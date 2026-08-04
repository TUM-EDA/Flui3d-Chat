import json
from networkx import MultiDiGraph, topological_sort
from typing import Any, Dict, List


class JsonConverter:
    """
    Converts a list of microfluidic chip graphs to a specified JSON schema.
    """

    def __init__(self) -> None:
        """
        Initializes the converter.
        """
        pass

    def convert_graphs(self, designs_with_ids: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Converts a list of graph data into the JSON schema format.

        Args:
            designs_with_ids (List[Dict[str, Any]]): A list of dictionaries, each containing an 'id'
                                     and a 'graph' in networkx node-link format.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each contains the 'id' and the
                  converted 'json' representation.
        """
        converted_graphs = [
            {"id": graph_data["id"], "json": self._convert_graph_to_schema(graph_data["graph"])}
            for graph_data in designs_with_ids
        ]
        return converted_graphs

    def _remove_junction_nodes(self, G: MultiDiGraph) -> MultiDiGraph:
        """
        Remove all junction nodes from the graph while keeping the edges and their attributes.
        
        Args:
            G (MultiDiGraph): The input graph.
            
        Returns:
            MultiDiGraph: A new graph with junction nodes removed and edges reconnected.
        """
        new_graph = MultiDiGraph(G.copy())
        junction_nodes = [node for node, data in new_graph.nodes(data=True) if "junction" in node]
        
        for junction in junction_nodes:
            incoming_edges = list(new_graph.in_edges(junction, data=True))
            outgoing_edges = list(new_graph.out_edges(junction, data=True))
            
            for src, _, attr_in in incoming_edges:
                for _, tgt, attr_out in outgoing_edges:
                    combined_attributes = {**attr_in, **attr_out}
                    new_graph.add_edge(src, tgt, **combined_attributes)
            new_graph.remove_node(junction)
        return new_graph

    def _source_port_name(self, G: MultiDiGraph, source: str, target: str) -> str:
        """
        Resolves the source-side port ID for an edge.

        Only DLD filters expose distinct output ports (``filter_N_smaller`` /
        ``filter_N_larger``), labelled on the edge via ``filter_connection_type``.
        Pillar-matrix filters (single output) and all other components use their
        plain node ID.

        Args:
            G (MultiDiGraph): The graph the edge belongs to.
            source (str): The source node ID.
            target (str): The target node ID.

        Returns:
            str: The schema-conformant source port ID.
        """
        if source.startswith("filter"):
            suffix = G.edges[source, target].get("filter_connection_type")
            if suffix:
                return f"{source}_{suffix}"
        return source

    def _extract_connections(self, G: MultiDiGraph) -> List[Dict[str, str]]:
        """
        Extract connections from the graph into the v2 schema format.

        Args:
            G (MultiDiGraph): The input graph without junction nodes.

        Returns:
            List[Dict[str, str]]: A list of conceptual connections, each represented as a dictionary.
        """
        connections = []
        # When the prompt generator recorded the exact order it narrates each connection
        # (`connection_reading_order`, a list of (source_node, target) pairs -- set by the connection
        # and path styles), list connections in that order. This is edge-level, needed because a node
        # can be introduced early yet have its out-edges narrated in later, separate clauses, which a
        # per-node order cannot express. Junction-incident edges are dropped first; unranked edges
        # (defensive) keep their original order via the stable sort.
        edge_order = G.graph.get("connection_reading_order")
        if edge_order is not None:
            rank = {tuple(pair): i for i, pair in enumerate(edge_order)}
            sorted_edges = sorted(
                (e for e in G.edges(data=True)
                 if not (e[0].startswith("junction") or e[1].startswith("junction"))),
                key=lambda e: rank.get((e[0], e[1]), len(rank)),
            )
        elif G.number_of_nodes() and all("reading_index" in d for _, d in G.nodes(data=True)):
            # Process style: it tags every node with `reading_index` (narration first-mention order).
            # Its prose is operation-centric -- an operation is stated once, naming all of its inputs --
            # so order connections by the consuming TARGET's reading position, then the source's. (Not
            # by source first: that would group ALL of a node's out-edges right after the node, even
            # when the prose narrates them at separate, later operations -- e.g. a filter's outputs
            # jumping ahead of a "distribute a second fluid" step narrated earlier.)
            node_order = {node: d["reading_index"] for node, d in G.nodes(data=True)}
            sorted_edges = sorted(G.edges(data=True), key=lambda e: (node_order[e[1]], node_order[e[0]]))
        else:
            # Untagged graph: fall back to a topological order.
            node_order = {node: i for i, node in enumerate(topological_sort(G))}
            sorted_edges = sorted(G.edges(data=True), key=lambda e: (node_order[e[0]], node_order[e[1]]))

        for source, target, edge_data in sorted_edges:
            if not (source.startswith("junction") or target.startswith("junction")):
                source_renamed = (
                    f"{source}_{edge_data['filter_connection_type']}"
                    if source.startswith("filter") and edge_data.get("filter_connection_type")
                    else source
                )
                connections.append({"source": source_renamed, "target": target})
        return connections

    def _extract_junctions(self, G: MultiDiGraph) -> List[Dict[str, Any]]:
        """
        Extract junctions from the graph into the v2 schema format.

        Each junction emits a ``sources`` list and a ``targets`` list. A split has
        one source and >=2 targets; a merge has >=2 sources and one target. Source
        ports resolve DLD filter output labels; targets are always plain node IDs.

        Args:
            G (MultiDiGraph): The input graph containing junction nodes.

        Returns:
            List[Dict[str, Any]]: A list of junctions and their associated connections.
        """
        junctions = []
        all_junctions = sorted([node for node in G.nodes if node.startswith("junction")], key=lambda x: int(x.split("_")[1]))

        for junction in all_junctions:
            node_data = G.nodes[junction]
            sources = [self._source_port_name(G, src, junction) for src, _ in G.in_edges(junction)]
            targets = [tgt for _, tgt in G.out_edges(junction)]
            junctions.append({"id": junction, "type": node_data["type"], "sources": sources, "targets": targets})
        return junctions

    def _convert_graph_to_schema(self, G: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert graph data to the new schema format.
        
        Args:
            G (Dict[str, Any]): The graph in networkx format.
            
        Returns:
            Dict[str, Any]: The graph represented in the new JSON schema format.
        """
        G_wo_junctions = self._remove_junction_nodes(G)

        component_params = {
            "mixers": [], "delays": [], "chambers": [],
            "filters": [], "droplets": [], "tesla_valves": [],
        }
        for node in topological_sort(G):
            data = G.nodes[node]
            if node.startswith("mixer"):
                mixer_type = data.get("type", "serpentine")
                if mixer_type == "ring":
                    component_params["mixers"].append({
                        "id": node, "type": "ring",
                        "diameter_um": data.get("diameter_um", 1000),
                        "num_circles": data.get("num_circles", 3),
                        "distance_between_circles_um": data.get("distance_between_circles_um", 200),
                    })
                else:
                    component_params["mixers"].append({
                        "id": node, "type": "serpentine",
                        "num_turnings": data.get("num_turnings", 4),
                        "amplitude_um": data.get("amplitude_um", 2000),
                        "distance_between_turnings_um": data.get("distance_between_turnings_um", 200),
                    })
            elif node.startswith("delay"):
                component_params["delays"].append({
                    "id": node, "type": "serpentine",
                    "num_turnings": data.get("num_turnings", 4),
                    "amplitude_um": data.get("amplitude_um", 2000),
                    "distance_between_turnings_um": data.get("distance_between_turnings_um", 200),
                })
            elif node.startswith("chamber"):
                component_params["chambers"].append({
                    "id": node,
                    "length_um": data.get("length_um", 4000),
                    "width_um": data.get("width_um", 3200),
                })
            elif node.startswith("filter"):
                filter_type = data.get("type", "dld")
                if filter_type == "pillar_matrix":
                    component_params["filters"].append({
                        "id": node, "type": "pillar_matrix",
                        "length_um": data.get("length_um", 4000),
                        "width_um": data.get("width_um", 3200),
                        "post_shape": data.get("post_shape", "circle"),
                        "post_diameter_um": data.get("post_diameter_um", 400),
                        "columns": data.get("columns", 3),
                        "rows": data.get("rows", 4),
                    })
                else:
                    component_params["filters"].append({
                        "id": node, "type": "dld",
                        "length_um": data.get("length_um", 4000),
                        "width_um": data.get("width_um", 3200),
                        "post_shape": data.get("post_shape", "circle"),
                        "post_diameter_um": data.get("post_diameter_um", 50),
                        "row_shift_fraction": data.get("row_shift_fraction", 0.20),
                        "critical_particle_diameter_um": data.get("critical_particle_diameter_um", 10),
                    })
            elif node.startswith("droplet"):
                component_params["droplets"].append({
                    "id": node, "type": data.get("type", "t_junction"),
                    "nozzle_width_um": data.get("nozzle_width_um", 100),
                })
            elif node.startswith("tesla_valve"):
                component_params["tesla_valves"].append({
                    "id": node,
                    "num_segment_pairs": data.get("num_segment_pairs", 2),
                    "segment_length_um": data.get("segment_length_um", 1000),
                    "segment_width_um": data.get("segment_width_um", 600),
                })

        # List each component_params bucket in ascending id order (mixer_1, mixer_2, mixer_3, ...)
        # rather than the topological build order, so the params read in numeric sequence.
        for bucket in component_params.values():
            bucket.sort(key=lambda c: int(c["id"].split("_")[-1]))

        return {
            "connections": self._extract_connections(G_wo_junctions),
            "junctions": self._extract_junctions(G),
            "component_params": component_params
        }