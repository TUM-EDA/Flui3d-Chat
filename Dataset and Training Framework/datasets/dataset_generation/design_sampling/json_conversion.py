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

    def _extract_connections(self, G: MultiDiGraph) -> List[Dict[str, str]]:
        """
        Extract connections from the graph into the new schema format.
        
        Args:
            G (MultiDiGraph): The input graph without junction nodes.
            
        Returns:
            List[Dict[str, str]]: A list of conceptual connections, each represented as a dictionary.
        """
        connections = []
        node_order = {node: i for i, node in enumerate(topological_sort(G))}
        sorted_edges = sorted(G.edges(data=True), key=lambda e: (node_order[e[0]], node_order[e[1]]))
        
        for source, target, edge_data in sorted_edges:
            if not (source.startswith("junction") or target.startswith("junction")):
                source_renamed = source
                target_renamed = target
                
                if target.startswith("droplet"):
                    target_renamed = f"{target}_{edge_data.get('droplet_connection_type', '')}"
                if source.startswith("filter"):
                    source_renamed = f"{source}_{edge_data.get('filter_connection_type', '')}"
                
                connections.append({"source": source_renamed, "target": target_renamed})
        return connections

    def _extract_junctions(self, G: MultiDiGraph) -> List[Dict[str, Any]]:
        """
        Extract junctions from the graph into the new schema format.
        
        Args:
            G (MultiDiGraph): The input graph containing junction nodes.
            
        Returns:
            List[Dict[str, Any]]: A list of junctions and their associated connections.
        """
        connections = []
        all_junctions = sorted([node for node in G.nodes if node.startswith("junction")], key=lambda x: int(x.split("_")[1]))
        
        for source in all_junctions:
            node_data = G.nodes[source]
            incoming_nodes = [edge[0] for edge in G.in_edges(source)]
            outgoing_nodes = [edge[1] for edge in G.out_edges(source)]
            
            junction_info = {"id": source, "type": node_data["type"]}
            
            if node_data["function"] == "combining":
                source_1, source_2 = (f"{n}_{G.edges[n, source].get('filter_connection_type', '')}" if n.startswith("filter") else n for n in incoming_nodes)
                target = f"{outgoing_nodes[0]}_{G.edges[source, outgoing_nodes[0]].get('droplet_connection_type', '')}" if outgoing_nodes[0].startswith("droplet") else outgoing_nodes[0]
                junction_info.update({"source_1": source_1, "source_2": source_2, "target": target})
            else:  # splitting
                source_1 = f"{incoming_nodes[0]}_{G.edges[incoming_nodes[0], source].get('filter_connection_type', '')}" if incoming_nodes[0].startswith("filter") else incoming_nodes[0]
                target_1, target_2 = (f"{n}_{G.edges[source, n].get('droplet_connection_type', '')}" if n.startswith("droplet") else n for n in outgoing_nodes)
                junction_info.update({"source": source_1, "target_1": target_1, "target_2": target_2})
            
            connections.append(junction_info)
        return connections

    def _convert_graph_to_schema(self, G: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert graph data to the new schema format.
        
        Args:
            G (Dict[str, Any]): The graph in networkx format.
            
        Returns:
            Dict[str, Any]: The graph represented in the new JSON schema format.
        """
        G_wo_junctions = self._remove_junction_nodes(G)

        component_params = {"mixers": [], "delays": [], "chambers": [], "filters": []}
        for node in topological_sort(G):
            data = G.nodes[node]
            if node.startswith("mixer"):
                component_params["mixers"].append({"id": node, "num_turnings": data.get("num_turnings", 4)})
            elif node.startswith("delay"):
                component_params["delays"].append({"id": node, "num_turnings": data.get("num_turnings", 4)})
            elif node.startswith("chamber"):
                component_params["chambers"].append({"id": node, "dimensions": {"length": data.get("length", 4000), "width": data.get("width", 3200)}})
            elif node.startswith("filter"):
                component_params["filters"].append({"id": node, "critical_particle_diameter": data.get("critical_particle_diameter", 10)})

        return {
            "connections": self._extract_connections(G_wo_junctions),
            "junctions": self._extract_junctions(G),
            "component_params": component_params
        }