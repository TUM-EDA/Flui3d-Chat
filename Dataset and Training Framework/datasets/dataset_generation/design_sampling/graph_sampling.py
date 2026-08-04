import random
import networkx as nx
import itertools
from typing import List, Dict, Tuple, Optional, Any, Set

class GraphGenerator:
    """
    Generates weakly connected directed acyclic graphs representing microfluidic chip designs (v2).
    """

    # Fixed (in_degree, out_degree) for components whose port counts never vary.
    STATIC_DEGREES = {
        'Inlet': (0, 1), 'Outlet': (1, 0), 'Chamber': (1, 1), 'Delay': (1, 1),
        'Mixer': (1, 1), 'Droplet': (1, 1), 'TeslaValve': (1, 1),
    }

    # Total junction width (sources + targets) sampling distribution.
    JUNCTION_WIDTH_DISTRIBUTION = {2: 0.50, 3: 0.25, 4: 0.15, 5: 0.10}

    # Cap on distinct graphs kept per component-combo, to bound the candidate population.
    MAX_GRAPHS_PER_COMBO = 4

    # Number of component combinations sampled per (n_regular, n_junc) size-cell. The full
    # candidate population would otherwise dwarf the few thousand designs we actually sample.
    COMBOS_PER_CELL = 150

    # Max number of junctions allowed in a single directed chain (junction -> junction -> ...).
    # Limits overly complex junction-only structures.
    MAX_JUNCTION_CHAIN = 3

    POST_SHAPES = ['circle', 'square', 'triangle', 'diamond', 'hexagon']

    # Maps a temporary combo prefix to its final, schema-conformant ID prefix.
    ID_PREFIX = {
        'Inlet': 'inlet', 'Outlet': 'outlet', 'Chamber': 'chamber', 'Delay': 'delay',
        'Mixer': 'mixer', 'Droplet': 'droplet', 'Filter': 'filter', 'TeslaValve': 'tesla_valve',
        'SplittingJunction': 'junction', 'CombiningJunction': 'junction',
    }

    def __init__(self, max_components: int = 20, max_junctions: int = 10, max_designs: int = 20000):
        """
        Initializes the GraphGenerator with component and design limits.

        Args:
            max_components (int): The maximum number of regular components in a design.
            max_junctions (int): The maximum number of junction components in a design.
            max_designs (int): The maximum number of unique designs to generate.
        """
        self.max_components = max_components
        self.max_junctions = max_junctions
        self.max_designs = max_designs

    def generate_designs(self) -> List[Dict[str, Any]]:
        """
        Generates a list of microfluidic chip designs as graphs.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                                  contains the design ID and its graph representation.
        """
        all_designs = self._generate_microfluidic_designs()
        limited_designs = self._limit_designs_by_max(all_designs)
        designs_with_ids = [{"id": idx + 1, "graph": design} for idx, design in enumerate(limited_designs)]

        return designs_with_ids

    def _sample_junction_width(self) -> int:
        """Draws a total junction width from JUNCTION_WIDTH_DISTRIBUTION."""
        widths = list(self.JUNCTION_WIDTH_DISTRIBUTION.keys())
        weights = list(self.JUNCTION_WIDTH_DISTRIBUTION.values())
        return random.choices(widths, weights=weights, k=1)[0]

    def _sample_node_meta(self, combo_nodes: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Samples per-node degree and subtype attributes for a combination of nodes.

        Filters and junctions have type/width-dependent degrees, so they are sampled
        before the graph is built. Mixers and droplets carry a subtype that does not
        affect their degree but is recorded for uniqueness and downstream params.

        Args:
            combo_nodes (List[str]): Temporary node names (e.g. ``Filter_3``).

        Returns:
            Dict[str, Dict[str, Any]]: Maps each node to ``{"degree": (in, out), "attrs": {...}}``.
        """
        meta: Dict[str, Dict[str, Any]] = {}
        for node in combo_nodes:
            ctype = node.split('_')[0]
            attrs: Dict[str, Any] = {}

            if ctype == 'Filter':
                ftype = random.choice(['dld', 'pillar_matrix'])
                attrs['type'] = ftype
                degree = (1, 2) if ftype == 'dld' else (1, 1)
            elif ctype == 'SplittingJunction':
                degree = (1, self._sample_junction_width())
            elif ctype == 'CombiningJunction':
                degree = (self._sample_junction_width(), 1)
            elif ctype == 'Mixer':
                attrs['type'] = random.choice(['serpentine', 'ring'])
                degree = (1, 1)
            elif ctype == 'Droplet':
                attrs['type'] = random.choice(['t_junction', 'flow_focusing'])
                degree = (1, 1)
            else:
                degree = self.STATIC_DEGREES[ctype]

            meta[node] = {"degree": degree, "attrs": attrs}
        return meta

    def _longest_junction_chain(self, G: nx.DiGraph) -> int:
        """
        Returns the maximum number of junctions in a single directed junction-only chain.

        Junction nodes are identified by their temporary combo prefix
        (``SplittingJunction`` / ``CombiningJunction``).

        Args:
            G (nx.DiGraph): The graph with temporary node names.

        Returns:
            int: The node count of the longest path through consecutive junction nodes
                 (0 if the graph has no junctions).
        """
        junction_nodes = [n for n in G.nodes()
                          if n.split('_')[0] in ('SplittingJunction', 'CombiningJunction')]
        if not junction_nodes:
            return 0
        return nx.dag_longest_path_length(G.subgraph(junction_nodes)) + 1

    def _has_duplicate_collapsed_connection(self, G: nx.DiGraph) -> bool:
        """
        Reports whether dissolving the junctions would leave two identical connections.

        The JSON converter removes every junction by reconnecting each of its inputs to
        each of its outputs. When two junction-mediated paths (or a junction path and a
        direct edge) join the same source port to the same target, that reconnection emits
        the same ``(source, target)`` connection twice. Once the junctions are gone the
        surplus edge carries no information, so such a wiring is rejected at sampling time
        rather than de-duplicated downstream. DLD filter outputs keep their distinct
        ``smaller`` / ``larger`` port labels, so genuinely different ports are not equated.

        Args:
            G (nx.DiGraph): A renamed graph with final IDs and filter port labels assigned.

        Returns:
            bool: True if the junction-collapsed graph contains a duplicate connection.
        """
        H = nx.MultiDiGraph(G.copy())
        for junction in [n for n in list(H.nodes()) if n.startswith('junction')]:
            in_edges = list(H.in_edges(junction, data=True))
            out_edges = list(H.out_edges(junction, data=True))
            for src, _, attr_in in in_edges:
                for _, tgt, attr_out in out_edges:
                    H.add_edge(src, tgt, **{**attr_in, **attr_out})
            H.remove_node(junction)

        seen: Set[Tuple[str, str]] = set()
        for u, v, data in H.edges(data=True):
            if u.startswith('junction') or v.startswith('junction'):
                continue
            if u.startswith('filter') and data.get('filter_connection_type'):
                key = (f"{u}_{data['filter_connection_type']}", v)
            else:
                key = (u, v)
            if key in seen:
                return True
            seen.add(key)
        return False

    def _generate_weakly_connected_digraph(self, nodes: List[str], degrees: Dict[str, Tuple[int, int]]) -> Optional[nx.DiGraph]:
        """
        Generates a weakly connected directed acyclic graph from a set of nodes and their degree constraints.

        Args:
            nodes (List[str]): A list of node identifiers.
            degrees (Dict[str, Tuple[int, int]]): A dictionary mapping node identifiers
                                                  to their (in-degree, out-degree) tuples.

        Returns:
            Optional[nx.DiGraph]: A generated graph if a valid one is found within the retry limit,
                                  otherwise None.
        """
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        retry_count = 0

        def reset_graph():
            nonlocal in_slots, out_slots, available_out_nodes, available_in_nodes, retry_count, stalled
            in_slots = {node: degrees[node][0] for node in nodes}
            out_slots = {node: degrees[node][1] for node in nodes}
            available_out_nodes = [node for node in nodes if out_slots[node] > 0]
            available_in_nodes = [node for node in nodes if in_slots[node] > 0]
            G.remove_edges_from(list(G.edges()))
            retry_count += 1
            stalled = 0

        in_slots = {node: degrees[node][0] for node in nodes}
        out_slots = {node: degrees[node][1] for node in nodes}

        if sum(in_slots.values()) != sum(out_slots.values()):
            return None

        available_out_nodes = [node for node in nodes if out_slots[node] > 0]
        available_in_nodes = [node for node in nodes if in_slots[node] > 0]

        # Once a node's only remaining partners are itself or already-linked targets, the
        # current partial graph is unsatisfiable. With multi-way junctions this dead-end is
        # common, so we count consecutive failed picks and rebuild once we are clearly stuck.
        stall_limit = 4 * len(nodes) + 50
        stalled = 0

        while available_out_nodes and available_in_nodes:
            source = random.choice(available_out_nodes)
            target = random.choice(available_in_nodes)

            if source != target and not G.has_edge(source, target):
                stalled = 0
                G.add_edge(source, target)
                out_slots[source] -= 1
                in_slots[target] -= 1

                if out_slots[source] == 0:
                    available_out_nodes.remove(source)
                if in_slots[target] == 0:
                    available_in_nodes.remove(target)

                if len(available_in_nodes) == len(available_out_nodes) == 0 and (not nx.is_weakly_connected(G) or not nx.is_directed_acyclic_graph(G)):
                    if retry_count >= 100:
                        return None
                    reset_graph()

            else:
                stalled += 1
                if stalled >= stall_limit:
                    if retry_count >= 100:
                        return None
                    reset_graph()
        return G

    def _normalize_graph_edges(self, G: nx.DiGraph) -> Tuple[Any, ...]:
        """
        Creates a canonical representation of the graph for de-duplication.

        The key combines the type-level edge multiset with a per-node descriptor that
        captures subtype and degree, so that two chips with identical topology but
        different junction widths or component subtypes are treated as distinct.

        Args:
            G (nx.DiGraph): The input graph (subtype attributes already assigned).

        Returns:
            Tuple[Any, ...]: A hashable canonical representation.
        """
        edges = sorted((u.split('_')[0], v.split('_')[0]) for u, v in G.edges())
        node_desc = sorted(
            (n.split('_')[0], G.nodes[n].get('type', ''), G.in_degree(n), G.out_degree(n))
            for n in G.nodes()
        )
        return (tuple(edges), tuple(node_desc))

    def _apply_default_params(self, data: Dict[str, Any], category: str, subtype: Optional[str]) -> None:
        """Writes the v2 default parameters for a component onto its node attribute dict."""
        if category == 'mixer':
            if subtype == 'ring':
                data.update(diameter_um=1000, num_circles=3, distance_between_circles_um=200)
            else:  # serpentine
                data.update(num_turnings=4, amplitude_um=2000, distance_between_turnings_um=200)
        elif category == 'delay':
            data.update(num_turnings=4, amplitude_um=2000, distance_between_turnings_um=200)
        elif category == 'chamber':
            data.update(length_um=4000, width_um=3200)
        elif category == 'filter':
            if subtype == 'pillar_matrix':
                data.update(length_um=4000, width_um=3200, post_shape='circle',
                            post_diameter_um=400, columns=3, rows=4)
            else:  # dld
                data.update(length_um=4000, width_um=3200, post_shape='circle',
                            post_diameter_um=50, row_shift_fraction=0.20,
                            critical_particle_diameter_um=10)
        elif category == 'tesla_valve':
            data.update(num_segment_pairs=2, segment_length_um=1000, segment_width_um=600)
        elif category == 'droplet':
            data.update(nozzle_width_um=100)

    def _rename_and_set_attributes(self, G: nx.DiGraph) -> Optional[nx.DiGraph]:
        """
        Renames nodes to standardized v2 IDs and assigns their default attributes.

        Numbering follows a deterministic lexicographic topological order, so upstream
        components receive lower indices. Junctions (split and merge) share the single
        ``junction_N`` counter and may connect directly to one another.

        Args:
            G (nx.DiGraph): The input graph with temporary node names and sampled subtypes.

        Returns:
            Optional[nx.DiGraph]: A new graph with renamed nodes and assigned attributes,
                                  or None if the graph is not a DAG.
        """
        try:
            order = list(nx.lexicographical_topological_sort(G, key=str))
        except nx.NetworkXUnfeasible:
            return None

        counters = {prefix: 0 for prefix in set(self.ID_PREFIX.values())}
        name_dict = {}
        for node in order:
            prefix = self.ID_PREFIX[node.split('_')[0]]
            counters[prefix] += 1
            name_dict[node] = f"{prefix}_{counters[prefix]}"
        nx.relabel_nodes(G, name_dict, copy=False)

        for node in list(G.nodes()):
            data = G.nodes[node]
            if node.startswith('mixer'):
                self._apply_default_params(data, 'mixer', data.get('type'))
            elif node.startswith('delay'):
                data['type'] = 'serpentine'
                self._apply_default_params(data, 'delay', 'serpentine')
            elif node.startswith('chamber'):
                self._apply_default_params(data, 'chamber', None)
            elif node.startswith('filter'):
                self._apply_default_params(data, 'filter', data.get('type'))
            elif node.startswith('tesla_valve'):
                self._apply_default_params(data, 'tesla_valve', None)
            elif node.startswith('droplet'):
                self._apply_default_params(data, 'droplet', data.get('type'))
            elif node.startswith('junction'):
                data['function'] = 'splitting' if G.out_degree(node) > G.in_degree(node) else 'combining'
                # Y-junctions are binary-only (a 1->2 split or 2->1 merge): a multi-way Y is not
                # realizable at the synthesis level. Any wider junction must therefore be a
                # T-junction. The fan width is max(in_degree, out_degree).
                junction_width = max(G.in_degree(node), G.out_degree(node))
                data['type'] = (random.choice(['T-junction', 'Y-junction'])
                                if junction_width == 2 else 'T-junction')

        # Label the two output ports of every DLD filter.
        for node in list(G.nodes()):
            if node.startswith('filter') and G.nodes[node].get('type') == 'dld':
                for i, (u, v) in enumerate(sorted(G.out_edges(node))):
                    G.edges[u, v]['filter_connection_type'] = 'larger' if i == 0 else 'smaller'

        H = nx.DiGraph()
        H.add_nodes_from(sorted(G.nodes(data=True)))
        H.add_edges_from(G.edges(data=True))
        return H

    def _create_randomized_graph(self, G: nx.DiGraph) -> Optional[nx.DiGraph]:
        """
        Creates a new graph by randomly altering the parameters of some components.

        Each component group is independently re-rolled with probability 0.5, drawing
        from the v2 random ranges (DOMAIN_KNOWLEDGE.md Table 2.2).

        Args:
            G (nx.DiGraph): The input graph.

        Returns:
            Optional[nx.DiGraph]: A new graph with randomized attributes, or None if no
                                  attributes were changed.
        """
        randomized_G = G.copy()
        changed = False
        for node, data in randomized_G.nodes(data=True):
            if node.startswith('mixer'):
                if random.random() < 0.5:
                    if data.get('type') == 'ring':
                        data['diameter_um'] = random.randint(200, 3000)
                        data['num_circles'] = random.randint(1, 8)
                        data['distance_between_circles_um'] = random.randint(50, 500)
                    else:  # serpentine
                        data['num_turnings'] = random.randint(1, 20)
                        data['amplitude_um'] = random.randint(1000, 7500)
                        data['distance_between_turnings_um'] = random.randint(50, 500)
                    changed = True
            elif node.startswith('delay'):
                if random.random() < 0.5:
                    data['num_turnings'] = random.randint(1, 20)
                    data['amplitude_um'] = random.randint(1000, 7500)
                    data['distance_between_turnings_um'] = random.randint(50, 500)
                    changed = True
            elif node.startswith('chamber'):
                if random.random() < 0.5:
                    data['length_um'] = random.randint(100, 6000)
                    data['width_um'] = random.randint(100, 6000)
                    changed = True
            elif node.startswith('filter'):
                if random.random() < 0.5:
                    if data.get('type') == 'pillar_matrix':
                        data['length_um'] = random.randint(1000, 15000)
                        data['width_um'] = random.randint(500, 6000)
                        data['post_shape'] = random.choice(self.POST_SHAPES)
                        data['post_diameter_um'] = random.randint(100, 800)
                        data['columns'] = random.randint(2, 12)
                        data['rows'] = random.randint(2, 12)
                    else:  # dld
                        data['length_um'] = random.randint(2000, 20000)
                        data['width_um'] = random.randint(1000, 6000)
                        data['post_shape'] = random.choice(self.POST_SHAPES)
                        data['post_diameter_um'] = random.randint(5, 50)
                        data['row_shift_fraction'] = round(random.uniform(0.02, 0.30), 2)
                        data['critical_particle_diameter_um'] = round(random.uniform(5, 25), 2)
                    changed = True
            elif node.startswith('tesla_valve'):
                if random.random() < 0.5:
                    data['num_segment_pairs'] = random.randint(1, 10)
                    data['segment_length_um'] = random.randint(1000, 3500)
                    data['segment_width_um'] = random.randint(500, 2500)
                    changed = True
            elif node.startswith('droplet'):
                if random.random() < 0.5:
                    data['nozzle_width_um'] = random.randint(10, 200)
                    changed = True
        return randomized_G if changed else None

    def _generate_microfluidic_designs(self) -> List[Dict[str, Any]]:
        """
        Generates a comprehensive list of microfluidic graph designs based on component combinations.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing a generated graph
                                  and its total number of components.
        """
        all_graphs = []
        regular_components = ['Inlet', 'Outlet', 'Chamber', 'Delay', 'Mixer', 'Droplet', 'TeslaValve', 'Filter']
        junctions = ['SplittingJunction', 'CombiningJunction']

        for n_regular in range(2, self.max_components + 1):
            for n_junc in range(0, self.max_junctions + 1):
                # First, generate all possible combinations for the current counts
                all_combinations = [
                    regular_combo + junc_combo
                    for regular_combo in itertools.combinations_with_replacement(regular_components, n_regular)
                    for junc_combo in itertools.combinations_with_replacement(junctions, n_junc)
                ]

                # If the list of combinations is large, sample from it
                if len(all_combinations) >= self.COMBOS_PER_CELL:
                    selected_combinations = random.choices(all_combinations, k=self.COMBOS_PER_CELL)
                else:
                    selected_combinations = all_combinations

                # Go through the selected combinations to generate graphs
                for full_combo in selected_combinations:
                    if full_combo.count('Inlet') < 1 or full_combo.count('Outlet') < 1 or sum(1 for c in full_combo if c not in ['Inlet', 'Outlet']) < 1 or sum(1 for c in full_combo if c not in ['Inlet', 'Outlet', 'SplittingJunction', 'CombiningJunction']) > 10:
                        continue

                    combo_nodes = [f"{c}_{i}" for i, c in enumerate(full_combo)]
                    unique_graphs = set()

                    for _ in range(30):
                        # We only keep a few graphs per combo: the dataset samples a small
                        # subset and the candidate population would otherwise explode.
                        if len(unique_graphs) >= self.MAX_GRAPHS_PER_COMBO:
                            break

                        node_meta = self._sample_node_meta(combo_nodes)
                        combo_degrees = {node: node_meta[node]["degree"] for node in combo_nodes}
                        G = self._generate_weakly_connected_digraph(combo_nodes, combo_degrees)
                        if G is None:
                            # Re-sampled widths/types may rebalance the degree sequence; keep trying.
                            continue

                        if self._longest_junction_chain(G) > self.MAX_JUNCTION_CHAIN:
                            # Reject overly long junction-to-junction chains; try another wiring.
                            continue

                        for node in combo_nodes:
                            for key, value in node_meta[node]["attrs"].items():
                                G.nodes[node][key] = value

                        normalized = self._normalize_graph_edges(G)
                        if normalized in unique_graphs:
                            continue

                        renamed_G = self._rename_and_set_attributes(G)
                        if renamed_G is None:
                            continue

                        # Reject wirings whose junction chains would collapse two routes onto
                        # the same (source, target) pair: once the converter dissolves the
                        # junctions that edge is an indistinguishable duplicate. Re-roll instead
                        # of admitting a design with a meaningless repeated connection.
                        if self._has_duplicate_collapsed_connection(renamed_G):
                            continue

                        unique_graphs.add(normalized)
                        all_graphs.append({"num_components": n_regular + n_junc, "graph": renamed_G})
                        randomized_G = self._create_randomized_graph(renamed_G)
                        if randomized_G:
                            all_graphs.append({"num_components": n_regular + n_junc, "graph": randomized_G})

        return all_graphs

    def _limit_designs_by_max(self, designs: List[Dict[str, Any]]) -> List[nx.DiGraph]:
        """
        Limits the number of generated designs to a specified maximum, ensuring a balanced
        distribution across different component counts.

        Args:
            designs (List[Dict[str, Any]]): The full list of generated designs.

        Returns:
            List[nx.DiGraph]: A list of graphs, sampled and shuffled to meet the max_designs limit.
        """
        random.shuffle(designs)
        designs_by_component_count = {n: [] for n in range(3, self.max_components + self.max_junctions + 1)}
        for design in designs:
            component_count = design["num_components"]
            if component_count in designs_by_component_count:
                designs_by_component_count[component_count].append(design["graph"])

        limited_designs = []
        designs_to_sample_from = {n: list(graphs) for n, graphs in designs_by_component_count.items()}

        while len(limited_designs) < self.max_designs:
            # Find categories that still have designs left
            active_categories = [n for n, graphs in designs_to_sample_from.items() if graphs]

            if not active_categories:
                break  # Stop if no designs are left anywhere

            # Calculate how many designs we still need
            remaining_allocation = self.max_designs - len(limited_designs)

            # Calculate an even allocation for the remaining active categories (must take at least 1)
            alloc_per_category = max(1, remaining_allocation // len(active_categories))

            # Flag to break the outer loop if we're done
            finished = False
            for n in active_categories:
                available_designs = designs_to_sample_from[n]

                # Determine how many designs to take from this category
                num_to_take = min(len(available_designs), alloc_per_category)

                # Randomly sample the designs
                taken_designs = random.sample(available_designs, num_to_take)
                limited_designs.extend(taken_designs)

                # Correctly remove the *specific* designs that were taken
                taken_set = set(taken_designs)
                designs_to_sample_from[n] = [d for d in available_designs if d not in taken_set]

                # Check if we have reached the maximum number of designs
                if len(limited_designs) >= self.max_designs:
                    finished = True
                    break

            if finished:
                break

        # Trim the list to the exact max_designs count, in case the last extend went over
        final_designs = limited_designs[:self.max_designs]

        # Shuffle the final list to mix designs from all categories
        random.shuffle(final_designs)

        return final_designs
