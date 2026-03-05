import random
import networkx as nx
import itertools
from typing import List, Dict, Tuple, Optional, Any, Set

class GraphGenerator:
    """
    Generates weakly connected directed acyclic graphs representing microfluidic chip designs.
    """

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
        self.components = {
            'Inlet': (0, 1), 'Outlet': (1, 0), 'Chamber': (1, 1),
            'SplittingJunction': (1, 2), 'CombiningJunction': (2, 1),
            'Delay': (1, 1), 'Mixer': (1, 1), 'Droplet': (2, 1), 'Filter': (1, 2)
        }

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
            nonlocal in_slots, out_slots, available_out_nodes, available_in_nodes, retry_count
            in_slots = {node: degrees[node][0] for node in nodes}
            out_slots = {node: degrees[node][1] for node in nodes}
            available_out_nodes = [node for node in nodes if out_slots[node] > 0]
            available_in_nodes = [node for node in nodes if in_slots[node] > 0]
            G.remove_edges_from(list(G.edges()))
            retry_count += 1

        in_slots = {node: degrees[node][0] for node in nodes}
        out_slots = {node: degrees[node][1] for node in nodes}

        if sum(in_slots.values()) != sum(out_slots.values()):
            return None

        available_out_nodes = [node for node in nodes if out_slots[node] > 0]
        available_in_nodes = [node for node in nodes if in_slots[node] > 0]

        while available_out_nodes and available_in_nodes:
            source = random.choice(available_out_nodes)
            target = random.choice(available_in_nodes)

            if source != target and not G.has_edge(source, target):
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

            elif (source == target or G.has_edge(source, target)) and len(available_in_nodes) == len(available_out_nodes) == 1:
                if retry_count >= 100:
                    return None
                reset_graph()
        return G

    def _normalize_graph_edges(self, G: nx.DiGraph) -> Tuple[Tuple[str, str], ...]:
        """
        Creates a canonical representation of the graph's edges by sorting them.

        Args:
            G (nx.DiGraph): The input graph.

        Returns:
            Tuple[Tuple[str, str], ...]: A sorted tuple of edge tuples, where each edge is
                                         represented by the types of its source and target nodes.
        """
        normalized_edges = []
        for u, v in G.edges():
            u_type = u.split('_')[0]
            v_type = v.split('_')[0]
            normalized_edges.append((u_type, v_type))
        normalized_edges.sort()
        return tuple(normalized_edges)

    def _replace_mixing_and_splitting_units(self, graph: nx.DiGraph) -> Optional[nx.DiGraph]:
        """
        Consolidates connected sequences of combining or splitting junctions into single unit nodes.

        Args:
            graph (nx.DiGraph): The input graph with individual junction nodes.

        Returns:
            Optional[nx.DiGraph]: A new graph with junction units replaced, or None if the
                                  edge count is invalid after replacement.
        """
        graph = graph.copy()
        num_edges = len(graph.edges())

        def replace_node_group(graph, nodes, new_node_name):
            nonlocal num_edges
            graph.add_node(new_node_name)
            graph.nodes[new_node_name]["junctions"] = []
            for node in nodes:
                graph.nodes[new_node_name]["junctions"].append(node)
                for pred in graph.predecessors(node):
                    if pred is not new_node_name:
                        graph.add_edge(pred, new_node_name)
                for succ in graph.successors(node):
                    if succ is not new_node_name:
                        graph.add_edge(new_node_name, succ)
            num_edges -= (len(nodes) - 1)
            graph.remove_nodes_from(nodes)

        def add_neighbors_to_connected_group(node, connected_group, component_type):
            for neighbor in nx.all_neighbors(graph, node):
                if neighbor not in connected_group and neighbor.startswith(component_type):
                    connected_group.add(neighbor)
                    add_neighbors_to_connected_group(neighbor, connected_group, component_type)

        for component_type, new_name in [('Combining', 'combining_unit'), ('Splitting', 'splitting_unit')]:
            groups_to_replace = []
            for name, attributes in graph.nodes(data=True):
                if all(name not in group for group in groups_to_replace) and name.startswith(component_type):
                    connected_group = {name}
                    add_neighbors_to_connected_group(name, connected_group, component_type)
                    if connected_group:
                        groups_to_replace.append(list(connected_group))
            
            for i, group in enumerate(groups_to_replace, 1):
                replace_node_group(graph, group, f"{new_name}_{i}")

        if num_edges != len(graph.edges()):
            return None
        return graph

    def _rename_and_set_attributes(self, G: nx.DiGraph) -> Optional[nx.DiGraph]:
        """
        Renames nodes in the graph to a standardized format and sets their attributes.

        Args:
            G (nx.DiGraph): The input graph with temporary node names.

        Returns:
            Optional[nx.DiGraph]: A new graph with renamed nodes and assigned attributes,
                                  or None if the graph is invalid.
        """
        components = {'Inlet': 0, 'Outlet': 0, 'Chamber': 0, 'Delay': 0, 'Mixer': 0,
                      'Droplet': 0, 'Filter': 0, 'junction': 0}
        name_dict = {}

        def handle_linear_subgraph(graph):
            if not graph.nodes():
                return
            root = [n for n in graph.nodes if graph.in_degree(n) == 0][0]
            current_node = root
            while True:
                component_type = current_node.split('_')[0]
                new_name = f"{component_type.lower()}_{components[component_type] + 1}"
                components[component_type] += 1
                name_dict[current_node] = new_name
                successors = list(graph.successors(current_node))
                if not successors:
                    break
                current_node = successors[0]

        def rename_graph_recursive(original_graph, replaced_graph):
            
            original_graph = original_graph.copy()
            replaced_graph = replaced_graph.copy()

            topo_sort = list(nx.topological_sort(replaced_graph))
            highest_order_node = topo_sort[-1]
            target_nodes = sorted([node for node in replaced_graph.nodes if any(node.startswith(p) for p in ['splitting', 'combining', 'Droplet', 'Filter'])], key=str.lower)
            
            furthest_node = None
            max_distance = -1
            for node in target_nodes:

                all_paths = list(nx.all_simple_paths(replaced_graph, source=node, target=highest_order_node))

                if len(all_paths)==0:
                    continue

                path_length = max(len(path) - 1 for path in all_paths)
                if path_length > max_distance:
                    max_distance = path_length
                    furthest_node = node

            if not furthest_node:
                handle_linear_subgraph(replaced_graph)
                return

            if furthest_node.startswith("combining") or furthest_node.startswith("Droplet"):
                junction_nodes = replaced_graph.nodes[furthest_node].get("junctions", [])
                if junction_nodes:
                    subgraph = original_graph.subgraph(junction_nodes)
                    for node in nx.topological_sort(subgraph):
                        name_dict[node] = f"junction_{components['junction'] + 1}"
                        components['junction'] += 1

                predecessors = list(nx.ancestors(replaced_graph, furthest_node))
                subgraph = replaced_graph.subgraph(predecessors)
                connected_components = list(nx.weakly_connected_components(subgraph))

                components_to_process = []

                # --- Specific logic for Droplet node ordering ---
                if furthest_node.startswith("Droplet"):
                    # 1. Relabel incoming edges
                    in_edges = list(G.in_edges(furthest_node))
                    if len(in_edges) == 2:
                        G.edges[in_edges[0]]["droplet_connection_type"] = "dispersed"
                        G.edges[in_edges[1]]["droplet_connection_type"] = "continuous"
                        dispersed_pred = in_edges[0][0]
                        continuous_pred = in_edges[1][0]

                        # 2. Find the components corresponding to the dispersed and continuous inputs
                        dispersed_component = next((c for c in connected_components if dispersed_pred in c), None)
                        continuous_component = next((c for c in connected_components if continuous_pred in c), None)

                        # 3. Order these two main components based on length (longer first)
                        main_components = []
                        if dispersed_component and continuous_component:
                            if len(dispersed_component) >= len(continuous_component):
                                main_components = [dispersed_component, continuous_component]
                            else:
                                main_components = [continuous_component, dispersed_component]
                            components_to_process.extend(main_components)

                        # 4. Add any other predecessor components, sorted by length
                        processed_nodes = set().union(*main_components)
                        remaining_components = [c for c in connected_components if not processed_nodes.intersection(c)]
                        components_to_process.extend(sorted(remaining_components, key=len, reverse=True))

                    # 5. Relabel the droplet node
                    name_dict[furthest_node] = f"droplet_{components['Droplet'] + 1}"
                    components['Droplet'] += 1
                # --- General logic for combining nodes ---
                else:
                    components_to_process = sorted(connected_components, key=len, reverse=True)

                for c in components_to_process:
                    handle_linear_subgraph(subgraph.subgraph(c))
                
                predecessors.append(furthest_node)
                replaced_graph.remove_nodes_from(predecessors)
                rename_graph_recursive(original_graph, replaced_graph)

            elif furthest_node.startswith("splitting") or furthest_node.startswith("Filter"):
                junction_nodes = replaced_graph.nodes[furthest_node].get("junctions", [])
                if junction_nodes:
                    subgraph = original_graph.subgraph(junction_nodes)
                    for node in nx.topological_sort(subgraph):
                        name_dict[node] = f"junction_{components['junction'] + 1}"
                        components['junction'] += 1

                if furthest_node.startswith("Filter"):
                    for i, (_, v) in enumerate(sorted(G.out_edges(furthest_node))):
                        G.edges[furthest_node, v]["filter_connection_type"] = "larger" if i == 0 else "smaller"
                    name_dict[furthest_node] = f"filter_{components['Filter'] + 1}"
                    components['Filter'] += 1

                predecessors = list(nx.ancestors(replaced_graph, furthest_node))
                handle_linear_subgraph(replaced_graph.subgraph(predecessors))
                
                replaced_graph.remove_node(furthest_node)
                replaced_graph.remove_nodes_from(predecessors)
                
                for c in sorted(nx.weakly_connected_components(replaced_graph), key=lambda comp: (len(comp), list(nx.topological_sort(replaced_graph.subgraph(comp)))[0].lower())):
                    rename_graph_recursive(original_graph, replaced_graph.subgraph(c))

        replaced_graph = self._replace_mixing_and_splitting_units(G)
        if not replaced_graph:
            return None
        
        rename_graph_recursive(G, replaced_graph)
        nx.relabel_nodes(G, name_dict, False)

        for node in list(G.nodes()):
            component_type = node.split('_')[0]
            if component_type == 'delay' or component_type == 'mixer':
                G.nodes[node]['num_turnings'] = 4
            elif component_type == 'chamber':
                G.nodes[node].update({'length': 4000, 'width': 3200})
            elif component_type == 'filter':
                G.nodes[node]['critical_particle_diameter'] = 10
            elif component_type == "junction":
                G.nodes[node]['function'] = 'splitting' if G.out_degree(node) == 2 else 'combining'
                G.nodes[node]['type'] = random.choice(['T-junction', 'Y-junction'])
        
        H = nx.DiGraph()
        H.add_nodes_from(sorted(G.nodes(data=True)))
        H.add_edges_from(G.edges(data=True))
        return H

    def _create_randomized_graph(self, G: nx.DiGraph) -> Optional[nx.DiGraph]:
        """
        Creates a new graph by randomly altering the attributes of some nodes in the original graph.

        Args:
            G (nx.DiGraph): The input graph.

        Returns:
            Optional[nx.DiGraph]: A new graph with randomized attributes, or None if no
                                  attributes were changed.
        """
        randomized_G = G.copy()
        changed = False
        for node, data in randomized_G.nodes(data=True):
            if 'num_turnings' in data and random.random() < 0.5:
                data['num_turnings'] = random.randint(1, 20)
                changed = True
            if 'length' in data and random.random() < 0.5:
                data['length'] = random.randint(100, 6000)
                data['width'] = random.randint(100, 6000)
                changed = True
            if 'critical_particle_diameter' in data and random.random() < 0.5:
                data['critical_particle_diameter'] = round(random.uniform(5, 15), 2)
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
        regular_components = [name for name in self.components if 'Junction' not in name]
        junctions = [name for name in self.components if 'Junction' in name]

        for n_regular in range(2, self.max_components + 1):
            for n_junc in range(0, self.max_junctions + 1):
                # First, generate all possible combinations for the current counts
                all_combinations = [
                    regular_combo + junc_combo
                    for regular_combo in itertools.combinations_with_replacement(regular_components, n_regular)
                    for junc_combo in itertools.combinations_with_replacement(junctions, n_junc)
                ]

                # If the list of combinations is large, sample from it
                if len(all_combinations) >= 1000:
                    selected_combinations = random.choices(all_combinations, k=1000)
                else:
                    selected_combinations = all_combinations

                # Go through the selected combinations to generate graphs
                for full_combo in selected_combinations:
                    if full_combo.count('Inlet') < 1 or full_combo.count('Outlet') < 1 or sum(1 for c in full_combo if c not in ['Inlet', 'Outlet']) < 1 or sum(1 for c in full_combo if c not in ['Inlet', 'Outlet', 'SplittingJunction', 'CombiningJunction']) > 10:
                        continue

                    combo_nodes = [f"{c}_{i}" for i, c in enumerate(full_combo)]
                    combo_degrees = {node: self.components[node.split('_')[0]] for node in combo_nodes}
                    unique_graphs = set()

                    for _ in range(20):
                        G = self._generate_weakly_connected_digraph(combo_nodes, combo_degrees)
                        if G is None:
                            break
                        normalized_edges = self._normalize_graph_edges(G)
                        if normalized_edges not in unique_graphs:
                            unique_graphs.add(normalized_edges)
                            renamed_G = self._rename_and_set_attributes(G)
                            if renamed_G:
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