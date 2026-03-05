import random
import re
from itertools import groupby
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from .prompt_generation import PromptGenerator

class ProcessOrientedPromptGenerator(PromptGenerator):
    """Generates prompts describing the processes occurring on the chip."""

    # A dictionary of synonyms for various actions to make the prompts sound more varied.
    ACTION_SYNONYMS = {
        "let react": ["let react", "initiate a reaction", "allow to react"],
        "mix": ["mix", "blend", "homogenize"],
        "mixture": ["mixture", "blend", "homogenized solution"],
        "delay": ["delay", "hold", "retain"],
        "delayed": ["delayed", "held", "retained"],
        "combine": ["combine", "merge", "join", "unite"],
        "split": ["split", "divide", "separate", "distribute", "segment"],
        "separate smaller and larger particles of": ["separate smaller and larger particles of", "filter", "separate particles of", "filter particles of", "divide smaller and larger particles of", "split smaller and larger particles of", "sieve", "sieve particles of"],
        "form droplets": ["form droplets", "generate droplets", "produce droplets", "form microdroplets", "generate microdroplets", "produce microdroplets"],
        "droplet formation": ["droplet formation", "droplet generation", "droplet production", "microdroplet formation", "microdroplet generation", "microdroplet production"],
        "smaller particle stream": ["smaller particle stream", "smaller particle flow", "flow with smaller particles", "stream with smaller particles"],
        "larger particle stream": ["larger particle stream", "larger particle flow", "flow with larger particles", "stream with larger particles"]
    }

    # A list of introductory phrases to start the prompts.
    PROMPT_BEGINNINGS = [
        "Design a microfluidic chip that",
        "Create a blueprint for a microfluidic device capable of",
        "Imagine a microfluidic system that",
        "Develop a concept for a microfluidic chip intended for",
        "Generate the layout for a microfluidic device suitable for",
        "Draft the architecture of a microfluidic chip focused on",
        "Propose a microfluidic chip design aimed at",
        "Describe the components and connections for a microfluidic system that",
        "Envision a microfluidic platform that",
        "Outline a microfluidic design that"
    ]

    
    def _generate_for_single_graph(self, graph: nx.DiGraph, graph_id: str) -> Tuple[Dict, Dict]:
        """Main method to generate a process-oriented prompt for a single chip graph."""

        # Add component details to some descriptions
        detailed = random.choice([True, False])
        
        # This style uses its own specific logic to pre-process the graph, merging adjacent junctions.
        processed_graph = self._replace_combining_and_splitting_units(graph)

        # Convert the processed graph into a single node that contains the full textual description of the process.
        output_node = self._graph_to_text(processed_graph, detailed=detailed)
        action_description = list(output_node.nodes(data=True))[0][1].get("action", "")

        # Refine and augment description
        action_description = self._modify_combine_mix_pairs(action_description)
        action_description = self._modify_combine_react_pairs(action_description)
        action_description = self._replace_with_random_synonym(action_description, self.MODULE_SYNONYMS)
        action_description = self._replace_with_random_action_synonyms(action_description)
        action_description = re.sub(r'\s+', ' ', action_description).strip()
        action_description = re.sub(r',\s*([A-Z])', lambda m: f", {m.group(1).lower()}", action_description)

        # Add a random introductory phrase to the description to form the complete prompt.
        full_prompt = self._generate_full_prompt(action_description)

        # Returns two versions: a raw action description (for an auxiliary LLM) and the full, polished prompt (for direct use).
        return (
            {"id": graph_id, "prompt": action_description},
            {"id": graph_id, "prompt": full_prompt}
        )
    
    # --- The following methods contain the complex, recursive logic specific to this process-oriented style ---
    def _replace_combining_and_splitting_units(self, graph: nx.DiGraph) -> Optional[nx.DiGraph]:
        """
        Pre-processes the graph by merging adjacent combining or splitting junctions
        into single conceptual units. This simplifies the graph structure before the main
        text generation logic, as described in the thesis.
        
        Args:
            graph: The input microfluidic chip graph.
        
        Returns:
            A new graph with junction groups replaced by single nodes, or None if an
            invalid structure is detected.
        """
        graph = graph.copy()
        edge_count = len(graph.edges())

        def replace_node_group(nodes_to_replace, new_node_name):
            # Helper function to replace a group of nodes with a single new node,
            # consolidating their connections and attributes.
            nonlocal edge_count
            graph.add_node(new_node_name, junction_types=[])
            
            for node in sorted(nodes_to_replace):
                graph.nodes[new_node_name]["junction_types"].append(graph.nodes[node]["type"])
                for pred in graph.predecessors(node):
                    if pred != new_node_name:
                        graph.add_edge(pred, new_node_name, **graph.get_edge_data(pred, node, {}))
                for succ in graph.successors(node):
                    if succ != new_node_name:
                        graph.add_edge(new_node_name, succ, **graph.get_edge_data(node, succ, {}))
            
            edge_count -= (len(nodes_to_replace) - 1)
            graph.remove_nodes_from(nodes_to_replace)

        # Iterate through both 'combining' and 'splitting' junction types.
        for component_type, function, new_name_base in [('junction', 'combining', 'combining_unit'), ('junction', 'splitting', 'splitting_unit')]:
            visited_nodes = set()
            counter = 1
            for node_name, attributes in list(graph.nodes(data=True)):
                if node_name not in visited_nodes and node_name.startswith(component_type) and attributes.get("function") == function:
                    # Find all connected junctions of the same type.
                    group = set()
                    q = [node_name]
                    visited_in_group = {node_name}
                    while q:
                        curr = q.pop(0)
                        group.add(curr)
                        for neighbor in nx.all_neighbors(graph, curr):
                            if neighbor not in visited_in_group and neighbor.startswith(component_type) and graph.nodes[neighbor].get("function") == function:
                                visited_in_group.add(neighbor)
                                q.append(neighbor)
                    
                    # Replace the identified group with a single conceptual node.
                    replace_node_group(group, f"{new_name_base}_{counter}")
                    visited_nodes.update(group)
                    counter += 1

        # Sanity check to ensure the graph structure remains valid.
        if edge_count != len(graph.edges()):
            return None
        return graph

    def _graph_to_text(self, graph: nx.DiGraph, detailed: bool = False) -> nx.DiGraph:
        """
        Recursively converts a graph into a single node containing the process description.
        This is the main dispatcher function. It identifies the most complex operation
        (the "branching node" topologically distant from the outlets) and calls the
        appropriate handler for that node type (e.g., combining, splitting). If no
        branching nodes exist, it processes the graph as a simple linear sequence.
        
        Args:
            graph: The microfluidic chip graph to be converted.
            detailed: Flag to include more component details in the description.

        Returns:
            A new graph containing a single node with the full process description.
        """
        graph = graph.copy()

        topo_sort = list(nx.topological_sort(graph))
        highest_order_node = topo_sort[-1]

        # Identify all "branching" nodes (junctions, filters, droplet generators).
        target_nodes = sorted([n for n in graph.nodes if n.startswith(('splitting', 'combining', 'droplet', 'filter'))], key=str.lower)
        
        # Find the branching node that is "furthest" from the outlets to structure the description
        # around the most influential, early-stage operations.
        furthest_node, max_distance = None, -1
        for node in target_nodes:
            if nx.has_path(graph, source=node, target=highest_order_node):
                all_paths = list(nx.all_simple_paths(graph, source=node, target=highest_order_node))
                if all_paths:
                    path_length = max(len(p) - 1 for p in all_paths)
                    if path_length > max_distance:
                        max_distance, furthest_node = path_length, node
        
        # Dispatch to the appropriate handler based on the furthest node's type.
        if furthest_node:
            if furthest_node.startswith("combining"): return self._combining_to_text(graph, furthest_node, detailed)
            if furthest_node.startswith("splitting"): return self._splitting_to_text(graph, furthest_node, detailed)
            if furthest_node.startswith("filter"): return self._filter_to_text(graph, furthest_node, detailed)
            if furthest_node.startswith("droplet"): return self._droplet_to_text(graph, furthest_node, detailed)
        
        # If no branching nodes are found, the graph is a simple linear sequence.
        return self._handle_linear_subgraph(graph, detailed)

    def _handle_linear_subgraph(self, graph: nx.DiGraph, detailed: bool, counter: int = 0) -> nx.DiGraph:
        """
        Recursively processes a non-branching, linear chain of components.
        It converts the sequence of operations (e.g., inlet -> mix -> react -> outlet)
        into a single, cohesive narrative sentence fragment.
        
        Args:
            graph: A graph representing a linear sequence of components.
            detailed: Flag to include more hardware details.
            counter: Tracks recursion depth to adjust grammar.
        
        Returns:
            A graph with a single node summarizing the linear process.
        """
        graph = graph.copy()
        # Outlets are terminal nodes, so they are removed to find the next step.
        graph.remove_nodes_from([n for n in graph.nodes if n.startswith('outlet')])
        
        # Find the root of the linear sequence (a node with no inputs).
        root_nodes = [n for n, d in graph.in_degree() if d == 0]
        if not root_nodes: return graph # Should not happen in a DAG fragment
        node_name = root_nodes[0]
        node_attributes = graph.nodes[node_name]

        # Initialize the description if the root is an inlet.
        if node_name.startswith("inlet"):
            inlet_number = int(node_name.split("_")[1])
            prefix = f"a {self._number_prefix(inlet_number-1)} " if inlet_number > 1 else "a "
            node_attributes.update({
                "action": "", "output_type": f"{prefix}fluid", "number_nodes": 0
            })

        # Base case: If only one node is left, the sequence is fully described.
        if len(graph.nodes()) <= 1:
            if counter > 0:
                node_attributes["action"] += "."
            return graph

        # Get the next component in the linear chain.
        successor = list(graph.successors(node_name))[0]
        succ_attrs = graph.nodes[successor]

        # Extract the current state of the description.
        prev_action = node_attributes.get("action", "")
        prev_output_type = node_attributes.get("output_type", "")
        prev_num_nodes = node_attributes.get("number_nodes", 0)

        # Define the descriptive terms for each component type.
        action_details = {
            "chamber": ("let react", "reaction product", {"length": 'μm length', "width": 'μm width'}),
            "mixer": ("mix", "mixture", {"num_turnings": ' turnings'}),
            "delay": ("delay", "delayed liquid", {"num_turnings": ' turnings'}),
        }

        # Generate the next part of the sentence based on the successor's type.
        for comp_type, (verb, product, attrs) in action_details.items():
            if successor.startswith(comp_type):
                module_details = ''
                # Add hardware details if they are non-default or if 'detailed' mode is on.
                if not self._check_default_values(successor, succ_attrs):
                    details_str = " and ".join([f'{succ_attrs[k]}{v}' for k, v in attrs.items()])
                    module_details = f' (using {comp_type}_synonym with {details_str})'
                elif detailed and random.random() < 0.5:
                    module_details = f' (using {comp_type}_synonym)'

                # Construct the action phrase, handling grammar for the start of a sentence vs. mid-sentence.
                if counter == 0:
                    # Special handling for "chamber" due to grammar
                    if comp_type == "chamber":
                        base_action = f"Let {prev_output_type} react{module_details}"
                    # Standard handling for all other types
                    else:
                        base_action = f"{verb.capitalize()} {prev_output_type}{module_details}"
                    
                    succ_attrs["action"] = base_action if not prev_action else f"{prev_action} Then, {base_action.lower()}"
                else:
                    succ_attrs["action"] = f"{prev_action}, then {verb}{module_details}"
                
                # Update the description of what the fluid has become (e.g., "the mixture").
                succ_attrs["output_type"] = f"the {product}"
                break
        
        # Add more context to the output type for clarity if the input was a special stream (e.g., "the mixture of the smaller particle stream").
        match = re.search(r"(\w+\s+particle stream.*|\w+\s+stream.*|droplets)", prev_output_type)
        if match:
             succ_attrs["output_type"] += f' of the {match.group(1)}'

        # Update attributes for the next recursive step.
        succ_attrs["number_nodes"] = prev_num_nodes + 1
        graph.remove_node(node_name)
        return self._handle_linear_subgraph(graph, detailed, counter + 1)

    def _combining_to_text(self, graph: nx.DiGraph, furthest_node: str, detailed: bool) -> nx.DiGraph:
        """
        Handles a combining unit by recursively processing its multiple input paths.
        It generates descriptions for each incoming branch, then weaves them together
        into a single narrative describing the combination event.
        
        Args:
            graph: The current microfluidic chip graph.
            furthest_node: The ID of the combining unit being processed.
            detailed: Flag to include more hardware details.
        
        Returns:
            The result of the next recursive call to _graph_to_text.
        """
        graph = graph.copy()

        # --- 1. Identify and Process Predecessor Subgraphs ---
    
        # Find all ancestor nodes that feed into the combining unit.
        predecessors = list(nx.ancestors(graph, furthest_node))
        predecessor_subgraph = graph.subgraph(predecessors).copy()

        # Recursively generate text for each independent path leading to the combination point.
        processed_subgraphs = []
        for component in nx.weakly_connected_components(predecessor_subgraph):
            component_subgraph = predecessor_subgraph.subgraph(component).copy()
            # _handle_linear_subgraph will condense this path into a single summary node.
            processed_subgraphs.append(self._handle_linear_subgraph(component_subgraph, detailed))

        # Remove the already-processed ancestor nodes from the main graph.
        graph.remove_nodes_from(predecessors)

        # Sort the processed subgraphs to ensure a deterministic and logical narrative order.
        # The sort is based on complexity (number of nodes) first, then alphabetically.
        def get_sort_key(subg):
            # Helper to extract sort criteria from the single-node subgraph.
            node_name, node_data = list(subg.nodes(data=True))[0]
            return -node_data.get('number_nodes', 0), node_name

        processed_subgraphs.sort(key=get_sort_key)
        
        # --- 2. Initialize Attributes and Counters for Text Generation ---

        node_attributes = graph.nodes[furthest_node]
        
        # Optionally add specific hardware details about the junction types used.
        module_details = ''
        if detailed and random.random() < 0.5:
            module_details = f' (using {self._describe_junctions(node_attributes["junction_types"])})'

        # Initialize the attributes for the new combined node.
        node_attributes.update({
            "action": "",
            "output_type": "the combination",
            "number_nodes": 1,
        })

        # These lists and counters will help build the final description from the different incoming branches.
        further_fluids = []
        interim_actions = []
        number_inlets = 0
        counter = 0          # Counts incoming "solutions" from other processes.
        further_counter = 0  # Counts incoming streams/droplets from other processes.

        # --- 3. Gather and Categorize Information from Sorted Subgraphs ---
        
        for subgraph in processed_subgraphs:
            # Extract the summary data from the single node representing a processed path.
            node_name, attrs = list(subgraph.nodes(data=True))[0]
            num_nodes = int(attrs.get("number_nodes", 0))
            output_type = attrs.get("output_type", "")
            action = attrs.get("action", "")
            
            is_stream_or_droplet = "stream" in output_type or "droplets" in output_type

            # Categorize the incoming branch to decide how to describe it.
            if num_nodes > 0 and not is_stream_or_droplet:
                # This branch represents a product of a previous process (e.g., a mixture).
                if action: interim_actions.append(action)
                counter += 1
                node_attributes["number_nodes"] += num_nodes
                
            elif node_name.startswith(("splitstream", "separatedstream")):
                # This branch is a stream from a split or filter.
                further_fluids.append(output_type)
                
            elif is_stream_or_droplet:
                # This branch is a generated stream or set of droplets.
                if action: interim_actions.append(action)
                further_fluids.append(output_type)
                further_counter += 1
                node_attributes["number_nodes"] += num_nodes
                
            else:
                # This branch is a direct fluid inlet.
                number_inlets += 1

        # --- 4. Generate the Final Descriptive Text ---
        
        # First, describe any parallel processes that occurred using "In the meantime...".
        if interim_actions:
            node_attributes["action"] = " In the meantime, ".join(filter(None, interim_actions))

        # Second, generate the main 'combine' action clause, with grammar dependent on what is being combined.
        action_clause = ""
        if number_inlets == 0:
            if counter == 0:
                action_clause = ' Then, combine'
            else:
                solution_text = 'the solution' if counter == 1 else f'the {counter} solutions'
                action_clause = f' Then, combine {solution_text}'
        else:
            if counter == 0:
                inlet_text = f'{number_inlets} fluid{"s" if number_inlets > 1 else ""}'
                action_clause = f' Then, combine {inlet_text}' if further_fluids else f'Combine {inlet_text}'
            elif counter == 1:
                inlet_text = f'{number_inlets} other fluid{"s" if number_inlets > 1 else ""}'
                action_clause = f' Then, combine the solution with {inlet_text}'
            else:
                inlet_text = f'{number_inlets} other fluid{"s" if number_inlets > 1 else ""}'
                action_clause = f' Then, combine the {counter} solutions with {inlet_text}'

        # Third, append details about any other incoming fluids.
        if further_fluids:
            further_action_text = ", and ".join(f"with {fluid}" for fluid in further_fluids)
            if "with" in action_clause:
                action_clause += f" and {further_action_text}"
            elif counter == 0 and number_inlets == 0:
                action_clause += ' ' + further_action_text.replace(", and", "", 1)
            else:
                action_clause += f" {further_action_text}"

            # Apply specific phrasing adjustments
            if counter == 0:
                if number_inlets == 0:
                    action_clause = action_clause.replace("with", "", 1)
                action_clause = action_clause.replace("fluid", "other fluid")

        # Finalize the action clause with hardware details and a period.
        action_clause += f"{module_details}."

        ## --- 5. Finalize Node Attributes and Return for Next Recursion ---
        
        # Append the generated action strings to the node's attributes.
        node_attributes["action"] += action_clause

        # Relabel the node with a prefix.
        mapping = {furthest_node: f'aaac-done_{furthest_node}'}
        graph = nx.relabel_nodes(graph, mapping)

        return self._graph_to_text(graph, detailed)

    def _splitting_to_text(self, graph: nx.DiGraph, furthest_node: str, detailed: bool) -> nx.DiGraph:
        """
        Handles a splitting unit by recursively processing its multiple output paths
        and composing the results into a cohesive narrative. It describes the incoming
        process, the split action, and then the separate processes for each resulting stream.
        
        Args:
            graph: The current microfluidic chip graph.
            furthest_node: The ID of the splitting unit being processed.
            detailed: Flag to include more hardware details.
        
        Returns:
            A new graph containing a single node with the full process description.
        """
        graph = graph.copy()
        module_details = f' (using {self._describe_junctions(graph.nodes[furthest_node]["junction_types"])})' if detailed and random.random() < 0.5 else ''
        outgoing_edge_data = {neighbor: data for _, neighbor, data in graph.edges(furthest_node, data=True)}
        
        # First, process the single path that leads into the splitting junction.
        predecessors = list(nx.ancestors(graph, furthest_node))
        incoming_subgraph = self._handle_linear_subgraph(graph.subgraph(predecessors).copy(), detailed)
        
        # Remove the parts of the graph that have already been processed.
        graph.remove_nodes_from(predecessors + [furthest_node])

        # Identify and sort the separate downstream branches for deterministic processing.
        weakly_connected_components = sorted(
            nx.weakly_connected_components(graph),
            key=lambda c: (len(self._filter_nodes(c)), list(nx.topological_sort(graph.subgraph(self._filter_nodes(c))))[0].lower() if self._filter_nodes(c) else "")
        )

        respond_graphs, next_splitting_count, stream_counter = [], 0, 0
        # Process each downstream branch (component) separately.
        for i, component in enumerate(weakly_connected_components):
            subgraph = graph.subgraph(component).copy()

            # Add extra context if a stream from a previous splitting junction is used as an input here.
            splitstream_nodes = [node for node in component if node.startswith("splitstream")]
            for node in splitstream_nodes:
                # Node name format: e.g., "splitstream_..._splitting_unit_3"
                # The ID is the last part of the original furthest_node name.
                splitting_junction_id = int(node.split("_")[5])
                if " junction" not in subgraph.nodes[node]["output_type"]:
                    context = f' of the {self._number_prefix(splitting_junction_id - 1)} splitting junction_synonym'
                    subgraph.nodes[node]["output_type"] += context

                    
            outgoing_nodes = [node for node in outgoing_edge_data if node in component]

            # Perform a topological sort to find the highest-order node.
            topological_order = list(nx.topological_sort(subgraph))
            highest_order_node = topological_order[-1]

            # Sort the outgoing nodes based on their distance from the end of the subgraph.
            outgoing_nodes = sorted(
                outgoing_nodes,
                key=lambda node: nx.shortest_path_length(subgraph, target=highest_order_node, source=node) if nx.has_path(subgraph, node, highest_order_node) else -1,
                reverse=True
            )
            
            # Insert a placeholder node (e.g., 'splitstream_0_0_...') to represent the start of this specific output branch.
            # This node is given a descriptive name like "the first stream".
            for j, out_node in enumerate(outgoing_nodes):
                new_node_name = f"splitstream_{i}_{j}_{furthest_node}"
                node_attrs = {
                    "action": "",
                    "output_type": f"the {self._number_prefix(stream_counter)} stream",
                    "number_nodes": 0
                }
                if next_splitting_count > 0:
                    split_id = int(furthest_node.split("_")[2]) - 1
                    node_attrs["output_type"] += f' of the {self._number_prefix(split_id)} splitting junction_synonym'
                
                subgraph.add_node(new_node_name, **node_attrs)
                subgraph.add_edge(new_node_name, out_node, **outgoing_edge_data[out_node])
                stream_counter += 1
            
            next_splitting_count += sum(1 for node in component if node.startswith("splitting"))
            # Recursively call the main function to generate the text for this entire branch.
            respond_graphs.append(self._graph_to_text(subgraph, detailed))
        
        # Retrieve the description of the fluid/process that is being split.
        incoming_props = list(incoming_subgraph.nodes(data=True))[0][1]
        
        # Start building the final combined action description.
        action_parts = []
        if incoming_props.get("number_nodes", 0) == 0:
             action_parts.append(f"Split {incoming_props['output_type']} into {len(outgoing_edge_data)} streams{module_details}.")
        else:
            action_parts.append(f"{incoming_props['action']} Then, split {incoming_props['output_type']} into {len(outgoing_edge_data)} streams{module_details}.")

        number_nodes = incoming_props.get("number_nodes", 0) + 1

        # Collect the results from all the processed downstream branches.
        outlets = []
        res_action_parts = []
        for res_graph in respond_graphs:
            res_node = list(res_graph.nodes)[0]
            res_props = res_graph.nodes[res_node]
            if res_node.startswith("splitstream"):
                # If a stream just goes to an outlet, describe it simply.
                outlets.append(f'Route {res_props["output_type"]} to outlet_synonym.')
            else:
                # Otherwise, append the full description of that branch's process.
                res_action_parts.append(res_props["action"])

            number_nodes += res_props["number_nodes"]
        
        # Assemble the final, complete action string.
        final_action = " ".join(action_parts)

        # Add a summary of outlet routing if multiple streams go to outlets.
        if len(outlets) > 1:
            final_action += f' Route {len(outlets)} streams to separate outlet_synonyms. '
        elif len(outlets) == 1:
            final_action += f' {outlets[0]} '
        else: 
            final_action += ' '

        final_action += " ".join(res_action_parts)

        # Return a new single-node graph representing the fully described splitting process.
        combined_graph = nx.DiGraph()
        combined_graph.add_node("combined_node", action=final_action, output_type="solution", number_nodes=number_nodes)
        return combined_graph
    
    def _filter_to_text(self, graph: nx.DiGraph, furthest_node: str, detailed: bool) -> nx.DiGraph:
        """
        Handles a filter component by processing its two distinct output paths
        (smaller and larger particles) and composing the results into a narrative.
        This follows the same recursive pattern as splitting.
        
        Args:
            graph: The current microfluidic chip graph.
            furthest_node: The ID of the filter being processed.
            detailed: Flag to include more hardware details.
        
        Returns:
            A new graph containing a single node with the full process description.
        """
        graph = graph.copy()
        filter_attrs = graph.nodes[furthest_node]
        module_details = ''
        # Add specific parameter details if non-default.
        if not self._check_default_values(furthest_node, filter_attrs):
            module_details = f' (using filter_synonym with {filter_attrs["critical_particle_diameter"]}μm critical particle diameter)'
        elif detailed and random.random() < 0.5:
            module_details = f' (using filter_synonym)'

        outgoing_edge_data = {neighbor: data for _, neighbor, data in graph.edges(furthest_node, data=True)}
        
        predecessors = list(nx.ancestors(graph, furthest_node))
        incoming_subgraph = self._handle_linear_subgraph(graph.subgraph(predecessors).copy(), detailed)
        
        graph.remove_nodes_from(predecessors + [furthest_node])
        
        respond_graphs = []
        next_filter_count = 0

        # Sort downstream components for deterministic processing.
        weakly_connected_components = sorted(
            nx.weakly_connected_components(graph),
            key=lambda c: (len(self._filter_nodes(c)), list(nx.topological_sort(graph.subgraph(self._filter_nodes(c))))[0].lower() if self._filter_nodes(c) else "")
        )

        for i, component in enumerate(weakly_connected_components):
            subgraph = graph.subgraph(component).copy()

            # Add context if a particle stream from a previous filter is used here.
            separatedstream_nodes = [node for node in component if node.startswith("separatedstream")]
            for node in separatedstream_nodes:
                # Node name format: e.g., "separatedstream_..._filter_3"
                # The ID is the last part of the original furthest_node name.
                filter_id = int(node.split("_")[4])
                if not re.search(r"of the \w+ filter", subgraph.nodes[node]["output_type"]):
                    context = f' of the {self._number_prefix(filter_id - 1)} filter_synonym'
                    subgraph.nodes[node]["output_type"] += context

            outgoing_nodes = [node for node in outgoing_edge_data if node in component]

            # Perform a topological sort to find the highest-order node.
            topological_order = list(nx.topological_sort(subgraph))
            highest_order_node = topological_order[-1]

            # Sort outgoing nodes by their distance from the highest-order node.
            outgoing_nodes = sorted(
                outgoing_nodes,
                key=lambda node: nx.shortest_path_length(subgraph, target=highest_order_node, source=node) if nx.has_path(subgraph, node, highest_order_node) else -1,
                reverse=True
            )
            
            # Insert placeholder nodes for the "smaller" and "larger" particle streams.
            for j, out_node in enumerate(outgoing_nodes):
                particle_type = outgoing_edge_data[out_node]["filter_connection_type"]
                new_node_name = f"separatedstream_{i}_{j}_{furthest_node}"

                filter_info = ""
                if next_filter_count > 0:
                    filter_id = int(furthest_node.split("_")[1]) - 1
                    filter_info = f' of the {self._number_prefix(filter_id)} filter_synonym'
                
                subgraph.add_node(new_node_name, action="", output_type=f"the {particle_type} particle stream{filter_info}", number_nodes=0)
                subgraph.add_edge(new_node_name, out_node, **outgoing_edge_data[out_node])

            next_filter_count += sum(1 for node in component if node.startswith("filter"))
                
            # Recursively generate text for each downstream branch.
            respond_graphs.append(self._graph_to_text(subgraph, detailed))
        
        incoming_props = list(incoming_subgraph.nodes(data=True))[0][1]
        
        # Assemble the final description.
        action_parts = []
        if incoming_props.get("number_nodes", 0) == 0:
            action_parts.append(f"Separate smaller and larger particles of {incoming_props['output_type']}{module_details}.")
        else:
            action_parts.append(f"{incoming_props['action']} Separate smaller and larger particles of {incoming_props['output_type']}{module_details}.")
        
        number_nodes = incoming_props.get("number_nodes", 0) + 1

        for res_graph in respond_graphs:
            res_props = list(res_graph.nodes(data=True))[0][1]
            action_parts.append(res_props["action"])
            number_nodes += res_props["number_nodes"]
        
        # Return a new single-node graph representing the fully described filtering process.
        combined_graph = nx.DiGraph()
        combined_graph.add_node("combined_node", action=" ".join(action_parts), output_type="solution", number_nodes=number_nodes)
        return combined_graph

    def _droplet_to_text(self, graph: nx.DiGraph, furthest_node: str, detailed: bool) -> nx.DiGraph:
        """
        Handles a droplet generator by processing its continuous and dispersed
        phase inputs and composing a description of the droplet formation process.
        Like combining, this involves two distinct input paths.
        
        Args:
            graph: The current microfluidic chip graph.
            furthest_node: The ID of the droplet generator being processed.
            detailed: Flag to include more hardware details.
        
        Returns:
            The result of the next recursive call to _graph_to_text.
        """
        graph = graph.copy()
        continuous_subgraph, dispersed_subgraph = None, None
        nodes_to_remove = set()

        # 1. Isolate and process the incoming continuous and dispersed fluid paths separately.
        for source, _, attrs in graph.in_edges(furthest_node, data=True):
            predecessors = [source] + list(nx.ancestors(graph, source))
            nodes_to_remove.update(predecessors)
            sub = graph.subgraph(predecessors).copy()
            if attrs.get("droplet_connection_type") == "continuous":
                continuous_subgraph = self._handle_linear_subgraph(sub, detailed)
            elif attrs.get("droplet_connection_type") == "dispersed":
                dispersed_subgraph = self._handle_linear_subgraph(sub, detailed)
        
        graph.remove_nodes_from(nodes_to_remove)
        
        # 2. Extract the properties from the processed paths.
        cont_name, cont_props = list(continuous_subgraph.nodes(data=True))[0]
        disp_name, disp_props = list(dispersed_subgraph.nodes(data=True))[0]

        node_attrs = graph.nodes[furthest_node]

        # 3. Build the action, starting with the dispersed phase (the fluid being turned into droplets).
        action = ""
        number_nodes = 1 + disp_props.get("number_nodes", 0)

        if disp_props.get("number_nodes", 0) > 0:
            # Case: The dispersed phase has its own prior processing steps.
            disp_action = disp_props.get("action", "")
            if "stream" not in disp_name:
                action = f"{disp_action} Then, form droplets from this"
            else:
                action = f"{disp_action} Then, form droplets from {disp_props.get('output_type')}"
        else:
            # Case: The dispersed phase is a simple input.
            action = f"Form droplets from {disp_props.get('output_type')}"

        # 4. Incorporate the continuous phase into the description.
        number_nodes += cont_props.get("number_nodes", 0)
        cont_nodes_count = cont_props.get("number_nodes", 0)

        if cont_nodes_count > 0:
            # Case: The continuous phase also has its own processing steps.
            # This logic decides how the two process descriptions are woven together grammatically.
            is_special_case = (
                not ("stream" in cont_name or "done" in cont_name)
                and cont_nodes_count <= disp_props.get("number_nodes", 0)
            )

            if is_special_case:
                # Special Case: Continuous phase process is simpler and described as a secondary step.
                add_info = ' for the former droplet formation' if "droplet" in cont_props.get("action", "") else ''
                cont_action_text = cont_props.get("action", "a.")
                action += f'. For creating the continuous phase, {cont_action_text[0].lower() + cont_action_text[1:-1]} and use this as continuous phase{add_info}.'
            else:
                # Normal Case: Continuous phase is the primary process.
                add_info = ''
                if "done_drop" in cont_name:
                    drop_index = int(cont_name.split("_")[3]) - 1
                    add_info = f' of the {self._number_prefix(drop_index)} droplet formation'

                action = f"{cont_props.get('action', '')} In the meantime, {action} (using {cont_props.get('output_type')}{add_info} as continuous phase)."

        elif "stream" in cont_props.get("output_type", ""):
            # Case: Continuous phase is a simple stream from a previous split/filter.
            action += f" (using {cont_props.get('output_type')} as continuous phase)."
        else:
            # Case: Continuous phase is a simple input (e.g., from an inlet) -> (dont mention specific continuous at all)
            action += "."
        
        # 5. Finalize the node and continue the recursive graph-to-text conversion.
        node_attrs.update({
            "action": action,
            "output_type": "the droplets",
            "number_nodes": number_nodes
        })
        
        # Relabel the node to ensure it's processed correctly in the recursion.
        graph = nx.relabel_nodes(graph, {furthest_node: f'aaad-done_drop_{furthest_node}'})
        return self._graph_to_text(graph, detailed)
    
    def _check_default_values(self, node_name: str, node_attributes: Dict) -> bool:
        """Checks if a component's attributes match the predefined defaults to decide if they need to be mentioned in the prompt."""
        defaults_map = {'mixer': {'num_turnings': self.DEFAULT_ATTRIBUTES['num_turnings']},'delay': {'num_turnings': self.DEFAULT_ATTRIBUTES['num_turnings']}, 'chamber': {'length': self.DEFAULT_ATTRIBUTES['length'], 'width': self.DEFAULT_ATTRIBUTES['width']}, 'filter': {'critical_particle_diameter': self.DEFAULT_ATTRIBUTES['critical_particle_diameter']}}
        for comp_type, defaults in defaults_map.items():
            if node_name.startswith(comp_type):
                return all(node_attributes.get(attr) == val for attr, val in defaults.items())
        return True

    def _describe_junctions(self, junction_list: List[str]) -> str:
        """Creates a natural language summary of a list of junction types (e.g., "2 T-junctions and 1 Y-junction")."""
        grouped = [f"{len(group)} {k}_synonym{'s' if len(group) > 1 else ''}" for k, g in groupby(junction_list) if (group := list(g))]
        if len(grouped) > 1:
            return ", ".join(grouped[:-1]) + " and " + grouped[-1]
        return grouped[0] if grouped else ""

    def _number_prefix(self, n: int) -> str:
        """Converts an integer into an ordinal word (e.g., 2 -> "second") for more natural language."""
        num_map = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
        return num_map[n] if n < len(num_map) else f"{n+1}th"

    def _filter_nodes(self, component_nodes: List[str], ignore_patterns: Tuple[str, ...] = ('splitstream', 'separatedstream')) -> List[str]:
        """Filters out temporary placeholder nodes (like 'splitstream') from a list of node names."""
        return [node for node in component_nodes if not any(node.startswith(p) for p in ignore_patterns)]

    def _modify_combine_mix_pairs(self, text: str) -> str:
        """Post-processing rule to make language more natural. Condenses "combine... then mix" into a single "mix" action."""
        pattern = r"(\bcombine\b[^.]*)(\.\s*Then, mix the combination)"
        def replacement(match):
            return match.group(1).replace("combine", "mix").replace("Combine", "Mix") if random.choice([True, False]) else match.group(0)
        return re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    def _modify_combine_react_pairs(self, text: str) -> str:
        """Post-processing rule to condense "combine... then react" into a single "let react with" action."""
        pattern = r"([^.]*\bcombine\b.[^.]*)(\.\s*Then, let the combination react)"
        # Replace all occurrences of the pattern in the string
        def replacement(m):
            # Randomly decide whether to replace
            if random.choice([True, False]):
                # Extract everything before and after "combine"
                before_combine = re.split(r'\bcombine\b', m.group(1), flags=re.IGNORECASE)[0]
                after_combine = re.split(r'\bcombine\b', m.group(1), flags=re.IGNORECASE)[-1].strip()
                
                # Extract everything after the first "with"
                if "with" in after_combine:
                    parts = after_combine.split("with", 1)
                    stuff_after_combine = parts[0].strip()  # Stuff after "combine"
                    stuff_after_with = parts[1].strip()  # Stuff after the first "with"

                    if before_combine:
                        return f"{before_combine}let {stuff_after_combine} react with {stuff_after_with}"
                    else:
                        return f"Let {stuff_after_combine} react with {stuff_after_with}"
                else:
                    # If there is no "with", use just the stuff after "combine"
                    if before_combine:
                        return f"{before_combine}let {after_combine} react"
                    else:
                        return f"Let {after_combine} react"
            else:
                return m.group(0)

        return re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    def _replace_with_random_synonym(self, text: str, synonyms_dict: Dict) -> str:
        """Replaces placeholder module names (e.g., 'mixer_synonym') with a random synonym."""
        for key, synonyms in synonyms_dict.items():
            text = text.replace(f"{key}_synonym", random.choice(synonyms))
        return text
    
    def _replace_with_random_action_synonyms(self, text: str) -> str:
        """Replaces action keywords with randomly chosen synonyms to increase linguistic diversity."""
        selected_synonyms = {}
        def replace_match(match):
            word, word_lower = match.group(0), match.group(0).lower()
            if word_lower in self.ACTION_SYNONYMS:
                if word_lower not in selected_synonyms:
                    # Choose a random synonym for this action word for the entire prompt.
                    chosen_index = random.randint(0, len(self.ACTION_SYNONYMS[word_lower]) - 1)
                    selected_synonyms[word_lower] = self.ACTION_SYNONYMS[word_lower][chosen_index]
                    # Ensure related words (e.g., "mix" and "mixture") use corresponding synonyms.
                    if word_lower == "mix": selected_synonyms["mixture"] = self.ACTION_SYNONYMS["mixture"][chosen_index]
                    elif word_lower == "delay": selected_synonyms["delayed"] = self.ACTION_SYNONYMS["delayed"][chosen_index]
                    elif word_lower == "form droplets": selected_synonyms["droplet formation"] = self.ACTION_SYNONYMS["droplet formation"][chosen_index]
                synonym = selected_synonyms[word_lower]
                # Preserve capitalization.
                return synonym.capitalize() if word[0].isupper() else synonym
            return word
        return re.sub(r'\b(' + '|'.join(self.ACTION_SYNONYMS.keys()) + r')\b', replace_match, text, flags=re.IGNORECASE)

    def _generate_full_prompt(self, action_description: str) -> str:
        """Combines the generated action description with a random introductory phrase to create the final prompt."""
        beginning = random.choice(self.PROMPT_BEGINNINGS)
        action_description = action_description[0].lower() + action_description[1:] if action_description else ""
        return f"{beginning} {action_description}"