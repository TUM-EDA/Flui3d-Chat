import json
import jsonschema
import re
import networkx as nx
from collections import defaultdict, deque
from itertools import permutations
import copy

class MicrofluidicDesignComparator:
    def __init__(self, schema_path):
        """Initialize the comparator by loading the JSON schema."""
        self.schema = self._load_json_schema(schema_path)
        # Define base rewards and penalties (tune these as needed)
        self.REWARD_MAX_STRUCTURE = 5.0
        self.REWARD_CONNECTIONS = 20.0
        self.REWARD_JUNCTIONS = 20.0
        self.REWARD_PARAMS = 10.0
        self.PENALTY_CONNECTION_MISMATCH = 4      # Penalty per missing/extra connection
        self.PENALTY_JUNCTION_UNIMPLEMENTED = 8   # Penalty per connection listed but not possible
        self.PENALTY_JUNCTION_SPURIOUS = 8        # Penalty per path via junctions not listed
        self.PENALTY_JUNCTION_EXTRA = 2           # Penalty per junction beyond minimum needed
        self.PENALTY_PARAM_MISMATCH = 2           # Penalty per incorrect parameter value

    def _load_json_schema(self, schema_path):
        """Load the JSON schema from a file."""
        try:
            with open(schema_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Schema file not found at {schema_path}")
            raise
        except json.JSONDecodeError:
            print(f"Error: Schema file at {schema_path} is not valid JSON.")
            raise

    def _validate_json(self, instance, design_name="Design"):
        """Validate a JSON instance against the schema."""
        try:
            jsonschema.validate(instance=instance, schema=self.schema)
            return True
        except jsonschema.ValidationError as e:
            print(f"{design_name} validation error: {e.message}")
            return False
        except Exception as e:
            print(f"An unexpected error occurred during validation of {design_name}: {e}")
            return False

    def _parse_component_id(self, component_id):
        """Parses a component ID into type, number, and optional suffix."""
        parts = component_id.split('_')
        if len(parts) < 2:
            return None, None, None # Invalid format
        
        component_type = parts[0]
        try:
            component_num = int(parts[1])
        except ValueError:
             return None, None, None # Invalid format (number part isn't integer)

        suffix = None
        if len(parts) > 2:
            # Handle specific cases like 'filter_X_smaller', 'droplet_X_continuous'
            if component_type in ["filter", "droplet"]:
                 suffix = parts[2]
            else:
                 pass

        return component_type, component_num, suffix

    def _get_connection_representation(self, conn_dict):
        """Converts a connection dict to a comparable tuple representation."""
        src_type, src_num, src_suffix = self._parse_component_id(conn_dict.get("source", ""))
        tgt_type, tgt_num, tgt_suffix = self._parse_component_id(conn_dict.get("target", ""))

        if src_type is None or tgt_type is None:
            return None # Invalid connection format

        # Return a tuple that ignores the number for comparison purposes initially
        # We'll use the numbers separately for mapping
        return (
            (src_type, src_suffix), # Source type/suffix pair
            src_num,
            (tgt_type, tgt_suffix), # Target type/suffix pair
            tgt_num
        )

    def _apply_mapping_to_connections(self, connections_set, mapping):
        """Applies a component number mapping to a set of connection representations."""
        mapped_connections = set()
        for (src_type_suffix, src_num, tgt_type_suffix, tgt_num) in connections_set:
            src_type, src_suffix = src_type_suffix
            tgt_type, tgt_suffix = tgt_type_suffix

            mapped_src_num = mapping.get(src_type, {}).get(src_num, src_num) # Default to original if no map
            mapped_tgt_num = mapping.get(tgt_type, {}).get(tgt_num, tgt_num) # Default to original if no map

            mapped_connections.add((
                (src_type, src_suffix),
                mapped_src_num,
                (tgt_type, tgt_suffix),
                mapped_tgt_num
            ))
        return mapped_connections

    def _compare_connections(self, reference_design, generated_design):
        """
        Compares the connections sections, allowing for consistent component renaming.
        Returns the score and the best mapping found.
        """
        ref_conns_raw = reference_design.get("connections", [])
        gen_conns_raw = generated_design.get("connections", [])

        ref_conns = set()
        ref_components = defaultdict(set) # {comp_type: {num1, num2}}
        for conn in ref_conns_raw:
            rep = self._get_connection_representation(conn)
            if rep:
                ref_conns.add(rep)
                ref_components[rep[0][0]].add(rep[1]) # Add source number by type
                ref_components[rep[2][0]].add(rep[3]) # Add target number by type


        gen_conns = set()
        gen_components = defaultdict(set) # {comp_type: {num1, num2}}
        for conn in gen_conns_raw:
            rep = self._get_connection_representation(conn)
            if rep:
                gen_conns.add(rep)
                gen_components[rep[0][0]].add(rep[1]) # Add source number by type
                gen_components[rep[2][0]].add(rep[3]) # Add target number by type

        best_mapping = {}
        max_matched_count = -1

        # Generate potential mappings only for types present in both designs
        types_to_map = ref_components.keys() & gen_components.keys()
        
        # List to hold permutations for each type
        type_permutations = []
        
        valid_mapping_possible = True
        for comp_type in types_to_map:
            ref_nums = sorted(list(ref_components[comp_type]))
            gen_nums = sorted(list(gen_components[comp_type]))

            if len(ref_nums) != len(gen_nums):
                 # Cannot have a perfect 1-to-1 mapping if counts differ.
                 # This scenario implies components are fundamentally missing/extra
                 # We will handle this via mismatch penalties later, proceed without permutations for this type.
                 pass
            elif not ref_nums:
                 continue # No components of this type
            else:
                 # Generate all possible mappings (permutations) for this type
                 perms = list(permutations(gen_nums))
                 type_permutations.append({'type': comp_type, 'ref_nums': ref_nums, 'perms': perms})
                 
        # If no types need mapping or permutations are impossible
        if not type_permutations:
             num_combinations = 1 # Only one combination: the empty mapping
        else:
             num_combinations = 1
             for p_info in type_permutations:
                  num_combinations *= len(p_info['perms'])
        
        # Limit combinations to avoid excessive computation
        MAX_COMBINATIONS = 100000 # Adjust as needed
        if num_combinations > MAX_COMBINATIONS:
            print(f"Warning: Too many mapping combinations ({num_combinations}). Skipping exhaustive check.")
            # Fallback: maybe check identity mapping only or a limited random sample?
            # For now, we'll just use an empty mapping which results in direct comparison.
            num_combinations = 1 
            type_permutations = []


        # Iterate through all combinations of permutations across types
        if num_combinations == 1 and not type_permutations:
            # Handle the case with no permutations needed or fallback
            current_mapping = {} # Empty or identity mapping effect
            mapped_gen_conns = self._apply_mapping_to_connections(gen_conns, current_mapping)
            matched_count = len(ref_conns.intersection(mapped_gen_conns))
            max_matched_count = matched_count
            best_mapping = current_mapping
        else:
            # Build combinations iteratively or using product
            indices = [0] * len(type_permutations)
            for i in range(num_combinations):
                current_mapping = {}
                # Construct mapping for this combination
                temp_indices = list(indices) # Work with a copy
                for p_idx in range(len(type_permutations) - 1, -1, -1):
                    p_info = type_permutations[p_idx]
                    current_perm_index = temp_indices[p_idx]
                    comp_type = p_info['type']
                    ref_nums = p_info['ref_nums']
                    chosen_perm = p_info['perms'][current_perm_index]
                    current_mapping[comp_type] = dict(zip(chosen_perm, ref_nums))
    	        
                # Evaluate this mapping
                mapped_gen_conns = self._apply_mapping_to_connections(gen_conns, current_mapping)
                matched_count = len(ref_conns.intersection(mapped_gen_conns))

                if matched_count > max_matched_count:
                    max_matched_count = matched_count
                    best_mapping = copy.deepcopy(current_mapping) # Store the best mapping

                # Increment indices for next combination
                for j in range(len(indices)):
                    indices[j] += 1
                    if indices[j] < len(type_permutations[j]['perms']):
                        break # Stop incrementing if we haven't rolled over
                    indices[j] = 0 # Roll over

        # Calculate score based on the best mapping found
        final_mapped_gen_conns = self._apply_mapping_to_connections(gen_conns, best_mapping)
        
        missing_conns = ref_conns - final_mapped_gen_conns
        extra_conns = final_mapped_gen_conns - ref_conns

        num_mismatches = len(missing_conns) + len(extra_conns)
        score = max(0.0, self.REWARD_CONNECTIONS - num_mismatches * self.PENALTY_CONNECTION_MISMATCH)

        return score, best_mapping


    def _is_junction_node(self, node_id):
        """Helper to check if a node ID represents a junction."""
        return isinstance(node_id, str) and node_id.startswith("junction_")

    def _calculate_minimum_junctions(self, defined_connections_set):
        """
        Calculates the minimum number of standard T/Y junctions required to
        implement the connections defined in the connections list.

        The logic counts the total number of elementary merge (2->1) and split
        (1->2) operations required at non-junction nodes based on the
        defined connections. Each standard junction performs one such operation.
        """
        if not defined_connections_set:
            return 0

        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        all_nodes = set()

        for src, tgt in defined_connections_set:
            if src: # Check if src is not None or empty
                out_degree[src] += 1
                all_nodes.add(src)
            if tgt: # Check if tgt is not None or empty
                in_degree[tgt] += 1
                all_nodes.add(tgt)

        total_merges_needed = 0
        total_splits_needed = 0

        for node in all_nodes:
            # Junctions are needed based on component connectivity requirements
            if not self._is_junction_node(node):
                if in_degree[node] > 1:
                    # Need N-1 merge operations for N inputs
                    total_merges_needed += (in_degree[node] - 1)
                if out_degree[node] > 1:
                    # Need N-1 split operations for N outputs
                    total_splits_needed += (out_degree[node] - 1)

        # Each elementary merge or split requires one junction.
        min_junctions = total_merges_needed + total_splits_needed
        return min_junctions


    def _compare_junctions(self, generated_design):
        """
        Evaluates the junction setup based on the generated design.
        Checks if junctions correctly implement connections and if they are minimal.
        """
        connections = generated_design.get("connections", [])
        junctions = generated_design.get("junctions", [])

        defined_connections_set = set()
        all_connection_nodes = set()
        for conn in connections:
            src = conn.get("source")
            tgt = conn.get("target")
            if src and tgt:
                defined_connections_set.add((src, tgt))
                all_connection_nodes.add(src)
                all_connection_nodes.add(tgt)

        if not connections and not junctions:
            return self.REWARD_JUNCTIONS
        if not connections and junctions:
             num_extra = len(junctions)
             score = max(0.0, self.REWARD_JUNCTIONS - num_extra * self.PENALTY_JUNCTION_EXTRA)
             return score

        # --- Graph Construction ---
        flow_graph = nx.MultiDiGraph()
        initial_in_degree = defaultdict(int)
        initial_out_degree = defaultdict(int)

        # 1. Start with a graph containing only connections
        for src, tgt in defined_connections_set:
            flow_graph.add_node(src)
            flow_graph.add_node(tgt)
            # Add edge with a default key 0 for MultiDiGraph compatibility
            flow_graph.add_edge(src, tgt, key=0)
            initial_out_degree[src] += 1
            initial_in_degree[tgt] += 1

        # 2. Identify and remove edges that need to be implemented by junctions
        edges_to_remove = set()
        for node in list(flow_graph.nodes()):
            # If node needs merging, remove its incoming connection edges
            if initial_in_degree[node] > 1:
                for u, v, key in flow_graph.in_edges(node, keys=True):
                     edges_to_remove.add((u, v, key))
            # If node needs splitting, remove its outgoing connection edges
            if initial_out_degree[node] > 1:
                for u, v, key in flow_graph.out_edges(node, keys=True):
                     edges_to_remove.add((u, v, key))

        flow_graph.remove_edges_from(list(edges_to_remove))

        # 3. Add junction nodes and edges

        # --- Pre-processing: Create a map of all raw junction definitions for quick lookup ---
        raw_junction_defs_map = {j.get("id"): j for j in junctions if j.get("id")}

        # List to store junctions that pass reciprocity checks
        reciprocity_passed_junctions = []

        # --- Pass 1: Junction-to-Junction Reciprocity Validation ---
        # For each junction, if it connects to other junctions, verify reciprocal connections.
        # If not reciprocal, the current junction is entirely disregarded.
        for current_junc_data in junctions:
            current_junc_id = current_junc_data.get("id")
            if not current_junc_id:
                continue

            is_reciprocity_valid = True  # Assume valid until a check fails

            # Helper function to check if a peer junction reciprocates the connection
            def check_peer_reciprocity(peer_id, current_junc_id_for_peer, is_current_expecting_peer_as_source, is_current_expecting_peer_as_target):
                # peer_id: ID of the other junction.
                # current_junc_id_for_peer: ID of the current junction, which peer_id should reference.
                # is_current_expecting_peer_as_source: True if current expects peer_id to be its source.
                # is_current_expecting_peer_as_target: True if current expects peer_id to be its target.
                
                peer_def = raw_junction_defs_map.get(peer_id)
                if not peer_def:
                    # This case implies an inconsistency if self._is_junction_node(peer_id) was true
                    # but peer_id is not in raw_junction_defs_map. For safety, treat as non-reciprocated.
                    return False

                reciprocated = False
                if is_current_expecting_peer_as_source:  # Current is TARGET of Peer; Peer is SOURCE to Current
                    # Peer must list current_junc_id_for_peer as one of its targets
                    if "source" in peer_def:  # Peer is SPLIT (has target_1, target_2)
                        if peer_def.get("target_1") == current_junc_id_for_peer or \
                        peer_def.get("target_2") == current_junc_id_for_peer:
                            reciprocated = True
                    elif "source_1" in peer_def:  # Peer is MERGE (has one target)
                        if peer_def.get("target") == current_junc_id_for_peer:
                            reciprocated = True
                
                elif is_current_expecting_peer_as_target:  # Current is SOURCE to Peer; Peer is TARGET of Current
                    # Peer must list current_junc_id_for_peer as one of its sources
                    if "source" in peer_def:  # Peer is SPLIT (has one source)
                        if peer_def.get("source") == current_junc_id_for_peer:
                            reciprocated = True
                    elif "source_1" in peer_def:  # Peer is MERGE (has source_1, source_2)
                        if peer_def.get("source_1") == current_junc_id_for_peer or \
                        peer_def.get("source_2") == current_junc_id_for_peer:
                            reciprocated = True
                return reciprocated

            if "source" in current_junc_data:  # Current junction is SPLIT
                peer_src_id = current_junc_data.get("source")
                if peer_src_id and self._is_junction_node(peer_src_id): # Check only if peer is a junction
                    if not check_peer_reciprocity(peer_src_id, current_junc_id, is_current_expecting_peer_as_source=True, is_current_expecting_peer_as_target=False):
                        is_reciprocity_valid = False
                
                if is_reciprocity_valid:
                    peer_tgt1_id = current_junc_data.get("target_1")
                    if peer_tgt1_id and self._is_junction_node(peer_tgt1_id):
                        if not check_peer_reciprocity(peer_tgt1_id, current_junc_id, is_current_expecting_peer_as_source=False, is_current_expecting_peer_as_target=True):
                            is_reciprocity_valid = False
                
                if is_reciprocity_valid:
                    peer_tgt2_id = current_junc_data.get("target_2")
                    if peer_tgt2_id and self._is_junction_node(peer_tgt2_id):
                        if not check_peer_reciprocity(peer_tgt2_id, current_junc_id, is_current_expecting_peer_as_source=False, is_current_expecting_peer_as_target=True):
                            is_reciprocity_valid = False
            
            elif "target" in current_junc_data:  # Current junction is MERGE
                peer_src1_id = current_junc_data.get("source_1")
                if peer_src1_id and self._is_junction_node(peer_src1_id):
                    if not check_peer_reciprocity(peer_src1_id, current_junc_id, is_current_expecting_peer_as_source=True, is_current_expecting_peer_as_target=False):
                        is_reciprocity_valid = False

                if is_reciprocity_valid:
                    peer_src2_id = current_junc_data.get("source_2")
                    if peer_src2_id and self._is_junction_node(peer_src2_id):
                        if not check_peer_reciprocity(peer_src2_id, current_junc_id, is_current_expecting_peer_as_source=True, is_current_expecting_peer_as_target=False):
                            is_reciprocity_valid = False
                
                if is_reciprocity_valid:
                    peer_tgt_id = current_junc_data.get("target")
                    if peer_tgt_id and self._is_junction_node(peer_tgt_id):
                        if not check_peer_reciprocity(peer_tgt_id, current_junc_id, is_current_expecting_peer_as_source=False, is_current_expecting_peer_as_target=True):
                            is_reciprocity_valid = False

            if is_reciprocity_valid:
                reciprocity_passed_junctions.append(current_junc_data)

        all_junction_nodes = set()
        all_junction_sources = set()
        all_junction_targets = set()
        junction_definitions_map = {}

        # --- Pass 2: Only add first junction if multiple junctions try to connect to same source/target ---
        for junc_data in reciprocity_passed_junctions:
            junc_id = junc_data.get("id")

            can_add_this_junction = True

            if "source" in junc_data:
                component_src = junc_data.get("source")
                component_tgt1 = junc_data.get("target_1")
                component_tgt2 = junc_data.get("target_2")

                if component_src and not self._is_junction_node(component_src):
                    if component_src in all_junction_sources:
                        can_add_this_junction = False
                
                if can_add_this_junction and component_tgt1 and not self._is_junction_node(component_tgt1):
                    if component_tgt1 in all_junction_targets:
                        can_add_this_junction = False
                
                if can_add_this_junction and component_tgt2 and not self._is_junction_node(component_tgt2):
                    if component_tgt2 in all_junction_targets:
                        can_add_this_junction = False

                if can_add_this_junction:
                    if component_src and not self._is_junction_node(component_src):
                        all_junction_sources.add(component_src)
                    if component_tgt1 and not self._is_junction_node(component_tgt1):
                        all_junction_targets.add(component_tgt1)
                    if component_tgt2 and not self._is_junction_node(component_tgt2):
                        all_junction_targets.add(component_tgt2)
                    
            elif "target" in junc_data:
                component_src1 = junc_data.get("source_1")
                component_src2 = junc_data.get("source_2")
                component_tgt = junc_data.get("target")

                if component_src1 and not self._is_junction_node(component_src1):
                    if component_src1 in all_junction_sources:
                        can_add_this_junction = False
                
                if can_add_this_junction and component_src2 and not self._is_junction_node(component_src2):
                    if component_src2 in all_junction_sources:
                        can_add_this_junction = False
                
                if can_add_this_junction and component_tgt and not self._is_junction_node(component_tgt):
                    if component_tgt in all_junction_targets:
                        can_add_this_junction = False

                if can_add_this_junction:
                    if component_src1 and not self._is_junction_node(component_src1):
                        all_junction_sources.add(component_src1)
                    if component_src2 and not self._is_junction_node(component_src2):
                        all_junction_sources.add(component_src2)
                    if component_tgt and not self._is_junction_node(component_tgt):
                        all_junction_targets.add(component_tgt)

            if can_add_this_junction:
                all_junction_nodes.add(junc_id)
                junction_definitions_map[junc_id] = junc_data
                flow_graph.add_node(junc_id)

        # Add junction edges
        # Iterate over the successfully validated and processed junctions.
        for current_junc_def in junction_definitions_map.values():
            current_junc_id = current_junc_def.get("id")

            # --- Processing for a SPLIT junction (source -> current_junc_id -> target_1, target_2) ---
            if "source" in current_junc_def:
                component_src = current_junc_def.get("source")
                component_tgt1 = current_junc_def.get("target_1")
                component_tgt2 = current_junc_def.get("target_2")

                # Edge from component_src to current_junc_id
                if component_src:
                    if self._is_junction_node(component_src):
                        # If component_src is a junction, it must be in junction_definitions_map (i.e., valid & reciprocal).
                        if component_src in junction_definitions_map:
                            flow_graph.add_edge(component_src, current_junc_id)
                    else: # component_src is a regular component
                        flow_graph.add_edge(component_src, current_junc_id)

                # Edge from current_junc_id to component_tgt1
                if component_tgt1:
                    if self._is_junction_node(component_tgt1):
                        if component_tgt1 in junction_definitions_map:
                            flow_graph.add_edge(current_junc_id, component_tgt1)
                    else: # component_tgt1 is a regular component
                        flow_graph.add_edge(current_junc_id, component_tgt1)

                # Edge from current_junc_id to component_tgt2
                if component_tgt2:
                    if self._is_junction_node(component_tgt2):
                        if component_tgt2 in junction_definitions_map:
                            flow_graph.add_edge(current_junc_id, component_tgt2)
                    else: # component_tgt2 is a regular component
                        flow_graph.add_edge(current_junc_id, component_tgt2)

            # --- Processing for a MERGE junction (source_1, source_2 -> current_junc_id -> target) ---
            elif "source_1" in current_junc_def:
                component_src1 = current_junc_def.get("source_1")
                component_src2 = current_junc_def.get("source_2")
                component_tgt = current_junc_def.get("target")

                # Edge from component_src1 to current_junc_id
                if component_src1:
                    if self._is_junction_node(component_src1):
                        if component_src1 in junction_definitions_map:
                            flow_graph.add_edge(component_src1, current_junc_id)
                    else: # component_src1 is a regular component
                        flow_graph.add_edge(component_src1, current_junc_id)

                # Edge from component_src2 to current_junc_id
                if component_src2:
                    if self._is_junction_node(component_src2):
                        if component_src2 in junction_definitions_map:
                            flow_graph.add_edge(component_src2, current_junc_id)
                    else: # component_src2 is a regular component
                        flow_graph.add_edge(component_src2, current_junc_id)
                
                # Edge from current_junc_id to component_tgt
                if component_tgt:
                    if self._is_junction_node(component_tgt):
                        if component_tgt in junction_definitions_map:
                            flow_graph.add_edge(current_junc_id, component_tgt)
                    else: # component_tgt is a regular component
                        flow_graph.add_edge(current_junc_id, component_tgt)
        # --- End of Junction Processing ---

        # --- Path Validation ---
        unimplemented_connections = set()
        spurious_connections = set()

        # 1. Check if all DEFINED connections have a valid path in the FINAL graph
        for source, target in defined_connections_set:
            connection_implemented = False
            # Check if the direct connection still exists (wasn't removed)
            if flow_graph.has_edge(source, target):
                 connection_implemented = True
            else:
                # If direct edge removed/absent, check for path via junctions
                if nx.has_path(flow_graph, source, target):
                    try:
                        for path in nx.all_simple_paths(flow_graph, source=source, target=target):
                            # Check if all intermediate nodes are junctions
                            if len(path) > 1: # Path has intermediate nodes
                                is_junction_path = True
                                for i in range(1, len(path) - 1):
                                    if not self._is_junction_node(path[i]):
                                        is_junction_path = False
                                        break
                                if is_junction_path:
                                    connection_implemented = True
                                    break # Found one valid path implementation
                        # If loop finishes without setting connection_implemented = True, it remains False.
                    except nx.NetworkXNoPath:
                        pass # Should not happen if has_path is true
                    except Exception as e: # Catch potential errors during pathfinding
                        print(f"Warning: Error finding paths for {source}->{target}: {e}")
                        pass

            if not connection_implemented:
                unimplemented_connections.add((source, target))


        # 2. Check for SPURIOUS paths (exist ONLY via junctions, but not DEFINED)
        component_nodes = all_connection_nodes - all_junction_nodes

        for source in component_nodes:
            for target in component_nodes:
                if source == target: continue

                is_defined = (source, target) in defined_connections_set
                if is_defined: continue # Only looking for paths that are NOT defined

                path_via_junction_found = False
                if nx.has_path(flow_graph, source, target):
                    try:
                        for path in nx.all_simple_paths(flow_graph, source=source, target=target):
                            if len(path) > 1: # Path has intermediate nodes
                                path_uses_junction = False
                                all_intermediate_are_junctions = True
                                for i in range(1, len(path) - 1):
                                    node = path[i]
                                    if self._is_junction_node(node):
                                        path_uses_junction = True
                                    else:
                                        # If any intermediate is NOT a junction, this path is invalid for this check
                                        all_intermediate_are_junctions = False
                                        break
                                # A spurious connection is one formed ONLY by junctions
                                if path_uses_junction and all_intermediate_are_junctions:
                                     path_via_junction_found = True
                                     break # Found one such path
                    except nx.NetworkXNoPath:
                         pass
                    except Exception as e:
                         print(f"Warning: Error finding paths for spurious check {source}->{target}: {e}")
                         pass # Skip pair if pathfinding fails

                # If a path ONLY via junctions was found, and it wasn't defined -> Spurious
                if path_via_junction_found:
                     spurious_connections.add((source, target))


        # --- Scoring ---
        validity_penalty = (len(unimplemented_connections) * self.PENALTY_JUNCTION_UNIMPLEMENTED +
                            len(spurious_connections) * self.PENALTY_JUNCTION_SPURIOUS)

        min_junctions_needed = self._calculate_minimum_junctions(defined_connections_set)
        num_provided_junctions = len(all_junction_nodes)
        minimality_penalty = max(0, num_provided_junctions - min_junctions_needed) * self.PENALTY_JUNCTION_EXTRA

        total_penalty = validity_penalty + minimality_penalty
        score = max(0.0, self.REWARD_JUNCTIONS - total_penalty)

        return score

    def _compare_component_params(self, reference_design, generated_design, mapping):
        """
        Compares parameters of components used in the generated connections,
        using the provided mapping to find corresponding reference components.
        """
        ref_params = reference_design.get("component_params", {})
        gen_params = generated_design.get("component_params", {})
        gen_connections = generated_design.get("connections", [])

        # 1. Identify components used in generated connections
        used_gen_components = set()
        for conn in gen_connections:
             src_id = conn.get("source")
             tgt_id = conn.get("target")
             src_type, _, _ = self._parse_component_id(src_id)
             tgt_type, _, _ = self._parse_component_id(tgt_id)
             # Only add components that have parameters (exclude inlet, outlet, potentially others)
             param_types = {"mixers", "delays", "chambers", "filters"} # Add others if they get params
             if src_type and src_type + "s" in param_types : used_gen_components.add(src_id)
             if tgt_type and tgt_type + "s" in param_types : used_gen_components.add(tgt_id)


        mismatched_params_count = 0

        # 2. Iterate through used generated components and compare params
        for gen_comp_id in used_gen_components:
            gen_type, gen_num, _ = self._parse_component_id(gen_comp_id)
            gen_type_plural = gen_type + "s" # e.g., "mixer" -> "mixers"

            # Find the parameters for this generated component
            gen_comp_data = None
            for comp in gen_params.get(gen_type_plural, []):
                if comp.get("id") == gen_comp_id:
                    gen_comp_data = {k: v for k, v in comp.items() if k != "id"}
                    break
            
            if gen_comp_data is None:
                 # Parameter data missing for a used component in generated design
                 mismatched_params_count += 1 # Penalize
                 continue


            # Find the corresponding reference component using the mapping
            ref_num = gen_num # Default if type not in mapping
            if gen_type in mapping:
                # Mapping is ref_num -> gen_num. We need the reverse.
                inverted_map = {v: k for k, v in mapping[gen_type].items()}
                ref_num = inverted_map.get(gen_num, gen_num) # Find ref_num corresponding to gen_num

            ref_comp_id = f"{gen_type}_{ref_num}" # Reconstruct potential ref ID

            # Find the parameters for the corresponding reference component
            ref_comp_data = None
            for comp in ref_params.get(gen_type_plural, []):
                if comp.get("id") == ref_comp_id:
                    ref_comp_data = {k: v for k, v in comp.items() if k != "id"}
                    break
            
            if ref_comp_data is None:
                 # Corresponding reference component params not found?
                 # This implies the component itself is effectively "extra" or mismatched
                 # The connection comparison should have penalized this already.
                 continue

            # Compare the actual parameters
            all_keys = set(gen_comp_data.keys()) | set(ref_comp_data.keys())
            for key in all_keys:
                gen_val = gen_comp_data.get(key)
                ref_val = ref_comp_data.get(key)

                if isinstance(gen_val, dict) and isinstance(ref_val, dict):
                     if gen_val != ref_val: # Simple dict comparison
                          mismatched_params_count += 1
                         
                elif gen_val != ref_val:
                    mismatched_params_count += 1
                    


        score = max(0.0, self.REWARD_PARAMS - mismatched_params_count * self.PENALTY_PARAM_MISMATCH)
        
        return score

    def _check_design_structure(self, output_str):
        """
        Evaluates the design structure.
        Applies penalties for invalid string patterns.
        """
        print("--------------------------------------")
        penalty_value = 0.125 # Keep penalty relative to section score
        max_section_score = self.REWARD_MAX_STRUCTURE / 3

        try:
            design = json.loads(output_str)
        except json.JSONDecodeError as e:
            print(f"JSON decoding error during structure check: {e}")
            return 0.0

        if not isinstance(design, dict):
            print("Structure check error: Input is not a JSON object.")
            return 0.0

        total_score = 0.0

        # Section 1: connections
        section_score = 0.0
        connections_struct_ok = False
        if "connections" in design and isinstance(design["connections"], list):
            connections_struct_ok = True
            for conn in design["connections"]:
                if not (isinstance(conn, dict) and "source" in conn and "target" in conn):
                    connections_struct_ok = False
                    break
        if connections_struct_ok:
            section_score = max_section_score # Start with full score for structure
            # Pattern checks with penalties
            source_pat = re.compile(r'^((inlet|mixer|delay|chamber|droplet)_[0-9]+|filter_[0-9]+_(smaller|larger)|junction_[0-9]+)$')
            target_pat = re.compile(r'^((outlet|mixer|delay|chamber|filter)_[0-9]+|droplet_[0-9]+_(continuous|dispersed)|junction_[0-9]+)$')
            source_pat_strict = re.compile(r'^((inlet|mixer|delay|chamber|droplet)_[0-9]+|filter_[0-9]+_(smaller|larger))$')
            target_pat_strict = re.compile(r'^((outlet|mixer|delay|chamber|filter)_[0-9]+|droplet_[0-9]+_(continuous|dispersed))$')

            for conn in design["connections"]:
                src = conn.get("source", "")
                tgt = conn.get("target", "")
                if not (isinstance(src, str) and source_pat_strict.match(src)):
                    section_score -= penalty_value
                if not (isinstance(tgt, str) and target_pat_strict.match(tgt)):
                    section_score -= penalty_value
        total_score += max(0.0, section_score) # Add score for this section, min 0

        # Section 2: junctions
        section_score = 0.0
        junctions_struct_ok = False
        if "junctions" in design and isinstance(design["junctions"], list):
            junctions_struct_ok = True
            # Define the exact key sets required for each junction type
            split_keys = {"id", "type", "source", "target_1", "target_2"}
            merge_keys = {"id", "type", "source_1", "source_2", "target"}

            for junc in design["junctions"]:
                # A junction must be a dictionary
                if not isinstance(junc, dict):
                    junctions_struct_ok = False
                    break

                # Get the set of keys from the current junction
                current_keys = set(junc.keys())

                # Check if the junction's keys are an exact match for EITHER a split or a merge.
                # This single condition is sufficient.
                if not (current_keys == split_keys or current_keys == merge_keys):
                    junctions_struct_ok = False
                    break

        if junctions_struct_ok:
            section_score = max_section_score # Start with full score for structure
            # Pattern checks with penalties
            id_pat = re.compile(r'^junction_[0-9]+$')
            type_allowed = {"T-junction", "Y-junction"}
            # Junctions can connect to/from components OR other junctions
            src_pat = re.compile(r'^((inlet|mixer|delay|chamber|droplet|junction)_[0-9]+|filter_[0-9]+_(smaller|larger))$')
            tgt_pat = re.compile(r'^((outlet|mixer|delay|chamber|filter|junction)_[0-9]+|droplet_[0-9]+_(continuous|dispersed))$')

            for junc in design["junctions"]:
                if not (isinstance(junc.get("id"), str) and id_pat.match(junc["id"])):
                    section_score -= penalty_value
                if junc.get("type") not in type_allowed:
                    section_score -= penalty_value

                if "source" in junc: # Split type
                    if not (isinstance(junc.get("source"), str) and src_pat.match(junc["source"])): section_score -= penalty_value
                    if not (isinstance(junc.get("target_1"), str) and tgt_pat.match(junc["target_1"])): section_score -= penalty_value
                    if not (isinstance(junc.get("target_2"), str) and tgt_pat.match(junc["target_2"])): section_score -= penalty_value
                elif "source_1" in junc: # Merge type
                    if not (isinstance(junc.get("source_1"), str) and src_pat.match(junc["source_1"])): section_score -= penalty_value
                    if not (isinstance(junc.get("source_2"), str) and src_pat.match(junc["source_2"])): section_score -= penalty_value
                    if not (isinstance(junc.get("target"), str) and tgt_pat.match(junc["target"])): section_score -= penalty_value
        total_score += max(0.0, section_score) # Add score for this section, min 0


        # Section 3: component_params
        section_score = 0.0
        comp_struct_ok = False
        required_sections = ["mixers", "delays", "chambers", "filters"] # Base required sections
        if "component_params" in design and isinstance(design["component_params"], dict):
            cp = design["component_params"]
            # Check if all *required* sections are present and are lists
            comp_struct_ok = all(key in cp and isinstance(cp[key], list) for key in required_sections)

            if comp_struct_ok:
                # Check structure within each list
                try:
                    for mixer in cp.get("mixers", []):
                        if not (isinstance(mixer, dict) and "id" in mixer and "num_turnings" in mixer): raise ValueError("Mixer structure")
                    for delay in cp.get("delays", []):
                         if not (isinstance(delay, dict) and "id" in delay and "num_turnings" in delay): raise ValueError("Delay structure")
                    for chamber in cp.get("chambers", []):
                         if not (isinstance(chamber, dict) and "id" in chamber and "dimensions" in chamber and
                                isinstance(chamber["dimensions"], dict) and
                                "length" in chamber["dimensions"] and "width" in chamber["dimensions"]): raise ValueError("Chamber structure")
                    for filt in cp.get("filters", []):
                         if not (isinstance(filt, dict) and "id" in filt and "critical_particle_diameter" in filt): raise ValueError("Filter structure")

                except ValueError as e:
                    comp_struct_ok = False

        if comp_struct_ok:
            section_score = max_section_score # Start with full score for structure
             # Pattern checks with penalties
            mixer_pat = re.compile(r'^mixer_[0-9]+$')
            delay_pat = re.compile(r'^delay_[0-9]+$')
            chamber_pat = re.compile(r'^chamber_[0-9]+$')
            filter_pat = re.compile(r'^filter_[0-9]+$')

            cp = design["component_params"]
            for mixer in cp.get("mixers", []):
                 if not (isinstance(mixer.get("id"), str) and mixer_pat.match(mixer["id"])): section_score -= penalty_value
            for delay in cp.get("delays", []):
                 if not (isinstance(delay.get("id"), str) and delay_pat.match(delay["id"])): section_score -= penalty_value
            for chamber in cp.get("chambers", []):
                 if not (isinstance(chamber.get("id"), str) and chamber_pat.match(chamber["id"])): section_score -= penalty_value
            for filt in cp.get("filters", []):
                 if not (isinstance(filt.get("id"), str) and filter_pat.match(filt["id"])): section_score -= penalty_value

        total_score += max(0.0, section_score) # Add score for this section, min 0

        print(f"Structure Score: {total_score} / {self.REWARD_MAX_STRUCTURE}")
        return total_score


    def evaluate_design(self, reference_json_str, generated_json_str):
        """
        Evaluates the generated microfluidic design against a reference.
        First checks structure, then compares connections, junctions, and params
        based on the new strategy.
        """
        # 1. Basic Parsing and Validation
        try:
            reference_design = json.loads(reference_json_str)
        except json.JSONDecodeError:
            print("Error: Reference JSON is invalid.")
            return 0.0

        try:
            generated_design = json.loads(generated_json_str)
        except json.JSONDecodeError:
            print("Error: Generated JSON is invalid.")
            return 0.0

        # 2. Check Structure of Generated Design
        structure_score = self._check_design_structure(generated_json_str)

        if not self._validate_json(generated_design, "Generated Design"):
             print("Generated design failed schema validation.")
             return structure_score
            
        # 3. Detailed Comparison (only if structure is perfect)
        print("Generated design passed structure check. Performing detailed comparison...")
        
        # Compare Connections (gets score and mapping)
        connection_score, best_mapping = self._compare_connections(reference_design, generated_design)

        # Compare Junctions (based *only* on generated design)
        junction_score = self._compare_junctions(generated_design)

        # Compare Parameters (using the mapping from connections)
        param_score = self._compare_component_params(reference_design, generated_design, best_mapping)

        # 4. Calculate Final Score
        # Total score is the sum of scores from the detailed comparison parts
        total_score = structure_score + connection_score + junction_score + param_score
        
        max_possible_score = self.REWARD_MAX_STRUCTURE + self.REWARD_CONNECTIONS + self.REWARD_JUNCTIONS + self.REWARD_PARAMS
        print(f"Detailed Scores: Connections={connection_score:.2f}, Junctions={junction_score:.2f}, Params={param_score:.2f}")
        print(f"Final Score: {total_score:.2f} / {max_possible_score:.2f}")

        return total_score