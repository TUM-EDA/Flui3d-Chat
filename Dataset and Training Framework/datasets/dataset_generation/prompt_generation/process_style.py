import random
import re
from typing import Dict, List, Optional, Tuple

import networkx as nx

from .prompt_generation import PromptGenerator

class ProcessOrientedPromptGenerator(PromptGenerator):
    """Generates prompts describing the processes occurring on the chip."""

    # Ordinal words for naming fluids/streams/filters ("the second filter"). Shared by
    # _number_prefix, _ordinal_index and the _canonicalize_reading_order relabel.
    ORDINAL_WORDS = ["first", "second", "third", "fourth", "fifth",
                     "sixth", "seventh", "eighth", "ninth", "tenth"]

    # Per-design counts of splits / DLD filters / droplet generators, (re)set at the top of
    # _generate_for_single_graph. When >=2 of a kind are in play a bare "the first stream" / "the
    # larger particle stream" / "the droplets" is ambiguous, so every such reference is anchored to
    # its split/filter/source fluid; see _split_possessive, _particle_stream_label and the droplet
    # branch of _handle_linear_subgraph.
    _n_splits = 0
    _n_dld = 0
    _n_droplets = 0

    # Action verb synonyms, chosen once per prompt for consistency. The "separate smaller
    # and larger particles of" family is reserved for DLD filters (two size-sorted outputs);
    # the single-output pillar-matrix filter has its own "filter out particles" verb so the
    # two filter types stay distinguishable in prose. "direct" is the tesla-valve verb.
    ACTION_SYNONYMS = {
        "let react": ["let react", "initiate a reaction", "allow to react"],
        "mix": ["mix", "blend", "homogenize"],
        "mixture": ["mixture", "blend", "homogenized solution"],
        "delay": ["delay", "hold", "retain"],
        "delayed": ["delayed", "held", "retained"],
        "combine": ["combine", "merge", "join", "unite"],
        "split": ["split", "divide", "separate", "distribute", "segment"],
        "separate smaller and larger particles of": ["separate smaller and larger particles of", "separate the smaller and larger particles of", "sort by size the particles of", "divide smaller and larger particles of", "fractionate by size the particles of", "size-separate the particles of"],
        "filter out particles": ["filter out particles", "remove particles", "sieve out particles", "strain out particles", "screen out particles", "separate out particles"],
        "direct": ["direct", "route", "guide", "channel", "pass"],
        "form droplets": ["form droplets", "generate droplets", "produce droplets", "form microdroplets", "generate microdroplets", "produce microdroplets"],
        "droplet formation": ["droplet formation", "droplet generation", "droplet production", "microdroplet formation", "microdroplet generation", "microdroplet production"],
        "smaller particle stream": ["smaller particle stream", "smaller particle flow", "flow with smaller particles", "stream with smaller particles"],
        "larger particle stream": ["larger particle stream", "larger particle flow", "flow with larger particles", "stream with larger particles"]
    }

    # Natural-language phrasing of each v2 component parameter, used to surface NON-default
    # (deliberately randomized) values inside a "(using ... with ...)" aside. Defaults are
    # suppressed, matching the structural styles. Keyed by the v2 attribute name; "{v}" is the
    # value. Iterated in this order, so a component's params read in a stable, logical sequence.
    # "{s}" expands to a plural "s" unless the value is 1 (so "1 turning", "4 turnings"); it is
    # ignored by phrasings that do not use it.
    PARAM_PHRASING = {
        "num_turnings": "{v} turning{s}",
        "amplitude_um": "{v} µm amplitude",
        "distance_between_turnings_um": "{v} µm between turnings",
        "diameter_um": "{v} µm diameter",
        "num_circles": "{v} circle{s}",
        "distance_between_circles_um": "{v} µm between circles",
        "length_um": "{v} µm length",
        "width_um": "{v} µm width",
        "post_shape": "{v}-shaped posts",
        "post_diameter_um": "{v} µm post diameter",
        "columns": "{v} column{s}",
        "rows": "{v} row{s}",
        "row_shift_fraction": "a row shift fraction of {v}",
        "critical_particle_diameter_um": "{v} µm critical particle diameter",
        "num_segment_pairs": "{v} segment pair{s}",
        "segment_length_um": "{v} µm segment length",
        "segment_width_um": "{v} µm segment width",
        "nozzle_width_um": "{v} µm nozzle width",
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
        if processed_graph is None:
            # Merging adjacent same-function junctions collapsed two routes onto one edge
            # (rare; e.g. a source feeding the same merge group twice). Fall back to treating
            # every junction as its own conceptual unit, which never collapses an edge.
            processed_graph = self._replace_combining_and_splitting_units(graph, merge_adjacent=False)

        # How many distinct splits / DLD filters / droplet generators the narration will describe.
        # With >=2 of a kind, "the first stream" / "the larger particle stream" / "the droplets" could
        # belong to more than one of them, so every such reference is anchored to its split/filter/source
        # fluid (_split_possessive, _particle_stream_label, droplet branch of _handle_linear_subgraph);
        # with <=1 there is no ambiguity and references stay bare and natural.
        self._n_splits = sum(1 for n in processed_graph.nodes if n.startswith("splitting"))
        self._n_dld = sum(1 for n in processed_graph.nodes
                          if n.startswith("filter") and processed_graph.nodes[n].get("type") == "dld")
        self._n_droplets = sum(1 for n in processed_graph.nodes if n.startswith("droplet"))

        # Convert the processed graph into a single node that contains the full textual description of the process.
        output_node = self._graph_to_text(processed_graph, detailed=detailed)
        output_data = list(output_node.nodes(data=True))[0][1]
        action_description = output_data.get("action", "") + (output_data.get("outlet_token") or "")

        # Record each node's narration position as a `reading_index` attribute BEFORE the renumber
        # passes strip the markers, so the JSON converter can list connections in prompt order. The
        # attribute rides through the id relabels below (relabel_nodes preserves node data).
        self._assign_reading_index(graph, action_description)

        # Renumber component IDs into the order the narration introduces them, so the prose reads in
        # sequence and every reference points at exactly that node in the JSON built from this graph.
        # All passes mutate `graph` in place (its IDs are the JSON's). Outlets run first because that
        # pass also strips the internal @@OUTLET@@ markers; the inlet/filter pass then works on clean
        # text (those carry a spoken ordinal to rewrite too); the pass-through pass renumbers the
        # remaining components (mixer/delay/chamber/droplet/tesla_valve), which carry no spoken
        # ordinal, so their JSON ids alone ascend in reading order -- e.g. the second tesla valve
        # narrated becomes tesla_valve_2, not tesla_valve_4.
        action_description = self._renumber_outlets_by_reading_order(graph, action_description)
        action_description = self._canonicalize_reading_order(graph, action_description)
        action_description = self._renumber_passthrough_by_reading_order(graph, action_description)

        # Refine and augment description
        action_description = self._modify_combine_mix_pairs(action_description)
        action_description = self._modify_combine_react_pairs(action_description)
        # Action verbs are synonymized BEFORE module nouns: the verb "delay" and the noun
        # "delay channel" share the word "delay", so resolving nouns first would let the verb
        # pass corrupt the noun ("retain channel"). While the noun is still a "<cat>_synonym"
        # token its trailing "_" blocks the verb regex's word boundary, so verbs are replaced
        # but noun tokens are not; the nouns are then resolved and never seen by the verb pass.
        action_description = self._replace_with_random_action_synonyms(action_description)
        action_description = self._replace_with_random_synonym(action_description, self.MODULE_SYNONYMS)
        action_description = self._dedupe_filter_phrasing(action_description)
        # Resolve the split anchors LAST: each "@@SPLITPOSS@@" becomes "the Nth split's" with N in
        # the order the splits were introduced. Done after the synonym passes so the literal noun
        # "split" is never mistaken for the "split" action verb and synonymised away.
        action_description = self._resolve_split_anchors(action_description)
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
    def _replace_combining_and_splitting_units(self, graph: nx.DiGraph, merge_adjacent: bool = True) -> Optional[nx.DiGraph]:
        """
        Pre-processes the graph by collapsing combining or splitting junctions into single
        conceptual "units". A merge of N inputs / split into N outputs becomes one operation
        in the process narration, regardless of how many physical T/Y-junctions implement it.

        With ``merge_adjacent`` (default) a maximal group of touching same-function junctions
        becomes one unit ("combine 4 fluids"). With it disabled each junction becomes its own
        unit; this never collapses two routes onto one edge, so it is the safe fallback when
        merging would (the caller retries with it off).

        Each unit records ``junction_specs``: the ``(width, T/Y-shape)`` of every junction it
        absorbed, so the hardware aside can still name them (e.g. "a 3-way fork").

        Args:
            graph: The input microfluidic chip graph.
            merge_adjacent: Whether to fuse touching same-function junctions into one unit.

        Returns:
            A new graph with junctions replaced by unit nodes, or None if merging collapsed
            the edge count (only possible when ``merge_adjacent`` is True).
        """
        graph = graph.copy()
        edge_count = len(graph.edges())

        def replace_node_group(nodes_to_replace, new_node_name):
            # Helper function to replace a group of nodes with a single new node,
            # consolidating their connections and attributes.
            nonlocal edge_count
            # Capture each junction's width and T/Y shape BEFORE any rewiring: adding edges to
            # the new unit would otherwise inflate the in/out-degree of group members.
            specs = [(max(graph.in_degree(n), graph.out_degree(n)), graph.nodes[n]["type"])
                     for n in sorted(nodes_to_replace)]
            graph.add_node(new_node_name, junction_specs=specs)

            for node in sorted(nodes_to_replace):
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
                    if merge_adjacent:
                        # Find all connected junctions of the same function.
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
                    else:
                        # Fallback: each junction is its own unit (no fusion, no collapse).
                        group = {node_name}

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

        # Identify all "branching" nodes: combine/split units and DLD filters (which fan a
        # stream into two size-sorted outputs). In v2 droplet generators and pillar-matrix
        # filters are single-input/single-output pass-throughs, so they are NOT branchers --
        # they are narrated as ordinary steps in the linear handler instead.
        target_nodes = sorted(
            [n for n in graph.nodes
             if n.startswith(('splitting', 'combining'))
             or (n.startswith('filter') and graph.nodes[n].get('type') == 'dld')],
            key=str.lower)
        
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
        # Outlets are terminal nodes, so they are removed to find the next step. Capturing them
        # first lets this chain carry an outlet marker (stripped again by
        # _renumber_outlets_by_reading_order): the marker rides the summary node into the assembled
        # text, so every outlet ends up tokenised in true narration order no matter which recursive
        # branch removed it.
        terminal_outlets = [n for n in graph.nodes if n.startswith('outlet')]
        graph.remove_nodes_from(terminal_outlets)
        
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
                "action": "", "output_type": f"{prefix}fluid@@RO:{node_name}@@", "number_nodes": 0
            })

        # Tag this chain with the outlet it terminates at. Only the first (outermost) call sees the
        # outlet; deeper recursive calls inherit the token via succ_attrs further down.
        if terminal_outlets:
            node_attributes["outlet_token"] = "".join(f"@@OUTLET:{o}@@" for o in terminal_outlets)

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
        # The originating special stream/droplet this chain stays attributed to (e.g. a DLD
        # "smaller particle stream" or a split "first stream"), or None for an ordinary fluid.
        prev_origin = node_attributes.get("stream_origin")

        # Verb, resulting fluid, and object-attachment mode for each linear (single-output)
        # component. The component is implied by its verb -- "mix" -> mixer, "form droplets" ->
        # droplet generator, "filter out particles" -> pillar-matrix filter -- except the tesla
        # valve, which has no implying verb and so is named in the clause itself. A product of
        # None marks a flow element (the tesla valve) that does not transform the fluid, so the
        # running output_type is carried through. Non-default params / a non-default type ride a
        # "(using ...)" aside built by _module_detail. The mode selects the lead-clause grammar:
        #   None    -> transitive: "Mix <fluid>"            / chain "..., then mix"
        #   "from"  -> "Form droplets from <fluid>"          / chain "..., then form droplets"
        #   "react" -> chamber grammar: "Let <fluid> react"  / chain "..., then let react"
        #   "tesla" -> tesla valve, named in the clause (handled specially)
        action_details = {
            "chamber":     ("let react", "reaction product", "react"),
            "mixer":       ("mix", "mixture", None),
            "delay":       ("delay", "delayed liquid", None),
            "droplet":     ("form droplets", "droplets", "from"),
            "filter":      ("filter out particles", "filtered fluid", "from"),
            "tesla_valve": ("direct", None, "tesla"),
        }

        # Generate the next part of the sentence based on the successor's type.
        for comp_type, (verb, product, mode) in action_details.items():
            if not successor.startswith(comp_type):
                continue

            # Build the leading clause (sentence start) and the chained clause (mid-sentence).
            if mode == "tesla":
                params = self._param_phrases(successor, succ_attrs)
                aside = f" (with {self._join_and(params)})" if params else ""
                lead = f"Direct {prev_output_type} through a tesla_valve_synonym{aside}"
                chain = f"direct through a tesla_valve_synonym{aside}"
            else:
                detail = self._module_detail(successor, succ_attrs, detailed)
                if mode == "react":
                    lead = f"Let {prev_output_type} react{detail}"
                elif mode == "from":
                    lead = f"{verb.capitalize()} from {prev_output_type}{detail}"
                else:
                    lead = f"{verb.capitalize()} {prev_output_type}{detail}"
                chain = f"{verb}{detail}"

            # Sentence start vs. continuation of an existing clause.
            if counter == 0:
                succ_attrs["action"] = lead if not prev_action else f"{prev_action} Then, {lead[0].lower() + lead[1:]}"
            else:
                succ_attrs["action"] = f"{prev_action}, then {chain}"

            # Mark this component's first mention so its JSON id can be renumbered into reading
            # order. Inlets and (pillar/DLD) filters also carry a spoken ordinal that
            # _canonicalize_reading_order rewrites; the pass-through components (mixer, delay,
            # chamber, droplet, tesla_valve) have no spoken ordinal, so the marker only reorders
            # their ids via _renumber_passthrough_by_reading_order.
            succ_attrs["action"] += f"@@RO:{successor}@@"

            # Update what the fluid has become and which originating special stream/droplet it
            # stays attributed to ("the reaction product of the smaller particle stream"). A flow
            # element (product None) leaves the fluid -- and its attribution -- unchanged. The
            # attribution is composed onto the fresh product each step rather than appended to the
            # previous label, so it never nests/explodes across a long chain.
            if product is None:
                succ_attrs["output_type"] = prev_output_type
                succ_attrs["stream_origin"] = prev_origin
            elif comp_type == "droplet":
                # Forming droplets makes "droplets" the new origin. A droplet set carried from a split
                # or DLD stream is already anchored to it ("the droplets of the smaller particle
                # stream"), and never to itself ("droplets of the droplets"). Otherwise the set is a
                # bare "the droplets" -- natural and kept when it is the ONLY droplet set in the design.
                # With >=2 droplet generators a bare "the droplets" is ambiguous (which set?), so anchor
                # this one to the fluid it is formed from ("the droplets of the third fluid" / "of the
                # mixture"), mirroring the split/DLD stream disambiguation. The source ordinal rides
                # through and is renumbered to reading order by _canonicalize_reading_order; the
                # first-mention marker is stripped here (it is preserved in the action) so none doubles.
                if prev_origin and not prev_origin.startswith("droplets"):
                    origin = prev_origin
                elif self._n_droplets >= 2 and not prev_output_type.startswith("the droplets"):
                    origin = re.sub(r"@@[A-Z]+:[^@]+@@", "", self._definite(prev_output_type))
                    origin = origin[len("the "):] if origin.startswith("the ") else origin
                else:
                    origin = None
                succ_attrs["output_type"] = f"the droplets of the {origin}" if origin else "the droplets"
                # Propagate the anchor into the origin tag too, but only with >=2 droplet sets, so a
                # downstream product stays attributable ("the reaction product of the droplets of the
                # third fluid") instead of a bare "the reaction product of the droplets" that collides
                # with another droplet set; a lone droplet set keeps the bare, natural origin.
                succ_attrs["stream_origin"] = (
                    f"droplets of the {origin}" if origin and self._n_droplets >= 2 else "droplets")
            else:
                succ_attrs["output_type"] = f"the {product} of the {prev_origin}" if prev_origin else f"the {product}"
                succ_attrs["stream_origin"] = prev_origin
            break

        # Update attributes for the next recursive step. The terminal-outlet marker rides forward
        # onto the summary node so it survives to the caller (and into the assembled text).
        succ_attrs["outlet_token"] = node_attributes.get("outlet_token")
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
        
        # Name the physical junction(s) only when that adds information the operation does not
        # already imply -- a non-default (Y) shape or a multi-junction fan; see _junction_aside.
        module_details = self._junction_aside(node_attributes["junction_specs"])

        # Initialize the attributes for the new combined node.
        node_attributes.update({
            "action": "",
            "output_type": "the combination",
            "number_nodes": 1,
        })

        # These lists and counters will help build the final description from the different incoming branches.
        further_fluids = []
        interim_actions = []
        interim_outputs = []  # product label of each interim branch (e.g. "the delayed liquid"), named in the merge
        inlet_marks = []     # @@RO@@ markers for direct inlets (rendered anonymously as "other fluids").
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
                # This branch represents a product of a previous process (e.g., a mixture). Name it in
                # the merge by its actual product label rather than an opaque "the solution", so the
                # reader can tell which earlier-described stream it is. Strip the internal markers and
                # make it definite (a pass-through-only branch keeps its inlet label, e.g. "a second
                # fluid" -> "the second fluid").
                if action: interim_actions.append(action)
                interim_outputs.append(self._definite(re.sub(r"@@[A-Z]+:[^@]+@@", "", output_type)))
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
                # This branch is a direct fluid inlet, rendered anonymously as "N other fluids".
                # Tag it so reading-order numbering still places it by where it is narrated.
                number_inlets += 1
                if node_name.startswith("inlet"):
                    inlet_marks.append(f"@@RO:{node_name}@@")

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
                action_clause = f' Then, combine {self._join_solution_labels(interim_outputs)}'
        else:
            if counter == 0:
                inlet_text = f'{number_inlets} fluid{"s" if number_inlets > 1 else ""}'
                action_clause = f' Then, combine {inlet_text}' if further_fluids else f'Combine {inlet_text}'
            else:
                inlet_text = f'{number_inlets} other fluid{"s" if number_inlets > 1 else ""}'
                action_clause = f' Then, combine {self._join_solution_labels(interim_outputs)} with {inlet_text}'

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
                # Mark the direct inlets as "other fluids" (they are combined WITH the further fluids).
                # Target only the inlet COUNT ("3 fluids" -> "3 other fluids"); a blanket replace also
                # corrupts fluid LABELS among the further fluids -- "the filtered fluid of the first
                # stream" -> "...filtered other fluid...", "the droplets of the second fluid" ->
                # "...second other fluid".
                action_clause = re.sub(r"(\d+) fluid", r"\1 other fluid", action_clause)

        # Finalize the action clause with hardware details, the direct-inlet markers, and a period.
        action_clause += f"{module_details}{''.join(inlet_marks)}."

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
        module_details = self._junction_aside(graph.nodes[furthest_node]["junction_specs"])
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

        respond_graphs, stream_counter = [], 0
        # Process each downstream branch (component) separately. A stream carried in from an earlier
        # split already wears that split's anchor (set when it was created below), so it needs no
        # extra disambiguation here.
        for i, component in enumerate(weakly_connected_components):
            subgraph = graph.subgraph(component).copy()

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
            
            # Insert a placeholder node (e.g., 'splitstream_0_0_...') to represent the start of this
            # specific output branch, named "the first stream" -- anchored to its split ("the second
            # split's first stream") when the design has >=2 splits, otherwise left bare.
            for j, out_node in enumerate(outgoing_nodes):
                new_node_name = f"splitstream_{i}_{j}_{furthest_node}"
                stream_label = f"the {self._split_possessive(furthest_node)}{self._number_prefix(stream_counter)} stream"
                subgraph.add_node(new_node_name, action="", output_type=stream_label,
                                  number_nodes=0, stream_origin=stream_label[len("the "):])
                subgraph.add_edge(new_node_name, out_node, **outgoing_edge_data[out_node])
                stream_counter += 1

            # Recursively call the main function to generate the text for this entire branch.
            respond_graphs.append(self._graph_to_text(subgraph, detailed))
        
        # Retrieve the description of the fluid/process that is being split.
        incoming_props = list(incoming_subgraph.nodes(data=True))[0][1]
        
        # Start building the final combined action description. "@@SPLITDEF@@" records where this
        # split is introduced so _resolve_split_anchors can number the splits in the order they
        # appear; it is stripped from the final text.
        splitdef = f"@@SPLITDEF:{int(furthest_node.split('_')[2])}@@" if self._n_splits >= 2 else ""
        action_parts = []
        if incoming_props.get("number_nodes", 0) == 0:
             action_parts.append(f"Split {incoming_props['output_type']} into {len(outgoing_edge_data)} streams{module_details}{splitdef}.")
        else:
            action_parts.append(f"{incoming_props['action']} Then, split {self._definite(incoming_props['output_type'])} into {len(outgoing_edge_data)} streams{module_details}{splitdef}.")

        number_nodes = incoming_props.get("number_nodes", 0) + 1

        # Collect the results from all the processed downstream branches.
        outlets = []
        outlet_tokens = []
        res_action_parts = []
        for res_graph in respond_graphs:
            res_node = list(res_graph.nodes)[0]
            res_props = res_graph.nodes[res_node]
            token = res_props.get("outlet_token") or ""
            if res_node.startswith("splitstream"):
                # If a stream just goes to an outlet, describe it simply.
                outlets.append(f'Route {res_props["output_type"]} to outlet_synonym{token}.')
                outlet_tokens.append(token)
            else:
                # Otherwise, append the full description of that branch's process.
                res_action_parts.append(res_props["action"] + token)

            number_nodes += res_props["number_nodes"]

        # Assemble the final, complete action string.
        final_action = " ".join(action_parts)

        # Add a summary of outlet routing if multiple streams go to outlets. The per-outlet markers
        # are appended even when the prose collapses to a count, so no outlet drops out of the text.
        if len(outlets) > 1:
            final_action += f' Route {len(outlets)} streams to separate outlet_synonyms.{"".join(outlet_tokens)} '
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
        # DLD-only handler (pillar-matrix filters are single-output and narrated linearly). The
        # aside surfaces the DLD type qualifier and any non-default geometry/critical-diameter.
        module_details = self._module_detail(furthest_node, filter_attrs, detailed)

        outgoing_edge_data = {neighbor: data for _, neighbor, data in graph.edges(furthest_node, data=True)}
        
        predecessors = list(nx.ancestors(graph, furthest_node))
        incoming_subgraph = self._handle_linear_subgraph(graph.subgraph(predecessors).copy(), detailed)
        
        graph.remove_nodes_from(predecessors + [furthest_node])
        
        respond_graphs = []

        # Sort downstream components for deterministic processing.
        weakly_connected_components = sorted(
            nx.weakly_connected_components(graph),
            key=lambda c: (len(self._filter_nodes(c)), list(nx.topological_sort(graph.subgraph(self._filter_nodes(c))))[0].lower() if self._filter_nodes(c) else "")
        )

        for i, component in enumerate(weakly_connected_components):
            subgraph = graph.subgraph(component).copy()

            # A particle stream carried in from an earlier filter already wears that filter's anchor
            # (set when it was created below), so it needs no extra disambiguation here.
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
            
            # Insert placeholder nodes for the "smaller" and "larger" particle streams -- anchored to
            # their filter ("the second filter's larger particle stream") when the design has >=2 DLD
            # filters, otherwise left bare. The filter ordinal is the JSON filter id; like any other
            # "the Nth filter" reference it is renumbered into reading order by _canonicalize_reading_order.
            for j, out_node in enumerate(outgoing_nodes):
                particle_type = outgoing_edge_data[out_node]["filter_connection_type"]
                new_node_name = f"separatedstream_{i}_{j}_{furthest_node}"
                stream_label = self._particle_stream_label(furthest_node, particle_type)
                subgraph.add_node(new_node_name, action="", output_type=stream_label,
                                  number_nodes=0, stream_origin=stream_label[len("the "):])
                subgraph.add_edge(new_node_name, out_node, **outgoing_edge_data[out_node])

            # Recursively generate text for each downstream branch.
            respond_graphs.append(self._graph_to_text(subgraph, detailed))
        
        incoming_props = list(incoming_subgraph.nodes(data=True))[0][1]
        
        # Assemble the final description.
        action_parts = []
        if incoming_props.get("number_nodes", 0) == 0:
            action_parts.append(f"Separate smaller and larger particles of {incoming_props['output_type']}{module_details}@@RO:{furthest_node}@@.")
        else:
            action_parts.append(f"{incoming_props['action']} Separate smaller and larger particles of {self._definite(incoming_props['output_type'])}{module_details}@@RO:{furthest_node}@@.")
        
        number_nodes = incoming_props.get("number_nodes", 0) + 1

        for res_graph in respond_graphs:
            res_props = list(res_graph.nodes(data=True))[0][1]
            action_parts.append(res_props["action"] + (res_props.get("outlet_token") or ""))
            number_nodes += res_props["number_nodes"]
        
        # Return a new single-node graph representing the fully described filtering process.
        combined_graph = nx.DiGraph()
        combined_graph.add_node("combined_node", action=" ".join(action_parts), output_type="solution", number_nodes=number_nodes)
        return combined_graph

    def _junction_aside(self, specs: List[Tuple[int, str]]) -> str:
        """A "(using ...)" hardware aside naming the physical junction(s) behind a combine/split
        operation, or "" to leave them unstated.

        The operation clause already conveys the function (combine/split) and the width (via the
        fluid/stream count), so a lone junction of the default T shape adds nothing and is left
        unnamed. An aside is emitted only when it carries non-implied structure:
          * a non-default (Y) shape -- otherwise the JSON `type` could not be recovered; or
          * a fan built from several chained junctions -- e.g. a 4-way split realized as
            "a binary T-junction and a 3-way fork", a decomposition the bare "split into 4
            streams" would hide.
        Thus the aside is present iff the junctions are anything other than a single T-junction.
        """
        informative = len(specs) > 1 or any(shape != self.DEFAULT_JUNCTION_SHAPE for _, shape in specs)
        return f" (using {self._describe_junctions(specs)})" if informative else ""

    def _describe_junctions(self, specs: List[Tuple[int, str]]) -> str:
        """Summarizes the physical junctions absorbed by a combine/split unit as a hardware
        aside, e.g. "a 3-way fork and 2 binary T-junctions". Width and T/Y shape come from the
        base ``_describe_junction`` helper; identical descriptions are collapsed with a count.

        Args:
            specs: ``(width, shape)`` pairs, one per absorbed junction (from ``junction_specs``).
        """
        descs = [self._describe_junction(width, shape) for width, shape in specs]
        parts, seen = [], []
        for desc in descs:
            if desc in seen:
                continue
            seen.append(desc)
            count = descs.count(desc)
            if count == 1:
                parts.append(desc)
            else:
                # "a binary Y-junction" -> "2 binary Y-junctions"
                noun = desc.split(" ", 1)[1] if desc.startswith(("a ", "an ")) else desc
                parts.append(f"{count} {noun}s")
        return self._join_and(parts)

    @staticmethod
    def _join_and(items: List[str]) -> str:
        """Joins items into an English list: ``[a, b, c]`` -> ``"a, b and c"``."""
        items = list(items)
        if len(items) <= 1:
            return items[0] if items else ""
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f" and {items[-1]}"

    def _join_solution_labels(self, labels: List[str]) -> str:
        """Join the interim-product labels of a merge (e.g. "the delayed liquid", "the mixture").

        Identical labels are collapsed with a count ("the mixture" twice -> "2 mixtures") so a merge
        never reads "the mixture and the mixture"; labels carrying a distinguishing origin (e.g. "the
        reaction product of the first stream") are unique and listed as-is.
        """
        parts, seen = [], []
        for label in labels:
            if label in seen:
                continue
            seen.append(label)
            count = labels.count(label)
            if count == 1:
                parts.append(label)
            else:
                noun = label[len("the "):] if label.startswith("the ") else label
                parts.append(f"{count} {noun}s")
        return self._join_and(parts)

    @staticmethod
    def _definite(output_type: str) -> str:
        """Render a fluid label definite for RE-mention as a later operation's object.

        The carried-through inlet fluid is introduced indefinitely ("a fluid", "a second
        fluid"), and a pure flow element (a tesla valve, ``product=None``) leaves that label
        unchanged. When a branch handler then re-mentions it -- "Direct a fluid through a
        tesla valve. Then split <X> into 2 streams." -- a bare "a fluid" reads as a fresh,
        unrelated fluid and hides that the split acts on the SAME stream. Turning it definite
        ("the fluid" / "the second fluid") makes the back-reference explicit while preserving
        any ordinal disambiguator. Labels that are already definite ("the mixture", "the
        smaller particle stream", "the droplets") are returned unchanged.
        """
        return f"the {output_type[2:]}" if output_type.startswith("a ") else output_type

    def _type_qualifier(self, category: str, ctype: Optional[str]) -> str:
        """Picks a prose qualifier for a component's subtype, or "" to leave it unstated.

        Non-default subtypes (ring mixer, flow-focusing droplet) are always stated -- the JSON
        ``type`` could not otherwise be recovered from the verb. Default subtypes (serpentine
        mixer, t-junction droplet) are usually left implicit. Filter type is optional because
        the verb already distinguishes DLD ("separate smaller and larger") from pillar-matrix
        ("filter out particles").
        """
        qualifiers = self.TYPE_QUALIFIERS.get((category, ctype))
        if not qualifiers:
            return ""
        if category in self.DEFAULT_TYPE:
            if ctype == self.DEFAULT_TYPE[category] and random.random() >= 0.18:
                return ""
        elif random.random() >= 0.5:  # filter: type optional (conveyed by the verb)
            return ""
        return random.choice(qualifiers)

    def _param_phrases(self, node: str, data: Dict) -> List[str]:
        """Natural-language phrasings of a component's NON-default parameters (e.g.
        "8 turnings", "5000 µm amplitude"). Defaults are suppressed so only deliberately
        randomized values surface, matching the structural styles. Order follows PARAM_PHRASING.
        """
        category = self._node_type(node)
        defaults = self.DEFAULT_ATTRIBUTES.get((category, data.get("type")),
                                               self.DEFAULT_ATTRIBUTES.get((category, None), {}))
        return [self.PARAM_PHRASING[k].format(v=data[k], s="" if data[k] == 1 else "s")
                for k in self.PARAM_PHRASING
                if k in data and k in defaults and data[k] != defaults[k]]

    def _module_detail(self, node: str, data: Dict, detailed: bool) -> str:
        """Optional "(using a <typed component> with <params>)" aside for a linear/DLD component.

        Emitted when the component carries a stated type qualifier (e.g. a ring mixer, whose
        type must be conveyed) or non-default parameters, and occasionally in detailed mode.
        The component noun rides a "<category>_synonym" token so every mention of a category
        resolves to one synonym per prompt; the type qualifier (when stated) is composed in
        front (e.g. "ring mixer_synonym"). Returns "" when there is nothing to add.
        """
        category = self._node_type(node)
        qualifier = self._type_qualifier(category, data.get("type"))
        params = self._param_phrases(node, data)
        noun = f"{qualifier} {category}_synonym".strip()
        if params:
            return f" (using {noun} with {self._join_and(params)})"
        if qualifier or (detailed and random.random() < 0.5):
            return f" (using {noun})"
        return ""

    def _number_prefix(self, n: int) -> str:
        """Converts a 0-based integer into an ordinal word (e.g., 1 -> "second") for natural language."""
        return self.ORDINAL_WORDS[n] if n < len(self.ORDINAL_WORDS) else f"{n+1}th"

    def _ordinal_index(self, word: Optional[str]) -> int:
        """Inverse of _number_prefix: an ordinal word (or None, meaning "first") -> its 0-based index."""
        if not word:
            return 0
        return self.ORDINAL_WORDS.index(word) if word in self.ORDINAL_WORDS else int(word[:-2]) - 1

    # Prose-ordinal patterns used to REWRITE the spoken ordinal once nodes are renumbered (the
    # narration ORDER itself comes from @@RO@@ first-mention markers, see _canonicalize_reading_order).
    # `ord` captures the ordinal word, or None for an implicit "first". "fluid" is never synonymized;
    # a DLD filter is referenced by ordinal only in the possessive "the Nth filter's <...> particle
    # stream", whose literal "filter's" marks it as a filter ref (never the "filter_synonym" noun).
    # The inlet pattern also captures the article (`art`): a fluid is introduced indefinitely
    # ("a fluid") but RE-mentioned definitely ("the fluid", see _definite), and both forms must
    # still be renumbered to reading order -- so the article is matched and preserved, not assumed.
    _READING_ORDER_PATTERNS = {
        "inlet": r"\b(?P<art>a|the) (?:(?P<ord>{ords}) )?fluid\b",
        "filter": r"\bthe (?P<ord>{ords}) filter's\b",
    }

    def _canonicalize_reading_order(self, graph: nx.DiGraph, action_description: str) -> str:
        """Relabel inlet and filter node IDs so they ascend in the order the narration first mentions
        them, and rewrite any spoken ordinals in ``action_description`` to agree.

        Order comes from "@@RO:<id>@@" first-mention markers the narration emits for every inlet and
        filter -- including ones that carry NO ordinal in the prose (a pillar filter narrated only as
        "strain out particles", or an inlet folded into a merge as "1 other fluid"). Reading prose
        ordinals alone would miss those and dump them at the end of the numbering, so e.g. a pillar
        filter narrated first could end up ``filter_3``. With markers each node is numbered by where
        it is actually narrated, and the spoken ordinals ("a second fluid", "the second filter's")
        are rewritten to match -- so the prose reads in sequence *and* every reference points at
        exactly ``inlet_N``/``filter_N`` in the JSON built from this graph. Each style converts its
        own graphs, so only this style's designs are affected.

        ``graph`` is mutated in place because its node IDs *are* the target JSON's IDs -- the caller
        must convert it to JSON only after generating the prompt.
        """
        ords = "|".join(self.ORDINAL_WORDS) + r"|\d+th"
        relabel = {}
        for prefix, template in self._READING_ORDER_PATTERNS.items():
            # Narration order from the first-mention markers (captures un-ordinalled mentions too).
            ordered = []
            for found in re.findall(rf"@@RO:{prefix}_(\d+)@@", action_description):
                num = int(found)
                if num not in ordered:
                    ordered.append(num)

            present = sorted(int(n.split("_")[1]) for n in graph.nodes if n.split("_")[0] == prefix)
            if not present:
                continue
            # Narrated nodes first (in reading order), then any a marker never reached (defensive).
            ordered = ordered + [i for i in present if i not in ordered]
            old_to_new = {old: new for new, old in enumerate(ordered, start=1)}
            relabel.update({f"{prefix}_{old}": f"{prefix}_{new}" for old, new in old_to_new.items()})

            # Rewrite the ordinal where the node IS named in prose; un-named mentions (merge inlets,
            # pillar filters) just follow their relabeled node id.
            pattern = re.compile(template.format(ords=ords))

            def _renumber(match, old_to_new=old_to_new, prefix=prefix):
                new = old_to_new[self._ordinal_index(match.group("ord")) + 1]
                # inlet_1 carries no ordinal ("a/the fluid"); filter ordinals are always spelled out.
                word = "" if prefix == "inlet" and new == 1 else self._number_prefix(new - 1) + " "
                # Preserve the inlet's article so a definite re-mention stays "the ... fluid".
                return f"{match.group('art')} {word}fluid" if prefix == "inlet" else f"the {word}filter's"

            action_description = pattern.sub(_renumber, action_description)

        action_description = re.sub(r"@@RO:(?:inlet|filter)_\d+@@", "", action_description)
        if relabel:
            # Two-phase via temporary names so an in-place permutation never collides (swapping
            # inlet_1 <-> inlet_2 directly with copy=False would clobber one of them).
            stash = {old: f"__reorder__{i}" for i, old in enumerate(relabel)}
            nx.relabel_nodes(graph, stash, copy=False)
            nx.relabel_nodes(graph, {stash[old]: new for old, new in relabel.items()}, copy=False)
        return action_description

    def _renumber_outlets_by_reading_order(self, graph: nx.DiGraph, action_description: str) -> str:
        """Renumber outlet node IDs so they ascend in the order the narration routes flow to them,
        then strip the internal ``@@OUTLET:..@@`` markers from the text.

        Outlets are never named by ordinal in the prose (always "an outlet"/"the output"), but each
        dataset example is a (prompt, JSON) pair, so the *order* outlets receive flow should still be
        canonical: the first outlet the process feeds is ``outlet_1``, the next ``outlet_2``, and so
        on. ``_handle_linear_subgraph`` tags every terminal chain with an ``@@OUTLET:<id>@@`` marker
        that rides the summary node into the assembled text, so reading the markers left-to-right
        recovers the true narration order even across nested branches.

        ``graph`` is mutated in place because its node IDs *are* the target JSON's IDs -- the caller
        converts it to JSON only after generating the prompt, so the relabel reaches the JSON.
        Idempotent: once the IDs are already in reading order the derived map is the identity.
        """
        order = []
        for found in re.findall(r"@@OUTLET:outlet_(\d+)@@", action_description):
            num = int(found)
            if num not in order:
                order.append(num)
        present = sorted(int(n.split("_")[1]) for n in graph.nodes if n.split("_")[0] == "outlet")
        # Narrated outlets first (in reading order), then any an @@OUTLET@@ marker never reached
        # (defensive -- every outlet should be tagged), keeping their existing order.
        ordered = order + [i for i in present if i not in order]
        relabel = {f"outlet_{old}": f"outlet_{new}"
                   for new, old in enumerate(ordered, start=1) if old != new}
        if relabel:
            # Two-phase via temporary names so the in-place permutation never clobbers a node
            # (renumbering outlet_1 <-> outlet_2 directly with copy=False would lose one).
            stash = {old: f"__outlet_reorder__{i}" for i, old in enumerate(relabel)}
            nx.relabel_nodes(graph, stash, copy=False)
            nx.relabel_nodes(graph, {stash[old]: new for old, new in relabel.items()}, copy=False)
        return re.sub(r"@@OUTLET:outlet_\d+@@", "", action_description)

    def _assign_reading_index(self, graph: nx.DiGraph, action_description: str) -> None:
        """Tag every node with a ``reading_index`` = the position at which the narration first
        mentions it, so the JSON converter can emit ``connections`` in the order the prompt reads.

        The order comes from the same ``@@RO:<id>@@`` / ``@@OUTLET:<id>@@`` first-mention markers
        the renumber passes consume; this must therefore run before those passes strip them. Nodes
        a marker never reached (defensive; e.g. junction nodes, which connections dissolve away) are
        appended after the narrated ones. The attribute is preserved across the later id relabels
        (``relabel_nodes`` keeps node data), and the converter never serialises it into the JSON.
        """
        order = []
        for found in re.findall(r"@@(?:RO|OUTLET):([a-z_]+_\d+)@@", action_description):
            if found not in order:
                order.append(found)
        idx = {node: i for i, node in enumerate(order)}
        nxt = len(order)
        for node in graph.nodes:
            if node in idx:
                graph.nodes[node]["reading_index"] = idx[node]
            else:
                graph.nodes[node]["reading_index"] = nxt
                nxt += 1

    # Pass-through components carry no spoken ordinal in the prose (they are referred to by a
    # synonym, e.g. "a tesla valve"), so only their JSON ids are reordered -- no text rewriting.
    _PASSTHROUGH_PREFIXES = ("mixer", "delay", "chamber", "droplet", "tesla_valve")

    def _renumber_passthrough_by_reading_order(self, graph: nx.DiGraph, action_description: str) -> str:
        """Renumber mixer/delay/chamber/droplet/tesla_valve IDs to the order the narration first
        mentions them, then strip the consumed ``@@RO@@`` markers for those types.

        These components are never named by ordinal in the prose, so -- like outlets -- only the
        node ids are reordered (no spoken ordinal to rewrite). Each dataset example is a
        (prompt, JSON) pair, so a component narrated second should be ``<prefix>_2`` in the paired
        JSON, not an arbitrary index left over from graph construction. Order comes from the
        ``@@RO:<id>@@`` first-mention markers every narrated component now emits, read left to
        right across the assembled (possibly nested-branch) text.

        ``graph`` is mutated in place because its node IDs *are* the target JSON's IDs -- the caller
        converts it to JSON only after generating the prompt. Idempotent once ids are in order.
        """
        relabel = {}
        for prefix in self._PASSTHROUGH_PREFIXES:
            ordered = []
            for found in re.findall(rf"@@RO:{prefix}_(\d+)@@", action_description):
                num = int(found)
                if num not in ordered:
                    ordered.append(num)
            present = sorted(int(n.split("_")[-1]) for n in graph.nodes if n.startswith(f"{prefix}_"))
            if not present:
                continue
            # Narrated nodes first (in reading order), then any a marker never reached (defensive).
            ordered = ordered + [i for i in present if i not in ordered]
            relabel.update({f"{prefix}_{old}": f"{prefix}_{new}"
                            for new, old in enumerate(ordered, start=1) if old != new})

        action_description = re.sub(r"@@RO:(?:mixer|delay|chamber|droplet|tesla_valve)_\d+@@", "", action_description)
        if relabel:
            # Two-phase via temporary names so the in-place permutation never clobbers a node.
            stash = {old: f"__pt_reorder__{i}" for i, old in enumerate(relabel)}
            nx.relabel_nodes(graph, stash, copy=False)
            nx.relabel_nodes(graph, {stash[old]: new for old, new in relabel.items()}, copy=False)
        return action_description

    def _split_possessive(self, split_unit_node: str) -> str:
        """Possessive anchor for a stream's split, e.g. ``@@SPLITPOSS:3@@ `` -> "the second split's ".

        Returns "" when the design has <2 splits (a bare "the first stream" is then unambiguous).
        The split id rides a marker that _resolve_split_anchors rewrites to the split's reading-order
        ordinal, so the anchor reads in the order the splits are introduced.
        """
        if self._n_splits < 2:
            return ""
        return f"@@SPLITPOSS:{int(split_unit_node.split('_')[2])}@@ "

    def _particle_stream_label(self, filter_node: str, particle_type: str) -> str:
        """Name a DLD output stream, e.g. "the smaller particle stream" -- anchored to its filter
        ("the second filter's smaller particle stream") when the design has >=2 DLD filters.

        The filter ordinal is the JSON ``filter_N`` id; _canonicalize_reading_order later renumbers
        both it and the matching node into reading order (its pattern matches "the Nth filter's").
        """
        base = f"{particle_type} particle stream"
        if self._n_dld < 2:
            return f"the {base}"
        return f"the {self._number_prefix(int(filter_node.split('_')[1]) - 1)} filter's {base}"

    def _resolve_split_anchors(self, text: str) -> str:
        """Resolve the split-anchor markers, numbering splits in the order they are introduced.

        ``@@SPLITDEF:<id>@@`` marks where each split's "split ... into N streams" clause appears, so
        the left-to-right order of those markers is the splits' narration order. Each
        ``@@SPLITPOSS:<id>@@`` anchoring a stream then becomes "<that order> split's" (a possessive),
        making every stream reference self-contained. Both markers are stripped. Runs after the
        synonym passes so the literal noun "split" is never mangled by the action-verb synonymiser.
        """
        order = []
        for found in re.findall(r"@@SPLITDEF:(\d+)@@", text):
            uid = int(found)
            if uid not in order:
                order.append(uid)
        index = {uid: i for i, uid in enumerate(order, start=1)}
        text = re.sub(r"@@SPLITPOSS:(\d+)@@",
                      lambda m: f"{self._number_prefix(index.get(int(m.group(1)), 1) - 1)} split's", text)
        return re.sub(r"@@SPLITDEF:\d+@@", "", text)

    def _dedupe_filter_phrasing(self, text: str) -> str:
        """Drop a filter type-qualifier word that the randomly chosen filter synonym then echoes.

        The qualifier (``particle-separation``, ``microstructured``, ...) and the noun synonym
        (``separation filter``, ``particle filter``, ``microfilter``, ...) are picked independently
        and concatenated, so a pairing can read "particle-separation separation filter" or
        "particle-separation particle filter". The verb already marks the filter type, so the
        echo is pure redundancy -- collapse it to a single mention ("particle-separation filter").
        """
        # Hyphenated qualifier whose first or second word is repeated as the noun's modifier.
        text = re.sub(r'\b(\w+)-(\w+) (?:\1|\2) filter\b', r'\1-\2 filter', text)
        # "micro" stutter from the lone single-word qualifier that overlaps "microfilter".
        text = text.replace('microstructured microfilter', 'microstructured filter')
        return text

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