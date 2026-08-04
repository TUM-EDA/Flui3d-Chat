import random
from typing import Dict, List, Tuple

import networkx as nx

from .prompt_generation import StructuralPromptGenerator

class PathOrientedPromptGenerator(StructuralPromptGenerator):
    """Generates prompts describing the fluidic paths through the chip."""

    # A list of introductory phrases for prompts that also include a list of components.
    PROMPT_BEGINNINGS_WITH_COMPONENT_LIST = [
        "Design a microfluidic chip that uses the following components and achieves these fluid paths:",
        "Create a microfluidic device with",
        "Based on the described setup of fluid paths, generate a microfluidic chip:",
        "Develop a layout for a microfluidic chip using the components described and ensuring the following fluid flow paths:",
        "Compose a schematic for a microfluidic chip with",
        "Generate a microfluidic chip design using",
        "Construct a functional microfluidic chip by arranging the listed components to achieve the specified fluid flow paths:",
        "Using the described components and the fluid flow patterns, create a microfluidic chip:",
        "Develop a configuration for a microfluidic chip where the listed components enable these fluid paths:",
        "Design a microfluidic system based on",
        "Design a microfluidic chip with"
    ]

    # A list of introductory phrases for prompts that focus only on the fluid paths without an initial component list.
    PROMPT_BEGINNINGS_WO_COMPONENT_LIST = [
        "Design a microfluidic chip based on the following fluid paths between components:",
        "Generate a microfluidic chip layout to achieve the following fluid flow patterns:",
        "Create a functional microfluidic chip by ensuring the specified fluid paths:",
        "Develop a microfluidic system design where the fluid follows these paths:",
        "Construct a microfluidic chip based on the given fluid flow paths:",
        "Using the outlined fluid paths, generate a schematic for a microfluidic chip design:",
        "Design a microfluidic chip layout where the functionality is determined by these fluid flow paths:",
        "Compose a configuration for a microfluidic chip based on these fluid dynamics:",
        "Build a microfluidic chip design that supports the following fluid paths:",
        "Create a microfluidic chip to enable the described fluid flow patterns:",
        "Design a microfluidic chip to implement these fluid paths:"
    ]

    def _generate_for_single_graph(self, graph: nx.DiGraph, graph_id: str) -> Tuple[Dict, Dict]:
        """
        Generates a path-oriented prompt set for a single graph.
        """
        # Generate the core path-based description from the graph representation.
        core_description, processed_graph, ordered_edges = self._generate_core_description(graph)

        # Record the order the prose narrates each connection (`ordered_edges`, a list of
        # (source, target) node pairs from walking the paths) on the (original) graph, so the JSON
        # converter lists `connections` in that exact order. This is EDGE-based, not node-based: a
        # node can be introduced early as a path endpoint (e.g. a filter that ends one path) yet have
        # its own out-edges narrated much later in separate paths, which a per-node order cannot
        # express (it would group all of a node's out-edges right after the node). Walking the paths
        # also sees the droplet generators / second-position filters the prose leaves unnamed, so
        # every edge lands at its true narration position. Only reorders the JSON; the prompt text is
        # untouched.
        graph.graph["connection_reading_order"] = ordered_edges

        # Count the number of each type of component in the chip.
        module_counts = self._get_module_counts(processed_graph)
        
        # Generate other descriptive parts, such as component lists and their attributes.
        prefix_counts, prefix_detailed, attributes_text, suffix_text = self._generate_structural_descriptions(processed_graph)
        
        # Create a dictionary of different prompt versions by combining the generated text parts.
        versions = self._create_prompt_versions(
            core_description, prefix_counts, prefix_detailed, attributes_text, suffix_text, processed_graph
        )

        # --- Generate final prompt for direct use (wo_llm) ---
        selected_version_text = random.choice(list(versions.values()))

        # Randomly select a naming scheme (e.g., "inlet 1", "first inlet").
        schema_choice = random.choice([0, 1, 2, 3])

        # Randomly choose synonyms for component names to increase variety.
        selected_synonyms = {module: random.choice(options) for module, options in self.MODULE_SYNONYMS.items()}
        
        # Construct the final prompt by applying the chosen naming scheme and synonyms.
        final_prompt = self._replace_module_names(selected_version_text, module_counts, selected_synonyms, schema=schema_choice)
        final_prompt = self._capitalize_sentences(final_prompt)

        # --- Generate description for auxiliary LLM ---
        # This creates a structured, intermediate prompt that will be fed to another LLM (like GPT-4o mini)
        # to be "naturalized" into a more human-like, assay-style description.
        schema_choice_desc = random.choice([0, 1, 2, 3])
        synonyms_desc = {module: random.choice(options) for module, options in self.MODULE_SYNONYMS.items()}
        
        llm_path_desc = self._replace_module_names(f"{core_description}.", module_counts, synonyms_desc, schema=schema_choice_desc)
        llm_comp_desc = ""
        if suffix_text:
            llm_comp_desc = self._replace_module_names(f"{suffix_text}.", module_counts, synonyms_desc, schema=schema_choice_desc)
        
        llm_path_desc = self._capitalize_sentences(llm_path_desc)
        llm_comp_desc = self._capitalize_sentences(llm_comp_desc)

        combined_desc = f"{llm_path_desc}{' ' + llm_comp_desc if llm_comp_desc else ''}"

        # Return two dictionaries: one for the auxiliary LLM and one for direct use.
        return (
            {"id": graph_id, "prompt": combined_desc.strip()},
            {"id": graph_id, "prompt": final_prompt}
        )


    def _find_random_paths(self, graph: nx.DiGraph) -> List[List[str]]:
        """
        This method finds a set of random paths through the chip graph.
        The goal is to select a collection of paths that covers every connection (edge) in the graph at least once.
        It also applies certain constraints to ensure the paths are realistic for microfluidic processes.
        """
        def is_valid_path(path):
            # Checks if a generated path follows positional rules for certain components.
            if len(path) < 2: return False
            for i, node in enumerate(path):
                # Droplet generators and filters read naturally only at a path boundary
                # (start, second, or last node), so they are never buried mid-path. In v2 a
                # droplet generator is a single-input/single-output component, so the old
                # "dispersed phase inlet" entry constraint no longer applies.
                if node.startswith("droplet_") and not (i in (0, 1) or i == len(path) - 1): return False
                if node.startswith("filter_") and not (i in (0, 1) or i == len(path) - 1): return False
            return True

        def is_subsequence(sub, full_list):
            # Helper function to check if one path is just a smaller part of another existing path.
            len_sub = len(sub)
            return any(full_list[i:i+len_sub] == sub for i in range(len(full_list) - len_sub + 1))

        all_edges = set(graph.edges)
        covered_edges = set()
        paths = []

        # Keep generating random paths until every edge in the graph has been included in at least one path.
        while covered_edges != all_edges:
            start_node = random.choice(list(graph.nodes))
            current_path = [start_node]
            current_node = start_node
            
            # Randomly walk through the graph from the start node to create a path.
            while True:
                successors = list(graph.successors(current_node))
                if not successors or random.random() < 0.2:
                    break
                current_node = random.choice(successors)
                current_path.append(current_node)
            
            # If the path is valid and not a subsequence of an existing path, add it to our list.
            if is_valid_path(current_path) and not any(is_subsequence(current_path, p) for p in paths):
                paths.append(current_path)
                # Mark the edges covered by this new path.
                for u, v in zip(current_path, current_path[1:]):
                    covered_edges.add((u, v))
        return paths

    def _generate_path_descriptions(self, graph: nx.DiGraph, paths: List[List[str]]) -> Tuple[str, List[Tuple[str, str]]]:
        """
        This method converts the list of paths (which are lists of node names) into natural language sentences.
        It uses a set of templates to handle different types of paths, especially those involving special components like filters and droplet generators.

        Only DLD filters expose size-sorted output ports (``filter_connection_type`` =
        ``smaller``/``larger``); single-output pillar-matrix filters and every other component
        carry no port label and are narrated as ordinary pass-through nodes. In v2 a droplet
        generator has a single input/output, so it no longer carries a phase-inlet label.

        Returns the joined description and the narration-order edge list: every (source, target)
        node pair, in the order it first appears while walking the paths in the SAME sorted order the
        prose uses. These pairs are the (junction-free) connections, so the caller can hand them to
        the JSON converter to list connections in prose order; see ``connection_reading_order``.
        """
        descriptions_with_deps = []

        def port(u: str, v: str) -> str:
            """The size-sorted output label of a DLD filter edge, or None for any other edge."""
            return (graph.get_edge_data(u, v) or {}).get("filter_connection_type")

        def stream(label: str) -> str:
            """Phrases a DLD output fraction as a particle-stream noun.

            A single fraction renders as e.g. "smaller particles". A compound label
            ("smaller and larger") arises when both of a filter's outputs recombine
            downstream (junction collapse merges the two ports onto one edge); it renders
            as "both particle fractions" so the recombination is still conveyed without the
            awkward "smaller and larger particles ..." doubling (and its "... of larger
            particles" stacking when the filter feeds another filter).
            """
            return "both particle fractions" if " and " in label else f"{label} particles"

        for path in paths:
            current_desc = ""
            start_port = port(path[0], path[1]) if len(path) > 1 else None
            second_port = port(path[1], path[2]) if len(path) > 2 else None

            # --- Path Start Logic ---
            # Special templates apply only to DLD filters (which carry a size-sorted port) and to
            # droplet generators; pillar-matrix filters and everything else fall to the general case.
            if path[0].startswith("filter_") and start_port:
                # Case: Path starts at a DLD filter, following one of its size-sorted outputs.
                current_desc = f"{stream(start_port)} from {path[0]}"
                if len(path) > 2:
                    if path[1].startswith("filter_") and second_port:
                        current_desc = f"{stream(second_port)} of {current_desc}"
                        if len(path) > 3: current_desc += "".join(f" through {node}" for node in path[2:-1])
                    elif path[1].startswith("droplet_"):
                        current_desc = f"drops_replace made of {current_desc}"
                        if len(path) > 3: current_desc += "".join(f" through {node}" for node in path[2:-1])
                    else:
                        current_desc += "".join(f" through {node}" for node in path[1:-1])

            elif len(path) > 2 and path[1].startswith("filter_") and second_port:
                # Case: A DLD filter is the second node. This phrasing deliberately leaves the
                # filter unnamed (for prompt variety): "particles OF THE FLUID from <X>" attributes
                # the size split to the fluid, not to <X>, so it no longer reads as if <X> itself
                # sorted the particles, while the separation stays implicit.
                current_desc = f"{stream(second_port)} of the fluid from {path[0]}"
                if len(path) > 3: current_desc += "".join(f" through {node}" for node in path[2:-1])

            elif len(path) > 2 and path[1].startswith("droplet_"):
                # Case: Droplet generator is the second node; the path carries the droplets it forms.
                start_desc = f"fluid entering {path[0]}" if path[0].startswith("inlet_") else f"fluid exiting {path[0]}"
                current_desc = f"guide drops_replace made from {start_desc}"
                if len(path) > 3: current_desc += "".join(f" through {node}" for node in path[2:-1])

            else:  # General path case (incl. single-output pillar-matrix filters and tesla valves).
                current_desc = f"from {path[0]}"
                if len(path) > 2: current_desc += "".join(f" through {node}" for node in path[1:-1])

            # --- Path End Logic ---
            # Every component is entered the same way; a droplet generator no longer has a
            # phase-specific inlet in v2, so the path simply ends "to <node>".
            if len(path) > 1:
                current_desc += f" to {path[-1]}"

            ancestor_count = len(nx.ancestors(graph, path[0]))
            descriptions_with_deps.append((ancestor_count, path, current_desc))

        # --- Final Sorting and Joining ---
        # Sort the path descriptions based on their dependencies to create a logical flow. The key is
        # only the ancestor count (a stable sort keeps ties in discovery order); never compare the
        # path lists themselves.
        descriptions_with_deps.sort(key=lambda x: x[0])

        # Narration-order edge list: first appearance of each (source, target) pair while walking the
        # sorted paths edge by edge. Unlike a text scan, this walk sees the droplet generators /
        # second-position filters the prose leaves unnamed, placing each edge at its true position in
        # the flow; and being edge-level it keeps a node's out-edges where they are actually narrated
        # rather than grouped at the node's first mention.
        ordered_edges: List[Tuple[str, str]] = []
        seen = set()
        for _, path, _ in descriptions_with_deps:
            for edge in zip(path, path[1:]):
                if edge not in seen:
                    seen.add(edge)
                    ordered_edges.append(edge)

        description = ". ".join(desc for _, _, desc in descriptions_with_deps)
        return description, ordered_edges

    def _generate_core_description(self, graph: nx.DiGraph) -> Tuple[str, nx.DiGraph, List[Tuple[str, str]]]:
        """Generates a path-based description of the chip's functionality.

        Returns the description, the junction-free processed graph, and the narration-order edge list
        (used to order the JSON connections in prose order; see ``connection_reading_order``).
        """

        # First, remove junction nodes from the graph to create paths that describe the conceptual flow between functional components.
        processed_graph = self._remove_junction_nodes(graph)

        # Find a set of random paths that cover all the connections in the simplified graph.
        paths = self._find_random_paths(processed_graph)

        # Convert these paths into a single string of natural language descriptions plus the
        # narration-order edge walk.
        description, ordered_edges = self._generate_path_descriptions(processed_graph, paths)
        return description, processed_graph, ordered_edges