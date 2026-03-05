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
        core_description, processed_graph = self._generate_core_description(graph)

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
            # Checks if a generated path follows specific rules for certain components.
            if len(path) < 2: return False
            for i, node in enumerate(path):
                # A droplet generator can only be at the start, second, or end position in a path.
                if node.startswith("droplet_") and not (i in (0, 1) or i == len(path) - 1): return False
                # If a droplet generator is the second node, it must be connected via its "dispersed" phase inlet.
                if node.startswith("droplet_") and i == 1 and i != len(path) - 1:
                    edge_data = graph.get_edge_data(path[i - 1], node)
                    if edge_data.get("droplet_connection_type") != "dispersed": return False
                # A filter has the same positional constraints as a droplet generator.
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

    def _generate_path_descriptions(self, graph: nx.DiGraph, paths: List[List[str]]) -> str:
        """
        This method converts the list of paths (which are lists of node names) into natural language sentences.
        It uses a set of templates to handle different types of paths, especially those involving special components like filters and droplet generators.
        """
        descriptions_with_deps = []

        for path in paths:
            current_desc = ""
            
            # --- Path Start Logic ---
            # This block handles how the description of a path begins, with special templates for filters and droplet generators.
            if path[0].startswith("filter_"):
                # Case: Path starts with a filter.
                if len(path) > 1:
                    edge_data = graph.get_edge_data(path[0], path[1], {})
                    f_type = edge_data.get("filter_connection_type")
                    current_desc = f"{f_type} particles from {path[0]}"

                if len(path) > 2:
                    if path[1].startswith("filter_"):
                        edge_data = graph.get_edge_data(path[1], path[2], {})
                        f_type = edge_data.get("filter_connection_type")
                        current_desc = f"{f_type} particles of {current_desc}"
                        if len(path) > 3: current_desc += "".join(f" through {node}" for node in path[2:-1])
                    elif path[1].startswith("droplet_"):
                        current_desc = f"drops_replace made of {current_desc}"
                        if len(path) > 3: current_desc += "".join(f" through {node}" for node in path[2:-1])
                    else:
                        current_desc += "".join(f" through {node}" for node in path[1:-1])

            elif len(path) > 2 and path[1].startswith("filter_"):
                # Case: Filter is the second node in the path.
                edge_data = graph.get_edge_data(path[1], path[2], {})
                f_type = edge_data.get("filter_connection_type")
                current_desc = f"{f_type} particles from {path[0]}"
                if len(path) > 3: current_desc += "".join(f" through {node}" for node in path[2:-1])

            elif len(path) > 2 and path[1].startswith("droplet_"):
                # Case: Droplet generator is the second node in the path.
                start_desc = f"fluid entering {path[0]}" if path[0].startswith("inlet_") else f"fluid exiting {path[0]}"
                current_desc = f"guide drops_replace made from {start_desc}"
                if len(path) > 3: current_desc += "".join(f" through {node}" for node in path[2:-1])

            else:  # General path case
                current_desc = f"from {path[0]}"
                if len(path) > 2: current_desc += "".join(f" through {node}" for node in path[1:-1])

            # --- Path End Logic ---
            # This block handles how the description of a path ends.
            if len(path) > 1:
                if path[-1].startswith("droplet_"):
                    # Case: Path ends at a droplet generator.
                    edge_data = graph.get_edge_data(path[-2], path[-1], {})
                    d_type = edge_data.get("droplet_connection_type")
                    current_desc += f" to {d_type} phase inlet of {path[-1]}"
                else:
                    current_desc += f" to {path[-1]}"

            ancestor_count = len(nx.ancestors(graph, path[0]))
            descriptions_with_deps.append((ancestor_count, current_desc))

        # --- Final Sorting and Joining ---
        # Sort the path descriptions based on their dependencies to create a logical flow.
        descriptions_with_deps.sort(key=lambda x: x[0])
        return ". ".join(desc for _, desc in descriptions_with_deps)

    def _generate_core_description(self, graph: nx.DiGraph) -> Tuple[str, nx.DiGraph]:
        """Generates a path-based description of the chip's functionality."""

        # First, remove junction nodes from the graph to create paths that describe the conceptual flow between functional components.
        processed_graph = self._remove_junction_nodes(graph)

        # Find a set of random paths that cover all the connections in the simplified graph.
        paths = self._find_random_paths(processed_graph)

        # Convert these paths into a single string of natural language descriptions.
        description = self._generate_path_descriptions(processed_graph, paths)
        return description, processed_graph