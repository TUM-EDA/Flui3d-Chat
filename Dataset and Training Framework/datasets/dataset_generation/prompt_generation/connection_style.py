import random
from collections import defaultdict
from typing import Dict, List, Tuple

import networkx as nx

from .prompt_generation import StructuralPromptGenerator

class ConnectionOrientedPromptGenerator(StructuralPromptGenerator):
    """Generates prompts describing the chip's component connections."""

    # A list of introductory phrases for prompts that include a list of components.
    # This helps in creating more varied prompts.
    PROMPT_BEGINNINGS_WITH_COMPONENT_LIST = [
        "Design a microfluidic chip that uses the following components and connections:",
        "Create a microfluidic device with",
        "Based on the described setup of components, generate a microfluidic chip design:",
        "Develop a layout for a microfluidic chip using the components described:",
        "Compose a schematic for a microfluidic chip with",
        "Generate a microfluidic chip design using",
        "Construct a functional microfluidic chip by arranging the listed components with the specified connections:",
        "Using the described components and their connections, create a microfluidic chip:",
        "Develop a configuration for a microfluidic chip where the following components are linked according to the provided specifications:",
        "Design a microfluidic system based on",
        "Design a microfluidic chip with"
    ]

    # A list of introductory phrases for prompts that do not list the components upfront.
    # This provides another way to structure the prompt, focusing solely on the connections.
    PROMPT_BEGINNINGS_WO_COMPONENT_LIST = [
        "Design a microfluidic chip based on the following connections between components:",
        "Generate a microfluidic chip layout where the connections between components are described as follows:",
        "Create a functional microfluidic chip by arranging components according to these connection details:",
        "Develop a microfluidic system design that implements the following component connections:",
        "Construct a microfluidic chip by following the specified connections between components:",
        "Using the outlined connections, generate a schematic for a microfluidic chip design:",
        "Design a microfluidic chip layout where the functionality is defined by the given connections:",
        "Compose a configuration for a microfluidic chip based on these interconnections between components:",
        "Build a microfluidic chip design.",
        "Create a microfluidic chip.",
        "Design a microfluidic chip."
    ]

    def _generate_for_single_graph(self, graph: nx.DiGraph, graph_id: str) -> Tuple[Dict, Dict]:
        """
        Generates a connection-oriented prompt set for a single graph.
        """
        # Generate the core textual description of the connections from the graph.
        core_description, processed_graph = self._generate_core_description(graph)

        # Get the count of each type of module (e.g., 2 mixers, 3 inlets).
        module_counts = self._get_module_counts(processed_graph)
        
        # Generate other structural parts of the prompt, like component lists and parameter specifications.
        prefix_counts, prefix_detailed, attributes_text, suffix_text = self._generate_structural_descriptions(processed_graph)
        
        # Create several different versions of the final prompt by combining the generated parts in various ways.
        versions = self._create_prompt_versions(
            core_description, prefix_counts, prefix_detailed, attributes_text, suffix_text, processed_graph
        )

        # Randomly select one of the generated prompt versions to use as the final prompt.
        selected_version_text = random.choice(list(versions.values()))

        # Randomly choose a naming scheme for components (e.g., "mixer 1", "first mixer").
        schema_choice = random.choice([0, 1, 2, 3])

         # Randomly select synonyms for module types to increase linguistic diversity (e.g., "chamber" vs. "reaction unit").
        selected_synonyms = {module: random.choice(options) for module, options in self.MODULE_SYNONYMS.items()}
        
        # Apply the chosen naming scheme and synonyms to the selected prompt text.
        final_prompt = self._replace_module_names(selected_version_text, module_counts, selected_synonyms, schema=schema_choice)

        # Capitalize the first letter of each sentence in the prompt for correct grammar.
        final_prompt = self._capitalize_sentences(final_prompt)

        # For this specific style, both the auxiliary LLM and direct-use outputs
        # use the same final prompt.
        prompts_for_llm = {"id": graph_id, "prompt": final_prompt}
        prompts_wo_llm = {"id": graph_id, "prompt": final_prompt}
        
        return prompts_for_llm, prompts_wo_llm
        

    def _generate_core_description(self, graph: nx.DiGraph) -> Tuple[str, nx.DiGraph]:
        """
        This method generates the main part of the prompt: a natural language description of the component connections.
        It traverses the graph representation of the chip and builds a textual description.
        """
        # Randomly decide whether to include 'junction' components explicitly in the description.
        # If not, the LLM will have to infer their existence, making the task more challenging.
        with_junctions = random.choice([True, False])
        processed_graph = graph if with_junctions else self._remove_junction_nodes(graph)

        # A list to hold the descriptions of each connection, along with their dependencies (what needs to be described first).
        descriptions_with_deps = []
        visited_nodes = []
        visited_edges = set()

        # A helper function to get the type and ID from a node name like "inlet_1".
        def extract_type_and_id(node):
            node_type, node_id = node.split("_")
            return node_type, int(node_id)

        # Sort the nodes to ensure the traversal is always in a topological consistent order.
        sorted_nodes = sorted(processed_graph.nodes, key=extract_type_and_id)

        # Start the traversal from the inlets, which are the entry points of the chip.
        inlets = [node for node in sorted_nodes if node.startswith("inlet")]

        # Traverse the graph starting from each inlet.
        for inlet in inlets:
            stack = [inlet]
            while stack:
                current = stack.pop()

                # Get the incoming and outgoing connections for the current node.
                incoming_edges = [
                    (source, current, processed_graph.get_edge_data(source, current))
                    for source in processed_graph.predecessors(current)
                    if source in inlets or source in stack or source in visited_nodes
                ]

                outgoing_edges = [
                    (current, target, processed_graph.get_edge_data(current, target))
                    for target in processed_graph.successors(current)
                ]

                # This section handles nodes that have more than one input.
                # It groups the sources and creates a description like "connect inlet_1 and inlet_2 to mixer_1".
                if len(incoming_edges) > 1:
                    source_texts, target_texts = [], []
                    for source, target, edge_data in incoming_edges:
                        if (source, target) in visited_edges:
                            continue
                        visited_nodes.append(source)
                        visited_edges.add((source, target))

                        # Handle special connection types for filters, which have different outlets (for smaller and larger particles).
                        if "filter_connection_type" in edge_data:
                            f_type = edge_data["filter_connection_type"]
                            source_texts.append(f"the {f_type} particle outlet{'s' if 'and' in f_type else ''} of {source}")
                        else:
                            source_texts.append(f"{source}")

                        # Handle special connection types for droplet generators, which have distinct inlets for the two different liquids.
                        if "droplet_connection_type" in edge_data:
                            d_type = edge_data["droplet_connection_type"]
                            target_texts.append(f"the {d_type} phase inlet{'s' if 'and' in d_type else ''} of {target}")
                    
                    if source_texts:
                        if target_texts:
                            grouped_targets = defaultdict(list)
                            for src_txt, tgt_txt in zip(source_texts, target_texts):
                                grouped_targets[tgt_txt].append(src_txt)
                            
                            combined = [f"{' and '.join(srcs)} to {tgt}" for tgt, srcs in grouped_targets.items()]
                            descriptions_with_deps.append({
                                "text": "connect " + " and ".join(combined),
                                "dependencies": list(nx.ancestors(processed_graph, current))
                            })
                        else:
                            descriptions_with_deps.append({
                                "text": f"connect {' and '.join(source_texts)} to {current}",
                                "dependencies": list(nx.ancestors(processed_graph, current))
                            })

                # This section handles linear connections (one input, one output) and splits (one input, multiple outputs).
                if outgoing_edges:
                    source_texts, target_texts = [], []
                    for source, target, edge_data in outgoing_edges:
                        if (source, target) in visited_edges:
                            continue
                        
                        stack.append(target)
                        visited_nodes.append(source)

                        # Delays describing an edge that leads into a multi-input
                        # component until that component is processed.
                        if processed_graph.in_degree(target) > 1:
                            continue
                        
                        visited_edges.add((source, target))
                        edge_data = edge_data or {}
                        
                        # Special handling for filter and droplet generator connections, similar to the incoming edge logic.
                        if "filter_connection_type" in edge_data:
                            f_type = edge_data["filter_connection_type"]
                            source_texts.append(f"the {f_type} particle outlet{'s' if 'and' in f_type else ''} of {source}")
                        
                        if "droplet_connection_type" in edge_data:
                            d_type = edge_data["droplet_connection_type"]
                            target_texts.append(f"the {d_type} phase inlet{'s' if 'and' in d_type else ''} of {target}")
                        else:
                            target_texts.append(f"{target}")

                    if target_texts:
                        if source_texts:
                            grouped_sources = defaultdict(list)
                            for src_txt, tgt_txt in zip(source_texts, target_texts):
                                grouped_sources[src_txt].append(tgt_txt)
                            
                            combined = [f"{src} to {' and '.join(tgts)}" for src, tgts in grouped_sources.items()]
                            descriptions_with_deps.append({
                                "text": "connect " + " and ".join(combined),
                                "dependencies": [current] + list(nx.ancestors(processed_graph, current))
                            })
                        else:
                            descriptions_with_deps.append({
                                "text": f"connect {current} to {' and '.join(target_texts)}",
                                "dependencies": [current] + list(nx.ancestors(processed_graph, current))
                            })

        # Sort the generated description parts based on their dependencies to create a coherent narrative.
        descriptions_with_deps.sort(key=lambda item: len(item["dependencies"]))

        # Join the sorted parts into a single string.
        description_text = ". ".join(item["text"] for item in descriptions_with_deps)

        return description_text, processed_graph