import random
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Set

import networkx as nx

class PromptGenerator:
    """
    A base class for generating various styles of microfluidic chip design prompts.

    This class provides shared functionalities, data, and a common interface
    for its subclasses, each of which implements a specific prompt generation style.

    Attributes:
        graphs_data (List[Dict[str, Any]]): A list of dictionaries, where each
            dictionary represents a graph with its 'id' and 'graph' data.
    """

    # A dictionary of synonyms for different microfluidic components.
    # This is used to create more varied prompts.
    MODULE_SYNONYMS = {
        "inlet": ["inlet", "fluid inlet", "entry point", "input port", "fluid entry", "input"],
        "outlet": ["outlet", "fluid outlet", "exit point", "output port", "fluid exit", "output"],
        "chamber": ["chamber", "reaction chamber", "microchamber", "reaction vessel", "reaction unit"],
        "mixer": ["mixer", "serpentine", "serpentine channel", "twisted channel", "winding path", "curved channel", "curved microchannel", "serpentine microchannel", "twisted microchannel", "serpentine mixer", "mixing unit", "mixing channel", "mixing microchannel", "blender", "mixing ring"],
        "delay": ["delay", "serpentine", "serpentine channel", "twisted channel", "winding path", "curved channel", "curved microchannel", "serpentine microchannel", "twisted microchannel", "delay channel", "delay microchannel", "delaying channel", "delaying microchannel"],
        "filter": ["filter", "particle filter", "separation filter", "DLD filter", "microfilter"],
        "droplet": ["droplet generator", "microdroplet generator"],
        "junction": ["junction", "intersection", "crossing"],
    }

    # A dictionary holding the default values for component parameters.
    # These are used when a prompt doesn't specify a value.
    DEFAULT_ATTRIBUTES = {
        "length": 4000,
        "width": 3200,
        "num_turnings": 4,
        "critical_particle_diameter": 10
    }

    def __init__(self, graphs_data: List[Dict[str, Any]]):
        """
        Initializes the PromptGenerator with graph and chip data.

        Args:
            graphs_data: A list of graph entries, each a dict with 'id' and 'graph'.
        """
        self.graphs_data = graphs_data

    def generate_prompts(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Generates prompts for all graphs.

        This is the main public method to be called. It iterates through all
        graphs and uses the subclass-specific implementation to generate prompts.

        Returns:
            A tuple containing two lists of dictionaries:
            - The first list is formatted for an auxiliary LLM.
            - The second list is formatted for direct use (without an auxiliary LLM).
        """
        prompts_for_llm = []
        prompts_wo_llm = []

        for entry in self.graphs_data:
            graph = entry['graph']
            graph_id = entry['id']

            
            for_llm, wo_llm = self._generate_for_single_graph(graph, graph_id)
            prompts_for_llm.append(for_llm)
            prompts_wo_llm.append(wo_llm)

        return prompts_for_llm, prompts_wo_llm

    def _generate_for_single_graph(self, graph: nx.DiGraph, graph_id: str) -> Tuple[Dict, Dict]:
        """
        Abstract method to generate prompts for a single graph.

        Subclasses must implement this method to define the specific logic for
        their prompt style.

        Args:
            graph: The NetworkX graph of the microfluidic chip.
            graph_id: The ID of the graph.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @staticmethod
    def _capitalize_sentences(text: str) -> str:
        """Capitalizes the first letter of every sentence in the given text."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        capitalized = ' '.join(sentence[0].upper() + sentence[1:] if sentence else '' for sentence in sentences)
        return capitalized

    def _get_module_counts(self, graph: nx.DiGraph) -> Dict[str, int]:
        """Counts the number of modules of each type in the graph."""
        module_counts = defaultdict(int)
        for node in graph.nodes():
            node_type = node.split("_")[0]
            if node_type in self.MODULE_SYNONYMS:
                module_counts[node_type] += 1
        return dict(module_counts)

    def _replace_module_names(self, version_text: str, module_counts: Dict[str, int],
                              selected_synonyms: Dict[str, str], schema: int = 0) -> str:
        """Replaces module names in text according to a specified schema."""
        if schema == 0:
            return version_text.replace("drops_replace", "droplets")

        def get_replacement(module_type: str, module_id: int) -> str:
            if schema == 1:
                ordinal_map = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
                return f"{ordinal_map[module_id - 1]} {module_type}" if module_id <= len(ordinal_map) else f"{module_id}th {module_type}"
            elif schema == 2:
                return f"{module_type} {module_id}"
            elif schema == 3:
                return f"{module_type} {chr(64 + module_id)}"


        def replace_suffix(match: re.Match) -> str:
            module_type, module_id = match.group(1), int(match.group(2))
            if module_counts.get(module_type) > 1 or random.random() < 0.5:
                return get_replacement(module_type, module_id)
            return module_type

        def replace_module_without_suffix(match: re.Match) -> str:
            return selected_synonyms.get(match.group(1))

        pattern_with_id = r"(inlet|outlet|chamber|mixer|delay|filter|droplet|junction)_(\d+)"
        version_text = re.sub(pattern_with_id, replace_suffix, version_text)

        pattern_without_id = r"(inlet|outlet|chamber|mixer|delay|filter|droplet|junction)"
        version_text = re.sub(pattern_without_id, replace_module_without_suffix, version_text)

        return version_text.replace("drops_replace", "droplets")



class StructuralPromptGenerator(PromptGenerator):
    """
    A generator for prompts that describe the structure of the chip,
    such as connections or paths. This class contains shared logic for
    connection- and path-oriented styles.
    """

    # A dictionary of synonyms.
    MODULE_SYNONYMS = {
        "inlet": ["inlet", "fluid inlet", "entry point", "input port", "fluid entry", "input"],
        "outlet": ["outlet", "fluid outlet", "exit point", "output port", "fluid exit", "output"],
        "chamber": ["chamber", "reaction chamber", "microchamber", "reaction vessel", "reaction unit"],
        "mixer": ["mixer", "mixing serpentine", "mixing serpentine channel", "mixing twisted channel", "mixing winding path", "mixing curved channel", "mixing curved microchannel", "mixing serpentine microchannel", "mixing twisted microchannel", "serpentine mixer", "mixing unit", "mixing channel", "mixing microchannel", "blender", "mixing ring"],
        "delay": ["delay", "delaying serpentine", "delaying serpentine channel", "delaying twisted channel", "delaying winding path", "delaying curved channel", "delaying curved microchannel", "delaying serpentine microchannel", "delaying twisted microchannel", "delay channel", "delay microchannel", "delaying channel", "delaying microchannel"],
        "filter": ["filter", "particle filter", "separation filter", "DLD filter", "microfilter"],
        "droplet": ["droplet generator", "microdroplet generator"],
        "junction": ["junction", "intersection", "crossing"],
    }

    PROMPT_BEGINNINGS_WITH_COMPONENT_LIST: List[str] = []
    PROMPT_BEGINNINGS_WO_COMPONENT_LIST: List[str] = []

    @staticmethod
    def _remove_junction_nodes(graph: nx.DiGraph) -> nx.DiGraph:
        """Removes junction nodes while preserving connectivity and attributes."""
        new_graph = graph.copy()
        junction_nodes = [node for node in new_graph.nodes if "junction" in node]

        for junction in junction_nodes:
            incoming = list(new_graph.in_edges(junction, data=True))
            outgoing = list(new_graph.out_edges(junction, data=True))
            
            for src, _, attr_in in incoming:
                for _, tgt, attr_out in outgoing:
                    combined_attrs = {**attr_in, **attr_out}
                    if new_graph.has_edge(src, tgt):
                        for key, value in combined_attrs.items():
                            if key in new_graph[src][tgt]:
                                new_graph[src][tgt][key] = f"{new_graph[src][tgt][key]} and {value}"
                            else:
                                new_graph[src][tgt][key] = value
                    else:
                        new_graph.add_edge(src, tgt, **combined_attrs)
            new_graph.remove_node(junction)
        return new_graph

    def _generate_structural_descriptions(self, graph: nx.DiGraph) -> Tuple[str, str, str, str]:
        """
        Generates various descriptive parts of the chip's structure.

        This includes component counts, detailed component lists with attributes,
        and descriptions with/without inline attributes.

        Args:
            graph: The NetworkX graph of the chip.

        Returns:
            A tuple containing:
            - prefix_counts: A string listing component counts.
            - prefix_detailed: A detailed string of components and their attributes.
            - attributes_text: Attributes (used as inline attributes later).
            - suffix_text: A string of suffix attribute descriptions.
        """
        node_types = defaultdict(list)
        attributes_text = {}
        for node, data in graph.nodes(data=True):
            node_type = node.split("_")[0]
            if node_type in self.MODULE_SYNONYMS:
                node_types[node_type].append(node)

            non_default_attrs = []
            for attr, value in data.items():
                if node_type == 'junction' and attr == 'function':
                    continue
                
                is_metric = "length" in attr or "width" in attr or "critical_particle_diameter" in attr
                unit = " µm" if is_metric else ""
                
                if attr in self.DEFAULT_ATTRIBUTES and value != self.DEFAULT_ATTRIBUTES[attr]:
                    non_default_attrs.append(f"{attr.replace('_', ' ')}: {value}{unit}")
                elif attr not in self.DEFAULT_ATTRIBUTES and random.random() < 0.5:
                    non_default_attrs.append(f"{attr.replace('_', ' ')}: {value}{unit}")
            
            if non_default_attrs:
                attributes_text[node] = f" ({', '.join(non_default_attrs)})"

        prefix_counts = ", ".join(
            f"{len(modules)} {module_type}{'s' if len(modules) > 1 else ''}"
            for module_type, modules in sorted(node_types.items())
        )

        prefix_detailed = "\n".join(
            f"{len(modules)} {module_type}{'s' if len(modules) > 1 else ''}:\n" +
            ", ".join(f"{node}{attributes_text.get(node, '')}" for node in modules)
            for module_type, modules in sorted(node_types.items())
        )

        suffix_attributes = []
        for node, text in attributes_text.items():
            node_type = node.split("_")[0]
            attrs = text.replace(" (", "").replace(")", "").split(", ")
            if node_type == 'chamber':
                length = next((a.replace(":", " of") for a in attrs if "length" in a), None)
                width = next((a.replace(":", " of") for a in attrs if "width" in a), None)
                if length and width: suffix_attributes.append(f"{node} should have a {length} and a {width}")
                elif length: suffix_attributes.append(f"{node} should have a {length}")
                elif width: suffix_attributes.append(f"{node} should have a {width}")
            elif node_type in ['mixer', 'delay']:
                turnings = next((a.replace("num turnings: ", "") for a in attrs if "num turnings" in a), None)
                if turnings: suffix_attributes.append(f"{node} should have {turnings} turnings")
            elif node_type == 'filter':
                diameter = next((a.replace(":", " of") for a in attrs if "critical particle diameter" in a), None)
                if diameter: suffix_attributes.append(f"{node} should have a {diameter}")
            elif node_type == 'junction':
                j_type = next((a.replace("type: ", "") for a in attrs if "type" in a), None)
                if j_type: suffix_attributes.append(f"{node} should be a {j_type}")
        
        return prefix_counts, prefix_detailed, attributes_text, ". ".join(suffix_attributes)


    def _create_prompt_versions(self, descriptions: str, prefix_counts: str, prefix_detailed: str,
                                attributes_text: Dict[str, str], suffix_text: str, graph: nx.DiGraph) -> Dict[str, str]:
        """Creates five different versions of the prompt for diversity."""
        
        # Create a version of the main description that has component attributes inserted directly into the text.
        mentioned_nodes = set()
        updated_descriptions_parts = []
        for desc_part in descriptions.split(". "):
            for node in graph.nodes:
                if node in desc_part and node not in mentioned_nodes:
                    desc_part = desc_part.replace(node, f"{node}{attributes_text.get(node, '')}", 1)
                    mentioned_nodes.add(node)
            updated_descriptions_parts.append(desc_part)
        descriptions_with_attributes = ". ".join(updated_descriptions_parts)
        
        # Version 1: Core description + a list of attributes at the end.
        v1 = f"{random.choice(self.PROMPT_BEGINNINGS_WO_COMPONENT_LIST)} {descriptions}. {suffix_text}.".replace(" .", "")

        # Version 2: Core description with attributes mixed in.
        v2 = f"{random.choice(self.PROMPT_BEGINNINGS_WO_COMPONENT_LIST)} {descriptions_with_attributes}."

        # Version 3: List of component counts at the beginning + core description + attributes at the end.
        v3 = f"{random.choice(self.PROMPT_BEGINNINGS_WITH_COMPONENT_LIST)} {prefix_counts}. {descriptions}. {suffix_text}.".replace(" .", "")

        # Version 4: List of component counts at the beginning + core description with attributes mixed in.
        v4 = f"{random.choice(self.PROMPT_BEGINNINGS_WITH_COMPONENT_LIST)} {prefix_counts}. {descriptions_with_attributes}."

        # Version 5: Detailed list of components and their attributes at the beginning + core description.
        v5 = f"{random.choice(self.PROMPT_BEGINNINGS_WITH_COMPONENT_LIST)} {prefix_detailed}\n\n{descriptions}."

        return {"v1": v1, "v2": v2, "v3": v3, "v4": v4, "v5": v5}
        
    def _generate_for_single_graph(self, graph: nx.DiGraph, graph_id: str) -> Tuple[Dict, Dict]:
        """
        Abstract method to generate prompts for a single graph.

        Subclasses must implement this method to define the specific logic for
        their prompt style.

        Args:
            graph: The NetworkX graph of the microfluidic chip.
            graph_id: The ID of the graph.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _generate_core_description(self, graph: nx.DiGraph) -> Tuple[str, nx.DiGraph]:
        """
        Abstract method to generate the core natural language description string.
        Must be implemented by subclasses.
        
        Returns:
            A tuple containing the core description string and the processed graph.
        """
        raise NotImplementedError("Subclasses must implement this method.")