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

    # Sentinel for "this component category/type has no registered default for this attribute".
    _UNKNOWN = object()

    # A dictionary of synonyms for different microfluidic components, keyed by ID prefix.
    # This is used to create more varied prompts. Type-conditional wording (ring vs serpentine
    # mixer, DLD vs pillar-matrix filter, ...) lives in TYPE_SYNONYMS instead, since those
    # variants share an ID prefix and can only be distinguished via the node's `type` attribute.
    MODULE_SYNONYMS = {
        "inlet": ["inlet", "fluid inlet", "entry point", "input port", "fluid entry", "input"],
        "outlet": ["outlet", "fluid outlet", "exit point", "output port", "fluid exit", "output"],
        "chamber": ["chamber", "reaction chamber", "microchamber", "reaction vessel", "reaction unit"],
        "mixer": ["mixer", "mixing unit", "mixing channel", "mixing microchannel", "micromixer", "blender"],
        "delay": ["delay", "serpentine", "serpentine channel", "twisted channel", "winding path", "curved channel", "curved microchannel", "serpentine microchannel", "twisted microchannel", "delay channel", "delay microchannel", "delaying channel", "delaying microchannel"],
        "filter": ["filter", "particle filter", "separation filter", "microfilter"],
        "droplet": ["droplet generator", "microdroplet generator"],
        "tesla_valve": ["tesla valve", "tesla diode", "fluidic diode", "non-return valve", "passive valve"],
        "junction": ["junction", "intersection", "crossing"],
    }

    # Type-conditional qualifiers, keyed by (ID prefix, `type`). These compose with the neutral
    # MODULE_SYNONYMS noun for the same category (e.g. "ring" + "blender" -> "ring blender"),
    # so the type is always conveyed in prose without contradicting the JSON `type` field.
    # Consulted by builders that have access to the node's `type` attribute.
    TYPE_QUALIFIERS = {
        ("mixer", "serpentine"): ["serpentine", "winding", "twisted"],
        ("mixer", "ring"): ["ring", "circular", "looped", "toroidal", "ring-shaped"],
        ("filter", "dld"): ["DLD", "size-separation", "particle-separation", "deterministic lateral displacement"],
        ("filter", "pillar_matrix"): ["pillar-matrix", "pillar-array", "microstructured", "sieving"],
        ("droplet", "t_junction"): ["T-junction"],
        ("droplet", "flow_focusing"): ["flow-focusing", "hydrodynamic-focusing"],
    }

    # Per-category default type; this type may be left unstated in prompts because it is the
    # canonical choice. Non-default types (e.g. ring mixer) are always stated. Filters are not
    # listed here: their type is always optional because the output-port count already encodes it.
    DEFAULT_TYPE = {"mixer": "serpentine", "droplet": "t_junction"}

    # Noun pool for multi-way (width > 2) junctions. Multi-way junctions are always T-junctions
    # (Y-junctions are binary-only), so the noun is a shape-neutral or T-shaped width descriptor --
    # never a "Y" word, which would misname a T-junction. The width is composed in by
    # _describe_junction (e.g. "3-way fan"); binary junctions use the literal
    # "T-junction" / "Y-junction" wording instead.
    JUNCTION_WIDTH_SYNONYMS = ["fan", "fork", "T-fork", "T-junction"]

    # Canonical junction shape; like DEFAULT_TYPE for components, a lone junction of this shape
    # may be left unnamed in prose (the operation already implies a plain junction), while the
    # non-default shape is stated so the JSON `type` stays recoverable. T-junctions are the
    # canonical microfluidic junction (cf. the t_junction droplet default).
    DEFAULT_JUNCTION_SHAPE = "T-junction"

    # Default parameter values per (ID prefix, `type`). A parameter sitting at its default is
    # suppressed from the prompt prose; only non-default (e.g. randomized) values are surfaced.
    # Keyed by type because length_um's default differs across categories (chamber/DLD/pillar).
    # Entries with type None apply to categories that have no subtype.
    DEFAULT_ATTRIBUTES = {
        ("chamber", None): {"length_um": 4000, "width_um": 3200},
        ("mixer", "serpentine"): {"num_turnings": 4, "amplitude_um": 2000, "distance_between_turnings_um": 200},
        ("mixer", "ring"): {"diameter_um": 1000, "num_circles": 3, "distance_between_circles_um": 200},
        ("delay", "serpentine"): {"num_turnings": 4, "amplitude_um": 2000, "distance_between_turnings_um": 200},
        ("filter", "dld"): {"length_um": 4000, "width_um": 3200, "post_shape": "circle", "post_diameter_um": 50, "row_shift_fraction": 0.20, "critical_particle_diameter_um": 10},
        ("filter", "pillar_matrix"): {"length_um": 4000, "width_um": 3200, "post_shape": "circle", "post_diameter_um": 400, "columns": 3, "rows": 4},
        ("tesla_valve", None): {"num_segment_pairs": 2, "segment_length_um": 1000, "segment_width_um": 600},
        ("droplet", "t_junction"): {"nozzle_width_um": 100},
        ("droplet", "flow_focusing"): {"nozzle_width_um": 100},
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

    @staticmethod
    def _node_type(node: str) -> str:
        """Returns a node's category prefix, e.g. ``tesla_valve_3`` -> ``tesla_valve``.

        Splitting on the first underscore is wrong for multi-word prefixes like
        ``tesla_valve``, so the trailing ``_<number>`` is stripped instead.
        """
        return re.sub(r"_\d+$", "", node)

    def _describe_junction(self, width: int, shape: str) -> str:
        """Phrases a junction for prose, e.g. ``"a 3-way fork"`` or ``"a binary T-junction"``.

        Args:
            width: The number of branches (fan-out for a split, fan-in for a merge).
                   A binary junction has width 2; multi-way junctions have width 3-5.
            shape: The junction's geometric ``type`` (``"T-junction"`` or ``"Y-junction"``).
                   Multi-way junctions are always ``"T-junction"`` (Y-junctions are binary-only),
                   so ``shape`` only steers the wording in the binary branch.

        Returns:
            A noun phrase (including its article) describing the junction.
        """
        if width <= 2:
            return random.choice([f"a {shape}", f"a binary {shape}"])
        noun = random.choice(self.JUNCTION_WIDTH_SYNONYMS)
        return f"a {width}-way {noun}"

    def _describe_component(self, category: str, ctype: str = None) -> str:
        """Phrases a component for prose, composing a type qualifier with a neutral noun.

        For typed categories the qualifier (e.g. ``ring``) is combined with a neutral
        noun from MODULE_SYNONYMS (e.g. ``blender``) to yield ``"ring blender"``, so the
        type is always conveyed without contradicting the JSON ``type`` field. Untyped
        categories fall back to a plain neutral synonym.

        Args:
            category: The component ID prefix (e.g. ``"mixer"``, ``"filter"``).
            ctype: The node's ``type`` attribute, if any (e.g. ``"ring"``, ``"dld"``).

        Returns:
            A noun phrase describing the component.
        """
        nouns = self.MODULE_SYNONYMS.get(category, [category.replace("_", " ")])
        qualifiers = self.TYPE_QUALIFIERS.get((category, ctype))
        if qualifiers:
            return f"{random.choice(qualifiers)} {random.choice(nouns)}"
        return random.choice(nouns)

    def _get_module_counts(self, graph: nx.DiGraph) -> Dict[str, int]:
        """Counts the number of modules of each type in the graph."""
        module_counts = defaultdict(int)
        for node in graph.nodes():
            node_type = self._node_type(node)
            if node_type in self.MODULE_SYNONYMS:
                module_counts[node_type] += 1
        return dict(module_counts)

    def _replace_module_names(self, version_text: str, module_counts: Dict[str, int],
                              selected_synonyms: Dict[str, str], schema: int = 0) -> str:
        """Replaces module names in text according to a specified schema."""
        if schema == 0:
            return version_text.replace("drops_replace", "droplets")

        def get_replacement(module_type: str, module_id: int) -> str:
            display = module_type.replace("_", " ")
            if schema == 1:
                ordinal_map = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
                return f"{ordinal_map[module_id - 1]} {display}" if module_id <= len(ordinal_map) else f"{module_id}th {display}"
            elif schema == 2:
                return f"{display} {module_id}"
            elif schema == 3:
                return f"{display} {chr(64 + module_id)}"


        def replace_suffix(match: re.Match) -> str:
            module_type, module_id = match.group(1), int(match.group(2))
            if (module_counts.get(module_type) or 0) > 1 or random.random() < 0.5:
                return get_replacement(module_type, module_id)
            return module_type.replace("_", " ")

        def replace_module_without_suffix(match: re.Match) -> str:
            module_type = match.group(1)
            return selected_synonyms.get(module_type, module_type.replace("_", " "))

        # tesla_valve must precede the single-word alternatives so the longer token wins.
        pattern_with_id = r"(tesla_valve|inlet|outlet|chamber|mixer|delay|filter|droplet|junction)_(\d+)"
        version_text = re.sub(pattern_with_id, replace_suffix, version_text)

        # The negative lookbehind keeps the fixed nomenclature "T-junction"/"Y-junction" intact:
        # their "junction" must not be synonymized to "crossing"/"intersection".
        pattern_without_id = r"(?<!-)(tesla_valve|inlet|outlet|chamber|mixer|delay|filter|droplet|junction)"
        version_text = re.sub(pattern_without_id, replace_module_without_suffix, version_text)

        return version_text.replace("drops_replace", "droplets")



class StructuralPromptGenerator(PromptGenerator):
    """
    A generator for prompts that describe the structure of the chip,
    such as connections or paths. This class contains shared logic for
    connection- and path-oriented styles.
    """

    # A dictionary of synonyms. Mixer terms are type-neutral (ring vs serpentine is conveyed
    # via TYPE_QUALIFIERS); delay terms may stay serpentine-flavored since all delays are serpentine.
    MODULE_SYNONYMS = {
        "inlet": ["inlet", "fluid inlet", "entry point", "input port", "fluid entry", "input"],
        "outlet": ["outlet", "fluid outlet", "exit point", "output port", "fluid exit", "output"],
        "chamber": ["chamber", "reaction chamber", "microchamber", "reaction vessel", "reaction unit"],
        "mixer": ["mixer", "mixing unit", "mixing channel", "mixing microchannel", "micromixer", "blender"],
        "delay": ["delay", "delaying serpentine", "delaying serpentine channel", "delaying twisted channel", "delaying winding path", "delaying curved channel", "delaying curved microchannel", "delaying serpentine microchannel", "delaying twisted microchannel", "delay channel", "delay microchannel", "delaying channel", "delaying microchannel"],
        "filter": ["filter", "particle filter", "microfilter"],
        "droplet": ["droplet generator", "microdroplet generator"],
        "tesla_valve": ["tesla valve", "tesla diode", "fluidic diode", "non-return valve", "passive valve"],
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
        type_tags = {}
        param_attrs = {}
        for node, data in graph.nodes(data=True):
            node_type = self._node_type(node)
            if node_type in self.MODULE_SYNONYMS:
                node_types[node_type].append(node)

            subtype = data.get("type")
            defaults = self.DEFAULT_ATTRIBUTES.get(
                (node_type, subtype), self.DEFAULT_ATTRIBUTES.get((node_type, None), {}))

            # Optionally state the component's type as a leading qualifier (e.g. "(pillar-array)"),
            # so it rides the same per-version slots as params and the component is named only once.
            # Default types are stated only occasionally (serpentine mixer, t_junction droplet);
            # non-default types are always stated; filter type is optional since the port count
            # already encodes it.
            qualifiers = self.TYPE_QUALIFIERS.get((node_type, subtype))
            if qualifiers:
                if node_type in self.DEFAULT_TYPE:
                    state = subtype != self.DEFAULT_TYPE[node_type] or random.random() < 0.18
                else:  # filter
                    state = random.random() < 0.5
                if state:
                    type_tags[node] = random.choice(qualifiers)

            params = []
            for attr, value in data.items():
                # `function` is structural; a component's `type` is conveyed via the qualifier
                # tag above. A junction's `type` is its T/Y shape and is kept as a normal param.
                # `reading_index` is narration-order bookkeeping used only to order the JSON
                # connections -- it is not a component parameter and must never be surfaced.
                if attr in ('function', 'reading_index'):
                    continue
                if attr == 'type':
                    # A component's `type` rides its qualifier tag; a junction's T/Y shape
                    # is stated inline in the connection narration (Style A, explicit mode),
                    # so neither is repeated as a "type:" param here.
                    continue

                # v2 keys already encode units (e.g. length_um); strip the suffix for the
                # display name and append the µm unit, giving "length: 4000 µm".
                if attr.endswith('_um'):
                    display, unit = attr[:-3].replace('_', ' '), " µm"
                else:
                    display, unit = attr.replace('_', ' '), ""

                default = defaults.get(attr, self._UNKNOWN)
                if default is self._UNKNOWN:
                    if random.random() < 0.5:
                        params.append(f"{display}: {value}{unit}")
                elif value != default:
                    params.append(f"{display}: {value}{unit}")

            param_attrs[node] = params
            parts = ([type_tags[node]] if node in type_tags else []) + params
            if parts:
                attributes_text[node] = f" ({', '.join(parts)})"

        prefix_counts = ", ".join(
            f"{len(modules)} {module_type.replace('_', ' ')}{'s' if len(modules) > 1 else ''}"
            for module_type, modules in sorted(node_types.items())
        )

        prefix_detailed = "\n".join(
            f"{len(modules)} {module_type.replace('_', ' ')}{'s' if len(modules) > 1 else ''}:\n" +
            ", ".join(f"{node}{attributes_text.get(node, '')}" for node in modules)
            for module_type, modules in sorted(node_types.items())
        )

        # Suffix sentences for the versions that route attributes to the end instead of inline.
        # Derived from the same per-node data as the inline path, so every v2 parameter is
        # surfaced (the old hand-written whitelist silently dropped ring/tesla/pillar params).
        # The type qualifier is phrased as "... of the <qualifier> type" rather than appending a
        # category noun, which previously produced "droplet generator generator" once the
        # renamer expanded the bare module word inside the appended noun.
        suffix_sentences = []
        for node in attributes_text:
            node_type = self._node_type(node)
            # A junction's T/Y shape is conveyed by the connection narration, so it gets no
            # redundant suffix line (it stays available in the detailed component list, v5).
            if node_type == 'junction':
                continue
            if node in type_tags:
                suffix_sentences.append(f"{node} should be of the {type_tags[node]} type")
            if param_attrs.get(node):
                suffix_sentences.append(f"{node} should have {', '.join(param_attrs[node])}")

        return prefix_counts, prefix_detailed, attributes_text, ". ".join(suffix_sentences)


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