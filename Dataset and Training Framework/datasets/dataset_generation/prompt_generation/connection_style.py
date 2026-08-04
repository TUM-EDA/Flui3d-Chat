import random
import re
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
        # Generate the core textual description of the connections from the graph. This also records,
        # on the graph, the order the prose narrates each connection (`connection_reading_order`), so
        # the JSON converter lists `connections` in that order; it does not alter the prompt text.
        core_description, processed_graph, protected = self._generate_core_description(graph)

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

        # Restore the protected helper phrases (junction/type/port wording) now that the
        # renamer has run, so their module words are not re-synonymized.
        for token, phrase in protected.items():
            final_prompt = final_prompt.replace(token, phrase)

        # Capitalize the first letter of each sentence in the prompt for correct grammar.
        final_prompt = self._capitalize_sentences(final_prompt)

        # For this specific style, both the auxiliary LLM and direct-use outputs
        # use the same final prompt.
        prompts_for_llm = {"id": graph_id, "prompt": final_prompt}
        prompts_wo_llm = {"id": graph_id, "prompt": final_prompt}
        
        return prompts_for_llm, prompts_wo_llm
        

    @staticmethod
    def _join_and(items: List[str]) -> str:
        """Joins items into an English list: ``[a, b, c]`` -> ``"a, b and c"``."""
        items = list(items)
        if len(items) <= 1:
            return items[0] if items else ""
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f" and {items[-1]}"

    # Verb pools for split/merge fans, keeping the junction's direction in the prose.
    _SPLIT_VERBS = ["split", "branch", "divide"]
    _MERGE_VERBS = ["merge", "combine", "join", "pool"]

    DEFAULT_JUNCTION_TYPE = "T-junction"

    # In explicit mode the non-default Y is always surfaced; the default T's shape is
    # surfaced only sometimes. This is the per-T probability of stating it.
    _EXPLICIT_T_TYPE_PROB = 0.5

    def _tag_explicit_junctions(self, text: str, graph: nx.DiGraph) -> str:
        """Tags each junction's first mention with its T/Y shape, e.g. ``junction_1 (a Y-junction)``.

        Explicit mode only: junctions are narrated as named waypoints, so the shape rides
        inline rather than in a ``"via"`` phrase. The non-default Y is always stated, the
        default T only sometimes (``_EXPLICIT_T_TYPE_PROB``). Width is left to the
        surrounding ``"connect ... to ..."`` clause (how many things meet at the junction),
        so no fan/fork synonym is used. The hyphen in ``"<X>-junction"`` keeps the renamer's
        ``(?<!-)`` lookbehind from re-synonymizing the shape word.
        """
        tagged = set()

        def add_tag(match: re.Match) -> str:
            jid = match.group(0)
            if jid in tagged:
                return jid
            tagged.add(jid)
            jtype = graph.nodes[jid].get("type", "T-junction")
            if jtype == self.DEFAULT_JUNCTION_TYPE and random.random() >= self._EXPLICIT_T_TYPE_PROB:
                return jid
            return f"{jid} (a {jtype})"

        return re.sub(r"junction_\d+", add_tag, text)

    def _group_via(self, members: List[str], graph: nx.DiGraph) -> str:
        """Builds a grouped ``"via ..."`` phrase for the junctions of one chain.

        Identical junction descriptions are collapsed with a count
        (``"via 2 binary Y-junctions"``); mixed chains are listed
        (``"via a binary T-junction and a binary Y-junction"``). Widths use
        ``_describe_junction`` so multi-way synonyms (``"3-way fork"``) survive.
        """
        descs = [
            self._describe_junction(
                max(graph.in_degree(j), graph.out_degree(j)),
                graph.nodes[j].get("type", "T-junction"),
            )
            for j in members
        ]
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
        return "via " + self._join_and(parts)

    def _generate_core_description(self, graph: nx.DiGraph) -> Tuple[str, nx.DiGraph, Dict[str, str]]:
        """
        Builds the natural-language connection description for a chip graph.

        Two narration modes are chosen per prompt (~50/50):

        * **Explicit** -- junctions stay in the graph as named waypoints (counted and
          listed like any component) and the flow is walked through them step by step
          ("connect A and B to junction_1. connect junction_1 to junction_2 ..."). Each
          junction's T/Y shape is stated inline at most once, on its first mention: the
          non-default Y always, the default T only sometimes. No ``"via"`` and no width
          word -- the fan-out/-in is conveyed by how many things meet at the junction.
        * **Implicit** -- junctions are removed and each junction chain is flattened to
          its real endpoints. A grouped ``"via ..."`` marker is appended only to mark a
          non-default setup: an all-T chain stays silent, but if any Y is present the marker
          describes every junction in that chain (the Y *and* any co-occurring T), so a
          multi-junction fan is not reduced to a single "via a Y-junction". Only this mode
          uses ``"via"``.

        Only junctions of the SAME flow direction (split-trees / merge-trees) collapse into
        one implicit fan; mixing a split with a merge (a diamond) would put a component on
        both sides, so external endpoints are found by walking junction-only paths to the
        nearest components.

        Wording produced by the helpers (the DLD "particle outlet of" port label, the
        implicit ``"via ..."`` phrase) is wrapped in placeholder tokens so the later
        ``_replace_module_names`` pass cannot re-synonymize the module words inside them.
        Node IDs are left outside the tokens so they are still renamed/numbered. The caller
        restores the tokens afterwards via the returned map.
        """
        explicit_junctions = random.random() < 0.50
        G = graph

        protected: Dict[str, str] = {}

        def protect(phrase: str) -> str:
            token = f"\x00{len(protected)}\x00"
            protected[token] = phrase
            return token

        def is_junction(node: str) -> bool:
            return node.startswith("junction")

        def source_port(u: str, v: str) -> str:
            """Source-side port label for edge u->v (DLD filters expose smaller/larger outlets)."""
            ftype = (G.get_edge_data(u, v) or {}).get("filter_connection_type")
            return f"{protect(f'the {ftype} particle outlet of')} {u}" if ftype else u

        order = {node: i for i, node in enumerate(nx.lexicographical_topological_sort(G, key=str))}
        sentences: List[Tuple[int, str]] = []

        if explicit_junctions:
            # Junctions stay in the graph: counted, listed, and narrated as named waypoints,
            # so every edge (incl. junction-incident ones) is an ordinary edge below.
            processed_graph = G
            routed_edges = set(G.edges())
        else:
            # Junctions are hidden; only component-to-component edges are narrated directly,
            # and each junction chain is flattened to its real endpoints here.
            processed_graph = self._remove_junction_nodes(graph)
            routed_edges = {(u, v) for u, v in G.edges() if not is_junction(u) and not is_junction(v)}

            junctions = [n for n in G.nodes if is_junction(n)]

            def junction_dir(j: str) -> str:
                indeg, outdeg = G.in_degree(j), G.out_degree(j)
                if indeg == 1 and outdeg >= 2:
                    return "split"
                if outdeg == 1 and indeg >= 2:
                    return "merge"
                return "other"

            parent = {j: j for j in junctions}

            def find(x: str) -> str:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for u, v in G.edges():
                if (is_junction(u) and is_junction(v)
                        and junction_dir(u) == junction_dir(v) != "other"):
                    parent[find(u)] = find(v)

            chains = defaultdict(list)
            for j in junctions:
                chains[find(j)].append(j)

            def reach_targets(members: List[str]) -> List[str]:
                """Nearest non-junction nodes downstream of the chain (its real targets)."""
                result, seen_n, seen_j, stack = [], set(), set(members), list(members)
                while stack:
                    for x in G.successors(stack.pop()):
                        if is_junction(x):
                            if x not in seen_j:
                                seen_j.add(x); stack.append(x)
                        elif x not in seen_n:
                            seen_n.add(x); result.append(x)
                return sorted(result, key=lambda n: order[n])

            def reach_source_ports(members: List[str]) -> List[Tuple[str, str]]:
                """Real source ports upstream of the chain, as (node, filter_outlet) pairs.

                The outlet is read from the edge by which the source actually enters a
                junction, so a DLD filter feeding the chain through both outlets yields two
                distinct ports (one "smaller", one "larger") instead of being collapsed to a
                single, arbitrarily-labelled one. Ordinary sources carry a None outlet.
                """
                ports, seen_keys, seen_j, stack = [], set(), set(members), list(members)
                while stack:
                    j = stack.pop()
                    for u in G.predecessors(j):
                        if is_junction(u):
                            if u not in seen_j:
                                seen_j.add(u); stack.append(u)
                        else:
                            ftype = (G.get_edge_data(u, j) or {}).get("filter_connection_type")
                            if (u, ftype) not in seen_keys:
                                seen_keys.add((u, ftype)); ports.append((u, ftype))
                return sorted(ports, key=lambda pr: order[pr[0]])

            # Group chains that flatten to the SAME (sources, targets): when a merge junction
            # feeds directly into a split junction (a "diamond"), the merge-side chain and the
            # split-side chain reach identical external endpoints, so each would otherwise
            # narrate the very same routing sentence. Key on the token-free (src_ports, ext_tgt)
            # signature -- the `protect` tokens inside `srcs` differ per chain, so the raw ports
            # are the stable key -- and combine every grouped chain's junctions into one marker.
            chain_groups: Dict[Tuple, Dict] = {}
            for members in chains.values():
                members.sort(key=lambda n: order[n])
                src_ports = reach_source_ports(members)
                ext_tgt = reach_targets(members)
                if not src_ports or not ext_tgt:
                    continue
                key = (tuple(src_ports), tuple(ext_tgt))
                grp = chain_groups.get(key)
                if grp is None:
                    chain_groups[key] = {"src_ports": src_ports, "ext_tgt": ext_tgt,
                                         "members": list(members)}
                else:
                    grp["members"].extend(members)

            for grp in chain_groups.values():
                src_ports, ext_tgt, members = grp["src_ports"], grp["ext_tgt"], grp["members"]

                # Each source keeps the DLD outlet label of the edge that enters the chain,
                # so a filter feeding through both outlets is named once per outlet.
                srcs = [f"{protect(f'the {ft} particle outlet of')} {u}" if ft else u
                        for u, ft in src_ports]
                if len(src_ports) == 1 and len(ext_tgt) >= 2:
                    verb = random.choice([None] + self._SPLIT_VERBS)
                    routing = (f"connect {srcs[0]} to {self._join_and(ext_tgt)}" if verb is None
                               else f"{verb} {srcs[0]} into {self._join_and(ext_tgt)}")
                elif len(src_ports) >= 2 and len(ext_tgt) == 1:
                    verb = random.choice([None] + self._MERGE_VERBS)
                    routing = (f"connect {self._join_and(srcs)} to {ext_tgt[0]}" if verb is None
                               else f"{verb} {self._join_and(srcs)} into {ext_tgt[0]}")
                else:
                    routing = f"connect {self._join_and(srcs)} to {self._join_and(ext_tgt)}"

                # "via ..." marks a non-default setup only: an all-T chain stays silent (T is
                # the default, and the fan-in/-out is already conveyed by how many endpoints the
                # clause lists). But once any non-default Y forces the marker to appear, describe
                # EVERY junction across the grouped chain(s) -- the Y(s) AND any co-occurring
                # (default) T -- so a multi-junction fan is not misleadingly reduced to "via a
                # Y-junction" when more than one junction actually does the merge/split. The
                # marker is protected from the later renamer.
                if any(G.nodes[j].get("type") != self.DEFAULT_JUNCTION_TYPE for j in members):
                    routing = f"{routing} {protect(self._group_via(members, G))}"

                # Order by the LAST source to appear, so the merge/split is narrated only
                # once every component it consumes has been introduced (e.g. "connect t1 to
                # t2" precedes "pool ... and t2 into t3"). The sentence carries the junction-free
                # connections it states (every source to every target of this flattened fan), so the
                # JSON can be listed in narration order.
                grp_edges = [(u, t) for u, _ in src_ports for t in ext_tgt]
                sentences.append((max(order[u] for u, _ in src_ports), routing, grp_edges))

        # --- Edges narrated directly (all edges when explicit, plain edges when implicit). ---
        emitted_edges = set()

        for v in sorted(G.nodes, key=lambda n: order[n]):
            ins = [u for u in G.predecessors(v) if (u, v) in routed_edges]
            if len(ins) > 1:
                ins_sorted = sorted(ins, key=lambda n: order[n])
                srcs = self._join_and([source_port(u, v) for u in ins_sorted])
                # Keyed by the last input introduced (see junction-chain note above).
                sentences.append((max(order[u] for u in ins), f"connect {srcs} to {v}",
                                  [(u, v) for u in ins_sorted]))
                emitted_edges.update((u, v) for u in ins)

        for u in sorted(G.nodes, key=lambda n: order[n]):
            outs = [v for v in G.successors(u) if (u, v) in routed_edges and (u, v) not in emitted_edges]
            if not outs:
                continue
            outs.sort(key=lambda n: order[n])
            by_port = defaultdict(list)
            for v in outs:
                by_port[source_port(u, v)].append(v)
            clauses = [f"{port} to {self._join_and(tgts)}" for port, tgts in by_port.items()]
            sentences.append((order[u], "connect " + " and ".join(clauses), [(u, v) for v in outs]))
            emitted_edges.update((u, v) for v in outs)

        sentences.sort(key=lambda item: item[0])
        description_text = ". ".join(text for _, text, _ in sentences)

        # Record the order the prose narrates each junction-free connection, so the JSON converter can
        # list `connections` in that order (see JsonConverter.connection_reading_order). Each sentence
        # carries the connections it states, in mention order.
        self._record_connection_order(G, explicit_junctions,
                                      [e for _, _, edges in sentences for e in edges])

        # In explicit mode, surface each junction's T/Y shape inline, once, at first mention.
        if explicit_junctions:
            description_text = self._tag_explicit_junctions(description_text, G)

        # Component types are conveyed uniformly via _generate_structural_descriptions (as a
        # qualifier on the component's single rendered name), not woven into the connections here.
        return description_text, processed_graph, protected

    def _record_connection_order(self, G: nx.DiGraph, explicit_junctions: bool,
                                 narrated: List[Tuple[str, str]]) -> None:
        """Store on ``G`` the order the prose narrates each junction-free connection, as a list of
        (source, target) node pairs in ``G.graph["connection_reading_order"]``.

        ``narrated`` is the edges the sentences state, in narration order. In IMPLICIT mode those are
        already the junction-free connections (junction chains were flattened to their endpoints), so
        their first-appearance order is taken directly. In EXPLICIT mode the prose walks through the
        junctions, so ``narrated`` holds raw component->junction->component edges; each junction-free
        connection (a, c) is then ranked by where its path's FIRST hop out of ``a`` and LAST hop into
        ``c`` are narrated -- which keeps, e.g., the inputs of one merge together and in stated order.
        Per-node ordering cannot express this (a node introduced early may have its out-edges narrated
        much later). Only affects JSON ordering; the prompt text is untouched.
        """
        if explicit_junctions:
            raw_idx: Dict[Tuple[str, str], int] = {}
            for edge in narrated:
                raw_idx.setdefault(edge, len(raw_idx))
            unranked = len(raw_idx)
            # A flattened connection a->c contracts a chain whose intermediates are ALL junctions
            # (a path through another component would be two separate connections), so resolve its
            # hops on the subgraph that keeps only a, c and the junctions -- otherwise a shortest
            # path could detour through a component and pick a hop belonging to a different edge.
            junctions = [n for n in G if str(n).startswith("junction")]
            ranked = []
            seen = set()
            for a, c in self._remove_junction_nodes(G).edges():
                if a.startswith("junction") or c.startswith("junction") or (a, c) in seen:
                    continue
                seen.add((a, c))
                path = nx.shortest_path(G.subgraph([a, c] + junctions), a, c)
                first_hop, last_hop = (path[0], path[1]), (path[-2], path[-1])
                ranked.append(((raw_idx.get(first_hop, unranked), raw_idx.get(last_hop, unranked)), (a, c)))
            ranked.sort(key=lambda item: item[0])
            order = [ac for _, ac in ranked]
        else:
            order, seen = [], set()
            for edge in narrated:
                if edge not in seen:
                    seen.add(edge)
                    order.append(edge)
        G.graph["connection_reading_order"] = order