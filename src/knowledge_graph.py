"""
Phase 4 - the knowledge graph. This is MEMORY 2, and the reason this project exists.

THE MENTAL MODEL
----------------
A road map. Every entity is a city. Every relation is a one-way road between two
cities, and the road has a name:

    DETR ──[outperforms]──> Faster R-CNN ──[proposed by]──> Ren et al.

Vector search (Memory 1) can only ask "which city looks most like this
description?". A graph can be WALKED: start at a city, follow roads, arrive
somewhere you could never have described in advance. That walk is what answers
multi-hop questions.

WHY DIRECTED, AND WHY IT MATTERS SO MUCH
----------------------------------------
    (DETR, outperforms, Faster R-CNN)  is TRUE
    (Faster R-CNN, outperforms, DETR)  is FALSE

Same three words, opposite claims. An undirected graph cannot tell them apart,
so it would happily answer "which models beat DETR?" with "Faster R-CNN" - a
confident, cited, exactly-backwards answer. Direction is not a technicality; it
carries half the meaning of the edge.

WHY MultiDiGraph AND NOT DiGraph
--------------------------------
Two entities can be related in several distinct ways, each stated in a different
paper:

    Mask R-CNN --[extends]--> Faster R-CNN     (mask_rcnn.pdf, p.1)
    Mask R-CNN --[based on]--> Faster R-CNN    (detr.pdf, p.3)

A plain DiGraph stores at most one edge per ordered pair, so the second fact
would silently overwrite the first and we would lose both a relation type and a
citation. MultiDiGraph keeps every edge with its own provenance - which is
exactly what Phase 6 needs to cite graph-derived claims.

WHAT "HOPS" MEANS
-----------------
The number of edges you traverse.

    0 hops: DETR
    1 hop:  DETR --[outperforms]--> Faster R-CNN
    2 hops: DETR --[outperforms]--> Faster R-CNN --[proposed by]--> Ren et al.
    3 hops: ... --[worked on]--> instance segmentation

At 1 hop you learn what a paper directly says. At 2-3 hops you learn things NO
paper says - facts that exist only in the combination of several papers. That is
the entire value proposition, and it is unreachable by any amount of text search.
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import GRAPH_PATH, banner, ensure_dirs, setup_console  # noqa: E402

# Fuzzy-matching threshold (0-100) for MERGING two entity strings into one node.
# Deliberately strict, and deliberately measured with fuzz.ratio rather than
# fuzz.WRatio - see the long note in _canonicalise_entities.
MERGE_THRESHOLD = 90

# Entity strings that are grammatically fine but useless as graph nodes.
_STOP_ENTITIES = {
    "it", "we", "they", "this", "that", "these", "those", "he", "she", "i",
    "the", "a", "an", "our", "their", "its", "which", "who", "what",
    "one", "two", "three", "both", "all", "each", "such", "other", "others",
    "table", "figure", "fig", "section", "sec", "eq", "equation", "paper",
    "method", "approach", "model", "models", "result", "results", "work",
    "example", "case", "time", "number", "order", "way", "use", "using",
}

_VERSION_RE = re.compile(r"(v?\d+(?:\.\d+)?)\s*$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Entity normalisation
# ---------------------------------------------------------------------------

def normalise_entity(text: str) -> str:
    """
    Reduce a surface form to a comparison key.

    "the DETR model" / "DETR" / "detr." all have to collide, or the graph ends up
    with three disconnected nodes for one thing and no path runs through any of
    them. This is the cheap, exact half of deduplication; fuzzy matching handles
    the rest.
    """
    text = text.strip().lower()
    text = re.sub(r"^(the|a|an|our|their|its|this|these)\s+", "", text)
    text = re.sub(r"[\"'`()\[\]{}]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,;:-")


def is_valid_entity(text: str) -> bool:
    """Reject entity strings that would only add noise to the graph."""
    normalised = normalise_entity(text)
    if len(normalised) < 2 or len(normalised) > 80:
        return False
    if normalised in _STOP_ENTITIES:
        return False
    if not any(c.isalpha() for c in normalised):
        return False               # pure numbers: "37.4", "2015"
    # Mostly digits with a stray letter is a table cell, not an entity.
    if sum(c.isdigit() for c in normalised) > len(normalised) * 0.5:
        return False
    if len(normalised.split()) > 8:
        return False               # a sentence fragment, not an entity

    # Maths notation leaking out of equations: "dk", "L L", "sqrt-dk", "pepos+k".
    # These reached the graph as real nodes with 100+ edges each.
    tokens = normalised.split()
    if "et al" not in normalised and all(len(token) <= 2 for token in tokens):
        # The "et al" exemption is load-bearing, not a nicety: "he et al" is
        # three two-letter tokens, so the bare rule silently deleted author
        # citations - the single entity type the authorship questions depend on.
        return False               # "L L", "dk", "h t"
    if any(char in normalised for char in "√∑∏∫≤≥≈∈⊂×÷±"):
        return False
    if len(set(tokens)) == 1 and len(tokens) > 1:
        return False               # "layer layer"

    return True


def _version_stem(normalised: str) -> tuple[str, str | None]:
    """Split 'yolov3' into ('yolo', 'v3'); returns (stem, version or None)."""
    match = _VERSION_RE.search(normalised)
    if not match:
        return normalised, None
    return normalised[: match.start()].strip(" -"), match.group(1)


_INITIALS_RE = re.compile(r"^(?:[a-z]\.?\s+)+")


def _strip_initials(normalised: str) -> str:
    """
    'a. vaswani' -> 'vaswani',  'k. he' -> 'he'.

    Author names are the one case where a pure similarity score reliably fails:
    "A. Vaswani" and "Vaswani" are unambiguously one person, but they share only
    73% of their characters, well under any threshold safe enough to use
    generally. Handling initials as an explicit rule lets the similarity
    threshold stay strict everywhere else.
    """
    return _INITIALS_RE.sub("", normalised).strip()


def _canonicalise_entities(counts: Counter) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """
    Decide which entity strings are the SAME thing and which are merely RELATED.

    Returns (mapping, variant_pairs):
      mapping       normalised form -> canonical normalised form
      variant_pairs (canonical_a, canonical_b) to be joined by a "variant_of" edge

    THE DISTINCTION THAT MATTERS
    PROJECT_CONTEXT.md asks for two things in one breath: "'YOLO' and 'YOLOv3'
    should be connected" and "'Vaswani' and 'A. Vaswani' should merge". Those are
    different operations and conflating them corrupts the graph:

      MERGE   "Vaswani" and "A. Vaswani" are one person written two ways. One node.
      CONNECT "YOLO" and "YOLOv3" are DIFFERENT MODELS with different accuracy
              and speed. Merging them would make "which models outperform YOLOv3?"
              return YOLOv3's own results. They deserve separate nodes joined by
              an explicit variant_of edge, so a traversal can still walk between
              them when that is useful.

    So: high similarity AND no conflicting version number -> merge.
        high similarity BUT different version numbers      -> link as variants.

    We canonicalise toward the MOST FREQUENT surface form, on the reasoning that
    the spelling a corpus uses most often is the one a user is most likely to type.

    WHY fuzz.ratio AND NOT fuzz.WRatio
    ----------------------------------
    The first version of this used WRatio, which is rapidfuzz's "smart" scorer -
    it blends several strategies including partial_ratio, so a SUBSTRING scores
    very highly. That is right for search and badly wrong for entity resolution,
    because containment is not identity. It produced 2,257 bogus variant links:

        'layer'   <-> '101-layer residual net'
        'object'  <-> 'a failure case with overlapping objects'
        'scale'   <-> 'multi-scale feature extraction'
        'cnn'     <-> 'cnn activations'

    41% of all edges in the graph were that noise. fuzz.ratio is plain edit
    distance over the whole string, so a short string inside a long one scores
    low, exactly as it should.

    NOTE the deliberate asymmetry with query time: building the graph is STRICT
    (a wrong merge corrupts the data permanently), while find_entity() below is
    LENIENT and still uses WRatio (a user typing "Faster RCNN" must reach the
    node, and a wrong match there costs one bad search, not a corrupted graph).
    """
    from rapidfuzz import fuzz, process

    # Most frequent first: earlier entries become the canonical forms.
    ordered = [entity for entity, _ in counts.most_common()]

    canonical_forms: list[str] = []
    mapping: dict[str, str] = {}
    variant_pairs: set[tuple[str, str]] = set()

    # Index by initials-stripped form so author-name variants can be caught
    # without loosening the similarity threshold for everything else.
    by_stripped: dict[str, str] = {}

    for entity in ordered:
        stripped = _strip_initials(entity) or entity

        # --- rule 1: author initials ("a. vaswani" == "vaswani") ----------
        # Order-independent on purpose: whichever spelling the corpus uses more
        # often is seen first and becomes canonical, and the other maps onto it.
        # An earlier version required the entity to BE the initialled form, so
        # the merge only happened in one of the two possible orderings.
        if stripped in by_stripped and by_stripped[stripped] != entity:
            mapping[entity] = by_stripped[stripped]
            continue

        if not canonical_forms:
            canonical_forms.append(entity)
            mapping[entity] = entity
            by_stripped.setdefault(stripped, entity)
            continue

        # --- rule 2: near-identical strings --------------------------------
        # rapidfuzz is the C-speed backend; this is the hot loop and a
        # pure-Python matcher would take minutes over thousands of entities.
        match = process.extractOne(
            entity, canonical_forms, scorer=fuzz.ratio, score_cutoff=MERGE_THRESHOLD
        )

        if match is not None:
            candidate = match[0]
            entity_stem, entity_version = _version_stem(entity)
            candidate_stem, candidate_version = _version_stem(candidate)

            versions_conflict = (
                entity_version is not None
                and candidate_version is not None
                and entity_version != candidate_version
            ) or (
                # "yolo" vs "yolov3": one carries a version, the other does not,
                # and the stems match. Distinct models, not spellings.
                (entity_version is None) != (candidate_version is None)
                and entity_stem == candidate_stem
            )

            if not versions_conflict:
                mapping[entity] = candidate
                continue

        # --- not merged: its own node -------------------------------------
        canonical_forms.append(entity)
        mapping[entity] = entity
        by_stripped.setdefault(stripped, entity)

    # --- rule 3: variant links, ONLY for genuine version differences -------
    # This is the narrow case the link was invented for: YOLO/YOLOv3,
    # ResNet-50/ResNet-101, CIFAR-10/CIFAR-100 - different things a reader would
    # still want to traverse between. Anything broader floods the graph (an
    # earlier similarity-based rule produced 2,257 links, 41% of all edges).
    #
    # Done as a POST-PASS over the finished canonical list rather than inside the
    # loop: linking as we go only caught pairs where the versioned form happened
    # to be processed second, so "yolov3" before "yolo" silently produced no link.
    by_stem: dict[str, list[tuple[str, str | None]]] = {}
    for form in canonical_forms:
        stem, version = _version_stem(form)
        if stem:
            by_stem.setdefault(stem, []).append((form, version))

    for group in by_stem.values():
        if len(group) < 2:
            continue
        for i, (left, left_version) in enumerate(group):
            for right, right_version in group[i + 1:]:
                if left_version != right_version:
                    variant_pairs.add(tuple(sorted((left, right))))

    return mapping, sorted(variant_pairs)


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------

class KnowledgeGraph:
    """A directed multigraph of entities and the relations between them."""

    def __init__(self, graph: nx.MultiDiGraph | None = None) -> None:
        self.graph = graph if graph is not None else nx.MultiDiGraph()

    # -- construction ------------------------------------------------------

    @classmethod
    def from_triples(cls, triples: list[dict], verbose: bool = True) -> "KnowledgeGraph":
        """Build a graph from the triple records produced by knowledge_extractor.py."""
        instance = cls()
        instance.build(triples, verbose=verbose)
        return instance

    def build(self, triples: list[dict], verbose: bool = True) -> None:
        # Pass 1: count every valid entity surface form so canonicalisation can
        # prefer the most common spelling.
        counts: Counter = Counter()
        usable: list[dict] = []
        for triple in triples:
            if not (is_valid_entity(triple["subject"]) and is_valid_entity(triple["object"])):
                continue
            usable.append(triple)
            counts[normalise_entity(triple["subject"])] += 1
            counts[normalise_entity(triple["object"])] += 1

        if verbose:
            dropped = len(triples) - len(usable)
            print(f"  {len(usable):,} usable triples ({dropped:,} dropped: "
                  f"stopword/numeric/over-long entities)")
            print(f"  {len(counts):,} distinct entity strings -> deduplicating ...")

        mapping, variant_pairs = _canonicalise_entities(counts)
        canonical_count = len(set(mapping.values()))
        if verbose:
            print(f"  {canonical_count:,} canonical entities after merging "
                  f"({len(counts) - canonical_count:,} merged away)")
            print(f"  {len(variant_pairs):,} variant_of links (e.g. YOLO <-> YOLOv3)")

        # Display names: the most frequent ORIGINAL spelling for each canonical key.
        display_counts: dict[str, Counter] = {}
        for triple in usable:
            for role in ("subject", "object"):
                original = triple[role].strip()
                canonical = mapping[normalise_entity(original)]
                display_counts.setdefault(canonical, Counter())[original] += 1

        # Pass 2: add nodes and edges.
        for triple in usable:
            subject_key = mapping[normalise_entity(triple["subject"])]
            object_key = mapping[normalise_entity(triple["object"])]
            if subject_key == object_key:
                continue                      # self-loops carry no information

            for key, original in ((subject_key, triple["subject"]),
                                  (object_key, triple["object"])):
                if key not in self.graph:
                    self.graph.add_node(
                        key,
                        name=display_counts[key].most_common(1)[0][0],
                        aliases=set(),
                        mentions=0,
                        sources=set(),
                    )
                node = self.graph.nodes[key]
                node["aliases"].add(original.strip())
                node["mentions"] += 1
                node["sources"].add(triple["source_file"])

            self.graph.add_edge(
                subject_key,
                object_key,
                relation=triple["relation"],
                source_file=triple["source_file"],
                page=triple.get("page", 1),
                sentence=triple.get("sentence", ""),
                extractor=triple.get("extractor", "unknown"),
            )

        # Variant links, added only between entities that actually made it in.
        for left, right in variant_pairs:
            if left in self.graph and right in self.graph:
                self.graph.add_edge(left, right, relation="variant_of",
                                    source_file="(entity resolution)", page=0,
                                    sentence="", extractor="dedup")
                self.graph.add_edge(right, left, relation="variant_of",
                                    source_file="(entity resolution)", page=0,
                                    sentence="", extractor="dedup")

    # -- lookup ------------------------------------------------------------

    def find_entity(self, query: str, threshold: int = 80) -> str | None:
        """
        Resolve a user-typed string to a node key: exact first, then fuzzy.

        Phase 5 depends on this. A user asks about "Faster RCNN" and the node is
        keyed "faster r-cnn"; without fuzzy fallback the traversal starts nowhere
        and the graph contributes nothing to the answer.
        """
        normalised = normalise_entity(query)
        if normalised in self.graph:
            return normalised

        # Any alias of any node.
        for node_key, data in self.graph.nodes(data=True):
            if any(normalise_entity(alias) == normalised for alias in data.get("aliases", ())):
                return node_key

        from rapidfuzz import fuzz, process
        match = process.extractOne(
            normalised, list(self.graph.nodes), scorer=fuzz.WRatio, score_cutoff=threshold
        )
        return match[0] if match else None

    def find_entities(self, query: str, limit: int = 5, threshold: int = 75) -> list[tuple[str, float]]:
        """Return several candidate nodes with scores - used for query expansion."""
        from rapidfuzz import fuzz, process
        normalised = normalise_entity(query)
        matches = process.extract(
            normalised, list(self.graph.nodes), scorer=fuzz.WRatio,
            limit=limit, score_cutoff=threshold,
        )
        return [(name, score) for name, score, _ in matches]

    def display_name(self, node_key: str) -> str:
        return self.graph.nodes[node_key].get("name", node_key) if node_key in self.graph else node_key

    # -- traversal ---------------------------------------------------------

    def get_neighbors(self, entity: str, hops: int = 1, max_results: int = 200) -> list[dict]:
        """
        Every node reachable within `hops` edges, with the PATH taken to reach it.

        The path is the whole point. Knowing that "Ren et al." is 2 hops from
        "DETR" is trivia; knowing it is reachable BY the route

            DETR --[outperforms]--> Faster R-CNN --[proposed by]--> Ren et al.

        is a citable chain of reasoning, and that chain is what Phase 6 hands to
        the LLM as evidence.

        Traverses edges in BOTH directions - "who beat DETR" needs incoming edges
        just as much as "what did DETR beat" needs outgoing ones - but records
        which way each edge actually points so direction is never lost.
        """
        start = self.find_entity(entity)
        if start is None:
            return []

        results: list[dict] = []
        visited = {start}
        # Breadth-first: (node, path-so-far). BFS rather than DFS because we want
        # the SHORTEST route to each node - the shortest chain is the most
        # believable explanation.
        frontier: list[tuple[str, list[dict]]] = [(start, [])]

        for hop in range(1, hops + 1):
            next_frontier: list[tuple[str, list[dict]]] = []

            for node, path in frontier:
                for neighbour, direction in self._adjacent(node):
                    if neighbour in visited:
                        continue

                    for edge in self._edges_between(node, neighbour, direction):
                        step = {
                            "from": node,
                            "to": neighbour,
                            "relation": edge["relation"],
                            "direction": direction,
                            "source_file": edge.get("source_file", ""),
                            "page": edge.get("page", 0),
                            "sentence": edge.get("sentence", ""),
                            "extractor": edge.get("extractor", ""),
                        }
                        results.append({
                            "entity": neighbour,
                            "display": self.display_name(neighbour),
                            "hop_distance": hop,
                            "path": path + [step],
                        })
                        if len(results) >= max_results:
                            return results

                    visited.add(neighbour)
                    # Only the first edge to a node continues the walk, so we do
                    # not explode combinatorially on densely connected hubs.
                    first_edge = next(iter(self._edges_between(node, neighbour, direction)), None)
                    if first_edge:
                        next_frontier.append((neighbour, path + [{
                            "from": node, "to": neighbour,
                            "relation": first_edge["relation"], "direction": direction,
                            "source_file": first_edge.get("source_file", ""),
                            "page": first_edge.get("page", 0),
                            "sentence": first_edge.get("sentence", ""),
                            "extractor": first_edge.get("extractor", ""),
                        }]))

            frontier = next_frontier
            if not frontier:
                break

        return results

    def _adjacent(self, node: str):
        """Yield (neighbour, direction) for both outgoing and incoming edges."""
        for neighbour in self.graph.successors(node):
            yield neighbour, "outgoing"
        for neighbour in self.graph.predecessors(node):
            yield neighbour, "incoming"

    def _edges_between(self, node: str, neighbour: str, direction: str) -> list[dict]:
        source, target = (node, neighbour) if direction == "outgoing" else (neighbour, node)
        if not self.graph.has_edge(source, target):
            return []
        return list(self.graph[source][target].values())

    def find_path(self, entity_a: str, entity_b: str, max_hops: int = 4) -> list[dict] | None:
        """
        Shortest chain of relations connecting two entities, or None.

        Searches the UNDIRECTED projection to find a route, then reports the real
        direction of each edge. Reason: "how are DETR and Ren et al. related?" has
        an obvious answer that no purely forward-directed walk would find, because
        the middle edge points the other way.
        """
        start = self.find_entity(entity_a)
        end = self.find_entity(entity_b)
        if start is None or end is None or start == end:
            return None

        undirected = self.graph.to_undirected(as_view=True)
        try:
            node_path = nx.shortest_path(undirected, start, end)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

        if len(node_path) - 1 > max_hops:
            return None

        steps: list[dict] = []
        for left, right in zip(node_path, node_path[1:]):
            if self.graph.has_edge(left, right):
                edge = next(iter(self.graph[left][right].values()))
                direction = "outgoing"
            else:
                edge = next(iter(self.graph[right][left].values()))
                direction = "incoming"
            steps.append({
                "from": left, "to": right,
                "relation": edge["relation"], "direction": direction,
                "source_file": edge.get("source_file", ""),
                "page": edge.get("page", 0),
                "sentence": edge.get("sentence", ""),
            })
        return steps

    def get_subgraph(self, entity: str, hops: int = 2, max_nodes: int = 60) -> nx.MultiDiGraph:
        """Extract the neighbourhood around an entity - used by graph_viz.py."""
        start = self.find_entity(entity)
        if start is None:
            return nx.MultiDiGraph()

        nodes = {start}
        frontier = {start}
        for _ in range(hops):
            next_frontier: set[str] = set()
            for node in frontier:
                for neighbour, _direction in self._adjacent(node):
                    if neighbour not in nodes and len(nodes) < max_nodes:
                        nodes.add(neighbour)
                        next_frontier.add(neighbour)
            frontier = next_frontier
            if not frontier:
                break

        return self.graph.subgraph(nodes).copy()

    # -- reporting ---------------------------------------------------------

    def stats(self) -> dict:
        degrees = dict(self.graph.degree())
        relations = Counter(data["relation"] for _, _, data in self.graph.edges(data=True))
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "relations": relations,
            "top_nodes": sorted(degrees.items(), key=lambda kv: kv[1], reverse=True)[:25],
            "density": nx.density(self.graph),
            "components": nx.number_weakly_connected_components(self.graph),
        }

    # -- persistence -------------------------------------------------------

    def save(self, path: Path = GRAPH_PATH) -> None:
        """
        Serialise with pickle.

        Note: networkx REMOVED nx.write_gpickle in 3.0. The roadmap's .gpickle
        filename is kept for continuity, but the call is plain pickle.
        Sets (aliases, sources) are converted to sorted lists so the file stays
        stable across runs and readable by other tools.
        """
        ensure_dirs()
        serialisable = self.graph.copy()
        for _, data in serialisable.nodes(data=True):
            data["aliases"] = sorted(data.get("aliases", ()))
            data["sources"] = sorted(data.get("sources", ()))
        with open(path, "wb") as handle:
            pickle.dump(serialisable, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Graph saved to {path}")

    @classmethod
    def load(cls, path: Path = GRAPH_PATH) -> "KnowledgeGraph":
        if not Path(path).exists():
            raise SystemExit(f"{path} not found. Run: python src/knowledge_graph.py")
        with open(path, "rb") as handle:
            graph = pickle.load(handle)
        for _, data in graph.nodes(data=True):
            data["aliases"] = set(data.get("aliases", ()))
            data["sources"] = set(data.get("sources", ()))
        return cls(graph)


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def format_path(kg: KnowledgeGraph, path: list[dict]) -> str:
    """Render a traversal as 'DETR --[outperforms]--> Faster R-CNN --[...]--> ...'."""
    if not path:
        return ""
    pieces = [kg.display_name(path[0]["from"])]
    for step in path:
        arrow = f"--[{step['relation']}]-->" if step["direction"] == "outgoing" \
            else f"<--[{step['relation']}]--"
        pieces.append(f" {arrow} {kg.display_name(step['to'])}")
    return "".join(pieces)


def main() -> None:
    setup_console()

    parser = argparse.ArgumentParser(description="Build and explore the knowledge graph")
    parser.add_argument("--rebuild", action="store_true", help="rebuild from triples.json")
    parser.add_argument("--entity", help="explore a specific entity")
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--path", nargs=2, metavar=("FROM", "TO"),
                        help="find the shortest path between two entities")
    args = parser.parse_args()

    if args.rebuild or not GRAPH_PATH.exists():
        from src.knowledge_extractor import load_triples
        print("Building graph from triples.json ...")
        triples = load_triples()
        print(f"  {len(triples):,} triples loaded")
        kg = KnowledgeGraph.from_triples(triples)
        kg.save()
    else:
        kg = KnowledgeGraph.load()
        print(f"Loaded existing graph from {GRAPH_PATH}")

    stats = kg.stats()
    print(banner("GRAPH STATS"))
    print(f"  nodes (entities):       {stats['nodes']:,}")
    print(f"  edges (relations):      {stats['edges']:,}")
    print(f"  density:                {stats['density']:.5f}")
    print(f"  connected components:   {stats['components']:,}")

    print(banner("MOST CONNECTED ENTITIES (highest degree)"))
    print("  These are the hubs - traversals through them reach the most.\n")
    for node, degree in stats["top_nodes"]:
        sources = len(kg.graph.nodes[node].get("sources", ()))
        print(f"     {degree:>4} edges  {kg.display_name(node):<40} (in {sources} papers)")

    print(banner("MOST COMMON RELATION TYPES"))
    for relation, count in stats["relations"].most_common(20):
        print(f"     {count:>5}  {relation}")

    if args.path:
        print(banner(f"PATH: {args.path[0]}  ->  {args.path[1]}"))
        path = kg.find_path(*args.path)
        print(f"  {format_path(kg, path)}" if path else "  no path found")
        if path:
            print("\n  evidence for each hop:")
            for step in path:
                print(f"     [{step['source_file']} p.{step['page']}] {step['sentence'][:110]}")
        return

    # Demonstrate 1, 2 and 3 hop traversal on the best-connected entity.
    target = args.entity or kg.display_name(stats["top_nodes"][0][0])
    print(banner(f'TRAVERSAL DEMO: "{target}"'))

    for hops in (1, 2, 3):
        neighbours = kg.get_neighbors(target, hops=hops, max_results=400)
        at_this_hop = [n for n in neighbours if n["hop_distance"] == hops]
        print(f"\n  --- {hops} hop{'s' if hops > 1 else ''}: "
              f"{len(at_this_hop)} connections reachable in exactly {hops} step(s) ---")
        for item in at_this_hop[:6]:
            print(f"     {format_path(kg, item['path'])}")

    print(banner("WHY THIS IS THE WHOLE POINT"))
    print("  At 1 hop you learn what a single paper directly states - which you")
    print("  could also have got from vector search.")
    print("\n  At 2 and 3 hops you get facts NO paper contains. Read the chains")
    print("  above: each one crosses documents, and the connection exists only")
    print("  in the combination. No chunk holds it, so no similarity search can")
    print("  retrieve it, at any chunk size or embedding quality.")
    print("\n  Phase 5 turns a user's question into a starting entity and walks")
    print("  these paths automatically.")


if __name__ == "__main__":
    main()
