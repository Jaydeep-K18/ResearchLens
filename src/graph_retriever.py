"""
Phase 5, Step 5.1 - answering natural language with a graph walk.

THE JOB
-------
Phase 4 built a graph and gave us functions to walk it, but they take an ENTITY.
Users type sentences. This module bridges that gap:

    "Which models outperform YOLO?"
        -> find the entities mentioned            -> {YOLO}
        -> find where they live in the graph      -> node "yolo"
        -> notice the relation the user asked for -> "outperforms"
        -> walk outward, preferring those edges
        -> return the paths, with the sentence behind each hop as evidence

THE HARD PART IS STEP ONE
-------------------------
notebooks/02_ner_demo.py established that spaCy NER is unreliable here: it missed
DETR and YOLOv3 entirely and tagged "Faster" as a person. If entity extraction
fails, the graph contributes NOTHING to the answer - the traversal starts nowhere
and the whole second memory sits idle. So we use four strategies at once and keep
whatever matches the graph:

    1. spaCy NER          - good for author names and organisations
    2. noun chunks        - catches "the DETR model" that NER skipped
    3. capitalised/acronym patterns - catches DETR, YOLOv3, COCO, ResNet-50
    4. fuzzy match against the graph's own node list

Strategy 4 is the one that carries the load, and it works because of something
worth noticing: after Phase 4, the graph's node list IS a domain vocabulary
extracted from this corpus. We do not need a model that knows what DETR is - the
corpus already told us, and we wrote it down.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.knowledge_graph import KnowledgeGraph, format_path, normalise_entity  # noqa: E402
from src.utils import banner, setup_console  # noqa: E402

DEFAULT_HOPS = 2
DEFAULT_MAX_EVIDENCE = 30

# Words in a question that signal which relation the user cares about. Matching
# these lets us PREFER the right edges instead of returning the neighbourhood
# indiscriminately - the difference between answering the question and dumping
# everything near the entity.
_RELATION_CUES: dict[str, tuple[str, ...]] = {
    "outperforms": ("outperform", "beat", "better than", "surpass", "exceed",
                    "superior", "wins", "improve over", "stronger"),
    "underperforms": ("worse", "slower", "underperform", "beaten by", "lose"),
    "achieves": ("achieve", "accuracy", "score", "map", "performance", "result",
                 "reach", "attain"),
    "trained_on": ("trained", "training data", "pretrained", "pre-trained"),
    "evaluated_on": ("evaluated", "benchmark", "tested", "dataset", "evaluation"),
    "proposes": ("propose", "author", "who wrote", "introduced by", "created by",
                 "researcher", "developed by", "behind"),
    "introduces": ("introduce", "present", "contribute"),
    "uses": ("use", "employ", "based on", "rely", "component", "architecture"),
    "extends": ("extend", "build on", "improve", "successor", "variant"),
    "compared_to": ("compare", "versus", "vs", "against", "difference"),
}

# Stopword-ish tokens that are never useful as graph entry points.
_QUERY_STOPWORDS = {
    "which", "what", "who", "where", "when", "how", "why", "does", "did", "do",
    "is", "are", "was", "were", "the", "a", "an", "of", "in", "on", "for", "to",
    "and", "or", "that", "this", "these", "those", "models", "model", "paper",
    "papers", "authors", "author", "research", "work", "method", "methods",
    "approach", "system", "also", "other", "any", "all", "some", "their", "its",
}

_ACRONYM_RE = re.compile(r"\b[A-Z][A-Za-z]*(?:[-\d]+[A-Za-z\d]*)*\b")

_nlp_cache: dict[str, object] = {}


def _get_nlp():
    if "nlp" not in _nlp_cache:
        import spacy
        _nlp_cache["nlp"] = spacy.load("en_core_web_sm")
    return _nlp_cache["nlp"]


# ---------------------------------------------------------------------------
# Step 1: query -> graph entities
# ---------------------------------------------------------------------------

def candidate_mentions(query: str) -> list[str]:
    """Collect every string in the query that might name an entity."""
    nlp = _get_nlp()
    doc = nlp(query)
    candidates: list[str] = []

    # 1. NER - reliable for people and organisations.
    candidates.extend(ent.text for ent in doc.ents)

    # 2. Noun chunks - catches technical terms NER skipped.
    candidates.extend(chunk.text for chunk in doc.noun_chunks)

    # 3. Capitalised tokens and acronyms. This is what actually catches DETR,
    #    YOLOv3, ResNet-50 and COCO, none of which NER recognises.
    candidates.extend(_ACRONYM_RE.findall(query))

    # 4. Bare content words, as a last resort for lowercase queries.
    candidates.extend(
        token.text for token in doc
        if token.pos_ in {"NOUN", "PROPN"} and token.text.lower() not in _QUERY_STOPWORDS
    )

    cleaned: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        text = candidate.strip()
        normalised = normalise_entity(text)
        if not normalised or normalised in _QUERY_STOPWORDS or len(normalised) < 2:
            continue

        # Drop candidates made ENTIRELY of question words. The single-token check
        # above misses multi-word chunks: spaCy hands back "Which models" as one
        # noun chunk, which is not in the stopword set as a phrase, and fuzzy
        # matching then happily resolved it to a graph node called "mode".
        tokens = normalised.split()
        if all(token in _QUERY_STOPWORDS for token in tokens):
            continue

        if normalised in seen:
            continue
        seen.add(normalised)
        cleaned.append(text)
    return cleaned


def resolve_entities(query: str, kg: KnowledgeGraph, threshold: int = 82,
                     max_entities: int = 4) -> list[dict]:
    """
    Map the query's candidate mentions onto actual graph nodes.

    Returns [{mention, node, display, score}], best first. Longer mentions win
    ties: matching "Faster R-CNN" is more informative than matching "R-CNN".
    """
    resolved: dict[str, dict] = {}

    for mention in candidate_mentions(query):
        for node, score in kg.find_entities(mention, limit=3, threshold=threshold):
            existing = resolved.get(node)
            # Prefer the higher score, then the longer (more specific) mention.
            if existing is None or (score, len(mention)) > (existing["score"], len(existing["mention"])):
                resolved[node] = {
                    "mention": mention,
                    "node": node,
                    "display": kg.display_name(node),
                    "score": score,
                }

    ranked = sorted(resolved.values(), key=lambda r: (r["score"], len(r["mention"])), reverse=True)
    return ranked[:max_entities]


def detect_relation_cues(query: str) -> list[str]:
    """Which relation types is this question about? Empty list means 'no preference'."""
    lowered = query.lower()
    return [relation for relation, cues in _RELATION_CUES.items()
            if any(cue in lowered for cue in cues)]


# ---------------------------------------------------------------------------
# Step 2: walk the graph
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    kg: KnowledgeGraph,
    hops: int = DEFAULT_HOPS,
    max_evidence: int = DEFAULT_MAX_EVIDENCE,
    verbose: bool = False,
) -> list[dict]:
    """
    Walk the graph from every entity the query mentions and return ranked evidence.

    Each evidence item is shaped to slot straight into the hybrid retriever
    alongside vector results:

        {
          "text":             the source sentence - what the LLM actually reads
          "triple":           (subject, relation, object)
          "path_str":         "DETR --[outperforms]--> Faster R-CNN"
          "source_file", "page":  citation
          "hop_distance":     how many edges from the query entity
          "score":            relevance, see below
          "retrieval_method": "graph"
        }
    """
    entities = resolve_entities(query, kg)
    if not entities:
        if verbose:
            print("  no query entities matched the graph")
        return []

    cues = detect_relation_cues(query)
    if verbose:
        print(f"  query entities: {[e['display'] for e in entities]}")
        print(f"  relation cues:  {cues or '(none - returning general neighbourhood)'}")

    evidence: list[dict] = []
    seen: set[tuple] = set()

    for entity in entities:
        neighbours = kg.get_neighbors(entity["node"], hops=hops, max_results=400)

        for neighbour in neighbours:
            path = neighbour["path"]
            if not path:
                continue
            final = path[-1]

            # Entity-resolution artefacts are plumbing, not evidence.
            if final["relation"] == "variant_of":
                continue

            key = (final["from"], final["relation"], final["to"], final["source_file"])
            if key in seen:
                continue
            seen.add(key)

            # --- scoring ------------------------------------------------
            # 1. Nearer hops are stronger evidence: a direct statement beats a
            #    chain of inferences.
            score = 1.0 / neighbour["hop_distance"]
            # 2. Big boost when the edge is the relation the user asked about.
            #    This is what makes "which models outperform YOLO?" return
            #    outperforms edges rather than YOLO's whole neighbourhood.
            if cues and final["relation"] in cues:
                score += 1.2
            elif cues:
                score -= 0.15
            # 3. How well the starting entity matched the query.
            score += (entity["score"] / 100.0) * 0.4
            # 4. Domain-extractor relations are more meaningful for these
            #    questions than REBEL's generic "instance of"/"part of".
            if final.get("extractor") == "domain":
                score += 0.25

            subject_display = kg.display_name(final["from"])
            object_display = kg.display_name(final["to"])
            if final["direction"] == "incoming":
                subject_display, object_display = object_display, subject_display

            evidence.append({
                "text": final["sentence"] or f"{subject_display} {final['relation']} {object_display}",
                "triple": (subject_display, final["relation"], object_display),
                "path_str": format_path(kg, path),
                "source_file": final["source_file"],
                "page": final["page"],
                "hop_distance": neighbour["hop_distance"],
                "start_entity": entity["display"],
                "relation": final["relation"],
                "extractor": final.get("extractor", ""),
                "score": score,
                "retrieval_method": "graph",
            })

    evidence.sort(key=lambda item: item["score"], reverse=True)
    for rank, item in enumerate(evidence):
        item["rank"] = rank
    return evidence[:max_evidence]


def print_evidence(evidence: list[dict], limit: int = 12) -> None:
    if not evidence:
        print("  (no graph evidence found)")
        return
    for item in evidence[:limit]:
        subject, relation, obj = item["triple"]
        print(f"\n  [{item['rank']}] score={item['score']:.2f}  "
              f"{item['hop_distance']} hop(s)  via {item['start_entity']}")
        print(f"      {subject}  --[{relation}]-->  {obj}")
        print(f"      path: {item['path_str']}")
        print(f"      [{item['source_file']} p.{item['page']}] {item['text'][:130]}")


def main() -> None:
    setup_console()

    parser = argparse.ArgumentParser(description="Graph-based retrieval from natural language")
    parser.add_argument("--query", "-q", help="a single question to run")
    parser.add_argument("--hops", type=int, default=DEFAULT_HOPS)
    args = parser.parse_args()

    kg = KnowledgeGraph.load()
    stats = kg.stats()
    print(f"Graph: {stats['nodes']:,} entities, {stats['edges']:,} relations")

    queries = [args.query] if args.query else [
        "Which models outperform YOLO?",
        "Who proposed Faster R-CNN?",
        "What datasets was the Vision Transformer trained on?",
        "What accuracy does Swin Transformer achieve?",
    ]

    for query in queries:
        print(banner(f'QUERY: "{query}"'))
        evidence = retrieve(query, kg, hops=args.hops, verbose=True)
        print_evidence(evidence)

    print(banner("WHAT JUST HAPPENED"))
    print("  No embeddings were involved. Nothing was compared for text")
    print("  similarity. The system found the entity the question names, then")
    print("  WALKED the relationships out from it.")
    print("\n  Notice the relation cues: asking 'which models outperform X'")
    print("  boosts outperforms edges, so the answer is the models that beat X -")
    print("  not everything that happens to sit near X in the graph.")
    print("\n  Step 5.2 merges this with vector search, so the system has both")
    print("  'text that resembles the question' and 'facts structurally connected")
    print("  to it'. Those are different kinds of evidence, and hard questions")
    print("  need both.")


if __name__ == "__main__":
    main()
