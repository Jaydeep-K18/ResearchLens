"""
Phase 5, Steps 5.2 + 5.3 - merging the two memories, then re-ranking for precision.

THE ARCHITECTURE IN ONE PICTURE

        query
          |
    +-----+-----+
    |           |
  VECTOR      GRAPH            <- run independently, they answer different questions
  "text like  "facts connected
   this"       to this"
    |           |
    +-----+-----+
          |
       MERGE + DEDUPLICATE     <- one list, provenance preserved
          |
    CROSS-ENCODER RE-RANK      <- a slower, sharper judge reorders the shortlist
          |
       top-K results  -> Phase 6 hands these to Gemini

WHY MERGE AT ALL
----------------
The two retrievers fail in opposite directions, which is exactly why they
combine well:

  Vector search  finds passages that RESEMBLE the question. It cannot follow a
                 chain, so multi-hop questions silently return topical-looking
                 chunks that do not contain the answer.
  Graph search   follows chains across documents. But it only knows what the
                 extractors captured, so anything phrased as ordinary prose -
                 definitions, explanations, caveats - is invisible to it.

A simple question is answered by the first. A multi-hop question needs the
second. A real question usually needs some of both, and the system does not know
in advance which - so it runs both and lets the re-ranker sort it out.

WHY RE-RANK (Step 5.3)
----------------------
This is the part worth understanding properly. Both retrieval methods produce
scores, but those scores are not comparable and neither is very precise.

  BI-ENCODER (the embedding model from Phase 2):
      embed(query) ................ 384 numbers, computed WITHOUT seeing any document
      embed(chunk) ................ 384 numbers, computed WITHOUT seeing the query
      similarity = dot product
  Because the chunk's vector never depends on the query, it can be computed once,
  in advance, for the whole corpus. That is what makes searching 40,000 chunks
  fast. The cost is that the model must compress the chunk's entire meaning into
  one vector, before knowing what will be asked of it.

  CROSS-ENCODER (this step):
      score = model("query [SEP] chunk")   <- both together, in one forward pass
  Every layer lets query tokens attend to chunk tokens. The model can notice that
  the chunk says "DETR" and the query asks about "DETR", and that the chunk
  discusses training cost while the query asks about accuracy. It is far more
  accurate. It is also impossible to precompute: N candidates means N forward
  passes, at query time.

So the standard pattern, and the one used here:
    fast + approximate over EVERYTHING  ->  slow + precise over the TOP FEW
Retrieve ~20 candidates with embeddings and graph walks, then spend ~0.5s letting
the cross-encoder put the best 8 in the right order.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import GRAPH_PATH, banner, setup_console  # noqa: E402

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_TOP_K = 8
DEFAULT_VECTOR_K = 10
DEFAULT_GRAPH_HOPS = 2

_cross_encoder_cache: dict[str, object] = {}


def get_cross_encoder(name: str = CROSS_ENCODER_MODEL):
    """Load (once) the re-ranking model. ~80MB, CPU-friendly."""
    if name not in _cross_encoder_cache:
        from sentence_transformers import CrossEncoder
        _cross_encoder_cache[name] = CrossEncoder(name)
    return _cross_encoder_cache[name]


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def _normalise_scores(items: list[dict], key: str = "score") -> None:
    """
    Rescale a group's scores to 0-1, in place.

    Necessary because the two retrievers produce scores on incompatible scales:
    vector similarities land around 0.4-0.7, while graph scores are a hand-built
    sum that can exceed 2. Merging the raw numbers would let graph evidence
    outrank vector evidence purely because of how each formula happens to be
    scaled - a units bug, not a relevance judgement. Normalising each group first
    makes "best of its kind" comparable across kinds.
    """
    if not items:
        return
    values = [item[key] for item in items]
    lowest, highest = min(values), max(values)
    spread = highest - lowest
    for item in items:
        item["normalised_score"] = 1.0 if spread == 0 else (item[key] - lowest) / spread


def _text_overlap(left: str, right: str) -> float:
    """Jaccard similarity over word sets - cheap and good enough for dedup."""
    left_words = set(left.lower().split())
    right_words = set(right.lower().split())
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / len(left_words | right_words)


def deduplicate(results: list[dict], threshold: float = 0.75) -> list[dict]:
    """
    Drop near-duplicate results from the same source file.

    Duplication is guaranteed here, by construction: chunks overlap by 100
    characters (Phase 1), and a graph evidence sentence is frequently a sentence
    that also sits inside a retrieved chunk. Without this, the top-8 can be five
    restatements of one fact, which both wastes the LLM's context and makes a
    single source look like corroboration from several.

    Results are assumed to arrive best-first, so the first copy seen is kept.
    """
    kept: list[dict] = []
    for candidate in results:
        duplicate = False
        for existing in kept:
            if candidate["source_file"] != existing["source_file"]:
                continue
            if _text_overlap(candidate["text"], existing["text"]) >= threshold:
                # Record that another method also found this - useful signal for
                # the LLM and for the UI in Phase 7.
                existing.setdefault("also_found_by", set()).add(candidate["retrieval_method"])
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


# ---------------------------------------------------------------------------
# The retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    def __init__(self, store=None, kg=None, cross_encoder_name: str = CROSS_ENCODER_MODEL):
        from src.knowledge_graph import KnowledgeGraph
        from src.vector_store import VectorStore

        self.store = store if store is not None else VectorStore()
        self.cross_encoder_name = cross_encoder_name

        # A missing graph is a normal state, not an error: a fresh workspace has
        # documents embedded before extraction finishes, and the user should be
        # able to ask questions in the meantime. This used to call
        # KnowledgeGraph.load() unconditionally, which raises SystemExit when the
        # pickle is absent - so constructing a retriever on an empty workspace
        # killed the process. Retrieval degrades to vector-only when kg is None.
        if kg is not None:
            self.kg = kg
        elif GRAPH_PATH.exists():
            self.kg = KnowledgeGraph.load()
        else:
            self.kg = None

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        vector_k: int = DEFAULT_VECTOR_K,
        graph_hops: int = DEFAULT_GRAPH_HOPS,
        rerank: bool = True,
        verbose: bool = False,
    ) -> dict:
        """
        Run both retrievers, merge, re-rank, and return the top-K.

        Returns a dict with the intermediate stages kept intact
        (vector_results, graph_results, merged, reranked) because Phase 6's
        LangGraph state and Phase 7's UI both want to show the pipeline's work,
        not just its conclusion.
        """
        from src import graph_retriever

        timings: dict[str, float] = {}

        # --- the two retrievers ------------------------------------------
        # Conceptually parallel; run sequentially here because both are CPU-bound
        # and threading them would just contend for the same cores. Phase 6's
        # LangGraph expresses the parallelism explicitly at the graph level.
        started = time.time()
        vector_results = self.store.search(query, top_k=vector_k)
        timings["vector"] = time.time() - started

        started = time.time()
        # No graph yet (fresh workspace, extraction still pending) means
        # vector-only retrieval rather than a crash.
        graph_results = [] if self.kg is None else graph_retriever.retrieve(
            query, self.kg, hops=graph_hops, max_evidence=vector_k * 2, verbose=verbose
        )
        timings["graph"] = time.time() - started

        # --- merge --------------------------------------------------------
        _normalise_scores(vector_results)
        _normalise_scores(graph_results)

        merged = sorted(
            vector_results + graph_results,
            key=lambda item: item["normalised_score"],
            reverse=True,
        )
        for rank, item in enumerate(merged):
            item["merged_rank"] = rank

        merged = deduplicate(merged)

        # --- re-rank ------------------------------------------------------
        started = time.time()
        if rerank and merged:
            reranked = self.rerank(query, merged)
        else:
            reranked = merged
        timings["rerank"] = time.time() - started

        return {
            "query": query,
            "vector_results": vector_results,
            "graph_results": graph_results,
            "merged": merged,
            "reranked": reranked[:top_k],
            "timings": timings,
        }

    def rerank(self, query: str, results: list[dict]) -> list[dict]:
        """Score every (query, passage) pair with the cross-encoder and re-sort."""
        model = get_cross_encoder(self.cross_encoder_name)

        # For graph evidence, give the cross-encoder the triple AS WELL AS the
        # sentence. The sentence alone often omits what makes it relevant - "It
        # outperforms the baseline by 2.1 points" is uninformative without
        # knowing that "it" is DETR. The triple supplies exactly that.
        passages: list[str] = []
        for item in results:
            if item["retrieval_method"] == "graph" and item.get("triple"):
                subject, relation, obj = item["triple"]
                passages.append(f"{subject} {relation.replace('_', ' ')} {obj}. {item['text']}")
            else:
                passages.append(item["text"])

        scores = model.predict([(query, passage) for passage in passages])

        for item, score in zip(results, scores):
            item["rerank_score"] = float(score)

        reranked = sorted(results, key=lambda item: item["rerank_score"], reverse=True)
        for rank, item in enumerate(reranked):
            item["final_rank"] = rank
        return reranked


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_comparison(result: dict, limit: int = 8) -> None:
    """Show what each method contributed and how re-ranking changed the order."""
    vector_count = len(result["vector_results"])
    graph_count = len(result["graph_results"])
    print(f"\n  vector search: {vector_count} results in {result['timings']['vector']:.2f}s")
    print(f"  graph search:  {graph_count} results in {result['timings']['graph']:.2f}s")
    print(f"  after merge + dedup: {len(result['merged'])}")
    print(f"  re-ranking:    {result['timings']['rerank']:.2f}s")

    print(f"\n  {'final':<7}{'was':<7}{'move':<8}{'method':<9}{'ce score':<10}source")
    print("  " + "-" * 74)
    for item in result["reranked"][:limit]:
        was = item.get("merged_rank", -1)
        now = item.get("final_rank", 0)
        delta = was - now
        move = f"{delta:+d}" if delta else "-"
        score = item.get("rerank_score", 0.0)
        origin = f"{item['source_file']} p.{item.get('page', '?')}"
        print(f"  {now:<7}{was:<7}{move:<8}{item['retrieval_method']:<9}{score:<10.3f}{origin}")

    print("\n  top results:")
    for item in result["reranked"][:limit]:
        text = " ".join(item["text"].split())[:140]
        tag = item["retrieval_method"].upper()
        extra = ""
        if item["retrieval_method"] == "graph":
            subject, relation, obj = item["triple"]
            extra = f"  [{subject} --{relation}--> {obj}, {item['hop_distance']} hop]"
        print(f"\n    [{item.get('final_rank', 0)}] {tag}{extra}")
        print(f"        {text}")


def main() -> None:
    setup_console()

    parser = argparse.ArgumentParser(description="Hybrid vector + graph retrieval")
    parser.add_argument("--query", "-q")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    retriever = HybridRetriever()
    print(f"Vector store: {retriever.store.count():,} chunks")
    stats = retriever.kg.stats()
    print(f"Graph:        {stats['nodes']:,} entities, {stats['edges']:,} relations")

    if args.query:
        queries = [(args.query, "")]
    else:
        queries = [
            ("What is the self-attention mechanism?",
             "SIMPLE - stated in one place. Vector search should dominate."),
            ("Which models outperform Faster R-CNN, and who proposed them?",
             "MULTI-HOP - needs a chain across papers. Graph should contribute."),
        ]

    for query, note in queries:
        print(banner(f'QUERY: "{query}"'))
        if note:
            print(f"  {note}")
        result = retriever.retrieve(query, top_k=args.top_k, rerank=not args.no_rerank,
                                    verbose=True)
        print_comparison(result, limit=args.top_k)

        methods = [item["retrieval_method"] for item in result["reranked"]]
        print(f"\n  contribution to the final top-{args.top_k}: "
              f"{methods.count('vector')} vector, {methods.count('graph')} graph")


if __name__ == "__main__":
    main()
