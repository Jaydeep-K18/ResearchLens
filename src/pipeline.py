"""
Phase 6 - the full KG-RAG pipeline, orchestrated with LangGraph.

THE FLOW
--------
                          query
                            |
                 +----------+----------+
                 |                     |
         retrieve_vector        retrieve_graph        <- parallel; different
         (ChromaDB, Phase 2)    (graph walk, 5.1)        state keys, no conflict
                 |                     |
                 +----------+----------+
                            |
                    merge_and_rerank                  <- dedupe + cross-encoder (5.3)
                            |
                        synthesize                    <- Gemini Flash, grounded prompt
                            |
                      format_output                   <- citations + provenance
                            |
                           END

Everything before `synthesize` was built in Phases 1-5. This file is where the
two memories finally produce a single cited answer, and where the prompt does
the work of keeping that answer honest.

WHY THE PROMPT IS THE MOST IMPORTANT CODE IN THIS FILE
------------------------------------------------------
Retrieval decides what the model CAN say. The prompt decides what it DOES say.
Phase 2 established that a vector store always returns its k nearest neighbours
whether or not any of them is relevant - "nearest" is not "near". So the only
thing standing between irrelevant context and a confident wrong answer is an
instruction to admit when the context is insufficient.

The prompt below also does something specific to this project: it labels each
piece of evidence with HOW it was retrieved, and asks the model to treat the two
kinds differently. Text retrieved by similarity is something a paper states.
A graph path is something the SYSTEM inferred by connecting several papers. The
second is more powerful and more fragile, and an honest answer distinguishes
them rather than presenting an inference as a quotation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Annotated

from typing_extensions import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llm import LLMUnavailable, get_llm  # noqa: E402
from src.utils import banner, setup_console  # noqa: E402

# 12 rather than 8. Multi-hop questions in this domain are usually COMPOUND -
# "which models outperform X, AND who proposed them" is two sub-questions sharing
# one evidence budget. With 8 slots the first half filled every one, pushing the
# authorship edges (which were correctly retrieved, at ranks 8-11) below the
# cutoff, and the model then reported that the authors were not in the evidence.
# Query decomposition is the real fix and is listed in the README's future work;
# a slightly larger budget is the cheap version.
DEFAULT_TOP_K = 12


def merge_timings(left: dict | None, right: dict | None) -> dict:
    """
    Reducer for the `timings` key. See the note in KGRagState below.
    """
    return {**(left or {}), **(right or {})}


class KGRagState(TypedDict, total=False):
    """
    Everything the pipeline carries from end to end.

    Intermediates are kept deliberately: Phase 7's UI shows the vector chunks,
    the graph evidence and the re-ranking separately, and evaluate.py needs to
    know which method contributed what. A pipeline that returns only its answer
    cannot be inspected, and an uninspectable retrieval system is impossible to
    debug or to demo.

    WHY `timings` IS ANNOTATED AND THE OTHERS ARE NOT
    -------------------------------------------------
    By default a state key is a "last value" channel that accepts exactly ONE
    write per step. retrieve_vector and retrieve_graph run in the SAME step, so:

        vector_results / graph_results   different keys -> fine
        timings                          BOTH nodes write it -> error

    LangGraph does not silently pick a winner; it raises InvalidUpdateError.
    That is the right behaviour - a silently dropped write would mean losing
    half the timing data with no indication anything went wrong.

    The fix is to give the key a REDUCER: a function saying how to combine
    concurrent writes. Annotated[dict, merge_timings] tells LangGraph to merge
    the two dicts instead of rejecting them.
    """
    query: str
    vector_results: list
    graph_results: list
    merged_results: list
    reranked_results: list
    answer: str
    sources: list
    timings: Annotated[dict, merge_timings]
    error: str | None


# ---------------------------------------------------------------------------
# The prompt (Step 6.3)
# ---------------------------------------------------------------------------

SYNTHESIS_PROMPT = """You are a research assistant answering questions about a corpus of computer vision papers. You have been given evidence retrieved by two different systems. Answer using ONLY that evidence.

# HOW TO READ THE EVIDENCE

Each item is labelled with how it was retrieved:

- [TEXT] - a passage found by semantic similarity to the question. This is
  something a paper literally says. Treat it as a direct quotation.

- [GRAPH] - a relationship the system extracted from the papers and connected
  across documents. It shows a chain like
      DETR --[outperforms]--> Faster R-CNN --[proposed by]--> Ren et al.
  along with the sentence each link came from. A multi-hop chain is an INFERENCE
  the system assembled - no single paper states it. It is often the only route
  to the answer, and it is also the more fragile kind of evidence.

# RULES

1. Use ONLY the evidence below. Never draw on outside knowledge, even if you are
   confident and the evidence is thin.
2. Cite every factual claim inline as [Source: filename.pdf, p.N], using the
   labels exactly as shown in the evidence.
3. When your answer depends on connecting several sources, show the chain
   explicitly. For example: "DETR outperforms Faster R-CNN [Source: detr.pdf,
   p.1], and Faster R-CNN was proposed by Ren et al. [Source: faster_rcnn.pdf,
   p.2], so the authors in question are Ren et al."
4. Distinguish what papers STATE from what the graph CONNECTS. Use wording like
   "X reports that..." for [TEXT] evidence and "connecting these sources
   suggests..." for multi-hop [GRAPH] chains.
5. If the evidence does not answer the question, SAY SO PLAINLY and state what
   is missing. Do not pad, do not guess, do not answer a nearby question instead.
   A precise "the corpus does not cover this" is a correct and useful answer.
6. Be concrete. Prefer specific model names, numbers and authors over general
   statements.

# EVIDENCE

{context}

# QUESTION

{question}

# ANSWER
"""


def format_evidence(results: list[dict]) -> str:
    """Lay out the retrieved evidence with its retrieval method and citation."""
    blocks: list[str] = []

    for i, item in enumerate(results, start=1):
        citation = f"[Source: {item['source_file']}, p.{item.get('page', '?')}]"

        if item["retrieval_method"] == "graph":
            subject, relation, obj = item.get("triple", ("", "", ""))
            hops = item.get("hop_distance", 1)
            chain = item.get("path_str", "")
            blocks.append(
                f"[{i}] [GRAPH] {citation}  ({hops} hop{'s' if hops > 1 else ''})\n"
                f"    relationship: {subject} --[{relation}]--> {obj}\n"
                f"    chain: {chain}\n"
                f"    evidence sentence: {' '.join(item['text'].split())}"
            )
        else:
            section = item.get("section", "")
            heading = f"  (section: {section})" if section else ""
            blocks.append(
                f"[{i}] [TEXT] {citation}{heading}\n"
                f"    {' '.join(item['text'].split())}"
            )

    return "\n\n".join(blocks)


def collect_sources(results: list[dict]) -> list[dict]:
    """Deduplicated list of (file, page) actually used, for the UI and eval."""
    seen: dict[tuple[str, int], dict] = {}
    for item in results:
        key = (item["source_file"], item.get("page", 0))
        if key not in seen:
            seen[key] = {
                "source_file": item["source_file"],
                "page": item.get("page", 0),
                "methods": set(),
            }
        seen[key]["methods"].add(item["retrieval_method"])
    return [
        {**entry, "methods": sorted(entry["methods"])}
        for entry in sorted(seen.values(), key=lambda e: (e["source_file"], e["page"]))
    ]


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------

class KGRagPipeline:
    """Builds and runs the LangGraph workflow."""

    def __init__(self, retriever=None, top_k: int = DEFAULT_TOP_K,
                 vector_k: int = 10, graph_hops: int = 2) -> None:
        from src.hybrid_retriever import HybridRetriever

        self.retriever = retriever if retriever is not None else HybridRetriever()
        self.top_k = top_k
        self.vector_k = vector_k
        self.graph_hops = graph_hops
        self.workflow = self._build()

    # -- nodes -------------------------------------------------------------

    def _retrieve_vector(self, state: KGRagState) -> dict:
        started = time.time()
        results = self.retriever.store.search(state["query"], top_k=self.vector_k)
        return {
            "vector_results": results,
            "timings": {"vector": time.time() - started},
        }

    def _retrieve_graph(self, state: KGRagState) -> dict:
        from src import graph_retriever

        started = time.time()
        results = graph_retriever.retrieve(
            state["query"], self.retriever.kg,
            hops=self.graph_hops, max_evidence=self.vector_k * 2,
        )
        return {
            "graph_results": results,
            "timings": {"graph": time.time() - started},
        }

    def _merge_and_rerank(self, state: KGRagState) -> dict:
        from src.hybrid_retriever import _normalise_scores, deduplicate

        started = time.time()
        vector_results = list(state.get("vector_results") or [])
        graph_results = list(state.get("graph_results") or [])

        _normalise_scores(vector_results)
        _normalise_scores(graph_results)

        merged = sorted(vector_results + graph_results,
                        key=lambda item: item["normalised_score"], reverse=True)
        for rank, item in enumerate(merged):
            item["merged_rank"] = rank
        merged = deduplicate(merged)

        reranked = self.retriever.rerank(state["query"], merged) if merged else []

        return {
            "merged_results": merged,
            "reranked_results": reranked[: self.top_k],
            "timings": {"merge_and_rerank": time.time() - started},
        }

    def _synthesize(self, state: KGRagState) -> dict:
        started = time.time()
        results = state.get("reranked_results") or []

        if not results:
            return {
                "answer": "No evidence was retrieved for this question. The corpus "
                          "may not cover this topic, or the index may be empty.",
                "error": None,
                "timings": {"synthesize": time.time() - started},
            }

        prompt = SYNTHESIS_PROMPT.format(
            context=format_evidence(results), question=state["query"]
        )

        llm = get_llm()
        try:
            answer = llm.generate(prompt, max_tokens=1600)
            return {
                "answer": answer,
                "error": None,
                "timings": {"synthesize": time.time() - started},
            }
        except LLMUnavailable as exc:
            return {
                "answer": "",
                "error": str(exc),
                "timings": {"synthesize": time.time() - started},
            }

    def _format_output(self, state: KGRagState) -> dict:
        return {"sources": collect_sources(state.get("reranked_results") or [])}

    # -- graph construction ------------------------------------------------

    def _build(self):
        from langgraph.graph import END, START, StateGraph

        builder = StateGraph(KGRagState)
        builder.add_node("retrieve_vector", self._retrieve_vector)
        builder.add_node("retrieve_graph", self._retrieve_graph)
        builder.add_node("merge_and_rerank", self._merge_and_rerank)
        builder.add_node("synthesize", self._synthesize)
        builder.add_node("format_output", self._format_output)

        # Fan out: both retrievers start from the query and run concurrently.
        # They write different state keys (vector_results / graph_results), which
        # is what makes the parallel write safe.
        builder.add_edge(START, "retrieve_vector")
        builder.add_edge(START, "retrieve_graph")

        # Join: merge_and_rerank waits for BOTH branches before running.
        builder.add_edge("retrieve_vector", "merge_and_rerank")
        builder.add_edge("retrieve_graph", "merge_and_rerank")

        builder.add_edge("merge_and_rerank", "synthesize")
        builder.add_edge("synthesize", "format_output")
        builder.add_edge("format_output", END)

        return builder.compile()

    # -- public API --------------------------------------------------------

    def run(self, query: str) -> KGRagState:
        started = time.time()
        # Per-node timings are merged by the reducer on the `timings` key; we
        # only add the wall-clock total, which nothing else writes.
        state: KGRagState = self.workflow.invoke({"query": query})
        state["timings"] = {**(state.get("timings") or {}), "total": time.time() - started}
        return state

    def diagram(self) -> str:
        try:
            return self.workflow.get_graph().draw_ascii()
        except Exception:  # noqa: BLE001 - drawing needs an optional dependency
            return ("START -> [retrieve_vector | retrieve_graph] -> merge_and_rerank\n"
                    "      -> synthesize -> format_output -> END")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_result(state: KGRagState, show_evidence: bool = True) -> None:
    vector_count = len(state.get("vector_results") or [])
    graph_count = len(state.get("graph_results") or [])
    reranked = state.get("reranked_results") or []
    methods = [item["retrieval_method"] for item in reranked]

    print(f"\n  retrieval: {vector_count} vector + {graph_count} graph "
          f"-> {len(state.get('merged_results') or [])} merged "
          f"-> top {len(reranked)}")
    print(f"  final mix: {methods.count('vector')} vector, {methods.count('graph')} graph")

    timings = state.get("timings") or {}
    if timings:
        parts = ", ".join(f"{name} {value:.2f}s" for name, value in timings.items())
        print(f"  timings:   {parts}")

    if show_evidence and reranked:
        print("\n  evidence handed to the model:")
        for i, item in enumerate(reranked, start=1):
            tag = item["retrieval_method"].upper()
            if item["retrieval_method"] == "graph":
                subject, relation, obj = item.get("triple", ("", "", ""))
                detail = f"{subject} --{relation}--> {obj}  ({item.get('hop_distance')} hop)"
            else:
                detail = " ".join(item["text"].split())[:95]
            print(f"    [{i}] {tag:<6} {item['source_file']} p.{item.get('page')}  {detail}")

    print("\n  " + "=" * 74)
    if state.get("error"):
        print(f"  [no answer generated] {state['error']}")
    else:
        print("  ANSWER:\n")
        for line in (state.get("answer") or "").splitlines():
            print(f"    {line}")
    print("  " + "=" * 74)

    sources = state.get("sources") or []
    if sources:
        print("\n  sources used:")
        for source in sources:
            print(f"    - {source['source_file']} p.{source['page']}  "
                  f"({', '.join(source['methods'])})")


TEST_QUESTIONS = [
    ("What is the self-attention mechanism?",
     "SIMPLE - one paper states it directly."),
    ("What datasets was the Vision Transformer trained and evaluated on?",
     "MEDIUM - facts spread across a paper, needs the trained_on/evaluated_on relations."),
    ("Which models outperform Faster R-CNN, and who proposed those models?",
     "HARD - a multi-hop chain: model -> outperforms -> model -> proposed by -> author."),
]


def main() -> None:
    setup_console()

    parser = argparse.ArgumentParser(description="Run the full KG-RAG pipeline")
    parser.add_argument("--question", "-q")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--show-prompt", action="store_true",
                        help="print the exact prompt sent to the model")
    args = parser.parse_args()

    pipeline = KGRagPipeline(top_k=args.top_k, graph_hops=args.hops)

    print(banner("PIPELINE STRUCTURE"))
    print(pipeline.diagram())

    llm = get_llm()
    if not llm.available:
        print(banner("NO LLM KEY - RETRIEVAL ONLY"))
        print(llm.setup_hint())

    if args.interactive:
        print(banner("KG-RAG - interactive (type 'quit' to exit)"))
        while True:
            try:
                question = input("\n  question> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  bye.")
                return
            if question.lower() in {"quit", "exit", "q"}:
                return
            if question:
                print_result(pipeline.run(question))
        return

    questions = [(args.question, "")] if args.question else TEST_QUESTIONS

    for question, note in questions:
        print(banner(f'QUESTION: "{question}"'))
        if note:
            print(f"  {note}")
        state = pipeline.run(question)

        if args.show_prompt:
            print(banner("EXACT PROMPT SENT TO THE MODEL"))
            print(SYNTHESIS_PROMPT.format(
                context=format_evidence(state.get("reranked_results") or []),
                question=question,
            ))

        print_result(state)


if __name__ == "__main__":
    main()
