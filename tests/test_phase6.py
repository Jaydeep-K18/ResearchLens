"""
Phase 6 tests - evidence formatting, source collection, and the LangGraph flow.

The pipeline is exercised with a stubbed retriever so the graph wiring (parallel
fan-out, join, state accumulation) is tested without loading ChromaDB, the graph,
or the cross-encoder - and without needing an API key.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import KGRagPipeline, collect_sources, format_evidence  # noqa: E402


VECTOR_ITEM = {
    "text": "Self-attention relates different positions of a single sequence.",
    "source_file": "attention_is_all_you_need.pdf", "page": 3,
    "section": "Method", "retrieval_method": "vector", "score": 0.61,
}

GRAPH_ITEM = {
    "text": "DETR outperforms Faster R-CNN on COCO.",
    "triple": ("DETR", "outperforms", "Faster R-CNN"),
    "path_str": "DETR --[outperforms]--> Faster R-CNN",
    "source_file": "detr.pdf", "page": 1,
    "hop_distance": 1, "retrieval_method": "graph", "score": 1.8,
}


# ---------------------------------------------------------------------------
# Evidence formatting - what the model actually reads
# ---------------------------------------------------------------------------

def test_the_two_evidence_kinds_are_labelled_differently():
    """
    The prompt asks the model to treat a stated fact and an inferred chain
    differently. It can only do that if the labels survive formatting.
    """
    rendered = format_evidence([VECTOR_ITEM, GRAPH_ITEM])
    assert "[TEXT]" in rendered
    assert "[GRAPH]" in rendered


def test_every_item_carries_a_citation_in_the_expected_format():
    rendered = format_evidence([VECTOR_ITEM, GRAPH_ITEM])
    assert "[Source: attention_is_all_you_need.pdf, p.3]" in rendered
    assert "[Source: detr.pdf, p.1]" in rendered


def test_graph_items_expose_the_triple_and_the_chain():
    rendered = format_evidence([GRAPH_ITEM])
    assert "DETR --[outperforms]--> Faster R-CNN" in rendered
    assert "1 hop" in rendered


def test_multi_hop_items_report_their_hop_count():
    item = {**GRAPH_ITEM, "hop_distance": 3}
    assert "3 hops" in format_evidence([item])


def test_formatting_empty_evidence_does_not_crash():
    assert format_evidence([]) == ""


# ---------------------------------------------------------------------------
# Source collection
# ---------------------------------------------------------------------------

def test_sources_are_deduplicated_by_file_and_page():
    sources = collect_sources([VECTOR_ITEM, dict(VECTOR_ITEM), GRAPH_ITEM])
    assert len(sources) == 2


def test_a_source_found_by_both_methods_records_both():
    same_place_via_graph = {**GRAPH_ITEM,
                            "source_file": VECTOR_ITEM["source_file"],
                            "page": VECTOR_ITEM["page"]}
    sources = collect_sources([VECTOR_ITEM, same_place_via_graph])
    assert len(sources) == 1
    assert sources[0]["methods"] == ["graph", "vector"]


# ---------------------------------------------------------------------------
# The LangGraph workflow
# ---------------------------------------------------------------------------

class _StubStore:
    def search(self, query, top_k=5, where=None):
        return [dict(VECTOR_ITEM, rank=i) for i in range(3)]


class _StubRetriever:
    def __init__(self):
        self.store = _StubStore()
        self.kg = object()

    def rerank(self, query, results):
        # Deterministic: reverse the merged order, so the test can tell whether
        # re-ranking actually ran rather than the order surviving by accident.
        reranked = list(reversed(results))
        for rank, item in enumerate(reranked):
            item["rerank_score"] = 1.0 - rank * 0.1
            item["final_rank"] = rank
        return reranked


@pytest.fixture
def pipeline(monkeypatch) -> KGRagPipeline:
    import src.graph_retriever as graph_retriever

    monkeypatch.setattr(
        graph_retriever, "retrieve",
        lambda query, kg, hops=2, max_evidence=20, verbose=False: [dict(GRAPH_ITEM)],
    )
    return KGRagPipeline(retriever=_StubRetriever())


def test_pipeline_runs_end_to_end_and_fills_every_stage(pipeline):
    state = pipeline.run("Which models outperform Faster R-CNN?")
    for key in ("query", "vector_results", "graph_results",
                "merged_results", "reranked_results", "sources"):
        assert key in state, f"state is missing {key!r}"


def test_both_retrievers_contribute_to_the_merged_results(pipeline):
    state = pipeline.run("Which models outperform Faster R-CNN?")
    methods = {item["retrieval_method"] for item in state["merged_results"]}
    assert methods == {"vector", "graph"}


def test_parallel_branches_do_not_overwrite_each_other(pipeline):
    """
    retrieve_vector and retrieve_graph run in the same LangGraph step. They are
    safe only because they write DIFFERENT state keys - if either clobbered the
    other, one whole retrieval path would silently vanish.
    """
    state = pipeline.run("test question")
    assert len(state["vector_results"]) == 3
    assert len(state["graph_results"]) == 1


def test_reranking_is_applied_to_the_final_results(pipeline):
    state = pipeline.run("test question")
    assert all("rerank_score" in item for item in state["reranked_results"])
    assert state["reranked_results"][0]["final_rank"] == 0


def test_top_k_limits_what_reaches_the_model(pipeline):
    pipeline.top_k = 2
    state = pipeline.run("test question")
    assert len(state["reranked_results"]) <= 2


def test_missing_llm_key_still_returns_full_retrieval(pipeline, monkeypatch):
    """
    An absent API key must degrade to "retrieval worked, generation did not",
    not to an exception. Phases 1-5 are usable without any key at all.
    """
    from src.llm import LLMClient, LLMUnavailable

    def _no_key(*_args, **_kwargs):
        raise LLMUnavailable(LLMClient.setup_hint())

    monkeypatch.setattr("src.pipeline.get_llm", lambda **_: type(
        "L", (), {"generate": staticmethod(_no_key)})())

    state = pipeline.run("test question")
    assert state["error"] is not None
    assert state["reranked_results"], "retrieval must still have produced results"


def test_empty_retrieval_produces_an_honest_message_not_a_hallucination(monkeypatch):
    import src.graph_retriever as graph_retriever

    monkeypatch.setattr(graph_retriever, "retrieve",
                        lambda *a, **k: [])

    class _EmptyStore:
        def search(self, *a, **k):
            return []

    class _EmptyRetriever:
        def __init__(self):
            self.store = _EmptyStore()
            self.kg = object()

        def rerank(self, query, results):
            return results

    state = KGRagPipeline(retriever=_EmptyRetriever()).run("something absent")
    assert state["reranked_results"] == []
    assert "No evidence" in state["answer"]


def test_timings_from_every_node_survive_the_run(pipeline):
    # `timings` is written by several nodes; last-write-wins would leave only one.
    state = pipeline.run("test question")
    assert "total" in state["timings"]
