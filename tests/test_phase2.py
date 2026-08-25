"""
Phase 2 tests - vector store behaviour and LLM client configuration.

These build a throwaway ChromaDB in a temp directory, so they never touch
data/chroma_db/. They do load the real embedding model (~90MB, cached after the
first run) because the properties worth testing - that similar text actually
scores higher - are meaningless against a fake embedder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm import LLMClient, LLMUnavailable, _real_key  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402

CHUNKS = [
    {
        "chunk_id": "yolo.pdf::fixed::0000",
        "text": "YOLO is a real-time object detection system that processes images in a single pass.",
        "source_file": "yolo.pdf", "chunk_index": 0, "page": 1,
        "section": "Introduction", "strategy": "fixed", "start_char": 0, "end_char": 83,
    },
    {
        "chunk_id": "transformer.pdf::fixed::0000",
        "text": "The Transformer uses self-attention mechanisms to weigh the importance of input tokens.",
        "source_file": "transformer.pdf", "chunk_index": 0, "page": 3,
        "section": "Method", "strategy": "fixed", "start_char": 0, "end_char": 87,
    },
    {
        "chunk_id": "cooking.pdf::fixed::0000",
        "text": "Mash three ripe bananas, add flour and sugar, then bake for forty minutes.",
        "source_file": "cooking.pdf", "chunk_index": 0, "page": 1,
        "section": "", "strategy": "fixed", "start_char": 0, "end_char": 73,
    },
]


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> VectorStore:
    directory = tmp_path_factory.mktemp("chroma_test")
    vector_store = VectorStore(persist_dir=directory, collection_name="test_chunks")
    vector_store.add_chunks(CHUNKS, show_progress=False)
    return vector_store


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def test_all_chunks_are_stored(store):
    assert store.count() == len(CHUNKS)


def test_adding_the_same_chunks_again_upserts_rather_than_duplicates(store):
    # chunk_id is the ChromaDB id, so re-ingestion must be idempotent - otherwise
    # every rebuild during development would silently double the corpus.
    before = store.count()
    store.add_chunks(CHUNKS, show_progress=False)
    assert store.count() == before


def test_metadata_needed_for_citation_survives_the_round_trip(store):
    hit = store.search("self-attention over input tokens", top_k=1)[0]
    assert hit["source_file"] == "transformer.pdf"
    assert hit["page"] == 3                    # required for "[Source: x.pdf, p.3]"
    assert hit["section"] == "Method"
    assert hit["retrieval_method"] == "vector"


# ---------------------------------------------------------------------------
# Search quality
# ---------------------------------------------------------------------------

def test_semantically_related_query_ranks_first(store):
    # No shared words with the stored chunk - this only works if embeddings are
    # capturing meaning rather than matching tokens.
    hits = store.search("How fast can a network find objects in a picture?", top_k=3)
    assert hits[0]["source_file"] == "yolo.pdf"


def test_scores_are_similarities_not_distances(store):
    """
    Regression guard: Chroma returns DISTANCE (lower = closer) and we invert it
    to a similarity (higher = better). Getting this backwards would silently
    rank the worst chunk first everywhere in the system.
    """
    hits = store.search("real-time object detection", top_k=3)
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True), "results must be sorted best-first"
    assert hits[0]["score"] > hits[-1]["score"]
    assert hits[0]["source_file"] == "yolo.pdf"


def test_query_with_a_match_outscores_a_query_with_none(store):
    # NOTE: "banana recipe" would be the obvious off-topic probe, but this
    # fixture deliberately contains a banana chunk, so that query legitimately
    # scores ~0.74. The off-corpus query has to be genuinely absent from all
    # three chunks for the comparison to mean anything.
    matched = store.search("object detection in real time", top_k=1)[0]["score"]
    unmatched = store.search("lattice gauge theory in quantum chromodynamics",
                             top_k=1)[0]["score"]
    assert matched > unmatched


def test_each_query_finds_its_own_topic(store):
    """Each of the three stored topics is retrievable by a paraphrase of itself."""
    expected = {
        "how quickly can a model spot objects": "yolo.pdf",
        "weighting the importance of input tokens": "transformer.pdf",
        "baking with mashed fruit and flour": "cooking.pdf",
    }
    for query, source in expected.items():
        assert store.search(query, top_k=1)[0]["source_file"] == source, query


def test_search_always_returns_k_results_even_when_nothing_is_relevant(store):
    """
    The property that makes basic RAG fail silently, pinned as a test.

    A vector database returns its k NEAREST neighbours. "Nearest" does not mean
    "near". Nothing in the search stage knows the corpus has no answer - which is
    exactly why Phase 6's prompt has to instruct the model to admit ignorance.
    """
    hits = store.search("quantum chromodynamics lattice gauge theory", top_k=3)
    assert len(hits) == 3
    assert all(h["text"] for h in hits)


def test_empty_store_returns_no_results(tmp_path):
    empty = VectorStore(persist_dir=tmp_path / "empty", collection_name="empty")
    assert empty.search("anything at all", top_k=5) == []


def test_metadata_filter_restricts_results(store):
    hits = store.search("detection", top_k=5, where={"source_file": "cooking.pdf"})
    assert all(h["source_file"] == "cooking.pdf" for h in hits)


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

def test_placeholder_keys_are_treated_as_missing(monkeypatch):
    # .env ships with "your_key_here". Treating that as a real key would produce
    # a confusing 401 instead of the setup instructions.
    monkeypatch.setenv("GEMINI_API_KEY", "your_key_here")
    assert _real_key("GEMINI_API_KEY") is None

    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    assert _real_key("GEMINI_API_KEY") is None

    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyRealLookingKey123")
    assert _real_key("GEMINI_API_KEY") == "AIzaSyRealLookingKey123"


def test_client_without_keys_raises_actionable_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "your_key_here")
    monkeypatch.setenv("GROQ_API_KEY", "your_key_here")

    client = LLMClient()
    client.gemini_key = None       # bypass any real .env loaded at import time
    client.groq_key = None

    assert not client.available
    with pytest.raises(LLMUnavailable) as excinfo:
        client.generate("hello")
    assert "aistudio.google.com" in str(excinfo.value)
