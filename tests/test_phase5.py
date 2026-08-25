"""
Phase 5 tests - query understanding, merging, deduplication, re-ranking.

The merge logic is where two independently-built subsystems meet, which makes it
exactly the place where a silent scale bug can make one of them useless. Those
cases get the most attention here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph_retriever import (  # noqa: E402
    candidate_mentions,
    detect_relation_cues,
    resolve_entities,
    retrieve,
)
from src.hybrid_retriever import _normalise_scores, _text_overlap, deduplicate  # noqa: E402
from src.knowledge_graph import KnowledgeGraph  # noqa: E402


def triple(subject, relation, obj, source="paper.pdf", page=1):
    return {
        "subject": subject, "relation": relation, "object": obj,
        "source_file": source, "page": page,
        "sentence": f"{subject} {relation.replace('_', ' ')} {obj}.",
        "extractor": "domain",
    }


@pytest.fixture(scope="module")
def kg() -> KnowledgeGraph:
    return KnowledgeGraph.from_triples([
        triple("DETR", "outperforms", "Faster R-CNN", "detr.pdf", 1),
        triple("Mask R-CNN", "outperforms", "Faster R-CNN", "mask_rcnn.pdf", 1),
        triple("Ren et al.", "proposes", "Faster R-CNN", "faster_rcnn.pdf", 2),
        triple("He et al.", "proposes", "Mask R-CNN", "mask_rcnn.pdf", 1),
        triple("Faster R-CNN", "uses", "VGG-16", "faster_rcnn.pdf", 3),
        triple("Vision Transformer", "trained_on", "JFT-300M", "vit.pdf", 4),
        triple("Swin Transformer", "achieves", "87.3 top-1 accuracy", "swin.pdf", 5),
    ], verbose=False)


# ---------------------------------------------------------------------------
# Query understanding
# ---------------------------------------------------------------------------

def test_acronyms_and_model_names_are_found_as_candidates():
    """
    The whole reason candidate_mentions uses four strategies: spaCy NER misses
    DETR, YOLOv3 and COCO entirely (see notebooks/02_ner_demo.py). If entity
    extraction fails, the graph contributes nothing at all to the answer.
    """
    mentions = " | ".join(candidate_mentions("Which models outperform DETR on COCO?")).lower()
    assert "detr" in mentions
    assert "coco" in mentions


def test_hyphenated_and_versioned_names_survive():
    mentions = " | ".join(candidate_mentions("How does ResNet-50 compare to YOLOv3?")).lower()
    assert "resnet-50" in mentions
    assert "yolov3" in mentions


def test_question_words_are_not_treated_as_entities():
    mentions = {m.lower() for m in candidate_mentions("Which models are better?")}
    assert "which" not in mentions
    assert "what" not in mentions


def test_relation_cues_are_detected_from_phrasing():
    assert "outperforms" in detect_relation_cues("Which models outperform YOLO?")
    assert "outperforms" in detect_relation_cues("What beats Faster R-CNN?")
    assert "proposes" in detect_relation_cues("Who proposed DETR?")
    assert "trained_on" in detect_relation_cues("What was ViT trained on?")


def test_no_cues_for_a_neutral_question():
    assert detect_relation_cues("Tell me about DETR") == []


def test_entities_resolve_to_graph_nodes(kg):
    resolved = resolve_entities("Which models outperform Faster R-CNN?", kg)
    assert any(entry["display"] == "Faster R-CNN" for entry in resolved)


def test_misspelled_entity_still_resolves(kg):
    resolved = resolve_entities("what beats Faster RCNN", kg)
    assert any("Faster" in entry["display"] for entry in resolved)


# ---------------------------------------------------------------------------
# Graph retrieval and cue-driven ranking
# ---------------------------------------------------------------------------

def test_relation_cue_promotes_the_matching_edges(kg):
    """
    Without cue matching, asking "which models outperform X" returns X's entire
    neighbourhood - its authors, its backbone, its datasets - and the actual
    answer is buried. The cue boost is what makes the retrieval answer the
    question rather than describe the entity.
    """
    evidence = retrieve("Which models outperform Faster R-CNN?", kg, hops=2)
    assert evidence
    assert evidence[0]["relation"] == "outperforms"


def test_authorship_question_promotes_proposes_edges(kg):
    evidence = retrieve("Who proposed Faster R-CNN?", kg, hops=1)
    assert evidence
    assert evidence[0]["relation"] == "proposes"


def test_every_evidence_item_carries_a_citation(kg):
    for item in retrieve("Which models outperform Faster R-CNN?", kg, hops=2):
        assert item["source_file"]
        assert item["retrieval_method"] == "graph"
        assert item["text"]


def test_query_with_no_known_entity_returns_nothing(kg):
    assert retrieve("What is the boiling point of mercury?", kg, hops=2) == []


def test_two_hop_reaches_authors_of_competing_models(kg):
    """
    The multi-hop shape that matters:
        Faster R-CNN <--[outperforms]-- Mask R-CNN <--[proposes]-- He et al.
    Two hops from the entity in the question to an author never mentioned in it.
    """
    evidence = retrieve("Which models outperform Faster R-CNN and who wrote them?",
                        kg, hops=2)
    reached = {item["triple"][0] for item in evidence} | {item["triple"][2] for item in evidence}
    assert "He et al." in reached


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def test_score_normalisation_maps_a_group_onto_zero_to_one():
    items = [{"score": 0.4}, {"score": 0.6}, {"score": 0.5}]
    _normalise_scores(items)
    assert items[0]["normalised_score"] == 0.0
    assert items[1]["normalised_score"] == 1.0
    assert 0 < items[2]["normalised_score"] < 1


def test_normalisation_handles_identical_scores_without_dividing_by_zero():
    items = [{"score": 0.5}, {"score": 0.5}]
    _normalise_scores(items)
    assert all(item["normalised_score"] == 1.0 for item in items)


def test_normalisation_makes_the_two_retrievers_comparable():
    """
    Regression guard for a units bug. Vector similarities land around 0.4-0.7;
    graph scores are a hand-built sum that can exceed 2. Merging the RAW numbers
    would rank graph evidence above vector evidence purely because of how each
    formula is scaled - not because it is more relevant.
    """
    vector = [{"score": 0.65}, {"score": 0.60}, {"score": 0.55}]
    graph = [{"score": 2.4}, {"score": 1.1}, {"score": 0.9}]

    _normalise_scores(vector)
    _normalise_scores(graph)

    # Best of each kind must tie at the top, not be decided by raw magnitude.
    assert max(v["normalised_score"] for v in vector) == 1.0
    assert max(g["normalised_score"] for g in graph) == 1.0


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def test_text_overlap_is_symmetric_and_bounded():
    assert _text_overlap("a b c", "a b c") == 1.0
    assert _text_overlap("a b c", "x y z") == 0.0
    assert _text_overlap("", "anything") == 0.0


def test_near_duplicates_from_the_same_file_are_collapsed():
    """
    Duplication is guaranteed by construction: chunks overlap by 100 characters,
    and graph evidence sentences often sit inside retrieved chunks. Left in, the
    top-8 can be five restatements of one fact - which wastes context AND makes
    one source look like several agreeing.
    """
    results = [
        {"text": "DETR outperforms Faster R-CNN on the COCO benchmark clearly",
         "source_file": "detr.pdf", "retrieval_method": "vector"},
        {"text": "DETR outperforms Faster R-CNN on the COCO benchmark",
         "source_file": "detr.pdf", "retrieval_method": "graph"},
    ]
    assert len(deduplicate(results)) == 1


def test_deduplication_records_the_other_method_that_found_it():
    # Agreement between the two retrievers is useful signal, not noise to discard.
    results = [
        {"text": "DETR outperforms Faster R-CNN on the COCO benchmark clearly",
         "source_file": "detr.pdf", "retrieval_method": "vector"},
        {"text": "DETR outperforms Faster R-CNN on the COCO benchmark",
         "source_file": "detr.pdf", "retrieval_method": "graph"},
    ]
    kept = deduplicate(results)
    assert "graph" in kept[0]["also_found_by"]


def test_identical_text_from_different_files_is_kept():
    # Two papers making the same claim is corroboration, not duplication.
    results = [
        {"text": "ResNet uses residual connections", "source_file": "resnet.pdf",
         "retrieval_method": "vector"},
        {"text": "ResNet uses residual connections", "source_file": "detr.pdf",
         "retrieval_method": "vector"},
    ]
    assert len(deduplicate(results)) == 2


def test_genuinely_different_passages_are_both_kept():
    results = [
        {"text": "DETR uses a transformer encoder decoder architecture",
         "source_file": "detr.pdf", "retrieval_method": "vector"},
        {"text": "The bipartite matching loss forces unique predictions",
         "source_file": "detr.pdf", "retrieval_method": "vector"},
    ]
    assert len(deduplicate(results)) == 2


# ---------------------------------------------------------------------------
# Cross-encoder re-ranking (loads the real ~80MB model)
# ---------------------------------------------------------------------------

def test_cross_encoder_ranks_the_relevant_passage_first():
    """
    The point of Step 5.3: a cross-encoder reads query and passage TOGETHER, so
    it can judge relevance an independently-computed embedding pair cannot.
    """
    from src.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever.__new__(HybridRetriever)   # skip loading store/graph
    retriever.cross_encoder_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    results = [
        {"text": "We use a batch size of 256 and a weight decay of 0.0001.",
         "source_file": "a.pdf", "retrieval_method": "vector"},
        {"text": "Self-attention relates different positions of a single sequence "
                 "in order to compute a representation of that sequence.",
         "source_file": "b.pdf", "retrieval_method": "vector"},
        {"text": "Images are resized so the shorter side is 600 pixels.",
         "source_file": "c.pdf", "retrieval_method": "vector"},
    ]

    reranked = retriever.rerank("What is self-attention?", results)
    assert reranked[0]["source_file"] == "b.pdf"
    assert reranked[0]["final_rank"] == 0
    assert all("rerank_score" in item for item in reranked)
