"""
Phase 4 tests - graph construction, entity resolution, traversal, persistence.

Built on a small hand-written triple set so every assertion is about a fact we
can verify by eye. The traversal tests encode the project's core claim: that a
3-hop chain crossing three documents is reachable, and that edge direction
survives the walk.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.knowledge_graph import (  # noqa: E402
    KnowledgeGraph,
    format_path,
    is_valid_entity,
    normalise_entity,
)


def triple(subject, relation, obj, source="paper.pdf", page=1, extractor="domain"):
    return {
        "subject": subject, "relation": relation, "object": obj,
        "source_file": source, "page": page,
        "sentence": f"{subject} {relation} {obj}.", "extractor": extractor,
    }


# The chain this corpus is built to exercise:
#   DETR --outperforms--> Faster R-CNN --proposes(rev)--> Ren et al. --> segmentation
TRIPLES = [
    triple("DETR", "outperforms", "Faster R-CNN", "detr.pdf", 1),
    triple("Ren et al.", "proposes", "Faster R-CNN", "faster_rcnn.pdf", 2),
    triple("Ren et al.", "introduces", "instance segmentation", "mask_rcnn.pdf", 3),
    triple("the DETR model", "uses", "Transformer", "detr.pdf", 2),
    triple("A. Vaswani", "proposes", "Transformer", "attention.pdf", 1),
    triple("Vaswani", "introduces", "self-attention", "attention.pdf", 2),
    triple("YOLOv3", "outperforms", "SSD", "yolov3.pdf", 1),
    triple("YOLO", "uses", "Darknet", "yolov3.pdf", 1),
    triple("Swin Transformer", "trained_on", "ImageNet", "swin.pdf", 4),
    # Junk that must be filtered out:
    triple("we", "uses", "it"),
    triple("37.4", "achieves", "39.8"),
    triple("the", "uses", "a"),
]


@pytest.fixture(scope="module")
def kg() -> KnowledgeGraph:
    return KnowledgeGraph.from_triples(TRIPLES, verbose=False)


# ---------------------------------------------------------------------------
# Entity normalisation and validation
# ---------------------------------------------------------------------------

def test_normalisation_collapses_surface_variants():
    assert normalise_entity("The DETR model") == "detr model"
    assert normalise_entity("  DETR.  ") == "detr"
    assert normalise_entity('"Faster R-CNN"') == "faster r-cnn"


def test_stopwords_numbers_and_fragments_are_rejected():
    assert not is_valid_entity("we")
    assert not is_valid_entity("the")
    assert not is_valid_entity("37.4")
    assert not is_valid_entity("2015")
    assert not is_valid_entity("a much longer phrase that is really a whole sentence fragment here")
    assert is_valid_entity("DETR")
    assert is_valid_entity("Faster R-CNN")


def test_junk_triples_never_reach_the_graph(kg):
    for junk in ("we", "it", "37.4", "39.8", "the"):
        assert kg.find_entity(junk) != junk or junk not in kg.graph


# ---------------------------------------------------------------------------
# Entity resolution: merge vs variant
# ---------------------------------------------------------------------------

def test_spelling_variants_of_one_person_merge_into_one_node(kg):
    # "A. Vaswani" and "Vaswani" are one person written two ways.
    assert kg.find_entity("A. Vaswani") == kg.find_entity("Vaswani")


def test_versioned_models_stay_separate(kg):
    """
    The distinction that matters: YOLO and YOLOv3 are DIFFERENT MODELS with
    different accuracy and speed. Merging them - which naive fuzzy matching does,
    since the strings are 89% similar - would make "which models outperform
    YOLOv3?" return YOLOv3's own results.
    """
    assert kg.find_entity("YOLO") != kg.find_entity("YOLOv3")


def test_versioned_models_are_still_linked_as_variants(kg):
    yolo = kg.find_entity("YOLO")
    yolov3 = kg.find_entity("YOLOv3")
    relations = {
        data["relation"]
        for _s, _t, data in kg.graph.edges(data=True)
        if {_s, _t} == {yolo, yolov3}
    }
    assert "variant_of" in relations


def test_fuzzy_lookup_finds_a_misspelled_entity(kg):
    # A user typing "Faster RCNN" must still reach the "faster r-cnn" node, or
    # the graph contributes nothing to that question.
    assert kg.find_entity("Faster RCNN") == kg.find_entity("Faster R-CNN")


def test_aliases_are_recorded_on_the_node(kg):
    node = kg.graph.nodes[kg.find_entity("Vaswani")]
    assert any("Vaswani" in alias for alias in node["aliases"])


# ---------------------------------------------------------------------------
# Traversal - the core capability
# ---------------------------------------------------------------------------

def test_one_hop_finds_directly_stated_facts(kg):
    neighbours = kg.get_neighbors("DETR", hops=1)
    reached = {n["display"] for n in neighbours if n["hop_distance"] == 1}
    assert "Faster R-CNN" in reached


def test_three_hop_chain_crosses_three_documents(kg):
    """
    THE project's central claim, as an assertion.

        DETR --[outperforms]--> Faster R-CNN --[proposes]--> Ren et al.
             --[introduces]--> instance segmentation

    Those three facts live in three different papers. No chunk contains the
    chain, so no vector search at any chunk size can retrieve it.
    """
    neighbours = kg.get_neighbors("DETR", hops=3, max_results=500)
    reached = {n["display"] for n in neighbours}
    assert "instance segmentation" in reached

    hop_counts = {n["display"]: n["hop_distance"] for n in neighbours}
    assert hop_counts["instance segmentation"] == 3

    chain = next(n for n in neighbours if n["display"] == "instance segmentation")
    files = {step["source_file"] for step in chain["path"]}
    assert len(files) == 3, f"expected a chain across 3 papers, got {files}"


def test_hop_distance_is_the_shortest_route(kg):
    neighbours = kg.get_neighbors("DETR", hops=3, max_results=500)
    for neighbour in neighbours:
        assert neighbour["hop_distance"] == len(neighbour["path"])


def test_every_hop_carries_its_own_citation(kg):
    # Phase 6 must cite graph-derived claims exactly as it cites chunks.
    for neighbour in kg.get_neighbors("DETR", hops=2, max_results=100):
        for step in neighbour["path"]:
            if step["relation"] == "variant_of":
                continue
            assert step["source_file"], "a traversal step lost its source file"
            assert step["sentence"], "a traversal step lost its evidence sentence"


def test_edge_direction_is_preserved_through_traversal(kg):
    """
    (DETR outperforms Faster R-CNN) and (Faster R-CNN outperforms DETR) are
    opposite claims. If direction were lost, "which models beat DETR?" would
    confidently answer "Faster R-CNN" - cited, and exactly backwards.
    """
    rendered = [
        format_path(kg, n["path"])
        for n in kg.get_neighbors("DETR", hops=1)
        if n["display"] == "Faster R-CNN"
    ]
    assert any("--[outperforms]--> Faster R-CNN" in text for text in rendered)
    assert not any("<--[outperforms]-- Faster R-CNN" in text for text in rendered)


def test_find_path_returns_the_connecting_chain(kg):
    path = kg.find_path("DETR", "instance segmentation")
    assert path is not None
    assert len(path) == 3
    assert path[0]["relation"] == "outperforms"


def test_find_path_returns_none_for_unrelated_entities(kg):
    assert kg.find_path("DETR", "a thing that is definitely not in this graph") is None


def test_unknown_entity_yields_no_neighbours(kg):
    assert kg.get_neighbors("quantum chromodynamics", hops=2) == []


def test_subgraph_is_bounded_by_max_nodes(kg):
    subgraph = kg.get_subgraph("DETR", hops=3, max_nodes=4)
    assert subgraph.number_of_nodes() <= 4


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_save_and_load_round_trip(kg, tmp_path):
    # networkx 3.x REMOVED write_gpickle/read_gpickle; we use plain pickle and
    # convert the set-valued attributes. This guards that conversion.
    path = tmp_path / "graph.gpickle"
    kg.save(path)

    reloaded = KnowledgeGraph.load(path)
    assert reloaded.graph.number_of_nodes() == kg.graph.number_of_nodes()
    assert reloaded.graph.number_of_edges() == kg.graph.number_of_edges()
    assert isinstance(reloaded.graph.nodes[reloaded.find_entity("DETR")]["aliases"], set)
    assert reloaded.find_path("DETR", "instance segmentation") is not None


def test_stats_report_the_expected_shape(kg):
    stats = kg.stats()
    assert stats["nodes"] > 0 and stats["edges"] > 0
    assert "outperforms" in stats["relations"]
    assert len(stats["top_nodes"]) > 0
