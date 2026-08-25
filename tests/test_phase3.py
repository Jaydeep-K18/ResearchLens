"""
Phase 3 tests - REBEL markup decoding, sentence filtering, domain relation extraction.

None of these load REBEL itself (1.6GB, ~1.3s per sentence). The parser is tested
against captured markup strings, which is both faster and a better test: it pins
the FORMAT contract, so a transformers upgrade that changes decoding behaviour
fails here loudly rather than silently producing zero triples.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.knowledge_extractor import is_useful_sentence  # noqa: E402
from src.relation_extractor import _parse_generated, extract_domain_triples  # noqa: E402


def triples_of(nlp, sentence: str) -> set[tuple[str, str, str]]:
    return {
        (t["subject"], t["relation"], t["object"])
        for t in extract_domain_triples(nlp(sentence))
    }


# ---------------------------------------------------------------------------
# REBEL markup parsing
# ---------------------------------------------------------------------------

def test_parses_a_single_triple():
    markup = "<s><triplet> DETR <subj> Faster R-CNN <obj> outperforms</s>"
    assert _parse_generated(markup) == [
        {"subject": "DETR", "relation": "outperforms", "object": "Faster R-CNN"}
    ]


def test_parses_multiple_objects_sharing_one_subject():
    # REBEL reuses a <triplet> block for several (object, relation) pairs.
    markup = ("<triplet> DETR <subj> Faster R-CNN <obj> outperforms "
              "<subj> transformer <obj> uses")
    parsed = _parse_generated(markup)
    assert len(parsed) == 2
    assert {t["object"] for t in parsed} == {"Faster R-CNN", "transformer"}
    assert all(t["subject"] == "DETR" for t in parsed)


def test_parses_multiple_triplet_blocks():
    markup = ("<triplet> DETR <subj> COCO <obj> evaluated on "
              "<triplet> ResNet <subj> He <obj> developer")
    parsed = _parse_generated(markup)
    assert len(parsed) == 2
    assert parsed[0]["subject"] == "DETR"
    assert parsed[1]["subject"] == "ResNet"


def test_special_tokens_are_stripped_from_field_values():
    markup = "<s><pad><triplet> ResNet <subj> He <obj> developer</s>"
    parsed = _parse_generated(markup)
    assert parsed[0]["subject"] == "ResNet"
    assert parsed[0]["object"] == "He"
    assert parsed[0]["relation"] == "developer"


def test_incomplete_generation_yields_nothing_rather_than_garbage():
    # A truncated generation must not produce a triple with an empty field -
    # that would become an unnamed node in the graph.
    assert _parse_generated("<triplet> DETR <subj> Faster R-CNN") == []
    assert _parse_generated("") == []
    assert _parse_generated("just some words with no markup") == []


def test_duplicate_triples_from_different_beams_are_collapsed():
    markup = ("<triplet> DETR <subj> COCO <obj> evaluated on "
              "<triplet> DETR <subj> COCO <obj> evaluated on")
    assert len(_parse_generated(markup)) == 1


# ---------------------------------------------------------------------------
# Sentence filtering
# ---------------------------------------------------------------------------

def test_normal_prose_is_kept():
    assert is_useful_sentence(
        "DETR uses a transformer encoder-decoder architecture and matches "
        "Faster R-CNN on the COCO object detection benchmark."
    )


def test_too_short_and_too_long_sentences_are_dropped():
    assert not is_useful_sentence("We propose DETR.")
    assert not is_useful_sentence("word " * 150)


def test_table_rows_are_dropped():
    # These pass a word count but contain no relation, and each one would cost
    # ~1.3 seconds of REBEL time for nothing.
    assert not is_useful_sentence("37.4 39.8 41.2 42.0 43.5 44.1 45.0 46.2 47.1 48.0")
    assert not is_useful_sentence("AP AP50 AP75 APS APM APL 38.2 59.1 41.0 20.1 41.1 50.2")


def test_symbol_heavy_equation_fragments_are_dropped():
    assert not is_useful_sentence("x = W_q h_t + b_q , k = W_k h_s + b_k , 0 < i < n , j += 1")


# ---------------------------------------------------------------------------
# Domain relation extraction - the relations REBEL structurally cannot produce
# ---------------------------------------------------------------------------

def test_extracts_outperforms():
    assert ("ResNet-101", "outperforms", "VGG-16") in triples_of(
        _shared_nlp(), "ResNet-101 outperforms VGG-16 on ImageNet."
    )


def test_outperforms_survives_a_shared_subject_conjunction():
    """
    Regression: in "DETR uses X and outperforms Y", spaCy attaches "outperforms"
    as a conj with NO subject of its own - the subject is stated once and shared.
    Without the conjunction walk the outperforms triple vanished entirely, which
    is the single relation type this extractor exists to capture.
    """
    found = triples_of(_shared_nlp(),
                       "DETR uses a transformer architecture and outperforms Faster R-CNN on COCO.")
    assert ("DETR", "outperforms", "Faster R-CNN") in found
    assert ("DETR", "uses", "transformer architecture") in found


def test_passive_voice_direction_is_flipped_correctly():
    # "X was proposed by Y" means Y proposed X, not X proposed Y. Getting this
    # backwards would make every authorship edge in the graph point the wrong way.
    found = triples_of(_shared_nlp(), "Faster R-CNN was proposed by Ren et al.")
    assert ("Ren et al.", "proposes", "Faster R-CNN") in found
    assert ("Faster R-CNN", "proposes", "Ren et al.") not in found


def test_prepositional_object_attached_to_the_direct_object_is_found():
    """
    Regression: spaCy attaches "on" to the direct object, not the verb, in
    "We evaluate Mask R-CNN on the COCO dataset". Searching only the verb's
    children dropped nearly every evaluated_on/trained_on triple.
    """
    assert ("Mask R-CNN", "evaluated_on", "COCO dataset") in triples_of(
        _shared_nlp(), "We evaluate Mask R-CNN on the COCO dataset."
    )


def test_pronoun_subjects_are_rejected():
    # "We" / "it" would become graph nodes connecting unrelated papers.
    for subject, _relation, _obj in triples_of(_shared_nlp(), "We use a residual connection."):
        assert subject.lower() not in {"we", "it", "they"}


def test_modifier_prepositions_are_not_mistaken_for_objects():
    """
    Regression: allowing prepositional objects for every relation produced
        (Vaswani, proposes, 2017)           from "... in 2017"
        (ResNet-101, outperforms, ImageNet) from "... on ImageNet"
    where the phrase modifies the claim rather than being its object.
    """
    found = triples_of(_shared_nlp(),
                       "Vaswani proposed the Transformer architecture in 2017.")
    assert not any(obj == "2017" for _s, _r, obj in found)

    found = triples_of(_shared_nlp(), "ResNet-101 outperforms VGG-16 on ImageNet.")
    assert not any(obj == "ImageNet" for _s, _r, obj in found)


def test_comparative_adjective_both_parse_shapes():
    """
    spaCy parses "than X" two different ways, and both appear in one sentence:
      "faster than YOLOv3"     -> than=mark, object is an advcl
      "more accurate than SSD" -> than=prep, object is a pobj
    Handling only one shape silently drops half of all comparisons.
    """
    found = triples_of(
        _shared_nlp(),
        "Our model is faster than YOLOv3 while being more accurate than SSD.",
    )
    objects = {obj for _s, relation, obj in found if relation == "outperforms"}
    assert "YOLOv3" in objects


def test_et_al_is_kept_with_the_author_name():
    # spaCy chunks "He et al." down to "He", which would collide with the pronoun.
    found = triples_of(_shared_nlp(), "He et al. introduced deep residual learning.")
    assert any(subject == "He et al." for subject, _r, _o in found)


def test_trained_on_is_extracted():
    assert ("Vision Transformer", "trained_on", "JFT-300M") in triples_of(
        _shared_nlp(), "The Vision Transformer is trained on JFT-300M."
    )


_NLP = None


def _shared_nlp():
    """Load spaCy once for the whole module (pytest fixtures cannot be used inline)."""
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
    return _NLP
