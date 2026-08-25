"""
Phase 1 tests - ingestion (extraction + cleaning) and chunking.

These are deliberately fast and PDF-free: every test builds its own synthetic
document, so the suite runs in under a second and does not depend on what
happens to be sitting in data/raw/. Two of them (test_semantic_chunker_does_not_
runaway, test_reference_removal_keeps_appendix) encode real bugs found while
building this phase - they exist so those bugs cannot come back.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunker import (  # noqa: E402
    build_text_with_page_map,
    chunk_fixed,
    chunk_semantic,
    page_for_offset,
)
from src.ingestion import (  # noqa: E402
    clean_text,
    find_reference_span,
    find_repeating_lines,
    strip_references,
)


def make_doc(pages: list[str], filename: str = "test.pdf") -> dict:
    return {
        "filename": filename,
        "path": f"data/raw/{filename}",
        "page_count": len(pages),
        "raw_pages": pages,
        "pages": pages,
        "full_text": "\n\n".join(pages),
    }


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def test_hyphenation_across_line_break_is_rejoined():
    assert "segmentation" in clean_text("instance segmen-\ntation masks")


def test_ligatures_are_normalised_to_ascii():
    # PDFs store "fi"/"fl" as single glyphs; the graph matches entities by string,
    # so these must become plain letters or entities silently split in two.
    assert clean_text("ﬁne-tuning the ﬂow of EfﬁcientNet") == (
        "fine-tuning the flow of EfficientNet"
    )


def test_standalone_page_numbers_are_dropped():
    cleaned = clean_text("Real body sentence.\n\n7\n\nMore body text.")
    assert "\n7\n" not in cleaned
    assert "Real body sentence." in cleaned
    assert "More body text." in cleaned


def test_paragraph_breaks_survive_but_line_wrapping_does_not():
    # The blank line is load-bearing: chunk_semantic splits on it.
    cleaned = clean_text("First paragraph line one\nwrapped line two.\n\nSecond paragraph.")
    assert "\n\n" in cleaned
    assert "line one wrapped line two" in cleaned   # intra-paragraph newline -> space


def test_repeating_header_detected_across_pages():
    pages = [f"Swin Transformer Paper\nBody text for page {i}.\nFooter {i}" for i in range(6)]
    repeating = find_repeating_lines(pages)
    assert "swin transformer paper" in repeating
    assert "footer #" in repeating          # digits normalised to '#'


def test_repeating_lines_ignored_for_short_documents():
    # With 2 pages "appears on most pages" is meaningless - we would delete real text.
    assert find_repeating_lines(["Header\nBody one", "Header\nBody two"]) == set()


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------

def test_reference_removal_cuts_to_end_when_no_appendix():
    text = ("Body of the paper. " * 40) + "\n\nReferences\n\n[1] He et al. Deep residual learning."
    assert "He et al." not in strip_references(text)
    assert "Body of the paper." in strip_references(text)


def test_reference_removal_keeps_appendix():
    # Regression: cutting everything from "References" onward deleted ViT's
    # appendix, which is a third of that paper.
    text = (
        ("Body of the paper. " * 40)
        + "\n\nReferences\n\n[1] He et al. Deep residual learning.\n"
        + "\n\nAppendix\n\nExtra experimental detail worth keeping."
    )
    cleaned = strip_references(text)
    assert "He et al." not in cleaned
    assert "Extra experimental detail worth keeping." in cleaned


def test_reference_heading_in_body_text_is_ignored():
    # "References" near the start is a section list, not the bibliography.
    text = "References are discussed below.\n\n" + ("Real body content. " * 60)
    assert find_reference_span(text) is None


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def test_fixed_chunks_overlap_by_the_requested_amount():
    doc = make_doc(["word " * 600])
    chunks = chunk_fixed(doc, chunk_size=500, overlap=100, snap=False)
    assert len(chunks) > 2
    for previous, following in zip(chunks, chunks[1:]):
        shared = previous["end_char"] - following["start_char"]
        assert shared == 100, f"expected 100 chars of overlap, got {shared}"


def test_fixed_chunks_cover_the_whole_document():
    doc = make_doc(["word " * 600])
    chunks = chunk_fixed(doc, chunk_size=500, overlap=100, snap=False)
    assert chunks[0]["start_char"] == 0
    # No gaps: each chunk starts before the previous one ended.
    for previous, following in zip(chunks, chunks[1:]):
        assert following["start_char"] < previous["end_char"]


def test_overlap_must_be_smaller_than_chunk_size():
    doc = make_doc(["text " * 200])
    try:
        chunk_fixed(doc, chunk_size=100, overlap=100)
    except ValueError:
        return
    raise AssertionError("expected ValueError when overlap >= chunk_size")


def test_semantic_chunker_does_not_runaway_on_text_without_periods():
    # Regression: table rows have no sentence terminators, so the "last sentence"
    # was the whole chunk, which got carried forward and compounded. Mask R-CNN
    # went from 56K source characters to 516K of chunks.
    rows = "\n\n".join(f"AP{i} AP50 AP75 APS APM APL {i * 7} {i * 3}" for i in range(300))
    doc = make_doc([rows])
    chunks = chunk_semantic(doc, max_chunk_size=900)
    produced = sum(c["n_chars"] for c in chunks)
    assert produced < len(rows) * 2, (
        f"runaway duplication: {len(rows)} source chars produced {produced} chunk chars"
    )


def test_semantic_chunker_respects_max_size():
    paragraphs = "\n\n".join("This is a normal research sentence about detection. " * 3
                            for _ in range(40))
    doc = make_doc([paragraphs])
    for chunk in chunk_semantic(doc, max_chunk_size=900):
        assert chunk["n_chars"] <= 1300, f"chunk of {chunk['n_chars']} chars blew the budget"


def test_semantic_chunker_tags_sections_and_does_not_merge_across_them():
    doc = make_doc([
        "1. Introduction\n\n" + ("Intro sentence here. " * 20)
        + "\n\n2. Method\n\n" + ("Method sentence here. " * 20)
    ])
    chunks = chunk_semantic(doc, max_chunk_size=900)
    sections = {c["section"] for c in chunks}
    assert "Introduction" in sections and "Method" in sections
    for chunk in chunks:
        # No chunk may contain text from both sections.
        assert not ("Intro sentence" in chunk["text"] and "Method sentence" in chunk["text"])


# ---------------------------------------------------------------------------
# Page attribution (needed for "[Source: detr.pdf, p.4]" in Phase 6)
# ---------------------------------------------------------------------------

def test_page_numbers_are_one_indexed_and_correct():
    doc = make_doc(["page one text", "page two text", "page three text"])
    text, spans = build_text_with_page_map(doc)
    assert page_for_offset(0, spans) == 1
    assert page_for_offset(text.index("page two"), spans) == 2
    assert page_for_offset(text.index("page three"), spans) == 3


def test_blank_pages_are_skipped_in_the_page_map():
    # Pages emptied by reference removal must not shift later page numbers.
    doc = make_doc(["page one text", "", "page three text"])
    text, spans = build_text_with_page_map(doc)
    assert page_for_offset(text.index("page three"), spans) == 3


def test_every_chunk_carries_the_metadata_later_phases_need():
    doc = make_doc(["Some research content. " * 60], filename="detr.pdf")
    for chunk in chunk_fixed(doc):
        for key in ("chunk_id", "text", "source_file", "chunk_index",
                    "start_char", "end_char", "page", "strategy", "n_chars"):
            assert key in chunk, f"chunk is missing {key!r}"
        assert chunk["source_file"] == "detr.pdf"
        assert chunk["page"] >= 1


def test_chunk_ids_are_unique():
    doc = make_doc(["Some research content. " * 200])
    for strategy in (chunk_fixed, chunk_semantic):
        chunks = strategy(doc)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), f"{strategy.__name__} produced duplicate chunk_ids"
