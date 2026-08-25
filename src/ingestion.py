"""
Phase 1, Steps 1.1 + 1.2 - PDF text extraction and cleaning.

WHAT THIS MODULE DOES
---------------------
Turns a folder of PDFs into clean, structured text that every later phase
consumes. It is the very first link in the chain, and the single highest-leverage
file in this project:

    bad text here  ->  bad chunks    ->  bad embeddings ->  bad retrieval
                   ->  bad sentences ->  bad triples    ->  a garbage graph

No embedding model, no re-ranker, and no LLM prompt can recover information that
was mangled at extraction time. Cleaning matters more than any fancy model you
add later.

THE DOCUMENT SCHEMA
-------------------
Produced by extract_pdf(), consumed by chunker.py and knowledge_extractor.py:

    {
      "filename":   "detr.pdf",        # basename - becomes the citation string
      "path":       "data/raw/detr.pdf",
      "page_count": 26,
      "pages":      ["page 1 text", "page 2 text", ...],   # cleaned, 0-indexed
      "full_text":  "the whole cleaned document as one string",
      "raw_pages":  [...],             # pre-cleaning, kept for before/after diffs
    }

We keep `pages` as a list (not just `full_text`) for one specific reason:
attribution. In Phase 6 the LLM must cite "[Source: detr.pdf, p.4]", so every
chunk and every extracted triple has to remember which page it came from. Losing
page boundaries here makes accurate citation impossible later.

WHY PyMuPDF: it is fast (C under the hood), handles the two-column academic
layouts arXiv papers use, and exposes per-block geometry - which we need, because
reading order in a two-column PDF is not the order the text is stored in.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf as fitz  # modern import name; "fitz" alias kept for readability

# Allow running this file directly (`python src/ingestion.py`) as well as
# importing it as `src.ingestion` from elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import RAW_DIR, banner, setup_console  # noqa: E402


# ---------------------------------------------------------------------------
# STEP 1.1 - extraction
# ---------------------------------------------------------------------------

def _extract_page_text(page: fitz.Page) -> str:
    """
    Pull text off a single page in correct READING order.

    The naive call - page.get_text() - returns text in whatever order the blocks
    happen to sit in the PDF content stream. For a one-column paper that is
    usually fine. For the two-column layout most CVPR/NeurIPS papers use, it
    frequently interleaves the left and right columns, producing sentences like
    "We propose DETR, a new to 91.2% mAP on COCO method that ...".

    That single failure would poison BOTH memories: the chunk text becomes
    gibberish, and REBEL in Phase 3 would extract relations between things that
    were never actually related.

    So instead we ask for the text BLOCKS (each with its bounding box), work out
    whether the page is one or two columns, and re-order accordingly.
    """
    # Each block is a tuple: (x0, y0, x1, y1, text, block_no, block_type)
    # block_type 0 = text, 1 = image. We only want text.
    blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
    if not blocks:
        return ""

    page_width = page.rect.width
    midline = page_width / 2

    # A block "spans" the page if it is wider than 60% of it - titles, abstracts
    # and full-width figures do this. Those are never column content.
    def is_spanning(block) -> bool:
        return (block[2] - block[0]) > 0.6 * page_width

    column_blocks = [b for b in blocks if not is_spanning(b)]

    # Two-column heuristic: we need a decent number of narrow blocks sitting
    # clearly on BOTH sides of the midline. If almost everything is on one side,
    # the page is single-column and naive top-to-bottom order is already correct.
    left = [b for b in column_blocks if (b[0] + b[2]) / 2 < midline]
    right = [b for b in column_blocks if (b[0] + b[2]) / 2 >= midline]
    two_column = len(left) >= 2 and len(right) >= 2

    if two_column:
        # Read the whole left column top-to-bottom, then the whole right one.
        # Spanning blocks (title/abstract) fold into the left stream at their
        # vertical position, which puts a page-top title before both columns -
        # exactly where a human would read it.
        def sort_key(block):
            center_x = (block[0] + block[2]) / 2
            in_right_column = (not is_spanning(block)) and center_x >= midline
            return (1 if in_right_column else 0, round(block[1], 1))
    else:
        def sort_key(block):
            return (0, round(block[1], 1), round(block[0], 1))

    ordered = sorted(blocks, key=sort_key)

    # Blocks are paragraph-ish units, so join them with a blank line. That blank
    # line is load-bearing: chunker.py's semantic chunker splits on exactly this
    # paragraph boundary.
    return "\n\n".join(b[4].strip() for b in ordered)


def extract_pdf(pdf_path: str | Path) -> dict | None:
    """
    Extract one PDF into the document dict described in the module docstring.

    Returns None (with a warning) if the file cannot be read - a single corrupted
    PDF in a 300-paper corpus must not kill an overnight batch job.
    """
    pdf_path = Path(pdf_path)
    try:
        with fitz.open(pdf_path) as doc:
            raw_pages = [_extract_page_text(page) for page in doc]
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any bad PDF is skippable
        print(f"  [WARN] could not read {pdf_path.name}: {type(exc).__name__}: {exc}")
        return None

    if not any(p.strip() for p in raw_pages):
        # Almost always a scanned / image-only PDF. The real fix is OCR, which is
        # out of scope; we just refuse to pass empty text downstream.
        print(f"  [WARN] {pdf_path.name} produced no text (scanned image PDF?) - skipping")
        return None

    return {
        "filename": pdf_path.name,
        "path": str(pdf_path),
        "page_count": len(raw_pages),
        "raw_pages": raw_pages,
        "pages": raw_pages,          # replaced by clean_document()
        "full_text": "\n\n".join(raw_pages),
    }


# ---------------------------------------------------------------------------
# STEP 1.2 - cleaning
# ---------------------------------------------------------------------------

# A line that is nothing but a page number: "7", "- 7 -", "Page 7", "7 of 26".
_PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:page\s+)?[-\[\(]?\s*\d{1,4}\s*(?:of\s+\d{1,4})?\s*[-\]\)]?\s*$",
    re.IGNORECASE,
)

# "References" / "Bibliography" alone on a line, optionally numbered.
_REFERENCES_RE = re.compile(
    r"^[ \t]*(?:\d+\.?[ \t]*|[IVXLC]+\.?[ \t]*)?(references|bibliography|works cited)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

# The start of an appendix / supplementary material section. Many papers put
# this AFTER the references, so "delete everything from References onwards"
# would silently throw away real content (ViT's appendix is a third of the paper).
_APPENDIX_RE = re.compile(
    r"^[ \t]*(?:"
    # "Appendix", "Appendix A", "Supplementary Material" - case-insensitive.
    r"(?i:appendix(?:[ \t]+[A-Z0-9][\w \-]{0,40})?"
    r"|supplementary(?:[ \t]+(?:material|information))?)"
    # "A. Additional Results" - lettered appendix sections. Case-SENSITIVE on
    # purpose: a lowercase "a. something" is a list item, not a heading.
    r"|[A-H]\.?[ \t]+[A-Z][A-Za-z ]{3,40}"
    r")[ \t]*$",
    re.MULTILINE,
)

# A word broken across a line by hyphenation: "detec-\ntion" -> "detection".
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\s*\n\s*(\w)")


def _normalise_for_comparison(line: str) -> str:
    """
    Collapse a line to a comparable fingerprint for header/footer detection.

    Digits become '#' so that "Page 3" and "Page 17" - the same footer on
    different pages - are recognised as the same thing.
    """
    line = re.sub(r"\d+", "#", line.strip().lower())
    return re.sub(r"\s+", " ", line)


def find_repeating_lines(pages: list[str], scan_lines: int = 3, threshold: float = 0.5) -> set[str]:
    """
    Detect running headers and footers by CROSS-PAGE REPETITION.

    The insight: you cannot tell a header from body text by looking at one page.
    "Swin Transformer: Hierarchical Vision Transformer" is a perfectly ordinary
    sentence - unless it appears at the top of all 14 pages, in which case it is a
    running header, and repeating it 14 times pollutes both memories (14
    near-identical chunks crowding out real content in the vector store, and junk
    edges in the graph).

    So we look at the first and last `scan_lines` lines of every page, fingerprint
    them, and flag anything appearing on more than `threshold` of the pages.
    """
    if len(pages) < 3:
        # With 1-2 pages "appears on most pages" is meaningless, and we would
        # start deleting real content.
        return set()

    counts: Counter[str] = Counter()
    for page in pages:
        lines = [ln for ln in page.split("\n") if ln.strip()]
        candidates = lines[:scan_lines] + lines[-scan_lines:]
        # set() so a line repeated twice on ONE page still counts once - we are
        # measuring "on how many pages", not "how many times overall".
        for line in {_normalise_for_comparison(ln) for ln in candidates}:
            if line:
                counts[line] += 1

    min_pages = max(3, int(len(pages) * threshold))
    return {line for line, count in counts.items() if count >= min_pages}


def find_reference_span(text: str) -> tuple[int, int] | None:
    """
    Locate the References/Bibliography section as a (start, end) character span.

    Why bother? A references section is 50-100 entries of author names and venue
    titles. Left in, it is actively harmful in both memories:

      - Vector store: dozens of chunks of pure citation soup. They match any
        author-flavoured query and drown out the actual findings.
      - Knowledge graph: REBEL happily turns "K. He, X. Zhang. Deep residual
        learning. CVPR 2016." into triples, filling the graph with thousands of
        low-value bibliographic edges that swamp the real relationships.

    Two document layouts have to be handled, and getting this wrong is costly in
    opposite directions:

      Layout A   [ body ][ references ]                 -> cut to the end
      Layout B   [ body ][ references ][ appendix ]     -> cut ONLY the middle

    Naively cutting everything from "References" onward destroys the appendix,
    and in a paper like ViT the appendix is a third of the content. So we find
    the heading, then look for an appendix heading after it and stop there.

    We ignore matches in the first 30% of the document, because "References"
    also appears in ordinary body text and section lists.

    Returns None if no references section is found.
    """
    if not text:
        return None

    earliest = int(len(text) * 0.30)
    matches = [m for m in _REFERENCES_RE.finditer(text) if m.start() >= earliest]
    if not matches:
        return None

    start = matches[0].start()

    # Does an appendix begin after the references? If so, the references section
    # ends there rather than at the end of the document.
    appendix = _APPENDIX_RE.search(text, pos=matches[0].end())
    end = appendix.start() if appendix else len(text)

    return (start, end)


def strip_references(text: str) -> str:
    """Remove the references section, keeping any appendix that follows it."""
    span = find_reference_span(text)
    if span is None:
        return text
    start, end = span
    return (text[:start].rstrip() + "\n\n" + text[end:].lstrip()).strip()


def clean_text(
    text: str,
    repeating_lines: set[str] | None = None,
    drop_references: bool = True,
) -> str:
    """
    Clean one page (or one whole document) of extracted PDF text.

    Order matters. We fix hyphenation BEFORE dropping lines, because a hyphen
    break can straddle a line we are about to delete; and we normalise whitespace
    LAST, once all the deletions have left ragged blank lines behind.
    """
    if not text:
        return ""

    # 0. Unicode normalisation (NFKC).
    #    PDFs store "fi" and "fl" as single LIGATURE glyphs, so extraction yields
    #    "Ef<U+FB01>cientNet" and "<U+FB02>ow", not "EfficientNet" and "flow".
    #    That is invisible when you print it but catastrophic downstream: the graph
    #    in Phase 4 matches entities by string, so "Ef<U+FB01>cientNet" and
    #    "EfficientNet" become two unrelated nodes, silently splitting the very
    #    relationships we built the graph to follow. NFKC folds ligatures into
    #    plain ASCII letters and normalises assorted lookalike dashes and quotes.
    text = unicodedata.normalize("NFKC", text)

    # 1. Rejoin words split across a line break: "segmen-\ntation" -> "segmentation".
    #    Trade-off: this also joins genuinely hyphenated compounds that happen to
    #    fall at a line end ("state-of-\nthe-art" -> "state-ofthe-art"). Rare, and
    #    far less damaging than leaving every long word fragmented.
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)

    # 2. Drop running headers/footers and standalone page numbers, line by line.
    kept: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if _PAGE_NUMBER_RE.match(stripped):
            continue
        if repeating_lines and _normalise_for_comparison(stripped) in repeating_lines:
            continue
        kept.append(stripped)
    text = "\n".join(kept)

    # 3. Optionally cut the references section.
    if drop_references:
        text = strip_references(text)

    # 4. Normalise whitespace - but PRESERVE the blank line between paragraphs.
    #    A single "\n" inside a paragraph is just PDF line-wrapping and becomes a
    #    space; "\n\n" is a real paragraph break, and the semantic chunker in
    #    chunker.py depends on it surviving this function.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)   # unwrap intra-paragraph newlines
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def clean_document(doc: dict) -> dict:
    """
    Clean a whole document, using cross-page evidence to find headers/footers.

    This is why cleaning is a DOCUMENT-level operation and not a per-page one:
    find_repeating_lines() can only work by comparing pages against each other.
    """
    repeating = find_repeating_lines(doc["raw_pages"])

    # Per page: strip headers/footers but do NOT cut references, because the
    # references section usually starts mid-page. We want to find its start once,
    # in the full document, rather than guess at it 26 separate times.
    cleaned_pages = [
        clean_text(page, repeating_lines=repeating, drop_references=False)
        for page in doc["raw_pages"]
    ]

    # Join the pages, remembering exactly where each one starts and ends in the
    # joined string. We need that mapping because the references section is found
    # in the JOINED text (it can straddle a page break), but has to be removed
    # from the individual PAGES too - otherwise `pages` and `full_text` disagree
    # and Phase 6 would cite a page number for text we already deleted.
    separator = "\n\n"
    spans: list[tuple[int, int]] = []
    cursor = 0
    for page in cleaned_pages:
        spans.append((cursor, cursor + len(page)))
        cursor += len(page) + len(separator)
    joined = separator.join(cleaned_pages)

    ref_span = find_reference_span(joined)
    if ref_span is None:
        trimmed_pages = list(cleaned_pages)
        full_text = joined
    else:
        ref_start, ref_end = ref_span
        trimmed_pages = []
        for page, (page_start, page_end) in zip(cleaned_pages, spans):
            # Subtract the reference span from this page's own span.
            overlap_start = max(page_start, ref_start)
            overlap_end = min(page_end, ref_end)
            if overlap_start >= overlap_end:
                trimmed_pages.append(page)             # page untouched
            else:
                head = page[: overlap_start - page_start]
                tail = page[overlap_end - page_start :]
                trimmed_pages.append((head + " " + tail).strip())
        full_text = (joined[:ref_start].rstrip() + separator + joined[ref_end:].lstrip()).strip()

    doc = dict(doc)
    doc["pages"] = trimmed_pages
    doc["full_text"] = full_text
    doc["repeating_lines_removed"] = sorted(repeating)
    doc["references_removed"] = ref_span is not None
    return doc


# ---------------------------------------------------------------------------
# Corpus-level entry point
# ---------------------------------------------------------------------------

def load_corpus(raw_dir: str | Path = RAW_DIR, clean: bool = True) -> list[dict]:
    """Extract (and by default clean) every PDF in a folder."""
    raw_dir = Path(raw_dir)
    pdf_paths = sorted(raw_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"[!] No PDFs found in {raw_dir}. Run: python scripts/fetch_papers.py")
        return []

    documents: list[dict] = []
    for pdf_path in pdf_paths:
        doc = extract_pdf(pdf_path)
        if doc is None:
            continue
        if clean:
            doc = clean_document(doc)
        documents.append(doc)
    return documents


def _print_before_after(doc: dict, n_chars: int = 400) -> None:
    """Show the effect of cleaning on one document - Step 1.2's deliverable."""
    raw = "\n\n".join(doc["raw_pages"])
    print(f"\n{'=' * 78}\n{doc['filename']}  ({doc['page_count']} pages)\n{'=' * 78}")

    removed = doc.get("repeating_lines_removed", [])
    if removed:
        print(f"Running headers/footers removed ({len(removed)}):")
        for line in removed[:5]:
            print(f"    - {line!r}")

    shrink = 100 * (1 - len(doc["full_text"]) / max(len(raw), 1))
    print(f"Characters: {len(raw):,} raw -> {len(doc['full_text']):,} clean  ({shrink:.1f}% removed)")
    print(f"\n--- BEFORE (first {n_chars} chars) ---\n{raw[:n_chars]}")
    print(f"\n--- AFTER  (first {n_chars} chars) ---\n{doc['full_text'][:n_chars]}")


if __name__ == "__main__":
    setup_console()
    print("Loading corpus from data/raw/ ...\n")
    docs = load_corpus(clean=True)

    if not docs:
        raise SystemExit(1)

    print(banner("CORPUS STATS"))
    print(f"{'file':<34}{'pages':>6}{'raw chars':>12}{'clean chars':>13}{'removed':>9}{'refs cut':>10}")
    print("-" * 84)
    total_pages = total_raw = total_clean = 0
    for doc in docs:
        raw_len = len("\n\n".join(doc["raw_pages"]))
        clean_len = len(doc["full_text"])
        pct = 100 * (1 - clean_len / max(raw_len, 1))
        refs = "yes" if doc.get("references_removed") else "NO"
        print(f"{doc['filename']:<34}{doc['page_count']:>6}{raw_len:>12,}{clean_len:>13,}"
              f"{pct:>8.1f}%{refs:>10}")
        total_pages += doc["page_count"]
        total_raw += raw_len
        total_clean += clean_len

    print("-" * 84)
    label = f"TOTAL ({len(docs)} docs)"
    print(f"{label:<34}{total_pages:>6}{total_raw:>12,}{total_clean:>13,}"
          f"{100 * (1 - total_clean / max(total_raw, 1)):>8.1f}%")

    # Step 1.2 asks for a before/after comparison on real papers.
    for doc in docs[:2]:
        _print_before_after(doc)
