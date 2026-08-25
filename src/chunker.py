"""
Phase 1, Step 1.3 - splitting cleaned documents into retrievable chunks.

WHY CHUNK AT ALL?
-----------------
When someone asks a question in Phase 2, we do not search whole papers - we
search small pieces of them. Two reasons:

1. PRECISION. A 55,000-character paper is "about" fifty things. Its embedding is
   the average of all fifty, which is a blurry vector close to nothing in
   particular. A 500-character chunk is about ONE thing, so its embedding is
   sharp and matches sharply.

2. CONTEXT BUDGET. In Phase 6 we hand the retrieved text to Gemini. We can
   afford to send 8 small chunks; we cannot send 8 whole papers.

The trade-off runs in both directions:
    chunks too big   -> imprecise matches, wasted context, diluted embeddings
    chunks too small -> the answer gets split across pieces and each piece,
                        read alone, is meaningless ("It achieves 42.0 AP." - what does?)

500 characters (~80 words, ~3 sentences) is a good default for research prose.

WHY OVERLAP?
------------
Consider this text with a hard cut at character 500:

    ... DETR matches the performance of Faster R-CNN. | The key idea is a
        set-based global loss that forces unique predictions ...

Chunk A ends with "matches the performance of Faster R-CNN." and chunk B starts
with "The key idea is...". A question like "how does DETR achieve parity with
Faster R-CNN?" matches NEITHER well - the claim is in A and the explanation is
in B, and no single chunk holds both.

With 100 characters of overlap, chunk B *starts* 100 characters earlier, so it
contains the tail of A. The boundary sentence now lives, complete, in at least
one chunk. Overlap is cheap insurance (20% more storage) against losing meaning
exactly at the seams.

THE CHUNK SCHEMA (consumed by vector_store.py in Phase 2)

    {
      "chunk_id":     "detr.pdf::fixed::0012",   # unique, stable, human-readable
      "text":         "...",
      "source_file":  "detr.pdf",
      "chunk_index":  12,
      "start_char":   6000,     # offset into the document's cleaned text
      "end_char":     6500,
      "page":         4,        # 1-indexed - REQUIRED for "[Source: detr.pdf, p.4]"
      "section":      "Method", # semantic chunker only; "" for fixed
      "strategy":     "fixed",
      "n_chars":      500,
    }

Note `page` is not in the original spec, but Phase 6 must cite page numbers, and
a chunk that has forgotten which page it came from can never be cited accurately.
The cheapest place to record it is here, at creation time.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import banner, setup_console  # noqa: E402

DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 100

# A section heading in a research paper: "3 Method", "3.1. Related Work",
# "Abstract", "Introduction". Used by the semantic chunker to avoid gluing the
# end of one section onto the start of the next.
_HEADING_RE = re.compile(
    r"^(?:"
    r"(?:\d+(?:\.\d+)*\.?\s+[A-Z][\w\- ]{2,60})"          # "3.1 Related Work"
    r"|(?:[A-Z][a-z]+(?:\s+[A-Za-z\-]+){0,4})"            # "Introduction", "Related Work"
    r")\s*$"
)

_KNOWN_HEADINGS = {
    "abstract", "introduction", "related work", "background", "method",
    "methods", "methodology", "approach", "experiments", "experimental setup",
    "results", "discussion", "conclusion", "conclusions", "ablation study",
    "implementation details", "limitations", "future work", "acknowledgments",
}


# ---------------------------------------------------------------------------
# Page mapping
# ---------------------------------------------------------------------------

def build_text_with_page_map(doc: dict) -> tuple[str, list[tuple[int, int, int]]]:
    """
    Rebuild the document text from its pages, remembering where each page lives.

    Returns (text, spans) where spans is a list of (start, end, page_number)
    with page_number 1-indexed.

    We deliberately rebuild from `doc["pages"]` rather than reuse
    `doc["full_text"]`: the two are equivalent in content, but only this version
    gives us exact character offsets per page, which is what turns a chunk offset
    back into "page 4" for citation.
    """
    parts: list[str] = []
    spans: list[tuple[int, int, int]] = []
    cursor = 0
    separator = "\n\n"

    for page_index, page_text in enumerate(doc["pages"]):
        if not page_text.strip():
            continue                       # blank page (e.g. it was all references)
        parts.append(page_text)
        spans.append((cursor, cursor + len(page_text), page_index + 1))
        cursor += len(page_text) + len(separator)

    return separator.join(parts), spans


def page_for_offset(offset: int, spans: list[tuple[int, int, int]]) -> int:
    """Map a character offset back to its 1-indexed page number."""
    for start, end, page_number in spans:
        if start <= offset < end:
            return page_number
    return spans[-1][2] if spans else 1


# ---------------------------------------------------------------------------
# Strategy 1: fixed-size overlapping chunks
# ---------------------------------------------------------------------------

def _snap_to_word_boundary(text: str, position: int, window: int = 40) -> int:
    """
    Nudge a cut point to the nearest sentence or word boundary.

    A blind cut at character 500 lands mid-word about one time in six, producing
    chunks that begin "...ntation masks are predicted". The embedding model has
    never seen "ntation" and wastes a token on garbage. Sliding the cut by a few
    characters costs nothing and keeps every chunk starting on a real word.

    Preference order: end of a sentence > end of a word > the raw position.
    """
    if position <= 0 or position >= len(text):
        return position

    lower = max(0, position - window)
    upper = min(len(text), position + window)
    segment = text[lower:upper]

    # Prefer a sentence end (". " / "? " / "! ") near the target.
    sentence_ends = [m.end() for m in re.finditer(r"[.!?]\s", segment)]
    if sentence_ends:
        best = min(sentence_ends, key=lambda e: abs((lower + e) - position))
        return lower + best

    # Otherwise the nearest whitespace.
    spaces = [m.start() for m in re.finditer(r"\s", segment)]
    if spaces:
        best = min(spaces, key=lambda s: abs((lower + s) - position))
        return lower + best + 1

    return position


def chunk_fixed(
    doc: dict,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    snap: bool = True,
) -> list[dict]:
    """
    Slide a fixed-size window over the document, stepping by (size - overlap).

    With size=500 and overlap=100 the windows are
        [0:500], [400:900], [800:1300], ...
    so each chunk shares its first 100 characters with the previous one.
    """
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")

    text, page_spans = build_text_with_page_map(doc)
    if not text.strip():
        return []

    step = chunk_size - overlap
    chunks: list[dict] = []
    start = 0
    index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        if snap and end < len(text):
            end = _snap_to_word_boundary(text, end)

        piece = text[start:end].strip()
        if len(piece) >= 50:            # skip slivers left by snapping
            chunks.append({
                "chunk_id": f"{doc['filename']}::fixed::{index:04d}",
                "text": piece,
                "source_file": doc["filename"],
                "chunk_index": index,
                "start_char": start,
                "end_char": end,
                "page": page_for_offset(start, page_spans),
                "section": "",
                "strategy": "fixed",
                "n_chars": len(piece),
            })
            index += 1

        if end >= len(text):
            break
        start += step

    return chunks


# ---------------------------------------------------------------------------
# Strategy 2: semantic (structure-aware) chunks
# ---------------------------------------------------------------------------

def _looks_like_heading(line: str) -> bool:
    """Heuristic: is this short line a section heading rather than prose?"""
    stripped = line.strip()
    if not (3 <= len(stripped) <= 70):
        return False
    if stripped.endswith((".", ",", ";", ":")) and not re.match(r"^\d", stripped):
        return False

    normalised = re.sub(r"^\d+(\.\d+)*\.?\s*", "", stripped).strip().lower()
    if normalised in _KNOWN_HEADINGS:
        return True
    # Numbered headings are reliable; bare capitalised phrases are not, so we
    # only accept those that match the known-heading vocabulary above.
    return bool(re.match(r"^\d+(\.\d+)*\.?\s+[A-Z]", stripped)) and len(stripped.split()) <= 8


def chunk_semantic(
    doc: dict,
    max_chunk_size: int = 900,
    min_chunk_size: int = 200,
    overlap_sentences: int = 1,
) -> list[dict]:
    """
    Split on the document's OWN structure instead of on a character count.

    The fixed chunker is structurally blind: it will happily produce a chunk
    whose first half is the end of "Related Work" and whose second half is the
    start of "Method". That chunk is about two unrelated things, so its embedding
    sits between two meanings and matches neither well.

    This chunker instead:
      1. splits the text into paragraphs (the blank lines ingestion.py preserved),
      2. tracks section headings and never merges across one,
      3. greedily packs whole paragraphs together up to `max_chunk_size`,
      4. carries the last sentence of each chunk into the next as overlap.

    Result: chunks with variable length but coherent meaning, each tagged with
    the section it came from. Compare the two strategies in the __main__ block.
    """
    text, page_spans = build_text_with_page_map(doc)
    if not text.strip():
        return []

    # Walk paragraphs, tracking each one's offset so citations still work.
    paragraphs: list[tuple[str, int]] = []
    cursor = 0
    for block in text.split("\n\n"):
        offset = text.find(block, cursor)
        if offset == -1:
            offset = cursor
        paragraphs.append((block, offset))
        cursor = offset + len(block)

    chunks: list[dict] = []
    index = 0
    current_section = ""
    buffer: list[str] = []          # NEW paragraphs since the last flush
    pending_tail = ""               # overlap carried over from the previous chunk
    buffer_start = 0
    buffer_len = 0

    def flush(force: bool = False) -> None:
        """
        Emit the buffered paragraphs as one chunk.

        `buffer` holds only paragraphs added since the last flush; the overlap
        text lives separately in `pending_tail`. Keeping them apart matters: if
        the carried tail sat in `buffer`, a flush with no new content would
        re-emit that tail as a chunk of its own, and the next flush would re-emit
        *its* tail, duplicating text endlessly.
        """
        nonlocal buffer, buffer_len, index, buffer_start, pending_tail
        if not buffer:
            return                                   # nothing new to emit

        body = "\n\n".join(([pending_tail] if pending_tail else []) + buffer).strip()
        if len(body) < min_chunk_size and not force:
            return                                   # keep accumulating

        if body:
            chunks.append({
                "chunk_id": f"{doc['filename']}::semantic::{index:04d}",
                "text": body,
                "source_file": doc["filename"],
                "chunk_index": index,
                "start_char": buffer_start,
                "end_char": buffer_start + len(body),
                "page": page_for_offset(buffer_start, page_spans),
                "section": current_section,
                "strategy": "semantic",
                "n_chars": len(body),
            })
            index += 1

        # Carry the tail sentence(s) forward so meaning is not lost at the seam -
        # the same insurance the fixed chunker gets from character overlap.
        pending_tail = _last_sentences(body, overlap_sentences) if overlap_sentences else ""
        buffer = []
        buffer_len = 0

    for block, offset in paragraphs:
        stripped = block.strip()
        if not stripped:
            continue

        if _looks_like_heading(stripped):
            flush(force=True)            # a heading is a hard boundary
            current_section = re.sub(r"^\d+(\.\d+)*\.?\s*", "", stripped).strip()
            # Do not drag the previous section's tail across the boundary - that
            # is exactly the cross-section contamination this chunker exists to
            # prevent.
            pending_tail = ""
            buffer, buffer_len, buffer_start = [], 0, offset
            continue

        if not buffer:
            buffer_start = offset

        # A single paragraph longer than the budget cannot be packed - fall back
        # to fixed-size splitting *within* that paragraph.
        if len(stripped) > max_chunk_size:
            flush(force=True)
            sub_doc = {"filename": doc["filename"], "pages": [stripped]}
            for sub in chunk_fixed(sub_doc, chunk_size=max_chunk_size,
                                   overlap=DEFAULT_OVERLAP):
                chunks.append({
                    **sub,
                    "chunk_id": f"{doc['filename']}::semantic::{index:04d}",
                    "chunk_index": index,
                    "start_char": offset + sub["start_char"],
                    "end_char": offset + sub["end_char"],
                    "page": page_for_offset(offset + sub["start_char"], page_spans),
                    "section": current_section,
                    "strategy": "semantic",
                })
                index += 1
            pending_tail = ""
            buffer, buffer_len, buffer_start = [], 0, offset + len(stripped)
            continue

        if len(pending_tail) + buffer_len + len(stripped) > max_chunk_size:
            flush(force=True)
            buffer_start = offset

        buffer.append(stripped)
        buffer_len += len(stripped) + 2

    flush(force=True)
    return chunks


def _last_sentences(text: str, count: int, max_chars: int = 250) -> str:
    """
    Return the final `count` sentences of a chunk, for use as overlap.

    The `max_chars` cap is not cosmetic - it prevents a real runaway. Research
    papers are full of text with NO sentence terminators: table rows
    ("AP AP50 AP75 APS APM APL"), equations, figure axis labels. For those,
    re.split finds no boundary and returns the whole text as one "sentence", so
    the tail would be the ENTIRE chunk. That whole chunk then gets carried into
    the next one, which also has no sentence end, so it is carried again... and
    the text compounds. On Mask R-CNN (a table-heavy paper) this turned 56K
    characters of source into 516K characters of chunks.

    Capping the carried tail keeps overlap doing its job - preserving meaning at
    the seam - without letting it become unbounded duplication.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    tail = " ".join(sentences[-count:]) if sentences else ""

    if len(tail) > max_chars:
        tail = tail[-max_chars:]
        # Do not start the overlap mid-word.
        space = tail.find(" ")
        if space != -1:
            tail = tail[space + 1:]

    return tail


# ---------------------------------------------------------------------------
# Corpus-level helper
# ---------------------------------------------------------------------------

def chunk_corpus(docs: list[dict], strategy: str = "fixed", **kwargs) -> list[dict]:
    """Chunk a whole corpus with the chosen strategy ('fixed' or 'semantic')."""
    chunker = {"fixed": chunk_fixed, "semantic": chunk_semantic}[strategy]
    all_chunks: list[dict] = []
    for doc in docs:
        all_chunks.extend(chunker(doc, **kwargs))
    return all_chunks


if __name__ == "__main__":
    setup_console()
    from src.ingestion import load_corpus

    print("Loading and cleaning corpus ...")
    docs = load_corpus(clean=True)
    if not docs:
        raise SystemExit(1)

    for strategy in ("fixed", "semantic"):
        chunks = chunk_corpus(docs, strategy=strategy)
        print(banner(f"STRATEGY: {strategy}"))
        print(f"{'file':<34}{'chunks':>8}{'avg chars':>11}{'min':>7}{'max':>7}")
        print("-" * 67)
        for doc in docs:
            own = [c for c in chunks if c["source_file"] == doc["filename"]]
            if not own:
                continue
            lengths = [c["n_chars"] for c in own]
            print(f"{doc['filename']:<34}{len(own):>8}{sum(lengths) / len(lengths):>11.0f}"
                  f"{min(lengths):>7}{max(lengths):>7}")
        lengths = [c["n_chars"] for c in chunks]
        print("-" * 67)
        print(f"{'TOTAL':<34}{len(chunks):>8}{sum(lengths) / len(lengths):>11.0f}"
              f"{min(lengths):>7}{max(lengths):>7}")

    # Show a real chunk from each strategy, plus the overlap in action.
    fixed = chunk_corpus(docs, strategy="fixed")
    semantic = chunk_corpus(docs, strategy="semantic")

    print(banner("SAMPLE CHUNK - fixed"))
    sample = fixed[12]
    for key in ("chunk_id", "source_file", "page", "start_char", "end_char", "n_chars"):
        print(f"  {key:<12} {sample[key]}")
    print(f"\n  {sample['text']}")

    print(banner("OVERLAP IN ACTION (fixed)"))
    a, b = fixed[12], fixed[13]
    print(f"  chunk 12 ends ....: ...{a['text'][-90:]}")
    print(f"  chunk 13 starts ..: {b['text'][:90]}...")
    print(f"\n  chunk 12 spans chars {a['start_char']}-{a['end_char']}")
    print(f"  chunk 13 spans chars {b['start_char']}-{b['end_char']}"
          f"  -> {a['end_char'] - b['start_char']} chars shared")

    print(banner("SAMPLE CHUNK - semantic"))
    with_section = next((c for c in semantic if c["section"]), semantic[0])
    for key in ("chunk_id", "source_file", "page", "section", "n_chars"):
        print(f"  {key:<12} {with_section[key]}")
    print(f"\n  {with_section['text'][:600]}")
