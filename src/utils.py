"""
Small shared helpers used by every module in src/.

This file exists to avoid copy-pasting the same six lines into ten modules.
It holds no project logic - just paths and console setup.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths - resolved from this file's location, so every module agrees on where
# the project root is no matter which directory you run python from.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "chroma_db"

TRIPLES_PATH = PROCESSED_DIR / "triples.json"
GRAPH_PATH = PROCESSED_DIR / "knowledge_graph.gpickle"
GRAPH_VIZ_PATH = PROCESSED_DIR / "graph_viz.html"
EVAL_RESULTS_PATH = DATA_DIR / "eval_results.json"


def ensure_dirs() -> None:
    """Create the data folders if they do not exist yet."""
    for directory in (RAW_DIR, PROCESSED_DIR):
        directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------

def setup_console() -> None:
    """
    Make stdout/stderr UTF-8 safe on Windows.

    Why this is needed: the Windows console defaults to the cp1252 code page,
    which cannot represent most of the characters that show up in real research
    PDFs - the footnote star, em dashes, Greek letters in equations,
    ligatures. Printing a raw extracted sentence then dies with

        UnicodeEncodeError: 'charmap' codec can't encode character '\\u2217'

    That is a *display* failure, not a data failure - the text itself is fine.
    Reconfiguring the stream to UTF-8 (with errors="replace" as a backstop for
    fonts the terminal cannot draw) makes every __main__ block in this project
    printable. Call it once at the top of any script that prints document text.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None and hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # Stream already detached or not reconfigurable (e.g. piped in a way
            # that does not support it). Not worth crashing over.
            pass


def banner(title: str, width: int = 80, char: str = "=") -> str:
    """Consistent section header for the __main__ blocks."""
    return f"\n{char * width}\n{title}\n{char * width}"
