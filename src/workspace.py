"""
The workspace: the user's documents, and a manifest recording what has been done to each.

WHY THIS FILE EXISTS
--------------------
Before this module, every ingest was O(whole corpus). Uploading one PDF re-extracted
and re-cleaned every PDF already present, re-chunked them, re-embedded them, re-split
every sentence with spaCy (twice), and rebuilt the graph from scratch. At eight
documents that was merely wasteful. At the fifty this project is now designed for it
is untenable: the REBEL pass alone is five to six hours, and the old checkpoint was
invalidated by the arrival of any new file, so it restarted from sentence zero.

The fix is a manifest. It records, per document, what has already been done - so the
system can answer "what is left to do?" instead of redoing everything.

WHAT IS DELIBERATELY *NOT* INCREMENTAL
--------------------------------------
Graph construction. `_canonicalise_entities()` in knowledge_graph.py resolves each
entity toward the most frequent surface form ACROSS THE WHOLE CORPUS, so adding a
document can legitimately change the canonical key of an entity that already exists.
That is genuine whole-corpus dependence, not an implementation shortcut, and the graph
is therefore rebuilt from all per-document triples as the final step of any ingest.
It is the only stage that still costs O(corpus), and at ~10s for eight papers that is
an acceptable price.

IDENTITY IS THE CONTENT HASH, NOT THE FILENAME
----------------------------------------------
Uploads used to be written straight to `RAW_DIR / upload.name`, so two different papers
both called `paper.pdf` silently overwrote one another - and since the filename becomes
the citation string the model prints ("[Source: paper.pdf, p.4]"), the citations became
ambiguous with no error anywhere. Documents are keyed by SHA-256 here: re-uploading the
same bytes is a no-op, and two different files that share a name are stored under
disambiguated filenames.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import (  # noqa: E402
    CHATS_DIR,
    CHROMA_DIR,
    MANIFEST_PATH,
    PROCESSED_DIR,
    RAW_DIR,
    TRIPLES_DIR,
    WORKSPACE_DIR,
    ensure_dirs,
)

# Extractor names recorded in a document's `extractors_run`.
DOMAIN = "domain"
REBEL = "rebel"

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


# ---------------------------------------------------------------------------
# Manifest IO
# ---------------------------------------------------------------------------

def load_manifest() -> dict[str, dict]:
    """Return {doc_id: record}. An absent manifest is an empty workspace, not an error."""
    if not MANIFEST_PATH.exists():
        return {}
    try:
        raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # A corrupt manifest must not brick the app. Losing the manifest costs a
        # re-ingest; refusing to start costs the user everything.
        return {}
    return raw.get("documents", {})


def save_manifest(documents: dict[str, dict]) -> None:
    ensure_dirs()
    MANIFEST_PATH.write_text(
        json.dumps({"version": 1, "documents": documents}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_hash(path: Path | str) -> str:
    """SHA-256 of a file's bytes, read in chunks so a 50MB PDF does not sit in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def safe_filename(name: str) -> str:
    """Make an uploaded name safe to use as a path component and a citation string."""
    cleaned = _UNSAFE_CHARS.sub("_", Path(name).name).strip("._")
    if not cleaned.lower().endswith(".pdf"):
        cleaned += ".pdf"
    return cleaned or "document.pdf"


# ---------------------------------------------------------------------------
# Adding documents
# ---------------------------------------------------------------------------

def _unique_filename(desired: str, documents: dict[str, dict]) -> str:
    """
    Resolve a filename collision by suffixing, never by overwriting.

    The filename is the citation string, so two distinct papers must never share
    one. `paper.pdf` colliding becomes `paper-2.pdf`.
    """
    taken = {record["filename"] for record in documents.values()}
    if desired not in taken and not (RAW_DIR / desired).exists():
        return desired

    stem, suffix = desired[:-4], desired[-4:]
    counter = 2
    while True:
        candidate = f"{stem}-{counter}{suffix}"
        if candidate not in taken and not (RAW_DIR / candidate).exists():
            return candidate
        counter += 1


def add_document(payload: bytes, original_name: str) -> tuple[str, bool]:
    """
    Store an uploaded PDF in the workspace.

    Returns (doc_id, is_new). is_new=False means these exact bytes are already in
    the workspace and nothing was written - re-uploading a file is a no-op rather
    than a duplicate or an overwrite.
    """
    ensure_dirs()
    documents = load_manifest()

    sha = bytes_hash(payload)
    doc_id = sha[:12]
    if doc_id in documents:
        return doc_id, False

    filename = _unique_filename(safe_filename(original_name), documents)
    (RAW_DIR / filename).write_bytes(payload)

    documents[doc_id] = {
        "doc_id": doc_id,
        "filename": filename,
        "original_name": original_name,
        "sha256": sha,
        "n_bytes": len(payload),
        "n_pages": 0,
        "n_chunks": 0,
        "n_triples": 0,
        "extractors_run": [],
        "status": "pending",
        "error": "",
        "added_at": _now(),
    }
    save_manifest(documents)
    return doc_id, True


def add_document_from_path(path: Path | str) -> tuple[str, bool]:
    """Same as add_document, for files already on disk (CLI use, fetch_papers.py)."""
    path = Path(path)
    return add_document(path.read_bytes(), path.name)


def adopt_existing_pdfs() -> int:
    """
    Register any PDF sitting in RAW_DIR that the manifest does not know about.

    Covers two real cases: a user dropping files into the folder by hand, and a
    workspace created before the manifest existed.
    """
    known = {record["filename"] for record in load_manifest().values()}
    adopted = 0
    for pdf in sorted(RAW_DIR.glob("*.pdf")):
        if pdf.name in known:
            continue
        documents = load_manifest()
        sha = file_hash(pdf)
        doc_id = sha[:12]
        if doc_id in documents:
            continue
        documents[doc_id] = {
            "doc_id": doc_id, "filename": pdf.name, "original_name": pdf.name,
            "sha256": sha, "n_bytes": pdf.stat().st_size, "n_pages": 0,
            "n_chunks": 0, "n_triples": 0, "extractors_run": [],
            "status": "pending", "error": "", "added_at": _now(),
        }
        save_manifest(documents)
        adopted += 1
    return adopted


# ---------------------------------------------------------------------------
# Updating and querying
# ---------------------------------------------------------------------------

def update_document(doc_id: str, **fields) -> None:
    documents = load_manifest()
    if doc_id not in documents:
        return
    documents[doc_id].update(fields)
    save_manifest(documents)


def mark_extractor_run(doc_id: str, extractor: str) -> None:
    documents = load_manifest()
    if doc_id not in documents:
        return
    run = documents[doc_id].setdefault("extractors_run", [])
    if extractor not in run:
        run.append(extractor)
    save_manifest(documents)


def documents_missing(extractor: str) -> list[dict]:
    """
    Documents that have not had `extractor` run over them yet.

    This is what makes the overnight REBEL job resumable at DOCUMENT granularity.
    The mechanism it replaces compared a whole-corpus sentence count, so adding a
    single PDF discarded the checkpoint and restarted a five-hour job at zero.
    """
    return [
        record for record in load_manifest().values()
        if extractor not in record.get("extractors_run", [])
    ]


def documents_pending() -> list[dict]:
    return [r for r in load_manifest().values() if r.get("status") != "processed"]


def workspace_stats() -> dict:
    documents = load_manifest()
    return {
        "n_documents": len(documents),
        "n_chunks": sum(r.get("n_chunks", 0) for r in documents.values()),
        "n_triples": sum(r.get("n_triples", 0) for r in documents.values()),
        "n_failed": sum(1 for r in documents.values() if r.get("status") == "failed"),
        "awaiting_rebel": len(documents_missing(REBEL)),
    }


def is_empty() -> bool:
    return not load_manifest()


# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------

def remove_document(doc_id: str, store=None) -> bool:
    """
    Remove one document and everything derived from it.

    Deletes, in order: its embeddings, its triples file, its PDF, its manifest
    entry. The caller rebuilds the graph afterwards (see the module docstring on
    why the graph cannot be updated incrementally).

    `store` is an optional VectorStore; passing the already-loaded instance avoids
    paying the embedding model's load time a second time.
    """
    documents = load_manifest()
    record = documents.get(doc_id)
    if record is None:
        return False

    filename = record["filename"]

    if store is None:
        from src.vector_store import VectorStore
        store = VectorStore()
    store.delete_by_source(filename)

    triples_file = TRIPLES_DIR / f"{doc_id}.json"
    triples_file.unlink(missing_ok=True)
    (RAW_DIR / filename).unlink(missing_ok=True)

    del documents[doc_id]
    save_manifest(documents)
    return True


def delete_workspace() -> None:
    """
    Delete the entire workspace: PDFs, both memories, the manifest, and all chats.

    EVAL_RESULTS_PATH lives outside WORKSPACE_DIR precisely so this cannot touch
    it - those results are committed benchmark evidence, not user data.
    """
    if WORKSPACE_DIR.exists():
        shutil.rmtree(WORKSPACE_DIR, ignore_errors=True)
    ensure_dirs()


# ---------------------------------------------------------------------------
# Per-document triples storage
# ---------------------------------------------------------------------------

def save_document_triples(doc_id: str, triples: list[dict]) -> None:
    ensure_dirs()
    (TRIPLES_DIR / f"{doc_id}.json").write_text(
        json.dumps(triples, ensure_ascii=False), encoding="utf-8"
    )


def load_document_triples(doc_id: str) -> list[dict]:
    path = TRIPLES_DIR / f"{doc_id}.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def load_all_triples() -> list[dict]:
    """
    Every triple in the workspace, concatenated from the per-document files.

    This is what the graph build consumes. Order follows the manifest so a rebuild
    is deterministic - entity canonicalisation is frequency-ordered, but ties are
    broken by insertion order, and a graph that changes shape between identical
    rebuilds would be impossible to reason about.
    """
    triples: list[dict] = []
    for doc_id in load_manifest():
        triples.extend(load_document_triples(doc_id))
    return triples


def rebuild_graph(verbose: bool = False):
    """Rebuild the knowledge graph from all per-document triples and save it."""
    from src.knowledge_graph import KnowledgeGraph

    triples = load_all_triples()
    graph = KnowledgeGraph.from_triples(triples, verbose=verbose)
    graph.save()
    return graph


if __name__ == "__main__":
    from src.utils import banner, setup_console
    setup_console()

    adopted = adopt_existing_pdfs()
    if adopted:
        print(f"Adopted {adopted} PDF(s) already in {RAW_DIR}")

    print(banner("WORKSPACE"))
    stats = workspace_stats()
    print(f"  location   : {WORKSPACE_DIR}")
    print(f"  documents  : {stats['n_documents']}")
    print(f"  chunks     : {stats['n_chunks']:,}")
    print(f"  triples    : {stats['n_triples']:,}")
    print(f"  failed     : {stats['n_failed']}")
    print(f"  awaiting REBEL: {stats['awaiting_rebel']}")

    documents = load_manifest()
    if documents:
        print(f"\n  {'document':<44}{'chunks':>8}{'triples':>9}  extractors")
        print("  " + "-" * 76)
        for record in documents.values():
            extractors = ",".join(record.get("extractors_run", [])) or "-"
            print(f"  {record['filename'][:42]:<44}{record.get('n_chunks', 0):>8}"
                  f"{record.get('n_triples', 0):>9}  {extractors}")
    else:
        print("\n  Empty. Upload PDFs in the app, or run:")
        print("     python scripts/fetch_papers.py     (optional 8-paper demo corpus)")
