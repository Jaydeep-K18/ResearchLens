"""
Workspace tests - document identity, incremental processing, and clean removal.

These cover the two crashes that made upload-first use impossible before this
module existed, so neither can come back:

  1. An empty workspace raising SystemExit out of load_triples(), which killed
     the very first upload on a clean install - after the vector store had
     already been written to.
  2. Uploads keyed by filename, so two different papers both called `paper.pdf`
     silently overwrote each other. Since the filename becomes the citation
     string the model prints, that made citations point at the wrong document
     with no error anywhere.

Every test runs against a temporary workspace. The module-level path constants
are monkeypatched, because src.workspace imports them by value at import time -
patching src.utils alone would not be seen.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import workspace  # noqa: E402


@pytest.fixture
def temp_workspace(tmp_path, monkeypatch):
    """Redirect the whole workspace at a tmp dir for the duration of one test."""
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    triples = processed / "triples"
    chats = tmp_path / "chats"
    for directory in (raw, processed, triples, chats):
        directory.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(workspace, "RAW_DIR", raw)
    monkeypatch.setattr(workspace, "PROCESSED_DIR", processed)
    monkeypatch.setattr(workspace, "TRIPLES_DIR", triples)
    monkeypatch.setattr(workspace, "MANIFEST_PATH", processed / "manifest.json")
    monkeypatch.setattr(workspace, "WORKSPACE_DIR", tmp_path)
    monkeypatch.setattr(workspace, "ensure_dirs", lambda: None)
    return tmp_path


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

def test_empty_workspace_reports_empty_rather_than_raising(temp_workspace):
    assert workspace.load_manifest() == {}
    assert workspace.is_empty()
    assert workspace.load_all_triples() == []
    assert workspace.workspace_stats()["n_documents"] == 0


def test_load_triples_returns_empty_on_a_fresh_workspace(temp_workspace, monkeypatch):
    """
    The regression that crashed the first upload.

    load_triples() used to `raise SystemExit` on a missing file. The upload
    handler called it to merge with existing triples, and on a fresh workspace
    there is by definition nothing to merge with - so the first upload died
    partway through, after the vector store had been mutated.
    """
    from src import knowledge_extractor

    monkeypatch.setattr(knowledge_extractor, "TRIPLES_PATH", temp_workspace / "nope.json")
    assert knowledge_extractor.load_triples() == []


def test_corrupt_manifest_does_not_brick_the_workspace(temp_workspace):
    # Losing the manifest costs a re-ingest. Refusing to start costs everything.
    workspace.MANIFEST_PATH.write_text("{not json", encoding="utf-8")
    assert workspace.load_manifest() == {}


# ---------------------------------------------------------------------------
# Document identity
# ---------------------------------------------------------------------------

def test_same_bytes_are_added_once(temp_workspace):
    payload = b"%PDF-1.4 fake content"
    first_id, first_new = workspace.add_document(payload, "paper.pdf")
    second_id, second_new = workspace.add_document(payload, "paper.pdf")

    assert first_new is True
    assert second_new is False, "re-uploading identical bytes must be a no-op"
    assert first_id == second_id
    assert len(workspace.load_manifest()) == 1


def test_different_files_sharing_a_name_do_not_overwrite(temp_workspace):
    """
    The filename IS the citation string ("[Source: paper.pdf, p.4]"), so two
    distinct documents must never share one.
    """
    a_id, _ = workspace.add_document(b"%PDF first document", "paper.pdf")
    b_id, _ = workspace.add_document(b"%PDF second, different", "paper.pdf")

    assert a_id != b_id
    documents = workspace.load_manifest()
    assert len(documents) == 2

    filenames = {record["filename"] for record in documents.values()}
    assert len(filenames) == 2, f"filenames collided: {filenames}"
    assert "paper.pdf" in filenames
    # Both files must actually exist on disk - neither clobbered the other.
    for name in filenames:
        assert (workspace.RAW_DIR / name).exists()


def test_unsafe_filenames_are_sanitised(temp_workspace):
    doc_id, _ = workspace.add_document(b"%PDF x", "../../etc/pa ss wd.pdf")
    record = workspace.load_manifest()[doc_id]
    assert "/" not in record["filename"]
    assert "\\" not in record["filename"]
    assert record["filename"].endswith(".pdf")
    # The original is kept for display, only the stored name is sanitised.
    assert record["original_name"] == "../../etc/pa ss wd.pdf"


# ---------------------------------------------------------------------------
# Incremental work tracking
# ---------------------------------------------------------------------------

def test_documents_missing_drives_the_resumable_rebel_job(temp_workspace):
    """
    Resumability is per DOCUMENT now. The mechanism this replaced compared a
    whole-corpus sentence count, so adding one PDF discarded the checkpoint and
    restarted a five-hour job from zero.
    """
    a_id, _ = workspace.add_document(b"%PDF a", "a.pdf")
    b_id, _ = workspace.add_document(b"%PDF b", "b.pdf")

    assert len(workspace.documents_missing(workspace.REBEL)) == 2

    workspace.mark_extractor_run(a_id, workspace.REBEL)
    remaining = workspace.documents_missing(workspace.REBEL)
    assert [r["doc_id"] for r in remaining] == [b_id]

    # Marking twice must not duplicate the entry.
    workspace.mark_extractor_run(a_id, workspace.REBEL)
    assert workspace.load_manifest()[a_id]["extractors_run"].count(workspace.REBEL) == 1


def test_stats_aggregate_across_documents(temp_workspace):
    a_id, _ = workspace.add_document(b"%PDF a", "a.pdf")
    b_id, _ = workspace.add_document(b"%PDF b", "b.pdf")
    workspace.update_document(a_id, n_chunks=100, n_triples=40, status="processed")
    workspace.update_document(b_id, n_chunks=50, n_triples=10, status="failed")

    stats = workspace.workspace_stats()
    assert stats["n_documents"] == 2
    assert stats["n_chunks"] == 150
    assert stats["n_triples"] == 50
    assert stats["n_failed"] == 1


# ---------------------------------------------------------------------------
# Per-document triples and removal
# ---------------------------------------------------------------------------

def test_triples_are_stored_per_document(temp_workspace):
    a_id, _ = workspace.add_document(b"%PDF a", "a.pdf")
    b_id, _ = workspace.add_document(b"%PDF b", "b.pdf")

    workspace.save_document_triples(a_id, [{"subject": "X", "relation": "r", "object": "Y"}])
    workspace.save_document_triples(b_id, [{"subject": "P", "relation": "r", "object": "Q"}])

    assert len(workspace.load_document_triples(a_id)) == 1
    assert len(workspace.load_all_triples()) == 2


def test_removing_a_document_removes_only_its_data(temp_workspace):
    a_id, _ = workspace.add_document(b"%PDF a", "a.pdf")
    b_id, _ = workspace.add_document(b"%PDF b", "b.pdf")
    workspace.save_document_triples(a_id, [{"subject": "X", "relation": "r", "object": "Y"}])
    workspace.save_document_triples(b_id, [{"subject": "P", "relation": "r", "object": "Q"}])

    class FakeStore:
        def __init__(self):
            self.deleted = []

        def delete_by_source(self, source_file):
            self.deleted.append(source_file)
            return 1

    store = FakeStore()
    assert workspace.remove_document(a_id, store=store) is True

    assert store.deleted == ["a.pdf"], "must delete the embeddings by source file"
    assert not (workspace.RAW_DIR / "a.pdf").exists()
    assert workspace.load_document_triples(a_id) == []
    # b is untouched.
    assert b_id in workspace.load_manifest()
    assert len(workspace.load_document_triples(b_id)) == 1
    assert len(workspace.load_all_triples()) == 1


def test_removing_an_unknown_document_is_a_no_op(temp_workspace):
    assert workspace.remove_document("does-not-exist", store=None) is False


def test_adopt_picks_up_pdfs_dropped_in_by_hand(temp_workspace):
    (workspace.RAW_DIR / "manual.pdf").write_bytes(b"%PDF dropped in by hand")
    assert workspace.adopt_existing_pdfs() == 1
    assert any(r["filename"] == "manual.pdf" for r in workspace.load_manifest().values())
    # Adopting twice must not duplicate.
    assert workspace.adopt_existing_pdfs() == 0
