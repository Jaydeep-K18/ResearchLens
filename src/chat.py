"""
Chats: many conversations over one shared workspace.

THE MODEL
---------
There is exactly ONE workspace - the uploaded PDFs and the two memories built from
them. Inside it there can be many chats. Every chat queries the same documents and
the same knowledge graph, but each keeps its OWN conversation history, so a follow-up
question is resolved against the chat it was asked in and nothing else.

That separation is the point. Two chats can explore the same corpus from different
angles without one contaminating the other's follow-ups, and deleting a chat costs
only that conversation, never the expensive shared index.

Deleting the WORKSPACE deletes everything, chats included - see
workspace.delete_workspace(). Chats live under WORKSPACE_DIR for exactly that reason.

STORAGE
-------
One JSON file per chat, plus an index for ordering and titles. Files rather than a
database because a chat is small, append-mostly, and read by exactly one process;
SQLite would be ceremony for no benefit at this scale.
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import CHATS_DIR, ensure_dirs  # noqa: E402

INDEX_PATH_NAME = "index.json"

# How many prior turns to hand the follow-up resolver. Enough for "it"/"those" to
# have a referent, small enough that the condensing prompt stays cheap and focused.
HISTORY_TURNS_FOR_CONDENSE = 6


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _index_path() -> Path:
    return CHATS_DIR / INDEX_PATH_NAME


def _chat_path(chat_id: str) -> Path:
    return CHATS_DIR / f"{chat_id}.json"


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

def _load_index() -> list[dict]:
    path = _index_path()
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("chats", [])
    except (json.JSONDecodeError, OSError):
        return []


def _save_index(chats: list[dict]) -> None:
    ensure_dirs()
    _index_path().write_text(
        json.dumps({"version": 1, "chats": chats}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def list_chats() -> list[dict]:
    """Chat summaries, most recently used first."""
    return sorted(_load_index(), key=lambda c: c.get("updated_at", ""), reverse=True)


# ---------------------------------------------------------------------------
# Chat lifecycle
# ---------------------------------------------------------------------------

def create_chat(title: str = "New chat") -> str:
    ensure_dirs()
    chat_id = uuid.uuid4().hex[:12]
    now = _now()

    _chat_path(chat_id).write_text(
        json.dumps({"chat_id": chat_id, "title": title, "created_at": now,
                    "updated_at": now, "messages": []}, ensure_ascii=False),
        encoding="utf-8",
    )
    chats = _load_index()
    chats.append({"chat_id": chat_id, "title": title,
                  "created_at": now, "updated_at": now, "n_messages": 0})
    _save_index(chats)
    return chat_id


def load_chat(chat_id: str) -> dict | None:
    path = _chat_path(chat_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def delete_chat(chat_id: str) -> bool:
    """Delete one conversation. The shared workspace is untouched."""
    path = _chat_path(chat_id)
    existed = path.exists()
    path.unlink(missing_ok=True)
    _save_index([c for c in _load_index() if c["chat_id"] != chat_id])
    return existed


def rename_chat(chat_id: str, title: str) -> None:
    chat = load_chat(chat_id)
    if chat is None:
        return
    chat["title"] = title
    _write_chat(chat)

    chats = _load_index()
    for entry in chats:
        if entry["chat_id"] == chat_id:
            entry["title"] = title
    _save_index(chats)


def _write_chat(chat: dict) -> None:
    ensure_dirs()
    chat["updated_at"] = _now()
    _chat_path(chat["chat_id"]).write_text(
        json.dumps(chat, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

def append_message(chat_id: str, role: str, content: str, **extra) -> None:
    """
    Add a message. `extra` carries assistant-only fields: sources, evidence,
    timings, resolved_query.
    """
    chat = load_chat(chat_id)
    if chat is None:
        return

    chat["messages"].append({"role": role, "content": content,
                             "timestamp": _now(), **extra})

    # Auto-title from the first question. A sidebar full of "New chat" is useless
    # once there are more than two conversations.
    if role == "user" and chat.get("title") in (None, "", "New chat"):
        chat["title"] = content[:48] + ("..." if len(content) > 48 else "")

    _write_chat(chat)

    chats = _load_index()
    for entry in chats:
        if entry["chat_id"] == chat_id:
            entry["title"] = chat["title"]
            entry["updated_at"] = chat["updated_at"]
            entry["n_messages"] = len(chat["messages"])
    _save_index(chats)


def history_for_condense(chat_id: str, turns: int = HISTORY_TURNS_FOR_CONDENSE) -> list[dict]:
    """
    The recent turns used to resolve a follow-up, oldest first.

    Only role and content - evidence and timings are irrelevant to resolving what
    "it" refers to, and would just cost prompt tokens.
    """
    chat = load_chat(chat_id)
    if chat is None:
        return []
    return [{"role": m["role"], "content": m["content"]}
            for m in chat["messages"][-turns:]]


def delete_all_chats() -> int:
    """Remove every chat. Called as part of deleting the workspace."""
    removed = 0
    if CHATS_DIR.exists():
        for path in CHATS_DIR.glob("*.json"):
            path.unlink(missing_ok=True)
            removed += 1
    return removed
