"""
Chat tests - isolation between conversations, and follow-up resolution.

The property that matters most here is ISOLATION. Many chats share one
workspace, so they share the documents and the graph - but a follow-up asked in
one chat must be resolved against that chat's history and nothing else. If
histories leaked, "how is it trained?" in a fresh chat would resolve against a
completely unrelated conversation and retrieve the wrong thing, which would look
like a retrieval bug rather than a state bug.

No test here calls an LLM. condense_question is exercised with a stub, because
what needs pinning is the control flow around it - when it runs, and what
happens when it misbehaves - not the model's phrasing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import chat as chat_store  # noqa: E402


@pytest.fixture
def temp_chats(tmp_path, monkeypatch):
    chats = tmp_path / "chats"
    chats.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(chat_store, "CHATS_DIR", chats)
    monkeypatch.setattr(chat_store, "ensure_dirs", lambda: None)
    return chats


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def test_create_load_and_list(temp_chats):
    chat_id = chat_store.create_chat()
    chat = chat_store.load_chat(chat_id)

    assert chat is not None
    assert chat["chat_id"] == chat_id
    assert chat["messages"] == []
    assert [c["chat_id"] for c in chat_store.list_chats()] == [chat_id]


def test_loading_a_missing_chat_returns_none_rather_than_raising(temp_chats):
    assert chat_store.load_chat("nope") is None
    assert chat_store.history_for_condense("nope") == []


def test_delete_removes_only_that_chat(temp_chats):
    keep = chat_store.create_chat()
    drop = chat_store.create_chat()
    chat_store.append_message(keep, "user", "kept question")

    assert chat_store.delete_chat(drop) is True
    remaining = [c["chat_id"] for c in chat_store.list_chats()]

    assert remaining == [keep]
    assert chat_store.load_chat(drop) is None
    assert len(chat_store.load_chat(keep)["messages"]) == 1


def test_chat_is_auto_titled_from_its_first_question(temp_chats):
    # A sidebar full of "New chat" is unusable past two conversations.
    chat_id = chat_store.create_chat()
    chat_store.append_message(chat_id, "user", "Which method performs best on the benchmark?")

    title = chat_store.load_chat(chat_id)["title"]
    assert title != "New chat"
    assert title.startswith("Which method performs best")


def test_long_first_question_is_truncated_for_the_title(temp_chats):
    chat_id = chat_store.create_chat()
    chat_store.append_message(chat_id, "user", "x" * 200)
    assert len(chat_store.load_chat(chat_id)["title"]) < 60


# ---------------------------------------------------------------------------
# Isolation - the property the whole model depends on
# ---------------------------------------------------------------------------

def test_histories_do_not_leak_between_chats(temp_chats):
    first = chat_store.create_chat()
    second = chat_store.create_chat()

    chat_store.append_message(first, "user", "Tell me about method X.")
    chat_store.append_message(first, "assistant", "X is a detector.")

    assert len(chat_store.history_for_condense(first)) == 2
    assert chat_store.history_for_condense(second) == [], \
        "a new chat must start with no history, or follow-ups resolve against the wrong thread"


def test_history_is_capped_and_ordered_oldest_first(temp_chats):
    chat_id = chat_store.create_chat()
    for index in range(10):
        chat_store.append_message(chat_id, "user", f"question {index}")

    history = chat_store.history_for_condense(chat_id, turns=4)
    assert len(history) == 4
    assert history[0]["content"] == "question 6"
    assert history[-1]["content"] == "question 9"


def test_history_carries_only_role_and_content(temp_chats):
    # Evidence and timings are irrelevant to resolving what "it" refers to, and
    # would just cost prompt tokens.
    chat_id = chat_store.create_chat()
    chat_store.append_message(chat_id, "assistant", "answer",
                              state={"huge": "payload"}, sources=[1, 2, 3])

    assert set(chat_store.history_for_condense(chat_id)[0]) == {"role", "content"}


def test_assistant_extras_are_persisted_on_the_message(temp_chats):
    chat_id = chat_store.create_chat()
    chat_store.append_message(chat_id, "assistant", "answer", state={"timings": {"total": 1.0}})

    message = chat_store.load_chat(chat_id)["messages"][0]
    assert message["state"]["timings"]["total"] == 1.0


def test_delete_all_chats(temp_chats):
    chat_store.create_chat()
    chat_store.create_chat()
    assert chat_store.delete_all_chats() >= 2
    assert chat_store.list_chats() == []


# ---------------------------------------------------------------------------
# Follow-up resolution
# ---------------------------------------------------------------------------

def test_no_history_means_no_llm_call_and_no_rewrite(monkeypatch):
    """First turns must not pay the condensing latency."""
    from src import pipeline

    def explode(*args, **kwargs):
        raise AssertionError("condense_question must not call the LLM without history")

    monkeypatch.setattr(pipeline, "get_llm", explode)
    assert pipeline.condense_question([], "What is X?") == "What is X?"


def test_follow_up_is_rewritten_using_history(monkeypatch):
    from src import pipeline

    class StubLLM:
        available = True

        def generate(self, prompt, **kwargs):
            assert "Tell me about DETR" in prompt, "history must reach the prompt"
            return "How is DETR trained?"

    monkeypatch.setattr(pipeline, "get_llm", lambda: StubLLM())
    history = [{"role": "user", "content": "Tell me about DETR"},
               {"role": "assistant", "content": "DETR is a detector."}]

    assert pipeline.condense_question(history, "How is it trained?") == "How is DETR trained?"


def test_condense_failure_falls_back_to_the_original_question(monkeypatch):
    """A follow-up answered poorly beats a chat that errors out."""
    from src import pipeline

    class BrokenLLM:
        available = True

        def generate(self, prompt, **kwargs):
            raise RuntimeError("provider down")

    monkeypatch.setattr(pipeline, "get_llm", lambda: BrokenLLM())
    history = [{"role": "user", "content": "earlier"}]

    assert pipeline.condense_question(history, "How is it trained?") == "How is it trained?"


def test_implausible_rewrites_are_rejected(monkeypatch):
    """
    A model that answers the question instead of rewriting it, or returns
    nothing, must not silently replace the user's query.
    """
    from src import pipeline

    class RamblingLLM:
        available = True

        def __init__(self, reply):
            self.reply = reply

        def generate(self, prompt, **kwargs):
            return self.reply

    history = [{"role": "user", "content": "earlier"}]

    monkeypatch.setattr(pipeline, "get_llm", lambda: RamblingLLM("   "))
    assert pipeline.condense_question(history, "How is it trained?") == "How is it trained?"

    monkeypatch.setattr(pipeline, "get_llm", lambda: RamblingLLM("word " * 500))
    assert pipeline.condense_question(history, "How is it trained?") == "How is it trained?"


def test_missing_llm_key_does_not_break_follow_ups(monkeypatch):
    from src import pipeline

    class NoKey:
        available = False

        def generate(self, *a, **k):
            raise AssertionError("must not generate without a key")

    monkeypatch.setattr(pipeline, "get_llm", lambda: NoKey())
    assert pipeline.condense_question([{"role": "user", "content": "x"}], "and?") == "and?"


def test_evidence_containing_sets_can_be_persisted(temp_chats):
    """
    Regression: the answer was generated, shown, and then lost.

    deduplicate() records `also_found_by` as a SET, and that evidence is stored
    verbatim on the assistant message. json.dumps raises TypeError on a set, so
    the write failed after the answer had already been rendered - leaving the
    user's question in the transcript with no reply and a traceback in the log.
    """
    chat_id = chat_store.create_chat()
    chat_store.append_message(
        chat_id, "assistant", "the answer",
        state={"reranked_results": [{"text": "x", "also_found_by": {"graph", "vector"}}]},
    )

    reloaded = chat_store.load_chat(chat_id)
    assert reloaded is not None, "the chat must survive being written"
    stored = reloaded["messages"][0]["state"]["reranked_results"][0]["also_found_by"]
    assert sorted(stored) == ["graph", "vector"]
