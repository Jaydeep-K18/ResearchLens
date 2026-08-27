"""
The KG-RAG app: upload your own documents, then ask questions across all of them.

Run me:  streamlit run src/app.py

THE MODEL
---------
One WORKSPACE holds the PDFs you upload and the two memories built from them - a
vector index for meaning and a knowledge graph for relationships. Inside that
workspace you can keep many CHATS. Every chat searches the same documents, but
each has its own conversation history, so a follow-up question is resolved
against the chat it was asked in and nothing else.

Deleting a chat costs you that conversation. Deleting the workspace deletes
everything: PDFs, both memories, and every chat.

WHY THE UI SHOWS ITS WORKING
----------------------------
The interesting thing about this system is not that it answers questions - a
40-line script does that. It is that you can see the two retrieval paths
disagree, watch the re-ranker reorder them, and read the multi-hop chain behind
an answer no single document contains. So every answer expands into the vector
chunks, the graph evidence, and what re-ranking changed.

A NOTE ON PROCESSING TIME
-------------------------
Uploading runs the FAST path: chunk, embed, and extract relations by dependency
parsing. That is roughly two minutes for fifty papers.

Full REBEL extraction is a different order of magnitude - about 1.3 seconds per
sentence, so five to six hours for fifty papers. A browser tab is the wrong place
for that, so it is a separate resumable command you run once, overnight:

    python src/knowledge_extractor.py --rebel

The graph works immediately after upload and gets richer after that job runs.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import (  # noqa: E402
    EVAL_RESULTS_PATH,
    GRAPH_PATH,
    GRAPH_VIZ_PATH,
    ensure_dirs,
)

MAX_UPLOAD_FILES = 50

st.set_page_config(
    page_title="KG-RAG",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
  .evidence-card {
      border-left: 3px solid #4A9EFF; padding: 10px 14px; margin: 8px 0;
      background: rgba(74,158,255,.06); border-radius: 0 6px 6px 0;
  }
  .evidence-card.graph { border-left-color: #4CD97B; background: rgba(76,217,123,.06); }
  .chain {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: .82rem; color: #7fd1a0; word-break: break-word;
  }
  .cite { color: #8a94a6; font-size: .78rem; }
  .pill {
      display: inline-block; padding: 1px 8px; border-radius: 10px;
      font-size: .7rem; font-weight: 600; margin-right: 6px;
  }
  .pill.vector { background: rgba(74,158,255,.18); color: #7ab8ff; }
  .pill.graph  { background: rgba(76,217,123,.18); color: #7fd1a0; }
  .doc-row { font-size: .8rem; padding: 2px 0; }
  .muted { color: #8a94a6; font-size: .78rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
# st.cache_resource keeps ONE instance across reruns. Streamlit re-executes this
# whole script on every interaction, so without caching each keystroke would
# reload a 90MB embedding model, an 80MB cross-encoder and the graph.

@st.cache_resource(show_spinner="Loading vector store ...")
def load_store():
    from src.vector_store import VectorStore
    return VectorStore()


@st.cache_resource(show_spinner="Loading knowledge graph ...")
def load_graph():
    from src.knowledge_graph import KnowledgeGraph
    if not GRAPH_PATH.exists():
        return None
    return KnowledgeGraph.load()


@st.cache_resource(show_spinner="Loading retrieval pipeline ...")
def load_pipeline():
    """
    The pipeline works with or without a graph.

    A workspace can legitimately have documents embedded but no graph yet, and
    the user should be able to ask questions in the meantime. Passing kg=None
    degrades to vector-only retrieval instead of refusing to build.
    """
    from src.hybrid_retriever import HybridRetriever
    from src.pipeline import KGRagPipeline

    retriever = HybridRetriever(store=load_store(), kg=load_graph())
    return KGRagPipeline(retriever=retriever)


def clear_caches() -> None:
    """Drop cached indexes after the corpus changes, so the next read is fresh."""
    load_store.clear()
    load_graph.clear()
    load_pipeline.clear()


def llm_status() -> tuple[bool, str]:
    from src.llm import LLMClient
    client = LLMClient()
    return client.available, client.status()


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def process_uploads(uploads) -> None:
    """
    Save uploaded PDFs into the workspace and process only what is new.

    Wrapped end to end: one malformed PDF among fifty must not abandon the batch
    or leave the workspace half-written. Per-document failures are recorded in
    the manifest and reported; the rest still complete.
    """
    from src import workspace
    from src.knowledge_extractor import ingest_documents

    if len(uploads) > MAX_UPLOAD_FILES:
        st.sidebar.error(
            f"{len(uploads)} files selected. This build handles up to "
            f"{MAX_UPLOAD_FILES} at once - add them in batches."
        )
        return

    new_ids: list[str] = []
    skipped = 0
    with st.status(f"Adding {len(uploads)} file(s) ...", expanded=True) as status:
        for upload in uploads:
            doc_id, is_new = workspace.add_document(upload.getbuffer().tobytes(), upload.name)
            if is_new:
                new_ids.append(doc_id)
            else:
                skipped += 1
        if skipped:
            status.write(f"Skipped {skipped} file(s) already in this workspace.")

        if not new_ids:
            status.update(label="Nothing new to process.", state="complete")
            return

        def report(index, total, filename, stage):
            status.write(f"[{index}/{total}] {filename} - {stage}")

        status.update(label=f"Processing {len(new_ids)} document(s) ...")
        try:
            summary = ingest_documents(new_ids, use_rebel=False, progress=report)
        except Exception as exc:  # noqa: BLE001
            status.update(label="Processing failed", state="error")
            st.sidebar.error(f"Processing failed: {exc}")
            return

        status.write("Rebuilding knowledge graph ...")
        try:
            workspace.rebuild_graph()
        except Exception as exc:  # noqa: BLE001
            status.write(f"Graph rebuild failed: {exc}")

        label = (f"Done - {summary['processed']} processed, "
                 f"{summary['chunks']:,} chunks, {summary['triples']:,} relations")
        if summary["failed"]:
            label += f", {summary['failed']} failed"
        status.update(label=label, state="complete")

    clear_caches()
    st.rerun()


def reprocess_document(doc_id: str) -> None:
    from src import workspace
    from src.knowledge_extractor import ingest_documents

    with st.spinner("Re-processing ..."):
        ingest_documents([doc_id], use_rebel=False)
        workspace.rebuild_graph()
    clear_caches()
    st.rerun()


def remove_document(doc_id: str) -> None:
    from src import workspace

    with st.spinner("Removing ..."):
        workspace.remove_document(doc_id, store=load_store())
        workspace.rebuild_graph()
    clear_caches()
    st.rerun()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_chat_list() -> str | None:
    """Chat switcher. Returns the active chat id."""
    from src import chat as chat_store

    st.sidebar.subheader("Chats")

    chats = chat_store.list_chats()
    if st.sidebar.button("New chat", width="stretch", type="primary"):
        st.session_state.active_chat = chat_store.create_chat()
        st.rerun()

    if not chats:
        st.sidebar.caption("No chats yet.")
        return st.session_state.get("active_chat")

    ids = [c["chat_id"] for c in chats]
    active = st.session_state.get("active_chat")
    if active not in ids:
        active = ids[0]
        st.session_state.active_chat = active

    labels = {c["chat_id"]: (c["title"] or "Untitled")[:38] for c in chats}
    chosen = st.sidebar.radio(
        "Conversations", ids, index=ids.index(active),
        format_func=lambda cid: labels.get(cid, cid), label_visibility="collapsed",
    )
    if chosen != active:
        st.session_state.active_chat = chosen
        st.rerun()

    if st.sidebar.button("Delete this chat", width="stretch"):
        chat_store.delete_chat(chosen)
        st.session_state.pop("active_chat", None)
        st.rerun()

    return chosen


def render_documents() -> None:
    from src import workspace

    st.sidebar.subheader("Documents")

    uploads = st.sidebar.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploads and st.sidebar.button("Process documents", type="primary", width="stretch"):
        process_uploads(uploads)

    documents = workspace.load_manifest()
    if not documents:
        st.sidebar.caption("No documents yet.")
        return

    stats = workspace.workspace_stats()
    a, b, c = st.sidebar.columns(3)
    a.metric("Docs", stats["n_documents"])
    b.metric("Chunks", f"{stats['n_chunks']:,}")
    c.metric("Relations", f"{stats['n_triples']:,}")

    if stats["awaiting_rebel"]:
        st.sidebar.caption(
            f"{stats['awaiting_rebel']} document(s) have fast-path relations only. "
            "For the full taxonomic graph run `python src/knowledge_extractor.py --rebel` "
            "(slow - about 1.3s per sentence)."
        )

    with st.sidebar.expander(f"Manage {len(documents)} document(s)"):
        for record in documents.values():
            failed = record.get("status") == "failed"
            marker = "!" if failed else ""
            st.markdown(
                f"<div class='doc-row'><b>{marker}{record['filename'][:34]}</b><br>"
                f"<span class='muted'>{record.get('n_chunks', 0)} chunks &middot; "
                f"{record.get('n_triples', 0)} relations &middot; "
                f"{', '.join(record.get('extractors_run', [])) or 'not processed'}</span></div>",
                unsafe_allow_html=True,
            )
            if failed and record.get("error"):
                st.caption(f"Error: {record['error'][:120]}")
            left, right = st.columns(2)
            if left.button("Remove", key=f"rm_{record['doc_id']}", width="stretch"):
                remove_document(record["doc_id"])
            if right.button("Redo", key=f"re_{record['doc_id']}", width="stretch"):
                reprocess_document(record["doc_id"])
            st.divider()


def render_sidebar() -> tuple[str | None, dict]:
    from src import workspace

    st.sidebar.title("KG-RAG")
    st.sidebar.caption("Ask questions across your own documents")

    active_chat = render_chat_list()
    st.sidebar.divider()
    render_documents()
    st.sidebar.divider()

    ready, status = llm_status()
    if ready:
        st.sidebar.success(f"LLM ready - {status}")
    else:
        st.sidebar.error("No LLM key. Retrieval works; answers need a key in `.env`.")

    settings = {
        "top_k": st.sidebar.slider("Results sent to the LLM", 3, 20, 12),
        "vector_k": st.sidebar.slider("Vector candidates", 5, 25, 10),
        "hops": st.sidebar.slider("Graph hops", 1, 3, 2,
                                  help="How far to walk from the entities in your question. "
                                       "3 hops finds more, and more noise."),
    }

    st.sidebar.divider()
    with st.sidebar.expander("Danger zone"):
        st.caption("Deletes every PDF, both memories, and all chats. "
                   "Benchmark results are kept.")
        if st.checkbox("I understand this cannot be undone"):
            if st.button("Delete workspace", type="primary", width="stretch"):
                from src import chat as chat_store
                chat_store.delete_all_chats()
                workspace.delete_workspace()
                st.session_state.clear()
                clear_caches()
                st.rerun()

    return active_chat, settings


# ---------------------------------------------------------------------------
# Evidence rendering
# ---------------------------------------------------------------------------

def render_evidence_item(item: dict, index: int) -> None:
    method = item["retrieval_method"]
    citation = f"{item['source_file']} &middot; p.{item['page']}"
    if method == "graph":
        hops = item.get("hop_distance", 1)
        subject, relation, obj = item.get("triple", ("", "", ""))
        st.markdown(
            f"<div class='evidence-card graph'>"
            f"<span class='pill graph'>GRAPH</span>"
            f"<span class='cite'>{citation} &middot; {hops} hop{'s' if hops > 1 else ''}</span>"
            f"<div class='chain'>{subject} —[{relation}]&rarr; {obj}</div>"
            f"<div style='margin-top:6px'>{item['text'][:400]}</div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='evidence-card'>"
            f"<span class='pill vector'>TEXT</span>"
            f"<span class='cite'>{citation}</span>"
            f"<div style='margin-top:6px'>{item['text'][:400]}</div></div>",
            unsafe_allow_html=True,
        )


def render_answer_details(state: dict) -> None:
    """The expandable machinery behind one answer."""
    reranked = state.get("reranked_results", []) or []
    vector = state.get("vector_results", []) or []
    graph = state.get("graph_results", []) or []
    timings = state.get("timings", {}) or {}

    methods = [item["retrieval_method"] for item in reranked]
    cols = st.columns(4)
    cols[0].metric("Vector hits", len(vector))
    cols[1].metric("Graph evidence", len(graph))
    cols[2].metric("Graph in final", f"{methods.count('graph')}/{len(reranked)}")
    cols[3].metric("Time", f"{timings.get('total', 0):.1f}s")

    if state.get("resolved_query") and state["resolved_query"] != state.get("original_query"):
        st.caption(f"Searched for: *{state['resolved_query']}*")

    sources = state.get("sources", []) or []
    if sources:
        st.caption("Sources: " + "  |  ".join(
            f"{s['source_file']} p.{s['page']}" for s in sources))

    with st.expander(f"Evidence sent to the model ({len(reranked)} items)"):
        st.caption("This is exactly what the model saw. Everything in the answer "
                   "should be traceable to one of these.")
        for index, item in enumerate(reranked):
            render_evidence_item(item, index)

    if graph:
        with st.expander(f"All graph evidence ({len(graph)})"):
            st.caption("Relationships reached by walking out from entities named in "
                       "your question. Multi-hop chains are facts no single document states.")
            for index, item in enumerate(graph):
                render_evidence_item(item, index)


# ---------------------------------------------------------------------------
# Chat tab
# ---------------------------------------------------------------------------

def render_onboarding() -> None:
    st.header("Upload documents to get started")
    st.markdown(
        "This system builds **two memories** of the documents you upload:\n\n"
        "- a **vector index**, which finds passages that resemble your question\n"
        "- a **knowledge graph**, which follows relationships *across* documents\n\n"
        "That second memory is what lets it answer questions whose answer is in "
        "no single document — like *“which method outperforms X, and who wrote it?”*, "
        "where the two facts live in different papers and the connection lives in neither.\n\n"
        "**Use the sidebar to upload PDFs.** Up to "
        f"{MAX_UPLOAD_FILES} at a time; roughly two minutes for fifty papers."
    )
    st.info(
        "No sample documents are included — this is your workspace. If you want a "
        "corpus to try it on, `python scripts/fetch_papers.py` downloads eight "
        "open-access papers that cite and benchmark against each other.",
        icon="💡",
    )


def render_chat_tab(active_chat: str | None, settings: dict) -> None:
    from src import chat as chat_store
    from src import workspace

    if workspace.is_empty():
        render_onboarding()
        return

    if active_chat is None:
        active_chat = chat_store.create_chat()
        st.session_state.active_chat = active_chat
        st.rerun()

    chat = chat_store.load_chat(active_chat)
    if chat is None:
        st.session_state.pop("active_chat", None)
        st.rerun()
        return

    if load_graph() is None:
        st.warning(
            "No knowledge graph yet — answering with vector search only. "
            "Upload documents or re-process to build the graph.",
            icon="⚠️",
        )

    for message in chat["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("state"):
                render_answer_details(message["state"])

    question = st.chat_input("Ask anything about your documents")
    if not question:
        return

    chat_store.append_message(active_chat, "user", question)
    with st.chat_message("user"):
        st.markdown(question)

    # History EXCLUDING the question just asked - it is the thing being resolved,
    # not context for resolving itself.
    history = chat_store.history_for_condense(active_chat)[:-1]

    pipeline = load_pipeline()
    pipeline.top_k = settings["top_k"]
    pipeline.vector_k = settings["vector_k"]
    pipeline.graph_hops = settings["hops"]

    with st.chat_message("assistant"):
        with st.spinner("Searching both memories ..."):
            try:
                state = pipeline.run(question, history=history or None)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not answer: {exc}")
                return

        answer = state.get("answer") or state.get("error") or "No answer produced."
        st.markdown(answer)
        render_answer_details(state)

    chat_store.append_message(
        active_chat, "assistant", answer,
        state={
            "reranked_results": state.get("reranked_results", []),
            "vector_results": state.get("vector_results", []),
            "graph_results": state.get("graph_results", []),
            "sources": state.get("sources", []),
            "timings": state.get("timings", {}),
            "original_query": state.get("original_query", question),
            "resolved_query": state.get("resolved_query", question),
        },
    )
    st.rerun()


# ---------------------------------------------------------------------------
# Graph tab
# ---------------------------------------------------------------------------

def render_graph_tab() -> None:
    from src.knowledge_graph import format_path

    graph = load_graph()
    if graph is None:
        st.info("No knowledge graph yet. Upload documents to build one.")
        return

    stats = graph.stats()
    st.caption(f"{stats['nodes']:,} entities and {stats['edges']:,} relations "
               "extracted from your documents.")

    # Centre options come from the user's OWN graph, not a hardcoded list of
    # sample-corpus entities. top_nodes is (node_key, degree) sorted by degree.
    top_nodes = [graph.display_name(key) for key, _ in stats.get("top_nodes", [])[:40]]

    controls = st.columns([2, 1, 1])
    centre = controls[0].selectbox("Centre the view on an entity", top_nodes) if top_nodes else None
    hops = controls[1].slider("Hops", 1, 3, 2)
    max_nodes = controls[2].slider("Max nodes", 20, 150, 70)

    if centre and st.button("Render graph"):
        from src.graph_viz import build_visualisation
        with st.spinner("Building visualisation ..."):
            build_visualisation(graph, centre, hops=hops, max_nodes=max_nodes)
        st.components.v1.html(GRAPH_VIZ_PATH.read_text(encoding="utf-8"), height=620)

    st.divider()
    st.subheader("Find a connection between two entities")
    st.caption("This is the capability vector search structurally cannot provide: "
               "a chain of relationships linking two things no single document "
               "discusses together.")
    left, right, button = st.columns([2, 2, 1])
    source = left.text_input("From")
    target = right.text_input("To")
    if button.button("Find path") and source and target:
        path = graph.find_path(source, target)
        if not path:
            st.warning("No path found between those entities.")
        else:
            st.success(f"Connected in {len(path)} hop(s)")
            st.code(format_path(graph, path))
            for step in path:
                st.caption(f"{step['source_file']} p.{step['page']}")
                st.markdown(f"> {step['sentence']}")


# ---------------------------------------------------------------------------
# Benchmark tab (developer-facing, kept out of the main workflow)
# ---------------------------------------------------------------------------

def render_benchmark_tab() -> None:
    import json

    st.caption(
        "An optional developer benchmark, not part of using the app. It scores this "
        "system against plain vector-only RAG on a fixed set of questions written for "
        "the eight-paper demo corpus — so these numbers describe that corpus, not the "
        "documents you have uploaded."
    )

    if not EVAL_RESULTS_PATH.exists():
        st.info(
            "No benchmark results. To reproduce them:\n\n"
            "```\npython scripts/fetch_papers.py\npython src/evaluate.py\n```"
        )
        return

    data = json.loads(EVAL_RESULTS_PATH.read_text(encoding="utf-8"))
    summary = data.get("summary", {})

    rows = []
    for tier in ("simple", "medium", "hard", "ALL"):
        if tier not in summary:
            continue
        entry = summary[tier]
        rows.append({
            "Tier": tier,
            "n": entry["n"],
            "Basic RAG": round(entry["basic_rag"]["overall"], 2),
            "KG-RAG": round(entry["kg_rag"]["overall"], 2),
            "Delta": round(entry["delta_overall"], 2),
            "Graph share": f"{entry['graph_share']:.0%}",
        })
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)

    with st.expander("Per-question results"):
        for record in data.get("results", []):
            basic = record["basic_rag"]["scores"]
            kgrag = record["kg_rag"]["scores"]
            header = (f"[{record['difficulty']}] {record['question'][:80]} "
                      f"- basic {basic['correctness']}/5 vs KG-RAG {kgrag['correctness']}/5")
            with st.expander(header):
                st.markdown("**Reference answer**")
                st.caption(record["expected"])
                left, right = st.columns(2)
                with left:
                    st.markdown("**Basic RAG**")
                    st.markdown(record["basic_rag"]["answer"] or "_none_")
                with right:
                    st.markdown("**KG-RAG**")
                    st.markdown(record["kg_rag"]["answer"] or "_none_")


# ---------------------------------------------------------------------------

def main() -> None:
    ensure_dirs()

    from src import workspace
    # Pick up PDFs a user dropped into the folder by hand, so the manifest and
    # the filesystem cannot silently disagree about what the corpus contains.
    workspace.adopt_existing_pdfs()

    active_chat, settings = render_sidebar()

    chat_tab, graph_tab, benchmark_tab = st.tabs(
        ["Chat", "Knowledge graph", "Benchmark"]
    )
    with chat_tab:
        render_chat_tab(active_chat, settings)
    with graph_tab:
        render_graph_tab()
    with benchmark_tab:
        render_benchmark_tab()


if __name__ == "__main__":
    main()
