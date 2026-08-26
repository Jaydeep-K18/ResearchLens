"""
Phase 7, Step 7.1 - the Streamlit UI.

Run me:  streamlit run src/app.py

WHAT THIS IS FOR
----------------
Not "a chatbot". The interesting thing about this system is not that it answers
questions - a 40-line script does that. It is that you can SEE the two retrieval
paths disagree, watch the re-ranker reorder them, and read the multi-hop chain
that produced an answer no single document contains.

So the UI is built around showing the machinery, not hiding it:

  - every answer expands into the vector chunks, the graph evidence, and what
    re-ranking did to the ordering
  - a side-by-side toggle runs basic RAG and KG-RAG on the same question, which
    is the single most convincing thing to demonstrate
  - the graph tab is explorable, because a knowledge graph you cannot look at is
    a claim rather than an artefact

A NOTE ON PROCESSING UPLOADS
----------------------------
Full REBEL extraction runs at ~1.3 seconds per sentence on CPU, so a single
20-page paper is several minutes and a browser tab is the wrong place for it.
The upload flow therefore defaults to the fast dependency-parse extractor and
offers REBEL as an explicit opt-in, with the honest warning attached. Large
corpora belong in the command-line batch job.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import (  # noqa: E402
    CHROMA_DIR,
    GRAPH_PATH,
    GRAPH_VIZ_PATH,
    RAW_DIR,
    TRIPLES_PATH,
    ensure_dirs,
)

st.set_page_config(
    page_title="KG-RAG - Knowledge Graph Augmented Retrieval",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
  .stApp { }
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
    from src.hybrid_retriever import HybridRetriever
    from src.pipeline import KGRagPipeline
    graph = load_graph()
    if graph is None:
        return None
    retriever = HybridRetriever(store=load_store(), kg=graph)
    return KGRagPipeline(retriever=retriever)


def llm_status() -> tuple[bool, str]:
    from src.llm import LLMClient
    client = LLMClient()
    return client.available, client.status()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> dict:
    st.sidebar.title("KG-RAG")
    st.sidebar.caption("Knowledge-graph augmented retrieval over research papers")

    # ---- corpus status ----
    store = load_store()
    graph = load_graph()
    pdfs = sorted(RAW_DIR.glob("*.pdf")) if RAW_DIR.exists() else []

    st.sidebar.subheader("Index status")
    col_a, col_b = st.sidebar.columns(2)
    col_a.metric("PDFs", len(pdfs))
    col_b.metric("Chunks", f"{store.count():,}")

    if graph is not None:
        stats = graph.stats()
        col_c, col_d = st.sidebar.columns(2)
        col_c.metric("Entities", f"{stats['nodes']:,}")
        col_d.metric("Relations", f"{stats['edges']:,}")
    else:
        st.sidebar.warning("No knowledge graph yet. Build it with:\n\n"
                           "`python src/knowledge_extractor.py`\n\n"
                           "`python src/knowledge_graph.py --rebuild`")

    available, status = llm_status()
    if available:
        st.sidebar.success(f"LLM ready - {status}")
    else:
        st.sidebar.error("No LLM key. Retrieval works; answers need a key in `.env`.\n\n"
                         "Get one free at aistudio.google.com/apikey")

    st.sidebar.divider()

    # ---- upload ----
    st.sidebar.subheader("Add documents")
    uploads = st.sidebar.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True,
        label_visibility="collapsed",
    )

    run_rebel = st.sidebar.checkbox(
        "Also run REBEL extraction", value=False,
        help="REBEL adds richer relations but runs at ~1.3s per sentence on CPU - "
             "several minutes per paper. Leave off for the fast dependency-parse "
             "extractor; use the command-line batch job for large corpora.",
    )

    if uploads and st.sidebar.button("Process documents", type="primary",
                                     width='stretch'):
        process_uploads(uploads, run_rebel)

    st.sidebar.divider()

    # ---- retrieval settings ----
    st.sidebar.subheader("Retrieval settings")
    settings = {
        # Default 12, not 8: compound questions ("which models beat X, AND who
        # proposed them") split one evidence budget across two sub-questions.
        # At 8, correctly-retrieved author edges were verified in testing to
        # rank 8-11 - just past the cutoff - so the pipeline reported "the
        # evidence does not name the authors" with the answer one slot away.
        "top_k": st.sidebar.slider("Results sent to the LLM", 3, 15, 12),
        "vector_k": st.sidebar.slider("Vector candidates", 5, 25, 10),
        "hops": st.sidebar.slider("Graph hops", 1, 3, 2,
                                  help="How far to walk from the query entities. "
                                       "3 hops finds more, and more noise."),
        "compare": st.sidebar.toggle("Compare with basic RAG", value=False,
                                     help="Run both systems side by side - the "
                                          "clearest demonstration of what the graph adds."),
    }
    return settings


def process_uploads(uploads, run_rebel: bool) -> None:
    """Save uploaded PDFs and rebuild both indexes."""
    ensure_dirs()
    progress = st.sidebar.progress(0.0, text="Saving uploads ...")

    saved: list[Path] = []
    for upload in uploads:
        destination = RAW_DIR / upload.name
        destination.write_bytes(upload.getbuffer())
        saved.append(destination)
    progress.progress(0.15, text=f"Saved {len(saved)} PDFs. Extracting text ...")

    from src.chunker import chunk_corpus
    from src.ingestion import load_corpus

    docs = load_corpus(clean=True)
    progress.progress(0.35, text=f"{len(docs)} documents cleaned. Chunking ...")

    chunks = chunk_corpus(docs, strategy="fixed")
    progress.progress(0.45, text=f"{len(chunks)} chunks. Embedding (this takes a moment) ...")

    store = load_store()
    store.add_chunks(chunks, show_progress=False)
    progress.progress(0.65, text="Vector store updated. Extracting relations ...")

    from src.knowledge_extractor import extract_all, save_triples

    triples = extract_all(use_rebel=run_rebel, use_domain=True, resume=False)
    save_triples(triples)
    progress.progress(0.9, text=f"{len(triples)} triples. Building graph ...")

    from src.knowledge_graph import KnowledgeGraph

    graph = KnowledgeGraph.from_triples(triples, verbose=False)
    graph.save()
    progress.progress(1.0, text="Done.")

    # Drop the cached instances so the new indexes are picked up.
    load_graph.clear()
    load_pipeline.clear()
    st.sidebar.success(f"Processed {len(docs)} documents, {len(chunks)} chunks, "
                       f"{len(triples)} triples.")
    st.rerun()


# ---------------------------------------------------------------------------
# Evidence rendering
# ---------------------------------------------------------------------------

def render_evidence_item(item: dict, index: int) -> None:
    method = item["retrieval_method"]
    citation = f"{item['source_file']} &middot; p.{item.get('page', '?')}"
    text = " ".join(item["text"].split())

    if method == "graph":
        subject, relation, obj = item.get("triple", ("", "", ""))
        hops = item.get("hop_distance", 1)
        st.markdown(
            f"""<div class="evidence-card graph">
            <span class="pill graph">GRAPH</span>
            <span class="cite">{citation} &middot; {hops} hop{'s' if hops > 1 else ''}</span>
            <div style="margin:6px 0"><b>{subject}</b> &mdash;[{relation}]&rarr; <b>{obj}</b></div>
            <div class="chain">{item.get('path_str', '')}</div>
            <div style="margin-top:8px;font-size:.88rem">{text}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        section = item.get("section") or ""
        section_label = f" &middot; {section}" if section else ""
        st.markdown(
            f"""<div class="evidence-card">
            <span class="pill vector">TEXT</span>
            <span class="cite">{citation}{section_label}</span>
            <div style="margin-top:8px;font-size:.88rem">{text}</div>
            </div>""",
            unsafe_allow_html=True,
        )


def render_answer_tab(settings: dict) -> None:
    pipeline = load_pipeline()

    st.title("Ask the corpus")
    st.caption("Vector search finds text that resembles your question. "
               "Graph traversal finds facts structurally connected to it. "
               "Both run, then a cross-encoder decides what actually matters.")

    if pipeline is None:
        st.warning("The knowledge graph has not been built yet, so only vector "
                   "search is available. Build it with `python src/knowledge_extractor.py` "
                   "then `python src/knowledge_graph.py --rebuild`.")
        return

    examples = [
        "Which models outperform Faster R-CNN, and who proposed them?",
        "What is the self-attention mechanism?",
        "Trace the architectural lineage from the Transformer to Swin Transformer.",
        "Which authors of detection papers also worked on segmentation?",
    ]
    chosen = st.selectbox("Try an example, or write your own below", [""] + examples,
                          format_func=lambda x: x or "-- pick an example --")

    question = st.text_input("Your question", value=chosen,
                             placeholder="e.g. Which models outperform YOLO?")

    if not st.button("Ask", type="primary") or not question.strip():
        return

    if settings["compare"]:
        render_comparison(question, settings)
        return

    pipeline.top_k = settings["top_k"]
    pipeline.vector_k = settings["vector_k"]
    pipeline.graph_hops = settings["hops"]

    with st.spinner("Running vector search and graph traversal ..."):
        started = time.time()
        state = pipeline.run(question)
        elapsed = time.time() - started

    render_state(state, elapsed)


def render_state(state: dict, elapsed: float) -> None:
    reranked = state.get("reranked_results") or []
    methods = [item["retrieval_method"] for item in reranked]

    if state.get("error"):
        st.error(state["error"])
    else:
        st.markdown("### Answer")
        st.markdown(state.get("answer") or "_no answer_")

    cols = st.columns(4)
    cols[0].metric("Vector hits", len(state.get("vector_results") or []))
    cols[1].metric("Graph evidence", len(state.get("graph_results") or []))
    cols[2].metric("Graph in final mix", f"{methods.count('graph')}/{len(methods)}")
    cols[3].metric("Time", f"{elapsed:.1f}s")

    sources = state.get("sources") or []
    if sources:
        st.markdown("**Sources cited:** " + " &nbsp;|&nbsp; ".join(
            f"`{s['source_file']} p.{s['page']}`" for s in sources
        ))

    st.divider()

    with st.expander(f"Final evidence sent to the model ({len(reranked)} items, "
                     f"after re-ranking)", expanded=False):
        st.caption("This is exactly what the LLM saw. Everything in the answer "
                   "should be traceable to one of these.")
        for index, item in enumerate(reranked):
            render_evidence_item(item, index)

    with st.expander(f"Vector search results ({len(state.get('vector_results') or [])})"):
        st.caption("Chunks whose embedding is closest to the question's embedding. "
                   "Note that these are the NEAREST chunks, which is not the same "
                   "as relevant ones.")
        for item in state.get("vector_results") or []:
            st.markdown(f"**{item['score']:.3f}** &middot; `{item['source_file']} "
                        f"p.{item['page']}`")
            st.caption(" ".join(item["text"].split())[:320])

    graph_results = state.get("graph_results") or []
    with st.expander(f"Graph evidence ({len(graph_results)})"):
        st.caption("Relationships reached by walking out from entities named in "
                   "your question. Multi-hop chains are facts no single paper states.")
        if not graph_results:
            st.info("No graph evidence - the question may not name an entity that "
                    "exists in the graph.")
        for item in graph_results:
            subject, relation, obj = item.get("triple", ("", "", ""))
            st.markdown(f"**{subject}** &mdash;[{relation}]&rarr; **{obj}** "
                        f"&nbsp; `{item['hop_distance']} hop` "
                        f"&nbsp; `{item['source_file']} p.{item['page']}`")
            st.caption(item.get("path_str", ""))

    with st.expander("What re-ranking changed"):
        st.caption("The cross-encoder reads the question and each passage TOGETHER, "
                   "which is more accurate than comparing two separately-computed "
                   "embeddings. Rows that moved up were under-ranked by the first pass.")
        rows = []
        for item in reranked:
            was = item.get("merged_rank", 0)
            now = item.get("final_rank", 0)
            rows.append({
                "final": now, "was": was, "move": was - now,
                "method": item["retrieval_method"],
                "ce score": round(item.get("rerank_score", 0), 3),
                "source": f"{item['source_file']} p.{item.get('page')}",
            })
        if rows:
            st.dataframe(rows, width='stretch', hide_index=True)


def render_comparison(question: str, settings: dict) -> None:
    from src.basic_rag import answer_question

    pipeline = load_pipeline()
    store = load_store()

    left, right = st.columns(2)

    with left:
        st.markdown("### Basic RAG")
        st.caption("Vector search only - the generic approach.")
        with st.spinner("Running basic RAG ..."):
            basic = answer_question(question, store, top_k=5, show_chunks=False)
        if basic.get("error"):
            st.error(basic["error"])
        else:
            st.markdown(basic["answer"])
        with st.expander(f"{len(basic.get('chunks', []))} chunks retrieved"):
            for chunk in basic.get("chunks", []):
                st.markdown(f"**{chunk['score']:.3f}** `{chunk['source_file']} "
                            f"p.{chunk['page']}`")
                st.caption(" ".join(chunk["text"].split())[:260])

    with right:
        st.markdown("### KG-RAG")
        st.caption("Vector search + graph traversal + cross-encoder re-ranking.")
        pipeline.top_k = settings["top_k"]
        pipeline.graph_hops = settings["hops"]
        with st.spinner("Running KG-RAG ..."):
            state = pipeline.run(question)
        if state.get("error"):
            st.error(state["error"])
        else:
            st.markdown(state.get("answer") or "_no answer_")

        reranked = state.get("reranked_results") or []
        graph_count = sum(1 for r in reranked if r["retrieval_method"] == "graph")
        st.caption(f"{graph_count} of {len(reranked)} final evidence items came "
                   f"from graph traversal.")
        with st.expander("Graph chains used"):
            for item in reranked:
                if item["retrieval_method"] == "graph":
                    st.markdown(f"`{item.get('path_str', '')}`")


# ---------------------------------------------------------------------------
# Graph tab
# ---------------------------------------------------------------------------

def render_graph_tab() -> None:
    import streamlit.components.v1 as components

    from src.graph_viz import build_visualisation, infer_entity_type

    graph = load_graph()
    st.title("Knowledge graph")

    if graph is None:
        st.warning("No graph built yet.")
        return

    stats = graph.stats()
    st.caption(f"{stats['nodes']:,} entities and {stats['edges']:,} relations extracted "
               f"from the corpus. Nodes are coloured by a type inferred from the "
               f"relations they participate in, and sized by how connected they are.")

    controls = st.columns([3, 1, 1])
    hub_names = [graph.display_name(node) for node, _ in stats["top_nodes"][:40]]
    entity = controls[0].selectbox("Centre the view on an entity", hub_names)
    hops = controls[1].slider("Hops", 1, 3, 2)
    max_nodes = controls[2].slider("Max nodes", 20, 150, 70)

    if st.button("Render graph", type="primary"):
        with st.spinner("Laying out the graph ..."):
            path = build_visualisation(graph, entity=entity, hops=hops,
                                       max_nodes=max_nodes, output=GRAPH_VIZ_PATH,
                                       height="720px")
        components.html(path.read_text(encoding="utf-8"), height=740, scrolling=False)

    st.divider()

    # ---- entity inspector ----
    st.subheader("Look up an entity")
    query = st.text_input("Entity name", placeholder="e.g. DETR, ResNet, COCO")
    if query:
        node = graph.find_entity(query)
        if node is None:
            st.warning(f"No entity matching {query!r}. Try a different spelling.")
        else:
            data = graph.graph.nodes[node]
            st.markdown(f"### {data.get('name', node)}")
            info = st.columns(3)
            info[0].metric("Connections", graph.graph.degree(node))
            info[1].metric("Mentions", data.get("mentions", 0))
            info[2].metric("Type", infer_entity_type(graph, node))

            aliases = sorted(data.get("aliases", ()))
            if aliases:
                st.caption("Also written as: " + ", ".join(f"`{a}`" for a in aliases[:12]))
            sources = sorted(data.get("sources", ()))
            if sources:
                st.caption("Appears in: " + ", ".join(f"`{s}`" for s in sources))

            outgoing = [(t, d["relation"]) for _s, t, d in graph.graph.out_edges(node, data=True)]
            incoming = [(s, d["relation"]) for s, _t, d in graph.graph.in_edges(node, data=True)]

            left, right = st.columns(2)
            with left:
                st.markdown("**Outgoing**")
                for target, relation in sorted(set(outgoing))[:25]:
                    st.markdown(f"&mdash;[{relation}]&rarr; {graph.display_name(target)}")
            with right:
                st.markdown("**Incoming**")
                for source, relation in sorted(set(incoming))[:25]:
                    st.markdown(f"{graph.display_name(source)} &mdash;[{relation}]&rarr;")

    st.divider()

    # ---- path finder: the multi-hop demo ----
    st.subheader("Find a connection between two entities")
    st.caption("This is the capability vector search structurally cannot provide: "
               "a chain of relationships linking two things no single document "
               "discusses together.")
    path_cols = st.columns([2, 2, 1])
    start = path_cols[0].text_input("From", placeholder="DETR")
    end = path_cols[1].text_input("To", placeholder="instance segmentation")
    if path_cols[2].button("Find path") and start and end:
        from src.knowledge_graph import format_path
        path = graph.find_path(start, end)
        if not path:
            st.warning("No path found within the hop limit.")
        else:
            st.success(f"Connected in {len(path)} hop{'s' if len(path) > 1 else ''}")
            st.code(format_path(graph, path), language=None)
            for step in path:
                st.markdown(f"- **{graph.display_name(step['from'])}** "
                            f"&mdash;[{step['relation']}]&rarr; "
                            f"**{graph.display_name(step['to'])}** "
                            f"&nbsp;`{step['source_file']} p.{step['page']}`")
                if step.get("sentence"):
                    st.caption(f"\"{step['sentence'][:250]}\"")


# ---------------------------------------------------------------------------
# Evaluation tab
# ---------------------------------------------------------------------------

def render_eval_tab() -> None:
    import json

    from src.utils import EVAL_RESULTS_PATH

    st.title("Evaluation")
    st.caption("Twenty questions in three difficulty tiers, run through both "
               "systems and scored by an LLM judge against hand-written reference "
               "answers.")

    if not EVAL_RESULTS_PATH.exists():
        st.info("No results yet. Run `python src/evaluate.py` to generate them.")
        return

    data = json.loads(EVAL_RESULTS_PATH.read_text(encoding="utf-8"))
    summary = data["summary"]

    rows = []
    for tier in ("simple", "medium", "hard", "ALL"):
        if tier not in summary:
            continue
        entry = summary[tier]
        rows.append({
            "tier": tier, "n": entry["n"],
            "basic RAG": entry["basic_rag"]["overall"],
            "KG-RAG": entry["kg_rag"]["overall"],
            "delta": entry["delta_overall"],
            "graph share": f"{entry['graph_share']:.0%}",
        })
    st.dataframe(rows, width='stretch', hide_index=True)

    st.markdown("A near-zero delta on **simple** questions is the correct result - "
                "it means adding the graph did not damage what already worked. "
                "The **hard** tier is where the graph has to earn its place.")

    st.divider()
    st.subheader("Per-question results")
    for record in data["results"]:
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
                st.caption(f"correctness {basic['correctness']} &middot; completeness "
                           f"{basic['completeness']} &middot; citations "
                           f"{basic['citation_accuracy']}")
                st.markdown(record["basic_rag"]["answer"] or "_none_")
            with right:
                st.markdown("**KG-RAG**")
                st.caption(f"correctness {kgrag['correctness']} &middot; completeness "
                           f"{kgrag['completeness']} &middot; citations "
                           f"{kgrag['citation_accuracy']}")
                st.markdown(record["kg_rag"]["answer"] or "_none_")


# ---------------------------------------------------------------------------

def main() -> None:
    ensure_dirs()
    settings = render_sidebar()

    ask_tab, graph_tab, eval_tab = st.tabs(
        ["Ask", "Knowledge graph", "Evaluation"]
    )
    with ask_tab:
        render_answer_tab(settings)
    with graph_tab:
        render_graph_tab()
    with eval_tab:
        render_eval_tab()


main()
