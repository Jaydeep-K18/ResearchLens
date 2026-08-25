"""
Phase 6, Step 6.1 - LangGraph in miniature.

Run me:  python notebooks/03_langgraph_intro.py

WHAT LANGGRAPH IS
-----------------
A way to express a multi-step workflow as a GRAPH instead of as nested function
calls. Three concepts, and that is genuinely all of it:

  STATE  A dict that flows through the whole workflow. Each step reads what it
         needs and returns only the keys it changed; LangGraph merges the result
         back in. The state is the single shared workspace.

  NODE   A plain Python function: state in, partial-state-update out. Nothing
         magic - the functions below are ordinary and testable on their own.

  EDGE   "after this node, run that one". Edges define execution order. Two edges
         out of one node means both run, and they run IN PARALLEL.

THE OBVIOUS OBJECTION
---------------------
"Why not just call the functions in sequence?"

    results = retrieve(query)
    ranked  = rerank(query, results)
    answer  = synthesize(query, ranked)

For a straight line, that is honestly fine, and you should not reach for a
framework to get one. LangGraph earns its place in this project for four
specific reasons:

  1. PARALLELISM, DECLARED. Our pipeline runs vector search and graph search
     independently on the same query. As nested calls you would hand-roll
     threads and join them. As a graph you draw two edges, and the runtime
     handles the fan-out and the join.

  2. THE STATE IS THE AUDIT TRAIL. Every intermediate result stays in the state
     dict. When Phase 7's UI shows "here are the vector chunks, here is the graph
     evidence, here is what re-ranking did", it is just reading state keys.
     Nested calls throw intermediates away as soon as they are consumed.

  3. THE STRUCTURE IS INSPECTABLE. The pipeline can print its own diagram, and a
     node can be swapped or bypassed without rewriting call sites.

  4. IT EXTENDS TO CONDITIONAL FLOW. Adding "if retrieval found nothing, rewrite
     the query and try again" is one conditional edge. In nested calls it is a
     control-flow rewrite.

Point 1 is the real reason here. The rest is why it stays worth it.
"""

from __future__ import annotations

import sys
from pathlib import Path

from typing_extensions import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import banner, setup_console  # noqa: E402


# ---------------------------------------------------------------------------
# 1. The state
# ---------------------------------------------------------------------------
# A TypedDict describing every key the workflow can carry. This is a contract,
# not just documentation: it is how you know what a node is allowed to read.
class DemoState(TypedDict):
    user_input: str
    processed: str
    word_count: int
    output: str


# ---------------------------------------------------------------------------
# 2. The nodes
# ---------------------------------------------------------------------------
# Each takes the whole state and returns ONLY the keys it changed. LangGraph
# merges that partial update into the state - a node never mutates state in
# place, which is what makes parallel branches safe.

def receive_input(state: DemoState) -> dict:
    print(f"  [node: receive_input]   state in  -> {dict(state)}")
    cleaned = state["user_input"].strip()
    update = {"processed": cleaned}
    print(f"                          returning -> {update}")
    return update


def transform(state: DemoState) -> dict:
    print(f"  [node: transform]       state in  -> {dict(state)}")
    update = {
        "processed": state["processed"].upper(),
        "word_count": len(state["processed"].split()),
    }
    print(f"                          returning -> {update}")
    return update


def format_output(state: DemoState) -> dict:
    print(f"  [node: format_output]   state in  -> {dict(state)}")
    update = {"output": f"{state['processed']}  ({state['word_count']} words)"}
    print(f"                          returning -> {update}")
    return update


# ---------------------------------------------------------------------------
# 3. A parallel example - the shape our real pipeline uses
# ---------------------------------------------------------------------------

class ParallelState(TypedDict):
    query: str
    vector_hits: list[str]
    graph_hits: list[str]
    combined: list[str]


def fake_vector_search(state: ParallelState) -> dict:
    print("  [node: vector_search] running (writes 'vector_hits')")
    return {"vector_hits": [f"chunk about {state['query']}", "another chunk"]}


def fake_graph_search(state: ParallelState) -> dict:
    print("  [node: graph_search]  running (writes 'graph_hits')")
    return {"graph_hits": [f"{state['query']} --outperforms--> something"]}


def combine(state: ParallelState) -> dict:
    print("  [node: combine]       running (reads both)")
    return {"combined": state["vector_hits"] + state["graph_hits"]}


def main() -> None:
    setup_console()

    from langgraph.graph import END, START, StateGraph

    # =====================================================================
    # Example 1 - a linear three-step graph
    # =====================================================================
    print(banner("EXAMPLE 1: A LINEAR 3-STEP GRAPH"))

    builder = StateGraph(DemoState)
    builder.add_node("receive_input", receive_input)
    builder.add_node("transform", transform)
    builder.add_node("format_output", format_output)

    # START and END are LangGraph's built-in sentinel nodes.
    builder.add_edge(START, "receive_input")
    builder.add_edge("receive_input", "transform")
    builder.add_edge("transform", "format_output")
    builder.add_edge("format_output", END)

    workflow = builder.compile()

    print("\n  structure:  START -> receive_input -> transform -> format_output -> END\n")
    print("  executing, printing the state at every node:\n")

    result = workflow.invoke({"user_input": "  hello knowledge graph world  "})

    print(f"\n  final state: {dict(result)}")
    print(f"  final output: {result['output']!r}")

    print("\n  Notice how the state ACCUMULATED. 'user_input' was set at the start")
    print("  and is still there at the end; each node added its own keys without")
    print("  destroying anyone else's. That accumulation is the whole point - it")
    print("  is why Phase 7 can show every intermediate stage of the pipeline.")

    # =====================================================================
    # Example 2 - parallel branches
    # =====================================================================
    print(banner("EXAMPLE 2: PARALLEL BRANCHES (our actual pipeline shape)"))

    parallel_builder = StateGraph(ParallelState)
    parallel_builder.add_node("vector_search", fake_vector_search)
    parallel_builder.add_node("graph_search", fake_graph_search)
    parallel_builder.add_node("combine", combine)

    # TWO edges out of START = both nodes run in the same step, in parallel.
    parallel_builder.add_edge(START, "vector_search")
    parallel_builder.add_edge(START, "graph_search")
    # TWO edges into combine = it waits for both to finish (a join).
    parallel_builder.add_edge("vector_search", "combine")
    parallel_builder.add_edge("graph_search", "combine")
    parallel_builder.add_edge("combine", END)

    parallel_workflow = parallel_builder.compile()

    print("""
  structure:
                    START
                   /     \\
        vector_search   graph_search      <- these two run in parallel
                   \\     /
                    combine               <- waits for both
                      |
                     END
""")
    parallel_result = parallel_workflow.invoke({"query": "DETR"})
    print(f"\n  combined: {parallel_result['combined']}")

    print("\n  The two search nodes never see each other's output and never")
    print("  conflict, because each writes a DIFFERENT state key. That is the")
    print("  rule for parallel branches: if two nodes running at the same time")
    print("  wrote the same key, LangGraph would raise an error rather than")
    print("  silently letting one clobber the other.")

    # =====================================================================
    print(banner("HOW THIS MAPS ONTO src/pipeline.py"))
    print("""
  DemoState                 ->  KGRagState (query, vector_results, graph_results,
                                merged_results, reranked_results, answer, sources)

  receive_input             ->  retrieve_vector   (Phase 2 ChromaDB search)
                            ->  retrieve_graph    (Phase 5.1 graph walk)
  transform                 ->  merge_and_rerank  (Phase 5.3 cross-encoder)
                            ->  synthesize        (Gemini Flash, grounded prompt)
  format_output             ->  format_output     (citations + provenance)

  Same three concepts. The only difference is that the nodes do real work.
""")


if __name__ == "__main__":
    main()
