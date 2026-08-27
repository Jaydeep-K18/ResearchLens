"""
Phase 2, Step 2.3 - basic RAG. The generic version that exists in 10,000 repos.

WHAT THIS IS
------------
The complete standard RAG loop, in about 40 lines of real logic:

    question -> embed -> find 5 nearest chunks -> paste them into a prompt
             -> ask the LLM to answer using only those chunks -> print

That is it. That is what every "chat with your PDF" tutorial builds, and it
genuinely works well for one particular shape of question: "what does this
corpus say about X?", where X is discussed in one place, in words similar to the
question.

WHY IT IS IN THIS PROJECT
-------------------------
Two reasons, and neither is "because we need a chatbot":

1. IT IS THE BASELINE. In Phase 7 we score this system and the full KG-RAG
   system on the same 20 questions. Without this file there is no comparison,
   and without a comparison there is no evidence the extra five phases were
   worth building. A KG-RAG project with no baseline is an assertion; with one,
   it is a result.

2. IT SHOWS THE FAILURE FIRSTHAND. Run --demo-limitation below and watch what
   happens on a multi-hop question. You need to see it fail on real papers, not
   just read that it would.

THE CEILING, PRECISELY
----------------------
Basic RAG can perform exactly ONE lookup: "text similar to this text". A question
whose answer requires connecting fact A in paper 1 to fact B in paper 2 through a
relationship that is never stated in either has no chunk to retrieve. The system
does not know it is missing something - it retrieves its 5 nearest neighbours,
they look topically plausible, and the LLM either hedges or invents a connection.

That is a structural limit of vector-only retrieval, and it is what Phases 3-6 fix.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llm import LLMUnavailable, get_llm  # noqa: E402
from src.utils import banner, setup_console  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402

DEFAULT_TOP_K = 5

# The instruction block is the only defence basic RAG has against hallucination.
# Note what each line is doing:
#   - "ONLY the context"    stops the model answering from its training memory
#   - "say so explicitly"   gives it a licence to fail, so it does not invent
#   - "[Source: file, p.N]" makes every claim checkable against the corpus
PROMPT_TEMPLATE = """You are a research assistant answering questions about a collection of documents the user has uploaded.

Answer the QUESTION using ONLY the CONTEXT below. Follow these rules strictly:
1. Use only facts present in the CONTEXT. Do not use outside knowledge.
2. Cite every factual claim as [Source: filename, p.N] using the labels shown.
3. If the CONTEXT does not contain enough information to answer, say so
   explicitly and state what is missing. Do not guess or fill gaps.
4. Be specific and concise. Prefer concrete numbers and names over generalities.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def format_context(chunks: list[dict]) -> str:
    """Lay the retrieved chunks out so the model can cite them precisely."""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        label = f"[{i}] [Source: {chunk['source_file']}, p.{chunk['page']}]"
        text = " ".join(chunk["text"].split())
        blocks.append(f"{label}\n{text}")
    return "\n\n".join(blocks)


def answer_question(
    question: str,
    store: VectorStore,
    top_k: int = DEFAULT_TOP_K,
    show_chunks: bool = True,
) -> dict:
    """
    The full basic-RAG loop: retrieve -> prompt -> generate.

    Returns {question, chunks, answer, provider, error} so Phase 7 can score it.
    """
    chunks = store.search(question, top_k=top_k)

    if show_chunks:
        print(f"\n  Retrieved {len(chunks)} chunks by vector similarity:")
        for chunk in chunks:
            snippet = " ".join(chunk["text"].split())[:110]
            print(f"    [{chunk['rank']}] {chunk['score']:.3f}  "
                  f"{chunk['source_file']} p.{chunk['page']}  {snippet}...")

    if not chunks:
        return {"question": question, "chunks": [], "answer": "", "provider": None,
                "error": "vector store is empty - run: python src/vector_store.py"}

    prompt = PROMPT_TEMPLATE.format(context=format_context(chunks), question=question)

    llm = get_llm()
    try:
        text = llm.generate(prompt)
        return {"question": question, "chunks": chunks, "answer": text,
                "provider": llm.last_provider, "error": None}
    except LLMUnavailable as exc:
        return {"question": question, "chunks": chunks, "answer": "",
                "provider": None, "error": str(exc)}


def print_answer(result: dict) -> None:
    print("\n  " + "-" * 70)
    if result["error"]:
        print(f"  [no answer generated] {result['error']}")
    else:
        print(f"  ANSWER  (via {result['provider']}):\n")
        for line in result["answer"].splitlines():
            print(f"    {line}")
    print("  " + "-" * 70)


# ---------------------------------------------------------------------------
# The demonstration that motivates the rest of the project
# ---------------------------------------------------------------------------

MULTI_HOP_QUESTION = (
    "Which authors of object detection papers also contributed to segmentation research?"
)

SIMPLE_QUESTION = "What is the main architectural idea behind the Transformer?"


def demo_limitation(store: VectorStore) -> None:
    """
    Run one question basic RAG handles well and one it cannot, back to back.

    The contrast is the point. Both retrievals look equally healthy - similar
    scores, plausible sources. Only one of them actually contains the answer.
    """
    print(banner("1. A QUESTION BASIC RAG HANDLES WELL"))
    print(f"  {SIMPLE_QUESTION}")
    print("\n  Why it works: the answer sits in ONE place, phrased in words close")
    print("  to the question. That is a single similarity lookup - exactly the")
    print("  one move vector search has.")
    print_answer(answer_question(SIMPLE_QUESTION, store))

    print(banner("2. A QUESTION BASIC RAG STRUCTURALLY CANNOT HANDLE"))
    print(f"  {MULTI_HOP_QUESTION}")
    print("\n  Why it fails: answering needs a CHAIN -")
    print("      find detection papers -> get their author lists")
    print("      -> find segmentation papers -> get THEIR author lists")
    print("      -> intersect the two sets")
    print("  No single chunk contains that intersection, because no author ever")
    print("  wrote 'I worked on both detection and segmentation' in a paper.")
    print("  The fact exists only in the RELATIONSHIPS BETWEEN documents.")
    result = answer_question(MULTI_HOP_QUESTION, store)
    print_answer(result)

    print(banner("WHAT TO NOTICE"))
    print("  Look at the retrieved chunks for question 2. They are topically")
    print("  reasonable - detection papers, segmentation papers - and their")
    print("  similarity scores are not obviously worse than question 1's.")
    print("\n  Retrieval did not fail loudly. It failed SILENTLY, returning the")
    print("  5 nearest chunks as always, none of which holds the connection.")
    print("  The LLM then either hedges or invents one.")
    print("\n  You cannot fix this with a better prompt, a bigger embedding model,")
    print("  or a different chunk size. The information was never in any single")
    print("  chunk to retrieve.")
    print("\n  The fix is a second index that stores the RELATIONSHIPS themselves,")
    print("  so the system can walk from paper to author to paper.")
    print("  That is Phase 3 (extract relations) and Phase 4 (build the graph).")


def interactive_loop(store: VectorStore, top_k: int) -> None:
    print(banner("BASIC RAG - interactive"))
    print("  Ask a question about the corpus. Type 'quit' (or Ctrl-C) to exit.")
    print(f"  Corpus: {store.count()} chunks.\n")

    while True:
        try:
            question = input("  question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  bye.")
            return
        if question.lower() in {"quit", "exit", "q"}:
            print("  bye.")
            return
        if not question:
            continue
        print_answer(answer_question(question, store, top_k=top_k))
        print()


def main() -> None:
    setup_console()

    parser = argparse.ArgumentParser(description="Basic RAG over the paper corpus")
    parser.add_argument("--question", "-q", help="ask one question and exit")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--demo-limitation", action="store_true",
                        help="run the simple vs multi-hop comparison")
    args = parser.parse_args()

    store = VectorStore()
    if store.count() == 0:
        raise SystemExit("Vector store is empty. Run first:  python src/vector_store.py")

    llm = get_llm()
    if not llm.available:
        print(banner("NO LLM KEY CONFIGURED"))
        print(llm.setup_hint())
        print("\nRetrieval still works - showing what WOULD be sent to the model.\n")

    if args.demo_limitation:
        demo_limitation(store)
    elif args.question:
        print_answer(answer_question(args.question, store, top_k=args.top_k))
    else:
        interactive_loop(store, args.top_k)


if __name__ == "__main__":
    main()
