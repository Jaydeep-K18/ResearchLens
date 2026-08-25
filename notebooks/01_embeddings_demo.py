"""
Phase 2, Step 2.1 - what an embedding actually IS.

Run me:  python notebooks/01_embeddings_demo.py

THE ONE IDEA
------------
An embedding turns a piece of text into a fixed-length list of numbers - here,
384 of them - such that texts with similar MEANING end up numerically close
together.

Think of those 384 numbers as coordinates. On a 2D map, "Mumbai" and "Pune" are
close and "Mumbai" and "Reykjavik" are far. An embedding model does the same
thing, but with 384 axes instead of 2, and the axes encode meaning rather than
geography. "Dog" and "puppy" land near each other. "Dog" and "algebra" do not.

Nobody hand-designed those 384 axes. The model (all-MiniLM-L6-v2, 23M
parameters) learned them from a billion sentence pairs. No axis means anything
on its own - only the distances matter.

WHY THIS MATTERS FOR THE PROJECT
--------------------------------
This is the entire mechanism behind Phase 2's retrieval. When you ask a
question, we embed the question, then find the document chunks whose 384 numbers
are closest to the question's 384 numbers. That is what "vector search" is.

Watch for the key result below: sentences 1 and 2 score around 0.85 similarity
while sharing almost NO words. That gap between "shares words" and "means the
same thing" is exactly what keyword search cannot cross and embeddings can.

Then watch for the LIMITATION at the bottom of the output. It is the reason the
other five phases of this project exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import banner, setup_console  # noqa: E402

SENTENCES = [
    "YOLO is a real-time object detection system",          # 0
    "You Only Look Once detects objects quickly",           # 1
    "The weather is sunny today",                           # 2
    "Transformers use self-attention mechanisms",           # 3
    "Self-attention allows models to weigh input importance",  # 4
    "I like pizza",                                         # 5
]


def main() -> None:
    setup_console()

    print(banner("LOADING THE MODEL"))
    print("all-MiniLM-L6-v2: 23M parameters, ~90MB.")
    print("First run downloads it to your HuggingFace cache; later runs are instant.")
    print("It runs comfortably on CPU - no GPU needed anywhere in this project.\n")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # encode() does the whole job: tokenise -> run the transformer -> pool the
    # token vectors into ONE vector per sentence.
    # normalize_embeddings=True scales every vector to length 1, which means the
    # dot product between two vectors IS their cosine similarity. That saves a
    # division later and is what ChromaDB expects in Phase 2.2.
    embeddings = model.encode(SENTENCES, normalize_embeddings=True)

    print(banner("WHAT CAME BACK"))
    print(f"Shape: {embeddings.shape}  ->  {len(SENTENCES)} sentences x {embeddings.shape[1]} numbers each")
    print(f"\nSentence 0 is {SENTENCES[0]!r}")
    print("Its first 8 coordinates (of 384):")
    print(f"  {embeddings[0][:8].round(4).tolist()}")
    print("\nThose numbers are meaningless individually. Only distances between")
    print("whole vectors carry information. That is the whole trick.")

    # Cosine similarity of unit vectors == their dot product. Values run from
    # -1 (opposite) through 0 (unrelated) to 1 (identical).
    similarity = embeddings @ embeddings.T

    print(banner("PAIRWISE SIMILARITY MATRIX"))
    print("     " + "".join(f"{i:>8}" for i in range(len(SENTENCES))))
    for i, row in enumerate(similarity):
        print(f"  {i}  " + "".join(f"{value:>8.3f}" for value in row))

    print("\nThe sentences:")
    for i, sentence in enumerate(SENTENCES):
        print(f"  {i}: {sentence}")

    print(banner("READING THE NUMBERS"))

    def show(i: int, j: int, note: str) -> None:
        shared = _shared_content_words(SENTENCES[i], SENTENCES[j])
        print(f"\n  [{i}] {SENTENCES[i]}")
        print(f"  [{j}] {SENTENCES[j]}")
        print(f"      similarity = {similarity[i][j]:.3f}")
        print(f"      words in common: {', '.join(shared) if shared else 'NONE'}")
        print(f"      -> {note}")

    show(0, 1, "HIGHEST-scoring pair with zero shared words. The model knows 'YOLO'\n"
               "         and 'You Only Look Once' are the same system. Keyword search\n"
               "         scores this pair exactly ZERO.")
    show(3, 4, "HIGH. 'self-attention' and 'weigh input importance' are the same "
               "idea\n         phrased two ways.")
    show(0, 3, "MIDDLE. Both are deep-learning topics, so they share a neighbourhood,\n"
               "         but they are about different things.")
    show(2, 5, "LOW. Two unrelated everyday sentences. Note it is not 0.000 - real\n"
               "         embedding spaces are lumpy, and everything is a bit similar\n"
               "         to everything. What matters is RANKING, not absolute values.")
    show(0, 5, "LOW. Object detection vs pizza. Exactly what you would hope.")

    # ---------------------------------------------------------------------
    # A word about absolute numbers - worth internalising early.
    # ---------------------------------------------------------------------
    related = float(similarity[0][1])
    unrelated = float(similarity[0][5])

    print(banner("A TRAP: ABSOLUTE SIMILARITY VALUES ARE MEANINGLESS ALONE"))
    print(f"The YOLO pair scored {related:.3f}. You will see tutorials (and this")
    print("project's own spec) claim ~0.85 for a pair like that. Ours does not,")
    print("and nothing is wrong.")
    print("\nCosine similarity has no absolute scale. Every model has its own")
    print("distribution: some spread scores across the full 0-1 range, others")
    print("(like MiniLM) compress everything into a narrow band. Comparing a raw")
    print("score against a number you read somewhere tells you nothing.")
    print("\nWhat is meaningful is the CONTRAST within one model:")
    print(f"    related pair   (0 vs 1): {related:.3f}")
    print(f"    unrelated pair (0 vs 5): {unrelated:.3f}")
    print(f"    ratio: {related / max(unrelated, 1e-6):.1f}x")
    print("\nThat separation is what retrieval actually depends on. Ranking is")
    print("what we use; the raw number is just a means to sort by. This is also")
    print("why Phase 5 adds a cross-encoder re-ranker: when the top candidates")
    print("all sit in a narrow band, you need a sharper judge to order them.")

    print(banner("THE POINT"))
    print(f"Sentences 0 and 1 share no meaningful words yet score {related:.3f} -")
    print(f"the highest of any pair here, and {related / max(unrelated, 1e-6):.0f}x the unrelated baseline.")
    print("\nThe model learned MEANING, not word matching. That is why we can")
    print("ask 'how fast is object detection?' and retrieve a chunk that never")
    print("uses either of those words.")

    # ---------------------------------------------------------------------
    # The limitation that motivates Phases 3-6.
    # ---------------------------------------------------------------------
    print(banner("NOW THE LIMITATION - WHY THIS PROJECT HAS FIVE MORE PHASES"))

    facts = [
        "DETR outperforms Faster R-CNN on the COCO benchmark.",
        "Faster R-CNN was proposed by Ren et al.",
        "Ren et al. also worked on instance segmentation.",
    ]
    question = "Who worked on segmentation and also proposed a model that DETR beat?"

    fact_vectors = model.encode(facts, normalize_embeddings=True)
    question_vector = model.encode([question], normalize_embeddings=True)[0]
    scores = fact_vectors @ question_vector

    print(f"\nQuestion: {question}\n")
    print("Our three 'chunks' and how similar each is to the question:")
    for fact, score in zip(facts, scores):
        print(f"  {score:.3f}   {fact}")

    print("\nThe answer is 'Ren et al.' - but no single chunk says so. You have to")
    print("chain three facts together:")
    print("    DETR --beats--> Faster R-CNN --proposed by--> Ren et al. --worked on--> segmentation")
    print("\nVector search retrieves chunks INDEPENDENTLY. It has exactly one move:")
    print("'find text similar to this text'. It cannot follow a chain, because")
    print("nothing in the 384 numbers encodes 'and then follow that link'.")
    print("\nThat is a STRUCTURAL limit, not a tuning problem. No chunk size, no")
    print("bigger embedding model, and no cleverer prompt fixes it.")
    print("\nThe fix is a second memory that stores the LINKS themselves - a")
    print("knowledge graph. That is Phases 3 to 6.")


def _shared_content_words(a: str, b: str) -> list[str]:
    """Words the two sentences literally share, ignoring stopwords."""
    stop = {"is", "a", "the", "to", "in", "of", "and", "i", "on", "for", "use", "allows"}
    words_a = {w.strip(".,").lower() for w in a.split()} - stop
    words_b = {w.strip(".,").lower() for w in b.split()} - stop
    return sorted(words_a & words_b)


if __name__ == "__main__":
    main()
