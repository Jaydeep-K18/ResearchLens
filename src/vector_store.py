"""
Phase 2, Step 2.2 - the vector store. This is MEMORY 1 of the two the system keeps.

WHAT A VECTOR DATABASE ACTUALLY IS
----------------------------------
Phase 2.1 showed that a sentence can be turned into 384 numbers, and that texts
meaning similar things land at nearby coordinates. A vector database is simply a
database built around that fact. It stores:

    id  ->  (384 numbers, the original text, some metadata)

and answers exactly one kind of question well:

    "given these 384 numbers, which stored rows have the closest 384 numbers?"

That is all vector search is. Your question becomes a point in 384-dimensional
space, and the database hands back its nearest neighbours.

WHY IT IS NOT JUST A for-LOOP
-----------------------------
With 929 chunks you genuinely could compare against all of them - that is under
a millisecond of numpy. The reason to use ChromaDB is what happens at 300 papers
(~40,000 chunks) and beyond: it builds an HNSW index, a layered graph structure
that finds approximate nearest neighbours in roughly log time instead of linear.
"Approximate" is the trade - it may miss a true neighbour occasionally, in
exchange for staying fast as the corpus grows. It also handles persistence,
metadata filtering, and batching for us.

WHY WE PASS OUR OWN EMBEDDINGS
------------------------------
ChromaDB can embed text for you. We do not let it. We call
SentenceTransformer.encode() ourselves and hand ChromaDB the finished vectors,
because:
  - the SAME model instance is reused in Phase 5's hybrid retriever, so we load
    those 90MB once instead of twice;
  - the embedding step becomes explicit and inspectable rather than hidden;
  - we are never surprised by a Chroma upgrade quietly changing the default model
    and silently invalidating every vector already on disk.

WHERE THIS SITS IN THE ARCHITECTURE
-----------------------------------
    chunker.py -> vector_store.py -> basic_rag.py         (Phase 2 baseline)
                                  -> hybrid_retriever.py  (Phase 5, merged with the graph)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import CHROMA_DIR, banner, setup_console  # noqa: E402

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "kg_rag_chunks"
DEFAULT_BATCH_SIZE = 128

# Cached across calls so repeated instantiation does not reload 90MB of weights.
_model_cache: dict[str, object] = {}


def get_embedding_model(name: str = EMBEDDING_MODEL_NAME):
    """Load (once) and return the sentence-transformer used everywhere in this project."""
    if name not in _model_cache:
        from sentence_transformers import SentenceTransformer
        _model_cache[name] = SentenceTransformer(name)
    return _model_cache[name]


class VectorStore:
    """A thin, explicit wrapper over a persistent ChromaDB collection."""

    def __init__(
        self,
        persist_dir: Path | str = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        model_name: str = EMBEDDING_MODEL_NAME,
    ) -> None:
        import chromadb

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.model = get_embedding_model(model_name)

        # PersistentClient writes to disk, so the embeddings survive between runs.
        # Embedding the corpus takes real time; doing it once is the whole point.
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

        # hnsw:space="cosine" tells Chroma to measure distance as
        # 1 - cosine_similarity. It matters that this matches how we normalise
        # our vectors (unit length), so that "distance 0" means "identical
        # meaning" and scores are comparable to the Phase 2.1 demo.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # -- writing -----------------------------------------------------------

    def add_chunks(self, chunks: list[dict], batch_size: int = DEFAULT_BATCH_SIZE,
                   show_progress: bool = True) -> int:
        """
        Embed chunks and store them.

        Uses chunk_id as the ChromaDB id and UPSERTS, so re-running ingestion
        updates existing rows instead of creating duplicates. That makes this
        function safe to run repeatedly while iterating - which you will.
        """
        if not chunks:
            return 0

        from tqdm import tqdm

        # ChromaDB metadata values must be str/int/float/bool - None is rejected.
        # Our `section` field is "" for fixed chunks, which is why chunker.py uses
        # an empty string rather than None.
        def to_metadata(chunk: dict) -> dict:
            return {
                "source_file": chunk["source_file"],
                "chunk_index": int(chunk["chunk_index"]),
                "page": int(chunk.get("page", 1)),
                "section": str(chunk.get("section") or ""),
                "strategy": str(chunk.get("strategy", "fixed")),
                "start_char": int(chunk.get("start_char", 0)),
                "end_char": int(chunk.get("end_char", 0)),
            }

        batches = range(0, len(chunks), batch_size)
        iterator = tqdm(batches, desc="Embedding chunks", unit="batch") if show_progress else batches

        for start in iterator:
            batch = chunks[start:start + batch_size]
            texts = [c["text"] for c in batch]

            # normalize_embeddings=True gives unit-length vectors, which is what
            # cosine distance assumes. Skipping it makes long chunks look
            # systematically more or less similar purely because of their length.
            vectors = self.model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )

            self.collection.upsert(
                ids=[c["chunk_id"] for c in batch],
                embeddings=[v.tolist() for v in vectors],
                documents=texts,
                metadatas=[to_metadata(c) for c in batch],
            )

        return len(chunks)

    # -- reading -----------------------------------------------------------

    def search(self, query: str, top_k: int = 5, where: dict | None = None) -> list[dict]:
        """
        Embed the query and return the top_k most similar chunks.

        Returns dicts shaped for the hybrid retriever in Phase 5:

            {
              "chunk_id", "text", "source_file", "page", "section",
              "score",            # cosine similarity, higher = better
              "distance",         # what Chroma actually returned
              "retrieval_method", # always "vector" here
              "rank",
            }
        """
        if self.count() == 0:
            return []

        query_vector = self.model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0]

        results = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=min(top_k, self.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Chroma returns parallel lists nested one level deep (one per query).
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        hits: list[dict] = []
        for rank, (chunk_id, text, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances)
        ):
            hits.append({
                "chunk_id": chunk_id,
                "text": text,
                "source_file": metadata.get("source_file", "unknown"),
                "page": metadata.get("page", 1),
                "section": metadata.get("section", ""),
                # With hnsw:space="cosine", distance = 1 - cosine_similarity,
                # so inverting it recovers the familiar similarity score.
                "score": 1.0 - float(distance),
                "distance": float(distance),
                "retrieval_method": "vector",
                "rank": rank,
            })
        return hits

    # -- housekeeping ------------------------------------------------------

    def count(self) -> int:
        return self.collection.count()

    def stats(self) -> dict:
        """Summary of what is currently stored."""
        total = self.count()
        if total == 0:
            return {"total_chunks": 0, "files": {}, "strategies": {}}

        everything = self.collection.get(include=["metadatas"])
        files: dict[str, int] = {}
        strategies: dict[str, int] = {}
        for metadata in everything["metadatas"]:
            files[metadata.get("source_file", "?")] = files.get(metadata.get("source_file", "?"), 0) + 1
            key = metadata.get("strategy", "?")
            strategies[key] = strategies.get(key, 0) + 1
        return {"total_chunks": total, "files": files, "strategies": strategies}

    def reset(self) -> None:
        """Delete and recreate the collection. Used when re-chunking from scratch."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )


# ---------------------------------------------------------------------------
# Build helper - used here, by app.py in Phase 7, and by evaluate.py
# ---------------------------------------------------------------------------

def build_from_corpus(strategy: str = "fixed", reset: bool = False,
                      store: VectorStore | None = None) -> VectorStore:
    """Run the whole Phase 1 -> Phase 2 path: PDFs -> clean text -> chunks -> ChromaDB."""
    from src.chunker import chunk_corpus
    from src.ingestion import load_corpus

    store = store or VectorStore()
    if reset:
        store.reset()

    print("Loading and cleaning PDFs ...")
    docs = load_corpus(clean=True)
    if not docs:
        raise SystemExit("No documents found. Run: python scripts/fetch_papers.py")

    chunks = chunk_corpus(docs, strategy=strategy)
    print(f"{len(docs)} documents -> {len(chunks)} chunks ({strategy} strategy)")

    store.add_chunks(chunks)
    print(f"Stored. Collection now holds {store.count()} chunks.")
    return store


if __name__ == "__main__":
    setup_console()

    strategy = sys.argv[1] if len(sys.argv) > 1 else "fixed"
    store = build_from_corpus(strategy=strategy, reset=True)

    stats = store.stats()
    print(banner("VECTOR STORE CONTENTS"))
    print(f"  total chunks: {stats['total_chunks']}")
    print(f"  strategies:   {stats['strategies']}")
    print("  per file:")
    for filename, count in sorted(stats["files"].items()):
        print(f"     {filename:<36} {count:>5}")

    # The three probes from the roadmap. The third is the important one: a good
    # retriever must also be able to find NOTHING convincingly.
    probes = [
        ("object detection speed", "should surface YOLO / real-time detection text"),
        ("attention mechanism", "should surface Transformer / self-attention text"),
        ("banana recipe", "should surface nothing relevant - watch the scores drop"),
    ]

    for query, expectation in probes:
        print(banner(f'QUERY: "{query}"'))
        print(f"  expectation: {expectation}\n")
        for hit in store.search(query, top_k=4):
            snippet = " ".join(hit["text"].split())[:150]
            print(f"  [{hit['rank']}] score={hit['score']:.3f}  "
                  f"{hit['source_file']} p.{hit['page']}")
            print(f"      {snippet}...\n")

    print(banner("NOTE ON THE 'banana recipe' SCORES"))
    print("They are lower than the real queries, but they are not zero, and the")
    print("chunks returned are not empty. A vector database ALWAYS returns its")
    print("k nearest neighbours - 'nearest' does not mean 'near'. Nothing in the")
    print("query stage knows the corpus has no answer.")
    print("\nThat is why Phase 6's prompt explicitly instructs Gemini to say when")
    print("the context does not contain the answer: the guard against confidently")
    print("answering from irrelevant chunks lives in the PROMPT, not the search.")
