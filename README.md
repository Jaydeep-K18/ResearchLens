# KG-RAG: Knowledge Graph-Augmented Retrieval

## Description

Standard RAG ("chat with your PDF") works well for simple lookups, but it can't answer **multi-hop questions** that need connecting facts across several documents — for example, *"Which YOLO variants were outperformed by ViT-based models, and who wrote those papers?"*

KG-RAG solves this by building two memories of a research-paper corpus:

- **Vector store** — semantic search over text chunks (the "meaning" index)
- **Knowledge graph** — `(subject, relation, object)` facts extracted from every sentence (the "relationship" index)

At query time, both are searched together, the results are re-ranked, and an LLM writes a grounded answer with cited sources.

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python |
| PDF & NLP | PyMuPDF, spaCy |
| Relation extraction | REBEL (Hugging Face Transformers) |
| Embeddings & vector DB | sentence-transformers (all-MiniLM-L6-v2), ChromaDB |
| Knowledge graph | NetworkX, thefuzz (entity deduplication), pyvis |
| Retrieval & orchestration | Cross-encoder (ms-marco-MiniLM-L-6-v2), LangGraph |
| LLM | Gemini Flash (primary), Groq / Llama 3 (fallback) |
| UI | Streamlit |

## Architecture

<img width="431" height="283" alt="krag archi" src="https://github.com/user-attachments/assets/2a54214b-ad58-469d-b0d0-545ed4a956bf" />


## Key Features

- **Hybrid retrieval** — vector search and graph traversal run in parallel, then get merged and re-ranked
- **Multi-hop reasoning** — follows 1–3 hop relationship chains across documents
- **Cited answers** — every answer points back to its source file and page
- **Interactive graph view** — explore entities and relations visually, filter and search nodes
- **Basic RAG vs KG-RAG comparison** — side-by-side answers, plus an evaluation harness scored with LLM-as-judge
- **Runs on a CPU-only laptop** — no GPU needed; heavy generation goes through free-tier APIs


## Challenges Faced

1. **Basic RAG couldn't connect the dots** — For multi-hop questions, vector search returned chunks about YOLO and chunks about ViT separately but never the link between them. The LLM would either say the context wasn't enough or guess a connection that wasn't in the papers. Tuning chunk size and prompts didn't fix it, which is why the knowledge graph was added instead.
2. **Noisy facts from research PDFs** — Two-column layouts, equations, tables and reference lists produced broken sentences, so REBEL extracted junk triples. The same model also appeared under different names ("ViT", "Vision Transformer", "ViT-B/16") and ended up as separate nodes. This needed text cleaning, skipping reference sections, filtering weak triples and fuzzy entity merging. Since REBEL runs at 2–3 seconds per sentence on CPU, every fix meant a long re-run, so extraction was done in overnight batches.
3. **Getting both paths to work together** — Graph traversal from hub nodes like "COCO" or "object detection" pulled in hundreds of loosely related facts and drowned out good text chunks. On other questions, spaCy didn't recognise model names in the query, so the graph returned nothing at all. Capping hop depth, falling back to vector-only search when no entity matched, and re-ranking both kinds of evidence with a cross-encoder into a single list made the combined results reliable.

## Future Scope

1. Move the graph from NetworkX to **Neo4j** to handle much larger corpora and richer graph queries.
2. Extend beyond CV/ML papers to other domains such as **medical or legal research**, with domain-specific relation extraction.
