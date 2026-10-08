# KG-RAG: Knowledge Graph-Augmented Retrieval System

## Description

KG-RAG is a question-answering system for research papers. Normal RAG can find relevant text, but it struggles with multi-hop questions that need facts connected across several papers.

This project solves that by searching two sources together: a **vector store** for meaning and a **knowledge graph** for relationships. An LLM then combines the results into a clear answer with sources.

## Tech Stack

- **Language:** Python
- **NLP:** spaCy, REBEL (Hugging Face Transformers)
- **Vector Search:** Sentence Transformers, ChromaDB
- **Knowledge Graph:** NetworkX, pyvis
- **Orchestration:** LangGraph, Cross-Encoder Re-ranking
- **LLM:** Gemini Flash, Groq
- **UI:** Streamlit

## Architecture
<img width="431" height="283" alt="krag archi" src="https://github.com/user-attachments/assets/83422d50-edd8-4b25-9c94-81708ebd3c27" />


## Key Features

- Answers multi-hop questions across multiple research papers
- Combines vector search and knowledge graph search
- Gives answers with source citations
- Interactive knowledge graph visualization
- Side-by-side comparison of Basic RAG vs KG-RAG
- Runs on a CPU-only laptop




## Challenges Faced

- Wrong but confident answers: The LLM sometimes made up links between papers when the retrieved context was weak.
- Noisy knowledge graph: PDF tables and references produced junk facts, and the same model showed up under different names.
- Retrieval missing the connection: Vector search missed the link between facts, and graph search pulled in too much.
- Slow processing on CPU: Relation extraction was slow without a GPU, so papers were processed in overnight batches.
## Future Scope

1. Move the knowledge graph to Neo4j to handle larger datasets.
2. Extend the system to other domains like medical or legal research.
