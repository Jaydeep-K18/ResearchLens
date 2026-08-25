# PROJECT_CONTEXT.md — KG-RAG: Knowledge Graph-Augmented Retrieval System

> **READ THIS ENTIRE FILE BEFORE WRITING ANY CODE.**
> This is the single source of truth for what we're building, why, how, and in
> what order. Every architectural decision, library choice, and constraint is
> documented here. If anything is unclear, ask before proceeding — don't guess.

---

## 1. Who is building this and on what hardware

### The builder
- Final-year B.Tech (Information Technology) student at VJTI Mumbai, graduating 2027.
- Has 2 published Computer Vision research papers — comfortable with Python,
  PyTorch, and ML concepts, but **new to RAG, knowledge graphs, vector
  databases, and LLM orchestration**. This is a learning project, not just a
  build. Every step must be explained, not just executed.
- Domain expertise: object detection, image segmentation, transformers in
  vision — the test corpus will be CV/ML research papers the builder can
  evaluate answer quality on, because he knows the field.

### Hardware constraints (non-negotiable)
- **CPU:** AMD Ryzen 7 5700U (8 cores, 16 threads)
- **RAM:** 16GB
- **GPU:** Integrated AMD Radeon Graphics — **no dedicated GPU at all**
- **Storage:** ~150GB free SSD
- **Consequence:** everything must run on CPU. No local LLM inference for
  anything beyond small utility models (embeddings, NER, cross-encoders — all
  under ~500M parameters). All heavy LLM text generation goes through APIs.

### API access for LLM generation
- **Primary:** Gemini Flash API (free tier, generous rate limits) via
  `google-generativeai` package
- **Secondary/backup:** Groq API (Llama 3.x, free tier) via `groq` package
- API keys stored in `.env` file, loaded via `python-dotenv`, **never hardcoded**

### Development approach
- **Incremental, phase by phase.** Do NOT jump ahead or generate the whole
  project at once.
- Build one module → test it → explain it → confirm understanding → move on.
- Each phase ends with the builder being able to explain, in his own words,
  what was built and why — not just have working code.
- Follow the phase order strictly (1 → 2 → 3 → 4 → 5 → 6 → 7). Earlier
  phases are dependencies for later ones.

---

## 2. The problem this project solves

### What is RAG?
RAG (Retrieval-Augmented Generation) is a pattern where, instead of asking
an LLM to answer from its training memory (which hallucinates), you first
SEARCH a database of your own documents to find relevant passages, then give
those passages to the LLM as context, and ask it to answer using ONLY that
context. Think of it as giving the LLM an open-book exam instead of a
closed-book exam.

### What is standard/basic RAG?
1. Take documents → split into small chunks of text
2. Convert each chunk into a list of numbers (an "embedding") that captures
   its meaning
3. Store all embeddings in a vector database (ChromaDB)
4. When user asks a question → convert question to embedding → find the
   chunks with the most similar embeddings → give those chunks to an LLM →
   LLM writes an answer based only on those chunks

This works for single-fact lookups ("What is the attention mechanism?",
"What results did ResNet report?"). **This is what every "chat with your
PDF" app, every YouTube RAG tutorial, and every beginner portfolio project
does.** It takes 1–2 days to build and has ~80 lines of core code.

### Where standard RAG fails — the actual problem
Standard RAG **cannot answer multi-hop questions** — questions where the
answer requires connecting information across multiple documents through
relationships that aren't expressed as text similarity.

**Example question standard RAG cannot answer:**
> "Which YOLO variants were outperformed by Vision Transformer-based models,
> and who are the researchers behind those winning papers?"

This requires following a CHAIN of relationships:
1. Find YOLO papers in the corpus
2. Find papers that benchmark against YOLO (comparison relationship)
3. Among those, find which are ViT-based (category relationship)
4. Among those, find which ones beat YOLO (performance relationship)
5. Find the authors of those winning papers (authorship relationship)

No single chunk of text contains this full chain. No amount of chunk-size
tuning, better embeddings, or fancier prompts fixes this. It's a
**structural limitation** of vector-only retrieval — you can only do ONE
similarity lookup, not follow a chain of connections.

### Why this matters in the real world
- **PhD literature reviews:** researchers spend 3–6 months manually mapping
  how papers in a field relate. This does it in hours.
- **Drug/medical research:** "which drug trials cited compound X AND reported
  conflicting results?" requires traversing relationships across thousands
  of studies.
- **Finding research gaps:** if paper A contradicts paper B and nobody has
  resolved it, that unresolved gap is a publishable opportunity. This tool
  surfaces those gaps automatically.
- **Industry relevance:** Microsoft (GraphRAG), Google, Neo4j, and Notion
  are actively building KG-RAG systems in production. Papers on KG-RAG
  appear at ACL, EMNLP, and ICLR.

---

## 3. What we are building — the solution

### In plain terms
A system that ingests a large corpus of research papers (100s of PDFs),
builds two parallel "memories" of them, and lets a user ask both simple and
complex multi-hop questions across the entire corpus, getting cited,
grounded answers.

### The two memories (this is the core architectural insight)

**Memory 1 — Vector store ("meaning" index):**
Every document is split into small overlapping chunks, each chunk is
converted into an embedding (a list of 384 numbers that captures meaning),
and stored in ChromaDB. At query time, the question is also embedded and
the most semantically similar chunks are retrieved. This is standard RAG.
It handles "what does paper X say about Y?" type questions well.

**Memory 2 — Knowledge graph ("relationship" index):**
Every sentence in every document is run through an NLP model (REBEL) that
extracts structured facts as triples: `(subject, relation, object)` — e.g.,
`(DETR, outperforms, Faster R-CNN)`, `(Swin Transformer, proposed by, Liu et al.)`.
These triples become nodes (entities) and labeled edges (relationships) in a
graph (NetworkX). At query time, entities are extracted from the question,
and the system WALKS the graph — following chains of relationships across
documents (1-hop, 2-hop, 3-hop traversals). This is what enables multi-hop
reasoning.

**At query time, BOTH memories are searched simultaneously, results are
merged, re-ranked by a cross-encoder for precision, and fed to Gemini Flash
for synthesis into a cited answer.**

Anyone can build Memory 1 — it's the generic project everyone has.
Memory 2 + the combination is what makes this project genuinely hard,
interesting, and unique. **Memory 2 is the actual project.**

### What a real session looks like
1. User uploads 300 CV research papers
2. System processes them (takes time — runs overnight for full corpus):
   - Chunks + embeds all text → ChromaDB
   - Extracts triples from all sentences → NetworkX graph
3. User asks: "Which YOLO variants were outperformed by Vision Transformer
   models, and who were the researchers behind those ViT papers?"
4. System:
   - Vector search finds chunks mentioning YOLO and ViT (but can't connect them)
   - Graph traversal: YOLO papers → "outperformed-by" edges → ViT papers →
     "authored-by" edges → specific researchers (3 hops)
   - Cross-encoder re-ranks all results by relevance
   - Gemini Flash synthesizes a specific, cited answer with source files and
     page numbers

---

## 4. Full architecture diagram

```
                              ┌─────────────────────────┐
                              │   Raw documents (PDFs)   │
                              │   in data/raw/ folder    │
                              └────────────┬────────────┘
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     │                                           │
        ▼ PATH A — Vector (standard RAG)            ▼ PATH B — Graph (the differentiator)
┌───────────────────────────┐            ┌───────────────────────────────┐
│ PyMuPDF text extraction   │            │ spaCy sentence segmentation   │
│ + text cleaning           │            │ + Named Entity Recognition    │
│ (src/ingestion.py)        │            │ (entities = graph nodes)      │
└─────────────┬─────────────┘            └──────────────┬────────────────┘
              │                                         │
              ▼                                         ▼
┌───────────────────────────┐            ┌───────────────────────────────┐
│ Overlapping chunking      │            │ REBEL relation extraction     │
│ (~500 chars, 100 overlap) │            │ sentence → (subj, rel, obj)  │
│ (src/chunker.py)          │            │ (src/relation_extractor.py)   │
└─────────────┬─────────────┘            └──────────────┬────────────────┘
              │                                         │
              ▼                                         ▼
┌───────────────────────────┐            ┌───────────────────────────────┐
│ sentence-transformers     │            │ NetworkX directed graph       │
│ all-MiniLM-L6-v2 (23M)   │            │ nodes = entities              │
│ → 384-dim embeddings      │            │ edges = labeled relations     │
│ (src/vector_store.py)     │            │ + fuzzy entity deduplication  │
└─────────────┬─────────────┘            │ (src/knowledge_graph.py)      │
              │                          └──────────────┬────────────────┘
              ▼                                         │
┌───────────────────────────┐                           │
│ ChromaDB                  │                           │
│ persistent, on disk       │                           │
│ at data/chroma_db/        │                           │
└─────────────┬─────────────┘                           │
              │                                         │
              └────────────────────┬────────────────────┘
                                   │
                      ┌────────────▼────────────┐
                      │      USER QUERY         │
                      └────────────┬────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                        ▼
┌───────────────────────────┐            ┌───────────────────────────────┐
│ Vector search             │            │ spaCy entity extraction from  │
│ query → embedding →       │            │ query → fuzzy match to graph  │
│ top-k similar chunks      │            │ nodes → traverse 1-3 hops →  │
│ (src/vector_store.py)     │            │ collect subgraph evidence     │
│                           │            │ (src/graph_retriever.py)      │
└─────────────┬─────────────┘            └──────────────┬────────────────┘
              │                                         │
              └────────────────────┬────────────────────┘
                                   ▼
                      ┌─────────────────────────┐
                      │ Merge + deduplicate      │
                      │ Cross-encoder re-rank    │
                      │ ms-marco-MiniLM-L-6-v2   │
                      │ (src/hybrid_retriever.py)│
                      └────────────┬────────────┘
                                   ▼
                      ┌─────────────────────────┐
                      │ LangGraph orchestration  │
                      │ → Gemini Flash synthesis │
                      │ → cited, grounded answer │
                      │ (src/pipeline.py)        │
                      └────────────┬────────────┘
                                   ▼
                      ┌─────────────────────────┐
                      │ Streamlit UI + pyvis     │
                      │ graph visualization      │
                      │ (src/app.py)             │
                      └─────────────────────────┘
```

---

## 5. Complete tech stack

Every library here is tested and verified to work on CPU with 16GB RAM.

| Purpose | Library/Model | Size/Notes |
|---|---|---|
| PDF text extraction | `PyMuPDF` (fitz) | Fast, handles most PDF layouts reliably |
| NLP / NER / sentence splitting | `spaCy` + `en_core_web_sm` | Small model (~12MB), fast on CPU |
| Relation extraction | `transformers` + `Babelscape/rebel-large` (HuggingFace) | ~1.2GB download; runs on CPU at ~2-3 sec/sentence — slow but batchable, run overnight for large corpora |
| Text embeddings | `sentence-transformers` + `all-MiniLM-L6-v2` | 23M params, produces 384-dim vectors, very fast on CPU |
| Vector database | `ChromaDB` | Persistent (on disk at `data/chroma_db/`), zero setup, no server needed |
| Graph storage | `NetworkX` | Pure Python, in-memory graph + pickle serialization to disk. Sufficient at this scale; Neo4j is optional future work, not needed for v1 |
| Entity deduplication | `thefuzz` | Fuzzy string matching to merge entity variants ("YOLO" / "YOLOv3", "Vaswani" / "A. Vaswani") |
| Cross-encoder re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~80M params, CPU-feasible (~50ms per query-passage pair) |
| Pipeline orchestration | `LangGraph` | Multi-step agentic workflow as a graph of operations with shared state |
| LLM generation (primary) | Gemini Flash API via `google-generativeai` | Free tier, generous rate limits, no local compute |
| LLM generation (backup) | Groq API (Llama 3.x) via `groq` | Free tier, fallback if Gemini is rate-limited |
| Graph visualization | `pyvis` | Interactive HTML graph rendering — key interview demo piece |
| Web UI | `Streamlit` | Fast to build, professional-looking, good for demos |
| Environment/config | `python-dotenv` | API keys in `.env`, never hardcoded, never committed |
| Progress bars | `tqdm` | For long batch jobs (triple extraction, embedding) |
| Data validation | `Pydantic` | Structured schemas for chunks, triples, query results |

### Package versions (save as requirements.txt)

```
PyMuPDF==1.24.0
sentence-transformers==3.0.0
chromadb==0.5.0
spacy==3.7.0
transformers==4.42.0
torch==2.3.0
networkx==3.3
pyvis==0.3.2
langgraph==0.2.0
langchain-google-genai==1.0.0
google-generativeai==0.7.0
streamlit==1.36.0
python-dotenv==1.0.1
thefuzz==0.22.1
tqdm==4.66.0
pydantic==2.8.0
```

Install torch CPU-only: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
Download spaCy model: `python -m spacy download en_core_web_sm`

---

## 6. Key concepts glossary

The builder is new to all of these. Explain each one the FIRST time it
appears in code, using plain language and concrete examples.

- **Embedding:** Converting text into a fixed-length list of numbers (a vector,
  e.g. 384 numbers) such that texts with similar MEANING end up numerically
  close together. "Dog" and "puppy" → close coordinates. "Dog" and "algebra"
  → far apart. The model learned meaning, not word matching — "YOLO is a
  real-time object detection system" and "You Only Look Once detects objects
  quickly" have ~0.85 similarity despite sharing almost no words.

- **Vector database (ChromaDB):** A database optimized to store embeddings and
  quickly find the nearest ones to a query embedding. When you search, your
  question becomes 384 numbers, and ChromaDB finds the chunks whose 384
  numbers are closest. That's vector search.

- **Chunking:** Splitting long documents into smaller pieces (~500 characters)
  before embedding, because retrieval needs to be precise. Overlap (e.g. 100
  chars) prevents cutting sentences at boundaries. Smaller chunks = more
  precise matches, but too small = lose context.

- **Named Entity Recognition (NER):** An NLP model reads a sentence and
  identifies all the "things" in it — people (PERSON), organizations (ORG),
  concepts. "Vaswani et al. at Google proposed the Transformer" →
  Vaswani (PERSON), Google (ORG), Transformer (CONCEPT). These entities
  become the NODES of the knowledge graph.

- **Relation extraction (REBEL):** A transformer model that reads a sentence
  and outputs structured `(subject, relation, object)` triples. "DETR uses
  transformers for object detection and outperforms Faster R-CNN on COCO" →
  `(DETR, uses, transformers for object detection)` AND
  `(DETR, outperforms, Faster R-CNN)`. NER finds the things. REBEL finds
  the relationships between things. These triples become the EDGES of the
  knowledge graph.

- **Knowledge graph:** A network of entities (nodes) connected by labeled
  relationships (edges). Like a road map: every entity is a city, every
  relation is a named road between cities. You can traverse it — follow
  roads from city to city to reach destinations that aren't directly
  connected.

- **Multi-hop reasoning / traversal:** Answering a question that requires
  following more than one relationship in sequence.
  - 1 hop: DETR → outperforms → Faster R-CNN (direct connection)
  - 2 hops: DETR → outperforms → Faster R-CNN → proposed by → Ren et al.
  - 3 hops: adds another connection step
  This is what vector search STRUCTURALLY cannot do and graph traversal CAN.

- **Cross-encoder re-ranking:** After fast-but-approximate initial retrieval
  (embeddings encode query and passage SEPARATELY), a slower-but-more-accurate
  model reads query AND candidate passage TOGETHER and directly scores
  relevance. Used as a second pass to re-order the top results for precision.

- **LangGraph:** A framework for building multi-step AI workflows as a graph
  of operations. Each node is a processing step, edges define execution order,
  and a shared state dict passes between steps accumulating results. Better
  than calling functions in sequence because it handles parallel execution
  (vector + graph search simultaneously) and makes the pipeline explicit.

- **Triple:** A structured fact in the form `(subject, relation, object)` with
  metadata (source file, page, original sentence). The atomic unit of the
  knowledge graph.

---

## 7. Project folder structure

```
kg-rag/
├── data/
│   ├── raw/                      # User drops PDF papers here
│   ├── processed/
│   │   ├── triples.json          # Extracted (subj, rel, obj) triples
│   │   ├── knowledge_graph.gpickle  # Serialized NetworkX graph
│   │   └── graph_viz.html        # Interactive pyvis visualization
│   └── chroma_db/                # ChromaDB persistent vector store (auto-created)
├── notebooks/
│   ├── 01_embeddings_demo.py     # Phase 2: understand embeddings hands-on
│   ├── 02_ner_demo.py            # Phase 3: understand NER hands-on
│   └── 03_langgraph_intro.py     # Phase 6: understand LangGraph hands-on
├── src/
│   ├── ingestion.py              # Phase 1: PDF extraction + text cleaning
│   ├── chunker.py                # Phase 1: overlapping text chunking
│   ├── vector_store.py           # Phase 2: ChromaDB embedding + search
│   ├── basic_rag.py              # Phase 2: generic RAG baseline (for comparison)
│   ├── relation_extractor.py     # Phase 3: REBEL triple extraction per sentence
│   ├── knowledge_extractor.py    # Phase 3: batch pipeline over all docs → triples.json
│   ├── knowledge_graph.py        # Phase 4: NetworkX graph + traversal functions
│   ├── graph_viz.py              # Phase 4: pyvis interactive visualization
│   ├── graph_retriever.py        # Phase 5: NL query → entity extraction → graph traversal
│   ├── hybrid_retriever.py       # Phase 5: vector + graph merge + cross-encoder re-rank
│   ├── pipeline.py               # Phase 6: LangGraph orchestrated full pipeline
│   ├── evaluate.py               # Phase 7: 20-question evaluation harness
│   └── app.py                    # Phase 7: Streamlit UI with graph visualization
├── tests/
├── .env                          # API keys (NEVER committed — in .gitignore)
├── .gitignore                    # venv/, data/raw/, __pycache__/, .env
├── requirements.txt
├── PROJECT_CONTEXT.md            # This file
└── README.md                     # Phase 7: interview-ready documentation
```

---

## 8. Build phases — detailed step-by-step with exact prompts

### Concepts learned per phase

| Phase | Week | New concepts |
|-------|------|-------------|
| 1 | 1 | Text extraction from PDFs, text cleaning, chunking strategies, why overlap matters |
| 2 | 2 | What embeddings are, vector databases, similarity search, basic RAG end-to-end, seeing its limitations |
| 3 | 3–4 | Named Entity Recognition, relation extraction with REBEL, structured vs unstructured knowledge |
| 4 | 4 | Graph databases, nodes/edges, directed graphs, traversal, hops, graph visualization |
| 5 | 5 | Hybrid retrieval, merging two result types, cross-encoder re-ranking, why re-ranking helps |
| 6 | 6 | LangGraph orchestration, state machines, prompt engineering for cited answers |
| 7 | 7–8 | Streamlit UI, RAG evaluation methodology, documentation, interview preparation |

---

### PHASE 1: Document Ingestion & Chunking (Week 1)

**Goal:** Take raw PDFs → extract clean text → split into searchable chunks.

**Step 1.1 — PDF text extraction**

Create `src/ingestion.py`:
- Use PyMuPDF to open each PDF in `data/raw/`, extract text page by page.
- Return a dict per PDF: `{filename, full_text, page_count, pages: [per_page_text]}`.
- Handle errors gracefully — skip corrupted PDFs, log warnings.
- Main block: process `data/raw/`, print stats (file count, total pages, total chars).

Test: drop 5-10 CV research papers into `data/raw/` and run. Inspect output —
messy headers, footers, references are expected at this stage.

After building, ask: "Show me the output of the first 500 characters from each
PDF. Are there any issues with the extraction?"

**Step 1.2 — Text cleaning**

Add `clean_text()` function to `src/ingestion.py`:
- Remove repeating headers/footers (detect via cross-page repetition).
- Remove References/Bibliography section entirely.
- Normalize whitespace, fix broken hyphenation (words split across lines).
- Remove page numbers.
- Show before/after comparison on real papers.

Why this matters: cleaning matters more than any fancy model you'll add later.
Bad input text = bad everything downstream. Garbage in, garbage out.

**Step 1.3 — Chunking**

Create `src/chunker.py`:
- Fixed-size chunker: `chunk_size=500` chars, `overlap=100` chars.
- Each chunk is a dict: `{chunk_id, text, source_file, start_char, end_char, chunk_index}`.
- Also build a smarter "semantic chunker" variant that splits on paragraph/section
  boundaries instead of fixed character counts.
- Main block: chunk all processed PDFs, print total chunks per doc, average
  chunk length, a sample chunk.
- Explain in comments WHY overlapping chunks are used: overlap ensures we don't
  cut a sentence in half and lose meaning at the boundary. Without overlap,
  a key sentence right at a boundary gets split between two chunks and neither
  chunk captures the full meaning.

**Key concept — WHY chunks?** Later, when someone asks a question, we won't
search the full paper. We'll search these small chunks to find the 3–5 most
relevant ones. Smaller chunks = more precise matches. But too small = lose
context. 500 chars is a good starting point.

**Phase 1 checkpoint:**
- [ ] PDFs extracting cleanly
- [ ] Text cleaning working (headers, footers, references removed)
- [ ] Chunks generating with metadata
- [ ] Understanding of WHY we chunk text

Commit: `git add -A && git commit -m "phase 1: document ingestion and chunking"`

---

### PHASE 2: Embeddings & Vector Search — Basic RAG (Week 2)

**Goal:** Build the GENERIC version first (the thing everyone has) so we
have a baseline to beat and the builder understands embeddings deeply.

**Step 2.1 — Understanding embeddings (learning exercise)**

Create `notebooks/01_embeddings_demo.py`:
- Load `all-MiniLM-L6-v2` model from sentence-transformers.
- Embed these 6 specific sentences:
  1. "YOLO is a real-time object detection system"
  2. "You Only Look Once detects objects quickly"
  3. "The weather is sunny today"
  4. "Transformers use self-attention mechanisms"
  5. "Self-attention allows models to weigh input importance"
  6. "I like pizza"
- Print pairwise similarity matrix.
- Show that sentences 1 and 2 have high similarity (~0.85) even though they
  share almost no words. Sentences 3 and 6 will be near 0. The model learned
  MEANING, not just word matching.
- Add detailed comments explaining what each number means.

Key explanation to give: each sentence becomes a list of 384 numbers. These
numbers are coordinates in a 384-dimensional space. Sentences with similar
meanings end up at nearby coordinates. "Dog" and "puppy" → close. "Dog" and
"algebra" → far. That's the entire idea behind embeddings.

**Step 2.2 — ChromaDB vector store**

Create `src/vector_store.py`:
- Create a persistent ChromaDB collection at `data/chroma_db/`.
- `add_chunks(chunks)` function: takes chunks from Phase 1, embeds them using
  all-MiniLM-L6-v2, stores in ChromaDB with metadata (source_file, chunk_index).
- `search(query, top_k=5)` function: takes a query string, embeds it, returns
  top-k most similar chunks with their similarity scores.
- Main block: ingest all chunks, test with 3 sample queries, print results.
- Add comments explaining what ChromaDB is doing under the hood.

Test queries to try:
- "object detection speed" → should find YOLO-related chunks
- "attention mechanism" → should find transformer-related chunks
- "banana recipe" → should find nothing relevant (low scores)

Key explanation: ChromaDB is just a database optimized for storing and searching
embeddings. When you search, it converts your question into 384 numbers, then
finds the chunks whose 384 numbers are closest. That's vector search.

**Step 2.3 — Basic RAG (the generic version everyone builds)**

Create `src/basic_rag.py`:
- Take user question → search ChromaDB for top-5 chunks → construct prompt:
  "Answer this question using ONLY the context provided. If the context doesn't
  contain the answer, say so. Cite which source file each fact comes from."
- Send prompt + retrieved chunks to Gemini Flash API (use `google-generativeai`
  package with API key from `.env`).
- Print: the question, retrieved chunks with scores, and the final answer.
- Make it interactive — loop asking for questions until user types 'quit'.
- Use `python-dotenv` to load API keys.

**CRITICAL TEST — THE MOTIVATION FOR THIS WHOLE PROJECT:**
After building basic RAG, try this multi-hop question:
> "Which authors of object detection papers also contributed to segmentation research?"

It WILL fail or hallucinate. It can only find text chunks that are individually
similar to the question. It cannot CONNECT information across papers. The builder
must experience this failure firsthand to understand why we need the knowledge
graph. This is where most projects stop. We keep going.

**Phase 2 checkpoint:**
- [ ] Understand what embeddings are and why they work
- [ ] ChromaDB storing and searching document chunks
- [ ] Basic RAG working end-to-end with Gemini Flash
- [ ] Seen its limitations firsthand on multi-hop questions

Commit: `git add -A && git commit -m "phase 2: embeddings, vector store, basic RAG"`

---

### PHASE 3: Named Entity Recognition & Relation Extraction (Weeks 3–4)

**Goal:** Extract structured knowledge from raw text. This is where KG-RAG
diverges from every generic project.

**Step 3.1 — NER with spaCy (learning exercise)**

Create `notebooks/02_ner_demo.py`:
- Install spaCy and download `en_core_web_sm`.
- Take 5 real sentences from the extracted papers.
- Run spaCy NER on each sentence.
- Print every entity found with its label (PERSON, ORG, WORK_OF_ART, etc.).
- Also print a formatted visualization.
- Use real sentences from the CV papers if available.
- Explain what NER is doing and why we need it.

What's happening: spaCy reads a sentence and highlights all the "things" —
people's names, organizations, technical terms. "Vaswani et al. at Google
proposed the Transformer" → entities: Vaswani (PERSON), Google (ORG),
Transformer (PRODUCT/CONCEPT). These entities become the NODES of the
knowledge graph.

**Step 3.2 — Relation extraction with REBEL**

Create `src/relation_extractor.py`:
- Install `transformers` (if not already).
- Load the REBEL model (`Babelscape/rebel-large`) from HuggingFace.
- Function that takes a sentence → returns list of `(subject, relation, object)` triples.
- Test on 10 real sentences from the papers.
- Print each triple clearly: `Subject → Relation → Object`.
- Handle sentences where REBEL finds nothing gracefully (return empty list).
- IMPORTANT NOTE: REBEL is ~1.2GB download. It runs on CPU but is slow
  (~2-3 sec per sentence). That's fine for batch processing.
- Add comments explaining: what is a "triple"? what does REBEL actually do
  internally? Why is this different from NER?

What's happening: REBEL reads "DETR uses transformers for object detection and
outperforms Faster R-CNN on COCO" and outputs TWO triples:
- (DETR) → uses → (transformers for object detection)
- (DETR) → outperforms → (Faster R-CNN)

These triples become the EDGES of the knowledge graph.
NER finds the things. REBEL finds the relationships between things.
Together they give you structured knowledge from raw text.

**Step 3.3 — Batch extraction pipeline**

Create `src/knowledge_extractor.py`:
- Take all cleaned documents from Phase 1.
- Split each into sentences using spaCy sentence segmentation.
- Run REBEL on each sentence to extract triples.
- For each triple, store metadata: `{subject, relation, object, source_file,
  page_number, sentence_text}` (for attribution later).
- Save all triples to `data/processed/triples.json`.
- Print stats: total triples extracted, most common entities, most common
  relation types.
- Add a progress bar using tqdm (since this will take time).
- Optimization: skip very short sentences (<10 words) and very long ones
  (>100 words) — REBEL struggles with both. Process in batches.
- On 300 papers, this may take several hours on CPU — it is designed to be
  run overnight as a batch job. The output is a big JSON file of
  `(subject, relation, object, source)` entries.

**Key concept — WHY is this hard?** Language is ambiguous. "The model was
beaten by the new approach" — REBEL needs to figure out that "the model" is
the subject and "the new approach" is the object, and the relation is
"beaten by." This is genuinely hard NLP, not string matching.

**Phase 3 checkpoint:**
- [ ] spaCy NER finding entities in papers
- [ ] REBEL extracting (subject, relation, object) triples
- [ ] Full batch pipeline producing `triples.json`
- [ ] Understanding the difference between text search and structured knowledge

Commit: `git add -A && git commit -m "phase 3: NER and relation extraction pipeline"`

---

### PHASE 4: Knowledge Graph Construction (Week 4)

**Goal:** Build the actual graph from extracted triples and learn to traverse it.

**Step 4.1 — Building the graph**

Create `src/knowledge_graph.py`:
- Load `triples.json` from Phase 3.
- Build a NetworkX **directed** graph:
  - Nodes = unique entities (subjects and objects from triples)
  - Edges = relations, with attributes: `relation_type`, `source_file`, `sentence_text`
- Entity deduplication: merge similar names using `thefuzz` fuzzy matching:
  - "YOLO" and "YOLOv3" should be connected
  - "Vaswani" and "A. Vaswani" should merge
- Print graph stats: node count, edge count, highest-degree nodes (most connected),
  most common relation types.
- Save graph to `data/processed/knowledge_graph.gpickle`.
- Explain: what "directed graph" means and why direction matters — `(A outperforms B)`
  is NOT the same as `(B outperforms A)`.

What's happening: you're building the road map. Every entity is a city. Every
relation is a road between cities, and the road has a name (like "outperforms"
or "cites" or "proposed by").

**Step 4.2 — Graph traversal (the multi-hop magic)**

Add these functions to `src/knowledge_graph.py`:
- `get_neighbors(entity, hops=1)` — find all nodes within N hops of a given
  entity. Return the full path taken.
- `find_path(entity_a, entity_b)` — find the shortest path between two entities,
  showing every hop and relation.
- `get_subgraph(entity, hops=2)` — extract a small subgraph around an entity
  showing all connections within 2 hops.
- Test each function with real entities from the built graph.
- Print results in readable format:
  `DETR --[outperforms]--> Faster R-CNN --[proposed by]--> Ren et al.`
- Explain what "hops" means with a concrete example.

What's happening: THIS is what makes KG-RAG possible. When someone asks a
multi-hop question, you extract the entities from their question, then WALK
the graph to find connected information.
- 1 hop: direct connection (DETR → outperforms → Faster R-CNN)
- 2 hops: one step removed (DETR → outperforms → Faster R-CNN → proposed by → Ren)
- 3 hops: two steps removed (and so on)
The graph gives you answers that no amount of text searching can find.

**Step 4.3 — Graph visualization**

Create `src/graph_viz.py`:
- Use pyvis to render an interactive HTML visualization.
- Nodes colored by type (PERSON=blue, PAPER=green, CONCEPT=orange, etc.).
- Edge labels show the relation type.
- Node size reflects degree (how connected it is).
- Users can zoom, pan, click nodes to see details.
- Generate for subgraph around a major entity (e.g., "Transformer" or whatever
  exists in the graph) with 2 hops.
- Save to `data/processed/graph_viz.html`.
- This is a KEY interview demo piece — makes the project visually impressive.

**Phase 4 checkpoint:**
- [ ] Knowledge graph built from extracted triples
- [ ] Can traverse the graph with 1, 2, 3 hops
- [ ] Interactive visualization working
- [ ] Understand WHY graph traversal answers questions text search can't

Commit: `git add -A && git commit -m "phase 4: knowledge graph construction and traversal"`

---

### PHASE 5: Hybrid Retrieval — Combining Both Paths (Week 5)

**Goal:** Make the two retrieval systems work together at query time.

**Step 5.1 — Graph-based retrieval from natural language**

Create `src/graph_retriever.py`:
- Take a natural language query.
- Extract entities from the query using spaCy NER.
- For each entity found, search the knowledge graph:
  - Exact match first, then fuzzy match as fallback
  - Get all nodes within 2 hops
  - Collect the source sentences for every edge traversed
- Return a list of "graph evidence" items, each with:
  `{triple (subject, relation, object), source_file, sentence, hop_distance}`.
- Test with: "Which models outperform YOLO?" and print results.

**Step 5.2 — Hybrid retrieval (the combination)**

Create `src/hybrid_retriever.py`:
- Take a query → run BOTH retrievals simultaneously:
  - Vector search (from Phase 2, `src/vector_store.py`) → top 10 chunks
  - Graph search (from Step 5.1, `src/graph_retriever.py`) → all graph evidence
- Combine into a single ranked list. Each result has:
  `{text, source, retrieval_method ('vector' or 'graph'), relevance_score}`.
- Remove near-duplicates (same source file + high text overlap).
- Return top-K combined results (default K=8).
- Test with BOTH:
  - A simple question (where vector search should win)
  - A multi-hop question (where graph search should win)
- Print which method contributed what to each answer.

**Step 5.3 — Cross-encoder re-ranking**

Update `src/hybrid_retriever.py` to add re-ranking:
- Load cross-encoder model: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- After combining vector + graph results, re-rank ALL of them using the
  cross-encoder.
- The cross-encoder takes (query, passage) pairs and directly scores how
  relevant the passage actually is to the query — more accurate than the
  initial retrieval scores.
- Return results sorted by cross-encoder score.
- Print: original rank vs re-ranked position for each result to show improvement.
- Explain in comments: why do we need re-ranking? How is a cross-encoder
  different from the embedding model used earlier?

Key concept — WHY re-rank? The embedding model (Phase 2) is fast but
approximate — it encodes query and passage SEPARATELY. The cross-encoder is
slow but precise — it reads query AND passage TOGETHER and directly judges
relevance. We use embeddings for the fast first pass, then the cross-encoder
to re-order the top results for precision.

**Phase 5 checkpoint:**
- [ ] Graph-based retrieval working from natural language queries
- [ ] Both retrieval methods running and combining results
- [ ] Cross-encoder re-ranking improving result quality
- [ ] Can see which method (vector vs graph) contributes to each answer

Commit: `git add -A && git commit -m "phase 5: hybrid retrieval with cross-encoder re-ranking"`

---

### PHASE 6: LLM Synthesis with LangGraph (Week 6)

**Goal:** Orchestrate the full pipeline and generate cited, grounded answers.

**Step 6.1 — LangGraph basics (learning exercise)**

Create `notebooks/03_langgraph_intro.py`:
- Install `langgraph`.
- Create a simple 3-step graph:
  - Step 1: take user input
  - Step 2: process it (just uppercase for now)
  - Step 3: format output
- Show the graph executing step by step with state printed at each node.
- Explain in comments: what is LangGraph? Why not just call functions in
  sequence? What does the "state" concept mean?

What's happening: LangGraph is a framework for building multi-step AI workflows
as a graph of operations. Each node is a step, edges define what runs next.
The state is a dict that passes between steps, accumulating results. It's better
than linear function calls because it handles parallel execution (we need vector
+ graph search running simultaneously) and makes the pipeline explicit and debuggable.

**Step 6.2 — The KG-RAG pipeline as a LangGraph**

Create `src/pipeline.py`:
- State dict: `{query, vector_results, graph_results, merged_results,
  reranked_results, answer, sources}`
- Nodes (processing steps):
  1. `retrieve_vector` — run vector search (from Phase 2)
  2. `retrieve_graph` — run graph search (from Phase 5.1)
  3. `merge_and_rerank` — combine + cross-encoder re-rank (from Phase 5.3)
  4. `synthesize` — send top results to Gemini Flash with grounded prompt
  5. `format_output` — structure the final response with citations
- Edges: query → [retrieve_vector, retrieve_graph] running in PARALLEL
  → merge_and_rerank → synthesize → format_output.
- Test with 3 questions of increasing complexity.

**Step 6.3 — Prompt engineering for synthesis**

Update the `synthesize` step in `src/pipeline.py` with a well-engineered
prompt for Gemini Flash:
- Clearly separate CONTEXT (retrieved chunks and triples) from the QUESTION.
- Instruct the model to:
  - Answer ONLY from the provided context
  - Cite every claim as `[Source: filename.pdf, p.X]`
  - If the answer requires connecting information from multiple sources,
    explicitly show the reasoning chain
  - If the context is insufficient, say what's missing
  - Distinguish between facts stated in papers vs inferred connections
    from the knowledge graph
- Include the retrieval method (vector vs graph) in the context so the LLM
  knows which evidence came from text similarity vs structural relationships.
- Test with the 3 questions and compare answer quality before and after
  the prompt improvements.

**Phase 6 checkpoint:**
- [ ] LangGraph pipeline orchestrating the full flow
- [ ] Gemini Flash generating cited, grounded answers
- [ ] Multi-hop questions getting better answers than Phase 2's basic RAG
- [ ] Can trace exactly which retrieval method contributed to each answer

Commit: `git add -A && git commit -m "phase 6: LangGraph pipeline with LLM synthesis"`

---

### PHASE 7: UI, Evaluation & Polish (Weeks 7–8)

**Goal:** Build the demo, prove it works with numbers, document everything.

**Step 7.1 — Streamlit UI**

Create `src/app.py` — a Streamlit app with:
- **Sidebar:**
  - Upload PDFs section (batch upload)
  - "Process documents" button that runs full ingestion + graph building
  - Progress bars during processing
  - Stats display: total docs, chunks, entities, triples
- **Main area:**
  - Search box for questions
  - Answer displayed with cited sources highlighted
  - Below the answer: expandable sections showing:
    - (a) Retrieved vector chunks with similarity scores
    - (b) Retrieved graph evidence with hop counts
    - (c) Re-ranked combined results
  - Toggle to compare "Basic RAG" vs "KG-RAG" answers side by side
- **Separate tab:** Interactive knowledge graph visualization (pyvis embedded)
  - Filter by entity type, relation type
  - Click a node to see all its connections
  - Search for specific entities
- Make it visually clean and professional — this is what interviewers see.

**Step 7.2 — Evaluation harness**

Create `src/evaluate.py`:
- Define 20 test questions in 3 categories (using actual CV papers from
  the builder's corpus so he can judge answer quality expertly):
  - Simple (single-doc answer): 7 questions
  - Medium (cross-doc, 1-2 hops): 7 questions
  - Hard (multi-hop, 3+ hops): 6 questions
- For each question, manually provide the expected answer.
- Run BOTH Basic RAG (Phase 2) and KG-RAG (Phase 6) on all 20.
- Use Gemini Flash as a judge: given (question, expected_answer, system_answer),
  score 1-5 on: correctness, completeness, citation accuracy.
- Print a comparison table and save to `data/eval_results.json`.
- This gives HARD NUMBERS to show in interviews:
  "KG-RAG scored 4.2/5 on multi-hop questions vs 1.8/5 for basic RAG"

**Step 7.3 — README and documentation**

Create a comprehensive `README.md` that includes:
1. Project title and one-line description
2. Architecture diagram (ASCII or Mermaid)
3. What problem this solves (with concrete multi-hop example)
4. How it works (simplified 5-step explanation)
5. Results: evaluation scores comparison table (basic RAG vs KG-RAG)
6. Tech stack with version numbers
7. Setup instructions (step by step)
8. Screenshots/GIFs of the Streamlit UI
9. Limitations and future work
10. Builder's name and links

This README is the first thing recruiters/interviewers see on GitHub. It
should make someone want to try it within 30 seconds of opening the repo.

**Phase 7 checkpoint:**
- [ ] Streamlit UI fully functional and professional-looking
- [ ] Evaluation results showing KG-RAG > Basic RAG on multi-hop questions
- [ ] README that makes someone want to try it in 30 seconds
- [ ] Project deployed on GitHub with clean commit history

Commit: `git add -A && git commit -m "phase 7: UI, evaluation, documentation"`

---

## 9. Success criteria

Build 20 test questions using the builder's actual CV paper corpus (object
detection, transformers, segmentation papers) across 3 difficulty tiers.
Run both basic RAG and KG-RAG on all 20, score with LLM-as-judge.

**Expected outcomes:**
- Simple questions: both systems score similarly (~4/5)
- Medium questions (1-2 hops): KG-RAG noticeably better (~3.8 vs ~2.5)
- Hard questions (3+ hops): KG-RAG dramatically better (~4.2 vs ~1.8)

This comparison table is the single most important artifact for interviews —
quantitative proof the extra complexity is justified, not academic overengineering.

---

## 10. How Claude Code should behave during this project

1. **Explain every new concept** the first time it appears (embeddings, NER,
   triples, graph traversal, re-ranking, LangGraph state). Assume the builder
   has NOT studied RAG or knowledge graphs formally. Use plain language and
   concrete examples from CV/ML papers he already knows.

2. **Build incrementally.** One module at a time. Test each piece with sample
   input → expected output before moving to the next step. Prefer small,
   testable increments over large multi-file generations.

3. **After writing code, explain:** what it does, why this approach was chosen
   over alternatives, and how it connects to the overall KG-RAG architecture —
   all in plain language, before moving on.

4. **Flag CPU-slow operations** (especially REBEL batch extraction over hundreds
   of docs) and suggest running as overnight batch jobs rather than blocking
   interactive development.

5. **API keys in `.env` always.** Use `python-dotenv` for loading. Never
   hardcode keys. Never include `.env` in git.

6. **When the builder asks "why"**, answer conceptually first (the intuition),
   then point to the specific code (the implementation).

7. **When something breaks**, diagnose step by step — show current state,
   explain what's wrong, fix it, verify the fix.

8. **The learning loop for every step:** write code → builder reads every line
   → explain anything unclear → run and see output → if broken, fix step by
   step → before moving on, summarize what was built and how it connects to
   the overall architecture.

---

## 11. What NOT to build (scope boundaries)

- **No local LLM fine-tuning** — not needed, not hardware-feasible, adds no value.
- **No GPU inference attempts** — there is no GPU. Everything is CPU or API.
- **Don't over-engineer the graph database** — NetworkX with pickle serialization
  is sufficient at this scale. Neo4j is optional future work for the README's
  "future work" section, not a v1 requirement.
- **No multi-user auth, cloud deployment, or production infra** — this is a
  portfolio/research-grade project, not a SaaS product.
- **No frontend framework beyond Streamlit** — keep it simple. Demo quality
  comes from the SYSTEM (graph viz, evaluation results, side-by-side comparison),
  not CSS polish.

---

## 12. One-time setup commands (run before starting Phase 1)

```bash
# Create project folder and structure
mkdir kg-rag && cd kg-rag
mkdir -p data/raw data/processed src tests notebooks

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Initialize git
git init
echo -e "venv/\ndata/raw/\n__pycache__/\n.env\ndata/chroma_db/\ndata/processed/" > .gitignore

# Create .env for API keys
echo "GEMINI_API_KEY=your_key_here" > .env
echo "GROQ_API_KEY=your_key_here" >> .env

# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Add 5-10 CV/ML research papers to data/raw/ for testing
# Good candidates the builder knows well:
# - Attention Is All You Need (Vaswani et al.)
# - YOLOv3 (Redmon & Farhadi)
# - DETR (Carion et al.)
# - Swin Transformer (Liu et al.)
# - Faster R-CNN (Ren et al.)
# - ResNet (He et al.)
# - Vision Transformer / ViT (Dosovitskiy et al.)
```
