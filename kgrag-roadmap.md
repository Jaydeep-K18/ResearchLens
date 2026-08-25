# KG-RAG Project — Complete Build Roadmap

> **Your hardware:** Ryzen 7 5700U · 16GB RAM · No GPU · Gemini Flash API + Groq API
> **Total time:** 6–8 weeks · ~2 hours/day
> **Tool:** Claude Code for every step

---

## Before you start — one-time setup

Open your terminal and run these. This is your project foundation.

```bash
# 1. Create project folder
mkdir kg-rag && cd kg-rag

# 2. Create a virtual environment (keeps your system clean)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Create folder structure
mkdir -p data/raw data/processed src tests notebooks

# 4. Initialize git
git init
echo "venv/\ndata/raw/\n__pycache__/\n.env" > .gitignore
```

Create a `.env` file for your API keys:
```
GEMINI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

---

## Concepts you'll learn (in order)

You don't need to study these beforehand. Each phase teaches them:

| Phase | New concepts |
|-------|-------------|
| 1 | Text extraction, chunking strategies, what embeddings are |
| 2 | Vector databases, similarity search, basic RAG |
| 3 | Named Entity Recognition (NER), relation extraction |
| 4 | Graph databases, nodes, edges, traversal |
| 5 | Hybrid retrieval, cross-encoder re-ranking |
| 6 | LangGraph orchestration, prompt engineering |
| 7 | UI, visualization, demo preparation |

---

## PHASE 1: Document Ingestion & Chunking (Week 1)

### What you'll learn
- How to extract clean text from PDFs programmatically
- What "chunking" means and why chunk size matters
- How to handle messy real-world documents

### What to tell Claude Code

Start Claude Code in your project folder, then give it these prompts one by one. After each step, READ the code it writes. Ask it to explain anything you don't understand.

---

**Step 1.1 — PDF text extraction**

```
Prompt for Claude Code:
"Create src/ingestion.py — a module that takes a folder of PDFs
and extracts clean text from each one using PyMuPDF. For each PDF,
return a dict with: filename, full_text, page_count, and a list of
per-page text. Handle errors gracefully (skip corrupted PDFs).
Add a main block that processes data/raw/ and prints stats.
Install any needed packages."
```

**What's happening:** PyMuPDF opens each PDF page by page and pulls
out the text. This is the rawest step — garbage in, garbage out.

**Test it:** Drop 5–10 research papers into `data/raw/` and run it.
Look at the output. You'll notice some text is messy — headers,
footers, references sections, weird formatting. That's normal.

**Ask Claude Code:** "Show me the output of the first 500 characters
from each PDF. Are there any issues with the extraction?"

---

**Step 1.2 — Text cleaning**

```
Prompt for Claude Code:
"Add a clean_text() function to src/ingestion.py that:
- Removes headers/footers that repeat on every page
- Removes the References/Bibliography section
- Normalizes whitespace and fixes broken hyphenation
- Removes page numbers
Test it on our extracted PDFs and show me before/after comparison."
```

**What's happening:** Real PDFs are messy. Cleaning matters more
than any fancy model you'll add later. Bad text = bad everything.

---

**Step 1.3 — Chunking**

```
Prompt for Claude Code:
"Create src/chunker.py with a function that splits cleaned text
into overlapping chunks. Parameters: chunk_size (default 500 chars),
overlap (default 100 chars). Each chunk should be a dict with:
chunk_id, text, source_file, start_char, end_char, chunk_index.

Also add a smarter 'semantic chunker' that tries to split on
paragraph/section boundaries instead of fixed character counts.

Add a main block that chunks our processed PDFs and prints:
- Total chunks per document
- Average chunk length
- A sample chunk

Explain in comments WHY we use overlapping chunks."
```

**What's happening:** An LLM can't read a whole paper at once during
retrieval. We split it into small pieces. Overlap ensures we don't
cut a sentence in half and lose meaning at the boundary.

**Key concept — WHY chunks?** Later, when someone asks a question,
we won't search the full paper. We'll search these small chunks
to find the 3–5 most relevant ones. Smaller chunks = more precise
matches. But too small = lose context. 500 chars is a good start.

---

### Phase 1 checkpoint

At this point you should have:
- [ ] PDFs extracting cleanly
- [ ] Text cleaning working
- [ ] Chunks generating with metadata
- [ ] Understanding of WHY we chunk text

**Commit:** `git add -A && git commit -m "phase 1: document ingestion and chunking"`

---

## PHASE 2: Embeddings & Vector Search (Week 2)

### What you'll learn
- What embeddings actually are (the core concept behind all modern AI search)
- How vector databases work
- How to build basic RAG (the generic version first)

---

**Step 2.1 — Understanding embeddings**

```
Prompt for Claude Code:
"Create notebooks/01_embeddings_demo.py — a script that:
1. Installs sentence-transformers
2. Loads the all-MiniLM-L6-v2 model (small, runs on CPU)
3. Takes these 6 sentences and embeds them:
   - 'YOLO is a real-time object detection system'
   - 'You Only Look Once detects objects quickly'
   - 'The weather is sunny today'
   - 'Transformers use self-attention mechanisms'
   - 'Self-attention allows models to weigh input importance'
   - 'I like pizza'
4. Prints the similarity score between every pair
5. Shows which sentences are most/least similar

Add detailed comments explaining what each number means."
```

**What's happening:** This is the 'aha' moment. You'll see that
sentences 1 and 2 have high similarity (~0.85) even though they
share almost no words. Sentence 3 and 6 will be near 0. The model
learned MEANING, not just word matching.

**Key concept — WHAT is an embedding?** Each sentence becomes a
list of 384 numbers. These numbers are coordinates in a 384-
dimensional space. Sentences with similar meanings end up at nearby
coordinates. "Dog" and "puppy" → close. "Dog" and "algebra" → far.
That's it. That's the whole idea.

---

**Step 2.2 — ChromaDB vector store**

```
Prompt for Claude Code:
"Create src/vector_store.py that:
1. Installs chromadb
2. Creates a persistent ChromaDB collection at data/chroma_db/
3. Has an add_chunks() function that takes our chunks from Phase 1,
   embeds them using all-MiniLM-L6-v2, and stores them in ChromaDB
   with metadata (source_file, chunk_index)
4. Has a search() function that takes a query string, embeds it,
   and returns the top-k most similar chunks with their similarity
   scores
5. Main block: ingest all our chunks, then test with 3 sample
   queries and print results

Add comments explaining what ChromaDB is doing under the hood."
```

**What's happening:** ChromaDB is just a database optimized for
storing and searching embeddings. When you search, it converts your
question into 384 numbers, then finds the chunks whose 384 numbers
are closest. That's vector search.

**Test it yourself:** Try queries like:
- "object detection speed" → should find YOLO-related chunks
- "attention mechanism" → should find transformer-related chunks
- "banana recipe" → should find nothing relevant (low scores)

---

**Step 2.3 — Basic RAG (the generic version)**

```
Prompt for Claude Code:
"Create src/basic_rag.py that:
1. Takes a user question
2. Searches ChromaDB for top-5 relevant chunks (from step 2.2)
3. Constructs a prompt: 'Answer this question using ONLY the
   context provided. If the context doesn't contain the answer,
   say so. Cite which source file each fact comes from.'
4. Sends the prompt + retrieved chunks to Gemini Flash API
   (use google-generativeai package with our .env key)
5. Prints: the question, retrieved chunks with scores, and the
   final answer
6. Make it interactive — loop asking for questions until 'quit'

Use python-dotenv to load API keys."
```

**What's happening:** THIS is basic RAG. This is what every
tutorial teaches. Search → retrieve chunks → feed to LLM → answer.
You now have the generic PDF Q&A app that exists in 10,000 repos.

**Try it and notice the limitation:** Ask a multi-hop question like
"Which authors of object detection papers also contributed to
segmentation research?" — it will fail or hallucinate. It can only
find text chunks that are individually similar to your question.
It cannot CONNECT information across papers.

**This is where most people stop. We keep going.**

---

### Phase 2 checkpoint

- [ ] Understand what embeddings are and why they work
- [ ] ChromaDB storing and searching your document chunks
- [ ] Basic RAG working end-to-end
- [ ] Seen its limitations firsthand on multi-hop questions

**Commit:** `git add -A && git commit -m "phase 2: embeddings, vector store, basic RAG"`

---

## PHASE 3: Named Entity Recognition & Relation Extraction (Weeks 3–4)

### What you'll learn
- How NLP models identify "things" in text (NER)
- How REBEL extracts structured facts from sentences
- The difference between unstructured text and structured knowledge

### This is where KG-RAG diverges from every generic project.

---

**Step 3.1 — Named Entity Recognition with spaCy**

```
Prompt for Claude Code:
"Create notebooks/02_ner_demo.py that:
1. Installs spacy and downloads en_core_web_sm
2. Takes 5 sentences from our extracted papers
3. Runs spaCy NER on each sentence
4. Prints every entity found with its label (PERSON, ORG, etc.)
5. Also prints a formatted visualization

Use real sentences from our CV papers if we have them. Explain
what NER is doing and why we need it."
```

**What's happening:** spaCy reads a sentence and highlights all
the "things" — people's names, organizations, technical terms.
"Vaswani et al. at Google proposed the Transformer" → entities:
Vaswani (PERSON), Google (ORG), Transformer (PRODUCT/CONCEPT).

These entities become the NODES of our knowledge graph.

---

**Step 3.2 — Relation extraction with REBEL**

```
Prompt for Claude Code:
"Create src/relation_extractor.py that:
1. Installs transformers
2. Loads the REBEL model (Babelscape/rebel-large) from HuggingFace
3. Takes a sentence and extracts (subject, relation, object) triples
4. Test it on 10 sentences from our papers
5. Print each triple clearly: Subject → Relation → Object
6. Handle sentences where REBEL finds nothing gracefully

Add comments explaining: what is a 'triple'? what does REBEL
actually do internally? Why is this different from NER?

IMPORTANT: REBEL is ~1.2GB. It runs on CPU but is slow (~2-3 sec
per sentence. That's fine for batch processing."
```

**What's happening:** This is the key step. REBEL reads:
"DETR uses transformers for object detection and outperforms
Faster R-CNN on COCO" and outputs TWO triples:
- (DETR) → uses → (transformers for object detection)
- (DETR) → outperforms → (Faster R-CNN)

These triples become the EDGES of our knowledge graph.

NER finds the things. REBEL finds the relationships between things.
Together they give you structured knowledge from raw text.

---

**Step 3.3 — Batch extraction pipeline**

```
Prompt for Claude Code:
"Create src/knowledge_extractor.py that:
1. Takes all our cleaned documents from Phase 1
2. Splits each into sentences (using spaCy sentence segmentation)
3. Runs REBEL on each sentence to extract triples
4. For each triple, also store: source_file, page_number,
   sentence_text (for attribution later)
5. Saves all triples to data/processed/triples.json
6. Prints stats: total triples extracted, most common entities,
   most common relation types
7. Add a progress bar (tqdm) since this will take time

Optimize: skip very short sentences (<10 words) and very long
ones (>100 words) — REBEL struggles with both. Process in batches."
```

**What's happening:** We're reading every sentence in every paper
and pulling out structured facts. On 300 papers this might take
several hours on CPU — run it overnight. The output is a big JSON
file of (subject, relation, object, source) entries.

**Key concept — WHY is this hard?** Language is ambiguous.
"The model was beaten by the new approach" — REBEL needs to figure
out that "the model" is the subject and "the new approach" is the
object, and the relation is "beaten by." This is genuinely hard NLP.

---

### Phase 3 checkpoint

- [ ] spaCy NER finding entities in your papers
- [ ] REBEL extracting (subject, relation, object) triples
- [ ] Full batch pipeline producing triples.json
- [ ] Understanding the difference between text search and structured knowledge

**Commit:** `git add -A && git commit -m "phase 3: NER and relation extraction pipeline"`

---

## PHASE 4: Knowledge Graph Construction (Week 4)

### What you'll learn
- What a graph database actually is
- How to build and query one with NetworkX
- Graph traversal — the "hops" concept

---

**Step 4.1 — Building the graph**

```
Prompt for Claude Code:
"Create src/knowledge_graph.py that:
1. Loads triples.json from Phase 3
2. Builds a NetworkX directed graph where:
   - Nodes = unique entities (subjects and objects from triples)
   - Edges = relations, with attributes: relation_type,
     source_file, sentence_text
3. Merges similar entity names (e.g., 'YOLO' and 'YOLOv3' should
   be connected, 'Vaswani' and 'A. Vaswani' should merge)
   Use fuzzy matching with thefuzz library.
4. Prints graph stats: node count, edge count, most connected
   nodes (highest degree), most common relation types
5. Saves the graph to data/processed/knowledge_graph.gpickle

Explain what 'directed graph' means and why direction matters
for relations like 'outperforms' (A outperforms B ≠ B outperforms A)."
```

**What's happening:** You're building the road map. Every entity
is a city. Every relation is a road between cities, and the road
has a name (like "outperforms" or "cites" or "proposed by").

---

**Step 4.2 — Graph traversal (the multi-hop magic)**

```
Prompt for Claude Code:
"Add these functions to src/knowledge_graph.py:

1. get_neighbors(entity, hops=1) — find all nodes within N hops
   of a given entity. Return the full path taken.

2. find_path(entity_a, entity_b) — find the shortest path between
   two entities, showing every hop and relation.

3. get_subgraph(entity, hops=2) — extract a small subgraph around
   an entity showing all connections within 2 hops.

Test each function with real entities from our graph.
Print the results in a readable format like:
  DETR --[outperforms]--> Faster R-CNN --[proposed by]--> Ren et al.

Explain what 'hops' means with a concrete example."
```

**What's happening:** THIS is what makes KG-RAG possible. When
someone asks a multi-hop question, you extract the entities from
their question, then WALK the graph to find connected information.

1 hop: direct connection (DETR → outperforms → Faster R-CNN)
2 hops: one step removed (DETR → outperforms → Faster R-CNN → proposed by → Ren)
3 hops: two steps removed (and so on)

The graph gives you answers that no amount of text searching can.

---

**Step 4.3 — Graph visualization**

```
Prompt for Claude Code:
"Create src/graph_viz.py that:
1. Installs pyvis
2. Takes our NetworkX graph (or a subgraph)
3. Renders an interactive HTML visualization where:
   - Nodes are colored by type (PERSON=blue, PAPER=green, etc.)
   - Edge labels show the relation type
   - You can zoom, pan, click nodes to see details
   - Node size reflects how connected it is
4. Generate a visualization of the subgraph around 'Transformer'
   (or whatever major entity exists in our graph) with 2 hops
5. Save to data/processed/graph_viz.html

This will be a key demo piece for interviews."
```

---

### Phase 4 checkpoint

- [ ] Knowledge graph built from extracted triples
- [ ] Can traverse the graph with 1, 2, 3 hops
- [ ] Interactive visualization working
- [ ] Understand WHY graph traversal answers questions text search can't

**Commit:** `git add -A && git commit -m "phase 4: knowledge graph construction and traversal"`

---

## PHASE 5: Hybrid Retrieval — Combining Both Paths (Week 5)

### What you'll learn
- How to run vector search and graph search in parallel
- What a cross-encoder re-ranker does and why it matters
- The architecture that makes KG-RAG actually work

---

**Step 5.1 — Graph-based retrieval**

```
Prompt for Claude Code:
"Create src/graph_retriever.py that:
1. Takes a natural language query
2. Extracts entities from the query using spaCy NER
3. For each entity found, searches our knowledge graph:
   - Exact match first, then fuzzy match
   - Gets all nodes within 2 hops
   - Collects the source sentences for every edge traversed
4. Returns a list of 'graph evidence' items, each with:
   - the triple (subject, relation, object)
   - the source file and sentence
   - the hop distance from the query entity
5. Test with: 'Which models outperform YOLO?' and print results"
```

---

**Step 5.2 — Hybrid retrieval (the combination)**

```
Prompt for Claude Code:
"Create src/hybrid_retriever.py that:
1. Takes a query
2. Runs BOTH retrievals in parallel:
   - Vector search (from Phase 2) → top 10 chunks
   - Graph search (from Phase 5.1) → all graph evidence
3. Combines results into a single ranked list
4. Each result has: text, source, retrieval_method ('vector' or
   'graph'), relevance_score
5. Remove near-duplicates (same source + high text overlap)
6. Return top-K combined results (default K=8)

Test with both a simple question (vector search should win) and
a multi-hop question (graph search should win). Print which
method contributed what."
```

---

**Step 5.3 — Cross-encoder re-ranking**

```
Prompt for Claude Code:
"Update src/hybrid_retriever.py to add re-ranking:
1. Install cross-encoder model: cross-encoder/ms-marco-MiniLM-L-6-v2
2. After combining vector + graph results, re-rank ALL of them
   using the cross-encoder
3. The cross-encoder takes (query, passage) pairs and scores how
   relevant the passage actually is to the query — more accurate
   than the initial retrieval scores
4. Return results sorted by cross-encoder score
5. Print: original rank vs re-ranked position for each result

Explain in comments: why do we need re-ranking? How is a cross-
encoder different from the embedding model we used earlier?"
```

**Key concept — WHY re-rank?** The embedding model (Phase 2) is
fast but approximate — it encodes query and passage separately.
The cross-encoder is slow but precise — it reads query AND passage
together and directly judges relevance. We use embeddings for the
fast first pass, then the cross-encoder to re-order the top results.

---

### Phase 5 checkpoint

- [ ] Graph-based retrieval working from natural language queries
- [ ] Both retrieval methods running and combining results
- [ ] Cross-encoder re-ranking improving result quality
- [ ] Can see which method (vector vs graph) contributes to each answer

**Commit:** `git add -A && git commit -m "phase 5: hybrid retrieval with cross-encoder re-ranking"`

---

## PHASE 6: LLM Synthesis with LangGraph (Week 6)

### What you'll learn
- How to orchestrate multi-step AI workflows
- Prompt engineering for grounded, cited answers
- Building a proper pipeline with LangGraph

---

**Step 6.1 — LangGraph basics**

```
Prompt for Claude Code:
"Create notebooks/03_langgraph_intro.py that:
1. Installs langgraph
2. Creates a simple 3-step graph:
   Step 1: take user input
   Step 2: process it (just uppercase for now)
   Step 3: format output
3. Show the graph executing step by step with state printed
4. Explain in comments: what is LangGraph? Why not just call
   functions in sequence? What does the 'state' concept mean?"
```

**What's happening:** LangGraph is a framework for building
multi-step AI workflows as a graph of operations. Each node is
a step, edges define what runs next. The state is a dict that
passes between steps, accumulating results.

---

**Step 6.2 — The KG-RAG pipeline as a LangGraph**

```
Prompt for Claude Code:
"Create src/pipeline.py that builds our full KG-RAG pipeline
as a LangGraph workflow:

State: { query, vector_results, graph_results, merged_results,
         reranked_results, answer, sources }

Nodes:
1. retrieve_vector — run vector search (Phase 2)
2. retrieve_graph — run graph search (Phase 5.1)
3. merge_and_rerank — combine + cross-encoder (Phase 5.3)
4. synthesize — send top results to Gemini Flash with a prompt
   that says: answer the question using ONLY the provided context,
   cite sources as [filename, page], if the context doesn't
   contain enough information say so explicitly
5. format_output — structure the final response

Edges: query → [retrieve_vector, retrieve_graph] (parallel)
       → merge_and_rerank → synthesize → format_output

Test with 3 questions of increasing complexity."
```

---

**Step 6.3 — Prompt engineering**

```
Prompt for Claude Code:
"Update the synthesize step in src/pipeline.py with a well-
engineered prompt for Gemini Flash. The prompt should:

1. Clearly separate CONTEXT (our retrieved chunks/triples) from
   the QUESTION
2. Instruct the model to:
   - Answer ONLY from the provided context
   - Cite every claim as [Source: filename.pdf, p.X]
   - If the answer requires connecting information from multiple
     sources, explicitly show the reasoning chain
   - If the context is insufficient, say what's missing
   - Distinguish between facts stated in papers vs inferred
     connections from the knowledge graph
3. Include the retrieval method (vector vs graph) in the context
   so the LLM knows which evidence came from text similarity vs
   structural relationships

Test with our 3 questions and compare answer quality before
and after the prompt improvements."
```

---

### Phase 6 checkpoint

- [ ] LangGraph pipeline orchestrating the full flow
- [ ] Gemini Flash generating cited, grounded answers
- [ ] Multi-hop questions getting better answers than Phase 2's basic RAG
- [ ] Can trace exactly which retrieval method contributed to each answer

**Commit:** `git add -A && git commit -m "phase 6: LangGraph pipeline with LLM synthesis"`

---

## PHASE 7: UI, Evaluation & Polish (Weeks 7–8)

### What you'll learn
- Building a demo-worthy UI with Streamlit
- How to evaluate RAG systems properly
- Interview-ready presentation of your work

---

**Step 7.1 — Streamlit UI**

```
Prompt for Claude Code:
"Create src/app.py — a Streamlit app with:

1. Sidebar:
   - Upload PDFs section (batch upload)
   - 'Process documents' button that runs ingestion + graph building
   - Progress bars during processing
   - Stats: total docs, chunks, entities, triples

2. Main area:
   - Search box for questions
   - Answer displayed with cited sources highlighted
   - Below the answer: expandable sections showing:
     a) Retrieved vector chunks (with similarity scores)
     b) Retrieved graph evidence (with hop counts)
     c) Re-ranked combined results
   - Toggle to compare 'Basic RAG' vs 'KG-RAG' answers side by side

3. Separate tab: Interactive knowledge graph visualization (pyvis)
   - Filter by entity type, relation type
   - Click a node to see all its connections
   - Search for specific entities

Make it visually clean and professional."
```

---

**Step 7.2 — Evaluation**

```
Prompt for Claude Code:
"Create src/evaluate.py that:
1. Defines 20 test questions in 3 categories:
   - Simple (single-doc answer): 7 questions
   - Medium (cross-doc, 1-2 hops): 7 questions
   - Hard (multi-hop, 3+ hops): 6 questions
2. For each question, manually provide the expected answer
3. Runs both Basic RAG (Phase 2) and KG-RAG (Phase 6) on all 20
4. Uses Gemini Flash as a judge: given (question, expected_answer,
   system_answer), score 1-5 on: correctness, completeness,
   citation accuracy
5. Print a comparison table and save to data/eval_results.json

This gives you HARD NUMBERS to show in interviews:
'KG-RAG scored 4.2/5 on multi-hop questions vs 1.8/5 for basic RAG'"
```

---

**Step 7.3 — README and documentation**

```
Prompt for Claude Code:
"Create a comprehensive README.md that includes:
1. Project title and one-line description
2. Architecture diagram (ASCII or Mermaid)
3. What problem this solves (with concrete example)
4. How it works (simplified 5-step explanation)
5. Results: evaluation scores table (basic RAG vs KG-RAG)
6. Tech stack with version numbers
7. Setup instructions (step by step)
8. Screenshots/GIFs of the Streamlit UI
9. Limitations and future work
10. Your name and links

This is the first thing recruiters/interviewers see on GitHub."
```

---

### Phase 7 checkpoint

- [ ] Streamlit UI fully functional and professional-looking
- [ ] Evaluation results showing KG-RAG > Basic RAG on multi-hop questions
- [ ] README that makes someone want to try it in 30 seconds
- [ ] Project deployed on GitHub with clean commit history

**Commit:** `git add -A && git commit -m "phase 7: UI, evaluation, documentation"`

---

## How to use Claude Code effectively for this project

### The learning loop

For every step above, follow this exact pattern:

```
1. Give Claude Code the prompt listed above
2. READ every line of code it writes — don't just run it
3. If you don't understand something, ask:
   "Explain what lines 23-35 are doing in simple terms"
4. Run it and see the output
5. If something breaks, paste the error and ask Claude Code to fix it
6. Before moving to the next step, ask:
   "Summarize what we just built and how it connects to the
    overall KG-RAG architecture"
```

### Useful Claude Code commands for learning

```
"Explain this function like I'm new to NLP"
"What would happen if I changed chunk_size to 200? Why?"
"Draw an ASCII diagram of how data flows through this module"
"What are the 3 most important things to understand in this file?"
"Show me the difference between what basic RAG returns and what
 our graph retriever returns for this query"
"Why did you choose this library over alternatives?"
```

### When you get stuck

```
"I'm stuck on [specific issue]. Show me the current state of
 the code, explain what's wrong, and fix it step by step."

"This works but I don't understand WHY. Walk me through what
 happens when I call search('attention mechanism') — trace the
 full execution path."
```

---

## Package versions (tested on CPU, 16GB RAM)

```
PyMuPDF==1.24.0
sentence-transformers==3.0.0
chromadb==0.5.0
spacy==3.7.0
transformers==4.42.0       # for REBEL
torch==2.3.0               # CPU-only: pip install torch --index-url https://download.pytorch.org/whl/cpu
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

Save as `requirements.txt`. Install with:
`pip install -r requirements.txt`
`python -m spacy download en_core_web_sm`

---

## Final folder structure

```
kg-rag/
├── data/
│   ├── raw/                 # Your PDF papers go here
│   ├── processed/
│   │   ├── triples.json     # Extracted relations
│   │   ├── knowledge_graph.gpickle
│   │   └── graph_viz.html   # Interactive visualization
│   └── chroma_db/           # Vector store (auto-created)
├── notebooks/
│   ├── 01_embeddings_demo.py
│   ├── 02_ner_demo.py
│   └── 03_langgraph_intro.py
├── src/
│   ├── ingestion.py         # Phase 1: PDF extraction + cleaning
│   ├── chunker.py           # Phase 1: Text chunking
│   ├── vector_store.py      # Phase 2: ChromaDB embeddings
│   ├── basic_rag.py         # Phase 2: Generic RAG (for comparison)
│   ├── relation_extractor.py # Phase 3: REBEL triple extraction
│   ├── knowledge_extractor.py # Phase 3: Batch extraction pipeline
│   ├── knowledge_graph.py   # Phase 4: Graph construction + traversal
│   ├── graph_viz.py         # Phase 4: Pyvis visualization
│   ├── graph_retriever.py   # Phase 5: Graph-based retrieval
│   ├── hybrid_retriever.py  # Phase 5: Combined retrieval + re-ranking
│   ├── pipeline.py          # Phase 6: LangGraph orchestration
│   ├── evaluate.py          # Phase 7: Evaluation framework
│   └── app.py               # Phase 7: Streamlit UI
├── tests/
├── .env                     # API keys (not committed)
├── .gitignore
├── requirements.txt
└── README.md
```
