# KG-RAG — Knowledge-Graph-Augmented Retrieval over Research Papers

**Upload your own research papers. Ask questions across all of them at once.**

Standard RAG can search a corpus. It cannot reason *across* it. This system builds two parallel
memories of the documents you upload — a vector index for meaning and a knowledge graph for
relationships — and walks the graph to answer questions whose answers exist in no single
document.

No preloaded data: the workspace starts empty and you fill it. Runs entirely on CPU, no GPU
anywhere.

---

## The problem, concretely

Ask a normal "chat with your PDFs" system this:

> *Which models outperform Faster R-CNN, and who proposed those models?*

It embeds the question, retrieves the five most similar chunks, and hands them to an LLM.
Those chunks will be topically plausible — detection papers, author lists — and the answer
will be vague or invented. Not because the retrieval was tuned badly, but because **no chunk
of text contains the answer.** Answering requires a chain:

```
Faster R-CNN  ←[outperforms]—  Mask R-CNN  ←[proposed by]—  He et al.
```

Fact 1 is in `mask_rcnn.pdf`. Fact 2 is in its author block. The *connection* is in neither.

Vector search has exactly one move: *find text similar to this text*. It executes that move
once and stops. Following a chain is not a harder version of that move — it is a different
operation, and no chunk size, embedding model, or prompt provides it.

This project adds the missing operation.

<details>
<summary><b>See it fail firsthand</b> (this is the whole motivation)</summary>

```bash
python src/basic_rag.py --demo-limitation
```

Runs one question basic RAG handles well and one it cannot, back to back. The instructive
part is that **the retrieval scores look identical** — around 0.50 in both cases. Retrieval
does not fail loudly. It returns its *k* nearest neighbours as always, none of which holds
the answer, and the LLM hedges or invents a connection.
</details>

---

## Architecture

Two independent memories, built once, searched together.

```mermaid
flowchart TD
    PDF[PDFs you upload]

    PDF --> ING[ingestion.py<br/>column-aware extraction<br/>+ cleaning]

    ING --> CH[chunker.py<br/>overlapping chunks]
    ING --> SENT[spaCy sentence split]

    CH --> EMB[all-MiniLM-L6-v2<br/>384-dim embeddings]
    EMB --> VS[(ChromaDB<br/>MEMORY 1: meaning)]

    SENT --> REB[REBEL<br/>Wikidata relations]
    SENT --> DOM[dependency parser<br/>outperforms / trained_on / ...]
    REB --> TR[per-document triples]
    DOM --> TR
    TR --> KG[(NetworkX MultiDiGraph<br/>MEMORY 2: relationships)]

    Q[user question] --> VR[vector search]
    Q --> GR[graph_retriever.py<br/>entity match + walk 1-3 hops]
    VS --> VR
    KG --> GR

    VR --> MG[merge + dedupe]
    GR --> MG
    MG --> RR[cross-encoder re-rank<br/>ms-marco-MiniLM-L-6-v2]
    RR --> SY[Gemini Flash<br/>grounded, cited synthesis]
    SY --> ANS[answer with citations]

    style VS fill:#1f4e79,color:#fff
    style KG fill:#1e5e3a,color:#fff
    style ANS fill:#5c4317,color:#fff
```

Retrieval and synthesis are orchestrated as a **LangGraph** workflow, with the two retrievers
running as parallel branches that join at the re-ranker.

---

## How it works, in five steps

1. **Extract and clean.** PyMuPDF pulls text *in reading order* — two-column academic layouts
   interleave columns if you extract naively. Running headers are found by cross-page
   repetition, references are stripped while appendices are preserved, and PDF ligatures are
   normalised (`Efﬁcient` → `Efficient`, which otherwise splits one graph node into two).

2. **Build Memory 1.** Text is split into overlapping ~500-character chunks, embedded with
   `all-MiniLM-L6-v2`, and stored in ChromaDB. Every chunk keeps its source file and page so
   claims can be cited.

3. **Build Memory 2.** Every sentence goes through two relation extractors, producing
   `(subject, relation, object)` triples with provenance. These become a directed multigraph:
   entities are nodes, relations are labelled edges. Similar entity names are merged
   (`A. Vaswani` = `Vaswani`) — but *versioned* names are deliberately kept separate
   (`YOLO` ≠ `YOLOv3`) and linked instead.

4. **Retrieve both ways.** The question is embedded *and* parsed for entities. Vector search
   finds resembling text; graph traversal walks 1–3 hops out from the matched entities,
   preferring edges that match the relation the question asks about. Results are merged,
   de-duplicated, and re-ranked by a cross-encoder that reads question and passage *together*.

5. **Synthesise.** Gemini Flash receives the evidence with each item labelled by how it was
   retrieved, and is instructed to distinguish what a paper **states** from what the graph
   **connects**, to cite everything, and to say plainly when the evidence is insufficient.

**Processing is incremental.** A manifest records what has been done to each document, keyed by
content hash, so adding one paper to a corpus of forty processes one paper. The one stage that
stays whole-corpus is the graph build — entity resolution canonicalises toward the most frequent
surface form *across* the corpus, so adding a document can legitimately change an existing
entity's canonical name. That is genuine whole-corpus dependence, not an implementation shortcut.

**Follow-ups are resolved before retrieval.** Retrieval is stateless: *"how is it trained?"*
embeds to nothing useful and names no entity. So within a chat, a follow-up is first rewritten
into a standalone question using that chat's history. Both forms are kept and the resolved one
is shown in the UI — a bad rewrite would otherwise look like a retrieval bug.

---

## Two findings worth reading before you trust the design

**REBEL cannot express the relations this domain needs.** `Babelscape/rebel-large` is trained
on Wikipedia aligned to **Wikidata properties**, so its labels come from that closed
vocabulary: `instance of`, `part of`, `developer`, `use`. There is no `outperforms` property
and never will be. Run `python src/relation_extractor.py` and read the vocabulary it reports.

A graph built on REBEL alone is taxonomically tidy and useless for comparative reasoning — the
flagship question above has no edge to walk. So a second extractor runs over the same
sentences, using spaCy's **dependency parse** to capture `outperforms`, `trained_on`,
`evaluated_on`, `achieves`, `proposes`, `extends`. Dependency parsing rather than regex,
because a regex finds the word "outperforms" but cannot tell you which noun is doing the
outperforming — and direction is the entire value of a directed edge.

**Absolute similarity scores mean nothing on their own.** The canonical demo pairs
*"YOLO is a real-time object detection system"* with *"You Only Look Once detects objects
quickly"* and expects ~0.85. Ours scores **0.48**, and nothing is wrong: cosine similarity has
no absolute scale, and MiniLM compresses its scores into a narrow band. What matters is the
*contrast* — 0.48 against 0.05 for an unrelated pair, a 9.6× separation. This is also why the
cross-encoder earns its place: when candidates sit in a narrow band, you need a sharper judge
to order them.

---

## Results (optional benchmark)

> This is a **developer benchmark, not part of using the app.** It measures the system against
> plain vector-only RAG on a fixed question set written for the optional eight-paper demo corpus.
> The numbers describe *that* corpus — not whatever documents you upload. Reproduce with
> `python scripts/fetch_papers.py` first.

Twenty questions in three tiers, run through **both** systems and scored 1–5 by an LLM judge
against hand-written reference answers.

```bash
python src/evaluate.py
python src/evaluate.py --basic-top-k 12 --output data/eval_results_controlled.json
python scripts/update_readme_results.py
```

The second command is the one that makes the first believable. KG-RAG hands the LLM 12
evidence items; standard basic RAG hands it 5. So part of any gap is simply *more context*,
not the graph — and on single-document questions, which need no traversal at all, it can only
be that. Giving basic RAG the same 12 items isolates what graph retrieval actually contributes.

<!-- RESULTS_TABLE_START -->
**System vs system** — full KG-RAG against the standard basic-RAG baseline:

| Tier | n | Basic RAG | KG-RAG | Delta | Graph share of evidence |
|---|---:|---:|---:|---:|---:|
| Simple (single-doc) | 7 | 4.29 | 4.95 | +0.66 | 26% |
| Medium (cross-doc, 1-2 hops) | 7 | 4.10 | 4.81 | +0.71 | 33% |
| **Hard (multi-hop, 3+)** | 6 | **2.34** | **3.22** | **+0.88** | 50% |
| All | 20 | 3.63 | 4.38 | +0.75 | 36% |

<sub>From `data/eval_results.json` (20 questions, 6.9 min).</sub>

**Controlled** — basic RAG given the *same* 12 evidence items, so the only remaining difference is graph retrieval:

| Tier | n | Basic RAG | KG-RAG | Delta | Graph share of evidence |
|---|---:|---:|---:|---:|---:|
| Simple (single-doc) | 7 | 4.47 | 4.86 | +0.39 | 26% |
| Medium (cross-doc, 1-2 hops) | 7 | 4.29 | 4.81 | +0.52 | 33% |
| **Hard (multi-hop, 3+)** | 6 | **3.17** | **3.39** | **+0.22** | 50% |
| All | 20 | 4.02 | 4.40 | +0.38 | 36% |

<sub>From `data/eval_results_controlled.json` (20 questions, 6.4 min).</sub>

<sub>Each score is the mean of correctness, completeness and citation accuracy, judged 1-5 against a hand-written reference answer. Per-question answers and the judge's reasoning are in the JSON files, so any number here can be checked by hand.</sub>
<!-- RESULTS_TABLE_END -->

**How to read it.** The `simple` tier is not padding — a delta near zero there is the *correct*
result, showing the graph did not damage what already worked. The `hard` tier is where the
graph has to earn its place. The `graph share` column is the honesty check: it reports how much
of the final evidence actually came from traversal. If hard questions score well with a graph
share near zero, the gap came from somewhere else and the claim is unsupported.

The evaluation is designed so it can fail. If KG-RAG matched basic RAG on multi-hop questions,
that would mean five phases of work were not worth it, and that is what the numbers would say.

---

## What it looks like

The knowledge graph is generated as a self-contained interactive HTML file — drag nodes, zoom,
hover for an entity's aliases and source papers, click an edge for the sentence it came from:

```bash
python src/graph_viz.py --entity "Faster R-CNN" --hops 2
start data/processed/graph_viz.html
```

Nodes are coloured by a type inferred from the relations they participate in (see below), and
sized by degree. On the eight-paper corpus that classifies 2,766 entities as 238 models,
94 datasets, 84 metrics, 22 people, 24 organisations, and the rest concepts.

> **Screenshots:** add PNGs of the Streamlit UI and the graph to `docs/` and link them here.
> Two worth capturing: the side-by-side *Basic RAG vs KG-RAG* comparison, and a multi-hop
> chain in the graph tab's path finder.

### Typical query cost (CPU, 8-paper corpus)

| Stage | Time |
|---|---|
| Vector search | 0.2 s |
| Graph traversal (2 hops) | 2.2 s |
| Merge + cross-encoder re-rank | 6.9 s |
| **Total before LLM call** | **~9 s** |

Re-ranking dominates, because scoring ~26 candidate passages with a cross-encoder means 26
forward passes on CPU. That is the deliberate trade described above: slow and precise, applied
only to the shortlist.

---

## Tech stack

| Purpose | Library / model | Version | Notes |
|---|---|---|---|
| PDF extraction | PyMuPDF | 1.28.2 | Block geometry for column-aware reading order |
| NLP / parsing | spaCy + `en_core_web_sm` | 3.8.16 | Sentence splitting and dependency parses |
| Relation extraction | `Babelscape/rebel-large` | via transformers 4.57.6 | ~1.6GB, ~1.3 s/sentence on CPU |
| Embeddings | `all-MiniLM-L6-v2` | via sentence-transformers 5.7.0 | 23M params, 384 dims |
| Vector store | ChromaDB | 1.5.9 | Persistent, cosine space |
| Graph | NetworkX | 3.6.1 | `MultiDiGraph` + pickle |
| Entity resolution | thefuzz / RapidFuzz | 0.22.1 / 3.14.5 | C-speed fuzzy matching |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | sentence-transformers | ~80M params |
| Orchestration | LangGraph | 1.2.11 | Parallel retrieval branches |
| LLM | Gemini Flash via `google-genai` | 2.19.0 | Groq as fallback |
| Visualisation | pyvis | 0.3.2 | Self-contained interactive HTML |
| UI | Streamlit | 1.62.0 | |
| Runtime | PyTorch | 2.13.0+**cpu** | CPU-only build, deliberately |

> **Deviation from the original spec:** `google-generativeai` is deprecated upstream; this uses
> the supported `google-genai` SDK. Other pinned versions were 2024-era and no longer resolve.

Developed on a Ryzen 7 5700U, 16GB RAM, **no dedicated GPU**. Every model choice is constrained
by that: nothing above ~400M parameters runs locally, and all heavy generation goes to an API.

---

## Setup

```bash
git clone <your-repo-url> && cd kg-rag
python -m venv venv && venv\Scripts\activate        # Linux/macOS: source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Add your API key:

```bash
cp .env.example .env       # then edit .env and paste your key
```

Get a free one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
**Retrieval works with no key at all** — only answer generation needs it.

Then start the app:

```bash
streamlit run src/app.py
```

**There is no preloaded data.** The workspace starts empty; upload your own PDFs in the sidebar.

---

## How you actually use it

### The workspace and chats

One **workspace** holds your uploaded PDFs and the two memories built from them. Inside it you
can keep many **chats**. Every chat searches the same documents, but each keeps its own
conversation history — so a follow-up like *"how is it trained?"* is resolved against the chat
you asked it in, and nothing else.

Deleting a chat costs you that conversation. Deleting the workspace deletes everything: PDFs,
both memories, and every chat.

### Uploading

Drop up to 50 PDFs into the sidebar and press **Process documents**. Only documents you have
not already added are processed — re-uploading the same file is a no-op, and adding one paper to
a corpus of forty processes one paper, not forty-one.

Documents are identified by content hash, so two different papers that happen to share a
filename are stored separately rather than overwriting each other. That matters because the
filename becomes the citation string the model prints.

### The two-speed extraction trade-off

This is the one thing worth understanding before you use it at scale.

| | Fast path (on upload) | Full extraction (`--rebel`) |
|---|---|---|
| What it extracts | `outperforms`, `trained_on`, `achieves`, authorship | + `subclass of`, `part of`, `use` |
| 50 papers | **~2 minutes** | **~5–6 hours** |
| Where it runs | in the app | command line |

Uploading runs the fast path only, so the workspace is queryable within minutes. REBEL adds the
taxonomic relations that connect the graph into one traversable component — worth having, but
not worth blocking a browser tab for hours. Run it once, overnight:

```bash
python src/knowledge_extractor.py --rebel
```

It processes one document at a time and records progress after each, so it is safe to interrupt
and resume — and adding new PDFs later only queues *those*.

### Optional: a demo corpus

If you want something to try it on immediately, this downloads eight open-access arXiv papers
chosen because they cite and benchmark against each other (which is what makes the graph
interesting). It is not required, and nothing in the app depends on it:

```bash
python scripts/fetch_papers.py
```

These same eight papers are the corpus the committed benchmark results describe.

---

## Try it

The app is the main way in. These CLI entry points work against whatever is in your workspace,
and are useful for inspecting one stage at a time:

```bash
python src/workspace.py                     # what is loaded, and what still needs REBEL
python src/pipeline.py -i                   # ask questions without the UI
python src/graph_retriever.py -q "your question here"    # graph walk only, no LLM needed
python src/knowledge_graph.py --path "entity A" "entity B"
python src/knowledge_extractor.py --rebel   # the overnight extraction job
```

Two demos that need the optional eight-paper corpus, because they name specific papers:

```bash
python src/basic_rag.py --demo-limitation   # watch vector-only RAG fail on a multi-hop question
python src/evaluate.py                      # the benchmark
```

The learning scripts, in order:

```bash
python notebooks/01_embeddings_demo.py    # what an embedding actually is
python notebooks/02_ner_demo.py           # what NER gets right and badly wrong here
python notebooks/03_langgraph_intro.py    # state, nodes, edges, parallel branches
```

---

## Tests

```bash
pytest tests/ -q
```

150 tests, no API key required. Several encode real bugs found during development, so they
cannot come back:

- the semantic chunker turning 56K characters of Mask R-CNN into 516K of chunks, because
  table rows have no sentence terminators and the "last sentence" carried forward was the
  entire chunk
- reference-stripping deleting ViT's appendix along with its bibliography
- the domain extractor silently returning **zero** triples because spaCy's lemmatizer had been
  disabled for speed, and verb lookup is by lemma
- LangGraph raising `InvalidUpdateError` because two parallel nodes wrote the same state key
- the evaluation harness scoring a failed API call as a 1/5 answer, which turned 17 dead
  requests into a clean-looking table reading "1.00 across every tier" — the most dangerous
  bug in the project, because it produced confident, specific, wrong numbers

---

## Limitations

Stated plainly, because a portfolio project that claims no weaknesses is not credible.

- **Extraction quality bounds everything.** REBEL produces confident errors
  (`SSD --[has part]--> YOLOv3`). The graph inherits them, and no downstream component can
  detect that an edge is wrong. Precision on the domain relations is good; recall is not
  measured.
- **Entity resolution is a heuristic.** Fuzzy matching merges `A. Vaswani` with `Vaswani` and
  keeps `YOLO` apart from `YOLOv3`, but `the DETR model` remains a separate node from `DETR`,
  linked rather than merged. Aggressive merging risks corrupting the graph; conservative
  merging fragments it. There is no clean answer at this scale.
- **The graph only knows what was extracted.** Anything expressed as ordinary prose —
  definitions, caveats, motivation — is invisible to traversal. This is exactly why the
  vector memory is kept rather than replaced.
- **LLM-as-judge is imperfect.** Judges favour longer answers and their own phrasing, and
  Gemini judging Gemini shares blind spots. Both systems face the same judge and the same
  rubric, so the *difference* is more trustworthy than either absolute score. Treat a 0.3 gap
  as noise.
- **The headline comparison has a confound, which is why there are two tables.** KG-RAG is
  not only "basic RAG plus a graph" — it also gets a larger evidence budget and a
  cross-encoder re-ranker. The controlled run matches the budget; the remaining gap is
  attributable to the graph, and it is smaller than the headline number. Quote the controlled
  table when the claim is specifically about knowledge graphs.
- **n=20 with a single judge pass.** No repeated trials, no confidence intervals, no second
  judge. These are directional results on a small question set, not a benchmark.
- **Free-tier quota shapes what is measurable.** `gemini-3.6-flash` allows 20 requests *per
  day*; one evaluation run needs ~80. The harness pins a `-lite` model and aborts loudly
  rather than scoring failed calls — an earlier run silently reported "1.00 across every
  tier", which was 17 dead API calls wearing the costume of a finding.
- **Designed for ~50 documents, verified at a smaller scale.** Ingest is incremental and the
  projections come from measured per-document costs, but a full 50-paper run has not been
  executed end to end. The bottleneck is known: entity resolution is O(n²) in distinct entity
  strings, and it is the one stage that cannot be made incremental, because canonicalisation
  depends on corpus-wide frequency. NetworkX is in-memory; Neo4j would be the move past roughly
  100k nodes.
- **Follow-ups are resolved by an LLM rewrite, which can be wrong.** "How is it trained?" becomes
  a standalone question before retrieval. When the rewrite misreads the referent, the answer is
  confidently about the wrong thing. The resolved question is displayed for exactly this reason —
  but it is a real failure mode that did not exist when every question was independent.
- **Chats share one workspace, and there is only one.** No multiple corpora, no per-user
  separation, no auth. Deleting the workspace is irreversible.
- **Fast-path uploads produce a thinner graph.** Uploading extracts comparative and authorship
  relations but not REBEL's taxonomic ones, so multi-hop reach is lower until the overnight
  `--rebel` job runs. The sidebar says how many documents are in that state; it is a real
  difference in capability, not just in edge count.

## Future work

- Precision/recall measurement of extracted triples against a hand-labelled sample
- Neo4j backend, and Cypher instead of hand-written traversal
- Learned entity linking rather than fuzzy string matching
- Query decomposition: split a multi-hop question into sub-questions and walk each separately
- Contradiction detection — surfacing where two papers disagree is the genuinely novel
  application of a corpus-scale knowledge graph

---

## Repository layout

```
src/
  ingestion.py           extraction + cleaning          Phase 1
  chunker.py             fixed + semantic chunking      Phase 1
  vector_store.py        ChromaDB, Memory 1             Phase 2
  basic_rag.py           the baseline to beat           Phase 2
  llm.py                 Gemini/Groq with fallback
  relation_extractor.py  REBEL + domain extractor       Phase 3
  knowledge_extractor.py batch pipeline -> triples.json Phase 3
  knowledge_graph.py     graph + traversal, Memory 2    Phase 4
  graph_viz.py           pyvis visualisation            Phase 4
  graph_retriever.py     natural language -> walk       Phase 5
  hybrid_retriever.py    merge + cross-encoder re-rank  Phase 5
  pipeline.py            LangGraph orchestration        Phase 6
  evaluate.py            20-question harness            Phase 7
  app.py                 Streamlit UI: chat + docs      Phase 7
  workspace.py           manifest, incremental ingest
  chat.py                many chats over one workspace
notebooks/               three explanatory scripts
tests/                   150 tests
scripts/fetch_papers.py  OPTIONAL demo corpus
```

---

## Author

**Jaydeep** — B.Tech Information Technology, VJTI Mumbai (2027)

Built as a study of why retrieval architecture matters more than model size: every model here
is small enough to run on an integrated GPU-less laptop, and the capability gain comes entirely
from how the retrieval is structured.
