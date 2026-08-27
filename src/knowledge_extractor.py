"""
Phase 3, Step 3.3 - the batch extraction pipeline: corpus -> triples.json.

WHAT THIS DOES
--------------
Reads every cleaned document from Phase 1, splits it into sentences, runs BOTH
relation extractors over each sentence, and writes every triple - with the
provenance needed to cite it later - to data/processed/triples.json.

    PDFs -> clean text -> sentences -> [REBEL] + [domain extractor] -> triples.json

This is the slow, run-it-once step. On our 8-paper corpus it is roughly 2,450
sentences at ~1.3s each, so about an hour. On 300 papers it is an overnight job.
Nothing at query time ever runs this - Phase 4 loads the finished JSON.

WHY TWO EXTRACTORS
------------------
See the long comment in relation_extractor.py. Briefly: REBEL emits Wikidata
relations ("instance of", "developer", "based on") and structurally cannot emit
"outperforms", because Wikidata has no such property. The domain extractor covers
exactly the comparative and evaluative relations that CV papers are made of.
Together they produce a graph you can actually traverse to answer
"which models beat YOLO, and who wrote them?".

WHY PROVENANCE IS STORED ON EVERY TRIPLE
----------------------------------------
Each triple carries source_file, page and the original sentence. Three reasons:

  1. Citation. Phase 6 must print "[Source: detr.pdf, p.4]" for graph-derived
     claims, exactly as it does for vector-retrieved ones.
  2. Evidence. A traversal returns a path; the sentences behind that path are
     what actually gets shown to the LLM. Without them, the graph could only say
     THAT two things are related, never on what basis.
  3. Debugging. When a triple looks wrong, the sentence tells you instantly
     whether the extractor misread it or the paper really said that.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import (  # noqa: E402
    PROCESSED_DIR,
    RAW_DIR,
    TRIPLES_PATH,
    banner,
    ensure_dirs,
    setup_console,
)
from src.relation_extractor import extract_domain_triples  # noqa: E402

# REBEL degrades at both extremes: very short sentences carry no relation, and
# very long ones get truncated mid-thought and produce scrambled triples.
MIN_WORDS = 10
MAX_WORDS = 100

# Checkpoint frequency. A 50-minute job that loses everything on a crash (or a
# closed laptop lid) is a bad job.
CHECKPOINT_EVERY = 250

_CHECKPOINT_PATH = PROCESSED_DIR / "triples_partial.json"


def is_useful_sentence(text: str) -> bool:
    """
    Filter out the text that looks like prose to a splitter but is not.

    Research PDFs are full of table rows, equation fragments and figure axis
    labels. They pass a word-count check but contain no relation, and feeding
    them to REBEL wastes ~1.3 seconds each AND pollutes the graph with nodes like
    "37.4 39.8 41.2". At ~2,450 sentences, dropping these is worth real minutes.
    """
    words = text.split()
    if not (MIN_WORDS <= len(words) <= MAX_WORDS):
        return False

    letters = sum(c.isalpha() for c in text)
    if letters / max(len(text), 1) < 0.6:
        return False                     # number/symbol soup: a table row

    # Needs at least a few ordinary lowercase words to be a sentence.
    lowercase_words = sum(1 for w in words if w.isalpha() and w.islower())
    if lowercase_words < 4:
        return False

    # Long runs of numbers separated by spaces are table rows.
    if re.search(r"(\b\d+\.?\d*\s+){4,}", text):
        return False

    return True


# ---------------------------------------------------------------------------
# Paper metadata: title and authors
# ---------------------------------------------------------------------------
#
# WHY THIS IS A SEPARATE MECHANISM FROM THE SENTENCE EXTRACTORS
#
# Both relation extractors read SENTENCES, and authorship is almost never stated
# in one. A paper does not say "Mask R-CNN was proposed by He et al." - it says
# "we propose", and prints the authors in a title block that is not a sentence at
# all. The references section would have carried that information, but Phase 1
# deliberately strips it (it was drowning both memories in citation soup).
#
# The result, before this function existed: the pipeline answered "which models
# outperform Faster R-CNN, and who proposed them?" with the models correctly
# cited and "the provided evidence does not state or name the authors". Honest,
# and a direct miss on one of the project's stated use cases.
#
# So authors are read from the title block instead, and joined to the rest of the
# graph through a paper node:
#
#     Kaiming He --[authored]--> "Mask R-CNN" --[presents]--> Mask R-CNN
#                                (paper node)                 (model entity)
#
# which makes "who proposed the model that beat X" a walkable chain.

_SECTION_HEADINGS = {
    "introduction", "abstract", "related work", "background", "method",
    "methods", "conclusion", "results", "experiments", "contents",
}

# The surname must be Capitalised-then-LOWERCASE. That single constraint rejects
# the model names that kept being read as authors - "Fast R-CNN", "Mask R-CNN" -
# because "R-CNN" has no lowercase run after its capital letter.
_AUTHOR_NAME_RE = re.compile(
    r"\b[A-Z][a-z]{1,15}"                    # first name
    r"(?:\s+[A-Z]\.)?"                       # optional middle initial
    r"\s+[A-Z][a-zÀ-ÿ'´]{1,20}\b"           # surname, accents allowed
)

# Words that make a capitalised pair an institution, not a person.
_AFFILIATION_WORDS = {
    "research", "university", "institute", "lab", "labs", "laboratory",
    "college", "school", "department", "technology", "inc", "corporation",
    "google", "microsoft", "facebook", "meta", "openai", "nvidia", "deepmind",
    "brain", "ai", "intelligence", "science", "sciences", "center", "centre",
    "academy", "corp", "team", "group", "berkeley", "stanford", "mit", "toronto",
    "equal", "contribution", "correspondence", "author", "work", "done",
    # Field-neutral institution vocabulary. Unlike subject terminology, the words
    # organisations are named with are stable across disciplines, so this list
    # generalises where a domain blocklist cannot: it catches "National Bureau"
    # (economics), "Max Planck" (physics), "General Hospital" (medicine) without
    # knowing anything about those fields.
    "bureau", "national", "international", "european", "federal", "state",
    "foundation", "association", "society", "ministry", "hospital", "clinic",
    "agency", "council", "consortium", "trust", "partnership", "division",
    "faculty", "campus", "planck", "cnrs", "inria", "cern", "nasa", "nih",
}

# Ordinary English words that the "Capitalised Capitalised" pattern happily
# matches inside a title or a sentence: "Attention Is", "Deep Residual",
# "Region Proposal". Widening the scan window to catch awkward layouts means
# more prose is in range, so these have to be excluded explicitly.
_NON_NAME_WORDS = {
    "is", "are", "was", "were", "all", "you", "need", "the", "a", "an", "and",
    "or", "for", "with", "using", "towards", "real", "time", "deep", "residual",
    "learning", "image", "recognition", "object", "detection", "region",
    "proposal", "networks", "network", "transformer", "transformers", "attention",
    "incremental", "improvement", "hierarchical", "vision", "shifted", "window",
    "windows", "end", "words", "scale", "self", "mask", "faster", "swin",
    "introduction", "abstract", "figure", "table", "we", "our", "this", "that",
    "in", "on", "at", "by", "of", "to", "from", "code", "models", "available",
    # Technical noun pairs from abstracts that also pass the name shape:
    # "Index Terms", "Convolutional Neural", "Selective Search".
    "index", "terms", "convolutional", "neural", "selective", "search",
    "visual", "computing", "fully", "fast", "sparse", "dense", "feature",
    "pyramid", "single", "shot", "multi", "box", "boxes", "state", "art",
    "keywords", "contents", "supplementary", "material", "appendix",
    # Conference/venue footers: "31st Conference on Neural Information
    # Processing Systems (NIPS 2017), Long Beach, CA, USA".
    "processing", "systems", "conference", "proceedings", "nips", "neurips",
    "advances", "annual", "beach", "long", "usa", "ca", "workshop", "published",
    "preprint", "under", "review", "submitted",
}


def _looks_like_body_phrase(candidate: str, body: str) -> bool:
    """
    True if this "name" is really a phrase from the document's prose.

    The structural signal that makes author extraction domain-neutral. The
    pattern that finds names - Capitalised word, Capitalised word - also matches
    "Gene Expression", "Monetary Policy", "Nash Equilibrium", "Random Forest".
    Blocklisting them does not scale: the existing list is essentially the titles
    of the eight demo papers, and would need rewriting for every new field.

    A real author name appears once or twice on page one and then rarely again.
    A technical phrase recurs throughout the body. Counting occurrences
    distinguishes them without knowing anything about the subject.
    """
    return body.count(candidate) >= 3


def extract_paper_metadata(doc: dict, max_authors: int = 12) -> tuple[str, list[str]]:
    """
    Read a paper's title and author list from its first page.

    Returns (title, authors). Either may be empty if the layout defeats us -
    this is a best-effort heuristic over the one region of a paper that is
    reliably formatted the same way across arXiv preprints.

    False authors are worse than missing ones here: each becomes a graph node
    with `authored` edges into the paper, poisoning exactly the multi-hop author
    chains this system exists to answer. So the filters below err toward
    rejecting.
    """
    if not doc.get("pages"):
        return "", []

    # Scan a generous window of the first page rather than "everything before
    # the abstract". Three different real layouts break the narrower rule:
    #   faster_rcnn  title wraps across two lines, authors on the third
    #   attention    one author per line, interleaved with affiliation lines
    #   vision_transformer  two-column reading order puts the whole author block
    #                       AFTER the abstract and introduction
    lines = [
        line.strip() for line in doc["pages"][0].split("\n")
        if line.strip() and not line.strip().lower().startswith("arxiv:")
    ]
    if not lines:
        return "", []

    # Title: the first line, plus a continuation line if the title clearly wraps
    # ("Faster R-CNN: Towards Real-Time Object" / "Detection with Region
    # Proposal Networks"). A continuation is short, has no author names in it,
    # and does not end the sentence.
    title = lines[0].strip(" .,")
    if len(lines) > 1:
        following = lines[1].strip()
        is_section_heading = (
            len(following.split()) == 1
            or following.lower().strip(" .:") in _SECTION_HEADINGS
        )
        if (len(following) < 70 and not is_section_heading
                and not _AUTHOR_NAME_RE.search(following)):
            title = f"{title} {following}".strip(" .,")

    # The author block is CONTIGUOUS: names appear together near the top and
    # then stop. Everything that looked like a name further down the page was
    # prose ("Index Terms", "Selective Search", "Long Beach" from a venue line).
    # So we stop once names run out - but tolerate a two-line gap, because
    # Attention Is All You Need interleaves an affiliation line between every
    # pair of author lines.
    # Body text used to test whether a candidate is really a recurring technical
    # phrase. Skip page one - the author block itself lives there.
    body = "\n".join(doc["pages"][1:6]) if len(doc.get("pages", [])) > 1 else ""

    authors: list[str] = []
    gap = 0
    for line in lines[1:16]:
        cleaned = re.sub(r"\{[^}]*\}", " ", line)          # {a,b}@microsoft.com
        cleaned = re.sub(r"\S+@\S+", " ", cleaned)         # bare emails
        cleaned = re.sub(r"[∗†‡*¹²³]", " ", cleaned)

        # Truncate the line at the first affiliation word, and scan only what
        # comes before it. Author names precede their affiliation, never follow
        # it, so everything after that word is institutional.
        #
        # Rejecting the whole line instead is too blunt - real papers put both on
        # one line ("Kaiming He Xiangyu Zhang Shaoqing Ren Jian Sun Microsoft
        # Research"), and discarding it loses four real authors. Checking only
        # the matched pair is too lenient - "Max Planck Institute for Molecular
        # Genetics" rejects "Max Planck" and then returns "Molecular Genetics"
        # as an author. Truncation handles both.
        words = cleaned.split()
        for position, word in enumerate(words):
            if word.lower().strip(".,") in _AFFILIATION_WORDS:
                cleaned = " ".join(words[:position])
                break

        found_here = 0
        for match in _AUTHOR_NAME_RE.finditer(cleaned):
            name = re.sub(r"\s+", " ", match.group(0)).strip()
            words = {word.lower().strip(".") for word in name.split()}
            if words & _AFFILIATION_WORDS or words & _NON_NAME_WORDS:
                continue                   # "Microsoft Research", "Attention Is"
            # Domain-neutral check: a phrase that recurs through the body is
            # terminology, not a person. This is what keeps "Gene Expression"
            # and "Monetary Policy" out of the graph on corpora the blocklists
            # above know nothing about.
            if body and _looks_like_body_phrase(name, body):
                continue
            found_here += 1
            if name not in authors:
                authors.append(name)
            if len(authors) >= max_authors:
                return title, authors

        if found_here:
            gap = 0
        elif authors:
            gap += 1
            # Tolerate three name-free lines, not two: Attention Is All You Need
            # puts TWO affiliation lines between consecutive author lines, so a
            # tighter gap stopped after the first two of its eight authors.
            if gap >= 3:
                break

    return title, authors


def build_metadata_triples(docs: list[dict], triples: list[dict],
                           top_entities: int = 6) -> list[dict]:
    """
    Turn each paper's title/author block into triples, and attach the paper node
    to the entities that paper is actually about.

    The "presents" edges are what connect this metadata to the rest of the graph.
    Without them the authors would sit in their own little island, connected to a
    title node and nothing else - technically extracted, but unreachable from any
    question about a model.
    """
    metadata_triples: list[dict] = []

    # Which entities does each paper talk about most? Reuse the triples we
    # already extracted rather than re-reading the text.
    mentions: dict[str, Counter] = {}
    for triple in triples:
        counter = mentions.setdefault(triple["source_file"], Counter())
        counter[triple["subject"]] += 1
        counter[triple["object"]] += 1

    for doc in docs:
        title, authors = extract_paper_metadata(doc)
        if not title:
            continue

        sentence = f"{title}. Authors: {', '.join(authors)}." if authors else title

        for author in authors:
            metadata_triples.append({
                "subject": author, "relation": "authored", "object": title,
                "source_file": doc["filename"], "page": 1,
                "sentence": sentence, "extractor": "metadata",
            })

        for entity, _count in mentions.get(doc["filename"], Counter()).most_common(top_entities):
            if entity.strip().lower() == title.strip().lower():
                continue
            metadata_triples.append({
                "subject": title, "relation": "presents", "object": entity,
                "source_file": doc["filename"], "page": 1,
                "sentence": sentence, "extractor": "metadata",
            })

    return metadata_triples


def split_into_sentences(docs: list[dict], nlp) -> list[dict]:
    """
    Split every document into sentences, keeping the page each one came from.

    Why spaCy rather than text.split("."): research prose is full of periods that
    do not end sentences - "et al.", "Fig. 3", "Sec. 4.2", "0.5 mAP", "R-CNN".
    Naive splitting shatters those into fragments, and a fragment fed to REBEL
    produces either nothing or nonsense.
    """
    from src.chunker import build_text_with_page_map, page_for_offset

    sentences: list[dict] = []
    for doc in docs:
        text, page_spans = build_text_with_page_map(doc)
        if not text.strip():
            continue

        parsed = nlp(text)
        for index, span in enumerate(parsed.sents):
            clean = " ".join(span.text.split())
            if not is_useful_sentence(clean):
                continue
            sentences.append({
                "text": clean,
                "source_file": doc["filename"],
                "page": page_for_offset(span.start_char, page_spans),
                "sent_index": index,
            })
    return sentences


def get_nlp():
    """
    The shared spaCy pipeline, configured once.

    Keep the parser AND the lemmatizer: extract_domain_triples looks verbs up by
    lemma, so "outperforms"/"outperformed"/"outperform" all map to one relation.
    Disabling the lemmatizer for speed once made lemma_ unreliable and silently
    produced ZERO domain triples - the pipeline ran clean and found nothing.
    NER stays off: it is unreliable on this text (notebooks/02_ner_demo.py).
    """
    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    nlp.max_length = 3_000_000
    return nlp


def extract_document(doc: dict, nlp, rebel=None, batch_size: int = 8) -> list[dict]:
    """
    Run extraction over ONE cleaned document and return its triples.

    This is the unit the whole incremental design rests on. Everything it needs is
    inside `doc`, because document cleaning has no cross-document state -
    find_repeating_lines compares pages within a single document, and
    find_reference_span works on that document's own text. So extracting one
    document in isolation is equivalent to extracting it as part of a corpus.

    `rebel` is an optional loaded RelationExtractor. Passing None runs the fast
    path only (dependency parsing plus title-block metadata), which is what the
    UI does: at ~50 documents REBEL is five to six hours and the fast path is
    about two minutes.
    """
    sentences = split_into_sentences([doc], nlp)
    if not sentences:
        return []

    triples: list[dict] = []

    # -- fast path: dependency parsing ------------------------------------
    texts = [item["text"] for item in sentences]
    for item, parsed in zip(sentences, nlp.pipe(texts, batch_size=64)):
        for triple in extract_domain_triples(parsed):
            triples.append({
                **triple,
                "source_file": item["source_file"],
                "page": item["page"],
                "sentence": item["text"],
                "extractor": "domain",
            })

    # -- slow path: REBEL, only when a model was handed in -----------------
    if rebel is not None:
        for start in range(0, len(sentences), batch_size):
            batch = sentences[start:start + batch_size]
            for item, found in zip(batch, rebel.extract_batch(
                    [s["text"] for s in batch], batch_size=batch_size)):
                for triple in found:
                    triples.append({
                        **triple,
                        "source_file": item["source_file"],
                        "page": item["page"],
                        "sentence": item["text"],
                        "extractor": "rebel",
                    })

    # -- title block: paper node and authorship ----------------------------
    triples.extend(build_metadata_triples([doc], triples))
    return triples


def ingest_documents(doc_ids: list[str] | None = None, use_rebel: bool = False,
                     progress=None) -> dict:
    """
    Process workspace documents that need it, one document at a time.

    `progress(index, total, filename, stage)` is an optional callback so the UI can
    report "document 12 of 50" instead of sitting silent through one long call.

    Returns a summary dict. The graph is NOT rebuilt here - the caller does that
    once, after all documents are in, because graph construction is the one stage
    that is genuinely whole-corpus.
    """
    from src.chunker import chunk_fixed
    from src.ingestion import clean_document, extract_pdf
    from src.vector_store import VectorStore
    from src import workspace

    manifest = workspace.load_manifest()
    if doc_ids is None:
        doc_ids = [
            doc_id for doc_id, record in manifest.items()
            if record.get("status") != "processed"
            or (use_rebel and workspace.REBEL not in record.get("extractors_run", []))
        ]
    if not doc_ids:
        return {"processed": 0, "failed": 0, "chunks": 0, "triples": 0}

    nlp = get_nlp()
    store = VectorStore()

    rebel = None
    if use_rebel:
        from src.relation_extractor import RelationExtractor
        rebel = RelationExtractor()

    processed = failed = total_chunks = total_triples = 0

    for index, doc_id in enumerate(doc_ids, start=1):
        record = manifest.get(doc_id)
        if record is None:
            continue
        filename = record["filename"]
        if progress:
            progress(index, len(doc_ids), filename, "reading")

        try:
            raw = extract_pdf(RAW_DIR / filename)
            if raw is None:
                raise ValueError("could not extract text (scanned or encrypted?)")
            doc = clean_document(raw)

            if progress:
                progress(index, len(doc_ids), filename, "embedding")
            chunks = chunk_fixed(doc)
            # Delete first: a re-processed document may yield FEWER chunks than
            # before, and upsert only overwrites the ids it is handed, so the old
            # high-numbered ids would survive as orphans pointing at stale text.
            store.delete_by_source(filename)
            store.add_chunks(chunks, show_progress=False)

            if progress:
                progress(index, len(doc_ids), filename,
                         "extracting relations (REBEL)" if rebel else "extracting relations")
            triples = extract_document(doc, nlp, rebel=rebel)
            workspace.save_document_triples(doc_id, triples)

            extractors = [workspace.DOMAIN] + ([workspace.REBEL] if rebel else [])
            workspace.update_document(
                doc_id,
                n_pages=len(doc.get("pages", [])),
                n_chunks=len(chunks),
                n_triples=len(triples),
                extractors_run=sorted(set(record.get("extractors_run", []) + extractors)),
                status="processed",
                error="",
            )
            processed += 1
            total_chunks += len(chunks)
            total_triples += len(triples)

        except Exception as exc:  # noqa: BLE001 - one bad PDF must not stop the batch
            workspace.update_document(doc_id, status="failed", error=str(exc)[:300])
            failed += 1
            if progress:
                progress(index, len(doc_ids), filename, f"FAILED: {str(exc)[:80]}")

    return {"processed": processed, "failed": failed,
            "chunks": total_chunks, "triples": total_triples}


def extract_all(
    limit: int | None = None,
    batch_size: int = 8,
    num_beams: int = 3,
    use_rebel: bool = True,
    use_domain: bool = True,
    resume: bool = True,
) -> list[dict]:
    """Run the whole extraction pipeline and return the list of triple records."""
    import spacy
    from tqdm import tqdm

    from src.ingestion import load_corpus
    from src.relation_extractor import RelationExtractor, extract_domain_triples

    ensure_dirs()

    print("Loading and cleaning corpus ...")
    docs = load_corpus(clean=True)
    if not docs:
        raise SystemExit("No documents. Run: python scripts/fetch_papers.py")

    # Keep the parser (dependency labels) AND the lemmatizer: extract_domain_triples
    # looks verbs up by lemma, so "outperforms"/"outperformed"/"outperform" all map
    # to one relation. Disabling the lemmatizer for speed made lemma_ unreliable and
    # silently produced ZERO domain triples - the pipeline ran clean and found
    # nothing, which is the worst kind of bug.
    # NER stays off: it is unreliable on this text (notebooks/02_ner_demo.py) and
    # nothing here uses it.
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    nlp.max_length = 3_000_000

    print("Splitting into sentences ...")
    sentences = split_into_sentences(docs, nlp)
    print(f"  {len(sentences):,} usable sentences from {len(docs)} documents")

    if limit:
        sentences = sentences[:limit]
        print(f"  limited to {len(sentences):,} for this run")

    triples: list[dict] = []
    start_index = 0

    # ---- resume from a checkpoint if one exists -------------------------
    if resume and _CHECKPOINT_PATH.exists():
        checkpoint = json.loads(_CHECKPOINT_PATH.read_text(encoding="utf-8"))
        if checkpoint.get("total_sentences") == len(sentences):
            triples = checkpoint["triples"]
            start_index = checkpoint["completed"]
            print(f"  resuming from checkpoint: {start_index:,}/{len(sentences):,} "
                  f"sentences already done, {len(triples):,} triples so far")
        else:
            print("  checkpoint is for a different corpus - starting fresh")

    # ---- the domain extractor (fast: pure spaCy, no neural generation) ---
    if use_domain and start_index == 0:
        print("\nRunning the domain relation extractor (dependency parsing) ...")
        started = time.time()
        texts = [s["text"] for s in sentences]
        domain_count = 0
        for sentence, parsed in zip(
            sentences,
            tqdm(nlp.pipe(texts, batch_size=64), total=len(texts),
                 desc="  domain", unit="sent"),
        ):
            for triple in extract_domain_triples(parsed):
                triples.append({
                    **triple,
                    "source_file": sentence["source_file"],
                    "page": sentence["page"],
                    "sentence": sentence["text"],
                    "extractor": "domain",
                })
                domain_count += 1
        print(f"  {domain_count:,} domain triples in {time.time() - started:.0f}s")

    # ---- REBEL (slow: neural seq2seq generation) ------------------------
    if use_rebel:
        extractor = RelationExtractor(num_beams=num_beams, num_threads=8)
        extractor._load()

        print(f"\nRunning REBEL over {len(sentences) - start_index:,} sentences "
              f"(batch_size={batch_size}, num_beams={num_beams}) ...")
        started = time.time()

        remaining = sentences[start_index:]
        progress = tqdm(total=len(remaining), desc="  REBEL", unit="sent")

        for offset in range(0, len(remaining), batch_size):
            batch = remaining[offset:offset + batch_size]
            batch_triples = extractor.extract_batch([s["text"] for s in batch],
                                                    batch_size=batch_size)
            for sentence, found in zip(batch, batch_triples):
                for triple in found:
                    triples.append({
                        **triple,
                        "source_file": sentence["source_file"],
                        "page": sentence["page"],
                        "sentence": sentence["text"],
                        "extractor": "rebel",
                    })
            progress.update(len(batch))

            completed = start_index + offset + len(batch)
            if completed % CHECKPOINT_EVERY < batch_size:
                _save_checkpoint(triples, completed, len(sentences))

        progress.close()
        elapsed = time.time() - started
        per_sentence = elapsed / max(len(remaining), 1)
        print(f"  REBEL finished in {elapsed / 60:.1f} min ({per_sentence:.2f}s/sentence)")

    # ---- paper titles and authors (cheap, and runs last so the "presents"
    # edges can be aimed at whichever entities the extractors actually found)
    metadata = build_metadata_triples(docs, triples)
    triples.extend(metadata)
    print(f"\n  {len(metadata):,} metadata triples (paper titles, authorship)")

    return triples


def _save_checkpoint(triples: list[dict], completed: int, total: int) -> None:
    _CHECKPOINT_PATH.write_text(
        json.dumps({"completed": completed, "total_sentences": total, "triples": triples}),
        encoding="utf-8",
    )


def save_triples(triples: list[dict], path: Path = TRIPLES_PATH) -> None:
    ensure_dirs()
    path.write_text(json.dumps(triples, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {len(triples):,} triples to {path}")
    if _CHECKPOINT_PATH.exists():
        _CHECKPOINT_PATH.unlink()          # the real file exists now


def load_triples(path: Path = TRIPLES_PATH) -> list[dict]:
    """
    Every triple in the workspace.

    Prefers the per-document files; falls back to the legacy single triples.json
    so a workspace built before the split keeps working.

    Returns [] for an absent file rather than raising. It used to `raise
    SystemExit`, which crashed the very first upload on a clean install - the
    upload handler called this to merge, and on a fresh workspace there is by
    definition nothing to merge with. An empty workspace is a normal state, not
    an error.
    """
    from src.workspace import load_all_triples

    per_document = load_all_triples()
    if per_document:
        return per_document
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def print_stats(triples: list[dict]) -> None:
    if not triples:
        print("No triples extracted.")
        return

    print(banner("EXTRACTION STATS"))
    print(f"  total triples: {len(triples):,}")

    by_extractor = Counter(t["extractor"] for t in triples)
    print(f"  by extractor:  {dict(by_extractor)}")

    by_file = Counter(t["source_file"] for t in triples)
    print("\n  per document:")
    for filename, count in by_file.most_common():
        print(f"     {filename:<34} {count:>6}")

    print(banner("MOST COMMON RELATION TYPES"))
    print(f"  {'relation':<28}{'count':>8}   extractor")
    print("  " + "-" * 52)
    relation_source: dict[str, Counter] = {}
    for triple in triples:
        relation_source.setdefault(triple["relation"], Counter())[triple["extractor"]] += 1
    for relation, count in Counter(t["relation"] for t in triples).most_common(25):
        sources = "+".join(sorted(relation_source[relation]))
        print(f"  {relation:<28}{count:>8}   {sources}")

    print(banner("MOST CONNECTED ENTITIES"))
    entities = Counter()
    for triple in triples:
        entities[triple["subject"]] += 1
        entities[triple["object"]] += 1
    print(f"  {len(entities):,} distinct entity strings (before Phase 4 deduplication)\n")
    for entity, count in entities.most_common(25):
        print(f"     {count:>5}  {entity}")

    print(banner("SAMPLE TRIPLES WITH PROVENANCE"))
    for triple in triples[:4]:
        print(f"\n  {triple['subject']}  --[{triple['relation']}]-->  {triple['object']}")
        print(f"     via {triple['extractor']}, {triple['source_file']} p.{triple['page']}")
        print(f"     \"{triple['sentence'][:130]}...\"")


def run_rebel_batch(num_beams: int = 3, batch_size: int = 8) -> None:
    """
    The overnight job: run REBEL over every document that has not had it yet.

    This is the slow half of the two-speed design. Uploading a document runs the
    fast dependency-parse path (about two minutes for fifty papers) so the
    workspace is queryable immediately; this adds REBEL's taxonomic relations
    (subclass of, part of, use) which are what connect the graph into one
    traversable component. At ~1.3 s/sentence it is five to six hours for fifty
    papers, which is why it lives here and not in a browser tab.

    Resumable at DOCUMENT granularity. Each document's triples are written as
    soon as it finishes, and the manifest records that REBEL ran on it - so an
    interrupted job resumes at the next unprocessed document. The mechanism this
    replaced stored a flat index into a corpus-wide sentence list validated by an
    exact sentence-count match, so adding a single PDF invalidated the checkpoint
    and restarted the whole job from zero.
    """
    from src import workspace
    from src.relation_extractor import RelationExtractor

    pending = workspace.documents_missing(workspace.REBEL)
    if not pending:
        total = len(workspace.load_manifest())
        print(f"Nothing to do - REBEL has run on all {total} document(s).")
        return

    print(banner("REBEL BATCH EXTRACTION"))
    print(f"  {len(pending)} document(s) to process.")
    print("  ~1.3s per sentence on CPU; a typical paper is ~300 sentences.")
    print("  Safe to interrupt - progress is saved after each document.\n")

    nlp = get_nlp()
    rebel = RelationExtractor(num_beams=num_beams)
    started = time.time()

    for index, record in enumerate(pending, start=1):
        doc_id, filename = record["doc_id"], record["filename"]
        print(f"[{index}/{len(pending)}] {filename} ...", flush=True)

        try:
            from src.ingestion import clean_document, extract_pdf

            raw = extract_pdf(RAW_DIR / filename)
            if raw is None:
                raise ValueError("could not extract text")
            doc = clean_document(raw)

            # Replace this document's triples wholesale: the new run includes
            # both the fast-path relations and REBEL's, so keeping the old file
            # would duplicate every domain triple.
            triples = extract_document(doc, nlp, rebel=rebel, batch_size=batch_size)
            workspace.save_document_triples(doc_id, triples)
            workspace.update_document(
                doc_id, n_triples=len(triples), status="processed", error="",
                extractors_run=sorted(set(record.get("extractors_run", [])
                                          + [workspace.DOMAIN, workspace.REBEL])),
            )
            print(f"    {len(triples):,} triples  ({(time.time() - started) / 60:.1f} min elapsed)")
        except Exception as exc:  # noqa: BLE001 - one bad document must not end the night
            workspace.update_document(doc_id, status="failed", error=str(exc)[:300])
            print(f"    FAILED: {str(exc)[:120]}")

    print("\nRebuilding knowledge graph ...")
    graph = workspace.rebuild_graph(verbose=True)
    print(f"Done in {(time.time() - started) / 60:.1f} min. "
          f"Graph: {graph.graph.number_of_nodes():,} entities, "
          f"{graph.graph.number_of_edges():,} relations.")


def main() -> None:
    setup_console()

    parser = argparse.ArgumentParser(description="Batch relation extraction over the corpus")
    parser.add_argument("--rebel", action="store_true",
                        help="run REBEL over every workspace document that has not had it "
                             "yet, one document at a time. Resumable; run it overnight.")
    parser.add_argument("--limit", type=int, help="only process the first N sentences")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=3,
                        help="more beams = more triples but slower (3 is a good default)")
    parser.add_argument("--no-rebel", action="store_true",
                        help="skip REBEL, run only the fast domain extractor")
    parser.add_argument("--no-domain", action="store_true")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore any existing checkpoint")
    parser.add_argument("--stats-only", action="store_true",
                        help="just print stats for an existing triples.json")
    args = parser.parse_args()

    if args.stats_only:
        print_stats(load_triples())
        return

    if args.rebel:
        run_rebel_batch(num_beams=args.num_beams, batch_size=args.batch_size)
        print_stats(load_triples())
        return

    triples = extract_all(
        limit=args.limit,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        use_rebel=not args.no_rebel,
        use_domain=not args.no_domain,
        resume=not args.no_resume,
    )
    save_triples(triples)
    print_stats(triples)


if __name__ == "__main__":
    main()
