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
from src.utils import PROCESSED_DIR, TRIPLES_PATH, banner, ensure_dirs, setup_console  # noqa: E402

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
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python src/knowledge_extractor.py")
    return json.loads(path.read_text(encoding="utf-8"))


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


def main() -> None:
    setup_console()

    parser = argparse.ArgumentParser(description="Batch relation extraction over the corpus")
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
