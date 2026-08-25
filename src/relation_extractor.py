"""
Phase 3, Step 3.2 - REBEL: turning sentences into (subject, relation, object) triples.

WHAT IS A TRIPLE?
-----------------
The atomic unit of a knowledge graph. Three parts:

    (DETR)  --[outperforms]-->  (Faster R-CNN)
    subject      relation           object

Subject and object become NODES. The relation becomes a labelled, DIRECTED EDGE.
Direction is not decoration - "DETR outperforms Faster R-CNN" and "Faster R-CNN
outperforms DETR" use identical words and mean opposite things.

HOW IS THIS DIFFERENT FROM NER?
-------------------------------
NER (Step 3.1) finds the things:      DETR, Faster R-CNN, COCO
REBEL finds how they are connected:   DETR --outperforms--> Faster R-CNN

Nodes without edges cannot be traversed, and traversal is the whole reason this
project exists. NER gives you a pile of nouns; REBEL gives you a graph.

WHAT REBEL ACTUALLY IS, INTERNALLY
----------------------------------
This surprises people: REBEL is not a classifier. It is a BART sequence-to-
sequence model - the same architecture used for summarisation and translation -
fine-tuned to "translate" English into a made-up markup language that encodes
triples:

    input:  "DETR outperforms Faster R-CNN on COCO"
    output: "<triplet> DETR <subj> Faster R-CNN <obj> outperforms"

So relation extraction is reframed as translation. The advantages: one model
emits any number of triples per sentence, handles overlapping relations, and
needs no pre-computed entity spans. The cost: the output is free-form generated
text that we have to PARSE, and a malformed generation yields nothing.

Everything in _parse_generated() below exists to decode that markup.

COST ON CPU
-----------
rebel-large is ~1.6GB (400M parameters) and does beam-search generation. Expect
roughly 0.5-2 seconds per sentence on this hardware. That is fine for a batch
job and unusable interactively - which is exactly why Step 3.3 runs it ONCE over
the corpus and saves the output to triples.json. Nothing at query time ever
calls REBEL.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import banner, setup_console  # noqa: E402

MODEL_NAME = "Babelscape/rebel-large"

# REBEL's markup vocabulary. These MUST survive decoding - see the note in
# _generate() about skip_special_tokens.
TRIPLET_TOKEN = "<triplet>"
SUBJECT_TOKEN = "<subj>"
OBJECT_TOKEN = "<obj>"


def _parse_generated(generated: str) -> list[dict]:
    """
    Decode REBEL's linearised markup into triples.

    The format is a flat token stream with three markers:

        <triplet> SUBJECT <subj> OBJECT <obj> RELATION
                  <subj> OTHER_OBJECT <obj> OTHER_RELATION      <- same subject reused
        <triplet> NEW_SUBJECT <subj> ...                        <- new subject

    So a single <triplet> block can carry several (object, relation) pairs that
    all share one subject. The parser is a small state machine: each marker
    switches which field the following tokens are appended to, and a completed
    triple is emitted whenever we hit the next marker.

    This function is adapted from the reference decoder on the model card; the
    format is not documented anywhere else, and getting it subtly wrong yields
    plausible-looking but scrambled triples.
    """
    triples: list[dict] = []
    subject = relation = obj = ""
    current = None

    # The tokenizer emits these around the real content; they carry no meaning.
    cleaned = (
        generated.replace("<s>", "")
        .replace("</s>", "")
        .replace("<pad>", "")
        .strip()
    )

    def flush() -> None:
        if subject.strip() and relation.strip() and obj.strip():
            triples.append({
                "subject": subject.strip(),
                "relation": relation.strip(),
                "object": obj.strip(),
            })

    for token in cleaned.split():
        if token == TRIPLET_TOKEN:
            flush()                       # close the previous triple, if any
            current = "subject"
            subject = relation = obj = ""
        elif token == SUBJECT_TOKEN:
            flush()                       # same subject, new (object, relation) pair
            current = "object"
            obj = relation = ""
        elif token == OBJECT_TOKEN:
            current = "relation"
            relation = ""
        elif current == "subject":
            subject += " " + token
        elif current == "object":
            obj += " " + token
        elif current == "relation":
            relation += " " + token

    flush()                               # the final triple has no marker after it

    # Deduplicate while preserving order - beam search often returns the same
    # triple from several beams.
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict] = []
    for triple in triples:
        key = (triple["subject"].lower(), triple["relation"].lower(), triple["object"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(triple)
    return unique


class RelationExtractor:
    """Wraps rebel-large. Loads lazily so importing this module stays cheap."""

    def __init__(self, model_name: str = MODEL_NAME, num_beams: int = 3,
                 max_length: int = 256, num_threads: int | None = None) -> None:
        self.model_name = model_name
        self.num_beams = num_beams
        self.max_length = max_length
        self.num_threads = num_threads
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        # Use all physical cores. torch defaults conservatively in some builds,
        # and this is a pure-CPU workload where threads are the only lever.
        if self.num_threads:
            torch.set_num_threads(self.num_threads)

        print(f"Loading {self.model_name} (~1.6GB, first run downloads it) ...")
        started = time.time()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self._model.eval()                 # no dropout, no gradient bookkeeping
        print(f"  loaded in {time.time() - started:.1f}s")

    @property
    def tokenizer(self):
        self._load()
        return self._tokenizer

    @property
    def model(self):
        self._load()
        return self._model

    def _generate(self, sentences: list[str]) -> list[str]:
        """Run the model over a batch and return raw generated markup strings."""
        import torch

        inputs = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )

        with torch.no_grad():              # inference only - halves memory use
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=self.num_beams,
                # Return every beam, not just the best one. Different beams often
                # surface different valid triples from the same sentence, so this
                # meaningfully increases recall for free (the beams were computed
                # either way).
                num_return_sequences=self.num_beams,
                early_stopping=True,
            )

        # CRITICAL: skip_special_tokens=False.
        # <triplet>, <subj> and <obj> are registered as special tokens. Decoding
        # with the usual skip_special_tokens=True strips exactly the markers the
        # parser needs, and every sentence silently yields zero triples.
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=False)

    def extract(self, sentence: str) -> list[dict]:
        """Extract triples from one sentence. Returns [] when REBEL finds nothing."""
        return self.extract_batch([sentence])[0]

    def extract_batch(self, sentences: list[str], batch_size: int = 8) -> list[list[dict]]:
        """
        Extract from many sentences at once.

        Batching matters a lot here: the per-call overhead (tokenising, kicking
        off the graph, beam bookkeeping) is amortised across the batch, and
        matrix multiplies over a padded batch use the CPU far better than
        one-at-a-time calls. Keep batches modest, though - padding to the longest
        sentence in the batch means one 100-word outlier makes every other
        sentence in that batch pay for its length.
        """
        if not sentences:
            return []

        self._load()
        results: list[list[dict]] = []

        for start in range(0, len(sentences), batch_size):
            batch = sentences[start:start + batch_size]
            decoded = self._generate(batch)

            # generate() returned num_beams sequences per input, flattened.
            # Regroup them so beam outputs stay attached to their sentence.
            for i in range(len(batch)):
                window = decoded[i * self.num_beams:(i + 1) * self.num_beams]
                merged: list[dict] = []
                seen: set[tuple[str, str, str]] = set()
                for generated in window:
                    for triple in _parse_generated(generated):
                        key = (triple["subject"].lower(), triple["relation"].lower(),
                               triple["object"].lower())
                        if key not in seen:
                            seen.add(key)
                            merged.append(triple)
                results.append(merged)

        return results


# ---------------------------------------------------------------------------
# The domain relation extractor - REBEL's blind spot, covered
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS (read before assuming it is over-engineering)
#
# REBEL was trained on Wikipedia aligned to WIKIDATA properties. Its relation
# labels are drawn from that closed vocabulary: "instance of", "part of",
# "developer", "use", "followed by". Run this module's main block and read the
# vocabulary it reports - "outperforms" is not in it and never will be, because
# Wikidata has no such property.
#
# That is a problem specific to this project. The motivating question in
# PROJECT_CONTEXT.md is:
#
#     "Which YOLO variants were outperformed by Vision Transformer models?"
#
# Answering it requires walking an "outperforms" edge. If no such edge is ever
# created, the graph cannot answer the question that justifies the graph. REBEL
# alone gets us a graph of "part of" and "instance of" - taxonomically tidy and
# useless for comparative reasoning.
#
# So we run a SECOND extractor over the same sentences, targeting the handful of
# relations that actually carry meaning in a CV paper: who beat whom, what scored
# what, what was trained on what, who proposed what.
#
# WHY DEPENDENCY PARSING RATHER THAN REGEX
# A regex for "outperforms" matches the word but cannot tell you WHICH noun is
# doing the outperforming. In "unlike SSD, which YOLOv3 outperforms, RetinaNet
# is slower", a regex sees three model names and one verb. spaCy's parser gives
# the grammatical subject and object directly, so direction comes out right -
# and direction is the entire value of a directed edge.

# verb lemma -> canonical relation name. Mapping many surface forms onto one
# label is what makes the graph traversable: a query for "outperforms" should
# find edges created from "surpasses" and "beats" too.
_VERB_RELATIONS: dict[str, str] = {
    "outperform": "outperforms",
    "surpass": "outperforms",
    "beat": "outperforms",
    "exceed": "outperforms",
    "outdo": "outperforms",
    "improve": "improves_on",
    "extend": "extends",
    "build": "based_on",
    "achieve": "achieves",
    "obtain": "achieves",
    "attain": "achieves",
    "reach": "achieves",
    "report": "reports",
    "use": "uses",
    "employ": "uses",
    "adopt": "uses",
    "leverage": "uses",
    "utilize": "uses",
    "apply": "uses",
    "introduce": "introduces",
    "propose": "proposes",
    "present": "proposes",
    "train": "trained_on",
    "evaluate": "evaluated_on",
    "test": "evaluated_on",
    "benchmark": "evaluated_on",
    "compare": "compared_to",
    "combine": "combines",
    "replace": "replaces",
    "outweigh": "outperforms",
}

# Comparative adjectives: "X is more accurate THAN Y", "X is faster THAN Y".
_COMPARATIVE_ADJECTIVES = {
    "better", "faster", "stronger", "superior", "higher", "greater",
    "more", "accurate", "efficient", "worse", "slower", "lower",
}

# Prepositions that carry the object for particular relations.
_RELATION_PREPOSITIONS = {
    # "on" ONLY. Allowing "with"/"using" here conflated two different facts:
    #   "trained ON ImageNet"  -> a dataset      (what we want)
    #   "trained WITH SGD"     -> an optimiser   (not a dataset at all)
    # Both became trained_on edges, so a question about training data could
    # traverse into optimiser trivia. Those now get their own relation.
    "trained_on": {"on"},
    "trained_with": {"with", "using"},
    "evaluated_on": {"on", "against"},
    "compared_to": {"to", "with", "against"},
    "based_on": {"on", "upon"},
    "improves_on": {"on", "upon", "over"},
}


def _clean_span(text: str) -> str:
    """Trim determiners and whitespace so 'the DETR model' becomes 'DETR model'."""
    words = text.strip().split()
    while words and words[0].lower() in {"the", "a", "an", "our", "their", "its", "this", "these"}:
        words = words[1:]
    return " ".join(words).strip(" ,;:.")


def _extend_with_et_al(token, doc, text: str) -> str:
    """
    Reattach a truncated "et al." to an author name.

    spaCy parses "He et al. introduced ..." with "He" as the subject and "et"/"al"
    as separate tokens, so the noun chunk is just "He" - which then becomes a
    graph node indistinguishable from the pronoun "he". Author attribution is one
    of the relations this project most wants ("who wrote the paper that..."), so
    the citation form is worth reconstructing.
    """
    # Already carries the citation form (spaCy sometimes chunks it in) - just
    # normalise the trailing period so "Ren et al" and "Ren et al." do not become
    # two separate graph nodes.
    stripped = text.rstrip(". ")
    if stripped.lower().endswith("et al"):
        return stripped + "."

    following = doc[token.i + 1: token.i + 4]
    words = [t.text.lower().strip(".") for t in following]
    if words[:2] == ["et", "al"]:
        return f"{text} et al."
    return text


def _noun_span(token, doc) -> str:
    """
    Expand a token to the noun phrase it belongs to.

    Using the bare token gives "R-CNN" where we want "Faster R-CNN"; using the
    whole subtree gives "Faster R-CNN on the COCO dataset with ResNet backbone",
    which is a sentence fragment, not an entity. Noun chunks are the right
    granularity, with a compound-walk fallback for tokens spaCy did not chunk.
    """
    span_text = None
    end_token = token
    for chunk in doc.noun_chunks:
        if chunk.start <= token.i < chunk.end:
            span_text = _clean_span(chunk.text)
            end_token = doc[chunk.end - 1]
            break

    if span_text is None:
        parts = [child.text for child in token.lefts if child.dep_ in {"compound", "amod"}]
        parts.append(token.text)
        span_text = _clean_span(" ".join(parts))

    return _extend_with_et_al(end_token, doc, span_text)


def _subject_of(verb, doc, follow_conjunction: bool = True) -> str | None:
    """
    Find the grammatical subject of a verb.

    `follow_conjunction` handles the shared-subject case. In

        "DETR uses a transformer architecture and outperforms Faster R-CNN"

    spaCy attaches "outperforms" to "uses" as a conj and does NOT give it its own
    nsubj - the subject is stated once and shared. Without this walk we lose the
    outperforms triple entirely, which is exactly the relation type this
    extractor exists to capture.
    """
    for child in verb.children:
        if child.dep_ in {"nsubj", "nsubjpass"}:
            return _noun_span(child, doc)

    if follow_conjunction and verb.dep_ == "conj":
        head = verb.head
        # Bounded walk - conjunction chains are short, and a cycle would hang.
        for _ in range(4):
            subject = _subject_of(head, doc, follow_conjunction=False)
            if subject:
                return subject
            if head.dep_ != "conj":
                break
            head = head.head
    return None


def _passive_agent(verb, doc) -> str | None:
    """The 'by Y' in 'X was proposed by Y'."""
    for child in verb.children:
        if child.dep_ == "agent":
            for grandchild in child.children:
                if grandchild.dep_ == "pobj":
                    return _noun_span(grandchild, doc)
    return None


def _prep_objects(anchors, allowed: set[str], doc) -> list[str]:
    """
    Collect prepositional objects hanging off any of `anchors`.

    Both the verb AND its direct object have to be searched. spaCy attaches the
    prepositional phrase to whichever it judges the modifier belongs to, and for

        "We evaluate Mask R-CNN on the COCO dataset"

    it attaches "on" to "CNN", not to "evaluate". Searching only the verb's
    children silently drops every evaluated_on/trained_on triple of that shape -
    which is most of them.
    """
    found: list[str] = []
    for anchor in anchors:
        for child in anchor.children:
            if child.dep_ == "prep" and child.text.lower() in allowed:
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        found.append(_noun_span(grandchild, doc))
    return found


def extract_domain_triples(doc) -> list[dict]:
    """
    Pull CV-specific relations out of one parsed spaCy sentence.

    `doc` is a spaCy Doc (or Span) that has been through the dependency parser.
    Returns the same {subject, relation, object} shape REBEL produces, so both
    sources flow into the graph builder unchanged.
    """
    triples: list[dict] = []

    def add(subject: str | None, relation: str, obj: str | None) -> None:
        if not subject or not obj:
            return
        if len(subject) < 2 or len(obj) < 2:
            return
        if subject.lower() == obj.lower():
            return
        # Pronoun subjects ("we", "it", "they") produce meaningless nodes.
        if subject.lower() in {"we", "it", "they", "this", "that", "these", "those", "i"}:
            return
        triples.append({"subject": subject, "relation": relation, "object": obj,
                        "extractor": "domain"})

    for token in doc:
        # --- verb-driven relations ---------------------------------------
        if token.pos_ in {"VERB", "AUX"}:
            relation = _VERB_RELATIONS.get(token.lemma_.lower())
            if relation:
                subject = _subject_of(token, doc)
                is_passive = any(c.dep_ == "auxpass" for c in token.children)

                # "Faster R-CNN was proposed by Ren et al."
                # -> (Ren et al.) proposes (Faster R-CNN), not the reverse.
                if is_passive:
                    agent = _passive_agent(token, doc)
                    if agent and subject:
                        add(agent, relation, subject)
                        continue

                direct_objects = [c for c in token.children if c.dep_ in {"dobj", "attr", "oprd"}]
                allowed = _RELATION_PREPOSITIONS.get(relation)

                # "We evaluate Mask R-CNN on COCO" -> (Mask R-CNN, evaluated_on, COCO).
                # The grammatical subject is the authors ("we"), which is a useless
                # graph node. For these relations the interesting subject is the
                # direct object, and the prepositional phrase is the real object.
                pronoun_subject = subject is None or subject.lower() in {
                    "we", "it", "they", "this", "that", "these", "those", "i", "he", "she",
                }
                if pronoun_subject and allowed and direct_objects:
                    inner_subject = _noun_span(direct_objects[0], doc)
                    for obj in _prep_objects([token] + direct_objects, allowed, doc):
                        add(inner_subject, relation, obj)
                    continue

                # direct object: "DETR outperforms Faster R-CNN"
                for child in direct_objects:
                    add(subject, relation, _noun_span(child, doc))

                # Prepositional objects ONLY for relations whose meaning lives in
                # the preposition ("trained ON COCO", "compared TO SSD"). Allowing
                # them everywhere produced junk like
                #     (Vaswani, proposes, 2017)          <- "in 2017"
                #     (ResNet-101, outperforms, ImageNet) <- "on ImageNet"
                # where the prepositional phrase is a modifier, not the object.
                if allowed:
                    for obj in _prep_objects([token] + direct_objects, allowed, doc):
                        add(subject, relation, obj)

                # One verb, two relations: "train" yields trained_on for "on X"
                # and trained_with for "with/using X", so a dataset and an
                # optimiser never end up behind the same edge label.
                if relation == "trained_on":
                    for obj in _prep_objects([token] + direct_objects,
                                             _RELATION_PREPOSITIONS["trained_with"], doc):
                        add(subject, "trained_with", obj)

        # --- comparative adjectives: "X is faster than Y" ------------------
        if token.pos_ in {"ADJ", "ADV"} and token.text.lower() in _COMPARATIVE_ADJECTIVES:
            head = token.head
            subject = _subject_of(head, doc) if head.pos_ in {"VERB", "AUX"} else None
            if subject is None:
                subject = _subject_of(token, doc)

            worse = token.text.lower() in {"worse", "slower", "lower"}
            relation = "underperforms" if worse else "outperforms"

            # spaCy parses "than X" two different ways depending on how it reads
            # the comparison, and both appear in one sentence of our test set:
            #   "faster than YOLOv3"       -> than=mark, YOLOv3=advcl child of "faster"
            #   "more accurate than SSD"   -> than=prep, SSD=pobj child of "than"
            # Handling only one silently drops half of all comparisons.
            for child in token.children:
                if child.dep_ == "prep" and child.text.lower() == "than":
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            add(subject, relation, _noun_span(grandchild, doc))
                elif child.dep_ in {"advcl", "npadvmod", "conj"}:
                    if any(g.dep_ == "mark" and g.text.lower() == "than"
                           for g in child.children):
                        add(subject, relation, _noun_span(child, doc))

    # Deduplicate.
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict] = []
    for triple in triples:
        key = (triple["subject"].lower(), triple["relation"], triple["object"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(triple)
    return unique


# ---------------------------------------------------------------------------
# Demo / benchmark
# ---------------------------------------------------------------------------

TEST_SENTENCES = [
    "DETR uses a transformer encoder-decoder architecture and outperforms Faster R-CNN on COCO.",
    "Vaswani and Shazeer at Google Brain proposed the Transformer architecture in 2017.",
    "Swin Transformer achieves 87.3 top-1 accuracy on ImageNet-1K.",
    "He et al. introduced deep residual learning at Microsoft Research.",
    "YOLOv3 is as accurate as SSD but three times faster.",
    "Faster R-CNN was proposed by Ren et al. and uses a region proposal network.",
    "The Vision Transformer applies a standard Transformer encoder directly to image patches.",
    "Mask R-CNN extends Faster R-CNN by adding a branch for predicting segmentation masks.",
    "ResNet won the ILSVRC 2015 classification competition.",
    "We trained our models on the COCO 2017 detection dataset.",
]


def main() -> None:
    setup_console()

    extractor = RelationExtractor(num_threads=8)

    # Load the model BEFORE starting the clock. On the very first run this step
    # downloads 1.6GB, which took 9 minutes here - timing that as if it were
    # inference reported 55s/sentence when the true figure is ~1.3s.
    load_started = time.time()
    extractor._load()
    print(f"  (model ready in {time.time() - load_started:.1f}s - not counted below)")

    print(banner("EXTRACTING TRIPLES FROM 10 SENTENCES"))
    started = time.time()
    all_triples = extractor.extract_batch(TEST_SENTENCES, batch_size=5)
    elapsed = time.time() - started

    total = 0
    empty = 0
    for sentence, triples in zip(TEST_SENTENCES, all_triples):
        print(f"\n  {sentence}")
        if not triples:
            print("     (no triples found)")
            empty += 1
            continue
        for triple in triples:
            print(f"     {triple['subject']}  --[{triple['relation']}]-->  {triple['object']}")
            total += 1

    print(banner("PERFORMANCE"))
    print(f"  {len(TEST_SENTENCES)} sentences in {elapsed:.1f}s "
          f"= {elapsed / len(TEST_SENTENCES):.2f}s per sentence")
    print(f"  {total} triples extracted, {empty} sentences yielded nothing")
    print(f"\n  Extrapolating: a 3,000-sentence corpus takes about "
          f"{elapsed / len(TEST_SENTENCES) * 3000 / 60:.0f} minutes.")
    print("  That is why Step 3.3 is a batch job that writes triples.json, and")
    print("  why nothing at query time ever calls this model.")

    # ---------------------------------------------------------------------
    # The relation vocabulary - read this before designing graph queries.
    # ---------------------------------------------------------------------
    relations = sorted({t["relation"] for triples in all_triples for t in triples})
    print(banner("THE RELATION VOCABULARY REBEL ACTUALLY USES"))
    print(f"  {relations}\n")
    print("  REBEL was trained on Wikipedia text aligned to WIKIDATA properties,")
    print("  so its relation labels come from that fixed vocabulary - things like")
    print("  'developer', 'instance of', 'part of', 'manufacturer'.")
    print("\n  It does NOT invent domain relations. If you were expecting")
    print("  'outperforms' or 'evaluated on', check the list above against what")
    print("  you actually got - this is the single most important thing to")
    print("  understand before Phase 4, because the graph can only be traversed")
    print("  along edges that actually exist.")


if __name__ == "__main__":
    main()
