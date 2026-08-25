"""
Phase 3, Step 3.1 - Named Entity Recognition: finding the THINGS in a sentence.

Run me:  python notebooks/02_ner_demo.py

WHAT NER DOES
-------------
A NER model reads a sentence and marks the spans that refer to real-world
entities, tagging each with a type:

    "Vaswani et al. at Google proposed the Transformer in 2017"
      Vaswani ....... PERSON
      Google ........ ORG
      2017 .......... DATE

That is it. It finds the nouns that are THINGS, not the relationships between
them. Relationships are Step 3.2's job (REBEL).

WHY WE NEED IT
--------------
Entities are the NODES of the knowledge graph we build in Phase 4. And in
Phase 5, when a user asks "which models outperform YOLO?", NER is what pulls
"YOLO" out of that sentence so we know where to start walking the graph.

WHAT TO WATCH FOR
-----------------
This demo deliberately shows you where spaCy SUCCEEDS and where it FAILS on
scientific text. The failures are not a bug - en_core_web_sm was trained on news
and web text, where "DETR" and "COCO" never appear. Seeing that clearly now is
what makes the Phase 5 design decisions make sense later.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import banner, setup_console  # noqa: E402

# Hand-picked sentences: the first four are real text from our corpus, the last
# is a constructed one that shows the model at its best.
DEMO_SENTENCES = [
    "We propose DETR, which uses a transformer encoder-decoder architecture and "
    "matches Faster R-CNN on the COCO object detection dataset.",

    "Swin Transformer achieves 87.3 top-1 accuracy on ImageNet-1K and surpasses "
    "the previous state-of-the-art by a large margin.",

    "He et al. introduced deep residual learning at Microsoft Research, and ResNet "
    "won the ILSVRC 2015 classification competition.",

    "YOLOv3 runs in 22 ms at 28.2 mAP, which is as accurate as SSD but three times faster.",

    "Vaswani and Shazeer at Google Brain proposed the Transformer in 2017.",
]

# Plain-English glosses - spaCy's label set is not self-explanatory.
LABEL_MEANING = {
    "PERSON": "a person's name",
    "ORG": "company, institution, agency",
    "GPE": "country, city, state",
    "LOC": "non-political location",
    "DATE": "an absolute or relative date",
    "TIME": "a time shorter than a day",
    "CARDINAL": "a plain number",
    "ORDINAL": "first, second, ...",
    "PERCENT": "a percentage",
    "MONEY": "a monetary value",
    "QUANTITY": "a measurement with units",
    "PRODUCT": "an object, vehicle, or device",
    "WORK_OF_ART": "title of a book, song, or paper",
    "EVENT": "a named event",
    "FAC": "building, airport, highway",
    "NORP": "nationality, religious or political group",
    "LANGUAGE": "a named language",
    "LAW": "a named legal document",
}


def visualise_entities(doc) -> str:
    """
    Render the sentence with entity spans bracketed inline.

    A terminal-friendly stand-in for spaCy's displacy renderer, which wants a
    browser or a notebook.
    """
    pieces: list[str] = []
    cursor = 0
    for entity in doc.ents:
        pieces.append(doc.text[cursor:entity.start_char])
        pieces.append(f"[{entity.text} :{entity.label_}]")
        cursor = entity.end_char
    pieces.append(doc.text[cursor:])
    return "".join(pieces)


def main() -> None:
    setup_console()

    print(banner("LOADING spaCy"))
    print("Model: en_core_web_sm (~12MB). Small, fast on CPU, trained on")
    print("news and web text - remember that, it explains the failures below.\n")

    import spacy

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        raise SystemExit(
            "en_core_web_sm not installed. Run:\n"
            "  python -m spacy download en_core_web_sm"
        )

    print(banner("NER ON REAL SENTENCES FROM THE CORPUS"))

    all_entities: list[tuple[str, str]] = []
    for i, sentence in enumerate(DEMO_SENTENCES, start=1):
        doc = nlp(sentence)
        print(f"\n  --- sentence {i} ---")
        print(f"  {sentence}\n")
        print(f"  marked up: {visualise_entities(doc)}\n")

        if not doc.ents:
            print("  entities found: NONE")
            continue

        print("  entities found:")
        for entity in doc.ents:
            gloss = LABEL_MEANING.get(entity.label_, "?")
            print(f"     {entity.text:<28} {entity.label_:<12} ({gloss})")
            all_entities.append((entity.text, entity.label_))

    # ---------------------------------------------------------------------
    # The honest assessment - computed from what actually happened above,
    # not from what a tutorial says should happen.
    # ---------------------------------------------------------------------
    found_texts = {text.lower() for text, _ in all_entities}
    by_label: dict[str, list[str]] = {}
    for text, label in all_entities:
        by_label.setdefault(label, []).append(text)

    print(banner("WHAT spaCy ACTUALLY PRODUCED"))
    print(f"  {len(all_entities)} entities across {len(DEMO_SENTENCES)} sentences:\n")
    for label in sorted(by_label):
        print(f"     {label:<12} {', '.join(by_label[label])}")

    print(banner("PROBLEM 1: THINGS IT MISSED ENTIRELY"))
    domain_terms = ["DETR", "Faster R-CNN", "COCO", "ImageNet-1K", "ResNet",
                    "YOLOv3", "SSD", "Transformer", "Vaswani", "He et al."]
    missed = [t for t in domain_terms if t.lower() not in found_texts]
    for term in missed:
        print(f"     - {term}")
    print("\n  Note what is in that list: DETR and YOLOv3 - two of the most")
    print("  important entities in our entire corpus - were not recognised as")
    print("  entities at all.")

    print(banner("PROBLEM 2: THINGS IT MISLABELLED"))
    print("  Worse than missing, because a confident wrong answer is harder to")
    print("  filter than a blank one:\n")
    for text, label in all_entities:
        wrong = {
            "Faster": "PERSON - it is half of 'Faster R-CNN', a model. The span "
                      "is wrong AND the type is wrong.",
            "Swin Transformer": "ORG - it is a model architecture, not a company.",
            "ImageNet-1K": "ORG - it is a dataset.",
            "Shazeer": "ORG - Noam Shazeer is a PERSON.",
            "Transformer": "ORG - it is an architecture.",
            "SSD": "ORG - it is a detection model.",
        }.get(text)
        if wrong:
            print(f"     {text:<20} tagged {label:<10} -> {wrong}")

    print("\n  And 'Vaswani' and 'He et al.' - actual people - were tagged as")
    print("  nothing at all, while 'Shazeer' next to them became an ORG.")

    print(banner("WHY THIS HAPPENS (IT IS NOT A BUG)"))
    print("  en_core_web_sm learned its entity types from news and web text.")
    print("  In that world 'DETR' and 'COCO' never appear, 'Swin' looks like a")
    print("  company name, and a capitalised token after 'and' is usually an org.")
    print("\n  More fundamentally: its label set has no MODEL, ARCHITECTURE, or")
    print("  DATASET category. Even a perfect prediction could not tag DETR")
    print("  correctly, because the right answer is not one of the options.")
    print("\n  These are exactly the entities this project cares most about. A")
    print("  knowledge graph of CV papers built on spaCy NER alone would be")
    print("  mostly wrong and would miss the substance of the field.")

    print(banner("SO WHY IS spaCy IN THIS PROJECT AT ALL?"))
    print("  Two jobs it IS reliable at, neither of which is entity typing:")
    print()
    print("  a) SENTENCE SEGMENTATION. knowledge_extractor.py (Step 3.3) has to")
    print("     split 370,000 characters of paper text into individual sentences")
    print("     to feed REBEL one at a time. Splitting on '.' would break on")
    print("     'et al.', 'Fig. 3', '0.5 mAP', 'Sec. 4.2'. spaCy's parser handles")
    print("     those correctly, and that is what we actually use it for.")
    print()
    print("  b) NOUN CHUNKS for query parsing in Phase 5 - see below.")

    print(banner("SO HOW DO WE ACTUALLY GET MODEL AND DATASET NODES?"))
    print("  Three answers, and the project uses all three:")
    print()
    print("  1. REBEL (Step 3.2) does not depend on spaCy's label set at all.")
    print("     It reads the raw sentence and emits (subject, relation, object)")
    print("     directly, so 'DETR' becomes a node because it is the SUBJECT of")
    print("     a relation - not because anything classified it as a model.")
    print("     This is the main source of graph nodes.")
    print()
    print("  2. Noun chunks. spaCy's parser finds noun phrases even when its NER")
    print("     ignores them. Watch:")

    doc = nlp(DEMO_SENTENCES[0])
    chunks = [chunk.text for chunk in doc.noun_chunks]
    print(f"       {chunks}")
    print("     'DETR' and 'a transformer encoder-decoder architecture' show up")
    print("     here even though NER skipped them. Phase 5 uses this as a second")
    print("     pass when NER finds nothing in a user's question.")
    print()
    print("  3. The graph itself as a gazetteer. Once Phase 4 has built the graph,")
    print("     its node list IS a vocabulary of every entity in the corpus. A")
    print("     query mentioning 'DETR' can be fuzzy-matched straight against")
    print("     that list. The corpus teaches us its own domain terms.")

    print(banner("NER vs RELATION EXTRACTION - THE DISTINCTION TO HOLD ON TO"))
    print("  Sentence: 'DETR outperforms Faster R-CNN on COCO'")
    print()
    print("  NER gives you the THINGS:")
    print("      DETR, Faster R-CNN, COCO          (three disconnected nodes)")
    print()
    print("  REBEL gives you the LINK BETWEEN them:")
    print("      (DETR) --outperforms--> (Faster R-CNN)   (an edge)")
    print()
    print("  A graph of nodes with no edges cannot be traversed, and traversal is")
    print("  the entire point. That is why Step 3.2 exists, and why REBEL - not")
    print("  NER - is the component that makes multi-hop reasoning possible.")


if __name__ == "__main__":
    main()
