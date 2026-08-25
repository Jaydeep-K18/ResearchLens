"""
Phase 7, Step 7.2 - the evaluation harness. This file produces the project's evidence.

WHY THIS IS THE MOST IMPORTANT FILE FOR AN INTERVIEW
-----------------------------------------------------
Everything else in this repository is a claim: "a knowledge graph makes
multi-hop retrieval possible". This file is the test of that claim, and it is
designed so it could FAIL. If KG-RAG scores the same as basic RAG on multi-hop
questions, five phases of work were not worth it, and the honest thing is to
report that.

Twenty questions in three difficulty tiers, run through BOTH systems, scored by
an LLM judge on three axes. The expected outcome, from PROJECT_CONTEXT.md:

    simple  questions: roughly equal          (vector search is enough)
    medium  questions: KG-RAG somewhat ahead  (1-2 hops start to matter)
    hard    questions: KG-RAG clearly ahead   (3+ hops are impossible otherwise)

The simple tier is not padding. If KG-RAG were WORSE there, that would mean the
graph evidence is crowding out good text evidence - a real regression that the
headline multi-hop number would otherwise hide.

ON USING AN LLM AS A JUDGE
--------------------------
Imperfect, and worth being clear about why we do it anyway. Judges are known to
favour longer answers and their own phrasing, and Gemini judging a Gemini answer
shares blind spots. Three things keep it useful here:

  - Both systems are judged by the same judge on the same rubric, so bias
    applies equally to both. We report a DIFFERENCE, which is more robust than
    either absolute score.
  - Every question has a hand-written expected answer, so the judge is comparing
    against ground truth rather than deciding what it thinks is true.
  - Per-question scores are saved to eval_results.json, so any claim in the
    README can be traced back to a specific answer and checked by hand.

The scores are evidence, not proof. Treat a 0.3 gap as noise and a 2-point gap
as a finding.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llm import LLMUnavailable, get_llm  # noqa: E402
from src.utils import EVAL_RESULTS_PATH, banner, setup_console  # noqa: E402

# Free-tier Gemini Flash allows roughly 15 requests/minute. A full run is
# ~80 calls (20 questions x 2 systems x [answer + judge]), so we pace ourselves
# rather than relying on the retry logic to absorb a wall of 429s.
DEFAULT_DELAY = 4.0


# ---------------------------------------------------------------------------
# The test set
# ---------------------------------------------------------------------------
# Written against the eight papers in data/raw/. Expected answers are the
# GROUND TRUTH the judge compares against, so they are deliberately factual and
# specific - a vague expected answer makes any answer look correct.

QUESTIONS: list[dict] = [
    # ---------------- SIMPLE: answerable from one paper, stated plainly -----
    {
        "id": 1, "difficulty": "simple",
        "question": "What is the main architectural innovation of the Transformer?",
        "expected": "The Transformer dispenses with recurrence and convolutions entirely and "
                    "relies solely on attention mechanisms, specifically multi-head "
                    "self-attention, to draw global dependencies between input and output. "
                    "This makes it far more parallelisable than RNN-based models.",
    },
    {
        "id": 2, "difficulty": "simple",
        "question": "What BLEU score does the Transformer achieve on the WMT 2014 "
                    "English-to-German translation task?",
        "expected": "The big Transformer model achieves 28.4 BLEU on WMT 2014 "
                    "English-to-German, improving over the previous best results including "
                    "ensembles by over 2 BLEU.",
    },
    {
        "id": 3, "difficulty": "simple",
        "question": "What problem do residual connections in ResNet solve?",
        "expected": "They address the degradation problem: as plain deep networks get deeper, "
                    "training accuracy saturates and then degrades - not from overfitting. "
                    "Residual connections let layers fit a residual mapping F(x) = H(x) - x, "
                    "making very deep networks (50-152 layers) easier to optimise.",
    },
    {
        "id": 4, "difficulty": "simple",
        "question": "How does Faster R-CNN generate region proposals?",
        "expected": "With a Region Proposal Network (RPN), a fully convolutional network that "
                    "shares convolutional features with the detection network and predicts "
                    "object bounds and objectness scores at each position using anchor boxes "
                    "of multiple scales and aspect ratios. This makes proposals nearly cost-free.",
    },
    {
        "id": 5, "difficulty": "simple",
        "question": "What does Mask R-CNN add on top of Faster R-CNN?",
        "expected": "A third branch that predicts a segmentation mask for each Region of "
                    "Interest, in parallel with the existing branches for classification and "
                    "bounding box regression. It also replaces RoIPool with RoIAlign to fix "
                    "misalignment.",
    },
    {
        "id": 6, "difficulty": "simple",
        "question": "What is the key idea behind DETR's approach to object detection?",
        "expected": "DETR treats detection as a direct set prediction problem, using a "
                    "transformer encoder-decoder with a set-based global loss that forces "
                    "unique predictions via bipartite matching. This removes hand-designed "
                    "components such as non-maximum suppression and anchor generation.",
    },
    {
        "id": 7, "difficulty": "simple",
        "question": "How does the Vision Transformer apply a Transformer to images?",
        "expected": "ViT splits an image into fixed-size patches, linearly embeds each patch, "
                    "adds position embeddings, and feeds the resulting sequence of vectors to a "
                    "standard Transformer encoder - treating image patches the same way NLP "
                    "treats tokens. A classification token is prepended for the class prediction.",
    },

    # ---------------- MEDIUM: cross-document, 1-2 hops ----------------------
    {
        "id": 8, "difficulty": "medium",
        "question": "Which models in this corpus were evaluated on the COCO dataset?",
        "expected": "DETR, Mask R-CNN, Faster R-CNN and YOLOv3 all report results on COCO. "
                    "Swin Transformer also evaluates on COCO for detection and segmentation.",
    },
    {
        "id": 9, "difficulty": "medium",
        "question": "Which papers in this corpus build directly on the Transformer architecture?",
        "expected": "The Vision Transformer (ViT) applies a standard Transformer encoder to "
                    "image patches; DETR uses a transformer encoder-decoder for detection; and "
                    "Swin Transformer builds a hierarchical vision transformer with shifted "
                    "windows. All three derive from Vaswani et al.'s Transformer.",
    },
    {
        "id": 10, "difficulty": "medium",
        "question": "What datasets was the Vision Transformer pre-trained on, and how did "
                    "dataset scale affect its performance?",
        "expected": "ViT was pre-trained on ImageNet-1k, ImageNet-21k and JFT-300M. It "
                    "underperforms ResNets when trained on small datasets, because it lacks the "
                    "inductive biases of convolutions, but overtakes them once pre-trained on "
                    "large datasets - scale substitutes for inductive bias.",
    },
    {
        "id": 11, "difficulty": "medium",
        "question": "How does Swin Transformer differ from the Vision Transformer?",
        "expected": "Swin builds hierarchical feature maps and computes self-attention within "
                    "local shifted windows, giving linear complexity in image size, whereas ViT "
                    "uses a single-scale global self-attention with quadratic complexity. This "
                    "makes Swin usable as a general-purpose backbone for detection and "
                    "segmentation, not just classification.",
    },
    {
        "id": 12, "difficulty": "medium",
        "question": "What is RoIAlign and which problem with RoIPool does it fix?",
        "expected": "RoIAlign removes the harsh quantisation of RoIPool, which rounds RoI "
                    "boundaries and bins to the feature-map grid and causes misalignment between "
                    "the RoI and the extracted features. RoIAlign uses bilinear interpolation at "
                    "sampled points instead, which matters greatly for pixel-accurate masks.",
    },
    {
        "id": 13, "difficulty": "medium",
        "question": "Which backbone networks are used by the detection models in this corpus?",
        "expected": "ResNet (and ResNeXt) backbones, often with a Feature Pyramid Network, are "
                    "used by Mask R-CNN and DETR; Faster R-CNN uses ZF and VGG-16 in the "
                    "original paper; YOLOv3 uses Darknet-53. Swin Transformer is itself proposed "
                    "as a backbone.",
    },
    {
        "id": 14, "difficulty": "medium",
        "question": "Which models does DETR compare itself against, and how does it perform?",
        "expected": "DETR compares against Faster R-CNN baselines on COCO and demonstrates "
                    "comparable or better accuracy - notably better on large objects, attributed "
                    "to the transformer's global reasoning, but worse on small objects.",
    },

    # ---------------- HARD: 3+ hops, genuinely cross-document ---------------
    {
        "id": 15, "difficulty": "hard",
        "question": "Which models outperform or improve on Faster R-CNN, and who proposed "
                    "those models?",
        "expected": "Mask R-CNN extends and outperforms Faster R-CNN and was proposed by He, "
                    "Gkioxari, Dollar and Girshick. DETR matches or beats Faster R-CNN "
                    "baselines and was proposed by Carion et al. Faster R-CNN itself was "
                    "proposed by Ren, He, Girshick and Sun.",
    },
    {
        "id": 16, "difficulty": "hard",
        "question": "Which authors of object detection papers in this corpus also contributed "
                    "to segmentation research?",
        "expected": "Kaiming He and Ross Girshick appear on both Faster R-CNN (detection) and "
                    "Mask R-CNN (instance segmentation). Kaiming He is also an author of ResNet. "
                    "Shaoqing Ren and Jian Sun co-authored Faster R-CNN and ResNet.",
    },
    {
        "id": 17, "difficulty": "hard",
        "question": "Trace the architectural lineage from the original Transformer to Swin "
                    "Transformer.",
        "expected": "Vaswani et al. proposed the Transformer for machine translation using "
                    "self-attention. Dosovitskiy et al. adapted it to vision as ViT by treating "
                    "image patches as tokens. Liu et al. then proposed Swin Transformer, which "
                    "addresses ViT's quadratic cost and lack of multi-scale features using "
                    "hierarchical shifted-window attention, making it a general backbone.",
    },
    {
        "id": 18, "difficulty": "hard",
        "question": "Which model that uses a ResNet backbone is outperformed by a "
                    "transformer-based model, and on what benchmark?",
        "expected": "Faster R-CNN and Mask R-CNN use ResNet backbones and are outperformed on "
                    "COCO by transformer-based approaches - DETR matches/exceeds Faster R-CNN "
                    "baselines, and Swin Transformer as a backbone substantially exceeds "
                    "ResNet-based detectors on COCO box and mask AP.",
    },
    {
        "id": 19, "difficulty": "hard",
        "question": "Kaiming He co-authored which papers in this corpus, and what is the "
                    "connection between them?",
        "expected": "Kaiming He co-authored ResNet, Faster R-CNN and Mask R-CNN. The connection "
                    "is a chain: ResNet provides the backbone architecture used by Faster R-CNN, "
                    "which Mask R-CNN then extends with a mask branch for instance segmentation. "
                    "The same author line runs from classification through detection to "
                    "segmentation.",
    },
    {
        "id": 20, "difficulty": "hard",
        "question": "Among the models evaluated on COCO, which are transformer-based, and which "
                    "convolutional model did each of them compare against?",
        "expected": "DETR is transformer-based and compares against Faster R-CNN on COCO. Swin "
                    "Transformer is transformer-based and compares against ResNet/ResNeXt-based "
                    "detectors on COCO. Both use a convolutional detector family as the baseline "
                    "they aim to surpass.",
    },
]


# ---------------------------------------------------------------------------
# The judge
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are grading a retrieval-augmented question answering system against a reference answer. Be strict and consistent.

QUESTION:
{question}

REFERENCE ANSWER (ground truth, written by a domain expert):
{expected}

SYSTEM ANSWER:
{answer}

Score the SYSTEM ANSWER on three axes, each an integer from 1 to 5:

correctness       - Are the stated facts right? 5 = everything correct.
                    1 = mostly wrong or hallucinated. Penalise invented facts
                    heavily. An answer that correctly says it cannot answer from
                    the given context scores 2, not 1 - it is honest but unhelpful.

completeness      - How much of the reference answer is covered? 5 = all key
                    points. 1 = almost nothing.

citation_accuracy - Are claims cited as [Source: filename, p.N], and do the cited
                    files plausibly match the claims? 5 = consistently cited.
                    1 = no citations at all.

Respond with ONLY a JSON object, no other text and no markdown fence:
{{"correctness": <int>, "completeness": <int>, "citation_accuracy": <int>, "reason": "<one sentence>"}}
"""


def parse_judge_response(text: str) -> dict:
    """
    Pull the scores out of the judge's reply.

    LLMs wrap JSON in markdown fences, add preambles, or trail explanation
    despite instructions, so we extract the first {...} block rather than trusting
    the whole response to parse. A judge failure must not kill a 15-minute run.
    """
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return {"correctness": 0, "completeness": 0, "citation_accuracy": 0,
                "reason": "judge response could not be parsed", "parse_failed": True}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"correctness": 0, "completeness": 0, "citation_accuracy": 0,
                "reason": "judge returned invalid JSON", "parse_failed": True}

    scores = {}
    for axis in ("correctness", "completeness", "citation_accuracy"):
        try:
            scores[axis] = max(1, min(5, int(data.get(axis, 0))))
        except (TypeError, ValueError):
            scores[axis] = 0
    scores["reason"] = str(data.get("reason", ""))[:300]
    scores["parse_failed"] = False
    return scores


def judge_answer(question: str, expected: str, answer: str, delay: float) -> dict:
    if not answer or not answer.strip():
        return {"correctness": 1, "completeness": 1, "citation_accuracy": 1,
                "reason": "system produced no answer", "parse_failed": False}

    llm = get_llm()
    prompt = JUDGE_PROMPT.format(question=question, expected=expected, answer=answer)
    try:
        response = llm.generate(prompt, temperature=0.0, max_tokens=300)
    except LLMUnavailable as exc:
        return {"correctness": 0, "completeness": 0, "citation_accuracy": 0,
                "reason": f"judge unavailable: {exc}", "parse_failed": True}
    finally:
        time.sleep(delay)

    return parse_judge_response(response)


# ---------------------------------------------------------------------------
# Running the two systems
# ---------------------------------------------------------------------------

def run_evaluation(limit: int | None = None, delay: float = DEFAULT_DELAY,
                   difficulties: set[str] | None = None) -> dict:
    from src.basic_rag import answer_question
    from src.pipeline import KGRagPipeline
    from src.vector_store import VectorStore

    llm = get_llm()
    if not llm.available:
        raise SystemExit(llm.setup_hint())

    store = VectorStore()
    if store.count() == 0:
        raise SystemExit("Vector store is empty. Run: python src/vector_store.py")

    pipeline = KGRagPipeline()

    questions = QUESTIONS
    if difficulties:
        questions = [q for q in questions if q["difficulty"] in difficulties]
    if limit:
        questions = questions[:limit]

    print(f"Evaluating {len(questions)} questions through 2 systems "
          f"({len(questions) * 4} LLM calls at {delay}s spacing)")
    print(f"Estimated time: {len(questions) * 4 * (delay + 2) / 60:.0f} minutes\n")

    records: list[dict] = []

    for index, item in enumerate(questions, start=1):
        print(f"[{index}/{len(questions)}] ({item['difficulty']}) {item['question'][:70]}...")

        # --- basic RAG ---------------------------------------------------
        basic = answer_question(item["question"], store, show_chunks=False)
        time.sleep(delay)
        basic_scores = judge_answer(item["question"], item["expected"],
                                    basic.get("answer", ""), delay)

        # --- KG-RAG ------------------------------------------------------
        state = pipeline.run(item["question"])
        time.sleep(delay)
        kgrag_scores = judge_answer(item["question"], item["expected"],
                                    state.get("answer", ""), delay)

        reranked = state.get("reranked_results") or []
        methods = [r["retrieval_method"] for r in reranked]

        records.append({
            "id": item["id"],
            "difficulty": item["difficulty"],
            "question": item["question"],
            "expected": item["expected"],
            "basic_rag": {
                "answer": basic.get("answer", ""),
                "scores": basic_scores,
                "n_chunks": len(basic.get("chunks", [])),
            },
            "kg_rag": {
                "answer": state.get("answer", ""),
                "scores": kgrag_scores,
                "n_vector": len(state.get("vector_results") or []),
                "n_graph": len(state.get("graph_results") or []),
                "final_mix": {"vector": methods.count("vector"),
                              "graph": methods.count("graph")},
                "sources": state.get("sources", []),
            },
        })

        print(f"      basic  c={basic_scores['correctness']} "
              f"cm={basic_scores['completeness']} ci={basic_scores['citation_accuracy']}"
              f"   |   kg-rag c={kgrag_scores['correctness']} "
              f"cm={kgrag_scores['completeness']} ci={kgrag_scores['citation_accuracy']}"
              f"   (graph contributed {methods.count('graph')}/{len(methods)})")

    return {"results": records, "summary": summarise(records)}


def _mean(values: list[float]) -> float:
    return round(statistics.mean(values), 2) if values else 0.0


def summarise(records: list[dict]) -> dict:
    """Average each axis per difficulty tier, per system."""
    summary: dict = {}
    tiers = ["simple", "medium", "hard", "ALL"]

    for tier in tiers:
        subset = records if tier == "ALL" else [r for r in records if r["difficulty"] == tier]
        if not subset:
            continue

        tier_summary: dict = {"n": len(subset)}
        for system in ("basic_rag", "kg_rag"):
            axes = {}
            for axis in ("correctness", "completeness", "citation_accuracy"):
                axes[axis] = _mean([r[system]["scores"][axis] for r in subset
                                    if not r[system]["scores"].get("parse_failed")])
            axes["overall"] = _mean(list(axes.values()))
            tier_summary[system] = axes

        tier_summary["delta_overall"] = round(
            tier_summary["kg_rag"]["overall"] - tier_summary["basic_rag"]["overall"], 2
        )
        # How much the graph actually contributed to the final evidence.
        mixes = [r["kg_rag"]["final_mix"] for r in subset]
        total = sum(m["vector"] + m["graph"] for m in mixes)
        tier_summary["graph_share"] = round(
            sum(m["graph"] for m in mixes) / total, 2) if total else 0.0

        summary[tier] = tier_summary

    return summary


def print_summary(summary: dict) -> None:
    print(banner("RESULTS: BASIC RAG vs KG-RAG"))
    print(f"  {'tier':<9}{'n':>3}   {'basic':>7}{'kg-rag':>9}{'delta':>8}   "
          f"{'graph share':>12}")
    print("  " + "-" * 56)
    for tier in ("simple", "medium", "hard", "ALL"):
        if tier not in summary:
            continue
        data = summary[tier]
        basic = data["basic_rag"]["overall"]
        kgrag = data["kg_rag"]["overall"]
        delta = data["delta_overall"]
        marker = "  <--" if tier == "hard" and delta > 0.5 else ""
        print(f"  {tier:<9}{data['n']:>3}   {basic:>7.2f}{kgrag:>9.2f}{delta:>+8.2f}   "
              f"{data['graph_share']:>11.0%}{marker}")

    print(banner("BREAKDOWN BY AXIS"))
    for tier in ("simple", "medium", "hard", "ALL"):
        if tier not in summary:
            continue
        data = summary[tier]
        print(f"\n  {tier} (n={data['n']})")
        print(f"    {'axis':<20}{'basic':>8}{'kg-rag':>9}{'delta':>8}")
        for axis in ("correctness", "completeness", "citation_accuracy"):
            basic = data["basic_rag"][axis]
            kgrag = data["kg_rag"][axis]
            print(f"    {axis:<20}{basic:>8.2f}{kgrag:>9.2f}{kgrag - basic:>+8.2f}")


def main() -> None:
    setup_console()

    parser = argparse.ArgumentParser(description="Evaluate basic RAG vs KG-RAG")
    parser.add_argument("--limit", type=int, help="only run the first N questions")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                        help="seconds between LLM calls (free-tier rate limiting)")
    parser.add_argument("--difficulty", nargs="+",
                        choices=["simple", "medium", "hard"],
                        help="only run these tiers")
    parser.add_argument("--report-only", action="store_true",
                        help="re-print the summary from a saved eval_results.json")
    args = parser.parse_args()

    if args.report_only:
        if not EVAL_RESULTS_PATH.exists():
            raise SystemExit(f"{EVAL_RESULTS_PATH} not found - run the evaluation first.")
        saved = json.loads(EVAL_RESULTS_PATH.read_text(encoding="utf-8"))
        print_summary(saved["summary"])
        return

    started = time.time()
    output = run_evaluation(
        limit=args.limit,
        delay=args.delay,
        difficulties=set(args.difficulty) if args.difficulty else None,
    )
    output["meta"] = {
        "questions": len(output["results"]),
        "elapsed_minutes": round((time.time() - started) / 60, 1),
    }

    EVAL_RESULTS_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved per-question results to {EVAL_RESULTS_PATH}")

    print_summary(output["summary"])

    print(banner("HOW TO READ THIS"))
    print("  The 'delta' column is the headline number, and the tiers matter:")
    print("    simple  - a delta near zero is the RIGHT result. It means adding")
    print("              the graph did not damage what already worked.")
    print("    hard    - this is where the graph has to earn its place. A large")
    print("              positive delta is the finding this project exists to show.")
    print("\n  'graph share' is how much of the final evidence came from graph")
    print("  traversal. If it is near zero on hard questions, the graph is not")
    print("  actually contributing and any score gap is coming from somewhere else.")


if __name__ == "__main__":
    main()
