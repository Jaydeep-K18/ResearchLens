"""
Inject the evaluation results from data/eval_results.json into README.md.

Why a script rather than pasting the numbers by hand: the README is the document
recruiters read, and a hand-copied table drifts from the JSON the moment the
evaluation is re-run. This keeps the published claim and the saved evidence in
sync, and makes it obvious that the table is generated rather than asserted.

Run after `python src/evaluate.py`:

    python scripts/update_readme_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import EVAL_RESULTS_PATH, PROJECT_ROOT  # noqa: E402

README_PATH = PROJECT_ROOT / "README.md"
START_MARKER = "<!-- RESULTS_TABLE_START -->"
END_MARKER = "<!-- RESULTS_TABLE_END -->"


LABELS = {
    "simple": "Simple (single-doc)",
    "medium": "Medium (cross-doc, 1-2 hops)",
    "hard": "**Hard (multi-hop, 3+)**",
    "ALL": "All",
}


def _table(data: dict, source_name: str) -> str:
    summary = data["summary"]
    meta = data.get("meta", {})

    lines = [
        "| Tier | n | Basic RAG | KG-RAG | Delta | Graph share of evidence |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for tier in ("simple", "medium", "hard", "ALL"):
        if tier not in summary:
            continue
        entry = summary[tier]
        basic = entry["basic_rag"]["overall"]
        kgrag = entry["kg_rag"]["overall"]
        delta = entry["delta_overall"]
        bold = tier == "hard"
        mark = "**" if bold else ""
        lines.append(
            f"| {LABELS[tier]} | {entry['n']} | {mark}{basic:.2f}{mark} | "
            f"{mark}{kgrag:.2f}{mark} | {mark}{delta:+.2f}{mark} | "
            f"{entry['graph_share']:.0%} |"
        )

    failed = sum(summary[t]["kg_rag"].get("failed", 0)
                 for t in ("simple", "medium", "hard") if t in summary)
    if failed:
        lines.append("")
        lines.append(f"> **Incomplete run:** {failed} question(s) produced no answer "
                     f"(API failure) and are excluded from these means. Re-run before "
                     f"quoting these numbers.")

    lines.append("")
    lines.append(f"<sub>From `{source_name}` "
                 f"({meta.get('questions', '?')} questions, "
                 f"{meta.get('elapsed_minutes', '?')} min).</sub>")
    return "\n".join(lines)


def build_table(data: dict, controlled: dict | None = None) -> str:
    """
    Render the headline table, and - when a controlled run exists - a second one
    that matches the evidence budgets.

    Why two tables. The default comparison pits the full KG-RAG system (12
    evidence items, cross-encoder re-ranked, graph included) against the standard
    basic-RAG baseline (5 chunks). That is the honest SYSTEM-vs-SYSTEM number,
    and it is what a user would actually experience - but it does not isolate the
    graph, because KG-RAG also gets 2.4x the context. The controlled run gives
    basic RAG the same 12 items, so the remaining difference is attributable to
    graph retrieval rather than to context budget.
    """
    parts = [
        "**System vs system** — full KG-RAG against the standard basic-RAG baseline:",
        "",
        _table(data, "data/eval_results.json"),
    ]

    if controlled:
        parts += [
            "",
            "**Controlled** — basic RAG given the *same* 12 evidence items, so the only "
            "remaining difference is graph retrieval:",
            "",
            _table(controlled, "data/eval_results_controlled.json"),
        ]

    parts += [
        "",
        "<sub>Each score is the mean of correctness, completeness and citation accuracy, "
        "judged 1-5 against a hand-written reference answer. Per-question answers and the "
        "judge's reasoning are in the JSON files, so any number here can be checked by "
        "hand.</sub>",
    ]
    return "\n".join(parts)


def main() -> None:
    if not EVAL_RESULTS_PATH.exists():
        raise SystemExit(
            f"{EVAL_RESULTS_PATH} not found.\nRun the evaluation first: python src/evaluate.py"
        )

    data = json.loads(EVAL_RESULTS_PATH.read_text(encoding="utf-8"))

    controlled_path = EVAL_RESULTS_PATH.parent / "eval_results_controlled.json"
    controlled = (json.loads(controlled_path.read_text(encoding="utf-8"))
                  if controlled_path.exists() else None)

    table = build_table(data, controlled)

    readme = README_PATH.read_text(encoding="utf-8")
    if START_MARKER not in readme or END_MARKER not in readme:
        raise SystemExit(f"Could not find the {START_MARKER} / {END_MARKER} markers in README.md")

    before = readme.split(START_MARKER)[0]
    after = readme.split(END_MARKER)[1]
    README_PATH.write_text(
        f"{before}{START_MARKER}\n{table}\n{END_MARKER}{after}", encoding="utf-8"
    )

    print("README.md results table updated:\n")
    print(table)


if __name__ == "__main__":
    main()
