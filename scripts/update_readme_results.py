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


def build_table(data: dict) -> str:
    summary = data["summary"]
    meta = data.get("meta", {})

    lines = [
        "| Tier | n | Basic RAG | KG-RAG | Delta | Graph share of evidence |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    label = {"simple": "Simple (single-doc)",
             "medium": "Medium (cross-doc, 1-2 hops)",
             "hard": "**Hard (multi-hop, 3+)**",
             "ALL": "All"}

    for tier in ("simple", "medium", "hard", "ALL"):
        if tier not in summary:
            continue
        entry = summary[tier]
        basic = entry["basic_rag"]["overall"]
        kgrag = entry["kg_rag"]["overall"]
        delta = entry["delta_overall"]
        bold = tier == "hard"
        fmt = (lambda v: f"**{v:.2f}**") if bold else (lambda v: f"{v:.2f}")
        lines.append(
            f"| {label[tier]} | {entry['n']} | {fmt(basic)} | {fmt(kgrag)} | "
            f"{'**' if bold else ''}{delta:+.2f}{'**' if bold else ''} | "
            f"{entry['graph_share']:.0%} |"
        )

    lines.append("")
    lines.append("*Scores are the mean of correctness, completeness and citation accuracy, "
                 "each judged 1-5 against a hand-written reference answer.*")

    if meta:
        lines.append(f"*Generated from `data/eval_results.json` "
                     f"({meta.get('questions', '?')} questions, "
                     f"{meta.get('elapsed_minutes', '?')} min run).*")

    return "\n".join(lines)


def main() -> None:
    if not EVAL_RESULTS_PATH.exists():
        raise SystemExit(
            f"{EVAL_RESULTS_PATH} not found.\nRun the evaluation first: python src/evaluate.py"
        )

    data = json.loads(EVAL_RESULTS_PATH.read_text(encoding="utf-8"))
    table = build_table(data)

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
