"""
Phase 7 tests - the evaluation harness.

Two things matter here and neither involves calling an LLM:

  1. The judge's reply must be parsed robustly. A full evaluation run is ~80 API
     calls over 15 minutes; one unparseable response must not take the run down.
  2. The summary arithmetic must be right, because those numbers go in the README
     and get quoted in interviews. A scoring bug here would be the worst kind -
     it produces confident, specific, wrong claims.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import QUESTIONS, parse_judge_response, summarise  # noqa: E402


# ---------------------------------------------------------------------------
# The question set
# ---------------------------------------------------------------------------

def test_there_are_twenty_questions_in_the_specified_tiers():
    assert len(QUESTIONS) == 20
    tiers = [q["difficulty"] for q in QUESTIONS]
    assert tiers.count("simple") == 7
    assert tiers.count("medium") == 7
    assert tiers.count("hard") == 6


def test_question_ids_are_unique():
    ids = [q["id"] for q in QUESTIONS]
    assert len(ids) == len(set(ids))


def test_every_question_has_a_substantive_reference_answer():
    # A vague reference answer makes any system answer look correct, which
    # quietly destroys the whole comparison.
    for question in QUESTIONS:
        assert len(question["expected"]) > 80, f"reference answer too thin: {question['id']}"
        # Most are interrogative, but a few are imperative ("Trace the lineage
        # from X to Y."), which is a perfectly good multi-hop prompt.
        assert question["question"].rstrip().endswith(("?", ".")), question["id"]
        assert len(question["question"]) > 30, question["id"]


# NOTE: there is deliberately no test asserting that the "hard" questions are
# genuinely multi-hop. Whether a question requires three hops is a judgement
# about meaning, and every string heuristic for it is wrong in both directions:
# Q17 ("Trace the lineage from the Transformer to Swin Transformer") names two
# entities but contains no comma or "and", while Q15 ("Which models outperform
# Faster R-CNN, and who proposed those models?") names only ONE entity and is
# unambiguously three hops. A test that pattern-matches the phrasing would pass
# vacuously while verifying nothing.
#
# The real check is empirical and lives in the evaluation itself: the
# `graph_share` column reports how much of the final evidence came from graph
# traversal. If the hard tier scores well with a graph share near zero, the tier
# is not exercising multi-hop retrieval - and that shows up in the results
# rather than in a test.


# ---------------------------------------------------------------------------
# Judge response parsing
# ---------------------------------------------------------------------------

def test_parses_clean_json():
    scores = parse_judge_response(
        '{"correctness": 4, "completeness": 3, "citation_accuracy": 5, "reason": "good"}'
    )
    assert scores["correctness"] == 4
    assert scores["completeness"] == 3
    assert scores["citation_accuracy"] == 5
    assert not scores["parse_failed"]


def test_parses_json_wrapped_in_a_markdown_fence():
    # LLMs add fences despite being told not to. This is the single most common
    # real-world failure mode of "respond with only JSON".
    scores = parse_judge_response(
        '```json\n{"correctness": 5, "completeness": 4, "citation_accuracy": 3, '
        '"reason": "x"}\n```'
    )
    assert scores["correctness"] == 5
    assert not scores["parse_failed"]


def test_parses_json_with_a_chatty_preamble():
    scores = parse_judge_response(
        'Sure! Here is my evaluation:\n\n'
        '{"correctness": 2, "completeness": 2, "citation_accuracy": 1, "reason": "weak"}\n'
        'Let me know if you need more detail.'
    )
    assert scores["correctness"] == 2
    assert not scores["parse_failed"]


def test_unparseable_response_fails_softly():
    # Must return a record, not raise - one bad judge reply cannot kill a
    # 15-minute run.
    scores = parse_judge_response("I am unable to evaluate this answer.")
    assert scores["parse_failed"]
    assert scores["correctness"] == 0


def test_malformed_json_fails_softly():
    scores = parse_judge_response('{"correctness": 4, "completeness":}')
    assert scores["parse_failed"]


def test_out_of_range_scores_are_clamped():
    # A judge that returns 10/5 would silently inflate the averages.
    scores = parse_judge_response(
        '{"correctness": 10, "completeness": -3, "citation_accuracy": 5, "reason": "x"}'
    )
    assert scores["correctness"] == 5
    assert scores["completeness"] == 1


def test_non_numeric_scores_do_not_crash():
    scores = parse_judge_response(
        '{"correctness": "four", "completeness": 3, "citation_accuracy": 3, "reason": "x"}'
    )
    assert scores["correctness"] == 0


# ---------------------------------------------------------------------------
# Summary arithmetic - these numbers end up in the README
# ---------------------------------------------------------------------------

def record(difficulty, basic, kgrag, vector=4, graph=4):
    def scores(value):
        return {"correctness": value, "completeness": value,
                "citation_accuracy": value, "parse_failed": False}
    return {
        "difficulty": difficulty,
        "basic_rag": {"scores": scores(basic)},
        "kg_rag": {"scores": scores(kgrag),
                   "final_mix": {"vector": vector, "graph": graph}},
    }


def test_tier_averages_and_delta_are_computed_correctly():
    records = [
        record("simple", 4, 4),
        record("simple", 5, 5),
        record("hard", 2, 4),
        record("hard", 2, 5),
    ]
    summary = summarise(records)

    assert summary["simple"]["n"] == 2
    assert summary["simple"]["basic_rag"]["overall"] == 4.5
    assert summary["simple"]["delta_overall"] == 0.0

    assert summary["hard"]["basic_rag"]["overall"] == 2.0
    assert summary["hard"]["kg_rag"]["overall"] == 4.5
    assert summary["hard"]["delta_overall"] == 2.5


def test_all_tier_aggregates_every_record():
    records = [record("simple", 4, 4), record("medium", 3, 4), record("hard", 2, 5)]
    assert summarise(records)["ALL"]["n"] == 3


def test_graph_share_reports_how_much_the_graph_contributed():
    """
    The honesty check on the whole evaluation: if the graph contributed nothing
    to the final evidence, any score difference came from somewhere else and the
    headline claim is unsupported.
    """
    records = [record("hard", 2, 5, vector=2, graph=6)]
    assert summarise(records)["hard"]["graph_share"] == 0.75

    records = [record("hard", 2, 5, vector=8, graph=0)]
    assert summarise(records)["hard"]["graph_share"] == 0.0


def test_failed_judge_records_are_excluded_from_averages():
    # A parse failure scores 0; averaging it in would drag the mean down and
    # misreport the system as worse than it is.
    good = record("hard", 4, 4)
    bad = record("hard", 0, 0)
    bad["basic_rag"]["scores"]["parse_failed"] = True
    bad["kg_rag"]["scores"]["parse_failed"] = True

    summary = summarise([good, bad])
    assert summary["hard"]["basic_rag"]["overall"] == 4.0


def test_empty_tier_is_omitted_rather_than_reported_as_zero():
    summary = summarise([record("simple", 4, 4)])
    assert "hard" not in summary
    assert "simple" in summary


# ---------------------------------------------------------------------------
# Failed generations must never be scored as bad answers
# ---------------------------------------------------------------------------

def test_empty_answer_is_marked_failed_not_scored_one():
    """
    The most important property in this file.

    An empty answer means generation never ran - almost always an exhausted API
    quota. Scoring it 1/5 turns an infrastructure failure into a data point, and
    a column of 1.0s reads exactly like a real (bad) result.

    The first full run here hit its quota at question 4 and reported
    "medium 1.00, hard 1.00" for BOTH systems. That is 17 failed API calls
    wearing the costume of a finding.
    """
    from src.evaluate import judge_answer

    scores = judge_answer("q", "expected", "", delay=0)
    assert scores["parse_failed"] is True
    assert scores["correctness"] == 0
    assert "NO ANSWER" in scores["reason"]


def test_failed_generations_are_excluded_from_the_means():
    good = record("hard", 4, 4)
    failed = record("hard", 0, 0)
    for system in ("basic_rag", "kg_rag"):
        failed[system]["scores"]["parse_failed"] = True

    summary = summarise([good, failed])
    assert summary["hard"]["kg_rag"]["overall"] == 4.0     # not (4+0)/2
    assert summary["hard"]["kg_rag"]["scored"] == 1
    assert summary["hard"]["kg_rag"]["failed"] == 1


def test_summary_reports_how_many_questions_were_actually_scored():
    records = [record("simple", 4, 4), record("simple", 3, 5)]
    summary = summarise(records)
    assert summary["simple"]["kg_rag"]["scored"] == 2
    assert summary["simple"]["kg_rag"]["failed"] == 0
