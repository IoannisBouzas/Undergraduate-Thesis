"""Regression fixtures for the null-judge-label policy.

Policy of record: a PRESENT-BUT-NULL judge_correct means "judged run, row
unscored" -> the row is EXCLUDED from every accuracy denominator (matching
judge_meta n_scored). It is never counted as incorrect and never falls back
to lexical metrics. Fixture rows cover judge_correct in [1, 0, None].

Run: python -m tests.test_null_judge_policy   (or pytest -q test_null_judge_policy.py)
"""

from __future__ import annotations

import json
import os
import tempfile

FIXTURE_ROWS = [
    {"qid": "q1", "mode": "vectorrag", "judge_correct": 1, "f1": 0.0, "contains": 0.0},
    {"qid": "q2", "mode": "vectorrag", "judge_correct": 0, "f1": 1.0, "contains": 1.0},
    # null with PERFECT lexical scores: any fallback or as-incorrect handling changes accuracy
    {"qid": "q3", "mode": "vectorrag", "judge_correct": None, "f1": 1.0, "contains": 1.0},
]


def test_analyze_run_is_correct():
    from energy.analyze_run import is_correct
    assert is_correct(FIXTURE_ROWS[0], "contains", 0.3) == 1.0
    assert is_correct(FIXTURE_ROWS[1], "contains", 0.3) == 0.0, \
        "judge says wrong; lexical must not override"
    assert is_correct(FIXTURE_ROWS[2], "contains", 0.3) is None, \
        "null judge label must be missing, never lexical fallback"
    # no judge key at all -> lexical fallback still allowed (unjudged run)
    assert is_correct({"qid": "q4", "contains": 1.0}, "contains", 0.3) == 1.0


def _write_run_dir(root: str, name: str) -> str:
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "judged.jsonl"), "w", encoding="utf-8") as f:
        for r in FIXTURE_ROWS:
            f.write(json.dumps(r) + "\n")
    return d


def test_aggregate_results_summarize():
    from analysis.aggregate_results import summarize
    with tempfile.TemporaryDirectory() as tmp:
        d = _write_run_dir(tmp, "fixture_run")
        rec = summarize(d)
    assert rec["n"] == 3
    assert rec["n_judged"] == 2, "null row must not enter the judge denominator"
    assert rec["judge"] == 0.5, f"accuracy must be 1/2, got {rec['judge']}"


def test_stats_report_load():
    from analysis import stats_report
    with tempfile.TemporaryDirectory() as tmp:
        _write_run_dir(os.path.join(tmp, "results"), "fixture_run")
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            labels = stats_report.load("fixture_run")
        finally:
            os.chdir(cwd)
    assert labels == {"q1": 1, "q2": 0}, \
        "null row must be dropped from McNemar pairing, not scored 0"


def test_pilot_prefix_compare_load_judged():
    from pathlib import Path
    from analysis.pilot_prefix_compare import load_judged
    with tempfile.TemporaryDirectory() as tmp:
        d = _write_run_dir(tmp, "fixture_run")
        labels = load_judged(Path(d))
    assert labels == {"q1": True, "q2": False}, \
        "null row must be excluded from judged labels"


if __name__ == "__main__":
    for fn in (test_analyze_run_is_correct, test_aggregate_results_summarize,
               test_stats_report_load, test_pilot_prefix_compare_load_judged):
        fn()
        print(f"PASS {fn.__name__}")
    print("all null-judge-policy fixtures pass")
