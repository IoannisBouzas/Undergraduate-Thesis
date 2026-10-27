#!/usr/bin/env python3
"""Chapter 4 supporting tables, regenerated from banked run artifacts. No model calls.

Three tables that the existing aggregators do not emit, each needed by a claim in
docs/thesis/ch4_memory_strategy_evaluation.tex:

  1. results/locomo_category_table.csv
     Per-category judged accuracy on LoCoMo, the counterpart of lme_category_table.csv.
     Regroups judged.jsonl by the `category` field already stamped on every row.
     LoCoMo cells are n=50 per category against LongMemEval's n=10, so this is the
     better-powered category analysis and it was previously uncomputed.

  2. results/evidence_completeness_table.csv
     Refines completeness_table.csv. That table's `evidence_recall` is a PER-QUESTION
     "at least one evidence turn present" rate (context_completeness.py:152 sets
     evidence_in_ctx = int(ev_hits > 0)). It is often misread as turn-level recall, and
     the two differ substantially: E2B vector at k=50 is 85% of questions with >=1 turn
     but 69.1% of evidence turns. This table reports both, and splits judged accuracy
     three ways - complete evidence / partial evidence / none - because the two-way split
     merges "the reader failed" with "only part of the evidence arrived".
     Reads the per-run completeness.jsonl written by context_completeness.py; it does NOT
     re-parse prompt logs, so it cannot change any banked completeness value.

  3. results/extraction_stats_table.csv
     Graph extraction health per extractor, aggregated over the ten LoCoMo conversations
     from ctx_meta.extraction_stats. Use THIS rather than counting extraction_failures.jsonl
     lines: that file was not written by the run version that produced the Gemma cells, so
     a line count reports zero failures for Gemma when the true counts are 9 and 7.

Usage: python -m analysis.ch4_tables       -> writes the three CSVs and prints them
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS = Path(__file__).resolve().parents[1] / "results"
LOCOMO_CATEGORIES = ["1", "2", "3", "4"]


def rows_of(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def pct(num, den):
    return round(100 * num / den, 1) if den else None


# ------------------------------------------------------------------ 1. LoCoMo categories

def locomo_category_table():
    out = []
    for d in sorted(RESULTS.iterdir()):
        if not d.is_dir() or not d.name.startswith("lap_locomo") or not (d / "judged.jsonl").exists():
            continue
        rows = [r for r in rows_of(d / "judged.jsonl") if r.get("judge_correct") is not None]
        if not rows:
            continue
        by = {}
        for r in rows:
            by.setdefault(str(r.get("category")), []).append(int(bool(r["judge_correct"])))
        rec = {"run": d.name, "mode": rows[0].get("mode"), "model": rows[0].get("model")}
        scored = []
        for c in LOCOMO_CATEGORIES:
            xs = by.get(c, [])
            scored += xs
            rec[f"cat{c}"] = pct(sum(xs), len(xs))
            rec[f"cat{c}_n"] = len(xs)
        rec["overall"] = pct(sum(scored), len(scored))
        rec["n"] = len(scored)
        out.append(rec)
    return out


# ------------------------------------------------- 2. evidence completeness (all/partial/none)

def evidence_completeness_table():
    out = []
    for d in sorted(RESULTS.iterdir()):
        if not d.is_dir() or not (d / "completeness.jsonl").exists():
            continue
        rows = rows_of(d / "completeness.jsonl")
        if not rows:
            continue
        n = len(rows)
        turns_hit = sum(r["evidence_turns_hit"] for r in rows)
        turns_tot = sum(r["n_evidence_turns"] for r in rows)

        def bucket(r):
            if r["evidence_turns_hit"] == 0:
                return "none"
            return "all" if r["evidence_turns_hit"] == r["n_evidence_turns"] else "partial"

        groups = {"all": [], "partial": [], "none": []}
        for r in rows:
            groups[bucket(r)].append(r)

        rec = {
            "run": d.name,
            "n": n,
            # the value completeness_table.csv calls `evidence_recall`
            "q_any_evidence_pct": pct(len(groups["all"]) + len(groups["partial"]), n),
            "q_all_evidence_pct": pct(len(groups["all"]), n),
            "turn_level_recall_pct": pct(turns_hit, turns_tot),
            "evidence_turns_total": turns_tot,
        }
        for name, xs in groups.items():
            scored = [r["judge_correct"] for r in xs if r["judge_correct"] is not None]
            rec[f"acc_{name}_pct"] = pct(sum(bool(v) for v in scored), len(scored))
            rec[f"n_{name}"] = len(scored)
        out.append(rec)
    return out


# ------------------------------------------------------------ 3. graph extraction health

def extraction_stats_table():
    out = []
    for d in sorted(RESULTS.iterdir()):
        if not d.is_dir() or not (d / "judged.jsonl").exists():
            continue
        per_conversation = {}
        model = None
        for r in rows_of(d / "judged.jsonl"):
            stats = (r.get("ctx_meta") or {}).get("extraction_stats")
            if stats:
                per_conversation[r.get("sample_id")] = stats
                model = model or r.get("model")
        if not per_conversation:
            continue
        total = sum(v["batches_total"] for v in per_conversation.values())
        failed = sum(v["batches_failed"] for v in per_conversation.values())
        empty = sum(v.get("batches_empty", 0) for v in per_conversation.values())
        out.append({
            "run": d.name,
            "extractor": model,
            "conversations": len(per_conversation),
            "batches_total": total,
            "batches_failed": failed,
            "failed_pct": pct(failed, total),
            "batches_empty": empty,
            "empty_pct": pct(empty, total),
            "triples_extracted": sum(v["triples_extracted"] for v in per_conversation.values()),
        })
    return out


# ------------------------------------------------------------------------------- driver

def write(name, rows):
    if not rows:
        print(f"{name}: no rows")
        return
    path = RESULTS / name
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} runs)")


def main():
    tables = [
        ("locomo_category_table.csv", locomo_category_table()),
        ("evidence_completeness_table.csv", evidence_completeness_table()),
        ("extraction_stats_table.csv", extraction_stats_table()),
    ]
    for name, rows in tables:
        write(name, rows)

    print("\n--- graph extraction health (LoCoMo v3 cells) ---")
    for r in tables[2][1]:
        if r["run"].startswith("lap_locomo200_") and r["run"].endswith("graphrag_v3_prefix"):
            print(f"  {r['extractor']:14s} {r['batches_failed']:4d}/{r['batches_total']} failed "
                  f"({r['failed_pct']:4.1f}%)  empty {r['batches_empty']:3d}  "
                  f"triples {r['triples_extracted']}")


if __name__ == "__main__":
    main()
