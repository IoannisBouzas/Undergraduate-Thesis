#!/usr/bin/env python3
"""Per-category accuracy and paired transitions between two memory conditions.

Regroups existing judged.jsonl rows (zero model calls), for LoCoMo and LongMemEval alike. For each
comparison BASE -> NEW and each question category it reports the correct counts, the discordant
pairs (fixed = BASE wrong and NEW right, broken = BASE right and NEW wrong) and the two-sided exact
McNemar p. At 50 questions per LoCoMo category and 10 per LongMemEval type, the discordant counts
are the defensible unit: an accuracy delta alone says nothing about whether one condition fixed
questions the other missed, or merely traded them. All p-values here are exploratory (raw, no
correction), and LoCoMo questions cluster within ten conversations, which these tests ignore.

A run argument may be "RUN_A|RUN_B": a question counts as correct if either run answered it. With
the v3 and v4 graph runs this is the post hoc selection oracle of Chapter 7 (spe_oracle_bound.json).

Usage (from the repository root):
  python -m analysis.category_breakdown                      # the thesis comparison set, both benchmarks
  python -m analysis.category_breakdown BASE NEW [NEW ...]   # any runs in results/
  python -m analysis.category_breakdown BASE NEW --list      # also print every flipped question
Writes results/category_transitions.csv for the default set.
"""

import csv
import json
import sys
from pathlib import Path

from analysis.stats_report import check_pairable, mcnemar_exact

RESULTS = Path("results")

# Dataset field -> category name, as in tab:ch4_locomo_categories. The numeric field does not
# follow the order of the LoCoMo paper's prose, so the map is fixed here, not inferred.
LOCOMO_CATEGORIES = {"1": "Multi-hop", "2": "Temporal", "3": "Open-domain", "4": "Single-hop"}
LME_CATEGORIES = {
    "single-session-user": "Single-session user",
    "single-session-assistant": "Single-session assistant",
    "single-session-preference": "Preference",
    "multi-session": "Multi-session",
    "knowledge-update": "Knowledge update",
    "temporal-reasoning": "Temporal reasoning",
}


def _load_one(run):
    rows = {}
    for line in (RESULTS / run / "judged.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("judge_correct") is None:
            print(f"WARNING: {run}: unscored row {r.get('qid')} excluded", file=sys.stderr)
            continue
        # qid, not stable_qid: older runs lack the latter. LoCoMo qid depends on --limit, so
        # check_pairable refuses cross-limit pairs and compare() checks the question text.
        rows[r["qid"]] = r
    return rows


def load(run):
    """{question key: row}; "A|B" merges two runs, correct if either is."""
    parts = run.split("|")
    merged = _load_one(parts[0])
    for other_run in parts[1:]:
        if not check_pairable(parts[0], other_run):
            sys.exit(1)
        other = _load_one(other_run)
        assert other.keys() == merged.keys(), f"{run}: question sets differ"
        merged = {k: {**r, "judge_correct": bool(r["judge_correct"]) or bool(other[k]["judge_correct"])}
                  for k, r in merged.items()}
    return merged


def category_name(row):
    c = str(row.get("category", "?"))
    return LOCOMO_CATEGORIES.get(c) or LME_CATEGORIES.get(c) or c


def compare(base_run, new_run):
    """One row per category plus an "All" row."""
    if not check_pairable(base_run.split("|")[0], new_run.split("|")[0]):
        sys.exit(1)
    base, new = load(base_run), load(new_run)
    assert base.keys() == new.keys(), f"{base_run} vs {new_run}: question sets differ"
    assert all(base[k]["question"] == new[k]["question"] for k in base), "same key, different question"
    names = list(LOCOMO_CATEGORIES.values()) + list(LME_CATEGORIES.values())
    cats = sorted({category_name(r) for r in base.values()}, key=lambda c: names.index(c) if c in names else 99)
    out = []
    for cat in cats + ["All"]:
        keys = [k for k in base if cat == "All" or category_name(base[k]) == cat]
        b = [bool(base[k]["judge_correct"]) for k in keys]
        n_ = [bool(new[k]["judge_correct"]) for k in keys]
        fixed = sum(1 for x, y in zip(b, n_) if not x and y)
        broken = sum(1 for x, y in zip(b, n_) if x and not y)
        out.append({
            "base": base_run, "new": new_run, "category": cat, "n": len(keys),
            "base_correct": sum(b), "new_correct": sum(n_), "fixed": fixed, "broken": broken,
            "p_exact": round(mcnemar_exact(fixed, broken), 4),
            "fixed_qids": [k for k, x, y in zip(keys, b, n_) if not x and y],
            "broken_qids": [k for k, x, y in zip(keys, b, n_) if x and not y],
        })
    return out, base, new


def print_table(rows):
    print(f"\n{rows[0]['base']}  ->  {rows[0]['new']}")
    print(f"  {'category':26s} {'n':>3s}  {'base':>9s}  {'new':>9s}  {'fixed':>5s} {'broken':>6s}  {'p':>6s}")
    for r in rows:
        pct = lambda k: f"{r[k]:3d} {100 * r[k] / r['n']:4.0f}%"
        print(f"  {r['category']:26s} {r['n']:3d}  {pct('base_correct')}  {pct('new_correct')}  "
              f"{r['fixed']:5d} {r['broken']:6d}  {r['p_exact']:6.3f}")


def print_flips(rows, base, new):
    for r in rows[:-1]:
        for label, keys in (("fixed", r["fixed_qids"]), ("broken", r["broken_qids"])):
            for k in keys:
                print(f"  [{r['category']}] {label}: {base[k]['question']}\n"
                      f"      gold: {base[k]['gold']} | base: {base[k]['pred']} | new: {new[k]['pred']}")


def default_comparisons():
    pairs = []
    for m in ("e2b", "e4b", "q35_08b", "q35_2b", "q35_4b"):
        run = lambda s: f"lap_locomo200_{m}_{s}"
        pairs += [(run("fullcontext"), run("graphrag_v3_prefix")),
                  (run("graphrag_v3_prefix"), run("graphrag_v4_prefix")),
                  (run("graphrag_v4_prefix"), run("vr_prefix")),
                  (run("fullcontext"), run("vr_prefix"))]
    for m in ("e2b", "e4b"):  # SPE and the selection oracle exist for the Gemma readers only
        run = lambda s: f"lap_locomo200_{m}_{s}"
        v3, v4 = run("graphrag_v3_prefix"), run("graphrag_v4_prefix")
        for spe in ("spe2fix_gap_b240", "spe2fix_gap_b370", "spe1fix_b240", "spe1fix_b370"):
            pairs += [(v3, run(spe)), (v4, run(spe))]
        pairs.append((v4, f"{v3}|{v4}"))
    for m in ("e2b", "e4b"):
        run = lambda s: f"lme_strat60_{m}_{s}"
        pairs += [(run("fc"), run("graphrag_v3_prefix")),
                  (run("graphrag_v3_prefix"), run("graphrag_v4_prefix")),
                  (run("graphrag_v4_prefix"), run("vr_prefix")),
                  (run("fc"), run("vr_prefix"))]
    for m in ("q35_08b", "q35_2b", "q35_4b"):
        pairs.append((f"lme_strat60_{m}_fc", f"lme_strat60_{m}_vr_prefix"))
    return pairs


def check_against_master(rows):
    """The "All" row must reproduce the published run accuracy, or the join is wrong."""
    master = {r["dir"]: float(r["judge"]) for r in csv.DictReader((RESULTS / "master_table.csv").open(encoding="utf-8"))}
    total = rows[-1]
    for run, key in ((total["base"], "base_correct"), (total["new"], "new_correct")):
        if "|" not in run and run in master:
            assert abs(total[key] / total["n"] - master[run]) < 1e-9, f"{run}: {total[key]}/{total['n']} vs master {master[run]}"


def main(argv):
    show_flips = "--list" in argv
    args = [a for a in argv if a != "--list"]
    pairs = [(args[0], new) for new in args[1:]] if len(args) >= 2 else default_comparisons()
    all_rows = []
    for base_run, new_run in pairs:
        rows, base, new = compare(base_run, new_run)
        check_against_master(rows)
        print_table(rows)
        if show_flips:
            print_flips(rows, base, new)
        all_rows += rows
    if len(args) < 2:
        path = RESULTS / "category_transitions.csv"
        fields = ["base", "new", "category", "n", "base_correct", "new_correct", "fixed", "broken", "p_exact"]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nwrote {path} ({len(all_rows)} rows, {len(pairs)} comparisons)")


if __name__ == "__main__":
    main(sys.argv[1:])
