#!/usr/bin/env python3
"""Consolidate every results/*/judged.jsonl into one master table.

Box-quality metrics only (judge accuracy, tokens, box latency). On-device energy lives
separately in the phone runs (see project memory / validation_notes.md). Rework-proof:
reads whatever judged dirs exist on disk at call time.

Usage: python -m analysis.aggregate_results            # pretty table to stdout
       python -m analysis.aggregate_results --csv out.csv
"""
import json, glob, os, argparse, sys, statistics as st

# --- population category distributions, for the weighted accuracy column ------------------
# Both benchmarks are sampled BALANCED-BY-CATEGORY (equal n per category) so that each
# category gets equal statistical power — that is what makes the per-category analysis in
# Ch 2c possible. But neither benchmark is naturally balanced, so the plain mean over a
# balanced sample is NOT an estimate of accuracy on the benchmark as a whole, which is what
# a reader comparing against published LoCoMo/LongMemEval numbers will assume.
# `judge_w` re-weights the per-category rates by the true population shares.
# Counts measured from the source files (locomo10.json, longmemeval_s_cleaned.json),
# LoCoMo excluding category 5 (adversarial, answer=None) per standard practice.
LOCOMO_POP = {"1": 282, "2": 321, "3": 96, "4": 841}           # 1540 answerable of 1986
LME_POP = {"knowledge-update": 78, "multi-session": 133, "single-session-assistant": 56,
           "single-session-preference": 30, "single-session-user": 70,
           "temporal-reasoning": 133}                           # 500 total


def population_for(rows):
    """Pick the population table by matching observed category labels. None = don't weight."""
    cats = {str(r.get("category") or r.get("source_category")) for r in rows}
    cats.discard("None")
    if cats and cats <= set(LOCOMO_POP):
        return LOCOMO_POP
    if cats and cats <= set(LME_POP):
        return LME_POP
    return None


def weighted_accuracy(rows):
    """Population-weighted judge accuracy, or nan when categories don't map to a known
    benchmark or a category is missing from the sample (partial reweighting would be
    misleading, so refuse rather than guess)."""
    pop = population_for(rows)
    if not pop:
        return float("nan")
    by = {}
    for r in rows:
        v = r.get("judge_correct")
        if v is None:
            continue
        by.setdefault(str(r.get("category") or r.get("source_category")), []).append(int(bool(v)))
    if not by or set(by) != set(pop):
        return float("nan")
    tot = sum(pop.values())
    return sum(pop[c] / tot * (sum(v) / len(v)) for c, v in by.items())


def rows_of(d):
    p = os.path.join(d, "judged.jsonl")
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out

def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else float("nan")

def summarize(d):
    rows = rows_of(d)
    if not rows:
        return None
    n = len(rows)
    r0 = rows[0]
    # Null judge_correct = unscored (judge error/unparseable), NOT incorrect.
    # Policy of record: exclude from the accuracy denominator (matches judge_meta
    # n_scored) and warn loudly — rejudge_missing.py scores them properly.
    jc_vals = [r.get("judge_correct") for r in rows]
    jc_scored = [int(bool(v)) for v in jc_vals if v is not None]
    n_null = len(jc_vals) - len(jc_scored)
    if n_null:
        print(f"WARNING: {os.path.basename(d)}: {n_null} unscored judge row(s) excluded "
              f"(denominator {len(jc_scored)}); run rejudge_missing.py", file=sys.stderr)
    jc = st.mean(jc_scored) if jc_scored else float("nan")
    con = mean([1 if (r.get("contains") or 0) > 0 else 0 for r in rows])
    f1 = mean([float(r.get("f1") or 0) for r in rows])
    pt = mean([r.get("prompt_tokens") for r in rows])
    ct = mean([r.get("completion_tokens") for r in rows])
    # box query latency: prefer api_total_seconds, fall back to generate_seconds
    lat = mean([r.get("api_total_seconds") or r.get("generate_seconds") for r in rows])
    return dict(
        dir=os.path.basename(d),
        dataset=r0.get("run_id", ""),  # keep run_id; dataset inferred from name
        mode=r0.get("mode", ""),
        model=r0.get("model", ""),
        n=n, n_judged=len(jc_scored),
        judge=jc, judge_w=weighted_accuracy(rows), contains=con, f1=f1,
        prompt_tok=pt, comp_tok=ct, box_lat_s=lat,
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/master_table.csv",
                    help="output CSV (default results/master_table.csv; pass '' to skip)")
    ap.add_argument("--glob", default="results/*")
    args = ap.parse_args()

    dirs = sorted(g for g in glob.glob(args.glob) if os.path.isdir(g))
    recs = [s for s in (summarize(d) for d in dirs) if s]

    hdr = ["dir", "mode", "model", "n", "n_judged", "judge", "judge_w", "contains", "f1", "prompt_tok", "comp_tok", "box_lat_s"]
    w = {"dir": 32, "mode": 11, "model": 11, "n": 4, "n_judged": 8, "judge": 6, "judge_w": 7, "contains": 8,
         "f1": 5, "prompt_tok": 10, "comp_tok": 8, "box_lat_s": 9}

    def fmt(rec):
        cells = []
        for h in hdr:
            v = rec[h]
            if h in ("judge", "judge_w", "contains"):
                v = "-" if v != v else f"{v*100:.0f}%"
            elif h == "f1":
                v = f"{v:.2f}"
            elif h in ("prompt_tok", "comp_tok", "box_lat_s"):
                v = f"{v:.1f}"
            cells.append(str(v).ljust(w[h]))
        return " ".join(cells)

    print(" ".join(h.ljust(w[h]) for h in hdr))
    print("-" * (sum(w.values()) + len(w)))
    for rec in recs:
        print(fmt(rec))

    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=hdr)
            wr.writeheader()
            for rec in recs:
                wr.writerow({h: rec[h] for h in hdr})
        print(f"\nwrote {args.csv} ({len(recs)} rows)")

if __name__ == "__main__":
    main()
