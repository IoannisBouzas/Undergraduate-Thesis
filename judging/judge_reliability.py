#!/usr/bin/env python3
"""Measure the 70b judge's run-to-run reliability on IDENTICAL inputs.

Motivation (2026-07-26): `lap_locomo100_e2b_vr_prefix_k10` and `lap_locomo100_e2b_vr_parity`
are the same configuration and produced byte-identical predictions (0/100 mismatches), yet
their judged accuracies differed — 30.0% vs 29.0%, one flipped verdict. judge.py already
sends temperature=0, so this is not a sampling setting: it is GPU-level nondeterminism
(batching/KV layout changes the floating-point reduction order, so greedy argmax can flip on
a near-tie). The flipped row was a borderline paraphrase, which is where near-ties live.

That single observation is an anecdote. This turns it into a measured instrument
characteristic: rejudge the same predictions R times and report the per-row flip rate and the
spread in cell accuracy.

This is NOT a defect to hide. Every LLM-as-judge result in the literature has this property;
almost none report it. Reporting it is a rigor win, and it sets the resolution floor below
which accuracy differences must not be interpreted.

Usage:
  python -m judging.judge_reliability --runs lap_locomo100_e2b_vr_prefix_k2 lap_locomo100_e2b_vr_prefix_k5 --reps 3
Writes results/judge_reliability.{json,md}. Re-judges into a scratch dir; never touches the
run's own judged.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics as st
import subprocess
import sys
import tempfile

JUDGE = "llama3.1:70b-instruct-q4_K_M"
BASE_URL = "http://localhost:18080/v1"


def load_verdicts(judged_path):
    out = {}
    for line in open(judged_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        v = r.get("judge_correct")
        if v is not None:
            out[r.get("qid")] = int(bool(v))
    return out


def rejudge_once(run_dir, dataset, tmp_root, rep):
    """Copy predictions into a scratch dir and judge them fresh. The scratch dir keeps the
    run's real judged.jsonl untouched, so a failed rep can never corrupt a banked result."""
    scratch = os.path.join(tmp_root, f"rep{rep}")
    os.makedirs(scratch, exist_ok=True)
    for f in os.listdir(run_dir):
        if f.endswith(".jsonl") and "judged" not in f and "extraction" not in f:
            shutil.copy2(os.path.join(run_dir, f), scratch)
        elif f.endswith("_summary.json"):
            shutil.copy2(os.path.join(run_dir, f), scratch)
    cmd = [sys.executable, "judge.py", "--run-dir", scratch, "--dataset", dataset,
           "--judge-model", JUDGE, "--base-url", BASE_URL, "--prompt-style", "official"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    jp = os.path.join(scratch, "judged.jsonl")
    if p.returncode != 0 or not os.path.exists(jp):
        print(f"    rep{rep} FAILED rc={p.returncode}: {(p.stderr or '')[-200:]}", file=sys.stderr)
        return None
    return load_verdicts(jp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--dataset", default="locomo")
    args = ap.parse_args()

    report = {"judge_model": JUDGE, "reps": args.reps, "runs": {}}
    for run in args.runs:
        run_dir = os.path.join("results", run)
        if not os.path.isdir(run_dir):
            print(f"SKIP {run}: no such dir", file=sys.stderr)
            continue
        print(f"=== {run} ({args.reps} independent judgings)")
        tmp_root = tempfile.mkdtemp(prefix=f"jrel_{run}_")
        try:
            reps = []
            for i in range(args.reps):
                v = rejudge_once(run_dir, args.dataset, tmp_root, i)
                if v:
                    reps.append(v)
                    print(f"    rep{i}: acc={100*sum(v.values())/len(v):.1f}% (n={len(v)})")
            if len(reps) < 2:
                print(f"    too few successful reps, skipping {run}", file=sys.stderr)
                continue
            common = sorted(set.intersection(*(set(r) for r in reps)))
            # a row is UNSTABLE if the reps disagree about it
            unstable = [q for q in common if len({r[q] for r in reps}) > 1]
            accs = [100 * sum(r[q] for q in common) / len(common) for r in reps]
            rec = {
                "n_rows": len(common),
                "n_unstable": len(unstable),
                "flip_rate_pct": 100 * len(unstable) / len(common) if common else 0.0,
                "acc_per_rep": accs,
                "acc_range_pts": max(accs) - min(accs),
                "acc_sd_pts": st.pstdev(accs) if len(accs) > 1 else 0.0,
                "unstable_qids": unstable[:20],
            }
            report["runs"][run] = rec
            print(f"    -> {rec['n_unstable']}/{rec['n_rows']} rows unstable "
                  f"({rec['flip_rate_pct']:.1f}%), accuracy range {rec['acc_range_pts']:.1f} pts")
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    if report["runs"]:
        flips = [r["flip_rate_pct"] for r in report["runs"].values()]
        ranges = [r["acc_range_pts"] for r in report["runs"].values()]
        report["summary"] = {
            "mean_flip_rate_pct": st.mean(flips),
            "max_acc_range_pts": max(ranges),
            "regime": "within-session (reps share one warm server process)",
            "interpretation": (
                "WITHIN-SESSION ONLY. All reps hit the same already-loaded 70b instance, where "
                "the batching/KV layout is fixed and greedy decoding is exactly reproducible — "
                "so a 0.0 range here does NOT mean the judge is deterministic in general, and "
                "must not be used as the noise floor. The reported noise floor comes from the "
                "ACROSS-SESSION regime (server restarted between judgings): 1 flipped verdict "
                "in 100 rows, i.e. about +/-1 pt at n=100. See results/judge_reliability.md. "
                "Paired tests remain valid when both arms are scored in the same pass."),
        }
    os.makedirs("results", exist_ok=True)
    with open("results/judge_reliability.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    with open("results/judge_reliability.md", "w", encoding="utf-8") as fh:
        fh.write(f"# Judge reliability — {JUDGE}, temperature=0, {args.reps} reps\n\n")
        fh.write("**WITHIN-SESSION regime only** (all reps share one warm server process). "
                 "The across-session floor is ~+/-1 pt at n=100 and is NOT measured here — "
                 "see the checked-in results/judge_reliability.md before citing anything below.\n\n")
        fh.write("| run | rows | unstable | flip rate | acc range (pts) |\n|---|---|---|---|---|\n")
        for run, r in report["runs"].items():
            fh.write(f"| {run} | {r['n_rows']} | {r['n_unstable']} | "
                     f"{r['flip_rate_pct']:.1f}% | {r['acc_range_pts']:.1f} |\n")
        if "summary" in report:
            fh.write(f"\nMean flip rate **{report['summary']['mean_flip_rate_pct']:.1f}%**; "
                     f"max accuracy range **{report['summary']['max_acc_range_pts']:.1f} pts**.\n\n")
            fh.write(report["summary"]["interpretation"] + "\n")
    print("\nwrote results/judge_reliability.{json,md}")


if __name__ == "__main__":
    main()
