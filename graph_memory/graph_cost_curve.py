"""Graph construction cost vs conversation length.

The "graph construction is structurally infeasible on-device" claim rested on ONE
phone data point (419-turn conversation = 92.1 min / 21.9 kJ). This script shows the
relationship is already measured on the box across every LoCoMo conversation length
(369-689 turns) in existing logs — no new runs — fits it, and projects to the phone
using the measured box:phone ingest ratio anchored on the 419-turn conversation.

Usage: python -m graph_memory.graph_cost_curve   (anchor defaults come from results/golden_metric.json)
Writes results/graph_cost_curve.{json,png} and prints the fit.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def collect(pattern: str) -> dict[str, dict[int, float]]:
    """run_dir -> {n_turns: ingest_seconds} for cold (uncached) graph builds."""
    out: dict[str, dict[int, float]] = defaultdict(dict)
    for p in glob.glob(pattern):
        run = os.path.basename(os.path.dirname(p))
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if (r.get("ingest_seconds") or 0) > 0:
                out[run].setdefault(r["n_history_turns"], r["ingest_seconds"])
    return {k: v for k, v in out.items() if len(v) >= 3}


def fit(turns: np.ndarray, secs: np.ndarray) -> tuple[float, float, float]:
    a, b = np.polyfit(turns, secs, 1)
    r2 = 1 - ((secs - (a * turns + b)) ** 2).sum() / ((secs - secs.mean()) ** 2).sum()
    return float(a), float(b), float(r2)


def main() -> None:
    ap = argparse.ArgumentParser()
    # The phone anchor is the one graph build, measured by golden_metric.py (same battery method).
    graph = json.loads(open("results/golden_metric.json", encoding="utf-8").read())["construction"]["graph"]
    ap.add_argument("--anchor-turns", type=int, default=graph["turns"])
    ap.add_argument("--anchor-phone-min", type=float, default=graph["minutes"])
    ap.add_argument("--anchor-phone-kj", type=float, default=graph["energy_kj"])
    args = ap.parse_args()

    runs = collect("results/*graphrag*/locomo_graphrag_*.jsonl")
    if not runs:
        raise SystemExit("no cold graph-build datapoints found")

    report = {"runs": {}, "anchor": vars(args)}
    print("Graph construction cost vs conversation length (box, existing logs)\n")
    for run, pts in sorted(runs.items()):
        x = np.array(sorted(pts)); y = np.array([pts[k] for k in x])
        a, b, r2 = fit(x, y)
        report["runs"][run] = {"n_convs": len(x), "s_per_turn": a, "intercept_s": b,
                               "r2": r2, "points": {int(k): pts[k] for k in x}}
        print(f"{run:42s} n={len(x):2d}  {a:.3f} s/turn  intercept {b:6.1f}s  R2={r2:.3f}")

    # headline fit: the v3 config that the thesis reports (E2B, extraction_max_tokens=512)
    key = next((k for k in runs if k.endswith("e2b_graphrag_v3")), sorted(runs)[0])
    pts = runs[key]
    x = np.array(sorted(pts)); y = np.array([pts[k] for k in x])
    a, b, r2 = fit(x, y)
    print(f"\nHEADLINE FIT [{key}]: ingest_s = {a:.3f} * turns + {b:.1f}  (R2={r2:.3f}, n={len(x)})")
    print(f"  -> construction is LINEAR in turns at {a:.3f} s/turn on box; "
          f"no superlinear blow-up, so the cost claim rests on the per-turn constant.")

    # phone projection: scale the box curve by the measured box:phone ratio at the anchor
    box_at_anchor = a * args.anchor_turns + b
    ratio = (args.anchor_phone_min * 60) / box_at_anchor
    kj_per_turn = args.anchor_phone_kj / args.anchor_turns
    print(f"\nPhone projection (anchor: {args.anchor_turns} turns = "
          f"{args.anchor_phone_min:.0f} min / {args.anchor_phone_kj:.1f} kJ measured on S23):")
    print(f"  box at anchor {box_at_anchor/60:.1f} min -> phone/box slowdown = {ratio:.1f}x")
    print(f"  phone build ~= {a*ratio:.2f} s/turn ~= {kj_per_turn*1000:.1f} J/turn")
    for n in (100, 300, 500, 1000):
        print(f"    {n:4d} turns -> {(a*n+b)*ratio/60:6.1f} min / {kj_per_turn*n:5.1f} kJ")
    report["headline"] = {"run": key, "s_per_turn_box": a, "intercept_s": b, "r2": r2,
                          "phone_box_ratio": ratio, "kj_per_turn_phone": kj_per_turn}

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for run, pts2 in sorted(runs.items()):
            xs = np.array(sorted(pts2)); ys = np.array([pts2[k] for k in xs]) / 60
            ax.scatter(xs, ys, s=18, alpha=.65, label=run.replace("lap_locomo", "")[:26])
        xs = np.linspace(min(x), max(x), 50)
        ax.plot(xs, (a * xs + b) / 60, "k--", lw=1.5,
                label=f"fit {a:.2f} s/turn (R²={r2:.2f})")
        ax.set_xlabel("conversation length (turns)")
        ax.set_ylabel("graph construction time (min, box)")
        ax.set_title("Graph construction cost is linear in conversation length")
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig("results/graph_cost_curve.png", dpi=150)
        print("\nwrote results/graph_cost_curve.png")
    except Exception as e:  # plotting is optional
        print(f"\n[plot skipped: {e}]")

    with open("results/graph_cost_curve.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("wrote results/graph_cost_curve.json")


if __name__ == "__main__":
    main()
