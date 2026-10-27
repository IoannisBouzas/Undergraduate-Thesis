#!/usr/bin/env python3
"""Conversation-cluster uncertainty for LoCoMo pairs (questions cluster by conversation).

LoCoMo n=200 is 10 conversations x ~20 questions, so judged rows are NOT independent:
questions from one conversation share a speaker pair, a persona, and a haystack. Wilson
CIs and McNemar both treat the 200 rows as 200 independent items and therefore understate
uncertainty. LongMemEval strat60 is 60 sessions x 1 question, so it has no such structure
and needs no correction -- that asymmetry is itself reportable.

This script quantifies the cost with a cluster bootstrap (resample CONVERSATIONS with
replacement, not questions) and reports the design effect

    deff = (cluster-bootstrap SE / item-level SE)^2

for both the single-cell accuracy and the paired accuracy difference.

Caveat that belongs in the thesis next to any number this prints: with only 10 clusters
the bootstrap is coarse. Percentile intervals are reported (BCa is not trustworthy at
10 draws). Treat deff as an order-of-magnitude correction factor, not a precise estimate.

Usage: python -m analysis.cluster_uncertainty                 # default headline pairs
       python -m analysis.cluster_uncertainty --pairs A B     # ad-hoc (dir names under results/)
"""
import json, math, random, argparse, collections, sys, os

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

B_BOOT = 20000
SEED = 7

# Pairs worth characterising: one headline, one moderate, one marginal, one k-sweep step.
# Deliberately small -- the point is to measure the design effect, not to re-run the grid.
DEFAULT_PAIRS = [
    ("lap_locomo200_e2b_vr_prefix", "lap_locomo200_e2b_fullcontext"),
    ("lap_locomo200_e2b_vr_prefix", "lap_locomo200_e2b_graphrag_v3_prefix"),
    ("lap_locomo200_e2b_vr_think_prefix", "lap_locomo200_e2b_vr_prefix"),
    ("lap_locomo200_e2b_fc_offprompt", "lap_locomo200_e2b_fullcontext"),
    ("lap_locomo100_e2b_vr_prefix_k10", "lap_locomo100_e2b_vr_prefix_k5"),
]


def load(d):
    """qid -> (conversation id, judged verdict). Pairing key stays qid: both arms of every
    pair here share a --limit, which is the condition stats_report.py already enforces."""
    path = os.path.join("results", d, "judged.jsonl")
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return {r["qid"]: (r.get("sample_id"), 1 if r.get("judge_correct") else 0) for r in rows}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def cluster_boot(units, rng):
    """units: cluster -> list of per-item values. Resample clusters with replacement,
    keeping every question inside a drawn conversation (that is what carries the
    within-conversation correlation)."""
    keys = list(units)
    out = []
    for _ in range(B_BOOT):
        samp = []
        for _ in keys:
            samp.extend(units[rng.choice(keys)])
        out.append(sum(samp) / len(samp))
    out.sort()
    lo = out[int(0.025 * B_BOOT)]
    hi = out[int(0.975 * B_BOOT)]
    se = (out[int(0.84 * B_BOOT)] - out[int(0.16 * B_BOOT)]) / 2  # robust SE from the spread
    return lo, hi, se


def report(a_name, b_name, rng):
    a, b = load(a_name), load(b_name)
    qs = sorted(set(a) & set(b))
    n = len(qs)
    if not n:
        print(f"SKIP {a_name} vs {b_name}: no shared qids")
        return

    by_conv = collections.defaultdict(list)
    for q in qs:
        by_conv[a[q][0]].append(a[q][1])
    k = sum(a[q][1] for q in qs)
    item_se = math.sqrt((k / n) * (1 - k / n) / n)
    wlo, whi = wilson(k, n)
    blo, bhi, bse = cluster_boot(by_conv, rng)

    dcl = collections.defaultdict(list)
    for q in qs:
        dcl[a[q][0]].append(a[q][1] - b[q][1])
    diffs = [a[q][1] - b[q][1] for q in qs]
    d = sum(diffs) / n
    item_se_d = math.sqrt(sum((x - d) ** 2 for x in diffs) / (n - 1) / n)
    dlo, dhi, dse = cluster_boot(dcl, rng)
    n01 = sum(1 for x in diffs if x == 1)
    n10 = sum(1 for x in diffs if x == -1)

    print(f"\n{a_name}  vs  {b_name}   (n={n} paired, {len(by_conv)} conversations)")
    print(f"  acc(A) {k/n:.3f}   item Wilson [{wlo:.3f},{whi:.3f}] (w={whi-wlo:.3f})"
          f"   cluster [{blo:.3f},{bhi:.3f}] (w={bhi-blo:.3f})   deff={(bse/item_se)**2:.2f}")
    print(f"  diff   {d:+.3f}   discordant {n01}/{n10}"
          f"   item [{d-1.96*item_se_d:+.3f},{d+1.96*item_se_d:+.3f}]"
          f"   cluster [{dlo:+.3f},{dhi:+.3f}]   deff={(dse/item_se_d)**2:.2f}")
    crosses = dlo <= 0 <= dhi
    print(f"  cluster CI {'INCLUDES 0 — not separable once conversation structure is honoured'
                          if crosses else 'excludes 0 — survives clustering'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    rng = random.Random(SEED)
    pairs = [tuple(args.pairs)] if args.pairs else DEFAULT_PAIRS
    print("CONVERSATION-CLUSTER UNCERTAINTY — cluster bootstrap over LoCoMo conversations")
    print(f"B={B_BOOT}, seed={SEED}, percentile intervals (10 clusters: coarse by construction)")
    print("=" * 78)
    for a, b in pairs:
        try:
            report(a, b, rng)
        except FileNotFoundError as e:
            print(f"\nSKIP {a} vs {b}: {e}")
    print("\nLongMemEval strat60 = 60 sessions x 1 question => no conversation clustering; "
          "item-level CIs are already correct there.")


if __name__ == "__main__":
    main()
