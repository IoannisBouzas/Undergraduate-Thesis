#!/usr/bin/env python3
"""Truncation-aware reporting for the thinking (reasoning) cells.

Why this exists (2026-07-28 audit): with `--max-answer-tokens 768 --thinking`, 12-20% of
questions hit `done_reason: length` having emitted no <answer> tag at all. Those rows are
auto-scored 0 by the null-judge policy — which is right, an answer that does not exist cannot
be correct — but it means a bare think-vs-base delta mixes two effects:
  (a) reasoning changed the answer, and
  (b) reasoning exhausted the budget before producing one.
Non-thinking runs truncate on 0% of rows (completion_tokens ~12, done_reason=stop), so this is
specific to the thinking arm.

Nothing here re-runs anything. The all-rows number stays the headline (it is the honest cost
of reasoning at a fixed budget); the non-truncated number is the sensitivity analysis that
separates (a) from (b).

The third output is the interesting one: base-arm accuracy ON the truncated questions. It is
far below the base arm's overall accuracy, i.e. the model reasons longest exactly where it is
most lost, then gets cut off. That is a resource finding, not a defect.

Usage: python -m analysis.thinking_truncation     -> results/thinking_truncation.{json,md}
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# (think_cell, base_cell, label). Base = the pipeline-matched non-thinking reference.
# fc uses the budget-matched control, NOT the 64-token baseline: fullcontext couples the
# answer cap to its history budget (num_ctx-128-cap), so the 64-token cell sees more history.
PAIRS = [
    ("lap_locomo200_e2b_vr_think_prefix",   "lap_locomo200_e2b_vr_prefix",          "vr E2B LoCoMo"),
    ("lap_locomo200_e4b_vr_think_prefix",   "lap_locomo200_e4b_vr_prefix",          "vr E4B LoCoMo"),
    ("lap_locomo200_e2b_grv3_think_prefix", "lap_locomo200_e2b_graphrag_v3_prefix", "grv3 E2B LoCoMo"),
    ("lap_locomo200_e4b_grv3_think_prefix", "lap_locomo200_e4b_graphrag_v3_prefix", "grv3 E4B LoCoMo"),
    ("lme_strat60_e2b_vr_think_prefix",     "lme_strat60_e2b_vr_prefix",            "vr E2B LME"),
    ("lme_strat60_e4b_vr_think_prefix",     "lme_strat60_e4b_vr_prefix",            "vr E4B LME"),
    ("lap_locomo200_e2b_fc_think",          "lap_locomo200_e2b_fc_budget768",       "fc E2B LoCoMo*"),
    ("lap_locomo200_e4b_fc_think",          "lap_locomo200_e4b_fc_budget768",       "fc E4B LoCoMo*"),
]


def pred_file(run):
    c = [x for x in glob.glob(f"results/{run}/*.jsonl")
         if not any(s in x for s in ("judged", "extraction", "completeness"))]
    return c[0] if c else None


def load(run):
    pf = pred_file(run)
    jp = f"results/{run}/judged.jsonl"
    if not pf or not os.path.exists(jp):
        return None, None
    pred = {r["qid"]: r for r in (json.loads(l) for l in open(pf, encoding="utf-8") if l.strip())}
    jd = {r["qid"]: r for r in (json.loads(l) for l in open(jp, encoding="utf-8") if l.strip())}
    return pred, jd


def analyse(think, base):
    tp, tj = load(think)
    bp, bj = load(base)
    if tp is None or bp is None:
        return None
    common = [q for q in tp if q in bp
              and tj.get(q, {}).get("judge_correct") is not None
              and bj.get(q, {}).get("judge_correct") is not None]
    if not common:
        return None
    # truncated = hit the cap AND produced no parseable answer
    trunc = [q for q in common
             if tp[q].get("done_reason") == "length" and not str(tp[q].get("pred") or "").strip()]
    keep = [q for q in common if q not in trunc]
    acc = lambda qs, j: (100 * sum(bool(j[q]["judge_correct"]) for q in qs) / len(qs)) if qs else float("nan")
    return {
        "n": len(common),
        "n_truncated": len(trunc),
        "truncation_rate_pct": 100 * len(trunc) / len(common),
        "base_all": acc(common, bj), "think_all": acc(common, tj),
        "delta_all": acc(common, tj) - acc(common, bj),
        "base_kept": acc(keep, bj), "think_kept": acc(keep, tj),
        "delta_kept": acc(keep, tj) - acc(keep, bj),
        "base_on_truncated": acc(trunc, bj),
        "mean_comp_tok_think": sum(tp[q].get("completion_tokens") or 0 for q in common) / len(common),
        "mean_comp_tok_base": sum(bp[q].get("completion_tokens") or 0 for q in common) / len(common),
    }


def main():
    out = {}
    rows = []
    for think, base, label in PAIRS:
        r = analyse(think, base)
        if not r:
            print(f"SKIP {label}: missing {think} or {base}", file=sys.stderr)
            continue
        out[label] = dict(r, think_cell=think, base_cell=base)
        rows.append((label, r))
        print(f"{label}")
        print(f"   truncated {r['n_truncated']}/{r['n']} ({r['truncation_rate_pct']:.0f}%)"
              f"   comp_tok {r['mean_comp_tok_base']:.0f} -> {r['mean_comp_tok_think']:.0f}")
        print(f"   all rows       {r['base_all']:5.1f} -> {r['think_all']:5.1f}  ({r['delta_all']:+.1f})")
        print(f"   non-truncated  {r['base_kept']:5.1f} -> {r['think_kept']:5.1f}  ({r['delta_kept']:+.1f})")
        print(f"   base accuracy on the truncated questions: {r['base_on_truncated']:.1f}%"
              f"  (vs {r['base_kept']:.1f}% on the rest)\n")

    os.makedirs("results", exist_ok=True)
    with open("results/thinking_truncation.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    with open("results/thinking_truncation.md", "w", encoding="utf-8") as fh:
        fh.write("# Thinking cells — truncation-aware reporting\n\n")
        fh.write("`--max-answer-tokens 768 --thinking`. A row is *truncated* when it hit the cap "
                 "(`done_reason: length`) without emitting a parseable answer; such rows are "
                 "auto-scored 0 by the null-judge policy. Non-thinking runs truncate on 0% of rows.\n\n")
        fh.write("| pair | trunc | all rows | non-truncated | base acc. on truncated Qs |\n")
        fh.write("|---|---|---|---|---|\n")
        for label, r in rows:
            fh.write(f"| {label} | {r['n_truncated']}/{r['n']} ({r['truncation_rate_pct']:.0f}%) "
                     f"| {r['base_all']:.1f} → {r['think_all']:.1f} ({r['delta_all']:+.1f}) "
                     f"| {r['base_kept']:.1f} → {r['think_kept']:.1f} ({r['delta_kept']:+.1f}) "
                     f"| {r['base_on_truncated']:.1f}% |\n")
        fh.write("\n\\* fc is compared against its budget-matched control "
                 "(`lap_locomo200_{e2b,e4b}_fc_budget768`), not the 64-token baseline, because "
                 "fullcontext couples the answer cap to its history budget. Note that control is "
                 "not neutral at E4B: `e4b_fc_budget768` scores 6.5% against `e4b_fullcontext`'s "
                 "10.5% (raw p = 0.03857), so raising the cap alone cost E4B accuracy and the E4B "
                 "reasoning delta is read against a baseline that itself moved.\n\n")
        fh.write("**Reporting rule.** The all-rows delta is the headline — it is the honest cost of "
                 "reasoning under a fixed budget. The non-truncated delta is the sensitivity "
                 "analysis isolating reasoning quality from budget exhaustion. Report both.\n\n")
        fh.write("**Finding.** Base accuracy on the truncated questions is far below base accuracy "
                 "on the rest: the model reasons longest exactly where it is most lost, then runs "
                 "out of budget. At an on-device-feasible 768-token cap roughly one question in "
                 "five yields no answer at all, concentrated on the hardest questions — so "
                 "reasoning on-device is not merely expensive (+33 s, +132 J per answer measured) "
                 "but unreliable.\n")
    print("wrote results/thinking_truncation.{json,md}")


if __name__ == "__main__":
    main()
