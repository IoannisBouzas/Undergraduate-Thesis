#!/usr/bin/env python3
"""One reproducible statistics report for all headline comparisons.

Per run: judge accuracy with Wilson 95% CI.
Per pair (paired by qid on judged.jsonl): exact McNemar test, discordant counts,
odds ratio, and Cohen's g effect size.

--------------------------------------------------------------------------------------
MULTIPLE-COMPARISONS FAMILIES (fixed 2026-08-13)
--------------------------------------------------------------------------------------
The pairs are split into two families:

  CONFIRMATORY (Holm-corrected as one family) — called the "primary" family in the thesis.
               It was designated during the analysis, after results were available, so it
               is NOT pre-registered. Only the parity threshold and the two SPE criteria were
               fixed before the runs that test them.
  EXPLORATORY  (everything else) — raw p only, reported and labelled as exploratory.

Every pair below points at a RECORD cell (prefixed retrieval, prompt_style v3), except the
prefix-repair pairs, whose entire purpose is to compare the old pipeline against the new.

CLUSTER DEPENDENCE. LoCoMo n=200 is 10 conversations x ~20 questions, so its rows are not
independent; LongMemEval strat60 is 60 sessions x 1 question and has no such structure.
Confirmatory LoCoMo pairs therefore also carry a conversation-cluster bootstrap CI from
cluster_uncertainty.py. Measured design effects run ~1.1-1.6 on the paired differences
(one outlier at 3.5 on fc_offprompt), so the item-level intervals are optimistic by roughly
15-25% on the SE. No confirmatory claim changes sign or loses separation from zero under
clustering; two exploratory ones (thinking, k10-vs-k5) stop being separable, which is what
Holm already said about them.

Usage: python -m analysis.stats_report                # report to stdout + results/stats_report.txt
       python -m analysis.stats_report --pairs a b    # ad-hoc single pair (dirs under results/)
       python -m analysis.stats_report --no-cluster   # skip the bootstrap (fast)
"""
import json, os, math, argparse, sys, glob, random
from itertools import count

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# =====================================================================================
# CONFIRMATORY FAMILY (the thesis's "primary" family) — Holm-corrected as one family.
# Fixed 2026-08-13, after results were available: not pre-registered. Adding a pair here
# changes the correction; do it deliberately.
# All arms are record cells (prefixed retrieval, prompt_style v3, 70b judge).
# =====================================================================================
CONFIRMATORY = [
    # 1-2. the strategy gap: retrieval beats stuffing the window, both sizes
    ("lap_locomo200_e2b_vr_prefix", "lap_locomo200_e2b_fullcontext"),
    ("lap_locomo200_e4b_vr_prefix", "lap_locomo200_e4b_fullcontext"),
    # 3-4. retrieval beats the SLM-extracted graph, both sizes
    ("lap_locomo200_e2b_vr_prefix", "lap_locomo200_e2b_graphrag_v3_prefix"),
    ("lap_locomo200_e4b_vr_prefix", "lap_locomo200_e4b_graphrag_v3_prefix"),
    # 5. scale at fixed strategy — the other half of "strategy beats scale"
    ("lap_locomo200_e4b_vr_prefix", "lap_locomo200_e2b_vr_prefix"),
    # 6. causal ladder: does appending the source turn to each triple close the gap?
    ("lap_locomo200_e2b_graphrag_v4_prefix", "lap_locomo200_e2b_graphrag_v3_prefix"),
    # 7. SPE falsification — the pre-registered negative result (Ch 9)
    ("lap_locomo200_e2b_spe2fix_gap_b370", "lap_locomo200_e2b_graphrag_v4_prefix"),
    # 8. cross-benchmark: does the strategy gap transfer to 500-turn LME haystacks?
    #    (LME strat60 = 60 sessions x 1 question => unclustered, item-level CI is correct)
    ("lme_strat60_e2b_vr_prefix", "lme_strat60_e2b_fc"),
    # 9. cross-family: is the ordering a Gemma artifact?
    ("lap_locomo200_q35_4b_vr_prefix", "lap_locomo200_q35_4b_fullcontext"),
]

# 10th confirmatory test, computed elsewhere: the box<->phone parity gate.
# It joins results/ to results_phone/ on stable_qid, so it cannot be a pair in this list.
# DELIBERATELY EXCLUDED FROM THE HOLM CORRECTION: it is an equivalence-style gate where the
# desired outcome is a LARGE p (no detectable directional bias). Holm inflates p, which for
# this test moves it toward the outcome we want — i.e. correcting it would make the gate
# easier to pass, not harder. It is reported uncorrected, with its own pre-registered rule
# (agreement >= 90% AND McNemar p > 0.05; result 2026-08-04: FAIL on the agreement half).
EXTERNAL_CONFIRMATORY = {
    "name": "box<->phone parity gate (parity_check.py, 2026-08-04)",
    "source": "results/parity_check_2026-08-04.md",
    "result": "agreement 83/100 = 83.0% (kappa 0.621), box 30.0% vs phone 37.0%, "
              "discordant 5/12, McNemar exact p = 0.1435 -> FAIL (agreement half)",
}

# =====================================================================================
# EXPLORATORY — raw p only, NO correction. Sweeps, controls, robustness grids, ablations.
# These support mechanism discussion; they must never be written up as confirmatory.
# =====================================================================================
EXPLORATORY = [
    # --- prefix repair (old pipeline -> record pipeline). The one place an unprefixed arm
    #     is correct on purpose: the comparison IS old-vs-new. ---
    ("lap_locomo200_e2b_vr_prefix", "lap_locomo200_e2b_vectorrag"),
    ("lap_locomo200_e4b_vr_prefix", "lap_locomo200_e4b_vectorrag"),
    ("lap_locomo200_e2b_graphrag_v3_prefix", "lap_locomo200_e2b_graphrag_v3"),
    ("lap_locomo200_e2b_graphrag_v4_prefix", "lap_locomo200_e2b_graphrag_v4"),
    ("lap_locomo200_e4b_graphrag_v3_prefix", "lap_locomo200_e4b_graphrag_v3"),
    ("lap_locomo200_e4b_graphrag_v4_prefix", "lap_locomo200_e4b_graphrag_v4"),
    ("lme_strat60_e2b_vr_prefix", "lme_strat60_e2b_vr"),
    ("lme_strat60_e4b_vr_prefix", "lme_strat60_e4b_vr"),
    # --- rest of the strategy ordering, on the record pipeline ---
    ("lme_strat60_e4b_vr_prefix", "lme_strat60_e4b_fc"),
    ("lap_locomo200_e2b_vr_prefix", "lap_locomo200_e2b_graphrag_v4_prefix"),
    ("lap_locomo200_e4b_vr_prefix", "lap_locomo200_e4b_graphrag_v4_prefix"),
    # --- graph ladder v2 -> v3 -> v4, prefixed throughout ---
    ("lap_locomo200_e2b_graphrag_v3_prefix", "lap_locomo200_e2b_graphrag_v2_prefix"),
    ("lap_locomo200_e4b_graphrag_v4_prefix", "lap_locomo200_e4b_graphrag_v3_prefix"),
    # --- SPE family (the falsified algorithm; #7 above carries the headline) ---
    # repair effect: does fixing the two audit defects move the answer? (both scales)
    ("lap_locomo200_e2b_spe2fix_gap_b370", "lap_locomo200_e2b_spe2_gap_b370"),
    ("lap_locomo200_e4b_spe2fix_gap_b370", "lap_locomo200_e4b_spe2_gap_b370"),
    # matched-budget head-to-head at the second scale (E2B is confirmatory #7)
    ("lap_locomo200_e4b_spe2fix_gap_b370", "lap_locomo200_e4b_graphrag_v4_prefix"),
    # tight-budget arm = pre-registered criterion 2 ("> triple-only under ~600 real tokens"),
    # record pipeline, both scales. Added 2026-08-29: the criterion was
    # registered but had never been given a paired test.
    ("lap_locomo200_e2b_spe2fix_gap_b240", "lap_locomo200_e2b_graphrag_v3_prefix"),
    ("lap_locomo200_e4b_spe2fix_gap_b240", "lap_locomo200_e4b_graphrag_v3_prefix"),
    # --- SPE on the PRE-CORRECTION pipeline (added 2026-08-29). Both arms of every pair
    #     below are unprefixed, so each comparison is internally matched; they must never be
    #     pooled with the prefixed record cells above. These carry the branch of the Ch 7
    #     argument where the selector DOES mix representations (~1 expansion/query) — the
    #     repaired selector expands nothing, so the corrected cells alone would test only one
    #     corner of the pre-registered hypothesis.
    # criterion 1, both selector generations, matched to expand-all
    ("lap_locomo200_e2b_spe_b370", "lap_locomo200_e2b_graphrag_v4"),
    ("lap_locomo200_e2b_spe2_gap_b370", "lap_locomo200_e2b_graphrag_v4"),
    ("lap_locomo200_e4b_spe2_gap_b370", "lap_locomo200_e4b_graphrag_v4"),
    # criterion 2, both selector generations, against triple-only
    ("lap_locomo200_e2b_spe_b240", "lap_locomo200_e2b_graphrag_v3"),
    ("lap_locomo200_e2b_spe2_gap_b240", "lap_locomo200_e2b_graphrag_v3"),
    ("lap_locomo200_e4b_spe2_gap_b240", "lap_locomo200_e4b_graphrag_v3"),
    # does the zero-parameter loss-aware utility beat the hand-weighted one at equal budget?
    ("lap_locomo200_e2b_spe2_gap_b370", "lap_locomo200_e2b_spe_b370"),
    ("lap_locomo200_e2b_spe2_gap_b240", "lap_locomo200_e2b_spe_b240"),
    # --- reader x graph crossover (added 2026-08-30). Both new cells are query-time only:
    #     the frozen v3_prefix graph caches were reused via --cache-dir (ingest_seconds 0.0,
    #     extraction_stats identical to the source cell), so the ONLY thing that changes across
    #     the 2x2 is which model extracted the graph and which model reads it. Built to answer
    #     "is the E4B-over-E2B v3 gap a graph effect or a reader effect?", which the two
    #     diagonal cells alone cannot separate.
    # graph effect, reader held fixed:
    ("lap_locomo200_e2bread_e4bgraph_v3_prefix", "lap_locomo200_e2b_graphrag_v3_prefix"),
    ("lap_locomo200_e4b_graphrag_v3_prefix", "lap_locomo200_e4bread_e2bgraph_v3_prefix"),
    # reader effect, graph held fixed:
    ("lap_locomo200_e4bread_e2bgraph_v3_prefix", "lap_locomo200_e2b_graphrag_v3_prefix"),
    ("lap_locomo200_e4b_graphrag_v3_prefix", "lap_locomo200_e2bread_e4bgraph_v3_prefix"),
    # --- reasoning grid, prefixed baselines (the 2026-07-26/30 re-runs) ---
    ("lap_locomo200_e2b_vr_think_prefix", "lap_locomo200_e2b_vr_prefix"),
    ("lap_locomo200_e4b_vr_think_prefix", "lap_locomo200_e4b_vr_prefix"),
    ("lap_locomo200_e2b_grv3_think_prefix", "lap_locomo200_e2b_graphrag_v3_prefix"),
    ("lap_locomo200_e4b_grv3_think_prefix", "lap_locomo200_e4b_graphrag_v3_prefix"),
    ("lme_strat60_e2b_vr_think_prefix", "lme_strat60_e2b_vr_prefix"),
    ("lme_strat60_e4b_vr_think_prefix", "lme_strat60_e4b_vr_prefix"),
    # --- budget-matched thinking controls ---
    # For fc ONLY, max_answer_tokens shrinks the history budget (num_ctx-128-cap), so the
    # 64-token baseline is NOT the right reference for a 768-token thinking run. The
    # budget768 control holds context constant and isolates thinking.
    ("lap_locomo200_e2b_fc_think", "lap_locomo200_e2b_fc_budget768"),    # CLEAN thinking effect
    ("lap_locomo200_e2b_fc_budget768", "lap_locomo200_e2b_fullcontext"), # pure context-size effect
    ("lap_locomo200_e4b_fc_think", "lap_locomo200_e4b_fc_budget768"),    # control landed 2026-08
    ("lap_locomo200_e4b_fc_budget768", "lap_locomo200_e4b_fullcontext"),
    # retrieval-based modes are invariant to the cap (verified 200/200 identical predictions),
    # so their think-vs-baseline comparisons were never confounded; kept for completeness
    ("lap_locomo200_e2b_vr_budget768", "lap_locomo200_e2b_vr_prefix"),
    ("lap_locomo200_e2b_grv3_budget768", "lap_locomo200_e2b_graphrag_v3_prefix"),
    # --- prompt robustness (official LoCoMo prompt vs qa_short_answer_v3), both sizes ---
    ("lap_locomo200_e2b_vr_offprompt", "lap_locomo200_e2b_vr_prefix"),
    ("lap_locomo200_e2b_grv3_offprompt", "lap_locomo200_e2b_graphrag_v3_prefix"),
    ("lap_locomo200_e2b_fc_offprompt", "lap_locomo200_e2b_fullcontext"),
    ("lap_locomo200_e4b_vr_offprompt", "lap_locomo200_e4b_vr_prefix"),
    ("lap_locomo200_e4b_grv3_offprompt", "lap_locomo200_e4b_graphrag_v3_prefix"),
    ("lap_locomo200_e4b_fc_offprompt", "lap_locomo200_e4b_fullcontext"),
    # --- sampling control ---
    ("lap_locomo200_e2b_vr_temp1", "lap_locomo200_e2b_vr_prefix"),
    # --- max_triples sweep, now prefixed end to end (mt4 / mt12=grv3_prefix / mt24) ---
    ("lap_locomo200_e2b_gr_mt4_prefix", "lap_locomo200_e2b_graphrag_v3_prefix"),
    ("lap_locomo200_e2b_gr_mt24_prefix", "lap_locomo200_e2b_graphrag_v3_prefix"),
    # graph-depth lever extended to a second Gemma reader and to Qwen (2026-09-03).
    # Each pair holds the reader's own graph fixed; only query-time depth changes.
    ("lap_locomo200_e4b_gr_mt4_prefix", "lap_locomo200_e4b_graphrag_v3_prefix"),
    ("lap_locomo200_e4b_gr_mt24_prefix", "lap_locomo200_e4b_graphrag_v3_prefix"),
    ("lap_locomo200_q35_4b_gr_mt4_prefix", "lap_locomo200_q35_4b_graphrag_v3_prefix"),
    ("lap_locomo200_q35_4b_gr_mt24_prefix", "lap_locomo200_q35_4b_graphrag_v3_prefix"),
    # Does more retrieval rescue an extraction-starved graph? 0.8b/2b fail ~4 batches in 5, so
    # a null here isolates extraction as the binding constraint at those tiers.
    ("lap_locomo200_q35_2b_gr_mt24_prefix", "lap_locomo200_q35_2b_graphrag_v3_prefix"),
    ("lap_locomo200_q35_08b_gr_mt24_prefix", "lap_locomo200_q35_08b_graphrag_v3_prefix"),
    # --- prefixed top-k sweep, both benchmarks, both sizes ---
    # Adjacent-k pairs test where the curve stops paying. All same --limit, so pairable.
    ("lap_locomo100_e2b_vr_prefix_k5",  "lap_locomo100_e2b_vr_prefix_k2"),
    ("lap_locomo100_e2b_vr_prefix_k10", "lap_locomo100_e2b_vr_prefix_k5"),
    ("lap_locomo100_e2b_vr_prefix_k20", "lap_locomo100_e2b_vr_prefix_k10"),
    ("lap_locomo100_e2b_vr_prefix_k50", "lap_locomo100_e2b_vr_prefix_k20"),
    ("lap_locomo100_e4b_vr_prefix_k50", "lap_locomo100_e4b_vr_prefix_k2"),
    # scale at matched k — the "strategy beats scale" contrast along the whole curve
    ("lap_locomo100_e4b_vr_prefix_k2",  "lap_locomo100_e2b_vr_prefix_k2"),
    ("lap_locomo100_e4b_vr_prefix_k10", "lap_locomo100_e2b_vr_prefix_k10"),
    ("lap_locomo100_e4b_vr_prefix_k50", "lap_locomo100_e2b_vr_prefix_k50"),
    # Qwen k-sweep (2026-09-03): same shared embedding store as the Gemma sweeps, so the
    # retrieved turns are byte-identical at each depth and only the reader differs.
    ("lap_locomo100_q35_4b_vr_prefix_k5",  "lap_locomo100_q35_4b_vr_prefix_k2"),
    ("lap_locomo100_q35_4b_vr_prefix_k10", "lap_locomo100_q35_4b_vr_prefix_k5"),
    ("lap_locomo100_q35_4b_vr_prefix_k20", "lap_locomo100_q35_4b_vr_prefix_k10"),
    ("lap_locomo100_q35_4b_vr_prefix_k50", "lap_locomo100_q35_4b_vr_prefix_k20"),
    ("lap_locomo100_q35_4b_vr_prefix_k50", "lap_locomo100_q35_4b_vr_prefix_k2"),
    # cross-family at matched depth: reader effect with the evidence held identical
    ("lap_locomo100_q35_4b_vr_prefix_k10", "lap_locomo100_e2b_vr_prefix_k10"),
    ("lap_locomo100_q35_4b_vr_prefix_k50", "lap_locomo100_e2b_vr_prefix_k50"),
    # small-Qwen depth: does the lever survive at 2B and 0.8B, or is there a reader floor?
    ("lap_locomo100_q35_2b_vr_prefix_k50",  "lap_locomo100_q35_2b_vr_prefix_k2"),
    ("lap_locomo100_q35_08b_vr_prefix_k50", "lap_locomo100_q35_08b_vr_prefix_k2"),
    ("lap_locomo100_q35_2b_vr_prefix_k10",  "lap_locomo100_q35_2b_vr_prefix_k2"),
    ("lap_locomo100_q35_08b_vr_prefix_k10", "lap_locomo100_q35_08b_vr_prefix_k2"),
    # LME: does the k lever transfer to 500-turn haystacks?
    ("lme_strat60_e2b_vr_prefix_k5", "lme_strat60_e2b_vr_prefix_k2"),
    ("lme_strat60_e2b_vr_prefix",    "lme_strat60_e2b_vr_prefix_k5"),  # k10 vs k5: E2B saturation
    ("lme_strat60_e4b_vr_prefix",    "lme_strat60_e4b_vr_prefix_k5"),  # k10 vs k5: E4B still climbing
    # LME cross-family transfer (2026-09-03)
    ("lme_strat60_q35_4b_vr_prefix", "lme_strat60_q35_4b_fc"),
    ("lme_strat60_q35_2b_vr_prefix", "lme_strat60_q35_2b_fc"),
    ("lme_strat60_q35_08b_vr_prefix", "lme_strat60_q35_08b_fc"),
    # --- Qwen cross-family, rest of the grid (#9 above carries the headline) ---
    ("lap_locomo200_q35_4b_vr_prefix", "lap_locomo200_q35_4b_graphrag_v3_prefix"),
    ("lap_locomo200_q35_4b_graphrag_v4_prefix", "lap_locomo200_q35_4b_graphrag_v3_prefix"),
    ("lap_locomo200_q35_2b_vr_prefix", "lap_locomo200_q35_2b_fullcontext"),
    ("lap_locomo200_q35_2b_vr_prefix", "lap_locomo200_q35_2b_graphrag_v3_prefix"),
    ("lap_locomo200_q35_08b_vr_prefix", "lap_locomo200_q35_08b_fullcontext"),
    ("lap_locomo200_q35_2b_graphrag_v4_prefix", "lap_locomo200_q35_2b_graphrag_v3_prefix"),
    # --- extraction-budget probes: is the small-model graph floor a budget artifact? ---
    ("lap_locomo200_q35_2b_grv3_extok1024", "lap_locomo200_q35_2b_graphrag_v3_prefix"),
    ("lap_locomo200_q35_08b_grv3_extok1024", "lap_locomo200_q35_08b_graphrag_v3_prefix"),
]

# NOTE: there is deliberately NO flat `PAIRS = CONFIRMATORY + EXPLORATORY` in this module.
# A 65-entry flat list is precisely what a future run would Holm-correct across, silently
# undoing the 2026-08-13 split. Verified 2026-08-13 that no other script imports one
# (test_null_judge_policy.py uses stats_report.load only; thinking_truncation.py has its own).

# Direction notes for pairs whose result is a NON-significant difference. Both are cases
# where failing to reject supports the narrative, which is the most attackable move in the
# section — so the report states the effect size and the direction rather than leaving a
# bare p-value that a reader has to interpret in our favour.
DIRECTION_NOTES = {
    ("lap_locomo200_e4b_vr_prefix", "lap_locomo200_e2b_vr_prefix"):
        "READ AS EFFECT SIZE, NOT AS A NULL. Doubling parameters buys +4.5 pts and does not "
        "separate from zero after correction; the k=2->k=10 retrieval lever buys +11 to +14 pts "
        "on the same benchmark and pipeline. 'Strategy beats scale' compares two measured "
        "magnitudes — do NOT write 'scale does not help'.",
    ("lap_locomo200_e2b_spe2fix_gap_b370", "lap_locomo200_e2b_graphrag_v4_prefix"):
        "FAILED PRE-REGISTERED PREDICTION, NOT AN EQUIVALENCE. SPE-2 is 6.0 pts WORSE than v4 "
        "(27.0 vs 33.0), i.e. the point estimate moves against its own hypothesis. Report as "
        "'SPE did not improve over v4 and trended worse' — do NOT write 'SPE is equivalent to v4'.",
    ("lap_locomo200_e2b_spe2fix_gap_b240", "lap_locomo200_e2b_graphrag_v3_prefix"):
        "PRE-REGISTERED CRITERION 2: NUMERICAL THRESHOLD MET, GAIN INCONCLUSIVE. 23.0% > 22.0% at "
        "545.7 < 600 prompt tokens, but the +1.5 pts over triple-only is not distinguishable from "
        "zero and the same comparison is nominally NEGATIVE at E4B (25.5 vs 27.5). Report as "
        "'passed numerically but statistically inconclusive' — do NOT write 'SPE beats v3 at low "
        "budget'.",
    ("lap_locomo200_e4bread_e2bgraph_v3_prefix", "lap_locomo200_e2b_graphrag_v3_prefix"):
        "READ AS EFFECT SIZE, NOT AS A NULL. Doubling the reader on a FIXED E2B graph buys +0.5 pts "
        "(10 vs 9 discordants); swapping the extractor on a fixed reader buys +4.5 to +5.5 on the "
        "same 2x2. Report as 'the gap is carried by the graph, not the reader' — do NOT write "
        "'reader scale does not matter'. Also do NOT attribute the graph effect to any single "
        "property of the E4B graph (node count, abstention rate, triple count): the crossover "
        "isolates WHICH EXTRACTOR, not WHICH PROPERTY of its output. That needs a further "
        "controlled experiment.",
    ("lap_locomo200_e4b_graphrag_v3_prefix", "lap_locomo200_e2bread_e4bgraph_v3_prefix"):
        "READ AS EFFECT SIZE, NOT AS A NULL. Doubling the reader on a FIXED E4B graph buys +1.5 pts "
        "(10 vs 7 discordants), against +4.5 to +5.5 for swapping the extractor. Same caution as "
        "the sibling reader-effect pair: this bounds the reader contribution, it does not explain "
        "the graph contribution.",
}

def run_limit(d):
    """--limit the run was produced with (None if the summary predates the field)."""
    for p in glob.glob(os.path.join("results", d, "*_summary.json")):
        try:
            return json.load(open(p, encoding="utf-8")).get("config", {}).get("limit")
        except Exception:
            return None
    return None


def check_pairable(a, b):
    """LoCoMo qid embeds a global sampling counter, so the same qid denotes DIFFERENT
    questions under different --limit (verified 2026-07-22). Refuse to pair across
    limits rather than silently compare unrelated questions; use stable_qid for that."""
    la, lb = run_limit(a), run_limit(b)
    if la is not None and lb is not None and la != lb:
        print(f"ERROR: refusing to pair {a} (--limit {la}) with {b} (--limit {lb}): "
              f"LoCoMo qids are limit-dependent. Join on stable_qid instead.", file=sys.stderr)
        return False
    return True


def load(d, key="qid"):
    p = os.path.join("results", d, "judged.jsonl")
    if not os.path.exists(p):
        return None
    out = {}
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            v = r.get("judge_correct")
            if v is None:
                # unscored, not incorrect — excluding drops the pair from McNemar
                print(f"WARNING: {d}: unscored judge row {r.get('qid')} excluded; "
                      f"run rejudge_missing.py", file=sys.stderr)
                continue
            out[r.get(key) or r.get("qid") or r.get("question_id") or r.get("id")] = int(bool(v))
    return out

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 3
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, ctr - hw, ctr + hw

def mcnemar_exact(b, c):
    """Two-sided exact binomial on discordant pairs (b = A-only correct, c = B-only correct)."""
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n) * 2
    return min(1.0, p)

def holm(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj

def cluster_diff_ci(a, b, rng, n_boot=20000):
    """Conversation-cluster bootstrap 95% CI on the paired accuracy difference (A - B).

    Resamples CONVERSATIONS with replacement, keeping every question inside a drawn
    conversation, because within-conversation correlation is what the item-level interval
    ignores. Returns (lo, hi, deff, n_clusters), or None when the cell has no usable
    sample_id (LongMemEval strat60 is 1 question per session, so clustering is a no-op).
    """
    pa = os.path.join("results", a, "judged.jsonl")
    pb = os.path.join("results", b, "judged.jsonl")
    conv, va, vb = {}, {}, {}
    for path, store in ((pa, va), (pb, vb)):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("judge_correct") is None:
                continue
            store[r.get("qid")] = int(bool(r["judge_correct"]))
            if path == pa:
                conv[r.get("qid")] = r.get("sample_id")
    common = sorted(set(va) & set(vb))
    if not common or any(conv.get(q) is None for q in common):
        return None
    units = {}
    for q in common:
        units.setdefault(conv[q], []).append(va[q] - vb[q])
    if len(units) >= len(common):        # one question per cluster => nothing to correct
        return None

    diffs = [va[q] - vb[q] for q in common]
    n = len(diffs)
    d = sum(diffs) / n
    item_se = math.sqrt(sum((x - d) ** 2 for x in diffs) / (n - 1) / n)

    keys = list(units)
    draws = []
    for _ in range(n_boot):
        samp = []
        for _ in keys:
            samp.extend(units[rng.choice(keys)])
        draws.append(sum(samp) / len(samp))
    draws.sort()
    lo, hi = draws[int(0.025 * n_boot)], draws[int(0.975 * n_boot)]
    boot_se = (draws[int(0.84 * n_boot)] - draws[int(0.16 * n_boot)]) / 2
    deff = (boot_se / item_se) ** 2 if item_se else float("nan")
    return lo, hi, deff, len(units)


def analyse(a, b, emit):
    """Compute one pair, or emit a SKIP line and return None."""
    if not check_pairable(a, b):
        emit(f"SKIP {a} vs {b}: different --limit, qids not comparable (see stderr)")
        return None
    A, B = load(a), load(b)
    if A is None or B is None:
        emit(f"SKIP {a} vs {b}: missing judged.jsonl ({'A' if A is None else 'B'})")
        return None
    common = sorted(set(A) & set(B))
    if not common:
        emit(f"SKIP {a} vs {b}: no common qids")
        return None
    ka, na = sum(A[q] for q in common), len(common)
    kb = sum(B[q] for q in common)
    b_only = sum(1 for q in common if A[q] and not B[q])
    c_only = sum(1 for q in common if B[q] and not A[q])
    p = mcnemar_exact(b_only, c_only)
    pa, la, ua = wilson(ka, na)
    pb, lb, ub = wilson(kb, na)
    g = (b_only / (b_only + c_only) - 0.5) if (b_only + c_only) else 0.0  # Cohen's g
    return dict(a=a, b=b, n=na, pa=pa, la=la, ua=ua, pb=pb, lb=lb, ub=ub,
                bo=b_only, co=c_only, p=p, g=g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs=2, default=None)
    ap.add_argument("--no-cluster", action="store_true",
                    help="skip the conversation-cluster bootstrap on confirmatory pairs")
    args = ap.parse_args()
    rng = random.Random(7)

    lines = []
    def emit(s=""):
        lines.append(s); print(s)

    emit("STATS REPORT — judge = llama3.1:70b-instruct-q4_K_M, paired by qid")
    emit("Family split locked 2026-08-13: Holm applies to the CONFIRMATORY family only.")
    emit("=" * 78)

    # --- ad-hoc single pair: no family, no correction ----------------------------------
    if args.pairs:
        r = analyse(args.pairs[0], args.pairs[1], emit)
        if r:
            emit(f"\n{r['a']}  vs  {r['b']}   (n={r['n']} paired)   [AD-HOC, uncorrected]")
            emit(f"  A: {r['pa']*100:5.1f}%  [{r['la']*100:.1f}, {r['ua']*100:.1f}]"
                 f"   B: {r['pb']*100:5.1f}%  [{r['lb']*100:.1f}, {r['ub']*100:.1f}]")
            emit(f"  discordant A+/B- = {r['bo']}, A-/B+ = {r['co']}"
                 f"   McNemar exact p = {r['p']:.4g}   Cohen's g = {r['g']:+.2f}")
        return

    # --- confirmatory family -----------------------------------------------------------
    emit("\n" + "=" * 78)
    emit("CONFIRMATORY FAMILY — Holm-corrected across these tests only")
    emit("=" * 78)
    conf = [r for r in (analyse(a, b, emit) for a, b in CONFIRMATORY) if r]
    adj = holm([r["p"] for r in conf]) if conf else []
    for r, ap_ in zip(conf, adj):
        emit(f"\n{r['a']}  vs  {r['b']}   (n={r['n']} paired)")
        emit(f"  A: {r['pa']*100:5.1f}%  [{r['la']*100:.1f}, {r['ua']*100:.1f}]"
             f"   B: {r['pb']*100:5.1f}%  [{r['lb']*100:.1f}, {r['ub']*100:.1f}]")
        emit(f"  discordant A+/B- = {r['bo']}, A-/B+ = {r['co']}   McNemar exact p = {r['p']:.4g}"
             f"   Holm-adj p = {ap_:.4g}   Cohen's g = {r['g']:+.2f}")
        if not args.no_cluster:
            cc = cluster_diff_ci(r["a"], r["b"], rng)
            if cc is None:
                emit("  cluster: n/a — 1 question per session, item-level CI already correct")
            else:
                lo, hi, deff, nc = cc
                d = r["pa"] - r["pb"]
                verdict = "excludes 0" if not (lo <= 0 <= hi) else "INCLUDES 0"
                emit(f"  cluster ({nc} conversations): diff {d:+.3f}"
                     f"   95% [{lo:+.3f}, {hi:+.3f}] {verdict}   design effect {deff:.2f}")
                # CONSERVATIVE RULE (locked 2026-08-13). Some pairs come out with deff < 1, i.e.
                # the cluster interval is TIGHTER than the item-level one. Checked 2026-08-13 on
                # the scale pair: this is not an artefact of the percentile SE estimator (the
                # bootstrap SD gives deff 0.72 against the percentile spread's 0.67 — both < 1).
                # It reflects genuinely low between-conversation variance in the paired
                # DIFFERENCE, compounded by the known downward bias of a cluster bootstrap at
                # G=10. Either way a robustness check must never manufacture a conclusion the
                # primary test refuses, so where deff < 1 the Holm-adjusted McNemar stays
                # primary and the cluster interval is corroboration only.
                if deff < 1.0:
                    emit("    ^ deff < 1: cluster interval is TIGHTER than item-level (low"
                         " between-conversation variance in the difference, plus known downward"
                         " bias of a cluster bootstrap at G=10). Primary inference stays the"
                         " Holm-adjusted McNemar above; do NOT read this interval as upgrading"
                         " a non-significant pair.")
        note = DIRECTION_NOTES.get((r["a"], r["b"]))
        if note:
            emit(f"  !! {note}")

    emit(f"\n[10th confirmatory test — {EXTERNAL_CONFIRMATORY['name']}]")
    emit(f"  {EXTERNAL_CONFIRMATORY['result']}")
    emit(f"  source: {EXTERNAL_CONFIRMATORY['source']}")
    emit("  Excluded from the Holm correction by design: this is an equivalence-style gate")
    emit("  where a large p is the desired outcome, so correcting it would loosen the gate.")

    # --- exploratory -------------------------------------------------------------------
    emit("\n" + "=" * 78)
    emit("EXPLORATORY — raw p only, NOT corrected, NOT confirmatory evidence")
    emit("=" * 78)
    expl = [r for r in (analyse(a, b, emit) for a, b in EXPLORATORY) if r]
    for r in expl:
        emit(f"\n{r['a']}  vs  {r['b']}   (n={r['n']} paired)")
        emit(f"  A: {r['pa']*100:5.1f}%  [{r['la']*100:.1f}, {r['ua']*100:.1f}]"
             f"   B: {r['pb']*100:5.1f}%  [{r['lb']*100:.1f}, {r['ub']*100:.1f}]")
        emit(f"  discordant A+/B- = {r['bo']}, A-/B+ = {r['co']}"
             f"   McNemar exact p = {r['p']:.4g} (raw)   Cohen's g = {r['g']:+.2f}")
        # Exploratory pairs get direction notes too. Until 2026-08-30 this lookup existed only
        # in the confirmatory loop, so notes attached to exploratory pairs were silently inert
        # — exactly the pairs most in need of one, since a non-significant exploratory result
        # is the easiest thing in the report to over-read.
        note = DIRECTION_NOTES.get((r["a"], r["b"]))
        if note:
            emit(f"  !! {note}")

    emit("\n" + "=" * 78)
    emit("Notes: Wilson 95% CIs (item-level); exact two-sided McNemar on discordants.")
    emit(f"Holm correction across {len(conf)} confirmatory difference tests; the parity gate is")
    emit("the 10th family member and is reported uncorrected for the reason stated above.")
    emit(f"{len(expl)} exploratory pairs carry raw p and must be labelled exploratory in prose.")
    emit("LoCoMo confirmatory pairs also carry a conversation-cluster bootstrap CI (10")
    emit("conversations x ~20 questions); with 10 clusters that interval is coarse by")
    emit("construction — percentile, not BCa. LongMemEval strat60 needs no correction.")
    emit("Rerun after any new judged run: python -m analysis.stats_report")
    os.makedirs("results", exist_ok=True)
    with open("results/stats_report.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\nwrote results/stats_report.txt")

if __name__ == "__main__":
    main()
