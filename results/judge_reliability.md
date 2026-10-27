# Judge reliability — llama3.1:70b-instruct-q4_K_M, temperature=0

The judge is scored on two regimes, because they give different answers and the thesis lives
in the second one.

## Regime 1 — WITHIN-SESSION (3 reps, back-to-back, same warm server process)

`judge_reliability.py --reps 3`. Each rep is a separate `judge.py` invocation, but all three
hit the same already-loaded 70b instance within a few minutes of each other.

| run | rows | unstable | flip rate | acc range (pts) |
|---|---|---|---|---|
| lap_locomo200_e2b_vr_prefix | 200 | 0 | 0.0% | 0.0 |
| lap_locomo200_e2b_graphrag_v3_prefix | 200 | 0 | 0.0% | 0.0 |
| lap_locomo100_e2b_vr_prefix_k10 | 100 | 0 | 0.0% | 0.0 |

**0 / 500 row-judgings flipped.** Against a stable, warm server, `temperature=0` judging is
exactly reproducible. Per-rep accuracies were 36.0 / 21.5 / 30.0 — identical to the banked
values, so this also re-confirms those three cells.

## Regime 2 — ACROSS-SESSION (same config judged days apart, server restarted between)

`lap_locomo100_e2b_vr_prefix_k10` (judged 2026-07-25) and `lap_locomo100_e2b_vr_parity`
(judged 2026-07-23) are the same configuration. Re-verified from the banked `judged.jsonl`
files on 2026-07-29:

```
common = 100 rows | prediction mismatches = 0 | verdict flips = 1
FLIP  locomo_conv-30_c1_1
  gold "By the water, with natural light and Marley flooring"
  pred "A room with a view of the ocean and natural light."
  k10 -> 1 , parity -> 0
accuracy 30.0% vs 29.0%
```

The reader is greedy and reproduced byte-identically (0/100 prediction mismatches), so the
1-point gap is entirely the judge. **1 / 100 row-judgings flipped.**

## Why the two regimes differ, and which one to report

`judge.py` sends `temperature: 0` (line 162), so this is not a sampling setting. It is
GPU-level nondeterminism: batching and KV-cache layout change the floating-point reduction
order, so greedy argmax can flip on a near-tie. Within one server lifetime that layout is
fixed, which is why Regime 1 is perfectly stable. Across a restart — different resident
models, different batch shapes, different memory layout — it is not. The one row that flipped
is a borderline paraphrase, which is exactly where near-ties live.

**The thesis is in Regime 2.** Cells were judged across many sessions spanning weeks, on a box
whose resident model set changed constantly. Regime 1's 0.0% therefore does NOT license
interpreting arbitrarily small differences.

**Reporting rule.**
1. Report the noise floor as **≈ ±1 pt at n = 100** (1 flipped verdict in 100), measured
   across sessions. A single-cell point estimate is one draw, not a fixed value.
2. Unpaired differences at or below that floor are not interpretable, regardless of McNemar
   p on a single judging pass.
3. **Paired McNemar tests are unaffected** when both arms were scored in the same judging
   pass — Regime 1 shows the within-pass comparison is exact.
4. The cheapest way to make any single comparison immune is to re-judge both arms together;
   `judge_reliability.py` re-judges into a scratch dir and never touches banked results.

## Limitation of this measurement

Regime 2 rests on n = 1 duplicate-config pair (100 rows) found opportunistically in banked
results, not on a designed replication. `judge_reliability.py` measures Regime 1 only — it
cannot restart the server between reps. A proper across-session estimate would re-judge the
same cells in several sessions separated by server restarts; until that exists, treat ±1 pt
at n = 100 as a single-observation estimate of the floor, not a tight bound.

Do not report this as a defect. Every LLM-as-judge result carries this property and almost
none characterise it; reporting it is a rigor win and it sets the resolution below which
accuracy differences must not be read.
