# Box↔Phone Parity Gate — RESULT (first run, 2026-08-04)

Run: `python parity_check.py results/lap_locomo100_e2b_vr_parity results_phone/phone_locomo100_e2b_vectorrag_n100`

Both halves verified config-matched (NOT confounded by the prefix blocker of 2026-07-25):
`embed_prefix=nomic, prompt_style=v3, top_k=10, num_ctx=4096`, gemma4:e2b vectorrag, n=100.

## Verdict: **FAIL** (pre-registered rule: agreement ≥90% AND McNemar p>0.05)

```
paired on stable_qid: 100
judged-label agreement: 83/100 = 83.0%   Cohen's kappa = 0.621
  box accuracy 30.0%   phone accuracy 37.0%
  discordant: box-only-correct 5, phone-only-correct 12  (total 17)  McNemar exact p = 0.1435
  resolution: one flipped row moves agreement by 1.0 pts
  VERDICT: FAIL — report phone-measured accuracy; treat box join as sensitivity only
  note: p > 0.05 is failure to detect systematic bias at n_disc=17, not proof of its absence.
exact answer-string agreement: 40/100 = 40.0%
```

## Reading
- Agreement 83% is **below** the 90% pre-registration threshold → the gate does **not** license
  importing box accuracy as phone accuracy. The McNemar half passes (p=0.14, and with n_disc=17
  the test HAS power), so there is **no evidence of *systematic* directional bias** — but the
  disagreement is large and roughly two-sided (phone-only-correct 12 vs box-only-correct 5).
- Direction is notable: **phone (quantized llama.cpp) is not worse** — 37% vs box 30%. So the
  quantized on-device reader does not degrade accuracy here; it produces *different* answers
  (exact-string agreement only 40%), which the pre-registered gate correctly refuses to treat
  as interchangeable.
- Cohen's κ=0.62 is "substantial" agreement but below the strict operational bar we set.

## Consequence for the golden metric
The 26.6× correct-per-kJ headline joined **box** accuracy to **phone** energy. That join is
**no longer licensed** for the accuracy numerator. Options, in order of preference:
1. **Phone-measured accuracy only.** For the vr E2B n=100 cell we now HAVE a direct phone
   accuracy (37%). Report the golden metric on phone-measured numerators wherever a phone cell
   exists; treat box accuracy as a sensitivity bound, not the primary number.
2. fc has no phone n=100 cell (11–20 h/cell). Use phone fc n=20 with the n-dependence caveat,
   or report the ratio as bounded, NOT as a box-import.
3. Keep the box join only as a labelled sensitivity analysis, never as the headline.

## Slide framing (have this ready — do NOT improvise at the podium)
"We pre-registered a parity gate to decide whether S23 (llama.cpp, quantized) accuracy could be
treated as identical to box (Ollama) accuracy. It did not pass the 90% bar, so we report
**phone-measured accuracy directly** and use the box only as a sensitivity check. Notably the
phone was not less accurate — the quantized reader simply produces different wordings that our
strict gate declines to equate."

## Open follow-ups
- Recompute `golden_metric.py` numerators from phone-measured accuracy; re-derive the vr/fc ratio
  and its error bar under the phone-only accuracy policy.
- Decide fc numerator (phone n=20 vs disclosed bound).
- This does NOT threaten the energy measurements themselves (energy is a phone-native quantity).
