# Thinking cells — truncation-aware reporting

`--max-answer-tokens 768 --thinking`. A row is *truncated* when it hit the cap (`done_reason: length`) without emitting a parseable answer; such rows are auto-scored 0 by the null-judge policy. Non-thinking runs truncate on 0% of rows.

| pair | trunc | all rows | non-truncated | base acc. on truncated Qs |
|---|---|---|---|---|
| vr E2B LoCoMo | 36/200 (18%) | 36.0 → 41.5 (+5.5) | 42.7 → 50.6 (+7.9) | 5.6% |
| vr E4B LoCoMo | 37/200 (18%) | 40.5 → 43.5 (+3.0) | 46.0 → 53.4 (+7.4) | 16.2% |
| grv3 E2B LoCoMo | 39/200 (20%) | 21.5 → 23.5 (+2.0) | 23.6 → 29.2 (+5.6) | 12.8% |
| grv3 E4B LoCoMo | 69/200 (34%) | 27.5 → 23.5 (-4.0) | 33.6 → 35.9 (+2.3) | 15.9% |
| vr E2B LME | 9/60 (15%) | 48.3 → 58.3 (+10.0) | 54.9 → 68.6 (+13.7) | 11.1% |
| vr E4B LME | 8/60 (13%) | 56.7 → 60.0 (+3.3) | 61.5 → 69.2 (+7.7) | 25.0% |
| fc E2B LoCoMo* | 24/200 (12%) | 9.0 → 4.5 (-4.5) | 9.1 → 5.1 (-4.0) | 8.3% |
| fc E4B LoCoMo* | 7/200 (4%) | 6.5 → 6.0 (-0.5) | 6.2 → 6.2 (+0.0) | 14.3% |

\* fc is compared against its budget-matched control (`lap_locomo200_{e2b,e4b}_fc_budget768`), not the 64-token baseline, because fullcontext couples the answer cap to its history budget. Note that control is not neutral at E4B: `e4b_fc_budget768` scores 6.5% against `e4b_fullcontext`'s 10.5% (raw p = 0.03857), so raising the cap alone cost E4B accuracy and the E4B reasoning delta is read against a baseline that itself moved.

**Reporting rule.** The all-rows delta is the headline — it is the honest cost of reasoning under a fixed budget. The non-truncated delta is the sensitivity analysis isolating reasoning quality from budget exhaustion. Report both.

**Finding.** Base accuracy on the truncated questions is far below base accuracy on the rest: the model reasons longest exactly where it is most lost, then runs out of budget. At an on-device-feasible 768-token cap roughly one question in five yields no answer at all, concentrated on the hardest questions — so reasoning on-device is not merely expensive (+33 s, +132 J per answer measured) but unreliable.
