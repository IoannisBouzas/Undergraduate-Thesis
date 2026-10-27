"""Box-vs-phone quality parity check: do the phone and workstation give the same answers?

The thesis measures QUALITY on the box (Ollama) and RESOURCES on the phone
(llama.cpp Q4_K_M). Joining them is only legitimate if the two readers answer
the same questions the same way. This script pairs a box run and a phone run
on `stable_qid` — NOT `qid`, which embeds a limit-dependent sampling counter —
and reports agreement.

Usage:
    python -m energy.parity_check results/<box_run> results_phone/<phone_run>

Reports: judged-label agreement, Cohen's kappa, McNemar on disagreements
(tests systematic bias, not just noise), and exact-string answer agreement.
Decision rule (pre-registered): label agreement >= 90% and McNemar p > 0.05
licenses joining box accuracy to phone energy; below that, report phone-measured
accuracy only and treat the box join as a sensitivity analysis.

IMPORTANT — judge both arms in the SAME judging session before running this.
The 70b judge is exactly reproducible within one warm server process but flips ~1
verdict per 100 rows ACROSS sessions (results/judge_reliability.md). A box arm judged
weeks before the phone arm therefore carries ~1 pt of pure judge noise against a 90%
threshold, which this script would attribute to quantization. Re-judge the box cell
back-to-back with the phone cell, then run this.

The pre-registered decision rule is NOT modified here. Added 2026-07-29: the script now
also reports how many pairs were dropped as unscored and how many discordant pairs the
McNemar test actually had, and annotates the verdict when that count is too small for
p > 0.05 to mean anything. Annotation only — the PASS/FAIL rule is unchanged.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# Windows consoles default to cp1252 and mangle the em dashes in the verdict strings
# (same guard as stats_report.py).
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def is_prediction_file(path: Path) -> bool:
    """True when the file's rows have the shape of harness predictions.

    Allow-list on CONTENT, not on filename. A deny-list of known sidecars is what let
    `completeness.jsonl` — which carries qid and judge_correct but no question/pred, and
    sorts alphabetically before every real prediction filename — get picked as the
    prediction file here and mis-ingested in judge.py (fixed there 2026-07-28). Any future
    analysis script that drops a .jsonl into a run dir re-arms a deny-list; this does not.
    """
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                return "question" in row and "pred" in row
    except (OSError, json.JSONDecodeError):
        return False
    return False  # empty file


def load(run_dir: Path) -> dict[str, dict]:
    """stable_qid -> row (judged.jsonl preferred, falls back to raw rows)."""
    src = run_dir / "judged.jsonl"
    if not src.exists():
        cands = [p for p in sorted(run_dir.glob("*.jsonl")) if is_prediction_file(p)]
        if not cands:
            sys.exit(f"no prediction rows in {run_dir} "
                     f"(looked for *.jsonl rows carrying both 'question' and 'pred')")
        src = cands[0]
    out, no_stable = {}, 0
    for line in src.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        sq = r.get("stable_qid")
        if not sq:
            no_stable += 1
            continue
        out[sq] = r
    if no_stable:
        print(f"WARNING: {run_dir.name}: {no_stable} rows lack stable_qid "
              f"(produced before the 2026-07-22 fix) — rerun that side to enable parity.",
              file=sys.stderr)
    return out


# Below this many discordant pairs the exact two-sided McNemar test cannot produce
# p <= 0.05 for ANY split (most extreme case k=0 gives p = 2 * 0.5**n_disc; n=5 -> 0.0625,
# n=6 -> 0.0312). So "p > 0.05" below 6 carries no information.
MIN_DISCORDANT_FOR_POWER = 6


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, sum(math.comb(n, i) for i in range(k + 1)) * 0.5 ** n * 2)


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    box, phone = Path(sys.argv[1]), Path(sys.argv[2])
    B, P = load(box), load(phone)
    common = sorted(set(B) & set(P))
    if not common:
        sys.exit("no shared stable_qid — did both sides run the same dataset with the "
                 "post-2026-07-22 harness?")
    print(f"box   = {box.name}  ({len(B)} rows)")
    print(f"phone = {phone.name}  ({len(P)} rows)")
    print(f"paired on stable_qid: {len(common)}\n")

    both = [(B[q].get("judge_correct"), P[q].get("judge_correct")) for q in common]
    scored = [(b, p) for b, p in both if b is not None and p is not None]
    n_dropped = len(both) - len(scored)
    if n_dropped:
        print(f"WARNING: {n_dropped}/{len(both)} pairs dropped as unscored (null judge_correct "
              f"on one or both sides) — run rejudge_missing.py before trusting the verdict.",
              file=sys.stderr)
    if scored:
        agree = sum(1 for b, p in scored if bool(b) == bool(p))
        n = len(scored)
        bc = sum(1 for b, p in scored if b and not p)   # box right, phone wrong
        cb = sum(1 for b, p in scored if p and not b)
        pa = agree / n
        pb = sum(bool(b) for b, _ in scored) / n
        pp = sum(bool(p) for _, p in scored) / n
        pe = pb * pp + (1 - pb) * (1 - pp)
        kappa = (pa - pe) / (1 - pe) if pe != 1 else float("nan")
        p_mc = mcnemar_exact(bc, cb)
        n_disc = bc + cb
        print(f"judged-label agreement: {agree}/{n} = {pa*100:.1f}%   Cohen's kappa = {kappa:.3f}")
        print(f"  box accuracy {pb*100:.1f}%   phone accuracy {pp*100:.1f}%")
        print(f"  discordant: box-only-correct {bc}, phone-only-correct {cb}  (total {n_disc})  "
              f"McNemar exact p = {p_mc:.4f}")
        print(f"  resolution: n={n} scored pairs, so one flipped row moves agreement by "
              f"{100/n:.1f} pts")
        verdict = ("PASS — joining box accuracy to phone energy is licensed"
                   if pa >= 0.90 and p_mc > 0.05 else
                   "FAIL — report phone-measured accuracy; treat box join as sensitivity only")
        print(f"  VERDICT: {verdict}")
        # Power annotation. The exact two-sided test is p = 2 * 0.5**n_disc at its most
        # extreme (k=0), so with fewer than 6 discordant pairs p > 0.05 is arithmetically
        # guaranteed no matter how lopsided the split is — the test CANNOT reject. Passing
        # the McNemar half is then vacuous, not evidence of parity. Reported, not enforced:
        # the pre-registered rule stands as written and the verdict above is unmodified.
        if n_disc < MIN_DISCORDANT_FOR_POWER:
            print(f"  ** POWER WARNING: only {n_disc} discordant pairs. Exact McNemar cannot "
                  f"reach p<=0.05 below {MIN_DISCORDANT_FOR_POWER}, so 'p > 0.05' here means "
                  f"'no data', not 'no bias'. Rest the verdict on the agreement rate.")
        elif p_mc > 0.05:
            print(f"  note: p > 0.05 is failure to detect systematic bias at n_disc={n_disc}, "
                  f"not proof of its absence.")
    else:
        print("no judged labels on one/both sides — run judge.py on both, then rerun.")

    exact = sum(1 for q in common
                if (B[q].get("pred") or "").strip().lower() == (P[q].get("pred") or "").strip().lower())
    print(f"\nexact answer-string agreement: {exact}/{len(common)} = {exact/len(common)*100:.1f}% "
          f"(strict; quantization is expected to change wording without changing correctness)")


if __name__ == "__main__":
    main()
