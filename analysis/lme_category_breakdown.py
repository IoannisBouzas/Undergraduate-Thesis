"""Per-question-type accuracy breakdown for the LongMemEval runs.

Regroups existing judged.jsonl rows by the dataset's question_type (already stamped on each
row as `category`) - zero model calls. Shows WHERE each memory mode loses: temporal-reasoning,
knowledge-update, multi-session, etc.

Usage: python -m analysis.lme_category_breakdown            # all lme_* run dirs
Writes results/lme_category_table.csv and prints a mode x category matrix.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"

CATEGORY_ORDER = [
    "single-session-user", "single-session-assistant", "single-session-preference",
    "multi-session", "knowledge-update", "temporal-reasoning",
]


def main():
    runs = {}
    for d in sorted(RESULTS.iterdir()):
        if not d.name.startswith("lme_") or not (d / "judged.jsonl").exists():
            continue
        rows = [json.loads(l) for l in (d / "judged.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        by_cat = defaultdict(list)
        for r in rows:
            if r.get("judge_correct") is not None:
                by_cat[r.get("category", "?")].append(r["judge_correct"])
        runs[d.name] = by_cat

    cats = [c for c in CATEGORY_ORDER if any(c in bc for bc in runs.values())]
    extra = sorted({c for bc in runs.values() for c in bc} - set(cats))
    cats += extra

    header = f"{'run':22s}" + "".join(f"{c[:18]:>20s}" for c in cats) + f"{'overall':>10s}"
    print(header)
    print("-" * len(header))
    out_rows = []
    for name, by_cat in runs.items():
        line = f"{name:22s}"
        rec = {"run": name}
        all_scores = []
        for c in cats:
            xs = by_cat.get(c, [])
            all_scores += xs
            cell = f"{sum(xs)}/{len(xs)}" if xs else "-"
            line += f"{cell:>20s}"
            rec[c] = round(sum(xs) / len(xs), 4) if xs else None
            rec[c + "_n"] = len(xs)
        overall = round(sum(all_scores) / len(all_scores), 4) if all_scores else None
        rec["overall"] = overall
        out_rows.append(rec)
        print(line + f"{overall!s:>10s}")

    out = RESULTS / "lme_category_table.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
