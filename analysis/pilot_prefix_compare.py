"""Compare two runs of run_memory_eval.py that differ only in --embed-prefix.

Usage:
    python -m analysis.pilot_prefix_compare results/<run_noprefix> results/<run_prefix>

Pairs questions by qid and reports:
  - retrieval overlap@k between the two arms (primary pilot decision metric:
    pure retrieval effect, no judge noise)
  - lexical accuracy deltas (f1 / em / contains) — sanity only, never "accuracy"
  - 70b judge accuracy delta when judged.jsonl exists in both run dirs
    (judge standard: all reported accuracy = judge_correct)

Decision guide printed at the end: mean overlap >= ~0.9 with no judged flips
=> prefixes don't materially change retrieval; report as robustness footnote.
Lower overlap or judged flips => schedule the full prefixed reruns.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_rows(run_dir: Path) -> dict[str, dict]:
    """qid -> row from the main per-question jsonl (skip judged/diagnostic sidecars)."""
    rows: dict[str, dict] = {}
    for jp in sorted(run_dir.glob("*.jsonl")):
        if jp.name == "judged.jsonl" or jp.name.endswith("_failures.jsonl"):
            continue
        with jp.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "qid" in row:
                    rows[str(row["qid"])] = row
    if not rows:
        sys.exit(f"no result rows found in {run_dir}")
    return rows


def load_judged(run_dir: Path) -> dict[str, bool] | None:
    jp = run_dir / "judged.jsonl"
    if not jp.exists():
        return None
    out: dict[str, bool] = {}
    with jp.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "qid" in row and row.get("judge_correct") is not None:
                out[str(row["qid"])] = bool(row["judge_correct"])
    return out or None


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    dir_a, dir_b = Path(sys.argv[1]), Path(sys.argv[2])
    rows_a, rows_b = load_rows(dir_a), load_rows(dir_b)
    qids = sorted(set(rows_a) & set(rows_b))
    if not qids:
        sys.exit("no shared qids between the two runs")
    print(f"paired questions: {len(qids)} "
          f"(A only: {len(set(rows_a) - set(rows_b))}, B only: {len(set(rows_b) - set(rows_a))})")
    print(f"A = {dir_a.name}\nB = {dir_b.name}\n")

    # --- retrieval overlap ---------------------------------------------------
    overlaps: list[float] = []
    missing_idx = 0
    changed: list[tuple[str, float]] = []
    for q in qids:
        ia = rows_a[q].get("ctx_meta", {}).get("kept_turn_indices")
        ib = rows_b[q].get("ctx_meta", {}).get("kept_turn_indices")
        if ia is None or ib is None:
            missing_idx += 1
            continue
        sa, sb = set(ia), set(ib)
        denom = max(len(sa), len(sb), 1)
        ov = len(sa & sb) / denom
        overlaps.append(ov)
        if ov < 1.0:
            changed.append((q, ov))
    if overlaps:
        overlaps_sorted = sorted(overlaps)
        mean_ov = sum(overlaps) / len(overlaps)
        print(f"retrieval overlap@k  (|A intersect B| / k, n={len(overlaps)})")
        print(f"  mean {mean_ov:.3f}   min {overlaps_sorted[0]:.3f}   "
              f"median {overlaps_sorted[len(overlaps_sorted) // 2]:.3f}")
        print(f"  questions with identical retrieval: "
              f"{sum(1 for o in overlaps if o == 1.0)}/{len(overlaps)}")
        if changed:
            print("  changed retrieval (qid: overlap):")
            for q, ov in sorted(changed, key=lambda x: x[1])[:20]:
                print(f"    {q}: {ov:.2f}")
    if missing_idx:
        print(f"  [warn] {missing_idx} pairs lack kept_turn_indices "
              f"(runs predate the logging patch?)")
    print()

    # --- lexical sanity metrics ---------------------------------------------
    def mean(rows: dict[str, dict], key: str) -> float:
        vals = [rows[q].get(key) for q in qids]
        vals = [v for v in vals if isinstance(v, (int, float))]
        return sum(vals) / len(vals) if vals else float("nan")

    print("lexical (sanity only, not accuracy-of-record):")
    for key in ("f1", "em", "contains"):
        a, b = mean(rows_a, key), mean(rows_b, key)
        print(f"  {key:9s} A {a:.3f}   B {b:.3f}   delta {b - a:+.3f}")
    print()

    # --- judge accuracy (method of record) -----------------------------------
    ja, jb = load_judged(dir_a), load_judged(dir_b)
    if ja and jb:
        shared = [q for q in qids if q in ja and q in jb]
        acc_a = sum(ja[q] for q in shared) / len(shared)
        acc_b = sum(jb[q] for q in shared) / len(shared)
        flips_ab = [q for q in shared if ja[q] and not jb[q]]  # correct -> wrong
        flips_ba = [q for q in shared if jb[q] and not ja[q]]  # wrong -> correct
        print(f"70b judge accuracy (n={len(shared)}):")
        print(f"  A {acc_a:.3f}   B {acc_b:.3f}   delta {acc_b - acc_a:+.3f}")
        print(f"  flips A-correct->B-wrong: {len(flips_ab)} {flips_ab}")
        print(f"  flips A-wrong->B-correct: {len(flips_ba)} {flips_ba}")
    else:
        print("judge labels missing in one or both runs — run judge.py on both dirs "
              "for accuracy-of-record (lexical deltas above are sanity only).")
    print()
    if overlaps:
        mean_ov = sum(overlaps) / len(overlaps)
        if mean_ov >= 0.9:
            print("VERDICT HINT: retrieval nearly unchanged -> robustness-footnote "
                  "territory (confirm with judge flips above).")
        else:
            print("VERDICT HINT: retrieval materially changed -> schedule full "
                  "prefixed reruns (vr + graph dense cells).")


if __name__ == "__main__":
    main()
