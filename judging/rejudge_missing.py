"""Re-judge ONLY rows with judge_correct == null across judged runs.

Nulls appear when the judge call errored or its verdict was unparseable
(judge.py leaves None "for review"). Aggregators must exclude them, but the
right fix is to score them: this script re-calls the judge for exactly those
rows, rewrites judged.jsonl in place, and updates judge_meta.json counters.

Usage (box up at :18080):
    python -m judging.rejudge_missing                    # all results/*/judged.jsonl
    python -m judging.rejudge_missing results/<run> ...  # specific run dirs
    python -m judging.rejudge_missing --dry-run          # list nulls, no judge calls

Judge settings (model/prompt-style/base-url) are read from each run's
judge_meta.json so the provenance stamp stays homogeneous within a run.
"""

from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime, timezone
from pathlib import Path

from judge import build_prompt, call_judge, degenerate_pred, parse_label

DEFAULT_BASE_URL = "http://localhost:18080/v1"


def rejudge_dir(run_dir: Path, dry_run: bool) -> None:
    judged_path = run_dir / "judged.jsonl"
    meta_path = run_dir / "judge_meta.json"
    if not judged_path.exists():
        return
    rows = [json.loads(l) for l in judged_path.open(encoding="utf-8") if l.strip()]
    nulls = [r for r in rows if r.get("judge_correct") is None]
    if not nulls:
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    dataset = meta.get("dataset") or ("longmemeval" if run_dir.name.startswith("lme") else "locomo")
    model = meta.get("judge_model", "llama3.1:70b-instruct-q4_K_M")
    style = meta.get("prompt_style", "official")
    base_url = meta.get("base_url", DEFAULT_BASE_URL)
    max_tokens = meta.get("max_tokens", 256)
    effort = meta.get("reasoning_effort")

    print(f"{run_dir.name}: {len(nulls)} null row(s) -> {[r.get('qid') for r in nulls]}")
    if dry_run:
        return

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    still_null = 0
    for row in nulls:
        if degenerate_pred(row.get("pred", "")):
            # empty/truncated answer (done_reason=length): unambiguously wrong,
            # a chat judge cannot grade it (asks for the missing response)
            label = 0
            row["judge_note"] = "empty_prediction_auto_wrong"
        else:
            prompt = build_prompt(row, dataset, style)
            try:
                content, reasoning = call_judge(base_url, model, prompt, max_tokens, effort)
                label = parse_label(content)
                if label is None:
                    label = parse_label(reasoning)
            except Exception as e:
                print(f"  [judge error] {row.get('qid')}: {e}")
                label = None
        row["judge_correct"] = label
        row["judge_model"] = model
        row["judge_prompt_style"] = style
        row["judge_base_url"] = base_url
        row["judged_at"] = stamp
        row["rejudged_missing"] = True  # provenance: scored later than the main pass
        if label is None:
            still_null += 1
        print(f"  {row.get('qid')}: judge_correct={label}")

    with judged_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    scored = [r for r in rows if r.get("judge_correct") is not None]
    meta.update({
        "n_rows": len(rows),
        "n_scored": len(scored),
        "judge_accuracy": round(sum(r["judge_correct"] for r in scored) / len(scored), 4)
        if scored else None,
        "rejudged_missing_at": stamp,
    })
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  updated: n_scored={meta['n_scored']} judge_accuracy={meta['judge_accuracy']}"
          + (f"  STILL NULL: {still_null}" if still_null else ""))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="*", help="run dirs; default = all results/*/ with judged.jsonl")
    p.add_argument("--dry-run", action="store_true", help="list null rows, no judge calls")
    args = p.parse_args()
    dirs = [Path(d) for d in args.run_dirs] if args.run_dirs else \
        sorted(Path(p).parent for p in glob.glob("results/*/judged.jsonl"))
    for d in dirs:
        rejudge_dir(d, args.dry_run)


if __name__ == "__main__":
    main()
