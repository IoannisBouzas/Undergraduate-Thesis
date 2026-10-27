"""Context-completeness analysis: did retrieval put the gold evidence in front of the reader?

Decomposes end-to-end failure into retrieval failure vs reader failure, offline, judge-free,
from artifacts that every run already wrote (debug_prompt_response.log + judged.jsonl).
No model calls. Inspired by the "measure context completeness first" framing (Zep, 2026),
grounded in the datasets' own evidence labels:

  - LongMemEval: per-turn `has_answer` flags inside `answer_session_ids` sessions.
  - LoCoMo: per-question `evidence` dia_ids -> turn texts.

Metrics per run:
  evidence_recall   = fraction of questions with >=1 evidence-turn text present in the prompt
  answer_in_ctx     = fraction with the (normalized) gold answer string present in the prompt
                      (secondary/lexical - dates and paraphrases legitimately miss)
  acc | evidence    = judge accuracy on questions where evidence WAS in context (reader-bound)
  acc | no evidence = judge accuracy where evidence was NOT in context (retrieval-bound;
                      any hits here are guesses/paraphrase artifacts)

Note on graphrag: its context is triples, not turns, so verbatim evidence-turn recall is
expected ~0 by construction - that IS the finding (representation discards the evidence).
answer_in_ctx is the fairer completeness proxy for graphrag.

Usage:
  python -m analysis.context_completeness results/lme_s50_e2b_vr results/lap_locomo200_e2b_* ...
  python -m analysis.context_completeness --all      # every results/* dir with judged.jsonl + prompt log
Writes per-run completeness.jsonl (per-question rows) and results/completeness_table.csv.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

LME_DATASET = ROOT / "longmemeval_s_cleaned.json"
LOCOMO_DATASET = ROOT / "locomo10.json"

_ws = re.compile(r"\s+")


def norm(s: str) -> str:
    return _ws.sub(" ", (s or "").lower()).strip()


# ---------------------------------------------------------------- evidence maps

def lme_evidence_map():
    """question text (normalized) -> {qid, evidence_turn_texts, answer, category}"""
    data = json.loads(LME_DATASET.read_text(encoding="utf-8"))
    out = {}
    for q in data:
        sid_to_sess = dict(zip(q["haystack_session_ids"], q["haystack_sessions"]))
        ev_turns = []
        for sid in q["answer_session_ids"]:
            sess = sid_to_sess.get(sid, [])
            flagged = [t["content"] for t in sess if t.get("has_answer")]
            # fall back to the whole answer session if nothing is flagged
            ev_turns.extend(flagged or [t["content"] for t in sess])
        out[norm(q["question"])] = {
            "qid": q["question_id"],
            "evidence": [norm(t) for t in ev_turns if t and len(norm(t)) >= 20],
            "answer": str(q["answer"]),
            "category": q["question_type"],
        }
    return out


def locomo_evidence_map():
    """question text (normalized) -> {evidence_turn_texts, answer, category} (per sample)."""
    data = json.loads(LOCOMO_DATASET.read_text(encoding="utf-8"))
    out = {}
    for sample in data:
        dia = {}
        for key, val in sample["conversation"].items():
            if isinstance(val, list):
                for t in val:
                    if isinstance(t, dict) and "dia_id" in t:
                        dia[t["dia_id"]] = t.get("text", "")
        for qa in sample["qa"]:
            ev = [norm(dia[d]) for d in qa.get("evidence", []) if d in dia]
            out[(sample["sample_id"], norm(str(qa["question"])))] = {
                "evidence": [e for e in ev if len(e) >= 10],
                "answer": str(qa.get("answer", qa.get("adversarial_answer", ""))),
                "category": qa.get("category"),
            }
    return out


# ---------------------------------------------------------------- prompt log parser

def parse_prompt_log(path: Path):
    """-> {question text (normalized): prompt text (normalized)}, last occurrence wins
    (a resumed run re-logs its first questions)."""
    prompts = {}
    question, buf, in_prompt = None, [], False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("=== Q: ") and line.rstrip().endswith("==="):
            if question is not None and buf:
                prompts[norm(question)] = norm("\n".join(buf))
            question = line[len("=== Q: "):].rstrip().removesuffix("===").strip()
            buf, in_prompt = [], False
        elif line.startswith("--- PROMPT ---"):
            in_prompt = True
        elif line.startswith("--- ") and line.rstrip().endswith("---"):
            in_prompt = False
        elif in_prompt:
            buf.append(line)
    if question is not None and buf:
        prompts[norm(question)] = norm("\n".join(buf))
    return prompts


# ---------------------------------------------------------------- per-run analysis

def analyze_run(run_dir: Path, lme_map, locomo_map):
    judged = run_dir / "judged.jsonl"
    log = run_dir / "debug_prompt_response.log"
    if not judged.exists() or not log.exists():
        return None
    rows = [json.loads(l) for l in judged.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        return None
    is_lme = any(run_dir.glob("longmemeval_*.jsonl"))
    prompts = parse_prompt_log(log)

    out_rows, missing_prompt, missing_ev = [], 0, 0
    for r in rows:
        qn = norm(r["question"])
        prompt = prompts.get(qn)
        if prompt is None:
            missing_prompt += 1
            continue
        if is_lme:
            info = lme_map.get(qn)
        else:
            info = locomo_map.get((r.get("sample_id"), qn))
        if not info or not info["evidence"]:
            missing_ev += 1
            continue
        ev_hits = sum(1 for e in info["evidence"] if e in prompt)
        gold = norm(str(r.get("gold", "")))
        out_rows.append({
            "qid": r.get("qid"),
            "category": r.get("category"),
            "judge_correct": r.get("judge_correct"),
            "n_evidence_turns": len(info["evidence"]),
            "evidence_turns_hit": ev_hits,
            "evidence_in_ctx": int(ev_hits > 0),
            "answer_in_ctx": int(bool(gold) and len(gold) >= 2 and gold in prompt),
        })

    if not out_rows:
        return None
    (run_dir / "completeness.jsonl").write_text(
        "\n".join(json.dumps(x) for x in out_rows) + "\n", encoding="utf-8")

    n = len(out_rows)
    with_ev = [x for x in out_rows if x["evidence_in_ctx"]]
    without = [x for x in out_rows if not x["evidence_in_ctx"]]
    scored_w = [x for x in with_ev if x["judge_correct"] is not None]
    scored_wo = [x for x in without if x["judge_correct"] is not None]

    def acc(xs):
        return round(sum(x["judge_correct"] for x in xs) / len(xs), 4) if xs else None

    return {
        "run": run_dir.name,
        "mode": rows[0].get("mode"),
        "model": rows[0].get("model"),
        "dataset": "longmemeval" if is_lme else "locomo",
        "n": n,
        "evidence_recall": round(len(with_ev) / n, 4),
        "answer_in_ctx": round(sum(x["answer_in_ctx"] for x in out_rows) / n, 4),
        "acc_given_evidence": acc(scored_w),
        "n_given_evidence": len(scored_w),
        "acc_given_no_evidence": acc(scored_wo),
        "n_given_no_evidence": len(scored_wo),
        "skipped_no_prompt": missing_prompt,
        "skipped_no_evidence_label": missing_ev,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="*")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    dirs = ([d for d in sorted(RESULTS.iterdir())
             if d.is_dir() and (d / "judged.jsonl").exists()
             and (d / "debug_prompt_response.log").exists()]
            if args.all else [Path(d) for d in args.run_dirs])
    if not dirs:
        sys.exit("no run dirs (pass paths or --all)")

    lme_map = lme_evidence_map() if LME_DATASET.exists() else {}
    locomo_map = locomo_evidence_map() if LOCOMO_DATASET.exists() else {}

    summaries = []
    for d in dirs:
        s = analyze_run(d, lme_map, locomo_map)
        if s:
            summaries.append(s)
            print(f"{s['run']:44s} n={s['n']:<4d} ev_recall={s['evidence_recall']:.2f} "
                  f"ans_in_ctx={s['answer_in_ctx']:.2f} "
                  f"acc|ev={s['acc_given_evidence']} (n={s['n_given_evidence']}) "
                  f"acc|no_ev={s['acc_given_no_evidence']} (n={s['n_given_no_evidence']})")
        else:
            print(f"{d.name:44s} SKIPPED (missing artifacts or no matched rows)")

    if summaries:
        out = RESULTS / "completeness_table.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            w.writeheader()
            w.writerows(summaries)
        print(f"\nwrote {out} ({len(summaries)} runs)")


if __name__ == "__main__":
    main()
