#!/usr/bin/env python3
"""Blind human audit of the 70b judge.

Samples 100 judged answers stratified across (run, judge verdict), strips every
condition identifier, shuffles (seed 42), writes:
  results/judge_audit_sheet.csv — id, question, gold, model_answer, human_label (blank)
  results/judge_audit_key.csv   — id -> run dir, qid, judge_correct  (DO NOT open while labeling)
Label correct / incorrect / ambiguous. Second annotator: rows 1-25 (overlap set).
Then: python -m judging.build_audit_sheet --score results/judge_audit_sheet_filled.csv
"""
import json, os, csv, random, argparse

RUNS = [  # spread over modes, models, datasets
    "lap_locomo200_e2b_vectorrag", "lap_locomo200_e2b_fullcontext",
    "lap_locomo200_e2b_graphrag_v3", "lap_locomo200_e4b_vectorrag",
    "lme_strat60_e2b_vr", "lme_strat60_e4b_vr", "lme_strat60_e2b_fc",
    "lap_locomo200_e2b_graphrag_v4",
]
PER_RUN = 13  # ~half judged-correct, half judged-incorrect where available

def rows_of(d):
    p = os.path.join("results", d, "judged.jsonl")
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p, encoding="utf-8"):
        if line.strip():
            out.append(json.loads(line))
    return out

def build():
    rng = random.Random(42)
    picked = []
    for d in RUNS:
        rows = rows_of(d)
        if not rows:
            print(f"skip {d}: no judged.jsonl"); continue
        n_null = sum(1 for r in rows if r.get("judge_correct") is None)
        if n_null:
            print(f"WARNING: {d}: {n_null} unscored judge row(s) excluded from audit pool; "
                  f"run rejudge_missing.py first")
        rows = [r for r in rows if r.get("judge_correct") is not None]
        cor = [r for r in rows if r.get("judge_correct")]
        inc = [r for r in rows if not r.get("judge_correct")]
        rng.shuffle(cor); rng.shuffle(inc)
        half = PER_RUN // 2
        sel = cor[:half + PER_RUN % 2] + inc[:half]
        for r in sel:
            picked.append((d, r))
    rng.shuffle(picked)
    picked = picked[:100]
    with open("results/judge_audit_sheet.csv", "w", newline="", encoding="utf-8-sig") as fh, \
         open("results/judge_audit_key.csv", "w", newline="", encoding="utf-8") as kh:
        w = csv.writer(fh); k = csv.writer(kh)
        w.writerow(["id", "question", "gold_answer", "model_answer", "human_label(correct/incorrect/ambiguous)"])
        k.writerow(["id", "run", "qid", "judge_correct"])
        for i, (d, r) in enumerate(picked, 1):
            w.writerow([i, r.get("question", ""), r.get("gold", ""), r.get("pred") or r.get("raw_pred", ""), ""])
            k.writerow([i, d, r.get("qid"), int(bool(r.get("judge_correct")))])
    print(f"wrote results/judge_audit_sheet.csv ({len(picked)} rows; label all; "
          f"2nd annotator does rows 1-25) and judge_audit_key.csv (keep closed)")

def score(filled):
    key = {row["id"]: row for row in csv.DictReader(open("results/judge_audit_key.csv", encoding="utf-8"))}
    n = agree = fp = fn = amb = 0
    for row in csv.DictReader(open(filled, encoding="utf-8-sig")):
        lab = (row.get("human_label(correct/incorrect/ambiguous)") or "").strip().lower()
        if not lab:
            continue
        j = int(key[row["id"]]["judge_correct"])
        if lab.startswith("amb"):
            amb += 1; continue
        h = 1 if lab.startswith("c") else 0
        n += 1
        agree += (h == j)
        fp += (j == 1 and h == 0)  # judge said correct, human says wrong
        fn += (j == 0 and h == 1)
    # Cohen's kappa over the decisive (non-ambiguous) labels
    pairs = []
    for row in csv.DictReader(open(filled, encoding="utf-8-sig")):
        lab = (row.get("human_label(correct/incorrect/ambiguous)") or "").strip().lower()
        if lab and not lab.startswith("amb"):
            pairs.append((int(key[row["id"]]["judge_correct"]), 1 if lab.startswith("c") else 0))
    pj = sum(j for j, _ in pairs) / n
    ph = sum(h for _, h in pairs) / n
    pe = pj * ph + (1 - pj) * (1 - ph)
    kappa = (agree / n - pe) / (1 - pe)
    print(f"labeled n={n} (+{amb} ambiguous)  agreement={agree/n*100:.1f}%  kappa={kappa:.3f}  "
          f"judge FP={fp} ({fp/n*100:.1f}%)  judge FN={fn} ({fn/n*100:.1f}%)")
    out = {"filled_sheet": filled, "n_decisive": n, "n_ambiguous": amb, "agree": agree,
           "agreement": agree / n, "cohen_kappa": kappa, "judge_false_pos": fp, "judge_false_neg": fn,
           "judge_model": "llama3.1:70b-instruct-q4_K_M", "annotators": 1}
    json.dump(out, open("results/judge_audit_result.json", "w", encoding="utf-8"), indent=1)
    print("wrote results/judge_audit_result.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", default=None, help="filled sheet CSV to score against the key")
    a = ap.parse_args()
    score(a.score) if a.score else build()
