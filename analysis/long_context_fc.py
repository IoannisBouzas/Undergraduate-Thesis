"""Untruncated full-context cells against the record conditions.

Reads judged runs only (no model calls). For each reader/benchmark, pairs the long-window
full-context run with the record truncated full-context, vector and graph+sources runs on qid,
reports accuracy, mean prompt tokens, discordant counts and the exact McNemar p, and checks that
no long-window prompt hit its context limit. Writes results/long_context_fc.json.
"""
import json, os
from math import comb

CELLS = [  # (label, benchmark, long-window run, context limit, truncated fc, vector, graph v4)
    ("Gemma 4 E2B", "LoCoMo", "lap_locomo200_e2b_fullcontext_ctx128k", 131072,
     "lap_locomo200_e2b_fullcontext", "lap_locomo200_e2b_vr_prefix", "lap_locomo200_e2b_graphrag_v4_prefix"),
    ("Gemma 4 E4B", "LoCoMo", "lap_locomo200_e4b_fullcontext_ctx128k", 131072,
     "lap_locomo200_e4b_fullcontext", "lap_locomo200_e4b_vr_prefix", "lap_locomo200_e4b_graphrag_v4_prefix"),
    ("Qwen3.5 4B", "LoCoMo", "lap_locomo200_q35_4b_fullcontext_ctx128k", 131072,
     "lap_locomo200_q35_4b_fullcontext", "lap_locomo200_q35_4b_vr_prefix", "lap_locomo200_q35_4b_graphrag_v4_prefix"),
    ("Gemma 4 E2B", "LongMemEval-S", "lme_strat60_e2b_fc_ctx128k", 131072,
     "lme_strat60_e2b_fc", "lme_strat60_e2b_vr_prefix", "lme_strat60_e2b_graphrag_v4_prefix"),
    ("Gemma 4 E4B", "LongMemEval-S", "lme_strat60_e4b_fc_ctx128k", 131072,
     "lme_strat60_e4b_fc", "lme_strat60_e4b_vr_prefix", "lme_strat60_e4b_graphrag_v4_prefix"),
    ("Qwen3.5 4B", "LongMemEval-S", "lme_strat60_q35_4b_fc_ctx256k", 262144,
     "lme_strat60_q35_4b_fc", "lme_strat60_q35_4b_vr_prefix", None),
]


def judged(run):
    rows = [json.loads(l) for l in open(f"results/{run}/judged.jsonl", encoding="utf-8") if l.strip()]
    assert all(r.get("judge_correct") is not None for r in rows), run
    return {r["qid"]: r for r in rows}


def mcnemar(a, b):
    n = a + b
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


def mean_tokens(rows):
    t = [r.get("prompt_tokens") for r in rows.values() if r.get("prompt_tokens") is not None]
    return sum(t) / len(t) if t else None


out = []
for label, bench, run, limit, fc, vr, gr in CELLS:
    new = judged(run)
    meta = json.load(open(f"results/{run}/judge_meta.json"))
    assert meta["judge_model"] == "llama3.1:70b-instruct-q4_K_M"
    toks = [r["prompt_tokens"] for r in new.values()]
    cell = {"reader": label, "benchmark": bench, "run": run, "n": len(new), "ctx_limit": limit,
            "accuracy": sum(bool(r["judge_correct"]) for r in new.values()) / len(new),
            "prompt_tokens": {"min": min(toks), "mean": sum(toks) / len(toks), "max": max(toks)},
            "prompts_at_limit": sum(t >= limit for t in toks), "vs": {}}
    for name, other in (("truncated_fc", fc), ("vector", vr), ("graph_v4", gr)):
        if other is None or not os.path.exists(f"results/{other}/judged.jsonl"):
            continue
        o = judged(other)
        assert set(o) == set(new), (run, other)  # same questions, joined on qid within one limit
        a = sum(1 for q in new if new[q]["judge_correct"] and not o[q]["judge_correct"])
        b = sum(1 for q in new if o[q]["judge_correct"] and not new[q]["judge_correct"])
        cell["vs"][name] = {"run": other, "accuracy": sum(bool(r["judge_correct"]) for r in o.values()) / len(o),
                            "mean_prompt_tokens": mean_tokens(o), "long_only": a, "other_only": b,
                            "mcnemar_p": mcnemar(a, b)}
    out.append(cell)
    v = cell["vs"].get("vector", {})
    print(f"{bench:13s} {label:12s} long-FC {cell['accuracy']*100:5.1f}%  vector {v.get('accuracy', 0)*100:5.1f}%  "
          f"({v.get('long_only')}/{v.get('other_only')}, p={v.get('mcnemar_p', 1):.3g})  "
          f"tokens {cell['prompt_tokens']['mean']:,.0f} vs {v.get('mean_prompt_tokens') or 0:,.0f}  "
          f"at-limit {cell['prompts_at_limit']}")

json.dump(out, open("results/long_context_fc.json", "w", encoding="utf-8"), indent=1)
print("wrote results/long_context_fc.json")
