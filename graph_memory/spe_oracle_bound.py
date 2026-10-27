"""Post hoc oracle bound for Chapter 7: per-question routing between GraphRAG v3 and v4.

For each question keep the v4 answer when v4 was judged correct and v3 was not, else the cheaper v3
answer. It reads the judged labels, so it is an upper bound on question-level selection between the
two existing runs, not a policy. Writes results/spe_oracle_bound.json.
"""
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"


def load(run):
    lines = (RESULTS / run / "judged.jsonl").read_text(encoding="utf-8").splitlines()
    return {(r["sample_id"], r["question"]): r for r in map(json.loads, filter(str.strip, lines))}


def bound(model):
    return bound_for(f"lap_locomo200_{model}_graphrag_v3_prefix", f"lap_locomo200_{model}_graphrag_v4_prefix")


def bound_for(v3_run, v4_run):
    """The same bound for any matched v3/v4 pair (fig10 uses its own unprefixed pair)."""
    v3, v4 = load(v3_run), load(v4_run)
    keys = sorted(v3.keys() & v4.keys())
    n = len(keys)
    c3 = [bool(v3[k]["judge_correct"]) for k in keys]
    c4 = [bool(v4[k]["judge_correct"]) for k in keys]
    t3 = [v3[k]["prompt_tokens"] for k in keys]
    t4 = [v4[k]["prompt_tokens"] for k in keys]
    only4 = [b and not a for a, b in zip(c3, c4)]
    acc3, acc4, tok3, tok4 = sum(c3) / n, sum(c4) / n, sum(t3) / n, sum(t4) / n
    oracle_tokens = sum(b if o else a for a, b, o in zip(t3, t4, only4)) / n
    p = sum(only4) / n  # share of questions the oracle expands
    return {
        "n": n, "v3_acc": acc3, "v4_acc": acc4, "v3_tokens": tok3, "v4_tokens": tok4,
        "v4_only_correct": sum(only4),
        "v3_only_correct": sum(a and not b for a, b in zip(c3, c4)),
        "oracle_acc": sum(a or b for a, b in zip(c3, c4)) / n,
        "oracle_tokens": oracle_tokens,
        "oracle_token_reduction_vs_v4": 1 - oracle_tokens / tok4,
        "random_router_acc": (1 - p) * acc3 + p * acc4,
        "random_router_tokens": (1 - p) * tok3 + p * tok4,
    }


if __name__ == "__main__":
    out = {model: bound(model) for model in ("e2b", "e4b")}
    # The discordant counts must equal those of the paired v3-vs-v4 tests reported in Chapter 6.
    assert (out["e2b"]["v4_only_correct"], out["e2b"]["v3_only_correct"]) == (27, 4)
    assert (out["e4b"]["v4_only_correct"], out["e4b"]["v3_only_correct"]) == (12, 2)
    # LongMemEval-S (n=60): no SPE run exists there; this is only the headroom for a v3/v4 selector.
    for model in ("e2b", "e4b"):
        out[f"lme_{model}"] = bound_for(f"lme_strat60_{model}_graphrag_v3_prefix", f"lme_strat60_{model}_graphrag_v4_prefix")
    assert (out["lme_e2b"]["v4_only_correct"], out["lme_e2b"]["v3_only_correct"]) == (11, 1)
    assert (out["lme_e4b"]["v4_only_correct"], out["lme_e4b"]["v3_only_correct"]) == (9, 3)
    (RESULTS / "spe_oracle_bound.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
