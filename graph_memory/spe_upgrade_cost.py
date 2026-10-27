"""Token costs behind Chapter 7's account of why SPE-2 never expanded (Section "Why SPE retained only triples").

For every triple SPE-2 selected at B=370: the cost C of its line and the marginal cost dC of upgrading it to its
retrieved source turns, in the selector's own units. Also a lower bound on the cost of each question's whole
candidate pool. Offline: selected lines come from the run logs, spans from the graph caches, and the top-10 turns
from the GraphRAG v4 rows, which share the caches and vector_k=10. Writes results/spe_upgrade_cost.json.
"""
import json
import re
import statistics as st
from pathlib import Path

from graph_memory.reembed_graph_caches import _MainAliasUnpickler

RESULTS = Path("results")


def tok(s):  # run_memory_eval approx_tokens
    return max(1, len(s) // 4)


def rows(run):
    path = next((RESULTS / run).glob("locomo_graphrag_*.jsonl"))
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def contexts(run, run_rows):
    # The log was appended across reruns; its last pass matches the jsonl rows.
    log = (RESULTS / run / "debug_prompt_response.log").read_text(encoding="utf-8")
    blocks = re.split(r"^=== Q: ", log, flags=re.M)[1:][-len(run_rows):]
    assert [b.split(" ===", 1)[0] for b in blocks] == [r["question"] for r in run_rows]
    return [re.search(r"Conversation history:\n(.*?)\n\nQuestion:", b, flags=re.S).group(1) for b in blocks]


def line_of(mem, u, v, d):  # same rendering as _context_spe
    ts = d.get("timestamp", "")
    return f"{'[' + ts + '] ' if ts else ''}System: {mem._format_node(u)} {d['relation']} {mem._format_node(v)}."


def costs(model):
    run = f"lap_locomo200_{model}_spe2fix_gap_b370"
    spe = rows(run)
    top10 = {(r["sample_id"], r["question"]): set(r["ctx_meta"]["kept_turn_indices"])
             for r in rows(f"lap_locomo200_{model}_graphrag_v4_prefix")}
    mems, C, dC, n_turns, pool, ambiguous = {}, [], [], [], [], 0
    for r, ctx in zip(spe, contexts(run, spe)):
        h = r["history_hash"]
        if h not in mems:
            with next((RESULTS / run / "graph_cache").glob(h + "*.pkl")).open("rb") as f:
                mems[h] = _MainAliasUnpickler(f).load()
        mem, top = mems[h], top10[(r["sample_id"], r["question"])]
        hits = {}  # rendered line -> retrieved turns in its source batches, one set per candidate edge
        for u, v, _, d in mem.graph.edges(keys=True, data=True):
            turns = {i for a, b in d.get("src_spans") or [] for i in top.intersection(range(a, b))}
            if turns:
                hits.setdefault(line_of(mem, u, v, d), []).append(turns)
        line_costs = sorted(tok(line) + 1 for line, edges in hits.items() for _ in edges)
        assert min(len(line_costs), 40) == r["ctx_meta"]["candidate_edges"]  # same candidates as the run
        pool.append(sum(line_costs[:40]))  # the pool is the 40 most similar, so it costs at least the 40 cheapest
        selected = [line for line in ctx.splitlines() if line.strip()]
        assert len(selected) == r["ctx_meta"]["kept_facts"] and r["ctx_meta"]["kept_source_turns"] == 0
        for line in selected:
            ambiguous += len(hits[line]) > 1
            turns = set().union(*hits[line])
            C.append(tok(line) + 1)
            dC.append(sum(tok("Source: " + mem.turn_texts[i]) + 1 for i in turns))
            n_turns.append(len(turns))
    ratio = [d / c for d, c in zip(dC, C)]
    return {
        "selected_facts": len(C), "ambiguous_lines": ambiguous,
        "triple_cost_mean": st.mean(C),
        "upgrade_cost_mean": st.mean(dC), "upgrade_cost_median": st.median(dC),
        "share_upgrades_multi_turn": sum(n > 1 for n in n_turns) / len(n_turns),
        "ratio_median": st.median(ratio), "share_ratio_below_2": sum(x < 2 for x in ratio) / len(ratio),
        "pool_cost_lower_bound_mean": st.mean(pool),
    }


if __name__ == "__main__":
    out = {model: costs(model) for model in ("e2b", "e4b")}
    e2b, e4b = out["e2b"], out["e4b"]
    # The values Chapter 7 cites.
    assert e2b["selected_facts"] == 4136 and round(e2b["triple_cost_mean"], 1) == 17.5
    assert round(e2b["upgrade_cost_mean"], 1) == 75.4 and e2b["upgrade_cost_median"] == 68
    assert (round(e2b["ratio_median"], 1), round(e4b["ratio_median"], 1)) == (3.9, 3.5)
    assert (round(100 * e2b["share_ratio_below_2"]), round(100 * e4b["share_ratio_below_2"])) == (10, 13)
    assert (round(e2b["pool_cost_lower_bound_mean"]), round(e4b["pool_cost_lower_bound_mean"])) == (727, 814)
    (RESULTS / "spe_upgrade_cost.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
