"""Why SPE-2 never expands: measure the gap distribution and the three competing densities.

Replicates _context_spe's candidate construction and greedy argmax exactly (run_memory_eval.py
:914-1075), on the prefixed record caches, and records which change type wins each greedy step.

Read-only. Writes nothing into results/.
"""
import json, pickle, re, string, sys, os
from pathlib import Path
import numpy as np

sys.path.insert(0, os.getcwd())
import run_memory_eval as R

sys.modules["__main__"].GraphRAGMemory = R.GraphRAGMemory

RUN = Path("results/lap_locomo200_e2b_spe2fix_gap_b370")
BUDGET = 370
VECTOR_K = 10
N_QUESTIONS = 48
HOST = "http://localhost:18080/api"

DOC, QRY = R.NOMIC_DOC_PREFIX, R.NOMIC_QUERY_PREFIX


def tok(s):
    return max(1, len(s) // 4)


import requests

_ecache = {}


def embed(text):
    if text in _ecache:
        return _ecache[text]
    r = requests.post(HOST + "/embeddings",
                      json={"model": "nomic-embed-text:latest", "prompt": text}, timeout=120)
    r.raise_for_status()
    d = r.json()
    emb = d.get("embedding") or d["data"][0]["embedding"]
    v = np.asarray(emb, dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-12)
    _ecache[text] = v
    return v


def format_node(mem, n):
    return mem._format_node(n)


def candidates_for(mem, qv, vector_k):
    """Exactly _context_spe's candidate construction."""
    M = np.stack(mem.turn_embs)
    sims = M @ qv
    k = min(vector_k, len(sims))
    top_idx = np.argsort(-sims)[:k]
    top_set = {int(i) for i in top_idx}
    sim_of = {int(i): float(sims[i]) for i in top_idx}
    cands = []
    for u, v, key, data in mem.graph.edges(keys=True, data=True):
        spans = data.get("src_spans") or []
        hit = sorted({i for a, b in spans for i in top_set.intersection(range(a, b))},
                     key=lambda i: -sim_of[i])
        if not hit:
            continue
        ts = data.get("timestamp", "")
        prefix = f"[{ts}] " if ts else ""
        line = f"{prefix}System: {format_node(mem, u)} {data['relation']} {format_node(mem, v)}."
        cands.append({"line": line, "turns": hit, "sim": sim_of[hit[0]]})
    cands.sort(key=lambda c: -c["sim"])
    return cands[:40], sim_of


def greedy(mem, cands, qv, budget):
    """Exactly the implemented loop. Returns (step_types, n_expanded, spent)."""
    for c in cands:
        c["u_triple"] = float(embed(DOC + c["line"]) @ qv)
        c["u_source"] = c["sim"]
        c["gap"] = c["u_source"] - c["u_triple"]
    selected_turns, spent = set(), 0
    remaining = [{"cand": c, "level": 0} for c in cands]
    steps = []
    while True:
        best = None
        for item in remaining:
            c = item["cand"]
            options = [False, True] if item["level"] == 0 else [True]
            for exp in options:
                cost = 0 if item["level"] == 1 and exp else tok(c["line"]) + 1
                if exp:
                    cost += sum(tok("Source: " + mem.turn_texts[i]) + 1
                                for i in c["turns"] if i not in selected_turns)
                    if item["level"] == 1:
                        cost = max(1, cost)
                if spent + cost > budget or cost <= 0:
                    continue
                u = (c["u_source"] - c["u_triple"]) if (item["level"] == 1 and exp) \
                    else (c["u_source"] if exp else c["u_triple"])
                if u <= 0:
                    continue
                ratio = u / cost
                kind = ("upgrade" if item["level"] == 1 else
                        ("omit_to_expanded" if exp else "omit_to_triple"))
                if best is None or ratio > best[0]:
                    best = (ratio, item, exp, cost, u, kind)
        if best is None:
            break
        ratio, item, exp, cost, u, kind = best
        steps.append((kind, ratio, cost, u))
        spent += cost
        if exp:
            selected_turns.update(item["cand"]["turns"])
            remaining.remove(item)
        else:
            item["level"] = 1
    return steps, len(selected_turns), spent


def main():
    caches = sorted((RUN / "graph_cache").glob("*embp_nomic*.pkl"))
    print("prefixed caches:", len(caches))
    import hashlib

    def hist_hash(history):
        h = hashlib.sha256()
        for t in history:
            h.update(f"{t.speaker}:{t.text}:{t.session_id}:{t.turn_idx}:{t.timestamp}".encode("utf-8"))
        return h.hexdigest()

    examples = R.load_locomo(Path("locomo10.json"), 200)
    by_conv = {}
    for ex in examples:
        by_conv.setdefault(ex.sample_id, []).append(ex)
    print("conversations:", len(by_conv))

    per_conv = max(1, N_QUESTIONS // max(1, len(by_conv)))
    sample = []
    for cid, exs in sorted(by_conv.items()):
        sample.extend(exs[:per_conv])
    print("sampled questions:", len(sample), "across", len(by_conv), "conversations")

    gaps, u_tri, u_src = [], [], []
    dens_new_triple, dens_upgrade, dens_omit_exp = [], [], []
    cost_triple, cost_expand = [], []
    all_steps, expanded_counts = [], []
    upgrade_ever_available = 0
    n_done = 0

    mem_cache = {}
    for ex in sample:
        # find the cache whose history matches this conversation
        key = hist_hash(ex.history)
        if key not in mem_cache:
            mem = None
            for c in caches:
                if c.name.startswith(key):
                    with open(c, "rb") as f:
                        mem = pickle.load(f)
                    break
            mem_cache[key] = mem
        mem = mem_cache[key]
        if mem is None:
            continue
        qv = embed(QRY + ex.question)
        cands, _ = candidates_for(mem, qv, VECTOR_K)
        if not cands:
            continue
        steps, n_exp, spent = greedy(mem, cands, qv, BUDGET)
        all_steps.extend(s[0] for s in steps)
        expanded_counts.append(n_exp)
        for c in cands:
            gaps.append(c["gap"]); u_tri.append(c["u_triple"]); u_src.append(c["u_source"])
            ct = tok(c["line"]) + 1
            ce = sum(tok("Source: " + mem.turn_texts[i]) + 1 for i in c["turns"])
            cost_triple.append(ct); cost_expand.append(ce)
            dens_new_triple.append(c["u_triple"] / ct)
            dens_omit_exp.append(c["u_source"] / (ct + ce))
            if c["gap"] > 0:
                upgrade_ever_available += 1
                dens_upgrade.append(c["gap"] / max(1, ce))
        n_done += 1
        if n_done % 8 == 0:
            print("  ...", n_done, "questions")

    def q(a, name, fmt="%.4f"):
        a = np.asarray(a, dtype=float)
        if a.size == 0:
            print(f"{name:26s} (empty)"); return
        print(f"{name:26s} n={a.size:6d}  mean={fmt % a.mean()}  "
              f"p10={fmt % np.percentile(a,10)}  med={fmt % np.median(a)}  "
              f"p90={fmt % np.percentile(a,90)}  max={fmt % a.max()}")

    print("\n===== candidate-level distributions (B=%d, k=%d) =====" % (BUDGET, VECTOR_K))
    q(u_src, "u_source = sim(q, turn)")
    q(u_tri, "u_triple = sim(q, triple)")
    q(gaps, "gap = u_source - u_triple")
    print(f"{'gap > 0 share':26s} {100.0*upgrade_ever_available/max(1,len(gaps)):.1f}%")
    q(cost_triple, "cost: add triple (tok)", "%.1f")
    q(cost_expand, "cost: add sources (tok)", "%.1f")
    print("\n===== the three competing densities (utility per token) =====")
    q(dens_new_triple, "omit->triple   u_tri/C")
    q(dens_omit_exp, "omit->expanded u_src/C")
    q(dens_upgrade, "triple->expand gap/C")
    med_t = np.median(dens_new_triple); med_u = np.median(dens_upgrade) if dens_upgrade else 0
    print(f"\nmedian(omit->triple) / median(upgrade) = {med_t/max(1e-12,med_u):.1f}x")
    med_e = np.median(dens_omit_exp)
    print(f"median(omit->triple) / median(omit->expanded) = {med_t/max(1e-12,med_e):.1f}x")

    from collections import Counter
    print("\n===== which change type won each greedy step =====")
    cnt = Counter(all_steps)
    tot = sum(cnt.values())
    for k2, v2 in cnt.most_common():
        print(f"  {k2:18s} {v2:6d}  ({100.0*v2/max(1,tot):.1f}%)")
    print(f"\nquestions simulated: {n_done}; mean source turns expanded: "
          f"{np.mean(expanded_counts) if expanded_counts else 0:.3f}")


if __name__ == "__main__":
    main()
