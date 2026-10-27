#!/usr/bin/env python3
"""Ego retrieval (graph v1 and the phone graph pilot) breaks score ties in an order that changes
from one Python process to the next; vector-first retrieval (graph v2 onwards) does not.

The ego scorer gives every edge the same score unless an endpoint or the relation matches the
query, so a hub's edges tie. Tied edges are kept in the order networkx iterates the candidate
subgraph, and for a subgraph view smaller than half the graph that is the iteration order of a
Python set of node names, which string hashing (PYTHONHASHSEED) changes per process.

Writes results/ego_tie_order.json with three measurements:
  - cutoff_ties: per ego run, questions whose 12th and 13th candidates tie on score, so the kept
    facts depend on the process's hash seed;
  - caroline_seed_sweep: fig21's question re-run under N hash seeds against the pilot's recorded prompt;
  - vector_first_seed_check: record-pipeline caches queried under three seeds must give identical contexts.

Usage: python -m graph_memory.ego_tie_order_check [--seeds 50]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import run_memory_eval

sys.modules["__main__"].GraphRAGMemory = run_memory_eval.GraphRAGMemory

ROOT = Path(__file__).resolve().parents[1]
EGO_RUNS = ["results/lap_locomo200_e2b_graphrag", "results/lap_locomo200_e4b_graphrag", "results_phone/_smoke_recovered/phone_smoke_gr"]
RECORD_RUNS = ["lap_locomo200_e2b_graphrag_v3_prefix", "lap_locomo200_e4b_graphrag_v4_prefix"]


class _TokenCount:
    @staticmethod
    def approx_tokens(text: str) -> int:
        return max(1, len(text) // 4)


def ego_ranking(mem, question: str) -> list:
    """Every ego candidate as (u, v, key, score, data), best first, captured from the deployed code."""
    captured: list[list] = []

    def spy(items, *args, **kwargs):
        out = sorted(items, *args, **kwargs)
        captured.append(out)
        return out

    run_memory_eval.sorted = spy
    try:
        mem.context_for(question)
    finally:
        del run_memory_eval.sorted
    return next((c for c in captured if c and len(c[0]) == 5), [])


def cutoff_ties(run: str) -> dict:
    run_dir = ROOT / run
    rows = [json.loads(line) for line in next(run_dir.glob("*_graphrag_*.jsonl")).open(encoding="utf-8") if line.strip()]
    memories: dict[str, object] = {}
    tied: list[str] = []
    for row in rows:
        if row["history_hash"] not in memories:
            pkl = next((run_dir / "graph_cache").glob(f"{row['history_hash']}_*.pkl"))
            mem = pickle.load(pkl.open("rb"))
            mem.backend = _TokenCount()
            memories[row["history_hash"]] = mem
        mem = memories[row["history_hash"]]
        ranked = ego_ranking(mem, row["question"])
        budget = mem.max_triples
        if len(ranked) > budget and ranked[budget - 1][3] == ranked[budget][3]:
            tied.append(row.get("stable_qid") or row["qid"])  # stable_qid: qid depends on --limit
    return {"questions": len(rows), "tie_across_cutoff": len(tied), "tied_questions": tied}


def _run_seeded(code: str, seed: int) -> str:
    env = dict(os.environ, PYTHONHASHSEED=str(seed))
    return subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.strip().splitlines()[-1]


CAROLINE_CHILD = """
import json, pickle
from figures import make_extended_figures as a
mem = pickle.load(open(a.PHONE_PILOT_GRAPH, "rb")); mem.backend = a._TokenCount()
print(json.dumps(mem.context_for(a.CASE_QUESTION)[0].splitlines()))
"""

VECTOR_FIRST_CHILD = """
import hashlib, pickle, sys
from pathlib import Path
import run_memory_eval as rfb
sys.modules["__main__"].GraphRAGMemory = rfb.GraphRAGMemory
from graph_memory.ego_tie_order_check import RECORD_RUNS, _TokenCount
digest = hashlib.sha256()
for run in RECORD_RUNS:
    for pkl in sorted(Path("results", run, "graph_cache").glob("*.pkl")):
        mem = pickle.load(pkl.open("rb")); mem.backend = _TokenCount()
        mem.append_source_turns = run.endswith("v4_prefix")
        for i in range(0, len(mem.turn_embs), 7):  # stored turn embeddings stand in for queries (no server)
            rfb.embed_query = lambda backend, q, v=mem.turn_embs[i]: v
            digest.update(mem.context_for("stand-in")[0].encode())
print(digest.hexdigest())
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=50)
    args = parser.parse_args()

    from figures import make_extended_figures as addons  # imported late: it loads matplotlib

    pilot = sorted(addons._pilot_prompt_facts(addons.CASE_QUESTION))
    runs = [json.loads(_run_seeded(CAROLINE_CHILD, seed)) for seed in range(args.seeds)]
    digests = {_run_seeded(VECTOR_FIRST_CHILD, seed) for seed in (0, 1, 12345)}
    ties = {run: cutoff_ties(run) for run in EGO_RUNS if (ROOT / run / "graph_cache").exists()}
    e2b, e4b = (set(ties[run]["tied_questions"]) for run in EGO_RUNS[:2])
    result = {
        "cutoff_ties": ties,
        "locomo_v1_tied_in_both_readers": len(e2b & e4b),
        "caroline_seed_sweep": {
            "question": addons.CASE_QUESTION,
            "seeds": args.seeds,
            "adoption_triple_kept": sum(any("adoption process" in fact for fact in facts) for facts in runs),
            "pilot_prompt_reproduced": sum(sorted(facts) == pilot for facts in runs),
        },
        "vector_first_seed_check": {"runs": RECORD_RUNS, "seeds": [0, 1, 12345], "identical": len(digests) == 1},
    }
    out = ROOT / "results" / "ego_tie_order.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
