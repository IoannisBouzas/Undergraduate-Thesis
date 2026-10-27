"""How speaker-centred are the SLM-extracted LoCoMo graphs?

For each extractor's prefixed v4-style graph caches (same prompt triple_json_v2, extraction
budget 512, 10 LoCoMo conversations), measure:
  speaker_edge_share   fraction of edges with at least one speaker endpoint
  speaker_endpoint_share  fraction of all edge endpoints that are a speaker node
  speaker_deg_vs_median   mean speaker degree / median non-speaker degree
  top2_are_speakers       conversations whose two highest-degree nodes are the two speakers
Writes results/speaker_hub_stats.json.

Usage: .venv/Scripts/python.exe speaker_hub_stats.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from graph_memory.reembed_graph_caches import _MainAliasUnpickler  # caches pickled from __main__

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "gemma4-e2b": "lap_locomo200_e2b_gr_mt24_prefix",
    "gemma4-e4b": "lap_locomo200_e4b_gr_mt24_prefix",
    "qwen3.5-0.8b": "lap_locomo200_q35_08b_gr_mt24_prefix",
    "qwen3.5-2b": "lap_locomo200_q35_2b_gr_mt24_prefix",
    "qwen3.5-4b": "lap_locomo200_q35_4b_gr_mt24_prefix",
}


def graph_stats(mem) -> dict:
    g, sp = mem.graph, mem.speakers
    edges = list(g.edges())
    deg = {n: len(set(g.predecessors(n)) | set(g.successors(n))) for n in g.nodes()}
    others = [d for n, d in deg.items() if n not in sp]
    sp_deg = [deg[s] for s in sp if s in deg]
    top2 = set(sorted(deg, key=lambda n: (-deg[n], n))[:2])
    return {
        "edges": len(edges),
        "speaker_edge_share": sum(1 for u, v in edges if u in sp or v in sp) / len(edges),
        "speaker_endpoint_share": sum((u in sp) + (v in sp) for u, v in edges) / (2 * len(edges)),
        "speaker_deg_vs_median": statistics.mean(sp_deg) / statistics.median(others),
        "top2_are_speakers": int(top2 == set(sp)),
    }


def main() -> None:
    out = {}
    for model, run in RUNS.items():
        per = []
        for p in sorted((ROOT / "results" / run / "graph_cache").glob("*_extok_512.pkl")):
            with p.open("rb") as f:
                per.append(graph_stats(_MainAliasUnpickler(f).load()))
        out[model] = {
            "conversations": len(per),
            "edges_mean": round(statistics.mean(s["edges"] for s in per), 1),
            "speaker_edge_share": round(statistics.mean(s["speaker_edge_share"] for s in per), 3),
            "speaker_endpoint_share": round(statistics.mean(s["speaker_endpoint_share"] for s in per), 3),
            "speaker_deg_vs_median": round(statistics.mean(s["speaker_deg_vs_median"] for s in per), 1),
            "top2_are_speakers": f"{sum(s['top2_are_speakers'] for s in per)}/{len(per)}",
        }
        print(model, out[model])
    (ROOT / "results" / "speaker_hub_stats.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
