#!/usr/bin/env python3
"""Why does a vector-RAG sitting draw more average power than a full-context sitting?

Two candidate explanations: the embedding work that builds the vector memory, and the duty cycle
(short queries near peak clock against long prefill at a settled clock). This script splits each
sitting into its phases with golden_metric.phase_split, the same battery method as every other
energy number in the thesis:

  * build phase - every cold ingestion window (six histories per vector sitting). The reader is
                  idle; only the embedder works.
  * query phase - the stretches between and after the build windows, where the questions run.

Stretches that began while the charge counter was paused at 100% are left out of the energy.
It also reports the median of the maximum core frequency over the query stretches, which is where
a settled clock shows up.

Writes results/phone_power_phases.json. Reads only banked phone runs.
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from energy.golden_metric import phase_split

ROOT = Path(__file__).resolve().parents[1]
CELLS = {
    ("Qwen3.5 0.8B", "Vector RAG"): "phone_locomo20_q35_08b_vectorrag",
    ("Qwen3.5 0.8B", "Full context"): "phone_locomo20_q35_08b_fullcontext",
    ("Qwen3.5 2B", "Vector RAG"): "phone_locomo20_q35_2b_vectorrag",
    ("Qwen3.5 2B", "Full context"): "phone_locomo20_q35_2b_fullcontext",
    ("Qwen3.5 4B", "Vector RAG"): "phone_locomo20_q35_4b_vectorrag",
    ("Qwen3.5 4B", "Full context"): "phone_locomo20_q35_4b_fullcontext",
    ("Gemma 4 E2B", "Vector RAG"): "phone_locomo20_e2b_vectorrag",
    ("Gemma 4 E2B", "Full context"): "phone_locomo20_e2b_fullcontext",
    ("Gemma 4 E4B", "Vector RAG"): "phone_locomo20_e4b_vectorrag",
    ("Gemma 4 E4B", "Full context"): "phone_locomo20_e4b_fullcontext",
}


def _median_max_core_mhz(run_dir: Path, stretches: list[dict]) -> float | None:
    """Median of the fastest core's frequency over the query stretches (not a battery quantity)."""
    spans = [(x["start"], x["end"]) for x in stretches if x["phase"] == "query"]
    with (run_dir / "resources.csv").open(encoding="utf-8") as handle:
        freqs = [float(r["cpu_freq_max_mhz"]) for r in csv.DictReader(handle)
                 if r.get("cpu_freq_max_mhz") and float(r["cpu_freq_max_mhz"]) > 0
                 and any(a <= float(r["timestamp"]) <= b for a, b in spans)]
    return statistics.median(freqs) if freqs else None


def main() -> None:
    out: dict[str, dict] = {}
    for (reader, strategy), run in CELLS.items():
        run_dir = ROOT / "results_phone" / run
        split = phase_split(run_dir)
        phases = {}
        for kind in ("build", "query"):
            ph = split[kind]
            phases[kind] = None if not ph["seconds"] else {
                "minutes": ph["seconds"] / 60, "energy_j": ph["joules"], "mean_w": ph["mean_w"],
                "units": ph["units"]}
        phases["query_median_max_core_mhz"] = _median_max_core_mhz(run_dir, split["stretches"])
        phases["stretches_left_out_paused_counter"] = len(split["skipped"])
        out[f"{reader} | {strategy}"] = {"run": run, **phases}

    path = ROOT / "results" / "phone_power_phases.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"{'cell':30s} {'build W':>8s} {'build min':>10s} {'query W':>8s} {'query MHz':>10s} {'left out':>9s}")
    for name, cell in out.items():
        b, q = cell["build"], cell["query"]
        fmt = lambda x, k, f: format(x[k], f) if x else "-"
        print(f"{name:30s} {fmt(b, 'mean_w', '8.2f'):>8s} {fmt(b, 'minutes', '10.1f'):>10s} "
              f"{fmt(q, 'mean_w', '8.2f'):>8s} {cell['query_median_max_core_mhz'] or 0:10.0f} "
              f"{cell['stretches_left_out_paused_counter']:9d}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
