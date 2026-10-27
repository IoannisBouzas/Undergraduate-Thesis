"""Resource reporting from EXISTING logs (no new runs).

Derives, per mode: peak/mean RAM (phone resources.csv), index sizes on disk
(vector + graph pickles), ingestion time, query latency p50/p95 (phone and box),
and cold-vs-warm fullcontext cost. Writes results/resource_stats.md.
"""

from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

PHONE = {
    "E2B vectorrag":   "results_phone/phone_locomo20_e2b_vectorrag",
    "E2B fullcontext": "results_phone/phone_locomo20_e2b_fullcontext",
    "E4B vectorrag":   "results_phone/phone_locomo20_e4b_vectorrag",
    "E4B fullcontext": "results_phone/phone_locomo20_e4b_fullcontext",
}
BOX = {
    "E2B vr (prefix)":   "results/lap_locomo200_e2b_vr_prefix",
    "E2B grv3 (prefix)": "results/lap_locomo200_e2b_graphrag_v3_prefix",
    "E2B fc":            "results/lap_locomo200_e2b_fullcontext",
}


def rows_of(d: Path) -> list[dict]:
    out = []
    for jp in d.glob("locomo_*.jsonl"):
        if jp.name.endswith("_failures.jsonl"):
            continue
        for line in jp.open(encoding="utf-8"):
            if line.strip():
                out.append(json.loads(line))
    return out


def pctile(vals: list[float], p: float) -> float:
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(round(p / 100 * (len(vals) - 1))))]


def ram_stats(d: Path, t0: float, t1: float):
    used = []
    with (d / "resources.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                t = float(row["timestamp"]); r = float(row["ram_used_mb"])
            except (ValueError, KeyError):
                continue
            if t0 <= t <= t1:
                used.append(r)
    return (max(used), st.mean(used)) if used else (None, None)


def span(rows: list[dict]):
    ts = [float(r[k]) for r in rows for k in
          ("ingest_start_ts", "query_start_ts", "ingest_end_ts", "query_end_ts") if r.get(k)]
    return min(ts), max(ts)


def cache_size_mb(d: Path) -> float:
    return sum(p.stat().st_size for p in (d / "graph_cache").glob("*.pkl")) / 1e6


def main() -> None:
    lines = ["# Resource statistics (derived from existing logs, no new runs)", ""]

    lines += ["## Phone (S23 Ultra, n=20 runs)", "",
              "| cell | RAM peak MB | RAM mean MB | ingest s (sum) | query lat p50 s | p95 s |",
              "|---|---|---|---|---|---|"]
    for name, d in PHONE.items():
        d = Path(d)
        rows = rows_of(d)
        t0, t1 = span(rows)
        peak, mean = ram_stats(d, t0, t1)
        ing = sum(r.get("ingest_seconds") or 0 for r in rows)
        lat = [(r.get("retrieval_seconds") or 0) + (r.get("generate_seconds") or 0) for r in rows]
        lines.append(f"| {name} | {peak:.0f} | {mean:.0f} | {ing:.0f} | "
                     f"{pctile(lat,50):.1f} | {pctile(lat,95):.1f} |")

    lines += ["", "## Index sizes on disk (per 10-conversation LoCoMo corpus)", "",
              "| index | MB |", "|---|---|"]
    lines.append(f"| vector (turns+embeddings, E2B prefix) | {cache_size_mb(Path('results/lap_locomo200_e2b_vr_prefix')):.1f} |")
    lines.append(f"| graph v3 (triples+spans+embeddings, E2B prefix) | {cache_size_mb(Path('results/lap_locomo200_e2b_graphrag_v3_prefix')):.1f} |")

    lines += ["", "## Box query latency + ingest (n=200 runs, api-reported)", "",
              "| cell | ingest s (cold convs) | query lat p50 s | p95 s |", "|---|---|---|---|"]
    for name, d in BOX.items():
        rows = rows_of(Path(d))
        ing = [r["ingest_seconds"] for r in rows if (r.get("ingest_seconds") or 0) > 0]
        lat = [(r.get("api_total_seconds") or r.get("generate_seconds") or 0)
               + (r.get("retrieval_seconds") or 0) for r in rows]
        lines.append(f"| {name} | {sum(ing):.0f} ({len(ing)} cold) | "
                     f"{pctile(lat,50):.2f} | {pctile(lat,95):.2f} |")

    # cold vs warm fullcontext on phone: q1 of each conversation (cold KV) vs rest
    rows = rows_of(Path(PHONE["E2B fullcontext"]))
    bysample: dict[str, list] = {}
    for r in rows:
        bysample.setdefault(r["sample_id"], []).append(r)
    cold = [v[0]["generate_seconds"] for v in bysample.values()]
    warm = [x["generate_seconds"] for v in bysample.values() for x in v[1:]]
    lines += ["", "## Cold vs warm (E2B fullcontext phone, per-conversation first answer vs rest)", "",
              f"- cold (first answer/conv): mean {st.mean(cold):.0f} s (n={len(cold)})",
              f"- warm (subsequent):        mean {st.mean(warm):.0f} s (n={len(warm)})"]

    out = Path("results/resource_stats.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
