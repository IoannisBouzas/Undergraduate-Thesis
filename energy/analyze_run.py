"""
Turn a run's raw logs into thesis metrics.

Joins the inference rows (results/<run_id>/*.jsonl, each with query_start_ts /
query_end_ts) against the resource samples (results/<run_id>/resources.csv,
each with an absolute `timestamp`) to compute, per question and per mode:

  - energy per answer        (two methods, see below)
  - latency / tokens         (prompt, generated, retrieved)
  - quality                  (f1/em/contains now; judge label if present)
  - correct answers per joule and per 1k tokens   <- the headline efficiency axes
  - thermal / throttling summary

Both logs must share one clock. They do when the profiler runs ON the phone
beside the inference (same time source). If the profiler ran on the laptop over
adb, pass --clock-offset (laptop_ts - phone_ts) to align them.

Energy: golden_metric.battery_measurement over the run span, the one method the thesis uses.
Per-question energy is not reported: the charge counter steps only about every 30 s. If no
energy was captured, energy is null and latency/token/thermal metrics are still produced.
"""

import argparse
import csv
import glob
import json
import statistics
from pathlib import Path

from energy.golden_metric import battery_measurement


def load_rows(run_dir: Path) -> list[dict]:
    # Prefer the judged file (it carries judge_correct); otherwise the raw rows.
    # Reading both would double-count every question.
    judged = run_dir / "judged.jsonl"
    paths = [str(judged)] if judged.exists() else glob.glob(str(run_dir / "*.jsonl"))
    rows: list[dict] = []
    for jp in paths:
        with open(jp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_resources(path: Path, clock_offset: float = 0.0) -> list[dict]:
    samples: list[dict] = []
    if not path.exists():
        return samples
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ts = _f(r.get("timestamp"))
            if ts is None:
                continue
            samples.append({
                "ts": ts + clock_offset,
                "cc_uah": _f(r.get("battery_charge_counter_uah")),
                "voltage_mv": _f(r.get("battery_voltage_mv")),
                "soc_temp_c": _f(r.get("soc_temp_c")),
                "cpu_freq_avg_mhz": _f(r.get("cpu_freq_avg_mhz")),
                "gpu_busy_pct": _f(r.get("gpu_busy_pct")),
                "battery_source": r.get("battery_source", ""),
                "plugged": r.get("battery_plugged", ""),
            })
    samples.sort(key=lambda s: s["ts"])
    return samples


def window(samples: list[dict], t0, t1) -> list[dict]:
    if t0 is None or t1 is None:
        return []
    return [s for s in samples if t0 <= s["ts"] <= t1]


def is_correct(row: dict, metric: str, f1_threshold: float) -> float | None:
    # Prefer an LLM-judge label when the judge step has been wired in.
    # A present-but-null label means "judged run, row unscored" -> return missing
    # (None), never the lexical fallback: mixing metrics inside one judged run
    # would silently violate the judge standard. rejudge_missing.py scores nulls.
    if "judge_correct" in row:
        v = row["judge_correct"]
        if v is None:
            return None
        return 1.0 if v in (1, True, "1", "yes", "CORRECT") else 0.0
    if metric == "f1":
        return 1.0 if (_f(row.get("f1")) or 0.0) >= f1_threshold else 0.0
    return 1.0 if (_f(row.get(metric)) or 0.0) >= 1.0 else 0.0


def retrieved_tokens(row: dict):
    cm = row.get("ctx_meta") or {}
    return cm.get("retrieved_tokens")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 4) if xs else None


def analyze(run_dir: Path, correct_metric: str, f1_threshold: float, clock_offset: float) -> dict:
    rows = load_rows(run_dir)
    samples = load_resources(run_dir / "resources.csv", clock_offset)
    have_energy = any(s["cc_uah"] is not None for s in samples)
    stamps = [_f(r.get(k)) for r in rows for k in ("ingest_start_ts", "query_start_ts", "ingest_end_ts", "query_end_ts")]
    stamps = [t for t in stamps if t]
    try:  # resource timestamps + clock_offset = run timestamps
        run_battery = battery_measurement(run_dir, min(stamps) - clock_offset, max(stamps) - clock_offset)
    except (ValueError, FileNotFoundError):
        run_battery = None

    per_row = []
    for row in rows:
        t0, t1 = _f(row.get("query_start_ts")), _f(row.get("query_end_ts"))
        win = window(samples, t0, t1)
        rec = {
            "qid": row.get("qid"),
            "mode": row.get("mode"),
            "model": row.get("model"),
            "category": row.get("category"),
            "correct": is_correct(row, correct_metric, f1_threshold),
            "f1": _f(row.get("f1")),
            "prompt_tokens": _f(row.get("prompt_tokens")),
            "completion_tokens": _f(row.get("completion_tokens")),
            "total_tokens": _f(row.get("total_tokens")),
            "retrieved_tokens": _f(retrieved_tokens(row)),
            "generate_seconds": _f(row.get("generate_seconds")),
            "retrieval_seconds": _f(row.get("retrieval_seconds")),
            "ttft_seconds": _f(row.get("api_prompt_eval_seconds")),
            "peak_temp_c": max([s["soc_temp_c"] for s in win if s["soc_temp_c"] is not None], default=None),
            "n_samples": len(win),
        }
        per_row.append(rec)

    # aggregate per mode
    modes = sorted({r["mode"] for r in per_row if r["mode"]})
    by_mode = {}
    for m in modes:
        sub = [r for r in per_row if r["mode"] == m]
        n = len(sub)
        # accuracy denominator = scored rows only (matches judge_meta n_scored);
        # unscored rows disclosed via n_unscored, their energy still counts in sums
        scored = [r for r in sub if r["correct"] is not None]
        n_scored = len(scored)
        sum_correct = sum(r["correct"] for r in scored)
        sum_tokens = sum(r["total_tokens"] for r in sub if r["total_tokens"]) or None
        # One mode per run directory: the whole-run energy belongs to that mode.
        sum_energy = run_battery.energy_kj * 1000 if run_battery and len(modes) == 1 else None
        by_mode[m] = {
            "n": n,
            "n_scored": n_scored,
            "n_unscored": n - n_scored,
            "accuracy": round(sum_correct / n_scored, 4) if n_scored else None,
            "f1_mean": mean([r["f1"] for r in sub]),
            "tokens_per_answer": mean([r["total_tokens"] for r in sub]),
            "retrieved_tokens_per_answer": mean([r["retrieved_tokens"] for r in sub]),
            "latency_s_mean": mean([r["generate_seconds"] for r in sub]),
            "ttft_s_mean": mean([r["ttft_seconds"] for r in sub]),
            "energy_j_per_answer": round(sum_energy / n, 4) if sum_energy else None,
            "correct_per_joule": round(sum_correct / sum_energy, 4) if sum_energy else None,
            "correct_per_1k_tokens": round(sum_correct / sum_tokens * 1000, 4) if sum_tokens else None,
            "peak_temp_c": max([r["peak_temp_c"] for r in sub if r["peak_temp_c"] is not None], default=None),
        }

    # whole-run thermal / throttle + energy-by-coulomb
    temps = [s["soc_temp_c"] for s in samples if s["soc_temp_c"] is not None]
    freqs = [s["cpu_freq_avg_mhz"] for s in samples if s["cpu_freq_avg_mhz"] is not None]
    run_summary = {
        "n_rows": len(rows),
        "n_resource_samples": len(samples),
        "energy_captured": have_energy,
        "battery_source": (samples[0]["battery_source"] if samples else None),
        "plugged_any": any(str(s["plugged"]).lower() == "true" for s in samples),
        "temp_idle_to_peak_c": [round(min(temps), 1), round(max(temps), 1)] if temps else None,
        "cpu_freq_peak_to_min_mhz": [round(max(freqs), 1), round(min(freqs), 1)] if freqs else None,
        "throttled": (max(freqs) - min(freqs) > 300) if freqs else None,
        "run_charge_mah": round(run_battery.charge_mah, 3) if run_battery else None,
        "run_energy_kj": round(run_battery.energy_kj, 4) if run_battery else None,
    }

    return {"run_dir": str(run_dir), "run_summary": run_summary, "by_mode": by_mode, "per_row": per_row}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="results/<run_id> directory")
    p.add_argument("--correct-metric", choices=["f1", "em", "contains"], default="f1")
    p.add_argument("--f1-threshold", type=float, default=0.5)
    p.add_argument("--clock-offset", type=float, default=0.0,
                   help="laptop_ts - phone_ts, if profiler ran on the laptop over adb")
    p.add_argument("--no-per-row", action="store_true", help="omit per-row detail from the JSON")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    result = analyze(run_dir, args.correct_metric, args.f1_threshold, args.clock_offset)

    out = {k: v for k, v in result.items() if not (args.no_per_row and k == "per_row")}
    (run_dir / "analysis.json").write_text(json.dumps(out, indent=2))

    rs = result["run_summary"]
    print(f"== {run_dir} ==")
    print(f"rows={rs['n_rows']}  resource_samples={rs['n_resource_samples']}  "
          f"energy_captured={rs['energy_captured']}  battery_source={rs['battery_source']}")
    if rs["plugged_any"]:
        print("  WARNING: some samples were taken while PLUGGED IN -> energy invalid.")
    print(f"  thermal: idle->peak={rs['temp_idle_to_peak_c']}C  "
          f"freq peak->min={rs['cpu_freq_peak_to_min_mhz']}MHz  throttled={rs['throttled']}")
    if rs["run_energy_kj"] is not None:
        print(f"  run energy: {rs['run_energy_kj']} kJ ({rs['run_charge_mah']} mAh)")
    print("  per mode:")
    for m, d in result["by_mode"].items():
        print(f"    {m:11s} n={d['n']:>4} acc={d['accuracy']} f1={d['f1_mean']} "
              f"tok/ans={d['tokens_per_answer']} lat={d['latency_s_mean']}s "
              f"J/ans={d['energy_j_per_answer']} correct/J={d['correct_per_joule']} "
              f"correct/1k_tok={d['correct_per_1k_tokens']}")
    print(f"  wrote {run_dir / 'analysis.json'}")


if __name__ == "__main__":
    main()
