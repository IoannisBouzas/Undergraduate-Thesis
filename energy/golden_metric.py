"""Golden metric: correct answers per kJ, per memory strategy, on-device.

Method of record: phone 70b-judged accuracy
(LoCoMo n=20 stratified) x phone battery telemetry clipped to the unplugged run
span. Charge use is the direct Coulomb-counter delta. Energy pairs each charge
interval with the average measured battery voltage over that interval. Answers
per charge use the phone's 5,000 mAh capacity and do not assume a fixed voltage.

Also prints a box-joined sensitivity variant using the prefixed box accuracy
with the same phone energy. Because the box-phone parity check failed
(2026-08-04: 83% agreement < 90%; results/parity_check_2026-08-04.md),
box accuracy may NOT be joined to phone energy for the headline. This variant is
reported as a labelled sensitivity ONLY; the phone-measured metric above is the record.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

BATTERY_CAPACITY_MAH = 5_000.0

AUXILIARY_RUNS = {
    # Longer-workload observation used only for the Chapter 5 amortization appendix.
    # It is not a matched n=20 versus n=100 intervention.
    "E2B_vectorrag_n100": "phone_locomo100_e2b_vectorrag_n100",
}

CELLS = {  # dir -> (mode, model, box accuracy for the sensitivity-only line)
    # Box accuracy = LoCoMo n=200 prefixed RECORD (lap_locomo200_{model}_{vr_prefix,fullcontext}),
    # NOT the n=20 box twin — the box-join uses record rates by design (cf. E2B fc 0.075 = n=200,
    # not the 0.10 available at n=20). It is an intensive rate x rate sensitivity, cross-n by
    # construction, and not used for the headline since the parity check failed. E4B fc 0.105 = record
    # lap_locomo200_e4b_fullcontext (was 0.060, which matched fc_think, not record fc — fixed).
    "phone_locomo20_e2b_vectorrag":   ("vectorrag",   "E2B", 0.360),
    "phone_locomo20_e2b_fullcontext": ("fullcontext", "E2B", 0.075),
    "phone_locomo20_e4b_vectorrag":   ("vectorrag",   "E4B", 0.405),
    "phone_locomo20_e4b_fullcontext": ("fullcontext", "E4B", 0.105),
    # Cross-family on-device tier, added 2026-08-25 (judged 08-24). box accuracy is None on
    # purpose: the parity gate was only ever run for E2B, it FAILED, and no qwen box-phone
    # agreement measurement exists — so there is nothing licensing a box join here, not even
    # as a labelled sensitivity. Phone-measured accuracy only.
    "phone_locomo20_q35_08b_vectorrag": ("vectorrag",   "Q35-0.8B", None),
    # 0.8b fullcontext scored 0/20 (judged 2026-08-25). Kept in the table on purpose: it is the
    # measurement that shows full context does not merely cost more at the smallest tier, it
    # returns nothing. acc=0 makes J/correct undefined and the vr/fc ratio unbounded; both are
    # printed as such rather than silently dropped.
    "phone_locomo20_q35_08b_fullcontext": ("fullcontext", "Q35-0.8B", None),
    "phone_locomo20_q35_2b_vectorrag":  ("vectorrag",   "Q35-2B",   None),
    "phone_locomo20_q35_2b_fullcontext": ("fullcontext", "Q35-2B",  None),
    "phone_locomo20_q35_4b_vectorrag":  ("vectorrag",   "Q35-4B",   None),
    "phone_locomo20_q35_4b_fullcontext": ("fullcontext", "Q35-4B",  None),
}

# Independent replicate sittings. Start charge and temperature were logged, not controlled.
# Predictions are byte-identical across replicates (temperature 0 -> deterministic),
# so replicates carry ENERGY variance only; accuracy variance is structurally zero.
REPLICATES = {
    ("E2B", "vectorrag"):   ["phone_locomo20_e2b_vectorrag",
                             "phone_locomo20_e2b_vectorrag_rep2",
                             "phone_locomo20_e2b_vectorrag_rep3"],
    ("E2B", "fullcontext"): ["phone_locomo20_e2b_fullcontext",
                             "phone_locomo20_e2b_fullcontext_rep2",
                             "phone_locomo20_e2b_fullcontext_rep3"],
    # Added 2026-07-30, closing the "E4B ratio has no error bar" gap. All three sittings share
    # an identical config (top_k 10, ctx 4096, embed_prefix none, max_answer 64), so they are
    # true replicates of the banked rep1 rather than a new condition.
    ("E4B", "vectorrag"):   ["phone_locomo20_e4b_vectorrag",
                             "phone_locomo20_e4b_vectorrag_rep2",
                             "phone_locomo20_e4b_vectorrag_rep3"],
    # E4B fullcontext remains a SINGLE sitting (~4 h/cell). Disclosed, not measured.
    # Qwen3.5 tier, added 2026-08-25: three vector-RAG sittings per size, same settings. Their
    # fullcontext arms are SINGLE sittings, so those ratios carry no error bar either — stated, not hidden.
    ("Q35-0.8B", "vectorrag"): ["phone_locomo20_q35_08b_vectorrag",
                                "phone_locomo20_q35_08b_vectorrag_rep2",
                                "phone_locomo20_q35_08b_vectorrag_rep3"],
    ("Q35-2B", "vectorrag"):   ["phone_locomo20_q35_2b_vectorrag",
                                "phone_locomo20_q35_2b_vectorrag_rep2",
                                "phone_locomo20_q35_2b_vectorrag_rep3"],
    ("Q35-4B", "vectorrag"):   ["phone_locomo20_q35_4b_vectorrag",
                                "phone_locomo20_q35_4b_vectorrag_rep2",
                                "phone_locomo20_q35_4b_vectorrag_rep3"],
}


def run_span(run_dir: Path) -> tuple[float, float]:
    ts = []
    for jp in run_dir.glob("locomo_*.jsonl"):
        for line in jp.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                for k in ("ingest_start_ts", "query_start_ts"):
                    if r.get(k): ts.append(float(r[k]))
                for k in ("ingest_end_ts", "query_end_ts"):
                    if r.get(k): ts.append(float(r[k]))
    return min(ts), max(ts)


@dataclass(frozen=True)
class BatteryMeasurement:
    """Direct charge use and voltage-weighted energy over one run span."""

    charge_mah: float
    energy_kj: float
    effective_voltage_v: float
    sample_count: int


def battery_measurement(
    run_dir: Path, t0: float, t1: float, csv_name: str = "resources.csv"
) -> BatteryMeasurement:
    """The one phone energy method: every energy number in the thesis comes from here."""
    samples: list[tuple[float, float, float]] = []
    with (run_dir / csv_name).open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                t = float(row["timestamp"])
                cc = float(row["battery_charge_counter_uah"])
                voltage_mv = float(row["battery_voltage_mv"])
            except (TypeError, ValueError, KeyError):
                continue
            usable = all(math.isfinite(value) for value in (t, cc, voltage_mv))
            unplugged = str(row.get("battery_plugged", "")).lower() != "true"
            if t0 <= t <= t1 and usable and unplugged and voltage_mv > 0:
                samples.append((t, cc, voltage_mv))
    samples.sort(key=lambda sample: sample[0])
    if len(samples) < 2:
        raise ValueError(f"{run_dir.name}: <2 usable unplugged samples in span")

    charge_mah = (samples[0][1] - samples[-1][1]) / 1000.0
    energy_j = 0.0
    for (_, charge_0_uah, voltage_0_mv), (_, charge_1_uah, voltage_1_mv) in zip(
        samples, samples[1:]
    ):
        delta_uah = charge_0_uah - charge_1_uah
        mean_voltage_v = (voltage_0_mv + voltage_1_mv) / 2000.0
        energy_j += delta_uah * mean_voltage_v * 0.0036
    if charge_mah <= 0 or energy_j <= 0:
        raise ValueError(f"{run_dir.name}: non-positive battery discharge in span")
    return BatteryMeasurement(
        charge_mah=charge_mah,
        energy_kj=energy_j / 1000.0,
        effective_voltage_v=energy_j / (charge_mah * 3.6),
        sample_count=len(samples),
    )


def cc_delta_kj(run_dir: Path, t0: float, t1: float) -> float:
    """Return voltage-weighted battery energy in kJ for compatibility."""

    return battery_measurement(run_dir, t0, t1).energy_kj


def _run_rows(run_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for jp in sorted(run_dir.glob("locomo_*.jsonl"))
        for line in jp.open(encoding="utf-8")
        if line.strip()
    ]


def full_charge_pause_end(run_dir: Path, t0: float) -> float | None:
    """When a run starts at 100% the counter holds its full value for a while, so energy in that
    stretch is not recorded. Returns the first time the counter falls, or None if the run did not
    start at 100%. Read from the logged level and counter, so no start rule is needed."""
    with (run_dir / "resources.csv").open(encoding="utf-8") as f:
        rows = [(float(r["timestamp"]), float(r["battery_charge_counter_uah"]), r["battery_level_pct"])
                for r in csv.DictReader(f) if r.get("battery_charge_counter_uah")]
    after = [r for r in sorted(rows) if r[0] >= t0]
    if not after or float(after[0][2]) < 100:
        return None
    return next((t for t, cc, _ in after if cc < after[0][1]), None)


def phase_split(run_dir: Path) -> dict:
    """Energy of the build phase (each cold ingestion window) and of the query phase (the stretches
    between and after them), each measured with battery_measurement. Any stretch that began while
    the counter was paused at full charge is left out, and listed."""
    rows = _run_rows(run_dir)
    t0, t1 = run_span(run_dir)
    pause_end = full_charge_pause_end(run_dir, t0)
    builds = sorted(
        (float(r["ingest_start_ts"]), float(r["ingest_end_ts"]), int(r["n_history_turns"]), r["sample_id"])
        for r in rows
        if (r.get("ingest_seconds") or 0) > 1  # a cached history embeds nothing
    )
    if len({b[3] for b in builds}) != len(builds):
        raise ValueError(f"{run_dir.name}: a history was built twice")
    edges = [t0] + [t for b in builds for t in b[:2]] + [t1]
    stretches = []  # (kind, start, end, units)
    for a, b, turns, _ in builds:
        stretches.append(("build", a, b, turns))
    for a, b in zip(edges[::2], edges[1::2]):
        answers = sum(1 for r in rows if a <= float(r["query_start_ts"]) < b)
        if answers:
            stretches.append(("query", a, b, answers))
    out = {"stretches": [
        {"phase": k, "start": a, "end": b, "kept": pause_end is None or a >= pause_end}
        for k, a, b, _ in stretches
    ]}
    out["skipped"] = [s for s in out["stretches"] if not s["kept"]]
    for kind in ("build", "query"):
        joules = seconds = units = 0.0
        for k, a, b, n in stretches:
            if k != kind or (pause_end is not None and a < pause_end):
                continue
            # Known limit: the counter steps about every 30 s, so each stretch boundary can shift
            # about one step (~68 J) between neighbours; a stretch with no step inside reads 0 J.
            try:
                joules += battery_measurement(run_dir, a, b).energy_kj * 1000
            except ValueError:
                pass
            seconds += b - a
            units += n
        out[kind] = {"joules": joules, "seconds": seconds, "units": units,
                     "mean_w": joules / seconds if seconds else None,
                     "per_unit": joules / units if units else None}
    return out


GRAPH_PILOT_DIR = Path("results_phone") / "_smoke_recovered"


def graph_build() -> dict:
    """The one phone graph build (419 turns of conv-26), same method as every other number."""
    run = json.loads(
        (GRAPH_PILOT_DIR / "phone_smoke_gr" / "locomo_graphrag_gemma4-e2b.jsonl")
        .read_text(encoding="utf-8").splitlines()[0]
    )
    t0, t1 = float(run["ingest_start_ts"]), float(run["ingest_end_ts"])
    energy_kj = battery_measurement(GRAPH_PILOT_DIR, t0, t1, csv_name="phone_gr.csv").energy_kj
    turns = int(run["n_history_turns"])
    return {"energy_kj": energy_kj, "turns": turns, "minutes": (t1 - t0) / 60,
            "j_per_turn": energy_kj * 1000 / turns}


def correct_per_charge(
    *, accuracy: float, n_answers: int, total_charge_mah: float
) -> float:
    """Estimate correct answers per full charge from directly measured charge use."""

    if n_answers <= 0 or total_charge_mah <= 0:
        raise ValueError("n_answers and total_charge_mah must be positive")
    mah_per_answer = total_charge_mah / n_answers
    return accuracy / mah_per_answer * BATTERY_CAPACITY_MAH


def main() -> None:
    rows = []
    for d, (mode, model, box_acc) in CELLS.items():
        rd = Path("results_phone") / d
        meta = json.load((rd / "judge_meta.json").open())
        acc, n = meta["judge_accuracy"], meta["n_scored"]
        t0, t1 = run_span(rd)
        battery = battery_measurement(rd, t0, t1)
        kj = battery.energy_kj
        j_ans = kj * 1000 / n
        mah_ans = battery.charge_mah / n
        rows.append(
            dict(
                mode=mode,
                model=model,
                phone_acc=acc,
                n=n,
                total_mah=battery.charge_mah,
                total_kj=kj,
                effective_voltage_v=battery.effective_voltage_v,
                battery_samples=battery.sample_count,
                energy_method="voltage_weighted_coulomb_counting",
                span_s=t1 - t0,
                # Mean whole-device power over the same span the energy is taken from. It is a
                # per-cell measurement, not a device constant: the range across cells is what
                # Chapter 3 quotes instead of a single "about 4 W".
                mean_w=kj * 1000 / (t1 - t0),
                mah_ans=mah_ans,
                j_ans=j_ans,
                j_correct=j_ans / acc if acc else None,
                cpkj=acc / j_ans * 1000,
                per_charge=correct_per_charge(
                    accuracy=acc,
                    n_answers=n,
                    total_charge_mah=battery.charge_mah,
                ),
                box_cpkj=(box_acc / j_ans * 1000) if box_acc is not None else None,
            )
        )
    print("GOLDEN METRIC — phone-measured (method of record)")
    print(f"{'cell':22s} {'acc':>5s} {'mAh':>7s} {'kJ':>7s} {'J/ans':>7s} "
          f"{'J/correct':>9s} {'corr/kJ':>8s} {'per-charge':>10s}")
    for r in rows:
        # j_correct is None when a cell scored zero correct answers (energy per correct answer is
        # undefined, not infinite-and-printable). Render it explicitly rather than crashing.
        jc = f"{r['j_correct']:9.0f}" if r["j_correct"] is not None else f"{'n/a':>9s}"
        print(f"{r['model']+' '+r['mode']:22s} {r['phone_acc']:5.2f} "
              f"{r['total_mah']:7.1f} {r['total_kj']:7.2f} {r['j_ans']:7.0f} "
              f"{jc} {r['cpkj']:8.3f} {r['per_charge']:10.1f}")
    # One ratio per reader size that has BOTH arms measured on the phone. Order is fixed so
    # the printed table is diffable across runs.
    for model in ("E2B", "E4B", "Q35-0.8B", "Q35-2B", "Q35-4B"):
        vr = next((r for r in rows if r["model"] == model and r["mode"] == "vectorrag"), None)
        fc = next((r for r in rows if r["model"] == model and r["mode"] == "fullcontext"), None)
        if not (vr and fc):
            continue
        # A zero-accuracy fullcontext arm makes the ratio unbounded, not enormous. Report the
        # bound and the two rates instead of dividing by zero.
        if fc["cpkj"] == 0:
            print(f"{model}: vr/fc correct-per-kJ = UNBOUNDED "
                  f"(fullcontext scored 0/{fc['n']}; vr {vr['cpkj']:.3f} vs fc 0.000 correct/kJ)")
            continue
        line = f"{model}: vr/fc correct-per-kJ = {vr['cpkj']/fc['cpkj']:.1f}x"
        if vr["box_cpkj"] is not None and fc["box_cpkj"] is not None:
            line += (f"  (box-joined sensitivity ONLY: {vr['box_cpkj']/fc['box_cpkj']:.1f}x "
                     f"— not used for the headline: parity check failed)")
        else:
            line += "  (single fullcontext sitting, no error bar; no box join — see CELLS note)"
        print(line)
    # --- replicate-based uncertainty -----------------------------------------
    # Accuracy is deterministic across sittings, so the headline ratio's uncertainty
    # comes entirely from between-run energy variance. Propagate it explicitly.
    import statistics as stat
    rep_report = {}
    print("\nBETWEEN-RUN UNCERTAINTY (independent sittings)")
    per_mode: dict[tuple[str, str], list[float]] = {}
    per_mode_accs: dict[tuple[str, str], list[float]] = {}
    for (model, mode), dirs in REPLICATES.items():
        # A replicate group is usable only once every sitting is judged — n_scored is the
        # per-answer denominator. Skip loudly rather than crashing on a freshly pulled cell.
        unjudged = [d for d in dirs
                    if not (Path("results_phone") / d / "judge_meta.json").exists()]
        if unjudged:
            print(f"  {model} {mode:12s} SKIPPED — not yet judged: {', '.join(unjudged)}")
            continue
        js, mahs, accs, phases = [], [], [], []
        for d in dirs:
            rd = Path("results_phone") / d
            if mode == "vectorrag":
                phases.append({"run": d, **phase_split(rd)})
            meta = json.load((rd / "judge_meta.json").open())
            judged = [
                json.loads(line)
                for line in (rd / "judged.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            scored_labels = [row["judge_correct"] for row in judged if row.get("judge_correct") is not None]
            if len(scored_labels) != meta["n_scored"]:
                raise ValueError(
                    f"{d}: judge_meta n_scored={meta['n_scored']} but judged.jsonl has "
                    f"{len(scored_labels)} scored rows"
                )
            t0, t1 = run_span(rd)
            battery = battery_measurement(rd, t0, t1)
            js.append(battery.energy_kj * 1000 / meta["n_scored"])
            mahs.append(battery.charge_mah / meta["n_scored"])
            accs.append(sum(int(bool(label)) for label in scored_labels) / len(scored_labels))
        per_mode[(model, mode)] = js
        per_mode_accs[(model, mode)] = accs
        mean_j, sd_j = stat.mean(js), stat.stdev(js)
        mean_mah, sd_mah = stat.mean(mahs), stat.stdev(mahs)
        # Correct answers a full charge buys, off the replicate MEAN rather than one sitting.
        # This is the resource-aware ranking: it need not follow the accuracy ranking, because a
        # cheaper reader can answer more questions per charge than a better one. Per-answer energy
        # at n=20 includes building the memory (about 60% of an E2B vector sitting, phase_split).
        # A separate n=100 Vector-RAG run used less energy per answer, mainly because it spread
        # the build over more questions, but it is not a matched run-length experiment.
        # Only emitted when accuracy really is constant across the sittings. E2B fullcontext is
        # [0.05, 0.0526, 0.05] -- the 0.0526 sitting scored 19 rows, not 20 -- so quoting one
        # sitting's accuracy there would hide a choice inside a headline-shaped number.
        per_charge = (
            accs[0] / mean_mah * BATTERY_CAPACITY_MAH
            if len(set(accs)) == 1
            else None
        )
        pc_s = f"  correct/charge {per_charge:5.1f}" if per_charge is not None else \
               "  correct/charge n/a (accuracy not constant across sittings)"
        print(f"  {model} {mode:12s} J/answer {mean_j:6.0f} +/- {sd_j:4.0f} "
              f"(CV {sd_j/mean_j*100:4.1f}%, n={len(js)} sittings)  acc {accs} -> deterministic"
              f"{pc_s}")
        rep_report[f"{model}_{mode}"] = {
            "j_per_answer": js,
            "mean": mean_j,
            "sd": sd_j,
            "cv": sd_j / mean_j,
            "mah_per_answer": mahs,
            "mean_mah_per_answer": mean_mah,
            "sd_mah_per_answer": sd_mah,
            "accuracy_by_sitting": accs,
            "correct_per_charge_at_mean_mah": per_charge,
        }
        if phases:
            build = [p["build"]["per_unit"] for p in phases]
            query = [p["query"]["per_unit"] for p in phases]
            rep_report[f"{model}_{mode}"].update(
                phases=phases,
                build_j_per_turn=stat.mean(build), build_j_per_turn_sd=stat.stdev(build),
                query_j_per_answer=stat.mean(query), query_j_per_answer_sd=stat.stdev(query),
            )
            print(f"    phases: build {stat.mean(build):.3f} +/- {stat.stdev(build):.3f} J/turn, "
                  f"query-only {stat.mean(query):.0f} +/- {stat.stdev(query):.0f} J/answer "
                  f"(skipped {sum(len(p['skipped']) for p in phases)} stretches paused at 100%)")

    for model in sorted({m for m, _ in per_mode}):
        vr_j, fc_j = per_mode.get((model, "vectorrag")), per_mode.get((model, "fullcontext"))
        if not (vr_j and fc_j):
            continue
        vr_accs = per_mode_accs[(model, "vectorrag")]
        fc_accs = per_mode_accs[(model, "fullcontext")]
        # ratio per sitting-pair (same list index)
        # Use each sitting's own scored denominator. This matters for the E2B full-context
        # replicate that completed 19 answers: its accuracy is 1/19, not 1/20.
        ratios = [
            (acc_vr / v) / (acc_fc / f)
            for v, f, acc_vr, acc_fc in zip(vr_j, fc_j, vr_accs, fc_accs)
        ]
        mr, sr = stat.mean(ratios), stat.stdev(ratios)
        print(f"  {model} GOLDEN METRIC vr/fc correct-per-kJ = {mr:.1f}x +/- {sr:.1f} "
              f"(per sitting: {', '.join(f'{r:.1f}' for r in ratios)})")
        rep_report[f"{model}_ratio"] = {"by_sitting": ratios, "mean": mr, "sd": sr}

    auxiliary_report = {}
    print("\nAUXILIARY PHONE RUNS (descriptive, outside the primary n=20 grid)")
    for label, directory in AUXILIARY_RUNS.items():
        run_dir = Path("results_phone") / directory
        meta = json.load((run_dir / "judge_meta.json").open())
        judged = [
            json.loads(line)
            for line in (run_dir / "judged.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        scored = [row["judge_correct"] for row in judged if row.get("judge_correct") is not None]
        if len(scored) != meta["n_scored"]:
            raise ValueError(
                f"{directory}: judge_meta n_scored={meta['n_scored']} but judged.jsonl has "
                f"{len(scored)} scored rows"
            )
        accuracy = sum(int(bool(value)) for value in scored) / len(scored)
        t0, t1 = run_span(run_dir)
        battery = battery_measurement(run_dir, t0, t1)
        record = {
            "run": directory,
            "n_scored": len(scored),
            "accuracy": accuracy,
            "charge_mah": battery.charge_mah,
            "energy_kj": battery.energy_kj,
            "joules_per_answer": battery.energy_kj * 1000 / len(scored),
            "effective_voltage_v": battery.effective_voltage_v,
            "evidence_status": "descriptive_run_length_observation",
        }
        auxiliary_report[label] = record
        print(
            f"  {label}: n={record['n_scored']}, acc={accuracy:.3f}, "
            f"energy={record['energy_kj']:.2f} kJ, "
            f"J/answer={record['joules_per_answer']:.1f}"
        )

    # Memory construction per dialogue turn: the one graph build against the E2B vector builds.
    graph = graph_build()
    vector = rep_report["E2B_vectorrag"]
    by_sitting = [p["build"]["per_unit"] for p in vector["phases"]]
    construction = {
        "graph": graph,
        "vector": {"runs": [p["run"] for p in vector["phases"]], "j_per_turn_by_sitting": by_sitting,
                   "j_per_turn": vector["build_j_per_turn"], "sd": vector["build_j_per_turn_sd"]},
        "ratio": graph["j_per_turn"] / vector["build_j_per_turn"],
        "ratio_by_sitting": [graph["j_per_turn"] / v for v in by_sitting],
    }
    print(f"\nCONSTRUCTION: graph {graph['j_per_turn']:.2f} J/turn ({graph['energy_kj']:.3f} kJ, "
          f"{graph['turns']} turns) vs vector {vector['build_j_per_turn']:.3f} J/turn -> "
          f"{construction['ratio']:.1f}x (by sitting "
          f"{min(construction['ratio_by_sitting']):.1f}-{max(construction['ratio_by_sitting']):.1f}x)")

    out = Path("results/golden_metric.json")
    out.write_text(
        json.dumps(
            {"cells": rows, "replicates": rep_report, "auxiliary": auxiliary_report,
             "construction": construction},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
