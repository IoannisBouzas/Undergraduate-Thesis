#!/usr/bin/env python3
"""Generate the claim-aligned thesis figure package.

The package deliberately separates confirmatory results, mechanism studies,
external-validity results, phone efficiency, and the early phone GraphRAG pilot.
Every plotted value is written to a source-data manifest, while
``figure_trace.yaml`` records the claim and limitation boundary for each figure.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from graph_memory import spe_oracle_bound
from energy.golden_metric import battery_measurement
from figures.visual_style import PAPER_PALETTE, PRESENTATION_PALETTE, configure_paper_style, paper_size, savefig_retry

# Set by configure_style. The plot functions take no style argument, so the handful of choices that
# differ between thesis and slides (canvas size, palette, in-figure notes) read it from here.
_STYLE = "paper"


MANIFEST_COLUMNS = [
    "figure",
    "panel",
    "dataset",
    "run",
    "condition",
    "series",
    "metric",
    "value",
    "error",
    "low",
    "high",
    "unit",
    "n",
    "replicates",
    "evidence",
    "source",
]

METHOD_STYLE = {
    "Full context": {"color": "#CC3311", "marker": "s"},
    "GraphRAG v3": {"color": "#009988", "marker": "^"},
    "GraphRAG v4": {"color": "#EE7733", "marker": "D"},
    "Vector RAG": {"color": "#0077BB", "marker": "o"},
}
ORACLE_COLOR = "#6A3D9A"
NUMBER_LABEL_BOX = {
    "boxstyle": "round,pad=0.10",
    "facecolor": "white",
    "edgecolor": "#CBD5E1",
    "linewidth": 0.5,
    "alpha": 0.82,
}
MODEL_STYLE = {
    "Gemma E2B": {"color": "#0077BB", "marker": "o"},
    "Gemma E4B": {"color": "#EE7733", "marker": "s"},
}
MODEL_ORDER = ["Gemma E2B", "Gemma E4B", "Qwen 0.8B", "Qwen 2B", "Qwen 4B"]
MODEL_CODES = {
    "Gemma E2B": "E2B",
    "Gemma E4B": "E4B",
    "Qwen 0.8B": "Q35-0.8B",
    "Qwen 2B": "Q35-2B",
    "Qwen 4B": "Q35-4B",
}
METHOD_ORDER = ["Full context", "GraphRAG v3", "GraphRAG v4", "Vector RAG"]
TOP_K = [2, 5, 10, 20, 50]

STRATEGY_RUNS: dict[str, dict[str, str]] = {
    "Gemma E2B": {
        "Full context": "lap_locomo200_e2b_fullcontext",
        "GraphRAG v3": "lap_locomo200_e2b_graphrag_v3_prefix",
        "GraphRAG v4": "lap_locomo200_e2b_graphrag_v4_prefix",
        "Vector RAG": "lap_locomo200_e2b_vr_prefix",
    },
    "Gemma E4B": {
        "Full context": "lap_locomo200_e4b_fullcontext",
        "GraphRAG v3": "lap_locomo200_e4b_graphrag_v3_prefix",
        "GraphRAG v4": "lap_locomo200_e4b_graphrag_v4_prefix",
        "Vector RAG": "lap_locomo200_e4b_vr_prefix",
    },
    "Qwen 0.8B": {
        "Full context": "lap_locomo200_q35_08b_fullcontext",
        "GraphRAG v3": "lap_locomo200_q35_08b_graphrag_v3_prefix",
        "GraphRAG v4": "lap_locomo200_q35_08b_graphrag_v4_prefix",
        "Vector RAG": "lap_locomo200_q35_08b_vr_prefix",
    },
    "Qwen 2B": {
        "Full context": "lap_locomo200_q35_2b_fullcontext",
        "GraphRAG v3": "lap_locomo200_q35_2b_graphrag_v3_prefix",
        "GraphRAG v4": "lap_locomo200_q35_2b_graphrag_v4_prefix",
        "Vector RAG": "lap_locomo200_q35_2b_vr_prefix",
    },
    "Qwen 4B": {
        "Full context": "lap_locomo200_q35_4b_fullcontext",
        "GraphRAG v3": "lap_locomo200_q35_4b_graphrag_v3_prefix",
        "GraphRAG v4": "lap_locomo200_q35_4b_graphrag_v4_prefix",
        "Vector RAG": "lap_locomo200_q35_4b_vr_prefix",
    },
}

TOPK_RUNS = {
    "Gemma E2B": {k: f"lap_locomo100_e2b_vr_prefix_k{k}" for k in TOP_K},
    "Gemma E4B": {k: f"lap_locomo100_e4b_vr_prefix_k{k}" for k in TOP_K},
}

# The record ladder: v1 retrieves by matching names and embeds nothing, so the task prefixes do
# not apply to it; every later rung is the prefixed record run of its stage.
REPAIR_RUNS = [
    ("GraphRAG v1\nbaseline", "lap_locomo200_e2b_graphrag", "GraphRAG v3"),
    ("GraphRAG v2\nvector-first", "lap_locomo200_e2b_graphrag_v2_prefix", "GraphRAG v3"),
    ("GraphRAG v3\nextraction repair", "lap_locomo200_e2b_graphrag_v3_prefix", "GraphRAG v3"),
    ("GraphRAG v4\n+ source turns", "lap_locomo200_e2b_graphrag_v4_prefix", "GraphRAG v4"),
    ("Vector RAG\nraw turns", "lap_locomo200_e2b_vr_prefix", "Vector RAG"),
]

# Retrieval cells use the prefixed record runs, as tab:ch4_lme does; full context has no embedding
# step. The pre-repair graph cells (lme_strat60_*_gr) are cited in Ch 4 text only.
LME_RUNS = {
    (model, method): f"lme_strat60_{code}_{suffix}"
    for model, code in (("Gemma E2B", "e2b"), ("Gemma E4B", "e4b"))
    for method, suffix in (
        ("Full context", "fc"),
        ("GraphRAG v3", "graphrag_v3_prefix"),
        ("GraphRAG v4", "graphrag_v4_prefix"),
        ("Vector RAG", "vr_prefix"),
    )
}
# fig07 only: the Qwen readers were run on LongMemEval under full context and vector retrieval
# (no graph cell exists). fig04 stays Gemma-only, as its caption says.
LME_QWEN_RUNS = {
    (model, method): f"lme_strat60_{code}_{suffix}"
    for model, code in (("Qwen 0.8B", "q35_08b"), ("Qwen 2B", "q35_2b"), ("Qwen 4B", "q35_4b"))
    for method, suffix in (("Full context", "fc"), ("Vector RAG", "vr_prefix"))
}

LME_CATEGORIES = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "knowledge-update",
    "temporal-reasoning",
]
LME_CATEGORY_LABELS = [
    "Single-session user",
    "Single-session assistant",
    "Preference",
    "Multi-session",
    "Knowledge update",
    "Temporal reasoning",
]

# fig10 shows the Ch 7 record comparison and nothing else: every run below must use the prefixed
# retrieval pipeline (embed_prefix=nomic) and the repaired selector, so it matches tab:spe_results
# and the prefixed oracle Ch 7 quotes. Until 2026-09-19 this figure showed the pre-correction,
# unprefixed development sequence instead, which disagreed with every number in the chapter.
# SPE-1 was only ever run unprefixed, so it stays out until a prefixed SPE-1 run exists.
# The guard is spe_story_records(), which reads embed_prefix from each run's config; a run-id
# substring check does NOT work, because the repaired SPE run ids (*_spe2fix_*) carry no "_prefix".
SPE_PREFIX = "nomic"
SPE_POINTS: dict[str, list[tuple[str, str, str]]] = {
    model: [
        ("GraphRAG v3", "GraphRAG v3", f"lap_locomo200_{code}_graphrag_v3_prefix"),
        ("SPE-2 b240", "SPE-2", f"lap_locomo200_{code}_spe2fix_gap_b240"),
        ("SPE-2 b370", "SPE-2", f"lap_locomo200_{code}_spe2fix_gap_b370"),
        ("GraphRAG v4", "GraphRAG v4", f"lap_locomo200_{code}_graphrag_v4_prefix"),
        ("Vector RAG", "Vector RAG", f"lap_locomo200_{code}_vr_prefix"),
    ]
    for model, code in (("Gemma E2B", "e2b"), ("Gemma E4B", "e4b"))
}

GRAPH_BUILD_RUNS = {
    "E2B GraphRAG v1": "lap_locomo100_e2b_graphrag",
    "E2B GraphRAG v2": "lap_locomo100_e2b_graphrag_v2",
    "E2B GraphRAG v3": "lap_locomo100_e2b_graphrag_v3",
    "E4B GraphRAG v1": "lap_locomo100_e4b_graphrag",
    "E4B GraphRAG v3": "lap_locomo200_e4b_graphrag_v3",
}

PHONE_RUNS: dict[tuple[str, str], list[str]] = {
    ("Gemma E2B", "vectorrag"): [
        "phone_locomo20_e2b_vectorrag",
        "phone_locomo20_e2b_vectorrag_rep2",
        "phone_locomo20_e2b_vectorrag_rep3",
    ],
    ("Gemma E2B", "fullcontext"): [
        "phone_locomo20_e2b_fullcontext",
        "phone_locomo20_e2b_fullcontext_rep2",
        "phone_locomo20_e2b_fullcontext_rep3",
    ],
    ("Gemma E4B", "vectorrag"): [
        "phone_locomo20_e4b_vectorrag",
        "phone_locomo20_e4b_vectorrag_rep2",
        "phone_locomo20_e4b_vectorrag_rep3",
    ],
    ("Gemma E4B", "fullcontext"): ["phone_locomo20_e4b_fullcontext"],
    ("Qwen 0.8B", "vectorrag"): [
        "phone_locomo20_q35_08b_vectorrag",
        "phone_locomo20_q35_08b_vectorrag_rep2",
        "phone_locomo20_q35_08b_vectorrag_rep3",
    ],
    ("Qwen 0.8B", "fullcontext"): ["phone_locomo20_q35_08b_fullcontext"],
    ("Qwen 2B", "vectorrag"): [
        "phone_locomo20_q35_2b_vectorrag",
        "phone_locomo20_q35_2b_vectorrag_rep2",
        "phone_locomo20_q35_2b_vectorrag_rep3",
    ],
    ("Qwen 2B", "fullcontext"): ["phone_locomo20_q35_2b_fullcontext"],
    ("Qwen 4B", "vectorrag"): [
        "phone_locomo20_q35_4b_vectorrag",
        "phone_locomo20_q35_4b_vectorrag_rep2",
        "phone_locomo20_q35_4b_vectorrag_rep3",
    ],
    ("Qwen 4B", "fullcontext"): ["phone_locomo20_q35_4b_fullcontext"],
}


def _read_csv(path: Path, key: str | None = None) -> Any:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    if key is None:
        return rows
    indexed = {row[key]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"duplicate {key} in {path}")
    return indexed


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def _record(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {column: "" for column in MANIFEST_COLUMNS}
    row.update(overrides)
    return row


def wilson_interval(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0 or correct < 0 or correct > n:
        raise ValueError(f"invalid binomial inputs: {correct}/{n}")
    p = correct / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def _accuracy_record(
    *,
    figure: str,
    panel: str,
    dataset: str,
    run: str,
    condition: str,
    series: str,
    accuracy: float,
    n: int,
    source: str,
    evidence: str,
) -> dict[str, object]:
    correct = round(accuracy * n)
    low, high = wilson_interval(correct, n)
    return _record(
        figure=figure,
        panel=panel,
        dataset=dataset,
        run=run,
        condition=condition,
        series=series,
        metric="judge_accuracy",
        value=accuracy,
        low=low,
        high=high,
        unit="proportion",
        n=n,
        replicates=1,
        evidence=evidence,
        source=source,
    )


def strategy_records(project_root: Path) -> list[dict[str, object]]:
    path = project_root / "results" / "master_table.csv"
    master = _read_csv(path, "dir")
    out: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        for method in METHOD_ORDER:
            run = STRATEGY_RUNS[model][method]
            row = master[run]
            if int(row["n"]) != 200 or int(row["n_judged"]) != 200:
                raise ValueError(f"{run}: expected 200 fully judged rows")
            out.append(
                _accuracy_record(
                    figure="fig01_strategy_landscape",
                    panel="LoCoMo accuracy",
                    dataset="locomo",
                    run=run,
                    condition=model,
                    series=method,
                    accuracy=float(row["judge"]),
                    n=200,
                    source=path.relative_to(project_root).as_posix(),
                    evidence="direct_box",
                )
            )
    return out


def retrieval_records(project_root: Path) -> list[dict[str, object]]:
    master_path = project_root / "results" / "master_table.csv"
    completeness_path = project_root / "results" / "completeness_table.csv"
    master = _read_csv(master_path, "dir")
    completeness = _read_csv(completeness_path, "run")
    out: list[dict[str, object]] = []

    for model in ("Gemma E2B", "Gemma E4B"):
        for k in TOP_K:
            run = TOPK_RUNS[model][k]
            row = master[run]
            if int(row["n"]) != 100 or int(row["n_judged"]) != 100:
                raise ValueError(f"{run}: expected 100 fully judged rows")
            out.append(
                _accuracy_record(
                    figure="fig02_retrieval_depth",
                    panel="Accuracy",
                    dataset="locomo",
                    run=run,
                    condition=str(k),
                    series=model,
                    accuracy=float(row["judge"]),
                    n=100,
                    source=master_path.relative_to(project_root).as_posix(),
                    evidence="direct_box",
                )
            )

    for k in TOP_K:
        e2_run = TOPK_RUNS["Gemma E2B"][k]
        e4_run = TOPK_RUNS["Gemma E4B"][k]
        e2_master, e4_master = master[e2_run], master[e4_run]
        if not math.isclose(float(e2_master["prompt_tok"]), float(e4_master["prompt_tok"]), abs_tol=0.02):
            raise ValueError(f"k={k}: prompt-token footprint is not reader-independent")
        e2_comp, e4_comp = completeness[e2_run], completeness[e4_run]
        recall = float(e2_comp["evidence_recall"])
        if not math.isclose(recall, float(e4_comp["evidence_recall"]), abs_tol=1e-9):
            raise ValueError(f"k={k}: evidence recall differs by reader")
        run_pair = f"{e2_run}|{e4_run}"
        out.extend(
            [
                _record(
                    figure="fig02_retrieval_depth",
                    panel="Evidence recall",
                    dataset="locomo",
                    run=run_pair,
                    condition=str(k),
                    series="Reader-independent",
                    metric="evidence_recall",
                    value=recall,
                    unit="proportion",
                    n=100,
                    replicates=1,
                    evidence="dataset_labelled",
                    source=completeness_path.relative_to(project_root).as_posix(),
                ),
                _record(
                    figure="fig02_retrieval_depth",
                    panel="Prompt footprint",
                    dataset="locomo",
                    run=run_pair,
                    condition=str(k),
                    series="Reader-independent",
                    metric="prompt_tokens",
                    value=float(e2_master["prompt_tok"]),
                    unit="tokens_per_answer",
                    n=100,
                    replicates=1,
                    evidence="direct_box",
                    source=master_path.relative_to(project_root).as_posix(),
                ),
            ]
        )
    return out


def repair_records(project_root: Path) -> list[dict[str, object]]:
    path = project_root / "results" / "master_table.csv"
    master = _read_csv(path, "dir")
    out: list[dict[str, object]] = []
    for stage, run, style_method in REPAIR_RUNS:
        row = master[run]
        if int(row["n"]) != 200 or int(row["n_judged"]) != 200:
            raise ValueError(f"{run}: expected n=200")
        out.append(
            _accuracy_record(
                figure="fig03_graph_repair",
                panel="Accuracy",
                dataset="locomo",
                run=run,
                condition=stage,
                series=style_method,
                accuracy=float(row["judge"]),
                n=200,
                source=path.relative_to(project_root).as_posix(),
                evidence="direct_box",
            )
        )
        out.append(
            _record(
                figure="fig03_graph_repair",
                panel="Prompt footprint",
                dataset="locomo",
                run=run,
                condition=stage,
                series=style_method,
                metric="prompt_tokens",
                value=float(row["prompt_tok"]),
                unit="tokens_per_answer",
                n=200,
                replicates=1,
                evidence="direct_box",
                source=path.relative_to(project_root).as_posix(),
            )
        )
    return out


def lme_records(project_root: Path) -> list[dict[str, object]]:
    master_path = project_root / "results" / "master_table.csv"
    category_path = project_root / "results" / "lme_category_table.csv"
    master = _read_csv(master_path, "dir")
    categories = _read_csv(category_path, "run")
    out: list[dict[str, object]] = []

    for (model, method), run in LME_RUNS.items():
        row = master[run]
        if int(row["n"]) != 60 or int(row["n_judged"]) != 60:
            raise ValueError(f"{run}: expected balanced n=60")
        out.append(
            _accuracy_record(
                figure="fig04_longmemeval_transfer",
                panel="Overall",
                dataset="longmemeval-s",
                run=run,
                condition=model,
                series=method,
                accuracy=float(row["judge"]),
                n=60,
                source=master_path.relative_to(project_root).as_posix(),
                evidence="direct_box",
            )
        )
        category_row = categories[run]
        for category in LME_CATEGORIES:
            n_category = int(category_row[f"{category}_n"])
            if n_category != 10:
                raise ValueError(f"{run}/{category}: expected n=10")
            accuracy = float(category_row[category])
            out.append(
                _accuracy_record(
                    figure="fig04_longmemeval_transfer",
                    panel="Question types",
                    dataset="longmemeval-s",
                    run=run,
                    condition=category,
                    series=f"{method}|{model}",
                    accuracy=accuracy,
                    n=10,
                    source=category_path.relative_to(project_root).as_posix(),
                    evidence="direct_box_category_descriptive",
                )
            )
    return out


def _phone_energy_spec(
    data: dict[str, Any], model_code: str, mode: str, cell: dict[str, Any]
) -> tuple[list[float], float, float]:
    key = f"{model_code}_{mode}"
    replicate = data.get("replicates", {}).get(key)
    if replicate and replicate.get("j_per_answer"):
        values = [float(value) for value in replicate["j_per_answer"]]
        return values, float(replicate["mean"]), float(replicate["sd"])
    value = float(cell["j_ans"])
    return [value], value, 0.0


def phone_efficiency_records(project_root: Path) -> list[dict[str, object]]:
    path = project_root / "results" / "golden_metric.json"
    data = _read_json(path)
    cells = {(str(cell["model"]), str(cell["mode"])): cell for cell in data["cells"]}
    out: list[dict[str, object]] = []
    mode_label = {"fullcontext": "Full context", "vectorrag": "Vector RAG"}

    for model in MODEL_ORDER:
        model_code = MODEL_CODES[model]
        for mode in ("fullcontext", "vectorrag"):
            cell = cells[(model_code, mode)]
            n = int(cell["n"])
            correct = round(float(cell["phone_acc"]) * n)
            accuracy = correct / n
            energy_values, energy_mean, energy_sd = _phone_energy_spec(data, model_code, mode, cell)
            low_acc, high_acc = wilson_interval(correct, n)
            cpkj = accuracy * 1000.0 / energy_mean
            method = mode_label[mode]
            common = dict(
                figure="fig05_phone_efficiency",
                dataset="locomo",
                run=f"phone_{model_code}_{mode}",
                condition=model,
                series=method,
                n=n,
                evidence="direct_phone",
                source=path.relative_to(project_root).as_posix(),
            )
            out.extend(
                [
                    _record(
                        **common,
                        panel="Correct per energy",
                        metric="correct_per_kj",
                        value=cpkj,
                        low=low_acc * 1000.0 / energy_mean,
                        high=high_acc * 1000.0 / energy_mean,
                        unit="correct_answers_per_kj",
                        replicates=len(energy_values),
                    ),
                    _record(
                        **common,
                        panel="Correct per energy",
                        metric="phone_accuracy",
                        value=accuracy,
                        low=low_acc,
                        high=high_acc,
                        unit="proportion",
                        replicates=1,
                    ),
                    _record(
                        **common,
                        panel="Energy per answer",
                        metric="joules_per_answer",
                        value=energy_mean,
                        error=energy_sd,
                        unit="joules",
                        replicates=len(energy_values),
                    ),
                ]
            )
            for sitting, value in enumerate(energy_values, start=1):
                sitting_common = dict(common)
                sitting_common["run"] = f"phone_{model_code}_{mode}_sitting_{sitting}"
                out.append(
                    _record(
                        **sitting_common,
                        panel="Energy per answer",
                        metric="joules_per_answer_sitting",
                        value=value,
                        unit="joules",
                        replicates=1,
                    )
                )
    return out


def phone_run_length_records(project_root: Path) -> list[dict[str, object]]:
    """Source-of-record rows for the appendix-only n=100 E2B observation."""
    path = project_root / "results" / "golden_metric.json"
    data = _read_json(path)
    auxiliary = data["auxiliary"]["E2B_vectorrag_n100"]
    n20_mean = float(data["replicates"]["E2B_vectorrag"]["mean"])
    n100_joules = float(auxiliary["joules_per_answer"])
    common = dict(
        figure="app_ch5_run_length",
        panel="Run-length observation",
        dataset="locomo",
        run=str(auxiliary["run"]),
        condition="Gemma E2B Vector RAG, n=100",
        series="Direct phone observation",
        n=int(auxiliary["n_scored"]),
        replicates=1,
        evidence="direct_phone_auxiliary",
        source=path.relative_to(project_root).as_posix(),
    )
    return [
        _record(**common, metric="phone_accuracy", value=float(auxiliary["accuracy"]), unit="proportion"),
        _record(**common, metric="energy_kj", value=float(auxiliary["energy_kj"]), unit="kilojoules"),
        _record(**common, metric="charge_mah", value=float(auxiliary["charge_mah"]), unit="milliampere_hours"),
        _record(**common, metric="joules_per_answer", value=n100_joules, unit="joules"),
        _record(
            **common,
            metric="n20_to_n100_joules_per_answer_ratio",
            value=n20_mean / n100_joules,
            unit="ratio",
        ),
    ]


def _nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.abs(values - target).argmin())


def _rolling_median(values: np.ndarray, radius: int = 7) -> np.ndarray:
    return np.array(
        [np.median(values[max(0, i - radius) : min(len(values), i + radius + 1)]) for i in range(len(values))],
        dtype=float,
    )


def phone_pilot_records(project_root: Path) -> list[dict[str, object]]:
    csv_path = project_root / "results_phone" / "_smoke_recovered" / "phone_gr.csv"
    run_path = (
        project_root
        / "results_phone"
        / "_smoke_recovered"
        / "phone_smoke_gr"
        / "locomo_graphrag_gemma4-e2b.jsonl"
    )
    judged_path = run_path.parent / "judged.jsonl"
    resource_rows = _read_csv(csv_path)
    run_rows = _read_jsonl(run_path)
    judged_rows = _read_jsonl(judged_path)
    first = run_rows[0]
    ingest_start = float(first["ingest_start_ts"])
    ingest_end = float(first["ingest_end_ts"])
    query_end = max(float(row["query_end_ts"]) for row in run_rows)

    timestamps = np.array([float(row["timestamp"]) for row in resource_rows], dtype=float)
    soc_temp = np.array([float(row["soc_temp_c"]) for row in resource_rows], dtype=float)
    battery_temp = np.array([float(row["battery_temp_c"]) for row in resource_rows], dtype=float)
    charge_uah = np.array([float(row["battery_charge_counter_uah"]) for row in resource_rows], dtype=float)
    voltage_mv = np.array([float(row["battery_voltage_mv"]) for row in resource_rows], dtype=float)
    t0 = timestamps[0]
    start_idx = _nearest_index(timestamps, ingest_start)
    ingest_end_idx = _nearest_index(timestamps, ingest_end)
    query_end_idx = _nearest_index(timestamps, query_end)
    # Selected includes from true idle baseline (index 0) through query end
    selected = np.arange(0, query_end_idx + 1)

    selected_ts = timestamps[selected]
    elapsed_min = (selected_ts - t0) / 60.0
    soc_smooth = _rolling_median(soc_temp[selected])
    battery_smooth = _rolling_median(battery_temp[selected])
    interval_charge_uah = charge_uah[selected[:-1]] - charge_uah[selected[1:]]
    interval_voltage_v = (voltage_mv[selected[:-1]] + voltage_mv[selected[1:]]) / 2000.0
    cumulative_kj = np.r_[0.0, np.cumsum(interval_charge_uah * interval_voltage_v * 0.0036)] / 1000.0
    cumulative_kj = np.maximum(0.0, cumulative_kj)
    downsample = np.unique(np.r_[np.arange(0, len(selected), 15), len(selected) - 1])

    out: list[dict[str, object]] = []
    source = csv_path.relative_to(project_root).as_posix()
    for idx in downsample:
        minute = float(elapsed_min[idx])
        if selected_ts[idx] < ingest_start:
            phase = "warmup"
        elif selected_ts[idx] <= ingest_end:
            phase = "construction"
        else:
            phase = "query"
        for metric, value, unit in (
            ("soc_temperature", soc_smooth[idx], "celsius"),
            ("battery_temperature", battery_smooth[idx], "celsius"),
            ("cumulative_energy", cumulative_kj[idx], "kilojoules"),
        ):
            out.append(
                _record(
                    figure="fig06_phone_graphrag_pilot",
                    panel="Timeline",
                    dataset="locomo",
                    run="phone_smoke_gr",
                    condition=f"{minute:.5f}",
                    series=phase,
                    metric=metric,
                    value=float(value),
                    unit=unit,
                    n=1,
                    replicates=1,
                    evidence="direct_phone_pilot",
                    source=source,
                )
            )

    # Reported energies use the thesis's one battery method; the cumulative trace above only draws it.
    construction_kj = battery_measurement(csv_path.parent, ingest_start, ingest_end, csv_name=csv_path.name).energy_kj
    query_kj = battery_measurement(csv_path.parent, ingest_end, query_end, csv_name=csv_path.name).energy_kj
    # soc_temp_c is the hottest of all readable thermal zones (profile_resources.get_soc_temp), and
    # cpu_freq_max_mhz the fastest of all eight cores, so both are device-wide maxima.
    freq_max = np.array([float(row["cpu_freq_max_mhz"]) for row in resource_rows], dtype=float)
    construction = slice(start_idx, ingest_end_idx + 1)
    peak_idx = int(np.argmax(soc_temp[selected]))
    top_clock_s = (timestamps[selected][freq_max[selected] >= 3300.0] - t0).max()
    turns = int(first["n_history_turns"])
    extraction = first["ctx_meta"]["extraction_stats"]
    construction_start_min = (ingest_start - t0) / 60.0
    construction_end_min = (ingest_end - t0) / 60.0
    construction_duration_min = (ingest_end - ingest_start) / 60.0
    query_end_min = (query_end - t0) / 60.0

    summaries = {
        "workload_start_min": 0.0,
        "construction_start_min": construction_start_min,
        "construction_end_min": construction_end_min,
        "construction_duration_min": construction_duration_min,
        "query_end_min": query_end_min,
        "construction_energy_kj": construction_kj,
        "query_energy_kj": query_kj,
        "construction_joules_per_turn": construction_kj * 1000.0 / turns,
        "history_turns": turns,
        "extraction_batches_total": int(extraction["batches_total"]),
        "extraction_batches_failed": int(extraction["batches_failed"]),
        "triples_extracted": int(extraction["triples_extracted"]),
        "judge_correct": sum(int(row["judge_correct"]) for row in judged_rows),
        "judge_n": len(judged_rows),
        "soc_temperature_start": float(soc_temp[0]),
        "battery_temperature_start": float(battery_temp[0]),
        "soc_temperature_peak": float(soc_temp[selected].max()),
        "soc_temperature_peak_elapsed_s": float(timestamps[selected][peak_idx] - t0),
        "soc_temperature_construction_median": float(np.median(soc_temp[construction])),
        "cpu_freq_max_construction_median_mhz": float(np.median(freq_max[construction])),
        "cpu_freq_top_last_s": float(top_clock_s),
        "battery_temperature_peak": float(battery_temp[selected].max()),
        "battery_level_start": float(resource_rows[0]["battery_level_pct"]),
        "battery_level_end": float(resource_rows[query_end_idx]["battery_level_pct"]),
    }
    units = {
        "workload_start_min": "minutes",
        "construction_start_min": "minutes",
        "construction_end_min": "minutes",
        "construction_duration_min": "minutes",
        "query_end_min": "minutes",
        "construction_energy_kj": "kilojoules",
        "query_energy_kj": "kilojoules",
        "construction_joules_per_turn": "joules_per_turn",
        "history_turns": "turns",
        "extraction_batches_total": "batches",
        "extraction_batches_failed": "batches",
        "triples_extracted": "triples",
        "judge_correct": "answers",
        "judge_n": "answers",
        "soc_temperature_start": "celsius",
        "battery_temperature_start": "celsius",
        "soc_temperature_peak": "celsius",
        "soc_temperature_peak_elapsed_s": "seconds",
        "soc_temperature_construction_median": "celsius",
        "cpu_freq_max_construction_median_mhz": "megahertz",
        "cpu_freq_top_last_s": "seconds",
        "battery_temperature_peak": "celsius",
        "battery_level_start": "percent",
        "battery_level_end": "percent",
    }
    for metric, value in summaries.items():
        out.append(
            _record(
                figure="fig06_phone_graphrag_pilot",
                panel="Pilot summary",
                dataset="locomo",
                run="phone_smoke_gr",
                condition="early GraphRAG phone pilot",
                series="Direct S23 measurement",
                metric=metric,
                value=value,
                unit=units[metric],
                n=1 if metric not in {"judge_correct", "judge_n"} else len(judged_rows),
                replicates=1,
                evidence="direct_phone_pilot",
                source=(judged_path if metric.startswith("judge_") else csv_path).relative_to(project_root).as_posix(),
            )
        )
    return out


def accuracy_cost_records(project_root: Path) -> list[dict[str, object]]:
    path = project_root / "results" / "master_table.csv"
    master = _read_csv(path, "dir")
    source = path.relative_to(project_root).as_posix()
    out: list[dict[str, object]] = []
    groups: list[tuple[str, str, int, list[tuple[str, str, str]]]] = [
        (
            "LoCoMo n=200",
            "locomo",
            200,
            [
                (model, method, STRATEGY_RUNS[model][method])
                for model in MODEL_ORDER
                for method in METHOD_ORDER
            ],
        ),
        (
            "LongMemEval-S n=60",
            "longmemeval",
            60,
            [(model, method, run) for (model, method), run in {**LME_RUNS, **LME_QWEN_RUNS}.items()],
        ),
    ]
    for panel, dataset, expected_n, cells in groups:
        for model, method, run in cells:
            row = master[run]
            if int(row["n"]) != expected_n or int(row["n_judged"]) != expected_n:
                raise ValueError(f"{run}: expected fully judged n={expected_n}")
            common = dict(
                figure="fig07_accuracy_cost_frontiers",
                panel=panel,
                dataset=dataset,
                run=run,
                condition=model,
                series=method,
                n=expected_n,
                replicates=1,
                evidence="direct_box",
                source=source,
            )
            accuracy = float(row["judge"])
            low, high = wilson_interval(round(accuracy * expected_n), expected_n)
            out.extend(
                [
                    _record(
                        **common,
                        metric="judge_accuracy",
                        value=accuracy,
                        low=low,
                        high=high,
                        unit="proportion",
                    ),
                    _record(
                        **common,
                        metric="prompt_tokens",
                        value=float(row["prompt_tok"]),
                        unit="tokens_per_answer",
                    ),
                ]
            )
    return out


def retrieval_outcome_records(project_root: Path) -> list[dict[str, object]]:
    path = project_root / "results" / "completeness_table.csv"
    completeness = _read_csv(path, "run")
    source = path.relative_to(project_root).as_posix()
    labels = (
        "Evidence + correct",
        "Evidence + wrong",
        "No evidence + correct",
        "No evidence + wrong",
    )
    out: list[dict[str, object]] = []
    for model in ("Gemma E2B", "Gemma E4B"):
        for k in TOP_K:
            run = TOPK_RUNS[model][k]
            row = completeness[run]
            n = int(row["n"])
            if n != 100:
                raise ValueError(f"{run}: expected n=100")
            n_evidence = int(row["n_given_evidence"])
            n_no_evidence = int(row["n_given_no_evidence"])
            correct_evidence = round(float(row["acc_given_evidence"]) * n_evidence)
            correct_no_evidence = round(float(row["acc_given_no_evidence"]) * n_no_evidence)
            counts = (
                correct_evidence,
                n_evidence - correct_evidence,
                correct_no_evidence,
                n_no_evidence - correct_no_evidence,
            )
            if sum(counts) != n:
                raise ValueError(f"{run}: outcome counts do not sum to {n}")
            for label, count in zip(labels, counts):
                out.append(
                    _record(
                        figure="fig08_retrieval_outcomes",
                        panel=model,
                        dataset="locomo",
                        run=run,
                        condition=model,
                        series=label,
                        metric="outcome_share",
                        value=count / n,
                        unit="proportion",
                        n=n,
                        replicates=1,
                        evidence="dataset_labelled_plus_70b_judge",
                        source=source,
                    )
                )
    return out


def thinking_tradeoff_records(project_root: Path) -> list[dict[str, object]]:
    path = project_root / "results" / "thinking_truncation.json"
    data = _read_json(path)
    source = path.relative_to(project_root).as_posix()
    out: list[dict[str, object]] = []
    for condition, cell in data.items():
        n = int(cell["n"])
        dataset = "longmemeval" if "LME" in condition else "locomo"
        panel = "LongMemEval-S" if dataset == "longmemeval" else "LoCoMo"
        for series, run_key, acc_key, token_key in (
            ("Base", "base_cell", "base_all", "mean_comp_tok_base"),
            ("Thinking", "think_cell", "think_all", "mean_comp_tok_think"),
        ):
            accuracy = float(cell[acc_key]) / 100.0
            low, high = wilson_interval(round(accuracy * n), n)
            common = dict(
                figure="fig09_thinking_tradeoff",
                panel=panel,
                dataset=dataset,
                run=str(cell[run_key]),
                condition=condition,
                series=series,
                n=n,
                replicates=1,
                evidence="direct_box_matched",
                source=source,
            )
            out.extend(
                [
                    _record(
                        **common,
                        metric="judge_accuracy",
                        value=accuracy,
                        low=low,
                        high=high,
                        unit="proportion",
                    ),
                    _record(
                        **common,
                        metric="completion_tokens",
                        value=float(cell[token_key]),
                        unit="tokens_per_answer",
                    ),
                ]
            )
        out.append(
            _record(
                figure="fig09_thinking_tradeoff",
                panel=panel,
                dataset=dataset,
                run=str(cell["think_cell"]),
                condition=condition,
                series="Thinking",
                metric="thinking_truncation_rate",
                value=float(cell["truncation_rate_pct"]) / 100.0,
                unit="proportion",
                n=n,
                replicates=1,
                evidence="direct_box_done_reason",
                source=source,
            )
        )
    return out


def _run_embed_prefix(project_root: Path, run: str) -> str | None:
    """The retrieval prefix a run was actually produced with, read from its own config.

    Run ids do not carry this reliably: the repaired SPE cells use prefixed retrieval but are
    named `*_spe2fix_*`, with no "_prefix" in the id, so a substring check passed them straight
    into an unprefixed comparison (caught 2026-08-29). Config is the only trustworthy source."""
    summaries = sorted((project_root / "results" / run).glob("*_summary.json"))
    if not summaries:
        raise ValueError(f"{run}: no run summary, cannot verify the retrieval pipeline")
    config = _read_json(summaries[0]).get("config", {})
    return config.get("embed_prefix")


def spe_story_records(project_root: Path) -> list[dict[str, object]]:
    path = project_root / "results" / "master_table.csv"
    master = _read_csv(path, "dir")
    source = path.relative_to(project_root).as_posix()
    out: list[dict[str, object]] = []
    for model, points in SPE_POINTS.items():
        for label, family, run in points:
            prefix = _run_embed_prefix(project_root, run)
            if prefix != SPE_PREFIX:
                raise ValueError(
                    f"{run}: the SPE comparison uses the prefixed record pipeline, but its config "
                    f"sets embed_prefix={prefix!r}. Mixing pipelines on one axis compares the "
                    f"retrieval repair with the selection rule."
                )
            row = master[run]
            n = int(row["n"])
            if n != 200 or int(row["n_judged"]) != n:
                raise ValueError(f"{run}: expected fully judged n=200")
            common = dict(
                figure="fig10_spe_story",
                panel=model,
                dataset="locomo",
                run=run,
                condition=model,
                series=family,
                n=n,
                replicates=1,
                evidence="direct_box_record",
                source=source,
            )
            accuracy = float(row["judge"])
            low, high = wilson_interval(round(accuracy * n), n)
            out.extend(
                [
                    _record(
                        **common,
                        metric="judge_accuracy",
                        value=accuracy,
                        low=low,
                        high=high,
                        unit="proportion",
                    ),
                    _record(
                        **common,
                        metric="prompt_tokens",
                        value=float(row["prompt_tok"]),
                        unit="tokens_per_answer",
                    ),
                ]
            )
        # Post hoc oracle on this panel's own prefixed v3/v4 pair: the values Ch 7 quotes
        # (results/spe_oracle_bound.json, 35.0% and 33.5%).
        v3_run = next(run for _, family, run in points if family == "GraphRAG v3")
        v4_run = next(run for _, family, run in points if family == "GraphRAG v4")
        oracle = spe_oracle_bound.bound_for(v3_run, v4_run)
        common = dict(
            figure="fig10_spe_story",
            panel=model,
            dataset="locomo",
            run="oracle_bound",
            condition=model,
            series="Oracle (post hoc)",
            n=int(oracle["n"]),
            replicates=1,
            evidence="post_hoc_upper_bound",
            source=f"results/{v3_run}/judged.jsonl + results/{v4_run}/judged.jsonl",
        )
        out.append(_record(**common, metric="judge_accuracy", value=float(oracle["oracle_acc"]), unit="proportion"))
        out.append(_record(**common, metric="prompt_tokens", value=float(oracle["oracle_tokens"]), unit="tokens_per_answer"))
    return out


def graph_build_records(
    project_root: Path, pilot_rows: list[dict[str, object]] | None = None
) -> list[dict[str, object]]:
    path = project_root / "results" / "graph_cost_curve.json"
    data = _read_json(path)
    source = path.relative_to(project_root).as_posix()
    out: list[dict[str, object]] = []
    for label, run in GRAPH_BUILD_RUNS.items():
        cell = data["runs"][run]
        for turns, seconds in cell["points"].items():
            out.append(
                _record(
                    figure="fig11_graph_build_scaling",
                    panel="Box scaling",
                    dataset="locomo",
                    run=run,
                    condition=str(turns),
                    series=label,
                    metric="construction_seconds",
                    value=float(seconds),
                    unit="seconds",
                    n=1,
                    replicates=1,
                    evidence="direct_box_conversation_build",
                    source=source,
                )
            )
        for metric, value, unit in (
            ("seconds_per_turn", cell["s_per_turn"], "seconds_per_turn"),
            ("intercept_seconds", cell["intercept_s"], "seconds"),
            ("fit_r2", cell["r2"], "proportion"),
        ):
            out.append(
                _record(
                    figure="fig11_graph_build_scaling",
                    panel="Fit summary",
                    dataset="locomo",
                    run=run,
                    condition=label,
                    series=label,
                    metric=metric,
                    value=float(value),
                    unit=unit,
                    n=int(cell["n_convs"]),
                    replicates=1,
                    evidence="linear_fit_to_direct_box_builds",
                    source=source,
                )
            )

    if pilot_rows is None:
        pilot_rows = phone_pilot_records(project_root)
    summary = {
        str(row["metric"]): row
        for row in pilot_rows
        if row["panel"] == "Pilot summary"
    }
    turns = int(float(summary["history_turns"]["value"]))
    phone_source = str(summary["construction_duration_min"]["source"])
    for metric, value, unit in (
        # ingest_start_ts to ingest_end_ts, matching fig06 and Chapter 5. construction_end_min is
        # measured from run launch and carries ~45 s of pre-ingest setup that the box builds beside
        # it do not include, so it is not like-for-like here.
        ("construction_seconds", float(summary["construction_duration_min"]["value"]) * 60.0, "seconds"),
        ("construction_energy_kj", float(summary["construction_energy_kj"]["value"]), "kilojoules"),
        ("history_turns", turns, "turns"),
    ):
        out.append(
            _record(
                figure="fig11_graph_build_scaling",
                panel="Phone anchor",
                dataset="locomo",
                run="phone_smoke_gr",
                condition=f"{turns} turns",
                series="Direct S23 pilot",
                metric=metric,
                value=value,
                unit=unit,
                n=1,
                replicates=1,
                evidence="direct_phone_pilot",
                source=phone_source,
            )
        )
    return out


def _phone_run_resource_metrics(run_dir: Path) -> tuple[float, float, float]:
    run_files = sorted(run_dir.glob("locomo_*.jsonl"))
    if len(run_files) != 1:
        raise ValueError(f"{run_dir.name}: expected one raw LoCoMo JSONL, found {len(run_files)}")
    run_rows = _read_jsonl(run_files[0])
    query_seconds = [float(row["query_end_ts"]) - float(row["query_start_ts"]) for row in run_rows]
    starts = [float(row.get("ingest_start_ts") or row["query_start_ts"]) for row in run_rows]
    ends = [float(row["query_end_ts"]) for row in run_rows]
    t0, t1 = min(starts), max(ends)
    soc_values: list[float] = []
    ram_values: list[float] = []
    with (run_dir / "resources.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                timestamp = float(row["timestamp"])
                soc = float(row["soc_temp_c"])
                ram = float(row["ram_used_mb"])
            except (KeyError, TypeError, ValueError):
                continue
            if t0 <= timestamp <= t1 and str(row.get("battery_plugged", "")).lower() != "true":
                if math.isfinite(soc):
                    soc_values.append(soc)
                if math.isfinite(ram):
                    ram_values.append(ram)
    if not soc_values or not ram_values:
        raise ValueError(f"{run_dir.name}: no usable resource samples in run span")
    return float(np.median(query_seconds)), max(soc_values), max(ram_values)


def _mean_sd(values: list[float]) -> tuple[float, float]:
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean, sd


def phone_workload_records(project_root: Path) -> list[dict[str, object]]:
    golden_path = project_root / "results" / "golden_metric.json"
    data = _read_json(golden_path)
    cells = {(str(cell["model"]), str(cell["mode"])): cell for cell in data["cells"]}
    mode_label = {"fullcontext": "Full context", "vectorrag": "Vector RAG"}
    out: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        model_code = MODEL_CODES[model]
        for mode in ("fullcontext", "vectorrag"):
            run_names = PHONE_RUNS[(model, mode)]
            resource_values = [
                _phone_run_resource_metrics(project_root / "results_phone" / run_name)
                for run_name in run_names
            ]
            latency_mean, latency_sd = _mean_sd([values[0] for values in resource_values])
            temp_mean, temp_sd = _mean_sd([values[1] for values in resource_values])
            ram_mean, ram_sd = _mean_sd([values[2] for values in resource_values])
            cell = cells[(model_code, mode)]
            energy_values, energy_mean, energy_sd = _phone_energy_spec(data, model_code, mode, cell)
            if len(energy_values) != len(run_names):
                raise ValueError(f"{model}/{mode}: resource and energy replicate counts differ")
            n = int(cell["n"])
            accuracy = float(cell["phone_acc"])
            low, high = wilson_interval(round(accuracy * n), n)
            method = mode_label[mode]
            run = "|".join(run_names)
            resource_source = " + ".join(
                (Path("results_phone") / run_name / "resources.csv").as_posix()
                for run_name in run_names
            )
            common = dict(
                figure="fig12_phone_workload_dashboard",
                dataset="locomo",
                run=run,
                condition=model,
                series=method,
                evidence="direct_phone",
            )
            out.extend(
                [
                    _record(
                        **common,
                        panel="Accuracy",
                        metric="phone_accuracy",
                        value=accuracy,
                        low=low,
                        high=high,
                        unit="proportion",
                        n=n,
                        replicates=len(run_names),
                        source=golden_path.relative_to(project_root).as_posix(),
                    ),
                    _record(
                        **common,
                        panel="Energy",
                        metric="joules_per_answer",
                        value=energy_mean,
                        error=energy_sd,
                        unit="joules",
                        n=n,
                        replicates=len(energy_values),
                        source=golden_path.relative_to(project_root).as_posix(),
                    ),
                    _record(
                        **common,
                        panel="Latency",
                        metric="query_latency_seconds",
                        value=latency_mean,
                        error=latency_sd,
                        unit="seconds",
                        n=n * len(run_names),
                        replicates=len(run_names),
                        source=resource_source,
                    ),
                    _record(
                        **common,
                        panel="Thermal",
                        metric="peak_soc_temperature",
                        value=temp_mean,
                        error=temp_sd,
                        unit="celsius",
                        n=len(run_names),
                        replicates=len(run_names),
                        source=resource_source,
                    ),
                    _record(
                        **common,
                        panel="Memory",
                        metric="peak_ram_mb",
                        value=ram_mean,
                        error=ram_sd,
                        unit="megabytes",
                        n=len(run_names),
                        replicates=len(run_names),
                        source=resource_source,
                    ),
                ]
            )
            # One row per sitting, drawn as grey dots. Answers are identical across sittings
            # (greedy decoding), so only the resource axes carry sittings, not accuracy.
            for run_name, energy, (latency, temperature, ram) in zip(run_names, energy_values, resource_values):
                sitting = {**common, "run": run_name, "n": n, "replicates": 1}
                for panel, metric, value, unit, source in (
                    ("Energy", "joules_per_answer", energy, "joules", golden_path.relative_to(project_root).as_posix()),
                    ("Latency", "query_latency_seconds", latency, "seconds", f"results_phone/{run_name}/locomo_*.jsonl"),
                    ("Thermal", "peak_soc_temperature", temperature, "celsius", f"results_phone/{run_name}/resources.csv"),
                    ("Memory", "peak_ram_mb", ram, "megabytes", f"results_phone/{run_name}/resources.csv"),
                ):
                    out.append(_record(**sitting, panel=panel, metric=f"{metric}_sitting", value=value, unit=unit, source=source))
    return out


def collect_source_data(project_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    records.extend(strategy_records(project_root))
    records.extend(retrieval_records(project_root))
    records.extend(repair_records(project_root))
    records.extend(lme_records(project_root))
    records.extend(phone_efficiency_records(project_root))
    records.extend(phone_run_length_records(project_root))
    pilot_rows = phone_pilot_records(project_root)
    records.extend(pilot_rows)
    records.extend(accuracy_cost_records(project_root))
    records.extend(retrieval_outcome_records(project_root))
    records.extend(thinking_tradeoff_records(project_root))
    records.extend(spe_story_records(project_root))
    records.extend(graph_build_records(project_root, pilot_rows))
    records.extend(phone_workload_records(project_root))
    figures = {str(row["figure"]) for row in records}
    expected = {
        "fig01_strategy_landscape",
        "fig02_retrieval_depth",
        "fig03_graph_repair",
        "fig04_longmemeval_transfer",
        "fig05_phone_efficiency",
        "fig06_phone_graphrag_pilot",
        "fig07_accuracy_cost_frontiers",
        "fig08_retrieval_outcomes",
        "fig09_thinking_tradeoff",
        "fig10_spe_story",
        "fig11_graph_build_scaling",
        "fig12_phone_workload_dashboard",
        "app_ch5_run_length",
    }
    if figures != expected:
        raise ValueError(f"figure set mismatch: {figures ^ expected}")
    return records


def configure_style(style: str = "paper") -> None:
    global _STYLE
    _STYLE = style
    # One palette per style, shared with the other figure scripts through visual_style. Mutated in
    # place because the plot functions read METHOD_STYLE directly.
    palette = PAPER_PALETTE if style == "paper" else PRESENTATION_PALETTE
    for method in ("Full context", "GraphRAG v3", "GraphRAG v4", "Vector RAG"):
        METHOD_STYLE[method]["color"] = palette[method]
    # Readers are a different kind of entity from strategies. On slides they borrow two strategy
    # hues; in the thesis they are ordered by size on a lightness ramp, darker for the larger model.
    reader_colors = (
        {"Gemma E2B": "#0077BB", "Gemma E4B": "#EE7733"}
        if style == "presentation"
        else {"Gemma E2B": "#8A96A6", "Gemma E4B": "#1E293B"}
    )
    for model, color in reader_colors.items():
        MODEL_STYLE[model]["color"] = color
    if style == "paper":
        configure_paper_style()
    else:
        plt.rcdefaults()
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 10,
                "axes.titlesize": 12,
                "axes.titleweight": "bold",
                "axes.labelsize": 10.5,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 8.8,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.linewidth": 0.8,
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "savefig.facecolor": "white",
                "savefig.bbox": "tight",
                "savefig.pad_inches": 0.08,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "svg.fonttype": "none",
            }
        )


def _fs(presentation: tuple[float, float], paper_height: float) -> tuple[float, float]:
    """Canvas size: the slide canvas for presentation, the thesis text width for paper."""
    return presentation if _STYLE == "presentation" else paper_size(paper_height)


INK = "#1E293B"


def _ink(series_color: str) -> str:
    """Value labels: series-colored on slides, ink in the thesis. A colored marker beside the
    label already carries identity, and light hues such as amber do not read as text on white."""
    return series_color if _STYLE == "presentation" else INK


def _note(target: Any, *args: Any, **kwargs: Any) -> Any:
    """An explanatory note drawn inside the figure. Slides keep it; in the thesis the \\caption
    carries it, so a copy inside the figure would only shrink with it and drift from the caption."""
    return target.text(*args, **kwargs) if _STYLE == "presentation" else None


def _layout(fig: Any, **presentation_margins: float) -> None:
    """Slides keep their hand-set margins. Thesis figures use constrained layout instead: their text
    is larger relative to the canvas, so margins tuned for the slide canvas no longer fit."""
    if _STYLE == "presentation":
        fig.subplots_adjust(**presentation_margins)
    else:
        fig.set_layout_engine("constrained")


def _fig_legend(fig: Any, *args: Any, bbox_to_anchor: Any = None, **kwargs: Any) -> Any:
    """A legend below the panels: pinned in figure coordinates on slides, given reserved space by
    constrained layout in the thesis so it cannot collide with the axis labels."""
    if _STYLE == "presentation":
        return fig.legend(*args, bbox_to_anchor=bbox_to_anchor, **kwargs)
    fig.set_layout_engine("constrained")
    kwargs["loc"] = "outside lower center"
    kwargs.pop("fontsize", None)
    return fig.legend(*args, **kwargs)


def _save_all(fig: Any, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix in ("pdf", "svg", "png"):
        path = output_dir / f"{stem}.{suffix}"
        savefig_retry(fig, path, dpi=300 if suffix == "png" else None)
        paths.append(path)
    plt.close(fig)
    return paths


def _rows(records: Iterable[dict[str, object]], figure: str, metric: str | None = None) -> list[dict[str, object]]:
    return [
        row
        for row in records
        if row["figure"] == figure and (metric is None or row["metric"] == metric)
    ]


def plot_strategy_landscape_bars(records: list[dict[str, object]], output_dir: Path) -> list[Path]:
    rows = _rows(records, "fig01_strategy_landscape", "judge_accuracy")
    lookup = {(str(row["condition"]), str(row["series"])): row for row in rows}
    x = np.arange(len(MODEL_ORDER))
    bar_width = 0.18

    fig, ax = plt.subplots(figsize=_fs((7.8, 3.4), 3.0))

    for idx, method in enumerate(METHOD_ORDER):
        offset = (idx - 1.5) * bar_width
        vals = []
        lows = []
        highs = []
        for model in MODEL_ORDER:
            row = lookup[(model, method)]
            v = float(row["value"]) * 100
            l = float(row["low"]) * 100
            h = float(row["high"]) * 100
            vals.append(v)
            lows.append(v - l)
            highs.append(h - v)

        style = METHOD_STYLE[method]
        ax.bar(
            x + offset,
            vals,
            bar_width,
            yerr=[lows, highs],
            capsize=2.5,
            error_kw={"elinewidth": 1.1, "ecolor": "#334155"},
            color=style["color"],
            label=method,
            edgecolor="#FFFFFF",
            linewidth=0.5,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, fontweight="semibold", fontsize=8.8)
    ax.set_ylabel("70B-judge accuracy (%)", fontsize=9.0)
    ax.set_ylim(0, 52)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", ncol=4, frameon=False, bbox_to_anchor=(0.0, 1.14), fontsize=8.5)
    fig.tight_layout()
    return _save_all(fig, output_dir, "fig01_strategy_landscape_bars")


def plot_strategy_landscape(records: list[dict[str, object]], output_dir: Path) -> list[Path]:
    rows = _rows(records, "fig01_strategy_landscape", "judge_accuracy")
    lookup = {(str(row["condition"]), str(row["series"])): row for row in rows}
    fig, ax = plt.subplots(figsize=_fs((9.4, 5.2), 3.9))
    y_base = np.arange(len(MODEL_ORDER), dtype=float)
    offsets = {method: offset for method, offset in zip(METHOD_ORDER, (-0.27, -0.09, 0.09, 0.27))}

    for i in range(len(MODEL_ORDER)):
        if i % 2 == 0:
            ax.axhspan(i - 0.48, i + 0.48, color="#F6F7F8", zorder=0)
    ax.axhline(1.5, color="#B8BDC2", linewidth=1.0, zorder=1)

    for method in METHOD_ORDER:
        style = METHOD_STYLE[method]
        for i, model in enumerate(MODEL_ORDER):
            row = lookup[(model, method)]
            value = float(row["value"]) * 100
            low = float(row["low"]) * 100
            high = float(row["high"]) * 100
            y = y_base[i] + offsets[method]
            ax.errorbar(
                value,
                y,
                xerr=[[value - low], [high - value]],
                fmt=style["marker"],
                color=style["color"],
                markerfacecolor="white" if method == "GraphRAG v4" else style["color"],
                markeredgewidth=1.6,
                markersize=7.5,
                capsize=2.5,
                linewidth=1.25,
                zorder=3,
            )
            # At the upper whisker cap, not on the whisker, so the label covers neither marker nor interval.
            ax.annotate(
                f"{value:.1f}",
                (high, y),
                xytext=(3, 0),
                textcoords="offset points",
                va="center",
                color=_ink(style["color"]),
                fontsize=8.0,
                fontweight="bold",
                bbox=NUMBER_LABEL_BOX,
                zorder=5,
            )

    ax.set_yticks(y_base, MODEL_ORDER)
    ax.set_xlim(0, max(float(row["high"]) for row in rows) * 100 + 6)
    ax.set_ylim(len(MODEL_ORDER) - 0.45, -0.55)
    ax.set_xlabel("70B-judge accuracy (%)")
    ax.set_ylabel("")
    ax.xaxis.grid(True, color="#E2E5E8", linewidth=0.8)
    ax.set_axisbelow(True)
    legend = [
        Line2D(
            [0],
            [0],
            marker=METHOD_STYLE[method]["marker"],
            color=METHOD_STYLE[method]["color"],
            markerfacecolor="white" if method == "GraphRAG v4" else METHOD_STYLE[method]["color"],
            linewidth=0,
            markersize=7,
            label=method,
        )
        for method in METHOD_ORDER
    ]
    ax.legend(handles=legend, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    _note(ax, 0.0, -0.285, "Wilson 95% intervals; n=200 questions per cell. GraphRAG v3 = triples only; v4 restores source turns.", transform=ax.transAxes, fontsize=8.5, color="#555B60")
    _layout(fig, left=0.16, right=0.98, top=0.98, bottom=0.25)
    saved = _save_all(fig, output_dir, "fig01_strategy_landscape")
    # Also save as whiskers variant for explicit selection
    _save_all(fig, output_dir, "fig01_strategy_landscape_whiskers")
    return saved


def plot_retrieval_depth(records: list[dict[str, object]], output_dir: Path) -> list[Path]:
    acc_rows = _rows(records, "fig02_retrieval_depth", "judge_accuracy")
    recall_rows = _rows(records, "fig02_retrieval_depth", "evidence_recall")
    token_rows = _rows(records, "fig02_retrieval_depth", "prompt_tokens")
    acc = {(str(row["series"]), int(str(row["condition"]))): row for row in acc_rows}
    recall = {int(str(row["condition"])): float(row["value"]) for row in recall_rows}
    tokens = {int(str(row["condition"])): float(row["value"]) for row in token_rows}
    x = np.arange(len(TOP_K))
    fig = plt.figure(figsize=_fs((9.4, 6.1), 4.4))
    grid = fig.add_gridspec(2, 1, height_ratios=[2.25, 1], hspace=0.08)
    ax = fig.add_subplot(grid[0])
    ax_recall = fig.add_subplot(grid[1], sharex=ax)

    for model in ("Gemma E2B", "Gemma E4B"):
        style = MODEL_STYLE[model]
        values, lows, highs = [], [], []
        for k in TOP_K:
            row = acc[(model, k)]
            values.append(float(row["value"]) * 100)
            lows.append(float(row["low"]) * 100)
            highs.append(float(row["high"]) * 100)
        values_arr = np.array(values)
        ax.plot(x, values_arr, color=style["color"], marker=style["marker"], linewidth=2.1, markersize=7, label=model)
        ax.errorbar(x, values_arr, yerr=[values_arr - np.array(lows), np.array(highs) - values_arr], fmt="none", ecolor=style["color"], capsize=3, linewidth=1.15)
        label_dy = -2.2 if model == "Gemma E2B" else 2.2
        label_va = "top" if label_dy < 0 else "bottom"
        # The thesis prints every one of these values in tab:ch4_topk, directly above the figure,
        # so its version shows the shape and leaves the numbers to the table.
        for xi, value in (zip(x, values) if _STYLE == "presentation" else ()):
            ax.text(
                xi,
                value + label_dy,
                f"{value:.0f}",
                ha="center",
                va=label_va,
                color=_ink(style["color"]),
                fontsize=8.5,
                fontweight="bold",
                bbox=NUMBER_LABEL_BOX,
                zorder=5,
            )

    recall_values = np.array([recall[k] * 100 for k in TOP_K])
    ax_recall.fill_between(x, 0, recall_values, color="#D8E8F2", alpha=0.85)
    ax_recall.plot(x, recall_values, color="#333333", marker="o", linewidth=2.0)
    for xi, value in (zip(x, recall_values) if _STYLE == "presentation" else ()):
        ax_recall.text(xi, value + 4.2, f"{value:.0f}%", ha="center", fontsize=8.5, fontweight="bold")

    ax.set_ylabel("70B-judge accuracy (%)")
    ax.set_ylim(0, 57)
    ax.legend(loc="upper left", ncol=2)
    ax.yaxis.grid(True, color="#E2E5E8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelbottom=False)
    ax_recall.set_ylabel("Evidence\nrecall (%)")
    ax_recall.set_ylim(0, 100)
    ax_recall.yaxis.grid(True, color="#E2E5E8", linewidth=0.8)
    ax_recall.set_axisbelow(True)
    ax_recall.set_xticks(
        x,
        [f"k={k}\n{tokens[k]:,.0f} tokens" for k in TOP_K],
    )
    ax_recall.set_xlabel("Retrieved turns and measured mean prompt footprint per answer")
    _note(ax_recall, 0.0, -0.48, "Accuracy whiskers: Wilson 95% intervals. Evidence recall is dataset-labelled and reader-independent.", transform=ax_recall.transAxes, fontsize=8.5, color="#555B60")
    _layout(fig, left=0.11, right=0.98, top=0.98, bottom=0.19)
    return _save_all(fig, output_dir, "fig02_retrieval_depth")


def plot_graph_repair(records: list[dict[str, object]], output_dir: Path) -> list[Path]:
    acc_rows = _rows(records, "fig03_graph_repair", "judge_accuracy")
    token_rows = _rows(records, "fig03_graph_repair", "prompt_tokens")
    acc = {str(row["condition"]): row for row in acc_rows}
    tokens = {str(row["condition"]): row for row in token_rows}
    stages = [stage for stage, _, _ in REPAIR_RUNS]
    y = np.arange(len(stages))
    fig, (ax_acc, ax_tok) = plt.subplots(1, 2, figsize=_fs((9.4, 4.7), 3.0), sharey=True, gridspec_kw={"width_ratios": [1.2, 1]})

    graph_accuracy = [float(acc[stage]["value"]) * 100 for stage in stages[:4]]
    ax_acc.plot(graph_accuracy, y[:4], color="#AAB0B5", linewidth=2.0, zorder=1)
    graph_tokens = [float(tokens[stage]["value"]) for stage in stages[:4]]
    ax_tok.plot(graph_tokens, y[:4], color="#AAB0B5", linewidth=2.0, zorder=1)

    for idx, stage in enumerate(stages):
        acc_row = acc[stage]
        tok_row = tokens[stage]
        method = str(acc_row["series"])
        style = METHOD_STYLE[method]
        value = float(acc_row["value"]) * 100
        low = float(acc_row["low"]) * 100
        high = float(acc_row["high"]) * 100
        ax_acc.errorbar(value, idx, xerr=[[value - low], [high - value]], fmt=style["marker"], color=style["color"], markerfacecolor="white" if method == "GraphRAG v4" else style["color"], markeredgewidth=1.5, markersize=8, capsize=3, linewidth=1.3, zorder=3)
        # Offsets in points, so a label clears its whisker cap or marker at any axis range.
        ax_acc.annotate(f"{value:.1f}%", (high, idx), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8.8, color=_ink(style["color"]), fontweight="bold", bbox=NUMBER_LABEL_BOX, zorder=5)
        token_value = float(tok_row["value"])
        ax_tok.scatter(token_value, idx, color=style["color"], marker=style["marker"], s=62, facecolors="white" if method == "GraphRAG v4" else style["color"], linewidths=1.5, zorder=3)
        ax_tok.annotate(f"{token_value:.0f}", (token_value, idx), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8.8, color=_ink(style["color"]), fontweight="bold", bbox=NUMBER_LABEL_BOX, zorder=5)

    ax_acc.set_yticks(y, stages)
    ax_acc.set_ylim(len(stages) - 0.55, -0.55)
    # Room for the widest upper interval and its label; a fixed 42 clipped Vector RAG's (42.8%).
    ax_acc.set_xlim(0, max(float(acc[s]["high"]) for s in stages) * 100 + 10)
    ax_acc.set_xlabel("70B-judge accuracy (%)")
    ax_acc.set_title("Answer quality", loc="left")
    ax_acc.xaxis.grid(True, color="#E2E5E8", linewidth=0.8)
    ax_tok.set_xlim(450, 900)
    ax_tok.set_xlabel("Mean prompt tokens per answer")
    ax_tok.set_title("Query footprint", loc="left")
    ax_tok.xaxis.grid(True, color="#E2E5E8", linewidth=0.8)
    for axis in (ax_acc, ax_tok):
        axis.set_axisbelow(True)
        axis.axhline(3.5, color="#C3C7CB", linewidth=1.0)
    _note(fig, 0.22, 0.035, "Record repair ladder (LoCoMo n=200, Gemma E2B); v1 embeds nothing, later rungs use the task prefixes. Wilson 95% intervals.", fontsize=8.4, color="#555B60")
    _layout(fig, left=0.22, right=0.98, top=0.9, bottom=0.19, wspace=0.16)
    return _save_all(fig, output_dir, "fig03_graph_repair")


def plot_longmemeval_transfer(records: list[dict[str, object]], output_dir: Path) -> list[Path]:
    overall_rows = _rows(records, "fig04_longmemeval_transfer", "judge_accuracy")
    overall_rows = [row for row in overall_rows if row["panel"] == "Overall"]
    category_rows = [row for row in _rows(records, "fig04_longmemeval_transfer", "judge_accuracy") if row["panel"] == "Question types"]
    overall = {(str(row["condition"]), str(row["series"])): row for row in overall_rows}
    category = {(str(row["condition"]), str(row["series"])): row for row in category_rows}
    methods = METHOD_ORDER
    method_labels = {method: method for method in methods}
    models = ["Gemma E2B", "Gemma E4B"]
    # The colorbar below is built for whichever layout engine is active when it is created, so the
    # thesis version has to be constrained from the start rather than switched over in _layout.
    fig = plt.figure(figsize=_fs((11.2, 5.2), 5.6), layout="constrained" if _STYLE == "paper" else None)
    if _STYLE == "presentation":
        grid = fig.add_gridspec(1, 2, width_ratios=[0.9, 1.7], wspace=0.45)
    else:  # both panels carry a long label column; side by side at the text width they leave no plot
        grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.9])
    ax = fig.add_subplot(grid[0])
    ax_heat = fig.add_subplot(grid[1])

    y = np.arange(len(methods))
    offsets = {"Gemma E2B": -0.11, "Gemma E4B": 0.11}
    for model in models:
        model_style = MODEL_STYLE[model]
        for idx, method in enumerate(methods):
            row = overall[(model, method)]
            value, low, high = (float(row[key]) * 100 for key in ("value", "low", "high"))
            yy = idx + offsets[model]
            ax.errorbar(value, yy, xerr=[[value - low], [high - value]], fmt=model_style["marker"], color=METHOD_STYLE[method]["color"], markerfacecolor=model_style["color"], markeredgecolor="white", markeredgewidth=0.7, markersize=8, capsize=3, linewidth=1.3)
            if _STYLE == "presentation":  # thesis: the values are in tab:ch4_lme, and at the text width
                # the two readers' labels of one method overlap
                ax.text(high + 1.8, yy, f"{value:.1f}", va="center", fontsize=8.6, color=_ink(METHOD_STYLE[method]["color"]), fontweight="bold", bbox=NUMBER_LABEL_BOX, zorder=5)
    ax.set_yticks(y, [method_labels[method] for method in methods])
    ax.set_ylim(len(methods) - 0.5, -0.5)
    ax.set_xlim(0, 75)
    ax.set_xlabel("Overall accuracy (%)")
    ax.set_title("Balanced benchmark result", loc="left")
    ax.xaxis.grid(True, color="#E2E5E8", linewidth=0.8)
    ax.set_axisbelow(True)
    model_legend = [Line2D([0], [0], marker=MODEL_STYLE[model]["marker"], color="none", markerfacecolor=MODEL_STYLE[model]["color"], markeredgecolor="white", markersize=8, label=model) for model in models]
    ax.legend(handles=model_legend, loc="upper right")

    columns = [(method, model) for method in methods for model in models]
    matrix = np.array(
        [
            [float(category[(cat, f"{method}|{model}")]["value"]) for method, model in columns]
            for cat in LME_CATEGORIES
        ]
    )
    im = ax_heat.imshow(matrix, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax_heat.text(j, i, f"{value * 100:.0f}", ha="center", va="center", color="white" if value >= 0.55 else "#111111", fontsize=9, fontweight="bold")
    ax_heat.set_yticks(np.arange(len(LME_CATEGORY_LABELS)), LME_CATEGORY_LABELS)
    # Reader under each column and the method once per group: with eight columns the combined
    # two-line labels run into each other in both styles.
    ax_heat.set_xticks(np.arange(len(columns)), [model.replace("Gemma ", "") for _, model in columns])
    for i, method in enumerate(methods):
        center = i * len(models) + (len(models) - 1) / 2
        ax_heat.annotate(method_labels[method], xy=(center, 0), xycoords=("data", "axes fraction"),
                         xytext=(0, -20), textcoords="offset points", ha="center", va="top")
    ax_heat.set_title("Question-type accuracy (n=10 per cell)", loc="left")
    ax_heat.tick_params(length=0)
    for boundary in np.arange(len(models), len(columns), len(models)) - 0.5:
        ax_heat.axvline(boundary, color="#94A3B8", linewidth=1.8, linestyle="-")
    colorbar = fig.colorbar(im, ax=ax_heat, fraction=0.035, pad=0.02)
    colorbar.set_label("Accuracy")
    colorbar.set_ticks([0, 0.5, 1])
    colorbar.set_ticklabels(["0%", "50%", "100%"])
    _note(fig, 0.08, 0.03, "LongMemEval-S stratified n=60: 10 questions from each of six types. Overall whiskers are Wilson 95% intervals; category cells are descriptive.", fontsize=8.6, color="#555B60")
    _layout(fig, left=0.15, right=0.97, top=0.91, bottom=0.17)
    return _save_all(fig, output_dir, "fig04_longmemeval_transfer")


def plot_phone_efficiency(records: list[dict[str, object]], output_dir: Path) -> list[Path]:
    cpkj_rows = _rows(records, "fig05_phone_efficiency", "correct_per_kj")
    energy_rows = _rows(records, "fig05_phone_efficiency", "joules_per_answer")
    sitting_rows = _rows(records, "fig05_phone_efficiency", "joules_per_answer_sitting")
    cpkj = {(str(row["condition"]), str(row["series"])): row for row in cpkj_rows}
    energy = {(str(row["condition"]), str(row["series"])): row for row in energy_rows}
    sittings: dict[tuple[str, str], list[float]] = {}
    for row in sitting_rows:
        sittings.setdefault((str(row["condition"]), str(row["series"])), []).append(float(row["value"]))

    fig, (ax_rate, ax_energy) = plt.subplots(1, 2, figsize=_fs((11.2, 5.2), 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
    y_base = np.arange(len(MODEL_ORDER), dtype=float)
    offsets = {"Full context": -0.12, "Vector RAG": 0.12}
    for i in range(len(MODEL_ORDER)):
        if i % 2 == 0:
            for axis in (ax_rate, ax_energy):
                axis.axhspan(i - 0.45, i + 0.45, color="#F6F7F8", zorder=0)
    for i, model in enumerate(MODEL_ORDER):
        for method in ("Full context", "Vector RAG"):
            row = cpkj[(model, method)]
            style = METHOD_STYLE[method]
            value, low, high = (float(row[key]) for key in ("value", "low", "high"))
            yy = i + offsets[method]
            ax_rate.errorbar(value, yy, xerr=[[value - low], [high - value]], fmt=style["marker"], color=style["color"], markersize=7.5, capsize=3, linewidth=1.2, zorder=3)
            # Past the upper whisker cap, so the label covers neither the marker nor its interval.
            ax_rate.annotate(f"{value:.2f}", (high, yy), xytext=(3, 0), textcoords="offset points", va="center", color=_ink(style["color"]), fontsize=8.2, fontweight="bold", bbox=NUMBER_LABEL_BOX, zorder=5)

            e_row = energy[(model, method)]
            e_value = float(e_row["value"])
            e_error = float(e_row["error"] or 0.0)
            points = sittings[(model, method)]
            jitter = np.linspace(-0.045, 0.045, len(points)) if len(points) > 1 else np.array([0.0])
            ax_energy.errorbar(e_value, yy, xerr=e_error if e_error > 0 else None, fmt=style["marker"], color=style["color"], markeredgewidth=1.5, markersize=7.5, capsize=3, linewidth=1.2, zorder=3)
            # Sitting dots sit above the mean marker so a one-sitting cell still shows its single dot.
            ax_energy.scatter(points, np.full(len(points), yy) + jitter, s=16, color="#5F666B", edgecolors="white", linewidths=0.5, zorder=4)

    ax_rate.set_yticks(y_base, MODEL_ORDER)
    ax_energy.set_yticks(y_base, [])
    for axis in (ax_rate, ax_energy):
        axis.set_ylim(len(MODEL_ORDER) - 0.5, -0.5)
        axis.set_axisbelow(True)
    ax_rate.set_xlim(0, 1.75)
    ax_rate.set_xlabel("Correct answers per kilojoule")
    ax_rate.set_title("Correct answers per unit of energy", loc="left")
    ax_rate.xaxis.grid(True, color="#E2E5E8", linewidth=0.8)
    ax_energy.set_xscale("log")
    ax_energy.set_xlim(180, 3300)
    # 2,000 and 3,000 run together on a log axis at the thesis panel width.
    energy_ticks = [200, 300, 500, 1000, 2000, 3000] if _STYLE == "presentation" else [200, 300, 500, 1000, 2000]
    ax_energy.set_xticks(energy_ticks)
    ax_energy.set_xticklabels([f"{tick:,}" for tick in energy_ticks])
    ax_energy.tick_params(axis="x", which="minor", labelbottom=False)
    ax_energy.set_xlabel("Joules per answer on S23 (log scale)")
    ax_energy.set_title("Direct energy measurements", loc="left")
    ax_energy.xaxis.grid(True, which="both", color="#E2E5E8", linewidth=0.8)
    legend = [Line2D([0], [0], marker=METHOD_STYLE[method]["marker"], color=METHOD_STYLE[method]["color"], linewidth=0, markersize=7, label=method) for method in ("Full context", "Vector RAG")]
    legend.append(Line2D([0], [0], marker="o", color="#5F666B", markeredgecolor="white", linewidth=0, markersize=5, label="one sitting"))
    _fig_legend(fig, handles=legend, loc="lower center", bbox_to_anchor=(0.66, 0.075), ncol=3, fontsize=8.2, frameon=False)
    _note(fig, 0.075, 0.02, "Rate intervals propagate Wilson 95% accuracy intervals through the measured mean energy. Energy whiskers are between-sitting SD.\nFull context was measured in one sitting for every reader except Gemma E2B; every vector cell has three. Phone vector retrieval is unprefixed (Chapter 3).", fontsize=8.5, color="#555B60")
    _layout(fig, left=0.16, right=0.98, top=0.91, bottom=0.22, wspace=0.18)
    return _save_all(fig, output_dir, "fig05_phone_efficiency")


def plot_phone_graphrag_pilot(records: list[dict[str, object]], output_dir: Path) -> list[Path]:
    timeline = [row for row in _rows(records, "fig06_phone_graphrag_pilot") if row["panel"] == "Timeline"]
    summary_rows = [row for row in _rows(records, "fig06_phone_graphrag_pilot") if row["panel"] == "Pilot summary"]
    summary = {str(row["metric"]): float(row["value"]) for row in summary_rows}
    times = sorted({float(str(row["condition"])) for row in timeline})
    series = {
        metric: {float(str(row["condition"])): float(row["value"]) for row in timeline if row["metric"] == metric}
        for metric in ("soc_temperature", "battery_temperature", "cumulative_energy")
    }
    soc = np.array([series["soc_temperature"][time] for time in times])
    battery = np.array([series["battery_temperature"][time] for time in times])
    energy = np.array([series["cumulative_energy"][time] for time in times])
    construction_start = summary.get("construction_start_min", 0.75)
    construction_end = summary["construction_end_min"]
    construction_dur = summary.get("construction_duration_min", construction_end - construction_start)
    query_end = summary["query_end_min"]

    fig = plt.figure(figsize=_fs((11.2, 5.5), 4.2))
    # The slide carries a side panel of headline numbers. In the thesis every one of them is in the
    # caption, and the panel would take a quarter of the text width from the traces.
    if _STYLE == "presentation":
        grid = fig.add_gridspec(2, 2, width_ratios=[3.8, 1.35], height_ratios=[1, 1], hspace=0.18, wspace=0.14)
    else:
        grid = fig.add_gridspec(2, 1, height_ratios=[1, 1])
    ax_temp = fig.add_subplot(grid[0, 0])
    ax_energy = fig.add_subplot(grid[1, 0], sharex=ax_temp)
    ax_facts = fig.add_subplot(grid[:, 1]) if _STYLE == "presentation" else None

    for axis in (ax_temp, ax_energy):
        axis.axvspan(0, construction_start, color="#FFF3E0", alpha=0.85, zorder=0)
        axis.axvspan(construction_start, construction_end, color="#D9F0EA", alpha=0.8, zorder=0)
        axis.axvspan(construction_end, query_end, color="#DCEAF4", alpha=0.9, zorder=0)
        axis.axvline(construction_start, color="#D4A373", linestyle=":", linewidth=0.9)
        axis.axvline(construction_end, color="#778087", linestyle="--", linewidth=1)
        axis.grid(True, axis="y", color="#E2E5E8", linewidth=0.8)
        axis.set_axisbelow(True)

    ax_temp.plot(times, soc, color="#CC3311", linewidth=1.6, label="Hottest thermal zone")
    ax_temp.plot(times, battery, color="#0077BB", linewidth=2.0, label="Battery")
    construction_median = summary["soc_temperature_construction_median"]
    ax_temp.axhline(construction_median, color="#CC3311", linestyle=":", alpha=0.45, linewidth=1.0)
    ax_temp.set_ylabel("Temperature (°C)")
    ax_temp.set_ylim(25, 103)
    # In the thesis the legend sits in the empty band between the two traces, where it hides no data.
    ax_temp.legend(loc="center left", bbox_to_anchor=(0.02, 0.52) if _STYLE == "presentation" else (0.30, 0.34),
                   ncol=1 if _STYLE == "presentation" else 2, framealpha=0.90, fontsize=8.0 if _STYLE == "presentation" else None)
    ax_temp.set_title("Thermal trace (Galaxy S23 Ultra, Snapdragon 8 Gen 2)", loc="left")
    ax_temp.tick_params(axis="x", labelbottom=False)

    # Phase labels. At the thesis width the load and query phases are too narrow to hold text; the
    # shading still marks them and the caption names them.
    if _STYLE == "presentation":
        ax_temp.text(construction_start / 2, 97.5, "load", ha="center", va="top", fontsize=7.8, color="#B06500", fontweight="bold")
    ax_temp.text((construction_start + construction_end) / 2, 97.5, f"graph construction ({construction_dur:.1f} min, 419 turns)", ha="center", va="top", fontsize=8.5, color="#176E5D", fontweight="bold")
    if _STYLE == "presentation":
        ax_temp.text((construction_end + query_end) / 2, 97.5, "4 queries", ha="center", va="top", fontsize=8.0, color="#25658D", fontweight="bold")

    # Thermal annotations. The trace has no governor state, so none of them attributes a cause.
    # The first sample is not an idle baseline: the servers were already loaded and the harness
    # starts 2 s after the sampler.
    soc_start = summary["soc_temperature_start"]
    # The caption gives the first sample; at the thesis width its label lands on the battery trace.
    if _STYLE == "presentation":
        ax_temp.annotate(f"First sample: {soc_start:.1f}°C",
                         xy=(0, soc_start), xytext=(2.0, 36),
                         arrowprops=dict(arrowstyle="->", color="#CC3311", lw=0.8),
                         fontsize=7.8, color="#CC3311", fontweight="bold")
    # Label from the full-resolution summary; anchor on the plotted (downsampled) maximum, which
    # can sit a fraction of a degree below it. The side panel and Chapter 5 quote the same value.
    soc_peak = summary["soc_temperature_peak"]
    peak_idx = int(np.argmax(soc))
    ax_temp.annotate(f"Peak: {soc_peak:.1f}°C at {summary['soc_temperature_peak_elapsed_s']:.0f} s",
                     xy=(times[peak_idx], soc[peak_idx]), xytext=(7.0, 88),
                     arrowprops=dict(arrowstyle="->", color="#8B0000", lw=0.8),
                     fontsize=7.6, color="#8B0000", fontweight="bold")
    clock_ghz = summary["cpu_freq_max_construction_median_mhz"] / 1000.0
    ax_temp.text(42, 64.0, f"Construction median {construction_median:.1f}°C; fastest core mostly {clock_ghz:.2f} GHz",
                 fontsize=7.8, color="#882200", style="italic", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75))
    bat_peak = summary["battery_temperature_peak"]
    ax_temp.annotate(f"Battery peak: {bat_peak:.1f}°C",
                     xy=(construction_end - 5, bat_peak), xytext=(construction_end - 22, 35),
                     arrowprops=dict(arrowstyle="->", color="#0077BB", lw=0.8),
                     fontsize=7.6, color="#0077BB", fontweight="bold")

    ax_energy.plot(times, energy, color="#6A3D9A", linewidth=2.5)
    ax_energy.fill_between(times, 0, energy, color="#6A3D9A", alpha=0.10)
    ax_energy.set_ylabel("Cumulative energy (kJ)")
    ax_energy.set_xlabel("Elapsed minutes since run launch")
    ax_energy.set_xlim(0, query_end)
    ax_energy.set_ylim(0, max(22.5, float(energy.max()) * 1.08))
    ax_energy.set_title("Charge-counter energy", loc="left")
    ax_energy.scatter([construction_end], [summary["construction_energy_kj"]], color="#6A3D9A", s=48, zorder=4)
    ax_energy.annotate(f"{summary['construction_energy_kj']:.1f} kJ", (construction_end, summary["construction_energy_kj"]), xytext=(-8, 10), textcoords="offset points", ha="right", color="#6A3D9A", fontweight="bold", bbox=NUMBER_LABEL_BOX)

    failed = summary["extraction_batches_failed"]
    total = summary["extraction_batches_total"]
    facts = [
        (f"{summary['history_turns']:.0f}", "history turns"),
        (f"{construction_dur:.1f} min", "construction"),
        (f"{summary['construction_energy_kj']:.1f} kJ", "construction energy"),
        (f"{summary['construction_joules_per_turn']:.1f} J", "per history turn"),
        (f"{failed / total * 100:.0f}%", f"failed extraction batches\n({failed:.0f}/{total:.0f})"),
        (f"{summary['judge_correct']:.0f}/{summary['judge_n']:.0f}", "judged correct\n(descriptive pilot)"),
    ]
    if ax_facts is not None:
        ax_facts.axis("off")
        y = 0.96
        for value, label in facts:
            ax_facts.text(0.02, y, value, transform=ax_facts.transAxes, fontsize=18, fontweight="bold", color="#183642", va="top")
            ax_facts.text(0.02, y - 0.06, label, transform=ax_facts.transAxes, fontsize=9, color="#59646B", va="top")
            y -= 0.13
        ax_facts.text(0.02, 0.02,
                      f"First sample: {soc_start:.1f}°C hottest zone, {summary['battery_temperature_start']:.1f}°C battery\n"
                      f"Peak: {soc_peak:.1f}°C at {summary['soc_temperature_peak_elapsed_s']:.0f} s; 3.36 GHz only until {summary['cpu_freq_top_last_s']:.0f} s\n"
                      f"Construction median: {construction_median:.1f}°C (fastest core {clock_ghz:.2f} GHz)\n"
                      f"Battery peak: {bat_peak:.1f}°C",
                      transform=ax_facts.transAxes, fontsize=7.8, color="#485258", va="bottom",
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="#F3F5F6", edgecolor="none"))
    _note(fig, 0.065, 0.015, f"Direct S23 pilot (419 turns). Temperature is the hottest of all thermal zones: {soc_peak:.1f}°C at peak, then a {construction_median:.1f}°C construction median while the fastest core mostly ran at {clock_ghz:.2f} GHz. Governor state was not logged.", fontsize=8.2, color="#555B60")
    _layout(fig, left=0.08, right=0.98, top=0.93, bottom=0.12)
    return _save_all(fig, output_dir, "fig06_phone_graphrag_pilot")


def plot_accuracy_cost_frontiers(records: list[dict[str, object]], output_dir: Path, gemma_only: bool = False) -> list[Path]:
    """fig07; gemma_only draws the two Gemma panels side by side for slides."""
    rows = _rows(records, "fig07_accuracy_cost_frontiers")
    accuracy = {str(row["run"]): row for row in rows if row["metric"] == "judge_accuracy"}
    tokens = {str(row["run"]): row for row in rows if row["metric"] == "prompt_tokens"}
    # Rows are benchmarks, columns reader families: one reader family per panel keeps the paths readable.
    panels = [
        ("LoCoMo n=200", "Gemma", ["Gemma E2B", "Gemma E4B"]),
        ("LoCoMo n=200", "Qwen3.5", ["Qwen 0.8B", "Qwen 2B", "Qwen 4B"]),
        ("LongMemEval-S n=60", "Gemma", ["Gemma E2B", "Gemma E4B"]),
        ("LongMemEval-S n=60", "Qwen3.5", ["Qwen 0.8B", "Qwen 2B", "Qwen 4B"]),
    ]
    method_order = METHOD_ORDER
    if gemma_only:
        panels = [panel for panel in panels if panel[1] == "Gemma"]
        fig, row = plt.subplots(1, 2, figsize=_fs((12.4, 5.0), 3.4), sharex=True, sharey=True)
        axes = row.reshape(1, 2)
    else:
        fig, axes = plt.subplots(2, 2, figsize=_fs((12.4, 9.0), 6.2), sharex=True, sharey=True)
    # Tag offset in points, keyed by (panel, family, method, top reader of the group); default right.
    tag_offsets = {
        ("LoCoMo n=200", "Gemma", "GraphRAG v3", "E4B"): (-7, 0),
        ("LoCoMo n=200", "Gemma", "GraphRAG v3", "E2B"): (-7, 0),
        ("LoCoMo n=200", "Gemma", "GraphRAG v4", "E2B"): (9, -7),
        ("LoCoMo n=200", "Qwen3.5", "GraphRAG v4", "4B"): (-8, 0),
        ("LoCoMo n=200", "Qwen3.5", "GraphRAG v3", "4B"): (-7, 0),
        ("LoCoMo n=200", "Qwen3.5", "GraphRAG v3", "2B"): (7, -11),
    }
    label_positions = {
        ("LoCoMo n=200", "Gemma E4B"): (1250, 45.0),
        ("LoCoMo n=200", "Gemma E2B"): (1250, 36.0),
        ("LoCoMo n=200", "Qwen 4B"): (1250, 44.0),
        ("LoCoMo n=200", "Qwen 2B"): (1250, 31.0),
        ("LoCoMo n=200", "Qwen 0.8B"): (1250, 18.0),
        ("LongMemEval-S n=60", "Gemma E4B"): (3150, 62.0),
        ("LongMemEval-S n=60", "Gemma E2B"): (3150, 43.0),
        ("LongMemEval-S n=60", "Qwen 4B"): (3150, 55.0),
        ("LongMemEval-S n=60", "Qwen 2B"): (3150, 40.0),
        ("LongMemEval-S n=60", "Qwen 0.8B"): (3150, 22.0),
    }
    # Long-window full-context cells live outside the source-data rows; read them from their own
    # record (long_context_fc.py) so every plotted value traces to results/long_context_fc.json.
    long_window: dict[tuple[str, str], tuple[float, float, float, float]] = {}
    long_path = Path(__file__).resolve().parents[1] / "results" / "long_context_fc.json"
    if long_path.exists():
        panel_of = {"LoCoMo": "LoCoMo n=200", "LongMemEval-S": "LongMemEval-S n=60"}
        reader_of = {"Gemma 4 E2B": "Gemma E2B", "Gemma 4 E4B": "Gemma E4B", "Qwen3.5 4B": "Qwen 4B"}
        for cell in json.loads(long_path.read_text(encoding="utf-8")):
            n = int(cell["n"])
            correct = round(float(cell["accuracy"]) * n)
            low, high = wilson_interval(correct, n)
            long_window[(panel_of[cell["benchmark"]], reader_of[cell["reader"]])] = (
                float(cell["prompt_tokens"]["mean"]), 100.0 * correct / n, 100.0 * low, 100.0 * high)

    for ax, (panel, family, models) in zip(axes.flat, panels):
        panel_rows = [row for row in accuracy.values() if row["panel"] == panel]
        panel_tags: list[tuple[str, float, float, str]] = []
        for model in models:
            model_rows = [row for row in panel_rows if row["condition"] == model]
            model_rows.sort(key=lambda row: method_order.index(str(row["series"])))
            xs = [float(tokens[str(row["run"])]["value"]) for row in model_rows]
            ys = [float(row["value"]) * 100.0 for row in model_rows]
            for row, x, y in zip(model_rows, xs, ys):
                method = str(row["series"])
                style = METHOD_STYLE[method]
                low = float(row["low"]) * 100.0
                high = float(row["high"]) * 100.0
                ax.errorbar(
                    x,
                    y,
                    yerr=[[y - low], [high - y]],
                    fmt=style["marker"],
                    color=style["color"],
                    markerfacecolor="white" if method == "GraphRAG v4" else style["color"],
                    markeredgewidth=1.2,
                    markersize=7.5,
                    capsize=2.5,
                    linewidth=1.0,
                    zorder=3,
                )
                if method == "Vector RAG":
                    ax.annotate(
                        model,
                        (x, y),
                        xytext=label_positions[(panel, model)],
                        textcoords="data",
                        fontsize=8.0,
                        color="#263238",
                        bbox=NUMBER_LABEL_BOX,
                        arrowprops=dict(arrowstyle="-", color="#A0AAB0", linewidth=0.7),
                        zorder=5,
                    )
                else:
                    panel_tags.append((method, x, y, model.split()[-1]))
            # Long-window full context (131k window, sec:ch4_long_fc): hollow square, dotted arrow
            # from the truncated full-context point of the same reader.
            long_fc = long_window.get((panel, model))
            truncated = [row for row in model_rows if row["series"] == "Full context"]
            if long_fc and truncated:
                lx, ly, llow, lhigh = long_fc
                # Both Gemma readers share one prompt length: dodge them apart and name them.
                lx *= {"Gemma E2B": 0.88, "Gemma E4B": 1.12}.get(model, 1.0)
                right = model == "Gemma E4B" and panel.startswith("LoCoMo")
                ax.annotate(model.split()[-1], (lx, ly), xytext=(7 if right else -7, 0), textcoords="offset points",
                            ha="left" if right else "right", va="center", fontsize=7.5, color="#263238", zorder=5)
                tx = float(tokens[str(truncated[0]["run"])]["value"])
                ty = float(truncated[0]["value"]) * 100.0
                ax.annotate("", xy=(lx, ly), xytext=(tx, ty),
                            arrowprops=dict(arrowstyle="->", linestyle=":", color=METHOD_STYLE["Full context"]["color"],
                                            linewidth=0.9, shrinkA=4, shrinkB=5), zorder=2)
                ax.errorbar(lx, ly, yerr=[[ly - llow], [lhigh - ly]], fmt="s",
                            color=METHOD_STYLE["Full context"]["color"], markerfacecolor="white",
                            markeredgewidth=1.4, markersize=7.5, capsize=2.5, linewidth=1.0, zorder=3)
        # No connecting lines, so graph and full-context points name their reader. Points of one
        # method that sit together share one tag, readers listed top to bottom.
        for method in method_order:
            points = sorted((t for t in panel_tags if t[0] == method), key=lambda t: -t[2])
            groups: list[list[tuple[str, float, float, str]]] = []
            for point in points:
                if groups and abs(point[2] - groups[-1][-1][2]) < 4.5 and abs(np.log(point[1] / groups[-1][-1][1])) < 0.3:
                    groups[-1].append(point)
                else:
                    groups.append([point])
            for group in groups:
                _, x, y, _ = group[0]
                dx, dy = tag_offsets.get((panel, family, method, group[0][3]), (6, 0))
                ax.annotate(", ".join(t[3] for t in group), (x, y), xytext=(dx, dy), textcoords="offset points",
                            ha="left" if dx > 0 else "right" if dx < 0 else "center", va="center",
                            fontsize=6.8, color="#455A64", zorder=5)
        ax.set_xscale("log")
        ax.set_xlim(240, 160000)
        ax.set_xticks([256, 1024, 4096, 16384, 65536, 131072])
        ax.set_xticklabels(["256", "1k", "4k", "16k", "64k", "128k"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(0, 70)
        ax.set_title(f"{panel} · {family}", loc="left")
        ax.grid(axis="both", color="#DDE1E3", linewidth=0.8)
    for ax in axes[-1]:
        ax.set_xlabel("Mean prompt tokens per answer (log scale)")
    for ax in axes[:, 0]:
        ax.set_ylabel("70B-judge accuracy (%)")
    axes[0, 0].annotate(
        "Pareto-better",
        xy=(300, 61),
        xytext=(850, 51),
        arrowprops=dict(arrowstyle="->", color="#59646B", linewidth=1.1),
        fontsize=8.5,
        color="#59646B",
    )
    method_handles = [
        Line2D(
            [0],
            [0],
            marker=METHOD_STYLE[method]["marker"],
            color=METHOD_STYLE[method]["color"],
            markerfacecolor="white" if method == "GraphRAG v4" else METHOD_STYLE[method]["color"],
            linewidth=0,
            markersize=7,
            label="Full context (4,096 window)" if method == "Full context" else method,
        )
        for method in method_order
    ]
    if long_window:
        method_handles.append(Line2D([0], [0], marker="s", color=METHOD_STYLE["Full context"]["color"],
                                     markerfacecolor="white", markeredgewidth=1.4, linewidth=0, markersize=7,
                                     label="Full context (131k window)"))
    if gemma_only:
        # The slide states the scope in its own text; the legend alone explains the markers.
        _fig_legend(fig, handles=method_handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=8.2)
        _layout(fig, left=0.07, right=0.99, top=0.92, bottom=0.25, wspace=0.08)
        return _save_all(fig, output_dir, "fig07_gemma_long_window")
    _fig_legend(fig, handles=method_handles, loc="lower center", ncol=5 if _STYLE == "presentation" else 3, frameon=False, bbox_to_anchor=(0.5, 0.045), fontsize=8.2)
    _note(fig,
        0.07,
        0.008,
        "Rows are benchmark populations and are not compared with each other. Whiskers are Wilson 95% intervals; Vector RAG points name their reader, per-reader values are tabulated in Chapter 4.\n"
        "Solid squares: full context cut to the 4,096-token window. Hollow squares: long-window full context (131k window; Gemma E2B, E4B, Qwen 4B), dotted arrow from the same reader's cut point.\n"
        "LongMemEval graph cells are the repaired prefixed runs; no Qwen graph run exists there.",
        fontsize=8.3,
        color="#59646B",
    )
    _layout(fig, left=0.08, right=0.99, top=0.95, bottom=0.17, wspace=0.08, hspace=0.18)
    return _save_all(fig, output_dir, "fig07_accuracy_cost_frontiers")


def plot_retrieval_outcomes(records: list[dict[str, object]], output_dir: Path) -> list[Path]:
    rows = _rows(records, "fig08_retrieval_outcomes", "outcome_share")
    order = [
        "Evidence + correct",
        "Evidence + wrong",
        "No evidence + correct",
        "No evidence + wrong",
    ]
    colors = {
        "Evidence + correct": "#0077BB",
        "Evidence + wrong": "#88CCEE",
        "No evidence + correct": "#EE7733",
        "No evidence + wrong": "#D7DADC",
    }
    hatches = {"Evidence + correct": "", "Evidence + wrong": "///", "No evidence + correct": "", "No evidence + wrong": "///"}
    fig, axes = plt.subplots(1, 2, figsize=_fs((12.4, 5.2), 3.2), sharey=True)
    for ax, model in zip(axes, ("Gemma E2B", "Gemma E4B")):
        bottoms = np.zeros(len(TOP_K), dtype=float)
        for outcome in order:
            values = []
            for k in TOP_K:
                run = TOPK_RUNS[model][k]
                row = next(item for item in rows if item["run"] == run and item["series"] == outcome)
                values.append(float(row["value"]) * 100.0)
            bars = ax.bar(
                np.arange(len(TOP_K)),
                values,
                bottom=bottoms,
                width=0.68,
                color=colors[outcome],
                edgecolor="white",
                linewidth=0.7,
                # Color already separates the four outcomes; in print the hatching only adds texture.
                hatch=hatches[outcome] if _STYLE == "presentation" else "",
                label=outcome,
            )
            for bar, value, bottom in zip(bars, values, bottoms):
                if value >= 7.5:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bottom + value / 2,
                        f"{value:.0f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="#102027" if outcome != "Evidence + correct" else "white",
                        fontweight="bold",
                    )
            bottoms += np.array(values)
        ax.set_xticks(np.arange(len(TOP_K)))
        ax.set_xticklabels([f"k={k}" for k in TOP_K])
        ax.set_ylim(0, 100)
        ax.set_title(model, loc="left")
        ax.set_xlabel("Retrieved turns")
        # No gridlines: every segment carries its own value, and lines drawn across the bars hide them.
        ax.grid(False)
    axes[0].set_ylabel("Share of questions (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    _fig_legend(fig, handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.025))
    _note(fig, 
        0.07,
        0.005,
        "LoCoMo n=100 per reader and k. Evidence presence uses dataset labels; correctness uses the 70B judge. Hatched segments are incorrect answers.",
        fontsize=8.5,
        color="#59646B",
    )
    _layout(fig, left=0.08, right=0.99, top=0.91, bottom=0.23, wspace=0.10)
    return _save_all(fig, output_dir, "fig08_retrieval_outcomes")


def plot_thinking_tradeoff(records: list[dict[str, object]], output_dir: Path) -> list[Path]:
    rows = _rows(records, "fig09_thinking_tradeoff")
    accuracy = [row for row in rows if row["metric"] == "judge_accuracy"]
    tokens = [row for row in rows if row["metric"] == "completion_tokens"]
    trunc = {str(row["condition"]): float(row["value"]) for row in rows if row["metric"] == "thinking_truncation_rate"}
    conditions = list(dict.fromkeys(str(row["condition"]) for row in accuracy))
    short = {
        "vr E2B LoCoMo": "VR · E2B · LoCoMo",
        "vr E4B LoCoMo": "VR · E4B · LoCoMo",
        "grv3 E2B LoCoMo": "GraphRAG v3 · E2B · LoCoMo",
        "grv3 E4B LoCoMo": "GraphRAG v3 · E4B · LoCoMo",
        "vr E2B LME": "VR · E2B · LongMemEval",
        "vr E4B LME": "VR · E4B · LongMemEval",
        # The asterisk is data, not decoration: full context is compared against its budget-matched
        # control (768-token answer cap), not the 64-token baseline (results/thinking_truncation.md).
        "fc E2B LoCoMo*": "Full context · E2B · LoCoMo*",
        "fc E4B LoCoMo*": "Full context · E4B · LoCoMo*",
    }
    y = np.arange(len(conditions))
    fig, (ax_acc, ax_cost) = plt.subplots(1, 2, figsize=_fs((13.4, 6.0), 3.8), sharey=True, gridspec_kw={"width_ratios": [1.05, 1.0]})
    for yi, condition in zip(y, conditions):
        base = next(row for row in accuracy if row["condition"] == condition and row["series"] == "Base")
        think = next(row for row in accuracy if row["condition"] == condition and row["series"] == "Thinking")
        base_v, think_v = float(base["value"]) * 100.0, float(think["value"]) * 100.0
        delta = think_v - base_v
        line_color = "#009988" if delta >= 0 else "#CC3311"
        ax_acc.plot([base_v, think_v], [yi, yi], color=line_color, linewidth=2.2, zorder=1)
        for row, value, color, marker in ((base, base_v, "#6F777B", "o"), (think, think_v, "#7B3FA1", "D")):
            low, high = float(row["low"]) * 100.0, float(row["high"]) * 100.0
            ax_acc.errorbar(
                value,
                yi,
                xerr=[[value - low], [high - value]],
                fmt=marker,
                color=color,
                markersize=6.5,
                capsize=2.5,
                linewidth=1.0,
                zorder=3,
            )
        # Past both upper whisker caps, so the label covers neither marker nor interval.
        whisker_end = max(float(r["high"]) for r in (base, think)) * 100.0
        ax_acc.annotate(
            f"{delta:+.1f} pt",
            (whisker_end, yi),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.2,
            color=_ink(line_color),
            bbox=NUMBER_LABEL_BOX,
        )

        base_tok = next(row for row in tokens if row["condition"] == condition and row["series"] == "Base")
        think_tok = next(row for row in tokens if row["condition"] == condition and row["series"] == "Thinking")
        base_t, think_t = float(base_tok["value"]), float(think_tok["value"])
        ax_cost.plot([base_t, think_t], [yi, yi], color="#AAB0B4", linewidth=1.7, zorder=1)
        ax_cost.scatter(base_t, yi, color="#6F777B", marker="o", s=45, zorder=3)
        ax_cost.scatter(think_t, yi, color="#7B3FA1", marker="D", s=46, zorder=3)
        ax_cost.annotate(
            # Round half up (18.5% -> 19%); an f-string rounds half to even and showed 18%.
            f"×{think_t / base_t:.0f}; {int(round(trunc[condition] * 100, 6) + 0.5)}% trunc.",
            (think_t, yi),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.0,
            color="#4D3A57",
            bbox=NUMBER_LABEL_BOX,
        )

    ax_acc.set_yticks(y)
    ax_acc.set_yticklabels([short.get(condition, condition) for condition in conditions])
    ax_acc.invert_yaxis()
    ax_acc.set_xlim(0, max(float(row["high"]) for row in accuracy) * 100.0 + 13)
    ax_acc.set_xlabel("70B-judge accuracy (%)")
    ax_acc.set_title("Observed quality change", loc="left")
    ax_acc.grid(axis="x", color="#DDE1E3", linewidth=0.8)
    ax_cost.set_xscale("log")
    ax_cost.set_xlim(7, 1300)
    ax_cost.set_xticks([10, 30, 100, 300, 1000])
    ax_cost.set_xticklabels(["10", "30", "100", "300", "1k"])
    ax_cost.set_xlabel("Mean completion tokens per answer (log scale)" if _STYLE == "presentation" else "Completion tokens per answer (log)")
    ax_cost.set_title("Decode expansion and truncation", loc="left")
    ax_cost.grid(axis="x", color="#DDE1E3", linewidth=0.8)
    handles = [
        Line2D([0], [0], marker="o", color="#6F777B", linewidth=0, label="Base"),
        Line2D([0], [0], marker="D", color="#7B3FA1", linewidth=0, label="Thinking"),
    ]
    _fig_legend(fig, handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.035))
    _note(fig, 
        0.055,
        0.008,
        "Matched cells only. Accuracy uses all questions; thinking truncation (length stop with no answer) therefore suppresses some observed gains. Whiskers are Wilson 95% intervals.",
        fontsize=8.4,
        color="#59646B",
    )
    _layout(fig, left=0.25, right=0.985, top=0.92, bottom=0.17, wspace=0.08)
    return _save_all(fig, output_dir, "fig09_thinking_tradeoff")


def plot_spe_story(records: list[dict[str, object]], output_dir: Path) -> list[Path]:
    rows = _rows(records, "fig10_spe_story")
    accuracy = {str(row["run"]): row for row in rows if row["metric"] == "judge_accuracy"}
    tokens = {str(row["run"]): row for row in rows if row["metric"] == "prompt_tokens"}
    if _STYLE == "presentation":
        family_style = {
            "GraphRAG v3": ("#59646B", "^", "-"),
            "SPE-1": ("#33BBEE", "o", "--"),
            "SPE-2": ("#009988", "o", "-"),
            "GraphRAG v4": ("#CC3311", "D", "-"),
            "Vector RAG": ("#0077BB", "*", "-"),
        }
        oracle_color = ORACLE_COLOR
    else:
        # The same colors as every other thesis figure, so red never means GraphRAG v4 here and full
        # context elsewhere. SPE-1 and SPE-2 are variants of one method: one hue, told apart by line
        # style. The oracle is a reference read from judged labels, not a method, so it is ink.
        spe_color = "#56B4E9"
        family_style = {
            "GraphRAG v3": (PAPER_PALETTE["GraphRAG v3"], "^", "-"),
            "SPE-1": (spe_color, "o", "--"),
            "SPE-2": (spe_color, "o", "-"),
            "GraphRAG v4": (PAPER_PALETTE["GraphRAG v4"], "D", "-"),
            "Vector RAG": (PAPER_PALETTE["Vector RAG"], "*", "-"),
        }
        oracle_color = INK
    fig, axes = plt.subplots(1, 2, figsize=_fs((12.8, 5.3), 3.2), sharey=True)
    for ax, model in zip(axes, ("Gemma E2B", "Gemma E4B")):
        points = SPE_POINTS[model]
        for family in family_style:
            family_points = [(label, run) for label, point_family, run in points if point_family == family]
            if not family_points:
                continue
            xs = [float(tokens[run]["value"]) for _, run in family_points]
            ys = [float(accuracy[run]["value"]) * 100.0 for _, run in family_points]
            color, marker, linestyle = family_style[family]
            if len(xs) > 1:
                order_idx = np.argsort(xs)
                ax.plot(np.array(xs)[order_idx], np.array(ys)[order_idx], color=color, linestyle=linestyle, linewidth=1.6, zorder=1)
            for (label, run), x, y in zip(family_points, xs, ys):
                row = accuracy[run]
                low, high = float(row["low"]) * 100.0, float(row["high"]) * 100.0
                ax.errorbar(
                    x,
                    y,
                    yerr=[[y - low], [high - y]],
                    fmt=marker,
                    color=color,
                    markerfacecolor="white" if family == "GraphRAG v4" else color,
                    markeredgewidth=1.2,
                    markersize=8 if marker != "*" else 11,
                    capsize=2.2,
                    linewidth=1.0,
                    zorder=3,
                )
                compact = label.replace("SPE-1 ", "S1-").replace("SPE-2 ", "S2-")
                offset = (7, 5)
                ha = "left"
                if "GraphRAG v3" in label:
                    offset = (-10, 8) if model == "Gemma E4B" else (-10, -6)
                    ha = "right"
                elif "SPE-1" in label and "b240" in label:
                    offset = (8, -8)
                    ha = "left"
                elif "SPE-1" in label and "b370" in label:
                    offset = (8, -8)
                    ha = "left"
                elif "SPE-2" in label and "b240" in label:
                    offset = (-10, -8) if model == "Gemma E4B" else (-10, 9)
                    ha = "right"
                elif "SPE-2" in label and "b370" in label:
                    offset = (-10, 8)
                    ha = "right"
                elif "GraphRAG v4" in label:
                    # Below its lower whisker: to the right it sat beside the Vector RAG star.
                    offset = (0, -30)
                    ha = "center"
                elif "Vector RAG" in label:
                    offset = (-10, 11)
                    ha = "right"
                ax.annotate(compact, (x, y), xytext=offset, textcoords="offset points", ha=ha, va="center", fontsize=7.6, color=_ink(color), bbox=NUMBER_LABEL_BOX, zorder=5)
        oracle = {
            str(row["metric"]): float(row["value"])
            for row in rows
            if row["series"] == "Oracle (post hoc)" and row["condition"] == model
        }
        ox, oy = oracle["prompt_tokens"], oracle["judge_accuracy"] * 100.0
        ax.scatter([ox], [oy], marker="h", s=95, facecolor="white", edgecolor=oracle_color, linewidth=1.7, zorder=4)
        ax.annotate(f"Oracle {oy:.1f}%", (ox, oy), xytext=(-10, 10), textcoords="offset points", ha="right", va="center", fontsize=7.6, color=_ink(oracle_color), bbox=NUMBER_LABEL_BOX, zorder=5)
        ax.set_xlim(460 if _STYLE == "presentation" else 380, 895)  # room for left-hand labels at the thesis width
        ax.set_ylim(0, 52)
        ax.set_title(model, loc="left")
        ax.set_xlabel("Mean prompt tokens per answer")
        ax.grid(color="#DDE1E3", linewidth=0.8)
    axes[0].set_ylabel("70B-judge accuracy (%)")
    plotted = {family for points in SPE_POINTS.values() for _, family, _ in points}
    handles = [
        Line2D([0], [0], color=color, marker=marker, linestyle=linestyle, label=family)
        for family, (color, marker, linestyle) in family_style.items()
        if family in plotted
    ]
    handles.append(Line2D([0], [0], marker="h", color=oracle_color, markerfacecolor="white", markeredgewidth=1.5, linewidth=0, markersize=8, label="Oracle v3/v4 (post hoc)"))
    _fig_legend(fig, handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.06))
    _note(fig, 
        0.07,
        0.006,
        "Record pipeline (prefixed retrieval, repaired selector), matched LoCoMo n=200 cells as in the Chapter 7 table. No SPE cell reaches source-restored GraphRAG v4 or raw-turn Vector RAG.\n"
        "The oracle keeps v4's answer only where v4 alone was judged correct: an upper bound read from judged labels, not a policy. Whiskers are Wilson 95% intervals.",
        fontsize=8.2,
        color="#59646B",
    )
    _layout(fig, left=0.08, right=0.99, top=0.92, bottom=0.27, wspace=0.10)
    return _save_all(fig, output_dir, "fig10_spe_story")


def plot_graph_build_scaling(records: list[dict[str, object]], output_dir: Path) -> list[Path]:
    rows = _rows(records, "fig11_graph_build_scaling")
    points = [row for row in rows if row["panel"] == "Box scaling" and row["metric"] == "construction_seconds"]
    slopes = {str(row["series"]): float(row["value"]) for row in rows if row["metric"] == "seconds_per_turn"}
    intercepts = {str(row["series"]): float(row["value"]) for row in rows if row["metric"] == "intercept_seconds"}
    r2 = {str(row["series"]): float(row["value"]) for row in rows if row["metric"] == "fit_r2"}
    phone_seconds = float(next(row for row in rows if row["panel"] == "Phone anchor" and row["metric"] == "construction_seconds")["value"])
    phone_energy = float(next(row for row in rows if row["panel"] == "Phone anchor" and row["metric"] == "construction_energy_kj")["value"])
    colors = {
        "E2B GraphRAG v1": "#88CCEE",
        "E2B GraphRAG v2": "#33BBEE",
        "E2B GraphRAG v3": "#0077BB",
        "E4B GraphRAG v1": "#EEAA77",
        "E4B GraphRAG v3": "#CC3311",
    }
    markers = {"E2B GraphRAG v1": "o", "E2B GraphRAG v2": "s", "E2B GraphRAG v3": "D", "E4B GraphRAG v1": "^", "E4B GraphRAG v3": "v"}
    fig, (ax_scale, ax_anchor) = plt.subplots(1, 2, figsize=_fs((13.0, 5.2), 3.0), gridspec_kw={"width_ratios": [1.3, 0.9]})
    for label in GRAPH_BUILD_RUNS:
        series_rows = [row for row in points if row["series"] == label]
        xs = np.array([float(row["condition"]) for row in series_rows])
        ys = np.array([float(row["value"]) for row in series_rows]) / 60.0
        ax_scale.scatter(xs, ys, color=colors[label], marker=markers[label], s=30, alpha=0.75, zorder=3)
        line_x = np.linspace(xs.min(), xs.max(), 100)
        line_y = (slopes[label] * line_x + intercepts[label]) / 60.0
        ax_scale.plot(line_x, line_y, color=colors[label], linewidth=1.5, label=f"{label}  (R²={r2[label]:.2f})")
    ax_scale.set_xlabel("Conversation length (turns)")
    ax_scale.set_ylabel("Graph build time on box (min)")
    ax_scale.set_title("Ten-conversation scaling curves", loc="left")
    ax_scale.grid(color="#DDE1E3", linewidth=0.8)
    # Thesis: below the panels, because inside the axes it covered the E4B points around 420-560 turns.
    if _STYLE == "presentation":
        ax_scale.legend(frameon=False, fontsize=7.7, loc="upper left")
    else:
        _fig_legend(fig, *ax_scale.get_legend_handles_labels(), ncol=3, frameon=False)

    anchor_turns = 419.0
    labels = list(GRAPH_BUILD_RUNS) + ["Direct S23 pilot"]
    minutes = [(slopes[label] * anchor_turns + intercepts[label]) / 60.0 for label in GRAPH_BUILD_RUNS] + [phone_seconds / 60.0]
    bar_colors = [colors[label] for label in GRAPH_BUILD_RUNS] + ["#7B3FA1"]
    yy = np.arange(len(labels))
    # Dots, not bars: on a log axis a bar's length from the arbitrary left edge is not proportional.
    ax_anchor.scatter(minutes, yy, color=bar_colors, s=46, zorder=3)
    ax_anchor.set_yticks(yy)
    ax_anchor.set_yticklabels(labels)
    ax_anchor.invert_yaxis()
    ax_anchor.set_xscale("log")
    ax_anchor.set_xlim(1.5, 140)
    ax_anchor.set_xticks([2, 5, 10, 30, 60, 120])
    ax_anchor.set_xticklabels(["2", "5", "10", "30", "60", "120"])
    ax_anchor.set_xlabel("Construction time at 419 turns (min, log scale)" if _STYLE == "presentation" else "Minutes at 419 turns (log)")
    ax_anchor.set_title("Box fits versus direct phone pilot", loc="left")
    ax_anchor.grid(axis="x", color="#DDE1E3", linewidth=0.8)
    for yi, value in zip(yy, minutes):
        ax_anchor.annotate(f"{value:.1f}", (value, yi), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.1, bbox=NUMBER_LABEL_BOX)
    _note(
        ax_anchor,
        0.98,
        0.04,
        f"Phone construction energy: {phone_energy:.1f} kJ",
        transform=ax_anchor.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.4,
        color="#5E2D79",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F4EDF8", edgecolor="none"),
    )
    _note(fig, 
        0.06,
        0.012,
        "Box points are direct per-conversation builds with linear fits. The S23 point is one early 419-turn pilot and is not configuration-identical to the later box v3/v4 cells.",
        fontsize=8.4,
        color="#59646B",
    )
    _layout(fig, left=0.07, right=0.99, top=0.91, bottom=0.16, wspace=0.32)
    return _save_all(fig, output_dir, "fig11_graph_build_scaling")


def plot_phone_workload_dashboard(records: list[dict[str, object]], output_dir: Path) -> list[Path]:
    rows = _rows(records, "fig12_phone_workload_dashboard")
    metric_specs = [
        ("phone_accuracy", "Accuracy", "Accuracy (%)", 100.0, False),
        ("joules_per_answer", "Energy", "Joules per answer", 1.0, True),
        ("query_latency_seconds", "Latency", "Median query time (min)", 1.0 / 60.0, True),
        ("peak_soc_temperature", "Thermal", "Peak of the hottest thermal zone (°C)", 1.0, False),
        ("peak_ram_mb", "Memory", "Peak system RAM used (GB)", 1.048576 / 1000.0, False),  # MiB -> GB (1e9 B)
    ]
    # Three panels across leave each about two inches at the thesis width, too little for a log axis's
    # tick labels, so the thesis stacks the five panels two across.
    nrows, ncols = (2, 3) if _STYLE == "presentation" else (3, 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=_fs((14.0, 8.0), 7.6))
    axes_flat = list(axes.flat)
    y = np.arange(len(MODEL_ORDER))
    offsets = {"Full context": -0.12, "Vector RAG": 0.12}
    for ax, (metric, panel, xlabel, scale, log_axis) in zip(axes_flat, metric_specs):
        metric_rows = [row for row in rows if row["metric"] == metric]
        for method in ("Full context", "Vector RAG"):
            style = METHOD_STYLE[method]
            method_rows = [next(row for row in metric_rows if row["condition"] == model and row["series"] == method) for model in MODEL_ORDER]
            values = np.array([float(row["value"]) * scale for row in method_rows])
            low_errors = []
            high_errors = []
            for row, value in zip(method_rows, values):
                if row["low"] != "" and row["high"] != "":
                    low_errors.append(value - float(row["low"]) * scale)
                    high_errors.append(float(row["high"]) * scale - value)
                else:
                    error = float(row["error"] or 0.0) * scale
                    low_errors.append(error)
                    high_errors.append(error)
            for yi, model, value, lo, hi in zip(y, MODEL_ORDER, values, low_errors, high_errors):
                ax.errorbar(
                    value,
                    yi + offsets[method],
                    xerr=[[lo], [hi]],
                    fmt=style["marker"],
                    color=style["color"],
                    markeredgewidth=1.3,
                    markersize=6.8,
                    capsize=2.5,
                    linewidth=1.0,
                    zorder=3,
                )
                sittings = [
                    float(row["value"]) * scale
                    for row in rows
                    if row["metric"] == f"{metric}_sitting" and row["condition"] == model and row["series"] == method
                ]
                if sittings:
                    jitter = np.linspace(-0.05, 0.05, len(sittings)) if len(sittings) > 1 else np.array([0.0])
                    ax.scatter(sittings, yi + offsets[method] + jitter, s=13, color="#5F666B", edgecolors="white", linewidths=0.4, zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels(MODEL_ORDER)
        ax.invert_yaxis()
        ax.set_title(panel, loc="left")
        ax.set_xlabel(xlabel)
        if log_axis:
            ax.set_xscale("log")
        ax.grid(axis="x", color="#DDE1E3", linewidth=0.8)
    axes_flat[1].set_xticks([200, 300, 500, 1000, 2000, 3000])
    axes_flat[1].set_xticklabels(["200", "300", "500", "1k", "2k", "3k"])
    axes_flat[1].xaxis.set_minor_formatter(NullFormatter())
    # 10 and 15 sit too close on a log axis at the thesis panel width to both carry a label.
    latency_ticks = [0.5, 1, 2, 5, 10, 15] if _STYLE == "presentation" else [0.5, 1, 2, 5, 10]
    axes_flat[2].set_xticks(latency_ticks)
    axes_flat[2].set_xticklabels([f"{tick:g}" for tick in latency_ticks])
    axes_flat[2].xaxis.set_minor_formatter(NullFormatter())
    # Reader names on the first column of each row only.
    label_columns = (0, 3) if _STYLE == "presentation" else (0, 2, 4)
    for i, ax in enumerate(axes_flat[:5]):
        ax.tick_params(axis="y", labelleft=i in label_columns)
    note_ax = axes_flat[5]
    note_ax.axis("off")
    if _STYLE == "presentation":
        note_ax.text(0.03, 0.94, "Measurement key", transform=note_ax.transAxes, fontsize=12, fontweight="bold", va="top")
    note_ax.legend(
        handles=[
            Line2D([0], [0], marker="s", color=METHOD_STYLE["Full context"]["color"], linewidth=0, label="Full context"),
            Line2D([0], [0], marker="o", color=METHOD_STYLE["Vector RAG"]["color"], linewidth=0, label="Vector RAG"),
            Line2D([0], [0], marker="o", color="#5F666B", markeredgecolor="white", linewidth=0, markersize=5, label="One sitting"),
        ],
        loc="upper left" if _STYLE == "presentation" else "center",
        bbox_to_anchor=(0.0, 0.84) if _STYLE == "presentation" else None,
        frameon=False,
        fontsize=9.2 if _STYLE == "presentation" else None,
    )
    _note(
        note_ax,
        0.03,
        0.48,
        "Accuracy whiskers: Wilson 95% interval (n=20).\nAnswers are identical across sittings\n(greedy decoding), so sittings measure\nenergy, time, heat and RAM only.\n\nOther whiskers: between-sitting SD.\nFull context has one sitting except E2B.\n\nThermal: hottest of all thermal zones,\nnot battery temperature.",
        transform=note_ax.transAxes,
        fontsize=9,
        color="#59646B",
        va="top",
        linespacing=1.35,
    )
    _note(fig, 
        0.055,
        0.012,
        "Direct S23 LoCoMo n=20 measurements. Grey dots are individual sittings. Resource peaks are summarized within each run span. Phone vector retrieval is unprefixed (Chapter 3).",
        fontsize=8.5,
        color="#59646B",
    )
    _layout(fig, left=0.13, right=0.99, top=0.94, bottom=0.10, hspace=0.42, wspace=0.22)
    return _save_all(fig, output_dir, "fig12_phone_workload_dashboard")


def render_all(records: list[dict[str, object]], output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for renderer in (
        plot_strategy_landscape,
        plot_retrieval_depth,
        plot_graph_repair,
        plot_longmemeval_transfer,
        plot_phone_efficiency,
        plot_phone_graphrag_pilot,
        plot_accuracy_cost_frontiers,
        plot_retrieval_outcomes,
        plot_thinking_tradeoff,
        plot_spe_story,
        plot_graph_build_scaling,
        plot_phone_workload_dashboard,
    ):
        paths.extend(renderer(records, output_dir))
    return paths


def write_source_manifest(records: list[dict[str, object]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "source_data.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    return path


def _script_hash(project_root: Path) -> str:
    return hashlib.sha256((project_root / "figures" / "make_thesis_visuals.py").read_bytes()).hexdigest()


def write_trace(project_root: Path, output_dir: Path) -> Path:
    digest = _script_hash(project_root)
    limitations = {
        "fig01_strategy_landscape": [
            "Wilson intervals describe marginal accuracy; paired method differences are tested separately with exact McNemar tests.",
            "GraphRAG v3 and v4 are distinct representations and must not be pooled as one condition.",
        ],
        "fig02_retrieval_depth": [
            "The n=100 top-k sweep is a different evaluation subset from the n=200 strategy landscape.",
            "Evidence recall is dataset-labelled retrieval coverage, not judged answer correctness.",
        ],
        "fig03_graph_repair": [
            "The matched repair ladder is Gemma E2B only and should not be generalized as a family-wide causal effect.",
            "v1 retrieves by entity matching and embeds nothing, so its cell is unaffected by the task prefixes that the later rungs use.",
            "v4 restores verbatim source turns as well as retaining graph structure; it is not a pure extraction repair.",
        ],
        "fig04_longmemeval_transfer": [
            "LongMemEval-S uses a stratified n=60 subset rather than the full n=500 benchmark.",
            "Category cells contain n=10 and are descriptive; the original n=50 single-category runs are excluded.",
            "Vector and graph cells are the prefixed record runs (repaired graph pipeline); full context uses no embeddings.",
            "The pre-repair LongMemEval graph cells (79-81% of extraction batches failed) are cited in the Chapter 4 text, not plotted.",
        ],
        "fig05_phone_efficiency": [
            "Accuracy has n=20 per cell; Wilson uncertainty is larger than the between-sitting energy variation for low-correct cells.",
            "Energy replication is uneven: vector cells and E2B full context have three sittings; other full-context cells have one.",
            "Variation in the Vector-RAG/full-context ratio across model families is not statistically distinguishable at n=20.",
            "Sittings that started at a full battery held the charge counter for 161-380 s, so their energy is under-counted; the estimated gap is disclosed in the appendix.",
        ],
        "fig06_phone_graphrag_pilot": [
            "This is one early-configuration GraphRAG pilot, not the final v3/v4 GPU configuration.",
            "The four query outputs are descriptive and cannot support a comparative accuracy claim.",
            "Temperature is the hottest of all readable thermal zones, not one SoC sensor, and must not be reported as battery temperature.",
            "The first sample is not an idle baseline: the servers were already loaded when sampling began.",
        ],
        "fig07_accuracy_cost_frontiers": [
            "LoCoMo n=200 and balanced LongMemEval-S n=60 are separate benchmark populations; absolute positions must not be compared across rows.",
            "Mean prompt tokens are a model-input cost proxy, not a direct energy measurement for these box runs.",
            "Workstation LoCoMo full-context prompts are capped at exactly 4,096 tokens because Ollama trims the overflow; the phone ran the same history untrimmed.",
            "LongMemEval graph cells are the repaired prefixed runs, and no Qwen graph cell exists on LongMemEval.",
        ],
        "fig08_retrieval_outcomes": [
            "The decomposition uses the LoCoMo n=100 top-k sweep rather than the n=200 strategy grid.",
            "Evidence presence comes from dataset-labelled evidence and correctness from the 70B judge; the association does not establish causality.",
        ],
        "fig09_thinking_tradeoff": [
            "Thinking runs truncate 12--35% of answers; all-question accuracy includes those failures.",
            "Completion tokens are a decode-cost proxy, and no phone energy was measured for the thinking intervention.",
        ],
        "fig10_spe_story": [
            "SPE is a pre-registered negative result; SPE-1 was only run on the unprefixed development pipeline, so it is absent until a prefixed SPE-1 run exists.",
            "Every cell is the prefixed record pipeline at n=200 (the Chapter 7 table), verified against each run's own embed_prefix config rather than its run id; the unprefixed development cells must not be pooled with it.",
            "The oracle is a post hoc upper bound read from judged labels on the same prefixed v3/v4 runs, not an achievable policy.",
        ],
        "fig11_graph_build_scaling": [
            "The direct phone pilot and box construction runs are not configuration-identical, so their separation is descriptive rather than a hardware-only causal estimate.",
            "There is one direct phone graph build and therefore no phone-side variance estimate.",
        ],
        "fig12_phone_workload_dashboard": [
            "Accuracy has n=20 per cell, and several full-context workloads have only one energy sitting.",
            "Thermal values are the hottest of all readable thermal zones, not battery temperature; RAM is system-wide (MemTotal minus MemAvailable), not the reader process.",
            "Energy, latency, RAM, and thermal summaries cover the complete recorded run span and are not isolated component measurements.",
            "Sittings that started at a full battery held the charge counter for 161-380 s, so their energy is under-counted; the estimated gap is disclosed in the appendix.",
        ],
        "app_ch5_run_length": [
            "The n=100 E2B Vector-RAG run and the n=20 replicates use different question compositions and were not executed as matched sittings.",
            "The observed 1.85x difference is descriptive evidence of run-length sensitivity, not a causal amortization factor.",
        ],
    }
    claims = {
        "fig01_strategy_landscape": "Across five SLM readers, memory representation changes LoCoMo accuracy more than modest reader scaling; Vector RAG is the highest point estimate in every reader cell.",
        "fig02_retrieval_depth": "Increasing Vector-RAG retrieval depth raises evidence coverage and answer accuracy while increasing prompt-token cost.",
        "fig03_graph_repair": "Extraction repair improves GraphRAG modestly, while restoring source detail produces the largest repair-stage gain and approaches Vector RAG at a similar token footprint.",
        "fig04_longmemeval_transfer": "The retrieval advantage extends to a category-balanced LongMemEval-S subset, although performance varies sharply by question type.",
        "fig05_phone_efficiency": "On the S23, Vector RAG has a higher point estimate of correct answers per kilojoule than full context in all five measured readers.",
        "fig06_phone_graphrag_pilot": "GraphRAG executed on the S23, but this early SLM-extracted configuration required roughly 92 minutes and 21 kJ to construct one 419-turn graph and suffered high extraction failure.",
        "fig07_accuracy_cost_frontiers": "Within each benchmark and reader family, retrieval-based memories move the operating point toward higher accuracy at lower prompt cost than full context; the benchmark rows are not directly comparable.",
        "fig08_retrieval_outcomes": "Increasing retrieval depth primarily reduces no-evidence failures, while evidence-present errors remain, separating retrieval limitations from downstream reading limitations.",
        "fig09_thinking_tradeoff": "Thinking spends one to two orders of magnitude more completion tokens; observed gains depend on retrieval quality and are partly suppressed by truncation.",
        "fig10_spe_story": "On the prefixed record pipeline, no SPE-2 cell surpasses source-restored GraphRAG v4 or raw-turn Vector RAG at comparable prompt footprints; a post hoc v3/v4 oracle shows question-level headroom the selector did not find.",
        "fig11_graph_build_scaling": "Box graph-construction time grows approximately linearly with conversation length; the direct phone pilot is far slower and consumes roughly 21 kJ, although configurations differ.",
        "fig12_phone_workload_dashboard": "Across five phone readers, Vector RAG occupies a more favorable accuracy, energy, and latency operating region than full context; RAM and thermal peaks remain device-level measurements.",
        "app_ch5_run_length": "A separate n=100 E2B Vector-RAG run used 297.6 J per answer, compared with 550.7 J per answer across the n=20 replicate mean, but the comparison is observational.",
    }
    sources = {
        "fig01_strategy_landscape": "results/master_table.csv",
        "fig02_retrieval_depth": "results/master_table.csv + results/completeness_table.csv",
        "fig03_graph_repair": "results/master_table.csv",
        "fig04_longmemeval_transfer": "results/master_table.csv + results/lme_category_table.csv",
        "fig05_phone_efficiency": "results/golden_metric.json",
        "fig06_phone_graphrag_pilot": "results_phone/_smoke_recovered/phone_gr.csv + results_phone/_smoke_recovered/phone_smoke_gr/*.jsonl",
        "fig07_accuracy_cost_frontiers": "results/master_table.csv",
        "fig08_retrieval_outcomes": "results/completeness_table.csv",
        "fig09_thinking_tradeoff": "results/thinking_truncation.json",
        "fig10_spe_story": "results/master_table.csv + results/lap_locomo200_e{2,4}b_graphrag_v{3,4}_prefix/judged.jsonl",
        "fig11_graph_build_scaling": "results/graph_cost_curve.json + results_phone/_smoke_recovered/phone_gr.csv + results_phone/_smoke_recovered/phone_smoke_gr/*.jsonl",
        "fig12_phone_workload_dashboard": "results/golden_metric.json + results_phone/phone_locomo20_*/resources.csv + results_phone/phone_locomo20_*/*.jsonl",
        "app_ch5_run_length": "results/golden_metric.json + results_phone/phone_locomo100_e2b_vectorrag_n100/resources.csv + results_phone/phone_locomo100_e2b_vectorrag_n100/*.jsonl",
    }
    lines = ["figure_table_trace:"]
    for figure in claims:
        lines.extend(
            [
                f'  - artifact_id: "{figure}"',
                "    source_data:",
                f'      dataset_id: "{figure}"',
                f'      file: "{sources[figure]}"',
                f'    transformation: {{script: "figures/make_thesis_visuals.py", hash: "{digest}"}}',
                f"    caption_claim: {json.dumps(claims[figure])}",
                "    supported_manuscript_claims:",
                f"      - claim: {json.dumps(claims[figure])}",
                '        locator: "Results chapter; exact section to be assigned during thesis integration"',
                "    limitations:",
            ]
        )
        lines.extend(f"      - {json.dumps(item)}" for item in limitations[figure])
    path = output_dir / "figure_trace.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_readme(output_dir: Path) -> Path:
    text = """# Thesis visual package

This package clusters the experiment record by the claim each experiment can support. It is
intended for both the thesis and slides; slide titles should be added outside the
figure body so the same vector artifact remains publication-ready.

## Main figures and caption drafts

### Figure 1. Memory representation dominates modest reader scaling on LoCoMo

70B-judge accuracy for Full context, triple-only GraphRAG v3, source-restored GraphRAG v4,
and Vector RAG across five SLM readers. Points are observed proportions and whiskers are Wilson
95% confidence intervals (n=200 per condition). Vector RAG has the highest point estimate in every
reader cell. The figure does not claim that every adjacent pair is statistically distinguishable.

### Figure 2. Retrieval depth trades prompt cost for evidence coverage and accuracy

Prefixed Vector-RAG sweep on the same balanced LoCoMo n=100 subset for Gemma E2B and E4B.
Accuracy whiskers are Wilson 95% intervals. Evidence recall is computed from dataset-labelled
evidence and is reader-independent; token values are measured mean prompt tokens per answer.

### Figure 3. Graph repair separates extraction failure from representation loss

Matched Gemma E2B LoCoMo n=200 repair sequence on the record runs. v2 changes retrieval ordering, v3
expands the extraction budget, and v4 restores verbatim source turns. The largest accuracy gain
arrives when discarded source detail is restored, at a prompt footprint close to Vector RAG.

### Figure 4. Retrieval transfers to a category-balanced LongMemEval-S subset

Overall and question-type 70B-judge accuracy on stratified LongMemEval-S n=60 (10 questions per
each of six types) for the two Gemma readers. Vector RAG and GraphRAG v3/v4 use the prefixed
record runs with the repaired graph pipeline. Overall whiskers are Wilson 95% intervals;
category cells are descriptive. The earlier lme_s50 runs are excluded because they contain only
single-session-user questions.

### Figure 5. Direct S23 measurements of correct answers per unit of energy

Correct answers per kilojoule and joules per answer for five phone-native reader cells. Rate
intervals propagate Wilson 95% accuracy intervals through the measured mean energy. Grey dots are
individual sittings and energy whiskers are between-sitting SD; full context has one sitting for
every reader except Gemma E2B. The figure supports the direction of the efficiency result, not a
model-family interpretation of the ratio spread.

### Figure 6. An early phone GraphRAG build was executable but operationally impractical

Direct charge-counter and thermal trace for phone_smoke_gr: one 419-turn conversation, followed by
four descriptive queries. The strict construction boundary is taken from ingest_start_ts to
ingest_end_ts. The red trace is the hottest of all readable thermal zones and the blue trace the
battery; 95.4 C is not battery temperature. The first sample is not an idle baseline.

### Figure 7. Accuracy--cost frontiers recur across two benchmark populations

Accuracy versus mean prompt tokens for the LoCoMo n=200 current-record grid (top row) and the
balanced LongMemEval-S n=60 transfer grid (bottom row), with Gemma readers on the left and Qwen3.5
readers on the right. The rows show a recurring within-benchmark tradeoff, not an absolute
comparison between benchmarks. Prompt tokens are a box-run cost proxy rather than a direct
phone-energy measurement; workstation LoCoMo full-context prompts are capped at 4,096 tokens by
Ollama. LongMemEval graph cells are the repaired prefixed runs; no Qwen graph cell exists there.

### Figure 8. Retrieval depth separates missing evidence from downstream answer failure

Outcome composition for the LoCoMo n=100 top-k sweep, decomposing every question into evidence
present/absent and judged correct/wrong. Higher k chiefly contracts the no-evidence share; the
remaining evidence-present errors show that retrieval coverage alone does not guarantee a correct
answer. Evidence labels and judge decisions are observational rather than a causal error diagnosis.

### Figure 9. Thinking adds decode cost and remains exposed to truncation

Matched thinking-off/on comparisons across seven reader--memory cells. Accuracy is shown beside
mean completion tokens and the observed thinking-truncation rate. Thinking uses one to two orders
of magnitude more completion tokens, while 12--35% truncation means the displayed all-question
accuracy already includes incomplete outputs. No direct phone energy was measured for this study.

### Figure 10. SPE does not reach source-restored graphs or vector retrieval

The Chapter 7 record comparison on LoCoMo n=200 (prefixed retrieval, repaired selector): triple-only
GraphRAG v3, SPE-2 at two budgets, source-restored GraphRAG v4 and Vector RAG. No SPE cell reaches
GraphRAG v4 or Vector RAG at comparable prompt footprints. A post hoc oracle on the same v3/v4 runs
(35.0% and 33.5% at about 560 tokens) marks the question-level headroom; it reads judged labels and
is not a policy. SPE-1 was only run on the unprefixed development pipeline and is not shown.

### Figure 11. Graph construction scales gently on the box but steeply in the phone pilot

Observed construction times across 50 box graph builds with per-variant fitted trends, paired with
the direct 419-turn phone_smoke_gr construction anchor and its charge-counter energy. The phone
point establishes practical feasibility and cost for the tested early configuration; because the
phone and box pipelines are not configuration-identical, it is not a hardware-only speed ratio.

### Figure 12. Phone workloads expose a multidimensional deployment tradeoff

Accuracy, whole-run energy per answer, latency, peak RAM, and peak thermal readings for Full
context and Vector RAG across five S23-native readers. Replicate means and SD are used where
multiple sittings survive, and grey dots show each sitting; answers are identical across sittings,
so sittings only vary the resource axes. The thermal panel is the hottest of all readable thermal
zones, and all resource metrics cover the recorded run span.

## Evidence clusters deliberately kept separate

- **Confirmatory comparison:** LoCoMo n=200 current-record cells.
- **Mechanism:** top-k retrieval sweep and GraphRAG repair ladder.
- **External validity:** balanced LongMemEval-S n=60.
- **Deployment measurement:** phone-native n=20 efficiency cells.
- **Feasibility pilot:** early phone GraphRAG n=4 plus its full resource trace.
- **Cross-cutting synthesis:** benchmark-specific accuracy--cost frontiers and the phone workload dashboard.
- **Extended/appendix evidence:** retrieval outcome decomposition, SPE development history,
  thinking/truncation, graph construction scaling, prompt/prefix, and temperature robustness runs.

## Files

- `source_data.csv`: every plotted value, interval, replicate point, and downsampled timeline point.
- `figure_trace.yaml`: source-to-claim trace with the plotting-script SHA-256 and limitations.
- PDF and SVG: vector thesis/presentation assets.
- PNG: 300-DPI review and slide-preview assets.

## Regeneration

```bash
python -B -m figures.make_thesis_visuals
python -B -m tests.test_thesis_visuals
```
"""
    path = output_dir / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate thesis visuals in paper, presentation, or both formats.")
    parser.add_argument("--style", choices=["paper", "presentation", "both"], default="both")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "results" / "thesis_visuals"
    presentation_dir = project_root / "results" / "presentation_visuals"

    records = collect_source_data(project_root)
    manifest = write_source_manifest(records, output_dir)

    if args.style in ("paper", "both"):
        configure_style("paper")
        paths = render_all(records, output_dir)
        plot_strategy_landscape_bars(records, output_dir)
        readme = write_readme(output_dir)
        trace = write_trace(project_root, output_dir)
        print(f"wrote {len(paths)} paper figure files into {output_dir}")
        print(f"source manifest: {manifest}")
        print(f"captions: {readme}")
        print(f"figure trace: {trace}")

    if args.style in ("presentation", "both"):
        configure_style("presentation")
        presentation_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "presentation").mkdir(parents=True, exist_ok=True)
        render_all(records, presentation_dir)
        render_all(records, output_dir / "presentation")
        plot_strategy_landscape_bars(records, presentation_dir)
        plot_strategy_landscape_bars(records, output_dir / "presentation")
        plot_accuracy_cost_frontiers(records, presentation_dir, gemma_only=True)
        print(f"wrote presentation figure files into {presentation_dir}")


if __name__ == "__main__":
    main()
