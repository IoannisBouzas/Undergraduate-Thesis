"""Regression checks for the claim-aligned thesis visual package."""

from __future__ import annotations

import csv
import tempfile
from collections import defaultdict
from pathlib import Path

from figures import make_thesis_visuals as visuals


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def records() -> list[dict[str, object]]:
    return visuals.collect_source_data(PROJECT_ROOT)


def test_strategy_landscape_has_all_current_record_cells() -> None:
    rows = [
        row
        for row in records()
        if row["figure"] == "fig01_strategy_landscape"
        and row["metric"] == "judge_accuracy"
    ]
    assert len(rows) == 20
    assert {row["condition"] for row in rows} == set(visuals.MODEL_ORDER)
    assert {row["series"] for row in rows} == set(visuals.METHOD_ORDER)
    assert all(int(row["n"]) == 200 for row in rows)
    qwen2_v4 = next(
        row
        for row in rows
        if row["condition"] == "Qwen 2B" and row["series"] == "GraphRAG v4"
    )
    assert float(qwen2_v4["value"]) == 0.09


def test_longmemeval_uses_only_balanced_strat60_cells() -> None:
    rows = [row for row in records() if row["figure"] == "fig04_longmemeval_transfer"]
    assert rows
    assert not any("lme_s50" in str(row["run"]) for row in rows)
    overall = [row for row in rows if row["panel"] == "Overall"]
    categories = [row for row in rows if row["panel"] == "Question types"]
    assert len(overall) == 8 and all(int(row["n"]) == 60 for row in overall)
    assert len(categories) == 48 and all(int(row["n"]) == 10 for row in categories)
    # retrieval cells must be the prefixed record runs that tab:ch4_lme reports
    assert all(str(row["run"]).endswith("_prefix") for row in rows if not str(row["series"]).startswith("Full context"))
    accuracy = {(row["condition"], row["series"]): round(float(row["value"]) * 100, 1) for row in overall}
    assert accuracy[("Gemma E2B", "GraphRAG v4")] == 41.7 and accuracy[("Gemma E4B", "GraphRAG v3")] == 23.3


def test_phone_efficiency_has_five_models_and_accuracy_uncertainty() -> None:
    rows = [
        row
        for row in records()
        if row["figure"] == "fig05_phone_efficiency"
        and row["metric"] == "correct_per_kj"
    ]
    assert len(rows) == 10
    assert {row["condition"] for row in rows} == set(visuals.MODEL_ORDER)
    assert all(float(row["high"]) > float(row["low"]) >= 0 for row in rows)
    assert all(
        next(row for row in rows if row["condition"] == model and row["series"] == "Vector RAG")["value"]
        > next(row for row in rows if row["condition"] == model and row["series"] == "Full context")["value"]
        for model in visuals.MODEL_ORDER
    )


def test_phone_graphrag_pilot_uses_recovered_direct_artifacts() -> None:
    rows = [row for row in records() if row["figure"] == "fig06_phone_graphrag_pilot"]
    summary = {
        row["metric"]: float(row["value"])
        for row in rows
        if row["panel"] == "Pilot summary"
    }
    assert 92.0 < summary["construction_end_min"] < 93.5
    assert 91.5 < summary["construction_duration_min"] < 92.5
    assert 21.0 < summary["construction_energy_kj"] < 22.5
    assert 51.0 < summary["construction_joules_per_turn"] < 53.5
    assert summary["history_turns"] == 419
    assert summary["extraction_batches_failed"] == 58
    assert summary["extraction_batches_total"] == 84
    assert summary["judge_correct"] == 0 and summary["judge_n"] == 4
    assert summary["battery_temperature_peak"] < summary["soc_temperature_peak"]
    assert summary["soc_temperature_start"] < 50.0
    # Values the fig06 caption and Ch 5 quote: the hottest zone's construction median (not the
    # retired 71.5 C "plateau"), the usual fastest-core clock, and when the peak and 3.36 GHz happen.
    assert round(summary["soc_temperature_construction_median"], 1) == 74.6
    assert round(summary["cpu_freq_max_construction_median_mhz"], 1) == 1593.6
    assert 73.0 < summary["soc_temperature_peak_elapsed_s"] < 75.0
    assert 59.0 < summary["cpu_freq_top_last_s"] < 61.0


def test_accuracy_cost_frontiers_keep_benchmark_populations_separate() -> None:
    rows = [row for row in records() if row["figure"] == "fig07_accuracy_cost_frontiers"]
    accuracy = [row for row in rows if row["metric"] == "judge_accuracy"]
    prompt = [row for row in rows if row["metric"] == "prompt_tokens"]
    locomo = [row for row in accuracy if row["dataset"] == "locomo"]
    lme = [row for row in accuracy if row["dataset"] == "longmemeval"]
    assert len(locomo) == 20 and all(int(row["n"]) == 200 for row in locomo)
    # 8 Gemma cells (full context, graph v3, graph v4, vector) + 6 Qwen cells (no Qwen graph run exists)
    assert len(lme) == 14 and all(int(row["n"]) == 60 for row in lme)
    assert len(prompt) == len(accuracy) == 34
    assert not any("lme_s50" in str(row["run"]) for row in rows)
    lme_graph = [row for row in lme if str(row["series"]).startswith("GraphRAG")]
    assert {row["condition"] for row in lme_graph} == {"Gemma E2B", "Gemma E4B"}
    assert all(str(row["run"]).endswith("_prefix") for row in lme_graph)  # repaired cells, not lme_*_gr
    assert {row["series"] for row in lme if str(row["condition"]).startswith("Qwen")} == {"Full context", "Vector RAG"}


def test_retrieval_outcomes_form_complete_compositions() -> None:
    rows = [row for row in records() if row["figure"] == "fig08_retrieval_outcomes"]
    assert len(rows) == 40
    grouped: dict[tuple[object, object], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["run"])].append(row)
    assert len(grouped) == 10
    expected_series = {
        "Evidence + correct",
        "Evidence + wrong",
        "No evidence + correct",
        "No evidence + wrong",
    }
    for group in grouped.values():
        assert {row["series"] for row in group} == expected_series
        assert abs(sum(float(row["value"]) for row in group) - 1.0) < 1e-12
        assert all(int(row["n"]) == 100 for row in group)
    e2b_k2 = {
        row["series"]: float(row["value"])
        for row in rows
        if row["condition"] == "Gemma E2B" and row["run"].endswith("prefix_k2")
    }
    assert e2b_k2 == {
        "Evidence + correct": 0.10,
        "Evidence + wrong": 0.16,
        "No evidence + correct": 0.08,
        "No evidence + wrong": 0.66,
    }


def test_thinking_tradeoff_has_all_matched_cells_and_costs() -> None:
    rows = [row for row in records() if row["figure"] == "fig09_thinking_tradeoff"]
    accuracy = [row for row in rows if row["metric"] == "judge_accuracy"]
    tokens = [row for row in rows if row["metric"] == "completion_tokens"]
    truncation = [row for row in rows if row["metric"] == "thinking_truncation_rate"]
    assert len({row["condition"] for row in rows}) == 8
    assert len(accuracy) == 16 and len(tokens) == 16 and len(truncation) == 8
    for condition in {str(row["condition"]) for row in rows}:
        base = next(row for row in tokens if row["condition"] == condition and row["series"] == "Base")
        think = next(row for row in tokens if row["condition"] == condition and row["series"] == "Thinking")
        assert float(think["value"]) / float(base["value"]) > 10
    assert all(0 < float(row["value"]) < 0.4 for row in truncation)


def test_spe_story_uses_the_prefixed_record_cells() -> None:
    rows = [row for row in records() if row["figure"] == "fig10_spe_story"]
    accuracy = [row for row in rows if row["metric"] == "judge_accuracy"]
    prompt = [row for row in rows if row["metric"] == "prompt_tokens"]
    assert len(accuracy) == len(prompt) == 12  # 5 record cells per reader + one post hoc oracle each
    assert all(int(row["n"]) == 200 for row in rows)
    assert {row["condition"] for row in accuracy} == {"Gemma E2B", "Gemma E4B"}
    assert {row["series"] for row in accuracy} == {"GraphRAG v3", "SPE-2", "GraphRAG v4", "Vector RAG", "Oracle (post hoc)"}
    # One pipeline, the prefixed record that tab:spe_results reports. Verify from each run's OWN
    # config: the repaired SPE cells are named `*_spe2fix_*` with no "_prefix" in the id.
    for run in {str(row["run"]) for row in rows} - {"oracle_bound"}:
        assert visuals._run_embed_prefix(PROJECT_ROOT, run) == "nomic", run
    by_run = {(row["condition"], row["series"], row["metric"]): float(row["value"]) for row in accuracy}
    # Chapter 7 table values: E2B v3 21.5, SPE-2 B240 23.0 / B370 27.0 (two cells), v4 33.0, VR 36.0
    assert by_run[("Gemma E2B", "GraphRAG v4", "judge_accuracy")] == 0.33
    assert by_run[("Gemma E2B", "Vector RAG", "judge_accuracy")] == 0.36
    assert by_run[("Gemma E4B", "Vector RAG", "judge_accuracy")] == 0.405
    # The oracle is the prefixed pair Chapter 7 quotes (results/spe_oracle_bound.json).
    oracle = {
        (row["condition"], row["metric"]): float(row["value"])
        for row in rows
        if row["series"] == "Oracle (post hoc)"
    }
    assert oracle[("Gemma E2B", "judge_accuracy")] == 0.35 and round(oracle[("Gemma E2B", "prompt_tokens")], 1) == 560.3
    assert oracle[("Gemma E4B", "judge_accuracy")] == 0.335 and round(oracle[("Gemma E4B", "prompt_tokens")], 1) == 555.6


def test_spe_story_rejects_a_cell_from_the_other_pipeline() -> None:
    """The guard must fail loudly if an unprefixed development cell is ever added to SPE_POINTS."""
    original = visuals.SPE_POINTS["Gemma E2B"]
    visuals.SPE_POINTS["Gemma E2B"] = original + [
        ("SPE-1 b240", "SPE-1", "lap_locomo200_e2b_spe_b240")
    ]
    try:
        raised = False
        try:
            visuals.spe_story_records(PROJECT_ROOT)
        except ValueError as exc:
            raised = "embed_prefix" in str(exc)
        assert raised, "an unprefixed run was accepted into the prefixed SPE comparison"
    finally:
        visuals.SPE_POINTS["Gemma E2B"] = original


def test_graph_scaling_uses_the_direct_phone_anchor() -> None:
    rows = [row for row in records() if row["figure"] == "fig11_graph_build_scaling"]
    points = [row for row in rows if row["panel"] == "Box scaling" and row["metric"] == "construction_seconds"]
    slopes = [row for row in rows if row["metric"] == "seconds_per_turn"]
    phone = next(
        row
        for row in rows
        if row["panel"] == "Phone anchor" and row["metric"] == "construction_seconds"
    )
    assert len(points) == 50 and len(slopes) == 5
    assert phone["run"] == "phone_smoke_gr"
    assert phone["condition"] == "419 turns"
    assert 5_400 < float(phone["value"]) < 5_600
    assert "_smoke_recovered" in str(phone["source"])
    # The bar must be the ingest-start-to-ingest-end span that fig06 and Chapter 5 quote, not the
    # run-launch-to-ingest-end span, which carries ~45 s of setup the box builds beside it exclude.
    pilot_duration_min = next(
        float(row["value"])
        for row in records()
        if row["figure"] == "fig06_phone_graphrag_pilot"
        and row["metric"] == "construction_duration_min"
    )
    assert abs(float(phone["value"]) / 60.0 - pilot_duration_min) < 0.01


def test_phone_workload_dashboard_covers_readers_and_resource_axes() -> None:
    rows = [row for row in records() if row["figure"] == "fig12_phone_workload_dashboard"]
    metrics = {
        "phone_accuracy",
        "joules_per_answer",
        "query_latency_seconds",
        "peak_soc_temperature",
        "peak_ram_mb",
    }
    sitting_metrics = {f"{metric}_sitting" for metric in metrics - {"phone_accuracy"}}
    assert {row["metric"] for row in rows} == metrics | sitting_metrics
    # 22 sittings: three for every vector cell and for E2B full context, one for the other four
    # full-context cells. Accuracy has no sitting rows: answers are identical across sittings.
    assert len(rows) == 50 + 4 * 22
    for metric in metrics:
        metric_rows = [row for row in rows if row["metric"] == metric]
        assert len(metric_rows) == 10
        assert {row["condition"] for row in metric_rows} == set(visuals.MODEL_ORDER)
        assert {row["series"] for row in metric_rows} == {"Full context", "Vector RAG"}
    for metric in sitting_metrics:
        assert len([row for row in rows if row["metric"] == metric]) == 22
    single = {(row["condition"], row["series"]) for row in rows if row["metric"] == "joules_per_answer" and int(row["replicates"]) == 1}
    assert single == {("Gemma E4B", "Full context"), ("Qwen 0.8B", "Full context"), ("Qwen 2B", "Full context"), ("Qwen 4B", "Full context")}
    e2b_vr_energy = next(
        row
        for row in rows
        if row["metric"] == "joules_per_answer"
        and row["condition"] == "Gemma E2B"
        and row["series"] == "Vector RAG"
    )
    assert 545 < float(e2b_vr_energy["value"]) < 560
    assert int(e2b_vr_energy["replicates"]) == 3


def test_manifest_trace_and_all_formats_render() -> None:
    rows = records()
    visuals.configure_style()
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        manifest = visuals.write_source_manifest(rows, output_dir)
        paths = visuals.render_all(rows, output_dir)
        readme = visuals.write_readme(output_dir)
        trace = visuals.write_trace(PROJECT_ROOT, output_dir)
        with manifest.open(encoding="utf-8", newline="") as handle:
            saved = list(csv.DictReader(handle))
        assert len(saved) == len(rows)
        assert len(paths) == 36
        assert {path.suffix for path in paths} == {".pdf", ".svg", ".png"}
        assert {path.stem for path in paths} == {
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
        }
        assert all(path.stat().st_size > 5_000 for path in paths)
        assert "lme_s50" in readme.read_text(encoding="utf-8")
        trace_text = trace.read_text(encoding="utf-8")
        assert trace_text.count("artifact_id:") == 13
        assert trace_text.count("limitations:") == 13


if __name__ == "__main__":
    tests = [
        test_strategy_landscape_has_all_current_record_cells,
        test_longmemeval_uses_only_balanced_strat60_cells,
        test_phone_efficiency_has_five_models_and_accuracy_uncertainty,
        test_phone_graphrag_pilot_uses_recovered_direct_artifacts,
        test_accuracy_cost_frontiers_keep_benchmark_populations_separate,
        test_retrieval_outcomes_form_complete_compositions,
        test_thinking_tradeoff_has_all_matched_cells_and_costs,
        test_spe_story_uses_the_prefixed_record_cells,
        test_spe_story_rejects_a_cell_from_the_other_pipeline,
        test_graph_scaling_uses_the_direct_phone_anchor,
        test_phone_workload_dashboard_covers_readers_and_resource_axes,
        test_manifest_trace_and_all_formats_render,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
