#!/usr/bin/env python3
r"""Generates the extended thesis figures (fig15-fig21) directly into results/thesis_visuals/:

1. fig19_model_family_scaling: Gemma 4 vs Qwen 3.5 — parameter scaling vs memory representation.
2. fig16_kv_ram_wall: Calculated Gemma KV cache against measured free RAM on the Samsung Galaxy S23.
3. fig17_amortization_frontier: Multi-turn dynamic break-even analysis (conversation length vs queries).
4. fig18_prompt_robustness: Prompt formatting invariance across scales.
5. fig15_parity_gate: Server box vs phone parity diagnostic.
6. fig20_phone_prefill_decode_memory: Measured E2B query time against prompt length on the S23.
7. fig21_extracted_subgraph_case_study: S23 graph pilot, why ego retrieval missed "What did Caroline research?".

Supports dual styles:
  - Paper: Clean publication styling, minimal ink, no in-canvas suptitle banners.
  - Presentation: Bold takeaway suptitles, slide-ready callout boxes, status badges.

Usage:
    python -B -m figures.make_extended_figures [--style {paper,presentation,both}]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np

import run_memory_eval
sys.modules["__main__"].GraphRAGMemory = run_memory_eval.GraphRAGMemory

from figures.visual_style import (
    paper_size,
    configure_paper_style,
    configure_presentation_style,
    PAPER_PALETTE,
    PRESENTATION_PALETTE,
    NUMBER_LABEL_BOX_PAPER,
    NUMBER_LABEL_BOX_PRESENTATION,
    save_figure,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THESIS_DIR = PROJECT_ROOT / "results" / "thesis_visuals"
PRESENTATION_DIR = PROJECT_ROOT / "results" / "presentation_visuals"

COLORS = PRESENTATION_PALETTE

# llama-server -c on the phone for every num_ctx=4096 run (PHONE_PLAYBOOK.md): llama.cpp rejects a
# prompt longer than its context, so the server got twice the harness budget.
PHONE_SERVER_CTX = 8192

# Measured E2B phone runs for fig20. Excluded: *_old_0707 (copies of the primary sittings),
# *_udq4kxl (different quantization) and *_ctx8192 (warm prompt-cache reuse after the first question).
PHONE_E2B_LATENCY_RUNS = {
    "Full context": ["phone_locomo20_e2b_fullcontext", "phone_locomo20_e2b_fullcontext_rep2", "phone_locomo20_e2b_fullcontext_rep3"],
    "Vector RAG, k=10": ["phone_locomo20_e2b_vectorrag", "phone_locomo20_e2b_vectorrag_rep2", "phone_locomo20_e2b_vectorrag_rep3", "phone_locomo100_e2b_vectorrag_n100"],
    "Vector RAG, k=2": ["phone_locomo20_e2b_vectorrag_k2"],
    "Graph pilot (early config)": ["_smoke_recovered/phone_smoke_gr"],
}


def e2b_kv_gb(ctx):
    """f16 KV cache of Gemma 4 E2B, calculated from the phone's GGUF header shapes
    (kv_shared_prefill_bound.py): 3 global layers grow with context, 12 sliding-window layers are
    capped at 512 tokens, 20 layers reuse earlier KV. A calculation, not a runtime measurement."""
    glob = 3 * 1 * (512 + 512) * 2 * ctx
    loc = 12 * 1 * (256 + 256) * 2 * np.minimum(ctx, 512)
    return (glob + loc) / 1e9


def e4b_kv_gb(ctx):
    """Same calculation for Gemma 4 E4B: 4 global and 20 sliding-window layers, 2 KV heads, 18 shared."""
    glob = 4 * 2 * (512 + 512) * 2 * ctx
    loc = 20 * 2 * (256 + 256) * 2 * np.minimum(ctx, 512)
    return (glob + loc) / 1e9


def _free_ram_gb(model: str) -> tuple[float, float, float]:
    """Measured MemTotal, the highest system-RAM peak across the model's phone sittings, and the
    RAM still free at that peak, in GB (1e9 B). resources.csv logs MiB of MemTotal - MemAvailable."""
    from figures.make_thesis_visuals import PHONE_RUNS, _phone_run_resource_metrics

    runs = PHONE_RUNS[(model, "fullcontext")] + PHONE_RUNS[(model, "vectorrag")]
    peak_mib = max(_phone_run_resource_metrics(PROJECT_ROOT / "results_phone" / run)[2] for run in runs)
    with (PROJECT_ROOT / "results_phone" / runs[0] / "resources.csv").open(encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    total_mib = float(row["ram_used_mb"]) + float(row["ram_free_mb"])
    to_gb = 1024 * 1024 / 1e9
    return total_mib * to_gb, peak_mib * to_gb, (total_mib - peak_mib) * to_gb


def _phone_query_times(runs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Prompt tokens and query seconds (retrieval + generation) for every answered phone question."""
    tokens, seconds = [], []
    for run in runs:
        path = next((PROJECT_ROOT / "results_phone" / run).glob("locomo_*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("generate_seconds") and row.get("prompt_tokens"):
                tokens.append(float(row["prompt_tokens"]))
                seconds.append(float(row["generate_seconds"]) + float(row.get("retrieval_seconds") or 0.0))
    return np.array(tokens), np.array(seconds)


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Calculates 95% two-sided Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    spread = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


# =============================================================================
# Visual 1: Reasoning vs Non-Reasoning ("The Thinking Illusion")
# =============================================================================
def plot_reasoning_tradeoff(output_dirs: list[Path], style: str = "paper") -> list[Path]:
    if style == "paper":
        configure_paper_style()
    else:
        configure_presentation_style()

    trunc_path = PROJECT_ROOT / "results" / "thinking_truncation.json"
    with trunc_path.open(encoding="utf-8") as f:
        data = json.load(f)

    conditions = [
        ("vr E2B LoCoMo", "VR · E2B\n(LoCoMo)"),
        ("vr E4B LoCoMo", "VR · E4B\n(LoCoMo)"),
        ("grv3 E2B LoCoMo", "GraphRAG v3 · E2B\n(LoCoMo)"),
        ("grv3 E4B LoCoMo", "GraphRAG v3 · E4B\n(LoCoMo)"),
        ("vr E2B LME", "VR · E2B\n(LongMemEval)"),
        ("vr E4B LME", "VR · E4B\n(LongMemEval)"),
    ]

    y = np.arange(len(conditions))
    fig = plt.figure(figsize=(11.5, 4.8))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 0.8], wspace=0.28)

    ax_acc = fig.add_subplot(grid[0])
    ax_cost = fig.add_subplot(grid[1], sharey=ax_acc)
    ax_trunc = fig.add_subplot(grid[2], sharey=ax_acc)

    for i, (key, label) in enumerate(conditions):
        cell = data[key]
        n = cell["n"]
        base_acc = cell["base_all"]
        think_acc = cell["think_all"]
        base_low, base_high = wilson_interval(round(base_acc / 100.0 * n), n)
        think_low, think_high = wilson_interval(round(think_acc / 100.0 * n), n)
        delta = think_acc - base_acc

        line_col = "#009988" if delta >= 0 else "#CC3311"
        ax_acc.plot([base_acc, think_acc], [i, i], color=line_col, lw=2.0, zorder=1)
        ax_acc.errorbar(
            base_acc, i,
            xerr=[[base_acc - base_low * 100], [base_high * 100 - base_acc]],
            fmt="o", color=COLORS["Base"], markersize=6.5, capsize=3, linewidth=1.1, zorder=3,
        )
        ax_acc.errorbar(
            think_acc, i,
            xerr=[[think_acc - think_low * 100], [think_high * 100 - think_acc]],
            fmt="D", color=COLORS["Thinking"], markersize=6.5, capsize=3, linewidth=1.1, zorder=3,
        )

        base_tok = cell["mean_comp_tok_base"]
        think_tok = cell["mean_comp_tok_think"]
        ax_cost.plot([base_tok, think_tok], [i, i], color="#AAB0B4", lw=1.8, zorder=1)
        ax_cost.scatter(base_tok, i, color=COLORS["Base"], marker="o", s=45, zorder=3)
        ax_cost.scatter(think_tok, i, color=COLORS["Thinking"], marker="D", s=45, zorder=3)

        trunc_rate = cell["truncation_rate_pct"]
        ax_trunc.barh(i, trunc_rate, height=0.52, color="#D55E00", alpha=0.75, edgecolor="#A04000", zorder=3)

    ax_acc.set_yticks(y)
    ax_acc.set_yticklabels([label for _, label in conditions], fontsize=8.5)
    ax_acc.invert_yaxis()
    ax_acc.set_xlim(0, 75)
    ax_acc.set_xlabel("70B-Judge accuracy (%)")
    ax_acc.set_title("(a) Accuracy impact", loc="left", fontsize=9.5)
    ax_acc.grid(axis="x", color="#E2E5E8", lw=0.8)

    ax_cost.set_xscale("log")
    ax_cost.set_xlim(6, 1500)
    ax_cost.set_xticks([10, 30, 100, 300, 1000])
    ax_cost.set_xticklabels(["10", "30", "100", "300", "1k"])
    ax_cost.set_xlabel("Mean completion tokens (log)")
    ax_cost.set_title("(b) Decode token cost", loc="left", fontsize=9.5)
    ax_cost.grid(axis="x", color="#E2E5E8", lw=0.8)
    plt.setp(ax_cost.get_yticklabels(), visible=False)

    ax_trunc.set_xlim(0, 45)
    ax_trunc.set_xlabel("Generation truncations (%)")
    ax_trunc.set_title("(c) Truncation rate", loc="left", fontsize=9.5)
    ax_trunc.grid(axis="x", color="#E2E5E8", lw=0.8)
    plt.setp(ax_trunc.get_yticklabels(), visible=False)

    legend_elements = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["Base"], markersize=7, label="Base (concise)"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=COLORS["Thinking"], markersize=7, label="Thinking (reasoning)"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False, fontsize=8.8)

    if style == "presentation":
        fig.suptitle("Test-time reasoning explodes decode tokens by ~50× but truncation erodes gains", x=0.08, y=1.12, ha="left", fontsize=13.0, fontweight="bold")
    fig.tight_layout()

    return save_figure(fig, "fig5_reasoning_tradeoff", output_dirs)


# =============================================================================
# Visual 2: Model Families Scaling (Gemma 4 vs Qwen 3.5) — Figure 19
# =============================================================================
def plot_model_family_scaling(output_dirs: list[Path], style: str = "paper") -> list[Path]:
    if style == "paper":
        configure_paper_style()
    else:
        configure_presentation_style()

    # Read from the judged record runs, as fig17 does, so a rejudge reaches this figure too.
    with open(PROJECT_ROOT / "results" / "master_table.csv", encoding="utf-8") as fh:
        judged = {row["dir"]: float(row["judge"]) for row in csv.DictReader(fh)}
    run_suffix = {
        "Full context": "fullcontext",
        "GraphRAG v3": "graphrag_v3_prefix",
        "GraphRAG v4": "graphrag_v4_prefix",
        "Vector RAG": "vr_prefix",
    }
    families = {
        "Qwen 3.5": ([0.8, 2.0, 4.0], ["0.8B", "2B", "4B"], ["q35_08b", "q35_2b", "q35_4b"]),
        "Gemma 4": ([2.6, 4.3], ["E2B", "E4B"], ["e2b", "e4b"]),
    }
    models_data = {
        family: {
            "sizes": sizes,
            "labels": labels,
            "methods": {method: [judged[f"lap_locomo200_{run}_{suffix}"] for run in runs] for method, suffix in run_suffix.items()},
        }
        for family, (sizes, labels, runs) in families.items()
    }

    fig, axes = plt.subplots(1, 2, figsize=((7.6, 3.4) if style == "presentation" else paper_size(3.0)), sharey=True)
    palette = PAPER_PALETTE if style == "paper" else PRESENTATION_PALETTE
    # Hollow diamond for v4, as in fig03. The x offset (parameters, in B) keeps whiskers that share a
    # value apart; the low graph and full-context cells overlap at Qwen 0.8B and 2B.
    method_styles = {
        "Full context": {"color": palette["Full context"], "marker": "s", "linestyle": "--", "dx": -0.09, "label": "Full context"},
        "GraphRAG v3": {"color": palette["GraphRAG v3"], "marker": "^", "linestyle": "-.", "dx": -0.03, "label": "GraphRAG v3 (triples)"},
        "GraphRAG v4": {"color": palette["GraphRAG v4"], "marker": "D", "linestyle": ":", "dx": 0.03, "label": "GraphRAG v4 (+ source turns)"},
        "Vector RAG": {"color": palette["Vector RAG"], "marker": "o", "linestyle": "-", "dx": 0.09, "label": "Vector RAG"},
    }

    subpanel_labels = ["(a) Qwen 3.5 family", "(b) Gemma 4 family"]

    for idx, (ax, (family_name, data)) in enumerate(zip(axes, models_data.items())):
        sizes = data["sizes"]
        labels = data["labels"]
        for method, values in data["methods"].items():
            st = method_styles[method]
            val_pct = [v * 100 for v in values]
            lows, highs = [], []
            for v in values:
                l, h = wilson_interval(round(v * 200), 200)
                lows.append(l * 100)
                highs.append(h * 100)

            ax.errorbar(
                np.array(sizes) + st["dx"], val_pct,
                yerr=[np.array(val_pct) - np.array(lows), np.array(highs) - np.array(val_pct)],
                color=st["color"],
                marker=st["marker"],
                markerfacecolor="white" if method == "GraphRAG v4" else st["color"],
                markeredgewidth=1.3,
                linestyle=st["linestyle"],
                linewidth=1.5,
                markersize=5.5,
                capsize=2.5,
                elinewidth=1.0,
                label=st["label"] if idx == 0 else None,
                zorder=3,
            )

        ax.set_title(subpanel_labels[idx], loc="left", fontsize=9.5, fontweight="semibold" if style == "presentation" else "normal", pad=8)
        ax.set_xticks(sizes)
        ax.set_xticklabels(labels, fontsize=8.8)
        ax.set_xlabel("Model parameter scale", fontsize=9.0)
        ax.set_xlim(min(sizes) - 0.45, max(sizes) + 0.45)
        ax.set_ylim(0, 52)
        ax.yaxis.grid(True, linestyle="--" if style == "presentation" else "-", alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("70B-judge accuracy (%)", fontsize=9.0)
    # Two legend rows: reserve the top band for them so they cannot sit on the panel titles.
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=8.5)

    if style == "presentation":
        fig.suptitle("Across both SLM families, memory representation outweighs a 5× parameter scaling", x=0.08, y=1.06, ha="left", fontsize=13.0, fontweight="bold")

    fig.tight_layout(rect=(0, 0, 1, 0.86))
    saved = save_figure(fig, "fig19_model_family_scaling", output_dirs)
    return saved


# =============================================================================
# Visual 3: On-Device RAM Wall & Prefill Ceiling (Samsung S23) — Figure 16
# =============================================================================
def plot_kv_ram_wall(output_dirs: list[Path], style: str = "paper") -> list[Path]:
    if style == "paper":
        configure_paper_style()
    else:
        configure_presentation_style()

    ctx = np.logspace(np.log10(512), np.log10(131072), 300)
    fig, ax = plt.subplots(figsize=((7.6, 4.6) if style == "presentation" else paper_size(3.8)))
    # The two free-RAM lines sit close together, so one label goes below its line and one above.
    curves = (("Gemma E2B", e2b_kv_gb, "#0072B2", "-", 0.86, "top"), ("Gemma E4B", e4b_kv_gb, "#E69F00", "--", 1.10, "bottom"))
    for model, kv_gb, color, line, label_shift, label_va in curves:
        ax.plot(ctx, kv_gb(ctx), line, color=color, lw=2.0, label="%s KV cache (calculated from its GGUF header)" % model.replace("Gemma", "Gemma 4"))
        # The server reserves KV for its whole -c window at load, so this is what every main-campaign
        # phone run held (one four-question pilot used -c 16384).
        ax.scatter([PHONE_SERVER_CTX], [kv_gb(PHONE_SERVER_CTX)], color=color, s=38, zorder=5)
        ax.annotate("%.0f MB" % (kv_gb(PHONE_SERVER_CTX) * 1000), (PHONE_SERVER_CTX, kv_gb(PHONE_SERVER_CTX)),
                    xytext=(-8, 7), textcoords="offset points", ha="right", fontsize=7.6, color=color)
        total_gb, peak_gb, free_gb = _free_ram_gb(model)
        ax.axhline(free_gb, color=color, ls=":", lw=1.2)
        ax.text(560, free_gb * label_shift, "RAM still free at %s's highest measured peak: %.1f GB (%.1f of %.1f GB used)" % (model, free_gb, peak_gb, total_gb),
                ha="left", va=label_va, fontsize=7.4, color=color)

    ax.axvline(PHONE_SERVER_CTX, color="#BFC5CA", ls=":", lw=0.9)
    ax.text(PHONE_SERVER_CTX * 1.06, 0.0055, "phone server context\n(-c 8192, main campaign)", ha="left", fontsize=7.4, color="#555")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(512, 140000)
    ax.set_ylim(0.004, 20)
    ax.set_xticks([512, 2048, 8192, 32768, 131072])
    ax.set_xticklabels(["512", "2k", "8k", "32k", "128k"], fontsize=8.5)
    ax.set_xlabel("Server context window (tokens, log scale)", fontsize=9.0)
    ax.set_ylabel("KV-cache size in GB (f16, log scale)", fontsize=9.0)

    if style == "presentation":
        ax.set_title("Calculated KV cache vs. measured free RAM on the S23", loc="left", pad=10)

    ax.grid(True, which="both", ls=":", color="#E2E8F0", lw=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=False, fontsize=7.8)
    fig.tight_layout()

    saved = save_figure(fig, "fig16_kv_ram_wall", output_dirs)
    return saved


# =============================================================================
# Visual 4: Multi-Turn Amortization Frontier — Figure 17
# =============================================================================
HISTORY_TURNS = 419  # conv-26, the phone graph-build pilot's history
AMORTIZATION_RUNS = {
    "Full context": "lap_locomo200_e2b_fullcontext",
    "Vector RAG": "lap_locomo200_e2b_vr_prefix",
    "GraphRAG v3": "lap_locomo200_e2b_graphrag_v3_prefix",
    "GraphRAG v4": "lap_locomo200_e2b_graphrag_v4_prefix",
}


def _amortization_inputs() -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Construction kJ, query kJ and accuracy per method for the E2B projection in fig17.

    Every energy input comes from results/golden_metric.json (one battery method). Vector query
    energy is its query-only stretches, so its build is not counted twice; full context has no
    build, so its whole-run mean is used. No phone graph QA run exists, so graph query energy is
    vector RAG's scaled by workstation prompt tokens. Accuracy is the workstation n=200 record.
    """
    golden = json.loads((PROJECT_ROOT / "results" / "golden_metric.json").read_text(encoding="utf-8"))
    replicates, construction = golden["replicates"], golden["construction"]
    with open(PROJECT_ROOT / "results" / "master_table.csv", encoding="utf-8") as fh:
        master = {row["dir"]: row for row in csv.DictReader(fh)}
    graph_build_kj = construction["graph"]["energy_kj"]
    tokens = {m: float(master[run]["prompt_tok"]) for m, run in AMORTIZATION_RUNS.items()}
    acc = {m: float(master[run]["judge"]) for m, run in AMORTIZATION_RUNS.items()}
    vr_kj = replicates["E2B_vectorrag"]["query_j_per_answer"] / 1000.0
    e_query = {
        "Full context": replicates["E2B_fullcontext"]["mean"] / 1000.0,
        "Vector RAG": vr_kj,
        "GraphRAG v3": vr_kj * tokens["GraphRAG v3"] / tokens["Vector RAG"],
        "GraphRAG v4": vr_kj * tokens["GraphRAG v4"] / tokens["Vector RAG"],
    }
    e_ingest = {
        "Full context": 0.0,
        "Vector RAG": construction["vector"]["j_per_turn"] * HISTORY_TURNS / 1000.0,
        "GraphRAG v3": graph_build_kj,
        "GraphRAG v4": graph_build_kj,  # v4 reuses the v3 graph
    }
    return e_ingest, e_query, acc


def plot_amortization_frontier(output_dirs: list[Path], style: str = "paper") -> list[Path]:
    if style == "paper":
        configure_paper_style()
    else:
        configure_presentation_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=((7.6, 3.8) if style == "presentation" else paper_size(3.2)))
    Q = np.arange(1, 101)
    e_ingest, e_query, acc = _amortization_inputs()
    linestyles = {
        "Full context": "-",
        "Vector RAG": "-",
        "GraphRAG v3": "--",
        "GraphRAG v4": "-.",
    }
    palette = PAPER_PALETTE if style == "paper" else PRESENTATION_PALETTE

    # The legend says which query energies are phone measurements and which are scaled estimates.
    provenance = {
        "Full context": "Full context (query energy measured)",
        "Vector RAG": "Vector RAG (query energy measured)",
        "GraphRAG v3": "GraphRAG v3 (query energy estimated)",
        "GraphRAG v4": "GraphRAG v4 (query energy estimated)",
    }
    for method in ["Full context", "Vector RAG", "GraphRAG v3", "GraphRAG v4"]:
        total_e = e_ingest[method] + Q * e_query[method]
        ax1.plot(Q, total_e, label=provenance[method], color=palette[method], linestyle=linestyles[method], linewidth=1.8)

    # Where each graph's cumulative energy drops below full context's (the caption quotes both).
    crossovers = []
    for method in ("GraphRAG v3", "GraphRAG v4"):
        q = e_ingest[method] / (e_query["Full context"] - e_query[method])
        e = e_ingest[method] + q * e_query[method]
        ax1.plot(q, e, marker="o", color="#334155", markersize=4, zorder=4)
        crossovers.append((q, e))
    (q3, e3), (q4, _) = crossovers
    # Upper left, above every line: the full-context line reaches this height only past Q = 70.
    ax1.annotate(
        "Graph overtakes full context:\n" r"v3 at $Q \approx %.0f$, v4 at $Q \approx %.0f$" % (q3, q4),
        xy=(q3, e3),
        xytext=(3, 145),
        ha="left",
        va="top",
        arrowprops=dict(arrowstyle="->", color="#475569", lw=0.8),
        fontsize=7.5,
        color="#334155",
    )

    ax1.set_title("(a) Projected cumulative energy", loc="left", fontsize=9.5, fontweight="semibold" if style == "presentation" else "normal")
    ax1.set_xlabel("Query count ($Q$)", fontsize=9.0)
    ax1.set_ylabel("Total energy (kJ)", fontsize=9.0)
    ax1.set_xlim(1, 100)
    ax1.set_ylim(0, 150)
    ax1.yaxis.grid(True, linestyle="--" if style == "presentation" else "-", alpha=0.7)
    ax1.spines[["top", "right"]].set_visible(False)
    # One legend for both panels, above them: inside panel (a) the full-context line crosses it.
    fig.legend(*ax1.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=8.0)

    for method in ["Full context", "Vector RAG", "GraphRAG v3", "GraphRAG v4"]:
        per_correct = (e_ingest[method] + Q * e_query[method]) / (Q * acc[method])
        ax2.plot(Q, per_correct, label=method, color=palette[method], linestyle=linestyles[method], linewidth=1.8)
    ax2.text(98, 90, "Accuracy: workstation n=200,\nassumed to hold on the phone", ha="right", va="top", fontsize=7.5, color="#334155")

    ax2.set_title("(b) Projected kJ per correct answer", loc="left", fontsize=9.5, fontweight="semibold" if style == "presentation" else "normal")
    ax2.set_xlabel("Query count ($Q$)", fontsize=9.0)
    ax2.set_ylabel("kJ / correct answer (log scale)", fontsize=9.0)
    ax2.set_yscale("log")
    ax2.set_xlim(1, 100)
    ax2.set_ylim(0.3, 100)
    ax2.yaxis.grid(True, which="both", linestyle="--" if style == "presentation" else "-", alpha=0.5)
    ax2.spines[["top", "right"]].set_visible(False)

    if style == "presentation":
        fig.suptitle("Amortization Analysis: When does upfront construction pay for itself?", x=0.08, y=1.03, ha="left", fontsize=13.0, fontweight="bold")

    # A projection, not a measurement: both styles say so inside the figure, not only in the caption.
    fig.tight_layout(rect=(0, 0.13, 1, 0.87))
    fig.text(
        0.01, 0.005,
        "PROJECTION for Gemma 4 E2B on a 419-turn history, not a phone measurement.\n"
        "Phone-measured: full-context and vector-RAG query energy (vector without its build), vector build, pilot graph build.\n"
        "Estimated: graph query energy, scaled from vector RAG by prompt tokens. Accuracy: workstation n=200 record.",
        fontsize=6.8, color="#334155", va="bottom", linespacing=1.3,
    )
    saved = save_figure(fig, "fig17_amortization_frontier", output_dirs)
    return saved


# =============================================================================
# Visual 5: Prompt Robustness Slopegraph — Figure 18
# =============================================================================
def plot_prompt_robustness(output_dirs: list[Path], style: str = "paper") -> list[Path]:
    if style == "paper":
        configure_paper_style()
    else:
        configure_presentation_style()

    fig, (ax_e2b, ax_e4b) = plt.subplots(1, 2, figsize=((7.6, 3.4) if style == "presentation" else paper_size(3.0)), sharey=True)
    palette = PAPER_PALETTE if style == "paper" else PRESENTATION_PALETTE

    models_data = [
        (ax_e2b, "(a) Gemma E2B (2.3B effective)", {
            "Vector RAG": (0.360, 0.320, palette["Vector RAG"], "o"),
            "GraphRAG v3": (0.215, 0.175, palette["GraphRAG v3"], "^"),
            "Full context": (0.075, 0.015, palette["Full context"], "s"),
        }),
        (ax_e4b, "(b) Gemma E4B (4.5B effective)", {
            "Vector RAG": (0.405, 0.350, palette["Vector RAG"], "o"),
            "GraphRAG v3": (0.275, 0.245, palette["GraphRAG v3"], "^"),
            "Full context": (0.105, 0.055, palette["Full context"], "s"),
        }),
    ]

    for ax, title, methods in models_data:
        ax.set_title(title, loc="left", fontsize=9.5, fontweight="semibold" if style == "presentation" else "normal")
        ax.set_xlim(-0.35, 1.35)
        ax.set_ylim(-1, 48)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Our prompt", "Official LoCoMo\nprompt"], fontsize=8.5)
        ax.yaxis.grid(True, linestyle="--" if style == "presentation" else "-", alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)

        for method_name, (std_p, off_p, color, marker) in methods.items():
            std_y = std_p * 100.0
            off_y = off_p * 100.0
            std_lo, std_hi = [v * 100.0 for v in wilson_interval(int(round(std_p * 200)), 200)]
            off_lo, off_hi = [v * 100.0 for v in wilson_interval(int(round(off_p * 200)), 200)]

            ax.plot([0, 1], [std_y, off_y], "-", color=color, lw=1.8, zorder=3, alpha=0.9)
            ax.errorbar(
                [0, 1], [std_y, off_y],
                yerr=[[std_y - std_lo, off_y - off_lo], [std_hi - std_y, off_hi - off_y]],
                fmt=marker, color=color, markerfacecolor=color, markeredgewidth=1.0,
                markersize=6.5, capsize=2.5, lw=1.0, zorder=4
            )

            diff = off_y - std_y
            ax.text(0 - 0.05, std_y, f"{std_y:.1f}%", ha="right", va="center", fontsize=7.8, color=color, fontweight="semibold")
            ax.text(1 + 0.05, off_y, f"{off_y:.1f}%", ha="left", va="center", fontsize=7.8, color=color, fontweight="semibold")

            mid_y = (std_y + off_y) / 2.0
            ax.text(0.48, mid_y + 1.2, method_name, ha="center", va="bottom", fontsize=7.5, color=color, fontweight="semibold")

    ax_e2b.set_ylabel("70B-Judge Accuracy (%)", fontsize=9.0)

    # No suptitle: the slide carries its own title.

    fig.tight_layout()
    saved = save_figure(fig, "fig18_prompt_robustness", output_dirs)
    return saved


# =============================================================================
# Visual 6: Parity Gate Diagnostic — Figure 15
# =============================================================================
def plot_parity_gate(output_dirs: list[Path], style: str = "paper") -> list[Path]:
    if style == "paper":
        configure_paper_style()
    else:
        configure_presentation_style()

    fig, (ax_bar, ax_mat) = plt.subplots(1, 2, figsize=((7.4, 3.2) if style == "presentation" else paper_size(2.8)), gridspec_kw={"width_ratios": [1.1, 0.9]})

    # Panel A: Parity threshold compliance
    y_pos = [0, 1]
    metrics = ["Exact string match", "Judged agreement"]
    vals = [40.0, 83.0]
    colors = ["#94A3B8", "#D97706"]

    ax_bar.barh(y_pos, vals, height=0.40, color=colors, alpha=0.9, zorder=2)
    ax_bar.errorbar([83.0], [1], xerr=[[83.0 - 74.5], [89.0 - 83.0]], fmt="none", color="#1E293B", capsize=3.5, lw=1.3, zorder=3)

    ax_bar.axvline(90.0, color="#DC2626", linestyle="--", linewidth=1.2, zorder=3)
    ax_bar.text(90.0, 1.42, "90% threshold", color="#DC2626", fontsize=7.8, ha="center", va="bottom", fontweight="semibold")

    # Centred in the bar, clear of the 74.5-89.0% interval drawn over the bar's right end.
    ax_bar.text(83.0 / 2, 1, "83.0%", va="center", ha="center", fontsize=8.2, color="white", fontweight="bold")
    ax_bar.text(40.0 / 2, 0, "40.0%", va="center", ha="center", fontsize=8.2, color="white", fontweight="bold")

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(metrics, fontsize=8.8)
    ax_bar.set_xlim(0, 100)
    ax_bar.set_ylim(-0.5, 1.6)
    ax_bar.set_xlabel("Agreement rate (%) on $n=100$", fontsize=9.0)
    ax_bar.set_title("(a) Parity gate compliance", loc="left", fontsize=9.5, fontweight="semibold" if style == "presentation" else "normal")
    ax_bar.xaxis.grid(True, linestyle="--" if style == "presentation" else "-", alpha=0.7)
    ax_bar.spines[["top", "right"]].set_visible(False)

    # Panel B: 2x2 Decision Matrix
    conf_mat = np.array([[25, 5], [12, 58]])
    # Grey, not green: shade encodes the share of questions, and green read "both wrong" as good.
    cax = ax_mat.imshow(conf_mat, cmap=plt.cm.Greys, aspect="auto", vmin=0, vmax=80)

    cell_labels = [
        ["Both correct\n25%", "Box only\n5%"],
        ["Phone only\n12%", "Both wrong\n58%"],
    ]
    for r in range(2):
        for c in range(2):
            text_color = "white" if conf_mat[r, c] > 35 else "#1E293B"
            ax_mat.text(c, r, cell_labels[r][c], ha="center", va="center", fontsize=8.2, color=text_color, fontweight="semibold")

    ax_mat.set_xticks([0, 1])
    ax_mat.set_xticklabels(["Phone pass", "Phone fail"], fontsize=8.5)
    ax_mat.set_yticks([0, 1])
    ax_mat.set_yticklabels(["Box pass", "Box fail"], fontsize=8.5)
    ax_mat.set_title("(b) Platform decision matrix", loc="left", fontsize=9.5, fontweight="semibold" if style == "presentation" else "normal")
    ax_mat.tick_params(top=False, bottom=True, labelbottom=True)

    if style == "presentation":
        fig.suptitle("Box ↔ S23 Phone Parity Gate: Why Platform Measurements Are Kept Separate", x=0.06, y=1.02, ha="left", fontsize=13.0, fontweight="bold")

    fig.tight_layout()
    saved = save_figure(fig, "fig15_parity_gate", output_dirs)
    return saved


# =============================================================================
# Visual 7: Prefill vs Decode Latency & RAM Breakdown — Figure 20
# =============================================================================
def plot_phone_prefill_decode_memory(output_dirs: list[Path], style: str = "paper") -> list[Path]:
    if style == "paper":
        configure_paper_style()
    else:
        configure_presentation_style()

    fig, ax_lat = plt.subplots(figsize=((7.2, 4.8) if style == "presentation" else paper_size(4.2)))

    # Measured: every answered E2B phone question, prompt length against query time. The phone logs
    # only total generation time, so prefill and decode are not separated here.
    slides = style == "presentation"
    series_style = {
        "Full context": dict(color="#CC3311" if slides else PAPER_PALETTE["Full context"], marker="s"),
        "Vector RAG, k=10": dict(color="#0077BB" if slides else PAPER_PALETTE["Vector RAG"], marker="o"),
        "Vector RAG, k=2": dict(color="#7FB8DE" if slides else "#56B4E9", marker="o"),
        # An early graph configuration, not GraphRAG v3: the thesis draws it in the gray that
        # fig04 and fig07 use for pre-repair graph cells, not in v3's green.
        "Graph pilot (early config)": dict(color="#009988" if slides else "#7A8288", marker="^"),
    }
    medians = {}
    for label, runs in PHONE_E2B_LATENCY_RUNS.items():
        tokens, seconds = _phone_query_times(runs)
        medians[label] = (float(np.median(tokens)), float(np.median(seconds)))
        style_kw = series_style[label]
        ax_lat.scatter(tokens, seconds, s=13, alpha=0.5, edgecolors="none", zorder=3, **style_kw)
        ax_lat.scatter([np.median(tokens)], [np.median(seconds)], s=75, edgecolors="white", linewidths=0.9, zorder=4,
                       label="%s: %d questions" % (label, len(seconds)), **style_kw)

    (fc_tok, fc_s), (k2_tok, k2_s) = medians["Full context"], medians["Vector RAG, k=2"]
    ax_lat.text(6300, 10.5, "Medians: full context has %.0f× the prompt tokens of k=2\n(%s vs %s) and takes %.0f× as long (%.1f vs %.1f s)"
                % (fc_tok / k2_tok, f"{fc_tok:,.0f}", f"{k2_tok:,.0f}", fc_s / k2_s, fc_s, k2_s),
                ha="right", va="bottom", fontsize=7.4, color="#59646B")

    ax_lat.set_xscale("log")
    ax_lat.set_yscale("log")
    ax_lat.set_xlim(220, 6500)
    ax_lat.set_ylim(8, 1000)
    ax_lat.set_xticks([250, 500, 1000, 2000, 4000])
    ax_lat.set_xticklabels(["250", "500", "1k", "2k", "4k"], fontsize=8.5)
    ax_lat.set_yticks([10, 30, 100, 300, 1000])
    ax_lat.set_yticklabels(["10", "30", "100", "300", "1,000"], fontsize=8.5)
    ax_lat.minorticks_off()
    ax_lat.set_xlabel("Prompt tokens per question (log scale)", fontsize=9.0)
    ax_lat.set_ylabel("Query time (s, log scale)", fontsize=9.0)
    ax_lat.set_title("Measured query time on S23, Gemma 4 E2B", loc="left", fontsize=9.5, fontweight="semibold" if style == "presentation" else "normal")
    ax_lat.grid(True, which="major", linestyle=":", alpha=0.5)
    ax_lat.set_axisbelow(True)
    ax_lat.legend(loc="upper left", fontsize=7.2, framealpha=0.92)

    fig.tight_layout()
    saved = save_figure(fig, "fig20_phone_prefill_decode_memory", output_dirs)
    return saved


# =============================================================================
# Visual 8: Extracted Subgraph Case Study — Figure 21
# =============================================================================
PHONE_PILOT_DIR = PROJECT_ROOT / "results_phone" / "_smoke_recovered" / "phone_smoke_gr"
PHONE_PILOT_GRAPH = (
    PHONE_PILOT_DIR
    / "graph_cache"
    / "a7096b83a38236d627ef52afcb57137abbfaab6e1227132bd1da1aa579638e16_graphrag_ext_gemma4-e2b_triple_json_v2_d_1.0_p_0.15_batch_5_max_12_ner_heuristic.pkl"
)
CASE_QUESTION = "What did Caroline research?"


class _TokenCount:
    """The pickled memory carries no backend; ego retrieval only needs its token estimate."""

    @staticmethod
    def approx_tokens(text: str) -> int:
        return max(1, len(text) // 4)


def _pilot_prompt_facts(question: str) -> list[str]:
    """The retrieved facts exactly as the phone pilot's reader saw them (debug_prompt_response.log)."""
    log = (PHONE_PILOT_DIR / "debug_prompt_response.log").read_text(encoding="utf-8")
    block = log.split(f"=== Q: {question} ===", 1)[1]
    return block.split("Conversation history:\n", 1)[1].split("\n\nQuestion:", 1)[0].splitlines()


def _caroline_case() -> dict:
    """fig21's case, measured: the phone pilot's conv-26 graph re-queried with the deployed ego retrieval.

    Scores are deterministic, but 72 candidates tie and the order among tied edges follows Python's
    per-process string hashing (networkx iterates small subgraph views over a set), so a re-query can
    keep different tied facts. The kept set is therefore read from the pilot's recorded prompt, and the
    re-query only supplies the scores; both counts must match the recorded ctx_meta (asserted).
    Known limit: the per-candidate scores are captured by wrapping run_memory_eval's sorted() for one
    context_for call; if the ego ranking code changes, the assertions here fail."""
    with PHONE_PILOT_GRAPH.open("rb") as handle:
        mem = pickle.load(handle)
    mem.backend = _TokenCount()
    captured: list[list] = []

    def spy(items, *args, **kwargs):
        out = sorted(items, *args, **kwargs)
        captured.append(out)
        return out

    run_memory_eval.sorted = spy
    try:
        _, meta = mem.context_for(CASE_QUESTION)
    finally:
        del run_memory_eval.sorted
    ranked = next(c for c in captured if c and len(c[0]) == 5)  # (u, v, key, score, data), best first

    with (PHONE_PILOT_DIR / "judged.jsonl").open(encoding="utf-8") as handle:
        record = next(r for r in map(json.loads, handle) if r["question"] == CASE_QUESTION)
    for key in ("candidate_nodes", "candidate_edges", "kept_facts"):
        assert meta[key] == record["ctx_meta"][key], key
    assert len(ranked) == meta["candidate_edges"]

    with (PROJECT_ROOT / "locomo10.json").open(encoding="utf-8") as handle:
        conv = next(c for c in json.load(handle) if c["sample_id"] == "conv-26")
    qa = next(q for q in conv["qa"] if q["question"] == CASE_QUESTION)
    turns = [t for k, v in conv["conversation"].items() if isinstance(v, list) for t in v]
    evidence = next(t for t in turns if t["dia_id"] == qa["evidence"][0])

    def fact(row) -> str:  # the context line run_memory_eval writes for a kept edge
        u, v, _, _, d = row
        prefix = f"[{d['timestamp']}] " if d.get("timestamp") else ""
        return f"{prefix}System: {mem._format_node(u)} {d['relation']} {mem._format_node(v)}."

    pilot_facts = set(_pilot_prompt_facts(CASE_QUESTION))
    in_pilot = [fact(row) in pilot_facts for row in ranked]
    assert sum(in_pilot) == len(pilot_facts) == meta["kept_facts"]
    # pilot's kept facts first within each score level; the order inside a tie is not reproducible
    order = sorted(range(len(ranked)), key=lambda i: (-ranked[i][3], not in_pilot[i], fact(ranked[i])))
    ranked = [ranked[i] for i in order]
    in_pilot = [in_pilot[i] for i in order]

    graph = mem.graph
    speakers = {"caroline", "melanie"}
    budget = mem.max_triples
    scores = [row[3] for row in ranked]
    assert in_pilot == [i < budget for i in range(len(ranked))]  # the pilot kept the top scores
    cutoff = scores[budget - 1]
    kept = ranked[:budget]
    edge_texts = [f"{u} {d.get('relation', '')} {v}" for u, v, d in graph.edges(data=True)]
    return {
        "scores": scores,
        "exchange": [{u, v} == speakers for u, v, _, _, _ in ranked],
        "budget": budget,
        "cutoff": cutoff,
        "tied": sum(1 for s in scores if s == cutoff),
        "tied_kept": sum(1 for s in scores[:budget] if s == cutoff),
        "kept_exchanges": sum(1 for u, v, _, _, _ in kept if {u, v} == speakers),
        "adoption_rank": next(i for i, row in enumerate(ranked, 1) if "adoption process" in row[1]),
        "adoption_score": next(row[3] for row in ranked if "adoption process" in row[1]),
        "research_in_graph": any(s in text for text in edge_texts for s in ("research", "agenc")),
        "adoption_triples": [
            (u, d.get("relation", ""), v) for u, v, d in graph.edges(data=True) if "adopt" in f"{u} {v}"
        ],
        "candidates": meta["candidate_edges"],
        "caroline_degree": graph.degree("caroline"),
        "hub_share": sum(1 for u, v in graph.edges() if u in speakers or v in speakers) / graph.number_of_edges(),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "batches_failed": mem.extraction_stats["batches_failed"],
        "batches_total": mem.extraction_stats["batches_total"],
        "evidence_id": evidence["dia_id"],
        "evidence_text": evidence["text"],
        "gold": qa["answer"],
        "pred": record["pred"],
        "judge_correct": record["judge_correct"],
    }


def plot_extracted_subgraph_case_study(output_dirs: list[Path], style: str = "paper") -> list[Path]:
    if style == "paper":
        configure_paper_style()
    else:
        configure_presentation_style()
    case = _caroline_case()

    fig, (ax_rank, ax_flow) = plt.subplots(1, 2, figsize=((7.4, 3.7) if style == "presentation" else paper_size(3.6)), gridspec_kw={"width_ratios": [1.0, 1.25]})

    # (a) every candidate fact, best first, as the deployed ego retrieval scored it
    exchange_color, other_color, dropped_color = "#7A8288", "#0072B2", "#CBD5E1"
    for rank, (score, exchange) in enumerate(zip(case["scores"], case["exchange"]), 1):
        kept = rank <= case["budget"]
        color = (exchange_color if exchange else other_color) if kept else dropped_color
        ax_rank.scatter(rank, score, s=16, color=color, edgecolor="none", zorder=3)
    adoption_score = case["adoption_score"]
    ax_rank.scatter(case["adoption_rank"], adoption_score, s=46, marker="D", color=PAPER_PALETTE["fail"], zorder=4)
    ax_rank.annotate(
        "Adoption triple: tied,\nnot kept in the pilot",
        xy=(case["adoption_rank"], adoption_score), xytext=(28, 0.82), fontsize=7.2,
        arrowprops=dict(arrowstyle="->", color="#1E293B", lw=0.8),
    )
    ax_rank.axvline(case["budget"] + 0.5, color="#1E293B", linestyle="--", linewidth=0.9, zorder=2)
    # Above the legend, which sits in the empty lower right.
    ax_rank.text(case["budget"] + 1.5, 0.47, "%d-fact\nbudget" % case["budget"], fontsize=7.2, color="#1E293B")
    ax_rank.text(
        case["candidates"], case["cutoff"] + 0.035,
        "%d candidates tie at %.2f" % (case["tied"], case["cutoff"]), fontsize=7.2, ha="right", color="#1E293B",
    )
    ax_rank.set_xlim(0, case["candidates"] + 2)
    ax_rank.set_ylim(0, 1.05)
    ax_rank.set_xlabel("Candidate fact, ranked by retrieval score", fontsize=8.5)
    ax_rank.set_ylabel("Retrieval score", fontsize=8.5)
    ax_rank.set_title("(a) How the 12 facts were chosen", loc="left", fontsize=9.0, fontweight="semibold" if style == "presentation" else "normal")
    ax_rank.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax_rank.spines[["top", "right"]].set_visible(False)
    ax_rank.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", color=exchange_color, label="Kept: Caroline–Melanie exchange"),
            Line2D([], [], marker="o", linestyle="", color=other_color, label="Kept: other fact"),
            Line2D([], [], marker="o", linestyle="", color=dropped_color, label="Not kept"),
        ],
        loc="lower right", bbox_to_anchor=(1.0, 0.12), fontsize=6.8, frameon=False, handletextpad=0.2,
    )

    # (b) the failure chain, every line computed above or read from the run record
    ax_flow.axis("off")
    ax_flow.set_title("(b) The measured failure", loc="left", fontsize=9.0, fontweight="semibold" if style == "presentation" else "normal")
    box = dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor="#94A3B8", linewidth=0.8)
    fail_box = dict(boxstyle="round,pad=0.35", facecolor="#FDEDEC", edgecolor=PAPER_PALETTE["fail"], linewidth=0.8)
    quote = case["evidence_text"].split(" and give")[0] + "…"
    adoption = "\n".join("   %s –%s→ %s" % t for t in case["adoption_triples"])
    steps = [
        (0.99, box, "1. Evidence (LoCoMo conv-26, turn %s)\nCaroline: \"%s\"\nGold answer: %s"
         % (case["evidence_id"], quote, case["gold"])),
        (0.76, box, "2. Phone extraction (Gemma 4 E2B)\nNo node or relation mentions \"research\" or \"agenc\".\n"
         "The only adoption triples:\n%s\n%d of %d extraction batches failed on the phone."
         % (adoption, case["batches_failed"], case["batches_total"])),
        (0.44, box, "3. Ego retrieval (one hop, %d facts)\nQuery matches hub Caroline (degree %d): %d candidates.\n"
         "%d tie; iteration order, not relevance, fills %d slots.\n"
         "%d of 12 kept facts are Caroline–Melanie exchanges."
         % (case["budget"], case["caroline_degree"], case["candidates"], case["tied"], case["tied_kept"],
            case["kept_exchanges"])),
        (0.22, fail_box, "4. Answer: \"%s\" (judged %s)"
         % (case["pred"], "correct" if case["judge_correct"] else "wrong")),
    ]
    for y, style_box, text in steps:
        # 56 characters keep the widest box inside the canvas at 6.8 pt.
        text = "\n".join(textwrap.fill(line, 56, subsequent_indent="  ") for line in text.split("\n"))
        ax_flow.text(0.0, y, text, transform=ax_flow.transAxes, fontsize=6.8, va="top", ha="left",
                     bbox=style_box, linespacing=1.25)

    fig.tight_layout()
    return save_figure(fig, "fig21_extracted_subgraph_case_study", output_dirs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the extended thesis figures in paper, presentation, or both formats.")
    parser.add_argument("--style", choices=["paper", "presentation", "both"], default="both")
    args = parser.parse_args()

    generated: list[Path] = []
    if args.style in ("paper", "both"):
        # Paper targets directly into THESIS_DIR
        out_dirs = [THESIS_DIR]
        generated.extend(plot_model_family_scaling(out_dirs, style="paper"))
        generated.extend(plot_kv_ram_wall(out_dirs, style="paper"))
        generated.extend(plot_amortization_frontier(out_dirs, style="paper"))
        generated.extend(plot_prompt_robustness(out_dirs, style="paper"))
        generated.extend(plot_parity_gate(out_dirs, style="paper"))
        generated.extend(plot_phone_prefill_decode_memory(out_dirs, style="paper"))
        generated.extend(plot_extracted_subgraph_case_study(out_dirs, style="paper"))

    if args.style in ("presentation", "both"):
        # Presentation targets into PRESENTATION_DIR and THESIS_DIR / "presentation"
        out_dirs = [PRESENTATION_DIR, THESIS_DIR / "presentation"]
        generated.extend(plot_model_family_scaling(out_dirs, style="presentation"))
        generated.extend(plot_kv_ram_wall(out_dirs, style="presentation"))
        generated.extend(plot_amortization_frontier(out_dirs, style="presentation"))
        generated.extend(plot_prompt_robustness(out_dirs, style="presentation"))
        generated.extend(plot_parity_gate(out_dirs, style="presentation"))
        generated.extend(plot_phone_prefill_decode_memory(out_dirs, style="presentation"))
        generated.extend(plot_extracted_subgraph_case_study(out_dirs, style="presentation"))

    print(f"Generated {len(generated)} files for style '{args.style}'")


if __name__ == "__main__":
    main()
