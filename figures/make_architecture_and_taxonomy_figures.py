#!/usr/bin/env python3
"""Generates Visual A (LongMemEval-S accuracy by question type) and Visual D (3-Way Architecture Diagram)
for the thesis.

Supports dual styles:
  - Paper: Clean publication-ready styling, minimal ink, no in-canvas title banners.
  - Presentation: Bold takeaway titles, zone highlights, slide callout boxes.

Alternative Encodings for Visual A (Figure 14):
  - Cleveland dot / range plot (fig14_category_taxonomy_dotplot)
  - Heatmap matrix (fig14_category_taxonomy_heatmap)
  - Grouped horizontal bars (fig14_category_taxonomy_bars)

Outputs:
  - fig13_architecture_diagram.png, .pdf, .svg (Visual D)
  - fig14_category_taxonomy.png, .pdf, .svg (Visual A)
  - fig14_category_taxonomy_dotplot.png, .pdf, .svg
  - fig14_category_taxonomy_heatmap.png, .pdf, .svg
  - fig14_category_taxonomy_bars.png, .pdf, .svg
"""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np

from figures.visual_style import (
    configure_paper_style,
    configure_presentation_style,
    PAPER_PALETTE,
    PRESENTATION_PALETTE,
    NUMBER_LABEL_BOX_PAPER,
    NUMBER_LABEL_BOX_PRESENTATION,
    paper_size,
    save_figure,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THESIS_DIR = PROJECT_ROOT / "results" / "thesis_visuals"
PRESENTATION_DIR = PROJECT_ROOT / "results" / "presentation_visuals"


# =============================================================================
# Visual A: Alternative 1 - Cleveland Dot / Range Plot (Paper Preferred)
# =============================================================================
def plot_category_taxonomy_dotplot(output_dirs: list[Path], is_primary: bool = True) -> list[Path]:
    configure_paper_style()
    categories = [
        "Single-session assistant",
        "Single-session user",
        "Knowledge update",
        "Temporal reasoning",
        "Multi-session reasoning",
        "Single-session preference",
    ]
    vr_e4b = [90.0, 90.0, 80.0, 20.0, 50.0, 10.0]
    vr_e2b = [90.0, 80.0, 70.0, 20.0, 10.0, 20.0]
    # GraphRAG v3, prefixed (lme_strat60_e4b_graphrag_v3_prefix in results/lme_category_table.csv,
    # permuted into this category order)
    gr_e4b = [20.0, 40.0, 30.0, 0.0, 10.0, 40.0]
    fc_e4b = [0.0, 0.0, 10.0, 0.0, 10.0, 10.0]

    y = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=paper_size(3.6))  # thesis-only figure: drawn at the text width

    # Range lines connecting min and max for each category
    for yi in y:
        all_vals = [vr_e4b[yi], vr_e2b[yi], gr_e4b[yi], fc_e4b[yi]]
        ax.plot([min(all_vals), max(all_vals)], [yi, yi], color="#CBD5E1", linewidth=2.2, zorder=1)

    # Many cells tie at 0, 10 or 20%: each series gets its own lane in the row, one marker height
    # apart, so three markers at the same value stay separately visible. Diamonds mean GraphRAG v4 in
    # fig03/fig19, so the smaller vector reader is a hollow circle instead.
    ax.scatter(fc_e4b, y - 0.3, color=PAPER_PALETTE["Full context"], marker="s", s=30, label="Full context (E4B)", zorder=3)
    ax.scatter(gr_e4b, y - 0.1, color=PAPER_PALETTE["GraphRAG v3"], marker="^", s=36, label="Graph triples v3 (E4B)", zorder=3)
    ax.scatter(vr_e2b, y + 0.1, facecolors="white", edgecolors=PAPER_PALETTE["Vector RAG"], linewidths=1.3, marker="o", s=30, label="Vector RAG (E2B)", zorder=3)
    ax.scatter(vr_e4b, y + 0.3, color=PAPER_PALETTE["Vector RAG"], marker="o", s=34, label="Vector RAG (E4B)", zorder=4)

    # Separator between the two observed groups; the notes name the question types, not a cause.
    ax.axhline(2.5, color="#94A3B8", linestyle=":", linewidth=1.0)
    ax.text(102, 1.0, "Single-session\nfacts & updates", va="center", ha="left", fontsize=8.0, color="#475569", style="italic")
    ax.text(102, 4.0, "Temporal,\nmulti-session\n& preference", va="center", ha="left", fontsize=8.0, color="#475569", style="italic")

    ax.set_yticks(y)
    ax.set_yticklabels(categories)
    ax.invert_yaxis()
    ax.set_xlim(-2, 105)
    ax.set_xlabel("Question-type accuracy on LongMemEval-S (%, $n=10$ per cell)")
    ax.xaxis.grid(True)
    ax.spines[["top", "right"]].set_visible(False)
    # Below the axes, so it cannot cover the right-hand zone notes.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    fig.tight_layout()

    saved = []
    saved.extend(save_figure(fig, "fig14_category_taxonomy_dotplot", output_dirs, close=False))
    if is_primary:
        saved.extend(save_figure(fig, "fig14_category_taxonomy", output_dirs, close=False))
    plt.close(fig)
    return saved


# =============================================================================
# Visual A: Alternative 2 - Heatmap Matrix (Paper Alternative)
# =============================================================================
def plot_category_taxonomy_heatmap(output_dirs: list[Path]) -> list[Path]:
    configure_paper_style()
    categories = [
        "Single-session assistant",
        "Single-session user",
        "Knowledge update",
        "Temporal reasoning",
        "Multi-session reasoning",
        "Single-session preference",
    ]
    methods = [
        "Full Context\n(E4B)",
        "Graph triples\n(E4B)",
        "Vector RAG\n(E2B)",
        "Vector RAG\n(E4B)",
    ]
    matrix = np.array([
        [0.0, 20.0, 90.0, 90.0],
        [0.0, 40.0, 80.0, 90.0],
        [10.0, 30.0, 70.0, 80.0],
        [0.0, 0.0, 20.0, 20.0],
        [10.0, 10.0, 10.0, 50.0],
        [10.0, 40.0, 20.0, 10.0],
    ])

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    cax = ax.imshow(matrix, cmap=plt.cm.Blues, aspect="auto", vmin=0, vmax=100)

    for r in range(len(categories)):
        for c in range(len(methods)):
            val = matrix[r, c]
            text_color = "white" if val > 55 else "#1E293B"
            ax.text(
                c, r, f"{val:.0f}%",
                ha="center", va="center", fontsize=8.8,
                color=text_color,
                fontweight="semibold" if val >= 70 else "normal",
            )

    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, fontsize=8.5)
    ax.set_yticks(np.arange(len(categories)))
    ax.set_yticklabels(categories, fontsize=8.5)
    ax.tick_params(top=False, bottom=True, labelbottom=True)
    ax.axhline(2.5, color="#475569", linewidth=1.4)

    cbar = fig.colorbar(cax, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Accuracy (%)", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()

    saved = save_figure(fig, "fig14_category_taxonomy_heatmap", output_dirs)
    return saved


# =============================================================================
# Visual A: Alternative 3 - Grouped Horizontal Bars (Presentation Format)
# =============================================================================
def plot_category_taxonomy_bars(output_dirs: list[Path], is_primary: bool = False) -> list[Path]:
    configure_presentation_style()
    categories = [
        "Single-session\nassistant",
        "Single-session\nuser",
        "Knowledge\nupdate",
        "Temporal\nreasoning",
        "Multi-session\nreasoning",
        "Single-session\npreference",
    ]
    vr_e4b = [90.0, 90.0, 80.0, 20.0, 50.0, 10.0]
    vr_e2b = [90.0, 80.0, 70.0, 20.0, 10.0, 20.0]
    # GraphRAG v3, prefixed (lme_strat60_e4b_graphrag_v3_prefix in results/lme_category_table.csv,
    # permuted into this category order)
    gr_e4b = [20.0, 40.0, 30.0, 0.0, 10.0, 40.0]
    fc_e4b = [0.0, 0.0, 10.0, 0.0, 10.0, 10.0]

    y = np.arange(len(categories))
    height = 0.20
    fig, ax = plt.subplots(figsize=(10.8, 5.8))

    b1 = ax.barh(y - 1.5 * height, vr_e4b, height, color=PRESENTATION_PALETTE["Vector RAG"], label="Vector RAG (Gemma E4B)", edgecolor="#005588", zorder=3)
    b2 = ax.barh(y - 0.5 * height, vr_e2b, height, color="#44AAEE", label="Vector RAG (Gemma E2B)", edgecolor="#2288CC", zorder=3)
    b3 = ax.barh(y + 0.5 * height, gr_e4b, height, color=PRESENTATION_PALETTE["GraphRAG v3"], label="Graph triples (Gemma E4B)", edgecolor="#006B60", hatch="//", zorder=3)
    b4 = ax.barh(y + 1.5 * height, fc_e4b, height, color=PRESENTATION_PALETTE["Full context"], label="Full Context (Gemma E4B)", edgecolor="#992200", hatch="\\\\", zorder=3)

    for bar_group, vals, col in [
        (b1, vr_e4b, PRESENTATION_PALETTE["Vector RAG"]),
        (b2, vr_e2b, "#0077BB"),
        (b3, gr_e4b, PRESENTATION_PALETTE["GraphRAG v3"]),
        (b4, fc_e4b, PRESENTATION_PALETTE["Full context"]),
    ]:
        for bar, v in zip(bar_group, vals):
            if v > 0:
                ax.annotate(
                    f"{v:.0f}%",
                    (bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2),
                    va="center", fontsize=7.8, fontweight="bold", color=col,
                    bbox=NUMBER_LABEL_BOX_PRESENTATION, zorder=5,
                )
            else:
                ax.annotate(
                    "0%",
                    (1.5, bar.get_y() + bar.get_height() / 2),
                    va="center", fontsize=7.2, color="#888888", zorder=5,
                )

    ax.axhspan(-0.5, 2.5, color="#E8F4F8", alpha=0.5, zorder=0)
    ax.text(102, 1.0, "Single-session facts & updates\n(Vector RAG 70–90%;\ngraph 20–40%, full context 0–10%)",
            va="center", ha="left", fontsize=8.2, fontweight="bold", color="#005588",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#0077BB", alpha=0.9))

    ax.axhspan(2.5, 5.5, color="#FDF2E9", alpha=0.5, zorder=0)
    ax.text(102, 4.0, "Temporal, multi-session & preference\n(every strategy 0–50%;\ngraph highest on preference)",
            va="center", ha="left", fontsize=8.2, fontweight="bold", color="#A04000",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#E67E22", alpha=0.9))

    ax.set_yticks(y)
    ax.set_yticklabels(categories, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Question-Type Accuracy (%) — LongMemEval-S (n=10 per cell, 70B judge)")
    ax.set_title("Accuracy by question type: two observed groups", loc="left", pad=12)
    ax.grid(axis="x", color="#E2E5E8", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=8.6)

    fig.text(0.08, -0.03, "LongMemEval-S stratified n=60, ten questions per type, so one question moves a cell by ten points. The groups are observed accuracy levels, not a tested cause.", fontsize=8.3, color="#555B60")

    saved = []
    saved.extend(save_figure(fig, "fig14_category_taxonomy_bars", output_dirs, close=False))
    if is_primary:
        saved.extend(save_figure(fig, "fig14_category_taxonomy", output_dirs, close=False))
    plt.close(fig)
    return saved


# =============================================================================
# Visual D: 3-Way Memory Pipeline Architecture Diagram
# =============================================================================
def _plot_architecture_diagram_paper(output_dirs: list[Path]) -> list[Path]:
    """The thesis version of fig13: the same schematic at the 6.3 in text width.

    At that width the slide's three lines per box no longer fit at a readable size, so each box keeps
    a short title and one line, and the detail moves to the Chapter 3 caption, which already carries
    all of it (4,096-token budget, top-10 turns in time order, five-turn extraction batches, v3 and
    v4, both platforms, and the 70B judge). A method schematic: no result numbers.
    """
    configure_paper_style()
    fig, ax = plt.subplots(figsize=paper_size(3.5))
    # A drawing, not a plot: the axes take the whole canvas, or the default side margins would
    # leave it narrower than the text width and LaTeX would scale it up.
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    ax.set_xlim(0, 102)
    ax.set_ylim(-2, 100)
    ax.axis("off")
    ink, muted, edge = "#1E293B", "#475569", "#64748B"

    def box(x, y, w, h, title, line, fill):
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                            facecolor=fill, edgecolor=edge, linewidth=0.7, zorder=2))
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=8.0, fontweight="bold", color=ink, zorder=3)
        if line:
            ax.text(x + w / 2, y + h * 0.28, line, ha="center", va="center", fontsize=7.5, color=muted, zorder=3)

    def arrow(x1, x2, y):
        ax.annotate("", xy=(x2, y), xytext=(x1, y), zorder=4,
                    arrowprops=dict(arrowstyle="->", color=edge, lw=0.9, mutation_scale=9))

    for x, header in ((25, "Ingestion"), (57, "Context at query time"), (88, "Reader")):
        ax.text(x, 95, header, ha="center", va="center", fontsize=8.0, fontweight="bold", color=muted)

    rows = [  # y centre, row label, palette key, tint, ingestion box, context box
        (80, "Full\ncontext", "Full context", "#FBE3D6", ("Dialogue history", "every turn, in order"),
         ("Prompt window", "latest turns that fit")),
        (59, "Vector\nretrieval", "Vector RAG", "#D6E6F2", ("Turn embeddings", "one vector per turn"),
         ("Retrieved turns", "top 10, in time order")),
        (38, "Graph\nmemory", "GraphRAG v3", "#D4EFE6", ("Triple graph", "from 5-turn batches"),
         ("Retrieved facts", "v3 triples; v4 + sources")),
    ]
    for y, label, key, tint, ingest, context in rows:
        ax.text(0.5, y, label, ha="left", va="center", fontsize=8.0, fontweight="bold", color=PAPER_PALETTE[key])
        box(13, y - 7, 24, 14, *ingest, tint)
        arrow(37.8, 43.2, y)
        box(44, y - 7, 26, 14, *context, tint)
        arrow(70.8, 75.2, y)
    box(76, 31, 24, 56, "SLM reader", "", "#EEF1F4")
    ax.text(88, 52, "one answer prompt\nfor every pathway\n\nfive readers,\nQ4_K_M weights",
            ha="center", va="center", fontsize=7.5, color=muted, linespacing=1.3, zorder=3)

    ax.plot([0.5, 101.5], [24.5, 24.5], color="#CBD5E1", linewidth=0.6)
    box(13, 3, 38, 15, "(B) Workstation", "Ollama · accuracy, all five readers", "#F4F6F8")
    box(55, 3, 45, 15, "(C) Phone: Galaxy S23 Ultra", "llama.cpp · answers, energy, heat, RAM", "#F4F6F8")
    ax.text(0.5, 10.5, "Two\nplatforms", ha="left", va="center", fontsize=8.0, fontweight="bold", color=muted)
    return save_figure(fig, "fig13_architecture_diagram", output_dirs)


# =============================================================================
# fig22: how each memory strategy looks up evidence at question time
# =============================================================================
def plot_lookup_methods(output_dirs: list[Path]) -> list[Path]:
    """Thesis fig22: what each strategy stores, what it does at question time, what the reader gets.

    A schematic with toy turns, no result numbers. Every parameter matches Chapter 3: full context
    keeps the newest turns within 4,096 - 192 tokens; vector retrieval takes the top 10 turns by
    cosine similarity and re-sorts them by time; the graph (v2-v4) selects edges whose source batch
    overlaps the top 10 turns, ranks them by best-turn similarity times edge weight, keeps 12, and
    v4 appends each source turn once. Drawn at the text width so it prints 1:1.
    """
    configure_paper_style()
    fig, ax = plt.subplots(figsize=paper_size(5.0))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ink, muted, edge, grey = "#1E293B", "#475569", "#64748B", "#CBD5E1"
    body = 7.5

    columns = [  # x left, palette key, tint, title
        (1.0, "Full context", "#FBE3D6", "Full context"),
        (34.3, "Vector RAG", "#D6E6F2", "Vector retrieval"),
        (67.6, "GraphRAG v3", "#D4EFE6", "Graph memory (v2–v4)"),
    ]
    width = 31.4
    rows = [(68, 22, "Stored at ingestion"), (36, 26, "At question time"), (6, 24, "Given to the reader")]
    n_turns = 12

    def panel(x, y, h, tint, label):
        ax.add_patch(patches.FancyBboxPatch((x, y), width, h, boxstyle="round,pad=0.4,rounding_size=1.0",
                                            facecolor=tint, edgecolor=edge, linewidth=0.6, zorder=1))
        ax.text(x + 1.2, y + h - 2.2, label, ha="left", va="center", fontsize=6.9, color=muted, style="italic", zorder=3)

    def strip(x, y, colors, h=3.2):
        step = (width - 4) / n_turns
        for i, c in enumerate(colors):
            ax.add_patch(patches.Rectangle((x + 2 + i * step, y), step * 0.86, h, facecolor=c, edgecolor=edge, linewidth=0.4, zorder=2))
        return step

    def note(x, y, text, fontsize=body, **kw):
        ax.text(x + width / 2, y, text, ha="center", va="center", fontsize=fontsize, color=ink, linespacing=1.25, zorder=3, **kw)

    for x, key, tint, title in columns:
        ax.text(x + width / 2, 96.5, title, ha="center", va="center", fontsize=8.5, fontweight="bold", color=PAPER_PALETTE[key])
        for y, h, label in rows:
            panel(x, y, h, tint, label)
        for y_top, y_bottom in ((68, 62), (36, 30)):
            ax.annotate("", xy=(x + width / 2, y_bottom + 0.6), xytext=(x + width / 2, y_top - 0.6),
                        arrowprops=dict(arrowstyle="->", color=edge, lw=0.9, mutation_scale=9), zorder=4)

    strong = {"Full context": "#E8A27C", "Vector RAG": "#6FA8D6", "GraphRAG v3": "#5DBE9C"}

    # Full context: every turn in order; the window keeps the newest turns.
    x = columns[0][0]
    strip(x, 76, [strong["Full context"]] * n_turns)
    note(x, 72.3, "every turn, in time order")
    kept = 5
    step = strip(x, 51, ["#EEF1F4"] * (n_turns - kept) + [strong["Full context"]] * kept)
    split = x + 2 + (n_turns - kept) * step - step * 0.07
    ax.text(x + 2 + (n_turns - kept) * step / 2, 56.5, "oldest dropped", ha="center", va="center", fontsize=6.9, color=muted)
    ax.text(split + kept * step / 2, 56.5, "newest kept", ha="center", va="center", fontsize=6.9, color=PAPER_PALETTE["Full context"], fontweight="bold")
    ax.plot([split, split], [49.5, 58.5], color=PAPER_PALETTE["Full context"], lw=0.9, ls="--", zorder=3)
    note(x, 43, "no search: keep the newest turns\nthat fit 4,096 − 192 tokens")
    strip(x, 20, ["#EEF1F4"] * (n_turns - kept) + [strong["Full context"]] * kept, h=2.6)
    note(x, 13.5, "the newest turns, in time order;\nolder evidence is not seen")

    # Vector retrieval: one vector per turn; cosine scores; top k re-sorted by time.
    x = columns[1][0]
    step = strip(x, 76, [strong["Vector RAG"]] * n_turns)
    for i in range(n_turns):
        ax.plot(x + 2 + i * step + step * 0.43, 81.4, marker="o", markersize=2.2, color=PAPER_PALETTE["Vector RAG"], zorder=3)
    note(x, 72.3, "one embedding per turn")
    scores = [0.30, 0.82, 0.25, 0.40, 0.90, 0.20, 0.35, 0.28, 0.76, 0.33, 0.22, 0.38]
    top = sorted(range(n_turns), key=lambda i: -scores[i])[:3]
    for i, s in enumerate(scores):
        ax.add_patch(patches.Rectangle((x + 2 + i * step, 50), step * 0.86, 7.5 * s,
                                       facecolor=strong["Vector RAG"] if i in top else "#C9D3DD", edgecolor="none", zorder=2))
    ax.plot([x + 2, x + width - 2], [50, 50], color=edge, lw=0.5)
    note(x, 43, "embed the question; cosine\nsimilarity to every turn; top k = 10")
    chosen = sorted(top)
    step = strip(x, 20, ["#EEF1F4"] * n_turns, h=2.6)
    for i in chosen:
        ax.add_patch(patches.Rectangle((x + 2 + i * step, 20), step * 0.86, 2.6, facecolor=strong["Vector RAG"], edgecolor=edge, linewidth=0.4, zorder=3))
    note(x, 13.5, "the retrieved turns,\nre-sorted into time order")

    # Graph memory: triples from 5-turn batches with source links; edges selected through top turns.
    x = columns[2][0]
    step = strip(x, 74.5, ["#EEF1F4"] * n_turns, h=2.6)
    for b in range(0, n_turns, 5):
        b_end = min(b + 5, n_turns)
        ax.plot([x + 2 + b * step, x + 2 + b_end * step - step * 0.14], [73.4, 73.4], color=PAPER_PALETTE["GraphRAG v3"], lw=1.0)
    nodes = {"A": (x + 7, 82.5), "B": (x + 14, 85.5), "C": (x + 21, 82.0), "D": (x + 27, 85.0)}
    for a, b in (("A", "B"), ("B", "C"), ("C", "D"), ("A", "C")):
        ax.plot(*zip(nodes[a], nodes[b]), color=PAPER_PALETTE["GraphRAG v3"], lw=0.8, zorder=2)
    for nx, ny in nodes.values():
        ax.plot(nx, ny, marker="o", markersize=4.2, color=PAPER_PALETTE["GraphRAG v3"], markeredgecolor="white", markeredgewidth=0.5, zorder=3)
    note(x, 70.2, "(subject, relation, object) triples from\n5-turn batches; each keeps its batch", fontsize=6.9)
    step = strip(x, 52, ["#EEF1F4"] * n_turns, h=2.6)
    for i in (1, 8):
        ax.add_patch(patches.Rectangle((x + 2 + i * step, 52), step * 0.86, 2.6, facecolor=strong["Vector RAG"], edgecolor=edge, linewidth=0.4, zorder=3))
    ax.text(x + width / 2, 57.0, "top 10 turns, then their batches' edges", ha="center", va="center", fontsize=6.9, color=muted)
    note(x, 44, "rank edges by turn similarity ×\nedge weight; keep 12")
    ax.text(x + 2.0, 21.5, "v3: 12 triples\n(s, r, o)  (s, r, o)  …", ha="left", va="center", fontsize=body, color=ink, linespacing=1.25)
    ax.text(x + 2.0, 12.5, "v4: the same 12 triples\n+ their source turns, each once", ha="left", va="center", fontsize=body, color=ink, linespacing=1.25)
    return save_figure(fig, "fig22_lookup_methods", output_dirs)


def plot_architecture_diagram(output_dirs: list[Path], style: str = "paper") -> list[Path]:
    if style == "paper":
        return _plot_architecture_diagram_paper(output_dirs)
    configure_presentation_style()

    # A method schematic: no result numbers. Results live in Chapters 4-5, where phone (n=20) and
    # workstation (n=200) accuracies are reported separately under the parity rule.
    fig, ax = plt.subplots(figsize=(13.0, 6.6))
    ax.set_xlim(0, 102)
    ax.set_ylim(-4, 100)
    ax.axis("off")

    def draw_box(x, y, w, h, title, subtitle, cost, color, hatch=""):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.8,rounding_size=1.5",
            facecolor=color,
            edgecolor="#4F5D65",
            linewidth=1.1,
            hatch=hatch,
            alpha=0.92,
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=9.2, fontweight="bold", color="#1C2833", zorder=3)
        if subtitle:
            ax.text(x + w / 2, y + h * 0.40, subtitle, ha="center", va="center", fontsize=7.8, color="#4A5568", zorder=3)
        if cost:
            ax.text(x + w / 2, y + h * 0.16, cost, ha="center", va="center", fontsize=7.6, color="#4A5568", zorder=3)

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#4F5D65", lw=1.5, mutation_scale=13),
            zorder=4,
        )
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 2.0, label, ha="center", va="bottom", fontsize=7.8, color="#2C3E50", fontweight="semibold", bbox=NUMBER_LABEL_BOX_PAPER, zorder=5)

    if style == "presentation":
        fig.suptitle("Three memory pathways, one pipeline on two platforms", x=0.04, y=0.995, ha="left", fontsize=13.5, fontweight="bold", color="#1A252C")

    # (A) memory pathways: only the context differs; the reader and answer prompt are shared.
    ax.text(1, 97.5, "(A) Memory pathways", ha="left", fontsize=9.2, fontweight="bold", color="#1A252C")
    ax.text(23, 91, "1. INGESTION (WRITE TIME)", ha="center", fontsize=9.0, fontweight="bold", color="#34495E")
    ax.text(55, 91, "2. CONTEXT AT QUERY TIME", ha="center", fontsize=9.0, fontweight="bold", color="#34495E")
    ax.text(88, 91, "3. READER", ha="center", fontsize=9.0, fontweight="bold", color="#34495E")

    rows = [
        (77, "Full context\n(truncated)", "Full context", "#FADBD8",
         ("Raw dialogue history", "every turn, time-ordered", "no ingestion step"), "render turns",
         ("Prompt window", "most recent turns that fit", "4,096-token input window")),
        (56, "Vector retrieval\n(dense turns)", "Vector RAG", "#D4E6F1",
         ("Turn embeddings", "one vector per turn", "Nomic Embed Text v1.5"), "cosine top-k",
         ("Retrieved turns", "10 most similar turns", "re-sorted into time order")),
        (35, "Graph memory\n(SLM triples)", "GraphRAG v3", "#D1F2EB",
         ("Triple graph", "reader extracts (s, r, o)", "from 5-turn batches"), "vector-first",
         ("Retrieved facts", "v3: up to 12 triples", "v4: triples + source turns")),
    ]
    for y_mid, label, palette_key, fill, ingest, arrow_label, context in rows:
        ax.text(1, y_mid, label, va="center", ha="left", fontsize=8.8, fontweight="bold", color=PRESENTATION_PALETTE[palette_key])
        draw_box(14, y_mid - 7, 18, 14, *ingest, fill)
        draw_arrow(32, y_mid, 45, y_mid, arrow_label)
        draw_box(45, y_mid - 7, 20, 14, *context, fill)
        draw_arrow(65, y_mid, 78, y_mid)

    reader = patches.FancyBboxPatch(
        (78, 28), 20, 56,
        boxstyle="round,pad=0.8,rounding_size=1.5",
        facecolor="#EAECEE",
        edgecolor="#4F5D65",
        linewidth=1.1,
        alpha=0.92,
        zorder=2,
    )
    ax.add_patch(reader)
    ax.text(88, 70, "SLM reader", ha="center", va="center", fontsize=9.2, fontweight="bold", color="#1C2833", zorder=3)
    ax.text(88, 56, "same answer prompt\nfor every pathway\n\nGemma 4 E2B · E4B\nQwen3.5 0.8B · 2B · 4B\nQ4_K_M weights",
            ha="center", va="center", fontsize=7.8, color="#4A5568", linespacing=1.35, zorder=3)

    # (B) and (C): the same pipeline runs on both platforms; answers are judged on the workstation.
    ax.plot([1, 101], [23.5, 23.5], color="#BDC3C7", linewidth=0.8)
    ax.text(1, 11, "One pipeline,\ntwo platforms", va="center", ha="left", fontsize=8.8, fontweight="bold", color="#34495E")
    draw_box(14, 4, 38, 14, "(B) Workstation", "Ollama · all five readers", "LoCoMo n=200 · LongMemEval-S n=60: accuracy", "#F4F6F7")
    draw_box(56, 4, 42, 14, "(C) Phone: Samsung Galaxy S23 Ultra", "llama.cpp in Termux · LoCoMo n=20 slice",
             "answers plus energy, latency, RAM and SoC temperature", "#F4F6F7")
    ax.text(56, -1.2, "A llama3.1:70b judge on the workstation scores the answers from both platforms; "
                      "the two platforms' accuracies are reported separately.",
            ha="center", va="center", fontsize=7.6, color="#59646B")

    saved = save_figure(fig, "fig13_architecture_diagram", output_dirs)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Visual A and D in paper, presentation, or both formats.")
    parser.add_argument("--style", choices=["paper", "presentation", "both"], default="both")
    args = parser.parse_args()

    generated: list[Path] = []
    if args.style in ("paper", "both"):
        # Paper targets
        out_dirs = [THESIS_DIR]
        generated.extend(plot_category_taxonomy_dotplot(out_dirs, is_primary=True))
        generated.extend(plot_category_taxonomy_heatmap(out_dirs))
        generated.extend(plot_category_taxonomy_bars(out_dirs, is_primary=False))
        generated.extend(plot_architecture_diagram(out_dirs, style="paper"))
        generated.extend(plot_lookup_methods(out_dirs))

    if args.style in ("presentation", "both"):
        # Presentation targets
        out_dirs = [PRESENTATION_DIR, THESIS_DIR / "presentation"]
        generated.extend(plot_category_taxonomy_bars(out_dirs, is_primary=True))
        generated.extend(plot_architecture_diagram(out_dirs, style="presentation"))

    print(f"Generated {len(generated)} files for style '{args.style}'")


if __name__ == "__main__":
    main()
