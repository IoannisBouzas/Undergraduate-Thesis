#!/usr/bin/env python3
"""Generate publication-grade Venn diagram and case analysis visual for the SPE Oracle Story.
Matches the layout, typography, and styling of top-tier academic systems/NLP papers (e.g. OSDI/NeurIPS/ACL).
Outputs both PDF and high-res PNG to results/thesis_visuals/.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_spe_oracle_visual(output_dir="results/thesis_visuals"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Set font to clean system serif/sans
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })

    # Total figure size: 13.6 in wide x 5.8 in high
    fig = plt.figure(figsize=(13.6, 5.8), facecolor="white")

    # Layout: Left = Venn (width=0.38), Right = Case Card (width=0.57)
    # Give ample margin at the bottom for clean spacing
    ax_venn = fig.add_axes([0.01, 0.08, 0.38, 0.86])
    ax_venn.set_aspect("equal")
    ax_venn.axis("off")
    ax_venn.set_xlim(-1.45, 1.45)
    ax_venn.set_ylim(-1.35, 1.35)

    ax_case = fig.add_axes([0.41, 0.08, 0.57, 0.86])
    ax_case.axis("off")
    ax_case.set_xlim(0, 1)
    ax_case.set_ylim(0, 1)

    # --- LEFT PANEL: VENN DIAGRAM ---
    r = 0.80
    c_v3 = (-0.38, 0.28)      # Top-Left: GraphRAG v3
    c_v4 = (0.38, 0.28)       # Top-Right: GraphRAG v4
    c_spe = (0.00, -0.34)     # Bottom: SPE-2

    # Palettes matching thesis visual identity
    col_v3 = "#854D0E"    # Deep Amber / Brown
    col_v4 = "#1D4ED8"    # Royal Blue
    col_spe = "#15803D"   # Deep Forest Green

    # Soft transparent fills
    circ_v3 = patches.Circle(c_v3, r, edgecolor=col_v3, facecolor="#FEF3C7", alpha=0.35, linewidth=2.0, zorder=1)
    circ_v4 = patches.Circle(c_v4, r, edgecolor=col_v4, facecolor="#DBEAFE", alpha=0.35, linewidth=2.0, zorder=1)
    circ_spe = patches.Circle(c_spe, r, edgecolor=col_spe, facecolor="#DCFCE7", alpha=0.35, linewidth=2.0, zorder=1)

    ax_venn.add_patch(circ_v3)
    ax_venn.add_patch(circ_v4)
    ax_venn.add_patch(circ_spe)

    # Clean borders redraw
    ax_venn.add_patch(patches.Circle(c_v3, r, edgecolor=col_v3, facecolor="none", linewidth=2.0, zorder=2))
    ax_venn.add_patch(patches.Circle(c_v4, r, edgecolor=col_v4, facecolor="none", linewidth=2.0, zorder=2))
    ax_venn.add_patch(patches.Circle(c_spe, r, edgecolor=col_spe, facecolor="none", linewidth=2.0, zorder=2))

    # Set Labels outside circles
    ax_venn.text(-0.95, 1.05, "GraphRAG v3\n(Triples)", color=col_v3, fontsize=13, fontweight="bold",
                 ha="center", va="bottom", linespacing=1.1)
    ax_venn.text(0.95, 1.05, "GraphRAG v4\n(+ Sources)", color=col_v4, fontsize=13, fontweight="bold",
                 ha="center", va="bottom", linespacing=1.1)
    ax_venn.text(0.00, -1.22, "+ SPE-2 (Policy)", color=col_spe, fontsize=13, fontweight="bold",
                 ha="center", va="top")

    # Overlap Numbers (200 LoCoMo questions, Gemma 4 E2B)
    # v3 only
    ax_venn.text(-0.70, 0.52, "1", fontsize=15, fontweight="bold", color="#1E293B", ha="center", va="center")
    # v4 only (15 source restoration wins missed by SPE)
    ax_venn.text(0.70, 0.52, "15", fontsize=16, fontweight="bold", color=col_v4, ha="center", va="center")
    # SPE only (8 wins from larger triple budget B=370)
    ax_venn.text(0.00, -0.74, "8", fontsize=15, fontweight="bold", color=col_spe, ha="center", va="center")
    # v3 & v4 only
    ax_venn.text(0.00, 0.54, "8", fontsize=15, fontweight="bold", color="#1E293B", ha="center", va="center")
    # v3 & SPE only (3 compact triple wins preserved)
    ax_venn.text(-0.35, -0.12, "3", fontsize=15, fontweight="bold", color=col_v3, ha="center", va="center")
    # v4 & SPE only (12 source expansion wins captured by SPE)
    ax_venn.text(0.35, -0.12, "12", fontsize=15, fontweight="bold", color=col_v4, ha="center", va="center")
    # All three (31 common successes)
    ax_venn.text(0.00, 0.12, "31", fontsize=20, fontweight="bold", color="#0F172A", ha="center", va="center")

    # --- RIGHT PANEL: CASE STUDY CARD ---
    # Outer white rounded card with green outline (matching uploaded style)
    box_rect = patches.FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        edgecolor="#15803D",
        facecolor="#FFFFFF",
        linewidth=1.6,
        zorder=1
    )
    ax_case.add_patch(box_rect)

    # Column alignment coordinates
    col_lbl = 0.035
    col_sym = 0.200
    col_txt = 0.235

    # ================= CASE 1 =================
    y = 0.93
    ax_case.text(col_lbl, y, "Case 1. Factual qualifier recovery (LoCoMo conv-48, Temporal)",
                 fontsize=11.2, fontweight="bold", color="#15803D", va="top")
    
    y -= 0.058
    ax_case.text(col_lbl, y, 'Task: "What kind of project was Jolene working on in the beginning of January 2023?"',
                 fontsize=9.8, color="#0F172A", va="top")

    y -= 0.052
    ax_case.text(col_lbl, y, 'v3 (Triples):', fontsize=9.6, fontweight="bold", color="#854D0E", va="top")
    ax_case.text(col_sym, y, '✗', fontsize=11, fontweight="bold", color="#DC2626", fontfamily="DejaVu Sans", va="top")
    ax_case.text(col_txt, y, '"A big project"  [domain qualifier lost during triple extraction]',
                 fontsize=9.5, color="#1E293B", va="top")

    y -= 0.048
    ax_case.text(col_lbl, y, 'v4 (+Sources):', fontsize=9.6, fontweight="bold", color="#1D4ED8", va="top")
    ax_case.text(col_sym, y, '✓', fontsize=11, fontweight="bold", color="#16A34A", fontfamily="DejaVu Sans", va="top")
    ax_case.text(col_txt, y, '"electrical engineering project"  [verbatim source dialogue restored]',
                 fontsize=9.5, color="#1E293B", va="top")

    y -= 0.048
    ax_case.text(col_lbl, y, 'SPE-2:', fontsize=9.6, fontweight="bold", color="#475569", va="top")
    ax_case.text(col_sym, y, '✗', fontsize=11, fontweight="bold", color="#DC2626", fontfamily="DejaVu Sans", va="top")
    ax_case.text(col_txt, y, '"A big project"  [greedy selector retained only triples under budget]',
                 fontsize=9.5, color="#475569", va="top")

    y -= 0.055
    why_1 = (
        'Why: Triple extraction abstracted away the technical qualifier ("electrical engineering"),\n'
        'retaining only the coarse relation. Restoring verbatim dialogue recovered the required factual fidelity.'
    )
    ax_case.text(col_lbl, y, why_1, fontsize=9.2, color="#334155", va="top", linespacing=1.2)

    # Dividing line
    line_y = 0.50
    line = patches.ConnectionPatch(
        (0.02, line_y), (0.98, line_y),
        coordsA="data", coordsB="data",
        axesA=ax_case, axesB=ax_case,
        color="#E2E8F0", linewidth=1.1, linestyle="-"
    )
    ax_case.add_patch(line)

    # ================= CASE 2 =================
    y = line_y - 0.045
    ax_case.text(col_lbl, y, "Case 2. Distractor noise avoidance (LoCoMo conv-44, Single-hop)",
                 fontsize=11.2, fontweight="bold", color="#15803D", va="top")

    y -= 0.058
    ax_case.text(col_lbl, y, 'Task: "What kind of flowers does Audrey have a tattoo of?"',
                 fontsize=9.8, color="#0F172A", va="top")

    y -= 0.052
    ax_case.text(col_lbl, y, 'v3 (Triples):', fontsize=9.6, fontweight="bold", color="#854D0E", va="top")
    ax_case.text(col_sym, y, '✓', fontsize=11, fontweight="bold", color="#16A34A", fontfamily="DejaVu Sans", va="top")
    ax_case.text(col_txt, y, '"sunflowers"  [isolated the exact entity-attribute relation cleanly]',
                 fontsize=9.5, color="#1E293B", va="top")

    y -= 0.048
    ax_case.text(col_lbl, y, 'v4 (+Sources):', fontsize=9.6, fontweight="bold", color="#1D4ED8", va="top")
    ax_case.text(col_sym, y, '✗', fontsize=11, fontweight="bold", color="#DC2626", fontfamily="DejaVu Sans", va="top")
    ax_case.text(col_txt, y, '"Peruvian Lilies"  [reader misled by neighboring turn context]',
                 fontsize=9.5, color="#1E293B", va="top")

    y -= 0.048
    ax_case.text(col_lbl, y, 'SPE-2:', fontsize=9.6, fontweight="bold", color="#475569", va="top")
    ax_case.text(col_sym, y, '✓', fontsize=11, fontweight="bold", color="#16A34A", fontfamily="DejaVu Sans", va="top")
    ax_case.text(col_txt, y, '"sunflowers"  [benefited from compact triple representation]',
                 fontsize=9.5, color="#15803D", va="top")

    y -= 0.055
    why_2 = (
        'Why: Expand-all introduced dialogue mentioning Peruvian Lilies in another conversational context,\n'
        'confusing the 2B reader. Compact triples shielded the reader from extraneous distractor text.'
    )
    ax_case.text(col_lbl, y, why_2, fontsize=9.2, color="#334155", va="top", linespacing=1.2)

    # Save versions: with and without caption
    # 1. Main visual for LaTeX \caption{} use
    pdf_path = out_path / "fig_spe_oracle_case_analysis.pdf"
    png_path = out_path / "fig_spe_oracle_case_analysis.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"Generated clean visual:\n  {pdf_path}\n  {png_path}")

    # 2. Standalone presentation version with caption included at bottom
    fig_caption = (
        "Figure 7.X Case analysis of GraphRAG v3, GraphRAG v4, and SPE-2 on LoCoMo (n=200, Gemma 4 E2B). "
        "Left: success-set Venn diagram across the three memory representations, "
        "where the colored numbers mark the two oracle opportunity patterns: 27 questions gained by source restoration (15 unique to v4 + 12 shared with SPE-2) "
        "and 4 questions preserved only by compact triples (1 unique to v3 + 3 shared with SPE-2). "
        "The post hoc oracle bound achieves 35.0% accuracy (70/200) at 560.3 prompt tokens (31.1% fewer tokens than v4). "
        "Right: sampled dialogue cases illustrating each trade-off pattern."
    )
    fig.text(0.01, 0.01, fig_caption, wrap=True, fontsize=8.8, color="#0F172A", linespacing=1.2)
    pdf_cap_path = out_path / "fig_spe_oracle_case_analysis_captioned.pdf"
    png_cap_path = out_path / "fig_spe_oracle_case_analysis_captioned.png"
    plt.savefig(pdf_cap_path, bbox_inches="tight")
    plt.savefig(png_cap_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Generated captioned visual:\n  {pdf_cap_path}\n  {png_cap_path}")

if __name__ == "__main__":
    create_spe_oracle_visual()
