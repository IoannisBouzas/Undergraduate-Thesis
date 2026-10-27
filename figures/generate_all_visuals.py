#!/usr/bin/env python3
"""Master figure pipeline: generate all thesis figures in paper and presentation styles.

Formats:
  1. Academic Paper Style ('paper'):
     - Minimalist, de-spined publication style for the thesis manuscript.
     - No in-canvas suptitles / banners (captions are handled by LaTeX \\caption{}).
     - Standardized subpanel labels: (a), (b), (c).
     - Subtler confidence intervals and unobscured markers.
     - Written to results/thesis_visuals/ as the primary drop-in targets for LaTeX.

  2. Defense Presentation Style ('presentation'):
     - High-contrast, slide-ready styling for talks.
     - Prominent takeaway suptitles and callout cards.
     - Written to results/presentation_visuals/ and results/thesis_visuals/presentation/.

  3. Alternative Encodings:
     - Figure 14: Both Cleveland Dot Plot (fig14_category_taxonomy_dotplot) and Heatmap Matrix (fig14_category_taxonomy_heatmap).
     - Figure 01: Both Grouped Vertical Bars (fig01_strategy_landscape_bars) and Whisker Scatter (fig01_strategy_landscape_whiskers).
     - Figure 15: Clean 2-panel diagnostic with confusion matrix.
     - Figure 17: Clean multi-turn break-even curves.
     - Figure 19: Clean 2-panel model family scaling curves.

Usage:
    python -B -m figures.generate_all_visuals [--style {paper,presentation,both}]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_pipeline(style: str = "both") -> None:
    scripts = [
        ("make_benchmark_anatomy_visual.py", "Figure 00: Benchmark Anatomy & Multi-Session Timeline"),
        ("make_thesis_visuals.py", "Figures 01–12: Core Thesis Visuals (Strategy Grid, Phone Workloads)"),
        ("make_architecture_and_taxonomy_figures.py", "Figures 13–14: 3-Way Architecture & Failure Taxonomy"),
        ("make_extended_figures.py", "Figures 15–21: Parity Gate, KV RAM Wall, Amortization, Scaling"),
    ]

    print(f"\n{'='*75}")
    print(f"  DUAL-FORMAT VISUAL GENERATION PIPELINE (Style: {style.upper()})")
    print(f"{'='*75}\n")

    start_time = time.time()
    for script, description in scripts:
        print(f"--> Running {script} ({description})...")
        cmd = [sys.executable, "-B", "-m", f"figures.{script[:-3]}", "--style", style]
        res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"FAILED: {script}")
            print(res.stderr)
            sys.exit(res.returncode)
        else:
            out_lines = [line for line in res.stdout.strip().split("\n") if line.strip()]
            for line in out_lines[-3:]:
                print(f"    {line}")

    elapsed = time.time() - start_time
    print(f"\n{'='*75}")
    print(f"SUCCESS: All visual packages generated in {elapsed:.1f}s")
    print(f"  Primary Thesis (Paper):       {PROJECT_ROOT / 'results' / 'thesis_visuals'}")
    print(f"  Slides (Presentation): {PROJECT_ROOT / 'results' / 'presentation_visuals'}")
    print(f"{'='*75}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all thesis figures in paper, presentation, or both formats.")
    parser.add_argument("--style", choices=["paper", "presentation", "both"], default="both")
    args = parser.parse_args()
    run_pipeline(args.style)


if __name__ == "__main__":
    main()
