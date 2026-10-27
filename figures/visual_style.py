#!/usr/bin/env python3
r"""Shared visual styling module supporting dual formats:
1. Academic Publication Style ('paper'):
   - Clean, minimal-ink, de-spined axes.
   - NO embedded suptitle banners (avoiding duplicate titles with LaTeX \caption{}).
   - Standardized subpanel tags: (a), (b), (c).
   - Subtle gridlines and unobscured confidence intervals / error whiskers.
2. Presentation Style ('presentation'):
   - Bold takeaway suptitles and prominent figure titles.
   - High-contrast callout boxes and slide-ready annotation cards.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence
import matplotlib
import matplotlib.pyplot as plt

# Colorblind-safe, publication-grade palettes
PAPER_PALETTE = {
    "Full context": "#D55E00",   # Vermilion / Rust
    "GraphRAG v3":  "#009E73",   # Bluish Green / Teal
    "GraphRAG v4":  "#E69F00",   # Amber / Orange
    "Vector RAG":   "#0072B2",   # Navy / Deep Blue
    "Thinking":     "#7B3FA1",   # Deep Purple
    "Base":         "#475569",   # Slate Grey
    "Gemma":        "#0072B2",   # Deep Blue
    "Qwen":         "#E69F00",   # Amber
    "pass":         "#10B981",   # Emerald
    "fail":         "#EF4444",   # Crimson
    "warn":         "#F59E0B",   # Amber
}

PRESENTATION_PALETTE = {
    "Full context": "#CC3311",   # Vermilion red
    "GraphRAG v3":  "#009988",   # Teal
    "GraphRAG v4":  "#EE7733",   # Orange
    "Vector RAG":   "#0077BB",   # Strong blue
    "Thinking":     "#7B3FA1",   # Deep purple
    "Base":         "#59646B",   # Slate grey
    "Gemma":        "#0072B2",   # Blue
    "Qwen":         "#E69F00",   # Amber
}

NUMBER_LABEL_BOX_PRESENTATION = {
    "boxstyle": "round,pad=0.14",
    "facecolor": "white",
    "edgecolor": "none",
    "alpha": 0.88,
}

NUMBER_LABEL_BOX_PAPER = {
    "boxstyle": "round,pad=0.10",
    "facecolor": "white",
    "edgecolor": "#CBD5E1",
    "linewidth": 0.5,
    "alpha": 0.80,
}


# Thesis text block (A4, 2.5 cm margins). Paper figures are drawn at this width so that they are
# included at \textwidth with no scaling and every point size below is the printed size.
PAPER_WIDTH_IN = 6.3


def paper_size(height_in: float) -> tuple[float, float]:
    """Figure size for a thesis figure: the text width, and a height chosen per figure."""
    return (PAPER_WIDTH_IN, height_in)


def configure_paper_style() -> None:
    """Configure Matplotlib for thesis figures drawn at their printed size."""
    # Start from defaults: rcParams.update is cumulative, so without this a style applied earlier
    # in the same process leaks its unset keys into this one.
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Helvetica Neue", "Arial"],
        "mathtext.fontset": "dejavusans",
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.titleweight": "normal",
        "axes.labelsize": 9.0,
        "axes.labelweight": "normal",
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "figure.titlesize": 9.0,
        "figure.titleweight": "normal",
        "lines.linewidth": 1.2,
        "lines.markersize": 4.5,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#334155",
        "xtick.color": "#334155",
        "ytick.color": "#334155",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "grid.color": "#E2E8F0",
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def configure_presentation_style() -> None:
    """Configure Matplotlib for bold, slide-ready presentation figures."""
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica Neue", "Arial"],
        "mathtext.fontset": "dejavusans",
        "font.size": 10.0,
        "figure.titlesize": 13.5,
        "figure.titleweight": "bold",
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.0,
        "axes.labelweight": "normal",
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 9.0,
        "lines.linewidth": 1.8,
        "lines.markersize": 7.0,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#71787E",
        "grid.color": "#E2E5E8",
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def savefig_retry(fig: plt.Figure, path: Path, **kwargs: object) -> None:
    """fig.savefig, retried briefly. On this Windows machine antivirus and OneDrive sync can hold a
    file for a moment after it is written, which surfaces as OSError errno 22 or 13 when the next
    run rewrites it. Any other error, or a lock that outlasts the retries, is raised."""
    for attempt in range(6):
        try:
            fig.savefig(path, **kwargs)
            return
        except OSError as exc:
            if exc.errno not in (13, 22) or attempt == 5:
                raise
            time.sleep(0.5 * (attempt + 1))


def save_figure(
    fig: plt.Figure,
    stem: str,
    output_dirs: Path | Sequence[Path],
    formats: Sequence[str] = ("png", "pdf", "svg"),
    dpi: int = 300,
    close: bool = True,
) -> list[Path]:
    """Save a figure across requested directory targets and formats."""
    if isinstance(output_dirs, Path):
        output_dirs = [output_dirs]

    saved_paths: list[Path] = []
    for out_dir in output_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            dest = out_dir / f"{stem}.{fmt}"
            savefig_retry(fig, dest, dpi=dpi if fmt == "png" else None)
            saved_paths.append(dest)

    if close:
        plt.close(fig)

    return saved_paths
