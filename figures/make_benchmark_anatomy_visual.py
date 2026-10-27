#!/usr/bin/env python3
"""Generates the Benchmark Anatomy & Multi-Session Timeline Visual for the thesis:

Figure: fig00_benchmark_anatomy (.png, .pdf, .svg)
- Panel A: one real LoCoMo history (conv-26) and where the evidence for one of its questions sits
- Panel B: benchmark facts for LoCoMo QA and LongMemEval-S, computed from the evaluated data files

Supports dual styles:
  - Paper: Clean publication layout, no redundant canvas suptitle.
  - Presentation: Bold banner suptitle.

Outputs saved to:
  - results/thesis_visuals/
  - results/presentation_visuals/
"""

from __future__ import annotations

import argparse
import json
import textwrap
from datetime import datetime
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from figures.visual_style import PAPER_WIDTH_IN, save_figure

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THESIS_VIS_DIR = PROJECT_ROOT / "results" / "thesis_visuals"
PRESENTATION_DIR = PROJECT_ROOT / "results" / "presentation_visuals"

# Typography & global styling
plt.rcParams.update({
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica Neue", "Arial"],
    "font.family": "sans-serif",
    "figure.titlesize": 13.5,
    "figure.titleweight": "bold",
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

COLORS = {
    "fullcontext": "#CC3311",   # Red
    "vectorrag": "#0077BB",     # Blue
    "graphrag": "#009988",      # Teal
    "session_bg": "#F4F6F7",
    "needle_init": "#F5EEF8",
    "needle_update": "#D5F5E3",
    "query_bg": "#FCF3CF",
    "card_bg": "#FAFAFA",
    "border": "#7F8C8D",
}

# Paper prints at the 6.7 in text width: a narrower canvas with larger type keeps labels >= ~6.5 pt.
_FONT_SCALE = 1.0


def fs(size: float) -> float:
    return size * _FONT_SCALE


NUMBER_LABEL_BOX = {
    "boxstyle": "round,pad=0.14",
    "facecolor": "white",
    "edgecolor": "none",
    "alpha": 0.88,
}


def draw_box(ax, x, y, w, h, title, subtitle, footer, facecolor, edgecolor="#5D6D7E", title_color="#1C2833", lw=1.2, hatch=""):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.6,rounding_size=1.2",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        hatch=hatch,
        alpha=0.95,
        zorder=2,
    )
    ax.add_patch(rect)
    if title:
        ax.text(x + w / 2, y + h * 0.72, title, ha="center", va="center", fontsize=fs(9.2), fontweight="bold", color=title_color, zorder=3)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.44, subtitle, ha="center", va="center", fontsize=fs(7.8), color="#34495E", zorder=3)
    if footer:
        ax.text(x + w / 2, y + h * 0.18, footer, ha="center", va="center", fontsize=fs(7.6), fontweight="bold", color="#1B4F72", zorder=3)


def draw_arrow(ax, x1, y1, x2, y2, label="", color="#5D6D7E"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.5, mutation_scale=12),
        zorder=4,
    )
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.6, label, ha="center", va="bottom", fontsize=fs(7.5), color="#2C3E50", fontweight="bold", bbox=NUMBER_LABEL_BOX, zorder=5)


EXAMPLE_CONV = "conv-26"
EXAMPLE_QUESTION = "What did Caroline research?"
FC_RUN = "lap_locomo200_e2b_fullcontext"  # its ctx_meta gives the turns the 4,096-token window kept


def _session_date(stamp: str) -> datetime:
    # LoCoMo stamps look like "1:56 pm on 8 May, 2023"
    return datetime.strptime(stamp.split(" on ")[1].replace(",", ""), "%d %B %Y")


def _fmt(day: datetime) -> str:
    return f"{day.day} {day:%B %Y}"


def _example() -> dict:
    """Session layout, evidence turn and full-context window for the example question."""
    conv = next(c for c in json.loads((PROJECT_ROOT / "locomo10.json").read_text(encoding="utf-8")) if c["sample_id"] == EXAMPLE_CONV)
    talk = conv["conversation"]
    sessions, start = [], 1
    for idx in sorted(int(k.split("_")[1]) for k in talk if k.startswith("session_") and not k.endswith("date_time")):
        turns = talk[f"session_{idx}"]
        sessions.append({"idx": idx, "date": _session_date(talk[f"session_{idx}_date_time"]), "first": start, "last": start + len(turns) - 1, "turns": turns})
        start += len(turns)
    qa = next(q for q in conv["qa"] if q["question"] == EXAMPLE_QUESTION)
    (evidence_id,) = qa["evidence"]
    ev_session = next(s for s in sessions if any(t["dia_id"] == evidence_id for t in s["turns"]))
    ev_offset = next(i for i, t in enumerate(ev_session["turns"]) if t["dia_id"] == evidence_id)
    ev_turn = ev_session["turns"][ev_offset]
    with open(PROJECT_ROOT / "results" / FC_RUN / "judged.jsonl", encoding="utf-8") as fh:
        kept = next(json.loads(line)["ctx_meta"]["kept_turns"] for line in fh if json.loads(line)["question"] == EXAMPLE_QUESTION)
    return {
        "sessions": sessions,
        "total": start - 1,
        "qa": qa,
        "evidence_id": evidence_id,
        "evidence_turn": ev_session["first"] + ev_offset,
        "evidence_session": ev_session,
        "evidence_speaker": ev_turn["speaker"],
        "evidence_text": ev_turn["text"].replace("�", "—"),
        "fc_kept": kept,
    }


def _corpus_stats() -> dict:
    locomo = json.loads((PROJECT_ROOT / "locomo10.json").read_text(encoding="utf-8"))
    turns, sessions, words = [], [], []
    for conv in locomo:
        talk = conv["conversation"]
        keys = [k for k in talk if k.startswith("session_") and not k.endswith("date_time")]
        turns.append(sum(len(talk[k]) for k in keys))
        sessions.append(len(keys))
        words.append(sum(len(t["text"].split()) for k in keys for t in talk[k]))
    lme = json.loads((PROJECT_ROOT / "longmemeval_s_strat60.json").read_text(encoding="utf-8"))
    lme_turns = [sum(len(s) for s in item["haystack_sessions"]) for item in lme]
    lme_sessions = [len(item["haystack_sessions"]) for item in lme]
    lme_words = [sum(len(m["content"].split()) for s in item["haystack_sessions"] for m in s) for item in lme]
    return {
        "n_conv": len(locomo), "turns": turns, "sessions": sessions, "words": words,
        "n_lme": len(lme), "lme_turns": lme_turns, "lme_sessions": lme_sessions, "lme_words": lme_words,
    }


def _plot_benchmark_anatomy_paper() -> plt.Figure:
    """Thesis fig00, drawn at the 6.3 in text width so it prints 1:1 (no text below 7.5 pt).

    Only the conv-26 history is drawn. The per-strategy boxes moved to fig22 (lookup methods) and
    the corpus facts to a LaTeX table in Chapter 3, both computed by _corpus_stats().
    """
    global _FONT_SCALE
    _FONT_SCALE = 1.0
    fig = plt.figure(figsize=(PAPER_WIDTH_IN, 3.0))
    ax = fig.add_axes([0.005, 0.005, 0.99, 0.99])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ex = _example()
    sessions, total = ex["sessions"], ex["total"]
    first, ev, last = sessions[0], ex["evidence_session"], sessions[-1]
    middle = [s for s in sessions if first["idx"] < s["idx"] < last["idx"] and s is not ev]
    qa = ex["qa"]

    def gap(a, b):
        return f"+{(b['date'] - a['date']).days} days"

    ax.text(1, 95, f"LoCoMo {EXAMPLE_CONV}: {len(sessions)} sessions, {total} turns", fontsize=9.5, fontweight="bold", color="#1C2833", va="center")
    ax.text(1, 84.5, f"Question asked after the last session: “{qa['question']}”\ngold answer: “{qa['answer']}”  ·  evidence: one turn ({ex['evidence_id']})",
            fontsize=8.0, color="#59646B", va="center", linespacing=1.3)
    quote = textwrap.fill(textwrap.shorten(ex["evidence_text"], width=44, placeholder="…"), width=24)
    by, bh = 30, 42
    draw_box(ax, 1.5, by, 20, bh, f"Session {first['idx']}\n{_fmt(first['date'])}", f"turns {first['first']}–{first['last']}", "other conversation", "#EBEDEF", edgecolor="#95A5A6", title_color="#2C3E50")
    draw_arrow(ax, 22.2, by + bh / 2, 26.3, by + bh / 2, gap(first, ev))
    draw_box(ax, 27, by, 22, bh, f"Session {ev['idx']}\n{_fmt(ev['date'])}", f"turn {ex['evidence_turn']}, {ex['evidence_speaker']}:\n“{quote}”",
             "evidence turn", "#FADBD8", edgecolor="#E74C3C", title_color="#900C3F")
    draw_arrow(ax, 49.7, by + bh / 2, 53.8, by + bh / 2, gap(ev, middle[0]))
    draw_box(ax, 54.5, by, 21, bh, f"Sessions {middle[0]['idx']}–{middle[-1]['idx']}",
             f"{_fmt(middle[0]['date'])} –\n{_fmt(middle[-1]['date'])}\nturns {middle[0]['first']}–{middle[-1]['last']}",
             f"{middle[-1]['last'] - middle[0]['first'] + 1} other turns", "#EBEDEF", edgecolor="#95A5A6", title_color="#2C3E50")
    draw_arrow(ax, 76.2, by + bh / 2, 80.3, by + bh / 2, gap(middle[-1], last))
    draw_box(ax, 81, by, 17.5, bh, f"Session {last['idx']}\n{_fmt(last['date'])}", f"turns {last['first']}–{last['last']},\nthen the question", "question asked", "#FCF3CF", edgecolor="#F39C12", title_color="#7D6608")

    x0, x1, sy = 1.5, 98.5, 10.0

    def tx(turn: float) -> float:
        return x0 + (turn - 1) / total * (x1 - x0)

    ax.add_patch(patches.Rectangle((x0, sy), x1 - x0, 6, facecolor="#EBEDEF", edgecolor="#95A5A6", lw=0.8, zorder=2))
    fc_first = total - ex["fc_kept"] + 1
    ax.add_patch(patches.Rectangle((tx(fc_first), sy), x1 - tx(fc_first), 6, facecolor="#F5B7B1", edgecolor="none", zorder=3))
    for s in sessions[1:]:
        ax.plot([tx(s["first"])] * 2, [sy, sy + 6], color="white", lw=1.0, zorder=4)
    ax.plot([tx(ex["evidence_turn"])] * 2, [sy - 1.5, sy + 7.5], color="#C0392B", lw=2.2, zorder=5)
    ax.text(x0, sy + 9.5, f"All {total} turns to scale; white ticks mark session boundaries", fontsize=7.8, color="#59646B", ha="left", va="center")
    ax.text(tx(ex["evidence_turn"]), sy - 5.5, f"evidence: turn {ex['evidence_turn']}", fontsize=7.8, color="#C0392B", ha="left", va="center", fontweight="bold")
    ax.text(x1, sy - 5.5, f"full-context window: last {ex['fc_kept']} turns ({fc_first}–{total})", fontsize=7.8, color="#A93226", ha="right", va="center", fontweight="bold")
    return fig


def plot_benchmark_anatomy(style: str = "paper") -> plt.Figure:
    if style == "paper":
        return _plot_benchmark_anatomy_paper()
    global _FONT_SCALE
    paper = style == "paper"
    _FONT_SCALE = 1.08 if paper else 1.0
    # Paper: 8.6 in wide, printed at 6.7 in (scale 0.78). Presentation keeps the 13 in slide canvas.
    figsize = (8.6, 10.6) if paper else (13.0, 9.0)
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [1.2, 1.0]})
    sep = chr(10) if paper else "  ·  "  # paper breaks the question line in two

    for ax in (ax_top, ax_bottom):
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis("off")

    ex = _example()
    sessions, total = ex["sessions"], ex["total"]
    first, ev, last = sessions[0], ex["evidence_session"], sessions[-1]
    middle = [s for s in sessions if first["idx"] < s["idx"] < last["idx"] and s is not ev]
    qa = ex["qa"]

    # Panel A: one real LoCoMo history and where the evidence for one of its questions sits
    ax_top.text(2, 95, f"(a) One LoCoMo history: {EXAMPLE_CONV}, {len(sessions)} sessions, {total} turns", fontsize=fs(11.5), fontweight="bold", color="#1C2833")
    ax_top.text(2, 86.5 if paper else 89.5, f"Question asked after the last session: “{qa['question']}”{sep}gold answer: “{qa['answer']}”  ·  evidence: one turn ({ex['evidence_id']})", fontsize=fs(8.6), color="#59646B", va="center" if paper else "baseline")

    def gap(a, b):
        return f"+{(b['date'] - a['date']).days} days"

    # Session titles break after the session number on paper, where each box is about 1.8 in wide.
    tsep = "\n" if paper else " · "
    evidence_quote = textwrap.fill(textwrap.shorten(ex["evidence_text"], width=64, placeholder="…"), width=22 if paper else 34)
    by = 48 if paper else 52  # session-box bottom; paper needs room for the two-line question
    draw_box(ax_top, 2, by, 21, 30, f"Session {first['idx']}{tsep}{_fmt(first['date'])}", f"turns {first['first']}–{first['last']}", "other conversation", "#EBEDEF", edgecolor="#95A5A6", title_color="#2C3E50")
    draw_arrow(ax_top, 23, by + 15, 27, by + 15, gap(first, ev))
    draw_box(ax_top, 27, by, 21, 30, f"Session {ev['idx']}{tsep}{_fmt(ev['date'])}",
             f"turn {ex['evidence_turn']} ({ex['evidence_id']}), {ex['evidence_speaker']}:\n“{evidence_quote}”",
             "Evidence turn" if paper else "Evidence turn (the needle)", "#FADBD8", edgecolor="#E74C3C", title_color="#900C3F")
    draw_arrow(ax_top, 48, by + 15, 52, by + 15, gap(ev, middle[0]))
    draw_box(ax_top, 52, by, 22, 30, f"Sessions {middle[0]['idx']}–{middle[-1]['idx']}",
             f"{_fmt(middle[0]['date'])} –{chr(10) if paper else ' '}{_fmt(middle[-1]['date'])}\nturns {middle[0]['first']}–{middle[-1]['last']}",
             f"{middle[-1]['last'] - middle[0]['first'] + 1} other turns" if paper else f"{middle[-1]['last'] - middle[0]['first'] + 1} turns of other conversation",
             "#EBEDEF", edgecolor="#95A5A6", title_color="#2C3E50")
    draw_arrow(ax_top, 74, by + 15, 78, by + 15, gap(middle[-1], last))
    draw_box(ax_top, 78, by, 20, 30, f"Session {last['idx']}{tsep}{_fmt(last['date'])}",
             f"turns {last['first']}–{last['last']}\nthen the question{chr(10) if paper else ' '}is asked",
             f"Gold: {qa['answer']}" if paper else f"Gold answer: {qa['answer']}", "#FCF3CF", edgecolor="#F39C12", title_color="#7D6608")

    # To-scale strip of all turns: session boundaries, the evidence turn and the full-context window
    x0, x1 = 2.0, 98.0

    def tx(turn: float) -> float:
        return x0 + (turn - 1) / total * (x1 - x0)

    sy = 36 if paper else 41  # strip bottom; paper moves the lower rows down by five units
    ax_top.add_patch(patches.Rectangle((x0, sy), x1 - x0, 4, facecolor="#EBEDEF", edgecolor="#95A5A6", lw=0.8, zorder=2))
    fc_first = total - ex["fc_kept"] + 1
    ax_top.add_patch(patches.Rectangle((tx(fc_first), sy), x1 - tx(fc_first), 4, facecolor="#F5B7B1", edgecolor="none", zorder=3))
    for s in sessions[1:]:
        ax_top.plot([tx(s["first"])] * 2, [sy, sy + 4], color="white", lw=1.0, zorder=4)
    ax_top.plot([tx(ex["evidence_turn"])] * 2, [sy - 1, sy + 5], color="#C0392B", lw=2.4, zorder=5)
    ax_top.text(x0, sy + 6.3, f"All {total} turns to scale; white ticks mark session boundaries", fontsize=fs(7.8), color="#59646B", ha="left")
    ax_top.text(tx(ex["evidence_turn"]), sy - 3.4, f"evidence: turn {ex['evidence_turn']}", fontsize=fs(7.8), color="#C0392B", ha="center", fontweight="bold")
    ax_top.text(x1, sy - 3.4, f"full-context window: last {ex['fc_kept']} turns ({fc_first}–{total})", fontsize=fs(7.8), color="#A93226", ha="right", fontweight="bold")

    ax_top.text(2, sy - 9.5, "How each memory handles the history:", fontsize=fs(9.5), fontweight="bold", color="#1C2833")
    gy, gh = (1, 21) if paper else (3, 24)
    if paper:
        fc_text = f"all turns in time order;\noldest dropped to fit 4,096 tokens\n(here the last {ex['fc_kept']} of {total} remain)"
        vr_text = "each turn embedded once (Nomic v1.5);\nthe question embedded at query time;\nthe 10 most similar turns returned"
        gr_text = "the reader extracts (subject, relation,\nobject) triples from 5-turn batches;\nup to 12 facts (v4 adds source turns)"
        gr_foot = "turn-anchored facts, no traversal"
    else:
        fc_text = f"all turns in time order in the prompt;\noldest turns dropped to fit 4,096 tokens\n(here only the last {ex['fc_kept']} of {total} remain)"
        vr_text = "each turn embedded once (Nomic Embed v1.5);\nthe question is embedded at query time;\nthe 10 most similar turns are returned"
        gr_text = "the reader extracts (subject, relation, object)\ntriples from 5-turn batches; up to 12 facts\nare returned (v4 adds their source turns)"
        gr_foot = "turn-anchored facts, no graph traversal"
    draw_box(ax_top, 2, gy, 30, gh, "1. Full context", fc_text, "no retrieval step", "#FDEDEC", edgecolor=COLORS["fullcontext"], title_color=COLORS["fullcontext"])
    draw_box(ax_top, 35, gy, 30, gh, "2. Vector retrieval", vr_text, "dense turn retrieval", "#E8F8F5", edgecolor=COLORS["vectorrag"], title_color=COLORS["vectorrag"])
    draw_box(ax_top, 68, gy, 30, gh, "3. Graph memory", gr_text, gr_foot, "#FEF9E7", edgecolor=COLORS["graphrag"], title_color=COLORS["graphrag"])

    # Panel B: benchmark facts, computed from the evaluated files
    st = _corpus_stats()
    brk = "\n      " if paper else " "  # paper wraps long bullets onto an indented second line
    ax_bottom.text(2, 94, "(b) The two evaluated benchmarks", fontsize=fs(11.5), fontweight="bold", color="#1C2833")
    ax_bottom.text(2, 88, "Sizes computed from the evaluated data files; tokens depend on the reader's tokenizer and are not shown", fontsize=fs(8.6), color="#59646B")

    draw_box(ax_bottom, 2, 8, 47, 74, "", "", "", "#F8F9FA", edgecolor="#2980B9", lw=1.5)
    ax_bottom.text(25.5, 76, "LoCoMo (conversational QA)", ha="center", fontsize=fs(10.5), fontweight="bold", color="#1B4F72")
    locomo_bullets = [
        r"$\bf{•\ Source:}$ Maharana et al., ACL 2024",
        rf"$\bf{{•\ Structure:}}$ {st['n_conv']} conversations, {min(st['turns'])}–{max(st['turns'])} turns{brk}(mean {mean(st['turns']):.0f}), {min(st['sessions'])}–{max(st['sessions'])} sessions",
        rf"$\bf{{•\ Size:}}$ {min(st['words']) / 1000:.1f}k–{max(st['words']) / 1000:.1f}k words per conversation{brk}(mean {mean(st['words']) / 1000:.1f}k)",
        r"$\bf{•\ Task:}$ question answering only; category 5 excluded",
        rf"$\bf{{•\ Workstation:}}$ 200 questions,{brk}50 from each of categories 1–4",
        rf"$\bf{{•\ Phone:}}$ a 20-question slice;{brk}one to three sittings per cell",
    ]
    for idx, bullet in enumerate(locomo_bullets):
        ax_bottom.text(4.5, 68 - idx * 9.5, bullet, ha="left", va="center", fontsize=fs(8.2), color="#2C3E50")
    ax_bottom.text(25.5, 13, "Primary benchmark: strategies and phone measurements" if paper else "Primary benchmark: strategy comparison and phone measurements", ha="center", fontsize=fs(8.0), fontweight="bold", color="#2980B9")

    draw_box(ax_bottom, 51, 8, 47, 74, "", "", "", "#F8F9FA", edgecolor="#27AE60", lw=1.5)
    ax_bottom.text(74.5, 76, "LongMemEval-S (long-term memory)" if paper else "LongMemEval-S (long-term interactive memory)", ha="center", fontsize=fs(10.5), fontweight="bold", color="#145A32")
    lme_bullets = [
        rf"$\bf{{•\ Subset:}}$ {st['n_lme']} questions from the cleaned S release,{brk}stratified by type",
        rf"$\bf{{•\ Structure:}}$ {min(st['lme_turns'])}–{max(st['lme_turns'])} turns per history{brk}(mean {mean(st['lme_turns']):.0f}), about {mean(st['lme_sessions']):.0f} sessions",
        rf"$\bf{{•\ Size:}}$ about {mean(st['lme_words']) / 1000:.0f}k words per history",
        r"$\bf{•\ Six\ question\ types,}$ 10 questions each:",
        r"   single-session user  ·  single-session assistant",
        r"   single-session preference  ·  multi-session",
        r"   knowledge update  ·  temporal reasoning",
    ]
    lme_step = 8.4 if paper else 7.6
    for idx, bullet in enumerate(lme_bullets):
        ax_bottom.text(53.5, 69 - idx * lme_step if paper else 68 - idx * lme_step, bullet, ha="left", va="center", fontsize=fs(8.0), color="#2C3E50")
    ax_bottom.text(74.5, 13, "Secondary benchmark: does the ordering transfer?" if paper else "Secondary benchmark: does the strategy ordering transfer?", ha="center", fontsize=fs(8.0), fontweight="bold", color="#27AE60")

    if style == "presentation":
        fig.suptitle("Benchmark Corpus Architecture & Multi-Session Conversation Anatomy", x=0.02, y=0.99, ha="left", fontsize=13.5, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.98])
    else:
        plt.tight_layout()

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark anatomy figure in paper, presentation, or both formats.")
    parser.add_argument("--style", choices=["paper", "presentation", "both"], default="both")
    args = parser.parse_args()

    THESIS_VIS_DIR.mkdir(parents=True, exist_ok=True)
    if args.style in ("paper", "both"):
        fig = plot_benchmark_anatomy(style="paper")
        save_figure(fig, "fig00_benchmark_anatomy", [THESIS_VIS_DIR])

    if args.style in ("presentation", "both"):
        fig = plot_benchmark_anatomy(style="presentation")
        save_figure(fig, "fig00_benchmark_anatomy", [PRESENTATION_DIR, THESIS_VIS_DIR / "presentation"])

    print(f"Generated fig00_benchmark_anatomy for style '{args.style}'")


if __name__ == "__main__":
    main()
