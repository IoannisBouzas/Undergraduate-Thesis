"""DEPRECATED 2026-09-27. Not the thesis record: GRAPH_TOK and VR_TOK below are hard-coded
prompt tokens (546/788/712) from before the record runs, so the graph query energies and
crossings in its outputs are stale. The record amortization figure is fig17, drawn by
make_extended_figures.py (_amortization_inputs: golden_metric.json + master_table.csv tokens).
Outputs now go to results/_deprecated_amortization_frontier_<model>.{json,png}.

Amortization frontier: which memory strategy is cheapest per CORRECT answer,
as a function of conversation length L (turns) and number of queries Q asked of it.

Every energy input is read from results/golden_metric.json, which measures all of them with
golden_metric.battery_measurement (the phone's charge counter, voltage-weighted):
  - build energy per turn: the E2B vector builds and the one 419-turn graph build;
  - query energy per answer: vector query-only stretches (build windows excluded, so the build is
    not counted twice); full context has no build, so its whole-run energy per answer is used.
Accuracies are the corrected-pipeline record (prefixed retrieval, 70b judge, n=200).

Model:  E_total(mode, L, Q) = E_ingest_per_turn(mode) * L  +  Q * E_query(mode)
        E_per_correct(mode, L, Q) = E_total / (Q * accuracy(mode))

The frontier is the lower envelope of E_per_correct over modes; the decision map is its
argmin, i.e. the static policy a budget-aware controller would implement.

Usage: python -m energy.amortization_frontier [--model e2b|e4b] [--phone-acc]
Writes results/_deprecated_amortization_frontier_<model>.{json,png}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# ---- measured constants (S23, joules), all from golden_metric.py -----------------
_GM = json.loads(Path(__file__).resolve().parents[1].joinpath("results").joinpath("golden_metric.json").read_text(encoding="utf-8"))
_C, _R = _GM["construction"], _GM["replicates"]
_E4B_FC = next(c for c in _GM["cells"] if c["model"] == "E4B" and c["mode"] == "fullcontext")
E_ING = {"fullcontext": 0.0,  # no build: the raw history IS the memory
         "vectorrag": _C["vector"]["j_per_turn"],
         "graph_v3": _C["graph"]["j_per_turn"],
         "graph_v4": _C["graph"]["j_per_turn"]}  # v4 = v3 graph + query-time source turns
E_ING_SD = {"vectorrag": _C["vector"]["sd"], "graph_v3": 0.0, "graph_v4": 0.0}  # one graph build
E_QUERY = {"e2b": {"fullcontext": _R["E2B_fullcontext"]["mean"],
                   "vectorrag": _R["E2B_vectorrag"]["query_j_per_answer"]},
           "e4b": {"fullcontext": _E4B_FC["j_ans"],
                   "vectorrag": _R["E4B_vectorrag"]["query_j_per_answer"]}}
E_QUERY_SD = {"e2b": {"fullcontext": _R["E2B_fullcontext"]["sd"],
                      "vectorrag": _R["E2B_vectorrag"]["query_j_per_answer_sd"]},
              "e4b": {"fullcontext": 0.0,  # one sitting
                      "vectorrag": _R["E4B_vectorrag"]["query_j_per_answer_sd"]}}

# graph query energy is ESTIMATED by prompt-token scaling from vr (no phone graph QA run):
GRAPH_TOK = {"graph_v3": 546.0, "graph_v4": 788.0}
VR_TOK = 712.0

# accuracy: corrected pipeline (prefixed retrieval), 70b judge, LoCoMo n=200, box.
# e4b fullcontext 0.105 = lap_locomo200_e4b_fullcontext record (was 0.060, which matched
# fc_think not record fc — fixed 2026-08-04). Box accuracy is a DE-LICENSED sensitivity input
# after the parity FAIL; the --phone-acc figure (fc/vr only) is the record.
ACC_BOX = {"e2b": {"fullcontext": 0.075, "vectorrag": 0.360, "graph_v3": 0.215, "graph_v4": 0.330},
           "e4b": {"fullcontext": 0.105, "vectorrag": 0.405, "graph_v3": 0.275, "graph_v4": 0.325}}
# phone-measured accuracy (n=20, same questions on-device) — sensitivity variant only
ACC_PHONE = {"e2b": {"fullcontext": 0.05, "vectorrag": 0.50},
             "e4b": {"fullcontext": 0.05, "vectorrag": 0.40}}

MODES = ["fullcontext", "vectorrag", "graph_v3", "graph_v4"]
COLOR = {"fullcontext": "#d62728", "vectorrag": "#1f77b4",
         "graph_v3": "#2ca02c", "graph_v4": "#9467bd"}


def e_query(model: str, mode: str) -> float:
    if mode in ("graph_v3", "graph_v4"):
        return E_QUERY[model]["vectorrag"] * GRAPH_TOK[mode] / VR_TOK  # disclosed estimate
    return E_QUERY[model][mode]


def per_correct(model: str, acc: dict, mode: str, L, Q) -> np.ndarray:
    """kJ per expected correct answer."""
    total = E_ING[mode] * L + Q * e_query(model, mode)
    return total / (Q * acc[mode]) / 1000.0


def breakeven_Q(model: str, acc: dict, a: str, b: str, L: float) -> float | None:
    """Q where mode a becomes cheaper per correct answer than mode b (None = never)."""
    ea, eb = e_query(model, a) / acc[a], e_query(model, b) / acc[b]      # slope
    ia, ib = E_ING[a] * L / acc[a], E_ING[b] * L / acc[b]                # intercept
    if ea >= eb:
        return None  # a's per-query cost per correct never beats b's -> never amortizes
    d = (ia - ib) / (eb - ea)
    return d if d > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["e2b", "e4b"], default="e2b")
    ap.add_argument("--phone-acc", action="store_true",
                    help="use phone-measured accuracy (n=20) instead of box record (n=200)")
    args = ap.parse_args()
    model = args.model
    acc = dict(ACC_BOX[model])
    if args.phone_acc:
        acc.update(ACC_PHONE[model])
        modes = ["fullcontext", "vectorrag"]
    else:
        modes = MODES

    print(f"AMORTIZATION FRONTIER — {model.upper()}, "
          f"accuracy = {'phone n=20' if args.phone_acc else 'box n=200 (prefixed record)'}\n")

    # 1) asymptote: Q -> infinity, ingest fully amortised
    print("Asymptotic cost (Q -> inf, ingest amortised away):")
    asym = {m: e_query(model, m) / acc[m] / 1000.0 for m in modes}
    for m in sorted(asym, key=asym.get):
        print(f"   {m:12s} {asym[m]:6.2f} kJ per correct answer"
              f"   (E_query {e_query(model, m):6.0f} J / acc {acc[m]:.3f})")
    best_asym = min(asym, key=asym.get)
    print(f"   -> asymptotic winner: {best_asym}\n")

    # 2) break-even analysis vs the asymptotic winner, per conversation length
    print(f"Break-even Q (queries needed to beat {best_asym} on energy per correct answer):")
    lengths = [50, 100, 300, 500, 1000]
    for m in modes:
        if m == best_asym:
            continue
        row = []
        for L in lengths:
            q = breakeven_Q(model, acc, m, best_asym, L)
            row.append("never" if q is None else f"{q:.0f}")
        print(f"   {m:12s} " + "  ".join(f"L={L}: {v:>6s}" for L, v in zip(lengths, row)))
    print()

    # 2b) inverse analysis — when one mode dominates the whole plane, the informative
    # question is what would have to change for a loser to win. Two levers:
    #   (a) accuracy: a mode beats the winner asymptotically iff
    #       acc_m > acc_win * E_query_m / E_query_win  (ingest-independent)
    #   (b) ingest: only matters if the mode already wins asymptotically; otherwise
    #       even a FREE index cannot rescue it.
    print("Requirement analysis (what would make each loser competitive):")
    for m in modes:
        if m == best_asym:
            continue
        need = acc[best_asym] * e_query(model, m) / e_query(model, best_asym)
        gap = need - acc[m]
        if gap <= 0:
            print(f"   {m:12s} already cheaper per correct at large Q")
            continue
        print(f"   {m:12s} needs accuracy >= {need*100:.1f}% (has {acc[m]*100:.1f}%, "
              f"gap {gap*100:+.1f} pts) — even with ZERO ingest cost")
    print(f"   note: ingest cost is not the binding constraint for any loser above; "
          f"their per-query cost per correct answer already exceeds {best_asym}'s.\n")
    report_extra = {m: {"accuracy_needed": acc[best_asym] * e_query(model, m) / e_query(model, best_asym),
                        "accuracy_have": acc[m]}
                    for m in modes if m != best_asym}

    # 3) decision map over (L, Q)
    Ls = np.linspace(20, 1000, 220)
    Qs = np.logspace(0, 3, 220)
    LL, QQ = np.meshgrid(Ls, Qs)
    stack = np.stack([per_correct(model, acc, m, LL, QQ) for m in modes])
    winner = np.argmin(stack, axis=0)
    frontier = stack.min(axis=0)
    uniq, counts = np.unique(winner, return_counts=True)
    print("Decision-map coverage (share of the (L,Q) grid each mode wins):")
    for i, c in sorted(zip(uniq, counts), key=lambda x: -x[1]):
        print(f"   {modes[i]:12s} {c / winner.size * 100:5.1f}%")

    report = {
        "model": model, "accuracy_source": "phone_n20" if args.phone_acc else "box_n200_prefixed",
        "e_ingest_j_per_turn": {m: E_ING[m] for m in modes},
        "e_query_j": {m: e_query(model, m) for m in modes},
        "accuracy": {m: acc[m] for m in modes},
        "asymptotic_kj_per_correct": asym,
        "asymptotic_winner": best_asym,
        "breakeven_Q_vs_winner": {
            m: {str(L): breakeven_Q(model, acc, m, best_asym, L) for L in lengths}
            for m in modes if m != best_asym},
        "decision_map_share": {modes[i]: float(c / winner.size) for i, c in zip(uniq, counts)},
        "requirement_analysis": report_extra,
        "notes": ("graph query energy is token-scaled from vr (no on-device graph QA run); "
                  "vr query energy = query-only stretches, fc = whole run, both from golden_metric.json; accuracies are box "
                  "unless phone_n20 — box join pending parity_check.py"),
    }
    out_json = f"results/_deprecated_amortization_frontier_{model}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out_json}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

        L0 = 400.0  # representative LoCoMo conversation
        Qline = np.logspace(0, 3, 400)
        for m in modes:
            ax1.plot(Qline, per_correct(model, acc, m, L0, Qline), color=COLOR[m], lw=2,
                     ls="--" if m.startswith("graph") else "-",
                     label=m + (" (query J est.)" if m.startswith("graph") else ""))
        ax1.plot(Qline, np.min([per_correct(model, acc, m, L0, Qline) for m in modes], axis=0),
                 color="k", lw=1, alpha=.35, label="frontier (lower envelope)")
        ax1.set_xscale("log"); ax1.set_yscale("log")
        ax1.set_xlabel("queries Q asked of one conversation")
        ax1.set_ylabel("kJ per correct answer")
        ax1.set_title(f"Amortization curves, L={L0:.0f}-turn conversation ({model.upper()})", fontsize=10.5)
        ax1.legend(fontsize=7.5); ax1.grid(alpha=.25, which="both")

        cmap = ListedColormap([COLOR[m] for m in modes])
        ax2.pcolormesh(Ls, Qs, winner, cmap=cmap, vmin=0, vmax=len(modes) - 1, shading="auto")
        ax2.set_yscale("log")
        ax2.set_xlabel("conversation length L (turns)")
        ax2.set_ylabel("queries Q")
        ax2.set_title("Cheapest strategy per correct answer\n(= static policy for the controller)",
                      fontsize=10.5)
        for i, m in enumerate(modes):
            if (winner == i).any():
                ax2.scatter([], [], color=COLOR[m], label=m, marker="s")
        ax2.legend(fontsize=7.5, loc="lower right")
        fig.tight_layout()
        png = f"results/_deprecated_amortization_frontier_{model}.png"
        fig.savefig(png, dpi=150)
        print(f"wrote {png}")
    except Exception as e:
        print(f"[plot skipped: {e}]")


if __name__ == "__main__":
    main()
