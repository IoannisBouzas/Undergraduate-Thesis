"""How far the phone vector/full-context efficiency ratio can fall on 20 questions.

Resamples the 20 paired phone questions with replacement (energy per answer held at the
measured means in results/golden_metric.json) and reports the 2.5th percentile of the
correct-answers-per-kJ ratio. Writes results/phone_ratio_bootstrap.json.

    python -m energy.phone_ratio_bootstrap
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"Q35-0.8B": "q35_08b", "Q35-2B": "q35_2b", "Q35-4B": "q35_4b", "E2B": "e2b", "E4B": "e4b"}
DRAWS = 20000


def labels(run):
    out = {}
    for line in (ROOT / "results_phone" / run / "judged.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out[(r.get("sample_id"), r["question"])] = bool(r["judge_correct"])
    return out


def j_per_answer(gm, key, mode):
    rep = gm["replicates"].get(f"{key}_{mode}")
    if rep:
        return rep["mean"]
    return next(c["j_ans"] for c in gm["cells"] if c["model"] == key and c["mode"] == mode)


def main():
    gm = json.loads((ROOT / "results" / "golden_metric.json").read_text(encoding="utf-8"))
    rng = random.Random(0)
    rows = {}
    for key, tag in MODELS.items():
        vr, fc = labels(f"phone_locomo20_{tag}_vectorrag"), labels(f"phone_locomo20_{tag}_fullcontext")
        qs = sorted(set(vr) & set(fc))
        j_vr, j_fc = j_per_answer(gm, key, "vectorrag"), j_per_answer(gm, key, "fullcontext")
        ratios = []
        for _ in range(DRAWS):
            s = [rng.choice(qs) for _ in qs]
            a_vr, a_fc = sum(vr[q] for q in s), sum(fc[q] for q in s)
            ratios.append(float("inf") if a_fc == 0 else (a_vr / a_fc) * (j_fc / j_vr))
        ratios.sort()
        rows[key] = {
            "n": len(qs),
            "vector_correct": sum(vr[q] for q in qs),
            "fullcontext_correct": sum(fc[q] for q in qs),
            "fullcontext_only_correct": sum(fc[q] and not vr[q] for q in qs),
            "energy_ratio_fc_over_vr": j_fc / j_vr,
            "ratio_p2_5": ratios[int(0.025 * DRAWS)],
            "ratio_median": ratios[DRAWS // 2],
        }
        # the thesis claims the direction holds in every pair; fail loudly if the data stop saying so
        assert rows[key]["ratio_p2_5"] > 1, key
    out = ROOT / "results" / "phone_ratio_bootstrap.json"
    out.write_text(json.dumps({"draws": DRAWS, "seed": 0, "cells": rows}, indent=1), encoding="utf-8")
    for k, r in rows.items():
        print(f"{k:9s} vr {r['vector_correct']}/20 fc {r['fullcontext_correct']}/20 "
              f"fc-only {r['fullcontext_only_correct']} floor {r['ratio_p2_5']:.1f}x")


if __name__ == "__main__":
    main()
