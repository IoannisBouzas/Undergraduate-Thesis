#!/usr/bin/env python3
"""Test whether LoCoMo correctness varies across the four evaluated question families.

The Chapter 4 sample contains 50 questions from each family and reuses those questions
across the eight primary Gemma reader-memory cells.  For each cell, this script uses
Pearson's 4 x 2 chi-square statistic (family x correct/incorrect) as an omnibus measure.

The p-value comes from a conversation-stratified randomization test rather than the
ordinary independent-item chi-square reference distribution.  Within each of the ten
conversations, the script shuffles the observed correctness labels across that
conversation's question-family labels.  This preserves both the conversation's total
number of correct answers and its family composition.  It therefore asks whether the
observed family separation is larger than expected after accounting for conversation-
level difficulty.  The eight p-values are Holm-adjusted as one declared family.

Usage:
    python -m analysis.question_family_stats

Output:
    results/question_family_stats.csv
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
CATEGORIES = ("1", "2", "3", "4")
N_PERMUTATIONS = 100_000
SEED = 2_026_090_6

CELLS = (
    ("Gemma E2B", "Full context", "lap_locomo200_e2b_fullcontext"),
    ("Gemma E2B", "Graph triples", "lap_locomo200_e2b_graphrag_v3_prefix"),
    ("Gemma E2B", "Graph + sources", "lap_locomo200_e2b_graphrag_v4_prefix"),
    ("Gemma E2B", "Vector RAG", "lap_locomo200_e2b_vr_prefix"),
    ("Gemma E4B", "Full context", "lap_locomo200_e4b_fullcontext"),
    ("Gemma E4B", "Graph triples", "lap_locomo200_e4b_graphrag_v3_prefix"),
    ("Gemma E4B", "Graph + sources", "lap_locomo200_e4b_graphrag_v4_prefix"),
    ("Gemma E4B", "Vector RAG", "lap_locomo200_e4b_vr_prefix"),
)


def read_rows(run: str) -> list[dict]:
    path = RESULTS / run / "judged.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scored = [row for row in rows if row.get("judge_correct") is not None]
    if len(scored) != 200:
        raise ValueError(f"{run}: expected 200 scored rows, found {len(scored)}")
    counts = {category: 0 for category in CATEGORIES}
    for row in scored:
        category = str(row.get("category"))
        if category not in counts:
            raise ValueError(f"{run}: unexpected category {category!r}")
        counts[category] += 1
        if row.get("sample_id") is None:
            raise ValueError(f"{run}: row lacks sample_id")
    if set(counts.values()) != {50}:
        raise ValueError(f"{run}: expected 50 questions per family, found {counts}")
    return scored


def chi_square(successes: np.ndarray, totals: np.ndarray) -> np.ndarray:
    """Pearson statistic for one or many 4 x 2 family-by-correctness tables."""
    successes = np.asarray(successes, dtype=float)
    totals = np.asarray(totals, dtype=float)
    total_success = successes.sum(axis=-1, keepdims=True)
    total_n = totals.sum()
    expected_success = total_success * totals / total_n
    expected_failure = (total_n - total_success) * totals / total_n
    failures = totals - successes
    return (
        ((successes - expected_success) ** 2 / expected_success)
        + ((failures - expected_failure) ** 2 / expected_failure)
    ).sum(axis=-1)


def chi_square_df3_survival(statistic: float) -> float:
    """Upper-tail probability for chi-square with three degrees of freedom."""
    z = statistic / 2.0
    return math.erfc(math.sqrt(z)) + 2.0 * math.sqrt(z) * math.exp(-z) / math.sqrt(math.pi)


def randomization_test(rows: list[dict], seed: int) -> dict[str, object]:
    totals = np.array([50, 50, 50, 50], dtype=np.int16)
    observed_success = np.array(
        [
            sum(
                int(bool(row["judge_correct"]))
                for row in rows
                if str(row["category"]) == category
            )
            for category in CATEGORIES
        ],
        dtype=np.int16,
    )
    observed_stat = float(chi_square(observed_success, totals))

    by_conversation: dict[str, list[dict]] = {}
    for row in rows:
        by_conversation.setdefault(str(row["sample_id"]), []).append(row)

    rng = np.random.default_rng(seed)
    permuted_success = np.zeros((N_PERMUTATIONS, len(CATEGORIES)), dtype=np.int16)
    for conversation_rows in by_conversation.values():
        labels = np.array(
            [int(bool(row["judge_correct"])) for row in conversation_rows],
            dtype=np.int8,
        )
        categories = np.array(
            [CATEGORIES.index(str(row["category"])) for row in conversation_rows],
            dtype=np.int8,
        )
        order = rng.random((N_PERMUTATIONS, len(labels))).argsort(axis=1)
        shuffled_labels = labels[order]
        for category_index in range(len(CATEGORIES)):
            permuted_success[:, category_index] += shuffled_labels[
                :, categories == category_index
            ].sum(axis=1)

    permuted_stat = chi_square(permuted_success, totals)
    exceedances = int(np.count_nonzero(permuted_stat >= observed_stat - 1e-12))
    p_randomization = (exceedances + 1) / (N_PERMUTATIONS + 1)
    return {
        "successes": observed_success.tolist(),
        "chi_square": observed_stat,
        "p_randomization": p_randomization,
        "p_asymptotic": chi_square_df3_survival(observed_stat),
        "conversations": len(by_conversation),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def main() -> None:
    records = []
    for index, (reader, memory, run) in enumerate(CELLS):
        result = randomization_test(read_rows(run), SEED + index)
        records.append(
            {
                "reader": reader,
                "memory": memory,
                "run": run,
                "n": 200,
                "conversations": result["conversations"],
                "cat1_correct": result["successes"][0],
                "cat2_correct": result["successes"][1],
                "cat3_correct": result["successes"][2],
                "cat4_correct": result["successes"][3],
                "per_category_n": 50,
                "pearson_chi_square": result["chi_square"],
                "df": 3,
                "permutations": N_PERMUTATIONS,
                "p_conversation_randomization": result["p_randomization"],
                "p_asymptotic_diagnostic": result["p_asymptotic"],
            }
        )

    adjusted = holm_adjust(
        [float(record["p_conversation_randomization"]) for record in records]
    )
    for record, p_adjusted in zip(records, adjusted):
        record["p_holm_8"] = p_adjusted

    output = RESULTS / "question_family_stats.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    print(f"wrote {output}")
    for record in records:
        print(
            f"{record['reader']:10s} {record['memory']:15s} "
            f"chi2(3)={record['pearson_chi_square']:.3f}, "
            f"p_perm={record['p_conversation_randomization']:.6g}, "
            f"p_holm={record['p_holm_8']:.6g}"
        )


if __name__ == "__main__":
    main()
