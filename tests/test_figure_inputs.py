"""Checks for the data loaders behind fig00, fig16, fig17, fig20 and fig21, whose numbers the captions and chapters quote."""

from __future__ import annotations

import json

import numpy as np

from figures import make_benchmark_anatomy_visual as anatomy
from figures import make_extended_figures as addons


def test_amortization_inputs_match_the_appendix() -> None:
    ingest, query, _ = addons._amortization_inputs()
    assert abs(ingest["GraphRAG v3"] - 21.898) < 0.01  # appendix: 21.898 kJ graph build
    assert round(ingest["GraphRAG v3"] / (query["Full context"] - query["GraphRAG v3"])) == 17  # caption: Q≈17
    assert round(ingest["GraphRAG v4"] / (query["Full context"] - query["GraphRAG v4"])) == 19  # caption: Q≈19
    assert abs(ingest["Vector RAG"] - 0.853) < 0.001  # caption: 0.85 kJ vector build (2.04 J/turn x 419)


def test_phone_query_times_match_ch5() -> None:
    _, seconds = addons._phone_query_times(addons.PHONE_E2B_LATENCY_RUNS["Full context"])
    assert len(seconds) == 59 and 400 < np.median(seconds) < 440  # Ch5: 411-431 s per sitting
    vr_tokens, _ = addons._phone_query_times(addons.PHONE_E2B_LATENCY_RUNS["Vector RAG, k=10"][:3])
    assert abs(np.median(vr_tokens) - 678) <= 1  # fig16 star label


def test_kv_sizes_and_free_ram_match_the_captions() -> None:
    # fig16 and Ch 5 quote the calculated cache at the phone's 8,192-token server context and at 131k
    assert round(addons.e2b_kv_gb(8192) * 1000) == 57 and round(addons.e4b_kv_gb(8192) * 1000) == 155
    assert round(float(addons.e2b_kv_gb(131072)), 2) == 0.81
    # ...and the measured RAM left at each model's highest phone peak, out of MemTotal (10,998 MiB)
    total, _, free = addons._free_ram_gb("Gemma E2B")
    assert round(total, 1) == 11.5 and round(free, 1) == 4.4
    assert round(addons._free_ram_gb("Gemma E4B")[2], 1) == 5.0


def test_query_time_medians_match_fig20_and_ch5() -> None:
    fc_tokens, fc_seconds = addons._phone_query_times(addons.PHONE_E2B_LATENCY_RUNS["Full context"])
    k2_tokens, k2_seconds = addons._phone_query_times(addons.PHONE_E2B_LATENCY_RUNS["Vector RAG, k=2"])
    assert (np.median(fc_tokens), np.median(k2_tokens)) == (4905, 294)
    assert (round(float(np.median(fc_seconds)), 1), round(float(np.median(k2_seconds)), 1)) == (418.5, 20.8)


def test_benchmark_anatomy_uses_real_conv26() -> None:
    example = anatomy._example()
    assert (example["total"], len(example["sessions"]), example["evidence_turn"], example["fc_kept"]) == (419, 19, 26, 79)
    stats = anatomy._corpus_stats()
    assert round(sum(stats["turns"]) / len(stats["turns"])) == 588  # the corpus mean, not conv-26's 419


def test_caroline_case_matches_fig21_and_ch4() -> None:
    case = addons._caroline_case()
    assert (case["candidates"], case["tied"], case["tied_kept"], case["kept_exchanges"]) == (77, 72, 9, 9)
    assert not case["research_in_graph"] and len(case["adoption_triples"]) == 2
    assert (case["caroline_degree"], round(case["hub_share"], 2), case["batches_failed"], case["batches_total"]) == (73, 0.85, 58, 84)
    assert (case["pred"], case["judge_correct"], case["gold"]) == ("Lgbtq support group", 0, "Adoption agencies")
    ties = json.loads((addons.PROJECT_ROOT / "results" / "ego_tie_order.json").read_text(encoding="utf-8"))
    assert (ties["caroline_seed_sweep"]["seeds"], ties["caroline_seed_sweep"]["adoption_triple_kept"]) == (50, 31)
    assert ties["cutoff_ties"]["results/lap_locomo200_e2b_graphrag"]["tie_across_cutoff"] == 164
    assert ties["cutoff_ties"]["results/lap_locomo200_e4b_graphrag"]["tie_across_cutoff"] == 164
    assert ties["locomo_v1_tied_in_both_readers"] == 138
    assert ties["vector_first_seed_check"]["identical"]
    # "workstation vector retrieval misses the evidence turn at every k from 2 to 50": D2:8 is turn 25 of conv-26
    for model in ("e2b", "e4b"):
        for k in (2, 5, 10, 20, 50):
            path = addons.PROJECT_ROOT / "results" / f"lap_locomo100_{model}_vr_prefix_k{k}" / "judged.jsonl"
            row = next(r for r in map(json.loads, path.open(encoding="utf-8")) if r["question"] == addons.CASE_QUESTION)
            assert 25 not in row["ctx_meta"]["kept_turn_indices"], (model, k)


if __name__ == "__main__":
    for test in (
        test_amortization_inputs_match_the_appendix,
        test_phone_query_times_match_ch5,
        test_kv_sizes_and_free_ram_match_the_captions,
        test_query_time_medians_match_fig20_and_ch5,
        test_benchmark_anatomy_uses_real_conv26,
        test_caroline_case_matches_fig21_and_ch4,
    ):
        test()
        print(f"PASS {test.__name__}")
