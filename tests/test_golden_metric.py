"""Regression tests for the phone battery golden metric."""

from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from energy import golden_metric


class GoldenMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.run_directory = Path(self._temporary_directory.name)

        with (self.run_directory / "resources.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "timestamp",
                    "battery_charge_counter_uah",
                    "battery_voltage_mv",
                    "battery_plugged",
                ],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "timestamp": 0,
                        "battery_charge_counter_uah": 1_000_000,
                        "battery_voltage_mv": 4_400,
                        "battery_plugged": "False",
                    },
                    {
                        "timestamp": 1,
                        "battery_charge_counter_uah": 999_000,
                        "battery_voltage_mv": 4_200,
                        "battery_plugged": "False",
                    },
                    {
                        "timestamp": 2,
                        "battery_charge_counter_uah": 997_000,
                        "battery_voltage_mv": 3_800,
                        "battery_plugged": "False",
                    },
                ]
            )

    def test_cc_delta_kj_uses_voltage_at_each_charge_interval(self) -> None:
        energy_kj = golden_metric.cc_delta_kj(self.run_directory, 0, 2)

        # 1,000 uAh at 4.3 V, then 2,000 uAh at 4.0 V.
        self.assertTrue(math.isclose(energy_kj, 0.04428, rel_tol=1e-12))

    def test_battery_measurement_preserves_direct_charge_delta(self) -> None:
        measurement = golden_metric.battery_measurement(self.run_directory, 0, 2)

        self.assertTrue(math.isclose(measurement.charge_mah, 3.0, rel_tol=1e-12))
        self.assertTrue(math.isclose(measurement.energy_kj, 0.04428, rel_tol=1e-12))
        self.assertTrue(
            math.isclose(measurement.effective_voltage_v, 4.1, rel_tol=1e-12)
        )

    def test_correct_per_charge_uses_measured_mah_per_answer(self) -> None:
        result = golden_metric.correct_per_charge(
            accuracy=0.5,
            n_answers=20,
            total_charge_mah=20.0,
        )

        self.assertTrue(math.isclose(result, 2_500.0, rel_tol=1e-12))

    def test_phase_split_leaves_out_stretches_begun_during_full_charge_pause(self) -> None:
        run = Path(self._temporary_directory.name) / "paused_run"
        run.mkdir()
        # Starts at 100%: the counter holds until t=15, then falls 1,000 uAh per second at 4.0 V.
        with (run / "resources.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "battery_charge_counter_uah", "battery_voltage_mv",
                             "battery_plugged", "battery_level_pct"])
            for t in range(0, 41):
                writer.writerow([t, 5_000_000 - 1_000 * max(0, t - 15), 4_000, "False", 100])
        rows = [  # history A built 0-10 s (paused), asked at 12 s; history B built 20-30 s, asked at 32 s
            {"sample_id": "A", "ingest_seconds": 10, "ingest_start_ts": 0, "ingest_end_ts": 10,
             "n_history_turns": 100, "query_start_ts": 12, "query_end_ts": 14},
            {"sample_id": "B", "ingest_seconds": 10, "ingest_start_ts": 20, "ingest_end_ts": 30,
             "n_history_turns": 50, "query_start_ts": 32, "query_end_ts": 40},
        ]
        (run / "locomo_vectorrag_fixture.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8")

        split = golden_metric.phase_split(run)

        # Only B's build (10 s x 1,000 uAh x 4.0 V = 144 J over 50 turns) and the 30-40 s stretch count.
        self.assertEqual(split["build"]["units"], 50)
        self.assertTrue(math.isclose(split["build"]["per_unit"], 144.0 / 50, rel_tol=1e-9))
        self.assertTrue(math.isclose(split["query"]["joules"], 144.0, rel_tol=1e-9))
        self.assertEqual(len(split["skipped"]), 2)  # A's build and A's question stretch


if __name__ == "__main__":
    unittest.main()
