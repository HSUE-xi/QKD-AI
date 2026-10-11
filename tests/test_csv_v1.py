"""Checks for the A/B CSV v1 handoff and BB84 physics baselines."""

from __future__ import annotations

import csv
import io
import math
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import Mock

from src.a_group.generate_bb84_dataset import (
    BASIC_LABELS,
    FIELD_NAMES,
    LABELS,
    TRUTH_FIELD_NAMES,
    BlockConditions,
    fiber_transmittance,
    generate_dataset,
    simulate_sample,
    write_csv,
)
from src.b_group.train_decision_tree import (
    FEATURE_COLUMNS,
    load_dataset,
    train_test_split,
)


class CsvV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows, cls.truth = generate_dataset(
            samples_per_label=12,
            photons_per_sample=200,
            run_size=3,
            test_fraction=0.2,
            seed=7,
        )

    def test_schema_pairing_and_public_counts(self) -> None:
        self.assertEqual(len(self.rows), 36)
        self.assertEqual(Counter(row["label"] for row in self.rows), dict.fromkeys(BASIC_LABELS, 12))
        self.assertEqual([row["sample_id"] for row in self.rows], list(range(1, 37)))

        for row, truth in zip(self.rows, self.truth):
            self.assertEqual(list(row), FIELD_NAMES)
            self.assertEqual(list(truth), TRUTH_FIELD_NAMES)
            self.assertEqual(
                (row["sample_id"], row["run_id"], row["block_index"]),
                (truth["sample_id"], truth["run_id"], truth["block_index"]),
            )
            self.assertEqual(row["z_test_count"] + row["x_test_count"], row["test_count"])
            self.assertEqual(
                row["z_test_error_count"] + row["x_test_error_count"],
                row["test_error_count"],
            )
            self.assertGreater(row["z_test_count"], 0)
            self.assertGreater(row["x_test_count"], 0)
            self.assertLessEqual(row["test_count"], row["sifted_count"])
            self.assertLessEqual(row["sifted_count"], row["click_count"])
            self.assertLessEqual(row["click_count"], row["sent_count"])
            self.assertAlmostEqual(
                row["qber"], row["test_error_count"] / row["test_count"], places=6,
            )
            self.assertAlmostEqual(
                row["click_rate"], row["click_count"] / row["sent_count"], places=6,
            )

        self.assertFalse(
            (set(TRUTH_FIELD_NAMES) - {"sample_id", "run_id", "block_index"})
            & set(FIELD_NAMES)
        )
        self.assertFalse({"sample_id", "run_id", "block_index", "label"} & set(FEATURE_COLUMNS))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.csv"
            write_csv(self.rows, path, FIELD_NAMES)
            with path.open(newline="", encoding="utf-8") as file:
                self.assertEqual(csv.DictReader(file).fieldnames, FIELD_NAMES)

    def test_generation_is_repeatable_and_splits_by_run(self) -> None:
        self.assertEqual(
            (self.rows, self.truth),
            generate_dataset(12, 200, run_size=3, test_fraction=0.2, seed=7),
        )
        train, test = train_test_split(self.rows, test_size=0.25, seed=99)
        self.assertFalse({row["run_id"] for row in train} & {row["run_id"] for row in test})
        self.assertEqual(len(train) + len(test), len(self.rows))
        self.assertEqual({row["label"] for row in train}, set(BASIC_LABELS))
        self.assertEqual({row["label"] for row in test}, set(BASIC_LABELS))

    def test_diagnostic_runs_keep_baselines_and_faults_together(self) -> None:
        rows, truth = generate_dataset(12, 200, run_size=4, seed=11, stage="diagnostic")
        self.assertEqual({row["label"] for row in rows}, set(LABELS))
        self.assertEqual(len(rows), len(truth))
        by_run: dict[int, list[dict[str, int | float | str]]] = {}
        for row in rows:
            by_run.setdefault(int(row["run_id"]), []).append(row)
        for run in by_run.values():
            ordered = sorted(run, key=lambda row: int(row["block_index"]))
            self.assertEqual([row["block_index"] for row in ordered], list(range(len(run))))
            self.assertEqual(ordered[0]["label"], "normal")
            self.assertEqual(len({row["label"] for row in ordered[1:]}), 1)

        train, test = train_test_split(rows, test_size=0.25, seed=99)
        self.assertFalse({row["run_id"] for row in train} & {row["run_id"] for row in test})
        self.assertEqual({row["label"] for row in train}, set(LABELS))
        self.assertEqual({row["label"] for row in test}, set(LABELS))

        truth_by_run: dict[int, list[dict[str, int | float | str]]] = {}
        for entry in truth:
            truth_by_run.setdefault(int(entry["run_id"]), []).append(entry)
        for run in truth_by_run.values():
            ordered = sorted(run, key=lambda row: int(row["block_index"]))
            if ordered[0]["scenario"] == "fiber_aging":
                self.assertGreater(
                    float(ordered[0]["channel_transmittance"]),
                    float(ordered[-1]["channel_transmittance"]),
                )
            if ordered[0]["scenario"] == "polarization_drift":
                self.assertEqual(ordered[0]["phase_angle_rad"], 0)
                self.assertGreater(float(ordered[-1]["phase_angle_rad"]), 0)

    def test_physics_baselines(self) -> None:
        def sample(conditions: BlockConditions, seed: int = 3) -> dict[str, int | float | str]:
            observed, _ = simulate_sample(
                1, 1, 0, conditions, 100_000, 0.5, seed, seed, "basic", random.Random(seed),
            )
            return observed

        ideal = sample(BlockConditions("normal", "normal", 0.0, 1.0))
        self.assertEqual(ideal["qber"], 0)
        self.assertEqual(ideal["click_rate"], 1)
        self.assertAlmostEqual(float(ideal["basis_match_rate"]), 0.5, delta=0.01)

        loss = sample(BlockConditions("normal", "normal", 0.0, 0.4))
        self.assertEqual(loss["qber"], 0)
        self.assertAlmostEqual(float(loss["click_rate"]), 0.4, delta=0.01)

        full_eve = sample(BlockConditions("intercept_resend", "intercept_resend", 0.0, 1.0,
                                          intercept_fraction=1.0))
        partial_eve = sample(BlockConditions("intercept_resend", "intercept_resend", 0.0, 1.0,
                                             intercept_fraction=0.2))
        self.assertAlmostEqual(float(full_eve["qber"]), 0.25, delta=0.02)
        self.assertAlmostEqual(float(partial_eve["qber"]), 0.05, delta=0.01)

        phase = 0.6
        drift = sample(BlockConditions("polarization_drift", "polarization_drift", 0.0, 1.0,
                                       phase_angle_rad=phase))
        self.assertEqual(drift["z_error_rate"], 0)
        self.assertAlmostEqual(float(drift["x_error_rate"]), math.sin(phase / 2) ** 2,
                               delta=0.02)

    def test_fiber_formula_and_matched_qber_causes(self) -> None:
        self.assertEqual(fiber_transmittance(0, 0.2), 1)
        self.assertAlmostEqual(fiber_transmittance(20, 0.2), 0.398107, places=5)

        # 三種原因設定為理論總 QBER 約 5%，但 Z/X 型態不必相同。
        phase = 2 * math.asin(math.sqrt(0.1))
        conditions = [
            BlockConditions("depolarizing_noise", "depolarizing_noise", 0.1, 1.0),
            BlockConditions("intercept_resend", "intercept_resend", 0.0, 1.0,
                            intercept_fraction=0.2),
            BlockConditions("polarization_drift", "polarization_drift", 0.0, 1.0,
                            phase_angle_rad=phase),
        ]
        rows = [simulate_sample(1, 1, 0, item, 100_000, 0.5, 23, 23, "diagnostic",
                                random.Random(23))[0] for item in conditions]
        for row in rows:
            self.assertAlmostEqual(float(row["qber"]), 0.05, delta=0.012)
        self.assertGreater(float(rows[2]["x_error_rate"]), 0.08)
        self.assertEqual(rows[2]["z_error_rate"], 0)

    def test_loader_accepts_observables_and_rejects_truth(self) -> None:
        def csv_path(fieldnames: list[str], rows: list[dict[str, object]]) -> Mock:
            stream = io.StringIO()
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            path = Mock()
            path.exists.return_value = True
            path.open.return_value = io.StringIO(stream.getvalue())
            return path

        path = csv_path(FIELD_NAMES, self.rows)
        parsed = load_dataset(path)
        self.assertEqual(len(parsed), len(self.rows))
        self.assertEqual(parsed[0]["run_id"], self.rows[0]["run_id"])

        leaked = [{**row, "eve_present": 1} for row in self.rows]
        with self.assertRaisesRegex(ValueError, "Truth columns"):
            load_dataset(csv_path([*FIELD_NAMES, "eve_present"], leaked))
        leaked_seed = [{**row, "seed": 7} for row in self.rows]
        with self.assertRaisesRegex(ValueError, "Truth columns"):
            load_dataset(csv_path([*FIELD_NAMES, "seed"], leaked_seed))
        leaked_fraction = [{**row, "intercept_fraction": 1} for row in self.rows]
        with self.assertRaisesRegex(ValueError, "Truth columns"):
            load_dataset(csv_path([*FIELD_NAMES, "intercept_fraction"], leaked_fraction))

    def test_rejects_too_few_photons(self) -> None:
        with self.assertRaisesRegex(ValueError, "increase --photons-per-sample"):
            generate_dataset(1, 1, run_size=1, seed=3)
        with self.assertRaisesRegex(ValueError, "at least 2"):
            generate_dataset(1, 100, run_size=1, stage="diagnostic")


if __name__ == "__main__":
    unittest.main()
