"""Regression checks for model handoff and intercept-sweep fixes."""

import contextlib
import csv
import hashlib
import importlib.util
import io
import os
import pickle
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src" / "b_group"))
sys.path.insert(0, str(REPO_DIR / "src" / "a_group"))
from generate_bb84_dataset import FIELD_NAMES, generate_dataset
import plot_detection_vs_f as sweep


class DemoHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "src" / "b_group").mkdir(parents=True)
        (self.root / "data").mkdir()
        for name in ("train_decision_tree.py", "qkd_model.py", "relative_features.py"):
            shutil.copyfile(REPO_DIR / "src" / "b_group" / name,
                            self.root / "src" / "b_group" / name)
        shutil.copyfile(REPO_DIR / "load_model_example.py", self.root / "load_model_example.py")
        rows, _ = generate_dataset(18, 300, run_size=4, seed=42, stage="diagnostic")
        self.dataset = self.root / "data" / "custom.csv"
        with self.dataset.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELD_NAMES)
            writer.writeheader()
            writer.writerows(rows)

    def run_python(self, *arguments):
        environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
        return subprocess.run([sys.executable, "-B", *arguments], cwd=self.root,
                              env=environment, capture_output=True, text=True,
                              encoding="utf-8", timeout=60)

    def train(self, *arguments):
        result = self.run_python("src/b_group/train_decision_tree.py", "--input",
                                 "data/custom.csv", *arguments)
        self.assertEqual(result.returncode, 0, result.stderr)
        with (self.root / "models" / "decision_tree_qkd.pkl").open("rb") as file:
            return pickle.load(file)

    def test_custom_splits_load_only_saved_test_runs(self):
        for seed, relative in ((0, False), (7, True)):
            with self.subTest(seed=seed, relative=relative):
                arguments = ["--seed", str(seed), "--test-size", "0.4"]
                if relative:
                    arguments.append("--relative")
                bundle = self.train(*arguments)
                metadata = bundle["training_metadata"]
                self.assertFalse(Path(metadata["dataset_path"]).is_absolute())
                self.assertEqual(metadata["seed"], seed)
                self.assertEqual(metadata["test_size"], 0.4)
                self.assertEqual(metadata["dataset_sha256"],
                                 hashlib.sha256(self.dataset.read_bytes()).hexdigest())
                self.assertEqual(len(bundle["feature_columns"]), 10 if relative else 7)
                result = self.run_python("load_model_example.py")
                self.assertEqual(result.returncode, 0, result.stderr)
                selected = next(line for line in result.stdout.splitlines() if line.startswith("run "))
                run_id = int(selected.split()[1])
                self.assertIn(run_id, metadata["test_run_ids"])

    def test_changed_dataset_is_rejected(self):
        self.train("--relative")
        with self.dataset.open("a", encoding="utf-8") as file:
            file.write("\n")
        result = self.run_python("load_model_example.py")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("differs from training data", result.stderr)

    def test_legacy_seven_feature_bundle_remains_loadable(self):
        bundle = self.train()
        bundle.pop("training_metadata")
        bundle.pop("relative_features")
        with (self.root / "models" / "decision_tree_qkd.pkl").open("wb") as file:
            pickle.dump(bundle, file)
        shutil.copyfile(self.dataset, self.root / "data" / "qkd_dataset_diagnostic_10000.csv")
        result = self.run_python("load_model_example.py")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("無法確認", result.stdout)


class SweepRegressionTests(unittest.TestCase):
    def test_invalid_arguments_fail_before_simulation(self):
        for arguments in (["--photons-per-sample", "1"], ["--blocks-per-f", "0"],
                          ["--test-fraction", "0"], ["--test-fraction", "nan"],
                          ["--fractions", "nan"]):
            with self.subTest(arguments=arguments):
                with patch.object(sys, "argv", ["sweep", *arguments]), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as result:
                        sweep.parse_args()
                    self.assertEqual(result.exception.code, 2)

    def test_failed_sampling_is_bounded(self):
        with patch.object(sweep, "simulate_sample", side_effect=ValueError("subset")) as simulate:
            with self.assertRaisesRegex(ValueError, "50 attempts"):
                sweep.simulate_blocks(.1, 1, 2, .2, random.Random(1))
            self.assertEqual(simulate.call_count, 50)

    def test_retry_counter_resets_after_success(self):
        row = dict.fromkeys(sweep.FEATURE_COLUMNS, .01)
        responses = ([ValueError("subset")] * 49 + [(row, {})]) * 2
        with patch.object(sweep, "simulate_sample", side_effect=responses):
            self.assertEqual(len(sweep.simulate_blocks(.1, 2, 1000, .2, random.Random(1))), 2)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is optional")
    def test_false_alarm_annotation_requires_zero_interception(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.axes import Axes

        row = dict.fromkeys(sweep.FEATURE_COLUMNS, .01)
        with tempfile.TemporaryDirectory() as temporary:
            for fractions, expected in ((["0.1", "0.2"], 0), (["0.1", "0"], 1)):
                with self.subTest(fractions=fractions):
                    arguments = ["sweep", "--fractions", *fractions, "--blocks-per-f", "2",
                                 "--output-csv", str(Path(temporary) / "table.csv"),
                                 "--output-png", str(Path(temporary) / "plot.png")]
                    with contextlib.ExitStack() as stack:
                        stack.enter_context(patch.object(sys, "argv", arguments))
                        stack.enter_context(patch.object(sweep, "load_dataset", return_value=[row]))
                        stack.enter_context(patch.object(sweep, "train_test_split", return_value=([row], [])))
                        stack.enter_context(patch.object(sweep.SimpleDecisionTree, "fit"))
                        stack.enter_context(patch.object(sweep.SimpleDecisionTree, "predict",
                                                         return_value=["normal", "normal"]))
                        stack.enter_context(patch.object(sweep, "simulate_blocks",
                                                         return_value=[row, row]))
                        annotation = stack.enter_context(patch.object(Axes, "annotate"))
                        output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                        sweep.main()
                        self.assertEqual(annotation.call_count, expected)
                        self.assertEqual("f = 0 那一列" in output.getvalue(), bool(expected))
                        if expected:
                            self.assertEqual(annotation.call_args.kwargs["xy"], (0, 0))
                    plt.close("all")


if __name__ == "__main__":
    unittest.main()
