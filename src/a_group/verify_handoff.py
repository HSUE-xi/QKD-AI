"""Evaluate the actual A-group CSV and render the intercept-sweep results.

Uses repository-owned features matching the review definitions.
Run from the repository root; --public-only does not require private truth files.
"""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
import platform
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold

from src.a_group.generate_bb84_dataset import FIELD_NAMES, LABELS, TRUTH_FIELD_NAMES
from src.b_group.train_decision_tree import load_dataset, train_test_split
from src.a_group.handoff_features import F7, REL, add_relative_features

REPO = Path(__file__).resolve().parents[2]


def read_observed(path: Path) -> list[dict]:
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELD_NAMES:
            raise ValueError(f'Unexpected schema: {path}')
        return [{key: value if key == 'label' else float(value)
                 for key, value in row.items()} for row in reader]


def check_dataset(path: Path, truth_path: Path | None) -> dict:
    rows = read_observed(path)
    if truth_path is not None:
        with truth_path.open(encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames == TRUTH_FIELD_NAMES
            truths = list(reader)
        assert len(rows) == len(truths)
        for row, truth in zip(rows, truths):
            assert all(row[k] == int(truth[k]) for k in ('sample_id', 'run_id', 'block_index'))
    by_run = defaultdict(list)
    for row in rows:
        assert row['sent_count'] == 10000
        assert 0 <= row['test_error_count'] <= row['test_count'] <= row['sifted_count'] <= row['click_count'] <= row['sent_count']
        assert row['z_test_count'] + row['x_test_count'] == row['test_count']
        assert row['z_test_error_count'] + row['x_test_error_count'] == row['test_error_count']
        for errors, count, rate in [('test_error_count', 'test_count', 'qber'),
                                    ('z_test_error_count', 'z_test_count', 'z_error_rate'),
                                    ('x_test_error_count', 'x_test_count', 'x_error_rate'),
                                    ('click_count', 'sent_count', 'click_rate')]:
            assert 0 <= row[errors] <= row[count]
            assert abs(row[rate] - row[errors] / row[count]) <= 0.00000051
        by_run[row['run_id']].append(row)
    assert len({row['sample_id'] for row in rows}) == len(rows)
    for run in by_run.values():
        ordered = sorted(run, key=lambda row: row['block_index'])
        assert [row['block_index'] for row in ordered] == list(range(len(run)))
        assert ordered[0]['label'] == 'normal'
    train, test = train_test_split(load_dataset(path), 0.25, 42)
    assert not ({row['run_id'] for row in train} & {row['run_id'] for row in test})
    counts = [row['test_count'] for row in rows]
    return dict(rows=len(rows), runs=len(by_run), truth_pairing_checked=truth_path is not None,
                labels=dict(Counter(row['label'] for row in rows)),
                test_count_min=min(counts), test_count_median=float(np.median(counts)),
                test_count_max=max(counts), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def evaluate_actual_csv() -> dict:
    rows = read_observed(REPO / 'data/qkd_dataset_diagnostic_10000.csv')
    add_relative_features(rows)
    labels = list(LABELS)
    target = np.array([row['label'] for row in rows])
    groups = np.array([row['run_id'] for row in rows])
    results = {}
    for name, columns in [('qber_only', ['qber']), ('F7', F7), ('REL', REL)]:
        inputs = np.array([[row[column] for column in columns] for row in rows])
        predicted = np.empty_like(target)
        for train, test in GroupKFold(5).split(inputs, target, groups):
            model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=0)
            predicted[test] = model.fit(inputs[train], target[train]).predict(inputs[test])
        attack = target == 'intercept_resend'
        results[name] = dict(
            accuracy=float(accuracy_score(target, predicted)),
            labels=labels, confusion_matrix=confusion_matrix(target, predicted, labels=labels).tolist(),
            recall={label: float(np.mean(predicted[target == label] == label)) for label in labels},
            intercept_recall=float(np.mean(predicted[attack] == 'intercept_resend')),
            intercept_false_positive_rate=float(np.mean(predicted[~attack] == 'intercept_resend')),
        )
        print(f'Actual CSV {name}: accuracy={results[name]["accuracy"]:.4f}', flush=True)
    return results


def plot_sweep() -> None:
    with (REPO / 'docs/a-group-intercept-detection.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    fig, axis = plt.subplots(figsize=(8, 5), layout='constrained')
    for kind, label in [('random_basis', 'Random-basis intercept'), ('fixed_z', 'Fixed-Z intercept')]:
        selected = [row for row in rows if row['kind'] == kind]
        fraction = np.array([float(row['intercept_fraction']) for row in selected])
        rate = np.array([float(row['detection_rate']) for row in selected])
        low = np.array([float(row['ci95_low']) for row in selected])
        high = np.array([float(row['ci95_high']) for row in selected])
        axis.plot(fraction, rate, 'o-', label=label)
        axis.fill_between(fraction, low, high, alpha=0.12)
    threshold = float(rows[0]['qber_threshold'])
    axis.set(xlabel='Intercept fraction f', ylabel='Run alarm probability', ylim=(-0.02, 1.02),
             title=f'Photon-level simulation, 10,000 photons/block\nQBER threshold {threshold:.2%}; shaded 95% Wilson intervals')
    axis.grid(alpha=0.2)
    axis.legend()
    fig.savefig(REPO / 'docs/a-group-intercept-detection.png', dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public-only', action='store_true',
                        help='Check public CSVs and evaluate models without private truth pairing.')
    args = parser.parse_args()
    checks = {}
    for name in ('diagnostic', 'intercept_sweep'):
        checks[name] = check_dataset(REPO / f'data/qkd_dataset_{name}_10000.csv',
                                     None if args.public_only else REPO / f'data/qkd_truth_{name}_10000.csv')
    assert checks['diagnostic']['rows'] == 2500
    assert set(checks['diagnostic']['labels'].values()) == {500}
    results = dict(environment=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__),
                   dataset_checks=checks, actual_csv_five_fold=evaluate_actual_csv())
    output = REPO / 'docs/a-group-validation-results.json'
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    plot_sweep()
    print(f'All checks passed. Results written to {output}')


if __name__ == '__main__':
    main()
