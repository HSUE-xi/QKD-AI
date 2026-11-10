"""Train a simple Decision Tree model for the QKD-AI MVP.

The first project version intentionally uses only the Python standard library,
so every teammate scan run it immediately after cloning the repository. B group
can replace this with scikit-learn later without changing the CSV contract.
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from qkd_model import TARGET_COLUMN, SimpleDecisionTree


# 僅使用 CSV v1 的可觀測欄位；ID、排序欄位與 A 組真值都不是特徵。
FEATURE_COLUMNS = [
    "qber",
    "z_error_rate",
    "x_error_rate",
    "click_rate",
    "test_count",
    "z_test_count",
    "x_test_count",
]
REQUIRED_COLUMNS = ["sample_id", "run_id", "block_index", *FEATURE_COLUMNS, TARGET_COLUMN]
HIDDEN_COLUMNS = {
    "seed", "dataset_seed", "stage", "scenario", "noise_level", "distance_km",
    "attenuation_db_per_km", "aging_delta_db_per_km", "phase_angle_rad",
    "intercept_fraction", "channel_transmittance", "eve_present", "depolarized_count",
}





def load_dataset(path: Path) -> list[dict[str, Any]]:
    """讀取 CSV v1，拒收混入模擬真值的資料檔。"""

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Run src/a_group/generate_bb84_dataset.py first."
        )

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        # 欄位缺失代表 A 組 CSV 介面和 B 組訓練需求沒有對齊。
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        leaked_columns = HIDDEN_COLUMNS.intersection(reader.fieldnames or [])
        if leaked_columns:
            raise ValueError(f"Truth columns must not appear in training data: {', '.join(sorted(leaked_columns))}")

        rows: list[dict[str, Any]] = []
        for row in reader:
            parsed = {feature: float(row[feature]) for feature in FEATURE_COLUMNS}
            parsed["sample_id"] = int(row["sample_id"])
            parsed["run_id"] = int(row["run_id"])
            parsed["block_index"] = int(row["block_index"])
            parsed[TARGET_COLUMN] = row[TARGET_COLUMN]
            rows.append(parsed)

    if not rows:
        raise ValueError("Dataset is empty.")
    return rows


def train_test_split(
    rows: list[dict[str, Any]],
    test_size: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按整個 run 的故障種類切分，避免基準段與故障段跨資料集。"""

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    runs_by_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        runs_by_id[row["run_id"]].append(row)

    grouped: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for run in runs_by_id.values():
        fault_labels = {row[TARGET_COLUMN] for row in run if row[TARGET_COLUMN] != "normal"}
        if len(fault_labels) > 1:
            raise ValueError("A run must contain at most one non-normal cause.")
        grouped[next(iter(fault_labels), "normal")].append(run)

    rng = random.Random(seed)
    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    for label, runs in grouped.items():
        runs = list(runs)
        if len(runs) < 2:
            raise ValueError(f"Label {label} needs at least two runs; lower --run-size.")
        rng.shuffle(runs)
        test_count = min(len(runs) - 1, max(1, math.ceil(len(runs) * test_size)))
        test_rows.extend(row for run in runs[:test_count] for row in run)
        train_rows.extend(row for run in runs[test_count:] for row in run)

    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    return train_rows, test_rows


def confusion_matrix(labels: list[str], actual: list[str], predicted: list[str]) -> list[list[int]]:
    """建立混淆矩陣，用來觀察不同成因的誤判。"""

    label_to_index = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]

    for actual_label, predicted_label in zip(actual, predicted):
        matrix[label_to_index[actual_label]][label_to_index[predicted_label]] += 1

    return matrix


def print_confusion_matrix(labels: list[str], matrix: list[list[int]]) -> None:
    """用純文字表格印出混淆矩陣。"""

    width = max(8, *(len(label) for label in labels)) + 2
    print(" " * width + "".join(label.rjust(width) for label in labels))
    for label, row in zip(labels, matrix):
        print(label.rjust(width) + "".join(str(value).rjust(width) for value in row))


def print_classification_report(labels: list[str], matrix: list[list[int]]) -> None:
    """根據混淆矩陣計算 precision、recall 和 f1。"""

    width = max(8, *(len(label) for label in labels))
    print(f"{'label'.rjust(width)}{'precision'.rjust(12)}{'recall'.rjust(12)}{'f1'.rjust(12)}")

    for index, label in enumerate(labels):
        true_positive = matrix[index][index]
        predicted_total = sum(row[index] for row in matrix)
        actual_total = sum(matrix[index])
        precision = true_positive / predicted_total if predicted_total else 0.0
        recall = true_positive / actual_total if actual_total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(f"{label.rjust(width)}{precision:12.4f}{recall:12.4f}{f1:12.4f}")


def eavesdrop_metrics(
    actual: list[str],
    predicted: list[str],
    attack: str = "intercept_resend",
    normal: str = "normal",
    noise: str = "depolarizing_noise",
) -> dict[str, float]:
    """針對竊聽類別，計算抓到、漏報、與雜訊混淆等指標。"""

    attack_predictions = [p for a, p in zip(actual, predicted) if a == attack]
    attack_total = len(attack_predictions)
    if attack_total == 0:
        raise ValueError(f"No '{attack}' samples in the test set.")

    caught = sum(p == attack for p in attack_predictions)
    missed = sum(p == normal for p in attack_predictions)
    as_noise = sum(p == noise for p in attack_predictions)
    other = attack_total - caught - missed - as_noise

    predicted_attack = sum(p == attack for p in predicted)
    actual_normal = sum(a == normal for a in actual)
    false_alarm = sum(a == normal and p == attack for a, p in zip(actual, predicted))

    return {
        "attack_total": attack_total,
        "recall": caught / attack_total,                  # 抓到
        "miss_rate": missed / attack_total,               # 漏報：判成 normal
        "as_noise_rate": as_noise / attack_total,         # 判成去極化雜訊
        "other_cause_rate": other / attack_total,         # 判成其他成因（五類才會出現）
        "precision": caught / predicted_attack if predicted_attack else 0.0,
        "false_alarm_rate": false_alarm / actual_normal if actual_normal else 0.0,  # normal 被判成竊聽
    }


def print_eavesdrop_report(title: str, metrics: dict[str, float]) -> None:
    """印出竊聽誤判指標。"""

    print(title)
    print(f"  Actual intercept_resend blocks: {int(metrics['attack_total'])}")
    print(f"  Recall (caught):                {metrics['recall']:.4f}")
    print(f"  Miss rate (-> normal):          {metrics['miss_rate']:.4f}")
    print(f"  Confused with noise:            {metrics['as_noise_rate']:.4f}")
    print(f"  Other cause:                    {metrics['other_cause_rate']:.4f}")
    print(f"  Precision:                      {metrics['precision']:.4f}")
    print(f"  False alarm (normal -> attack): {metrics['false_alarm_rate']:.4f}")


def macro_f1(matrix: list[list[int]]) -> float:
    """每類 F1 的平均，避免筆數多的類別掩蓋少數類別。"""

    scores = []
    for index in range(len(matrix)):
        tp = matrix[index][index]
        predicted_total = sum(row[index] for row in matrix)
        actual_total = sum(matrix[index])
        precision = tp / predicted_total if predicted_total else 0.0
        recall = tp / actual_total if actual_total else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def parse_args() -> argparse.Namespace:
    """定義訓練腳本的命令列參數。"""

    parser = argparse.ArgumentParser(description="Train a QKD eavesdropping detector.")
    parser.add_argument("--input", type=Path, default=Path("data/qkd_dataset.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/decision_tree_qkd.pkl"))
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """程式入口：讀資料、切分資料、訓練模型、印出評估結果並儲存模型。"""

    args = parse_args()
    rows = load_dataset(args.input)
    train_rows, test_rows = train_test_split(rows, args.test_size, args.seed)

    # 使用簡易決策樹完成 MVP；後續可替換成 scikit-learn 的 DecisionTreeClassifier。
    model = SimpleDecisionTree(
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        feature_columns=FEATURE_COLUMNS,
    )
    model.fit(train_rows)

    # 在測試集上評估模型，確認資料格式和基本分類流程都能正常運作。
    actual = [row[TARGET_COLUMN] for row in test_rows]
    predicted = model.predict(test_rows)
    labels = sorted({row[TARGET_COLUMN] for row in rows})
    correct = sum(1 for actual_label, predicted_label in zip(actual, predicted) if actual_label == predicted_label)
    accuracy = correct / len(test_rows)
    matrix = confusion_matrix(labels, actual, predicted)

    print(f"Rows: {len(rows)}")
    print(f"Train rows: {len(train_rows)}")
    print(f"Test rows: {len(test_rows)}")
    print(f"Features: {', '.join(FEATURE_COLUMNS)}")
    print(f"Accuracy: {accuracy:.4f}")
    print()
    print("Confusion matrix")
    print_confusion_matrix(labels, matrix)
    print()
    print("Classification report")
    print_classification_report(labels, matrix)

    # QBER 基準：相同切分、相同參數，但只給模型 qber 一個特徵。
    baseline = SimpleDecisionTree(
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        feature_columns=["qber"],
    )
    baseline.fit(train_rows)
    baseline_predicted = baseline.predict(test_rows)
    baseline_matrix = confusion_matrix(labels, actual, baseline_predicted)
    baseline_accuracy = sum(a == p for a, p in zip(actual, baseline_predicted)) / len(actual)

    print()
    print(f"{'':24}{'accuracy':>10}{'macro-F1':>10}")
    print(f"{'Full model':24}{accuracy:10.4f}{macro_f1(matrix):10.4f}")
    print(f"{'QBER-only baseline':24}{baseline_accuracy:10.4f}{macro_f1(baseline_matrix):10.4f}")
    print()
    print_eavesdrop_report("Full model", eavesdrop_metrics(actual, predicted))
    print()
    print_eavesdrop_report("QBER-only baseline", eavesdrop_metrics(actual, baseline_predicted))
    print()
    print("Feature importance")
    for feature, importance in model.feature_importance().items():
        print(f"- {feature}: {importance:.4f}")

    # 儲存模型和欄位資訊，讓後續整合時知道模型需要哪些輸入欄位。
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as file:
        pickle.dump(
            {
                "model": model,
                "feature_columns": FEATURE_COLUMNS,
                "target_column": TARGET_COLUMN,
                "labels": labels,
            },
            file,
        )
    print()
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
