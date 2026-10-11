"""Train a simple Decision Tree model for the QKD-AI MVP.

The first project version intentionally uses only the Python standard library,
so every teammate can run it immediately after cloning the repository. B group
can replace this with scikit-learn later without changing the CSV contract.
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
TARGET_COLUMN = "label"
REQUIRED_COLUMNS = ["sample_id", "run_id", "block_index", *FEATURE_COLUMNS, TARGET_COLUMN]
HIDDEN_COLUMNS = {
    "seed", "dataset_seed", "stage", "scenario", "noise_level", "distance_km",
    "attenuation_db_per_km", "aging_delta_db_per_km", "phase_angle_rad",
    "intercept_fraction", "channel_transmittance", "eve_present", "depolarized_count",
}


@dataclass
class Node:
    """決策樹中的單一節點；若沒有 feature，就代表這是一個葉節點。"""

    prediction: str
    feature: str | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None

    def predict(self, row: dict[str, float]) -> str:
        """從目前節點開始，依照 threshold 往左或往右走，直到得到預測 label。"""

        if self.feature is None or self.threshold is None:
            return self.prediction

        if row[self.feature] <= self.threshold:
            if self.left is None:
                return self.prediction
            return self.left.predict(row)

        if self.right is None:
            return self.prediction
        return self.right.predict(row)


@dataclass
class SimpleDecisionTree:
    """簡易版決策樹分類器，用標準函式庫完成 MVP 驗收流程。"""

    max_depth: int
    min_samples_leaf: int
    feature_columns: list[str]
    root: Node | None = None
    split_counts: Counter[str] | None = None

    def fit(self, rows: list[dict[str, Any]]) -> None:
        """訓練決策樹，並記錄每個特徵被拿來切分的次數。"""

        self.split_counts = Counter()
        self.root = self._build(rows, depth=0)

    def predict_one(self, row: dict[str, float]) -> str:
        """預測單筆資料的 label。"""

        if self.root is None:
            raise ValueError("Model has not been trained yet.")
        return self.root.predict(row)

    def predict(self, rows: list[dict[str, Any]]) -> list[str]:
        """批次預測多筆資料。"""

        return [self.predict_one(row) for row in rows]

    def feature_importance(self) -> dict[str, float]:
        """用各特徵被用來切分的比例，提供簡化版 feature importance。"""

        if not self.split_counts:
            return {feature: 0.0 for feature in self.feature_columns}

        total = sum(self.split_counts.values())
        return {
            feature: self.split_counts[feature] / total
            for feature in self.feature_columns
        }

    def _build(self, rows: list[dict[str, Any]], depth: int) -> Node:
        """遞迴建立決策樹；每一層挑選能最大幅降低 Gini impurity 的切分。"""

        labels = [row[TARGET_COLUMN] for row in rows]
        prediction = majority_label(labels)

        # 達到最大深度，或目前資料已經都屬於同一類，就停止切分。
        if depth >= self.max_depth or len(set(labels)) == 1:
            return Node(prediction=prediction)

        # 找不到有效切分時，用目前節點的多數類別作為預測結果。
        split = best_split(rows, self.feature_columns, self.min_samples_leaf)
        if split is None:
            return Node(prediction=prediction)

        feature, threshold, left_rows, right_rows = split
        if self.split_counts is not None:
            self.split_counts[feature] += 1

        return Node(
            prediction=prediction,
            feature=feature,
            threshold=threshold,
            left=self._build(left_rows, depth + 1),
            right=self._build(right_rows, depth + 1),
        )


def majority_label(labels: list[str]) -> str:
    """回傳目前資料中數量最多的 label，作為葉節點的預測值。"""

    return Counter(labels).most_common(1)[0][0]


def gini(labels: list[str]) -> float:
    """計算 Gini impurity；數值越低代表分類越純。"""

    total = len(labels)
    if total == 0:
        return 0.0

    counts = Counter(labels)
    impurity = 1.0
    for count in counts.values():
        probability = count / total
        impurity -= probability * probability
    return impurity


def weighted_gini(left_labels: list[str], right_labels: list[str]) -> float:
    """計算切分後左右兩邊加權平均的 Gini impurity。"""

    total = len(left_labels) + len(right_labels)
    return (len(left_labels) / total) * gini(left_labels) + (len(right_labels) / total) * gini(right_labels)


def candidate_thresholds(rows: list[dict[str, Any]], feature: str) -> list[float]:
    """從某個特徵的排序值中，產生所有可能的切分門檻。"""

    values = sorted({float(row[feature]) for row in rows})
    if len(values) < 2:
        return []

    return [
        (values[index] + values[index + 1]) / 2
        for index in range(len(values) - 1)
    ]


def best_split(
    rows: list[dict[str, Any]],
    feature_columns: list[str],
    min_samples_leaf: int,
) -> tuple[str, float, list[dict[str, Any]], list[dict[str, Any]]] | None:
    """找出目前節點最佳的 feature 和 threshold。"""

    current_labels = [row[TARGET_COLUMN] for row in rows]
    current_gini = gini(current_labels)
    best: tuple[float, str, float, list[dict[str, Any]], list[dict[str, Any]]] | None = None

    for feature in feature_columns:
        for threshold in candidate_thresholds(rows, feature):
            # 依照門檻把資料分成左右兩群，模擬決策樹的一次切分。
            left_rows = [row for row in rows if float(row[feature]) <= threshold]
            right_rows = [row for row in rows if float(row[feature]) > threshold]

            # 避免切出太小的葉節點，降低 overfit 風險。
            if len(left_rows) < min_samples_leaf or len(right_rows) < min_samples_leaf:
                continue

            # gain 越大代表這次切分越能降低資料混雜程度。
            split_gini = weighted_gini(
                [row[TARGET_COLUMN] for row in left_rows],
                [row[TARGET_COLUMN] for row in right_rows],
            )
            gain = current_gini - split_gini

            if best is None or gain > best[0]:
                best = (gain, feature, threshold, left_rows, right_rows)

    if best is None or best[0] <= 0:
        return None

    _, feature, threshold, left_rows, right_rows = best
    return feature, threshold, left_rows, right_rows


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
