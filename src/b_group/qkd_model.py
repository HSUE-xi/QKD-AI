"""QKD-AI 決策樹模型：Node、SimpleDecisionTree 與切分用的工具函式。

獨立成模組，pickle 才會記成 qkd_model.SimpleDecisionTree，
別的程式只要 import 得到這個模組就能載入模型。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

TARGET_COLUMN = "label"


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

