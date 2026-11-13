"""B4：用梯度提升做五類診斷，和決策樹比較。"""

from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier

from train_decision_tree import (
    FEATURE_COLUMNS,
    confusion_matrix,
    eavesdrop_metrics,
    load_dataset,
    macro_f1,
    train_test_split,
)


def to_matrix(rows):
    """把每一筆 dict 照 FEATURE_COLUMNS 的順序轉成 list，給 scikit-learn 用。"""
    return [[row[f] for f in FEATURE_COLUMNS] for row in rows]


# 1. 讀資料、切分（跟決策樹用同一份切分）
rows = load_dataset(Path("data/qkd_dataset_diagnostic.csv"))
train_rows, test_rows = train_test_split(rows, test_size=0.25, seed=42)

X_train = to_matrix(train_rows)
y_train = [row["label"] for row in train_rows]
X_test = to_matrix(test_rows)
actual = [row["label"] for row in test_rows]

# 2. 訓練
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# 3. 預測、評估
predicted = list(model.predict(X_test))
labels = sorted(set(actual))
accuracy = sum(a == p for a, p in zip(actual, predicted)) / len(actual)
matrix = confusion_matrix(labels, actual, predicted)
attack = eavesdrop_metrics(actual, predicted)

print(f"準確率 {accuracy:.1%}  macro-F1 {macro_f1(matrix):.4f}")
print(f"竊聽：抓到率 {attack['recall']:.1%}  誤報率 {attack['false_alarm_rate']:.1%}")