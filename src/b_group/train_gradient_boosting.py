"""B4：用梯度提升做五類診斷，和決策樹比較。"""

from collections import defaultdict
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier

from train_decision_tree import (
    FEATURE_COLUMNS,
    SimpleDecisionTree,
    confusion_matrix,
    load_dataset,
    macro_f1,
    print_confusion_matrix,
    train_test_split,
)


def to_matrix(rows):
    """把每一筆 dict 照 FEATURE_COLUMNS 的順序轉成 list，給 scikit-learn 用。"""
    return [[row[f] for f in FEATURE_COLUMNS] for row in rows]


def flag_rate_by_class(rows, flagged):
    """每一類 block 裡，被判成竊聽（flagged 為 True）的比例。
"""
    counts = defaultdict(int)    # 每類總共幾個
    hits = defaultdict(int)      # 每類被判成竊聽的有幾個
    for row, flag in zip(rows, flagged):
        counts[row["label"]] += 1
        if flag:
            hits[row["label"]] += 1
    return {label: hits[label] / counts[label] for label in sorted(counts)}


def overall_false_alarm(rows, flagged):
    """所有非竊聽 block 裡，被判成竊聽的比例。"""
    pairs = [(row, flag) for row, flag in zip(rows, flagged) 
if row["label"] != "intercept_resend"]
    return sum(flag for row, flag in pairs) / len(pairs)

# 1. 讀資料、切分
rows = load_dataset(Path("data/qkd_dataset_diagnostic.csv"))
train_rows, test_rows = train_test_split(rows, test_size=0.25, seed=42)
actual = [row["label"] for row in test_rows]
labels = sorted(set(actual))

# 2. 決策樹
tree = SimpleDecisionTree(max_depth=4, min_samples_leaf=5, feature_columns=FEATURE_COLUMNS)
tree.fit(train_rows)
tree_predicted = [tree.predict_one(row) for row in test_rows]

# 3. 梯度提升（這段你原本就寫好了，變數名稱改成 gbm_predicted）
X_train = to_matrix(train_rows)
y_train = [row["label"] for row in train_rows]
X_test = to_matrix(test_rows)
gbm = GradientBoostingClassifier(random_state=42)
gbm.fit(X_train, y_train)
gbm_predicted = list(gbm.predict(X_test))

# 4. 兩個模型放進同一張表
methods = [
    ("決策樹（7 特徵）", tree_predicted),
    ("梯度提升（7 特徵）", gbm_predicted),
]

print("| 模型 | 準確率 | macro-F1 | 竊聽檢出率 | 正常區塊誤報率 | 非竊聽整體誤報率 | 去極化 | 偏振漂移 | 光纖老化 |")
print("|---|---|---|---|---|---|---|---|---|")
for name, predicted in methods:
    accuracy = sum(a == p for a, p in zip(actual, predicted)) / len(actual)
    f1 = macro_f1(confusion_matrix(labels, actual, predicted))
    flagged = [p == "intercept_resend" for p in predicted]          # 空格 1
    rate = flag_rate_by_class(test_rows, flagged)
    overall = overall_false_alarm(test_rows, flagged)                                # 空格 2
    print(f"| {name} | {accuracy:.1%} | {f1:.4f} | {rate['intercept_resend']:.1%} | "
          f"{rate['normal']:.1%} | {overall:.1%} | {rate['depolarizing_noise']:.1%} | "
          f"{rate['polarization_drift']:.1%} | {rate['fiber_aging']:.1%} |")

# 5. 混淆矩陣
for name, predicted in methods:
    print()
    print(f"{name} 混淆矩陣（列：真實，欄：預測）")
    print_confusion_matrix(labels, confusion_matrix(labels, actual, predicted))