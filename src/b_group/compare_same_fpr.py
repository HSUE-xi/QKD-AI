"""B3/B5：在相同誤報率下，比較完整模型和 QBER 門檻偵測器抓到竊聽的比例。"""

import argparse
from collections import defaultdict
from pathlib import Path

from train_decision_tree import (
    FEATURE_COLUMNS,
    SimpleDecisionTree,
    load_dataset,
    train_test_split,
)

def flag_rate_by_class(rows, flagged):
    """每一類 block 裡，被判成竊聽（flagged 為 True）的比例。"""
    counts = defaultdict(int)    # 每類總共幾個
    hits = defaultdict(int)      # 每類被判成竊聽的有幾個
    for row, flag in zip(rows, flagged):
        counts[row["label"]] += 1
        if flag:
            hits[row["label"]] += 1
    return {label: hits[label] / counts[label] for label in sorted(counts)}

def overall_false_alarm(rows, flagged):
    """所有非竊聽 block 裡，被判成竊聽的比例。"""
    pairs = [(row, flag) for row, flag in zip(rows, flagged) if row["label"] != "intercept_resend"]
    return sum(flag for row, flag in pairs) / len(pairs)

# 1. 讀資料、切分（參數跟訓練程式一樣，才會切出同一份測試集）
parser = argparse.ArgumentParser(description="B3/B5：協定門檻、QBER 門檻、決策樹的竊聽偵測比較")
parser.add_argument("--input", type=Path, default=Path("data/qkd_dataset_diagnostic.csv"))
args = parser.parse_args()

rows = load_dataset(args.input)
train_rows, test_rows = train_test_split(rows, test_size=0.25, seed=42)   
# 2. 訓練完整模型（參數跟訓練程式預設值一樣）
model = SimpleDecisionTree(
    max_depth=4,
    min_samples_leaf=5,
    feature_columns=FEATURE_COLUMNS,
)
model.fit(train_rows)

# 3. 預測測試集，算竊聽指標
actual = [row["label"] for row in test_rows]
predicted = [model.predict_one(row) for row in test_rows]
target = 0.05
candidates = sorted({row["qber"] for row in train_rows if row["label"] == "normal"})
best_t=None
for t in candidates:
    if flag_rate_by_class(train_rows, [row["qber"] > t for row in train_rows])["normal"] <= target:
        best_t = t
        break   # 由小到大，第一個符合的就是最低門檻

if best_t is None:
    raise ValueError("找不到符合目標誤報率的門檻")

# 協定門檻（BB84 常用 11%）
protocol_t = 0.11

# 決策樹：預測結果是 intercept_resend 就算「響」
tree_flagged = [p == "intercept_resend" for p in predicted]

# QBER 門檻：qber > 門檻 就算「響」
qber_flagged = [row["qber"] > best_t for row in test_rows]

print()
methods = [
    ("完整模型（決策樹，7 特徵）", "—", tree_flagged),
    ("QBER 門檻（目標 5%，訓練集挑選）", f"{best_t:.4f}", qber_flagged),
    ("協定門檻", "0.11", [row["qber"] > protocol_t for row in test_rows]),
]

# 非竊聽的類別：資料裡有哪些就列哪些
labels = sorted({row["label"] for row in test_rows})
non_attack = [label for label in labels if label != "intercept_resend"]

header = ["方法", "門檻", "竊聽檢出率", "非竊聽整體誤報率"] + [f"{label} 誤報率" for label in non_attack]
print("| " + " | ".join(header) + " |")
print("|" + "---|" * len(header))

for name, threshold, flagged in methods:
    rate = flag_rate_by_class(test_rows, flagged)
    overall = overall_false_alarm(test_rows, flagged)
    cells = [name, threshold, f"{rate['intercept_resend']:.1%}", f"{overall:.1%}"]
    cells += [f"{rate[label]:.1%}" for label in non_attack]
    print("| " + " | ".join(cells) + " |")