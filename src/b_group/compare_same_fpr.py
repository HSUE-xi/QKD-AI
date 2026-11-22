"""B3/B5：協定門檻、QBER 門檻、決策樹（原 7 特徵／加相對特徵）的竊聽偵測比較。"""

import argparse
from collections import defaultdict
from pathlib import Path

from relative_features import RELATIVE_COLUMNS, add_relative_features
from train_decision_tree import (
    FEATURE_COLUMNS,
    SimpleDecisionTree,
    confusion_matrix,
    load_dataset,
    macro_f1,
    print_confusion_matrix,
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


def train_tree(train_rows, feature_columns):
    """用同樣的參數訓練一棵決策樹，只換特徵，比較才公平。"""
    model = SimpleDecisionTree(max_depth=4, min_samples_leaf=5, feature_columns=feature_columns)
    model.fit(train_rows)
    return model


# 1. 讀資料、加相對特徵、切分（參數跟訓練程式一樣，才會切出同一份測試集）
parser = argparse.ArgumentParser(description="B3/B5：協定門檻、QBER 門檻、決策樹的竊聽偵測比較")
parser.add_argument("--input", type=Path, default=Path("data/qkd_dataset_diagnostic.csv"))
args = parser.parse_args()

rows = load_dataset(args.input)
add_relative_features(rows)                      

# 原本的 FEATURE_COLUMNS 不能改（其他程式和 C 組的模型都在用），另外做一個新的 list
EXTENDED_FEATURES = FEATURE_COLUMNS + RELATIVE_COLUMNS                
train_rows, test_rows = train_test_split(rows, test_size=0.25, seed=42)

# 2. 訓練兩棵樹：只差在特徵
tree7 = train_tree(train_rows, FEATURE_COLUMNS)
tree10 = train_tree(train_rows, EXTENDED_FEATURES)              

# 3. QBER 門檻：在訓練集挑「正常區塊誤報率 ≤ 5%」的最低門檻
target = 0.05
candidates = sorted({row["qber"] for row in train_rows if row["label"] == "normal"})
best_t = None
for t in candidates:
    if flag_rate_by_class(train_rows, [row["qber"] > t for row in train_rows])["normal"] <= target:
        best_t = t
        break   # 由小到大，第一個符合的就是最低門檻
if best_t is None:
    raise ValueError("找不到符合目標誤報率的門檻")

protocol_t = 0.11   # 協定門檻（BB84 常用 11%）

# 4. 每個方法都轉成「有沒有響」（判成 intercept_resend 就算響）
tree7_flagged = [tree7.predict_one(row) == "intercept_resend" for row in test_rows]
tree10_flagged = [tree10.predict_one(row) == "intercept_resend" for row in test_rows]       

methods = [
    ("決策樹（原 7 特徵）", "—", tree7_flagged),
    ("決策樹（7 特徵＋相對特徵）", "—", tree10_flagged),       
    ("QBER 門檻（目標 5%，訓練集挑選）", f"{best_t:.4f}", [row["qber"] > best_t for row in test_rows]),
    ("協定門檻", f"{protocol_t}", [row["qber"] > protocol_t for row in test_rows]),
]

# 5. 印表格：欄位依資料中的類別自動產生
labels = sorted({row["label"] for row in test_rows})
non_attack = [label for label in labels if label != "intercept_resend"]

print()
header = ["方法", "門檻", "竊聽檢出率", "非竊聽整體誤報率"] + [f"{label} 誤報率" for label in non_attack]
print("| " + " | ".join(header) + " |")
print("|" + "---|" * len(header))
for name, threshold, flagged in methods:
    rate = flag_rate_by_class(test_rows, flagged)
    overall = overall_false_alarm(test_rows, flagged)
    cells = [name, threshold, f"{rate['intercept_resend']:.1%}", f"{overall:.1%}"]
    cells += [f"{rate[label]:.1%}" for label in non_attack]
    print("| " + " | ".join(cells) + " |")

# 6. 五類診斷：每一類被「正確判成自己那一類」的比例（各類 recall）
actual = [row["label"] for row in test_rows]
models = [
    ("決策樹（原 7 特徵）", [tree7.predict_one(row) for row in test_rows]),
    ("決策樹（7 特徵＋相對特徵）", [tree10.predict_one(row) for row in test_rows]),
]

print()
header = ["模型", "準確率", "macro-F1"] + [f"{label} recall" for label in labels]
print("| " + " | ".join(header) + " |")
print("|" + "---|" * len(header))
for name, predicted in models:
    matrix = confusion_matrix(labels, actual, predicted)   # 列：真實，欄：預測
    accuracy = sum(a == p for a, p in zip(actual, predicted)) / len(actual)
    recalls = []
    for i, label in enumerate(labels):
        correct =  matrix[i][i]          # 空格 1：混淆矩陣裡，這一類「猜對」的是哪一格？
        actual_total =sum(matrix[i])     # 空格 2：這一類「真實總數」是哪一列加起來？
        recalls.append(correct / actual_total)
    cells = [name, f"{accuracy:.1%}", f"{macro_f1(matrix):.4f}"] + [f"{r:.1%}" for r in recalls]
    print("| " + " | ".join(cells) + " |")

for name, predicted in models:
    print()
    print(f"{name} 混淆矩陣（列：真實，欄：預測）")
    print_confusion_matrix(labels, confusion_matrix(labels, actual, predicted))
