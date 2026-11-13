"""B4：比較五類各特徵的平均值，找出每種故障的物理指紋。"""

from collections import defaultdict
from pathlib import Path
from statistics import mean

from train_decision_tree import load_dataset

FEATURES = ["qber", "z_error_rate", "x_error_rate", "click_rate"]

rows = load_dataset(Path("data/qkd_dataset_diagnostic.csv"))

# 1. 依 label 分組
groups = defaultdict(list)
for row in rows:
    groups[row["label"]].append(row)

# 2. 印出每一類、每個特徵的平均值
print("label".ljust(20) + "".join(f.rjust(14) for f in FEATURES))
for label in sorted(groups):
    blocks = groups[label]
    values = [mean(block[f] for block in blocks) for f in FEATURES]
    print(label.ljust(20) + "".join(f"{v:14.4f}" for v in values))