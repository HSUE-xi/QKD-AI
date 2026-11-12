"""B5：在相同誤報率下，比較完整模型和 QBER 門檻偵測器抓到竊聽的比例。"""


from pathlib import Path

from train_decision_tree import (
    FEATURE_COLUMNS,
    SimpleDecisionTree,
    eavesdrop_metrics,
    load_dataset,
    train_test_split,
)

def qber_false_alarm(rows, threshold):
    """正常的 block 裡，qber > threshold（被誤判成竊聽）的比例。"""
    normal = [row for row in rows if row["label"] == "normal"]
    return sum(row["qber"] > threshold for row in normal) / len(normal)


def qber_recall(rows, threshold):
    """竊聽的 block 裡，qber > threshold（被抓到）的比例。"""
    intercept_resend = [row for row in rows if row["label"] == "intercept_resend"]
    return sum(row["qber"] > threshold for row in intercept_resend) / len(intercept_resend)

# 1. 讀資料、切分（參數跟訓練程式一樣，才會切出同一份測試集）
rows = load_dataset(Path("data/qkd_dataset_diagnostic.csv"))
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
full = eavesdrop_metrics(actual, predicted)
print("完整模型：抓到率", full["recall"], "誤報率", full["false_alarm_rate"])
for t in [0.0, 0.02, 0.05, 0.11]:
    print(f"門檻 {t}：誤報率 {qber_false_alarm(test_rows, t):.1%}  抓到率 {qber_recall(test_rows, t):.1%}")
target = full["false_alarm_rate"]   # 完整模型的誤報率，5.7%

candidates = sorted({row["qber"] for row in train_rows if row["label"] == "normal"})
for t in candidates:
    if qber_false_alarm(train_rows, t) <= target:
        best_t = t
        break   # 找到第一個符合的就停下來，Python 用哪個字跳出迴圈？

# 協定門檻（BB84 常用 11%）
protocol_t = 0.11

print()
print("| 方法 | 門檻 | 誤報率 | 抓到率 |")
print("|---|---|---|---|")
print(f"| 完整模型（決策樹，7 特徵） | — | {full['false_alarm_rate']:.1%} | {full['recall']:.1%} |")
print(f"| QBER 門檻（訓練集挑選） | {best_t:.4f} | {qber_false_alarm(test_rows, best_t):.1%} |  {qber_recall(test_rows, best_t):.1%}|")
print(f"| 協定門檻 | {protocol_t} |{qber_false_alarm(test_rows, protocol_t):.1%}| {qber_recall(test_rows, protocol_t):.1%}|")