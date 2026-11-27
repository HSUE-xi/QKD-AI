"""載入 B 組模型並預測的範例（給 C 組串 demo 用）。

先訓練，產生 models/decision_tree_qkd.pkl：
- 展示模型（10 特徵）：python src/b_group/train_decision_tree.py --input data/qkd_dataset_diagnostic_10000.csv --relative
- 舊版（7 特徵）：    python src/b_group/train_decision_tree.py --input data/qkd_dataset_diagnostic_10000.csv

再執行：python load_model_example.py
"""

import hashlib
import pickle
import sys
from pathlib import Path

REPO_DIR = Path(__file__).parent
# 讓 Python 也去 src/b_group 找 qkd_model.py、relative_features.py、train_decision_tree.py
sys.path.insert(0, str(REPO_DIR / "src" / "b_group"))

from relative_features import add_relative_features
from train_decision_tree import load_dataset, train_test_split

# 給人看的顯示名稱；程式和 CSV 的標籤字串不改
DISPLAY_NAME = {
    "normal": "正常",
    "depolarizing_noise": "去極化雜訊",
    "fiber_aging": "光纖老化",
    "polarization_drift": "Z 軸相位型漂移",
    "intercept_resend": "QBER 異常（可能竊聽）",
}

with open(REPO_DIR / "models" / "decision_tree_qkd.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
uses_relative = bundle.get("relative_features", False)   # 舊的 .pkl 沒有這個鍵，代表 7 特徵模型
print("模型需要的特徵：", bundle["feature_columns"])
print("需要相對基準特徵：", uses_relative)

# 優先依模型紀錄的資料版本與測試 run 選取展示樣本。
metadata = bundle.get("training_metadata")
if metadata is not None:
    dataset_path = Path(metadata["dataset_path"])
    if not dataset_path.is_absolute():
        dataset_path = REPO_DIR / dataset_path
    if hashlib.sha256(dataset_path.read_bytes()).hexdigest() != metadata["dataset_sha256"]:
        raise ValueError("Demo dataset differs from training data; restore it or retrain the model.")
    rows = load_dataset(dataset_path)
    test_run_ids = set(metadata["test_run_ids"])
    test_rows = [row for row in rows if row["run_id"] in test_run_ids]
else:
    print("舊模型未記錄資料來源與測試切分，以下僅示範載入及預測，無法確認樣本是否未參與訓練。")
    rows = load_dataset(REPO_DIR / "data" / "qkd_dataset_diagnostic_10000.csv")
    _, test_rows = train_test_split(rows, test_size=0.25, seed=42)

if not test_rows:
    raise ValueError("No held-out runs found in the demo dataset.")
# 五類資料優先展示光纖老化；三類模型使用測試集的其他 run。
run_id = next(
    (row["run_id"] for row in test_rows if row["label"] == "fiber_aging"),
    test_rows[0]["run_id"],
)
run = [row for row in test_rows if row["run_id"] == run_id]

# 10 特徵模型：預測前先對「整個 run」算相對特徵（需要 block 0 當基準、只用過去與現在的區塊）
if uses_relative:
    add_relative_features(run)

print()
print(f"run {run_id} 的逐區塊診斷：")
for row in sorted(run, key=lambda row: row["block_index"]):
    predicted = model.predict_one(row)
    print(f"  block {row['block_index']}：預測 {DISPLAY_NAME[predicted]}（真實 {DISPLAY_NAME[row['label']]}）")
