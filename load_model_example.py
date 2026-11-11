import pickle
import sys
from pathlib import Path

REPO_DIR = Path(__file__).parent
# 讓 Python 也去 src/b_group 找 qkd_model.py
sys.path.insert(0, str(REPO_DIR / "src" / "b_group"))

with open(REPO_DIR / "models" / "decision_tree_qkd.pkl", "rb") as f:
    bundle = pickle.load(f)

sample = {
    "qber": 0.125,
    "z_error_rate": 0.107143,
    "x_error_rate": 0.15,
    "click_rate": 0.47,
    "test_count": 33,
    "z_test_count": 23,
    "x_test_count": 10,
}
model = bundle["model"]
print(bundle["labels"])
print(bundle["feature_columns"])
result = model.predict_one(sample)
print("預測結果：", result)