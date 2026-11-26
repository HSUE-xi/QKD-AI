# B 組展示模型交付說明（給 C 組）

日期：2026-11-25。展示模型選定為 **10 特徵決策樹**（原 7 特徵＋3 個相對基準特徵）。

## 1. 產生模型檔

`.pkl` 不上傳 GitHub（`.gitignore` 排除），請在專案根目錄自己訓練一次，只需要 Python 標準函式庫，不需要 scikit-learn：

```powershell
python src/b_group/train_decision_tree.py --input data/qkd_dataset_diagnostic_10000.csv --relative
```

輸出 `models/decision_tree_qkd.pkl`。測試集（依 run 切分，`seed=42`）五類準確率約 71.1%、macro-F1 約 0.70。

`.pkl` 內容是一個 dict：

| 鍵 | 內容 |
|---|---|
| `model` | `SimpleDecisionTree`（定義在 `src/b_group/qkd_model.py`） |
| `feature_columns` | 模型需要的 10 個特徵名稱 |
| `labels` | 五類標籤 |
| `relative_features` | `True`：預測前要先算相對基準特徵 |
| `training_metadata` | 訓練資料路徑及 SHA-256、切分種子、測試比例與測試 run ID |

## 2. 輸入特徵規格

模型一次判斷**一個區塊（block）**，但**相對基準特徵需要同一個 run 從 block 0 開始的區塊**：

| 特徵 | 來源 |
|---|---|
| `qber`、`z_error_rate`、`x_error_rate`、`click_rate`、`test_count`、`z_test_count`、`x_test_count` | CSV v1 原有欄位（`load_dataset()` 讀入） |
| `click_ratio` | 這個區塊的 `click_rate` ÷ 同一 run 的 block 0 `click_rate` |
| `cum_z`、`cum_x` | 同一 run 從 block 1 累加到目前區塊的 Z／X 錯誤數 ÷ 測試數（不含 block 0、不用未來區塊） |

規則：

1. 每個 run 必須有 **block 0**（故障前的正常基準），否則 `add_relative_features()` 會報錯
2. 預測第 k 個區塊前，先把同一 run 的 block 0 到 k 交給 `add_relative_features()`；它只用過去與現在的區塊，所以即時系統可以一個區塊一個區塊地算
3. 不要自己重寫特徵算法，直接 `import`：`from relative_features import add_relative_features`
4. 標籤字串（`polarization_drift` 等）不改，只在畫面顯示時轉成中文

## 3. 載入與預測

完整範例：`load_model_example.py`（專案根目錄）。重點：

```python
import pickle, sys
from pathlib import Path

REPO_DIR = Path(__file__).parent
sys.path.insert(0, str(REPO_DIR / "src" / "b_group"))   # 讓 pickle 找得到 qkd_model
from relative_features import add_relative_features

with open(REPO_DIR / "models" / "decision_tree_qkd.pkl", "rb") as f:
    bundle = pickle.load(f)

add_relative_features(run)            # run：同一 run 從 block 0 開始的區塊（dict 的 list）
for row in run:
    print(bundle["model"].predict_one(row))
```

執行 `python load_model_example.py` 會依 `.pkl` 儲存的測試 run ID 選取樣本，並先核對 CSV 的 SHA-256。五類模型優先挑光纖老化的 run，三類模型則挑測試集的其他 run。更改 `--input`、`--seed` 或 `--test-size` 後，重新訓練的模型會保存對應設定。專案內的 CSV 使用相對路徑，方便換電腦重跑；專案外的 CSV 使用絕對路徑。資料內容改變時需還原原檔或重新訓練。

舊 `.pkl` 缺少 `training_metadata` 時仍可載入。範例會明確標示無法確認展示樣本是否曾參與訓練；若要確認測試集來源，請重新訓練一次。

## 4. C 組串 demo 的建議流程

demo 程式的角色是「**取得資料 → 前處理 → 呼叫模型 → 顯示結果**」。畫面可以自己設計，但**中間兩步一定要用 B 組的同一份函式**，訓練和展示的特徵算法才會一致：

| 步驟 | 用什麼 | 能不能自己改 |
|---|---|---|
| 取得資料 | 換成 demo 要展示的資料（建議用 `load_dataset()` 讀，會檢查欄位並轉成數字；展示的 run 從測試集挑） | ✅ 可以換 |
| 前處理 | `add_relative_features(run)`，`run` 要包含同一 run 從 block 0 開始的區塊 | ❌ 不要重寫 |
| 呼叫模型 | 載入 `.pkl`，`bundle["model"].predict_one(row)` | ❌ 不要重寫 |
| 顯示結果 | 標籤轉成第 5 節的顯示文字；畫面樣式自由 | 文字照第 5 節，樣式自由 |

建議做法：

1. 先**原封不動**執行 `python load_model_example.py`，能印出逐區塊診斷，代表環境、模型、特徵都沒問題
2. 以它為基礎，只換掉「挑一個測試集的 run」那一段資料來源，其他步驟照用
3. 不要把 B 組的程式**複製**一份到別處；直接 `import`，B 組之後修正時，`git pull` 就會一起更新

如果自己重寫特徵算法（例如累積錯誤率把 block 0 也算進去），程式**不會報錯**，但特徵數值跟訓練時不同，準確率會默默下降（training-serving skew）。

## 5. 畫面顯示文字（照教授意見的新說法）

| 標籤 | 顯示 |
|---|---|
| `normal` | 正常 |
| `depolarizing_noise` | 去極化雜訊 |
| `fiber_aging` | 光纖老化 |
| `polarization_drift` | Z 軸相位型漂移 |
| `intercept_resend` | QBER 異常（可能竊聽） |

展示說明：

> AI 協助診斷光纖老化與 Z 軸相位型漂移的可能成因。QBER 異常一律保守視為可能竊聽。漂移預測仍須保留安全警示，因固定 Z 基底攔截也可能呈現相同特徵。

限制：去極化雜訊和攔截重送在目前參數下分不開（兩類互相誤判仍多），所以模型判成去極化時，也不能當成安全。

## 6. 結果數字（說明書、影片可用）

見 PR #9：`python src/b_group/compare_same_fpr.py --input data/qkd_dataset_diagnostic_10000.csv` 會印出竊聽偵測表、五類診斷表（準確率、macro-F1、各類 recall）和混淆矩陣。

| 模型 | 準確率 | macro-F1 | 去極化 | 老化 | 竊聽 | 正常 | 漂移 |
|---|---|---|---|---|---|---|---|
| 決策樹（原 7 特徵） | 52.7% | 0.4828 | 62.7% | 5.6% | 43.7% | 74.6% | 78.7% |
| 決策樹（10 特徵，展示模型） | 71.1% | 0.7011 | 32.5% | 92.9% | 73.0% | 77.9% | 79.5% |
