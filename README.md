# QKD-AI

本專案模擬 BB84 量子金鑰分配鏈路，提供可觀測的區塊統計給 B 組分析通道劣化與攔截重送。A 組負責模擬與資料品質；模型訓練及評估由 B 組負責。分組與整體目標見[專案總覽](docs/project-overview.md)。

## 專案結構

```text
QKD-AI/
  data/
    qkd_dataset.csv
    qkd_truth.csv          # 本機產生，不提交
  docs/
    a-group-work-plan.md
    bb84-basic-principle.md
    csv-format.md
    project-overview.md
    meeting-notes/
      2026-10-09.md
  models/
    decision_tree_qkd.pkl  # 本機產生，不提交
  src/
    a_group/
      generate_bb84_dataset.py
    b_group/
      train_decision_tree.py
  tests/
    test_csv_v1.py
```

## 執行環境

使用 Python 3.10 以上版本；目前模擬器、交接驗收腳本和測試都只需標準函式庫。以下命令在專案根目錄執行。

## A 組：產生 CSV v1

A 組輸出給 B 組的 `data/qkd_dataset.csv` 與僅供 A 組驗證的 `data/qkd_truth.csv`。欄位、標籤及真值隔離規則見 [CSV 格式規格](docs/csv-format.md)。10/18 基本情境：

```bash
python src/a_group/generate_bb84_dataset.py --stage basic --seed 42
```

10/25 診斷情境含光纖老化、偏振漂移和部分攔截：

```bash
python src/a_group/generate_bb84_dataset.py --stage diagnostic --seed 42
```

兩種模式使用同一份 CSV v1 欄位；第二個命令會覆寫第一個命令的輸出。預設基本版每類 500 列、共 1,500 列；診斷版五類各約 500 列。可用 `--samples-per-label`、`--photons-per-sample`、`--run-size` 調整規模；固定參數與種子可重現資料。`data/qkd_dataset.csv` 可納入 Git 供 B 組取得，真值檔被忽略，**不要加入訓練資料或提交**。

驗證模擬與資料介面：

```bash
python -m unittest discover -s tests -v
```

## B 組：交接驗收

現有決策樹腳本只用來確認 CSV 能讀取、按完整 `run_id` 切分並印出混淆矩陣；正式特徵工程和模型比較由 B 組進行。

```bash
python src/b_group/train_decision_tree.py
```

預設讀取 `data/qkd_dataset.csv`，印出分類結果，並於本機輸出 `models/decision_tree_qkd.pkl`。分類準確率不能證明模型能在真實 QKD 鏈路可靠辨認竊聽；必須與只看 QBER 的基準比較。

## 文件與協作

- [A 組工作計畫](docs/a-group-work-plan.md)：里程碑、物理假設與驗收。
- [BB84 基本物理原理](docs/bb84-basic-principle.md)：協定入門。
- [10/9 會議紀錄](docs/meeting-notes/2026-10-09.md)：原始分工。

GitHub repo 是共同來源；欄位與標籤以 [CSV 格式規格](docs/csv-format.md) 為準。A、B 組確認後才將樣本檔作為正式交接版本；新功能先開 issue，避免不同工具各自修改規格。
