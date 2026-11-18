# A 組資料與驗證交接

日期為 2026-11-17。原始專案版本為 `1d5e4b2`，本次加入固定基底攔截選項、掃描資料產生程式與交接驗證程式。執行環境為 Python 3.13.9、NumPy 2.3.5、scikit-learn 1.7.2。核心模擬繼續使用標準函式庫逐光子模擬。

## 交付內容

| 檔案 | 用途 |
| --- | --- |
| `data/qkd_dataset_diagnostic_10000.csv` | 五類診斷資料，共 2,500 筆，每類 500 筆，每區塊 10,000 光子，種子 42。供 B 組重訓。 |
| `data/qkd_dataset_intercept_sweep_10000.csv` | 攔截掃描資料，共 3,300 筆、1,100 個 run。每 run 有 1 個正常基準和 2 個評估區塊。供情境評估。 |
| `docs/a-group-intercept-detection.csv` | 已固定告警規則的 run 檢出率、誤報率、95% Wilson 信賴區間。 |
| `docs/a-group-intercept-detection.png` | C 組可使用的檢出率曲線。 |
| `docs/a-group-validation-results.json` | 資料檢查、檔案 SHA-256、五折混淆矩陣與各類召回率。 |

CSV 沿用 v1 欄位及標籤。兩份獨立真值檔以 `qkd_truth_` 開頭留在 A 組本機，交接包僅含公開資料。掃描設定與標籤只供評估分組。B 組模型特徵取自可觀測統計。兩份資料的 ID 各自獨立，合併時請先重新編排 ID 以保留 run 邊界。

診斷資料的測試位元最小值、中位數、最大值為 210、483、810。每基底都有樣本。中位區塊的一個錯誤使 QBER 增加約 0.207 個百分點。

## 物理假設與展示文字

採理想單光子、理想偵測器與獨立通道作用。光纖傳輸率為 `10 ** (-alpha * distance_km / 10)`。展示範圍為 5–25 km，基準衰減為 0.2 dB/km，老化增量上限為 0.08 dB/km。理想偵測器下純損耗降低點擊率，QBER 由接收到的光子估計。

暗計數檢查採偵測效率 10%、暗計數機率 10^-5。在 25 km、衰減 0.28 dB/km 時，暗計數 QBER 約 0.0501%。這個數字對應指定的效率、暗計數與距離條件。

請將漂移情境顯示為「Z 軸相位型漂移」。相位角 theta 在理想無雜訊時使 X 基底錯誤率為 `sin(theta / 2) ** 2`，Z 基底錯誤率為零。這是固定旋轉軸的特例。CSV 標籤仍沿用 `polarization_drift` 以維持既有讀取介面。

Eve 隨機選 Z/X 基底攔截比例 f，在無其他雜訊時的期望 QBER 為 f/4。固定 Z 基底攔截時，Z 基底錯誤率為零、X 基底為 f/2、總 QBER 為 f/4，呈現與相位型漂移相似的基底特徵。

C 組可使用以下文字。

> AI 協助診斷光纖老化與 Z 軸相位型漂移的可能成因。QBER 異常一律保守視為可能竊聽。漂移預測仍須保留安全警示，因固定 Z 基底攔截也可能呈現相同特徵。

去極化和部分攔截在目前參數範圍下高度重疊，展示請同時呈現混淆矩陣與 QBER 告警。C 組負責將文字與圖表整合進簡報、README 和系統畫面。B 組規則表請採用同一原則。

## A 組實際 CSV 的模型核對

本次以 A 組逐光子產生的高光子數診斷 CSV 計算。三種特徵組使用相同五折 GroupKFold，按完整 run 切分。模型為 HistGradientBoostingClassifier，max_iter=150、learning_rate=0.08、random_state=0。這是 A 組交接驗證，B 組正式模型交付可依此重跑。

| 特徵 | 五類準確率 |
| --- | --- |
| 僅 QBER | 39.88% |
| 原有 7 特徵 | 53.44% |
| 加相對基準特徵 | 71.40% |

模型核對使用選用套件 numpy、scikit-learn、matplotlib，版本列於 `requirements-validation.txt`。相對特徵已實作於 `src/a_group/handoff_features.py`，定義與老師附件一致，A 組交接驗證不再依賴附件資料夾。完整重跑老師的 `02`、`03` 仍使用原始審查附件。

| 類別 | 相對特徵模型召回率 |
| --- | --- |
| 正常 | 73.4% |
| 去極化 | 50.6% |
| 老化 | 91.2% |
| Z 軸相位型漂移 | 84.8% |
| 攔截重送 | 57.0% |

相對特徵模型將其他類別誤判為攔截的比例為 11.95%。這是五類分類輸出的統計，QBER 告警曲線另有校準和操作點。

老師附件的 `02`、`03` 已在 Anaconda 環境跑完。其資料由附件 surrogate.py 產生。`02` 在 1,000／10,000／100,000 光子下的原 7 特徵準確率為 40.2%／55.0%／62.1%，相對特徵為 61.8%／74.2%／81.0%，去極化對部分攔截 AUC 為 0.52／0.52／0.57。`03` 在 10,000 光子時，固定 Z 攔截約 98% 被判成漂移。老師替身資料和本組逐光子資料分別報告。

## 攔截掃描與告警操作點

掃描 f 為 0、0.02、0.04、0.08、0.12、0.16、0.24、0.4、0.6、0.8、1。兩種攻擊各自每個比例 50 個 run，每 run 3 區塊。每區塊送出 10,000 光子，公開測試比例 20%，資料種子為 20261117。背景去極化 p 在 0–0.02 間均勻取樣，距離在 5–25 km 間均勻取樣。

50 組通道設定在各 f 間共用，便於成對比較，區塊的亂數種子獨立。每組前 20 個通道保留作校準，後 30 個通道作評估。只使用 f=0 隨機基底組的 20 個校準 run 選門檻。各曲線點間有成對通道相關性，圖中區間表示各點的二項比例不確定度。

每 run 的告警分數為兩個活動區塊的最大 QBER。校準樣本的經驗 95 百分位數得到門檻 0.012270，當分數嚴格大於 1.227% 時告警。這是指定背景條件下的實驗操作點，實際系統須依自己的正常通道重新校準。校準用 95 百分位數也會受有限校準樣本影響。

| f | 隨機基底告警率 | 固定 Z 基底告警率 |
| --- | --- | --- |
| 0 | 0/30（0%） | 1/30（3.33%） |
| 0.02 | 13/30（43.33%） | 15/30（50%） |
| 0.04 | 27/30（90%） | 26/30（86.67%） |
| 0.08 至 1 | 各測點 30/30（100%） | 各測點 30/30（100%） |

f=0 的結果是誤報率。0/30 的 95% Wilson 區間為 0–11.35%，1/30 為 0.59–16.67%。30/30 的區間為 88.65–100%，展示需一併附上樣本數。去極化與漂移也可能觸發告警，展示說明採「可能竊聽」的保守解讀。

## 重現指令

在專案根目錄執行以下 PowerShell 指令。診斷檔與真值分開指定。

```powershell
$qkdPython = 'D:\Program Files\anaconda\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '2'
& $qkdPython -B src/a_group/generate_bb84_dataset.py --stage diagnostic --seed 42 --photons-per-sample 10000 --output data/qkd_dataset_diagnostic_10000.csv --truth-output data/qkd_truth_diagnostic_10000.csv
& $qkdPython -B -m src.a_group.generate_intercept_sweep
& $qkdPython -B -m src.a_group.verify_handoff
& $qkdPython -B -m unittest discover -s tests -v
```

模型核對使用選用套件 numpy、scikit-learn、matplotlib，版本列於 `requirements-validation.txt`。相對特徵已實作於 `src/a_group/handoff_features.py`，定義與老師附件一致，A 組交接驗證不再依賴附件資料夾。完整重跑老師的 `02`、`03` 仍使用原始審查附件。

## 驗收紀錄與分工

8 項專案測試全數通過，新增固定 Z 攔截的 Z/X 指紋測試。原始 1,000 光子診斷資料仍可用種子 42 逐列重現。兩份新 CSV 通過欄位、計數、比例、唯一 ID、真值一對一配對、基準區塊與 B 組讀取及 run 切分檢查。B 組既有簡易決策樹也已實跑新版 CSV，按 run 保留測試集的準確率為 52.73%，並以另一個 Python 程序載入模型完成預測。

A 組的資料、物理驗證、掃描曲線與交接文件已備妥。B 組接手正式模型、相對特徵整合與規則表更新，C 組接手 README 展示說明、簡報及告警畫面整合。

## 從 GitHub 下載後重跑（不需老師附件或真值檔）

在專案根目錄使用 Python 3.11 以上環境。以下 `python` 可替換成自己的 Python 執行檔路徑。

```powershell
python -m pip install -r requirements-validation.txt
$env:OMP_NUM_THREADS = '2'
python -B -m src.a_group.verify_handoff --public-only
python -B -m unittest discover -s tests -v
```

此模式使用已提交的兩份公開 CSV 及 `docs/a-group-intercept-detection.csv`，檢查資料、重跑五折模型評估並重畫曲線。結果的 `truth_pairing_checked` 為 `false`，表示沒有檢查私有真值配對，不代表真值配對驗證通過。

若要完整重產資料與告警門檻，先依上節指令執行兩支產生器，再執行不帶 `--public-only` 的驗證指令，檢查本機真值配對。真值檔不得提交。核心模擬及既有測試僅使用標準函式庫，不必安裝選用評估套件。

Git 僅對 `data/qkd_dataset_diagnostic_10000.csv` 和 `data/qkd_dataset_intercept_sweep_10000.csv` 新增忽略例外；`data/qkd_truth*.csv` 繼續被忽略。
