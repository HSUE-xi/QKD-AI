# CSV v1 交接規格

本文件是 A 組與 B 組的資料介面。A 組產生 UTF-8 CSV：`data/qkd_dataset.csv` 可供 B 組讀取，`data/qkd_truth.csv` 僅供 A 組驗證。兩檔以 `sample_id` 一對一對應；**不得把真值檔的欄位加入模型輸入**。產生方式見 [README](../README.md)。

## 可觀測資料

每列代表一個 BB84 傳輸區塊。預設發送 1,000 個理想單光子，從基底比對後的位元隨機抽取約 20% 作公開測試。錯誤率只用這些公開測試位元計算。正式欄位順序固定為：

```csv
sample_id,run_id,block_index,sent_count,click_count,sifted_count,test_count,z_test_count,x_test_count,test_error_count,z_test_error_count,x_test_error_count,qber,z_error_rate,x_error_rate,click_rate,basis_match_rate,label
```

| 欄位 | 定義 |
|---|---|
| `sample_id` | 唯一整數 ID，用來對應 A 組真值。 |
| `run_id` | 同一模擬批次的 ID；切分訓練與測試時，整個 run 必須留在同一側。 |
| `block_index` | 同一 run 內從 0 開始的傳輸順序；僅供排序與計算過去資料的趨勢。 |
| `sent_count`、`click_count`、`sifted_count` | 發送、Bob 點擊、基底匹配的次數，依序不增。此版偵測器理想，點擊等於光子抵達。 |
| `test_count`、`z_test_count`、`x_test_count` | 公開測試位元總數及 Z/X 基底數；後兩者相加等於總數。 |
| `test_error_count`、`z_test_error_count`、`x_test_error_count` | 公開測試錯誤總數及 Z/X 錯誤數；後兩者相加等於總數。 |
| `qber` | `test_error_count / test_count`。 |
| `z_error_rate`、`x_error_rate` | 各基底錯誤數除以各基底測試數。 |
| `click_rate` | `click_count / sent_count`。 |
| `basis_match_rate` | `sifted_count / click_count`；主要用於協定驗證。 |
| `label` | 下表的成因類別；是訓練目標，不是特徵。 |

比率介於 0 與 1，輸出到小數第六位。每列必須有 Z 與 X 測試位元；無法估計時重抽，重試失敗則報錯，不以 0 代替未知值。`sample_id`、`run_id`、`block_index`、`label` 均不可直接當成模型特徵。

## 標籤與階段

| `label` | 定義 | 10/18 `basic` | 10/25 `diagnostic` |
|---|---|---|---|
| `normal` | 低擾動、無 Eve；診斷模式亦包含故障前的基準區塊。 | 有 | 有 |
| `depolarizing_noise` | 簡化去極化通道使 Bob 的結果隨機化；不等同所有自然雜訊。 | 有 | 有 |
| `fiber_aging` | 光纖衰減係數逐區塊升高；純損耗本身不造成 bit 錯誤。 | 無 | 有 |
| `polarization_drift` | 簡化的 Z 軸相位／偏振漂移，主要影響 X 基底。 | 無 | 有 |
| `intercept_resend` | Eve 攔截重送；基本模式為完整攔截，診斷模式兼有部分與完整攔截。 | 有 | 有 |

同一 `run_id` 在診斷模式先有一個 `normal` 基準區塊，再進入單一故障或攻擊情境。距離與基準通道設定跨類別重疊，避免絕對點擊率成為固定答案；資料仍會保留無法單靠現有觀測值區分的難例。診斷模式以每個非正常類別約 `--samples-per-label` 個故障區塊為目標，並補足正常區塊；極小資料集的正常筆數可能因每個 run 的基準段而較多。

## A 組真值檔

`data/qkd_truth.csv` 的欄位順序：

```csv
sample_id,run_id,block_index,seed,dataset_seed,stage,scenario,noise_level,distance_km,attenuation_db_per_km,aging_delta_db_per_km,phase_angle_rad,intercept_fraction,channel_transmittance,eve_present,depolarized_count
```

其中 `seed` 是單一區塊的種子，`dataset_seed` 用於重產整份資料；其餘欄位為模擬器設定或內部計數。`scenario` 記錄 run 的預定成因，因此基準區塊的 `label` 仍可為 `normal`。`eve_present` 和 `intercept_fraction` 是答案資訊，不能出現在 B 組資料。真值檔被 Git 忽略，不應提交。

## 交接檢查

1. 公開 CSV 欄名與順序完全一致；兩檔 `sample_id` 唯一且一對一對應。
2. 計數加總及比率可由公開欄位重算；不出現沒有分母卻填 0 的錯誤率。
3. B 組按完整 `run_id` 切分，不能把同一 run 的基準段與故障段拆到不同資料集。
4. 趨勢特徵僅由當下及過去區塊計算；評估時須對照只看 QBER 的基準，並明列誤報與漏報。

若日後誘餌態需要新觀測欄位，須先與 B 組約定 v2；不得變更已交接的 v1 欄位語意。
