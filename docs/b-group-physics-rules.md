# B組物理規則表(五類診斷版)

## 五類平均值表
|label|qber|z_error_rate|x_error_rate|click_rate|
|---|---|---|---|---|                              
|depolarizing_noise    |      0.0394 |       0.0384     |   0.0398 | 0.5112 |
|fiber_aging         |        0.0042  |      0.0038    |    0.0047 | 0.4583|
|intercept_resend     |      0.0659    |    0.0672    |    0.0651 | 0.5154|
|normal             |         0.0048  |  0.0050      |0.0047 | 0.5050|
|polarization_drift|          0.0242   |     0.0039 |       0.0444 | 0.5189|

## 規則表
|類別|指紋|物理原因|判斷規則|
|---|---|---|---|
|depolarizing_noise |  QBER ↑（約 4%），Z ≈ X |  	隨機雜訊，兩個基底一樣受影響	   |    QBER 偏高（但通常低於竊聽）;Z ≈ X，click_rate 正常|
|fiber_aging         | QBER 正常，click_rate ↓ 約 9% |  光纖損耗變大，光子消失但不出錯 |  QBER 正常，但 click_rate 明顯偏低|
|intercept_resend     |QBER ↑↑（約 6.6%），Z ≈ X | Eve 用錯基底量測，兩個基底一樣受影響 | QBER 最高，Z ≈ X，click_rate 正常:部分攔截時會跟去極化雜訊重疊|
|normal                |QBER ≈ 0.5%，Z ≈ X，click ≈ 0.505 |    通道正常  | QBER 低、Z ≈ X、click_rate 正常|
|polarization_drift   |X 錯誤率 ≫ Z 錯誤率    | 偏振轉歪，主要影響 X 基底  |    X 基底錯誤率 ≫ Z 基底錯誤率|

## 限制
竊聽和去極化雜訊的指紋很像（Z ≈ X、click_rate 正常），只差在 QBER 高低；部分攔截會把 QBER 壓低，跟去極化雜訊重疊，所以需要用模型分辨。
以上規則是目前模擬設定下的觀察，物理解讀請 A 組一起核對。

## 混淆矩陣
```
決策樹（7 特徵） 混淆矩陣（列：真實，欄：預測）
                      depolarizing_noise         fiber_aging    intercept_resend              normal  polarization_drift
  depolarizing_noise                  17                  11                  66                  16                  16
         fiber_aging                   7                  54                   5                  53                   7
    intercept_resend                  18                  23                  67                  11                   7
              normal                   7                  42                   7                  63                   3
  polarization_drift                   8                  14                  11                  38                  51

梯度提升（7 特徵） 混淆矩陣（列：真實，欄：預測）
                      depolarizing_noise         fiber_aging    intercept_resend              normal  polarization_drift
  depolarizing_noise                  44                   5                  42                   8                  27
         fiber_aging                   6                  40                   5                  54                  21
    intercept_resend                  38                  13                  44                  15                  16
              normal                  13                  28                   3                  58                  20
  polarization_drift                  12                  12                   8                  34                  56
  ```