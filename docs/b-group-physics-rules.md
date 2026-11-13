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