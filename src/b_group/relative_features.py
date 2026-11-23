"""B 組相對基準特徵：只用同一個 run 的「過去和現在」區塊，不用未來的區塊。"""

from collections import defaultdict

RELATIVE_COLUMNS = ["click_ratio", "cum_z", "cum_x"]


def add_relative_features(rows):
    """在每一筆 row 加上 click_ratio、cum_z、cum_x 三個特徵（直接修改 rows）。"""
    # 1. 依 run 分組
    runs = defaultdict(list)
    for row in rows:
        runs[row["run_id"]].append(row)

    for run in runs.values():
        # 2. 依時間順序排好，第一個就是基準
        run.sort(key=lambda row: row["block_index"])
        base = run[0]
        if base["block_index"] != 0:
            raise ValueError("每個 run 排第一的區塊必須是 block 0")

        # 3. 一個 block 一個 block 往後走，邊走邊累積
        z_err = x_err = z_cnt = x_cnt = 0
        for row in run:
            row["click_ratio"] = row['click_rate'] / base['click_rate'] if base['click_rate'] > 0 else 1.0
            if row["block_index"] > 0:
                z_err += round(row["z_error_rate"] * row["z_test_count"])
                x_err += round(row["x_error_rate"] * row["x_test_count"])
                z_cnt += row["z_test_count"]
                x_cnt += row["x_test_count"]
            row["cum_z"] = z_err / z_cnt if z_cnt else 0.0
            row["cum_x"] = x_err / x_cnt if x_cnt else 0.0
    return rows