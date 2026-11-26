"""畫出「竊聽檢出率 vs. 攔截比例 f」曲線。

流程：
1. 讀 A 組交接的 data/qkd_dataset_diagnostic.csv（五類診斷版，每區塊約 1,000 光子），
   用與 train_decision_tree.py 相同的 run 切分與參數，訓練完整模型（原 7 特徵）和只看 QBER 的基準模型。
   注意：這裡的完整模型是 7 特徵版，不是展示用的 10 特徵模型。相對基準特徵需要同一 run 的 block 0，
   而本程式產生的是彼此獨立的單一區塊，所以無法套用相對特徵。
2. 對每個指定的 f，呼叫 A 組的 simulate_sample 產生新的攔截區塊。
   通道條件與 diagnostic 模式的正常區塊相同：距離 5–25 km、0.2 dB/km、
   背景去極化雜訊 0–2%。
3. 檢出率 = 被模型判成 intercept_resend 的比例。f = 0 那一點代表沒有 Eve，
   所以它的值就是誤報率。
4. 輸出曲線圖 (PNG) 與數據表 (CSV)，附 95% Wilson 信賴區間。

f 只用來決定 x 軸位置，從頭到尾不會當成模型特徵。
另外產生的測試區塊與訓練資料完全獨立，不會有資料洩漏。

執行（在專案根目錄）：
    python src/b_group/plot_detection_vs_f.py
需要 matplotlib（見 requirements-validation.txt）；沒有的話加 --no-plot，只輸出 CSV。
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR / "src" / "b_group"))
sys.path.insert(0, str(REPO_DIR / "src" / "a_group"))

from generate_bb84_dataset import (  # noqa: E402
    BASE_ATTENUATION_DB_PER_KM,
    BlockConditions,
    fiber_transmittance,
    simulate_sample,
)
from qkd_model import SimpleDecisionTree  # noqa: E402
from train_decision_tree import FEATURE_COLUMNS, load_dataset, train_test_split  # noqa: E402

ATTACK_LABEL = "intercept_resend"
# 預設 f 網格：小 f 區域（偵測極限所在）取密一點，大 f 區域取疏一點。
DEFAULT_FRACTIONS = [round(0.02 * i, 2) for i in range(16)] + [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """比例的 95% Wilson 信賴區間；在比例接近 0 或 1 時比常態近似穩定。"""

    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def simulate_blocks(
    fraction: float,
    count: int,
    photons: int,
    test_fraction: float,
    rng: random.Random,
) -> list[dict[str, float]]:
    """在攔截比例 fraction 下產生 count 個獨立區塊，只回傳可觀測欄位。"""

    rows: list[dict[str, float]] = []
    failed_attempts = 0
    while len(rows) < count:
        distance_km = rng.uniform(5, 25)
        conditions = BlockConditions(
            scenario=ATTACK_LABEL if fraction > 0 else "normal",
            label=ATTACK_LABEL if fraction > 0 else "normal",
            noise_level=rng.uniform(0, 0.02),
            transmittance=fiber_transmittance(distance_km, BASE_ATTENUATION_DB_PER_KM),
            distance_km=distance_km,
            attenuation_db_per_km=BASE_ATTENUATION_DB_PER_KM,
            intercept_fraction=fraction,
        )
        block_seed = rng.randrange(2**63)
        try:
            observed, _truth = simulate_sample(
                0, 0, 0, conditions, photons, test_fraction,
                block_seed, 0, "diagnostic", random.Random(block_seed),
            )
        except ValueError as error:
            failed_attempts += 1
            if failed_attempts >= 50:
                raise ValueError(
                    "Could not form a valid test subset after 50 attempts; "
                    "increase --photons-per-sample."
                ) from error
            continue
        failed_attempts = 0
        rows.append({feature: float(observed[feature]) for feature in FEATURE_COLUMNS})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot detection rate versus intercept fraction f.")
    parser.add_argument("--input", type=Path, default=REPO_DIR / "data" / "qkd_dataset_diagnostic.csv",
                        help="訓練用的 CSV v1（預設五類 diagnostic 資料，每區塊約 1,000 光子）")
    parser.add_argument("--fractions", type=float, nargs="+", default=DEFAULT_FRACTIONS,
                        help="要掃描的 f 值，例如 --fractions 0 0.05 0.1 0.2 0.5 1")
    parser.add_argument("--blocks-per-f", type=int, default=400, help="每個 f 產生幾個測試區塊")
    parser.add_argument("--photons-per-sample", type=int, default=1000,
                        help="每區塊光子數；應與訓練資料相同，否則 test_count 類特徵會失真")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42, help="訓練切分種子，與 train_decision_tree.py 一致")
    parser.add_argument("--sweep-seed", type=int, default=2026, help="產生掃描測試區塊的種子")
    parser.add_argument("--output-png", type=Path, default=REPO_DIR / "results" / "detection_vs_f.png")
    parser.add_argument("--output-csv", type=Path, default=REPO_DIR / "results" / "detection_vs_f.csv")
    parser.add_argument("--no-plot", action="store_true", help="只輸出 CSV，不需要 matplotlib")
    args = parser.parse_args()
    if args.blocks_per_f <= 0:
        parser.error("--blocks-per-f must be positive.")
    if args.photons_per_sample < 2:
        parser.error("--photons-per-sample must be at least 2.")
    if not 0 < args.test_fraction < 1:
        parser.error("--test-fraction must be between 0 and 1.")
    if any(not 0 <= fraction <= 1 for fraction in args.fractions):
        parser.error("f must be between 0 and 1.")
    return args


def main() -> None:
    args = parse_args()

    # 1. 用與 train_decision_tree.py 相同的切分和參數訓練兩個模型。
    rows = load_dataset(args.input)
    train_rows, _ = train_test_split(rows, args.test_size, args.seed)
    trained_photons = {row["test_count"] for row in train_rows}
    models = {}
    for name, features in (("Full model", FEATURE_COLUMNS), ("QBER-only baseline", ["qber"])):
        model = SimpleDecisionTree(args.max_depth, args.min_samples_leaf, features)
        model.fit(train_rows)
        models[name] = model
    print(f"Trained on {len(train_rows)} rows from {args.input}")
    print(f"(training test_count range: {min(trained_photons):.0f}-{max(trained_photons):.0f})")

    # 2. 對每個 f 產生新區塊並計算檢出率。
    rng = random.Random(args.sweep_seed)
    results: list[dict[str, float]] = []
    print(f"\n{'f':>6}{'QBER':>9}" + "".join(f"{name:>22}" for name in models))
    for fraction in sorted(set(args.fractions)):
        if not 0 <= fraction <= 1:
            raise ValueError("f must be between 0 and 1.")
        blocks = simulate_blocks(fraction, args.blocks_per_f, args.photons_per_sample,
                                 args.test_fraction, rng)
        record: dict[str, float] = {
            "intercept_fraction": fraction,
            "blocks": len(blocks),
            "theoretical_qber": fraction / 4,
            "mean_observed_qber": sum(b["qber"] for b in blocks) / len(blocks),
        }
        line = f"{fraction:6.2f}{record['mean_observed_qber']:9.4f}"
        for name, model in models.items():
            predictions = model.predict(blocks)
            caught = sum(p == ATTACK_LABEL for p in predictions)
            flagged = sum(p != "normal" for p in predictions)
            low, high = wilson_interval(caught, len(blocks))
            key = "full" if name == "Full model" else "qber_only"
            record[f"{key}_detection_rate"] = caught / len(blocks)
            record[f"{key}_ci_low"] = low
            record[f"{key}_ci_high"] = high
            record[f"{key}_flagged_abnormal_rate"] = flagged / len(blocks)
            line += f"{caught / len(blocks):12.3f} [{low:.2f},{high:.2f}]"
        results.append(record)
        print(line)
    no_eve = next((record for record in results if record["intercept_fraction"] == 0), None)
    if no_eve is not None:
        print("\nf = 0 那一列沒有 Eve，檢出率即為誤報率。")

    # 3. 輸出數據表。
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        for record in results:
            writer.writerow({k: round(v, 6) if isinstance(v, float) else v for k, v in record.items()})
    print(f"Saved table to {args.output_csv}")

    if args.no_plot:
        return

    # 4. 畫圖。
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fractions = [r["intercept_fraction"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.axvspan(0.08, 0.24, color="0.92", zorder=0, label="Partial-intercept range in training (8–24%)")
    styles = {"full": ("Full model (7 features)", "#1f5fa8", "o"),
              "qber_only": ("QBER-only baseline", "#d0641c", "s")}
    for key, (label, color, marker) in styles.items():
        rate = [r[f"{key}_detection_rate"] for r in results]
        low = [r[f"{key}_ci_low"] for r in results]
        high = [r[f"{key}_ci_high"] for r in results]
        ax.fill_between(fractions, low, high, color=color, alpha=0.15, linewidth=0)
        ax.plot(fractions, rate, marker=marker, markersize=4, color=color, label=label)
    ax.set_xlabel("Intercept fraction f  (theoretical QBER = f / 4)")
    ax.set_ylabel("Detection rate  (predicted intercept_resend)")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"Eavesdropping detection vs. intercept fraction "
                 f"({args.blocks_per_f} blocks per f, {args.photons_per_sample} photons/block)",
                 fontsize=10)
    if no_eve is not None:
        ax.annotate("f = 0 (no Eve): false-alarm rate",
                    xy=(0, no_eve["full_detection_rate"]), xytext=(0.32, 0.12),
                    fontsize=8, color="0.3", arrowprops={"arrowstyle": "->", "color": "0.5"})
    ax.legend(loc="lower right", fontsize=8)

    secondary = ax.secondary_xaxis("top", functions=(lambda f: f / 4, lambda q: q * 4))
    secondary.set_xlabel("Theoretical QBER")

    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png)
    print(f"Saved figure to {args.output_png}")


if __name__ == "__main__":
    main()
