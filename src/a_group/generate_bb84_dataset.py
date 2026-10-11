"""Generate observable BB84 block data and separate simulation ground truth."""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path


# B 組只能使用這份資料中的公開測試統計；欄位定義見 docs/csv-format.md。
FIELD_NAMES = [
    "sample_id",
    "run_id",
    "block_index",
    "sent_count",
    "click_count",
    "sifted_count",
    "test_count",
    "z_test_count",
    "x_test_count",
    "test_error_count",
    "z_test_error_count",
    "x_test_error_count",
    "qber",
    "z_error_rate",
    "x_error_rate",
    "click_rate",
    "basis_match_rate",
    "label",
]
TRUTH_FIELD_NAMES = [
    "sample_id",
    "run_id",
    "block_index",
    "seed",
    "dataset_seed",
    "stage",
    "scenario",
    "noise_level",
    "distance_km",
    "attenuation_db_per_km",
    "aging_delta_db_per_km",
    "phase_angle_rad",
    "intercept_fraction",
    "channel_transmittance",
    "eve_present",
    "depolarized_count",
]

LABELS = (
    "normal",
    "depolarizing_noise",
    "fiber_aging",
    "polarization_drift",
    "intercept_resend",
)
BASIC_LABELS = ("normal", "depolarizing_noise", "intercept_resend")
BASES = ("Z", "X")
LOSS_RANGE = (0.0, 0.12)
BASE_ATTENUATION_DB_PER_KM = 0.2


@dataclass(frozen=True)
class BlockConditions:
    """保存單一區塊的通道與攻擊設定；只寫入 A 組真值檔。"""

    scenario: str
    label: str
    noise_level: float
    transmittance: float
    distance_km: float = 0.0
    attenuation_db_per_km: float = 0.0
    aging_delta_db_per_km: float = 0.0
    phase_angle_rad: float = 0.0
    intercept_fraction: float = 0.0


def random_bit(rng: random.Random) -> int:
    """隨機產生 Alice、Bob 或 Eve 使用的 bit。"""

    return rng.randint(0, 1)


def random_basis(rng: random.Random) -> str:
    """從 BB84 的 Z / X 基底中隨機選擇一組。"""

    return rng.choice(BASES)


def rate(count: int, total: int) -> float:
    """計算已驗證分母非零的比例。"""

    return round(count / total, 6)


def fiber_transmittance(distance_km: float, attenuation_db_per_km: float) -> float:
    """由距離與 dB/km 衰減係數計算理想光纖抵達機率。"""

    if distance_km < 0 or attenuation_db_per_km < 0:
        raise ValueError("Distance and attenuation must be non-negative.")
    return 10 ** (-attenuation_db_per_km * distance_km / 10)


def simulate_sample(
    sample_id: int,
    run_id: int,
    block_index: int,
    conditions: BlockConditions,
    photons: int,
    test_fraction: float,
    seed: int,
    dataset_seed: int,
    stage: str,
    rng: random.Random,
) -> tuple[dict[str, int | float | str], dict[str, int | float | str]]:
    """模擬一個傳輸區塊，分別回傳可觀測資料及模擬真值。"""

    sifted: list[tuple[str, bool]] = []
    click_count = 0
    depolarized_count = 0
    phase_flip_probability = math.sin(conditions.phase_angle_rad / 2) ** 2

    for _ in range(photons):
        # Alice 準備態，Bob 獨立選擇測量基底；損失的光子沒有點擊事件。
        alice_bit = random_bit(rng)
        alice_basis = random_basis(rng)
        bob_basis = random_basis(rng)
        if rng.random() >= conditions.transmittance:
            continue
        click_count += 1

        transmitted_bit = alice_bit
        transmitted_basis = alice_basis

        # Eve 攔截後以自己選的基底重送，錯基底測量得到隨機結果。
        if rng.random() < conditions.intercept_fraction:
            eve_basis = random_basis(rng)
            transmitted_bit = alice_bit if eve_basis == alice_basis else random_bit(rng)
            transmitted_basis = eve_basis

        # Z 軸相位漂移只改變 X 基底的同基底測量機率；去極化另行作用。
        if rng.random() < conditions.noise_level:
            bob_bit = random_bit(rng)
            depolarized_count += 1
        elif bob_basis == transmitted_basis:
            phase_flip = bob_basis == "X" and rng.random() < phase_flip_probability
            bob_bit = transmitted_bit ^ int(phase_flip)
        else:
            bob_bit = random_bit(rng)

        # 只有 Alice / Bob 基底相同的點擊事件能進入篩選後資料。
        if bob_basis == alice_basis:
            sifted.append((alice_basis, bob_bit != alice_bit))

    if click_count == 0 or len(sifted) < 2:
        raise ValueError("Not enough detected and sifted bits for parameter estimation.")

    # 從篩選後資料抽測試位元；若任一基底沒有測試樣本，重抽而非填 0。
    test_count = min(len(sifted), max(2, round(len(sifted) * test_fraction)))
    for _ in range(20):
        tested = rng.sample(sifted, test_count)
        z_test_count = sum(basis == "Z" for basis, _ in tested)
        x_test_count = test_count - z_test_count
        if z_test_count and x_test_count:
            break
    else:
        raise ValueError("Test subset does not contain both bases.")

    test_error_count = sum(is_error for _, is_error in tested)
    z_test_error_count = sum(is_error for basis, is_error in tested if basis == "Z")
    x_test_error_count = test_error_count - z_test_error_count

    observed: dict[str, int | float | str] = {
        "sample_id": sample_id,
        "run_id": run_id,
        "block_index": block_index,
        "sent_count": photons,
        "click_count": click_count,
        "sifted_count": len(sifted),
        "test_count": test_count,
        "z_test_count": z_test_count,
        "x_test_count": x_test_count,
        "test_error_count": test_error_count,
        "z_test_error_count": z_test_error_count,
        "x_test_error_count": x_test_error_count,
        "qber": rate(test_error_count, test_count),
        "z_error_rate": rate(z_test_error_count, z_test_count),
        "x_error_rate": rate(x_test_error_count, x_test_count),
        "click_rate": rate(click_count, photons),
        "basis_match_rate": rate(len(sifted), click_count),
        "label": conditions.label,
    }
    truth: dict[str, int | float | str] = {
        "sample_id": sample_id,
        "run_id": run_id,
        "block_index": block_index,
        "seed": seed,
        "dataset_seed": dataset_seed,
        "stage": stage,
        "scenario": conditions.scenario,
        "noise_level": round(conditions.noise_level, 6),
        "distance_km": round(conditions.distance_km, 6),
        "attenuation_db_per_km": round(conditions.attenuation_db_per_km, 6),
        "aging_delta_db_per_km": round(conditions.aging_delta_db_per_km, 6),
        "phase_angle_rad": round(conditions.phase_angle_rad, 6),
        "intercept_fraction": round(conditions.intercept_fraction, 6),
        "channel_transmittance": round(conditions.transmittance, 6),
        "eve_present": int(conditions.intercept_fraction > 0),
        "depolarized_count": depolarized_count,
    }
    return observed, truth


def generate_dataset(
    samples_per_label: int,
    photons_per_sample: int,
    run_size: int = 10,
    test_fraction: float = 0.2,
    seed: int = 42,
    stage: str = "basic",
) -> tuple[list[dict[str, int | float | str]], list[dict[str, int | float | str]]]:
    """產生基本或診斷情境；同一 run 的區塊共用基準通道。"""

    if samples_per_label <= 0 or photons_per_sample <= 0 or run_size <= 0:
        raise ValueError("Sample, photon and run counts must be positive.")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    if stage not in ("basic", "diagnostic"):
        raise ValueError("stage must be basic or diagnostic.")
    if stage == "diagnostic" and run_size < 2:
        raise ValueError("diagnostic stage needs --run-size of at least 2.")

    rng = random.Random(seed)
    paired_rows: list[tuple[dict[str, int | float | str], dict[str, int | float | str]]] = []
    run_id = 0

    def add_block(current_run: int, block_index: int, conditions: BlockConditions) -> None:
        # 各區塊使用獨立種子，可用真值檔的參數單獨重現該區塊。
        for _ in range(50):
            block_seed = rng.randrange(2**63)
            try:
                paired_rows.append(simulate_sample(
                    0, current_run, block_index, conditions, photons_per_sample,
                    test_fraction, block_seed, seed, stage, random.Random(block_seed),
                ))
                return
            except ValueError:
                continue
        raise ValueError("Could not form a valid test subset; increase --photons-per-sample.")

    if stage == "basic":
        # 三類共用相同損耗分布；完整攔截重送是此階段的基準攻擊。
        for label in BASIC_LABELS:
            for start in range(0, samples_per_label, run_size):
                run_id += 1
                block_size = min(run_size, samples_per_label - start)
                noise = rng.uniform(0.04, 0.12) if label == "depolarizing_noise" else rng.uniform(0, 0.02)
                transmittance = 1 - rng.uniform(*LOSS_RANGE)
                for block_index in range(block_size):
                    add_block(run_id, block_index, BlockConditions(
                        scenario=label, label=label, noise_level=noise,
                        transmittance=transmittance,
                        intercept_fraction=float(label == "intercept_resend"),
                    ))
    else:
        baseline_count = 0
        for label in LABELS[1:]:
            remaining = samples_per_label
            while remaining:
                run_id += 1
                active_count = min(run_size - 1, remaining)
                distance_km = rng.uniform(5, 25)
                baseline_noise = rng.uniform(0, 0.02)
                active_noise = rng.uniform(0.04, 0.12) if label == "depolarizing_noise" else baseline_noise
                aging_delta = rng.uniform(0.02, 0.08) if label == "fiber_aging" else 0.0
                phase_max = rng.uniform(0.5, 0.8) if label == "polarization_drift" else 0.0
                intercept_fraction = (
                    1.0 if rng.random() < 0.2 else rng.uniform(0.08, 0.24)
                ) if label == "intercept_resend" else 0.0

                # 每個故障 run 先有一個正常基準區塊，之後擾動逐步或立即出現。
                for block_index in range(active_count + 1):
                    is_baseline = block_index == 0
                    progress = block_index / active_count
                    attenuation = BASE_ATTENUATION_DB_PER_KM + (
                        aging_delta * progress if not is_baseline else 0.0
                    )
                    phase = phase_max * progress if not is_baseline else 0.0
                    add_block(run_id, block_index, BlockConditions(
                        scenario=label,
                        label="normal" if is_baseline else label,
                        noise_level=baseline_noise if is_baseline else active_noise,
                        transmittance=fiber_transmittance(distance_km, attenuation),
                        distance_km=distance_km,
                        attenuation_db_per_km=attenuation,
                        aging_delta_db_per_km=aging_delta,
                        phase_angle_rad=phase,
                        intercept_fraction=0.0 if is_baseline else intercept_fraction,
                    ))
                baseline_count += 1
                remaining -= active_count

        # 補足正常類目標筆數；極小資料集可能因每個 run 的基準列而超額。
        normal_needed = max(0, samples_per_label - baseline_count)
        for start in range(0, normal_needed, run_size):
            run_id += 1
            block_size = min(run_size, normal_needed - start)
            distance_km = rng.uniform(5, 25)
            noise = rng.uniform(0, 0.02)
            conditions = BlockConditions(
                scenario="normal", label="normal", noise_level=noise,
                transmittance=fiber_transmittance(distance_km, BASE_ATTENUATION_DB_PER_KM),
                distance_km=distance_km,
                attenuation_db_per_km=BASE_ATTENUATION_DB_PER_KM,
            )
            for block_index in range(block_size):
                add_block(run_id, block_index, conditions)

    # run_id 也打散，避免編號大小本身透露標籤所在的產生順序。
    shuffled_run_ids = list(range(1, run_id + 1))
    rng.shuffle(shuffled_run_ids)
    for observed, truth in paired_rows:
        new_run_id = shuffled_run_ids[int(observed["run_id"]) - 1]
        observed["run_id"] = new_run_id
        truth["run_id"] = new_run_id

    # 一起打散資料與真值，再分配 sample_id，確保兩檔永遠可一對一對應。
    rng.shuffle(paired_rows)
    for sample_id, (observed, truth) in enumerate(paired_rows, start=1):
        observed["sample_id"] = sample_id
        truth["sample_id"] = sample_id

    return [pair[0] for pair in paired_rows], [pair[1] for pair in paired_rows]


def write_csv(rows: list[dict[str, int | float | str]], output: Path, fields: list[str]) -> None:
    """依固定欄位順序寫出 UTF-8 CSV。"""

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """定義 A 組資料產生器的命令列參數。"""

    parser = argparse.ArgumentParser(description="Generate BB84 CSV v1 data.")
    parser.add_argument("--stage", choices=("basic", "diagnostic"), default="basic")
    parser.add_argument("--samples-per-label", type=int, default=500)
    parser.add_argument("--photons-per-sample", type=int, default=1000)
    parser.add_argument("--run-size", type=int, default=10)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=Path("data/qkd_dataset.csv"))
    parser.add_argument("--truth-output", type=Path, default=Path("data/qkd_truth.csv"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """產生可交接資料和獨立真值檔，並印出結果摘要。"""

    args = parse_args()
    if args.output.resolve() == args.truth_output.resolve():
        raise ValueError("Dataset and truth output paths must be different.")

    rows, truth_rows = generate_dataset(
        args.samples_per_label, args.photons_per_sample,
        args.run_size, args.test_fraction, args.seed, args.stage,
    )
    write_csv(rows, args.output, FIELD_NAMES)
    write_csv(truth_rows, args.truth_output, TRUTH_FIELD_NAMES)

    print(f"Wrote {len(rows)} observable rows to {args.output}")
    print(f"Wrote {len(truth_rows)} truth rows to {args.truth_output}")
    print(f"Stage: {args.stage}")
    print(f"Labels: {', '.join(sorted({row['label'] for row in rows}))}")


if __name__ == "__main__":
    main()
