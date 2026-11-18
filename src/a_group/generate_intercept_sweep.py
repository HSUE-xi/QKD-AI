"""Generate photon-level intercept sweeps and an independently calibrated QBER alarm."""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path

from src.a_group.generate_bb84_dataset import (
    FIELD_NAMES, TRUTH_FIELD_NAMES, BlockConditions,
    fiber_transmittance, simulate_sample, write_csv,
)

FRACTIONS = (0.0, 0.02, 0.04, 0.08, 0.12, 0.16, 0.24, 0.4, 0.6, 0.8, 1.0)


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    proportion = successes / total
    z_squared = 1.959964 ** 2
    denominator = 1 + z_squared / total
    center = (proportion + z_squared / (2 * total)) / denominator
    radius = math.sqrt(proportion * (1 - proportion) / total + z_squared / (4 * total ** 2))
    radius *= math.sqrt(z_squared) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--photons', type=int, default=10000)
    parser.add_argument('--runs', type=int, default=50)
    parser.add_argument('--calibration-runs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20261117)
    args = parser.parse_args()
    if args.photons < 10000 or not 1 < args.calibration_runs < args.runs:
        parser.error('Use >= 10000 photons and 1 < calibration-runs < runs.')

    rng = random.Random(args.seed)
    # Independent channels per trial. Reuse their settings across f for paired comparisons.
    channels = [(rng.uniform(5, 25), rng.uniform(0, 0.02)) for _ in range(args.runs)]
    rows, truths, scores = [], [], []
    run_id = 0
    for kind, eve_basis in [('random_basis', None), ('fixed_z', 'Z')]:
        for fraction in FRACTIONS:
            for trial, (distance, noise) in enumerate(channels):
                run_id += 1
                active = []
                for block in range(3):
                    current_fraction = fraction if block else 0.0
                    label = 'intercept_resend' if current_fraction else 'normal'
                    condition = BlockConditions(
                        scenario=f'{kind}_intercept_sweep', label=label,
                        noise_level=noise, transmittance=fiber_transmittance(distance, 0.2),
                        distance_km=distance, attenuation_db_per_km=0.2,
                        intercept_fraction=current_fraction, eve_basis=eve_basis,
                    )
                    block_seed = rng.randrange(2 ** 63)
                    row, truth = simulate_sample(
                        len(rows) + 1, run_id, block, condition, args.photons, 0.2,
                        block_seed, args.seed, 'intercept_sweep', random.Random(block_seed),
                    )
                    rows.append(row)
                    truths.append(truth)
                    if block:
                        active.append(float(row['qber']))
                scores.append(dict(kind=kind, fraction=fraction, trial=trial, score=max(active)))
            print(f'Generated {kind}, f={fraction:g}', flush=True)

    calibration = sorted(s['score'] for s in scores if s['kind'] == 'random_basis'
                         and s['fraction'] == 0 and s['trial'] < args.calibration_runs)
    # Empirical 95th percentile, nearest-rank. Alarm is strictly greater than threshold.
    threshold = calibration[math.ceil(0.95 * len(calibration)) - 1]
    groups = defaultdict(list)
    for score in scores:
        if score['trial'] >= args.calibration_runs:
            groups[(score['kind'], score['fraction'])].append(score['score'] > threshold)
    summary = []
    for (kind, fraction), alarms in groups.items():
        low, high = wilson_interval(sum(alarms), len(alarms))
        summary.append(dict(kind=kind, intercept_fraction=fraction, runs=len(alarms),
                            alarms=sum(alarms), detection_rate=sum(alarms) / len(alarms),
                            ci95_low=low, ci95_high=high, qber_threshold=threshold))
    write_csv(rows, Path('data/qkd_dataset_intercept_sweep_10000.csv'), FIELD_NAMES)
    write_csv(truths, Path('data/qkd_truth_intercept_sweep_10000.csv'), TRUTH_FIELD_NAMES)
    output = Path('docs/a-group-intercept-detection.csv')
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f'Wrote {len(rows)} rows, {run_id} runs; calibrated threshold={threshold:.6f}')


if __name__ == '__main__':
    main()
