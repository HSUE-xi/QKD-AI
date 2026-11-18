"""Causal features matching the teacher's review definitions.

The first block of each run is an independently measured normal baseline.
Count ratios equal click-rate ratios only when sent_count is constant per run.
No labels or truth metadata are used as model inputs.
"""

from collections import defaultdict

import numpy as np

F7 = ['qber', 'z_error_rate', 'x_error_rate', 'click_rate', 'test_count',
      'z_test_count', 'x_test_count']
REL = F7 + ['click_ratio', 'cum_z', 'cum_x', 'cum_q', 'cum_ratio',
            'ratio_slope', 'base_q']


def add_relative_features(rows):
    """Add baseline and past/current block features in place, excluding future blocks."""
    runs = defaultdict(list)
    for row in rows:
        runs[row['run_id']].append(row)
    for run in runs.values():
        run.sort(key=lambda row: row['block_index'])
        base = run[0]
        if base['block_index'] != 0 or base['click_count'] <= 0:
            raise ValueError('Each run needs block 0 with a positive baseline click count.')
        if any(row['sent_count'] != base['sent_count'] for row in run):
            raise ValueError('Review-compatible count ratios require constant sent_count per run.')
        z_errors = x_errors = z_count = x_count = 0
        ratios = []
        for row in run:
            row['click_ratio'] = row['click_count'] / base['click_count']
            ratios.append(row['click_ratio'])
            if row['block_index'] > 0:
                z_errors += row['z_test_error_count']
                x_errors += row['x_test_error_count']
                z_count += row['z_test_count']
                x_count += row['x_test_count']
            row['cum_z'] = z_errors / z_count if z_count else 0.0
            row['cum_x'] = x_errors / x_count if x_count else 0.0
            row['cum_q'] = (z_errors + x_errors) / (z_count + x_count) if z_count + x_count else 0.0
            row['cum_ratio'] = float(np.mean(ratios[1:])) if len(ratios) > 1 else 1.0
            row['ratio_slope'] = float(np.polyfit(range(len(ratios)), ratios, 1)[0]) if len(ratios) > 2 else 0.0
            row['base_q'] = base['qber']
    return rows
