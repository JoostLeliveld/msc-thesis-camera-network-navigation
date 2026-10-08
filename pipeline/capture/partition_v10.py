#!/usr/bin/env python3
"""Roles for the v10 lane-grid positions (METHOD amendment v10, "Roles").

Rule, declared before the capture and never informed by errors:
  * every v9 role is kept, and the sealed v9 audit is unchanged;
  * each new position falls into its v9 2 x 2 m stratum (`captures/v9/strata_v9.json`: its
    own cell, the stratum a small cell was merged into, or the nearest stratum centre);
  * inside a stratum the new positions are split over D_dev, D_R and D_mu in the
    proportions of that stratum's v9 working set (largest remainder), in the order
    sha256(seed, position_key): D_dev first, then D_R, the rest D_mu.
Output: `captures/v10/partition_v10.csv` (every v9 row, then the new rows).

    python3 pipeline/capture/partition_v10.py
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'pipeline/capture')]
from plan_rebalance import SEED, cell_of, largest_remainder  # noqa: E402

CAPTURES = REPO / 'logs/thesis/captures'
V9_PARTITION = CAPTURES / 'v9/partition_v9.csv'
V9_STRATA = CAPTURES / 'v9/strata_v9.json'
LANE_POSES = CAPTURES / 'v10/lane/capture_poses_v10_lane_kept.json'
OUT = CAPTURES / 'v10/partition_v10.csv'
WORKING = ('D_dev', 'D_R', 'D_mu')


def stratum_map():
    strata = json.loads(V9_STRATA.read_text(encoding='utf-8'))
    own = {tuple(s['cell']): tuple(s['cell']) for s in strata['strata']}
    merged = {tuple(int(v) for v in k.strip('()').split(',')): tuple(v) for k, v in strata['merged_cells'].items()}
    centres = {s: np.array([(s[0] + .5) * strata['cell_m'], (s[1] + .5) * strata['cell_m']]) for s in own}

    def of(x, y):
        c = cell_of(x, y)
        if c in own:
            return own[c]
        if c in merged:
            return merged[c]
        return min(centres, key=lambda s: float(np.linalg.norm(centres[s] - np.array([x, y]))))
    return of


def order_key(position_key):
    return hashlib.sha256(f'{SEED}:{position_key}'.encode()).hexdigest()


def partition(v9_rows, v9_positions, lane_positions):
    """v9_rows: [{position_key, role, source}]; *_positions: {key: (x, y)}."""
    of = stratum_map()
    counts = collections.defaultdict(collections.Counter)
    for r in v9_rows:
        if r['role'] in WORKING and r['position_key'] in v9_positions:
            counts[of(*v9_positions[r['position_key']])][r['role']] += 1
    new_by_stratum = collections.defaultdict(list)
    for key, (x, y) in lane_positions.items():
        new_by_stratum[of(x, y)].append(key)
    assigned = {}
    for s, keys in new_by_stratum.items():
        weights = {role: counts[s][role] for role in WORKING}
        if sum(weights.values()) == 0:
            weights = {'D_dev': 0, 'D_R': 0, 'D_mu': 1}
        quota = largest_remainder(len(keys), weights)
        ordered = sorted(keys, key=order_key)
        i = 0
        for role in WORKING:
            for key in ordered[i:i + quota[role]]:
                assigned[key] = role
            i += quota[role]
    return assigned


def main() -> int:
    import dataset  # noqa: E402  (v9 loader; reads the v9 partition)
    v9_rows = list(csv.DictReader(V9_PARTITION.open(newline='', encoding='utf-8')))
    v9_positions = {}
    for r in dataset.load_rows():
        v9_positions.setdefault(r['position_key'], (float(r['robot_x']), float(r['robot_y'])))
    lane = json.loads(LANE_POSES.read_text(encoding='utf-8'))
    lane_positions = {}
    for p in lane:
        lane_positions.setdefault(p['position_key'], (float(p['x']), float(p['y'])))
    if set(lane_positions) & set(v9_positions):
        raise RuntimeError('lane position keys collide with v9 keys')
    assigned = partition(v9_rows, v9_positions, lane_positions)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', newline='', encoding='utf-8') as handle:
        w = csv.DictWriter(handle, fieldnames=['position_key', 'role', 'source'])
        w.writeheader()
        for r in v9_rows:
            w.writerow(r)
        for key in sorted(assigned):
            w.writerow({'position_key': key, 'role': assigned[key], 'source': 'lane_v10'})
    print(json.dumps(dict(collections.Counter(assigned.values()))))
    return 0


if __name__ == '__main__':
    sys.path.insert(0, str(REPO / 'pipeline'))
    raise SystemExit(main())
