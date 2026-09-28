#!/usr/bin/env python3
"""Roles for dataset v11: one dataset, every role an even sample of the same area (METHOD amendment v11).

Rule, declared before any v11 fit and never informed by errors:
  * the pool is every captured position, whatever capture it came from;
  * a position is valid when it has four captured headings and the 0.80 x 0.55 m footprint
    clears the collision scene by the capture body clearance at every heading inside the site
    (`plan_fill_v11.footprint_rule`); invalid positions get the role `excluded`;
  * role densities per m^2 of valid area are those of `plan_fill_v11.DENSITY`: test
    (final_audit) 1, development 2, covariance fit 8 (derived from R2), correction fit 8;
    each role's total, density times the valid area, is spread over the v9 2 x 2 m strata in
    proportion to valid area (largest remainder);
  * inside a stratum the valid positions are ordered by sha256(seed, position_key),
    regardless of capture, and take the quotas in the order final_audit, D_dev, D_R, D_mu; a
    stratum short of positions falls short in D_mu. Valid positions beyond the quotas get the
    role `unused` and enter no fit or evaluation.
Output: `captures/v11/partition_v11.csv` and `captures/v11/partition_v11_summary.json`.

    python3 pipeline/capture/partition_v11.py
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'pipeline/capture'), str(REPO / 'pipeline')]
from plan_rebalance import largest_remainder  # noqa: E402
from partition_v10 import stratum_map  # noqa: E402
from plan_fill_v11 import DENSITY, footprint_rule  # noqa: E402

CAPTURES = REPO / 'logs/thesis/captures'
V9_STRATA = CAPTURES / 'v9/strata_v9.json'
OUT_DIR = CAPTURES / 'v11'
SEED = 20260925
ORDER = ('final_audit', 'D_dev', 'D_R', 'D_mu')


def order_key(position_key: str) -> str:
    return hashlib.sha256(f'{SEED}:{position_key}'.encode()).hexdigest()


def pool():
    import dataset  # noqa: E402
    out = {}
    for r in dataset.load_rows(apply_partition=False):
        p = out.setdefault(r['position_key'], {'x': float(r['robot_x']), 'y': float(r['robot_y']),
                                              'source': r['capture_source'], 'yaws': set()})
        p['yaws'].add(int(r['yaw_idx']))
    return out


def partition(positions):
    fits = footprint_rule()
    strata = json.loads(V9_STRATA.read_text(encoding='utf-8'))
    area = {tuple(s['cell']): float(s['valid_area_m2']) for s in strata['strata']}
    of = stratum_map()
    roles, members = {}, collections.defaultdict(list)
    for key, p in positions.items():
        if len(p['yaws']) == 4 and fits(p['x'], p['y']):
            members[of(p['x'], p['y'])].append(key)
        else:
            roles[key] = 'excluded'
    if set(members) - set(area):
        raise RuntimeError('a valid position maps to a stratum without an area')
    total_area = sum(area.values())
    quota = {role: largest_remainder(round(DENSITY[role] * total_area), area) for role in ORDER}
    for s in sorted(area):
        keys = sorted(members.get(s, []), key=order_key)
        cursor = 0
        for role in ORDER:
            for key in keys[cursor:cursor + quota[role][s]]:
                roles[key] = role
            cursor = min(len(keys), cursor + quota[role][s])
        for key in keys[cursor:]:
            roles[key] = 'unused'
    return roles, quota


def main() -> int:
    positions = pool()
    roles, quota = partition(positions)
    if set(roles) != set(positions):
        raise RuntimeError('partition does not cover the pool exactly')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / 'partition_v11.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['position_key', 'role', 'source'])
        writer.writeheader()
        for key in sorted(roles):
            writer.writerow({'position_key': key, 'role': roles[key], 'source': positions[key]['source']})
    got = collections.Counter(roles.values())
    summary = {'seed': SEED, 'density_per_m2': DENSITY,
               'quota': {r: sum(q.values()) for r, q in quota.items()},
               'assigned': dict(got),
               'short_of_quota': {r: sum(quota[r].values()) - got[r] for r in ORDER}}
    (OUT_DIR / 'partition_v11_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
