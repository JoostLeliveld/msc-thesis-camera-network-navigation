#!/usr/bin/env python3
"""Plan the v11 fill capture: every 2 x 2 m stratum up to one even density (METHOD amendment v11).

Design, declared before capture and never informed by errors. Every role is an even sample
of the same valid area; only its density differs:
  * covariance fit 8 / m^2, derived from R2: K / (pi (2 l_R)^2) with K = 16, l_R = 0.4 m;
  * correction fit 8 / m^2 (the same resolution, a design choice);
  * development 2 / m^2 and test 1 / m^2.
The total target is 19 / m^2 of valid area. A position is valid when the 0.80 x 0.55 m
footprint clears the collision scene by the capture body clearance at every heading and lies
inside the site (the plan_topup rule), and when it has four captured headings.

Every stratum (the v9 2 x 2 m strata, `captures/v9/strata_v9.json`) below
ceil(19 * valid area) valid positions is filled to it with new positions at valid 0.1 m
sub-grid points farthest from every existing position, never closer than 0.15 m to one.
Headings: four, 90 degrees apart, seeded random offset. Keys `Nnnnn`.

    python3 pipeline/capture/plan_fill_v11.py
"""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'pipeline/capture'), str(REPO / 'pipeline'), str(REPO / 'src/unav_common')]
from unav_common.occlusion_geometry import profile_collision_scene  # noqa: E402
from unav_common.rectangular_footprint import RectangularFootprint  # noqa: E402
from plan_topup import BODY_CLEARANCE, HEADINGS, SITE_HALF_X, SITE_HALF_Y, HALF_DIAGONAL  # noqa: E402
from partition_v10 import stratum_map  # noqa: E402

CAPTURES = REPO / 'logs/thesis/captures'
V9_STRATA = CAPTURES / 'v9/strata_v9.json'
OUT = CAPTURES / 'v11'
WORLD = REPO / 'src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf'
PROFILES = REPO / 'src/experiments/config/world_profiles.yaml'
DENSITY = {'final_audit': 1.0, 'D_dev': 2.0, 'D_R': 16 / (math.pi * 0.8 ** 2), 'D_mu': 8.0}
TOTAL_DENSITY = sum(DENSITY.values())
MIN_SPACING_M = 0.15
SEED = 20260925


def footprint_rule():
    scene = profile_collision_scene(str(WORLD), yaml.safe_load(PROFILES.read_text())['worlds']['warehouse_v2.world.sdf'])
    footprint = RectangularFootprint(tuple(scene.prisms), length=0.80, width=0.55)

    def fits(x, y, headings=HEADINGS):
        if abs(x) > SITE_HALF_X - HALF_DIAGONAL or abs(y) > SITE_HALF_Y - HALF_DIAGONAL:
            return False
        return all(footprint.clearance((x, y, float(h))) >= BODY_CLEARANCE for h in headings)
    return fits


def existing_positions():
    """Every captured position with its headings, before any role."""
    import dataset  # noqa: E402
    pos = {}
    for r in dataset.load_rows(apply_partition=False):
        p = pos.setdefault(r['position_key'], {'x': float(r['robot_x']), 'y': float(r['robot_y']),
                                              'yaws': set()})
        p['yaws'].add(int(r['yaw_idx']))
    return pos


def main() -> int:
    fits = footprint_rule()
    strata = json.loads(V9_STRATA.read_text(encoding='utf-8'))
    area = {tuple(s['cell']): float(s['valid_area_m2']) for s in strata['strata']}
    of = stratum_map()
    pos = existing_positions()
    valid = {k: p for k, p in pos.items() if len(p['yaws']) == 4 and fits(p['x'], p['y'])}
    excluded = sorted(set(pos) - set(valid))
    count = collections.Counter(of(p['x'], p['y']) for p in valid.values())

    # candidate sub-grid points per stratum (a stratum can own merged cells)
    candidates = collections.defaultdict(list)
    for x in np.arange(-11.25 + 0.05, 11.25, 0.1):
        for y in np.arange(-9.25 + 0.05, 9.25, 0.1):
            s = of(x, y)
            if s in area and fits(x, y):
                candidates[s].append((round(float(x), 3), round(float(y), 3)))
    placed = np.array([[p['x'], p['y']] for p in pos.values()])
    new = []
    short = {}
    for s in sorted(area):
        need = math.ceil(TOTAL_DENSITY * area[s]) - count[s]
        pts = np.array(candidates.get(s, []))
        for _ in range(max(0, need)):
            if not len(pts):
                break
            d = np.min(np.hypot(pts[:, None, 0] - placed[None, :, 0], pts[:, None, 1] - placed[None, :, 1]), axis=1)
            best = int(np.argmax(d))
            if d[best] < MIN_SPACING_M:
                break
            new.append(tuple(pts[best]))
            placed = np.vstack([placed, pts[best]])
        got = sum(1 for p in new if of(*p) == s)
        if need > got:
            short[str(s)] = need - got

    rng = np.random.default_rng(SEED)
    poses, kept = [], []
    for x, y in new:
        offset = float(rng.uniform(0, math.pi / 2))
        yaws = [(offset + k * math.pi / 2) % (2 * math.pi) for k in range(4)]
        if not fits(x, y, yaws):
            continue
        key = f'N{len(kept):04d}'
        kept.append({'position_key': key, 'x': x, 'y': y})
        for k, yaw in enumerate(yaws):
            poses.append({'x': x, 'y': y, 'yaw': yaw, 'stratum': 'unassigned', 'position_id': len(kept) - 1,
                          'position_key': key, 'yaw_idx': k, 'heading_id': k,
                          'heading_degrees': round(math.degrees(yaw), 6), 'kind': 'fill_v11'})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'fill').mkdir(exist_ok=True)
    (OUT / 'fill/capture_poses_v11_fill.json').write_text(json.dumps(poses, indent=1))
    summary = {'rule_source': 'pipeline/capture/plan_fill_v11.py', 'density_per_m2': DENSITY,
               'total_density_per_m2': TOTAL_DENSITY, 'valid_area_m2': sum(area.values()),
               'existing_positions': len(pos), 'existing_valid': len(valid),
               'excluded_existing': {'count': len(excluded),
                                     'two_headings': sum(1 for k in excluded if len(pos[k]['yaws']) != 4),
                                     'keys': excluded},
               'new_positions': len(kept), 'new_poses': len(poses),
               'strata_still_short': short}
    (OUT / 'fill/fill_plan.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'excluded_existing'} |
                     {'excluded_existing': {k: v for k, v in summary['excluded_existing'].items() if k != 'keys'}},
                     indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
