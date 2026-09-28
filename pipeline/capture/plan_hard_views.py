#!/usr/bin/env python3
"""Plan the v10 capture: a lane grid plus a spaced hard-view top-up (METHOD amendment v10).

Every candidate view is classified geometrically, identically for existing and new poses,
with the capture's own robot box (0.80 x 0.55 x 0.35 m) and a ray test against the
full-height collision boxes:
  edge           the projected box is cut by, or within 20 px of, the image border
  bottom hidden  the ray from the camera to the footprint centre at 5 cm is blocked
  small          the projected box is under 30 px tall
  easy           in view and none of those
The campaign routes are never read.

1. Lane grid: 0.5 m grid over the driveable region at the four aisle headings, keeping
   poses whose body clears every object by 0.05 m, minus poses already captured within
   0.25 m and 15 deg.
2. Spaced top-up: 0.25 m grid, 8 headings, at least 0.15 m from every sealed-audit
   position; greedy by the number of edge views that close the per-camera deficit to
   +75 % of the existing edge views, never duplicating an existing or lane pose, top-up
   positions at least 0.5 m apart and at most 4 headings per position.

    python3 pipeline/capture/plan_hard_views.py --out logs/thesis/captures/v10
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'pipeline'), str(REPO / 'pipeline/capture'),
                str(REPO / 'src/unav_common'), str(REPO / 'src/reliability')]
import dataset  # noqa: E402
from score_collisions import DriveableRegion, WORLD_PROFILES, WORLD  # noqa: E402
from capture_positions import _project_robot_bbox, _camera_specs  # noqa: E402
from unav_common.occlusion_geometry import profile_collision_scene  # noqa: E402

EDGE_PX, SMALL_PX, BOTTOM_Z = 20, 30, 0.05
BODY_CLEARANCE, LANE_STEP, TOPUP_STEP = 0.05, 0.5, 0.25
COVER_R, COVER_DEG, AUDIT_SEP = 0.25, 15.0, 0.15
EDGE_TARGET, MIN_SPACING, MAX_HEADINGS = 0.75, 0.5, 4
KINDS = ('edge', 'bottom hidden', 'small', 'easy')


class ViewClassifier:
    def __init__(self):
        self.specs, _, _ = _camera_specs(WORLD_PROFILES, WORLD.name)
        profile = yaml.safe_load(open(WORLD_PROFILES))['worlds'][WORLD.name]
        self.prisms = profile_collision_scene(str(WORLD), profile).prisms
        self.P = np.array([[p.xmin, p.xmax, p.ymin, p.ymax, p.zmin, p.zmax] for p in self.prisms])
        self.cameras = [s.camera_id for s in self.specs]

    def _blocked(self, a, b):
        d = b - a
        P = self.P
        with np.errstate(divide='ignore', invalid='ignore'):
            lo = np.stack([(P[:, 0] - a[0]) / d[0], (P[:, 2] - a[1]) / d[1], (P[:, 4] - a[2]) / d[2]], 1)
            hi = np.stack([(P[:, 1] - a[0]) / d[0], (P[:, 3] - a[1]) / d[1], (P[:, 5] - a[2]) / d[2]], 1)
        t0 = np.nanmax(np.minimum(lo, hi), 1)
        t1 = np.nanmin(np.maximum(lo, hi), 1)
        inside = lambda q: ((P[:, 0] - .05 <= q[0]) & (q[0] <= P[:, 1] + .05)
                            & (P[:, 2] - .05 <= q[1]) & (q[1] <= P[:, 3] + .05))
        return bool(np.any((t0 <= t1) & (t1 > 0) & (t0 < 0.999) & ~inside(b) & ~inside(a)))

    def __call__(self, x, y, yaw):
        out = {}
        for spec in self.specs:
            box = _project_robot_bbox(spec.camera, x=x, y=y, yaw=yaw, z=0.0,
                                      box_length=0.80, box_width=0.55, box_height=0.35)
            if box is None:
                out[spec.camera_id] = 'out'
                continue
            x0, y0, x1, y1 = box
            W, H = spec.image_width, spec.image_height
            if not (x1 > 0 and y1 > 0 and x0 < W and y0 < H):
                out[spec.camera_id] = 'out'
            elif min(x0, y0, W - x1, H - y1) < EDGE_PX:
                out[spec.camera_id] = 'edge'
            elif self._blocked(np.asarray(spec.camera.cam_pos, float), np.array([x, y, BOTTOM_Z])):
                out[spec.camera_id] = 'bottom hidden'
            elif y1 - y0 < SMALL_PX:
                out[spec.camera_id] = 'small'
            else:
                out[spec.camera_id] = 'easy'
        return out


def _grid(region, site, step, headings, keep=lambda x, y: True):
    poses = []
    for x in np.arange(site['xmin'] + step / 2, site['xmax'], step):
        for y in np.arange(site['ymin'] + step / 2, site['ymax'], step):
            if not keep(x, y):
                continue
            for h in headings:
                o, b = region.clearance((x, y, h))
                if o >= BODY_CLEARANCE and b >= BODY_CLEARANCE:
                    poses.append((x, y, h))
    return np.array(poses)


def _covered(poses, known):
    tree = cKDTree(known[:, :2])
    return np.array([bool(i) and np.any(np.degrees(np.abs(
        (known[i, 2] - h + np.pi) % (2 * np.pi) - np.pi)) < COVER_DEG)
        for (x, y, h), i in zip(poses, tree.query_ball_point(poses[:, :2], COVER_R))])


def plan():
    classify = ViewClassifier()
    region = DriveableRegion.for_world()
    profile = yaml.safe_load(open(WORLD_PROFILES))['worlds'][WORLD.name]
    site = [r for r in profile['known_2d_regions'] if r.get('type') == 'site_boundary'][0]
    rows = pd.DataFrame(dataset.load_rows())
    audit = rows[rows.stratum == 'final_audit'][['robot_x', 'robot_y']].astype(float).drop_duplicates().values
    existing = rows[rows.stratum != 'final_audit'][['robot_x', 'robot_y', 'robot_yaw']].astype(float).drop_duplicates().values

    lane_all = _grid(region, site, LANE_STEP, (0, math.pi / 2, math.pi, 3 * math.pi / 2))
    lane = lane_all[~_covered(lane_all, existing)]
    views_existing = pd.DataFrame([classify(*p) for p in existing])
    views_lane = pd.DataFrame([classify(*p) for p in lane])
    cams = classify.cameras
    edge_existing = np.array([(views_existing[c] == 'edge').sum() for c in cams])
    edge_lane = np.array([(views_lane[c] == 'edge').sum() for c in cams])
    need = np.maximum(np.round(EDGE_TARGET * edge_existing) - edge_lane, 0).astype(float)

    audit_tree = cKDTree(audit)
    cand = _grid(region, site, TOPUP_STEP, [k * math.pi / 4 for k in range(8)],
                 keep=lambda x, y: audit_tree.query([x, y])[0] >= AUDIT_SEP)
    views_cand = pd.DataFrame([classify(*p) for p in cand])
    E = np.stack([(views_cand[c] == 'edge').values for c in cams], 1).astype(int)
    avail = ~_covered(cand, np.vstack([existing, lane]))
    key = [tuple(np.round(p, 3)) for p in cand[:, :2]]
    per_position, positions, chosen = {}, [], []
    while need.sum() > 0:
        gain = (E * (need > 0)).sum(1) * avail
        i = int(gain.argmax())
        if gain[i] == 0:
            break
        k, p = key[i], cand[i, :2]
        if k not in per_position and positions and np.min(np.hypot(*(np.array(positions) - p).T)) < MIN_SPACING:
            avail[i] = False
            continue
        chosen.append(i)
        avail[i] = False
        need = np.maximum(need - E[i], 0)
        if k not in per_position:
            per_position[k] = 0
            positions.append(p)
        per_position[k] += 1
        if per_position[k] >= MAX_HEADINGS:
            avail[[j for j in range(len(cand)) if key[j] == k]] = False
    topup = cand[chosen]
    return dict(lane=lane, topup=topup, existing=existing, cameras=cams,
                views=dict(existing=views_existing, lane=views_lane,
                           topup=views_cand.iloc[chosen].reset_index(drop=True)),
                unmet=dict(zip(cams, need.astype(int).tolist())))


def pose_file(lane, topup):
    entries, index = [], {}
    for kind, poses in (('lane_grid', lane), ('hard_view_topup', topup)):
        for x, y, h in poses:
            k = (round(float(x), 3), round(float(y), 3))
            if k not in index:
                index[k] = {'pid': len(index), 'n': 0}
            slot = index[k]
            entries.append({'x': k[0], 'y': k[1], 'yaw': float(h), 'stratum': 'unassigned',
                            'position_id': slot['pid'], 'position_key': 'L%04d' % slot['pid'],
                            'yaw_idx': slot['n'], 'heading_id': slot['n'],
                            'heading_degrees': round(math.degrees(h), 6), 'kind': kind})
            slot['n'] += 1
    return entries


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    result = plan()
    poses = pose_file(result['lane'], result['topup'])
    path = out / 'capture_poses_v10.json'
    path.write_text(json.dumps(poses, indent=1) + '\n', encoding='utf-8')
    counts = {name: {c: {k: int((v[c] == k).sum()) for k in KINDS} for c in result['cameras']}
              for name, v in result['views'].items()}
    manifest = {
        'schema': 'capture_plan_v10.v1', 'implementation': 'pipeline/capture/plan_hard_views.py',
        'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'pose_file': path.name, 'pose_file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'lane_poses': int(len(result['lane'])), 'topup_poses': int(len(result['topup'])),
        'positions': len({(p['x'], p['y']) for p in poses}),
        'rule': {'edge_px': EDGE_PX, 'small_px': SMALL_PX, 'bottom_z_m': BOTTOM_Z,
                 'body_clearance_m': BODY_CLEARANCE, 'lane_step_m': LANE_STEP,
                 'topup_step_m': TOPUP_STEP, 'covered_r_m': COVER_R, 'covered_deg': COVER_DEG,
                 'audit_separation_m': AUDIT_SEP, 'edge_target_increase': EDGE_TARGET,
                 'min_spacing_m': MIN_SPACING, 'max_headings': MAX_HEADINGS},
        'unmet_edge_deficit': result['unmet'], 'view_counts': counts,
        'campaign_routes_read': False,
    }
    (out / 'capture_plan_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ('lane_poses', 'topup_poses', 'positions', 'unmet_edge_deficit')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
