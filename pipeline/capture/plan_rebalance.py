#!/usr/bin/env python3
"""Plan the spatially balanced dataset v9: strata, a fresh sealed audit, role re-split, fill.

Why: in v8 the southern cross-aisle held about half of all positions and 59 % of the final
audit, and camera C had 14 audit positions. v9 makes every evaluation role spatially
uniform over the space the robot can occupy, without deleting any captured position.

Rule (declared before capture; seeded, nothing chosen by looking at errors):
1. Valid centres: 0.1 m grid points where the 0.80 x 0.55 m footprint clears the collision
   scene by the capture body clearance at every heading, inside the site keep-in region
   (the same test as the v8 top-up).
2. Strata: 2 x 2 m cells; a cell with less than MIN_STRATUM_M2 of valid centres joins the
   nearest stratum. A stratum's weight is its valid-centre area.
3. Quotas: the fresh audit (150), D_dev (437) and D_R (704) are split over strata in
   proportion to area (largest remainder). The totals equal v8's.
4. Existing positions (all of v8, the opened v8 audit included) are ordered inside their
   stratum by sha256(seed, position_key): the first fill D_dev, the next D_R, the rest D_mu.
   No position is dropped.
5. Where a stratum has too few positions for its D_dev + D_R quota, fill positions are drawn
   uniformly from its valid centres, at least MIN_SPACING_M from every captured position.
6. The fresh audit is drawn the same way inside each stratum, at least MIN_SPACING_M from
   every captured, fill or detector-training position and from each other (0.15 m: the
   capture grid spacing, and the median distance of the v8 audit to its nearest training
   position). Its images do not exist yet, so nothing about it has been seen.
Headings follow v5/v8: four, 90 degrees apart, with a seeded offset.

Writes logs/thesis/captures/v9/: capture_poses_v9.json (fill + audit), partition_v9.csv
(every position -> role), strata_v9.json, rebalance_plan.json, rebalance_map.png.

    python3 pipeline/capture/plan_rebalance.py
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import math
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = pathlib.Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src" / "unav_common")]
from pipeline import dataset  # noqa: E402
from unav_common.occlusion_geometry import profile_collision_scene  # noqa: E402
from unav_common.rectangular_footprint import RectangularFootprint  # noqa: E402

OUT = REPO / "logs/thesis/captures/v9"
WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
DETECTOR_POSITIONS = REPO / "pipeline/capture/camera_capture_positions_v3.csv"
BODY_CLEARANCE = 0.0499
HEADINGS = np.linspace(0.0, math.pi, 36, endpoint=False)
SITE_HALF_X, SITE_HALF_Y = 11.25, 9.25
HALF_DIAGONAL = 0.5 * math.hypot(0.80, 0.55)
GRID_M, CELL_M = 0.1, 2.0
MIN_STRATUM_M2 = 0.5
MIN_SPACING_M = 0.15
QUOTA = {"final_audit": 150, "D_dev": 437, "D_R": 704}
SEED = 20260924
ROLE_COLOUR = {"D_mu": "#56B4E9", "D_R": "#CC79A7", "D_dev": "#E69F00", "final_audit": "#1a1a1a"}


def valid_centres():
    profile = yaml.safe_load((REPO / "src/experiments/config/world_profiles.yaml").read_text())["worlds"][WORLD.name]
    fp = RectangularFootprint(tuple(profile_collision_scene(str(WORLD), profile).prisms), length=0.80, width=0.55)
    xs = np.arange(-SITE_HALF_X + HALF_DIAGONAL, SITE_HALF_X - HALF_DIAGONAL + 1e-9, GRID_M)
    ys = np.arange(-SITE_HALF_Y + HALF_DIAGONAL, SITE_HALF_Y - HALF_DIAGONAL + 1e-9, GRID_M)
    pts = [(float(x), float(y)) for x in xs for y in ys
           if all(fp.clearance((float(x), float(y), float(h))) >= BODY_CLEARANCE for h in HEADINGS)]
    return np.round(np.array(pts), 3), fp


def cell_of(x, y):
    return (math.floor(x / CELL_M), math.floor(y / CELL_M))


def largest_remainder(total, weights):
    keys = list(weights)
    w = np.array([weights[k] for k in keys], float)
    exact = total * w / w.sum()
    base = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - base))[: total - base.sum()]:
        base[i] += 1
    return dict(zip(keys, base.tolist()))


def order_key(position_key):
    return hashlib.sha256(f"{SEED}:{position_key}".encode()).hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    centres, fp = valid_centres()
    # strata
    area = collections.Counter(cell_of(x, y) for x, y in centres)
    cell_area = {c: n * GRID_M ** 2 for c, n in area.items()}
    small = {c for c, a in cell_area.items() if a < MIN_STRATUM_M2}
    big = [c for c in cell_area if c not in small]
    centre_of = lambda c: np.array([(c[0] + 0.5) * CELL_M, (c[1] + 0.5) * CELL_M])
    stratum_of_cell = {c: c for c in big}
    for c in small:
        stratum_of_cell[c] = min(big, key=lambda b: float(np.linalg.norm(centre_of(b) - centre_of(c))))
    strata = sorted(set(stratum_of_cell.values()))
    s_area = collections.Counter()
    s_centres = collections.defaultdict(list)
    for x, y in centres:
        s = stratum_of_cell[cell_of(x, y)]
        s_area[s] += GRID_M ** 2
        s_centres[s].append((x, y))
    quotas = {role: largest_remainder(n, dict(s_area)) for role, n in QUOTA.items()}

    # existing positions (unique physical positions of v8, every role)
    positions = {}
    for r in dataset.load_rows(v8_only=True):
        positions.setdefault(r["position_key"], (float(r["robot_x"]), float(r["robot_y"]), r["stratum"]))
    existing_xy = np.array([(x, y) for x, y, _ in positions.values()])
    by_stratum = collections.defaultdict(list)
    for key, (x, y, old) in positions.items():
        c = cell_of(x, y)
        s = stratum_of_cell.get(c)
        if s is None:   # a captured centre outside today's valid set: nearest stratum
            s = min(strata, key=lambda b: float(np.linalg.norm(centre_of(b) - np.array([x, y]))))
        by_stratum[s].append(key)

    rng = np.random.default_rng(SEED)
    placed = [tuple(p) for p in existing_xy]
    detector_xy = [(float(r["x_m"]), float(r["y_m"])) for r in csv.DictReader(DETECTOR_POSITIONS.open())]

    def draw(s, n, avoid):
        if n <= 0:
            return []
        cand = np.array(s_centres[s])
        chosen = []
        for i in rng.permutation(len(cand)):
            p = cand[i] + rng.uniform(-GRID_M / 2, GRID_M / 2, 2)
            if not all(fp.clearance((float(p[0]), float(p[1]), float(h))) >= BODY_CLEARANCE for h in HEADINGS):
                continue
            ref = np.array(avoid + chosen) if (avoid or chosen) else np.empty((0, 2))
            if len(ref) and np.min(np.hypot(ref[:, 0] - p[0], ref[:, 1] - p[1])) < MIN_SPACING_M:
                continue
            chosen.append((round(float(p[0]), 3), round(float(p[1]), 3)))
            if len(chosen) == n:
                break
        return chosen

    partition, fill, audit, shortfall = {}, [], [], {}
    for s in strata:
        keys = sorted(by_stratum.get(s, []), key=order_key)
        n_dev, n_r = quotas["D_dev"][s], quotas["D_R"][s]
        for i, k in enumerate(keys):
            partition[k] = "D_dev" if i < n_dev else "D_R" if i < n_dev + n_r else "D_mu"
        missing = max(0, n_dev + n_r - len(keys))
        if missing:
            roles = (["D_dev"] * max(0, n_dev - len(keys)) + ["D_R"] * n_r)[-missing:]
            pts = draw(s, missing, placed)
            shortfall[str(s)] = missing - len(pts)
            for p, role in zip(pts, roles):
                fill.append((p, role))
                placed.append(p)
    for s in strata:
        pts = draw(s, quotas["final_audit"][s], placed + detector_xy)
        if len(pts) < quotas["final_audit"][s]:
            shortfall[f"audit {s}"] = quotas["final_audit"][s] - len(pts)
        for p in pts:
            audit.append(p)
            placed.append(p)

    poses = []
    new_positions = [(p, role, f"W{i:04d}") for i, (p, role) in enumerate(fill)] + \
                    [(p, "final_audit", f"V{i:04d}") for i, p in enumerate(audit)]
    for index, ((x, y), role, key) in enumerate(new_positions):
        offset = float(rng.uniform(0, math.pi / 2))
        for k in range(4):
            yaw = (offset + k * math.pi / 2) % (2 * math.pi)
            poses.append({"x": x, "y": y, "yaw": yaw, "stratum": role, "position_id": index,
                          "position_key": key, "yaw_idx": k, "heading_id": k,
                          "heading_degrees": round(math.degrees(yaw), 6),
                          "kind": "v9_audit" if role == "final_audit" else "v9_fill"})
        partition[key] = role
    (OUT / "capture_poses_v9.json").write_text(json.dumps(poses, indent=1))
    with (OUT / "partition_v9.csv").open("w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["position_key", "role", "source"])
        for k in sorted(partition):
            w.writerow([k, partition[k], "v8" if k in positions else "v9"])
    (OUT / "strata_v9.json").write_text(json.dumps({
        "cell_m": CELL_M, "grid_m": GRID_M, "min_stratum_m2": MIN_STRATUM_M2,
        "strata": [{"cell": list(s), "valid_area_m2": round(s_area[s], 3),
                    **{role: quotas[role][s] for role in QUOTA},
                    "existing": len(by_stratum.get(s, []))} for s in strata],
        "merged_cells": {str(c): list(stratum_of_cell[c]) for c in small}}, indent=1))
    counts = collections.Counter(partition.values())
    summary = {"rule_source": str(HERE.relative_to(REPO)), "seed": SEED, "quota": QUOTA,
               "strata": len(strata), "valid_area_m2": round(sum(s_area.values()), 2),
               "existing_positions": len(positions), "fill_positions": len(fill),
               "fill_by_role": dict(collections.Counter(r for _, r in fill)),
               "audit_positions": len(audit), "unmet_quota": {k: v for k, v in shortfall.items() if v},
               "roles_v9": dict(counts),
               "old_v8_audit_now": dict(collections.Counter(partition[k] for k, v in positions.items() if v[2] == "final_audit"))}
    (OUT / "rebalance_plan.json").write_text(json.dumps(summary, indent=1))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.6), constrained_layout=True)
    for ax, title in zip(axes, ("v8 roles (as used so far)", "v9 roles (planned)")):
        for p in profile_collision_scene(str(WORLD), yaml.safe_load((REPO / "src/experiments/config/world_profiles.yaml").read_text())["worlds"][WORLD.name]).prisms:
            ax.add_patch(Rectangle((p.xmin, p.ymin), p.xmax - p.xmin, p.ymax - p.ymin, fc="#e4e2dc", ec="#bdbab2", lw=0.4))
        for s in strata:
            ax.add_patch(Rectangle((s[0] * CELL_M, s[1] * CELL_M), CELL_M, CELL_M, fill=False, ec="#dddddd", lw=0.3))
        ax.set_xlim(-12, 12); ax.set_ylim(-10, 10); ax.set_aspect("equal"); ax.set_title(title)
    for role, colour in ROLE_COLOUR.items():
        a = np.array([(x, y) for x, y, r in positions.values() if r == role])
        axes[0].scatter(a[:, 0], a[:, 1], s=4 if role != "final_audit" else 10, c=colour, marker="s" if role == "final_audit" else "o", lw=0, label=f"{role} {len(a)}")
        b = [positions[k][:2] for k, r in partition.items() if r == role and k in positions]
        b += [p for p, r in fill if r == role] + ([tuple(p) for p in audit] if role == "final_audit" else [])
        b = np.array(b)
        axes[1].scatter(b[:, 0], b[:, 1], s=4 if role != "final_audit" else 10, c=colour, marker="s" if role == "final_audit" else "o", lw=0, label=f"{role} {len(b)}")
    f = np.array([p for p, _ in fill]) if fill else np.empty((0, 2))
    if len(f):
        axes[1].scatter(f[:, 0], f[:, 1], s=26, facecolors="none", edgecolors="#D55E00", lw=0.8, label=f"new fill {len(f)}")
    q = np.array(audit)
    axes[1].scatter(q[:, 0], q[:, 1], s=30, facecolors="none", edgecolors="k", lw=0.8, marker="s", label=f"new audit {len(q)}")
    for ax in axes:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=6, fontsize=8, frameon=False)
    fig.savefig(OUT / "rebalance_map.png", dpi=110)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
