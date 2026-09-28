#!/usr/bin/env python3
"""Plan the uniform top-up of the reference-position dataset (v8), and map it.

Target density, derived from R2 rather than tuned: K / (pi (2 l_R)^2) = 8 positions per
m^2 (K=16, l_R=0.4 m), scaled per 1 m cell by the fraction of the cell where the
0.80 x 0.55 m footprint clears the collision scene by the capture body clearance at every
heading. Every cell below its target is topped up to it, anywhere in the warehouse. No
position is removed; cells above the cap are handled by density weights in reporting.

The operating domain is the site boundary (+-11.25 x +-9.25 m, the planner's keep-in
region): the footprint must lie inside it at every heading, so the centre stays within
the boundary minus the footprint half-diagonal. v5 respects the same limit.

Existing positions are v5 plus the 12 camera-C supplement positions already captured. New
points are placed at footprint-valid sub-grid points farthest from every existing
position. Each new position inherits the role of its nearest v5 position, so near-duplicate
views never cross roles; a position whose nearest v5 neighbour is final_audit is dropped,
which leaves the sealed audit set unchanged. Headings follow v5: four, 90 degrees apart,
with a seeded random offset.

    python3 pipeline/capture/plan_topup.py
"""
from __future__ import annotations

import collections
import csv
import json
import math
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import yaml  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = pathlib.Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src" / "unav_common"))
from unav_common.occlusion_geometry import profile_collision_scene  # noqa: E402
from unav_common.rectangular_footprint import RectangularFootprint  # noqa: E402

L = REPO / "logs/thesis/captures"
V5_POSITIONS = L / "v5/capture_positions_v5.csv"
SUPPLEMENT_POSES = L / "v8/capture_poses_supplement.json"
OUT = L / "v8"
WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
K, ELL = 16, 0.4
CAP = K / (math.pi * (2 * ELL) ** 2)
BODY_CLEARANCE = 0.0499
HEADINGS = np.linspace(0.0, math.pi, 36, endpoint=False)  # the footprint is pi-periodic
SITE_HALF_X, SITE_HALF_Y = 11.25, 9.25
HALF_DIAGONAL = 0.5 * math.hypot(0.80, 0.55)
SEED = 20260923
CAMS = {"A": (-11.45, -9.45), "B": (-1.5, -9.72), "C": (-6.95, 9.45),
        "D": (11.45, 7.2), "E": (11.45, -9.45)}
ROLE_COLOUR = {"D_mu": "#2f8f5b", "D_R": "#1b6ca8", "D_dev": "#e8a33d", "final_audit": "#8e44ad"}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    scene = profile_collision_scene(str(WORLD), yaml.safe_load((REPO / "src/experiments/config/world_profiles.yaml").read_text())["worlds"]["warehouse_v2.world.sdf"])
    footprint = RectangularFootprint(tuple(scene.prisms), length=0.80, width=0.55)

    def fits(x, y):
        if abs(x) > SITE_HALF_X - HALF_DIAGONAL or abs(y) > SITE_HALF_Y - HALF_DIAGONAL:
            return False
        return all(footprint.clearance((x, y, float(h))) >= BODY_CLEARANCE for h in HEADINGS)

    v5 = list(csv.DictReader(V5_POSITIONS.open()))
    v5_xy = np.array([[float(r["x"]), float(r["y"])] for r in v5])
    v5_role = [r["role"] for r in v5]
    supplement = sorted({(p["x"], p["y"]) for p in json.loads(SUPPLEMENT_POSES.read_text())})
    existing = np.vstack([v5_xy, np.array(supplement)])
    cells = collections.Counter((math.floor(x), math.floor(y)) for x, y in existing)

    sub = np.arange(0.05, 1.0, 0.1)
    new = []
    for i in range(-12, 12):
        for j in range(-10, 10):
            free = [(i + u, j + v) for u in sub for v in sub if fits(i + u, j + v)]
            target = int(round(CAP * len(free) / len(sub) ** 2))
            need = target - cells.get((i, j), 0)
            if need <= 0 or not free:
                continue
            placed = np.vstack([existing] + ([np.array(new)] if new else []))
            candidates = np.array(free)
            for _ in range(need):
                dist = np.min(np.hypot(candidates[:, None, 0] - placed[None, :, 0],
                                       candidates[:, None, 1] - placed[None, :, 1]), axis=1)
                best = int(np.argmax(dist))
                if dist[best] < 0.1:
                    break
                point = tuple(np.round(candidates[best], 3))
                new.append(point)
                placed = np.vstack([placed, point])

    rng = np.random.default_rng(SEED)
    poses, kept, dropped, heading_rejects = [], [], [], []
    for x, y in new:
        d = np.hypot(v5_xy[:, 0] - x, v5_xy[:, 1] - y)
        role = v5_role[int(np.argmin(d))]
        if role == "final_audit":
            dropped.append((x, y))
            continue
        offset = float(rng.uniform(0, math.pi / 2))
        yaws = [(offset + k * math.pi / 2) % (2 * math.pi) for k in range(4)]
        if any(footprint.clearance((x, y, yaw)) < BODY_CLEARANCE for yaw in yaws):
            heading_rejects.append((x, y))
            continue
        index = len(kept)
        kept.append({"position_key": f"U{index:04d}", "x": x, "y": y, "role": role,
                     "nearest_v5_m": round(float(d.min()), 3)})
        for k in range(4):
            yaw = (offset + k * math.pi / 2) % (2 * math.pi)
            poses.append({"x": x, "y": y, "yaw": yaw, "stratum": role, "position_id": index,
                          "position_key": f"U{index:04d}", "yaw_idx": k, "heading_id": k,
                          "heading_degrees": round(math.degrees(yaw), 6),
                          "kind": "uniform_topup"})
    (OUT / "capture_poses_v8_topup.json").write_text(json.dumps(poses, indent=1))
    summary = {
        "rule_source": str(HERE.relative_to(REPO)), "cap_per_m2": CAP,
        "existing_positions": {"v5": len(v5), "supplement": len(supplement)},
        "planned_new_positions": len(new), "kept": len(kept), "dropped_final_audit_neighbour": len(dropped),
        "rejected_at_drawn_heading": len(heading_rejects),
        "kept_by_role": dict(collections.Counter(p["role"] for p in kept)),
        "max_nearest_v5_m": max(p["nearest_v5_m"] for p in kept),
        "positions": kept, "dropped": dropped,
    }
    (OUT / "topup_plan.json").write_text(json.dumps(summary, indent=1))

    fig, ax = plt.subplots(figsize=(12, 9), constrained_layout=True)
    for pr in scene.prisms:
        if "wall" in pr.name:
            continue
        ax.add_patch(Rectangle((pr.xmin, pr.ymin), pr.xmax - pr.xmin, pr.ymax - pr.ymin,
                               fc="#e6dcc6", ec="#9c8358", lw=0.5, zorder=0))
    for role, colour in ROLE_COLOUR.items():
        pts = v5_xy[[r == role for r in v5_role]]
        ax.scatter(pts[:, 0], pts[:, 1], s=5, c=colour, alpha=0.35, lw=0, label=f"v5 {role} ({len(pts)})")
    sp = np.array(supplement)
    ax.scatter(sp[:, 0], sp[:, 1], s=30, c="#2f8f5b", marker="s", edgecolors="k", lw=0.6,
               label=f"captured C supplement, D_mu ({len(sp)})", zorder=3)
    for role in ("D_mu", "D_R", "D_dev"):
        pts = [(p["x"], p["y"]) for p in kept if p["role"] == role]
        if pts:
            ax.scatter(*zip(*pts), s=36, c=ROLE_COLOUR[role], edgecolors="k", lw=0.7,
                       label=f"new {role} (+{len(pts)})", zorder=3)
    if dropped:
        ax.scatter(*zip(*dropped), s=46, c="none", edgecolors="#c0392b", marker="X", lw=1.2,
                   label=f"dropped, next to final audit ({len(dropped)})", zorder=4)
    for name, (x, y) in CAMS.items():
        ax.plot(x, y, "s", color="#1f3b73", ms=9, zorder=5)
        ax.annotate(name, (x, y), xytext=(5, -12) if y > 0 else (5, 5),
                    textcoords="offset points", fontsize=12, weight="bold", color="#1f3b73")
    ax.set_xlim(-12, 12); ax.set_ylim(-10, 10); ax.set_aspect("equal")
    total = len(v5) + len(supplement) + len(kept)
    ax.set_title(f"v8 uniform dataset: {len(v5)} v5 + {len(supplement)} supplement + {len(kept)} "
                 f"top-up = {total} positions", fontsize=13)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=9, frameon=False)
    fig.savefig(OUT / "topup_plan_map.png", dpi=140)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("positions", "dropped")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
