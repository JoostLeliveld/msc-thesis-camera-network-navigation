#!/usr/bin/env python3
"""Offline collision score: did the robot footprint ever leave the driveable region?

The driveable region is the world profile's site boundary minus every collision object of
the world (the model groups and the included loose objects), at zero margin. A run
collides when the 0.80 x 0.55 m footprint at its true pose leaves that region at any
instant. The true poses come from the run's ground_truth_pose.csv (every sample, in
arrival order); each consecutive pair is checked with a certified sweep, so a crossing
between two samples is caught. Ground truth is used here only, after the run.

    python3 pipeline/score_collisions.py RUN_DIR [RUN_DIR ...] [--out scores.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src/unav_common"))
from unav_common.occlusion_geometry import profile_collision_scene  # noqa: E402
from unav_common.rectangular_footprint import RectangularFootprint  # noqa: E402

WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
WORLD_PROFILES = REPO / "src/experiments/config/world_profiles.yaml"
ROBOT_TARGET = REPO / "pipeline/capture/robot_target_manifest.json"


class DriveableRegion:
    """Site boundary (keep-in) minus collision boxes (keep-out) for one robot footprint."""

    def __init__(self, obstacles, boundary, length: float, width: float):
        self.obstacles = RectangularFootprint(obstacles, length=length, width=width)
        self.boundary = RectangularFootprint([boundary], length=length, width=width, keep_in=True)

    @classmethod
    def for_world(cls) -> "DriveableRegion":
        profile = yaml.safe_load(WORLD_PROFILES.read_text(encoding="utf-8"))["worlds"][WORLD.name]
        sites = [r for r in profile["known_2d_regions"] if r.get("type") == "site_boundary"]
        if len(sites) != 1:
            raise ValueError("world profile must declare exactly one site_boundary region")
        s = sites[0]
        boundary = SimpleNamespace(xmin=float(s["xmin"]), xmax=float(s["xmax"]),
                                   ymin=float(s["ymin"]), ymax=float(s["ymax"]))
        body = json.loads(ROBOT_TARGET.read_text(encoding="utf-8"))["physical_contract"]
        return cls(profile_collision_scene(str(WORLD), profile).prisms, boundary,
                   float(body["body_length_m"]), float(body["body_width_m"]))

    def clearance(self, pose) -> tuple[float, float]:
        """(obstacle clearance, boundary clearance); negative means outside the region."""
        return self.obstacles.clearance(pose), self.boundary.clearance(pose)

    def segment_clear(self, start, end) -> bool:
        """Certified: the footprint stays in the region along the straight pose segment."""
        return (self.obstacles.sweep_clearance(start, end) >= 0.0
                and self.boundary.sweep_clearance(start, end) >= 0.0)


def read_poses(path: Path) -> list[tuple[float, float, float, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [(float(r["stamp_s"]), float(r["x"]), float(r["y"]), float(r["yaw"]))
                for r in csv.DictReader(handle)]


def score_poses(region: DriveableRegion, poses) -> dict:
    if not poses:
        return {"scored": False, "reason": "no ground-truth poses"}
    min_obstacle = min_boundary = math.inf
    first_exit = None
    for i, (stamp, x, y, yaw) in enumerate(poses):
        obstacle, boundary = region.clearance((x, y, yaw))
        min_obstacle, min_boundary = min(min_obstacle, obstacle), min(min_boundary, boundary)
        left = obstacle < 0.0 or boundary < 0.0
        if not left and i > 0:
            left = not region.segment_clear(poses[i - 1][1:], (x, y, yaw))
        if left and first_exit is None:
            first_exit = {"sample": i, "stamp_s": stamp, "x": x, "y": y, "yaw": yaw,
                          "kind": ("obstacle" if obstacle < 0.0 else
                                   "boundary" if boundary < 0.0 else "between_samples")}
    gaps = [b[0] - a[0] for a, b in zip(poses, poses[1:])]
    return {
        "scored": True,
        "collision": first_exit is not None,
        "first_exit": first_exit,
        "min_obstacle_clearance_m": min_obstacle,
        "min_boundary_clearance_m": min_boundary,
        "samples": len(poses),
        "max_sample_gap_s": max(gaps) if gaps else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    region = DriveableRegion.for_world()
    scores = {}
    for run in args.run_dirs:
        path = run / "ground_truth_pose.csv"
        scores[str(run)] = (score_poses(region, read_poses(path)) if path.is_file()
                            else {"scored": False, "reason": f"missing {path.name}"})
    text = json.dumps(scores, indent=2, allow_nan=False, default=str)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
