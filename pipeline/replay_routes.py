#!/usr/bin/env python3
"""Replay every solved route through the ff_fb follower with the campaign settings.

METHOD amendment E: every solved route must keep the robot footprint inside the driveable
region when followed by the real follower code (`EfeAgentNode._ff_fb_plan`) on the true
state, before the campaign. The footprint is swept between steps against the physical
region (zero margin, `pipeline/score_collisions.py`) and, for information, against the
planner's inflated obstacles and shrunk boundary.

Writes a follower_replay_check.json beside the selected routes; exits non-zero if any route fails to
arrive or leaves the physical region.

    python3 pipeline/replay_routes.py --routes logs/thesis/final_campaign/routes
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src/planning"), str(REPO / "src/unav_common"), str(REPO / "src/experiments")]
from experiments.core.world_profiles import (  # noqa: E402
    serialize_collision_geometry_from_world, serialize_driveable_geometry_from_profile)
from planning.nodes.efe_agent_node import EfeAgentNode, _tracking_waypoints, _waypoint_reached_or_passed  # noqa: E402
from unav_common.occlusion_geometry import scene_from_json  # noqa: E402
from unav_common.rectangular_footprint import RectangularFootprint, constant_twist_pose  # noqa: E402
from pipeline.score_collisions import DriveableRegion  # noqa: E402

WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
ARRIVAL_M = 0.20


def setup():
    profile = yaml.safe_load((REPO / "src/experiments/config/world_profiles.yaml").read_text())["worlds"][WORLD.name]
    cfg = yaml.safe_load((REPO / "pipeline/execution_template.yaml").read_text())
    coll = serialize_collision_geometry_from_world(
        str(WORLD), model_names=tuple(profile["collision_model_names"]),
        include_names=tuple(profile["collision_include_names"]), profile=profile)
    planner = (RectangularFootprint(scene_from_json(coll).prisms, 0.80, 0.55),
               RectangularFootprint(scene_from_json(serialize_driveable_geometry_from_profile(profile)).prisms,
                                    0.80, 0.55, keep_in=True))
    tasks = {t["name"]: t for t in yaml.safe_load((REPO / "pipeline/tasks.yaml").read_text())["tasks"][WORLD.name]}
    return cfg, planner, DriveableRegion.for_world(), tasks


def replay(cfg, planner, physical, start, goal, waypoints, max_steps=1200):
    node = SimpleNamespace(
        local_horizon=12, dt=float(cfg["dt"]), v_max=float(cfg["v_max"]),
        waypoint_spacing_m=float(cfg["waypoint_spacing_m"]),
        simple_tracker_yaw_gate_rad=float(cfg["simple_tracker_yaw_gate_rad"]),
        ff_fb_turn_rate_limit_rad_s=float(cfg["ff_fb_turn_rate_limit_rad_s"]),
        ff_fb_corner_crawl_speed_mps=float(cfg["ff_fb_corner_crawl_speed_mps"]),
        ff_fb_pivot_heading_error_rad=float(cfg["ff_fb_pivot_heading_error_rad"]),
        _waypoints=waypoints, _wp_idx=1)
    node._waypoint_array = lambda xy: _tracking_waypoints(node._waypoints, node._wp_idx, xy)
    state = np.asarray((start["x"], start["y"], start["yaw"]), float)
    worst = {"physical": np.inf, "planner": np.inf}
    for step in range(max_steps):
        while node._wp_idx < len(waypoints) - 1 and _waypoint_reached_or_passed(
                waypoints, node._wp_idx, state[:2], arrival_radius_m=float(cfg["waypoint_arrival_radius_m"])):
            node._wp_idx += 1
        cmd = EfeAgentNode._ff_fb_plan(node, state)[0]
        end = constant_twist_pose(state, cmd, node.dt)
        worst["physical"] = min(worst["physical"],
                                physical.obstacles.sweep_clearance(state, end, control=cmd, dt=node.dt),
                                physical.boundary.sweep_clearance(state, end, control=cmd, dt=node.dt))
        worst["planner"] = min(worst["planner"], *(p.sweep_clearance(state, end, control=cmd, dt=node.dt) for p in planner))
        state = end
        if np.linalg.norm(state[:2] - (goal["x"], goal["y"])) <= ARRIVAL_M:
            return True, (step + 1) * node.dt, worst
    return False, max_steps * node.dt, worst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=Path, default=REPO / "logs/thesis/final_campaign/routes",
                        help="Directory containing one solved-route directory per task.")
    args = parser.parse_args()
    routes = args.routes.resolve()
    cfg, planner, physical, tasks = setup()
    rows, failing = [], 0
    for tdir in sorted(p for p in routes.iterdir() if p.is_dir() and not p.name.endswith(".incomplete")):
        t = tasks[tdir.name]
        for rf in sorted(tdir.glob("*.route.json")):
            wps = [tuple(p) for p in json.loads(rf.read_text())]
            ok, sim_s, w = replay(cfg, planner, physical, t["start"], t["goal"], wps)
            bad = bool((not ok) or w["physical"] < 0)
            failing += bad
            rows.append({"task": tdir.name, "condition": rf.name.split(".")[0], "arrived": bool(ok),
                         "sim_time_s": round(sim_s, 2), "min_physical_clearance_m": round(float(w["physical"]), 4),
                         "min_planner_clearance_m": round(float(w["planner"]), 4), "failed": bad})
            print(f"{tdir.name:46s} {rows[-1]['condition']:20s} arrived={ok!s:5s} phys={w['physical']:7.3f}"
                  f" plan={w['planner']:7.3f}{'  <-- FAIL' if bad else ''}")
    (routes / "follower_replay_check.json").write_text(
        json.dumps({"routes": rows, "failing": failing}, indent=1) + "\n")
    print(f"failing: {failing} of {len(rows)}")
    return 1 if failing or not rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
