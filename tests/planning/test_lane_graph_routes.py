from __future__ import annotations

import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from planning.planners.base_planner import UnicyclePlannerBase
from unav_common.lane_graph_routes import generate_route_seeds
from unav_common.occlusion_geometry import (
    parse_collision_scene_from_world,
    scene_from_json,
    signed_distance_to_union_xy,
)
from experiments.core.world_profiles import serialize_driveable_geometry_from_profile

REPO = Path(__file__).resolve().parents[2]


def _geometry(name: str, xmin: float, xmax: float, ymin: float, ymax: float) -> dict:
    return {
        "name": name,
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "zmin": 0.0,
        "zmax": 1.0,
    }
















def test_route_seed_waypoint_arrival_scales_with_global_control_step() -> None:
    planner = object.__new__(UnicyclePlannerBase)
    planner.horizon = 80
    planner.dt = 0.4
    planner.v_max = 0.6
    planner.v_min = -0.6
    planner.w_max = 1.0
    planner.w_min = -1.0
    planner.process_noise_xy = 0.01
    planner.process_noise_theta = 0.02

    controls = planner._controls_for_waypoints(
        np.array([0.0, 0.0, 0.0]),
        [(1.0, 0.0), (1.0, 1.0), (2.0, 1.0)],
    ).reshape(-1, 2)

    state = np.array([0.0, 0.0, 0.0])
    covariance = np.eye(3) * 1e-6
    for control in controls:
        state, covariance = planner.predict(state, covariance, control)
    assert np.linalg.norm(state[:2] - np.array([2.0, 1.0])) < 0.35


def test_multistart_terminal_gate_prefers_goal_reaching_candidate() -> None:
    planner = object.__new__(UnicyclePlannerBase)
    planner.optimizer_terminal_goal_tolerance_m = 0.25

    assert planner._terminal_goal_feasible(
        {"terminal_goal_distance_pred": 0.25}
    )
    assert not planner._terminal_goal_feasible(
        {"terminal_goal_distance_pred": 12.0}
    )
    assert planner._prefer_candidate(
        candidate_valid=True,
        candidate_goal_feasible=True,
        candidate_cost=1000.0,
        incumbent_valid=True,
        incumbent_goal_feasible=False,
        incumbent_cost=1.0,
    )


def test_multistart_terminal_gate_keeps_safety_ahead_of_goal_completion() -> None:
    planner = object.__new__(UnicyclePlannerBase)
    planner.optimizer_terminal_goal_tolerance_m = 0.25

    assert not planner._prefer_candidate(
        candidate_valid=False,
        candidate_goal_feasible=True,
        candidate_cost=1.0,
        incumbent_valid=True,
        incumbent_goal_feasible=False,
        incumbent_cost=1000.0,
    )
