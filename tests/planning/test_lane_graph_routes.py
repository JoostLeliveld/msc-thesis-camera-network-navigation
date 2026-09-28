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


def test_four_cam_lanes_are_inset_from_obstacles_like_paper1() -> None:
    """Lanes must be a conservative corridor, not the raw geometric gap.

    Paper 1 draws its lanes inset a measured median 0.25 m from obstacle faces;
    that inset is where keep_in's safety margin actually comes from. Drawing the
    4-camera lanes flush against the rack faces left nothing after corner
    rounding (0.046 m planned clearance vs the paper's 0.358 m floor).
    """
    profiles = yaml.safe_load(
        (REPO / "src/experiments/config/world_profiles.yaml").read_text("utf-8")
    )
    regions = profiles["worlds"]["warehouse_full_4cam.world.sdf"]["known_2d_regions"]
    lanes = [r for r in regions if r.get("type") == "traversable"]
    # site_boundary is the keep-in envelope, not an obstacle: lanes live inside it.
    blockers = [
        r for r in regions
        if str(r.get("type", "")).startswith("non_driveable")
    ]
    assert lanes, "no traversable lanes declared"

    for lane in lanes:
        for blocker in blockers:
            x_overlap = not (blocker["xmax"] <= lane["xmin"] or blocker["xmin"] >= lane["xmax"])
            y_overlap = not (blocker["ymax"] <= lane["ymin"] or blocker["ymin"] >= lane["ymax"])
            assert not (x_overlap and y_overlap), (
                f"lane {lane['name']} overlaps {blocker['name']}"
            )


def test_four_cam_named_no_go_regions_are_safety_envelopes_not_exact_fits() -> None:
    profiles = yaml.safe_load(
        (REPO / "src/experiments/config/world_profiles.yaml").read_text("utf-8")
    )
    regions = profiles["worlds"]["warehouse_full_4cam.world.sdf"]["known_2d_regions"]
    blockers = {
        region["name"]: region
        for region in regions
        if str(region.get("type", "")).startswith("non_driveable")
    }
    expected = {
        "west_wall_backed_one_sided_shelf": (-12.23, -11.04, -6.82, 7.22),
        "central_support_pillar": (-0.57, 0.57, -1.47, -0.33),
    }
    assert set(blockers) == set(expected)
    for name, bounds in expected.items():
        region = blockers[name]
        actual = tuple(float(region[key]) for key in ("xmin", "xmax", "ymin", "ymax"))
        assert actual == bounds


def test_four_cam_sdf_collisions_do_not_occupy_declared_through_lanes() -> None:
    profiles = yaml.safe_load(
        (REPO / "src/experiments/config/world_profiles.yaml").read_text("utf-8")
    )
    regions = profiles["worlds"]["warehouse_full_4cam.world.sdf"]["known_2d_regions"]
    lanes = [region for region in regions if region.get("type") == "traversable"]
    scene = parse_collision_scene_from_world(
        str(REPO / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf"),
        model_names=("warehouse_rack_occluders",),
    )
    overlaps = []
    for prism in scene.prisms:
        for lane in lanes:
            overlap_x = min(prism.xmax, lane["xmax"]) - max(prism.xmin, lane["xmin"])
            overlap_y = min(prism.ymax, lane["ymax"]) - max(prism.ymin, lane["ymin"])
            if overlap_x > 1.0e-6 and overlap_y > 1.0e-6:
                overlaps.append((prism.name, lane["name"], overlap_x, overlap_y))
    assert not overlaps


def test_four_cam_safety_envelopes_do_not_overlap_driveable_union() -> None:
    """The cyan planner map must not cross any green operational envelope."""
    profiles = yaml.safe_load(
        (REPO / "src/experiments/config/world_profiles.yaml").read_text("utf-8")
    )
    regions = profiles["worlds"]["warehouse_full_4cam.world.sdf"]["known_2d_regions"]
    lanes = [region for region in regions if region.get("type") == "traversable"]
    scene = parse_collision_scene_from_world(
        str(REPO / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf"),
        model_names=("warehouse_rack_occluders",),
    )

    margin = 0.32
    overlaps = []
    for prism in scene.prisms:
        envelope = {
            "xmin": prism.xmin - margin,
            "xmax": prism.xmax + margin,
            "ymin": prism.ymin - margin,
            "ymax": prism.ymax + margin,
        }
        for lane in lanes:
            overlap_x = min(envelope["xmax"], lane["xmax"]) - max(envelope["xmin"], lane["xmin"])
            overlap_y = min(envelope["ymax"], lane["ymax"]) - max(envelope["ymin"], lane["ymin"])
            if overlap_x > 1.0e-6 and overlap_y > 1.0e-6:
                overlaps.append((prism.name, lane["name"], overlap_x, overlap_y))
    assert not overlaps


def test_four_cam_external_props_clear_driveable_union() -> None:
    """Decorative apron props must remain outside cyan planner corridors."""
    profiles = yaml.safe_load(
        (REPO / "src/experiments/config/world_profiles.yaml").read_text("utf-8")
    )
    regions = profiles["worlds"]["warehouse_full_4cam.world.sdf"]["known_2d_regions"]
    lanes = [region for region in regions if region.get("type") == "traversable"]
    world = ET.parse(
        REPO / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf"
    ).getroot()
    includes = {
        include.findtext("name", ""): include
        for include in world.findall(".//world/include")
    }
    footprint_sizes = {
        "bucket_dock": (0.42, 0.42),
        "trashcan_nw": (0.55, 0.55),
        "clutterA_sw": (0.90, 0.75),
        "clutterC_ne": (1.00, 0.80),
        "clutterD_dock": (0.90, 0.80),
        "palletjack_apron": (1.35, 0.65),
        "desk_qc_ne": (1.55, 0.85),
    }

    overlaps = []
    clearance = 0.05
    for name, (size_x, size_y) in footprint_sizes.items():
        pose = [float(value) for value in includes[name].findtext("pose", "").split()]
        x, y, yaw = pose[0], pose[1], pose[5]
        half_x = 0.5 * (abs(math.cos(yaw)) * size_x + abs(math.sin(yaw)) * size_y)
        half_y = 0.5 * (abs(math.sin(yaw)) * size_x + abs(math.cos(yaw)) * size_y)
        bounds = {
            "xmin": x - half_x - clearance,
            "xmax": x + half_x + clearance,
            "ymin": y - half_y - clearance,
            "ymax": y + half_y + clearance,
        }
        for lane in lanes:
            overlap_x = min(bounds["xmax"], lane["xmax"]) - max(bounds["xmin"], lane["xmin"])
            overlap_y = min(bounds["ymax"], lane["ymax"]) - max(bounds["ymin"], lane["ymin"])
            if overlap_x > 1.0e-6 and overlap_y > 1.0e-6:
                overlaps.append((name, lane["name"], overlap_x, overlap_y))
    assert not overlaps


def test_four_cam_cross_aisles_retain_useful_width_between_envelopes() -> None:
    profiles = yaml.safe_load(
        (REPO / "src/experiments/config/world_profiles.yaml").read_text("utf-8")
    )
    regions = {
        region["name"]: region
        for region in profiles["worlds"]["warehouse_full_4cam.world.sdf"]["known_2d_regions"]
    }
    for name in ("lower_cross_aisle", "upper_cross_aisle"):
        width = float(regions[name]["ymax"]) - float(regions[name]["ymin"])
        assert np.isclose(width, 0.96)


def test_four_cam_task_control_points_are_driveable_and_not_blocked() -> None:
    profiles = yaml.safe_load(
        (REPO / "src/experiments/config/world_profiles.yaml").read_text("utf-8")
    )
    regions = profiles["worlds"]["warehouse_full_4cam.world.sdf"]["known_2d_regions"]
    lanes = [region for region in regions if region.get("type") == "traversable"]
    blockers = [
        region for region in regions
        if str(region.get("type", "")).startswith("non_driveable")
    ]
    tasks = yaml.safe_load(
        (REPO / "src/experiments/config/tasks.yaml").read_text("utf-8")
    )["tasks"]["warehouse_full_4cam.world.sdf"]

    issues = []
    for task in tasks:
        points = [("start", task.get("start")), ("goal", task.get("goal"))]
        points.extend(
            (f"waypoint[{index}]", point)
            for index, point in enumerate(task.get("waypoints", []))
        )
        for role, point in points:
            if not point:
                continue
            x, y = float(point["x"]), float(point["y"])
            inside_lane = any(
                lane["xmin"] <= x <= lane["xmax"]
                and lane["ymin"] <= y <= lane["ymax"]
                for lane in lanes
            )
            inside_blocker = any(
                blocker["xmin"] <= x <= blocker["xmax"]
                and blocker["ymin"] <= y <= blocker["ymax"]
                for blocker in blockers
            )
            if not inside_lane or inside_blocker:
                issues.append((task["name"], role, x, y, inside_lane, inside_blocker))
    assert not issues


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
