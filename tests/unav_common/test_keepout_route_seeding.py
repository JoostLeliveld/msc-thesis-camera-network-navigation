"""Route seeding must work against the keep-out (obstacle) map.

The declared map is the obstacle set: one box per collision footprint grown by a
fixed margin, everything else traversable. These tests pin the three things that
silently produced ZERO routes while being individually plausible:

  1. the payload must be tagged keepout_region, or the seeder applies keep-in
     semantics and every segment fails;
  2. corridor axes must be declared, because inferring them from obstacle boxes
     would place centre-lines inside obstacles;
  3. a segment through an obstacle must be rejected, and one down a clear aisle
     accepted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/unav_common"))
sys.path.insert(0, str(REPO / "src/experiments"))

from experiments.core.world_profiles import (  # noqa: E402
    serialize_collision_geometry_from_world,
    serialize_driveable_geometry_from_profile,
)
from unav_common.lane_graph_routes import (  # noqa: E402
    KEEPOUT_MODEL, _route_centres_from_geometry, _segment_free,
    generate_route_seeds, repair_route_seeds_for_footprint,
)
from unav_common.occlusion_geometry import scene_from_json  # noqa: E402

WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
PROFILES = REPO / "src/experiments/config/world_profiles.yaml"


@pytest.fixture(scope="module")
def geometry() -> str:
    profile = yaml.safe_load(PROFILES.read_text())["worlds"]["warehouse_v2.world.sdf"]
    return serialize_collision_geometry_from_world(
        str(WORLD),
        model_names=tuple(profile["collision_model_names"]),
        include_names=tuple(profile["collision_include_names"]),
        profile=profile,
    )


def test_payload_is_tagged_keepout_and_carries_axes(geometry: str) -> None:
    payload = json.loads(geometry)
    assert payload["model_name"] == KEEPOUT_MODEL
    assert payload["geometric_safety_margin_m"] == pytest.approx(0.10)
    assert len(payload["prisms"]) == 56, "include_names must be applied"
    assert payload["route_horizontal_centres"]
    assert payload["route_vertical_centres"]


def test_actual_collision_geometry_is_expanded_by_locked_margin(geometry: str) -> None:
    payload = json.loads(geometry)
    east_wall = next(
        item for item in payload['prisms']
        if item['name'] == 'warehouse_shell/wall_east:collision'
    )
    assert east_wall['xmin'] == pytest.approx(11.8)
    assert east_wall['xmax'] == pytest.approx(12.2)
    assert east_wall['ymin'] == pytest.approx(-10.1)
    assert east_wall['ymax'] == pytest.approx(10.1)


def test_keepout_geometry_without_declared_axes_is_refused() -> None:
    payload = {"model_name": KEEPOUT_MODEL, "prisms": [
        {"name": "o", "xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0,
         "zmin": 0.0, "zmax": 0.1}]}
    blob = json.dumps(payload)
    with pytest.raises(ValueError, match="must be declared"):
        _route_centres_from_geometry(blob, scene_from_json(blob).prisms)


def test_segment_rejects_obstacle_and_accepts_clear_aisle(geometry: str) -> None:
    prisms = scene_from_json(geometry).prisms
    # straight through bin_office at (-8.71, -7.45)
    assert not _segment_free(prisms, (-9.5, -7.45), (-8.0, -7.45), keep_out=True)
    # along the southern apron, clear of everything
    assert _segment_free(prisms, (-10.0, -6.25), (10.0, -6.25), keep_out=True)


@pytest.mark.parametrize(
    "start,goal",
    [((-8.1, -8.7), (9.5, 6.5)), ((-10.5, 0.0), (10.5, 0.0)),
     ((0.0, -8.5), (0.0, 8.5)), ((-10.0, 8.0), (10.0, -8.0))],
)
def test_route_seeds_exist_and_avoid_obstacles(geometry: str, start, goal) -> None:
    seeds = generate_route_seeds(geometry, start, goal)
    assert seeds, f"no route seeds for {start} -> {goal}"
    prisms = scene_from_json(geometry).prisms
    for seed in seeds:
        pts = [(float(x), float(y)) for x, y in seed["waypoints"]]
        for a, b in zip(pts, pts[1:]):
            assert _segment_free(prisms, a, b, keep_out=True), \
                f"seed {seed['name']} crosses an obstacle"


def test_near_start_lane_snap_is_collapsed_for_fair_seed_comparison(
        geometry: str) -> None:
    start = (-10.7, 8.55)
    seeds = generate_route_seeds(geometry, start, (9.6, -1.4))
    above = next(seed for seed in seeds if seed['name'] == 'above_cross_aisle')
    first = np.asarray(above['waypoints'][0], dtype=float)
    # The old first waypoint was (-10.7, 8.625), only 7.5 cm away. It forced
    # a turn-move-turn beside the boundary before beginning the actual route.
    assert np.linalg.norm(first - np.asarray(start)) > 0.10
    prisms = scene_from_json(geometry).prisms
    assert _segment_free(prisms, start, tuple(first), keep_out=True)


def test_blind_corridor_seeds_do_not_retrace_the_same_lane(geometry: str) -> None:
    start = np.asarray((-10.7, 8.55), dtype=float)
    seeds = generate_route_seeds(geometry, start, (9.6, -1.4))
    for seed in seeds:
        points = np.vstack((start, np.asarray(seed['waypoints'], dtype=float)))
        for a, b, c in zip(points, points[1:], points[2:]):
            ab, bc = b - a, c - b
            cross = ab[0] * bc[1] - ab[1] * bc[0]
            assert not (abs(cross) < 1e-9 and np.dot(ab, bc) < 0.0), \
                f"seed {seed['name']} retraces the same lane"


def test_feasible_routes_do_not_detour_beyond_both_endpoint_ordinates(
        geometry: str) -> None:
    start = np.asarray((-3.05, -5.2), dtype=float)
    goal = np.asarray((10.6, 2.54), dtype=float)
    seeds = generate_route_seeds(geometry, start, goal)
    assert seeds
    lower, upper = sorted((start[1], goal[1]))
    for seed in seeds:
        ys = np.asarray(seed['waypoints'], dtype=float)[:, 1]
        assert ys.min() >= lower - 1e-6
        assert ys.max() <= upper + 1e-6


def test_progressing_upper_aisle_seeds_are_repaired_by_geometry_only(
        geometry: str) -> None:
    profile = yaml.safe_load(PROFILES.read_text())["worlds"]["warehouse_v2.world.sdf"]
    boundary = serialize_driveable_geometry_from_profile(profile)
    start = (-10.7, 8.55, 0.0)
    seeds = generate_route_seeds(geometry, start[:2], (9.6, -1.4))
    repaired = {
        seed['name']: seed for seed in repair_route_seeds_for_footprint(
            seeds, geometry, boundary, start,
            robot_length_m=0.80, robot_width_m=0.55,
            target_clearance_m=0.02,
        )
    }

    connector = repaired['above_connector']
    assert connector['geometry_repaired'] is True
    assert connector['repaired_vertical_x_m'] == pytest.approx(9.925)
    assert connector['footprint_seed_clearance_m'] >= 0.02
    assert any(np.allclose(point, [9.925, 2.425])
               for point in connector['waypoints'])

    # A feasible cross-aisle lies between the endpoint ordinates, so the
    # generator must not offer the more northerly aisle as an away-from-goal
    # detour. This is a geometric plausibility rule, not a camera preference.
    assert 'above_cross_aisle_2' not in repaired

    # A hard-valid seed stays byte-for-byte geometrically unchanged; repair is
    # not allowed to encode a camera-model preference or soft-cost preference.
    raw_main = next(seed for seed in seeds if seed['name'] == 'below_main_aisle')
    assert repaired['below_main_aisle']['geometry_repaired'] is False
    assert repaired['below_main_aisle']['waypoints'] == raw_main['waypoints']
