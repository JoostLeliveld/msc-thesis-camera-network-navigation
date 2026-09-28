"""ROS/Gazebo-free integration replay for route generation and ff_fb execution."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from experiments.core.world_profiles import (
    serialize_collision_geometry_from_world,
    serialize_driveable_geometry_from_profile,
)
from planning.nodes.efe_agent_node import (
    EfeAgentNode,
    _tracking_waypoints,
    _waypoint_reached_or_passed,
)
from unav_common.lane_graph_routes import generate_route_seeds
from unav_common.occlusion_geometry import scene_from_json
from unav_common.rectangular_footprint import RectangularFootprint, constant_twist_pose


ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / 'src/experiments/config/world_profiles.yaml'
WORLD_PATH = ROOT / 'src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf'
TASK_PATH = ROOT / 'pipeline/tasks.yaml'


@pytest.fixture(scope='module')
def offline_scene():
    profile = yaml.safe_load(PROFILE_PATH.read_text())['worlds'][WORLD_PATH.name]
    collision_json = serialize_collision_geometry_from_world(
        str(WORLD_PATH),
        model_names=tuple(profile['collision_model_names']),
        include_names=tuple(profile['collision_include_names']),
        profile=profile,
    )
    boundary_json = serialize_driveable_geometry_from_profile(profile)
    collision = RectangularFootprint(
        scene_from_json(collision_json).prisms, length=0.80, width=0.55,
    )
    boundary = RectangularFootprint(
        scene_from_json(boundary_json).prisms,
        length=0.80, width=0.55, keep_in=True,
    )
    tasks = yaml.safe_load(TASK_PATH.read_text())['tasks'][WORLD_PATH.name]
    return collision_json, collision, boundary, tasks


def route_length(start_xy, seed):
    points = [np.asarray(start_xy, dtype=float)] + [
        np.asarray(point, dtype=float) for point in seed['waypoints']
    ]
    return sum(float(np.linalg.norm(b - a)) for a, b in zip(points, points[1:]))


def execute_route(start, goal, waypoints, collision, boundary, *, max_steps=300):
    node = SimpleNamespace(
        local_horizon=12, dt=0.25, v_max=1.0,
        waypoint_spacing_m=0.20, simple_tracker_yaw_gate_rad=0.60,
        _waypoints=waypoints, _wp_idx=1,
    )
    node._waypoint_array = lambda xy: _tracking_waypoints(
        node._waypoints, node._wp_idx, xy,
    )
    state = np.asarray((start['x'], start['y'], start['yaw']), dtype=float)
    minimum_clearance = float('inf')
    for step in range(max_steps):
        while (
            node._wp_idx < len(waypoints) - 1
            and _waypoint_reached_or_passed(
                waypoints, node._wp_idx, state[:2], arrival_radius_m=0.10,
            )
        ):
            node._wp_idx += 1
        command = EfeAgentNode._ff_fb_plan(node, state)[0]
        end = constant_twist_pose(state, command, node.dt)
        clearances = (
            collision.sweep_clearance(
                state, end, control=command, dt=node.dt,
            ),
            boundary.sweep_clearance(
                state, end, control=command, dt=node.dt,
            ),
        )
        minimum_clearance = min(minimum_clearance, *clearances)
        if min(clearances) < 0.0:
            return False, state, step, minimum_clearance
        state = end
        if np.linalg.norm(state[:2] - (goal['x'], goal['y'])) <= 0.20:
            return True, state, step + 1, minimum_clearance
    return False, state, max_steps, minimum_clearance




def _declared_routes(task):
    start = (task['start']['x'], task['start']['y'])
    for seed in task['route_seeds']:
        yield seed['name'], [start] + [tuple(point) for point in seed['waypoints']]


def test_every_declared_route_candidate_executes_without_touching_geometry(offline_scene):
    """Each route the solver may choose is followed by ff_fb with the true footprint."""
    _collision_json, collision, boundary, tasks = offline_scene
    assert len(tasks) == 5
    failures = []
    for task in tasks:
        for name, waypoints in _declared_routes(task):
            arrived, final_state, _steps, clearance = execute_route(
                task['start'], task['goal'], waypoints, collision, boundary, max_steps=600)
            if not arrived or clearance < 0.0:
                failures.append((task['name'], name, arrived, round(clearance, 3),
                                 [round(v, 2) for v in final_state.tolist()]))
    assert not failures, failures


@pytest.mark.parametrize('dx,dy', [(0.05, 0.0), (-0.05, 0.0),
                                   (0.0, 0.05), (0.0, -0.05)])
def test_closed_loop_recovers_from_five_centimetre_handoff_displacement(
        offline_scene, dx, dy):
    _collision_json, collision, boundary, tasks = offline_scene
    task = tasks[0]
    name, waypoints = next(_declared_routes(task))
    displaced = dict(task['start'], x=task['start']['x'] + dx, y=task['start']['y'] + dy)
    arrived, final_state, _steps, clearance = execute_route(
        displaced, task['goal'], waypoints, collision, boundary, max_steps=600)
    assert arrived, (name, dx, dy, final_state.tolist(), clearance)
    assert clearance >= 0.0
