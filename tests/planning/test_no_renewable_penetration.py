"""Execution must enforce the oriented body, including between control poses."""
from types import SimpleNamespace
import numpy as np
import pytest
from planning.nodes.efe_agent_node import EfeAgentNode
from planning.planners.base_planner import UnicyclePlannerBase
from unav_common.rectangular_footprint import RectangularFootprint


def node_for(model, lane=False):
    planner = SimpleNamespace(collision_cost_model=None if lane else object(),
        nogo_cost_model=SimpleNamespace(enabled=True) if lane else None,
        _footprint_collision_model=model)
    planner.collision_clearance_state_np = lambda s: UnicyclePlannerBase.collision_clearance_state_np(planner, s)
    planner.collision_sweep_clearance_np = model.sweep_clearance
    planner.driveable_clearance_state_np = model.clearance
    planner.driveable_sweep_clearance_np = model.sweep_clearance
    return SimpleNamespace(planner=planner, dt=.25)


def prism(xmin, ymin, xmax, ymax):
    return SimpleNamespace(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)


@pytest.mark.parametrize('wall_distance, expected', [(.405, 0), (.49, 1)])
def test_body_collision_uses_rectangle_not_centre_or_circle(wall_distance, expected):
    node = node_for(RectangularFootprint([prism(wall_distance, -2, 1, 2)]))
    result = EfeAgentNode._simple_plan_safe_to_execute(node, np.array([[.04, 0.]]), np.zeros(3))
    assert result.safe_steps == expected


@pytest.mark.parametrize('lane', [False, True])
def test_outward_motion_has_no_renewable_allowance(lane):
    scene = [prism(-2, -2, 2, .276)] if lane else [prism(-2, .276, 2, 2)]
    node = node_for(RectangularFootprint(scene, keep_in=lane), lane)
    for _ in range(30):
        result = EfeAgentNode._simple_plan_safe_to_execute(node, np.array([[0., .1]]), np.zeros(3))
        assert result.safe_steps == 0


def test_existing_body_overlap_does_not_authorize_recovery_motion():
    node = node_for(RectangularFootprint([prism(.39, -2, 1, 2)]))
    result = EfeAgentNode._simple_plan_safe_to_execute(node, np.array([[-.04, 0.]]), np.zeros(3))
    assert result.safe_steps == 0


def test_tracker_checks_turn_between_clear_endpoints():
    node = node_for(RectangularFootprint([prism(.42, -.02, .44, .02)]))
    node.dt = 1.
    result = EfeAgentNode._simple_plan_safe_to_execute(node, np.array([[0., np.pi]]), np.zeros(3))
    assert result.safe_steps == 0
