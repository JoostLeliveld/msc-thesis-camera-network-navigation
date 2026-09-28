from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from planning.nodes.efe_agent_node import (
    EfeAgentNode,
    _compress_collinear_waypoints,
    _ff_fb_arrival_speed_cap,
    _ff_fb_forward_speed,
    _ff_fb_path_guidance,
    _geometric_route_time_cost,
    _preview_corner_speed_limit,
    _route_length_from,
    _route_states,
    _tracking_waypoints,
    _waypoint_reached_or_passed,
)
from planning.core.dynamics import unicycle_step


def test_straight_path_keeps_one_metre_per_second_ceiling():
    path = np.asarray(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)))
    assert _preview_corner_speed_limit(
        path, 0, np.asarray((0.0, 0.0)), v_max=1.0
    ) == pytest.approx(1.0)


def test_ninety_degree_corner_triggers_braking_before_turn():
    path = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (1.0, 2.0)))
    limit = _preview_corner_speed_limit(
        path, 1, np.asarray((0.80, 0.0)), v_max=1.0
    )
    assert 0.30 < limit < 1.0


def test_corner_outside_preview_does_not_slow_early_straight():
    path = np.asarray(((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)))
    assert _preview_corner_speed_limit(
        path, 0, np.asarray((0.0, 0.0)), v_max=1.0, preview_m=0.9
    ) == pytest.approx(1.0)


def test_tracker_retains_previous_corner_as_cross_track_anchor():
    route = [(-10.0, 7.75), (-3.57, 7.75), (-3.57, 6.0)]
    path = _tracking_waypoints(route, 1, np.asarray((-8.0, 7.75)))
    assert path.tolist() == [
        [-10.0, 7.75],
        [-3.57, 7.75],
        [-3.57, 6.0],
    ]


def test_tracker_uses_current_state_before_first_route_waypoint():
    route = [(-10.0, -2.0), (-3.57, -2.0), (-3.57, 6.0)]
    path = _tracking_waypoints(route, 0, np.asarray((-10.0, -6.0)))
    assert path.tolist() == [
        [-10.0, -6.0],
        [-10.0, -2.0],
        [-3.57, -2.0],
        [-3.57, 6.0],
    ]


def test_dense_waypoint_is_advanced_after_belief_crosses_its_plane():
    route = [(0.0, 0.0), (0.2, 0.0), (0.4, 0.0)]
    assert _waypoint_reached_or_passed(
        route, 1, np.asarray((0.31, 0.03)), arrival_radius_m=0.1,
    )


def test_first_waypoint_is_advanced_after_belief_crosses_its_outgoing_plane():
    route = [(0.2, 0.0), (0.4, 0.0), (0.6, 0.0)]
    assert _waypoint_reached_or_passed(
        route, 0, np.asarray((0.31, 0.03)), arrival_radius_m=0.1,
    )


def test_first_waypoint_is_not_advanced_before_its_outgoing_plane():
    route = [(0.2, 0.0), (0.4, 0.0), (0.6, 0.0)]
    assert not _waypoint_reached_or_passed(
        route, 0, np.asarray((0.09, 0.03)), arrival_radius_m=0.1,
    )


def test_dense_waypoint_is_not_advanced_before_its_plane():
    route = [(0.0, 0.0), (0.2, 0.0), (0.4, 0.0)]
    assert not _waypoint_reached_or_passed(
        route, 1, np.asarray((0.09, 0.03)), arrival_radius_m=0.1,
    )


def test_ff_fb_collapses_dense_straight_samples_but_retains_corner():
    dense = np.asarray(((0.0, 0.0), (0.2, 0.0), (0.4, 0.0),
                        (0.4, 0.2), (0.4, 0.4)))
    assert _compress_collinear_waypoints(dense).tolist() == [
        [0.0, 0.0], [0.4, 0.0], [0.4, 0.4],
    ]


def test_geometric_route_length_includes_start_to_first_waypoint():
    assert _route_length_from(
        np.asarray((-10.0, -6.0)),
        [(-10.0, 2.0), (-3.5, 2.0), (-3.5, 6.0)],
    ) == pytest.approx(18.5)


def test_geometric_route_states_include_start_and_corner_headings():
    states = _route_states(
        np.asarray((0.0, 0.0, -1.0)), [(0.0, 2.0), (3.0, 2.0)]
    )
    assert states[:, :2].tolist() == [[0.0, 0.0], [0.0, 2.0], [3.0, 2.0]]
    assert states[0, 2] == pytest.approx(np.pi / 2.0)
    assert states[1, 2] == pytest.approx(0.0)


def test_ff_fb_pivots_only_for_a_near_reversal():
    assert _ff_fb_forward_speed(
        0.30, 0.30, -0.8, 2.70, v_max=1.0, yaw_gate_rad=0.65,
        crawl_speed_mps=0.18, pivot_heading_error_rad=2.60,
    ) == 0.0


def test_ff_fb_reduces_speed_smoothly_for_moderate_heading_error():
    at_gate = _ff_fb_forward_speed(
        0.30, 0.30, -0.8, 0.65, v_max=1.0, yaw_gate_rad=0.65,
        crawl_speed_mps=0.18, pivot_heading_error_rad=2.60,
    )
    through_turn = _ff_fb_forward_speed(
        0.30, 0.30, -0.8, 1.20, v_max=1.0, yaw_gate_rad=0.65,
        crawl_speed_mps=0.18, pivot_heading_error_rad=2.60,
    )
    right_angle = _ff_fb_forward_speed(
        0.30, 0.30, -0.8, np.pi / 2.0, v_max=1.0, yaw_gate_rad=0.65,
        crawl_speed_mps=0.18, pivot_heading_error_rad=2.60,
    )
    assert at_gate > through_turn > right_angle > 0.0
    assert right_angle == pytest.approx(0.18)


def test_ff_fb_uses_arc_for_a_nominal_right_angle_corner():
    speed = _ff_fb_forward_speed(
        0.30, 0.30, 0.8, np.pi / 2.0, v_max=1.0, yaw_gate_rad=0.65,
        crawl_speed_mps=0.18, pivot_heading_error_rad=2.60,
    )
    assert speed == pytest.approx(0.18)


def test_ff_fb_keeps_one_metre_per_second_on_aligned_straight():
    assert _ff_fb_forward_speed(
        1.0, 1.0, 0.0, 0.0, v_max=1.0, yaw_gate_rad=0.60
    ) == pytest.approx(1.0)


def test_ff_fb_respects_arrival_or_corner_speed_cap():
    assert _ff_fb_forward_speed(
        1.0, 0.30, 0.0, 0.0, v_max=1.0, yaw_gate_rad=0.60
    ) == pytest.approx(0.30)


def test_ff_fb_does_not_brake_when_target_does_not_require_capture():
    assert _ff_fb_arrival_speed_cap(
        0.2, must_capture=False, v_max=1.0,
    ) == pytest.approx(1.0)


def test_ff_fb_brakes_for_retained_corner_or_final_target():
    assert _ff_fb_arrival_speed_cap(
        0.2, must_capture=True, v_max=1.0,
    ) == pytest.approx(0.16)


def test_ff_fb_corner_capture_brakes_before_tight_ninety_degree_turn():
    node = SimpleNamespace(
        local_horizon=12,
        dt=0.25,
        v_max=1.0,
        waypoint_spacing_m=0.2,
        simple_tracker_yaw_gate_rad=0.60,
        _waypoints=[(0.0, 0.0), (1.0, 0.0), (1.0, 2.0)],
        _wp_idx=1,
    )
    node._waypoint_array = lambda state_xy: _tracking_waypoints(
        node._waypoints, node._wp_idx, state_xy,
    )
    controls = EfeAgentNode._ff_fb_plan(
        node, np.asarray((0.81, 0.0, 0.0)),
    )
    # The fillet remains a moving arc, but heading-based reduction prevents the
    # robot from carrying straight-line speed through the outside of the turn:
    # the turn-overshoot cap keeps the swing within the 0.10 m safety margin.
    assert 0.05 < controls[0, 0] < 0.50


def test_ff_fb_path_guidance_preserves_centreline_speed():
    heading_error, angular_velocity, speed_cap = _ff_fb_path_guidance(
        np.pi / 2.0, np.pi / 2.0, 0.0, v_max=1.0, w_limit=0.75,
    )
    assert heading_error == pytest.approx(0.0)
    assert angular_velocity == pytest.approx(0.0)
    assert speed_cap == pytest.approx(1.0)


def test_ff_fb_path_guidance_turns_back_and_slows_when_right_of_northbound_path():
    heading_error, angular_velocity, speed_cap = _ff_fb_path_guidance(
        np.pi / 2.0, np.pi / 2.0, -0.10, v_max=1.0, w_limit=0.75,
    )
    assert heading_error > 0.0
    assert angular_velocity > 0.0
    assert speed_cap < 0.56


def test_ff_fb_path_guidance_is_mirror_symmetric():
    right_error, right_w, right_speed = _ff_fb_path_guidance(
        0.0, 0.0, -0.08, v_max=1.0, w_limit=0.75,
    )
    left_error, left_w, left_speed = _ff_fb_path_guidance(
        0.0, 0.0, 0.08, v_max=1.0, w_limit=0.75,
    )
    assert right_error == pytest.approx(-left_error)
    assert right_w == pytest.approx(-left_w)
    assert right_speed == pytest.approx(left_speed)


def test_ff_fb_closed_loop_converges_from_lateral_departure_without_crossing_farther_out():
    node = SimpleNamespace(
        local_horizon=12,
        dt=0.25,
        v_max=1.0,
        waypoint_spacing_m=0.2,
        simple_tracker_yaw_gate_rad=0.60,
        _waypoints=[(0.0, 0.0), (0.0, 10.0)],
        _wp_idx=1,
    )
    node._waypoint_array = lambda state_xy: _tracking_waypoints(
        node._waypoints, node._wp_idx, state_xy,
    )
    state = np.asarray((0.20, 0.0, np.pi / 2.0))
    lateral_positions = []
    for _ in range(60):
        controls = EfeAgentNode._ff_fb_plan(node, state)
        state = unicycle_step(state, controls[0], node.dt)
        lateral_positions.append(float(state[0]))
        if state[1] >= 9.8:
            break
    assert max(lateral_positions) <= 0.20 + 1.0e-9
    assert abs(state[0]) < 0.02


def test_geometric_time_cost_prefers_heading_aligned_route_over_180_pivot():
    start = np.asarray((0.0, 0.0, -np.pi / 2.0))
    short_with_uturn = [(0.0, 4.0), (4.0, 4.0), (4.0, 8.0)]
    wider_aligned_route = [(0.0, -2.0), (4.0, -2.0), (4.0, 8.0)]
    assert _route_length_from(start[:2], short_with_uturn) < _route_length_from(
        start[:2], wider_aligned_route
    )
    assert _geometric_route_time_cost(
        start, wider_aligned_route, v_max=1.0
    ) < _geometric_route_time_cost(start, short_with_uturn, v_max=1.0)


def test_turn_overshoot_cap_bounds_lateral_travel_to_the_safety_margin():
    import math
    from planning.nodes.efe_agent_node import TURN_OVERSHOOT_BUDGET_M, _ff_fb_turn_overshoot_cap
    w = 0.8
    assert _ff_fb_turn_overshoot_cap(0.0, w_limit=w) == math.inf
    for error in (0.2, 0.6, 1.0, math.pi / 2):
        v = _ff_fb_turn_overshoot_cap(error, w_limit=w)
        # radius v / w times (1 - cos e) is exactly the budget
        assert math.isclose((v / w) * (1 - math.cos(error)), TURN_OVERSHOOT_BUDGET_M)
    # beyond 90 degrees the cap saturates at the right-angle value
    assert _ff_fb_turn_overshoot_cap(2.5, w_limit=w) == _ff_fb_turn_overshoot_cap(math.pi / 2, w_limit=w)


def test_turn_overshoot_budget_equals_the_world_geometric_safety_margin():
    import inspect
    from experiments.core import world_profiles
    from planning.nodes.efe_agent_node import TURN_OVERSHOOT_BUDGET_M
    for fn in (world_profiles.serialize_collision_geometry_from_world,
               world_profiles.serialize_driveable_geometry_from_profile):
        assert inspect.signature(fn).parameters["safety_margin_m"].default == TURN_OVERSHOOT_BUDGET_M
