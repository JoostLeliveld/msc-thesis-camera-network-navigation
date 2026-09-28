"""Regressions for state-clock consistency and concurrent runtime ownership."""
import threading
import json
from types import SimpleNamespace

import numpy as np
import pytest

from planning.nodes.efe_agent_node import EfeAgentNode
from test_planner_node_correction_wiring import _Clock, stamp
from test_planner_node_state_correction import make_state_node
from unav_common.terminal_stop import (
    TerminalStopRequest,
    terminal_stop_ack_from_json,
    terminal_stop_request_to_json,
)


class Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def publish_node():
    node = make_state_node(belief_stamp_s=9., now_s=10.)
    # These transaction tests exercise timestamp/revision ownership, not the
    # camera_xy_only odometry-heading adapter.
    node.heading_update_mode = 'coupled'
    node.odom_vel = np.zeros(2)
    node.planner_belief_pub = Publisher()
    node._resolve_plan_frame_id = lambda: 'map_bev'
    node._cmd_log = [(9., 0., 0.)]
    return node


def test_belief_header_is_the_prediction_target_even_if_computation_takes_time():
    node = publish_node()

    def prediction(m, P, u, age, target, *, motion_snapshot=None):
        assert node._stamp_to_float(target) == pytest.approx(10.)
        node._clock.seconds = 10.2
        return m, P

    node._predict_belief_to_now = prediction
    node._belief_publish_tick()
    assert len(node.planner_belief_pub.messages) == 1
    assert node._stamp_to_float(node.planner_belief_pub.messages[0].header.stamp) == pytest.approx(10.)


def test_prediction_from_superseded_anchor_is_not_published():
    node = publish_node()

    def prediction(m, P, u, age, target, *, motion_snapshot=None):
        node._commit_belief(np.array([1., 2., 0.]), np.eye(3), stamp(9.8))
        return m, P

    node._predict_belief_to_now = prediction
    node._belief_publish_tick()
    assert not node.planner_belief_pub.messages


@pytest.mark.parametrize('change', [
    {'schema_version': 99}, {'frame_id': 'camera_optical'},
    {'covariance_m2': [[.01, .1], [.1, .01]]},
    {'covariance_m2': [[.01, .001], [0., .01]]},
])
def test_malformed_envelope_cannot_enter_the_filter(change):
    node = make_state_node()
    node.require_state_correction_envelope = True
    node._seen_state_source_batch_ids = set()
    node._resolve_plan_frame_id = lambda: 'map_bev'
    errors = []
    node._fatal_experiment_stop = lambda context, exc: errors.append(str(exc))
    payload = dict(schema_version=1, frame_id='map_bev', source_batch_id='test_frame',
                   correction_stamp=9.95, xy=[.05, 0.], covariance_m2=[[.01, 0.], [0., .01]])
    payload.update(change)
    before = node.belief_m.copy()
    node._state_correction_envelope_cb(SimpleNamespace(data=json.dumps(payload)))
    assert errors
    np.testing.assert_array_equal(node.belief_m, before)
    assert not node.correction_assimilation_pub.published


def test_metric_corrections_cannot_compute_from_the_same_prior_concurrently():
    node = make_state_node(belief_stamp_s=9.9, now_s=10.3)
    node._correction_lock = threading.RLock()
    first_entered, release_first = threading.Event(), threading.Event()
    second_entered = threading.Event()
    errors = []

    def replay(m, P, from_stamp, to_stamp, u, dt, *, motion_snapshot=None):
        if node._stamp_to_float(to_stamp) < 10.1:
            first_entered.set()
            if not release_first.wait(2.):
                raise RuntimeError('test did not release first update')
        else:
            second_entered.set()
        return m, P, {}

    node._replay_cmd_log_interval = replay

    def update(t):
        try:
            node._apply_metric_correction(stamp(t), np.array([.05, 0.]), np.eye(2)*.03)
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=update, args=(10.,))
    second = threading.Thread(target=update, args=(10.2,))
    first.start()
    assert first_entered.wait(1.)
    second.start()
    overlapped = second_entered.wait(.1)
    release_first.set()
    first.join(2.); second.join(2.)
    assert not first.is_alive() and not second.is_alive()
    assert not errors
    assert not overlapped, 'two camera updates read an uncommitted prior'
    assert node._stamp_to_float(node.belief_stamp) == pytest.approx(10.2)


def command_node():
    node = object.__new__(EfeAgentNode)
    node._data_lock = threading.RLock()
    node._clock = _Clock(10.)
    node.get_clock = lambda: node._clock
    node.dt = .25
    node._active_controls = np.array([[.2, .1]])
    node._active_plan_started_at = _Clock(9.9).now()
    node._active_controls_original_len = 1
    node._last_latency_skip_steps = 0
    node._last_latency_skip_s = 0.
    node._current_wp_idx = 0
    node._current_wp_count = 2
    node._current_wp_target = np.array([1., 0.])
    for name in ('_current_wp_dist', '_current_desired_yaw', '_current_yaw_error',
                 '_current_tracking_yaw', '_current_tracking_yaw_source'):
        setattr(node, name, 0.)
    node.cmd_pub = Publisher()
    node.active_execution_diag_pub = Publisher()
    node.last_cmd = np.zeros(2)
    return node


def test_stop_cannot_be_undone_by_a_previously_snapshotted_command():
    node = command_node()
    clock = node._clock
    called = False

    def now():
        nonlocal called
        if not called:
            called = True
            node._publish_safe_stop_command()
        return _Clock(10.).now()

    clock.now = now
    node._publish_active_plan_command()
    assert node.cmd_pub.messages
    assert all(m.linear.x == 0 and m.angular.z == 0 for m in node.cmd_pub.messages)


def test_expiry_of_old_tape_cannot_clear_a_new_tape():
    node = command_node()
    replacement = np.array([[.1, -.2]])

    def now():
        node._active_controls = replacement
        node._active_plan_started_at = _Clock(10.5).now()
        return _Clock(10.5).now()

    node._clock.now = now
    node._publish_active_plan_command()
    assert node._active_controls is replacement
    assert not node.cmd_pub.messages


def test_inflight_plan_cannot_publish_motion_after_a_fatal_integrity_stop():
    node = command_node()
    node._fatal_stop_triggered = True
    node._pending_plan_started_at = None
    node.latency_compensate_plan_handoff = False
    node._result_safe_to_execute = lambda result: (True, '')
    node._after_plan_result(SimpleNamespace(controls=np.array([[.2, .3]])))
    assert node.cmd_pub.messages
    assert all(m.linear.x == m.angular.z == 0. for m in node.cmd_pub.messages)
    assert node._active_controls is None


def test_clock_rewind_cannot_restart_the_pre_reset_command_tape():
    node = command_node()
    node._clock.seconds = 0.
    node._publish_active_plan_command()
    assert node._active_controls is None
    assert node.cmd_pub.messages
    assert all(m.linear.x == m.angular.z == 0. for m in node.cmd_pub.messages)


def test_safe_stop_clears_the_held_execution_diagnostic():
    node = command_node()
    node._publish_safe_stop_command()
    assert node.active_execution_diag_pub.messages
    data = node.active_execution_diag_pub.messages[-1].data
    assert data[1] == 0.  # remaining tape time
    assert data[3] == 0.  # current tape length
    assert data[5] == data[6] == 0.  # command actually being published


def test_terminal_request_latches_planner_stop_before_acknowledgement():
    node = command_node()
    node._terminal_stop_requested = False
    node._terminal_stop_request_id = ''
    node._terminal_stop_ack_pub = Publisher()
    request = TerminalStopRequest('request-1', 'run-1', 'goal_reached', 10)

    node._terminal_stop_request_cb(
        SimpleNamespace(data=terminal_stop_request_to_json(request))
    )

    assert node._terminal_stop_requested
    assert node._active_controls is None
    assert node.cmd_pub.messages[-1].linear.x == 0.0
    assert node.cmd_pub.messages[-1].angular.z == 0.0
    ack = terminal_stop_ack_from_json(node._terminal_stop_ack_pub.messages[-1].data)
    assert ack.request_id == request.request_id
    assert ack.component == 'planner'
    assert ack.status == 'stop_command_published'


@pytest.mark.parametrize('target', [(0., 1.), (0., -1.)])
def test_simple_tracker_uses_the_declared_actuator_angular_limits(target):
    node = SimpleNamespace(local_horizon=12, dt=.25, v_max=.22,
                           w_min=-1., w_max=1., simple_tracker_yaw_gate_rad=.6)
    controls = EfeAgentNode._simple_local_plan(node, np.zeros(3), np.asarray(target))
    assert np.max(controls[:, 1]) <= 1.
    assert np.min(controls[:, 1]) >= -1.
