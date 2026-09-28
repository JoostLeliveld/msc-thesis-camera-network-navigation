"""Independent timing contracts for the remaining runtime repairs.

Real node methods, controlled ROS clocks, and event/lock barriers; no ROS graph,
sleep, live simulator, or retained experimental data. These assert the required
behavior, unlike the historical audit probes that assert defective baselines.
"""
from __future__ import annotations

from collections import deque
import json
import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.clock import ClockType
from rclpy.time import Time

from reliability.nodes.camera_manager_node import CameraManagerNode
from reliability.common_time import MotionPose
from test_planner_node_correction_wiring import _Clock, _Logger, _Publisher, stamp
from test_planner_node_state_correction import make_state_node, state_msg
from test_runtime_transactions import Publisher


def _linear_prediction(m, P, u, dt):
    """A checkable dynamics oracle for snapshot ownership, not filter calibration."""
    mean, covariance = m.copy(), P.copy()
    mean[0] += float(u[0]) * dt
    mean[2] += float(u[1]) * dt
    covariance += np.eye(3) * .001 * dt
    return mean, covariance


def _node(*, anchor=9.9, now=10., initialized=True):
    n = make_state_node(belief_stamp_s=anchor, now_s=now)
    n.heading_update_mode = 'coupled'
    n.require_state_correction_envelope = True
    n.state_reanchor_m = 0.
    n.pixel_timeout_s = .5
    n._seen_state_source_batch_ids = set()
    n._seen_map_observation_stamps = {}
    n.use_diagnostic_odom_localization = False
    n.use_odom_for_predict = True
    n.odom_vel = np.array([.2, .0])
    n.planner_belief_pub = Publisher()
    n.belief_state_pub = Publisher()
    n._CMD_LOG_MAX_S = 60.
    n.stale_belief_inflate_m2_per_s = 0.
    n.stale_belief_inflate_cap_m2 = 0.
    n._bev_affine = None
    n.bev_y_calibration_offset_m = 0.
    n._latest_detection_diag = None
    n._goal_received_logged = False
    n._goal_signature = None
    n._goal_progress_start_dist_m = None
    # Seed support including an actual preceding sample; timestamp/order tests
    # must not silently depend on an unsupported-motion fallback policy.
    n._odom_log = [(anchor - .1, .2, 0.), (anchor, .2, 0.)]
    n._odom_heading_log = [(round((anchor - .1)*1e9), 0.), (round(anchor*1e9), 0.)]
    n.planner.predict = _linear_prediction
    n._timing_fatal = []

    def fatal(reason, exc=None):
        n._timing_fatal.append((reason, exc))
        n._fatal_stop_triggered = True
        raise RuntimeError(reason) from exc

    n._fatal_experiment_stop = fatal
    if not initialized:
        n.belief_m = n.belief_S = n.belief_stamp = None
    elif hasattr(n, '_commit_belief'):
        # Install the fixture posterior through the same owner as production;
        # later tests must not rely on mutating authoritative-state mirrors.
        n._commit_belief(n.belief_m, n.belief_S, n.belief_stamp)
    return n


def _envelope(*, batch='timing:batch', capture=9.95):
    return SimpleNamespace(data=json.dumps(dict(
        schema_version=1, frame_id='map_bev', source_batch_id=batch,
        correction_stamp=capture, xy=[.1, 0.], covariance_m2=[[.03, 0.], [0., .03]],
    )))


def _wait(event):
    assert event.wait(5.), 'the deterministic synchronization barrier was not reached'


def _spawn(fn, *, name='timing-worker'):
    errors = []

    def run():
        try:
            fn()
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run, name=name, daemon=True)
    worker.start()
    return worker, errors


def _join(worker):
    thread, errors = worker
    thread.join(5.)
    assert not thread.is_alive(), 'callback did not leave its synchronization barrier'
    if errors:
        raise errors[0]


def _allow_integrity_stop(n, callback):
    try:
        return callback()
    except RuntimeError:
        assert n._timing_fatal, 'unexpected runtime failure was not an integrity stop'
        return None


def test_manager_snapshot_never_exposes_future_odometry_but_releases_it_later():
    manager = object.__new__(CameraManagerNode)
    manager._input_lock = threading.RLock()
    manager._manager_epoch = 'manager-test'
    manager.frame_id = 'map_bev'
    manager.odometry_frame_id = 'odom'
    manager.odometry_to_map_yaw_rad = 0.
    manager._odom_history = deque([
        MotionPose(9_900_000_000, (0., 0.), 0., 'odom', 'manager-test')
    ], maxlen=600)
    future = MotionPose(10_200_000_000, (.06, 0.), 0., 'odom', 'manager-test')
    manager._pending_odom_history = {future.stamp_ns: future}
    manager._pending_odom_capacity = 4096
    manager._clock = _Clock(10.)
    manager.get_clock = lambda: manager._clock

    with manager._input_lock:
        before = CameraManagerNode._motion_snapshot(manager)
    assert tuple(p.stamp_ns for p in before.samples) == (9_900_000_000,)
    assert tuple(manager._pending_odom_history) == (10_200_000_000,)

    manager._clock.seconds = 10.2
    with manager._input_lock:
        after = CameraManagerNode._motion_snapshot(manager)
    assert tuple(p.stamp_ns for p in after.samples) == (
        9_900_000_000, 10_200_000_000)
    assert not manager._pending_odom_history


def test_manager_shutdown_record_does_not_consult_invalid_ros_clock(tmp_path):
    from unav_common.camera_outcomes import OutcomeJournal, read_journal

    manager = object.__new__(CameraManagerNode)
    manager._outcome_journal = OutcomeJournal(tmp_path / 'manager.jsonl', 'manager-test')
    manager._batch_clock_high_water_s = 14.25
    manager.get_clock = lambda: pytest.fail('shutdown must not consult the ROS clock')
    CameraManagerNode._append_session_stopped(manager)
    manager._outcome_journal.close()

    stopped = list(read_journal(tmp_path / 'manager.jsonl'))[-1]
    assert stopped['status'] == 'session_stopped'
    assert stopped['stage'] == 'manager'
    assert stopped['publish_stamp_s'] == 14.25


class _ObservedRLock:
    """Signal contention, avoiding a sleep to infer that the second callback ran."""

    def __init__(self, progressed):
        self.lock = threading.RLock()
        self.progressed = progressed

    def __enter__(self):
        if not self.lock.acquire(blocking=False):
            self.progressed.set()
            self.lock.acquire()
        return self

    def __exit__(self, *_):
        self.lock.release()


def test_a_slow_public_prediction_cannot_replace_a_newer_target():
    n = _node()
    entered, release = threading.Event(), threading.Event()
    predict = n.planner.predict
    paused = False

    def delayed(m, P, u, dt):
        nonlocal paused
        if threading.current_thread().name == 'old-publication' and not paused:
            paused = True
            entered.set()
            _wait(release)
        return predict(m, P, u, dt)

    n.planner.predict = delayed
    first = _spawn(n._belief_publish_tick, name='old-publication')
    try:
        _wait(entered)
        n._clock.seconds = 10.2
        n._belief_publish_tick()
    finally:
        release.set()
    _join(first)
    targets = [n._stamp_to_float(m.header.stamp) for m in n.planner_belief_pub.messages]
    assert targets and targets[-1] == pytest.approx(10.2)
    assert targets == sorted(targets), 'an older prediction replaced newer public state'


def test_rewind_invalidates_a_prediction_already_in_flight():
    n = _node(anchor=100., now=100.1)
    entered, release = threading.Event(), threading.Event()
    predict = n.planner.predict

    def delayed(m, P, u, dt):
        entered.set()
        _wait(release)
        return predict(m, P, u, dt)

    n.planner.predict = delayed
    worker = _spawn(lambda: _allow_integrity_stop(n, n._belief_publish_tick))
    try:
        _wait(entered)
        n._clock.seconds = 1.
    finally:
        release.set()
    _join(worker)
    assert not n.planner_belief_pub.messages, 'pre-reset work published in the new clock epoch'


def test_rewind_during_correction_prevents_a_pre_reset_posterior_commit():
    n = _node(anchor=100., now=100.1)
    original = n.belief_m.copy(), n.belief_S.copy(), n.belief_stamp
    entered, release = threading.Event(), threading.Event()
    predict = n.planner.predict

    def delayed(m, P, u, dt):
        entered.set()
        _wait(release)
        return predict(m, P, u, dt)

    n.planner.predict = delayed
    worker = _spawn(lambda: _allow_integrity_stop(
        n, lambda: n._state_correction_envelope_cb(_envelope(capture=100.05))))
    try:
        _wait(entered)
        n._clock.seconds = 1.
    finally:
        release.set()
    _join(worker)
    np.testing.assert_array_equal(n.belief_m, original[0])
    np.testing.assert_array_equal(n.belief_S, original[1])
    assert n.belief_stamp == original[2], 'correction committed after its clock epoch ended'
    rows = [json.loads(value) for value in n.correction_assimilation_pub.published]
    assert not any(row['status'] == 'accepted' for row in rows)


def test_same_anchor_correction_supersedes_an_inflight_projection():
    n = _node()
    entered, release = threading.Event(), threading.Event()
    predict = n.planner.predict
    paused = False

    def delayed(m, P, u, dt):
        nonlocal paused
        if threading.current_thread().name == 'old-revision' and not paused:
            paused = True
            entered.set()
            _wait(release)
        return predict(m, P, u, dt)

    n.planner.predict = delayed
    first = _spawn(n._belief_publish_tick, name='old-revision')
    try:
        _wait(entered)
        n._apply_metric_correction(stamp(9.9), np.array([.1, 0.]), np.eye(2)*.03,
                                   allow_same_stamp=True)
        assert n._stamp_to_float(n.belief_stamp) == pytest.approx(9.9)
        n._belief_publish_tick()
        latest, = n.planner_belief_pub.messages
    finally:
        release.set()
    _join(first)
    assert n.planner_belief_pub.messages == [latest], 'same-time correction did not supersede old work'


def test_future_anchor_never_becomes_a_fresh_planning_belief():
    n = _node(anchor=100., now=100.1)
    n._clock.seconds = 1.
    result = _allow_integrity_stop(n, n._resolve_belief_for_planning)
    if result is not None:
        mean, covariance, _ = result
        assert mean is None and covariance is None, 'future anchor was returned as age-zero state'


def test_clock_domain_conversion_failure_is_not_fresh():
    n = _node()
    n.get_clock = lambda: SimpleNamespace(now=lambda: Time(
        seconds=1000., clock_type=ClockType.SYSTEM_TIME))
    try:
        age = n._stamp_age_s(stamp(1.))
    except (TypeError, ValueError):
        return  # Explicit refusal also satisfies the boundary.
    assert not n._metric_correction_is_fresh(age), 'failed time conversion was treated as age zero'


def test_command_receipt_and_held_value_have_one_commit_order():
    n = _node()
    entered, release, progressed = threading.Event(), threading.Event(), threading.Event()
    n._data_lock = _ObservedRLock(progressed)

    class Clock:
        calls = 0

        def now(self):
            self.calls += 1
            first = self.calls == 1
            value = _Clock(10. if first else 11.).now()
            if first:
                entered.set()
                _wait(release)
            return value

    clock = Clock()
    n.get_clock = lambda: clock
    a, b = Twist(), Twist()
    a.linear.x, b.linear.x = .1, .2
    first = _spawn(lambda: n._cmd_cb(a))

    def second_callback():
        n._cmd_cb(b)
        progressed.set()

    second = None
    try:
        _wait(entered)
        second = _spawn(second_callback)
        # The second call either commits or is blocked by a repaired transaction.
        _wait(progressed)
    finally:
        release.set()
    _join(first)
    if second is not None:
        _join(second)
    times = [event[0] for event in n._cmd_log]
    assert times == sorted(times), 'command history is callback completion order'
    assert n.last_cmd[0] == pytest.approx(.2), 'old receipt overwrote the newer held command'


@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf])
def test_nonfinite_command_cannot_mutate_motion_inputs(value):
    n = _node()
    before, history = n.last_cmd.copy(), list(n._cmd_log)
    message = Twist()
    message.linear.x = value
    _allow_integrity_stop(n, lambda: n._cmd_cb(message))
    np.testing.assert_array_equal(n.last_cmd, before)
    assert n._cmd_log == history


def test_goal_and_progress_signature_are_one_event():
    n = _node()
    entered, release, progressed = threading.Event(), threading.Event(), threading.Event()
    n._data_lock = _ObservedRLock(progressed)
    first_goal = PoseStamped()
    first_goal.header.frame_id = 'map_bev'
    first_goal.pose.position.x = 1.

    class DelayedGoal:
        header = first_goal.header
        reads = 0

        @property
        def pose(self):
            self.reads += 1
            if self.reads == 1:
                entered.set()
                _wait(release)
            return first_goal.pose

    second_goal = PoseStamped()
    second_goal.header.frame_id = 'map_bev'
    second_goal.pose.position.x = 2.
    first = _spawn(lambda: n._goal_cb(DelayedGoal()))

    def second_callback():
        n._goal_cb(second_goal)
        progressed.set()

    second = None
    try:
        _wait(entered)
        second = _spawn(second_callback)
        _wait(progressed)
    finally:
        release.set()
    _join(first)
    if second is not None:
        _join(second)
    signature = (n.goal_msg.header.frame_id, n.goal_msg.pose.position.x, n.goal_msg.pose.position.y)
    assert n._goal_signature == signature, 'goal and progress state came from different callbacks'


def test_full_published_state_and_covariance_share_the_declared_target():
    n = _node()
    initial = np.array([1., 2., .7])
    covariance = np.array([[.4, .03, .02], [.03, .3, -.01], [.02, -.01, .2]])
    if hasattr(n, '_commit_belief'):
        n._commit_belief(initial, covariance, stamp(9.9))
    else:
        n.belief_m, n.belief_S = initial.copy(), covariance.copy()
    n._odom_log = [(9.8, .2, .3), (9.9, .2, .3)]
    n._belief_publish_tick()
    message, = n.planner_belief_pub.messages
    target = n._stamp_to_float(message.header.stamp)
    age = target - 9.9
    assert target == pytest.approx(10.)
    assert message.header.frame_id == 'map_bev'
    assert message.pose.pose.position.x == pytest.approx(initial[0] + .2 * age)
    assert message.pose.pose.position.y == initial[1]
    yaw = 2. * math.atan2(message.pose.pose.orientation.z, message.pose.pose.orientation.w)
    assert yaw == pytest.approx(initial[2] + .3 * age)
    matrix = np.asarray(message.pose.covariance).reshape(6, 6)
    np.testing.assert_allclose(matrix[np.ix_([0, 1, 5], [0, 1, 5])], covariance + np.eye(3)*.001*age)
    np.testing.assert_array_equal(n.belief_m, initial)
    np.testing.assert_array_equal(n.belief_S, covariance)
    assert n._stamp_to_float(n.belief_stamp) == pytest.approx(9.9)


def test_camera_xy_only_publishes_the_same_odom_heading_used_for_planning():
    """The manager's hull prior and the controller must see one heading state."""
    n = _node()
    n.heading_update_mode = 'camera_xy_only'
    n.use_pixel_correction = False
    n.odom_yaw_offset_rad = 0.0
    n.process_noise_theta = .02
    n._odom_origin_stamp_s = 0.0
    n._odom_log = [(9.8, .2, .3), (9.9, .2, .3)]
    n._odom_heading_log = [(9_800_000_000, 1.17), (9_900_000_000, 1.20)]

    planned_m, planned_P, meta = n._resolve_belief_for_planning()
    assert meta['heading_anchor_applied']
    n._belief_publish_tick()

    event = _last_belief_event(n)
    pose = n.planner_belief_pub.messages[-1]
    _assert_event_and_pose_agree(event, pose)
    assert event['mean'][2] == pytest.approx(1.23)
    assert event['mean'][2] == pytest.approx(planned_m[2])
    np.testing.assert_allclose(event['covariance'], planned_P)
    np.testing.assert_allclose(np.asarray(event['covariance'])[:2, 2], 0.0)
    np.testing.assert_allclose(np.asarray(event['covariance'])[2, :2], 0.0)


def test_compatibility_message_cannot_relabel_committed_belief():
    n = _node()
    pose = state_msg(0., 0., seconds=9.9)
    pose.header.frame_id = 'map_bev'
    n._state_cb(pose)
    predict = n.planner.predict
    injected = False

    def wrong_frame_arrives(m, P, u, dt):
        nonlocal injected
        if not injected:
            injected = True
            wrong = state_msg(50., 50., seconds=9.95)
            wrong.header.frame_id = 'camera_optical'
            _allow_integrity_stop(n, lambda: n._state_cb(wrong))
        return predict(m, P, u, dt)

    n.planner.predict = wrong_frame_arrives
    _allow_integrity_stop(n, n._belief_publish_tick)
    assert all(msg.header.frame_id == 'map_bev' for msg in n.planner_belief_pub.messages)


def test_correction_during_planning_cannot_change_only_input_metadata():
    n = _node()
    entered, release = threading.Event(), threading.Event()
    predict = n.planner.predict
    paused = False
    returned = []

    def delayed(m, P, u, dt):
        nonlocal paused
        if threading.current_thread().name == 'planning-snapshot' and not paused:
            paused = True
            entered.set()
            _wait(release)
        return predict(m, P, u, dt)

    n.planner.predict = delayed
    worker = _spawn(lambda: returned.append(n._resolve_belief_for_planning()), name='planning-snapshot')
    try:
        _wait(entered)
        n._state_correction_envelope_cb(_envelope())
        corrected_x = float(n.belief_m[0])
    finally:
        release.set()
    _join(worker)
    mean, _, meta = returned[0]
    assert mean is not None, 'routine camera update discarded planning without a replacement snapshot'
    anchor = n._stamp_to_float(meta['belief_stamp'])
    if math.isclose(float(mean[0]), .02):
        assert anchor == pytest.approx(9.9), 'old prediction was relabelled with the newer anchor'
    else:
        # Reprojection is also valid, provided metadata and state agree.
        assert anchor == pytest.approx(9.95)
        assert mean[0] == pytest.approx(corrected_x + .2*.05)


def test_diagnostic_failure_cannot_forget_a_committed_batch_identity():
    n = _node()

    def fail(_message):
        raise RuntimeError('injected diagnostic publication failure')

    n.pixel_correction_diag_pub = SimpleNamespace(publish=fail)
    _allow_integrity_stop(n, lambda: n._state_correction_envelope_cb(_envelope(batch='committed')))
    assert n.belief_m[0] > .02, 'test did not reach a posterior-changing commit'
    assert 'committed' in n._seen_state_source_batch_ids, 'posterior committed without its identity'
    retained = n._correction_outcomes['committed']
    assert retained['source_batch_id'] == 'committed'
    assert retained['status'] == 'accepted', 'retained terminal outcome does not describe the commit'
    assert retained['schema_version'] == 2
    assert retained['frame_id'] == 'map_bev'
    assert retained['state_stamp_ns'] == 9_950_000_000
    assert retained['apply_stamp_ns'] == 10_000_000_000
    np.testing.assert_array_equal(retained['posterior_mean'], n.belief_m)
    np.testing.assert_array_equal(retained['posterior_covariance'], n.belief_S)
    before = (n.belief_m.copy(), n.belief_S.copy(), n.belief_stamp)
    n.pixel_correction_diag_pub = _Publisher()
    _allow_integrity_stop(n, lambda: n._state_correction_envelope_cb(_envelope(batch='committed')))
    np.testing.assert_array_equal(n.belief_m, before[0])
    np.testing.assert_array_equal(n.belief_S, before[1])
    assert n.belief_stamp == before[2]
    assert n._correction_outcomes['committed'] == retained
    rows = [json.loads(value) for value in n.correction_assimilation_pub.published]
    assert not any(row['reason'] == 'not_newer_than_belief' for row in rows), \
        'redelivery reclassified a committed correction as never assimilated'


def test_terminal_outcome_retains_commit_time_through_slow_diagnostics():
    n = _node()

    def slow_diagnostic(_message):
        n._clock.seconds = 10.2

    n.pixel_correction_diag_pub = SimpleNamespace(publish=slow_diagnostic)
    n._state_correction_envelope_cb(_envelope())
    row, = [json.loads(value) for value in n.correction_assimilation_pub.published]
    assert n._clock.seconds == 10.2
    assert row['apply_stamp_ns'] == 10_000_000_000
    assert row['apply_stamp'] == pytest.approx(10.)
    assert row['correction_stamp_ns'] == row['state_stamp_ns'] == 9_950_000_000
    np.testing.assert_array_equal(row['posterior_mean'], n.belief_m)
    np.testing.assert_array_equal(row['posterior_covariance'], n.belief_S)


def test_simultaneous_duplicate_envelopes_have_one_terminal_commit():
    n = _node()
    entered, release, contended = threading.Event(), threading.Event(), threading.Event()
    n._correction_lock = _ObservedRLock(contended)
    predict = n.planner.predict

    def delayed(m, P, u, dt):
        entered.set()
        _wait(release)
        return predict(m, P, u, dt)

    n.planner.predict = delayed
    first = _spawn(lambda: n._state_correction_envelope_cb(_envelope()))
    second = None
    try:
        _wait(entered)
        second = _spawn(lambda: _allow_integrity_stop(
            n, lambda: n._state_correction_envelope_cb(_envelope())))
        _wait(contended)
    finally:
        release.set()
    _join(first)
    if second is not None:
        _join(second)
    rows = [json.loads(value) for value in n.correction_assimilation_pub.published]
    assert len(rows) == 1 and rows[0]['status'] == 'accepted'
    assert rows[0]['source_batch_id'] == 'timing:batch'
    assert len(n._timing_fatal) == 1


def test_same_state_time_does_not_make_a_new_belief_a_duplicate():
    n = _node()
    n._belief_publish_tick()
    n._state_correction_envelope_cb(_envelope())
    n._belief_publish_tick()
    first, second = n.planner_belief_pub.messages
    assert first.header.stamp == second.header.stamp
    assert second.pose.pose.position.x > first.pose.pose.position.x
    manager = SimpleNamespace(frame_id='map_bev', _belief_query_history=deque(),
                              get_logger=lambda: _Logger())
    CameraManagerNode._belief_query_callback(manager, first)
    CameraManagerNode._belief_query_callback(manager, second)
    assert manager._belief_query_history[-1][1][0] == pytest.approx(second.pose.pose.position.x), \
        'equal timestamp hid a newer posterior from the belief consumer'


def _last_belief_event(n):
    assert n.belief_state_pub.messages, 'the explicit belief-state topic published no event'
    return json.loads(n.belief_state_pub.messages[-1].data)


def _assert_event_and_pose_agree(event, pose):
    assert event['schema_version'] == 1
    assert event['valid'] and event['motion_supported']
    assert isinstance(event['epoch'], str) and event['epoch']
    assert isinstance(event['revision'], int) and event['revision'] >= 0
    assert event['frame_id'] == pose.header.frame_id == 'map_bev'
    assert event['state_stamp_ns'] == pose.header.stamp.sec*1_000_000_000 + pose.header.stamp.nanosec
    yaw = 2.*math.atan2(pose.pose.pose.orientation.z, pose.pose.pose.orientation.w)
    np.testing.assert_allclose(event['mean'], [pose.pose.pose.position.x, pose.pose.pose.position.y, yaw])
    covariance = np.asarray(pose.pose.covariance).reshape(6, 6)
    np.testing.assert_allclose(event['covariance'], covariance[np.ix_([0, 1, 5], [0, 1, 5])])


def test_explicit_belief_event_distinguishes_revisions_at_the_same_target():
    n = _node()
    n._belief_publish_tick()
    first = _last_belief_event(n)
    _assert_event_and_pose_agree(first, n.planner_belief_pub.messages[-1])
    n._state_correction_envelope_cb(_envelope())
    n._belief_publish_tick()
    second = _last_belief_event(n)
    _assert_event_and_pose_agree(second, n.planner_belief_pub.messages[-1])
    assert first['state_stamp_ns'] == second['state_stamp_ns'] == 10_000_000_000
    assert first['epoch'] == second['epoch']
    assert first['revision'] < second['revision']
    assert first['anchor_stamp_ns'] == 9_900_000_000
    assert second['anchor_stamp_ns'] == 9_950_000_000


def test_distinct_correction_at_the_same_anchor_increments_revision():
    n = _node(now=9.9)
    n._belief_publish_tick()
    first = _last_belief_event(n)
    n._apply_metric_correction(stamp(9.9), np.array([.1, 0.]), np.eye(2)*.03,
                               allow_same_stamp=True, source_batch_id='same-anchor-camera')
    n._belief_publish_tick()
    second = _last_belief_event(n)
    assert second['epoch'] == first['epoch']
    assert second['anchor_stamp_ns'] == first['anchor_stamp_ns'] == 9_900_000_000
    assert second['state_stamp_ns'] == first['state_stamp_ns']
    assert second['revision'] > first['revision']
    assert second['mean'][0] > first['mean'][0]
    assert second['covariance'][0][0] < first['covariance'][0][0]


def test_rewind_revokes_the_previously_published_valid_status():
    n = _node(anchor=100., now=100.1)
    n._belief_publish_tick()
    before = _last_belief_event(n)
    assert before['valid']
    count = len(n.planner_belief_pub.messages)
    n._clock.seconds = 1.
    _allow_integrity_stop(n, n._belief_publish_tick)
    after = _last_belief_event(n)
    assert not after['valid'], 'retained status still declares a pre-reset belief valid'
    assert after['invalid_reason']
    assert len(n.planner_belief_pub.messages) == count


def test_delayed_uninitialized_status_cannot_revoke_a_new_bootstrap():
    n = _node(initialized=False)
    entered, release = threading.Event(), threading.Event()
    publish_invalid = n._publish_invalid_belief

    def delayed(reason, *args, **kwargs):
        entered.set()
        _wait(release)
        return publish_invalid(reason, *args, **kwargs)

    n._publish_invalid_belief = delayed
    worker = _spawn(n._belief_publish_tick)
    try:
        _wait(entered)
        n._state_correction_envelope_cb(_envelope(batch='new-bootstrap'))
        assert n.belief_m is not None
    finally:
        release.set()
    _join(worker)
    events = [json.loads(message.data) for message in n.belief_state_pub.messages]
    assert not any(event.get('invalid_reason') == 'uninitialized' for event in events), \
        'stale uninitialized work published after the new belief committed'
    n._publish_invalid_belief = publish_invalid
    n._belief_publish_tick()
    assert _last_belief_event(n)['valid']


def test_canonical_manager_consumer_keeps_revision_and_invalidity():
    from reliability.manager_state import AdmissionBeliefHistory

    n = _node()
    manager = SimpleNamespace(frame_id='map_bev', _input_lock=threading.RLock(),
        _admission_beliefs=AdmissionBeliefHistory('map_bev'),
        _belief_query_history=deque(maxlen=400), _canonical_belief_seen=False,
        _has_operational_anchor=False, get_clock=lambda: n._clock,
        get_logger=lambda: _Logger())
    n._belief_publish_tick()
    first_event = n.belief_state_pub.messages[-1]
    first_pose = n.planner_belief_pub.messages[-1]
    CameraManagerNode._belief_state_callback(manager, first_event)

    n._state_correction_envelope_cb(_envelope())
    n._belief_publish_tick()
    second_event = n.belief_state_pub.messages[-1]
    second_pose = n.planner_belief_pub.messages[-1]
    CameraManagerNode._belief_state_callback(manager, second_event)
    assert manager._admission_beliefs.latest.revision == _last_belief_event(n)['revision']
    assert manager._belief_query_history[-1][1][0] == pytest.approx(second_pose.pose.pose.position.x)
    # Late canonical and compatibility deliveries cannot restore the old revision.
    CameraManagerNode._belief_state_callback(manager, first_event)
    CameraManagerNode._belief_query_callback(manager, first_pose)
    assert manager._belief_query_history[-1][1][0] == pytest.approx(second_pose.pose.pose.position.x)

    # A forward clock jump beyond the held odometry support produces an explicit
    # invalid event. It must revoke the prior without publishing a false Pose.
    n._clock.seconds = 12.
    pose_count = len(n.planner_belief_pub.messages)
    n._belief_publish_tick()
    assert not _last_belief_event(n)['valid']
    assert len(n.planner_belief_pub.messages) == pose_count
    CameraManagerNode._belief_state_callback(manager, n.belief_state_pub.messages[-1])
    assert not manager._admission_beliefs.latest.valid
    assert not manager._belief_query_history
    CameraManagerNode._belief_state_callback(manager, second_event)
    CameraManagerNode._belief_query_callback(manager, second_pose)
    assert not manager._admission_beliefs.latest.valid
    assert not manager._belief_query_history
