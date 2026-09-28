"""State-owner regressions using runtime callbacks and independent algebra.

No ROS graph or simulator. The main trace uses the real planner prediction and
node correction; a matrix-exponential continuous SDE supplies the Q oracle.
"""
import json
import math

import numpy as np
import pytest
from scipy.linalg import expm

from planning.planners.base_planner import UnicyclePlannerBase
from planning.core.motion_history import MotionHistorySnapshot, plan_replay
from test_planner_node_correction_wiring import make_node, stamp
from test_planner_node_state_correction import state_msg
from test_outage_motion_replay import _odom_msg
from test_timing_publication_invariants import _node


def _real_dynamics(n):
    planner = object.__new__(UnicyclePlannerBase)
    planner.dt = .25
    planner.process_noise_xy = .01
    planner.process_noise_theta = .02
    planner.coherent_drift = False
    n.planner = planner
    return n


def _oracle_step(m, P, v, w, dt):
    theta = m[2]
    c, s = math.cos(theta), math.sin(theta)
    A = np.zeros((3,3)); A[0,2], A[1,2] = -v*s, v*c
    G = np.array([[c, 0.], [s, 0.], [0., 1.]])
    W = G @ np.diag([.01**2, .02**2]) @ G.T
    M = expm(np.block([[A, W], [np.zeros((3,3)), -A.T]])*dt)
    F, Q = M[:3,:3], M[:3,3:] @ M[:3,:3].T
    result = m + np.array([v*c*dt, v*s*dt, w*dt])
    result[2] = math.atan2(math.sin(result[2]), math.cos(result[2]))
    return result, F @ P @ F.T + Q


def test_complete_runtime_trace_matches_independent_full_covariance_oracle():
    n = _real_dynamics(_node(anchor=9.4, now=10.))
    m = np.array([1., -2., .4])
    P = np.array([[.04,.005,.002],[.005,.06,-.003],[.002,-.003,.025]])
    n._commit_belief(m, P, stamp(9.4))
    revision_before = n._belief_record.revision
    n._odom_log = []; n._odom_heading_log = []
    for t,v,w,yaw in [(9.35,.2,0.,.4),(9.6,0.,.8,.4),(9.75,.15,0.,.52),(10.,0.,0.,.52)]:
        n._odom_cb(_odom_msg(t,v,w,yaw))
    prior, prior_P = m.copy(), P.copy()
    for v,w,dt in [(.2,0.,.2),(0.,.8,.15),(.15,0.,.25)]:
        prior, prior_P = _oracle_step(prior, prior_P, v,w,dt)
    R = np.array([[.015,.003],[.003,.025]])
    innovation = np.array([.02,-.01]); z = prior[:2] + innovation
    H = np.array([[1.,0.,0.],[0.,1.,0.]])
    gain = np.linalg.solve(H @ prior_P @ H.T + R, H @ prior_P).T
    posterior = prior + gain @ innovation
    J = np.eye(3)-gain @ H
    posterior_P = J @ prior_P @ J.T + gain @ R @ gain.T
    n._apply_metric_correction(stamp(10.), z, R, source_batch_id='trace:physical-event')
    terminal = json.loads(n.correction_assimilation_pub.published[-1])
    assert terminal['status'] == 'accepted'
    np.testing.assert_allclose(terminal['prior_mean'], prior, atol=1e-13)
    np.testing.assert_allclose(terminal['prior_covariance'], prior_P, atol=1e-13)
    np.testing.assert_allclose(terminal['posterior_mean'], posterior, atol=1e-13)
    np.testing.assert_allclose(terminal['posterior_covariance'], posterior_P, atol=1e-13)
    np.testing.assert_allclose(n.belief_m, posterior, atol=1e-13)
    np.testing.assert_allclose(n.belief_S, posterior_P, atol=1e-13)
    assert terminal['source_batch_id'] == 'trace:physical-event'
    assert terminal['state_stamp_ns'] == terminal['correction_stamp_ns'] == 10_000_000_000
    assert terminal['frame_id'] == 'map_bev' and terminal['motion_supported']
    assert terminal['revision_before'] == revision_before
    assert terminal['revision_after'] > revision_before
    n._belief_publish_tick()
    event = json.loads(n.belief_state_pub.messages[-1].data)
    np.testing.assert_allclose(event['mean'], posterior)
    np.testing.assert_allclose(event['covariance'], posterior_P)
    assert event['state_stamp_ns'] == terminal['state_stamp_ns']
    assert event['revision'] == terminal['revision_after']


def test_delayed_bootstrap_during_turn_uses_capture_heading_once():
    n = _real_dynamics(_node(anchor=9.4, now=10., initialized=False))
    n.odom_yaw_offset_rad = .3
    n._odom_log = []; n._odom_heading_log = []
    n._odom_cb(_odom_msg(9.8,0.,1.,.8))
    n._odom_cb(_odom_msg(10.,0.,1.,1.))
    n._apply_metric_correction(stamp(9.8), np.zeros(2), np.eye(2)*.03,
                               source_batch_id='delayed-bootstrap')
    assert n.belief_m[2] == pytest.approx(1.1)
    n._belief_publish_tick()
    event = json.loads(n.belief_state_pub.messages[-1].data)
    assert event['mean'][2] == pytest.approx(1.3)
    assert event['anchor_stamp_ns'] == 9_800_000_000
    assert event['state_stamp_ns'] == 10_000_000_000


@pytest.mark.parametrize('history', [[], [(9.95,.2,1.)], [(8.,.2,1.)]])
def test_unsupported_motion_has_one_mean_assumption_in_correction_and_publication(history):
    n = _real_dynamics(_node())
    n._odom_log, n._cmd_log = history, []
    n._belief_publish_tick()
    prediction = json.loads(n.belief_state_pub.messages[-1].data)
    assert not prediction['valid'] and not prediction['motion_supported']
    n._apply_metric_correction(stamp(10.), np.array(prediction['mean'][:2]), np.eye(2)*.03,
                               source_batch_id='unsupported')
    terminal = json.loads(n.correction_assimilation_pub.published[-1])
    np.testing.assert_allclose(terminal['prior_mean'], prediction['mean'])
    np.testing.assert_allclose(terminal['prior_covariance'], prediction['covariance'])
    assert not terminal['motion_supported']
    assert n._resolve_belief_for_planning()[0] is None


def test_outcome_and_snapshot_arrays_do_not_alias_committed_state():
    n = _node()
    outcome = n._apply_metric_correction(stamp(9.95), np.array([.1,0.]), np.eye(2)*.03,
                                         source_batch_id='immutable')
    before = n._belief_record
    outcome.next_m[:] = 40.; outcome.next_S[:] = 50.
    np.testing.assert_allclose(n.belief_m, before.mean)
    np.testing.assert_allclose(n.belief_S, before.covariance)
    with pytest.raises(TypeError):
        n._correction_outcomes['immutable']['posterior_mean'][0] = 99.
    with pytest.raises(ValueError):
        n.belief_m[0] = 99.


@pytest.mark.parametrize('failure', ['nan', 'indefinite', 'shape'])
def test_invalid_held_prediction_cannot_replace_last_valid_anchor(failure):
    n = _node()
    before = n._belief_record
    def invalid(m,P,u,dt):
        if failure == 'nan': m[0] = math.nan
        elif failure == 'indefinite': P[0,0] = -1.
        else: m = m[:2]
        return m,P
    n.planner.predict = invalid
    with pytest.raises(RuntimeError):
        n._apply_metric_correction(stamp(9.95), np.array([.1,0.]), np.eye(2)*.03,
                                   source_batch_id='invalid-held')
    assert n._belief_record is before
    assert n._correction_outcomes['invalid-held']['reason'] == 'invalid_prediction'
    assert n._correction_outcomes['invalid-held']['posterior_mean'] == before.mean


def test_pixel_event_identity_prevents_repeat_information_and_backdating():
    n = make_node(meas=(.1,0.))
    n._apply_pixel_correction(stamp(9.95))
    record = n._belief_record
    n._apply_pixel_correction(stamp(9.95))
    n._apply_pixel_correction(stamp(9.94))
    assert n._belief_record is record
    assert len(n.correction_assimilation_pub.published) == 1


def test_legacy_state_is_committed_once_on_arrival_and_never_by_planning():
    n = _node()
    n.state_correction_ekf = False
    pose = state_msg(.1,.2,seconds=9.95); pose.header.frame_id = 'map_bev'
    n._state_cb(pose)
    record = n._belief_record
    assert record.stamp_ns == 9_950_000_000
    for _ in range(3):
        n._resolve_belief_for_planning()
    assert n._belief_record is record
    n._state_cb(state_msg(20.,30.,seconds=9.94))
    assert n._belief_record is record


def test_one_nanosecond_motion_interval_is_integrated_exactly_once():
    snapshot = MotionHistorySnapshot.capture([(1., .2, .3)], [], True)
    plan = plan_replay(snapshot, 1_000_000_000, 1_000_000_001, 1.5)
    assert plan.segments == ((1_000_000_000,1_000_000_001,.2,.3),)
    assert plan.support.supported


def test_future_odometry_is_causally_buffered_then_committed_in_source_order():
    n = _node()
    n._odom_accepted_stamp_ns = 9_900_000_000
    history = list(n._odom_log)
    n._odom_cb(_odom_msg(11.,9.,8.,2.))
    assert n._odom_log == history
    assert n._odom_buffered_future == 1
    assert tuple(n._pending_odom_events) == (11_000_000_000,)
    # A second future sample may arrive earlier in source time. Neither sample
    # becomes current merely because it has been delivered.
    n._odom_cb(_odom_msg(10.8,.4,.3,.7))
    assert n._odom_log == history
    assert tuple(sorted(n._pending_odom_events)) == (
        10_800_000_000, 11_000_000_000)

    n._clock.seconds = 11.
    with n._data_lock:
        n._observe_belief_clock_locked()
        snapshot = n._motion_snapshot_locked()
    assert snapshot.odom[-2:] == ((10.8, .4, .3), (11., 9., 8.))
    assert snapshot.headings[-2:] == ((10_800_000_000, .7), (11_000_000_000, 2.))
    assert not n._pending_odom_events


@pytest.mark.parametrize('mutation', ['zero_quaternion','nonunit_quaternion','pose_frame','twist_frame'])
def test_semantically_invalid_odometry_never_mutates_history_or_heading(mutation):
    n = _node()
    msg = _odom_msg(9.95,.2,.3,.5)
    if mutation == 'zero_quaternion':
        msg.pose.pose.orientation.z = msg.pose.pose.orientation.w = 0.
    elif mutation == 'nonunit_quaternion':
        msg.pose.pose.orientation.w = 4.
    elif mutation == 'pose_frame':
        msg.header.frame_id = 'camera_optical'
    else:
        msg.child_frame_id = 'map_bev'
    history, headings = list(n._odom_log), list(n._odom_heading_log)
    n._odom_cb(msg)
    assert n._odom_log == history
    assert n._odom_heading_log == headings
    assert n._odom_refused_invalid == 1
