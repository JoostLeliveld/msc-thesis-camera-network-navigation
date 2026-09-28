"""Independent audit-06 invariants; no changes to declared recovery thresholds.

Uses analytic motion and fixed synthetic R to test wiring rather than camera quality.
Long camera outages are distinct from missing-motion gaps: complete odometry support
allows replay, while absent or stale support is still refused.
"""
import json
import math

import numpy as np
import pytest
from std_msgs.msg import String

from reliability.fusion import map_observations_to_json
from planning.core.motion_history import MotionHistorySnapshot, plan_replay
from test_planner_node_correction_wiring import stamp
from test_planner_node_per_camera_correction import observation
from test_planner_node_state_correction import make_state_node


class ExactMotion:
    """Independent closed-form constant-input unicycle, small positive Q."""
    @staticmethod
    def predict(m, P, cmd, dt):
        x, y, yaw = np.asarray(m, dtype=float)
        v, w = cmd
        if abs(w) < 1e-12:
            x += v * dt * math.cos(yaw)
            y += v * dt * math.sin(yaw)
        else:
            x += v / w * (math.sin(yaw + w * dt) - math.sin(yaw))
            y += v / w * (math.cos(yaw) - math.cos(yaw + w * dt))
        return np.array([x, y, math.atan2(math.sin(yaw+w*dt), math.cos(yaw+w*dt))]), np.asarray(P) + np.diag([1e-4, 1e-4, 4e-4])*dt


def node(**kwargs):
    n = make_state_node(pixel_timeout_s=.5, **kwargs)
    n.state_reanchor_m = 0.
    n.heading_update_mode = 'coupled'
    n.max_predict_speed_mps = .22
    n._seen_map_observation_stamps = {}
    n._seen_state_source_batch_ids = set()
    n._resolve_plan_frame_id = lambda: 'map_bev'
    return n


def apply(n, t, xy, batch):
    n._clock.seconds = t + .01
    n._apply_metric_correction(stamp(t), np.asarray(xy), np.eye(2)*.01,
                               source_batch_id=batch)
    return json.loads(n.correction_assimilation_pub.published[-1])


def test_normal_update_during_supported_turn_accepts_correct_measurement():
    n = node(belief_stamp_s=9.8, now_s=10.01)
    n.use_odom_for_predict = True
    n.planner = ExactMotion()
    n._odom_log = [(9.8, .2, 1.), (9.9, .2, 1.), (10., .2, 1.)]
    expected = [.2*math.sin(.2), .2*(1-math.cos(.2)), .2]
    row = apply(n, 10., expected[:2], 'turn')
    assert row['status'] == 'accepted'
    assert row['nis'] == pytest.approx(0., abs=1e-15)
    np.testing.assert_allclose(n.belief_m, expected, atol=1e-10)
    assert row['belief_stamp_after'] == 10.


def test_complete_blind_turn_with_dense_odometry_accepts_first_return():
    n = node(belief_stamp_s=0.)
    n.use_odom_for_predict = True
    n.planner = ExactMotion()
    n._odom_log = [(k/10, .2 if k < 20 or k >= 40 else 0.,
                    math.pi/4 if 20 <= k < 40 else 0.) for k in range(103)]
    first = apply(n, 10., [.4, 1.2], 'return')
    assert first['status'] == 'accepted'
    assert first['reason'] == 'accepted'
    np.testing.assert_allclose(n.belief_m, [.4, 1.2, math.pi/2], atol=1e-10)
    assert len(n.correction_assimilation_pub.published) == 1


def test_repeated_rejection_then_valid_evidence_recovers_without_snap():
    n = node(belief_cov=.01, belief_stamp_s=0.)
    for k in range(1, 5):
        row = apply(n, k*.2, [2., 0.], f'outlier-{k}')
        assert row['status'] == 'rejected'
        assert row['reason'] == 'nis_too_large'
        assert n.belief_m[0] == 0.
    row = apply(n, 1., [0., 0.], 'recovery')
    assert row['status'] == 'accepted'
    assert len(n.correction_assimilation_pub.published) == 5


def test_duplicate_rejected_direct_batch_adds_no_further_inflation():
    n = node(belief_cov=.01)
    batch = [observation('camera_A', 2., 0., seconds=9.95, var=.01)]
    n._apply_map_observations(batch)
    assert n.belief_m[0] == 0.
    before = n.belief_S.copy()
    before_stamp = n._stamp_to_float(n.belief_stamp)
    n._apply_map_observations(batch)
    np.testing.assert_array_equal(n.belief_S, before)
    assert n._stamp_to_float(n.belief_stamp) == before_stamp


def test_repeated_frames_from_one_camera_cannot_form_reanchor_quorum():
    n = node(belief_cov=.01)
    n.state_reanchor_m = 2.
    n._apply_map_observations([
        observation('camera_A', 3., 0., seconds=9.94, var=.01),
        observation('camera_A', 3., 0., seconds=9.95, var=.01)])
    assert n.belief_m[0] == pytest.approx(0.)


def test_stale_agreeing_cameras_do_not_reanchor_or_inflate():
    n = node(belief_stamp_s=10., now_s=12.)
    n.state_reanchor_m = 2.
    before_m, before_P = n.belief_m.copy(), n.belief_S.copy()
    n._apply_map_observations([
        observation('camera_A', 3., 0., seconds=9., var=.01),
        observation('camera_B', 3., 0., seconds=9., var=.01)])
    np.testing.assert_array_equal(n.belief_m, before_m)
    np.testing.assert_array_equal(n.belief_S, before_P)
    assert n._stamp_to_float(n.belief_stamp) == 10.


def test_direct_bootstrap_is_not_a_rejection_inflation_event():
    n = node()
    n.belief_m = n.belief_S = n.belief_stamp = None
    n._apply_map_observations([observation('camera_A', 1., 0., seconds=9.95, var=.01)])
    # Existing direct path allows one-camera initialization; no policy change here.
    np.testing.assert_allclose(n.belief_m[:2], [1., 0.])
    np.testing.assert_allclose(n.belief_S[:2, :2], np.eye(2)*.01, atol=1e-12)


def test_direct_gap_refusal_inflates_only_once_like_fused_event():
    direct = node(belief_stamp_s=0.)
    fused = node(belief_stamp_s=0.)
    for n in (direct, fused):
        n.use_odom_for_predict = True
        n._odom_log = [(k/10, 0., 0.) for k in range(101)]
    direct._apply_map_observations([observation('camera_A', 0., 0., seconds=9.95, var=.01)])
    fused._apply_metric_correction(stamp(9.95), np.zeros(2), np.eye(2)*.01)
    np.testing.assert_allclose(direct.belief_S, fused.belief_S, atol=1e-12)


@pytest.mark.parametrize('frame,covariance', [
    ('foreign', [[.01, 0.], [0., .01]]),
    ('map_bev', [[.01, 1.], [1., .01]]),
    ('map_bev', [[.01, .001], [0., .01]]),
    ('map_bev', [[float('nan'), 0.], [0., .01]])])
def test_malformed_direct_transport_never_mutates_filter_or_watermarks(frame, covariance):
    n = node()
    n.state_correction_mode = 'per_camera'
    failures = []
    n._fatal_experiment_stop = lambda *args: failures.append(args)
    raw = json.loads(map_observations_to_json([
        observation('camera_A', .1, 0., seconds=9.95, var=.01)], frame_id=frame))
    raw['observations'][0]['covariance_m2'] = covariance
    message = String(); message.data = json.dumps(raw)
    before_m, before_P = n.belief_m.copy(), n.belief_S.copy()
    n._map_observations_cb(message)
    assert failures
    np.testing.assert_array_equal(n.belief_m, before_m)
    np.testing.assert_array_equal(n.belief_S, before_P)
    assert n._seen_map_observation_stamps == {}


def test_consistent_simultaneous_cameras_contribute_once_independent_of_list_order():
    batch = [observation(c, .01, 0., seconds=9.95, var=.01)
             for c in ('camera_A', 'camera_B')]
    final = []
    for inputs in (batch, list(reversed(batch))):
        n = node(belief_cov=.05)
        n._apply_map_observations(inputs)
        assert n.belief_S[0, 0] == pytest.approx(1/(1/.05 + 2/.01))
        final.append((n.belief_m.copy(), n.belief_S.copy()))
    for a, b in zip(final[0], final[1]): np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('missing', ['prefix', 'interior', 'tail'])
def test_missing_motion_is_distinct_from_complete_camera_outage(missing):
    history = [(k/10, .2, .1) for k in range(101)]
    complete = MotionHistorySnapshot.capture(history, [], True)
    assert plan_replay(complete, 0, 10_000_000_000, 1.5).support.supported
    damaged = {'prefix': history[20:],
               'interior': [e for e in history if not 2. <= e[0] < 5.],
               'tail': history[:70]}[missing]
    plan = plan_replay(MotionHistorySnapshot.capture(damaged, [], True),
                       0, 10_000_000_000, 1.5)
    assert not plan.support.supported
    assert plan.support.gaps
    assert sum(b-a for a,b,*_ in plan.segments) == 10_000_000_000


@pytest.mark.parametrize('mutation', [
    {'frame_id': 'foreign'}, {'schema_version': 99},
    {'xy': [float('nan'), 0.]},
    {'covariance_m2': [[.01, .1], [.1, .01]]}])
def test_malformed_fused_envelope_is_not_statistical_recovery(mutation):
    n = node()
    n.require_state_correction_envelope = True
    failures = []
    n._fatal_experiment_stop = lambda *args: failures.append(args)
    before_m, before_P = n.belief_m.copy(), n.belief_S.copy()
    raw = dict(schema_version=1, frame_id='map_bev', source_batch_id='malformed',
               correction_stamp=9.95, xy=[0.,0.], covariance_m2=[[.01,0.],[0.,.01]])
    raw.update(mutation)
    message = String(); message.data = json.dumps(raw)
    n._state_correction_envelope_cb(message)
    assert failures
    np.testing.assert_array_equal(n.belief_m, before_m)
    np.testing.assert_array_equal(n.belief_S, before_P)
    assert n._seen_state_source_batch_ids == set()
