"""Camera absence must not discard available motion, especially a blind turn."""
import math

import numpy as np
import pytest

from planning.core.dynamics import unicycle_step
from planning.core.motion_history import MotionHistorySnapshot, covers_interval
from test_planner_node_state_correction import make_state_node
from test_planner_node_correction_wiring import stamp


def _odom_msg(t, v=0., w=0., yaw=0.):
    """One timestamped odometry message, as the node's callback expects it."""
    from nav_msgs.msg import Odometry
    msg = Odometry()
    msg.header.stamp = stamp(t)
    msg.header.frame_id = 'odom'
    msg.child_frame_id = 'base_footprint'
    msg.twist.twist.linear.x = float(v)
    msg.twist.twist.angular.z = float(w)
    msg.pose.pose.orientation.z = math.sin(yaw / 2.)
    msg.pose.pose.orientation.w = math.cos(yaw / 2.)
    return msg


@pytest.mark.parametrize('entries,start,end,expected', [
    ([(0., .2, 0.), (1., .2, 0.), (2., .2, 0.)], .1, 2.5, True),
    ([(1., .2, 0.), (2., .2, 0.)], .1, 2.5, False),
    ([(0., .2, 0.), (2., .2, 0.)], .1, 2.5, False),
    ([(0., .2, 0.), (1., .2, 0.)], .1, 3., False),
    ([(0., .2, 0.), (1., math.nan, 0.)], .1, 1.2, False),
    ([(0., .2, 0.), (1., .2, 0.), (.5, .2, 0.)], .1, 1.2, False),
    ([(0., .2, 0.)], 2., 2.1, False),
    ([], .1, 2., False),
])
def test_motion_coverage(entries, start, end, expected):
    assert covers_interval(entries, start, end, 1.5) is expected


class MotionModel:
    @staticmethod
    def predict(m, P, u, dt):
        # Independent exact unicycle mean oracle, with nonzero process growth.
        return unicycle_step(m, u, dt), P + np.diag([.01, .01, .02])**2 * dt


def test_returning_camera_cannot_erase_a_recorded_blind_turn():
    node = make_state_node(belief_stamp_s=0., now_s=10.1)
    node.heading_update_mode = 'coupled'
    node.use_odom_for_predict = True
    node.planner = MotionModel()
    # Drive east, turn north, then drive north. Every input sample is present;
    # the camera alone is absent for ten seconds.
    node._odom_log = [(float(t), .2 if t < 2 or t >= 4 else 0.,
                       math.pi/4 if 2 <= t < 4 else 0.)
                      for t in np.arange(0., 10.01, .1)]
    expected, covariance = node._predict_belief_to_now(
        node.belief_m.copy(), node.belief_S.copy(), node.last_cmd, 10., stamp(10.))
    node._advance_belief_over_outage(stamp(10.), 10.)
    np.testing.assert_allclose(node.belief_m, expected, atol=1e-10)
    np.testing.assert_allclose(node.belief_S, covariance, atol=1e-10)
    assert node.belief_m[2] == pytest.approx(math.pi/2)
    assert node._stamp_to_float(node.belief_stamp) == pytest.approx(10.)


def test_unsupported_motion_is_not_treated_as_a_complete_replay():
    node = make_state_node(belief_stamp_s=0., now_s=10.1)
    node.use_odom_for_predict = True
    node.planner = MotionModel()
    node._odom_log = [(9., .2, 0.), (10., .2, 0.)]
    before = node.belief_S.copy()
    node._advance_belief_over_outage(stamp(10.), 10.)
    # This is the existing conservative fallback, not the complete-history case.
    assert node.belief_m[0] <= .3
    assert node.belief_S[0, 0] > before[0, 0] + 1.


def test_coverage_check_and_replay_cannot_see_different_histories():
    """A trim arriving between the check and the replay must not erase the turn.

    The coverage check accepted a full 60 s history containing a 0.2 rad turn.
    When the check and the replay were two separate reads of the live buffer, an
    odometry callback landing in between could trim that turn away, so the belief
    stamp advanced 60 s over motion that was never integrated.
    """
    node = make_state_node(belief_stamp_s=0., now_s=60.)
    node.use_odom_for_predict = True
    node.heading_update_mode = 'coupled'
    node.planner = MotionModel()
    node._CMD_LOG_MAX_S = 60.
    node._odom_log = [(k/10., 0., 1. if k < 2 else 0.) for k in range(600)]
    assert covers_interval(node._odom_log, 0., 59.9, 1.5)

    predict = node._predict_belief_to_now
    trimmed = []

    def input_arrives_after_coverage_check(*args, **kwargs):
        # Runs after the coverage read and before replay. Without a shared
        # snapshot this trims the already-verified turn out of the buffer.
        node._clock.seconds = 60.2
        node._odom_cb(_odom_msg(60.2))
        trimmed.append(node._odom_log[0][0])
        return predict(*args, **kwargs)

    node._predict_belief_to_now = input_arrives_after_coverage_check
    node._advance_belief_over_outage(stamp(59.9), 59.9)

    # The interfering callback really did trim the verified turn from the buffer.
    assert trimmed and trimmed[0] == pytest.approx(.2)

    # The 0.2 rad turn recorded in the checked history is still integrated.
    assert node.belief_m[2] == pytest.approx(.2, abs=1e-10)
    assert node._stamp_to_float(node.belief_stamp) == pytest.approx(59.9)


def test_snapshot_fixes_both_the_source_choice_and_the_fallback_entries():
    """Freezing only the preferred buffer would leave the fallback mutable.

    Prediction falls back from odometry to commands when odometry has no samples
    for the interval. Both lists and the source flag are therefore captured.
    """
    node = make_state_node(belief_stamp_s=0., now_s=10.1)
    node.use_odom_for_predict = True
    node.planner = MotionModel()
    node._odom_log = []
    node._cmd_log = [(float(t), .2, 0.) for t in np.arange(0., 10.01, .1)]

    snapshot = MotionHistorySnapshot.capture(
        node._odom_log, node._cmd_log, node.use_odom_for_predict)
    # Mutate BOTH live buffers after capture, including the fallback.
    node._odom_log.append((5., 9., 9.))
    node._cmd_log.clear()
    node.use_odom_for_predict = False

    m, _ = node._predict_belief_to_now(
        node.belief_m.copy(), node.belief_S.copy(), node.last_cmd, 10., stamp(10.),
        motion_snapshot=snapshot)

    # Command fallback over the frozen entries: 0.2 m/s for 10 s.
    assert m[0] == pytest.approx(2., abs=1e-9)
    assert node._latest_prediction_source == pytest.approx(2.)


def _node_with_odom_buffer():
    node = make_state_node(belief_stamp_s=9.4, now_s=10.1)
    node.use_odom_for_predict = True
    node.planner = MotionModel()
    node._CMD_LOG_MAX_S = 60.
    node._odom_log = []
    return node


def test_late_odometry_is_refused_rather_than_inserted_out_of_order():
    """9.4, 9.7, then a late 9.5 must leave the accepted history at 9.4, 9.7.

    Appending the late sample put a backwards entry in the buffer, which replay
    then integrated as extra motion: a 0.6 s interval covered 0.8 s of duration,
    overstating travel by a third and inflating heading process noise with it.
    Refusal is chosen over sorted insertion because a retrospective insert would
    change a history an accepted correction was already computed from.
    """
    node = _node_with_odom_buffer()
    for t in (9.4, 9.7, 9.5):
        node._odom_cb(_odom_msg(t, .2, 0.))

    assert [entry[0] for entry in node._odom_log] == pytest.approx([9.4, 9.7])
    assert node._odom_refused_old == 1

    m, _ = node._predict_belief_to_now(
        node.belief_m.copy(), node.belief_S.copy(), node.last_cmd, .6, stamp(10.))
    # 0.2 m/s over the real 0.6 s interval, not the 0.8 s the duplicate created.
    assert m[0] == pytest.approx(.12, abs=1e-9)


def test_an_equal_stamp_message_adds_no_motion_and_keeps_the_first_event():
    node = _node_with_odom_buffer()
    node._odom_cb(_odom_msg(9.4, .2, 0.))
    node._odom_cb(_odom_msg(9.4, 5.0, 5.0))

    assert [entry[0] for entry in node._odom_log] == pytest.approx([9.4])
    # The first event at that timestamp is retained; the conflicting one is not.
    assert node._odom_log[0][1] == pytest.approx(.2)
    assert node.odom_vel[0] == pytest.approx(.2)
    assert node._odom_refused_duplicate == 1


def test_a_late_callback_cannot_pair_its_yaw_with_another_events_velocity():
    """Yaw, velocity, origin and history are one commit or none.

    Yaw and the origin stamp used to be written before the lock and before any
    ordering check, so a delayed message could leave the estimator holding one
    event's heading beside a newer event's velocity.
    """
    node = _node_with_odom_buffer()
    node._clock.seconds = 11.
    node._odom_cb(_odom_msg(11., .2, 0., yaw=.5))
    node._odom_cb(_odom_msg(10., 9., 9., yaw=-1.3))

    assert node._latest_odom_yaw == pytest.approx(.5)
    assert node.odom_vel[0] == pytest.approx(.2)
    assert [entry[0] for entry in node._odom_log] == pytest.approx([11.])


def test_a_malformed_message_does_not_poison_the_watermark():
    node = _node_with_odom_buffer()
    node._odom_cb(_odom_msg(9.4, math.nan, 0.))
    assert node._odom_log == []
    assert node._odom_refused_invalid == 1
    # A later valid message is still accepted: the refusal set no watermark.
    node._odom_cb(_odom_msg(9.5, .2, 0.))
    assert [entry[0] for entry in node._odom_log] == pytest.approx([9.5])
