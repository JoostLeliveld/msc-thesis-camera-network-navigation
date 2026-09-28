"""The anchor follows the clock at a fixed lag, so a long camera gap cannot grow the replay."""
import math

import numpy as np

from test_outage_motion_replay import MotionModel
from test_planner_node_state_correction import make_state_node
from test_planner_node_correction_wiring import stamp


def blind_drive_node(duration_s=12.0):
    """A robot that drives and turns with odometry only, never seen by a camera."""
    node = make_state_node(belief_stamp_s=0., now_s=0.)
    node.heading_update_mode = 'coupled'
    node.use_odom_for_predict = True
    node.planner = MotionModel()
    node._odom_log = [(float(t), .2, math.pi / 8 if 3 <= t < 7 else 0.)
                      for t in np.arange(0., duration_s + 1e-9, .02)]
    with node._data_lock:
        node._ensure_belief_runtime_locked()
    return node


def replay_to(node, t_s):
    """The belief a query sees at t_s: replay from the current anchor."""
    record = node._belief_record
    age = t_s - record.stamp_ns * 1e-9
    return node._predict_belief_to_now(*record.arrays(), node.last_cmd, age, stamp(t_s))


def test_advancing_the_anchor_leaves_the_belief_at_now_unchanged():
    reference = blind_drive_node()
    m_ref, P_ref = replay_to(reference, 12.)
    node = blind_drive_node()
    for t in np.arange(.1, 12.01, .1):
        node._clock.seconds = float(t)
        node._advance_belief_anchor()
    m, P = replay_to(node, 12.)
    # Only the split of odometry segments at anchor times differs.
    np.testing.assert_allclose(m, m_ref, atol=2e-3)
    np.testing.assert_allclose(P, P_ref, rtol=1e-6, atol=1e-9)


def test_replay_length_stays_bounded_during_a_camera_gap():
    node = blind_drive_node()
    for t in np.arange(.1, 12.01, .1):
        node._clock.seconds = float(t)
        node._advance_belief_anchor()
    lag = 12. - node._belief_record.stamp_ns * 1e-9
    assert lag <= 2 * node.pixel_timeout_s + node._ANCHOR_ADVANCE_MIN_STEP_S + .1 + 1e-9


def test_every_admissible_correction_still_lies_after_the_anchor():
    node = blind_drive_node()
    node._clock.seconds = 12.
    node._advance_belief_anchor()
    oldest_admissible_s = 12. - node.pixel_timeout_s
    assert node._belief_record.stamp_ns * 1e-9 < oldest_admissible_s


def test_an_interval_without_odometry_is_not_committed():
    node = blind_drive_node(duration_s=4.)
    node._clock.seconds = 12.
    before = node._belief_record
    node._advance_belief_anchor()
    assert node._belief_record is before


def test_no_advance_before_the_lag_is_reached():
    node = blind_drive_node()
    node._clock.seconds = 2 * node.pixel_timeout_s + .2
    before = node._belief_record
    node._advance_belief_anchor()
    assert node._belief_record is before
