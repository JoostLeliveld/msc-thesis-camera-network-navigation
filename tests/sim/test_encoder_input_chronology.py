"""Old encoder input must not consume an interval or move its anchor.

The encoder integrates one interval per accepted message. When the interval
anchor advanced before the validity gate, an old or duplicate message rebased it
and the next genuine message integrated a shortened interval, silently deleting
real motion from the encoder estimate. These tests hold the whole node state --
pose, covariance, slip states and random generator -- not just the heading.
"""
from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest

from nav_msgs.msg import Odometry
from builtin_interfaces.msg import Time as TimeMsg

from sim.encoder_noise_node import EncoderNoiseNode


def _stamp(seconds: float) -> TimeMsg:
    msg = TimeMsg()
    msg.sec = int(seconds)
    msg.nanosec = int(round((seconds - int(seconds)) * 1e9))
    return msg


def _odom(t, v=0., w=0., yaw=0.) -> Odometry:
    msg = Odometry()
    msg.header.stamp = _stamp(t)
    msg.header.frame_id = 'odom'
    msg.child_frame_id = 'base_footprint'
    msg.twist.twist.linear.x = float(v)
    msg.twist.twist.angular.z = float(w)
    msg.pose.pose.orientation.z = math.sin(yaw / 2.)
    msg.pose.pose.orientation.w = math.cos(yaw / 2.)
    return msg


def _encoder() -> EncoderNoiseNode:
    """The configured node with deterministic zero innovations.

    Every configured mean and the covariance logic are retained; only the random
    draw is fixed, so a difference between two streams is a real state difference.
    """
    node = object.__new__(EncoderNoiseNode)
    for key, value in dict(
            enabled=True, linear_slip_mean=.02, linear_slip_std=.05,
            angular_slip_mean=0., angular_slip_std=.03, linear_additive_std=.004,
            angular_additive_std=.020, correlation_alpha=.80,
            stop_linear_deadband=1e-4, stop_angular_deadband=1e-4, max_dt_s=.5,
            initial_position_std_m=.01, initial_yaw_std_rad=.01,
            linear_scale_bias_std=.02, covariance_floor_m2=1e-8,
            covariance_floor_yaw_rad2=1e-8).items():
        setattr(node, key, value)
    node._rng = SimpleNamespace(gauss=lambda mean, std: 0.)
    node._last_stamp = None
    node._last_stamp_ns = None
    node._linear_slip_state = node._angular_slip_state = 0.
    node._pose_cov = node._initial_pose_covariance()
    node._linear_scale_jacobian = [0., 0., 0.]
    node.messages = []
    node._pub = SimpleNamespace(publish=node.messages.append)
    return node


def _state(node):
    """Everything an old message must not be able to touch."""
    return dict(
        x=node._pose_x, y=node._pose_y, theta=node._pose_theta,
        cov=np.array(node._pose_cov, dtype=float).copy(),
        linear_slip=node._linear_slip_state,
        angular_slip=node._angular_slip_state,
        jacobian=list(node._linear_scale_jacobian),
        anchor_ns=node._last_stamp_ns,
        published=len(node.messages),
    )


def _assert_same_state(a, b):
    assert a['x'] == pytest.approx(b['x'])
    assert a['y'] == pytest.approx(b['y'])
    assert a['theta'] == pytest.approx(b['theta'])
    np.testing.assert_allclose(a['cov'], b['cov'], atol=1e-15)
    assert a['linear_slip'] == pytest.approx(b['linear_slip'])
    assert a['angular_slip'] == pytest.approx(b['angular_slip'])
    assert a['jacobian'] == pytest.approx(b['jacobian'])
    assert a['anchor_ns'] == b['anchor_ns']
    assert a['published'] == b['published']


def test_an_old_message_does_not_shorten_the_next_interval():
    """0, .2, .1, .3 s at 1 rad/s is 0.3 rad of turn, not 0.4.

    The late 0.1 s message used to rebase the anchor, so the following 0.3 s
    message integrated 0.2 s instead of 0.1 s and invented a tenth of a radian.
    """
    node = _encoder()
    for t in (0., .2, .1, .3):
        node._odom_cb(_odom(t, w=1.))
    assert node._pose_theta == pytest.approx(.3, abs=1e-12)


def test_an_inserted_old_message_changes_no_node_state_at_all():
    clean = _encoder()
    for t in (0., .2, .3):
        clean._odom_cb(_odom(t, w=1., v=.2))

    disturbed = _encoder()
    for t in (0., .2, .1, .3):
        disturbed._odom_cb(_odom(t, w=1., v=.2))

    _assert_same_state(_state(clean), _state(disturbed))


def test_a_duplicate_message_consumes_no_interval_and_no_random_draw():
    clean = _encoder()
    for t in (0., .2, .3):
        clean._odom_cb(_odom(t, w=1., v=.2))

    duplicated = _encoder()
    for t in (0., .2, .2, .3):
        duplicated._odom_cb(_odom(t, w=1., v=.2))

    _assert_same_state(_state(clean), _state(duplicated))


def test_a_malformed_message_leaves_the_anchor_and_pose_untouched():
    node = _encoder()
    node._odom_cb(_odom(0., w=1.))
    before = _state(node)
    node._odom_cb(_odom(.1, w=math.nan))
    _assert_same_state(before, _state(node))
    # The stream still advances normally afterwards.
    node._odom_cb(_odom(.1, w=1.))
    assert node._pose_theta == pytest.approx(.1, abs=1e-12)


def test_a_large_gap_still_rebases_the_baseline_so_the_encoder_resumes():
    """Compatibility: the skipped interval is preserved, and so is resumption.

    A positive gap beyond ``max_dt_s`` deliberately rebases the interval baseline.
    Without that, every later message would also exceed the cap and the encoder
    would freeze permanently. The one radian turned during the omitted interval
    is NOT integrated, and this assertion is not a claim that the published pose
    is correct across the gap -- that missing-motion validity defect is open.
    """
    node = _encoder()
    for t in (0., .1, 1.1, 1.2):
        node._odom_cb(_odom(t, w=1.))

    assert node._pose_theta == pytest.approx(.2, abs=1e-12)
    # The 1.0 s interval was skipped, and the node did resume on the next message.
    assert node._last_stamp_ns == 1_200_000_000
    assert node.messages, 'the encoder must keep publishing after a long gap'
