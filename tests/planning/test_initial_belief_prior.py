"""The belief starts at the declared task start, not at a camera bootstrap."""
import json
import math

import numpy as np
import pytest

from test_outage_motion_replay import _odom_msg
from test_planner_node_state_correction import _envelope, _envelope_node, state_msg


def prior_node(*, now_s=10.0, enabled=True):
    node = _envelope_node(now_s=now_s)
    node._odom_log = []
    node._odom_heading_log = []
    node._odom_accepted_stamp_ns = None
    node._CMD_LOG_MAX_S = 60.0
    node.odom_frame_id, node.odom_child_frame_id = 'odom', 'base_footprint'
    node._latest_odom_yaw = None
    node.odom_yaw_offset_rad = 0.3
    node.initial_belief_from_task_start = enabled
    node.initial_belief_xyyaw = [1.0, 2.0, 0.3]
    node.initial_belief_sigma_xy_m = 0.10
    node.initial_belief_sigma_theta_rad = math.radians(15.0)
    with node._data_lock:
        node._ensure_belief_runtime_locked()
        node._belief_record = None
    return node


def test_first_odometry_starts_the_belief_at_the_declared_start():
    node = prior_node()
    node._odom_cb(_odom_msg(9.9, yaw=0.0))
    record = node._belief_record
    assert record is not None
    np.testing.assert_allclose(record.mean[:2], [1.0, 2.0])
    assert record.mean[2] == pytest.approx(0.3)          # odometry yaw 0 + declared start yaw
    np.testing.assert_allclose(np.diag(record.covariance),
                               [0.01, 0.01, math.radians(15.0) ** 2], rtol=1e-6)
    assert record.stamp_ns == round(9.9e9)
    assert not node.correction_assimilation_pub.published   # not a correction


def test_the_first_camera_batch_is_an_ordinary_update():
    node = prior_node()
    node._odom_cb(_odom_msg(9.9, yaw=0.0))
    node._odom_cb(_odom_msg(9.95, yaw=0.0))
    node._state_cb(state_msg(1.02, 2.01, seconds=9.95))
    node._state_correction_envelope_cb(_envelope(
        source_batch_id='camera_A+camera_B@9950000000', correction_stamp=9.95, xy=(1.02, 2.01)))
    events = [json.loads(p) for p in node.correction_assimilation_pub.published]
    assert len(events) == 1
    assert events[0]['status'] == 'accepted'


def test_disabled_keeps_the_camera_bootstrap():
    node = prior_node(enabled=False)
    node._odom_cb(_odom_msg(9.9, yaw=0.0))
    assert node._belief_record is None


def test_the_prior_is_committed_only_once():
    node = prior_node()
    node._odom_cb(_odom_msg(9.9, yaw=0.0))
    first = node._belief_record
    node._odom_cb(_odom_msg(9.95, yaw=0.0))
    assert node._belief_record is first
