"""The planner node actually routes its pixel corrections through the shared chain.

``test_belief_correction.py`` pins the extracted module. This file pins the
*wiring*: that ``UnicyclePlannerNode._apply_pixel_correction`` still commits the
belief, publishes the same diagnostics vector, and honours each gate -- the
things a module-level test cannot see.

The node is built with ``object.__new__`` and hand-stubbed attributes rather
than ``rclpy.init``: no ROS graph, no spin, no Gazebo.
"""

from __future__ import annotations

import math
import threading

import numpy as np
import pytest

from builtin_interfaces.msg import Time as TimeMsg
from rclpy.clock import ClockType
from rclpy.time import Time

from planning.core import belief_correction as bc
from planning.nodes.unicycle_planner_node import UnicyclePlannerNode


NIS_THRESHOLD = 9.21
MAX_JUMP_M = 0.5


def stamp(seconds: float) -> TimeMsg:
    msg = TimeMsg()
    msg.sec = int(seconds)
    msg.nanosec = int(round((seconds - int(seconds)) * 1e9))
    return msg


class _Clock:
    def __init__(self, seconds: float):
        self.seconds = seconds

    def now(self) -> Time:
        # ROS_TIME to match Time.from_msg on header stamps; the node's
        # _stamp_age_s swallows the TypeError a clock-type mismatch raises and
        # would silently report age 0, disabling the staleness gate under test.
        return Time(nanoseconds=int(self.seconds * 1e9), clock_type=ClockType.ROS_TIME)


class _Logger:
    def __init__(self):
        self.messages = []

    def _record(self, level, message):
        self.messages.append((level, str(message)))

    def info(self, message):
        self._record('info', message)

    def warn(self, message):
        self._record('warn', message)

    def error(self, message):
        self._record('error', message)


class _Publisher:
    def __init__(self):
        self.published = []

    def publish(self, msg):
        self.published.append(
            msg.data if isinstance(msg.data, str) else list(msg.data)
        )


class _Camera:
    H = np.eye(2)

    @staticmethod
    def world_to_pixel(x, y, z):
        return 100.0 + x, 200.0 + y, True


class _Planner:
    """Linear pixel observation h(x) = [x, y] + offset, so the update is checkable."""

    camera = _Camera()

    def __init__(self, r_eff=0.04, p_vis=0.9):
        self.r_eff = float(r_eff)
        self.p_vis = float(p_vis)

    def observation_model_with_visibility(self, m_pred, S_pred):
        R_eff = np.eye(2) * self.r_eff
        return self.p_vis, R_eff, np.asarray(S_pred, dtype=float).copy(), 1.0

    def approx_observation(self, m, S, method=None, R_override=None):
        S = np.asarray(S, dtype=float)
        H = np.hstack([np.eye(2), np.zeros((2, 1))])
        mu_y = H @ np.asarray(m, dtype=float)
        Sigma_y = H @ S @ H.T + np.asarray(R_override, dtype=float)
        Gamma = S @ H.T
        return mu_y, Sigma_y, Gamma

    def predict(self, m, S, cmd, dt):
        return np.asarray(m, dtype=float).copy(), np.asarray(S, dtype=float).copy()


def make_node(
    *,
    belief_xy=(0.0, 0.0),
    meas=(0.05, 0.0),
    belief_cov=0.05,
    r_eff=0.04,
    now_s=10.0,
    belief_stamp_s=9.9,
    max_jump_m=MAX_JUMP_M,
    nis_threshold=NIS_THRESHOLD,
    pixel_timeout_s=1.25,
):
    node = object.__new__(UnicyclePlannerNode)
    node._data_lock = threading.RLock()
    node._correction_lock = threading.RLock()
    node.belief_m = np.array([belief_xy[0], belief_xy[1], 0.0], dtype=float)
    node.belief_S = np.eye(3) * belief_cov
    node.belief_stamp = stamp(belief_stamp_s)
    node._last_correction_stamp = None
    node.last_cmd = np.zeros(2)
    node.pixel_meas = np.array(meas, dtype=float)

    node.dt = 0.25
    # The vehicle's speed ceiling. Used to bound how far the robot could have gone
    # across a gap the belief may not replay, so the covariance can state that.
    node.v_max = 0.22
    node.pixel_timeout_s = pixel_timeout_s
    node.skip_stale_pixel_correction = True
    node.pixel_max_correction_jump_m = max_jump_m
    node.pixel_correction_nis_threshold = nis_threshold
    node.cov_eig_floor = 1e-9
    node.min_state_cov = 1e-6
    node.pixel_correction_min_interval_s = 0.0
    node.approx_method = 'ET1'
    node.pixel_correction_approx = 'AUTO'
    # Off on the single-camera path: honest_campaign_v1 is locked and ran
    # without the prediction cap, so its evidence must stay reproducible.
    node.max_predict_speed_mps = 0.0

    node.planner = _Planner(r_eff=r_eff)
    node.global_planner = None
    node.use_odom_for_predict = False
    node._odom_log = []
    node._cmd_log = []

    node.debug_runtime = False
    node.slow_correction_ms = 1e9
    node._last_stale_log = -1e9
    node._last_shape_mismatch_log = -1e9
    node._last_correction_log = -1e9
    node._last_slow_correction_log = -1e9

    # Heading comes from map-frame odometry; its variance is that odometry's drift
    # since the run began. None here means no odometry has arrived yet.
    node._odom_origin_stamp_s = None
    node.process_noise_theta = 0.02

    # An integrity failure stops the run rather than degrading into a planner that
    # silently never corrects again; the latch makes that stop happen once.
    node._fatal_stop_triggered = False

    node._clock = _Clock(now_s)
    node._logger = _Logger()
    node.pixel_correction_diag_pub = _Publisher()
    node.correction_assimilation_pub = _Publisher()
    node.get_clock = lambda: node._clock
    node.get_logger = lambda: node._logger
    return node


# Diagnostics vector layout, from _publish_pixel_correction_diagnostics.
IDX_XY_UPDATE_NORM = 8
IDX_NIS = 29
IDX_ACCEPTED = 30
IDX_REJECT_CODE = 31


def only_diag(node):
    assert len(node.pixel_correction_diag_pub.published) == 1
    return node.pixel_correction_diag_pub.published[0]


def test_accepted_correction_commits_belief_and_publishes_accepted_diagnostics():
    node = make_node(meas=(0.05, 0.0))
    node._apply_pixel_correction(stamp(9.95), source='callback')

    # Belief moved toward the measurement and the stamp advanced.
    assert node.belief_m[0] > 0.0
    assert node.belief_m[0] < 0.05
    assert node.belief_stamp.sec == 9 and node.belief_stamp.nanosec == 950000000
    assert node._last_correction_stamp == node.belief_stamp
    assert node._last_correction_stamp is not node.belief_stamp

    diag = only_diag(node)
    assert diag[IDX_ACCEPTED] == 1.0
    assert diag[IDX_REJECT_CODE] == bc.ACCEPTED_CODE
    assert diag[IDX_XY_UPDATE_NORM] == pytest.approx(abs(node.belief_m[0]))
    assert math.isfinite(diag[IDX_NIS])


def test_jump_limiter_rejects_and_commits_only_the_prediction():
    """The gate the multicam path silently dropped -- it must still bite here."""
    # Loose prior + tight R -> gain ~1 -> a 3 m innovation moves the belief ~3 m.
    node = make_node(meas=(3.0, 0.0), belief_cov=10.0, r_eff=1e-4)
    before_m = node.belief_m.copy()
    before_stamp = node.belief_stamp

    node._apply_pixel_correction(stamp(9.95), source='callback')

    np.testing.assert_array_equal(node.belief_m, before_m)
    assert node.belief_stamp == stamp(9.95)
    assert before_stamp == stamp(9.9)
    assert node._last_correction_stamp is None

    diag = only_diag(node)
    assert diag[IDX_ACCEPTED] == 0.0
    assert diag[IDX_REJECT_CODE] == bc.REJECT_CODES['jump_too_large']
    assert diag[IDX_XY_UPDATE_NORM] > MAX_JUMP_M
    assert any('exceeds' in m and 'rejecting' in m for _, m in node._logger.messages)


def test_nis_gate_rejects_a_gross_outlier():
    # Tight prior + tight R -> big NIS, but the resulting update stays under the
    # jump limit, so the NIS gate is what fires.
    node = make_node(meas=(0.4, 0.0), belief_cov=1e-3, r_eff=1e-3)
    before_m = node.belief_m.copy()

    node._apply_pixel_correction(stamp(9.95), source='callback')

    np.testing.assert_array_equal(node.belief_m, before_m)
    diag = only_diag(node)
    assert diag[IDX_ACCEPTED] == 0.0
    assert diag[IDX_REJECT_CODE] == bc.REJECT_CODES['nis_too_large']
    assert diag[IDX_NIS] > NIS_THRESHOLD
    assert diag[IDX_XY_UPDATE_NORM] < MAX_JUMP_M


def test_stale_measurement_rejected_before_any_planner_work():
    node = make_node()
    node.planner.approx_observation = lambda *a, **k: pytest.fail("must not linearize")
    before_m = node.belief_m.copy()

    node._apply_pixel_correction(stamp(1.0), source='callback')   # age ~9 s

    np.testing.assert_array_equal(node.belief_m, before_m)
    diag = only_diag(node)
    assert diag[IDX_ACCEPTED] == 0.0
    assert diag[IDX_REJECT_CODE] == bc.REJECT_CODES['stale_age']
    assert any('time-inconsistent' in m for _, m in node._logger.messages)


def test_throttled_correction_publishes_nothing():
    node = make_node()
    node.pixel_correction_min_interval_s = 5.0
    node._last_correction_stamp = stamp(9.94)

    node._apply_pixel_correction(stamp(9.95), source='timer')

    assert node.pixel_correction_diag_pub.published == []


def test_gates_are_built_from_node_parameters():
    node = make_node(max_jump_m=0.7, nis_threshold=5.0, pixel_timeout_s=2.0)
    gates = node._correction_gates()
    assert gates.max_jump_m == 0.7
    assert gates.nis_threshold == 5.0
    assert gates.pixel_timeout_s == 2.0
    assert gates.dt_nominal_s == node.dt
    assert gates.min_state_cov == node.min_state_cov
    assert gates.cov_eig_floor == node.cov_eig_floor
    assert gates.skip_stale is True


def test_snapshot_is_the_shared_dataclass_carrying_the_measurement_stamp():
    node = make_node()
    stamp_msg = stamp(9.95)
    snapshot = node._snapshot_pixel_correction_inputs(stamp_msg)
    assert isinstance(snapshot, bc.CorrectionSnapshot)
    assert snapshot.meas_stamp == stamp_msg
    assert snapshot.meas_stamp is not stamp_msg
    assert snapshot.belief_stamp == node.belief_stamp
    assert snapshot.belief_stamp is not node.belief_stamp
    np.testing.assert_array_equal(snapshot.meas, node.pixel_meas)

    node.pixel_meas = None
    assert node._snapshot_pixel_correction_inputs(stamp_msg) is None


def test_regularize_delegates_to_the_shared_implementation():
    node = make_node()
    S = np.array([[1e-9, 0.0, 0.0], [0.0, 2.0, 0.5], [0.0, 0.5, 3.0]])
    np.testing.assert_allclose(
        node._regularize_state_covariance(S),
        bc.regularize_covariance(S, node.min_state_cov),
        rtol=0, atol=0,
    )
    assert node._regularize_state_covariance(S)[0, 0] == node.min_state_cov
