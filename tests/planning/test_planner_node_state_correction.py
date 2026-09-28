"""The fused correction path runs the same guarded chain as the pixel path.

``_apply_state_correction`` reaches ``planning.core.belief_correction`` through
``FusedMapMeasurementSource``, so both measurement paths share one gate chain. These
tests pin the three properties that made a second, separately written chain dangerous:

1. the jump limiter must actually be read on this path, not merely configured;
2. a NIS rejection must not inflate S so far that the gate is open for the next
   correction -- a self-widening gate stops rejecting anything;
3. every rejection must carry a reason code, or a gate doing nothing is
   indistinguishable in the log from a gate that is working.

Plus the behaviours the separate chain got RIGHT, which must survive the merge:
bootstrap, re-anchor on divergence, and never freezing the belief on a rejection.
"""

from __future__ import annotations

import json
import math
import threading

import numpy as np
import pytest

from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import String

from planning.core import belief_correction as bc

from test_planner_node_correction_wiring import (
    IDX_ACCEPTED,
    IDX_NIS,
    IDX_REJECT_CODE,
    IDX_XY_UPDATE_NORM,
    make_node,
    stamp,
)

IDX_MEASUREMENT_SPACE = 44
IDX_PREDICT_CLIPPED = 45

STATE_REANCHOR_M = 2.0
REJECT_INFLATE_M2 = 0.05


def state_msg(x, y, *, seconds, var=0.03):
    msg = PoseWithCovarianceStamped()
    msg.header.stamp = stamp(seconds)
    msg.header.frame_id = 'map'
    msg.pose.pose.position.x = float(x)
    msg.pose.pose.position.y = float(y)
    msg.pose.pose.orientation.w = 1.0
    cov = [0.0] * 36
    cov[0] = float(var)
    cov[7] = float(var)
    cov[35] = float(math.pi ** 2)
    msg.pose.covariance = cov
    return msg


def make_state_node(*, belief_xy=(0.0, 0.0), belief_cov=0.05, now_s=10.0,
                    belief_stamp_s=9.9, odom_yaw=0.25, **kwargs):
    node = make_node(
        belief_xy=belief_xy, belief_cov=belief_cov, now_s=now_s,
        belief_stamp_s=belief_stamp_s, **kwargs
    )
    node.state_correction_ekf = True
    node.state_correction_mode = 'fused'
    node.use_pixel_correction = False
    node.state_reanchor_m = STATE_REANCHOR_M
    node.state_max_predict_dt_s = 1.5
    node.state_reject_inflate_m2 = REJECT_INFLATE_M2
    node.state_max_correction_jump_m = 0.0   # NIS is the gate on the metric path
    node.odom_yaw_offset_rad = 0.0
    node._latest_odom_yaw = odom_yaw
    # Stamped preceding yaw support, rather than an unclocked latest-yaw field.
    node._odom_log = [(belief_stamp_s, 0., 0.)]
    node._odom_heading_log = [(round(belief_stamp_s*1e9), odom_yaw)] if odom_yaw is not None else []
    node.state_msg = None
    return node


def only_diag(node):
    assert len(node.pixel_correction_diag_pub.published) == 1, (
        "the multicam path must publish exactly one reason-coded diagnostic "
        "per correction -- it published none before the consolidation"
    )
    return node.pixel_correction_diag_pub.published[0]


# --------------------------------------------------------------------------
# Behaviours the separate chain had right; they must survive the merge
# --------------------------------------------------------------------------

def test_bootstraps_the_belief_from_the_first_fresh_correction():
    node = make_state_node()
    node.belief_m = None
    node.belief_S = None
    node.belief_stamp = None

    node._apply_state_correction(state_msg(1.0, 2.0, seconds=9.95))

    assert node.belief_m is not None
    assert node.belief_m[0] == pytest.approx(1.0)
    assert node.belief_m[1] == pytest.approx(2.0)
    # /state/bev carries no heading in multicam; it comes from map-frame odom.
    assert node.belief_m[2] == pytest.approx(0.25)


def test_ignores_a_correction_not_newer_than_the_belief():
    node = make_state_node(belief_stamp_s=9.95)
    before = node.belief_m.copy()
    node._apply_state_correction(state_msg(1.0, 0.0, seconds=9.95))
    np.testing.assert_array_equal(node.belief_m, before)
    diag = only_diag(node)
    assert diag[IDX_REJECT_CODE] == bc.REJECT_CODES['not_newer_than_belief']


def test_source_batch_gets_one_terminal_assimilation_record():
    node = make_state_node()
    node._apply_metric_correction(
        stamp(9.95), np.array([0.05, 0.0]), np.eye(2) * 0.03,
        source_batch_id='strict:camera_A@9950000000',
    )
    assert len(node.correction_assimilation_pub.published) == 1
    payload = json.loads(node.correction_assimilation_pub.published[0])
    assert payload['source_batch_id'] == 'strict:camera_A@9950000000'
    assert payload['status'] == 'accepted'
    assert payload['accepted'] is True


def test_an_implausible_replay_gap_is_rejected_but_the_clock_still_advances():
    """A gap this long means the prior cannot be replayed, so there is no valid
    innovation to gate on. Snapping to the measurement here used to be the one path
    that bypassed both the outlier gate and the re-anchor threshold: one camera's word
    could move the belief anywhere, unchecked.

    The measurement is refused, but the belief STAMP must still advance. Refusing
    without advancing is an absorbing state: the gap grows by one detector period every
    cycle, so every later correction is refused for the same reason. Measured on a live
    drive before this was fixed -- one 1.4 s startup gap produced 158 consecutive
    refusals, the drive finished on odometry alone, and it ended in a collision.
    Widening the covariance does not help, because the gate is on dt, not on covariance.
    """
    node = make_state_node(belief_stamp_s=5.0, now_s=10.0)
    before_S = node.belief_S.copy()
    measurement = np.array([3.0, 4.0], dtype=float)

    # dt = 4.9 s > state_max_predict_dt_s; replaying would fling the prediction.
    node._apply_state_correction(state_msg(3.0, 4.0, seconds=9.9))

    # refused: the belief did not snap to what the camera said
    assert np.linalg.norm(node.belief_m[:2] - measurement) > 1.0
    # widened
    assert node.belief_S[0, 0] > before_S[0, 0]
    assert node.belief_S[1, 1] > before_S[1, 1]
    # and, crucially, the clock moved, so the next correction is gateable
    assert node._stamp_to_float(node.belief_stamp) == pytest.approx(9.9)


def test_a_refused_correction_never_locks_the_filter_out():
    """The absorbing state, end to end: refuse once, then feed the ordinary 5 Hz
    stream and require that it starts being accepted again."""
    node = make_state_node(belief_stamp_s=5.0, now_s=10.0)

    # one long gap, refused
    node._apply_state_correction(state_msg(3.0, 4.0, seconds=9.9))

    assert np.linalg.norm(node.belief_m[:2]) < 0.5, "must not snap to the refused reading"
    widened = node.belief_S[0, 0]

    # then the detector's normal cadence resumes, all agreeing on x = 0.30
    for i in range(1, 6):
        t_s = 9.9 + 0.2 * i
        node._clock.seconds = t_s + 0.05
        node._apply_state_correction(state_msg(0.30, 0.0, seconds=t_s))

    # the belief converged on the cameras and its uncertainty came back down, which can
    # only happen if those corrections were assimilated rather than refused in turn
    assert node.belief_m[0] == pytest.approx(0.30, abs=0.02), (
        f"belief x is {node.belief_m[0]:.3f}, not the 0.30 every later camera reported: "
        "the filter never recovered from one refused correction"
    )
    assert node.belief_S[0, 0] < widened / 10.0
    assert node._stamp_to_float(node.belief_stamp) == pytest.approx(10.9)


def test_planning_prediction_is_read_only_so_delayed_correction_still_lands():
    """A planning tick at now must not move the committed belief past an image
    that is still travelling through the detector pipeline."""
    node = make_state_node(belief_stamp_s=9.6, now_s=10.0)
    before_stamp = node._stamp_to_float(node.belief_stamp)

    node._predict_belief_to_now(
        node.belief_m, node.belief_S, node.last_cmd,
        belief_age_s=0.4, now_msg=stamp(10.0),
    )
    assert node._stamp_to_float(node.belief_stamp) == pytest.approx(before_stamp)

    node._apply_state_correction(state_msg(0.05, 0.0, seconds=9.8))
    assert node._stamp_to_float(node.belief_stamp) == pytest.approx(9.8)
    assert node.belief_m[0] > 0.0


def test_coupled_heading_is_not_replaced_only_for_planning():
    node = make_state_node(odom_yaw=1.2)
    node.heading_update_mode = 'coupled'
    mean = np.array([1.0, 2.0, 0.35])
    covariance = np.array([
        [0.2, 0.0, 0.03],
        [0.0, 0.2, -0.02],
        [0.03, -0.02, 0.1],
    ])
    anchored_m, anchored_S = node._anchor_belief_yaw_for_planning(
        mean, covariance, stamp(10.0))
    np.testing.assert_allclose(anchored_m, mean)
    np.testing.assert_allclose(anchored_S, covariance)


def test_reanchors_when_the_belief_has_diverged():
    """A diverged BELIEF must snap to the correction, not reject it.

    Rejecting would lock the belief out of recovery and let it dead-reckon away.
    """
    node = make_state_node(belief_xy=(0.0, 0.0))
    node._apply_state_correction(state_msg(5.0, 0.0, seconds=9.95))   # 5 m > 2 m

    assert node.belief_m[0] == pytest.approx(5.0)
    diag = only_diag(node)
    assert diag[IDX_REJECT_CODE] == bc.REJECT_CODES['diverged']
    assert diag[IDX_ACCEPTED] == 0.0


def test_accepted_correction_commits_and_keeps_heading_from_odom():
    node = make_state_node(belief_xy=(0.0, 0.0), odom_yaw=0.4)
    node._apply_state_correction(state_msg(0.05, 0.0, seconds=9.95))

    assert 0.0 < node.belief_m[0] < 0.05
    assert node.belief_m[2] == pytest.approx(0.4)     # camera_xy_only
    diag = only_diag(node)
    assert diag[IDX_ACCEPTED] == 1.0
    assert diag[IDX_REJECT_CODE] == bc.ACCEPTED_CODE
    assert diag[IDX_MEASUREMENT_SPACE] == bc.SPACE_MAP_XY


def test_camera_xy_only_commit_keeps_the_belief_covariance_psd():
    """Never splice prior heading correlations into a posterior xy block.

    The live multicamera showcase exposed this exact failure: the resulting
    hybrid covariance became indefinite, then NIS went negative (which is
    mathematically impossible for a valid innovation covariance).
    """
    node = make_state_node()
    S_pred = np.array([
        [1.0, 0.0, 0.9],
        [0.0, 1.0, 0.0],
        [0.9, 0.0, 1.0],
    ])
    outcome = bc.CorrectionOutcome(
        reason=bc.RejectReason.ACCEPTED,
        m_pred=np.zeros(3),
        S_pred=S_pred,
        next_m=np.zeros(3),
        next_S=np.diag([0.01, 0.01, 0.10]),
    )

    node._commit_metric_correction_outcome(
        stamp(9.95), np.diag([0.02, 0.02]), outcome
    )

    assert np.min(np.linalg.eigvalsh(node.belief_S)) >= node.cov_eig_floor
    np.testing.assert_allclose(node.belief_S[:2, 2], 0.0, atol=1e-15)
    np.testing.assert_allclose(node.belief_S[2, :2], 0.0, atol=1e-15)
    assert node.belief_S[2, 2] == pytest.approx(S_pred[2, 2])


def test_camera_xy_only_commit_uses_the_odometry_heading_variance():
    """One heading model: the committed heading variance is the one the read-out reports.

    Keeping the predicted record variance instead carried the declared initial heading
    prior through the whole run, and prediction turned it into metres of cross-track
    uncertainty in camera-free stretches.
    """
    node = make_state_node()
    node._odom_origin_stamp_s = 0.0
    S_pred = np.diag([1.0, 1.0, 0.3])
    outcome = bc.CorrectionOutcome(
        reason=bc.RejectReason.ACCEPTED,
        m_pred=np.zeros(3),
        S_pred=S_pred,
        next_m=np.zeros(3),
        next_S=np.diag([0.01, 0.01, 0.10]),
    )

    node._commit_metric_correction_outcome(
        stamp(9.95), np.diag([0.02, 0.02]), outcome
    )

    expected = node._map_frame_heading_variance(stamp(9.95))
    assert expected < S_pred[2, 2]
    assert node.belief_S[2, 2] == pytest.approx(expected)
    assert np.min(np.linalg.eigvalsh(node.belief_S)) >= node.cov_eig_floor


# --------------------------------------------------------------------------
# The three confirmed gaps
# --------------------------------------------------------------------------

def test_gap1_the_gate_is_now_read_on_the_multicam_path():
    """GAP 1 revisited.

    The audit's finding was that this path read NO update gate at all: the
    configured limit was ignored and nothing bounded the correction. The fix is
    that a gate now runs -- but on this path it is NIS, not a jump limit. See
    ``test_jump_limiter_is_off_on_the_metric_path_by_design`` for why.
    """
    node = make_state_node(belief_xy=(0.0, 0.0), belief_cov=1e-3)
    before = node.belief_m.copy()

    node._apply_state_correction(state_msg(1.5, 0.0, seconds=9.95, var=1e-3))

    diag = only_diag(node)
    assert diag[IDX_REJECT_CODE] == bc.REJECT_CODES['nis_too_large']
    # Belief held at the prediction, NOT snapped to the correction.
    assert node.belief_m[0] == pytest.approx(before[0], abs=1e-9)


def test_jump_limiter_is_off_on_the_metric_path_by_design():
    """A jump limit here DEADLOCKS recovery, so the metric path does not use one.

    NIS and a jump limit fire in opposite regimes: with a confident belief NIS
    rejects a large innovation while the gain (and so the update) is small; with
    an uncertain belief NIS passes while the update is large. So the jump limit
    only ever fires when a big correction is WARRANTED.

    Worse, it is self-reinforcing: rejecting inflates S, which raises the gain,
    which raises the update, which trips the limit again. A belief genuinely
    1.5 m off would never recover until drift pushed the innovation past
    state_reanchor_m -- it would have to get worse first.
    """
    node = make_state_node()
    assert node.state_max_correction_jump_m == 0.0
    assert node._correction_gates(metric_measurement=True).max_jump_m == 0.0
    # The locked paper-1 pixel path keeps its limit; it is frozen method.
    assert node._correction_gates().max_jump_m == pytest.approx(0.5)


def test_a_genuinely_lost_belief_recovers_monotonically():
    """The case the jump limiter used to block: belief 1.5 m off, cameras agree.

    Every correction says the same thing, so the belief -- not the measurement --
    is what is wrong. It must converge, and within a bounded number of samples.
    """
    node = make_state_node(belief_xy=(0.0, 0.0), belief_cov=1e-3)
    node.state_max_predict_dt_s = 100.0

    accepted_at = None
    for k in range(12):
        t = 9.95 + 0.2 * k
        node._clock.seconds = t + 0.05
        node._apply_state_correction(state_msg(1.5, 0.0, seconds=t, var=1e-3))
        if node.pixel_correction_diag_pub.published[-1][IDX_ACCEPTED] == 1.0:
            accepted_at = k + 1
            break
    assert accepted_at is not None, "belief never recovered -- the gate deadlocks"
    assert accepted_at <= 8, f"recovery took {accepted_at} corrections"
    assert node.belief_m[0] == pytest.approx(1.5, abs=0.1)


def test_rejection_advances_the_stamp_and_inflates_but_does_not_freeze():
    """The property the separate chain existed to protect -- keep it."""
    node = make_state_node(belief_xy=(0.0, 0.0), belief_cov=1e-3)
    S_before = node.belief_S.copy()

    node._apply_state_correction(state_msg(1.5, 0.0, seconds=9.95, var=1e-3))

    assert node.belief_stamp.nanosec == 950000000      # stamp advanced
    assert node.belief_S[0, 0] > S_before[0, 0]        # covariance grew
    # ...but NOT by the old +1.0 m^2 sledgehammer.
    assert node.belief_S[0, 0] - S_before[0, 0] < 0.5


def test_the_measured_1_87m_dead_band_jump_stays_rejected_after_a_rejection():
    """GAP 2 -- the exact failure the parity audit measured.

    1.87 m in one 0.1 s sample (physically impossible at v_max 0.6 m/s), sitting
    in a dead band: too large for the *inflated* NIS gate to stop, but below
    state_reanchor_m = 2.0 so the re-anchor never fired either. The old path
    inflated S by +1.0 m^2 on reject, so the next correction saw S_y ~ 1.0,
    scored NIS ~3.5, and was applied at gain ~0.999 -- an un-gated snap.

    With the mild inflation the second one is still rejected (NIS ~67).
    """
    node = make_state_node(belief_xy=(0.0, 0.0), belief_cov=1e-3)
    assert 1.87 < node.state_reanchor_m, "must stay inside the dead band"

    node._apply_state_correction(state_msg(1.87, 0.0, seconds=9.95, var=1e-3))
    assert only_diag(node)[IDX_ACCEPTED] == 0.0
    assert node.belief_S[0, 0] < 1.0, "inflation must not approach the old +1.0 m^2"

    node.pixel_correction_diag_pub.published.clear()
    node._clock.seconds = 10.05
    node._apply_state_correction(state_msg(1.87, 0.0, seconds=10.0, var=1e-3))

    second = only_diag(node)
    assert second[IDX_ACCEPTED] == 0.0, (
        "the 1.87 m outlier was accepted on the second correction -- the "
        "reject-inflation has neutered the gate, which is the original bug"
    )
    assert node.belief_m[0] < 0.5, "the belief snapped to the phantom pose"


def test_a_modest_persistent_disagreement_is_eventually_accepted():
    """The other half of the trade-off, asserted so it cannot silently regress.

    A rejection must never *freeze* the belief. When corrections keep saying the
    belief is off by a modest amount, the belief is the thing that is probably
    wrong, so the growing covariance should let the next one through -- as a
    bounded update under the jump limit, not a snap. The gross case is covered
    by the state_reanchor_m guard, the impossible case by the test above.
    """
    node = make_state_node(belief_xy=(0.0, 0.0), belief_cov=1e-3)

    node._apply_state_correction(state_msg(0.35, 0.0, seconds=9.95, var=1e-3))
    assert only_diag(node)[IDX_REJECT_CODE] == bc.REJECT_CODES['nis_too_large']

    node.pixel_correction_diag_pub.published.clear()
    node._clock.seconds = 10.05
    node._apply_state_correction(state_msg(0.35, 0.0, seconds=10.0, var=1e-3))

    assert only_diag(node)[IDX_ACCEPTED] == 1.0
    assert node.belief_m[0] == pytest.approx(0.35, abs=0.05)
    assert only_diag(node)[IDX_XY_UPDATE_NORM] <= node.pixel_max_correction_jump_m


def test_every_outcome_publishes_a_reason_coded_diagnostic():
    """GAP 3: this path published no rejection diagnostics at all."""
    cases = [
        (state_msg(0.05, 0.0, seconds=9.95), bc.ACCEPTED_CODE),
        (state_msg(5.00, 0.0, seconds=9.95), bc.REJECT_CODES['diverged']),
    ]
    for msg, expected_code in cases:
        node = make_state_node(belief_xy=(0.0, 0.0))
        node._apply_state_correction(msg)
        diag = only_diag(node)
        assert diag[IDX_REJECT_CODE] == expected_code
        assert diag[IDX_MEASUREMENT_SPACE] == bc.SPACE_MAP_XY, (
            "a CSV reader must be able to tell that innov/meas are metres here, "
            "not pixels -- the pixel_corr_* columns are shared by both stacks"
        )


# --------------------------------------------------------------------------
# Kinematic plausibility cap on the prediction
# --------------------------------------------------------------------------

def test_prediction_clamp_bounds_an_implausible_replayed_step():
    node = make_state_node(belief_xy=(0.0, 0.0), belief_stamp_s=9.9)
    node.max_predict_speed_mps = 0.6

    # Replay invents 5 m of travel over the 0.05 s interval.
    def _runaway_replay(m, S, from_stamp, to_stamp, cmd, dt, *, motion_snapshot=None):
        m = np.asarray(m, dtype=float).copy()
        m[0] += 5.0
        return m, np.asarray(S, dtype=float), {}

    node._replay_cmd_log_interval = _runaway_replay
    node._apply_state_correction(state_msg(0.05, 0.0, seconds=9.95))

    diag = only_diag(node)
    clipped = diag[IDX_PREDICT_CLIPPED]
    assert clipped > 4.0, "the invented step should have been clamped"
    # Cap = v_max * dt + margin = 0.6 * 0.05 + 0.05 = 0.08 m.
    assert node.belief_m[0] < 1.0


def test_prediction_clamp_is_off_by_default_so_paper1_is_untouched():
    node = make_state_node(belief_xy=(0.0, 0.0))
    assert node.max_predict_speed_mps == 0.0
    assert not math.isfinite(node._correction_gates().max_predict_step_m(0.1))


def test_gates_expose_the_divergence_guard_only_for_metric_measurements():
    node = make_state_node()
    assert node._correction_gates(metric_measurement=True).reanchor_innov_m == STATE_REANCHOR_M
    # A pixel innovation has no metric scale, so the guard must stay off there.
    assert node._correction_gates().reanchor_innov_m == 0.0


# --------------------------------------------------------------------------
# Anisotropic measurement covariance must survive into the filter
# --------------------------------------------------------------------------

def anisotropic_state_msg(x, y, *, seconds, vxx, vyy, vxy):
    """A /state/bev pose whose xy covariance has real cross-correlation.

    The commissioned profile propagates the pixel covariance through the
    projection as J R Jt, so an oblique camera yields a long thin ellipse whose
    axes are NOT aligned with map x/y -- i.e. a non-zero cov[1] and cov[6].
    """
    msg = state_msg(x, y, seconds=seconds)
    cov = list(msg.pose.covariance)
    cov[0] = float(vxx)
    cov[1] = float(vxy)
    cov[6] = float(vxy)
    cov[7] = float(vyy)
    msg.pose.covariance = cov
    return msg


def test_fused_path_preserves_map_frame_cross_covariance():
    """Reading only cov[0]/cov[7] would silently isotropise the measurement."""
    node = make_state_node()
    R = node._state_measurement_cov(
        anisotropic_state_msg(0.0, 0.0, seconds=9.95, vxx=0.04, vyy=0.0025, vxy=0.008)
    )
    assert R[0, 1] == pytest.approx(0.008)
    assert R[1, 0] == pytest.approx(0.008)
    assert R[0, 0] == pytest.approx(0.04)
    assert R[1, 1] == pytest.approx(0.0025)


def test_cross_covariance_changes_the_update_so_dropping_it_is_not_cosmetic():
    """An anisotropic R must steer the correction differently to its diagonal.

    If this ever stops holding, the covariance profile is not reaching the
    filter and the whole J R Jt propagation is decorative.
    """
    def belief_after(vxy):
        node = make_state_node(belief_xy=(0.0, 0.0), belief_cov=0.05)
        node._apply_state_correction(
            anisotropic_state_msg(0.10, 0.10, seconds=9.95,
                                  vxx=0.04, vyy=0.0025, vxy=vxy)
        )
        return node.belief_m[:2].copy()

    correlated = belief_after(0.008)
    diagonal = belief_after(0.0)
    assert not np.allclose(correlated, diagonal, atol=1e-6), (
        "the cross-covariance made no difference -- it is being dropped "
        "somewhere between /state/bev and the Kalman update"
    )


def test_non_positive_definite_covariance_falls_back_to_the_diagonal():
    node = make_state_node()
    # |vxy| > sqrt(vxx*vyy) -> indefinite.
    R = node._state_measurement_cov(
        anisotropic_state_msg(0.0, 0.0, seconds=9.95, vxx=0.01, vyy=0.01, vxy=0.5)
    )
    assert R[0, 1] == 0.0 and R[1, 0] == 0.0
    assert R[0, 0] == pytest.approx(0.01)
    assert np.min(np.linalg.eigvalsh(R)) > 0.0
    assert any('positive definite' in m for _, m in node._logger.messages)


def test_belief_bootstrap_preserves_cross_covariance():
    node = make_state_node()
    node.belief_m = None
    node.belief_S = None
    node.belief_stamp = None
    node._apply_state_correction(
        anisotropic_state_msg(1.0, 2.0, seconds=9.95, vxx=0.04, vyy=0.0025, vxy=0.008)
    )
    assert node.belief_S[0, 1] == pytest.approx(0.008)
    assert node.belief_S[1, 0] == pytest.approx(0.008)


def test_legacy_diagonal_covariance_still_works_unchanged():
    """The legacy constant-metric profile publishes no cross terms."""
    node = make_state_node()
    R = node._state_measurement_cov(state_msg(0.0, 0.0, seconds=9.95, var=0.0064))
    np.testing.assert_allclose(R, np.diag([0.0064, 0.0064]))


# --------------------------------------------------------------------------
# Unmodelled-error (inter-camera disagreement) inflation
# --------------------------------------------------------------------------






def test_coupled_mode_lets_a_position_fix_move_the_heading():
    """The opt-in mode keeps the posterior the update produced, heading included.

    ``camera_xy_only`` anchors heading to odometry and deletes the position-heading
    correlation at every correction, so heading uncertainty only ever grows. In ``coupled``
    the same correction is allowed to use that correlation, which is the whole point: driving
    with a heading error puts you sideways, so a position fix carries information about
    heading. Measured offline on five recorded drives, this takes the heading error from
    0.81-6.60 deg to 0.45-1.07 deg.
    """
    S_pred = np.array([
        [0.04, 0.0, 0.05],
        [0.0, 0.04, 0.0],
        [0.05, 0.0, 0.20],
    ])
    posterior = np.array([
        [0.01, 0.0, 0.02],
        [0.0, 0.01, 0.0],
        [0.02, 0.0, 0.09],
    ])
    def fresh():
        # the commit writes its result back into the outcome for the diagnostics, so each
        # node needs its own copy or the second one inherits the first one's heading
        return bc.CorrectionOutcome(
            reason=bc.RejectReason.ACCEPTED,
            m_pred=np.zeros(3),
            S_pred=S_pred.copy(),
            next_m=np.array([0.05, 0.0, 0.33]),
            next_S=posterior.copy(),
        )
    R = np.diag([1.0e-4, 1.0e-4])

    anchored = make_state_node(odom_yaw=0.25)
    anchored.heading_update_mode = 'camera_xy_only'
    anchored._commit_metric_correction_outcome(stamp(10.0), R, fresh())
    assert anchored.belief_m[2] == pytest.approx(0.25)        # odometry wins
    assert anchored.belief_S[0, 2] == 0.0                     # correlation deleted
    assert anchored.belief_S[2, 2] == pytest.approx(S_pred[2, 2])   # heading never sharpens

    coupled = make_state_node(odom_yaw=0.25)
    coupled.heading_update_mode = 'coupled'
    coupled._commit_metric_correction_outcome(stamp(10.0), R, fresh())
    assert coupled.belief_m[2] == pytest.approx(0.33)          # the fix moved heading
    assert coupled.belief_S[0, 2] != 0.0                       # correlation kept
    assert coupled.belief_S[2, 2] < S_pred[2, 2]               # heading actually sharpened
    assert np.min(np.linalg.eigvalsh(coupled.belief_S)) > 0.0  # still a covariance matrix


# --------------------------------------------------------------------------
# The heading prior must describe the heading it actually holds
# --------------------------------------------------------------------------

def test_bootstrap_heading_variance_is_the_odometry_drift_not_pi_squared():
    """The belief's heading MEAN is map-frame odometry, so its variance is odometry's.

    Calling it non-informative while using it as the mean is a contradiction, and with
    a coupled update it is a costly one: an xy measurement against a pi^2 heading prior
    swings the heading by more than 90 degrees on the first correction. On the frozen
    route that put the belief 112 degrees out and drove the robot into a rack 1.7 m in.
    """
    node = make_state_node(belief_stamp_s=9.9, now_s=10.0)
    node.belief_m = None
    node.belief_S = None
    node.belief_stamp = None
    node._odom_origin_stamp_s = 0.0          # odometry began at t=0
    node.process_noise_theta = 0.02          # rad per root second, as the campaign runs

    node._apply_state_correction(state_msg(1.0, 2.0, seconds=9.95))

    # 0.02^2 rad^2/s of drift over 9.95 s, and nowhere near pi^2
    assert node.belief_S[2, 2] == pytest.approx(0.02 ** 2 * 9.95, rel=1e-6)
    assert node.belief_S[2, 2] < 0.01
    assert node.belief_m[2] == pytest.approx(0.25)   # the map-frame odom heading


def test_heading_variance_falls_back_to_non_informative_without_odometry():
    """No odometry means no heading knowledge, and the belief must say so."""
    node = make_state_node()
    node.belief_m = None
    node.belief_S = None
    node.belief_stamp = None
    node._odom_origin_stamp_s = None

    node._apply_state_correction(state_msg(1.0, 2.0, seconds=9.95))

    assert node.belief_S[2, 2] == pytest.approx(math.pi ** 2)


def test_heading_variance_never_exceeds_knowing_nothing():
    node = make_state_node(belief_stamp_s=9.9, now_s=10.0)
    node.belief_m = None
    node.belief_S = None
    node.belief_stamp = None
    node._odom_origin_stamp_s = -1.0e7        # an absurdly long drive
    node.process_noise_theta = 0.02

    node._apply_state_correction(state_msg(1.0, 2.0, seconds=9.95))

    assert node.belief_S[2, 2] == pytest.approx(math.pi ** 2)


def _envelope(*, source_batch_id, correction_stamp, xy, var=0.03, frame_id='map_bev'):
    """One state-correction envelope, carrying batch identity and capture time."""
    message = String()
    message.data = json.dumps(dict(
        schema_version=1, frame_id=frame_id,
        source_batch_id=source_batch_id, correction_stamp=float(correction_stamp),
        xy=[float(xy[0]), float(xy[1])],
        covariance_m2=[[float(var), 0.], [0., float(var)]]))
    return message


def _envelope_node(*, now_s=10.0):
    from planning.planners.base_planner import UnicyclePlannerBase
    planner = object.__new__(UnicyclePlannerBase)
    planner.dt = .25
    planner.process_noise_xy = .01
    planner.process_noise_theta = .02
    planner.coherent_drift = False

    node = make_state_node(now_s=now_s)
    node.planner = planner
    node.heading_update_mode = 'coupled'
    node.belief_m = node.belief_S = node.belief_stamp = None
    node.require_state_correction_envelope = True
    node._seen_state_source_batch_ids = set()
    node.state_reanchor_m = 0.
    node.stale_belief_inflate_m2_per_s = 0.
    node.stale_belief_inflate_cap_m2 = 0.
    node._resolve_plan_frame_id = lambda: 'map_bev'
    node._fatal_experiment_stop = lambda *a: (_ for _ in ()).throw(AssertionError(a))
    return node


def test_envelope_mode_does_not_bootstrap_from_the_compatibility_pose():
    """The anonymous pose must not initialize the estimator under the envelope contract.

    The compatibility pose carries no source batch identity and no capture time.
    Bootstrapping from it initialized the belief from an anonymous measurement and
    recorded no identified event, so the envelope that followed was then judged
    'not newer than belief' and dropped -- the physical bootstrap disappeared from
    the ledger entirely.
    """
    node = _envelope_node()
    node._state_cb(state_msg(1., 2., seconds=9.95))
    assert node.belief_m is None

    m, S, meta = node._resolve_state_belief_ekf(stamp(10.))
    assert m is None and S is None
    assert meta['measurement_available'] is False
    assert node.belief_m is None
    assert not node.correction_assimilation_pub.published


def test_the_envelope_itself_bootstraps_and_records_one_identified_event():
    node = _envelope_node()
    node._state_cb(state_msg(1., 2., seconds=9.95))
    node._resolve_state_belief_ekf(stamp(10.))

    node._state_correction_envelope_cb(_envelope(
        source_batch_id='camera_A+camera_B@9950000000',
        correction_stamp=9.95, xy=(1., 2.)))

    assert node.belief_m is not None
    np.testing.assert_allclose(node.belief_m[:2], [1., 2.], atol=1e-9)
    events = [json.loads(p) for p in node.correction_assimilation_pub.published]
    assert len(events) == 1
    assert events[0]['status'] == 'accepted_bootstrap'
    assert events[0]['source_batch_id'] == 'camera_A+camera_B@9950000000'
    assert events[0]['correction_stamp'] == pytest.approx(9.95)


def test_the_reverse_delivery_order_gives_the_same_initialized_state():
    node = _envelope_node()
    node._state_correction_envelope_cb(_envelope(
        source_batch_id='camera_A+camera_B@9950000000',
        correction_stamp=9.95, xy=(1., 2.)))
    node._state_cb(state_msg(1., 2., seconds=9.95))
    node._resolve_state_belief_ekf(stamp(10.))

    np.testing.assert_allclose(node.belief_m[:2], [1., 2.], atol=1e-9)
    events = [json.loads(p) for p in node.correction_assimilation_pub.published]
    assert len([e for e in events if e['accepted']]) == 1


def test_an_invalid_envelope_plus_a_usable_pose_still_does_not_initialize():
    node = _envelope_node()
    stops = []
    node._fatal_experiment_stop = lambda *a: stops.append(a)
    node._state_cb(state_msg(1., 2., seconds=9.95))

    bad = String()
    bad.data = json.dumps(dict(schema_version=1, frame_id='map_bev',
                               source_batch_id='', correction_stamp=9.95))
    node._state_correction_envelope_cb(bad)
    node._resolve_state_belief_ekf(stamp(10.))

    assert node.belief_m is None
    assert not any(json.loads(p)['accepted']
                   for p in node.correction_assimilation_pub.published)


def test_legacy_non_envelope_mode_keeps_its_compatibility_bootstrap():
    node = _envelope_node()
    node.require_state_correction_envelope = False
    node._state_cb(state_msg(1., 2., seconds=9.95))
    node._resolve_state_belief_ekf(stamp(10.))
    np.testing.assert_allclose(node.belief_m[:2], [1., 2.], atol=1e-9)


@pytest.mark.parametrize(
    ("accepted_camera_ids", "expected"),
    [(["camera_A"], False), (["camera_A", "camera_B"], True)],
)
def test_schema2_envelope_only_allows_quorum_reanchor(
    accepted_camera_ids, expected
):
    """A lone camera may update normally but may not snap a lost belief."""
    from planning.nodes.unicycle_planner_node import (
        _correction_envelope_allows_reanchor,
    )

    assert _correction_envelope_allows_reanchor(
        {"schema_version": 2, "accepted_camera_ids": accepted_camera_ids}
    ) is expected


def test_schema1_envelope_retains_legacy_reanchor_semantics():
    from planning.nodes.unicycle_planner_node import (
        _correction_envelope_allows_reanchor,
    )

    assert _correction_envelope_allows_reanchor({"schema_version": 1}) is True


def test_odometry_heading_variance_follows_the_encoder_model():
    """Heading variance accumulates sigma_w^2(v, w) along the odometry, not q^2 * t."""
    from planning.core.encoder_noise_model import ENCODER_PSD, encoder_psd
    node = make_state_node()
    node.planner.process_noise_psd = dict(ENCODER_PSD)
    node._odom_origin_stamp_s = 0.0
    node._odom_heading_var_log = None
    for k in range(501):                    # 10 s at 50 Hz: 5 s straight, 5 s turning
        t = 0.02 * k
        node._accumulate_odometry_heading_variance(t, 0.8, 0.0 if t <= 5.0 + 1e-9 else 0.5)
    straight = encoder_psd(0.8, 0.0)[1] * 5.0
    turning = encoder_psd(0.8, 0.5)[1] * 5.0
    assert node._map_frame_heading_variance(stamp(5.0)) == pytest.approx(
        max(np.radians(0.5) ** 2, straight), rel=1e-3)
    assert node._map_frame_heading_variance(stamp(10.0)) == pytest.approx(straight + turning, rel=1e-3)
