"""Regression harness for the shared belief-correction chain.

Step 1 of the single-cam/multicam consolidation extracted paper-1's gate chain
out of ``unicycle_planner_node`` into ``planning.core.belief_correction``. That
path backs **locked** evidence (``honest_campaign_v1``, never rerun), so the
extraction has to be behaviour-preserving, not merely plausible.

Two kinds of coverage here:

- ``test_golden_trace_*`` replays every correction the locked campaign logged
  through :func:`evaluate_gates` and asserts the verdict is bit-identical to
  what the pre-refactor node recorded. This is the characterization test: it
  fails if the extraction changed a single decision.
- the unit tests pin the arithmetic and gate ordering against verbatim copies of
  the pre-refactor formulas (the ``legacy_*`` oracles below), including the
  closed-form map-xy update the multicam path used -- proving the one shared
  ``compute_update`` reproduces *both* stacks.

Only ``pixel_corr_*`` diagnostic columns are read from the campaign CSVs; no position
columns, so the column-choice hazards of the campaign logs do not apply here.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from planning.core import belief_correction as bc


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCKED_CAMPAIGN = REPO_ROOT / "logs" / "visibility_comparison" / "honest_campaign_v1"

# From warehouse_visibility_campaign_honest_v1.yaml (the locked campaign config).
# run_manifest.json records pixel_correction_nis_threshold and pixel_timeout_s
# but not the jump limit, so it is pinned from the config here.
LOCKED_MAX_JUMP_M = 0.5
LOCKED_PIXEL_TIMEOUT_S = 1.25
LOCKED_DT_S = 0.25

PRE_UPDATE_REASONS = {
    bc.RejectReason.STALE_AGE.value,
    bc.RejectReason.DT_IMPLAUSIBLE.value,
    bc.RejectReason.MISSING_SNAPSHOT.value,
    bc.RejectReason.UPDATE_FAILED.value,
}


def _float(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


# --------------------------------------------------------------------------
# Verbatim pre-refactor implementations, kept as oracles.
# --------------------------------------------------------------------------

def legacy_pixel_uv_update(m_pred, S_eff, meas, mu_y, Sigma_y, Gamma, gain_scale, cov_eig_floor):
    """``_compute_pixel_uv_update`` as it stood before the extraction."""
    mu_y = np.asarray(mu_y, dtype=float).reshape(-1)
    meas = np.asarray(meas, dtype=float).reshape(-1)
    Sigma_y = np.asarray(Sigma_y, dtype=float)
    Gamma = np.asarray(Gamma, dtype=float)
    innov = meas - mu_y
    Sigma_y = (Sigma_y + Sigma_y.T) / 2.0
    Sigma_inv = np.linalg.pinv(Sigma_y)
    K = Gamma @ Sigma_inv
    next_m = m_pred + gain_scale * (K @ innov)
    next_S = S_eff - gain_scale * (Gamma @ Sigma_inv @ Gamma.T)
    next_S = 0.5 * (next_S + next_S.T)
    eig_min = np.min(np.linalg.eigvalsh(next_S))
    if eig_min < cov_eig_floor:
        w, v = np.linalg.eigh(next_S)
        next_S = (v * np.maximum(w, cov_eig_floor)) @ v.T
    return next_m, next_S, innov, K


def legacy_state_update(m_pred, S_pred, z, R):
    """``_apply_state_correction``'s closed-form map-xy update (multicam path)."""
    m_pred = np.asarray(m_pred, dtype=float)
    S_pred = np.asarray(S_pred, dtype=float)
    innov = np.asarray(z, dtype=float) - m_pred[:2]
    S_y = S_pred[:2, :2] + np.asarray(R, dtype=float)
    S_y_inv = np.linalg.inv(S_y)
    K = S_pred[:, :2] @ S_y_inv
    m_upd = m_pred + K @ innov
    S_upd = S_pred - K @ S_pred[:2, :]
    return m_upd, S_upd, innov, K


def legacy_regularize(S, min_state_cov):
    """``_regularize_state_covariance`` as it stood before the extraction."""
    S = np.asarray(S, dtype=float).copy()
    if min_state_cov > 0.0:
        for i in range(min(3, S.shape[0])):
            if S[i, i] < min_state_cov:
                S[i, i] = min_state_cov
    return (S + S.T) / 2.0


# --------------------------------------------------------------------------
# Wire contract
# --------------------------------------------------------------------------

def test_reject_codes_are_frozen():
    """Campaign CSVs and experiment_logger decode these integers.

    Codes 1-6 are load-bearing for every already-recorded campaign: they must
    never be renumbered or reused. New reasons append above 6.
    """
    assert {k: v for k, v in bc.REJECT_CODES.items() if v <= 6.0} == {
        'stale_age': 1.0,
        'dt_implausible': 2.0,
        'missing_snapshot': 3.0,
        'update_failed': 4.0,
        'jump_too_large': 5.0,
        'nis_too_large': 6.0,
    }
    # Codes are unique and none collides with accepted (0) or unknown (99).
    codes = list(bc.REJECT_CODES.values())
    assert len(codes) == len(set(codes))
    assert bc.ACCEPTED_CODE not in codes and bc.UNKNOWN_REJECT_CODE not in codes
    # Every enum member except ACCEPTED has a code, so none can silently
    # decode as 'unknown' in a CSV.
    assert {r.value for r in bc.RejectReason if r is not bc.RejectReason.ACCEPTED} == set(
        bc.REJECT_CODES
    )
    assert bc.reject_code(bc.RejectReason.ACCEPTED) == 0.0
    assert bc.reject_code('callback') == 99.0
    assert bc.reject_code('') == 99.0
    assert bc.reject_code(None) == 99.0
    # Accepts the enum and the bare string the node used to pass.
    for reason, code in bc.REJECT_CODES.items():
        assert bc.reject_code(reason) == code
        assert bc.reject_code(bc.RejectReason(reason)) == code


# --------------------------------------------------------------------------
# Gate chain
# --------------------------------------------------------------------------

@pytest.fixture
def gates():
    return bc.CorrectionGates(
        pixel_timeout_s=LOCKED_PIXEL_TIMEOUT_S,
        dt_nominal_s=LOCKED_DT_S,
        skip_stale=True,
        max_jump_m=LOCKED_MAX_JUMP_M,
        nis_threshold=9.21,
    )


def test_gate_thresholds_match_paper1(gates):
    assert gates.future_tolerance_s == pytest.approx(1.25)
    assert gates.max_dt_s == pytest.approx(2.5)          # max(2*1.25, 4*0.25, 0.5)
    assert not gates.age_is_invalid(1.25)                # boundary is inclusive-pass
    assert gates.age_is_invalid(1.26)
    assert not gates.age_is_invalid(-1.25)
    assert gates.age_is_invalid(-1.26)
    assert not gates.dt_is_implausible(2.5)
    assert gates.dt_is_implausible(2.51)
    assert not gates.jump_is_too_large(0.5)
    assert gates.jump_is_too_large(0.51)
    assert not gates.nis_is_too_large(9.21)
    assert gates.nis_is_too_large(9.22)


def test_gate_order_is_first_wins(gates):
    """Every gate simultaneously tripped -> the earliest reason, as before."""
    assert evaluate_all(gates, age=99.0, dt_s=99.0, jump=99.0, nis=99.0) is bc.RejectReason.STALE_AGE
    assert evaluate_all(gates, dt_s=99.0, jump=99.0, nis=99.0) is bc.RejectReason.DT_IMPLAUSIBLE
    assert evaluate_all(gates, jump=99.0, nis=99.0) is bc.RejectReason.JUMP_TOO_LARGE
    assert evaluate_all(gates, nis=99.0) is bc.RejectReason.NIS_TOO_LARGE
    assert evaluate_all(gates) is bc.RejectReason.ACCEPTED


def evaluate_all(gates, *, age=0.0, dt_s=0.1, jump=0.0, nis=0.0):
    return bc.evaluate_gates(gates, age=age, dt_s=dt_s, xy_update_norm_m=jump, nis=nis)


def test_snapshot_and_update_failures_slot_between_dt_and_jump(gates):
    assert bc.evaluate_gates(gates, snapshot_missing=True) is bc.RejectReason.MISSING_SNAPSHOT
    assert bc.evaluate_gates(gates, update_failed=True) is bc.RejectReason.UPDATE_FAILED
    assert bc.evaluate_gates(
        gates, snapshot_missing=True, update_failed=True
    ) is bc.RejectReason.MISSING_SNAPSHOT
    # dt still outranks both
    assert bc.evaluate_gates(
        gates, dt_s=99.0, snapshot_missing=True
    ) is bc.RejectReason.DT_IMPLAUSIBLE


def test_incremental_evaluation_matches_one_shot(gates):
    """apply_correction calls evaluate_gates stagewise with NaN placeholders.

    That is only sound if unknown-as-NaN can never trip a gate, so partial
    evaluation agrees with the full-record evaluation.
    """
    cases = [
        dict(age=0.1, dt_s=0.2, xy_update_norm_m=0.05, nis=1.0),
        dict(age=9.0, dt_s=0.2, xy_update_norm_m=0.05, nis=1.0),
        dict(age=0.1, dt_s=9.0, xy_update_norm_m=0.05, nis=1.0),
        dict(age=0.1, dt_s=0.2, xy_update_norm_m=5.0, nis=1.0),
        dict(age=0.1, dt_s=0.2, xy_update_norm_m=0.05, nis=50.0),
    ]
    for case in cases:
        one_shot = bc.evaluate_gates(gates, **case)
        staged = bc.evaluate_gates(gates, age=case['age'])
        if staged is bc.RejectReason.ACCEPTED:
            staged = bc.evaluate_gates(gates, age=case['age'], dt_s=case['dt_s'])
        if staged is bc.RejectReason.ACCEPTED:
            staged = bc.evaluate_gates(gates, **case)
        assert staged is one_shot, case


def test_nan_inputs_never_trip_a_gate(gates):
    assert bc.evaluate_gates(gates) is bc.RejectReason.ACCEPTED
    assert bc.evaluate_gates(
        gates, age=math.nan, dt_s=math.nan, xy_update_norm_m=math.nan, nis=math.nan
    ) is bc.RejectReason.ACCEPTED
    # NIS specifically guarded on isfinite in the original code.
    assert not gates.nis_is_too_large(math.inf)


def test_gates_disabled_by_nonpositive_thresholds():
    off = bc.CorrectionGates(max_jump_m=0.0, nis_threshold=0.0, skip_stale=False)
    assert not off.jump_is_too_large(1e6)
    assert not off.nis_is_too_large(1e6)
    assert not off.age_is_invalid(1e6)


# --------------------------------------------------------------------------
# Shared Kalman update reproduces BOTH stacks' arithmetic
# --------------------------------------------------------------------------

def test_compute_update_matches_legacy_pixel_formula():
    rng = np.random.default_rng(0)
    for _ in range(25):
        m_pred = rng.normal(size=3)
        A = rng.normal(size=(3, 3))
        S_eff = A @ A.T + 0.1 * np.eye(3)
        Gamma = rng.normal(size=(3, 2))
        B = rng.normal(size=(2, 2))
        Sigma_y = B @ B.T + 0.5 * np.eye(2)
        meas = rng.normal(size=2)
        mu_y = rng.normal(size=2)
        # Unit gain preserves the historical optimal-Kalman arithmetic. Scaled
        # gains are checked against Joseph form separately, not the old defect.
        gain_scale = 1.0

        lin = bc.Linearization(
            z=meas, mu_y=mu_y, Gamma=Gamma, Sigma_y=Sigma_y,
            R_eff=np.eye(2), S_eff=S_eff, gain_scale=gain_scale,
        )
        got = bc.compute_update(m_pred, lin, cov_eig_floor=1e-9)
        exp_m, exp_S, exp_innov, exp_K = legacy_pixel_uv_update(
            m_pred, S_eff, meas, mu_y, Sigma_y, Gamma, gain_scale, 1e-9
        )
        # The shared path wraps theta; the legacy path did too, on the same line.
        exp_m[2] = math.atan2(math.sin(exp_m[2]), math.cos(exp_m[2]))
        np.testing.assert_allclose(got.next_m, exp_m, rtol=0, atol=1e-12)
        np.testing.assert_allclose(got.next_S, exp_S, rtol=0, atol=1e-12)
        np.testing.assert_allclose(got.innov, exp_innov, rtol=0, atol=1e-12)
        np.testing.assert_allclose(got.K, exp_K, rtol=0, atol=1e-12)


def test_compute_update_matches_legacy_map_xy_formula():
    """The fused map-xy linearization reduces to the multicam closed form.

    This is what makes step 2 (pointing the multicam path at the shared chain)
    a rewiring rather than a numerics change.
    """
    rng = np.random.default_rng(1)
    for _ in range(25):
        m_pred = rng.normal(size=3)
        A = rng.normal(size=(3, 3))
        S_pred = A @ A.T + 0.1 * np.eye(3)
        z = m_pred[:2] + rng.normal(scale=0.05, size=2)
        R = np.diag(rng.uniform(0.01, 0.2, size=2))

        snapshot = bc.CorrectionSnapshot(
            belief_m=m_pred, belief_S=S_pred, belief_stamp=None,
            cmd=np.zeros(2), meas=z,
        )
        source = bc.FusedMapMeasurementSource(
            snapshot_fn=lambda: snapshot, measurement_cov_fn=lambda: R
        )
        lin = source.linearize(m_pred, S_pred, snapshot)
        got = bc.compute_update(m_pred, lin, cov_eig_floor=1e-12)
        exp_m, exp_S, exp_innov, exp_K = legacy_state_update(m_pred, S_pred, z, R)

        np.testing.assert_allclose(got.next_m[:2], exp_m[:2], rtol=0, atol=1e-10)
        np.testing.assert_allclose(got.next_S, exp_S, rtol=0, atol=1e-10)
        np.testing.assert_allclose(got.innov, exp_innov, rtol=0, atol=1e-12)
        np.testing.assert_allclose(got.K, exp_K, rtol=0, atol=1e-10)


def test_compute_update_rejects_shape_mismatch_and_reports():
    messages = []
    lin = bc.Linearization(
        z=np.zeros(3), mu_y=np.zeros(2), Gamma=np.zeros((3, 2)),
        Sigma_y=np.eye(2), R_eff=np.eye(2), S_eff=np.eye(3),
    )
    assert bc.compute_update(
        np.zeros(3), lin, cov_eig_floor=1e-9, on_shape_error=messages.append
    ) is None
    assert len(messages) == 1 and 'shape mismatch' in messages[0]

    bad_cov = bc.Linearization(
        z=np.zeros(2), mu_y=np.zeros(2), Gamma=np.zeros((3, 2)),
        Sigma_y=np.eye(3), R_eff=np.eye(2), S_eff=np.eye(3),
    )
    assert bc.compute_update(
        np.zeros(3), bad_cov, cov_eig_floor=1e-9, on_shape_error=messages.append
    ) is None
    assert 'covariance shape mismatch' in messages[1]


def test_nis_matches_legacy_definition():
    innov = np.array([0.3, -0.4])
    S_y = np.array([[0.02, 0.001], [0.001, 0.03]])
    expected = float(innov @ np.linalg.inv(S_y) @ innov)
    assert bc.normalized_innovation_squared(innov, S_y) == pytest.approx(expected)
    assert math.isnan(bc.normalized_innovation_squared(innov, None))
    assert math.isnan(bc.normalized_innovation_squared(innov, np.zeros((2, 2))))


def test_regularize_covariance_matches_legacy():
    rng = np.random.default_rng(2)
    for min_cov in (0.0, 1e-6, 0.05):
        for _ in range(10):
            A = rng.normal(size=(3, 3))
            S = A @ A.T * 1e-3
            np.testing.assert_allclose(
                bc.regularize_covariance(S, min_cov),
                legacy_regularize(S, min_cov),
                rtol=0, atol=0,
            )


def test_yaw_report_does_not_modify_the_estimate():
    m_pred = np.array([1.0, 2.0, 0.5])
    next_m = np.array([1.1, 2.1, 0.7])
    next_S = np.eye(3)
    info = bc.yaw_report(next_m, next_S, m_pred)
    assert info['next_m'] is next_m
    assert info['next_S'] is next_S
    assert info['yaw_correction_applied'] is False
    assert info['theta_update_from_uv_rad'] == pytest.approx(0.2)
    assert info['theta_update_total_rad'] == pytest.approx(0.2)
    assert math.isnan(info['innov_theta'])
    assert math.isnan(info['k_theta_theta'])


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------

class _StubSource:
    label = 'stub'

    def __init__(self, snapshot, lin):
        self._snapshot = snapshot
        self._lin = lin
        self.linearize_calls = 0

    def snapshot(self):
        return self._snapshot

    def linearize(self, m_pred, S_pred, snapshot):
        self.linearize_calls += 1
        return self._lin


def _identity_replay(m, S, from_stamp, to_stamp, cmd, dt):
    return np.asarray(m, dtype=float), np.asarray(S, dtype=float), {
        'cmd_replay_count': 0.0,
        'cmd_replay_duration_s': float(dt),
        'cmd_replay_used_fallback': 0.0,
        'motion_replay_source_code': 1.0,
    }


def _stub_pair(*, z_offset, S_scale=1.0, R=0.01):
    m_pred = np.array([0.0, 0.0, 0.0])
    S_pred = np.eye(3) * S_scale
    z = np.array([z_offset, 0.0])
    snapshot = bc.CorrectionSnapshot(
        belief_m=m_pred, belief_S=S_pred, belief_stamp=None,
        cmd=np.zeros(2), meas=z, meas_stamp=None,
    )
    lin = bc.Linearization(
        z=z, mu_y=m_pred[:2].copy(), Gamma=S_pred[:, :2].copy(),
        Sigma_y=S_pred[:2, :2] + np.eye(2) * R, R_eff=np.eye(2) * R,
        S_eff=S_pred.copy(), gain_scale=1.0,
    )
    return snapshot, lin


def test_apply_correction_accepts_and_carries_diagnostics(gates):
    snapshot, lin = _stub_pair(z_offset=0.05)
    source = _StubSource(snapshot, lin)
    outcome = bc.apply_correction(
        source=source, gates=gates, replay=_identity_replay, age=0.1, dt_s=0.2
    )
    assert outcome.accepted
    assert outcome.reject_code == 0.0
    assert outcome.next_m is not None and outcome.next_S is not None
    assert outcome.xy_update_norm_m > 0.0
    assert math.isfinite(outcome.nis)
    assert outcome.replay_meta['motion_replay_source_code'] == 1.0
    assert outcome.yaw_info['yaw_correction_applied'] is False


def test_apply_correction_short_circuits_before_touching_the_source(gates):
    snapshot, lin = _stub_pair(z_offset=0.05)
    source = _StubSource(snapshot, lin)
    outcome = bc.apply_correction(
        source=source, gates=gates, replay=_identity_replay, age=99.0, dt_s=0.2
    )
    assert outcome.reason is bc.RejectReason.STALE_AGE
    assert outcome.reject_code == 1.0
    assert source.linearize_calls == 0
    assert outcome.next_m is None


def test_apply_correction_jump_limiter_fires_and_withholds_the_commit(gates):
    # Huge innovation, loose prior -> gain ~1 -> update far exceeds 0.5 m.
    snapshot, lin = _stub_pair(z_offset=5.0, S_scale=10.0, R=0.01)
    outcome = bc.apply_correction(
        source=_StubSource(snapshot, lin), gates=gates,
        replay=_identity_replay, age=0.1, dt_s=0.2,
    )
    assert outcome.reason is bc.RejectReason.JUMP_TOO_LARGE
    assert outcome.reject_code == 5.0
    assert outcome.next_m is None      # caller must not commit
    assert outcome.next_S is None
    # Diagnostics still populated so the rejection is visible in the CSV.
    assert outcome.xy_update_norm_m > gates.max_jump_m
    assert math.isfinite(outcome.nis)
    assert outcome.m_pred is not None and outcome.innov is not None


def test_apply_correction_missing_snapshot(gates):
    class _NoSnapshot:
        label = 'none'

        def snapshot(self):
            return None

        def linearize(self, m_pred, S_pred, snapshot):  # pragma: no cover
            raise AssertionError("must not linearize without a snapshot")

    outcome = bc.apply_correction(
        source=_NoSnapshot(), gates=gates, replay=_identity_replay, age=0.1, dt_s=0.2
    )
    assert outcome.reason is bc.RejectReason.MISSING_SNAPSHOT
    assert outcome.reject_code == 3.0


def test_apply_correction_update_failed_still_carries_r_eff(gates):
    """The node's old path reported R_eff on update_failed -- keep that."""
    snapshot, lin = _stub_pair(z_offset=0.05)
    lin.mu_y = np.zeros(3)          # force a shape mismatch inside compute_update
    outcome = bc.apply_correction(
        source=_StubSource(snapshot, lin), gates=gates,
        replay=_identity_replay, age=0.1, dt_s=0.2,
    )
    assert outcome.reason is bc.RejectReason.UPDATE_FAILED
    assert outcome.reject_code == 4.0
    assert outcome.R_eff is not None
    assert outcome.m_pred is not None


# --------------------------------------------------------------------------
# Golden trace against the locked campaign
# --------------------------------------------------------------------------

def _locked_corrections():
    """Unique logged corrections from honest_campaign_v1, deduped by apply stamp.

    The logger latches the last diagnostics message, so consecutive CSV rows
    repeat the same correction.
    """
    records = []
    for csv_path in sorted(LOCKED_CAMPAIGN.glob("*/*/*/*/experiment.csv")):
        seen = set()
        with csv_path.open() as handle:
            for row in csv.DictReader(handle):
                key = row.get('pixel_corr_apply_stamp', '')
                if not key or key == 'nan' or key in seen:
                    continue
                seen.add(key)
                records.append((csv_path, row))
    return records


LOCKED_RECORDS = _locked_corrections() if LOCKED_CAMPAIGN.is_dir() else []

requires_locked_campaign = pytest.mark.skipif(
    not LOCKED_RECORDS,
    reason=f"locked campaign not present at {LOCKED_CAMPAIGN} (logs/ is gitignored)",
)


@requires_locked_campaign
def test_golden_trace_dataset_is_the_expected_shape():
    """Guard the guard: the trace must not silently shrink to nothing."""
    assert len(LOCKED_RECORDS) > 5000
    reasons = {row['pixel_corr_reject_reason'] for _, row in LOCKED_RECORDS}
    assert 'accepted' in reasons
    # At least one real rejection, or the gate chain is untested by this trace.
    assert reasons - {'accepted'}


@requires_locked_campaign
def test_golden_trace_verdicts_are_unchanged():
    """Replay every locked correction; the shared chain must agree exactly."""
    gates = bc.CorrectionGates(
        pixel_timeout_s=LOCKED_PIXEL_TIMEOUT_S,
        dt_nominal_s=LOCKED_DT_S,
        skip_stale=True,
        max_jump_m=LOCKED_MAX_JUMP_M,
        nis_threshold=9.21,
    )

    mismatches = []
    replayed = 0
    pre_update = 0
    for csv_path, row in LOCKED_RECORDS:
        logged_reason = row['pixel_corr_reject_reason']
        jump = _float(row, 'pixel_corr_xy_update_norm_m')
        if not math.isfinite(jump):
            # Rejected before the update ran; its inputs (age, dt) are not
            # logged, so the verdict is not reconstructible from the CSV.
            assert logged_reason in PRE_UPDATE_REASONS, (csv_path, logged_reason)
            pre_update += 1
            continue

        nis = _float(row, 'pixel_corr_nis')
        threshold = _float(row, 'pixel_corr_nis_threshold')
        assert threshold == pytest.approx(gates.nis_threshold), csv_path

        got = bc.evaluate_gates(gates, age=0.0, dt_s=0.0, xy_update_norm_m=jump, nis=nis)
        replayed += 1
        if got.value != logged_reason:
            mismatches.append((str(csv_path), logged_reason, got.value, jump, nis))
        # The wire code must round-trip too, not just the name.
        logged_code = _float(row, 'pixel_corr_reject_reason_code')
        if math.isfinite(logged_code) and got.value == logged_reason:
            assert bc.reject_code(got) == logged_code, (csv_path, logged_reason)

    assert replayed > 5000, f"only replayed {replayed} corrections"
    assert not mismatches, (
        f"{len(mismatches)}/{replayed} verdicts changed; first 5: {mismatches[:5]}"
    )
    print(f"\ngolden trace: {replayed} corrections replayed, "
          f"{pre_update} pre-update rejections skipped (inputs not logged)")


@requires_locked_campaign
def test_golden_trace_exercises_the_nis_gate():
    """Document what the locked trace actually covers, so gaps stay visible.

    honest_campaign_v1 contains a genuine NIS rejection but never trips the
    0.5 m jump limiter (max observed update 0.205 m) -- the jump gate is
    covered by unit tests only.
    """
    jumps = [_float(row, 'pixel_corr_xy_update_norm_m') for _, row in LOCKED_RECORDS]
    jumps = [j for j in jumps if math.isfinite(j)]
    reasons = [row['pixel_corr_reject_reason'] for _, row in LOCKED_RECORDS]

    assert max(jumps) < LOCKED_MAX_JUMP_M, "trace now covers the jump gate -- update this note"
    assert reasons.count('nis_too_large') >= 1
    assert 'jump_too_large' not in reasons
