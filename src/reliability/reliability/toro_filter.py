"""Toro-style constant-velocity Kalman baseline B2 (plan 01 / pre-reg §3).

The faithful comparison method of Toro Diz et al.: a **constant-velocity**
Kalman filter (state ``[x, y, vx, vy]``, no odometry) whose per-measurement
covariance is the **nearest static calibration point** (``ToroCovarianceModel``),
with a **validated-FOV gate** and axis-specific 2x2 measurement covariance.

This is deliberately a *separate* estimator from :func:`reliability.replay.run_replay`
(which is a position-only, odometry-driven filter). Keeping Toro's own CV/no-odom
temporal model distinct is the point of pre-registration item B2a/B2b: it isolates
whether any gain over Toro comes from reliability modelling or merely from a
different temporal handling. Results are scored through the SAME
:func:`reliability.replay.compute_replay_metrics` so B2 is paired-comparable with
every other method (identical ATE/NIS/NEES definitions on identical eval frames).

Two temporal variants (pre-reg §3):

* ``b2b_sequential`` (default) — each observation is applied at its own timestamp
  with a CV predict between updates (timestamped sequential updates).
* ``b2a_simultaneous`` — observations sharing a frame are treated as one
  time-binned batch (Toro's 0.08 s binning), predicted once to the frame time
  then updated in sequence at dt = 0.

Pure library: no ROS, no numpy. ``q`` (the CV white-noise-acceleration intensity)
is tuned on validation data ONLY and shared across all conditions (plan 01 gate:
no per-method process noise).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

from reliability.contracts import ContractValidationError
from reliability.replay import (
    EvaluationFrame,
    ReplayFrame,
    ReplayMetrics,
    ReplayStep,
    compute_replay_metrics,
)
from reliability.toro_baseline import ToroCovarianceModel, constant_velocity_predict

_BINNINGS = ("b2a_simultaneous", "b2b_sequential")


@dataclass(frozen=True)
class ToroFilterConfig:
    """Configuration for :func:`run_toro_filter`.

    ``q`` is the shared CV process intensity (validation-tuned, never per-method).
    ``fov_gate`` reproduces Toro's rejection of measurements outside the validated
    field of view. ``nis_gate`` is off by default because Toro's own gate is the
    FOV hull, not an innovation test — enable it only to study the interaction.
    """

    q: float = 1.0
    initial_position_std_m: float = 0.5
    initial_velocity_std_ms: float = 1.0
    binning: str = "b2b_sequential"
    fov_gate: bool = True
    nis_gate: float | None = None
    max_error_divergence_m: float = 1.0

    def __post_init__(self) -> None:
        for name in ("q", "initial_position_std_m", "initial_velocity_std_ms"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ContractValidationError(f"{name} must be finite and non-negative")
        if self.binning not in _BINNINGS:
            raise ContractValidationError(
                f"unknown binning {self.binning!r}; choices are {_BINNINGS}"
            )
        if self.nis_gate is not None:
            gate = float(self.nis_gate)
            if not math.isfinite(gate) or gate <= 0.0:
                raise ContractValidationError("nis_gate must be finite and positive when set")


@dataclass(frozen=True)
class ToroFilterResult:
    config: ToroFilterConfig
    steps: tuple[ReplayStep, ...]
    metrics: ReplayMetrics | None = None
    accepted_total: int = 0
    rejected_total: int = 0


def run_toro_filter(
    frames: Sequence[ReplayFrame],
    config: ToroFilterConfig,
    model: ToroCovarianceModel,
    *,
    evaluation_frames: Sequence[EvaluationFrame] | None = None,
) -> ToroFilterResult:
    """Run the Toro-style CV Kalman baseline over recorded replay frames.

    One :class:`ReplayStep` is emitted per frame (at ``frame.timestamp_s``) so
    steps align to the same evaluation frames as every other method. The step
    covariance is the position block (top-left 2x2) of the CV state covariance.
    """

    if not isinstance(model, ToroCovarianceModel):
        raise ContractValidationError("model must be a ToroCovarianceModel")
    if not frames:
        return ToroFilterResult(config=config, steps=tuple(), metrics=None)

    ordered = sorted(frames, key=lambda frame: frame.timestamp_s)
    p0 = float(config.initial_position_std_m) ** 2
    v0 = float(config.initial_velocity_std_ms) ** 2
    state = (ordered[0].odometry_xy_m[0], ordered[0].odometry_xy_m[1], 0.0, 0.0)
    cov = (
        (max(p0, 1.0e-9), 0.0, 0.0, 0.0),
        (0.0, max(p0, 1.0e-9), 0.0, 0.0),
        (0.0, 0.0, max(v0, 1.0e-9), 0.0),
        (0.0, 0.0, 0.0, max(v0, 1.0e-9)),
    )
    filter_time = ordered[0].timestamp_s
    steps: list[ReplayStep] = []
    accepted_total = 0
    rejected_total = 0
    camera_ids = set(model.camera_ids)

    for frame in ordered:
        accepted: list[str] = []
        rejected: list[str] = []
        nis_by_camera: dict[str, float] = {}

        if config.binning == "b2a_simultaneous":
            state, cov, filter_time = _predict_to(state, cov, filter_time, frame.timestamp_s, config.q)
            observations = list(frame.observations)
        else:  # b2b_sequential
            observations = sorted(frame.observations, key=lambda obs: obs.timestamp_s)

        for obs in observations:
            if config.binning == "b2b_sequential":
                state, cov, filter_time = _predict_to(
                    state, cov, filter_time, obs.timestamp_s, config.q
                )
            outcome, state, cov, nis = _toro_update(
                state, cov, obs, model, camera_ids, config
            )
            if outcome == "accept":
                accepted.append(obs.camera_id)
                nis_by_camera[obs.camera_id] = nis
            else:
                rejected.append(obs.camera_id)
                nis_by_camera[obs.camera_id] = nis

        if config.binning == "b2b_sequential":
            state, cov, filter_time = _predict_to(
                state, cov, filter_time, frame.timestamp_s, config.q
            )

        accepted_total += len(accepted)
        rejected_total += len(rejected)
        steps.append(
            ReplayStep(
                timestamp_s=frame.timestamp_s,
                mean_xy_m=(state[0], state[1]),
                covariance_m2=((cov[0][0], cov[0][1]), (cov[1][0], cov[1][1])),
                accepted_camera_ids=tuple(accepted),
                rejected_camera_ids=tuple(rejected),
                nis_by_camera=nis_by_camera,
            )
        )

    metrics = None
    if evaluation_frames:
        metrics = compute_replay_metrics(
            steps,
            evaluation_frames,
            accepted_total=accepted_total,
            rejected_total=rejected_total,
            max_error_divergence_m=config.max_error_divergence_m,
        )
    return ToroFilterResult(
        config=config,
        steps=tuple(steps),
        metrics=metrics,
        accepted_total=accepted_total,
        rejected_total=rejected_total,
    )


def _predict_to(state, cov, filter_time, target_time, q):
    dt = float(target_time) - float(filter_time)
    if dt <= 0.0:
        return state, cov, float(filter_time) if dt < 0.0 else float(target_time)
    state, cov = constant_velocity_predict(state, cov, dt, q)
    return state, cov, float(target_time)


def _toro_update(state, cov, obs, model, camera_ids, config):
    """Return (outcome, state, cov, nis). outcome in {'accept', 'reject'}.

    Reject when the camera has no calibration support, when FOV gating fails, or
    when the optional NIS gate trips. The measurement covariance is the nearest
    static calibration point's 2x2 (Toro nearest-point rule, queried at the
    measured position — no ground truth).
    """

    z = obs.xy_m
    if obs.camera_id not in camera_ids:
        return "reject", state, cov, math.nan
    if config.fov_gate and not model.in_validated_fov(obs.camera_id, z):
        return "reject", state, cov, math.nan

    r_mat = model.covariance_for(obs.camera_id, z)
    innovation = (z[0] - state[0], z[1] - state[1])
    # S = H P H^T + R = P[:2,:2] + R
    s_mat = (
        (cov[0][0] + r_mat[0][0], cov[0][1] + r_mat[0][1]),
        (cov[1][0] + r_mat[1][0], cov[1][1] + r_mat[1][1]),
    )
    s_inv = _inv_2x2(s_mat)
    nis = _quad2(innovation, s_inv)
    if config.nis_gate is not None and math.isfinite(nis) and nis > float(config.nis_gate):
        return "reject", state, cov, nis

    state_post, cov_post = _joseph_position_update(state, cov, innovation, s_inv, r_mat)
    return "accept", state_post, cov_post, nis


# ---------------------------------------------------------------------------
# Kalman update with H = position selection ([I2 | 0]); Joseph form for PSD.
# ---------------------------------------------------------------------------


def _joseph_position_update(state, cov, innovation, s_inv, r_mat):
    # K = P H^T S^-1, H^T selects columns 0,1 so P H^T = first two columns of P.
    p_ht = [[cov[i][0], cov[i][1]] for i in range(4)]  # 4x2
    k = [
        [
            p_ht[i][0] * s_inv[0][0] + p_ht[i][1] * s_inv[1][0],
            p_ht[i][0] * s_inv[0][1] + p_ht[i][1] * s_inv[1][1],
        ]
        for i in range(4)
    ]  # 4x2
    new_state = tuple(state[i] + k[i][0] * innovation[0] + k[i][1] * innovation[1] for i in range(4))

    # A = I - K H  (4x4): K occupies columns 0,1.
    a = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    for i in range(4):
        a[i][0] -= k[i][0]
        a[i][1] -= k[i][1]

    # P+ = A P A^T + K R K^T
    ap = _mm(a, [list(row) for row in cov])          # 4x4
    apat = _mm(ap, _transpose(a))                     # 4x4
    kr = [[k[i][0] * r_mat[0][0] + k[i][1] * r_mat[1][0],
           k[i][0] * r_mat[0][1] + k[i][1] * r_mat[1][1]] for i in range(4)]  # 4x2
    krkt = [[kr[i][0] * k[j][0] + kr[i][1] * k[j][1] for j in range(4)] for i in range(4)]  # 4x4
    new_cov = tuple(
        tuple(0.5 * ((apat[i][j] + krkt[i][j]) + (apat[j][i] + krkt[j][i])) for j in range(4))
        for i in range(4)
    )
    return new_state, new_cov


def _mm(a, b):
    n, m, p = len(a), len(b), len(b[0])
    return [[sum(a[i][k] * b[k][j] for k in range(m)) for j in range(p)] for i in range(n)]


def _transpose(a):
    return [[a[j][i] for j in range(len(a))] for i in range(len(a[0]))]


def _inv_2x2(a):
    det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    if abs(det) <= 1.0e-15:
        raise ContractValidationError("innovation covariance is singular")
    inv_det = 1.0 / det
    return ((a[1][1] * inv_det, -a[0][1] * inv_det), (-a[1][0] * inv_det, a[0][0] * inv_det))


def _quad2(v, a_inv):
    iv = (a_inv[0][0] * v[0] + a_inv[0][1] * v[1], a_inv[1][0] * v[0] + a_inv[1][1] * v[1])
    return float(v[0] * iv[0] + v[1] * iv[1])
