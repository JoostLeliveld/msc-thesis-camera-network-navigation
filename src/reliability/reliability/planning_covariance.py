"""Planning adapter: predicted reliability -> future observation covariance (plan 10, §13).

HARD RULE (thesis architecture): planning consumes ONLY spatially predictable
reliability heads (availability/quality LCBs evaluated at candidate future
states) plus slow per-camera health averages (EWMA h-bar). The current frame's
detector confidence is NEVER copied across the planning horizon, and no API
parameter in this module may accept per-future-state confidence. A guard test
asserts the public signatures stay free of confidence-like parameters.

Forms implemented (pure functions, stdlib only; planner node imports these):

    r_plan_i(s) = a_lcb_i(s) * q_lcb_i(s) * h_bar_i
    R_plan_i(s) = R_visible_i + (1 - r_plan_i(s))^gamma * (R_miss_i - R_visible_i)

    information form:  Lambda+ = (P-)^-1 + sum_i r_i (R_i)^-1   (H = I position model)
    sequential form:   P+ <- P - r_i * P (P + R_i)^-1 P         (per camera, in order)

The two belief-propagation candidates coincide at r in {0, 1}; for fractional
r they differ (Jensen gap: sequential >= information in the Loewner order) —
plan 10 compares their calibration on held-out rollouts.
"""

from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence

from reliability.contracts import ContractValidationError, _finite_float, _finite_probability
from reliability.fusion import (
    _mat_add,
    _mat_inv_2x2,
    _mat_mul,
    _mat_sub,
    _matrix_2x2,
    _pair,
    _symmetrize,
    _validate_spd,
)

Matrix2x2 = tuple[tuple[float, float], tuple[float, float]]


def plan_reliability(availability_lcb: float, quality_lcb: float, mean_health: float) -> float:
    """r_plan = a_lcb * q_lcb * h_bar; every factor validated in [0, 1].

    ``mean_health`` is the slow health EWMA (plan 08) — a per-camera scalar,
    never a per-state or per-frame quantity.
    """

    a = _finite_probability(availability_lcb, field_name="availability_lcb")
    q = _finite_probability(quality_lcb, field_name="quality_lcb")
    h = _finite_probability(mean_health, field_name="mean_health")
    return float(a * q * h)


def plan_covariance(
    r_plan: float,
    r_visible: Sequence[Sequence[float]],
    r_miss: Sequence[Sequence[float]],
    gamma: float = 1.0,
) -> Matrix2x2:
    """R_plan = R_visible + (1 - r_plan)^gamma * (R_miss - R_visible).

    Endpoints must be SPD and satisfy ``R_miss >= R_visible`` in the full
    Loewner order: ``R_miss - R_visible`` must be positive semidefinite.
    Equality and singular positive-semidefinite differences are allowed.
    ``gamma`` must be > 0. The result is a convex combination of the SPD
    endpoints, returned symmetrized (hence symmetric positive definite).
    """

    r = _finite_probability(r_plan, field_name="r_plan")
    g = _finite_float(gamma, field_name="gamma")
    if g <= 0.0:
        raise ContractValidationError(f"gamma must be > 0, got {g}")
    visible = _matrix_2x2(r_visible, "r_visible")
    miss = _matrix_2x2(r_miss, "r_miss")
    _validate_spd(visible, "r_visible")
    _validate_spd(miss, "r_miss")
    difference = _symmetrize(_mat_sub(miss, visible))
    if not _is_psd_2x2(difference):
        raise ContractValidationError(
            "r_miss - r_visible must be positive semidefinite: "
            "missing a camera can never shrink covariance in any direction"
        )
    weight = (1.0 - r) ** g
    blended = _mat_add(visible, _mat_scale(weight, difference))
    out = _symmetrize(blended)
    _validate_spd(out, "r_plan_xy")
    return out


def expected_information_update(
    cov_prior: Sequence[Sequence[float]],
    camera_terms: Sequence[tuple[float, Sequence[Sequence[float]]]],
) -> Matrix2x2:
    """Multi-camera expected-information update (H = I position model).

    Lambda+ = (P-)^-1 + sum_i r_i (R_i)^-1 ; returns (Lambda+)^-1 symmetrized.
    Terms with r_i = 0 contribute nothing (pure prediction); r_i = 1 is a
    full-weight update. Prior and every conditional covariance must be SPD.
    """

    prior = _matrix_2x2(cov_prior, "cov_prior")
    _validate_spd(prior, "cov_prior")
    information = _mat_inv_2x2(prior)
    for index, (r_plan, r_cond) in enumerate(camera_terms):
        r = _finite_probability(r_plan, field_name=f"camera_terms[{index}].r_plan")
        cond = _matrix_2x2(r_cond, f"camera_terms[{index}].r_cond")
        _validate_spd(cond, f"camera_terms[{index}].r_cond")
        if r == 0.0:
            continue
        information = _mat_add(information, _mat_scale(r, _mat_inv_2x2(cond)))
    return _symmetrize(_mat_inv_2x2(_symmetrize(information)))


def sequential_expected_update(
    cov_prior: Sequence[Sequence[float]],
    camera_terms: Sequence[tuple[float, Sequence[Sequence[float]]]],
) -> Matrix2x2:
    """Per-camera §13.3 form, applied sequentially: P+ = P - r * P (P + R)^-1 P.

    Each step is the Bernoulli(r)-expected covariance of a Kalman update that
    arrives with probability r. Terms with r = 0 leave P unchanged.
    """

    cov = _matrix_2x2(cov_prior, "cov_prior")
    _validate_spd(cov, "cov_prior")
    for index, (r_plan, r_cond) in enumerate(camera_terms):
        r = _finite_probability(r_plan, field_name=f"camera_terms[{index}].r_plan")
        cond = _matrix_2x2(r_cond, f"camera_terms[{index}].r_cond")
        _validate_spd(cond, f"camera_terms[{index}].r_cond")
        if r == 0.0:
            continue
        gain = _mat_mul(cov, _mat_inv_2x2(_mat_add(cov, cond)))
        reduction = _mat_mul(gain, cov)
        cov = _symmetrize(_mat_sub(cov, _mat_scale(r, reduction)))
        _validate_spd(cov, f"camera_terms[{index}] posterior")
    return cov


def batch_plan_query(
    states_xy: Sequence[Sequence[float]],
    per_camera: Mapping[str, Callable[[tuple[float, float]], tuple[float, float]]],
    healths: Mapping[str, float],
    r_visible: Mapping[str, Sequence[Sequence[float]]],
    r_miss: Mapping[str, Sequence[Sequence[float]]],
    gamma: float = 1.0,
) -> list[dict[str, dict[str, object]]]:
    """Planner-query adapter (§19): candidate states -> per-state, per-camera R_plan.

    ``per_camera`` maps camera_id -> callable(state_xy) -> (availability_lcb,
    quality_lcb): the spatial reliability heads evaluated at a FUTURE state.
    ``healths`` maps camera_id -> slow health EWMA. Returns one dict per state:
    {camera_id: {"reliability": r_plan, "r_plan_xy": 2x2 covariance}}.
    Pure function — no node state, no clocks, no current-frame inputs.
    """

    for camera_id in per_camera:
        for mapping, name in ((healths, "healths"), (r_visible, "r_visible"), (r_miss, "r_miss")):
            if camera_id not in mapping:
                raise ContractValidationError(f"{name} is missing camera_id {camera_id!r}")
    results: list[dict[str, dict[str, object]]] = []
    for state_index, state in enumerate(states_xy):
        state_xy = _pair(state, f"states_xy[{state_index}]")
        per_state: dict[str, dict[str, object]] = {}
        for camera_id, head in per_camera.items():
            availability_lcb, quality_lcb = head(state_xy)
            reliability = plan_reliability(availability_lcb, quality_lcb, healths[camera_id])
            per_state[camera_id] = {
                "reliability": reliability,
                "r_plan_xy": plan_covariance(
                    reliability, r_visible[camera_id], r_miss[camera_id], gamma
                ),
            }
        results.append(per_state)
    return results


def _mat_scale(scalar: float, matrix: Matrix2x2) -> Matrix2x2:
    return (
        (scalar * matrix[0][0], scalar * matrix[0][1]),
        (scalar * matrix[1][0], scalar * matrix[1][1]),
    )


def _is_psd_2x2(matrix: Matrix2x2) -> bool:
    """Return whether a symmetric 2x2 matrix is PSD, up to roundoff."""

    a, b = matrix[0]
    _, d = matrix[1]
    scale = max(1.0, abs(a), abs(b), abs(d))
    min_eigenvalue = 0.5 * (a + d) - math.hypot(0.5 * (a - d), b)
    return min_eigenvalue >= -1.0e-12 * scale
