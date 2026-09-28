"""Conditional measurement covariance R_cond (plan 06, section 9 / E2).

Per-camera anisotropic 2x2 covariance of usable observations — the quantity
the filter actually consumes. Strictly separate from reliability (the GP is
never re-labelled as covariance). MVP tier: one 2x2 per camera from residuals
with shrinkage toward a pooled target, plus the covariance-specific evaluation
metrics MNLL, chi-square coverage, and sharpness.

Conventions (pre-registered defaults):

* Sample covariance uses the **population divisor n** (the maximum-likelihood
  estimator). The small-sample bias this introduces is exactly what the
  shrinkage step is there to absorb, and it keeps the estimator defined for
  n = 2; documented here so the choice is auditable.
* Default shrinkage weight follows the fixed-count rule
  ``lambda = min(1, k / (k + n))`` with ``k = 10`` — the pre-registered
  default from plan 06 (Ledoit-Wolf-style count rule, no data-dependent
  tuning).

Pure library: no ROS, no numpy. 2x2 math reuses :mod:`reliability.fusion`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Sequence

from reliability.contracts import ContractValidationError
from reliability.fusion import (
    _matrix_2x2,
    _pair,
    _quad_form_inverse_2x2,
    _validate_spd,
)


SHRINKAGE_COUNT_K = 10.0
"""Pre-registered fixed count for the default shrinkage rule lambda = k/(k+n)."""

_PSD_FLOOR_EPS = 1.0e-9

Matrix2 = tuple[tuple[float, float], tuple[float, float]]

_ALLOWED_FRAMES = ("uv", "xy")

_LOG_TWO_PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class CovarianceEstimate:
    """Shrunk per-camera 2x2 residual covariance with its provenance."""

    camera_id: str
    covariance: Matrix2
    sample_count: int
    frame: str
    shrinkage_lambda: float

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ContractValidationError("camera_id must be a non-empty string")
        covariance = _matrix_2x2(self.covariance, "covariance")
        _validate_spd(covariance, "covariance")
        object.__setattr__(self, "covariance", covariance)
        sample_count = _positive_sample_count(self.sample_count)
        object.__setattr__(self, "sample_count", sample_count)
        if self.frame not in _ALLOWED_FRAMES:
            raise ContractValidationError("frame must be 'uv' or 'xy'")
        lam = float(self.shrinkage_lambda)
        if not math.isfinite(lam) or lam < 0.0 or lam > 1.0:
            raise ContractValidationError("shrinkage_lambda must be in [0, 1]")
        object.__setattr__(self, "shrinkage_lambda", lam)


def default_shrinkage_lambda(sample_count: int, *, k: float = SHRINKAGE_COUNT_K) -> float:
    """Fixed-count shrinkage rule ``lambda = k / (k + n)`` (default k = 10).

    ``sample_count`` must be a positive integer (booleans and numeric floats
    are rejected). ``k`` must be finite and non-negative; ``k = 0`` disables
    shrinkage and returns exactly zero.
    """

    n = _positive_sample_count(sample_count)
    if isinstance(k, bool):
        raise ContractValidationError("k must be finite and non-negative")
    try:
        k_value = float(k)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContractValidationError("k must be finite and non-negative") from exc
    if not math.isfinite(k_value) or k_value < 0.0:
        raise ContractValidationError("k must be finite and non-negative")
    if k_value == 0.0:
        return 0.0
    try:
        n_value = float(n)
    except OverflowError:
        # Any integer too large for a finite float dominates every finite k.
        return 0.0
    return 1.0 / (1.0 + n_value / k_value)


def estimate_conditional_covariance(
    residuals: Sequence[Sequence[float]],
    camera_id: str,
    frame: str,
    shrinkage_target: Sequence[Sequence[float]],
    shrinkage_lambda: float | None = None,
) -> CovarianceEstimate:
    """Estimate a per-camera 2x2 residual covariance with shrinkage.

    ``R_hat = (1 - lambda) * S + lambda * T`` where ``S`` is the sample
    covariance of ``residuals`` (population divisor n — see module docstring)
    and ``T`` is ``shrinkage_target`` (e.g. the global pooled matrix). When
    ``shrinkage_lambda`` is None the pre-registered default
    ``min(1, 10 / (10 + n))`` is used. A PSD floor adds ``eps * I`` when the
    smallest eigenvalue drops below ``eps = 1e-9``, so the returned covariance
    is always symmetric positive definite. Raises for fewer than 2 residuals.
    """

    points = [_pair(residual, f"residuals[{index}]") for index, residual in enumerate(residuals)]
    n = len(points)
    if n < 2:
        raise ContractValidationError("at least 2 residuals are required")
    target = _matrix_2x2(shrinkage_target, "shrinkage_target")
    _validate_spd(target, "shrinkage_target")
    if shrinkage_lambda is None:
        lam = default_shrinkage_lambda(n)
    else:
        lam = float(shrinkage_lambda)
        if not math.isfinite(lam) or lam < 0.0 or lam > 1.0:
            raise ContractValidationError("shrinkage_lambda must be in [0, 1]")

    mean_x = sum(point[0] for point in points) / n
    mean_y = sum(point[1] for point in points) / n
    sxx = sum((point[0] - mean_x) ** 2 for point in points) / n
    syy = sum((point[1] - mean_y) ** 2 for point in points) / n
    sxy = sum((point[0] - mean_x) * (point[1] - mean_y) for point in points) / n
    sample = ((sxx, sxy), (sxy, syy))

    shrunk = (
        (
            (1.0 - lam) * sample[0][0] + lam * target[0][0],
            (1.0 - lam) * sample[0][1] + lam * target[0][1],
        ),
        (
            (1.0 - lam) * sample[1][0] + lam * target[1][0],
            (1.0 - lam) * sample[1][1] + lam * target[1][1],
        ),
    )
    floored = _psd_floor(shrunk)
    return CovarianceEstimate(
        camera_id=camera_id,
        covariance=floored,
        sample_count=n,
        frame=frame,
        shrinkage_lambda=lam,
    )


def matrix_nll(
    residuals: Sequence[Sequence[float]],
    covariances: Sequence[Sequence[Sequence[float]]],
) -> float:
    """Mean 2D Gaussian negative log-likelihood (MNLL, plan 06 E2).

    ``mean over t of 0.5 * (e_t' R_t^-1 e_t + log|R_t| + 2 log 2 pi)``.
    """

    pairs = _paired_residual_covariances(residuals, covariances)
    total = 0.0
    for residual, covariance in pairs:
        maha = _quad_form_inverse_2x2(residual, covariance)
        total += 0.5 * (maha + _log_det_2x2(covariance) + 2.0 * _LOG_TWO_PI)
    return total / len(pairs)


def chi2_coverage(
    residuals: Sequence[Sequence[float]],
    covariances: Sequence[Sequence[Sequence[float]]],
    q: float = 0.95,
) -> float:
    """Fraction of residuals with ``e' R^-1 e <= chi2_{2, q}``.

    The 2-dof chi-square quantile is analytic: ``-2 * ln(1 - q)``.
    """

    q_f = float(q)
    if not math.isfinite(q_f) or q_f <= 0.0 or q_f >= 1.0:
        raise ContractValidationError("q must be in (0, 1)")
    threshold = -2.0 * math.log(1.0 - q_f)
    pairs = _paired_residual_covariances(residuals, covariances)
    covered = sum(
        1 for residual, covariance in pairs
        if _quad_form_inverse_2x2(residual, covariance) <= threshold
    )
    return covered / len(pairs)


def sharpness(covariances: Sequence[Sequence[Sequence[float]]]) -> float:
    """Mean ``log|R|`` — reported next to coverage (a huge-R model passes
    coverage trivially, so both are always quoted together)."""

    matrices = [_validated_matrix(covariance, index) for index, covariance in enumerate(covariances)]
    if not matrices:
        raise ContractValidationError("covariances must be non-empty")
    return sum(_log_det_2x2(matrix) for matrix in matrices) / len(matrices)


def _paired_residual_covariances(
    residuals: Sequence[Sequence[float]],
    covariances: Sequence[Sequence[Sequence[float]]],
) -> list[tuple[tuple[float, float], Matrix2]]:
    points = [_pair(residual, f"residuals[{index}]") for index, residual in enumerate(residuals)]
    matrices = [_validated_matrix(covariance, index) for index, covariance in enumerate(covariances)]
    if not points:
        raise ContractValidationError("residuals must be non-empty")
    if len(points) != len(matrices):
        raise ContractValidationError("residuals and covariances must have equal length")
    return list(zip(points, matrices))


def _positive_sample_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ContractValidationError("sample_count must be a positive integer")
    sample_count = int(value)
    if sample_count < 1:
        raise ContractValidationError("sample_count must be a positive integer")
    return sample_count


def _validated_matrix(value: Sequence[Sequence[float]], index: int) -> Matrix2:
    matrix = _matrix_2x2(value, f"covariances[{index}]")
    _validate_spd(matrix, f"covariances[{index}]")
    return matrix


def _log_det_2x2(matrix: Matrix2) -> float:
    det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    if det <= 0.0:
        raise ContractValidationError("covariance determinant must be positive")
    return math.log(det)


def _min_eigenvalue_2x2(matrix: Matrix2) -> float:
    a = matrix[0][0]
    b = 0.5 * (matrix[0][1] + matrix[1][0])
    d = matrix[1][1]
    half_trace = 0.5 * (a + d)
    radius = math.sqrt((0.5 * (a - d)) ** 2 + b * b)
    return half_trace - radius


def _psd_floor(matrix: Matrix2, eps: float = _PSD_FLOOR_EPS) -> Matrix2:
    symmetric = (
        (matrix[0][0], 0.5 * (matrix[0][1] + matrix[1][0])),
        (0.5 * (matrix[0][1] + matrix[1][0]), matrix[1][1]),
    )
    min_eig = _min_eigenvalue_2x2(symmetric)
    if min_eig >= eps:
        return symmetric
    shift = eps - min_eig + eps
    return (
        (symmetric[0][0] + shift, symmetric[0][1]),
        (symmetric[1][0], symmetric[1][1] + shift),
    )
