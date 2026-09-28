from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "reliability"))

from reliability.contracts import ContractValidationError  # noqa: E402
from reliability.fusion import _validate_spd  # noqa: E402
from reliability.conditional_covariance import (  # noqa: E402
    CovarianceEstimate,
    chi2_coverage,
    default_shrinkage_lambda,
    estimate_conditional_covariance,
    matrix_nll,
    sharpness,
)


TARGET = ((4.0, 1.0), (1.0, 4.0))

# Zero-mean residuals with population covariance [[0.5, 0], [0, 2.0]].
RESIDUALS = [(1.0, 0.0), (-1.0, 0.0), (0.0, 2.0), (0.0, -2.0)]
SAMPLE_COV = ((0.5, 0.0), (0.0, 2.0))


# ---------------------------------------------------------------------------
# Shrinkage limits and defaults
# ---------------------------------------------------------------------------


def test_lambda_zero_returns_sample_covariance() -> None:
    estimate = estimate_conditional_covariance(
        RESIDUALS, "camera_A", "uv", TARGET, shrinkage_lambda=0.0
    )
    for i in range(2):
        for j in range(2):
            assert estimate.covariance[i][j] == pytest.approx(SAMPLE_COV[i][j], abs=1e-12)
    assert estimate.shrinkage_lambda == 0.0
    assert estimate.sample_count == 4
    assert estimate.camera_id == "camera_A"
    assert estimate.frame == "uv"


def test_lambda_one_returns_target() -> None:
    estimate = estimate_conditional_covariance(
        RESIDUALS, "camera_A", "xy", TARGET, shrinkage_lambda=1.0
    )
    for i in range(2):
        for j in range(2):
            assert estimate.covariance[i][j] == pytest.approx(TARGET[i][j], abs=1e-12)
    assert estimate.frame == "xy"


def test_intermediate_lambda_is_convex_combination() -> None:
    lam = 0.25
    estimate = estimate_conditional_covariance(
        RESIDUALS, "camera_A", "uv", TARGET, shrinkage_lambda=lam
    )
    for i in range(2):
        for j in range(2):
            expected = (1.0 - lam) * SAMPLE_COV[i][j] + lam * TARGET[i][j]
            assert estimate.covariance[i][j] == pytest.approx(expected, abs=1e-12)


def test_default_lambda_follows_count_rule() -> None:
    # Pre-registered default: lambda = min(1, k / (k + n)) with k = 10.
    estimate = estimate_conditional_covariance(RESIDUALS, "camera_A", "uv", TARGET)
    assert estimate.shrinkage_lambda == pytest.approx(10.0 / 14.0)
    assert default_shrinkage_lambda(4) == pytest.approx(10.0 / 14.0)
    assert default_shrinkage_lambda(90) == pytest.approx(0.1)
    # More samples -> less shrinkage toward the target.
    assert default_shrinkage_lambda(1000) < default_shrinkage_lambda(10)


def test_default_lambda_validates_sample_count_and_k() -> None:
    for bad_count in (True, False, 0, -1, 1.0, 2.5):
        with pytest.raises(ContractValidationError):
            default_shrinkage_lambda(bad_count)

    assert default_shrinkage_lambda(4, k=0.0) == 0.0
    for bad_k in (True, -0.1, math.nan, math.inf, -math.inf):
        with pytest.raises(ContractValidationError):
            default_shrinkage_lambda(4, k=bad_k)


def test_sample_covariance_uses_population_divisor() -> None:
    # Two residuals, mean (0, 0): population covariance divides by n = 2.
    estimate = estimate_conditional_covariance(
        [(1.0, 1.0), (-1.0, -1.0)], "camera_A", "uv", TARGET, shrinkage_lambda=0.5
    )
    # S = [[1, 1], [1, 1]] with divisor n (unbiased n-1 would give [[2, 2], [2, 2]]).
    expected = ((0.5 * 1.0 + 0.5 * 4.0, 0.5 * 1.0 + 0.5 * 1.0),
                (0.5 * 1.0 + 0.5 * 1.0, 0.5 * 1.0 + 0.5 * 4.0))
    for i in range(2):
        for j in range(2):
            assert estimate.covariance[i][j] == pytest.approx(expected[i][j], abs=1e-12)


# ---------------------------------------------------------------------------
# PSD guarantees and validation
# ---------------------------------------------------------------------------


def test_result_is_spd_even_for_degenerate_residuals() -> None:
    # Perfectly collinear residuals -> singular sample covariance; with
    # lambda = 0 only the PSD floor can rescue it.
    collinear = [(1.0, 1.0), (-1.0, -1.0), (2.0, 2.0), (-2.0, -2.0)]
    estimate = estimate_conditional_covariance(
        collinear, "camera_A", "uv", TARGET, shrinkage_lambda=0.0
    )
    _validate_spd(estimate.covariance, "floored covariance")
    a, b = estimate.covariance[0]
    _, d = estimate.covariance[1]
    assert a * d - b * b > 0.0


def test_result_is_spd_for_identical_residuals() -> None:
    estimate = estimate_conditional_covariance(
        [(0.3, -0.1), (0.3, -0.1)], "camera_A", "uv", TARGET, shrinkage_lambda=0.0
    )
    _validate_spd(estimate.covariance, "floored covariance")


def test_fewer_than_two_residuals_raises() -> None:
    with pytest.raises(ContractValidationError):
        estimate_conditional_covariance([], "camera_A", "uv", TARGET)
    with pytest.raises(ContractValidationError):
        estimate_conditional_covariance([(0.1, 0.2)], "camera_A", "uv", TARGET)


def test_invalid_lambda_frame_and_target_raise() -> None:
    with pytest.raises(ContractValidationError):
        estimate_conditional_covariance(
            RESIDUALS, "camera_A", "uv", TARGET, shrinkage_lambda=1.5
        )
    with pytest.raises(ContractValidationError):
        estimate_conditional_covariance(
            RESIDUALS, "camera_A", "uv", TARGET, shrinkage_lambda=-0.1
        )
    with pytest.raises(ContractValidationError):
        estimate_conditional_covariance(RESIDUALS, "camera_A", "bev", TARGET)
    with pytest.raises(ContractValidationError):
        estimate_conditional_covariance(
            RESIDUALS, "camera_A", "uv", ((1.0, 2.0), (2.0, 1.0))
        )


def test_covariance_estimate_dataclass_validates() -> None:
    with pytest.raises(ContractValidationError):
        CovarianceEstimate(
            camera_id="camera_A",
            covariance=((1.0, 2.0), (2.0, 1.0)),
            sample_count=5,
            frame="uv",
            shrinkage_lambda=0.5,
        )
    for bad_count in (True, 1.0, 2.5, 0, -1):
        with pytest.raises(ContractValidationError):
            CovarianceEstimate(
                camera_id="camera_A",
                covariance=((1.0, 0.0), (0.0, 1.0)),
                sample_count=bad_count,
                frame="uv",
                shrinkage_lambda=0.5,
            )
    with pytest.raises(ContractValidationError):
        CovarianceEstimate(
            camera_id="",
            covariance=((1.0, 0.0), (0.0, 1.0)),
            sample_count=5,
            frame="uv",
            shrinkage_lambda=0.5,
        )


# ---------------------------------------------------------------------------
# MNLL
# ---------------------------------------------------------------------------


def test_matrix_nll_matches_hand_computed_two_point_case() -> None:
    identity = ((1.0, 0.0), (0.0, 1.0))
    residuals = [(1.0, 0.0), (0.0, 2.0)]
    # Per point: 0.5 * (e' e + log 1 + 2 log 2 pi); mahalanobis 1.0 and 4.0.
    expected = 0.5 * ((1.0 + 4.0) / 2.0) + math.log(2.0 * math.pi)
    assert matrix_nll(residuals, [identity, identity]) == pytest.approx(expected, abs=1e-12)


def test_matrix_nll_anisotropic_hand_computed() -> None:
    cov = ((4.0, 0.0), (0.0, 0.25))
    residuals = [(2.0, 0.5)]
    # e' R^-1 e = 4/4 + 0.25/0.25 = 2; log|R| = log(1) = 0.
    expected = 0.5 * (2.0 + 0.0 + 2.0 * math.log(2.0 * math.pi))
    assert matrix_nll(residuals, [cov]) == pytest.approx(expected, abs=1e-12)


def test_matrix_nll_length_mismatch_and_empty_raise() -> None:
    identity = ((1.0, 0.0), (0.0, 1.0))
    with pytest.raises(ContractValidationError):
        matrix_nll([(1.0, 0.0)], [identity, identity])
    with pytest.raises(ContractValidationError):
        matrix_nll([], [])


# ---------------------------------------------------------------------------
# Chi-square coverage
# ---------------------------------------------------------------------------


def test_chi2_coverage_matches_nominal_on_synthetic_gaussians() -> None:
    rng = random.Random(20260717)
    true_cov = ((0.25, 0.0), (0.0, 1.0))
    n = 4000
    residuals = [(rng.gauss(0.0, 0.5), rng.gauss(0.0, 1.0)) for _ in range(n)]
    covariances = [true_cov] * n
    coverage = chi2_coverage(residuals, covariances, q=0.95)
    assert abs(coverage - 0.95) <= 0.03
    coverage_50 = chi2_coverage(residuals, covariances, q=0.50)
    assert abs(coverage_50 - 0.50) <= 0.03


def test_chi2_coverage_boundary_cases() -> None:
    identity = ((1.0, 0.0), (0.0, 1.0))
    threshold = -2.0 * math.log(1.0 - 0.95)  # ~5.9915
    inside = (math.sqrt(threshold) - 1e-6, 0.0)
    outside = (math.sqrt(threshold) + 1e-3, 0.0)
    assert chi2_coverage([inside], [identity]) == 1.0
    assert chi2_coverage([outside], [identity]) == 0.0
    assert chi2_coverage([inside, outside], [identity, identity]) == 0.5


def test_chi2_coverage_huge_covariance_trivially_covers() -> None:
    # Motivates always reporting sharpness alongside coverage.
    huge = ((1.0e6, 0.0), (0.0, 1.0e6))
    residuals = [(3.0, -4.0), (10.0, 10.0)]
    assert chi2_coverage(residuals, [huge, huge]) == 1.0


def test_chi2_coverage_rejects_bad_quantile() -> None:
    identity = ((1.0, 0.0), (0.0, 1.0))
    with pytest.raises(ContractValidationError):
        chi2_coverage([(0.0, 0.0)], [identity], q=0.0)
    with pytest.raises(ContractValidationError):
        chi2_coverage([(0.0, 0.0)], [identity], q=1.0)


# ---------------------------------------------------------------------------
# Sharpness
# ---------------------------------------------------------------------------


def test_sharpness_arithmetic() -> None:
    identity = ((1.0, 0.0), (0.0, 1.0))
    four_i = ((4.0, 0.0), (0.0, 4.0))
    # log|I| = 0, log|4I| = log 16; mean = log(16) / 2.
    assert sharpness([identity, four_i]) == pytest.approx(math.log(16.0) / 2.0, abs=1e-12)
    assert sharpness([identity]) == pytest.approx(0.0, abs=1e-12)


def test_sharpness_empty_raises() -> None:
    with pytest.raises(ContractValidationError):
        sharpness([])
