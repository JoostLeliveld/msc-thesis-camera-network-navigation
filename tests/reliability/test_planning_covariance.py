from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "reliability"))

from reliability.contracts import ContractValidationError  # noqa: E402
from reliability import planning_covariance  # noqa: E402
from reliability.planning_covariance import (  # noqa: E402
    batch_plan_query,
    expected_information_update,
    plan_covariance,
    plan_reliability,
    sequential_expected_update,
)


R_VISIBLE = ((0.05, 0.01), (0.01, 0.08))
R_MISS = ((0.60, 0.02), (0.02, 0.90))
PRIOR = ((0.30, 0.05), (0.05, 0.40))


def _trace(m) -> float:
    return float(m[0][0] + m[1][1])


def _assert_symmetric_psd(m, atol: float = 1.0e-12) -> None:
    assert abs(m[0][1] - m[1][0]) <= 1.0e-9
    det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
    assert m[0][0] >= -atol
    assert m[1][1] >= -atol
    assert det >= -atol


def _assert_close_2x2(a, b, atol: float = 1.0e-9) -> None:
    for i in range(2):
        for j in range(2):
            assert a[i][j] == pytest.approx(b[i][j], abs=atol)


def _full_information_posterior(prior, r_cond):
    """(P^-1 + R^-1)^-1 computed independently of the module under test."""

    def inv(m):
        det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
        return (
            (m[1][1] / det, -m[0][1] / det),
            (-m[1][0] / det, m[0][0] / det),
        )

    ip = inv(prior)
    ir = inv(r_cond)
    lam = ((ip[0][0] + ir[0][0], ip[0][1] + ir[0][1]), (ip[1][0] + ir[1][0], ip[1][1] + ir[1][1]))
    return inv(lam)


def test_plan_reliability_product_and_bounds():
    assert plan_reliability(0.5, 0.8, 0.9) == pytest.approx(0.36)
    assert plan_reliability(0.0, 1.0, 1.0) == 0.0
    assert plan_reliability(1.0, 1.0, 1.0) == 1.0
    for bad in (-0.01, 1.01, math.nan):
        with pytest.raises(ContractValidationError):
            plan_reliability(bad, 0.5, 0.5)
        with pytest.raises(ContractValidationError):
            plan_reliability(0.5, bad, 0.5)
        with pytest.raises(ContractValidationError):
            plan_reliability(0.5, 0.5, bad)


def test_r_zero_returns_r_miss_and_prior_unchanged():
    _assert_close_2x2(plan_covariance(0.0, R_VISIBLE, R_MISS), R_MISS)
    _assert_close_2x2(expected_information_update(PRIOR, [(0.0, R_VISIBLE)]), PRIOR)
    _assert_close_2x2(sequential_expected_update(PRIOR, [(0.0, R_VISIBLE)]), PRIOR)


def test_r_one_returns_r_visible_and_full_update():
    _assert_close_2x2(plan_covariance(1.0, R_VISIBLE, R_MISS), R_VISIBLE)
    expected = _full_information_posterior(PRIOR, R_VISIBLE)
    _assert_close_2x2(expected_information_update(PRIOR, [(1.0, R_VISIBLE)]), expected)
    _assert_close_2x2(sequential_expected_update(PRIOR, [(1.0, R_VISIBLE)]), expected)


def test_covariance_trace_monotone_nonincreasing_in_r():
    rs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for form in (
        lambda r: plan_covariance(r, R_VISIBLE, R_MISS, 1.7),
        lambda r: expected_information_update(PRIOR, [(r, R_VISIBLE)]),
        lambda r: sequential_expected_update(PRIOR, [(r, R_VISIBLE)]),
    ):
        traces = [_trace(form(r)) for r in rs]
        for lo, hi in zip(traces[1:], traces[:-1]):
            assert lo <= hi + 1.0e-12


def test_single_camera_forms_agree_at_endpoints_and_jensen_gap_between():
    # At r=0 and r=1 the information and sequential forms are identical.
    for r in (0.0, 1.0):
        _assert_close_2x2(
            expected_information_update(PRIOR, [(r, R_VISIBLE)]),
            sequential_expected_update(PRIOR, [(r, R_VISIBLE)]),
        )
    # For fractional r they genuinely differ even for ONE camera (Jensen gap:
    # sequential is E[covariance] of a Bernoulli(r) update, information form
    # scales precision) — sequential dominates in trace; both PSD, both <= prior.
    info = expected_information_update(PRIOR, [(0.5, R_VISIBLE)])
    seq = sequential_expected_update(PRIOR, [(0.5, R_VISIBLE)])
    assert max(abs(info[i][j] - seq[i][j]) for i in range(2) for j in range(2)) > 1.0e-6
    _assert_symmetric_psd(info)
    _assert_symmetric_psd(seq)
    assert _trace(info) <= _trace(seq) + 1.0e-12
    assert _trace(seq) <= _trace(PRIOR) + 1.0e-12


def test_multi_camera_forms_differ_both_psd_and_below_prior():
    terms = [(0.7, R_VISIBLE), (0.4, ((0.10, -0.02), (-0.02, 0.06)))]
    info = expected_information_update(PRIOR, terms)
    seq = sequential_expected_update(PRIOR, terms)
    assert max(abs(info[i][j] - seq[i][j]) for i in range(2) for j in range(2)) > 1.0e-6
    _assert_symmetric_psd(info)
    _assert_symmetric_psd(seq)
    assert _trace(info) <= _trace(PRIOR) + 1.0e-12
    assert _trace(seq) <= _trace(PRIOR) + 1.0e-12


def test_psd_outputs_under_extreme_inputs():
    # Note: fusion._mat_inv_2x2 guards det <= 1e-12 as singular, so "tiny"
    # stays just above that floor (1e-4 diag => det 1e-8).
    tiny = ((1.0e-4, 0.0), (0.0, 1.0e-4))
    huge = ((1.0e4, 9.9e3), (9.9e3, 1.0e4))  # strongly correlated, SPD
    for r in (0.0, 1.0e-9, 0.5, 1.0 - 1.0e-9, 1.0):
        _assert_symmetric_psd(plan_covariance(r, tiny, huge, 0.1))
        _assert_symmetric_psd(plan_covariance(r, tiny, huge, 8.0))
        _assert_symmetric_psd(expected_information_update(huge, [(r, tiny)]))
        _assert_symmetric_psd(sequential_expected_update(huge, [(r, tiny)]))
        _assert_symmetric_psd(expected_information_update(tiny, [(r, huge)]))
        _assert_symmetric_psd(sequential_expected_update(tiny, [(r, huge)]))


def test_full_loewner_violation_and_bad_gamma_raise():
    smaller_than_visible = ((0.01, 0.0), (0.0, 0.01))
    with pytest.raises(ContractValidationError):
        plan_covariance(0.5, R_VISIBLE, smaller_than_visible)
    # Both diagonal entries grow, but the changed correlation makes
    # R_miss - R_visible indefinite (eigenvalues 2.8 and -0.8).
    correlated_visible = ((1.0, 0.9), (0.9, 1.0))
    correlated_miss = ((2.0, -0.9), (-0.9, 2.0))
    with pytest.raises(ContractValidationError):
        plan_covariance(0.5, correlated_visible, correlated_miss)
    with pytest.raises(ContractValidationError):
        plan_covariance(0.5, R_VISIBLE, R_MISS, gamma=0.0)
    with pytest.raises(ContractValidationError):
        plan_covariance(0.5, R_VISIBLE, R_MISS, gamma=-1.0)
    not_spd = ((0.05, 0.10), (0.10, 0.05))
    with pytest.raises(ContractValidationError):
        plan_covariance(0.5, not_spd, R_MISS)
    with pytest.raises(ContractValidationError):
        expected_information_update(not_spd, [(0.5, R_VISIBLE)])
    with pytest.raises(ContractValidationError):
        sequential_expected_update(PRIOR, [(0.5, not_spd)])


def test_loewner_order_allows_equal_endpoints_and_semidefinite_difference():
    visible = ((1.0, 0.0), (0.0, 1.0))
    _assert_close_2x2(plan_covariance(0.4, visible, visible), visible)

    # Difference [[1, 1], [1, 1]] has eigenvalues 0 and 2, so it is PSD but
    # not positive definite. This is a valid (non-strict) Loewner increase.
    miss = ((2.0, 1.0), (1.0, 2.0))
    _assert_close_2x2(
        plan_covariance(0.5, visible, miss),
        ((1.5, 0.5), (0.5, 1.5)),
    )


def test_batch_plan_query_two_states_two_cameras():
    states = [(0.0, 0.0), (4.0, -1.0)]
    per_camera = {
        "camera_A": lambda xy: (0.9, 0.8),
        "camera_B": lambda xy: (0.5 if xy[0] > 2.0 else 0.1, 0.6),
    }
    healths = {"camera_A": 1.0, "camera_B": 0.5}
    r_visible = {"camera_A": R_VISIBLE, "camera_B": R_VISIBLE}
    r_miss = {"camera_A": R_MISS, "camera_B": R_MISS}
    out = batch_plan_query(states, per_camera, healths, r_visible, r_miss, gamma=1.0)

    assert len(out) == 2
    for per_state in out:
        assert set(per_state) == {"camera_A", "camera_B"}
        for entry in per_state.values():
            assert set(entry) == {"reliability", "r_plan_xy"}
            assert 0.0 <= entry["reliability"] <= 1.0
            _assert_symmetric_psd(entry["r_plan_xy"])

    assert out[0]["camera_A"]["reliability"] == pytest.approx(0.9 * 0.8 * 1.0)
    assert out[0]["camera_B"]["reliability"] == pytest.approx(0.1 * 0.6 * 0.5)
    assert out[1]["camera_B"]["reliability"] == pytest.approx(0.5 * 0.6 * 0.5)
    # Values match a direct plan_covariance call with the same constants.
    _assert_close_2x2(
        out[1]["camera_B"]["r_plan_xy"],
        plan_covariance(0.5 * 0.6 * 0.5, R_VISIBLE, R_MISS, 1.0),
    )
    # Higher reliability at state 1 for camera_B => smaller planned covariance.
    assert _trace(out[1]["camera_B"]["r_plan_xy"]) < _trace(out[0]["camera_B"]["r_plan_xy"])

    with pytest.raises(ContractValidationError):
        batch_plan_query(states, per_camera, {"camera_A": 1.0}, r_visible, r_miss, 1.0)


def test_no_public_parameter_accepts_per_state_confidence():
    """Future-confidence leak guard (plan 10 gate): the planner-facing API must
    not expose any per-future-state confidence/health-snapshot argument."""

    forbidden = ("confidence", "conf", "detector_score", "score", "health_snapshot")
    public_functions = [
        obj
        for name, obj in vars(planning_covariance).items()
        if not name.startswith("_") and inspect.isfunction(obj) and obj.__module__ == planning_covariance.__name__
    ]
    assert len(public_functions) >= 5
    for func in public_functions:
        for param in inspect.signature(func).parameters:
            lowered = param.lower()
            for token in forbidden:
                assert token not in lowered, f"{func.__name__} exposes confidence-like parameter {param!r}"
