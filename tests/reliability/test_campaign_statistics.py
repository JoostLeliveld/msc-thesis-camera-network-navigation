from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "reliability"))

from reliability.campaign_statistics import (  # noqa: E402
    Leaf,
    NominalSummary,
    PRIMARY_COMPARISONS,
    beats_toro_degraded,
    beats_toro_nominal,
    beats_toro_navigation,
    classify_comparison,
    error_severity_auc,
    hierarchical_bootstrap,
    mean,
    median,
    paired_bootstrap,
    paired_differences,
    summarize_paired,
    wilson_interval,
)


# --------------------------------------------------------------------------- #
# Wilson interval
# --------------------------------------------------------------------------- #
def test_wilson_reference_value_8_of_10() -> None:
    est = wilson_interval(8, 10)
    assert est.point == pytest.approx(0.8)
    # Reference Wilson score interval (no continuity correction) for 8/10 @95%.
    assert est.low == pytest.approx(0.4902, abs=1e-3)
    assert est.high == pytest.approx(0.9433, abs=1e-3)
    assert 0.0 <= est.low < est.point < est.high <= 1.0


def test_wilson_does_not_collapse_at_extremes() -> None:
    full = wilson_interval(10, 10)
    assert full.point == 1.0
    assert full.high == pytest.approx(1.0)
    assert full.low > 0.0  # Wald would give a degenerate zero-width interval
    zero = wilson_interval(0, 10)
    assert zero.point == 0.0
    assert zero.low == pytest.approx(0.0)
    assert zero.high > 0.0


def test_wilson_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        wilson_interval(3, 0)
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
    with pytest.raises(ValueError):
        wilson_interval(-1, 10)
    with pytest.raises(ValueError):
        wilson_interval(5, 10, z=0.0)


# --------------------------------------------------------------------------- #
# Paired differences
# --------------------------------------------------------------------------- #
def test_paired_differences_values_and_order() -> None:
    proposed = {"r2s1": 0.30, "r1s0": 0.10}
    baseline = {"r1s0": 0.25, "r2s1": 0.50}
    diffs = paired_differences(proposed, baseline)
    assert [k for k, _ in diffs] == ["r1s0", "r2s1"]  # sorted by key
    assert dict(diffs)["r1s0"] == pytest.approx(-0.15)
    assert dict(diffs)["r2s1"] == pytest.approx(-0.20)


def test_paired_differences_requires_identical_keys() -> None:
    with pytest.raises(ValueError):
        paired_differences({"a": 1.0, "b": 2.0}, {"a": 1.0})


# --------------------------------------------------------------------------- #
# mean / median helpers
# --------------------------------------------------------------------------- #
def test_mean_median_and_empty_guards() -> None:
    assert mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert median([3.0, 1.0, 2.0]) == pytest.approx(2.0)
    assert median([1.0, 2.0, 3.0, 4.0]) == pytest.approx(2.5)
    with pytest.raises(ValueError):
        mean([])
    with pytest.raises(ValueError):
        median([])


# --------------------------------------------------------------------------- #
# Bootstraps
# --------------------------------------------------------------------------- #
def _leaves_shifted(shift: float) -> list[Leaf]:
    leaves = []
    for r in ("A", "B", "C"):
        for s in ("0", "1"):
            for e, jitter in enumerate((-0.1, 0.1)):
                leaves.append(Leaf(route=r, seed=s, episode=str(e), delta=shift + jitter))
    return leaves


def test_hierarchical_bootstrap_deterministic_and_brackets_point() -> None:
    leaves = _leaves_shifted(-2.0)
    a = hierarchical_bootstrap(leaves, seed=7)
    b = hierarchical_bootstrap(leaves, seed=7)
    assert (a.point, a.low, a.high) == (b.point, b.low, b.high)
    assert a.point == pytest.approx(-2.0, abs=1e-9)
    assert a.low <= a.point <= a.high
    # Clearly-negative shift, lower-is-better: every unit favourable, CI < 0.
    assert a.proportion_favorable == pytest.approx(1.0)
    assert a.high < 0.0


def test_hierarchical_bootstrap_seed_changes_replicates() -> None:
    # Spread the deltas so resampling actually moves the CI percentiles;
    # a near-constant set would give seed-insensitive endpoints (correctly).
    spread = (-3.0, -2.4, -1.8, -1.1, -0.4, 0.3)
    leaves = []
    for r in ("A", "B", "C"):
        for s in ("0", "1"):
            for e, d in enumerate(spread):
                leaves.append(Leaf(route=r, seed=s, episode=str(e), delta=d))
    a = hierarchical_bootstrap(leaves, seed=1)
    b = hierarchical_bootstrap(leaves, seed=2)
    assert (a.low, a.high) != (b.low, b.high)


def test_hierarchical_wider_than_flat_under_strong_clustering() -> None:
    # Route A ≈ -10, route B ≈ +10: between-cluster variance dominates.
    leaves = []
    for e in range(4):
        leaves.append(Leaf(route="A", seed="0", episode=str(e), delta=-10.0 + 0.01 * e))
        leaves.append(Leaf(route="B", seed="0", episode=str(e), delta=10.0 - 0.01 * e))
    deltas = [leaf.delta for leaf in leaves]
    hier = hierarchical_bootstrap(leaves, seed=3, n_boot=3000)
    flat = paired_bootstrap(deltas, seed=3, n_boot=3000)
    # Resampling whole routes exposes the ±10 swing that iid leaf resampling hides.
    assert (hier.high - hier.low) > (flat.high - flat.low)


def test_paired_bootstrap_recovers_mean_and_brackets() -> None:
    deltas = [-1.0, -0.5, -1.5, -0.8, -1.2]
    boot = paired_bootstrap(deltas, seed=0, n_boot=2000)
    assert boot.point == pytest.approx(mean(deltas))
    assert boot.low <= boot.point <= boot.high
    assert boot.proportion_favorable == pytest.approx(1.0)  # all negative, lower better


def test_summarize_paired_ci_excludes_zero_on_clear_shift() -> None:
    proposed = {f"u{i}": 0.10 + 0.001 * i for i in range(12)}
    baseline = {f"u{i}": 0.30 + 0.001 * i for i in range(12)}
    summary = summarize_paired(proposed, baseline, lower_is_better=True, seed=0)
    assert summary.n == 12
    assert summary.mean_delta == pytest.approx(-0.20, abs=1e-9)
    assert summary.proportion_improved == pytest.approx(1.0)
    assert summary.ci_excludes_zero is True
    assert summary.ci_high < 0.0


def test_summarize_paired_ci_includes_zero_when_no_effect() -> None:
    # Symmetric deltas around zero → CI should straddle zero.
    proposed = {f"u{i}": 1.0 + (0.5 if i % 2 else -0.5) for i in range(10)}
    baseline = {f"u{i}": 1.0 for i in range(10)}
    summary = summarize_paired(proposed, baseline, seed=0)
    assert summary.mean_delta == pytest.approx(0.0, abs=1e-9)
    assert summary.ci_excludes_zero is False


# --------------------------------------------------------------------------- #
# error-severity AUC
# --------------------------------------------------------------------------- #
def test_error_severity_auc_trapezoid() -> None:
    assert error_severity_auc([0, 1, 2], [1, 2, 4]) == pytest.approx(4.5)


def test_error_severity_auc_averages_duplicate_severities() -> None:
    assert error_severity_auc([0, 0, 1], [1, 3, 5]) == pytest.approx(3.5)


def test_error_severity_auc_guards() -> None:
    with pytest.raises(ValueError):
        error_severity_auc([0, 1], [1])
    with pytest.raises(ValueError):
        error_severity_auc([0], [1])
    with pytest.raises(ValueError):
        error_severity_auc([1, 1], [2, 3])  # only one distinct severity


# --------------------------------------------------------------------------- #
# Multiple-comparison discipline
# --------------------------------------------------------------------------- #
def test_classify_comparison() -> None:
    assert PRIMARY_COMPARISONS == {"full_vs_toro", "full_vs_gp_only", "fusion_vs_selection"}
    assert classify_comparison("full_vs_toro") == "primary"
    assert classify_comparison("full_vs_constant") == "exploratory"


# --------------------------------------------------------------------------- #
# §17 beats-Toro decision rules
# --------------------------------------------------------------------------- #
def _nominal(p95: float, nll: float, cov: float, rt: bool = True) -> NominalSummary:
    return NominalSummary(
        median_run_p95_ate=p95, localization_nll=nll, coverage_c95=cov, real_time_ok=rt
    )


def test_beats_toro_nominal_all_pass() -> None:
    proposed = _nominal(0.20, -1.5, 0.94)
    toro = _nominal(0.25, -1.0, 0.80)
    v = beats_toro_nominal(proposed, toro, max_error_increase_ruled_out=True)
    assert v.passed is True
    assert all(v.criteria.values())


def test_beats_toro_nominal_single_failure_blocks() -> None:
    proposed = _nominal(0.20, -1.5, 0.94)
    toro = _nominal(0.25, -1.0, 0.80)
    # Max-error increase not ruled out → overall fails despite other wins.
    v = beats_toro_nominal(proposed, toro, max_error_increase_ruled_out=False)
    assert v.passed is False
    assert v.criteria["no_meaningful_max_error_increase"] is False
    assert v.criteria["lower_median_p95_ate"] is True


def test_beats_toro_nominal_coverage_closer_logic() -> None:
    # proposed over-covers slightly (0.99), toro under-covers badly (0.70):
    # |0.99-0.95|=0.04 < |0.70-0.95|=0.25 → proposed closer to nominal.
    proposed = _nominal(0.20, -1.5, 0.99)
    toro = _nominal(0.25, -1.0, 0.70)
    v = beats_toro_nominal(proposed, toro, max_error_increase_ruled_out=True)
    assert v.criteria["coverage_closer_to_nominal"] is True


def test_beats_toro_nominal_not_realtime_blocks() -> None:
    proposed = _nominal(0.20, -1.5, 0.94, rt=False)
    toro = _nominal(0.25, -1.0, 0.80)
    v = beats_toro_nominal(proposed, toro, max_error_increase_ruled_out=True)
    assert v.passed is False


def test_beats_toro_degraded_truth_table() -> None:
    assert beats_toro_degraded(
        lower_error_severity_auc=True,
        lower_max_error=True,
        faster_fault_isolation=True,
        lower_nav_failure_rate=True,
    ).passed is True
    assert beats_toro_degraded(
        lower_error_severity_auc=True,
        lower_max_error=False,
        faster_fault_isolation=True,
        lower_nav_failure_rate=True,
    ).passed is False


def test_beats_toro_navigation_tolerances_and_contacts() -> None:
    ok = beats_toro_navigation(
        better_breach_free_completion=True,
        fewer_geometry_breaches=False,
        path_length_overhead=0.08,
        travel_time_overhead=0.05,
        physics_contacts_increase=0,
        max_path_length_overhead=0.10,
        max_travel_time_overhead=0.10,
    )
    assert ok.passed is True

    # Path overhead exceeds tolerance → blocked.
    over = beats_toro_navigation(
        better_breach_free_completion=True,
        fewer_geometry_breaches=True,
        path_length_overhead=0.15,
        travel_time_overhead=0.05,
        physics_contacts_increase=0,
        max_path_length_overhead=0.10,
        max_travel_time_overhead=0.10,
    )
    assert over.passed is False
    assert over.criteria["path_length_within_tolerance"] is False

    # More physics contacts → blocked regardless of safety gains.
    contact = beats_toro_navigation(
        better_breach_free_completion=True,
        fewer_geometry_breaches=True,
        path_length_overhead=0.0,
        travel_time_overhead=0.0,
        physics_contacts_increase=2,
        max_path_length_overhead=0.10,
        max_travel_time_overhead=0.10,
    )
    assert contact.passed is False

    # Neither safety metric improved → blocked.
    unsafe = beats_toro_navigation(
        better_breach_free_completion=False,
        fewer_geometry_breaches=False,
        path_length_overhead=0.0,
        travel_time_overhead=0.0,
        physics_contacts_increase=0,
        max_path_length_overhead=0.10,
        max_travel_time_overhead=0.10,
    )
    assert unsafe.passed is False
