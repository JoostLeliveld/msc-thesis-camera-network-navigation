from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "reliability"))

import math  # noqa: E402
import random  # noqa: E402

from reliability.confidence_calibration import (  # noqa: E402
    COEFFICIENT_CAP,
    CalibrationError,
    IsotonicCalibrator,
    LogisticCalibrator,
    MULTIVARIATE_FEATURE_ORDER,
    MultivariateLogisticCalibrator,
    reliability_curve,
)
from reliability.contracts import ContractValidationError  # noqa: E402


def _brute_force_isotonic(labels: list[float]) -> list[float]:
    """Closed-form isotonic regression: f_i = max_{j<=i} min_{k>=i} mean(y_j..y_k)."""

    n = len(labels)
    fits = []
    for i in range(n):
        best = -math.inf
        for j in range(i + 1):
            worst = math.inf
            for k in range(i, n):
                segment = labels[j : k + 1]
                worst = min(worst, sum(segment) / len(segment))
            best = max(best, worst)
        fits.append(best)
    return fits


def _sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


# ---------------------------------------------------------------------------
# IsotonicCalibrator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "labels",
    [
        [0, 1, 0, 1],
        [1, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1, 0, 1, 0, 1],
        [1],
        [0, 1, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 0],
    ],
)
def test_pav_matches_brute_force(labels: list[int]) -> None:
    scores = [(i + 1) / (len(labels) + 1) for i in range(len(labels))]
    calibrator = IsotonicCalibrator().fit(scores, labels)
    expected = _brute_force_isotonic([float(label) for label in labels])
    got = [calibrator.predict(score) for score in scores]
    assert got == pytest.approx(expected)


def test_isotonic_monotone_predictions_and_clamping() -> None:
    rng = random.Random(7)
    scores = [rng.random() for _ in range(300)]
    labels = [1 if rng.random() < score else 0 for score in scores]
    calibrator = IsotonicCalibrator().fit(scores, labels)

    grid = [i / 200.0 for i in range(201)]
    predictions = [calibrator.predict(value) for value in grid]
    assert all(later >= earlier for earlier, later in zip(predictions, predictions[1:]))
    assert all(0.0 <= p <= 1.0 for p in predictions)

    breakpoints = calibrator.breakpoints
    # Below/above the fitted range clamps to first/last block value.
    assert calibrator.predict(-10.0) == breakpoints[0][1]
    assert calibrator.predict(10.0) == breakpoints[-1][1]
    # Breakpoints are strictly increasing in score and value.
    bp_scores = [score for score, _ in breakpoints]
    bp_values = [value for _, value in breakpoints]
    assert bp_scores == sorted(set(bp_scores))
    assert all(later > earlier for earlier, later in zip(bp_values, bp_values[1:]))


def test_isotonic_pools_tied_scores() -> None:
    calibrator = IsotonicCalibrator().fit([0.3, 0.3, 0.7], [0, 1, 1])
    assert calibrator.predict(0.3) == pytest.approx(0.5)
    assert calibrator.predict(0.7) == pytest.approx(1.0)


def test_isotonic_rejects_bad_inputs() -> None:
    with pytest.raises(ContractValidationError):
        IsotonicCalibrator().fit([], [])
    with pytest.raises(ContractValidationError):
        IsotonicCalibrator().fit([0.1, 0.2], [1])
    with pytest.raises(ContractValidationError):
        IsotonicCalibrator().fit([0.1, 0.2], [0, 0.5])
    with pytest.raises(ContractValidationError):
        IsotonicCalibrator().fit([0.1, math.nan], [0, 1])
    with pytest.raises(CalibrationError):
        IsotonicCalibrator().predict(0.5)


# ---------------------------------------------------------------------------
# LogisticCalibrator
# ---------------------------------------------------------------------------


def test_logistic_recovers_known_coefficients() -> None:
    rng = random.Random(0)
    true_a0, true_a1 = -0.4, 1.3
    scores: list[float] = []
    labels: list[int] = []
    for _ in range(2000):
        c = 0.05 + 0.9 * rng.random()
        p = _sigmoid(true_a0 + true_a1 * _logit(c))
        scores.append(c)
        labels.append(1 if rng.random() < p else 0)
    calibrator = LogisticCalibrator().fit(scores, labels)
    a0, a1 = calibrator.coefficients
    assert abs(a0 - true_a0) < 0.15
    assert abs(a1 - true_a1) < 0.15


def test_logistic_perfect_separation_is_deterministic() -> None:
    scores = [0.05, 0.1, 0.2, 0.8, 0.9, 0.95]
    labels = [0, 0, 0, 1, 1, 1]
    first = LogisticCalibrator().fit(scores, labels)
    second = LogisticCalibrator().fit(scores, labels)
    assert first.coefficients == second.coefficients
    assert all(math.isfinite(value) for value in first.coefficients)
    assert all(abs(value) <= COEFFICIENT_CAP for value in first.coefficients)
    for score in (0.01, 0.5, 0.99):
        p = first.predict(score)
        assert 0.0 < p < 1.0
    assert first.predict(0.9) > first.predict(0.1)


def test_logistic_single_class_raises() -> None:
    with pytest.raises(ContractValidationError):
        LogisticCalibrator().fit([0.2, 0.5, 0.8], [1, 1, 1])
    with pytest.raises(ContractValidationError):
        LogisticCalibrator().fit([0.2, 0.5, 0.8], [0, 0, 0])


def test_logistic_rejects_bad_inputs() -> None:
    with pytest.raises(ContractValidationError):
        LogisticCalibrator().fit([], [])
    with pytest.raises(ContractValidationError):
        LogisticCalibrator().fit([0.1, 1.2], [0, 1])  # score outside [0, 1]
    with pytest.raises(ContractValidationError):
        LogisticCalibrator().fit([0.1, 0.9], [0, 2])  # non-binary label
    with pytest.raises(CalibrationError):
        LogisticCalibrator().predict(0.5)


# ---------------------------------------------------------------------------
# MultivariateLogisticCalibrator
# ---------------------------------------------------------------------------


def _make_multivariate_dataset(
    n: int, seed: int
) -> tuple[list[tuple[float, float, float, float, float]], list[int]]:
    rng = random.Random(seed)
    true = (0.3, 1.0, 0.4, -0.8, 0.5, -0.2)
    features: list[tuple[float, float, float, float, float]] = []
    labels: list[int] = []
    for _ in range(n):
        confidence = 0.05 + 0.9 * rng.random()
        bbox_area = math.exp(rng.uniform(4.0, 8.0))
        u_hat = rng.uniform(-1.0, 1.0)
        v_hat = rng.uniform(-1.0, 1.0)
        range_m = rng.uniform(2.0, 10.0)
        z = (
            true[0]
            + true[1] * _logit(confidence)
            + true[2] * (math.log(bbox_area) - 6.0)
            + true[3] * u_hat
            + true[4] * v_hat
            + true[5] * (range_m - 6.0)
        )
        features.append((confidence, bbox_area, u_hat, v_hat, range_m))
        labels.append(1 if rng.random() < _sigmoid(z) else 0)
    return features, labels


def test_multivariate_fits_and_recovers_slope_signs() -> None:
    features, labels = _make_multivariate_dataset(2000, seed=11)
    calibrator = MultivariateLogisticCalibrator().fit(features, labels)
    coefficients = calibrator.coefficients
    assert len(coefficients) == 6
    # Slope signs of the generating model: +logit(c), +log(area), -u, +v, -range.
    assert coefficients[1] > 0.0
    assert coefficients[2] > 0.0
    assert coefficients[3] < 0.0
    assert coefficients[4] > 0.0
    assert coefficients[5] < 0.0
    assert abs(coefficients[1] - 1.0) < 0.15


def test_multivariate_dict_and_tuple_features_are_equivalent() -> None:
    features, labels = _make_multivariate_dataset(200, seed=5)
    dict_features = [dict(zip(MULTIVARIATE_FEATURE_ORDER, item)) for item in features]
    from_tuples = MultivariateLogisticCalibrator().fit(features, labels)
    from_dicts = MultivariateLogisticCalibrator().fit(dict_features, labels)
    assert from_tuples.coefficients == pytest.approx(from_dicts.coefficients)
    probe = features[0]
    assert from_tuples.predict(probe) == pytest.approx(
        from_dicts.predict(dict(zip(MULTIVARIATE_FEATURE_ORDER, probe)))
    )


def test_multivariate_rejects_bad_inputs() -> None:
    good = (0.7, 500.0, 0.1, -0.2, 5.0)
    with pytest.raises(ContractValidationError):
        MultivariateLogisticCalibrator().fit([good, good], [1, 1])  # single class
    with pytest.raises(ContractValidationError):
        MultivariateLogisticCalibrator().fit([(0.7, -1.0, 0.1, -0.2, 5.0), good], [0, 1])  # area <= 0
    with pytest.raises(ContractValidationError):
        MultivariateLogisticCalibrator().fit([(0.7, 500.0, 0.1, -0.2), good], [0, 1])  # short tuple
    with pytest.raises(ContractValidationError):
        MultivariateLogisticCalibrator().fit(
            [{"confidence": 0.7, "bbox_area": 500.0}, good], [0, 1]
        )  # missing keys
    with pytest.raises(CalibrationError):
        MultivariateLogisticCalibrator().predict(good)


# ---------------------------------------------------------------------------
# Serialization round-trips
# ---------------------------------------------------------------------------


def test_isotonic_serialization_round_trip() -> None:
    rng = random.Random(3)
    scores = [rng.random() for _ in range(120)]
    labels = [1 if rng.random() < score else 0 for score in scores]
    calibrator = IsotonicCalibrator().fit(scores, labels)
    restored = IsotonicCalibrator.from_json(calibrator.to_json())
    assert restored.breakpoints == calibrator.breakpoints
    for value in [i / 50.0 for i in range(51)] + [-1.0, 2.0]:
        assert restored.predict(value) == calibrator.predict(value)


def test_logistic_serialization_round_trip() -> None:
    rng = random.Random(4)
    scores = [0.05 + 0.9 * rng.random() for _ in range(200)]
    labels = [1 if rng.random() < score else 0 for score in scores]
    calibrator = LogisticCalibrator().fit(scores, labels)
    payload = calibrator.to_dict()
    assert payload["class"] == "LogisticCalibrator"
    restored = LogisticCalibrator.from_json(calibrator.to_json())
    assert restored.coefficients == calibrator.coefficients
    for value in [i / 20.0 for i in range(21)]:
        assert restored.predict(value) == calibrator.predict(value)


def test_multivariate_serialization_round_trip() -> None:
    features, labels = _make_multivariate_dataset(300, seed=9)
    calibrator = MultivariateLogisticCalibrator().fit(features, labels)
    payload = calibrator.to_dict()
    assert payload["class"] == "MultivariateLogisticCalibrator"
    assert payload["feature_order"] == list(MULTIVARIATE_FEATURE_ORDER)
    restored = MultivariateLogisticCalibrator.from_json(calibrator.to_json())
    assert restored.coefficients == calibrator.coefficients
    for probe in features[:10]:
        assert restored.predict(probe) == calibrator.predict(probe)


def test_from_dict_rejects_wrong_class() -> None:
    calibrator = LogisticCalibrator().fit([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    payload = calibrator.to_dict()
    with pytest.raises(ContractValidationError):
        MultivariateLogisticCalibrator.from_dict(payload)
    with pytest.raises(ContractValidationError):
        IsotonicCalibrator.from_dict(payload)


# ---------------------------------------------------------------------------
# reliability_curve
# ---------------------------------------------------------------------------


def test_reliability_curve_hand_case() -> None:
    scores = [0.05, 0.15, 0.18, 0.95, 1.0]
    labels = [0, 1, 0, 1, 1]
    curve = reliability_curve(scores, labels, n_bins=10)
    assert len(curve) == 3  # empty bins skipped
    assert curve[0] == pytest.approx((0.05, 0.05, 0.0, 1))
    assert curve[1] == pytest.approx((0.15, (0.15 + 0.18) / 2.0, 0.5, 2))
    # score == 1.0 falls into the last bin
    assert curve[2] == pytest.approx((0.95, (0.95 + 1.0) / 2.0, 1.0, 2))
    assert sum(entry[3] for entry in curve) == len(scores)


def test_reliability_curve_rejects_bad_inputs() -> None:
    with pytest.raises(ContractValidationError):
        reliability_curve([], [], n_bins=10)
    with pytest.raises(ContractValidationError):
        reliability_curve([0.5], [1], n_bins=0)
    with pytest.raises(ContractValidationError):
        reliability_curve([0.5, 0.6], [1], n_bins=10)
    with pytest.raises(ContractValidationError):
        reliability_curve([0.5, 1.4], [1, 0], n_bins=10)
