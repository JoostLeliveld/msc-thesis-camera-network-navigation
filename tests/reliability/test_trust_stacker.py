from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "reliability"))

import math  # noqa: E402
import random  # noqa: E402

from reliability.confidence_calibration import CalibrationError  # noqa: E402
from reliability.contracts import ContractValidationError, LeakageError  # noqa: E402
from reliability.trust_stacker import (  # noqa: E402
    TRUST_FEATURE_ORDER,
    TrustFeatures,
    TrustStacker,
    split_groups,
)


TRUE_BETA = (0.2, 1.1, 0.9, 1.4, 0.7)  # matches TRUST_FEATURE_ORDER


def _sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _make_dataset(
    n: int = 2000, seed: int = 3
) -> tuple[list[TrustFeatures], list[int], list[str]]:
    """Synthetic data generated from the true stacker model."""

    rng = random.Random(seed)
    features: list[TrustFeatures] = []
    labels: list[int] = []
    groups: list[str] = []
    for idx in range(n):
        quality = 0.05 + 0.9 * rng.random()
        confidence = 0.05 + 0.9 * rng.random()
        health = 0.05 + 0.9 * rng.random()
        uncertainty = 2.0 * rng.random()
        z = (
            TRUE_BETA[0]
            + TRUE_BETA[1] * _logit(quality)
            + TRUE_BETA[2] * _logit(confidence)
            + TRUE_BETA[3] * (-uncertainty)
            + TRUE_BETA[4] * _logit(health)
        )
        features.append(
            TrustFeatures(
                spatial_quality=quality,
                spatial_uncertainty=uncertainty,
                calibrated_confidence=confidence,
                camera_health=health,
            )
        )
        labels.append(1 if rng.random() < _sigmoid(z) else 0)
        groups.append(f"run{idx % 8}")
    return features, labels, groups


# ---------------------------------------------------------------------------
# Fitting behaviour
# ---------------------------------------------------------------------------


def test_recovers_signs_of_known_coefficients() -> None:
    features, labels, groups = _make_dataset()
    stacker = TrustStacker().fit(features, labels, groups)
    coefficients = stacker.coefficients
    assert len(coefficients) == len(TRUST_FEATURE_ORDER)
    # All true slopes are positive in the design parameterization
    # (the uncertainty feature is already negated in the design row).
    assert coefficients[1] > 0.0  # logit(spatial_quality)
    assert coefficients[2] > 0.0  # logit(calibrated_confidence)
    assert coefficients[3] > 0.0  # -spatial_uncertainty
    assert coefficients[4] > 0.0  # logit(camera_health)
    for fitted, true in zip(coefficients[1:], TRUE_BETA[1:]):
        assert abs(fitted - true) < 0.5
    assert stacker.group_ids == tuple(sorted({f"run{i}" for i in range(8)}))


def test_monotone_response_in_spatial_quality() -> None:
    features, labels, groups = _make_dataset()
    stacker = TrustStacker().fit(features, labels, groups)
    qualities = [0.02 + 0.96 * i / 40.0 for i in range(41)]
    taus = [
        stacker.predict(
            TrustFeatures(
                spatial_quality=quality,
                spatial_uncertainty=0.5,
                calibrated_confidence=0.6,
                camera_health=0.9,
            )
        )
        for quality in qualities
    ]
    assert all(later >= earlier for earlier, later in zip(taus, taus[1:]))
    assert all(0.0 < tau < 1.0 for tau in taus)


def test_predictions_stay_in_open_interval_at_degenerate_inputs() -> None:
    features, labels, groups = _make_dataset(n=500, seed=6)
    stacker = TrustStacker().fit(features, labels, groups)
    for extreme in (
        TrustFeatures(1.0, 0.0, 1.0, 1.0),
        TrustFeatures(0.0, 5.0, 0.0, 0.0),
    ):
        tau = stacker.predict(extreme)
        assert 0.0 < tau < 1.0


def test_single_class_labels_raise() -> None:
    features, _, groups = _make_dataset(n=50, seed=1)
    with pytest.raises(ContractValidationError):
        TrustStacker().fit(features, [1] * len(features), groups)
    with pytest.raises(ContractValidationError):
        TrustStacker().fit(features, [0] * len(features), groups)


def test_fit_validates_lengths_and_groups() -> None:
    features, labels, groups = _make_dataset(n=20, seed=2)
    with pytest.raises(ContractValidationError):
        TrustStacker().fit(features, labels[:-1], groups)
    with pytest.raises(ContractValidationError):
        TrustStacker().fit(features, labels, groups[:-1])
    with pytest.raises(ContractValidationError):
        TrustStacker().fit([], [], [])
    with pytest.raises(CalibrationError):
        TrustStacker().predict(TrustFeatures(0.5, 0.5, 0.5, 0.5))


def test_trust_features_validation() -> None:
    with pytest.raises(ContractValidationError):
        TrustFeatures(1.5, 0.1, 0.5, 0.9)  # quality out of [0, 1]
    with pytest.raises(ContractValidationError):
        TrustFeatures(0.5, -0.1, 0.5, 0.9)  # negative uncertainty
    with pytest.raises(ContractValidationError):
        TrustFeatures(0.5, math.nan, 0.5, 0.9)  # non-finite
    with pytest.raises(ContractValidationError):
        TrustFeatures(0.5, 0.1, 0.5, 1.2)  # health out of [0, 1]


# ---------------------------------------------------------------------------
# Grouped split guard
# ---------------------------------------------------------------------------


def _small_grouped_dataset() -> tuple[list[TrustFeatures], list[int], list[str]]:
    features = [TrustFeatures(0.5, 0.5, 0.5, 0.9) for _ in range(6)]
    labels = [0, 1, 0, 1, 0, 1]
    groups = ["a", "a", "b", "b", "c", "c"]
    return features, labels, groups


def test_split_groups_partitions_by_group() -> None:
    features, labels, groups = _small_grouped_dataset()
    train, holdout = split_groups(features, labels, groups, holdout_groups=["c"])
    train_features, train_labels, train_groups = train
    holdout_features, holdout_labels, holdout_groups = holdout
    assert set(train_groups) == {"a", "b"}
    assert set(holdout_groups) == {"c"}
    assert len(train_features) == len(train_labels) == 4
    assert len(holdout_features) == len(holdout_labels) == 2
    assert not set(train_groups) & set(holdout_groups)


def test_split_groups_raises_on_missing_holdout_group() -> None:
    features, labels, groups = _small_grouped_dataset()
    with pytest.raises(ContractValidationError):
        split_groups(features, labels, groups, holdout_groups=["z"])


def test_split_groups_raises_on_train_holdout_overlap() -> None:
    features, labels, groups = _small_grouped_dataset()
    with pytest.raises(LeakageError):
        split_groups(
            features,
            labels,
            groups,
            holdout_groups=["c"],
            train_groups=["a", "c"],
        )


def test_split_groups_raises_when_no_train_groups_remain() -> None:
    features, labels, groups = _small_grouped_dataset()
    with pytest.raises(ContractValidationError):
        split_groups(features, labels, groups, holdout_groups=["a", "b", "c"])


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_serialization_round_trip_predicts_identically() -> None:
    features, labels, groups = _make_dataset(n=800, seed=8)
    stacker = TrustStacker().fit(features, labels, groups)
    payload = stacker.to_dict()
    assert payload["class"] == "TrustStacker"
    assert payload["feature_order"] == list(TRUST_FEATURE_ORDER)
    restored = TrustStacker.from_json(stacker.to_json())
    assert restored.coefficients == stacker.coefficients
    assert restored.group_ids == stacker.group_ids
    for probe in features[:20]:
        assert restored.predict(probe) == stacker.predict(probe)


def test_from_dict_rejects_wrong_class_or_feature_order() -> None:
    features, labels, groups = _make_dataset(n=200, seed=4)
    stacker = TrustStacker().fit(features, labels, groups)
    payload = stacker.to_dict()
    bad_class = dict(payload)
    bad_class["class"] = "LogisticCalibrator"
    with pytest.raises(ContractValidationError):
        TrustStacker.from_dict(bad_class)
    bad_order = dict(payload)
    bad_order["feature_order"] = list(reversed(TRUST_FEATURE_ORDER))
    with pytest.raises(ContractValidationError):
        TrustStacker.from_dict(bad_order)
