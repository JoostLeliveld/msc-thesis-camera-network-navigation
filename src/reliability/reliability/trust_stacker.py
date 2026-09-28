"""Trust stacker tau (multicamera plan 05).

Combines the spatial GP quality prior, calibrated per-frame confidence, GP
uncertainty, and camera health into one trust probability for the CURRENT
observation:

    tau = sigmoid( b0 + bq * logit(q_hat) + bc * logit(p_c)
                   - bu * sigma_q + bh * logit(h) )

This is a stacking model fitted by logistic IRLS on grouped data (never the
split that trained the GP or the calibrator), not a naive product — the GP
prior and calibrated confidence are correlated. Pure library code: stdlib +
numpy (via the shared IRLS core), no sklearn/scipy, no ROS.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence

from reliability.confidence_calibration import (
    LOGIT_EPS,
    CalibrationError,
    _as_list,
    _binary_labels,
    _check_payload_class,
    _clipped_logit,
    _finite,
    _fit_logistic_irls,
    _open_interval_probability,
    _payload_coefficients,
    _payload_eps,
    _probability,
    _sigmoid,
)
from reliability.contracts import ContractValidationError, LeakageError


TRUST_FEATURE_ORDER = (
    "intercept",
    "logit_spatial_quality",
    "logit_calibrated_confidence",
    "neg_spatial_uncertainty",
    "logit_camera_health",
)
"""Design-row order used by TrustStacker: coefficients[k] multiplies the
k-th entry of [1, logit(spatial_quality), logit(calibrated_confidence),
-spatial_uncertainty, logit(camera_health)] (logits eps-clipped)."""


@dataclass(frozen=True)
class TrustFeatures:
    """Per-observation inputs to the trust stacker.

    - spatial_quality: GP quality-prior mean q_hat(s) in [0, 1]
    - spatial_uncertainty: GP predictive std sigma_q(s), finite and >= 0
    - calibrated_confidence: p_c from the confidence calibrator, in [0, 1]
    - camera_health: h from the camera-health layer, in [0, 1]
      (use 1.0 until the health layer lands)
    """

    spatial_quality: float
    spatial_uncertainty: float
    calibrated_confidence: float
    camera_health: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "spatial_quality", _probability(self.spatial_quality, "spatial_quality")
        )
        uncertainty = _finite(self.spatial_uncertainty, "spatial_uncertainty")
        if uncertainty < 0.0:
            raise ContractValidationError(
                f"spatial_uncertainty must be >= 0, got {uncertainty}"
            )
        object.__setattr__(self, "spatial_uncertainty", uncertainty)
        object.__setattr__(
            self,
            "calibrated_confidence",
            _probability(self.calibrated_confidence, "calibrated_confidence"),
        )
        object.__setattr__(
            self, "camera_health", _probability(self.camera_health, "camera_health")
        )


def _validate_features(features: Any, field_name: str = "features") -> list[TrustFeatures]:
    out: list[TrustFeatures] = []
    for idx, item in enumerate(_as_list(features, field_name)):
        if not isinstance(item, TrustFeatures):
            raise ContractValidationError(
                f"{field_name}[{idx}] must be a TrustFeatures instance, got {type(item).__name__}"
            )
        out.append(item)
    return out


def _group_strings(groups: Any, field_name: str = "groups") -> list[str]:
    out: list[str] = []
    for idx, item in enumerate(_as_list(groups, field_name)):
        text = str(item).strip()
        if not text:
            raise ContractValidationError(f"{field_name}[{idx}] must be a non-empty group id")
        out.append(text)
    return out


class TrustStacker:
    """Logistic stacker producing a trust probability tau in (0, 1).

    Feature map per observation (order = TRUST_FEATURE_ORDER):
    ``[1, logit(spatial_quality), logit(calibrated_confidence),
    -spatial_uncertainty, logit(camera_health)]`` with eps-clipped logits so
    degenerate 0/1 inputs stay finite.
    """

    def __init__(self, *, eps: float = LOGIT_EPS, max_iterations: int = 100) -> None:
        eps = _finite(eps, "eps")
        if not 0.0 < eps < 0.5:
            raise ContractValidationError(f"eps must be in (0, 0.5), got {eps}")
        max_iterations = int(max_iterations)
        if max_iterations < 1:
            raise ContractValidationError("max_iterations must be >= 1")
        self._eps = eps
        self._max_iterations = max_iterations
        self._coefficients: tuple[float, ...] | None = None
        self._group_ids: tuple[str, ...] = ()

    @property
    def coefficients(self) -> tuple[float, ...]:
        """Fitted coefficients in TRUST_FEATURE_ORDER (5 values)."""

        if self._coefficients is None:
            raise CalibrationError("TrustStacker is not fitted")
        return self._coefficients

    @property
    def group_ids(self) -> tuple[str, ...]:
        """Sorted unique group ids seen during fit (fit provenance)."""

        return self._group_ids

    def _design_row(self, features: TrustFeatures) -> list[float]:
        return [
            1.0,
            _clipped_logit(features.spatial_quality, self._eps),
            _clipped_logit(features.calibrated_confidence, self._eps),
            -features.spatial_uncertainty,
            _clipped_logit(features.camera_health, self._eps),
        ]

    def fit(
        self,
        features: Sequence[TrustFeatures],
        labels: Sequence[float],
        groups: Sequence[Any],
    ) -> "TrustStacker":
        feats = _validate_features(features)
        ys = _binary_labels(labels)
        group_ids = _group_strings(groups)
        if not feats:
            raise ContractValidationError("TrustStacker.fit: features must be non-empty")
        if len(feats) != len(ys):
            raise ContractValidationError(
                f"TrustStacker.fit: features and labels must have equal length, "
                f"got {len(feats)} vs {len(ys)}"
            )
        if len(group_ids) != len(ys):
            raise ContractValidationError(
                f"TrustStacker.fit: groups and labels must have equal length, "
                f"got {len(group_ids)} vs {len(ys)}"
            )
        unique = set(ys)
        if unique != {0.0, 1.0}:
            raise ContractValidationError(
                f"TrustStacker.fit: labels must contain both classes (0 and 1), "
                f"got only {sorted(unique)}"
            )
        rows = [self._design_row(item) for item in feats]
        self._coefficients = _fit_logistic_irls(rows, ys, max_iterations=self._max_iterations)
        self._group_ids = tuple(sorted(set(group_ids)))
        return self

    def predict(self, features: TrustFeatures) -> float:
        coefficients = self.coefficients
        if not isinstance(features, TrustFeatures):
            raise ContractValidationError(
                f"predict expects a TrustFeatures instance, got {type(features).__name__}"
            )
        row = self._design_row(features)
        z = sum(coef * value for coef, value in zip(coefficients, row))
        return _open_interval_probability(_sigmoid(z))

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": type(self).__name__,
            "coefficients": [float(value) for value in self.coefficients],
            "eps": float(self._eps),
            "feature_order": list(TRUST_FEATURE_ORDER),
            "group_ids": list(self._group_ids),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrustStacker":
        _check_payload_class(payload, cls.__name__)
        feature_order = payload.get("feature_order", list(TRUST_FEATURE_ORDER))
        if tuple(feature_order) != TRUST_FEATURE_ORDER:
            raise ContractValidationError(
                f"{cls.__name__} payload feature_order {tuple(feature_order)!r} does not match "
                f"{TRUST_FEATURE_ORDER!r}"
            )
        out = cls(eps=_payload_eps(payload, cls.__name__))
        out._coefficients = _payload_coefficients(
            payload, len(TRUST_FEATURE_ORDER), cls.__name__
        )
        out._group_ids = tuple(
            sorted({str(item) for item in _as_list(payload.get("group_ids", []), "group_ids")})
        )
        return out

    @classmethod
    def from_json(cls, text: str) -> "TrustStacker":
        return cls.from_dict(json.loads(text))


def split_groups(
    features: Sequence[TrustFeatures],
    labels: Sequence[float],
    groups: Sequence[Any],
    holdout_groups: Sequence[Any],
    train_groups: Sequence[Any] | None = None,
) -> tuple[
    tuple[tuple[TrustFeatures, ...], tuple[float, ...], tuple[str, ...]],
    tuple[tuple[TrustFeatures, ...], tuple[float, ...], tuple[str, ...]],
]:
    """Grouped train/holdout split with leakage guards.

    Returns ``((train_features, train_labels, train_groups),
    (holdout_features, holdout_labels, holdout_groups))``.

    - Raises ``ContractValidationError`` if any requested holdout (or train)
      group id is absent from ``groups``, or if either side ends up empty.
    - Raises ``LeakageError`` if explicit ``train_groups`` overlap
      ``holdout_groups``. When ``train_groups`` is None, train = all groups
      not in the holdout set (disjoint by construction).
    """

    feats = _validate_features(features)
    ys = _binary_labels(labels)
    group_ids = _group_strings(groups)
    if not feats:
        raise ContractValidationError("split_groups: features must be non-empty")
    if len(feats) != len(ys) or len(group_ids) != len(ys):
        raise ContractValidationError(
            "split_groups: features, labels, and groups must have equal length, got "
            f"{len(feats)}/{len(ys)}/{len(group_ids)}"
        )

    available = set(group_ids)
    holdout = set(_group_strings(holdout_groups, "holdout_groups"))
    missing_holdout = sorted(holdout - available)
    if missing_holdout:
        raise ContractValidationError(
            "split_groups: holdout groups absent from data: " + ", ".join(missing_holdout)
        )

    if train_groups is None:
        train = available - holdout
    else:
        train = set(_group_strings(train_groups, "train_groups"))
        overlap = sorted(train & holdout)
        if overlap:
            raise LeakageError(
                "split_groups: train and holdout groups overlap: " + ", ".join(overlap)
            )
        missing_train = sorted(train - available)
        if missing_train:
            raise ContractValidationError(
                "split_groups: train groups absent from data: " + ", ".join(missing_train)
            )
    if not train:
        raise ContractValidationError("split_groups: no training groups remain after holdout")

    train_idx = [idx for idx, group in enumerate(group_ids) if group in train]
    holdout_idx = [idx for idx, group in enumerate(group_ids) if group in holdout]
    if not holdout_idx:
        raise ContractValidationError("split_groups: holdout selection is empty")

    def _take(indices: list[int]) -> tuple[
        tuple[TrustFeatures, ...], tuple[float, ...], tuple[str, ...]
    ]:
        return (
            tuple(feats[idx] for idx in indices),
            tuple(ys[idx] for idx in indices),
            tuple(group_ids[idx] for idx in indices),
        )

    return (_take(train_idx), _take(holdout_idx))
