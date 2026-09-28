"""Detector-confidence calibration (multicamera plan 04).

Maps a raw detector confidence ``c`` to a calibrated usability probability
``p = P(usable | detection, c)``. Raw confidence is never used directly as a
probability. Pure library code: stdlib + numpy only (no sklearn/scipy, no ROS).

Calibrators:

- ``IsotonicCalibrator`` — monotone (nondecreasing) step function fitted with
  the pool-adjacent-violators algorithm; stepwise-constant prediction with
  clamping outside the fitted score range.
- ``LogisticCalibrator`` — ``logit(p) = a0 + a1 * logit(c)`` fitted by
  Newton-Raphson / IRLS on eps-clipped logits.
- ``MultivariateLogisticCalibrator`` — the same IRLS core over the feature
  vector ``[1, logit(c), log(bbox_area), u_hat, v_hat, range_m]``; tests
  whether raw confidence stays informative once geometry is known.

All calibrators serialize to/from JSON dict payloads that include the class
name and fitted parameters, so artifacts are self-describing and round-trip.
"""

from __future__ import annotations

import bisect
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from reliability.contracts import ContractValidationError


LOGIT_EPS = 1.0e-6
"""Default clip for logits: probabilities are clipped to [eps, 1 - eps]."""

COEFFICIENT_CAP = 30.0
"""Absolute bound on IRLS coefficients. Under (quasi-)perfect separation the
logistic MLE does not exist; iterates are clipped to this bound and the capped
solution is returned deterministically instead of diverging."""

MULTIVARIATE_FEATURE_ORDER = ("confidence", "bbox_area", "u_hat", "v_hat", "range_m")
"""Input order for MultivariateLogisticCalibrator tuple features (and required
keys for dict features). The internal design row is
[1, logit(confidence), log(bbox_area), u_hat, v_hat, range_m]."""


class CalibrationError(ValueError):
    """Raised when a calibrator cannot be fitted (non-convergence) or used
    before fitting."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _finite(value: Any, field_name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"{field_name} must be numeric, got {value!r}") from exc
    if not math.isfinite(out):
        raise ContractValidationError(f"{field_name} must be finite, got {value!r}")
    return out


def _probability(value: Any, field_name: str) -> float:
    out = _finite(value, field_name)
    if not 0.0 <= out <= 1.0:
        raise ContractValidationError(f"{field_name} must be in [0, 1], got {out}")
    return out


def _as_list(values: Any, field_name: str) -> list[Any]:
    if isinstance(values, (str, bytes)):
        raise ContractValidationError(f"{field_name} must be a sequence, got {values!r}")
    if hasattr(values, "tolist"):
        values = values.tolist()
    try:
        return list(values)
    except TypeError as exc:
        raise ContractValidationError(f"{field_name} must be a sequence, got {values!r}") from exc


def _finite_scores(scores: Any, field_name: str = "scores") -> list[float]:
    return [_finite(item, f"{field_name}[{idx}]") for idx, item in enumerate(_as_list(scores, field_name))]


def _probability_scores(scores: Any, field_name: str = "scores") -> list[float]:
    return [_probability(item, f"{field_name}[{idx}]") for idx, item in enumerate(_as_list(scores, field_name))]


def _binary_labels(labels: Any, field_name: str = "labels") -> list[float]:
    out: list[float] = []
    for idx, item in enumerate(_as_list(labels, field_name)):
        value = _finite(item, f"{field_name}[{idx}]")
        if value not in (0.0, 1.0):
            raise ContractValidationError(f"{field_name}[{idx}] must be binary (0 or 1), got {item!r}")
        out.append(value)
    return out


def _check_paired(xs: Sequence[float], ys: Sequence[float], context: str) -> None:
    if len(xs) != len(ys):
        raise ContractValidationError(
            f"{context}: scores and labels must have equal length, got {len(xs)} vs {len(ys)}"
        )
    if not xs:
        raise ContractValidationError(f"{context}: scores and labels must be non-empty")


def _require_both_classes(labels: Sequence[float], context: str) -> None:
    unique = set(labels)
    if unique != {0.0, 1.0}:
        raise ContractValidationError(
            f"{context}: labels must contain both classes (0 and 1), got only {sorted(unique)}"
        )


def _clipped_logit(probability: float, eps: float = LOGIT_EPS) -> float:
    p = min(max(probability, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


_PROBABILITY_FLOOR = 1.0e-12


def _open_interval_probability(p: float) -> float:
    """Keep predicted probabilities strictly inside (0, 1) even when the
    sigmoid saturates in floating point (e.g. capped separation fits)."""

    return min(max(p, _PROBABILITY_FLOOR), 1.0 - _PROBABILITY_FLOOR)


# ---------------------------------------------------------------------------
# Shared IRLS core
# ---------------------------------------------------------------------------


def _fit_logistic_irls(
    design: Sequence[Sequence[float]],
    labels: Sequence[float],
    *,
    max_iterations: int = 100,
    tolerance: float = 1.0e-8,
    coefficient_cap: float = COEFFICIENT_CAP,
    ridge: float = 1.0e-8,
    saturation_tolerance: float = 1.0e-6,
) -> tuple[float, ...]:
    """Newton-Raphson / IRLS for logistic regression.

    Deterministic separation handling (perfect separation has no finite MLE):
    - if every training probability matches its label to within
      ``saturation_tolerance``, the current iterate is accepted;
    - iterates are clipped elementwise to ``[-coefficient_cap,
      coefficient_cap]`` and a cap-saturated iterate is returned immediately.

    Raises ``CalibrationError`` when the step does not shrink below
    ``tolerance`` within ``max_iterations``.
    """

    X = np.asarray(design, dtype=float)
    y = np.asarray(labels, dtype=float)
    if X.ndim != 2 or X.shape[0] != y.shape[0] or X.shape[0] == 0:
        raise ContractValidationError("IRLS design and labels must be non-empty with matching rows")
    n_features = X.shape[1]
    beta = np.zeros(n_features, dtype=float)
    for _ in range(int(max_iterations)):
        z = X @ beta
        p = 0.5 * (1.0 + np.tanh(0.5 * z))  # numerically stable sigmoid
        if float(np.max(np.abs(y - p))) < saturation_tolerance:
            # (Quasi-)perfect separation: the fit already reproduces every
            # label; stop deterministically instead of diverging.
            return tuple(float(value) for value in beta)
        weights = p * (1.0 - p)
        gradient = X.T @ (y - p)
        hessian = X.T @ (X * weights[:, None]) + ridge * np.eye(n_features)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as exc:
            raise CalibrationError("IRLS normal equations are singular") from exc
        new_beta = np.clip(beta + step, -coefficient_cap, coefficient_cap)
        if bool(np.any(np.abs(new_beta) >= coefficient_cap)):
            return tuple(float(value) for value in new_beta)
        if float(np.max(np.abs(new_beta - beta))) < tolerance:
            return tuple(float(value) for value in new_beta)
        beta = new_beta
    raise CalibrationError(f"IRLS did not converge within {max_iterations} iterations")


# ---------------------------------------------------------------------------
# Isotonic calibrator (pool-adjacent-violators)
# ---------------------------------------------------------------------------


class IsotonicCalibrator:
    """Nondecreasing step-function calibrator fitted with PAV.

    ``predict`` is stepwise-constant: a query score takes the value of the
    fitted block whose start score is the largest one <= the query. Queries
    below the smallest fitted score clamp to the first block value; queries
    above the largest clamp to the last.
    """

    def __init__(self) -> None:
        self._breakpoints: tuple[tuple[float, float], ...] | None = None
        self._starts: list[float] = []

    @property
    def breakpoints(self) -> tuple[tuple[float, float], ...]:
        """Fitted (block_start_score, calibrated_probability) pairs, with
        strictly increasing scores and strictly increasing probabilities."""

        if self._breakpoints is None:
            raise CalibrationError("IsotonicCalibrator is not fitted")
        return self._breakpoints

    def fit(self, scores: Sequence[float], labels: Sequence[float]) -> "IsotonicCalibrator":
        xs = _finite_scores(scores)
        ys = _binary_labels(labels)
        _check_paired(xs, ys, "IsotonicCalibrator.fit")

        order = sorted(range(len(xs)), key=lambda idx: xs[idx])
        # Pool tied scores first so the fit is a function of the score.
        tied: list[list[float]] = []  # [score, label_sum, weight]
        for idx in order:
            score, label = xs[idx], ys[idx]
            if tied and tied[-1][0] == score:
                tied[-1][1] += label
                tied[-1][2] += 1.0
            else:
                tied.append([score, label, 1.0])

        # Pool-adjacent-violators: merge while the previous block mean is not
        # strictly below the current one.
        blocks: list[list[float]] = []  # [start_score, label_sum, weight]
        for score, label_sum, weight in tied:
            blocks.append([score, label_sum, weight])
            while len(blocks) >= 2 and (
                blocks[-2][1] * blocks[-1][2] >= blocks[-1][1] * blocks[-2][2]
            ):
                _, merged_sum, merged_weight = blocks.pop()
                blocks[-1][1] += merged_sum
                blocks[-1][2] += merged_weight

        self._set_breakpoints(
            tuple((start, label_sum / weight) for start, label_sum, weight in blocks)
        )
        return self

    def _set_breakpoints(self, breakpoints: tuple[tuple[float, float], ...]) -> None:
        self._breakpoints = breakpoints
        self._starts = [start for start, _ in breakpoints]

    def predict(self, score: float) -> float:
        breakpoints = self.breakpoints
        value = _finite(score, "score")
        idx = bisect.bisect_right(self._starts, value) - 1
        if idx < 0:
            idx = 0
        return breakpoints[idx][1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": type(self).__name__,
            "breakpoints": [[float(score), float(prob)] for score, prob in self.breakpoints],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IsotonicCalibrator":
        _check_payload_class(payload, cls.__name__)
        raw = payload.get("breakpoints")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
            raise ContractValidationError("IsotonicCalibrator payload requires non-empty breakpoints")
        breakpoints: list[tuple[float, float]] = []
        for idx, item in enumerate(raw):
            pair = _as_list(item, f"breakpoints[{idx}]")
            if len(pair) != 2:
                raise ContractValidationError(f"breakpoints[{idx}] must be a (score, probability) pair")
            score = _finite(pair[0], f"breakpoints[{idx}][0]")
            prob = _probability(pair[1], f"breakpoints[{idx}][1]")
            if breakpoints and (score <= breakpoints[-1][0] or prob < breakpoints[-1][1]):
                raise ContractValidationError(
                    "breakpoints must have strictly increasing scores and nondecreasing probabilities"
                )
            breakpoints.append((score, prob))
        out = cls()
        out._set_breakpoints(tuple(breakpoints))
        return out

    @classmethod
    def from_json(cls, text: str) -> "IsotonicCalibrator":
        return cls.from_dict(json.loads(text))


# ---------------------------------------------------------------------------
# Logistic calibrators
# ---------------------------------------------------------------------------


def _check_payload_class(payload: Mapping[str, Any], expected: str) -> None:
    if not isinstance(payload, Mapping):
        raise ContractValidationError(f"{expected} payload must be a mapping")
    found = payload.get("class")
    if found != expected:
        raise ContractValidationError(f"{expected} payload has class {found!r}, expected {expected!r}")


def _payload_coefficients(payload: Mapping[str, Any], expected_count: int, context: str) -> tuple[float, ...]:
    raw = _as_list(payload.get("coefficients"), f"{context}.coefficients")
    if len(raw) != expected_count:
        raise ContractValidationError(
            f"{context} payload requires {expected_count} coefficients, got {len(raw)}"
        )
    return tuple(_finite(item, f"{context}.coefficients[{idx}]") for idx, item in enumerate(raw))


def _payload_eps(payload: Mapping[str, Any], context: str) -> float:
    eps = _finite(payload.get("eps", LOGIT_EPS), f"{context}.eps")
    if not 0.0 < eps < 0.5:
        raise ContractValidationError(f"{context}.eps must be in (0, 0.5), got {eps}")
    return eps


class LogisticCalibrator:
    """Platt-style calibrator: ``logit(p) = a0 + a1 * logit(c)``.

    ``c`` is clipped to ``[eps, 1 - eps]`` before the logit. Fitting requires
    both label classes; the IRLS core caps iterations and either converges,
    returns a deterministic coefficient-capped solution under perfect
    separation, or raises ``CalibrationError``.
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

    @property
    def coefficients(self) -> tuple[float, float]:
        """(a0, a1) for logit(p) = a0 + a1 * logit(c)."""

        if self._coefficients is None:
            raise CalibrationError("LogisticCalibrator is not fitted")
        return (self._coefficients[0], self._coefficients[1])

    def fit(self, scores: Sequence[float], labels: Sequence[float]) -> "LogisticCalibrator":
        xs = _probability_scores(scores)
        ys = _binary_labels(labels)
        _check_paired(xs, ys, "LogisticCalibrator.fit")
        _require_both_classes(ys, "LogisticCalibrator.fit")
        design = [[1.0, _clipped_logit(score, self._eps)] for score in xs]
        self._coefficients = _fit_logistic_irls(design, ys, max_iterations=self._max_iterations)
        return self

    def predict(self, score: float) -> float:
        a0, a1 = self.coefficients
        c = _probability(score, "score")
        return _open_interval_probability(_sigmoid(a0 + a1 * _clipped_logit(c, self._eps)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": type(self).__name__,
            "coefficients": [float(value) for value in self.coefficients],
            "eps": float(self._eps),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LogisticCalibrator":
        _check_payload_class(payload, cls.__name__)
        out = cls(eps=_payload_eps(payload, cls.__name__))
        out._coefficients = _payload_coefficients(payload, 2, cls.__name__)
        return out

    @classmethod
    def from_json(cls, text: str) -> "LogisticCalibrator":
        return cls.from_dict(json.loads(text))


class MultivariateLogisticCalibrator:
    """IRLS logistic calibrator over confidence + detection geometry.

    Design row (in this exact order):
    ``[1, logit(confidence), log(bbox_area), u_hat, v_hat, range_m]``.

    ``fit``/``predict`` accept features either as mappings with the keys in
    ``MULTIVARIATE_FEATURE_ORDER`` (``confidence``, ``bbox_area``, ``u_hat``,
    ``v_hat``, ``range_m``) or as 5-tuples in that same order.
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

    @property
    def coefficients(self) -> tuple[float, ...]:
        """Six coefficients matching [1, logit(c), log(bbox_area), u_hat, v_hat, range_m]."""

        if self._coefficients is None:
            raise CalibrationError("MultivariateLogisticCalibrator is not fitted")
        return self._coefficients

    def _design_row(self, feature: Any, field_name: str) -> list[float]:
        if isinstance(feature, Mapping):
            missing = [key for key in MULTIVARIATE_FEATURE_ORDER if key not in feature]
            if missing:
                raise ContractValidationError(
                    f"{field_name} is missing feature keys: {', '.join(missing)}"
                )
            values = [feature[key] for key in MULTIVARIATE_FEATURE_ORDER]
        else:
            values = _as_list(feature, field_name)
            if len(values) != len(MULTIVARIATE_FEATURE_ORDER):
                raise ContractValidationError(
                    f"{field_name} must have {len(MULTIVARIATE_FEATURE_ORDER)} entries in order "
                    f"{MULTIVARIATE_FEATURE_ORDER}, got {len(values)}"
                )
        confidence = _probability(values[0], f"{field_name}.confidence")
        bbox_area = _finite(values[1], f"{field_name}.bbox_area")
        if bbox_area <= 0.0:
            raise ContractValidationError(f"{field_name}.bbox_area must be > 0, got {bbox_area}")
        u_hat = _finite(values[2], f"{field_name}.u_hat")
        v_hat = _finite(values[3], f"{field_name}.v_hat")
        range_m = _finite(values[4], f"{field_name}.range_m")
        return [
            1.0,
            _clipped_logit(confidence, self._eps),
            math.log(bbox_area),
            u_hat,
            v_hat,
            range_m,
        ]

    def fit(self, features: Sequence[Any], labels: Sequence[float]) -> "MultivariateLogisticCalibrator":
        rows = [
            self._design_row(feature, f"features[{idx}]")
            for idx, feature in enumerate(_as_list(features, "features"))
        ]
        ys = _binary_labels(labels)
        _check_paired(rows, ys, "MultivariateLogisticCalibrator.fit")
        _require_both_classes(ys, "MultivariateLogisticCalibrator.fit")
        self._coefficients = _fit_logistic_irls(rows, ys, max_iterations=self._max_iterations)
        return self

    def predict(self, feature: Any) -> float:
        coefficients = self.coefficients
        row = self._design_row(feature, "feature")
        z = sum(coef * value for coef, value in zip(coefficients, row))
        return _open_interval_probability(_sigmoid(z))

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": type(self).__name__,
            "coefficients": [float(value) for value in self.coefficients],
            "eps": float(self._eps),
            "feature_order": list(MULTIVARIATE_FEATURE_ORDER),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MultivariateLogisticCalibrator":
        _check_payload_class(payload, cls.__name__)
        feature_order = payload.get("feature_order", list(MULTIVARIATE_FEATURE_ORDER))
        if tuple(feature_order) != MULTIVARIATE_FEATURE_ORDER:
            raise ContractValidationError(
                f"{cls.__name__} payload feature_order {tuple(feature_order)!r} does not match "
                f"{MULTIVARIATE_FEATURE_ORDER!r}"
            )
        out = cls(eps=_payload_eps(payload, cls.__name__))
        out._coefficients = _payload_coefficients(
            payload, len(MULTIVARIATE_FEATURE_ORDER) + 1, cls.__name__
        )
        return out

    @classmethod
    def from_json(cls, text: str) -> "MultivariateLogisticCalibrator":
        return cls.from_dict(json.loads(text))


# ---------------------------------------------------------------------------
# Reliability curve
# ---------------------------------------------------------------------------


def reliability_curve(
    scores: Sequence[float],
    labels: Sequence[float],
    n_bins: int = 10,
) -> list[tuple[float, float, float, int]]:
    """Bin scores into ``n_bins`` equal-width bins over [0, 1].

    Returns one ``(bin_center, mean_score, mean_label, count)`` tuple per
    non-empty bin, in bin order. Empty bins are skipped. Scores of exactly 1.0
    fall in the last bin.
    """

    xs = _probability_scores(scores)
    ys = _binary_labels(labels)
    _check_paired(xs, ys, "reliability_curve")
    n_bins = int(n_bins)
    if n_bins < 1:
        raise ContractValidationError(f"n_bins must be >= 1, got {n_bins}")

    score_sums = [0.0] * n_bins
    label_sums = [0.0] * n_bins
    counts = [0] * n_bins
    for score, label in zip(xs, ys):
        idx = min(int(score * n_bins), n_bins - 1)
        score_sums[idx] += score
        label_sums[idx] += label
        counts[idx] += 1

    out: list[tuple[float, float, float, int]] = []
    for idx in range(n_bins):
        if counts[idx] == 0:
            continue
        center = (idx + 0.5) / n_bins
        out.append((center, score_sums[idx] / counts[idx], label_sums[idx] / counts[idx], counts[idx]))
    return out
