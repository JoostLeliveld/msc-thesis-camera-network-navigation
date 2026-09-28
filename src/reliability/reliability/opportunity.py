"""Operational-only expected-observation opportunities and LOO usability labels.

This module intentionally does not project from Gazebo truth or read an
``evaluation_only`` export.  A camera miss is meaningful only when a frozen
operational prediction says that the camera had an opportunity to see the
robot.  Usability labels are likewise derived from a leave-one-camera-out
reference that explicitly excludes the camera being labelled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

from reliability.contracts import (
    ContractValidationError,
    LeakageError,
    OperationalReliabilitySample,
    reject_evaluation_only_keys,
)


@dataclass(frozen=True)
class OpportunityConfig:
    """Pre-registered geometry, liveness, and association gates."""

    image_width_px: int
    image_height_px: int
    valid_margin_px: float = 0.0
    ellipse_sigma: float = 2.0
    min_ellipse_inside_fraction: float = 0.8
    min_predicted_height_px: float = 12.0
    max_measurement_age_s: float = 0.5
    max_association_delta_s: float = 0.10

    def __post_init__(self) -> None:
        if self.image_width_px <= 0 or self.image_height_px <= 0:
            raise ContractValidationError("image dimensions must be positive")
        for name in (
            "valid_margin_px",
            "ellipse_sigma",
            "min_predicted_height_px",
            "max_measurement_age_s",
            "max_association_delta_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ContractValidationError(f"{name} must be finite and non-negative")
        fraction = float(self.min_ellipse_inside_fraction)
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ContractValidationError("min_ellipse_inside_fraction must be in (0, 1]")
        if self.valid_margin_px * 2.0 >= min(self.image_width_px, self.image_height_px):
            raise ContractValidationError("valid_margin_px leaves no valid image region")


@dataclass(frozen=True)
class OpportunityPrediction:
    """Operational prediction sidecar for one possible camera observation."""

    sample_id: str
    camera_id: str
    predicted_uv: tuple[float, float]
    predicted_cov_uv: tuple[tuple[float, float], tuple[float, float]]
    predicted_height_px: float
    stream_live: bool
    association_delta_s: float

    def __post_init__(self) -> None:
        if not self.sample_id or not self.camera_id:
            raise ContractValidationError("OpportunityPrediction needs sample_id and camera_id")
        _finite_pair(self.predicted_uv, field="predicted_uv")
        _validate_covariance(self.predicted_cov_uv, field="predicted_cov_uv")
        for name in ("predicted_height_px", "association_delta_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ContractValidationError(f"{name} must be finite and non-negative")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OpportunityPrediction":
        if not isinstance(payload, Mapping):
            raise ContractValidationError("OpportunityPrediction payload must be a mapping")
        reject_evaluation_only_keys(payload, context="OpportunityPrediction")
        required = {
            "sample_id",
            "camera_id",
            "predicted_uv",
            "predicted_cov_uv",
            "predicted_height_px",
            "stream_live",
            "association_delta_s",
        }
        unknown = set(payload) - required
        missing = required - set(payload)
        if missing or unknown:
            details = []
            if missing:
                details.append("missing " + ", ".join(sorted(missing)))
            if unknown:
                details.append("unknown " + ", ".join(sorted(unknown)))
            raise ContractValidationError("OpportunityPrediction " + "; ".join(details))
        return cls(**dict(payload))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OpportunityRow:
    """One training-row candidate; only emitted when opportunity is true."""

    sample_id: str
    camera_id: str
    run_id: str
    timestamp_s: float
    belief_xy_m: tuple[float, float] | None
    belief_cov_xy_m2: tuple[tuple[float, float], tuple[float, float]] | None
    predicted_uv: tuple[float, float]
    predicted_cov_uv: tuple[tuple[float, float], tuple[float, float]]
    predicted_height_px: float
    ellipse_inside_fraction: float
    inside_valid_region: bool
    stream_healthy: bool
    detection_received: bool
    association_valid: bool
    raw_confidence: float | None
    measured_uv: tuple[float, float] | None
    availability_label: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("belief_xy_m", "predicted_uv", "measured_uv"):
            value = payload[key]
            if value is not None:
                payload[key] = list(value)
        for key in ("belief_cov_xy_m2", "predicted_cov_uv"):
            value = payload[key]
            if value is not None:
                payload[key] = [list(row) for row in value]
        return payload


@dataclass(frozen=True)
class LOOReference:
    """Per-sample operational reference produced without the labelled camera."""

    sample_id: str
    excluded_camera_id: str
    reference_uv: tuple[float, float]
    reference_cov_uv: tuple[tuple[float, float], tuple[float, float]]
    label_source: str = "leave_one_camera_out"

    def __post_init__(self) -> None:
        if not self.sample_id or not self.excluded_camera_id:
            raise ContractValidationError("LOOReference needs sample_id and excluded_camera_id")
        if self.label_source != "leave_one_camera_out":
            raise ContractValidationError("LOOReference label_source must be leave_one_camera_out")
        _finite_pair(self.reference_uv, field="reference_uv")
        _validate_covariance(self.reference_cov_uv, field="reference_cov_uv")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LOOReference":
        if not isinstance(payload, Mapping):
            raise ContractValidationError("LOOReference payload must be a mapping")
        reject_evaluation_only_keys(payload, context="LOOReference")
        required = {
            "sample_id",
            "excluded_camera_id",
            "reference_uv",
            "reference_cov_uv",
            "label_source",
        }
        unknown = set(payload) - required
        missing = required - set(payload)
        if missing or unknown:
            details = []
            if missing:
                details.append("missing " + ", ".join(sorted(missing)))
            if unknown:
                details.append("unknown " + ", ".join(sorted(unknown)))
            raise ContractValidationError("LOOReference " + "; ".join(details))
        return cls(**dict(payload))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_opportunity_row(
    sample: OperationalReliabilitySample,
    prediction: OpportunityPrediction,
    *,
    config: OpportunityConfig,
) -> OpportunityRow | None:
    """Emit a row only if the camera had a valid operational opportunity.

    A robot outside the predicted valid region is intentionally not a negative
    availability example.  This prevents unsupported geometry from being
    relabelled as a detector failure.
    """

    if prediction.sample_id != sample.sample_id:
        raise ContractValidationError("prediction sample_id does not match operational sample")
    camera_id = str(sample.metadata.get("camera_id", ""))
    if not camera_id:
        raise ContractValidationError("operational sample lacks metadata.camera_id")
    if prediction.camera_id != camera_id:
        raise ContractValidationError("prediction camera_id does not match operational sample")

    ellipse_fraction = ellipse_inside_fraction(prediction, config=config)
    u, v = prediction.predicted_uv
    margin = config.valid_margin_px
    inside_center = (
        margin <= u <= config.image_width_px - margin
        and margin <= v <= config.image_height_px - margin
    )
    inside_valid = inside_center and ellipse_fraction >= config.min_ellipse_inside_fraction
    stream_healthy = bool(prediction.stream_live) and sample.measurement_age_s <= config.max_measurement_age_s
    association_window_valid = prediction.association_delta_s <= config.max_association_delta_s
    scale_valid = prediction.predicted_height_px >= config.min_predicted_height_px
    if not (inside_valid and stream_healthy and association_window_valid and scale_valid):
        return None

    detected = _bool(sample.detector_result.get("detected", False))
    association_valid = detected and association_window_valid
    raw_confidence = _finite_or_none(sample.detector_result.get("raw_score"))
    belief_xy = _mapping_pair(sample.belief, "x_m", "y_m")
    belief_cov = _mapping_covariance(sample.belief, "covariance_xy_m2")
    return OpportunityRow(
        sample_id=sample.sample_id,
        camera_id=camera_id,
        run_id=sample.run_id,
        timestamp_s=sample.timestamp_s,
        belief_xy_m=belief_xy,
        belief_cov_xy_m2=belief_cov,
        predicted_uv=prediction.predicted_uv,
        predicted_cov_uv=prediction.predicted_cov_uv,
        predicted_height_px=prediction.predicted_height_px,
        ellipse_inside_fraction=ellipse_fraction,
        inside_valid_region=True,
        stream_healthy=True,
        detection_received=detected,
        association_valid=association_valid,
        raw_confidence=raw_confidence,
        measured_uv=sample.selected_pixel,
        availability_label=int(association_valid),
    )


def label_loo_usability(
    row: OpportunityRow,
    reference: LOOReference,
    *,
    max_residual_px: float,
) -> dict[str, Any]:
    """Append a non-circular usability label to an opportunity row."""

    if not math.isfinite(max_residual_px) or max_residual_px <= 0.0:
        raise ContractValidationError("max_residual_px must be finite and positive")
    if reference.sample_id != row.sample_id:
        raise ContractValidationError("LOO reference sample_id does not match opportunity row")
    if reference.excluded_camera_id != row.camera_id:
        raise LeakageError("LOO reference does not exclude the camera being labelled")

    payload = row.to_dict()
    payload["label_source"] = reference.label_source
    payload["reference_uv"] = list(reference.reference_uv)
    payload["reference_cov_uv"] = [list(item) for item in reference.reference_cov_uv]
    payload["reference_uncertainty_px"] = math.sqrt(
        max(reference.reference_cov_uv[0][0], reference.reference_cov_uv[1][1])
    )
    if not row.association_valid or row.measured_uv is None:
        payload.update({"e_loo_uv": None, "e_loo_norm_px": None, "usable_label": None})
        return payload

    residual = (
        row.measured_uv[0] - reference.reference_uv[0],
        row.measured_uv[1] - reference.reference_uv[1],
    )
    norm = math.hypot(*residual)
    payload.update(
        {
            "e_loo_uv": list(residual),
            "e_loo_norm_px": norm,
            "usable_label": int(norm <= max_residual_px),
        }
    )
    return payload


def ellipse_inside_fraction(prediction: OpportunityPrediction, *, config: OpportunityConfig) -> float:
    """Conservative five-point approximation of projected ellipse support."""

    u, v = prediction.predicted_uv
    sigma_u = math.sqrt(prediction.predicted_cov_uv[0][0])
    sigma_v = math.sqrt(prediction.predicted_cov_uv[1][1])
    radius = config.ellipse_sigma
    points = ((u, v), (u - radius * sigma_u, v), (u + radius * sigma_u, v), (u, v - radius * sigma_v), (u, v + radius * sigma_v))
    margin = config.valid_margin_px
    inside = sum(
        margin <= point_u <= config.image_width_px - margin
        and margin <= point_v <= config.image_height_px - margin
        for point_u, point_v in points
    )
    return inside / len(points)


def _finite_pair(value: Sequence[Any], *, field: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ContractValidationError(f"{field} must be a two-element sequence")
    pair = (float(value[0]), float(value[1]))
    if not all(math.isfinite(item) for item in pair):
        raise ContractValidationError(f"{field} must be finite")
    return pair


def _validate_covariance(value: Sequence[Sequence[Any]], *, field: str) -> None:
    if not isinstance(value, Sequence) or len(value) != 2 or any(not isinstance(row, Sequence) or len(row) != 2 for row in value):
        raise ContractValidationError(f"{field} must be a 2x2 matrix")
    a, b = (float(item) for item in value[0])
    c, d = (float(item) for item in value[1])
    if not all(math.isfinite(item) for item in (a, b, c, d)) or abs(b - c) > 1.0e-9:
        raise ContractValidationError(f"{field} must be finite and symmetric")
    if a < 0.0 or d < 0.0 or a * d - b * c < -1.0e-9:
        raise ContractValidationError(f"{field} must be positive semidefinite")


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return bool(value)


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mapping_pair(mapping: Mapping[str, Any], first: str, second: str) -> tuple[float, float] | None:
    first_value = _finite_or_none(mapping.get(first))
    second_value = _finite_or_none(mapping.get(second))
    if first_value is None or second_value is None:
        return None
    return (first_value, second_value)


def _mapping_covariance(mapping: Mapping[str, Any], key: str) -> tuple[tuple[float, float], tuple[float, float]] | None:
    value = mapping.get(key)
    if value is None:
        return None
    _validate_covariance(value, field=key)
    return ((float(value[0][0]), float(value[0][1])), (float(value[1][0]), float(value[1][1])))
