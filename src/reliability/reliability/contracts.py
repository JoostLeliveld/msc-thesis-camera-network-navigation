"""Typed Phase 0 reliability contracts.

These dataclasses are intentionally ROS-independent. They describe records that
future exporters/loaders can serialize without pulling runtime nodes, Gazebo
messages, or planner code into the reliability-learning surface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import base64
import binascii
import json
import math
from numbers import Integral
from typing import Any, ClassVar, Mapping, Sequence

from unav_common.config import parse_bool


SCHEMA_VERSION = "phase0.v1"


class ContractValidationError(ValueError):
    """Raised when a reliability contract violates its declared schema."""


class LeakageError(ContractValidationError):
    """Raised when evaluation-only evidence enters an operational contract."""


def _strict_bool(value: Any, *, field_name: str) -> bool:
    try:
        return parse_bool(value, field_name=field_name)
    except ValueError as exc:
        raise ContractValidationError(str(exc)) from exc


EVALUATION_ONLY_FIELD_NAMES = frozenset(
    {
        "gt_available",
        "gt_x",
        "gt_y",
        "gt_yaw",
        # raw-log GT-pose naming (perception.csv / experiment.csv). These are the ground
        # truth as logged; the token list only caught gt_*, so add the true_* names and the
        # GT-derived pos error explicitly. Mirrors reliability/config/leakage_firewall.yaml.
        "true_available",
        "true_x",
        "true_y",
        "true_yaw",
        "state_pos_error",
        "gazebo_ground_truth_pose",
        "ground_truth_pose",
        "ground_truth_projected_pixel",
        "ground_truth_localization_error_m",
        "belief_error_gt_m",
        "state_error_gt_m",
        "odom_map_gt_drift_m",
        "belief_yaw_error_gt_rad",
        "clearance_m",
        "min_clearance_m",
        "min_wall_distance_m",
        "min_obstacle_distance_m",
        "collision",
        "collision_any",
        "collision_contact",
        "collision_geom",
        "geometry_breach",
        "final_task_outcome",
        "goal_region_success",
        "final_goal_distance",
        "minimum_goal_distance",
        "localization_error_m",
        "localization_error_calibrated_m",
        "localization_error_captime_m",
        "state_error_captime_m",
        "oracle_visible",
        "oracle_bottom_u",
        "oracle_bottom_v",
        "oracle_occlusion_reason",
    }
)

EVALUATION_ONLY_TOKENS = (
    "ground_truth",
    "gazebo_truth",
    "oracle_",
    "_gt_",
    "_gt",
    "gt_",
    "clearance",
    "collision",
    "geometry_breach",
    "final_task_outcome",
    "localization_error",
)

# Prefixes that mark a column/key as evaluation-only regardless of the rest of
# the name. The token list above is a substring test, so it only caught the
# eval_ columns whose names happen to also contain "gt_" (eval_gt_x,
# eval_belief_error_gt_m). Signed-residual columns such as ``eval_res_x`` and
# ``eval_pred_world_x`` are just as GT-derived but contain no such token, so
# the ``eval_`` prefix is matched explicitly here. Anchored as a PREFIX, not a
# substring, so ordinary words that embed the letters (e.g. "retrieval_") are
# not falsely flagged. Mirrors "^eval_" in
# reliability/config/leakage_firewall.yaml.
EVALUATION_ONLY_PREFIXES = ("eval_",)


def _finite_float(value: Any, *, field_name: str, allow_nan: bool = False) -> float:
    if value is None and allow_nan:
        return math.nan
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"{field_name} must be numeric, got {value!r}") from exc
    if not allow_nan and not math.isfinite(out):
        raise ContractValidationError(f"{field_name} must be finite, got {value!r}")
    return out


def _finite_probability(value: Any, *, field_name: str) -> float:
    out = _finite_float(value, field_name=field_name)
    if out < 0.0 or out > 1.0:
        raise ContractValidationError(f"{field_name} must be in [0, 1], got {out}")
    return out


def _nonnegative_float(value: Any, *, field_name: str) -> float:
    out = _finite_float(value, field_name=field_name)
    if out < 0.0:
        raise ContractValidationError(f"{field_name} must be non-negative, got {out}")
    return out


def _array_like_payload(value: Any) -> Any:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return value.tolist()
    return value


def _as_pair(value: Any, *, field_name: str, allow_none: bool = False) -> tuple[float, float] | None:
    value = _array_like_payload(value)
    if value is None:
        if allow_none:
            return None
        raise ContractValidationError(f"{field_name} must be a 2-element sequence")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ContractValidationError(f"{field_name} must be a 2-element sequence")
    return (
        _finite_float(value[0], field_name=f"{field_name}[0]"),
        _finite_float(value[1], field_name=f"{field_name}[1]"),
    )


def _as_quad(value: Any, *, field_name: str, allow_none: bool = False) -> tuple[float, float, float, float] | None:
    value = _array_like_payload(value)
    if value is None:
        if allow_none:
            return None
        raise ContractValidationError(f"{field_name} must be a 4-element sequence")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ContractValidationError(f"{field_name} must be a 4-element sequence")
    return (
        _finite_float(value[0], field_name=f"{field_name}[0]"),
        _finite_float(value[1], field_name=f"{field_name}[1]"),
        _finite_float(value[2], field_name=f"{field_name}[2]"),
        _finite_float(value[3], field_name=f"{field_name}[3]"),
    )


def _as_matrix_2x2(value: Any, *, field_name: str) -> tuple[tuple[float, float], tuple[float, float]]:
    value = _array_like_payload(value)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ContractValidationError(f"{field_name} must be a 2x2 matrix")
    rows = []
    for i, row in enumerate(value):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != 2:
            raise ContractValidationError(f"{field_name} must be a 2x2 matrix")
        rows.append(
            (
                _finite_float(row[0], field_name=f"{field_name}[{i}][0]"),
                _finite_float(row[1], field_name=f"{field_name}[{i}][1]"),
            )
        )
    return (rows[0], rows[1])


def _validate_spd_2x2(matrix: tuple[tuple[float, float], tuple[float, float]], *, field_name: str) -> None:
    a, b = matrix[0]
    c, d = matrix[1]
    if not all(math.isfinite(value) for value in (a, b, c, d)):
        raise ContractValidationError(f"{field_name} must be finite")
    scale = max(abs(a), abs(b), abs(c), abs(d))
    if abs(b - c) > 1e-9 * scale:
        raise ContractValidationError(f"{field_name} must be symmetric")
    # Check a dimensionless determinant, so units cannot determine validity.
    if a <= 0.0 or d <= 0.0 or (a/scale)*(d/scale) - (b/scale)*(c/scale) <= 0.0:
        raise ContractValidationError(f"{field_name} must be symmetric positive definite")


def _plain_mapping(value: Mapping[str, Any] | None, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field_name} must be a mapping")
    return {str(key): val for key, val in value.items()}


def _plain_str_mapping(value: Mapping[str, str] | None, *, field_name: str) -> dict[str, str]:
    raw = _plain_mapping(value, field_name=field_name)
    return {str(key): str(val) for key, val in raw.items()}


def _field_names(cls: type) -> set[str]:
    return {item.name for item in fields(cls)}


def _unknown_fields(cls: type, payload: Mapping[str, Any]) -> set[str]:
    return set(str(key) for key in payload.keys()) - _field_names(cls)


def _contains_evaluation_key(key: str) -> bool:
    lowered = str(key).strip().lower()
    return (
        lowered in EVALUATION_ONLY_FIELD_NAMES
        or lowered.startswith(EVALUATION_ONLY_PREFIXES)
        or any(token in lowered for token in EVALUATION_ONLY_TOKENS)
    )


def _find_evaluation_keys(payload: Any, *, prefix: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if _contains_evaluation_key(key_text):
                hits.append(path)
            hits.extend(_find_evaluation_keys(value, prefix=path))
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        for idx, value in enumerate(payload):
            hits.extend(_find_evaluation_keys(value, prefix=f"{prefix}[{idx}]"))
    return hits


def reject_evaluation_only_keys(payload: Mapping[str, Any], *, context: str) -> None:
    """Reject nested evaluation-only keys in operational payloads."""

    hits = sorted(set(_find_evaluation_keys(payload)))
    if hits:
        joined = ", ".join(hits)
        raise LeakageError(f"{context} contains evaluation-only fields: {joined}")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return value


def _dataclass_to_json_dict(obj: Any) -> dict[str, Any]:
    return _to_jsonable(asdict(obj))


@dataclass(frozen=True)
class OperationalReliabilitySample:
    """Warehouse-realistic input record for availability/quality prediction."""

    schema_version: str = SCHEMA_VERSION
    sample_id: str = ""
    timestamp_s: float = 0.0
    run_id: str = ""
    task_id: str = ""
    seed: int = 0
    image_ref: str = ""
    image_hash: str = ""
    detector_result: dict[str, Any] = field(default_factory=dict)
    selected_pixel: tuple[float, float] | None = None
    projection_valid: bool = False
    measurement_age_s: float = 0.0
    measurement_stale: bool = False
    odometry: dict[str, Any] = field(default_factory=dict)
    state_estimate: dict[str, Any] = field(default_factory=dict)
    belief: dict[str, Any] = field(default_factory=dict)
    camera_relative_range_m: float = 0.0
    image_location: dict[str, Any] = field(default_factory=dict)
    recent_detector_history: tuple[bool, ...] = field(default_factory=tuple)
    config_hash: str = ""
    artifact_hashes: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    _strict_unknown_fields: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ContractValidationError(f"Unsupported schema_version {self.schema_version!r}")
        object.__setattr__(
            self,
            "timestamp_s",
            _nonnegative_float(self.timestamp_s, field_name="timestamp_s"),
        )
        object.__setattr__(
            self,
            "measurement_age_s",
            _nonnegative_float(self.measurement_age_s, field_name="measurement_age_s"),
        )
        object.__setattr__(
            self,
            "camera_relative_range_m",
            _finite_float(
                self.camera_relative_range_m,
                field_name="camera_relative_range_m",
                allow_nan=True,
            ),
        )
        if self.selected_pixel is not None:
            object.__setattr__(
                self,
                "selected_pixel",
                _as_pair(self.selected_pixel, field_name="selected_pixel", allow_none=True),
            )
        object.__setattr__(self, "detector_result", _plain_mapping(self.detector_result, field_name="detector_result"))
        object.__setattr__(self, "odometry", _plain_mapping(self.odometry, field_name="odometry"))
        object.__setattr__(self, "state_estimate", _plain_mapping(self.state_estimate, field_name="state_estimate"))
        object.__setattr__(self, "belief", _plain_mapping(self.belief, field_name="belief"))
        object.__setattr__(self, "image_location", _plain_mapping(self.image_location, field_name="image_location"))
        object.__setattr__(
            self,
            "projection_valid",
            _strict_bool(self.projection_valid, field_name="projection_valid"),
        )
        object.__setattr__(
            self,
            "measurement_stale",
            _strict_bool(self.measurement_stale, field_name="measurement_stale"),
        )
        object.__setattr__(
            self,
            "recent_detector_history",
            tuple(
                _strict_bool(value, field_name=f"recent_detector_history[{index}]")
                for index, value in enumerate(self.recent_detector_history)
            ),
        )
        object.__setattr__(
            self,
            "artifact_hashes",
            _plain_str_mapping(self.artifact_hashes, field_name="artifact_hashes"),
        )
        object.__setattr__(self, "metadata", _plain_mapping(self.metadata, field_name="metadata"))
        reject_evaluation_only_keys(self.to_dict(), context="OperationalReliabilitySample")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OperationalReliabilitySample":
        if not isinstance(payload, Mapping):
            raise ContractValidationError("OperationalReliabilitySample payload must be a mapping")
        unknown = _unknown_fields(cls, payload)
        if unknown:
            eval_unknown = sorted(key for key in unknown if _contains_evaluation_key(key))
            if eval_unknown:
                raise LeakageError(
                    "OperationalReliabilitySample contains evaluation-only fields: "
                    + ", ".join(eval_unknown)
                )
            raise ContractValidationError(
                "OperationalReliabilitySample contains unknown fields: " + ", ".join(sorted(unknown))
            )
        return cls(**dict(payload))

    @classmethod
    def from_json(cls, text: str) -> "OperationalReliabilitySample":
        return cls.from_dict(json.loads(text))

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_json_dict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class EvaluationOnlySample:
    """Evaluation-only record. These fields must never become model features."""

    schema_version: str = SCHEMA_VERSION
    sample_id: str = ""
    timestamp_s: float = 0.0
    run_id: str = ""
    task_id: str = ""
    seed: int = 0
    gazebo_ground_truth_pose: dict[str, Any] = field(default_factory=dict)
    ground_truth_projected_pixel: tuple[float, float] | None = None
    ground_truth_localization_error_m: float = math.nan
    clearance_m: float = math.nan
    collision: bool = False
    geometry_breach: bool = False
    final_task_outcome: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ContractValidationError(f"Unsupported schema_version {self.schema_version!r}")
        object.__setattr__(
            self,
            "timestamp_s",
            _nonnegative_float(self.timestamp_s, field_name="timestamp_s"),
        )
        if self.ground_truth_projected_pixel is not None:
            object.__setattr__(
                self,
                "ground_truth_projected_pixel",
                _as_pair(
                    self.ground_truth_projected_pixel,
                    field_name="ground_truth_projected_pixel",
                    allow_none=True,
                ),
            )
        object.__setattr__(
            self,
            "ground_truth_localization_error_m",
            _finite_float(
                self.ground_truth_localization_error_m,
                field_name="ground_truth_localization_error_m",
                allow_nan=True,
            ),
        )
        object.__setattr__(
            self,
            "clearance_m",
            _finite_float(self.clearance_m, field_name="clearance_m", allow_nan=True),
        )
        object.__setattr__(self, "collision", _strict_bool(self.collision, field_name="collision"))
        object.__setattr__(
            self,
            "geometry_breach",
            _strict_bool(self.geometry_breach, field_name="geometry_breach"),
        )
        object.__setattr__(
            self,
            "gazebo_ground_truth_pose",
            _plain_mapping(self.gazebo_ground_truth_pose, field_name="gazebo_ground_truth_pose"),
        )
        object.__setattr__(self, "metrics", _plain_mapping(self.metrics, field_name="metrics"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvaluationOnlySample":
        if not isinstance(payload, Mapping):
            raise ContractValidationError("EvaluationOnlySample payload must be a mapping")
        unknown = _unknown_fields(cls, payload)
        if unknown:
            raise ContractValidationError(
                "EvaluationOnlySample contains unknown fields: " + ", ".join(sorted(unknown))
            )
        return cls(**dict(payload))

    @classmethod
    def from_json(cls, text: str) -> "EvaluationOnlySample":
        return cls.from_dict(json.loads(text))

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_json_dict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class CameraObservation:
    """Operational observation from one external camera.

    Array-like inputs are accepted and normalized to immutable tuples so the
    contract remains JSON-safe and independent of ROS message classes.
    """

    schema_version: str = SCHEMA_VERSION
    camera_id: str = ""
    # Logical all-camera detector cycle. A native cycle can contain multiple
    # model calls; detector_invocation_id below identifies the physical call.
    # Historical records retain this field without invented member identities.
    source_batch_id: str = ""
    timestamp_s: float = 0.0
    pixel_uv: tuple[float, float] | None = None
    detection_valid: bool = False
    detector_score: float = 0.0
    detector_score_raw: float = math.nan
    bbox_xyxy: tuple[float, float, float, float] | None = None
    visibility_grid_16x16: tuple[float, ...] | None = None
    rgb_context_crop_96x96_zlib_b64: str | None = None
    bbox_bottom_uv: tuple[float, float] | None = None
    mask_bottom_uv: tuple[float, float] | None = None
    selected_pixel_source: str = "none"
    mask_available: bool = False
    mask_area_px: float = math.nan
    mask_polygon_points: float = math.nan
    yolo_inference_wall_ms: float = math.nan
    detector_callback_wall_ms: float = math.nan
    image_receive_stamp_s: float = math.nan
    inference_start_stamp_s: float = math.nan
    inference_finish_stamp_s: float = math.nan
    publish_stamp_s: float = math.nan
    frame_age_at_publish_s: float = math.nan
    measurement_age_s: float = 0.0
    calibration_id: str = ""
    image_frame_id: str = ""
    conditional_cov_uv: tuple[tuple[float, float], tuple[float, float]] = (
        (1.0, 0.0),
        (0.0, 1.0),
    )
    availability_probability: float = 1.0
    association_probability: float = 1.0
    producer_epoch: str = ""
    source_frame_id: str = ""
    capture_stamp_ns: int | None = None
    detector_invocation_id: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ContractValidationError(f"Unsupported schema_version {self.schema_version!r}")
        camera_id = str(self.camera_id).strip()
        if not camera_id:
            raise ContractValidationError("camera_id must be a non-empty string")
        object.__setattr__(self, "camera_id", camera_id)
        object.__setattr__(self, "source_batch_id", str(self.source_batch_id or "").strip())
        object.__setattr__(
            self,
            "timestamp_s",
            _nonnegative_float(self.timestamp_s, field_name="timestamp_s"),
        )
        identity_present = self.capture_stamp_ns is not None or any(
            value != "" for value in (self.producer_epoch, self.source_frame_id,
                                       self.detector_invocation_id)
        )
        if identity_present:
            for name in ("producer_epoch", "source_frame_id", "detector_invocation_id"):
                value = getattr(self, name)
                if not isinstance(value, str) or not value.strip() or value != value.strip():
                    raise ContractValidationError(f"{name} must be a non-empty canonical string")
            if not self.source_batch_id:
                raise ContractValidationError("physical source identity requires source_batch_id")
            ns = self.capture_stamp_ns
            if isinstance(ns, bool) or not isinstance(ns, Integral) or ns < 0:
                raise ContractValidationError("capture_stamp_ns must be a non-negative integer")
            object.__setattr__(self, "capture_stamp_ns", int(ns))
            capture_s = int(ns) / 1_000_000_000
            # Exact ns remain authoritative. Float seconds can lose an ULP at
            # epoch-sized clocks, but may not describe another capture instant.
            tolerance = max(1e-9, math.ulp(capture_s))
            if abs(self.timestamp_s - capture_s) > tolerance:
                raise ContractValidationError("timestamp_s differs from capture_stamp_ns")
        object.__setattr__(
            self,
            "detection_valid",
            _strict_bool(self.detection_valid, field_name="detection_valid"),
        )
        pixel = _as_pair(self.pixel_uv, field_name="pixel_uv", allow_none=True)
        if self.detection_valid and pixel is None:
            raise ContractValidationError("pixel_uv is required when detection_valid is true")
        object.__setattr__(self, "pixel_uv", pixel)
        object.__setattr__(
            self,
            "detector_score",
            _finite_probability(self.detector_score, field_name="detector_score"),
        )
        detector_score_raw = _finite_float(
            self.detector_score_raw,
            field_name="detector_score_raw",
            allow_nan=True,
        )
        if math.isfinite(detector_score_raw) and not 0.0 <= detector_score_raw <= 1.0:
            raise ContractValidationError(
                f"detector_score_raw must be NaN or in [0, 1], got {detector_score_raw}"
            )
        object.__setattr__(self, "detector_score_raw", detector_score_raw)
        object.__setattr__(
            self,
            "bbox_xyxy",
            _as_quad(self.bbox_xyxy, field_name="bbox_xyxy", allow_none=True),
        )
        visibility = self.visibility_grid_16x16
        if visibility is not None:
            if not isinstance(visibility, (tuple, list)) or len(visibility) != 256:
                raise ContractValidationError(
                    "visibility_grid_16x16 must be absent or contain exactly 256 values"
                )
            visibility = tuple(
                _finite_float(value, field_name="visibility_grid_16x16")
                for value in visibility
            )
            if any(value < 0.0 or value > 1.0 for value in visibility):
                raise ContractValidationError("visibility_grid_16x16 values must be in [0,1]")
            if not self.detection_valid or self.bbox_xyxy is None:
                raise ContractValidationError(
                    "visibility_grid_16x16 requires a valid detection and bounding box"
                )
        object.__setattr__(self, "visibility_grid_16x16", visibility)
        crop = self.rgb_context_crop_96x96_zlib_b64
        if crop is not None:
            if not isinstance(crop, str) or not crop:
                raise ContractValidationError(
                    "rgb_context_crop_96x96_zlib_b64 must be absent or non-empty"
                )
            try:
                base64.b64decode(crop, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ContractValidationError(
                    "rgb_context_crop_96x96_zlib_b64 is not valid base64"
                ) from exc
            if not self.detection_valid or self.bbox_xyxy is None:
                raise ContractValidationError(
                    "rgb_context_crop_96x96_zlib_b64 requires a valid detection and box"
                )
        object.__setattr__(self, "rgb_context_crop_96x96_zlib_b64", crop)
        object.__setattr__(
            self,
            "bbox_bottom_uv",
            _as_pair(self.bbox_bottom_uv, field_name="bbox_bottom_uv", allow_none=True),
        )
        object.__setattr__(
            self,
            "mask_bottom_uv",
            _as_pair(self.mask_bottom_uv, field_name="mask_bottom_uv", allow_none=True),
        )
        selected_source = str(self.selected_pixel_source).strip().lower()
        if selected_source not in {"none", "bbox_bottom", "mask_bottom"}:
            raise ContractValidationError(
                "selected_pixel_source must be none, bbox_bottom, or mask_bottom"
            )
        object.__setattr__(self, "selected_pixel_source", selected_source)
        object.__setattr__(
            self,
            "mask_available",
            _strict_bool(self.mask_available, field_name="mask_available"),
        )
        for field_name in (
            "mask_area_px",
            "mask_polygon_points",
            "yolo_inference_wall_ms",
            "detector_callback_wall_ms",
            "image_receive_stamp_s",
            "inference_start_stamp_s",
            "inference_finish_stamp_s",
            "publish_stamp_s",
            "frame_age_at_publish_s",
        ):
            value = _finite_float(
                getattr(self, field_name),
                field_name=field_name,
                allow_nan=True,
            )
            if field_name.endswith(("_ms", "area_px", "points", "age_at_publish_s")):
                if math.isfinite(value) and value < 0.0:
                    raise ContractValidationError(f"{field_name} must be non-negative or NaN")
            object.__setattr__(self, field_name, value)
        measurement_age_s = _finite_float(self.measurement_age_s, field_name="measurement_age_s")
        if measurement_age_s < 0.0:
            raise ContractValidationError("measurement_age_s must be non-negative")
        object.__setattr__(self, "measurement_age_s", measurement_age_s)
        matrix = _as_matrix_2x2(self.conditional_cov_uv, field_name="conditional_cov_uv")
        _validate_spd_2x2(matrix, field_name="conditional_cov_uv")
        object.__setattr__(self, "conditional_cov_uv", matrix)
        object.__setattr__(
            self,
            "availability_probability",
            _finite_probability(
                self.availability_probability,
                field_name="availability_probability",
            ),
        )
        object.__setattr__(
            self,
            "association_probability",
            _finite_probability(
                self.association_probability,
                field_name="association_probability",
            ),
        )
        reject_evaluation_only_keys(self.to_dict(), context="CameraObservation")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CameraObservation":
        if not isinstance(payload, Mapping):
            raise ContractValidationError("CameraObservation payload must be a mapping")
        unknown = _unknown_fields(cls, payload)
        if unknown:
            eval_unknown = sorted(key for key in unknown if _contains_evaluation_key(key))
            if eval_unknown:
                raise LeakageError(
                    "CameraObservation contains evaluation-only fields: "
                    + ", ".join(eval_unknown)
                )
            raise ContractValidationError(
                "CameraObservation contains unknown fields: " + ", ".join(sorted(unknown))
            )
        return cls(**dict(payload))

    @classmethod
    def from_json(cls, text: str) -> "CameraObservation":
        return cls.from_dict(json.loads(text))

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_json_dict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False)

    def quality(self, *, source_model: str = "camera_observation") -> "CameraQuality":
        return CameraQuality(
            camera_id=self.camera_id,
            p_available=self.availability_probability,
            conditional_cov_uv=self.conditional_cov_uv,
            association_confidence=self.association_probability,
            epistemic_score=0.0,
            stale=self.measurement_age_s > 0.0 and not self.detection_valid,
            source_model=source_model,
        )


@dataclass(frozen=True)
class CameraQuality:
    """Planner/estimator-facing quality summary for one camera."""

    schema_version: str = SCHEMA_VERSION
    camera_id: str = ""
    p_available: float = 1.0
    conditional_cov_uv: tuple[tuple[float, float], tuple[float, float]] = (
        (1.0, 0.0),
        (0.0, 1.0),
    )
    association_confidence: float = 1.0
    epistemic_score: float = 0.0
    stale: bool = False
    source_model: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ContractValidationError(f"Unsupported schema_version {self.schema_version!r}")
        object.__setattr__(
            self,
            "p_available",
            _finite_probability(self.p_available, field_name="p_available"),
        )
        matrix = _as_matrix_2x2(self.conditional_cov_uv, field_name="conditional_cov_uv")
        _validate_spd_2x2(matrix, field_name="conditional_cov_uv")
        object.__setattr__(self, "conditional_cov_uv", matrix)
        object.__setattr__(
            self,
            "association_confidence",
            _finite_probability(
                self.association_confidence,
                field_name="association_confidence",
            ),
        )
        epistemic_score = _finite_float(self.epistemic_score, field_name="epistemic_score")
        if epistemic_score < 0.0:
            raise ContractValidationError("epistemic_score must be non-negative")
        object.__setattr__(self, "epistemic_score", epistemic_score)
        object.__setattr__(self, "stale", _strict_bool(self.stale, field_name="stale"))
        reject_evaluation_only_keys(self.to_dict(), context="CameraQuality")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CameraQuality":
        if not isinstance(payload, Mapping):
            raise ContractValidationError("CameraQuality payload must be a mapping")
        unknown = _unknown_fields(cls, payload)
        if unknown:
            eval_unknown = sorted(key for key in unknown if _contains_evaluation_key(key))
            if eval_unknown:
                raise LeakageError(
                    "CameraQuality contains evaluation-only fields: " + ", ".join(eval_unknown)
                )
            raise ContractValidationError(
                "CameraQuality contains unknown fields: " + ", ".join(sorted(unknown))
            )
        return cls(**dict(payload))

    @classmethod
    def from_json(cls, text: str) -> "CameraQuality":
        return cls.from_dict(json.loads(text))

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_json_dict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class ReliabilityPrediction:
    """Availability and quality prediction, before covariance adaptation."""

    schema_version: str = SCHEMA_VERSION
    sample_id: str = ""
    model_id: str = ""
    timestamp_s: float = 0.0
    p_available: float = 1.0
    raw_probability: float = 1.0
    calibrated_probability: float = 1.0
    conditional_quality: dict[str, Any] = field(default_factory=dict)
    epistemic_uncertainty: float = 0.0
    staleness_s: float = 0.0
    provenance: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ContractValidationError(f"Unsupported schema_version {self.schema_version!r}")
        for name in ("p_available", "raw_probability", "calibrated_probability"):
            object.__setattr__(
                self,
                name,
                _finite_probability(getattr(self, name), field_name=name),
            )
        object.__setattr__(
            self,
            "timestamp_s",
            _nonnegative_float(self.timestamp_s, field_name="timestamp_s"),
        )
        epistemic_uncertainty = _finite_float(
            self.epistemic_uncertainty,
            field_name="epistemic_uncertainty",
        )
        if epistemic_uncertainty < 0.0:
            raise ContractValidationError("epistemic_uncertainty must be non-negative")
        object.__setattr__(self, "epistemic_uncertainty", epistemic_uncertainty)
        staleness_s = _finite_float(self.staleness_s, field_name="staleness_s")
        if staleness_s < 0.0:
            raise ContractValidationError("staleness_s must be non-negative")
        object.__setattr__(self, "staleness_s", staleness_s)
        object.__setattr__(
            self,
            "conditional_quality",
            _plain_mapping(self.conditional_quality, field_name="conditional_quality"),
        )

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_json_dict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReliabilityPrediction":
        if not isinstance(payload, Mapping):
            raise ContractValidationError("ReliabilityPrediction payload must be a mapping")
        unknown = _unknown_fields(cls, payload)
        if unknown:
            raise ContractValidationError(
                "ReliabilityPrediction contains unknown fields: " + ", ".join(sorted(unknown))
            )
        return cls(**dict(payload))

    @classmethod
    def from_json(cls, text: str) -> "ReliabilityPrediction":
        return cls.from_dict(json.loads(text))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class UpdateCovariance:
    """Filter-update covariance for an actually available image observation."""

    schema_version: str = SCHEMA_VERSION
    matrix_px2: tuple[tuple[float, float], tuple[float, float]] = ((1.0, 0.0), (0.0, 1.0))
    available: bool = True
    adapter_id: str = ""
    units: str = "px^2"
    source: str = ""
    epistemic_inflation: float = 1.0
    staleness_inflation: float = 1.0

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ContractValidationError(f"Unsupported schema_version {self.schema_version!r}")
        if self.units != "px^2":
            raise ContractValidationError("UpdateCovariance units must be 'px^2'")
        matrix = _as_matrix_2x2(self.matrix_px2, field_name="matrix_px2")
        _validate_spd_2x2(matrix, field_name="matrix_px2")
        object.__setattr__(self, "matrix_px2", matrix)
        object.__setattr__(self, "available", _strict_bool(self.available, field_name="available"))
        epistemic_inflation = _finite_float(
            self.epistemic_inflation,
            field_name="epistemic_inflation",
        )
        if epistemic_inflation < 1.0:
            raise ContractValidationError("epistemic_inflation must be >= 1")
        object.__setattr__(self, "epistemic_inflation", epistemic_inflation)
        staleness_inflation = _finite_float(
            self.staleness_inflation,
            field_name="staleness_inflation",
        )
        if staleness_inflation < 1.0:
            raise ContractValidationError("staleness_inflation must be >= 1")
        object.__setattr__(self, "staleness_inflation", staleness_inflation)

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_json_dict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UpdateCovariance":
        if not isinstance(payload, Mapping):
            raise ContractValidationError("UpdateCovariance payload must be a mapping")
        unknown = _unknown_fields(cls, payload)
        if unknown:
            raise ContractValidationError(
                "UpdateCovariance contains unknown fields: " + ", ".join(sorted(unknown))
            )
        return cls(**dict(payload))

    @classmethod
    def from_json(cls, text: str) -> "UpdateCovariance":
        return cls.from_dict(json.loads(text))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class PlanningCovariance:
    """Planner-facing covariance, separate from update covariance and no-go costs."""

    schema_version: str = SCHEMA_VERSION
    matrix_px2: tuple[tuple[float, float], tuple[float, float]] = ((1.0, 0.0), (0.0, 1.0))
    p_available: float = 1.0
    conditional_covariance_px2: tuple[tuple[float, float], tuple[float, float]] | None = None
    epistemic_score: float = 0.0
    staleness_score: float = 0.0
    provenance: str = ""
    timestamp_s: float = 0.0

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ContractValidationError(f"Unsupported schema_version {self.schema_version!r}")
        object.__setattr__(
            self,
            "p_available",
            _finite_probability(self.p_available, field_name="p_available"),
        )
        matrix = _as_matrix_2x2(self.matrix_px2, field_name="matrix_px2")
        _validate_spd_2x2(matrix, field_name="matrix_px2")
        object.__setattr__(self, "matrix_px2", matrix)
        if self.conditional_covariance_px2 is not None:
            cond = _as_matrix_2x2(
                self.conditional_covariance_px2,
                field_name="conditional_covariance_px2",
            )
            _validate_spd_2x2(cond, field_name="conditional_covariance_px2")
            object.__setattr__(self, "conditional_covariance_px2", cond)
        epistemic_score = _finite_float(self.epistemic_score, field_name="epistemic_score")
        if epistemic_score < 0.0:
            raise ContractValidationError("epistemic_score must be non-negative")
        object.__setattr__(self, "epistemic_score", epistemic_score)
        staleness_score = _finite_float(self.staleness_score, field_name="staleness_score")
        if staleness_score < 0.0:
            raise ContractValidationError("staleness_score must be non-negative")
        object.__setattr__(self, "staleness_score", staleness_score)
        object.__setattr__(
            self,
            "timestamp_s",
            _nonnegative_float(self.timestamp_s, field_name="timestamp_s"),
        )

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_json_dict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanningCovariance":
        if not isinstance(payload, Mapping):
            raise ContractValidationError("PlanningCovariance payload must be a mapping")
        unknown = _unknown_fields(cls, payload)
        if unknown:
            raise ContractValidationError(
                "PlanningCovariance contains unknown fields: " + ", ".join(sorted(unknown))
            )
        return cls(**dict(payload))

    @classmethod
    def from_json(cls, text: str) -> "PlanningCovariance":
        return cls.from_dict(json.loads(text))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False)
