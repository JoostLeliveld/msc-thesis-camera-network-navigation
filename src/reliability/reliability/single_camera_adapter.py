"""Adapters from the current single-camera runtime fields to camera contracts.

This module is deliberately pure Python and ROS-independent. It prepares the
Phase 1 interface without changing detector, EKF, or planner runtime behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from reliability.contracts import CameraObservation, CameraQuality, ContractValidationError


SELECTED_PIXEL_SOURCE_NONE = 0
SELECTED_PIXEL_SOURCE_BBOX_BOTTOM = 1
SELECTED_PIXEL_SOURCE_MASK_BOTTOM = 2


@dataclass(frozen=True)
class SingleCameraAdapterConfig:
    """Current-camera identifiers and covariance endpoints."""

    camera_id: str = "camera_A"
    calibration_id: str = "warehouse_aws_external_camera_v1"
    image_frame_id: str = "external_camera"
    r_visible_uv: float = 2.5
    r_miss_uv: float = 40.0
    min_probability: float = 1.0e-4


def selected_pixel_source_name(code: Any) -> str:
    try:
        value = float(code)
    except (TypeError, ValueError):
        return "none"
    if value >= 1.5:
        return "mask_bottom"
    if value >= 0.5:
        return "bbox_bottom"
    return "none"


def _finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _finite_with_default(value: Any, default: float) -> float:
    out = _finite_or_none(value)
    return float(default) if out is None else out


def _finite_or_nan(value: Any) -> float:
    out = _finite_or_none(value)
    return math.nan if out is None else float(out)


def _probability(value: Any, *, default: float = 0.0) -> float:
    out = _finite_with_default(value, default)
    return float(min(max(out, 0.0), 1.0))


def _bool_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    numeric = _finite_or_none(value)
    if numeric is not None:
        return numeric >= 0.5
    return bool(value)


def _first_finite(mapping: Mapping[str, Any], keys: Sequence[str], default: float) -> float:
    for key in keys:
        if key in mapping:
            value = _finite_or_none(mapping.get(key))
            if value is not None:
                return value
    return float(default)


def _optional_pair(mapping: Mapping[str, Any], u_key: str, v_key: str) -> tuple[float, float] | None:
    u = _finite_or_none(mapping.get(u_key))
    v = _finite_or_none(mapping.get(v_key))
    if u is None or v is None:
        return None
    return (u, v)


def _optional_bbox_xyxy(mapping: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    values = [
        _finite_or_none(mapping.get("bbox_xmin")),
        _finite_or_none(mapping.get("bbox_ymin")),
        _finite_or_none(mapping.get("bbox_xmax")),
        _finite_or_none(mapping.get("bbox_ymax")),
    ]
    if any(value is None for value in values):
        return None
    x0, y0, x1, y1 = (float(value) for value in values)
    return (x0, y0, x1, y1)


def _matrix_2x2(value: Any) -> tuple[tuple[float, float], tuple[float, float]]:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ContractValidationError("conditional covariance must be a 2x2 matrix")
    rows: list[tuple[float, float]] = []
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != 2:
            raise ContractValidationError("conditional covariance must be a 2x2 matrix")
        a = _finite_or_none(row[0])
        b = _finite_or_none(row[1])
        if a is None or b is None:
            raise ContractValidationError("conditional covariance must be finite")
        rows.append((a, b))
    return (rows[0], rows[1])


def precision_blend_covariance(
    trust: Any,
    *,
    r_visible_uv: float,
    r_miss_uv: float,
    min_probability: float = 1.0e-4,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Match the current planner's scalar-trust precision blend in pixel units."""

    t = _probability(trust, default=1.0)
    t = float(min(max(t, float(min_probability)), 1.0 - float(min_probability)))
    visible_var = max(float(r_visible_uv) ** 2, 1.0e-6)
    miss_var = max(float(r_miss_uv) ** 2, 1.0e-6)
    visible_prec = 1.0 / visible_var
    miss_prec = 1.0 / miss_var
    blended_prec = t * visible_prec + (1.0 - t) * miss_prec
    var = 1.0 / max(blended_prec, 1.0e-9)
    return ((float(var), 0.0), (0.0, float(var)))


def camera_observation_from_diagnostics(
    diagnostics: Mapping[str, Any],
    *,
    config: SingleCameraAdapterConfig | None = None,
    timestamp_s: float | None = None,
    measurement_age_s: float | None = None,
    conditional_cov_uv: Any | None = None,
    availability_probability: float | None = None,
    association_probability: float = 1.0,
) -> CameraObservation:
    """Convert current detector diagnostics into a camera observation contract."""

    cfg = config or SingleCameraAdapterConfig()
    if timestamp_s is None:
        timestamp_s = _first_finite(diagnostics, ("stamp", "diag_stamp", "pixel_pose_stamp"), 0.0)
    selected_score = _finite_or_none(diagnostics.get("yolo_score_selected"))
    raw_score = _finite_or_none(diagnostics.get("yolo_score_raw"))
    detector_score = _probability(selected_score if selected_score is not None else raw_score, default=0.0)

    if "yolo_detected_after_threshold" in diagnostics:
        detection_valid = _bool_flag(diagnostics.get("yolo_detected_after_threshold"))
    else:
        detection_valid = _bool_flag(diagnostics.get("detected", False))

    pixel_uv = _optional_pair(diagnostics, "u_mid", "v_mid")
    if pixel_uv is None:
        pixel_uv = _optional_pair(diagnostics, "pixel_pose_u", "pixel_pose_v")

    if measurement_age_s is None:
        measurement_age_s = _first_finite(
            diagnostics,
            (
                "measurement_age_s",
                "pixel_pose_age_s",
                "frame_age_at_publish_s",
                "detector_total_latency_s",
            ),
            0.0,
        )
    measurement_age_s = max(float(measurement_age_s), 0.0)

    if availability_probability is None:
        availability_probability = detector_score if detection_valid else 0.0
    if conditional_cov_uv is None:
        conditional_cov_uv = precision_blend_covariance(
            availability_probability,
            r_visible_uv=cfg.r_visible_uv,
            r_miss_uv=cfg.r_miss_uv,
            min_probability=cfg.min_probability,
        )

    bbox = _optional_bbox_xyxy(diagnostics)
    bbox_bottom_uv = None
    if bbox is not None:
        bbox_bottom_uv = (0.5 * (bbox[0] + bbox[2]), bbox[3])
    mask_bottom_uv = _optional_pair(diagnostics, "mask_bottom_u", "mask_bottom_v")
    selected_source = selected_pixel_source_name(diagnostics.get("selected_pixel_source_code", 0.0))
    mask_available = _bool_flag(diagnostics.get("mask_used", mask_bottom_uv is not None))

    return CameraObservation(
        camera_id=cfg.camera_id,
        timestamp_s=float(timestamp_s),
        pixel_uv=pixel_uv,
        detection_valid=detection_valid,
        detector_score=detector_score,
        detector_score_raw=_finite_or_nan(diagnostics.get("yolo_score_raw")),
        bbox_xyxy=bbox,
        bbox_bottom_uv=bbox_bottom_uv,
        mask_bottom_uv=mask_bottom_uv,
        selected_pixel_source=selected_source,
        mask_available=mask_available,
        mask_area_px=_finite_or_nan(diagnostics.get("mask_area_px")),
        mask_polygon_points=_finite_or_nan(diagnostics.get("mask_polygon_points")),
        yolo_inference_wall_ms=_finite_or_nan(diagnostics.get("yolo_inference_ms")),
        detector_callback_wall_ms=_finite_or_nan(diagnostics.get("detector_callback_ms")),
        image_receive_stamp_s=_finite_or_nan(diagnostics.get("yolo_receive_stamp")),
        inference_start_stamp_s=_finite_or_nan(diagnostics.get("yolo_start_stamp")),
        inference_finish_stamp_s=_finite_or_nan(diagnostics.get("yolo_finish_stamp")),
        publish_stamp_s=_finite_or_nan(diagnostics.get("yolo_publish_stamp")),
        frame_age_at_publish_s=_finite_or_nan(diagnostics.get("frame_age_at_publish_s")),
        measurement_age_s=measurement_age_s,
        calibration_id=cfg.calibration_id,
        image_frame_id=cfg.image_frame_id,
        conditional_cov_uv=_matrix_2x2(conditional_cov_uv),
        availability_probability=_probability(availability_probability, default=0.0),
        association_probability=_probability(association_probability, default=1.0),
    )


def camera_quality_from_planning_diagnostics(
    diagnostics: Mapping[str, Any],
    *,
    config: SingleCameraAdapterConfig | None = None,
    association_confidence: float = 1.0,
    stale: bool = False,
    source_model: str = "single_camera_gp",
) -> CameraQuality:
    """Convert current planner visibility diagnostics into camera quality."""

    cfg = config or SingleCameraAdapterConfig()
    p_available = _probability(
        diagnostics.get("p_vis_eff", diagnostics.get("p_vis", 1.0)),
        default=1.0,
    )
    r_plan = diagnostics.get("R_plan")
    if r_plan is None:
        r_plan = precision_blend_covariance(
            p_available,
            r_visible_uv=cfg.r_visible_uv,
            r_miss_uv=cfg.r_miss_uv,
            min_probability=cfg.min_probability,
        )
    return CameraQuality(
        camera_id=cfg.camera_id,
        p_available=p_available,
        conditional_cov_uv=_matrix_2x2(r_plan),
        association_confidence=_probability(association_confidence, default=1.0),
        epistemic_score=0.0,
        stale=stale,
        source_model=source_model,
    )


def camera_quality_from_observation(
    observation: CameraObservation,
    *,
    source_model: str = "single_camera_observation",
) -> CameraQuality:
    return observation.quality(source_model=source_model)
