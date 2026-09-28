from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "reliability"))

from reliability import (  # noqa: E402
    CameraQuality,
    ContractValidationError,
    SingleCameraAdapterConfig,
    camera_observation_from_diagnostics,
    camera_quality_from_observation,
    camera_quality_from_planning_diagnostics,
    precision_blend_covariance,
    selected_pixel_source_name,
)


def _diagnostics() -> dict:
    return {
        "stamp": 184.62,
        "detected": True,
        "u_mid": 392.0,
        "v_mid": 456.0,
        "yolo_score_raw": 0.81,
        "yolo_score_selected": 0.83,
        "yolo_detected_after_threshold": 1.0,
        "bbox_xmin": 360.0,
        "bbox_ymin": 410.0,
        "bbox_xmax": 424.0,
        "bbox_ymax": 456.0,
        "mask_area_px": 812.0,
        "mask_bottom_u": 391.0,
        "mask_bottom_v": 455.0,
        "mask_used": 1.0,
        "mask_polygon_points": 18.0,
        "selected_pixel_source_code": 1.0,
        "yolo_inference_ms": 42.5,
        "detector_callback_ms": 47.0,
        "yolo_receive_stamp": 184.60,
        "yolo_start_stamp": 184.61,
        "yolo_finish_stamp": 184.6525,
        "yolo_publish_stamp": 184.66,
        "frame_age_at_publish_s": 0.04,
    }


def test_camera_observation_from_current_diagnostics_preserves_selected_fields() -> None:
    obs = camera_observation_from_diagnostics(
        _diagnostics(),
        config=SingleCameraAdapterConfig(
            camera_id="camera_A",
            calibration_id="calib_v1",
            image_frame_id="external_camera",
            r_visible_uv=2.5,
            r_miss_uv=40.0,
        ),
        conditional_cov_uv=((6.25, 0.0), (0.0, 6.25)),
        availability_probability=0.74,
        association_probability=0.96,
    )

    assert obs.camera_id == "camera_A"
    assert obs.timestamp_s == 184.62
    assert obs.pixel_uv == (392.0, 456.0)
    assert obs.detection_valid is True
    assert obs.detector_score == 0.83
    assert obs.detector_score_raw == 0.81
    assert obs.bbox_xyxy == (360.0, 410.0, 424.0, 456.0)
    assert obs.bbox_bottom_uv == (392.0, 456.0)
    assert obs.mask_bottom_uv == (391.0, 455.0)
    assert obs.selected_pixel_source == "bbox_bottom"
    assert obs.mask_available is True
    assert obs.mask_area_px == 812.0
    assert obs.mask_polygon_points == 18.0
    assert obs.yolo_inference_wall_ms == 42.5
    assert obs.detector_callback_wall_ms == 47.0
    assert obs.image_receive_stamp_s == 184.60
    assert obs.inference_start_stamp_s == 184.61
    assert obs.inference_finish_stamp_s == 184.6525
    assert obs.publish_stamp_s == 184.66
    assert obs.frame_age_at_publish_s == 0.04
    assert obs.measurement_age_s == 0.04
    assert obs.calibration_id == "calib_v1"
    assert obs.image_frame_id == "external_camera"
    assert obs.conditional_cov_uv == ((6.25, 0.0), (0.0, 6.25))
    assert obs.availability_probability == 0.74
    assert obs.association_probability == 0.96


def test_camera_observation_handles_subthreshold_candidate_without_update_validity() -> None:
    diag = _diagnostics()
    diag["yolo_detected_after_threshold"] = 0.0
    diag["detected"] = False
    diag["yolo_score_raw"] = 0.18
    diag["yolo_score_selected"] = math.nan

    obs = camera_observation_from_diagnostics(diag)

    assert obs.detection_valid is False
    assert obs.detector_score == 0.18
    assert obs.pixel_uv == (392.0, 456.0)
    assert obs.availability_probability == 0.0


def test_camera_observation_falls_back_to_pixel_pose_columns() -> None:
    diag = {
        "diag_stamp": 10.0,
        "pixel_pose_u": 12.0,
        "pixel_pose_v": 34.0,
        "pixel_pose_age_s": 0.2,
        "yolo_detected_after_threshold": 1.0,
        "yolo_score_raw": 0.6,
    }

    obs = camera_observation_from_diagnostics(diag)

    assert obs.timestamp_s == 10.0
    assert obs.pixel_uv == (12.0, 34.0)
    assert obs.measurement_age_s == 0.2


def test_precision_blend_covariance_matches_current_planner_formula() -> None:
    cov = precision_blend_covariance(
        0.25,
        r_visible_uv=2.5,
        r_miss_uv=40.0,
        min_probability=0.0,
    )
    expected_var = 1.0 / (0.25 / (2.5**2) + 0.75 / (40.0**2))

    assert cov == ((pytest.approx(expected_var), 0.0), (0.0, pytest.approx(expected_var)))


def test_camera_quality_from_planning_diagnostics_preserves_r_plan() -> None:
    quality = camera_quality_from_planning_diagnostics(
        {
            "p_vis": 0.61,
            "p_vis_eff": 0.59,
            "R_plan": ((10.0, 0.2), (0.2, 12.0)),
        },
        source_model="single_camera_gp",
    )

    assert quality == CameraQuality(
        camera_id="camera_A",
        p_available=0.59,
        conditional_cov_uv=((10.0, 0.2), (0.2, 12.0)),
        association_confidence=1.0,
        epistemic_score=0.0,
        stale=False,
        source_model="single_camera_gp",
    )


def test_camera_quality_from_planning_diagnostics_can_recompute_r_plan() -> None:
    quality = camera_quality_from_planning_diagnostics(
        {"p_vis_eff": 1.0},
        config=SingleCameraAdapterConfig(r_visible_uv=2.5, r_miss_uv=40.0, min_probability=0.0),
    )

    assert quality.conditional_cov_uv == ((6.25, 0.0), (0.0, 6.25))


def test_camera_quality_from_observation_is_lossless_quality_projection() -> None:
    obs = camera_observation_from_diagnostics(
        _diagnostics(),
        conditional_cov_uv=((7.0, 0.0), (0.0, 8.0)),
        availability_probability=0.5,
        association_probability=0.9,
    )
    quality = camera_quality_from_observation(obs, source_model="diagnostics")

    assert quality.camera_id == obs.camera_id
    assert quality.p_available == obs.availability_probability
    assert quality.conditional_cov_uv == obs.conditional_cov_uv
    assert quality.association_confidence == obs.association_probability
    assert quality.source_model == "diagnostics"


def test_selected_pixel_source_code_mapping() -> None:
    assert selected_pixel_source_name(0.0) == "none"
    assert selected_pixel_source_name(1.0) == "bbox_bottom"
    assert selected_pixel_source_name(2.0) == "mask_bottom"
    assert selected_pixel_source_name("bad") == "none"


def test_adapter_rejects_bad_covariance() -> None:
    with pytest.raises(ContractValidationError, match="positive definite"):
        camera_observation_from_diagnostics(
            _diagnostics(),
            conditional_cov_uv=((1.0, 0.0), (0.0, -1.0)),
        )
