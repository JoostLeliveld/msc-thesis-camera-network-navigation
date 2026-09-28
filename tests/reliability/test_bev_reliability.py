from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "reliability"))

from reliability import (  # noqa: E402
    BEVCameraToken,
    BEVFeatureConfig,
    BEVReliabilityModel,
    BEVReliabilityTrainingExample,
    CalibratedPriorConfig,
    CameraCalibration,
    LeakageError,
    PriorGridSpec,
    bev_tokens_from_prior_maps,
    build_multicamera_geometry_prior,
    fit_bev_reliability_model,
    predict_bev_reliability_grid,
)


def _camera_a() -> CameraCalibration:
    return CameraCalibration(
        camera_id="camera_A",
        cam_pos_m=(0.0, -4.0, 3.0),
        look_at_m=(0.0, 0.0, 0.0),
        img_width_px=640,
        img_height_px=480,
        fov_h_rad=1.5708,
    )


def _camera_b() -> CameraCalibration:
    return CameraCalibration(
        camera_id="camera_B",
        cam_pos_m=(4.0, 0.0, 3.0),
        look_at_m=(0.0, 0.0, 0.0),
        img_width_px=640,
        img_height_px=480,
        fov_h_rad=1.5708,
    )


def _prior():
    grid = PriorGridSpec.from_bounds(x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0, nx=5, ny=5)
    cfg = CalibratedPriorConfig(
        min_border_margin_px=0.0,
        min_pixel_scale_px_per_m=0.0,
        good_range_m=10.0,
        range_decay_m=10.0,
    )
    return build_multicamera_geometry_prior([_camera_a(), _camera_b()], grid, cfg)


def test_camera_set_attention_prefers_operationally_reliable_camera() -> None:
    tokens = bev_tokens_from_prior_maps(
        _prior(),
        (0.0, 0.0),
        camera_measurements={
            "camera_A": {
                "detector_score": 0.05,
                "detection_valid": False,
                "measurement_age_s": 2.0,
                "measurement_stale": True,
                "recent_detector_history": [False, False, False],
            },
            "camera_B": {
                "detector_score": 0.95,
                "detection_valid": True,
                "measurement_age_s": 0.0,
                "measurement_stale": False,
                "recent_detector_history": [True, True, True],
            },
        },
    )

    prediction = BEVReliabilityModel().predict_set(tokens)

    assert prediction.best_camera_id == "camera_B"
    assert prediction.per_camera_probability["camera_B"] > prediction.per_camera_probability["camera_A"]
    assert prediction.attention_weights["camera_B"] > prediction.attention_weights["camera_A"]
    assert prediction.fused_probability >= prediction.best_probability
    assert prediction.fused_quality.conditional_cov_uv[0][0] > 0.0
    assert prediction.fused_quality.source_model.endswith(":set_union")


def test_bev_reliability_grid_exports_provider() -> None:
    grid = predict_bev_reliability_grid(BEVReliabilityModel(), _prior())

    assert grid.fused_probability.shape == (5, 5)
    assert grid.best_camera_id.shape == (5, 5)
    assert set(grid.attention_by_camera) == {"camera_A", "camera_B"}
    assert np.all(grid.fused_probability >= 0.0)
    assert np.all(grid.fused_probability <= 1.0)
    assert grid.attention_by_camera["camera_A"][2, 2] + grid.attention_by_camera["camera_B"][2, 2] == pytest.approx(1.0)

    provider = grid.to_grid_provider(camera_id="multicamera")
    quality = provider.query("multicamera", (0.0, 0.0))

    assert quality.camera_id == "multicamera"
    assert 0.0 < quality.p_available < 1.0
    assert quality.source_model.startswith("bev_reliability_linear_attention")


def test_fit_bev_reliability_model_learns_detector_signal() -> None:
    cfg = BEVFeatureConfig(feature_names=("bias", "detector_score", "detection_valid"))
    low = BEVCameraToken(
        camera_id="camera_A",
        xy_m=(0.0, 0.0),
        geometry_prior=0.5,
        detector_score=0.0,
        detection_valid=False,
    )
    high = BEVCameraToken(
        camera_id="camera_A",
        xy_m=(0.0, 0.0),
        geometry_prior=0.5,
        detector_score=1.0,
        detection_valid=True,
    )
    examples = tuple(
        [BEVReliabilityTrainingExample(low, 0.0) for _ in range(8)]
        + [BEVReliabilityTrainingExample(high, 1.0) for _ in range(8)]
    )

    model = fit_bev_reliability_model(examples, feature_config=cfg, steps=500, learning_rate=0.20)

    assert model.predict_token_probability(high) > 0.80
    assert model.predict_token_probability(low) < 0.30
    assert model.weights["detector_score"] > 0.0


def test_bev_reliability_rejects_evaluation_feature_leakage() -> None:
    with pytest.raises(LeakageError, match="gt_x"):
        BEVFeatureConfig(feature_names=("bias", "gt_x"))

    with pytest.raises(LeakageError, match="ground_truth_pose"):
        BEVCameraToken(
            camera_id="camera_A",
            xy_m=(0.0, 0.0),
            metadata={"ground_truth_pose": {"x": 0.0}},
        )

    with pytest.raises(LeakageError, match="localization_error"):
        bev_tokens_from_prior_maps(
            _prior(),
            (0.0, 0.0),
            camera_measurements={
                "camera_A": {
                    "detector_score": 0.5,
                    "localization_error_m": 0.1,
                }
            },
        )
