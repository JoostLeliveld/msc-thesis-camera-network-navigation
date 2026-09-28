from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "reliability"))

from reliability import (  # noqa: E402
    CalibratedPriorConfig,
    CameraCalibration,
    PriorGridSpec,
    build_camera_geometry_prior,
    build_multicamera_geometry_prior,
    evaluate_probability_grid,
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


def test_build_multicamera_prior_fuses_best_and_union() -> None:
    grid = PriorGridSpec.from_bounds(x_min=-2.0, x_max=2.0, y_min=-2.0, y_max=2.0, nx=7, ny=7)
    cfg = CalibratedPriorConfig(
        min_border_margin_px=0.0,
        min_pixel_scale_px_per_m=0.0,
        good_range_m=10.0,
        range_decay_m=10.0,
    )

    prior = build_multicamera_geometry_prior([_camera_a(), _camera_b()], grid, cfg)
    a = prior.camera_maps["camera_A"].probability
    b = prior.camera_maps["camera_B"].probability

    assert prior.union_probability.shape == (7, 7)
    assert np.all(prior.best_probability == pytest.approx(np.maximum(a, b)))
    assert np.all(prior.union_probability >= prior.best_probability)
    assert np.any(prior.coverage_count > 1.0)
    assert set(np.unique(prior.best_camera_id)) <= {"", "camera_A", "camera_B"}


def test_single_camera_prior_exports_grid_provider() -> None:
    grid = PriorGridSpec.from_bounds(x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0, nx=5, ny=5)
    prior = build_camera_geometry_prior(_camera_a(), grid)
    provider = prior.to_grid_provider()

    quality = provider.query("camera_A", (0.0, 0.0))

    assert quality.camera_id == "camera_A"
    assert 0.0 <= quality.p_available <= 1.0
    assert quality.source_model.startswith("calibrated_multicamera_prior")


def test_large_prior_keeps_feature_maps_finite_outside_fov() -> None:
    grid = PriorGridSpec.from_bounds(x_min=-12.0, x_max=12.0, y_min=-8.0, y_max=8.0, nx=17, ny=13)
    prior = build_camera_geometry_prior(_camera_a(), grid)

    assert not np.all(prior.fov_mask)
    for values in prior.feature_maps.values():
        assert np.all(np.isfinite(values))


def test_evaluate_probability_grid_reports_blind_exposure() -> None:
    probability = np.asarray([[0.9, 0.8, 0.1], [0.7, 0.2, 0.1]], dtype=float)
    labels = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)

    summary = evaluate_probability_grid(probability, labels, reliability_threshold=0.5)

    assert summary.sample_count == 6
    assert summary.blind_exposure_fraction == pytest.approx(1.0 / 6.0)
    assert summary.predicted_reliable_fraction == pytest.approx(3.0 / 6.0)
    assert summary.auroc > 0.5
