"""Correctness (plumbing) tests for the E6 calibration-parameter perturbation.

Synthetic camera geometry only: these verify the fault-model MATH (re-projection,
range-dependent bias, per-camera targeting), not any warehouse result. No claim
rests on synthetic data (repo hard rule).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "reliability"))

from reliability.calibration_perturbation import (  # noqa: E402
    CalibrationPerturbation,
    PinholeGroundCamera,
    calibration_drift_world_bias,
    perturb,
    perturb_camera_calibration,
    reproject_world,
)
from reliability.contracts import CameraQuality, ContractValidationError  # noqa: E402
from reliability.fusion import MapObservation  # noqa: E402
from reliability.replay import ReplayFrame  # noqa: E402


def _cam():
    # Oblique camera at 6 m looking down-and-along +x to the ground at (4, 0).
    return PinholeGroundCamera.looking_at((0.0, 0.0, 6.0), (4.0, 0.0, 0.0), fov_h_deg=90.0)


def _mag(b):
    return math.hypot(b[0], b[1])


def _obs(camera_id, xy):
    return MapObservation(
        camera_id=camera_id,
        timestamp_s=0.0,
        xy_m=(float(xy[0]), float(xy[1])),
        covariance_m2=((0.05**2, 0.0), (0.0, 0.05**2)),
        quality=CameraQuality(camera_id=camera_id, p_available=0.9),
    )


def test_roundtrip_identity_and_center_projection():
    cam = _cam()
    for p in [(2.0, 0.0), (4.0, 0.0), (8.0, 0.0), (4.0, 2.0)]:
        px = cam.world_to_pixel(p)
        assert px is not None
        rt = cam.pixel_to_ground(*px)
        assert rt == pytest.approx(p, abs=1e-6)
    # The look-at point images at the principal centre.
    assert cam.world_to_pixel((4.0, 0.0)) == pytest.approx((cam.cx, cam.cy), abs=1e-6)


def test_identity_perturbation_is_noop():
    cam = _cam()
    pert = CalibrationPerturbation()
    assert pert.is_identity is True
    assert calibration_drift_world_bias(cam, (4.0, 0.0), pert) == (0.0, 0.0)
    assert reproject_world(cam, (3.0, 1.0), pert) == pytest.approx((3.0, 1.0), abs=1e-6)


def test_yaw_bias_grows_with_range():
    # THE paper-relevant property: the same angular drift barely moves a near
    # point and badly moves a far one (bias flows through camera geometry).
    cam = _cam()
    pert = CalibrationPerturbation(yaw_deg=2.0)
    b2 = _mag(calibration_drift_world_bias(cam, (2.0, 0.0), pert))
    b4 = _mag(calibration_drift_world_bias(cam, (4.0, 0.0), pert))
    b8 = _mag(calibration_drift_world_bias(cam, (8.0, 0.0), pert))
    assert 0.0 < b2 < b4 < b8


def test_focal_error_is_zero_at_principal_centre():
    cam = _cam()
    pert = CalibrationPerturbation(focal_scale=1.02)
    # (4,0) images at the principal centre -> a radial focal scaling leaves it put.
    assert _mag(calibration_drift_world_bias(cam, (4.0, 0.0), pert)) < 1e-6
    # An off-centre point is displaced.
    assert _mag(calibration_drift_world_bias(cam, (2.0, 3.0), pert)) > 1e-3


def test_translation_shifts_ground_point():
    cam = _cam()
    bias = calibration_drift_world_bias(cam, (4.0, 0.0), CalibrationPerturbation(tx_m=0.1))
    assert bias[0] == pytest.approx(0.1, abs=0.02)
    assert abs(bias[1]) < 1e-6


def test_point_behind_camera_projects_to_none():
    cam = _cam()
    assert cam.world_to_pixel((-10.0, 0.0)) is None
    # A non-projectable point yields no injected bias (left unchanged).
    assert calibration_drift_world_bias(cam, (-10.0, 0.0), CalibrationPerturbation(yaw_deg=5.0)) == (0.0, 0.0)


def test_frame_injector_targets_only_named_camera():
    cam = _cam()
    frames = [
        ReplayFrame(
            timestamp_s=0.0,
            odometry_xy_m=(4.0, 0.0),
            observations=(_obs("camA", (8.0, 0.0)), _obs("camB", (8.0, 0.0))),
        )
    ]
    out = perturb_camera_calibration(frames, "camA", cam, CalibrationPerturbation(yaw_deg=2.0))
    obs_by_cam = {o.camera_id: o for o in out[0].observations}
    assert obs_by_cam["camA"].xy_m != (8.0, 0.0)  # camA drifted
    assert obs_by_cam["camB"].xy_m == (8.0, 0.0)  # camB untouched
    assert obs_by_cam["camA"].covariance_m2 == ((0.05**2, 0.0), (0.0, 0.05**2))  # cov preserved
    assert "calib_drift" in obs_by_cam["camA"].source


def test_frame_injector_identity_is_noop():
    cam = _cam()
    frames = [ReplayFrame(timestamp_s=0.0, odometry_xy_m=(4.0, 0.0), observations=(_obs("camA", (5.0, 1.0)),))]
    out = perturb_camera_calibration(frames, "camA", cam, CalibrationPerturbation())
    assert out[0].observations[0].xy_m == (5.0, 1.0)


def test_compound_perturbation_runs_finite():
    cam = _cam()
    pert = CalibrationPerturbation(yaw_deg=1.0, pitch_deg=0.5, tx_m=0.02, focal_scale=1.005)
    bias = calibration_drift_world_bias(cam, (6.0, 1.0), pert)
    assert math.isfinite(bias[0]) and math.isfinite(bias[1])
    assert _mag(bias) > 0.0


def test_validation():
    with pytest.raises(ContractValidationError):
        CalibrationPerturbation(focal_scale=0.0)
    with pytest.raises(ContractValidationError):
        CalibrationPerturbation(yaw_deg=float("nan"))
    with pytest.raises(ContractValidationError):
        PinholeGroundCamera.looking_at((0.0, 0.0, 6.0), (0.0, 0.0, 0.0), fov_h_deg=90.0)  # looks along up axis
    with pytest.raises(ContractValidationError):
        PinholeGroundCamera.looking_at((0.0, 0.0, 6.0), (4.0, 0.0, 0.0), fov_h_deg=200.0)  # bad fov
