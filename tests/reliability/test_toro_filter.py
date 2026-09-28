"""Correctness (plumbing) tests for the Toro-style CV Kalman baseline (B2).

These are synthetic fixtures that verify the FILTER MATH — CV tracking,
nearest-point covariance controlling the gain, FOV gating, the B2a/B2b temporal
variants, paired-comparable metrics. They are explicitly NOT paper evidence:
no warehouse claim rests on synthetic data (repo hard rule). The containment /
"beats Toro" evidence comes only from the real replay campaign.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "reliability"))

from reliability.contracts import CameraQuality, ContractValidationError  # noqa: E402
from reliability.fusion import MapObservation  # noqa: E402
from reliability.replay import EvaluationFrame, ReplayFrame  # noqa: E402
from reliability.toro_baseline import CalibrationPoint, ToroCovarianceModel  # noqa: E402
from reliability.toro_filter import ToroFilterConfig, run_toro_filter  # noqa: E402


def _obs(camera_id, t, xy, cov=((0.05**2, 0.0), (0.0, 0.05**2))):
    return MapObservation(
        camera_id=camera_id,
        timestamp_s=float(t),
        xy_m=(float(xy[0]), float(xy[1])),
        covariance_m2=cov,
        quality=CameraQuality(camera_id=camera_id, p_available=0.9),
    )


def _box_model(camera_id="camA", std=0.06, xspan=(0.0, 10.0), yspan=(-1.0, 1.0)):
    """A validated-FOV box: 4 corner calibration points, isotropic covariance."""
    var = std * std
    cov = ((var, 0.0), (0.0, var))
    pts = [
        CalibrationPoint(camera_id, (xspan[0], yspan[0]), cov, 50),
        CalibrationPoint(camera_id, (xspan[0], yspan[1]), cov, 50),
        CalibrationPoint(camera_id, (xspan[1], yspan[0]), cov, 50),
        CalibrationPoint(camera_id, (xspan[1], yspan[1]), cov, 50),
    ]
    return ToroCovarianceModel(pts)


def test_tracks_constant_velocity_target():
    model = _box_model(xspan=(-1.0, 11.0))  # wide enough that no in-track measurement is FOV-gated
    times = [0.5 * i for i in range(21)]  # 0..10 s
    frames, evals = [], []
    for t in times:
        truth = (t, 0.0)  # 1 m/s along x
        meas = (t + (0.01 if int(t * 2) % 2 == 0 else -0.01), 0.0)
        frames.append(ReplayFrame(timestamp_s=t, odometry_xy_m=truth, observations=(_obs("camA", t, meas),)))
        evals.append(EvaluationFrame(timestamp_s=t, truth_xy_m=truth))

    result = run_toro_filter(frames, ToroFilterConfig(q=0.5), model, evaluation_frames=evals)
    assert result.metrics is not None
    assert result.rejected_total == 0
    assert result.accepted_total == len(frames)
    assert result.metrics.rmse_m < 0.5
    assert result.metrics.final_error_m < 0.1  # converged after the velocity transient


def test_larger_toro_covariance_resists_a_biased_measurement():
    # Stationary truth at origin; every measurement biased +0.5 m in x.
    times = [float(i) for i in range(6)]
    frames, evals = [], []
    for t in times:
        frames.append(
            ReplayFrame(timestamp_s=t, odometry_xy_m=(0.0, 0.0), observations=(_obs("camA", t, (0.5, 0.0)),))
        )
        evals.append(EvaluationFrame(timestamp_s=t, truth_xy_m=(0.0, 0.0)))

    tight = run_toro_filter(frames, ToroFilterConfig(q=0.05), _box_model(std=0.02), evaluation_frames=evals)
    loose = run_toro_filter(frames, ToroFilterConfig(q=0.05), _box_model(std=1.0), evaluation_frames=evals)

    # A tight (small) nearest-point covariance trusts the biased measurement and
    # is pulled toward it; a loose covariance down-weights it and stays nearer truth.
    assert tight.metrics.rmse_m > loose.metrics.rmse_m
    assert loose.metrics.rmse_m < 0.5


def test_covariance_lookup_is_region_dependent():
    var_lo, var_hi = 0.03**2, 0.5**2
    pts = [
        CalibrationPoint("camA", (2.5, -1.0), ((var_lo, 0.0), (0.0, var_lo)), 50),
        CalibrationPoint("camA", (2.5, 1.0), ((var_lo, 0.0), (0.0, var_lo)), 50),
        CalibrationPoint("camA", (7.5, -1.0), ((var_hi, 0.0), (0.0, var_hi)), 50),
        CalibrationPoint("camA", (7.5, 1.0), ((var_hi, 0.0), (0.0, var_hi)), 50),
    ]
    model = ToroCovarianceModel(pts)
    assert model.covariance_for("camA", (2.0, 0.0))[0][0] == pytest.approx(var_lo)
    assert model.covariance_for("camA", (8.0, 0.0))[0][0] == pytest.approx(var_hi)


def test_fov_gate_rejects_out_of_calibrated_region_measurements():
    model = _box_model(yspan=(-1.0, 1.0))
    frames = [
        ReplayFrame(timestamp_s=0.0, odometry_xy_m=(1.0, 0.0), observations=(_obs("camA", 0.0, (1.0, 0.0)),)),
        ReplayFrame(timestamp_s=1.0, odometry_xy_m=(2.0, 0.0), observations=(_obs("camA", 1.0, (2.0, 5.0)),)),  # y=5 outside
    ]
    gated = run_toro_filter(frames, ToroFilterConfig(fov_gate=True), model)
    assert gated.steps[1].rejected_camera_ids == ("camA",)
    assert gated.steps[0].accepted_camera_ids == ("camA",)

    ungated = run_toro_filter(frames, ToroFilterConfig(fov_gate=False), model)
    assert ungated.steps[1].accepted_camera_ids == ("camA",)


def test_camera_without_calibration_is_rejected():
    model = _box_model("camA")
    frames = [ReplayFrame(timestamp_s=0.0, odometry_xy_m=(1.0, 0.0), observations=(_obs("camB", 0.0, (1.0, 0.0)),))]
    result = run_toro_filter(frames, ToroFilterConfig(), model)
    assert result.steps[0].rejected_camera_ids == ("camB",)
    assert result.accepted_total == 0


def test_b2a_and_b2b_both_run_with_finite_metrics():
    model = _box_model()
    frames, evals = [], []
    for i in range(8):
        t = float(i)
        truth = (0.5 * i, 0.0)
        obs = (
            _obs("camA", t, (0.5 * i + 0.02, 0.0)),
            _obs("camA", t + 0.03, (0.5 * i - 0.02, 0.0)),  # same-frame, slightly later
        )
        frames.append(ReplayFrame(timestamp_s=t, odometry_xy_m=truth, observations=obs))
        evals.append(EvaluationFrame(timestamp_s=t, truth_xy_m=truth))

    for binning in ("b2a_simultaneous", "b2b_sequential"):
        result = run_toro_filter(
            frames, ToroFilterConfig(q=0.5, binning=binning), model, evaluation_frames=evals
        )
        assert result.metrics is not None
        assert result.metrics.rmse_m == pytest.approx(result.metrics.rmse_m)  # finite
        assert result.metrics.rmse_m < 1.0
        assert result.accepted_total > 0


def test_empty_frames_returns_no_metrics():
    result = run_toro_filter([], ToroFilterConfig(), _box_model())
    assert result.steps == ()
    assert result.metrics is None


def test_config_validation():
    with pytest.raises(ContractValidationError):
        ToroFilterConfig(q=-1.0)
    with pytest.raises(ContractValidationError):
        ToroFilterConfig(binning="nonsense")
    with pytest.raises(ContractValidationError):
        ToroFilterConfig(nis_gate=0.0)


def test_model_type_is_validated():
    frames = [ReplayFrame(timestamp_s=0.0, odometry_xy_m=(0.0, 0.0))]
    with pytest.raises(ContractValidationError):
        run_toro_filter(frames, ToroFilterConfig(), object())
