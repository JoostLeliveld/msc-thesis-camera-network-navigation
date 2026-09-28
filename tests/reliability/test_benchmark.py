from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "reliability"))

from reliability import (  # noqa: E402
    CameraQuality,
    EvaluationFrame,
    MapObservation,
    ReplayFrame,
    ReplayMode,
    default_replay_benchmark_conditions,
    has_multicamera_observations,
    run_replay_benchmark,
)


def _quality(camera_id: str, p: float = 0.9) -> CameraQuality:
    return CameraQuality(
        camera_id=camera_id,
        p_available=p,
        conditional_cov_uv=((6.25, 0.0), (0.0, 6.25)),
        association_confidence=0.95,
        source_model="test",
    )


def _map_obs(camera_id: str, x: float, y: float, cov: float = 0.08**2) -> MapObservation:
    return MapObservation(
        camera_id=camera_id,
        timestamp_s=1.0,
        xy_m=(x, y),
        covariance_m2=((cov, 0.0), (0.0, cov)),
        quality=_quality(camera_id),
    )


def test_default_benchmark_conditions_include_required_replay_modes() -> None:
    single = default_replay_benchmark_conditions()
    multi = default_replay_benchmark_conditions(include_multicamera=True)

    assert [condition.config.mode for condition in single] == [
        ReplayMode.ODOM_ONLY,
        ReplayMode.SINGLE_FIXED_R,
        ReplayMode.SINGLE_FIXED_R_NIS,
        ReplayMode.DETECTOR_SCORE_R,
        ReplayMode.CURRENT_GP_R,
    ]
    assert [condition.config.mode for condition in multi[-4:]] == [
        ReplayMode.SEQUENTIAL_FUSION,
        ReplayMode.CONSERVATIVE_SELECTION,
        ReplayMode.HANDOVER_AWARE_SELECTION,
        ReplayMode.HYSTERETIC_HANDOVER_SELECTION,
    ]


def test_run_benchmark_auto_adds_multicamera_fusion_modes() -> None:
    frames = (
        ReplayFrame(
            timestamp_s=0.0,
            odometry_xy_m=(0.0, 0.0),
            observations=(_map_obs("camera_A", 0.0, 0.0),),
        ),
        ReplayFrame(
            timestamp_s=1.0,
            odometry_xy_m=(1.1, 0.0),
            observations=(
                _map_obs("camera_A", 1.0, 0.0),
                _map_obs("camera_B", 1.02, 0.0),
            ),
        ),
    )
    evaluation = (
        EvaluationFrame(timestamp_s=0.0, truth_xy_m=(0.0, 0.0)),
        EvaluationFrame(timestamp_s=1.0, truth_xy_m=(1.0, 0.0)),
    )

    assert has_multicamera_observations(frames)
    suite = run_replay_benchmark(frames, evaluation_frames=evaluation)
    summary = suite.summary()

    assert ReplayMode.SEQUENTIAL_FUSION.value in summary["results"]
    assert ReplayMode.CONSERVATIVE_SELECTION.value in summary["results"]
    assert ReplayMode.HANDOVER_AWARE_SELECTION.value in summary["results"]
    assert ReplayMode.HYSTERETIC_HANDOVER_SELECTION.value in summary["results"]
    assert summary["results"][ReplayMode.ODOM_ONLY.value]["rmse_m"] is not None
    assert summary["results"][ReplayMode.ODOM_ONLY.value]["p95_error_m"] is not None
    assert summary["results"][ReplayMode.ODOM_ONLY.value]["covariance_2sigma_coverage"] is not None
    assert summary["results"][ReplayMode.SEQUENTIAL_FUSION.value]["steps"] == 2


def test_run_benchmark_can_force_single_camera_suite() -> None:
    frames = (
        ReplayFrame(
            timestamp_s=0.0,
            odometry_xy_m=(0.0, 0.0),
            observations=(_map_obs("camera_A", 0.0, 0.0), _map_obs("camera_B", 0.0, 0.0)),
        ),
    )

    suite = run_replay_benchmark(frames, include_multicamera=False)
    names = [item.condition.name for item in suite.results]

    assert ReplayMode.SEQUENTIAL_FUSION.value not in names
    assert ReplayMode.HANDOVER_AWARE_SELECTION.value not in names
    assert ReplayMode.HYSTERETIC_HANDOVER_SELECTION.value not in names
    assert ReplayMode.CURRENT_GP_R.value in names
