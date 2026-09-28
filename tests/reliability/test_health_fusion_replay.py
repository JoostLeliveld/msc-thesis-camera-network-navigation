from __future__ import annotations

from itertools import permutations
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "reliability"))

from reliability import (  # noqa: E402
    CameraHealthConfig,
    CameraHealthMachine,
    CameraHealthState,
    CameraManagerConfig,
    CameraObservation,
    CameraQuality,
    EvaluationFrame,
    FixedCameraReliabilityProvider,
    FixedZone,
    HandoverUncertaintyConfig,
    MapObservation,
    ReplayConfig,
    ReplayFrame,
    ReplayMode,
    camera_disagreement_m,
    conservative_camera_score,
    run_replay,
    select_conservative_best_camera,
    select_best_static_reliability,
    select_fixed_zone,
    select_freshest_valid,
    select_highest_detector_score,
    select_primary_camera,
    sequential_kalman_update_2d,
)


def _obs(camera_id: str, score: float, age: float = 0.0, p: float | None = None) -> CameraObservation:
    return CameraObservation(
        camera_id=camera_id,
        timestamp_s=1.0,
        pixel_uv=(10.0, 20.0),
        detection_valid=True,
        detector_score=score,
        measurement_age_s=age,
        availability_probability=score if p is None else p,
        association_probability=0.9,
    )


def _quality(camera_id: str, p: float, cov: float = 0.1**2) -> CameraQuality:
    return CameraQuality(
        camera_id=camera_id,
        p_available=p,
        conditional_cov_uv=((4.0, 0.0), (0.0, 4.0)),
        association_confidence=0.9,
        epistemic_score=0.0,
        source_model="test",
    )


def _map_obs(
    camera_id: str,
    x: float,
    y: float,
    p: float = 0.9,
    cov: float = 0.1**2,
    timestamp_s: float = 1.0,
) -> MapObservation:
    return MapObservation(
        camera_id=camera_id,
        timestamp_s=timestamp_s,
        xy_m=(x, y),
        covariance_m2=((cov, 0.0), (0.0, cov)),
        quality=_quality(camera_id, p),
    )


def test_camera_health_transitions_through_loss_and_reacquire() -> None:
    machine = CameraHealthMachine(
        camera_id="camera_A",
        config=CameraHealthConfig(
            stale_age_s=1.0,
            degraded_miss_count=1,
            lost_miss_count=2,
            reacquire_required_hits=2,
        ),
    )

    assert machine.update(_obs("camera_A", 0.8)).state == CameraHealthState.TRACKING
    assert machine.update(CameraObservation(camera_id="camera_A", timestamp_s=2.0)).state == CameraHealthState.DEGRADED
    assert machine.update(CameraObservation(camera_id="camera_A", timestamp_s=3.0)).state == CameraHealthState.LOST
    assert machine.update(_obs("camera_A", 0.8)).state == CameraHealthState.REACQUIRING
    assert machine.update(_obs("camera_A", 0.8)).state == CameraHealthState.TRACKING


def test_camera_health_high_nis_degrades_without_truth() -> None:
    machine = CameraHealthMachine(
        camera_id="camera_A",
        config=CameraHealthConfig(degraded_high_nis_count=1, lost_high_nis_count=2),
    )

    assert machine.update(_obs("camera_A", 0.8), nis=12.0, accepted=False).state == CameraHealthState.DEGRADED
    assert machine.update(_obs("camera_A", 0.8), nis=13.0, accepted=False).state == CameraHealthState.LOST


def test_selection_policies_choose_expected_observations() -> None:
    a = _obs("camera_A", 0.6, age=0.3, p=0.4)
    b = _obs("camera_B", 0.8, age=0.1, p=0.7)
    c = _obs("camera_C", 0.7, age=0.02, p=0.2)

    assert select_primary_camera([a, b], primary_camera_id="camera_A") == a
    assert select_highest_detector_score([a, b, c]) == b
    assert select_freshest_valid([a, b, c]) == c
    assert select_fixed_zone(
        [a, b],
        state_xy=(2.0, 2.0),
        zones=[FixedZone(camera_id="camera_B", xmin=1.0, xmax=3.0, ymin=1.0, ymax=3.0)],
    ) == b
    assert select_conservative_best_camera(
        [a, b],
        qualities={"camera_A": _quality("camera_A", 0.3), "camera_B": _quality("camera_B", 0.8)},
    ) == b
    assert conservative_camera_score(b, _quality("camera_B", 0.8)) > conservative_camera_score(a, _quality("camera_A", 0.3))


def test_selection_policy_ties_do_not_depend_on_callback_order() -> None:
    a = _obs("camera_A", 0.8, age=0.1, p=0.7)
    b = _obs("camera_B", 0.8, age=0.1, p=0.7)
    qualities = {
        "camera_A": _quality("camera_A", 0.7),
        "camera_B": _quality("camera_B", 0.7),
    }

    for observations in permutations((a, b)):
        assert select_primary_camera(observations, primary_camera_id="absent").camera_id == "camera_A"
        assert select_fixed_zone(observations, state_xy=(0.0, 0.0), zones=()).camera_id == "camera_A"
        assert select_highest_detector_score(observations).camera_id == "camera_A"
        assert select_freshest_valid(observations).camera_id == "camera_A"
        assert select_best_static_reliability(observations, qualities).camera_id == "camera_A"
        assert select_conservative_best_camera(observations, qualities).camera_id == "camera_A"


def test_sequential_fusion_accepts_near_observation_and_rejects_outlier() -> None:
    result = sequential_kalman_update_2d(
        (0.0, 0.0),
        ((0.2**2, 0.0), (0.0, 0.2**2)),
        [_map_obs("camera_A", 0.05, 0.0), _map_obs("camera_B", 5.0, 5.0)],
        nis_gate=9.21,
    )

    assert result.accepted_camera_ids == ("camera_A",)
    assert result.rejected_camera_ids == ("camera_B",)
    assert result.mean_xy[0] == pytest.approx(0.04)
    assert result.mean_xy[1] == pytest.approx(0.0)
    assert result.nis_by_camera["camera_B"] > 9.21


def test_camera_disagreement_reports_pairwise_max() -> None:
    assert camera_disagreement_m([_map_obs("camera_A", 0.0, 0.0), _map_obs("camera_B", 3.0, 4.0)]) == 5.0


def test_replay_runs_required_modes_and_computes_metrics() -> None:
    frames = (
        ReplayFrame(timestamp_s=0.0, odometry_xy_m=(0.0, 0.0), observations=(_map_obs("camera_A", 0.0, 0.0),)),
        ReplayFrame(timestamp_s=1.0, odometry_xy_m=(1.1, 0.0), observations=(_map_obs("camera_A", 1.0, 0.0),)),
        ReplayFrame(timestamp_s=2.0, odometry_xy_m=(2.2, 0.0), observations=(_map_obs("camera_A", 2.0, 0.0),)),
    )
    evaluation = (
        EvaluationFrame(timestamp_s=0.0, truth_xy_m=(0.0, 0.0)),
        EvaluationFrame(timestamp_s=1.0, truth_xy_m=(1.0, 0.0)),
        EvaluationFrame(timestamp_s=2.0, truth_xy_m=(2.0, 0.0)),
    )

    odom_only = run_replay(frames, ReplayConfig(mode=ReplayMode.ODOM_ONLY), evaluation_frames=evaluation)
    fixed_r = run_replay(frames, ReplayConfig(mode=ReplayMode.SINGLE_FIXED_R_NIS, nis_gate=9.21), evaluation_frames=evaluation)

    assert odom_only.steps[-1].mean_xy_m == pytest.approx((2.2, 0.0))
    assert fixed_r.steps[-1].mean_xy_m[0] < odom_only.steps[-1].mean_xy_m[0]
    assert fixed_r.metrics is not None
    assert odom_only.metrics is not None
    assert fixed_r.metrics.rmse_m < odom_only.metrics.rmse_m


def test_replay_gates_large_bad_measurement() -> None:
    frames = (
        ReplayFrame(timestamp_s=0.0, odometry_xy_m=(0.0, 0.0), observations=tuple()),
        ReplayFrame(timestamp_s=1.0, odometry_xy_m=(1.0, 0.0), observations=(_map_obs("camera_A", 100.0, 100.0, cov=0.01**2),)),
    )

    result = run_replay(
        frames,
        ReplayConfig(mode=ReplayMode.SINGLE_FIXED_R_NIS, nis_gate=9.21, fixed_measurement_cov_m2=((0.01**2, 0.0), (0.0, 0.01**2))),
    )

    assert result.steps[-1].mean_xy_m == pytest.approx((1.0, 0.0))
    assert result.steps[-1].rejected_camera_ids == ("camera_A",)


def test_replay_current_gp_mode_uses_quality_provider_probability() -> None:
    frames = (
        ReplayFrame(timestamp_s=0.0, odometry_xy_m=(0.0, 0.0), observations=tuple()),
        ReplayFrame(timestamp_s=1.0, odometry_xy_m=(1.0, 0.0), observations=(_map_obs("camera_A", 0.0, 0.0),)),
    )
    low = run_replay(
        frames,
        ReplayConfig(
            mode=ReplayMode.CURRENT_GP_R,
            nis_gate=None,
            quality_providers={
                "camera_A": FixedCameraReliabilityProvider(camera_id="camera_A", p_available=0.1),
            },
        ),
    )
    high = run_replay(
        frames,
        ReplayConfig(
            mode=ReplayMode.CURRENT_GP_R,
            nis_gate=None,
            quality_providers={
                "camera_A": FixedCameraReliabilityProvider(camera_id="camera_A", p_available=0.95),
            },
        ),
    )

    assert high.steps[-1].mean_xy_m[0] < low.steps[-1].mean_xy_m[0]


def test_replay_handover_aware_selection_inflates_disagreeing_switch() -> None:
    frames = (
        ReplayFrame(
            timestamp_s=0.0,
            odometry_xy_m=(0.0, 0.0),
            observations=(_map_obs("camera_A", 0.0, 0.0, p=0.9),),
        ),
        ReplayFrame(
            timestamp_s=1.0,
            odometry_xy_m=(1.0, 0.0),
            observations=(
                _map_obs("camera_A", 1.0, 0.0, p=0.9),
                _map_obs("camera_B", 1.6, 0.0, p=0.99),
            ),
        ),
    )
    immediate = run_replay(frames, ReplayConfig(mode=ReplayMode.CONSERVATIVE_SELECTION, nis_gate=None))
    handover = run_replay(
        frames,
        ReplayConfig(
            mode=ReplayMode.HANDOVER_AWARE_SELECTION,
            nis_gate=None,
            handover_config=HandoverUncertaintyConfig(disagreement_gate_m=0.30, max_covariance_inflation=10.0),
        ),
    )

    assert immediate.steps[-1].accepted_camera_ids == ("camera_B",)
    assert handover.steps[-1].accepted_camera_ids == ("camera_B",)
    assert handover.steps[-1].handover_diagnostics[0]["switched"] is True
    assert "overlap_disagreement" in handover.steps[-1].handover_diagnostics[0]["reasons"]
    assert handover.steps[-1].handover_diagnostics[0]["covariance_inflation"] > 1.0
    assert handover.steps[-1].mean_xy_m[0] < immediate.steps[-1].mean_xy_m[0]


def test_replay_hysteretic_handover_holds_then_switches_and_records_manager_decision() -> None:
    frames = (
        ReplayFrame(
            timestamp_s=0.0,
            odometry_xy_m=(0.0, 0.0),
            observations=(_map_obs("camera_A", 0.0, 0.0, p=0.80, timestamp_s=0.0),),
        ),
        ReplayFrame(
            timestamp_s=1.0,
            odometry_xy_m=(1.0, 0.0),
            observations=(_map_obs("camera_A", 1.0, 0.0, p=0.70), _map_obs("camera_B", 1.02, 0.0, p=0.95)),
        ),
        ReplayFrame(
            timestamp_s=2.0,
            odometry_xy_m=(2.0, 0.0),
            observations=(
                # Stamps must track the frame time: the manager age-gates at
                # max_measurement_age_s, so second-frame observations carrying
                # the helper's default 1.0 s stamp would all be stale.
                _map_obs("camera_A", 2.0, 0.0, p=0.70, timestamp_s=2.0),
                _map_obs("camera_B", 2.02, 0.0, p=0.95, timestamp_s=2.0),
            ),
        ),
    )

    result = run_replay(
        frames,
        ReplayConfig(
            mode=ReplayMode.HYSTERETIC_HANDOVER_SELECTION,
            nis_gate=None,
            camera_manager_config=CameraManagerConfig(
                min_spatial_trust=0.10,
                candidate_score_margin=0.05,
                required_consecutive_better_frames=2,
            ),
        ),
    )

    assert result.steps[1].accepted_camera_ids == ("camera_A",)
    assert result.steps[1].camera_manager_decision["candidate_streak"] == 1
    assert result.steps[2].accepted_camera_ids == ("camera_B",)
    assert result.steps[2].camera_manager_decision["switched"] is True
    assert result.steps[2].handover_diagnostics[0]["switched"] is True


def test_replay_hysteretic_selection_uses_camera_specific_reliability_provider() -> None:
    frame = ReplayFrame(
        timestamp_s=0.0,
        odometry_xy_m=(0.0, 0.0),
        observations=(
            _map_obs("camera_A", 0.0, 0.0, p=0.50, timestamp_s=0.0),
            _map_obs("camera_B", 0.0, 0.0, p=0.99, timestamp_s=0.0),
        ),
    )

    result = run_replay(
        (frame,),
        ReplayConfig(
            mode=ReplayMode.HYSTERETIC_HANDOVER_SELECTION,
            nis_gate=None,
            quality_providers={
                "camera_A": FixedCameraReliabilityProvider(camera_id="camera_A", p_available=0.95),
                "camera_B": FixedCameraReliabilityProvider(camera_id="camera_B", p_available=0.05),
            },
            camera_manager_config=CameraManagerConfig(min_spatial_trust=0.10),
        ),
    )

    assert result.steps[0].accepted_camera_ids == ("camera_A",)
    assert result.steps[0].camera_manager_decision["selected_camera_id"] == "camera_A"
