"""Regression coverage for D1 dynamic-occlusion-aware camera fusion.

This is a deterministic synthetic test of the *operational* mechanism.  Actor
tracks are supplied to the filter; ground truth is supplied only afterwards to
score the replay.  It is not a replacement for a live Gazebo dynamic-actor run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "reliability"))

from reliability import (  # noqa: E402
    CameraQuality,
    DynamicActorState,
    DynamicOcclusionConfig,
    EvaluationFrame,
    MapObservation,
    ReplayConfig,
    ReplayFrame,
    ReplayMode,
    dynamic_occlusion_probability,
    run_replay,
)
from reliability.contracts import ContractValidationError  # noqa: E402


def _observation(t: float, xy: tuple[float, float]) -> MapObservation:
    return MapObservation(
        camera_id="camera_A", timestamp_s=t, xy_m=xy,
        covariance_m2=((0.08 ** 2, 0.0), (0.0, 0.08 ** 2)),
        quality=CameraQuality(camera_id="camera_A", p_available=0.95),
        source="synthetic_dynamic_regression",
    )


def _dynamic_crossing_replay():
    frames, evaluation = [], []
    for i in range(45):
        t = i * 0.2
        truth = (0.04 * i, 2.0)
        # A tracked person is exactly on the camera-to-robot ray for 16 frames.
        blocked = 14 <= i < 30
        actors = (
            DynamicActorState("person_0", (truth[0] * 0.5, -1.0), radius_m=0.35, confidence=0.98),
        ) if blocked else tuple()
        # The detector's small association displacement remains below the usual
        # per-frame NIS gate, so a static fusion baseline keeps accepting it.
        measured = (truth[0] + (0.23 if blocked else 0.0), truth[1])
        odom = (truth[0] + 0.008 * i, truth[1])
        frames.append(ReplayFrame(t, odom, (_observation(t, measured),), actors))
        evaluation.append(EvaluationFrame(t, truth))
    return tuple(frames), tuple(evaluation)


def test_actor_on_ray_has_high_occlusion_probability() -> None:
    probability, blockers = dynamic_occlusion_probability(
        (0.0, -4.0), (2.0, 2.0),
        (DynamicActorState("forklift", (1.0, -1.0), radius_m=0.40, confidence=0.9),),
    )
    assert probability == pytest.approx(0.9)
    assert blockers == ("forklift",)


def test_dynamic_occlusion_fusion_contains_gate_evading_association_error() -> None:
    frames, evaluation = _dynamic_crossing_replay()
    baseline = run_replay(
        frames,
        ReplayConfig(mode=ReplayMode.SEQUENTIAL_FUSION, nis_gate=9.21),
        evaluation_frames=evaluation,
    )
    dynamic = run_replay(
        frames,
        ReplayConfig(
            mode=ReplayMode.DYNAMIC_OCCLUSION_AWARE_FUSION,
            nis_gate=9.21,
            dynamic_occlusion_config=DynamicOcclusionConfig(
                camera_xy_m={"camera_A": (0.0, -4.0)}, covariance_gain=8.0,
            ),
        ),
        evaluation_frames=evaluation,
    )
    assert baseline.metrics is not None and dynamic.metrics is not None
    assert dynamic.metrics.rmse_m < baseline.metrics.rmse_m
    blocked = dynamic.steps[16].dynamic_occlusion_diagnostics[0]
    assert blocked["occlusion_probability"] > 0.9
    assert blocked["covariance_inflation"] > 8.0


def test_dynamic_mode_requires_calibrated_actor_response() -> None:
    frames, _ = _dynamic_crossing_replay()
    with pytest.raises(ContractValidationError, match="dynamic_occlusion_config"):
        run_replay(frames[:1], ReplayConfig(mode=ReplayMode.DYNAMIC_OCCLUSION_AWARE_FUSION))
