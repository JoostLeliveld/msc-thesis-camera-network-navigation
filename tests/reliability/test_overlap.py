from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "reliability"))

from reliability import (  # noqa: E402
    CameraQuality,
    ContractValidationError,
    MapObservation,
    ReplayFrame,
    match_overlap_observations,
    overlap_trust_from_frames,
    summarize_overlap_trust,
    validate_camera_overlap,
)


def _quality(camera_id: str) -> CameraQuality:
    return CameraQuality(
        camera_id=camera_id,
        p_available=0.9,
        conditional_cov_uv=((6.25, 0.0), (0.0, 6.25)),
        association_confidence=0.95,
        source_model="test",
    )


def _obs(camera_id: str, stamp: float, x: float, y: float) -> MapObservation:
    return MapObservation(
        camera_id=camera_id,
        timestamp_s=stamp,
        xy_m=(x, y),
        covariance_m2=((0.08**2, 0.0), (0.0, 0.08**2)),
        quality=_quality(camera_id),
    )


def test_match_overlap_observations_pairs_near_timestamps() -> None:
    pairs = match_overlap_observations(
        [
            _obs("camera_A", 1.00, 1.0, 0.0),
            _obs("camera_B", 1.03, 1.1, 0.0),
            _obs("camera_B", 2.00, 3.0, 0.0),
        ],
        max_time_delta_s=0.05,
    )

    assert len(pairs) == 1
    assert pairs[0].disagreement_m == pytest.approx(0.1)
    assert pairs[0].to_dict()["camera_a_id"] == "camera_A"


def test_validate_camera_overlap_reports_gate_result() -> None:
    frames = (
        ReplayFrame(
            timestamp_s=1.0,
            odometry_xy_m=(0.0, 0.0),
            observations=(
                _obs("camera_A", 1.0, 0.0, 0.0),
                _obs("camera_B", 1.0, 0.2, 0.0),
            ),
        ),
    )

    ok = validate_camera_overlap(frames, max_allowed_disagreement_m=0.25)
    bad = validate_camera_overlap(frames, max_allowed_disagreement_m=0.15)

    assert ok.pass_gate is True
    assert bad.pass_gate is False
    assert ok.to_dict()["pair_count"] == 1
    assert ok.rms_disagreement_m == pytest.approx(0.2)


def test_overlap_trust_reports_bias_and_outlier_rate() -> None:
    pairs = match_overlap_observations(
        [
            _obs("camera_A", 1.00, 0.0, 0.0),
            _obs("camera_B", 1.00, 0.1, 0.0),
            _obs("camera_A", 2.00, 1.0, 0.0),
            _obs("camera_B", 2.00, 1.1, 0.0),
            _obs("camera_A", 3.00, 2.0, 0.0),
            _obs("camera_B", 3.00, 3.0, 0.0),
        ],
        max_time_delta_s=0.01,
    )

    trust = summarize_overlap_trust(
        pairs,
        disagreement_gate_m=0.25,
        min_pair_count=3,
        max_outlier_rate=0.40,
    )

    assert trust.pair_count == 3
    assert trust.outlier_rate == pytest.approx(1.0 / 3.0)
    assert trust.mean_delta_b_minus_a_m[0] == pytest.approx(0.4)
    assert 0.0 < trust.pair_trust_score < 1.0
    assert trust.pass_gate is False
    assert trust.to_dict()["bias_norm_m"] == pytest.approx(0.4)


def test_overlap_trust_from_frames_uses_replay_frames() -> None:
    frames = (
        ReplayFrame(
            timestamp_s=1.0,
            odometry_xy_m=(0.0, 0.0),
            observations=(
                _obs("camera_A", 1.0, 0.0, 0.0),
                _obs("camera_B", 1.0, 0.05, 0.0),
            ),
        ),
    )

    trust = overlap_trust_from_frames(frames, disagreement_gate_m=0.10)

    assert trust.pass_gate is True
    assert trust.p90_disagreement_m == pytest.approx(0.05)


def test_validate_camera_overlap_rejects_bad_thresholds() -> None:
    with pytest.raises(ContractValidationError, match="non-negative"):
        validate_camera_overlap([], max_time_delta_s=-1.0)
