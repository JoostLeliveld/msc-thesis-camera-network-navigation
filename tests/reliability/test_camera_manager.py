from __future__ import annotations

import sys
from pathlib import Path
import math

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "reliability"))

from reliability import (  # noqa: E402
    CameraManager,
    CameraManagerConfig,
    CameraQuality,
    ContractValidationError,
    MapObservation,
    observation_age_s,
    operational_camera_score,
)


def _obs(
    camera_id: str,
    *,
    timestamp_s: float,
    x: float = 0.0,
    y: float = 0.0,
    p: float = 0.90,
    association: float = 0.95,
    epistemic: float = 0.0,
    stale: bool = False,
) -> MapObservation:
    return MapObservation(
        camera_id=camera_id,
        timestamp_s=timestamp_s,
        xy_m=(x, y),
        covariance_m2=((0.1**2, 0.0), (0.0, 0.1**2)),
        quality=CameraQuality(
            camera_id=camera_id,
            p_available=p,
            conditional_cov_uv=((4.0, 0.0), (0.0, 4.0)),
            association_confidence=association,
            epistemic_score=epistemic,
            stale=stale,
            source_model="test",
        ),
    )


def _manager(**kwargs) -> CameraManager:
    settings = dict(
        min_spatial_trust=0.10,
        min_association_confidence=0.50,
        max_measurement_age_s=0.20,
        candidate_score_margin=0.05,
        required_consecutive_better_frames=3,
        max_cross_camera_disagreement_m=0.30,
    )
    settings.update(kwargs)
    return CameraManager(CameraManagerConfig(**settings))


def test_initial_selection_uses_highest_operational_score() -> None:
    manager = _manager()

    decision = manager.select(
        timestamp_s=1.0,
        observations=(_obs("camera_A", timestamp_s=1.0, p=0.70), _obs("camera_B", timestamp_s=1.0, p=0.95)),
    )

    assert decision.selected_camera_id == "camera_B"
    assert decision.reasons == ("initial_selection",)
    assert not decision.switched


def test_eligible_observations_is_non_mutating_and_applies_all_release_gates() -> None:
    manager = _manager()
    candidates, scores, rejected = manager.eligible_observations(
        timestamp_s=1.0,
        observations=(
            _obs("camera_A", timestamp_s=1.0, p=0.90),
            _obs("camera_B", timestamp_s=0.70, p=0.95),
            _obs("camera_C", timestamp_s=1.0, association=0.20),
            _obs("camera_D", timestamp_s=1.0, p=0.01),
        ),
    )

    assert set(candidates) == {"camera_A"}
    assert scores["camera_A"] > 0.0
    assert "measurement_stale" in rejected["camera_B"]
    assert "association_below_minimum" in rejected["camera_C"]
    assert "spatial_trust_below_minimum" in rejected["camera_D"]
    assert manager.active_camera_id == ""


def test_handover_requires_same_better_candidate_for_required_frames() -> None:
    manager = _manager()
    assert manager.select(timestamp_s=0.0, observations=(_obs("camera_A", timestamp_s=0.0, p=0.80),)).selected_camera_id == "camera_A"

    first = manager.select(
        timestamp_s=1.0,
        observations=(_obs("camera_A", timestamp_s=1.0, p=0.70), _obs("camera_B", timestamp_s=1.0, p=0.95)),
    )
    second = manager.select(
        timestamp_s=2.0,
        observations=(_obs("camera_A", timestamp_s=2.0, p=0.70), _obs("camera_B", timestamp_s=2.0, p=0.95)),
    )
    third = manager.select(
        timestamp_s=3.0,
        observations=(_obs("camera_A", timestamp_s=3.0, p=0.70), _obs("camera_B", timestamp_s=3.0, p=0.95)),
    )

    assert first.selected_camera_id == second.selected_camera_id == "camera_A"
    assert first.candidate_camera_id == second.candidate_camera_id == "camera_B"
    assert (first.candidate_streak, second.candidate_streak) == (1, 2)
    assert third.selected_camera_id == "camera_B"
    assert third.switched and third.reasons == ("hysteresis_handover",)


def test_handover_hysteresis_counts_unique_candidate_frames_not_timer_ticks() -> None:
    manager = _manager(required_consecutive_better_frames=2)
    manager.select(timestamp_s=0.0, observations=(_obs("camera_A", timestamp_s=0.0, p=0.80),))

    first = manager.select(
        timestamp_s=1.0,
        observations=(
            _obs("camera_A", timestamp_s=1.0, p=0.70),
            _obs("camera_B", timestamp_s=1.0, p=0.95),
        ),
    )
    repeated = manager.select(
        timestamp_s=1.05,
        observations=(
            _obs("camera_A", timestamp_s=1.0, p=0.70),
            _obs("camera_B", timestamp_s=1.0, p=0.95),
        ),
    )
    newer = manager.select(
        timestamp_s=1.10,
        observations=(
            _obs("camera_A", timestamp_s=1.10, p=0.70),
            _obs("camera_B", timestamp_s=1.10, p=0.95),
        ),
    )

    assert first.selected_camera_id == repeated.selected_camera_id == "camera_A"
    assert first.candidate_streak == repeated.candidate_streak == 1
    assert newer.selected_camera_id == "camera_B"
    assert newer.switched


def test_alternating_candidates_do_not_accumulate_a_handover_streak() -> None:
    manager = _manager(required_consecutive_better_frames=2)
    manager.select(timestamp_s=0.0, observations=(_obs("camera_A", timestamp_s=0.0, p=0.80),))

    b = manager.select(
        timestamp_s=1.0,
        observations=(_obs("camera_A", timestamp_s=1.0, p=0.70), _obs("camera_B", timestamp_s=1.0, p=0.95)),
    )
    c = manager.select(
        timestamp_s=2.0,
        observations=(
            _obs("camera_A", timestamp_s=2.0, p=0.70),
            _obs("camera_B", timestamp_s=2.0, p=0.75),
            _obs("camera_C", timestamp_s=2.0, p=0.95),
        ),
    )

    assert b.selected_camera_id == c.selected_camera_id == "camera_A"
    assert b.candidate_camera_id == "camera_B"
    assert c.candidate_camera_id == "camera_C"
    assert c.candidate_streak == 1


def test_inconsistent_candidate_cannot_replace_healthy_active_camera() -> None:
    manager = _manager(required_consecutive_better_frames=1)
    manager.select(timestamp_s=0.0, observations=(_obs("camera_A", timestamp_s=0.0, x=0.0, p=0.80),))

    decision = manager.select(
        timestamp_s=1.0,
        observations=(
            _obs("camera_A", timestamp_s=1.0, x=1.0, p=0.70),
            _obs("camera_B", timestamp_s=1.0, x=1.7, p=0.95),
        ),
    )

    assert decision.selected_camera_id == "camera_A"
    assert decision.reasons == ("cross_camera_disagreement",)
    assert "cross_camera_disagreement" in decision.rejected_by_camera["camera_B"]


def test_stale_active_camera_immediately_uses_valid_fallback() -> None:
    manager = _manager()
    manager.select(timestamp_s=0.0, observations=(_obs("camera_A", timestamp_s=0.0, p=0.90),))

    decision = manager.select(
        timestamp_s=1.0,
        observations=(
            _obs("camera_A", timestamp_s=0.5, p=0.90),
            _obs("camera_B", timestamp_s=1.0, p=0.85),
        ),
    )

    assert decision.selected_camera_id == "camera_B"
    assert decision.switched and decision.fallback
    assert decision.reasons == ("active_camera_ineligible", "fallback_selection")
    assert "measurement_stale" in decision.rejected_by_camera["camera_A"]


def test_score_uses_age_and_epistemic_penalties() -> None:
    observation = _obs("camera_A", timestamp_s=0.5, p=1.0, association=1.0, epistemic=math.log(2.0))
    assert observation_age_s(observation, timestamp_s=1.0) == pytest.approx(0.5)
    assert operational_camera_score(observation, timestamp_s=1.0, age_decay_s=0.5) == pytest.approx(0.5 / math.e)


def test_future_dated_observation_is_ineligible_instead_of_perfectly_fresh() -> None:
    observation = _obs("camera_A", timestamp_s=1.01, p=1.0, association=1.0)
    assert math.isinf(observation_age_s(observation, timestamp_s=1.0))
    assert operational_camera_score(observation, timestamp_s=1.0, age_decay_s=0.5) == 0.0

    decision = _manager().select(timestamp_s=1.0, observations=(observation,))

    assert decision.selected_observation is None
    assert "measurement_future_dated" in decision.rejected_by_camera["camera_A"]


def test_manager_config_rejects_invalid_thresholds_and_camera_ids() -> None:
    with pytest.raises(ContractValidationError):
        CameraManagerConfig(required_consecutive_better_frames=0)
    with pytest.raises(ContractValidationError):
        CameraManagerConfig(min_spatial_trust=1.01)
    with pytest.raises(ContractValidationError):
        CameraManagerConfig(allowed_camera_ids=("camera_A", "camera_A"))


def test_manager_rejects_two_votes_from_one_physical_camera() -> None:
    observation = _obs("camera_A", timestamp_s=1.0)
    with pytest.raises(ContractValidationError, match="duplicate camera"):
        _manager().select(timestamp_s=1.0, observations=(observation, observation))


def test_equal_selection_scores_have_a_stable_camera_tiebreak() -> None:
    a = _obs("camera_A", timestamp_s=1.0, p=0.9)
    b = _obs("camera_B", timestamp_s=1.0, p=0.9)
    assert _manager().select(timestamp_s=1.0, observations=(a, b)).selected_camera_id == "camera_A"
    assert _manager().select(timestamp_s=1.0, observations=(b, a)).selected_camera_id == "camera_A"


def test_manager_cannot_receive_ground_truth_through_the_camera_quality_contract() -> None:
    with pytest.raises(ContractValidationError, match="evaluation-only"):
        CameraQuality.from_dict(
            {
                "camera_id": "camera_A",
                "p_available": 0.9,
                "conditional_cov_uv": [[4.0, 0.0], [0.0, 4.0]],
                "ground_truth_localization_error_m": 0.01,
            }
        )
