from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "reliability"))

from reliability import (  # noqa: E402
    CameraQuality,
    ContractValidationError,
    HandoverUncertaintyConfig,
    MapObservation,
    assess_handover_uncertainty,
    handover_adjusted_observation,
    inflate_map_observation_covariance,
)


def _quality(camera_id: str, p: float = 0.9, *, stale: bool = False, epistemic: float = 0.0) -> CameraQuality:
    return CameraQuality(
        camera_id=camera_id,
        p_available=p,
        conditional_cov_uv=((6.25, 0.0), (0.0, 6.25)),
        association_confidence=0.95,
        epistemic_score=epistemic,
        stale=stale,
        source_model="test",
    )


def _obs(camera_id: str, stamp: float, x: float, y: float, p: float = 0.9, cov: float = 0.08**2) -> MapObservation:
    return MapObservation(
        camera_id=camera_id,
        timestamp_s=stamp,
        xy_m=(x, y),
        covariance_m2=((cov, 0.0), (0.0, cov)),
        quality=_quality(camera_id, p),
        source="test",
    )


def test_same_camera_has_low_handover_uncertainty() -> None:
    selected = _obs("camera_A", 1.0, 0.0, 0.0)

    diagnostic = assess_handover_uncertainty(
        previous_camera_id="camera_A",
        selected_observation=selected,
        candidate_observations=(selected,),
    )

    assert diagnostic.switched is False
    assert diagnostic.overlap_available is False
    assert diagnostic.uncertainty_score < 0.05
    assert diagnostic.covariance_inflation < 1.5
    assert "camera_switch" not in diagnostic.reasons


def test_clean_overlap_handover_has_modest_inflation() -> None:
    a = _obs("camera_A", 1.00, 1.00, 0.00)
    b = _obs("camera_B", 1.02, 1.03, 0.00)

    diagnostic = assess_handover_uncertainty(
        previous_camera_id="camera_A",
        selected_observation=b,
        candidate_observations=(a, b),
    )

    assert diagnostic.switched is True
    assert diagnostic.overlap_available is True
    assert diagnostic.disagreement_m == pytest.approx(0.03)
    assert diagnostic.age_gap_s == pytest.approx(0.02)
    assert "camera_switch" in diagnostic.reasons
    assert "overlap_disagreement" not in diagnostic.reasons
    assert 1.0 < diagnostic.covariance_inflation < 4.0


def test_disagreeing_overlap_handover_inflates_covariance() -> None:
    a = _obs("camera_A", 1.00, 0.00, 0.00)
    b = _obs("camera_B", 1.00, 0.80, 0.00)
    cfg = HandoverUncertaintyConfig(disagreement_gate_m=0.30, max_covariance_inflation=10.0)

    diagnostic = assess_handover_uncertainty(
        previous_camera_id="camera_A",
        selected_observation=b,
        candidate_observations=(a, b),
        config=cfg,
    )

    assert diagnostic.overlap_available is True
    assert diagnostic.disagreement_m == pytest.approx(0.8)
    assert "overlap_disagreement" in diagnostic.reasons
    assert diagnostic.covariance_inflation >= 5.0


def test_switch_without_overlap_is_reported_as_unconfirmed() -> None:
    previous = _obs("camera_A", 0.70, 0.0, 0.0)
    selected = _obs("camera_B", 1.00, 0.1, 0.0)

    diagnostic = assess_handover_uncertainty(
        previous_camera_id="camera_A",
        previous_observation=previous,
        selected_observation=selected,
        candidate_observations=(selected,),
    )

    assert diagnostic.switched is True
    assert diagnostic.overlap_available is False
    assert diagnostic.age_gap_s == pytest.approx(0.30)
    assert "no_overlap_confirmation" in diagnostic.reasons
    assert diagnostic.covariance_inflation > 4.0


def test_low_quality_stale_selected_camera_adds_handover_risk() -> None:
    selected = MapObservation(
        camera_id="camera_B",
        timestamp_s=1.0,
        xy_m=(0.0, 0.0),
        covariance_m2=((0.08**2, 0.0), (0.0, 0.08**2)),
        quality=_quality("camera_B", p=0.35, stale=True, epistemic=0.5),
    )

    diagnostic = assess_handover_uncertainty(
        previous_camera_id="camera_A",
        selected_observation=selected,
        candidate_observations=(selected,),
    )

    assert "low_selected_quality" in diagnostic.reasons
    assert "selected_stale" in diagnostic.reasons
    assert diagnostic.selected_quality_score < 0.2
    assert diagnostic.covariance_inflation > 5.0


def test_handover_adjusted_observation_preserves_spd_and_source() -> None:
    a = _obs("camera_A", 1.0, 0.0, 0.0)
    b = _obs("camera_B", 1.0, 0.6, 0.0)

    adjusted, diagnostic = handover_adjusted_observation(
        previous_camera_id="camera_A",
        selected_observation=b,
        candidate_observations=(a, b),
        config=HandoverUncertaintyConfig(disagreement_gate_m=0.30, max_covariance_inflation=8.0),
    )

    assert adjusted is not None
    assert diagnostic.covariance_inflation > 1.0
    assert adjusted.covariance_m2[0][0] == pytest.approx(b.covariance_m2[0][0] * diagnostic.covariance_inflation)
    assert adjusted.covariance_m2[1][1] == pytest.approx(b.covariance_m2[1][1] * diagnostic.covariance_inflation)
    assert adjusted.source.endswith(":handover_inflated")


def test_inflate_map_observation_covariance_rejects_bad_factor() -> None:
    with pytest.raises(ContractValidationError, match="positive"):
        inflate_map_observation_covariance(_obs("camera_A", 1.0, 0.0, 0.0), 0.0)


def test_handover_config_rejects_bad_weights() -> None:
    with pytest.raises(ContractValidationError, match="in \\[0, 1\\]"):
        HandoverUncertaintyConfig(disagreement_weight=1.5)

    with pytest.raises(ContractValidationError, match="positive"):
        HandoverUncertaintyConfig(disagreement_gate_m=0.0)
