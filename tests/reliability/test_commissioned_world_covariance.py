"""The commissioned world-plane covariance profile reads its artifact, not a constant."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from reliability.nodes.camera_manager_node import (
    COMMISSIONED_COVARIANCE,
    COMMISSIONED_WORLD_COVARIANCE,
    SUPPORTED_COVARIANCE_PROFILES,
    commissioned_world_band,
    load_commissioned_world_covariance,
)

REPO = Path(__file__).resolve().parents[2]
ARTIFACT = (REPO / "logs/studies/perception_bayesian_gaussian/results"
            / "06_commissioning/commissioning.json")


def test_both_profiles_are_supported():
    """Adding the world profile must not remove the pixel one."""
    assert COMMISSIONED_COVARIANCE in SUPPORTED_COVARIANCE_PROFILES
    assert COMMISSIONED_WORLD_COVARIANCE in SUPPORTED_COVARIANCE_PROFILES


def test_profile_names_survive_launch_normalisation():
    """The launch layer lowercases this parameter, so a profile name with capitals is
    unreachable from a campaign config.

    This killed every run of the first live campaign: the manager refused
    'commissioned_world_r' because the constant said 'commissioned_world_R', and the
    failure appeared only as a node traceback inside a 40 MB launch log.
    """
    for name in SUPPORTED_COVARIANCE_PROFILES:
        assert name == name.strip().lower(), name


def test_band_assignment_is_monotone_in_confidence():
    edges = [0.2, 0.5, 0.9]
    assert commissioned_world_band(0.0, edges) == 0
    assert commissioned_world_band(0.3, edges) == 1
    assert commissioned_world_band(0.7, edges) == 2
    assert commissioned_world_band(0.99, edges) == 3
    previous = -1
    for step in range(101):
        band = commissioned_world_band(step / 100.0, edges)
        assert band >= previous
        previous = band


def test_missing_artifact_is_refused_not_guessed(tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps(
        {"confidence_edges": [], "models": {"radial": {"bias_parameters": {},
                                                       "commissioned": {}}}}))
    with pytest.raises(ValueError):
        load_commissioned_world_covariance(str(empty))


def test_confidence_field_the_override_reads_exists_on_the_contract():
    """The band lookup reads the detector confidence off a live CameraObservation.

    This is a real regression: the override was written against a `confidence` attribute
    that does not exist, so the manager crashed on its first detection and every run of
    the first live campaign died. A field-name error is invisible until a detection
    arrives, so it is pinned here against the actual contract.
    """
    from reliability.contracts import CameraObservation

    observation = CameraObservation(
        camera_id="camera_B", timestamp_s=1.0, pixel_uv=(640.0, 400.0),
        detection_valid=True, detector_score=0.93,
        bbox_xyxy=(600.0, 350.0, 680.0, 400.0), bbox_bottom_uv=(640.0, 400.0),
    )
    assert observation.detector_score == pytest.approx(0.93)
    # The band lookup must accept it without raising and return a usable index.
    band = commissioned_world_band(observation.detector_score, [0.5, 0.9])
    assert band == 2


@pytest.mark.skipif(not ARTIFACT.is_file(), reason="commissioning artifact not built")
def test_artifact_covariances_are_positive_definite_and_plausible():
    bias, table, edges, floor = load_commissioned_world_covariance(str(ARTIFACT))
    assert table and bias and edges
    for key, matrix in table.items():
        assert matrix[0][1] == pytest.approx(matrix[1][0]), key
        trace = matrix[0][0] + matrix[1][1]
        determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        assert trace > 0.0 and determinant > 0.0, key
        sd = math.sqrt(trace / 2.0)
        # Commissioned widths run 3.8-26.7 cm; anything outside a decimetre-scale band
        # means the artifact is not what this profile expects.
        assert 0.005 < sd < 1.0, (key, sd)
    assert floor > 0.0


@pytest.mark.skipif(not ARTIFACT.is_file(), reason="commissioning artifact not built")
def test_stated_width_grows_as_detector_confidence_falls():
    """The whole point of banding: an unsure detection must state more uncertainty."""
    _bias, table, _edges, _floor = load_commissioned_world_covariance(str(ARTIFACT))
    for camera in sorted({key[0] for key in table}):
        bands = sorted(key[1] for key in table if key[0] == camera)
        if len(bands) < 4:
            continue
        widths = [math.sqrt((table[(camera, b)][0][0]
                             + table[(camera, b)][1][1]) / 2.0) for b in bands]
        low = sum(widths[:2]) / 2.0
        high = sum(widths[-2:]) / 2.0
        assert low > high, (camera, low, high)
