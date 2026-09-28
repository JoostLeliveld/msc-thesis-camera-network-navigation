"""Boolean fields may not use Python container/string truthiness."""

import pytest

from reliability.bev_reliability import BEVCameraToken
from reliability.camera_manager import CameraManagerConfig
from reliability.contracts import (
    CameraObservation,
    CameraQuality,
    ContractValidationError,
    EvaluationOnlySample,
    OperationalReliabilitySample,
    UpdateCovariance,
)
from reliability.firewall import validate_config_sources


def test_false_strings_remain_false_across_operational_contracts():
    observation = CameraObservation(
        camera_id="camera_A",
        detection_valid="false",
        mask_available="off",
    )
    assert observation.detection_valid is False
    assert observation.mask_available is False
    assert CameraQuality(stale="false").stale is False

    sample = OperationalReliabilitySample(
        projection_valid="false",
        measurement_stale="off",
        recent_detector_history=("true", "false"),
    )
    assert sample.projection_valid is False
    assert sample.measurement_stale is False
    assert sample.recent_detector_history == (True, False)

    token = BEVCameraToken(
        camera_id="camera_A",
        xy_m=(0.0, 0.0),
        in_fov="false",
        detection_valid="off",
        measurement_stale="no",
    )
    assert (token.in_fov, token.detection_valid, token.measurement_stale) == (
        False, False, False,
    )


def test_manager_false_string_does_not_enable_fallback_or_consistency():
    config = CameraManagerConfig(
        require_consistency_when_source_available="false",
        fallback_on_active_camera_loss="off",
    )
    assert config.require_consistency_when_source_available is False
    assert config.fallback_on_active_camera_loss is False


@pytest.mark.parametrize("bad", ["fasle", "", 2, None, []])
def test_ambiguous_boolean_contract_values_are_rejected(bad):
    with pytest.raises(ContractValidationError, match="boolean"):
        CameraObservation(camera_id="camera_A", detection_valid=bad)
    with pytest.raises(ContractValidationError, match="boolean"):
        CameraManagerConfig(fallback_on_active_camera_loss=bad)
    with pytest.raises(ContractValidationError, match="boolean"):
        BEVCameraToken(camera_id="camera_A", xy_m=(0.0, 0.0), in_fov=bad)


def test_camera_observation_requires_physical_identity():
    with pytest.raises(ContractValidationError, match="camera_id"):
        CameraObservation(camera_id=" ")


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: CameraObservation(camera_id="camera_A", timestamp_s=-1.0),
        lambda: OperationalReliabilitySample(timestamp_s=-1.0),
        lambda: OperationalReliabilitySample(measurement_age_s=-0.1),
        lambda: EvaluationOnlySample(timestamp_s=-1.0),
    ],
)
def test_contract_time_cannot_run_before_its_epoch(constructor):
    with pytest.raises(ContractValidationError, match="non-negative"):
        constructor()


def test_evaluation_and_update_contract_flags_are_strict():
    sample = EvaluationOnlySample(collision="false", geometry_breach="off")
    assert sample.collision is False
    assert sample.geometry_breach is False
    assert UpdateCovariance(available="false").available is False

    with pytest.raises(ContractValidationError, match="boolean"):
        EvaluationOnlySample(collision="fasle")
    with pytest.raises(ContractValidationError, match="boolean"):
        UpdateCovariance(available=2)


def test_firewall_rejects_misspelled_forbidden_flag_instead_of_guessing():
    config = {
        "normal_runtime_forbidden_config_values": {
            "state_or_reliability_source_tokens": [],
            "forbidden_true_flags": ["use_ground_truth"],
        },
        "allowed_evaluation_contexts": [],
    }
    with pytest.raises(ValueError, match="boolean"):
        validate_config_sources(
            {"use_ground_truth": "fasle"},
            cfg=config,
        )
