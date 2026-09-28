"""Launch config strings must retain their YAML/ROS boolean meaning."""
import importlib.util
from pathlib import Path

import pytest

from experiments.core.visibility_launch_common import (
    DEFAULT_MANAGER_FUSION_RULE,
    _as_bool,
    _odom_yaw_offset_rad,
    manager_arm_settings,
)
from unav_common.config import local_controller_type


def base_config(**overrides):
    return {
        "pixel_timeout_s": .5,
        **overrides,
    }


@pytest.mark.parametrize("value", [False, "false", "False", "0", "off", "no"])
def test_manager_false_flags_stay_false(value):
    settings = manager_arm_settings(base_config(
        manager_require_gp_artifacts=value,
        manager_fusion_mode=value,
        manager_publish_map_observations=value,
        manager_require_source_batch_id=value,
        manager_commissioned_per_camera_sigma=value,
        manager_correction_timestamp_compensation=value,
        manager_require_consistency_when_source_available=value,
    ))
    for key, result in settings.items():
        if key in {
            "manager_require_gp_artifacts", "manager_fusion_mode",
            "manager_publish_map_observations", "manager_require_source_batch_id",
            "manager_commissioned_per_camera_sigma",
            "manager_correction_timestamp_compensation",
            "manager_require_consistency_when_source_available",
        }:
            assert result is False, key


@pytest.mark.parametrize("value", [True, "true", "True", "1", "on", "yes"])
def test_manager_true_flags_stay_true(value):
    settings = manager_arm_settings(base_config(manager_fusion_mode=value))
    assert settings["manager_fusion_mode"] is True


def test_default_fusion_rule_is_supported_and_matches_runtime():
    from reliability.measurement_fusion import (
        FUSION_RULE_INDEPENDENT,
        SUPPORTED_FUSION_RULES,
    )

    configured = manager_arm_settings(base_config())["manager_fusion_rule"]
    assert configured == DEFAULT_MANAGER_FUSION_RULE == FUSION_RULE_INDEPENDENT
    assert configured in SUPPORTED_FUSION_RULES


def test_manager_camera_roster_is_manifested_as_an_arm_setting():
    roster = "camera_A,camera_B,camera_C,camera_D,camera_E"
    settings = manager_arm_settings(base_config(manager_camera_ids=roster))
    assert settings["manager_camera_ids"] == roster


@pytest.mark.parametrize("value", ["fasle", "truthy", 2, None])
def test_invalid_boolean_values_are_rejected(value):
    with pytest.raises(ValueError, match="invalid boolean value"):
        _as_bool(value)


def test_local_controller_typo_is_rejected_instead_of_selecting_default():
    with pytest.raises(ValueError, match="unsupported local_controller_type"):
        local_controller_type("turn_then_g0")


def test_campaign_runner_does_not_enable_encoder_noise_from_false_string(tmp_path):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "campaign_config_semantics",
        root / "pipeline/campaign_runner.py",
    )
    campaign = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(campaign)
    detector = tmp_path / "detector.pt"
    detector.write_bytes(b"fixture")
    cfg = {
        "world": "warehouse_v2.world.sdf",
        "launch_file": "warehouse_primary_comparison.launch.py",
        "yolo_model": str(detector),
        "horizon": 10,
        "dt": .1,
        "goal_success_radius": .2,
        "run_timeout_after_first_cmd_s": 10,
        "odom_topic": "/odom_noisy",
        "use_encoder_noise": "false",
        "conditions": {"spatial_intact": {
            "camera_network_artifact_path": str(detector),
            "camera_network_active_camera_ids": "camera_A,camera_B,camera_C,camera_D,camera_E"}},
        "camera_network_objective": "metric_expected_belief",
    }
    command = campaign._build_launch_cmd(cfg, "task", "spatial_intact", 1, tmp_path)
    assert "odom_topic:=/odom" in command


def test_world_frame_encoder_yaw_is_not_offset_by_spawn_yaw():
    """The ground-truth-driven encoder already publishes map-frame yaw."""
    cfg = {
        "odom_topic": "/odom_noisy", "use_encoder_noise": True,
        "spawn": {"x": 1.0, "y": 2.0, "yaw": -1.5708},
    }
    assert _odom_yaw_offset_rad(cfg, "/odom_noisy") == 0.0
    assert _odom_yaw_offset_rad({**cfg, "use_encoder_noise": False}, "/odom") == pytest.approx(-1.5708)
