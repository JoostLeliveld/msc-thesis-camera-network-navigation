"""Shared launch helpers for the visibility-aware thesis pipeline."""

from __future__ import annotations

import json
import math
import hashlib
import io
import os
from pathlib import Path
from typing import Dict, List

import numpy as np

from launch.actions import IncludeLaunchDescription, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from unav_common.config import local_controller_type, parse_bev_affine_calibration, parse_bool


PAPER_LAUNCH_DEFAULTS: Dict[str, str] = {
    'planner': 'visibility_aware_efe',
    'world': 'warehouse_v2.world.sdf',
    'task': '',
    'seed': '0',
    'odom_wait_timeout_s': '60.0',
    'odom_wait_min_messages': '1',
    'odom_wait_require_pose_match': 'false',
    'use_pixel_correction': 'true',
    'pixel_topic': '/perception/pixel_pose',
    'pixel_timeout_s': '0.5',
    'pixel_correction_min_interval_s': '0.1',
    'pixel_correction_approx': 'AUTO',
    'skip_stale_pixel_correction': 'true',
    'bev_y_calibration_offset_m': '0.127',
    'bev_affine_calibration': '',
    'pixel_max_correction_jump_m': '0.0',
    'pixel_correction_nis_threshold': '9.21',
    # How the belief recovers when a correction is refused or cannot be replayed.
    # Declared here so every run's provenance states its recovery policy instead of
    # inheriting whatever the runtime default happened to be that week.
    'state_reanchor_m': '0.0',
    'state_max_predict_dt_s': '1.5',
    'state_reject_inflate_m2': '0.0',
    'stale_belief_inflate_m2_per_s': '0.0',
    'stale_belief_inflate_cap_m2': '0.0',
    'require_state_correction_envelope': 'false',
    'use_diagnostic_odom_localization': 'false',
    'cmd_publish_rate': '10.0',
    'command_noise_output_topic': '/cmd_vel',
    'min_state_cov': '1e-6',
    'plan_rate': '2.0',
    'belief_publish_rate': '10.0',
    'horizon': '40',
    'dt': '0.25',
    'v_max': '0.22',
    # Kinematic plausibility cap on the EKF prediction step. 0 = off, which is
    # the single-camera default: the locked honest_campaign_v1 ran without it
    # and its evidence must stay reproducible. The 4-cam configs set it to v_max.
    'max_predict_speed_mps': '0.0',
    # 'fused' = one pre-fused /state/bev pose per tick; 'per_camera' = fold the
    # per-camera map observations in sequentially, each with its own covariance
    # and its own gate decision. Same gate chain either way, so a clean A/B.
    'state_correction_mode': 'fused',
    # Jump limiter for the METRIC correction path. 0 = off (default): on a
    # linear-H metric measurement NIS is the correct gate, and a jump limit
    # deadlocks recovery because rejecting inflates S, which raises the gain,
    # which raises the update. Distinct from pixel_max_correction_jump_m.
    'state_max_correction_jump_m': '0.0',
    # Unmodelled-error term added in quadrature to every metric correction:
    # R' = R + sigma^2 I. Mostly INTER-CAMERA DISAGREEMENT, which the
    # per-camera pixel covariance cannot express. 0 = off (single-camera path).
    'control_weight': '0.0',
    # Risk and ambiguity are BOTH in nats, so their relative weight is 1 unless
    # there is a stated reason otherwise. The previous 3.0 had no derivation --
    # it entered under a commit called "Launch updates" -- and route choice is a
    # direct function of it, so it was an undeclared tuning knob deciding the
    # result. Reset 2026-09-18; see docs/PLANNER.md.
    'risk_weight_obs': '1.0',
    'ambiguity_weight': '1.0',
    'goal_sigma_uv': '2.0',
    'use_ambiguity': 'true',
    'use_obs_risk': 'true',
    'r_visible_uv': '2.5',
    'r_miss_uv': '120.0',
    'visibility_sigma_kappa': '1.0',
    'camera_network_objective': 'legacy_pixel_chart',
    # EFE preference precision. Set to the declared arrival tolerance
    # (goal_success_radius). At 0.15 the risk term is ~5x stronger and the
    # objective collapses toward shortest path. See docs/PLANNER.md.
    'network_goal_std_m': '0.10',
    # Goal-prior anneal start; see base_planner. -1 disables.
    'network_goal_std_start_m': '5.0',
    'kouw_et1_ambiguity': 'true',
    'network_goal_std_start_m': '-1.0',
    'camera_network_updates_per_step': '1',
    'optimizer_control_block_steps': '1',
    'goal_prior_u_std_start': '80.0',
    'goal_prior_v_std_start': '80.0',
    'goal_prior_u_std_final': '4.0',
    'goal_prior_v_std_final': '4.0',
    'goal_tightening_power': '0.45',
    'goal_progress_n_steps': '90',
    'observation_risk_scale': '1.25',
    'ambiguity_term_scale': '1.00',
    'discount_gamma': '0.995',
    'robot_collision_radius_m': '0.48541219597369',
    'robot_length_m': '0.8',
    'robot_width_m': '0.55',
    'bridge_camera_a': 'true',
    'bridge_camera_b': 'false',
    'bridge_camera_c': 'false',
    'bridge_camera_d': 'false',
    'lockstep': 'false',
    'lockstep_control_step_iterations': '100',
    'lockstep_camera_every_control_steps': '2',
    'lockstep_max_control_steps': '0',
    'use_command_noise': 'true',
    'use_encoder_noise': 'true',
    'use_odom_for_predict': 'true',
    'odom_topic': '/odom_noisy',
    # Actuation-noise power spectral densities (sigma_v, sigma_omega) used by the
    # unicycle Q_d closed form. These are DECLARED MODEL PARAMETERS, not the
    # encoder-noise specification and not fitted to data. Locked to the values of
    # the camera-ready IWAI paper so the thesis and the prior work grow covariance
    # identically. Do not change without recording the reason in
    # docs/PROCESS_NOISE.md.
    'process_noise_xy': '0.02',
    'process_noise_theta': '0.08',
    'process_noise_model': 'encoder',
    'optimizer_maxiter': '80',
    'optimizer_maxfun': '500',
    'optimizer_ftol': '1e-6',
    'optimizer_gtol': '1e-4',
    'optimizer_warm_start': 'true',
    'optimizer_multistart': 'false',
    'optimizer_multistart_include_direct': 'true',
    'optimizer_initial_routes_json': '',
    'optimizer_terminal_goal_tolerance_m': '0.0',
    'optimizer_route_seed_mode': 'explicit',
    'use_hierarchical': 'false',
    'global_planner_mode': 'efe',
    'preselected_route_json': '',
    'preselected_route_sha256': '',
    'preselected_route_source_path': '',
    'preselected_route_source_sha256': '',
    'preselected_route_clearance_m': '0.25',
    'preselected_route_endpoint_tolerance_m': '0.25',
    'preselected_route_sample_step_m': '0.04',
    'global_horizon': '60',
    'global_dt': '0.0',
    'local_horizon': '12',
    'local_plan_rate': '4.0',
    'local_optimizer_maxiter': '60',
    'global_use_ambiguity': 'true',
    'local_use_ambiguity': 'false',
    'local_use_obs_risk': 'true',
    'global_optimizer_multistart': 'true',
    'local_optimizer_multistart': 'true',
    'local_use_visibility_model': 'false',
    'local_use_belief_nogo_cost': 'false',
    'local_nogo_penalty_type': '',
    'local_nogo_safe_distance': '-1.0',
    'local_goal_prior_u_std_start': '-1.0',
    'local_goal_prior_v_std_start': '-1.0',
    'local_goal_prior_u_std_final': '-1.0',
    'local_goal_prior_v_std_final': '-1.0',
    'waypoint_spacing_m': '0.2',
    'waypoint_arrival_radius_m': '0.1',
    'local_replan_min_remaining_s': '0.0',
    'local_replan_on_waypoint_change': 'false',
    'latency_compensate_plan_handoff': 'false',
    'simple_tracker_yaw_gate_rad': '0.65',
    'ff_fb_turn_rate_limit_rad_s': '0.80',
    'ff_fb_corner_crawl_speed_mps': '0.18',
    'ff_fb_pivot_heading_error_rad': '2.60',
    'odom_heading_timeout_s': '0.75',
    'heading_update_mode': 'coupled',
    # Canonical thesis follower. Campaigns must still set this explicitly.
    'local_controller_type': 'ff_fb',
    'debug_runtime': 'false',
    'auto_stop_on_goal': 'true',
    'goal_success_radius': '0.20',
    'goal_success_hold_s': '2.0',
    'goal_stable_radius': '0.20',
    'goal_stable_hold_s': '2.0',
    'goal_stable_max_displacement_m': '0.04',
    'run_timeout_after_first_cmd_s': '75.0',
    'first_cmd_linear_eps': '0.02',
    'first_cmd_angular_eps': '0.10',
    'stuck_window_s': '8.0',
    'stuck_max_displacement_m': '0.08',
    'stuck_max_goal_improvement_m': '0.05',
    'stuck_cmd_fraction_min': '0.50',
    'stuck_idle_cmd_fraction_max': '0.10',
    'reset_world': 'false',
    # Commissioning drive: when false, the goal-mission + goal-marker nodes are
    # not launched, so no goal is published and the EFE planner stays silent
    # (its belief EKF still runs and publishes /planner_belief). Default true
    # preserves the frozen navigation comparison exactly.
    'enable_mission': 'true',
    'wait_for_belief_before_first_goal': 'false',
    'initial_belief_max_sigma_m': '0.0',
    'operational_belief_timeout_s': '0.5',
    'yolo_model': '',
    'outcome_journal_path': '',
    'manager_outcome_journal_path': '',
    'campaign_config_path': '',
    'yolo_device': '',
    'yolo_imgsz': '640',
    'yolo_conf_threshold': '0.25',
    'yolo_predict_conf_floor': '0.05',
    'yolo_iou_threshold': '0.45',
    'yolo_target_class': 'robot',
    'yolo_class_id': '-1',
    'yolo_use_masks': 'true',
    'yolo_min_mask_area_px': '12.0',
    'yolo_mask_bottom_band_px': '3.0',
    'yolo_min_bbox_area_px': '0.0',
    # Five-Hz camera rounds are 0.20 s apart. This must stay strictly below
    # one period so adjacent physical capture rounds can never merge.
    'yolo_max_batch_stamp_skew_s': '0.05',
    'yolo_debug_frame_dir': '',
    'yolo_debug_crop_dir': '',
    'log_dir': 'logs/experiments',
}

# Must match reliability.measurement_fusion without importing one ROS package
# into another package's launch-description import path. Historical campaigns
# name their rule explicitly; this default keeps ad-hoc launch and manifest
# generation aligned with the camera-manager node.
DEFAULT_MANAGER_FUSION_RULE = 'independent'

# Runtime collision closure for verified named include instances. This is kept
# outside the frozen camera/world profile contract so adding physical obstacles
# cannot silently refreeze calibration identity.
_COLLISION_INCLUDE_NAMES_BY_WORLD = {
    world: ('forklift_parked', 'pallet_jack', 'bin_office',
            'pallet_loose_1', 'pallet_loose_2')
    for world in (
        'warehouse_v2.world.sdf',
    )
}

_DETECTOR_CALIBRATION_WORLD_BY_WORLD = {}

# Command noise shape — paper-locked, not user-overridable.
# These values were calibrated for the TurtleBot3 in Gazebo and must not change between runs.
_COMMAND_NOISE_LINEAR_SLIP_MEAN: float = 0.03
_COMMAND_NOISE_LINEAR_SLIP_STD: float = 0.06
_COMMAND_NOISE_ANGULAR_SLIP_MEAN: float = 0.00
_COMMAND_NOISE_ANGULAR_SLIP_STD: float = 0.04
_COMMAND_NOISE_LINEAR_ADDITIVE_STD: float = 0.008
_COMMAND_NOISE_ANGULAR_ADDITIVE_STD: float = 0.035
_COMMAND_NOISE_CORRELATION_ALPHA: float = 0.85

# Encoder noise shape — paper-locked, not user-overridable.
# Models cheap wheel encoder imprecision on top of actuation slip.
# Independent AR(1) process (different RNG seed offset) so that belief and
# truth diverge at a realistic rate when camera observations are unavailable.
# Wheel-encoder odometry noise, set to a LOW-TRACTION warehouse condition.
# The previous values produced 1.3-2.2% position drift per metre travelled,
# which sits inside the 0.50-2.30% band published for CALIBRATED wheel odometry
# on good surfaces. A warehouse AMR on smooth concrete with payload shifts and
# dust belongs above that band, so these are scaled to target ~4% of distance.
# See docs/PROCESS_NOISE.md for the paired process-noise decision.
_ENCODER_NOISE_LINEAR_SLIP_MEAN: float = 0.05
_ENCODER_NOISE_LINEAR_SLIP_STD: float = 0.125
_ENCODER_NOISE_ANGULAR_SLIP_MEAN: float = 0.00
_ENCODER_NOISE_ANGULAR_SLIP_STD: float = 0.075
_ENCODER_NOISE_LINEAR_ADDITIVE_STD: float = 0.004
_ENCODER_NOISE_ANGULAR_ADDITIVE_STD: float = 0.050
_ENCODER_NOISE_CORRELATION_ALPHA: float = 0.80
# Systematic wheel errors of an off-the-shelf differential-drive robot, as isolated by
# UMBmark on a TRC LabMate (Borenstein & Feng): right/left diameter ratio 1.00121 and
# effective wheelbase 337.2 mm against 340 mm nominal. One robot, so fixed, not per seed.
_ENCODER_WHEEL_DIAMETER_RATIO_ERROR: float = 0.00121
_ENCODER_WHEELBASE_RATIO: float = 337.2 / 340.0
_ROBOT_WHEEL_SEPARATION_M: float = 0.44   # warehouse_amr.urdf.xacro wheel_sep

# Sensor pixel noise — paper-locked, not user-overridable.
_SENSOR_PIXEL_NOISE_SIGMA: float = 1.0


VISIBILITY_FALLBACK_DEFAULTS: Dict[str, object] = {
    'visibility_target_height_m': 0.0,
    'use_nogo_cost': 'true',
    'nogo_penalty_type': 'warning_band',
    # Robot half-width 0.275 + 0.05 lane-keeping margin. The former 0.55-0.585
    # (circumscribed radius) left the 1.10 m lanes a NEGATIVE lateral budget,
    # i.e. structurally infeasible before any uncertainty existed.
    'nogo_safe_distance': 0.0,
    # Equals nogo_warning_band: the already-declared 'close to the edge' scale.
    # At 1e-3 a 1 mm notional violation cost more than the entire risk term.
    'nogo_logbarrier_eps': 0.05,
    'nogo_warning_band': 0.05,
    'nogo_near_weight': 50.0,
    # The covariance -> clearance channel. With this off the obstacle term sees
    # only the mean path, so predicted belief growth costs nothing.
    'use_belief_nogo_cost': 'true',
    'nogo_belief_kappa': 1.0,
    'nogo_mode': 'keep_out',
    # Hit/miss expected-belief mixture in the EFE objective. 'false' reproduces
    # the published precision-blend planner bit-for-bit; only an experiment that
    # deliberately evaluates the new measurement model may set it true.
    'use_hit_miss_mixture': 'false',
}


def _as_bool(value: object) -> bool:
    try:
        return parse_bool(value)
    except ValueError as exc:
        raise ValueError(f"invalid boolean value {value!r}") from exc


def _launch_value(context, name: str, default_value: str) -> str:
    return LaunchConfiguration(name, default=default_value).perform(context)

def _profile_name_tuple(profile: Dict[str, object], plural_key: str, singular_key: str) -> tuple[str, ...]:
    raw = profile.get(plural_key, profile.get(singular_key, ()))
    if isinstance(raw, str):
        value = raw.strip()
        return (value,) if value else ()
    if isinstance(raw, (list, tuple)):
        return tuple(str(item).strip() for item in raw if str(item).strip())
    return ()


def _require_task_field(task, key):
    if key not in task:
        raise RuntimeError(f"Task is missing '{key}' field")
    return task[key]


def _camera_network_identity(
        path: str, expected_camera_ids, active_camera_ids=None) -> Dict[str, object]:
    """Read the exact network bytes and materialize the consumer expectations."""
    artifact_bytes = Path(path).read_bytes()
    digest = hashlib.sha256(artifact_bytes).hexdigest()
    try:
        with np.load(io.BytesIO(artifact_bytes), allow_pickle=False) as archive:
            metadata = json.loads(str(archive['metadata_json'].item()))
            artifact_camera_ids = tuple(str(value) for value in archive['camera_ids'])
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise RuntimeError(f'malformed camera-network artifact identity: {exc}') from exc
    source_hashes = metadata.get('source_hashes') if isinstance(metadata, dict) else None
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise RuntimeError('camera-network artifact has no nonempty source_hashes mapping')
    roster = tuple(str(value) for value in expected_camera_ids)
    if not roster or len(set(roster)) != len(roster) or set(roster) != set(artifact_camera_ids):
        raise RuntimeError(
            f'camera-network roster {artifact_camera_ids} differs from world profile {roster}'
        )
    active = tuple(str(value) for value in (active_camera_ids or roster))
    if not active or len(set(active)) != len(active) or not set(active).issubset(roster):
        raise RuntimeError(
            'active planning cameras must be a nonempty subset of the artifact roster')
    return {
        'camera_network_expected_sha256': digest,
        'camera_network_expected_source_hashes_json': json.dumps(
            source_hashes, sort_keys=True, separators=(',', ':')
        ),
        'camera_network_camera_ids': ','.join(roster),
        'camera_network_active_camera_ids': ','.join(active),
    }


def _state_estimator_metadata(cfg: Dict[str, object] | None = None) -> Dict[str, str]:
    return {
        'state_source_x': 'yolo_mask_or_bbox_homography',
        'state_source_y': 'yolo_mask_or_bbox_homography',
        'state_source_theta': 'none',
        'state_estimator_mode': 'yolo_camera_xy_only_no_direct_theta',
    }


def parse_common_launch_config(context) -> Dict[str, object]:
    """Parse the generic visibility-aware agent launch arguments."""
    seed_value = int(LaunchConfiguration('seed').perform(context))
    world_value = LaunchConfiguration('world').perform(context)
    visibility_enabled_default = 'true'
    use_visibility_raw = _launch_value(context, 'use_visibility_model', visibility_enabled_default).strip().lower()

    cfg: Dict[str, object] = {
        'use_sim_time': True,
        'planner': _launch_value(context, 'planner', PAPER_LAUNCH_DEFAULTS['planner']).strip(),
        'world': world_value,
        'world_profiles_path': LaunchConfiguration('world_profiles').perform(context),
        'tasks_yaml': LaunchConfiguration('tasks_yaml').perform(context),
        'task_name': _launch_value(context, 'task', PAPER_LAUNCH_DEFAULTS['task']).strip(),
        'comparison_method_id': _launch_value(context, 'comparison_method_id', '').strip(),
        'log_dir': _launch_value(context, 'log_dir', PAPER_LAUNCH_DEFAULTS['log_dir']).strip(),
        'campaign_config_path': _launch_value(
            context, 'campaign_config_path', PAPER_LAUNCH_DEFAULTS['campaign_config_path']
        ).strip(),
        'seed': seed_value,
        'perception_backend': 'yolo',
        'sensor_pixel_noise_sigma': _SENSOR_PIXEL_NOISE_SIGMA,
        'odom_wait_timeout_s': float(
            _launch_value(context, 'odom_wait_timeout_s', PAPER_LAUNCH_DEFAULTS['odom_wait_timeout_s'])
        ),
        'odom_wait_min_messages': max(1, int(float(_launch_value(context, 'odom_wait_min_messages', '1')))),
        'odom_wait_require_pose_match': _as_bool(_launch_value(context, 'odom_wait_require_pose_match', 'false')),
        'odom_wait_position_tolerance': 0.25,
        'odom_wait_yaw_tolerance': 0.5,
        'use_pixel_correction': _as_bool(_launch_value(context, 'use_pixel_correction', PAPER_LAUNCH_DEFAULTS['use_pixel_correction'])),
        # Multi-camera belief front-end (guarded; default off preserves single-cam path).
        'multicam_belief': _as_bool(_launch_value(context, 'multicam_belief', 'false')),
        'lockstep': _as_bool(_launch_value(context, 'lockstep', 'false')),
        'lockstep_control_step_iterations': int(_launch_value(
            context, 'lockstep_control_step_iterations', '100')),
        'lockstep_camera_every_control_steps': int(_launch_value(
            context, 'lockstep_camera_every_control_steps', '2')),
        'lockstep_max_control_steps': int(_launch_value(
            context, 'lockstep_max_control_steps', '0')),
        'multicam_scheduled': _as_bool(_launch_value(context, 'multicam_scheduled', 'false')),
        'scheduled_coverage_artifact': _launch_value(context, 'scheduled_coverage_artifact', '').strip(),
        'scheduled_report_std_m': float(_launch_value(context, 'scheduled_report_std_m', '0.15')),
        'scheduled_rate_hz': float(_launch_value(context, 'scheduled_rate_hz', '5.0')),
        'scheduled_selection_mode': _launch_value(
            context, 'scheduled_selection_mode', 'coverage_best_with_fallback'
        ).strip().lower(),
        'manager_gp_artifact_template': _launch_value(context, 'manager_gp_artifact_template', '').strip(),
        'manager_min_spatial_trust': float(_launch_value(context, 'manager_min_spatial_trust', '0.15')),
        'manager_decision_rate_hz': float(_launch_value(context, 'manager_decision_rate_hz', '5.0')),
        'manager_camera_ids': _launch_value(context, 'manager_camera_ids', '').strip(),
        'manager_fusion_mode': _as_bool(_launch_value(context, 'manager_fusion_mode', 'true')),
        'manager_fusion_disagreement_gate_m': float(
            _launch_value(context, 'manager_fusion_disagreement_gate_m', '0.6')
        ),
        'manager_require_source_batch_id': _as_bool(_launch_value(
            context, 'manager_require_source_batch_id', 'true')),
        'manager_bootstrap_min_cameras': int(_launch_value(
            context, 'manager_bootstrap_min_cameras', '2')),
        'manager_bootstrap_max_disagreement_m': float(_launch_value(
            context, 'manager_bootstrap_max_disagreement_m', '0.30')),
        'manager_use_task_start_as_bootstrap_prior': _as_bool(_launch_value(
            context, 'manager_use_task_start_as_bootstrap_prior', 'false')),
        'manager_bootstrap_prior_counts_as_support': _as_bool(_launch_value(
            context, 'manager_bootstrap_prior_counts_as_support', 'false')),
        'initial_belief_from_task_start': _as_bool(_launch_value(
            context, 'initial_belief_from_task_start', 'false')),
        'initial_belief_sigma_xy_m': float(_launch_value(context, 'initial_belief_sigma_xy_m', '0.10')),
        'initial_belief_sigma_theta_rad': float(_launch_value(
            context, 'initial_belief_sigma_theta_rad', str(math.radians(15.0)))),
        'manager_require_gp_artifacts': _as_bool(_launch_value(context, 'manager_require_gp_artifacts', 'true')),
        'manager_fusion_max_timestamp_spread_s': float(
            _launch_value(context, 'manager_fusion_max_timestamp_spread_s', '0.05')
        ),
        'manager_covariance_profile': _launch_value(
            context, 'manager_covariance_profile', 'commissioned_sigma_px'
        ).strip().lower(),
        'manager_commissioned_calibration_path': _launch_value(
            context, 'manager_commissioned_calibration_path', ''
        ),
        'manager_commissioned_world_covariance_path': _launch_value(
            context, 'manager_commissioned_world_covariance_path', ''
        ).strip(),
        'manager_commissioned_sigma_px': float(
            _launch_value(context, 'manager_commissioned_sigma_px', '0.0')
        ),
        'manager_commissioned_per_camera_sigma': _as_bool(_launch_value(
            context, 'manager_commissioned_per_camera_sigma', 'false')),
        'manager_fusion_common_mode_std_m': float(
            _launch_value(context, 'manager_fusion_common_mode_std_m', '0.0')),
        'manager_fusion_rule': _launch_value(
            context, 'manager_fusion_rule', DEFAULT_MANAGER_FUSION_RULE
        ).strip().lower(),
        'manager_correction_timestamp_compensation': _as_bool(_launch_value(
            context, 'manager_correction_timestamp_compensation', 'false')),
        'manager_sensor_gate_config_path': _launch_value(
            context, 'manager_sensor_gate_config_path', '').strip(),
        'manager_correction_residual_interval_s': float(_launch_value(
            context, 'manager_correction_residual_interval_s', '0.05')),
        'manager_correction_propagation_drift_std': float(_launch_value(
            context, 'manager_correction_propagation_drift_std', '0.05')),
        'manager_observation_model': _launch_value(
            context, 'manager_observation_model', 'raw_box'
        ).strip().lower(),
        'manager_availability_model_path': _launch_value(
            context, 'manager_availability_model_path', ''
        ).strip(),
        'manager_availability_model_expected_sha256': _launch_value(
            context, 'manager_availability_model_expected_sha256', ''
        ).strip(),
        'manager_learned_correction_path': _launch_value(
            context, 'manager_learned_correction_path', ''
        ).strip(),
        'manager_visibility_sensor_model_path': _launch_value(
            context, 'manager_visibility_sensor_model_path', ''
        ).strip(),
        'manager_visibility_sensor_model_expected_sha256': _launch_value(
            context, 'manager_visibility_sensor_model_expected_sha256', ''
        ).strip(),
        'manager_perception_sensor_model_path': _launch_value(
            context, 'manager_perception_sensor_model_path', ''
        ).strip(),
        'manager_perception_sensor_model_expected_sha256': _launch_value(
            context, 'manager_perception_sensor_model_expected_sha256', ''
        ).strip(),
        'manager_max_measurement_age_s': float(
            _launch_value(context, 'manager_max_measurement_age_s', '1.25')
        ),
        'manager_age_decay_s': float(_launch_value(context, 'manager_age_decay_s', '1.25')),
        'manager_min_association_confidence': float(
            _launch_value(context, 'manager_min_association_confidence', '0.30')
        ),
        'manager_required_consecutive_better_frames': int(
            _launch_value(context, 'manager_required_consecutive_better_frames', '1')
        ),
        'manager_max_cross_camera_disagreement_m': float(
            _launch_value(context, 'manager_max_cross_camera_disagreement_m', '1.0')
        ),
        'manager_require_consistency_when_source_available': _as_bool(
            _launch_value(context, 'manager_require_consistency_when_source_available', 'false')
        ),
        # Raw string so the multicam branch can default it ON while still letting
        # a config explicitly opt out (empty = unset -> multicam default True).
        'state_correction_ekf_raw': _launch_value(context, 'state_correction_ekf', '').strip(),
        'pixel_topic': _launch_value(context, 'pixel_topic', PAPER_LAUNCH_DEFAULTS['pixel_topic']).strip(),
        'pixel_timeout_s': float(_launch_value(context, 'pixel_timeout_s', PAPER_LAUNCH_DEFAULTS['pixel_timeout_s'])),
        'pixel_correction_min_interval_s': float(_launch_value(context, 'pixel_correction_min_interval_s', '0.0')),
        'bev_y_calibration_offset_m': float(_launch_value(context, 'bev_y_calibration_offset_m', PAPER_LAUNCH_DEFAULTS['bev_y_calibration_offset_m'])),
        'bev_affine_calibration': _launch_value(context, 'bev_affine_calibration', PAPER_LAUNCH_DEFAULTS['bev_affine_calibration']),
        'pixel_max_correction_jump_m': float(_launch_value(context, 'pixel_max_correction_jump_m', PAPER_LAUNCH_DEFAULTS['pixel_max_correction_jump_m'])),
        'pixel_correction_nis_threshold': float(_launch_value(context, 'pixel_correction_nis_threshold', PAPER_LAUNCH_DEFAULTS['pixel_correction_nis_threshold'])),
        'state_reanchor_m': float(_launch_value(
            context, 'state_reanchor_m', PAPER_LAUNCH_DEFAULTS['state_reanchor_m'])),
        'state_max_predict_dt_s': float(_launch_value(
            context, 'state_max_predict_dt_s',
            PAPER_LAUNCH_DEFAULTS['state_max_predict_dt_s'])),
        'state_reject_inflate_m2': float(_launch_value(
            context, 'state_reject_inflate_m2',
            PAPER_LAUNCH_DEFAULTS['state_reject_inflate_m2'])),
        'stale_belief_inflate_m2_per_s': float(_launch_value(
            context, 'stale_belief_inflate_m2_per_s',
            PAPER_LAUNCH_DEFAULTS['stale_belief_inflate_m2_per_s'])),
        'stale_belief_inflate_cap_m2': float(_launch_value(
            context, 'stale_belief_inflate_cap_m2',
            PAPER_LAUNCH_DEFAULTS['stale_belief_inflate_cap_m2'])),
        'require_state_correction_envelope': _as_bool(_launch_value(
            context, 'require_state_correction_envelope',
            PAPER_LAUNCH_DEFAULTS['require_state_correction_envelope'])),
        'use_diagnostic_odom_localization': _as_bool(_launch_value(
            context, 'use_diagnostic_odom_localization',
            PAPER_LAUNCH_DEFAULTS['use_diagnostic_odom_localization'])),
        'pixel_correction_approx': _launch_value(
            context,
            'pixel_correction_approx',
            PAPER_LAUNCH_DEFAULTS['pixel_correction_approx'],
        ).strip().upper(),
        'skip_stale_pixel_correction': _as_bool(_launch_value(context, 'skip_stale_pixel_correction', 'true')),
        'use_ambiguity': _as_bool(_launch_value(context, 'use_ambiguity', PAPER_LAUNCH_DEFAULTS['use_ambiguity'])),
        'use_obs_risk': _as_bool(_launch_value(context, 'use_obs_risk', PAPER_LAUNCH_DEFAULTS['use_obs_risk'])),
        'auto_stop_on_goal': _as_bool(_launch_value(context, 'auto_stop_on_goal', PAPER_LAUNCH_DEFAULTS['auto_stop_on_goal'])),
        'enable_mission': _as_bool(_launch_value(context, 'enable_mission', PAPER_LAUNCH_DEFAULTS['enable_mission'])),
        'wait_for_belief_before_first_goal': _as_bool(_launch_value(
            context,
            'wait_for_belief_before_first_goal',
            PAPER_LAUNCH_DEFAULTS['wait_for_belief_before_first_goal'],
        )),
        'initial_belief_max_sigma_m': float(_launch_value(
            context,
            'initial_belief_max_sigma_m',
            PAPER_LAUNCH_DEFAULTS['initial_belief_max_sigma_m'],
        )),
        'operational_belief_timeout_s': float(_launch_value(
            context, 'operational_belief_timeout_s',
            PAPER_LAUNCH_DEFAULTS['operational_belief_timeout_s'],
        )),
        'goal_success_radius': float(_launch_value(context, 'goal_success_radius', PAPER_LAUNCH_DEFAULTS['goal_success_radius'])),
        'goal_success_hold_s': float(_launch_value(context, 'goal_success_hold_s', PAPER_LAUNCH_DEFAULTS['goal_success_hold_s'])),
        'goal_stable_radius': float(_launch_value(context, 'goal_stable_radius', PAPER_LAUNCH_DEFAULTS['goal_stable_radius'])),
        'goal_stable_hold_s': float(_launch_value(context, 'goal_stable_hold_s', PAPER_LAUNCH_DEFAULTS['goal_stable_hold_s'])),
        'goal_stable_max_displacement_m': float(_launch_value(
            context,
            'goal_stable_max_displacement_m',
            PAPER_LAUNCH_DEFAULTS['goal_stable_max_displacement_m'],
        )),
        'run_timeout_after_first_cmd_s': float(_launch_value(context, 'run_timeout_after_first_cmd_s', PAPER_LAUNCH_DEFAULTS['run_timeout_after_first_cmd_s'])),
        'first_cmd_linear_eps': float(_launch_value(context, 'first_cmd_linear_eps', PAPER_LAUNCH_DEFAULTS['first_cmd_linear_eps'])),
        'first_cmd_angular_eps': float(_launch_value(context, 'first_cmd_angular_eps', PAPER_LAUNCH_DEFAULTS['first_cmd_angular_eps'])),
        'stuck_window_s': float(_launch_value(context, 'stuck_window_s', PAPER_LAUNCH_DEFAULTS['stuck_window_s'])),
        'stuck_max_displacement_m': float(_launch_value(context, 'stuck_max_displacement_m', PAPER_LAUNCH_DEFAULTS['stuck_max_displacement_m'])),
        'stuck_max_goal_improvement_m': float(_launch_value(
            context,
            'stuck_max_goal_improvement_m',
            PAPER_LAUNCH_DEFAULTS['stuck_max_goal_improvement_m'],
        )),
        'stuck_cmd_fraction_min': float(_launch_value(context, 'stuck_cmd_fraction_min', PAPER_LAUNCH_DEFAULTS['stuck_cmd_fraction_min'])),
        'stuck_idle_cmd_fraction_max': float(_launch_value(context, 'stuck_idle_cmd_fraction_max', PAPER_LAUNCH_DEFAULTS['stuck_idle_cmd_fraction_max'])),
        'process_noise_xy': float(_launch_value(context, 'process_noise_xy', PAPER_LAUNCH_DEFAULTS['process_noise_xy'])),
        'process_noise_theta': float(_launch_value(context, 'process_noise_theta', PAPER_LAUNCH_DEFAULTS['process_noise_theta'])),
        'process_noise_model': str(_launch_value(context, 'process_noise_model', PAPER_LAUNCH_DEFAULTS['process_noise_model'])),
        'optimizer_maxiter': int(_launch_value(context, 'optimizer_maxiter', PAPER_LAUNCH_DEFAULTS['optimizer_maxiter'])),
        'optimizer_maxfun': int(_launch_value(context, 'optimizer_maxfun', PAPER_LAUNCH_DEFAULTS['optimizer_maxfun'])),
        'optimizer_ftol': float(_launch_value(context, 'optimizer_ftol', PAPER_LAUNCH_DEFAULTS['optimizer_ftol'])),
        'optimizer_gtol': float(_launch_value(context, 'optimizer_gtol', PAPER_LAUNCH_DEFAULTS['optimizer_gtol'])),
        'optimizer_warm_start': _as_bool(_launch_value(context, 'optimizer_warm_start', PAPER_LAUNCH_DEFAULTS['optimizer_warm_start'])),
        'optimizer_multistart': _as_bool(_launch_value(context, 'optimizer_multistart', PAPER_LAUNCH_DEFAULTS['optimizer_multistart'])),
        'optimizer_multistart_include_direct': _as_bool(_launch_value(context, 'optimizer_multistart_include_direct', PAPER_LAUNCH_DEFAULTS['optimizer_multistart_include_direct'])),
        'optimizer_initial_routes_json': _launch_value(context, 'optimizer_initial_routes_json', PAPER_LAUNCH_DEFAULTS['optimizer_initial_routes_json']),
        'optimizer_terminal_goal_tolerance_m': float(_launch_value(
            context,
            'optimizer_terminal_goal_tolerance_m',
            PAPER_LAUNCH_DEFAULTS['optimizer_terminal_goal_tolerance_m'],
        )),
        'optimizer_route_seed_mode': _launch_value(context, 'optimizer_route_seed_mode', PAPER_LAUNCH_DEFAULTS['optimizer_route_seed_mode']),
        'use_hierarchical': _as_bool(_launch_value(context, 'use_hierarchical', PAPER_LAUNCH_DEFAULTS['use_hierarchical'])),
        'global_planner_mode': _launch_value(context, 'global_planner_mode', PAPER_LAUNCH_DEFAULTS['global_planner_mode']).strip().lower(),
        'preselected_route_json': _launch_value(
            context, 'preselected_route_json', PAPER_LAUNCH_DEFAULTS['preselected_route_json']
        ).strip(),
        'preselected_route_sha256': _launch_value(
            context, 'preselected_route_sha256', PAPER_LAUNCH_DEFAULTS['preselected_route_sha256']
        ).strip(),
        'preselected_route_source_path': _launch_value(
            context,
            'preselected_route_source_path',
            PAPER_LAUNCH_DEFAULTS['preselected_route_source_path'],
        ).strip(),
        'preselected_route_source_sha256': _launch_value(
            context,
            'preselected_route_source_sha256',
            PAPER_LAUNCH_DEFAULTS['preselected_route_source_sha256'],
        ).strip(),
        'preselected_route_clearance_m': float(_launch_value(
            context,
            'preselected_route_clearance_m',
            PAPER_LAUNCH_DEFAULTS['preselected_route_clearance_m'],
        )),
        'preselected_route_endpoint_tolerance_m': float(_launch_value(
            context,
            'preselected_route_endpoint_tolerance_m',
            PAPER_LAUNCH_DEFAULTS['preselected_route_endpoint_tolerance_m'],
        )),
        'preselected_route_sample_step_m': float(_launch_value(
            context,
            'preselected_route_sample_step_m',
            PAPER_LAUNCH_DEFAULTS['preselected_route_sample_step_m'],
        )),
        'global_horizon': int(_launch_value(context, 'global_horizon', PAPER_LAUNCH_DEFAULTS['global_horizon'])),
        'global_dt': float(_launch_value(context, 'global_dt', PAPER_LAUNCH_DEFAULTS['global_dt'])),
        'local_horizon': int(_launch_value(context, 'local_horizon', PAPER_LAUNCH_DEFAULTS['local_horizon'])),
        'local_plan_rate': float(_launch_value(context, 'local_plan_rate', PAPER_LAUNCH_DEFAULTS['local_plan_rate'])),
        'local_optimizer_maxiter': int(_launch_value(context, 'local_optimizer_maxiter', PAPER_LAUNCH_DEFAULTS['local_optimizer_maxiter'])),
        'global_use_ambiguity': _as_bool(_launch_value(context, 'global_use_ambiguity', PAPER_LAUNCH_DEFAULTS['global_use_ambiguity'])),
        'local_use_ambiguity': _as_bool(_launch_value(context, 'local_use_ambiguity', PAPER_LAUNCH_DEFAULTS['local_use_ambiguity'])),
        'local_use_obs_risk': _as_bool(_launch_value(context, 'local_use_obs_risk', PAPER_LAUNCH_DEFAULTS['local_use_obs_risk'])),
        'global_optimizer_multistart': _as_bool(_launch_value(
            context, 'global_optimizer_multistart', PAPER_LAUNCH_DEFAULTS['global_optimizer_multistart']
        )),
        'local_optimizer_multistart': _as_bool(_launch_value(
            context, 'local_optimizer_multistart', PAPER_LAUNCH_DEFAULTS['local_optimizer_multistart']
        )),
        'local_use_visibility_model': _as_bool(_launch_value(
            context, 'local_use_visibility_model', PAPER_LAUNCH_DEFAULTS['local_use_visibility_model']
        )),
        'local_use_belief_nogo_cost': _as_bool(_launch_value(
            context, 'local_use_belief_nogo_cost', PAPER_LAUNCH_DEFAULTS['local_use_belief_nogo_cost']
        )),
        'local_nogo_penalty_type': _launch_value(
            context, 'local_nogo_penalty_type', PAPER_LAUNCH_DEFAULTS['local_nogo_penalty_type']
        ).strip().lower(),
        'local_nogo_safe_distance': float(_launch_value(
            context, 'local_nogo_safe_distance', PAPER_LAUNCH_DEFAULTS['local_nogo_safe_distance']
        )),
        'local_goal_prior_u_std_start': float(_launch_value(
            context,
            'local_goal_prior_u_std_start',
            PAPER_LAUNCH_DEFAULTS['local_goal_prior_u_std_start'],
        )),
        'local_goal_prior_v_std_start': float(_launch_value(
            context,
            'local_goal_prior_v_std_start',
            PAPER_LAUNCH_DEFAULTS['local_goal_prior_v_std_start'],
        )),
        'local_goal_prior_u_std_final': float(_launch_value(
            context,
            'local_goal_prior_u_std_final',
            PAPER_LAUNCH_DEFAULTS['local_goal_prior_u_std_final'],
        )),
        'local_goal_prior_v_std_final': float(_launch_value(
            context,
            'local_goal_prior_v_std_final',
            PAPER_LAUNCH_DEFAULTS['local_goal_prior_v_std_final'],
        )),
        'waypoint_spacing_m': float(_launch_value(context, 'waypoint_spacing_m', PAPER_LAUNCH_DEFAULTS['waypoint_spacing_m'])),
        'waypoint_arrival_radius_m': float(_launch_value(context, 'waypoint_arrival_radius_m', PAPER_LAUNCH_DEFAULTS['waypoint_arrival_radius_m'])),
        'local_replan_min_remaining_s': float(_launch_value(
            context,
            'local_replan_min_remaining_s',
            PAPER_LAUNCH_DEFAULTS['local_replan_min_remaining_s'],
        )),
        'local_replan_on_waypoint_change': _as_bool(_launch_value(
            context,
            'local_replan_on_waypoint_change',
            PAPER_LAUNCH_DEFAULTS['local_replan_on_waypoint_change'],
        )),
        'latency_compensate_plan_handoff': _as_bool(_launch_value(
            context,
            'latency_compensate_plan_handoff',
            PAPER_LAUNCH_DEFAULTS['latency_compensate_plan_handoff'],
        )),
        'simple_tracker_yaw_gate_rad': float(_launch_value(
            context,
            'simple_tracker_yaw_gate_rad',
            PAPER_LAUNCH_DEFAULTS['simple_tracker_yaw_gate_rad'],
        )),
        'ff_fb_turn_rate_limit_rad_s': float(_launch_value(
            context,
            'ff_fb_turn_rate_limit_rad_s',
            PAPER_LAUNCH_DEFAULTS['ff_fb_turn_rate_limit_rad_s'],
        )),
        'ff_fb_corner_crawl_speed_mps': float(_launch_value(
            context,
            'ff_fb_corner_crawl_speed_mps',
            PAPER_LAUNCH_DEFAULTS['ff_fb_corner_crawl_speed_mps'],
        )),
        'ff_fb_pivot_heading_error_rad': float(_launch_value(
            context,
            'ff_fb_pivot_heading_error_rad',
            PAPER_LAUNCH_DEFAULTS['ff_fb_pivot_heading_error_rad'],
        )),
        'odom_heading_timeout_s': float(_launch_value(
            context,
            'odom_heading_timeout_s',
            PAPER_LAUNCH_DEFAULTS['odom_heading_timeout_s'],
        )),
        'heading_update_mode': _launch_value(
            context, 'heading_update_mode', PAPER_LAUNCH_DEFAULTS['heading_update_mode']
        ).strip().lower(),
        'local_controller_type': _launch_value(
            context, 'local_controller_type', PAPER_LAUNCH_DEFAULTS['local_controller_type']
        ).strip().lower(),
        'plan_rate': float(_launch_value(context, 'plan_rate', PAPER_LAUNCH_DEFAULTS['plan_rate'])),
        'cmd_publish_rate': float(_launch_value(
            context, 'cmd_publish_rate', PAPER_LAUNCH_DEFAULTS['cmd_publish_rate']
        )),
        'command_noise_output_topic': _launch_value(
            context, 'command_noise_output_topic', PAPER_LAUNCH_DEFAULTS['command_noise_output_topic']
        ).strip(),
        'belief_publish_rate': float(_launch_value(context, 'belief_publish_rate', PAPER_LAUNCH_DEFAULTS['belief_publish_rate'])),
        'horizon': int(_launch_value(context, 'horizon', PAPER_LAUNCH_DEFAULTS['horizon'])),
        'dt': float(_launch_value(context, 'dt', PAPER_LAUNCH_DEFAULTS['dt'])),
        'v_max': float(_launch_value(context, 'v_max', PAPER_LAUNCH_DEFAULTS['v_max'])),
        'max_predict_speed_mps': float(_launch_value(
            context, 'max_predict_speed_mps', PAPER_LAUNCH_DEFAULTS['max_predict_speed_mps']
        )),
        'state_max_correction_jump_m': float(_launch_value(
            context, 'state_max_correction_jump_m',
            PAPER_LAUNCH_DEFAULTS['state_max_correction_jump_m']
        )),
        'state_correction_mode': _launch_value(
            context, 'state_correction_mode', PAPER_LAUNCH_DEFAULTS['state_correction_mode']
        ).strip().lower(),
        'control_weight': float(_launch_value(context, 'control_weight', PAPER_LAUNCH_DEFAULTS['control_weight'])),
        'risk_weight_obs': float(_launch_value(context, 'risk_weight_obs', PAPER_LAUNCH_DEFAULTS['risk_weight_obs'])),
        'ambiguity_weight': float(_launch_value(context, 'ambiguity_weight', PAPER_LAUNCH_DEFAULTS['ambiguity_weight'])),
        'r_visible_uv': float(_launch_value(context, 'r_visible_uv', PAPER_LAUNCH_DEFAULTS['r_visible_uv'])),
        'r_miss_uv': float(_launch_value(context, 'r_miss_uv', PAPER_LAUNCH_DEFAULTS['r_miss_uv'])),
        'visibility_sigma_kappa': float(_launch_value(context, 'visibility_sigma_kappa', PAPER_LAUNCH_DEFAULTS['visibility_sigma_kappa'])),
        'goal_prior_u_std_start': float(_launch_value(context, 'goal_prior_u_std_start', PAPER_LAUNCH_DEFAULTS['goal_prior_u_std_start'])),
        'goal_prior_v_std_start': float(_launch_value(context, 'goal_prior_v_std_start', PAPER_LAUNCH_DEFAULTS['goal_prior_v_std_start'])),
        'goal_prior_u_std_final': float(_launch_value(context, 'goal_prior_u_std_final', PAPER_LAUNCH_DEFAULTS['goal_prior_u_std_final'])),
        'goal_prior_v_std_final': float(_launch_value(context, 'goal_prior_v_std_final', PAPER_LAUNCH_DEFAULTS['goal_prior_v_std_final'])),
        'goal_tightening_power': float(_launch_value(context, 'goal_tightening_power', PAPER_LAUNCH_DEFAULTS['goal_tightening_power'])),
        'goal_progress_n_steps': int(_launch_value(context, 'goal_progress_n_steps', PAPER_LAUNCH_DEFAULTS['goal_progress_n_steps'])),
        'observation_risk_scale': float(_launch_value(context, 'observation_risk_scale', PAPER_LAUNCH_DEFAULTS['observation_risk_scale'])),
        'ambiguity_term_scale': float(_launch_value(context, 'ambiguity_term_scale', PAPER_LAUNCH_DEFAULTS['ambiguity_term_scale'])),
        'discount_gamma': float(_launch_value(context, 'discount_gamma', PAPER_LAUNCH_DEFAULTS['discount_gamma'])),
        'use_visibility_model': _as_bool(
            visibility_enabled_default if use_visibility_raw in ('', 'auto', 'default') else use_visibility_raw
        ),
        'perception_use_geometry_occlusion': _as_bool(
            _launch_value(context, 'perception_use_geometry_occlusion', 'true')
        ),
        'visibility_target_height_m': float(_launch_value(context, 'visibility_target_height_m', str(VISIBILITY_FALLBACK_DEFAULTS['visibility_target_height_m']))),
        'visibility_geometry_json': _launch_value(context, 'visibility_geometry_json', ''),
        'collision_geometry_json': _launch_value(context, 'collision_geometry_json', ''),
        'driveable_geometry_json': _launch_value(context, 'driveable_geometry_json', ''),
        'visibility_artifact_path': _launch_value(context, 'visibility_artifact_path', ''),
        'camera_network_artifact_path': _launch_value(context, 'camera_network_artifact_path', ''),
        'camera_network_active_camera_ids': _launch_value(
            context, 'camera_network_active_camera_ids', ''),
        'camera_network_objective': _launch_value(
            context, 'camera_network_objective',
            PAPER_LAUNCH_DEFAULTS['camera_network_objective'],
        ).strip().lower(),
        'kouw_et1_ambiguity': _as_bool(_launch_value(
            context, 'kouw_et1_ambiguity', PAPER_LAUNCH_DEFAULTS['kouw_et1_ambiguity'])),
        'network_goal_std_start_m': float(_launch_value(
            context, 'network_goal_std_start_m',
            PAPER_LAUNCH_DEFAULTS['network_goal_std_start_m'],
        )),
        'network_goal_std_m': float(_launch_value(
            context, 'network_goal_std_m', PAPER_LAUNCH_DEFAULTS['network_goal_std_m'],
        )),
        'camera_network_updates_per_step': int(_launch_value(
            context, 'camera_network_updates_per_step',
            PAPER_LAUNCH_DEFAULTS['camera_network_updates_per_step'],
        )),
        'optimizer_control_block_steps': int(_launch_value(
            context, 'optimizer_control_block_steps',
            PAPER_LAUNCH_DEFAULTS['optimizer_control_block_steps'],
        )),
        'use_nogo_cost': _launch_value(context, 'use_nogo_cost', str(VISIBILITY_FALLBACK_DEFAULTS['use_nogo_cost'])).strip().lower(),
        'nogo_penalty_type': _launch_value(context, 'nogo_penalty_type', str(VISIBILITY_FALLBACK_DEFAULTS['nogo_penalty_type'])).strip().lower(),
        'nogo_safe_distance': float(_launch_value(context, 'nogo_safe_distance', str(VISIBILITY_FALLBACK_DEFAULTS['nogo_safe_distance']))),
        'nogo_logbarrier_eps': float(_launch_value(context, 'nogo_logbarrier_eps', str(VISIBILITY_FALLBACK_DEFAULTS['nogo_logbarrier_eps']))),
        'nogo_warning_band': float(_launch_value(context, 'nogo_warning_band', str(VISIBILITY_FALLBACK_DEFAULTS['nogo_warning_band']))),
        'nogo_near_weight': float(_launch_value(context, 'nogo_near_weight', str(VISIBILITY_FALLBACK_DEFAULTS['nogo_near_weight']))),
        'use_belief_nogo_cost': _as_bool(_launch_value(context, 'use_belief_nogo_cost', str(VISIBILITY_FALLBACK_DEFAULTS['use_belief_nogo_cost']))),
        'nogo_belief_kappa': float(_launch_value(context, 'nogo_belief_kappa', str(VISIBILITY_FALLBACK_DEFAULTS['nogo_belief_kappa']))),
        'nogo_mode': _launch_value(context, 'nogo_mode', str(VISIBILITY_FALLBACK_DEFAULTS['nogo_mode'])).strip().lower(),
        'use_hit_miss_mixture': _as_bool(_launch_value(context, 'use_hit_miss_mixture', str(VISIBILITY_FALLBACK_DEFAULTS['use_hit_miss_mixture']))),
        'goal_sigma_uv': float(_launch_value(context, 'goal_sigma_uv', PAPER_LAUNCH_DEFAULTS['goal_sigma_uv'])),
        'robot_length_m': float(_launch_value(context, 'robot_length_m', PAPER_LAUNCH_DEFAULTS['robot_length_m'])),
        'robot_width_m': float(_launch_value(context, 'robot_width_m', PAPER_LAUNCH_DEFAULTS['robot_width_m'])),
        'robot_collision_radius_m': float(
            _launch_value(
                context,
                'robot_collision_radius_m',
                PAPER_LAUNCH_DEFAULTS['robot_collision_radius_m'],
            )
        ),
        'bridge_camera_a': _as_bool(
            _launch_value(context, 'bridge_camera_a', PAPER_LAUNCH_DEFAULTS['bridge_camera_a'])
        ),
        'bridge_camera_b': _as_bool(
            _launch_value(context, 'bridge_camera_b', PAPER_LAUNCH_DEFAULTS['bridge_camera_b'])
        ),
        'bridge_camera_c': _as_bool(
            _launch_value(context, 'bridge_camera_c', PAPER_LAUNCH_DEFAULTS['bridge_camera_c'])
        ),
        'bridge_camera_d': _as_bool(
            _launch_value(context, 'bridge_camera_d', PAPER_LAUNCH_DEFAULTS['bridge_camera_d'])
        ),
        'use_command_noise': _as_bool(
            _launch_value(context, 'use_command_noise', PAPER_LAUNCH_DEFAULTS['use_command_noise'])
        ),
        'use_encoder_noise': _as_bool(
            _launch_value(context, 'use_encoder_noise', PAPER_LAUNCH_DEFAULTS['use_encoder_noise'])
        ),
        'use_odom_for_predict': _as_bool(
            _launch_value(context, 'use_odom_for_predict', PAPER_LAUNCH_DEFAULTS['use_odom_for_predict'])
        ),
        'odom_topic': _launch_value(context, 'odom_topic', PAPER_LAUNCH_DEFAULTS['odom_topic']).strip(),
        'headless': _as_bool(_launch_value(context, 'headless', 'false')),
        'nvidia_offload': _as_bool(_launch_value(context, 'nvidia_offload', 'true')),
        'reset_world': _as_bool(_launch_value(context, 'reset_world', PAPER_LAUNCH_DEFAULTS['reset_world'])),
        'command_noise_linear_slip_mean': float(
            _launch_value(context, 'command_noise_linear_slip_mean', _COMMAND_NOISE_LINEAR_SLIP_MEAN)
        ),
        'command_noise_linear_slip_std': float(
            _launch_value(context, 'command_noise_linear_slip_std', _COMMAND_NOISE_LINEAR_SLIP_STD)
        ),
        'command_noise_angular_slip_mean': float(
            _launch_value(context, 'command_noise_angular_slip_mean', _COMMAND_NOISE_ANGULAR_SLIP_MEAN)
        ),
        'command_noise_angular_slip_std': float(
            _launch_value(context, 'command_noise_angular_slip_std', _COMMAND_NOISE_ANGULAR_SLIP_STD)
        ),
        'command_noise_linear_additive_std': float(
            _launch_value(
                context, 'command_noise_linear_additive_std', _COMMAND_NOISE_LINEAR_ADDITIVE_STD
            )
        ),
        'command_noise_angular_additive_std': float(
            _launch_value(
                context, 'command_noise_angular_additive_std', _COMMAND_NOISE_ANGULAR_ADDITIVE_STD
            )
        ),
        'command_noise_correlation_alpha': float(
            _launch_value(context, 'command_noise_correlation_alpha', _COMMAND_NOISE_CORRELATION_ALPHA)
        ),
        'encoder_noise_linear_slip_mean': float(
            _launch_value(context, 'encoder_noise_linear_slip_mean', _ENCODER_NOISE_LINEAR_SLIP_MEAN)
        ),
        'encoder_noise_linear_slip_std': float(
            _launch_value(context, 'encoder_noise_linear_slip_std', _ENCODER_NOISE_LINEAR_SLIP_STD)
        ),
        'encoder_noise_angular_slip_mean': float(
            _launch_value(context, 'encoder_noise_angular_slip_mean', _ENCODER_NOISE_ANGULAR_SLIP_MEAN)
        ),
        'encoder_noise_angular_slip_std': float(
            _launch_value(context, 'encoder_noise_angular_slip_std', _ENCODER_NOISE_ANGULAR_SLIP_STD)
        ),
        'encoder_noise_linear_additive_std': float(
            _launch_value(context, 'encoder_noise_linear_additive_std', _ENCODER_NOISE_LINEAR_ADDITIVE_STD)
        ),
        'encoder_noise_angular_additive_std': float(
            _launch_value(context, 'encoder_noise_angular_additive_std', _ENCODER_NOISE_ANGULAR_ADDITIVE_STD)
        ),
        'encoder_noise_correlation_alpha': float(
            _launch_value(context, 'encoder_noise_correlation_alpha', _ENCODER_NOISE_CORRELATION_ALPHA)
        ),
        'encoder_wheel_diameter_ratio_error': float(
            _launch_value(context, 'encoder_wheel_diameter_ratio_error', _ENCODER_WHEEL_DIAMETER_RATIO_ERROR)
        ),
        'encoder_wheelbase_ratio': float(
            _launch_value(context, 'encoder_wheelbase_ratio', _ENCODER_WHEELBASE_RATIO)
        ),
        'min_state_cov': float(_launch_value(context, 'min_state_cov', '1e-6')),
        'debug_runtime': _as_bool(_launch_value(context, 'debug_runtime', 'false')),
        'enable_logging': _as_bool(_launch_value(context, 'enable_logging', 'true')),
        'use_rviz': _as_bool(_launch_value(context, 'use_rviz', 'false')),
        'rviz_config': _launch_value(context, 'rviz_config', ''),
        'yolo_model': _launch_value(context, 'yolo_model', PAPER_LAUNCH_DEFAULTS['yolo_model']),
        'outcome_journal_path': _launch_value(
            context, 'outcome_journal_path', PAPER_LAUNCH_DEFAULTS['outcome_journal_path']
        ).strip(),
        'manager_outcome_journal_path': _launch_value(
            context, 'manager_outcome_journal_path',
            PAPER_LAUNCH_DEFAULTS['manager_outcome_journal_path']
        ).strip(),
        'yolo_device': _launch_value(context, 'yolo_device', PAPER_LAUNCH_DEFAULTS['yolo_device']),
        'yolo_imgsz': int(_launch_value(context, 'yolo_imgsz', PAPER_LAUNCH_DEFAULTS['yolo_imgsz'])),
        'yolo_conf_threshold': float(_launch_value(context, 'yolo_conf_threshold', PAPER_LAUNCH_DEFAULTS['yolo_conf_threshold'])),
        'yolo_predict_conf_floor': float(_launch_value(
            context, 'yolo_predict_conf_floor', PAPER_LAUNCH_DEFAULTS['yolo_predict_conf_floor'])),
        'yolo_iou_threshold': float(_launch_value(context, 'yolo_iou_threshold', PAPER_LAUNCH_DEFAULTS['yolo_iou_threshold'])),
        'yolo_target_class': _launch_value(context, 'yolo_target_class', PAPER_LAUNCH_DEFAULTS['yolo_target_class']),
        'yolo_class_id': int(_launch_value(context, 'yolo_class_id', PAPER_LAUNCH_DEFAULTS['yolo_class_id'])),
        'yolo_use_masks': _as_bool(_launch_value(context, 'yolo_use_masks', PAPER_LAUNCH_DEFAULTS['yolo_use_masks'])),
        'yolo_min_mask_area_px': float(_launch_value(context, 'yolo_min_mask_area_px', PAPER_LAUNCH_DEFAULTS['yolo_min_mask_area_px'])),
        'yolo_mask_bottom_band_px': float(_launch_value(context, 'yolo_mask_bottom_band_px', PAPER_LAUNCH_DEFAULTS['yolo_mask_bottom_band_px'])),
        'yolo_min_bbox_area_px': float(_launch_value(context, 'yolo_min_bbox_area_px', PAPER_LAUNCH_DEFAULTS['yolo_min_bbox_area_px'])),
        'yolo_max_batch_stamp_skew_s': float(_launch_value(
            context, 'yolo_max_batch_stamp_skew_s',
            PAPER_LAUNCH_DEFAULTS['yolo_max_batch_stamp_skew_s'])),
        'yolo_debug_frame_dir': _launch_value(context, 'yolo_debug_frame_dir', ''),
        'yolo_debug_crop_dir': _launch_value(context, 'yolo_debug_crop_dir', ''),
        'yolo_use_torchscript': _as_bool(_launch_value(context, 'yolo_use_torchscript', 'false')),
        'yolo_runtime_backend': _launch_value(context, 'yolo_runtime_backend', 'native').strip().lower(),
        'yolo_compiled_model': _launch_value(context, 'yolo_compiled_model', '').strip(),
        'yolo_input_transport': _launch_value(context, 'yolo_input_transport', 'ros').strip().lower(),
        'yolo_runtime_trace_period_s': float(_launch_value(context, 'yolo_runtime_trace_period_s', '0.0')),
        'yolo_warmup_iters': int(_launch_value(context, 'yolo_warmup_iters', '3')),
        'yolo_inference_in_callback': _as_bool(_launch_value(context, 'yolo_inference_in_callback', 'true')),
    }
    if cfg['heading_update_mode'] not in ('camera_xy_only', 'coupled'):
        raise RuntimeError(
            "heading_update_mode must be 'camera_xy_only' or 'coupled'")
    try:
        cfg['local_controller_type'] = local_controller_type(cfg['local_controller_type'])
        parse_bev_affine_calibration(cfg['bev_affine_calibration'])
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    if not (0.0 <= cfg['yolo_max_batch_stamp_skew_s'] < 0.20):
        raise RuntimeError(
            "yolo_max_batch_stamp_skew_s must be non-negative and strictly below "
            "the 0.20 s camera period; otherwise adjacent capture rounds can merge"
        )
    if not (0.0 < cfg['yolo_predict_conf_floor'] <= cfg['yolo_conf_threshold'] <= 1.0):
        raise RuntimeError(
            "yolo_predict_conf_floor must be positive and no larger than the reported "
            "yolo_conf_threshold, which must not exceed one"
        )
    if (
        cfg['enable_logging']
        and (
            cfg['yolo_runtime_backend'] != 'native'
            or cfg['yolo_input_transport'] != 'ros'
        )
    ):
        raise RuntimeError(
            'compiled/direct-Gazebo YOLO is a diagnostic runtime successor and '
            'is blocked from evidence logging; set enable_logging:=false for commissioning'
        )

    return cfg


def resolve_world_setup(cfg: Dict[str, object]) -> Dict[str, object]:
    """Resolve world profile/task and derive camera/spawn launch parameters."""
    import json as _json
    from experiments.core.world_profiles import (
        load_profile,
        compute_camera_quaternion_from_rpy,
        compute_look_at_from_pose,
        resolve_profile_asset_path,
        serialize_collision_geometry_from_world,
        serialize_driveable_geometry_from_profile,
        serialize_occlusion_geometry_from_world,
    )
    from experiments.core.tasks import load_tasks, select_task

    profile, _intrinsics, world_path, camera_pose = load_profile(
        cfg['world_profiles_path'], cfg['world']
    )
    tasks_by_world = load_tasks(cfg['tasks_yaml'])
    task_name = str(cfg.get('task_name', '') or '').strip()
    if not task_name:
        task_name = str(profile.get('recommended_task', '') or '').strip()
    task = select_task(tasks_by_world, cfg['world'], task_name)

    # Evaluation tasks may declare the complete condition-neutral homotopy
    # set. Passing it unchanged prevents campaign seeds from degenerating into
    # several perturbations of one aisle.
    route_seeds = task.get('route_seeds') if isinstance(task, dict) else None
    if route_seeds:
        if not isinstance(route_seeds, list) or len(route_seeds) < 2:
            raise RuntimeError('task route_seeds must contain at least two routes')
        cfg['optimizer_initial_routes_json'] = _json.dumps(route_seeds)
        cfg['optimizer_route_seed_mode'] = 'explicit'

    start = _require_task_field(task, 'start')
    goal = _require_task_field(task, 'goal')
    for key in ('x', 'y', 'yaw'):
        if key not in start:
            raise RuntimeError(f"Task start missing '{key}'")
    for key in ('x', 'y'):
        if key not in goal:
            raise RuntimeError(f"Task goal missing '{key}'")

    profile_spawn = profile.get('spawn', {}) if isinstance(profile.get('spawn', {}), dict) else {}
    start_z = start.get('z', profile_spawn.get('z', 0.05))

    spawn = {
        'x': float(start['x']),
        'y': float(start['y']),
        'z': float(start_z),
        'yaw': float(start['yaw']),
    }
    goal_x = float(goal['x'])
    goal_y = float(goal['y'])
    # Optional multi-goal tour: task 'waypoints' = ordered list of {x,y}. The goal
    # node drives them in sequence; the FINAL waypoint is the mission goal (used
    # for success/auto-stop), so override goal_x/goal_y to it.
    waypoints = task.get('waypoints') if isinstance(task, dict) else None
    waypoints_json = ''
    if waypoints:
        pts = [[float(w['x']), float(w['y'])] for w in waypoints]
        waypoints_json = _json.dumps(pts)
        goal_x, goal_y = pts[-1][0], pts[-1][1]

    planner = str(cfg['planner'])
    if planner == 'auto':
        raise RuntimeError("planner must be explicit for current active runs; 'auto' was retired")
    if planner == 'constant_R_efe':
        cfg['use_visibility_model'] = False
        # C1 is still an EFE planner. It uses constant observation covariance
        # instead of the GP-conditioned covariance, but it does not remove the
        # ambiguity term.
        cfg['use_ambiguity'] = True
        cfg['use_obs_risk'] = True
    visibility_artifact_path = str(cfg.get('visibility_artifact_path', '') or '').strip()
    camera_network_artifact_path = str(cfg.get('camera_network_artifact_path', '') or '').strip()
    camera_network_objective = str(
        cfg.get('camera_network_objective', 'legacy_pixel_chart') or '').strip().lower()
    if camera_network_objective not in ('legacy_pixel_chart', 'metric_expected_belief'):
        raise RuntimeError('unknown camera_network_objective')
    if camera_network_objective == 'metric_expected_belief' and not camera_network_artifact_path:
        raise RuntimeError('metric_expected_belief requires camera_network_artifact_path')
    network_goal_std_m = float(cfg.get('network_goal_std_m', 0.10))
    if not np.isfinite(network_goal_std_m) or network_goal_std_m <= 0.:
        raise RuntimeError('network_goal_std_m must be finite and positive')
    network_updates = int(cfg.get('camera_network_updates_per_step', 1))
    if network_updates < 1:
        raise RuntimeError('camera_network_updates_per_step must be positive')
    cfg['camera_network_objective'] = camera_network_objective
    cfg['network_goal_std_m'] = network_goal_std_m
    # <= 0 means 'no anneal'; the planner takes None and keeps a constant prior.
    _goal_start = float(cfg.get('network_goal_std_start_m', -1.0) or -1.0)
    cfg['network_goal_std_start_m'] = _goal_start if _goal_start > 0. else -1.0
    cfg['camera_network_updates_per_step'] = network_updates
    control_block_steps = int(cfg.get('optimizer_control_block_steps', 1))
    if control_block_steps < 1:
        raise RuntimeError('optimizer_control_block_steps must be positive')
    cfg['optimizer_control_block_steps'] = control_block_steps
    if camera_network_artifact_path:
        if visibility_artifact_path:
            raise RuntimeError('choose one planner field: visibility_artifact_path or camera_network_artifact_path')
        if planner != 'visibility_aware_efe' or str(cfg.get('global_planner_mode', 'efe')).strip().lower() == 'preselected_route':
            raise RuntimeError('camera_network_artifact_path requires a solved visibility_aware_efe plan')
        if _as_bool(cfg.get('use_pixel_correction', False)):
            raise RuntimeError('camera network planning requires metric camera corrections')
        if _as_bool(cfg.get('use_hit_miss_mixture', False)):
            raise RuntimeError('camera-network objective owns its reporting-event model')
        camera_network_artifact_path = resolve_profile_asset_path(cfg['world_profiles_path'], camera_network_artifact_path)
        if not Path(camera_network_artifact_path).is_file():
            raise RuntimeError(f'camera_network_artifact_path does not exist: {camera_network_artifact_path}')
    # Only an actually solved visibility-aware global plan consumes this GP.
    # A preselected route may retain the same local/filter configuration, but
    # its route is hash-bound and no global EFE objective is evaluated.
    if (
        planner == 'visibility_aware_efe'
        and str(cfg.get('global_planner_mode', 'efe')).strip().lower()
        != 'preselected_route'
    ):
        if not visibility_artifact_path and not camera_network_artifact_path:
            raise RuntimeError(
                "visibility_artifact_path or camera_network_artifact_path must be provided explicitly — "
                "no fallback to world profile defaults is allowed."
            )
        if visibility_artifact_path:
            visibility_artifact_path = resolve_profile_asset_path(cfg['world_profiles_path'], visibility_artifact_path)
            if not Path(visibility_artifact_path).exists():
                raise RuntimeError(f"visibility_artifact_path does not exist: {visibility_artifact_path}")

    cam_pos = [camera_pose[0], camera_pose[1], camera_pose[2]]
    roll, pitch, yaw = camera_pose[3], camera_pose[4], camera_pose[5]
    look_at = compute_look_at_from_pose(cam_pos, roll, pitch, yaw)
    quat = compute_camera_quaternion_from_rpy(roll, pitch, yaw)
    spawn_quat = compute_camera_quaternion_from_rpy(0.0, 0.0, spawn['yaw'])

    intrinsics = dict(_intrinsics)
    profile_camera_ids = [
        str(item).strip() for item in
        (profile.get('camera_ids') or ['camera_A', 'camera_B', 'camera_C', 'camera_D'])
    ]
    profile_camera_models = [
        str(item).strip() for item in
        (profile.get('camera_model_includes') or [
            'external_camera', 'external_camera_b', 'external_camera_c', 'external_camera_d',
        ])
    ]
    if (
        not profile_camera_ids
        or len(profile_camera_ids) != len(profile_camera_models)
        or len(set(profile_camera_ids)) != len(profile_camera_ids)
        or len(set(profile_camera_models)) != len(profile_camera_models)
        or any(not item for item in (*profile_camera_ids, *profile_camera_models))
    ):
        raise RuntimeError(
            f"World profile camera_ids and camera_model_includes must be aligned, "
            f"non-empty, and unique for {cfg['world']}"
        )
    profile_camera_topics = [
        str(item).strip() for item in
        (profile.get('camera_image_topics') or [
            f'/{model}/image_raw' for model in profile_camera_models
        ])
    ]
    if (
        len(profile_camera_topics) != len(profile_camera_ids)
        or len(set(profile_camera_topics)) != len(profile_camera_topics)
        or any(not item for item in profile_camera_topics)
    ):
        raise RuntimeError(
            f"World profile camera_image_topics must align with camera_ids for {cfg['world']}"
        )
    network_identity = {
        'camera_network_expected_sha256': '',
        'camera_network_expected_source_hashes_json': '',
        'camera_network_camera_ids': '',
        'camera_network_active_camera_ids': '',
    }
    if camera_network_artifact_path:
        active_camera_ids = tuple(
            value.strip() for value in str(
                cfg.get('camera_network_active_camera_ids', '') or '').split(',')
            if value.strip()
        )
        network_identity = _camera_network_identity(
            camera_network_artifact_path, profile_camera_ids,
            active_camera_ids=active_camera_ids or None,
        )
    camera_params = {
        'cam_pos': cam_pos,
        'look_at': look_at,
        'img_width': int(intrinsics['img_width']),
        'img_height': int(intrinsics['img_height']),
        'fov_h_rad': float(intrinsics['fov_h_rad']),
    }
    tf_args = {
        'use_sim_time': 'true',
        'cam_x': str(cam_pos[0]),
        'cam_y': str(cam_pos[1]),
        'cam_z': str(cam_pos[2]),
        'cam_qx': str(quat[0]),
        'cam_qy': str(quat[1]),
        'cam_qz': str(quat[2]),
        'cam_qw': str(quat[3]),
        'odom_x': str(spawn['x']),
        'odom_y': str(spawn['y']),
        'odom_z': '0.0',
        'odom_qx': str(spawn_quat[0]),
        'odom_qy': str(spawn_quat[1]),
        'odom_qz': str(spawn_quat[2]),
        'odom_qw': str(spawn_quat[3]),
    }

    visibility_geometry_json = str(cfg.get('visibility_geometry_json', '') or '')
    collision_geometry_json = str(cfg.get('collision_geometry_json', '') or '')
    driveable_geometry_json = str(cfg.get('driveable_geometry_json', '') or '')
    if not driveable_geometry_json:
        # This argument is the hard keep-in authority.  It is always the site
        # boundary; collision_geometry_json independently supplies keep-out
        # geometry.  Old lane unions are not a safety boundary.
        driveable_geometry_json = serialize_driveable_geometry_from_profile(profile)
    raw_use_nogo_cost = str(cfg.get('use_nogo_cost', 'auto')).strip().lower()
    nogo_geometry_needed = (
        raw_use_nogo_cost in ('1', 'true', 't', 'yes', 'y', 'on')
    )
    geometry_needed = _as_bool(cfg.get('perception_use_geometry_occlusion', False)) or nogo_geometry_needed
    occlusion_model_names = _profile_name_tuple(
        profile,
        'occlusion_model_names',
        'occlusion_model_name',
    )
    collision_model_names = _profile_name_tuple(
        profile,
        'collision_model_names',
        'collision_model_name',
    )
    collision_include_names = _profile_name_tuple(
        profile,
        'collision_include_names',
        'collision_include_name',
    )
    if not collision_include_names:
        collision_include_names = _COLLISION_INCLUDE_NAMES_BY_WORLD.get(
            str(cfg['world']), ()
        )
    if (not visibility_geometry_json) and geometry_needed:
        visibility_geometry_json = serialize_occlusion_geometry_from_world(
            world_path,
            model_name=occlusion_model_names or 'warehouse_rack_occluders',
        )
    if not collision_geometry_json:
        if collision_model_names:
            collision_geometry_json = serialize_collision_geometry_from_world(
                world_path,
                model_names=collision_model_names,
                include_names=collision_include_names,
            )
        else:
            collision_geometry_json = serialize_collision_geometry_from_world(
                world_path, include_names=collision_include_names
            )

    global_planner_mode = str(cfg.get('global_planner_mode', 'efe') or 'efe').strip().lower()
    allowed_global_modes = ('efe', 'geometric_shortest_path', 'preselected_route')
    if global_planner_mode not in allowed_global_modes:
        raise RuntimeError(
            f"global_planner_mode must be one of: {', '.join(allowed_global_modes)}"
        )

    route_argument_names = (
        'preselected_route_json',
        'preselected_route_sha256',
        'preselected_route_source_path',
        'preselected_route_source_sha256',
    )
    route_args_present = any(str(cfg.get(name, '') or '').strip() for name in route_argument_names)
    preselected_route_validation_json = ''
    if global_planner_mode == 'preselected_route':
        if not _as_bool(cfg.get('use_hierarchical', False)):
            raise RuntimeError(
                "global_planner_mode='preselected_route' requires use_hierarchical:=true "
                "so the existing belief-based local waypoint tracker executes the route"
            )
        if waypoints_json:
            raise RuntimeError(
                "preselected_route accepts one start-to-goal polyline; task waypoint tours "
                "would change the goal during execution and are not allowed"
            )
        endpoint_tolerance_m = float(
            cfg.get('preselected_route_endpoint_tolerance_m', 0.25)
        )
        if endpoint_tolerance_m < 0.0 or endpoint_tolerance_m > 0.25:
            raise RuntimeError(
                "preselected_route_endpoint_tolerance_m must be within [0, 0.25] m "
                "under the frozen one-cell endpoint gate"
            )
        sample_step_m = float(cfg.get('preselected_route_sample_step_m', 0.04))
        if sample_step_m <= 0.0 or sample_step_m > 0.04:
            raise RuntimeError(
                "preselected_route_sample_step_m must be within (0, 0.04] m so "
                "between-vertex clearance is not under-sampled"
            )
        try:
            from unav_common.preselected_route import validate_preselected_route

            validated_route = validate_preselected_route(
                str(cfg.get('preselected_route_json', '') or ''),
                str(cfg.get('preselected_route_sha256', '') or ''),
                start_xy=(float(start['x']), float(start['y'])),
                goal_xy=(goal_x, goal_y),
                driveable_geometry_json=driveable_geometry_json,
                declared_clearance_m=float(
                    cfg.get('preselected_route_clearance_m', 0.25)
                ),
                source_path=str(cfg.get('preselected_route_source_path', '') or ''),
                expected_source_sha256=str(
                    cfg.get('preselected_route_source_sha256', '') or ''
                ),
                endpoint_tolerance_m=endpoint_tolerance_m,
                sample_step_m=sample_step_m,
            )
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"preselected route gate failed: {exc}") from exc
        # Downstream components receive only the verified canonical bytes and
        # resolved source path, never the caller's formatting/path aliases.
        cfg['preselected_route_json'] = validated_route.canonical_json
        cfg['preselected_route_sha256'] = validated_route.sha256
        cfg['preselected_route_source_path'] = validated_route.source_path
        cfg['preselected_route_source_sha256'] = validated_route.source_sha256
        preselected_route_validation_json = _json.dumps(
            validated_route.provenance_dict(),
            sort_keys=True,
            separators=(',', ':'),
            allow_nan=False,
        )
    elif route_args_present:
        raise RuntimeError(
            "preselected route arguments were supplied but global_planner_mode is not "
            "'preselected_route'; refusing to ignore a hash-bound route"
        )

    cfg = dict(cfg)
    cfg.update({
        'profile': profile,
        'task': task,
        'task_name': str(task.get('name', task_name)),
        'planner': planner,
        'spawn': spawn,
        'goal_x': goal_x,
        'goal_y': goal_y,
        'waypoints_json': waypoints_json,
        'start_x': float(start['x']),
        'start_y': float(start['y']),
        'start_yaw': float(start['yaw']),
        'camera_params': camera_params,
        'profile_camera_ids': profile_camera_ids,
        'profile_camera_model_includes': profile_camera_models,
        'profile_camera_image_topics': profile_camera_topics,
        'tf_args': tf_args,
        'world_path': world_path,
        'visibility_geometry_json': visibility_geometry_json,
        'collision_geometry_json': collision_geometry_json,
        'driveable_geometry_json': driveable_geometry_json,
        'visibility_artifact_path': visibility_artifact_path,
        'camera_network_artifact_path': camera_network_artifact_path,
        **network_identity,
        'preselected_route_validation_json': preselected_route_validation_json,
    })
    return cfg


def _odom_yaw_offset_rad(cfg: Dict[str, object], odom_topic: str) -> float:
    """Map raw local odometry yaw to map yaw without double-transforming /odom_noisy."""
    return (0.0 if _as_bool(cfg.get('use_encoder_noise', True)) and odom_topic == '/odom_noisy'
            else float(cfg['spawn']['yaw']))


def build_shared_nodes(cfg: Dict[str, object]) -> Dict[str, object]:
    """Create shared nodes/components for the thesis pipeline."""
    state_sources = _state_estimator_metadata(cfg)
    odom_topic = str(cfg.get('odom_topic') or '/odom_noisy')
    use_encoder_noise = _as_bool(cfg.get('use_encoder_noise', True))
    if not use_encoder_noise and odom_topic == '/odom_noisy':
        odom_topic = '/odom'
    # The modelled encoder is initialized from Gazebo's world-frame pose and integrates in
    # that frame, so /odom_noisy already carries map-frame yaw. Raw /odom starts at yaw zero
    # and still needs the declared spawn-yaw transform. Applying the offset to /odom_noisy
    # a second time made every non-zero-yaw task start with the wrong belief heading.
    odom_yaw_offset_rad = _odom_yaw_offset_rad(cfg, odom_topic)
    sim_pkg = FindPackageShare('sim')
    sim_launch_arguments = {
        'use_sim_time': 'true',
        'use_lidar': 'false',
        'show_pose_markers': 'false',
        'bridge_scan': 'false',
        'headless': 'true' if cfg.get('headless', False) else 'false',
        'nvidia_offload': 'true' if cfg.get('nvidia_offload', True) else 'false',
        'world': cfg['world'],
        'world_name': cfg['profile']['world_name'],
        'spawn_x': str(cfg['spawn']['x']),
        'spawn_y': str(cfg['spawn']['y']),
        'spawn_z': str(cfg['spawn']['z']),
        'spawn_yaw': str(cfg['spawn']['yaw']),
        'reset_world': 'true' if cfg.get('reset_world', False) else 'false',
        'bridge_camera_a': 'true' if cfg.get('bridge_camera_a', True) else 'false',
        'bridge_camera_b': 'true' if cfg.get('bridge_camera_b', False) else 'false',
        'bridge_camera_c': 'true' if cfg.get('bridge_camera_c', False) else 'false',
        'bridge_camera_d': 'true' if cfg.get('bridge_camera_d', False) else 'false',
    }
    if _as_bool(cfg.get('multicam_scheduled', False)) or _as_bool(cfg.get('multicam_belief', False)):
        # Both multi-camera front-ends need fresh RGB from EVERY camera the world profile
        # declares: the scheduled detector infers one view per cycle, and the batched
        # detector emits a batch only once every camera in its contract has contributed.
        # An unbridged camera therefore does not degrade the run, it silently starves it --
        # which is how camera E went missing when the profile grew from four cameras to
        # five. So the bridges are derived from the profile rather than listed by hand.
        for camera_id in cfg.get('profile_camera_ids', []):
            suffix = str(camera_id).removeprefix('camera_').lower()
            if suffix == 'a':
                sim_launch_arguments['bridge_camera_a'] = 'true'
            elif suffix in tuple('bcdefghijkl'):
                sim_launch_arguments[f'bridge_camera_{suffix}'] = 'true'

    bringup_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([sim_pkg, 'launch', 'bringup_sim.launch.py'])
        ),
        launch_arguments=sim_launch_arguments.items(),
    )

    perception_pkg = FindPackageShare('perception')
    tf_static = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([perception_pkg, 'launch', 'tf_static.launch.py'])
        ),
        launch_arguments=cfg['tf_args'].items(),
    )

    wait_for_odom = Node(
        package='sim',
        executable='wait_for_odom',
        name='wait_for_odom',
        output='screen',
        parameters=[{
            'topic': '/odom',
            'timeout_s': cfg['odom_wait_timeout_s'],
            'min_messages': cfg['odom_wait_min_messages'],
            'require_pose_match': cfg['odom_wait_require_pose_match'],
            'expected_x': 0.0,
            'expected_y': 0.0,
            'expected_yaw': 0.0,
            'position_tolerance': cfg['odom_wait_position_tolerance'],
            'yaw_tolerance': cfg['odom_wait_yaw_tolerance'],
        }],
    )

    command_noise_node = None
    if cfg.get('use_command_noise', True):
        command_noise_node = Node(
            package='sim',
            executable='actuation_noise_node',
            name='actuation_noise_node',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'enabled': True,
                'input_topic': '/cmd_vel_raw',
                'output_topic': cfg.get('command_noise_output_topic', '/cmd_vel'),
                'diagnostics_topic': '/cmd_vel_noise/diagnostics',
                'seed': cfg['seed'],
                'linear_slip_mean': cfg['command_noise_linear_slip_mean'],
                'linear_slip_std': cfg['command_noise_linear_slip_std'],
                'angular_slip_mean': cfg['command_noise_angular_slip_mean'],
                'angular_slip_std': cfg['command_noise_angular_slip_std'],
                'linear_additive_std': cfg['command_noise_linear_additive_std'],
                'angular_additive_std': cfg['command_noise_angular_additive_std'],
                'correlation_alpha': cfg['command_noise_correlation_alpha'],
                'linear_min': 0.0,
                'linear_max': cfg['v_max'],
                'angular_min': -1.0,
                'angular_max': 1.0,
            }],
        )

    encoder_noise_node = Node(
        package='sim',
        executable='encoder_noise_node',
        name='encoder_noise_node',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'enabled': True,
            'input_topic': '/odom',
            'output_topic': '/odom_noisy',
            'seed': cfg['seed'],
            'linear_slip_mean': cfg.get('encoder_noise_linear_slip_mean', _ENCODER_NOISE_LINEAR_SLIP_MEAN),
            'linear_slip_std': cfg.get('encoder_noise_linear_slip_std', _ENCODER_NOISE_LINEAR_SLIP_STD),
            'angular_slip_mean': cfg.get('encoder_noise_angular_slip_mean', _ENCODER_NOISE_ANGULAR_SLIP_MEAN),
            'angular_slip_std': cfg.get('encoder_noise_angular_slip_std', _ENCODER_NOISE_ANGULAR_SLIP_STD),
            'linear_additive_std': cfg.get(
                'encoder_noise_linear_additive_std',
                _ENCODER_NOISE_LINEAR_ADDITIVE_STD,
            ),
            'angular_additive_std': cfg.get(
                'encoder_noise_angular_additive_std',
                _ENCODER_NOISE_ANGULAR_ADDITIVE_STD,
            ),
            'correlation_alpha': cfg.get('encoder_noise_correlation_alpha', _ENCODER_NOISE_CORRELATION_ALPHA),
            'wheel_diameter_ratio_error': cfg.get(
                'encoder_wheel_diameter_ratio_error', _ENCODER_WHEEL_DIAMETER_RATIO_ERROR),
            'wheelbase_ratio': cfg.get('encoder_wheelbase_ratio', _ENCODER_WHEELBASE_RATIO),
            'wheel_separation_m': _ROBOT_WHEEL_SEPARATION_M,
            'input_source': 'ground_truth',
        }],
    )

    yolo_params = {
        # The detector was missing use_sim_time, so it ran on the wall clock while
        # the camera frame header.stamp + planner run on sim time. That made every
        # detector-side latency stamp (frame_age_at_publish_s, detector_total_latency_s,
        # yolo_receive/start/finish/publish_stamp) mix clock bases -> garbage ~1.78e9
        # epoch offsets in the logs. (The EKF was unaffected: it times off the sim
        # header.stamp, not the detector clock.)
        'use_sim_time': True,
        'pixel_noise_sigma': _SENSOR_PIXEL_NOISE_SIGMA,
        'seed': cfg['seed'],
        'model_path': cfg['yolo_model'],
        'device': cfg['yolo_device'],
        'image_size': cfg['yolo_imgsz'],
        'confidence_threshold': cfg['yolo_conf_threshold'],
        'iou_threshold': cfg['yolo_iou_threshold'],
        'class_name': cfg['yolo_target_class'],
        'class_id': cfg['yolo_class_id'],
        'use_masks': cfg['yolo_use_masks'],
        'mask_min_area': cfg['yolo_min_mask_area_px'],
        'mask_bottom_band_px': cfg['yolo_mask_bottom_band_px'],
        'min_bbox_area_px': cfg['yolo_min_bbox_area_px'],
        'debug_frame_dir': cfg.get('yolo_debug_frame_dir', ''),
        'debug_crop_dir': cfg.get('yolo_debug_crop_dir', ''),
        # TIMING fix knobs (see yolo_robot_detector_node).
        'use_torchscript': cfg.get('yolo_use_torchscript', False),
        'runtime_backend': cfg.get('yolo_runtime_backend', 'native'),
        'compiled_model_path': cfg.get('yolo_compiled_model', ''),
        'input_transport': cfg.get('yolo_input_transport', 'ros'),
        'runtime_trace_period_s': cfg.get('yolo_runtime_trace_period_s', 0.0),
        'warmup_iters': int(cfg.get('yolo_warmup_iters', 3)),
        'inference_in_callback': cfg.get('yolo_inference_in_callback', True),
    }
    # CPU-affinity isolation for the detector. In-run the YOLO forward inflates
    # 17ms(idle) -> ~168ms because the CUDA kernel-launch thread gets starved by
    # the EFE solver's BLAS threads (GPU util stays ~0% -> launch-bound, not
    # compute-bound). Pinning the detector to dedicated cores (e.g. "10,11") while
    # the rest of the run is confined elsewhere keeps the launch thread on-core.
    # Env-gated so it is a no-op unless DETECTOR_CPU_AFFINITY is set.
    _det_affinity = os.environ.get('DETECTOR_CPU_AFFINITY', '').strip()
    _det_prefix = f'taskset -c {_det_affinity}' if _det_affinity else None
    perception_node = Node(
        package='perception',
        executable='yolo_robot_detector_node',
        name='yolo_robot_detector_node',
        output='screen',
        parameters=[yolo_params],
        prefix=_det_prefix,
    )

    pixel_params = {
        'use_sim_time': True,
        'frame_id': 'map_bev',
        'pixel_noise_sigma': 0.0,
        'heading_pixel_noise_sigma': _SENSOR_PIXEL_NOISE_SIGMA,
        'transform_noise_sigma': 0.0,
        'use_odom_heading_fallback': True,
        'odom_topic': odom_topic,
        'odom_heading_timeout_s': cfg['odom_heading_timeout_s'],
        'odom_heading_sigma_rad': 0.08,
        'odom_yaw_offset_rad': odom_yaw_offset_rad,
        'infer_yaw_from_motion': False,
        'seed': cfg['seed'],
        'diagnostics_match_tolerance_s': 1e-3,
        # BEV calibration MUST be applied at the projection node. These were
        # previously only passed to the logger/planner, so the state node ran with
        # the default 0.0 and the south-bias correction never took effect.
        'bev_y_calibration_offset_m': cfg.get('bev_y_calibration_offset_m', 0.0),
        'bev_affine_calibration': cfg.get('bev_affine_calibration', ''),
        **cfg['camera_params'],
    }
    pixel_to_bev = Node(
        package='state',
        executable='pixel_to_bev_state_node',
        name='pixel_to_bev_state_node',
        output='screen',
        parameters=[pixel_params],
    )

    mission_params = {
        'use_sim_time': True,
        'frame_id': 'map_bev',
        'goal_x': cfg['goal_x'],
        'goal_y': cfg['goal_y'],
        'waypoints_json': cfg.get('waypoints_json', ''),
        'wait_for_belief_before_first_goal': cfg.get(
            'wait_for_belief_before_first_goal', False
        ),
        'initial_belief_max_sigma_m': cfg.get(
            'initial_belief_max_sigma_m', 0.0
        ),
        'operational_belief_timeout_s': cfg.get(
            'operational_belief_timeout_s', 0.5
        ),
    }
    mission_node = Node(
        package='experiments',
        executable='goal_mission_node',
        name='goal_mission_node',
        output='screen',
        parameters=[mission_params],
    )

    goal_marker_node = Node(
        package='experiments',
        executable='goal_marker_node',
        name='goal_marker_node',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'marker_topic': '/goal_marker',
            'marker_ns': 'goal',
            'scale': 0.35,
            'z': 0.08,
            'color_r': 0.0,
            'color_g': 0.35,
            'color_b': 1.0,
            'color_a': 1.0,
        }],
    )

    logger_node = None
    if cfg.get('enable_logging', True):
        logger_node = Node(
            package='experiments',
            executable='experiment_logger',
            name='experiment_logger',
            output='screen',
            on_exit=[Shutdown(reason='experiment_logger exited')],
            parameters=[{
                'use_sim_time': True,
                'log_dir': cfg['log_dir'],
                'seed': cfg['seed'],
                # Three settings that change what a drive IS and were reaching the
                # planner without reaching the manifest: how fast the robot may go
                # (so how much odometry drift accumulates between corrections and how
                # far it travels during an outage), whether the belief is predicted
                # from odometry or from the commanded velocity, and which local
                # controller tracks the route.
                'v_max': cfg['v_max'],
                'use_odom_for_predict': cfg['use_odom_for_predict'],
                'local_controller_type': cfg['local_controller_type'],
                'method': cfg['comparison_method_id'] or cfg['planner'],
                'perception_backend': cfg['perception_backend'],
                'world': cfg['world'],
                'world_sdf_path': str(cfg['world_path']),
                'task': cfg['task'].get('name', cfg['task_name'] or ''),
                'planner': cfg['planner'],
                'state_source_x': state_sources['state_source_x'],
                'state_source_y': state_sources['state_source_y'],
                'state_source_theta': state_sources['state_source_theta'],
                'state_estimator_mode': state_sources['state_estimator_mode'],
                'state_correction_mode': cfg.get('state_correction_mode', 'fused'),
                # The camera-manager settings that define the arm, from the SAME
                # function that configures the manager. Recorded in the manifest so a
                # result's arm identity does not live only in its directory name.
                'manager_settings_json': json.dumps(
                    dict(manager_arm_settings(cfg),
                         manager_active=_as_bool(cfg.get('multicam_belief', False))),
                    sort_keys=True),
                'campaign_config_path': cfg.get('campaign_config_path', ''),
                'outcome_journal_path': cfg.get('outcome_journal_path', ''),
                'manager_outcome_journal_path': cfg.get(
                    'manager_outcome_journal_path', ''),
                'use_pixel_correction': cfg['use_pixel_correction'],
                'pixel_timeout_s': cfg['pixel_timeout_s'],
                'use_ambiguity': cfg['use_ambiguity'],
                'use_obs_risk': cfg['use_obs_risk'],
                'use_visibility_model': cfg['use_visibility_model'],
                'visibility_artifact_path': cfg['visibility_artifact_path'],
                'camera_network_artifact_path': cfg.get('camera_network_artifact_path', ''),
                'camera_network_expected_sha256': cfg.get(
                    'camera_network_expected_sha256', ''),
                'camera_network_expected_source_hashes_json': cfg.get(
                    'camera_network_expected_source_hashes_json', ''),
                'camera_network_camera_ids': cfg.get('camera_network_camera_ids', ''),
                'camera_network_active_camera_ids': cfg.get(
                    'camera_network_active_camera_ids', ''),
                'camera_network_objective': cfg.get(
                    'camera_network_objective', 'legacy_pixel_chart'),
                'network_goal_std_m': cfg.get('network_goal_std_m', 0.10),
            'network_goal_std_start_m': cfg.get('network_goal_std_start_m', -1.0),
            'kouw_et1_ambiguity': cfg.get('kouw_et1_ambiguity', True),
                'network_goal_std_start_m': cfg.get('network_goal_std_start_m', -1.0),
            'kouw_et1_ambiguity': cfg.get('kouw_et1_ambiguity', True),
                'kouw_et1_ambiguity': cfg.get('kouw_et1_ambiguity', True),
                'camera_network_updates_per_step': cfg.get(
                    'camera_network_updates_per_step', 1),
                'optimizer_control_block_steps': cfg.get(
                    'optimizer_control_block_steps', 1),
                'risk_weight_obs': cfg['risk_weight_obs'],
                'ambiguity_weight': cfg['ambiguity_weight'],
                'goal_sigma_uv': cfg['goal_sigma_uv'],
                'r_visible_uv': cfg['r_visible_uv'],
                'r_miss_uv': cfg['r_miss_uv'],
                'visibility_sigma_kappa': cfg['visibility_sigma_kappa'],
                'goal_prior_u_std_start': cfg['goal_prior_u_std_start'],
                'goal_prior_v_std_start': cfg['goal_prior_v_std_start'],
                'goal_prior_u_std_final': cfg['goal_prior_u_std_final'],
                'goal_prior_v_std_final': cfg['goal_prior_v_std_final'],
                'goal_tightening_power': cfg['goal_tightening_power'],
                'goal_progress_n_steps': cfg['goal_progress_n_steps'],
                'observation_risk_scale': cfg['observation_risk_scale'],
                'ambiguity_term_scale': cfg['ambiguity_term_scale'],
                'discount_gamma': cfg['discount_gamma'],
                'v_max': cfg['v_max'],
                'visibility_target_height_m': cfg['visibility_target_height_m'],
                'visibility_geometry_json': cfg['visibility_geometry_json'],
                'collision_geometry_json': cfg['collision_geometry_json'],
                'perception_use_geometry_occlusion': cfg['perception_use_geometry_occlusion'],
                'use_nogo_cost': cfg.get('resolved_use_nogo_cost', False),
                'nogo_penalty_type': cfg['nogo_penalty_type'],
                'nogo_safe_distance': cfg['nogo_safe_distance'],
                'nogo_logbarrier_eps': cfg['nogo_logbarrier_eps'],
                'nogo_warning_band': cfg['nogo_warning_band'],
                'nogo_near_weight': cfg['nogo_near_weight'],
                'use_belief_nogo_cost': cfg['use_belief_nogo_cost'],
                'nogo_belief_kappa': cfg['nogo_belief_kappa'],
                'use_hit_miss_mixture': cfg.get('use_hit_miss_mixture', False),
                'nogo_mode': cfg.get('nogo_mode', 'keep_out'),
                'yolo_model': cfg['yolo_model'],
                'yolo_compiled_model': cfg.get('yolo_compiled_model', ''),
                'yolo_device': cfg['yolo_device'],
                'yolo_imgsz': cfg['yolo_imgsz'],
                'yolo_conf_threshold': cfg['yolo_conf_threshold'],
                'yolo_predict_conf_floor': cfg['yolo_predict_conf_floor'],
                'yolo_iou_threshold': cfg['yolo_iou_threshold'],
                'yolo_target_class': cfg['yolo_target_class'],
                'yolo_class_id': cfg['yolo_class_id'],
                'yolo_use_masks': cfg['yolo_use_masks'],
                'yolo_min_mask_area_px': cfg['yolo_min_mask_area_px'],
                'yolo_mask_bottom_band_px': cfg['yolo_mask_bottom_band_px'],
                'yolo_max_batch_stamp_skew_s': cfg['yolo_max_batch_stamp_skew_s'],
                'show_pose_markers': False,
                'diagnostics_match_tolerance_s': 1e-3,
                'bev_y_calibration_offset_m': cfg['bev_y_calibration_offset_m'],
                'bev_affine_calibration': cfg.get('bev_affine_calibration', ''),
                'pixel_correction_nis_threshold': cfg['pixel_correction_nis_threshold'],
                'state_reanchor_m': cfg['state_reanchor_m'],
                'state_max_predict_dt_s': cfg['state_max_predict_dt_s'],
                'state_reject_inflate_m2': cfg['state_reject_inflate_m2'],
                'stale_belief_inflate_m2_per_s': cfg['stale_belief_inflate_m2_per_s'],
                'stale_belief_inflate_cap_m2': cfg['stale_belief_inflate_cap_m2'],
                'require_state_correction_envelope': cfg['require_state_correction_envelope'],
                'world_profiles_path': cfg['world_profiles_path'],
                'tasks_yaml': cfg['tasks_yaml'],
                'auto_stop_on_goal': cfg['auto_stop_on_goal'],
                'goal_success_radius': cfg['goal_success_radius'],
                'goal_success_hold_s': cfg['goal_success_hold_s'],
                'operational_belief_timeout_s': cfg['operational_belief_timeout_s'],
                'goal_stable_radius': cfg['goal_stable_radius'],
                'goal_stable_hold_s': cfg['goal_stable_hold_s'],
                'goal_stable_max_displacement_m': cfg['goal_stable_max_displacement_m'],
                'plan_rate': cfg['plan_rate'],
                'cmd_publish_rate': cfg['cmd_publish_rate'],
                'horizon': cfg['horizon'],
                'dt': cfg['dt'],
                'control_weight': cfg['control_weight'],
                'process_noise_xy': cfg['process_noise_xy'],
                'process_noise_theta': cfg['process_noise_theta'],
                'process_noise_model': cfg['process_noise_model'],
                'optimizer_maxiter': cfg['optimizer_maxiter'],
                'optimizer_maxfun': cfg['optimizer_maxfun'],
                'optimizer_ftol': cfg['optimizer_ftol'],
                'optimizer_gtol': cfg['optimizer_gtol'],
                'optimizer_warm_start': cfg['optimizer_warm_start'],
                'optimizer_multistart': cfg['optimizer_multistart'],
                'optimizer_multistart_include_direct': cfg['optimizer_multistart_include_direct'],
                'optimizer_initial_routes_json': cfg['optimizer_initial_routes_json'],
                'optimizer_terminal_goal_tolerance_m': cfg['optimizer_terminal_goal_tolerance_m'],
                'optimizer_route_seed_mode': cfg['optimizer_route_seed_mode'],
                'use_hierarchical': cfg.get('use_hierarchical', False),
                'global_planner_mode': cfg.get('global_planner_mode', 'efe'),
                'preselected_route_json': cfg.get('preselected_route_json', ''),
                'preselected_route_sha256': cfg.get('preselected_route_sha256', ''),
                'preselected_route_source_path': cfg.get(
                    'preselected_route_source_path', ''
                ),
                'preselected_route_source_sha256': cfg.get(
                    'preselected_route_source_sha256', ''
                ),
                'preselected_route_clearance_m': cfg.get(
                    'preselected_route_clearance_m', 0.25
                ),
                'preselected_route_endpoint_tolerance_m': cfg.get(
                    'preselected_route_endpoint_tolerance_m', 0.25
                ),
                'preselected_route_sample_step_m': cfg.get(
                    'preselected_route_sample_step_m', 0.04
                ),
                'preselected_route_validation_json': cfg.get(
                    'preselected_route_validation_json', ''
                ),
                'global_horizon': cfg.get('global_horizon', 60),
                'global_dt': cfg.get('global_dt', 0.0),
                'local_horizon': cfg.get('local_horizon', 12),
                'local_plan_rate': cfg.get('local_plan_rate', 4.0),
                'local_optimizer_maxiter': cfg.get('local_optimizer_maxiter', 60),
                'global_use_ambiguity': cfg.get('global_use_ambiguity', True),
                'local_use_ambiguity': cfg.get('local_use_ambiguity', False),
                'local_use_obs_risk': cfg.get('local_use_obs_risk', True),
                'global_optimizer_multistart': cfg.get('global_optimizer_multistart', True),
                'local_optimizer_multistart': cfg.get('local_optimizer_multistart', True),
                'local_use_visibility_model': cfg.get('local_use_visibility_model', False),
                'local_use_belief_nogo_cost': cfg.get('local_use_belief_nogo_cost', False),
                'local_nogo_penalty_type': cfg.get('local_nogo_penalty_type', ''),
                'local_nogo_safe_distance': cfg.get('local_nogo_safe_distance', -1.0),
                # Record the resolved local goal prior, not the logger's -1
                # sentinel.  The planner already receives these values below;
                # omitting them here made otherwise valid runs fail campaign
                # resume/config-parity checks.
                'local_goal_prior_u_std_start': cfg.get('local_goal_prior_u_std_start', -1.0),
                'local_goal_prior_v_std_start': cfg.get('local_goal_prior_v_std_start', -1.0),
                'local_goal_prior_u_std_final': cfg.get('local_goal_prior_u_std_final', -1.0),
                'local_goal_prior_v_std_final': cfg.get('local_goal_prior_v_std_final', -1.0),
                'waypoint_spacing_m': cfg.get('waypoint_spacing_m', 0.2),
                'waypoint_arrival_radius_m': cfg.get('waypoint_arrival_radius_m', 0.1),
                'local_replan_min_remaining_s': cfg.get('local_replan_min_remaining_s', 0.0),
                'local_replan_on_waypoint_change': cfg.get('local_replan_on_waypoint_change', False),
                'latency_compensate_plan_handoff': cfg.get('latency_compensate_plan_handoff', False),
                'simple_tracker_yaw_gate_rad': cfg.get('simple_tracker_yaw_gate_rad', 0.65),
                'ff_fb_turn_rate_limit_rad_s': cfg.get('ff_fb_turn_rate_limit_rad_s', 0.80),
                'ff_fb_corner_crawl_speed_mps': cfg.get('ff_fb_corner_crawl_speed_mps', 0.18),
                'ff_fb_pivot_heading_error_rad': cfg.get('ff_fb_pivot_heading_error_rad', 2.60),
                'heading_update_mode': cfg['heading_update_mode'],
                'local_controller_type': cfg['local_controller_type'],
                'run_timeout_after_first_cmd_s': cfg['run_timeout_after_first_cmd_s'],
                'first_cmd_linear_eps': cfg['first_cmd_linear_eps'],
                'first_cmd_angular_eps': cfg['first_cmd_angular_eps'],
                'stuck_window_s': cfg['stuck_window_s'],
                'stuck_max_displacement_m': cfg['stuck_max_displacement_m'],
                'stuck_max_goal_improvement_m': cfg['stuck_max_goal_improvement_m'],
                'stuck_cmd_fraction_min': cfg['stuck_cmd_fraction_min'],
                'stuck_idle_cmd_fraction_max': cfg['stuck_idle_cmd_fraction_max'],
                'robot_collision_radius_m': cfg['robot_collision_radius_m'],
                'robot_length_m': cfg.get('robot_length_m', 0.8),
                'robot_width_m': cfg.get('robot_width_m', 0.55),
                'use_command_noise': cfg['use_command_noise'],
                'use_encoder_noise': cfg['use_encoder_noise'],
                'use_odom_for_predict': cfg['use_odom_for_predict'],
                'odom_topic': odom_topic,
                'command_noise_linear_slip_mean': cfg['command_noise_linear_slip_mean'],
                'command_noise_linear_slip_std': cfg['command_noise_linear_slip_std'],
                'command_noise_angular_slip_mean': cfg['command_noise_angular_slip_mean'],
                'command_noise_angular_slip_std': cfg['command_noise_angular_slip_std'],
                'command_noise_linear_additive_std': cfg['command_noise_linear_additive_std'],
                'command_noise_angular_additive_std': cfg['command_noise_angular_additive_std'],
                'command_noise_correlation_alpha': cfg['command_noise_correlation_alpha'],
                'encoder_noise_linear_slip_mean': cfg.get(
                    'encoder_noise_linear_slip_mean',
                    _ENCODER_NOISE_LINEAR_SLIP_MEAN,
                ),
                'encoder_noise_linear_slip_std': cfg.get(
                    'encoder_noise_linear_slip_std',
                    _ENCODER_NOISE_LINEAR_SLIP_STD,
                ),
                'encoder_noise_angular_slip_mean': cfg.get(
                    'encoder_noise_angular_slip_mean',
                    _ENCODER_NOISE_ANGULAR_SLIP_MEAN,
                ),
                'encoder_noise_angular_slip_std': cfg.get(
                    'encoder_noise_angular_slip_std',
                    _ENCODER_NOISE_ANGULAR_SLIP_STD,
                ),
                'encoder_noise_linear_additive_std': cfg.get(
                    'encoder_noise_linear_additive_std',
                    _ENCODER_NOISE_LINEAR_ADDITIVE_STD,
                ),
                'encoder_noise_angular_additive_std': cfg.get(
                    'encoder_noise_angular_additive_std',
                    _ENCODER_NOISE_ANGULAR_ADDITIVE_STD,
                ),
                'encoder_wheel_diameter_ratio_error': cfg.get(
                    'encoder_wheel_diameter_ratio_error', _ENCODER_WHEEL_DIAMETER_RATIO_ERROR),
                'encoder_wheelbase_ratio': cfg.get('encoder_wheelbase_ratio', _ENCODER_WHEELBASE_RATIO),
                'process_noise_model': cfg['process_noise_model'],
                'encoder_noise_correlation_alpha': cfg.get(
                    'encoder_noise_correlation_alpha',
                    _ENCODER_NOISE_CORRELATION_ALPHA,
                ),
                **cfg['camera_params'],
            }],
        )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', cfg['rviz_config']],
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    lockstep_scheduler = None
    if cfg.get('lockstep', False):
        lockstep_scheduler = Node(
            package='sim_command_guard',
            executable='lockstep_scheduler',
            name='lockstep_scheduler',
            output='screen',
            parameters=[{
                'world_name': cfg['profile']['world_name'],
                'control_step_iterations': cfg['lockstep_control_step_iterations'],
                'camera_every_control_steps': cfg['lockstep_camera_every_control_steps'],
                'max_control_steps': cfg['lockstep_max_control_steps'],
                # A planner that falls behind pauses the world instead of ageing.
                'wait_for_planner': True,
                # GPU model load and warm-up can exceed 30 s on this laptop. This
                # timeout also catches genuinely wedged runtime barriers once active.
                'barrier_timeout_s': 180.0,
            }],
            on_exit=[Shutdown(reason='lockstep scheduler exited')],
        )

    return {
        'bringup_sim': bringup_sim,
        'tf_static': tf_static,
        'wait_for_odom': wait_for_odom,
        'command_noise_node': command_noise_node,
        'encoder_noise_node': encoder_noise_node,
        'perception_node': perception_node,
        'pixel_to_bev': pixel_to_bev,
        'mission_node': mission_node,
        'goal_marker_node': goal_marker_node,
        'logger_node': logger_node,
        'rviz': rviz,
        'lockstep_scheduler': lockstep_scheduler,
    }


def manager_arm_settings(cfg: Dict[str, object]) -> Dict[str, object]:
    """The camera-manager settings that DEFINE an experiment arm, as plain scalars.

    Read twice from this one place: to configure ``camera_manager_node``, and to
    record in the run manifest. One source, so the manifest cannot disagree with what
    actually ran.

    Before this existed the manifest carried 152 keys and not one manager setting, so
    the fusion rule, the observation model, the decision rate and the timestamp
    compensation of every F1-F4 / O1-O2 result lived only in a directory name -- and
    the campaign runner's reuse check silently skips any key the manifest lacks, so a
    re-run could inherit a directory from a different arm.

    JSON-serialisable on purpose: no launch Substitutions, no camera-model paths.
    """

    settings = {
        'manager_camera_ids': str(cfg.get('manager_camera_ids', '') or ''),
        'manager_decision_rate_hz': float(cfg.get('manager_decision_rate_hz', 5.0)),
        'manager_require_gp_artifacts': _as_bool(cfg.get('manager_require_gp_artifacts', True)),
        'manager_fusion_mode': _as_bool(cfg.get('manager_fusion_mode', True)),
        'manager_publish_map_observations': _as_bool(
            cfg.get('manager_publish_map_observations',
                    str(cfg.get('state_correction_mode', 'fused')) == 'per_camera')
        ),
        'manager_fusion_disagreement_gate_m': float(
            cfg.get('manager_fusion_disagreement_gate_m', 0.6)),
        'manager_require_source_batch_id': _as_bool(
            cfg.get('manager_require_source_batch_id', True)),
        'manager_bootstrap_min_cameras': int(
            cfg.get('manager_bootstrap_min_cameras', 2)),
        'manager_bootstrap_max_disagreement_m': float(
            cfg.get('manager_bootstrap_max_disagreement_m', 0.30)),
        'manager_use_task_start_as_bootstrap_prior': _as_bool(
            cfg.get('manager_use_task_start_as_bootstrap_prior', False)),
        'manager_bootstrap_prior_counts_as_support': _as_bool(
            cfg.get('manager_bootstrap_prior_counts_as_support', False)),
        'manager_assume_initial_belief_anchor': _as_bool(cfg.get('initial_belief_from_task_start', False)),
        'manager_fusion_max_timestamp_spread_s': float(
            cfg.get('manager_fusion_max_timestamp_spread_s', 0.05)),
        'manager_covariance_profile': str(
            cfg.get('manager_covariance_profile', 'commissioned_sigma_px')),
        'manager_commissioned_calibration_path': str(
            cfg.get('manager_commissioned_calibration_path', '') or ''),
        'manager_commissioned_world_covariance_path': str(
            cfg.get('manager_commissioned_world_covariance_path', '') or ''),
        'manager_commissioned_sigma_px': float(cfg.get('manager_commissioned_sigma_px', 0.0)),
        'manager_commissioned_per_camera_sigma': _as_bool(
            cfg.get('manager_commissioned_per_camera_sigma', False)),
        'manager_fusion_common_mode_std_m': float(
            cfg.get('manager_fusion_common_mode_std_m', 0.0)),
        'manager_fusion_rule': str(cfg.get('manager_fusion_rule', DEFAULT_MANAGER_FUSION_RULE)),
        'manager_correction_timestamp_compensation': _as_bool(
            cfg.get('manager_correction_timestamp_compensation', False)),
        'manager_sensor_gate_config_path': str(
            cfg.get('manager_sensor_gate_config_path', '') or ''),
        'manager_correction_residual_interval_s': float(
            cfg.get('manager_correction_residual_interval_s', 0.05)),
        'manager_correction_propagation_drift_std': float(
            cfg.get('manager_correction_propagation_drift_std', 0.05)),
        'manager_observation_model': str(cfg.get('manager_observation_model', 'raw_box')),
        'manager_availability_model_path': str(
            cfg.get('manager_availability_model_path', '') or ''),
        'manager_availability_model_expected_sha256': str(
            cfg.get('manager_availability_model_expected_sha256', '') or ''),
        # Empty learned paths are valid for the raw-box commissioning condition.
        'manager_learned_correction_path': str(
            cfg.get('manager_learned_correction_path', '')),
        'manager_visibility_sensor_model_path': str(
            cfg.get('manager_visibility_sensor_model_path', '')),
        'manager_visibility_sensor_model_expected_sha256': str(
            cfg.get('manager_visibility_sensor_model_expected_sha256', '')),
        'manager_perception_sensor_model_path': str(
            cfg.get('manager_perception_sensor_model_path', '')),
        'manager_perception_sensor_model_expected_sha256': str(
            cfg.get('manager_perception_sensor_model_expected_sha256', '')),
        'manager_min_spatial_trust': float(cfg.get('manager_min_spatial_trust', 0.15)),
        'manager_max_measurement_age_s': float(
            cfg.get('manager_max_measurement_age_s', cfg['pixel_timeout_s'])),
        'manager_age_decay_s': float(cfg.get('manager_age_decay_s', cfg['pixel_timeout_s'])),
        'manager_min_association_confidence': float(
            cfg.get('manager_min_association_confidence', 0.30)),
        'manager_required_consecutive_better_frames': int(
            cfg.get('manager_required_consecutive_better_frames', 1)),
        'manager_max_cross_camera_disagreement_m': float(
            cfg.get('manager_max_cross_camera_disagreement_m', 1.0)),
        'manager_require_consistency_when_source_available': _as_bool(
            cfg.get('manager_require_consistency_when_source_available', False)),
        'manager_bias_floor_along_slope_m_per_m': float(
            cfg.get('manager_bias_floor_along_slope_m_per_m', 0.0)),
        # Both zero means the bias floor is off, which is the default. A single
        # positive slope is a floor with a zero axis; the manager refuses it at
        # start-up rather than dying at the first fusion.
        'manager_bias_floor_across_slope_m_per_m': float(
            cfg.get('manager_bias_floor_across_slope_m_per_m', 0.0)),
    }
    spawn = cfg.get('spawn') or {}
    settings['manager_bootstrap_prior_xyyaw'] = [
        float(spawn.get('x', 0.0)), float(spawn.get('y', 0.0)),
        float(spawn.get('yaw', 0.0)),
    ]
    return settings


#: manager_arm_settings key -> the camera_manager_node parameter it sets.
_MANAGER_PARAM_NAMES = {
    'manager_correction_propagation_drift_std': 'correction_propagation_drift_std_m_per_s',
}


def _manager_node_parameters(cfg: Dict[str, object]) -> Dict[str, object]:
    """The arm settings under the node's own parameter names."""

    out = {}
    for key, value in manager_arm_settings(cfg).items():
        # The node consumes a resolved string-array roster assembled from the
        # world profile above. The manifest retains the user's comma-separated
        # arm setting, but forwarding that scalar here would overwrite the
        # resolved ``camera_ids`` list in the Node parameter dictionary.
        if key == 'manager_camera_ids':
            continue
        name = _MANAGER_PARAM_NAMES.get(key)
        if name is None:
            assert key.startswith('manager_'), key
            name = key[len('manager_'):]
        out[name] = value
    learned_path = str(cfg.get('manager_learned_correction_path', '') or '').strip()
    visibility_path = str(
        cfg.get('manager_visibility_sensor_model_path', '') or ''
    ).strip()
    perception_path = str(
        cfg.get('manager_perception_sensor_model_path', '') or ''
    ).strip()
    availability_path = str(cfg.get('manager_availability_model_path', '') or '').strip()
    sensor_gate_path = str(cfg.get('manager_sensor_gate_config_path', '') or '').strip()
    world_covariance_path = str(
        cfg.get('manager_commissioned_world_covariance_path', '') or ''
    ).strip()
    if learned_path:
        learned = Path(learned_path).expanduser().resolve()
        out['learned_correction_path'] = str(learned)
        out['learned_correction_expected_sha256'] = hashlib.sha256(
            learned.read_bytes()
        ).hexdigest()
    if visibility_path:
        visibility = Path(visibility_path).expanduser().resolve()
        out['visibility_sensor_model_path'] = str(visibility)
        actual = hashlib.sha256(visibility.read_bytes()).hexdigest()
        declared = str(
            cfg.get('manager_visibility_sensor_model_expected_sha256', '') or ''
        )
        if declared and declared != actual:
            raise RuntimeError('visibility-residual sensor model differs from configured SHA-256')
        out['visibility_sensor_model_expected_sha256'] = declared or actual
    if perception_path:
        perception = Path(perception_path).expanduser().resolve()
        out['perception_sensor_model_path'] = str(perception)
        actual = hashlib.sha256(perception.read_bytes()).hexdigest()
        declared = str(
            cfg.get('manager_perception_sensor_model_expected_sha256', '') or ''
        )
        if declared and declared != actual:
            raise RuntimeError('perception sensor model differs from configured SHA-256')
        out['perception_sensor_model_expected_sha256'] = declared or actual
    if availability_path:
        availability = Path(availability_path).expanduser().resolve()
        out['availability_model_path'] = str(availability)
        actual = hashlib.sha256(availability.read_bytes()).hexdigest()
        declared = str(cfg.get('manager_availability_model_expected_sha256', '') or '')
        if declared and declared != actual:
            raise RuntimeError('Stage-08 availability artifact differs from configured SHA-256')
        out['availability_model_expected_sha256'] = declared or actual
    if sensor_gate_path:
        sensor_gate = Path(sensor_gate_path).expanduser().resolve()
        out['sensor_gate_config_path'] = str(sensor_gate)
        out['sensor_gate_config_expected_sha256'] = hashlib.sha256(
            sensor_gate.read_bytes()
        ).hexdigest()
    if world_covariance_path:
        covariance = Path(world_covariance_path).expanduser().resolve()
        out['commissioned_world_covariance_path'] = str(covariance)
        out['commissioned_world_covariance_expected_sha256'] = hashlib.sha256(
            covariance.read_bytes()
        ).hexdigest()
    return out


def _multicam_perception_nodes(cfg: Dict[str, object]) -> List[object]:
    if str(cfg.get('yolo_runtime_backend', 'native')) != 'native':
        raise RuntimeError(
            'paper multicam evidence requires native strict batching; asynchronous '
            'TorchScript batches do not have the all-camera batch identity contract'
        )
    if str(cfg.get('yolo_input_transport', 'ros')) != 'ros':
        raise RuntimeError(
            'paper multicam evidence requires the strict ROS camera batch transport'
        )
    """Multi-camera belief front-end (multicam_belief mode).

    Replaces the single-camera ``yolo_robot_detector_node`` + ``pixel_to_bev``
    with the batched multicamera detector and the ``camera_manager`` running in
    ``authority=active`` mode. The manager selects the best available camera
    per frame and republishes the world-frame correction to ``/state/bev`` --
    the same topic the planner's ``_state_cb`` consumes -- so no planner change
    is needed. Every camera's detection is projected with its own calibration,
    so the fused correction is genuinely multi-camera.
    """
    # The camera set is the perception layer's own contract, not a literal repeated here.
    # It was ('camera_A'..'camera_D') while the batch runtime was four-camera; the runtime
    # contract is now v2 and carries warehouse_v2's five wall cameras, so a stale literal
    # here refused every warehouse_v2 arm before Gazebo started.
    from perception.core.four_camera_runtime_contract import (  # noqa: PLC0415
        BATCHED_CAMERA_ORDER,
    )
    contract_camera_ids = tuple(BATCHED_CAMERA_ORDER)
    detector_calibration_world = _DETECTOR_CALIBRATION_WORLD_BY_WORLD.get(
        str(cfg['world']), str(cfg['world']).replace('.world.sdf', '')
    )
    profile_camera_ids = tuple(cfg.get('profile_camera_ids', ()))
    if profile_camera_ids != contract_camera_ids:
        raise RuntimeError(
            "multicam_belief uses the batched detector's runtime contract "
            f"{contract_camera_ids}, but the world profile declares {profile_camera_ids}. "
            "Refusing to silently omit cameras: the batch is emitted only when every "
            "camera in the contract has contributed a frame, so a mismatch would drop a "
            "camera's evidence without saying so."
        )
    sim_pkg = FindPackageShare('sim')
    world_sdf = PathJoinSubstitution([sim_pkg, 'gazebo_worlds', 'worlds', cfg['world']])
    batched = Node(
        package='perception',
        executable='batched_four_camera_yolo_node',
        name='batched_four_camera_yolo',
        output='screen',
        on_exit=[Shutdown(reason='batched multicamera detector exited')],
        parameters=[{
            'use_sim_time': True,
            # Every published observation carries a calibration identity of the form
            # "<world>_<camera>", and the node refuses to run without the world half. This
            # launch path never set it, so the batched detector died on start-up; the
            # commissioning launch derived it the same way.
            'calibration_world': detector_calibration_world,
            'model_path': cfg['yolo_model'],
            'runtime_backend': cfg.get('yolo_runtime_backend', 'native'),
            'compiled_model_path': cfg.get('yolo_compiled_model', ''),
            'device': cfg['yolo_device'],
            'image_size': cfg['yolo_imgsz'],
            'confidence_threshold': cfg['yolo_conf_threshold'],
            'iou_threshold': cfg['yolo_iou_threshold'],
            'class_name': cfg['yolo_target_class'],
            'class_id': cfg['yolo_class_id'],
            'use_masks': cfg['yolo_use_masks'],
            'mask_min_area': cfg['yolo_min_mask_area_px'],
            'mask_bottom_band_px': cfg['yolo_mask_bottom_band_px'],
            'min_bbox_area_px': cfg['yolo_min_bbox_area_px'],
            'debug_crop_dir': cfg.get('yolo_debug_crop_dir', ''),
            # Prune sub-threshold anchors before NMS+mask post-processing. At the
            # 0.25 reporting threshold this changes no reported detection but cuts
            # batch inference ~140 ms -> ~39 ms, which frees the P2000 to render
            # the four oblique cameras faster: measured 3.3 Hz -> 4.9 Hz per
            # camera (batched, all registered cameras in lockstep). See scheduled note.
            'predict_conf_floor': cfg['yolo_predict_conf_floor'],
            'warmup_iters': int(cfg.get('yolo_warmup_iters', 3)),
            # Capture stamp, not callback arrival time, defines a round. Keep
            # this below the 0.20 s camera period so round N and N+1 cannot merge.
            'max_batch_stamp_skew_s': cfg['yolo_max_batch_stamp_skew_s'],
            'max_pending_wall_s': 0.50,
            'synchronization_mode': 'strict',
            'input_transport': cfg.get('yolo_input_transport', 'ros'),
            # 0 disables. Non-zero periodically reports frames per camera, batcher
            # decisions and which cameras each unfinished round still waits on --
            # the only way to see a detector that has stopped producing batches.
            'runtime_trace_period_s': float(cfg.get('yolo_runtime_trace_period_s', 0.0)),
            'camera_observation_r_visible_uv': float(cfg.get('r_visible_uv', 2.5)),
            'camera_observation_r_miss_uv': float(cfg.get('r_miss_uv', 40.0)),
            'outcome_journal_path': str(cfg.get('outcome_journal_path', '') or ''),
        }],
    )
    # Which cameras the fusion manager uses. Empty -> all registered cameras; restrict
    # to one (e.g. "camera_A") for a single-camera localization baseline in the
    # fused-vs-single comparison. camera_model_includes must stay aligned 1:1 with
    # camera_ids (the manager asserts this), so derive it from the same selection.
    # Derived from the world profile, which already carries camera_ids aligned 1:1 with
    # camera_model_includes and is validated above. A hard-coded map here silently lacked
    # camera_E and would have dropped the fifth camera from the manager.
    _model_include_by_id = dict(zip(
        cfg.get('profile_camera_ids', ()), cfg.get('profile_camera_model_includes', ())))
    _mc_camera_ids = (
        [c.strip() for c in str(cfg.get('manager_camera_ids', '') or '').split(',') if c.strip()]
        or list(cfg.get('profile_camera_ids', ()))
    )
    _missing = [c for c in _mc_camera_ids if c not in _model_include_by_id]
    if _missing:
        raise RuntimeError(
            f"manager_camera_ids asks for {_missing}, which the world profile for "
            f"{cfg['world']} does not declare. Fix the profile or the request; do not "
            "guess a model include name."
        )
    _mc_model_includes = [_model_include_by_id[c] for c in _mc_camera_ids]
    _manager_odom_topic = str(cfg.get('odom_topic') or '/odom_noisy')
    if not _as_bool(cfg.get('use_encoder_noise', True)) and _manager_odom_topic == '/odom_noisy':
        _manager_odom_topic = '/odom'
    manager = Node(
        package='reliability',
        executable='camera_manager_node',
        name='camera_manager_active',
        output='screen',
        on_exit=[Shutdown(reason='camera manager exited')],
        parameters=[{
            'use_sim_time': True,
            'world_sdf': world_sdf,
            'authority': 'active',
            'active_output_topic': '/state/bev',
            'frame_id': 'map_bev',
            'odometry_topic': _manager_odom_topic,
            'odometry_frame_id': 'odom',
            'odometry_to_map_yaw_rad': float(cfg['spawn']['yaw']),
            'gp_artifact_template': str(cfg.get('manager_gp_artifact_template', '') or ''),
            'camera_ids': _mc_camera_ids,
            'camera_model_includes': _mc_model_includes,
            'camera_calibration_ids': [
                f'{detector_calibration_world}_{camera_id}'
                for camera_id in _mc_camera_ids
            ],
            'camera_image_frame_ids': list(_mc_camera_ids),
            'outcome_journal_path': str(
                cfg.get('manager_outcome_journal_path', '') or ''),
            **_manager_node_parameters(cfg),
        }],
    )
    return [batched, manager]


def _scheduled_detector_node(cfg: Dict[str, object]):
    """Reliability-aware scheduled detector: one inference/cycle on the
    coverage-best camera, hand-over by belief, ~3-4 Hz -> /state/bev."""
    sim_pkg = FindPackageShare('sim')
    world_sdf = PathJoinSubstitution([sim_pkg, 'gazebo_worlds', 'worlds', cfg['world']])
    return Node(
        package='perception', executable='scheduled_camera_detector_node',
        name='scheduled_camera_detector', output='screen',
        parameters=[{
            'use_sim_time': True,
            'model_path': cfg['yolo_model'],
            'world_sdf': world_sdf,
            'coverage_artifact': str(cfg.get('scheduled_coverage_artifact', '') or ''),
            'device': str(cfg.get('yolo_device', '0') or '0'),
            'imgsz': int(cfg.get('yolo_imgsz', 640)),
            'conf': float(cfg.get('yolo_conf_threshold', 0.05)),
            'iou': float(cfg.get('yolo_iou_threshold', 0.45)),
            'report_std_m': float(cfg.get('scheduled_report_std_m', 0.15)),
            'rate_hz': float(cfg.get('scheduled_rate_hz', 5.0)),
            'frame_id': 'map_bev',
            'spawn_x': float(cfg['spawn']['x']),
            'spawn_y': float(cfg['spawn']['y']),
            'camera_ids': list(cfg.get('profile_camera_ids', [])),
            'camera_model_includes': list(cfg.get('profile_camera_model_includes', [])),
            'camera_image_topics': list(cfg.get('profile_camera_image_topics', [])),
            'selection_mode': str(
                cfg.get('scheduled_selection_mode', 'coverage_best_with_fallback')
            ),
            'publish_camera_observation_json': True,
            'camera_calibration_id_prefix': str(cfg['profile']['world_name']),
        }],
    )


def build_agent_runtime_actions(cfg: Dict[str, object]) -> List[object]:
    """Create runtime actions for the visibility-aware agent launch."""
    odom_topic = str(cfg.get('odom_topic') or '/odom_noisy')
    if not _as_bool(cfg.get('use_encoder_noise', True)) and odom_topic == '/odom_noisy':
        odom_topic = '/odom'
    odom_yaw_offset_rad = _odom_yaw_offset_rad(cfg, odom_topic)
    raw_use_nogo_cost = cfg.get('use_nogo_cost', 'auto')
    if isinstance(raw_use_nogo_cost, str) and raw_use_nogo_cost in ('', 'auto', 'default'):
        resolved_use_nogo_cost = False
    else:
        resolved_use_nogo_cost = _as_bool(raw_use_nogo_cost)

    cfg = dict(cfg)
    multicam_belief = _as_bool(cfg.get('multicam_belief', False))
    multicam_scheduled = _as_bool(cfg.get('multicam_scheduled', False))
    if multicam_belief or multicam_scheduled:
        # Multi-camera correction arrives as a world-frame /state/bev message,
        # not a single-camera pixel_pose, so disable the pixel-correction path.
        cfg['use_pixel_correction'] = False
        # Fuse /state/bev as a proper recursive EKF (corrections applied on arrival
        # in _apply_state_correction: latency-compensated motion replay + Kalman
        # update + NIS gate with dead-reckon-and-grow recovery), matching the
        # single-camera pixel path. No hard resets. Default ON for multicam; opt
        # out with state_correction_ekf:false in the config.
        raw_ekf = str(cfg.get('state_correction_ekf_raw', '') or '').strip()
        cfg['state_correction_ekf'] = _as_bool(raw_ekf) if raw_ekf else True
        # Preserve the locked single-camera result (where this cap did not
        # exist), but prevent multicam fallback replay from inventing more
        # motion than the robot can physically execute.
        if float(cfg.get('max_predict_speed_mps', 0.0)) <= 0.0:
            cfg['max_predict_speed_mps'] = float(cfg['v_max'])
    cfg['resolved_use_nogo_cost'] = resolved_use_nogo_cost
    shared_nodes = build_shared_nodes(cfg)
    planner = cfg['planner']

    if planner not in ('visibility_aware_efe', 'constant_R_efe', 'geometric_shortest_path'):
        raise RuntimeError(
            "planner must be 'visibility_aware_efe', 'constant_R_efe' or "
            "'geometric_shortest_path' for agent launch"
        )

    planner_params = {
        'visibility_aware_efe': {
            'approx_method': 'ET1',
            'use_ambiguity': cfg['use_ambiguity'],
            'use_obs_risk': cfg['use_obs_risk'],
        },
        'constant_R_efe': {
            'approx_method': 'ET1',
            'use_ambiguity': True,
            'use_obs_risk': True,
        },
        # C0 conventional-navigation baseline: the one-shot global EFE solve is
        # skipped (global_planner_mode='geometric_shortest_path'), so the EFE
        # terms are unused; disable them so nothing is silently active.
        'geometric_shortest_path': {
            'approx_method': 'ET1',
            'use_ambiguity': False,
            'use_obs_risk': False,
        },
    }
    planner_uses_visibility = (
        _as_bool(cfg['use_visibility_model'])
        and planner not in ('constant_R_efe', 'geometric_shortest_path')
    )
    agent_node = Node(
        package='planning',
        executable='efe_agent',
        name=f'{planner}_agent',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'cmd_topic': '/cmd_vel_raw' if cfg.get('use_command_noise', True) else '/cmd_vel',
            'cmd_publish_rate': cfg['cmd_publish_rate'],
            'plan_rate': cfg['plan_rate'],
            'belief_publish_rate': cfg['belief_publish_rate'],
            'horizon': cfg['horizon'],
            'dt': cfg['dt'],
            'v_max': cfg['v_max'],
            'max_predict_speed_mps': cfg.get('max_predict_speed_mps', 0.0),
            'control_weight': cfg['control_weight'],
            'use_pixel_correction': cfg['use_pixel_correction'],
            'state_correction_ekf': _as_bool(cfg.get('state_correction_ekf', False)),
            'state_correction_mode': cfg.get('state_correction_mode', 'fused'),
            'state_max_correction_jump_m': cfg.get('state_max_correction_jump_m', 0.0),
            'pixel_topic': cfg['pixel_topic'],
            'pixel_timeout_s': cfg['pixel_timeout_s'],
            'pixel_correction_min_interval_s': cfg['pixel_correction_min_interval_s'],
            'pixel_correction_approx': cfg['pixel_correction_approx'],
            'skip_stale_pixel_correction': cfg['skip_stale_pixel_correction'],
            'bev_y_calibration_offset_m': cfg['bev_y_calibration_offset_m'],
            'bev_affine_calibration': cfg.get('bev_affine_calibration', ''),
            'pixel_max_correction_jump_m': cfg['pixel_max_correction_jump_m'],
            'pixel_correction_nis_threshold': cfg['pixel_correction_nis_threshold'],
            'state_reanchor_m': cfg['state_reanchor_m'],
            'state_max_predict_dt_s': cfg['state_max_predict_dt_s'],
            'state_reject_inflate_m2': cfg['state_reject_inflate_m2'],
            'stale_belief_inflate_m2_per_s': cfg['stale_belief_inflate_m2_per_s'],
            'stale_belief_inflate_cap_m2': cfg['stale_belief_inflate_cap_m2'],
            'require_state_correction_envelope': cfg['require_state_correction_envelope'],
            'use_diagnostic_odom_localization': cfg['use_diagnostic_odom_localization'],
            'odom_topic': odom_topic,
            'use_odom_for_predict': cfg['use_odom_for_predict'],
            'heading_update_mode': cfg['heading_update_mode'],
            # /odom_noisy already integrates the world-frame ground-truth pose; raw /odom
            # begins at zero yaw and needs the spawn transform.
            'odom_yaw_offset_rad': odom_yaw_offset_rad,
            # Declared initial prior at the task start (the task declaration, not ground truth).
            'initial_belief_from_task_start': _as_bool(cfg.get('initial_belief_from_task_start', False)),
            'initial_belief_xyyaw': [float(cfg['spawn']['x']), float(cfg['spawn']['y']),
                                     float(cfg['spawn']['yaw'])],
            'initial_belief_sigma_xy_m': float(cfg.get('initial_belief_sigma_xy_m', 0.10)),
            'initial_belief_sigma_theta_rad': float(cfg.get('initial_belief_sigma_theta_rad', math.radians(15.0))),
            'local_controller_type': cfg['local_controller_type'],
            'min_state_cov': cfg['min_state_cov'],
            'debug_runtime': cfg['debug_runtime'],
            'process_noise_xy': cfg['process_noise_xy'],
            'process_noise_theta': cfg['process_noise_theta'],
            'process_noise_model': cfg['process_noise_model'],
            'goal_sigma_uv': cfg['goal_sigma_uv'],
            'risk_weight_obs': cfg['risk_weight_obs'],
            'ambiguity_weight': cfg['ambiguity_weight'],
            'r_visible_uv': cfg['r_visible_uv'],
            'r_miss_uv': cfg['r_miss_uv'],
            'visibility_sigma_kappa': cfg['visibility_sigma_kappa'],
            'goal_prior_u_std_start': cfg['goal_prior_u_std_start'],
            'goal_prior_v_std_start': cfg['goal_prior_v_std_start'],
            'goal_prior_u_std_final': cfg['goal_prior_u_std_final'],
            'goal_prior_v_std_final': cfg['goal_prior_v_std_final'],
            'goal_tightening_power': cfg['goal_tightening_power'],
            'goal_progress_n_steps': cfg['goal_progress_n_steps'],
            'observation_risk_scale': cfg['observation_risk_scale'],
            'ambiguity_term_scale': cfg['ambiguity_term_scale'],
            'discount_gamma': cfg['discount_gamma'],
            'use_visibility_model': planner_uses_visibility,
            'visibility_target_height_m': cfg['visibility_target_height_m'],
            'visibility_geometry_json': cfg['visibility_geometry_json'],
            'collision_geometry_json': cfg['collision_geometry_json'],
            'driveable_geometry_json': cfg.get('driveable_geometry_json', ''),
            'nogo_mode': cfg.get('nogo_mode', 'keep_out'),
            'visibility_artifact_path': cfg['visibility_artifact_path'],
            'camera_network_artifact_path': cfg.get('camera_network_artifact_path', ''),
            'camera_network_expected_sha256': cfg.get(
                'camera_network_expected_sha256', ''),
            'camera_network_expected_source_hashes_json': cfg.get(
                'camera_network_expected_source_hashes_json', ''),
            'camera_network_camera_ids': cfg.get('camera_network_camera_ids', ''),
            'camera_network_active_camera_ids': cfg.get(
                'camera_network_active_camera_ids', ''),
            'camera_network_objective': cfg.get(
                'camera_network_objective', 'legacy_pixel_chart'),
            'network_goal_std_m': cfg.get('network_goal_std_m', 0.10),
            'network_goal_std_start_m': cfg.get('network_goal_std_start_m', -1.0),
            'kouw_et1_ambiguity': cfg.get('kouw_et1_ambiguity', True),
            'camera_network_updates_per_step': cfg.get(
                'camera_network_updates_per_step', 1),
            'optimizer_control_block_steps': cfg.get(
                'optimizer_control_block_steps', 1),
            'use_nogo_cost': cfg['resolved_use_nogo_cost'],
            'nogo_penalty_type': cfg['nogo_penalty_type'],
            'nogo_safe_distance': cfg['nogo_safe_distance'],
            'nogo_logbarrier_eps': cfg['nogo_logbarrier_eps'],
            'nogo_warning_band': cfg['nogo_warning_band'],
            'nogo_near_weight': cfg['nogo_near_weight'],
            'use_belief_nogo_cost': cfg['use_belief_nogo_cost'],
            'nogo_belief_kappa': cfg['nogo_belief_kappa'],
            'use_hit_miss_mixture': cfg.get('use_hit_miss_mixture', False),
            'robot_collision_radius_m': cfg['robot_collision_radius_m'],
            'robot_length_m': cfg.get('robot_length_m', 0.8),
            'robot_width_m': cfg.get('robot_width_m', 0.55),
            'optimizer_maxiter': cfg['optimizer_maxiter'],
            'optimizer_maxfun': cfg['optimizer_maxfun'],
            'optimizer_ftol': cfg['optimizer_ftol'],
            'optimizer_gtol': cfg['optimizer_gtol'],
            'optimizer_warm_start': cfg['optimizer_warm_start'],
            'optimizer_multistart': cfg['optimizer_multistart'],
            'optimizer_multistart_include_direct': cfg['optimizer_multistart_include_direct'],
            'optimizer_initial_routes_json': cfg['optimizer_initial_routes_json'],
            'optimizer_terminal_goal_tolerance_m': cfg['optimizer_terminal_goal_tolerance_m'],
            'optimizer_route_seed_mode': cfg['optimizer_route_seed_mode'],
            'use_hierarchical': cfg.get('use_hierarchical', False),
            'global_planner_mode': cfg.get('global_planner_mode', 'efe'),
            'preselected_route_json': cfg.get('preselected_route_json', ''),
            'preselected_route_sha256': cfg.get('preselected_route_sha256', ''),
            'preselected_route_source_path': cfg.get('preselected_route_source_path', ''),
            'preselected_route_source_sha256': cfg.get(
                'preselected_route_source_sha256', ''
            ),
            'preselected_route_validation_json': cfg.get(
                'preselected_route_validation_json', ''
            ),
            'global_horizon': cfg.get('global_horizon', 60),
            'global_dt': cfg.get('global_dt', 0.0),
            'local_horizon': cfg.get('local_horizon', 12),
            'local_plan_rate': cfg.get('local_plan_rate', 4.0),
            'local_optimizer_maxiter': cfg.get('local_optimizer_maxiter', 60),
            'global_use_ambiguity': cfg.get('global_use_ambiguity', True),
            'local_use_ambiguity': cfg.get('local_use_ambiguity', False),
            'local_use_obs_risk': cfg.get('local_use_obs_risk', True),
            'global_optimizer_multistart': cfg.get('global_optimizer_multistart', True),
            'local_optimizer_multistart': cfg.get('local_optimizer_multistart', True),
            'local_use_visibility_model': cfg.get('local_use_visibility_model', False),
            'local_use_belief_nogo_cost': cfg.get('local_use_belief_nogo_cost', False),
            'local_nogo_penalty_type': cfg.get('local_nogo_penalty_type', ''),
            'local_nogo_safe_distance': cfg.get('local_nogo_safe_distance', -1.0),
            'local_goal_prior_u_std_start': cfg.get('local_goal_prior_u_std_start', -1.0),
            'local_goal_prior_v_std_start': cfg.get('local_goal_prior_v_std_start', -1.0),
            'local_goal_prior_u_std_final': cfg.get('local_goal_prior_u_std_final', -1.0),
            'local_goal_prior_v_std_final': cfg.get('local_goal_prior_v_std_final', -1.0),
            'waypoint_spacing_m': cfg.get('waypoint_spacing_m', 0.2),
            'waypoint_arrival_radius_m': cfg.get('waypoint_arrival_radius_m', 0.1),
            'local_replan_min_remaining_s': cfg.get('local_replan_min_remaining_s', 0.0),
            'local_replan_on_waypoint_change': cfg.get('local_replan_on_waypoint_change', False),
            'latency_compensate_plan_handoff': cfg.get('latency_compensate_plan_handoff', False),
            'simple_tracker_yaw_gate_rad': cfg.get('simple_tracker_yaw_gate_rad', 0.65),
            'ff_fb_turn_rate_limit_rad_s': cfg.get('ff_fb_turn_rate_limit_rad_s', 0.80),
            'ff_fb_corner_crawl_speed_mps': cfg.get('ff_fb_corner_crawl_speed_mps', 0.18),
            'ff_fb_pivot_heading_error_rad': cfg.get('ff_fb_pivot_heading_error_rad', 2.60),
            **cfg['camera_params'],
            **planner_params[planner],
        }],
    )

    if multicam_scheduled:
        # Reliability-aware scheduled detector -> /state/bev (single-cam rate,
        # hand-over by belief).
        after_odom = [_scheduled_detector_node(cfg)]
    elif multicam_belief:
        # Guarded swap: 4-camera detector + active camera_manager -> /state/bev,
        # in place of the single-camera detector + pixel_to_bev.
        after_odom = list(_multicam_perception_nodes(cfg))
    else:
        after_odom = [
            shared_nodes['perception_node'],
            shared_nodes['pixel_to_bev'],
        ]
    # Commissioning drive (enable_mission=false): omit the goal publisher + marker
    # so no goal is ever published. The EFE planner then no-ops in _plan_once and
    # never emits /cmd_vel, while its belief EKF keeps predicting on odom and
    # correcting on pixel detections — an external coverage controller drives.
    if cfg.get('enable_mission', True):
        after_odom.append(shared_nodes['mission_node'])
        after_odom.append(shared_nodes['goal_marker_node'])
    if shared_nodes['logger_node'] is not None:
        after_odom.append(shared_nodes['logger_node'])
    if cfg['use_rviz']:
        after_odom.append(shared_nodes['rviz'])
    if shared_nodes.get('lockstep_scheduler') is not None:
        after_odom.append(shared_nodes['lockstep_scheduler'])

    start_after_odom = RegisterEventHandler(
        OnProcessExit(
            target_action=shared_nodes['wait_for_odom'],
            on_exit=after_odom,
        )
    )

    runtime_actions = [
        shared_nodes['bringup_sim'],
        shared_nodes['tf_static'],
    ]
    if shared_nodes.get('command_noise_node') is not None:
        runtime_actions.append(shared_nodes['command_noise_node'])
    if cfg.get('use_encoder_noise', True):
        runtime_actions.append(shared_nodes['encoder_noise_node'])
    runtime_actions.extend([
        agent_node,
        shared_nodes['wait_for_odom'],
        start_after_odom,
    ])
    return runtime_actions
