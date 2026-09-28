#!/usr/bin/env python3
import csv
from contextlib import nullcontext
import json
import hashlib
import math
import os
import tempfile
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np
import rclpy
import tf2_ros
import yaml
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray, String
from tf2_geometry_msgs import do_transform_pose

from experiments.core.manifest import create_run_dir, snapshot_configs, write_manifest
from experiments.core.camera_opportunity_log import CameraOpportunityLog, JsonlDeliveryLog
from experiments.core.world_profiles import load_profile, compute_look_at_from_pose
from perception.core.detection_diagnostics import (
    DETECTION_DIAGNOSTICS_TOPIC,
    diagnostics_from_message,
)
from reliability.fusion_event import FusedCorrectionEvent
from unav_common.config import parse_bev_affine_calibration
from unav_common.correction_ledger import validate_correction_ledger
from unav_common.mission_goal import MISSION_GOAL_TOPIC, mission_goal_from_json
from unav_common.occlusion_geometry import scene_from_json
from unav_common.terminal_stop import (
    TERMINAL_COMPONENTS,
    TERMINAL_STOP_ACK_TOPIC,
    TERMINAL_STOP_REQUEST_TOPIC,
    TerminalStopRequest,
    terminal_stop_ack_from_json,
    terminal_stop_request_to_json,
)


def _find_repo_root(start_dir: str) -> str:
    current = os.path.abspath(start_dir)
    while True:
        # A worktree has a .git FILE pointing at the main repository, not a directory.
        if os.path.exists(os.path.join(current, '.git')):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return start_dir
        current = parent


def _load_task_start_pose(tasks_yaml_path: str, world: str, task: str):
    path = str(tasks_yaml_path or '').strip()
    if not path or not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8') as handle:
        payload = yaml.safe_load(handle) or {}
    tasks = payload.get('tasks')
    if not isinstance(tasks, dict):
        return None
    world_tasks = tasks.get(str(world), [])
    if not isinstance(world_tasks, list):
        return None
    for entry in world_tasks:
        if not isinstance(entry, dict):
            continue
        if str(entry.get('name', '')).strip() != str(task).strip():
            continue
        start = entry.get('start')
        if not isinstance(start, dict):
            return None
        return (
            float(start['x']),
            float(start['y']),
            float(start.get('yaw', 0.0)),
        )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(str(text or '').encode('utf-8')).hexdigest()


def _sha256_file(path: str):
    candidate = str(path or '').strip()
    if not candidate or not os.path.isfile(candidate):
        return None
    digest = hashlib.sha256()
    try:
        with open(candidate, 'rb') as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _strict_json_value(value):
    """Convert runtime diagnostics to interoperable JSON without NaN/Infinity."""
    if isinstance(value, dict):
        return {str(key): _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict_json_value(value.tolist())
    if isinstance(value, np.generic):
        return _strict_json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json_atomic(path: str, payload) -> None:
    """Commit strict JSON by atomic replace and sync the containing directory."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.summary-', suffix='.tmp', dir=directory)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(_strict_json_value(payload), handle, indent=2,
                      sort_keys=True, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Atomic replacement is complete. Some filesystems do not permit
            # directory fsync; the summary records file-level completion.
            pass
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _split_prisms_by_prefix(prisms, prefix: str):
    token = str(prefix or '').strip()
    if not token:
        return tuple()
    return tuple(prism for prism in tuple(prisms or ()) if str(prism.name).startswith(token))


#: Every terminal outcome a correction may have. The planner publishes exactly one of
#: these per detector batch; anything else means a correction went unaccounted, which is
#: what invalidates a run. `accepted_bootstrap` seeds the belief, `reanchored` recovers a
#: diverged one, `rejected` failed a gate, `dropped` was refused before a prediction
#: existed (a stale stamp, or an outage longer than the replay cap).
KNOWN_ASSIMILATION_STATUSES = frozenset({
    "accepted", "accepted_bootstrap", "reanchored", "rejected", "dropped",
})
EVENT_DRAIN_QUIET_S = 0.20
EVENT_DRAIN_MAX_S = 8.0
TERMINAL_REST_WINDOW_S = 0.25
TERMINAL_REST_MAX_DISPLACEMENT_M = 0.01
STUCK_TURN_PROGRESS_MIN_RAD = 0.15
STUCK_ANGULAR_CMD_FRACTION_MIN = 0.25

class ExperimentLogger(Node):
    def __init__(self):
        super().__init__('experiment_logger')

        self.declare_parameter('log_dir', 'logs/experiments')
        self.declare_parameter('log_rate', 10.0)
        self.declare_parameter('seed', 0)
        self.declare_parameter('method', '')
        self.declare_parameter('perception_backend', '')
        self.declare_parameter('world', '')
        self.declare_parameter('world_sdf_path', '')
        self.declare_parameter('task', '')
        self.declare_parameter('planner', '')
        self.declare_parameter('state_source_x', 'unknown')
        self.declare_parameter('state_source_y', 'unknown')
        self.declare_parameter('state_source_theta', 'unknown')
        self.declare_parameter('state_estimator_mode', 'unknown')
        self.declare_parameter('state_correction_mode', 'fused')
        # The camera-manager settings that define the experiment arm, as JSON, from the
        # same launch helper that configures the manager. Folded into the manifest so a
        # result's arm identity is recorded in the run rather than in its path.
        self.declare_parameter('manager_settings_json', '')
        self.declare_parameter('campaign_config_path', '')
        self.declare_parameter('outcome_journal_path', '')
        self.declare_parameter('manager_outcome_journal_path', '')
        self.declare_parameter('operational_belief_timeout_s', 0.5)
        self.declare_parameter('use_pixel_correction', False)
        self.declare_parameter('pixel_timeout_s', 0.5)
        self.declare_parameter('use_ambiguity', False)
        self.declare_parameter('use_obs_risk', True)
        self.declare_parameter('world_profiles_path', '')
        self.declare_parameter('tasks_yaml', '')
        self.declare_parameter('log_plan_samples', True)
        self.declare_parameter('log_perception_samples', True)
        self.declare_parameter('auto_stop_on_goal', False)
        self.declare_parameter('goal_success_radius', 0.20)
        self.declare_parameter('goal_success_hold_s', 2.0)
        self.declare_parameter('goal_stable_radius', 0.20)
        self.declare_parameter('goal_stable_hold_s', 2.0)
        self.declare_parameter('goal_stable_max_displacement_m', 0.04)
        # Stuck detection is off inside goal_stable_radius; a belief that stays there
        # this long without satisfying either goal hold ends the run as its own failure.
        self.declare_parameter('goal_loiter_timeout_s', 15.0)
        self.declare_parameter('frame_id', 'map_bev')
        self.declare_parameter('frame_sanity_start_tolerance_m', 0.25)
        self.declare_parameter('frame_sanity_start_tolerance_yaw_rad', 0.5)
        self.declare_parameter('use_visibility_model', False)
        self.declare_parameter('visibility_artifact_path', '')
        self.declare_parameter('camera_network_artifact_path', '')
        self.declare_parameter('camera_network_expected_sha256', '')
        self.declare_parameter('camera_network_expected_source_hashes_json', '')
        self.declare_parameter('camera_network_camera_ids', '')
        self.declare_parameter('camera_network_active_camera_ids', '')
        self.declare_parameter('camera_network_objective', 'legacy_pixel_chart')
        self.declare_parameter('camera_network_updates_per_step', 1)
        self.declare_parameter('optimizer_control_block_steps', 1)
        self.declare_parameter('network_goal_std_m', 0.10)
        self.declare_parameter('risk_weight_obs', 1.0)
        self.declare_parameter('ambiguity_weight', 1.0)
        self.declare_parameter('goal_sigma_uv', 2.0)
        self.declare_parameter('r_visible_uv', 2.5)
        self.declare_parameter('r_miss_uv', 120.0)
        self.declare_parameter('visibility_sigma_kappa', 1.0)
        self.declare_parameter('plan_rate', 2.0)
        self.declare_parameter('horizon', 36)
        self.declare_parameter('dt', 0.2)
        self.declare_parameter('control_weight', 0.0)
        self.declare_parameter('process_noise_xy', 0.012)
        self.declare_parameter('process_noise_theta', 0.05)
        self.declare_parameter('goal_prior_u_std_start', 80.0)
        self.declare_parameter('goal_prior_v_std_start', 80.0)
        self.declare_parameter('goal_prior_u_std_final', 18.0)
        self.declare_parameter('goal_prior_v_std_final', 18.0)
        self.declare_parameter('goal_tightening_power', 0.45)
        self.declare_parameter('goal_progress_n_steps', 90)
        self.declare_parameter('observation_risk_scale', 1.25)
        self.declare_parameter('ambiguity_term_scale', 1.00)
        self.declare_parameter('discount_gamma', 0.995)
        self.declare_parameter('visibility_target_height_m', 0.0)
        self.declare_parameter('perception_use_geometry_occlusion', True)
        self.declare_parameter('visibility_geometry_json', '')
        self.declare_parameter('collision_geometry_json', '')
        self.declare_parameter('robot_collision_radius_m', 0.125)
        self.declare_parameter('robot_length_m', 0.8)
        self.declare_parameter('robot_width_m', 0.55)
        self.declare_parameter('use_command_noise', True)
        self.declare_parameter('use_encoder_noise', True)
        self.declare_parameter('command_noise_linear_slip_mean', 0.03)
        self.declare_parameter('command_noise_linear_slip_std', 0.06)
        self.declare_parameter('command_noise_angular_slip_mean', 0.0)
        self.declare_parameter('command_noise_angular_slip_std', 0.04)
        self.declare_parameter('command_noise_linear_additive_std', 0.008)
        self.declare_parameter('command_noise_angular_additive_std', 0.035)
        self.declare_parameter('command_noise_correlation_alpha', 0.85)
        self.declare_parameter('encoder_noise_linear_slip_mean', 0.02)
        self.declare_parameter('encoder_noise_linear_slip_std', 0.05)
        self.declare_parameter('encoder_noise_angular_slip_mean', 0.0)
        self.declare_parameter('encoder_noise_angular_slip_std', 0.03)
        self.declare_parameter('encoder_noise_linear_additive_std', 0.004)
        self.declare_parameter('encoder_noise_angular_additive_std', 0.020)
        self.declare_parameter('encoder_wheel_diameter_ratio_error', 0.0)
        self.declare_parameter('encoder_wheelbase_ratio', 1.0)
        self.declare_parameter('process_noise_model', 'constant_psd')
        self.declare_parameter('encoder_noise_correlation_alpha', 0.8)
        self.declare_parameter('optimizer_maxiter', 80)
        self.declare_parameter('optimizer_maxfun', 500)
        self.declare_parameter('optimizer_ftol', 1e-6)
        self.declare_parameter('optimizer_gtol', 1e-4)
        self.declare_parameter('optimizer_warm_start', True)
        self.declare_parameter('optimizer_multistart', False)
        self.declare_parameter('optimizer_multistart_include_direct', True)
        self.declare_parameter('optimizer_initial_routes_json', '')
        self.declare_parameter('optimizer_terminal_goal_tolerance_m', 0.0)
        self.declare_parameter('optimizer_route_seed_mode', 'explicit')
        self.declare_parameter('use_hierarchical', False)
        self.declare_parameter('global_planner_mode', 'efe')
        self.declare_parameter('preselected_route_json', '')
        self.declare_parameter('preselected_route_sha256', '')
        self.declare_parameter('preselected_route_source_path', '')
        self.declare_parameter('preselected_route_source_sha256', '')
        self.declare_parameter('preselected_route_clearance_m', 0.25)
        self.declare_parameter('preselected_route_endpoint_tolerance_m', 0.25)
        self.declare_parameter('preselected_route_sample_step_m', 0.04)
        self.declare_parameter('preselected_route_validation_json', '')
        self.declare_parameter('global_horizon', 60)
        self.declare_parameter('global_dt', 0.0)
        self.declare_parameter('local_horizon', 12)
        self.declare_parameter('local_plan_rate', 4.0)
        self.declare_parameter('local_optimizer_maxiter', 60)
        self.declare_parameter('global_use_ambiguity', True)
        self.declare_parameter('local_use_ambiguity', False)
        self.declare_parameter('local_use_obs_risk', True)
        self.declare_parameter('global_optimizer_multistart', True)
        self.declare_parameter('local_optimizer_multistart', True)
        self.declare_parameter('local_use_visibility_model', False)
        self.declare_parameter('local_use_belief_nogo_cost', False)
        self.declare_parameter('local_nogo_penalty_type', '')
        self.declare_parameter('local_nogo_safe_distance', -1.0)
        self.declare_parameter('local_goal_prior_u_std_start', -1.0)
        self.declare_parameter('local_goal_prior_v_std_start', -1.0)
        self.declare_parameter('local_goal_prior_u_std_final', -1.0)
        self.declare_parameter('local_goal_prior_v_std_final', -1.0)
        self.declare_parameter('waypoint_spacing_m', 1.0)
        self.declare_parameter('waypoint_arrival_radius_m', 0.35)
        self.declare_parameter('local_replan_min_remaining_s', 0.0)
        self.declare_parameter('local_replan_on_waypoint_change', False)
        self.declare_parameter('latency_compensate_plan_handoff', False)
        self.declare_parameter('cmd_publish_rate', 10.0)
        self.declare_parameter('heading_update_mode', 'camera_xy_only')
        # Recovery policy. Recorded because it decides what the belief does when a
        # correction is refused, which is not visible in any error column.
        self.declare_parameter('state_reanchor_m', 0.0)
        self.declare_parameter('state_max_predict_dt_s', 1.5)
        self.declare_parameter('state_reject_inflate_m2', 0.0)
        self.declare_parameter('stale_belief_inflate_m2_per_s', 0.0)
        self.declare_parameter('stale_belief_inflate_cap_m2', 0.0)
        self.declare_parameter('require_state_correction_envelope', False)
        self.declare_parameter('use_nogo_cost', False)
        self.declare_parameter('nogo_penalty_type', 'warning_band')
        self.declare_parameter('nogo_safe_distance', 0.0)
        self.declare_parameter('nogo_logbarrier_eps', 1e-3)
        self.declare_parameter('nogo_warning_band', 0.05)
        self.declare_parameter('nogo_near_weight', 50.0)
        self.declare_parameter('use_belief_nogo_cost', False)
        self.declare_parameter('nogo_belief_kappa', 1.0)
        # Frozen planner-observation-model provenance. The launch path already
        # forwards this parameter to both the planner and logger.
        self.declare_parameter('use_hit_miss_mixture', False)
        self.declare_parameter('nogo_mode', 'keep_out')
        self.declare_parameter('yolo_model', '')
        self.declare_parameter('yolo_compiled_model', '')
        self.declare_parameter('yolo_device', '')
        self.declare_parameter('yolo_imgsz', 640)
        self.declare_parameter('yolo_conf_threshold', 0.25)
        self.declare_parameter('yolo_predict_conf_floor', 0.05)
        self.declare_parameter('yolo_iou_threshold', 0.45)
        self.declare_parameter('yolo_target_class', 'robot')
        self.declare_parameter('yolo_class_id', -1)
        self.declare_parameter('yolo_use_masks', True)
        self.declare_parameter('yolo_min_mask_area_px', 12.0)
        self.declare_parameter('yolo_mask_bottom_band_px', 3.0)
        self.declare_parameter('yolo_max_batch_stamp_skew_s', 0.05)
        self.declare_parameter('show_pose_markers', False)
        self.declare_parameter('diagnostics_match_tolerance_s', 1e-3)
        self.declare_parameter('bev_y_calibration_offset_m', 0.0)
        self.declare_parameter('bev_affine_calibration', '')
        self.declare_parameter('pixel_correction_nis_threshold', 0.0)
        self.declare_parameter('odom_topic', '/odom_noisy')
        self.declare_parameter('run_dir_topic', '/experiment/run_dir')
        self.declare_parameter('run_timeout_after_first_cmd_s', 75.0)
        self.declare_parameter('first_cmd_linear_eps', 0.02)
        self.declare_parameter('first_cmd_angular_eps', 0.10)
        self.declare_parameter('v_max', 0.22)
        self.declare_parameter('use_odom_for_predict', True)
        self.declare_parameter('use_diagnostic_odom_localization', False)
        self.declare_parameter('local_controller_type', 'ff_fb')
        self.declare_parameter('stuck_window_s', 8.0)
        self.declare_parameter('stuck_max_displacement_m', 0.08)
        self.declare_parameter('stuck_max_goal_improvement_m', 0.05)
        self.declare_parameter('stuck_cmd_fraction_min', 0.50)
        self.declare_parameter('stuck_idle_cmd_fraction_max', 0.10)

        log_dir = self.get_parameter('log_dir').value
        self.seed = int(self.get_parameter('seed').value)
        self.method = str(self.get_parameter('method').value)
        self.perception_backend = str(self.get_parameter('perception_backend').value)
        self.world = self.get_parameter('world').value
        self.world_sdf_path = str(self.get_parameter('world_sdf_path').value or '').strip()
        if self.world_sdf_path and not os.path.isfile(self.world_sdf_path):
            raise RuntimeError('configured world_sdf_path does not name a readable file')
        self.task = self.get_parameter('task').value
        self.planner = self.get_parameter('planner').value
        self.state_source_x = str(self.get_parameter('state_source_x').value)
        self.state_source_y = str(self.get_parameter('state_source_y').value)
        self.state_source_theta = str(self.get_parameter('state_source_theta').value)
        self.state_estimator_mode = str(self.get_parameter('state_estimator_mode').value)
        self.state_correction_mode = str(
            self.get_parameter('state_correction_mode').value
        )
        self.manager_settings_json = str(
            self.get_parameter('manager_settings_json').value or '')
        self.campaign_config_path = str(
            self.get_parameter('campaign_config_path').value or '')
        self.outcome_journal_path = str(
            self.get_parameter('outcome_journal_path').value or '').strip()
        self.manager_outcome_journal_path = str(
            self.get_parameter('manager_outcome_journal_path').value or '').strip()
        self.operational_belief_timeout_s = float(
            self.get_parameter('operational_belief_timeout_s').value)
        if (not math.isfinite(self.operational_belief_timeout_s)
                or self.operational_belief_timeout_s <= 0.0):
            raise RuntimeError('operational_belief_timeout_s must be finite and positive')
        self.heading_update_mode = str(self.get_parameter('heading_update_mode').value)
        self.state_reanchor_m = float(self.get_parameter('state_reanchor_m').value)
        self.state_max_predict_dt_s = float(
            self.get_parameter('state_max_predict_dt_s').value)
        self.state_reject_inflate_m2 = float(
            self.get_parameter('state_reject_inflate_m2').value)
        self.stale_belief_inflate_m2_per_s = float(
            self.get_parameter('stale_belief_inflate_m2_per_s').value)
        self.stale_belief_inflate_cap_m2 = float(
            self.get_parameter('stale_belief_inflate_cap_m2').value)
        self.require_state_correction_envelope = bool(
            self.get_parameter('require_state_correction_envelope').value)
        self.use_pixel_correction = bool(self.get_parameter('use_pixel_correction').value)
        self.pixel_timeout_s = float(self.get_parameter('pixel_timeout_s').value)
        self.use_ambiguity = bool(self.get_parameter('use_ambiguity').value)
        self.use_obs_risk = bool(self.get_parameter('use_obs_risk').value)
        self.world_profiles_path = self.get_parameter('world_profiles_path').value
        self.tasks_yaml = self.get_parameter('tasks_yaml').value
        self.log_plan_samples = bool(self.get_parameter('log_plan_samples').value)
        self.log_perception_samples = bool(self.get_parameter('log_perception_samples').value)
        self.auto_stop_on_goal = bool(self.get_parameter('auto_stop_on_goal').value)
        self.goal_success_radius = float(self.get_parameter('goal_success_radius').value)
        self.goal_success_hold_s = float(self.get_parameter('goal_success_hold_s').value)
        self.goal_stable_radius = float(self.get_parameter('goal_stable_radius').value)
        self.goal_stable_hold_s = float(self.get_parameter('goal_stable_hold_s').value)
        self.goal_stable_max_displacement_m = float(
            self.get_parameter('goal_stable_max_displacement_m').value
        )
        self.goal_loiter_timeout_s = float(self.get_parameter('goal_loiter_timeout_s').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.frame_sanity_start_tolerance_m = float(self.get_parameter('frame_sanity_start_tolerance_m').value)
        self.frame_sanity_start_tolerance_yaw_rad = float(self.get_parameter('frame_sanity_start_tolerance_yaw_rad').value)
        self.use_visibility_model = bool(self.get_parameter('use_visibility_model').value)
        self.visibility_artifact_path = str(self.get_parameter('visibility_artifact_path').value)
        self.camera_network_artifact_path = str(self.get_parameter('camera_network_artifact_path').value)
        self.camera_network_expected_sha256 = str(
            self.get_parameter('camera_network_expected_sha256').value or '').strip()
        if self.camera_network_expected_sha256 and (
                len(self.camera_network_expected_sha256) != 64
                or any(char not in '0123456789abcdef'
                       for char in self.camera_network_expected_sha256)):
            raise RuntimeError(
                'camera_network_expected_sha256 must be a lowercase SHA-256')
        source_hashes_text = str(
            self.get_parameter('camera_network_expected_source_hashes_json').value or '').strip()
        try:
            self.camera_network_source_hashes = (
                json.loads(source_hashes_text) if source_hashes_text else {})
        except (TypeError, ValueError) as exc:
            raise RuntimeError('camera_network_expected_source_hashes_json is malformed') from exc
        if not isinstance(self.camera_network_source_hashes, dict) or not all(
                isinstance(key, str) and key
                and isinstance(value, str) and len(value) == 64
                and all(char in '0123456789abcdef' for char in value)
                for key, value in self.camera_network_source_hashes.items()):
            raise RuntimeError(
                'camera network source hashes must map names to lowercase SHA-256 strings')
        camera_ids_text = str(
            self.get_parameter('camera_network_camera_ids').value or '').strip()
        self.camera_network_camera_ids = [
            item.strip() for item in camera_ids_text.split(',') if item.strip()]
        if len(self.camera_network_camera_ids) != len(set(self.camera_network_camera_ids)):
            raise RuntimeError('camera_network_camera_ids contains duplicates')
        active_camera_ids_text = str(
            self.get_parameter('camera_network_active_camera_ids').value or '').strip()
        self.camera_network_active_camera_ids = [
            item.strip() for item in active_camera_ids_text.split(',') if item.strip()]
        if (self.camera_network_artifact_path
                and (not self.camera_network_active_camera_ids
                     or len(self.camera_network_active_camera_ids)
                     != len(set(self.camera_network_active_camera_ids)))):
            raise RuntimeError(
                'camera_network_active_camera_ids must be nonempty and unique')
        if not set(self.camera_network_active_camera_ids).issubset(
                self.camera_network_camera_ids):
            raise RuntimeError(
                'active planning cameras must be a subset of the artifact roster')
        self.camera_network_objective = str(
            self.get_parameter('camera_network_objective').value or '').strip().lower()
        self.camera_network_updates_per_step = int(
            self.get_parameter('camera_network_updates_per_step').value)
        self.optimizer_control_block_steps = int(
            self.get_parameter('optimizer_control_block_steps').value)
        self.network_goal_std_m = float(self.get_parameter('network_goal_std_m').value)
        if self.camera_network_objective not in ('legacy_pixel_chart', 'metric_expected_belief'):
            raise RuntimeError('unknown camera_network_objective')
        if self.camera_network_updates_per_step < 1:
            raise RuntimeError('camera_network_updates_per_step must be positive')
        if not np.isfinite(self.network_goal_std_m) or self.network_goal_std_m <= 0.:
            raise RuntimeError('network_goal_std_m must be finite and positive')
        self.risk_weight_obs = float(self.get_parameter('risk_weight_obs').value)
        self.ambiguity_weight = float(self.get_parameter('ambiguity_weight').value)
        self.goal_sigma_uv = float(self.get_parameter('goal_sigma_uv').value)
        self.r_visible_uv = float(self.get_parameter('r_visible_uv').value)
        self.r_miss_uv = float(self.get_parameter('r_miss_uv').value)
        self.visibility_sigma_kappa = float(self.get_parameter('visibility_sigma_kappa').value)
        self.plan_rate = float(self.get_parameter('plan_rate').value)
        self.horizon = int(self.get_parameter('horizon').value)
        self.dt = float(self.get_parameter('dt').value)
        self.control_weight = float(self.get_parameter('control_weight').value)
        self.process_noise_xy = float(self.get_parameter('process_noise_xy').value)
        self.process_noise_theta = float(self.get_parameter('process_noise_theta').value)
        self.goal_prior_u_std_start = float(self.get_parameter('goal_prior_u_std_start').value)
        self.goal_prior_v_std_start = float(self.get_parameter('goal_prior_v_std_start').value)
        self.goal_prior_u_std_final = float(self.get_parameter('goal_prior_u_std_final').value)
        self.goal_prior_v_std_final = float(self.get_parameter('goal_prior_v_std_final').value)
        self.goal_tightening_power = float(self.get_parameter('goal_tightening_power').value)
        self.goal_progress_n_steps = int(self.get_parameter('goal_progress_n_steps').value)
        self.observation_risk_scale = float(self.get_parameter('observation_risk_scale').value)
        self.ambiguity_term_scale = float(self.get_parameter('ambiguity_term_scale').value)
        self.discount_gamma = float(self.get_parameter('discount_gamma').value)
        self.visibility_target_height_m = float(self.get_parameter('visibility_target_height_m').value)
        self.perception_use_geometry_occlusion = bool(
            self.get_parameter('perception_use_geometry_occlusion').value
        )
        self.visibility_geometry_json = str(self.get_parameter('visibility_geometry_json').value)
        self.collision_geometry_json = str(self.get_parameter('collision_geometry_json').value)
        # Recorded for evidence identity: the planner's no-go term uses it.
        self.robot_collision_radius_m = float(self.get_parameter('robot_collision_radius_m').value)
        self.robot_length_m = float(self.get_parameter('robot_length_m').value)
        self.robot_width_m = float(self.get_parameter('robot_width_m').value)
        self.use_command_noise = bool(self.get_parameter('use_command_noise').value)
        self.use_encoder_noise = bool(self.get_parameter('use_encoder_noise').value)
        self.command_noise_linear_slip_mean = float(self.get_parameter('command_noise_linear_slip_mean').value)
        self.command_noise_linear_slip_std = float(self.get_parameter('command_noise_linear_slip_std').value)
        self.command_noise_angular_slip_mean = float(self.get_parameter('command_noise_angular_slip_mean').value)
        self.command_noise_angular_slip_std = float(self.get_parameter('command_noise_angular_slip_std').value)
        self.command_noise_linear_additive_std = float(self.get_parameter('command_noise_linear_additive_std').value)
        self.command_noise_angular_additive_std = float(self.get_parameter('command_noise_angular_additive_std').value)
        self.command_noise_correlation_alpha = float(self.get_parameter('command_noise_correlation_alpha').value)
        self.encoder_noise_linear_slip_mean = float(self.get_parameter('encoder_noise_linear_slip_mean').value)
        self.encoder_noise_linear_slip_std = float(self.get_parameter('encoder_noise_linear_slip_std').value)
        self.encoder_noise_angular_slip_mean = float(self.get_parameter('encoder_noise_angular_slip_mean').value)
        self.encoder_noise_angular_slip_std = float(self.get_parameter('encoder_noise_angular_slip_std').value)
        self.encoder_noise_linear_additive_std = float(self.get_parameter('encoder_noise_linear_additive_std').value)
        self.encoder_noise_angular_additive_std = float(self.get_parameter('encoder_noise_angular_additive_std').value)
        self.encoder_wheel_diameter_ratio_error = float(self.get_parameter('encoder_wheel_diameter_ratio_error').value)
        self.encoder_wheelbase_ratio = float(self.get_parameter('encoder_wheelbase_ratio').value)
        self.process_noise_model = str(self.get_parameter('process_noise_model').value)
        self.encoder_noise_correlation_alpha = float(self.get_parameter('encoder_noise_correlation_alpha').value)
        self.optimizer_maxiter = int(self.get_parameter('optimizer_maxiter').value)
        self.optimizer_maxfun = int(self.get_parameter('optimizer_maxfun').value)
        self.optimizer_ftol = float(self.get_parameter('optimizer_ftol').value)
        self.optimizer_gtol = float(self.get_parameter('optimizer_gtol').value)
        self.optimizer_warm_start = bool(self.get_parameter('optimizer_warm_start').value)
        self.optimizer_multistart = bool(self.get_parameter('optimizer_multistart').value)
        self.optimizer_multistart_include_direct = bool(
            self.get_parameter('optimizer_multistart_include_direct').value
        )
        self.optimizer_initial_routes_json = str(
            self.get_parameter('optimizer_initial_routes_json').value
        )
        self.optimizer_terminal_goal_tolerance_m = float(
            self.get_parameter('optimizer_terminal_goal_tolerance_m').value
        )
        self.optimizer_route_seed_mode = str(
            self.get_parameter('optimizer_route_seed_mode').value or 'explicit'
        )
        self.use_hierarchical = bool(self.get_parameter('use_hierarchical').value)
        self.global_planner_mode = str(
            self.get_parameter('global_planner_mode').value or 'efe'
        ).strip().lower()
        self.preselected_route_json = str(
            self.get_parameter('preselected_route_json').value or ''
        )
        self.preselected_route_sha256 = str(
            self.get_parameter('preselected_route_sha256').value or ''
        )
        self.preselected_route_source_path = str(
            self.get_parameter('preselected_route_source_path').value or ''
        )
        self.preselected_route_source_sha256 = str(
            self.get_parameter('preselected_route_source_sha256').value or ''
        )
        self.preselected_route_clearance_m = float(
            self.get_parameter('preselected_route_clearance_m').value
        )
        self.preselected_route_endpoint_tolerance_m = float(
            self.get_parameter('preselected_route_endpoint_tolerance_m').value
        )
        self.preselected_route_sample_step_m = float(
            self.get_parameter('preselected_route_sample_step_m').value
        )
        self.preselected_route_validation_json = str(
            self.get_parameter('preselected_route_validation_json').value or ''
        )
        self.global_horizon = int(self.get_parameter('global_horizon').value)
        self.global_dt = float(self.get_parameter('global_dt').value)
        self.local_horizon = int(self.get_parameter('local_horizon').value)
        self.local_plan_rate = float(self.get_parameter('local_plan_rate').value)
        self.local_optimizer_maxiter = int(self.get_parameter('local_optimizer_maxiter').value)
        self.global_use_ambiguity = bool(self.get_parameter('global_use_ambiguity').value)
        self.local_use_ambiguity = bool(self.get_parameter('local_use_ambiguity').value)
        self.local_use_obs_risk = bool(self.get_parameter('local_use_obs_risk').value)
        self.global_optimizer_multistart = bool(
            self.get_parameter('global_optimizer_multistart').value
        )
        self.local_optimizer_multistart = bool(
            self.get_parameter('local_optimizer_multistart').value
        )
        self.local_use_visibility_model = bool(
            self.get_parameter('local_use_visibility_model').value
        )
        self.local_use_belief_nogo_cost = bool(
            self.get_parameter('local_use_belief_nogo_cost').value
        )
        self.local_nogo_penalty_type = str(
            self.get_parameter('local_nogo_penalty_type').value or ''
        )
        self.local_nogo_safe_distance = float(
            self.get_parameter('local_nogo_safe_distance').value
        )
        self.local_goal_prior_u_std_start = float(
            self.get_parameter('local_goal_prior_u_std_start').value
        )
        self.local_goal_prior_v_std_start = float(
            self.get_parameter('local_goal_prior_v_std_start').value
        )
        self.local_goal_prior_u_std_final = float(
            self.get_parameter('local_goal_prior_u_std_final').value
        )
        self.local_goal_prior_v_std_final = float(
            self.get_parameter('local_goal_prior_v_std_final').value
        )
        self.waypoint_spacing_m = float(self.get_parameter('waypoint_spacing_m').value)
        self.waypoint_arrival_radius_m = float(
            self.get_parameter('waypoint_arrival_radius_m').value
        )
        self.local_replan_min_remaining_s = float(
            self.get_parameter('local_replan_min_remaining_s').value
        )
        self.local_replan_on_waypoint_change = bool(
            self.get_parameter('local_replan_on_waypoint_change').value
        )
        self.latency_compensate_plan_handoff = bool(
            self.get_parameter('latency_compensate_plan_handoff').value
        )
        self.cmd_publish_rate = float(self.get_parameter('cmd_publish_rate').value)
        self.use_nogo_cost = bool(self.get_parameter('use_nogo_cost').value)
        self.nogo_penalty_type = str(self.get_parameter('nogo_penalty_type').value)
        self.nogo_safe_distance = float(self.get_parameter('nogo_safe_distance').value)
        self.nogo_logbarrier_eps = float(self.get_parameter('nogo_logbarrier_eps').value)
        self.nogo_warning_band = float(self.get_parameter('nogo_warning_band').value)
        self.nogo_near_weight = float(self.get_parameter('nogo_near_weight').value)
        self.use_belief_nogo_cost = bool(self.get_parameter('use_belief_nogo_cost').value)
        self.nogo_belief_kappa = float(self.get_parameter('nogo_belief_kappa').value)
        self.use_hit_miss_mixture = bool(
            self.get_parameter('use_hit_miss_mixture').value
        )
        self.nogo_mode = str(self.get_parameter('nogo_mode').value or 'keep_out')
        self.yolo_model = str(self.get_parameter('yolo_model').value)
        self.yolo_compiled_model = str(
            self.get_parameter('yolo_compiled_model').value or '')
        self.yolo_device = str(self.get_parameter('yolo_device').value)
        self.yolo_imgsz = int(self.get_parameter('yolo_imgsz').value)
        self.yolo_conf_threshold = float(self.get_parameter('yolo_conf_threshold').value)
        self.yolo_predict_conf_floor = float(
            self.get_parameter('yolo_predict_conf_floor').value)
        self.yolo_iou_threshold = float(self.get_parameter('yolo_iou_threshold').value)
        self.yolo_target_class = str(self.get_parameter('yolo_target_class').value)
        self.yolo_class_id = int(self.get_parameter('yolo_class_id').value)
        self.yolo_use_masks = bool(self.get_parameter('yolo_use_masks').value)
        self.yolo_min_mask_area_px = float(self.get_parameter('yolo_min_mask_area_px').value)
        self.yolo_mask_bottom_band_px = float(self.get_parameter('yolo_mask_bottom_band_px').value)
        self.yolo_max_batch_stamp_skew_s = float(
            self.get_parameter('yolo_max_batch_stamp_skew_s').value)
        self.show_pose_markers = bool(self.get_parameter('show_pose_markers').value)
        self.diagnostics_match_tolerance_s = float(
            self.get_parameter('diagnostics_match_tolerance_s').value
        )
        self.bev_y_calibration_offset_m = float(
            self.get_parameter('bev_y_calibration_offset_m').value
        )
        self.bev_affine_calibration = str(
            self.get_parameter('bev_affine_calibration').value or ''
        ).strip()
        self._bev_affine = self._parse_bev_affine(self.bev_affine_calibration)
        self.pixel_correction_nis_threshold = float(
            self.get_parameter('pixel_correction_nis_threshold').value
        )
        self.odom_topic = str(self.get_parameter('odom_topic').value or '/odom_noisy')
        self.run_dir_topic = str(self.get_parameter('run_dir_topic').value).strip() or '/experiment/run_dir'
        self.run_timeout_after_first_cmd_s = float(self.get_parameter('run_timeout_after_first_cmd_s').value)
        self.first_cmd_linear_eps = float(self.get_parameter('first_cmd_linear_eps').value)
        self.first_cmd_angular_eps = float(self.get_parameter('first_cmd_angular_eps').value)
        self.v_max = float(self.get_parameter('v_max').value)
        self.use_odom_for_predict = bool(self.get_parameter('use_odom_for_predict').value)
        self.use_diagnostic_odom_localization = bool(
            self.get_parameter('use_diagnostic_odom_localization').value
        )
        self.local_controller_type = str(
            self.get_parameter('local_controller_type').value)
        self.stuck_window_s = float(self.get_parameter('stuck_window_s').value)
        self.stuck_max_displacement_m = float(self.get_parameter('stuck_max_displacement_m').value)
        self.stuck_max_goal_improvement_m = float(self.get_parameter('stuck_max_goal_improvement_m').value)
        self.stuck_cmd_fraction_min = float(self.get_parameter('stuck_cmd_fraction_min').value)
        self.stuck_idle_cmd_fraction_max = float(
            self.get_parameter('stuck_idle_cmd_fraction_max').value
        )

        # The camera model comes from the world profile and from nowhere else, so the
        # logger's projection is the same one the state and planner nodes use. There is
        # deliberately no node parameter for the camera pose or intrinsics: a second way
        # to say where a camera is, is a second thing that can disagree.

        run_info = create_run_dir(log_dir)
        self.run_id = run_info['run_id']
        self.run_dir = run_info['run_dir']

        self.log_path = os.path.join(self.run_dir, 'experiment.csv')

        repo_root = _find_repo_root(os.getcwd())
        self.repo_root = repo_root
        self.task_start_pose = _load_task_start_pose(self.tasks_yaml, self.world, self.task)
        profile, _intrinsics, _world_path, _camera_pose = load_profile(self.world_profiles_path, self.world)

        # Build the homography camera from the profile (same source the state/planner
        # nodes use) so pred_world_x/y match the true Gazebo camera. _camera_pose is
        # [x, y, z, roll, pitch, yaw]; look_at is derived exactly as the launch does.
        from unav_common.camera_model import ObliqueCameraModel
        _cam_pos = [float(_camera_pose[0]), float(_camera_pose[1]), float(_camera_pose[2])]
        _look_at = compute_look_at_from_pose(
            _cam_pos, float(_camera_pose[3]), float(_camera_pose[4]), float(_camera_pose[5])
        )
        self.camera_model = ObliqueCameraModel(
            cam_pos=np.array(_cam_pos, dtype=float),
            look_at=np.array(_look_at, dtype=float),
            img_width=int(_intrinsics['img_width']),
            img_height=int(_intrinsics['img_height']),
            fov_h_rad=float(_intrinsics['fov_h_rad']),
        )
        self.camera_pos_xy = np.asarray(_cam_pos[:2], dtype=float).reshape(2)
        self.get_logger().info(
            f"[camera_model] profile-built cam_pos={_cam_pos} look_at={_look_at} "
            f"img=({int(_intrinsics['img_width'])}x{int(_intrinsics['img_height'])}) "
            f"fov_h={float(_intrinsics['fov_h_rad']):.4f}"
        )

        visibility_defaults = dict(profile.get('visibility_defaults') or {})
        self.world_bounds = {
            'xmin': float(visibility_defaults.get('visibility_map_min_x', math.nan)),
            'xmax': float(visibility_defaults.get('visibility_map_max_x', math.nan)),
            'ymin': float(visibility_defaults.get('visibility_map_min_y', math.nan)),
            'ymax': float(visibility_defaults.get('visibility_map_max_y', math.nan)),
        }

        collision_scene = scene_from_json(self.collision_geometry_json)
        self._collision_prisms = tuple(collision_scene.prisms)
        try:
            manager_settings = json.loads(self.manager_settings_json or '{}')
            if not isinstance(manager_settings, dict):
                manager_settings = {}
        except (ValueError, TypeError):
            manager_settings = {}
        world_sdf_sha256 = _sha256_file(self.world_sdf_path)
        if self.world_sdf_path and world_sdf_sha256 is None:
            raise RuntimeError('configured world_sdf_path cannot be hashed')

        manifest_data = {
            'run_id': self.run_id,
            # Which logging conventions this run was written under. Bumped when a
            # column changes meaning, so an analysis can refuse a run it cannot score
            # instead of silently mixing definitions.
            #   1 = pre-2026-08-28. Errors scored against the truth held at LOG time;
            #       final_goal_distance measured from wheel odometry; one
            #       fusion_observations row per decision with no way to tell repeats
            #       of one detection apart.
            #   2 = errors scored against the truth at each estimate's own stamp,
            #       final_goal_distance from ground truth, fusion_observations carries
            #       obs_repeat / obs_seq / gt_*_at_obs / fused_stamp.
            #   3 = detector batches are first-class identities; camera observations
            #       record both capture-time and common-time values; experiment
            #       termination uses the operational belief (never ground truth).
            #   4 = every published fused correction carries source_batch_id through
            #       the filter and produces one terminal assimilation record.
            #   5 = fusion_observations.csv carries pred_h_px / pred_w_px, the box
            #       the hull model predicted from the pose the correction was made
            #       from, so the height ratio is reconstructable from a drive.
            #   6 = fusion_observations.csv carries raw_obs_x / raw_obs_y, the
            #       UNCORRECTED back-projection of the same box. obs_x/obs_y is what
            #       the runtime observation model decided the reading means; these are
            #       what the camera saw. With both, one drive supports replaying any
            #       interpretation on identical readings instead of needing a separate
            #       drive per interpretation.
            #   7 = raw per-camera opportunities (including misses and refusals
            #       upstream of manager selection) retained in camera_opportunities.jsonl.
            #   8 = raw correction/decision/stage deliveries are retained exactly;
            #       identity-bearing fused publications have their own ledger; terminal
            #       schema-2 posterior fields pass through; final summary follows a
            #       bounded drain, complete ledger reconciliation and atomic file close.
            #   9 = every planner-published belief prediction is retained with its
            #       state timestamp, anchor epoch/revision and full covariance. This
            #       makes simultaneous predictions from different correction revisions
            #       distinguishable instead of silently choosing one Pose message.
            'logging_schema_version': 9,
            'camera_opportunity_log': 'camera_opportunities.jsonl',
            'camera_opportunity_scope': 'all received detector outputs; not all scheduled sensor frames',
            'camera_opportunity_schema': 'camera_opportunity_log.v2',
            'runtime_event_delivery_ledger': 'runtime_event_deliveries.jsonl',
            'runtime_event_delivery_schema': 'runtime_event_delivery.v1',
            'belief_prediction_ledger': 'belief_predictions.jsonl',
            'belief_prediction_schema': 'planner_belief_prediction.v1',
            'detector_outcome_journal_path': self.outcome_journal_path,
            'manager_outcome_journal_path': self.manager_outcome_journal_path,
            'correction_publication_ledger': 'correction_publications.csv',
            'ground_truth_pose_ledger': 'ground_truth_pose.csv',
            'correction_publication_schema': 'correction_publications.v2',
            'correction_assimilation_schema': '1_or_2_passthrough',
            'committed_posterior_capability': (
                'terminal_schema_2_required_with_mean_full_covariance_frame_state_time_revision'),
            'command_diagnostic_semantics': (
                'cmd_raw_is_requested;cmd_is_ros_published;neither_is_physical_application'),
            'actuation_outcome_semantics': (
                'native_forwarding_boundary_only;physical_application_verified_false'),
            'terminal_stop_protocol': 'terminal_stop_request.v1+terminal_stop_ack.v1',
            'terminal_stop_verification': (
                'planner_latched_zero+native_guard_forwarded_zero+'
                f'operational_rest_{TERMINAL_REST_WINDOW_S:.2f}s_'
                f'{TERMINAL_REST_MAX_DISPLACEMENT_M:.3f}m'),
            'timestamp': datetime.now().isoformat(),
            'method': self.method or self.planner,
            'perception_backend': self.perception_backend,
            'world': self.world,
            'world_sdf_path': self.world_sdf_path,
            'world_sdf_sha256': world_sdf_sha256,
            'task': self.task,
            'planner': self.planner,
            'state_source_x': self.state_source_x,
            'state_source_y': self.state_source_y,
            'state_source_theta': self.state_source_theta,
            'state_estimator_mode': self.state_estimator_mode,
            'state_correction_mode': self.state_correction_mode,
            'heading_update_mode': self.heading_update_mode,
            'state_reanchor_m': self.state_reanchor_m,
            'state_max_predict_dt_s': self.state_max_predict_dt_s,
            'state_reject_inflate_m2': self.state_reject_inflate_m2,
            'stale_belief_inflate_m2_per_s': self.stale_belief_inflate_m2_per_s,
            'stale_belief_inflate_cap_m2': self.stale_belief_inflate_cap_m2,
            'require_state_correction_envelope': self.require_state_correction_envelope,
            'operational_belief_timeout_s': self.operational_belief_timeout_s,
            'use_pixel_correction': self.use_pixel_correction,
            'pixel_timeout_s': self.pixel_timeout_s,
            'use_ambiguity': self.use_ambiguity,
            'use_obs_risk': self.use_obs_risk,
            'use_visibility_model': self.use_visibility_model,
            'visibility_artifact_path': self.visibility_artifact_path,
            'camera_network_artifact_path': self.camera_network_artifact_path,
            'camera_network_active_camera_ids': list(
                self.camera_network_active_camera_ids),
            'camera_network_objective': self.camera_network_objective,
            'camera_network_updates_per_step': self.camera_network_updates_per_step,
            'network_goal_std_m': self.network_goal_std_m,
            'risk_weight_obs': self.risk_weight_obs,
            'ambiguity_weight': self.ambiguity_weight,
            'goal_sigma_uv': self.goal_sigma_uv,
            'r_visible_uv': self.r_visible_uv,
            'r_miss_uv': self.r_miss_uv,
            'visibility_sigma_kappa': self.visibility_sigma_kappa,
            'goal_prior_u_std_start': self.goal_prior_u_std_start,
            'goal_prior_v_std_start': self.goal_prior_v_std_start,
            'goal_prior_u_std_final': self.goal_prior_u_std_final,
            'goal_prior_v_std_final': self.goal_prior_v_std_final,
            'goal_tightening_power': self.goal_tightening_power,
            'goal_progress_n_steps': self.goal_progress_n_steps,
            'observation_risk_scale': self.observation_risk_scale,
            'ambiguity_term_scale': self.ambiguity_term_scale,
            'discount_gamma': self.discount_gamma,
            'visibility_target_height_m': self.visibility_target_height_m,
            'visibility_geometry_json': self.visibility_geometry_json,
            'visibility_geometry_sha256': _sha256_text(self.visibility_geometry_json),
            'collision_geometry_json': self.collision_geometry_json,
            'collision_geometry_sha256': _sha256_text(self.collision_geometry_json),
            'robot_collision_radius_m': self.robot_collision_radius_m,
            'robot_length_m': self.robot_length_m, 'robot_width_m': self.robot_width_m,
            'planner_collision_model': 'oriented_rectangle_swept_v1',
            'legacy_geometry_diagnostic_model': 'circle',
            'use_command_noise': self.use_command_noise,
            'use_encoder_noise': self.use_encoder_noise,
            'command_noise_linear_slip_mean': self.command_noise_linear_slip_mean,
            'command_noise_linear_slip_std': self.command_noise_linear_slip_std,
            'command_noise_angular_slip_mean': self.command_noise_angular_slip_mean,
            'command_noise_angular_slip_std': self.command_noise_angular_slip_std,
            'command_noise_linear_additive_std': self.command_noise_linear_additive_std,
            'command_noise_angular_additive_std': self.command_noise_angular_additive_std,
            'command_noise_correlation_alpha': self.command_noise_correlation_alpha,
            'encoder_noise_linear_slip_mean': self.encoder_noise_linear_slip_mean,
            'encoder_noise_linear_slip_std': self.encoder_noise_linear_slip_std,
            'encoder_noise_angular_slip_mean': self.encoder_noise_angular_slip_mean,
            'encoder_noise_angular_slip_std': self.encoder_noise_angular_slip_std,
            'encoder_noise_linear_additive_std': self.encoder_noise_linear_additive_std,
            'encoder_noise_angular_additive_std': self.encoder_noise_angular_additive_std,
            'encoder_wheel_diameter_ratio_error': self.encoder_wheel_diameter_ratio_error,
            'encoder_wheelbase_ratio': self.encoder_wheelbase_ratio,
            'encoder_noise_correlation_alpha': self.encoder_noise_correlation_alpha,
            'perception_use_geometry_occlusion': self.perception_use_geometry_occlusion,
            'use_nogo_cost': self.use_nogo_cost,
            'nogo_penalty_type': self.nogo_penalty_type,
            'nogo_safe_distance': self.nogo_safe_distance,
            'nogo_logbarrier_eps': self.nogo_logbarrier_eps,
            'nogo_warning_band': self.nogo_warning_band,
            'nogo_near_weight': self.nogo_near_weight,
            'use_belief_nogo_cost': self.use_belief_nogo_cost,
            'nogo_belief_kappa': self.nogo_belief_kappa,
            'use_hit_miss_mixture': self.use_hit_miss_mixture,
            'nogo_mode': self.nogo_mode,
            'yolo_model': self.yolo_model,
            'yolo_model_sha256': _sha256_file(self.yolo_model),
            'yolo_compiled_model': self.yolo_compiled_model,
            'yolo_compiled_model_sha256': _sha256_file(self.yolo_compiled_model),
            'yolo_device': self.yolo_device,
            'yolo_imgsz': self.yolo_imgsz,
            'yolo_conf_threshold': self.yolo_conf_threshold,
            'yolo_predict_conf_floor': self.yolo_predict_conf_floor,
            'yolo_iou_threshold': self.yolo_iou_threshold,
            'yolo_target_class': self.yolo_target_class,
            'yolo_class_id': self.yolo_class_id,
            'yolo_use_masks': self.yolo_use_masks,
            'yolo_min_mask_area_px': self.yolo_min_mask_area_px,
            'yolo_mask_bottom_band_px': self.yolo_mask_bottom_band_px,
            'yolo_max_batch_stamp_skew_s': self.yolo_max_batch_stamp_skew_s,
            'show_pose_markers': self.show_pose_markers,
            'diagnostics_match_tolerance_s': self.diagnostics_match_tolerance_s,
            'bev_y_calibration_offset_m': self.bev_y_calibration_offset_m,
            'bev_affine_calibration': self.bev_affine_calibration,
            'pixel_correction_nis_threshold': self.pixel_correction_nis_threshold,
            'odom_topic': self.odom_topic,
            'seed': self.seed,
            'state_pipeline': 'homography_to_bev',
            'observation_model': 'uv',
            'world_bounds': dict(self.world_bounds),
            'task_start_pose': {
                'x': float(self.task_start_pose[0]),
                'y': float(self.task_start_pose[1]),
                'yaw': float(self.task_start_pose[2]),
            } if self.task_start_pose is not None else None,
            'frame_sanity_start_tolerance_m': self.frame_sanity_start_tolerance_m,
            'frame_sanity_start_tolerance_yaw_rad': self.frame_sanity_start_tolerance_yaw_rad,
            'plan_rate': self.plan_rate,
            'horizon': self.horizon,
            'dt': self.dt,
            'control_weight': self.control_weight,
            'process_noise_xy': self.process_noise_xy,
            'process_noise_theta': self.process_noise_theta,
            'process_noise_model': self.process_noise_model,
            'optimizer_maxiter': self.optimizer_maxiter,
            'optimizer_maxfun': self.optimizer_maxfun,
            'optimizer_ftol': self.optimizer_ftol,
            'optimizer_gtol': self.optimizer_gtol,
            'optimizer_warm_start': self.optimizer_warm_start,
            'optimizer_multistart': self.optimizer_multistart,
            'optimizer_multistart_include_direct': self.optimizer_multistart_include_direct,
            'optimizer_initial_routes_json': self.optimizer_initial_routes_json,
            'optimizer_terminal_goal_tolerance_m': self.optimizer_terminal_goal_tolerance_m,
            'optimizer_route_seed_mode': self.optimizer_route_seed_mode,
            'optimizer_control_block_steps': self.optimizer_control_block_steps,
            'use_hierarchical': self.use_hierarchical,
            'global_planner_mode': self.global_planner_mode,
            'preselected_route_json': self.preselected_route_json,
            'preselected_route_sha256': self.preselected_route_sha256,
            'preselected_route_source_path': self.preselected_route_source_path,
            'preselected_route_source_sha256': self.preselected_route_source_sha256,
            'preselected_route_clearance_m': self.preselected_route_clearance_m,
            'preselected_route_endpoint_tolerance_m': (
                self.preselected_route_endpoint_tolerance_m
            ),
            'preselected_route_sample_step_m': self.preselected_route_sample_step_m,
            'preselected_route_validation_json': self.preselected_route_validation_json,
            'global_horizon': self.global_horizon,
            'global_dt': self.global_dt,
            'local_horizon': self.local_horizon,
            'local_plan_rate': self.local_plan_rate,
            'local_optimizer_maxiter': self.local_optimizer_maxiter,
            'global_use_ambiguity': self.global_use_ambiguity,
            'local_use_ambiguity': self.local_use_ambiguity,
            'local_use_obs_risk': self.local_use_obs_risk,
            'global_optimizer_multistart': self.global_optimizer_multistart,
            'local_optimizer_multistart': self.local_optimizer_multistart,
            'local_use_visibility_model': self.local_use_visibility_model,
            'local_use_belief_nogo_cost': self.local_use_belief_nogo_cost,
            'local_nogo_penalty_type': self.local_nogo_penalty_type,
            'local_nogo_safe_distance': self.local_nogo_safe_distance,
            'local_goal_prior_u_std_start': self.local_goal_prior_u_std_start,
            'local_goal_prior_v_std_start': self.local_goal_prior_v_std_start,
            'local_goal_prior_u_std_final': self.local_goal_prior_u_std_final,
            'local_goal_prior_v_std_final': self.local_goal_prior_v_std_final,
            'waypoint_spacing_m': self.waypoint_spacing_m,
            'waypoint_arrival_radius_m': self.waypoint_arrival_radius_m,
            'local_replan_min_remaining_s': self.local_replan_min_remaining_s,
            'local_replan_on_waypoint_change': self.local_replan_on_waypoint_change,
            'latency_compensate_plan_handoff': self.latency_compensate_plan_handoff,
            'cmd_publish_rate': self.cmd_publish_rate,
            'auto_stop_on_goal': self.auto_stop_on_goal,
            'goal_termination_reference': 'planner_belief',
            'geometry_collision_termination': False,
            'goal_success_radius': self.goal_success_radius,
            'goal_success_hold_s': self.goal_success_hold_s,
            'goal_stable_radius': self.goal_stable_radius,
            'goal_stable_hold_s': self.goal_stable_hold_s,
            'goal_stable_max_displacement_m': self.goal_stable_max_displacement_m,
            'goal_loiter_timeout_s': self.goal_loiter_timeout_s,
            'v_max': self.v_max,
            'use_odom_for_predict': self.use_odom_for_predict,
            'use_diagnostic_odom_localization': self.use_diagnostic_odom_localization,
            'local_controller_type': self.local_controller_type,
            'run_timeout_after_first_cmd_s': self.run_timeout_after_first_cmd_s,
            'first_cmd_linear_eps': self.first_cmd_linear_eps,
            'first_cmd_angular_eps': self.first_cmd_angular_eps,
            'stuck_window_s': self.stuck_window_s,
            'stuck_max_displacement_m': self.stuck_max_displacement_m,
            'stuck_max_goal_improvement_m': self.stuck_max_goal_improvement_m,
            'stuck_cmd_fraction_min': self.stuck_cmd_fraction_min,
            'stuck_idle_cmd_fraction_max': self.stuck_idle_cmd_fraction_max,
            'stuck_turn_progress_min_rad': STUCK_TURN_PROGRESS_MIN_RAD,
            'stuck_angular_cmd_fraction_min': STUCK_ANGULAR_CMD_FRACTION_MIN,
            **manager_settings,
        }
        manifest_data['visibility_artifact_sha256'] = _sha256_file(
            self.visibility_artifact_path)
        manifest_data['camera_network_artifact_sha256'] = _sha256_file(
            self.camera_network_artifact_path)
        actual_camera_network_sha256 = manifest_data['camera_network_artifact_sha256']
        if (self.camera_network_expected_sha256
                and actual_camera_network_sha256 != self.camera_network_expected_sha256):
            raise RuntimeError(
                'camera network artifact SHA-256 differs from the configured expectation')
        manifest_data['camera_network_expected_sha256'] = self.camera_network_expected_sha256
        manifest_data['camera_network_source_hashes'] = dict(
            self.camera_network_source_hashes)
        manifest_data['camera_network_camera_ids'] = list(self.camera_network_camera_ids)
        manifest_data['camera_network_active_camera_ids'] = list(
            self.camera_network_active_camera_ids)
        if self.camera_network_artifact_path:
            if self.camera_network_objective == 'metric_expected_belief':
                with np.load(self.camera_network_artifact_path, allow_pickle=False) as archive:
                    network_metadata = json.loads(str(archive['metadata_json'].item()))
                if network_metadata.get('schema') in (
                        'camera_network.thesis_stage09.v3',
                        'camera_network.final_bayesian_planning.v1',
                        'camera_network.matched_covariance_precision.v1'):
                    manifest_data['planner_field_semantics'] = (
                        'sum of active per-camera precision matrices obtained directly '
                        'from the matched runtime covariance')
                    manifest_data['planner_p_vis_semantics'] = (
                        'not defined; no availability or admission model enters planner precision')
                else:
                    manifest_data['planner_field_semantics'] = (
                        'metric expected belief over independent per-camera Bernoulli reports; '
                        'not the robust runtime fusion posterior')
                    manifest_data['planner_p_vis_semantics'] = (
                        'mean usable-detection probability across artifact cameras')
            else:
                manifest_data['planner_field_semantics'] = 'IWAI detector-score precision proxy; not a measurement covariance or calibrated posterior'
                manifest_data['planner_p_vis_semantics'] = 'mean expected detector score across artifact cameras; not probability of a usable observation'
        manifest_data['manager_commissioned_calibration_sha256'] = _sha256_file(
            str(manager_settings.get('manager_commissioned_calibration_path', '') or '')
        )
        manifest_data['manager_learned_correction_sha256'] = _sha256_file(
            str(manager_settings.get('manager_learned_correction_path', '') or '')
        )
        manifest_data['manager_visibility_sensor_model_sha256'] = _sha256_file(
            str(manager_settings.get('manager_visibility_sensor_model_path', '') or '')
        )
        manifest_data['manager_perception_sensor_model_sha256'] = _sha256_file(
            str(manager_settings.get('manager_perception_sensor_model_path', '') or '')
        )
        manifest_data['manager_sensor_gate_config_sha256'] = _sha256_file(
            str(manager_settings.get('manager_sensor_gate_config_path', '') or '')
        )
        # The commissioned world-plane covariance table, hashed for the same reason: a drive
        # must not be scoreable against a table that has since been refitted.
        manifest_data['manager_commissioned_world_covariance_sha256'] = _sha256_file(
            str(manager_settings.get(
                'manager_commissioned_world_covariance_path', '') or '')
        )
        manifest_data['campaign_config_path'] = self.campaign_config_path
        manifest_data['campaign_config_sha256'] = _sha256_file(
            self.campaign_config_path)
        self._manifest_data = dict(manifest_data)
        write_manifest(self.run_dir, self._manifest_data, repo_root)
        snapshot_configs(
            self.run_dir,
            [self.world_profiles_path, self.tasks_yaml, self.campaign_config_path],
        )

        self.state_msg = None
        self.planner_belief_msg = None
        self.odom_msg = None
        self.odom_noisy_msg = None
        # TRUE Gazebo pose (world frame == map_bev) from /ground_truth_tf, held as
        # latest (x, y). Lets us measure error vs GROUND TRUTH instead of vs /odom,
        # which is DiffDrive wheel odometry and itself drifts in turns.
        self._gt_xy = None
        self._gt_yaw = None  # TRUE heading from /ground_truth_tf (for GT heading error)
        self._gt_stamp = math.nan  # when the truth sample is valid, not the log clock
        self._gt_stamp_source = 'none'
        #: gaps between consecutive truth samples, so the residual alignment error left
        #: by stamping at receipt is measured in every run instead of being assumed
        self._gt_intervals = deque(maxlen=20000)
        # Buffer of (stamp_s, x, y, yaw) GROUND TRUTH, so an estimate can be scored
        # against the truth AT THE INSTANT IT DESCRIBES rather than at log time.
        #
        # Without this every error column paired a timestamped estimate with a LATER
        # truth: the belief publishes at 10 Hz and the logger samples 10 Hz one cycle
        # behind it, so `belief_error_gt_m` carried a fixed 100 ms of robot travel.
        # Measured on the six-arm drives that is 2.3x the real median error (2.75 cm
        # logged against 1.13 cm aligned) and 1.9x the NEES. The offset is nearly
        # identical on every arm, so it looked like a property of the camera network.
        self._gt_buf = deque(maxlen=2000)
        # Buffer of (stamp_s, x, y, yaw) WHEEL ODOMETRY in the map_bev frame. Kept for
        # the odometry-referenced diagnostics only. It is NOT truth: measured drift
        # against the Gazebo pose is 24 cm median and 2.4 m worst on these drives.
        self._odom_map_buf = deque(maxlen=600)
        self.obs_msg = None
        self.perception_diag = None
        self.heading_diag = None
        self.pixel_correction_diag = None
        self._assimilation_source_batches = set()
        self._assimilation_payloads = {}
        self._assimilation_count = 0
        self._assimilation_dropped_count = 0
        self.cmd_msg = None
        self.cmd_raw_msg = None
        self.cmd_noise_diag = None
        self.cmd_stamp_s = math.nan
        self.cmd_raw_stamp_s = math.nan
        self.cmd_noise_diag_stamp_s = math.nan
        self.goal_msg = None
        self.plan_msg = None
        self.planner_diag = None
        self.planner_diag_text = ''
        self.active_execution_diag = None
        self.efe_metrics = None
        self._goal_in_radius_since = None
        self._goal_stable_since = None
        self._goal_loiter_since = None
        self._goal_region_entered = False
        self._goal_region_first_stamp = math.nan
        self._motion_history = deque()
        self._stop_requested = False
        self._completed = False
        self._finalizing = False
        self._accepting_events = True
        self._event_streams_closed = False
        self._event_lock = threading.RLock()
        # Finalization runs on a background timer.  Serialize it with the periodic
        # CSV callback so a row cannot be assembled while the underlying stream is
        # being closed.
        self._log_write_lock = threading.RLock()
        self._last_event_wall_s = time.monotonic()
        self._finish_requested_wall_s = math.nan
        self._finish_reason = ''
        self._finish_stamp = math.nan
        self._late_event_count = 0
        self._runtime_event_counts = {}
        self._runtime_event_invalid_count = 0
        self._fusion_decision_payloads = {}
        self._correction_publications = {}
        self._data_file_close_errors = []
        self._terminal_stop_verified = False
        self._terminal_stop_event_id = ''
        self._terminal_zero_forwarded = False
        self._terminal_zero_stamp_s = math.nan
        self._terminal_stop_request_id = ''
        self._terminal_stop_request_payload = None
        self._terminal_stop_acks = {}
        self._mission_goal_state = None
        self._active_mission_goal_id = ''
        self._last_tf_warn_wall = 0.0
        self._frame_sanity_logged = False
        self._frame_sanity = {
            'recorded': False,
            'ok': None,
            'reason': 'pending',
            'source_frame': '',
            'odom_map_stamp': math.nan,
            'raw_odom_x': math.nan,
            'raw_odom_y': math.nan,
            'raw_odom_yaw': math.nan,
            'odom_map_x': math.nan,
            'odom_map_y': math.nan,
            'odom_map_yaw': math.nan,
            'task_start_x': float(self.task_start_pose[0]) if self.task_start_pose is not None else math.nan,
            'task_start_y': float(self.task_start_pose[1]) if self.task_start_pose is not None else math.nan,
            'task_start_yaw': float(self.task_start_pose[2]) if self.task_start_pose is not None else math.nan,
            'odom_map_start_error_m': math.nan,
            'raw_start_error_m': math.nan,
            'odom_map_start_yaw_error_rad': math.nan,
            'raw_start_yaw_error_rad': math.nan,
            'tolerance_m': self.frame_sanity_start_tolerance_m,
            'tolerance_yaw_rad': self.frame_sanity_start_tolerance_yaw_rad,
        }
        self._rewrite_manifest()

        self._first_cmd_stamp = None
        self._cumulative_path_length = 0.0
        self._last_path_pose = None
        self._min_goal_distance = float('inf')
        self._valid_run = True
        self._invalid_reason = ''
        #: reason -> count, for the refusal rate reported beside the accuracy
        self._assimilation_dropped_reasons: dict[str, int] = {}
        #: The worst stretch this drive went without a usable correction, in seconds.
        #: A property of the camera network on this route, and a headline for the
        #: availability work -- not a defect.
        self._longest_correction_gap_s = 0.0
        self._last_correction_stamp_s = None

        self._efe_risk_sum = 0.0
        self._efe_ambiguity_sum = 0.0
        self._efe_control_sum = 0.0
        self._efe_obstacle_sum = 0.0
        self._efe_count = 0
        self._solve_time_ms_sum = 0.0
        self._solve_count = 0
        self._p_vis_plan_sum = 0.0
        self._p_vis_plan_eff_sum = 0.0
        self._r_plan_u_std_sum = 0.0
        self._r_plan_v_std_sum = 0.0
        self._p_vis_count = 0
        self._p_vis_plan_below_0_2_count = 0
        self._p_vis_plan_eff_below_0_2_count = 0
        self._max_r_plan_std = 0.0
        self._state_error_odom_sum = 0.0
        self._belief_error_odom_sum = 0.0
        self._state_error_odom_count = 0
        self._belief_error_odom_count = 0
        self._state_error_odom_after_first_cmd_sum = 0.0
        self._belief_error_odom_after_first_cmd_sum = 0.0
        self._state_error_odom_after_first_cmd_count = 0
        self._belief_error_odom_after_first_cmd_count = 0
        self._odom_map_vs_odom_yaw_error_sum = 0.0
        self._odom_map_vs_state_yaw_error_sum = 0.0
        self._odom_map_vs_belief_yaw_error_sum = 0.0
        self._odom_map_vs_odom_yaw_error_count = 0
        self._truth_state_yaw_error_count = 0
        self._truth_belief_yaw_error_count = 0
        self._odom_map_vs_odom_yaw_error_after_first_cmd_sum = 0.0
        self._odom_map_vs_state_yaw_error_after_first_cmd_sum = 0.0
        self._odom_map_vs_belief_yaw_error_after_first_cmd_sum = 0.0
        self._odom_map_vs_odom_yaw_error_after_first_cmd_count = 0
        self._odom_map_vs_state_yaw_error_after_first_cmd_count = 0
        self._odom_map_vs_belief_yaw_error_after_first_cmd_count = 0
        # GROUND-TRUTH error means (vs the real Gazebo pose gt_x/gt_y), the honest
        # counterpart to the odom-based mean_truth_* means above.
        self._belief_error_gt_sum = 0.0
        self._belief_error_gt_count = 0
        self._state_error_gt_sum = 0.0
        self._state_error_gt_count = 0
        self._belief_error_gt_after_first_cmd_sum = 0.0
        self._belief_error_gt_after_first_cmd_count = 0
        self._state_error_gt_after_first_cmd_sum = 0.0
        self._state_error_gt_after_first_cmd_count = 0
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        run_dir_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.run_dir_pub = self.create_publisher(String, self.run_dir_topic, qos_profile=run_dir_qos)
        run_dir_msg = String()
        run_dir_msg.data = self.run_dir
        self.run_dir_pub.publish(run_dir_msg)

        goal_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        #: the manager's latest decision: which cameras went into the correction, and when
        self.fusion_decision = None
        self.fusion_decision_stamp = None
        self.runtime_event_path = os.path.join(
            self.run_dir, 'runtime_event_deliveries.jsonl')
        self.runtime_event_file = open(self.runtime_event_path, 'w', encoding='utf-8')
        self.runtime_event_log = JsonlDeliveryLog(self.runtime_event_file)
        self.belief_prediction_path = os.path.join(
            self.run_dir, 'belief_predictions.jsonl')
        self.belief_prediction_file = open(
            self.belief_prediction_path, 'w', encoding='utf-8')
        self._belief_prediction_count = 0
        self._belief_prediction_invalid_count = 0
        event_qos = QoSProfile(
            depth=4096,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        compatible_event_qos = QoSProfile(
            depth=4096,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        terminal_qos = QoSProfile(
            depth=16,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._terminal_stop_request_pub = self.create_publisher(
            String, TERMINAL_STOP_REQUEST_TOPIC, terminal_qos
        )
        self.create_subscription(
            String,
            TERMINAL_STOP_ACK_TOPIC,
            self._terminal_stop_ack_cb,
            terminal_qos,
        )
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.create_subscription(Odometry, '/odom_noisy', self._odom_noisy_cb, 10)
        self.create_subscription(TFMessage, '/ground_truth_tf', self._ground_truth_cb, 50)
        self.create_subscription(PoseWithCovarianceStamped, '/state/bev', self._state_cb, 10)
        self.create_subscription(
            String, '/reliability/camera_manager/decision', self._fusion_decision_cb, 10)
        self.create_subscription(
            String, '/reliability/camera_manager/fused_correction',
            self._fused_correction_cb, 100)
        for topic in ('/perception/camera_batch_outcome', MISSION_GOAL_TOPIC):
            self.create_subscription(
                String, topic,
                lambda msg, name=topic: self._runtime_outcome_cb(name, msg),
                event_qos,
            )
        # These active publishers are volatile today. A volatile request remains
        # compatible if a producer is upgraded to transient-local later, whereas a
        # transient-local request silently disconnects from the current bridge/manager.
        for topic in ('/reliability/camera_manager/batch_outcome',
                      '/sim/actuation_outcome'):
            self.create_subscription(
                String, topic,
                lambda msg, name=topic: self._runtime_outcome_cb(name, msg),
                compatible_event_qos,
            )
        self.create_subscription(Float64MultiArray, '/state/heading_diagnostics', self._heading_diag_cb, 10)
        self.create_subscription(String, '/planner/belief_state', self._belief_state_cb, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/planner_belief', self._planner_belief_cb, 10)
        self.create_subscription(PoseStamped, '/perception/pixel_pose', self._obs_cb, 10)
        self.create_subscription(
            Float64MultiArray,
            DETECTION_DIAGNOSTICS_TOPIC,
            self._diag_cb,
            10,
        )
        self.create_subscription(Twist, '/cmd_vel', self._cmd_cb, 10)
        self.create_subscription(Twist, '/cmd_vel_raw', self._cmd_raw_cb, 10)
        self.create_subscription(Float64MultiArray, '/cmd_vel_noise/diagnostics', self._cmd_noise_diag_cb, 10)
        self.create_subscription(PoseStamped, '/goal_bev', self._goal_cb, qos_profile=goal_qos)
        self.create_subscription(Path, '/plan_preview', self._plan_cb, 10)
        self.create_subscription(Float64MultiArray, '/planner/diagnostics', self._planner_diag_cb, 10)
        self.create_subscription(
            Float64MultiArray,
            '/planner/active_execution_diagnostics',
            self._active_execution_diag_cb,
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            '/planner/pixel_correction_diagnostics',
            self._pixel_correction_diag_cb,
            10,
        )
        self.create_subscription(
            String,
            '/planner/correction_assimilation',
            self._correction_assimilation_cb,
            10,
        )
        self.create_subscription(String, '/planner/diagnostics_text', self._planner_diag_text_cb, 10)
        self.create_subscription(Float64MultiArray, '/efe/metrics', self._efe_cb, 10)

        self.file = open(self.log_path, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            'stamp',
            'odom_map_available', 'odom_map_stamp', 'odom_map_x', 'odom_map_y', 'odom_map_yaw',
            'state_available', 'state_stamp', 'state_x', 'state_y', 'state_yaw',
            'state_age_s', 'state_fresh',
            'state_cov_xx', 'state_cov_xy', 'state_cov_yy', 'state_cov_yaw',
            'planner_belief_available', 'planner_belief_stamp',
            'planner_belief_age_s',
            'planner_belief_x', 'planner_belief_y', 'planner_belief_yaw',
            'planner_cov_x', 'planner_cov_xy', 'planner_cov_y', 'planner_cov_yaw',
            'est_available', 'est_x', 'est_y', 'est_yaw',
            'est_cov_xx', 'est_cov_xy', 'est_cov_yy',
            'state_pos_error_m', 'state_cov_trace', 'state_cov_det',
            'state_sigma_major_m', 'state_sigma_minor_m', 'state_entropy_xy',
            # Explicit unambiguous error columns:
            # state_error_odom_m  = ||truth - /state/bev||   (perception estimate vs ground truth)
            # belief_error_odom_m = ||truth - /planner_belief||  (planner internal state vs ground truth)
            'state_error_odom_m', 'belief_error_odom_m',
            'odom_available', 'odom_stamp', 'odom_x', 'odom_y', 'odom_yaw',
            'odom_v', 'odom_w',
            'odom_noisy_available', 'odom_noisy_stamp', 'odom_noisy_x', 'odom_noisy_y',
            'odom_noisy_yaw', 'odom_noisy_v', 'odom_noisy_w',
            'yaw_error_odom_map_vs_odom_rad', 'yaw_error_odom_map_vs_state_rad', 'yaw_error_odom_map_vs_belief_rad',
            'pixel_yaw_meas', 'heading_source_code', 'heading_source',
            'heading_diag_stamp', 'heading_diag_age_s',
            'state_heading_yaw_sigma', 'state_heading_odom_age_s',
            'planner_pixel_correction_available', 'planner_pixel_correction_stamp',
            'planner_pixel_correction_age_s',
            'pixel_corr_innov_u', 'pixel_corr_innov_v',
            'pixel_corr_xy_update_norm_m', 'pixel_corr_theta_update_from_uv_rad',
            'pixel_corr_nis',
            'pixel_corr_accepted', 'pixel_corr_reject_reason_code', 'pixel_corr_reject_reason',
            'pixel_corr_apply_stamp', 'pixel_corr_belief_input_stamp',
            'pixel_corr_cmd_replay_count', 'pixel_corr_cmd_replay_duration_s',
            'pixel_corr_cmd_replay_used_fallback',
            'pixel_corr_motion_replay_source_code', 'pixel_corr_motion_replay_source',
            'pixel_corr_nis_threshold',
            'pixel_heading_correction_applied', 'pixel_heading_meas_source',
            'pixel_heading_innov_rad',
            'pixel_heading_gain_theta', 'pixel_corr_theta_update_total_rad',
            'pixel_corr_pred_x', 'pixel_corr_pred_y', 'pixel_corr_pred_yaw',
            'pixel_corr_next_x', 'pixel_corr_next_y', 'pixel_corr_next_yaw',
            'pixel_corr_expected_after_u', 'pixel_corr_expected_after_v',
            'pixel_corr_expected_after_visible',
            'cmd_v', 'cmd_w',
            'cmd_raw_v', 'cmd_raw_w',
            'cmd_stamp', 'cmd_age_s', 'cmd_raw_stamp', 'cmd_raw_age_s',
            'cmd_noise_enabled',
            'cmd_noise_linear_multiplier', 'cmd_noise_angular_multiplier',
            'cmd_noise_linear_additive', 'cmd_noise_angular_additive',
            'cmd_noise_v_error', 'cmd_noise_w_error',
            'goal_x', 'goal_y', 'goal_dist',
            'operational_goal_dist_m', 'goal_termination_reference',
            'plan_points', 'plan_length',
            'optimizer_success', 'optimizer_status', 'optimizer_nit', 'optimizer_nfev', 'optimizer_message',
            'plan_time_ms', 'solve_time_ms',
            'measurement_available', 'belief_age_s',
            'p_vis_plan', 'p_vis_plan_eff',
            'r_plan_u_std', 'r_plan_v_std',
            'terminal_goal_distance_pred', 'terminal_goal_progress_m',
            'fraction_horizon_low_pvis', 'fraction_horizon_high_ambiguity',
            'min_predicted_obstacle_distance_m', 'rollout_valid',
            'efe_total', 'efe_risk', 'efe_ambiguity', 'efe_control', 'efe_obstacle',
            'efe_risk_mean', 'efe_risk_cov_trace', 'efe_risk_cov_logdet',
            'efe_delta_risk_visibility', 'efe_delta_ambiguity_visibility',
            'active_plan_age_s', 'active_plan_remaining_s', 'active_control_index',
            'active_controls_len', 'active_controls_original_len',
            'latency_skip_steps', 'latency_skip_s',
            'command_timer_period_s', 'planner_timer_period_s',
            'pending_plan_started_active_remaining_s',
            'exec_plan_age_s', 'exec_plan_remaining_s', 'exec_control_index',
            'exec_controls_len', 'exec_controls_original_len',
            'exec_cmd_v', 'exec_cmd_w', 'exec_latency_skip_steps',
            'exec_latency_skip_s',
            'exec_wp_idx', 'exec_wp_count', 'exec_wp_target_x', 'exec_wp_target_y',
            'exec_wp_dist_m', 'exec_desired_yaw', 'exec_yaw_error',
            'exec_tracking_yaw', 'exec_tracking_yaw_source',
            'valid_run', 'invalid_reason',
            'heading_update_mode',
            'pixel_corr_K_theta_u', 'pixel_corr_K_theta_v',
            # Shared single-cam/multicam correction chain (2026-07-29): innov and
            # meas are PIXELS when measurement_space==0, METRES when ==1.
            'pixel_corr_measurement_space', 'pixel_corr_predict_clipped_m',
            'pixel_corr_camera_index',
            'yaw_error_odom_noisy_vs_odom_map_rad',
            'state_bev_yaw_latest',
            'state_bev_cov_theta_theta', 'state_bev_cov_x_theta', 'state_bev_cov_y_theta',
            'planner_belief_cov_theta_theta', 'planner_belief_cov_x_theta', 'planner_belief_cov_y_theta',
            'planner_diag_prediction_source', 'planner_diag_prediction_dt',
            'planner_diag_u_pred_v', 'planner_diag_u_pred_omega', 'planner_diag_Q_theta_theta',
            'planner_diag_odom_delta_theta', 'planner_diag_cmd_delta_theta',
            'planner_diag_heading_anchor_applied', 'planner_diag_state_bev_yaw_ignored',
            # Ground-truth (vs TRUE Gazebo pose, not /odom wheel odometry).
            #
            # gt_x/gt_y/gt_yaw are the LATEST truth, valid at gt_stamp (its own stamp,
            # not the log clock). belief_error_gt_m and state_error_gt_m are scored
            # against the truth at the ESTIMATE's stamp, and gt_*_at_*_stamp are the
            # exact truth they used so any analysis can re-derive them. The
            # *_logtime_m twins are the old, latency-inflated definition, kept as a
            # diagnostic and for comparison with pre-2026-08-28 runs.
            'gt_available', 'gt_x', 'gt_y', 'gt_yaw', 'gt_stamp', 'gt_age_s',
            'belief_error_gt_m', 'state_error_gt_m', 'odom_map_gt_drift_m',
            'belief_yaw_error_gt_rad',
            'gt_x_at_belief_stamp', 'gt_y_at_belief_stamp',
            'gt_x_at_state_stamp', 'gt_y_at_state_stamp',
            'belief_error_gt_logtime_m', 'state_error_gt_logtime_m',
            'seed',
            # How many cameras actually went into the correction the filter just used, and
            # which ones. Without this the fusion comparison cannot plot error and claimed
            # uncertainty against the number of contributing cameras -- which is the axis the
            # whole experiment turns on. Read from the manager's own decision message, so it
            # is what happened rather than what the commissioned map expected.
            'fusion_cameras_n', 'fusion_cameras', 'fusion_decision_age_s',
            # candidates = cameras available to the rule; cameras = the ones it used. For a
            # single-best rule those differ by construction, so the cross-arm plot is read
            # against the candidates.
            'fusion_candidates_n', 'fusion_candidates'
        ])

        # One row per CAMERA per fused correction: where that camera put the robot, how sure
        # it was, whether the arm's rule used it, and what the rule produced from them all.
        # This is the mechanism the fusion figures draw; the main CSV only carries the result.
        self.fusion_obs_path = os.path.join(self.run_dir, 'fusion_observations.csv')
        self.fusion_obs_file = open(self.fusion_obs_path, 'w', newline='')
        self.fusion_obs_writer = csv.writer(self.fusion_obs_file)
        self.fusion_obs_writer.writerow([
            # `stamp` is when the DECISION reached the logger. The manager decides at
            # 20 Hz while the detector produces 5 Hz, so the same physical detection
            # appears on about four consecutive decisions. Counting rows therefore
            # counts each reading ~4x: measured 25656 rows for 6418 distinct readings
            # across five drives. `obs_repeat` == 0 selects one row per detection;
            # `obs_seq` numbers the distinct detections per camera. Every per-camera
            # statistic (n, bias, spread, NEES, any likelihood) must filter on
            # obs_repeat == 0, or it reports a quarter of its effective sample size as
            # four times as many independent readings.
            'stamp', 'decision_seq', 'source_batch_id', 'common_capture_stamp',
            'camera', 'used', 'obs_x', 'obs_y',
            'obs_cov_xx', 'obs_cov_xy', 'obs_cov_yy',
            'aligned_x', 'aligned_y',
            'aligned_cov_xx', 'aligned_cov_xy', 'aligned_cov_yy',
            'n_candidates', 'n_used',
            'fused_x', 'fused_y', 'fused_cov_xx', 'fused_cov_xy', 'fused_cov_yy',
            # Three different instants, so three different truths. `obs_x/obs_y`
            # describe the robot at `obs_stamp` (capture). `fused_x/fused_y` describe
            # it at `fused_stamp`, because the manager propagates the fused correction
            # forward and re-stamps it. `gt_x/gt_y` is the latest truth when the
            # decision arrived here. Scoring a camera against `gt_x` charges it the
            # whole pipeline delay -- ~200 ms, about 4.4 cm at 0.22 m/s -- and scoring
            # a camera and the fused answer against the SAME `gt_x` is what made the
            # fusion rule look better than the cameras it was combining.
            'gt_x', 'gt_y', 'gt_stamp',
            'gt_x_at_obs', 'gt_y_at_obs',
            'fused_stamp', 'gt_x_at_fused', 'gt_y_at_fused',
            'obs_seq', 'obs_repeat',
            # What the detector said about this reading. Recorded so the question "does the
            # detector's own confidence predict how wrong it was?" can be answered from a
            # drive; nothing in the runtime weights anything by these.
            'conf', 'conf_raw', 'bbox_h_px', 'bbox_w_px',
            # The box the hull model predicted from the pose the correction was made
            # from. Detected over predicted is the height ratio; without this column it
            # cannot be reconstructed from a drive.
            'pred_h_px', 'pred_w_px',
            # The UNCORRECTED back-projection of the same box, before the runtime
            # observation model rewrote it. `obs_x/obs_y` is what the steering model
            # decided the reading means; these two are what the camera actually saw. With
            # both, one recorded drive supports replaying any interpretation on identical
            # readings, instead of needing a separate drive per interpretation.
            'raw_obs_x', 'raw_obs_y',
            'range_m', 'obs_stamp',
        ])
        self._fusion_obs_last_stamp = None
        self._fusion_decision_seq = 0
        #: (camera, obs_stamp) -> how many times that detection has been written
        self._obs_repeat_count = {}
        #: camera -> how many DISTINCT detections it has contributed
        self._obs_seq_by_camera = {}

        # One terminal filter outcome per detector batch. This is the causal
        # join used by scoring and run-validity checks.
        # Every true robot pose sample, in arrival order, for the offline collision score
        # (footprint vs driveable region). Offline evaluation only: nothing reads it at runtime.
        self.ground_truth_pose_path = os.path.join(self.run_dir, 'ground_truth_pose.csv')
        self.ground_truth_pose_file = open(self.ground_truth_pose_path, 'w', newline='')
        self.ground_truth_pose_writer = csv.writer(self.ground_truth_pose_file)
        self.ground_truth_pose_writer.writerow(['stamp_s', 'stamp_source', 'x', 'y', 'yaw'])

        self.assimilation_path = os.path.join(
            self.run_dir, 'correction_assimilations.csv')
        self.assimilation_file = open(self.assimilation_path, 'w', newline='')
        self.assimilation_writer = csv.writer(self.assimilation_file)
        self.assimilation_writer.writerow([
            'source_batch_id', 'correction_stamp', 'apply_stamp',
            'status', 'reason', 'accepted', 'nis', 'belief_stamp_after',
            'schema_version', 'epoch', 'revision_before', 'revision_after',
            'frame_id', 'state_stamp_ns', 'posterior_mean',
            'posterior_covariance', 'valid', 'motion_supported',
            'logger_receive_stamp', 'correction_stamp_ns', 'apply_stamp_ns',
            'belief_stamp_before_ns', 'belief_stamp_after_ns',
            'source_event_id', 'source_epoch', 'source_publication_seq',
            'source_member_ids', 'source_payload_sha256',
        ])

        # The actual identity-bearing correction envelope is a distinct boundary
        # from the manager's later display/diagnostic decision message.
        self.correction_publication_path = os.path.join(
            self.run_dir, 'correction_publications.csv')
        self.correction_publication_file = open(
            self.correction_publication_path, 'w', newline='')
        self.correction_publication_writer = csv.writer(self.correction_publication_file)
        self.correction_publication_writer.writerow([
            'source_batch_id', 'event_id', 'epoch', 'publication_seq',
            'correction_stamp', 'correction_stamp_ns', 'common_capture_stamp',
            'common_capture_stamp_ns', 'publication_stamp_ns', 'logger_receive_stamp',
            'frame_id', 'schema_version', 'accepted_camera_ids', 'member_ids',
            'payload_sha256',
        ])

        self.plan_file = None
        self.plan_writer = None
        if self.log_plan_samples:
            self.plan_log_path = os.path.join(self.run_dir, 'plan_samples.csv')
            self.plan_file = open(self.plan_log_path, 'w', newline='')
            self.plan_writer = csv.writer(self.plan_file)
            self.plan_writer.writerow(['plan_stamp', 'point_idx', 'x', 'y'])

        self.perception_file = None
        self.perception_writer = None
        if self.log_perception_samples:
            self.perception_log_path = os.path.join(self.run_dir, 'perception.csv')
            self.perception_file = open(self.perception_log_path, 'w', newline='')
            self.perception_writer = csv.writer(self.perception_file)
            self.perception_writer.writerow([
                'diag_stamp',
                'log_stamp',
                'detected',
                'true_available',
                'true_x',
                'true_y',
                'true_yaw',
                'state_available',
                'state_x',
                'state_y',
                'state_yaw',
                'state_age_s',
                'state_fresh',
                'state_pos_error',
                'state_yaw_error_deg',
                'obs_u',
                'obs_v',
                'obs_yaw',
                'obs_yaw_error_deg',
                'pixel_pose_available',
                'pixel_pose_stamp',
                'pixel_pose_u',
                'pixel_pose_v',
                'pixel_pose_yaw',
                'pixel_pose_age_s',
                'pixel_pose_fresh',
                'pred_world_x',
                'pred_world_y',
                'localization_error_m',
                'pred_world_x_calibrated',
                'pred_world_y_calibrated',
                'localization_error_calibrated_m',
                # vs truth at the measurement's OWN capture time (latency-removed):
                # true detector quality. The _calibrated_m above is vs log-time truth
                # (latency-inflated in turns). state_error_captime_m = same for /state.
                'localization_error_captime_m',
                'state_error_captime_m',
                'bev_y_calibration_offset_m',
                'u_red',
                'v_red',
                'red_area_px',
                'u_blue',
                'v_blue',
                'blue_area_px',
                'separation_px',
                'border_margin_px',
                'yolo_score_raw',
                'yolo_score_selected',
                'yolo_detected_after_threshold',
                'yolo_best_class_id',
                'yolo_target_candidate_count',
                'bbox_area_px',
                'bbox_xmin',
                'bbox_ymin',
                'bbox_xmax',
                'bbox_ymax',
                'logit_margin',
                'class_entropy',
                'mask_area_px',
                'mask_bottom_u',
                'mask_bottom_v',
                'mask_used',
                'mask_polygon_points',
                'confidence_logit',
                'mask_compactness',
                'mask_border_frac',
                'mask_score',
                'selected_pixel_source_code',
                'yolo_raw_best_score',
                'yolo_selected_score',
                'yolo_num_target_candidates',
                'yolo_selected_class_id',
                'yolo_selected_pixel_source',
                'yolo_bbox_area',
                'yolo_mask_area',
                'yolo_inference_ms',
                'detector_callback_ms',
                'yolo_receive_stamp',
                'yolo_start_stamp',
                'yolo_finish_stamp',
                'yolo_publish_stamp',
                'yolo_latency_s',
                'frame_age_at_publish_s',
                'detector_total_latency_s',
                'camera_relative_bearing_deg',
                'seed',
            ])

        self.camera_opportunity_file = open(
            os.path.join(self.run_dir, 'camera_opportunities.jsonl'), 'w', encoding='utf-8')
        self.camera_opportunity_log = CameraOpportunityLog(self.camera_opportunity_file)
        self._camera_opportunity_subscriptions = [
            self.create_subscription(
                String, f'/perception/camera_observation/{camera}',
                lambda msg, c=camera: self._camera_opportunity_cb(c, msg),
                100)
            for camera in ('camera_A', 'camera_B', 'camera_C', 'camera_D', 'camera_E')
        ]

        rate = float(self.get_parameter('log_rate').value)
        self.create_timer(1.0 / max(rate, 0.1), self._log_once)
        self.get_logger().info(
            f'Experiment logger writing to {self.log_path} '
            f'(method={self.method or self.planner}, world={self.world}, task={self.task})'
        )
        self.get_logger().info(
            'State-estimator provenance: '
            f'mode={self.state_estimator_mode}, '
            f'x={self.state_source_x}, y={self.state_source_y}, theta={self.state_source_theta}'
        )
        if self.perception_file is not None:
            self.get_logger().info(f'Perception samples writing to {self.perception_log_path}')
        if self.auto_stop_on_goal:
            self.get_logger().info(
                f"Auto-stop enabled: goal radius <= {self.goal_success_radius:.3f} m "
                f"for {self.goal_success_hold_s:.2f} s; stable/idle goal radius <= "
                f"{self.goal_stable_radius:.3f} m for {self.goal_stable_hold_s:.2f} s"
            )
        if self.stuck_window_s > 0.0:
            self.get_logger().info(
                f"Stuck-stop enabled: {self.stuck_window_s:.2f}s window, "
                f"max displacement {self.stuck_max_displacement_m:.3f} m, "
                f"max goal improvement {self.stuck_max_goal_improvement_m:.3f} m, "
                f"active cmd fraction >= {self.stuck_cmd_fraction_min:.2f}, "
                f"idle cmd fraction <= {self.stuck_idle_cmd_fraction_max:.2f}"
            )

    @staticmethod
    def _stamp_to_float(stamp_msg) -> float:
        return float(stamp_msg.sec) + float(stamp_msg.nanosec) * 1e-9

    @staticmethod
    def _parse_bev_affine(raw: str):
        return parse_bev_affine_calibration(raw)

    def _apply_bev_calibration(self, x: float, y: float) -> tuple[float, float]:
        if self._bev_affine is not None:
            c = self._bev_affine
            return (
                c[0] * float(x) + c[1] * float(y) + c[2],
                c[3] * float(x) + c[4] * float(y) + c[5],
            )
        return float(x), float(y) + self.bev_y_calibration_offset_m

    @staticmethod
    def _yaw_from_quaternion(q) -> float:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def _covariance_metrics_2d(cov_xx: float, cov_xy: float, cov_yy: float):
        if not (math.isfinite(cov_xx) and math.isfinite(cov_yy)):
            return math.nan, math.nan, math.nan, math.nan, math.nan
        cov_xy = float(cov_xy) if math.isfinite(cov_xy) else 0.0
        trace = float(cov_xx + cov_yy)
        det = float(cov_xx * cov_yy - cov_xy * cov_xy)
        sigma_major = math.nan
        sigma_minor = math.nan
        entropy_xy = math.nan
        try:
            evals = np.linalg.eigvalsh(np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=float))
            evals = np.clip(np.asarray(evals, dtype=float), 0.0, None)
            sigma_minor = float(math.sqrt(evals[0]))
            sigma_major = float(math.sqrt(evals[1]))
        except np.linalg.LinAlgError:
            pass
        if det > 0.0:
            entropy_xy = float(0.5 * math.log(((2.0 * math.pi * math.e) ** 2) * det))
        return trace, det, sigma_major, sigma_minor, entropy_xy

    def _command_active(self, cmd_v: float, cmd_w: float) -> bool:
        return (
            abs(float(cmd_v)) >= self.first_cmd_linear_eps
            or abs(float(cmd_w)) >= self.first_cmd_angular_eps
        )

    def _remember_motion_sample(
        self,
        stamp: float,
        operational_x: float,
        operational_y: float,
        goal_dist: float,
        cmd_v: float,
        cmd_w: float,
        operational_yaw: float = math.nan,
    ) -> None:
        # Use the same operational state available to the controller. Ground
        # truth is evaluation-only and cannot decide when a run stops.
        # A goal is not required for terminal-rest verification: externally
        # controlled commissioning runs intentionally have enable_mission=false.
        if not (
            math.isfinite(stamp)
            and math.isfinite(operational_x)
            and math.isfinite(operational_y)
        ):
            return
        self._motion_history.append((
            float(stamp),
            float(operational_x),
            float(operational_y),
            float(goal_dist),
            1.0 if self._command_active(cmd_v, cmd_w) else 0.0,
            float(operational_yaw),
            1.0 if abs(float(cmd_w)) >= self.first_cmd_angular_eps else 0.0,
        ))
        keep_window_s = max(
            float(self.stuck_window_s),
            float(self.goal_success_hold_s),
            float(self.goal_stable_hold_s),
            1.0,
        ) + 1.0
        while self._motion_history and stamp - self._motion_history[0][0] > keep_window_s:
            self._motion_history.popleft()

    def _motion_window_stats(self, stamp: float, window_s: float):
        if window_s <= 0.0 or len(self._motion_history) < 2:
            return None
        window_start = stamp - window_s
        samples = [sample for sample in self._motion_history if sample[0] >= window_start]
        if len(samples) < 2:
            return None
        duration_s = float(samples[-1][0] - samples[0][0])
        if duration_s < min(max(0.5 * window_s, 0.5), window_s):
            return None
        displacement_m = float(math.hypot(samples[-1][1] - samples[0][1], samples[-1][2] - samples[0][2]))
        goal_improvement_m = float(samples[0][3] - samples[-1][3])
        cmd_fraction = float(sum(sample[4] for sample in samples) / len(samples))
        yaw_samples = [
            sample for sample in samples
            if len(sample) >= 7 and math.isfinite(sample[5])
        ]
        yaw_progress_rad = 0.0
        if len(yaw_samples) >= 2:
            yaw_progress_rad = abs(sum(
                self._wrap_angle(current[5] - previous[5])
                for previous, current in zip(yaw_samples[:-1], yaw_samples[1:])
            ))
        angular_cmd_fraction = float(
            sum(sample[6] for sample in samples if len(sample) >= 7) / len(samples)
        )
        return {
            'duration_s': duration_s,
            'displacement_m': displacement_m,
            'goal_improvement_m': goal_improvement_m,
            'cmd_fraction': cmd_fraction,
            'yaw_progress_rad': float(yaw_progress_rad),
            'angular_cmd_fraction': angular_cmd_fraction,
        }

    def _update_terminal_stop_verification(self, stamp: float) -> None:
        """Verify rest without consulting ground truth or redefining stop time."""
        if (not self._stop_requested or not self._terminal_zero_forwarded
                or not math.isfinite(self._terminal_zero_stamp_s)):
            return
        samples = [
            sample for sample in self._motion_history
            if sample[0] >= self._terminal_zero_stamp_s
        ]
        # The authoritative actuation-outcome zero and the cached /cmd_vel
        # subscription can arrive in either callback order.  A stale nonzero
        # sample after the forwarded-zero timestamp must delay the rest window,
        # not poison every later window permanently.  Start after the last
        # observed active command and still require a full idle interval.
        last_active = max(
            (index for index, sample in enumerate(samples) if sample[4] != 0.0),
            default=-1,
        )
        samples = samples[last_active + 1:]
        if len(samples) < 2:
            return
        duration_s = float(samples[-1][0] - samples[0][0])
        if duration_s < TERMINAL_REST_WINDOW_S:
            return
        x0, y0 = samples[0][1], samples[0][2]
        max_displacement_m = max(
            math.hypot(sample[1] - x0, sample[2] - y0) for sample in samples
        )
        commands_idle = all(sample[4] == 0.0 for sample in samples)
        if commands_idle and max_displacement_m <= TERMINAL_REST_MAX_DISPLACEMENT_M:
            self._terminal_stop_verified = True

    def _update_goal_region_state(self, stamp: float, goal_dist: float) -> None:
        if not (math.isfinite(stamp) and math.isfinite(goal_dist)):
            return
        if goal_dist <= self.goal_success_radius and not self._goal_region_entered:
            self._goal_region_entered = True
            self._goal_region_first_stamp = float(stamp)

    def _goal_region_reached(self) -> bool:
        return bool(self._goal_region_entered)

    def _maybe_finish_for_goal(self, stamp: float, goal_dist: float, cmd_v: float, cmd_w: float) -> bool:
        mission = getattr(self, '_mission_goal_state', None)
        if mission is not None and (not mission.is_final or mission.status != 'active'):
            self._goal_in_radius_since = None
            self._goal_stable_since = None
            self._goal_loiter_since = None
            return False
        if not (self.auto_stop_on_goal
                and (self.goal_msg is not None or mission is not None)
                and math.isfinite(goal_dist)):
            self._goal_in_radius_since = None
            self._goal_stable_since = None
            self._goal_loiter_since = None
            return False

        if goal_dist <= self.goal_success_radius:
            if self._goal_in_radius_since is None:
                self._goal_in_radius_since = stamp
            held_s = float(stamp - self._goal_in_radius_since)
            if held_s >= self.goal_success_hold_s:
                self.get_logger().info(
                    f"Goal reached (dist={goal_dist:.3f} m <= {self.goal_success_radius:.3f} m) "
                    f"and held for {held_s:.2f} s."
                )
                self._finish_run("goal_reached", stamp)
                return True
        else:
            self._goal_in_radius_since = None

        if goal_dist > self.goal_stable_radius:
            self._goal_stable_since = None
            self._goal_loiter_since = None
            return False
        if self._goal_loiter_since is None:
            self._goal_loiter_since = stamp
        elif float(stamp - self._goal_loiter_since) >= self.goal_loiter_timeout_s:
            self.get_logger().info(
                f"Goal loiter timeout: belief within {self.goal_stable_radius:.3f} m for "
                f"{float(stamp - self._goal_loiter_since):.2f} s without a goal hold."
            )
            self._finish_run("goal_loiter_timeout", stamp)
            return True

        stats = self._motion_window_stats(stamp, self.goal_stable_hold_s)
        stable_at_goal = (
            stats is not None
            and stats['displacement_m'] <= self.goal_stable_max_displacement_m
        )
        idle_at_goal = not self._command_active(cmd_v, cmd_w)
        if stable_at_goal or idle_at_goal:
            if self._goal_stable_since is None:
                self._goal_stable_since = stamp
            stable_held_s = float(stamp - self._goal_stable_since)
            if stable_held_s >= self.goal_stable_hold_s:
                mode = 'stable' if stable_at_goal else 'idle'
                self.get_logger().info(
                    f"Goal reached and {mode} "
                    f"(dist={goal_dist:.3f} m <= {self.goal_stable_radius:.3f} m) "
                    f"for {stable_held_s:.2f} s."
                )
                self._finish_run("goal_reached_stable", stamp)
                return True
        else:
            self._goal_stable_since = None
        return False

    def _maybe_finish_for_stuck(self, stamp: float, goal_dist: float) -> bool:
        if self._first_cmd_stamp is None or self.stuck_window_s <= 0.0:
            return False
        elapsed_after_first_cmd = float(stamp - self._first_cmd_stamp)
        if elapsed_after_first_cmd < self.stuck_window_s:
            return False
        if math.isfinite(goal_dist) and goal_dist <= self.goal_stable_radius:
            return False
        stats = self._motion_window_stats(stamp, self.stuck_window_s)
        if stats is None:
            return False
        no_motion = (
            stats['displacement_m'] <= self.stuck_max_displacement_m
            and stats['goal_improvement_m'] <= self.stuck_max_goal_improvement_m
        )
        active_stuck = stats['cmd_fraction'] >= self.stuck_cmd_fraction_min
        idle_stuck = stats['cmd_fraction'] <= self.stuck_idle_cmd_fraction_max
        turning_progress = (
            stats['angular_cmd_fraction'] >= STUCK_ANGULAR_CMD_FRACTION_MIN
            and stats['yaw_progress_rad'] >= STUCK_TURN_PROGRESS_MIN_RAD
        )
        stuck = no_motion and ((active_stuck and not turning_progress) or idle_stuck)
        if not stuck:
            return False
        mode = 'active' if active_stuck else 'idle'
        self.get_logger().info(
            f"Stuck termination ({mode}): "
            f"displacement={stats['displacement_m']:.3f} m <= {self.stuck_max_displacement_m:.3f} m, "
            f"goal_improvement={stats['goal_improvement_m']:.3f} m <= "
            f"{self.stuck_max_goal_improvement_m:.3f} m, "
            f"cmd_fraction={stats['cmd_fraction']:.2f}, "
            f"angular_cmd_fraction={stats['angular_cmd_fraction']:.2f}, "
            f"yaw_progress={stats['yaw_progress_rad']:.3f} rad, "
            f"active_threshold={self.stuck_cmd_fraction_min:.2f}, "
            f"idle_threshold={self.stuck_idle_cmd_fraction_max:.2f}."
        )
        self._finish_run("stuck", stamp)
        return True

    def _odom_cb(self, msg: Odometry):
        self.odom_msg = msg
        ok, st, x, y, yaw = self._latest_odom_map_pose()
        if ok and math.isfinite(st):
            self._odom_map_buf.append((float(st), float(x), float(y), float(yaw)))

    @staticmethod
    def _interpolate_pose_buffer(buf, stamp):
        """Interpolate a (stamp, x, y, yaw) buffer to `stamp`. Returns (ok,x,y,yaw).

        `ok` is False outside the buffered interval as well as when the buffer is
        empty: clamping to an endpoint would silently return a pose from a different
        instant and score it as if it were aligned, which is the exact defect this
        buffer exists to remove.
        """
        if not buf or not math.isfinite(stamp):
            return False, math.nan, math.nan, math.nan
        if stamp < buf[0][0] or stamp > buf[-1][0]:
            return False, math.nan, math.nan, math.nan
        prev = buf[0]
        for cur in buf:
            if cur[0] >= stamp:
                s0, x0, y0, yaw0 = prev
                s1, x1, y1, yaw1 = cur
                if s1 <= s0:
                    return True, x1, y1, yaw1
                a = (stamp - s0) / (s1 - s0)
                return (
                    True,
                    x0 + a * (x1 - x0),
                    y0 + a * (y1 - y0),
                    ExperimentLogger._wrap_angle(
                        yaw0 + a * ExperimentLogger._wrap_angle(yaw1 - yaw0)),
                )
            prev = cur
        return (True,) + tuple(buf[-1][1:])

    def _gt_at(self, stamp):
        """GROUND TRUTH interpolated to `stamp`. The reference every error must use."""
        return self._interpolate_pose_buffer(self._gt_buf, stamp)

    def _odom_map_at(self, stamp):
        """Wheel odometry in map_bev interpolated to `stamp`. Diagnostic only."""
        return self._interpolate_pose_buffer(self._odom_map_buf, stamp)

    def _error_against_gt_at(self, x, y, stamp):
        """Distance from (x, y) to the TRUE pose at the instant (x, y) describes.

        Returns (error_m, gt_x, gt_y) with NaNs when the estimate, its stamp, or the
        truth at that stamp is unavailable. Never falls back to the latest truth: a
        silent fallback is what made every error column read 100 ms late.
        """
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(stamp)):
            return math.nan, math.nan, math.nan
        ok, gx, gy, _gyaw = self._gt_at(stamp)
        if not ok:
            return math.nan, math.nan, math.nan
        return math.hypot(float(x) - gx, float(y) - gy), gx, gy

    def _odom_noisy_cb(self, msg: Odometry):
        self.odom_noisy_msg = msg

    def _ground_truth_cb(self, msg: TFMessage):
        # /world/<name>/dynamic_pose/info publishes every moving entity's world
        # pose. Keep the robot's. Holds last value while stationary (true pose
        # is constant then anyway).
        #
        # Every truth sample is TIMESTAMPED and buffered. Holding truth as a bare
        # latest value was the root cause of every misaligned error column: the only
        # instant it could then be paired with was the log clock.
        #
        # The transform's own stamp is preferred but is not available here: measured on
        # warehouse_v2, the ros_gz bridge publishes /ground_truth_tf with header.stamp
        # exactly 0 on every transform, because gz.msgs.Pose_V carries its time on the
        # message header and not on the individual poses. So the sample is stamped at
        # RECEIPT on the simulation clock. That stamp includes an unknown transport
        # delay. Inter-sample intervals below describe delivered cadence only; they do
        # not bound a constant or slowly varying bridge delay.
        receipt_s = float(self.get_clock().now().nanoseconds) * 1e-9
        for tr in msg.transforms:
            if tr.child_frame_id == 'turtlebot3':
                x = float(tr.transform.translation.x)
                y = float(tr.transform.translation.y)
                yaw = self._yaw_from_quaternion(tr.transform.rotation)
                header_stamp = self._stamp_to_float(tr.header.stamp)
                if math.isfinite(header_stamp) and header_stamp > 0.0:
                    stamp = header_stamp
                    self._gt_stamp_source = 'transform_header'
                else:
                    stamp = receipt_s
                    self._gt_stamp_source = 'receipt_sim_clock'
                self._gt_xy = (x, y)
                self._gt_yaw = yaw
                self._gt_stamp = stamp
                # Truth keeps arriving after the run ends; the file is closed by then.
                with getattr(self, '_log_write_lock', None) or nullcontext():
                    handle = getattr(self, 'ground_truth_pose_file', None)
                    if handle is not None and not handle.closed:
                        self.ground_truth_pose_writer.writerow(
                            [repr(stamp), self._gt_stamp_source, repr(x), repr(y), repr(yaw)])
                if math.isfinite(stamp) and (
                    not self._gt_buf or stamp > self._gt_buf[-1][0]
                ):
                    if self._gt_buf:
                        self._gt_intervals.append(stamp - self._gt_buf[-1][0])
                    self._gt_buf.append((stamp, x, y, yaw))

    def _record_runtime_delivery(self, topic: str, payload: str):
        """Append one raw ROS delivery before any semantic interpretation."""
        with self._event_lock:
            if not self._accepting_events or self._event_streams_closed:
                self._late_event_count += 1
                return None
            receipt = float(self.get_clock().now().nanoseconds) * 1e-9
            try:
                record = self.runtime_event_log.append(topic, payload, receipt)
            except Exception as exc:
                self._record_invalid(f'runtime_event_log_write_failed:{type(exc).__name__}')
                raise
            self._last_event_wall_s = time.monotonic()
            self._runtime_event_counts[topic] = self._runtime_event_counts.get(topic, 0) + 1
            if not record.get('valid_json_object', False):
                self._runtime_event_invalid_count += 1
            return record

    def _camera_opportunity_cb(self, camera: str, message: String) -> None:
        with self._event_lock:
            if not self._accepting_events or self._event_streams_closed:
                self._late_event_count += 1
                return
            receipt = float(self.get_clock().now().nanoseconds) * 1e-9
            try:
                record = self.camera_opportunity_log.append(camera, message.data, receipt)
            except Exception as exc:
                self._record_invalid(
                    f'camera_opportunity_log_write_failed:{type(exc).__name__}')
                raise
            self._last_event_wall_s = time.monotonic()
            if not record.get('valid_contract', False):
                self._record_invalid('invalid_camera_opportunity_delivery')
            elif record.get('conflicting_duplicate', False):
                self._record_invalid('conflicting_camera_opportunity_delivery')

    def _runtime_outcome_cb(self, topic: str, message: String) -> None:
        record = self._record_runtime_delivery(topic, message.data)
        if record is None:
            return
        payload = record.get('parsed_payload')
        if not isinstance(payload, dict):
            self._record_invalid(f'malformed_runtime_outcome:{topic}')
            return
        if topic == MISSION_GOAL_TOPIC:
            try:
                mission = mission_goal_from_json(message.data)
            except Exception:
                self._record_invalid('malformed_mission_goal_state')
                return
            previous_id = getattr(self, '_active_mission_goal_id', '')
            if previous_id and previous_id != mission.goal_id:
                self._goal_in_radius_since = None
                self._goal_stable_since = None
                self._goal_loiter_since = None
                self._goal_region_entered = False
                self._goal_region_first_stamp = math.nan
            self._active_mission_goal_id = mission.goal_id
            self._mission_goal_state = mission
            if mission.status == 'cancelled':
                cancellation_reason = mission.reason or 'unspecified'
                self._record_invalid(f'mission_cancelled:{cancellation_reason}')
                self._finish_run(
                    'mission_cancelled',
                    float(self.get_clock().now().nanoseconds) * 1e-9)
        elif topic == '/sim/actuation_outcome' and self._stop_requested:
            try:
                linear = float(payload.get('forwarded_linear'))
                angular = float(payload.get('forwarded_angular'))
                status = str(payload.get('status', '') or '')
                if (status in ('forwarded', 'forwarded_zero')
                        and abs(linear) <= 1e-12
                        and abs(angular) <= 1e-12):
                    stamp_ns = payload.get('forwarded_sim_stamp_ns')
                    stamp_is_valid = bool(
                        not isinstance(stamp_ns, bool)
                        and isinstance(stamp_ns, int)
                        and stamp_ns >= 0
                    )
                    # Latch the first timestamped zero after the stop request.
                    # A zero command is intentionally republished while stopped;
                    # replacing the anchor on every delivery would move the rest
                    # window forever and make verification impossible.
                    if (not self._terminal_zero_forwarded
                            or not math.isfinite(self._terminal_zero_stamp_s)):
                        self._terminal_zero_forwarded = True
                        self._terminal_stop_event_id = str(
                            payload.get('event_id', '') or '')
                        if stamp_is_valid:
                            self._terminal_zero_stamp_s = stamp_ns * 1e-9
                elif (self._terminal_zero_forwarded
                      and status == 'forwarded'
                      and (abs(linear) > 1e-12 or abs(angular) > 1e-12)):
                    self._terminal_stop_verified = False
                    self._record_invalid('nonzero_command_after_terminal_zero')
            except (TypeError, ValueError):
                self._record_invalid('malformed_actuation_outcome')

    def _terminal_stop_ack_cb(self, message: String) -> None:
        topic = TERMINAL_STOP_ACK_TOPIC
        record = self._record_runtime_delivery(topic, message.data)
        if record is None:
            return
        try:
            ack = terminal_stop_ack_from_json(message.data)
        except ValueError:
            self._record_invalid('malformed_terminal_stop_ack')
            return
        if not self._stop_requested or ack.request_id != self._terminal_stop_request_id:
            self._record_invalid('terminal_stop_ack_identity_mismatch')
            return
        previous = self._terminal_stop_acks.get(ack.component)
        if previous is not None and previous != ack:
            self._record_invalid('conflicting_terminal_stop_ack')
            return
        self._terminal_stop_acks[ack.component] = ack

    def _fused_correction_cb(self, message: String) -> None:
        topic = '/reliability/camera_manager/fused_correction'
        record = self._record_runtime_delivery(topic, message.data)
        if record is None:
            return
        payload = record.get('parsed_payload')
        try:
            if (not isinstance(payload, dict)
                    or type(payload.get('schema_version')) is not int
                    or payload.get('schema_version') not in (1, 2)):
                raise ValueError('unsupported correction publication schema')
            if payload['schema_version'] == 2:
                payload = FusedCorrectionEvent.from_json(message.data).payload
            source_batch_id = str(payload.get('source_batch_id', '') or '').strip()
            if not source_batch_id:
                raise ValueError('correction publication has no source_batch_id')
            correction_stamp = float(payload['correction_stamp'])
            if not math.isfinite(correction_stamp):
                raise ValueError('correction publication stamp is not finite')
            frame_id = str(payload.get('frame_id', '') or '').strip()
            if not frame_id:
                raise ValueError('correction publication has no frame_id')
            xy = np.asarray(payload['xy'], dtype=float)
            covariance = np.asarray(payload['covariance_m2'], dtype=float)
            if xy.shape != (2,) or covariance.shape != (2, 2):
                raise ValueError('correction publication payload shape is invalid')
            if not np.isfinite(xy).all() or not np.isfinite(covariance).all():
                raise ValueError('correction publication payload is not finite')
            if not np.allclose(covariance, covariance.T, rtol=1e-7, atol=1e-10):
                raise ValueError('correction publication covariance is not symmetric')
            np.linalg.cholesky(covariance)
            canonical = {
                'source_batch_id': source_batch_id,
                'correction_stamp': correction_stamp,
                'frame_id': frame_id,
                'payload': {'xy': xy.tolist(), 'covariance_m2': covariance.tolist()},
                'member_ids': list(
                    payload.get('member_ids')
                    or payload.get('physical_member_ids')
                    or payload.get('accepted_camera_ids')
                    or []),
            }
            for field in ('event_id', 'epoch', 'publication_seq', 'payload_sha256'):
                if field in payload:
                    canonical[field] = payload[field]
            if 'correction_stamp_ns' in payload:
                correction_stamp_ns = payload['correction_stamp_ns']
                if (isinstance(correction_stamp_ns, bool)
                        or not isinstance(correction_stamp_ns, int)
                        or correction_stamp_ns < 0):
                    raise ValueError('correction_stamp_ns is invalid')
                canonical['correction_stamp_ns'] = correction_stamp_ns
        except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
            self._record_invalid(f'invalid_correction_publication:{type(exc).__name__}')
            return
        previous = self._correction_publications.get(source_batch_id)
        if previous is not None:
            if previous != canonical:
                self._record_invalid('conflicting_correction_publication')
            return
        self._correction_publications[source_batch_id] = canonical
        receipt = float(record['receive_stamp_s'])
        self.correction_publication_writer.writerow([
            source_batch_id, payload.get('event_id', ''), payload.get('epoch', ''),
            payload.get('publication_seq', ''), correction_stamp,
            payload.get('correction_stamp_ns', ''),
            payload.get('common_capture_stamp', ''),
            payload.get('common_capture_stamp_ns', ''),
            payload.get('publication_stamp_ns', ''), receipt, frame_id,
            payload.get('schema_version'),
            json.dumps(payload.get('accepted_camera_ids') or [], separators=(',', ':')),
            json.dumps(canonical['member_ids'], separators=(',', ':')),
            payload.get('payload_sha256', ''),
        ])
        self.correction_publication_file.flush()

    def _fusion_decision_cb(self, message) -> None:
        """The camera manager's own account of which cameras went into this correction."""
        topic = '/reliability/camera_manager/decision'
        record = self._record_runtime_delivery(topic, message.data)
        if record is None:
            return
        payload = record.get('parsed_payload')
        if not isinstance(payload, dict):
            self._record_invalid('malformed_fusion_decision')
            return
        self.fusion_decision = payload
        self.fusion_decision_stamp = float(record['receive_stamp_s'])

        source_batch_id = str(payload.get('source_batch_id', '') or '').strip()
        if not source_batch_id:
            self._record_invalid('fusion_decision_without_source_batch_id')
            return
        try:
            signature = json.dumps(payload, sort_keys=True, allow_nan=False,
                                   separators=(',', ':'))
        except (ValueError, TypeError, OverflowError):
            self._record_invalid('nonfinite_fusion_decision')
            return
        previous = self._fusion_decision_payloads.get(source_batch_id)
        if previous is not None:
            if previous != signature:
                self._record_invalid('conflicting_fusion_decision')
            return
        self._fusion_decision_payloads[source_batch_id] = signature

        observations = payload.get('observations')
        if not observations or self.fusion_obs_writer is None:
            return
        stamp = self.fusion_decision_stamp
        self._fusion_obs_last_stamp = stamp
        self._fusion_decision_seq += 1
        decision_seq = self._fusion_decision_seq
        fused = payload.get('fused_xy') or [float('nan'), float('nan')]
        fcov = payload.get('fused_cov') or [[float('nan')] * 2] * 2
        gt = self._gt_xy if self._gt_xy is not None else (float('nan'), float('nan'))
        gt_stamp = float(self._gt_stamp)
        fused_stamp = float(payload.get('fused_stamp', float('nan')))
        common_capture_stamp = float(
            payload.get('common_capture_stamp', float('nan')))
        fok, fgx, fgy, _fyaw = self._gt_at(fused_stamp)
        if not fok:
            fgx = fgy = float('nan')
        used_ids = payload.get('accepted_camera_ids') or []
        for observation in observations:
            cov = observation.get('cov') or [[float('nan')] * 2] * 2
            xy = observation.get('xy') or [float('nan'), float('nan')]
            aligned_cov = observation.get('aligned_cov') or [[float('nan')] * 2] * 2
            aligned_xy = observation.get('aligned_xy') or [float('nan'), float('nan')]
            camera = str(observation.get('camera', '')).replace('camera_', '')
            obs_stamp = float(observation.get('obs_stamp', float('nan')))
            # The truth at the instant THIS CAMERA saw the robot. Written here, once,
            # so no downstream script has to re-derive it -- and so a script that
            # forgets to is visibly using the wrong column rather than silently
            # charging the pipeline delay to the sensor.
            ook, ogx, ogy, _oyaw = self._gt_at(obs_stamp)
            if not ook:
                ogx = ogy = float('nan')
            key = (source_batch_id, camera)
            repeat = self._obs_repeat_count.get(key, 0)
            self._obs_repeat_count[key] = repeat + 1
            if repeat == 0:
                self._obs_seq_by_camera[camera] = (
                    self._obs_seq_by_camera.get(camera, -1) + 1)
            obs_seq = self._obs_seq_by_camera.get(camera, -1)
            self.fusion_obs_writer.writerow([
                stamp, decision_seq, source_batch_id, common_capture_stamp, camera,
                1 if observation.get('used') else 0, xy[0], xy[1],
                cov[0][0], cov[0][1], cov[1][1],
                aligned_xy[0], aligned_xy[1],
                aligned_cov[0][0], aligned_cov[0][1], aligned_cov[1][1],
                len(observations), len(used_ids),
                fused[0], fused[1], fcov[0][0], fcov[0][1], fcov[1][1],
                gt[0], gt[1], gt_stamp,
                ogx, ogy,
                fused_stamp, fgx, fgy,
                obs_seq, repeat,
                observation.get('conf', float('nan')),
                observation.get('conf_raw', float('nan')),
                observation.get('bbox_h_px', float('nan')),
                observation.get('bbox_w_px', float('nan')),
                observation.get('pred_h_px', float('nan')),
                observation.get('pred_w_px', float('nan')),
                observation.get('raw_obs_x', float('nan')),
                observation.get('raw_obs_y', float('nan')),
                observation.get('range_m', float('nan')),
                obs_stamp,
            ])
        self.fusion_obs_file.flush()

    def _state_cb(self, msg: PoseWithCovarianceStamped):
        self.state_msg = msg

    def _planner_belief_cb(self, msg: PoseWithCovarianceStamped):
        self.planner_belief_msg = msg

    def _belief_state_cb(self, msg: String):
        """Retain identity-bearing planner predictions for unambiguous scoring."""
        receive_stamp = float(self.get_clock().now().nanoseconds) * 1e-9
        record = {'logger_receive_stamp': receive_stamp, 'valid_envelope': False}
        invalid_reason = ''
        try:
            payload = json.loads(msg.data)
            if not isinstance(payload, dict) or payload.get('schema_version') != 1:
                raise ValueError('unsupported planner belief schema')
            for key in ('initialized', 'valid', 'motion_supported'):
                if type(payload.get(key)) is not bool:
                    raise ValueError(f'planner belief {key} is not boolean')
            if payload['initialized']:
                if not isinstance(payload.get('epoch'), str) or not payload['epoch']:
                    raise ValueError('planner belief epoch is missing')
                if type(payload.get('revision')) is not int or payload['revision'] < 0:
                    raise ValueError('planner belief revision is invalid')
                for key in ('anchor_stamp_ns', 'state_stamp_ns'):
                    if type(payload.get(key)) is not int or payload[key] < 0:
                        raise ValueError(f'planner belief {key} is invalid')
                if payload['state_stamp_ns'] < payload['anchor_stamp_ns']:
                    raise ValueError('planner belief state precedes its anchor')
                mean = np.asarray(payload.get('mean'), dtype=float)
                covariance = np.asarray(payload.get('covariance'), dtype=float)
                if mean.shape != (3,) or covariance.shape != (3, 3):
                    raise ValueError('planner belief state shape is invalid')
                if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
                    raise ValueError('planner belief state is nonfinite')
                if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=1e-10):
                    raise ValueError('planner belief covariance is asymmetric')
                if float(np.linalg.eigvalsh(covariance).min()) < -1e-10:
                    raise ValueError('planner belief covariance is indefinite')
            record.update(valid_envelope=True, payload=payload)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            invalid_reason = f'{type(exc).__name__}:{exc}'
            record.update(error=invalid_reason, raw=str(msg.data))
        with self._log_write_lock:
            handle = getattr(self, 'belief_prediction_file', None)
            if handle is None or handle.closed:
                return
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
            handle.flush()
            self._belief_prediction_count += 1
            if invalid_reason:
                self._belief_prediction_invalid_count += 1
                self._record_invalid(f'invalid_planner_belief_prediction:{invalid_reason}')

    def _obs_cb(self, msg: PoseStamped):
        self.obs_msg = msg

    def _cmd_cb(self, msg: Twist):
        self.cmd_msg = msg
        self.cmd_stamp_s = float(self.get_clock().now().nanoseconds) * 1e-9

    def _cmd_raw_cb(self, msg: Twist):
        self.cmd_raw_msg = msg
        self.cmd_raw_stamp_s = float(self.get_clock().now().nanoseconds) * 1e-9

    def _cmd_noise_diag_cb(self, msg: Float64MultiArray):
        self.cmd_noise_diag = msg
        self.cmd_noise_diag_stamp_s = float(self.get_clock().now().nanoseconds) * 1e-9

    def _goal_cb(self, msg: PoseStamped):
        self.goal_msg = msg

    def _active_goal_xy(self):
        """Return coordinates atomically paired with mission identity when available."""
        mission = getattr(self, '_mission_goal_state', None)
        if mission is not None:
            return float(mission.x), float(mission.y)
        if self.goal_msg is not None:
            return (float(self.goal_msg.pose.position.x),
                    float(self.goal_msg.pose.position.y))
        return math.nan, math.nan

    def _plan_cb(self, msg: Path):
        self.plan_msg = msg
        if self.plan_writer is None:
            return
        if not msg.poses:
            return
        plan_stamp = self._stamp_to_float(msg.header.stamp)
        for i, pose_stamped in enumerate(msg.poses):
            p = pose_stamped.pose.position
            self.plan_writer.writerow([plan_stamp, i, p.x, p.y])
        self.plan_file.flush()

    def _efe_cb(self, msg: Float64MultiArray):
        self.efe_metrics = msg
        if msg.data and len(msg.data) >= 3:
            self._efe_risk_sum += float(msg.data[1])
            self._efe_ambiguity_sum += float(msg.data[2])
            if len(msg.data) >= 5:
                self._efe_control_sum += float(msg.data[3])
                self._efe_obstacle_sum += float(msg.data[4])
            self._efe_count += 1

    def _planner_diag_cb(self, msg: Float64MultiArray):
        self.planner_diag = msg
        if msg.data and len(msg.data) >= 6:
            solve_time_ms = float(msg.data[5])
            self._solve_time_ms_sum += solve_time_ms
            self._solve_count += 1
        if msg.data and len(msg.data) >= 12:
            p_vis_plan = float(msg.data[6])
            p_vis_plan_eff = float(msg.data[7])
            r_plan_u_std = float(msg.data[8])
            r_plan_v_std = float(msg.data[9])
            if math.isfinite(p_vis_plan):
                self._p_vis_plan_sum += p_vis_plan
                self._p_vis_plan_eff_sum += p_vis_plan_eff
                self._r_plan_u_std_sum += r_plan_u_std
                self._r_plan_v_std_sum += r_plan_v_std
                
                if p_vis_plan < 0.2:
                    self._p_vis_plan_below_0_2_count += 1
                if p_vis_plan_eff < 0.2:
                    self._p_vis_plan_eff_below_0_2_count += 1
                
                r_std_max = max(r_plan_u_std, r_plan_v_std)
                if r_std_max > self._max_r_plan_std:
                    self._max_r_plan_std = r_std_max
                
                self._p_vis_count += 1

    def _planner_diag_text_cb(self, msg: String):
        self.planner_diag_text = str(msg.data or '')

    def _active_execution_diag_cb(self, msg: Float64MultiArray):
        self.active_execution_diag = msg

    def _diag_cb(self, msg: Float64MultiArray):
        self.perception_diag = diagnostics_from_message(msg)
        self._log_perception_sample(self.perception_diag)

    def _heading_diag_cb(self, msg: Float64MultiArray):
        self.heading_diag = msg

    def _pixel_correction_diag_cb(self, msg: Float64MultiArray):
        self.pixel_correction_diag = msg

    def _correction_assimilation_cb(self, msg: String):
        topic = '/planner/correction_assimilation'
        record = self._record_runtime_delivery(topic, msg.data)
        if record is None:
            return
        payload = record.get('parsed_payload')
        if not isinstance(payload, dict):
            self._record_invalid('malformed_correction_assimilation')
            return
        source_batch_id = str(payload.get('source_batch_id', '') or '').strip()
        if not source_batch_id:
            self._record_invalid('assimilation_missing_source_batch_id')
            return
        if source_batch_id in self._assimilation_source_batches:
            self._record_invalid('duplicate_source_batch_assimilation')
            return
        schema_version = payload.get('schema_version')
        if type(schema_version) is not int or schema_version not in (1, 2):
            self._record_invalid('unsupported_correction_assimilation_schema')
            return
        try:
            json.dumps(payload, sort_keys=True, allow_nan=False, separators=(',', ':'))
        except (ValueError, TypeError, OverflowError):
            self._record_invalid('nonfinite_correction_assimilation')
            return
        status = str(payload.get('status', '') or '').strip()
        reason = str(payload.get('reason', '') or '').strip()
        accepted = payload.get('accepted')
        if status not in KNOWN_ASSIMILATION_STATUSES:
            self._record_invalid(f'unknown_assimilation_status:{status or "empty"}')
            return
        if status in ('rejected', 'dropped') and not reason:
            self._record_invalid('correction_refusal_without_reason')
            return
        if not isinstance(accepted, bool) or accepted != (
                status in ('accepted', 'accepted_bootstrap', 'reanchored')):
            self._record_invalid('assimilation_accepted_status_mismatch')
            return
        try:
            correction_stamp = float(payload['correction_stamp'])
            apply_stamp = float(payload['apply_stamp'])
            if (not math.isfinite(correction_stamp) or correction_stamp < 0.0
                    or not math.isfinite(apply_stamp) or apply_stamp < 0.0):
                raise ValueError('terminal timestamps must be finite and nonnegative')
            if schema_version == 2:
                epoch = payload['epoch']
                frame_id = payload['frame_id']
                revision_before = payload['revision_before']
                revision_after = payload['revision_after']
                correction_stamp_ns = payload['correction_stamp_ns']
                apply_stamp_ns = payload['apply_stamp_ns']
                state_stamp_ns = payload['state_stamp_ns']
                initialized = payload['initialized']
                if not isinstance(epoch, str) or not epoch.strip():
                    raise ValueError('schema-2 terminal has no epoch')
                if not isinstance(frame_id, str) or not frame_id.strip():
                    raise ValueError('schema-2 terminal has no frame_id')
                if (isinstance(revision_before, bool)
                        or not isinstance(revision_before, int)
                        or revision_before < 0
                        or isinstance(revision_after, bool)
                        or not isinstance(revision_after, int)
                        or revision_after < revision_before):
                    raise ValueError('schema-2 terminal revision is invalid')
                if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                       for value in (correction_stamp_ns, apply_stamp_ns)):
                    raise ValueError('schema-2 terminal exact timestamps are invalid')
                if (correction_stamp != correction_stamp_ns * 1e-9
                        or apply_stamp != apply_stamp_ns * 1e-9):
                    raise ValueError('schema-2 float/exact timestamps disagree')
                if accepted and apply_stamp_ns < correction_stamp_ns:
                    raise ValueError('schema-2 accepted terminal applies before correction')
                if type(initialized) is not bool:
                    raise ValueError('schema-2 terminal initialized is not boolean')
                if type(payload.get('valid')) is not bool:
                    raise ValueError('schema-2 terminal valid is not boolean')
                if type(payload.get('motion_supported')) is not bool:
                    raise ValueError('schema-2 terminal motion_supported is not boolean')
                if initialized:
                    posterior_mean = np.asarray(payload['posterior_mean'], dtype=float)
                    posterior_covariance = np.asarray(
                        payload['posterior_covariance'], dtype=float)
                    if (isinstance(state_stamp_ns, bool)
                            or not isinstance(state_stamp_ns, int)
                            or state_stamp_ns < 0):
                        raise ValueError('schema-2 terminal state_stamp_ns is invalid')
                    if (posterior_mean.shape != (3,)
                            or posterior_covariance.shape != (3, 3)):
                        raise ValueError('schema-2 terminal posterior shape is invalid')
                    if (not np.isfinite(posterior_mean).all()
                            or not np.isfinite(posterior_covariance).all()
                            or not np.allclose(posterior_covariance,
                                               posterior_covariance.T,
                                               rtol=1e-7, atol=1e-10)
                            or np.linalg.eigvalsh(
                                posterior_covariance).min() < -1e-10):
                        raise ValueError(
                            'schema-2 terminal posterior covariance is invalid')
                elif (accepted or payload.get('valid') or payload.get('motion_supported')
                      or state_stamp_ns is not None
                      or payload.get('posterior_mean') is not None
                      or payload.get('posterior_covariance') is not None):
                    raise ValueError('uninitialized terminal claims a posterior/update')
        except (KeyError, TypeError, ValueError, OverflowError, np.linalg.LinAlgError):
            self._record_invalid('invalid_correction_assimilation_contract')
            return

        self._assimilation_source_batches.add(source_batch_id)
        canonical_terminal = dict(payload)
        # `epoch` on terminal schema 2 is the recursive belief epoch. Publication
        # identity has its own producer epoch, copied through by the planner as
        # `source_epoch`; map that field only for cross-boundary ledger agreement.
        if payload.get('source_epoch') is not None:
            canonical_terminal['belief_epoch'] = payload.get('epoch')
            canonical_terminal['epoch'] = payload.get('source_epoch')
        if payload.get('source_payload_sha256') is not None:
            canonical_terminal['payload_sha256'] = payload.get(
                'source_payload_sha256')
        if payload.get('source_member_ids') is not None:
            canonical_terminal['member_ids'] = payload.get('source_member_ids')
        self._assimilation_payloads[source_batch_id] = canonical_terminal
        try:
            corr_stamp_s = correction_stamp
        except (TypeError, ValueError):
            corr_stamp_s = math.nan
        if math.isfinite(corr_stamp_s):
            if self._last_correction_stamp_s is not None:
                gap = corr_stamp_s - self._last_correction_stamp_s
                if gap > self._longest_correction_gap_s:
                    self._longest_correction_gap_s = float(gap)
            self._last_correction_stamp_s = corr_stamp_s
        self._assimilation_count += 1
        if status in ('rejected', 'dropped'):
            # A REFUSAL is not a broken evidence chain. The filter declined a measurement
            # it could not causally bridge -- most often a camera outage longer than the
            # replay cap -- recorded why, and carried on. That is the same class of event
            # as a NIS rejection, which has never invalidated a run.
            #
            # What would invalidate the run is a correction with no outcome, two outcomes,
            # or an unclassifiable one; those are checked above and on completion. The
            # refusal RATE and the longest gap are reported instead, because a warehouse
            # with a 17 s blind stretch is a finding about the camera network, not a
            # faulty drive -- and discarding those runs would throw away exactly the
            # low-coverage routes the comparison needs.
            if status == 'dropped':
                self._assimilation_dropped_count += 1
                self._assimilation_dropped_reasons[reason] = (
                    self._assimilation_dropped_reasons.get(reason, 0) + 1
                )
        def encoded(field):
            value = payload.get(field)
            if value is None:
                return ''
            return json.dumps(value, allow_nan=False, separators=(',', ':'))
        self.assimilation_writer.writerow([
            source_batch_id,
            payload.get('correction_stamp', math.nan),
            payload.get('apply_stamp', math.nan),
            status,
            reason,
            1 if accepted is True else 0,
            payload.get('nis', math.nan),
            payload.get('belief_stamp_after', math.nan),
            payload.get('schema_version', ''),
            payload.get('epoch', ''),
            payload.get('revision_before', ''),
            payload.get('revision_after', ''),
            payload.get('frame_id', ''),
            payload.get('state_stamp_ns', ''),
            encoded('posterior_mean'),
            encoded('posterior_covariance'),
            payload.get('valid', ''),
            payload.get('motion_supported', ''),
            record.get('receive_stamp_s', math.nan),
            payload.get('correction_stamp_ns', ''),
            payload.get('apply_stamp_ns', ''),
            payload.get('belief_stamp_before_ns', ''),
            payload.get('belief_stamp_after_ns', ''),
            payload.get('source_event_id', ''),
            payload.get('source_epoch', ''),
            payload.get('source_publication_seq', ''),
            encoded('source_member_ids'),
            payload.get('source_payload_sha256', ''),
        ])
        self.assimilation_file.flush()

    @staticmethod
    def extract_planar_covariances(cov):
        if cov is None or len(cov) < 36:
            return math.nan, math.nan, math.nan, math.nan, math.nan, math.nan
        cov_xx = float(cov[0])
        cov_xy = float(cov[1])
        cov_yy = float(cov[7])
        cov_x_theta = float(cov[30])
        cov_y_theta = float(cov[31])
        cov_theta_theta = float(cov[35])
        return cov_xx, cov_xy, cov_yy, cov_x_theta, cov_y_theta, cov_theta_theta

    @staticmethod
    def _heading_source_name(code: float) -> str:
        try:
            value = int(round(float(code)))
        except (TypeError, ValueError):
            value = 0
        return {
            1: 'pixel_heading',
            2: 'odom_heading_fallback',
            3: 'motion_heading_fallback',
            4: 'held_previous_heading',
        }.get(value, 'unknown')

    @staticmethod
    def _pixel_correction_reject_reason_name(code: float) -> str:
        try:
            value = int(round(float(code)))
        except (TypeError, ValueError):
            value = 99
        return {
            0: 'accepted',
            1: 'stale_age',
            2: 'dt_implausible',
            3: 'missing_snapshot',
            4: 'update_failed',
            5: 'jump_too_large',
            6: 'nis_too_large',
            7: 'diverged',
            8: 'not_newer_than_belief',
            9: 'replay_gap_too_large',
        }.get(value, 'unknown')

    @staticmethod
    def _pixel_correction_motion_replay_source_name(code: float) -> str:
        try:
            value = int(round(float(code)))
        except (TypeError, ValueError):
            value = -1
        return {
            0: 'none',
            1: 'odom_noisy',
            2: 'command_log',
            3: 'single_fallback',
        }.get(value, 'unknown')

    def _record_invalid(self, reason: str) -> None:
        reason = str(reason or '').strip()
        if not reason:
            return
        if self._valid_run:
            self._valid_run = False
            self._invalid_reason = reason
            return
        if reason not in self._invalid_reason.split('|'):
            self._invalid_reason = f'{self._invalid_reason}|{reason}' if self._invalid_reason else reason

    def _camera_relative_bearing_deg(self, odom_map_x: float, odom_map_y: float, odom_map_yaw: float) -> float:
        vec = np.asarray(self.camera_pos_xy, dtype=float) - np.array([float(odom_map_x), float(odom_map_y)], dtype=float)
        if np.linalg.norm(vec) <= 1e-9:
            return math.nan
        bearing_world = math.atan2(float(vec[1]), float(vec[0]))
        rel = self._wrap_angle(bearing_world - float(odom_map_yaw))
        return float(abs(math.degrees(rel)))

    def _latest_odom_map_pose(self):
        """Latest WHEEL ODOMETRY pose expressed in the map_bev frame.

        Named for what it is. It was called `_latest_truth_pose` and is not truth:
        `/odom` comes from the Gazebo DiffDrive plugin and accumulates about 1% of
        distance travelled, measured at 24 cm median and 2.4 m worst on these drives.
        Truth is `_gt_xy` / `_gt_buf`, from `/ground_truth_tf`.
        """
        if self.odom_msg is None:
            return False, math.nan, math.nan, math.nan, math.nan

        stamp = self._stamp_to_float(self.odom_msg.header.stamp)
        source_frame = (self.odom_msg.header.frame_id or 'odom').strip() or 'odom'
        pose_world = self.odom_msg.pose.pose
        if source_frame != self.frame_id:
            try:
                tf_msg = self._tf_buffer.lookup_transform(
                    self.frame_id,
                    source_frame,
                    rclpy.time.Time(),
                )
                pose_world = do_transform_pose(self.odom_msg.pose.pose, tf_msg)
            except tf2_ros.TransformException as exc:
                now_wall = time.monotonic()
                if now_wall - self._last_tf_warn_wall > 2.0:
                    self.get_logger().warn(
                        f"Truth pose unavailable until TF {source_frame}->{self.frame_id} exists: {exc}"
                    )
                    self._last_tf_warn_wall = now_wall
                return False, stamp, math.nan, math.nan, math.nan

        return (
            True,
            stamp,
            float(pose_world.position.x),
            float(pose_world.position.y),
            self._yaw_from_quaternion(pose_world.orientation),
        )

    def _latest_raw_odom_pose(self):
        if self.odom_msg is None:
            return False, math.nan, math.nan, math.nan, math.nan, ''
        pose = self.odom_msg.pose.pose
        source_frame = (self.odom_msg.header.frame_id or 'odom').strip() or 'odom'
        return (
            True,
            self._stamp_to_float(self.odom_msg.header.stamp),
            float(pose.position.x),
            float(pose.position.y),
            self._yaw_from_quaternion(pose.orientation),
            source_frame,
        )

    def _odom_record(self, msg: Odometry | None):
        if msg is None:
            return False, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan
        pose = msg.pose.pose
        twist = msg.twist.twist
        return (
            True,
            self._stamp_to_float(msg.header.stamp),
            float(pose.position.x),
            float(pose.position.y),
            self._yaw_from_quaternion(pose.orientation),
            float(twist.linear.x),
            float(twist.angular.z),
        )

    def _rewrite_manifest(self):
        payload = dict(self._manifest_data)
        payload['frame_sanity'] = dict(self._frame_sanity)
        write_manifest(self.run_dir, payload, self.repo_root)

    def _maybe_log_frame_sanity(self, now_stamp: float, cmd_v: float, cmd_w: float):
        if self._frame_sanity_logged:
            return
        if self.task_start_pose is None:
            self._frame_sanity_logged = True
            self._frame_sanity.update({
                'recorded': False,
                'ok': None,
                'reason': 'task_start_unavailable',
            })
            self._rewrite_manifest()
            self.get_logger().warn(
                f'Frame sanity check skipped because task start pose could not be loaded '
                f'from tasks_yaml={self.tasks_yaml!r} for world={self.world!r}, task={self.task!r}.'
            )
            return
        if self._first_cmd_stamp is not None:
            self._frame_sanity_logged = True
            self._frame_sanity.update({
                'recorded': False,
                'ok': None,
                'reason': 'first_command_started_before_sanity',
            })
            self._rewrite_manifest()
            self.get_logger().warn(
                'Frame sanity check could not be recorded before the first command; '
                'treat truth-frame validation as unavailable for this run.'
            )
            return
        if abs(cmd_v) >= self.first_cmd_linear_eps or abs(cmd_w) >= self.first_cmd_angular_eps:
            return

        raw_ok, _raw_stamp, raw_x, raw_y, raw_yaw, source_frame = self._latest_raw_odom_pose()
        true_ok, odom_map_stamp, odom_map_x, odom_map_y, odom_map_yaw = self._latest_odom_map_pose()
        if not (raw_ok and true_ok):
            return

        start_x, start_y, start_yaw = self.task_start_pose
        truth_start_error = float(math.hypot(odom_map_x - start_x, odom_map_y - start_y))
        raw_start_error = float(math.hypot(raw_x - start_x, raw_y - start_y))
        truth_start_yaw_error = abs(self._wrap_angle(odom_map_yaw - start_yaw))
        raw_start_yaw_error = abs(self._wrap_angle(raw_yaw - start_yaw))
        ok = bool(
            truth_start_error <= self.frame_sanity_start_tolerance_m
            and truth_start_yaw_error <= self.frame_sanity_start_tolerance_yaw_rad
        )

        self._frame_sanity_logged = True
        self._frame_sanity.update({
            'recorded': True,
            'ok': ok,
            'reason': 'ok' if ok else 'odom_map_start_mismatch',
            'source_frame': source_frame,
            'odom_map_stamp': odom_map_stamp,
            'raw_odom_x': raw_x,
            'raw_odom_y': raw_y,
            'raw_odom_yaw': raw_yaw,
            'odom_map_x': odom_map_x,
            'odom_map_y': odom_map_y,
            'odom_map_yaw': odom_map_yaw,
            'task_start_x': start_x,
            'task_start_y': start_y,
            'task_start_yaw': start_yaw,
            'odom_map_start_error_m': truth_start_error,
            'raw_start_error_m': raw_start_error,
            'odom_map_start_yaw_error_rad': truth_start_yaw_error,
            'raw_start_yaw_error_rad': raw_start_yaw_error,
            'tolerance_m': self.frame_sanity_start_tolerance_m,
            'tolerance_yaw_rad': self.frame_sanity_start_tolerance_yaw_rad,
            'recorded_at_log_stamp': now_stamp,
        })
        self._rewrite_manifest()

        message = (
            'Frame sanity check '
            f'({source_frame} -> {self.frame_id}): raw odom=({raw_x:.3f}, {raw_y:.3f}), '
            f'transformed truth=({odom_map_x:.3f}, {odom_map_y:.3f}), '
            f'task start=({start_x:.3f}, {start_y:.3f}), '
            f'truth_start_error={truth_start_error:.3f} m, '
            f'truth_start_yaw_error={truth_start_yaw_error:.3f} rad'
        )
        if ok:
            self.get_logger().info(message)
        else:
            self.get_logger().warn(
                message
                + (
                    f' exceeds tolerances {self.frame_sanity_start_tolerance_m:.3f} m and/or '
                    f'{self.frame_sanity_start_tolerance_yaw_rad:.3f} rad. '
                )
                + 'This usually means the map_bev->odom transform or odom frame assumption is wrong.'
            )

    def _latest_state_pose(self):
        if self.state_msg is None:
            return False, math.nan, math.nan, math.nan, math.nan
        stamp = self._stamp_to_float(self.state_msg.header.stamp)
        pose = self.state_msg.pose.pose
        return (
            True,
            stamp,
            float(pose.position.x),
            float(pose.position.y),
            self._yaw_from_quaternion(pose.orientation),
        )

    def _latest_planner_belief_pose(self):
        if self.planner_belief_msg is None:
            return False, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan
        stamp = self._stamp_to_float(self.planner_belief_msg.header.stamp)
        pose = self.planner_belief_msg.pose.pose
        cov = list(self.planner_belief_msg.pose.covariance)
        return (
            True,
            stamp,
            float(pose.position.x),
            float(pose.position.y),
            self._yaw_from_quaternion(pose.orientation),
            float(cov[0]) if len(cov) > 0 else math.nan,
            float(cov[1]) if len(cov) > 1 else math.nan,
            float(cov[7]) if len(cov) > 7 else math.nan,
            float(cov[35]) if len(cov) > 35 else math.nan,
        )

    def _latest_pixel_pose(self):
        if self.obs_msg is None:
            return False, math.nan, math.nan, math.nan, math.nan
        pose = self.obs_msg.pose
        return (
            True,
            self._stamp_to_float(self.obs_msg.header.stamp),
            float(pose.position.x),
            float(pose.position.y),
            self._yaw_from_quaternion(pose.orientation),
        )

    def _log_perception_sample(self, diag):
        if self.perception_writer is None:
            return

        true_ok, _truth_stamp, true_x, true_y, true_yaw = self._latest_odom_map_pose()
        state_ok, state_stamp, state_x, state_y, state_yaw = self._latest_state_pose()
        obs_ok, pixel_pose_stamp, pixel_pose_u, pixel_pose_v, pixel_pose_yaw = self._latest_pixel_pose()

        state_pos_error = math.nan
        state_yaw_error_deg = math.nan
        if true_ok and state_ok:
            state_pos_error = math.hypot(state_x - true_x, state_y - true_y)
            state_yaw_error_deg = math.degrees(self._wrap_angle(state_yaw - true_yaw))

        obs_yaw_error_deg = math.nan
        if true_ok and diag['detected'] and math.isfinite(diag['yaw_est']):
            obs_yaw_error_deg = math.degrees(self._wrap_angle(diag['yaw_est'] - true_yaw))
        camera_relative_bearing_deg = math.nan
        if true_ok:
            camera_relative_bearing_deg = self._camera_relative_bearing_deg(true_x, true_y, true_yaw)

        log_stamp = float(self.get_clock().now().nanoseconds) * 1e-9
        state_age_s = math.nan
        if state_ok and math.isfinite(state_stamp):
            state_age_s = max(log_stamp - state_stamp, 0.0)
        state_fresh = bool(
            state_ok
            and math.isfinite(state_age_s)
            and state_age_s <= max(float(self.pixel_timeout_s), 0.0)
        )
        pixel_pose_age_s = math.nan
        if obs_ok and math.isfinite(pixel_pose_stamp):
            pixel_pose_age_s = max(log_stamp - pixel_pose_stamp, 0.0)
        pixel_pose_fresh = bool(
            obs_ok
            and math.isfinite(pixel_pose_age_s)
            and pixel_pose_age_s <= max(float(self.pixel_timeout_s), 0.0)
        )

        # Compute predicted world position from image coordinates using homography
        pred_world_x = math.nan
        pred_world_y = math.nan
        localization_error_m = math.nan
        pred_world_x_calibrated = math.nan
        pred_world_y_calibrated = math.nan
        localization_error_calibrated_m = math.nan
        if obs_ok and math.isfinite(pixel_pose_u) and math.isfinite(pixel_pose_v):
            world = self.camera_model.pixel_to_world(float(pixel_pose_u), float(pixel_pose_v))
            if world is not None:
                pred_world_x = float(world[0])
                pred_world_y = float(world[1])
                pred_world_x_calibrated, pred_world_y_calibrated = self._apply_bev_calibration(
                    pred_world_x,
                    pred_world_y,
                )
                if true_ok:
                    localization_error_m = math.hypot(pred_world_x - true_x, pred_world_y - true_y)
                    localization_error_calibrated_m = math.hypot(
                        pred_world_x_calibrated - true_x,
                        pred_world_y_calibrated - true_y,
                    )

        # Capture-time errors: compare the measurement / state to GROUND TRUTH at THEIR
        # OWN timestamp instead of the current log time. This removes the pipeline
        # latency, so it reflects the TRUE detector / projection quality.
        #
        # These used to interpolate the odometry buffer, which is wheel odometry, so
        # they were scored against a reference up to 2.4 m from the real pose while
        # the comment claimed they showed true detector quality. They now use the
        # stamped ground-truth buffer. The *_calibrated_m and state_pos_error columns
        # above compare to log-time truth and are therefore LATENCY-INFLATED in turns.
        localization_error_captime_m = math.nan
        if obs_ok:
            localization_error_captime_m, _cgx, _cgy = self._error_against_gt_at(
                pred_world_x_calibrated, pred_world_y_calibrated, pixel_pose_stamp)
        state_error_captime_m = math.nan
        if state_ok:
            state_error_captime_m, _sgx, _sgy = self._error_against_gt_at(
                state_x, state_y, state_stamp)

        selected_pixel_source_code = float(diag.get('selected_pixel_source_code', math.nan))
        if selected_pixel_source_code >= 1.5:
            selected_pixel_source = 'mask_bottom'
        elif selected_pixel_source_code >= 0.5:
            selected_pixel_source = 'bbox_bottom'
        else:
            selected_pixel_source = 'none'
        yolo_raw_best_score = diag.get('yolo_score_raw', math.nan)
        yolo_selected_score = diag.get('yolo_score_selected', math.nan)
        yolo_num_target_candidates = diag.get('yolo_target_candidate_count', math.nan)
        yolo_selected_class_id = diag.get('yolo_best_class_id', math.nan)
        yolo_bbox_area = diag.get('bbox_area_px', math.nan)
        yolo_mask_area = diag.get('mask_area_px', math.nan)
        yolo_inference_ms = diag.get('yolo_inference_ms', math.nan)
        detector_callback_ms = diag.get('detector_callback_ms', math.nan)
        yolo_receive_stamp = diag.get('yolo_receive_stamp', math.nan)
        yolo_start_stamp = diag.get('yolo_start_stamp', math.nan)
        yolo_finish_stamp = diag.get('yolo_finish_stamp', math.nan)
        yolo_publish_stamp = diag.get('yolo_publish_stamp', math.nan)
        yolo_latency_s = diag.get('yolo_latency_s', math.nan)
        frame_age_at_publish_s = diag.get('frame_age_at_publish_s', math.nan)
        if math.isfinite(float(yolo_publish_stamp)):
            detector_total_latency_s = max(float(yolo_publish_stamp) - float(diag['stamp']), 0.0)
        else:
            detector_total_latency_s = (
                max(log_stamp - float(diag['stamp']), 0.0)
                if math.isfinite(float(diag.get('stamp', math.nan))) else math.nan
            )

        self.perception_writer.writerow([
            diag['stamp'],
            log_stamp,
            int(diag['detected']),
            int(true_ok),
            true_x,
            true_y,
            true_yaw,
            int(state_ok),
            state_x,
            state_y,
            state_yaw,
            state_age_s,
            1.0 if state_fresh else 0.0,
            state_pos_error,
            state_yaw_error_deg,
            diag['u_mid'],
            diag['v_mid'],
            diag['yaw_est'],
            obs_yaw_error_deg,
            int(obs_ok),
            pixel_pose_stamp,
            pixel_pose_u,
            pixel_pose_v,
            pixel_pose_yaw,
            pixel_pose_age_s,
            1.0 if pixel_pose_fresh else 0.0,
            pred_world_x,
            pred_world_y,
            localization_error_m,
            pred_world_x_calibrated,
            pred_world_y_calibrated,
            localization_error_calibrated_m,
            localization_error_captime_m,
            state_error_captime_m,
            self.bev_y_calibration_offset_m,
            diag['u_red'],
            diag['v_red'],
            diag['red_area_px'],
            diag['u_blue'],
            diag['v_blue'],
            diag['blue_area_px'],
            diag['separation_px'],
            diag['border_margin_px'],
            diag.get('yolo_score_raw', math.nan),
            diag.get('yolo_score_selected', math.nan),
            diag.get('yolo_detected_after_threshold', math.nan),
            diag.get('yolo_best_class_id', math.nan),
            diag.get('yolo_target_candidate_count', math.nan),
            diag.get('bbox_area_px', math.nan),
            diag.get('bbox_xmin', math.nan),
            diag.get('bbox_ymin', math.nan),
            diag.get('bbox_xmax', math.nan),
            diag.get('bbox_ymax', math.nan),
            diag.get('logit_margin', math.nan),
            diag.get('class_entropy', math.nan),
            diag.get('mask_area_px', math.nan),
            diag.get('mask_bottom_u', math.nan),
            diag.get('mask_bottom_v', math.nan),
            diag.get('mask_used', math.nan),
            diag.get('mask_polygon_points', math.nan),
            diag.get('confidence_logit', math.nan),
            diag.get('mask_compactness', math.nan),
            diag.get('mask_border_frac', math.nan),
            diag.get('mask_score', math.nan),
            selected_pixel_source_code,
            yolo_raw_best_score,
            yolo_selected_score,
            yolo_num_target_candidates,
            yolo_selected_class_id,
            selected_pixel_source,
            yolo_bbox_area,
            yolo_mask_area,
            yolo_inference_ms,
            detector_callback_ms,
            yolo_receive_stamp,
            yolo_start_stamp,
            yolo_finish_stamp,
            yolo_publish_stamp,
            yolo_latency_s,
            frame_age_at_publish_s,
            detector_total_latency_s,
            camera_relative_bearing_deg,
            self.seed,
        ])
        self.perception_file.flush()

    def _log_once(self):
        with self._log_write_lock:
            self._log_once_locked()

    def _log_once_locked(self):
        if getattr(self, '_event_streams_closed', False):
            return
        now_stamp = float(self.get_clock().now().nanoseconds) * 1e-9

        state_ok, state_stamp, state_x, state_y, state_yaw = self._latest_state_pose()
        state_age_s = math.nan
        if state_ok and math.isfinite(state_stamp):
            state_age_s = max(now_stamp - state_stamp, 0.0)
        state_fresh = bool(
            state_ok
            and math.isfinite(state_age_s)
            and state_age_s <= max(float(self.pixel_timeout_s), 0.0)
        )
        if self.state_msg is not None:
            cov = self.state_msg.pose.covariance
            cov_x = float(cov[0]) if len(cov) > 0 else math.nan
            cov_xy = float(cov[1]) if len(cov) > 1 else math.nan
            cov_y = float(cov[7]) if len(cov) > 7 else math.nan
            cov_yaw = float(cov[35]) if len(cov) > 35 else math.nan
        else:
            cov_x = cov_xy = cov_y = cov_yaw = math.nan

        true_ok, odom_map_stamp, true_x, true_y, true_yaw = self._latest_odom_map_pose()
        (
            planner_belief_ok,
            planner_belief_stamp,
            planner_belief_x,
            planner_belief_y,
            planner_belief_yaw,
            planner_cov_x,
            planner_cov_xy,
            planner_cov_y,
            planner_cov_yaw,
        ) = self._latest_planner_belief_pose()
        if not planner_belief_ok:
            planner_belief_stamp = math.nan
            planner_belief_x = planner_belief_y = planner_belief_yaw = math.nan
            planner_cov_x = planner_cov_xy = planner_cov_y = planner_cov_yaw = math.nan
        if planner_belief_ok and math.isfinite(planner_belief_stamp):
            planner_belief_age_s = max(0.0, now_stamp - float(planner_belief_stamp))
        else:
            planner_belief_age_s = math.nan

        if planner_belief_ok:
            est_available = 1.0
            est_x = planner_belief_x
            est_y = planner_belief_y
            est_yaw = planner_belief_yaw
            est_cov_xx = planner_cov_x
            est_cov_xy = planner_cov_xy
            est_cov_yy = planner_cov_y
        elif state_ok:
            est_available = 1.0
            est_x = float(state_x)
            est_y = float(state_y)
            est_yaw = float(state_yaw)
            est_cov_xx = float(cov_x)
            est_cov_xy = float(cov_xy)
            est_cov_yy = float(cov_y)
        else:
            est_available = 0.0
            est_x = est_y = est_yaw = math.nan
            est_cov_xx = est_cov_xy = est_cov_yy = math.nan

        state_pos_error_m = math.nan
        if true_ok and math.isfinite(est_x) and math.isfinite(est_y):
            state_pos_error_m = float(math.hypot(true_x - est_x, true_y - est_y))
        (
            state_cov_trace,
            state_cov_det,
            state_sigma_major_m,
            state_sigma_minor_m,
            state_entropy_xy,
        ) = self._covariance_metrics_2d(est_cov_xx, est_cov_xy, est_cov_yy)

        # ODOM-as-reference error signals (NOT ground truth — true_x/y = /odom,
        # which drifts from the real pose). Kept only for odom-drift diagnostics;
        # the honest errors are belief_error_gt_m / state_error_gt_m (vs gt_x/gt_y).
        # state_error_odom_m:  /odom vs /state/bev (perception output)
        # belief_error_odom_m: /odom vs /planner_belief (planner's internal belief)
        state_error_odom_m = math.nan
        after_first_cmd = bool(self._first_cmd_stamp is not None and now_stamp >= self._first_cmd_stamp)
        if true_ok and state_ok and math.isfinite(state_x) and math.isfinite(state_y):
            state_error_odom_m = float(math.hypot(true_x - state_x, true_y - state_y))
            self._state_error_odom_sum += state_error_odom_m
            if math.isfinite(state_error_odom_m):
                self._state_error_odom_count += 1
                if after_first_cmd:
                    self._state_error_odom_after_first_cmd_sum += state_error_odom_m
                    self._state_error_odom_after_first_cmd_count += 1
        belief_error_odom_m = math.nan
        if true_ok and planner_belief_ok and math.isfinite(planner_belief_x) and math.isfinite(planner_belief_y):
            belief_error_odom_m = float(math.hypot(true_x - planner_belief_x, true_y - planner_belief_y))
            self._belief_error_odom_sum += belief_error_odom_m
            if math.isfinite(belief_error_odom_m):
                self._belief_error_odom_count += 1
                if after_first_cmd:
                    self._belief_error_odom_after_first_cmd_sum += belief_error_odom_m
                    self._belief_error_odom_after_first_cmd_count += 1

        odom_ok, odom_stamp, odom_x, odom_y, odom_yaw, odom_v, odom_w = self._odom_record(self.odom_msg)
        (
            odom_noisy_ok,
            odom_noisy_stamp,
            odom_noisy_x,
            odom_noisy_y,
            odom_noisy_yaw,
            odom_noisy_v,
            odom_noisy_w,
        ) = self._odom_record(self.odom_noisy_msg)
        yaw_error_odom_noisy_vs_odom_map_rad = math.nan
        if true_ok and odom_noisy_ok and math.isfinite(odom_noisy_yaw):
            yaw_error_odom_noisy_vs_odom_map_rad = float(self._wrap_angle(odom_noisy_yaw - true_yaw))

        state_bev_yaw_latest = math.nan
        state_bev_cov_theta_theta = math.nan
        state_bev_cov_x_theta = math.nan
        state_bev_cov_y_theta = math.nan
        if self.state_msg is not None:
            state_bev_yaw_latest = self._yaw_from_quaternion(self.state_msg.pose.pose.orientation)
            _, _, _, state_bev_cov_x_theta, state_bev_cov_y_theta, state_bev_cov_theta_theta = (
                self.extract_planar_covariances(self.state_msg.pose.covariance)
            )

        planner_belief_cov_theta_theta = math.nan
        planner_belief_cov_x_theta = math.nan
        planner_belief_cov_y_theta = math.nan
        if self.planner_belief_msg is not None:
            _, _, _, planner_belief_cov_x_theta, planner_belief_cov_y_theta, planner_belief_cov_theta_theta = (
                self.extract_planar_covariances(self.planner_belief_msg.pose.covariance)
            )

        yaw_error_odom_map_vs_odom_rad = math.nan
        yaw_error_odom_map_vs_state_rad = math.nan
        yaw_error_odom_map_vs_belief_rad = math.nan
        if true_ok and odom_ok and math.isfinite(odom_yaw):
            yaw_error_odom_map_vs_odom_rad = float(self._wrap_angle(odom_yaw - true_yaw))
            self._odom_map_vs_odom_yaw_error_sum += abs(yaw_error_odom_map_vs_odom_rad)
            self._odom_map_vs_odom_yaw_error_count += 1
            if after_first_cmd:
                self._odom_map_vs_odom_yaw_error_after_first_cmd_sum += abs(yaw_error_odom_map_vs_odom_rad)
                self._odom_map_vs_odom_yaw_error_after_first_cmd_count += 1
        if true_ok and state_ok and math.isfinite(state_yaw):
            yaw_error_odom_map_vs_state_rad = float(self._wrap_angle(state_yaw - true_yaw))
            self._odom_map_vs_state_yaw_error_sum += abs(yaw_error_odom_map_vs_state_rad)
            self._truth_state_yaw_error_count += 1
            if after_first_cmd:
                self._odom_map_vs_state_yaw_error_after_first_cmd_sum += abs(yaw_error_odom_map_vs_state_rad)
                self._odom_map_vs_state_yaw_error_after_first_cmd_count += 1
        if true_ok and planner_belief_ok and math.isfinite(planner_belief_yaw):
            yaw_error_odom_map_vs_belief_rad = float(self._wrap_angle(planner_belief_yaw - true_yaw))
            self._odom_map_vs_belief_yaw_error_sum += abs(yaw_error_odom_map_vs_belief_rad)
            self._truth_belief_yaw_error_count += 1
            if after_first_cmd:
                self._odom_map_vs_belief_yaw_error_after_first_cmd_sum += abs(yaw_error_odom_map_vs_belief_rad)
                self._odom_map_vs_belief_yaw_error_after_first_cmd_count += 1

        heading_diag_stamp = math.nan
        heading_diag_age_s = math.nan
        pixel_yaw_meas = math.nan
        heading_source_code = math.nan
        heading_source = 'unknown'
        state_heading_yaw_sigma = math.nan
        state_heading_odom_age_s = math.nan
        if self.heading_diag is not None and self.heading_diag.data and len(self.heading_diag.data) >= 10:
            hdata = list(self.heading_diag.data)
            heading_diag_stamp = float(hdata[0])
            heading_source_code = float(hdata[1])
            pixel_yaw_meas = float(hdata[2])
            state_heading_yaw_sigma = float(hdata[6])
            state_heading_odom_age_s = float(hdata[9])
            heading_source = self._heading_source_name(heading_source_code)
            if math.isfinite(heading_diag_stamp):
                heading_diag_age_s = max(now_stamp - heading_diag_stamp, 0.0)

        planner_pixel_correction_available = 0.0
        planner_pixel_correction_stamp = math.nan
        planner_pixel_correction_age_s = math.nan
        pixel_corr_innov_u = math.nan
        pixel_corr_innov_v = math.nan
        pixel_corr_xy_update_norm_m = math.nan
        pixel_corr_theta_update_from_uv_rad = math.nan
        pixel_corr_nis = math.nan
        pixel_corr_accepted = math.nan
        pixel_corr_reject_reason_code = math.nan
        pixel_corr_reject_reason = 'unknown'
        pixel_corr_apply_stamp = math.nan
        pixel_corr_belief_input_stamp = math.nan
        pixel_corr_cmd_replay_count = math.nan
        pixel_corr_cmd_replay_duration_s = math.nan
        pixel_corr_cmd_replay_used_fallback = math.nan
        pixel_corr_motion_replay_source_code = math.nan
        pixel_corr_motion_replay_source = 'unknown'
        pixel_corr_nis_threshold = math.nan
        pixel_heading_correction_applied = math.nan
        pixel_heading_meas_source = math.nan
        pixel_heading_innov_rad = math.nan
        pixel_heading_gain_theta = math.nan
        pixel_corr_theta_update_total_rad = math.nan
        pixel_corr_pred_x = math.nan
        pixel_corr_pred_y = math.nan
        pixel_corr_pred_yaw = math.nan
        pixel_corr_next_x = math.nan
        pixel_corr_next_y = math.nan
        pixel_corr_next_yaw = math.nan
        pixel_corr_expected_after_u = math.nan
        pixel_corr_expected_after_v = math.nan
        pixel_corr_expected_after_visible = math.nan
        pixel_corr_K_theta_u = math.nan
        pixel_corr_K_theta_v = math.nan
        pixel_corr_measurement_space = math.nan
        pixel_corr_predict_clipped_m = math.nan
        pixel_corr_camera_index = math.nan
        if (
            self.pixel_correction_diag is not None
            and self.pixel_correction_diag.data
            and len(self.pixel_correction_diag.data) >= 20
        ):
            cdata = list(self.pixel_correction_diag.data)
            planner_pixel_correction_available = float(cdata[1])
            planner_pixel_correction_stamp = float(cdata[0])
            if math.isfinite(planner_pixel_correction_stamp):
                planner_pixel_correction_age_s = max(now_stamp - planner_pixel_correction_stamp, 0.0)
            pixel_corr_innov_u = float(cdata[6])
            pixel_corr_innov_v = float(cdata[7])
            pixel_corr_xy_update_norm_m = float(cdata[8])
            pixel_corr_theta_update_from_uv_rad = float(cdata[9])
            pixel_heading_correction_applied = float(cdata[10])
            pixel_heading_innov_rad = float(cdata[11])
            pixel_heading_gain_theta = float(cdata[12])
            pixel_corr_theta_update_total_rad = float(cdata[13])
            pixel_corr_pred_x = float(cdata[14])
            pixel_corr_pred_y = float(cdata[15])
            pixel_corr_pred_yaw = float(cdata[16])
            pixel_corr_next_x = float(cdata[17])
            pixel_corr_next_y = float(cdata[18])
            pixel_corr_next_yaw = float(cdata[19])
            if len(cdata) >= 29:
                pixel_heading_meas_source = float(cdata[28])
            if len(cdata) >= 30:
                pixel_corr_nis = float(cdata[29])
            if len(cdata) >= 38:
                pixel_corr_accepted = float(cdata[30])
                pixel_corr_reject_reason_code = float(cdata[31])
                pixel_corr_reject_reason = self._pixel_correction_reject_reason_name(
                    pixel_corr_reject_reason_code
                )
                pixel_corr_apply_stamp = float(cdata[32])
                pixel_corr_belief_input_stamp = float(cdata[33])
                pixel_corr_cmd_replay_count = float(cdata[34])
                pixel_corr_cmd_replay_duration_s = float(cdata[35])
                pixel_corr_cmd_replay_used_fallback = float(cdata[36])
                pixel_corr_nis_threshold = float(cdata[37])
            if len(cdata) >= 41:
                pixel_corr_expected_after_u = float(cdata[38])
                pixel_corr_expected_after_v = float(cdata[39])
                pixel_corr_expected_after_visible = float(cdata[40])
            if len(cdata) >= 42:
                pixel_corr_motion_replay_source_code = float(cdata[41])
                pixel_corr_motion_replay_source = self._pixel_correction_motion_replay_source_name(
                    pixel_corr_motion_replay_source_code
                )
            if len(cdata) >= 44:
                pixel_corr_K_theta_u = float(cdata[42])
                pixel_corr_K_theta_v = float(cdata[43])
            if len(cdata) >= 46:
                pixel_corr_measurement_space = float(cdata[44])
                pixel_corr_predict_clipped_m = float(cdata[45])
            if len(cdata) >= 47:
                pixel_corr_camera_index = float(cdata[46])

        cmd_v = self.cmd_msg.linear.x if self.cmd_msg else 0.0
        cmd_w = self.cmd_msg.angular.z if self.cmd_msg else 0.0
        cmd_raw_v = self.cmd_raw_msg.linear.x if self.cmd_raw_msg else cmd_v
        cmd_raw_w = self.cmd_raw_msg.angular.z if self.cmd_raw_msg else cmd_w
        cmd_stamp = self.cmd_stamp_s
        cmd_raw_stamp = self.cmd_raw_stamp_s
        cmd_age_s = max(now_stamp - cmd_stamp, 0.0) if math.isfinite(cmd_stamp) else math.nan
        cmd_raw_age_s = max(now_stamp - cmd_raw_stamp, 0.0) if math.isfinite(cmd_raw_stamp) else math.nan
        cmd_noise_enabled = math.nan
        cmd_noise_linear_multiplier = math.nan
        cmd_noise_angular_multiplier = math.nan
        cmd_noise_linear_additive = math.nan
        cmd_noise_angular_additive = math.nan
        if self.cmd_noise_diag is not None and self.cmd_noise_diag.data and len(self.cmd_noise_diag.data) >= 10:
            ndata = list(self.cmd_noise_diag.data)
            cmd_noise_enabled = float(ndata[1])
            cmd_raw_v = float(ndata[2])
            cmd_raw_w = float(ndata[3])
            cmd_noise_linear_multiplier = float(ndata[6])
            cmd_noise_angular_multiplier = float(ndata[7])
            cmd_noise_linear_additive = float(ndata[8])
            cmd_noise_angular_additive = float(ndata[9])
        cmd_noise_v_error = float(cmd_v - cmd_raw_v)
        cmd_noise_w_error = float(cmd_w - cmd_raw_w)
        self._maybe_log_frame_sanity(now_stamp, cmd_v, cmd_w)

        goal_x = math.nan
        goal_y = math.nan
        # Goal distance & executed path from the TRUE Gazebo pose ONLY (no /odom
        # fallback). goal_reached / min_goal_distance are outcome metrics; scoring
        # them on drifting wheel-odom would (like the collision metric) misreport
        # whether the TRUE robot reached the goal. If gt is unavailable -> NaN.
        goal_dist = math.nan
        goal_x, goal_y = self._active_goal_xy()
        if math.isfinite(goal_x) and math.isfinite(goal_y) and self._gt_xy is not None:
            goal_dist = math.hypot(goal_x - self._gt_xy[0], goal_y - self._gt_xy[1])

        operational_goal_dist_m = math.nan
        if (
            math.isfinite(goal_x)
            and math.isfinite(goal_y)
            and planner_belief_ok
            and math.isfinite(planner_belief_x)
            and math.isfinite(planner_belief_y)
        ):
            # This is the state the controller actually uses. It alone may
            # drive automatic goal and stuck termination.
            operational_goal_dist_m = math.hypot(
                goal_x - planner_belief_x, goal_y - planner_belief_y
            )

        current_pose = None
        if self._gt_xy is not None:
            current_pose = (float(self._gt_xy[0]), float(self._gt_xy[1]))
            if self._last_path_pose is not None:
                self._cumulative_path_length += math.hypot(current_pose[0] - self._last_path_pose[0], current_pose[1] - self._last_path_pose[1])
            self._last_path_pose = current_pose

        if math.isfinite(goal_dist):
            self._min_goal_distance = min(self._min_goal_distance, goal_dist)
            self._update_goal_region_state(now_stamp, goal_dist)

        self._remember_motion_sample(
            now_stamp,
            planner_belief_x,
            planner_belief_y,
            operational_goal_dist_m,
            cmd_v,
            cmd_w,
            operational_yaw=planner_belief_yaw,
        )
        self._update_terminal_stop_verification(now_stamp)

        plan_points = 0
        plan_length = 0.0
        if self.plan_msg and self.plan_msg.poses:
            plan_points = len(self.plan_msg.poses)
            for i in range(1, plan_points):
                p0 = self.plan_msg.poses[i - 1].pose.position
                p1 = self.plan_msg.poses[i].pose.position
                plan_length += math.hypot(p1.x - p0.x, p1.y - p0.y)

        stamp = now_stamp

        efe_total = 0.0
        efe_risk = 0.0
        efe_ambiguity = 0.0
        efe_control = 0.0
        efe_obstacle = 0.0
        efe_risk_mean = math.nan
        efe_risk_cov_trace = math.nan
        efe_risk_cov_logdet = math.nan
        efe_delta_risk_visibility = math.nan
        efe_delta_ambiguity_visibility = math.nan
        active_plan_age_s = math.nan
        active_plan_remaining_s = math.nan
        active_control_index = math.nan
        active_controls_len = math.nan
        active_controls_original_len = math.nan
        latency_skip_steps = math.nan
        latency_skip_s = math.nan
        command_timer_period_s = math.nan
        planner_timer_period_s = math.nan
        pending_plan_started_active_remaining_s = math.nan
        exec_plan_age_s = math.nan
        exec_plan_remaining_s = math.nan
        exec_control_index = math.nan
        exec_controls_len = math.nan
        exec_controls_original_len = math.nan
        exec_cmd_v = math.nan
        exec_cmd_w = math.nan
        exec_latency_skip_steps = math.nan
        exec_latency_skip_s = math.nan
        exec_wp_idx = math.nan
        exec_wp_count = math.nan
        exec_wp_target_x = math.nan
        exec_wp_target_y = math.nan
        exec_wp_dist_m = math.nan
        exec_desired_yaw = math.nan
        exec_yaw_error = math.nan
        exec_tracking_yaw = math.nan
        exec_tracking_yaw_source = math.nan
        optimizer_success = 0.0
        optimizer_status = 0.0
        optimizer_nit = 0.0
        optimizer_nfev = 0.0
        optimizer_message = self.planner_diag_text
        planner_diag_prediction_source = math.nan
        planner_diag_prediction_dt = math.nan
        planner_diag_u_pred_v = math.nan
        planner_diag_u_pred_omega = math.nan
        planner_diag_Q_theta_theta = math.nan
        planner_diag_odom_delta_theta = math.nan
        planner_diag_cmd_delta_theta = math.nan
        planner_diag_heading_anchor_applied = math.nan
        planner_diag_state_bev_yaw_ignored = math.nan
        plan_time_ms = 0.0
        solve_time_ms = 0.0
        measurement_available = math.nan
        belief_age_s = math.nan
        p_vis_plan = math.nan
        p_vis_plan_eff = math.nan
        r_plan_u_std = math.nan
        r_plan_v_std = math.nan
        terminal_goal_distance_pred = math.nan
        terminal_goal_progress_m = math.nan
        fraction_horizon_low_pvis = math.nan
        fraction_horizon_high_ambiguity = math.nan
        min_predicted_obstacle_distance_m = math.nan
        rollout_valid = math.nan
        if self.planner_diag and self.planner_diag.data and len(self.planner_diag.data) >= 6:
            optimizer_success = float(self.planner_diag.data[0])
            optimizer_status = float(self.planner_diag.data[1])
            optimizer_nit = float(self.planner_diag.data[2])
            optimizer_nfev = float(self.planner_diag.data[3])
            plan_time_ms = float(self.planner_diag.data[4])
            solve_time_ms = float(self.planner_diag.data[5])

            if len(self.planner_diag.data) >= 12:
                p_vis_plan = float(self.planner_diag.data[6])
                p_vis_plan_eff = float(self.planner_diag.data[7])
                r_plan_u_std = float(self.planner_diag.data[8])
                r_plan_v_std = float(self.planner_diag.data[9])
                measurement_available = float(self.planner_diag.data[10])
                belief_age_s = float(self.planner_diag.data[11])
            if len(self.planner_diag.data) >= 18:
                terminal_goal_distance_pred = float(self.planner_diag.data[12])
                terminal_goal_progress_m = float(self.planner_diag.data[13])
                fraction_horizon_low_pvis = float(self.planner_diag.data[14])
                fraction_horizon_high_ambiguity = float(self.planner_diag.data[15])
                min_predicted_obstacle_distance_m = float(self.planner_diag.data[16])
                rollout_valid = float(self.planner_diag.data[17])
            if len(self.planner_diag.data) >= 23:
                efe_risk_mean = float(self.planner_diag.data[18])
                efe_risk_cov_trace = float(self.planner_diag.data[19])
                efe_risk_cov_logdet = float(self.planner_diag.data[20])
                efe_delta_risk_visibility = float(self.planner_diag.data[21])
                efe_delta_ambiguity_visibility = float(self.planner_diag.data[22])
            if len(self.planner_diag.data) >= 33:
                active_plan_age_s = float(self.planner_diag.data[23])
                active_plan_remaining_s = float(self.planner_diag.data[24])
                active_control_index = float(self.planner_diag.data[25])
                active_controls_len = float(self.planner_diag.data[26])
                active_controls_original_len = float(self.planner_diag.data[27])
                latency_skip_steps = float(self.planner_diag.data[28])
                latency_skip_s = float(self.planner_diag.data[29])
                command_timer_period_s = float(self.planner_diag.data[30])
                planner_timer_period_s = float(self.planner_diag.data[31])
                pending_plan_started_active_remaining_s = float(self.planner_diag.data[32])
            if len(self.planner_diag.data) >= 42:
                planner_diag_prediction_source = float(self.planner_diag.data[33])
                planner_diag_prediction_dt = float(self.planner_diag.data[34])
                planner_diag_u_pred_v = float(self.planner_diag.data[35])
                planner_diag_u_pred_omega = float(self.planner_diag.data[36])
                planner_diag_Q_theta_theta = float(self.planner_diag.data[37])
                planner_diag_odom_delta_theta = float(self.planner_diag.data[38])
                planner_diag_cmd_delta_theta = float(self.planner_diag.data[39])
                planner_diag_heading_anchor_applied = float(self.planner_diag.data[40])
                planner_diag_state_bev_yaw_ignored = float(self.planner_diag.data[41])

        if (
            self.active_execution_diag
            and self.active_execution_diag.data
            and len(self.active_execution_diag.data) >= 9
        ):
            exec_plan_age_s = float(self.active_execution_diag.data[0])
            exec_plan_remaining_s = float(self.active_execution_diag.data[1])
            exec_control_index = float(self.active_execution_diag.data[2])
            exec_controls_len = float(self.active_execution_diag.data[3])
            exec_controls_original_len = float(self.active_execution_diag.data[4])
            exec_cmd_v = float(self.active_execution_diag.data[5])
            exec_cmd_w = float(self.active_execution_diag.data[6])
            exec_latency_skip_steps = float(self.active_execution_diag.data[7])
            exec_latency_skip_s = float(self.active_execution_diag.data[8])
            if len(self.active_execution_diag.data) >= 18:
                exec_wp_idx = float(self.active_execution_diag.data[9])
                exec_wp_count = float(self.active_execution_diag.data[10])
                exec_wp_target_x = float(self.active_execution_diag.data[11])
                exec_wp_target_y = float(self.active_execution_diag.data[12])
                exec_wp_dist_m = float(self.active_execution_diag.data[13])
                exec_desired_yaw = float(self.active_execution_diag.data[14])
                exec_yaw_error = float(self.active_execution_diag.data[15])
                exec_tracking_yaw = float(self.active_execution_diag.data[16])
                exec_tracking_yaw_source = float(self.active_execution_diag.data[17])

        if self.efe_metrics and self.efe_metrics.data and len(self.efe_metrics.data) >= 5:
            efe_total = float(self.efe_metrics.data[0])
            efe_risk = float(self.efe_metrics.data[1])
            efe_ambiguity = float(self.efe_metrics.data[2])
            efe_control = float(self.efe_metrics.data[3])
            efe_obstacle = float(self.efe_metrics.data[4])
            if len(self.efe_metrics.data) >= 23:
                efe_risk_mean = float(self.efe_metrics.data[18])
                efe_risk_cov_trace = float(self.efe_metrics.data[19])
                efe_risk_cov_logdet = float(self.efe_metrics.data[20])
                efe_delta_risk_visibility = float(self.efe_metrics.data[21])
                efe_delta_ambiguity_visibility = float(self.efe_metrics.data[22])

        valid_run = 1.0 if self._valid_run else 0.0
        invalid_reason = self._invalid_reason

        # --- GROUND-TRUTH errors (vs TRUE Gazebo pose, not /odom) ---
        # /odom ("odom_map_*") is DiffDrive wheel odometry and drifts in turns;
        # these *_gt columns are the honest errors against the real pose.
        #
        # TIME ALIGNMENT IS THE WHOLE POINT HERE. `belief_error_gt_m` and
        # `state_error_gt_m` score each estimate against the truth AT THE INSTANT THE
        # ESTIMATE DESCRIBES, taken from the stamped ground-truth buffer. They used to
        # use the latest held truth, i.e. the truth at LOG time, which is later by one
        # full publish cycle: the belief publishes at 10 Hz and the logger samples
        # 10 Hz behind it, so the column carried a fixed 100 ms of robot travel. That
        # inflated the median belief error by 2.3x (2.75 cm vs 1.13 cm) and NEES by
        # 1.9x on the six-arm drives, identically on every arm, so it read as a
        # property of the camera network rather than of the logger.
        #
        # The `*_logtime_m` twins keep the old, misaligned definition so a run can be
        # compared with the pre-2026-08-28 campaigns and so the size of the artefact
        # stays visible in the data instead of living in a memo.
        gt_ok = self._gt_xy is not None
        gt_x = self._gt_xy[0] if gt_ok else math.nan
        gt_y = self._gt_xy[1] if gt_ok else math.nan
        gt_yaw = self._gt_yaw if (gt_ok and self._gt_yaw is not None) else math.nan
        gt_stamp = float(self._gt_stamp)
        gt_age_s = (max(now_stamp - gt_stamp, 0.0)
                    if math.isfinite(gt_stamp) else math.nan)
        belief_error_gt_m = math.nan
        state_error_gt_m = math.nan
        belief_error_gt_logtime_m = math.nan
        state_error_gt_logtime_m = math.nan
        gt_x_at_belief_stamp = math.nan
        gt_y_at_belief_stamp = math.nan
        gt_x_at_state_stamp = math.nan
        gt_y_at_state_stamp = math.nan
        odom_map_gt_drift_m = math.nan
        belief_yaw_error_gt_rad = math.nan

        # Aligned errors need only the stamped buffer, never the held latest value.
        if planner_belief_ok:
            (belief_error_gt_m, gt_x_at_belief_stamp,
             gt_y_at_belief_stamp) = self._error_against_gt_at(
                planner_belief_x, planner_belief_y, planner_belief_stamp)
        if state_ok:
            (state_error_gt_m, gt_x_at_state_stamp,
             gt_y_at_state_stamp) = self._error_against_gt_at(
                state_x, state_y, state_stamp)
        if planner_belief_ok and math.isfinite(planner_belief_yaw):
            byaw_ok, _bgx, _bgy, gt_yaw_at_belief = self._gt_at(planner_belief_stamp)
            if byaw_ok:
                belief_yaw_error_gt_rad = self._wrap_angle(
                    planner_belief_yaw - gt_yaw_at_belief)

        if gt_ok:
            if planner_belief_ok and math.isfinite(planner_belief_x):
                belief_error_gt_logtime_m = math.hypot(
                    planner_belief_x - gt_x, planner_belief_y - gt_y)
            if state_ok and math.isfinite(state_x):
                state_error_gt_logtime_m = math.hypot(state_x - gt_x, state_y - gt_y)
            if true_ok and math.isfinite(true_x):
                odom_map_gt_drift_m = math.hypot(true_x - gt_x, true_y - gt_y)
        # Run-summary means accumulate the ALIGNED errors.
        if math.isfinite(belief_error_gt_m):
            self._belief_error_gt_sum += belief_error_gt_m
            self._belief_error_gt_count += 1
            if after_first_cmd:
                self._belief_error_gt_after_first_cmd_sum += belief_error_gt_m
                self._belief_error_gt_after_first_cmd_count += 1
        if math.isfinite(state_error_gt_m):
            self._state_error_gt_sum += state_error_gt_m
            self._state_error_gt_count += 1
            if after_first_cmd:
                self._state_error_gt_after_first_cmd_sum += state_error_gt_m
                self._state_error_gt_after_first_cmd_count += 1

        fusion_cameras_n = float('nan')
        fusion_cameras = ''
        fusion_candidates_n = float('nan')
        fusion_candidates = ''
        fusion_decision_age_s = float('nan')
        if self.fusion_decision is not None:
            accepted = self.fusion_decision.get('accepted_camera_ids') or []
            fusion_cameras_n = float(len(accepted))
            fusion_cameras = '|'.join(str(c).replace('camera_', '') for c in accepted)
            candidates = self.fusion_decision.get('synchronous_camera_ids')
            if candidates is None:
                n_fresh = self.fusion_decision.get('n_fresh')
                fusion_candidates_n = float(n_fresh) if n_fresh is not None else float('nan')
            else:
                fusion_candidates_n = float(len(candidates))
                fusion_candidates = '|'.join(
                    str(c).replace('camera_', '') for c in candidates)
            if self.fusion_decision_stamp is not None:
                fusion_decision_age_s = float(stamp) - float(self.fusion_decision_stamp)

        self.writer.writerow([
            stamp,
            1.0 if true_ok else 0.0, odom_map_stamp, true_x, true_y, true_yaw,
            1.0 if state_ok else 0.0, state_stamp, state_x, state_y, state_yaw,
            state_age_s, 1.0 if state_fresh else 0.0,
            cov_x, cov_xy, cov_y, cov_yaw,
            1.0 if planner_belief_ok else 0.0, planner_belief_stamp,
            planner_belief_age_s,
            planner_belief_x, planner_belief_y, planner_belief_yaw,
            planner_cov_x, planner_cov_xy, planner_cov_y, planner_cov_yaw,
            est_available, est_x, est_y, est_yaw,
            est_cov_xx, est_cov_xy, est_cov_yy,
            state_pos_error_m, state_cov_trace, state_cov_det,
            state_sigma_major_m, state_sigma_minor_m, state_entropy_xy,
            state_error_odom_m, belief_error_odom_m,
            1.0 if odom_ok else 0.0, odom_stamp, odom_x, odom_y, odom_yaw,
            odom_v, odom_w,
            1.0 if odom_noisy_ok else 0.0, odom_noisy_stamp, odom_noisy_x, odom_noisy_y,
            odom_noisy_yaw, odom_noisy_v, odom_noisy_w,
            yaw_error_odom_map_vs_odom_rad, yaw_error_odom_map_vs_state_rad, yaw_error_odom_map_vs_belief_rad,
            pixel_yaw_meas, heading_source_code, heading_source,
            heading_diag_stamp, heading_diag_age_s,
            state_heading_yaw_sigma, state_heading_odom_age_s,
            planner_pixel_correction_available, planner_pixel_correction_stamp,
            planner_pixel_correction_age_s,
            pixel_corr_innov_u, pixel_corr_innov_v,
            pixel_corr_xy_update_norm_m, pixel_corr_theta_update_from_uv_rad,
            pixel_corr_nis,
            pixel_corr_accepted, pixel_corr_reject_reason_code, pixel_corr_reject_reason,
            pixel_corr_apply_stamp, pixel_corr_belief_input_stamp,
            pixel_corr_cmd_replay_count, pixel_corr_cmd_replay_duration_s,
            pixel_corr_cmd_replay_used_fallback,
            pixel_corr_motion_replay_source_code, pixel_corr_motion_replay_source,
            pixel_corr_nis_threshold,
            pixel_heading_correction_applied, pixel_heading_meas_source,
            pixel_heading_innov_rad,
            pixel_heading_gain_theta, pixel_corr_theta_update_total_rad,
            pixel_corr_pred_x, pixel_corr_pred_y, pixel_corr_pred_yaw,
            pixel_corr_next_x, pixel_corr_next_y, pixel_corr_next_yaw,
            pixel_corr_expected_after_u, pixel_corr_expected_after_v,
            pixel_corr_expected_after_visible,
            cmd_v, cmd_w,
            cmd_raw_v, cmd_raw_w,
            cmd_stamp, cmd_age_s, cmd_raw_stamp, cmd_raw_age_s,
            cmd_noise_enabled,
            cmd_noise_linear_multiplier, cmd_noise_angular_multiplier,
            cmd_noise_linear_additive, cmd_noise_angular_additive,
            cmd_noise_v_error, cmd_noise_w_error,
            goal_x, goal_y, goal_dist,
            operational_goal_dist_m, 'planner_belief',
            plan_points, plan_length,
            optimizer_success, optimizer_status, optimizer_nit, optimizer_nfev, optimizer_message,
            plan_time_ms, solve_time_ms,
            measurement_available, belief_age_s,
            p_vis_plan, p_vis_plan_eff,
            r_plan_u_std, r_plan_v_std,
            terminal_goal_distance_pred, terminal_goal_progress_m,
            fraction_horizon_low_pvis, fraction_horizon_high_ambiguity,
            min_predicted_obstacle_distance_m, rollout_valid,
            efe_total, efe_risk, efe_ambiguity, efe_control, efe_obstacle,
            efe_risk_mean, efe_risk_cov_trace, efe_risk_cov_logdet,
            efe_delta_risk_visibility, efe_delta_ambiguity_visibility,
            active_plan_age_s, active_plan_remaining_s, active_control_index,
            active_controls_len, active_controls_original_len,
            latency_skip_steps, latency_skip_s,
            command_timer_period_s, planner_timer_period_s,
            pending_plan_started_active_remaining_s,
            exec_plan_age_s, exec_plan_remaining_s, exec_control_index,
            exec_controls_len, exec_controls_original_len,
            exec_cmd_v, exec_cmd_w, exec_latency_skip_steps,
            exec_latency_skip_s,
            exec_wp_idx, exec_wp_count, exec_wp_target_x, exec_wp_target_y,
            exec_wp_dist_m, exec_desired_yaw, exec_yaw_error,
            exec_tracking_yaw, exec_tracking_yaw_source,
            valid_run, invalid_reason,
            self.heading_update_mode,
            pixel_corr_K_theta_u, pixel_corr_K_theta_v,
            pixel_corr_measurement_space, pixel_corr_predict_clipped_m,
            pixel_corr_camera_index,
            yaw_error_odom_noisy_vs_odom_map_rad,
            state_bev_yaw_latest,
            state_bev_cov_theta_theta, state_bev_cov_x_theta, state_bev_cov_y_theta,
            planner_belief_cov_theta_theta, planner_belief_cov_x_theta, planner_belief_cov_y_theta,
            planner_diag_prediction_source, planner_diag_prediction_dt,
            planner_diag_u_pred_v, planner_diag_u_pred_omega, planner_diag_Q_theta_theta,
            planner_diag_odom_delta_theta, planner_diag_cmd_delta_theta,
            planner_diag_heading_anchor_applied, planner_diag_state_bev_yaw_ignored,
            1.0 if gt_ok else 0.0, gt_x, gt_y, gt_yaw, gt_stamp, gt_age_s,
            belief_error_gt_m, state_error_gt_m, odom_map_gt_drift_m,
            belief_yaw_error_gt_rad,
            gt_x_at_belief_stamp, gt_y_at_belief_stamp,
            gt_x_at_state_stamp, gt_y_at_state_stamp,
            belief_error_gt_logtime_m, state_error_gt_logtime_m,
            self.seed,
            fusion_cameras_n, fusion_cameras, fusion_decision_age_s,
            fusion_candidates_n, fusion_candidates,
        ])
        self.file.flush()

        if not self._stop_requested:
            if self._first_cmd_stamp is None:
                if self._command_active(cmd_v, cmd_w):
                    self._first_cmd_stamp = now_stamp
                    self.get_logger().info(f"First command detected. Starting {self.run_timeout_after_first_cmd_s:.1f}s timeout.")
            else:
                elapsed = now_stamp - self._first_cmd_stamp
                if elapsed >= self.run_timeout_after_first_cmd_s:
                    self._finish_run("timeout_after_first_cmd", now_stamp)
                    return

        if not self._stop_requested and self._maybe_finish_for_goal(
            now_stamp, operational_goal_dist_m, cmd_v, cmd_w
        ):
            return

        if not self._stop_requested and self._maybe_finish_for_stuck(
            now_stamp, operational_goal_dist_m
        ):
            return

    def _finish_run(self, reason: str, stamp: float = None):
        """Request a latched stop, then wait for acknowledgements and ledger drain."""
        with self._event_lock:
            if self._stop_requested:
                return
            self._stop_requested = True
            self._finish_reason = str(reason)
            self._finish_stamp = (
                float(stamp) if stamp is not None
                else float(self.get_clock().now().nanoseconds) * 1e-9
            )
            self._finish_requested_wall_s = time.monotonic()
            decision_stamp_ns = max(int(round(self._finish_stamp * 1e9)), 0)
            request = TerminalStopRequest(
                request_id=(
                    f"logger:{self.run_id}:terminal:{decision_stamp_ns}"
                ),
                run_id=self.run_id,
                reason=self._finish_reason,
                decision_stamp_ns=decision_stamp_ns,
            )
            self._terminal_stop_request_id = request.request_id
            self._terminal_stop_request_payload = request
        message = String()
        message.data = terminal_stop_request_to_json(request)
        self._terminal_stop_request_pub.publish(message)
        threading.Timer(0.05, self._finalize_run).start()

    def _correction_ledger_result(self):
        return validate_correction_ledger(
            list(self._correction_publications.values()),
            list(self._assimilation_payloads.values()),
        )

    def _flush_close_data_files(self):
        """Attempt every stream even when one flush/close fails."""
        lock = getattr(self, '_log_write_lock', None)
        if lock is None:  # Minimal unit-test fixtures and legacy callers.
            return ExperimentLogger._flush_close_data_files_locked(self)
        with lock:
            return ExperimentLogger._flush_close_data_files_locked(self)

    def _flush_close_data_files_locked(self):
        errors = []
        names = (
            'file', 'plan_file', 'perception_file', 'fusion_obs_file',
            'assimilation_file', 'correction_publication_file', 'ground_truth_pose_file',
            'camera_opportunity_file', 'runtime_event_file',
            'belief_prediction_file',
        )
        for name in names:
            handle = getattr(self, name, None)
            if handle is None or getattr(handle, 'closed', False):
                continue
            try:
                handle.flush()
                try:
                    os.fsync(handle.fileno())
                except (AttributeError, OSError):
                    pass
            except Exception as exc:
                errors.append(f'{name}:flush:{type(exc).__name__}:{exc}')
            try:
                handle.close()
            except Exception as exc:
                errors.append(f'{name}:close:{type(exc).__name__}:{exc}')
        self._data_file_close_errors = errors
        self._event_streams_closed = True
        return errors

    def _finalize_run(self):
        """Freeze the event cutoff, reconcile the ledger, and atomically commit summary."""
        with self._event_lock:
            if self._completed or self._finalizing:
                return
            now_wall = time.monotonic()
            requested = float(self._finish_requested_wall_s)
            elapsed = max(now_wall - requested, 0.0)
            quiet = max(now_wall - float(self._last_event_wall_s), 0.0)
            ledger_ready = self._correction_ledger_result().valid
            acknowledged = set(self._terminal_stop_acks)
            acknowledgements_ready = TERMINAL_COMPONENTS.issubset(acknowledged)
            terminal_ready = bool(
                ledger_ready
                and acknowledgements_ready
                and self._terminal_stop_verified
            )
            if ((not terminal_ready or quiet < EVENT_DRAIN_QUIET_S)
                    and elapsed < EVENT_DRAIN_MAX_S):
                delay = min(0.05, max(EVENT_DRAIN_MAX_S - elapsed, 0.01))
                threading.Timer(delay, self._finalize_run).start()
                return
            if not ledger_ready:
                self._record_invalid('terminal_stop_correction_ledger_not_drained')
            if not acknowledgements_ready:
                missing = sorted(TERMINAL_COMPONENTS - acknowledged)
                self._record_invalid(
                    'terminal_stop_ack_timeout:' + ','.join(missing)
                )
            if not self._terminal_stop_verified:
                self._record_invalid('terminal_stop_not_verified')
            self._finalizing = True
            self._accepting_events = False
            reason = self._finish_reason
            stamp = self._finish_stamp
            event_cutoff_wall_s = now_wall
            event_drain_elapsed_s = elapsed

        with self._log_write_lock:
            return self._finalize_run_locked(
                reason=reason,
                stamp=stamp,
                event_cutoff_wall_s=event_cutoff_wall_s,
                event_drain_elapsed_s=event_drain_elapsed_s,
            )

    def _finalize_run_locked(
        self, *, reason, stamp, event_cutoff_wall_s, event_drain_elapsed_s
    ):
        """Commit a terminal summary while the periodic row writer is excluded."""
        ledger = self._correction_ledger_result()
        ledger_required = bool(
            getattr(self, 'require_state_correction_envelope', False)
            and getattr(self, 'state_correction_mode', '') == 'fused')
        ledger_applicable = bool(
            ledger_required or self._correction_publications
            or self._assimilation_payloads)
        if ledger_applicable and not ledger.valid:
            codes = ','.join(sorted({issue.code for issue in ledger.errors}))
            self._record_invalid(f'correction_ledger_invalid:{codes}')
        schema2_terminal_count = sum(
            int(payload.get('schema_version', 1)) == 2
            for payload in self._assimilation_payloads.values())
        committed_posterior_complete = bool(
            not self._assimilation_payloads
            or schema2_terminal_count == len(self._assimilation_payloads))
        if ledger_required and not committed_posterior_complete:
            self._record_invalid('committed_posterior_missing')

        mean_efe_risk = self._efe_risk_sum / max(self._efe_count, 1) if self._efe_count > 0 else math.nan
        mean_efe_ambiguity = self._efe_ambiguity_sum / max(self._efe_count, 1) if self._efe_count > 0 else math.nan
        mean_efe_control = self._efe_control_sum / max(self._efe_count, 1) if self._efe_count > 0 else math.nan
        mean_efe_obstacle = self._efe_obstacle_sum / max(self._efe_count, 1) if self._efe_count > 0 else math.nan
        mean_solve_time_ms = self._solve_time_ms_sum / max(self._solve_count, 1) if self._solve_count > 0 else math.nan
        mean_p_vis_plan = self._p_vis_plan_sum / max(self._p_vis_count, 1) if self._p_vis_count > 0 else math.nan
        mean_p_vis_plan_eff = self._p_vis_plan_eff_sum / max(self._p_vis_count, 1) if self._p_vis_count > 0 else math.nan
        mean_r_plan_u_std = self._r_plan_u_std_sum / max(self._p_vis_count, 1) if self._p_vis_count > 0 else math.nan
        mean_r_plan_v_std = self._r_plan_v_std_sum / max(self._p_vis_count, 1) if self._p_vis_count > 0 else math.nan
        fraction_time_p_vis_below_0_2 = self._p_vis_plan_below_0_2_count / max(self._p_vis_count, 1) if self._p_vis_count > 0 else math.nan
        fraction_time_p_vis_eff_below_0_2 = self._p_vis_plan_eff_below_0_2_count / max(self._p_vis_count, 1) if self._p_vis_count > 0 else math.nan
        max_r_plan_std = self._max_r_plan_std if self._p_vis_count > 0 else math.nan
        mean_state_error_odom_m = (
            self._state_error_odom_sum / self._state_error_odom_count
            if self._state_error_odom_count > 0 else math.nan
        )
        mean_belief_error_odom_m = (
            self._belief_error_odom_sum / self._belief_error_odom_count
            if self._belief_error_odom_count > 0 else math.nan
        )
        mean_state_error_odom_after_first_cmd_m = (
            self._state_error_odom_after_first_cmd_sum
            / self._state_error_odom_after_first_cmd_count
            if self._state_error_odom_after_first_cmd_count > 0 else math.nan
        )
        mean_belief_error_odom_after_first_cmd_m = (
            self._belief_error_odom_after_first_cmd_sum
            / self._belief_error_odom_after_first_cmd_count
            if self._belief_error_odom_after_first_cmd_count > 0 else math.nan
        )
        # GROUND-TRUTH error means (the honest localization metric)
        mean_belief_error_gt_m = (
            self._belief_error_gt_sum / self._belief_error_gt_count
            if self._belief_error_gt_count > 0 else math.nan
        )
        mean_state_error_gt_m = (
            self._state_error_gt_sum / self._state_error_gt_count
            if self._state_error_gt_count > 0 else math.nan
        )
        mean_belief_error_gt_after_first_cmd_m = (
            self._belief_error_gt_after_first_cmd_sum
            / self._belief_error_gt_after_first_cmd_count
            if self._belief_error_gt_after_first_cmd_count > 0 else math.nan
        )
        mean_state_error_gt_after_first_cmd_m = (
            self._state_error_gt_after_first_cmd_sum
            / self._state_error_gt_after_first_cmd_count
            if self._state_error_gt_after_first_cmd_count > 0 else math.nan
        )
        mean_abs_odom_map_vs_odom_yaw_error_rad = (
            self._odom_map_vs_odom_yaw_error_sum / self._odom_map_vs_odom_yaw_error_count
            if self._odom_map_vs_odom_yaw_error_count > 0 else math.nan
        )
        mean_abs_odom_map_vs_state_yaw_error_rad = (
            self._odom_map_vs_state_yaw_error_sum / self._truth_state_yaw_error_count
            if self._truth_state_yaw_error_count > 0 else math.nan
        )
        mean_abs_odom_map_vs_belief_yaw_error_rad = (
            self._odom_map_vs_belief_yaw_error_sum / self._truth_belief_yaw_error_count
            if self._truth_belief_yaw_error_count > 0 else math.nan
        )
        mean_abs_odom_map_vs_odom_yaw_error_after_first_cmd_rad = (
            self._odom_map_vs_odom_yaw_error_after_first_cmd_sum
            / self._odom_map_vs_odom_yaw_error_after_first_cmd_count
            if self._odom_map_vs_odom_yaw_error_after_first_cmd_count > 0 else math.nan
        )
        mean_abs_odom_map_vs_state_yaw_error_after_first_cmd_rad = (
            self._odom_map_vs_state_yaw_error_after_first_cmd_sum
            / self._odom_map_vs_state_yaw_error_after_first_cmd_count
            if self._odom_map_vs_state_yaw_error_after_first_cmd_count > 0 else math.nan
        )
        mean_abs_odom_map_vs_belief_yaw_error_after_first_cmd_rad = (
            self._odom_map_vs_belief_yaw_error_after_first_cmd_sum
            / self._odom_map_vs_belief_yaw_error_after_first_cmd_count
            if self._odom_map_vs_belief_yaw_error_after_first_cmd_count > 0 else math.nan
        )

        elapsed_after_first_cmd_s = stamp - self._first_cmd_stamp if self._first_cmd_stamp is not None else 0.0
        if (
            self._first_cmd_stamp is not None
            and math.isfinite(self._goal_region_first_stamp)
        ):
            goal_region_after_first_cmd_s = float(self._goal_region_first_stamp - self._first_cmd_stamp)
        else:
            goal_region_after_first_cmd_s = math.nan

        # Distance to the goal at the end of the run, from the TRUE pose.
        #
        # This read `_latest_odom_map_pose()` -- wheel odometry -- while the
        # goal-reached DECISION twenty lines up uses `_gt_xy`. So a run could be
        # correctly recorded as `goal_reached` and simultaneously report having ended
        # metres away: measured 0.44/0.62/2.46/0.75 m for F1/F4/O1/O2 against true
        # distances of 0.03/0.03/0.07/0.13 m, the difference being exactly that run's
        # accumulated wheel slip. The odometry-referenced number is kept beside it,
        # named for what it is, because it is a useful drift diagnostic.
        final_goal_distance = math.nan
        final_goal_distance_odom = math.nan
        odom_ok, _odom_stamp, odom_pose_x, odom_pose_y, _odom_yaw = (
            self._latest_odom_map_pose())
        goal_x, goal_y = self._active_goal_xy()
        if math.isfinite(goal_x) and math.isfinite(goal_y):
            if self._gt_xy is not None:
                final_goal_distance = math.hypot(
                    goal_x - self._gt_xy[0], goal_y - self._gt_xy[1])
            if odom_ok and math.isfinite(odom_pose_x):
                final_goal_distance_odom = math.hypot(
                    goal_x - odom_pose_x, goal_y - odom_pose_y)

        # Collision is not decided here: it is scored offline from ground_truth_pose.csv
        # as the robot footprint leaving the driveable region.
        goal_region_success = bool(self._goal_region_reached() and self._valid_run)

        summary = {
            'completed': True,
            'completion_reason': reason,
            'termination_reference': 'planner_belief_or_timeout',
            'collision_scoring': 'offline_footprint_vs_driveable_region_from_ground_truth_pose_csv',
            'first_cmd_stamp': self._first_cmd_stamp if self._first_cmd_stamp is not None else math.nan,
            'stop_stamp': stamp,
            'elapsed_after_first_cmd_s': elapsed_after_first_cmd_s,
            'path_length_m': self._cumulative_path_length,
            'final_goal_distance': final_goal_distance,
            'final_goal_distance_reference': 'ground_truth',
            'goal_termination_reference': 'planner_belief',
            # `receipt_sim_clock` means the bridge gave no usable source stamp. The
            # interval fields describe delivered cadence, not transport latency.
            'gt_stamp_source': self._gt_stamp_source,
            'gt_source_time_available': self._gt_stamp_source == 'transform_header',
            'gt_transport_latency_measured': False,
            'gt_sample_interval_median_s': (
                float(np.median(self._gt_intervals)) if self._gt_intervals else math.nan),
            'gt_sample_interval_p95_s': (
                float(np.percentile(self._gt_intervals, 95))
                if self._gt_intervals else math.nan),
            'gt_sample_interval_max_s': (
                float(np.max(self._gt_intervals)) if self._gt_intervals else math.nan),
            'gt_samples': len(self._gt_buf),
            'final_goal_distance_odom_m': final_goal_distance_odom,
            'minimum_goal_distance': self._min_goal_distance if math.isfinite(self._min_goal_distance) else math.nan,
            'goal_success_radius': self.goal_success_radius,
            'goal_success_hold_s': self.goal_success_hold_s,
            'goal_stable_radius': self.goal_stable_radius,
            'goal_stable_hold_s': self.goal_stable_hold_s,
            'goal_stable_max_displacement_m': self.goal_stable_max_displacement_m,
            'goal_loiter_timeout_s': self.goal_loiter_timeout_s,
            'goal_region_entered': bool(self._goal_region_entered),
            'goal_region_first_stamp': self._goal_region_first_stamp,
            'goal_region_after_first_cmd_s': goal_region_after_first_cmd_s,
            'goal_region_success': goal_region_success,
            'stuck_window_s': self.stuck_window_s,
            'stuck_max_displacement_m': self.stuck_max_displacement_m,
            'stuck_max_goal_improvement_m': self.stuck_max_goal_improvement_m,
            'stuck_cmd_fraction_min': self.stuck_cmd_fraction_min,
            'stuck_idle_cmd_fraction_max': self.stuck_idle_cmd_fraction_max,
            'mean_solve_time_ms': mean_solve_time_ms,
            # EFE terms used by the paper objective.
            'mean_efe_risk': mean_efe_risk,
            'mean_efe_ambiguity': mean_efe_ambiguity,
            'mean_efe_control': mean_efe_control,
            'mean_efe_obstacle': mean_efe_obstacle,
            'mean_p_vis_plan': mean_p_vis_plan,
            'mean_p_vis_plan_eff': mean_p_vis_plan_eff,
            'fraction_time_p_vis_below_0_2': fraction_time_p_vis_below_0_2,
            'fraction_time_p_vis_eff_below_0_2': fraction_time_p_vis_eff_below_0_2,
            'mean_r_plan_u_std': mean_r_plan_u_std,
            'mean_r_plan_v_std': mean_r_plan_v_std,
            'max_r_plan_std': max_r_plan_std,
            # Explicit truth vs perception / planner belief errors
            'mean_state_error_odom_m': mean_state_error_odom_m,
            'mean_belief_error_odom_m': mean_belief_error_odom_m,
            'mean_state_error_odom_after_first_cmd_m': mean_state_error_odom_after_first_cmd_m,
            'mean_belief_error_odom_after_first_cmd_m': mean_belief_error_odom_after_first_cmd_m,
            # GROUND-TRUTH error means (honest; use these, not the mean_truth_* above)
            'mean_belief_error_gt_m': mean_belief_error_gt_m,
            'mean_state_error_gt_m': mean_state_error_gt_m,
            'mean_belief_error_gt_after_first_cmd_m': mean_belief_error_gt_after_first_cmd_m,
            'mean_state_error_gt_after_first_cmd_m': mean_state_error_gt_after_first_cmd_m,
            'mean_abs_odom_map_vs_odom_yaw_error_rad': mean_abs_odom_map_vs_odom_yaw_error_rad,
            'mean_abs_odom_map_vs_state_yaw_error_rad': mean_abs_odom_map_vs_state_yaw_error_rad,
            'mean_abs_odom_map_vs_belief_yaw_error_rad': mean_abs_odom_map_vs_belief_yaw_error_rad,
            'mean_abs_odom_map_vs_odom_yaw_error_after_first_cmd_rad': mean_abs_odom_map_vs_odom_yaw_error_after_first_cmd_rad,
            'mean_abs_odom_map_vs_state_yaw_error_after_first_cmd_rad': mean_abs_odom_map_vs_state_yaw_error_after_first_cmd_rad,
            'mean_abs_odom_map_vs_belief_yaw_error_after_first_cmd_rad': mean_abs_odom_map_vs_belief_yaw_error_after_first_cmd_rad,
            'valid_run': bool(self._valid_run),
            'invalid_reason': self._invalid_reason,
            'correction_assimilation_count': int(self._assimilation_count),
            'correction_assimilation_dropped_count': int(
                self._assimilation_dropped_count),
            # Refusals are reported, not treated as faults. The rate says how much of the
            # drive the cameras actually carried; the longest gap says how blind the worst
            # stretch of this route was. Both belong beside the accuracy, not in place of
            # a validity verdict.
            'correction_dropped_reasons': dict(self._assimilation_dropped_reasons),
            'correction_dropped_fraction': (
                float(self._assimilation_dropped_count) / float(self._assimilation_count)
                if self._assimilation_count else 0.0),
            'longest_correction_gap_s': float(self._longest_correction_gap_s),
            'correction_ledger': ledger.to_dict(),
            'correction_ledger_applicable': ledger_applicable,
            'correction_ledger_required': ledger_required,
            'schema2_terminal_count': schema2_terminal_count,
            'belief_prediction_count': self._belief_prediction_count,
            'belief_prediction_invalid_count': self._belief_prediction_invalid_count,
            'committed_posterior_complete': committed_posterior_complete,
            'runtime_event_delivery_count': int(getattr(self.runtime_event_log, 'rows', 0)),
            'runtime_event_counts_by_topic': dict(self._runtime_event_counts),
            'runtime_event_invalid_count': int(self._runtime_event_invalid_count),
            'event_drain_quiet_s': EVENT_DRAIN_QUIET_S,
            'event_drain_max_s': EVENT_DRAIN_MAX_S,
            'event_drain_elapsed_s': event_drain_elapsed_s,
            'event_cutoff_wall_s': event_cutoff_wall_s,
            'late_events_observed_before_summary_snapshot': int(self._late_event_count),
            'producer_quiescence_acknowledged': bool(
                TERMINAL_COMPONENTS.issubset(set(self._terminal_stop_acks))
            ),
            'terminal_stop_request_id': self._terminal_stop_request_id,
            'terminal_stop_request': (
                None if self._terminal_stop_request_payload is None
                else {
                    'request_id': self._terminal_stop_request_payload.request_id,
                    'run_id': self._terminal_stop_request_payload.run_id,
                    'reason': self._terminal_stop_request_payload.reason,
                    'decision_stamp_ns': (
                        self._terminal_stop_request_payload.decision_stamp_ns
                    ),
                }
            ),
            'terminal_stop_acknowledgements': {
                component: {
                    'status': ack.status,
                    'acknowledgement_stamp_ns': ack.acknowledgement_stamp_ns,
                    'detail': ack.detail,
                }
                for component, ack in sorted(self._terminal_stop_acks.items())
            },
            'terminal_stop_verified': bool(self._terminal_stop_verified),
            'terminal_zero_forwarded': bool(self._terminal_zero_forwarded),
            'terminal_stop_event_id': self._terminal_stop_event_id,
            'terminal_zero_stamp_s': self._terminal_zero_stamp_s,
            'terminal_rest_window_s': TERMINAL_REST_WINDOW_S,
            'terminal_rest_max_displacement_m': TERMINAL_REST_MAX_DISPLACEMENT_M,
            'mission_goal_id': (
                self._mission_goal_state.goal_id if self._mission_goal_state is not None else ''),
            'mission_epoch': (
                self._mission_goal_state.mission_epoch
                if self._mission_goal_state is not None else ''),
            'mission_goal_is_final': (
                self._mission_goal_state.is_final
                if self._mission_goal_state is not None else None),
            'mission_goal_status': (
                self._mission_goal_state.status
                if self._mission_goal_state is not None else ''),
            'mission_goal_reason': (
                self._mission_goal_state.reason
                if self._mission_goal_state is not None else ''),
            'mission_goal_status_stamp_ns': (
                self._mission_goal_state.status_stamp_ns
                if self._mission_goal_state is not None else None),
            'frame_sanity': dict(self._frame_sanity),
            'run_dir': self.run_dir
        }

        close_errors = self._flush_close_data_files()
        if close_errors:
            self._record_invalid('data_file_close_failure')
            summary['valid_run'] = False
            summary['invalid_reason'] = self._invalid_reason
        summary['data_files_closed'] = not close_errors
        summary['data_file_close_errors'] = list(close_errors)
        summary['evidence_complete'] = bool(
            not close_errors
            and (not ledger_applicable or ledger.valid)
            and (not ledger_required or committed_posterior_complete)
            and summary['producer_quiescence_acknowledged']
            and self._valid_run)
        summary_path = os.path.join(self.run_dir, 'run_summary.json')
        try:
            _write_json_atomic(summary_path, summary)
        except Exception as exc:
            self._finalizing = False
            self.get_logger().error(
                f'Failed to commit run summary atomically: {type(exc).__name__}: {exc}')
            threading.Timer(0.15, _safe_shutdown).start()
            return
        self._completed = True
        self._finalizing = False

        self.get_logger().info(f"Ending run. Reason: {reason}.")
        # rclpy.shutdown() must NOT be called from inside a timer/subscription callback
        # (it deadlocks on the global executor lock). Schedule it on a background thread
        # so this callback can return cleanly first.
        threading.Timer(0.15, _safe_shutdown).start()

    def destroy_node(self):
        try:
            if not getattr(self, '_completed', False):
                self._accepting_events = False
                close_errors = self._flush_close_data_files()
                summary_path = os.path.join(self.run_dir, 'run_summary.json')
                summary = {
                    'completed': False,
                    'completion_reason': 'interrupted',
                    'valid_run': bool(getattr(self, '_valid_run', True)),
                    'invalid_reason': str(getattr(self, '_invalid_reason', '') or ''),
                    'data_files_closed': not close_errors,
                    'data_file_close_errors': list(close_errors),
                    'terminal_stop_verified': bool(
                        getattr(self, '_terminal_stop_verified', False)),
                    'terminal_zero_forwarded': bool(
                        getattr(self, '_terminal_zero_forwarded', False)),
                    'frame_sanity': dict(getattr(self, '_frame_sanity', {})),
                    'run_dir': getattr(self, 'run_dir', '')
                }
                _write_json_atomic(summary_path, summary)
            else:
                self._flush_close_data_files()
        finally:
            super().destroy_node()


def _safe_shutdown():
    """Deferred shutdown helper — called from a background thread, not from a ROS callback.

    rclpy.shutdown() deadlocks if called from inside a timer or subscription callback
    because it tries to acquire the executor lock, which the calling callback already holds.
    Scheduling it on a Thread with a small delay allows the callback to return first.
    """
    try:
        rclpy.shutdown()
    except Exception:
        pass


def main(args=None):
    rclpy.init(args=args)
    node = ExperimentLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
