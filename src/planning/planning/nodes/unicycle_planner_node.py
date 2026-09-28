"""Thin ROS 2 wrapper around unicycle planners."""

import json
import math
import time
import threading
import traceback
from functools import wraps
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from rclpy.time import Time
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float64MultiArray, Header, String
from builtin_interfaces.msg import Time as TimeMsg

from perception.core.detection_diagnostics import (
    DETECTION_DIAGNOSTICS_TOPIC,
    diagnostics_from_message,
)
from reliability.fusion import map_observations_from_json

from planning.core.efe_utils import wrap_angle
from planning.core import belief_correction as bc
from planning.core.motion_history import MotionHistorySnapshot, covers_interval
from planning.core.motion_history import plan_replay
from planning.core.belief_state import BeliefRecord, MotionSupport, PredictionSnapshot, checked_state
from planning.core.dynamics import unicycle_jacobian
from copy import deepcopy
from types import MappingProxyType
import uuid
from unav_common.config import local_controller_type, parse_bev_affine_calibration
from unav_common.odometry_input import odometry_pose_yaw, odometry_pose_is_available

PIXEL_DIAG_K_THETA_U_IDX = 42
PIXEL_DIAG_K_THETA_V_IDX = 43

# A metric xy correction says nothing about heading; keep its yaw variance
# non-informative so no consumer mistakes it for a heading fix. Matches
# camera_manager_node.NONINFORMATIVE_YAW_VAR.
NONINFORMATIVE_YAW_VAR = float(math.pi ** 2)
_UNSPECIFIED_RECORD = object()


def _correction_envelope_allows_reanchor(payload) -> bool:
    """Whether one retained fused event can justify resetting a lost belief."""
    if payload.get('schema_version') == 1:
        return True
    accepted = payload.get('accepted_camera_ids', ())
    return bool(
        isinstance(accepted, list)
        and len(accepted) == len(set(accepted))
        and len(accepted) >= 2
    )


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 't', 'yes', 'y', 'on')
    return bool(value)


def _serialized_correction(method):
    """Give the recursive filter one writer while odometry and commands keep flowing."""
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._correction_lock:
            return method(self, *args, **kwargs)
    return call


class UnicyclePlannerNode(Node):
    """Base class for EFE/MPC planners using unicycle dynamics."""

    NODE_NAME = 'planner'
    PLANNER_CLASS = None
    PARAM_DEFAULT_OVERRIDES = {}

    def __init__(self):
        super().__init__(self.NODE_NAME)

        if self.PLANNER_CLASS is None:
            raise RuntimeError('PLANNER_CLASS is not set.')

        node_defaults = dict(getattr(self, 'PARAM_DEFAULT_OVERRIDES', {}) or {})

        def _declare_if_not(name, default_value):
            if name in node_defaults:
                default_value = node_defaults[name]
            if not self.has_parameter(name):
                self.declare_parameter(name, default_value)

        def _as_bool(value):
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in ('1', 'true', 't', 'yes', 'y', 'on')
            return bool(value)

        # Standalone defaults for the visibility-aware thesis planner node.

        # Planner params
        _declare_if_not('plan_rate', 1.0)
        _declare_if_not('belief_publish_rate', 10.0)
        _declare_if_not('horizon', 10)
        _declare_if_not('dt', 0.2)
        _declare_if_not('v_min', 0.0)
        _declare_if_not('v_max', 0.22)
        _declare_if_not('w_min', -1.0)
        _declare_if_not('w_max', 1.0)
        _declare_if_not('control_weight', 0.0)
        _declare_if_not('seed', 0)

        # Process/observation noise
        # Locked to the camera-ready IWAI actuation-noise PSDs; see
        # docs/PROCESS_NOISE.md before changing either value.
        _declare_if_not('process_noise_xy', 0.02)
        _declare_if_not('process_noise_theta', 0.08)
        # 'encoder' (default): Q set from the simulated encoder noise; 'constant_psd': legacy.
        _declare_if_not('process_noise_model', 'encoder')
        # Frozen process-model option. Campaign manifests must state its value.
        _declare_if_not('coherent_drift', False)

        # Goal observation covariance
        _declare_if_not('goal_sigma_uv', 2.0)

        # EFE weights
        _declare_if_not('risk_weight_obs', 1.0)
        _declare_if_not('ambiguity_weight', 1.0)
        _declare_if_not('approx_method', 'ET1')
        _declare_if_not('use_obs_risk', True)
        _declare_if_not('use_ambiguity', True)
        _declare_if_not('use_visibility_model', False)
        _declare_if_not('visibility_target_height_m', 0.0)
        _declare_if_not('visibility_geometry_json', '')
        _declare_if_not('collision_geometry_json', '')
        _declare_if_not('r_visible_uv', 2.5)
        _declare_if_not('r_miss_uv', 120.0)
        _declare_if_not('visibility_sigma_kappa', 1.0)
        _declare_if_not('goal_prior_u_std_start', 80.0)
        _declare_if_not('goal_prior_v_std_start', 80.0)
        _declare_if_not('goal_prior_u_std_final', 18.0)
        _declare_if_not('goal_prior_v_std_final', 18.0)
        _declare_if_not('goal_tightening_power', 0.9)
        _declare_if_not('goal_progress_n_steps', 90)
        _declare_if_not('observation_risk_scale', 1.0)
        _declare_if_not('ambiguity_term_scale', 1.00)
        _declare_if_not('discount_gamma', 0.995)
        _declare_if_not('use_nogo_cost', False)
        _declare_if_not('nogo_penalty_type', 'warning_band')
        _declare_if_not('nogo_weight', 0.0)
        # Legacy centre-point radius. Exact rectangular-footprint runs use the
        # body dimensions and explicit map inflation, so no implicit radius is
        # added here.
        _declare_if_not('nogo_safe_distance', 0.0)
        _declare_if_not('nogo_logbarrier_eps', 1e-3)
        _declare_if_not('nogo_warning_band', 0.05)
        _declare_if_not('nogo_near_weight', 50.0)
        _declare_if_not('use_belief_nogo_cost', False)
        _declare_if_not('nogo_belief_kappa', 1.0)
        # Hit/miss expected-belief mixture in the EFE objective. MUST stay False by
        # default: false reproduces the published single-camera precision-blend
        # planner bit-for-bit. True switches the objective to the Bernoulli
        # availability model (and stops reading r_miss_uv entirely).
        _declare_if_not('use_hit_miss_mixture', False)
        _declare_if_not('nogo_mode', 'keep_out')
        _declare_if_not('driveable_geometry_json', '')
        _declare_if_not('visibility_artifact_path', '')
        _declare_if_not('camera_network_artifact_path', '')
        _declare_if_not('camera_network_expected_sha256', '')
        _declare_if_not('camera_network_expected_source_hashes_json', '')
        _declare_if_not('camera_network_camera_ids', '')
        _declare_if_not('camera_network_active_camera_ids', '')
        _declare_if_not('camera_network_objective', 'legacy_pixel_chart')
        _declare_if_not('network_goal_std_m', 0.10)
        _declare_if_not('kouw_et1_ambiguity', True)
        _declare_if_not('network_goal_std_start_m', 5.0)
        _declare_if_not('network_goal_std_start_m', -1.0)
        _declare_if_not('camera_network_updates_per_step', 1)
        # The planner models the robot as a disc, so this is the CIRCUMSCRIBED
        # radius. warehouse_amr is 0.800 x 0.550 m -> hypot(0.400, 0.275) = 0.485.
        # (turtlebot3_burger was 0.125; pass it explicitly to reproduce a
        # pre-2026-08-20 campaign.)
        _declare_if_not('robot_collision_radius_m', 0.485)
        _declare_if_not('robot_length_m', 0.8)
        _declare_if_not('robot_width_m', 0.55)

        # Optimizer params
        _declare_if_not('optimizer_maxiter', 50)
        _declare_if_not('optimizer_maxfun', 500)
        _declare_if_not('optimizer_ftol', 1e-6)
        _declare_if_not('optimizer_gtol', 1e-4)
        _declare_if_not('optimizer_control_block_steps', 1)
        _declare_if_not('optimizer_warm_start', True)
        _declare_if_not('optimizer_multistart', False)
        _declare_if_not('optimizer_multistart_include_direct', True)
        _declare_if_not('optimizer_initial_routes_json', '')
        _declare_if_not('optimizer_terminal_goal_tolerance_m', 0.0)
        # Route-seed source for the multistart: 'explicit' uses
        # optimizer_initial_routes_json as-is; 'lane_graph' generates condition-
        # neutral lane-centre Manhattan seeds from the driveable map at the (one-shot)
        # global solve. See unav_common.lane_graph_routes.
        _declare_if_not('optimizer_route_seed_mode', 'explicit')
        # Two-stage (global-then-local) hierarchical planning
        _declare_if_not('use_hierarchical', False)
        _declare_if_not('global_horizon', 60)
        # Global-solve step size. 0.0 => use dt; a larger value coarsens the
        # one-shot global route solve (cost is linear in the number of steps),
        # trading discretization fidelity for a faster solve. The global plan is
        # reduced to spatial waypoints, so coarser steps mostly affect route
        # shape resolution, not the tracked path.
        _declare_if_not('global_dt', 0.0)
        _declare_if_not('local_horizon', 12)
        _declare_if_not('local_plan_rate', 4.0)
        _declare_if_not('local_optimizer_maxiter', 60)
        _declare_if_not('global_use_ambiguity', True)
        _declare_if_not('local_use_ambiguity', False)
        # Local executor observation-space goal risk. Default True keeps the locked
        # config behaviour; the belief-loop config sets this False so the local
        # layer is a STATE-space waypoint tracker (the observation-space pixel goal
        # prior is what produced the aisle-transition freeze and the ~0.42 m
        # final-approach stall). Condition-neutral; visibility stays in the global EFE.
        _declare_if_not('local_use_obs_risk', True)
        _declare_if_not('global_optimizer_multistart', True)
        _declare_if_not('local_optimizer_multistart', True)
        _declare_if_not('local_use_visibility_model', False)
        _declare_if_not('local_use_belief_nogo_cost', False)
        _declare_if_not('local_nogo_penalty_type', '')
        _declare_if_not('local_nogo_weight', -1.0)
        _declare_if_not('local_nogo_safe_distance', -1.0)
        _declare_if_not('local_goal_prior_u_std_start', -1.0)
        _declare_if_not('local_goal_prior_v_std_start', -1.0)
        _declare_if_not('local_goal_prior_u_std_final', -1.0)
        _declare_if_not('local_goal_prior_v_std_final', -1.0)
        _declare_if_not('waypoint_spacing_m', 0.2)
        _declare_if_not('waypoint_arrival_radius_m', 0.1)
        _declare_if_not('local_replan_min_remaining_s', 0.0)
        _declare_if_not('local_replan_on_waypoint_change', False)
        _declare_if_not('latency_compensate_plan_handoff', False)
        _declare_if_not('simple_tracker_yaw_gate_rad', 0.65)
        _declare_if_not('ff_fb_turn_rate_limit_rad_s', 0.80)
        _declare_if_not('ff_fb_corner_crawl_speed_mps', 0.18)
        _declare_if_not('ff_fb_pivot_heading_error_rad', 2.60)
        # Which waypoint-tracking law the simple local controller uses. All are
        # "execution plumbing" (track the global plan, no local EFE); they differ
        # only in HOW they track, to avoid the turn-then-go limit-cycle on sharp
        # turns. 'turn_then_go' = legacy. 'hyst_damp' = +hysteresis/damped-w/creep.
        # 'pure_pursuit' = lookahead. 'ff_fb' = path tangent/curvature feedforward
        # + cross-track feedback on belief.
        _declare_if_not('local_controller_type', 'ff_fb')

        # Pixel correction params
        _declare_if_not('use_pixel_correction', False)
        # Multicam /state/bev belief filter. When False the legacy path hard-resets
        # the belief xy to each raw fused correction (no smoothing, no latency
        # compensation). When True the /state/bev correction is fused through the
        # same predict->EKF-update->predict cycle the single-camera pixel path
        # uses: motion-replay latency compensation to the correction stamp, a
        # covariance-weighted Kalman update, NIS + jump gating.
        _declare_if_not('state_correction_ekf', False)
        # How the multicam measurements reach the filter.
        #   'fused'      -- one pre-fused /state/bev pose per tick (camera_manager
        #                   fuses in map space, then the planner does one update).
        #   'per_camera' -- the per-camera map observations, folded in
        #                   SEQUENTIALLY, each with its own covariance and its own
        #                   gate decision. One filter instead of two in series.
        # Both run the same gate chain, so this is a clean A/B.
        _declare_if_not('state_correction_mode', 'fused')
        _declare_if_not(
            'map_observations_topic', '/reliability/camera_manager/map_observations'
        )
        # Evidence-grade fused corrections use an envelope carrying the detector
        # batch identity beside the pose/covariance. The Pose topic remains for
        # visualisation and legacy consumers, but a paper run never infers
        # assimilation identity from a timestamp.
        _declare_if_not(
            'state_correction_envelope_topic',
            '/reliability/camera_manager/fused_correction',
        )
        _declare_if_not('require_state_correction_envelope', False)
        _declare_if_not('pixel_topic', '/perception/pixel_pose')
        _declare_if_not('cmd_topic', '/cmd_vel')
        _declare_if_not('cmd_publish_rate', 10.0)
        _declare_if_not('pixel_timeout_s', 0.5)
        _declare_if_not('pixel_correction_min_interval_s', 0.0)
        _declare_if_not('bev_y_calibration_offset_m', 0.0)
        _declare_if_not('bev_affine_calibration', '')
        _declare_if_not('pixel_max_correction_jump_m', 0.0)
        # DIAGNOSTIC ONLY: feed transformed raw ODOMETRY -- not ground truth -- as the
        # planner belief, bypassing perception entirely, to isolate the controller from
        # the estimator. MUST be false for any comparison or paper run; the campaign
        # runner and the leakage firewall both refuse a run that sets it true.
        _declare_if_not('use_diagnostic_odom_localization', False)
        _declare_if_not('diagnostic_odom_topic', '/odom')
        # Chi-squared (2-DOF) innovation gate: a pixel correction whose NIS exceeds
        # this threshold is rejected as a detector outlier. 0.0 = disabled; the
        # campaign sets 9.21 = chi2(2, 0.99).
        _declare_if_not('pixel_correction_nis_threshold', 9.21)
        _declare_if_not('pixel_correction_approx', 'AUTO')
        _declare_if_not('skip_stale_pixel_correction', True)
        _declare_if_not('odom_topic', '/odom_noisy')
        _declare_if_not('odom_frame_id', 'odom')
        _declare_if_not('odom_child_frame_id', 'base_footprint')
        _declare_if_not('use_odom_for_predict', True)
        # The long-standing behaviour stays the default. `coupled` is the better
        # estimator -- it keeps the posterior the update produced instead of deleting
        # the position-heading cross terms -- but it is still under test, so every
        # campaign declares its choice rather than inheriting one.
        _declare_if_not('heading_update_mode', 'camera_xy_only')
        # Spawn yaw (map_bev - odom). The single-camera path applies this in
        # pixel_to_bev_state_node (heading = odom_yaw + offset); the multicam
        # path replaces that node, so the planner must apply it itself or the
        # belief heading stays in the raw-odom frame and the robot plans/drives
        # ~90 deg off. Default 0 keeps the single-camera path unchanged.
        _declare_if_not('odom_yaw_offset_rad', 0.0)
        # Declared initial prior (METHOD amendment 2026-09-24 night): the belief starts at the
        # task's declared start pose, not at a camera bootstrap. Off keeps the camera bootstrap.
        _declare_if_not('initial_belief_from_task_start', False)
        _declare_if_not('initial_belief_xyyaw', [0.0, 0.0, 0.0])
        _declare_if_not('initial_belief_sigma_xy_m', 0.10)
        _declare_if_not('initial_belief_sigma_theta_rad', math.radians(15.0))
        # Divergence guards for the multicam /state/bev EKF. The fused correction
        # is camera-derived + manager-gated (reliable ~0.2 m), so if the predicted
        # belief lands implausibly far from a fresh correction the belief (or a
        # bad-stamp replay) has diverged -> hard re-anchor to the correction rather
        # than NIS-reject it (which locks the belief out of recovery). And cap the
        # unsupported motion-sample gap so a single far-future correction stamp
        # cannot jump the prediction tens of metres. A longer interval between
        # camera fixes is still replayable when timestamped odometry covers the
        # whole interval with no gap above this bound.
        _declare_if_not('state_reanchor_m', 0.0)
        _declare_if_not('state_max_predict_dt_s', 1.5)
        # Optional legacy covariance added on a rejected /state/bev correction.
        # The default is zero: the rejected observation contributes no information,
        # while time and motion propagation still add the configured process noise.
        # A fixed event-triggered jump makes transport failures look like physical
        # drift and progressively weakens the NIS gate.
        _declare_if_not('state_reject_inflate_m2', 0.0)
        # Optional extra covariance growth after the ordinary process model.
        # Zero is the evidence-grade default: an empirical staleness penalty must
        # never hide in control flow or be absent from the run manifest.
        _declare_if_not('stale_belief_inflate_m2_per_s', 0.0)
        _declare_if_not('stale_belief_inflate_cap_m2', 0.0)
        # Jump limiter for the METRIC (/state/bev, per-camera) path. Default OFF,
        # and deliberately separate from pixel_max_correction_jump_m.
        #
        # NIS and a jump limit fire in OPPOSITE regimes. Confident belief + large
        # innovation -> NIS is huge (rejects, correctly) while the gain is small
        # so the update is small and a jump limit would not fire. Uncertain
        # belief + large innovation -> NIS is moderate (passes) while the gain is
        # ~1 so the update is large and a jump limit DOES fire -- precisely when a
        # large correction is warranted. On this path H = [I2 | 0] is linear, so
        # NIS is the statistically correct gate and a jump limit is redundant at
        # best.
        #
        # Worse, it deadlocks: rejecting inflates S, which RAISES the gain, which
        # RAISES the update, so the limit keeps firing. A belief genuinely 1.5 m
        # off never recovers until drift pushes the innovation past
        # state_reanchor_m -- it has to get worse first. With NIS alone the same
        # case recovers monotonically in ~6 corrections (~1.2 s at 5 Hz).
        #
        # The pixel path keeps its 0.5 m limit: nonlinear observation model, and
        # it is locked paper-1 method backing honest_campaign_v1.
        _declare_if_not('state_max_correction_jump_m', 0.0)
        # Kinematic plausibility cap on the prediction, m/s. The motion replay
        # extrapolates the last command when odometry is missing (up to
        # state_max_predict_dt_s), inventing up to ~0.9 m of travel. 0 disables.
        # Left at 0 for the single-camera path: the locked campaign ran without
        # it and its evidence must stay reproducible.
        _declare_if_not('max_predict_speed_mps', 0.0)
        _declare_if_not('min_state_cov', 1e-6)
        _declare_if_not('debug_runtime', False)
        _declare_if_not('debug_log_period_s', 1.0)
        _declare_if_not('slow_plan_factor', 1.0)
        _declare_if_not('slow_correction_ms', 20.0)

        # Camera model params (must match sim)
        _declare_if_not('cam_pos', [-3.0, -3.0, 6.0])
        _declare_if_not('look_at', [1.5, 1.5, 0.0])
        _declare_if_not('img_width', 1280)
        _declare_if_not('img_height', 720)
        _declare_if_not('fov_h_rad', 1.5708)

        self.plan_rate = float(self.get_parameter('plan_rate').value)
        self.belief_publish_rate = float(self.get_parameter('belief_publish_rate').value)
        self.horizon = int(self.get_parameter('horizon').value)
        self.dt = float(self.get_parameter('dt').value)
        self.v_min = float(self.get_parameter('v_min').value)
        self.v_max = float(self.get_parameter('v_max').value)
        self.w_min = float(self.get_parameter('w_min').value)
        self.w_max = float(self.get_parameter('w_max').value)
        self.control_weight = float(self.get_parameter('control_weight').value)
        self.seed = int(self.get_parameter('seed').value)

        self.coherent_drift = bool(self.get_parameter('coherent_drift').value)
        self.process_noise_xy = float(self.get_parameter('process_noise_xy').value)
        self.process_noise_theta = float(self.get_parameter('process_noise_theta').value)
        self.process_noise_model = str(self.get_parameter('process_noise_model').value)

        self.goal_sigma_uv = float(self.get_parameter('goal_sigma_uv').value)

        self.risk_weight_obs = float(self.get_parameter('risk_weight_obs').value)
        self.ambiguity_weight = float(self.get_parameter('ambiguity_weight').value)
        self.approx_method = str(self.get_parameter('approx_method').value).upper()
        if self.approx_method not in ('ET1', 'ET2'):
            raise RuntimeError("approx_method must be one of: ET1, ET2")
        self.planner_path_summary = (
            f'approx_method={self.approx_method}, solver=casadi_symbolic_efe'
        )
        self.use_obs_risk = _as_bool(self.get_parameter('use_obs_risk').value)
        self.use_ambiguity = _as_bool(self.get_parameter('use_ambiguity').value)
        self.use_visibility_model = _as_bool(self.get_parameter('use_visibility_model').value)
        self.visibility_target_height_m = float(self.get_parameter('visibility_target_height_m').value)
        self.visibility_geometry_json = str(self.get_parameter('visibility_geometry_json').value)
        self.collision_geometry_json = str(self.get_parameter('collision_geometry_json').value)
        self.r_visible_uv = float(self.get_parameter('r_visible_uv').value)
        self.r_miss_uv = float(self.get_parameter('r_miss_uv').value)
        self.visibility_sigma_kappa = float(self.get_parameter('visibility_sigma_kappa').value)
        self.goal_prior_u_std_start = float(self.get_parameter('goal_prior_u_std_start').value)
        self.goal_prior_v_std_start = float(self.get_parameter('goal_prior_v_std_start').value)
        self.goal_prior_u_std_final = float(self.get_parameter('goal_prior_u_std_final').value)
        self.goal_prior_v_std_final = float(self.get_parameter('goal_prior_v_std_final').value)
        self.goal_tightening_power = float(self.get_parameter('goal_tightening_power').value)
        self.goal_progress_n_steps = int(self.get_parameter('goal_progress_n_steps').value)
        self.observation_risk_scale = float(self.get_parameter('observation_risk_scale').value)
        self.ambiguity_term_scale = float(self.get_parameter('ambiguity_term_scale').value)
        self.discount_gamma = float(self.get_parameter('discount_gamma').value)
        self.use_nogo_cost = _as_bool(self.get_parameter('use_nogo_cost').value)
        self.nogo_penalty_type = str(self.get_parameter('nogo_penalty_type').value).strip().lower()
        self.nogo_weight = float(self.get_parameter('nogo_weight').value)
        self.nogo_safe_distance = float(self.get_parameter('nogo_safe_distance').value)
        self.nogo_logbarrier_eps = float(self.get_parameter('nogo_logbarrier_eps').value)
        self.nogo_warning_band = float(self.get_parameter('nogo_warning_band').value)
        self.nogo_near_weight = float(self.get_parameter('nogo_near_weight').value)
        self.use_belief_nogo_cost = _as_bool(self.get_parameter('use_belief_nogo_cost').value)
        self.nogo_belief_kappa = float(self.get_parameter('nogo_belief_kappa').value)
        self.use_hit_miss_mixture = _as_bool(self.get_parameter('use_hit_miss_mixture').value)
        self.nogo_mode = str(self.get_parameter('nogo_mode').value or 'keep_out').strip().lower()
        self.driveable_geometry_json = str(self.get_parameter('driveable_geometry_json').value or '')
        self.visibility_artifact_path = str(self.get_parameter('visibility_artifact_path').value).strip()
        self.camera_network_artifact_path = str(self.get_parameter('camera_network_artifact_path').value).strip()
        self.camera_network_expected_sha256 = str(
            self.get_parameter('camera_network_expected_sha256').value
        ).strip()
        self.camera_network_expected_source_hashes_json = str(
            self.get_parameter('camera_network_expected_source_hashes_json').value
        ).strip()
        self.camera_network_camera_ids = str(
            self.get_parameter('camera_network_camera_ids').value
        ).strip()
        self.camera_network_active_camera_ids = str(
            self.get_parameter('camera_network_active_camera_ids').value
        ).strip()
        self.camera_network_objective = str(
            self.get_parameter('camera_network_objective').value
        ).strip().lower()
        self.network_goal_std_m = float(self.get_parameter('network_goal_std_m').value)
        self.kouw_et1_ambiguity = bool(self.get_parameter('kouw_et1_ambiguity').value)
        _goal_start = float(self.get_parameter('network_goal_std_start_m').value)
        self.network_goal_std_start_m = _goal_start if _goal_start > 0. else None
        self.camera_network_updates_per_step = int(
            self.get_parameter('camera_network_updates_per_step').value
        )
        self.robot_collision_radius_m = float(self.get_parameter('robot_collision_radius_m').value)
        self.robot_length_m = float(self.get_parameter('robot_length_m').value)
        self.robot_width_m = float(self.get_parameter('robot_width_m').value)

        self.optimizer_maxiter = int(self.get_parameter('optimizer_maxiter').value)
        self.optimizer_maxfun = int(self.get_parameter('optimizer_maxfun').value)
        self.optimizer_ftol = float(self.get_parameter('optimizer_ftol').value)
        self.optimizer_gtol = float(self.get_parameter('optimizer_gtol').value)
        self.optimizer_control_block_steps = int(
            self.get_parameter('optimizer_control_block_steps').value
        )
        self.optimizer_warm_start = _as_bool(self.get_parameter('optimizer_warm_start').value)
        self.optimizer_multistart = _as_bool(self.get_parameter('optimizer_multistart').value)
        self.optimizer_multistart_include_direct = _as_bool(
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
        self.use_hierarchical = _as_bool(self.get_parameter('use_hierarchical').value)
        self.global_horizon = int(self.get_parameter('global_horizon').value)
        _global_dt = float(self.get_parameter('global_dt').value)
        self.global_dt = _global_dt if _global_dt > 0.0 else float(self.get_parameter('dt').value)
        self.local_horizon = int(self.get_parameter('local_horizon').value)
        self.local_plan_rate = float(self.get_parameter('local_plan_rate').value)
        self.local_optimizer_maxiter = int(self.get_parameter('local_optimizer_maxiter').value)
        self.global_use_ambiguity = _as_bool(self.get_parameter('global_use_ambiguity').value)
        self.local_use_ambiguity = _as_bool(self.get_parameter('local_use_ambiguity').value)
        self.local_use_obs_risk = _as_bool(self.get_parameter('local_use_obs_risk').value)
        self.global_optimizer_multistart = _as_bool(
            self.get_parameter('global_optimizer_multistart').value
        )
        self.local_optimizer_multistart = _as_bool(
            self.get_parameter('local_optimizer_multistart').value
        )
        self.local_use_visibility_model = _as_bool(
            self.get_parameter('local_use_visibility_model').value
        )
        self.local_use_belief_nogo_cost = _as_bool(
            self.get_parameter('local_use_belief_nogo_cost').value
        )
        self.local_nogo_penalty_type = str(
            self.get_parameter('local_nogo_penalty_type').value or ''
        ).strip().lower()
        self.local_nogo_weight = float(self.get_parameter('local_nogo_weight').value)
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
        self.waypoint_arrival_radius_m = float(self.get_parameter('waypoint_arrival_radius_m').value)
        self.local_replan_min_remaining_s = max(
            0.0, float(self.get_parameter('local_replan_min_remaining_s').value)
        )
        self.local_replan_on_waypoint_change = _as_bool(
            self.get_parameter('local_replan_on_waypoint_change').value
        )
        self.latency_compensate_plan_handoff = _as_bool(
            self.get_parameter('latency_compensate_plan_handoff').value
        )
        self.simple_tracker_yaw_gate_rad = max(
            0.0, float(self.get_parameter('simple_tracker_yaw_gate_rad').value)
        )
        self.ff_fb_turn_rate_limit_rad_s = float(np.clip(
            self.get_parameter('ff_fb_turn_rate_limit_rad_s').value,
            0.01,
            max(abs(float(self.w_min)), abs(float(self.w_max))),
        ))
        self.ff_fb_corner_crawl_speed_mps = float(np.clip(
            self.get_parameter('ff_fb_corner_crawl_speed_mps').value,
            0.0,
            float(self.v_max),
        ))
        self.ff_fb_pivot_heading_error_rad = float(np.clip(
            self.get_parameter('ff_fb_pivot_heading_error_rad').value,
            self.simple_tracker_yaw_gate_rad,
            math.pi,
        ))
        try:
            self.local_controller_type = local_controller_type(
                self.get_parameter('local_controller_type').value
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

        self.use_pixel_correction = _as_bool(self.get_parameter('use_pixel_correction').value)
        self.state_correction_ekf = _as_bool(self.get_parameter('state_correction_ekf').value)
        if self.camera_network_artifact_path and self.use_pixel_correction:
            raise RuntimeError('camera network requires metric camera corrections; its IWAI cost proxy is not measurement noise')
        self.state_correction_mode = str(
            self.get_parameter('state_correction_mode').value
        ).strip().lower()
        if self.state_correction_mode not in ('fused', 'per_camera'):
            raise RuntimeError(
                "state_correction_mode must be 'fused' or 'per_camera', "
                f"got {self.state_correction_mode!r}"
            )
        self.map_observations_topic = str(self.get_parameter('map_observations_topic').value)
        self.state_correction_envelope_topic = str(
            self.get_parameter('state_correction_envelope_topic').value
        ).strip()
        self.require_state_correction_envelope = _as_bool(
            self.get_parameter('require_state_correction_envelope').value
        )
        if self.require_state_correction_envelope and not self.state_correction_envelope_topic:
            raise RuntimeError(
                "require_state_correction_envelope=true needs a non-empty envelope topic"
            )
        self.pixel_topic = self.get_parameter('pixel_topic').value
        self.cmd_topic = str(self.get_parameter('cmd_topic').value).strip() or '/cmd_vel'
        self.cmd_publish_rate = max(0.1, float(self.get_parameter('cmd_publish_rate').value))
        self.pixel_timeout_s = float(self.get_parameter('pixel_timeout_s').value)
        self.pixel_correction_min_interval_s = float(
            self.get_parameter('pixel_correction_min_interval_s').value
        )
        self.bev_y_calibration_offset_m = float(
            self.get_parameter('bev_y_calibration_offset_m').value
        )
        # Position-dependent affine BEV calibration (6 coeffs). When set it is the
        # single calibration used by the pixel correction, matching the state node
        # and experiment logger; the constant y-offset is only a legacy fallback.
        _affine_raw = str(self.get_parameter('bev_affine_calibration').value or '').strip()
        try:
            self._bev_affine = parse_bev_affine_calibration(_affine_raw)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        self.pixel_max_correction_jump_m = float(
            self.get_parameter('pixel_max_correction_jump_m').value
        )
        self.use_diagnostic_odom_localization = _as_bool(
            self.get_parameter('use_diagnostic_odom_localization').value
        )
        self.diagnostic_odom_topic = (
            str(self.get_parameter('diagnostic_odom_topic').value).strip() or '/odom'
        )
        self.pixel_correction_nis_threshold = float(
            self.get_parameter('pixel_correction_nis_threshold').value
        )
        self.pixel_correction_approx = str(
            self.get_parameter('pixel_correction_approx').value
        ).strip().upper()
        if self.pixel_correction_approx not in ('AUTO', 'ET1', 'ET2', 'UT'):
            raise RuntimeError("pixel_correction_approx must be one of: AUTO, ET1, ET2, UT")
        self.skip_stale_pixel_correction = _as_bool(
            self.get_parameter('skip_stale_pixel_correction').value
        )
        self.odom_topic = str(self.get_parameter('odom_topic').value)
        self.odom_frame_id = str(self.get_parameter('odom_frame_id').value)
        self.odom_child_frame_id = str(self.get_parameter('odom_child_frame_id').value)
        self.use_odom_for_predict = _as_bool(self.get_parameter('use_odom_for_predict').value)
        self.odom_yaw_offset_rad = float(self.get_parameter('odom_yaw_offset_rad').value)
        self.initial_belief_from_task_start = _as_bool(
            self.get_parameter('initial_belief_from_task_start').value)
        self.initial_belief_xyyaw = [float(v) for v in self.get_parameter('initial_belief_xyyaw').value]
        self.initial_belief_sigma_xy_m = float(self.get_parameter('initial_belief_sigma_xy_m').value)
        self.initial_belief_sigma_theta_rad = float(
            self.get_parameter('initial_belief_sigma_theta_rad').value)
        if self.initial_belief_from_task_start and (
                len(self.initial_belief_xyyaw) != 3
                or not all(math.isfinite(v) for v in self.initial_belief_xyyaw)
                or not self.initial_belief_sigma_xy_m > 0.0
                or not self.initial_belief_sigma_theta_rad > 0.0):
            raise ValueError('initial_belief_from_task_start needs a finite [x, y, yaw] and positive sigmas')
        self.state_reanchor_m = float(self.get_parameter('state_reanchor_m').value)
        self.state_max_predict_dt_s = float(self.get_parameter('state_max_predict_dt_s').value)
        self.state_reject_inflate_m2 = max(
            float(self.get_parameter('state_reject_inflate_m2').value), 0.0
        )
        self.stale_belief_inflate_m2_per_s = max(
            float(self.get_parameter('stale_belief_inflate_m2_per_s').value), 0.0
        )
        self.stale_belief_inflate_cap_m2 = max(
            float(self.get_parameter('stale_belief_inflate_cap_m2').value), 0.0
        )
        self.state_max_correction_jump_m = max(
            float(self.get_parameter('state_max_correction_jump_m').value), 0.0
        )
        self.max_predict_speed_mps = max(
            float(self.get_parameter('max_predict_speed_mps').value), 0.0
        )
        self._latest_odom_yaw = None
        #: stamp of the first odometry message, where the map-frame heading is the
        #: commissioned spawn heading and its drift has not started accumulating.
        self._odom_origin_stamp_s = None
        # Highest odometry stamp already folded into the estimator fields, in
        # integer nanoseconds. Kept private and checked under ``_data_lock`` so
        # two callbacks cannot both pass an earlier unlocked check.
        self._odom_accepted_stamp_ns = None
        # Diagnosis counters for refused odometry. These name a local input
        # disposition; they do not imply the retained history has complete
        # temporal support.
        self._odom_refused_old = 0
        self._odom_refused_duplicate = 0
        self._odom_refused_invalid = 0
        self.heading_update_mode = str(self.get_parameter('heading_update_mode').value).strip().lower()
        if self.heading_update_mode not in ('camera_xy_only', 'coupled'):
            raise RuntimeError(
                "heading_update_mode must be 'camera_xy_only' (heading anchored to odometry, "
                "cameras move x/y only) or 'coupled' (the camera update also moves heading "
                "through the position-heading covariance)")
        self.min_state_cov = float(self.get_parameter('min_state_cov').value)
        self.cov_eig_floor = 1e-9
        self._heading_anchor_applied = False
        self._state_bev_yaw_ignored = False
        self._latest_prediction_source = 0.0
        self._latest_prediction_dt = 0.0
        self._latest_u_pred_v = 0.0
        self._latest_u_pred_omega = 0.0
        self._latest_Q_theta_theta = 0.0
        self._latest_odom_delta_theta = 0.0
        self._latest_cmd_delta_theta = 0.0
        self.debug_runtime = _as_bool(self.get_parameter('debug_runtime').value)
        self.debug_log_period_s = max(0.2, float(self.get_parameter('debug_log_period_s').value))
        self.slow_plan_factor = max(0.1, float(self.get_parameter('slow_plan_factor').value))
        self.slow_correction_ms = max(0.1, float(self.get_parameter('slow_correction_ms').value))

        camera_params = {
            'cam_pos': self.get_parameter('cam_pos').value,
            'look_at': self.get_parameter('look_at').value,
            'img_width': int(self.get_parameter('img_width').value),
            'img_height': int(self.get_parameter('img_height').value),
            'fov_h_rad': float(self.get_parameter('fov_h_rad').value),
        }
        warm_start_shift_steps = self._warm_start_shift_steps_for_rate(self.plan_rate)

        self._camera_params = camera_params
        self._warm_start_shift_steps = warm_start_shift_steps
        self.optimizer_warm_start_shift_steps = warm_start_shift_steps
        self.planner = self._construct_planner()
        self._io_group = ReentrantCallbackGroup()
        # Odometry is an ordered evidence stream, not a latest-value display
        # topic.  Keep it separate from the reentrant command/goal callbacks so
        # concurrent executor threads cannot commit newer stamps before older
        # ones, and retain enough history to ride out short planner stalls.
        self._odom_group = MutuallyExclusiveCallbackGroup()
        self._correction_group = MutuallyExclusiveCallbackGroup()
        self._plan_group = MutuallyExclusiveCallbackGroup()
        self._data_lock = threading.RLock()
        self._correction_lock = threading.RLock()
        _declare_if_not('belief_frame_id', 'map_bev')
        self._belief_frame_id = str(self.get_parameter('belief_frame_id').value).strip()
        if not self._belief_frame_id:
            raise RuntimeError('belief_frame_id must name the authoritative map frame')

        # Subscriptions
        state_qos = QoSProfile(depth=1)
        state_qos.durability = DurabilityPolicy.VOLATILE
        odom_qos = QoSProfile(depth=500)
        odom_qos.durability = DurabilityPolicy.VOLATILE
        # Pose display topics may legitimately keep only the newest sample, but an
        # evidence-bearing correction envelope is an event stream: every published
        # source_batch_id must reach one terminal assimilation outcome.  A depth-one
        # queue silently discarded envelopes while the serialized EKF callback was
        # busy (especially with 960-pixel detector inference).  Keep a bounded but
        # comfortably deep reliable history for correction events.
        correction_event_qos = QoSProfile(depth=1000)
        correction_event_qos.durability = DurabilityPolicy.VOLATILE
        self.state_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/state/bev', self._state_cb, qos_profile=state_qos,
            callback_group=self._correction_group
        )
        self.state_correction_envelope_sub = None
        if (
            self.state_correction_ekf
            and self.state_correction_mode == 'fused'
            and self.require_state_correction_envelope
        ):
            self.state_correction_envelope_sub = self.create_subscription(
                String,
                self.state_correction_envelope_topic,
                self._state_correction_envelope_cb,
                qos_profile=correction_event_qos,
                callback_group=self._correction_group,
            )
        self.map_observations_sub = None
        if self.state_correction_ekf and self.state_correction_mode == 'per_camera':
            self.map_observations_sub = self.create_subscription(
                String, self.map_observations_topic, self._map_observations_cb,
                qos_profile=correction_event_qos, callback_group=self._correction_group
            )
            self.get_logger().info(
                "state_correction_mode=per_camera: folding per-camera map "
                f"observations from {self.map_observations_topic} into the belief "
                "sequentially (one filter, N updates)"
            )
        goal_qos = QoSProfile(depth=1)
        goal_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_bev', self._goal_cb, qos_profile=goal_qos,
            callback_group=self._io_group
        )
        self.pixel_sub = self.create_subscription(
            PoseStamped, self.pixel_topic, self._pixel_cb, qos_profile=state_qos,
            callback_group=self._correction_group
        )
        self.detection_diag_sub = self.create_subscription(
            Float64MultiArray, DETECTION_DIAGNOSTICS_TOPIC, self._detection_diag_cb, qos_profile=state_qos,
            callback_group=self._io_group
        )
        self.cmd_sub = self.create_subscription(
            Twist, self.cmd_topic, self._cmd_cb, qos_profile=state_qos,
            callback_group=self._io_group
        )
        self.odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self._odom_cb, qos_profile=odom_qos,
            callback_group=self._odom_group
        )

        # DIAGNOSTIC: transformed raw-odometry localization path.
        self.diagnostic_odom_pose = None  # (x, y, yaw) in plan frame
        self.diagnostic_odom_pose_stamp = None
        self._tf_buffer = None
        self._tf_listener = None
        self.diagnostic_odom_sub = None
        if self.use_diagnostic_odom_localization:
            import tf2_ros
            from tf2_geometry_msgs import do_transform_pose  # noqa: F401  (registers PoseStamped)
            self._tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
            self._do_transform_pose = do_transform_pose
            self.diagnostic_odom_sub = self.create_subscription(
                Odometry, self.diagnostic_odom_topic, self._diagnostic_odom_cb,
                qos_profile=odom_qos, callback_group=self._odom_group,
            )
            self.get_logger().warn(
                "*** use_diagnostic_odom_localization=TRUE — planner belief is "
                "transformed raw odometry (perception bypassed). DIAGNOSTIC ONLY; "
                "not valid for comparison runs. ***"
            )

        # Publishers
        path_qos = QoSProfile(depth=1)
        path_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.path_pub = self.create_publisher(Path, '/plan', qos_profile=path_qos)
        self.plan_preview_pub = self.create_publisher(Path, '/plan_preview', qos_profile=path_qos)
        self.planner_belief_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/planner_belief', qos_profile=path_qos
        )
        self.belief_state_pub = self.create_publisher(String, '/planner/belief_state', qos_profile=path_qos)
        self.odometry_processed_pub = self.create_publisher(Header, '/planner/odometry_processed', 100)
        self.metrics_pub = self.create_publisher(Float64MultiArray, '/efe/metrics', 10)
        self.planner_diag_pub = self.create_publisher(Float64MultiArray, '/planner/diagnostics', 10)
        self.planner_diag_text_pub = self.create_publisher(String, '/planner/diagnostics_text', 10)
        self.pixel_correction_diag_pub = self.create_publisher(
            Float64MultiArray, '/planner/pixel_correction_diagnostics', 10
        )
        self.correction_assimilation_pub = self.create_publisher(
            String, '/planner/correction_assimilation', 10
        )

        # State
        self.state_msg = None
        self.goal_msg = None
        self._goal_received_logged = False
        self.pixel_meas = None
        self.pixel_stamp = None
        self._latest_detection_diag = None
        self._last_correction_log = 0.0
        self._last_correction_stamp = None
        self._seen_state_source_batch_ids = set()
        self._seen_map_observation_stamps = {}
        self._last_stale_log = 0.0
        self._last_shape_mismatch_log = 0.0
        self._last_runtime_log = 0.0
        self._last_plan_entry_log = 0.0
        self._last_plan_return_log = 0.0
        self._last_slow_plan_log = 0.0
        self._last_slow_correction_log = 0.0
        self._fatal_stop_triggered = False
        self._goal_signature = None
        self._goal_progress_start_dist_m = None
        self.belief_m = None
        self.belief_S = None
        self.belief_stamp = None
        self.last_cmd = np.array([0.0, 0.0], dtype=float)
        self.odom_vel = np.array([0.0, 0.0], dtype=float)
        # Ring buffers of timestamped motion inputs for dead reckoning.  The
        # odometry log is preferred when use_odom_for_predict=True because it
        # represents the encoder/noisy-odometry estimate of what the robot did.
        # The command log remains as a fallback and for diagnostics.
        self._cmd_log: list[tuple[float, float, float]] = []
        self._odom_log: list[tuple[float, float, float]] = []
        self._odom_heading_log = []
        # DDS may deliver an odometry sample just before the corresponding /clock
        # callback is executed. Such a sample is not evidence at the current
        # instant, but discarding it creates an artificial hole in the later replay
        # history. Keep it outside the active history until the local simulation
        # clock reaches its source stamp. A bounded dictionary also permits two
        # future samples that arrive out of order to be committed in source order.
        self._pending_odom_events = {}
        self._PENDING_ODOM_MAX = 4096
        self._CMD_LOG_MAX_S: float = 60.0
        self._latest_measurement_available = False
        self._latest_belief_age_s = math.nan
        with self._data_lock:
            self._ensure_belief_runtime_locked()

        planner_rate = self.local_plan_rate if self.use_hierarchical else self.plan_rate
        self._plan_period_s = 1.0 / max(planner_rate, 0.1)
        self.create_timer(self._plan_period_s, self._plan_once, callback_group=self._plan_group)
        if self.belief_publish_rate > 0.0:
            self._belief_publish_period_s = 1.0 / max(self.belief_publish_rate, 0.1)
            self.create_timer(
                self._belief_publish_period_s,
                self._belief_publish_tick,
                callback_group=self._io_group,
            )
        self._pixel_correction_timer = None
        if self.use_pixel_correction and self.pixel_correction_min_interval_s > 0.0:
            correction_period = max(self.pixel_correction_min_interval_s, 0.02)
            self._pixel_correction_timer = self.create_timer(
                correction_period, self._pixel_correction_timer_cb, callback_group=self._correction_group
            )
        self.get_logger().info(f'Active planner path: {self.planner_path_summary}')
        self.get_logger().info(
            f"{self.NODE_NAME} started "
            f"({self.planner_path_summary}, "
            f"use_obs_risk={self.use_obs_risk}, use_ambiguity={self.use_ambiguity}, "
            f"goal_progress_n_steps={self.goal_progress_n_steps}, "
            f"use_visibility_model={self.use_visibility_model}, "
            f"use_nogo_cost={self.use_nogo_cost}, nogo_penalty_type={self.nogo_penalty_type}, "
            f"use_belief_nogo_cost={self.use_belief_nogo_cost}, "
            f"use_hit_miss_mixture={self.use_hit_miss_mixture}, "
            f"use_pixel_correction={self.use_pixel_correction}, "
            f"cmd_topic={self.cmd_topic}, "
            f"pixel_correction_approx={self.pixel_correction_approx}, "
            f"heading_update_mode={self.heading_update_mode}, "
            f"debug_runtime={self.debug_runtime})"
        )

    def _ensure_belief_runtime_locked(self):
        """Initialize owned metadata; also supports minimal object.__new__ fixtures.

        Production calls this once before timers start. Legacy m/P/time fields
        are adopted only on this first initialization; later writes use the
        central commit and the immutable record is authoritative.
        """
        if hasattr(self, '_belief_record'):
            return
        self._belief_epoch = uuid.uuid4().hex
        self._belief_revision = 0
        self._goal_revision = 0
        self._belief_invalid_reason = ''
        self._belief_clock_ns = None
        self._last_belief_publication = None
        self._belief_frame_id = getattr(self, '_belief_frame_id', 'map_bev')
        self._correction_outcomes = {}
        self._seen_state_source_batch_ids = getattr(self, '_seen_state_source_batch_ids', set())
        self._seen_state_source_members = set()
        self._odom_heading_log = getattr(self, '_odom_heading_log', [])
        self._belief_record = None
        if all(getattr(self, name, None) is not None for name in ('belief_m', 'belief_S', 'belief_stamp')):
            self._belief_record = BeliefRecord.create(
                self.belief_m, self.belief_S, self._stamp_ns(self.belief_stamp),
                self._belief_frame_id, self._belief_epoch, 0)

    @staticmethod
    def _stamp_ns(stamp_msg):
        ns = int(stamp_msg.sec) * 1_000_000_000 + int(stamp_msg.nanosec)
        if ns < 0 or not 0 <= int(stamp_msg.nanosec) < 1_000_000_000:
            raise ValueError('invalid ROS timestamp')
        return ns

    @staticmethod
    def _ns_stamp(ns):
        value = TimeMsg()
        value.sec, value.nanosec = divmod(int(ns), 1_000_000_000)
        return value

    def _observe_belief_clock_locked(self):
        self._ensure_belief_runtime_locked()
        try:
            ns = int(self.get_clock().now().nanoseconds)
            if ns < 0:
                raise ValueError('negative ROS clock')
        except (AttributeError, TypeError, ValueError, OverflowError):
            self._belief_invalid_reason = 'invalid_clock'
            return None
        previous = self._belief_clock_ns
        if previous is not None and ns < previous and not self._belief_invalid_reason:
            # A partial hot reset cannot reconcile the detector, odometry,
            # correction and command histories. Invalidate this epoch until
            # the owning runtime is restarted; do not relabel its old state.
            self._belief_epoch = uuid.uuid4().hex
            self._belief_invalid_reason = 'clock_rewind'
            self._last_belief_publication = None
        self._belief_clock_ns = ns
        return None if self._belief_invalid_reason else ns

    def _commit_belief(self, m, P, stamp_msg, *, motion_support=None):
        """The only recursive m/P/time writer after startup; validates before mutation."""
        with self._data_lock:
            self._ensure_belief_runtime_locked()
            now_ns = self._observe_belief_clock_locked()
            if now_ns is None:
                raise ValueError(f'cannot commit belief in invalid epoch: {self._belief_invalid_reason}')
            record = BeliefRecord.create(
                m, P, self._stamp_ns(stamp_msg), self._belief_frame_id,
                self._belief_epoch, self._belief_revision + 1, motion_support)
            if self._belief_record is not None and record.stamp_ns < self._belief_record.stamp_ns:
                raise ValueError('recursive belief time cannot move backwards')
            self._belief_record = record
            self._belief_revision = record.revision
            self._last_belief_commit_ns = now_ns
            # Compatibility mirrors never alias a returned correction outcome.
            self.belief_m, self.belief_S = record.arrays()
            self.belief_m.setflags(write=False)
            self.belief_S.setflags(write=False)
            self.belief_stamp = self._ns_stamp(record.stamp_ns)
            return record

    def _motion_snapshot_locked(self):
        now_ns = getattr(self, '_belief_clock_ns', None)
        if now_ns is not None:
            self._flush_pending_odom_locked(int(now_ns))
        return MotionHistorySnapshot.capture(
            self._odom_log, self._cmd_log, self.use_odom_for_predict,
            getattr(self, '_odom_heading_log', ()), getattr(self, '_odom_motion_gaps', ()))

    def _run_motion_replay(self, m, P, plan):
        m, P = checked_state(m, P)
        Q_total = np.zeros((3, 3))
        yaw_delta = 0.0
        for a, b, v, w in plan.segments:
            dt = (b - a) * 1e-9
            F = unicycle_jacobian(m, [v, w], dt)
            old_P = P
            m, P = self.planner.predict(m, P, np.array([v, w]), dt=dt)
            m, P = checked_state(m, P)
            Q_total = F @ Q_total @ F.T + (P - F @ old_P @ F.T)
            yaw_delta += w * dt
        return m, P, Q_total, yaw_delta

    def _publish_invalid_belief(self, reason, *, expected_record=_UNSPECIFIED_RECORD,
                              expected_epoch=None):
        with self._data_lock:
            self._ensure_belief_runtime_locked()
            record = self._belief_record
            if expected_record is not _UNSPECIFIED_RECORD and record is not expected_record:
                return
            if expected_epoch is not None and expected_epoch != self._belief_epoch:
                return
            if reason == 'uninitialized' and record is not None:
                return
            if reason == 'future_anchor' and record is not None:
                current_ns = self._observe_belief_clock_locked()
                if current_ns is not None and current_ns >= record.stamp_ns:
                    return
            payload = dict(schema_version=1, initialized=record is not None, epoch=self._belief_epoch,
                revision=self._belief_revision, frame_id=self._belief_frame_id,
                anchor_stamp_ns=record.stamp_ns if record else 0,
                state_stamp_ns=record.stamp_ns if record else 0,
                mean=list(record.mean) if record else None,
                covariance=[list(r) for r in record.covariance] if record else None,
                valid=False, invalid_reason=str(reason), motion_supported=False,
                motion_support=record.motion_support.to_dict() if record else None)
            publisher = getattr(self, 'belief_state_pub', None)
            if publisher is not None:
                msg = String(); msg.data = json.dumps(payload, allow_nan=False)
                publisher.publish(msg)

    def _current_belief_context(self):
        with self._data_lock:
            now_ns = self._observe_belief_clock_locked()
            record = self._belief_record
            valid = (now_ns is not None and record is not None
                     and record.epoch == self._belief_epoch and now_ns >= record.stamp_ns)
            support = record.motion_support if record else None
            if valid:
                plan = plan_replay(self._motion_snapshot_locked(), record.stamp_ns, now_ns,
                                   self.state_max_predict_dt_s)
                support = plan.support.following(support)
                valid = support.supported
            return dict(belief_epoch=self._belief_epoch, belief_revision=self._belief_revision,
                        belief_frame_id=self._belief_frame_id, goal_revision=self._goal_revision,
                        belief_valid=bool(valid), motion_supported=bool(support and support.supported),
                        belief_stamp_ns=record.stamp_ns if record else None,
                        prediction_stamp_ns=now_ns,
                        invalid_reason=self._belief_invalid_reason or ('' if valid else 'unavailable_belief'))

    def _belief_context_is_current(self, meta, *, require_revision=True):
        current = self._current_belief_context()
        keys = ['belief_epoch', 'belief_frame_id', 'goal_revision']
        if require_revision:
            keys.append('belief_revision')
        return bool(current['belief_valid'] and meta.get('belief_valid', False)
                    and all(meta.get(k) == current[k] for k in keys))

    @staticmethod
    def _freeze_outcome(value):
        if isinstance(value, dict):
            return MappingProxyType({k: UnicyclePlannerNode._freeze_outcome(v) for k, v in value.items()})
        if isinstance(value, (list, tuple)):
            return tuple(UnicyclePlannerNode._freeze_outcome(v) for v in value)
        return value

    @staticmethod
    def _outcome_json_value(value):
        if isinstance(value, (dict, MappingProxyType)):
            return {k: UnicyclePlannerNode._outcome_json_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [UnicyclePlannerNode._outcome_json_value(v) for v in value]
        return value

    def _retain_correction_outcome_locked(self, source_batch_id, stamp_msg, status, reason,
                                          before, outcome=None):
        """Called in the same data-lock transaction as the final state commit."""
        self._ensure_belief_runtime_locked()
        if not source_batch_id:
            return None
        if source_batch_id in self._correction_outcomes:
            return self._correction_outcomes[source_batch_id]
        after = self._belief_record
        is_prediction = outcome is not None and outcome.m_pred is not None and outcome.S_pred is not None
        prior_m = outcome.m_pred if is_prediction else (before.mean if before else None)
        prior_P = outcome.S_pred if is_prediction else (before.covariance if before else None)
        if prior_m is not None and not (np.isfinite(prior_m).all() and np.isfinite(prior_P).all()):
            prior_m = prior_P = None
            is_prediction = False
        stamp_ns = self._stamp_ns(stamp_msg)
        apply_ns = (self._last_belief_commit_ns if after is not before
                    else (self._belief_clock_ns if self._belief_clock_ns is not None else 0))
        nis = float(getattr(outcome, 'nis', math.nan))
        payload = dict(schema_version=2, source_batch_id=str(source_batch_id),
            correction_stamp=stamp_ns*1e-9, apply_stamp=apply_ns*1e-9,
            belief_stamp_after=after.stamp_ns*1e-9 if after else None,
            status=status, reason=reason, accepted=status in ('accepted','accepted_bootstrap','reanchored'),
            nis=nis if math.isfinite(nis) else None, epoch=self._belief_epoch,
            revision_before=before.revision if before else 0,
            revision_after=after.revision if after else 0, frame_id=self._belief_frame_id,
            correction_stamp_ns=stamp_ns, apply_stamp_ns=apply_ns,
            belief_stamp_before_ns=before.stamp_ns if before else None,
            belief_stamp_after_ns=after.stamp_ns if after else None,
            prior_kind='prediction' if is_prediction else 'anchor',
            prior_stamp_ns=stamp_ns if is_prediction else (before.stamp_ns if before else None),
            prior_mean=np.asarray(prior_m).tolist() if prior_m is not None else None,
            prior_covariance=np.asarray(prior_P).tolist() if prior_P is not None else None,
            posterior_mean=list(after.mean) if after else None,
            posterior_covariance=[list(row) for row in after.covariance] if after else None,
            state_stamp_ns=after.stamp_ns if after else None, initialized=after is not None,
            valid=bool(after and after.motion_support.supported and not self._belief_invalid_reason),
            motion_supported=bool(after and after.motion_support.supported),
            motion_support=after.motion_support.to_dict() if after else None)
        source = getattr(self, '_active_correction_envelope', None)
        if source is not None:
            payload['source_envelope'] = deepcopy(source)
            for key in ('event_id', 'epoch', 'publication_seq', 'member_ids', 'payload_sha256'):
                if key in source:
                    payload['source_' + key] = deepcopy(source[key])
            for member in source.get('members', ()):
                self._seen_state_source_members.add((member['camera_id'], member['producer_epoch'],
                                                      member['source_frame_id']))
        retained = self._freeze_outcome(payload)
        self._correction_outcomes[source_batch_id] = retained
        self._seen_state_source_batch_ids.add(source_batch_id)
        return retained

    def _publish_safe_stop_command(self):
        """Hook for agent mode; planner-only nodes can ignore."""
        return

    def _fatal_experiment_stop(self, reason: str, exc: Exception | None = None):
        if self._fatal_stop_triggered:
            return
        self._fatal_stop_triggered = True

        try:
            self._publish_safe_stop_command()
        except RuntimeError:
            pass

        detail = reason
        if exc is not None:
            detail = f"{reason}: {type(exc).__name__}: {exc}"
        self.get_logger().error(
            "Fatal experiment integrity failure. Publishing zero command and terminating node. "
            f"Reason: {detail}"
        )
        if exc is not None:
            try:
                tb = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                self.get_logger().error(tb.rstrip())
            except (TypeError, ValueError):
                pass

        # Stop this process so runs fail fast instead of continuing with invalid behavior.
        try:
            rclpy.shutdown()
        except RuntimeError:
            pass
        raise RuntimeError(detail) from exc

    @_serialized_correction
    def _state_cb(self, msg: PoseWithCovarianceStamped):
        with self._data_lock:
            self._ensure_belief_runtime_locked()
            self.state_msg = deepcopy(msg)
        # EKF mode: fold each fused correction into the belief on arrival (the
        # same architecture the single-camera pixel path uses -- corrections are
        # applied here, the planning loop only predicts the committed belief).
        # In per_camera mode the belief is corrected from the per-camera
        # observations instead; applying the fused pose too would fold the same
        # measurements in twice.
        if (
            self.state_correction_ekf
            and self.state_correction_mode == 'fused'
            and not self.require_state_correction_envelope
        ):
            try:
                self._apply_state_correction(msg)
            except Exception as exc:
                self._fatal_experiment_stop("state correction update failed", exc)
        elif not self.state_correction_ekf and not self.use_pixel_correction:
            self._init_belief_from_state(allow_replace=True)

    @_serialized_correction
    def _state_correction_envelope_cb(self, msg: String):
        """Apply one fused correction with its physical detector-batch identity.

        The pose-only topic cannot carry ``source_batch_id``. Evidence-grade runs
        therefore consume this envelope and leave ``/state/bev`` as a display and
        compatibility topic. Duplicate batch identities fail closed before they can
        enter the recursive belief twice.
        """
        if not (
            self.state_correction_ekf
            and self.state_correction_mode == 'fused'
            and self.require_state_correction_envelope
        ):
            return
        try:
            payload = json.loads(msg.data)
            if not isinstance(payload, dict):
                raise ValueError("correction envelope must be a JSON object")
            if type(payload.get('schema_version')) is not int or payload['schema_version'] not in (1, 2):
                raise ValueError("unsupported correction envelope schema")
            if payload['schema_version'] == 2:
                from reliability.fusion_event import FusedCorrectionEvent
                payload = FusedCorrectionEvent.from_json(msg.data).payload
            if payload.get('frame_id') != self._resolve_plan_frame_id():
                raise ValueError("correction envelope frame differs from robot belief frame")
            source_batch_id = str(payload.get('source_batch_id', '') or '').strip()
            if not source_batch_id:
                raise ValueError("correction envelope has no source_batch_id")
            correction_stamp = float(payload['correction_stamp'])
            xy = np.asarray(payload['xy'], dtype=float).reshape(-1)
            covariance = np.asarray(payload['covariance_m2'], dtype=float)
            if xy.size != 2 or covariance.shape != (2, 2):
                raise ValueError("correction envelope needs xy[2] and covariance_m2[2][2]")
            if not (math.isfinite(correction_stamp) and np.isfinite(xy).all()
                    and np.isfinite(covariance).all()):
                raise ValueError("correction envelope contains non-finite values")
            if not np.allclose(covariance, covariance.T, rtol=1e-7, atol=1e-10):
                raise ValueError("correction envelope covariance is not symmetric")
            np.linalg.cholesky(covariance)  # No silent repair of malformed sensor R.
            with self._data_lock:
                self._ensure_belief_runtime_locked()
                duplicate = source_batch_id in self._seen_state_source_batch_ids
                repeated_members = any((member['camera_id'], member['producer_epoch'], member['source_frame_id'])
                    in self._seen_state_source_members for member in payload.get('members', ()))
            if duplicate:
                raise ValueError(f"duplicate source_batch_id {source_batch_id!r}")
            if repeated_members:
                raise ValueError('physical camera frame was already processed in another correction event')
            self._active_correction_envelope = deepcopy(payload)
            try:
                # A large innovation can mean that the recursive belief has
                # escaped the robot.  Re-anchoring is only justified by a
                # corroborated fused measurement: one accepted camera is still
                # allowed through the ordinary Kalman/NIS path, but may not snap
                # the robot belief across the map.  Schema-1 envelopes predate
                # the retained membership list and keep their legacy behaviour.
                allow_reanchor = _correction_envelope_allows_reanchor(payload)
                self._apply_metric_correction(
                    (self._ns_stamp(payload['correction_stamp_ns']) if payload['schema_version'] == 2
                     else self._float_to_stamp(correction_stamp)), xy, covariance,
                    source_batch_id=source_batch_id,
                    allow_same_stamp=payload['schema_version'] == 2,
                    allow_reanchor=allow_reanchor,
                )
            finally:
                self._active_correction_envelope = None
        except Exception as exc:
            self._fatal_experiment_stop("fused correction envelope failed", exc)

    @_serialized_correction
    def _map_observations_cb(self, msg: String):
        if not (self.state_correction_ekf and self.state_correction_mode == 'per_camera'):
            return
        try:
            observations, frame_id = map_observations_from_json(msg.data)
            belief_frame_id = self._resolve_plan_frame_id()
            if frame_id != belief_frame_id:
                raise ValueError(
                    f"map-observation frame {frame_id!r} does not match belief frame {belief_frame_id!r}"
                )
        except Exception as exc:
            self._fatal_experiment_stop("malformed map-observation batch", exc)
            return
        try:
            self._apply_map_observations(observations)
        except Exception as exc:
            self._fatal_experiment_stop("per-camera correction update failed", exc)

    def _update_goal_progress_origin(self, msg: PoseStamped):
        signature = (
            (msg.header.frame_id or '').strip() or 'map_bev',
            float(msg.pose.position.x),
            float(msg.pose.position.y),
        )
        with self._data_lock:
            self._ensure_belief_runtime_locked()
            previous = self._goal_signature
            changed = (
                previous is None
                or signature[0] != previous[0]
                or abs(signature[1] - previous[1]) > 1e-9
                or abs(signature[2] - previous[2]) > 1e-9
            )
            if changed:
                self._goal_revision += 1
                self._goal_signature = signature
                self._goal_progress_start_dist_m = None

    def _warm_start_shift_steps_for_rate(self, plan_rate: float) -> int:
        return max(
            1,
            int(round((1.0 / max(float(plan_rate), 0.1)) / max(self.dt, 1e-3))),
        )

    def _construct_planner(self, **ov):
        """Build a planner from the node's params, with optional overrides.

        Overridable (for the two-stage global/local planners): horizon,
        visibility/no-go switches, goal tightening, optimizer settings, and
        warm-start shift. Keeping these as real overrides prevents hidden
        local/global planner behavior from diverging from the manifest.
        """
        def g(key):
            return ov[key] if key in ov else getattr(self, key)

        def g_default(key, default):
            # For optional planner kwargs that have no node-level attribute: use
            # the override if provided, otherwise the supplied default. The global
            # planner omits these so the LOCAL reference-tracking terms stay OFF
            # (0.0) for it, leaving the global EFE objective unchanged.
            return ov[key] if key in ov else default

        planner = self._build_planner_instance(g, g_default, _as_bool)
        # Gate 0's coherent encoder-drift terms. Set after construction so the flag does not
        # have to be threaded through every planner subclass's kwargs; OFF by default.
        planner.coherent_drift = bool(getattr(self, 'coherent_drift', False))
        return planner

    def _build_planner_instance(self, g, g_default, _as_bool):
        return self.PLANNER_CLASS(
            horizon=int(g('horizon')),
            dt=float(g_default('dt', self.dt)), v_min=self.v_min, v_max=self.v_max, w_min=self.w_min, w_max=self.w_max,
            control_weight=self.control_weight,
            process_noise_xy=self.process_noise_xy, process_noise_theta=self.process_noise_theta, process_noise_model=getattr(self, 'process_noise_model', 'encoder'),
            goal_sigma_uv=self.goal_sigma_uv,
            risk_weight_obs=self.risk_weight_obs, ambiguity_weight=self.ambiguity_weight,
            optimizer_maxiter=int(g('optimizer_maxiter')), optimizer_maxfun=int(g('optimizer_maxfun')),
            optimizer_ftol=float(g('optimizer_ftol')), optimizer_gtol=float(g('optimizer_gtol')),
            optimizer_warm_start=_as_bool(g('optimizer_warm_start')),
            optimizer_warm_start_shift_steps=int(g('optimizer_warm_start_shift_steps')),
            optimizer_multistart=_as_bool(g('optimizer_multistart')),
            optimizer_multistart_include_direct=_as_bool(g('optimizer_multistart_include_direct')),
            optimizer_initial_routes_json=g('optimizer_initial_routes_json'),
            optimizer_terminal_goal_tolerance_m=float(
                g('optimizer_terminal_goal_tolerance_m')
            ),
            approx_method=self.approx_method, use_obs_risk=_as_bool(g('use_obs_risk')),
            use_ambiguity=_as_bool(g('use_ambiguity')), seed=self.seed, camera_params=self._camera_params,
            use_visibility_model=_as_bool(g('use_visibility_model')),
            visibility_target_height_m=self.visibility_target_height_m,
            visibility_geometry_json=self.visibility_geometry_json,
            collision_geometry_json=self.collision_geometry_json,
            visibility_artifact_path=self.visibility_artifact_path,
            camera_network_artifact_path=(
                g_default('camera_network_artifact_path', getattr(self, 'camera_network_artifact_path', ''))
                if _as_bool(g('use_visibility_model')) else ''
            ),
            camera_network_expected_sha256=g_default(
                'camera_network_expected_sha256',
                getattr(self, 'camera_network_expected_sha256', ''),
            ),
            camera_network_expected_source_hashes=g_default(
                'camera_network_expected_source_hashes_json',
                getattr(self, 'camera_network_expected_source_hashes_json', ''),
            ),
            camera_network_camera_ids=g_default(
                'camera_network_camera_ids',
                getattr(self, 'camera_network_camera_ids', ''),
            ),
            camera_network_active_camera_ids=g_default(
                'camera_network_active_camera_ids',
                getattr(self, 'camera_network_active_camera_ids', ''),
            ),
            camera_network_objective=(
                g_default(
                    'camera_network_objective',
                    getattr(self, 'camera_network_objective', 'legacy_pixel_chart'),
                )
                if _as_bool(g('use_visibility_model')) else 'legacy_pixel_chart'
            ),
            network_goal_std_start_m=getattr(self, 'network_goal_std_start_m', None),
            kouw_et1_ambiguity=bool(g_default(
                'kouw_et1_ambiguity', getattr(self, 'kouw_et1_ambiguity', True))),
            network_goal_std_m=float(g_default(
                'network_goal_std_m',
                getattr(self, 'network_goal_std_m', 0.10),
            )),
            camera_network_updates_per_step=int(g_default(
                'camera_network_updates_per_step',
                getattr(self, 'camera_network_updates_per_step', 1),
            )),
            optimizer_control_block_steps=int(g_default(
                'optimizer_control_block_steps',
                getattr(self, 'optimizer_control_block_steps', 1),
            )),
            r_visible_uv=self.r_visible_uv, r_miss_uv=self.r_miss_uv,
            visibility_sigma_kappa=self.visibility_sigma_kappa,
            goal_prior_u_std_start=g('goal_prior_u_std_start'),
            goal_prior_v_std_start=g('goal_prior_v_std_start'),
            goal_prior_u_std_final=g('goal_prior_u_std_final'),
            goal_prior_v_std_final=g('goal_prior_v_std_final'),
            goal_tightening_power=g('goal_tightening_power'),
            goal_progress_n_steps=int(g('goal_progress_n_steps')),
            observation_risk_scale=float(g('observation_risk_scale')),
            ambiguity_term_scale=float(g('ambiguity_term_scale')), discount_gamma=float(g('discount_gamma')),
            use_nogo_cost=_as_bool(g('use_nogo_cost')), nogo_penalty_type=str(g('nogo_penalty_type')),
            nogo_weight=float(g('nogo_weight')), nogo_safe_distance=float(g('nogo_safe_distance')),
            nogo_logbarrier_eps=float(g('nogo_logbarrier_eps')),
            nogo_warning_band=float(g('nogo_warning_band')),
            nogo_near_weight=float(g('nogo_near_weight')),
            use_belief_nogo_cost=_as_bool(g('use_belief_nogo_cost')),
            enforce_planner_lock=bool(g_default('enforce_planner_lock', True)),
            nogo_belief_kappa=float(g('nogo_belief_kappa')),
            use_hit_miss_mixture=_as_bool(g('use_hit_miss_mixture')),
            nogo_mode=str(g('nogo_mode')), driveable_geometry_json=g('driveable_geometry_json'),
            robot_collision_radius_m=self.robot_collision_radius_m, runtime_debug=self.debug_runtime,
            robot_length_m=self.robot_length_m, robot_width_m=self.robot_width_m,
        )

    def _current_goal_progress_index(self, m0, goal_xy) -> float:
        current_dist = float(math.hypot(float(m0[0]) - float(goal_xy[0]), float(m0[1]) - float(goal_xy[1])))
        with self._data_lock:
            start_dist = self._goal_progress_start_dist_m
            if start_dist is None or (not math.isfinite(start_dist)) or start_dist <= 0.0:
                self._goal_progress_start_dist_m = current_dist
                start_dist = current_dist
        if (not math.isfinite(start_dist)) or start_dist <= 1e-9:
            return 0.0
        progress_fraction = max(min((start_dist - current_dist) / start_dist, 1.0), 0.0)
        return progress_fraction * float(max(self.goal_progress_n_steps, 1))

    def _goal_cb(self, msg: PoseStamped):
        if not all(math.isfinite(float(x)) for x in (msg.pose.position.x, msg.pose.position.y)):
            return
        with self._data_lock:
            self._ensure_belief_runtime_locked()
            self.goal_msg = deepcopy(msg)
            first_goal = not self._goal_received_logged
            if first_goal:
                self._goal_received_logged = True
            self._update_goal_progress_origin(self.goal_msg)
        if first_goal:
            self.get_logger().info(
                f"Received goal ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}) "
                f"frame='{msg.header.frame_id or 'map_bev'}'"
            )

    def _cmd_cb(self, msg: Twist):
        values = (float(msg.linear.x), float(msg.angular.z))
        if not all(math.isfinite(x) for x in values):
            return
        with self._data_lock:
            now_ns = self._observe_belief_clock_locked()
            if now_ns is None:
                return
            now_s = now_ns * 1e-9
            self.last_cmd = np.array(values, dtype=float)
            # Log the intended pre-noise command with its sim timestamp.
            self._cmd_log.append((now_s, *values))
            # Trim entries older than the ring-buffer horizon.
            cutoff = now_s - self._CMD_LOG_MAX_S
            while self._cmd_log and self._cmd_log[0][0] < cutoff:
                self._cmd_log.pop(0)

    @staticmethod
    def _yaw_from_quaternion(q) -> float:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _odom_cb(self, msg: Odometry):
        """Fold one odometry event into the estimator, or refuse all of it.

        Parse into locals first, then commit yaw, velocity, origin, history and
        watermark together under one lock. Previously yaw and the origin were
        written outside the lock and no ordering was enforced, so a late message
        could pair its yaw with another event's velocity and append backwards
        history that replay then integrated as real motion.

        Late input is refused rather than inserted in order: retrospective
        insertion would alter a history an accepted correction has already been
        computed from, without rewinding and replaying that correction. Refusing
        old input does NOT establish that the remaining history has complete
        temporal support; that validity question is separate and still open.
        """
        try:
            yaw = odometry_pose_yaw(msg,
                expected_frame=getattr(self, 'odom_frame_id', 'odom'),
                expected_child_frame=getattr(self, 'odom_child_frame_id', 'base_footprint'))
            pose_available = odometry_pose_is_available(msg)
            v_odom = float(msg.twist.twist.linear.x)
            w_odom = float(msg.twist.twist.angular.z)
        except (AttributeError, TypeError, ValueError):
            self._odom_refused_invalid = getattr(self, '_odom_refused_invalid', 0) + 1
            return
        if not all(math.isfinite(x) for x in (yaw, v_odom, w_odom)):
            self._odom_refused_invalid = getattr(self, '_odom_refused_invalid', 0) + 1
            return
        try:
            stamp_ns = self._stamp_ns(msg.header.stamp)
        except (AttributeError, TypeError, ValueError):
            # Do not invent a receipt-time measurement stamp: a wall-clock stamp
            # would place unstamped motion at the wrong point in the history.
            self._odom_refused_invalid = getattr(self, '_odom_refused_invalid', 0) + 1
            return
        stamp_s = stamp_ns * 1e-9

        with self._data_lock:
            now_ns = self._observe_belief_clock_locked()
            if now_ns is None:
                return
            accepted = getattr(self, '_odom_accepted_stamp_ns', None)
            if accepted is not None and stamp_ns <= accepted:
                # An exact duplicate and a conflicting equal-stamp message both add
                # no motion and no process noise. The first event at a timestamp is
                # the one retained.
                if stamp_ns == accepted:
                    self._odom_refused_duplicate = getattr(self, '_odom_refused_duplicate', 0) + 1
                else:
                    self._odom_refused_old = getattr(self, '_odom_refused_old', 0) + 1
                return
            event = (float(yaw), bool(pose_available), float(v_odom), float(w_odom))
            pending = getattr(self, '_pending_odom_events', None)
            if pending is None:
                self._pending_odom_events = pending = {}
            previous = pending.get(stamp_ns)
            if previous is not None:
                if previous != event:
                    self._odom_refused_conflict = getattr(
                        self, '_odom_refused_conflict', 0) + 1
                else:
                    self._odom_refused_duplicate = getattr(
                        self, '_odom_refused_duplicate', 0) + 1
                return
            capacity = int(getattr(self, '_PENDING_ODOM_MAX', 4096))
            if len(pending) >= capacity:
                # This is a bounded transport-reordering buffer, not an alternate
                # motion store. Refusing overflow is preferable to evicting an
                # earlier sample and silently manufacturing a replay gap.
                self._odom_refused_pending_overflow = getattr(
                    self, '_odom_refused_pending_overflow', 0) + 1
                return
            pending[stamp_ns] = event
            if stamp_ns > now_ns:
                self._odom_buffered_future = getattr(
                    self, '_odom_buffered_future', 0) + 1
            self._flush_pending_odom_locked(now_ns)
        # Received counts as processed: a buffered sample is committed as soon as
        # /clock reaches it, with no further work in this node.
        self._init_belief_from_task_prior(stamp_ns)
        self._publish_odometry_processed(stamp_ns)

    def _publish_odometry_processed(self, stamp_ns) -> None:
        """Tell the lockstep scheduler which odometry stamp this node's callback has handled.

        The scheduler holds the next simulation step until this stamp reaches the
        step's odometry, so a slow planner pauses the simulation instead of falling
        behind simulated time.
        """
        publisher = getattr(self, 'odometry_processed_pub', None)
        if publisher is None or stamp_ns is None:
            return
        msg = Header()
        msg.stamp = self._ns_stamp(int(stamp_ns))
        msg.frame_id = 'odom'
        publisher.publish(msg)

    def _flush_pending_odom_locked(self, now_ns: int) -> None:
        """Move only causally available odometry into the replay history.

        The caller holds ``_data_lock``. Future-stamped samples remain pending,
        so they cannot alter a current prediction, heading, or velocity. Once
        ``/clock`` catches up, they are committed in source-time order and become
        ordinary measured-motion evidence.
        """
        pending = getattr(self, '_pending_odom_events', None)
        if not pending:
            return
        for stamp_ns in sorted(t for t in pending if t <= int(now_ns)):
            yaw, pose_available, v_odom, w_odom = pending.pop(stamp_ns)
            accepted = getattr(self, '_odom_accepted_stamp_ns', None)
            if accepted is not None and stamp_ns <= accepted:
                self._odom_refused_old = getattr(self, '_odom_refused_old', 0) + 1
                continue
            stamp_s = stamp_ns * 1e-9
            self._latest_odom_yaw = yaw if pose_available else None
            self.odom_vel = np.array([v_odom, w_odom], dtype=float)
            if getattr(self, '_odom_origin_stamp_s', None) is None:
                self._odom_origin_stamp_s = float(stamp_s)
            self._accumulate_odometry_heading_variance(float(stamp_s), float(v_odom), float(w_odom))
            self._odom_log.append((stamp_s, v_odom, w_odom))
            if pose_available:
                self._odom_heading_log.append((stamp_ns, yaw))
            elif not getattr(self, '_odom_pose_unavailable', False):
                if not hasattr(self, '_odom_motion_gaps'):
                    self._odom_motion_gaps = []
                if accepted is not None:
                    self._odom_motion_gaps.append(
                        (accepted, stamp_ns, 'encoder_pose_unavailable'))
            self._odom_pose_unavailable = not pose_available
            self._odom_accepted_stamp_ns = stamp_ns
            cutoff = stamp_s - self._CMD_LOG_MAX_S
            while len(self._odom_log) > 1 and self._odom_log[1][0] <= cutoff:
                self._odom_log.pop(0)
            while (len(self._odom_heading_log) > 1
                   and self._odom_heading_log[1][0] * 1e-9 <= cutoff):
                self._odom_heading_log.pop(0)

    def _diagnostic_odom_cb(self, msg: Odometry):
        """DIAGNOSTIC: transform raw odom (truth) into the plan frame via TF."""
        source_frame = (msg.header.frame_id or 'odom').strip() or 'odom'
        plan_frame = self._resolve_plan_frame_id()
        pose_world = msg.pose.pose
        if source_frame != plan_frame:
            try:
                tf_msg = self._tf_buffer.lookup_transform(
                    plan_frame, source_frame, Time())
                pose_world = self._do_transform_pose(msg.pose.pose, tf_msg)
            except Exception:
                return
        x = float(pose_world.position.x)
        y = float(pose_world.position.y)
        yaw = self._yaw_from_quaternion(pose_world.orientation)
        with self._data_lock:
            self.diagnostic_odom_pose = (x, y, yaw)
            self.diagnostic_odom_pose_stamp = msg.header.stamp

    @staticmethod
    def _stamp_to_float(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _detection_diag_cb(self, msg: Float64MultiArray):
        try:
            diag = diagnostics_from_message(msg)
        except (KeyError, TypeError, ValueError):
            return
        with self._data_lock:
            self._latest_detection_diag = diag

    def _state_msg_to_belief(self, state_ref: PoseWithCovarianceStamped):
        """Convert the external state estimate into planner belief coordinates."""
        q = state_ref.pose.pose.orientation
        theta = self._yaw_from_quaternion(q)
        m = np.array([
            state_ref.pose.pose.position.x,
            state_ref.pose.pose.position.y,
            theta,
        ], dtype=float)

        cov = state_ref.pose.covariance
        R_xy = self._xy_covariance_from_pose(cov)
        S = np.diag([0.0, 0.0, cov[35] if len(cov) > 35 else 1e-6]).astype(float)
        S[:2, :2] = R_xy
        return m, self._regularize_state_covariance(S)

    def _xy_covariance_from_pose(self, cov):
        """Full 2x2 xy block of a ROS pose covariance, cross terms included.

        camera_manager publishes map-frame CROSS-covariance at indices 1 and 6
        under the anisotropic covariance profiles: pixel noise projected through
        an oblique camera is a long thin ellipse, not a circle, and its axes are
        not aligned with map x/y. Reading only the diagonal silently discards
        that -- the estimate is then weighted as if the error were isotropic,
        which is the very thing the profile exists to fix.

        Falls back to the diagonal if the symmetrized block is not positive
        definite, so a malformed message degrades loudly-but-safely rather than
        poisoning the filter.
        """
        vxx = max(float(cov[0]) if len(cov) > 0 else 0.0, 0.0)
        vyy = max(float(cov[7]) if len(cov) > 7 else 0.0, 0.0)
        vxy = 0.0
        if len(cov) > 6:
            vxy = 0.5 * (float(cov[1]) + float(cov[6]))
        R = np.array([[vxx, vxy], [vxy, vyy]], dtype=float)
        if not np.all(np.isfinite(R)):
            return np.diag([vxx, vyy]).astype(float)
        if vxy != 0.0 and np.min(np.linalg.eigvalsh(R)) <= 0.0:
            self._warn_stale_pixel_once(
                "pose covariance xy block is not positive definite "
                f"(xx={vxx:.4g}, xy={vxy:.4g}, yy={vyy:.4g}); using its diagonal"
            )
            return np.diag([vxx, vyy]).astype(float)
        return R

    def _state_msg_age_s(self, state_ref: PoseWithCovarianceStamped) -> float:
        try:
            return (self.get_clock().now() - Time.from_msg(state_ref.header.stamp)).nanoseconds * 1e-9
        except (AttributeError, TypeError, ValueError):
            return math.inf

    def _state_msg_is_fresh(self, state_ref: PoseWithCovarianceStamped) -> bool:
        age = self._state_msg_age_s(state_ref)
        future_tolerance_s = max(float(self.pixel_timeout_s), 0.25)
        return bool(math.isfinite(age) and age <= float(self.pixel_timeout_s) and age >= -future_tolerance_s)

    def _regularize_state_covariance(self, S):
        """Keep planner belief covariance positive enough for stable updates."""
        return bc.regularize_covariance(S, self.min_state_cov)

    @_serialized_correction
    def _init_belief_from_state(self, *, allow_replace=False):
        with self._data_lock:
            now_ns = self._observe_belief_clock_locked()
            before = self._belief_record
            state = deepcopy(self.state_msg)
            motion = self._motion_snapshot_locked()
            if before is not None and not allow_replace:
                return True
            if state is None or now_ns is None:
                return False
            target_ns = self._stamp_ns(state.header.stamp)
            if target_ns > now_ns or (before is not None and target_ns <= before.stamp_ns):
                return False
            if self.skip_stale_pixel_correction and not self._state_msg_is_fresh(state):
                return False
            if (state.header.frame_id or self._belief_frame_id) != self._belief_frame_id:
                raise ValueError('state initialization frame differs from belief frame')
        m, P = self._state_msg_to_belief(state)
        support = MotionSupport(target_ns, target_ns)
        if before is not None:
            plan = plan_replay(motion, before.stamp_ns, target_ns,
                               float(getattr(self, 'state_max_predict_dt_s', 1.5)))
            predicted, covariance, _, _ = self._run_motion_replay(*before.arrays(), plan)
            # Legacy position replacement contributes independent XY; retaining
            # unrelated old XY/yaw cross terms can produce an indefinite matrix.
            m[2], P[2,2] = predicted[2], covariance[2,2]
            P[:2,2] = 0.; P[2,:2] = 0.
            support = plan.support.following(before.motion_support)
        self._commit_belief(m, P, state.header.stamp, motion_support=support)
        return True

    def _matching_detection_diag_locked(self, stamp_msg):
        """Return the diagnostics message that belongs to a pixel observation."""
        if self._latest_detection_diag is None:
            return None
        diag_ref = dict(self._latest_detection_diag)
        try:
            stamp_s = self._stamp_to_float(stamp_msg)
            diag_stamp = float(diag_ref.get('stamp', math.nan))
        except (AttributeError, TypeError, ValueError):
            return None
        if (not math.isfinite(diag_stamp)) or abs(diag_stamp - stamp_s) > 1e-3:
            return None
        return diag_ref

    @_serialized_correction
    def _pixel_cb(self, msg: PoseStamped):
        u = msg.pose.position.x
        v = msg.pose.position.y
        # Apply the same BEV calibration the state node and logger use so all
        # nodes converge to the same world position from the same pixel: the
        # position-dependent affine when configured, else the legacy constant
        # y-offset. Re-project the calibrated world point back to a pixel.
        if self._bev_affine is not None or self.bev_y_calibration_offset_m != 0.0:
            try:
                camera = self.planner.camera
                xy = camera.pixel_to_world(u, v)
                if xy is not None:
                    if self._bev_affine is not None:
                        c = self._bev_affine
                        x_cal = c[0] * xy[0] + c[1] * xy[1] + c[2]
                        y_cal = c[3] * xy[0] + c[4] * xy[1] + c[5]
                    else:
                        x_cal, y_cal = xy[0], xy[1] + self.bev_y_calibration_offset_m
                    u_cal, v_cal, vis = camera.world_to_pixel(x_cal, y_cal, 0.0)
                    if vis:
                        u, v = u_cal, v_cal
            except Exception:
                pass
        with self._data_lock:
            self.pixel_meas = np.array([u, v], dtype=float)
            self.pixel_stamp = msg.header.stamp

        if not self.use_pixel_correction:
            return
        if self.pixel_correction_min_interval_s > 0.0:
            return
        self._apply_pixel_correction(msg.header.stamp, source='callback')

    @_serialized_correction
    def _pixel_correction_timer_cb(self):
        if not self.use_pixel_correction or self.pixel_correction_min_interval_s <= 0.0:
            return
        with self._data_lock:
            stamp_ref = deepcopy(self.pixel_stamp)
        if stamp_ref is None:
            return
        self._apply_pixel_correction(stamp_ref, source='timer')

    def _stamp_age_s(self, stamp_msg) -> float:
        try:
            return (self.get_clock().now() - Time.from_msg(stamp_msg)).nanoseconds * 1e-9
        except (AttributeError, TypeError, ValueError):
            return math.nan

    def _warn_stale_pixel_once(self, message: str):
        now_wall = time.monotonic()
        if now_wall - self._last_stale_log > 2.0:
            self.get_logger().warn(message)
            self._last_stale_log = now_wall

    def _pixel_correction_is_throttled(self, stamp_msg) -> bool:
        if self.pixel_correction_min_interval_s <= 0.0:
            return False
        with self._data_lock:
            last_correction_stamp = self._last_correction_stamp
        if last_correction_stamp is None:
            return False
        try:
            dt_since_correction = (
                Time.from_msg(stamp_msg) - Time.from_msg(last_correction_stamp)
            ).nanoseconds * 1e-9
        except (AttributeError, TypeError, ValueError):
            return False
        return bool(0.0 <= dt_since_correction < self.pixel_correction_min_interval_s)

    def _replay_cmd_log_interval(self, m0, S0, from_stamp, to_stamp,
                                   fallback_cmd, fallback_dt, *, motion_snapshot=None):
        start_ns, end_ns = self._stamp_ns(from_stamp), self._stamp_ns(to_stamp)
        if motion_snapshot is None:
            with self._data_lock:
                motion_snapshot = self._motion_snapshot_locked()
        plan = plan_replay(motion_snapshot, start_ns, end_ns,
                           float(self.state_max_predict_dt_s) if hasattr(self, 'state_max_predict_dt_s') else 1.5)
        m, P, Q, yaw = self._run_motion_replay(m0, S0, plan)
        return m, P, {
            'cmd_replay_count': float(plan.event_count),
            'cmd_replay_duration_s': (end_ns - start_ns) * 1e-9,
            'cmd_replay_used_fallback': float(not plan.support.supported),
            'motion_replay_source_code': {'odom': 1., 'command': 2., 'none': 3.}[plan.support.source],
            'motion_support': plan.support.to_dict(),
            'applied_process_covariance': Q.tolist(),
            'integrated_yaw_delta': yaw,
        }

    def _pixel_correction_dt_s(self, stamp_msg) -> float | None:
        with self._data_lock:
            self._ensure_belief_runtime_locked()
            record = self._belief_record
        if record is None:
            return None
        dt_ns = self._stamp_ns(stamp_msg) - record.stamp_ns
        if dt_ns <= 0:
            return None
        dt_s = dt_ns * 1e-9
        max_dt_s = max(2.0 * float(self.pixel_timeout_s), 4.0 * float(self.dt), 0.5)
        return dt_s if dt_s <= max_dt_s else None

    def _snapshot_pixel_correction_inputs(self, stamp_msg):
        with self._data_lock:
            self._ensure_belief_runtime_locked()
            record = self._belief_record
            meas = None if self.pixel_meas is None else self.pixel_meas.copy()
            measured_stamp = getattr(self, 'pixel_stamp', None)
            if measured_stamp is not None and self._stamp_ns(measured_stamp) != self._stamp_ns(stamp_msg):
                return None
            if record is None or meas is None:
                return None
            m, P = record.arrays()
            return bc.CorrectionSnapshot(
                belief_m=m, belief_S=P, belief_stamp=self._ns_stamp(record.stamp_ns),
                cmd=self.last_cmd.copy(), meas=meas, meas_stamp=deepcopy(stamp_msg),
                yaw_meas=None, yaw_sigma=math.nan, yaw_source=0.0,
                motion_snapshot=self._motion_snapshot_locked(), belief_record=record)

    def _correction_gates(
        self,
        *,
        metric_measurement: bool = False,
        allow_reanchor: bool = True,
        validated_replay_dt_s: float | None = None,
    ) -> bc.CorrectionGates:
        """Thresholds for the shared gate chain, from the node's parameters.

        ``metric_measurement`` enables the divergence guard, whose threshold is
        in metres and so only means anything when the measurement is a map
        position (the fused /state/bev path), not pixels.
        """
        metric_replay_ceiling = float(
            getattr(self, 'state_max_predict_dt_s', 1.5)
        )
        if metric_measurement and validated_replay_dt_s is not None:
            # The caller has already proved that timestamped motion inputs cover
            # this entire camera-outage interval. Widen only the total-dt gate;
            # plan_replay still uses state_max_predict_dt_s as its maximum input
            # sample gap, so stale/absent odometry cannot pass through here.
            metric_replay_ceiling = max(
                metric_replay_ceiling, float(validated_replay_dt_s)
            )
        return bc.CorrectionGates(
            pixel_timeout_s=float(self.pixel_timeout_s),
            dt_nominal_s=float(self.dt),
            skip_stale=bool(self.skip_stale_pixel_correction),
            max_jump_m=float(
                self.state_max_correction_jump_m if metric_measurement
                else self.pixel_max_correction_jump_m
            ),
            nis_threshold=float(self.pixel_correction_nis_threshold),
            cov_eig_floor=float(self.cov_eig_floor),
            min_state_cov=float(self.min_state_cov),
            reanchor_innov_m=(
                float(self.state_reanchor_m)
                if metric_measurement and allow_reanchor
                else 0.0
            ),
            max_predict_speed_mps=float(self.max_predict_speed_mps),
            # The fused metric path uses the configured ceiling, so the node's own
            # support pre-check and this gate chain cannot disagree. The pixel path
            # passes 0 and keeps its derived value, leaving that baseline unchanged.
            max_predict_dt_s=(
                metric_replay_ceiling if metric_measurement else 0.0
            ),
        )

    def _log_pixel_shape_error_once(self, message: str):
        now_wall = time.monotonic()
        if now_wall - self._last_shape_mismatch_log > 2.0:
            self.get_logger().error(message)
            self._last_shape_mismatch_log = now_wall

    project_to_psd = staticmethod(bc.project_to_psd)

    def _publish_pixel_correction_diagnostics(
        self,
        *,
        stamp_msg,
        age,
        dt_s,
        p_vis,
        gain_scale,
        innov,
        xy_update_norm_m,
        yaw_info,
        m_pred,
        next_m,
        meas,
        mu_y,
        R_eff,
        yaw_meas,
        yaw_sigma,
        yaw_source,
        nis=float('nan'),
        accepted=True,
        reject_reason_code=0.0,
        apply_stamp_s=math.nan,
        belief_input_stamp_s=math.nan,
        cmd_replay_count=math.nan,
        cmd_replay_duration_s=math.nan,
        cmd_replay_used_fallback=math.nan,
        motion_replay_source_code=math.nan,
        nis_threshold=math.nan,
        K_theta_u=math.nan,
        K_theta_v=math.nan,
        measurement_space=bc.SPACE_PIXEL_UV,
        predict_clipped_m=0.0,
        camera_index=math.nan,
    ):
        diag_msg = Float64MultiArray()
        r_eff = np.asarray(R_eff, dtype=float)
        if not math.isfinite(float(apply_stamp_s)):
            apply_stamp_s = float(self.get_clock().now().nanoseconds) * 1e-9
        expected_after_u = math.nan
        expected_after_v = math.nan
        expected_after_visible = math.nan
        try:
            expected_after_u, expected_after_v, visible = self.planner.camera.world_to_pixel(
                float(next_m[0]),
                float(next_m[1]),
                0.0,
            )
            expected_after_visible = 1.0 if bool(visible) else 0.0
        except Exception:
            expected_after_u = math.nan
            expected_after_v = math.nan
            expected_after_visible = math.nan
        diag_msg.data = [
            float(self._stamp_to_float(stamp_msg)),
            1.0,
            float(age),
            float(dt_s),
            float(p_vis),
            float(gain_scale),
            float(innov[0]) if innov.size > 0 else math.nan,
            float(innov[1]) if innov.size > 1 else math.nan,
            float(xy_update_norm_m),
            float(yaw_info['theta_update_from_uv_rad']),
            1.0 if yaw_info['yaw_correction_applied'] else 0.0,
            float(yaw_info['innov_theta']),
            float(yaw_info['k_theta_theta']),
            float(yaw_info['theta_update_total_rad']),
            float(m_pred[0]),
            float(m_pred[1]),
            float(m_pred[2]),
            float(next_m[0]),
            float(next_m[1]),
            float(next_m[2]),
            float(meas[0]) if meas.size > 0 else math.nan,
            float(meas[1]) if meas.size > 1 else math.nan,
            float(mu_y[0]) if mu_y.size > 0 else math.nan,
            float(mu_y[1]) if mu_y.size > 1 else math.nan,
            float(r_eff[0, 0]) if r_eff.ndim == 2 and r_eff.shape[0] > 0 and r_eff.shape[1] > 0 else math.nan,
            float(r_eff[1, 1]) if r_eff.ndim == 2 and r_eff.shape[0] > 1 and r_eff.shape[1] > 1 else math.nan,
            float(yaw_meas) if yaw_meas is not None and math.isfinite(float(yaw_meas)) else math.nan,
            float(yaw_sigma) if math.isfinite(float(yaw_sigma)) else math.nan,
            float(yaw_source),
            float(nis) if math.isfinite(float(nis)) else math.nan,
            1.0 if accepted else 0.0,
            float(reject_reason_code),
            float(apply_stamp_s),
            float(belief_input_stamp_s),
            float(cmd_replay_count),
            float(cmd_replay_duration_s),
            float(cmd_replay_used_fallback),
            float(nis_threshold),
            float(expected_after_u) if math.isfinite(float(expected_after_u)) else math.nan,
            float(expected_after_v) if math.isfinite(float(expected_after_v)) else math.nan,
            float(expected_after_visible) if math.isfinite(float(expected_after_visible)) else math.nan,
            float(motion_replay_source_code) if math.isfinite(float(motion_replay_source_code)) else math.nan,
            float(K_theta_u) if math.isfinite(float(K_theta_u)) else math.nan,
            float(K_theta_v) if math.isfinite(float(K_theta_v)) else math.nan,
            # Appended 2026-07-29 with the single/multicam chain consolidation.
            # The pixel_corr_* columns are now shared by both stacks, so a reader
            # needs to know which one produced the row: innov/meas are pixels
            # when 0, metres when 1.
            float(measurement_space),
            float(predict_clipped_m),
            # Which camera produced this correction in per_camera mode (index into
            # the batch, sorted by stamp); NaN for the fused/pixel paths.
            float(camera_index),
        ]
        self.pixel_correction_diag_pub.publish(diag_msg)

    _pixel_correction_reject_code = staticmethod(bc.reject_code)

    def _publish_pixel_correction_rejection(
        self,
        stamp_msg,
        *,
        reason: str,
        age=math.nan,
        dt_s=math.nan,
        m_pred=None,
        meas=None,
        mu_y=None,
        innov=None,
        xy_update_norm_m=math.nan,
        R_eff=None,
        nis=math.nan,
        belief_input_stamp_s=math.nan,
        cmd_replay_count=math.nan,
        cmd_replay_duration_s=math.nan,
        cmd_replay_used_fallback=math.nan,
        motion_replay_source_code=math.nan,
        measurement_space=bc.SPACE_PIXEL_UV,
        predict_clipped_m=0.0,
        camera_index=math.nan,
    ):
        nan_state = np.array([math.nan, math.nan, math.nan], dtype=float)
        nan_meas = np.array([math.nan, math.nan], dtype=float)
        yaw_info = {
            'theta_update_from_uv_rad': math.nan,
            'yaw_correction_applied': False,
            'innov_theta': math.nan,
            'k_theta_theta': math.nan,
            'theta_update_total_rad': math.nan,
        }
        self._publish_pixel_correction_diagnostics(
            stamp_msg=stamp_msg,
            age=age,
            dt_s=dt_s,
            p_vis=math.nan,
            gain_scale=math.nan,
            innov=np.asarray(innov if innov is not None else nan_meas, dtype=float),
            xy_update_norm_m=xy_update_norm_m,
            yaw_info=yaw_info,
            m_pred=np.asarray(m_pred if m_pred is not None else nan_state, dtype=float),
            next_m=nan_state,
            meas=np.asarray(meas if meas is not None else nan_meas, dtype=float),
            mu_y=np.asarray(mu_y if mu_y is not None else nan_meas, dtype=float),
            R_eff=np.asarray(R_eff if R_eff is not None else np.full((2, 2), math.nan), dtype=float),
            yaw_meas=math.nan,
            yaw_sigma=math.nan,
            yaw_source=0.0,
            nis=nis,
            accepted=False,
            reject_reason_code=self._pixel_correction_reject_code(reason),
            apply_stamp_s=float(self.get_clock().now().nanoseconds) * 1e-9,
            belief_input_stamp_s=belief_input_stamp_s,
            cmd_replay_count=cmd_replay_count,
            cmd_replay_duration_s=cmd_replay_duration_s,
            cmd_replay_used_fallback=cmd_replay_used_fallback,
            motion_replay_source_code=motion_replay_source_code,
            nis_threshold=float(self.pixel_correction_nis_threshold),
            measurement_space=measurement_space,
            predict_clipped_m=predict_clipped_m,
            camera_index=camera_index,
        )

    def _correction_reject_warning(self, outcome) -> str | None:
        """Operator-facing text for a rejection, or None if it warns elsewhere."""
        if outcome.reason is bc.RejectReason.JUMP_TOO_LARGE:
            return (
                f"Pixel correction jump {outcome.xy_update_norm_m:.3f} m exceeds "
                f"limit {self.pixel_max_correction_jump_m:.3f} m; rejecting"
            )
        if outcome.reason is bc.RejectReason.NIS_TOO_LARGE:
            return (
                f"Pixel correction NIS {outcome.nis:.2f} exceeds "
                f"threshold {self.pixel_correction_nis_threshold:.2f}; rejecting"
            )
        return None

    def _publish_correction_outcome(self, stamp_msg, outcome, *, camera_index=math.nan):
        """Publish diagnostics for one :class:`bc.CorrectionOutcome`."""
        snapshot = outcome.snapshot
        belief_input_stamp_s = (
            self._stamp_to_float(snapshot.belief_stamp) if snapshot is not None else math.nan
        )
        meta = outcome.replay_meta or {}
        replay_kwargs = {
            'belief_input_stamp_s': belief_input_stamp_s,
            'cmd_replay_count': float(meta.get('cmd_replay_count', math.nan)),
            'cmd_replay_duration_s': float(meta.get('cmd_replay_duration_s', math.nan)),
            'cmd_replay_used_fallback': float(meta.get('cmd_replay_used_fallback', math.nan)),
            'motion_replay_source_code': float(meta.get('motion_replay_source_code', math.nan)),
            'measurement_space': float(outcome.measurement_space),
            'predict_clipped_m': float(outcome.predict_clipped_m),
            'camera_index': float(camera_index),
        }
        if not outcome.accepted:
            self._publish_pixel_correction_rejection(
                stamp_msg,
                reason=outcome.reason,
                age=outcome.age,
                dt_s=outcome.dt_s,
                m_pred=outcome.m_pred,
                meas=outcome.meas,
                mu_y=outcome.mu_y,
                innov=outcome.innov,
                xy_update_norm_m=outcome.xy_update_norm_m,
                R_eff=outcome.R_eff,
                nis=outcome.nis,
                **replay_kwargs,
            )
            return

        K_theta_u = math.nan
        K_theta_v = math.nan
        K_mat = outcome.K
        if K_mat is not None and K_mat.shape[0] >= 3:
            if K_mat.shape[1] >= 1:
                K_theta_u = float(K_mat[2, 0])
            if K_mat.shape[1] >= 2:
                K_theta_v = float(K_mat[2, 1])

        self._publish_pixel_correction_diagnostics(
            stamp_msg=stamp_msg,
            age=outcome.age,
            dt_s=outcome.dt_s,
            p_vis=outcome.p_vis,
            gain_scale=outcome.gain_scale,
            innov=outcome.innov,
            xy_update_norm_m=outcome.xy_update_norm_m,
            yaw_info=outcome.yaw_info,
            m_pred=outcome.m_pred,
            next_m=outcome.next_m,
            meas=outcome.meas,
            mu_y=outcome.mu_y,
            R_eff=outcome.R_eff,
            yaw_meas=snapshot.yaw_meas if snapshot is not None else math.nan,
            yaw_sigma=snapshot.yaw_sigma if snapshot is not None else math.nan,
            yaw_source=snapshot.yaw_source if snapshot is not None else 0.0,
            nis=outcome.nis,
            accepted=True,
            reject_reason_code=bc.ACCEPTED_CODE,
            apply_stamp_s=float(self.get_clock().now().nanoseconds) * 1e-9,
            nis_threshold=float(self.pixel_correction_nis_threshold),
            K_theta_u=K_theta_u,
            K_theta_v=K_theta_v,
            **replay_kwargs,
        )

    @_serialized_correction
    def _apply_pixel_correction(self, stamp_msg, *, source='callback'):
        stamp_ns = self._stamp_ns(stamp_msg)
        event_id = f'pixel:{stamp_ns}'
        with self._data_lock:
            now_ns = self._observe_belief_clock_locked()
            if event_id in self._correction_outcomes:
                return
            before = self._belief_record
        age = self._stamp_age_s(stamp_msg)
        if now_ns is None or self._correction_gates().age_is_invalid(age):
            outcome = bc.CorrectionOutcome(reason=bc.RejectReason.STALE_AGE, age=age)
            with self._data_lock:
                self._retain_correction_outcome_locked(event_id, stamp_msg, 'dropped',
                    outcome.reason.value, before, outcome)
            self._warn_stale_pixel_once(f'Skipping time-inconsistent pixel measurement (age {age:.2f}s)')
            self._publish_correction_outcome(stamp_msg, outcome)
            self._publish_correction_assimilation(source_batch_id=event_id, stamp_msg=stamp_msg,
                status='dropped', reason=outcome.reason.value, outcome=outcome)
            return
        if before is None and not self._init_belief_from_state():
            return
        with self._data_lock:
            before = self._belief_record
        if stamp_ns <= before.stamp_ns or self._pixel_correction_is_throttled(stamp_msg):
            return
        snapshot = self._snapshot_pixel_correction_inputs(stamp_msg)
        if snapshot is None:
            return
        dt_s = (stamp_ns - before.stamp_ns)*1e-9
        corr_method = self.approx_method if self.pixel_correction_approx == 'AUTO' else self.pixel_correction_approx
        try:
            replay = lambda *args: self._replay_cmd_log_interval(
                *args, motion_snapshot=snapshot.motion_snapshot)
            if self._pixel_correction_dt_s(stamp_msg) is None:
                outcome = bc.CorrectionOutcome(reason=bc.RejectReason.DT_IMPLAUSIBLE,
                    age=age, dt_s=dt_s, snapshot=snapshot)
            else:
                outcome = bc.apply_correction(
                    source=bc.PixelMeasurementSource(planner=self.planner,
                        planner_for_obs=getattr(self, 'global_planner', None) or self.planner,
                        snapshot_fn=lambda: snapshot, corr_method=corr_method),
                    gates=self._correction_gates(), replay=replay, age=age, dt_s=dt_s,
                    on_shape_error=self._log_pixel_shape_error_once)
            if outcome.m_pred is None or outcome.S_pred is None:
                outcome.m_pred, outcome.S_pred, outcome.replay_meta = replay(
                    *before.arrays(), self._ns_stamp(before.stamp_ns), stamp_msg, snapshot.cmd, dt_s)
            m, P = checked_state(outcome.next_m, outcome.next_S) if outcome.accepted else checked_state(
                outcome.m_pred, outcome.S_pred)
            support = MotionSupport.from_dict(outcome.replay_meta['motion_support']).following(before.motion_support)
            with self._data_lock:
                record = self._commit_belief(m, P, stamp_msg, motion_support=support)
                if outcome.accepted:
                    outcome.next_m, outcome.next_S = record.arrays()
                    outcome.yaw_info = bc.yaw_report(outcome.next_m, outcome.next_S, outcome.m_pred)
                    self._last_correction_stamp = deepcopy(stamp_msg)
                self._retain_correction_outcome_locked(event_id, stamp_msg,
                    'accepted' if outcome.accepted else 'rejected', outcome.reason.value, before, outcome)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
            with self._data_lock:
                self._retain_correction_outcome_locked(event_id, stamp_msg, 'rejected',
                    self._belief_invalid_reason or 'invalid_prediction', before)
            self._fatal_experiment_stop('invalid pixel prediction or commit', exc)
            return
        warning = self._correction_reject_warning(outcome)
        if warning is not None:
            self._warn_stale_pixel_once(warning)
        self._publish_correction_outcome(stamp_msg, outcome)
        self._publish_correction_assimilation(source_batch_id=event_id, stamp_msg=stamp_msg,
            status='accepted' if outcome.accepted else 'rejected', reason=outcome.reason.value,
            outcome=outcome)

    def _belief_snapshot_for_planning(self):
        with self._data_lock:
            self._ensure_belief_runtime_locked()
            record = self._belief_record
            if record is None:
                return None
            m, P = record.arrays()
            return dict(m=m, S=P, stamp=self._ns_stamp(record.stamp_ns),
                        pixel_stamp=deepcopy(self.pixel_stamp), last_cmd=self.last_cmd.copy(),
                        belief_record=record, motion_snapshot=self._motion_snapshot_locked())

    def _belief_age_for_planning(self, now_msg, stamp_ref) -> float | None:
        try:
            dt_ns = self._stamp_ns(now_msg) - self._stamp_ns(stamp_ref)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None
        return dt_ns * 1e-9 if dt_ns >= 0 else None

    def _pixel_measurement_available_for_planning(self, now_msg, pixel_stamp_ref) -> bool:
        if pixel_stamp_ref is None:
            return False
        try:
            raw_measurement_age = (
                Time.from_msg(now_msg) - Time.from_msg(pixel_stamp_ref)
            ).nanoseconds * 1e-9
        except (AttributeError, TypeError, ValueError):
            raw_measurement_age = math.inf
        return bool(0.0 <= raw_measurement_age <= self.pixel_timeout_s)

    def _reset_prediction_diagnostics(self):
        self._latest_prediction_source = 0.0
        self._latest_prediction_dt = 0.0
        self._latest_u_pred_v = 0.0
        self._latest_u_pred_omega = 0.0
        self._latest_Q_theta_theta = 0.0
        self._latest_odom_delta_theta = 0.0
        self._latest_cmd_delta_theta = 0.0

    def _predict_belief_to_now(self, m0, S0, last_cmd, belief_age_s: float, now_msg,
                               *, motion_snapshot=None, diagnostics=None):
        if not math.isfinite(belief_age_s) or belief_age_s < 0:
            raise ValueError('invalid prediction interval')
        end_ns = self._stamp_ns(now_msg)
        start_ns = end_ns - round(belief_age_s * 1e9)
        if motion_snapshot is None:
            with self._data_lock:
                motion_snapshot = self._motion_snapshot_locked()
        plan = plan_replay(motion_snapshot, start_ns, end_ns,
                           float(getattr(self, 'state_max_predict_dt_s', 1.5)))
        m, P, Q, yaw = self._run_motion_replay(m0, S0, plan)
        self._latest_prediction_source = {'odom': 1., 'command': 2., 'none': 0.}[plan.support.source]
        self._latest_prediction_dt = belief_age_s
        self._latest_u_pred_v = plan.segments[-1][2] if plan.segments else 0.
        self._latest_u_pred_omega = plan.segments[-1][3] if plan.segments else 0.
        self._latest_Q_theta_theta = float(Q[2, 2])
        self._latest_odom_delta_theta = yaw if plan.support.source == 'odom' else 0.
        self._latest_cmd_delta_theta = yaw if plan.support.source == 'command' else 0.
        if diagnostics is not None:
            diagnostics.update(prediction_source={'odom': 1., 'command': 2., 'none': 0.}[plan.support.source],
                prediction_dt_s=belief_age_s, u_pred_v=plan.segments[-1][2] if plan.segments else 0.,
                u_pred_omega=plan.segments[-1][3] if plan.segments else 0.,
                Q_theta_theta=float(Q[2,2]), odom_delta_theta=yaw if plan.support.source == 'odom' else 0.,
                cmd_delta_theta=yaw if plan.support.source == 'command' else 0.)
        return m, P

    def _map_frame_heading(self, stamp_msg=None, *, motion_snapshot=None):
        """Use odometry heading at the requested state instant, never receipt time."""
        if stamp_msg is None:
            return None if self._latest_odom_yaw is None else float(wrap_angle(
                self._latest_odom_yaw + self.odom_yaw_offset_rad))
        target_ns = self._stamp_ns(stamp_msg)
        if motion_snapshot is None:
            with self._data_lock:
                motion_snapshot = self._motion_snapshot_locked()
        previous = next(((t, yaw) for t, yaw in reversed(motion_snapshot.headings)
                         if t <= target_ns), None)
        if previous is None:
            return None
        t, yaw = previous
        odom_only = MotionHistorySnapshot(motion_snapshot.odom, (), True,
                                          motion_snapshot.headings, motion_snapshot.gaps)
        plan = plan_replay(odom_only, t, target_ns,
                           float(getattr(self, 'state_max_predict_dt_s', 1.5)))
        if not plan.support.supported:
            return None
        delta = sum((b-a)*1e-9*w for a, b, _, w in plan.segments)
        return float(wrap_angle(yaw + delta + self.odom_yaw_offset_rad))

    def _map_frame_heading_variance(self, stamp_msg) -> float:
        """How wrong the map-frame odometry heading can be, at this instant.

        The belief's heading mean is taken from map-frame odometry, so its variance has
        to describe that same quantity. Calling it non-informative (pi^2) while using it
        as the mean is a contradiction, and a costly one: with a coupled update, an xy
        measurement against a pi^2 heading prior swings the heading by more than 90
        degrees on the first correction, and the robot then predicts its own motion in
        the wrong direction and drives into a rack. Measured on the frozen route: the
        belief heading jumped 0.28 -> 2.69 rad on one correction and stayed ~112 degrees
        wrong until the run ended in a collision 1.7 m later.

        The heading starts at the commissioned spawn heading -- the same knowledge the
        camera_xy_only mode leans on every update -- and drifts from there at the
        filter's own heading process noise. So the variance is that drift, integrated
        since odometry began, and it is bounded by the non-informative value so this can
        never claim more than knowing nothing.
        """
        floor_var = float(math.radians(0.5) ** 2)
        origin = self._odom_origin_stamp_s
        if origin is None:
            return NONINFORMATIVE_YAW_VAR
        try:
            elapsed_s = float(self._stamp_to_float(stamp_msg)) - float(origin)
        except (AttributeError, TypeError, ValueError):
            return NONINFORMATIVE_YAW_VAR
        if not math.isfinite(elapsed_s) or elapsed_s < 0.0:
            return NONINFORMATIVE_YAW_VAR
        log = getattr(self, '_odom_heading_var_log', None)
        if log and len(log) > 1:
            ts, vs = zip(*log)
            drift_var = float(np.interp(float(origin) + elapsed_s, ts, vs))
        else:
            drift_var = float(self.process_noise_theta) ** 2 * elapsed_s
        return float(min(NONINFORMATIVE_YAW_VAR, max(floor_var, drift_var)))

    def _accumulate_odometry_heading_variance(self, stamp_s, v, w) -> None:
        """Integrate the heading PSD of the process-noise model along the odometry.

        With the encoder model the heading PSD depends on the measured (v, w), so the
        odometry-heading variance is the integral of sigma_w^2(v, w) since odometry began,
        the same quantity the prediction step accumulates. The constant model reduces to
        process_noise_theta^2 * t.
        """
        log = getattr(self, '_odom_heading_var_log', None)
        if log is None:
            self._odom_heading_var_log = log = [(float(stamp_s), 0.0)]
            return
        t_prev, var_prev = log[-1]
        dt = float(stamp_s) - t_prev
        if not math.isfinite(dt) or dt <= 0.0:
            return
        psd = getattr(getattr(self, 'planner', None), 'process_noise_psd', None)
        if psd is not None:
            from planning.core.encoder_noise_model import encoder_psd
            rate = float(encoder_psd(v, w, psd)[1])
        else:
            rate = float(self.process_noise_theta) ** 2
        log.append((float(stamp_s), var_prev + rate * dt))

    def _odometry_heading_variance_or(self, stamp_msg, fallback) -> float:
        """The odometry-heading variance to commit in camera_xy_only mode.

        Before the first odometry sample there is no drift origin and
        _map_frame_heading_variance answers the non-informative pi^2; committing that
        would turn every later prediction into metres of cross-track uncertainty, so
        the caller's own variance is kept until odometry has started.
        """
        if getattr(self, '_odom_origin_stamp_s', None) is None:
            return float(fallback)
        return self._map_frame_heading_variance(stamp_msg)

    def _anchor_belief_yaw_for_planning(self, m0, S0, now_msg, *, motion_snapshot=None):
        """Heading anchor.

        Single-camera path (use_pixel_correction): pixel_to_bev already bakes the
        spawn-yaw offset into /state/bev, so leave the belief yaw alone.
        Multicam path (state_correction_ekf / no pixel correction): the manager's
        /state/bev carries no heading, so anchor the belief yaw to the map-frame
        odometry heading (odom_yaw + odom_yaw_offset_rad). Without this the belief
        heading stays in the raw-odom frame and the planner steers ~90 deg off.
        """
        self._heading_anchor_applied = False
        self._state_bev_yaw_ignored = True
        if (
            not self.use_pixel_correction
            and self.heading_update_mode == 'camera_xy_only'
        ):
            h = self._map_frame_heading(now_msg, motion_snapshot=motion_snapshot)
            if h is not None:
                m0 = np.asarray(m0, dtype=float).copy()
                S0 = np.asarray(S0, dtype=float).copy()
                m0[2] = h
                # The returned mean now uses an external odometry heading, not
                # the recursively predicted camera-coupled heading. Transform
                # the covariance to the same model instead of pairing two means
                # with one covariance.
                S0[:2, 2] = 0.0
                S0[2, :2] = 0.0
                S0[2, 2] = self._map_frame_heading_variance(now_msg)
                S0 = self._regularize_state_covariance(S0)
                self._heading_anchor_applied = True
        return m0, S0

    def _inflate_stale_planning_covariance(self, S0, belief_age_s: float):
        """Apply the explicitly configured prediction-only uncertainty penalty."""
        staleness_s = max(float(belief_age_s) - float(self.pixel_timeout_s), 0.0)
        rate = float(self.stale_belief_inflate_m2_per_s)
        cap = float(self.stale_belief_inflate_cap_m2)
        if staleness_s <= 0.0 or rate <= 0.0 or cap <= 0.0:
            return S0
        inflate = min(staleness_s * rate, cap)
        result = np.asarray(S0, dtype=float).copy()
        result[0, 0] += inflate
        result[1, 1] += inflate
        return self._regularize_state_covariance(result)

    def _resolve_pixel_corrected_belief_for_planning(self, now_msg):
        # All operational modes project the same owned record. Initialization
        # and measurement assimilation are callback transactions, never reads.
        m, P, meta = self._resolve_state_belief_ekf(now_msg)
        with self._data_lock:
            pixel_stamp = deepcopy(getattr(self, 'pixel_stamp', None))
        meta['measurement_available'] = self._pixel_measurement_available_for_planning(now_msg, pixel_stamp)
        return m, P, meta

    def _resolve_state_belief_for_planning(self):
        return self._resolve_state_belief_ekf(self.get_clock().now().to_msg())

    def _reanchor_belief_to_correction(self, state_msg, reason=""):
        """Re-anchor to a fused /state/bev correction (message form)."""
        self._reanchor_belief_to_xy(
            state_msg.header.stamp,
            np.array([
                float(state_msg.pose.pose.position.x),
                float(state_msg.pose.pose.position.y),
            ], dtype=float),
            self._state_measurement_cov(state_msg),
            reason=reason,
        )

    def _init_belief_from_task_prior(self, stamp_ns) -> None:
        """Start the belief at the declared task start, at the first accepted odometry stamp.

        The prior is N([x0, y0, theta0], diag(s_xy^2, s_xy^2, s_theta^2)) from the task
        declaration, never ground truth. Stamping it at the first odometry sample gives the
        replay motion support from t0, and every later camera batch is newer, so the first
        camera batch is an ordinary NIS-gated update. It is not a correction: no correction
        event is published.
        """
        if not getattr(self, 'initial_belief_from_task_start', False) or stamp_ns is None:
            return
        with self._data_lock:
            self._ensure_belief_runtime_locked()
            accepted_ns = getattr(self, '_odom_accepted_stamp_ns', None)
            if (self._belief_record is not None or accepted_ns is None
                    or self._observe_belief_clock_locked() is None):
                return
            x0, y0, yaw0 = self.initial_belief_xyyaw
            stamp = self._ns_stamp(int(accepted_ns))
            h = self._map_frame_heading(stamp)
            m = np.array([x0, y0, h if h is not None else wrap_angle(yaw0)], dtype=float)
            s_xy, s_th = self.initial_belief_sigma_xy_m, self.initial_belief_sigma_theta_rad
            P = np.diag([s_xy ** 2, s_xy ** 2, s_th ** 2])
            if h is not None and getattr(self, 'heading_update_mode', 'camera_xy_only') != 'coupled':
                # camera_xy_only: the heading mean is the map-frame odometry heading, so its
                # variance is that heading's drift variance, the same value the read-out
                # reports (_map_frame_heading_variance). Carrying the declared task prior
                # here instead leaves a second, larger heading variance inside the record
                # that prediction turns into cross-track uncertainty.
                P[2, 2] = self._odometry_heading_variance_or(stamp, P[2, 2])
            self._commit_belief(m, self._regularize_state_covariance(P), stamp)

    def _reanchor_belief_to_xy(self, stamp_msg, z_xy, R, reason=""):
        with self._data_lock:
            motion = self._motion_snapshot_locked()
            h = self._map_frame_heading(stamp_msg, motion_snapshot=motion)
            m = np.array([float(z_xy[0]), float(z_xy[1]), h if h is not None else 0.])
            yaw_var = self._map_frame_heading_variance(stamp_msg) if h is not None else NONINFORMATIVE_YAW_VAR
            P = np.diag([0., 0., yaw_var]); P[:2, :2] = np.asarray(R)
            self._commit_belief(m, self._regularize_state_covariance(P), stamp_msg)
            self._last_correction_stamp = deepcopy(stamp_msg)

    def _state_measurement_cov(self, state_msg):
        """Measurement covariance of one fused /state/bev correction.

        Taken as the manager states it. Nothing here inflates it: the commissioned
        covariance is the claim under test, and a bias the cameras repeat is bounded by
        the manager's commissioned bias floor, not by widening R per frame.
        """
        return self._xy_covariance_from_pose(state_msg.pose.covariance)

    def _snapshot_metric_correction_inputs(self, stamp_msg, z_xy):
        with self._data_lock:
            self._ensure_belief_runtime_locked()
            record = self._belief_record
            if record is None:
                return None
            m, P = record.arrays()
            return bc.CorrectionSnapshot(m, P, self._ns_stamp(record.stamp_ns),
                self.last_cmd.copy(), np.asarray(z_xy, dtype=float).reshape(-1)[:2].copy(),
                meas_stamp=deepcopy(stamp_msg), motion_snapshot=self._motion_snapshot_locked(),
                belief_record=record)

    def _apply_state_correction(self, state_msg):
        """Fold one FUSED /state/bev correction into the belief (message form)."""
        self._apply_metric_correction(
            state_msg.header.stamp,
            np.array([
                float(state_msg.pose.pose.position.x),
                float(state_msg.pose.pose.position.y),
            ], dtype=float),
            self._state_measurement_cov(state_msg),
        )

    @_serialized_correction
    def _apply_map_observations(self, observations):
        """Fold PER-CAMERA map observations in sequentially, one filter, N updates.

        The alternative -- collapsing them into a fused pose first -- runs two
        filters in series: ``camera_manager`` seeds from a median with an
        identity prior and no motion model, and the planner then treats that
        output as an independent measurement. Sequential updating into the one
        belief that actually has a prior and a motion model is the textbook form,
        and it lets each camera carry its own measurement covariance and be
        gated (and reason-coded) on its own.

        Applied in TIMESTAMP order, because a Kalman update is only valid against
        a prior predicted to that measurement's stamp; the chain predicts between
        them. Observations not newer than the belief are dropped rather than
        buffered -- with the belief already past them they carry no information
        the filter can use without a smoother.
        """
        ordered = sorted(observations, key=lambda o: (float(o.timestamp_s), str(o.camera_id)))
        if not ordered:
            return
        indexes = {id(obs): i for i, obs in enumerate(ordered)}
        with self._data_lock:
            if self._observe_belief_clock_locked() is None:
                return
            before = self._belief_record
        fresh = []
        for index, obs in enumerate(ordered):
            previous = self._seen_map_observation_stamps.get(str(obs.camera_id), -math.inf)
            if float(obs.timestamp_s) <= previous:
                self._publish_pixel_correction_rejection(
                    self._float_to_stamp(float(obs.timestamp_s)),
                    reason=bc.RejectReason.NOT_NEWER,
                    age=self._stamp_age_s(self._float_to_stamp(float(obs.timestamp_s))),
                    measurement_space=bc.SPACE_MAP_XY, camera_index=float(index),
                )
                continue
            self._seen_map_observation_stamps[str(obs.camera_id)] = float(obs.timestamp_s)
            stamp_msg = self._float_to_stamp(float(obs.timestamp_s))
            if (not self._metric_correction_is_fresh(self._stamp_age_s(stamp_msg))
                    or (before is not None and self._stamp_ns(stamp_msg) < before.stamp_ns)):
                self._publish_pixel_correction_rejection(stamp_msg,
                    reason=(bc.RejectReason.STALE_AGE if not self._metric_correction_is_fresh(
                        self._stamp_age_s(stamp_msg)) else bc.RejectReason.NOT_NEWER),
                    age=self._stamp_age_s(stamp_msg), measurement_space=bc.SPACE_MAP_XY,
                    camera_index=float(index))
                continue
            fresh.append(obs)
        ordered = fresh
        if not ordered:
            return  # Re-delivery is not new information and must not inflate P.

        # Divergence is a claim about the BELIEF, so it needs corroboration. A
        # lone camera reporting metres away is far more likely to be a bad
        # camera than proof the belief is lost -- snapping to it on its own word
        # would hand the belief to the worst observation in the batch. So the
        # per-observation re-anchor is OFF here, and instead a QUORUM of mutually
        # agreeing cameras can re-anchor the belief before the updates run.
        # (The fused path keeps the single-observation guard: that measurement
        # has already been through the manager's NIS + disagreement gates.)
        consumed = self._reanchor_on_camera_quorum(ordered)

        accepted_any = bool(consumed)
        rejected_any = False
        for index, obs in enumerate(ordered):
            if (str(obs.camera_id), float(obs.timestamp_s)) in consumed:
                continue
            outcome = self._apply_metric_correction(
                self._float_to_stamp(float(obs.timestamp_s)),
                np.array([float(obs.xy_m[0]), float(obs.xy_m[1])], dtype=float),
                self._observation_covariance(obs),
                camera_index=float(indexes[id(obs)]),
                label=str(obs.camera_id),
                source_batch_id='camera:' + json.dumps([str(obs.camera_id), round(obs.timestamp_s*1e9)],
                                                      separators=(',', ':')),
                allow_same_stamp=True,
                allow_reanchor=False,
                # Inflate at most once per batch, below, and only if NOTHING
                # anchored the belief: inflating per rejected camera would let a
                # single persistently bad camera balloon the covariance at
                # N x rate until it is itself waved through.
                inflate_on_reject=False,
            )
            accepted_any = accepted_any or bool(outcome is not None and outcome.accepted)
            rejected_any = rejected_any or bool(outcome is not None and not outcome.accepted
                and outcome.reason not in (bc.RejectReason.NOT_NEWER, bc.RejectReason.STALE_AGE))

        if rejected_any and not accepted_any:
            self._inflate_belief_after_rejection(
                f"no camera accepted ({len(ordered)} observation(s))"
            )
        # One terminal batch record includes any final configured inflation.
        # The optional wire has camera/time identities, but no detector event ID.
        event_id = 'per_camera:' + json.dumps([(str(o.camera_id), round(o.timestamp_s*1e9))
                                              for o in ordered], separators=(',', ':'))
        stamp_msg = self._float_to_stamp(max(o.timestamp_s for o in ordered))
        with self._data_lock:
            self._retain_correction_outcome_locked(event_id, stamp_msg,
                'accepted' if accepted_any else 'rejected',
                'camera_quorum' if consumed else 'per_camera_batch', before)
        self._publish_correction_assimilation(source_batch_id=event_id, stamp_msg=stamp_msg,
            status='accepted' if accepted_any else 'rejected', reason='per_camera_batch')

    def _observation_covariance(self, obs):
        """One camera's stated covariance, as it stated it (see _state_measurement_cov)."""
        return np.array(
            [[float(obs.covariance_m2[0][0]), float(obs.covariance_m2[0][1])],
             [float(obs.covariance_m2[1][0]), float(obs.covariance_m2[1][1])]],
            dtype=float,
        )

    def _reanchor_on_camera_quorum(self, ordered):
        """Re-anchor only when several mutually agreeing cameras say the belief is lost."""
        # Multiple frames from one physical camera cannot supply multiple votes.
        ordered = list({str(obs.camera_id): obs for obs in ordered}.values())
        if self.state_reanchor_m <= 0.0 or len(ordered) < 2:
            return set()
        with self._data_lock:
            self._ensure_belief_runtime_locked()
            before = self._belief_record
            belief_m = None if self.belief_m is None else self.belief_m.copy()
        if belief_m is None:
            return set()
        far = [
            obs for obs in ordered
            if math.hypot(float(obs.xy_m[0]) - float(belief_m[0]),
                          float(obs.xy_m[1]) - float(belief_m[1])) > self.state_reanchor_m
        ]
        if len(far) < max(2, (len(ordered) + 1) // 2):
            return set()
        # They must agree with EACH OTHER far better than they disagree with the
        # belief, else this is scattered noise rather than a lost belief.
        spread = max(
            math.hypot(float(a.xy_m[0]) - float(b.xy_m[0]), float(a.xy_m[1]) - float(b.xy_m[1]))
            for a in far for b in far
        )
        if spread >= self.state_reanchor_m:
            return set()
        xs = sorted(float(obs.xy_m[0]) for obs in far)
        ys = sorted(float(obs.xy_m[1]) for obs in far)
        median = np.array([xs[len(xs) // 2], ys[len(ys) // 2]], dtype=float)
        newest = max(far, key=lambda o: float(o.timestamp_s))
        self._reanchor_belief_to_xy(
            self._float_to_stamp(float(newest.timestamp_s)),
            median,
            self._observation_covariance(newest),
            reason=f"{len(far)}/{len(ordered)} cameras agree, spread {spread:.2f} m",
        )
        event_id = 'quorum:' + json.dumps([(str(o.camera_id), round(o.timestamp_s*1e9))
                                           for o in far], separators=(',', ':'))
        with self._data_lock:
            self._retain_correction_outcome_locked(event_id,
                self._float_to_stamp(newest.timestamp_s), 'reanchored', 'camera_quorum', before)
        return {(str(obs.camera_id), float(obs.timestamp_s)) for obs in far}

    def _inflate_belief_after_rejection(self, reason: str):
        inflate = float(self.state_reject_inflate_m2)
        if inflate <= 0:
            return
        with self._data_lock:
            self._ensure_belief_runtime_locked()
            record = self._belief_record
            if record is None:
                return
            m, P = record.arrays()
            P[0, 0] += inflate; P[1, 1] += inflate
            self._commit_belief(m, self._regularize_state_covariance(P), self._ns_stamp(record.stamp_ns),
                                motion_support=record.motion_support)

    def _advance_belief_over_outage(self, stamp_msg, dt_s: float) -> None:
        with self._data_lock:
            self._ensure_belief_runtime_locked()
            record = self._belief_record
            if record is None:
                return
            motion = self._motion_snapshot_locked()
        target_ns = self._stamp_ns(stamp_msg)
        full = plan_replay(motion, record.stamp_ns, target_ns, self.state_max_predict_dt_s)
        supported = full.support.supported
        replayed_s = dt_s if supported else min(dt_s, self.state_max_predict_dt_s)
        start_ns = target_ns - round(replayed_s * 1e9)
        plan = plan_replay(motion, start_ns, target_ns, self.state_max_predict_dt_s)
        m, P = self._predict_belief_to_now(*record.arrays(), self.last_cmd.copy(), replayed_s,
                                          stamp_msg, motion_snapshot=motion)
        gaps = list(plan.support.gaps)
        if start_ns > record.stamp_ns:
            gaps.insert(0, (record.stamp_ns, start_ns, 'replay_cap_unknown_motion'))
            reach = self.v_max * (start_ns-record.stamp_ns)*1e-9
            P = np.asarray(P).copy(); P[0,0] += reach**2; P[1,1] += reach**2
        support = MotionSupport(record.stamp_ns, target_ns, plan.support.source, tuple(gaps))
        self._commit_belief(m, self._regularize_state_covariance(P), stamp_msg,
                            motion_support=support.following(record.motion_support))

    def _publish_correction_assimilation(self, *, source_batch_id, stamp_msg,
                                          status, reason, outcome=None):
        if not source_batch_id:
            return
        with self._data_lock:
            retained = self._correction_outcomes[source_batch_id]
        message = String()
        message.data = json.dumps(self._outcome_json_value(retained), sort_keys=True, allow_nan=False)
        self.correction_assimilation_pub.publish(message)

    @_serialized_correction
    def _apply_metric_correction(self, stamp_msg, z_xy, R, *,
                                 camera_index=math.nan, label="fused", source_batch_id="",
                                 allow_reanchor=True, inflate_on_reject=True, allow_same_stamp=False):
        with self._data_lock:
            now_ns = self._observe_belief_clock_locked()
            before = self._belief_record
            if source_batch_id in self._correction_outcomes:
                return None  # no repeat information, inflation, or terminal reclassification
        stamp_ns = self._stamp_ns(stamp_msg)
        age = self._stamp_age_s(stamp_msg)
        dt_corr = (stamp_ns-before.stamp_ns)*1e-9 if before else 0.

        def finish(status, reason, outcome=None, commit=None):
            with self._data_lock:
                if commit is not None:
                    commit()
                self._retain_correction_outcome_locked(source_batch_id, stamp_msg, status,
                                                       str(reason), before, outcome)
            if status != 'accepted_bootstrap':
                if outcome is not None and outcome.snapshot is not None:
                    self._publish_correction_outcome(stamp_msg, outcome, camera_index=camera_index)
                else:
                    self._publish_pixel_correction_rejection(stamp_msg, reason=reason, age=age,
                        dt_s=dt_corr, measurement_space=bc.SPACE_MAP_XY, camera_index=camera_index,
                        belief_input_stamp_s=before.stamp_ns*1e-9 if before else math.nan)
            self._publish_correction_assimilation(source_batch_id=source_batch_id, stamp_msg=stamp_msg,
                                                 status=status, reason=reason, outcome=outcome)
            return outcome

        def dropped(reason):
            out = bc.CorrectionOutcome(reason=reason, age=age, dt_s=dt_corr,
                                        measurement_space=bc.SPACE_MAP_XY)
            return finish('dropped', reason.value, out)

        if now_ns is None:
            return dropped(bc.RejectReason.DT_IMPLAUSIBLE)
        if not self._metric_correction_is_fresh(age):
            return dropped(bc.RejectReason.STALE_AGE)
        if before is None:
            out = bc.CorrectionOutcome(reason=bc.RejectReason.ACCEPTED, age=age,
                                        measurement_space=bc.SPACE_MAP_XY)
            return finish('accepted_bootstrap', 'bootstrap', out,
                          lambda: self._reanchor_belief_to_xy(stamp_msg, z_xy, R))
        simultaneous = allow_same_stamp and stamp_ns == before.stamp_ns
        if stamp_ns <= before.stamp_ns and not simultaneous:
            return dropped(bc.RejectReason.NOT_NEWER)
        try:
            snapshot = self._snapshot_metric_correction_inputs(stamp_msg, z_xy)
            if snapshot is None:
                return dropped(bc.RejectReason.MISSING_SNAPSHOT)
            validated_replay_dt_s = None
            if dt_corr > self.state_max_predict_dt_s:
                full_plan = plan_replay(
                    snapshot.motion_snapshot,
                    before.stamp_ns,
                    stamp_ns,
                    self.state_max_predict_dt_s,
                )
                full_support = full_plan.support.following(before.motion_support)
                if full_support.supported:
                    validated_replay_dt_s = dt_corr
                else:
                    out = bc.CorrectionOutcome(reason=bc.RejectReason.REPLAY_GAP, age=age,
                                                dt_s=dt_corr, measurement_space=bc.SPACE_MAP_XY)
                    def advance():
                        self._advance_belief_over_outage(stamp_msg, dt_corr)
                        if inflate_on_reject:
                            self._inflate_belief_after_rejection('replay gap')
                    return finish('dropped', out.reason.value, out, advance)
            outcome = bc.apply_correction(
                source=bc.FusedMapMeasurementSource(snapshot_fn=lambda: snapshot,
                                                      measurement_cov_fn=lambda: R),
                gates=self._correction_gates(
                    metric_measurement=True,
                    allow_reanchor=allow_reanchor,
                    validated_replay_dt_s=validated_replay_dt_s,
                ),
                replay=lambda *args: self._replay_cmd_log_interval(
                    *args, motion_snapshot=snapshot.motion_snapshot), age=age, dt_s=dt_corr)
            status = ('accepted' if outcome.accepted else
                      'reanchored' if outcome.recover == bc.RECOVER_REANCHOR else 'rejected')
            return finish(status, outcome.reason.value, outcome,
                lambda: self._commit_metric_correction_outcome(stamp_msg, R, outcome,
                    label=label, inflate_on_reject=inflate_on_reject))
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
            # A failed numerical prediction is not a prediction to hold. The last
            # valid immutable anchor survives and the event is retained before stop.
            with self._data_lock:
                self._retain_correction_outcome_locked(source_batch_id, stamp_msg, 'rejected',
                    self._belief_invalid_reason or 'invalid_prediction', before)
            self._publish_correction_assimilation(source_batch_id=source_batch_id, stamp_msg=stamp_msg,
                status='rejected', reason='invalid_prediction')
            self._fatal_experiment_stop('invalid metric prediction or commit', exc)
            return None

    def _metric_correction_is_fresh(self, age: float) -> bool:
        future_tolerance_s = max(float(self.pixel_timeout_s), 0.25)
        return bool(
            math.isfinite(age)
            and age <= float(self.pixel_timeout_s)
            and age >= -future_tolerance_s
        )

    def _float_to_stamp(self, seconds: float):
        seconds = float(seconds)
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError('invalid measurement timestamp')
        return self._ns_stamp(round(seconds * 1e9))

    def _commit_metric_correction_outcome(self, stamp_msg, R, outcome, *, label="fused",
                                          inflate_on_reject=True):
        with self._data_lock:
            self._ensure_belief_runtime_locked()
        if outcome.accepted:
            m, P = checked_state(outcome.next_m, outcome.next_S)
            if getattr(self, 'heading_update_mode', 'camera_xy_only') != 'coupled':
                h = self._map_frame_heading(stamp_msg,
                    motion_snapshot=outcome.snapshot.motion_snapshot if outcome.snapshot else None)
                m[2] = float(h) if h is not None else float(outcome.m_pred[2])
                P[:2, 2] = 0.; P[2, :2] = 0.
                # One heading model: the committed variance is the odometry-heading
                # variance the read-out reports, not the predicted record variance,
                # which still holds the initial prior (see _init_belief_from_task_prior).
                P[2, 2] = (self._odometry_heading_variance_or(stamp_msg, outcome.S_pred[2, 2])
                           if h is not None else float(outcome.S_pred[2, 2]))
            P = self._regularize_state_covariance(P)
            support_data = outcome.replay_meta.get('motion_support')
            before = outcome.snapshot.belief_record if outcome.snapshot else self._belief_record
            support = (MotionSupport.from_dict(support_data) if support_data else
                       MotionSupport(before.stamp_ns, self._stamp_ns(stamp_msg)))
            record = self._commit_belief(m, P, stamp_msg,
                          motion_support=support.following(before.motion_support if before else None))
            outcome.next_m, outcome.next_S = record.arrays()
            outcome.yaw_info = bc.yaw_report(outcome.next_m, outcome.next_S, outcome.m_pred)
            self._last_correction_stamp = deepcopy(stamp_msg)
            return
        if outcome.recover == bc.RECOVER_REANCHOR:
            self._reanchor_belief_to_xy(stamp_msg, outcome.meas, R)
            return
        if outcome.m_pred is None or outcome.S_pred is None:
            behind = (self._stamp_ns(stamp_msg)-self._belief_record.stamp_ns)*1e-9
            if behind > 0:
                self._advance_belief_over_outage(stamp_msg, behind)
            return
        m, P = checked_state(outcome.m_pred, outcome.S_pred)
        if inflate_on_reject:
            P[0,0] += self.state_reject_inflate_m2; P[1,1] += self.state_reject_inflate_m2
        support_data = outcome.replay_meta.get('motion_support')
        before = outcome.snapshot.belief_record if outcome.snapshot else self._belief_record
        support = (MotionSupport.from_dict(support_data) if support_data else
                   MotionSupport(before.stamp_ns, self._stamp_ns(stamp_msg)))
        self._commit_belief(m, self._regularize_state_covariance(P), stamp_msg,
                            motion_support=support.following(before.motion_support if before else None))

    def _resolve_state_belief_ekf(self, now_msg):
        with self._data_lock:
            self._ensure_belief_runtime_locked()
            state_ref = deepcopy(getattr(self, 'state_msg', None))
            record = self._belief_record
            motion = self._motion_snapshot_locked()
            last_cmd = self.last_cmd.copy()
            goal_revision = self._goal_revision
            invalid = self._belief_invalid_reason
        self._state_bev_yaw_ignored = True
        if record is None:
            return None, None, {'measurement_available': False, 'belief_age_s': math.inf,
                                'belief_valid': False, 'invalid_reason': 'uninitialized'}
        target_ns = self._stamp_ns(now_msg)
        if invalid or target_ns < record.stamp_ns or record.epoch != self._belief_epoch:
            return None, None, {'belief_valid': False,
                                'invalid_reason': invalid or 'future_anchor'}
        age = (target_ns - record.stamp_ns) * 1e-9
        plan = plan_replay(motion, record.stamp_ns, target_ns, self.state_max_predict_dt_s)
        support = plan.support.following(record.motion_support)
        prediction_meta = {}
        m, P = self._predict_belief_to_now(*record.arrays(), last_cmd, age,
                                          now_msg, motion_snapshot=motion, diagnostics=prediction_meta)
        m, P = self._anchor_belief_yaw_for_planning(m, P, now_msg, motion_snapshot=motion)
        P = self._inflate_stale_planning_covariance(P, age)
        snapshot = PredictionSnapshot.create(record, m, P, target_ns, support,
                    valid=support.supported, invalid_reason='' if support.supported else 'unsupported_motion',
                    goal_revision=goal_revision)
        meta = snapshot.planner_meta()
        meta.update(prediction_meta)
        meta['heading_anchor_applied'] = bool(not self.use_pixel_correction
            and self.heading_update_mode == 'camera_xy_only'
            and self._map_frame_heading(now_msg, motion_snapshot=motion) is not None)
        meta.update(measurement_available=bool(state_ref is not None and self._state_msg_is_fresh(state_ref)),
                    belief_age_s=age, belief_stamp=self._ns_stamp(record.stamp_ns))
        return (m, P, meta) if snapshot.valid else (None, None, meta)

    def _resolve_diagnostic_odom_belief_for_planning(self):
        """DIAGNOSTIC: use raw ODOMETRY as the belief, with a near-zero covariance.

        Not ground truth -- the simulator's pose never reaches this node. This exists to
        take the estimator out of the loop so the controller can be looked at on its own,
        and it makes the belief a lie by construction (odometry drifts; the stated
        covariance says it does not). Never true in a comparison run.
        """
        with self._data_lock:
            diagnostic_pose = self.diagnostic_odom_pose
        if diagnostic_pose is None:
            return None, None, {}
        m0 = np.array(
            [diagnostic_pose[0], diagnostic_pose[1], diagnostic_pose[2]], dtype=float)
        S0 = np.diag([1e-4, 1e-4, 1e-4]).astype(float)
        return m0, S0, {
            'measurement_available': True,
            'belief_age_s': 0.0,
            'belief_valid': True,
            'motion_supported': True,
            'belief_epoch': 'diagnostic_odom',
            # Diagnostic odometry is a continuously sampled source rather than
            # a correction transaction. Keep one stable identity so route
            # admission can re-read the latest pose without confusing motion
            # with a concurrent estimator commit.
            'belief_revision': 0,
            'belief_frame_id': self._resolve_plan_frame_id(),
        }

    def _resolve_belief_for_planning(self):
        self._heading_anchor_applied = False
        self._state_bev_yaw_ignored = False
        self._reset_prediction_diagnostics()
        with self._data_lock:
            now_ns = self._observe_belief_clock_locked()
        if now_ns is None:
            return None, None, {'belief_valid': False, 'invalid_reason': self._belief_invalid_reason}
        now_msg = self._ns_stamp(now_ns)
        if self.use_diagnostic_odom_localization:
            m, P, meta = self._resolve_diagnostic_odom_belief_for_planning()
        elif self.use_pixel_correction:
            m, P, meta = self._resolve_pixel_corrected_belief_for_planning(now_msg)
        elif self.state_correction_ekf:
            m, P, meta = self._resolve_state_belief_ekf(now_msg)
        else:
            m, P, meta = self._resolve_state_belief_for_planning()
        if m is None or P is None:
            return None, None, meta
        if not meta.get('belief_valid', True):
            return None, None, meta
        self._latest_measurement_available = bool(meta.get('measurement_available', False))
        self._latest_belief_age_s = float(meta.get('belief_age_s', math.nan))
        return m, P, meta

    def _resolve_plan_frame_id(self):
        with self._data_lock:
            self._ensure_belief_runtime_locked()
            return self._belief_frame_id

    @staticmethod
    def _pose_covariance_from_state_covariance(S):
        pose_cov = [0.0] * 36
        if S is None:
            return pose_cov
        S = np.asarray(S, dtype=float)
        if S.shape[0] < 3 or S.shape[1] < 3:
            return pose_cov
        idx = (0, 1, 5)
        for i_src, i_dst in enumerate(idx):
            for j_src, j_dst in enumerate(idx):
                pose_cov[i_dst * 6 + j_dst] = float(S[i_src, j_src])
        return pose_cov

    def _build_path_message(self, result, goal_xy, *, append_goal=True, frame_id=None, stamp=None):
        path = Path()
        path.header.frame_id = frame_id or self._resolve_plan_frame_id()
        path.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()

        for state in result.states:
            p = PoseStamped()
            p.header = path.header
            p.pose.position.x = float(state[0])
            p.pose.position.y = float(state[1])
            p.pose.orientation.z = math.sin(0.5 * float(state[2]))
            p.pose.orientation.w = math.cos(0.5 * float(state[2]))
            path.poses.append(p)

        if append_goal:
            goal_pose = PoseStamped()
            goal_pose.header = path.header
            goal_pose.pose.position.x = float(goal_xy[0])
            goal_pose.pose.position.y = float(goal_xy[1])
            goal_pose.pose.orientation.w = 1.0
            path.poses.append(goal_pose)
        return path

    # Minimum step of an anchor advance, so the revision does not churn at the tick rate.
    _ANCHOR_ADVANCE_MIN_STEP_S = 0.5

    def _advance_belief_anchor(self):
        """Commit the odometry prediction as the anchor, a fixed lag behind the clock.

        Every belief query replays odometry from the anchor to now. With an anchor that
        moves only on camera corrections, that replay, and the lock it holds, grows with
        the camera gap until the node falls behind its own odometry. The anchor is
        therefore advanced to ``now - 2 * pixel_timeout_s``. Corrections older than
        ``pixel_timeout_s`` are refused, so every admissible correction, including one
        whose processing started up to ``pixel_timeout_s`` before this advance, still
        lies after the anchor. An interval without odometry support is not committed:
        it stays visible to every query as unsupported motion.
        """
        with self._data_lock:
            now_ns = self._observe_belief_clock_locked()
            record = self._belief_record
            if now_ns is None or record is None or record.epoch != self._belief_epoch:
                return
            target_ns = now_ns - round(2.0 * float(self.pixel_timeout_s) * 1e9)
            if target_ns - record.stamp_ns < round(self._ANCHOR_ADVANCE_MIN_STEP_S * 1e9):
                return
            plan = plan_replay(self._motion_snapshot_locked(), record.stamp_ns, target_ns,
                               self.state_max_predict_dt_s)
            support = plan.support.following(record.motion_support)
            if not support.supported:
                return
            m, P, _, _ = self._run_motion_replay(*record.arrays(), plan)
            self._commit_belief(m, P, self._ns_stamp(target_ns), motion_support=support)

    def _belief_publish_tick(self):
        self._advance_belief_anchor()
        with self._data_lock:
            now_ns = self._observe_belief_clock_locked()
            record = self._belief_record
            anchor_objects = (getattr(self, 'belief_m', None), getattr(self, 'belief_S', None))
            motion = self._motion_snapshot_locked()
            goal_revision = self._goal_revision
            reason = self._belief_invalid_reason
            epoch = self._belief_epoch
        if record is None or now_ns is None or (record is not None and now_ns < record.stamp_ns):
            self._publish_invalid_belief(reason or ('uninitialized' if record is None else 'future_anchor'),
                                         expected_record=record, expected_epoch=epoch)
            return
        try:
            plan = plan_replay(motion, record.stamp_ns, now_ns, self.state_max_predict_dt_s)
            support = plan.support.following(record.motion_support)
            m, P = self._predict_belief_to_now(*record.arrays(), self.last_cmd.copy(),
                        (now_ns-record.stamp_ns)*1e-9, self._ns_stamp(now_ns), motion_snapshot=motion)
            # ``camera_xy_only`` defines heading as map-frame odometry heading.
            # Planning already applies that definition in
            # ``_resolve_state_belief_ekf``.  The canonical prediction consumed by
            # the camera manager must expose the same state: otherwise a rapid
            # turn can leave the manager testing the observed hull against a
            # recursively propagated heading that the controller itself does not
            # use.  That inconsistency is fail-closed but self-locking--valid
            # detections are refused precisely when a position correction is
            # needed.  Anchor both the explicit envelope and compatibility pose;
            # x/y and the motion-support evidence are unchanged.
            m, P = self._anchor_belief_yaw_for_planning(
                m, P, self._ns_stamp(now_ns), motion_snapshot=motion)
            snapshot = PredictionSnapshot.create(record, m, P, now_ns, support,
                        valid=support.supported,
                        invalid_reason='' if support.supported else 'unsupported_motion',
                        goal_revision=goal_revision)
        except (ValueError, TypeError, ArithmeticError, np.linalg.LinAlgError) as exc:
            self._publish_invalid_belief('invalid_prediction', expected_record=record,
                                         expected_epoch=epoch)
            self._fatal_experiment_stop('invalid belief publication prediction', exc)
            return
        with self._data_lock:
            if self._observe_belief_clock_locked() is None:
                self._publish_invalid_belief(self._belief_invalid_reason)
                return
            if (self._belief_record is not record or self._belief_epoch != record.epoch
                    or self.belief_m is not anchor_objects[0] or self.belief_S is not anchor_objects[1]):
                return
            key = (record.epoch, now_ns, record.revision)
            last = self._last_belief_publication
            if last is not None and last[0] == key[0] and key[1:] <= last[1:]:
                return
            self._last_belief_publication = key
            publisher = getattr(self, 'belief_state_pub', None)
            if publisher is not None:
                msg = String(); msg.data = json.dumps(snapshot.to_dict(), allow_nan=False)
                publisher.publish(msg)
            if snapshot.valid:
                self.planner_belief_pub.publish(self._build_belief_message(
                    m, P, frame_id=record.frame_id, stamp=self._ns_stamp(now_ns)))

    def _build_belief_message(self, m0, S0, *, frame_id=None, stamp=None):
        belief = PoseWithCovarianceStamped()
        belief.header.frame_id = frame_id or self._resolve_plan_frame_id()
        belief.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
        belief.pose.pose.position.x = float(m0[0])
        belief.pose.pose.position.y = float(m0[1])
        belief.pose.pose.orientation.z = math.sin(0.5 * float(m0[2]))
        belief.pose.pose.orientation.w = math.cos(0.5 * float(m0[2]))
        belief.pose.covariance = self._pose_covariance_from_state_covariance(S0)
        return belief

    def _publish_plan_and_metrics(self, result, goal_xy, m0, S0, *, belief_meta=None):
        frame_id = self._resolve_plan_frame_id()
        stamp = self.get_clock().now().to_msg()
        preview_path = self._build_path_message(
            result, goal_xy, append_goal=False, frame_id=frame_id, stamp=stamp
        )
        # Both retained topics need the same rollout. Construct its ROS poses
        # once; the displayed path only adds the mission-goal marker.
        path = Path()
        path.header = preview_path.header
        path.poses = list(preview_path.poses)
        goal_pose = PoseStamped()
        goal_pose.header = path.header
        goal_pose.pose.position.x = float(goal_xy[0])
        goal_pose.pose.position.y = float(goal_xy[1])
        goal_pose.pose.orientation.w = 1.0
        path.poses.append(goal_pose)
        self.path_pub.publish(path)
        self.plan_preview_pub.publish(preview_path)

        metrics_msg = Float64MultiArray()
        metrics_msg.data = [
            float(result.total_cost),
            float(result.risk_cost),
            float(result.ambiguity_cost),
            float(result.control_cost),
            float(result.obstacle_cost),
            float(getattr(result, 'p_vis_plan', 1.0)),
            float(getattr(result, 'p_vis_plan_eff', 1.0)),
            float(getattr(result, 'r_plan_u_std', np.nan)),
            float(getattr(result, 'r_plan_v_std', np.nan)),
            1.0 if (belief_meta or {}).get('measurement_available', False) else 0.0,
            float((belief_meta or {}).get('belief_age_s', math.nan)),
            float(getattr(result, 'terminal_goal_distance_pred', np.nan)),
            float(getattr(result, 'terminal_goal_progress_m', np.nan)),
            float(getattr(result, 'fraction_horizon_low_pvis', np.nan)),
            float(getattr(result, 'fraction_horizon_high_ambiguity', np.nan)),
            float(getattr(result, 'min_predicted_obstacle_distance_m', np.nan)),
            1.0 if getattr(result, 'rollout_valid', True) else 0.0,
            float(getattr(result, 'risk_mean', np.nan)),
            float(getattr(result, 'risk_cov_trace', np.nan)),
            float(getattr(result, 'risk_cov_logdet', np.nan)),
            float(getattr(result, 'delta_risk_visibility', np.nan)),
            float(getattr(result, 'delta_ambiguity_visibility', np.nan)),
        ]
        self.metrics_pub.publish(metrics_msg)

    def _after_plan_result(self, result):
        """Hook for subclasses (e.g. agent node) to publish extra outputs."""
        return

    def _publish_planner_diagnostics(self, result, plan_elapsed_ms, *, belief_meta=None):
        active_plan_age_s = math.nan
        active_plan_remaining_s = math.nan
        active_control_index = math.nan
        active_controls_len = math.nan
        active_controls_original_len = math.nan
        latency_skip_steps = float(getattr(self, '_last_latency_skip_steps', 0))
        latency_skip_s = float(getattr(self, '_last_latency_skip_s', 0.0))
        command_timer_period_s = float(getattr(self, '_cmd_timer_period_s', math.nan))
        planner_timer_period_s = float(getattr(self, '_plan_period_s', math.nan))
        pending_active_remaining_s = float(
            getattr(self, '_pending_plan_started_active_remaining_s', math.nan)
        )
        if hasattr(self, '_active_controls') and hasattr(self, '_active_plan_started_at'):
            with self._data_lock:
                active_controls = None if self._active_controls is None else self._active_controls
                active_started_at = self._active_plan_started_at
                active_controls_original_len = float(
                    getattr(self, '_active_controls_original_len', 0)
                )
            if active_controls is not None and active_started_at is not None:
                step_dt = max(float(self.dt), 1e-3)
                active_plan_age_s = max(
                    (self.get_clock().now() - active_started_at).nanoseconds * 1e-9,
                    0.0,
                )
                active_controls_len = float(active_controls.shape[0])
                active_control_index = float(
                    min(int(active_plan_age_s / step_dt), active_controls.shape[0] - 1)
                )
                active_plan_remaining_s = max(
                    float(active_controls.shape[0]) * step_dt - active_plan_age_s,
                    0.0,
                )
        diag = Float64MultiArray()
        diag.data = [
            1.0 if getattr(result, 'optimizer_success', False) else 0.0,
            float(getattr(result, 'optimizer_status', 0)),
            float(getattr(result, 'optimizer_nit', 0)),
            float(getattr(result, 'optimizer_nfev', 0)),
            float(plan_elapsed_ms),
            float(getattr(result, 'solve_time_s', 0.0)) * 1000.0,
            float(getattr(result, 'p_vis_plan', 1.0)),
            float(getattr(result, 'p_vis_plan_eff', 1.0)),
            float(getattr(result, 'r_plan_u_std', np.nan)),
            float(getattr(result, 'r_plan_v_std', np.nan)),
            1.0 if (belief_meta or {}).get('measurement_available', False) else 0.0,
            float((belief_meta or {}).get('belief_age_s', math.nan)),
            float(getattr(result, 'terminal_goal_distance_pred', np.nan)),
            float(getattr(result, 'terminal_goal_progress_m', np.nan)),
            float(getattr(result, 'fraction_horizon_low_pvis', np.nan)),
            float(getattr(result, 'fraction_horizon_high_ambiguity', np.nan)),
            float(getattr(result, 'min_predicted_obstacle_distance_m', np.nan)),
            1.0 if getattr(result, 'rollout_valid', True) else 0.0,
            float(getattr(result, 'risk_mean', np.nan)),
            float(getattr(result, 'risk_cov_trace', np.nan)),
            float(getattr(result, 'risk_cov_logdet', np.nan)),
            float(getattr(result, 'delta_risk_visibility', np.nan)),
            float(getattr(result, 'delta_ambiguity_visibility', np.nan)),
            active_plan_age_s,
            active_plan_remaining_s,
            active_control_index,
            active_controls_len,
            active_controls_original_len,
            latency_skip_steps,
            latency_skip_s,
            command_timer_period_s,
            planner_timer_period_s,
            pending_active_remaining_s,
            float((belief_meta or {}).get('prediction_source', math.nan)),
            float((belief_meta or {}).get('prediction_dt_s', math.nan)),
            float((belief_meta or {}).get('u_pred_v', math.nan)),
            float((belief_meta or {}).get('u_pred_omega', math.nan)),
            float((belief_meta or {}).get('Q_theta_theta', math.nan)),
            float((belief_meta or {}).get('odom_delta_theta', math.nan)),
            float((belief_meta or {}).get('cmd_delta_theta', math.nan)),
            float(bool((belief_meta or {}).get('heading_anchor_applied', False))),
            1.0 if self._state_bev_yaw_ignored else 0.0,
        ]
        self.planner_diag_pub.publish(diag)
        diag_parts = [str(getattr(result, 'optimizer_message', '') or '').strip()]
        invalid_reason = str(getattr(result, 'invalid_reason', '') or '').strip()
        if invalid_reason:
            diag_parts.append(f'invalid_reason={invalid_reason}')
        self._publish_planner_status_text(' | '.join(part for part in diag_parts if part))

    def _publish_planner_status_text(self, text):
        """Publish status transitions promptly, with a 1 Hz unchanged heartbeat.

        Numeric diagnostics and per-event assimilation evidence keep their full
        cadence. The heartbeat also supplies late-joining volatile subscribers.
        """
        now_s = float(self.get_clock().now().nanoseconds) * 1e-9
        previous_s = getattr(self, '_last_status_text_stamp_s', -math.inf)
        if (text == getattr(self, '_last_status_text', None)
                and 0.0 <= now_s - previous_s < 1.0):
            return
        message = String()
        message.data = text
        self.planner_diag_text_pub.publish(message)
        self._last_status_text = text
        self._last_status_text_stamp_s = now_s

    def _snapshot_plan_inputs(self):
        with self._data_lock:
            self._ensure_belief_runtime_locked()
            return {
                'goal': deepcopy(self.goal_msg),
                'pixel_stamp': deepcopy(self.pixel_stamp),
                'state': deepcopy(self.state_msg),
                'goal_revision': self._goal_revision,
                'belief_epoch': self._belief_epoch,
            }

    def _validate_plan_frames(self, goal_ref, state_ref) -> tuple[str, str]:
        goal_frame = (goal_ref.header.frame_id or '').strip()
        state_frame = (state_ref.header.frame_id or '').strip() if state_ref is not None else ''
        if goal_frame and state_frame and goal_frame != state_frame:
            self._fatal_experiment_stop(
                "Frame mismatch between /goal_bev and /state/bev "
                f"(goal='{goal_frame}', state='{state_frame}')"
            )
        return goal_frame, state_frame

    def _goal_xy_from_msg(self, goal_ref: PoseStamped):
        return (
            float(goal_ref.pose.position.x),
            float(goal_ref.pose.position.y),
        )

    def _call_planner(self, m0, S0, goal_xy, progress_index, *, plan_start, now_wall):
        if self.debug_runtime and (now_wall - self._last_plan_entry_log) > self.debug_log_period_s:
            self.get_logger().info(
                "Entering planner.plan: "
                f"x0=({m0[0]:.2f},{m0[1]:.2f},{m0[2]:.2f}), "
                f"goal=({goal_xy[0]:.2f},{goal_xy[1]:.2f})"
            )
            self._last_plan_entry_log = now_wall
        try:
            # Unexpected planner failures should abort the run instead of allowing
            # an invalid experiment to continue as if it were valid evidence.
            result = self.planner.plan(
                m0, S0, goal_xy, progress_index=progress_index,
            )
        except Exception as exc:
            self._fatal_experiment_stop("Planner.solve raised an exception", exc)
            return None

        after_plan_wall = time.monotonic()
        if self.debug_runtime and (after_plan_wall - self._last_plan_return_log) > self.debug_log_period_s:
            elapsed_ms = max((time.perf_counter() - plan_start) * 1000.0, 0.0)
            self.get_logger().info(
                "Returned from planner.plan: "
                f"backend={getattr(result, 'backend', 'casadi') if result is not None else 'casadi'}, "
                f"elapsed_ms={elapsed_ms:.1f}, "
                f"success={getattr(result, 'optimizer_success', False) if result is not None else False}"
            )
            self._last_plan_return_log = after_plan_wall
        if result is None:
            self._fatal_experiment_stop("Planner returned no result")
            return None
        return result

    def _publish_plan_result_bundle(self, result, goal_xy, m0, S0, *, belief_meta, plan_elapsed_ms):
        self._publish_plan_and_metrics(result, goal_xy, m0, S0, belief_meta=belief_meta)
        self._after_plan_result(result)
        self._publish_planner_diagnostics(result, plan_elapsed_ms, belief_meta=belief_meta)

    def _warn_on_plan_health(self, result, plan_elapsed_ms, solve_elapsed_ms, *, now_wall):
        if self.debug_runtime and plan_elapsed_ms > (self.slow_plan_factor * self._plan_period_s * 1000.0):
            if now_wall - self._last_slow_plan_log > 2.0:
                self.get_logger().warn(
                    f"Slow plan cycle ({plan_elapsed_ms:.1f} ms, solver={solve_elapsed_ms:.1f} ms, "
                    f"period={self._plan_period_s * 1000.0:.1f} ms, backend={getattr(result, 'backend', 'unknown')})."
                )
                self._last_slow_plan_log = now_wall
        elif (not getattr(result, 'optimizer_success', True)) and (now_wall - self._last_slow_plan_log > 2.0):
            self.get_logger().warn(
                f"Optimizer reported non-success status={getattr(result, 'optimizer_status', 0)} "
                f"message='{getattr(result, 'optimizer_message', '')}'. "
                "Executing the selected solver-returned control sequence."
            )
            self._last_slow_plan_log = now_wall

    def _pixel_age_for_debug(self, pixel_stamp_ref):
        if pixel_stamp_ref is None:
            return None
        try:
            return (self.get_clock().now() - Time.from_msg(pixel_stamp_ref)).nanoseconds * 1e-9
        except (AttributeError, TypeError, ValueError):
            return None

    def _log_plan_debug_once(
        self,
        result,
        m0,
        goal_xy,
        *,
        plan_elapsed_ms,
        solve_elapsed_ms,
        goal_frame,
        state_frame,
        pixel_stamp_ref,
        now_wall,
    ):
        if not (self.debug_runtime and (now_wall - self._last_runtime_log) > self.debug_log_period_s):
            return
        pixel_age = self._pixel_age_for_debug(pixel_stamp_ref)
        self.get_logger().info(
            "Plan debug: "
            f"backend={getattr(result, 'backend', 'unknown')}, "
            f"success={getattr(result, 'optimizer_success', False)}, "
            f"status={getattr(result, 'optimizer_status', 0)}, "
            f"nit={getattr(result, 'optimizer_nit', 0)}, "
            f"nfev={getattr(result, 'optimizer_nfev', 0)}, "
            f"plan_ms={plan_elapsed_ms:.1f}, solve_ms={solve_elapsed_ms:.1f}, "
            f"x0=({m0[0]:.2f},{m0[1]:.2f},{m0[2]:.2f}), "
            f"goal=({goal_xy[0]:.2f},{goal_xy[1]:.2f}), "
            f"frames=({state_frame or 'n/a'}->{goal_frame or 'n/a'}), "
            f"u0=({result.controls[0, 0]:.3f},{result.controls[0, 1]:.3f}), "
            f"J={result.total_cost:.3f}, "
            f"pixel_age={pixel_age if pixel_age is not None else 'n/a'}"
        )
        self._last_runtime_log = now_wall

    def _plan_once(self):
        inputs = self._snapshot_plan_inputs()
        goal_ref = inputs['goal']
        pixel_stamp_ref = inputs['pixel_stamp']
        state_ref = inputs['state']
        if goal_ref is None:
            return

        now_wall = time.monotonic()
        goal_frame, state_frame = self._validate_plan_frames(goal_ref, state_ref)

        m0, S0, belief_meta = self._resolve_belief_for_planning()
        if m0 is None or S0 is None:
            return

        goal_xy = self._goal_xy_from_msg(goal_ref)
        progress_index = self._current_goal_progress_index(m0, goal_xy)

        plan_start = time.perf_counter()
        result = self._call_planner(
            m0, S0, goal_xy, progress_index, plan_start=plan_start, now_wall=now_wall
        )
        if result is None:
            return

        plan_elapsed_ms = max((time.perf_counter() - plan_start) * 1000.0, 0.0)
        solve_elapsed_ms = float(getattr(result, 'solve_time_s', 0.0)) * 1000.0
        self._publish_plan_result_bundle(
            result, goal_xy, m0, S0, belief_meta=belief_meta, plan_elapsed_ms=plan_elapsed_ms
        )
        self._warn_on_plan_health(result, plan_elapsed_ms, solve_elapsed_ms, now_wall=now_wall)
        self._log_plan_debug_once(
            result,
            m0,
            goal_xy,
            plan_elapsed_ms=plan_elapsed_ms,
            solve_elapsed_ms=solve_elapsed_ms,
            goal_frame=goal_frame,
            state_frame=state_frame,
            pixel_stamp_ref=pixel_stamp_ref,
            now_wall=now_wall,
        )
