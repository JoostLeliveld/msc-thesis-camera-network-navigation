"""Base planner classes (pure Python, no ROS)."""

from dataclasses import dataclass
import json
import math
import warnings
import os
import time
import numpy as np

from scipy.optimize import minimize

from planning.core.dynamics import unicycle_step, unicycle_jacobian, unicycle_process_noise
from planning.core.efe_utils import ET1, ET2, UT, ambiguity, risk_components, wrap_angle
from planning.core.nogo_cost import NogoCostConfig, NogoZoneCostModel
from planning.core.visibility_gp_map import GPVisibilityMapConfig, GPVisibilityMapModel
from unav_common.camera_model import ObliqueCameraModel
from planning.core.rollout import rollout_unicycle
from planning.core.plan_validation import (
    validate_controls, validate_covariance,
    validate_plan_result, validate_planning_inputs,
)


@dataclass
class PlanResult:
    controls: np.ndarray
    states: np.ndarray
    total_cost: float
    risk_cost: float
    ambiguity_cost: float
    control_cost: float
    risk_mean: float = 0.0
    risk_cov_trace: float = 0.0
    risk_cov_logdet: float = 0.0
    delta_risk_visibility: float = 0.0
    delta_ambiguity_visibility: float = 0.0
    obstacle_cost: float = 0.0
    backend: str = "unknown"
    optimizer_success: bool = False
    optimizer_status: int = 0
    optimizer_nit: int = 0
    optimizer_nfev: int = 0
    optimizer_message: str = ""
    solve_time_s: float = 0.0
    selected_source: str = ""
    p_vis_plan: float = 1.0
    p_vis_plan_eff: float = 1.0
    r_plan_u_std: float = np.nan
    r_plan_v_std: float = np.nan
    terminal_goal_distance_pred: float = np.nan
    terminal_goal_progress_m: float = np.nan
    fraction_horizon_low_pvis: float = np.nan
    fraction_horizon_high_ambiguity: float = np.nan
    min_predicted_obstacle_distance_m: float = np.nan
    rollout_valid: bool = False
    invalid_reason: str = ""


def extract_waypoints(states, spacing_m=1.0, include_goal=True):
    """Arc-length downsample a plan's xy states into waypoints.

    Drops a waypoint every `spacing_m` of cumulative path length; the final
    state (the plan's terminus / goal) is always included when `include_goal`.
    Returns a list of (x, y). Used to decompose a long-horizon global plan into
    targets for a short-horizon local tracker (planner-derived, not scripted).
    """
    pts = np.asarray(states, dtype=float)[:, :2]
    if pts.shape[0] == 0:
        return []
    spacing = max(float(spacing_m), 1e-3)
    # Interpolate along every segment, carrying residual arc length forward.
    # Selecting only existing global states left gaps up to v_max*global_dt
    # (1 m here), despite a requested 0.20 m execution spacing.
    waypoints = []
    last = pts[0]
    remaining = spacing
    for segment_start, segment_end in zip(pts, pts[1:]):
        cursor = np.asarray(segment_start, dtype=float).copy()
        delta = np.asarray(segment_end, dtype=float) - cursor
        length = float(np.linalg.norm(delta))
        if length <= 1.0e-12:
            last = np.asarray(segment_end, dtype=float)
            continue
        direction = delta / length
        travelled = 0.0
        while length - travelled + 1.0e-12 >= remaining:
            travelled += remaining
            point = cursor + direction * travelled
            waypoints.append((float(point[0]), float(point[1])))
            remaining = spacing
        remaining -= max(length - travelled, 0.0)
        if remaining <= 1.0e-12:
            remaining = spacing
        last = np.asarray(segment_end, dtype=float)
    if include_goal:
        if not waypoints or float(np.linalg.norm(np.asarray(waypoints[-1]) - last)) > 1e-3:
            waypoints.append((float(last[0]), float(last[1])))
    if not waypoints:
        waypoints.append((float(last[0]), float(last[1])))
    return waypoints


class UnicyclePlannerBase:
    """Shared unicycle planner logic. Subclasses define objective specifics."""

    def __init__(
        self,
        *,
        horizon,
        dt,
        v_min,
        v_max,
        w_min,
        w_max,
        control_weight,
        process_noise_xy,
        process_noise_theta,
        goal_sigma_uv,
        risk_weight_obs,
        ambiguity_weight,
        optimizer_maxiter,
        optimizer_gtol,
        optimizer_warm_start,
        optimizer_warm_start_shift_steps=1,
        approx_method=None,
        use_obs_risk=None,
        use_ambiguity=None,
        seed,
        camera_params,
        use_visibility_model=False,
        visibility_target_height_m=0.0,
        visibility_geometry_json='',
        collision_geometry_json='',
        visibility_artifact_path='',
        camera_network_artifact_path='',
        camera_network_expected_sha256='',
        camera_network_expected_source_hashes=None,
        camera_network_camera_ids=None,
        camera_network_active_camera_ids=None,
        camera_network_objective='legacy_pixel_chart',
        # Goal-prior anneal, broad early -> precise late. Without it the mean
        # (goal-distance) part of risk charges (d/sigma*)^2 per step from step
        # zero, which buries the belief terms: measured 1633 vs an ambiguity of
        # ~4 at 20 m from the goal. Annealing from 5.0 m cuts that first step to
        # 8.0, so the early rollout - where the corridor is chosen - is shaped by
        # the belief terms, and the prior still tightens so the route must arrive.
        # Precedent: Meera, Lanillos & Kouw (arXiv 2608.14466) anneal the
        # preference variance as their sole exploration control, tau^2 20 -> 0.6.
        network_goal_std_start_m=5.0,
        # Kouw (IWAI 2024) Lemma 1 ambiguity under the first-order extended
        # transform, evaluated on the availability-weighted commissioned R.
        # This is the thesis method; see docs/PLANNER.md.
        kouw_et1_ambiguity=True,
        # Charge risk once on the terminal belief rather than summing it per
        # step. Integrated risk is a length proxy and buries the visibility
        # signal; goal arrival is already a hard constraint, so risk does not
        # have to pull the robot toward the goal. See CasadiEfeParams and
        # docs/PLANNER.md.
        terminal_risk_only=False,
        network_goal_std_m=0.10,
        camera_network_updates_per_step=1,
        r_visible_uv=2.5,
        r_miss_uv=120.0,
        visibility_sigma_kappa=1.0,
        goal_prior_u_std_start=80.0,
        goal_prior_v_std_start=80.0,
        goal_prior_u_std_final=18.0,
        goal_prior_v_std_final=18.0,
        goal_tightening_power=0.9,
        goal_progress_n_steps=90,
        observation_risk_scale=1.0,
        ambiguity_term_scale=1.00,
        discount_gamma=0.995,
        optimizer_maxfun=500,
        optimizer_ftol=1e-6,
        optimizer_control_block_steps=1,
        optimizer_multistart=False,
        optimizer_multistart_include_direct=True,
        optimizer_initial_routes_json='',
        optimizer_terminal_goal_tolerance_m=0.0,
        use_nogo_cost=False,
        nogo_penalty_type='warning_band',
        nogo_safe_distance=0.0,
        nogo_logbarrier_eps=0.05,
        nogo_warning_band=0.05,
        nogo_near_weight=50.0,
        use_belief_nogo_cost=True,
        nogo_belief_kappa=1.0,
        nogo_mode='keep_out',
        driveable_geometry_json='',
        robot_collision_radius_m=0.125,
        robot_length_m=0.8,
        robot_width_m=0.55,
        use_hit_miss_mixture=False,
        runtime_debug=False,
        # True for the planner that solves the locked EFE objective. The
        # hierarchical local tracker passes False: it builds a planner object
        # for no-go geometry and warm-start seeds but never solves it, so the
        # locked constants do not apply to it. Defaults True so a caller that
        # forgets is still checked. See docs/PLANNER.md.
        enforce_planner_lock=True,
        process_noise_model='encoder',
    ):
        self.enforce_planner_lock = bool(enforce_planner_lock)
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.control_weight = float(control_weight)

        # LOCKED: actuation-noise PSDs from the camera-ready IWAI paper. The
        # unicycle Q_d closed form scales these with speed, heading and dt, so
        # they are the only two free process-noise numbers in the planner. A
        # different value is a method change and must be declared, not inherited
        # silently from a stale config. See docs/PROCESS_NOISE.md.
        _LOCKED_PROCESS_NOISE_XY = 0.02
        _LOCKED_PROCESS_NOISE_THETA = 0.08
        # Every value below was established by measurement on 2026-09-14 and is
        # recorded with its derivation in docs/PLANNER.md. A run that uses
        # any superseded value silently produces a DIFFERENT planner, which has
        # already cost one 80-run campaign. Warn loudly rather than let it pass.
        # Only the planner that actually SOLVES the locked objective is checked.
        # The hierarchical local tracker builds a planner object for its no-go
        # geometry and warm-start seeds but never solves it, and deliberately
        # runs without the belief-aware clearance term; warning about it trains
        # the reader to ignore the warning that matters.
        _enforce_lock = bool(enforce_planner_lock)
        for _name, _value, _locked in () if not _enforce_lock else (
            ('nogo_safe_distance', nogo_safe_distance, 0.0),
            ('nogo_logbarrier_eps', nogo_logbarrier_eps, 0.05),
            ('nogo_warning_band', nogo_warning_band, 0.05),
            ('network_goal_std_m', network_goal_std_m, 0.10),
            ('goal_tightening_power', goal_tightening_power, 0.9),
            ('observation_risk_scale', observation_risk_scale, 1.0),
            ('risk_weight_obs', risk_weight_obs, 1.0),
            ('ambiguity_weight', ambiguity_weight, 1.0),
            ('discount_gamma', discount_gamma, 0.995),
            # Above 1, each block of controls is averaged and repeated, which
            # destroys the seeder's alternating turn/drive steps: the best seed
            # then misses the goal by more than the terminal tolerance, no
            # candidate passes the hard gate, and a parked plan wins on raw
            # cost. This produced a Gazebo run that never moved.
            ('optimizer_control_block_steps', optimizer_control_block_steps, 1),
        ):
            if abs(float(_value) - _locked) > 1e-9:
                warnings.warn(
                    f'{_name}={float(_value)} overrides the locked value {_locked}; '
                    'see docs/PLANNER.md',
                    RuntimeWarning, stacklevel=2)
        if _enforce_lock and not bool(use_belief_nogo_cost):
            warnings.warn(
                'use_belief_nogo_cost is off: the clearance term will not see '
                'predicted belief growth. See docs/PLANNER.md',
                RuntimeWarning, stacklevel=2)
        if _enforce_lock and not bool(kouw_et1_ambiguity):
            warnings.warn(
                'kouw_et1_ambiguity is off: the ambiguity term falls back to a '
                'posterior-entropy sum that charges for route length. '
                'See docs/PLANNER.md',
                RuntimeWarning, stacklevel=2)
        if _enforce_lock and bool(terminal_risk_only):
            warnings.warn(
                'terminal_risk_only is on: the locked method uses normalized '
                'running risk. See docs/PLANNER.md',
                RuntimeWarning, stacklevel=2)
        self.process_noise_model = str(process_noise_model).strip().lower()
        if self.process_noise_model not in ('encoder', 'constant_psd'):
            raise ValueError("process_noise_model must be 'encoder' or 'constant_psd'")
        # 'encoder': Q set from the simulated encoder noise (docs/PROCESS_NOISE.md); the
        # constant process_noise_xy/theta below are then unused and not lock-checked.
        from planning.core.encoder_noise_model import ENCODER_PSD
        self.process_noise_psd = dict(ENCODER_PSD) if self.process_noise_model == 'encoder' else None
        _constant_q = self.process_noise_psd is None
        if _enforce_lock and _constant_q and abs(float(process_noise_xy) - _LOCKED_PROCESS_NOISE_XY) > 1e-9:
            warnings.warn(
                f'process_noise_xy={float(process_noise_xy)} overrides the locked '
                f'value {_LOCKED_PROCESS_NOISE_XY}; see docs/PROCESS_NOISE.md',
                RuntimeWarning, stacklevel=2)
        if _enforce_lock and _constant_q and abs(float(process_noise_theta) - _LOCKED_PROCESS_NOISE_THETA) > 1e-9:
            warnings.warn(
                f'process_noise_theta={float(process_noise_theta)} overrides the locked '
                f'value {_LOCKED_PROCESS_NOISE_THETA}; see docs/PROCESS_NOISE.md',
                RuntimeWarning, stacklevel=2)
        self.process_noise_xy = float(process_noise_xy)
        self.process_noise_theta = float(process_noise_theta)

        self.goal_sigma_uv = float(goal_sigma_uv)
        if self.goal_sigma_uv <= 0.0:
            raise ValueError("goal_sigma_uv must be > 0.0; do not use 0.0 as a sentinel")

        self.risk_weight_obs = float(risk_weight_obs)
        self.ambiguity_weight = float(ambiguity_weight)
        self.r_visible_uv = float(r_visible_uv)
        self.r_miss_uv = float(r_miss_uv)
        self.visibility_sigma_kappa = float(max(visibility_sigma_kappa, 1e-6))
        self.goal_prior_u_std_start = float(goal_prior_u_std_start)
        self.goal_prior_v_std_start = float(goal_prior_v_std_start)
        self.goal_prior_u_std_final = float(goal_prior_u_std_final)
        self.goal_prior_v_std_final = float(goal_prior_v_std_final)
        self.goal_tightening_power = float(max(goal_tightening_power, 1e-6))
        self.goal_progress_n_steps = int(max(goal_progress_n_steps, 1))
        self.observation_risk_scale = float(observation_risk_scale)
        self.ambiguity_term_scale = float(ambiguity_term_scale)
        self.discount_gamma = float(discount_gamma)
        self.robot_collision_radius_m = float(max(robot_collision_radius_m, 0.0))
        self.robot_length_m = float(robot_length_m)
        self.robot_width_m = float(robot_width_m)

        if approx_method is None:
            self.approx_method = 'ET1'
        else:
            self.approx_method = str(approx_method).upper()
        if self.approx_method not in ('ET1', 'ET2'):
            raise ValueError("approx_method must be 'ET1' or 'ET2'")

        if use_obs_risk is None:
            self.use_obs_risk = True
        else:
            self.use_obs_risk = bool(use_obs_risk)

        if use_ambiguity is None:
            self.use_ambiguity = True
        else:
            self.use_ambiguity = bool(use_ambiguity)

        self.optimizer_maxiter = int(optimizer_maxiter)
        self.optimizer_gtol = float(optimizer_gtol)
        self.optimizer_warm_start = bool(optimizer_warm_start)
        self.optimizer_warm_start_shift_steps = int(max(optimizer_warm_start_shift_steps, 1))
        self.optimizer_maxfun = int(max(optimizer_maxfun, 1))
        self.optimizer_ftol = float(max(optimizer_ftol, 1e-12))
        if (isinstance(optimizer_control_block_steps, bool)
                or int(optimizer_control_block_steps) != optimizer_control_block_steps
                or int(optimizer_control_block_steps) < 1):
            raise ValueError('optimizer_control_block_steps must be a positive integer')
        self.optimizer_control_block_steps = int(optimizer_control_block_steps)
        self.optimizer_multistart = self._as_bool_like(optimizer_multistart)
        self.optimizer_multistart_include_direct = self._as_bool_like(
            optimizer_multistart_include_direct
        )
        self.optimizer_initial_routes = self._parse_initial_routes(optimizer_initial_routes_json)
        self.optimizer_terminal_goal_tolerance_m = float(
            max(optimizer_terminal_goal_tolerance_m, 0.0)
        )
        self.rng = np.random.default_rng(int(seed))

        self.camera = ObliqueCameraModel(
            cam_pos=camera_params['cam_pos'],
            look_at=camera_params['look_at'],
            img_width=camera_params['img_width'],
            img_height=camera_params['img_height'],
            fov_h_rad=camera_params['fov_h_rad'],
        )

        self.runtime_debug = bool(runtime_debug)
        self.use_visibility_model = bool(use_visibility_model)
        self._visibility_min_prob = 1e-4
        self.visibility_model = None
        self.camera_network = None
        self.camera_network_objective = str(camera_network_objective or '').strip().lower()
        self.network_goal_std_start_m = (
            None if network_goal_std_start_m is None
            else float(network_goal_std_start_m))
        self.kouw_et1_ambiguity = bool(kouw_et1_ambiguity)
        self.terminal_risk_only = bool(terminal_risk_only)
        if self.camera_network_objective not in ('legacy_pixel_chart', 'metric_expected_belief'):
            raise ValueError(
                'camera_network_objective must be legacy_pixel_chart or metric_expected_belief'
            )
        self.network_goal_std_m = float(network_goal_std_m)
        if not np.isfinite(self.network_goal_std_m) or self.network_goal_std_m <= 0.:
            raise ValueError('network_goal_std_m must be finite and positive')
        if (isinstance(camera_network_updates_per_step, bool)
                or int(camera_network_updates_per_step) != camera_network_updates_per_step
                or int(camera_network_updates_per_step) < 1):
            raise ValueError('camera_network_updates_per_step must be a positive integer')
        self.camera_network_updates_per_step = int(camera_network_updates_per_step)
        network_path = str(camera_network_artifact_path or '').strip()
        if network_path:
            if not self.use_visibility_model:
                raise ValueError('camera network requires use_visibility_model=True')
            if str(visibility_artifact_path or '').strip():
                raise ValueError('choose a camera-network artifact or a single-camera field, not both')
            if use_hit_miss_mixture:
                raise ValueError('IWAI network score proxy is not the hit/miss measurement model')
            from planning.core.camera_network import CameraNetworkModel
            expected_sources = camera_network_expected_source_hashes
            if isinstance(expected_sources, str):
                expected_sources = json.loads(expected_sources) if expected_sources.strip() else None
            expected_ids = camera_network_camera_ids
            if isinstance(expected_ids, str):
                expected_ids = tuple(v.strip() for v in expected_ids.split(',') if v.strip())
            active_ids = camera_network_active_camera_ids
            if isinstance(active_ids, str):
                active_ids = tuple(v.strip() for v in active_ids.split(',') if v.strip())
            if active_ids is not None and not active_ids:
                active_ids = None
            self.camera_network = CameraNetworkModel(
                network_path, cameras=active_ids,
                expected_sha256=camera_network_expected_sha256 or None,
                expected_source_hashes=expected_sources, expected_camera_ids=expected_ids)
        elif self.camera_network_objective != 'legacy_pixel_chart':
            raise ValueError('metric_expected_belief requires a camera-network artifact')

        from unav_common.navigation_parameters import validate_navigation_parameters
        # Reject invalid tuning before the legacy clamps can silently change it.
        validate_navigation_parameters({
            'nogo_safe_distance': nogo_safe_distance,
            'nogo_logbarrier_eps': nogo_logbarrier_eps,
            'nogo_warning_band': nogo_warning_band, 'nogo_near_weight': nogo_near_weight,
            'nogo_belief_kappa': nogo_belief_kappa,
        })
        self.use_nogo_cost = bool(use_nogo_cost)
        self.nogo_penalty_type = str(nogo_penalty_type or 'warning_band').strip().lower()
        self.nogo_safe_distance = float(max(nogo_safe_distance, 0.0))
        self.nogo_logbarrier_eps = float(max(nogo_logbarrier_eps, 1e-6))
        self.nogo_warning_band = float(max(nogo_warning_band, 1e-6))
        self.nogo_near_weight = float(max(nogo_near_weight, 0.0))
        self.use_belief_nogo_cost = bool(use_belief_nogo_cost)
        # Hit/miss expected-belief mixture in the CasADi EFE objective. DEFAULT OFF:
        # off, the planner runs the frozen precision-blend path bit-for-bit (see
        # tests/planning/test_efe_hit_miss_mixture.py). On, detection availability
        # is modelled as Bernoulli instead of being laundered into R_plan, and
        # r_miss_uv is not read by the objective. NumPy selection evaluates the
        # same mixture; observation_model_with_visibility remains separate.
        self.use_hit_miss_mixture = bool(use_hit_miss_mixture)
        self.nogo_belief_kappa = float(max(nogo_belief_kappa, 1e-6))
        self.nogo_mode = str(nogo_mode or 'keep_out').strip().lower()
        self.driveable_geometry_json = str(driveable_geometry_json or '')
        # Numerical integration density for state-only obstacle costs. At the
        # maximum speed, adjacent samples are no more than 0.20 m apart.
        self.obstacle_substeps = max(
            1, int(math.ceil(self.v_max * self.dt / 0.20)))
        self.nogo_cost_model = None
        self.collision_cost_model = None

        self.g_obs = self.camera.g_uv

        self.R_visible = np.diag([
            self.r_visible_uv ** 2,
            self.r_visible_uv ** 2,
        ])
        self.R_miss = np.diag([
            self.r_miss_uv ** 2,
            self.r_miss_uv ** 2,
        ])
        self.R = self.R_visible.copy()

        if self.use_visibility_model and self.camera_network is None:
            vis_cfg = GPVisibilityMapConfig(
                artifact_path=str(visibility_artifact_path or ''),
                camera_pos=tuple(np.asarray(camera_params['cam_pos'], dtype=float).tolist()),
                target_height_m=float(visibility_target_height_m),
                min_prob=self._visibility_min_prob,
            )
            self.visibility_model = GPVisibilityMapModel(vis_cfg)

        if self.use_nogo_cost:
            # keep_in: penalise leaving the driveable lane union (safe_distance =
            # soft edge margin). keep_out: penalise proximity to occluder prisms.
            if self.nogo_mode == 'keep_in':
                nogo_geometry = self.driveable_geometry_json
            else:
                nogo_geometry = str(visibility_geometry_json or '')
            nogo_cfg = NogoCostConfig(
                penalty_type=self.nogo_penalty_type,
                safe_distance=self.nogo_safe_distance,
                logbarrier_eps=self.nogo_logbarrier_eps,
                warning_band=self.nogo_warning_band,
                near_weight=self.nogo_near_weight,
                # Give the cost the real body so it distinguishes driving
                # aligned with an aisle from crossing it at an angle.
                robot_half_length=0.5 * float(self.robot_length_m),
                robot_half_width=0.5 * float(self.robot_width_m),
                body_margin=0.0,
                geometry_json=nogo_geometry,
                mode=self.nogo_mode,
            )
            self.nogo_cost_model = NogoZoneCostModel(nogo_cfg)

        if str(collision_geometry_json or '').strip():
            # The exact swept rectangular-footprint check is the hard validity
            # gate.  Give L-BFGS-B a differentiable centre-distance surrogate as
            # well, otherwise it repeatedly converges to routes that merely pass
            # the zero-clearance gate and fail the release's 0.10 m body margin.
            collision_cfg = NogoCostConfig(
                penalty_type='warning_band',
                safe_distance=0.0,
                logbarrier_eps=self.nogo_logbarrier_eps,
                warning_band=self.nogo_warning_band,
                near_weight=self.nogo_near_weight,
                robot_half_length=0.5 * float(self.robot_length_m),
                robot_half_width=0.5 * float(self.robot_width_m),
                body_margin=0.0,
                geometry_json=str(collision_geometry_json or ''),
                mode='keep_out',
            )
            self.collision_cost_model = NogoZoneCostModel(collision_cfg)

        from unav_common.rectangular_footprint import RectangularFootprint
        self._footprint_collision_model = RectangularFootprint(
            self.collision_cost_model.prisms if self.collision_cost_model is not None else (),
            self.robot_length_m, self.robot_width_m)
        self._footprint_driveable_model = (
            RectangularFootprint(self.nogo_cost_model.prisms,
                                 self.robot_length_m, self.robot_width_m, keep_in=self.nogo_mode == 'keep_in')
            if self.nogo_cost_model is not None else None)
        self.prev_controls_flat = None
        self._prev_goal_xy = None
        self._casadi_valgrad_cache = {}

    @staticmethod
    def _as_bool_like(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in ('1', 'true', 't', 'yes', 'y', 'on')

    def _terminal_goal_feasible(self, diagnostics):
        """Return whether a rollout satisfies the optional terminal task gate."""
        if self.optimizer_terminal_goal_tolerance_m <= 0.0:
            return True
        distance = float(diagnostics.get('terminal_goal_distance_pred', math.inf))
        return bool(
            np.isfinite(distance)
            and distance <= self.optimizer_terminal_goal_tolerance_m
        )

    @property
    def control_bounds(self):
        return ((self.v_min, self.v_max), (self.w_min, self.w_max))

    def validate_trajectory_geometry(self, states, controls):
        """Check the current footprint model over each declared motion segment."""
        states = np.asarray(states, dtype=float)
        controls = np.asarray(controls, dtype=float)
        if (states.shape != (len(controls)+1, 3) or not np.isfinite(states).all()
                or controls.ndim != 2 or controls.shape[1] != 2 or not np.isfinite(controls).all()):
            return False, 'invalid_geometry_trajectory'
        for index, pose in enumerate(states):
            clearances = (self.collision_clearance_state_np(pose),
                          self.driveable_clearance_state_np(pose))
            if any(np.isnan(v) or v < 0 for v in clearances):
                return False, f'footprint_violation_at_state_{index}'
        from unav_common.rectangular_footprint import constant_twist_pose
        physical_state = states[0].copy()
        for index, (start, end, control) in enumerate(zip(states[:-1], states[1:], controls)):
            yaw_delta = float(control[1])*self.dt
            physical_end = constant_twist_pose(physical_state, control, self.dt)
            clearances = (
                self.collision_sweep_clearance_np(start,end,yaw_delta=yaw_delta,control=control,dt=self.dt),
                self.driveable_sweep_clearance_np(start,end,yaw_delta=yaw_delta,control=control,dt=self.dt),
                self.collision_sweep_clearance_np(physical_state,physical_end,yaw_delta=yaw_delta,control=control,dt=self.dt),
                self.driveable_sweep_clearance_np(physical_state,physical_end,yaw_delta=yaw_delta,control=control,dt=self.dt),
            )
            if any(np.isnan(v) or v < 0 for v in clearances):
                return False, f'footprint_violation_on_segment_{index}'
            physical_state = physical_end
        return True, ''

    def validate_result(self, result, m0, S0, goal_xy, *, require_complete=False):
        """Authoritative numerical/dynamics/geometry contract for this planner."""
        return validate_plan_result(
            result, initial_state=m0, initial_covariance=S0, goal_xy=goal_xy,
            dt=self.dt, control_bounds=self.control_bounds, expected_horizon=self.horizon,
            require_complete=require_complete,
            terminal_tolerance_m=self.optimizer_terminal_goal_tolerance_m,
            geometry_validator=self.validate_trajectory_geometry,
        )

    def _prefer_candidate(
        self,
        *,
        candidate_valid,
        candidate_goal_feasible,
        candidate_cost,
        incumbent_valid,
        incumbent_goal_feasible,
        incumbent_cost,
    ):
        """Lexicographic multistart choice: safety, task feasibility, then EFE."""
        if not np.isfinite(candidate_cost):
            return False
        if not np.isfinite(incumbent_cost):
            return True
        if bool(candidate_valid) != bool(incumbent_valid):
            return bool(candidate_valid)
        if (
            self.optimizer_terminal_goal_tolerance_m > 0.0
            and bool(candidate_goal_feasible) != bool(incumbent_goal_feasible)
        ):
            return bool(candidate_goal_feasible)
        return float(candidate_cost) < float(incumbent_cost)

    @staticmethod
    def _parse_initial_routes(raw):
        if raw is None:
            return []
        if isinstance(raw, (list, tuple)):
            payload = raw
        else:
            text = str(raw).strip()
            if not text:
                return []
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                return []
        routes = []
        if not isinstance(payload, (list, tuple)):
            return routes
        for idx, route in enumerate(payload):
            if not isinstance(route, dict):
                continue
            name = str(route.get('name', f'route_{idx}')).strip() or f'route_{idx}'
            waypoints_raw = route.get('waypoints', [])
            waypoints = []
            if isinstance(waypoints_raw, (list, tuple)):
                for wp in waypoints_raw:
                    try:
                        arr = np.asarray(wp, dtype=float).reshape(-1)
                    except (TypeError, ValueError):
                        continue
                    if arr.size >= 2 and np.all(np.isfinite(arr[:2])):
                        waypoints.append((float(arr[0]), float(arr[1])))
            if waypoints:
                routes.append({'name': name, 'waypoints': waypoints})
        return routes

    def _runtime_debug_print(self, message):
        if not self.runtime_debug:
            return
        try:
            print(message, flush=True)
        except (BrokenPipeError, OSError):
            # Launch wrappers can close stdout while planner work is still running.
            pass

    def process_noise(self, dt=None, theta=None, v=None, w=None):
        step_dt = self.dt if dt is None else float(dt)
        return unicycle_process_noise(
            self.process_noise_xy, self.process_noise_theta, step_dt,
            theta=theta, v=v, base_dt=self.dt,
            coherent_drift=getattr(self, 'coherent_drift', False),
            w=w, psd=getattr(self, 'process_noise_psd', None),
        )

    def predict(self, m, S, u, dt=None):
        step_dt = self.dt if dt is None else float(dt)
        m_next = unicycle_step(m, u, step_dt)
        F = unicycle_jacobian(m, u, step_dt)
        Q = self.process_noise(step_dt, theta=float(m[2]), v=float(u[0]), w=float(u[1]))
        S_next = F @ S @ F.T + Q
        return m_next, S_next

    def _approx_fn_for_method(self, method):
        m = str(method).upper()
        if m == 'ET1':
            return ET1
        if m == 'ET2':
            return ET2
        if m == 'UT':
            return UT
        raise RuntimeError(f"Unknown observation approximation method: {method}")

    def approx_observation(self, m, S, method=None, R_override=None):
        approx_method = self.approx_method if method is None else method
        fn = self._approx_fn_for_method(approx_method)
        R_use = self.R if R_override is None else np.asarray(R_override, dtype=float)
        return fn(m, S, self.g_obs, addmatrix=R_use, forceHermitian=True)

    @staticmethod
    def _expected_state_posterior_covariance(S, Sigma_y, Gamma):
        S = np.asarray(S, dtype=float)
        Sigma_y = np.asarray(Sigma_y, dtype=float)
        Gamma = np.asarray(Gamma, dtype=float)
        S = 0.5 * (S + S.T)
        Sigma_y = 0.5 * (Sigma_y + Sigma_y.T)
        try:
            update = Gamma @ np.linalg.solve(Sigma_y + 1e-9 * np.eye(2), Gamma.T)
        except np.linalg.LinAlgError:
            update = Gamma @ np.linalg.pinv(Sigma_y + 1e-9 * np.eye(2)) @ Gamma.T
        S_post = S - update
        return 0.5 * (S_post + S_post.T)

    def visibility_probability(self, m):
        if (not self.use_visibility_model) or (self.visibility_model is None):
            return 1.0
        try:
            p = float(self.visibility_model.prob_state_np(m))
        except RuntimeError:
            # Long-horizon line searches can briefly sample outside the fitted
            # GP grid. Treat that as conservative low visibility so candidate
            # scoring remains defined; physical feasibility is handled by the
            # driveable-region barrier.
            p = float(self._visibility_min_prob)
        return float(np.clip(p, self._visibility_min_prob, 1.0 - self._visibility_min_prob))

    def visibility_probability_belief(self, m, S):
        if (not self.use_visibility_model) or (self.visibility_model is None):
            return 1.0
        mean_xy = np.asarray(m[:2], dtype=float).reshape(2)
        cov_xy = np.asarray(S[:2, :2], dtype=float).reshape(2, 2)
        cov_xy = 0.5 * (cov_xy + cov_xy.T)
        kappa = max(float(self.visibility_sigma_kappa), 1e-6)
        scale = np.sqrt(2.0 + kappa)
        chol = np.linalg.cholesky(cov_xy + 1e-9 * np.eye(2))
        spread = scale * chol
        sigma_points = np.vstack([
            mean_xy,
            mean_xy + spread[:, 0],
            mean_xy - spread[:, 0],
            mean_xy + spread[:, 1],
            mean_xy - spread[:, 1],
        ])
        weights = np.array(
            [kappa / (2.0 + kappa)] + [1.0 / (2.0 * (2.0 + kappa))] * 4,
            dtype=float,
        )
        samples = np.column_stack([sigma_points[:, 0], sigma_points[:, 1], np.full(5, float(m[2]), dtype=float)])
        raw_probs = []
        for sample in samples:
            try:
                raw_probs.append(float(self.visibility_model.prob_state_np(sample)))
            except RuntimeError:
                raw_probs.append(float(self._visibility_min_prob))
        probs = np.clip(
            np.asarray(raw_probs, dtype=float),
            self._visibility_min_prob,
            1.0 - self._visibility_min_prob,
        )
        return float(np.clip(np.sum(weights * probs), self._visibility_min_prob, 1.0 - self._visibility_min_prob))

    @staticmethod
    def _smoothstep(x):
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    @staticmethod
    def _softplus(x):
        x = float(x)
        if x > 40.0:
            return x
        if x < -40.0:
            return math.exp(x)
        return math.log1p(math.exp(x))

    def _visibility_effective_score(self, p_vis):
        return float(np.clip(p_vis, self._visibility_min_prob, 1.0 - self._visibility_min_prob))

    def _blend_observation_covariance(self, trust):
        trust = float(np.clip(trust, self._visibility_min_prob, 1.0 - self._visibility_min_prob))
        # Precision Blending: var = 1 / (trust/var_vis + (1-trust)/var_miss)
        # This makes R much more responsive to visibility improvements.
        visible_prec = 1.0 / np.maximum(np.diag(self.R_visible), 1e-6)
        miss_prec = 1.0 / np.maximum(np.diag(self.R_miss), 1e-6)
        blended_prec = trust * visible_prec + (1.0 - trust) * miss_prec
        plan_var = 1.0 / np.maximum(blended_prec, 1e-9)
        return np.diag(plan_var).astype(float)

    def goal_obs_cov_for_progress(self, progress):
        if (self.camera_network is not None
                and self.camera_network_objective == 'metric_expected_belief'):
            start = self.network_goal_std_start_m
            if start is None or start == self.network_goal_std_m:
                return np.eye(2, dtype=float) * self.network_goal_std_m**2
            fast = float(np.clip(progress, 0.0, 1.0)) ** self.goal_tightening_power
            a = self._smoothstep(fast)
            sigma = (1.0 - a) * start + a * self.network_goal_std_m
            return np.eye(2, dtype=float) * sigma**2
        progress_fast = float(np.clip(progress, 0.0, 1.0)) ** self.goal_tightening_power
        a = self._smoothstep(progress_fast)
        sigma_u = (1.0 - a) * self.goal_prior_u_std_start + a * self.goal_prior_u_std_final
        sigma_v = (1.0 - a) * self.goal_prior_v_std_start + a * self.goal_prior_v_std_final
        return np.diag([sigma_u ** 2, sigma_v ** 2]).astype(float)

    def planning_visibility_diagnostics(self, m, S):
        # NOTE:
        # This function is planner-facing only.
        # It defines predictive observability/trust for route evaluation.
        # It must not be used as the sole source of measurement-update trust.
        if self.camera_network is not None:
            if self.camera_network_objective == 'metric_expected_belief':
                query = self.camera_network.query_belief(
                    m, S, self.visibility_sigma_kappa,
                )
                posterior, _entropy = self.camera_network.expected_belief(
                    m, S, self.visibility_sigma_kappa,
                )
                if self.camera_network.direct_information:
                    information = np.asarray(query['expected_information'], dtype=float)
                    support_key = (
                        'residual_support'
                        if 'residual_support' in query else 'opportunity_support'
                    )
                    return {
                        'p_vis': math.nan,
                        'p_vis_eff': math.nan,
                        'R_plan': np.asarray(posterior[:2, :2], dtype=float),
                        'r_plan_u_std': float(np.sqrt(posterior[0, 0])),
                        'r_plan_v_std': float(np.sqrt(posterior[1, 1])),
                        'expected_information_trace': float(
                            np.trace(information.sum(axis=0))),
                        support_key: np.asarray(query[support_key], dtype=float),
                        'p_vis_semantics': 'not_defined_for_direct_expected_information',
                        'R_plan_semantics': 'information_approximation_posterior_xy_covariance_m2',
                    }
                return {
                    'p_vis': float(np.mean(query['availability'])),
                    'p_vis_eff': float(np.mean(query['availability'])),
                    'R_plan': np.asarray(posterior[:2, :2], dtype=float),
                    'r_plan_u_std': float(np.sqrt(posterior[0, 0])),
                    'r_plan_v_std': float(np.sqrt(posterior[1, 1])),
                    'p_vis_semantics': 'mean_usable_detection_probability',
                    'R_plan_semantics': 'expected_posterior_xy_covariance_m2',
                }
            return self.camera_network.planning_diagnostics(
                m, S, self.camera.H, self.visibility_sigma_kappa)
        p_vis = self.visibility_probability_belief(m, S)
        p_vis_eff = self._visibility_effective_score(p_vis)
        if (not self.use_visibility_model) or (self.visibility_model is None):
            p_vis = 1.0
            p_vis_eff = 1.0
        R_plan = self._blend_observation_covariance(p_vis_eff)
        return {
            'p_vis': float(p_vis),
            'p_vis_eff': float(p_vis_eff),
            'R_plan': np.asarray(R_plan, dtype=float),
            'r_plan_u_std': float(np.sqrt(max(R_plan[0, 0], 0.0))),
            'r_plan_v_std': float(np.sqrt(max(R_plan[1, 1], 0.0))),
        }

    def obstacle_penalty(self, m, S=None):
        penalty = 0.0
        if self.nogo_cost_model is not None and self.nogo_cost_model.enabled:
            if self.use_belief_nogo_cost and S is not None:
                penalty += float(self.nogo_cost_model.penalty_belief_np(
                    m,
                    S,
                    kappa=self.nogo_belief_kappa,
                ))
            else:
                penalty += float(self.nogo_cost_model.penalty_state_np(m))
        if self.collision_cost_model is not None and self.collision_cost_model.enabled:
            penalty += float(self.collision_cost_model.penalty_state_np(m))
        return float(penalty)

    def collision_signed_distance_state_np(self, m):
        if self.collision_cost_model is None:
            return float('inf')
        return float(self.collision_cost_model.signed_distance_state_np(m))

    def collision_clearance_state_np(self, m):
        return self._footprint_collision_model.clearance(m)

    def collision_sweep_clearance_np(self, start, end, *, yaw_delta=None, control=None,
                                     dt=None, required_clearance=0.0):
        return self._footprint_collision_model.sweep_clearance(
            start, end, yaw_delta=yaw_delta, control=control, dt=dt,
            required_clearance=required_clearance)

    def driveable_clearance_state_np(self, m):
        if self._footprint_driveable_model is not None:
            return self._footprint_driveable_model.clearance(m)
        if self.nogo_cost_model is not None and self.nogo_cost_model.enabled:
            return self.nogo_cost_model.clearance_state_np(m)
        return math.inf

    def driveable_sweep_clearance_np(self, start, end, *, yaw_delta=None, control=None,
                                     dt=None, required_clearance=0.0):
        if self._footprint_driveable_model is not None:
            return self._footprint_driveable_model.sweep_clearance(
                start, end, yaw_delta=yaw_delta, control=control, dt=dt,
                required_clearance=required_clearance)
        return math.inf

    def collision_penetration_state_np(self, m):
        clearance = self.collision_clearance_state_np(m)
        if not math.isfinite(clearance):
            return 0.0
        return float(max(-clearance, 0.0))

    def _goal_distance_xy(self, state_xy, goal_xy):
        state_xy = np.asarray(state_xy, dtype=float).reshape(2)
        goal_xy = np.asarray(goal_xy, dtype=float).reshape(2)
        return float(np.linalg.norm(state_xy - goal_xy))

    @staticmethod
    def _trajectory_retraces_lane(states, *, midpoint_tolerance_m=0.15):
        """Reject non-adjacent segments that traverse one lane in reverse.

        Perpendicular crossings and nearby parallel aisles are allowed. A pair
        is retracing only when its segment midpoints coincide and its travel
        directions are nearly opposite.
        """
        xy = np.asarray(states, dtype=float)[:, :2]
        delta = np.diff(xy, axis=0)
        length = np.linalg.norm(delta, axis=1)
        moving = np.flatnonzero(length > 0.05)
        midpoint = 0.5 * (xy[:-1] + xy[1:])
        direction = np.zeros_like(delta)
        direction[moving] = delta[moving] / length[moving, None]
        for offset, first in enumerate(moving):
            for second in moving[offset + 1:]:
                if np.linalg.norm(midpoint[first] - midpoint[second]) > midpoint_tolerance_m:
                    continue
                if float(direction[first] @ direction[second]) < -0.95:
                    return True
        return False

    def _trajectory_plan_diagnostics(self, m0, S0, controls, goal_xy):
        m0, S0, goal_xy = validate_planning_inputs(m0, S0, goal_xy)
        goal_xy = np.asarray(goal_xy, dtype=float).reshape(2)
        # This private geometry diagnostic also probes trajectories outside the
        # actuator envelope. Public evaluation, selection and result admission
        # enforce bounds independently before relying on these diagnostics.
        controls = np.asarray(controls, dtype=float).reshape(self.horizon,2)
        if not np.isfinite(controls).all():
            raise ValueError('nonfinite diagnostic controls')
        m = np.asarray(m0, dtype=float).copy()
        S = np.asarray(S0, dtype=float).copy()
        p_vis_values = []
        ambiguity_std_values = []
        min_collision_clearance = float('inf')
        min_nogo_clearance = float('inf')
        # keep_in: raw mean inside-distance (positive = mean is inside the lane union),
        # WITHOUT safe_distance / belief-tube. Used for the hard validity gate so that a
        # feasible in-lane route is not rejected merely for grazing the soft standoff band.
        min_nogo_mean_inside = float('inf')
        min_driveable_body_clearance = float('inf')

        from unav_common.rectangular_footprint import constant_twist_pose
        physical_state = np.asarray(m0, dtype=float).copy()
        for u in controls:
            m_prev = np.asarray(m, dtype=float).copy()
            m, S = self.predict(m, S, u)
            validate_covariance(S, name='predicted covariance')
            sweep_collision = self.collision_sweep_clearance_np(m_prev, m, yaw_delta=float(u[1])*self.dt, control=u, dt=self.dt)
            if np.isnan(sweep_collision):
                raise ValueError('nonfinite collision sweep geometry')
            min_collision_clearance = min(min_collision_clearance, sweep_collision)
            min_driveable_body_clearance = min(
                min_driveable_body_clearance,
                self.driveable_sweep_clearance_np(m_prev, m, yaw_delta=float(u[1])*self.dt, control=u, dt=self.dt))
            physical_end = constant_twist_pose(physical_state, u, self.dt)
            min_collision_clearance = min(min_collision_clearance,
                self.collision_sweep_clearance_np(physical_state, physical_end,
                    yaw_delta=float(u[1])*self.dt, control=u, dt=self.dt))
            min_driveable_body_clearance = min(min_driveable_body_clearance,
                self.driveable_sweep_clearance_np(physical_state, physical_end,
                    yaw_delta=float(u[1])*self.dt, control=u, dt=self.dt))
            physical_state = physical_end
            vis_diag = self.planning_visibility_diagnostics(m, S)
            p_vis_values.append(float(vis_diag['p_vis']))
            ambiguity_std_values.append(
                float(max(vis_diag['r_plan_u_std'], vis_diag['r_plan_v_std']))
            )

            # Validate not only the discrete rollout states, but also the
            # straight segment between successive states. Without this, a plan
            # can "corner cut" through forbidden floor between two valid samples
            # and still be selected.
            seg_len = float(np.linalg.norm(np.asarray(m[:2], dtype=float) - np.asarray(m_prev[:2], dtype=float)))
            n_seg = max(1, int(math.ceil(seg_len / 0.05)))
            for alpha in np.linspace(1.0 / n_seg, 1.0, n_seg):
                m_seg = (1.0 - float(alpha)) * m_prev + float(alpha) * m
                min_collision_clearance = min(
                    min_collision_clearance,
                    self.collision_clearance_state_np(m_seg),
                )
                if self.nogo_cost_model is not None and self.nogo_cost_model.enabled:
                    nogo_clearance = self.nogo_cost_model.clearance_state_np(m_seg)
                    min_nogo_clearance = min(min_nogo_clearance, float(nogo_clearance))
                    if self.nogo_mode == 'keep_in':
                        min_nogo_mean_inside = min(
                            min_nogo_mean_inside,
                            float(self.nogo_cost_model.signed_distance_state_np(m_seg)),
                        )

        current_goal_distance = self._goal_distance_xy(np.asarray(m0[:2], dtype=float), goal_xy)
        terminal_goal_distance = self._goal_distance_xy(np.asarray(m[:2], dtype=float), goal_xy)
        terminal_goal_progress = float(current_goal_distance - terminal_goal_distance)
        low_pvis_fraction = (
            float(np.mean(np.asarray(p_vis_values, dtype=float) < 0.2))
            if p_vis_values else math.nan
        )
        ambiguity_threshold = math.sqrt(max(self.r_visible_uv, 1e-6) * max(self.r_miss_uv, 1e-6))
        high_ambiguity_fraction = (
            float(np.mean(np.asarray(ambiguity_std_values, dtype=float) >= ambiguity_threshold))
            if ambiguity_std_values else math.nan
        )
        min_clearance = min(min_collision_clearance, min_nogo_clearance)
        # Hard validity uses the same swept rectangular footprint as execution.
        # Soft centre/belief no-go costs remain diagnostics and preferences.
        collision_ok = min_collision_clearance >= 0.0
        nogo_ok = min_driveable_body_clearance >= 0.0
        states = rollout_unicycle(m0, controls, self.dt)
        geometry_valid, geometry_reason = self.validate_trajectory_geometry(states, controls)
        retraces_lane = self._trajectory_retraces_lane(states)
        rollout_valid = bool(collision_ok and nogo_ok and geometry_valid and not retraces_lane)
        invalid_reason = ''
        if not rollout_valid:
            invalid_reason = ('predicted_route_retracing' if retraces_lane else geometry_reason) or (
                'predicted_collision_geometry'
                if not collision_ok
                else 'predicted_driveable_region_violation'
            )
        return {
            'terminal_goal_distance_pred': float(terminal_goal_distance),
            'terminal_goal_progress_m': float(terminal_goal_progress),
            'fraction_horizon_low_pvis': low_pvis_fraction,
            'fraction_horizon_high_ambiguity': high_ambiguity_fraction,
            'min_predicted_obstacle_distance_m': (
                float(min_clearance)
            ),
            'rollout_valid': rollout_valid,
            'invalid_reason': invalid_reason,
        }

    def _resolve_plan_problem(self, m0, goal_xy):
        goal_theta = 0.0
        goal_state = self._goal_state(goal_xy, goal_theta)

        use_observation_risk = self.use_obs_risk
        use_ambiguity_term = self.use_ambiguity

        goal_obs = None
        goal_obs_cov = None
        if use_observation_risk or use_ambiguity_term:
            goal_obs = self._goal_obs(goal_state)
            goal_obs_cov = self._goal_obs_cov()

        return (
            goal_state,
            goal_obs,
            goal_obs_cov,
            use_observation_risk,
            use_ambiguity_term,
        )

    def evaluate_rollout_controls(self, m0, S0, goal_xy, controls, *, progress_index=0.0):
        """Evaluate one fixed control rollout using the same accounting as planner selection."""
        m0, S0, goal_xy = validate_planning_inputs(m0, S0, goal_xy)
        if not np.isfinite(progress_index):
            raise ValueError('goal progress index must be finite')
        (
            goal_state,
            goal_obs,
            goal_obs_cov,
            _use_observation_risk,
            _use_ambiguity_term,
        ) = self._resolve_plan_problem(m0, goal_xy)
        controls = validate_controls(np.asarray(controls, dtype=float).reshape(self.horizon, 2),
                                     control_bounds=self.control_bounds, expected_horizon=self.horizon)
        total_cost, metrics = self._evaluate_controls(
            controls.reshape(-1),
            np.asarray(m0, dtype=float),
            np.asarray(S0, dtype=float),
            goal_state,
            goal_obs,
            goal_obs_cov,
            True,
            progress_index=float(max(progress_index, 0.0)),
        )
        states = rollout_unicycle(np.asarray(m0, dtype=float), controls, self.dt)
        plan_diag = self._trajectory_plan_diagnostics(m0, S0, controls, goal_xy)

        p_vis_values = []
        p_vis_eff_values = []
        r_u_values = []
        r_v_values = []
        m = np.asarray(m0, dtype=float).copy()
        S = np.asarray(S0, dtype=float).copy()
        for u in controls:
            m, S = self.predict(m, S, u)
            vis_diag = self.planning_visibility_diagnostics(m, S)
            p_vis_values.append(float(vis_diag['p_vis']))
            p_vis_eff_values.append(float(vis_diag['p_vis_eff']))
            r_u_values.append(float(vis_diag['r_plan_u_std']))
            r_v_values.append(float(vis_diag['r_plan_v_std']))

        return {
            'controls': controls,
            'states': states,
            'total_cost': float(total_cost),
            'risk_cost': float(metrics.get('risk_cost', math.nan)),
            'ambiguity_cost': float(metrics.get('ambiguity_cost', math.nan)),
            'control_cost': float(metrics.get('control_cost', math.nan)),
            'obstacle_cost': float(metrics.get('obstacle_cost', 0.0)),
            'risk_mean': float(metrics.get('risk_mean', 0.0)),
            'risk_cov_trace': float(metrics.get('risk_cov_trace', 0.0)),
            'risk_cov_logdet': float(metrics.get('risk_cov_logdet', 0.0)),
            'delta_risk_visibility': float(metrics.get('delta_risk_visibility', 0.0)),
            'delta_ambiguity_visibility': float(metrics.get('delta_ambiguity_visibility', 0.0)),
            'terminal_goal_distance_pred': float(plan_diag['terminal_goal_distance_pred']),
            'terminal_goal_progress_m': float(plan_diag['terminal_goal_progress_m']),
            'fraction_horizon_low_pvis': float(plan_diag['fraction_horizon_low_pvis']),
            'fraction_horizon_high_ambiguity': float(plan_diag['fraction_horizon_high_ambiguity']),
            'min_predicted_obstacle_distance_m': float(plan_diag['min_predicted_obstacle_distance_m']),
            'rollout_valid': bool(plan_diag['rollout_valid']),
            'invalid_reason': str(plan_diag['invalid_reason']),
            'mean_p_vis_plan': float(np.mean(p_vis_values)) if p_vis_values else math.nan,
            'mean_p_vis_plan_eff': float(np.mean(p_vis_eff_values)) if p_vis_eff_values else math.nan,
            'mean_r_plan_u_std': float(np.mean(r_u_values)) if r_u_values else math.nan,
            'mean_r_plan_v_std': float(np.mean(r_v_values)) if r_v_values else math.nan,
        }

    def _nominal_controls_flat(self):
        return np.zeros(self.horizon * 2, dtype=float)

    def _shift_controls_flat(self, controls_flat, shift_steps=None):
        controls = np.asarray(controls_flat, dtype=float).reshape(self.horizon, 2)
        shift = int(self.optimizer_warm_start_shift_steps if shift_steps is None else shift_steps)
        shift = max(1, shift)
        if self.horizon <= 1:
            return controls.reshape(-1).copy()
        if shift >= self.horizon:
            shifted = np.repeat(controls[-1:, :], self.horizon, axis=0)
        else:
            shifted = np.empty_like(controls)
            shifted[:-shift, :] = controls[shift:, :]
            shifted[-shift:, :] = controls[-1, :]
        return shifted.reshape(-1)

    def _initial_controls_flat(self):
        if self.optimizer_warm_start and self.prev_controls_flat is not None:
            try:
                controls = validate_controls(np.asarray(self.prev_controls_flat).reshape(self.horizon,2),
                                             control_bounds=self.control_bounds, expected_horizon=self.horizon)
                return self._shift_controls_flat(controls.reshape(-1))
            except (ValueError, TypeError):
                self.prev_controls_flat = None
        return self._nominal_controls_flat()

    def _controls_for_waypoints(self, start_xy_yaw, waypoints):
        """Crude unicycle route seed used only as optimizer initialization."""
        controls = np.zeros((self.horizon, 2), dtype=float)
        wps = [np.asarray(wp, dtype=float).reshape(2) for wp in waypoints]
        if not wps:
            return controls.reshape(-1)
        # Drive the polyline EXACTLY: at each step either rotate toward the
        # next waypoint or advance along the current bearing, never both, and
        # never overshoot. Two defects made the previous seed unusable at the
        # global layer, where one step is v_max*dt = 1 m:
        #   - it switched target once within one step of a corner, so it began
        #     turning a metre early and the swept body left the lane;
        #   - it pivoted in place whenever the heading error exceeded a gate,
        #     and a 0.80x0.55 m body sweeping its circumscribed radius does not
        #     fit in a 1.10-1.30 m aisle.
        # Landing on each waypoint removes both: the seed turns only where the
        # polyline turns, and each rotation happens at a point the map declares
        # driveable. This is a route-candidate parameterisation for the
        # optimiser to start from, not the runtime controller.
        m = np.asarray(start_xy_yaw, dtype=float).reshape(-1)[:3].copy()
        S_dummy = np.eye(3, dtype=float) * 1e-6
        step = 0
        for target in wps:
            target = np.asarray(target, dtype=float).reshape(2)
            while step < self.horizon:
                delta = target - m[:2]
                distance = float(np.linalg.norm(delta))
                if distance <= 1.0e-8:
                    break
                yaw_error = wrap_angle(
                    math.atan2(float(delta[1]), float(delta[0])) - float(m[2]))
                if abs(yaw_error) > 1.0e-8:
                    command = np.array([
                        0.0,
                        float(np.clip(yaw_error / max(self.dt, 1e-6),
                                      self.w_min, self.w_max)),
                    ], dtype=float)
                else:
                    command = np.array([
                        float(np.clip(min(self.v_max, distance / max(self.dt, 1e-6)),
                                      self.v_min, self.v_max)),
                        0.0,
                    ], dtype=float)
                controls[step] = command
                m, _ = self.predict(m, S_dummy, command)
                step += 1
            if step >= self.horizon:
                break
        return controls.reshape(-1)

    @staticmethod
    def _densify_waypoints(start_xy, waypoints, *, spacing):
        """Return the same polyline with no leg longer than ``spacing``."""
        spacing = float(max(spacing, 1e-3))
        points = [np.asarray(start_xy, dtype=float).reshape(2)]
        points += [np.asarray(wp, dtype=float).reshape(2) for wp in waypoints]
        dense = []
        for first, second in zip(points[:-1], points[1:]):
            leg = float(np.linalg.norm(second - first))
            steps = max(1, int(math.ceil(leg / spacing)))
            for index in range(1, steps + 1):
                dense.append(first + (second - first) * (index / steps))
        return dense or [np.asarray(wp, dtype=float).reshape(2) for wp in waypoints]

    def _build_multistart_candidates(self, m0, goal_xy):
        """Build optional optimizer seeds; these are not mission waypoints."""
        candidates: list[tuple[str, np.ndarray]] = []
        if not self.optimizer_multistart:
            return candidates

        start = np.asarray(m0, dtype=float).reshape(-1)[:3]
        goal = np.asarray(goal_xy, dtype=float).reshape(2)

        if self.optimizer_warm_start and self.prev_controls_flat is not None:
            candidates.append(('cold', self._nominal_controls_flat()))

        if self.optimizer_multistart_include_direct:
            candidates.append(('direct_goal', self._controls_for_waypoints(start, [goal])))

        for route in self.optimizer_initial_routes:
            name = str(route.get('name', 'route'))
            waypoints = list(route.get('waypoints', []))
            if waypoints:
                candidates.append((
                    f'route:{name}',
                    self._controls_for_waypoints(start, waypoints),
                ))

        return candidates

    def _objective_scales(self, controls_flat, m0, S0, goal_state, goal_obs, goal_obs_cov):
        del controls_flat, m0, S0, goal_state, goal_obs, goal_obs_cov
        return 1.0, 1.0

    @staticmethod
    def _scaled_objective_from_metrics(metrics, objective_scales):
        risk_scale, ambiguity_scale = objective_scales
        risk_term = float(metrics.get('risk_cost', 0.0)) / float(max(risk_scale, 1e-9))
        ambiguity_term = float(metrics.get('ambiguity_cost', 0.0)) / float(max(ambiguity_scale, 1e-9))
        return (
            risk_term
            + ambiguity_term
            + float(metrics.get('control_cost', 0.0))
            + float(metrics.get('obstacle_cost', 0.0))
        )

    def _evaluate_candidate_controls(
        self,
        controls_flat,
        m0,
        S0,
        goal_state,
        goal_obs,
        goal_obs_cov,
        objective_scales,
        *,
        progress_index=0.0,
    ):
        controls_flat = np.asarray(controls_flat, dtype=float).reshape(self.horizon * 2)
        controls = controls_flat.reshape(self.horizon, 2)
        total_cost, metrics = self._evaluate_controls(
            controls_flat,
            m0,
            S0,
            goal_state,
            goal_obs,
            goal_obs_cov,
            True,
            progress_index=progress_index,
        )
        scaled_total = self._scaled_objective_from_metrics(metrics, objective_scales)
        return {
            'controls_flat': controls_flat,
            'total_cost': float(total_cost),
            'metrics': {str(k): float(v) for k, v in metrics.items()},
            'scaled_total': float(scaled_total),
        }

    def _checked_candidate_controls(self, controls_flat, m0, S0, goal_state,
                                    goal_obs, goal_obs_cov, objective_scales, *, progress_index):
        """Reject invalid numbers before they can become a selection incumbent."""
        controls = validate_controls(np.asarray(controls_flat, dtype=float).reshape(self.horizon,2),
                                     control_bounds=self.control_bounds, expected_horizon=self.horizon)
        candidate = self._evaluate_candidate_controls(
            controls.reshape(-1), m0, S0, goal_state, goal_obs, goal_obs_cov,
            objective_scales, progress_index=progress_index)
        values = [candidate['total_cost'], candidate['scaled_total'], *candidate['metrics'].values()]
        if not np.isfinite(values).all():
            raise ValueError('nonfinite candidate objective or components')
        diagnostics = self._trajectory_plan_diagnostics(m0,S0,controls,goal_state[:2])
        candidate['controls_flat'] = controls.reshape(-1).copy()
        return candidate, diagnostics

    def _autodiff_cache_key(
        self,
        goal_state,
        goal_obs,
        use_observation_risk,
        use_ambiguity_term,
    ):
        del goal_state
        return (
            self.approx_method,
            bool(use_ambiguity_term),
            bool(use_observation_risk),
            bool(self.use_visibility_model),
            float(self.control_weight),
            float(self.risk_weight_obs),
            float(self.ambiguity_weight),
            float(self.r_visible_uv),
            float(self.r_miss_uv),
            float(self.visibility_sigma_kappa),
            float(self.goal_prior_u_std_start),
            float(self.goal_prior_v_std_start),
            float(self.goal_prior_u_std_final),
            float(self.goal_prior_v_std_final),
            float(self.goal_tightening_power),
            int(self.goal_progress_n_steps),
            float(self.observation_risk_scale),
            float(self.ambiguity_term_scale),
            float(self.discount_gamma),
            float(self.robot_collision_radius_m),
            float(self.robot_length_m), float(self.robot_width_m),
            bool(self.use_nogo_cost),
            bool(self.use_belief_nogo_cost),
            bool(self.use_hit_miss_mixture),
            self.camera_network_objective,
            float(self.network_goal_std_m),
            float(self.network_goal_std_start_m)
            if self.network_goal_std_start_m is not None else -1.0,
            bool(self.kouw_et1_ambiguity),
            bool(self.terminal_risk_only),
            float(self.optimizer_terminal_goal_tolerance_m),
            int(self.camera_network_updates_per_step),
            int(self.optimizer_control_block_steps),
            int(self.obstacle_substeps),
            float(self.nogo_belief_kappa),
            self._geometry_cache_identity(self.nogo_cost_model),
            self._geometry_cache_identity(self.collision_cost_model),
            int(self.horizon),
            float(self.dt),
            int(np.asarray(goal_obs, dtype=float).shape[0]),
            tuple(self.visibility_model.signature) if self.visibility_model is not None else (),
            tuple(self.camera_network.signature) if self.camera_network is not None else (),
            float(self.process_noise_xy),
            float(self.process_noise_theta),
            tuple(sorted((getattr(self, 'process_noise_psd', None) or {}).items())),
            tuple(np.asarray(self.camera.H, dtype=float).reshape(-1)),
            tuple(np.asarray(self.R_visible, dtype=float).reshape(-1)),
            tuple(np.asarray(self.R_miss, dtype=float).reshape(-1)),
            os.environ.get('UNAV_CASADI_JIT', '0') == '1',
        )

    @staticmethod
    def _geometry_cache_identity(model):
        if model is None:
            return ()
        # Model.signature is a rounded human diagnostic; it is not an exact
        # identity for constants frozen into a symbolic graph.
        settings = tuple(getattr(model,name,None) for name in (
            'mode','penalty_type','safe_distance','logbarrier_eps',
            'warning_band','near_weight'))
        prisms = tuple(tuple(float(getattr(p,name)) for name in
                       ('xmin','xmax','ymin','ymax','zmin','zmax')) for p in model.prisms)
        arrays = tuple(tuple(np.asarray(getattr(model,name,()),dtype=float).reshape(-1))
                       for name in ('_xmins','_xmaxs','_ymins','_ymaxs','union_boundary_segments'))
        return settings,prisms,arrays

    def _get_casadi_valgrad(
        self,
        goal_state,
        goal_obs,
        *,
        use_observation_risk,
        use_ambiguity_term,
    ):
        from planning.core import casadi_efe

        if getattr(self, 'coherent_drift', False):
            raise ValueError('coherent drift has no matching symbolic planner dynamics')

        if not casadi_efe.casadi_available():
            raise RuntimeError("CasADi is not available")

        if goal_obs is None:
            goal_obs = self._goal_obs(goal_state)

        cache_key = self._autodiff_cache_key(
            goal_state,
            goal_obs,
            use_observation_risk,
            use_ambiguity_term,
        )
        valgrad = self._casadi_valgrad_cache.get(cache_key)
        self._runtime_debug_print(
            "[planner_debug] CasADi valgrad cache "
            f"{'hit' if valgrad is not None else 'miss'} "
            f"(horizon={self.horizon}, approx={self.approx_method})"
        )
        if valgrad is None:
            build_start = time.perf_counter()
            from planning.core.casadi_cache import FunctionCache, compile_function, function_key
            cache_dir = os.environ.get('UNAV_CASADI_CACHE_DIR', '').strip()
            jit = os.environ.get('UNAV_CASADI_JIT', '0') == '1'
            disk_cache = FunctionCache(cache_dir) if cache_dir else None
            disk_key = function_key(cache_key, jit=jit) if disk_cache else None
            cached = disk_cache.load(disk_key) if disk_cache else None
            if cached is not None:
                valgrad = casadi_efe._make_valgrad_wrapper(cached)
                valgrad.cache_info = dict(status='hit', key=disk_key, jit=jit,
                                          prepare_s=time.perf_counter() - build_start)
                self._casadi_valgrad_cache[cache_key] = valgrad
                return valgrad
            p_vis_ca = None
            if self.use_visibility_model and self.visibility_model is not None:
                p_vis_ca = self.visibility_model.make_prob_state_casadi()
            nogo_cost_ca = None
            nogo_belief_cost_ca = None
            state_cost_terms = []
            if self.nogo_cost_model is not None and self.nogo_cost_model.enabled:
                if self.use_belief_nogo_cost:
                    nogo_belief_cost_ca = self.nogo_cost_model.make_penalty_belief_casadi(
                        kappa=self.nogo_belief_kappa,
                    )
                else:
                    state_cost_terms.append(
                        self.nogo_cost_model.make_penalty_state_casadi())
            if self.collision_cost_model is not None and self.collision_cost_model.enabled:
                state_cost_terms.append(
                    self.collision_cost_model.make_penalty_state_casadi())
            if state_cost_terms:
                def nogo_cost_ca(m):
                    return sum(term(m) for term in state_cost_terms)
            params_ca = casadi_efe.CasadiEfeParams(
                # No static Q: the EFE loop rebuilds the exact Q_d(theta, v, dt) per step
                # from process_noise_xy/theta (see unicycle_process_noise_ca).
                R_visible=np.array(self.R_visible, dtype=float),
                R_miss=np.array(self.R_miss, dtype=float),
                control_weight=float(self.control_weight),
                risk_scale=float(self.risk_weight_obs * self.observation_risk_scale if use_observation_risk else 0.0),
                ambiguity_scale=float(self.ambiguity_weight * self.ambiguity_term_scale if use_ambiguity_term else 0.0),
                discount_gamma=float(self.discount_gamma),
                process_noise_xy=float(self.process_noise_xy),
                process_noise_theta=float(self.process_noise_theta),
                process_noise_psd=getattr(self, 'process_noise_psd', None),
                visibility_sigma_kappa=float(self.visibility_sigma_kappa),
                goal_prior_u_std_start=float(self.goal_prior_u_std_start),
                goal_prior_v_std_start=float(self.goal_prior_v_std_start),
                goal_prior_u_std_final=float(self.goal_prior_u_std_final),
                goal_prior_v_std_final=float(self.goal_prior_v_std_final),
                goal_tightening_power=float(self.goal_tightening_power),
                goal_progress_n_steps=int(self.goal_progress_n_steps),
                use_belief_nogo_cost=bool(self.use_belief_nogo_cost),
                time_horizon=int(self.horizon),
                dt=float(self.dt),
                Du=2,
                use_hit_miss_mixture=bool(self.use_hit_miss_mixture),
                terminal_risk_only=bool(self.terminal_risk_only),
                # R_cond / obs_bias stay None: no conditional-covariance field or
                # bias has been measured yet, so casadi_efe falls back to
                # R_visible and zero bias (documented in _r_cond_expr). Wire the
                # measured values here, not by editing casadi_efe.
                R_cond=None,
                obs_bias=None,
            )
            if (self.camera_network is not None
                    and self.camera_network_objective == 'metric_expected_belief'):
                valgrad = casadi_efe.make_metric_network_efe_valgrad_fn(
                    params_ca,
                    self.camera_network.make_expected_belief_casadi(
                        self.visibility_sigma_kappa,
                        opportunities=self.camera_network_updates_per_step,
                    ),
                    arrival_radius_m=float(self.optimizer_terminal_goal_tolerance_m),
                    effective_covariance=(
                        self.camera_network.make_effective_covariance_casadi(
                            self.visibility_sigma_kappa)
                        if self.kouw_et1_ambiguity else None),
                    reference_covariance=self.reference_observation_covariance(),
                    goal_std_m=self.network_goal_std_m,
                    goal_std_start_m=self.network_goal_std_start_m,
                    nogo_cost=nogo_cost_ca,
                    nogo_belief_cost=nogo_belief_cost_ca,
                    obstacle_substeps=self.obstacle_substeps,
                    control_block_steps=self.optimizer_control_block_steps,
                )
            else:
                valgrad = casadi_efe.make_efe_valgrad_fn(
                    params_ca,
                    self.camera.H,
                    approx=self.approx_method,
                    p_vis_state=p_vis_ca,
                    nogo_cost=nogo_cost_ca,
                    nogo_belief_cost=nogo_belief_cost_ca,
                    R_plan_state=(None if self.camera_network is None else
                        self.camera_network.make_proxy_covariance_casadi(
                            self.camera.H, self.visibility_sigma_kappa)),
                )
            if jit:
                # An unsupported compiler/backend must be explicit; never quietly
                # run a different execution profile than the campaign recorded.
                valgrad = casadi_efe._make_valgrad_wrapper(compile_function(valgrad.casadi_function))
            if disk_cache:
                disk_cache.save(disk_key, valgrad.casadi_function)
            valgrad.cache_info = dict(status='miss' if disk_cache else 'disabled', key=disk_key,
                                      jit=jit, prepare_s=time.perf_counter() - build_start)
            self._casadi_valgrad_cache[cache_key] = valgrad
            self._runtime_debug_print(
                "[planner_debug] CasADi valgrad function prepared in "
                f"{(time.perf_counter() - build_start) * 1000.0:.1f} ms"
            )

        return valgrad

    def observation_model_with_visibility(self, m_pred, S_pred):
        """Visibility-aware measurement shaping used by objective and correction."""
        if self.camera_network is not None:
            raise RuntimeError('camera-network planning proxy is not a fresh measurement; use metric map updates')
        S_pred = np.asarray(S_pred, dtype=float)
        if (not self.use_visibility_model) or (self.visibility_model is None):
            R_eff = np.asarray(self.R_visible, dtype=float)
            S_eff = S_pred.copy()
            return 1.0, R_eff, S_eff, 1.0

        diag = self.planning_visibility_diagnostics(m_pred, S_pred)
        R_eff = np.asarray(diag['R_plan'], dtype=float)
        S_eff = S_pred.copy()
        return float(diag['p_vis']), R_eff, S_eff, 1.0

    def _goal_state(self, goal_xy, theta):
        return np.array([goal_xy[0], goal_xy[1], theta], dtype=float)

    def _goal_obs(self, goal_state):
        if (self.camera_network is not None
                and self.camera_network_objective == 'metric_expected_belief'):
            return np.asarray(goal_state[:2], dtype=float)
        return np.asarray(self.g_obs(goal_state), dtype=float)

    def _goal_obs_cov(self):
        return self.goal_obs_cov_for_progress(0.0)

    def _mixture_stage_metrics(self, m, S, p_use, goal_obs, goal_cov):
        """NumPy counterpart of the existing optional symbolic hit/miss stage."""
        from planning.core.camera_network import projection_jacobian
        from planning.core.casadi_efe import INNOVATION_COV_FLOOR_PX2
        R = np.asarray(self.R_visible, dtype=float)
        mu, Sigma_hit, _Gamma = self.approx_observation(m,S,R_override=R)
        Sigma_miss = Sigma_hit-R
        # risk_ca's existing numerical jitter, unchanged in the symbolic model.
        target = (goal_obs, goal_cov+1e-9*np.eye(2))
        hit = risk_components(mu, (Sigma_hit+Sigma_hit.T)*.5+1e-9*np.eye(2), target)
        miss = risk_components(mu, (Sigma_miss+Sigma_miss.T)*.5+1e-9*np.eye(2), target)
        risk_parts = {key:p_use*hit[key]+(1-p_use)*miss[key] for key in hit}
        J = np.column_stack((projection_jacobian(self.camera.H,m),np.zeros(2)))
        prior = (S+S.T)*.5
        innovation = J@prior@J.T+R+INNOVATION_COV_FLOOR_PX2*np.eye(2)
        K = np.linalg.solve(innovation,J@prior).T
        A = np.eye(3)-K@J
        posterior = A@prior@A.T+K@R@K.T
        posterior = (posterior+posterior.T)*.5
        def entropy(P):
            return .5*(3*math.log(2*math.pi*math.e)+math.log(max(float(np.linalg.det(P)),1e-12)))
        hit_entropy, prior_entropy = entropy(posterior), entropy(prior)
        return (risk_parts, p_use*hit_entropy+(1-p_use)*prior_entropy,
                p_use*posterior+(1-p_use)*prior, hit['total'], hit_entropy)

    def _evaluate_controls(
        self,
        controls_flat,
        m0,
        S0,
        goal_state,
        goal_obs,
        goal_obs_cov,
        return_metrics=False,
        *,
        progress_index=0.0,
        R_baseline_override=None,
    ):
        del goal_obs_cov
        controls_flat = np.asarray(controls_flat, dtype=float)
        assert controls_flat.size == self.horizon * 2, f"controls_flat size {controls_flat.size} != expected {self.horizon * 2}"
        controls = controls_flat.reshape(self.horizon, 2)

        validate_planning_inputs(m0, S0, goal_state[:2])
        if not np.isfinite(controls).all():
            raise ValueError('nonfinite objective controls')

        if (self.camera_network is not None
                and self.camera_network_objective == 'metric_expected_belief'):
            return self._evaluate_metric_network_controls(
                controls, m0, S0, goal_state,
                return_metrics=return_metrics,
            )

        m = m0.copy()
        S = S0.copy()
        total_risk = 0.0
        total_amb = 0.0
        total_control = 0.0
        total_obstacle = 0.0
        total_risk_mean = 0.0
        total_risk_cov_trace = 0.0
        total_risk_cov_logdet = 0.0
        total_risk_const = 0.0
        total_delta_risk_visibility = 0.0
        total_delta_ambiguity_visibility = 0.0
        use_observation_risk = self.use_obs_risk
        use_ambiguity_term = self.use_ambiguity
        goal_xy = np.asarray(goal_state[:2], dtype=float).reshape(2)
        if self.use_hit_miss_mixture and goal_obs is None:
            # The optional branch helper computes component diagnostics even
            # when both weighted observation terms are disabled.
            goal_obs = self._goal_obs(goal_state)
        R_good = np.asarray(
            R_baseline_override
            if R_baseline_override is not None
            else np.diag([float(self.r_visible_uv) ** 2, float(self.r_visible_uv) ** 2]),
            dtype=float,
        )

        for t in range(self.horizon):
            u = controls[t]
            m, S = self.predict(m, S, u)
            validate_covariance(S, name='predicted covariance')
            vis_diag = self.planning_visibility_diagnostics(m, S)
            p_vis = vis_diag['p_vis']
            R_plan = vis_diag['R_plan']
            if self.use_hit_miss_mixture:
                weight_t = self.discount_gamma**t
                goal_cov = self.goal_obs_cov_for_progress(
                    (float(progress_index)+t)/max(self.goal_progress_n_steps,1))
                parts, expected_entropy, S_drive, hit_risk, hit_entropy = self._mixture_stage_metrics(
                    m,S,self._visibility_effective_score(p_vis),goal_obs,goal_cov)
                risk_scale = self.risk_weight_obs*self.observation_risk_scale if use_observation_risk else 0.
                ambiguity_scale = self.ambiguity_weight*self.ambiguity_term_scale if use_ambiguity_term else 0.
                total_risk += weight_t*risk_scale*parts['total']
                total_risk_mean += weight_t*risk_scale*parts['mean']
                total_risk_cov_trace += weight_t*risk_scale*parts['cov_trace']
                total_risk_cov_logdet += weight_t*risk_scale*parts['cov_logdet']
                total_risk_const += weight_t*risk_scale*parts['const']
                total_amb += weight_t*ambiguity_scale*expected_entropy
                total_delta_risk_visibility += weight_t*risk_scale*(parts['total']-hit_risk)
                total_delta_ambiguity_visibility += weight_t*ambiguity_scale*(expected_entropy-hit_entropy)
                total_obstacle += weight_t*self.obstacle_penalty(m,S_drive if self.use_belief_nogo_cost else S)
                total_control += weight_t*self.control_weight*float(u@u)
                continue
            mu_y = Sigma_y = Gamma = None
            if use_observation_risk or use_ambiguity_term or self.use_belief_nogo_cost:
                mu_y, Sigma_y, Gamma = self.approx_observation(
                    m,
                    S,
                    method=self.approx_method,
                    R_override=R_plan,
                )
            weight_t = self.discount_gamma ** t
            observation_risk = 0.0
            baseline_risk = 0.0
            ambiguity_current = 0.0
            ambiguity_baseline = 0.0
            Sigma_good = None
            Gamma_good = None
            if use_observation_risk and mu_y is not None:
                goal_cov_t = self.goal_obs_cov_for_progress(
                    (float(progress_index) + float(t)) / max(self.goal_progress_n_steps, 1)
                )
                risk_parts = risk_components(mu_y, Sigma_y, (goal_obs, goal_cov_t))
                risk_scale = self.risk_weight_obs * self.observation_risk_scale
                observation_risk = risk_scale * risk_parts['total']
                total_risk_mean += weight_t * risk_scale * risk_parts['mean']
                total_risk_cov_trace += weight_t * risk_scale * risk_parts['cov_trace']
                total_risk_cov_logdet += weight_t * risk_scale * risk_parts['cov_logdet']
                total_risk_const += weight_t * risk_scale * risk_parts['const']
                mu_good, Sigma_good, Gamma_good = self.approx_observation(
                    m,
                    S,
                    method=self.approx_method,
                    R_override=R_good,
                )
                baseline_parts = risk_components(mu_good, Sigma_good, (goal_obs, goal_cov_t))
                baseline_risk = risk_scale * baseline_parts['total']
            total_risk += weight_t * observation_risk
            if use_ambiguity_term and Sigma_y is not None:
                ambiguity_scale = self.ambiguity_weight * self.ambiguity_term_scale
                ambiguity_current = ambiguity_scale * ambiguity(Sigma_y, Gamma, S)
                if Sigma_good is None or Gamma_good is None:
                    _mu_good, Sigma_good, Gamma_good = self.approx_observation(
                        m,
                        S,
                        method=self.approx_method,
                        R_override=R_good,
                    )
                ambiguity_baseline = ambiguity_scale * ambiguity(Sigma_good, Gamma_good, S)
                total_amb += weight_t * ambiguity_current
            total_delta_risk_visibility += weight_t * (observation_risk - baseline_risk)
            total_delta_ambiguity_visibility += weight_t * (ambiguity_current - ambiguity_baseline)
            S_nogo = S
            if self.use_belief_nogo_cost and Sigma_y is not None and Gamma is not None:
                S_nogo = self._expected_state_posterior_covariance(S, Sigma_y, Gamma)
            total_obstacle += weight_t * self.obstacle_penalty(m, S_nogo)
            total_control += weight_t * self.control_weight * float(u[0] ** 2 + u[1] ** 2)

        total = total_risk + total_amb + total_control + total_obstacle
        if return_metrics:
            return total, {
                'risk_cost': float(total_risk),
                'ambiguity_cost': float(total_amb),
                'control_cost': float(total_control),
                'obstacle_cost': float(total_obstacle),
                'risk_mean': float(total_risk_mean),
                'risk_cov_trace': float(total_risk_cov_trace),
                'risk_cov_logdet': float(total_risk_cov_logdet),
                'risk_const': float(total_risk_const),
                'delta_risk_visibility': float(total_delta_risk_visibility),
                'delta_ambiguity_visibility': float(total_delta_ambiguity_visibility),
            }
        return total

    def _evaluate_metric_network_controls(
        self, controls, m0, S0, goal_state, *, return_metrics=False,
    ):
        """NumPy accounting for the world-XY camera-network objective."""
        m = np.asarray(m0, dtype=float).copy()
        S = np.asarray(S0, dtype=float).copy()
        goal_xy = np.asarray(goal_state[:2], dtype=float)
        goal_cov = np.eye(2, dtype=float) * self.network_goal_std_m**2
        anneal_goal_prior = (
            self.network_goal_std_start_m is not None
            and self.network_goal_std_start_m != self.network_goal_std_m)
        totals = dict(risk_cost=0., ambiguity_cost=0., control_cost=0.,
                      obstacle_cost=0., risk_mean=0., risk_cov_trace=0.,
                      risk_cov_logdet=0., risk_const=0.,
                      delta_risk_visibility=0., delta_ambiguity_visibility=0.)
        risk_scale = (self.risk_weight_obs * self.observation_risk_scale
                      if self.use_obs_risk else 0.)
        ambiguity_scale = (self.ambiguity_weight * self.ambiguity_term_scale
                           if self.use_ambiguity else 0.)
        arrival_radius = float(self.optimizer_terminal_goal_tolerance_m)
        arrival_softness = 0.25
        active = 1.0
        total_active_weight = 0.0
        for t, u in enumerate(controls):
            m_prev = np.asarray(m, dtype=float).copy()
            m, S = self.predict(m, S, u)
            validate_covariance(S, positive_definite=True, name='predicted covariance')
            weight_t = self.discount_gamma**t
            if arrival_radius > 0.:
                # Parked steps are not part of the plan; see the CasADi objective.
                reached = float(np.linalg.norm(m[:2] - goal_xy))
                active *= 1.0 / (1.0 + math.exp(
                    -(reached - arrival_radius) / arrival_softness))
            weight_t *= active
            total_active_weight += weight_t
            if anneal_goal_prior:
                goal_cov = self.goal_obs_cov_for_progress(
                    float(t) / float(max(self.goal_progress_n_steps, 1)))
            parts = risk_components(m[:2], S[:2, :2], (goal_xy, goal_cov))
            if self.terminal_risk_only:
                # Overwrite, so after the loop this is the FINAL belief's risk
                # alone and the term carries no duration. Twin of the CasADi
                # branch in casadi_efe.make_metric_network_efe_valgrad_fn.
                totals['risk_cost'] = weight_t * risk_scale * parts['total']
                for key in ('mean', 'cov_trace', 'cov_logdet', 'const'):
                    totals[f'risk_{key}'] = weight_t * risk_scale * parts[key]
            else:
                totals['risk_cost'] += weight_t * risk_scale * parts['total']
                for key in ('mean', 'cov_trace', 'cov_logdet', 'const'):
                    totals[f'risk_{key}'] += weight_t * risk_scale * parts[key]
            S_post, expected_entropy = self.camera_network.expected_belief(
                m, S, self.visibility_sigma_kappa,
                opportunities=self.camera_network_updates_per_step,
            )
            if self.kouw_et1_ambiguity:
                R_eff = self.camera_network.effective_observation_covariance(
                    m, S, self.visibility_sigma_kappa)
                sign, logdet = np.linalg.slogdet(0.5 * (R_eff + R_eff.T))
                if sign <= 0:
                    raise ValueError('effective observation covariance must be positive definite')
                # ANCHORED to the ideally observed pose. The raw Lemma 1 term is
                # an absolute differential entropy, so under the arrival gate its
                # route-independent constant becomes constant*T -- a duration
                # term whose sign is set by the units R_eff is written in. The
                # log-ratio is dimensionless and clipped so no route is paid for
                # lasting longer. See _anchored_ambiguity_ca in casadi_efe.
                expected_entropy = 0.5 * max(
                    logdet - self._reference_ambiguity_logdet(), 0.0)
            totals['ambiguity_cost'] += weight_t * ambiguity_scale * expected_entropy
            if self.use_belief_nogo_cost:
                obstacle = self.obstacle_penalty(m, S_post)
            else:
                obstacle = float(np.mean([
                    self.obstacle_penalty(unicycle_step(
                        m_prev, u, self.dt * substep / self.obstacle_substeps))
                    for substep in range(1, self.obstacle_substeps + 1)
                ]))
            totals['obstacle_cost'] += weight_t * obstacle
            totals['control_cost'] += weight_t * self.control_weight * float(u @ u)
            S = S_post
        normalizer = max(float(total_active_weight), 1e-8)
        for key in totals:
            totals[key] /= normalizer
        total = sum(totals[key] for key in (
            'risk_cost', 'ambiguity_cost', 'control_cost', 'obstacle_cost'))
        if return_metrics:
            return float(total), {key: float(value) for key, value in totals.items()}
        return float(total)

    # The ambiguity floor: ONE covariance, shared by every arm.
    #
    # The floor enters as a constant subtraction, so any value below the
    # tightest reachable R_eff gives identical rankings; only clipping a real
    # pose to zero destroys information. It must therefore be the SAME for every
    # arm being compared. Each arm's own commissioned best-R floor differs
    # (measured: 3.5 nats between the loosest and tightest of the current seven
    # arms), so anchoring each arm to itself would shift every arm by a
    # different constant and make the conditions incomparable -- the key thing an information-field
    # comparison must not do.
    #
    # 1.5 mm isotropic position sd, chosen inside a two-sided window:
    #
    #   upper bound  the tightest R_eff reachable on any arm is det 2.39e-11
    #                (~2.2 mm sd). The floor must stay below it or it clips real
    #                poses and deletes signal.
    #   lower bound  casadi_efe._logdet_small_pd clamps det at 1e-12. A floor at
    #                or under that clamp makes the CasADi log-determinant
    #                disagree with the NumPy one (measured 8.3e-6 per
    #                evaluation, compounding across opportunities), which breaks
    #                the 1e-8 agreement the two back-ends must hold.
    #
    # 1.5 mm sits 5x above the clamp and 4.7x below the tightest reachable pose.
    # The window is narrow because the commissioned cameras are very precise;
    # if a future arm is tighter still, widen the clamp rather than lowering
    # this, and re-check both bounds. A threshold with stated margins, not a
    # fitted value: verify_anchored_ambiguity.py reports the clip count (must be
    # 0) and the twins' agreement.
    AMBIGUITY_FLOOR_POSITION_SD_M = 1.5e-3

    def reference_observation_covariance(self):
        """The shared ambiguity floor. See CameraNetwork.reference_observation_covariance.

        A THRESHOLD, not an estimate: it sits below the tightest reachable
        R_eff of every arm being compared, and anywhere below that the exact
        value cancels out of the ranking. It is deliberately independent of the
        arm's own R model so the arms stay comparable.
        """
        variance = float(self.AMBIGUITY_FLOOR_POSITION_SD_M) ** 2
        return variance * np.eye(2)

    def _reference_ambiguity_logdet(self):
        cached = getattr(self, '_reference_logdet_cache', None)
        if cached is None:
            R_ref = self.reference_observation_covariance()
            sign, logdet = np.linalg.slogdet(0.5 * (R_ref + R_ref.T))
            if sign <= 0:
                raise ValueError('reference observation covariance must be positive definite')
            cached = float(logdet)
            self._reference_logdet_cache = cached
        return cached

    def plan(self, m0, S0, goal_xy, *, progress_index=0.0):
        t_plan_start = time.perf_counter()
        m0, S0, goal_xy = validate_planning_inputs(m0, S0, goal_xy)
        if not np.isfinite(self.dt) or self.dt <= 0 or self.horizon <= 0:
            raise ValueError('planning horizon and dt must be positive')
        if not (self.camera_network is not None
                and self.camera_network_objective == 'metric_expected_belief'):
            from planning.core.camera_network import projection_jacobian
            projection_jacobian(self.camera.H, m0)
            projection_jacobian(self.camera.H, np.r_[goal_xy, 0.])
        if not np.isfinite(progress_index):
            raise ValueError('goal progress index must be finite')
        progress_index = float(max(progress_index, 0.0))

        # Reset warm start when goal changes by more than 0.5 m to prevent
        # stale plans from creating a stuck local minimum after goal switch.
        goal_xy_arr = np.asarray(goal_xy, dtype=float).reshape(2)
        if self._prev_goal_xy is not None:
            if float(np.linalg.norm(goal_xy_arr - self._prev_goal_xy)) > 0.5:
                self.prev_controls_flat = None
        self._prev_goal_xy = goal_xy_arr.copy()

        (
            goal_state,
            goal_obs,
            goal_obs_cov,
            use_observation_risk,
            use_ambiguity_term,
        ) = self._resolve_plan_problem(m0, goal_xy)

        block_steps = self.optimizer_control_block_steps
        decision_blocks = int(math.ceil(self.horizon / block_steps))

        def compress_controls(full):
            full = np.asarray(full, dtype=float).reshape(self.horizon, 2)
            return np.asarray([
                np.mean(full[start:min(start + block_steps, self.horizon)], axis=0)
                for start in range(0, self.horizon, block_steps)
            ], dtype=float).reshape(-1)

        def expand_controls(sparse):
            sparse = np.asarray(sparse, dtype=float).reshape(decision_blocks, 2)
            return np.repeat(sparse, block_steps, axis=0)[:self.horizon].reshape(-1)

        bounds = []
        for _ in range(decision_blocks):
            bounds.append((self.v_min, self.v_max))
            bounds.append((self.w_min, self.w_max))

        x0_default = self._initial_controls_flat()
        init_candidates: list[tuple[str, np.ndarray]] = [
            ('warm_or_cold', compress_controls(x0_default)),
        ]
        for ms_name, ms_controls in self._build_multistart_candidates(m0, goal_xy):
            init_candidates.append((ms_name, compress_controls(ms_controls)))

        objective_scales = self._objective_scales(
            expand_controls(init_candidates[0][1]), m0, S0,
            goal_state, goal_obs, goal_obs_cov,
        )
        best_candidate = None
        best_init_name = ''
        backend_used = 'casadi'
        optimizer_success = False
        optimizer_status = 0
        optimizer_nit = 0
        optimizer_nfev = 0
        optimizer_message = ''

        try:
            fg_calls = {'count': 0}
            goal_obs_eval = np.asarray(goal_obs if goal_obs is not None else self._goal_obs(goal_state), dtype=float)
            valgrad = self._get_casadi_valgrad(
                goal_state,
                goal_obs,
                use_observation_risk=use_observation_risk,
                use_ambiguity_term=use_ambiguity_term,
            )

            def objective(u):
                u_arr = np.asarray(u, dtype=float)
                start = time.perf_counter()
                val_out, grad_out = valgrad(
                    u_arr,
                    m0,
                    S0,
                    goal_obs_eval,
                    np.asarray(goal_xy, dtype=float).reshape(2),
                    progress_index,
                )
                if fg_calls['count'] == 0:
                    self._runtime_debug_print(
                        "[planner_debug] First CasADi objective/gradient eval returned in "
                        f"{(time.perf_counter() - start) * 1000.0:.1f} ms "
                        f"with J={val_out:.3f}, grad_norm={np.linalg.norm(grad_out):.3f}"
                    )
                fg_calls['count'] += 1
                if (not np.isfinite(val_out) or np.asarray(grad_out).shape != u_arr.shape
                        or not np.isfinite(grad_out).all()):
                    raise ValueError('nonfinite objective or invalid gradient')
                return val_out, grad_out

            minimize_start = time.perf_counter()
            self._runtime_debug_print(
                "[planner_debug] Starting CasADi-backed scipy.optimize.minimize "
                f"(maxiter={self.optimizer_maxiter}, maxfun={self.optimizer_maxfun}, ftol={self.optimizer_ftol}, "
                f"init_candidates={len(init_candidates)}, control_blocks={decision_blocks}, "
                f"block_steps={block_steps})"
            )

            for init_name, x_init in init_candidates:
                attempt_start = time.perf_counter()
                def checked(raw):
                    try:
                        return self._checked_candidate_controls(
                            expand_controls(raw), m0, S0, goal_state, goal_obs, goal_obs_cov,
                            objective_scales, progress_index=progress_index)
                    except (ValueError, TypeError, RuntimeError, np.linalg.LinAlgError) as exc:
                        self._runtime_debug_print(f"[planner_debug] rejected numerical candidate: {exc}")
                        return None

                # Check the seed independently: an optimizer exception, malformed
                # x, or NaN result must not hide an already evaluable finite seed.
                seed_checked = checked(x_init)
                from types import SimpleNamespace
                result = SimpleNamespace(success=False, status=-1, nit=0, nfev=0,
                                         message='optimizer failed; evaluated seed', x=None)
                try:
                    result = minimize(
                        objective, np.asarray(x_init, dtype=float), jac=True,
                        method='L-BFGS-B', bounds=bounds,
                        options={'maxiter': self.optimizer_maxiter, 'maxfun': self.optimizer_maxfun,
                                 'ftol': self.optimizer_ftol, 'gtol': self.optimizer_gtol})
                    raw_valid = result.x is not None
                    for name in ('fun', 'jac'):
                        value = getattr(result, name, None)
                        if value is not None and not np.isfinite(value).all():
                            raw_valid = False
                    opt_checked = checked(result.x) if raw_valid else None
                except Exception as exc:
                    self._runtime_debug_print(f"[planner_debug] init={init_name!s} threw {type(exc).__name__}: {exc}")
                    opt_checked = None
                if opt_checked is None and seed_checked is None:
                    continue
                use_seed = opt_checked is None
                if opt_checked is not None and seed_checked is not None:
                    opt_candidate, opt_diag = opt_checked
                    seed_candidate, seed_diag = seed_checked
                    use_seed = self._prefer_candidate(
                        candidate_valid=seed_diag['rollout_valid'],
                        candidate_goal_feasible=self._terminal_goal_feasible(seed_diag),
                        candidate_cost=seed_candidate['total_cost'],
                        incumbent_valid=opt_diag['rollout_valid'],
                        incumbent_goal_feasible=self._terminal_goal_feasible(opt_diag),
                        incumbent_cost=opt_candidate['total_cost'])
                candidate, diag_attempt = seed_checked if use_seed else opt_checked
                candidate['optimizer_seed_fallback'] = use_seed
                ctrls_attempt = np.asarray(candidate['controls_flat'], dtype=float).reshape(self.horizon, 2)
                cand_valid = bool(diag_attempt['rollout_valid'])
                cand_goal_feasible = self._terminal_goal_feasible(diag_attempt)
                source_label = (
                    f'solver:shifted_warm_start' if (init_name == 'warm_or_cold' and self.prev_controls_flat is not None)
                    else f'solver:{init_name}'
                )
                if use_seed:
                    source_label = source_label.replace('solver:', 'seed:', 1)
                candidate.update({
                    'source': source_label,
                    'rollout_valid': cand_valid,
                    'terminal_goal_feasible': cand_goal_feasible,
                    'optimizer_result': result,
                })
                self._runtime_debug_print(
                    f"[planner_debug] init={init_name!s} solver finished "
                    f"J={candidate['total_cost']:.3f}, valid={cand_valid}, "
                    f"goal_gap={float(diag_attempt.get('terminal_goal_distance_pred', math.nan)):.3f}, "
                    f"goal_feasible={cand_goal_feasible}, "
                    f"min_clear={float(diag_attempt.get('min_predicted_obstacle_distance_m', math.nan)):.3f}, "
                    f"invalid={str(diag_attempt.get('invalid_reason', '')) or '-'}, "
                    f"success={bool(result.success)}, status={int(result.status)}, "
                    f"seed_fallback={bool(candidate.get('optimizer_seed_fallback', False))}, "
                    f"nit={int(getattr(result, 'nit', 0) or 0)}, "
                    f"nfev={int(getattr(result, 'nfev', 0) or 0)}, "
                    f"dt={(time.perf_counter() - attempt_start) * 1000.0:.0f}ms"
                )

                if best_candidate is None:
                    keep = True
                else:
                    best_valid = bool(best_candidate.get('rollout_valid', False))
                    best_goal_feasible = bool(
                        best_candidate.get('terminal_goal_feasible', True)
                    )
                    # The smooth no-go term shapes the continuous optimization,
                    # but the final multi-start choice should never prefer an
                    # invalid shortcut or an incomplete route over a candidate
                    # satisfying the task. This is condition-neutral feasibility
                    # handling, not route scripting.
                    keep = self._prefer_candidate(
                        candidate_valid=cand_valid,
                        candidate_goal_feasible=cand_goal_feasible,
                        candidate_cost=candidate['total_cost'],
                        incumbent_valid=best_valid,
                        incumbent_goal_feasible=best_goal_feasible,
                        incumbent_cost=best_candidate['total_cost'],
                    )
                if keep:
                    best_candidate = candidate
                    best_init_name = init_name
                    optimizer_success = bool(result.success)
                    optimizer_status = int(result.status)
                    optimizer_nit = int(getattr(result, 'nit', 0) or 0)
                    optimizer_nfev = int(getattr(result, 'nfev', 0) or 0)
                    optimizer_message = str(result.message or '')

            if best_candidate is None:
                raise RuntimeError("Planner optimizer returned no finite solution from any init")
            if not bool(best_candidate.get('rollout_valid', False)):
                stop_controls = np.zeros(self.horizon * 2, dtype=float)
                stop_candidate, stop_diag = self._checked_candidate_controls(
                    stop_controls, m0, S0, goal_state, goal_obs, goal_obs_cov,
                    objective_scales, progress_index=progress_index)
                if bool(stop_diag['rollout_valid']):
                    stop_candidate.update({
                        'source': 'safe_stop_invalid_rollout',
                        'rollout_valid': True,
                        'optimizer_result': best_candidate.get('optimizer_result'),
                    })
                    best_candidate = stop_candidate
                    best_init_name = 'safe_stop_invalid_rollout'
                    optimizer_success = False
                    optimizer_status = -2
                    optimizer_nit = 0
                    optimizer_nfev = 0
                    optimizer_message = (
                        'All optimized candidates violated the known driveable '
                        'region; selected zero-control safe stop.'
                    )
            self._runtime_debug_print(
                f"[planner_debug] Best optimizer init={best_init_name!s} "
                f"J={best_candidate['total_cost']:.3f}, valid={best_candidate.get('rollout_valid', False)}, "
                f"goal_feasible={best_candidate.get('terminal_goal_feasible', True)}"
            )
            self._runtime_debug_print(
                "[planner_debug] CasADi-backed minimize finished in "
                f"{(time.perf_counter() - minimize_start) * 1000.0:.1f} ms "
                f"(shared_fg_evals={fg_calls['count']})"
            )
        except Exception as exc:
            raise RuntimeError(
                "Planner optimization failed "
                f"(backend={backend_used}, approx={self.approx_method}, "
                f"horizon={self.horizon}, dt={self.dt}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if best_candidate is None:
            raise RuntimeError("Planner produced no candidate solution")

        best_controls_flat = np.asarray(best_candidate['controls_flat'], dtype=float)
        best_controls = best_controls_flat.reshape(self.horizon, 2).copy()
        total_cost = float(best_candidate['total_cost'])
        metrics = dict(best_candidate['metrics'])
        vis_diag = self.planning_visibility_diagnostics(m0, S0)

        states = rollout_unicycle(m0, best_controls, self.dt)
        plan_diag = self._trajectory_plan_diagnostics(m0, S0, best_controls, goal_xy)
        # Invalid routes are returned with explicit diagnostics when no valid stop
        # exists; they must not become the warm start for the next solve.
        self.prev_controls_flat = best_controls.reshape(-1).copy() if plan_diag['rollout_valid'] else None
        selected_source = str(best_candidate.get('source', ''))
        solve_time_s = float(max(time.perf_counter() - t_plan_start, 0.0))
        return PlanResult(
            controls=best_controls,
            states=states,
            total_cost=float(total_cost),
            risk_cost=float(metrics.get('risk_cost', 0.0)),
            ambiguity_cost=float(metrics.get('ambiguity_cost', 0.0)),
            control_cost=float(metrics.get('control_cost', 0.0)),
            obstacle_cost=float(metrics.get('obstacle_cost', 0.0)),
            risk_mean=float(metrics.get('risk_mean', 0.0)),
            risk_cov_trace=float(metrics.get('risk_cov_trace', 0.0)),
            risk_cov_logdet=float(metrics.get('risk_cov_logdet', 0.0)),
            delta_risk_visibility=float(metrics.get('delta_risk_visibility', 0.0)),
            delta_ambiguity_visibility=float(metrics.get('delta_ambiguity_visibility', 0.0)),
            backend=str(backend_used),
            optimizer_success=optimizer_success,
            optimizer_status=optimizer_status,
            optimizer_nit=optimizer_nit,
            optimizer_nfev=optimizer_nfev,
            optimizer_message=optimizer_message,
            solve_time_s=solve_time_s,
            selected_source=selected_source,
            p_vis_plan=float(vis_diag['p_vis']),
            p_vis_plan_eff=float(vis_diag['p_vis_eff']),
            r_plan_u_std=float(vis_diag['r_plan_u_std']),
            r_plan_v_std=float(vis_diag['r_plan_v_std']),
            terminal_goal_distance_pred=float(plan_diag['terminal_goal_distance_pred']),
            terminal_goal_progress_m=float(plan_diag['terminal_goal_progress_m']),
            fraction_horizon_low_pvis=float(plan_diag['fraction_horizon_low_pvis']),
            fraction_horizon_high_ambiguity=float(plan_diag['fraction_horizon_high_ambiguity']),
            min_predicted_obstacle_distance_m=float(plan_diag['min_predicted_obstacle_distance_m']),
            rollout_valid=bool(plan_diag['rollout_valid']),
            invalid_reason=str(plan_diag['invalid_reason']),
        )
