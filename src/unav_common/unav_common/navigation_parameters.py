"""Physical domains for explicitly configured navigation parameters.

Zero-valued disable/inherit sentinels are allowed only where documented.
This checks configuration, not controller stability or sensor calibration.
"""
import math


POSITIVE = frozenset({
    'robot_length_m', 'robot_width_m',
    'v_max', 'dt', 'horizon', 'global_horizon', 'local_horizon',
    'cmd_publish_rate', 'plan_rate', 'local_plan_rate', 'belief_publish_rate',
    'manager_decision_rate_hz', 'robot_collision_radius_m',
    'waypoint_spacing_m', 'waypoint_arrival_radius_m', 'goal_success_radius',
    'nogo_warning_band', 'nogo_logbarrier_eps', 'nogo_belief_kappa',
    'optimizer_maxiter', 'optimizer_maxfun', 'local_optimizer_maxiter',
    'optimizer_ftol', 'optimizer_gtol',
    'goal_prior_u_std_start', 'goal_prior_v_std_start',
    'goal_prior_u_std_final', 'goal_prior_v_std_final', 'goal_tightening_power',
})
NONNEGATIVE = frozenset({
    'global_dt', 'process_noise_xy', 'process_noise_theta',
    'state_reject_inflate_m2', 'stale_belief_inflate_m2_per_s',
    'stale_belief_inflate_cap_m2', 'goal_success_hold_s',
    'nogo_safe_distance', 'pixel_correction_nis_threshold',
    'nogo_weight', 'nogo_near_weight', 'control_weight', 'risk_weight_obs',
    'ambiguity_weight', 'observation_risk_scale', 'ambiguity_term_scale',
    'optimizer_terminal_goal_tolerance_m',
})
ENUMS = {'nogo_mode': {'keep_in', 'keep_out'}, 'nogo_penalty_type': {'warning_band'}}
PARAMETERS = POSITIVE | NONNEGATIVE | {'discount_gamma'} | ENUMS.keys()


def validate_navigation_parameters(values):
    """Reject invalid explicit values; leave absent values to launch defaults."""
    for key in PARAMETERS & values.keys():
        raw = values[key]
        if key in ENUMS:
            if raw not in ENUMS[key]:
                raise ValueError(f'{key} must be one of {sorted(ENUMS[key])}')
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f'{key} must be a finite number, got {raw!r}') from None
        if isinstance(raw, bool) or not math.isfinite(value):
            raise ValueError(f'{key} must be a finite number, got {raw!r}')
        if key in POSITIVE and value <= 0:
            raise ValueError(f'{key} must be positive')
        if key in NONNEGATIVE and value < 0:
            raise ValueError(f'{key} must be nonnegative')
        if (key.endswith('horizon') or key.endswith(('maxiter', 'maxfun'))) and not value.is_integer():
            raise ValueError(f'{key} must be an integer')
        if key == 'discount_gamma' and not 0 < value <= 1:
            raise ValueError('discount_gamma must be in (0, 1]')
