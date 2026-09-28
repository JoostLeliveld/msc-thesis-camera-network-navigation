"""Numerical contracts for planning inputs and solver results, without ROS.

Solver convergence is diagnostic. Feasibility comes from the represented motion,
the explicit result contract and the caller's geometry check. No covariance or
control is clipped here to turn an invalid candidate into a valid one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from planning.core.dynamics import unicycle_step


def immutable_array(value):
    """An owned float array whose bytes cannot be made writable by a consumer."""
    array = np.ascontiguousarray(value, dtype=float)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def validate_covariance(value, *, dimension=3, positive_definite=False, name='belief covariance'):
    array = np.asarray(value, dtype=float)
    if array.shape != (dimension, dimension) or not np.isfinite(array).all():
        raise ValueError(f'{name} must be finite {dimension}x{dimension}')
    # A rounding allowance, not a variance floor or a statistical tuning value.
    rounding = 64 * np.finfo(float).eps * max(1., float(np.linalg.norm(array, ord=2)))
    if np.max(np.abs(array-array.T)) > rounding:
        raise ValueError(f'{name} must be symmetric')
    smallest = float(np.linalg.eigvalsh((array+array.T)*.5)[0])
    if smallest < -rounding or (positive_definite and smallest <= 0.):
        kind = 'positive definite' if positive_definite else 'positive semidefinite'
        raise ValueError(f'{name} must be {kind}')
    return array.copy()


def validate_planning_inputs(initial_state, initial_covariance, goal_xy):
    state = np.asarray(initial_state, dtype=float)
    goal = np.asarray(goal_xy, dtype=float)
    if state.shape != (3,) or not np.isfinite(state).all():
        raise ValueError('initial state must be finite [x,y,yaw]')
    if goal.shape != (2,) or not np.isfinite(goal).all():
        raise ValueError('goal must be finite [x,y]')
    covariance = validate_covariance(initial_covariance)
    return state.copy(), covariance, goal.copy()


def validate_controls(controls, *, control_bounds, expected_horizon=None):
    controls = np.asarray(controls, dtype=float)
    bounds = np.asarray(control_bounds, dtype=float)
    if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError('control bounds must be finite ((v_min,v_max),(w_min,w_max))')
    if controls.ndim != 2 or controls.shape[1] != 2 or controls.shape[0] == 0:
        raise ValueError('controls must be nonempty Nx2')
    if expected_horizon is not None and controls.shape[0] != int(expected_horizon):
        raise ValueError('control horizon differs from request')
    if not np.isfinite(controls).all():
        raise ValueError('nonfinite controls')
    rounding = 64*np.finfo(float).eps*max(1., float(np.max(np.abs(bounds))))
    if np.any(controls < bounds[:, 0]-rounding) or np.any(controls > bounds[:, 1]+rounding):
        raise ValueError('controls exceed request bounds')
    return controls.copy()


@dataclass(frozen=True)
class PlanValidation:
    valid: bool
    reason: str = ''
    controls: np.ndarray | None = None
    states: np.ndarray | None = None
    terminal_goal_distance_m: float | None = None

    def __iter__(self):
        # Convenient bool/reason unpacking without throwing away validated data.
        yield self.valid
        yield self.reason


def validate_plan_result(
    result, *, initial_state, initial_covariance, goal_xy, dt, control_bounds,
    expected_horizon=None, require_complete=False, terminal_tolerance_m=0.,
    geometry_validator: Callable | None = None,
):
    """Validate a solver result and return immutable copies on success.

    ``geometry_validator(states, controls)`` returns ``(valid, reason)`` and must
    check the caller's actual geometry contract, including the first pose and
    swept motion. No callback means this is numerical validation only. +infinity
    clearance explicitly denotes absent constraints; NaN and -infinity are invalid.
    The optional terminal gate uses recomputed distance, never a permissive field
    default. Both successful and iteration-limited optimizers use these same checks.
    """
    try:
        state, _covariance, goal = validate_planning_inputs(initial_state, initial_covariance, goal_xy)
        step = float(dt)
        tolerance = float(terminal_tolerance_m)
        if not np.isfinite(step) or step <= 0:
            raise ValueError('planning dt must be finite and positive')
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError('terminal tolerance must be finite and nonnegative')
        validity = getattr(result, 'rollout_valid', None)
        if not isinstance(validity, (bool, np.bool_)) or not validity:
            raise ValueError('explicit valid rollout is required')
        controls = validate_controls(getattr(result, 'controls', []),
                                     control_bounds=control_bounds, expected_horizon=expected_horizon)
        states = np.asarray(getattr(result, 'states', []), dtype=float)
        if states.shape != (len(controls)+1, 3) or not np.isfinite(states).all():
            raise ValueError('states must be finite (N+1)x3')
        if not hasattr(result, 'total_cost') or not np.isfinite(float(result.total_cost)):
            raise ValueError('finite objective is required')
        for field in ('risk_cost', 'ambiguity_cost', 'control_cost', 'obstacle_cost',
                      'risk_mean', 'risk_cov_trace', 'risk_cov_logdet',
                      'delta_risk_visibility', 'delta_ambiguity_visibility'):
            if hasattr(result, field) and not np.isfinite(float(getattr(result, field))):
                raise ValueError(f'nonfinite objective component: {field}')
        if hasattr(result, 'min_predicted_obstacle_distance_m'):
            clearance = float(result.min_predicted_obstacle_distance_m)
            if np.isnan(clearance) or clearance == -np.inf:
                raise ValueError('invalid reported clearance')
        predicted = [state]
        for control in controls:
            predicted.append(unicycle_step(predicted[-1], control, step))
        predicted = np.asarray(predicted)
        if not np.isfinite(predicted).all():
            raise ValueError('nonfinite motion rollout')
        # Purely numerical consistency tolerance; not an allowed path deviation.
        if not np.allclose(states[:, :2], predicted[:, :2], atol=1e-8, rtol=1e-10):
            raise ValueError('states differ from request motion rollout')
        yaw_error = np.arctan2(np.sin(states[:, 2]-predicted[:, 2]),
                              np.cos(states[:, 2]-predicted[:, 2]))
        if np.max(np.abs(yaw_error)) > 1e-8:
            raise ValueError('headings differ from request motion rollout')
        distance = float(np.linalg.norm(states[-1, :2]-goal))
        if hasattr(result, 'terminal_goal_distance_pred'):
            reported = float(result.terminal_goal_distance_pred)
            if not np.isfinite(reported) or not np.isclose(reported, distance, atol=1e-8, rtol=1e-10):
                raise ValueError('invalid reported terminal distance')
        if require_complete and distance > tolerance:
            raise ValueError('incomplete route')
        if require_complete and str(getattr(result, 'selected_source', '')).startswith('safe_stop'):
            raise ValueError('safe stop is not a mission route')
        checked_controls, checked_states = immutable_array(controls), immutable_array(states)
        if geometry_validator is not None:
            valid, reason = geometry_validator(checked_states, checked_controls)
            if not isinstance(valid, (bool, np.bool_)) or not valid:
                raise ValueError(str(reason) or 'geometry validation failed')
        return PlanValidation(True, '', checked_controls, checked_states, distance)
    except (ValueError, TypeError, OverflowError, np.linalg.LinAlgError) as error:
        return PlanValidation(False, str(error))
