"""Checked local action selection; no estimator, map, or clearance relaxation."""
from dataclasses import dataclass
from enum import Enum

import numpy as np

from planning.core.dynamics import unicycle_step


class SafetyFailure(str, Enum):
    NONE = "none"
    INVALID_INPUT = "invalid_input"
    INVALID_GEOMETRY = "invalid_geometry"
    COLLISION = "collision"
    DRIVEABLE_CLEARANCE = "driveable_clearance"


@dataclass(frozen=True)
class ControlSafetyResult:
    """Safety policy is carried by ``failure``; ``reason`` is diagnostic text.

    Two-value unpacking is retained for the saved tracker inspection probes.
    Recovery never infers authority from a formatted message.
    """
    safe_steps: int
    reason: str = ""
    failure: SafetyFailure = SafetyFailure.NONE

    def __iter__(self):
        yield self.safe_steps
        yield self.reason

    @property
    def permits_rotation_attempt(self):
        return self.safe_steps == 0 and self.failure in (
            SafetyFailure.COLLISION, SafetyFailure.DRIVEABLE_CLEARANCE,
        )


@dataclass(frozen=True)
class GuardedControls:
    controls: np.ndarray
    safe_steps: int
    reason: str
    rotation_recovery: bool = False


def checked_tracker_controls(controls, state, target, *, dt, w_min, w_max,
                             safety_check, allow_rotation_recovery=False):
    """Try a bounded stationary turn only after the proposed first step is refused.

    Keep the same waypoint and the same gate. A blocked translation while already
    aligned remains blocked; this is not a route search or a collision exemption.
    The gate must model the robot's footprint for rotation as well as translation.
    The active platform uses a swept oriented rectangular footprint.
    """
    controls = np.asarray(controls, dtype=float)
    safety = safety_check(controls, state)
    original = GuardedControls(controls, safety.safe_steps, safety.reason)
    if not allow_rotation_recovery or not safety.permits_rotation_attempt:
        return original
    pose = np.asarray(state, dtype=float).copy()
    target = np.asarray(target, dtype=float)
    if (pose.shape != (3,) or target.shape != (2,) or controls.ndim != 2
            or controls.shape[1] != 2 or not np.isfinite(pose).all()
            or not np.isfinite(target).all()):
        return original
    turn = np.zeros_like(controls)
    for i in range(len(turn)):
        desired = np.arctan2(target[1]-pose[1], target[0]-pose[0])
        delta = np.arctan2(np.sin(desired-pose[2]), np.cos(desired-pose[2]))
        turn[i, 1] = np.clip(2.*delta, w_min, w_max)
        pose = unicycle_step(pose, turn[i], dt)
    if not len(turn) or abs(turn[0, 1]) < 1e-3:
        return original
    turn_safety = safety_check(turn, state)
    if turn_safety.safe_steps <= 0:
        return original
    return GuardedControls(turn, turn_safety.safe_steps, 'rotation_recovery_after:'+safety.reason, True)
