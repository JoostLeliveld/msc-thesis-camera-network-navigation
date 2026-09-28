"""Rollout helpers."""

import numpy as np

from planning.core.dynamics import unicycle_step


def rollout_unicycle(state0, controls, dt):
    """Roll out a sequence of unicycle controls. Returns array of states (H+1,3)."""
    states = [np.asarray(state0, dtype=float)]
    m = np.asarray(state0, dtype=float)
    for u in controls:
        m = unicycle_step(m, u, dt)
        states.append(m)
    return np.asarray(states, dtype=float)
