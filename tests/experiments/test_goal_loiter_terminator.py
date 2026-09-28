"""The goal loiter terminator closes the dead zone where stuck detection is off."""
from __future__ import annotations

from types import SimpleNamespace

from experiments.nodes.experiment_logger import ExperimentLogger


class _Log:
    def info(self, _message):
        pass


def logger(**overrides):
    value = SimpleNamespace(
        _mission_goal_state=None, auto_stop_on_goal=True, goal_msg=object(),
        goal_success_radius=0.10, goal_success_hold_s=2.0,
        goal_stable_radius=0.20, goal_stable_hold_s=2.0, goal_stable_max_displacement_m=0.04,
        goal_loiter_timeout_s=15.0,
        _goal_in_radius_since=None, _goal_stable_since=None, _goal_loiter_since=None,
        finished=[],
    )
    value.__dict__.update(overrides)
    value.get_logger = lambda: _Log()
    value._motion_window_stats = lambda stamp, window: {'displacement_m': 1.0}
    value._command_active = lambda v, w: True
    value._finish_run = lambda reason, stamp: value.finished.append((reason, stamp))
    return value


def step(value, stamp, dist):
    return ExperimentLogger._maybe_finish_for_goal(value, stamp, dist, 0.3, 0.0)


def test_moving_inside_stable_radius_but_outside_stop_radius_ends_as_loiter():
    value = logger()
    for i in range(0, 161):
        if step(value, i * 0.1, 0.15):
            break
    assert value.finished == [("goal_loiter_timeout", 15.0)]


def test_leaving_the_stable_radius_restarts_the_loiter_clock():
    value = logger()
    for i in range(0, 100):
        step(value, i * 0.1, 0.15)
    step(value, 10.0, 0.25)
    for i in range(101, 200):
        step(value, i * 0.1, 0.15)
    assert value.finished == []


def test_a_goal_hold_inside_the_stop_radius_still_wins():
    value = logger()
    for i in range(0, 40):
        if step(value, i * 0.1, 0.05):
            break
    assert value.finished == [("goal_reached", 2.0)]
