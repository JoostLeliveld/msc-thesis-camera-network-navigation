from types import SimpleNamespace
import pytest
from sim.actuation_noise_node import ActuationNoiseNode


def node_at(now):
    node = object.__new__(ActuationNoiseNode)
    node._last_input_stamp_s = 10.
    node._watchdog_stopped = False
    node.command_timeout_s = .5
    node._epoch_rewind_latched = False
    node._linear_slip_state = 1.0
    node._angular_slip_state = 1.0
    messages = []
    node._pub = SimpleNamespace(publish=messages.append)
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=int(now*1e9)))
    node.get_logger = lambda: SimpleNamespace(warn=lambda _: None, error=lambda _: None)
    return node, messages


@pytest.mark.parametrize('now', [10.6, 9.])
def test_silence_or_clock_reset_stops_once(now):
    node, messages = node_at(now)
    node._watchdog_tick()
    node._watchdog_tick()
    assert len(messages) == 1
    assert messages[0].linear.x == 0 and messages[0].angular.z == 0


def test_recent_command_is_not_interrupted():
    node, messages = node_at(10.3)
    node._watchdog_tick()
    assert not messages


def test_clock_rewind_latches_adapter_and_clears_correlated_noise():
    node, messages = node_at(9.0)
    node._watchdog_tick()
    assert node._epoch_rewind_latched
    assert node._linear_slip_state == 0.0
    assert node._angular_slip_state == 0.0
    assert len(messages) == 1
