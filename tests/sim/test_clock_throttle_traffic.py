from types import SimpleNamespace

from sim import clock_throttle_node as clock


def test_paused_clock_has_one_hz_heartbeat_but_changes_and_resets_are_not_dropped(monkeypatch):
    node = object.__new__(clock.ClockThrottleNode)
    node._latest_msg = None
    node._last_published_stamp = None
    node._last_publish_wall_s = -float('inf')
    node._published_count = 0
    node.duplicate_heartbeat_s = 1.0
    messages = []
    node._pub = SimpleNamespace(publish=messages.append)
    node.get_logger = lambda: SimpleNamespace(info=lambda _: None)
    now = [0.]
    monkeypatch.setattr(clock.time, 'monotonic', lambda: now[0])
    node._publish_latest()
    assert not messages
    node._latest_msg = SimpleNamespace(clock=SimpleNamespace(sec=10, nanosec=0))
    for i in range(100):
        now[0] = i / 50.
        node._publish_latest()
    assert len(messages) == 2
    for sec, ns in ((10, 1), (1, 0)):
        node._latest_msg = SimpleNamespace(clock=SimpleNamespace(sec=sec, nanosec=ns))
        node._publish_latest()
    assert [(m.clock.sec, m.clock.nanosec) for m in messages] == [(10, 0), (10, 0), (10, 1), (1, 0)]
