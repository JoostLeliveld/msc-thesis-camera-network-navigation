from types import SimpleNamespace

from planning.nodes.unicycle_planner_node import UnicyclePlannerNode


def test_status_changes_empty_status_and_clock_reset_are_immediate():
    node = object.__new__(UnicyclePlannerNode)
    now = [10.]
    messages = []
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=int(now[0]*1e9)))
    node.planner_diag_text_pub = SimpleNamespace(publish=lambda m: messages.append(m.data))
    for i in range(8):
        now[0] = 10. + i*.25
        node._publish_planner_status_text('safe prefix')
    assert messages == ['safe prefix', 'safe prefix']
    node._publish_planner_status_text('stopped')
    node._publish_planner_status_text('')
    assert messages[-2:] == ['stopped', '']
    now[0] = 0.
    node._publish_planner_status_text('')
    assert messages[-2:] == ['', '']
