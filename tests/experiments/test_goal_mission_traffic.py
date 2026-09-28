from types import SimpleNamespace
import json

from builtin_interfaces.msg import Time as TimeMsg
from std_msgs.msg import String

from experiments.nodes.goal_mission_node import GoalMissionNode


class _Duration:
    def __init__(self, nanoseconds):
        self.nanoseconds = nanoseconds


class _Time:
    def __init__(self, seconds):
        self.seconds = float(seconds)
        self.nanoseconds = int(round(self.seconds * 1e9))

    def __sub__(self, other):
        return _Duration(self.nanoseconds - other.nanoseconds)

    def to_msg(self):
        seconds = int(self.seconds)
        return TimeMsg(sec=seconds, nanosec=int(round((self.seconds - seconds) * 1e9)))


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


class _Logger:
    def info(self, _message):
        pass


def _mission(*, repeat_unchanged_goal=False):
    node = object.__new__(GoalMissionNode)
    wall = {'seconds': 10.0}
    node.wall_clock = SimpleNamespace(now=lambda: _Time(wall['seconds']))
    node.get_clock = lambda: node.wall_clock
    node.start_time = _Time(0.0)
    node.delay = 0.0
    node.sent_count = 0
    node.repeat_count = 0
    node.repeat_unchanged_goal = repeat_unchanged_goal
    node._last_published_wp_idx = None
    node.wait_for_belief = False
    node._belief_ready = False
    node._belief_xy = None
    node._belief_wait_logged = False
    node.waypoints = [(1.0, 2.0), (3.0, 4.0)]
    node.wp_idx = 0
    node.arrival_radius = 0.2
    node.frame_id = 'map_bev'
    node.initial_belief_max_sigma_m = 0.0
    node.operational_belief_timeout_s = 0.5
    node._initialize_mission_state()
    node.goal_pub = _Publisher()
    node.goal_state_pub = _Publisher()
    node.get_logger = lambda: _Logger()
    return node, wall


def test_unchanged_transient_local_goal_is_published_once():
    node, wall = _mission()

    for seconds in (10.0, 11.0, 12.0, 13.0):
        wall['seconds'] = seconds
        node._send_goal()

    assert node.sent_count == 1
    assert len(node.goal_pub.messages) == 1
    assert node.goal_pub.messages[0].pose.position.x == 1.0


def test_new_waypoint_is_published_immediately_after_advancing():
    node, _wall = _mission()
    node._send_goal()
    # Progress now consumes the complete operational state; a raw XY tuple is
    # intentionally insufficient. Keep the original traffic assertions below.
    belief = String()
    belief.data = json.dumps(dict(schema_version=1, epoch='traffic-test', revision=1,
        initialized=True, frame_id='map_bev', anchor_stamp_ns=10_000_000_000,
        state_stamp_ns=10_000_000_000, mean=[1.05, 2.0, 0.0],
        covariance=[[.01, 0., 0.], [0., .01, 0.], [0., 0., .01]],
        valid=True, invalid_reason='', motion_supported=True,
        motion_support=dict(start_stamp_ns=10_000_000_000, end_stamp_ns=10_000_000_000,
                            source='none', supported=True, gaps=[])))
    node._belief_cb(belief)

    node._send_goal()
    node._send_goal()

    assert node.sent_count == 2
    assert [msg.pose.position.x for msg in node.goal_pub.messages] == [1.0, 3.0]


def test_explicit_repeat_mode_preserves_periodic_publication():
    node, _wall = _mission(repeat_unchanged_goal=True)

    node._send_goal()
    node._send_goal()

    assert node.sent_count == 2
    assert len(node.goal_pub.messages) == 2
