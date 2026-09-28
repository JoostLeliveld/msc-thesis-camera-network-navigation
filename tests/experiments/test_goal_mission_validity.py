"""Mission decisions with deterministic clocks, no DDS or simulator process."""
import json
import threading
from types import SimpleNamespace

import pytest
from std_msgs.msg import String

from experiments.nodes.goal_mission_node import GoalMissionNode
from unav_common.mission_goal import mission_goal_from_json


class _Time:
    def __init__(self, ns):
        self.nanoseconds = ns

    def __sub__(self, other):
        return _Time(self.nanoseconds - other.nanoseconds)


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def _mission(*, waypoints=((1., 0.), (2., 0.)), wait=False, repeat=False, sigma_max=0.):
    node = object.__new__(GoalMissionNode)
    clock = dict(ros=10_000_000_003, wall=1_700_000_000_000_000_000)
    node.get_clock = lambda: SimpleNamespace(now=lambda: _Time(clock['ros']))
    node.wall_clock = SimpleNamespace(now=lambda: _Time(clock['wall']))
    node.start_time = _Time(clock['wall'] - 5_000_000_000)
    node.delay = 0.
    node.repeat_count = 0
    node.repeat_unchanged_goal = repeat
    node.wait_for_belief = wait
    node.initial_belief_max_sigma_m = sigma_max
    node.operational_belief_timeout_s = .5
    node.waypoints = waypoints
    node.arrival_radius = .2
    node.frame_id = 'map_bev'
    node._initialize_mission_state()
    node._fixture_anchor_stamps = {}
    node.goal_pub = _Publisher()
    node.goal_state_pub = _Publisher()
    node.get_logger = lambda: SimpleNamespace(info=lambda _text: None)
    return node, clock


def _belief(node, clock, *, xy=(1., 0.), **changes):
    epoch = changes.get('epoch', 'producer-a')
    anchor = node._fixture_anchor_stamps.setdefault(epoch, clock['ros'] - 100_000_000)
    data = dict(schema_version=1, epoch='producer-a', revision=1, initialized=True,
                frame_id='map_bev', anchor_stamp_ns=anchor,
                state_stamp_ns=clock['ros'], mean=[*xy, 0.],
                covariance=[[.01, 0., 0.], [0., .01, 0.], [0., 0., .01]],
                valid=True, invalid_reason='', motion_supported=True)
    data.update(changes)
    data.setdefault('motion_support', dict(start_stamp_ns=data['anchor_stamp_ns'],
        end_stamp_ns=data['state_stamp_ns'], source='odom', supported=True, gaps=[]))
    message = String()
    message.data = json.dumps(data)
    node._belief_cb(message)


def _goals(node):
    return [mission_goal_from_json(msg.data) for msg in node.goal_state_pub.messages]


@pytest.mark.parametrize('timeout', [0., -1., float('nan'), float('inf')])
def test_mission_delivery_deadline_must_be_finite_and_positive(timeout):
    node, _clock = _mission()
    node.operational_belief_timeout_s = timeout
    with pytest.raises(ValueError, match='finite and positive'):
        node._initialize_mission_state()


def test_goal_geometry_identity_and_creation_stamp_are_atomic_and_use_ros_time():
    node, clock = _mission(repeat=True)
    node._send_goal()
    original = _goals(node)[0]
    clock['ros'] += 1_000_000_000
    clock['wall'] += 3_000_000_000
    node._send_goal()
    assert _goals(node) == [original, original]
    assert original.stamp_ns == 10_000_000_003
    assert original.stamp_ns != clock['wall']
    assert not original.is_final
    for pose in node.goal_pub.messages:
        assert pose.header.stamp.sec * 1_000_000_000 + pose.header.stamp.nanosec == original.stamp_ns
        assert (pose.pose.position.x, pose.pose.position.y) == (original.x, original.y)
        assert pose.header.frame_id == original.frame_id


def test_startup_wall_delay_does_not_change_goal_time_domain():
    node, clock = _mission()
    node.delay = 6.
    node._send_goal()
    assert not node.goal_pub.messages
    clock['wall'] += 1_000_000_000
    node._send_goal()
    assert _goals(node)[0].stamp_ns == clock['ros']


@pytest.mark.parametrize('changes', [
    {'frame_id': 'odom'},
    {'mean': [1., 0., float('nan')]},
    {'covariance': [[.01, .1, 0.], [.1, .01, 0.], [0., 0., .01]]},
    {'covariance': [[.01, .1, 0.], [0., .01, 0.], [0., 0., .01]]},
    {'valid': False, 'invalid_reason': 'producer_invalid'},
    {'motion_supported': False}, {'motion_support': None},
    {'state_stamp_ns': 9_000_000_003, 'anchor_stamp_ns': 9_000_000_003},
    {'state_stamp_ns': 11_000_000_003},
])
def test_unusable_update_clears_readiness_and_cannot_advance_near_waypoint(changes):
    node, clock = _mission(wait=True)
    _belief(node, clock)
    assert node._belief_ready
    node._send_goal()
    _belief(node, clock, revision=2, **changes)
    node._send_goal()
    assert not node._belief_ready
    assert node._belief_xy is None
    assert node.wp_idx == 0
    assert len(node.goal_pub.messages) == 1


def test_held_belief_expires_without_callback_before_first_goal():
    node, clock = _mission(wait=True)
    _belief(node, clock)
    assert node._belief_ready
    clock['ros'] += 500_000_001
    node._send_goal()
    assert not node._belief_ready
    assert not node.goal_pub.messages


def test_old_camera_anchor_with_fresh_supported_prediction_releases_goal():
    node, clock = _mission(wait=True)
    _belief(node, clock, anchor_stamp_ns=0)
    node._send_goal()
    assert node._belief_ready
    assert len(node.goal_pub.messages) == 1


def test_actual_prediction_producer_payload_releases_then_invalidates_mission():
    from planning.core.belief_state import BeliefRecord, MotionSupport, PredictionSnapshot

    node, clock = _mission(wait=True)
    covariance = [[.01, 0., 0.], [0., .01, 0.], [0., 0., .01]]
    anchor = BeliefRecord.create([0., 0., 0.], covariance, 0, 'map_bev', 'producer-a', 1)
    prediction = PredictionSnapshot.create(anchor, [0., 0., 0.], covariance,
        clock['ros'], MotionSupport(0, clock['ros'], 'odom'))
    message = String()
    message.data = json.dumps(prediction.to_dict(), allow_nan=False)
    node._belief_cb(message)
    node._send_goal()
    assert node._belief_ready
    assert len(node.goal_pub.messages) == 1
    clock['ros'] += 100_000_000
    unsupported = MotionSupport(0, clock['ros'], 'odom', ((0, 1, 'missing_prefix'),))
    invalid = PredictionSnapshot.create(anchor, [1., 0., 0.], covariance,
        clock['ros'], unsupported, valid=False, invalid_reason='unsupported_motion')
    message.data = json.dumps(invalid.to_dict(), allow_nan=False)
    node._belief_cb(message)
    node._send_goal()
    assert not node._belief_ready
    assert node.wp_idx == 0


def test_tour_announces_every_short_duplicate_and_sharp_turn_target_in_order():
    points = ((1., 0.), (1., 0.), (1.1, 0.), (1.1, 1.))
    node, clock = _mission(waypoints=points)
    _belief(node, clock)
    node._maybe_advance()
    assert node.wp_idx == 0
    node._send_goal()
    for _ in range(4):
        node._send_goal()
    assert node.wp_idx == 0  # Held pre-goal sample cannot consume nearby targets.
    for index in range(len(points) - 1):
        clock['ros'] += 100_000_000
        _belief(node, clock, xy=points[index])  # New prediction, same anchor revision.
        node._send_goal()
        assert node.wp_idx == index + 1
        node._send_goal()
        assert len(node.goal_pub.messages) == index + 2
    goals = _goals(node)
    assert [(g.x, g.y) for g in goals] == list(points)
    assert [g.tour_index for g in goals] == [0, 1, 2, 3]
    assert [g.is_final for g in goals] == [False, False, False, True]
    assert len({g.goal_id for g in goals}) == 4


def test_new_correction_at_same_prediction_time_can_advance_published_goal():
    node, clock = _mission()
    _belief(node, clock, xy=(0., 0.))
    node._send_goal()
    _belief(node, clock, revision=2)
    node._send_goal()
    assert node.wp_idx == 1


def test_valid_belief_older_than_goal_creation_cannot_advance():
    node, clock = _mission()
    node._send_goal()
    _belief(node, clock, state_stamp_ns=clock['ros'] - 1)
    node._send_goal()
    assert node._belief_ready
    assert node.wp_idx == 0


@pytest.mark.parametrize('x, advances', [(1.199999, True), (1.2, True), (1.200001, False)])
def test_intermediate_arrival_boundary_uses_unchanged_radius(x, advances):
    node, clock = _mission()
    node._send_goal()
    _belief(node, clock, xy=(x, 0.))
    node._send_goal()
    assert node.wp_idx == int(advances)


@pytest.mark.parametrize('x', [1., 1.199999, 1.200001])
def test_final_target_is_held_for_outcome_owner_inside_or_outside_radius(x):
    node, clock = _mission(waypoints=((1., 0.),))
    node._send_goal()
    _belief(node, clock, xy=(x, 0.))
    node._send_goal()
    assert node.wp_idx == 0
    assert len(node.goal_pub.messages) == 1
    assert _goals(node)[0].is_final
    assert _goals(node)[0].status == 'active'  # Mission never asserts logger success.


def test_initial_sigma_release_limit_is_not_a_new_progress_threshold():
    node, clock = _mission(wait=True, sigma_max=.2)
    _belief(node, clock, xy=(0., 0.))
    node._send_goal()
    _belief(node, clock, revision=2,
            covariance=[[1., 0., 0.], [0., 1., 0.], [0., 0., .01]])
    node._send_goal()
    assert not node._belief_ready
    assert node.wp_idx == 1


def test_clock_rewind_latches_mission_even_when_a_new_belief_producer_appears():
    node, clock = _mission(wait=True)
    _belief(node, clock, xy=(0., 0.))
    node._send_goal()
    _belief(node, clock, revision=2)
    node._send_goal()
    before = _goals(node)[-1]
    assert before.tour_index == 1
    clock['ros'] = 1_000_000_000
    node._send_goal()
    assert len(_goals(node)) == 3
    cancelled = _goals(node)[-1]
    assert cancelled.identity_payload == before.identity_payload
    assert cancelled.status == 'cancelled'
    assert cancelled.status_stamp_ns == clock['ros']
    assert cancelled.reason == 'mission_clock_rewind_requires_restart'
    assert len(node.goal_pub.messages) == 2
    assert not node._belief_ready
    _belief(node, clock, revision=3)
    node._send_goal()
    assert len(_goals(node)) == 3
    _belief(node, clock, epoch='producer-b')
    node._send_goal()
    assert len(_goals(node)) == 3
    assert not node._belief_ready
    assert node._belief is None
    assert node._mission_invalid_reason == 'mission_clock_rewind_requires_restart'
    assert node.wp_idx == before.tour_index
    assert node._mission_epoch == before.mission_epoch


def test_new_mission_instance_after_restart_requires_fresh_belief_and_has_new_identity():
    previous, clock = _mission(wait=True)
    _belief(previous, clock)
    previous._send_goal()
    original = _goals(previous)[0]
    clock['ros'] = 1_000_000_000
    previous._send_goal()
    restarted, restarted_clock = _mission(wait=True, waypoints=previous.waypoints)
    restarted_clock['ros'] = clock['ros']
    restarted._send_goal()
    assert not restarted.goal_pub.messages
    _belief(restarted, restarted_clock, epoch='producer-b')
    restarted._send_goal()
    current = _goals(restarted)[0]
    assert current.mission_epoch != original.mission_epoch
    assert current.tour_index == 0
    assert current.stamp_ns == restarted_clock['ros']


def test_producer_restart_without_clock_rewind_cannot_resume_old_mission():
    node, clock = _mission(wait=True)
    _belief(node, clock, xy=(0., 0.))
    node._send_goal()
    _belief(node, clock, epoch='producer-b')
    node._send_goal()
    assert node._mission_invalid_reason == 'belief_epoch_changed_requires_restart'
    assert not node._belief_ready
    assert node._belief is None
    assert node.wp_idx == 0
    assert len(_goals(node)) == 2
    assert _goals(node)[-1].status == 'cancelled'
    assert len(node.goal_pub.messages) == 1


def test_rewind_before_first_publication_records_only_cancelled_configured_target():
    node, clock = _mission(wait=True)
    node._send_goal()
    assert not node.goal_pub.messages
    clock['ros'] = 1_000_000_000
    node._send_goal()
    assert not node.goal_pub.messages
    assert node.sent_count == 0
    cancelled = _goals(node)[0]
    assert cancelled.status == 'cancelled'
    assert cancelled.reason == 'before_first_publication:mission_clock_rewind_requires_restart'
    assert cancelled.status_stamp_ns == clock['ros']


def test_failed_cancellation_retries_same_event_once_and_stays_invalid():
    node, clock = _mission()
    node._send_goal()
    real_publish = node.goal_state_pub.publish
    failed = []

    def fail(message):
        failed.append(message.data)
        raise RuntimeError('transport unavailable')

    node.goal_state_pub.publish = fail
    clock['ros'] = 1_000_000_000
    node._send_goal()
    assert node._mission_invalid_reason
    assert not node._mission_cancellation_published
    node.goal_state_pub.publish = real_publish
    clock['ros'] += 100_000_000
    node._send_goal()
    node._send_goal()
    assert node._mission_cancellation_published
    assert len(_goals(node)) == 2
    assert node.goal_state_pub.messages[-1].data == failed[0]
    assert _goals(node)[-1].status_stamp_ns == 1_000_000_000
    assert len(node.goal_pub.messages) == 1


def test_failed_compatibility_publication_retries_same_identity_without_skipping():
    node, clock = _mission()
    real_publish = node.goal_pub.publish
    node.goal_pub.publish = lambda _message: (_ for _ in ()).throw(RuntimeError('publish failed'))
    with pytest.raises(RuntimeError, match='publish failed'):
        node._send_goal()
    first = _goals(node)[0]
    _belief(node, clock)
    node.goal_pub.publish = real_publish
    node._send_goal()
    assert node.wp_idx == 0
    assert _goals(node) == [first, first]
    assert node.sent_count == 1


def test_belief_callback_during_publication_cannot_mix_goal_snapshot():
    node, clock = _mission()
    in_publication, release, callback_started, callback_done = [threading.Event() for _ in range(4)]
    failures = []
    publish = node.goal_state_pub.publish

    def blocked_publish(message):
        publish(message)
        in_publication.set()
        assert release.wait(2)

    def send():
        try:
            node._send_goal()
        except Exception as exc:
            failures.append(exc)

    def update():
        callback_started.set()
        try:
            _belief(node, clock)
        except Exception as exc:
            failures.append(exc)
        finally:
            callback_done.set()

    node.goal_state_pub.publish = blocked_publish
    sending = threading.Thread(target=send)
    updating = threading.Thread(target=update)
    sending.start()
    assert in_publication.wait(2)
    updating.start()
    assert callback_started.wait(2)
    assert not callback_done.is_set()
    release.set()
    sending.join(2)
    updating.join(2)
    assert not sending.is_alive() and not updating.is_alive()
    assert not failures
    goal = _goals(node)[0]
    pose = node.goal_pub.messages[0]
    assert goal.tour_index == node.wp_idx == 0
    assert (pose.pose.position.x, pose.pose.position.y) == (goal.x, goal.y)
    assert callback_done.is_set()
