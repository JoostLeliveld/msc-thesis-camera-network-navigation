"""The mission identity and geometry are an indivisible versioned payload."""
from dataclasses import replace
import json

import pytest

from unav_common.mission_goal import (
    make_mission_goal, mission_goal_from_json, mission_goal_to_json, parse_mission_waypoints,
)


def goal(index=0, epoch='mission-a'):
    return make_mission_goal(mission_epoch=epoch, stamp_ns=10_123_456_789,
                             frame_id='map_bev', waypoints=((1., 2.), (1., 2.)), tour_index=index)


def test_serialization_keeps_nanosecond_stamp_identity_and_coordinates_together():
    before = goal()
    after = mission_goal_from_json(mission_goal_to_json(before))
    assert after == before
    assert after.stamp_ns == 10_123_456_789
    assert after.identity_payload == before.identity_payload
    assert not after.is_final


def test_repeated_coordinate_at_another_tour_index_is_a_new_goal():
    first, final = goal(), goal(1)
    assert (first.x, first.y) == (final.x, final.y)
    assert first.goal_id != final.goal_id
    assert final.is_final
    assert goal(epoch='mission-b').goal_id != first.goal_id


def test_same_id_cannot_hide_changed_geometry_from_consumer():
    first = goal()
    changed = replace(first, x=first.x+1.)
    assert changed.goal_id == first.goal_id
    assert changed.identity_payload != first.identity_payload
    assert replace(first, status='cancelled').identity_payload == first.identity_payload


def test_cancellation_time_is_separate_from_immutable_goal_creation_time():
    first = goal()
    cancelled = replace(first, status='cancelled', reason='clock_rewind_requires_restart',
                        status_stamp_ns=1_000_000_000)
    parsed = mission_goal_from_json(mission_goal_to_json(cancelled))
    assert parsed == cancelled
    assert parsed.identity_payload == first.identity_payload
    assert parsed.stamp_ns == 10_123_456_789
    assert parsed.status_stamp_ns == 1_000_000_000


def test_json_integer_coordinates_are_normalized_for_ros_geometry_fields():
    payload = json.loads(mission_goal_to_json(goal()))
    payload.update(x=1, y=2)
    parsed = mission_goal_from_json(json.dumps(payload))
    assert type(parsed.x) is float and type(parsed.y) is float


@pytest.mark.parametrize('change', [
    {'schema': 'mission_goal.v99'}, {'x': float('nan')}, {'y': True},
    {'goal_id': 'some-other-goal'}, {'tour_index': -1}, {'tour_count': 0},
    {'tour_index': True}, {'is_final': True}, {'frame_id': ''},
    {'goal_stamp': {'sec': 10, 'nanosec': 1_000_000_000}},
    {'goal_stamp': {'sec': -1, 'nanosec': 0}}, {'status': 'unknown'},
    {'goal_stamp': {'sec': 2_147_483_648, 'nanosec': 0}}, {'x': 10**500},
    {'status_stamp_ns': -1}, {'status_stamp_ns': True},
])
def test_invalid_goal_envelope_is_rejected(change):
    payload = json.loads(mission_goal_to_json(goal()))
    payload.update(change)
    with pytest.raises(ValueError):
        mission_goal_from_json(json.dumps(payload))


@pytest.mark.parametrize('raw', [
    '[[1,2],[3]]', '[[1,2],null]', '[[1,2],[3,4,5]]', '[[1,2],[NaN,4]]',
    '[[1,2],[true,4]]', '[]', '{}', 'not json',
])
def test_malformed_tour_never_returns_a_partial_mission(raw):
    with pytest.raises(ValueError):
        parse_mission_waypoints(raw, 10., 20.)


def test_empty_parameter_selects_validated_single_goal_and_retains_tour_order():
    assert parse_mission_waypoints('', 10., 20.) == ((10., 20.),)
    assert parse_mission_waypoints('[[2,1],[1,2],[2,1]]', 10., 20.) == ((2., 1.), (1., 2.), (2., 1.))
    with pytest.raises(ValueError):
        parse_mission_waypoints('', float('inf'), 0.)
