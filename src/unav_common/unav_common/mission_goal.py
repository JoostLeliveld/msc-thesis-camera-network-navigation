"""Atomic, ROS-independent mission goal transport.

The identity belongs to a mission epoch and tour index. Repeated publication
does not create a new goal or restart a success hold. Coordinates and identity
travel in one payload; /goal_bev is a compatibility view of the same goal.
"""
from dataclasses import dataclass
import json
import math


MISSION_GOAL_TOPIC = '/mission/goal_state'
MISSION_GOAL_SCHEMA = 'mission_goal.v1'


def _finite_number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{field} must be a finite number')
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f'{field} must be a finite number') from exc
    if not math.isfinite(number):
        raise ValueError(f'{field} must be a finite number')
    return number


def _integer(value, field, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{field} must be an integer >= {minimum}')
    return value


def parse_mission_waypoints(text, goal_x, goal_y):
    """Validate a whole tour before returning it; invalid input never truncates it."""
    if not isinstance(text, str):
        raise ValueError('waypoints_json must be a string')
    if not text.strip():
        raw = [[goal_x, goal_y]]
    else:
        try:
            raw = json.loads(text)
        except (TypeError, ValueError) as exc:
            raise ValueError('waypoints_json must be a JSON array of [x,y] points') from exc
    if not isinstance(raw, list) or not raw:
        raise ValueError('mission must contain at least one waypoint')
    points = []
    for index, point in enumerate(raw):
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f'waypoint {index} must contain exactly [x,y]')
        points.append(tuple(_finite_number(v, f'waypoint {index} {axis}')
                            for axis, v in zip(('x', 'y'), point)))
    return tuple(points)


@dataclass(frozen=True)
class MissionGoal:
    mission_epoch: str
    goal_id: str
    stamp_sec: int
    stamp_nanosec: int
    frame_id: str
    x: float
    y: float
    tour_index: int
    tour_count: int
    is_final: bool
    status: str = 'active'
    reason: str = ''
    status_stamp_ns: int | None = None

    def __post_init__(self):
        for name in ('mission_epoch', 'goal_id', 'frame_id'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f'{name} must be a nonempty string without surrounding whitespace')
        _integer(self.stamp_sec, 'stamp_sec')
        if self.stamp_sec > 2_147_483_647:
            raise ValueError('stamp_sec exceeds the ROS Time message range')
        _integer(self.stamp_nanosec, 'stamp_nanosec')
        if self.stamp_nanosec >= 1_000_000_000:
            raise ValueError('stamp_nanosec must be below one second')
        _integer(self.tour_index, 'tour_index')
        _integer(self.tour_count, 'tour_count', minimum=1)
        if self.tour_index >= self.tour_count:
            raise ValueError('tour_index is outside the mission')
        if type(self.is_final) is not bool or self.is_final != (self.tour_index == self.tour_count - 1):
            raise ValueError('is_final must describe the last tour index')
        if self.goal_id != f'{self.mission_epoch}:{self.tour_index}':
            raise ValueError('goal_id must identify its mission epoch and tour index')
        # ROS geometry fields require floats; JSON integers are valid inputs.
        object.__setattr__(self, 'x', _finite_number(self.x, 'x'))
        object.__setattr__(self, 'y', _finite_number(self.y, 'y'))
        if self.status not in ('active', 'completed', 'cancelled'):
            raise ValueError('unsupported mission goal status')
        if not isinstance(self.reason, str):
            raise ValueError('reason must be a string')
        if self.status_stamp_ns is not None:
            _integer(self.status_stamp_ns, 'status_stamp_ns')

    @property
    def stamp_ns(self):
        return self.stamp_sec * 1_000_000_000 + self.stamp_nanosec

    @property
    def identity_payload(self):
        """Fields fixed under one goal_id; status, reason and its time can change."""
        return (self.mission_epoch, self.goal_id, self.stamp_ns, self.frame_id,
                self.x, self.y, self.tour_index, self.tour_count, self.is_final)


def make_mission_goal(*, mission_epoch, stamp_ns, frame_id, waypoints, tour_index, reason=''):
    _integer(stamp_ns, 'stamp_ns')
    _integer(tour_index, 'tour_index')
    if not waypoints or tour_index >= len(waypoints):
        raise ValueError('tour_index is outside the mission')
    x, y = waypoints[tour_index]
    sec, nanosec = divmod(stamp_ns, 1_000_000_000)
    return MissionGoal(mission_epoch, f'{mission_epoch}:{tour_index}', sec, nanosec,
                       frame_id, x, y, tour_index, len(waypoints),
                       tour_index == len(waypoints) - 1, reason=reason, status_stamp_ns=stamp_ns)


def mission_goal_to_json(goal):
    if not isinstance(goal, MissionGoal):
        raise TypeError('expected MissionGoal')
    return json.dumps(dict(schema=MISSION_GOAL_SCHEMA, mission_epoch=goal.mission_epoch,
        goal_id=goal.goal_id, goal_stamp=dict(sec=goal.stamp_sec, nanosec=goal.stamp_nanosec),
        frame_id=goal.frame_id, x=goal.x, y=goal.y, tour_index=goal.tour_index,
        tour_count=goal.tour_count, is_final=goal.is_final, status=goal.status, reason=goal.reason,
        status_stamp_ns=goal.status_stamp_ns),
        allow_nan=False, sort_keys=True, separators=(',', ':'))


def mission_goal_from_json(text):
    try:
        data = json.loads(text)
        if not isinstance(data, dict) or data.get('schema') != MISSION_GOAL_SCHEMA:
            raise ValueError('unsupported mission goal schema')
        stamp = data['goal_stamp']
        return MissionGoal(data['mission_epoch'], data['goal_id'], stamp['sec'], stamp['nanosec'],
            data['frame_id'], data['x'], data['y'], data['tour_index'], data['tour_count'],
            data['is_final'], data['status'], data['reason'], data.get('status_stamp_ns'))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'invalid mission goal: {exc}') from exc
