"""Deterministic consumer regressions for operational state, not camera age."""
import json
from dataclasses import FrozenInstanceError

import pytest

from unav_common.operational_belief import (
    OperationalBeliefReceiver, operational_belief_from_json,
    operational_belief_invalid_reason,
)


NOW = 10_000_000_000


def _payload(**changes):
    data = dict(schema_version=1, epoch='producer-a', revision=1, initialized=True,
                frame_id='map_bev', anchor_stamp_ns=NOW - 200_000_000,
                state_stamp_ns=NOW, mean=[1., 2., .3],
                covariance=[[.04, .01, 0.], [.01, .09, 0.], [0., 0., .01]],
                valid=True, invalid_reason='', motion_supported=True)
    data.update(changes)
    data.setdefault('motion_support', dict(start_stamp_ns=data['anchor_stamp_ns'],
        end_stamp_ns=data['state_stamp_ns'], source='odom', supported=True, gaps=[]))
    return data


def _send(receiver, *, now=NOW, **changes):
    return receiver.receive_json(json.dumps(_payload(**changes)), now_ns=now)


def test_full_covariance_is_retained_and_snapshot_is_immutable():
    receiver = OperationalBeliefReceiver()
    assert _send(receiver)
    belief = receiver.usable(now_ns=NOW)
    assert belief.covariance[0][1] == .01
    with pytest.raises(FrozenInstanceError):
        belief.revision = 8
    support = belief.motion_support
    support['gaps'].append({'reason': 'mutated'})
    assert belief.motion_support['gaps'] == []


def test_fresh_predicted_state_remains_usable_with_old_camera_anchor():
    receiver = OperationalBeliefReceiver()
    assert _send(receiver, anchor_stamp_ns=0)
    assert receiver.usable(now_ns=NOW) is not None


def test_state_age_boundary_and_expiry_without_another_callback():
    receiver = OperationalBeliefReceiver()
    assert _send(receiver)
    assert receiver.usable(now_ns=NOW + 500_000_000) is not None
    assert receiver.usable(now_ns=NOW + 500_000_001) is None
    assert receiver.reason == 'stale_belief_state'


@pytest.mark.parametrize('changes', [
    {'mean': [float('nan'), 0., 0.]},
    {'mean': [0., 0., float('inf')]},
    {'mean': [True, 0., 0.]},
    {'mean': ['1', 0., 0.]},
    {'mean': [0., 0.]},
    {'covariance': [[.1, .2, 0.], [.2, .1, 0.], [0., 0., .1]]},
    {'covariance': [[.1, .02, 0.], [0., .1, 0.], [0., 0., .1]]},
    {'covariance': [[.1, 0., 0.], [0., .1, 0.], [0., 0., -.01]]},
    {'covariance': [[.1, 0., 0.], [0., .1, 0.], [0., 0., float('inf')]]},
    {'covariance': [[1e308] * 3 for _ in range(3)]},
    {'covariance': [[.1, 0., 0.], [0., .1, 0.]]},
    {'frame_id': 'odom'}, {'frame_id': ''},
    {'motion_supported': False}, {'motion_supported': 'true'},
    {'valid': False, 'invalid_reason': 'motion_outage'},
    {'valid': 'true'}, {'invalid_reason': 'contradictory_invalid_state'},
    {'initialized': False}, {'initialized': 'true'},
    {'state_stamp_ns': NOW + 1}, {'anchor_stamp_ns': NOW + 1},
    {'state_stamp_ns': None}, {'schema_version': 2}, {'schema_version': True},
    {'revision': True}, {'epoch': ''},
])
def test_unusable_new_event_clears_previous_usable_state(changes):
    receiver = OperationalBeliefReceiver()
    assert _send(receiver)
    update = dict(revision=2)
    update.update(changes)
    _send(receiver, **update)
    assert receiver.usable(now_ns=NOW) is None
    assert receiver.reason


@pytest.mark.parametrize('support', [
    None, {}, [], 'supported',
    dict(start_stamp_ns=NOW - 200_000_000, end_stamp_ns=NOW, source='odom', gaps=[]),
    dict(start_stamp_ns=NOW - 200_000_000, end_stamp_ns=NOW, source='odom', supported=False, gaps=[]),
    dict(start_stamp_ns=NOW - 200_000_000, end_stamp_ns=NOW, source='odom', supported='true', gaps=[]),
    dict(start_stamp_ns=NOW - 200_000_000, end_stamp_ns=NOW, source='odom', supported=True),
    dict(start_stamp_ns=NOW - 200_000_000, end_stamp_ns=NOW, source='odom', supported=True, gaps=None),
    dict(start_stamp_ns=NOW - 200_000_000, end_stamp_ns=NOW, source='odom', supported=True,
         gaps=[dict(start_stamp_ns=NOW - 1, end_stamp_ns=NOW, reason='stale_motion')]),
    dict(start_stamp_ns=NOW - 199_999_999, end_stamp_ns=NOW, source='odom', supported=True, gaps=[]),
    dict(start_stamp_ns=NOW - 200_000_000, end_stamp_ns=NOW - 1, source='odom', supported=True, gaps=[]),
    dict(start_stamp_ns=True, end_stamp_ns=NOW, source='odom', supported=True, gaps=[]),
    dict(start_stamp_ns=-1, end_stamp_ns=NOW, source='odom', supported=True, gaps=[]),
    dict(start_stamp_ns=NOW - 200_000_000, end_stamp_ns=float(NOW), source='odom', supported=True, gaps=[]),
    dict(start_stamp_ns=NOW - 200_000_000, end_stamp_ns=NOW, source='', supported=True, gaps=[]),
    dict(start_stamp_ns=NOW - 200_000_000, end_stamp_ns=NOW, source='none', supported=True, gaps=[]),
])
def test_missing_or_contradictory_nested_motion_support_is_not_usable(support):
    receiver = OperationalBeliefReceiver()
    assert _send(receiver)
    _send(receiver, revision=2, motion_support=support)
    assert receiver.usable(now_ns=NOW) is None


def test_zero_interval_requires_no_motion_and_psd_zero_covariance_is_legal():
    data = _payload(anchor_stamp_ns=NOW, covariance=[[0.] * 3 for _ in range(3)])
    data['motion_support']['source'] = 'none'
    belief = operational_belief_from_json(json.dumps(data))
    assert operational_belief_invalid_reason(belief, now_ns=NOW, expected_frame='map_bev') == ''


def test_later_prediction_with_same_anchor_revision_and_new_correction_same_time():
    receiver = OperationalBeliefReceiver()
    assert _send(receiver)
    assert _send(receiver, now=NOW + 1, state_stamp_ns=NOW + 1, mean=[2., 2., .3])
    assert receiver.usable(now_ns=NOW + 1).mean[0] == 2.
    assert _send(receiver, now=NOW + 1, revision=2, state_stamp_ns=NOW + 1, mean=[3., 2., .3])
    assert receiver.usable(now_ns=NOW + 1).mean[0] == 3.


def test_older_revision_or_older_prediction_cannot_overwrite_current_state():
    receiver = OperationalBeliefReceiver()
    assert _send(receiver, revision=2)
    assert not _send(receiver, revision=1, mean=[8., 8., 0.])
    assert not _send(receiver, revision=2, state_stamp_ns=NOW - 1, mean=[9., 9., 0.])
    assert receiver.usable(now_ns=NOW).mean[0] == 1.


def test_higher_revision_with_regressing_target_invalidates():
    receiver = OperationalBeliefReceiver()
    assert _send(receiver)
    assert not _send(receiver, revision=2, state_stamp_ns=NOW - 1)
    assert receiver.usable(now_ns=NOW) is None
    assert receiver.reason == 'belief_state_time_regressed'


def test_same_anchor_revision_cannot_move_anchor_even_at_new_prediction_time():
    receiver = OperationalBeliefReceiver()
    assert _send(receiver)
    assert not _send(receiver, now=NOW + 1, state_stamp_ns=NOW + 1,
                     anchor_stamp_ns=NOW - 100_000_000)
    assert receiver.usable(now_ns=NOW + 1) is None
    assert receiver.reason == 'conflicting_belief_anchor_revision'


def test_conflicting_identity_invalidates_and_duplicate_cannot_revive_it():
    receiver = OperationalBeliefReceiver()
    assert _send(receiver)
    assert not _send(receiver, mean=[2., 2., .3])
    assert receiver.usable(now_ns=NOW) is None
    assert receiver.reason == 'conflicting_belief_revision'
    assert not _send(receiver)
    assert receiver.usable(now_ns=NOW) is None
    assert _send(receiver, revision=2)
    assert receiver.usable(now_ns=NOW) is not None


def test_malformed_higher_revision_cannot_be_replaced_by_an_older_delivery():
    receiver = OperationalBeliefReceiver()
    assert _send(receiver)
    assert not _send(receiver, revision=3, state_stamp_ns='bad')
    assert not _send(receiver, revision=2)
    assert receiver.usable(now_ns=NOW) is None
    assert _send(receiver, revision=4)


def test_invalid_prebootstrap_and_postbootstrap_events_keep_initialization_distinct():
    receiver = OperationalBeliefReceiver()
    invalid = dict(valid=False, invalid_reason='uninitialized', revision=0,
                   initialized=False, mean=None, covariance=None, anchor_stamp_ns=None,
                   state_stamp_ns=None, motion_supported=False, motion_support=None)
    assert _send(receiver, **invalid)
    assert not receiver.latest.initialized
    assert receiver.usable(now_ns=NOW) is None
    assert _send(receiver, revision=1)
    invalid.update(revision=2, initialized=True, invalid_reason='motion_outage')
    assert _send(receiver, **invalid)
    assert receiver.latest.initialized
    assert receiver.usable(now_ns=NOW) is None


def test_new_epoch_closes_previous_epoch_and_clock_rewind_requires_another_epoch():
    receiver = OperationalBeliefReceiver()
    assert _send(receiver)
    assert _send(receiver, epoch='producer-b')
    assert not _send(receiver, epoch='producer-a', revision=10)
    assert receiver.usable(now_ns=NOW).epoch == 'producer-b'
    assert receiver.usable(now_ns=1_000_000_000) is None
    assert not _send(receiver, now=1_000_000_000, epoch='producer-b',
                     anchor_stamp_ns=1_000_000_000, state_stamp_ns=1_000_000_000)
    assert _send(receiver, now=1_000_000_000, epoch='producer-c',
                 anchor_stamp_ns=1_000_000_000, state_stamp_ns=1_000_000_000)
    assert receiver.usable(now_ns=1_000_000_000).epoch == 'producer-c'
