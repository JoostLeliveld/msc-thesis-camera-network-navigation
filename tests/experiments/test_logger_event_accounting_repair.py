"""Desired invariants for schema-8 logger delivery and closure accounting."""
from __future__ import annotations

import csv
import io
import json
import math
import threading
from collections import deque
from types import SimpleNamespace as NS

import pytest
from std_msgs.msg import String

from experiments.core.camera_opportunity_log import JsonlDeliveryLog
from experiments.nodes.experiment_logger import (
    ExperimentLogger, _write_json_atomic,
)
from unav_common.correction_ledger import validate_correction_ledger


class Clock:
    def __init__(self, seconds=10.0):
        self.seconds = seconds

    def now(self):
        return NS(nanoseconds=int(self.seconds * 1e9))


def node(tmp_path):
    value = object.__new__(ExperimentLogger)
    value.run_id = 'test-run'
    value._clock = Clock()
    value.get_clock = lambda: value._clock
    value._event_lock = threading.RLock()
    value._accepting_events = True
    value._event_streams_closed = False
    value._last_event_wall_s = 0.0
    value._late_event_count = 0
    value._runtime_event_counts = {}
    value._runtime_event_invalid_count = 0
    value._valid_run = True
    value._invalid_reason = ''
    value.runtime_event_file = io.StringIO()
    value.runtime_event_log = JsonlDeliveryLog(value.runtime_event_file)
    value._fusion_decision_payloads = {}
    value._correction_publications = {}
    value._assimilation_payloads = {}
    value._assimilation_source_batches = set()
    value._assimilation_count = 0
    value._assimilation_dropped_count = 0
    value._assimilation_dropped_reasons = {}
    value._last_correction_stamp_s = None
    value._longest_correction_gap_s = 0.0
    value._terminal_stop_verified = False
    value._terminal_stop_event_id = ''
    value._terminal_zero_forwarded = False
    value._terminal_zero_stamp_s = float('nan')
    value._terminal_stop_request_id = ''
    value._terminal_stop_request_payload = None
    value._terminal_stop_acks = {}
    value._motion_history = deque()
    value._published_terminal_stop_requests = []
    value._terminal_stop_request_pub = NS(
        publish=value._published_terminal_stop_requests.append
    )
    value._stop_requested = False
    value._mission_goal_state = None
    value._contact_channel_status = None
    value._active_mission_goal_id = ''
    value._goal_in_radius_since = None
    value._goal_stable_since = None
    value._goal_region_entered = False
    value._goal_region_first_stamp = float('nan')
    value._fusion_obs_last_stamp = None
    value._fusion_decision_seq = 0
    value._obs_repeat_count = {}
    value._obs_seq_by_camera = {}
    value._gt_xy = None
    value._gt_stamp = float('nan')
    value._gt_at = lambda _stamp: (False, float('nan'), float('nan'), float('nan'))
    value.fusion_obs_file = io.StringIO()
    value.fusion_obs_writer = csv.writer(value.fusion_obs_file)
    value.assimilation_file = io.StringIO()
    value.assimilation_writer = csv.writer(value.assimilation_file)
    value.correction_publication_file = io.StringIO()
    value.correction_publication_writer = csv.writer(value.correction_publication_file)
    value.run_dir = str(tmp_path)
    return value


def decision(batch, x):
    observation = dict(camera='camera_A', used=True, xy=[x, 0.],
                       cov=[[1., 0.], [0., 1.]], aligned_xy=[x, 0.],
                       aligned_cov=[[1., 0.], [0., 1.]], obs_stamp=x)
    return String(data=json.dumps(dict(
        source_batch_id=batch, common_capture_stamp=x, fused_stamp=x,
        fused_xy=[x, 0.], fused_cov=[[1., 0.], [0., 1.]],
        accepted_camera_ids=['camera_A'], observations=[observation])))


def terminal(batch, *, status='accepted', reason='accepted'):
    return String(data=json.dumps(dict(
        schema_version=1, source_batch_id=batch, correction_stamp=1.,
        apply_stamp=1.1, status=status, reason=reason,
        accepted=status in ('accepted', 'accepted_bootstrap', 'reanchored'),
        nis=0., belief_stamp_after=1.)))


def publication(batch):
    return String(data=json.dumps(dict(
        schema_version=1, source_batch_id=batch, correction_stamp=1.,
        frame_id='map_bev', xy=[1., 2.],
        covariance_m2=[[1., 0.], [0., 1.]],
        accepted_camera_ids=['camera_A'])))


def terminal_v2(batch):
    return String(data=json.dumps(dict(
        schema_version=2, source_batch_id=batch, correction_stamp=1.,
        apply_stamp=1.1, status='accepted', reason='accepted', accepted=True,
        nis=0., belief_stamp_after=1., epoch='planner:1', revision_before=4,
        revision_after=5, frame_id='map_bev', state_stamp_ns=1_100_000_000,
        correction_stamp_ns=1_000_000_000, apply_stamp_ns=1_100_000_000,
        initialized=True,
        posterior_mean=[1., 2., 0.],
        posterior_covariance=[[1., 0., 0.], [0., 1., 0.], [0., 0., .1]],
        valid=True, motion_supported=True)))


def test_distinct_decisions_survive_same_logger_clock(tmp_path):
    value = node(tmp_path)
    value._fusion_decision_cb(decision('a', 1.))
    value._fusion_decision_cb(decision('b', 2.))
    assert value._fusion_decision_seq == 2
    assert set(value._fusion_decision_payloads) == {'a', 'b'}
    assert len(value.fusion_obs_file.getvalue().splitlines()) == 2


def test_raw_delivery_retains_duplicate_while_canonical_terminal_is_unique(tmp_path):
    value = node(tmp_path)
    message = terminal('a')
    value._correction_assimilation_cb(message)
    value._correction_assimilation_cb(message)
    deliveries = [json.loads(line) for line in value.runtime_event_file.getvalue().splitlines()]
    assert len(deliveries) == 2
    assert value._assimilation_count == 1
    assert set(value._assimilation_payloads) == {'a'}
    assert not value._valid_run
    assert 'duplicate_source_batch_assimilation' in value._invalid_reason


def test_publication_and_terminal_reconcile_by_identity(tmp_path):
    value = node(tmp_path)
    value._fused_correction_cb(publication('a'))
    value._correction_assimilation_cb(terminal('a'))
    result = value._correction_ledger_result()
    assert result.valid and result.accepted_update_ids == ('a',)


def test_written_files_support_independent_fault_accounting(tmp_path):
    value = node(tmp_path)
    for batch in ('accepted', 'rejected', 'dropped', 'missing'):
        value._fused_correction_cb(publication(batch))
    value._correction_assimilation_cb(terminal('accepted'))
    value._correction_assimilation_cb(
        terminal('rejected', status='rejected', reason='nis_gate'))
    value._correction_assimilation_cb(
        terminal('dropped', status='dropped', reason='replay_gap'))
    value._correction_assimilation_cb(terminal('accepted'))  # raw retransmission
    value._correction_assimilation_cb(String(data='{malformed'))

    publications = []
    for row in csv.reader(io.StringIO(value.correction_publication_file.getvalue())):
        publications.append(dict(source_batch_id=row[0], correction_stamp=float(row[4]),
                                 frame_id=row[10], member_ids=json.loads(row[13]),
                                 payload={'present': 1.0}))
    outcomes = []
    for row in csv.reader(io.StringIO(value.assimilation_file.getvalue())):
        outcomes.append(dict(source_batch_id=row[0], correction_stamp=float(row[1]),
                             apply_stamp=float(row[2]), status=row[3], reason=row[4],
                             accepted=bool(int(row[5]))))
    reconstructed = validate_correction_ledger(publications, outcomes)
    assert reconstructed.missing_ids == ('missing',)
    assert reconstructed.accepted_update_ids == ('accepted',)
    assert reconstructed.refusal_counts == {'dropped': 1, 'rejected': 1}
    raw = [json.loads(line) for line in value.runtime_event_file.getvalue().splitlines()]
    assert len(raw) == 9
    assert sum(r.get('valid_json_object') is False for r in raw) == 1
    assert len(outcomes) == 3  # duplicate/malformed deliveries are never second outcomes


def test_reason_is_required_for_both_refusal_statuses(tmp_path):
    for status in ('rejected', 'dropped'):
        value = node(tmp_path/status)
        value._correction_assimilation_cb(terminal('a', status=status, reason=''))
        assert not value._valid_run
        assert 'correction_refusal_without_reason' in value._invalid_reason


def test_schema2_terminal_preserves_reconstructable_posterior(tmp_path):
    value = node(tmp_path)
    value._correction_assimilation_cb(terminal_v2('a'))
    assert value._valid_run
    assert value._assimilation_payloads['a']['revision_after'] == 5
    assert '[[1.0,0.0,0.0]' in value.assimilation_file.getvalue()


def test_schema2_exact_time_accepts_large_epoch_float_projection(tmp_path):
    value = node(tmp_path)
    payload = json.loads(terminal_v2('large').data)
    correction_ns = 1_787_000_000_123_456_789
    apply_ns = correction_ns + 1_000_000
    payload.update(correction_stamp_ns=correction_ns, apply_stamp_ns=apply_ns,
                   correction_stamp=correction_ns * 1e-9,
                   apply_stamp=apply_ns * 1e-9,
                   state_stamp_ns=apply_ns, belief_stamp_after=apply_ns * 1e-9)
    value._correction_assimilation_cb(String(data=json.dumps(payload)))
    assert value._valid_run and 'large' in value._assimilation_payloads


def test_malformed_schema2_terminal_is_raw_only(tmp_path):
    value = node(tmp_path)
    payload = json.loads(terminal_v2('a').data)
    payload['posterior_covariance'] = [[1., 2.], [2., 1.]]
    value._correction_assimilation_cb(String(data=json.dumps(payload)))
    assert not value._valid_run
    assert not value._assimilation_payloads
    assert len(value.runtime_event_file.getvalue().splitlines()) == 1


def test_uninitialized_reasoned_drop_is_a_canonical_terminal_without_posterior(tmp_path):
    value = node(tmp_path)
    payload = json.loads(terminal_v2('uninitialized').data)
    payload.update(status='dropped', reason='uninitialized', accepted=False,
                   initialized=False, state_stamp_ns=None, posterior_mean=None,
                   posterior_covariance=None, valid=False, motion_supported=False)
    value._correction_assimilation_cb(String(data=json.dumps(payload)))
    assert value._valid_run
    assert value._assimilation_payloads['uninitialized']['status'] == 'dropped'


def test_actuation_outcome_records_forwarded_zero_without_claiming_application(tmp_path):
    value = node(tmp_path)
    value._stop_requested = True
    payload = dict(schema_version=1, event_id='guard:8', status='forwarded_zero',
                   forwarded_linear=0.0, forwarded_angular=0.0,
                   physical_application_verified=False)
    value._runtime_outcome_cb('/sim/actuation_outcome',
                              String(data=json.dumps(payload)))
    assert value._terminal_zero_forwarded
    assert not value._terminal_stop_verified
    assert value._terminal_stop_event_id == 'guard:8'


def test_republished_terminal_zero_preserves_first_rest_window_anchor(tmp_path):
    value = node(tmp_path)
    value._stop_requested = True
    for event_id, stamp_ns in (('guard:8', 10_000_000_000),
                               ('guard:9', 10_200_000_000)):
        payload = dict(
            schema_version=1,
            event_id=event_id,
            status='forwarded_zero',
            forwarded_linear=0.0,
            forwarded_angular=0.0,
            forwarded_sim_stamp_ns=stamp_ns,
            physical_application_verified=False,
        )
        value._runtime_outcome_cb(
            '/sim/actuation_outcome', String(data=json.dumps(payload)))

    assert value._terminal_zero_stamp_s == 10.0
    assert value._terminal_stop_event_id == 'guard:8'


def test_terminal_motion_history_does_not_require_a_mission_goal(tmp_path):
    value = node(tmp_path)
    value.stuck_window_s = 8.0
    value.goal_success_hold_s = 0.5
    value.goal_stable_hold_s = 1.0
    value.first_cmd_linear_eps = 0.01
    value.first_cmd_angular_eps = 0.01

    value._remember_motion_sample(10.0, 1.0, 2.0, float('nan'), 0.0, 0.0)

    assert len(value._motion_history) == 1
    stamp, x, y, goal_dist, command_active, yaw, angular_active = value._motion_history[0]
    assert (stamp, x, y, command_active) == (10.0, 1.0, 2.0, 0.0)
    assert math.isnan(goal_dist)
    assert math.isnan(yaw)
    assert angular_active == 0.0


def test_stuck_detector_does_not_stop_an_intentional_pivot(tmp_path):
    value = node(tmp_path)
    value.first_cmd_linear_eps = 0.01
    value.first_cmd_angular_eps = 0.01
    value.stuck_window_s = 1.0
    value.goal_success_hold_s = 0.5
    value.goal_stable_hold_s = 0.5
    value.stuck_max_displacement_m = 0.1
    value.stuck_max_goal_improvement_m = 0.1
    value.stuck_cmd_fraction_min = 0.8
    value.stuck_idle_cmd_fraction_max = 0.1
    value.goal_stable_radius = 0.2
    value._first_cmd_stamp = 10.0
    value._finish_run = lambda *_args: pytest.fail('pivot was classified as stuck')
    for index in range(11):
        value._remember_motion_sample(
            10.0 + 0.1 * index, 1.0, 2.0, 4.0, 0.0, 0.5,
            operational_yaw=0.05 * index,
        )
    assert not value._maybe_finish_for_stuck(11.0, 4.0)


def test_stuck_detector_still_stops_commanded_turn_without_yaw_progress(tmp_path):
    value = node(tmp_path)
    value.first_cmd_linear_eps = 0.01
    value.first_cmd_angular_eps = 0.01
    value.stuck_window_s = 1.0
    value.goal_success_hold_s = 0.5
    value.goal_stable_hold_s = 0.5
    value.stuck_max_displacement_m = 0.1
    value.stuck_max_goal_improvement_m = 0.1
    value.stuck_cmd_fraction_min = 0.8
    value.stuck_idle_cmd_fraction_max = 0.1
    value.goal_stable_radius = 0.2
    value._first_cmd_stamp = 10.0
    finished = []
    value._finish_run = lambda reason, stamp: finished.append((reason, stamp))
    value.get_logger = lambda: NS(info=lambda *_args: None)
    for index in range(11):
        value._remember_motion_sample(
            10.0 + 0.1 * index, 1.0, 2.0, 4.0, 0.0, 0.5,
            operational_yaw=0.0,
        )
    assert value._maybe_finish_for_stuck(11.0, 4.0)
    assert finished == [('stuck', 11.0)]


def test_terminal_stop_verification_uses_operational_rest_not_ground_truth(tmp_path):
    value = node(tmp_path)
    value._stop_requested = True
    value._terminal_zero_forwarded = True
    value._terminal_zero_stamp_s = 10.0
    value._motion_history.extend([
        (10.00, 1.000, 2.000, 0.2, 0.0),
        (10.10, 1.002, 2.001, 0.2, 0.0),
        (10.26, 1.003, 2.002, 0.2, 0.0),
    ])
    value._update_terminal_stop_verification(10.26)
    assert value._terminal_stop_verified


def test_terminal_stop_verification_refuses_motion_or_nonzero_command(tmp_path):
    for samples in (
        [(10.0, 1.0, 2.0, 0.2, 0.0), (10.3, 1.02, 2.0, 0.2, 0.0)],
        [(10.0, 1.0, 2.0, 0.2, 0.0), (10.3, 1.0, 2.0, 0.2, 1.0)],
    ):
        value = node(tmp_path)
        value._stop_requested = True
        value._terminal_zero_forwarded = True
        value._terminal_zero_stamp_s = 10.0
        value._motion_history.extend(samples)
        value._update_terminal_stop_verification(10.3)
        assert not value._terminal_stop_verified


def test_terminal_stop_verification_starts_rest_after_last_nonzero_sample(tmp_path):
    value = node(tmp_path)
    value._stop_requested = True
    value._terminal_zero_forwarded = True
    value._terminal_zero_stamp_s = 10.0
    value._motion_history.extend([
        # A cached pre-stop command can be observed after the authoritative
        # forwarded-zero outcome because the subscriptions are asynchronous.
        (10.05, 1.0000, 2.0000, 0.2, 1.0),
        (10.10, 1.0002, 2.0001, 0.2, 0.0),
        (10.22, 1.0003, 2.0001, 0.2, 0.0),
        (10.37, 1.0004, 2.0002, 0.2, 0.0),
    ])
    value._update_terminal_stop_verification(10.37)
    assert value._terminal_stop_verified


def test_stop_request_defers_completion_and_finalization(monkeypatch, tmp_path):
    value = node(tmp_path)
    value._completed = False
    value._finalizing = False
    timers = []
    monkeypatch.setattr('experiments.nodes.experiment_logger.threading.Timer',
                        lambda delay, call: timers.append((delay, call)) or NS(start=lambda: None))
    value._finish_run('collision', 12.)
    assert value._stop_requested and not value._completed
    assert value._accepting_events
    assert value._published_terminal_stop_requests
    assert value._terminal_stop_request_id.startswith('logger:test-run:terminal:')
    assert timers and timers[0][1] == value._finalize_run


def test_atomic_summary_failure_preserves_previous_file(monkeypatch, tmp_path):
    path = tmp_path/'run_summary.json'
    path.write_text('{"previous": true}\n')
    monkeypatch.setattr('experiments.nodes.experiment_logger.json.dump',
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(OSError):
        _write_json_atomic(str(path), {'completed': True})
    assert json.loads(path.read_text()) == {'previous': True}
    assert not list(tmp_path.glob('.summary-*.tmp'))


def test_close_attempts_every_file_after_one_failure(tmp_path):
    class Broken(io.StringIO):
        def flush(self):
            raise OSError('flush failed')

        def close(self):
            self.was_closed = True
            raise OSError('close failed')

    value = NS(file=Broken(), plan_file=io.StringIO(), perception_file=None,
               fusion_obs_file=io.StringIO(), assimilation_file=io.StringIO(),
               correction_publication_file=io.StringIO(), camera_opportunity_file=io.StringIO(),
               runtime_event_file=io.StringIO(), _data_file_close_errors=[],
               _event_streams_closed=False)
    errors = ExperimentLogger._flush_close_data_files(value)
    assert len(errors) == 2
    assert value.plan_file.closed and value.fusion_obs_file.closed
    assert value.assimilation_file.closed and value.runtime_event_file.closed
    assert value._event_streams_closed
