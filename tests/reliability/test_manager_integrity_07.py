"""Manager identity, motion support and publication invariants (audit 07 repair)."""
from dataclasses import replace
from itertools import permutations
import math
import json
import hashlib
import threading
from collections import deque
from types import SimpleNamespace as NS

import pytest

from reliability.contracts import CameraObservation, ContractValidationError
from reliability.common_time import MotionPose, MotionPoseSnapshot
from reliability.manager_state import AdmissionBeliefHistory
from reliability.fusion import MapObservation
from reliability.fusion_event import FusedCorrectionEvent, canonical_json, publish_fused_event
from reliability.nodes.camera_manager_node import _bootstrap_agreeing_group


def identified_observation(**overrides):
    fields = dict(camera_id="camera_A", source_batch_id="epoch-1/cycle/2",
                  producer_epoch="epoch-1", source_frame_id="frame:epoch-1:camera_A:10000000000:abc",
                  capture_stamp_ns=10_000_000_000,
                  detector_invocation_id="epoch-1/cycle/2/chunk/0", timestamp_s=10.)
    fields.update(overrides)
    return CameraObservation(**fields)


def test_physical_identity_round_trips_without_float_timestamp_reconstruction():
    stamp_ns = 1_700_000_000_123_456_789
    observation = identified_observation(capture_stamp_ns=stamp_ns,
                                         timestamp_s=stamp_ns / 1e9)
    loaded = CameraObservation.from_json(observation.to_json())
    assert loaded.capture_stamp_ns == stamp_ns
    assert loaded.source_frame_id == observation.source_frame_id
    assert loaded.detector_invocation_id == observation.detector_invocation_id


def test_frozen_legacy_observations_remain_readable_without_invented_identity():
    legacy = CameraObservation.from_dict(dict(camera_id="camera_A", timestamp_s=10.))
    assert legacy.capture_stamp_ns is None
    assert legacy.producer_epoch == legacy.detector_invocation_id == legacy.source_frame_id == ""


@pytest.mark.parametrize("field,value", [
    ("capture_stamp_ns", True), ("capture_stamp_ns", 10e9),
    ("capture_stamp_ns", -1), ("capture_stamp_ns", None),
    ("capture_stamp_ns", 9_000_000_000), ("producer_epoch", ""),
    ("source_frame_id", ""), ("detector_invocation_id", ""),
    ("producer_epoch", 123), ("source_batch_id", ""),
])
def test_partial_or_malformed_physical_identity_is_rejected(field, value):
    with pytest.raises(ContractValidationError):
        identified_observation(**{field: value})


def test_metadata_cannot_change_capture_time_through_dataclass_replacement():
    with pytest.raises(ContractValidationError, match="capture"):
        replace(identified_observation(), timestamp_s=10.04)


def motion(stamp, x, y=0., frame="odom", epoch="manager-1"):
    return MotionPose(round(stamp*1e9), (x, y), 0., frame, epoch)


def history(entries, yaw=0.):
    return MotionPoseSnapshot.capture(entries, target_frame="map_bev", source_frame="odom",
                                      epoch="manager-1", source_to_target_yaw=yaw)


def test_bootstrap_prior_can_support_one_agreeing_camera_but_never_prior_alone():
    prior = (-7.9, -8.7)
    quality_c = identified_observation(camera_id="camera_C").quality()
    quality_b = identified_observation(camera_id="camera_B").quality()
    close = MapObservation(
        "camera_C", 1.0, (-7.854, -8.685), ((0.01, 0.0), (0.0, 0.01)), quality_c
    )
    far = MapObservation(
        "camera_B", 1.0, (-7.2, -8.7), ((0.01, 0.0), (0.0, 0.01)), quality_b
    )
    assert _bootstrap_agreeing_group([close], 0.30, prior_xy=prior) == [close]
    assert _bootstrap_agreeing_group([far], 0.30, prior_xy=prior) == []
    assert _bootstrap_agreeing_group([], 0.30, prior_xy=prior) == []


def test_prior_free_bootstrap_still_rejects_a_single_camera():
    single = MapObservation(
        "camera_C", 1.0, (0.0, 0.0), ((0.01, 0.0), (0.0, 0.01)),
        identified_observation(camera_id="camera_C").quality(),
    )
    assert _bootstrap_agreeing_group([single], 0.30) == []


def test_supported_interpolation_is_order_invariant_and_preserves_frame_rotation():
    samples = [motion(10., 0.), motion(10.02, .0044), motion(10.05, .011)]
    for order in permutations(samples):
        result = history(order, yaw=math.pi/2).displacement(10_005_000_000, 10_040_000_000,
                                                         max_gap_s=.05)
        assert result.supported
        assert result.delta_xy_m == pytest.approx((0., .0077))
        assert result.sample_stamps_ns == (10_000_000_000, 10_020_000_000, 10_050_000_000)


@pytest.mark.parametrize("samples,reason", [
    ([], "empty_motion_history"), ([motion(9.9, 0.)], "missing_tail_support"),
    ([motion(10.3, 0.)], "missing_prefix_support"),
    ([motion(10., 0.), motion(10.04, .0088)], "motion_gap"),
])
def test_nearby_stale_future_and_gapped_motion_cannot_relabel_a_camera(samples, reason):
    result = history(samples).displacement(10_000_000_000, 10_040_000_000, max_gap_s=.02)
    assert not result.supported
    assert result.reason == reason
    assert result.delta_xy_m is None


def test_motion_snapshot_is_immutable_when_live_history_changes():
    live = [motion(10., 0.), motion(10.04, .0088)]
    snapshot = history(live)
    live.clear()
    assert snapshot.displacement(10_000_000_000, 10_040_000_000,
                                 max_gap_s=.05).delta_xy_m == pytest.approx((.0088, 0.))


def test_equal_capture_times_require_no_motion_evidence():
    result = history([]).displacement(10_000_000_000, 10_000_000_000, max_gap_s=.05)
    assert result.supported and result.delta_xy_m == (0., 0.)


@pytest.mark.parametrize("samples", [
    [motion(10., 0.), motion(10., 1.)],
    [motion(10., 0.), motion(10.04, 1., frame="camera_optical")],
    [motion(10., 0.), motion(10.04, 1., epoch="old")],
])
def test_conflicting_identity_frames_or_epochs_cannot_enter_motion_snapshot(samples):
    with pytest.raises(ContractValidationError):
        history(samples)


def belief(**overrides):
    payload = dict(schema_version=1, epoch="robot-1", revision=1, frame_id="map_bev",
                   anchor_stamp_ns=9_900_000_000, state_stamp_ns=10_000_000_000,
                   mean=[1., 2., 0.], covariance=[[.01, 0., 0.], [0., .01, 0.], [0., 0., .01]],
                   valid=True, initialized=True, invalid_reason="", motion_supported=True,
                   motion_support=dict(start_stamp_ns=9_900_000_000, end_stamp_ns=10_100_000_000,
                                       source="odom", supported=True, gaps=[]))
    payload.update(overrides)
    return payload


def test_same_time_newer_revision_replaces_prior_and_old_revision_cannot_restore_it():
    cache = AdmissionBeliefHistory("map_bev")
    assert cache.offer(belief())
    assert cache.offer(belief(revision=2, mean=[1.4, 2., 0.]))
    assert cache.poses() == ((10., (1.4, 2., 0.)),)
    assert cache.predictions() == ((
        10., (1.4, 2., 0.),
        ((.01, 0., 0.), (0., .01, 0.), (0., 0., .01)),
    ),)
    assert not cache.offer(belief(state_stamp_ns=10_100_000_000))
    assert cache.poses() == ((10., (1.4, 2., 0.)),)


def test_invalid_newer_belief_clears_admission_readiness_and_retains_watermark():
    cache = AdmissionBeliefHistory("map_bev")
    cache.offer(belief())
    cache.offer(belief(revision=2, valid=False, motion_supported=False,
                       invalid_reason="motion_gap", mean=None, covariance=None))
    assert cache.poses() == ()
    assert cache.latest.invalid_reason == "motion_gap"
    assert not cache.offer(belief())
    assert not cache.offer(belief(state_stamp_ns=10_100_000_000))


@pytest.mark.parametrize("extra", [
    {"schema_version": True}, {"frame_id": "camera_optical"}, {"revision": True},
    {"anchor_stamp_ns": 11_000_000_000}, {"motion_supported": "true"},
    {"covariance": [[.01, .1, 0.], [.1, .01, 0.], [0., 0., .01]]},
])
def test_invalid_operational_belief_envelopes_are_rejected(extra):
    with pytest.raises(ContractValidationError):
        AdmissionBeliefHistory("map_bev").offer(belief(**extra))


def test_same_revision_conflict_and_cross_epoch_require_integrity_failure():
    cache = AdmissionBeliefHistory("map_bev")
    cache.offer(belief())
    for extra in ({"mean": [3., 2., 0.]}, {"epoch": "robot-old"}):
        with pytest.raises(ContractValidationError):
            cache.offer(belief(**extra))


def test_malformed_higher_revision_clears_prior_and_retains_refusal_watermark():
    cache = AdmissionBeliefHistory("map_bev")
    cache.offer(belief())
    with pytest.raises(ContractValidationError):
        cache.offer(belief(revision=2, covariance=[[float("nan")]*3]*3))
    assert cache.poses() == ()
    assert not cache.offer(belief())
    assert cache.poses() == ()


def test_nullable_invalid_target_never_compares_none_to_integer_time():
    cache = AdmissionBeliefHistory("map_bev")
    cache.offer(belief())
    # Same-key inconsistent validity is an explicit integrity error, not TypeError.
    with pytest.raises(ContractValidationError):
        cache.offer(belief(state_stamp_ns=None, valid=False, invalid_reason="unsupported",
                           motion_supported=False, mean=None, covariance=None))
    assert cache.poses() == ()
    assert cache.offer(belief(revision=2, anchor_stamp_ns=None, state_stamp_ns=None, valid=False,
                              invalid_reason="unsupported", motion_supported=False, mean=None, covariance=None))
    assert cache.poses() == ()


def correction_event():
    a = identified_observation()
    b = identified_observation(camera_id="camera_B", source_frame_id="frame:epoch-1:camera_B:10000000000:def")
    readings = [MapObservation(c.camera_id, c.timestamp_s, (1., 2.), ((.01, 0.), (0., .02)), c.quality())
                for c in (a, b)]
    return FusedCorrectionEvent.create(source_batch_id=a.source_batch_id, epoch="manager-1", publication_seq=1,
        frame_id="map_bev", common_capture_stamp_ns=10_000_000_000, correction_stamp_ns=10_000_000_000,
        xy=(1., 2.), covariance_m2=((.01, 0.), (0., .02)), accepted_camera_ids=["camera_A", "camera_B"],
        contracts=[a, b], capture_observations=readings, aligned_observations=readings,
        motion_support_by_camera={}, model={"mean": "synthetic", "R": "fixed"})


def test_fusion_envelope_roundtrip_preserves_identity_and_immutable_values():
    event = correction_event()
    assert FusedCorrectionEvent.from_json(event.json) == event
    view = event.payload
    view["xy"][0] = 99.
    assert event.payload["xy"] == [1., 2.]
    assert event.payload["members"][0]["capture_observation"]["timestamp_s"] == 10.


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(accepted_camera_ids=["camera_A", "camera_A"]),
    lambda p: p.update(common_capture_stamp_ns=11_000_000_000),
    lambda p: p.update(member_ids=["wrong"]),
    lambda p: p.update(schema_version=True),
    lambda p: p.update(covariance_m2=[[.01, .1], [.1, .01]]),
])
def test_rehashed_malformed_envelope_cannot_pass_semantic_validation(mutation):
    payload = correction_event().payload
    mutation(payload)
    del payload["payload_sha256"]
    payload["payload_sha256"] = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    with pytest.raises(ContractValidationError):
        FusedCorrectionEvent.from_json(canonical_json(payload))


def test_post_envelope_diagnostic_failure_still_publishes_decision_and_retains_outcome():
    records, delivered = [], []
    class Journal:
        append = staticmethod(records.append)
    def failed():
        raise RuntimeError("compatibility sink failed")
    with pytest.raises(RuntimeError, match="compatibility"):
        publish_fused_event(correction_event(), journal=Journal(), publish_envelope=delivered.append,
                            publish_decision=lambda: delivered.append("decision"),
                            publish_compatibility=[failed, lambda: delivered.append("second compatibility")])
    assert delivered[1:] == ["decision", "second compatibility"]
    assert records[0]["status"] == "fusion_prepared"
    assert records[0]["envelope"]["event_id"] == "manager-1:fusion:1"
    errors = [r for r in records if r["status"] == "publication_error"]
    assert len(errors) == 1 and errors[0]["surface"] == "compatibility_0"


def test_failed_journal_prevents_any_publication():
    class FailedJournal:
        def append(self, event):
            raise OSError("disk full")
    delivered = []
    with pytest.raises(OSError):
        publish_fused_event(correction_event(), journal=FailedJournal(), publish_envelope=delivered.append,
                            publish_decision=lambda: delivered.append("decision"))
    assert not delivered


class Publisher:
    def __init__(self, fail=False):
        self.messages, self.fail = [], fail
    def publish(self, message):
        if self.fail:
            raise RuntimeError("controlled publication failure")
        self.messages.append(message)


def manager_node(tmp_path):
    from reliability.nodes.camera_manager_node import CameraManagerNode
    from reliability.camera_manager import CameraManager, CameraManagerConfig
    from reliability.source_batch_buffer import SourceBatchBuffer
    from unav_common.camera_outcomes import OutcomeJournal
    n = object.__new__(CameraManagerNode)
    n._manager_epoch = "manager-1"
    n._terminal_stopped = False
    n._terminal_stop_request_id = ""
    n._session_stopped_appended = False
    n._outcome_transport_enabled = True
    n._terminal_stop_ack_pub = Publisher()
    n._input_lock, n._decision_lock = threading.RLock(), threading.RLock()
    n._decision_snapshot = None
    n._fusion_publication_seq = 0
    n.frame_id, n.odometry_frame_id, n.odometry_to_map_yaw_rad = "map_bev", "odom", 0.
    n._outcome_journal = OutcomeJournal(tmp_path / "manager.jsonl", n._manager_epoch)
    n.get_clock = lambda: NS(now=lambda: NS(nanoseconds=10_000_000_000))
    n.get_logger = lambda: NS(
        debug=lambda _: None, info=lambda _: None, warn=lambda _: None, error=lambda _: None)
    n.batch_outcome_pub = Publisher()
    n._batch_clock_high_water_s = None
    n.camera_ids = ["camera_A", "camera_B"]
    n._source_batch_buffer = SourceBatchBuffer(n.camera_ids, on_event=n._publish_batch_outcome)
    n._completed_source_batches = deque()
    n._completed_source_batch_capacity = 64
    n.require_source_batch_id = True
    n._latest, n._ready_source_batch_id = {}, None
    n._ready_source_batch_stamp_s, n._last_decided_source_batch_id = -math.inf, None
    n._belief_query_history = deque([(9.95, (0., 0., 0.)), (9.99, (.4, 0., 0.))], maxlen=400)
    n._belief_prediction_history = deque(maxlen=400)
    n._admission_beliefs = AdmissionBeliefHistory(n.frame_id)
    n._canonical_belief_seen, n._has_operational_anchor = False, True
    n._odom_history = deque([motion(9.95, 0.), motion(9.99, 0.)], maxlen=600)
    n.manager = CameraManager(CameraManagerConfig(allowed_camera_ids=tuple(n.camera_ids)))
    n.fusion_mode, n.authority, n.fusion_rule = True, "active", "joint_network"
    n.fusion_max_timestamp_spread_s, n.reliability_query_max_time_delta_s = .05, .35
    n.propagation_drift_std, n.correction_residual_interval_s = .05, .05
    n._bootstrap_camera_ids = set()
    n.bootstrap_min_cameras, n.bootstrap_max_disagreement_m = 2, .91
    n.fusion_disagreement_gate_m = .6
    n.camera_models = {c: NS(cam_pos=(0., 0., 5.)) for c in n.camera_ids}
    n._bias_floors = lambda _: None
    n.timestamp_compensation, n.fusion_common_mode_std_m = False, 0.
    n.observation_model, n.covariance_profile = "synthetic", "fixed_R"
    n._gate_rejections, n._reliability_query_source_by_camera = {}, {}
    n._camera_mapping_reasons = {}
    n._measurement_model_status_by_camera, n._detection_extras_by_camera = {}, {}
    n.decision_pub, n.selected_pub, n.active_pub, n.fused_correction_pub = (Publisher() for _ in range(4))
    n.map_observations_pub = None
    contracts = [identified_observation(camera_id=c, capture_stamp_ns=round(t*1e9), timestamp_s=t,
                                       source_frame_id=f"frame:epoch-1:{c}:{round(t*1e9)}:image")
                 for c, t in zip(n.camera_ids, (9.95, 9.99))]
    readings = [MapObservation(c.camera_id, c.timestamp_s, (1., 2.), ((.01, 0.), (0., .02)), c.quality())
                for c in contracts]
    for c in contracts:
        n._observation_callback(c.camera_id)(NS(data=c.to_json()))
    n._map_observations = lambda _: [
        replace(reading, timestamp_s=contract.timestamp_s)
        for reading, contract in zip(readings, n._decision_snapshot.contracts)
    ]
    return n, contracts, readings


def test_terminal_request_quiesces_manager_and_binds_durable_marker(tmp_path):
    from unav_common.camera_outcomes import read_journal
    from unav_common.terminal_stop import (
        TerminalStopRequest,
        terminal_stop_ack_from_json,
        terminal_stop_request_to_json,
    )

    n, _, _ = manager_node(tmp_path)
    request = TerminalStopRequest('request-1', 'run-1', 'goal_reached', 10)
    n._terminal_stop_request_cb(
        NS(data=terminal_stop_request_to_json(request))
    )
    try:
        assert n._terminal_stopped
        assert not n._outcome_transport_enabled
        ack = terminal_stop_ack_from_json(n._terminal_stop_ack_pub.messages[-1].data)
        assert ack.component == 'camera_manager'
        assert ack.request_id == request.request_id
        stopped = list(read_journal(n._outcome_journal.path))[-1]
        assert stopped['status'] == 'session_stopped'
        assert stopped['terminal_stop_request_id'] == request.request_id
    finally:
        n._outcome_journal.close()


def test_real_manager_uses_two_stationary_odom_endpoints_never_belief_jump(tmp_path):
    n, _, _ = manager_node(tmp_path)
    try:
        n._decide()
        event = FusedCorrectionEvent.from_json(n.fused_correction_pub.messages[0].data)
        old = event.payload["members"][0]
        assert old["capture_observation"]["xy_m"] == old["common_observation"]["xy_m"] == [1., 2.]
        assert old["capture_stamp_ns"] == 9_950_000_000
        assert old["common_observation"]["timestamp_s"] == 9.99
        assert old["common_observation"]["covariance_m2"][0][0] == pytest.approx(.010004)
        n._decide()
        assert len(n.fused_correction_pub.messages) == 1
    finally:
        n._outcome_journal.close()


def test_manager_bootstraps_from_declared_start_plus_one_agreeing_camera(tmp_path):
    n, _, readings = manager_node(tmp_path)
    n._has_operational_anchor = False
    n.bootstrap_prior_pose = (1.0, 2.0, 0.0)
    n.bootstrap_prior_counts_as_support = True
    n.bootstrap_min_cameras = 2
    n.bootstrap_max_disagreement_m = 0.30
    n._map_observations = lambda _: readings[:1]
    try:
        n._decide()
        assert len(n.fused_correction_pub.messages) == 1
        decision = json.loads(n.decision_pub.messages[0].data)
        assert decision["accepted_camera_ids"] == ["camera_A"]
        assert decision["bootstrap_evidence"] == {
            "camera_count": 1,
            "camera_ids": ["camera_A"],
            "max_disagreement_m": 0.30,
            "prior_used": True,
            "prior_xy": [1.0, 2.0],
            "required_support_count": 2,
            "support_count": 2,
        }
    finally:
        n._outcome_journal.close()


def test_timestamp_compensation_stops_at_latest_supported_odometry(tmp_path):
    """Image callbacks may precede odometry at the manager's current ROS time."""
    n, _, _ = manager_node(tmp_path)
    n.timestamp_compensation = True
    try:
        n._decide()
        assert len(n.fused_correction_pub.messages) == 1
        event = FusedCorrectionEvent.from_json(n.fused_correction_pub.messages[0].data).payload
        assert event["common_capture_stamp_ns"] == 9_990_000_000
        assert event["correction_stamp_ns"] == 9_990_000_000
        assert event["correction_motion_support"]["supported"] is True
        assert event["correction_motion_support"]["reason"] == "same_instant"
        decision = json.loads(n.decision_pub.messages[0].data)
        assert decision["propagation"].startswith(
            "applied_to_latest_supported_odometry")
    finally:
        n._outcome_journal.close()


def test_manager_freezes_members_before_new_batch_callback(tmp_path):
    n, contracts, readings = manager_node(tmp_path)
    first_id = contracts[0].source_batch_id
    def mapping(_):
        for c in contracts:
            newer = replace(c, source_batch_id="epoch-1/cycle/3", timestamp_s=10., capture_stamp_ns=10_000_000_000,
                            source_frame_id=c.source_frame_id+":new", detector_invocation_id="epoch-1/cycle/3/chunk/0")
            n._observation_callback(c.camera_id)(NS(data=newer.to_json()))
        return readings
    n._map_observations = mapping
    try:
        n._decide()
        event = FusedCorrectionEvent.from_json(n.fused_correction_pub.messages[0].data).payload
        assert event["source_batch_id"] == first_id
        assert {m["source_observation"]["source_batch_id"] for m in event["members"]} == {first_id}
        assert n._ready_source_batch_id == "epoch-1/cycle/3"
    finally:
        n._outcome_journal.close()


def test_manager_consumes_every_completed_batch_in_fifo_order(tmp_path):
    """Two detector rounds between timer ticks must not overwrite the first."""
    n, contracts, readings = manager_node(tmp_path)
    first_id = contracts[0].source_batch_id
    second_id = "epoch-1/cycle/3"
    for c in contracts:
        newer = replace(
            c,
            source_batch_id=second_id,
            timestamp_s=10.0,
            capture_stamp_ns=10_000_000_000,
            source_frame_id=c.source_frame_id + ":new",
            detector_invocation_id=second_id + "/chunk/0",
        )
        n._observation_callback(c.camera_id)(NS(data=newer.to_json()))
    n._map_observations = lambda _: [
        replace(reading, timestamp_s=contract.timestamp_s)
        for reading, contract in zip(readings, n._decision_snapshot.contracts)
    ]
    try:
        n._decide()
        events = [FusedCorrectionEvent.from_json(m.data).payload
                  for m in n.fused_correction_pub.messages]
        assert [event["source_batch_id"] for event in events] == [first_id, second_id]
        assert n._ready_source_batch_id is None
        assert not any(
            json.loads(m.data).get("status") == "superseded_before_decision"
            for m in n.batch_outcome_pub.messages
        )
    finally:
        n._outcome_journal.close()


def test_real_manager_retains_decision_after_compatibility_failure_and_never_retries(tmp_path):
    from unav_common.camera_outcomes import read_journal
    n, contracts, _ = manager_node(tmp_path)
    n.selected_pub.fail = True
    try:
        with pytest.raises(RuntimeError):
            n._decide()
        n._decide()
        assert len(n.fused_correction_pub.messages) == len(n.decision_pub.messages) == 1
        assert n._last_decided_source_batch_id == contracts[0].source_batch_id
        rows = list(read_journal(n._outcome_journal.path))
        assert any(r["status"] == "fusion_prepared" for r in rows)
        assert any(r["status"] == "publication_error" for r in rows)
        assert rows[-1]["status"] == "decision_error"
    finally:
        n._outcome_journal.close()
