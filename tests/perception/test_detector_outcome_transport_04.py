"""Outcome durability/identity regressions using extracted production methods."""
import ast
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/perception'))
NODE = ROOT / 'src/perception/perception/nodes/batched_four_camera_yolo_node.py'


def method(name, **env):
    tree = ast.parse(NODE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BatchedFourCameraYoloNode')
    f = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), f], type_ignores=[])
    scope = dict(json=json, String=lambda: NS(), replace=replace, **env)
    exec(compile(ast.fix_missing_locations(module), str(NODE), 'exec'), scope)
    return scope[name]


def test_outcome_is_journaled_before_failing_transport(tmp_path):
    recorded = []
    class Journal:
        def append(self, event):
            recorded.append(dict(event))
            return dict(event, event_id='producer:1')
    def fail(message):
        assert recorded, 'transport attempted before durable journal append'
        raise RuntimeError('publisher unavailable')
    node = NS(_producer_epoch='producer', _outcome_journal=Journal(),
              get_clock=lambda: NS(now=lambda: NS(nanoseconds=1)),
              get_logger=lambda: NS(info=lambda _: None, debug=lambda _: None),
              batch_outcome_publisher=NS(publish=fail))
    with pytest.raises(RuntimeError, match='publisher unavailable'):
        method('_publish_batch_outcome')(node, dict(status='aborted', source_batch_id='cycle'))
    assert recorded[0]['source_batch_id'] == 'cycle'

from perception.core.detector_outcomes import (
    OutcomeJournal, read_journal, journal_path, image_content_sha256, source_frame_id,
    frame_member, OUTCOME_HISTORY_DEPTH,
)
from perception.core.four_camera_batch import FourCameraBatcher, PendingFrame, BatchContractError


def test_durable_journal_replays_after_publisher_failure(tmp_path):
    path = tmp_path / 'events.jsonl'
    writer = OutcomeJournal(path, 'producer')
    node = NS(_outcome_journal=writer, get_clock=lambda: NS(now=lambda: NS(nanoseconds=100)),
              get_logger=lambda: NS(info=lambda _: None, debug=lambda _: None),
              batch_outcome_publisher=NS(publish=lambda _: (_ for _ in ()).throw(RuntimeError('no publisher'))))
    with pytest.raises(RuntimeError):
        method('_publish_batch_outcome')(node, dict(status='inference_error', source_batch_id='cycle'))
    writer.close()
    rows = list(read_journal(path))
    assert len(rows) == 1 and rows[0]['event_id'] == 'producer:1'
    assert rows[0]['schema_version'] == 'camera_batch_outcome.v2'
    assert rows[0]['source_batch_id'] == 'cycle'
    assert rows[0]['journal_path'] == str(path)


def test_fsync_precedes_return_and_hash_chain_retains_distinct_equal_time_events(tmp_path, monkeypatch):
    import unav_common.camera_outcomes as outcomes
    writer = OutcomeJournal(tmp_path / 'events.jsonl', 'producer')
    calls = []
    sync = outcomes.os.fsync
    def synced(fd):
        calls.append(fd)
        sync(fd)
    monkeypatch.setattr(outcomes.os, 'fsync', synced)
    first = writer.append(dict(status='selected', publish_stamp_s=1))
    second = writer.append(dict(status='aborted', publish_stamp_s=1))
    assert len(calls) == 2
    assert first['event_id'] != second['event_id']
    assert second['previous_event_sha256'] == first['event_sha256']
    assert list(read_journal(writer.path)) == [first, second]
    writer.close()
    with pytest.raises(FileExistsError): OutcomeJournal(writer.path, 'new-process')


def test_partial_write_poison_and_torn_tail_are_explicit(tmp_path, monkeypatch):
    import unav_common.camera_outcomes as outcomes
    writer = OutcomeJournal(tmp_path / 'events.jsonl', 'producer')
    write = outcomes.os.write
    calls = []
    def fail_after_prefix(fd, data):
        calls.append(1)
        if len(calls) == 1: return write(fd, data[:10])
        raise OSError('injected disk failure')
    monkeypatch.setattr(outcomes.os, 'write', fail_after_prefix)
    with pytest.raises(OSError): writer.append(dict(status='selected'))
    with pytest.raises(RuntimeError, match='unavailable'): writer.append(dict(status='aborted'))
    writer.close()
    with pytest.raises(ValueError, match='incomplete'): list(read_journal(writer.path))


def test_interrupt_delivered_after_fsync_does_not_poison_journal(tmp_path, monkeypatch):
    """A shutdown signal may interrupt return, never the durable transaction."""
    import unav_common.camera_outcomes as outcomes

    writer = OutcomeJournal(tmp_path / 'events.jsonl', 'producer')
    restore_calls = []

    monkeypatch.setattr(outcomes, '_block_termination_signals', lambda: {'old-mask'})
    def interrupt_once(previous_mask):
        restore_calls.append(previous_mask)
        if len(restore_calls) == 1:
            raise KeyboardInterrupt('pending SIGINT delivered on unmask')
    monkeypatch.setattr(outcomes, '_restore_signal_mask', interrupt_once)

    with pytest.raises(KeyboardInterrupt, match='pending SIGINT'):
        writer.append(dict(status='selected'))
    # The first record was already fsynced and the writer's hash-chain state was
    # committed before SIGINT became deliverable. Shutdown can therefore record
    # its own terminal event instead of inheriting a falsely poisoned journal.
    stopped = writer.append(dict(status='session_stopped'))
    assert stopped['event_id'] == 'producer:2'
    assert [row['status'] for row in read_journal(writer.path)] == [
        'selected', 'session_stopped'
    ]
    writer.close()


def test_journal_capacity_stops_without_silent_rotation(tmp_path):
    writer = OutcomeJournal(tmp_path / 'events.jsonl', 'producer', max_bytes=1)
    with pytest.raises(RuntimeError, match='byte limit'): writer.append(dict(status='selected'))
    assert writer.path.read_bytes() == b''
    with pytest.raises(RuntimeError, match='unavailable'): writer.append(dict(status='aborted'))
    writer.close()


def test_paths_are_explicit_or_in_tracked_ros_log_dir(tmp_path):
    with pytest.raises(ValueError, match='required'): journal_path('', 'epoch', {})
    assert journal_path('', 'epoch', {'ROS_LOG_DIR': str(tmp_path)}) == tmp_path / 'camera_outcomes/epoch.jsonl'
    assert journal_path(str(tmp_path / '{producer_epoch}.jsonl'), 'epoch', {}) == tmp_path / 'epoch.jsonl'


def test_frame_content_identity_and_conflicting_same_stamp_fail_closed():
    values = dict(encoding='rgb8', width=1, height=1, step=3)
    digest = image_content_sha256(**values, data=b'123')
    different = image_content_sha256(**values, data=b'456')
    assert digest != different
    fid = source_frame_id('process', 'camera_A', 1, digest)
    assert fid != source_frame_id('process', 'camera_A', 1, different)
    events = []
    b = FourCameraBatcher(on_event=events.append)
    f = PendingFrame('camera_A', 1, 0, 0, object(), fid, digest)
    b.offer(f)
    assert b.offer(replace(f)).status == 'duplicate'
    with pytest.raises(BatchContractError, match='conflicting image bytes'):
        b.offer(replace(f, content_sha256=different,
                        source_frame_id=source_frame_id('process', 'camera_A', 1, different)))
    assert events[-1]['status'] == 'conflicting_duplicate_image'


def test_real_callback_batch_inference_and_observation_identity(tmp_path):
    import math
    import time
    import numpy as np
    sys.path.insert(0, str(ROOT / 'src/reliability'))
    from reliability.contracts import CameraObservation
    from perception.core.four_camera_batch import (
        CAMERA_ORDER, stamp_parts_to_ns, MAX_FUTURE_IMAGE_STAMP_S, validate_batch_results,
    )
    from perception.core.yolo_selection import select_best_detection

    # Execute the actual decoder function without importing ROS's Image type.
    tree = ast.parse((ROOT / 'src/perception/perception/core/ros_image.py').read_text())
    decoder = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    decoder.args.args[0].annotation = None
    decoder.returns = None
    decoder_env = {'np': np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[decoder], type_ignores=[])), '<decoder>', 'exec'), decoder_env)

    def array(values):
        value = np.asarray(values)
        return NS(detach=lambda: NS(cpu=lambda: NS(numpy=lambda: value)))
    class Boxes:
        def __init__(self, tag):
            self.xyxy = array([[tag, 0, tag + 1, 2]])
            self.conf = array([.9])
            self.cls = array([0])
        def __len__(self): return 1

    journal = OutcomeJournal(tmp_path / 'events.jsonl', 'epoch')
    node = NS(_producer_epoch='epoch', _cycle_sequence=0, _outcome_journal=journal,
              inference_chunk=2, torchscript_detection_only=False,
              synchronization_mode='strict', _clock_s=lambda: 2,
              _strict_frames={}, _strict_last_receive_wall={}, _strict_decisions={}, _strict_batches=0,
              image_size=960, predict_conf_floor=.05, iou_threshold=.45, device='cpu',
              target_ids={0}, confidence_threshold=.25, use_masks=False, mask_min_area=0,
              mask_bottom_band_px=3, min_bbox_area_px=0, pixel_noise_sigma=0,
              get_clock=lambda: NS(now=lambda: NS(nanoseconds=2_000_000_000)),
                  get_logger=lambda: NS(info=lambda _: None, debug=lambda _: None), batch_outcome_publisher=NS(publish=lambda _: None),
              _warn_bounded=lambda *a: None,
              _observation_configs={c: c for c in CAMERA_ORDER})
    def fatal(message, cause=None): raise RuntimeError(message) from cause
    node._fatal = fatal
    env = dict(np=np, math=math, time=time, PendingFrame=PendingFrame, CAMERA_ORDER=CAMERA_ORDER,
               stamp_parts_to_ns=stamp_parts_to_ns, image_content_sha256=image_content_sha256,
               source_frame_id=source_frame_id, frame_member=frame_member, BatchContractError=BatchContractError,
               MAX_FUTURE_IMAGE_STAMP_S=MAX_FUTURE_IMAGE_STAMP_S, validate_batch_results=validate_batch_results,
               _DirectGzImagePayload=type('Unused', (), {}), image_msg_to_bgr8=decoder_env[decoder.name],
               _BatchTiming=lambda **kw: NS(**kw), select_best_detection=select_best_detection,
               diagnostics_from_message=lambda value: value)
    for name in ['_publish_batch_outcome', '_image_callback', '_process_batch', '_process_frames',
                 '_process_frames_once', '_predict_batch', '_prepare_result', '_prepare_selection', '_observation_message']:
        setattr(node, name, method(name, **env).__get__(node))
    node.batcher = FourCameraBatcher(on_event=node._publish_batch_outcome)
    calls, observations = [], []
    def predict(**kw):
        # Started intent must already be durable before an actual model call.
        rows = list(read_journal(journal.path))
        assert rows[-1]['status'] == 'inference_started'
        tags = [int(im[0, 0, 0]) for im in kw['source']]
        calls.append(tags)
        return [NS(boxes=Boxes(tag), masks=None) for tag in tags]
    node.model = NS(predict=predict)
    node._camera_observation_from_diagnostics = lambda value, config: CameraObservation(
        camera_id=config, timestamp_s=1.000000007,
        pixel_uv=(value['selected_u'], value['selected_v']), detection_valid=True)
    def publish(item, image, selection, timing, *, source_batch_id):
        message = node._observation_message(item.camera_id, selection, source_batch_id=source_batch_id)
        observations.append(CameraObservation.from_json(message.data))
    node._publish_result = publish  # transport stub; serialization above is actual method
    images = []
    for i, cid in reversed(list(enumerate(CAMERA_ORDER))):
        msg = NS(encoding='bgr8', height=2, width=8, step=24,
                 data=bytes([i] * 48), header=NS(stamp=NS(sec=1, nanosec=7)))
        images.append((cid, msg))
        node._image_callback(cid, msg)
    assert calls == [[0, 1], [2, 3], [4]]
    assert [o.camera_id for o in observations] == list(CAMERA_ORDER)
    assert [o.pixel_uv for o in observations] == [(i + .5, 2) for i in range(5)]
    assert len({o.source_batch_id for o in observations}) == 1
    assert len({o.detector_invocation_id for o in observations}) == 3
    assert len({o.source_frame_id for o in observations}) == 5
    assert all(o.capture_stamp_ns == 1_000_000_007 and o.producer_epoch == 'epoch' for o in observations)
    for cid, msg in images: node._image_callback(cid, msg)
    assert len(calls) == 3 and len(observations) == 5
    rows = list(read_journal(journal.path))
    assert sum(r['status'] == 'published' for r in rows) == 1
    assert sum(r['status'] == 'member_published' for r in rows) == 5
    assert sum(r['status'] == 'image_duplicate' for r in rows) == 5
    journal.close()


def test_outcome_qos_is_reliable_retained_and_bounded():
    tree = ast.parse(NODE.read_text())
    qos = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_outcome_qos')
    qos.returns = None
    env = dict(QoSProfile=lambda **kw: kw, OUTCOME_HISTORY_DEPTH=OUTCOME_HISTORY_DEPTH,
               HistoryPolicy=NS(KEEP_LAST='last'), ReliabilityPolicy=NS(RELIABLE='reliable'),
               DurabilityPolicy=NS(TRANSIENT_LOCAL='retained'))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[qos], type_ignores=[])), '<qos>', 'exec'), env)
    assert env['_outcome_qos']() == dict(history='last', depth=4096,
                                        reliability='reliable', durability='retained')


@pytest.mark.parametrize("exception_type", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("error_message", ["injected model fault", ""])
def test_model_exception_has_durable_started_error_and_cycle_abort(tmp_path, exception_type, error_message):
    import time
    import math
    import numpy as np
    from perception.core.four_camera_batch import validate_batch_results, CAMERA_ORDER
    writer = OutcomeJournal(tmp_path / 'events.jsonl', 'epoch')
    node = NS(_producer_epoch='epoch', _cycle_sequence=0, _outcome_journal=writer,
              torchscript_detection_only=False, inference_chunk=2, image_size=960,
              predict_conf_floor=.05, iou_threshold=.45, device='cpu', _clock_s=lambda: 1,
              get_clock=lambda: NS(now=lambda: NS(nanoseconds=1_000_000_000)),
                  get_logger=lambda: NS(info=lambda _: None, debug=lambda _: None), batch_outcome_publisher=NS(publish=lambda _: None))
    def fail(**kwargs): raise exception_type(error_message)
    node.model = NS(predict=fail)
    env = dict(np=np, time=time, math=math, frame_member=frame_member, CAMERA_ORDER=CAMERA_ORDER,
               BatchContractError=BatchContractError, validate_batch_results=validate_batch_results)
    for name in ['_publish_batch_outcome', '_predict_batch', '_process_frames']:
        setattr(node, name, method(name, **env).__get__(node))
    node._process_frames_once = lambda frames, **kw: node._predict_batch([f.payload for f in frames])
    frame = PendingFrame('camera_A', 100, 1, 0, np.zeros((2, 2, 3)), 'frame-one', 'a' * 64)
    with pytest.raises(exception_type, match=error_message or None): node._process_frames((frame,))
    rows = list(read_journal(writer.path))
    assert [r['status'] for r in rows] == ['selected', 'inference_started', 'inference_error', 'aborted']
    assert rows[1]['invocation_id'] == rows[2]['invocation_id']
    assert rows[2]['reason'] == rows[3]['reason'] == (error_message or exception_type.__name__)
    assert len({r['source_batch_id'] for r in rows}) == 1
    assert not any(r['status'] == 'member_published' for r in rows)
    writer.close()


@pytest.mark.parametrize('alteration', ['change_payload', 'remove_record'])
def test_journal_integrity_does_not_accept_changed_or_missing_events(tmp_path, alteration):
    writer = OutcomeJournal(tmp_path / 'events.jsonl', 'producer')
    writer.append(dict(status='selected'))
    writer.append(dict(status='inference_started'))
    writer.append(dict(status='inference_error'))
    writer.close()
    lines = writer.path.read_text().splitlines()
    if alteration == 'change_payload':
        event = json.loads(lines[1]); event['status'] = 'inference_completed'
        lines[1] = json.dumps(event)
    else:
        lines.pop(1)
    writer.path.write_text('\n'.join(lines) + '\n')
    with pytest.raises(ValueError, match='hash chain'): list(read_journal(writer.path))


def test_shutdown_accounts_for_unselected_pending_images_once():
    events = []
    batcher = FourCameraBatcher(on_event=events.append)
    frame = PendingFrame('camera_A', 100, 1, 0, object(), 'frame-id', 'a' * 64)
    batcher.offer(frame)
    batcher.close(); batcher.close()
    assert len(events) == 1
    assert events[0]['status'] == 'incomplete_shutdown'
    assert events[0]['members'][0]['source_frame_id'] == 'frame-id'
    assert events[0]['missing_camera_ids'] == ['camera_B', 'camera_C', 'camera_D', 'camera_E']


def test_shutdown_journal_does_not_depend_on_live_dds_or_ros_clock(tmp_path):
    writer = OutcomeJournal(tmp_path / 'events.jsonl', 'epoch')
    node = NS(_outcome_journal=writer, _outcome_transport_enabled=False,
              _clock_high_water_s=12.5,
              get_clock=lambda: pytest.fail('shutdown must not consult the ROS clock'),
                  get_logger=lambda: NS(info=lambda _: None, debug=lambda _: None),
              batch_outcome_publisher=NS(publish=lambda _: pytest.fail('DDS must not be used after quiescence')))
    method('_append_session_stopped')(node)
    writer.close()
    stopped = list(read_journal(writer.path))[-1]
    assert stopped['status'] == 'session_stopped'
    assert stopped['publish_stamp_s'] == 12.5


def test_terminal_request_quiesces_detector_and_binds_durable_marker(tmp_path):
    import threading
    from perception.core.four_camera_batch import FourCameraBatcher
    from unav_common.terminal_stop import (
        TerminalStopAck,
        TerminalStopRequest,
        terminal_stop_ack_from_json,
        terminal_stop_ack_to_json,
        terminal_stop_request_from_json,
        terminal_stop_request_to_json,
    )

    writer = OutcomeJournal(tmp_path / 'events.jsonl', 'epoch')
    published = []
    node = NS(
        _outcome_journal=writer,
        _outcome_transport_enabled=True,
        _clock_high_water_s=12.5,
        _terminal_stopped=False,
        _terminal_stop_request_id='',
        _session_stopped_appended=False,
        _terminal_lifecycle_lock=threading.RLock(),
        _async_pending_lock=threading.RLock(),
        _async_pending={},
        get_clock=lambda: NS(now=lambda: NS(nanoseconds=12_500_000_000)),
        get_logger=lambda: NS(info=lambda _: None, debug=lambda _: None),
        batch_outcome_publisher=NS(publish=lambda message: published.append(message)),
        _terminal_stop_ack_pub=NS(publish=lambda message: published.append(message)),
    )
    env = dict(
        terminal_stop_request_from_json=terminal_stop_request_from_json,
        TerminalStopAck=TerminalStopAck,
        terminal_stop_ack_to_json=terminal_stop_ack_to_json,
        frame_member=frame_member,
    )
    for name in ('_publish_batch_outcome', '_append_session_stopped',
                 '_terminal_stop_request_cb'):
        setattr(node, name, method(name, **env).__get__(node))
    node._fatal = lambda message, cause=None: pytest.fail(message)
    node.batcher = FourCameraBatcher(on_event=node._publish_batch_outcome)
    request = TerminalStopRequest('request-1', 'run-1', 'goal_reached', 10)

    node._terminal_stop_request_cb(
        NS(data=terminal_stop_request_to_json(request))
    )
    try:
        assert node._terminal_stopped
        assert not node._outcome_transport_enabled
        ack = terminal_stop_ack_from_json(published[-1].data)
        assert ack.component == 'detector'
        assert ack.request_id == request.request_id
        stopped = list(read_journal(writer.path))[-1]
        assert stopped['status'] == 'session_stopped'
        assert stopped['terminal_stop_request_id'] == request.request_id
    finally:
        writer.close()


def test_async_drop_and_all_expired_drain_remain_observable():
    import threading
    from perception.core.four_camera_batch import CAMERA_ORDER, MAX_FUTURE_IMAGE_STAMP_S
    events = []
    node = NS(_async_pending={}, _async_pending_lock=threading.Lock(),
              _async_last_seen_stamp_ns={c: -1 for c in CAMERA_ORDER},
              _async_last_seen_content_sha256={c: '' for c in CAMERA_ORDER},
              max_pending_wall_s=.5, _publish_batch_outcome=events.append)
    offer = method('_offer_async_frame', frame_member=frame_member)
    frame = PendingFrame('camera_A', 100, 1, 0, object(), 'old', 'a' * 64)
    offer(node, frame)
    offer(node, replace(frame, stamp_ns=200, receive_wall_s=.1, source_frame_id='new'))
    offer(node, replace(frame, stamp_ns=200, receive_wall_s=.2, source_frame_id='new'))
    drain = method('_drain_async_pending', frame_member=frame_member, CAMERA_ORDER=CAMERA_ORDER,
                   MAX_FUTURE_IMAGE_STAMP_S=MAX_FUTURE_IMAGE_STAMP_S,
                   time=NS(perf_counter=lambda: 1))
    drain(node)
    assert [e['status'] for e in events] == ['frame_replaced', 'image_duplicate', 'incomplete_timeout']
    assert events[-1]['members'][0]['source_frame_id'] == 'new'
    assert not node._async_pending


def test_shared_journal_preserves_manager_stage_and_concurrent_sequence(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from unav_common.camera_outcomes import OutcomeJournal as CommonJournal
    writer = CommonJournal(tmp_path / 'manager.jsonl', 'manager-process')
    def append(camera):
        for _ in range(5):
            writer.append(dict(stage='manager', status='received', camera_id=camera, publish_stamp_s=1))
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(append, ['left', 'right']))
    writer.close()
    rows = list(read_journal(writer.path))
    assert [r['event_seq'] for r in rows] == list(range(1, 11))
    assert len({r['event_id'] for r in rows}) == 10
    assert all(r['stage'] == 'manager' and r['producer_epoch'] == 'manager-process' for r in rows)
    assert {r['camera_id'] for r in rows} == {'left', 'right'}


def test_journal_overload_preserves_all_prior_durable_records(tmp_path):
    writer = OutcomeJournal(tmp_path / 'events.jsonl', 'producer', max_bytes=2048)
    saved = []
    with pytest.raises(RuntimeError, match='byte limit'):
        for index in range(100):
            saved.append(writer.append(dict(stage='detector', status='selected', source_batch_id=str(index))))
    writer.close()
    assert 0 < len(saved) < 100
    assert list(read_journal(writer.path)) == saved
    assert writer.path.stat().st_size <= 2048


def test_host_inference_timing_excludes_journal_and_transport_delay():
    from perception.core.four_camera_batch import CAMERA_ORDER, validate_batch_results
    wall, events = [0], []
    def record(event):
        events.append(event)
        wall[0] += 10  # controlled expensive fsync/transport, outside model call
    def predict(**kw):
        wall[0] += 2
        return [NS(boxes=[])]
    node = NS(inference_chunk=2, image_size=960, predict_conf_floor=.05, iou_threshold=.45,
              device='cpu', _active_cycle=('cycle', ['camera_A']),
              _active_members={'camera_A': {'camera_id': 'camera_A'}},
              _clock_s=lambda: wall[0], _publish_batch_outcome=record, model=NS(predict=predict))
    method('_predict_batch', CAMERA_ORDER=CAMERA_ORDER, BatchContractError=BatchContractError,
           validate_batch_results=validate_batch_results, time=NS(perf_counter=lambda: wall[0]))(node, [object()])
    assert events[0]['inference_scheduled_stamp_s'] == 0
    assert events[1]['inference_start_stamp_s'] == 10
    assert events[1]['inference_finish_stamp_s'] == 12
    assert events[1]['inference_wall_ms'] == 2000
