"""Deterministic audit probes. Assertions describe observed behavior, including defects.
AST extraction executes unchanged methods without importing ROS/model runtimes.
"""
import ast
from collections import deque
import math
from pathlib import Path
import sys
import time
import threading
from types import SimpleNamespace as NS
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/perception'))
from perception.core.four_camera_batch import CAMERA_ORDER, FourCameraBatcher, PendingFrame, BatchContractError, validate_batch_results
sys.path.insert(0, str(ROOT / 'src/reliability'))
from reliability.source_batch_buffer import SourceBatchBuffer
from reliability.contracts import CameraObservation, ContractValidationError
from perception.core.detector_outcomes import frame_member


def method(path, cls, name, **env):
    tree = ast.parse((ROOT / path).read_text())
    c = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    f = next(n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name)
    code = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), f], type_ignores=[])
    ns = dict(math=math, np=np, time=time, CAMERA_ORDER=CAMERA_ORDER, BatchContractError=BatchContractError, validate_batch_results=validate_batch_results, frame_member=frame_member, **env)
    exec(compile(ast.fix_missing_locations(code), str(ROOT / path), 'exec'), ns)
    return ns[name]

BATCH = 'src/perception/perception/nodes/batched_four_camera_yolo_node.py'
MANAGER = 'src/reliability/reliability/nodes/camera_manager_node.py'
SINGLE = 'src/perception/perception/nodes/yolo_robot_detector_node.py'
SCHEDULED = 'src/perception/perception/nodes/scheduled_camera_detector_node.py'


def frame(cid, stamp, wall=0, value=0):
    return PendingFrame(cid, stamp, stamp / 1e9, wall, np.full((2, 3, 3), value, np.uint8))


def receiver():
    callback = method(MANAGER, 'CameraManagerNode', '_observation_callback',
                      CameraObservation=CameraObservation, ContractValidationError=ContractValidationError)
    receive = method(MANAGER, 'CameraManagerNode', '_receive_observation',
                     CameraObservation=CameraObservation, ContractValidationError=ContractValidationError,
                     deque=deque)
    s = NS(camera_ids=list(CAMERA_ORDER), _pending_source_batches={}, _latest={},
           _ready_source_batch_stamp_s=-math.inf, _ready_source_batch_id=None,
           _last_decided_source_batch_id=None, require_source_batch_id=True,
           get_logger=lambda: NS(warn=lambda _: None))
    s.events = []
    s._input_lock, s._decision_lock = threading.RLock(), threading.RLock()
    s._snapshot_inputs = lambda bid: NS(source_batch_id=bid, contracts=tuple(s._latest.values()))
    s._receive_observation = lambda cid, msg: receive(s, cid, msg)
    s._publish_batch_outcome = s.events.append
    s._check_batch_clock = lambda: None
    s._source_batch_buffer = SourceBatchBuffer(CAMERA_ORDER, on_event=s.events.append)
    def send(cid, bid, stamp, value=0):
        observation = CameraObservation(camera_id=cid, source_batch_id=bid, timestamp_s=stamp,
            pixel_uv=(value, 0.), producer_epoch='audit04', capture_stamp_ns=round(stamp*1e9),
            source_frame_id=f'frame:audit04:{cid}:{round(stamp*1e9)}:{value}',
            detector_invocation_id=f'{bid}/chunk/0')
        callback(s, cid)(NS(data=observation.to_json()))
    return s, send


def test_staggered_rounds_order_duplicates_and_delayed_input():
    b = FourCameraBatcher()
    for i, c in enumerate(reversed(CAMERA_ORDER)):
        d = b.offer(frame(c, 1_000_000_000, i * .01, CAMERA_ORDER.index(c)))
    assert [int(f.payload[0, 0, 0]) for f in d.batch] == list(range(5))
    assert b.offer(frame(CAMERA_ORDER[0], 1_000_000_000)).status == 'duplicate'
    assert b.offer(frame(CAMERA_ORDER[0], 999_999_999)).status == 'out_of_order'


def test_distinct_rounds_inside_tolerance_can_merge():
    b = FourCameraBatcher()
    b.offer(frame(CAMERA_ORDER[0], 1_000_000_000, value=1))
    for c in CAMERA_ORDER[1:]:
        d = b.offer(frame(c, 1_040_000_000, value=2))
    assert [int(f.payload[0, 0, 0]) for f in d.batch] == [1, 2, 2, 2, 2]


def test_equal_stamp_distinct_image_is_indistinguishable():
    b = FourCameraBatcher()
    b.offer(frame(CAMERA_ORDER[0], 100, value=1))
    assert b.offer(frame(CAMERA_ORDER[0], 100, value=2)).status == 'duplicate'


def test_clock_reset_and_restart_identity_collision():
    ids = []
    for b in [FourCameraBatcher(), FourCameraBatcher()]:
        for c in CAMERA_ORDER:
            d = b.offer(frame(c, 100_000_000_000))
        ids.append('strict:' + ','.join(f'{f.camera_id}@{f.stamp_ns}' for f in d.batch))
        assert all(b.offer(frame(c, 0)).status == 'out_of_order' for c in CAMERA_ORDER)
    assert ids[0] == ids[1]


def test_absent_camera_expiry_reports_healthy_cameras_and_requires_call():
    b = FourCameraBatcher()
    for c in CAMERA_ORDER[:-1]:
        b.offer(frame(c, 100))
    assert b.bucket_report == ((100, CAMERA_ORDER[:-1]),)
    assert b.expire(1) == CAMERA_ORDER[:-1]
    assert b.pending_camera_ids == ()


def test_burst_is_bounded_and_superseded_rounds_are_reported():
    events = []
    b = FourCameraBatcher(max_stamp_skew_s=0, on_event=events.append)
    for i in range(1000):
        b.offer(frame(CAMERA_ORDER[0], i, wall=0))
    assert len(b.bucket_report) == 32
    for c in CAMERA_ORDER[1:]:
        d = b.offer(frame(c, 999, wall=.1))
    assert d.batch is not None and d.dropped_camera_ids == (CAMERA_ORDER[0],)
    assert len(events) == 999
    assert {e["status"] for e in events} == {"incomplete_capacity", "superseded_by_complete_batch"}
    assert b.bucket_report == ()


def test_partial_manager_batches_bounded_and_evictions_identified():
    s, send = receiver()
    for i in range(1000):
        for c in CAMERA_ORDER[:-1]:
            send(c, str(i), i)
    assert len(s._source_batch_buffer.pending) == 64
    assert len(s.events) == 936
    assert all(e["missing_camera_ids"] == [CAMERA_ORDER[-1]] for e in s.events)
    assert s._ready_source_batch_id is None


def test_manager_equal_time_new_identity_and_reset_discarded():
    s, send = receiver()
    for bid, stamp in [('first', 100), ('distinct', 100), ('reset', 0)]:
        for c in CAMERA_ORDER:
            send(c, bid, stamp)
    assert s._ready_source_batch_id == 'first'


def test_duplicate_delivery_and_fast_ticks_do_not_repeat_active_decision():
    s, send = receiver()
    calls = []
    s.get_clock = lambda: NS(now=lambda: NS(nanoseconds=2_000_000_000))
    s._map_observations = lambda now: list(s._latest.values())
    s._camera_mapping_reasons = {}
    s._publish_map_observations = lambda obs: None
    s.fusion_mode, s.active_pub = True, object()
    s._decide_fused = lambda *args, **kw: calls.append(kw['source_batch_id'])
    once = method(MANAGER, 'CameraManagerNode', '_decide_once')
    s._decide_once = lambda bid: once(s, bid)
    next_ready = method(MANAGER, 'CameraManagerNode', '_decide_next_ready_batch')
    s._decide_next_ready_batch = lambda: next_ready(s)
    tick = method(MANAGER, 'CameraManagerNode', '_decide')
    for _ in range(3):
        for c in CAMERA_ORDER:
            send(c, 'physical', 1)
        for _ in range(20):
            tick(s)
    assert calls == ['physical']


def test_chunk_result_count_mismatch_fails_before_next_chunk():
    predict = method(BATCH, 'BatchedFourCameraYoloNode', '_predict_batch')
    calls = []
    def fake(**kw):
        ids = [int(x[0, 0, 0]) for x in kw['source']]
        calls.append(ids)
        return {0: [0], 2: [2, 3, 99], 4: [4]}[ids[0]]
    s = NS(inference_chunk=2, image_size=960, predict_conf_floor=0, iou_threshold=.45,
           device='cpu', model=NS(predict=fake))
    with pytest.raises(BatchContractError, match="expected 2"):
        predict(s, [frame(c, 0, value=i).payload for i, c in enumerate(CAMERA_ORDER)])
    assert calls == [[0, 1]]


def test_single_detector_repeated_delivery_reinfers_same_image():
    process = method(SINGLE, 'YoloRobotDetectorNode', '_process_image', image_msg_to_bgr8=lambda m: m.image)
    calls = []
    s = NS(debug_frame_dir='', get_clock=lambda: NS(now=lambda: NS(nanoseconds=10)),
           _predict=lambda im: calls.append(im) or [], _publish_diagnostics=lambda *a: None)
    msg = NS(image=np.zeros((2, 3, 3)), header=NS(stamp=1))
    process(s, msg, 0); process(s, msg, 0)
    assert len(calls) == 2


def test_scheduled_exception_claims_frame_without_terminal_observation():
    tick = method(SCHEDULED, 'ScheduledCameraDetector', '_tick')
    calls, outputs = [], []
    def fail(**kw):
        calls.append(kw); raise RuntimeError('controlled inference failure')
    s = NS(belief=(0, 0), cams=[('synthetic', '', '')], _cov_at=lambda *a: 1,
           selection_mode='coverage_best_with_fallback', min_cov=0,
           latest={'synthetic': object()}, _stamp_s=lambda m: 1,
           _last_processed_stamp_s={'synthetic': -math.inf},
           bridge=NS(imgmsg_to_cv2=lambda *a: np.zeros((2,3,3))),
           model=NS(predict=fail), imgsz=960, conf=.25, iou=.45, device='cpu',
           get_logger=lambda: NS(warn=lambda m: None), _publish_observation=lambda **kw: outputs.append(kw))
    tick(s); tick(s)
    assert len(calls) == 1 and outputs == []


def test_decoder_padding_and_rgb_coordinates():
    tree = ast.parse((ROOT / 'src/perception/perception/core/ros_image.py').read_text())
    f = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    f.returns = None
    for arg in f.args.args: arg.annotation = None
    ns = {'np': np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[f], type_ignores=[])), '<decoder>', 'exec'), ns)
    msg = NS(encoding='rgb8', height=2, width=1, step=4, data=bytes([1,2,3,99,4,5,6,88]))
    out = ns[f.name](msg)
    assert out.tolist() == [[[3,2,1]], [[6,5,4]]]
    assert out.flags.c_contiguous
    msg.step = 2
    with pytest.raises(ValueError): ns[f.name](msg)


def test_slow_inference_has_no_post_inference_age_rejection_and_partial_publish():
    from perception.core.four_camera_batch import MAX_FUTURE_IMAGE_STAMP_S, validate_batch_results
    process = method(BATCH, 'BatchedFourCameraYoloNode', '_process_frames_once',
                     MAX_FUTURE_IMAGE_STAMP_S=MAX_FUTURE_IMAGE_STAMP_S,
                     _DirectGzImagePayload=type('Unused', (), {}),
                     image_msg_to_bgr8=lambda x: x, _BatchTiming=lambda **kw: NS(**kw))
    clock, published = [1.0], []
    def predict(images):
        clock[0] = 101.0  # controlled 100-second inference, no real sleep
        return [NS(boxes=[]) for _ in images]
    def publish(item, image, selection, timing, **kw):
        published.append((item.camera_id, clock[0] - item.stamp_ns / 1e9))
        if item.camera_id == CAMERA_ORDER[1]: raise RuntimeError('publication failure')
    s = NS(_clock_s=lambda: clock[0], torchscript_detection_only=False,
           _active_members={c: {"camera_id": c} for c in CAMERA_ORDER},
           _publish_batch_outcome=lambda event: None,
           _predict_batch=predict, _prepare_result=lambda r: {}, _publish_result=publish)
    with pytest.raises(RuntimeError, match='publication failure'):
        process(s, tuple(frame(c, 1_000_000_000) for c in CAMERA_ORDER), source_batch_id="test")
    assert published == [(CAMERA_ORDER[0], 100.0), (CAMERA_ORDER[1], 100.0)]


def test_detector_cycle_abort_is_identified_and_restart_ids_differ():
    process = method(BATCH, 'BatchedFourCameraYoloNode', '_process_frames')
    identifiers = []
    for epoch in ['process-one', 'process-two']:
        events = []
        def fail(*args, **kw):
            raise RuntimeError('controlled partial publication')
        node = NS(_producer_epoch=epoch, _cycle_sequence=0, inference_chunk=2,
                  torchscript_detection_only=False,
                  _publish_batch_outcome=events.append, _process_frames_once=fail)
        with pytest.raises(RuntimeError):
            process(node, tuple(frame(c, 100) for c in CAMERA_ORDER))
        assert [e['status'] for e in events] == ['selected', 'aborted']
        assert events[0]['source_batch_id'] == events[1]['source_batch_id']
        assert node._active_cycle is None
        identifiers.append(events[0]['source_batch_id'])
    assert identifiers[0] != identifiers[1]


def test_detector_clock_reset_is_explicit_failure_not_frame_lockout():
    from perception.core.four_camera_batch import MAX_FUTURE_IMAGE_STAMP_S
    clock = method(BATCH, 'BatchedFourCameraYoloNode', '_clock_s',
                   MAX_FUTURE_IMAGE_STAMP_S=MAX_FUTURE_IMAGE_STAMP_S)
    stamp = [100_000_000_000]
    def fatal(message): raise RuntimeError(message)
    node = NS(_clock_high_water_s=None, _clock_lock=threading.Lock(), get_clock=lambda: NS(now=lambda: NS(nanoseconds=stamp[0])),
              _fatal=fatal)
    assert clock(node) == 100
    stamp[0] = 0
    with pytest.raises(RuntimeError, match='coordinated runtime restart'):
        clock(node)


def test_chunk_invocation_outcomes_preserve_camera_order_and_empty_misses():
    predict = method(BATCH, 'BatchedFourCameraYoloNode', '_predict_batch')
    events = []
    node = NS(inference_chunk=2, image_size=960, predict_conf_floor=0, iou_threshold=.45,
              device='cpu', _active_cycle=('cycle', list(CAMERA_ORDER)),
              _active_members={c: {'camera_id': c} for c in CAMERA_ORDER},
              _clock_s=lambda: 1, _publish_batch_outcome=events.append,
              model=NS(predict=lambda **kw: [NS(boxes=[], tag=int(im[0,0,0])) for im in kw['source']]))
    results = predict(node, [frame(c, 0, value=i).payload for i,c in enumerate(CAMERA_ORDER)])
    assert [r.tag for r in results] == list(range(5))
    assert [e['status'] for e in events] == ['inference_started', 'inference_completed'] * 3
    events = [e for e in events if e['status'] == 'inference_completed']
    assert [e['invocation_id'] for e in events] == ['cycle/chunk/0', 'cycle/chunk/1', 'cycle/chunk/2']
    assert [c for e in events for c in e['camera_ids']] == list(CAMERA_ORDER)
    assert all(e['status'] == 'inference_completed' for e in events)


def test_manager_failed_decision_cannot_retry_after_partial_publication():
    tick = method(MANAGER, 'CameraManagerNode', '_decide')
    calls, events = [], []
    def fail(bid):
        calls.append(bid)
        raise RuntimeError('controlled publisher failure')
    node = NS(_ready_source_batch_id='cycle', _last_decided_source_batch_id=None,
              _decide_once=fail, _publish_batch_outcome=events.append,
              _input_lock=threading.RLock(), _decision_lock=threading.RLock(),
              _snapshot_inputs=lambda bid: NS(source_batch_id=bid))
    with pytest.raises(RuntimeError): tick(node)
    tick(node)
    assert calls == ['cycle'] and events[0]['status'] == 'decision_error'


def test_active_launch_shuts_down_on_detector_or_manager_exit():
    tree = ast.parse((ROOT / 'src/experiments/experiments/core/visibility_launch_common.py').read_text())
    construction = {}
    for call in ast.walk(tree):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'Node':
            kw = {k.arg: k.value for k in call.keywords}
            name = kw.get('name')
            if isinstance(name, ast.Constant): construction[name.value] = kw
    for name in ['batched_four_camera_yolo', 'camera_manager_active']:
        shutdown = construction[name]['on_exit'].elts[0]
        assert shutdown.func.id == 'Shutdown'


def test_manager_wall_timer_reset_reports_integrity_failure():
    check = method(MANAGER, 'CameraManagerNode', '_check_batch_clock')
    events = []
    node = NS(_batch_clock_high_water_s=100, _publish_batch_outcome=events.append,
              get_clock=lambda: NS(now=lambda: NS(nanoseconds=0)))
    with pytest.raises(RuntimeError, match='coordinated runtime restart'): check(node)
    assert events == [dict(status='clock_reset', reason='coordinated runtime restart required')]
