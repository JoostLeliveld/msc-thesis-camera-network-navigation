from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
PERCEPTION_SRC = ROOT / "src" / "perception"
if str(PERCEPTION_SRC) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_SRC))

from perception.core.four_camera_batch import (
    CAMERA_ORDER,
    BatchContractError,
    FourCameraBatcher,
    PendingFrame,
    frame_age_at_publish_s,
    stamp_parts_to_ns,
    validate_batch_results,
)


def _frame(camera_id: str, stamp_ns: int, wall_s: float = 0.0) -> PendingFrame:
    return PendingFrame(
        camera_id=camera_id,
        stamp_ns=stamp_ns,
        receive_stamp_s=stamp_ns * 1.0e-9,
        receive_wall_s=wall_s,
        payload=f"{camera_id}:{stamp_ns}",
    )


class _ValidResult:
    boxes = ()


class _MissingBoxesResult:
    boxes = None


def test_batch_is_emitted_in_fixed_order_independent_of_callback_order() -> None:
    batcher = FourCameraBatcher(max_stamp_skew_s=0.01, max_pending_wall_s=1.0)
    decisions = [
        batcher.offer(_frame("camera_D", 1_000_000_000, 0.00)),
        batcher.offer(_frame("camera_B", 1_000_000_000, 0.01)),
        batcher.offer(_frame("camera_A", 1_000_000_000, 0.02)),
        batcher.offer(_frame("camera_E", 1_000_000_000, 0.03)),
        batcher.offer(_frame("camera_C", 1_000_000_000, 0.04)),
    ]

    assert [decision.status for decision in decisions] == (
        ["accepted_waiting"] * (len(CAMERA_ORDER) - 1) + ["batch_ready"]
    )
    assert decisions[-1].batch is not None
    assert tuple(item.camera_id for item in decisions[-1].batch) == CAMERA_ORDER
    assert tuple(item.payload for item in decisions[-1].batch) == tuple(
        f"{camera_id}:1000000000" for camera_id in CAMERA_ORDER
    )


def test_pending_frame_is_latest_only_and_a_used_stamp_is_never_reused() -> None:
    batcher = FourCameraBatcher(max_stamp_skew_s=0.01, max_pending_wall_s=1.0)
    assert batcher.offer(_frame("camera_A", 100, 0.00)).status == "accepted_waiting"
    assert batcher.offer(_frame("camera_A", 200, 0.01)).status == "accepted_replaced"
    for camera_id in ("camera_B", "camera_C", "camera_D"):
        assert batcher.offer(_frame(camera_id, 200, 0.02)).batch is None
    ready = batcher.offer(_frame("camera_E", 200, 0.03))

    assert ready.status == "batch_ready"
    assert ready.batch is not None
    assert [item.stamp_ns for item in ready.batch] == [200] * len(CAMERA_ORDER)
    assert batcher.offer(_frame("camera_A", 200, 0.04)).status == "duplicate"
    assert batcher.offer(_frame("camera_A", 199, 0.05)).status == "out_of_order"
    assert batcher.pending_camera_ids == ()


def test_frames_from_different_rounds_are_never_batched_together() -> None:
    """A late frame must not be able to break the round that is still forming.

    This used to assert that the batcher REJECTED the set for stamp skew and dropped
    the odd camera out.  That rejection was the production failure: with one pending
    slot per camera, a newer frame overwrote its camera mid-cycle and the set that
    completed mixed one camera's round with another's.  In a live five-camera run
    every batch was rejected this way and the node published nothing, while the
    cameras themselves were delivering byte-identical stamps in 101 of 102 rounds.

    Grouping by stamp makes the guarantee structural rather than a tolerance test:
    frames a period apart land in different rounds, so they cannot be batched
    together at all, and the earlier round completes when its missing frame arrives.
    Staleness is still bounded, by wall-clock expiry.
    """
    batcher = FourCameraBatcher(max_stamp_skew_s=0.05, max_pending_wall_s=1.0)
    batcher.offer(_frame("camera_A", 1_000_000_000, 0.00))
    for camera_id in ("camera_B", "camera_C", "camera_D"):
        batcher.offer(_frame(camera_id, 1_200_000_000, 0.01))
    decision = batcher.offer(_frame("camera_E", 1_200_000_000, 0.02))

    # the later round is one frame short, and the earlier round is untouched
    assert decision.status == "accepted_waiting"
    assert decision.batch is None
    assert set(batcher.pending_camera_ids) == set(CAMERA_ORDER)

    ready = batcher.offer(_frame("camera_A", 1_200_000_000, 0.03))
    assert ready.status == "batch_ready"
    assert ready.batch is not None
    assert [item.stamp_ns for item in ready.batch] == [1_200_000_000] * len(CAMERA_ORDER)
    # and the stranded earlier round is gone, not left to surface as a later drop
    assert batcher.pending_camera_ids == ()


def test_pending_wall_timeout_prevents_a_paused_camera_frame_from_later_use() -> None:
    batcher = FourCameraBatcher(max_stamp_skew_s=1.0, max_pending_wall_s=0.10)
    batcher.offer(_frame("camera_A", 10, 0.00))
    batcher.offer(_frame("camera_B", 10, 0.01))
    batcher.offer(_frame("camera_C", 10, 0.02))
    batcher.offer(_frame("camera_D", 10, 0.03))

    decision = batcher.offer(_frame("camera_E", 10, 0.20))

    assert decision.batch is None
    assert decision.dropped_camera_ids == ("camera_A", "camera_B", "camera_C", "camera_D")
    assert batcher.pending_camera_ids == ("camera_E",)


def test_stamp_validation_is_exact_and_rejects_malformed_values() -> None:
    assert stamp_parts_to_ns(12, 345) == 12_000_000_345
    for sec, nanosec in [(-1, 0), (0, -1), (0, 1_000_000_000), ("1", 0), (True, 0)]:
        with pytest.raises(BatchContractError):
            stamp_parts_to_ns(sec, nanosec)


def test_future_image_stamps_are_an_integrity_fault_not_a_zero_age_frame() -> None:
    assert frame_age_at_publish_s(publish_stamp_s=10.0, image_stamp_s=10.003) == 0.0
    with pytest.raises(BatchContractError, match="materially in the future"):
        frame_age_at_publish_s(publish_stamp_s=10.0, image_stamp_s=10.006)


def test_result_validation_fails_closed_before_camera_identity_can_shift() -> None:
    valid = [_ValidResult() for _ in CAMERA_ORDER]
    assert validate_batch_results(valid) == tuple(valid)
    # written as "one short" rather than a literal count, so the batch width can
    # change without this test silently checking the wrong thing
    short = valid[:-1]
    for malformed in (
        None,
        short,
        [*short, None],
        [*short, object()],
        [*short, _MissingBoxesResult()],
        "not-results",
    ):
        with pytest.raises(BatchContractError):
            validate_batch_results(malformed)


def test_runtime_source_has_one_native_model_and_the_complete_operational_contract() -> None:
    source = (
        ROOT
        / "src/perception/perception/nodes/batched_four_camera_yolo_node.py"
    ).read_text(encoding="utf-8")

    # Exactly one Ultralytics model is constructed, and the compiled diagnostic
    # path owns exactly one CUDA copy of its graph. A second construction of
    # either kind puts a second compiled graph beside Gazebo's renderer on the
    # 4 GiB laptop GPU. (Asserted on the construction itself rather than on one
    # spelling of the load path, which the torchscript metadata split changed.)
    assert source.count("= YOLO(") == 1
    assert source.count("torch.jit.load(") == 1
    # One inference CYCLE per timestamp batch, but the cycle may be split into
    # consecutive chunks: measured on this 4 GiB card, five 960-px images in one
    # call reserve 2558 MiB and run out of memory beside Gazebo's five cameras,
    # while chunks of two reserve 1418 MiB and fit. What must stay true is that
    # every image is predicted exactly once and the results are reassembled in
    # camera order -- so that is what is asserted, rather than the batch width of
    # a single predict call.
    assert source.count("self.model.predict(**kwargs)") == 1
    assert '"source": group' in source
    assert '"batch": len(group)' in source
    assert "results.extend(validate_batch_results(part, len(group)))" in source
    assert "if len(results) != len(images_bgr):" in source
    assert source.count("else self._predict_batch(images)") == 1
    assert "use_torchscript" not in source
    assert "ground_truth" not in source
    assert "oracle_" not in source
    assert "self.subscriptions = []" not in source
    assert "self.camera_subscriptions = []" in source
    assert "_publish_failed_batch" not in source
    assert "publishing misses" not in source
    assert 'self.get_logger().fatal(message)' in source
    assert 'raise RuntimeError(message) from cause' in source
    for fatal_fault in (
        "malformed input contract",
        "image conversion failed",
        "batch inference failed",
        "malformed multicamera results",
        "malformed result contract",
        "CameraObservation contract failed",
    ):
        assert fatal_fault in source
    for topic in (
        "/external_camera/image_raw",
        "/external_camera_b/image_raw",
        "/external_camera_c/image_raw",
        "/external_camera_d/image_raw",
    ):
        assert topic in source
    for timing_field in (
        "yolo_inference_ms",
        "detector_callback_ms",
        "yolo_receive_stamp",
        "yolo_start_stamp",
        "yolo_finish_stamp",
        "yolo_publish_stamp",
        "yolo_latency_s",
        "frame_age_at_publish_s",
    ):
        assert timing_field in source
    assert 'f"/perception/{camera_id}/pixel_pose"' in source
    assert 'f"/perception/{camera_id}/detection_diagnostics"' in source
    assert 'f"/perception/camera_observation/{camera_id}"' in source
    assert 'RUNTIME_CONTRACT_TOPIC = "/perception/four_camera_detector_runtime_contract"' in (
        ROOT / "src/perception/perception/core/four_camera_runtime_contract.py"
    ).read_text(encoding="utf-8")
    assert "DurabilityPolicy.TRANSIENT_LOCAL" in source
    assert "ReliabilityPolicy.RELIABLE" in source
    assert "self._publish_runtime_contract()" in source
    assert "model checkpoint bytes changed while the batched detector was loading" in source
    assert "source_hashes_from_paths" in source
    assert "future image-stamp integrity fault" in source
    assert 'self.declare_parameter("synchronization_mode", "strict")' in source
    assert 'self.declare_parameter("async_coalesce_wall_s", 0.02)' in source
    assert "self._drain_async_pending" in source
    assert "diagnostic-only" in source
    assert "MultiThreadedExecutor(num_threads=2)" in source
    assert "ReentrantCallbackGroup" in source
    assert "MutuallyExclusiveCallbackGroup" in source
    assert 'self.declare_parameter("input_transport", "ros")' in source
    assert "_gz_image_callback" in source
    assert "subscribe_raw(" in source
    assert '"ignition.msgs.Image"' in source
    assert "direct_gz input is diagnostic-only" in source
    assert "compiled runtime is diagnostic-only" in source
    assert "compiled model bytes changed while the batched detector was loading" in source
    assert "runtime_trace " in source
