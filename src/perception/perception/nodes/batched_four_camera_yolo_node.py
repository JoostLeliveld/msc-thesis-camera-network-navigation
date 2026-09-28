"""Single-model, deterministic multicamera YOLO inference runtime.

This node is intentionally specific to the commissioned camera registry.
It holds a bounded set of timestamp rounds and invokes one native
Ultralytics model with one image per registered camera in contract order. It never fills a
batch by repeating a previous image.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import base64
import math
import json
import zlib
import uuid
from pathlib import Path
import threading
import time
from typing import Any, NoReturn

import cv2
import numpy as np
import rclpy
import torch
from geometry_msgs.msg import PoseStamped
from rclpy.clock import Clock as RclpyClock, ClockType
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Header, String
from ultralytics import YOLO

import perception.core.four_camera_batch as four_camera_batch_module
import perception.core.four_camera_runtime_contract as runtime_contract_module
import perception.core.yolo_selection as yolo_selection_module
import unav_common.camera_outcomes as camera_outcomes_module
import perception.core.detector_outcomes as detector_outcomes_module
from perception.core.detector_outcomes import (
    OutcomeJournal, OUTCOME_HISTORY_DEPTH, DEFAULT_JOURNAL_MAX_BYTES,
    image_content_sha256, source_frame_id, frame_member, journal_path,
)
from perception.core.detection_diagnostics import (
    diagnostics_from_message,
    diagnostics_message,
)
from perception.core.four_camera_batch import (
    CAMERA_ORDER,
    BatchContractError,
    FourCameraBatcher,
    MAX_FUTURE_IMAGE_STAMP_S,
    PendingFrame,
    frame_age_at_publish_s,
    stamp_parts_to_ns,
    validate_batch_results,
)
from perception.core.four_camera_runtime_contract import (
    EXECUTABLE_SOURCE_KEY,
    FOUR_CAMERA_BATCH_SOURCE_KEY,
    RUNTIME_CONTRACT_SOURCE_KEY,
    RUNTIME_CONTRACT_TOPIC,
    YOLO_SELECTION_SOURCE_KEY,
    build_batched_runtime_contract,
    runtime_contract_json,
    sha256_file,
    source_hashes_from_paths,
)
from perception.core.ros_image import image_msg_to_bgr8
from perception.core.yolo_selection import select_best_detection, target_class_ids
from perception.nodes.yolo_robot_detector_node import (
    _as_bool,
    _confidence_logit,
    _selected_pixel_source_code,
)
from unav_common.terminal_stop import (
    TERMINAL_STOP_ACK_TOPIC,
    TERMINAL_STOP_REQUEST_TOPIC,
    TerminalStopAck,
    terminal_stop_ack_to_json,
    terminal_stop_request_from_json,
)


CAMERA_TOPICS = {
    "camera_A": "/external_camera/image_raw",
    "camera_B": "/external_camera_b/image_raw",
    "camera_C": "/external_camera_c/image_raw",
    "camera_D": "/external_camera_d/image_raw",
    "camera_E": "/external_camera_e/image_raw",
}


@dataclass(frozen=True)
class _DirectGzImagePayload:
    """BGR image and reconstructed ROS header from a Gazebo transport image."""

    header: Header
    image_bgr: np.ndarray


def _runtime_contract_qos() -> QoSProfile:
    """Retain exactly one detector identity for recorders that start later."""

    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def _outcome_qos() -> QoSProfile:
    return QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=OUTCOME_HISTORY_DEPTH,
                      reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL)


@dataclass(frozen=True)
class _CameraOutput:
    pixel_publisher: Any
    diagnostics_publisher: Any
    observation_publisher: Any


@dataclass(frozen=True)
class _BatchTiming:
    predict_start_stamp_s: float = math.nan
    predict_finish_stamp_s: float = math.nan
    inference_wall_ms: float = math.nan


class BatchedFourCameraYoloNode(Node):
    """Run one native YOLO model over strict or diagnostic contract-ordered inputs."""

    def __init__(self) -> None:
        super().__init__("batched_four_camera_yolo_node")

        self.declare_parameter("model_path", "")
        self.declare_parameter("runtime_backend", "native")
        self.declare_parameter("compiled_model_path", "")
        self.declare_parameter("torchscript_detection_only", False)
        self.declare_parameter("device", "")
        self.declare_parameter("cpu_num_threads", 2)
        self.declare_parameter("cpu_num_interop_threads", 1)
        self.declare_parameter("opencv_num_threads", 1)
        # 960, which is what every checkpoint was trained at. See the launch files
        # for the measurement behind it; the short version is that the measurement
        # this node produces IS the bottom edge of the mask, the mask is built at the
        # inference resolution, and a five-image batch at 960 costs 99 ms of the
        # 200 ms the 5 Hz cameras allow.
        self.declare_parameter("image_size", 960)
        self.declare_parameter("confidence_threshold", 0.25)
        # Anchor pre-filter passed to Ultralytics ``predict`` (``conf``). At the
        # default 0.0 every anchor survives to NMS and the segmentation head
        # builds a mask for each survivor -- ~140 ms/batch on the P2000. Raising
        # this floor below ``confidence_threshold`` prunes only anchors that
        # would be thresholded out anyway, so the final detections are identical
        # while NMS+mask cost drops ~3.5x (140 ms -> 39 ms). Kept at 0.0 by
        # default so the strict evidence path (which reads raw_best_score on
        # deep misses for availability modelling) is byte-for-byte unchanged.
        self.declare_parameter("predict_conf_floor", 0.0)
        self.declare_parameter("iou_threshold", 0.45)
        self.declare_parameter("class_name", "robot")
        self.declare_parameter("class_id", -1)
        self.declare_parameter("use_masks", True)
        self.declare_parameter("mask_min_area", 12.0)
        self.declare_parameter("mask_bottom_band_px", 3.0)
        self.declare_parameter("min_bbox_area_px", 0.0)
        self.declare_parameter("debug_frame_dir", "")
        self.declare_parameter("debug_crop_dir", "")
        self.declare_parameter("pixel_noise_sigma", 0.0)
        self.declare_parameter("seed", 0)
        self.declare_parameter("warmup_iters", 3)
        self.declare_parameter("max_batch_stamp_skew_s", 0.05)
        self.declare_parameter("max_pending_wall_s", 0.50)
        self.declare_parameter("synchronization_mode", "strict")
        self.declare_parameter("async_coalesce_wall_s", 0.02)
        self.declare_parameter("input_transport", "ros")
        self.declare_parameter("runtime_trace_period_s", 0.0)
        self.declare_parameter("outcome_journal_path", "")
        self.declare_parameter("outcome_journal_max_bytes", DEFAULT_JOURNAL_MAX_BYTES)
        # Diagnostic covariance attached to detector output. The camera manager replaces
        # this with the matched commissioned R of the selected correction model.
        self.declare_parameter("camera_observation_r_visible_uv", 2.5)
        self.declare_parameter("camera_observation_r_miss_uv", 40.0)
        # Which world's camera calibration these observations claim. This used to be
        # the literal "warehouse_full_4cam", so an observation captured in any other
        # world silently carried the wrong calibration identity.
        self.declare_parameter("calibration_world", "")
        # How many images of the timestamp batch go through the GPU at once. 0 means
        # the whole batch in one call. See _predict_batch for the memory measurement
        # that makes this necessary rather than optional on a 4 GiB card.
        self.declare_parameter("inference_chunk", 2)

        raw_model_path = str(self.get_parameter("model_path").value).strip()
        if not raw_model_path:
            raise RuntimeError("model_path must be set for batched_four_camera_yolo_node")
        model_path = Path(raw_model_path).expanduser()
        if not model_path.is_file():
            raise RuntimeError(f"model_path does not exist: {model_path}")
        if model_path.suffix.lower() != ".pt":
            raise RuntimeError(
                "batched commissioning requires a native Ultralytics .pt checkpoint; "
                f"got {model_path}"
            )
        self.model_path = model_path.resolve()
        self.runtime_backend = str(self.get_parameter("runtime_backend").value).strip().lower()
        raw_compiled_model_path = str(self.get_parameter("compiled_model_path").value).strip()
        if self.runtime_backend not in {"native", "torchscript"}:
            raise RuntimeError("runtime_backend must be 'native' or 'torchscript'")
        self.compiled_model_path = None
        if self.runtime_backend == "torchscript":
            compiled_path = Path(raw_compiled_model_path).expanduser()
            if not compiled_path.is_file() or compiled_path.suffix.lower() != ".torchscript":
                raise RuntimeError(
                    "torchscript backend requires compiled_model_path ending in .torchscript"
                )
            self.compiled_model_path = compiled_path.resolve()
        detection_only_value = self.get_parameter("torchscript_detection_only").value
        self.torchscript_detection_only = (
            detection_only_value
            if isinstance(detection_only_value, bool)
            else _as_bool(detection_only_value)
        )
        # Keep this a string end to end.  In particular, ``0`` means CUDA GPU
        # zero to Ultralytics but would violate this declared ROS parameter if
        # launch serialized it as an integer.
        self.device = str(self.get_parameter("device").value).strip()
        self.cpu_num_threads = max(1, int(self.get_parameter("cpu_num_threads").value))
        self.cpu_num_interop_threads = max(
            1, int(self.get_parameter("cpu_num_interop_threads").value)
        )
        self.opencv_num_threads = max(1, int(self.get_parameter("opencv_num_threads").value))
        torch.set_num_threads(self.cpu_num_threads)
        try:
            torch.set_num_interop_threads(self.cpu_num_interop_threads)
        except RuntimeError as exc:
            self.get_logger().warn(f"could not set PyTorch interop threads: {exc}")
        cv2.setNumThreads(self.opencv_num_threads)
        # Attest the effective process settings, not merely requested ROS
        # parameters. If a runtime cannot apply a setting, the observed
        # contract must differ and the recorder will fail closed.
        self.actual_cpu_num_threads = int(torch.get_num_threads())
        self.actual_cpu_num_interop_threads = int(torch.get_num_interop_threads())
        self.actual_opencv_num_threads = int(cv2.getNumThreads())

        self.image_size = int(self.get_parameter("image_size").value)
        self.confidence_threshold = float(self.get_parameter("confidence_threshold").value)
        self.inference_chunk = int(self.get_parameter("inference_chunk").value)
        if not 0 <= self.inference_chunk <= len(CAMERA_ORDER):
            raise RuntimeError(
                f"inference_chunk must be between 0 and {len(CAMERA_ORDER)}"
            )
        self.calibration_world = str(self.get_parameter("calibration_world").value or "").strip()
        if not self.calibration_world:
            raise RuntimeError(
                "calibration_world must be set (for example 'warehouse_v2'): the "
                "calibration identity of every published observation depends on it"
            )
        self.predict_conf_floor = float(self.get_parameter("predict_conf_floor").value)
        self.iou_threshold = float(self.get_parameter("iou_threshold").value)
        self.class_name = str(self.get_parameter("class_name").value)
        self.class_id = int(self.get_parameter("class_id").value)
        use_masks_value = self.get_parameter("use_masks").value
        self.use_masks = use_masks_value if isinstance(use_masks_value, bool) else _as_bool(use_masks_value)
        self.mask_min_area = float(self.get_parameter("mask_min_area").value)
        self.mask_bottom_band_px = float(self.get_parameter("mask_bottom_band_px").value)
        self.min_bbox_area_px = float(self.get_parameter("min_bbox_area_px").value)
        self.pixel_noise_sigma = float(self.get_parameter("pixel_noise_sigma").value)
        self.rng = np.random.default_rng(int(self.get_parameter("seed").value))
        self.warmup_iters = max(0, int(self.get_parameter("warmup_iters").value))
        self.r_visible_uv = float(self.get_parameter("camera_observation_r_visible_uv").value)
        self.r_miss_uv = float(self.get_parameter("camera_observation_r_miss_uv").value)
        self.debug_frame_dir = str(self.get_parameter("debug_frame_dir").value).strip()
        if self.debug_frame_dir:
            Path(self.debug_frame_dir).expanduser().mkdir(parents=True, exist_ok=True)
        self.debug_crop_dir = str(self.get_parameter("debug_crop_dir").value).strip()
        if self.debug_crop_dir:
            Path(self.debug_crop_dir).expanduser().mkdir(parents=True, exist_ok=True)

        if self.image_size <= 0:
            raise RuntimeError("image_size must be positive")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise RuntimeError("confidence_threshold must be in [0, 1]")
        if not 0.0 <= self.predict_conf_floor <= self.confidence_threshold:
            raise RuntimeError(
                "predict_conf_floor must be in [0, confidence_threshold] so it can "
                "never suppress a detection that clears the reporting threshold"
            )
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise RuntimeError("iou_threshold must be in [0, 1]")
        if self.pixel_noise_sigma < 0.0 or self.min_bbox_area_px < 0.0:
            raise RuntimeError("pixel_noise_sigma and min_bbox_area_px must be non-negative")

        self.max_batch_stamp_skew_s = float(
            self.get_parameter("max_batch_stamp_skew_s").value
        )
        self.max_pending_wall_s = float(self.get_parameter("max_pending_wall_s").value)
        self.synchronization_mode = str(
            self.get_parameter("synchronization_mode").value
        ).strip().lower()
        self.async_coalesce_wall_s = float(
            self.get_parameter("async_coalesce_wall_s").value
        )
        self.input_transport = str(self.get_parameter("input_transport").value).strip().lower()
        self.runtime_trace_period_s = float(self.get_parameter("runtime_trace_period_s").value)
        if self.synchronization_mode not in {"strict", "asynchronous"}:
            raise RuntimeError("synchronization_mode must be 'strict' or 'asynchronous'")
        if self.input_transport not in {"ros", "direct_gz"}:
            raise RuntimeError("input_transport must be 'ros' or 'direct_gz'")
        if self.input_transport == "direct_gz" and self.synchronization_mode != "asynchronous":
            raise RuntimeError("direct_gz input is diagnostic-only and requires asynchronous mode")
        if self.runtime_backend != "native" and self.synchronization_mode != "asynchronous":
            raise RuntimeError("compiled runtime is diagnostic-only and requires asynchronous mode")
        if self.torchscript_detection_only and (
            self.runtime_backend != "torchscript"
            or self.compiled_model_path is None
            or self.use_masks
            or self.confidence_threshold <= 0.0
        ):
            raise RuntimeError(
                "torchscript_detection_only requires the compiled torchscript backend, "
                "use_masks=false, and a positive confidence_threshold"
            )
        if not math.isfinite(self.runtime_trace_period_s) or self.runtime_trace_period_s < 0.0:
            raise RuntimeError("runtime_trace_period_s must be finite and non-negative")
        if not math.isfinite(self.async_coalesce_wall_s) or self.async_coalesce_wall_s <= 0.0:
            raise RuntimeError("async_coalesce_wall_s must be finite and positive")
        self._producer_epoch = uuid.uuid4().hex
        self._cycle_sequence = 0
        self._clock_high_water_s = None
        self._clock_lock = threading.Lock()
        self._terminal_lifecycle_lock = threading.RLock()
        self._terminal_stopped = False
        self._terminal_stop_request_id = ""
        self._session_stopped_appended = False
        self._outcome_journal = OutcomeJournal(
            journal_path(str(self.get_parameter("outcome_journal_path").value), self._producer_epoch),
            self._producer_epoch,
            max_bytes=int(self.get_parameter("outcome_journal_max_bytes").value),
        )
        self.batch_outcome_publisher = self.create_publisher(
            String, "/perception/camera_batch_outcome", _outcome_qos()
        )
        terminal_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=16,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._terminal_stop_ack_pub = self.create_publisher(
            String, TERMINAL_STOP_ACK_TOPIC, terminal_qos
        )
        self.create_subscription(
            String,
            TERMINAL_STOP_REQUEST_TOPIC,
            self._terminal_stop_request_cb,
            terminal_qos,
        )
        self._publish_batch_outcome(dict(
            status="session_started", camera_ids=list(CAMERA_ORDER),
            outcome_source_sha256=sha256_file(Path(detector_outcomes_module.__file__)),
            journal_source_sha256=sha256_file(Path(camera_outcomes_module.__file__)),
            frame_identity_semantics="producer_epoch_camera_exact_stamp_image_content",
            source_batch_semantics="logical_cycle", invocation_semantics="one_model_call",
        ))
        self.batcher = FourCameraBatcher(
            camera_order=CAMERA_ORDER,
            max_stamp_skew_s=self.max_batch_stamp_skew_s,
            max_pending_wall_s=self.max_pending_wall_s,
            on_event=self._publish_batch_outcome,
        )
        self._async_pending: dict[str, PendingFrame] = {}
        self._async_last_seen_stamp_ns = {camera_id: -1 for camera_id in CAMERA_ORDER}
        self._async_last_seen_content_sha256 = {camera_id: "" for camera_id in CAMERA_ORDER}
        self._async_pending_lock = threading.Lock()
        self._direct_gz_received = 0
        self._async_batches_processed = 0
        self._async_frames_processed = 0
        self._async_last_batch_size = 0
        self._async_input_group = None
        self._async_processing_group = None
        if self.synchronization_mode == "asynchronous":
            # Camera callbacks must continue retaining fresh images while the
            # shared model is busy. The timer itself remains mutually
            # exclusive, so there is still exactly one inference at a time.
            self._async_input_group = ReentrantCallbackGroup()
            self._async_processing_group = MutuallyExclusiveCallbackGroup()
        self._warning_counts: dict[str, int] = {}
        #: strict-path liveness: frames offered per camera and what the batcher
        #: decided. Without these a batcher that never completes a round is silent.
        self._strict_frames: dict[str, int] = {}
        self._strict_last_receive_wall: dict[str, float] = {}
        self._strict_decisions: dict[str, int] = {}
        self._strict_batches = 0

        # The source checkpoint remains mandatory in every mode. The compiled
        # path is an asynchronous diagnostic candidate and is separately
        # attested; strict evidence runs remain native-only.
        model_sha256_before_load = sha256_file(self.model_path)
        self.compiled_model_sha256 = None
        model_load_path = self.model_path
        if self.compiled_model_path is not None:
            self.compiled_model_sha256 = sha256_file(self.compiled_model_path)
            model_load_path = self.compiled_model_path
        # A TorchScript file does not retain Ultralytics' task metadata.  This
        # checkpoint is a segmentation model; loading it without the explicit
        # task silently selects the detect predictor, discards masks and applies
        # the wrong interpretation to the raw output tensor.  Keep the task
        # explicit for the compiled diagnostic path so its normal Results
        # contract remains equivalent to the source checkpoint.
        # The raw detection-only path owns exactly one CUDA copy of the
        # compiled graph below.  Load the source checkpoint only for immutable
        # class metadata here, otherwise asking ``YOLO(...).names`` would load
        # a second compiled graph and exhaust the 4 GiB laptop GPU beside
        # Gazebo's renderer.
        metadata_model_path = (
            self.model_path if self.torchscript_detection_only else model_load_path
        )
        self.model = YOLO(
            str(metadata_model_path),
            task="segment" if self.runtime_backend == "torchscript" and not self.torchscript_detection_only else None,
        )
        self.target_ids = target_class_ids(
            getattr(self.model, "names", {}), self.class_name, self.class_id
        )
        if self.target_ids == set():
            raise RuntimeError(
                "class filter does not match any YOLO class names: "
                f"class_name={self.class_name!r}, class_id={self.class_id}, "
                f"names={getattr(self.model, 'names', {})!r}"
            )
        if self.torchscript_detection_only and (
            self.target_ids is None
            or len(self.target_ids) != 1
            or len(getattr(self.model, "names", {})) != 1
        ):
            raise RuntimeError(
                "torchscript_detection_only is restricted to the one-class commissioned checkpoint"
            )
        self._torchscript_detection_model = None
        self._torchscript_detection_device = None
        if self.torchscript_detection_only:
            # The source checkpoint is segmentation-trained, but this explicit
            # runtime publishes bbox-bottom observations only. Avoid building
            # unused instance masks by applying class-aware NMS directly to
            # the compiled graph's raw detection tensor.
            if not self.device or not torch.cuda.is_available():
                raise RuntimeError(
                    "torchscript_detection_only requires an explicit CUDA device"
                )
            device_name = self.device if self.device.startswith("cuda") else f"cuda:{self.device}"
            self._torchscript_detection_device = torch.device(device_name)
            self._torchscript_detection_model = torch.jit.load(
                str(self.compiled_model_path), map_location=self._torchscript_detection_device
            ).eval()

        try:
            from reliability.single_camera_adapter import (
                SingleCameraAdapterConfig,
                camera_observation_from_diagnostics,
            )
        except Exception as exc:
            raise RuntimeError(
                "the CameraObservation adapter is required by the batched runtime"
            ) from exc
        self._camera_observation_from_diagnostics = camera_observation_from_diagnostics
        self._observation_configs = {
            camera_id: SingleCameraAdapterConfig(
                camera_id=camera_id,
                calibration_id=f"{self.calibration_world}_{camera_id}",
                image_frame_id=camera_id,
                r_visible_uv=self.r_visible_uv,
                r_miss_uv=self.r_miss_uv,
            )
            for camera_id in CAMERA_ORDER
        }

        if self.warmup_iters:
            dummy_batch = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in CAMERA_ORDER]
            warmup_start = time.perf_counter()
            for _ in range(self.warmup_iters):
                try:
                    if self.torchscript_detection_only:
                        self._predict_detection_only(dummy_batch)
                    else:
                        validate_batch_results(self._predict_batch(dummy_batch), len(CAMERA_ORDER))
                except Exception as exc:
                    raise RuntimeError(
                        f"{len(CAMERA_ORDER)}-image detector warmup failed: {exc}"
                    ) from exc
            self.get_logger().info(
                f"four-image warmup: {self.warmup_iters} iterations in "
                f"{(time.perf_counter() - warmup_start) * 1.0e3:.0f} ms"
            )

        # Hash after the model is loaded and warmed, then reject a checkpoint
        # replacement race rather than attesting bytes that were not the model
        # used for the first live batch.
        self.model_sha256 = sha256_file(self.model_path)
        if self.model_sha256 != model_sha256_before_load:
            raise RuntimeError(
                "model checkpoint bytes changed while the batched detector was loading"
            )
        if self.compiled_model_path is not None:
            compiled_hash_after_load = sha256_file(self.compiled_model_path)
            if compiled_hash_after_load != self.compiled_model_sha256:
                raise RuntimeError(
                    "compiled model bytes changed while the batched detector was loading"
                )
        if self.synchronization_mode == "strict":
            self._publish_runtime_contract()
        else:
            self.get_logger().warn(
                "asynchronous shared-model mode is diagnostic-only: it does not "
                "publish the strict multicamera runtime contract and is therefore "
                "ineligible for evidence recording"
            )

        self.outputs: dict[str, _CameraOutput] = {}
        for camera_id in CAMERA_ORDER:
            self.outputs[camera_id] = _CameraOutput(
                pixel_publisher=self.create_publisher(
                    PoseStamped, f"/perception/{camera_id}/pixel_pose", 10
                ),
                diagnostics_publisher=self.create_publisher(
                    Float64MultiArray,
                    f"/perception/{camera_id}/detection_diagnostics",
                    10,
                ),
                observation_publisher=self.create_publisher(
                    String, f"/perception/camera_observation/{camera_id}", 10
                ),
            )

        if self.synchronization_mode == "asynchronous":
            self._async_wall_clock = RclpyClock(clock_type=ClockType.SYSTEM_TIME)
            self.create_timer(
                self.async_coalesce_wall_s,
                self._drain_async_pending,
                clock=self._async_wall_clock,
                callback_group=self._async_processing_group,
            )

        # ``Node.subscriptions`` is an rclpy-managed read-only property.
        # Keep our references under a node-local name so the subscriptions
        # remain alive without shadowing the ROS API at startup.
        self.camera_subscriptions = []
        self._gz_transport_node = None
        if self.input_transport == "direct_gz":
            # Subscribe before ROS conversion so B--D can bypass three large
            # gz->ROS image copies. Outputs remain the normal ROS contracts.
            from gz.msgs10.image_pb2 import Image as GzImage
            from gz.transport13 import Node as GzTransportNode, SubscribeOptions

            self._gz_image_type = GzImage
            self._gz_transport_node = GzTransportNode()
            for camera_id in CAMERA_ORDER:
                self._gz_transport_node.subscribe_raw(
                    CAMERA_TOPICS[camera_id],
                    lambda raw, _info, selected_camera=camera_id: self._gz_raw_image_callback(
                        selected_camera, raw
                    ),
                    # Gazebo Fortress still advertises ignition.msgs.Image
                    # even though its wire payload is protobuf-compatible with
                    # gz.msgs.Image. Raw subscription avoids the type-name
                    # handshake mismatch without a ROS conversion.
                    "ignition.msgs.Image",
                    SubscribeOptions(),
                )
            self.get_logger().info(
                "subscribed directly to Gazebo image topics for camera_A through camera_D"
            )
        else:
            for camera_id in CAMERA_ORDER:
                self.camera_subscriptions.append(
                    self.create_subscription(
                        Image,
                        CAMERA_TOPICS[camera_id],
                        lambda msg, selected_camera=camera_id: self._image_callback(
                            selected_camera, msg
                        ),
                        1,
                        callback_group=self._async_input_group,
                    )
                )

        self.get_logger().info(
            "Batched multicamera YOLO started "
            f"(model={model_load_path}, backend={self.runtime_backend}, "
            f"device={self.device or 'auto'}, "
            f"batch_order={','.join(CAMERA_ORDER)}, batch_size={len(CAMERA_ORDER)}, "
            f"imgsz={self.image_size}, masks={self.use_masks}, "
            f"detection_only={self.torchscript_detection_only}, "
            f"torch_threads={self.actual_cpu_num_threads}, "
            f"torch_interop={self.actual_cpu_num_interop_threads}, "
            f"opencv_threads={self.actual_opencv_num_threads}, "
            f"input_policy={self.synchronization_mode}-{self.input_transport}-new-latest-only)"
        )
        self._liveness_clock = RclpyClock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(0.25, self._batch_liveness_tick, clock=self._liveness_clock)
        if self.runtime_trace_period_s > 0.0:
            trace_clock = RclpyClock(clock_type=ClockType.SYSTEM_TIME)
            self.create_timer(
                self.runtime_trace_period_s,
                self._log_runtime_trace,
                clock=trace_clock,
            )

    def _publish_runtime_contract(self) -> None:
        """Publish the retained identity only after a usable model is ready.

        The recorder subscribes to this topic before accepting detector or
        odometry rows.  It compares the JSON byte-for-byte (via the embedded
        digest) with its frozen CLI/configuration expectation, so this is not a
        best-effort diagnostic announcement.
        """

        try:
            source_hashes = source_hashes_from_paths(
                {
                    EXECUTABLE_SOURCE_KEY: Path(__file__),
                    FOUR_CAMERA_BATCH_SOURCE_KEY: Path(four_camera_batch_module.__file__),
                    RUNTIME_CONTRACT_SOURCE_KEY: Path(runtime_contract_module.__file__),
                    YOLO_SELECTION_SOURCE_KEY: Path(yolo_selection_module.__file__),
                }
            )
            self.runtime_contract = build_batched_runtime_contract(
                model_path=self.model_path,
                model_sha256=self.model_sha256,
                imgsz=self.image_size,
                confidence_threshold=self.confidence_threshold,
                iou_threshold=self.iou_threshold,
                use_masks=self.use_masks,
                shared_device=self.device,
                cpu_threads=self.actual_cpu_num_threads,
                interop_threads=self.actual_cpu_num_interop_threads,
                opencv_threads=self.actual_opencv_num_threads,
                max_batch_stamp_skew_s=self.max_batch_stamp_skew_s,
                max_pending_wall_s=self.max_pending_wall_s,
                source_hashes=source_hashes,
            )
            self.runtime_contract_publisher = self.create_publisher(
                String,
                RUNTIME_CONTRACT_TOPIC,
                _runtime_contract_qos(),
            )
            message = String()
            message.data = runtime_contract_json(self.runtime_contract)
            self.runtime_contract_publisher.publish(message)
        except Exception as exc:
            raise RuntimeError(
                "could not publish the required multicamera detector runtime contract"
            ) from exc
        self.get_logger().info(
            "published retained multicamera detector runtime contract "
            f"sha256={self.runtime_contract['contract_sha256']} "
            f"topic={RUNTIME_CONTRACT_TOPIC}"
        )

    def _clock_s(self) -> float:
        with self._clock_lock:
            stamp = float(self.get_clock().now().nanoseconds) * 1.0e-9
            previous = self._clock_high_water_s
            if previous is not None and stamp < previous - MAX_FUTURE_IMAGE_STAMP_S:
                self._fatal("simulation clock moved backwards; coordinated runtime restart required")
            self._clock_high_water_s = stamp if previous is None else max(stamp, previous)
            return stamp

    def _publish_batch_outcome(self, event) -> None:
        if getattr(self, "_terminal_stopped", False):
            return
        # This local durable record precedes publication. Reliable DDS history
        # supports bounded late delivery; the journal covers process/transport loss.
        payload = self._outcome_journal.append(dict(
            event, stage="detector",
            publish_stamp_s=float(self.get_clock().now().nanoseconds) * 1e-9,
        ))
        message = String()
        message.data = json.dumps(payload, sort_keys=True, allow_nan=False)
        # The append-only outcome journal and ROS topic are the evidence surfaces.
        # Duplicating every full JSON event at INFO creates multi-gigabyte campaign
        # consoles and materially slows five-camera 960 px inference.
        self.get_logger().debug("camera_batch_outcome " + message.data)
        if getattr(self, "_outcome_transport_enabled", True):
            self.batch_outcome_publisher.publish(message)


    def _batch_liveness_tick(self) -> None:
        if getattr(self, "_terminal_stopped", False):
            return
        self._clock_s()  # detect reset even if input publication stops
        if self.synchronization_mode != "strict":
            return
        self.batcher.expire(time.perf_counter())
        now_wall = time.perf_counter()
        missing = [c for c in CAMERA_ORDER
                   if c not in self._strict_last_receive_wall
                   or now_wall - self._strict_last_receive_wall[c] > self.max_pending_wall_s]
        if missing:
            self._warn_bounded("missing_or_slow_cameras",
                               f"no recent image received from cameras: {missing}")

    def _predict_batch(self, images_bgr: list[np.ndarray]):
        """One inference cycle over the whole timestamp batch, in camera order.

        The cycle may be split into consecutive chunks, because activation memory --
        not compute -- is what limits this machine. Measured on the 4 GiB P2000 with
        Gazebo rendering five 1280x720 oblique cameras (which holds about 1.9 GiB of
        the card), reserved GPU memory for one cycle of five images is:

            imgsz 640: 708 MiB whole    | imgsz 960: 2558 MiB whole -> OOM in a live run
            imgsz 960 in chunks of two: 1418 MiB, which fits
            imgsz 1280: does not fit at any chunk size beside the simulator

        Chunking changes nothing observable: the batch is still one strict
        same-timestamp set, the results are still assembled in CAMERA_ORDER, and the
        reported inference time is still the wall time of the complete cycle.
        """
        if not 1 <= len(images_bgr) <= len(CAMERA_ORDER):
            raise BatchContractError(
                f"predict requires between 1 and {len(CAMERA_ORDER)} images"
            )
        chunk = self.inference_chunk if self.inference_chunk > 0 else len(images_bgr)
        results: list[Any] = []
        for start in range(0, len(images_bgr), chunk):
            group = list(images_bgr[start:start + chunk])
            kwargs = {
                "source": group,
                "imgsz": self.image_size,
                "conf": self.predict_conf_floor,
                "iou": self.iou_threshold,
                "batch": len(group),
                "stream": False,
                "verbose": False,
            }
            if self.device:
                kwargs["device"] = self.device
            cycle = getattr(self, "_active_cycle", None)
            scheduled_stamp = self._clock_s() if cycle is not None else None
            invocation_id = f"{cycle[0]}/chunk/{start // chunk}" if cycle is not None else None
            invocation_members = ([self._active_members[c] for c in cycle[1][start:start + len(group)]]
                                  if cycle is not None else [])
            if cycle is not None:
                self._publish_batch_outcome(dict(source_batch_id=cycle[0], invocation_id=invocation_id,
                                                 members=invocation_members, status="inference_started",
                                                 inference_scheduled_stamp_s=scheduled_stamp))
            started_stamp = self._clock_s() if cycle is not None else None
            started_wall = time.perf_counter()
            try:
                part = self.model.predict(**kwargs)
                finished_wall = time.perf_counter()
                finished_stamp = self._clock_s() if cycle is not None else None
                if part is None:
                    raise BatchContractError("batch inference returned no result sequence")
                results.extend(validate_batch_results(part, len(group)))
            except BaseException as exc:
                if cycle is not None:
                    self._publish_batch_outcome(dict(
                        source_batch_id=cycle[0], invocation_id=invocation_id, members=invocation_members,
                        camera_ids=cycle[1][start:start + len(group)], status="inference_error",
                        inference_start_stamp_s=started_stamp, reason=str(exc) or type(exc).__name__))
                raise
            if cycle is not None:
                self._publish_batch_outcome(dict(
                    source_batch_id=cycle[0], invocation_id=invocation_id, members=invocation_members,
                    camera_ids=cycle[1][start:start + len(group)], status="inference_completed",
                    inference_start_stamp_s=started_stamp, inference_finish_stamp_s=finished_stamp,
                    inference_wall_ms=(finished_wall - started_wall) * 1000.0,
                    timing_semantics="host_model_call_start_return"))
        if len(results) != len(images_bgr):
            raise BatchContractError(
                f"chunked inference returned {len(results)} results for "
                f"{len(images_bgr)} images"
            )
        return results

    def _predict_detection_only(self, images_bgr: list[np.ndarray]) -> list[dict[str, Any]]:
        """Return compiled bbox-bottom selections without constructing masks."""

        if self._torchscript_detection_model is None or self._torchscript_detection_device is None:
            raise BatchContractError("detection-only runtime was not initialized")
        prepared: list[np.ndarray] = []
        transforms: list[tuple[float, int, int, int, int]] = []
        for image in images_bgr:
            height, width = image.shape[:2]
            gain = min(float(self.image_size) / width, float(self.image_size) / height)
            resized_width, resized_height = round(width * gain), round(height * gain)
            pad_x = round((self.image_size - resized_width) / 2.0 - 0.1)
            pad_y = round((self.image_size - resized_height) / 2.0 - 0.1)
            canvas = np.full((self.image_size, self.image_size, 3), 114, dtype=np.uint8)
            canvas[
                pad_y:pad_y + resized_height,
                pad_x:pad_x + resized_width,
            ] = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
            prepared.append(np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)))
            transforms.append((gain, pad_x, pad_y, width, height))
        tensor = torch.from_numpy(np.stack(prepared)).to(
            device=self._torchscript_detection_device, dtype=torch.float32
        ).div_(255.0)
        cycle = getattr(self, "_active_cycle", None)
        scheduled_stamp = self._clock_s() if cycle is not None else None
        if cycle is not None:
            self._publish_batch_outcome(dict(source_batch_id=cycle[0], invocation_id=f"{cycle[0]}/chunk/0",
                                             members=list(self._active_members.values()), status="inference_started",
                                             inference_scheduled_stamp_s=scheduled_stamp))
        started_stamp = self._clock_s() if cycle is not None else None
        started_wall = time.perf_counter()
        try:
            with torch.inference_mode():
                raw = self._torchscript_detection_model(tensor)
                finished_wall = time.perf_counter()
                finished_stamp = self._clock_s() if cycle is not None else None
                predictions = raw[0] if isinstance(raw, (tuple, list)) else raw
        except BaseException as exc:
            if cycle is not None:
                self._publish_batch_outcome(dict(source_batch_id=cycle[0], invocation_id=f"{cycle[0]}/chunk/0",
                                                 members=list(self._active_members.values()), status="inference_error",
                                                 inference_start_stamp_s=started_stamp, reason=str(exc) or type(exc).__name__))
            raise
        if cycle is not None:
            self._publish_batch_outcome(dict(source_batch_id=cycle[0], invocation_id=f"{cycle[0]}/chunk/0",
                                             members=list(self._active_members.values()), status="inference_completed",
                                             inference_start_stamp_s=started_stamp, inference_finish_stamp_s=finished_stamp,
                                             inference_wall_ms=(finished_wall - started_wall) * 1000.0,
                                             timing_semantics="host_model_call_start_return"))
        if predictions.ndim != 3 or predictions.shape[0] != len(images_bgr) or predictions.shape[1] < 5:
            raise BatchContractError(
                f"unexpected one-class raw prediction shape {tuple(predictions.shape)}"
            )
        selections: list[dict[str, Any]] = []
        for prediction, (gain, pad_x, pad_y, width, height) in zip(predictions, transforms, strict=True):
            # In the one-class export the score channel is index four. NMS
            # always retains its global maximum first, so directly selecting
            # that element is exactly the bbox-only observable while avoiding
            # multi-second suppression over irrelevant low-score anchors.
            score, index = torch.max(prediction[4], dim=0)
            score = float(score.item())
            if score < self.confidence_threshold:
                selections.append({
                    "detected": False, "detected_after_threshold": False,
                    "raw_best_score": 0.0, "selected_score": 0.0, "confidence": 0.0,
                    "best_class_id": math.nan, "class_id": math.nan, "bbox_xyxy": None,
                    "mask_area": math.nan, "mask_bottom_u": math.nan, "mask_bottom_v": math.nan,
                    "selected_u": math.nan, "selected_v": math.nan,
                    "selected_pixel_source": "none", "mask_available": 0,
                    "n_candidates": 0, "detection": None,
                })
                continue
            best_xywh = prediction[:4, index].detach().cpu().numpy().astype(float)
            center_x, center_y, box_width, box_height = best_xywh
            bbox = np.asarray([
                center_x - box_width / 2.0,
                center_y - box_height / 2.0,
                center_x + box_width / 2.0,
                center_y + box_height / 2.0,
            ])
            bbox[[0, 2]] = (bbox[[0, 2]] - float(pad_x)) / gain
            bbox[[1, 3]] = (bbox[[1, 3]] - float(pad_y)) / gain
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0.0, float(width - 1))
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0.0, float(height - 1))
            selections.append({
                "detected": True, "detected_after_threshold": True,
                "raw_best_score": score, "selected_score": score, "confidence": score,
                "best_class_id": 0, "class_id": 0, "bbox_xyxy": bbox,
                "mask_area": math.nan, "mask_bottom_u": math.nan, "mask_bottom_v": math.nan,
                "bbox_bottom_u": float(0.5 * (bbox[0] + bbox[2])),
                "bbox_bottom_v": float(bbox[3]),
                "selected_u": float(0.5 * (bbox[0] + bbox[2])), "selected_v": float(bbox[3]),
                "selected_pixel_source": "bbox_bottom", "mask_available": 0,
                "n_candidates": 1, "detection": None,
            })
        return selections

    def _warn_bounded(self, key: str, message: str) -> None:
        count = self._warning_counts.get(key, 0) + 1
        self._warning_counts[key] = count
        if count <= 3 or count & (count - 1) == 0:
            self.get_logger().warn(f"{message} (count={count})")

    def _fatal(self, message: str, cause: Exception | None = None) -> NoReturn:
        """Log once and terminate rather than forge an availability miss."""

        self.get_logger().fatal(message)
        if getattr(self, "_outcome_journal", None) is not None:
            cycle = getattr(self, "_active_cycle", None)
            try:
                self._publish_batch_outcome(dict(status="producer_error", reason=message,
                                                 source_batch_id=cycle[0] if cycle is not None else ""))
            except Exception as journal_error:
                # The primary fault must remain visible even if durability/transport
                # is itself the failed subsystem. Do not continue processing.
                self.get_logger().error(f"could not transport producer_error: {journal_error}")
        if cause is None:
            raise RuntimeError(message)
        raise RuntimeError(message) from cause

    def _image_callback(self, camera_id: str, msg: Image) -> None:
        if getattr(self, "_terminal_stopped", False):
            return
        receive_wall_s = time.perf_counter()
        receive_stamp_s = self._clock_s()
        try:
            stamp_ns = stamp_parts_to_ns(msg.header.stamp.sec, msg.header.stamp.nanosec)
            content_hash = image_content_sha256(encoding=msg.encoding, height=msg.height,
                                                width=msg.width, step=msg.step, data=msg.data)
            frame = PendingFrame(
                camera_id=camera_id,
                stamp_ns=stamp_ns,
                receive_stamp_s=receive_stamp_s,
                receive_wall_s=receive_wall_s,
                payload=msg,
                content_sha256=content_hash,
                source_frame_id=source_frame_id(self._producer_epoch, camera_id, stamp_ns, content_hash),
            )
            self._publish_batch_outcome(dict(status="image_received", members=[frame_member(frame)]))
            if self.synchronization_mode == "asynchronous":
                self._offer_async_frame(frame)
                return
            decision = self.batcher.offer(frame)
        except Exception as exc:
            self._fatal(f"malformed input contract for {camera_id}: {exc}", exc)

        self._strict_last_receive_wall[camera_id] = receive_wall_s
        self._strict_frames[camera_id] = self._strict_frames.get(camera_id, 0) + 1
        self._strict_decisions[decision.status] = (
            self._strict_decisions.get(decision.status, 0) + 1
        )

        if decision.status in {"duplicate", "out_of_order", "stamp_skew"}:
            self._publish_batch_outcome(dict(status="image_" + decision.status,
                                             members=[frame_member(frame)]))
            self._warn_bounded(
                camera_id + ":" + decision.status,
                f"{decision.status} frame rejected for {camera_id}; "
                f"dropped={decision.dropped_camera_ids}",
            )
        elif decision.dropped_camera_ids:
            self._warn_bounded(
                "pending_expired",
                f"expired pending frames: {decision.dropped_camera_ids}",
            )
        if decision.batch is not None:
            self._strict_batches += 1
            self._process_batch(decision.batch)

    def _gz_image_callback(self, camera_id: str, msg: Any) -> None:
        """Accept one Gazebo RGB image without a ros_gz_bridge conversion."""

        if getattr(self, "_terminal_stopped", False):
            return

        receive_wall_s = time.perf_counter()
        receive_stamp_s = self._clock_s()
        try:
            sec = int(msg.header.stamp.sec)
            nanosec = int(msg.header.stamp.nsec)
            stamp_ns = stamp_parts_to_ns(sec, nanosec)
            width = int(msg.width)
            height = int(msg.height)
            step = int(msg.step)
            if width <= 0 or height <= 0 or step != width * 3:
                raise BatchContractError(
                    f"unexpected Gazebo image geometry {width}x{height}, step={step}"
                )
            data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            if data.size != height * step:
                raise BatchContractError(
                    f"Gazebo image data length {data.size} does not match {height}x{step}"
                )
            image = data.reshape(height, width, 3)
            # Gazebo camera output is RGB_INT8 in the commissioned worlds.
            # Convert once here to preserve the detector's existing BGR input
            # contract and reject every other format rather than guessing.
            if int(msg.pixel_format_type) != 3:  # gz.msgs.RGB_INT8
                raise BatchContractError(
                    f"unexpected Gazebo pixel format {int(msg.pixel_format_type)}; expected RGB_INT8"
                )
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            header = Header()
            header.stamp.sec = sec
            header.stamp.nanosec = nanosec
            header.frame_id = "image"
            content_hash = image_content_sha256(encoding="rgb8", height=height, width=width,
                                                step=step, data=msg.data)
            frame = PendingFrame(
                camera_id=camera_id, stamp_ns=stamp_ns,
                receive_stamp_s=receive_stamp_s, receive_wall_s=receive_wall_s,
                payload=_DirectGzImagePayload(header=header, image_bgr=bgr),
                content_sha256=content_hash,
                source_frame_id=source_frame_id(self._producer_epoch, camera_id, stamp_ns, content_hash),
            )
            self._publish_batch_outcome(dict(status="image_received", members=[frame_member(frame)]))
            self._offer_async_frame(frame)
            self._direct_gz_received += 1
        except Exception as exc:
            self._fatal(f"malformed direct Gazebo image for {camera_id}: {exc}", exc)

    def _gz_raw_image_callback(self, camera_id: str, raw: bytes) -> None:
        """Decode the legacy-advertised Gazebo image wire payload locally."""

        try:
            msg = self._gz_image_type.FromString(raw)
        except Exception as exc:
            self._fatal(f"could not decode direct Gazebo image for {camera_id}: {exc}", exc)
        self._gz_image_callback(camera_id, msg)

    def _offer_async_frame(self, frame: PendingFrame) -> None:
        """Retain one fresh frame per camera without cross-camera stamp matching.

        A short wall-clock coalescing timer sends the currently pending subset
        through the one shared model. Every result retains its source stamp; a
        downstream fusion policy, not this detector, decides whether readings
        from different cameras are close enough to combine.
        """

        events = []
        conflict = False
        with self._async_pending_lock:
            expired = [pending for pending in self._async_pending.values()
                       if frame.receive_wall_s - pending.receive_wall_s > self.max_pending_wall_s]
            for pending in expired:
                self._async_pending.pop(pending.camera_id)
            if expired:
                events.append(dict(status="incomplete_timeout", members=[frame_member(f) for f in expired]))
            last_seen = self._async_last_seen_stamp_ns[frame.camera_id]
            if frame.stamp_ns > last_seen:
                previous = self._async_pending.get(frame.camera_id)
                if previous is not None:
                    events.append(dict(status="frame_replaced", members=[frame_member(previous)]))
                self._async_last_seen_stamp_ns[frame.camera_id] = frame.stamp_ns
                self._async_last_seen_content_sha256[frame.camera_id] = frame.content_sha256
                self._async_pending[frame.camera_id] = frame
            else:
                conflict = (frame.stamp_ns == last_seen and bool(frame.content_sha256)
                            and bool(self._async_last_seen_content_sha256[frame.camera_id])
                            and frame.content_sha256 != self._async_last_seen_content_sha256[frame.camera_id])
                status = ("conflicting_duplicate_image" if conflict else
                          "image_duplicate" if frame.stamp_ns == last_seen else "image_out_of_order")
                events.append(dict(status=status, members=[frame_member(frame)]))
        for event in events:
            self._publish_batch_outcome(event)
        if conflict:
            self._fatal(f"conflicting image bytes at same capture stamp for {frame.camera_id}")

    def _drain_async_pending(self) -> None:
        if getattr(self, "_terminal_stopped", False):
            return
        now_wall_s = time.perf_counter()
        with self._async_pending_lock:
            if not self._async_pending:
                return
            expired = [pending for pending in self._async_pending.values()
                       if now_wall_s - pending.receive_wall_s > self.max_pending_wall_s]
            for pending in expired:
                self._async_pending.pop(pending.camera_id)
            batch = tuple(self._async_pending[camera_id] for camera_id in CAMERA_ORDER
                          if camera_id in self._async_pending)
            wait_for_clock = bool(batch) and (
                self._clock_s() + MAX_FUTURE_IMAGE_STAMP_S < max(f.stamp_ns for f in batch) * 1e-9
            )
            if not wait_for_clock:
                for item in batch:
                    self._async_pending.pop(item.camera_id)
        if expired:
            self._publish_batch_outcome(dict(status="incomplete_timeout",
                                             members=[frame_member(f) for f in expired]))
        if not batch:
            return
        if wait_for_clock:
            self._warn_bounded("async_clock_wait", "waiting for simulation clock before publishing asynchronous frames")
            return
        self._async_batches_processed += 1
        self._async_frames_processed += len(batch)
        self._async_last_batch_size = len(batch)
        self._process_frames(batch)

    def _log_runtime_trace(self) -> None:
        """Emit low-rate local counters without subscribing to image topics."""

        if self.synchronization_mode == "strict":
            waiting = "; ".join(
                f"{stamp_ns}ns present={list(cameras)} missing={[c for c in CAMERA_ORDER if c not in cameras]}"
                for stamp_ns, cameras in self.batcher.bucket_report
            )
            self.get_logger().info(
                "runtime_trace "
                f"transport={self.input_transport} mode=strict "
                f"frames={self._strict_frames} "
                f"decisions={self._strict_decisions} "
                f"batches={self._strict_batches} "
                f"open_rounds=[{waiting}]"
            )
            return
        self.get_logger().info(
            "runtime_trace "
            f"transport={self.input_transport} direct_gz_received={self._direct_gz_received} "
            f"async_batches={self._async_batches_processed} "
            f"async_frames={self._async_frames_processed} "
            f"last_batch_size={self._async_last_batch_size}"
        )

    def _process_batch(self, batch: tuple[PendingFrame, ...]) -> None:
        if tuple(item.camera_id for item in batch) != CAMERA_ORDER:
            self._fatal("internal multicamera batch order violation")
        self._process_frames(batch)

    def _process_frames(self, batch: tuple[PendingFrame, ...]) -> None:
        if getattr(self, "_terminal_stopped", False):
            return
        if not batch or len({item.camera_id for item in batch}) != len(batch):
            self._fatal("internal camera micro-batch identity violation")
        self._cycle_sequence += 1
        source_batch_id = f"strict:{self._producer_epoch}:{self._cycle_sequence}:" + ",".join(
            f"{item.camera_id}@{item.stamp_ns}" for item in batch
        )
        members = [item.camera_id for item in batch]
        chunk = len(batch) if self.torchscript_detection_only or self.inference_chunk == 0 else self.inference_chunk
        self._active_members = {
            item.camera_id: dict(frame_member(item), producer_epoch=self._producer_epoch,
                                 detector_invocation_id=f"{source_batch_id}/chunk/{index // chunk}")
            for index, item in enumerate(batch)
        }
        self._active_cycle = (source_batch_id, members)
        try:
            self._publish_batch_outcome(dict(source_batch_id=source_batch_id,
                                             members=list(self._active_members.values()),
                                             status="selected", camera_ids=members,
                                             inference_chunk=self.inference_chunk))
            status = self._process_frames_once(batch, source_batch_id=source_batch_id)
        except BaseException as exc:
            self._publish_batch_outcome(dict(source_batch_id=source_batch_id,
                                             status="aborted", reason=str(exc) or type(exc).__name__, camera_ids=members,
                                             members=list(self._active_members.values())))
            raise
        finally:
            self._active_cycle = None
        self._publish_batch_outcome(dict(source_batch_id=source_batch_id,
                                         status=status, camera_ids=members,
                                         members=list(self._active_members.values())))

    def _process_frames_once(self, batch, *, source_batch_id):
        # Startup clock race: before the first /clock tick under use_sim_time,
        # ``now()`` reads 0 while camera images already carry sim stamps (e.g.
        # 4.8 s), so publishing would trip the future-stamp integrity fault. The
        # async drain guards this explicitly; mirror it here for the strict path
        # so a batch that forms before the clock catches up is dropped (bounded
        # warning) rather than crashing the detector. Only fires during the brief
        # startup window where the clock is genuinely behind the image.
        latest_source_stamp_s = max(item.stamp_ns for item in batch) * 1.0e-9
        if self._clock_s() + MAX_FUTURE_IMAGE_STAMP_S < latest_source_stamp_s:
            self._warn_bounded(
                "strict_clock_wait",
                "waiting for simulation clock before publishing multicamera batch",
            )
            return "dropped_clock_wait"

        images: list[np.ndarray] = []
        try:
            for item in batch:
                image = (
                    item.payload.image_bgr
                    if isinstance(item.payload, _DirectGzImagePayload)
                    else image_msg_to_bgr8(item.payload)
                )
                if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
                    raise BatchContractError(
                        f"{item.camera_id} image conversion returned shape {image.shape!r}"
                    )
                images.append(image)
        except Exception as exc:
            self._fatal(f"multicamera image conversion failed: {exc}", exc)

        predict_start_stamp_s = self._clock_s()
        predict_start_wall_s = time.perf_counter()
        try:
            selections = (
                self._predict_detection_only(images)
                if self.torchscript_detection_only
                else None
            )
            raw_results = None if selections is not None else self._predict_batch(images)
        except Exception as exc:
            elapsed_ms = max((time.perf_counter() - predict_start_wall_s) * 1.0e3, 0.0)
            self._fatal(
                f"multicamera batch inference failed after {elapsed_ms:.1f} ms: {exc}",
                exc,
            )

        predict_finish_stamp_s = self._clock_s()
        timing = _BatchTiming(
            predict_start_stamp_s=predict_start_stamp_s,
            predict_finish_stamp_s=predict_finish_stamp_s,
            inference_wall_ms=max(
                (time.perf_counter() - predict_start_wall_s) * 1.0e3, 0.0
            ),
        )
        prepared: list[dict[str, Any]] = []
        try:
            results = None if selections is not None else validate_batch_results(raw_results, len(batch))
        except BatchContractError as exc:
            self._fatal(f"malformed multicamera results: {exc}", exc)
        for index, (item, image) in enumerate(zip(batch, images, strict=True)):
            try:
                prepared.append(
                    self._prepare_selection(selections[index])
                    if selections is not None
                    else self._prepare_result(results[index])
                )
            except Exception as exc:
                self._fatal(
                    f"malformed result contract for {item.camera_id}: {exc}",
                    exc,
                )
        # Validate and prepare all selections before publishing any camera from
        # this batch.  A malformed B/C/D result therefore cannot leave a
        # plausible partial A-only evidence row behind.
        for item, image, selection in zip(batch, images, prepared, strict=True):
            member = self._active_members[item.camera_id]
            self._publish_batch_outcome(dict(source_batch_id=source_batch_id, members=[member],
                                             status="member_publication_started",
                                             detection_valid=bool(selection.get("detected_after_threshold", False))))
            try:
                self._publish_result(item, image, selection, timing, source_batch_id=source_batch_id)
            except Exception as exc:
                self._publish_batch_outcome(dict(source_batch_id=source_batch_id, members=[member],
                                                 status="member_publication_error", reason=str(exc) or type(exc).__name__,
                                                 detection_valid=bool(selection.get("detected_after_threshold", False))))
                raise
            self._publish_batch_outcome(dict(source_batch_id=source_batch_id, members=[member],
                                             status="member_published",
                                             detection_valid=bool(selection.get("detected_after_threshold", False))))
        return "published"

    def _prepare_result(self, result: Any) -> dict[str, Any]:
        selection = select_best_detection(
            result,
            target_ids=self.target_ids,
            confidence_threshold=self.confidence_threshold,
            use_masks=self.use_masks,
            mask_min_area=self.mask_min_area,
            mask_bottom_band_px=self.mask_bottom_band_px,
        )
        return self._prepare_selection(selection)

    def _prepare_selection(self, selection: dict[str, Any]) -> dict[str, Any]:
        selection = dict(selection)
        bbox = selection.get("bbox_xyxy")
        if bbox is not None and self.min_bbox_area_px > 0.0:
            x0, y0, x1, y1 = np.asarray(bbox, dtype=float).reshape(4)
            bbox_area = float(max(x1 - x0, 0.0) * max(y1 - y0, 0.0))
            if bbox_area < self.min_bbox_area_px:
                selection["detected_after_threshold"] = False

        if bbox is not None:
            selected_u = float(selection.get("selected_u", math.nan))
            selected_v = float(selection.get("selected_v", math.nan))
            if self.pixel_noise_sigma > 0.0:
                selected_u += float(self.rng.normal(0.0, self.pixel_noise_sigma))
                selected_v += float(self.rng.normal(0.0, self.pixel_noise_sigma))
            selection["selected_u"] = selected_u
            selection["selected_v"] = selected_v
            if not math.isfinite(selected_u) or not math.isfinite(selected_v):
                raise BatchContractError("selected pixel is non-finite")

        return selection

    def _publish_result(
        self,
        item: PendingFrame,
        image_bgr: np.ndarray,
        selection: dict[str, Any],
        timing: _BatchTiming,
        *,
        source_batch_id: str,
    ) -> None:
        if bool(selection.get("detected_after_threshold", False)):
            selected_u = float(selection.get("selected_u", math.nan))
            selected_v = float(selection.get("selected_v", math.nan))
            if not math.isfinite(selected_u) or not math.isfinite(selected_v):
                self._fatal(f"non-finite selected pixel for {item.camera_id}")
        self._write_debug_frame(item, image_bgr, selection)
        self._publish_camera(
            item,
            image_bgr,
            selection,
            timing,
            source_batch_id=source_batch_id,
        )
        if not bool(selection.get("detected_after_threshold", False)):
            return

        selected_u = float(selection.get("selected_u", math.nan))
        selected_v = float(selection.get("selected_v", math.nan))
        pose = PoseStamped()
        pose.header = item.payload.header
        pose.header.frame_id = "image"
        pose.pose.position.x = selected_u
        pose.pose.position.y = selected_v
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0
        try:
            self.outputs[item.camera_id].pixel_publisher.publish(pose)
        except Exception as exc:
            self._fatal(f"pixel-pose publish failed for {item.camera_id}: {exc}", exc)

    def _publish_camera(
        self,
        item: PendingFrame,
        image_bgr: np.ndarray,
        selection: dict[str, Any] | None,
        timing: _BatchTiming,
        *,
        source_batch_id: str,
    ) -> None:
        selection = dict(selection or {})
        stamp_s = item.stamp_ns * 1.0e-9
        bbox = selection.get("bbox_xyxy")
        if bbox is None:
            x0 = y0 = x1 = y1 = math.nan
        else:
            x0, y0, x1, y1 = [
                float(value) for value in np.asarray(bbox, dtype=float).reshape(4)
            ]
        selected_u = float(selection.get("selected_u", math.nan))
        selected_v = float(selection.get("selected_v", math.nan))
        raw_score = float(selection.get("raw_best_score", 0.0) or 0.0)
        selected_score = float(selection.get("selected_score", raw_score) or raw_score)
        mask_available = int(selection.get("mask_available", 0) or 0)
        mask_area = float(selection.get("mask_area", math.nan))
        mask_bottom_u = (
            float(selection.get("mask_bottom_u", math.nan)) if mask_available else math.nan
        )
        mask_bottom_v = (
            float(selection.get("mask_bottom_v", math.nan)) if mask_available else math.nan
        )
        polygon = (selection.get("detection") or {}).get("polygon")
        mask_points = float(polygon.shape[0]) if mask_available and polygon is not None else math.nan
        height, width = image_bgr.shape[:2]
        visibility_grid = None
        rgb_context_crop = None
        if bbox is not None and bool(selection.get("detected_after_threshold", False)):
            from unav_common.visibility_patch import visibility_grid_from_bgr_frame
            visibility_grid = tuple(
                float(value) for value in visibility_grid_from_bgr_frame(
                    image_bgr, bbox
                ).reshape(-1)
            )
            bx0, by0, bx1, by1 = map(float, bbox)
            bw, bh = max(bx1 - bx0, 1.0), max(by1 - by0, 1.0)
            cx0 = max(0, int(math.floor(bx0 - 0.5 * bw)))
            cy0 = max(0, int(math.floor(by0 - 0.5 * bh)))
            cx1 = min(width, int(math.ceil(bx1 + 0.5 * bw)))
            cy1 = min(height, int(math.ceil(by1 + 0.5 * bh)))
            if cx1 > cx0 and cy1 > cy0:
                crop_rgb = cv2.cvtColor(
                    image_bgr[cy0:cy1, cx0:cx1], cv2.COLOR_BGR2RGB
                )
                crop_rgb = cv2.resize(crop_rgb, (96, 96), interpolation=cv2.INTER_LINEAR)
                crop_chw = np.ascontiguousarray(crop_rgb.transpose(2, 0, 1), dtype=np.uint8)
                rgb_context_crop = base64.b64encode(
                    zlib.compress(crop_chw.tobytes(), level=3)
                ).decode("ascii")
        if math.isfinite(selected_u) and math.isfinite(selected_v) and height > 0 and width > 0:
            border_margin = float(
                min(
                    selected_u,
                    selected_v,
                    float(width) - 1.0 - selected_u,
                    float(height) - 1.0 - selected_v,
                )
            )
        else:
            border_margin = math.nan
        bbox_area = (
            float(max(x1 - x0, 0.0) * max(y1 - y0, 0.0))
            if math.isfinite(x0)
            else math.nan
        )
        publish_stamp_s = self._clock_s()
        callback_ms = max((time.perf_counter() - item.receive_wall_s) * 1.0e3, 0.0)
        if math.isfinite(timing.predict_start_stamp_s) and math.isfinite(
            timing.predict_finish_stamp_s
        ):
            latency_s = max(
                timing.predict_finish_stamp_s - timing.predict_start_stamp_s, 0.0
            )
        else:
            latency_s = math.nan
        try:
            frame_age_s = frame_age_at_publish_s(
                publish_stamp_s=publish_stamp_s,
                image_stamp_s=stamp_s,
            )
        except BatchContractError as exc:
            self._fatal(
                f"future image-stamp integrity fault for {item.camera_id}: {exc}",
                exc,
            )
        detected = bool(selection.get("detected_after_threshold", False))
        message = diagnostics_message(
            stamp=stamp_s,
            detected=detected,
            u_mid=selected_u,
            v_mid=selected_v,
            yaw_est=math.nan,
            u_red=math.nan,
            v_red=math.nan,
            red_area_px=math.nan,
            u_blue=math.nan,
            v_blue=math.nan,
            blue_area_px=math.nan,
            separation_px=math.nan,
            border_margin_px=border_margin,
            yolo_score_raw=raw_score,
            yolo_score_selected=selected_score,
            yolo_detected_after_threshold=1.0 if detected else 0.0,
            yolo_best_class_id=float(selection.get("best_class_id", math.nan)),
            yolo_target_candidate_count=float(selection.get("n_candidates", 0)),
            bbox_area_px=bbox_area,
            bbox_xmin=x0,
            bbox_ymin=y0,
            bbox_xmax=x1,
            bbox_ymax=y1,
            logit_margin=math.nan,
            class_entropy=math.nan,
            mask_area_px=mask_area,
            mask_bottom_u=mask_bottom_u,
            mask_bottom_v=mask_bottom_v,
            mask_used=1.0 if mask_available else 0.0,
            mask_polygon_points=mask_points,
            confidence_logit=_confidence_logit(raw_score) if raw_score > 0.0 else math.nan,
            mask_compactness=math.nan,
            mask_border_frac=math.nan,
            mask_score=selected_score if mask_available else math.nan,
            selected_pixel_source_code=_selected_pixel_source_code(
                selection.get("selected_pixel_source", "none")
            ),
            yolo_inference_ms=timing.inference_wall_ms,
            detector_callback_ms=callback_ms,
            yolo_receive_stamp=item.receive_stamp_s,
            yolo_start_stamp=timing.predict_start_stamp_s,
            yolo_finish_stamp=timing.predict_finish_stamp_s,
            yolo_publish_stamp=publish_stamp_s,
            yolo_latency_s=latency_s,
            frame_age_at_publish_s=frame_age_s,
        )
        output = self.outputs[item.camera_id]
        observation_message = self._observation_message(
            item.camera_id, message, source_batch_id=source_batch_id,
            visibility_grid=visibility_grid,
            rgb_context_crop=rgb_context_crop,
        )
        try:
            output.diagnostics_publisher.publish(message)
            output.observation_publisher.publish(observation_message)
        except Exception as exc:
            self._fatal(f"operational publish failed for {item.camera_id}: {exc}", exc)

    def _observation_message(
        self,
        camera_id: str,
        diagnostics: Float64MultiArray,
        *,
        source_batch_id: str,
        visibility_grid: tuple[float, ...] | None = None,
        rgb_context_crop: str | None = None,
    ) -> String:
        try:
            observation = self._camera_observation_from_diagnostics(
                diagnostics_from_message(diagnostics),
                config=self._observation_configs[camera_id],
            )
            identity = self._active_members[camera_id]
            observation = replace(observation, source_batch_id=source_batch_id,
                                  producer_epoch=identity["producer_epoch"],
                                  source_frame_id=identity["source_frame_id"],
                                  capture_stamp_ns=identity["capture_stamp_ns"],
                                  detector_invocation_id=identity["detector_invocation_id"],
                                  visibility_grid_16x16=visibility_grid,
                                  rgb_context_crop_96x96_zlib_b64=rgb_context_crop)
            message = String()
            message.data = observation.to_json()
            return message
        except Exception as exc:
            self._fatal(
                f"CameraObservation contract failed for {camera_id}: {exc}", exc
            )

    def _write_debug_frame(
        self, item: PendingFrame, image_bgr: np.ndarray, selection: dict[str, Any]
    ) -> None:
        if not self.debug_frame_dir and not self.debug_crop_dir:
            return
        try:
            bbox = selection.get("bbox_xyxy")
            if self.debug_crop_dir and bbox is not None:
                from PIL import Image as PilImage

                x0, y0, x1, y1 = np.asarray(bbox, dtype=float).reshape(4)
                height, width = image_bgr.shape[:2]
                box_width, box_height = x1 - x0, y1 - y0
                rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                crop = PilImage.fromarray(rgb).crop((
                    int(max(0.0, x0 - 0.5 * box_width)),
                    int(max(0.0, y0 - 0.5 * box_height)),
                    int(min(float(width), x1 + 0.5 * box_width)),
                    int(min(float(height), y1 + 0.5 * box_height)),
                )).resize((96, 96), PilImage.Resampling.BILINEAR)
                filename = f"{item.camera_id}_{item.stamp_ns:019d}.npz"
                np.savez_compressed(
                    Path(self.debug_crop_dir).expanduser() / filename,
                    source_frame_id=np.asarray(item.source_frame_id),
                    producer_epoch=np.asarray(self._producer_epoch),
                    capture_stamp_ns=np.asarray(item.stamp_ns, dtype=np.int64),
                    crop=np.asarray(crop, dtype=np.uint8).transpose(2, 0, 1),
                    bbox_xyxy=np.asarray(bbox, dtype=np.float64),
                    selected_uv=np.asarray((
                        float(selection.get("selected_u", math.nan)),
                        float(selection.get("selected_v", math.nan)),
                    ), dtype=np.float64),
                    image_shape=np.asarray((height, width), dtype=np.int64),
                    confidence=np.asarray(
                        float(selection.get("selected_score", math.nan)), dtype=np.float64),
                )
            if self.debug_frame_dir:
                image = image_bgr.copy()
                if bbox is not None:
                    x0, y0, x1, y1 = np.asarray(bbox, dtype=float).reshape(4)
                    cv2.rectangle(
                        image, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 1
                    )
                u = float(selection.get("selected_u", math.nan))
                v = float(selection.get("selected_v", math.nan))
                if math.isfinite(u) and math.isfinite(v):
                    cv2.circle(image, (int(u), int(v)), 2, (0, 255, 255), -1)
                filename = f"{item.camera_id}_{item.stamp_ns:019d}.png"
                cv2.imwrite(str(Path(self.debug_frame_dir).expanduser() / filename), image)
        except Exception as exc:
            self._warn_bounded(item.camera_id + ":debug", f"debug frame write failed: {exc}")

    def _terminal_stop_request_cb(self, message) -> None:
        """Drain detector-owned work and durably close its outcome stream."""
        try:
            request = terminal_stop_request_from_json(message.data)
        except ValueError as exc:
            self._fatal("invalid terminal stop request", exc)
            return
        with self._terminal_lifecycle_lock:
            if self._terminal_stopped:
                if request.request_id != self._terminal_stop_request_id:
                    self._fatal("conflicting terminal stop request identity")
                return
            self._terminal_stop_request_id = request.request_id
            self._outcome_transport_enabled = False
            self.batcher.close()
            with self._async_pending_lock:
                pending = list(self._async_pending.values())
                self._async_pending.clear()
            if pending:
                self._publish_batch_outcome(dict(
                    status="incomplete_terminal_stop",
                    terminal_stop_request_id=request.request_id,
                    members=[frame_member(frame) for frame in pending],
                ))
            self._append_session_stopped(request_id=request.request_id)
            self._terminal_stopped = True
        ack = TerminalStopAck(
            request_id=request.request_id,
            component="detector",
            status="outcome_stream_quiescent",
            acknowledgement_stamp_ns=int(self.get_clock().now().nanoseconds),
            detail="durable_session_stopped",
        )
        response = String()
        response.data = terminal_stop_ack_to_json(ack)
        self._terminal_stop_ack_pub.publish(response)

    def _append_session_stopped(self, *, request_id=""):
        """Append the terminal record without consulting any ROS entity."""
        if getattr(self, "_session_stopped_appended", False):
            return None
        result = self._outcome_journal.append(dict(
            status="session_stopped",
            transport="journal_only",
            stage="detector",
            terminal_stop_request_id=str(request_id or ""),
            publish_stamp_s=float(
                getattr(self, "_clock_high_water_s", 0.0) or 0.0),
        ))
        self._session_stopped_appended = True
        return result

    def destroy_node(self):
        # main stops the executor first. DDS may already be unavailable, so final
        # pending drops and the final event sequence belong in the journal only.
        self._outcome_transport_enabled = False
        try:
            with self._terminal_lifecycle_lock:
                if not self._terminal_stopped:
                    self.batcher.close()
                    with self._async_pending_lock:
                        pending = list(self._async_pending.values())
                        self._async_pending.clear()
                    if pending:
                        self._publish_batch_outcome(dict(
                            status="incomplete_shutdown",
                            members=[frame_member(f) for f in pending],
                        ))
                    # The ROS context may already be invalid here. Closing evidence
                    # must depend only on the journal, not on a ROS publisher.
                    self._append_session_stopped()
                    self._terminal_stopped = True
        finally:
            self._outcome_journal.close()
            result = super().destroy_node()
        return result


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BatchedFourCameraYoloNode()
    executor = None
    try:
        if node.synchronization_mode == "asynchronous":
            executor = MultiThreadedExecutor(num_threads=2)
            executor.add_node(node)
            executor.spin()
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if executor is not None:
            executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
