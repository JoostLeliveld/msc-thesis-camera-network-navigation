import math
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Header, String
from ultralytics import YOLO

from perception.core.detection_diagnostics import (
    DETECTION_DIAGNOSTICS_TOPIC,
    diagnostics_from_message,
    diagnostics_message,
)
from perception.core.ros_image import image_msg_to_bgr8
from perception.core.yolo_selection import select_best_detection, target_class_ids


SELECTED_PIXEL_SOURCE_NONE = 0.0
SELECTED_PIXEL_SOURCE_BBOX_BOTTOM = 1.0
SELECTED_PIXEL_SOURCE_MASK_BOTTOM = 2.0


def _as_bool(value) -> bool:
    return str(value).strip().lower() in ('1', 'true', 't', 'yes', 'y', 'on')


def _confidence_logit(score: float, eps: float = 1e-6) -> float:
    p = float(np.clip(score, eps, 1.0 - eps))
    return float(math.log(p / (1.0 - p)))


def _selected_pixel_source_code(source: str) -> float:
    source = str(source or '').strip().lower()
    if source == 'mask_bottom':
        return SELECTED_PIXEL_SOURCE_MASK_BOTTOM
    if source == 'bbox_bottom':
        return SELECTED_PIXEL_SOURCE_BBOX_BOTTOM
    return SELECTED_PIXEL_SOURCE_NONE


class YoloRobotDetectorNode(Node):
    """Run one local YOLO model and publish the selected image-space robot observation."""

    def __init__(self):
        super().__init__('yolo_robot_detector_node')

        self.declare_parameter('model_path', '')
        self.declare_parameter('device', '')
        # Bound each detector process explicitly.  An uncapped CPU detector
        # previously consumed the host thread pool, slowed Gazebo to ~0.16 RTF,
        # and reduced the three GPU detector output rates as collateral damage.
        self.declare_parameter('cpu_num_threads', 0)
        self.declare_parameter('cpu_num_interop_threads', 1)
        self.declare_parameter('image_size', 640)
        self.declare_parameter('confidence_threshold', 0.25)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('class_name', 'robot')
        self.declare_parameter('class_id', -1)
        self.declare_parameter('use_masks', True)
        self.declare_parameter('mask_min_area', 12.0)
        self.declare_parameter('mask_bottom_band_px', 3.0)
        # Occlusion gate: reject a detection whose bbox area (px^2) is below this,
        # i.e. the robot is partially occluded / too small to localize reliably.
        # An occlusion-truncated box gives a confidently-WRONG bottom-centre that
        # poisons the EKF (no recovery); rejecting it yields no update -> belief
        # grows honestly and recovers on the next clean detection. 0 disables.
        self.declare_parameter('min_bbox_area_px', 0.0)
        # Debug: if set, save each frame with the detected box + selected point
        # drawn, named by header stamp, so the boxes can be inspected offline.
        self.declare_parameter('debug_frame_dir', '')
        self.declare_parameter('pixel_noise_sigma', 0.0)
        self.declare_parameter('seed', 0)
        # TIMING: load the TorchScript export of the model (forward = one C++
        # dispatch that holds the GIL once for the whole forward, instead of
        # per-layer eager dispatch). Pre/post-processing stay identical via the
        # Ultralytics YOLO() wrapper, so detections are bit-identical. Falls back
        # to model_path when the sibling .torchscript is missing.
        self.declare_parameter('use_torchscript', False)
        # Diagnostic successor: keep the native .pt checkpoint as the declared
        # source identity, while explicitly loading a fixed-shape compiled
        # artifact.  Defaults retain the established single-camera runtime.
        self.declare_parameter('runtime_backend', 'native')
        self.declare_parameter('compiled_model_path', '')
        self.declare_parameter('input_transport', 'ros')
        self.declare_parameter('runtime_trace_period_s', 0.0)
        # TIMING: run N dummy inferences at startup to pay the one-off lazy CUDA
        # / cuDNN init + (TorchScript) JIT specialization cost off the hot path,
        # so the first live frame is not a multi-hundred-ms stall.
        self.declare_parameter('warmup_iters', 3)
        # TIMING: run inference synchronously in the image callback (single
        # python thread) instead of a daemon worker. The worker shared the GIL
        # with rclpy.spin and PyTorch's per-layer dispatch thrashed against it,
        # inflating in-run inference ~10x. With queue depth 1, frames that arrive
        # during inference are dropped by the middleware, preserving the
        # latest-frame-only policy with zero intra-process GIL contention.
        self.declare_parameter('inference_in_callback', True)
        # Phase-1 camera-interface pass-through. Disabled by default so the
        # frozen C1/C2 runtime keeps identical topic contracts unless requested.
        self.declare_parameter('publish_camera_observation_json', False)
        self.declare_parameter('image_topic', '/external_camera/image_raw')
        self.declare_parameter('pixel_pose_topic', '/perception/pixel_pose')
        self.declare_parameter('diagnostics_topic', DETECTION_DIAGNOSTICS_TOPIC)
        self.declare_parameter('camera_observation_topic', '')
        self.declare_parameter('camera_id', 'camera_A')
        self.declare_parameter('camera_calibration_id', 'warehouse_aws_external_camera_v1')
        self.declare_parameter('camera_image_frame_id', 'external_camera')
        self.declare_parameter('camera_observation_r_visible_uv', 2.5)
        self.declare_parameter('camera_observation_r_miss_uv', 40.0)

        model_path = Path(str(self.get_parameter('model_path').value).strip()).expanduser()
        if not str(model_path):
            raise RuntimeError('model_path must be set for yolo_robot_detector_node')
        if not model_path.is_file():
            raise RuntimeError(f'model_path does not exist: {model_path}')

        self.model_path = model_path.resolve()
        if self.model_path.suffix.lower() != '.pt':
            raise RuntimeError('model_path must be the native source .pt checkpoint')
        self.runtime_backend = str(self.get_parameter('runtime_backend').value).strip().lower()
        self.compiled_model_path = None
        raw_compiled_path = str(self.get_parameter('compiled_model_path').value).strip()
        if self.runtime_backend not in ('native', 'torchscript'):
            raise RuntimeError("runtime_backend must be 'native' or 'torchscript'")
        if self.runtime_backend == 'torchscript':
            candidate = Path(raw_compiled_path).expanduser()
            if not candidate.is_file() or candidate.suffix.lower() != '.torchscript':
                raise RuntimeError('torchscript backend requires compiled_model_path ending in .torchscript')
            self.compiled_model_path = candidate.resolve()
        self.input_transport = str(self.get_parameter('input_transport').value).strip().lower()
        if self.input_transport not in ('ros', 'direct_gz'):
            raise RuntimeError("input_transport must be 'ros' or 'direct_gz'")
        legacy_torchscript = self.get_parameter('use_torchscript').value
        if self.runtime_backend == 'torchscript' and _as_bool(legacy_torchscript):
            raise RuntimeError('use_torchscript and runtime_backend=torchscript are mutually exclusive')
        self.runtime_trace_period_s = float(self.get_parameter('runtime_trace_period_s').value)
        if not math.isfinite(self.runtime_trace_period_s) or self.runtime_trace_period_s < 0.0:
            raise RuntimeError('runtime_trace_period_s must be finite and non-negative')
        self.device = str(self.get_parameter('device').value).strip()
        self.cpu_num_threads = max(0, int(self.get_parameter('cpu_num_threads').value))
        self.cpu_num_interop_threads = max(
            0, int(self.get_parameter('cpu_num_interop_threads').value)
        )
        if self.cpu_num_threads > 0:
            torch.set_num_threads(self.cpu_num_threads)
        if self.cpu_num_interop_threads > 0:
            try:
                torch.set_num_interop_threads(self.cpu_num_interop_threads)
            except RuntimeError as exc:
                # PyTorch permits setting this only before parallel work starts.
                # A launch should still fail visibly in timing gates rather than
                # taking down the detector solely because another library made
                # an early one-time call.
                self.get_logger().warn(f'could not set PyTorch interop threads: {exc}')
        self.image_size = int(self.get_parameter('image_size').value)
        self.confidence_threshold = float(self.get_parameter('confidence_threshold').value)
        self.iou_threshold = float(self.get_parameter('iou_threshold').value)
        self.class_name = str(self.get_parameter('class_name').value)
        self.class_id = int(self.get_parameter('class_id').value)
        self.use_masks = bool(self.get_parameter('use_masks').value) if isinstance(self.get_parameter('use_masks').value, bool) else _as_bool(self.get_parameter('use_masks').value)
        self.mask_min_area = float(self.get_parameter('mask_min_area').value)
        self.mask_bottom_band_px = float(self.get_parameter('mask_bottom_band_px').value)
        self.min_bbox_area_px = float(self.get_parameter('min_bbox_area_px').value)
        self.debug_frame_dir = str(self.get_parameter('debug_frame_dir').value).strip()
        if self.debug_frame_dir:
            import os as _os
            _os.makedirs(self.debug_frame_dir, exist_ok=True)
        self.pixel_noise_sigma = float(self.get_parameter('pixel_noise_sigma').value)
        self.rng = np.random.default_rng(int(self.get_parameter('seed').value))
        self.use_torchscript = bool(self.get_parameter('use_torchscript').value) if isinstance(self.get_parameter('use_torchscript').value, bool) else _as_bool(self.get_parameter('use_torchscript').value)
        self.warmup_iters = int(self.get_parameter('warmup_iters').value)
        self.inference_in_callback = bool(self.get_parameter('inference_in_callback').value) if isinstance(self.get_parameter('inference_in_callback').value, bool) else _as_bool(self.get_parameter('inference_in_callback').value)
        self.publish_camera_observation_json = bool(self.get_parameter('publish_camera_observation_json').value) if isinstance(self.get_parameter('publish_camera_observation_json').value, bool) else _as_bool(self.get_parameter('publish_camera_observation_json').value)
        self.image_topic = str(self.get_parameter('image_topic').value)
        self.pixel_pose_topic = str(self.get_parameter('pixel_pose_topic').value)
        self.diagnostics_topic = str(self.get_parameter('diagnostics_topic').value)
        self.camera_observation_topic = str(self.get_parameter('camera_observation_topic').value).strip()
        self.camera_id = str(self.get_parameter('camera_id').value)
        self.camera_calibration_id = str(self.get_parameter('camera_calibration_id').value)
        self.camera_image_frame_id = str(self.get_parameter('camera_image_frame_id').value)
        self.camera_observation_r_visible_uv = float(self.get_parameter('camera_observation_r_visible_uv').value)
        self.camera_observation_r_miss_uv = float(self.get_parameter('camera_observation_r_miss_uv').value)
        self._camera_observation_warned = False
        self._last_inference_ms = math.nan
        self._last_callback_ms = math.nan
        self._last_receive_stamp_s = math.nan
        self._last_predict_start_stamp_s = math.nan
        self._last_predict_finish_stamp_s = math.nan
        self._latest_image_shape = (0, 0)
        self._pending_image = None
        self._pending_receive_stamp_s = math.nan
        self._image_lock = threading.Lock()
        self._image_event = threading.Event()
        self._worker_stop = threading.Event()

        load_path = self.model_path
        if self.compiled_model_path is not None:
            load_path = self.compiled_model_path
        elif self.use_torchscript:
            ts_path = self.model_path.with_suffix('.torchscript')
            if ts_path.is_file():
                load_path = ts_path
            else:
                self.get_logger().warn(
                    f'use_torchscript=true but {ts_path} is missing; '
                    f'falling back to eager {self.model_path}'
                )
        self.model = YOLO(
            str(load_path),
            task='segment' if self.runtime_backend == 'torchscript' else None,
        )
        self.get_logger().info(
            f'PyTorch thread budget: intraop={torch.get_num_threads()} '
            f'interop={torch.get_num_interop_threads()} device={self.device or "auto"}'
        )
        self.target_ids = target_class_ids(getattr(self.model, 'names', {}), self.class_name, self.class_id)
        if self.target_ids == set():
            names = getattr(self.model, 'names', {})
            raise RuntimeError(
                f'class filter does not match any YOLO class names: class_name={self.class_name!r}, '
                f'class_id={self.class_id}, names={names!r}'
            )

        # Warmup off the hot path (lazy CUDA/cuDNN init, TorchScript JIT specialize).
        if self.warmup_iters > 0:
            _dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
            _wt0 = time.perf_counter()
            for _ in range(self.warmup_iters):
                try:
                    self._predict(_dummy)
                except Exception as exc:  # pragma: no cover - defensive
                    self.get_logger().warn(f'warmup predict failed: {exc}')
                    break
            self.get_logger().info(
                f'detector warmup: {self.warmup_iters} iters in '
                f'{(time.perf_counter() - _wt0) * 1e3:.0f}ms'
            )

        self.pixel_pub = self.create_publisher(PoseStamped, self.pixel_pose_topic, 10)
        self.diag_pub = self.create_publisher(Float64MultiArray, self.diagnostics_topic, 10)
        self.camera_observation_pub = None
        if self.publish_camera_observation_json:
            camera_observation_topic = self.camera_observation_topic or f'/perception/camera_observation/{self.camera_id}'
            self.camera_observation_pub = self.create_publisher(
                String,
                camera_observation_topic,
                10,
            )
        self._direct_gz_received = 0
        self._gz_transport_node = None
        if self.input_transport == 'direct_gz':
            from gz.msgs10.image_pb2 import Image as GzImage
            from gz.transport13 import Node as GzTransportNode, SubscribeOptions

            self._gz_image_type = GzImage
            self._gz_transport_node = GzTransportNode()
            self._gz_transport_node.subscribe_raw(
                self.image_topic,
                self._direct_gz_raw_image_cb,
                # Fortress advertises ignition.msgs.Image but shares the wire
                # format with gz.msgs.Image. Raw subscription avoids the
                # type-name handshake while removing the gz->ROS image copy.
                'ignition.msgs.Image',
                SubscribeOptions(),
            )
            self.get_logger().warn(
                'direct Gazebo input is a diagnostic successor: preserve source/output '
                'validation and do not use it for evidence recording'
            )
        else:
            self.create_subscription(Image, self.image_topic, self._image_cb, 1)
        # Worker thread is only started in the legacy (contended) path. Default
        # is single-threaded inference in the callback (no GIL contention).
        self._worker = None
        if not self.inference_in_callback:
            self._worker = threading.Thread(target=self._inference_worker, daemon=True)
            self._worker.start()

        self.get_logger().info(
            f'YOLO runtime detector started (model={load_path}, conf={self.confidence_threshold:.2f}, '
            f'iou={self.iou_threshold:.2f}, use_masks={self.use_masks}, '
            f'task=seg/det, device={self.device or "auto"}, '
            f'torchscript={self.use_torchscript}, backend={self.runtime_backend}, '
            f'input_transport={self.input_transport}, '
            f'inference_in_callback={self.inference_in_callback}, '
            f'camera_id={self.camera_id}, image_topic={self.image_topic}, '
            f'pixel_pose_topic={self.pixel_pose_topic}, diagnostics_topic={self.diagnostics_topic}, '
            'input_policy=latest-frame-only)'
        )
        if self.runtime_trace_period_s > 0.0:
            self.create_timer(self.runtime_trace_period_s, self._log_runtime_trace)

    def _predict(self, image_bgr: np.ndarray):
        kwargs = {
            'source': image_bgr,
            'imgsz': self.image_size,
            'conf': 0.0,
            'iou': self.iou_threshold,
            'verbose': False,
        }
        if self.device:
            kwargs['device'] = self.device
        return self.model.predict(**kwargs)

    def _publish_diagnostics(
        self,
        stamp_msg,
        selection: dict | None,
        pose_extra: dict | None = None,
    ) -> None:
        selection = dict(selection or {})
        pose_extra = dict(pose_extra or {})
        stamp = float(stamp_msg.sec) + float(stamp_msg.nanosec) * 1e-9
        detected_after_threshold = bool(selection.get('detected_after_threshold', False))
        bbox = selection.get('bbox_xyxy')
        if bbox is None:
            x0 = y0 = x1 = y1 = math.nan
        else:
            x0, y0, x1, y1 = [float(v) for v in np.asarray(bbox, dtype=float).reshape(4)]
        selected_u = float(selection.get('selected_u', math.nan))
        selected_v = float(selection.get('selected_v', math.nan))
        raw_score = float(selection.get('raw_best_score', 0.0) or 0.0)
        selected_score = float(selection.get('selected_score', raw_score) or raw_score)
        mask_area = float(selection.get('mask_area', math.nan))
        mask_available = int(selection.get('mask_available', 0) or 0)
        mask_bottom_u = float(selection.get('mask_bottom_u', math.nan)) if mask_available else math.nan
        mask_bottom_v = float(selection.get('mask_bottom_v', math.nan)) if mask_available else math.nan
        mask_points = float(
            selection.get('detection', {}).get('polygon').shape[0]
        ) if mask_available and selection.get('detection', {}).get('polygon') is not None else math.nan
        if math.isfinite(selected_u) and math.isfinite(selected_v):
            h = float(getattr(self, '_latest_image_shape', (0, 0))[0] or 0.0)
            w = float(getattr(self, '_latest_image_shape', (0, 0))[1] or 0.0)
            border_margin = float(min(selected_u, selected_v, w - 1.0 - selected_u, h - 1.0 - selected_v))
        else:
            border_margin = math.nan
        bbox_area = float(max(x1 - x0, 0.0) * max(y1 - y0, 0.0)) if math.isfinite(x0) else math.nan
        yaw_est = float(pose_extra.get('yaw_est', math.nan))
        u_front = float(pose_extra.get('u_front', math.nan))
        v_front = float(pose_extra.get('v_front', math.nan))
        u_rear = float(pose_extra.get('u_rear', math.nan))
        v_rear = float(pose_extra.get('v_rear', math.nan))
        if (
            math.isfinite(u_front) and math.isfinite(v_front)
            and math.isfinite(u_rear) and math.isfinite(v_rear)
        ):
            separation_px = float(math.hypot(u_front - u_rear, v_front - v_rear))
        else:
            separation_px = math.nan
        publish_stamp_s = float(self.get_clock().now().nanoseconds) * 1e-9
        if math.isfinite(self._last_predict_start_stamp_s) and math.isfinite(self._last_predict_finish_stamp_s):
            yolo_latency_s = max(self._last_predict_finish_stamp_s - self._last_predict_start_stamp_s, 0.0)
        elif math.isfinite(self._last_inference_ms):
            yolo_latency_s = max(self._last_inference_ms * 1e-3, 0.0)
        else:
            yolo_latency_s = math.nan
        frame_age_at_publish_s = max(publish_stamp_s - stamp, 0.0) if math.isfinite(stamp) else math.nan
        diag_msg = diagnostics_message(
            stamp=stamp,
            detected=detected_after_threshold,
            u_mid=selected_u,
            v_mid=selected_v,
            yaw_est=yaw_est,
            u_red=u_front,
            v_red=v_front,
            red_area_px=math.nan,
            u_blue=u_rear,
            v_blue=v_rear,
            blue_area_px=math.nan,
            separation_px=separation_px,
            border_margin_px=border_margin,
            yolo_score_raw=raw_score,
            yolo_score_selected=selected_score,
            yolo_detected_after_threshold=1.0 if detected_after_threshold else 0.0,
            yolo_best_class_id=float(selection.get('best_class_id', math.nan)),
            yolo_target_candidate_count=float(selection.get('n_candidates', 0)),
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
            selected_pixel_source_code=_selected_pixel_source_code(selection.get('selected_pixel_source', 'none')),
            yolo_inference_ms=self._last_inference_ms,
            detector_callback_ms=self._last_callback_ms,
            yolo_receive_stamp=self._last_receive_stamp_s,
            yolo_start_stamp=self._last_predict_start_stamp_s,
            yolo_finish_stamp=self._last_predict_finish_stamp_s,
            yolo_publish_stamp=publish_stamp_s,
            yolo_latency_s=yolo_latency_s,
            frame_age_at_publish_s=frame_age_at_publish_s,
        )
        self.diag_pub.publish(diag_msg)
        self._publish_camera_observation_json(diag_msg)

    def _publish_camera_observation_json(self, diag_msg: Float64MultiArray) -> None:
        if self.camera_observation_pub is None:
            return
        try:
            from reliability.single_camera_adapter import (
                SingleCameraAdapterConfig,
                camera_observation_from_diagnostics,
            )

            config = SingleCameraAdapterConfig(
                camera_id=self.camera_id,
                calibration_id=self.camera_calibration_id,
                image_frame_id=self.camera_image_frame_id,
                r_visible_uv=self.camera_observation_r_visible_uv,
                r_miss_uv=self.camera_observation_r_miss_uv,
            )
            obs = camera_observation_from_diagnostics(
                diagnostics_from_message(diag_msg),
                config=config,
            )
            msg = String()
            msg.data = obs.to_json()
            self.camera_observation_pub.publish(msg)
        except Exception as exc:  # pragma: no cover - defensive runtime bridge
            if not self._camera_observation_warned:
                self.get_logger().warn(f'camera observation JSON publish failed: {exc}')
                self._camera_observation_warned = True

    def _image_cb(self, msg: Image):
        receive_stamp_s = float(self.get_clock().now().nanoseconds) * 1e-9
        if self.inference_in_callback:
            # Single-thread: run inference inline. While this blocks, depth-1 QoS
            # keeps only the newest incoming frame, so the next spin picks up the
            # latest available image (latest-frame-only, no GIL contender).
            self._process_image(msg, receive_stamp_s)
            return
        with self._image_lock:
            self._pending_image = msg
            self._pending_receive_stamp_s = receive_stamp_s
        self._image_event.set()

    def _direct_gz_raw_image_cb(self, raw: bytes, _info) -> None:
        try:
            gz_msg = self._gz_image_type.FromString(raw)
            width, height, step = int(gz_msg.width), int(gz_msg.height), int(gz_msg.step)
            if width <= 0 or height <= 0 or step != width * 3:
                raise RuntimeError(f'unexpected Gazebo image geometry {width}x{height}, step={step}')
            if int(gz_msg.pixel_format_type) != 3:  # gz.msgs.RGB_INT8
                raise RuntimeError(f'unexpected Gazebo pixel format {int(gz_msg.pixel_format_type)}')
            data = np.frombuffer(bytes(gz_msg.data), dtype=np.uint8)
            if data.size != height * step:
                raise RuntimeError(f'image data size {data.size} does not match {height}x{step}')
            image = data.reshape(height, width, 3)[:, :, ::-1].copy()
            msg = Image()
            msg.header = Header()
            msg.header.stamp.sec = int(gz_msg.header.stamp.sec)
            msg.header.stamp.nanosec = int(gz_msg.header.stamp.nsec)
            msg.header.frame_id = 'image'
            msg.height = height
            msg.width = width
            msg.encoding = 'bgr8'
            msg.step = step
            msg.data = image.tobytes()
            self._direct_gz_received += 1
            self._image_cb(msg)
        except Exception as exc:
            self.get_logger().error(f'malformed direct Gazebo image: {exc}')

    def _log_runtime_trace(self) -> None:
        self.get_logger().info(
            f'runtime_trace transport={self.input_transport} '
            f'direct_gz_received={self._direct_gz_received} '
            f'last_inference_ms={self._last_inference_ms:.1f} '
            f'last_callback_ms={self._last_callback_ms:.1f}'
        )

    def _take_latest_image(self):
        with self._image_lock:
            msg = self._pending_image
            receive_stamp_s = self._pending_receive_stamp_s
            self._pending_image = None
            self._pending_receive_stamp_s = math.nan
        return msg, receive_stamp_s

    def _inference_worker(self):
        while not self._worker_stop.is_set():
            self._image_event.wait(timeout=0.1)
            if self._worker_stop.is_set():
                break
            self._image_event.clear()
            msg, receive_stamp_s = self._take_latest_image()
            if msg is None:
                continue
            self._process_image(msg, receive_stamp_s)
            with self._image_lock:
                has_newer_image = self._pending_image is not None
            if has_newer_image:
                self._image_event.set()

    def _process_image(self, msg: Image, receive_stamp_s: float):
        callback_start = time.perf_counter()
        self._last_receive_stamp_s = receive_stamp_s
        image_bgr = image_msg_to_bgr8(msg)
        self._latest_image_shape = image_bgr.shape[:2]
        if self.debug_frame_dir:
            self._latest_image_bgr = image_bgr
            try:
                import cv2 as _cv2, os as _os
                _raw = _os.path.join(self.debug_frame_dir, 'raw'); _os.makedirs(_raw, exist_ok=True)
                _st = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
                _cv2.imwrite(_os.path.join(_raw, f'frame_{_os.getpid()}_{_st:.3f}.png'), image_bgr)
            except Exception:
                pass
        predict_start = time.perf_counter()
        self._last_predict_start_stamp_s = float(self.get_clock().now().nanoseconds) * 1e-9
        results = self._predict(image_bgr)
        self._last_predict_finish_stamp_s = float(self.get_clock().now().nanoseconds) * 1e-9
        self._last_inference_ms = max((time.perf_counter() - predict_start) * 1000.0, 0.0)
        self._last_callback_ms = max((time.perf_counter() - callback_start) * 1000.0, 0.0)
        if not results:
            self._publish_diagnostics(msg.header.stamp, None)
            return

        self._handle_seg_result(msg, results[0])

    def destroy_node(self):
        self._worker_stop.set()
        self._image_event.set()
        worker = getattr(self, '_worker', None)
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)
        return super().destroy_node()

    # NOTE: _inference_worker / _take_latest_image below are only used when
    # inference_in_callback=False (legacy contended path, kept for A/B timing).

    def _handle_seg_result(self, msg: Image, result) -> None:
        selection = select_best_detection(
            result,
            target_ids=self.target_ids,
            confidence_threshold=float(self.confidence_threshold),
            use_masks=self.use_masks,
            mask_min_area=float(self.mask_min_area),
            mask_bottom_band_px=float(self.mask_bottom_band_px),
        )
        if selection.get('bbox_xyxy') is None:
            self._publish_diagnostics(msg.header.stamp, selection)
            return

        # Occlusion gate: a too-small box = partial occlusion -> biased bottom-centre.
        # Drop it to a non-detection so the EKF gets no (poisoning) update.
        if self.min_bbox_area_px > 0.0:
            _b = np.asarray(selection['bbox_xyxy'], dtype=float).reshape(4)
            _area = float(max(_b[2] - _b[0], 0.0) * max(_b[3] - _b[1], 0.0))
            if _area < self.min_bbox_area_px:
                selection['detected_after_threshold'] = False
                self._publish_diagnostics(msg.header.stamp, selection)
                return

        selected_u = float(selection.get('selected_u', math.nan))
        selected_v = float(selection.get('selected_v', math.nan))

        if self.pixel_noise_sigma > 0.0:
            selected_u += float(self.rng.normal(0.0, self.pixel_noise_sigma))
            selected_v += float(self.rng.normal(0.0, self.pixel_noise_sigma))
        selection['selected_u'] = selected_u
        selection['selected_v'] = selected_v
        if self.debug_frame_dir and getattr(self, '_latest_image_bgr', None) is not None:
            try:
                import cv2, os as _os
                im = self._latest_image_bgr.copy()
                b = np.asarray(selection['bbox_xyxy'], dtype=float).reshape(4)
                cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 1)
                if math.isfinite(selected_u) and math.isfinite(selected_v):
                    cv2.circle(im, (int(selected_u), int(selected_v)), 2, (0, 255, 255), -1)
                st = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
                cv2.imwrite(_os.path.join(self.debug_frame_dir, f'frame_{_os.getpid()}_{st:.3f}.png'), im)
            except Exception:
                pass
        self._publish_diagnostics(msg.header.stamp, selection)
        if not bool(selection.get('detected_after_threshold', False)):
            return

        out = PoseStamped()
        out.header = msg.header
        out.header.frame_id = 'image'
        out.pose.position.x = float(selected_u)
        out.pose.position.y = float(selected_v)
        out.pose.position.z = 0.0
        out.pose.orientation.w = 1.0
        self.pixel_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = YoloRobotDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
