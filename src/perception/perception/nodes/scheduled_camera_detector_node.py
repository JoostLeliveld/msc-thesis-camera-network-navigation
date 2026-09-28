#!/usr/bin/env python3
"""Reliability-aware scheduled multi-camera detector.

Instead of running four detectors every cycle (~1 Hz, batched, belief drifts
between corrections), this runs ONE inference per cycle on the camera the
coverage map says best sees the robot's current belief position, and hands over
to the next camera as the robot drives. Single-image inference restores
~3-4 Hz corrections (single-camera rate) while keeping full-warehouse coverage
through hand-over. It projects the detection by inverse perspective mapping --
box bottom-centre onto the floor plane, no parameters -- and publishes the
world-frame correction directly to /state/bev.
"""
from __future__ import annotations
import math
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO

from reliability.projection import camera_model_from_world
from reliability.contracts import CameraObservation

DEFAULT_CAMERA_SPECS = [
    ("camera_A", "external_camera", "/external_camera/image_raw"),
    ("camera_B", "external_camera_b", "/external_camera_b/image_raw"),
    ("camera_C", "external_camera_c", "/external_camera_c/image_raw"),
    ("camera_D", "external_camera_d", "/external_camera_d/image_raw"),
]
# Backwards-compatible module constant used by older launch/tests.
CAMS = DEFAULT_CAMERA_SPECS


def _camera_specs(camera_ids, model_includes, image_topics):
    ids = [str(value).strip() for value in camera_ids]
    models = [str(value).strip() for value in model_includes]
    topics = [str(value).strip() for value in image_topics]
    if not ids or len(ids) != len(models) or len(ids) != len(topics):
        raise ValueError("camera_ids, camera_model_includes, and camera_image_topics must align")
    if any(not value for value in (*ids, *models, *topics)):
        raise ValueError("camera IDs, model includes, and image topics must be non-empty")
    if len(set(ids)) != len(ids) or len(set(models)) != len(models) or len(set(topics)) != len(topics):
        raise ValueError("camera IDs, model includes, and image topics must each be unique")
    return list(zip(ids, models, topics))


class ScheduledCameraDetector(Node):
    def __init__(self) -> None:
        super().__init__("scheduled_camera_detector")
        gp = self.declare_parameter
        gp("model_path", ""); gp("world_sdf", ""); gp("coverage_artifact", "")
        gp("device", "0"); gp("imgsz", 640)
        gp("conf", 0.05); gp("iou", 0.45)
        gp("report_std_m", 0.15); gp("rate_hz", 5.0); gp("frame_id", "map_bev")
        gp("spawn_x", 0.0); gp("spawn_y", 0.0); gp("min_coverage", 0.02)
        gp("selection_mode", "coverage_best_with_fallback")
        gp("publish_camera_observation_json", True)
        gp("camera_calibration_id_prefix", "scheduled_multicamera")
        gp("camera_observation_r_visible_uv", 2.5)
        gp("camera_observation_r_miss_uv", 40.0)
        gp("camera_ids", [item[0] for item in DEFAULT_CAMERA_SPECS])
        gp("camera_model_includes", [item[1] for item in DEFAULT_CAMERA_SPECS])
        gp("camera_image_topics", [item[2] for item in DEFAULT_CAMERA_SPECS])
        g = lambda n: self.get_parameter(n).value
        self.cams = _camera_specs(
            g("camera_ids"), g("camera_model_includes"), g("camera_image_topics")
        )
        world = str(g("world_sdf")); self.device = str(g("device"))
        self.imgsz = int(g("imgsz")); self.conf = float(g("conf")); self.iou = float(g("iou"))
        self.rvar = float(g("report_std_m")) ** 2
        self.frame_id = str(g("frame_id")); self.min_cov = float(g("min_coverage"))
        self.belief = (float(g("spawn_x")), float(g("spawn_y")))
        self.selection_mode = str(g("selection_mode")).strip().lower()
        if self.selection_mode not in {"coverage_best_with_fallback", "round_robin"}:
            raise ValueError(
                "selection_mode must be coverage_best_with_fallback or round_robin"
            )
        self.publish_observations = bool(g("publish_camera_observation_json"))
        self.calibration_id_prefix = str(g("camera_calibration_id_prefix")).strip()
        self.r_visible_uv = float(g("camera_observation_r_visible_uv"))
        self.r_miss_uv = float(g("camera_observation_r_miss_uv"))

        self.model = YOLO(str(g("model_path")))
        self.cam_models = {
            cid: camera_model_from_world(world, include_name=mname)
            for cid, mname, _ in self.cams
        }
        d = np.load(str(g("coverage_artifact")))
        self.xs = np.asarray(d["xs"], float); self.ys = np.asarray(d["ys"], float)
        self.cov = {
            cid: np.asarray(d[f"P_camera_{cid.split('_', 1)[1]}_map"], float)
            for cid, _, _ in self.cams
        }

        self.bridge = CvBridge()
        self.latest = {cid: None for cid, _, _ in self.cams}
        self._last_processed_stamp_s = {cid: -math.inf for cid, _, _ in self.cams}
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=2)
        for cid, _, topic in self.cams:
            self.create_subscription(Image, topic, self._img_cb(cid), qos)
        self.create_subscription(PoseWithCovarianceStamped, "/planner_belief", self._belief_cb, 10)
        self.pub = self.create_publisher(PoseWithCovarianceStamped, "/state/bev", 10)
        self.observation_pubs = {
            cid: self.create_publisher(
                String, f"/perception/camera_observation/{cid}", 10
            )
            for cid, _, _ in self.cams
        } if self.publish_observations else {}
        # warm up the model off the hot path
        try:
            self.model.predict(source=np.zeros((720, 1280, 3), np.uint8), imgsz=self.imgsz,
                               conf=0.0, iou=self.iou, verbose=False,
                               device=(self.device or None))
        except Exception as exc:
            self.get_logger().warn(f"warmup failed: {exc}")
        self._counts = {cid: 0 for cid, _, _ in self.cams}
        self._round_robin_index = 0
        self.create_timer(1.0 / max(float(g("rate_hz")), 0.5), self._tick)
        self.get_logger().info("scheduled_camera_detector: one inference/cycle on the coverage-best camera -> /state/bev")

    def _img_cb(self, cid):
        def cb(msg: Image):
            self.latest[cid] = msg  # store raw; convert only the image we infer
        return cb

    def _belief_cb(self, msg: PoseWithCovarianceStamped):
        self.belief = (float(msg.pose.pose.position.x), float(msg.pose.pose.position.y))

    def _cov_at(self, cid, x, y):
        xi = int(np.clip(np.searchsorted(self.xs, x) - 1, 0, len(self.xs) - 1))
        yi = int(np.clip(np.searchsorted(self.ys, y) - 1, 0, len(self.ys) - 1))
        return float(self.cov[cid][yi, xi])

    @staticmethod
    def _stamp_s(msg: Image) -> float:
        return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1.0e-9

    def _publish_observation(
        self,
        *,
        cid: str,
        image_msg: Image,
        inference_ms: float,
        score: float = 0.0,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> None:
        publisher = self.observation_pubs.get(cid)
        if publisher is None:
            return
        stamp_s = self._stamp_s(image_msg)
        publish_s = float(self.get_clock().now().nanoseconds) * 1.0e-9
        detected = bbox is not None
        pixel = None
        if bbox is not None:
            x1, _y1, x2, y2 = bbox
            pixel = (float((x1 + x2) / 2.0), float(y2))
        r_uv = self.r_visible_uv if detected else self.r_miss_uv
        observation = CameraObservation(
            camera_id=cid,
            timestamp_s=stamp_s,
            pixel_uv=pixel,
            detection_valid=detected,
            detector_score=float(score if detected else 0.0),
            detector_score_raw=float(score if detected else 0.0),
            bbox_xyxy=bbox,
            bbox_bottom_uv=pixel,
            selected_pixel_source="bbox_bottom" if detected else "none",
            yolo_inference_wall_ms=float(inference_ms),
            image_receive_stamp_s=stamp_s,
            inference_finish_stamp_s=publish_s,
            publish_stamp_s=publish_s,
            frame_age_at_publish_s=max(publish_s - stamp_s, 0.0),
            measurement_age_s=max(publish_s - stamp_s, 0.0),
            calibration_id=f"{self.calibration_id_prefix}_{cid}",
            image_frame_id=cid,
            conditional_cov_uv=((r_uv * r_uv, 0.0), (0.0, r_uv * r_uv)),
            availability_probability=1.0 if detected else 0.0,
            association_probability=1.0 if detected else 0.0,
        )
        message = String()
        message.data = observation.to_json()
        publisher.publish(message)

    def _tick(self):
        bx, by = self.belief
        ranked = sorted(
            ((self._cov_at(cid, bx, by), cid) for cid, _, _ in self.cams), reverse=True
        )
        if self.selection_mode == "round_robin":
            ordered_ids = [cid for cid, _, _ in self.cams]
            cid = ordered_ids[self._round_robin_index % len(ordered_ids)]
            self._round_robin_index += 1
            ranked = [(self._cov_at(cid, bx, by), cid)]
        # Try cameras in coverage order until one detects (needed for robustness:
        # the best-coverage camera misses ~30% of the time, and a bootstrap with
        # zero corrections deadlocks the planner). Convert only the image we infer.
        for score, cid in ranked:
            if score < self.min_cov or self.latest[cid] is None:
                continue
            image_stamp_s = self._stamp_s(self.latest[cid])
            if image_stamp_s <= self._last_processed_stamp_s[cid]:
                continue
            # Claim the frame before inference. A detector exception must not
            # cause an infinite retry loop on one corrupt/stale message.
            self._last_processed_stamp_s[cid] = image_stamp_s
            try:
                img = self.bridge.imgmsg_to_cv2(self.latest[cid], "bgr8")
                inference_start = time.perf_counter()
                res = self.model.predict(source=img, imgsz=self.imgsz, conf=self.conf,
                                         iou=self.iou, verbose=False, device=(self.device or None))
                inference_ms = (time.perf_counter() - inference_start) * 1000.0
            except Exception as exc:
                self.get_logger().warn(f"predict failed on {cid}: {exc}"); return
            boxes = res[0].boxes
            if boxes is None or len(boxes) == 0:
                self._publish_observation(
                    cid=cid,
                    image_msg=self.latest[cid],
                    inference_ms=inference_ms,
                )
                continue  # this camera missed; try next-best
            xyxy = boxes.xyxy.cpu().numpy(); confs = boxes.conf.cpu().numpy()
            k = int(np.argmax(confs))
            x1, y1, x2, y2 = xyxy[k]
            bbox = (float(x1), float(y1), float(x2), float(y2))
            self._publish_observation(
                cid=cid,
                image_msg=self.latest[cid],
                inference_ms=inference_ms,
                score=float(confs[k]),
                bbox=bbox,
            )
            u = float((x1 + x2) / 2.0); v = float(y2)  # bottom-centre
            # Inverse perspective mapping, no parameters (e7, 2026-08-07).
            world = self.cam_models[cid].pixel_to_world(u, v)
            if world is None:
                continue
            now = self.get_clock().now()
            m = PoseWithCovarianceStamped()
            m.header.stamp = now.to_msg(); m.header.frame_id = self.frame_id
            m.pose.pose.position.x = float(world[0]); m.pose.pose.position.y = float(world[1])
            m.pose.pose.orientation.w = 1.0
            cov = [0.0] * 36; cov[0] = self.rvar; cov[7] = self.rvar; cov[35] = 1.0e6
            m.pose.covariance = cov
            self.pub.publish(m)
            self._counts[cid] += 1
            if sum(self._counts.values()) % 25 == 0:
                self.get_logger().info(f"handover counts {self._counts} belief=({bx:.1f},{by:.1f}) -> {cid}")
            return  # one correction per cycle


def main(args=None) -> int:
    rclpy.init(args=args)
    node = ScheduledCameraDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    main()
