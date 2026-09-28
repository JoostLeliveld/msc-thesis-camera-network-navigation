#!/usr/bin/env python3
"""Capture one frame from the plan-view camera of the warehouse world.

plan_view_camera sits 42 m above the origin and looks straight down. The frame
is the background of the thesis setup figure (make_thesis_setup.py).

The camera is not part of warehouse_v2.world.sdf. Start the simulation with
rendering on a GPU display (headless rendering produces a corrupt frame), add
the camera, bridge its topic and capture one frame:

    ros2 launch sim bringup_sim.launch.py world:=warehouse_v2.world.sdf
    ros2 run ros_gz_sim create -world warehouse_v2 -name plan_view_camera \
        -file src/sim/models/plan_view_camera/model.sdf
    ros2 run ros_gz_bridge parameter_bridge \
        "/plan_view_camera/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image"
    python3 figures/capture_plan_view.py --out logs/thesis/figures/gazebo_plan_view.png
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import cv2  # type: ignore
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

TOPIC = '/plan_view_camera/image_raw'


class OneShot(Node):
    def __init__(self, topic: str) -> None:
        super().__init__('plan_view_frame_capture')
        self.frame = None
        self.bridge = CvBridge()
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.create_subscription(Image, topic, self._on_image, qos)

    def _on_image(self, message: Image) -> None:
        if self.frame is None:
            self.frame = self.bridge.imgmsg_to_cv2(message, 'bgr8')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=pathlib.Path, required=True)
    parser.add_argument('--timeout', type=float, default=60.0)
    args = parser.parse_args()

    rclpy.init()
    node = OneShot(TOPIC)
    deadline = node.get_clock().now().nanoseconds + int(args.timeout * 1e9)
    while node.frame is None and node.get_clock().now().nanoseconds < deadline:
        rclpy.spin_once(node, timeout_sec=0.5)

    frame = node.frame
    node.destroy_node()
    rclpy.shutdown()

    if frame is None:
        print(f'no frame on {TOPIC} within {args.timeout:.0f} s; '
              'is the simulation running?', file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), frame)
    print(f'wrote {args.out} ({frame.shape[1]}x{frame.shape[0]})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
