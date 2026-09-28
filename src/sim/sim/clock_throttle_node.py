#!/usr/bin/env python3
"""Throttle Gazebo's high-rate simulation clock before ROS nodes consume it."""

import math
import time

import rclpy
from rclpy.clock import Clock as RclpyClock
from rclpy.clock import ClockType
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rosgraph_msgs.msg import Clock


def _clock_qos() -> QoSProfile:
    # Match rclpy TimeSource's /clock subscription: depth=1, best effort.
    return QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)


class ClockThrottleNode(Node):
    def __init__(self):
        super().__init__('clock_throttle_node')

        self.declare_parameter('input_topic', '/clock_full')
        self.declare_parameter('output_topic', '/clock')
        self.declare_parameter('publish_rate_hz', 50.0)
        self.declare_parameter('duplicate_heartbeat_s', 1.0)

        self.input_topic = str(self.get_parameter('input_topic').value)
        self.output_topic = str(self.get_parameter('output_topic').value)
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        if not math.isfinite(self.publish_rate_hz) or self.publish_rate_hz <= 0.0:
            raise RuntimeError('publish_rate_hz must be a positive finite value')
        self.duplicate_heartbeat_s = float(self.get_parameter('duplicate_heartbeat_s').value)
        if not math.isfinite(self.duplicate_heartbeat_s) or self.duplicate_heartbeat_s <= 0.0:
            raise RuntimeError('duplicate_heartbeat_s must be a positive finite value')

        qos = _clock_qos()
        self._latest_msg = None
        self._received_count = 0
        self._published_count = 0
        self._last_published_stamp = None
        self._last_publish_wall_s = -math.inf

        self.create_subscription(Clock, self.input_topic, self._clock_cb, qos)
        self._pub = self.create_publisher(Clock, self.output_topic, qos)

        self._wall_clock = RclpyClock(clock_type=ClockType.SYSTEM_TIME)
        self.create_timer(
            1.0 / self.publish_rate_hz,
            self._publish_latest,
            clock=self._wall_clock,
        )

        self.get_logger().info(
            f"Throttling {self.input_topic} -> {self.output_topic} "
            f"at {self.publish_rate_hz:.1f} Hz"
        )

    def _clock_cb(self, msg: Clock):
        self._latest_msg = msg
        self._received_count += 1
        if self._received_count == 1:
            stamp = msg.clock.sec + msg.clock.nanosec * 1e-9
            self.get_logger().info(f"Received first input clock at t={stamp:.6f}s")

    def _publish_latest(self):
        message = self._latest_msg
        if message is None:
            return
        stamp_key = (message.clock.sec, message.clock.nanosec)
        now = time.monotonic()
        # Changed timestamps (including a reset) retain the original cadence.
        # A paused clock only needs a heartbeat for newly joined ROS nodes.
        if (stamp_key == self._last_published_stamp
                and now - self._last_publish_wall_s < self.duplicate_heartbeat_s):
            return
        self._pub.publish(message)
        self._last_published_stamp = stamp_key
        self._last_publish_wall_s = now
        self._published_count += 1
        if self._published_count == 1:
            stamp = self._latest_msg.clock.sec + self._latest_msg.clock.nanosec * 1e-9
            self.get_logger().info(f"Published first throttled clock at t={stamp:.6f}s")


def main(args=None):
    rclpy.init(args=args)
    node = ClockThrottleNode()
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
