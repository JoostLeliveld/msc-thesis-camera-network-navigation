#!/usr/bin/env python3
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rosgraph_msgs.msg import Clock


class WaitForClock(Node):
    def __init__(self):
        super().__init__('wait_for_clock')
        self.declare_parameter('topic', '/clock')
        self.declare_parameter('timeout_s', 0.0)
        self.declare_parameter('min_messages', 3)

        self.topic = self.get_parameter('topic').value
        self.timeout_s = float(self.get_parameter('timeout_s').value)
        self.min_messages = max(2, int(self.get_parameter('min_messages').value))
        self.received = False
        self.valid_count = 0
        self.last_stamp_ns = None

        clock_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Clock, self.topic, self._cb, clock_qos)
        if self.timeout_s > 0.0:
            self.get_logger().info(f"Waiting for {self.min_messages} advancing messages on {self.topic} (timeout {self.timeout_s:.1f}s)")
        else:
            self.get_logger().info(f"Waiting for {self.min_messages} advancing messages on {self.topic} (no timeout)")

    def _cb(self, msg: Clock):
        stamp_ns = int(msg.clock.sec) * 1_000_000_000 + int(msg.clock.nanosec)
        if not 0 <= int(msg.clock.nanosec) < 1_000_000_000:
            self.valid_count = 0
            self.last_stamp_ns = None
            return
        if self.last_stamp_ns is None or stamp_ns > self.last_stamp_ns:
            self.valid_count += 1
        elif stamp_ns < self.last_stamp_ns:
            self.valid_count = 1
        self.last_stamp_ns = stamp_ns
        if not self.received and self.valid_count >= self.min_messages:
            self.received = True
            self.get_logger().info(
                f"Received {self.min_messages} advancing messages on {self.topic}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = WaitForClock()
    start = time.monotonic()

    def _safe_shutdown():
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except RuntimeError:
            pass

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.received:
                node.destroy_node()
                _safe_shutdown()
                return 0
            if node.timeout_s > 0.0 and (time.monotonic() - start) > node.timeout_s:
                node.get_logger().error(f"Timeout waiting for {node.topic}")
                node.destroy_node()
                _safe_shutdown()
                return 1
    except KeyboardInterrupt:
        node.destroy_node()
        _safe_shutdown()
        return 0


if __name__ == '__main__':
    sys.exit(main())
