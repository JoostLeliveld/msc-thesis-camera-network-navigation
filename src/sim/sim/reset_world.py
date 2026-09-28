#!/usr/bin/env python3
import sys
import time

import rclpy
from rclpy.node import Node
from ros_gz_interfaces.srv import ControlWorld


class ResetWorld(Node):
    def __init__(self):
        super().__init__('reset_world')
        self.declare_parameter('world_name', 'default')
        self.declare_parameter('timeout_s', 5.0)
        self.declare_parameter('reset_all', True)

        self.world_name = self.get_parameter('world_name').value
        self.timeout_s = float(self.get_parameter('timeout_s').value)
        self.reset_all = bool(self.get_parameter('reset_all').value)

        self.service_name = f'/world/{self.world_name}/control'
        self.client = self.create_client(ControlWorld, self.service_name)
        self.future = None
        self.done = False
        self.failed = False

        self.get_logger().info(f"Waiting for {self.service_name} to reset world")
        self.create_timer(0.2, self._tick)

    def _tick(self):
        if self.done:
            return
        if self.future is None:
            if not self.client.wait_for_service(timeout_sec=0.1):
                return
            req = ControlWorld.Request()
            req.world_control.reset.all = self.reset_all
            self.future = self.client.call_async(req)
            self.get_logger().info('World reset request sent')
            return

        if self.future.done():
            try:
                result = self.future.result()
            except Exception as exc:  # noqa: BLE001
                self.failed = True
                self.get_logger().error(f'World reset service failed: {exc}')
            else:
                if hasattr(result, 'success') and not result.success:
                    self.failed = True
                    self.get_logger().error('World reset reported failure')
                else:
                    self.get_logger().info('World reset complete')
            self.done = True


def main(args=None):
    rclpy.init(args=args)
    node = ResetWorld()
    start = time.monotonic()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.done:
                node.destroy_node()
                rclpy.shutdown()
                return 1 if node.failed else 0
            if node.timeout_s > 0.0 and (time.monotonic() - start) > node.timeout_s:
                node.get_logger().error(f"Timeout waiting for {node.service_name}")
                node.destroy_node()
                rclpy.shutdown()
                return 1
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()
        return 0


if __name__ == '__main__':
    sys.exit(main())
