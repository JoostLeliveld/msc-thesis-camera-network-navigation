#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker


class GoalMarkerNode(Node):
    def __init__(self):
        super().__init__('goal_marker_node')

        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', True)
        if not self.has_parameter('marker_topic'):
            self.declare_parameter('marker_topic', '/goal_marker')
        if not self.has_parameter('marker_ns'):
            self.declare_parameter('marker_ns', 'goal')
        if not self.has_parameter('scale'):
            self.declare_parameter('scale', 0.25)
        if not self.has_parameter('z'):
            self.declare_parameter('z', 0.08)
        if not self.has_parameter('color_r'):
            self.declare_parameter('color_r', 0.0)
        if not self.has_parameter('color_g'):
            self.declare_parameter('color_g', 0.35)
        if not self.has_parameter('color_b'):
            self.declare_parameter('color_b', 1.0)
        if not self.has_parameter('color_a'):
            self.declare_parameter('color_a', 1.0)

        marker_topic = str(self.get_parameter('marker_topic').value).strip() or '/goal_marker'
        self.marker_ns = str(self.get_parameter('marker_ns').value).strip() or 'goal'
        self.scale = float(self.get_parameter('scale').value)
        self.marker_z = float(self.get_parameter('z').value)
        self.color_r = float(self.get_parameter('color_r').value)
        self.color_g = float(self.get_parameter('color_g').value)
        self.color_b = float(self.get_parameter('color_b').value)
        self.color_a = float(self.get_parameter('color_a').value)

        sub_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        pub_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_bev', self._goal_cb, qos_profile=sub_qos
        )
        self.marker_pub = self.create_publisher(Marker, marker_topic, qos_profile=pub_qos)

        self.get_logger().info(
            f"Goal marker relay started (/goal_bev -> {marker_topic}, ns='{self.marker_ns}')"
        )

    def _goal_cb(self, msg: PoseStamped):
        marker = Marker()
        marker.header = msg.header
        marker.ns = self.marker_ns
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = msg.pose.position.x
        marker.pose.position.y = msg.pose.position.y
        marker.pose.position.z = self.marker_z
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.scale
        marker.scale.y = self.scale
        marker.scale.z = self.scale
        marker.color.r = self.color_r
        marker.color.g = self.color_g
        marker.color.b = self.color_b
        marker.color.a = self.color_a
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 0
        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = GoalMarkerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
