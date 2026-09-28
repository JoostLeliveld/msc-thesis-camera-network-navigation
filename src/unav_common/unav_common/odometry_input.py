"""Semantic validation of the planar odometry input boundary; no ROS import."""
import math


def odometry_pose_yaw(message, *, expected_frame='odom', expected_child_frame='base_footprint'):
    if message.header.frame_id != expected_frame or message.child_frame_id != expected_child_frame:
        raise ValueError('odometry pose/twist frames differ from declared frames')
    p, q = message.pose.pose.position, message.pose.pose.orientation
    values = tuple(float(v) for v in (p.x, p.y, p.z, q.x, q.y, q.z, q.w))
    if not all(math.isfinite(v) for v in values):
        raise ValueError('nonfinite odometry pose')
    if abs(sum(v*v for v in values[3:]) - 1.) > 1.e-6:
        raise ValueError('odometry orientation must be a unit quaternion')
    return math.atan2(2.*(q.w*q.z + q.x*q.y), 1.-2.*(q.y*q.y + q.z*q.z))


def odometry_pose_is_available(message):
    """Infinite planar pose variance explicitly denotes omitted encoder motion.

    Twist may remain usable. This tests availability, not calibration of P.
    """
    return all(math.isfinite(float(message.pose.covariance[i])) for i in (0, 7, 35))
