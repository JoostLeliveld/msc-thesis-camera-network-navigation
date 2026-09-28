#!/usr/bin/env python3
from dataclasses import replace
import math
import threading
import uuid

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

from unav_common.mission_goal import (
    MISSION_GOAL_TOPIC, make_mission_goal, mission_goal_to_json, parse_mission_waypoints,
)
from unav_common.operational_belief import (
    OPERATIONAL_BELIEF_TOPIC, OPERATIONAL_BELIEF_TIMEOUT_S, OperationalBeliefReceiver,
)


class GoalMissionNode(Node):
    """Publishes the mission goal. Single-goal by default; if ``waypoints_json``
    is a non-empty JSON list of [x,y], drives them as an ordered multi-goal tour:
    the current waypoint is published to /goal_bev, and advances to the next once
    the belief is within ``arrival_radius_m`` of it. The FINAL waypoint is held so
    the logger can distinguish final success from arrival at an intermediate goal.
    Goal identity and coordinates also travel atomically on /mission/goal_state.
    """

    def __init__(self):
        super().__init__('goal_mission_node')

        self.declare_parameter('goal_x', 3.0)
        self.declare_parameter('goal_y', 3.0)
        self.declare_parameter('delay_seconds', 3.0)
        self.declare_parameter('repeat_rate', 1.0)
        self.declare_parameter('repeat_count', 0)
        self.declare_parameter('repeat_unchanged_goal', False)
        self.declare_parameter('frame_id', 'map_bev')
        self.declare_parameter('waypoints_json', '')
        self.declare_parameter('arrival_radius_m', 0.6)
        self.declare_parameter('wait_for_belief_before_first_goal', False)
        self.declare_parameter('initial_belief_max_sigma_m', 0.0)
        self.declare_parameter('operational_belief_timeout_s', OPERATIONAL_BELIEF_TIMEOUT_S)

        self.delay = float(self.get_parameter('delay_seconds').value)
        self.repeat_rate = float(self.get_parameter('repeat_rate').value)
        self.repeat_count = int(self.get_parameter('repeat_count').value)
        self.repeat_unchanged_goal = bool(
            self.get_parameter('repeat_unchanged_goal').value
        )
        self.frame_id = self.get_parameter('frame_id').value
        self.arrival_radius = float(self.get_parameter('arrival_radius_m').value)
        self.wait_for_belief = bool(
            self.get_parameter('wait_for_belief_before_first_goal').value
        )
        self.initial_belief_max_sigma_m = max(
            float(self.get_parameter('initial_belief_max_sigma_m').value), 0.0
        )
        self.operational_belief_timeout_s = float(self.get_parameter('operational_belief_timeout_s').value)

        # Reject the complete configured tour on malformed input. A parsed prefix
        # is not a substitute mission and must never become an executable goal.
        self.waypoints = parse_mission_waypoints(
            self.get_parameter('waypoints_json').value,
            self.get_parameter('goal_x').value, self.get_parameter('goal_y').value,
        )
        self._initialize_mission_state()
        self.wall_clock = Clock(clock_type=ClockType.SYSTEM_TIME)
        self.start_time = self.wall_clock.now()

        goal_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.goal_state_pub = self.create_publisher(String, MISSION_GOAL_TOPIC, qos_profile=goal_qos)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_bev', qos_profile=goal_qos)
        self.create_subscription(String, OPERATIONAL_BELIEF_TOPIC, self._belief_cb, goal_qos)

        period = 1.0 / max(self.repeat_rate, 0.1)
        self.create_timer(period, self._send_goal, clock=self.wall_clock)
        self.get_logger().info(
            f"Mission ready. {len(self.waypoints)} waypoint(s) {self.waypoints} in frame "
            f"'{self.frame_id}', first in {self.delay}s (arrival_radius {self.arrival_radius} m, "
            f"wait_for_belief={self.wait_for_belief}, "
            f"initial_sigma_max={self.initial_belief_max_sigma_m:.3f} m, "
            f"repeat_unchanged_goal={self.repeat_unchanged_goal})"
        )

    def _initialize_mission_state(self):
        """Initialize before subscriptions/timers can observe mission state."""
        self._mission_lock = threading.RLock()
        self._mission_epoch = uuid.uuid4().hex
        self._active_goal = None
        self._last_clock_ns = None
        self._mission_invalid_reason = ''
        self._mission_cancellation_pending = None
        self._mission_cancellation_published = False
        self._belief_epoch = None
        self.wp_idx = 0
        self._belief_xy = None
        self._belief_ready = False
        self._belief_sigma_m = math.inf
        self._belief_wait_logged = False
        self.sent_count = 0
        self._last_published_wp_idx = None
        self._goal_belief_identity_at_issue = None
        self._belief = None
        timeout = self.operational_belief_timeout_s
        if not math.isfinite(timeout) or timeout <= 0.:
            raise ValueError('operational_belief_timeout_s must be finite and positive')
        self._belief_receiver = OperationalBeliefReceiver(
            expected_frame=self.frame_id,
            max_age_s=timeout,
        )

    def _belief_cb(self, msg: String):
        with self._mission_lock:
            now_ns = self._observe_clock_locked()
            if now_ns is None:
                return
            self._belief_receiver.receive_json(msg.data, now_ns=now_ns)
            event = self._belief_receiver.latest
            if event is not None:
                if self._belief_epoch is not None and event.epoch != self._belief_epoch:
                    self._invalidate_mission_locked('belief_epoch_changed_requires_restart', now_ns)
                    return
                self._belief_epoch = event.epoch
            self._refresh_belief_locked(now_ns)
            if self._belief_ready and self._belief_wait_logged:
                self.get_logger().info(
                    f"Initial belief ready (max xy sigma={self._belief_sigma_m:.3f} m); "
                    "releasing mission goal"
                )
                self._belief_wait_logged = False

    def _refresh_belief_locked(self, now_ns):
        self._belief = self._belief_receiver.usable(now_ns=now_ns)
        if self._belief is None:
            self._belief_xy = None
            self._belief_sigma_m = math.inf
            self._belief_ready = False
            return
        self._belief_xy = self._belief.mean[:2]
        self._belief_sigma_m = math.sqrt(max(0., self._belief.covariance[0][0], self._belief.covariance[1][1]))
        self._belief_ready = (self.initial_belief_max_sigma_m <= 0.
                             or self._belief_sigma_m <= self.initial_belief_max_sigma_m)

    def _maybe_advance(self):
        with self._mission_lock:
            now_ns = self._observe_clock_locked()
            if now_ns is None:
                return
            self._refresh_belief_locked(now_ns)
            return self._maybe_advance_locked()

    def _maybe_advance_locked(self):
        """Advance to the next waypoint once the belief reaches the current one
        (except the last, which is held for the success/auto-stop logic)."""
        if (self.wp_idx >= len(self.waypoints) - 1 or self._belief is None
                or self._last_published_wp_idx != self.wp_idx or self._active_goal is None):
            return
        # Announce every goal before testing arrival. One held sample cannot
        # consume a tour of close/duplicate targets; a later prediction or a
        # new correction at the same target time can provide the next decision.
        if (self._belief.state_stamp_ns < self._active_goal.stamp_ns
                or self._belief.identity == self._goal_belief_identity_at_issue):
            return
        gx, gy = self.waypoints[self.wp_idx]
        bx, by = self._belief_xy
        if ((bx - gx) ** 2 + (by - gy) ** 2) ** 0.5 <= self.arrival_radius:
            self.wp_idx += 1
            self._active_goal = None
            self.get_logger().info(
                f"reached waypoint {self.wp_idx}/{len(self.waypoints) - 1} -> "
                f"advancing to ({self.waypoints[self.wp_idx][0]:.2f}, {self.waypoints[self.wp_idx][1]:.2f})"
            )

    def _send_goal(self):
        with self._mission_lock:
            self._send_goal_locked()

    def _invalidate_mission_locked(self, reason, now_ns):
        self._mission_invalid_reason = reason
        self._belief_xy = None
        self._belief_ready = False
        self._belief_sigma_m = math.inf
        self._belief = None
        self.get_logger().info(reason)
        active = self._active_goal
        if active is None:
            # Record a cancelled configured target, without ever claiming it was
            # issued as active. The last valid ROS time names this identity.
            active = make_mission_goal(mission_epoch=self._mission_epoch,
                stamp_ns=self._last_clock_ns or 0, frame_id=self.frame_id,
                waypoints=self.waypoints, tour_index=self.wp_idx)
            reason = 'before_first_publication:' + reason
        self._mission_cancellation_pending = replace(active, status='cancelled', reason=reason,
            status_stamp_ns=now_ns if now_ns >= 0 else None)
        self._publish_cancellation_locked()

    def _publish_cancellation_locked(self):
        if self._mission_cancellation_published or self._mission_cancellation_pending is None:
            return
        message = String()
        message.data = mission_goal_to_json(self._mission_cancellation_pending)
        try:
            self.goal_state_pub.publish(message)
        except Exception as exc:
            # Keep the invalid latch and retry the same event on the next tick.
            self.get_logger().info(f'Mission cancellation publication failed; will retry: {exc}')
            return
        self._mission_cancellation_published = True

    def _observe_clock_locked(self):
        if self._mission_invalid_reason:
            self._publish_cancellation_locked()
            return None
        now_ns = self.get_clock().now().nanoseconds
        rewound = self._last_clock_ns is not None and now_ns < self._last_clock_ns
        if now_ns < 0 or rewound:
            self._invalidate_mission_locked('mission_clock_rewind_requires_restart' if rewound
                                            else 'invalid_mission_clock_requires_restart', now_ns)
            self._belief_receiver.usable(now_ns=now_ns)
            return None
        self._last_clock_ns = now_ns
        self._belief_receiver.usable(now_ns=now_ns)
        return now_ns

    def _send_goal_locked(self):
        now_ns = self._observe_clock_locked()
        if now_ns is None:
            return
        self._refresh_belief_locked(now_ns)
        elapsed = (self.wall_clock.now() - self.start_time).nanoseconds * 1e-9
        if elapsed < self.delay:
            return
        if self.sent_count == 0 and self.wait_for_belief and not self._belief_ready:
            if not self._belief_wait_logged:
                sigma = (
                    f"{self._belief_sigma_m:.3f} m"
                    if math.isfinite(self._belief_sigma_m)
                    else "unavailable"
                )
                self.get_logger().info(
                    f"Holding first goal until planner belief is ready "
                    f"(max xy sigma={sigma}, required <= "
                    f"{self.initial_belief_max_sigma_m:.3f} m)"
                )
                self._belief_wait_logged = True
            return
        if self.repeat_count > 0 and self.sent_count >= self.repeat_count:
            return
        self._maybe_advance_locked()
        if (
            not getattr(self, 'repeat_unchanged_goal', False)
            and getattr(self, '_last_published_wp_idx', None) == self.wp_idx
        ):
            return
        if self._active_goal is None:
            self._active_goal = make_mission_goal(
                mission_epoch=self._mission_epoch, stamp_ns=now_ns,
                frame_id=self.frame_id, waypoints=self.waypoints, tour_index=self.wp_idx,
            )
            self._goal_belief_identity_at_issue = self._belief.identity if self._belief else None
        active = self._active_goal
        gx, gy = active.x, active.y
        goal = PoseStamped()
        goal.header.stamp.sec = active.stamp_sec
        goal.header.stamp.nanosec = active.stamp_nanosec
        goal.header.frame_id = active.frame_id
        goal.pose.position.x = gx
        goal.pose.position.y = gy
        goal.pose.orientation.w = 1.0
        state = String()
        state.data = mission_goal_to_json(active)
        # The atomic envelope is authoritative. A compatibility publication
        # failure retries the same identity/payload on the next tick.
        self.goal_state_pub.publish(state)
        self.goal_pub.publish(goal)
        self.sent_count += 1
        self._last_published_wp_idx = self.wp_idx
        self.get_logger().info(
            f"Goal published (wp {self.wp_idx + 1}/{len(self.waypoints)}) "
            f"at ({gx:.3f}, {gy:.3f}) frame='{self.frame_id}'"
        )


def main(args=None):
    rclpy.init(args=args)
    node = GoalMissionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
