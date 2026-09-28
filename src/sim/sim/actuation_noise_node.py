"""Command-space actuation noise for Gazebo experiments.

The planner publishes an intended command on ``/cmd_vel_raw``.  This node
publishes the command received by Gazebo on ``/cmd_vel`` after applying a small
bounded actuator/slip disturbance.  This makes the physical rollout less
perfect without conflating planner belief process noise with real robot motion.
"""

import math
import random

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class ActuationNoiseNode(Node):
    def __init__(self):
        super().__init__('actuation_noise_node')

        self.declare_parameter('enabled', True)
        self.declare_parameter('input_topic', '/cmd_vel_raw')
        self.declare_parameter('output_topic', '/cmd_vel')
        self.declare_parameter('diagnostics_topic', '/cmd_vel_noise/diagnostics')
        self.declare_parameter('seed', 0)
        self.declare_parameter('command_timeout_s', 0.5)

        # Fractional speed loss and temporally correlated slip/noise.
        self.declare_parameter('linear_slip_mean', 0.03)
        self.declare_parameter('linear_slip_std', 0.06)
        self.declare_parameter('angular_slip_mean', 0.00)
        self.declare_parameter('angular_slip_std', 0.04)
        self.declare_parameter('linear_additive_std', 0.008)
        self.declare_parameter('angular_additive_std', 0.035)
        self.declare_parameter('correlation_alpha', 0.85)

        # Do not inject drift into an intentional stop.
        self.declare_parameter('stop_linear_deadband', 1e-4)
        self.declare_parameter('stop_angular_deadband', 1e-4)

        # Match the TurtleBot3 command envelope used by the planner.
        self.declare_parameter('linear_min', 0.0)
        self.declare_parameter('linear_max', 0.22)
        self.declare_parameter('angular_min', -1.0)
        self.declare_parameter('angular_max', 1.0)

        self.enabled = bool(self.get_parameter('enabled').value)
        input_topic = str(self.get_parameter('input_topic').value)
        output_topic = str(self.get_parameter('output_topic').value)
        diagnostics_topic = str(self.get_parameter('diagnostics_topic').value)
        self.linear_slip_mean = float(self.get_parameter('linear_slip_mean').value)
        self.linear_slip_std = max(float(self.get_parameter('linear_slip_std').value), 0.0)
        self.angular_slip_mean = float(self.get_parameter('angular_slip_mean').value)
        self.angular_slip_std = max(float(self.get_parameter('angular_slip_std').value), 0.0)
        self.linear_additive_std = max(float(self.get_parameter('linear_additive_std').value), 0.0)
        self.angular_additive_std = max(float(self.get_parameter('angular_additive_std').value), 0.0)
        self.correlation_alpha = min(max(float(self.get_parameter('correlation_alpha').value), 0.0), 0.999)
        self.stop_linear_deadband = max(float(self.get_parameter('stop_linear_deadband').value), 0.0)
        self.stop_angular_deadband = max(float(self.get_parameter('stop_angular_deadband').value), 0.0)
        self.linear_min = float(self.get_parameter('linear_min').value)
        self.linear_max = float(self.get_parameter('linear_max').value)
        self.angular_min = float(self.get_parameter('angular_min').value)
        self.angular_max = float(self.get_parameter('angular_max').value)
        self.command_timeout_s = float(self.get_parameter('command_timeout_s').value)
        if not math.isfinite(self.command_timeout_s) or self.command_timeout_s <= 0.0:
            raise ValueError('command_timeout_s must be finite and positive')
        self._last_input_stamp_s = None
        self._watchdog_stopped = True
        self._epoch_rewind_latched = False

        seed = int(self.get_parameter('seed').value)
        self._rng = random.Random(seed)
        self._linear_slip_state = 0.0
        self._angular_slip_state = 0.0

        self._pub = self.create_publisher(Twist, output_topic, 1)
        self._diag_pub = self.create_publisher(Float64MultiArray, diagnostics_topic, 10)
        self.create_subscription(Twist, input_topic, self._cmd_cb, 1)
        self.create_timer(min(0.1, self.command_timeout_s/2.), self._watchdog_tick)
        self.publish_stop('startup')

        self.get_logger().info(
            'Actuation noise node started: '
            f'{input_topic} -> {output_topic}, enabled={self.enabled}, seed={seed}, '
            f'linear_slip_mean={self.linear_slip_mean:.3f}, linear_slip_std={self.linear_slip_std:.3f}, '
            f'linear_additive_std={self.linear_additive_std:.3f} m/s, '
            f'angular_slip_std={self.angular_slip_std:.3f}, '
            f'angular_additive_std={self.angular_additive_std:.3f} rad/s, '
            f'command_timeout={self.command_timeout_s:.3f}s (simulation clock)'
        )

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        if lower > upper:
            lower, upper = upper, lower
        return min(max(float(value), float(lower)), float(upper))

    def _update_correlated_state(self, old_value: float, std: float) -> float:
        if std <= 0.0:
            return 0.0
        innovation_scale = math.sqrt(max(1.0 - self.correlation_alpha * self.correlation_alpha, 0.0))
        return (
            self.correlation_alpha * float(old_value)
            + innovation_scale * self._rng.gauss(0.0, std)
        )

    def _cmd_cb(self, msg: Twist) -> None:
        now_s = self.get_clock().now().nanoseconds * 1e-9
        if self._epoch_rewind_latched:
            return
        if self._last_input_stamp_s is not None and now_s < self._last_input_stamp_s:
            self._epoch_rewind_latched = True
            self._last_input_stamp_s = now_s
            self._linear_slip_state = 0.0
            self._angular_slip_state = 0.0
            self.publish_stop('simulation_time_rewind')
            self.get_logger().error(
                'Simulation clock rewound; command adapter is latched at zero until restart'
            )
            return
        v_cmd = float(msg.linear.x)
        w_cmd = float(msg.angular.z)
        if not (math.isfinite(v_cmd) and math.isfinite(w_cmd)):
            self._pub.publish(Twist())
            self._watchdog_stopped = True
            self.get_logger().error('Non-finite input command: publishing zero velocity')
            return
        self._last_input_stamp_s = now_s

        stop_cmd = (
            abs(v_cmd) <= self.stop_linear_deadband
            and abs(w_cmd) <= self.stop_angular_deadband
        )

        if (not self.enabled) or stop_cmd:
            self._linear_slip_state *= self.correlation_alpha
            self._angular_slip_state *= self.correlation_alpha
            v_out = 0.0 if stop_cmd else self._clamp(v_cmd, self.linear_min, self.linear_max)
            w_out = 0.0 if stop_cmd else self._clamp(w_cmd, self.angular_min, self.angular_max)
            linear_multiplier = 1.0
            angular_multiplier = 1.0
            linear_additive = 0.0
            angular_additive = 0.0
        else:
            self._linear_slip_state = self._update_correlated_state(
                self._linear_slip_state, self.linear_slip_std
            )
            self._angular_slip_state = self._update_correlated_state(
                self._angular_slip_state, self.angular_slip_std
            )

            linear_multiplier = max(0.0, 1.0 - self.linear_slip_mean + self._linear_slip_state)
            angular_multiplier = 1.0 - self.angular_slip_mean + self._angular_slip_state

            linear_additive = (
                self._rng.gauss(0.0, self.linear_additive_std)
                if abs(v_cmd) > self.stop_linear_deadband else 0.0
            )
            # Wheel mismatch can induce yaw error during both arcs and nominally straight driving.
            angular_additive_active = (
                abs(v_cmd) > self.stop_linear_deadband
                or abs(w_cmd) > self.stop_angular_deadband
            )
            angular_additive = (
                self._rng.gauss(0.0, self.angular_additive_std)
                if angular_additive_active else 0.0
            )

            v_out = self._clamp(
                v_cmd * linear_multiplier + linear_additive,
                self.linear_min,
                self.linear_max,
            )
            w_out = self._clamp(
                w_cmd * angular_multiplier + angular_additive,
                self.angular_min,
                self.angular_max,
            )

        noisy = Twist()
        noisy.linear.x = float(v_out)
        noisy.angular.z = float(w_out)
        self._pub.publish(noisy)
        self._watchdog_stopped = bool(stop_cmd)

        stamp = float(self.get_clock().now().nanoseconds) * 1e-9
        diag = Float64MultiArray()
        diag.data = [
            stamp,
            1.0 if self.enabled else 0.0,
            v_cmd,
            w_cmd,
            float(v_out),
            float(w_out),
            float(linear_multiplier),
            float(angular_multiplier),
            float(linear_additive),
            float(angular_additive),
            float(self._linear_slip_state),
            float(self._angular_slip_state),
        ]
        self._diag_pub.publish(diag)

    def _watchdog_tick(self):
        """Stop from this separate process if the planner stops sending commands.

        Twist has no source timestamp: this bounds silence after receipt, not
        arbitrary transport delay. Use the same simulated clock as robot motion.
        """
        if self._last_input_stamp_s is None or self._watchdog_stopped:
            return
        elapsed = self.get_clock().now().nanoseconds * 1e-9 - self._last_input_stamp_s
        if elapsed < 0.0:
            self._epoch_rewind_latched = True
            self._linear_slip_state = 0.0
            self._angular_slip_state = 0.0
            self.publish_stop('simulation_time_rewind')
            self.get_logger().error(
                'Simulation clock rewound; command adapter is latched at zero until restart'
            )
        elif elapsed > self.command_timeout_s:
            self.publish_stop('command_timeout')
            self.get_logger().warn(f'Command watchdog stop: input age {elapsed:.3f}s')

    def publish_stop(self, _reason: str) -> None:
        self._pub.publish(Twist())
        self._watchdog_stopped = True


def main(args=None):
    rclpy.init(args=args)
    node = ActuationNoiseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop('shutdown')
        try:
            rclpy.spin_once(node, timeout_sec=0.05)
        except Exception:  # noqa: BLE001
            pass
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except RuntimeError:
            pass


if __name__ == '__main__':
    main()
