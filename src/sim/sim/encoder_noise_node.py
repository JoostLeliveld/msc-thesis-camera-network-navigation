"""Simulated wheel-encoder odometry with realistic slip and drift.

Gazebo's diff-drive plugin publishes a perfect integral of whatever velocity
it was commanded.  Real cheap encoders accumulate independent slip errors on
top of that.  This node corrupts the reported velocity with a correlated
multiplicative slip model (same AR(1) structure as actuation_noise_node) and
republishes an integrated pose that drifts away from the simulator truth.

The planner subscribes to /odom_noisy instead of /odom and uses the noisy
velocity for its EKF predict step, making the belief diverge from truth
whenever camera observations stop arriving.

Topic wiring:
  /odom          <- Gazebo DiffDrive odometry (input_source 'odometry')
  /ground_truth_tf <- Gazebo world poses (input_source 'ground_truth', the default)
  /odom_noisy    -> consumed by the planner's predict step and heading correction

The DiffDrive odometry integrates the wheel joints, so the simulated wheel-floor
slip already corrupts it, by an amount no parameter describes. With input_source
'ground_truth' the encoder starts from the true body velocity instead, and the
noise declared here (slip, additive noise, systematic wheel errors) is the only
odometry error: the filter's process noise can then be set from this model.
"""

import math
import random
from typing import Any

try:  # Keep covariance propagation importable for non-ROS analysis/tests.
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.time import Time
    from tf2_msgs.msg import TFMessage
except ImportError:  # pragma: no cover - runtime launch always supplies ROS.
    rclpy = None
    Odometry = Any
    Node = object
    Time = Any
    TFMessage = Any


def _wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def systematic_wheel_velocities(v: float, w: float, diameter_ratio_error: float,
                                wheelbase_ratio: float, wheel_separation_m: float):
    """Body velocities the encoders report for a robot with systematic wheel errors.

    First-order differential-drive kinematics with right/left wheel diameters
    D(1 + e/2) and D(1 - e/2) and a true wheelbase rho times the nominal one, while the
    odometry assumes equal diameters and the nominal wheelbase b:
        v_enc = v - (e b / 4) w,     w_enc = rho w - (e / b) v.
    The two errors are the ones UMBmark isolates (Borenstein & Feng): unequal wheel
    diameters bend straight driving, a wrong wheelbase scales every turn.
    """
    b = float(wheel_separation_m)
    e = float(diameter_ratio_error)
    return (v - 0.25 * e * b * w, float(wheelbase_ratio) * w - e * v / b)


def body_velocity_from_poses(x0: float, y0: float, th0: float,
                            x1: float, y1: float, th1: float, dt: float):
    """True forward and angular velocity between two consecutive world poses.

    Forward speed is the displacement projected on the mean heading; the angular
    rate is the wrapped heading change. Exact for a unicycle at constant (v, w)
    to second order in dt.
    """
    dth = math.atan2(math.sin(th1 - th0), math.cos(th1 - th0))
    mid = th0 + 0.5 * dth
    v = ((x1 - x0) * math.cos(mid) + (y1 - y0) * math.sin(mid)) / dt
    return v, dth / dt


def _yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _quaternion_from_yaw(yaw: float):
    from geometry_msgs.msg import Quaternion
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


class EncoderNoiseNode(Node):
    """Republishes /odom with independent encoder slip added to the velocity.

    Pose is integrated from the first received odom message; it drifts from
    the Gazebo ground truth at a rate determined by the slip parameters.
    """

    def __init__(self):
        super().__init__('encoder_noise_node')

        self.declare_parameter('enabled', True)
        self.declare_parameter('input_topic', '/odom')
        self.declare_parameter('output_topic', '/odom_noisy')
        self.declare_parameter('input_frame_id', 'odom')
        self.declare_parameter('input_child_frame_id', 'base_footprint')
        self.declare_parameter('seed', 0)

        # Multiplicative slip on the true velocity (independent of actuation slip).
        # mean > 0 models systematic under-reading (e.g. wheel compression).
        self.declare_parameter('linear_slip_mean', 0.02)
        self.declare_parameter('linear_slip_std', 0.05)
        self.declare_parameter('angular_slip_mean', 0.00)
        self.declare_parameter('angular_slip_std', 0.03)

        # Additive white noise (vibration, quantisation).
        self.declare_parameter('linear_additive_std', 0.004)
        self.declare_parameter('angular_additive_std', 0.020)

        # AR(1) temporal correlation of the slip state.
        self.declare_parameter('correlation_alpha', 0.80)
        # 'ground_truth': true body velocity from /ground_truth_tf (see module docstring);
        # 'odometry': the DiffDrive wheel odometry on input_topic.
        self.declare_parameter('input_source', 'ground_truth')
        self.declare_parameter('ground_truth_topic', '/ground_truth_tf')
        self.declare_parameter('ground_truth_child_frame_id', 'turtlebot3')
        # The declared noise is per encoder sample at 50 Hz; ground truth arrives faster, so
        # it is decimated to this period (a faster rate would change the noise density).
        self.declare_parameter('encoder_period_s', 0.02)
        # Systematic wheel errors (a property of the robot, not random per run):
        # right/left diameter mismatch e and true/nominal wheelbase ratio. Zero and one
        # reproduce the previous, systematically calibrated encoder.
        self.declare_parameter('wheel_diameter_ratio_error', 0.0)
        self.declare_parameter('wheelbase_ratio', 1.0)
        self.declare_parameter('wheel_separation_m', 0.44)

        # Deadband: do not inject drift on a hard stop command.
        self.declare_parameter('stop_linear_deadband', 1e-4)
        self.declare_parameter('stop_angular_deadband', 1e-4)

        # Maximum plausible dt between consecutive odom messages (seconds).
        # Messages with larger gaps are skipped to avoid huge integration jumps.
        self.declare_parameter('max_dt_s', 0.5)

        # A propagated covariance accompanies the corrupted encoder pose.  It
        # is operational metadata: it depends only on the encoder-noise model
        # and commanded odometry, never on Gazebo truth.  Commissioning logs
        # use it as the uncertain-input GP covariance and later check its
        # calibration against evaluation-only truth.
        self.declare_parameter('initial_position_std_m', 0.01)
        self.declare_parameter('initial_yaw_std_rad', 0.01)
        # The encoder reading intentionally contains a systematic scale error
        # (``linear_slip_mean``).  It is not independent white noise: over a
        # straight aisle it accumulates coherently.  Carry a conservative,
        # operationally declared residual scale uncertainty in addition to the
        # white/AR process covariance, otherwise the reported ellipse is
        # spuriously narrow along the driving direction.
        self.declare_parameter('linear_scale_bias_std', -1.0)
        self.declare_parameter('covariance_floor_m2', 1.0e-8)
        self.declare_parameter('covariance_floor_yaw_rad2', 1.0e-8)

        self.enabled = bool(self.get_parameter('enabled').value)
        input_topic = str(self.get_parameter('input_topic').value)
        output_topic = str(self.get_parameter('output_topic').value)
        self.input_frame_id = str(self.get_parameter('input_frame_id').value)
        self.input_child_frame_id = str(self.get_parameter('input_child_frame_id').value)

        self.linear_slip_mean = float(self.get_parameter('linear_slip_mean').value)
        self.linear_slip_std = max(0.0, float(self.get_parameter('linear_slip_std').value))
        self.angular_slip_mean = float(self.get_parameter('angular_slip_mean').value)
        self.angular_slip_std = max(0.0, float(self.get_parameter('angular_slip_std').value))
        self.linear_additive_std = max(0.0, float(self.get_parameter('linear_additive_std').value))
        self.wheel_diameter_ratio_error = float(self.get_parameter('wheel_diameter_ratio_error').value)
        self.wheelbase_ratio = float(self.get_parameter('wheelbase_ratio').value)
        self.wheel_separation_m = float(self.get_parameter('wheel_separation_m').value)
        if self.wheel_separation_m <= 0.0 or self.wheelbase_ratio <= 0.0:
            raise ValueError('wheel_separation_m and wheelbase_ratio must be positive')
        self.angular_additive_std = max(0.0, float(self.get_parameter('angular_additive_std').value))
        self.correlation_alpha = min(max(float(self.get_parameter('correlation_alpha').value), 0.0), 0.999)
        self.stop_linear_deadband = max(0.0, float(self.get_parameter('stop_linear_deadband').value))
        self.stop_angular_deadband = max(0.0, float(self.get_parameter('stop_angular_deadband').value))
        self.max_dt_s = max(0.01, float(self.get_parameter('max_dt_s').value))
        self.initial_position_std_m = max(0.0, float(self.get_parameter('initial_position_std_m').value))
        self.initial_yaw_std_rad = max(0.0, float(self.get_parameter('initial_yaw_std_rad').value))
        declared_scale_std = float(self.get_parameter('linear_scale_bias_std').value)
        self.linear_scale_bias_std = (
            max(0.0, declared_scale_std)
            if declared_scale_std >= 0.0
            else abs(self.linear_slip_mean)
        )
        self.covariance_floor_m2 = max(0.0, float(self.get_parameter('covariance_floor_m2').value))
        self.covariance_floor_yaw_rad2 = max(
            0.0, float(self.get_parameter('covariance_floor_yaw_rad2').value)
        )

        seed = int(self.get_parameter('seed').value)
        # Use a different seed offset from actuation_noise_node so the two
        # slip processes are independent even with the same base seed.
        self._rng = random.Random(seed + 31337)
        self._linear_slip_state = 0.0
        self._angular_slip_state = 0.0

        # Encoder-integrated pose (world frame).  Initialised from first odom.
        self._pose_x: float | None = None
        self._pose_y: float | None = None
        self._pose_theta: float | None = None
        self._last_stamp = None
        # Interval anchor in integer nanoseconds. Advanced only after an interval
        # has actually been integrated, or by the deliberate large-gap rebase.
        self._last_stamp_ns = None
        self._pose_cov = self._initial_pose_covariance()
        # Jacobian of [x, y, yaw] with respect to a constant residual encoder
        # scale error.  Keeping it separately avoids adding a fully correlated
        # bias term repeatedly into the white-noise covariance recursion.
        self._linear_scale_jacobian = [0.0, 0.0, 0.0]

        self._pub = self.create_publisher(Odometry, output_topic, 10)
        self.input_source = str(self.get_parameter('input_source').value).strip().lower()
        if self.input_source not in ('ground_truth', 'odometry'):
            raise ValueError("input_source must be 'ground_truth' or 'odometry'")
        self._gt_child = str(self.get_parameter('ground_truth_child_frame_id').value)
        self._gt_prev = None
        self.encoder_period_ns = int(round(float(self.get_parameter('encoder_period_s').value) * 1e9))
        if self.input_source == 'odometry':
            self.create_subscription(Odometry, input_topic, self._odom_cb, 10)
        else:
            self.create_subscription(TFMessage, str(self.get_parameter('ground_truth_topic').value),
                                     self._ground_truth_cb, 50)

        self.get_logger().info(
            f'Encoder noise node: {self.input_source}:{input_topic if self.input_source == "odometry" else self.get_parameter("ground_truth_topic").value} -> {output_topic}, enabled={self.enabled}, '
            f'seed={seed}, lin_slip_mean={self.linear_slip_mean:.3f}, '
            f'lin_slip_std={self.linear_slip_std:.3f}, '
            f'ang_slip_std={self.angular_slip_std:.3f}, '
            f'alpha={self.correlation_alpha:.2f}'
        )

    def _update_correlated_state(self, old_value: float, std: float) -> float:
        if std <= 0.0:
            return 0.0
        innov_scale = math.sqrt(max(1.0 - self.correlation_alpha ** 2, 0.0))
        return self.correlation_alpha * old_value + innov_scale * self._rng.gauss(0.0, std)

    def _initial_pose_covariance(self):
        pos_var = max(self.initial_position_std_m ** 2, self.covariance_floor_m2)
        yaw_var = max(self.initial_yaw_std_rad ** 2, self.covariance_floor_yaw_rad2)
        return [[pos_var, 0.0, 0.0], [0.0, pos_var, 0.0], [0.0, 0.0, yaw_var]]

    def _propagate_pose_covariance(self, *, theta: float, v_true: float, w_true: float, dt: float,
                                   v_integrated=None, noise_active=True) -> tuple[float, float]:
        """Propagate planar encoder uncertainty for one noisy integration step.

        The AR(1) state has stationary standard deviation ``*_slip_std``.  Its
        low-frequency accumulation is represented by the finite correlation
        inflation below.  The separately propagated scale-bias Jacobian covers
        the declared residual systematic encoder scale uncertainty; it uses no
        simulator truth and is reported as part of the operational covariance.
        """

        c = math.cos(theta)
        s = math.sin(theta)
        velocity = v_true if v_integrated is None else float(v_integrated)
        f = (
            (1.0, 0.0, -velocity * dt * s),
            (0.0, 1.0, velocity * dt * c),
            (0.0, 0.0, 1.0),
        )
        p = self._pose_cov
        fp = [[sum(f[i][k] * p[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        propagated = [[sum(fp[i][k] * f[j][k] for k in range(3)) for j in range(3)] for i in range(3)]

        correlation_inflation = (1.0 + self.correlation_alpha) / max(1.0 - self.correlation_alpha, 1.0e-6)
        var_v = (v_true * self.linear_slip_std) ** 2 * correlation_inflation + self.linear_additive_std ** 2
        var_w = (w_true * self.angular_slip_std) ** 2 * correlation_inflation + self.angular_additive_std ** 2
        if not noise_active:
            var_v = var_w = 0.
        g_v = (dt * c, dt * s, 0.0)
        g_w = (0.0, 0.0, dt)
        for i in range(3):
            for j in range(3):
                propagated[i][j] += g_v[i] * var_v * g_v[j] + g_w[i] * var_w * g_w[j]
        for i in range(3):
            propagated[i][i] = max(
                propagated[i][i],
                self.covariance_floor_yaw_rad2 if i == 2 else self.covariance_floor_m2,
            )
        self._pose_cov = propagated
        previous_jacobian = list(getattr(self, '_linear_scale_jacobian', [0.0, 0.0, 0.0]))
        scale_increment = (dt * c * v_true, dt * s * v_true, 0.0) if noise_active else (0., 0., 0.)
        self._linear_scale_jacobian = [
            sum(f[i][k] * previous_jacobian[k] for k in range(3)) + scale_increment[i]
            for i in range(3)
        ]
        return var_v, var_w

    def _published_pose_covariance(self):
        """Return process covariance plus the declared coherent scale term."""

        p = self._pose_cov
        jacobian = list(getattr(self, '_linear_scale_jacobian', [0.0, 0.0, 0.0]))
        scale_std = max(0.0, float(getattr(self, 'linear_scale_bias_std', 0.0)))
        return [
            [p[i][j] + scale_std ** 2 * jacobian[i] * jacobian[j] for j in range(3)]
            for i in range(3)
        ]

    def _write_covariances(self, message: Odometry, *, var_v: float, var_w: float) -> None:
        """Write planar covariance into ROS's 6x6 pose/twist conventions."""

        p = self._published_pose_covariance()
        pose_cov = [0.0] * 36
        pose_cov[0] = p[0][0]
        pose_cov[1] = pose_cov[6] = p[0][1]
        pose_cov[5] = pose_cov[30] = p[0][2]
        pose_cov[7] = p[1][1]
        pose_cov[11] = pose_cov[31] = p[1][2]
        pose_cov[35] = p[2][2]
        message.pose.covariance = pose_cov

        twist_cov = [0.0] * 36
        twist_cov[0] = max(var_v, self.covariance_floor_m2)
        twist_cov[35] = max(var_w, self.covariance_floor_yaw_rad2)
        message.twist.covariance = twist_cov

    def _ground_truth_cb(self, msg) -> None:
        """Turn two consecutive true poses into an odometry sample for _odom_cb.

        The bridge leaves the per-transform stamps at zero, so the sample is stamped
        at receipt on the simulation clock, as the experiment logger does.
        """
        for tr in msg.transforms:
            if tr.child_frame_id != self._gt_child:
                continue
            q = tr.transform.rotation
            x, y = float(tr.transform.translation.x), float(tr.transform.translation.y)
            th = float(_yaw_from_quaternion(q))
            now_ns = int(self.get_clock().now().nanoseconds)
            prev = self._gt_prev
            period = getattr(self, 'encoder_period_ns', 20_000_000)
            if prev is not None and now_ns - prev[0] < period:
                return                      # decimate to the encoder rate
            self._gt_prev = (now_ns, x, y, th)
            if prev is None:
                return
            v, w = body_velocity_from_poses(prev[1], prev[2], prev[3], x, y, th,
                                            (now_ns - prev[0]) * 1e-9)
            out = Odometry()
            out.header.stamp = Time(nanoseconds=now_ns).to_msg()
            out.header.frame_id = self.input_frame_id
            out.child_frame_id = self.input_child_frame_id
            out.pose.pose.position.x, out.pose.pose.position.y = x, y
            out.pose.pose.orientation = q
            out.twist.twist.linear.x, out.twist.twist.angular.z = v, w
            self._odom_cb(out)
            return

    def _odom_cb(self, msg: Odometry) -> None:
        """Integrate one encoder interval, or leave every field untouched.

        Validate before mutating. An old or duplicate message previously moved
        ``_last_stamp`` before the ``dt <= 0`` gate returned, so it rebased the
        interval anchor and the following genuine message integrated a shortened
        interval -- silently deleting real motion from the encoder estimate.

        The stamp is copied rather than aliased: the incoming message is owned by
        the executor and must not become this node's interval anchor by reference.

        This node runs on a single-threaded executor with the default callback
        group, so no cross-callback lock is required here.
        """
        try:
            if (msg.header.frame_id != getattr(self, 'input_frame_id', 'odom')
                    or msg.child_frame_id != getattr(self, 'input_child_frame_id', 'base_footprint')):
                return
            q = msg.pose.pose.orientation
            quaternion = tuple(float(v) for v in (q.x,q.y,q.z,q.w))
            if (not all(math.isfinite(v) for v in quaternion)
                    or abs(sum(v*v for v in quaternion)-1.) > 1.e-6):
                return
            if msg.header.stamp.sec < 0 or not 0 <= msg.header.stamp.nanosec < 1_000_000_000:
                return
            stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
            pose_x = float(msg.pose.pose.position.x)
            pose_y = float(msg.pose.pose.position.y)
            pose_theta = float(_yaw_from_quaternion(msg.pose.pose.orientation))
            v_true = float(msg.twist.twist.linear.x)
            w_true = float(msg.twist.twist.angular.z)
        except (AttributeError, TypeError, ValueError):
            return
        if not all(math.isfinite(x) for x in (pose_x, pose_y, pose_theta, v_true, w_true)):
            return
        stamp = Time(nanoseconds=stamp_ns).to_msg()

        # First message: initialise encoder pose from Gazebo truth.
        if getattr(self, '_last_stamp_ns', None) is None:
            self._pose_x = pose_x
            self._pose_y = pose_y
            self._pose_theta = pose_theta
            self._last_stamp_ns = stamp_ns
            self._last_stamp = stamp
            return

        dt = (stamp_ns - self._last_stamp_ns) * 1e-9

        if dt <= 0.0:
            # Old or duplicate input. It must not move the anchor, the pose, the
            # covariance, the slip states or the random generator.
            return

        if dt > self.max_dt_s:
            # A positive large gap DOES rebase the interval baseline, deliberately:
            # holding the old baseline forever would make every later interval
            # exceed the cap and freeze the encoder permanently. This is an interval
            # baseline reset, NOT evidence that the pose is supported across the
            # gap -- the omitted motion remains an unresolved validity defect.
            self._last_stamp_ns = stamp_ns
            self._last_stamp = stamp
            return

        # From here the interval is valid and will be integrated exactly once.
        # Committing the anchor here rather than earlier keeps a refused message
        # from consuming the interval.
        self._last_stamp_ns = stamp_ns
        self._last_stamp = stamp

        stop = (abs(v_true) <= self.stop_linear_deadband
                and abs(w_true) <= self.stop_angular_deadband)

        # Evolve correlated slip states regardless of motion to keep continuity.
        self._linear_slip_state = self._update_correlated_state(
            self._linear_slip_state, self.linear_slip_std)
        self._angular_slip_state = self._update_correlated_state(
            self._angular_slip_state, self.angular_slip_std)

        if (not self.enabled) or stop:
            v_enc = 0.0 if stop else v_true
            w_enc = 0.0 if stop else w_true
        else:
            v_mult = max(0.0, 1.0 - self.linear_slip_mean + self._linear_slip_state)
            w_mult = 1.0 - self.angular_slip_mean + self._angular_slip_state
            v_sys, w_sys = systematic_wheel_velocities(
                v_true, w_true, getattr(self, 'wheel_diameter_ratio_error', 0.0),
                getattr(self, 'wheelbase_ratio', 1.0), getattr(self, 'wheel_separation_m', 0.44))
            v_enc = v_sys * v_mult + self._rng.gauss(0.0, self.linear_additive_std)
            # Angular encoder error also fires when the robot is turning with v=0.
            moving = (abs(v_true) > self.stop_linear_deadband
                      or abs(w_true) > self.stop_angular_deadband)
            w_enc = (w_sys * w_mult + self._rng.gauss(0.0, self.angular_additive_std)
                     if moving else w_true)

        # Propagate covariance before applying the mean motion so its Jacobian
        # uses the previous heading, matching the Euler integration below.
        var_v, var_w = self._propagate_pose_covariance(
            theta=self._pose_theta,
            v_true=v_true,
            w_true=w_true,
            dt=dt,
            v_integrated=v_enc,
            noise_active=self.enabled and not stop,
        )

        # Integrate noisy velocity.
        self._pose_x += v_enc * dt * math.cos(self._pose_theta)
        self._pose_y += v_enc * dt * math.sin(self._pose_theta)
        self._pose_theta = _wrap_angle(self._pose_theta + w_enc * dt)

        # Publish noisy odometry.
        out = Odometry()
        out.header.stamp = stamp
        out.header.frame_id = msg.header.frame_id or 'odom'
        out.child_frame_id = msg.child_frame_id or 'base_footprint'
        out.pose.pose.position.x = self._pose_x
        out.pose.pose.position.y = self._pose_y
        out.pose.pose.position.z = 0.0
        out.pose.pose.orientation = _quaternion_from_yaw(self._pose_theta)
        out.twist.twist.linear.x = v_enc
        out.twist.twist.angular.z = w_enc
        self._write_covariances(out, var_v=var_v, var_w=var_w)
        self._pub.publish(out)


def main(args=None):
    if rclpy is None:
        raise RuntimeError('rclpy is required; source the ROS workspace before launching encoder_noise_node')
    rclpy.init(args=args)
    node = EncoderNoiseNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except RuntimeError:
            pass


if __name__ == '__main__':
    main()
