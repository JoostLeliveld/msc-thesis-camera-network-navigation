"""ROS 2 node for EFE agent that publishes cmd_vel directly (unicycle dynamics)."""

import json
import math
import time
import threading
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

import os

import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, DurabilityPolicy

from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Float64MultiArray, String

from planning.nodes.unicycle_planner_node import UnicyclePlannerNode
from planning.planners.base_planner import UnicyclePlannerBase, extract_waypoints
from planning.core.dynamics import unicycle_step
from planning.core.efe_utils import wrap_angle
from planning.core.plan_validation import validate_plan_result
from planning.core.tracker_guard import ControlSafetyResult, SafetyFailure
from unav_common.config import local_controller_type
from unav_common.mission_goal import MISSION_GOAL_TOPIC, mission_goal_from_json
from unav_common.terminal_stop import (
    TERMINAL_STOP_ACK_TOPIC,
    TERMINAL_STOP_REQUEST_TOPIC,
    TerminalStopAck,
    terminal_stop_ack_to_json,
    terminal_stop_request_from_json,
)


def _preview_corner_speed_limit(
    waypoints: np.ndarray,
    nearest_index: int,
    xy: np.ndarray,
    *,
    v_max: float,
    preview_m: float = 0.90,
    max_decel_mps2: float = 0.90,
    corner_speed_mps: float = 0.30,
    min_corner_angle_rad: float = 0.35,
) -> float:
    """Braking-feasible speed for the first meaningful corner ahead.

    The global path is densely sampled, so this scans the upcoming polyline
    instead of waiting for a large heading error at the corner itself.
    """

    points = np.asarray(waypoints, dtype=float)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
        return float(v_max)
    j = int(np.clip(nearest_index, 0, len(points) - 1))
    first_corner_index = max(j, 1)
    distance_m = float(
        np.linalg.norm(
            points[first_corner_index, :2] - np.asarray(xy, dtype=float)[:2]
        )
    )
    limit = float(v_max)
    for k in range(first_corner_index, len(points) - 1):
        if k > first_corner_index:
            distance_m += float(np.linalg.norm(points[k, :2] - points[k - 1, :2]))
        if distance_m > preview_m:
            break
        incoming = points[k, :2] - points[k - 1, :2]
        outgoing = points[k + 1, :2] - points[k, :2]
        in_norm = float(np.linalg.norm(incoming))
        out_norm = float(np.linalg.norm(outgoing))
        if in_norm < 1.0e-6 or out_norm < 1.0e-6:
            continue
        cosine = float(np.clip(np.dot(incoming, outgoing) / (in_norm * out_norm), -1.0, 1.0))
        corner_angle = math.acos(cosine)
        if corner_angle < min_corner_angle_rad:
            continue
        allowed = math.sqrt(
            max(corner_speed_mps, 0.0) ** 2
            + 2.0 * max(max_decel_mps2, 0.0) * max(distance_m, 0.0)
        )
        limit = min(limit, allowed)
    return float(np.clip(limit, 0.0, v_max))


def _ff_fb_forward_speed(
    nominal_v: float,
    corner_cap: float,
    angular_velocity: float,
    heading_error: float,
    *,
    v_max: float,
    yaw_gate_rad: float,
    crawl_speed_mps: float = 0.18,
    pivot_heading_error_rad: float = 2.60,
) -> float:
    """Choose FF/FB translation speed, using a bounded arc through corners.

    Maintaining even the old 0.25 m/s floor during a 90-degree waypoint turn
    makes a differential-drive robot orbit a close waypoint.  The path tangent
    then keeps rotating and the waypoint may never be reached.  A mathematical
    zero-radius pivot is also a poor match for this platform: counter-rotating
    wheels advance encoder yaw faster than the physical body in Gazebo.  A
    low-speed arc keeps heading observable while retaining the 1 m/s ceiling
    on aligned straights. Only near-reversals require a pivot.
    """
    error = abs(float(heading_error))
    pivot_error = max(float(pivot_heading_error_rad), float(yaw_gate_rad) + 1.0e-6)
    if error > pivot_error:
        return 0.0
    lateral_cap = (
        float(v_max)
        if abs(float(angular_velocity)) < 1.0e-6
        else 0.65 / abs(float(angular_velocity))
    )
    base_cap = min(float(nominal_v), float(corner_cap), lateral_cap, float(v_max))
    if error > float(yaw_gate_rad):
        arc_end_error = max(
            min(0.5 * math.pi, pivot_error),
            float(yaw_gate_rad) + 1.0e-6,
        )
        fraction = float(np.clip(
            (error - float(yaw_gate_rad))
            / (arc_end_error - float(yaw_gate_rad)),
            0.0,
            1.0,
        ))
        smooth_fraction = fraction * fraction * (3.0 - 2.0 * fraction)
        crawl_cap = min(float(crawl_speed_mps), float(corner_cap), lateral_cap)
        return float((1.0 - smooth_fraction) * base_cap + smooth_fraction * crawl_cap)
    return float(np.clip(
        base_cap,
        0.05,
        float(v_max),
    ))


# Lateral overshoot allowed while the follower removes a heading error. It equals the
# 0.10 m geometric safety margin the world profile keeps between the planner's free
# space and every collision object and the site boundary
# (experiments.core.world_profiles, safety_margin_m); a test keeps the two equal.
TURN_OVERSHOOT_BUDGET_M = 0.10


def _ff_fb_turn_overshoot_cap(heading_error: float, *, w_limit: float,
                              budget_m: float = TURN_OVERSHOOT_BUDGET_M) -> float:
    """Speed at which removing `heading_error` stays within `budget_m` of lateral travel.

    Turning at the rate limit w on a circle of radius v / w, a heading error e is removed
    after a lateral displacement of (v / w)(1 - cos e). Bounding it by the budget gives
    v <= w * budget / (1 - cos e). Without this cap the follower carried 0.5-0.8 m/s into
    90-degree corners and swung about 0.5 m wide, out of the driveable region.
    """
    one_minus_cos = 1.0 - math.cos(min(abs(float(heading_error)), 0.5 * math.pi))
    if one_minus_cos <= 1.0e-9:
        return math.inf
    return abs(float(w_limit)) * float(budget_m) / one_minus_cos


def _ff_fb_arrival_speed_cap(
    target_distance_m: float, *, must_capture: bool, v_max: float,
) -> float:
    """Brake when the active target is a retained corner or mission endpoint.

    The FF/FB tracker removes collinear densification points before calling
    this helper.  Its remaining intermediate targets are therefore real route
    corners, not arbitrary 0.2 m samples.  Entering a sharp corner at the
    preview limiter's nonzero corner speed left too much distance for a held
    250 ms command plus drivetrain lag; the belief could switch segments while
    the physical robot overshot the driveable support boundary.
    """
    if not must_capture:
        return float(v_max)
    # Use less than one remaining-distance per second.  With the 4 Hz local
    # update, held-command latency, wheel slip and braking lag this leaves a
    # useful capture margin instead of arriving at a corner with residual
    # forward momentum.
    return float(min(v_max, max(0.05, 0.8 * float(target_distance_m))))


def _ff_fb_path_guidance(
    path_yaw: float,
    state_yaw: float,
    cross_track_m: float,
    *,
    v_max: float,
    w_limit: float,
) -> tuple[float, float, float]:
    """Return damped path-heading feedback and a cross-track speed cap.

    Feeding cross-track error directly into angular velocity made the 1 m/s
    follower alternate between the two sides of a narrow aisle.  Convert it to
    a bounded heading offset first (a Stanley-style guidance law), then close a
    heading loop.  Slow continuously as lateral error grows so actuator lag and
    the next 4 Hz replan cannot carry the body through the driveable boundary.
    """

    speed_scale = max(float(v_max), 0.05)
    guidance_yaw = float(path_yaw) - math.atan2(
        1.5 * float(cross_track_m), speed_scale,
    )
    heading_error = wrap_angle(guidance_yaw - float(state_yaw))
    angular_velocity = float(np.clip(
        1.8 * heading_error,
        -abs(float(w_limit)),
        abs(float(w_limit)),
    ))
    # Keep the declared 1 m/s ceiling on the centreline, but shed speed early
    # enough that a 5--10 cm lateral departure is corrected rather than
    # amplified.  The low floor permits recovery without forcing appreciable
    # translation while the body is returning to the route.
    cross_track_cap = float(v_max) * math.exp(-6.0 * abs(float(cross_track_m)))
    cross_track_cap = float(np.clip(cross_track_cap, 0.05, float(v_max)))
    return heading_error, angular_velocity, min(cross_track_cap, float(v_max))


def _tracking_waypoints(
    waypoints,
    waypoint_index: int,
    state_xy: np.ndarray,
) -> np.ndarray | None:
    """Return the remaining path with the current state as its first point."""

    if not waypoints:
        return None
    target_index = int(np.clip(waypoint_index, 0, len(waypoints) - 1))
    # Before the first waypoint the live state is the only valid segment
    # origin. After a waypoint transition, retain the previous route corner as
    # the segment anchor. Prepending the live state on every cycle erased
    # cross-track error and made a displaced robot drive parallel to the aisle
    # centreline instead of converging back to it.
    start = max(target_index - 1, 0)
    remaining = np.asarray(
        [(float(w[0]), float(w[1])) for w in waypoints[start:]], dtype=float
    )
    state = np.asarray(state_xy, dtype=float)[:2]
    if remaining.size == 0:
        return None
    if target_index == 0 and float(np.linalg.norm(remaining[0] - state)) > 1.0e-6:
        remaining = np.vstack((state, remaining))
    return remaining


def _waypoint_reached_or_passed(
    waypoints,
    waypoint_index: int,
    state_xy: np.ndarray,
    *,
    arrival_radius_m: float,
) -> bool:
    """Advance dense intermediate samples once the belief crosses their plane.

    Requiring a noisy belief to enter a small circle around every densified
    sample can leave a sample behind the robot. A neutral polyline follower
    should then continue along the route, not turn back to capture that sample.
    The final waypoint is deliberately handled elsewhere by the mission goal
    criterion.
    """

    if not waypoints:
        return False
    index = int(np.clip(waypoint_index, 0, len(waypoints) - 1))
    state = np.asarray(state_xy, dtype=float)[:2]
    target = np.asarray(waypoints[index], dtype=float)[:2]
    if float(np.linalg.norm(state - target)) < float(arrival_radius_m):
        return True
    if index == 0:
        if len(waypoints) < 2:
            return False
        # The first densified route point has no retained predecessor.  Use
        # its outgoing tangent as the crossing plane; otherwise an overshoot
        # leaves waypoint zero permanently behind and makes the tracker turn
        # around to retrace an already completed piece of route.
        segment = np.asarray(waypoints[1], dtype=float)[:2] - target
        origin = target
    else:
        origin = np.asarray(waypoints[index - 1], dtype=float)[:2]
        segment = target - origin
    segment_sq = float(segment @ segment)
    if segment_sq <= 1.0e-12:
        return True
    threshold = 0.0 if index == 0 else segment_sq
    return float((state - origin) @ segment) >= threshold


def _compress_collinear_waypoints(points: np.ndarray) -> np.ndarray:
    """Keep route corners/endpoints while removing exact densification points."""

    path = np.asarray(points, dtype=float)
    if path.ndim != 2 or path.shape[0] <= 2:
        return path
    kept = [path[0]]
    for index in range(1, len(path) - 1):
        incoming = path[index] - kept[-1]
        outgoing = path[index + 1] - path[index]
        incoming_norm = float(np.linalg.norm(incoming))
        outgoing_norm = float(np.linalg.norm(outgoing))
        if incoming_norm <= 1.0e-9 or outgoing_norm <= 1.0e-9:
            continue
        cosine = float((incoming @ outgoing) / (incoming_norm * outgoing_norm))
        cross = abs(float(incoming[0] * outgoing[1] - incoming[1] * outgoing[0]))
        sine = cross / (incoming_norm * outgoing_norm)
        if cosine > 0.999999 and sine < 1.0e-6:
            continue
        kept.append(path[index])
    kept.append(path[-1])
    return np.asarray(kept, dtype=float)


def _route_length_from(start_xy: np.ndarray, waypoints) -> float:
    """Polyline length including the otherwise implicit start-to-first leg."""

    points = np.asarray(
        [np.asarray(start_xy, dtype=float)[:2], *waypoints], dtype=float
    )
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        return float("inf")
    return float(np.sum(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)))


def _geometric_route_time_cost(
    start_state: np.ndarray,
    waypoints,
    *,
    v_max: float,
    pivot_rate_rad_s: float = 0.75,
) -> float:
    """Estimated traversal time: translation plus in-place heading changes."""

    start = np.asarray(start_state, dtype=float)
    points = np.asarray([start[:2], *waypoints], dtype=float)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        return float("inf")
    segments = np.diff(points[:, :2], axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    keep = lengths > 1.0e-6
    if not np.any(keep):
        return 0.0
    headings = np.arctan2(segments[keep, 1], segments[keep, 0])
    previous = float(start[2]) if start.size >= 3 else float(headings[0])
    turn_radians = 0.0
    for heading in headings:
        turn_radians += abs(wrap_angle(float(heading) - previous))
        previous = float(heading)
    travel_s = float(np.sum(lengths)) / max(float(v_max), 1.0e-3)
    pivot_s = turn_radians / max(float(pivot_rate_rad_s), 1.0e-3)
    return travel_s + pivot_s


def _route_states(start_state: np.ndarray, waypoints) -> np.ndarray:
    """Build drawable x/y/yaw states for a geometric waypoint route."""

    start = np.asarray(start_state, dtype=float)
    points = np.asarray([start[:2], *waypoints], dtype=float)
    states = np.zeros((len(points), 3), dtype=float)
    states[:, :2] = points[:, :2]
    states[0, 2] = float(start[2]) if start.size >= 3 else 0.0
    for index in range(len(points) - 1):
        delta = points[index + 1] - points[index]
        states[index, 2] = math.atan2(float(delta[1]), float(delta[0]))
    if len(points) > 1:
        states[-1, 2] = states[-2, 2]
    return states


def _polyline_states(points, *, initial_yaw: float = 0.0) -> np.ndarray:
    """Build drawable states from an exact polyline without adding/removing points."""

    route = np.asarray(points, dtype=float)
    if route.ndim != 2 or route.shape[0] < 2 or route.shape[1] != 2:
        raise ValueError("polyline must have shape (N,2), N >= 2")
    states = np.zeros((len(route), 3), dtype=float)
    states[:, :2] = route
    states[0, 2] = float(initial_yaw)
    for index in range(len(route) - 1):
        delta = route[index + 1] - route[index]
        states[index, 2] = math.atan2(float(delta[1]), float(delta[0]))
    states[-1, 2] = states[-2, 2]
    return states


class EfeAgentNode(UnicyclePlannerNode):
    NODE_NAME = 'efe_agent'
    PLANNER_CLASS = UnicyclePlannerBase

    def __init__(self):
        super().__init__()

        if not self.has_parameter('cmd_topic'):
            self.declare_parameter('cmd_topic', '/cmd_vel')
        self.cmd_topic = self.get_parameter('cmd_topic').value
        # Global route source for the hierarchical planner. 'preselected_route'
        # executes one launch-validated, hash-bound polyline and never invokes a
        # global EFE or shortest-path solve.
        if not self.has_parameter('global_planner_mode'):
            self.declare_parameter('global_planner_mode', 'efe')
        self.global_planner_mode = str(
            self.get_parameter('global_planner_mode').value or 'efe'
        ).strip().lower()
        if self.global_planner_mode not in (
            'efe', 'geometric_shortest_path', 'preselected_route'
        ):
            raise RuntimeError(
                "global_planner_mode must be 'efe', 'geometric_shortest_path', "
                "or 'preselected_route'"
            )

        route_parameter_defaults = {
            'preselected_route_json': '',
            'preselected_route_sha256': '',
            'preselected_route_source_path': '',
            'preselected_route_source_sha256': '',
            'preselected_route_validation_json': '',
        }
        for name, default in route_parameter_defaults.items():
            if not self.has_parameter(name):
                self.declare_parameter(name, default)
        self.preselected_route_json = str(
            self.get_parameter('preselected_route_json').value or ''
        )
        self.preselected_route_sha256 = str(
            self.get_parameter('preselected_route_sha256').value or ''
        )
        self.preselected_route_source_path = str(
            self.get_parameter('preselected_route_source_path').value or ''
        )
        self.preselected_route_source_sha256 = str(
            self.get_parameter('preselected_route_source_sha256').value or ''
        )
        self.preselected_route_validation_json = str(
            self.get_parameter('preselected_route_validation_json').value or ''
        )
        self._preselected_route_points = None
        self._preselected_route_provenance = None
        if self.global_planner_mode == 'preselected_route':
            if not self.use_hierarchical:
                raise RuntimeError(
                    "preselected_route requires use_hierarchical=true so the "
                    "belief-based local tracker executes the frozen polyline"
                )
            try:
                launch_validation = json.loads(self.preselected_route_validation_json)
                if not isinstance(launch_validation, dict):
                    raise ValueError('validation record must be a JSON object')
                if launch_validation.get('validation_status') != 'passed':
                    raise ValueError("validation_status must be 'passed'")
                from unav_common.preselected_route import validate_preselected_route

                validated = validate_preselected_route(
                    self.preselected_route_json,
                    self.preselected_route_sha256,
                    start_xy=launch_validation['registered_start_xy'],
                    goal_xy=launch_validation['registered_goal_xy'],
                    driveable_geometry_json=self.driveable_geometry_json,
                    declared_clearance_m=float(
                        launch_validation['declared_clearance_m']
                    ),
                    source_path=self.preselected_route_source_path,
                    expected_source_sha256=self.preselected_route_source_sha256,
                    endpoint_tolerance_m=float(
                        launch_validation['endpoint_tolerance_m']
                    ),
                    sample_step_m=float(
                        launch_validation['clearance_sample_step_m']
                    ),
                )
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"preselected route failed planner-side identity/geometry revalidation: {exc}"
                ) from exc
            self.preselected_route_json = validated.canonical_json
            self.preselected_route_sha256 = validated.sha256
            self.preselected_route_source_path = validated.source_path
            self.preselected_route_source_sha256 = validated.source_sha256
            self._preselected_route_points = validated.points
            self._preselected_route_provenance = validated.provenance_dict()
        elif any((
            self.preselected_route_json,
            self.preselected_route_sha256,
            self.preselected_route_source_path,
            self.preselected_route_source_sha256,
            self.preselected_route_validation_json,
        )):
            raise RuntimeError(
                "preselected route parameters were supplied for a different "
                "global_planner_mode; refusing to ignore them"
            )
        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)
        terminal_qos = QoSProfile(depth=16, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._terminal_stop_ack_pub = self.create_publisher(
            String, TERMINAL_STOP_ACK_TOPIC, terminal_qos
        )
        self._terminal_stop_requested = False
        self._terminal_stop_request_id = ""
        self.create_subscription(
            String,
            TERMINAL_STOP_REQUEST_TOPIC,
            self._terminal_stop_request_cb,
            terminal_qos,
            callback_group=self._io_group,
        )
        self.active_execution_diag_pub = self.create_publisher(
            Float64MultiArray, '/planner/active_execution_diagnostics', 10
        )
        self._active_controls = None
        self._active_plan_started_at = None
        self._active_controls_original_len = 0
        self._last_local_plan_target = None
        self._pending_plan_started_at = None
        self._pending_plan_started_active_remaining_s = 0.0
        # Incremented under ``_data_lock`` by every explicit ordinary safe stop,
        # atomically with clearing the tape and publishing zero. A planning
        # request captures the value it started under; a result whose generation
        # no longer matches has been cancelled and must not install, publish or
        # clear anyone else's tape. The fatal latch is separate and stays latched.
        self._command_stop_generation = 0
        # The request context of the planning call currently in flight, set by the
        # ``_plan_once`` wrapper and cleared in its ``finally``. The existing
        # mutually exclusive planning group makes this scoped context single-writer.
        self._active_plan_request = None
        self._mission_goal = None
        self._retired_mission_epochs = set()
        mission_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._mission_goal_sub = self.create_subscription(
            String, MISSION_GOAL_TOPIC, self._mission_goal_cb, mission_qos,
            callback_group=self._io_group)
        self._last_latency_skip_steps = 0
        self._last_latency_skip_s = 0.0
        self._current_wp_idx = math.nan
        self._current_wp_count = math.nan
        self._current_wp_target = np.array([math.nan, math.nan], dtype=float)
        self._current_wp_dist = math.nan
        self._current_desired_yaw = math.nan
        self._current_yaw_error = math.nan
        self._current_tracking_yaw = math.nan
        self._current_tracking_yaw_source = 0.0  # 0=belief/state, 1=odom override
        self._cmd_timer_period_s = 1.0 / max(float(self.cmd_publish_rate), 0.1)
        self._cmd_timer = self.create_timer(
            self._cmd_timer_period_s,
            self._publish_active_plan_command,
            callback_group=self._io_group,
        )

        # Two-stage hierarchical planning: self.planner is the LOCAL (lean,
        # short-horizon) tracker; self.global_planner computes one frozen
        # long-horizon visibility-aware plan whose states become waypoints.
        self._hier_phase = 'GLOBAL'
        self._waypoints = None
        self._wp_idx = 0
        # Multi-goal support: the (x,y) the frozen global route currently targets.
        # Any changed goal identity revokes the old route, including a new mission
        # goal ID at identical coordinates. Replan before tracking the new goal.
        self._global_goal_xy = None
        # True once a global route has passed admission. Failed replans stop;
        # they must not substitute an unchecked straight route.
        self._global_solve_done = False
        if not self.has_parameter('goal_replan_move_m'):
            self.declare_parameter('goal_replan_move_m', 1.0)
        self.goal_replan_move_m = float(self.get_parameter('goal_replan_move_m').value)
        # persistent state for the alternative trackers (hyst_damp damping/hysteresis)
        self._ctrl_prev_w = 0.0
        self._ctrl_spin = False
        # Persist the one-shot global route artifacts (solved plan + waypoints +
        # which seed won + costs). The global route is chosen ONCE and never
        # replanned, so it is the route-choice evidence for the campaign.
        self._run_dir = None
        self._pending_global_artifact = None
        self._global_artifact_saved = False
        run_dir_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/experiment/run_dir', self._run_dir_cb, run_dir_qos)
        self.global_planner = None
        if self.use_hierarchical:
            local_nogo_penalty_type = self.local_nogo_penalty_type or self.nogo_penalty_type
            local_nogo_weight = (
                self.nogo_weight if self.local_nogo_weight < 0.0 else self.local_nogo_weight
            )
            local_nogo_safe_distance = (
                self.nogo_safe_distance
                if self.local_nogo_safe_distance < 0.0
                else self.local_nogo_safe_distance
            )
            local_goal_u_final = (
                self.goal_prior_u_std_final
                if self.local_goal_prior_u_std_final < 0.0
                else self.local_goal_prior_u_std_final
            )
            local_goal_v_final = (
                self.goal_prior_v_std_final
                if self.local_goal_prior_v_std_final < 0.0
                else self.local_goal_prior_v_std_final
            )
            local_goal_u_start = (
                local_goal_u_final
                if self.local_goal_prior_u_std_start < 0.0
                else self.local_goal_prior_u_std_start
            )
            local_goal_v_start = (
                local_goal_v_final
                if self.local_goal_prior_v_std_start < 0.0
                else self.local_goal_prior_v_std_start
            )
            # The local planner object is retained only for the simple tracker's
            # collision/no-go geometry (safety check) and warm-start seeds; it is
            # never solved (the simple geometric tracker produces the commands).
            self.planner = self._construct_planner(
                horizon=self.local_horizon,
                use_ambiguity=self.local_use_ambiguity,
                use_obs_risk=self.local_use_obs_risk,
                goal_progress_n_steps=self.local_horizon,
                goal_prior_u_std_start=local_goal_u_start,
                goal_prior_v_std_start=local_goal_v_start,
                goal_prior_u_std_final=local_goal_u_final,
                goal_prior_v_std_final=local_goal_v_final,
                optimizer_multistart=self.local_optimizer_multistart,
                optimizer_multistart_include_direct=True,
                optimizer_initial_routes_json='',
                optimizer_warm_start_shift_steps=self._warm_start_shift_steps_for_rate(
                    self.local_plan_rate
                ),
                optimizer_maxiter=self.local_optimizer_maxiter,
                optimizer_maxfun=self.local_optimizer_maxiter * 4,
                use_belief_nogo_cost=self.local_use_belief_nogo_cost,
                use_visibility_model=self.local_use_visibility_model,
                nogo_penalty_type=local_nogo_penalty_type,
                nogo_weight=local_nogo_weight,
                nogo_safe_distance=local_nogo_safe_distance,
                # Never solved, so the locked objective's constants do not apply
                # to it. Without this the lock warns on every run about a local
                # setting that is off by design, which teaches the reader to
                # ignore the warning that matters. See docs/PLANNER.md.
                enforce_planner_lock=False,
            )
            if self.global_planner_mode != 'preselected_route':
                self.global_planner = self._construct_planner(
                    horizon=self.global_horizon,
                    dt=self.global_dt,
                    use_ambiguity=self.global_use_ambiguity,
                    optimizer_multistart=self.global_optimizer_multistart,
                    optimizer_warm_start_shift_steps=self._warm_start_shift_steps_for_rate(
                        self.plan_rate
                    ),
                    # This is the planner that solves the locked objective.
                    enforce_planner_lock=True,
                )
            global_description = (
                f"preselected route {self.preselected_route_sha256[:12]} "
                "(no global planner constructed)"
                if self.global_planner_mode == 'preselected_route'
                else (
                    f"global H={self.global_horizon} (dt={self.global_dt:.3f}s, "
                    f"lookahead={self.global_horizon * self.global_dt:.1f}s, "
                    f"ambiguity={self.global_use_ambiguity}, "
                    f"multistart={self.global_optimizer_multistart})"
                )
            )
            self.get_logger().info(
                f"[hierarchical] {global_description} -> waypoints "
                f"(spacing {self.waypoint_spacing_m} m) -> local H={self.local_horizon} "
                f"(rate={self.local_plan_rate} Hz, ambiguity={self.local_use_ambiguity}, "
                f"visibility={self.local_use_visibility_model}, "
                f"belief_nogo={self.local_use_belief_nogo_cost}, "
                f"nogo={local_nogo_penalty_type}:{local_nogo_weight}, "
                f"safe={local_nogo_safe_distance}, "
                f"local_goal_std={local_goal_u_start:.2f}->{local_goal_u_final:.2f}/"
                f"{local_goal_v_start:.2f}->{local_goal_v_final:.2f}, "
                f"replan_min_remaining={self.local_replan_min_remaining_s:.2f}s, "
                f"latency_compensate={self.latency_compensate_plan_handoff}, "
                f"cmd_rate={self.cmd_publish_rate:.1f}Hz, "
                f"simple_yaw_gate={self.simple_tracker_yaw_gate_rad:.2f}rad, "
                f"multistart={self.local_optimizer_multistart}, "
                f"maxiter={self.local_optimizer_maxiter})"
            )

    def _run_dir_cb(self, msg: String):
        self._run_dir = str(msg.data or '').strip() or None
        # Flush any global-route artifact captured before the run dir was known.
        if self._run_dir and self._pending_global_artifact is not None:
            self._write_global_artifact(self._pending_global_artifact)

    def _densify_line(self, p0, p1):
        """Straight-line waypoint list from p0 to p1 at waypoint_spacing_m, ending
        exactly at p1. Fallback route when a multi-goal replan solve fails."""
        p0 = np.asarray(p0, dtype=float).reshape(2)
        p1 = np.asarray(p1, dtype=float).reshape(2)
        dist = float(np.linalg.norm(p1 - p0))
        spacing = max(float(getattr(self, 'waypoint_spacing_m', 0.12)), 1e-3)
        n = max(int(math.ceil(dist / spacing)), 1)
        return [(float(x), float(y))
                for x, y in (p0 + (p1 - p0) * (k / n) for k in range(1, n + 1))]

    def _save_global_plan_artifacts(self, rg, m0, final_goal):
        """Capture the one-shot solved global route and persist it (or defer until
        the run directory is known)."""
        if self._global_artifact_saved:
            return
        try:
            states = np.asarray(rg.states, dtype=float)
            waypoints = [(float(w[0]), float(w[1])) for w in (self._waypoints or [])]
            seeds = [
                {'name': str(r.get('name', '')), 'waypoints': [[float(a), float(b)] for a, b in r.get('waypoints', [])]}
                for r in getattr(self.global_planner, 'optimizer_initial_routes', []) or []
            ]
            meta = {
                'selected_source': str(getattr(rg, 'selected_source', '')),
                'route_seed_mode': str(getattr(self, 'optimizer_route_seed_mode', 'explicit')),
                'route_seeds': seeds,
                'total_cost': float(getattr(rg, 'total_cost', float('nan'))),
                'risk_cost': float(getattr(rg, 'risk_cost', float('nan'))),
                'ambiguity_cost': float(getattr(rg, 'ambiguity_cost', float('nan'))),
                'obstacle_cost': float(getattr(rg, 'obstacle_cost', float('nan'))),
                'rollout_valid': bool(getattr(rg, 'rollout_valid', True)),
                'terminal_goal_distance_pred': float(getattr(rg, 'terminal_goal_distance_pred', float('nan'))),
                'global_horizon': int(self.global_horizon),
                'start_xy_yaw': [float(m0[0]), float(m0[1]), float(m0[2])],
                'goal_xy': [float(final_goal[0]), float(final_goal[1])],
                'n_states': int(states.shape[0]),
                'n_waypoints': int(len(waypoints)),
            }
            artifact = {'states': states, 'waypoints': waypoints, 'meta': meta}
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"failed to capture global route artifact: {exc}")
            return
        if self._run_dir:
            self._write_global_artifact(artifact)
        else:
            self._pending_global_artifact = artifact

    def _save_preselected_route_artifacts(self, states, final_goal):
        """Persist the exact canonical bytes that the local tracker received."""

        if self._global_artifact_saved:
            return
        provenance = dict(self._preselected_route_provenance or {})
        meta = {
            'selected_source': 'preselected_route',
            'global_planner_mode': 'preselected_route',
            'global_solve_invoked': False,
            'route_sha256': self.preselected_route_sha256,
            'route_source_path': self.preselected_route_source_path,
            'route_source_sha256': self.preselected_route_source_sha256,
            'goal_xy': [float(final_goal[0]), float(final_goal[1])],
            'n_states': int(np.asarray(states).shape[0]),
            'n_waypoints': int(len(self._preselected_route_points or ())),
        }
        artifact = {
            'states': np.asarray(states, dtype=float),
            'waypoints': list(self._preselected_route_points or ()),
            'meta': meta,
            'canonical_route_json': self.preselected_route_json,
            'preselected_route_provenance': provenance,
        }
        if self._run_dir:
            self._write_global_artifact(artifact)
        else:
            self._pending_global_artifact = artifact

    def _write_global_artifact(self, artifact):
        try:
            run_dir = self._run_dir
            states = artifact['states']
            waypoints = artifact['waypoints']
            meta = artifact['meta']
            with open(os.path.join(run_dir, 'global_plan.csv'), 'w', encoding='utf-8') as f:
                f.write('point_idx,x,y,theta\n')
                for i, s in enumerate(states):
                    th = float(s[2]) if states.shape[1] > 2 else float('nan')
                    f.write(f'{i},{float(s[0]):.6f},{float(s[1]):.6f},{th:.6f}\n')
            with open(os.path.join(run_dir, 'global_waypoints.csv'), 'w', encoding='utf-8') as f:
                f.write('wp_idx,x,y\n')
                for i, w in enumerate(waypoints):
                    f.write(f'{i},{float(w[0]):.6f},{float(w[1]):.6f}\n')
            with open(os.path.join(run_dir, 'global_plan_meta.json'), 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2)
            if 'canonical_route_json' in artifact:
                # No newline: the file bytes themselves have the registered hash.
                with open(
                    os.path.join(run_dir, 'preselected_route.json'),
                    'w',
                    encoding='utf-8',
                    newline='',
                ) as f:
                    f.write(artifact['canonical_route_json'])
                with open(
                    os.path.join(run_dir, 'preselected_route_provenance.json'),
                    'w',
                    encoding='utf-8',
                ) as f:
                    json.dump(
                        artifact['preselected_route_provenance'],
                        f,
                        indent=2,
                        sort_keys=True,
                        allow_nan=False,
                    )
            self._global_artifact_saved = True
            self._pending_global_artifact = None
            self.get_logger().info(
                f"[hierarchical] saved global route artifacts -> global_plan.csv "
                f"({meta['n_states']} states), global_waypoints.csv ({meta['n_waypoints']} wp), "
                f"global_plan_meta.json (chose {meta['selected_source']})"
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"failed to write global route artifact: {exc}")

    def _active_plan_remaining_s(self) -> float:
        with self._data_lock:
            controls = None if self._active_controls is None else self._active_controls
            started_at = self._active_plan_started_at
        if controls is None or controls.size == 0 or started_at is None:
            return 0.0
        elapsed_s = max((self.get_clock().now() - started_at).nanoseconds * 1e-9, 0.0)
        return max(float(controls.shape[0]) * max(float(self.dt), 1e-3) - elapsed_s, 0.0)

    def _capture_plan_request(self):
        """Freeze the cancellation context before any expensive planning work.

        Captured BEFORE the planning callback snapshots goal/belief or predicts,
        so the generation belongs to the request, not to whatever the world looked
        like once the solve finished.
        """
        with self._data_lock:
            return {
                'stop_generation': int(self._command_stop_generation),
                'started_at': self.get_clock().now(),
                'goal': self._execution_goal_identity(),
                'config': self._execution_config_identity(),
                'owner_thread': threading.get_ident(),
            }

    def _execution_goal_identity(self):
        goal = getattr(self, 'goal_msg', None)
        if goal is None:
            return None
        mission = getattr(self, '_mission_goal', None)
        return (goal.header.frame_id, float(goal.pose.position.x), float(goal.pose.position.y),
                None if mission is None else (mission.mission_epoch, mission.goal_id))

    def _goal_cb(self, msg):
        with self._data_lock:
            if getattr(self, '_mission_goal', None) is not None:
                return  # Once present, the atomic envelope owns goal identity.
            previous = self._execution_goal_identity()
            super()._goal_cb(msg)
            if previous is not None and previous != self._execution_goal_identity():
                self._publish_safe_stop_command()

    def _mission_goal_cb(self, msg):
        try:
            goal = mission_goal_from_json(msg.data)
        except ValueError as error:
            self._fatal_experiment_stop('Invalid mission goal envelope', error)
            return
        with self._data_lock:
            old = getattr(self, '_mission_goal', None)
            retired = getattr(self, '_retired_mission_epochs', set())
            if goal.mission_epoch in retired:
                return
            if old is not None and old.goal_id == goal.goal_id:
                if old.identity_payload != goal.identity_payload:
                    self._fatal_experiment_stop('Conflicting mission goal identity')
                    return
                if old.status != 'active' and goal.status == 'active':
                    return
            if old is not None and old.mission_epoch == goal.mission_epoch and goal.tour_index < old.tour_index:
                return
            if old is not None and old.mission_epoch != goal.mission_epoch:
                retired.add(old.mission_epoch)
            self._retired_mission_epochs = retired
            changed = old is None or old.goal_id != goal.goal_id or old.status != goal.status
            self._mission_goal = goal
            if goal.status != 'active':
                self.goal_msg = None
                if changed:
                    self._publish_safe_stop_command()
                return
            pose = PoseStamped()
            pose.header.frame_id = goal.frame_id
            pose.header.stamp.sec, pose.header.stamp.nanosec = goal.stamp_sec, goal.stamp_nanosec
            pose.pose.position.x, pose.pose.position.y = goal.x, goal.y
            super()._goal_cb(pose)
            if changed:
                self._publish_safe_stop_command()

    def _terminal_stop_request_cb(self, msg):
        """Latch the experiment-owned terminal stop before acknowledging it."""
        try:
            request = terminal_stop_request_from_json(msg.data)
        except ValueError as error:
            self._fatal_experiment_stop("Invalid terminal stop request", error)
            return
        with self._data_lock:
            if self._terminal_stop_requested:
                if request.request_id != self._terminal_stop_request_id:
                    self._fatal_experiment_stop("Conflicting terminal stop request identity")
                return
            self._terminal_stop_requested = True
            self._terminal_stop_request_id = request.request_id
            self._publish_safe_stop_command(
                f'terminal_stop:{getattr(request, "reason", "unspecified")}'
            )
            generation = int(self._command_stop_generation)
        ack = TerminalStopAck(
            request_id=request.request_id,
            component="planner",
            status="stop_command_published",
            acknowledgement_stamp_ns=int(self.get_clock().now().nanoseconds),
            detail=f"command_stop_generation={generation}",
        )
        message = String()
        message.data = terminal_stop_ack_to_json(ack)
        self._terminal_stop_ack_pub.publish(message)

    def _execution_config_identity(self):
        # Values that change route/control feasibility, not diagnostic timestamps.
        names = ('v_min', 'v_max', 'w_min', 'w_max', 'dt', 'global_dt',
                 'global_planner_mode', 'local_controller_type', 'driveable_geometry_json',
                 'robot_collision_radius_m', 'nogo_safe_distance', 'local_nogo_safe_distance',
                 'camera_network_artifact_path')
        identity = [(name, repr(getattr(self, name, None))) for name in names]
        planner_names = ('dt', 'horizon', 'v_min', 'v_max', 'w_min', 'w_max',
                         'robot_length_m', 'robot_width_m', 'robot_collision_radius_m',
                         'use_nogo_cost', 'use_belief_nogo_cost', 'nogo_belief_kappa',
                         'control_weight', 'risk_weight_obs', 'ambiguity_weight',
                         'use_visibility_model', 'use_hit_miss_mixture')
        for label in ('planner', 'global_planner'):
            planner = getattr(self, label, None)
            identity.append((label, id(planner)))
            identity.extend((label+'.'+name, repr(getattr(planner, name, None))) for name in planner_names)
            for name in ('collision_cost_model', 'nogo_cost_model', 'camera_network', 'visibility_model'):
                model = getattr(planner, name, None)
                identity.append((label+'.'+name, repr(getattr(model, 'signature', None))))
        return tuple(identity)

    def _remember_request_belief(self, meta):
        request = getattr(self, '_active_plan_request', None)
        if (request is not None and request.get('owner_thread') == threading.get_ident()
                and 'belief_origin' not in request):
            request['belief_origin'] = dict(meta or {})

    def _global_solver_result_safe(self, result, goal):
        request = self._active_plan_request
        origin = request.get('state_origin') if request is not None else None
        if origin is None:
            return False, 'missing_solver_origin'
        validator = getattr(self.global_planner, 'validate_result', None)
        if validator is not None:
            return validator(result, origin[0], origin[1], goal, require_complete=True)
        return validate_plan_result(
            result, initial_state=origin[0], initial_covariance=origin[1], goal_xy=goal,
            dt=float(getattr(self.global_planner, 'dt', getattr(self, 'global_dt', 1.))),
            control_bounds=((self.v_min, self.v_max), (self.w_min, self.w_max)),
            require_complete=True,
            terminal_tolerance_m=float(self.global_planner.optimizer_terminal_goal_tolerance_m))

    def _global_route_candidate_safe(self, points, goal, current_m, result=None):
        """Check a candidate route, including its fresh entry leg.

        A live solver result has not passed the frozen-artifact gate, so every
        segment remains subject to admission-time swept-body validation.  A
        preselected route is different: its exact coordinates, source artifact,
        endpoints and full geometry were hash-bound and revalidated at startup.
        Repeating hundreds of swept-body checks here used to hold ``_data_lock``
        long enough to discard depth-one odometry.  For that mode, verify that
        the candidate is still the startup-validated polyline and validate only
        the live connector plus the first frozen leg.  Every command is still
        checked immediately before execution.
        """
        try:
            points = np.asarray(points, dtype=float)
            if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
                return False, 'malformed_route_points'
            # Preselected-route mode intentionally constructs no global planner.
            # Use the tolerance that already passed route identity/geometry
            # validation, and use the local planner for the fresh entry-leg
            # geometry checks below.
            planner = self.global_planner if self.global_planner is not None else self.planner
            if self.global_planner_mode == 'preselected_route':
                tolerance = float((self._preselected_route_provenance or {}).get(
                    'endpoint_tolerance_m',
                    getattr(planner, 'optimizer_terminal_goal_tolerance_m', 0.),
                ))
            else:
                tolerance = float(getattr(planner, 'optimizer_terminal_goal_tolerance_m', 0.))
            # Canonical JSON round-tripping can move a decimal endpoint by a
            # few ulps (e.g. 10.6 -> 10.600000000000001). A declared zero
            # geometric tolerance still needs a numerical comparison floor.
            if np.linalg.norm(points[-1] - np.asarray(goal)) > tolerance + 1.0e-9:
                return False, 'incomplete_route'
            points_to_check = points
            if self.global_planner_mode == 'preselected_route':
                validated_points = np.asarray(
                    getattr(self, '_preselected_route_points', ()), dtype=float
                )
                if (validated_points.shape != points.shape
                        or not np.array_equal(validated_points, points)):
                    return False, 'preselected_route_identity_mismatch'
                # point 0 checks the live current-pose connector; point 1 also
                # checks the actual initial turn and first frozen route leg.
                points_to_check = points[:2]
                # A route start already within the follower's arrival radius is
                # reached on the first tick: the follower never turns to face it.
                # Checking that turn (toward a few-centimetre offset, so toward an
                # arbitrary heading) rejected every run from a start in a narrow aisle.
                arrival = float(getattr(self, 'waypoint_arrival_radius_m', 0.0))
                if np.linalg.norm(points[0] - np.asarray(current_m[:2], dtype=float)) <= arrival:
                    points_to_check = points[1:2]
            pose = np.asarray(current_m[:3], dtype=float).copy()
            for point in points_to_check:
                delta = point - pose[:2]
                yaw = pose[2] if np.linalg.norm(delta) < 1e-12 else math.atan2(delta[1], delta[0])
                rotated = np.array([pose[0], pose[1], yaw])
                end = np.array([point[0], point[1], yaw])
                for start, finish in ((pose, rotated), (rotated, end)):
                    for method in ('collision_sweep_clearance_np', 'driveable_sweep_clearance_np'):
                        checker = getattr(planner, method, None)
                        if checker is None:
                            return False, 'missing_route_geometry_validator'
                        clearance = float(checker(start, finish))
                        if math.isnan(clearance) or clearance < 0.:
                            return False, method
                pose = end
            return True, ''
        except (ValueError, TypeError, AttributeError, OverflowError):
            return False, 'invalid_route_geometry'

    def _resolve_belief_for_planning(self):
        result = super()._resolve_belief_for_planning()
        if result[0] is not None:
            self._remember_request_belief(result[2])
            request = getattr(self, '_active_plan_request', None)
            if (request is not None and request.get('owner_thread') == threading.get_ident()
                    and 'state_origin' not in request):
                request['state_origin'] = (result[0].copy(), result[1].copy())
        return result

    def _reject_current_request(self, request, reason):
        # A rejected LOCAL replan is not a command to revoke the control tape
        # that was already admitted and is still being checked before every
        # publication.  Erasing that tape here turned an ordinary failed
        # replacement into a permanent zero-command deadlock.  The command
        # timer remains the owner of execution-time collision and driveable-area
        # rejection; it will stop immediately if the retained next step is no
        # longer safe.
        retained = False
        with self._data_lock:
            if not self._plan_request_is_current(request):
                return False
            retain_active_local_tape = bool(
                self.use_hierarchical
                and self._hier_phase == 'LOCAL'
                and self._active_controls is not None
                and np.asarray(self._active_controls).size > 0
                and self._active_plan_started_at is not None
            )
            if retain_active_local_tape:
                retained = True
            else:
                self._publish_safe_stop_command()
        suffix = '; retaining previously admitted local control tape' if retained else ''
        self._warn_once_about_expired_tape(f'{reason}{suffix}')
        return False

    def _fresh_request_belief(self, request):
        m, S, meta = self._resolve_belief_for_planning()
        origin = request.get('belief_origin', {})
        if (m is None or S is None or not meta.get('belief_valid', False)
                or not meta.get('motion_supported', False)):
            return None
        if meta.get('belief_epoch') != origin.get('belief_epoch'):
            return None
        frame = meta.get('belief_frame_id')
        if meta.get('belief_epoch') is None or not frame or frame != origin.get('belief_frame_id'):
            return None
        goal = request.get('goal')
        if goal is not None and goal[0] and frame != goal[0]:
            return None
        if not np.isfinite(m).all() or not np.isfinite(S).all():
            return None
        return m, S, meta

    def _controls_within_execution_bounds(self, controls):
        return (controls.ndim == 2 and controls.shape[0] > 0 and controls.shape[1] == 2
                and np.isfinite(controls).all()
                and np.all(controls[:, 0] >= getattr(self, 'v_min', 0.))
                and np.all(controls[:, 0] <= getattr(self, 'v_max', .22))
                and np.all(controls[:, 1] >= getattr(self, 'w_min', -1.))
                and np.all(controls[:, 1] <= getattr(self, 'w_max', 1.)))

    def _execution_belief_is_current(self, meta):
        if getattr(self, 'use_diagnostic_odom_localization', False):
            return (
                getattr(self, 'diagnostic_odom_pose', None) is not None
                and meta.get('belief_epoch') == 'diagnostic_odom'
                and meta.get('belief_revision') == 0
                and meta.get('belief_frame_id') == self._resolve_plan_frame_id()
            )
        checker = getattr(self, '_belief_context_is_current', None)
        if checker is not None and getattr(self, '_belief_record', None) is not None:
            return checker(meta, require_revision=True)
        return (meta.get('belief_epoch') == getattr(self, '_belief_epoch', None)
                and meta.get('belief_revision') == getattr(self, '_belief_revision', None))

    def _install_global_route(self, points, final_goal, *, result=None):
        """Admit the actual tracked route against a supported current belief.

        GLOBAL controls are never a command tape: no LOCAL duration bound applies.
        A slow result must pass the same fresh geometry check as a fast result.
        Correction ownership prevents a camera commit between revalidation and
        installation; odometry/clock/goal ownership shares the final data lock.
        """
        request = getattr(self, '_active_plan_request', None)
        reason = 'missing_global_request'
        with getattr(self, '_correction_lock', nullcontext()):
            with self._data_lock:
                if request is None or not self._plan_request_is_current(request):
                    return False
                age = (self.get_clock().now() - request['started_at']).nanoseconds * 1e-9
                if not math.isfinite(age) or age < 0. or getattr(self, '_fatal_stop_triggered', False):
                    return self._reject_current_request(request, 'invalid_global_request_epoch')
                fresh = self._fresh_request_belief(request)
                if fresh is None:
                    return self._reject_current_request(request, 'unsupported_global_belief')
                current_m, _, fresh_meta = fresh
                validation_started = time.perf_counter()
                safe, reason = self._global_route_candidate_safe(points, final_goal, current_m, result)
                validation_ms = (time.perf_counter() - validation_started) * 1000.0
                self.get_logger().info(
                    f"[hierarchical] route admission geometry took {validation_ms:.1f} ms "
                    f"(mode={self.global_planner_mode}, points={len(points)}, "
                    f"checked={min(len(points), 2) if self.global_planner_mode == 'preselected_route' else len(points)})"
                )
                if (not self._plan_request_is_current(request)
                        or not self._execution_belief_is_current(fresh_meta)):
                    return False
                if not safe:
                    return self._reject_current_request(request, 'Rejected global route: ' + reason)
                self._waypoints = [(float(p[0]), float(p[1])) for p in points]
                self._wp_idx = 0
                self._hier_phase = 'LOCAL'
                self._global_goal_xy = np.asarray(final_goal, dtype=float).copy()
                self._global_solve_done = True
                self._installed_route_request = dict(request)
                # Keep the first LOCAL handoff observable in campaign logs.  This
                # is intentionally one-shot and has no control-flow effect; it
                # distinguishes executor starvation from controller/safety work.
                self._first_local_handoff_pending = True
                # The warm start belongs to the route only after admission.
                try:
                    seed = self.planner._controls_for_waypoints(current_m[:3], [np.asarray(self._waypoints[0])])
                    self.planner.prev_controls_flat = np.asarray(seed, dtype=float).reshape(-1)
                except Exception:
                    self.planner.prev_controls_flat = None
                return True

    def _install_control_tape(self, controls, *, original_len, started_at=None,
                              latency_skip_steps=0, latency_skip_s=0.0,
                              log_prefix='Rejected local control tape at install'):
        """The single ownership boundary both execution routes install through.

        Generation comparison, expiry, tape replacement and the immediate
        publication are one transaction. Keeping two copies of this let the LOCAL
        and direct paths drift apart, which is how the computation-age gate ended
        up assigned on one route and read on neither.

        Returns 'cancelled', 'expired' or 'installed'.
        """
        controls = np.asarray(controls, dtype=float)
        request = getattr(self, '_active_plan_request', None)
        with getattr(self, '_correction_lock', nullcontext()), self._data_lock:
            # Generation FIRST: a cancelled result must not reach the stop below,
            # which would erase a replacement tape another owner already installed.
            if not self._plan_request_is_current(request):
                return 'cancelled'
            if not self._controls_within_execution_bounds(controls):
                self._publish_safe_stop_command()
                return 'invalid'
            if request is not None and 'belief_origin' in request:
                fresh = self._fresh_request_belief(request)
                if fresh is None:
                    self._publish_safe_stop_command()
                    return 'invalid_belief'
                incremental_local_guard = (
                    self.use_hierarchical and self._hier_phase == 'LOCAL'
                )
                # The admitted global route owns geometry for hierarchical LOCAL
                # tracking.  Rechecking geometry from a drifting belief turned
                # localization error in camera-blackout regions into a permanent
                # zero-command deadlock, even though the frozen route itself was
                # collision-free.  Direct/non-hierarchical execution still needs
                # full tape admission here.
                safety = (ControlSafetyResult(controls.shape[0]) if incremental_local_guard
                          else self._simple_plan_safe_to_execute(controls, fresh[0]))
                if (not self._plan_request_is_current(request)
                        or not self._execution_belief_is_current(fresh[2])):
                    return 'cancelled'
                if safety.safe_steps <= 0:
                    self._publish_safe_stop_command()
                    return 'unsafe'
                if not incremental_local_guard:
                    controls = controls[:safety.safe_steps]
            # The age is read here, under the install lock, so safety validation
            # and scheduling time count against the tape that actually installs.
            expired, why = self._plan_request_expired(request, controls.shape[0])
            if not expired:
                self._active_controls = controls.copy()
                self._active_plan_started_at = (
                    self.get_clock().now() if started_at is None else started_at)
                self._active_controls_original_len = int(original_len)
                self._last_latency_skip_steps = int(latency_skip_steps)
                self._last_latency_skip_s = float(latency_skip_s)
                self._active_tape_request = None if request is None else dict(request)
                if controls.size > 0:
                    self._publish_command(float(controls[0, 0]), float(controls[0, 1]))
                return 'installed'
            # Consume the stop in the same transaction as the expiry decision.
            self._publish_safe_stop_command()
        self._warn_once_about_expired_tape(f'{log_prefix}: {why}')
        return 'expired'

    def _warn_once_about_expired_tape(self, message: str) -> None:
        """Log outside the install lock, and never let logging break a safe stop."""
        try:
            self.get_logger().warn(message)
        except Exception:
            pass

    def _plan_request_expired(self, request, control_count) -> tuple:
        """Whether the computation took longer than the tape it produced covers.

        The existing rule -- ``age > steps * max(dt, 1e-3)`` -- with the strict
        ``>`` preserved. It is evaluated on the tape that will actually install,
        and independently of whether latency compensation is enabled: elapsed
        time is not evidence of executed motion, so this must not silently
        activate that separate path.

        A backward clock jump during computation rejects the request even when no
        old tape existed for the timer to invalidate. That is deliberate, and it
        is not complete filter-epoch reset handling.
        """
        if request is None:
            # Missing execution context must not silently disable validation.
            return True, 'missing_request_context'
        started_at = request.get('started_at')
        if started_at is None:
            return True, 'missing_request_start_time'
        age_s = (self.get_clock().now() - started_at).nanoseconds * 1e-9
        if not math.isfinite(age_s):
            return True, 'nonfinite_computation_age'
        if age_s < 0.0:
            return True, f'negative_computation_age:{age_s:.3f}'
        tape_duration_s = float(int(control_count)) * max(float(self.dt), 1e-3)
        if age_s > tape_duration_s:
            return True, f'stale_control_tape:{age_s:.3f}>{tape_duration_s:.3f}'
        return False, ''

    def _plan_request_is_current(self, request) -> bool:
        """Whether this result may still install. Caller must hold ``_data_lock``.

        A request captured after an ordinary stop is genuinely new and may resume;
        only work that began before the stop is cancelled.
        """
        if getattr(self, "_terminal_stop_requested", False):
            return False
        if request is None:
            return True
        origin = request.get('belief_origin')
        if origin is not None and hasattr(self, '_belief_epoch'):
            if origin.get('belief_epoch') != self._belief_epoch:
                return False
        return (int(request['stop_generation']) == int(self._command_stop_generation)
                and request.get('goal') == self._execution_goal_identity()
                and request.get('config', self._execution_config_identity()) == self._execution_config_identity())

    def _plan_once(self):
        """Wrap both execution routes in one cancellation context.

        Ordinary safe stops previously cleared the current tape but had no
        generation counter, so a solve that began before the stop could still
        install afterwards and drive the robot away from a commanded stop.
        """
        if getattr(self, "_terminal_stop_requested", False):
            return
        request = self._capture_plan_request()
        with self._data_lock:
            self._active_plan_request = request
        try:
            return self._plan_once_impl()
        except Exception:
            self._reject_current_request(request, 'Planning callback failed')
            raise
        finally:
            with self._data_lock:
                if self._active_plan_request is request:
                    self._active_plan_request = None

    def _plan_once_impl(self):
        if not self.use_hierarchical:
            return super()._plan_once()

        inputs = self._snapshot_plan_inputs()
        goal_ref = inputs['goal']
        if goal_ref is None:
            return
        now_wall = time.monotonic()
        m0, S0, belief_meta = self._resolve_belief_for_planning()
        if m0 is None or S0 is None:
            return
        self._remember_request_belief(belief_meta)
        if self._active_plan_request is not None:
            self._active_plan_request['state_origin'] = (m0.copy(), S0.copy())
        final_goal = self._goal_xy_from_msg(goal_ref)

        route_request = getattr(self, '_installed_route_request', None)
        if self._hier_phase == 'LOCAL' and route_request is not None:
            if (route_request.get('goal') != self._execution_goal_identity()
                    or route_request.get('config') != self._execution_config_identity()
                    or route_request['belief_origin'].get('belief_epoch') != belief_meta.get('belief_epoch')):
                self._hier_phase = 'GLOBAL'
                self._publish_safe_stop_command()
                return  # The next callback captures the new cancellation context.

        # Multi-goal: if the mission goal has advanced (a new waypoint published to
        # /goal_bev), re-enter the GLOBAL phase so a fresh route is planned to the
        # new goal. The frozen route only reaches the previous waypoint, so without
        # this the local tracker runs out of waypoints and the robot stalls.
        if (self._hier_phase == 'LOCAL'
                and self._global_goal_xy is not None
                and float(np.linalg.norm(final_goal - self._global_goal_xy))
                > 1.0e-9):
            if self.global_planner_mode == 'preselected_route':
                self._fatal_experiment_stop(
                    "Mission goal changed during preselected-route execution; "
                    "the frozen contract permits exactly one start-to-goal polyline"
                )
                return
            self.get_logger().info(
                f"[hierarchical] mission goal advanced "
                f"({self._global_goal_xy[0]:.2f},{self._global_goal_xy[1]:.2f}) -> "
                f"({final_goal[0]:.2f},{final_goal[1]:.2f}); replanning global route"
            )
            self._hier_phase = 'GLOBAL'

        if self._hier_phase == 'GLOBAL':
            if self.global_planner_mode == 'preselected_route':
                if not self._preselected_route_points:
                    self._fatal_experiment_stop(
                        "Preselected route was not available after startup validation"
                    )
                    return
                registered_goal = np.asarray(
                    self._preselected_route_provenance['registered_goal_xy'], dtype=float
                )
                if float(np.linalg.norm(final_goal - registered_goal)) > 1.0e-6:
                    self._fatal_experiment_stop(
                        "Mission goal does not equal the task goal used by the "
                        "preselected-route launch gate"
                    )
                    return

                # Preserve every selected coordinate exactly. In particular, do
                # not densify, smooth, truncate, or append the live belief/goal.
                candidate = [(float(point[0]), float(point[1]))
                             for point in self._preselected_route_points]
                if not self._install_global_route(candidate, final_goal):
                    return

                route_states = _polyline_states(
                    self._waypoints, initial_yaw=float(m0[2])
                )
                route_result = SimpleNamespace(states=route_states)
                self.path_pub.publish(
                    self._build_path_message(
                        route_result, final_goal, append_goal=False
                    )
                )
                self._save_preselected_route_artifacts(route_states, final_goal)
                self.get_logger().info(
                    f"[preselected_route] accepted {len(self._waypoints)} exact "
                    f"waypoints, sha256={self.preselected_route_sha256}; "
                    "switching to belief-based local tracking without a global solve"
                )
                return
            if str(getattr(self, 'global_planner_mode', 'efe')) == 'geometric_shortest_path':
                # C0 conventional-navigation baseline: pick the shortest-time valid
                # lane-graph route over the SAME driveable + no-go geometry as
                # C1/C2 and hand it to the SAME local tracker. No GP/visibility
                # input and no EFE solve -- the one-shot global optimisation is
                # skipped entirely. Route seeds come from the identical
                # generate_route_seeds call used by the EFE branch below.
                def _polyline_len(waypoints) -> float:
                    return _route_length_from(m0[:2], waypoints)

                def _route_time(waypoints) -> float:
                    return _geometric_route_time_cost(
                        m0,
                        waypoints,
                        v_max=float(self.v_max),
                    )

                seeds = []
                if str(getattr(self, 'driveable_geometry_json', '') or ''):
                    try:
                        from unav_common.lane_graph_routes import (
                            generate_route_seeds,
                            repair_route_seeds_for_footprint,
                        )
                        seed_geometry = (
                            self.collision_geometry_json
                            if str(getattr(self, 'collision_geometry_json', '') or '').strip()
                            else self.driveable_geometry_json
                        )
                        seeds = generate_route_seeds(
                            seed_geometry,
                            (float(m0[0]), float(m0[1])),
                            (float(final_goal[0]), float(final_goal[1])),
                        )
                        if str(getattr(self, 'collision_geometry_json', '') or '').strip():
                            seeds = repair_route_seeds_for_footprint(
                                seeds,
                                self.collision_geometry_json,
                                self.driveable_geometry_json,
                                m0,
                                robot_length_m=float(self.robot_length_m),
                                robot_width_m=float(self.robot_width_m),
                                target_clearance_m=0.02,
                            )
                    except Exception as exc:  # noqa: BLE001
                        self.get_logger().warn(
                            f"[geometric_shortest_path] lane-graph seed generation failed "
                            f"({exc}); falling back to straight start->goal route"
                        )
                        seeds = []
                if not seeds:
                    self._reject_current_request(self._active_plan_request, 'No validated geometric route')
                    return
                best = min(seeds, key=lambda seed: _route_time(seed['waypoints']))
                candidate = [(float(w[0]), float(w[1])) for w in best['waypoints']]
                if not self._install_global_route(candidate, final_goal):
                    return
                # C0 has no optimizer result, but consumers (including the live
                # dashboard) still need the route the planner actually chose.
                route_result = SimpleNamespace(
                    states=_route_states(m0[:3], self._waypoints)
                )
                self.path_pub.publish(
                    self._build_path_message(
                        route_result, final_goal, append_goal=False
                    )
                )
                return
            # Generate condition-neutral lane-graph route seeds from the driveable
            # map for this (one-shot) global solve. The global route is chosen once
            # and never replanned, so these seeds provide the nonconvex optimizer's
            # route-basin coverage. Generated from geometry + actual start + goal;
            # identical across conditions; no GP/visibility input.
            is_replan = self._global_solve_done
            fresh_seeds = False
            if (str(getattr(self, 'optimizer_route_seed_mode', 'explicit')) == 'lane_graph'
                    and str(getattr(self, 'driveable_geometry_json', '') or '')):
                try:
                    from unav_common.lane_graph_routes import (
                        generate_route_seeds,
                        repair_route_seeds_for_footprint,
                    )
                    seed_geometry = (
                        self.collision_geometry_json
                        if str(getattr(self, 'collision_geometry_json', '') or '').strip()
                        else self.driveable_geometry_json
                    )
                    seeds = generate_route_seeds(
                        seed_geometry,
                        (float(m0[0]), float(m0[1])),
                        (float(final_goal[0]), float(final_goal[1])),
                    )
                    if str(getattr(self, 'collision_geometry_json', '') or '').strip():
                        seeds = repair_route_seeds_for_footprint(
                            seeds,
                            self.collision_geometry_json,
                            self.driveable_geometry_json,
                            m0,
                            robot_length_m=float(self.robot_length_m),
                            robot_width_m=float(self.robot_width_m),
                            target_clearance_m=0.02,
                        )
                    if seeds:
                        self.global_planner.optimizer_initial_routes = (
                            self.global_planner._parse_initial_routes(json.dumps(seeds))
                        )
                        fresh_seeds = True
                        self.get_logger().info(
                            f"[hierarchical] lane-graph route seeds: "
                            f"{[s['name'] for s in seeds]}"
                        )
                    else:
                        self.get_logger().warn(
                            "lane-graph generated 0 seeds; keeping explicit "
                            "optimizer_initial_routes (check driveable_geometry_json covers start/goal)"
                        )
                except Exception as exc:  # noqa: BLE001
                    self.get_logger().warn(f"lane-graph seed generation failed ({exc}); using explicit seeds")
            # Multi-goal replan: the explicit optimizer_initial_routes are anchored to
            # the ORIGINAL start->goal, so reusing them for a new leg initialises the
            # EFE optimiser with a geometrically-wrong trajectory (covariance can go
            # non-PD -> LinAlgError). When no fresh lane-graph seeds were produced,
            # seed the replan with a direct route from the current belief to the new
            # goal so the optimiser starts from a sane, leg-appropriate trajectory.
            if is_replan and not fresh_seeds:
                direct = [{'name': 'replan_direct',
                           'waypoints': [(float(m0[0]), float(m0[1])),
                                         (float(final_goal[0]), float(final_goal[1]))]}]
                self.global_planner.optimizer_initial_routes = (
                    self.global_planner._parse_initial_routes(direct))
                self.global_planner.prev_controls_flat = None
                self.get_logger().info(
                    "[hierarchical] replan seeded with direct route "
                    f"({m0[0]:.2f},{m0[1]:.2f})->({final_goal[0]:.2f},{final_goal[1]:.2f})")
            plan_start = time.perf_counter()
            try:
                rg = self.global_planner.plan(m0, S0, final_goal)
            except Exception as exc:  # noqa: BLE001
                # An exception is not authority to install an unchecked straight route.
                self._reject_current_request(self._active_plan_request,
                                             f'Global planner raised: {type(exc).__name__}: {exc}')
                return
            # Validate solver evidence before waypoint extraction can hide malformed
            # or incomplete results. No unchecked terminal connector is appended.
            safe, reason = self._global_solver_result_safe(rg, final_goal)
            if not safe:
                self._reject_current_request(self._active_plan_request, 'Rejected global result: ' + reason)
                return
            candidate = extract_waypoints(
                rg.states, spacing_m=self.waypoint_spacing_m, include_goal=True)
            if not self._install_global_route(candidate, final_goal, result=rg):
                return
            self.get_logger().info(
                f"[hierarchical] global plan solved in {(time.perf_counter()-plan_start):.1f}s "
                f"(backend={getattr(rg, 'backend', '?')}, "
                f"nit={getattr(rg, 'optimizer_nit', 0)}, nfev={getattr(rg, 'optimizer_nfev', 0)}, "
                f"solve={getattr(rg, 'solve_time_s', 0.0):.1f}s) -> "
                f"{len(self._waypoints)} waypoints; switching to local tracking"
            )
            # Publish the global plan for visualization; do NOT follow it.
            self._publish_plan_and_metrics(rg, final_goal, m0, S0, belief_meta=belief_meta)
            # Persist the solved global route (plan states + waypoints + winning
            # seed + costs) to the run directory -- this is the one-shot route choice.
            self._save_global_plan_artifacts(rg, m0, final_goal)
            return

        # LOCAL phase: track the current planner-derived waypoint.
        trace_first_handoff = bool(
            getattr(self, '_first_local_handoff_pending', False)
        )
        if trace_first_handoff:
            self.get_logger().info('[hierarchical] first local callback entered')
        if not self._waypoints:
            return
        target = np.asarray(self._waypoints[self._wp_idx], dtype=float)
        prev_wp_idx = self._wp_idx
        while (
            self._wp_idx < len(self._waypoints) - 1
            and _waypoint_reached_or_passed(
                self._waypoints,
                self._wp_idx,
                m0[:2],
                arrival_radius_m=self.waypoint_arrival_radius_m,
            )
        ):
            self._wp_idx += 1
            target = np.asarray(self._waypoints[self._wp_idx], dtype=float)
        if self._wp_idx != prev_wp_idx:
            try:
                seed = self.planner._controls_for_waypoints(m0[:3], [target])
                self.planner.prev_controls_flat = np.asarray(seed, dtype=float).reshape(-1)
            except Exception:
                pass
        elif self.local_replan_min_remaining_s > 0.0:
            remaining_s = self._active_plan_remaining_s()
            if remaining_s > self.local_replan_min_remaining_s:
                return
        elif (
            self.local_replan_on_waypoint_change
            and self._last_local_plan_target is not None
            and np.allclose(target, self._last_local_plan_target, atol=1e-6)
            and self._active_plan_remaining_s() > 0.0
        ):
            return

        plan_start = time.perf_counter()
        self._pending_plan_started_at = self.get_clock().now()
        self._pending_plan_started_active_remaining_s = self._active_plan_remaining_s()
        self._last_local_plan_target = target.copy()

        m_track = m0.copy()
        S_track = S0.copy()
        tracking_yaw_source = 0.0

        dx = float(target[0] - m_track[0])
        dy = float(target[1] - m_track[1])
        desired_yaw = math.atan2(dy, dx)
        yaw_error = wrap_angle(desired_yaw - float(m_track[2]))
        self._current_wp_idx = float(self._wp_idx)
        self._current_wp_count = float(len(self._waypoints))
        self._current_wp_target = target.copy()
        self._current_wp_dist = float(math.hypot(dx, dy))
        self._current_desired_yaw = float(desired_yaw)
        self._current_yaw_error = float(yaw_error)
        self._current_tracking_yaw = float(m_track[2])
        self._current_tracking_yaw_source = float(tracking_yaw_source)

        controller_started = time.perf_counter()
        proposed_controls = self._dispatch_local_controller(m_track, target)
        controller_ms = (time.perf_counter() - controller_started) * 1000.0
        if trace_first_handoff:
            self.get_logger().info(
                f'[hierarchical] first local controls generated in {controller_ms:.3f} ms'
            )
        # The global route has already passed the complete swept-footprint and
        # driveable-area admission check.  Re-applying those tests to a local
        # command from the corrected runtime belief makes localization error a
        # second, inconsistent route veto: a normal correction can then erase a
        # healthy tape although the admitted physical route remains unchanged.
        # LOCAL therefore validates only the command representation and bounds;
        # collision evidence remains monitored independently by the experiment.
        controls = proposed_controls
        if trace_first_handoff:
            self.get_logger().info('[hierarchical] first local tape install entered')
        install_status = self._install_control_tape(
            controls,
            original_len=int(controls.shape[0]),
            log_prefix='[hierarchical] local control tape expired before install',
        )
        if trace_first_handoff:
            self.get_logger().info(
                f'[hierarchical] first local tape install finished: {install_status}'
            )
            self._first_local_handoff_pending = False
        return

    def _simple_local_plan(self, m0: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Proportional geometric controller — returns (local_horizon, 2) [v, w] array."""
        H = int(self.local_horizon)
        dt = float(self.dt)
        v_max = float(self.v_max)
        w_min, w_max = float(self.w_min), float(self.w_max)

        # Rotate-in-place when badly misaligned, then translate. Without this
        # gate the exp(-|yaw_err|) taper still gives ~0.13 m/s at 90 deg, so the
        # robot creeps forward through a large departure turn and can clip an
        # adjacent obstacle (e.g. C1 swinging into R5L on the initial east->north
        # turn). A standard differential-drive turn-then-go controller.
        yaw_gate = float(self.simple_tracker_yaw_gate_rad)

        controls = np.zeros((H, 2), dtype=float)
        state = m0[:3].copy().astype(float)
        tx, ty = float(target[0]), float(target[1])

        for i in range(H):
            dx, dy = tx - state[0], ty - state[1]
            dist = math.hypot(dx, dy)
            if dist < 0.05:
                break
            desired_yaw = math.atan2(dy, dx)
            yaw_err = wrap_angle(desired_yaw - state[2])
            w = float(np.clip(2.0 * yaw_err, w_min, w_max))
            if abs(yaw_err) > yaw_gate:
                v = 0.0  # rotate in place until aligned
            else:
                # Never travel farther than the active waypoint in one command.
                # At 1 m/s the former 0.25 m step was longer than the frozen
                # route's 0.20 m spacing and cut corners between waypoints.
                v = float(min(
                    v_max * math.exp(-abs(yaw_err)),
                    dist / max(dt, 1.0e-9),
                ))
            controls[i] = [v, w]
            state = unicycle_step(state, [v, w], dt)

        return controls

    def _dispatch_local_controller(self, m0: np.ndarray, target: np.ndarray) -> np.ndarray:
        ct = local_controller_type(getattr(self, 'local_controller_type', 'ff_fb'))
        if ct == 'hyst_damp':
            return self._hyst_damp_plan(m0, target)
        if ct == 'pure_pursuit':
            return self._pure_pursuit_plan(m0)
        if ct == 'ff_fb':
            return self._ff_fb_plan(m0)
        if ct in ('turn_then_go', 'turn_then_go_recovery'):
            return self._simple_local_plan(m0, target)
        raise AssertionError(f"unhandled local controller {ct!r}")

    def _waypoint_array(self, state_xy=None) -> np.ndarray | None:
        if state_xy is None:
            state_xy = self._waypoints[self._wp_idx] if self._waypoints else (0.0, 0.0)
        return _tracking_waypoints(self._waypoints, self._wp_idx, state_xy)

    def _hyst_damp_plan(self, m0: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Turn-then-go + hysteresis on the spin gate, rate-limited (damped) w, and a
        small forward creep instead of a full stop -- kills the sharp-turn limit-cycle."""
        H = int(self.local_horizon); dt = float(self.dt); v_max = float(self.v_max)
        gate = float(self.simple_tracker_yaw_gate_rad)
        controls = np.zeros((H, 2), dtype=float)
        state = m0[:3].copy().astype(float)
        tx, ty = float(target[0]), float(target[1])
        spin = bool(self._ctrl_spin); w_prev = float(self._ctrl_prev_w)
        for i in range(H):
            dx, dy = tx - state[0], ty - state[1]
            if math.hypot(dx, dy) < 0.05:
                break
            yaw_err = wrap_angle(math.atan2(dy, dx) - state[2])
            # hysteresis: enter spin above the gate, only leave below 0.30 rad
            spin = (abs(yaw_err) > gate) or (spin and abs(yaw_err) > 0.30)
            w_des = float(np.clip(1.0 * yaw_err, -1.0, 1.0))
            w = w_prev + float(np.clip(w_des - w_prev, -0.15, 0.15))  # damp / rate-limit
            v = 0.08 if spin else float(v_max * math.exp(-abs(yaw_err)))
            controls[i] = [v, w]
            state = unicycle_step(state, [v, w], dt)
            w_prev = w
            if i == 0:
                self._ctrl_spin = spin; self._ctrl_prev_w = w
        return controls

    def _pure_pursuit_plan(self, m0: np.ndarray) -> np.ndarray:
        """Lookahead path tracker over the global waypoint polyline (always moving)."""
        H = int(self.local_horizon); dt = float(self.dt); v_max = float(self.v_max)
        wps = self._waypoint_array(m0[:2])
        controls = np.zeros((H, 2), dtype=float)
        if wps is None:
            return controls
        Ld = max(3.0 * float(self.waypoint_spacing_m), 0.30)
        state = m0[:3].copy().astype(float)
        for i in range(H):
            j = int(np.argmin(np.hypot(wps[:, 0] - state[0], wps[:, 1] - state[1])))
            while j < len(wps) - 1 and np.hypot(*(wps[j] - state[:2])) < Ld:
                j += 1
            dx, dy = wps[j] - state[:2]
            if np.hypot(dx, dy) < 0.05:
                break
            alpha = wrap_angle(math.atan2(dy, dx) - state[2])
            L = max(float(np.hypot(dx, dy)), 1e-3)
            w = float(np.clip(
                2.0 * v_max * math.sin(alpha) / L,
                float(self.w_min),
                float(self.w_max),
            ))
            v = float(v_max * max(0.2, 1.0 - abs(alpha) / 1.2))
            controls[i] = [v, w]
            state = unicycle_step(state, [v, w], dt)
        return controls

    def _ff_fb_plan(self, m0: np.ndarray) -> np.ndarray:
        """Path feedback with straight-line speed and corner-aware braking."""
        H = int(self.local_horizon); dt = float(self.dt); v_max = float(self.v_max)
        wps = self._waypoint_array(m0[:2])
        controls = np.zeros((H, 2), dtype=float)
        if wps is None or len(wps) < 2:
            return controls
        # The frozen route is sampled every 0.2 m for collision validation.
        # Those samples are not control objectives. Track the equivalent
        # corner polyline so the controller cannot dominate the comparison by
        # repeatedly point-capturing dense samples.
        wps = _compress_collinear_waypoints(wps)
        state = m0[:3].copy().astype(float)
        j = 0
        for i in range(H):
            # Advance the simulated tracker after reaching or crossing a
            # corner. This prevents later controls in the short rollout from
            # turning back toward a waypoint which the rollout already passed.
            while j < len(wps) - 2:
                candidate = wps[j + 1] - wps[j]
                candidate_len = float(np.hypot(*candidate))
                if candidate_len < 1.0e-4:
                    j += 1
                    continue
                candidate_dist = float(np.linalg.norm(wps[j + 1] - state[:2]))
                candidate_along = float(
                    (state[:2] - wps[j]) @ (candidate / candidate_len)
                )
                if candidate_dist < 0.05 or candidate_along >= candidate_len:
                    j += 1
                    continue
                break
            j2 = min(j + 1, len(wps) - 1)
            seg = wps[j2] - wps[j]; seglen = float(np.hypot(*seg))
            if seglen < 1e-4:
                break
            to_target = wps[j2] - state[:2]
            target_dist = float(np.hypot(*to_target))
            final_segment = j2 == len(wps) - 1
            if final_segment and target_dist < 0.05:
                break
            along = float((state[:2] - wps[j]) @ (seg / seglen))
            capture_target = final_segment
            if capture_target:
                # Point capture is reserved for the mission endpoint. Ordinary
                # route corners are followed as continuous fillets below.
                tang = math.atan2(to_target[1], to_target[0])
                ct = 0.0
            else:
                tang = math.atan2(seg[1], seg[0])
                nh = np.array([-math.sin(tang), math.cos(tang)])
                ct = float((state[:2] - wps[j]) @ nh)    # + = left of path
                # Blend into the outgoing tangent before reaching a retained
                # polyline corner. This produces a bounded arc instead of
                # driving to the vertex and then trying to pivot in place.
                fillet_preview_m = max(0.60, 3.0 * float(self.waypoint_spacing_m))
                if j2 < len(wps) - 1 and target_dist < fillet_preview_m:
                    outgoing = wps[j2 + 1] - wps[j2]
                    if float(np.hypot(*outgoing)) > 1.0e-4:
                        outgoing_yaw = math.atan2(outgoing[1], outgoing[0])
                        blend = float(np.clip(
                            1.0 - target_dist / fillet_preview_m, 0.0, 1.0,
                        ))
                        delta = wrap_angle(outgoing_yaw - tang)
                        tang = wrap_angle(tang + blend * delta)
                        # Do not let the incoming-line cross-track term fight
                        # the deliberate departure onto the corner fillet.
                        ct *= 1.0 - blend
            w_limit = float(getattr(self, 'ff_fb_turn_rate_limit_rad_s', 0.80))
            he, w, cross_track_cap = _ff_fb_path_guidance(
                tang,
                state[2],
                ct,
                v_max=v_max,
                w_limit=w_limit,
            )
            # Fast wheel-counterrotation makes Gazebo's wheel odometry finish a
            # pivot before the physical body, which is especially damaging when
            # heading is intentionally odometry-only. Stay below the observed
            # traction-safe turn rate while preserving 1 m/s on straights.
            nominal_v = float(
                np.clip(
                    v_max * max(0.25, 1.0 - 1.5 * abs(ct)),
                    0.05,
                    v_max,
                )
            )
            corner_cap = _preview_corner_speed_limit(
                wps,
                j,
                state[:2],
                v_max=v_max,
                preview_m=max(0.90, 6.0 * float(self.waypoint_spacing_m)),
                corner_speed_mps=min(0.30, 0.40 * v_max),
            )
            arrival_cap = _ff_fb_arrival_speed_cap(
                target_dist,
                must_capture=capture_target,
                v_max=v_max,
            )
            # For unicycle motion lateral acceleration is v*|w|. This keeps a
            # high straight-line ceiling without entering tight turns at that
            # same speed. Large departure turns pivot in place, otherwise the
            # minimum forward speed can create a waypoint-orbit limit cycle.
            v = _ff_fb_forward_speed(
                nominal_v,
                min(corner_cap, arrival_cap, cross_track_cap,
                    _ff_fb_turn_overshoot_cap(he, w_limit=w_limit)),
                w,
                he,
                v_max=v_max,
                yaw_gate_rad=float(self.simple_tracker_yaw_gate_rad),
                crawl_speed_mps=float(getattr(self, 'ff_fb_corner_crawl_speed_mps', 0.18)),
                pivot_heading_error_rad=float(getattr(
                    self, 'ff_fb_pivot_heading_error_rad', 2.60,
                )),
            )
            controls[i] = [v, w]
            state = unicycle_step(state, [v, w], dt)
            if np.hypot(*(wps[-1] - state[:2])) < 0.05:
                break
        return controls

    def _simple_plan_safe_to_execute(self, controls: np.ndarray, m0: np.ndarray) -> ControlSafetyResult:
        """Accept only prefixes whose oriented body stays clear throughout motion.

        Physical overlap or an uncertified swept interval refuses the step.
        Stationary rotation recovery must pass the same footprint check.
        """
        controls = np.asarray(controls, dtype=float)
        if controls.ndim != 2 or controls.shape[0] == 0 or controls.shape[1] != 2:
            return ControlSafetyResult(0, 'empty_or_malformed_controls', SafetyFailure.INVALID_INPUT)
        if not np.all(np.isfinite(controls)):
            return ControlSafetyResult(0, 'nonfinite_controls', SafetyFailure.INVALID_INPUT)

        start = np.asarray(m0[:3], dtype=float)
        if start.shape != (3,) or not np.isfinite(start).all() or not math.isfinite(self.dt) or self.dt <= 0.0:
            return ControlSafetyResult(0, 'invalid_state_or_timestep', SafetyFailure.INVALID_INPUT)
        if self.planner.collision_cost_model is not None:
            start_coll = self.planner.collision_clearance_state_np(start)
        else:
            start_coll = float('inf')
        nogo = self.planner.nogo_cost_model
        collision_required_margin = max(
            float(getattr(self.planner, 'nogo_safe_distance', 0.0))
            - float(getattr(self.planner, 'robot_collision_radius_m', 0.0)),
            0.0,
        )
        # The driveable union is a routing/support mask, not collision
        # geometry.  Runtime tracking must remain inside it, while the declared
        # body buffer applies to physical obstacles.  Applying the physical
        # margin to both double-counts clearance at narrow mapped corners and
        # can stop a physically safe robot that has not left the lane.
        driveable_required_margin = 0.0
        if nogo is not None:
            start_nogo = self.planner.driveable_clearance_state_np(start)
        else:
            start_nogo = float('inf')
        # +inf represents an unconstrained/empty scene. NaN and -inf cannot
        # establish a safe prefix and must not silently disable the gate.
        if any(math.isnan(clearance) or clearance == -math.inf for clearance in (start_coll, start_nogo)):
            return ControlSafetyResult(0, 'invalid_initial_clearance', SafetyFailure.INVALID_GEOMETRY)
        if start_coll < collision_required_margin:
            return ControlSafetyResult(
                0, f'initial_collision_clearance_below_margin:{start_coll:.3f}',
                SafetyFailure.COLLISION)
        if start_nogo < driveable_required_margin:
            return ControlSafetyResult(
                0, f'initial_driveable_clearance_below_margin:{start_nogo:.3f}',
                SafetyFailure.DRIVEABLE_CLEARANCE)
        from unav_common.rectangular_footprint import constant_twist_pose
        state = start.copy()
        for i, u in enumerate(controls):
            previous_state = state.copy()
            state = unicycle_step(state, u, float(self.dt))
            if not np.isfinite(state).all():
                return ControlSafetyResult(i, 'nonfinite_predicted_state', SafetyFailure.INVALID_INPUT)
            if self.planner.collision_cost_model is not None:
                clearance = self.planner.collision_sweep_clearance_np(
                    previous_state, state, yaw_delta=float(u[1])*self.dt, control=u,
                    dt=self.dt, required_clearance=collision_required_margin)
                if math.isnan(clearance) or clearance == -math.inf:
                    return ControlSafetyResult(i, 'invalid_collision_clearance', SafetyFailure.INVALID_GEOMETRY)
                if clearance < collision_required_margin:
                    return ControlSafetyResult(i, f'collision_geometry_violation_step_{i}:{clearance:.3f}',
                                               SafetyFailure.COLLISION)
            if nogo is not None:
                clearance = self.planner.driveable_sweep_clearance_np(
                    previous_state, state, yaw_delta=float(u[1])*self.dt, control=u,
                    dt=self.dt, required_clearance=driveable_required_margin)
                if math.isnan(clearance) or clearance == -math.inf:
                    return ControlSafetyResult(i, 'invalid_driveable_clearance', SafetyFailure.INVALID_GEOMETRY)
                if clearance < driveable_required_margin:
                    return ControlSafetyResult(i, f'driveable_clearance_violation_step_{i}:{clearance:.3f}',
                                               SafetyFailure.DRIVEABLE_CLEARANCE)
            # Carry the exact held-command endpoint into the next interval.
            state = constant_twist_pose(previous_state, u, float(self.dt))
        return ControlSafetyResult(controls.shape[0])

    def _publish_command(self, v_cmd: float, w_cmd: float):
        cmd = Twist()
        cmd.linear.x = float(v_cmd)
        cmd.angular.z = float(w_cmd)
        with self._data_lock:
            # An in-flight solve may finish after a different callback declares
            # a fatal integrity stop. Publication is the final ownership boundary.
            invalid_command = not (math.isfinite(cmd.linear.x) and math.isfinite(cmd.angular.z))
            if (getattr(self, '_fatal_stop_triggered', False)
                    or getattr(self, '_terminal_stop_requested', False)
                    or invalid_command):
                cmd = Twist()
                if invalid_command:
                    self._command_stop_generation = int(getattr(self, '_command_stop_generation', 0)) + 1
                self._active_controls = None
                self._active_tape_request = None
                self._active_plan_started_at = None
                self._active_controls_original_len = 0
                self._publish_idle_execution_diagnostics()
            self.cmd_pub.publish(cmd)
            self.last_cmd = np.array([cmd.linear.x, cmd.angular.z], dtype=float)

    def _publish_idle_execution_diagnostics(self):
        """A stopped tape must not leave a held nonzero execution diagnostic."""
        if not hasattr(self, 'active_execution_diag_pub'):
            return
        target = getattr(self, '_current_wp_target', (math.nan, math.nan))
        diag = Float64MultiArray()
        diag.data = [0.]*9 + [
            float(getattr(self, '_current_wp_idx', math.nan)),
            float(getattr(self, '_current_wp_count', math.nan)),
            float(target[0]), float(target[1]),
            float(getattr(self, '_current_wp_dist', math.nan)),
            float(getattr(self, '_current_desired_yaw', math.nan)),
            float(getattr(self, '_current_yaw_error', math.nan)),
            float(getattr(self, '_current_tracking_yaw', math.nan)),
            float(getattr(self, '_current_tracking_yaw_source', math.nan)),
        ]
        try:
            self.active_execution_diag_pub.publish(diag)
        except Exception:
            # Diagnostics cannot prevent a fatal/ordinary zero command.
            pass

    def _result_safe_to_execute(self, result) -> tuple[bool, str]:
        """Return whether a solver result may replace the active control tape.

        Optimizer non-success can be acceptable for L-BFGS-B when the returned
        rollout is feasible and makes progress. It is not acceptable to execute
        controls that leave the known driveable domain, contain non-finite values,
        or fail to move the local tracker toward its waypoint.
        """
        request = getattr(self, '_active_plan_request', None)
        if request is not None and 'state_origin' in request:
            origin = request['state_origin']
            goal = request['goal']
            decision = validate_plan_result(
                result, initial_state=origin[0], initial_covariance=origin[1],
                goal_xy=goal[1:3], dt=self.dt,
                control_bounds=((self.v_min, self.v_max), (self.w_min, self.w_max)))
            if not decision.valid:
                return False, decision.reason
        controls = np.asarray(getattr(result, 'controls', []), dtype=float)
        if controls.ndim != 2 or controls.shape[0] == 0 or controls.shape[1] != 2:
            return False, 'empty_or_malformed_controls'
        if not np.all(np.isfinite(controls)):
            return False, 'nonfinite_controls'
        if not bool(getattr(result, 'rollout_valid', True)):
            reason = str(getattr(result, 'invalid_reason', '') or '').strip()
            return False, reason or 'invalid_rollout'
        clearance = float(getattr(result, 'min_predicted_obstacle_distance_m', math.nan))
        if math.isfinite(clearance) and clearance < 0.0:
            return False, f'negative_predicted_clearance:{clearance:.3f}'

        if self.use_hierarchical and self._hier_phase == 'LOCAL':
            current_dist = float(self._current_wp_dist)
            terminal_dist = float(getattr(result, 'terminal_goal_distance_pred', math.nan))
            # A unicycle must rotate to face the waypoint before it can reduce
            # distance, so a turning (low-translation) plan is legitimate progress,
            # not a freeze. Only require distance progress once roughly aligned;
            # while the heading error is large, allow the turn. This mirrors the
            # simple tracker's rotate-then-go gate (yaw_gate=0.6 rad) and applies
            # identically to all conditions.
            yaw_err = abs(float(getattr(self, '_current_yaw_error', 0.0)))
            require_distance_progress = yaw_err <= 0.6
            if (require_distance_progress and math.isfinite(current_dist)
                    and math.isfinite(terminal_dist)):
                if terminal_dist > current_dist - 1e-3:
                    return False, (
                        f'no_waypoint_progress:{terminal_dist:.3f}>={current_dist:.3f}'
                    )

        if self._pending_plan_started_at is not None:
            plan_latency_s = max(
                (self.get_clock().now() - self._pending_plan_started_at).nanoseconds * 1e-9,
                0.0,
            )
            tape_duration_s = float(controls.shape[0]) * max(float(self.dt), 1e-3)
            if plan_latency_s > tape_duration_s:
                return False, (
                    f'stale_control_tape:{plan_latency_s:.3f}>{tape_duration_s:.3f}'
                )

        return True, ''

    def _publish_active_plan_command(self):
        # Snapshot, expiry, publication and tape replacement share one ownership
        # boundary. A timer must not resurrect a tape after a concurrent stop.
        if getattr(self, "_terminal_stop_requested", False):
            return
        with getattr(self, '_correction_lock', nullcontext()), self._data_lock:
            self._publish_active_plan_command_locked()

    def _publish_active_plan_command_locked(self):
        with self._data_lock:
            controls_ref = self._active_controls
            controls = None if self._active_controls is None else self._active_controls.copy()
            started_at = self._active_plan_started_at
            original_len = int(self._active_controls_original_len)
        if controls is None or controls.size == 0 or started_at is None:
            return

        elapsed_s = (self.get_clock().now() - started_at).nanoseconds * 1e-9
        if self._active_controls is not controls_ref or self._active_plan_started_at is not started_at:
            return
        if not math.isfinite(elapsed_s) or elapsed_s < 0.0:
            # A reset does not make the previous epoch's tape new again.
            self._publish_safe_stop_command()
            return
        step_dt = max(float(self.dt), 1e-3)
        if elapsed_s >= controls.shape[0] * step_dt:
            # Do not keep replaying the terminal control of an exhausted local plan.
            # A slow or failed replan should leave the robot stopped, not coasting
            # into a boundary on stale controls.
            self._publish_safe_stop_command()
            return
        step_idx = min(int(elapsed_s / step_dt), controls.shape[0] - 1)
        request = getattr(self, '_active_tape_request', None)
        if request is not None and 'belief_origin' in request:
            if not self._plan_request_is_current(request):
                self._publish_safe_stop_command()
                return
            fresh = self._fresh_request_belief(request)
            if fresh is None:
                self._publish_safe_stop_command()
                return
            hierarchical_local = bool(self.use_hierarchical and self._hier_phase == 'LOCAL')
            # A hierarchical LOCAL tape follows a globally admitted route.  Its
            # geometry is deliberately not vetoed again from a potentially
            # drifted runtime belief: the experiment must expose a collision or
            # complete the route, not turn model error into an uninformative stop.
            safety = (ControlSafetyResult(1) if hierarchical_local else
                      self._simple_plan_safe_to_execute(controls[step_idx:step_idx + 1], fresh[0]))
            if (not self._plan_request_is_current(request)
                    or self._active_controls is not controls_ref
                    or not self._execution_belief_is_current(fresh[2])):
                return
            if safety.safe_steps <= 0:
                self._publish_safe_stop_command()
                return
        u = controls[step_idx]
        diag = Float64MultiArray()
        diag.data = [
            elapsed_s,
            float(max(controls.shape[0] * step_dt - elapsed_s, 0.0)),
            float(step_idx),
            float(controls.shape[0]),
            float(original_len),
            float(u[0]),
            float(u[1]),
            float(self._last_latency_skip_steps),
            float(self._last_latency_skip_s),
            float(self._current_wp_idx),
            float(self._current_wp_count),
            float(self._current_wp_target[0]),
            float(self._current_wp_target[1]),
            float(self._current_wp_dist),
            float(self._current_desired_yaw),
            float(self._current_yaw_error),
            float(self._current_tracking_yaw),
            float(self._current_tracking_yaw_source),
        ]
        self._publish_command(u[0], u[1])
        if getattr(self, '_fatal_stop_triggered', False):
            return
        try:
            self.active_execution_diag_pub.publish(diag)
        except Exception:
            pass

    def _after_plan_result(self, result):
        # Keep following the current planned control sequence until replanning replaces it.
        request = getattr(self, '_active_plan_request', None)
        with self._data_lock:
            # A result cancelled by an ordinary stop is discarded entirely,
            # including its rejection and stop side effects: it must neither
            # publish nonzero nor clear a newer replacement tape.
            if not self._plan_request_is_current(request):
                return
        safe, reason = self._result_safe_to_execute(result)
        if not safe:
            self._reject_current_request(request, f'Rejected local control tape: {reason}')
            return
        controls = np.asarray(result.controls, dtype=float)
        started_at = self.get_clock().now()
        skip_steps = 0
        latency_s = 0.0
        if self.latency_compensate_plan_handoff and self._pending_plan_started_at is not None:
            latency_s = max((started_at - self._pending_plan_started_at).nanoseconds * 1e-9, 0.0)
            latency_s = min(latency_s, max(float(self._pending_plan_started_active_remaining_s), 0.0))
            step_dt = max(float(self.dt), 1e-3)
            skip_steps = min(int(latency_s / step_dt), int(controls.shape[0]))
            fractional_s = max(latency_s - skip_steps * step_dt, 0.0)
            if skip_steps >= controls.shape[0]:
                self._reject_current_request(request, 'Latency compensation exhausted control tape')
                return
            if fractional_s > 0.0:
                started_at = started_at - Duration(seconds=fractional_s)
            controls = controls[skip_steps:]
        # A stop can occur after _result_safe_to_execute returns, so that call is
        # not the ownership barrier; the shared installation transaction is.
        self._install_control_tape(
            controls,
            original_len=int(result.controls.shape[0]),
            started_at=started_at,
            latency_skip_steps=int(skip_steps),
            latency_skip_s=float(latency_s),
            log_prefix='Rejected local control tape at install',
        )
        return

    def _publish_safe_stop_command(self, reason: str = ''):
        """Stop, and cancel any planning work that began before this moment.

        Clearing the tape alone was not enough: an in-flight solve could install
        its result immediately afterwards, so the stop lasted only until the
        solver returned. Bumping the generation inside the same locked block
        makes the cancellation atomic with the stop itself. The fatal latch is
        independent and remains latched.
        """
        if not hasattr(self, 'cmd_pub'):
            return
        with self._data_lock:
            had_active_tape = bool(
                self._active_controls is not None
                and np.asarray(self._active_controls).size > 0
            )
            if had_active_tape:
                detail = reason or 'unspecified_call_site'
                try:
                    self.get_logger().warn(
                        '[execution] clearing active control tape: '
                        f'reason={detail}, phase={getattr(self, "_hier_phase", "unknown")}, '
                        f'waypoint={getattr(self, "_wp_idx", -1)}/'
                        f'{len(getattr(self, "_waypoints", []))}'
                    )
                except (AttributeError, RuntimeError):
                    # Diagnostics must not alter stop semantics in lightweight
                    # transaction tests or during ROS teardown.
                    pass
            self._command_stop_generation = int(
                getattr(self, '_command_stop_generation', 0)) + 1
            self._active_controls = None
            self._active_tape_request = None
            self._active_plan_started_at = None
            self._active_controls_original_len = 0
            self._publish_command(0.0, 0.0)
            self._publish_idle_execution_diagnostics()


def main(args=None):
    rclpy.init(args=args)
    node = EfeAgentNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except RuntimeError:
            pass


if __name__ == '__main__':
    main()
