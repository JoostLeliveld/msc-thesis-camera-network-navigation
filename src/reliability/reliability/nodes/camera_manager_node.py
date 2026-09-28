#!/usr/bin/env python3
"""Paper-facing multi-camera measurement manager.

Each identified detector frame is processed once through the deterministic sensor gate,
raw ground projection, selected commissioned correction, and its matched covariance.
Admitted simultaneous camera measurements are fused before the single robot filter update.
Ground truth is forbidden from this runtime path.
"""

from __future__ import annotations

import collections
import hashlib
import itertools
from contextlib import nullcontext
from collections import deque
from dataclasses import replace
import json
import math
import time
import threading
import uuid
from pathlib import Path

from reliability.source_batch_buffer import SourceBatchBuffer
from reliability.camera_manager import CameraManager, CameraManagerConfig
from reliability.contracts import CameraObservation, ContractValidationError
from reliability.common_time import MotionPose, MotionPoseSnapshot
from reliability.manager_state import AdmissionBeliefHistory, ManagerInputs
from reliability.fusion_event import FusedCorrectionEvent, publish_fused_event
from unav_common.camera_outcomes import (
    OutcomeJournal, journal_path, OUTCOME_HISTORY_DEPTH, DEFAULT_JOURNAL_MAX_BYTES,
)
from unav_common.terminal_stop import (
    TERMINAL_STOP_ACK_TOPIC,
    TERMINAL_STOP_REQUEST_TOPIC,
    TerminalStopAck,
    terminal_stop_ack_to_json,
    terminal_stop_request_from_json,
)
from reliability.fusion import (
    MapObservation,
    map_observations_to_json,
)
# Runtime and replay share the ROS-free fusion implementation.
from reliability.measurement_fusion import (
    FUSION_RULE_BEST_SINGLE, FUSION_RULE_DISTANCE_ANGLE, FUSION_RULE_INDEPENDENT,
    FUSION_RULE_JOINT_NETWORK, SUPPORTED_FUSION_RULES,
    combine_measurements_2d as _combine_by_rule,
    gated_measurement_fusion_2d as _gated_fusion,
)
from reliability.handover import HandoverUncertaintyConfig, handover_adjusted_observation
from reliability.projection import (
    camera_model_from_world,
    project_observation_to_world,
    project_observation_to_world_with_covariance,
)
from reliability.observation_geometry import validate_observation_geometry
from reliability.observation_gates import (
    UsableObservationGateConfig,
    evaluate_sensor_gate,
)
from reliability.providers import GridMapReliabilityProvider
from reliability.replay import ReplayConfig, ReplayMode, _with_provider_quality

try:
    import rclpy
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from rclpy.clock import Clock, ClockType
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from std_msgs.msg import String
except ImportError:  # pragma: no cover - keeps the library importable without ROS
    rclpy = None
    Node = object
    PoseWithCovarianceStamped = None
    String = None


DEFAULT_CAMERA_IDS = ["camera_A", "camera_B", "camera_C", "camera_D"]
DEFAULT_MODEL_INCLUDES = [
    "external_camera",
    "external_camera_b",
    "external_camera_c",
    "external_camera_d",
]
# Yaw variance for the position-only active output: deliberately
# non-informative so no consumer can mistake the selection for a heading fix.
NONINFORMATIVE_YAW_VAR = float(math.pi**2)
#: Raw projected-pixel covariance profile retained as a commissioning baseline.
COMMISSIONED_COVARIANCE = "commissioned_sigma_px"
#: Candidate-specific covariance profiles loaded from locked commissioning artifacts.
COMMISSIONED_WORLD_COVARIANCE = "commissioned_world_r"
COMMISSIONED_REFERENCE_COVARIANCE = "commissioned_reference_r"
COMMISSIONED_VISIBILITY_COVARIANCE = "commissioned_visibility_r"
COMMISSIONED_PERCEPTION_COVARIANCE = "commissioned_perception_r"
SUPPORTED_COVARIANCE_PROFILES = (COMMISSIONED_COVARIANCE, COMMISSIONED_WORLD_COVARIANCE,
                                 COMMISSIONED_REFERENCE_COVARIANCE,
                                 COMMISSIONED_VISIBILITY_COVARIANCE,
                                 COMMISSIONED_PERCEPTION_COVARIANCE)
#: How a detector's box is turned into a statement about where the robot is.
OBSERVATION_MODEL_RAW_BOX = "raw_box"
#: The packaged neural box-feature correction, run forward on runtime-only inputs.
OBSERVATION_MODEL_LEARNED_NN = "learned_nn"
OBSERVATION_MODEL_VISIBILITY_PATCH = "visibility_patch"
OBSERVATION_MODEL_HIERARCHICAL_RESIDUAL = "hierarchical_residual"
OBSERVATION_MODEL_SPATIAL_RESIDUAL = "spatial_residual"
OBSERVATION_MODEL_JOINT_RGB_GAUSSIAN = "joint_rgb_gaussian"
OBSERVATION_MODEL_GLOBAL_RESIDUAL = "global_residual"
OBSERVATION_MODEL_PER_CAMERA_RESIDUAL = "per_camera_residual"
SUPPORTED_OBSERVATION_MODELS = (
    OBSERVATION_MODEL_RAW_BOX,
    OBSERVATION_MODEL_LEARNED_NN,
    OBSERVATION_MODEL_VISIBILITY_PATCH,
    OBSERVATION_MODEL_HIERARCHICAL_RESIDUAL,
    OBSERVATION_MODEL_SPATIAL_RESIDUAL,
    OBSERVATION_MODEL_JOINT_RGB_GAUSSIAN,
    OBSERVATION_MODEL_GLOBAL_RESIDUAL,
    OBSERVATION_MODEL_PER_CAMERA_RESIDUAL,
)


def _message_stamp_ns(message):
    stamp = message.header.stamp
    if (type(stamp.sec) is not int or type(stamp.nanosec) is not int
            or stamp.sec < 0 or not 0 <= stamp.nanosec < 1_000_000_000):
        raise ContractValidationError("invalid ROS capture stamp")
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def _message_yaw(message):
    q = message.pose.pose.orientation
    values = tuple(float(v) for v in (q.x, q.y, q.z, q.w))
    norm2 = sum(v*v for v in values)
    if not all(math.isfinite(v) for v in values) or abs(norm2 - 1.) > 1e-3:
        raise ContractValidationError("pose quaternion must be finite and normalized")
    x, y, z, w = (v/math.sqrt(norm2) for v in values)
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def _diagnostic_values(value):
    """Missing diagnostic scalars serialize as null; measurement validation is separate."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _diagnostic_values(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_diagnostic_values(v) for v in value]
    return value


def load_commissioned_sigma_px(calibration_path: str) -> float:
    """The frozen detector noise, in pixels, read from the commissioning artifact.

    Read rather than typed in, so an arm cannot be driven against a remembered number: the
    covariance every fusion rule is judged on comes from this one value.
    """

    payload = json.loads(Path(calibration_path).read_text())
    sigma_px = float(payload["calibration"]["sigma_px"])
    if not math.isfinite(sigma_px) or sigma_px <= 0.0:
        raise ValueError(f"sigma_px in {calibration_path} is not a positive number")
    return sigma_px


def load_commissioned_sigma_px_by_camera(calibration_path: str) -> dict[str, float]:
    """Read optional per-camera detector noise from a commissioning artifact."""

    payload = json.loads(Path(calibration_path).read_text())
    table = payload.get("calibration", {}).get("sigma_px_by_camera") or {}
    out = {}
    for camera_id, value in table.items():
        sigma = float(value)
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(f"sigma_px for {camera_id} in {calibration_path} is not positive")
        out[str(camera_id)] = sigma
    return out


def load_commissioned_world_covariance(commissioning_path: str):
    """The commissioned world-plane covariance table, read from its study artifact.

    Returns ``(bias_by_camera, table, confidence_edges, floor_m)`` where ``table`` maps
    ``(camera_id, band_index)`` to a 2x2 covariance in m^2. Read rather than typed in, so an
    arm cannot be driven against a remembered number.
    """

    payload = json.loads(Path(commissioning_path).read_text())
    block = payload["models"]["radial"]
    edges = [float(v) for v in payload["confidence_edges"]]
    bias = {camera: tuple(tuple(float(x) for x in row) for row in theta)
            for camera, theta in block["bias_parameters"].items()}
    table = {}
    for key, entry in block["commissioned"].items():
        camera_id, _, band = key.partition("|")
        index = int(band.lstrip("q"))
        matrix = entry["R_pred"]
        table[(camera_id, index)] = (
            (float(matrix[0][0]), float(matrix[0][1])),
            (float(matrix[1][0]), float(matrix[1][1])),
        )
    if not table:
        raise ValueError(f"{commissioning_path} contains no commissioned covariance bands")
    return bias, table, edges, float(payload.get("belief_lean_floor_m", 0.0))


def commissioned_world_band(confidence: float, edges) -> int:
    """Which frozen confidence band a reading falls in."""

    index = 0
    for edge in edges:
        if float(confidence) < edge:
            break
        index += 1
    return index


def commissioned_pixel_covariance(sigma_px: float):
    """``R_pix = sigma_px^2 I`` -- isotropic in pixels, identical for every camera."""

    variance = float(sigma_px) ** 2
    return ((variance, 0.0), (0.0, variance))


def offset_away_from_camera(world_xy, camera_position, offset_m: float):
    """Push a ground reading away from its camera by a fixed distance.

    The fixed-offset observation model, and the whole of it: the box's bottom edge is the
    robot's nearest point to the camera, so its floor projection lands short, along the
    viewing bearing. One number cannot follow the 11 cm that gap swings as the robot turns --
    which is what this arm exists to demonstrate rather than to hide.
    """

    dx = float(world_xy[0]) - float(camera_position[0])
    dy = float(world_xy[1]) - float(camera_position[1])
    norm = math.hypot(dx, dy)
    if norm <= 1.0e-9:
        return (float(world_xy[0]), float(world_xy[1]))
    scale = float(offset_m) / norm
    return (float(world_xy[0]) + dx * scale, float(world_xy[1]) + dy * scale)


def _nearest_state_pose(
    history: list[tuple[float, tuple[float, float, float]]] | deque,
    timestamp_s: float,
    *,
    max_delta_s: float,
) -> tuple[float, float, float] | None:
    """The operational belief pose nearest a camera capture time, yaw included.

    Returns ``None`` when the history is empty or the nearest entry is too old, and also
    when the entry predates the yaw-carrying history format -- an observation function
    evaluated at a guessed heading is not a correction.
    """

    if (
        not history
        or not math.isfinite(timestamp_s)
        or not math.isfinite(max_delta_s)
        or max_delta_s < 0.0
    ):
        return None
    nearest_stamp, nearest_pose = min(
        history, key=lambda row: abs(float(row[0]) - timestamp_s)
    )
    if abs(float(nearest_stamp) - timestamp_s) > max_delta_s:
        return None
    if len(nearest_pose) < 3:
        return None
    return float(nearest_pose[0]), float(nearest_pose[1]), float(nearest_pose[2])


def _nearest_state_prediction(
    history,
    timestamp_s: float,
    *,
    max_delta_s: float,
):
    """Nearest canonical belief mean and covariance at a camera capture time."""

    if (
        not history
        or not math.isfinite(timestamp_s)
        or not math.isfinite(max_delta_s)
        or max_delta_s < 0.0
    ):
        return None
    nearest_stamp, mean, covariance = min(
        history, key=lambda row: abs(float(row[0]) - timestamp_s)
    )
    if abs(float(nearest_stamp) - timestamp_s) > max_delta_s or len(mean) < 3:
        return None
    try:
        pose = tuple(float(mean[index]) for index in range(3))
        cov = tuple(tuple(float(covariance[row][column]) for column in range(3))
                    for row in range(3))
    except (TypeError, IndexError, ValueError):
        return None
    if not all(math.isfinite(value) for value in pose):
        return None
    if not all(math.isfinite(value) for row in cov for value in row):
        return None
    return pose, cov


def _nearest_state_xy(
    history: list[tuple[float, tuple[float, float]]] | deque,
    timestamp_s: float,
    *,
    max_delta_s: float,
) -> tuple[float, float] | None:
    """Return the operational state prediction nearest a camera capture time."""

    if (
        not history
        or not math.isfinite(timestamp_s)
        or not math.isfinite(max_delta_s)
        or max_delta_s < 0.0
    ):
        return None
    nearest_stamp, nearest_xy = min(
        history, key=lambda row: abs(float(row[0]) - timestamp_s)
    )
    if abs(float(nearest_stamp) - timestamp_s) > max_delta_s:
        return None
    return float(nearest_xy[0]), float(nearest_xy[1])


def propagate_correction_to_now(xy, covariance_m2, pose_then, pose_now, *,
                                drift_std_m_per_s: float, dt_s: float,
                                residual_interval_s: float = 0.0):
    """Carry a correction between two measured motion poses.

    This alternative must not be combined with estimator-side rewind/replay. Motion support
    over the interval supplies the displacement; declared propagation uncertainty is added
    to the covariance.
    """

    dx = float(pose_now[0]) - float(pose_then[0])
    dy = float(pose_now[1]) - float(pose_then[1])
    dt = max(float(dt_s), 0.0)
    grown = max(float(drift_std_m_per_s) * dt, 0.0) ** 2
    cxx = float(covariance_m2[0][0]) + grown
    cxy = float(covariance_m2[0][1])
    cyy = float(covariance_m2[1][1]) + grown

    # Declare residual publication-to-consumption delay along the direction of travel.
    speed = math.hypot(dx, dy) / dt if dt > 0.0 else 0.0
    residual = max(float(residual_interval_s), 0.0) * speed
    if residual > 0.0 and (dx or dy):
        norm = math.hypot(dx, dy)
        ux, uy = dx / norm, dy / norm
        along = residual ** 2
        cxx += along * ux * ux
        cxy += along * ux * uy
        cyy += along * uy * uy

    covariance = ((cxx, cxy), (cxy, cyy))
    return (float(xy[0]) + dx, float(xy[1]) + dy), covariance, (dx, dy)


def align_observations_to_common_time(
    observations,
    history,
    *,
    max_pose_delta_s: float,
    drift_std_m_per_s: float,
    max_motion_gap_s: float = 0.35,
    frame_id: str = "map_bev",
    stamp_ns_by_camera=None,
    support_by_camera=None,
):
    """Carry asynchronous camera observations to the newest capture instant.

    Fusion requires all operands to describe the same state. Camera rendering
    can be phase-shifted, so a detector batch may legitimately contain several
    capture stamps; averaging those positions directly creates motion bias.
    Runtime supplies an immutable labelled odometry snapshot. Unsupported
    operands retain their capture evidence and are explicitly refused. The tuple
    input adapter is retained for standalone library callers; runtime
    never feeds corrected belief positions through it.
    """

    if not observations:
        return [], [], math.nan
    stamps = {o.camera_id: round(float(o.timestamp_s)*1e9) for o in observations}
    if stamp_ns_by_camera is not None:
        stamps.update({o.camera_id: stamp_ns_by_camera[o.camera_id] for o in observations})
    target_ns = max(stamps.values())
    target_s = target_ns / 1e9
    if not isinstance(history, MotionPoseSnapshot):
        history = MotionPoseSnapshot.capture(
            [MotionPose(round(float(t)*1e9), tuple(p[:2]), float(p[2]), frame_id, "legacy_helper")
             for t, p in (history or ())], target_frame=frame_id, source_frame=frame_id,
            epoch="legacy_helper")
    aligned = []
    rejected = []
    for observation in observations:
        source_ns = stamps[observation.camera_id]
        source_s = source_ns / 1e9
        support = history.displacement(source_ns, target_ns, max_gap_s=max_motion_gap_s,
                                       max_endpoint_delta_s=max_pose_delta_s)
        if support_by_camera is not None:
            support_by_camera[observation.camera_id] = support.to_dict()
        if source_ns == target_ns:
            aligned.append(replace(observation, timestamp_s=target_s))
            continue
        if not support.supported:
            rejected.append(str(observation.camera_id))
            continue
        xy, covariance, _delta = propagate_correction_to_now(
            observation.xy_m,
            observation.covariance_m2,
            (0., 0.),
            support.delta_xy_m,
            drift_std_m_per_s=drift_std_m_per_s,
            dt_s=max(target_s - source_s, 0.0),
            residual_interval_s=0.0,
        )
        aligned.append(
            replace(
                observation,
                timestamp_s=target_s,
                xy_m=xy,
                covariance_m2=covariance,
            )
        )
    return aligned, sorted(rejected), target_s


def _group_spread_m(observations) -> float:
    """The widest disagreement between any two of these readings, in metres."""
    if len(observations) < 2:
        return 0.0
    return max(
        math.hypot(float(a.xy_m[0]) - float(b.xy_m[0]),
                   float(a.xy_m[1]) - float(b.xy_m[1]))
        for a, b in itertools.combinations(observations, 2)
    )


def _largest_agreeing_group(observations, max_disagreement_m: float):
    """The biggest set of readings that all agree with each other, within the bound.

    Used only to start the belief, where there is no prior to gate against. Ties are
    broken by the tightest group so the initial pose comes from the readings that agree
    best, and by camera id so the choice cannot depend on arrival order.
    """
    best: list = []
    best_spread = math.inf
    for size in range(len(observations), 1, -1):
        if size < len(best):
            break
        for candidate in itertools.combinations(observations, size):
            spread = _group_spread_m(candidate)
            if spread > max_disagreement_m:
                continue
            group = list(candidate)
            key = (len(group), -spread,
                   tuple(sorted(str(o.camera_id) for o in group)))
            if not best or key > (len(best), -best_spread,
                                  tuple(sorted(str(o.camera_id) for o in best))):
                best, best_spread = group, spread
        if best:
            break
    return best


def _bootstrap_agreeing_group(
    observations,
    max_disagreement_m: float,
    *,
    prior_xy=None,
):
    """Return the largest mutually consistent bootstrap group.

    Without a declared start prior this uses the conservative rule:
    a singleton is never a quorum.  With a start prior, every retained camera must
    independently fall within the same metric bound of that prior, and a singleton
    may be returned.  The caller still decides how many support sources are required.
    """
    candidates = list(observations)
    minimum_size = 2
    if prior_xy is not None:
        px, py = float(prior_xy[0]), float(prior_xy[1])
        candidates = [
            observation for observation in candidates
            if math.hypot(
                float(observation.xy_m[0]) - px,
                float(observation.xy_m[1]) - py,
            ) <= max_disagreement_m
        ]
        minimum_size = 1
    best: list = []
    best_spread = math.inf
    best_prior_residual = math.inf
    for size in range(len(candidates), minimum_size - 1, -1):
        for candidate in itertools.combinations(candidates, size):
            spread = _group_spread_m(candidate)
            if spread > max_disagreement_m:
                continue
            prior_residual = (
                max(
                    math.hypot(
                        float(observation.xy_m[0]) - float(prior_xy[0]),
                        float(observation.xy_m[1]) - float(prior_xy[1]),
                    )
                    for observation in candidate
                )
                if prior_xy is not None else 0.0
            )
            group = list(candidate)
            key = (
                len(group),
                -spread,
                -prior_residual,
                tuple(sorted(str(observation.camera_id) for observation in group)),
            )
            old_key = (
                len(best),
                -best_spread,
                -best_prior_residual,
                tuple(sorted(str(observation.camera_id) for observation in best)),
            )
            if not best or key > old_key:
                best = group
                best_spread = spread
                best_prior_residual = prior_residual
        if best:
            break
    return best


def _fusion_report_covariance(covariance_m2, *, common_mode_std_m: float = 0.0):
    """Add a locked shared-error component after camera fusion."""

    shared = max(float(common_mode_std_m), 0.0) ** 2
    return (
        (float(covariance_m2[0][0]) + shared, float(covariance_m2[0][1])),
        (float(covariance_m2[1][0]), float(covariance_m2[1][1]) + shared),
    )


def _synchronous_fusion_candidates(
    manager: CameraManager,
    *,
    now_s: float,
    observations: list[MapObservation],
    max_timestamp_spread_s: float,
) -> tuple[
    list[MapObservation],
    dict[str, float],
    dict[str, tuple[str, ...]],
    list[str],
]:
    """Apply manager eligibility and discard old cached camera frames."""
    if not math.isfinite(max_timestamp_spread_s) or max_timestamp_spread_s < 0.0:
        raise ValueError("max_timestamp_spread_s must be finite and non-negative")
    candidates, scores, rejected = manager.eligible_observations(
        timestamp_s=now_s, observations=observations
    )
    if not candidates:
        return [], scores, rejected, []
    newest_stamp_s = max(float(obs.timestamp_s) for obs in candidates.values())
    synchronous = [
        obs for obs in candidates.values()
        if newest_stamp_s - float(obs.timestamp_s) <= max_timestamp_spread_s
    ]
    time_skew_rejected = sorted(
        camera_id for camera_id, obs in candidates.items()
        if newest_stamp_s - float(obs.timestamp_s) > max_timestamp_spread_s
    )
    return synchronous, scores, rejected, time_skew_rejected


class CameraManagerNode(Node):
    """Shadow-mode (optionally active) live camera selection."""

    def __init__(self) -> None:
        super().__init__("camera_manager_node")
        self._input_lock = threading.RLock()
        self._decision_lock = threading.RLock()
        self._terminal_stopped = False
        self._terminal_stop_request_id = ""
        self._session_stopped_appended = False
        self._manager_epoch = uuid.uuid4().hex
        self._decision_snapshot = None
        self._fusion_publication_seq = 0
        self.declare_parameter("camera_ids", DEFAULT_CAMERA_IDS)
        self.declare_parameter(
            "observation_topic_template", "/perception/camera_observation/{camera_id}"
        )
        self.declare_parameter("world_sdf", "")
        self.declare_parameter("camera_model_includes", DEFAULT_MODEL_INCLUDES)
        self.declare_parameter("camera_calibration_ids", [""])
        self.declare_parameter("camera_image_frame_ids", [""])
        self.declare_parameter("gp_artifacts", [""])
        # Launch-friendly alternative to the aligned list: one string with a
        # {camera_id} placeholder, e.g.
        # ".../final_02/gp/{camera_id}/det_hit_expected_kernel_gp.npz".
        self.declare_parameter("gp_artifact_template", "")
        self.declare_parameter("availability_model_path", "")
        self.declare_parameter("availability_model_expected_sha256", "")
        self.declare_parameter("decision_rate_hz", 5.0)
        # Raw projection is the detected box-bottom ray intersected with the floor plane.
        self.declare_parameter("require_gp_artifacts", True)
        self.declare_parameter("frame_id", "map_bev")
        self.declare_parameter("authority", "shadow")
        self.declare_parameter("decision_topic", "/reliability/camera_manager/decision")
        self.declare_parameter("outcome_journal_path", "")
        self.declare_parameter("outcome_journal_max_bytes", DEFAULT_JOURNAL_MAX_BYTES)
        # read only to measure how far the robot moved while a correction was in flight
        self.declare_parameter("odometry_topic", "/odom_noisy")
        self.declare_parameter("odometry_frame_id", "odom")
        self.declare_parameter("odometry_to_map_yaw_rad", float("nan"))
        self.declare_parameter("selected_topic", "/reliability/camera_manager/selected_observation")
        self.declare_parameter("active_output_topic", "/state/bev")
        # GP reliability is a property of the predicted robot location, not of
        # the camera measurement being scored.  Query a timestamp-matched
        # operational belief; the measurement is used only during bootstrap.
        self.declare_parameter("reliability_query_topic", "/planner_belief")
        self.declare_parameter("belief_state_topic", "/planner/belief_state")
        self.declare_parameter("reliability_query_max_time_delta_s", 0.35)
        # Operational gates: defaults mirror CameraManagerConfig; override from
        # the frozen study/protocol config in the launch file, never here.
        defaults = CameraManagerConfig()
        # Covariance-weighted fusion of ALL in-view cameras (Joseph sequential
        # update with NIS/disagreement gating) instead of hard single-camera
        # selection. Default False preserves the selection path.
        self.declare_parameter("fusion_mode", False)
        # Publish the PER-CAMERA map observations alongside the fused pose, so a
        # consumer can fold them into its own filter one at a time instead of
        # receiving a single pre-fused pose. Purely additive: the fused/selected
        # outputs are unchanged, so this is safe to leave on.
        self.declare_parameter("publish_map_observations", False)
        self.declare_parameter(
            "map_observations_topic", "/reliability/camera_manager/map_observations"
        )
        self.declare_parameter(
            "fused_correction_topic", "/reliability/camera_manager/fused_correction"
        )
        self.declare_parameter("fusion_disagreement_gate_m", 0.6)
        # Evidence-grade batched fusion waits for every subscribed camera's
        # result from one detector invocation. This makes a manager timer rate
        # higher than the detector rate safe: cached pixels are never reused.
        self.declare_parameter("require_source_batch_id", False)
        self.declare_parameter("source_batch_timeout_wall_s", 2.0)
        self.declare_parameter("source_batch_capacity", 64)
        # Bootstrap is a separate, explicit quorum rule until a belief exists.
        self.declare_parameter("bootstrap_min_cameras", 2)
        self.declare_parameter("bootstrap_max_disagreement_m", 0.30)
        # Optional surveyed start pose used only during recursive-filter bootstrap.
        self.declare_parameter("use_task_start_as_bootstrap_prior", False)
        # When enabled, a surveyed/staged task start is one bootstrap support source.
        # At least one admitted camera must still agree with it; the prior can never
        # initialise the camera belief by itself.
        self.declare_parameter("bootstrap_prior_counts_as_support", False)
        self.declare_parameter("bootstrap_prior_xyyaw", [0.0, 0.0, 0.0])
        # The planner starts from a declared initial prior (METHOD amendment 2026-09-24
        # night): treat the belief as anchored from the start, so no batch waits for a
        # camera bootstrap quorum.
        self.declare_parameter("assume_initial_belief_anchor", False)
        # Optional paired covariance floor. Both axes must be zero or both positive.
        self.declare_parameter("bias_floor_along_slope_m_per_m", 0.0)
        self.declare_parameter("bias_floor_across_slope_m_per_m", 0.0)
        # Views fused into one correction must come from the same detector round.
        self.declare_parameter("fusion_max_timestamp_spread_s", 0.05)
        # The profile is explicit in every run manifest.
        self.declare_parameter("covariance_profile", COMMISSIONED_COVARIANCE)
        # The number is read from the artifact rather than typed in; set commissioned_sigma_px
        # only to override it deliberately, and say so in the run's provenance.
        self.declare_parameter("commissioned_calibration_path", "")
        self.declare_parameter("commissioned_sigma_px", 0.0)
        #: Path to the commissioned world-plane covariance artifact, required by
        #: covariance_profile=commissioned_world_R.
        self.declare_parameter("commissioned_world_covariance_path", "")
        # Optional per-camera raw projected-pixel baseline.
        self.declare_parameter("commissioned_per_camera_sigma", False)
        # The error the cameras make TOGETHER, as a standard deviation in metres, added to
        # the fused covariance AFTER combining. Independent fusion shrinks the stated
        # uncertainty like 1/N; a shared error does not shrink at all, so without this term
        # the fused answer grows confidently wrong as cameras are added. 0 means the model
        # claims the cameras err independently.
        #
        # Zero is retained until the current commissioning drives select a dependence model.
        self.declare_parameter("fusion_common_mode_std_m", 0.0)
        # Carry a correction from its capture timestamp to the estimator timestamp.
        self.declare_parameter("correction_timestamp_compensation", False)
        # Uncertainty added while carrying a correction forward in time.
        self.declare_parameter("correction_propagation_drift_std_m_per_s", 0.05)
        # Declared publication-to-consumption residual interval.
        self.declare_parameter("correction_residual_interval_s", 0.05)
        # Which rule turns several cameras into one measurement -- the treatment of the
        # fusion comparison. Named explicitly by every campaign; see SUPPORTED_FUSION_RULES.
        self.declare_parameter("fusion_rule", FUSION_RULE_INDEPENDENT)
        # Deterministic pre-estimator gate. The loaded config is rejected at startup if it
        # enables association, tracking, or localizer acceptance.
        self.declare_parameter("sensor_gate_config_path", "")
        self.declare_parameter("sensor_gate_config_expected_sha256", "")
        self.declare_parameter("observation_model", OBSERVATION_MODEL_RAW_BOX)
        # Where the packaged neural box correction lives. Required by the learned models
        # and ignored by every other one.
        self.declare_parameter("learned_correction_path", "")
        self.declare_parameter("learned_correction_expected_sha256", "")
        self.declare_parameter("visibility_sensor_model_path", "")
        self.declare_parameter("visibility_sensor_model_expected_sha256", "")
        self.declare_parameter("perception_sensor_model_path", "")
        self.declare_parameter("perception_sensor_model_expected_sha256", "")
        self.declare_parameter("commissioned_world_covariance_expected_sha256", "")
        self.declare_parameter("min_spatial_trust", defaults.min_spatial_trust)
        self.declare_parameter("min_association_confidence", defaults.min_association_confidence)
        self.declare_parameter("max_measurement_age_s", defaults.max_measurement_age_s)
        self.declare_parameter("age_decay_s", defaults.age_decay_s)
        self.declare_parameter("candidate_score_margin", defaults.candidate_score_margin)
        self.declare_parameter(
            "required_consecutive_better_frames", defaults.required_consecutive_better_frames
        )
        self.declare_parameter(
            "max_cross_camera_disagreement_m", defaults.max_cross_camera_disagreement_m
        )
        self.declare_parameter("max_overlap_time_delta_s", defaults.max_overlap_time_delta_s)
        self.declare_parameter(
            "require_consistency_when_source_available",
            defaults.require_consistency_when_source_available,
        )
        self.declare_parameter("fallback_on_active_camera_loss", defaults.fallback_on_active_camera_loss)

        self.camera_ids = [str(item) for item in self.get_parameter("camera_ids").value]
        template = str(self.get_parameter("observation_topic_template").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.authority = str(self.get_parameter("authority").value).strip().lower()
        if self.authority not in ("shadow", "active"):
            raise ValueError(f"authority must be 'shadow' or 'active', got {self.authority!r}")

        world_sdf = str(self.get_parameter("world_sdf").value)
        includes = [str(item) for item in self.get_parameter("camera_model_includes").value]
        if not world_sdf:
            raise ValueError("world_sdf parameter is required for projection")
        if len(includes) != len(self.camera_ids):
            raise ValueError("camera_model_includes must align with camera_ids")
        self.camera_models = {
            camera_id: camera_model_from_world(world_sdf, include_name=include)
            for camera_id, include in zip(self.camera_ids, includes)
        }
        calibration_ids = list(self.get_parameter("camera_calibration_ids").value)
        image_frame_ids = list(self.get_parameter("camera_image_frame_ids").value)
        if image_frame_ids == [""]:
            image_frame_ids = list(self.camera_ids)
        if (len(calibration_ids) != len(self.camera_ids) or len(image_frame_ids) != len(self.camera_ids)
                or any(not isinstance(v, str) or not v.strip() for v in (*calibration_ids, *image_frame_ids))):
            raise ValueError("explicit camera calibration/frame IDs must align with configured cameras")
        self.camera_calibration_ids_by_camera = dict(zip(self.camera_ids, calibration_ids))
        self.camera_image_frame_ids_by_camera = dict(zip(self.camera_ids, image_frame_ids))
        artifacts = [str(item) for item in self.get_parameter("gp_artifacts").value]
        if artifacts == [""]:
            artifacts = []
        template_artifact = str(self.get_parameter("gp_artifact_template").value)
        if not artifacts and template_artifact:
            artifacts = [
                template_artifact.format(camera_id=camera_id) for camera_id in self.camera_ids
            ]
        if artifacts and len(artifacts) != len(self.camera_ids):
            raise ValueError("gp_artifacts must be empty or align with camera_ids")
        availability_path = str(self.get_parameter("availability_model_path").value or "").strip()
        if artifacts and availability_path:
            raise ValueError("gp_artifacts and the Stage-08 availability model are alternative providers")
        if (bool(self.get_parameter("require_gp_artifacts").value)
                and len(artifacts) != len(self.camera_ids) and not availability_path):
            raise ValueError("one frozen GP artifact per camera is required by the commissioning contract")
        providers = {}
        for camera_id, artifact in zip(self.camera_ids, artifacts):
            if artifact:
                providers[camera_id] = GridMapReliabilityProvider.from_npz(
                    Path(artifact), camera_id=camera_id, out_of_bounds_policy="min"
                )
        self.commissioned_availability = None
        if availability_path:
            from reliability.commissioned_availability import CommissionedAvailabilityModel
            self.commissioned_availability = CommissionedAvailabilityModel(
                availability_path,
                expected_sha256=(
                    str(self.get_parameter("availability_model_expected_sha256").value)
                    or None
                ),
            )
            self.get_logger().info(
                f"frozen position-only availability loaded from {availability_path}; "
                f"sha256={self.commissioned_availability.sha256}"
            )
        # Runtime and replay share one provider configuration.
        self.replay_config = ReplayConfig(
            mode=ReplayMode.HYSTERETIC_HANDOVER_SELECTION,
            quality_providers=providers,
        )

        self.manager = CameraManager(
            CameraManagerConfig(
                min_spatial_trust=float(self.get_parameter("min_spatial_trust").value),
                min_association_confidence=float(
                    self.get_parameter("min_association_confidence").value
                ),
                max_measurement_age_s=float(self.get_parameter("max_measurement_age_s").value),
                age_decay_s=float(self.get_parameter("age_decay_s").value),
                candidate_score_margin=float(self.get_parameter("candidate_score_margin").value),
                required_consecutive_better_frames=int(
                    self.get_parameter("required_consecutive_better_frames").value
                ),
                max_cross_camera_disagreement_m=float(
                    self.get_parameter("max_cross_camera_disagreement_m").value
                ),
                max_overlap_time_delta_s=float(
                    self.get_parameter("max_overlap_time_delta_s").value
                ),
                require_consistency_when_source_available=bool(
                    self.get_parameter("require_consistency_when_source_available").value
                ),
                fallback_on_active_camera_loss=bool(
                    self.get_parameter("fallback_on_active_camera_loss").value
                ),
                allowed_camera_ids=tuple(self.camera_ids),
            )
        )
        self.handover_config = HandoverUncertaintyConfig()
        self._latest: dict[str, CameraObservation] = {}
        self.require_source_batch_id = bool(
            self.get_parameter("require_source_batch_id").value
        )
        self.bootstrap_min_cameras = int(
            self.get_parameter("bootstrap_min_cameras").value
        )
        self.bootstrap_max_disagreement_m = float(
            self.get_parameter("bootstrap_max_disagreement_m").value
        )
        self.use_task_start_as_bootstrap_prior = bool(
            self.get_parameter("use_task_start_as_bootstrap_prior").value
        )
        self.bootstrap_prior_counts_as_support = bool(
            self.get_parameter("bootstrap_prior_counts_as_support").value
        )
        bootstrap_prior = tuple(float(value) for value in
                                self.get_parameter("bootstrap_prior_xyyaw").value)
        if len(bootstrap_prior) != 3 or not all(math.isfinite(value) for value in bootstrap_prior):
            raise ValueError("bootstrap_prior_xyyaw must contain finite [x,y,yaw]")
        self.bootstrap_prior_pose = bootstrap_prior if self.use_task_start_as_bootstrap_prior else None
        if self.bootstrap_prior_counts_as_support and self.bootstrap_prior_pose is None:
            raise ValueError(
                "bootstrap_prior_counts_as_support requires "
                "use_task_start_as_bootstrap_prior=true"
            )
        if self.bootstrap_min_cameras < 1:
            raise ValueError("bootstrap_min_cameras must be at least one")
        if not math.isfinite(self.bootstrap_max_disagreement_m) or self.bootstrap_max_disagreement_m <= 0.0:
            raise ValueError("bootstrap_max_disagreement_m must be finite and positive")
        self._outcome_journal = OutcomeJournal(
            journal_path(str(self.get_parameter("outcome_journal_path").value), self._manager_epoch),
            self._manager_epoch, max_bytes=int(self.get_parameter("outcome_journal_max_bytes").value))
        self.batch_outcome_pub = self.create_publisher(String, "/reliability/camera_manager/batch_outcome",
            QoSProfile(depth=OUTCOME_HISTORY_DEPTH, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL))
        terminal_qos = QoSProfile(
            depth=16,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._terminal_stop_ack_pub = self.create_publisher(
            String, TERMINAL_STOP_ACK_TOPIC, terminal_qos
        )
        self.create_subscription(
            String,
            TERMINAL_STOP_REQUEST_TOPIC,
            self._terminal_stop_request_cb,
            terminal_qos,
        )
        self._source_batch_buffer = SourceBatchBuffer(
            self.camera_ids,
            timeout_s=float(self.get_parameter("source_batch_timeout_wall_s").value),
            capacity=int(self.get_parameter("source_batch_capacity").value),
            on_event=self._publish_batch_outcome,
        )
        # Completed detector transactions are events, not sampled state.  A single
        # ``latest`` slot loses a physical camera round whenever detector completion
        # happens twice between manager timer callbacks.  Retain every completed
        # transaction until one decision consumes it.  Capacity exhaustion is an
        # integrity failure (reported and raised), never permission to drop the oldest
        # camera measurement silently.
        self._completed_source_batches = deque()
        self._completed_source_batch_capacity = int(
            self.get_parameter("source_batch_capacity").value
        )
        self._batch_clock_high_water_s = None
        self._ready_source_batch_id: str | None = None
        self._ready_source_batch_stamp_s = -math.inf
        self._last_decided_source_batch_id: str | None = None
        # Counter for observations that arrive with no detector batch identity. Only
        # reachable when require_source_batch_id is false; a paper run sets it true,
        # because a correction that cannot be traced to one detector invocation
        # cannot be counted exactly once.
        self._unidentified_observation_generation = 0
        #: Cameras whose reading had no prior pose to gate against this round.
        #: Rebuilt by _map_observations; declared here so the attribute always exists.
        self._belief_query_history = deque(maxlen=400)
        self._belief_prediction_history = deque(maxlen=400)
        self._admission_beliefs = AdmissionBeliefHistory(self.frame_id)
        self._canonical_belief_seen = False
        self._has_operational_anchor = bool(self.get_parameter("assume_initial_belief_anchor").value)
        self.odometry_frame_id = str(self.get_parameter("odometry_frame_id").value)
        self.odometry_to_map_yaw_rad = float(self.get_parameter("odometry_to_map_yaw_rad").value)
        # Measured poses only. Corrected-belief jumps cannot support displacement.
        self._odom_history = deque(maxlen=600)
        # Source-stamped odometry may arrive before this node has processed the
        # matching /clock sample. It is held here, not exposed to a snapshot, until
        # the local simulation clock reaches the source stamp.
        self._pending_odom_history: dict[int, MotionPose] = {}
        self._pending_odom_capacity = 4096
        self._reliability_query_source_by_camera: dict[str, str] = {}
        self._previous_camera_id: str | None = None
        self._previous_observation: MapObservation | None = None

        self.decision_pub = self.create_publisher(
            String, str(self.get_parameter("decision_topic").value), 10
        )
        self.selected_pub = self.create_publisher(
            PoseWithCovarianceStamped, str(self.get_parameter("selected_topic").value), 10
        )
        self.active_pub = None
        if self.authority == "active":
            self.active_pub = self.create_publisher(
                PoseWithCovarianceStamped, str(self.get_parameter("active_output_topic").value), 10
            )
            self.get_logger().warn(
                "authority=active: selected observations WILL be published to "
                f"{self.get_parameter('active_output_topic').value}. This is only "
                "valid after the ch.09 release gates pass."
            )

        self.fusion_mode = bool(self.get_parameter("fusion_mode").value)
        self.map_observations_pub = None
        if bool(self.get_parameter("publish_map_observations").value):
            self.map_observations_pub = self.create_publisher(
                String, str(self.get_parameter("map_observations_topic").value), 10
            )
        # PoseWithCovarianceStamped cannot carry the physical detector-batch
        # identity. Publish the evidence-grade correction contract beside the
        # legacy/display pose topic.
        self.fused_correction_pub = self.create_publisher(
            String, str(self.get_parameter("fused_correction_topic").value), 10
        )
        # 0 disables the median gate (METHOD amendment 2026-09-24 night): fuse every camera.
        gate = float(self.get_parameter("fusion_disagreement_gate_m").value)
        self.fusion_disagreement_gate_m = math.inf if gate == 0.0 else gate
        self._measurement_model_status_by_camera: dict[str, str] = {}
        # What the detector itself said about each reading, kept beside the reading so the
        # log can ask whether the detector's own confidence predicts how wrong it was.
        # Confidence is NOT used to weight anything here -- it is recorded and nothing else.
        self._detection_extras_by_camera: dict[str, dict[str, float]] = {}
        self.bias_floor_along_slope = float(
            self.get_parameter("bias_floor_along_slope_m_per_m").value)
        self.bias_floor_across_slope = float(
            self.get_parameter("bias_floor_across_slope_m_per_m").value)
        if self.bias_floor_along_slope < 0.0 or self.bias_floor_across_slope < 0.0:
            raise RuntimeError("bias floor slopes must be non-negative")
        if (self.bias_floor_along_slope > 0.0) != (self.bias_floor_across_slope > 0.0):
            # Refused here rather than at the first fusion, where it surfaced as a
            # singular floor and killed the manager mid-run.
            raise RuntimeError(
                "the bias floor needs both slopes positive or both zero: "
                f"along={self.bias_floor_along_slope}, "
                f"across={self.bias_floor_across_slope}. One positive slope is an "
                "ellipse with a zero axis, which no filter can use."
            )
        self.fusion_max_timestamp_spread_s = float(
            self.get_parameter("fusion_max_timestamp_spread_s").value
        )
        if (
            not math.isfinite(self.fusion_max_timestamp_spread_s)
            or self.fusion_max_timestamp_spread_s < 0.0
        ):
            raise ValueError("fusion_max_timestamp_spread_s must be finite and non-negative")
        self.covariance_profile = str(
            self.get_parameter("covariance_profile").value
        ).strip().lower()
        if self.covariance_profile not in SUPPORTED_COVARIANCE_PROFILES:
            raise ValueError(
                "covariance_profile must be one of "
                f"{SUPPORTED_COVARIANCE_PROFILES}, got {self.covariance_profile!r}"
            )
        self.commissioned_sigma_px = 0.0
        self.fusion_common_mode_std_m = float(
            self.get_parameter("fusion_common_mode_std_m").value)
        # The commissioned sensor model, the only one: R_pix from the frozen calibration.
        self.commissioned_pixel_cov_by_camera: dict[str, tuple] = {}
        self.per_camera_sigma = bool(
            self.get_parameter("commissioned_per_camera_sigma").value)
        stated = float(self.get_parameter("commissioned_sigma_px").value)
        path = str(self.get_parameter("commissioned_calibration_path").value).strip()
        if self.covariance_profile in (
            COMMISSIONED_VISIBILITY_COVARIANCE,
            COMMISSIONED_PERCEPTION_COVARIANCE,
        ):
            # Used only to establish the raw ground intersection. The selected candidate's
            # matched world-plane covariance replaces it before fusion.
            self.commissioned_sigma_px = 1.0
            source = "unit intermediary; replaced by frozen world-plane covariance"
        elif stated > 0.0:
            self.commissioned_sigma_px = stated
            source = "the commissioned_sigma_px parameter"
        elif path:
            self.commissioned_sigma_px = load_commissioned_sigma_px(path)
            source = path
        else:
            raise ValueError(
                f"covariance_profile={self.covariance_profile} needs either "
                "commissioned_calibration_path or commissioned_sigma_px; refusing to "
                "invent the detector's noise"
            )
        self.commissioned_pixel_cov = commissioned_pixel_covariance(self.commissioned_sigma_px)
        self.commissioned_pixel_cov_by_camera = {}
        if self.per_camera_sigma and path:
            for camera_id, sigma in load_commissioned_sigma_px_by_camera(path).items():
                self.commissioned_pixel_cov_by_camera[camera_id] = (
                    commissioned_pixel_covariance(sigma))
            if self.commissioned_pixel_cov_by_camera:
                listing = ", ".join(
                    f"{k}={v:.3f}" for k, v in
                    sorted(load_commissioned_sigma_px_by_camera(path).items()))
                self.get_logger().info(
                    f"per-camera detector noise in use: {listing} px "
                    f"(the pooled {self.commissioned_sigma_px:.4f} px is the fallback "
                    f"for any camera commissioning did not measure)")
        if self.covariance_profile not in (
            COMMISSIONED_VISIBILITY_COVARIANCE,
            COMMISSIONED_PERCEPTION_COVARIANCE,
        ):
            self.get_logger().info(
                f"covariance_profile={self.covariance_profile}: R_pix = "
                f"({self.commissioned_sigma_px:.4f} px)^2 I from {source}, pushed through "
                f"each camera's geometry before any profile-specific replacement"
            )

        # Candidate-specific world-plane covariance is loaded only when selected.
        self.commissioned_world_bias: dict = {}
        self.commissioned_world_table: dict = {}
        self.commissioned_world_edges: list = []
        if self.covariance_profile == COMMISSIONED_WORLD_COVARIANCE:
            world_path = str(
                self.get_parameter("commissioned_world_covariance_path").value).strip()
            if not world_path:
                raise ValueError(
                    "covariance_profile=commissioned_world_R needs "
                    "commissioned_world_covariance_path; refusing to invent the "
                    "measurement covariance"
                )
            (self.commissioned_world_bias, self.commissioned_world_table,
             self.commissioned_world_edges, _floor) = (
                load_commissioned_world_covariance(world_path))
            widths = sorted(
                math.sqrt((m[0][0] + m[1][1]) / 2.0)
                for m in self.commissioned_world_table.values())
            self.get_logger().info(
                f"covariance_profile=commissioned_world_R: {len(self.commissioned_world_table)} "
                f"(camera, confidence band) covariances from {world_path}; stated "
                f"standard deviation spans {widths[0]*100:.1f}-{widths[-1]*100:.1f} cm, "
                f"and the projected pixel noise is not used"
            )

        self.timestamp_compensation = bool(
            self.get_parameter("correction_timestamp_compensation").value)
        self.propagation_drift_std = float(
            self.get_parameter("correction_propagation_drift_std_m_per_s").value)
        self.correction_residual_interval_s = float(
            self.get_parameter("correction_residual_interval_s").value)
        if self.correction_residual_interval_s < 0.0:
            raise ValueError("correction_residual_interval_s must be non-negative")
        if self.propagation_drift_std < 0.0:
            raise ValueError("correction_propagation_drift_std_m_per_s must be non-negative")
        if self.timestamp_compensation:
            self.get_logger().info(
                "correction_timestamp_compensation=true: corrections are carried forward from "
                "the pose they describe to the pose they are used on")
        self._propagation_status = ""

        self.fusion_rule = str(self.get_parameter("fusion_rule").value).strip().lower()
        if self.fusion_rule not in SUPPORTED_FUSION_RULES:
            raise ValueError(
                f"fusion_rule must be one of {SUPPORTED_FUSION_RULES}, "
                f"got {self.fusion_rule!r}"
            )
        if not self.fusion_mode:
            raise ValueError(
                f"fusion_rule={self.fusion_rule} needs fusion_mode=true; with selection "
                "only one camera is ever used and the rule would never run"
            )

        self.sensor_gate_config = None
        sensor_gate_path = str(
            self.get_parameter("sensor_gate_config_path").value or ""
        ).strip()
        if sensor_gate_path:
            path = Path(sensor_gate_path).expanduser().resolve()
            actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            expected_sha256 = str(
                self.get_parameter("sensor_gate_config_expected_sha256").value or ""
            ).strip()
            if expected_sha256 and actual_sha256 != expected_sha256:
                raise ValueError(
                    "sensor gate config SHA-256 mismatch: "
                    f"expected {expected_sha256}, got {actual_sha256}"
                )
            self.sensor_gate_config = UsableObservationGateConfig.from_yaml(str(path))
            self.sensor_gate_config.assert_belief_independent()
            self.get_logger().info(
                "loaded deterministic sensor gate "
                f"{self.sensor_gate_config.gate_id}; sha256={actual_sha256}"
            )
        if self.sensor_gate_config is None:
            self.get_logger().warn(
                "sensor_gate_config_path is empty: projected detections are retained without "
                "the thesis sensor gate; this is valid only for raw commissioning capture")
        self._gate_rejections = collections.Counter()

        self.observation_model = str(
            self.get_parameter("observation_model").value
        ).strip().lower()
        if self.observation_model not in SUPPORTED_OBSERVATION_MODELS:
            raise ValueError(
                "observation_model must be one of "
                f"{SUPPORTED_OBSERVATION_MODELS}, got {self.observation_model!r}"
            )
        #: Set only for the learned models; None means no learned correction is loaded.
        self.learned_correction = None
        self.visibility_sensor_model = None
        self.perception_sensor_model = None
        self.reference_calibration = None
        self._learned_gate_counts = collections.Counter()
        if self.observation_model == OBSERVATION_MODEL_LEARNED_NN:
            artifact = str(self.get_parameter("learned_correction_path").value or "")
            if not artifact:
                raise ValueError(
                    f"observation_model={self.observation_model} needs "
                    f"learned_correction_path; the model is a commissioned artifact, not a "
                    f"default")
            from reliability.learned_box_correction import LearnedBoxCorrection
            # Fail at startup, not per reading: a drive that silently ran without the
            # correction would look like the arm it is meant to be compared against.
            self.learned_correction = LearnedBoxCorrection(artifact,
                expected_sha256=str(self.get_parameter("learned_correction_expected_sha256").value) or None)
            self.get_logger().warn(
                f"observation_model={self.observation_model}: neural box correction loaded "
                f"from {artifact}"
            )
        if (self.observation_model == OBSERVATION_MODEL_VISIBILITY_PATCH
                or self.covariance_profile == COMMISSIONED_VISIBILITY_COVARIANCE):
            if not (self.observation_model == OBSERVATION_MODEL_VISIBILITY_PATCH
                    and self.covariance_profile == COMMISSIONED_VISIBILITY_COVARIANCE):
                raise ValueError(
                    "observation_model=visibility_patch and "
                    "covariance_profile=commissioned_visibility_r must be selected together"
                )
            artifact = str(self.get_parameter("visibility_sensor_model_path").value or "")
            if not artifact:
                raise ValueError("visibility_patch needs visibility_sensor_model_path")
            from reliability.commissioned_visibility import CommissionedVisibilitySensorModel
            self.visibility_sensor_model = CommissionedVisibilitySensorModel(
                artifact,
                expected_sha256=(
                    str(self.get_parameter("visibility_sensor_model_expected_sha256").value)
                    or None
                ),
            )
            self.get_logger().info(
                "loaded image-residual correction and matched covariance from "
                f"{artifact}; sha256={self.visibility_sensor_model.sha256}"
            )
        perception_models = {
            OBSERVATION_MODEL_GLOBAL_RESIDUAL,
            OBSERVATION_MODEL_PER_CAMERA_RESIDUAL,
            OBSERVATION_MODEL_HIERARCHICAL_RESIDUAL,
            OBSERVATION_MODEL_SPATIAL_RESIDUAL,
            OBSERVATION_MODEL_JOINT_RGB_GAUSSIAN,
        }
        if self.observation_model in perception_models:
            if self.covariance_profile != COMMISSIONED_PERCEPTION_COVARIANCE:
                raise ValueError(
                    f"observation_model={self.observation_model} requires "
                    "covariance_profile=commissioned_perception_r"
                )
            artifact = str(self.get_parameter("perception_sensor_model_path").value or "")
            if not artifact:
                raise ValueError(f"observation_model={self.observation_model} needs perception_sensor_model_path")
            from reliability.commissioned_perception import CommissionedPerceptionSensorModel
            self.perception_sensor_model = CommissionedPerceptionSensorModel(
                artifact,
                expected_sha256=(
                    str(self.get_parameter("perception_sensor_model_expected_sha256").value)
                    or None
                ),
            )
            if self.perception_sensor_model.method != self.observation_model:
                raise ValueError(
                    f"configured observation model {self.observation_model} does not match "
                    f"artifact method {self.perception_sensor_model.method}"
                )
            self.get_logger().info(
                f"loaded commissioned perception method {self.observation_model} from "
                f"{artifact}; sha256={self.perception_sensor_model.sha256}"
            )
        if self.covariance_profile == COMMISSIONED_REFERENCE_COVARIANCE:
            if self.observation_model != OBSERVATION_MODEL_LEARNED_NN:
                raise ValueError('commissioned_reference_r requires observation_model=learned_nn')
            from reliability.reference_calibration import ReferenceCalibration
            self.reference_calibration = ReferenceCalibration(
                str(self.get_parameter('commissioned_world_covariance_path').value),
                str(self.get_parameter('learned_correction_path').value), self.camera_models.keys(),
                loaded_mean_sha256=self.learned_correction.sha256,
                expected_sha256=str(self.get_parameter("commissioned_world_covariance_expected_sha256").value) or None)
            self.get_logger().info(
                f'NN reference calibration {self.reference_calibration.sha256}: '
                'subtract residual mean after NN, use frozen full metric R')
        self.get_logger().info(
            f"observation_model={self.observation_model}: raw YOLO box-bottom projection "
            "is the common measurement base"
        )

        if self.fusion_mode:
            self.get_logger().warn("fusion_mode=true: publishing covariance-weighted FUSION of all in-view cameras to /state/bev")

        self.reliability_query_max_time_delta_s = float(
            self.get_parameter("reliability_query_max_time_delta_s").value
        )
        if (
            not math.isfinite(self.reliability_query_max_time_delta_s)
            or self.reliability_query_max_time_delta_s < 0.0
        ):
            raise ValueError(
                "reliability_query_max_time_delta_s must be finite and non-negative"
            )
        self.create_subscription(
            PoseWithCovarianceStamped,
            str(self.get_parameter("reliability_query_topic").value),
            self._belief_query_callback,
            20,
        )
        self.create_subscription(String, str(self.get_parameter("belief_state_topic").value),
                                 self._belief_state_callback, 20)
        if self.timestamp_compensation or self.fusion_mode:
            from nav_msgs.msg import Odometry  # noqa: PLC0415
            self.create_subscription(
                Odometry, str(self.get_parameter("odometry_topic").value),
                self._odom_callback, 50)

        for camera_id in self.camera_ids:
            topic = template.format(camera_id=camera_id)
            self.create_subscription(String, topic, self._observation_callback(camera_id), 10)

        rate = max(0.1, float(self.get_parameter("decision_rate_hz").value))
        # Expiry must continue when simulation time pauses or all publishers vanish.
        self._batch_wall_clock = Clock(clock_type=ClockType.STEADY_TIME)
        # Decisions too: the lockstep scheduler holds the simulation clock until it
        # sees the decision for the batch it just stepped, so a simulation-clock
        # timer would never fire. Each completed batch is still decided exactly once,
        # and under lockstep at the paused simulation time, independent of the phase.
        self.create_timer(1.0 / rate, self._decide, clock=self._batch_wall_clock)
        self.create_timer(0.25, self._expire_source_batches, clock=self._batch_wall_clock)

    def _publish_batch_outcome(self, event) -> None:
        if getattr(self, "_terminal_stopped", False):
            return
        payload = self._outcome_journal.append(dict(event, stage="manager",
            publish_stamp_s=self.get_clock().now().nanoseconds * 1e-9))
        message = String()
        message.data = json.dumps(payload, sort_keys=True, allow_nan=False)
        # The append-only journal and topic retain the event.  Keep the duplicate
        # console representation available at DEBUG without making evidence runs
        # I/O-bound at five cameras and 5 Hz.
        self.get_logger().debug("camera_batch_outcome " + message.data)
        if getattr(self, "_outcome_transport_enabled", True):
            self.batch_outcome_pub.publish(message)

    def _publish_decision(self, payload):
        payload = _diagnostic_values(payload)
        self._publish_batch_outcome(dict(source_batch_id=payload.get("source_batch_id"),
            status="manager_decision", decision=payload))
        message = String()
        message.data = json.dumps(payload, sort_keys=True, allow_nan=False)
        self.decision_pub.publish(message)

    def _check_batch_clock(self) -> None:
        stamp = self.get_clock().now().nanoseconds * 1e-9
        previous = self._batch_clock_high_water_s
        if previous is not None and stamp < previous - 0.005:
            self._publish_batch_outcome(dict(status="clock_reset", reason="coordinated runtime restart required"))
            raise RuntimeError("simulation clock moved backwards; coordinated runtime restart required")
        self._batch_clock_high_water_s = stamp if previous is None else max(stamp, previous)

    def _expire_source_batches(self) -> None:
        if getattr(self, "_terminal_stopped", False):
            return
        with self._input_lock:
            self._check_batch_clock()
            self._source_batch_buffer.expire(time.monotonic())

    def _observation_callback(self, expected_camera_id: str):
        def callback(message) -> None:
            with self._input_lock:
                self._receive_observation(expected_camera_id, message)
        return callback

    def _receive_observation(self, expected_camera_id, message):
        if getattr(self, "_terminal_stopped", False):
            return
        self._check_batch_clock()
        try:
            observation = CameraObservation.from_json(message.data)
        except (ContractValidationError, ValueError, TypeError) as exc:
            self._publish_batch_outcome(dict(status="member_rejected", camera_id=expected_camera_id,
                                             reason=f"invalid_contract:{exc}"))
            return
        source_batch_id = observation.source_batch_id
        reason = None
        if observation.camera_id != expected_camera_id:
            reason = "topic_camera_mismatch"
        elif self.require_source_batch_id and (not source_batch_id or observation.capture_stamp_ns is None):
            reason = "missing_physical_source_identity"
        if reason is not None:
            self._publish_batch_outcome(dict(source_batch_id=source_batch_id, status="member_rejected",
                                             camera_id=expected_camera_id, reason=reason))
            return
        if source_batch_id:
            complete = self._source_batch_buffer.offer(observation, time.monotonic())
            if complete is not None:
                epochs = {o.producer_epoch for o in complete.values()}
                if len(epochs) != 1 or any(o.source_batch_id != source_batch_id for o in complete.values()):
                    raise ContractValidationError("detector batch mixes source epochs or cycle identities")
                queue = getattr(self, "_completed_source_batches", None)
                if queue is None:
                    queue = self._completed_source_batches = deque()
                capacity = getattr(
                    self, "_completed_source_batch_capacity", self._source_batch_buffer.capacity
                )
                if len(queue) >= capacity:
                    self._publish_batch_outcome(dict(
                        source_batch_id=source_batch_id,
                        status="completed_queue_overflow",
                        pending_completed_batches=len(queue),
                    ))
                    raise RuntimeError(
                        "completed camera-batch queue overflow; refusing to skip a physical frame"
                    )
                queue.append((source_batch_id, dict(complete)))
                if len(queue) == 1:
                    self._latest = dict(complete)
                    self._ready_source_batch_id = source_batch_id
                    self._ready_source_batch_stamp_s = max(
                        o.timestamp_s for o in complete.values()
                    )
            return
        self._latest[expected_camera_id] = observation
        self._unidentified_observation_generation += 1
        self._ready_source_batch_id = f"unidentified:{self._unidentified_observation_generation}"

    def _odom_callback(self, message) -> None:
        """Retain finite measured poses in one declared frame and process epoch."""
        if getattr(self, "_terminal_stopped", False):
            return
        stamp_ns = _message_stamp_ns(message)
        frame = message.header.frame_id
        if frame != self.odometry_frame_id:
            self._publish_batch_outcome(dict(status="motion_rejected", reason="wrong_odometry_frame",
                                             frame_id=frame, capture_stamp_ns=stamp_ns))
            return
        if frame != self.frame_id and not math.isfinite(self.odometry_to_map_yaw_rad):
            self._publish_batch_outcome(dict(status="motion_rejected", reason="undeclared_odometry_transform",
                                             frame_id=frame, capture_stamp_ns=stamp_ns))
            return
        now_ns = self.get_clock().now().nanoseconds
        sample = MotionPose(stamp_ns,
                            (float(message.pose.pose.position.x), float(message.pose.pose.position.y)),
                            _message_yaw(message), frame, self._manager_epoch)
        with self._input_lock:
            by_stamp = {p.stamp_ns: p for p in self._odom_history}
            if stamp_ns in by_stamp and by_stamp[stamp_ns] != sample:
                raise ContractValidationError("conflicting odometry poses at the same stamp")
            pending = self._pending_odom_history
            if stamp_ns in pending and pending[stamp_ns] != sample:
                raise ContractValidationError("conflicting pending odometry poses at the same stamp")
            if stamp_ns not in by_stamp and stamp_ns not in pending:
                if len(pending) >= self._pending_odom_capacity:
                    self._publish_batch_outcome(dict(
                        status="motion_rejected", reason="pending_odometry_overflow",
                        capture_stamp_ns=stamp_ns))
                    return
                pending[stamp_ns] = sample
            self._flush_pending_odometry_locked(now_ns)
        if stamp_ns > now_ns:
            self._publish_batch_outcome(dict(
                status="motion_buffered", reason="clock_delivery_reordering",
                capture_stamp_ns=stamp_ns, local_clock_stamp_ns=now_ns))

    def _flush_pending_odometry_locked(self, now_ns: int) -> None:
        """Expose buffered poses only after their source time is causally current."""
        pending = getattr(self, '_pending_odom_history', None)
        if not pending:
            return
        by_stamp = {p.stamp_ns: p for p in self._odom_history}
        for stamp_ns in sorted(t for t in pending if t <= int(now_ns)):
            sample = pending.pop(stamp_ns)
            if stamp_ns in by_stamp and by_stamp[stamp_ns] != sample:
                raise ContractValidationError("conflicting odometry poses at the same stamp")
            by_stamp[stamp_ns] = sample
        capacity = self._odom_history.maxlen
        self._odom_history = deque(
            (by_stamp[t] for t in sorted(by_stamp)[-capacity:]), maxlen=capacity)

    def _motion_snapshot(self):
        """Caller holds the input lock; no live buffers escape into alignment."""
        self._flush_pending_odometry_locked(self.get_clock().now().nanoseconds)
        yaw = 0. if self.odometry_frame_id == self.frame_id else self.odometry_to_map_yaw_rad
        # Missing declared transform cannot have supplied valid callback samples.
        if not math.isfinite(yaw):
            return None
        return MotionPoseSnapshot.capture(tuple(self._odom_history), target_frame=self.frame_id,
            source_frame=self.odometry_frame_id, epoch=self._manager_epoch, source_to_target_yaw=yaw)

    def _snapshot_inputs(self, source_batch_id):
        with self._input_lock:
            belief = self._admission_beliefs.latest
            identity = None if belief is None else (
                belief.epoch, belief.revision, belief.anchor_stamp_ns, belief.state_stamp_ns)
            return ManagerInputs(source_batch_id, tuple(self._latest[c] for c in sorted(self._latest)),
                                 tuple(self._belief_query_history), self._has_operational_anchor,
                                 identity, self._motion_snapshot(),
                                 tuple(self._belief_prediction_history))

    def _belief_state_callback(self, message) -> None:
        with self._input_lock:
            self._canonical_belief_seen = True
            try:
                payload = json.loads(message.data)
                if self._admission_beliefs.offer(payload, now_ns=self.get_clock().now().nanoseconds):
                    self._belief_query_history = deque(self._admission_beliefs.poses(), maxlen=400)
                    self._belief_prediction_history = deque(
                        self._admission_beliefs.predictions(), maxlen=400)
                    self._has_operational_anchor = self._admission_beliefs.latest.initialized
            except (TypeError, ValueError) as exc:
                self._admission_beliefs.invalidate(exc)
                self._belief_query_history.clear()
                self._belief_prediction_history.clear()
                # Unknown/malformed state is never cold-bootstrap permission.
                self._has_operational_anchor = True
                raise

    def _belief_query_callback(self, message) -> None:
        # Legacy producers lack revisions. Once the canonical stream is bound,
        # a compatibility topic can never restore an old or invalid prior.
        if getattr(self, "_canonical_belief_seen", False):
            return
        if message.header.frame_id != self.frame_id:
            self.get_logger().warn(
                "ignoring reliability query state in frame "
                f"{message.header.frame_id!r}; expected {self.frame_id!r}"
            )
            return
        timestamp_s = _message_stamp_ns(message) / 1e9
        # Yaw is carried beside x/y for covariance queries and belief propagation.
        yaw = _message_yaw(message)
        pose = (
            float(message.pose.pose.position.x),
            float(message.pose.pose.position.y),
            yaw,
        )
        if math.isfinite(timestamp_s) and all(math.isfinite(value) for value in pose):
            with getattr(self, "_input_lock", nullcontext()):
                by_stamp = dict(self._belief_query_history)
                by_stamp[timestamp_s] = pose
                capacity = self._belief_query_history.maxlen or 400
                self._belief_query_history = deque(sorted(by_stamp.items())[-capacity:], maxlen=capacity)
                self._has_operational_anchor = True

    def _map_observations(self, now_s: float) -> list[MapObservation]:
        observations: list[MapObservation] = []
        self._detection_extras_by_camera = {}
        self._measurement_model_status_by_camera = {}
        self._reliability_query_source_by_camera = {}
        self._camera_mapping_reasons = {}
        inputs = self._decision_snapshot or self._snapshot_inputs(self._ready_source_batch_id)
        for contract in inputs.contracts:
            camera_id = contract.camera_id
            try:
                validate_observation_geometry(contract, self.camera_models[camera_id],
                    expected_camera_id=camera_id,
                    expected_calibration_id=self.camera_calibration_ids_by_camera[camera_id],
                    expected_image_frame_id=self.camera_image_frame_ids_by_camera[camera_id],
                    require_bbox=True)
            except (ValueError, TypeError) as exc:
                self._camera_mapping_reasons[camera_id] = f"invalid_observation_geometry:{exc}"
                self._measurement_model_status_by_camera[camera_id] = "refused_invalid_observation_geometry"
                continue
            # The detector's noise is one commissioned number in PIXELS, identical for
            # every camera. Whatever pixel covariance the contract arrived with is
            # replaced by it here, so no arm can be driven against a remembered value,
            # and the geometry alone decides what it is worth in centimetres.
            stated = replace(contract, conditional_cov_uv=(
                self.commissioned_pixel_cov_by_camera.get(
                    camera_id, self.commissioned_pixel_cov)))
            projected = project_observation_to_world_with_covariance(
                stated, self.camera_models[camera_id]
            )
            if projected is None:
                self._camera_mapping_reasons[camera_id] = "detector_miss" if not contract.detection_valid else "projection_unavailable"
                continue
            world_xy, covariance_m2 = projected
            if self.sensor_gate_config is not None:
                bbox = contract.bbox_xyxy
                gate_result = evaluate_sensor_gate({
                    "frame_expected": True,
                    "frame_received": True,
                    "frame_age_ms": max(
                        0.0, 1000.0 * (float(now_s) - float(contract.timestamp_s))
                    ),
                    "detection_received": bool(contract.detection_valid),
                    "detector_class": "robot",
                    "detector_confidence": float(contract.detector_score),
                    "bbox_xmin": bbox[0] if bbox else None,
                    "bbox_ymin": bbox[1] if bbox else None,
                    "bbox_xmax": bbox[2] if bbox else None,
                    "bbox_ymax": bbox[3] if bbox else None,
                    "projection_valid": True,
                }, self.sensor_gate_config)
                if not gate_result.admitted:
                    reason = f"sensor_gate:{gate_result.reason}"
                    self._gate_rejections[reason] += 1
                    self._measurement_model_status_by_camera[camera_id] = f"refused_{reason}"
                    self._camera_mapping_reasons[camera_id] = reason
                    continue
            # commissioned_world_R replaces the projected pixel covariance with the
            # residual scatter measured on the warehouse floor for this camera at this
            # detector confidence, and subtracts the systematic offset commissioned with
            # it. Both come from the same artifact, so a corrected reading is never paired
            # with an uncorrected covariance.
            if self.covariance_profile == COMMISSIONED_WORLD_COVARIANCE:
                # `detector_score` is the YOLO confidence, the same quantity the offline
                # confidence bands were fitted on (set from the detector's own score in
                # scheduled_camera_detector_node).
                band = commissioned_world_band(
                    contract.detector_score, self.commissioned_world_edges)
                entry = self.commissioned_world_table.get((camera_id, band))
                theta = self.commissioned_world_bias.get(camera_id)
                if entry is None or theta is None:
                    # No commissioned statement for this camera and confidence: refuse
                    # rather than fall back on a covariance measured somewhere else.
                    self._gate_rejections["no_commissioned_world_covariance"] += 1
                    self._camera_mapping_reasons[camera_id] = "no_commissioned_world_covariance"
                    continue
                covariance_m2 = entry
                if contract.bbox_xyxy is not None:
                    height = max(
                        1.0, float(contract.bbox_xyxy[3]) - float(contract.bbox_xyxy[1]))
                    features = (1.0, 1.0 / height)
                    world_xy = (
                        float(world_xy[0])
                        - (features[0] * theta[0][0] + features[1] * theta[1][0]),
                        float(world_xy[1])
                        - (features[0] * theta[0][1] + features[1] * theta[1][1]),
                    )
            # The UNCORRECTED back-projection, kept before any observation model rewrites
            # `world_xy`. Recorded so a drive can be re-interpreted offline: without it the
            # only reading in the log is the one the steering model already corrected, so a
            # different interpretation cannot be replayed on the same drive and every
            # comparison would have to run its own trajectory. Diagnostic only -- nothing
            # downstream reads it, and no arm is driven against it.
            raw_world_xy = (float(world_xy[0]), float(world_xy[1]))
            source = f"live_contract:{self.covariance_profile}"
            prior_prediction = _nearest_state_prediction(
                inputs.belief_predictions,
                contract.timestamp_s,
                max_delta_s=self.reliability_query_max_time_delta_s,
            )
            if prior_prediction is not None:
                prior_pose, prior_covariance = prior_prediction
            else:
                prior_pose = _nearest_state_pose(
                    inputs.belief_poses,
                    contract.timestamp_s,
                    max_delta_s=self.reliability_query_max_time_delta_s,
                )
                prior_covariance = None
            bootstrap_prior_used = False
            if (prior_pose is None and not inputs.has_anchor
                    and self.bootstrap_prior_pose is not None):
                prior_pose = self.bootstrap_prior_pose
                bootstrap_prior_used = True
            if self.perception_sensor_model is not None:
                try:
                    camera_model = self.camera_models[camera_id]
                    world_xy, covariance_m2 = self.perception_sensor_model.correct_and_covariance(
                        camera_id,
                        world_xy,
                        contract.bbox_xyxy,
                        float(contract.detector_score),
                        contract.rgb_context_crop_96x96_zlib_b64,
                        camera_model.cam_pos,
                        (camera_model.img_width, camera_model.img_height),
                    )
                except (ValueError, TypeError, ArithmeticError) as exc:
                    self._gate_rejections["perception_model_unavailable"] += 1
                    self._measurement_model_status_by_camera[camera_id] = (
                        "refused_perception_model_unavailable"
                    )
                    self._camera_mapping_reasons[camera_id] = (
                        f"perception_model_unavailable:{type(exc).__name__}"
                    )
                    continue
                source = f"{source}:{self.observation_model}"
                measurement_model_status = f"{self.observation_model}_applied"
            elif self.visibility_sensor_model is not None:
                heading_required = self.visibility_sensor_model.requires_capture_heading
                if (contract.visibility_grid_16x16 is None
                        or (heading_required and prior_pose is None)):
                    reason = (
                        "visibility_grid_unavailable" if contract.visibility_grid_16x16 is None
                        else "capture_heading_unavailable"
                    )
                    self._gate_rejections[reason] += 1
                    self._measurement_model_status_by_camera[camera_id] = f"refused_{reason}"
                    self._camera_mapping_reasons[camera_id] = reason
                    continue
                try:
                    world_xy, covariance_m2 = self.visibility_sensor_model.correct_and_covariance(
                        camera_id,
                        world_xy,
                        contract.bbox_xyxy,
                        float(contract.detector_score),
                        contract.visibility_grid_16x16,
                        None if prior_pose is None else float(prior_pose[2]),
                        projection_covariance_world=covariance_m2,
                    )
                except (ValueError, TypeError, ArithmeticError) as exc:
                    self._gate_rejections["visibility_model_unavailable"] += 1
                    self._measurement_model_status_by_camera[camera_id] = (
                        "refused_visibility_model_unavailable"
                    )
                    self._camera_mapping_reasons[camera_id] = (
                        f"visibility_model_unavailable:{type(exc).__name__}"
                    )
                    continue
                source = f"{source}:visibility_patch"
                measurement_model_status = "visibility_patch_applied"
            elif self.learned_correction is not None:
                corrected_xy = self.learned_correction.correct(
                    camera_id, world_xy, contract.bbox_xyxy,
                    float(contract.detector_score))
                if corrected_xy is None:
                    # A reading the model cannot describe is refused rather than passed
                    # through uncorrected, which would silently mix two interpretations.
                    self._gate_rejections["learned_correction_unavailable"] += 1
                    self._measurement_model_status_by_camera[camera_id] = (
                        "refused_learned_correction_unavailable")
                    self._camera_mapping_reasons[camera_id] = "learned_correction_unavailable"
                    continue
                world_xy = corrected_xy
                if getattr(self, 'reference_calibration', None) is not None:
                    world_xy, covariance_m2 = self.reference_calibration.apply(camera_id, world_xy)
                source = f"{source}:learned_nn"
                measurement_model_status = "learned_nn_applied"
            else:
                measurement_model_status = "raw_box_projection"
            self._measurement_model_status_by_camera[camera_id] = measurement_model_status
            bbox = contract.bbox_xyxy
            cam_pos = self.camera_models[camera_id].cam_pos
            self._detection_extras_by_camera[camera_id] = {
                "conf": float(contract.detector_score),
                "conf_raw": float(contract.detector_score_raw),
                "bbox_h_px": (float(bbox[3] - bbox[1]) if bbox is not None else float("nan")),
                "bbox_w_px": (float(bbox[2] - bbox[0]) if bbox is not None else float("nan")),
                "covariance_query_heading_rad": (
                    float(prior_pose[2]) if prior_pose is not None else float("nan")),
                "range_m": float(math.hypot(world_xy[0] - float(cam_pos[0]),
                                            world_xy[1] - float(cam_pos[1]))),
                "raw_obs_x": raw_world_xy[0],
                "raw_obs_y": raw_world_xy[1],
            }
            base = MapObservation(
                camera_id=camera_id,
                timestamp_s=contract.timestamp_s,
                xy_m=world_xy,
                covariance_m2=covariance_m2,
                quality=_contract_quality(contract),
                source=source,
            )
            if prior_pose is None:
                query_xy = world_xy
                query_source = "measurement_bootstrap"
            elif bootstrap_prior_used:
                query_xy = (prior_pose[0], prior_pose[1])
                query_source = "declared_task_start_bootstrap_prior"
            else:
                query_xy = (prior_pose[0], prior_pose[1])
                query_source = "timestamp_matched_planner_belief"
            self._reliability_query_source_by_camera[camera_id] = query_source
            # Runtime quality decorates the commissioned measurement. Replay's
            # alternate isotropic covariance model must not overwrite its R.
            provider = _with_provider_quality(base, self.replay_config, query_xy, now_s)
            if self.commissioned_availability is not None and prior_pose is not None:
                probability = self.commissioned_availability.probability(
                    camera_id,
                    float(prior_pose[0]), float(prior_pose[1]),
                )
                quality = replace(
                    provider.quality,
                    p_available=probability,
                    source_model="commissioned_q_i_p",
                )
                # Availability controls selection/planning quality, never the conditional
                # R of a measurement that was actually supplied. Keep the matched runtime R.
                provider = replace(base, quality=quality)
            observations.append(replace(base, quality=provider.quality))
        return observations

    def _decide(self) -> None:
        if getattr(self, "_terminal_stopped", False):
            return
        with self._decision_lock:
            # Drain exactly the transactions that were ready at entry. A transaction
            # arriving re-entrantly during publication remains queued for the next
            # callback, while a short detector burst cannot overwrite an earlier one.
            queued = getattr(self, "_completed_source_batches", None)
            if queued is None:
                # Compatibility for library/test harnesses constructed without the
                # ROS-node initializer. Production nodes always use the FIFO above.
                source_batch_id = None
                try:
                    with self._input_lock:
                        source_batch_id = self._ready_source_batch_id
                        if (source_batch_id is None
                                or source_batch_id == self._last_decided_source_batch_id):
                            return
                        self._last_decided_source_batch_id = source_batch_id
                        self._decision_snapshot = self._snapshot_inputs(source_batch_id)
                    self._decide_once(source_batch_id)
                except Exception as exc:
                    self._publish_batch_outcome(dict(
                        source_batch_id=source_batch_id,
                        status="decision_error", reason=str(exc)))
                    raise
                finally:
                    self._decision_snapshot = None
                self._publish_batch_outcome(dict(
                    source_batch_id=source_batch_id, status="decision_completed"))
                return
            ready_count = len(queued) if queued is not None else 1
            for _ in range(ready_count):
                self._decide_next_ready_batch()

    def _decide_next_ready_batch(self) -> None:
        """Consume one immutable completed detector transaction at most once."""
        source_batch_id = None
        try:
            with self._input_lock:
                queued = getattr(self, "_completed_source_batches", None)
                if queued is not None:
                    if not queued:
                        self._ready_source_batch_id = None
                        return
                    source_batch_id, contracts = queued.popleft()
                    self._latest = dict(contracts)
                    self._ready_source_batch_id = source_batch_id
                    self._ready_source_batch_stamp_s = max(
                        o.timestamp_s for o in contracts.values()
                    )
                else:
                    source_batch_id = self._ready_source_batch_id
                if source_batch_id is None or source_batch_id == self._last_decided_source_batch_id:
                    return
                # Claim before any publication. A partial publication failure may not
                # cause the same physical frames to be assimilated a second time.
                self._last_decided_source_batch_id = source_batch_id
                self._decision_snapshot = self._snapshot_inputs(source_batch_id)
            source_batch_id = None
            try:
                source_batch_id = self._decision_snapshot.source_batch_id
                self._decide_once(source_batch_id)
            except Exception as exc:
                self._publish_batch_outcome(dict(source_batch_id=source_batch_id,
                                                 status="decision_error", reason=str(exc)))
                raise
            finally:
                self._decision_snapshot = None
            self._publish_batch_outcome(dict(source_batch_id=source_batch_id,
                                             status="decision_completed"))
        finally:
            queued = getattr(self, "_completed_source_batches", None)
            if queued is not None:
                with self._input_lock:
                    if queued:
                        next_id, next_contracts = queued[0]
                        self._latest = dict(next_contracts)
                        self._ready_source_batch_id = next_id
                        self._ready_source_batch_stamp_s = max(
                            o.timestamp_s for o in next_contracts.values()
                        )
                    else:
                        self._ready_source_batch_id = None

    def _decide_once(self, source_batch_id: str) -> None:
        self._decision_now_ns = self.get_clock().now().nanoseconds
        now_s = self._decision_now_ns * 1.0e-9
        observations = self._map_observations(now_s)
        mapped = {o.camera_id: o for o in observations}
        for contract in self._decision_snapshot.contracts:
            reading = mapped.get(contract.camera_id)
            self._publish_batch_outcome(dict(source_batch_id=source_batch_id, status="camera_mapping",
                camera_id=contract.camera_id, source_frame_id=contract.source_frame_id,
                detector_invocation_id=contract.detector_invocation_id,
                capture_stamp_ns=contract.capture_stamp_ns,
                disposition="mapped" if reading is not None else "refused",
                reason="" if reading is not None else self._camera_mapping_reasons.get(contract.camera_id, "mapping_unavailable"),
                capture_observation=None if reading is None else reading.to_dict()))
        self._publish_map_observations(observations)
        if self.fusion_mode and self.active_pub is not None:
            self._decide_fused(now_s, observations, source_batch_id=source_batch_id)
            self._last_decided_source_batch_id = source_batch_id
            return
        decision = self.manager.select(timestamp_s=now_s, observations=observations)
        # The handover switch is REPORTED, never applied: the commissioned covariance is the
        # claim under test, so a camera switch may not quietly widen it. The diagnostic says
        # when a switch happened and by how much it would have been inflated.
        _, diagnostic = handover_adjusted_observation(
            previous_camera_id=self._previous_camera_id,
            selected_observation=decision.selected_observation,
            candidate_observations=tuple(observations),
            previous_observation=self._previous_observation,
            config=self.handover_config,
        )
        selected = decision.selected_observation
        if selected is not None:
            self._previous_camera_id = selected.camera_id
            self._previous_observation = selected

        payload = decision.to_dict()
        payload["authority"] = self.authority
        payload["source_batch_id"] = source_batch_id
        payload["covariance_profile"] = self.covariance_profile
        payload["handover_diagnostic"] = diagnostic.to_dict()
        payload["gp_query_source_by_camera"] = dict(
            self._reliability_query_source_by_camera
        )
        payload["measurement_model_status_by_camera"] = dict(
            self._measurement_model_status_by_camera
        )
        self._publish_decision(payload)

        if selected is None:
            self._last_decided_source_batch_id = source_batch_id
            return
        pose = self._pose_message(selected)
        self.selected_pub.publish(pose)
        if self.active_pub is not None:
            self.active_pub.publish(pose)
        self._last_decided_source_batch_id = source_batch_id

    def _publish_map_observations(self, observations: list[MapObservation]) -> None:
        """Emit the per-camera observations for a downstream sequential filter.

        Each keeps its own covariance and physical camera identity. The optional
        direct-camera filter consumes these independently of the manager's
        prior-free fused-measurement output.
        """
        if self.map_observations_pub is None:
            return
        message = String()
        message.data = map_observations_to_json(observations, frame_id=self.frame_id)
        self.map_observations_pub.publish(message)

    def _bias_floors(self, observations):
        """Per-camera bias floors for this batch, or None when the floor is disabled.

        Each floor is built from the sighting's own range and bearing, taken from the
        camera's surveyed position and the observation it produced, because the measured
        bias tracks the camera's line of sight rather than the world frame.
        """
        if self.bias_floor_along_slope <= 0.0 and self.bias_floor_across_slope <= 0.0:
            return None
        from reliability.bias_floor import bias_floor_matrix, ray_bearing_rad

        floors = {}
        for observation in observations:
            model = self.camera_models.get(observation.camera_id)
            if model is None:
                continue
            camera_xy = (float(model.cam_pos[0]), float(model.cam_pos[1]))
            target_xy = (float(observation.xy_m[0]), float(observation.xy_m[1]))
            try:
                bearing = ray_bearing_rad(camera_xy, target_xy)
            except Exception:                                  # noqa: BLE001
                continue
            range_m = math.hypot(target_xy[0] - camera_xy[0], target_xy[1] - camera_xy[1])
            floors[observation.camera_id] = bias_floor_matrix(
                range_m, bearing,
                along_slope=self.bias_floor_along_slope,
                across_slope=self.bias_floor_across_slope,
            )
        return floors or None

    def _decide_fused(
        self,
        now_s: float,
        observations: list[MapObservation],
        *,
        source_batch_id: str,
    ) -> None:
        # Camera callbacks retain one latest observation each. Fuse only views
        # that satisfy the same operational gates as selection and whose stamps
        # occupy one time neighbourhood; otherwise an old cached view is
        # incorrectly treated as a simultaneous measurement of the robot.
        fresh, scores, rejected, time_skew_rejected = _synchronous_fusion_candidates(
            self.manager,
            now_s=now_s,
            observations=observations,
            max_timestamp_spread_s=self.fusion_max_timestamp_spread_s,
        )
        if not fresh:
            payload = {"authority": self.authority, "fusion_mode": True,
                       "source_batch_id": source_batch_id,
                       "accepted_camera_ids": [], "reasons": ["no_eligible_synchronous_observations"],
                       "scores_by_camera": scores,
                       "rejected_by_camera": {key: list(value) for key, value in rejected.items()},
                       "time_skew_rejected_camera_ids": time_skew_rejected,
                       "gp_query_source_by_camera": dict(
                           self._reliability_query_source_by_camera
                       ),
                       "measurement_model_status_by_camera": dict(
                           self._measurement_model_status_by_camera
                       )}
            self._publish_decision(payload)
            return
        original_by_camera = {
            str(observation.camera_id): observation for observation in fresh
        }
        inputs = self._decision_snapshot or self._snapshot_inputs(source_batch_id)
        history = inputs.motion
        capture_ns = {c.camera_id: c.capture_stamp_ns if c.capture_stamp_ns is not None
                      else round(c.timestamp_s*1e9) for c in inputs.contracts}
        # Library-only callers may supply observations without detector contracts.
        capture_ns.update({o.camera_id: round(o.timestamp_s*1e9) for o in fresh
                           if o.camera_id not in capture_ns})
        common_time_support = {}
        common_capture_ns = max(capture_ns[o.camera_id] for o in fresh)
        fresh, common_time_rejected, common_capture_s = align_observations_to_common_time(
            fresh,
            history,
            max_pose_delta_s=self.reliability_query_max_time_delta_s,
            drift_std_m_per_s=self.propagation_drift_std,
            max_motion_gap_s=self.reliability_query_max_time_delta_s,
            frame_id=self.frame_id,
            stamp_ns_by_camera=capture_ns,
            support_by_camera=common_time_support,
        )
        all_aligned_observations = tuple(fresh)
        for camera_id in common_time_rejected:
            rejected[camera_id] = tuple(rejected.get(camera_id, ())) + (
                "common_time_propagation_unavailable",
            )
        bootstrap_evidence = None
        if not inputs.has_anchor and fresh:
            # Prior-free initialisation is allowed only when several independent cameras
            # agree.  A separately declared, surveyed task start may instead count as one
            # support source, but only when at least one admitted camera also agrees with it.
            # The prior can never initialise the camera belief by itself.
            #
            # It looks for the largest group that mutually agrees, not for unanimity.
            # Requiring every camera in view to agree lets one mis-associated camera
            # block start-up entirely, which is the opposite of what a quorum is for.
            bootstrap_prior = (
                self.bootstrap_prior_pose
                if self.bootstrap_prior_counts_as_support else None
            )
            agreeing = _bootstrap_agreeing_group(
                fresh,
                self.bootstrap_max_disagreement_m,
                prior_xy=(bootstrap_prior[:2] if bootstrap_prior is not None else None),
            )
            spread = _group_spread_m(fresh)
            required_camera_count = max(
                1,
                self.bootstrap_min_cameras - (1 if bootstrap_prior is not None else 0),
            )
            support_count = len(agreeing) + (1 if bootstrap_prior is not None else 0)
            prior_rejected = sorted(
                str(observation.camera_id)
                for observation in fresh
                if bootstrap_prior is not None
                and math.hypot(
                    float(observation.xy_m[0]) - float(bootstrap_prior[0]),
                    float(observation.xy_m[1]) - float(bootstrap_prior[1]),
                ) > self.bootstrap_max_disagreement_m
            )
            if len(agreeing) < required_camera_count:
                payload = {
                    "authority": self.authority,
                    "fusion_mode": True,
                    "source_batch_id": source_batch_id,
                    "accepted_camera_ids": [],
                    "reasons": ["bootstrap_quorum_failed"],
                    "bootstrap_camera_count": len(fresh),
                    "bootstrap_agreeing_count": len(agreeing),
                    "bootstrap_spread_m": float(spread),
                    "bootstrap_max_disagreement_m": float(self.bootstrap_max_disagreement_m),
                    "bootstrap_prior_used": bootstrap_prior is not None,
                    "bootstrap_prior_xy": (
                        [float(bootstrap_prior[0]), float(bootstrap_prior[1])]
                        if bootstrap_prior is not None else None
                    ),
                    "bootstrap_prior_rejected_camera_ids": prior_rejected,
                    "bootstrap_required_camera_count": required_camera_count,
                    "bootstrap_support_count": support_count,
                }
                self._publish_decision(payload)
                return
            # Only the agreeing group initialises the belief; a camera outside it is
            # not evidence about where the robot is.
            outside = sorted(
                str(o.camera_id) for o in fresh if o not in agreeing)
            for camera_id in outside:
                rejected[camera_id] = tuple(rejected.get(camera_id, ())) + (
                    "outside_bootstrap_agreeing_group",
                )
            fresh = agreeing
            bootstrap_evidence = {
                "prior_used": bootstrap_prior is not None,
                "prior_xy": (
                    [float(bootstrap_prior[0]), float(bootstrap_prior[1])]
                    if bootstrap_prior is not None else None
                ),
                "camera_ids": sorted(str(observation.camera_id) for observation in agreeing),
                "camera_count": len(agreeing),
                "support_count": support_count,
                "required_support_count": self.bootstrap_min_cameras,
                "max_disagreement_m": float(self.bootstrap_max_disagreement_m),
            }
        if not fresh:
            payload = {
                "authority": self.authority,
                "fusion_mode": True,
                "source_batch_id": source_batch_id,
                "accepted_camera_ids": [],
                "reasons": ["no_common_time_admitted_observations"],
            }
            self._publish_decision(payload)
            return
        result = _gated_fusion(
            fresh,
            disagreement_gate_m=self.fusion_disagreement_gate_m,
            rule=self.fusion_rule,
            camera_positions_m={
                camera_id: model.cam_pos
                for camera_id, model in self.camera_models.items()
            },
            belief_floors=self._bias_floors(fresh),
        )
        if not result.accepted_camera_ids:
            payload = {
                "authority": self.authority,
                "fusion_mode": True,
                "source_batch_id": source_batch_id,
                "accepted_camera_ids": [],
                "rejected_camera_ids": list(result.rejected_camera_ids),
                "reasons": ["all_synchronous_observations_rejected"],
                "scores_by_camera": scores,
                "rejected_by_camera": {
                    key: list(value) for key, value in rejected.items()
                },
                "time_skew_rejected_camera_ids": time_skew_rejected,
                "covariance_profile": self.covariance_profile,
                "gp_query_source_by_camera": dict(
                    self._reliability_query_source_by_camera
                ),
                "measurement_model_status_by_camera": dict(
                    self._measurement_model_status_by_camera
                ),
            }
            self._publish_decision(payload)
            return
        ts = float(common_capture_s)

        # The fused correction describes where the robot was when those frames were taken.
        # Carry it forward to the newest instant supported by measured odometry.  The manager
        # must not extrapolate to its callback time: ROS can deliver the image batch before the
        # odometry callback at that same simulation instant.  The recursive estimator replays
        # the small remaining interval from this stamped correction to its own current time.
        # Propagating only over measured support removes camera latency without inventing a
        # zero-motion tail and also permits the initial, stationary two-camera bootstrap.
        mean_xy, propagated_cov = result.mean_xy, result.covariance_m2
        correction_ns = common_capture_ns
        correction_motion_support = None
        self._propagation_status = "disabled"
        if self.timestamp_compensation:
            decision_ns = getattr(self, "_decision_now_ns", round(now_s*1e9))
            target_ns = (
                min(decision_ns, history.samples[-1].stamp_ns)
                if history is not None and history.samples
                else decision_ns
            )
            support = None if history is None else history.displacement(
                common_capture_ns, target_ns, max_gap_s=self.reliability_query_max_time_delta_s)
            if support is None or not support.supported:
                # Timestamp compensation is part of the estimator contract. Publishing an
                # uncompensated old measurement here creates a deterministic lag bias while
                # still labelling the campaign "compensated", so fail closed for this batch.
                payload = {
                    "authority": self.authority,
                    "fusion_mode": True,
                    "source_batch_id": source_batch_id,
                    "common_capture_stamp": float(common_capture_s),
                    "accepted_camera_ids": [],
                    "reasons": ["timestamp_compensation_pose_unavailable"],
                    "motion_support": None if support is None else support.to_dict(),
                    "would_accept_camera_ids": list(result.accepted_camera_ids),
                }
                self._publish_decision(payload)
                return
            mean_xy, propagated_cov, delta = propagate_correction_to_now(
                result.mean_xy, result.covariance_m2, (0., 0.), support.delta_xy_m,
                drift_std_m_per_s=self.propagation_drift_std,
                dt_s=max(target_ns * 1.0e-9 - ts, 0.0),
                residual_interval_s=self.correction_residual_interval_s)
            self._propagation_status = (
                "applied_to_latest_supported_odometry "
                f"age={max(target_ns * 1.0e-9 - ts, 0.0):.3f}s "
                f"dx={delta[0]:+.3f} dy={delta[1]:+.3f}")
            # It describes the newest odometry-supported instant, which can be slightly
            # behind the manager callback time.  Preserve that physical timestamp so the
            # estimator can replay the residual interval exactly once.
            ts = target_ns * 1.0e-9
            correction_ns = target_ns
            correction_motion_support = support.to_dict()

        message = PoseWithCovarianceStamped()
        message.header.stamp.sec, message.header.stamp.nanosec = divmod(correction_ns, 1_000_000_000)
        message.header.frame_id = self.frame_id
        message.pose.pose.position.x = float(mean_xy[0])
        message.pose.pose.position.y = float(mean_xy[1])
        message.pose.pose.orientation.w = 1.0
        cov = [0.0] * 36
        report_covariance = _fusion_report_covariance(
            propagated_cov, common_mode_std_m=self.fusion_common_mode_std_m
        )
        cov[0] = report_covariance[0][0]
        cov[1] = report_covariance[0][1]
        cov[6] = report_covariance[1][0]
        cov[7] = report_covariance[1][1]
        cov[35] = NONINFORMATIVE_YAW_VAR
        message.pose.covariance = cov
        payload = {"authority": self.authority, "fusion_mode": True,
                   "source_batch_id": source_batch_id,
                   "common_capture_stamp": float(common_capture_s),
                   "common_time_rejected_camera_ids": common_time_rejected,
                   "common_time_support_by_camera": common_time_support,
                   "accepted_camera_ids": list(result.accepted_camera_ids),
                   "rejected_camera_ids": list(result.rejected_camera_ids),
                   "fused_xy": [float(mean_xy[0]), float(mean_xy[1])],
                   "fused_xy_before_propagation": [float(result.mean_xy[0]),
                                                   float(result.mean_xy[1])],
                   # The instant the fused answer describes -- `now` when propagation
                   # applied, the newest capture time when it did not. Published so a
                   # scorer can align the fused answer without guessing which of the
                   # two it is; the per-camera `obs_stamp`s are a different instant.
                   "fused_stamp": float(ts),
                   "propagation": self._propagation_status,
                   "gate_rejections": dict(self._gate_rejections),
                   "n_fresh": len(fresh),
                   # The cameras that were ON THE TABLE at this instant, before the arm's rule
                   # chose among them. accepted_camera_ids is what the rule USED, which for a
                   # single-best rule is always one camera -- so the two are not the same axis,
                   # and the fusion comparison has to be read against this one.
                   "synchronous_camera_ids": [str(o.camera_id) for o in fresh],
                   # Every camera's own answer at this instant -- where it put the robot and
                   # how sure it was -- beside the one the rule produced from them. Without
                   # this the fusion is a number with no visible mechanism: you can see that
                   # arms differ but not WHERE each camera landed or what the rule did with
                   # them. Small: five cameras at 5 Hz.
                   "observations": [
                       {"camera": str(o.camera_id),
                        # When the CAMERA saw it, not when the log wrote it. Without this the
                        # only truth a reading can be scored against is the truth at logging
                        # time, which is later by the whole detector-plus-manager delay -- and
                        # that delay reads as measurement error identically on every camera.
                        "obs_stamp": float(
                            original_by_camera[str(o.camera_id)].timestamp_s
                        ),
                        "xy": [
                            float(original_by_camera[str(o.camera_id)].xy_m[0]),
                            float(original_by_camera[str(o.camera_id)].xy_m[1]),
                        ],
                        "cov": [
                            [float(original_by_camera[str(o.camera_id)].covariance_m2[0][0]),
                             float(original_by_camera[str(o.camera_id)].covariance_m2[0][1])],
                            [float(original_by_camera[str(o.camera_id)].covariance_m2[1][0]),
                             float(original_by_camera[str(o.camera_id)].covariance_m2[1][1])],
                        ],
                        "aligned_xy": [float(o.xy_m[0]), float(o.xy_m[1])],
                        "aligned_cov": [
                            [float(o.covariance_m2[0][0]), float(o.covariance_m2[0][1])],
                            [float(o.covariance_m2[1][0]), float(o.covariance_m2[1][1])],
                        ],
                        "used": str(o.camera_id) in set(result.accepted_camera_ids),
                        **self._detection_extras_by_camera.get(str(o.camera_id), {})}
                       for o in fresh],
                   "fused_cov": [[float(report_covariance[0][0]), float(report_covariance[0][1])],
                                 [float(report_covariance[1][0]), float(report_covariance[1][1])]],
                   "scores_by_camera": scores,
                   "rejected_by_camera": {key: list(value) for key, value in rejected.items()},
                   "time_skew_rejected_camera_ids": time_skew_rejected,
                   "covariance_profile": self.covariance_profile,
                   "common_mode_std_m": self.fusion_common_mode_std_m,
                   "max_timestamp_spread_s": self.fusion_max_timestamp_spread_s}
        payload["bootstrap_evidence"] = bootstrap_evidence
        payload["gp_query_source_by_camera"] = dict(
            self._reliability_query_source_by_camera
        )
        payload["measurement_model_status_by_camera"] = dict(
            self._measurement_model_status_by_camera
        )
        self._fusion_publication_seq += 1
        event = FusedCorrectionEvent.create(
            source_batch_id=source_batch_id, epoch=self._manager_epoch,
            publication_seq=self._fusion_publication_seq, frame_id=self.frame_id,
            common_capture_stamp_ns=common_capture_ns, correction_stamp_ns=correction_ns,
            xy=mean_xy, covariance_m2=report_covariance, accepted_camera_ids=result.accepted_camera_ids,
            contracts=inputs.contracts, capture_observations=observations,
            aligned_observations=all_aligned_observations, motion_support_by_camera=common_time_support,
            model=dict(mean=self.observation_model, covariance_profile=self.covariance_profile,
                       reference_calibration_sha256=getattr(getattr(self, "reference_calibration", None), "sha256", None),
                       availability_model_sha256=getattr(getattr(self, "commissioned_availability", None), "sha256", None),
                       fusion_rule=self.fusion_rule, drift_std_m_per_s=self.propagation_drift_std,
                       common_mode_std_m=self.fusion_common_mode_std_m),
            belief_identity=inputs.belief_identity, common_time_xy=result.mean_xy,
            common_time_covariance_m2=result.covariance_m2,
            correction_motion_support=correction_motion_support)
        payload.update(fusion_event_id=event.payload["event_id"],
                       payload_sha256=event.payload["payload_sha256"],
                       epoch=self._manager_epoch, publication_seq=self._fusion_publication_seq,
                       common_capture_stamp_ns=common_capture_ns, correction_stamp_ns=correction_ns)
        def envelope_publish(text):
            envelope = String()
            envelope.data = text
            self.fused_correction_pub.publish(envelope)
        class Journal:
            def append(_journal, entry):
                self._publish_batch_outcome(entry)
        publish_fused_event(event, journal=Journal(), publish_envelope=envelope_publish,
            publish_decision=lambda: self._publish_decision(payload),
            publish_compatibility=(lambda: self.selected_pub.publish(message),
                                   lambda: self.active_pub.publish(message)))

    def _pose_message(self, observation: MapObservation):
        message = PoseWithCovarianceStamped()
        message.header.stamp.sec, message.header.stamp.nanosec = divmod(round(observation.timestamp_s*1e9), 1_000_000_000)
        message.header.frame_id = self.frame_id
        message.pose.pose.position.x = float(observation.xy_m[0])
        message.pose.pose.position.y = float(observation.xy_m[1])
        message.pose.pose.orientation.w = 1.0
        covariance = [0.0] * 36
        covariance[0] = float(observation.covariance_m2[0][0])
        covariance[1] = float(observation.covariance_m2[0][1])
        covariance[6] = float(observation.covariance_m2[1][0])
        covariance[7] = float(observation.covariance_m2[1][1])
        covariance[35] = NONINFORMATIVE_YAW_VAR
        message.pose.covariance = covariance
        return message

    def _terminal_stop_request_cb(self, message) -> None:
        """Stop producing corrections and durably acknowledge one terminal request."""
        try:
            request = terminal_stop_request_from_json(message.data)
        except ValueError as exc:
            raise RuntimeError("invalid terminal stop request") from exc
        with self._decision_lock, self._input_lock:
            if self._terminal_stopped:
                if request.request_id != self._terminal_stop_request_id:
                    raise RuntimeError("conflicting terminal stop request identity")
                return
            pending = self._ready_source_batch_id
            if pending is not None and pending != self._last_decided_source_batch_id:
                self._publish_batch_outcome(dict(
                    source_batch_id=pending,
                    status="superseded_by_terminal_stop",
                    terminal_stop_request_id=request.request_id,
                ))
            self._terminal_stop_request_id = request.request_id
            self._outcome_transport_enabled = False
            self._append_session_stopped(request_id=request.request_id)
            self._terminal_stopped = True
        ack = TerminalStopAck(
            request_id=request.request_id,
            component="camera_manager",
            status="correction_stream_quiescent",
            acknowledgement_stamp_ns=int(self.get_clock().now().nanoseconds),
            detail="durable_session_stopped",
        )
        response = String()
        response.data = terminal_stop_ack_to_json(ack)
        self._terminal_stop_ack_pub.publish(response)

    def _append_session_stopped(self, *, request_id=""):
        """Append the terminal record without consulting any ROS entity."""
        if getattr(self, "_session_stopped_appended", False):
            return None
        # The launch service may invalidate the rcl context before
        # ``destroy_node`` runs.  The last observed simulation high-water mark is
        # sufficient metadata for a journal-only terminal event.
        result = self._outcome_journal.append(dict(
            status="session_stopped",
            transport="journal_only",
            stage="manager",
            terminal_stop_request_id=str(request_id or ""),
            publish_stamp_s=float(
                getattr(self, "_batch_clock_high_water_s", 0.0) or 0.0),
        ))
        self._session_stopped_appended = True
        return result

    def destroy_node(self):
        """Durably close the manager event stream after ROS delivery stops."""
        self._outcome_transport_enabled = False
        try:
            self._append_session_stopped()
        finally:
            self._outcome_journal.close()
            result = super().destroy_node()
        return result


def _contract_quality(contract: CameraObservation):
    from reliability.contracts import CameraQuality

    return CameraQuality(
        camera_id=contract.camera_id,
        p_available=contract.availability_probability,
        conditional_cov_uv=contract.conditional_cov_uv,
        association_confidence=contract.association_probability,
        epistemic_score=0.0,
        stale=False,
        source_model="live_contract",
    )


def main(args=None) -> int:
    if rclpy is None:
        raise SystemExit("rclpy is required to run camera_manager_node")
    rclpy.init(args=args)
    node = CameraManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
