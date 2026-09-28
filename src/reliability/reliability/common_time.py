"""Supported displacement from immutable, frame-labelled odometry poses.

Interpolation uses measured odometry only. Its support metadata describes where
samples exist; the caller's existing drift covariance remains a separate model.
No corrected robot-belief position is an admissible motion input here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from reliability.contracts import ContractValidationError


@dataclass(frozen=True)
class MotionPose:
    stamp_ns: int
    xy_m: tuple[float, float]
    yaw_rad: float
    frame_id: str
    epoch: str

    def __post_init__(self):
        if type(self.stamp_ns) is not int or self.stamp_ns < 0:
            raise ContractValidationError("motion stamp must be non-negative integer ns")
        if not isinstance(self.frame_id, str) or not self.frame_id.strip():
            raise ContractValidationError("motion frame must be explicit")
        if not isinstance(self.epoch, str) or not self.epoch.strip():
            raise ContractValidationError("motion epoch must be explicit")
        if len(self.xy_m) != 2 or not all(math.isfinite(v) for v in (*self.xy_m, self.yaw_rad)):
            raise ContractValidationError("motion pose must be finite XY/yaw")
        object.__setattr__(self, "xy_m", tuple(float(v) for v in self.xy_m))


@dataclass(frozen=True)
class Displacement:
    start_ns: int
    end_ns: int
    frame_id: str
    source_frame_id: str
    epoch: str
    supported: bool
    reason: str
    delta_xy_m: tuple[float, float] | None = None
    sample_stamps_ns: tuple[int, ...] = ()
    rotation_rad: float | None = None

    def to_dict(self):
        return dict(start_ns=self.start_ns, end_ns=self.end_ns,
                    frame_id=self.frame_id, source_frame_id=self.source_frame_id,
                    epoch=self.epoch, supported=self.supported, reason=self.reason,
                    delta_xy_m=None if self.delta_xy_m is None else list(self.delta_xy_m),
                    sample_stamps_ns=list(self.sample_stamps_ns), rotation_rad=self.rotation_rad,
                    source="measured_odometry_pose", interpolation="piecewise_linear_xy",
                    covariance_model="configured_drift_inflation",
                    cross_camera_motion_covariance_modelled=False)


@dataclass(frozen=True)
class MotionPoseSnapshot:
    samples: tuple[MotionPose, ...]
    target_frame: str
    source_frame: str
    epoch: str
    source_to_target_yaw: float

    @classmethod
    def capture(cls, entries, *, target_frame, source_frame, epoch,
                source_to_target_yaw=0.):
        """Copy/sort while holding the owner's input lock; reject ambiguous samples."""
        if not all(isinstance(v, str) and v.strip() for v in (target_frame, source_frame, epoch)):
            raise ContractValidationError("motion snapshot needs explicit frames and epoch")
        if not math.isfinite(source_to_target_yaw):
            raise ContractValidationError("motion frame transform must be finite and declared")
        by_stamp = {}
        for sample in entries:
            if not isinstance(sample, MotionPose):
                raise ContractValidationError("motion snapshot requires labelled MotionPose samples")
            if sample.frame_id != source_frame or sample.epoch != epoch:
                raise ContractValidationError("motion snapshot mixes frames or epochs")
            if sample.stamp_ns in by_stamp and by_stamp[sample.stamp_ns] != sample:
                raise ContractValidationError("conflicting motion samples at one stamp")
            by_stamp[sample.stamp_ns] = sample
        return cls(tuple(by_stamp[t] for t in sorted(by_stamp)), target_frame,
                   source_frame, epoch, float(source_to_target_yaw))

    def displacement(self, start_ns: int, end_ns: int, *, max_gap_s: float,
                     max_endpoint_delta_s: float | None = None) -> Displacement:
        def result(supported, reason, delta=None, stamps=()):
            return Displacement(start_ns, end_ns, self.target_frame, self.source_frame,
                                self.epoch, supported, reason, delta, stamps,
                                self.source_to_target_yaw)
        if (type(start_ns) is not int or type(end_ns) is not int
                or start_ns < 0 or end_ns < start_ns):
            return result(False, "invalid_interval")
        if not math.isfinite(max_gap_s) or max_gap_s <= 0:
            return result(False, "invalid_support_gap")
        if max_endpoint_delta_s is None:
            max_endpoint_delta_s = max_gap_s
        if not math.isfinite(max_endpoint_delta_s) or max_endpoint_delta_s < 0:
            return result(False, "invalid_endpoint_tolerance")
        if start_ns == end_ns:
            return result(True, "same_instant", (0., 0.))
        if not self.samples:
            return result(False, "empty_motion_history")
        # No extrapolation, including holding the nearest pose at both endpoints.
        if self.samples[0].stamp_ns > start_ns:
            return result(False, "missing_prefix_support")
        if self.samples[-1].stamp_ns < end_ns:
            return result(False, "missing_tail_support")
        used = set()

        def at(stamp):
            for index, item in enumerate(self.samples):
                if item.stamp_ns == stamp:
                    used.add(index)
                    return item.xy_m
                if item.stamp_ns > stamp:
                    left = self.samples[index - 1]
                    if max(stamp-left.stamp_ns, item.stamp_ns-stamp) > max_endpoint_delta_s * 1e9 + .5:
                        return None
                    used.update((index-1, index))
                    fraction = (stamp-left.stamp_ns)/(item.stamp_ns-left.stamp_ns)
                    return tuple(a + fraction*(b-a) for a, b in zip(left.xy_m, item.xy_m))
            return None

        begin, finish = at(start_ns), at(end_ns)
        if begin is None or finish is None:
            return result(False, "endpoint_bracket_too_wide")
        first, last = min(used), max(used)
        interval = self.samples[first:last+1]
        stamps = tuple(p.stamp_ns for p in interval)
        if any(b-a > max_gap_s*1e9 + .5 for a, b in zip(stamps, stamps[1:])):
            return result(False, "motion_gap", stamps=stamps)
        dx, dy = finish[0]-begin[0], finish[1]-begin[1]
        c, s = math.cos(self.source_to_target_yaw), math.sin(self.source_to_target_yaw)
        return result(True, "supported", (c*dx-s*dy, s*dx+c*dy), stamps)
