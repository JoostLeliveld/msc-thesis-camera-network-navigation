"""Operational dynamic-occlusion model for fixed-camera localization.

The static reliability map answers where a camera *usually* works.  This module
adds the short-lived part of the problem: an actor tracker can report people,
forklifts, or pallet jacks that currently cross the camera-to-robot ray.  Actor
state is an input available at runtime; no simulator geometry or evaluation
truth is used here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from reliability.contracts import CameraQuality, ContractValidationError
from reliability.fusion import MapObservation


@dataclass(frozen=True)
class DynamicActorState:
    """A tracked, ground-plane actor footprint at one observation time."""

    actor_id: str
    xy_m: tuple[float, float]
    radius_m: float = 0.35
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not str(self.actor_id):
            raise ContractValidationError("actor_id must be non-empty")
        xy = _pair(self.xy_m, "xy_m")
        radius = _positive(self.radius_m, "radius_m")
        confidence = _probability(self.confidence, "confidence")
        object.__setattr__(self, "actor_id", str(self.actor_id))
        object.__setattr__(self, "xy_m", xy)
        object.__setattr__(self, "radius_m", radius)
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True)
class DynamicOcclusionConfig:
    """Geometry and conservative covariance response for dynamic blockers.

    ``camera_xy_m`` is the fixed-camera location from calibration.  Actors that
    overlap a camera-to-observation ground-plane ray reduce availability and
    inflate the entire 2D observation covariance.  It is intentionally an
    isotropic response: without a 3D actor shape/camera projection model, an
    anisotropic claim would be unjustified.
    """

    camera_xy_m: Mapping[str, tuple[float, float]]
    ray_margin_m: float = 0.20
    covariance_gain: float = 8.0
    max_covariance_inflation: float = 12.0

    def __post_init__(self) -> None:
        camera_xy = {str(camera_id): _pair(xy, f"camera_xy_m[{camera_id!r}]")
                     for camera_id, xy in self.camera_xy_m.items()}
        if not camera_xy:
            raise ContractValidationError("camera_xy_m must contain at least one camera")
        margin = _positive(self.ray_margin_m, "ray_margin_m")
        gain = _nonnegative(self.covariance_gain, "covariance_gain")
        maximum = _positive(self.max_covariance_inflation, "max_covariance_inflation")
        if maximum < 1.0:
            raise ContractValidationError("max_covariance_inflation must be at least 1")
        object.__setattr__(self, "camera_xy_m", camera_xy)
        object.__setattr__(self, "ray_margin_m", margin)
        object.__setattr__(self, "covariance_gain", gain)
        object.__setattr__(self, "max_covariance_inflation", maximum)


@dataclass(frozen=True)
class DynamicOcclusionDiagnostic:
    camera_id: str
    occlusion_probability: float
    covariance_inflation: float
    blocking_actor_ids: tuple[str, ...]


def dynamic_occlusion_probability(
    camera_xy_m: Sequence[float],
    target_xy_m: Sequence[float],
    actors: Sequence[DynamicActorState],
    *,
    ray_margin_m: float = 0.20,
) -> tuple[float, tuple[str, ...]]:
    """Probability that at least one tracked actor crosses a sight line.

    Actor events are combined as independent conservative blocker events.  A
    tracked centre exactly on the segment contributes its confidence; off-ray
    contributions fall off with the actor radius plus ``ray_margin_m``.
    """

    camera = _pair(camera_xy_m, "camera_xy_m")
    target = _pair(target_xy_m, "target_xy_m")
    margin = _positive(ray_margin_m, "ray_margin_m")
    dx, dy = target[0] - camera[0], target[1] - camera[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 1.0e-12:
        return 0.0, tuple()

    no_block = 1.0
    blockers: list[str] = []
    for actor in actors:
        ax, ay = actor.xy_m[0] - camera[0], actor.xy_m[1] - camera[1]
        u = (ax * dx + ay * dy) / length_sq
        if not 0.0 < u < 1.0:
            continue
        closest = (camera[0] + u * dx, camera[1] + u * dy)
        distance = math.hypot(actor.xy_m[0] - closest[0], actor.xy_m[1] - closest[1])
        sigma = actor.radius_m + margin
        contribution = actor.confidence * math.exp(-0.5 * (distance / sigma) ** 2)
        contribution = min(max(contribution, 0.0), 1.0)
        if contribution > 1.0e-3:
            blockers.append(actor.actor_id)
            no_block *= 1.0 - contribution
    return 1.0 - no_block, tuple(blockers)


def dynamic_occlusion_adjusted_observation(
    observation: MapObservation,
    actors: Sequence[DynamicActorState],
    config: DynamicOcclusionConfig,
) -> tuple[MapObservation, DynamicOcclusionDiagnostic]:
    """Apply the runtime dynamic-occlusion response to one map observation."""

    if observation.camera_id not in config.camera_xy_m:
        raise ContractValidationError(
            f"dynamic occlusion has no calibrated position for {observation.camera_id!r}"
        )
    probability, blockers = dynamic_occlusion_probability(
        config.camera_xy_m[observation.camera_id], observation.xy_m, actors,
        ray_margin_m=config.ray_margin_m,
    )
    odds = probability / max(1.0 - probability, 1.0e-6)
    inflation = min(1.0 + config.covariance_gain * odds, config.max_covariance_inflation)
    quality = observation.quality
    adjusted_quality = CameraQuality(
        camera_id=observation.camera_id,
        p_available=quality.p_available * (1.0 - probability),
        conditional_cov_uv=quality.conditional_cov_uv,
        association_confidence=quality.association_confidence,
        epistemic_score=quality.epistemic_score,
        stale=quality.stale,
        source_model="dynamic_occlusion",
    )
    cov = observation.covariance_m2
    adjusted = MapObservation(
        camera_id=observation.camera_id,
        timestamp_s=observation.timestamp_s,
        xy_m=observation.xy_m,
        covariance_m2=((cov[0][0] * inflation, cov[0][1] * inflation),
                       (cov[1][0] * inflation, cov[1][1] * inflation)),
        quality=adjusted_quality,
        source="dynamic_occlusion",
    )
    return adjusted, DynamicOcclusionDiagnostic(
        camera_id=observation.camera_id,
        occlusion_probability=probability,
        covariance_inflation=inflation,
        blocking_actor_ids=blockers,
    )


def _pair(value: Sequence[float], field_name: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ContractValidationError(f"{field_name} must be a 2-element sequence")
    out = (float(value[0]), float(value[1]))
    if not all(math.isfinite(item) for item in out):
        raise ContractValidationError(f"{field_name} must be finite")
    return out


def _positive(value: float, field_name: str) -> float:
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ContractValidationError(f"{field_name} must be finite and positive")
    return out


def _nonnegative(value: float, field_name: str) -> float:
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise ContractValidationError(f"{field_name} must be finite and non-negative")
    return out


def _probability(value: float, field_name: str) -> float:
    out = float(value)
    if not math.isfinite(out) or not 0.0 <= out <= 1.0:
        raise ContractValidationError(f"{field_name} must be a probability in [0, 1]")
    return out
