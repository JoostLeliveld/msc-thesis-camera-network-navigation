"""Per-camera reliability learning from detection outcomes.

The learning layer keeps one field per camera. This is deliberate: a miss,
false positive, or occlusion pattern from camera B must not contaminate camera
A's reliability field. Maps are fused only after each per-camera posterior has
been updated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np

from reliability.contracts import ContractValidationError
from reliability.prior import CameraPriorMap, MultiCameraPriorMap, fuse_camera_prior_maps


@dataclass(frozen=True)
class ReliabilityObservation:
    camera_id: str
    xy_m: tuple[float, float]
    detected_probability: float
    weight: float = 1.0
    timestamp_s: float = 0.0

    def __post_init__(self) -> None:
        if not str(self.camera_id).strip():
            raise ContractValidationError("camera_id must be non-empty")
        object.__setattr__(self, "camera_id", str(self.camera_id))
        object.__setattr__(self, "xy_m", _pair(self.xy_m, "xy_m"))
        object.__setattr__(self, "detected_probability", _probability(self.detected_probability, "detected_probability"))
        weight = _finite(self.weight, "weight")
        if weight < 0.0:
            raise ContractValidationError("weight must be non-negative")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "timestamp_s", _finite(self.timestamp_s, "timestamp_s"))


@dataclass(frozen=True)
class LearningConfig:
    prior_strength: float | Mapping[str, float] = 3.0
    alpha_floor: float = 1.0e-3
    beta_floor: float = 1.0e-3
    kernel_length_scale_m: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.prior_strength, Mapping):
            for key, value in self.prior_strength.items():
                if _finite(value, f"prior_strength[{key}]") < 0.0:
                    raise ContractValidationError("prior_strength values must be non-negative")
        elif _finite(self.prior_strength, "prior_strength") < 0.0:
            raise ContractValidationError("prior_strength must be non-negative")
        if _finite(self.alpha_floor, "alpha_floor") < 0.0 or _finite(self.beta_floor, "beta_floor") < 0.0:
            raise ContractValidationError("alpha_floor and beta_floor must be non-negative")
        if _finite(self.kernel_length_scale_m, "kernel_length_scale_m") < 0.0:
            raise ContractValidationError("kernel_length_scale_m must be non-negative")

    def strength_for(self, camera_id: str) -> float:
        if isinstance(self.prior_strength, Mapping):
            return float(self.prior_strength.get(camera_id, 0.0))
        return float(self.prior_strength)


@dataclass(frozen=True)
class LearnedCameraReliabilityMap:
    camera_id: str
    xs: tuple[float, ...]
    ys: tuple[float, ...]
    prior_probability: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    evidence_weight: np.ndarray

    def __post_init__(self) -> None:
        xs = _axis(self.xs, "xs")
        ys = _axis(self.ys, "ys")
        shape = (len(ys), len(xs))
        object.__setattr__(self, "camera_id", str(self.camera_id))
        object.__setattr__(self, "xs", xs)
        object.__setattr__(self, "ys", ys)
        object.__setattr__(self, "prior_probability", _grid(self.prior_probability, shape, "prior_probability"))
        object.__setattr__(self, "alpha", _grid(self.alpha, shape, "alpha"))
        object.__setattr__(self, "beta", _grid(self.beta, shape, "beta"))
        object.__setattr__(self, "evidence_weight", _grid(self.evidence_weight, shape, "evidence_weight"))

    @property
    def posterior_probability(self) -> np.ndarray:
        total = self.alpha + self.beta
        with np.errstate(divide="ignore", invalid="ignore"):
            posterior = np.divide(self.alpha, total, out=np.zeros_like(self.alpha), where=total > 0.0)
        return np.clip(posterior, 0.0, 1.0)

    def to_camera_prior_map(self) -> CameraPriorMap:
        return CameraPriorMap(
            camera_id=self.camera_id,
            xs=self.xs,
            ys=self.ys,
            probability=self.posterior_probability,
            fov_mask=np.asarray(self.prior_probability > 0.0, dtype=bool),
            feature_maps={"evidence_weight": self.evidence_weight},
        )

    def to_dict(self, *, include_maps: bool = False) -> dict[str, Any]:
        posterior = self.posterior_probability
        out: dict[str, Any] = {
            "camera_id": self.camera_id,
            "shape": [int(posterior.shape[0]), int(posterior.shape[1])],
            "mean_prior_probability": float(np.mean(self.prior_probability)),
            "mean_posterior_probability": float(np.mean(posterior)),
            "total_evidence_weight": float(np.sum(self.evidence_weight)),
        }
        if include_maps:
            out["xs"] = list(self.xs)
            out["ys"] = list(self.ys)
            out["prior_probability"] = self.prior_probability.tolist()
            out["posterior_probability"] = posterior.tolist()
            out["alpha"] = self.alpha.tolist()
            out["beta"] = self.beta.tolist()
            out["evidence_weight"] = self.evidence_weight.tolist()
        return out


@dataclass(frozen=True)
class PerCameraLearningResult:
    camera_maps: Mapping[str, LearnedCameraReliabilityMap]
    fused_posterior: MultiCameraPriorMap

    def __post_init__(self) -> None:
        maps = dict(self.camera_maps)
        if not maps:
            raise ContractValidationError("camera_maps must not be empty")
        object.__setattr__(self, "camera_maps", maps)

    def to_dict(self, *, include_maps: bool = False) -> dict[str, Any]:
        return {
            "camera_maps": {
                camera_id: cmap.to_dict(include_maps=include_maps)
                for camera_id, cmap in sorted(self.camera_maps.items())
            },
            "fused_posterior": self.fused_posterior.to_dict(include_maps=include_maps),
        }


def learn_per_camera_reliability(
    priors: Mapping[str, CameraPriorMap] | MultiCameraPriorMap,
    observations: Sequence[ReliabilityObservation],
    config: LearningConfig | None = None,
) -> PerCameraLearningResult:
    cfg = config or LearningConfig()
    prior_maps = dict(priors.camera_maps if isinstance(priors, MultiCameraPriorMap) else priors)
    if not prior_maps:
        raise ContractValidationError("priors must contain at least one camera")
    alpha: dict[str, np.ndarray] = {}
    beta: dict[str, np.ndarray] = {}
    evidence: dict[str, np.ndarray] = {}
    for camera_id, prior in prior_maps.items():
        strength = cfg.strength_for(camera_id)
        p = np.clip(np.asarray(prior.probability, dtype=float), 0.0, 1.0)
        alpha[camera_id] = p * strength + float(cfg.alpha_floor)
        beta[camera_id] = (1.0 - p) * strength + float(cfg.beta_floor)
        evidence[camera_id] = np.zeros_like(p)

    for obs in observations:
        if obs.camera_id not in prior_maps:
            raise ContractValidationError(f"Observation references unknown camera_id {obs.camera_id!r}")
        prior = prior_maps[obs.camera_id]
        weights = _cell_weights(
            prior.xs,
            prior.ys,
            obs.xy_m,
            kernel_length_scale_m=float(cfg.kernel_length_scale_m),
        )
        if not np.any(weights > 0.0):
            continue
        weighted = weights * float(obs.weight)
        alpha[obs.camera_id] += weighted * float(obs.detected_probability)
        beta[obs.camera_id] += weighted * (1.0 - float(obs.detected_probability))
        evidence[obs.camera_id] += weighted

    learned = {
        camera_id: LearnedCameraReliabilityMap(
            camera_id=camera_id,
            xs=prior.xs,
            ys=prior.ys,
            prior_probability=prior.probability,
            alpha=alpha[camera_id],
            beta=beta[camera_id],
            evidence_weight=evidence[camera_id],
        )
        for camera_id, prior in prior_maps.items()
    }
    fused = fuse_camera_prior_maps(
        {
            camera_id: cmap.to_camera_prior_map()
            for camera_id, cmap in learned.items()
        }
    )
    return PerCameraLearningResult(camera_maps=learned, fused_posterior=fused)


def observations_from_camera_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    camera_id: str,
    x_key: str = "state_x",
    y_key: str = "state_y",
    detected_key: str = "yolo_detected_after_threshold",
    score_key: str = "yolo_score_selected",
    use_score_as_probability: bool = False,
) -> tuple[ReliabilityObservation, ...]:
    observations: list[ReliabilityObservation] = []
    for idx, row in enumerate(rows):
        x = _optional_float(row.get(x_key))
        y = _optional_float(row.get(y_key))
        if x is None or y is None:
            continue
        if use_score_as_probability:
            detected = _optional_float(row.get(score_key))
            if detected is None:
                detected = 0.0
        else:
            detected = 1.0 if _truthy(row.get(detected_key)) else 0.0
        timestamp = _optional_float(row.get("timestamp_s", row.get("stamp", idx))) or float(idx)
        observations.append(
            ReliabilityObservation(
                camera_id=camera_id,
                xy_m=(x, y),
                detected_probability=detected,
                timestamp_s=timestamp,
            )
        )
    return tuple(observations)


def _cell_weights(
    xs: Sequence[float],
    ys: Sequence[float],
    xy_m: Sequence[float],
    *,
    kernel_length_scale_m: float,
) -> np.ndarray:
    x, y = _pair(xy_m, "xy_m")
    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    if x < xs_arr[0] or x > xs_arr[-1] or y < ys_arr[0] or y > ys_arr[-1]:
        return np.zeros((len(ys_arr), len(xs_arr)), dtype=float)
    if kernel_length_scale_m <= 0.0:
        ix = int(np.argmin(np.abs(xs_arr - x)))
        iy = int(np.argmin(np.abs(ys_arr - y)))
        weights = np.zeros((len(ys_arr), len(xs_arr)), dtype=float)
        weights[iy, ix] = 1.0
        return weights
    xx, yy = np.meshgrid(xs_arr, ys_arr)
    d2 = (xx - x) ** 2 + (yy - y) ** 2
    weights = np.exp(-0.5 * d2 / max(kernel_length_scale_m * kernel_length_scale_m, 1.0e-12))
    weights[weights < 1.0e-6] = 0.0
    return weights


def _axis(value: Sequence[float], field_name: str) -> tuple[float, ...]:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        raise ContractValidationError(f"{field_name} must contain at least two values")
    out = tuple(_finite(item, f"{field_name}[]") for item in value)
    for prev, current in zip(out, out[1:]):
        if current <= prev:
            raise ContractValidationError(f"{field_name} must be strictly increasing")
    return out


def _grid(value: Sequence[Sequence[float]], shape: tuple[int, int], field_name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != shape:
        raise ContractValidationError(f"{field_name} must have shape {shape}")
    if not np.all(np.isfinite(arr)):
        raise ContractValidationError(f"{field_name} must be finite")
    return arr


def _pair(value: Sequence[float], field_name: str) -> tuple[float, float]:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ContractValidationError(f"{field_name} must be a 2-element sequence")
    return (_finite(value[0], f"{field_name}[0]"), _finite(value[1], f"{field_name}[1]"))


def _probability(value: float, field_name: str) -> float:
    out = _finite(value, field_name)
    if not 0.0 <= out <= 1.0:
        raise ContractValidationError(f"{field_name} must be in [0, 1]")
    return out


def _finite(value: float, field_name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"{field_name} must be numeric") from exc
    if not math.isfinite(out):
        raise ContractValidationError(f"{field_name} must be finite")
    return out


def _optional_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    numeric = _optional_float(value)
    return bool(numeric is not None and numeric >= 0.5)
