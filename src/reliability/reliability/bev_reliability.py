"""Occlusion-aware BEV reliability features and camera-set attention.

This module is intentionally small and ROS-independent. It provides a
thesis-sized learning surface for calibrated external cameras: per-camera
operational feature tokens are scored by a linear reliability model, then fused
with soft attention over the available camera set.

The implementation does not consume raw images or evaluation labels. It is a
drop-in offline baseline that can later be replaced by a PyTorch network with
the same feature/token contract if the experiments justify it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np

from reliability.contracts import (
    CameraQuality,
    ContractValidationError,
    LeakageError,
    reject_evaluation_only_keys,
)
from reliability.firewall import validate_feature_columns
from reliability.prior import CameraPriorMap, MultiCameraPriorMap
from reliability.providers import GridMapReliabilityProvider
from reliability.single_camera_adapter import precision_blend_covariance
from unav_common.config import parse_bool


DEFAULT_BEV_FEATURE_NAMES = (
    "bias",
    "geometry_prior_logit",
    "detector_score",
    "detection_valid",
    "association_confidence",
    "freshness",
    "not_stale",
    "range_score",
    "border_score",
    "pixel_scale_score",
    "incidence_score",
    "recent_detection_rate",
    "overlap_agreement",
)

DEFAULT_BEV_FEATURE_WEIGHTS = {
    "bias": -1.65,
    "geometry_prior_logit": 0.45,
    "detector_score": 1.25,
    "detection_valid": 0.80,
    "association_confidence": 0.45,
    "freshness": 0.45,
    "not_stale": 0.45,
    "range_score": 0.25,
    "border_score": 0.20,
    "pixel_scale_score": 0.15,
    "incidence_score": 0.15,
    "recent_detection_rate": 0.35,
    "overlap_agreement": 0.25,
}


@dataclass(frozen=True)
class BEVFeatureConfig:
    """Normalization and fusion settings for BEV reliability tokens."""

    feature_names: tuple[str, ...] = DEFAULT_BEV_FEATURE_NAMES
    stale_decay_s: float = 0.50
    good_range_m: float = 6.0
    range_decay_m: float = 6.0
    full_border_margin_px: float = 40.0
    full_pixel_scale_px_per_m: float = 35.0
    min_ground_incidence_sin: float = 0.08
    overlap_gate_m: float = 0.30
    attention_temperature: float = 1.0
    fusion_mode: str = "union"
    r_visible_uv: float = 2.5
    r_miss_uv: float = 40.0
    min_probability: float = 1.0e-4

    def __post_init__(self) -> None:
        names = tuple(str(name).strip() for name in self.feature_names)
        if not names or any(not name for name in names):
            raise ContractValidationError("feature_names must contain non-empty names")
        if len(set(names)) != len(names):
            raise ContractValidationError("feature_names must be unique")
        validate_feature_columns(names)
        for field_name in (
            "stale_decay_s",
            "good_range_m",
            "range_decay_m",
            "full_border_margin_px",
            "full_pixel_scale_px_per_m",
            "min_ground_incidence_sin",
            "overlap_gate_m",
            "attention_temperature",
            "r_visible_uv",
            "r_miss_uv",
            "min_probability",
        ):
            value = _finite(getattr(self, field_name), field_name)
            if value < 0.0:
                raise ContractValidationError(f"{field_name} must be non-negative")
            object.__setattr__(self, field_name, value)
        if self.attention_temperature <= 0.0:
            raise ContractValidationError("attention_temperature must be positive")
        if self.r_visible_uv <= 0.0 or self.r_miss_uv <= 0.0:
            raise ContractValidationError("R standard deviations must be positive")
        if not 0.0 <= self.min_probability < 0.5:
            raise ContractValidationError("min_probability must be in [0, 0.5)")
        mode = str(self.fusion_mode).strip().lower()
        if mode not in {"union", "attention", "best"}:
            raise ContractValidationError("fusion_mode must be one of union, attention, best")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "fusion_mode", mode)


@dataclass(frozen=True)
class BEVCameraToken:
    """Operational feature token for one camera at one candidate BEV state."""

    camera_id: str
    xy_m: tuple[float, float]
    timestamp_s: float = 0.0
    geometry_prior: float = 0.5
    in_fov: bool = True
    detector_score: float = 0.0
    detection_valid: bool = False
    measurement_age_s: float = 0.0
    measurement_stale: bool = False
    association_confidence: float = 1.0
    camera_relative_range_m: float = math.nan
    border_margin_px: float = math.nan
    projection_scale_px_per_m: float = math.nan
    ground_incidence_sin: float = math.nan
    recent_detection_rate: float = math.nan
    overlap_disagreement_m: float = math.nan
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        camera_id = str(self.camera_id).strip()
        if not camera_id:
            raise ContractValidationError("camera_id must be non-empty")
        object.__setattr__(self, "camera_id", camera_id)
        object.__setattr__(self, "xy_m", _pair(self.xy_m, "xy_m"))
        object.__setattr__(self, "timestamp_s", _finite(self.timestamp_s, "timestamp_s"))
        object.__setattr__(self, "geometry_prior", _probability(self.geometry_prior, "geometry_prior"))
        try:
            in_fov = parse_bool(self.in_fov, field_name="in_fov")
            detection_valid = parse_bool(self.detection_valid, field_name="detection_valid")
            measurement_stale = parse_bool(self.measurement_stale, field_name="measurement_stale")
        except ValueError as exc:
            raise ContractValidationError(str(exc)) from exc
        object.__setattr__(self, "in_fov", in_fov)
        object.__setattr__(self, "detector_score", _probability(self.detector_score, "detector_score"))
        object.__setattr__(self, "detection_valid", detection_valid)
        age = _finite(self.measurement_age_s, "measurement_age_s")
        if age < 0.0:
            raise ContractValidationError("measurement_age_s must be non-negative")
        object.__setattr__(self, "measurement_age_s", age)
        object.__setattr__(self, "measurement_stale", measurement_stale)
        object.__setattr__(
            self,
            "association_confidence",
            _probability(self.association_confidence, "association_confidence"),
        )
        for field_name in (
            "camera_relative_range_m",
            "border_margin_px",
            "projection_scale_px_per_m",
            "ground_incidence_sin",
            "overlap_disagreement_m",
        ):
            value = _finite_or_nan(getattr(self, field_name), field_name)
            if math.isfinite(value) and value < 0.0:
                raise ContractValidationError(f"{field_name} must be non-negative")
            object.__setattr__(self, field_name, value)
        object.__setattr__(
            self,
            "recent_detection_rate",
            _probability_or_nan(self.recent_detection_rate, "recent_detection_rate"),
        )
        metadata = dict(self.metadata or {})
        reject_evaluation_only_keys(metadata, context="BEVCameraToken.metadata")
        object.__setattr__(self, "metadata", metadata)

    def feature_dict(self, config: BEVFeatureConfig | None = None) -> dict[str, float]:
        cfg = config or BEVFeatureConfig()
        prior = self.geometry_prior if self.in_fov else 0.0
        range_score = _range_score(
            self.camera_relative_range_m,
            good_range_m=cfg.good_range_m,
            range_decay_m=cfg.range_decay_m,
        )
        border_score = _linear_score(self.border_margin_px, cfg.full_border_margin_px)
        pixel_scale_score = _linear_score(self.projection_scale_px_per_m, cfg.full_pixel_scale_px_per_m)
        incidence_score = _incidence_score(self.ground_incidence_sin, cfg.min_ground_incidence_sin)
        recent_rate = (
            float(self.recent_detection_rate)
            if math.isfinite(self.recent_detection_rate)
            else (1.0 if self.detection_valid else 0.0)
        )
        overlap_agreement = _overlap_agreement(self.overlap_disagreement_m, cfg.overlap_gate_m)
        return {
            "bias": 1.0,
            "geometry_prior": float(prior),
            "geometry_prior_logit": _logit(prior),
            "detector_score": float(self.detector_score),
            "detection_valid": 1.0 if self.detection_valid else 0.0,
            "association_confidence": float(self.association_confidence),
            "freshness": math.exp(-self.measurement_age_s / max(cfg.stale_decay_s, 1.0e-9)),
            "not_stale": 0.0 if self.measurement_stale else 1.0,
            "range_score": range_score,
            "border_score": border_score,
            "pixel_scale_score": pixel_scale_score,
            "incidence_score": incidence_score,
            "recent_detection_rate": float(recent_rate),
            "overlap_agreement": overlap_agreement,
        }

    def feature_vector(self, config: BEVFeatureConfig | None = None) -> np.ndarray:
        cfg = config or BEVFeatureConfig()
        values = self.feature_dict(cfg)
        return np.asarray([values.get(name, 0.0) for name in cfg.feature_names], dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "xy_m": list(self.xy_m),
            "timestamp_s": self.timestamp_s,
            "geometry_prior": self.geometry_prior,
            "in_fov": self.in_fov,
            "detector_score": self.detector_score,
            "detection_valid": self.detection_valid,
            "measurement_age_s": self.measurement_age_s,
            "measurement_stale": self.measurement_stale,
            "association_confidence": self.association_confidence,
            "camera_relative_range_m": _none_nan(self.camera_relative_range_m),
            "border_margin_px": _none_nan(self.border_margin_px),
            "projection_scale_px_per_m": _none_nan(self.projection_scale_px_per_m),
            "ground_incidence_sin": _none_nan(self.ground_incidence_sin),
            "recent_detection_rate": _none_nan(self.recent_detection_rate),
            "overlap_disagreement_m": _none_nan(self.overlap_disagreement_m),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CameraSetReliabilityPrediction:
    """Reliability output for one BEV state and a set of calibrated cameras."""

    xy_m: tuple[float, float]
    timestamp_s: float
    camera_ids: tuple[str, ...]
    per_camera_probability: Mapping[str, float]
    attention_weights: Mapping[str, float]
    union_probability: float
    attended_probability: float
    best_camera_id: str
    best_probability: float
    fused_probability: float
    fused_quality: CameraQuality
    source_model: str = "bev_reliability_attention"

    def __post_init__(self) -> None:
        object.__setattr__(self, "xy_m", _pair(self.xy_m, "xy_m"))
        object.__setattr__(self, "timestamp_s", _finite(self.timestamp_s, "timestamp_s"))
        camera_ids = tuple(str(item) for item in self.camera_ids)
        if not camera_ids:
            raise ContractValidationError("camera_ids must not be empty")
        object.__setattr__(self, "camera_ids", camera_ids)
        probs = _probability_mapping(self.per_camera_probability, "per_camera_probability")
        weights = _probability_mapping(self.attention_weights, "attention_weights")
        for camera_id in camera_ids:
            if camera_id not in probs or camera_id not in weights:
                raise ContractValidationError("camera_ids must match probability and attention maps")
        object.__setattr__(self, "per_camera_probability", probs)
        object.__setattr__(self, "attention_weights", weights)
        object.__setattr__(self, "union_probability", _probability(self.union_probability, "union_probability"))
        object.__setattr__(self, "attended_probability", _probability(self.attended_probability, "attended_probability"))
        object.__setattr__(self, "best_probability", _probability(self.best_probability, "best_probability"))
        object.__setattr__(self, "fused_probability", _probability(self.fused_probability, "fused_probability"))
        if self.best_camera_id and self.best_camera_id not in camera_ids:
            raise ContractValidationError("best_camera_id must be one of camera_ids")

    def to_dict(self) -> dict[str, Any]:
        return {
            "xy_m": list(self.xy_m),
            "timestamp_s": self.timestamp_s,
            "camera_ids": list(self.camera_ids),
            "per_camera_probability": dict(self.per_camera_probability),
            "attention_weights": dict(self.attention_weights),
            "union_probability": self.union_probability,
            "attended_probability": self.attended_probability,
            "best_camera_id": self.best_camera_id,
            "best_probability": self.best_probability,
            "fused_probability": self.fused_probability,
            "fused_quality": self.fused_quality.to_dict(),
            "source_model": self.source_model,
        }


@dataclass(frozen=True)
class BEVReliabilityTrainingExample:
    """One operational training example for the lightweight reliability model."""

    token: BEVCameraToken
    target_reliable: float
    sample_weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_reliable", _probability(self.target_reliable, "target_reliable"))
        weight = _finite(self.sample_weight, "sample_weight")
        if weight < 0.0:
            raise ContractValidationError("sample_weight must be non-negative")
        object.__setattr__(self, "sample_weight", weight)
        metadata = dict(self.metadata or {})
        reject_evaluation_only_keys(metadata, context="BEVReliabilityTrainingExample.metadata")
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True)
class BEVReliabilityModel:
    """Small calibrated reliability network over per-camera BEV tokens."""

    feature_config: BEVFeatureConfig = field(default_factory=BEVFeatureConfig)
    weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_BEV_FEATURE_WEIGHTS))
    camera_biases: Mapping[str, float] = field(default_factory=dict)
    source_model: str = "bev_reliability_linear_attention"

    def __post_init__(self) -> None:
        cfg = self.feature_config
        weights = {str(key): _finite(value, f"weights[{key}]") for key, value in dict(self.weights).items()}
        validate_feature_columns(weights.keys())
        camera_biases = {
            str(key): _finite(value, f"camera_biases[{key}]")
            for key, value in dict(self.camera_biases).items()
        }
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "camera_biases", camera_biases)
        _ = self._weight_vector(cfg)

    def linear_score(self, token: BEVCameraToken) -> float:
        x = token.feature_vector(self.feature_config)
        w = self._weight_vector(self.feature_config)
        return float(np.dot(x, w) + float(self.camera_biases.get(token.camera_id, 0.0)))

    def predict_token_probability(self, token: BEVCameraToken) -> float:
        return _sigmoid(self.linear_score(token))

    def predict_quality(self, token: BEVCameraToken) -> CameraQuality:
        p = self.predict_token_probability(token)
        return CameraQuality(
            camera_id=token.camera_id,
            p_available=p,
            conditional_cov_uv=precision_blend_covariance(
                p,
                r_visible_uv=self.feature_config.r_visible_uv,
                r_miss_uv=self.feature_config.r_miss_uv,
                min_probability=self.feature_config.min_probability,
            ),
            association_confidence=token.association_confidence,
            epistemic_score=0.0,
            stale=token.measurement_stale,
            source_model=f"{self.source_model}:token",
        )

    def predict_set(self, tokens: Sequence[BEVCameraToken]) -> CameraSetReliabilityPrediction:
        if not tokens:
            raise ContractValidationError("tokens must not be empty")
        ordered = tuple(tokens)
        xy = ordered[0].xy_m
        for token in ordered[1:]:
            if math.hypot(token.xy_m[0] - xy[0], token.xy_m[1] - xy[1]) > 1.0e-6:
                raise ContractValidationError("all tokens in a camera set must share xy_m")
        camera_ids = tuple(token.camera_id for token in ordered)
        if len(set(camera_ids)) != len(camera_ids):
            raise ContractValidationError("camera set contains duplicate camera_id values")

        scores = np.asarray([self.linear_score(token) for token in ordered], dtype=float)
        probabilities = np.asarray([_sigmoid(score) for score in scores], dtype=float)
        attention = _softmax(scores / self.feature_config.attention_temperature)
        union = float(1.0 - np.prod(1.0 - probabilities))
        attended = float(np.dot(attention, probabilities))
        best_idx = int(np.argmax(probabilities))
        best = float(probabilities[best_idx])
        if self.feature_config.fusion_mode == "attention":
            fused = attended
        elif self.feature_config.fusion_mode == "best":
            fused = best
        else:
            fused = union

        fused_quality = CameraQuality(
            camera_id="multicamera",
            p_available=float(min(max(fused, self.feature_config.min_probability), 1.0 - self.feature_config.min_probability)),
            conditional_cov_uv=precision_blend_covariance(
                fused,
                r_visible_uv=self.feature_config.r_visible_uv,
                r_miss_uv=self.feature_config.r_miss_uv,
                min_probability=self.feature_config.min_probability,
            ),
            association_confidence=float(max(token.association_confidence for token in ordered)),
            epistemic_score=float(np.max(probabilities) - np.min(probabilities)),
            stale=all(token.measurement_stale for token in ordered),
            source_model=f"{self.source_model}:set_{self.feature_config.fusion_mode}",
        )

        return CameraSetReliabilityPrediction(
            xy_m=xy,
            timestamp_s=max(token.timestamp_s for token in ordered),
            camera_ids=camera_ids,
            per_camera_probability={
                token.camera_id: float(probabilities[idx])
                for idx, token in enumerate(ordered)
            },
            attention_weights={
                token.camera_id: float(attention[idx])
                for idx, token in enumerate(ordered)
            },
            union_probability=union,
            attended_probability=attended,
            best_camera_id=ordered[best_idx].camera_id,
            best_probability=best,
            fused_probability=float(fused),
            fused_quality=fused_quality,
            source_model=self.source_model,
        )

    def _weight_vector(self, config: BEVFeatureConfig) -> np.ndarray:
        return np.asarray([float(self.weights.get(name, 0.0)) for name in config.feature_names], dtype=float)


@dataclass(frozen=True)
class BEVReliabilityGridPrediction:
    """BEV grid output from the camera-set reliability model."""

    xs: tuple[float, ...]
    ys: tuple[float, ...]
    fused_probability: np.ndarray
    best_camera_id: np.ndarray
    attention_by_camera: Mapping[str, np.ndarray] = field(default_factory=dict)
    source_model: str = "bev_reliability_grid"

    def __post_init__(self) -> None:
        xs = _axis(self.xs, "xs")
        ys = _axis(self.ys, "ys")
        shape = (len(ys), len(xs))
        fused = np.asarray(self.fused_probability, dtype=float)
        if fused.shape != shape:
            raise ContractValidationError(f"fused_probability must have shape {shape}")
        if not np.all(np.isfinite(fused)):
            raise ContractValidationError("fused_probability must be finite")
        if np.any((fused < 0.0) | (fused > 1.0)):
            raise ContractValidationError("fused_probability must be in [0, 1]")
        best = np.asarray(self.best_camera_id, dtype=object)
        if best.shape != shape:
            raise ContractValidationError(f"best_camera_id must have shape {shape}")
        attention = {
            str(camera_id): _grid(value, shape, f"attention_by_camera[{camera_id}]")
            for camera_id, value in dict(self.attention_by_camera).items()
        }
        object.__setattr__(self, "xs", xs)
        object.__setattr__(self, "ys", ys)
        object.__setattr__(self, "fused_probability", fused)
        object.__setattr__(self, "best_camera_id", best)
        object.__setattr__(self, "attention_by_camera", attention)

    def to_grid_provider(self, *, camera_id: str = "multicamera") -> GridMapReliabilityProvider:
        return GridMapReliabilityProvider(
            camera_id=camera_id,
            artifact_path=f"<bev_reliability_grid:{camera_id}>",
            xs=self.xs,
            ys=self.ys,
            probability_map=tuple(tuple(float(v) for v in row) for row in self.fused_probability),
            map_key="P_bev_reliability_map",
            source_model=self.source_model,
            out_of_bounds_policy="min",
        )

    def to_dict(self, *, include_maps: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "shape": [int(self.fused_probability.shape[0]), int(self.fused_probability.shape[1])],
            "x_min": self.xs[0],
            "x_max": self.xs[-1],
            "y_min": self.ys[0],
            "y_max": self.ys[-1],
            "mean_fused_probability": float(np.mean(self.fused_probability)),
            "max_fused_probability": float(np.max(self.fused_probability)),
            "source_model": self.source_model,
        }
        if include_maps:
            out["xs"] = list(self.xs)
            out["ys"] = list(self.ys)
            out["fused_probability"] = self.fused_probability.tolist()
            out["best_camera_id"] = self.best_camera_id.tolist()
            out["attention_by_camera"] = {
                camera_id: grid.tolist()
                for camera_id, grid in sorted(self.attention_by_camera.items())
            }
        return out


def bev_token_from_prior_map(
    camera_map: CameraPriorMap,
    xy_m: Sequence[float],
    *,
    timestamp_s: float = 0.0,
    detector_score: float = 0.0,
    detection_valid: bool = False,
    measurement_age_s: float = 0.0,
    measurement_stale: bool = False,
    association_confidence: float = 1.0,
    recent_detector_history: Sequence[bool] = (),
    overlap_disagreement_m: float = math.nan,
    metadata: Mapping[str, Any] | None = None,
) -> BEVCameraToken:
    """Build one BEV token by sampling a calibrated per-camera prior map."""

    xy = _pair(xy_m, "xy_m")
    in_bounds = _contains_xy(camera_map.xs, camera_map.ys, xy)
    if in_bounds:
        geometry_prior = _bilinear(camera_map.xs, camera_map.ys, camera_map.probability, xy[0], xy[1])
        fov_value = _bilinear(camera_map.xs, camera_map.ys, camera_map.fov_mask.astype(float), xy[0], xy[1])
        feature_maps = {
            name: _bilinear(camera_map.xs, camera_map.ys, grid, xy[0], xy[1])
            for name, grid in camera_map.feature_maps.items()
        }
    else:
        geometry_prior = 0.0
        fov_value = 0.0
        feature_maps = {}
    return BEVCameraToken(
        camera_id=camera_map.camera_id,
        xy_m=xy,
        timestamp_s=timestamp_s,
        geometry_prior=geometry_prior,
        in_fov=fov_value >= 0.5,
        detector_score=detector_score,
        detection_valid=detection_valid,
        measurement_age_s=measurement_age_s,
        measurement_stale=measurement_stale,
        association_confidence=association_confidence,
        camera_relative_range_m=feature_maps.get("range_m", math.nan),
        border_margin_px=feature_maps.get("border_margin_px", math.nan),
        projection_scale_px_per_m=feature_maps.get("projection_scale_px_per_m", math.nan),
        ground_incidence_sin=feature_maps.get("ground_incidence_sin", math.nan),
        recent_detection_rate=_history_rate(recent_detector_history),
        overlap_disagreement_m=overlap_disagreement_m,
        metadata=metadata or {},
    )


def bev_tokens_from_prior_maps(
    priors: MultiCameraPriorMap | Mapping[str, CameraPriorMap],
    xy_m: Sequence[float],
    *,
    timestamp_s: float = 0.0,
    camera_measurements: Mapping[str, Mapping[str, Any]] | None = None,
    overlap_disagreement_m: float = math.nan,
) -> tuple[BEVCameraToken, ...]:
    """Build one operational token per camera for a candidate BEV state."""

    maps = dict(priors.camera_maps if isinstance(priors, MultiCameraPriorMap) else priors)
    if not maps:
        raise ContractValidationError("priors must contain at least one camera")
    measurements = dict(camera_measurements or {})
    tokens: list[BEVCameraToken] = []
    for camera_id, camera_map in sorted(maps.items()):
        measurement = dict(measurements.get(camera_id, {}))
        try:
            reject_evaluation_only_keys(measurement, context=f"camera_measurements[{camera_id}]")
        except LeakageError:
            raise
        tokens.append(
            bev_token_from_prior_map(
                camera_map,
                xy_m,
                timestamp_s=float(measurement.get("timestamp_s", timestamp_s)),
                detector_score=float(measurement.get("detector_score", 0.0)),
                detection_valid=_truthy(measurement.get("detection_valid", False)),
                measurement_age_s=float(measurement.get("measurement_age_s", 0.0)),
                measurement_stale=_truthy(measurement.get("measurement_stale", False)),
                association_confidence=float(measurement.get("association_confidence", 1.0)),
                recent_detector_history=tuple(measurement.get("recent_detector_history", ())),
                overlap_disagreement_m=float(measurement.get("overlap_disagreement_m", overlap_disagreement_m)),
                metadata={"source": "calibrated_prior_token"},
            )
        )
    return tuple(tokens)


def fit_bev_reliability_model(
    examples: Sequence[BEVReliabilityTrainingExample],
    *,
    feature_config: BEVFeatureConfig | None = None,
    initial_model: BEVReliabilityModel | None = None,
    learning_rate: float = 0.15,
    l2: float = 1.0e-3,
    steps: int = 250,
) -> BEVReliabilityModel:
    """Fit the lightweight linear BEV reliability model with logistic loss."""

    if not examples:
        raise ContractValidationError("examples must not be empty")
    cfg = feature_config or (initial_model.feature_config if initial_model else BEVFeatureConfig())
    lr = _finite(learning_rate, "learning_rate")
    reg = _finite(l2, "l2")
    if lr <= 0.0:
        raise ContractValidationError("learning_rate must be positive")
    if reg < 0.0:
        raise ContractValidationError("l2 must be non-negative")
    if int(steps) < 1:
        raise ContractValidationError("steps must be at least 1")

    x = np.vstack([example.token.feature_vector(cfg) for example in examples])
    y = np.asarray([example.target_reliable for example in examples], dtype=float)
    sample_weight = np.asarray([example.sample_weight for example in examples], dtype=float)
    if not np.any(sample_weight > 0.0):
        raise ContractValidationError("at least one example must have positive sample_weight")
    if initial_model is None:
        start_weights = dict(DEFAULT_BEV_FEATURE_WEIGHTS)
    else:
        start_weights = dict(initial_model.weights)
    w = np.asarray([float(start_weights.get(name, 0.0)) for name in cfg.feature_names], dtype=float)
    bias_idx = cfg.feature_names.index("bias") if "bias" in cfg.feature_names else -1
    denom = float(np.sum(sample_weight))

    for _ in range(int(steps)):
        pred = _sigmoid_array(x @ w)
        residual = (pred - y) * sample_weight
        grad = (x.T @ residual) / denom
        if reg > 0.0:
            penalty = w.copy()
            if bias_idx >= 0:
                penalty[bias_idx] = 0.0
            grad += reg * penalty
        w -= lr * grad

    return BEVReliabilityModel(
        feature_config=cfg,
        weights={name: float(w[idx]) for idx, name in enumerate(cfg.feature_names)},
        camera_biases=dict(initial_model.camera_biases) if initial_model else {},
        source_model="bev_reliability_linear_fit",
    )


def predict_bev_reliability_grid(
    model: BEVReliabilityModel,
    priors: MultiCameraPriorMap | Mapping[str, CameraPriorMap],
    *,
    timestamp_s: float = 0.0,
    camera_measurements: Mapping[str, Mapping[str, Any]] | None = None,
) -> BEVReliabilityGridPrediction:
    """Evaluate the model on every grid cell of a calibrated prior map."""

    maps = dict(priors.camera_maps if isinstance(priors, MultiCameraPriorMap) else priors)
    if not maps:
        raise ContractValidationError("priors must contain at least one camera")
    first = next(iter(maps.values()))
    xs = first.xs
    ys = first.ys
    for camera_id, camera_map in maps.items():
        if camera_map.xs != xs or camera_map.ys != ys:
            raise ContractValidationError(f"camera map {camera_id!r} does not share the grid")
    fused = np.zeros((len(ys), len(xs)), dtype=float)
    best = np.empty((len(ys), len(xs)), dtype=object)
    attention = {camera_id: np.zeros((len(ys), len(xs)), dtype=float) for camera_id in maps}
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            prediction = model.predict_set(
                bev_tokens_from_prior_maps(
                    maps,
                    (x, y),
                    timestamp_s=timestamp_s,
                    camera_measurements=camera_measurements,
                )
            )
            fused[iy, ix] = prediction.fused_probability
            best[iy, ix] = prediction.best_camera_id
            for camera_id, value in prediction.attention_weights.items():
                attention[camera_id][iy, ix] = value
    return BEVReliabilityGridPrediction(
        xs=xs,
        ys=ys,
        fused_probability=fused,
        best_camera_id=best,
        attention_by_camera=attention,
        source_model=model.source_model,
    )


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


def _contains_xy(xs: Sequence[float], ys: Sequence[float], xy_m: Sequence[float]) -> bool:
    x, y = _pair(xy_m, "xy_m")
    eps = 1.0e-9
    return bool(xs[0] - eps <= x <= xs[-1] + eps and ys[0] - eps <= y <= ys[-1] + eps)


def _bilinear(xs: Sequence[float], ys: Sequence[float], grid: Sequence[Sequence[float]], x: float, y: float) -> float:
    arr = np.asarray(grid, dtype=float)
    if arr.shape != (len(ys), len(xs)):
        raise ContractValidationError("grid shape must match xs/ys")
    x = min(max(float(x), xs[0]), xs[-1])
    y = min(max(float(y), ys[0]), ys[-1])
    ix = max(0, min(len(xs) - 2, int(np.searchsorted(xs, x, side="right") - 1)))
    iy = max(0, min(len(ys) - 2, int(np.searchsorted(ys, y, side="right") - 1)))
    x0, x1 = xs[ix], xs[ix + 1]
    y0, y1 = ys[iy], ys[iy + 1]
    wx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    wy = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
    q00 = float(arr[iy, ix])
    q10 = float(arr[iy, ix + 1])
    q01 = float(arr[iy + 1, ix])
    q11 = float(arr[iy + 1, ix + 1])
    value = (
        (1.0 - wx) * (1.0 - wy) * q00
        + wx * (1.0 - wy) * q10
        + (1.0 - wx) * wy * q01
        + wx * wy * q11
    )
    return float(value) if math.isfinite(value) else math.nan


def _pair(value: Sequence[float], field_name: str) -> tuple[float, float]:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ContractValidationError(f"{field_name} must be a 2-element sequence")
    return (_finite(value[0], f"{field_name}[0]"), _finite(value[1], f"{field_name}[1]"))


def _probability_mapping(values: Mapping[str, float], field_name: str) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise ContractValidationError(f"{field_name} must be a mapping")
    return {str(key): _probability(value, f"{field_name}[{key}]") for key, value in values.items()}


def _probability(value: float, field_name: str) -> float:
    out = _finite(value, field_name)
    if not 0.0 <= out <= 1.0:
        raise ContractValidationError(f"{field_name} must be in [0, 1]")
    return out


def _probability_or_nan(value: float, field_name: str) -> float:
    out = _finite_or_nan(value, field_name)
    if math.isnan(out):
        return out
    if not 0.0 <= out <= 1.0:
        raise ContractValidationError(f"{field_name} must be in [0, 1]")
    return out


def _finite_or_nan(value: float, field_name: str) -> float:
    if value is None:
        return math.nan
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"{field_name} must be numeric") from exc
    if math.isnan(out):
        return math.nan
    if not math.isfinite(out):
        raise ContractValidationError(f"{field_name} must be finite or NaN")
    return out


def _finite(value: float, field_name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"{field_name} must be numeric") from exc
    if not math.isfinite(out):
        raise ContractValidationError(f"{field_name} must be finite")
    return out


def _logit(value: float) -> float:
    p = min(max(float(value), 1.0e-4), 1.0 - 1.0e-4)
    return float(math.log(p / (1.0 - p)))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return float(1.0 / (1.0 + z))
    z = math.exp(value)
    return float(z / (1.0 + z))


def _sigmoid_array(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    out = np.empty_like(value)
    positive = value >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    out[~positive] = exp_value / (1.0 + exp_value)
    return out


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    shifted = logits - float(np.max(logits))
    weights = np.exp(shifted)
    denom = float(np.sum(weights))
    if denom <= 0.0 or not math.isfinite(denom):
        return np.full(logits.shape, 1.0 / float(logits.size), dtype=float)
    return weights / denom


def _range_score(value: float, *, good_range_m: float, range_decay_m: float) -> float:
    if not math.isfinite(value):
        return 0.5
    return float(math.exp(-max(float(value) - float(good_range_m), 0.0) / max(float(range_decay_m), 1.0e-9)))


def _linear_score(value: float, full_value: float) -> float:
    if not math.isfinite(value):
        return 0.5
    full = float(full_value)
    if full <= 0.0:
        return 1.0
    return float(min(max(float(value) / full, 0.0), 1.0))


def _incidence_score(value: float, minimum: float) -> float:
    if not math.isfinite(value):
        return 0.5
    minimum = min(max(float(minimum), 0.0), 0.99)
    value = min(max(float(value), 0.0), 1.0)
    return float(min(max((value - minimum) / max(1.0 - minimum, 1.0e-9), 0.0), 1.0))


def _overlap_agreement(disagreement_m: float, gate_m: float) -> float:
    if not math.isfinite(disagreement_m):
        return 0.5
    return float(math.exp(-max(float(disagreement_m), 0.0) / max(float(gate_m), 1.0e-9)))


def _history_rate(history: Sequence[bool]) -> float:
    if not history:
        return math.nan
    values = [1.0 if bool(item) else 0.0 for item in history]
    return float(sum(values) / len(values))


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    try:
        return float(value) >= 0.5
    except (TypeError, ValueError):
        return False


def _none_nan(value: float) -> float | None:
    return None if isinstance(value, float) and math.isnan(value) else float(value)
