"""Calibrated multi-camera day-zero reliability priors.

These utilities use known camera calibration only. They intentionally do not
consume detections, ground truth, CAD occluder geometry, or evaluation outcomes.
They are meant for offline research ablations before any planner-facing runtime
integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np

from reliability.contracts import ContractValidationError


@dataclass(frozen=True)
class CameraCalibration:
    camera_id: str
    cam_pos_m: tuple[float, float, float]
    look_at_m: tuple[float, float, float]
    img_width_px: int = 1280
    img_height_px: int = 720
    fov_h_rad: float = 1.5708
    up_hint: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        if not str(self.camera_id).strip():
            raise ContractValidationError("camera_id must be non-empty")
        object.__setattr__(self, "camera_id", str(self.camera_id))
        object.__setattr__(self, "cam_pos_m", _triple(self.cam_pos_m, "cam_pos_m"))
        object.__setattr__(self, "look_at_m", _triple(self.look_at_m, "look_at_m"))
        object.__setattr__(self, "up_hint", _triple(self.up_hint, "up_hint"))
        if int(self.img_width_px) <= 0 or int(self.img_height_px) <= 0:
            raise ContractValidationError("image dimensions must be positive")
        object.__setattr__(self, "img_width_px", int(self.img_width_px))
        object.__setattr__(self, "img_height_px", int(self.img_height_px))
        fov = _finite(self.fov_h_rad, "fov_h_rad")
        if not 0.0 < fov < math.pi:
            raise ContractValidationError("fov_h_rad must be in (0, pi)")
        object.__setattr__(self, "fov_h_rad", fov)
        _ = self.rotation_world_to_camera()

    @classmethod
    def from_gazebo_pose(
        cls,
        camera_id: str,
        pose_xyz_rpy: Sequence[float],
        *,
        img_width_px: int = 1280,
        img_height_px: int = 720,
        fov_h_rad: float = 1.5708,
    ) -> "CameraCalibration":
        pose = _as_tuple(pose_xyz_rpy, "pose_xyz_rpy", expected_len=6)
        cam_pos = pose[:3]
        look_at = compute_look_at_from_gazebo_pose(cam_pos, pose[3], pose[4], pose[5])
        return cls(
            camera_id=camera_id,
            cam_pos_m=cam_pos,
            look_at_m=look_at,
            img_width_px=img_width_px,
            img_height_px=img_height_px,
            fov_h_rad=fov_h_rad,
        )

    def intrinsics(self) -> np.ndarray:
        f = (float(self.img_width_px) / 2.0) / math.tan(float(self.fov_h_rad) / 2.0)
        cx = float(self.img_width_px) / 2.0
        cy = float(self.img_height_px) / 2.0
        return np.asarray([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=float)

    def rotation_world_to_camera(self) -> np.ndarray:
        cam_pos = np.asarray(self.cam_pos_m, dtype=float)
        look_at = np.asarray(self.look_at_m, dtype=float)
        up_hint = np.asarray(self.up_hint, dtype=float)
        z_cam = look_at - cam_pos
        z_norm = float(np.linalg.norm(z_cam))
        if z_norm <= 1.0e-12:
            raise ContractValidationError("look_at_m must differ from cam_pos_m")
        z_cam = z_cam / z_norm
        x_cam = np.cross(z_cam, up_hint)
        x_norm = float(np.linalg.norm(x_cam))
        if x_norm <= 1.0e-12:
            raise ContractValidationError("up_hint must not be parallel to viewing direction")
        x_cam = x_cam / x_norm
        y_cam = np.cross(z_cam, x_cam)
        y_cam = y_cam / float(np.linalg.norm(y_cam))
        return np.asarray([x_cam, y_cam, z_cam], dtype=float)

    def world_to_pixel(self, x_m: float, y_m: float, z_m: float = 0.0) -> tuple[float, float, bool]:
        cam_pos = np.asarray(self.cam_pos_m, dtype=float)
        world = np.asarray([float(x_m), float(y_m), float(z_m)], dtype=float)
        cam_pt = self.rotation_world_to_camera() @ (world - cam_pos)
        if cam_pt[2] <= 1.0e-10:
            return 0.0, 0.0, False
        uvw = self.intrinsics() @ cam_pt
        u = float(uvw[0] / uvw[2])
        v = float(uvw[1] / uvw[2])
        visible = bool(0.0 <= u < float(self.img_width_px) and 0.0 <= v < float(self.img_height_px))
        return u, v, visible

    def range_to(self, x_m: float, y_m: float, z_m: float = 0.0) -> float:
        cam_pos = np.asarray(self.cam_pos_m, dtype=float)
        point = np.asarray([float(x_m), float(y_m), float(z_m)], dtype=float)
        return float(np.linalg.norm(point - cam_pos))

    def ground_incidence_sin(self, x_m: float, y_m: float, z_m: float = 0.0) -> float:
        cam_pos = np.asarray(self.cam_pos_m, dtype=float)
        point = np.asarray([float(x_m), float(y_m), float(z_m)], dtype=float)
        ray = point - cam_pos
        norm = float(np.linalg.norm(ray))
        if norm <= 1.0e-12:
            return 0.0
        ray = ray / norm
        return float(abs(ray[2]))


@dataclass(frozen=True)
class PriorGridSpec:
    xs: tuple[float, ...]
    ys: tuple[float, ...]

    @classmethod
    def from_bounds(
        cls,
        *,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        nx: int,
        ny: int,
    ) -> "PriorGridSpec":
        if int(nx) < 2 or int(ny) < 2:
            raise ContractValidationError("nx and ny must be at least 2")
        return cls(
            xs=tuple(float(v) for v in np.linspace(float(x_min), float(x_max), int(nx))),
            ys=tuple(float(v) for v in np.linspace(float(y_min), float(y_max), int(ny))),
        )

    def __post_init__(self) -> None:
        xs = _axis(self.xs, "xs")
        ys = _axis(self.ys, "ys")
        object.__setattr__(self, "xs", xs)
        object.__setattr__(self, "ys", ys)


@dataclass(frozen=True)
class CalibratedPriorConfig:
    target_z_m: float = 0.0
    base_reliability: float = 0.98
    out_of_fov_probability: float = 0.0
    good_range_m: float = 6.0
    range_decay_m: float = 6.0
    min_border_margin_px: float = 20.0
    min_pixel_scale_px_per_m: float = 20.0
    min_ground_incidence_sin: float = 0.08
    min_probability: float = 0.0
    max_probability: float = 0.999

    def __post_init__(self) -> None:
        for field_name in (
            "target_z_m",
            "base_reliability",
            "out_of_fov_probability",
            "good_range_m",
            "range_decay_m",
            "min_border_margin_px",
            "min_pixel_scale_px_per_m",
            "min_ground_incidence_sin",
            "min_probability",
            "max_probability",
        ):
            _finite(getattr(self, field_name), field_name)
        if not 0.0 <= self.min_probability <= self.max_probability <= 1.0:
            raise ContractValidationError("probability bounds must satisfy 0 <= min <= max <= 1")


@dataclass(frozen=True)
class CameraPriorMap:
    camera_id: str
    xs: tuple[float, ...]
    ys: tuple[float, ...]
    probability: np.ndarray
    fov_mask: np.ndarray
    feature_maps: Mapping[str, np.ndarray] = field(default_factory=dict)
    config: CalibratedPriorConfig = field(default_factory=CalibratedPriorConfig)

    def __post_init__(self) -> None:
        xs = _axis(self.xs, "xs")
        ys = _axis(self.ys, "ys")
        probability = _grid(self.probability, len(ys), len(xs), "probability")
        fov_mask = np.asarray(self.fov_mask, dtype=bool)
        if fov_mask.shape != probability.shape:
            raise ContractValidationError("fov_mask must match probability shape")
        features = {
            str(name): _grid(value, len(ys), len(xs), f"feature_maps.{name}")
            for name, value in dict(self.feature_maps).items()
        }
        object.__setattr__(self, "camera_id", str(self.camera_id))
        object.__setattr__(self, "xs", xs)
        object.__setattr__(self, "ys", ys)
        object.__setattr__(self, "probability", probability)
        object.__setattr__(self, "fov_mask", fov_mask)
        object.__setattr__(self, "feature_maps", features)

    def to_dict(self, *, include_maps: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "camera_id": self.camera_id,
            "shape": [int(self.probability.shape[0]), int(self.probability.shape[1])],
            "x_min": self.xs[0],
            "x_max": self.xs[-1],
            "y_min": self.ys[0],
            "y_max": self.ys[-1],
            "coverage_fraction": float(np.mean(self.fov_mask)),
            "mean_probability": float(np.mean(self.probability)),
            "max_probability": float(np.max(self.probability)),
        }
        if include_maps:
            out["xs"] = list(self.xs)
            out["ys"] = list(self.ys)
            out["probability"] = self.probability.tolist()
            out["fov_mask"] = self.fov_mask.astype(int).tolist()
            out["feature_maps"] = {key: value.tolist() for key, value in self.feature_maps.items()}
        return out

    def to_grid_provider(self):
        from reliability.providers import GridMapReliabilityProvider

        return GridMapReliabilityProvider(
            camera_id=self.camera_id,
            artifact_path=f"<calibrated_prior:{self.camera_id}>",
            xs=self.xs,
            ys=self.ys,
            probability_map=tuple(tuple(float(v) for v in row) for row in self.probability),
            map_key="calibrated_geometry_prior",
            source_model="calibrated_multicamera_prior",
            out_of_bounds_policy="min",
        )


@dataclass(frozen=True)
class MultiCameraPriorMap:
    camera_maps: Mapping[str, CameraPriorMap]
    union_probability: np.ndarray
    best_probability: np.ndarray
    best_camera_id: np.ndarray
    coverage_count: np.ndarray

    def __post_init__(self) -> None:
        maps = dict(self.camera_maps)
        if not maps:
            raise ContractValidationError("camera_maps must not be empty")
        first = next(iter(maps.values()))
        shape = first.probability.shape
        for camera_id, cmap in maps.items():
            if cmap.probability.shape != shape:
                raise ContractValidationError(f"camera map {camera_id!r} shape does not match")
            if cmap.xs != first.xs or cmap.ys != first.ys:
                raise ContractValidationError("all camera maps must share xs/ys")
        object.__setattr__(self, "camera_maps", maps)
        object.__setattr__(self, "union_probability", _grid(self.union_probability, shape[0], shape[1], "union_probability"))
        object.__setattr__(self, "best_probability", _grid(self.best_probability, shape[0], shape[1], "best_probability"))
        best_camera_id = np.asarray(self.best_camera_id, dtype=object)
        if best_camera_id.shape != shape:
            raise ContractValidationError("best_camera_id must match probability shape")
        object.__setattr__(self, "best_camera_id", best_camera_id)
        object.__setattr__(self, "coverage_count", _grid(self.coverage_count, shape[0], shape[1], "coverage_count"))

    @property
    def xs(self) -> tuple[float, ...]:
        return next(iter(self.camera_maps.values())).xs

    @property
    def ys(self) -> tuple[float, ...]:
        return next(iter(self.camera_maps.values())).ys

    def to_dict(self, *, include_maps: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "camera_ids": sorted(self.camera_maps.keys()),
            "shape": [int(self.union_probability.shape[0]), int(self.union_probability.shape[1])],
            "union_coverage_fraction": float(np.mean(self.coverage_count > 0.0)),
            "multi_covered_fraction": float(np.mean(self.coverage_count > 1.0)),
            "mean_union_probability": float(np.mean(self.union_probability)),
            "mean_best_probability": float(np.mean(self.best_probability)),
            "camera_maps": {
                camera_id: cmap.to_dict(include_maps=False)
                for camera_id, cmap in sorted(self.camera_maps.items())
            },
        }
        if include_maps:
            out["xs"] = list(self.xs)
            out["ys"] = list(self.ys)
            out["union_probability"] = self.union_probability.tolist()
            out["best_probability"] = self.best_probability.tolist()
            out["best_camera_id"] = self.best_camera_id.tolist()
            out["coverage_count"] = self.coverage_count.tolist()
        return out


@dataclass(frozen=True)
class PriorEvaluationSummary:
    sample_count: int
    brier: float
    auroc: float
    predicted_reliable_fraction: float
    true_reliable_fraction: float
    blind_exposure_fraction: float
    mean_probability: float

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "sample_count": self.sample_count,
            "brier": _none_nan(self.brier),
            "auroc": _none_nan(self.auroc),
            "predicted_reliable_fraction": self.predicted_reliable_fraction,
            "true_reliable_fraction": self.true_reliable_fraction,
            "blind_exposure_fraction": self.blind_exposure_fraction,
            "mean_probability": self.mean_probability,
        }


def compute_look_at_from_gazebo_pose(
    cam_pos_xyz: Sequence[float],
    roll_rad: float,
    pitch_rad: float,
    yaw_rad: float,
) -> tuple[float, float, float]:
    cam_pos = _triple(cam_pos_xyz, "cam_pos_xyz")
    _finite(roll_rad, "roll_rad")
    pitch = _finite(pitch_rad, "pitch_rad")
    yaw = _finite(yaw_rad, "yaw_rad")
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    forward = (cp * cy, cp * sy, -sp)
    if forward[2] >= -1.0e-9:
        raise ContractValidationError("camera pose must point downward for ground-plane prior")
    t = -cam_pos[2] / forward[2]
    if t <= 0.0:
        raise ContractValidationError("camera forward ray must hit the ground in front")
    return (
        float(cam_pos[0] + t * forward[0]),
        float(cam_pos[1] + t * forward[1]),
        0.0,
    )


def build_camera_geometry_prior(
    calibration: CameraCalibration,
    grid: PriorGridSpec,
    config: CalibratedPriorConfig | None = None,
) -> CameraPriorMap:
    cfg = config or CalibratedPriorConfig()
    h = len(grid.ys)
    w = len(grid.xs)
    probability = np.zeros((h, w), dtype=float)
    fov_mask = np.zeros((h, w), dtype=bool)
    range_map = np.zeros((h, w), dtype=float)
    border_map = np.zeros((h, w), dtype=float)
    scale_map = np.zeros((h, w), dtype=float)
    incidence_map = np.zeros((h, w), dtype=float)

    for iy, y in enumerate(grid.ys):
        for ix, x in enumerate(grid.xs):
            u, v, visible = calibration.world_to_pixel(x, y, cfg.target_z_m)
            range_m = calibration.range_to(x, y, cfg.target_z_m)
            incidence = calibration.ground_incidence_sin(x, y, cfg.target_z_m)
            range_map[iy, ix] = range_m
            incidence_map[iy, ix] = incidence
            if not visible:
                probability[iy, ix] = cfg.out_of_fov_probability
                continue
            fov_mask[iy, ix] = True
            border = min(
                u,
                v,
                float(calibration.img_width_px) - 1.0 - u,
                float(calibration.img_height_px) - 1.0 - v,
            )
            border_map[iy, ix] = border
            scale = _projection_scale_px_per_m(calibration, x, y, cfg.target_z_m)
            scale_map[iy, ix] = scale
            scores = (
                _range_score(range_m, cfg.good_range_m, cfg.range_decay_m),
                _linear_score(border, cfg.min_border_margin_px),
                _linear_score(scale, cfg.min_pixel_scale_px_per_m),
                _incidence_score(incidence, cfg.min_ground_incidence_sin),
            )
            p = float(cfg.base_reliability)
            for score in scores:
                p *= score
            probability[iy, ix] = min(max(p, cfg.min_probability), cfg.max_probability)

    return CameraPriorMap(
        camera_id=calibration.camera_id,
        xs=grid.xs,
        ys=grid.ys,
        probability=probability,
        fov_mask=fov_mask,
        feature_maps={
            "range_m": range_map,
            "border_margin_px": border_map,
            "projection_scale_px_per_m": scale_map,
            "ground_incidence_sin": incidence_map,
        },
        config=cfg,
    )


def build_multicamera_geometry_prior(
    calibrations: Sequence[CameraCalibration],
    grid: PriorGridSpec,
    config: CalibratedPriorConfig | None = None,
) -> MultiCameraPriorMap:
    if not calibrations:
        raise ContractValidationError("calibrations must contain at least one camera")
    camera_maps = {
        calibration.camera_id: build_camera_geometry_prior(calibration, grid, config)
        for calibration in calibrations
    }
    return fuse_camera_prior_maps(camera_maps)


def fuse_camera_prior_maps(camera_maps: Mapping[str, CameraPriorMap]) -> MultiCameraPriorMap:
    maps = dict(camera_maps)
    if not maps:
        raise ContractValidationError("camera_maps must not be empty")
    ordered_ids = sorted(maps.keys())
    stack = np.stack([maps[camera_id].probability for camera_id in ordered_ids], axis=0)
    fov_stack = np.stack([maps[camera_id].fov_mask.astype(float) for camera_id in ordered_ids], axis=0)
    union = 1.0 - np.prod(1.0 - stack, axis=0)
    best_idx = np.argmax(stack, axis=0)
    best = np.max(stack, axis=0)
    best_ids = np.empty(best.shape, dtype=object)
    for idx, camera_id in enumerate(ordered_ids):
        best_ids[best_idx == idx] = camera_id
    best_ids[best <= 0.0] = ""
    return MultiCameraPriorMap(
        camera_maps=maps,
        union_probability=union,
        best_probability=best,
        best_camera_id=best_ids,
        coverage_count=np.sum(fov_stack, axis=0),
    )


def evaluate_probability_grid(
    probability: Sequence[Sequence[float]],
    labels: Sequence[Sequence[float]],
    *,
    mask: Sequence[Sequence[bool]] | None = None,
    reliability_threshold: float = 0.5,
) -> PriorEvaluationSummary:
    p = np.asarray(probability, dtype=float)
    y = np.asarray(labels, dtype=float)
    if p.shape != y.shape:
        raise ContractValidationError("probability and labels must have the same shape")
    if mask is None:
        valid = np.ones(p.shape, dtype=bool)
    else:
        valid = np.asarray(mask, dtype=bool)
        if valid.shape != p.shape:
            raise ContractValidationError("mask must match probability shape")
    valid &= np.isfinite(p) & np.isfinite(y)
    if not np.any(valid):
        return PriorEvaluationSummary(
            sample_count=0,
            brier=math.nan,
            auroc=math.nan,
            predicted_reliable_fraction=math.nan,
            true_reliable_fraction=math.nan,
            blind_exposure_fraction=math.nan,
            mean_probability=math.nan,
        )
    p1 = np.clip(p[valid], 0.0, 1.0)
    y1 = np.clip(y[valid], 0.0, 1.0)
    pred_reliable = p1 >= float(reliability_threshold)
    true_reliable = y1 >= float(reliability_threshold)
    blind_exposure = pred_reliable & ~true_reliable
    return PriorEvaluationSummary(
        sample_count=int(p1.size),
        brier=float(np.mean((p1 - y1) ** 2)),
        auroc=_binary_auc(p1, true_reliable.astype(int)),
        predicted_reliable_fraction=float(np.mean(pred_reliable)),
        true_reliable_fraction=float(np.mean(true_reliable)),
        blind_exposure_fraction=float(np.mean(blind_exposure)),
        mean_probability=float(np.mean(p1)),
    )


def _projection_scale_px_per_m(calibration: CameraCalibration, x: float, y: float, z: float) -> float:
    eps = 0.05
    u0, v0, ok0 = calibration.world_to_pixel(x, y, z)
    ux, vx, okx = calibration.world_to_pixel(x + eps, y, z)
    uy, vy, oky = calibration.world_to_pixel(x, y + eps, z)
    if not (ok0 and okx and oky):
        return 0.0
    du_dx = (ux - u0) / eps
    dv_dx = (vx - v0) / eps
    du_dy = (uy - u0) / eps
    dv_dy = (vy - v0) / eps
    return float(math.sqrt(max((du_dx * du_dx + dv_dx * dv_dx + du_dy * du_dy + dv_dy * dv_dy) / 2.0, 0.0)))


def _range_score(range_m: float, good_range_m: float, decay_m: float) -> float:
    good = max(float(good_range_m), 0.0)
    decay = max(float(decay_m), 1.0e-6)
    return float(math.exp(-max(float(range_m) - good, 0.0) / decay))


def _linear_score(value: float, full_value: float) -> float:
    full = float(full_value)
    if full <= 0.0:
        return 1.0
    return float(min(max(float(value) / full, 0.0), 1.0))


def _incidence_score(value: float, minimum: float) -> float:
    minimum = min(max(float(minimum), 0.0), 0.99)
    value = min(max(float(value), 0.0), 1.0)
    return float(min(max((value - minimum) / max(1.0 - minimum, 1.0e-6), 0.0), 1.0))


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return math.nan
    wins = 0.0
    for value in pos:
        wins += float(np.sum(value > neg))
        wins += 0.5 * float(np.sum(value == neg))
    return float(wins / (pos.size * neg.size))


def _axis(value: Sequence[float], field_name: str) -> tuple[float, ...]:
    out = _as_tuple(value, field_name)
    if len(out) < 2:
        raise ContractValidationError(f"{field_name} must contain at least two values")
    for prev, current in zip(out, out[1:]):
        if current <= prev:
            raise ContractValidationError(f"{field_name} must be strictly increasing")
    return out


def _grid(value: Sequence[Sequence[float]], height: int, width: int, field_name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (int(height), int(width)):
        raise ContractValidationError(f"{field_name} must have shape {(height, width)}")
    if not np.all(np.isfinite(arr)):
        raise ContractValidationError(f"{field_name} must be finite")
    return arr


def _triple(value: Sequence[float], field_name: str) -> tuple[float, float, float]:
    out = _as_tuple(value, field_name, expected_len=3)
    return (out[0], out[1], out[2])


def _as_tuple(value: Sequence[float], field_name: str, expected_len: int | None = None) -> tuple[float, ...]:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ContractValidationError(f"{field_name} must be a numeric sequence")
    if expected_len is not None and len(value) != expected_len:
        raise ContractValidationError(f"{field_name} must have length {expected_len}")
    return tuple(_finite(item, f"{field_name}[]") for item in value)


def _finite(value: float, field_name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"{field_name} must be numeric") from exc
    if not math.isfinite(out):
        raise ContractValidationError(f"{field_name} must be finite")
    return out


def _none_nan(value: float) -> float | None:
    return None if isinstance(value, float) and math.isnan(value) else float(value)
