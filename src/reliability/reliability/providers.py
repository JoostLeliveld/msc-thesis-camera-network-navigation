"""Camera-specific reliability providers for offline replay and future runtime use.

The providers in this module are ROS-independent and evaluation-free. They
consume only operational state summaries such as the current belief position and
return `CameraQuality` records that can be used by selection/fusion baselines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np

from reliability.contracts import CameraQuality, ContractValidationError
from reliability.single_camera_adapter import precision_blend_covariance


class CameraReliabilityProvider(Protocol):
    """Operational camera-quality query interface."""

    def query(
        self,
        camera_id: str,
        belief_xy: Sequence[float],
        timestamp_s: float = 0.0,
    ) -> CameraQuality:
        """Return a camera quality estimate for `camera_id` at `belief_xy`."""


@dataclass(frozen=True)
class FixedCameraReliabilityProvider:
    """Camera provider with a constant availability and pixel covariance."""

    camera_id: str
    p_available: float = 1.0
    conditional_cov_uv: tuple[tuple[float, float], tuple[float, float]] = (
        (2.5**2, 0.0),
        (0.0, 2.5**2),
    )
    association_confidence: float = 1.0
    epistemic_score: float = 0.0
    stale: bool = False
    source_model: str = "fixed_camera_reliability"

    def query(
        self,
        camera_id: str,
        belief_xy: Sequence[float],
        timestamp_s: float = 0.0,
    ) -> CameraQuality:
        _require_camera(camera_id, self.camera_id)
        _pair(belief_xy, "belief_xy")
        _finite(timestamp_s, "timestamp_s")
        return CameraQuality(
            camera_id=self.camera_id,
            p_available=self.p_available,
            conditional_cov_uv=self.conditional_cov_uv,
            association_confidence=self.association_confidence,
            epistemic_score=self.epistemic_score,
            stale=self.stale,
            source_model=self.source_model,
        )


@dataclass(frozen=True)
class GridMapReliabilityProvider:
    """Query a per-camera reliability grid from an `.npz` artifact.

    Artifacts follow the current thesis GP convention: 1-D `xs`, 1-D `ys`, and a
    probability grid indexed as `(y_index, x_index)`.
    """

    camera_id: str
    artifact_path: str | Path
    xs: tuple[float, ...]
    ys: tuple[float, ...]
    probability_map: tuple[tuple[float, ...], ...]
    map_key: str = "P_conservative_plan_map"
    epistemic_map: tuple[tuple[float, ...], ...] | None = None
    epistemic_map_key: str = "F_std_map"
    r_visible_uv: float = 2.5
    r_miss_uv: float = 40.0
    min_probability: float = 1.0e-4
    association_confidence: float = 1.0
    out_of_bounds_policy: str = "min"
    source_model: str = "gp_grid_reliability"
    artifact_sha256: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_npz(
        cls,
        artifact_path: str | Path,
        *,
        camera_id: str = "camera_A",
        map_key: str = "P_conservative_plan_map",
        epistemic_map_key: str = "F_std_map",
        r_visible_uv: float = 2.5,
        r_miss_uv: float = 40.0,
        min_probability: float = 1.0e-4,
        association_confidence: float = 1.0,
        out_of_bounds_policy: str = "min",
        source_model: str = "gp_grid_reliability",
    ) -> "GridMapReliabilityProvider":
        path = Path(artifact_path).expanduser()
        if not path.is_file():
            raise ContractValidationError(f"Reliability artifact not found: {path}")
        with np.load(path, allow_pickle=False) as data:
            if "xs" not in data.files or "ys" not in data.files:
                raise ContractValidationError(f"Reliability artifact {path} is missing xs/ys")
            if map_key not in data.files:
                raise ContractValidationError(f"Reliability artifact {path} is missing {map_key}")
            xs = _as_float_tuple(np.asarray(data["xs"], dtype=float).reshape(-1), "xs")
            ys = _as_float_tuple(np.asarray(data["ys"], dtype=float).reshape(-1), "ys")
            probability = _grid_to_tuple(
                np.asarray(data[map_key], dtype=float),
                expected_shape=(len(ys), len(xs)),
                field_name=map_key,
            )
            epistemic = None
            if epistemic_map_key and epistemic_map_key in data.files:
                epistemic = _grid_to_tuple(
                    np.asarray(data[epistemic_map_key], dtype=float),
                    expected_shape=(len(ys), len(xs)),
                    field_name=epistemic_map_key,
                )
        return cls(
            camera_id=camera_id,
            artifact_path=path.resolve(),
            xs=xs,
            ys=ys,
            probability_map=probability,
            map_key=map_key,
            epistemic_map=epistemic,
            epistemic_map_key=epistemic_map_key,
            r_visible_uv=r_visible_uv,
            r_miss_uv=r_miss_uv,
            min_probability=min_probability,
            association_confidence=association_confidence,
            out_of_bounds_policy=out_of_bounds_policy,
            source_model=source_model,
            artifact_sha256=_sha256_file(path),
            metadata={"artifact": str(path.resolve()), "map_key": map_key},
        )

    def __post_init__(self) -> None:
        xs = _as_float_tuple(self.xs, "xs")
        ys = _as_float_tuple(self.ys, "ys")
        _validate_axis(xs, "xs")
        _validate_axis(ys, "ys")
        probability = _grid_to_tuple(
            self.probability_map,
            expected_shape=(len(ys), len(xs)),
            field_name="probability_map",
        )
        clipped_probability = tuple(
            tuple(_clip_probability(value, self.min_probability) for value in row)
            for row in probability
        )
        epistemic = None
        if self.epistemic_map is not None:
            epistemic = _grid_to_tuple(
                self.epistemic_map,
                expected_shape=(len(ys), len(xs)),
                field_name="epistemic_map",
            )
            epistemic = tuple(tuple(max(float(value), 0.0) for value in row) for row in epistemic)
        policy = str(self.out_of_bounds_policy).strip().lower()
        if policy not in {"min", "clamp", "raise"}:
            raise ContractValidationError("out_of_bounds_policy must be one of min, clamp, raise")
        if not 0.0 <= float(self.min_probability) < 0.5:
            raise ContractValidationError("min_probability must be in [0, 0.5)")
        object.__setattr__(self, "artifact_path", str(Path(self.artifact_path)))
        object.__setattr__(self, "xs", xs)
        object.__setattr__(self, "ys", ys)
        object.__setattr__(self, "probability_map", clipped_probability)
        object.__setattr__(self, "epistemic_map", epistemic)
        object.__setattr__(self, "out_of_bounds_policy", policy)

    @property
    def x_min(self) -> float:
        return self.xs[0]

    @property
    def x_max(self) -> float:
        return self.xs[-1]

    @property
    def y_min(self) -> float:
        return self.ys[0]

    @property
    def y_max(self) -> float:
        return self.ys[-1]

    def contains_xy(self, xy: Sequence[float]) -> bool:
        x, y = _pair(xy, "xy")
        eps = 1.0e-9
        return bool(self.x_min - eps <= x <= self.x_max + eps and self.y_min - eps <= y <= self.y_max + eps)

    def probability_at(self, belief_xy: Sequence[float]) -> float:
        x, y, in_bounds = self._query_xy(belief_xy)
        if not in_bounds and self.out_of_bounds_policy == "min":
            return _clip_probability(self.min_probability, self.min_probability)
        return _clip_probability(
            _bilinear(self.xs, self.ys, self.probability_map, x, y),
            self.min_probability,
        )

    def epistemic_at(self, belief_xy: Sequence[float]) -> float:
        if self.epistemic_map is None:
            return 0.0
        x, y, in_bounds = self._query_xy(belief_xy)
        if not in_bounds and self.out_of_bounds_policy == "min":
            return 1.0
        return max(float(_bilinear(self.xs, self.ys, self.epistemic_map, x, y)), 0.0)

    def query(
        self,
        camera_id: str,
        belief_xy: Sequence[float],
        timestamp_s: float = 0.0,
    ) -> CameraQuality:
        _require_camera(camera_id, self.camera_id)
        _finite(timestamp_s, "timestamp_s")
        p_available = self.probability_at(belief_xy)
        return CameraQuality(
            camera_id=self.camera_id,
            p_available=p_available,
            conditional_cov_uv=precision_blend_covariance(
                p_available,
                r_visible_uv=self.r_visible_uv,
                r_miss_uv=self.r_miss_uv,
                min_probability=self.min_probability,
            ),
            association_confidence=self.association_confidence,
            epistemic_score=self.epistemic_at(belief_xy),
            stale=False,
            source_model=f"{self.source_model}:{self.map_key}",
        )

    def _query_xy(self, belief_xy: Sequence[float]) -> tuple[float, float, bool]:
        x, y = _pair(belief_xy, "belief_xy")
        in_bounds = self.contains_xy((x, y))
        if in_bounds:
            return x, y, True
        if self.out_of_bounds_policy == "raise":
            raise ContractValidationError(
                "Reliability query outside artifact support: "
                f"x={x:.3f}, y={y:.3f}, "
                f"support=[{self.x_min:.3f},{self.x_max:.3f}]x[{self.y_min:.3f},{self.y_max:.3f}]"
            )
        if self.out_of_bounds_policy == "clamp":
            return min(max(x, self.x_min), self.x_max), min(max(y, self.y_min), self.y_max), False
        return x, y, False


@dataclass(frozen=True)
class MultiCameraReliabilityProvider:
    """Dispatch quality queries to one provider per camera."""

    providers: Mapping[str, CameraReliabilityProvider]

    def __post_init__(self) -> None:
        if not self.providers:
            raise ContractValidationError("providers must contain at least one camera")
        object.__setattr__(self, "providers", dict(self.providers))

    def query(
        self,
        camera_id: str,
        belief_xy: Sequence[float],
        timestamp_s: float = 0.0,
    ) -> CameraQuality:
        if camera_id not in self.providers:
            raise ContractValidationError(f"Unknown camera_id {camera_id!r}")
        return self.providers[camera_id].query(camera_id, belief_xy, timestamp_s)

    def query_all(
        self,
        belief_xy: Sequence[float],
        timestamp_s: float = 0.0,
        camera_ids: Sequence[str] | None = None,
    ) -> dict[str, CameraQuality]:
        ids = tuple(camera_ids or self.providers.keys())
        return {camera_id: self.query(camera_id, belief_xy, timestamp_s) for camera_id in ids}


def quality_from_gp_artifact(
    artifact_path: str | Path,
    *,
    camera_id: str = "camera_A",
    belief_xy: Sequence[float],
    timestamp_s: float = 0.0,
    map_key: str = "P_conservative_plan_map",
    out_of_bounds_policy: str = "min",
) -> CameraQuality:
    """Convenience one-shot query for the current GP artifact format."""

    provider = GridMapReliabilityProvider.from_npz(
        artifact_path,
        camera_id=camera_id,
        map_key=map_key,
        out_of_bounds_policy=out_of_bounds_policy,
    )
    return provider.query(camera_id, belief_xy, timestamp_s)


def _require_camera(requested: str, expected: str) -> None:
    if str(requested) != str(expected):
        raise ContractValidationError(f"Provider for {expected!r} cannot answer camera {requested!r}")


def _finite(value: float, field_name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"{field_name} must be numeric") from exc
    if not math.isfinite(out):
        raise ContractValidationError(f"{field_name} must be finite")
    return out


def _pair(value: Sequence[float], field_name: str) -> tuple[float, float]:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ContractValidationError(f"{field_name} must be a 2-element sequence")
    return (_finite(value[0], f"{field_name}[0]"), _finite(value[1], f"{field_name}[1]"))


def _as_float_tuple(value: Sequence[float], field_name: str) -> tuple[float, ...]:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ContractValidationError(f"{field_name} must be a numeric sequence")
    out = tuple(_finite(item, f"{field_name}[]") for item in value)
    if not out:
        raise ContractValidationError(f"{field_name} must not be empty")
    return out


def _validate_axis(axis: Sequence[float], field_name: str) -> None:
    if len(axis) < 2:
        raise ContractValidationError(f"{field_name} must contain at least two values")
    for prev, current in zip(axis, axis[1:]):
        if current <= prev:
            raise ContractValidationError(f"{field_name} must be strictly increasing")


def _grid_to_tuple(
    value: Sequence[Sequence[float]],
    *,
    expected_shape: tuple[int, int],
    field_name: str,
) -> tuple[tuple[float, ...], ...]:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != expected_shape[0]:
        raise ContractValidationError(f"{field_name} must have shape {expected_shape}")
    rows: list[tuple[float, ...]] = []
    for row_idx, row in enumerate(value):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != expected_shape[1]:
            raise ContractValidationError(f"{field_name} must have shape {expected_shape}")
        rows.append(tuple(_finite(item, f"{field_name}[{row_idx}][]") for item in row))
    return tuple(rows)


def _clip_probability(value: float, min_probability: float) -> float:
    eps = max(float(min_probability), 0.0)
    return float(min(max(float(value), eps), 1.0 - eps))


def _bilinear(
    xs: Sequence[float],
    ys: Sequence[float],
    grid: Sequence[Sequence[float]],
    x: float,
    y: float,
) -> float:
    x = min(max(float(x), xs[0]), xs[-1])
    y = min(max(float(y), ys[0]), ys[-1])
    ix = max(0, min(len(xs) - 2, int(np.searchsorted(xs, x, side="right") - 1)))
    iy = max(0, min(len(ys) - 2, int(np.searchsorted(ys, y, side="right") - 1)))
    x0, x1 = xs[ix], xs[ix + 1]
    y0, y1 = ys[iy], ys[iy + 1]
    wx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    wy = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
    q00 = float(grid[iy][ix])
    q10 = float(grid[iy][ix + 1])
    q01 = float(grid[iy + 1][ix])
    q11 = float(grid[iy + 1][ix + 1])
    return (
        (1.0 - wx) * (1.0 - wy) * q00
        + wx * (1.0 - wy) * q10
        + (1.0 - wx) * wy * q01
        + wx * wy * q11
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()
