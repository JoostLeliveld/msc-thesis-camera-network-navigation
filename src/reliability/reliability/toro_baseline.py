"""Toro-style calibrated-covariance baseline (plan 01, B2a/B2b).

Re-implementation of the comparison method (Toro Diz et al.): per-camera 2x2
measurement covariance taken from the *nearest static calibration location*
(Euclidean nearest neighbour, no interpolation), a validated-FOV gate built
from the calibration points, exact time binning at 0.08 s (B2a), and a shared
constant-velocity process model for replay fusion.

Pure library: no ROS, no numpy required. 2x2 math reuses the hand-rolled
helpers from :mod:`reliability.fusion`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Iterable, Mapping, Sequence

from reliability.contracts import ContractValidationError
from reliability.fusion import (
    MapObservation,
    _matrix_2x2,
    _pair,
    _validate_spd,
)


_EDGE_EPS = 1.0e-9
_BIN_EPS = 1.0e-9
_PSD_REL_TOL = 1.0e-10

State4 = tuple[float, float, float, float]
Cov4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


@dataclass(frozen=True)
class CalibrationPoint:
    """One static calibration location for one camera.

    ``covariance_xy`` is the axis-specific 2x2 covariance measured at
    standstill at ``position_xy``; ``sample_count`` records how many samples
    produced it (calibration points are measured, not perfect poses).
    """

    camera_id: str
    position_xy: tuple[float, float]
    covariance_xy: tuple[tuple[float, float], tuple[float, float]]
    sample_count: int

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ContractValidationError("camera_id must be a non-empty string")
        position = _pair(self.position_xy, "position_xy")
        covariance = _matrix_2x2(self.covariance_xy, "covariance_xy")
        _validate_spd(covariance, "covariance_xy")
        if isinstance(self.sample_count, bool) or not isinstance(
            self.sample_count, Integral
        ):
            raise ContractValidationError("sample_count must be a positive integer")
        sample_count = int(self.sample_count)
        if sample_count < 1:
            raise ContractValidationError("sample_count must be a positive integer")
        object.__setattr__(self, "position_xy", position)
        object.__setattr__(self, "covariance_xy", covariance)
        object.__setattr__(self, "sample_count", sample_count)


class ToroCovarianceModel:
    """Nearest-calibration-point covariance model with a validated-FOV gate.

    ``covariance_for`` returns the covariance of the Euclidean-nearest
    calibration point (no interpolation; ties break to the earliest point in
    construction order). ``in_validated_fov`` tests membership in the convex
    hull of the camera's calibration points (boundary inclusive); with fewer
    than three non-collinear points the hull degenerates to a segment or a
    single point and membership means lying on it (within 1e-9).
    """

    def __init__(self, calibration_points: Iterable[CalibrationPoint]) -> None:
        points_by_camera: dict[str, list[CalibrationPoint]] = {}
        for point in calibration_points:
            if not isinstance(point, CalibrationPoint):
                raise ContractValidationError(
                    "calibration_points must contain CalibrationPoint instances"
                )
            points_by_camera.setdefault(point.camera_id, []).append(point)
        self._points_by_camera: Mapping[str, tuple[CalibrationPoint, ...]] = {
            camera_id: tuple(points) for camera_id, points in points_by_camera.items()
        }
        self._support_polygon_by_camera: dict[str, tuple[tuple[float, float], ...]] = {
            camera_id: _convex_hull([p.position_xy for p in points])
            for camera_id, points in self._points_by_camera.items()
        }

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return tuple(self._points_by_camera.keys())

    def calibration_points(self, camera_id: str) -> tuple[CalibrationPoint, ...]:
        return self._camera_points(camera_id)

    def covariance_for(
        self, camera_id: str, position_xy: Sequence[float]
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        query = _pair(position_xy, "position_xy")
        points = self._camera_points(camera_id)
        best_point = points[0]
        best_dist = math.hypot(
            query[0] - best_point.position_xy[0], query[1] - best_point.position_xy[1]
        )
        for point in points[1:]:
            dist = math.hypot(
                query[0] - point.position_xy[0], query[1] - point.position_xy[1]
            )
            if dist < best_dist:
                best_dist = dist
                best_point = point
        return best_point.covariance_xy

    def in_validated_fov(self, camera_id: str, position_xy: Sequence[float]) -> bool:
        query = _pair(position_xy, "position_xy")
        self._camera_points(camera_id)
        support_polygon = self._support_polygon_by_camera[camera_id]
        return _point_in_convex_hull(query, support_polygon)

    def _camera_points(self, camera_id: str) -> tuple[CalibrationPoint, ...]:
        points = self._points_by_camera.get(camera_id)
        if not points:
            raise ContractValidationError(
                f"no calibration points for camera {camera_id!r}"
            )
        return points


def bin_observations(
    observations: Iterable[MapObservation],
    bin_width_s: float = 0.08,
) -> list[list[MapObservation]]:
    """Group observations into exact fixed-width time bins (B2a).

    Bins are anchored at absolute time zero: an observation with timestamp
    ``t`` falls into bin ``floor(t / bin_width_s + 1e-9)`` — the small epsilon
    keeps timestamps that lie exactly on a bin edge (e.g. 0.16 with width
    0.08) in the *later* bin despite binary floating-point rounding.
    Observations are sorted by timestamp first, so out-of-order input is
    handled; only non-empty bins are returned, in time order.
    """

    width = float(bin_width_s)
    if not math.isfinite(width) or width <= 0.0:
        raise ContractValidationError("bin_width_s must be a finite positive float")
    stamped: list[tuple[float, MapObservation]] = []
    for obs in observations:
        timestamp = float(obs.timestamp_s)
        if not math.isfinite(timestamp):
            raise ContractValidationError("observation timestamp_s must be finite")
        stamped.append((timestamp, obs))
    stamped.sort(key=lambda item: item[0])
    bins: list[list[MapObservation]] = []
    current_index: int | None = None
    for timestamp, obs in stamped:
        bin_index = math.floor(timestamp / width + _BIN_EPS)
        if bin_index != current_index:
            bins.append([])
            current_index = bin_index
        bins[-1].append(obs)
    return bins


def constant_velocity_predict(
    state4: Sequence[float],
    cov4x4: Sequence[Sequence[float]],
    dt: float,
    q: float,
) -> tuple[State4, Cov4]:
    """Shared constant-velocity prediction step for replay fusion.

    State is ``[x, y, vx, vy]``. Process noise is the standard white-noise
    acceleration model with intensity ``q`` (per axis)::

        Q = q * [[dt^3/3, 0,      dt^2/2, 0     ],
                 [0,      dt^3/3, 0,      dt^2/2],
                 [dt^2/2, 0,      dt,     0     ],
                 [0,      dt^2/2, 0,      dt    ]]

    ``q`` must be tuned on validation data only and shared across all replay
    conditions (plan 01 gate: no per-method Q).
    """

    state = _vector_4(state4, "state4")
    cov = _matrix_4x4(cov4x4, "cov4x4")
    _validate_psd_4(cov, "cov4x4")
    dt_f = float(dt)
    q_f = float(q)
    if not math.isfinite(dt_f) or dt_f < 0.0:
        raise ContractValidationError("dt must be finite and non-negative")
    if not math.isfinite(q_f) or q_f < 0.0:
        raise ContractValidationError("q must be finite and non-negative")

    transition = (
        (1.0, 0.0, dt_f, 0.0),
        (0.0, 1.0, 0.0, dt_f),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    dt2 = dt_f * dt_f
    dt3 = dt2 * dt_f
    process_noise = (
        (q_f * dt3 / 3.0, 0.0, q_f * dt2 / 2.0, 0.0),
        (0.0, q_f * dt3 / 3.0, 0.0, q_f * dt2 / 2.0),
        (q_f * dt2 / 2.0, 0.0, q_f * dt_f, 0.0),
        (0.0, q_f * dt2 / 2.0, 0.0, q_f * dt_f),
    )
    new_state = (
        state[0] + state[2] * dt_f,
        state[1] + state[3] * dt_f,
        state[2],
        state[3],
    )
    propagated = _mat_mul_4(_mat_mul_4(transition, cov), _transpose_4(transition))
    new_cov = _symmetrize_4(_mat_add_4(propagated, process_noise))
    return new_state, new_cov


# ---------------------------------------------------------------------------
# Convex-hull helpers (cross products only, no scipy)
# ---------------------------------------------------------------------------


def _cross(
    origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> float:
    return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])


def _convex_hull(points: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    """Monotone-chain convex hull; returns CCW vertices without collinear points.

    Degenerate inputs return fewer than three vertices: one vertex for a
    single distinct point, two for collinear points (segment endpoints).
    """

    unique = sorted(set((float(p[0]), float(p[1])) for p in points))
    if len(unique) <= 2:
        return tuple(unique)
    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:  # all points collinear
        return (unique[0], unique[-1])
    return tuple(hull)


def _point_in_convex_hull(
    point: tuple[float, float], hull: Sequence[tuple[float, float]]
) -> bool:
    if len(hull) == 0:
        return False
    if len(hull) == 1:
        return math.hypot(point[0] - hull[0][0], point[1] - hull[0][1]) <= _EDGE_EPS
    if len(hull) == 2:
        return _point_on_segment(point, hull[0], hull[1])
    for i, vertex in enumerate(hull):
        nxt = hull[(i + 1) % len(hull)]
        if _cross(vertex, nxt, point) < -_EDGE_EPS:
            return False
    return True


def _point_on_segment(
    point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> bool:
    if abs(_cross(a, b, point)) > _EDGE_EPS:
        return False
    dot = (point[0] - a[0]) * (b[0] - a[0]) + (point[1] - a[1]) * (b[1] - a[1])
    seg_len_sq = (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2
    return -_EDGE_EPS <= dot <= seg_len_sq + _EDGE_EPS


# ---------------------------------------------------------------------------
# 4x4 helpers for the constant-velocity model
# ---------------------------------------------------------------------------


def _vector_4(value: Sequence[float], field_name: str) -> State4:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ContractValidationError(f"{field_name} must be a 4-element sequence")
    out = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in out):
        raise ContractValidationError(f"{field_name} must be finite")
    return out  # type: ignore[return-value]


def _matrix_4x4(value: Sequence[Sequence[float]], field_name: str) -> Cov4:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ContractValidationError(f"{field_name} must be a 4x4 matrix")
    rows: list[tuple[float, float, float, float]] = []
    for row in value:
        if hasattr(row, "tolist") and not isinstance(row, (str, bytes)):
            row = row.tolist()
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != 4:
            raise ContractValidationError(f"{field_name} must be a 4x4 matrix")
        parsed = tuple(float(item) for item in row)
        if not all(math.isfinite(item) for item in parsed):
            raise ContractValidationError(f"{field_name} must be finite")
        rows.append(parsed)  # type: ignore[arg-type]
    matrix = tuple(rows)
    for i in range(4):
        for j in range(i + 1, 4):
            if abs(matrix[i][j] - matrix[j][i]) > 1.0e-9:
                raise ContractValidationError(f"{field_name} must be symmetric")
    return matrix  # type: ignore[return-value]


def _validate_psd_4(matrix: Cov4, field_name: str) -> None:
    """Validate positive semidefiniteness with a pivoted LDL-style check.

    Schur-complement pivots whose magnitude is at most ``1e-10`` times the
    largest input magnitude are treated as zero. This permits exactly singular
    covariance matrices and small negative round-off at a semidefinite boundary
    while still rejecting materially indefinite symmetric matrices.
    """

    scale = max(abs(matrix[i][j]) for i in range(4) for j in range(4))
    if scale == 0.0:
        return
    tolerance = _PSD_REL_TOL * scale
    residual = [
        [0.5 * (matrix[i][j] + matrix[j][i]) for j in range(4)]
        for i in range(4)
    ]

    for column in range(4):
        # A diagonal pivot avoids dividing by an early zero pivot in a valid
        # semidefinite matrix. Apply the same permutation to rows and columns.
        pivot_index = max(
            range(column, 4), key=lambda index: residual[index][index]
        )
        if pivot_index != column:
            residual[column], residual[pivot_index] = (
                residual[pivot_index],
                residual[column],
            )
            for row in residual:
                row[column], row[pivot_index] = row[pivot_index], row[column]

        pivot = residual[column][column]
        if pivot < -tolerance:
            raise ContractValidationError(
                f"{field_name} must be positive semidefinite"
            )
        if pivot <= tolerance:
            if any(
                abs(residual[column][index]) > tolerance
                for index in range(column + 1, 4)
            ):
                raise ContractValidationError(
                    f"{field_name} must be positive semidefinite"
                )
            continue

        for row in range(column + 1, 4):
            for col in range(row, 4):
                updated = (
                    residual[row][col]
                    - residual[row][column]
                    * residual[col][column]
                    / pivot
                )
                residual[row][col] = updated
                residual[col][row] = updated


def _mat_mul_4(a: Cov4, b: Cov4) -> Cov4:
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4))
        for i in range(4)
    )  # type: ignore[return-value]


def _transpose_4(a: Cov4) -> Cov4:
    return tuple(tuple(a[j][i] for j in range(4)) for i in range(4))  # type: ignore[return-value]


def _mat_add_4(a: Cov4, b: Cov4) -> Cov4:
    return tuple(
        tuple(a[i][j] + b[i][j] for j in range(4)) for i in range(4)
    )  # type: ignore[return-value]


def _symmetrize_4(a: Cov4) -> Cov4:
    return tuple(
        tuple(0.5 * (a[i][j] + a[j][i]) for j in range(4)) for i in range(4)
    )  # type: ignore[return-value]
