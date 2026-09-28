"""Hash-bound validation for externally selected navigation polylines.

The closed-loop route experiment must execute the coordinates selected offline,
not a whitespace-dependent JSON blob and not a route repaired at launch time.
This module therefore owns one deterministic JSON representation and a fail-closed
geometry gate shared by campaign, launch, and planner code.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import math
from pathlib import Path
import re
from typing import Sequence

import numpy as np

from unav_common.occlusion_geometry import scene_from_json, signed_distance_to_union_xy


DEFAULT_ENDPOINT_TOLERANCE_M = 0.25
DEFAULT_SAMPLE_STEP_M = 0.04
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class PreselectedRouteError(ValueError):
    """Raised when a preselected route or its provenance fails validation."""


@dataclass(frozen=True)
class ValidatedPreselectedRoute:
    """A canonical route together with all launch-time gate measurements."""

    points: tuple[tuple[float, float], ...]
    canonical_json: str
    sha256: str
    source_path: str
    source_sha256: str
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    start_error_m: float
    goal_error_m: float
    endpoint_tolerance_m: float
    declared_clearance_m: float
    minimum_driveable_clearance_m: float
    sample_step_m: float
    clearance_sample_count: int
    length_m: float

    def provenance_dict(self) -> dict:
        """Return the JSON-ready record persisted beside the executed route."""

        return {
            "schema_version": 1,
            "route_json_canonicalization": (
                "json.dumps(float_xy,separators=(',',':'),ensure_ascii=False,allow_nan=False);"
                "negative_zero_normalized"
            ),
            "route_sha256": self.sha256,
            "route_points": [[x, y] for x, y in self.points],
            "route_point_count": len(self.points),
            "route_length_m": self.length_m,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "registered_start_xy": list(self.start_xy),
            "registered_goal_xy": list(self.goal_xy),
            "start_error_m": self.start_error_m,
            "goal_error_m": self.goal_error_m,
            "endpoint_tolerance_m": self.endpoint_tolerance_m,
            "declared_clearance_m": self.declared_clearance_m,
            "minimum_driveable_clearance_m": self.minimum_driveable_clearance_m,
            "clearance_sample_step_m": self.sample_step_m,
            "clearance_sample_count": self.clearance_sample_count,
            "validation_status": "passed",
        }


def _require_sha256(value: str, *, label: str) -> str:
    digest = str(value or "").strip()
    if _SHA256_RE.fullmatch(digest) is None:
        raise PreselectedRouteError(
            f"{label} must be exactly 64 lowercase hexadecimal characters"
        )
    return digest


def canonicalize_polyline_json(text: str) -> tuple[tuple[tuple[float, float], ...], str]:
    """Parse exactly one ``[[x,y], ...]`` polyline and serialize it canonically."""

    raw = str(text or "").strip()
    if not raw:
        raise PreselectedRouteError("preselected route JSON is empty")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PreselectedRouteError(f"preselected route is not valid JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise PreselectedRouteError(
            "preselected route must be exactly one JSON polyline (a list of [x,y] points)"
        )
    if len(payload) < 2:
        raise PreselectedRouteError("preselected route must contain at least two points")

    points: list[tuple[float, float]] = []
    for index, point in enumerate(payload):
        if not isinstance(point, list) or len(point) != 2:
            raise PreselectedRouteError(
                f"route point {index} must be a JSON list containing exactly [x,y]"
            )
        values: list[float] = []
        for axis, value in zip(("x", "y"), point):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise PreselectedRouteError(
                    f"route point {index} {axis} must be a finite JSON number"
                )
            number = float(value)
            if not math.isfinite(number):
                raise PreselectedRouteError(
                    f"route point {index} {axis} must be finite"
                )
            # JSON has no semantic distinction between positive and negative zero.
            # Normalizing it prevents two hashes for the same physical coordinate.
            values.append(0.0 if number == 0.0 else number)
        points.append((values[0], values[1]))

    for index, (a, b) in enumerate(zip(points, points[1:])):
        if math.hypot(b[0] - a[0], b[1] - a[1]) <= 1.0e-12:
            raise PreselectedRouteError(
                f"route points {index} and {index + 1} are duplicates; "
                "the executed polyline must be unambiguous"
            )

    canonical = json.dumps(
        [[x, y] for x, y in points],
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return tuple(points), canonical


def route_sha256(canonical_json: str) -> str:
    """SHA-256 of the UTF-8 canonical route bytes (with no trailing newline)."""

    return hashlib.sha256(str(canonical_json).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash an artifact without loading a potentially large geometry file at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source_artifact(path: str | Path, expected_sha256: str) -> tuple[str, str]:
    """Resolve and hash-bind the file from which the polyline was selected."""

    expected = _require_sha256(expected_sha256, label="preselected route source SHA-256")
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PreselectedRouteError(
            f"preselected route source artifact does not exist: {path}"
        ) from exc
    if not resolved.is_file():
        raise PreselectedRouteError(
            f"preselected route source artifact is not a regular file: {resolved}"
        )
    actual = sha256_file(resolved)
    if not hmac.compare_digest(actual, expected):
        raise PreselectedRouteError(
            f"preselected route source SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return str(resolved), actual


def sample_polyline(
    points: Sequence[Sequence[float]], *, maximum_step_m: float = DEFAULT_SAMPLE_STEP_M
) -> np.ndarray:
    """Sample every segment, including every vertex, at no more than ``maximum_step_m``."""

    step = float(maximum_step_m)
    if not math.isfinite(step) or step <= 0.0:
        raise PreselectedRouteError("clearance sample step must be finite and positive")
    route = np.asarray(points, dtype=float)
    if route.ndim != 2 or route.shape[0] < 2 or route.shape[1] != 2:
        raise PreselectedRouteError("route must have shape (N,2) with N >= 2")

    samples: list[np.ndarray] = [route[0].copy()]
    for p0, p1 in zip(route, route[1:]):
        distance = float(np.linalg.norm(p1 - p0))
        intervals = max(1, int(math.ceil(distance / step)))
        for index in range(1, intervals + 1):
            samples.append(p0 + (p1 - p0) * (index / intervals))
    return np.asarray(samples, dtype=float)


def validate_preselected_route(
    route_json: str,
    expected_route_sha256: str,
    *,
    start_xy: Sequence[float],
    goal_xy: Sequence[float],
    driveable_geometry_json: str,
    declared_clearance_m: float,
    source_path: str | Path,
    expected_source_sha256: str,
    endpoint_tolerance_m: float = DEFAULT_ENDPOINT_TOLERANCE_M,
    sample_step_m: float = DEFAULT_SAMPLE_STEP_M,
) -> ValidatedPreselectedRoute:
    """Validate identity, provenance, task endpoints, and full-segment clearance."""

    points, canonical = canonicalize_polyline_json(route_json)
    expected_route = _require_sha256(
        expected_route_sha256, label="preselected route SHA-256"
    )
    actual_route = route_sha256(canonical)
    if not hmac.compare_digest(actual_route, expected_route):
        raise PreselectedRouteError(
            f"preselected route SHA-256 mismatch: expected {expected_route}, got {actual_route}"
        )
    resolved_source, actual_source = verify_source_artifact(
        source_path, expected_source_sha256
    )

    start = np.asarray(start_xy, dtype=float).reshape(-1)
    goal = np.asarray(goal_xy, dtype=float).reshape(-1)
    if start.size != 2 or goal.size != 2 or not (
        np.all(np.isfinite(start)) and np.all(np.isfinite(goal))
    ):
        raise PreselectedRouteError("registered start and goal must each be finite [x,y]")

    tolerance = float(endpoint_tolerance_m)
    clearance_required = float(declared_clearance_m)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise PreselectedRouteError("endpoint tolerance must be finite and non-negative")
    if not math.isfinite(clearance_required) or clearance_required < 0.0:
        raise PreselectedRouteError("declared clearance must be finite and non-negative")

    route_array = np.asarray(points, dtype=float)
    start_error = float(np.linalg.norm(route_array[0] - start))
    goal_error = float(np.linalg.norm(route_array[-1] - goal))
    if start_error > tolerance + 1.0e-12:
        raise PreselectedRouteError(
            f"route start is {start_error:.6f} m from the registered task start; "
            f"maximum is {tolerance:.6f} m"
        )
    if goal_error > tolerance + 1.0e-12:
        raise PreselectedRouteError(
            f"route end is {goal_error:.6f} m from the registered task goal; "
            f"maximum is {tolerance:.6f} m"
        )

    scene = scene_from_json(driveable_geometry_json)
    if not scene.prisms:
        raise PreselectedRouteError(
            "driveable geometry is empty; route clearance cannot be validated"
        )
    samples = sample_polyline(points, maximum_step_m=sample_step_m)
    clearance = -np.asarray(
        signed_distance_to_union_xy(scene.prisms, samples, keep_in=True), dtype=float
    )
    if clearance.size == 0 or not np.all(np.isfinite(clearance)):
        raise PreselectedRouteError("driveable-clearance evaluation was non-finite")
    minimum_clearance = float(np.min(clearance))
    if minimum_clearance + 1.0e-9 < clearance_required:
        bad_index = int(np.argmin(clearance))
        bad = samples[bad_index]
        raise PreselectedRouteError(
            f"route driveable clearance is {minimum_clearance:.6f} m at "
            f"({bad[0]:.6f},{bad[1]:.6f}); declared minimum is "
            f"{clearance_required:.6f} m"
        )

    segment_lengths = np.linalg.norm(np.diff(route_array, axis=0), axis=1)
    return ValidatedPreselectedRoute(
        points=points,
        canonical_json=canonical,
        sha256=actual_route,
        source_path=resolved_source,
        source_sha256=actual_source,
        start_xy=(float(start[0]), float(start[1])),
        goal_xy=(float(goal[0]), float(goal[1])),
        start_error_m=start_error,
        goal_error_m=goal_error,
        endpoint_tolerance_m=tolerance,
        declared_clearance_m=clearance_required,
        minimum_driveable_clearance_m=minimum_clearance,
        sample_step_m=float(sample_step_m),
        clearance_sample_count=int(len(samples)),
        length_m=float(np.sum(segment_lengths)),
    )
