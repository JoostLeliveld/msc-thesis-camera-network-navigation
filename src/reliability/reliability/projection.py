"""Ground-plane projection of camera observations from world-SDF poses.

Pixel to ground is **inverse perspective mapping and nothing else**: the detector's
box bottom-centre ray is intersected with the floor plane.  There are no parameters, no
contact-plane constant and no per-camera calibration.

That is not a simplification for its own sake, it is the measured optimum.  On 1844 real
detections, every fitted correction this module once supported scored **worse** than
applying none (2026-08-07; that study predates the 2026-08-25 clean sheet, so the numbers
below are the reason the parameters were removed, not a current result):

    raw IPM, 0 parameters   66.6 mm     <- this code
    v4, 2 parameters        70.1 mm
    v3, 10 parameters       74.5 mm
    v2, 8 parameters        68.2 mm

The cross-bearing term inverted and amplified the very bias it existed to remove on both
cameras it was fitted for (camera C +18.8 mm -> -58.7 mm), because commissioning constants
fitted in one region do not transfer.  The along-bearing term was mostly cancelling the
0.05 m contact plane, so two fitted parameters were paying to undo one free operator
constant.  Deleted 2026-08-07 rather than carried: `_project_pixel_to_world`,
`load_projection_calibration`, `load_projection_contact_z`, `projection_kwargs_for_camera`
and `tests/reliability/test_projection_cross_bearing.py`.  Git history is the record; the
v2/v3/v4 artifacts remain in `logs/studies/multicamera_commissioning_bigwarehouse/` as inert
evidence, each with a `SUPERSEDED.md`.

``unav_common`` is imported lazily to keep this package importable in ROS-independent
tooling contexts.
"""

from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

from reliability.contracts import CameraObservation

Matrix2x2 = tuple[tuple[float, float], tuple[float, float]]


def camera_model_from_world(world_sdf: str | Path, *, include_name: str):
    """Build a ground-plane camera model from an SDF ``<include>`` pose."""

    from unav_common.camera_model import ObliqueCameraModel

    root = ET.parse(Path(world_sdf)).getroot()
    pose_text = None
    matches = []
    for include in root.findall("./world/include"):
        name = (include.findtext("name") or "").strip()
        uri = (include.findtext("uri") or "").strip()
        model_name = uri.removeprefix("model://").split("/", 1)[0]
        if include_name in {name, model_name}:
            matches.append(include)
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous camera include {include_name!r}")
    if matches:
        pose = matches[0].find("pose")
        if pose is not None and (pose.get("relative_to") or pose.get("degrees", "false").lower() == "true" or pose.get("rotation_format", "euler_rpy") != "euler_rpy"):
            raise RuntimeError("Camera include requires an absolute world pose in Euler radians")
        pose_text = matches[0].findtext("pose")
    if not pose_text:
        raise RuntimeError(f"Could not find a posed camera include {include_name!r} in {world_sdf}")
    values = [float(value) for value in pose_text.split()]
    if len(values) != 6 or not all(math.isfinite(v) for v in values):
        raise RuntimeError(f"Camera pose for {include_name!r} must contain six values")
    x, y, z, roll, pitch, yaw = values
    if roll != 0.0 or z <= 0.0:
        raise RuntimeError("Floor camera loader requires zero roll and positive camera height")
    forward = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch))
    if forward[2] >= -1.0e-6:
        raise RuntimeError(f"Camera {include_name!r} does not point down towards the ground")
    scale = -z / forward[2]
    look_at = (x + scale * forward[0], y + scale * forward[1], 0.0)
    return ObliqueCameraModel(
        cam_pos=(x, y, z),
        look_at=look_at,
        img_width=1280,
        img_height=720,
        fov_h_rad=1.5708,
    )


def project_observation_to_world(
    observation: CameraObservation,
    camera,
) -> tuple[float, float] | None:
    """Project a valid camera observation into the metric warehouse frame.

    Inverse perspective mapping: intersect the observation's ray with the floor plane.
    Returns ``None`` for an invalid or pixel-less observation, or when the ray does not
    meet the ground in front of the camera.
    """

    if not observation.detection_valid or observation.pixel_uv is None:
        return None
    return camera.pixel_to_world(observation.pixel_uv[0], observation.pixel_uv[1])


def project_observation_to_world_with_covariance(
    observation: CameraObservation,
    camera,
    *,
    jacobian_step_px: float = 0.5,
    min_eigenvalue_m2: float = 1.0e-12,
) -> tuple[tuple[float, float], Matrix2x2] | None:
    """Project an observation and its full pixel covariance into map coordinates.

    This is the strict paper-1 precision-covariance reproduction path:
    ``CameraObservation.conditional_cov_uv`` is already the historical
    trust-to-R precision blend in px².  The projection is differentiated
    numerically and the covariance is propagated as ``R_xy = J R_uv Jᵀ``.  The tiny
    eigenvalue floor is numerical only; it is many orders of magnitude below any
    physical measurement covariance.
    """

    if not observation.detection_valid or observation.pixel_uv is None:
        return None
    step = float(jacobian_step_px)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("jacobian_step_px must be finite and positive")
    floor = float(min_eigenvalue_m2)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("min_eigenvalue_m2 must be finite and positive")

    from reliability.observation_geometry import validate_pixel_covariance
    validate_pixel_covariance(observation.conditional_cov_uv)
    u, v = observation.pixel_uv
    centre = camera.pixel_to_world(u, v)
    if centre is None or not all(math.isfinite(float(v)) for v in centre):
        return None
    du = _projection_derivative(camera, u, v, axis=0, step=step, centre=centre)
    dv = _projection_derivative(camera, u, v, axis=1, step=step, centre=centre)
    if du is None or dv is None:
        return None

    # J rows are map x/y and columns are pixel u/v.
    j00, j10 = du
    j01, j11 = dv
    r_uv = observation.conditional_cov_uv
    a = float(r_uv[0][0])
    b = 0.5 * (float(r_uv[0][1]) + float(r_uv[1][0]))
    d = float(r_uv[1][1])
    xx = j00 * j00 * a + 2.0 * j00 * j01 * b + j01 * j01 * d
    xy = j00 * j10 * a + (j00 * j11 + j01 * j10) * b + j01 * j11 * d
    yy = j10 * j10 * a + 2.0 * j10 * j11 * b + j11 * j11 * d
    try:
        covariance = _floor_spd_2x2(((xx, xy), (xy, yy)), floor)
    except ValueError:
        # Unsupported numerical geometry creates no measurement.
        return None
    return centre, covariance


def _projection_derivative(
    camera,
    u: float,
    v: float,
    *,
    axis: int,
    step: float,
    centre: tuple[float, float],
) -> tuple[float, float] | None:
    """Central projection derivative with a one-sided image-edge fallback."""

    plus = camera.pixel_to_world(
        u + (step if axis == 0 else 0.0),
        v + (step if axis == 1 else 0.0),
    )
    minus = camera.pixel_to_world(
        u - (step if axis == 0 else 0.0),
        v - (step if axis == 1 else 0.0),
    )
    if plus is not None and minus is not None:
        return (
            (float(plus[0]) - float(minus[0])) / (2.0 * step),
            (float(plus[1]) - float(minus[1])) / (2.0 * step),
        )
    if plus is not None:
        return (
            (float(plus[0]) - float(centre[0])) / step,
            (float(plus[1]) - float(centre[1])) / step,
        )
    if minus is not None:
        return (
            (float(centre[0]) - float(minus[0])) / step,
            (float(centre[1]) - float(minus[1])) / step,
        )
    return None


def _floor_spd_2x2(matrix: Matrix2x2, floor: float) -> Matrix2x2:
    """Add only enough isotropic jitter to give the matrix a numerical SPD floor."""

    import numpy as np
    values = np.asarray(matrix, dtype=float)
    if values.shape != (2, 2) or not np.isfinite(values).all():
        raise ValueError("projection covariance must be finite 2x2")
    if not math.isfinite(float(floor)) or floor <= 0:
        raise ValueError("projection covariance floor must be finite and positive")
    if not np.allclose(values, values.T, atol=1e-12, rtol=0):
        raise ValueError("projection covariance must be symmetric")
    a = float(matrix[0][0])
    b = 0.5 * (float(matrix[0][1]) + float(matrix[1][0]))
    d = float(matrix[1][1])
    minimum = 0.5 * (a + d) - math.hypot(0.5 * (a - d), b)
    jitter = max(0.0, float(floor) - minimum)
    result = ((a + jitter, b), (b, d + jitter))
    if not np.isfinite(result).all():
        raise ValueError("projection covariance overflow")
    try:
        np.linalg.cholesky(result)
    except np.linalg.LinAlgError as exc:
        raise ValueError("projection covariance floor is not representable at this scale") from exc
    return result
