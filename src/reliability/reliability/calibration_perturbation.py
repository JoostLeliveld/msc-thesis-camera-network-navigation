"""Calibration-parameter perturbation for the E6 fault model (pre-reg §2).

The faithful E6 fault: perturb a camera's *calibration* (yaw/pitch/roll,
translation, principal-point, focal) and RE-PROJECT the recorded detection pixel
through the perturbed model. The same pixel then lands at a different world point,
and — because projection goes through the camera geometry — the world bias depends
on range and viewing angle. This is strictly stronger than the coarse constant
world-position bias (``bias_camera_position``): a 1 deg yaw drift barely moves a
near point and badly moves a far one, exactly the effect a real stale calibration
produces.

Images are unchanged (the detector still fires the same pixel); only the
estimator's calibration copy is perturbed — "controlled calibration-ablation
evidence" in the pre-registration. GROUND TRUTH IS NEVER USED.

Because the offline ``ReplayFrame`` stores the projected world point (not the
pixel), :func:`perturb_camera_calibration` recovers the pixel by projecting the
world point back through the *true* camera, then re-projects it through the
perturbed camera. On a real capture where the recorded pixel is available
(``load_commissioning_run`` re-projects from it), inject at load time with
:func:`reproject_world` on the recorded pixel for bit-faithfulness; the
perturbation math is identical either way.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from reliability.contracts import ContractValidationError
from reliability.fusion import MapObservation
from reliability.replay import ReplayFrame

Vec3 = tuple[float, float, float]
Mat3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


def _finite(value: float, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ContractValidationError(f"{name} must be finite")
    return out


@dataclass(frozen=True)
class PinholeGroundCamera:
    """A pinhole camera over a flat ground plane at ``ground_z_m``.

    ``rotation_cw`` maps world→camera (rows are the camera x=right, y=down,
    z=forward axes in world coordinates). Intrinsics are square-pixel
    ``fx/fy/cx/cy``. Build one with :meth:`looking_at` rather than by hand.
    """

    center_m: Vec3
    rotation_cw: Mat3
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    ground_z_m: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "center_m", tuple(_finite(v, "center_m") for v in self.center_m))
        rows = tuple(tuple(_finite(v, "rotation_cw") for v in row) for row in self.rotation_cw)
        if len(rows) != 3 or any(len(r) != 3 for r in rows):
            raise ContractValidationError("rotation_cw must be 3x3")
        object.__setattr__(self, "rotation_cw", rows)
        for name in ("fx", "fy", "cx", "cy", "ground_z_m"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ContractValidationError("fx, fy must be positive")

    @classmethod
    def looking_at(
        cls,
        center_m: Sequence[float],
        look_at_m: Sequence[float],
        *,
        fov_h_deg: float,
        width: int = 1280,
        height: int = 720,
        ground_z_m: float = 0.0,
        up: Sequence[float] = (0.0, 0.0, 1.0),
    ) -> "PinholeGroundCamera":
        c = np.asarray(center_m, dtype=float)
        target = np.asarray(look_at_m, dtype=float)
        up_v = np.asarray(up, dtype=float)
        forward = _normalize(target - c)  # camera +Z
        right = np.cross(forward, up_v)
        if float(np.linalg.norm(right)) < 1.0e-9:
            raise ContractValidationError(
                "camera forward is parallel to up; give a non-degenerate look_at/up "
                "(an oblique camera must not look exactly along the up axis)"
            )
        right = _normalize(right)  # camera +X
        down = np.cross(forward, right)  # camera +Y (image v grows downward)
        rotation_cw = np.vstack([right, down, forward])  # rows = camera axes in world
        fov = math.radians(_finite(fov_h_deg, "fov_h_deg"))
        if not 0.0 < fov < math.pi:
            raise ContractValidationError("fov_h_deg must be in (0, 180)")
        fx = (float(width) / 2.0) / math.tan(fov / 2.0)
        return cls(
            center_m=(float(c[0]), float(c[1]), float(c[2])),
            rotation_cw=_mat_to_tuple(rotation_cw),
            fx=fx,
            fy=fx,  # square pixels
            cx=float(width) / 2.0,
            cy=float(height) / 2.0,
            width=int(width),
            height=int(height),
            ground_z_m=float(ground_z_m),
        )

    def world_to_pixel(self, xy_m: Sequence[float], z_m: float | None = None) -> tuple[float, float] | None:
        """Project a world point to a pixel; ``None`` if it is behind the camera."""

        z = self.ground_z_m if z_m is None else float(z_m)
        p = np.asarray([float(xy_m[0]), float(xy_m[1]), z], dtype=float)
        r = _mat(self.rotation_cw)
        cam = r @ (p - np.asarray(self.center_m, dtype=float))
        if cam[2] <= 1.0e-9:
            return None
        u = self.fx * cam[0] / cam[2] + self.cx
        v = self.fy * cam[1] / cam[2] + self.cy
        return (float(u), float(v))

    def pixel_to_ground(self, u: float, v: float) -> tuple[float, float] | None:
        """Back-project a pixel to the ground plane; ``None`` if the ray misses it."""

        d_cam = np.asarray([(float(u) - self.cx) / self.fx, (float(v) - self.cy) / self.fy, 1.0], dtype=float)
        d_world = _mat(self.rotation_cw).T @ d_cam  # R_cw^T maps camera->world
        cz = float(self.center_m[2])
        if abs(d_world[2]) < 1.0e-12:
            return None
        t = (self.ground_z_m - cz) / d_world[2]
        if t <= 0.0:  # ground is behind the ray direction
            return None
        p = np.asarray(self.center_m, dtype=float) + t * d_world
        return (float(p[0]), float(p[1]))


@dataclass(frozen=True)
class CalibrationPerturbation:
    """A one-factor-or-compound calibration drift (pre-reg §2 ladder).

    Angles in degrees about the camera's own axes (yaw=about down, pitch=about
    right, roll=about forward); translation in metres of the camera centre;
    principal-point shift in pixels; ``focal_scale`` is a multiplicative focal
    error (1.0 = none, 1.01 = +1%). All default to a no-op.
    """

    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    tx_m: float = 0.0
    ty_m: float = 0.0
    tz_m: float = 0.0
    dcx_px: float = 0.0
    dcy_px: float = 0.0
    focal_scale: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "yaw_deg", "pitch_deg", "roll_deg", "tx_m", "ty_m", "tz_m",
            "dcx_px", "dcy_px", "focal_scale",
        ):
            _finite(getattr(self, name), name)
        if self.focal_scale <= 0.0:
            raise ContractValidationError("focal_scale must be positive")

    @property
    def is_identity(self) -> bool:
        return (
            self.yaw_deg == 0.0 and self.pitch_deg == 0.0 and self.roll_deg == 0.0
            and self.tx_m == 0.0 and self.ty_m == 0.0 and self.tz_m == 0.0
            and self.dcx_px == 0.0 and self.dcy_px == 0.0 and self.focal_scale == 1.0
        )


def perturb(camera: PinholeGroundCamera, perturbation: CalibrationPerturbation) -> PinholeGroundCamera:
    """Return a copy of ``camera`` with the calibration drift applied."""

    # Rotation drift in the camera frame: R_cw' = R_delta @ R_cw (rotate the camera
    # about its own axes; the camera centre is unchanged by rotation).
    r_delta = (
        _rot_z(math.radians(perturbation.roll_deg))
        @ _rot_y(math.radians(perturbation.yaw_deg))
        @ _rot_x(math.radians(perturbation.pitch_deg))
    )
    rotation_cw = r_delta @ _mat(camera.rotation_cw)
    center = (
        camera.center_m[0] + perturbation.tx_m,
        camera.center_m[1] + perturbation.ty_m,
        camera.center_m[2] + perturbation.tz_m,
    )
    return PinholeGroundCamera(
        center_m=center,
        rotation_cw=_mat_to_tuple(rotation_cw),
        fx=camera.fx * perturbation.focal_scale,
        fy=camera.fy * perturbation.focal_scale,
        cx=camera.cx + perturbation.dcx_px,
        cy=camera.cy + perturbation.dcy_px,
        width=camera.width,
        height=camera.height,
        ground_z_m=camera.ground_z_m,
    )


def reproject_world(
    camera: PinholeGroundCamera,
    xy_true_m: Sequence[float],
    perturbation: CalibrationPerturbation,
) -> tuple[float, float] | None:
    """World point after a calibration drift: recover the pixel through the true
    camera, then re-project it through the perturbed camera. ``None`` when the
    point is not projectable (behind the camera / ray misses the ground)."""

    pixel = camera.world_to_pixel(xy_true_m)
    if pixel is None:
        return None
    return perturb(camera, perturbation).pixel_to_ground(pixel[0], pixel[1])


def calibration_drift_world_bias(
    camera: PinholeGroundCamera,
    xy_true_m: Sequence[float],
    perturbation: CalibrationPerturbation,
) -> tuple[float, float]:
    """World-space bias (perturbed − true) induced at ``xy_true_m``.

    Returns ``(0.0, 0.0)`` when the point is not projectable (the fault leaves an
    unprojectable observation unchanged rather than inventing a bias)."""

    reprojected = reproject_world(camera, xy_true_m, perturbation)
    if reprojected is None:
        return (0.0, 0.0)
    return (reprojected[0] - float(xy_true_m[0]), reprojected[1] - float(xy_true_m[1]))


def perturb_camera_calibration(
    frames: Sequence[ReplayFrame],
    camera_id: str,
    camera: PinholeGroundCamera,
    perturbation: CalibrationPerturbation,
) -> list[ReplayFrame]:
    """E6 fault injector: re-project ``camera_id``'s observations through a drifted
    calibration, leaving every other camera untouched (parallels
    ``replay_sweeps.bias_camera_position`` but geometry-faithful).

    An observation whose world point is not projectable is left unchanged.
    """

    if perturbation.is_identity:
        return list(frames)
    out: list[ReplayFrame] = []
    for frame in frames:
        new_obs: list[MapObservation] = []
        for obs in frame.observations:
            if obs.camera_id != camera_id:
                new_obs.append(obs)
                continue
            reprojected = reproject_world(camera, obs.xy_m, perturbation)
            if reprojected is None:
                new_obs.append(obs)
                continue
            new_obs.append(
                MapObservation(
                    camera_id=obs.camera_id,
                    timestamp_s=obs.timestamp_s,
                    xy_m=reprojected,
                    covariance_m2=obs.covariance_m2,
                    quality=obs.quality,
                    source=f"{obs.source or 'obs'}+calib_drift",
                )
            )
        out.append(
            ReplayFrame(
                timestamp_s=frame.timestamp_s,
                odometry_xy_m=frame.odometry_xy_m,
                observations=tuple(new_obs),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# 3x3 helpers (numpy internally; tuples at the dataclass boundary).
# --------------------------------------------------------------------------- #


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1.0e-12:
        raise ContractValidationError("cannot normalize a zero-length vector")
    return v / n


def _mat(m: Mat3) -> np.ndarray:
    return np.asarray(m, dtype=float)


def _mat_to_tuple(m: np.ndarray) -> Mat3:
    return tuple(tuple(float(x) for x in row) for row in m)  # type: ignore[return-value]


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
