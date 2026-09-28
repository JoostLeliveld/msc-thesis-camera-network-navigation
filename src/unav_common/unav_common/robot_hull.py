"""Robot geometry used only to audit synthetic detector captures and labels.

This module is not an observation model and is not an input to any commissioned
correction. The thesis observation always begins with the detected bounding-box bottom
centre. These shapes remain solely because Stage-04 capture auditing must check whether
the rendered robot could produce a valid training label.

Two geometric supports are provided:

`BODY_PRISM`
    The 0.80 x 0.55 x 0.35 m bounding prism used for coarse capture support.

`VISUAL_HULL`
    Sampled visual links from the simulator URDF, used to audit rendered labels.

All coordinates are in the robot's base_footprint frame with the floor at z = 0.
`base_footprint -> base_link` is the xacro's 0.010 m base joint, so every base_link
number below is quoted with that 0.010 m already added.
"""

from __future__ import annotations

import math

import numpy as np

# --- xacro properties, verbatim -------------------------------------------------
BODY_L = 0.800
BODY_W = 0.550
DECK_TOP = 0.350          # deck_z1 0.340 in base_link + 0.010 base joint
BODY_H = 0.350            # the prism height D4 quotes
WHEEL_R = 0.100
WHEEL_SEP = 0.440
BASE_JOINT_Z = 0.010


def _box(length, width, height, centre):
    """The eight corners of an axis-aligned box.  Enough for a silhouette: the
    perspective image of a convex body's outline is the convex hull of its projected
    vertices, so extremes of the projection are attained at vertices."""
    hx, hy, hz = 0.5 * length, 0.5 * width, 0.5 * height
    cx, cy, cz = centre
    return np.array([[cx + sx * hx, cy + sy * hy, cz + sz * hz]
                     for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])


def _cylinder_y(radius, length, centre, facets=24):
    """A wheel: a cylinder whose axis lies along the robot's y axis."""
    cx, cy, cz = centre
    ang = np.linspace(0.0, 2.0 * math.pi, facets, endpoint=False)
    ring = np.stack([cx + radius * np.cos(ang), np.zeros_like(ang), cz + radius * np.sin(ang)], 1)
    out = []
    for dy in (-0.5 * length, 0.5 * length):
        shifted = ring.copy()
        shifted[:, 1] = cy + dy
        out.append(shifted)
    return np.concatenate(out)


def _sphere(radius, centre, count=32):
    """A caster.  Fibonacci points: no pole clustering, and the extremes of the
    projection are within a fraction of a pixel of the true outline at these ranges."""
    idx = np.arange(count) + 0.5
    phi = np.arccos(1.0 - 2.0 * idx / count)
    theta = math.pi * (1.0 + 5.0 ** 0.5) * idx
    unit = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)
    return np.asarray(centre) + radius * unit


def _visual_hull() -> np.ndarray:
    z = BASE_JOINT_Z
    parts = [
        # base_link chassis skirt: box 0.800 x 0.550 x 0.230 at base_link z 0.195
        _box(BODY_L, BODY_W, 0.230, (0.0, 0.0, 0.195 + z)),
        # hazard bumper band: body + 0.012 on each plan dimension, 0.040 tall at 0.100
        _box(BODY_L + 0.012, BODY_W + 0.012, 0.040, (0.0, 0.0, 0.100 + z)),
        # load deck: body - 0.020, 0.030 tall at 0.325
        _box(BODY_L - 0.020, BODY_W - 0.020, 0.030, (0.0, 0.0, 0.325 + z)),
        # recessed front sensor bar
        _box(0.030, 0.420, 0.070, (0.392, 0.0, 0.250 + z)),
        # deck slots
        _box(0.120, 0.070, 0.008, (0.300, 0.150, 0.341 + z)),
        _box(0.120, 0.070, 0.008, (0.300, -0.150, 0.341 + z)),
        # raised rear control cabinet -- the tallest thing on the robot, 0.400 m
        _box(0.170, 0.300, 0.050, (-0.280, 0.0, 0.365 + z)),
        # drive wheels: r 0.100, 0.050 wide, centred at base_link z = wheel_r - 0.010,
        # which puts the tyre bottom exactly on the floor
        _cylinder_y(WHEEL_R, 0.050, (0.0, 0.5 * WHEEL_SEP, WHEEL_R - 0.010 + z)),
        _cylinder_y(WHEEL_R, 0.050, (0.0, -0.5 * WHEEL_SEP, WHEEL_R - 0.010 + z)),
        # casters fore and aft, r 0.040 at base_link z 0.030 -- also floor-touching
        _sphere(0.040, (0.300, 0.0, 0.030 + z)),
        _sphere(0.040, (-0.300, 0.0, 0.030 + z)),
    ]
    return np.concatenate(parts)


BODY_PRISM = _box(BODY_L, BODY_W, BODY_H, (0.0, 0.0, 0.5 * BODY_H))
VISUAL_HULL = _visual_hull()
# Chassis without the ground-touching running gear: an ablation that isolates how much
# of the bottom row the wheels and casters are responsible for.
CHASSIS_ONLY = np.concatenate([
    _box(BODY_L, BODY_W, 0.230, (0.0, 0.0, 0.195 + BASE_JOINT_Z)),
    _box(BODY_L + 0.012, BODY_W + 0.012, 0.040, (0.0, 0.0, 0.100 + BASE_JOINT_Z)),
    _box(BODY_L - 0.020, BODY_W - 0.020, 0.030, (0.0, 0.0, 0.325 + BASE_JOINT_Z)),
    _box(0.170, 0.300, 0.050, (-0.280, 0.0, 0.365 + BASE_JOINT_Z)),
])

SHAPES = {"body_prism": BODY_PRISM, "visual_hull": VISUAL_HULL, "chassis_only": CHASSIS_ONLY}


def silhouette_box(camera, x, y, yaw, points=VISUAL_HULL):
    """Projected label-support box for offline capture auditing.

    Returns ``(u0, v0, u1, v1)`` or ``None`` when no part of the shape is in front of
    the camera.  Nothing is clipped to the image: a partly out-of-frame robot still has
    a well defined predicted box, and whether the reading is usable is a separate
    question answered by the capture's own gates.
    """
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    world = points @ rot.T + np.array([float(x), float(y), 0.0])
    in_camera = (world - camera.cam_pos) @ camera.R.T
    ahead = in_camera[:, 2] > 1.0e-6
    if not ahead.any():
        return None
    projected = (camera.K @ in_camera[ahead].T).T
    uv = projected[:, :2] / projected[:, 2:3]
    return (float(uv[:, 0].min()), float(uv[:, 1].min()),
            float(uv[:, 0].max()), float(uv[:, 1].max()))


def silhouette_bottom_centre(camera, x, y, yaw, points=VISUAL_HULL):
    """Bottom centre of the offline projected label-support box."""
    box = silhouette_box(camera, x, y, yaw, points)
    if box is None:
        return None
    u0, _v0, u1, v1 = box
    return 0.5 * (u0 + u1), v1
