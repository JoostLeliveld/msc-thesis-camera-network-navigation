"""Round-trip tests for ``PixelToBevTransformer.pixel_to_world_at_z``.

We project a known 3D world point through the camera's forward model, then
back-project the resulting pixel through the new at-z helper, and check we
recover the same (x, y).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
for rel in ('src/state', 'src/unav_common'):
    p = str((REPO_ROOT / rel).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)

from state.core.pixel_to_bev import PixelToBevTransformer
from unav_common.camera_model import ObliqueCameraModel


def _make_transformer():
    cam_pos = np.asarray([-2.45, -2.45, 2.80], dtype=float)
    look_at = np.asarray([0.0, 0.0, 0.0], dtype=float)
    return PixelToBevTransformer(
        cam_pos=cam_pos, look_at=look_at,
        img_width=1280, img_height=720, fov_h_rad=1.5708,
    )


def _project_world(camera: ObliqueCameraModel, xyz):
    cam_pt = camera.R @ (np.asarray(xyz, dtype=float) - camera.cam_pos)
    assert cam_pt[2] > 0
    h = camera.K @ cam_pt
    return float(h[0] / h[2]), float(h[1] / h[2])


def test_round_trip_at_marker_height_recovers_xy() -> None:
    transformer = _make_transformer()
    camera = transformer._make_camera(transformer.cam_pos, transformer.look_at)
    z_plane = 0.215
    for x_world, y_world in [(0.0, 0.0), (0.5, -0.3), (-1.2, 0.8), (1.5, 1.2)]:
        u, v = _project_world(camera, (x_world, y_world, z_plane))
        result = transformer.pixel_to_world_at_z(u, v, z_plane)
        assert result is not None
        assert math.isclose(result[0], x_world, abs_tol=1e-6)
        assert math.isclose(result[1], y_world, abs_tol=1e-6)


def test_z0_helper_matches_legacy_pixel_to_world() -> None:
    transformer = _make_transformer()
    camera = transformer._make_camera(transformer.cam_pos, transformer.look_at)
    u, v = _project_world(camera, (0.7, -0.4, 0.0))
    legacy = transformer.pixel_to_world(u, v)
    new = transformer.pixel_to_world_at_z(u, v, 0.0)
    assert legacy is not None and new is not None
    assert math.isclose(legacy[0], new[0], abs_tol=1e-6)
    assert math.isclose(legacy[1], new[1], abs_tol=1e-6)


def test_camera_world_to_pixel_respects_height() -> None:
    transformer = _make_transformer()
    camera = transformer._make_camera(transformer.cam_pos, transformer.look_at)
    xyz = (0.7, -0.4, 0.215)
    expected = _project_world(camera, xyz)
    actual = camera.world_to_pixel(*xyz)
    assert actual[2] is True
    assert math.isclose(actual[0], expected[0], abs_tol=1e-6)
    assert math.isclose(actual[1], expected[1], abs_tol=1e-6)


def test_returns_none_for_ray_parallel_to_plane() -> None:
    transformer = _make_transformer()
    # Pick u,v that produce a near-horizontal ray; for our angled camera this
    # corresponds to looking at the far edge of the image. We synthesise it by
    # asking for a z far ABOVE the camera so the ray must point downward but
    # we ask for an upper plane: t becomes negative.
    z_plane = transformer.cam_pos[2] + 5.0
    result = transformer.pixel_to_world_at_z(640, 360, z_plane)
    assert result is None


def test_yaw_from_back_projected_keypoints_matches_robot_yaw() -> None:
    """Project front/rear markers at robot pose, back-project at marker z,
    confirm BEV yaw matches the simulated robot yaw within numerical noise."""
    transformer = _make_transformer()
    camera = transformer._make_camera(transformer.cam_pos, transformer.look_at)
    z_plane = 0.215
    front_x, rear_x = 0.040, -0.090
    for x, y in [(0.0, 0.0), (1.0, -0.5), (-1.0, 0.8)]:
        for yaw in [0.0, 0.5, 1.2, math.pi - 0.3, -math.pi + 0.4]:
            c, s = math.cos(yaw), math.sin(yaw)
            front = (x + c * front_x, y + s * front_x, z_plane)
            rear = (x + c * rear_x,  y + s * rear_x,  z_plane)
            uf, vf = _project_world(camera, front)
            ur, vr = _project_world(camera, rear)
            f_xy = transformer.pixel_to_world_at_z(uf, vf, z_plane)
            r_xy = transformer.pixel_to_world_at_z(ur, vr, z_plane)
            assert f_xy is not None and r_xy is not None
            yaw_recovered = math.atan2(f_xy[1] - r_xy[1], f_xy[0] - r_xy[0])
            # wrap diff into (-pi, pi]
            d = math.atan2(math.sin(yaw_recovered - yaw), math.cos(yaw_recovered - yaw))
            assert abs(d) < 1e-3, f'yaw={yaw}, recovered={yaw_recovered}, diff={d}'
