import math
import numpy as np

from unav_common.camera_model import ObliqueCameraModel


class PixelToBevTransformer:
    def __init__(self, cam_pos, look_at, img_width, img_height, fov_h_rad, rng=None):
        self.cam_pos = np.array(cam_pos, dtype=float)
        self.look_at = np.array(look_at, dtype=float)
        self.img_width = int(img_width)
        self.img_height = int(img_height)
        self.fov_h_rad = float(fov_h_rad)
        self.rng = rng

    def _make_camera(self, cam_pos, look_at):
        return ObliqueCameraModel(
            cam_pos=cam_pos,
            look_at=look_at,
            img_width=self.img_width,
            img_height=self.img_height,
            fov_h_rad=self.fov_h_rad,
        )

    def pixel_to_world(self, u, v, transform_noise_sigma=0.0):
        cam_pos = self.cam_pos.copy()
        look_at = self.look_at.copy()
        if transform_noise_sigma > 0.0 and self.rng is not None:
            cam_pos += self.rng.normal(0.0, transform_noise_sigma, size=3)
            look_at += self.rng.normal(0.0, transform_noise_sigma, size=3)

        camera = self._make_camera(cam_pos, look_at)
        return camera.pixel_to_world(u, v)

    def pixel_covariance_to_metric(self, u, v, covariance_px2, transform_noise_sigma=0.0):
        """Map full pixel covariance using derivatives of one nominal camera.

        Transform noise means independent uncertainty in the configured camera
        position and look-at coordinates (metres). Its six derivatives are added
        once as J_geometry sigma² I J_geometry.T, without random camera draws.
        This is a first-order covariance, not an empirical residual calibration.
        """
        R = np.asarray(covariance_px2, dtype=float)
        if R.shape != (2, 2) or not np.isfinite(R).all() or not np.allclose(R, R.T, atol=1e-12, rtol=0):
            raise ValueError("pixel covariance must be finite symmetric 2x2")
        if np.linalg.eigvalsh(R).min() < 0:
            raise ValueError("pixel covariance must be positive semidefinite")
        sigma = float(transform_noise_sigma)
        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError("transform noise sigma must be finite and nonnegative")
        camera = self._make_camera(self.cam_pos, self.look_at)
        if camera.pixel_to_world(u, v) is None:
            return None
        A = camera.H_inv
        w = A @ np.array([float(u), float(v), 1.0])
        J = (A[:2, :2]*w[2] - w[:2, None]*A[2, :2]) / w[2]**2
        metric = J @ R @ J.T
        if sigma > 0:
            # Central differences: each column perturbs exactly one geometry
            # coordinate about the same nominal configuration.
            geometry = np.concatenate((self.cam_pos, self.look_at))
            columns = []
            for axis in range(6):
                step = np.cbrt(np.finfo(float).eps) * max(1.0, abs(geometry[axis]))
                plus, minus = geometry.copy(), geometry.copy()
                plus[axis] += step
                minus[axis] -= step
                try:
                    high = self._make_camera(plus[:3], plus[3:]).pixel_to_world(u, v)
                    low = self._make_camera(minus[:3], minus[3:]).pixel_to_world(u, v)
                except ValueError:
                    return None
                if high is None or low is None:
                    return None
                columns.append((np.asarray(high)-low)/(2*step))
            geometry_J = np.column_stack(columns)
            metric += sigma**2 * geometry_J @ geometry_J.T
        if not np.isfinite(metric).all():
            return None
        return (metric + metric.T) / 2

    def pixel_noise_to_metric(self, u, v, pixel_noise_sigma, transform_noise_sigma=0.0):
        """World-axis marginal standard deviations; full covariance API is above."""
        sigma = float(pixel_noise_sigma)
        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError("pixel noise sigma must be finite and nonnegative")
        covariance = self.pixel_covariance_to_metric(
            u, v, np.eye(2)*sigma**2, transform_noise_sigma)
        if covariance is None:
            return None
        return tuple(np.sqrt(np.maximum(np.diag(covariance), 0.0)))

    def pixel_to_world_at_z(self, u, v, z_plane, transform_noise_sigma=0.0):
        """Back-project a pixel to the world plane z=z_plane.

        The default ``pixel_to_world`` uses the planar homography for z=0; for
        elevated keypoints (e.g. front/rear marker discs at z ~= 0.215) using
        the ground-plane homography produces a biased BEV position. Here we
        intersect the camera ray with z=z_plane analytically.

        Returns (x, y) in world coordinates or None if the ray is parallel to
        the plane / behind the camera.
        """
        cam_pos = self.cam_pos.copy()
        look_at = self.look_at.copy()
        if transform_noise_sigma > 0.0 and self.rng is not None:
            cam_pos = cam_pos + self.rng.normal(0.0, transform_noise_sigma, size=3)
            look_at = look_at + self.rng.normal(0.0, transform_noise_sigma, size=3)
        camera = self._make_camera(cam_pos, look_at)
        return camera.pixel_to_world_at_z(u, v, z_plane)
