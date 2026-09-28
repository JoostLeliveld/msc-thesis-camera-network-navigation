"""Canonical R0--R2 covariances retain ray anisotropy at query time."""

import numpy as np

from reliability.commissioned_visibility import CommissionedVisibilitySensorModel


def test_canonical_ray_covariance_does_not_require_robot_heading():
    model = CommissionedVisibilitySensorModel.__new__(CommissionedVisibilitySensorModel)
    model.projection_sigma_px = None
    model.ray_r_samples = {"camera_A": {}}
    model.current_r_samples = None

    assert model.requires_capture_heading is False


def test_archived_heading_conditioned_covariance_declares_heading_dependency():
    model = CommissionedVisibilitySensorModel.__new__(CommissionedVisibilitySensorModel)
    model.projection_sigma_px = None
    model.ray_r_samples = None
    model.current_r_samples = None

    assert model.requires_capture_heading is True


def test_global_ray_covariance_rotates_into_world_at_planning_query():
    model = CommissionedVisibilitySensorModel.__new__(CommissionedVisibilitySensorModel)
    model.ray_r_samples = {"camera_A": {"xy": np.asarray([[0.0, 1.0]]),
                                        "moment": np.asarray([np.diag([4.0, 1.0])])}}
    model.current_r_samples = None
    model.runtime_covariance_model = "R0_global_full"
    model.ray_r_global = np.diag([4.0, 1.0])
    model.camera_xy = {"camera_A": np.asarray([0.0, 0.0])}

    along_x = model.planner_covariance("camera_A", 2.0, 0.0, 0.0)
    along_y = model.planner_covariance("camera_A", 0.0, 2.0, 0.0)

    np.testing.assert_allclose(along_x, np.diag([4.0, 1.0]))
    np.testing.assert_allclose(along_y, np.diag([1.0, 4.0]), atol=1e-12)
    np.testing.assert_allclose(np.linalg.eigvalsh(along_x),
                               np.linalg.eigvalsh(along_y))


def test_spatial_model_shrinks_ray_moment_to_per_camera_ray_prior():
    model = CommissionedVisibilitySensorModel.__new__(CommissionedVisibilitySensorModel)
    model.runtime_covariance_model = "R2_spatial_residual"
    model.ray_r_per_camera = {"camera_A": np.diag([2.0, 1.0])}
    model.ray_r_samples = {"camera_A": {
        "xy": np.asarray([[1.0, 0.0]]),
        "moment": np.asarray([np.diag([6.0, 3.0])]),
    }}
    model.current_r_neighbors = 1
    model.current_r_bandwidth_m = 1.0
    model.current_r_shrinkage = 1.0
    model.ray_r_prior_covariance = None

    result = model._ray_r_covariance("camera_A", np.asarray([1.0, 0.0]))
    np.testing.assert_allclose(result, np.diag([4.0, 2.0]))


def test_spatial_bayesian_model_uses_declared_covariance_prior():
    model = CommissionedVisibilitySensorModel.__new__(CommissionedVisibilitySensorModel)
    model.runtime_covariance_model = "R2_spatial_residual"
    model.ray_r_per_camera = {"camera_A": np.diag([2.0, 1.0])}
    model.ray_r_samples = {"camera_A": {
        "xy": np.asarray([[1.0, 0.0]]),
        "moment": np.asarray([np.diag([6.0, 3.0])]),
    }}
    model.current_r_neighbors = 1
    model.current_r_bandwidth_m = 1.0
    model.current_r_shrinkage = 1.0
    model.ray_r_prior_covariance = np.diag([100.0, 100.0])

    result = model._ray_r_covariance("camera_A", np.asarray([1.0, 0.0]))
    np.testing.assert_allclose(result, np.diag([53.0, 51.5]))
