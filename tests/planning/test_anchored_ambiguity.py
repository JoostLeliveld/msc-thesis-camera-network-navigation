"""The ambiguity term is anchored to an ideally observed pose.

The ET1 ambiguity is an ABSOLUTE differential entropy, so it carries a
route-independent additive constant whose sign follows the units R_eff is
written in. On a fixed horizon that constant cancels. Under the smooth arrival
gate duration is free, so it becomes ``constant * T`` -- a duration term that can
pay a route for lasting longer. These tests pin the properties that removes.
"""
import hashlib
import inspect
import json

import numpy as np
import pytest

from planning.core.camera_network import (
    AMBIGUITY_FLOOR_POSITION_SD_M,
    AMBIGUITY_INFORMATION_REGULARIZER_M2_INV,
    CameraNetworkModel,
)


def write_network(path, *, availability, r_scale=1.0, spatial=False):
    xs = ys = np.array([-3., 0., 3.])
    X, Y = np.meshgrid(xs, ys)
    score = np.stack([np.full((3, 3), .5), np.full((3, 3), .5)])
    R = r_scale * np.array([[[.01, .006], [.006, .16]], [[.12, -.008], [-.008, .015]]])
    if np.isscalar(availability):
        avail = np.full((2, 3, 3), float(availability))
    else:
        avail = np.asarray(availability, dtype=float)
    meta = dict(
        schema='camera_network.iwai.v1', reference='robot_ground_reference_xy',
        frame='map_bev', covariance_units='m2',
        score_target='detector_score_with_miss_zero',
        availability_target='valid_detection_finite_ground_projection',
        evidence='synthetic_test_fixture',
        source_hashes={'synthetic_fixture': hashlib.sha256(b'anchored-v1').hexdigest()})
    np.savez(path, xs=xs, ys=ys, camera_ids=['camera_A', 'camera_B'], score=score,
             availability=avail, R_cond_m2=R, R_miss_proxy_m2=R + 25 * np.eye(2),
             metadata_json=json.dumps(meta))
    return CameraNetworkModel(path)


FLOOR = AMBIGUITY_FLOOR_POSITION_SD_M ** 2 * np.eye(2)


def anchored(net, state, P):
    R_eff = net.effective_observation_covariance(state, P, 1.0)
    ratio = (np.linalg.slogdet(R_eff)[1] - np.linalg.slogdet(FLOOR)[1])
    return 0.5 * max(ratio, 0.0)


def test_constant_parameter_arm_is_constant_not_zero(tmp_path):
    """A constant-q, constant-R arm must give a CONSTANT per-step ambiguity.

    NOT zero: q=1 means every camera reports, not that the pose is perfectly
    localized. The resulting finite R_eff is real residual uncertainty the
    objective should still charge for. What makes such an arm select the
    shortest safe route is that the SAME amount is added to every candidate at
    every step, so only length and safety separate the candidates.
    """
    net = write_network(tmp_path / 'f.npz', availability=1.0)
    P = np.diag([.01, .01, .01])
    values = [anchored(net, np.array([x, y, 0.]), P)
              for x in (-2., 0., 2.) for y in (-2., 0., 2.)]
    assert np.ptp(values) == pytest.approx(0.0, abs=1e-12)
    assert min(values) > 0.0, 'a constant arm still carries residual uncertainty'


def test_the_floor_never_clips_a_real_pose(tmp_path):
    """The floor is a THRESHOLD below the operating range, not an estimate.

    Any floor below the tightest reachable R_eff gives identical rankings; a
    floor inside the range clips real poses to zero and deletes the signal.
    """
    net = write_network(tmp_path / 'f.npz', availability=0.5)
    P = np.diag([.01, .01, .01])
    rng = np.random.default_rng(3)
    for _ in range(50):
        state = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), 0.])
        R_eff = net.effective_observation_covariance(state, P, 1.0)
        assert np.linalg.slogdet(R_eff)[1] > np.linalg.slogdet(FLOOR)[1], (
            'floor is inside the operating range; it would clip a real pose')


def test_every_arm_shares_one_floor(tmp_path):
    """Per-arm floors would shift each arm by a different constant.

    Measured on the seven commissioned arms, per-arm best-R floors span 3.5
    nats. Anchoring each arm to itself would make the arms incomparable, which
    is precisely what a q/R comparison must not do.
    """
    tight = write_network(tmp_path / 'tight.npz', availability=0.5, r_scale=0.01)
    loose = write_network(tmp_path / 'loose.npz', availability=0.5, r_scale=100.0)
    state, P = np.array([0., 0., 0.]), np.diag([.01, .01, .01])
    # Same floor for both, so the arm with better R gets the lower ambiguity.
    assert anchored(tight, state, P) < anchored(loose, state, P)


def test_anchored_ambiguity_is_never_negative(tmp_path):
    """No route may be REWARDED for lasting longer."""
    net = write_network(tmp_path / 'f.npz', availability=1.0)
    P = np.diag([.01, .01, .01])
    rng = np.random.default_rng(0)
    for _ in range(50):
        state = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(0, 6.28)])
        assert anchored(net, state, P) >= 0.0


def test_poorly_observed_pose_costs_more_than_a_well_observed_one(tmp_path):
    """The term still prefers well-observed states -- the signal is preserved."""
    blind = write_network(tmp_path / 'blind.npz', availability=0.02)
    covered = write_network(tmp_path / 'covered.npz', availability=0.98)
    state, P = np.array([0., 0., 0.]), np.diag([.01, .01, .01])
    assert anchored(blind, state, P) > anchored(covered, state, P)


def test_zero_camera_information_has_fixed_ambiguity_covariance(tmp_path):
    """The inverse remains finite without borrowing a scale from process noise."""
    net = write_network(tmp_path / 'blind.npz', availability=0.0)
    P = np.diag([.01, .01, .01])
    actual = net.effective_observation_covariance(np.zeros(3), P)
    expected = np.eye(2) / AMBIGUITY_INFORMATION_REGULARIZER_M2_INV
    np.testing.assert_allclose(actual, expected)


def test_effective_covariance_has_no_process_noise_input():
    """Belief averaging may use P, but the inverse floor must not use Q or dt."""
    numpy_parameters = inspect.signature(
        CameraNetworkModel.effective_observation_covariance).parameters
    casadi_parameters = inspect.signature(
        CameraNetworkModel.make_effective_covariance_casadi).parameters
    assert 'no_report_var' not in numpy_parameters
    assert 'no_report_var' not in casadi_parameters
    assert 'process_noise' not in numpy_parameters
    assert 'process_noise' not in casadi_parameters
