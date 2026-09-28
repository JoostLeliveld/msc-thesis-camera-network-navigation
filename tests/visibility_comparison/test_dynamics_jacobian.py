from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
for rel in ('src/planning',):
    path = str((REPO_ROOT / rel).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)

from planning.core.casadi_efe import unicycle_jacobian_ca, unicycle_step_ca, unicycle_process_noise_ca
from planning.core.dynamics import unicycle_jacobian, unicycle_step, unicycle_process_noise


def _finite_difference_jacobian(state: np.ndarray, control: np.ndarray, dt: float, eps: float = 1e-6) -> np.ndarray:
    base = unicycle_step(state, control, dt)
    J = np.zeros((3, 3), dtype=float)
    for idx in range(3):
        perturbed = np.array(state, dtype=float)
        perturbed[idx] += eps
        J[:, idx] = (unicycle_step(perturbed, control, dt) - base) / eps
    return J


def test_unicycle_jacobian_matches_pre_step_finite_difference() -> None:
    state = np.array([1.2, -0.4, 0.7], dtype=float)
    control = np.array([0.35, -0.2], dtype=float)
    dt = 0.15

    analytic = unicycle_jacobian(state, control, dt)
    numeric = _finite_difference_jacobian(state, control, dt)
    np.testing.assert_allclose(analytic, numeric, atol=5e-6, rtol=5e-6)


def test_casadi_unicycle_jacobian_uses_pre_step_heading() -> None:
    ca = pytest.importorskip('casadi')
    state = ca.DM([0.8, 0.1, 0.45])
    control = ca.DM([0.4, 0.9])
    dt = 0.25

    jacobian = np.asarray(unicycle_jacobian_ca(state, control, dt), dtype=float)
    expected = np.eye(3, dtype=float)
    expected[0, 2] = -float(control[0]) * dt * math.sin(float(state[2]))
    expected[1, 2] = float(control[0]) * dt * math.cos(float(state[2]))
    np.testing.assert_allclose(jacobian, expected, atol=1e-9, rtol=1e-9)

    stepped = np.asarray(unicycle_step_ca(state, control, dt), dtype=float).reshape(-1)
    assert jacobian[0, 2] != pytest.approx(-float(control[0]) * dt * math.sin(float(stepped[2])))


def test_unicycle_process_noise_analytical() -> None:
    ca = pytest.importorskip('casadi')
    process_noise_xy = 0.01
    process_noise_theta = 0.02
    dt = 0.25
    theta = 0.45
    v = 0.4

    # 1. Test fallback when theta and v are None (should be diagonal)
    Q_fallback = unicycle_process_noise(process_noise_xy, process_noise_theta, dt)
    np.testing.assert_allclose(
        Q_fallback,
        np.diag([process_noise_xy**2, process_noise_xy**2, process_noise_theta**2]),
        atol=1e-9
    )

    # 2. Test analytical covariance
    Q_np = unicycle_process_noise(process_noise_xy, process_noise_theta, dt, theta=theta, v=v)
    
    # Check that it matches the CasADi implementation. Use ca.DM (numeric) so the
    # result is a concrete matrix convertible to float; ca.MX would stay symbolic and
    # cannot be passed to np.asarray.
    Q_ca = np.asarray(
        unicycle_process_noise_ca(
            ca.DM(process_noise_xy),
            ca.DM(process_noise_theta),
            ca.DM(dt),
            ca.DM(theta),
            ca.DM(v)
        ),
        dtype=float
    )
    np.testing.assert_allclose(Q_np, Q_ca, atol=1e-9)

    # 2b. Verify the SYMBOLIC (MX) path the EFE loop actually uses: build a ca.Function
    # from symbolic inputs, evaluate it, and confirm it matches the NumPy reference.
    th_s = ca.MX.sym('th')
    v_s = ca.MX.sym('v')
    Q_sym = unicycle_process_noise_ca(
        ca.MX(process_noise_xy), ca.MX(process_noise_theta), ca.MX(dt), th_s, v_s
    )
    Q_fn = ca.Function('Q_fn', [th_s, v_s], [Q_sym])
    Q_ca_sym = np.asarray(Q_fn(theta, v), dtype=float)
    np.testing.assert_allclose(Q_np, Q_ca_sym, atol=1e-9)

    # 3. Check specific analytical values
    c = math.cos(theta)
    s = math.sin(theta)
    sig_v2 = process_noise_xy ** 2
    sig_w2 = process_noise_theta ** 2

    expected_00 = sig_v2 * (c ** 2) * dt + (1.0 / 3.0) * (v ** 2) * (s ** 2) * sig_w2 * (dt ** 3)
    expected_01 = sig_v2 * c * s * dt - (1.0 / 3.0) * (v ** 2) * c * s * sig_w2 * (dt ** 3)
    expected_02 = -0.5 * v * s * sig_w2 * (dt ** 2)
    
    expected_11 = sig_v2 * (s ** 2) * dt + (1.0 / 3.0) * (v ** 2) * (c ** 2) * sig_w2 * (dt ** 3)
    expected_12 = 0.5 * v * c * sig_w2 * (dt ** 2)
    expected_22 = sig_w2 * dt

    expected_Q = np.array([
        [expected_00, expected_01, expected_02],
        [expected_01, expected_11, expected_12],
        [expected_02, expected_12, expected_22]
    ])
    np.testing.assert_allclose(Q_np, expected_Q, atol=1e-9)

    # 4. Check zero velocity limit (v = 0)
    Q_zero_v = unicycle_process_noise(process_noise_xy, process_noise_theta, dt, theta=theta, v=0.0)
    expected_zero_v = np.array([
        [sig_v2 * (c**2) * dt, sig_v2 * c * s * dt, 0.0],
        [sig_v2 * c * s * dt, sig_v2 * (s**2) * dt, 0.0],
        [0.0, 0.0, sig_w2 * dt]
    ])
    np.testing.assert_allclose(Q_zero_v, expected_zero_v, atol=1e-9)
