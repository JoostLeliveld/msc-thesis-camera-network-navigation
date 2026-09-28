"""Q set from the simulated encoder noise: consistency, both implementations, coverage.

The coverage test reproduces the encoder of sim/encoder_noise_node.py (AR(1) slip,
additive noise, systematic wheel errors) on known true motion, and checks that the Q
predicted by the planner's own closed form covers the resulting odometry error.
"""
import numpy as np
import pytest

from planning.core import encoder_noise_model as enm
from planning.core.dynamics import unicycle_jacobian, unicycle_process_noise, unicycle_step


def test_constants_match_the_simulator():
    lc = pytest.importorskip('experiments.core.visibility_launch_common')
    assert enm.LINEAR_SLIP_STD == lc._ENCODER_NOISE_LINEAR_SLIP_STD
    assert enm.ANGULAR_SLIP_STD == lc._ENCODER_NOISE_ANGULAR_SLIP_STD
    assert enm.LINEAR_ADDITIVE_STD == lc._ENCODER_NOISE_LINEAR_ADDITIVE_STD
    assert enm.ANGULAR_ADDITIVE_STD == lc._ENCODER_NOISE_ANGULAR_ADDITIVE_STD
    assert enm.SLIP_ALPHA == lc._ENCODER_NOISE_CORRELATION_ALPHA
    assert enm.WHEEL_DIAMETER_RATIO_ERROR == lc._ENCODER_WHEEL_DIAMETER_RATIO_ERROR
    assert enm.WHEELBASE_RATIO == pytest.approx(lc._ENCODER_WHEELBASE_RATIO)
    assert enm.WHEEL_SEPARATION_M == lc._ROBOT_WHEEL_SEPARATION_M


def test_numpy_and_casadi_q_agree():
    ca = pytest.importorskip('casadi')
    from planning.core.casadi_efe import unicycle_process_noise_ca
    for theta, v, w in [(0.3, 0.9, 0.0), (1.2, 0.5, 0.6), (-2.0, 0.0, 1.0)]:
        q_np = unicycle_process_noise(0.02, 0.08, 0.25, theta=theta, v=v, w=w, psd=enm.ENCODER_PSD)
        q_ca = np.array(ca.DM(unicycle_process_noise_ca(0.02, 0.08, 0.25, theta, v, w, enm.ENCODER_PSD)))
        np.testing.assert_allclose(q_np, q_ca, rtol=1e-12, atol=1e-15)


def test_heading_noise_follows_turning_not_time():
    q = lambda v, w: unicycle_process_noise(0, 0, 1.0, theta=0.0, v=v, w=w, psd=enm.ENCODER_PSD)[2, 2]
    assert q(0.0, 0.0) < q(1.0, 0.0) < q(0.0, 1.0)
    assert q(0.0, 1.0) > 10 * q(0.0, 0.0)


def _encoder_odometry(v, w, dt, n_seeds, rng):
    """Encoder velocities for known true (v, w) sequences: the simulator's noise model."""
    from sim.encoder_noise_node import systematic_wheel_velocities
    v_sys, w_sys = systematic_wheel_velocities(v, w, enm.WHEEL_DIAMETER_RATIO_ERROR,
                                               enm.WHEELBASE_RATIO, enm.WHEEL_SEPARATION_M)
    n = len(v)
    a = enm.SLIP_ALPHA
    k = np.sqrt(1 - a * a)
    sv = np.zeros(n_seeds); sw = np.zeros(n_seeds)
    ve = np.empty((n_seeds, n)); we = np.empty((n_seeds, n))
    for i in range(n):
        sv = a * sv + k * enm.LINEAR_SLIP_STD * rng.standard_normal(n_seeds)
        sw = a * sw + k * enm.ANGULAR_SLIP_STD * rng.standard_normal(n_seeds)
        ve[:, i] = v_sys[i] * (1 + sv) + enm.LINEAR_ADDITIVE_STD * rng.standard_normal(n_seeds)
        we[:, i] = w_sys[i] * (1 + sw) + enm.ANGULAR_ADDITIVE_STD * rng.standard_normal(n_seeds)
    return ve, we


@pytest.mark.parametrize('profile', ['straight', 'turns'])
def test_set_q_covers_the_encoder_drift(profile):
    dt = enm.ENCODER_PERIOD_S
    n = int(10.0 / dt)
    t = np.arange(n) * dt
    v = np.full(n, 0.9)
    w = np.zeros(n) if profile == 'straight' else 0.6 * np.sign(np.sin(2 * np.pi * t / 5.0))
    rng = np.random.default_rng(7)
    ve, we = _encoder_odometry(v, w, dt, 600, rng)
    # truth and encoder dead reckoning
    def integrate(vv, ww):
        th = np.cumsum(ww, axis=-1) * dt
        th_prev = th - ww * dt
        return (np.cumsum(vv * np.cos(th_prev), axis=-1) * dt,
                np.cumsum(vv * np.sin(th_prev), axis=-1) * dt, th)
    gx, gy, gth = integrate(v, w)
    ex, ey, eth = integrate(ve, we)
    m = np.zeros(3); P = np.zeros((3, 3))
    checks = {int(1 / dt) - 1: None, int(3 / dt) - 1: None, n - 1: None}
    for i in range(n):
        u = np.array([v[i], w[i]])
        F = unicycle_jacobian(m, u, dt)
        P = F @ P @ F.T + unicycle_process_noise(0, 0, dt, theta=float(m[2]), v=v[i], w=w[i],
                                                 psd=enm.ENCODER_PSD)
        m = np.asarray(unicycle_step(m, u, dt), float)
        if i in checks:
            d = np.stack([ex[:, i] - gx[i], ey[:, i] - gy[i]], -1)
            nees = np.einsum('ni,ij,nj->n', d, np.linalg.inv(P[:2, :2]), d)
            pos = np.mean(nees <= 5.991)
            head = np.mean(np.abs(eth[:, i] - gth[i]) <= 1.96 * np.sqrt(P[2, 2]))
            checks[i] = (pos, head)
    for i, (pos, head) in checks.items():
        assert pos >= 0.93, (profile, (i + 1) * dt, pos)
        assert head >= 0.93, (profile, (i + 1) * dt, head)
    print(profile, {round((i + 1) * dt): tuple(round(x, 3) for x in c) for i, c in checks.items()})


def test_campaign_templates_run_the_modelled_encoder_noise():
    """The campaign must run the noise Q is set from (the runner has its own fallbacks)."""
    import pathlib
    import yaml
    root = pathlib.Path(__file__).resolve().parents[2] / 'pipeline'
    for name in ('execution_template.yaml', 'route_planning_template.yaml'):
        cfg = yaml.safe_load((root / name).read_text())
        assert cfg['process_noise_model'] == 'encoder'
        assert cfg['encoder_noise_linear_slip_mean'] == 0.0
        assert cfg['encoder_noise_linear_slip_std'] == enm.LINEAR_SLIP_STD
        assert cfg['encoder_noise_angular_slip_mean'] == 0.0
        assert cfg['encoder_noise_angular_slip_std'] == enm.ANGULAR_SLIP_STD
        assert cfg['encoder_noise_linear_additive_std'] == enm.LINEAR_ADDITIVE_STD
        assert cfg['encoder_noise_angular_additive_std'] == enm.ANGULAR_ADDITIVE_STD
        assert cfg['encoder_noise_correlation_alpha'] == enm.SLIP_ALPHA
        assert cfg['encoder_wheel_diameter_ratio_error'] == enm.WHEEL_DIAMETER_RATIO_ERROR
        assert cfg['encoder_wheelbase_ratio'] == pytest.approx(enm.WHEELBASE_RATIO, abs=1e-6)
