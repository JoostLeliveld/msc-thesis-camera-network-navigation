"""Dynamics models used by planners."""

import math
import numpy as np

from planning.core.efe_utils import wrap_angle


def unicycle_step(state, control, dt):
    """Unicycle step in SE(2). state=[x,y,theta], control=[v,w]."""
    x, y, theta = state
    v, w = control
    x = x + v * dt * math.cos(theta)
    y = y + v * dt * math.sin(theta)
    theta = wrap_angle(theta + w * dt)
    return np.array([x, y, theta], dtype=float)


def unicycle_jacobian(state, control, dt):
    """Jacobian of unicycle dynamics wrt state."""
    _, _, theta = state
    v, _ = control
    F = np.eye(3)
    F[0, 2] = -v * dt * math.sin(theta)
    F[1, 2] = v * dt * math.cos(theta)
    return F


#: Frozen simulator process-model constants. They are configuration inputs, not fitted
#: thesis quantities.
COHERENT_SPEED_SCALE = 0.02          # sim/encoder_noise_node.py `linear_slip_mean`
COHERENT_HEADING_RAD = 0.0227        # 1.3 deg, the flat offset implied by cross-track drift


def coherent_drift_block(distance_m, theta, *, speed_scale=COHERENT_SPEED_SCALE,
                         heading_rad=COHERENT_HEADING_RAD):
    """Rank-2 covariance of encoder drift that does NOT average away over a window.

    `unicycle_process_noise` models only instantaneous white noise on (v, w). The encoder
    stream it must describe (`sim/encoder_noise_node.py`) also carries a systematic speed
    scale error (`linear_slip_mean`) and AR(1)-correlated slip (`correlation_alpha = 0.80`).
    Those are coherent: held roughly constant over a window rather than resampled each step.

    Both coherent terms scale with DISTANCE TRAVELLED, in the body frame:

        along-track   speed_scale * distance      (systematic speed error)
        cross-track   heading_rad * distance      (a held heading offset)

    These values remain frozen with the simulator configuration. The thesis does not
    estimate or select them.

    **This block is the covariance of the TOTAL drift after `distance_m`, not a per-step
    increment.** A held offset accumulates coherently: displacement grows linearly with
    distance, so variance grows as distance^2. Summing this block once per step instead
    would add variances (distance^2 -> n * (distance/n)^2), reintroducing exactly the
    sqrt(n) averaging the term exists to prevent. Callers that integrate step by step
    must therefore pass the DIFFERENCE between the block at the new
    cumulative distance and the block at the old one; `coherent_drift_increment` does that.

    Returns a 3x3 block (heading row/column zero: these terms are positional).
    """
    c, s = math.cos(float(theta)), math.sin(float(theta))
    fwd = (c, s)
    lat = (-s, c)
    along_var = (float(speed_scale) * float(distance_m)) ** 2
    cross_var = (float(heading_rad) * float(distance_m)) ** 2
    block = np.zeros((3, 3))
    for i in range(2):
        for j in range(2):
            block[i, j] = along_var * fwd[i] * fwd[j] + cross_var * lat[i] * lat[j]
    return block


def coherent_drift_increment(distance_before_m, distance_after_m, theta, **kwargs):
    """The coherent covariance a filter should ADD for one step of a longer drive.

    Because coherent drift grows as distance^2, the increment from `d0` to `d1` is
    `block(d1) - block(d0)`, not `block(d1 - d0)`. Over a whole window these telescope to
    exactly `block(total)`.
    """
    after = coherent_drift_block(distance_after_m, theta, **kwargs)
    before = coherent_drift_block(distance_before_m, theta, **kwargs)
    return after - before


def unicycle_process_noise(process_noise_xy, process_noise_theta, dt, theta=None, v=None,
                           base_dt=None, coherent_drift=False, distance_travelled_m=0.0,
                           w=None, psd=None):
    """Process noise matrix for unicycle.

    If theta and v are provided, uses the exact integrated analytical process noise covariance.
    Otherwise, falls back to the simplified diagonal covariance scaled by dt/base_dt.

    ``coherent_drift`` adds `coherent_drift_block` for the distance this step travels. It is
    OFF by default: switching it on changes the belief on every drive, so existing campaigns
    stay comparable until one opts in explicitly.
    """
    if theta is not None and v is not None:
        c = math.cos(float(theta))
        s = math.sin(float(theta))
        v = float(v)
        dt = float(dt)
        if psd is not None:   # input-dependent PSDs, planning.core.encoder_noise_model
            from planning.core.encoder_noise_model import encoder_psd
            sig_v2, sig_w2 = encoder_psd(v, 0.0 if w is None else float(w), psd)
        else:
            sig_v2 = float(process_noise_xy) ** 2
            sig_w2 = float(process_noise_theta) ** 2

        q00 = sig_v2 * (c ** 2) * dt + (1.0 / 3.0) * (v ** 2) * (s ** 2) * sig_w2 * (dt ** 3)
        q01 = sig_v2 * c * s * dt - (1.0 / 3.0) * (v ** 2) * c * s * sig_w2 * (dt ** 3)
        q02 = -0.5 * v * s * sig_w2 * (dt ** 2)

        q11 = sig_v2 * (s ** 2) * dt + (1.0 / 3.0) * (v ** 2) * (c ** 2) * sig_w2 * (dt ** 3)
        q12 = 0.5 * v * c * sig_w2 * (dt ** 2)

        q22 = sig_w2 * dt

        Q = np.array([
            [q00, q01, q02],
            [q01, q11, q12],
            [q02, q12, q22]
        ], dtype=float)
        if coherent_drift:
            travelled = abs(v) * dt
            Q = Q + coherent_drift_increment(
                float(distance_travelled_m),
                float(distance_travelled_m) + travelled,
                float(theta),
            )
        return Q

    Q = np.diag([
        process_noise_xy ** 2,
        process_noise_xy ** 2,
        process_noise_theta ** 2,
    ])
    if base_dt is not None and base_dt > 1e-9:
        Q = Q * max(dt / base_dt, 0.0)
    if coherent_drift and theta is not None and v is not None:
        travelled = abs(float(v)) * float(dt)
        Q = Q + coherent_drift_increment(
            float(distance_travelled_m),
            float(distance_travelled_m) + travelled,
            float(theta),
        )
    return Q


def linear_dynamics_matrices(dt):
    """Linear 4D dynamics matrices from the reference free_energy_agents.py."""
    A = np.array([
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    B = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [dt, 0.0],
        [0.0, dt],
    ])
    return A, B


def linear_process_noise(dt, rho):
    """Process noise covariance for the linear 4D dynamics (reference form)."""
    rho = np.asarray(rho, dtype=float).reshape(-1)
    if rho.size == 1:
        rho = np.array([rho[0], rho[0]], dtype=float)
    return np.array([
        [dt**3 / 3.0 * rho[0], 0.0, dt**2 / 2.0 * rho[0], 0.0],
        [0.0, dt**3 / 3.0 * rho[1], 0.0, dt**2 / 2.0 * rho[1]],
        [dt**2 / 2.0 * rho[0], 0.0, dt * rho[0], 0.0],
        [0.0, dt**2 / 2.0 * rho[1], 0.0, dt * rho[1]],
    ])
