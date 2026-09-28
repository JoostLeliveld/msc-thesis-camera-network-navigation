"""
EFE/MPC math utilities (numpy-only).

Ported from ICRA2026-WUnEmbodied-main/util.py and free_energy_agents.py,
with autodiff-specific pieces replaced by finite-difference and NumPy equivalents.
"""

import numpy as np


def wrap_angle(theta):
    """Wrap angle to [-pi, pi]."""
    while theta > np.pi:
        theta -= 2.0 * np.pi
    while theta < -np.pi:
        theta += 2.0 * np.pi
    return theta


def _ensure_psd(S, eps=1e-9):
    """Project matrix to PSD by clipping eigenvalues."""
    vals, vecs = np.linalg.eigh(S)
    vals = np.maximum(vals, eps)
    return (vecs @ np.diag(vals) @ vecs.T + (vecs @ np.diag(vals) @ vecs.T).T) / 2.0


def finite_diff_jacobian(g, x, eps=1e-6):
    """Finite-difference Jacobian of g at x. g: R^n -> R^m."""
    x = np.asarray(x, dtype=float)
    y0 = np.asarray(g(x), dtype=float)
    m = y0.size
    n = x.size
    J = np.zeros((m, n), dtype=float)
    for i in range(n):
        dx = np.zeros_like(x)
        dx[i] = eps
        y1 = np.asarray(g(x + dx), dtype=float)
        y2 = np.asarray(g(x - dx), dtype=float)
        J[:, i] = (y1 - y2) / (2.0 * eps)
    return J


def finite_diff_hessian_scalar(g, x, eps=1e-4):
    """Finite-difference Hessian for scalar g: R^n -> R."""
    x = np.asarray(x, dtype=float)
    n = x.size
    H = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            dx_i = np.zeros_like(x); dx_i[i] = eps
            dx_j = np.zeros_like(x); dx_j[j] = eps
            f_pp = g(x + dx_i + dx_j)
            f_pm = g(x + dx_i - dx_j)
            f_mp = g(x - dx_i + dx_j)
            f_mm = g(x - dx_i - dx_j)
            H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps * eps)
    return H


def sigma_points(m, P, alpha=1e-3, kappa=0.0):
    """Sigma points for Unscented Transform. Returns array (2N+1, N)."""
    m = np.asarray(m, dtype=float)
    P = np.asarray(P, dtype=float)
    n = m.size
    lam = alpha ** 2 * (n + kappa) - n
    scale = n + lam
    # Cholesky with jitter
    jitter = 1e-9
    for _ in range(5):
        try:
            S = np.linalg.cholesky(scale * P + jitter * np.eye(n))
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    else:
        S = np.linalg.cholesky(scale * _ensure_psd(P) + jitter * np.eye(n))

    sigmas = np.zeros((2 * n + 1, n), dtype=float)
    sigmas[0] = m
    for i in range(n):
        sigmas[1 + i] = m + S[:, i]
        sigmas[1 + n + i] = m - S[:, i]
    return sigmas


def ut_weights(n, alpha=1e-3, beta=2.0, kappa=0.0):
    """Weights for Unscented Transform."""
    lam = alpha ** 2 * (n + kappa) - n
    w0 = lam / (n + lam)
    Wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
    Wc = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
    Wm[0] = w0
    Wc[0] = w0 + (1.0 - alpha ** 2 + beta)
    return Wm, Wc


def UT(m, P, g, addmatrix=None, forceHermitian=False, alpha=1e-3, beta=2.0, kappa=0.0):
    """
    Unscented Transform for g(m,P).
    Returns (mu, Sigma, Gamma).
    """
    m = np.asarray(m, dtype=float)
    P = np.asarray(P, dtype=float)
    n = m.size
    sigmas = sigma_points(m, P, alpha=alpha, kappa=kappa)
    Wm, Wc = ut_weights(n, alpha=alpha, beta=beta, kappa=kappa)

    ys = np.array([np.asarray(g(s), dtype=float) for s in sigmas])
    mu = np.sum(Wm[:, None] * ys, axis=0)

    d_y = ys.shape[1]
    Sigma = np.zeros((d_y, d_y), dtype=float)
    Gamma = np.zeros((n, d_y), dtype=float)
    for i in range(2 * n + 1):
        dy = (ys[i] - mu).reshape(-1, 1)
        dx = (sigmas[i] - m).reshape(-1, 1)
        Sigma += Wc[i] * (dy @ dy.T)
        Gamma += Wc[i] * (dx @ dy.T)

    if addmatrix is not None:
        Sigma += addmatrix
    if forceHermitian:
        Sigma = (Sigma + Sigma.T) / 2.0
    return mu, Sigma, Gamma


def ET1(m, S, g, addmatrix=None, forceHermitian=False, eps=1e-6):
    """Extended Transform order 1 using finite-diff Jacobian."""
    m = np.asarray(m, dtype=float)
    S = np.asarray(S, dtype=float)
    Jm = finite_diff_jacobian(g, m, eps=eps)
    mE = np.asarray(g(m), dtype=float)
    SE = Jm @ S @ Jm.T
    CE = S @ Jm.T
    if addmatrix is not None:
        SE += addmatrix
    if forceHermitian:
        SE = (SE + SE.T) / 2.0
    return mE, SE, CE


def ET2(m, S, g, addmatrix=None, forceHermitian=False, eps=1e-4):
    """
    Extended Transform order 2 using finite-diff Hessians.
    This is approximate and intended for small dimensions.
    """
    m = np.asarray(m, dtype=float)
    S = np.asarray(S, dtype=float)
    y = np.asarray(g(m), dtype=float)
    n = m.size
    d = y.size
    Jm = finite_diff_jacobian(g, m, eps=eps)

    # Hessians per output dimension
    H = np.zeros((d, n, n), dtype=float)
    for i in range(d):
        def g_i(x):
            return np.asarray(g(x), dtype=float)[i]
        H[i] = finite_diff_hessian_scalar(g_i, m, eps=eps)

    aux1 = np.zeros(d, dtype=float)
    for i in range(d):
        aux1[i] = np.trace(H[i] @ S)

    aux2 = np.zeros((d, d), dtype=float)
    for i in range(d):
        for j in range(d):
            aux2[i, j] = np.trace(H[i] @ S @ H[j] @ S)

    mE = y + 0.5 * aux1
    SE = Jm @ S @ Jm.T + 0.5 * aux2
    CE = S @ Jm.T

    if addmatrix is not None:
        SE += addmatrix
    if forceHermitian:
        SE = (SE + SE.T) / 2.0
    return mE, SE, CE


def ambiguity(Sigma, Gamma, S):
    """Conditional entropy term for EFE."""
    S_inv = np.linalg.inv(S)
    Sigma_cond = Sigma - Gamma.T @ S_inv @ Gamma
    Sigma_cond = _ensure_psd(Sigma_cond)
    sign, logdet = np.linalg.slogdet(Sigma_cond)
    if sign <= 0:
        logdet = np.log(np.maximum(np.linalg.det(Sigma_cond), 1e-12))
    d = Sigma_cond.shape[0]
    return 0.5 * (d * np.log(2 * np.pi * np.e) + logdet)


def risk_components(mu, Sigma, goal, one_sided=True):
    """KL(N(mu,Sigma) || N(m*,S*)) split into interpretable pieces.

    ``one_sided`` clips the covariance argument up to the goal prior before
    scoring, so a prediction sharper than the task tolerance is treated as if it
    sat exactly at tolerance. Without it the KL pays the planner to let the
    belief drift whenever the prediction is sharper than the goal prior. See
    ``risk_ca`` for the full argument; the two must agree numerically.
    """
    m_star, S_star = goal
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    m_star = np.asarray(m_star, dtype=float)
    S_star = np.asarray(S_star, dtype=float)

    d = mu.size
    if one_sided:
        floor = float(np.exp(np.linalg.slogdet(S_star)[1] / d))
        deficit = floor - float(np.trace(Sigma)) / d
        if deficit > 0.0:
            Sigma = Sigma + deficit * np.eye(Sigma.shape[0])
    diff = (m_star - mu).reshape(-1, 1)

    sign_s, logdet_s = np.linalg.slogdet(Sigma)
    sign_t, logdet_t = np.linalg.slogdet(S_star)
    if sign_s <= 0:
        logdet_s = np.log(np.maximum(np.linalg.det(Sigma), 1e-12))
    if sign_t <= 0:
        logdet_t = np.log(np.maximum(np.linalg.det(S_star), 1e-12))

    term_trace = np.trace(np.linalg.solve(S_star, Sigma))
    term_quad = (diff.T @ np.linalg.solve(S_star, diff)).item()
    mean = 0.5 * term_quad
    cov_trace = 0.5 * float(term_trace)
    cov_logdet = 0.5 * float(logdet_t - logdet_s)
    const = -0.5 * float(d)
    total = mean + cov_trace + cov_logdet + const
    return {
        'total': float(total),
        'mean': float(mean),
        'cov_trace': float(cov_trace),
        'cov_logdet': float(cov_logdet),
        'const': float(const),
    }


def risk(mu, Sigma, goal):
    """KL(N(mu,Sigma) || N(m*,S*)) as risk term."""
    return risk_components(mu, Sigma, goal)['total']


def risk_obs(mu, Sigma, goal):
    """Alias for risk in observation space."""
    return risk(mu, Sigma, goal)
