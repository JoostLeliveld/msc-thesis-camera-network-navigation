"""CasADi utilities for symbolic visibility-aware EFE planning."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

try:
    import casadi as ca
except Exception:  # pragma: no cover - optional dependency
    ca = None


def casadi_available() -> bool:
    return ca is not None


# Innovation-covariance floor, in pixel^2, used ONLY on the hit/miss mixture path
# to keep the single ca.solve well posed. (0.05 px)^2 — a twentieth of a pixel,
# two orders of magnitude below the ~2.5 px conditional residual std this detector
# actually achieves, so it can never bind on real data. Deliberately NOT 1e-9:
# against the 1e3–1e5 px^2 magnitudes of J S J^T, a 1e-9 floor sits below float64
# round-off and buys nothing while pretending to be a safeguard.
INNOVATION_COV_FLOOR_PX2 = 0.05 ** 2


@dataclass
class CasadiEfeParams:
    # NOTE: there is no static `Q` field. Process noise is rebuilt per step inside the
    # EFE loop via unicycle_process_noise_ca(process_noise_xy, process_noise_theta, dt,
    # theta, v) — the exact integrated Q_d(theta, v, dt) — so a frozen Q would be unused.
    R_visible: object
    R_miss: object
    control_weight: float
    risk_scale: float
    ambiguity_scale: float
    discount_gamma: float
    process_noise_xy: float
    process_noise_theta: float
    visibility_sigma_kappa: float
    goal_prior_u_std_start: float
    goal_prior_v_std_start: float
    goal_prior_u_std_final: float
    goal_prior_v_std_final: float
    goal_tightening_power: float
    goal_progress_n_steps: int
    use_belief_nogo_cost: bool
    time_horizon: int
    dt: float
    Du: int

    # --- hit/miss expected-belief mixture -----------------------------------
    # DEFAULT OFF. With use_hit_miss_mixture=False every field below is ignored
    # and the planner runs the precision-blend path bit-for-bit unchanged — that
    # path is FROZEN METHOD for the published single-camera paper and backs the
    # locked honest_campaign_v1 (see tests/planning/test_efe_hit_miss_mixture.py).
    #
    # With the flag on, detection availability stops being laundered into the
    # measurement covariance and becomes what it actually is: a Bernoulli
    # variable. See hit_miss_posterior_ca for the model.
    use_hit_miss_mixture: bool = False
    # R_cond: conditional measurement covariance GIVEN a usable detection.
    # None -> falls back to R_visible (see _r_cond_expr). R_miss is never read on
    # the mixture path.
    R_cond: object = None
    # obs_bias: constant conditional measurement bias b in pixels, shape (2,1).
    # None -> zero. Plumbing for the companion measurement workstream; no value
    # is invented here.
    obs_bias: object = None
    # Charge risk ONCE on the terminal belief instead of summing it per step.
    #
    # Integrated risk is a length proxy. Measured on the route candidates for
    # thesis09_lane08E_to_lane12W: corr(length, risk) = +0.99 with a 12.6x
    # spread across candidates, against a 1.7x spread in ambiguity. Risk
    # therefore out-ranges the visibility signal by about 7x and decides the
    # route on its own, so the planner took an 18.05 m route with two blind
    # steps over a fully covered 20.45 m one -- ambiguity preferred the covered
    # route by 24 nats and integrated risk penalised it by 83.
    #
    # Goal arrival is a HARD CONSTRAINT (`_prefer_candidate` ranks terminal
    # feasibility above cost), so risk no longer has to pull the robot toward
    # the goal; that job is already done. Charging it on the terminal belief
    # alone keeps what risk is for -- arriving with a belief that matches the
    # goal prior -- and removes the per-step accumulation that encoded duration.
    # Compare "Terminal Matters" (arXiv 2605.09046), where uncertainty enters
    # only through the terminal cost.
    terminal_risk_only: bool = False
    # Input-dependent process-noise PSDs (encoder_noise_model.ENCODER_PSD); None keeps
    # the constant process_noise_xy/theta PSDs.
    process_noise_psd: object = None


def _require_casadi():
    if not casadi_available():
        raise RuntimeError("CasADi is not available")


def _clip_expr(x, lo, hi):
    return ca.fmin(ca.fmax(x, lo), hi)


def make_g_from_homography(H):
    """
    Return a CasADi-friendly planar homography observation function.
    H: (3,3) projective mapping -> returns Function(state(3,) -> uv(2,))
    """
    _require_casadi()
    H_ca = ca.DM(np.asarray(H, dtype=float))
    state = ca.MX.sym('state', 3)
    pt = ca.vertcat(state[0], state[1], 1.0)
    pix = ca.mtimes(H_ca, pt)
    uv = ca.vertcat(pix[0] / pix[2], pix[1] / pix[2])
    return ca.Function('homography_uv', [state], [uv], ['state'], ['uv'])


def wrap_angle_ca(theta):
    return ca.atan2(ca.sin(theta), ca.cos(theta))


def unicycle_step_ca(state, control, dt):
    x = state[0] + control[0] * dt * ca.cos(state[2])
    y = state[1] + control[0] * dt * ca.sin(state[2])
    theta = wrap_angle_ca(state[2] + control[1] * dt)
    return ca.vertcat(x, y, theta)


def unicycle_jacobian_ca(state, control, dt):
    theta = state[2]
    v = control[0]
    return ca.vertcat(
        ca.horzcat(1.0, 0.0, -v * dt * ca.sin(theta)),
        ca.horzcat(0.0, 1.0, v * dt * ca.cos(theta)),
        ca.horzcat(0.0, 0.0, 1.0),
    )


def unicycle_process_noise_ca(process_noise_xy, process_noise_theta, dt, theta, v, w=None, psd=None):
    c = ca.cos(theta)
    s = ca.sin(theta)
    if psd is not None:   # input-dependent PSDs, planning.core.encoder_noise_model
        from planning.core.encoder_noise_model import encoder_psd
        sig_v2, sig_w2 = encoder_psd(v, 0.0 if w is None else w, psd)
    else:
        sig_v2 = process_noise_xy ** 2
        sig_w2 = process_noise_theta ** 2

    # Q_d matrix elements
    # Row 0
    q00 = sig_v2 * (c ** 2) * dt + (1.0 / 3.0) * (v ** 2) * (s ** 2) * sig_w2 * (dt ** 3)
    q01 = sig_v2 * c * s * dt - (1.0 / 3.0) * (v ** 2) * c * s * sig_w2 * (dt ** 3)
    q02 = -0.5 * v * s * sig_w2 * (dt ** 2)

    # Row 1
    q10 = q01
    q11 = sig_v2 * (s ** 2) * dt + (1.0 / 3.0) * (v ** 2) * (c ** 2) * sig_w2 * (dt ** 3)
    q12 = 0.5 * v * c * sig_w2 * (dt ** 2)

    # Row 2
    q20 = q02
    q21 = q12
    q22 = sig_w2 * dt

    return ca.vertcat(
        ca.horzcat(q00, q01, q02),
        ca.horzcat(q10, q11, q12),
        ca.horzcat(q20, q21, q22)
    )


def _ensure_symmetric_pd(M, eps=1e-9):
    _require_casadi()
    dim = int(M.size1())
    return 0.5 * (M + M.T) + float(eps) * ca.DM.eye(dim)


def _det_2x2(M):
    return M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]


def _det_3x3(M):
    return (
        M[0, 0] * (M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1])
        - M[0, 1] * (M[1, 0] * M[2, 2] - M[1, 2] * M[2, 0])
        + M[0, 2] * (M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0])
    )


def _logdet_small_pd(M):
    dim = int(M.size1())
    if dim == 2 and int(M.size2()) == 2:
        det_val = _det_2x2(M)
    elif dim == 3 and int(M.size2()) == 3:
        det_val = _det_3x3(M)
    else:
        raise RuntimeError(f"Unsupported matrix size for CasADi logdet surrogate: {dim}x{int(M.size2())}")
    return ca.log(ca.fmax(det_val, 1e-12))


def _chol_2x2(M, eps=1e-9):
    M = 0.5 * (M + M.T)
    a = ca.fmax(M[0, 0], eps)
    l11 = ca.sqrt(a)
    l21 = M[1, 0] / l11
    diag22 = ca.fmax(M[1, 1] - l21 * l21, eps)
    l22 = ca.sqrt(diag22)
    return ca.vertcat(
        ca.horzcat(l11, 0.0),
        ca.horzcat(l21, l22),
    )


def _xy_visibility_sigma_points_ca(mean_xy, cov_xy, kappa=1.0):
    mean_xy = ca.reshape(mean_xy, 2, 1)
    cov_xy = 0.5 * (cov_xy + cov_xy.T)
    kappa = max(float(kappa), 1e-6)
    scale = math.sqrt(2.0 + kappa)
    chol = _chol_2x2(cov_xy + 1e-9 * ca.DM.eye(2))
    spread = scale * chol
    sigma_points = (
        mean_xy,
        mean_xy + spread[:, 0],
        mean_xy - spread[:, 0],
        mean_xy + spread[:, 1],
        mean_xy - spread[:, 1],
    )
    weights = (
        kappa / (2.0 + kappa),
        1.0 / (2.0 * (2.0 + kappa)),
        1.0 / (2.0 * (2.0 + kappa)),
        1.0 / (2.0 * (2.0 + kappa)),
        1.0 / (2.0 * (2.0 + kappa)),
    )
    return sigma_points, weights


def expected_visibility_ca(mean, cov, prob_state, *, kappa=1.0, lo=1e-4, hi=1.0 - 1e-4):
    """
    Approximate expected visibility via Unscented Transform (Sigma Points).
    mean: (3,) S: (3,3) -> return p_vis: float (scalar MX)
    """
    sigma_points_xy, weights = _xy_visibility_sigma_points_ca(mean[:2], cov[:2, :2], kappa=kappa)
    total = 0
    for sigma_xy, weight in zip(sigma_points_xy, weights):
        state = ca.vertcat(sigma_xy[0], sigma_xy[1], mean[2])
        total += float(weight) * _clip_expr(prob_state(state), lo, hi)
    return _clip_expr(total, lo, hi)


def _smoothstep_ca(x):
    x = _clip_expr(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _softplus_expr(x):
    x = _clip_expr(x, -60.0, 60.0)
    return ca.log(1.0 + ca.exp(x))


def _visibility_effective_score_ca(p_vis, params: CasadiEfeParams):
    del params
    return _clip_expr(p_vis, 1e-4, 1.0 - 1e-4)


def _blend_observation_covariance_ca(p_vis_eff, params: CasadiEfeParams):
    visible_prec = ca.vertcat(
        1.0 / ca.fmax(params.R_visible[0, 0], 1e-6),
        1.0 / ca.fmax(params.R_visible[1, 1], 1e-6),
    )
    miss_prec = ca.vertcat(
        1.0 / ca.fmax(params.R_miss[0, 0], 1e-6),
        1.0 / ca.fmax(params.R_miss[1, 1], 1e-6),
    )
    blended_prec = p_vis_eff * visible_prec + (1.0 - p_vis_eff) * miss_prec
    return ca.diag(1.0 / ca.fmax(blended_prec, 1e-9))


def goal_obs_cov_ca_for_progress(params: CasadiEfeParams, progress):
    progress_fast = ca.power(_clip_expr(progress, 0.0, 1.0), params.goal_tightening_power)
    a = _smoothstep_ca(progress_fast)
    sigma_u = (1.0 - a) * params.goal_prior_u_std_start + a * params.goal_prior_u_std_final
    sigma_v = (1.0 - a) * params.goal_prior_v_std_start + a * params.goal_prior_v_std_final
    return ca.diag(ca.vertcat(ca.power(sigma_u, 2), ca.power(sigma_v, 2)))


def et1_ca(m, S, R_eff, g, dg):
    """
    Extended Transform (1st order) linearization.
    m: (3,) S: (3,3) R_eff: (2,2) g/dg: mapping functions
    returns: mu_y: (2,) Sigma_y: (2,2) Gamma_xy: (3,2)
    """
    Jm = dg(m)
    mu = g(m)
    Sigma = ca.mtimes([Jm, S, Jm.T]) + R_eff
    Gamma = ca.mtimes(S, Jm.T)
    return mu, Sigma, Gamma


def et2_ca(m, S, R_eff, g, dg, d2g):
    """
    Second-order extended transform using CasADi Jacobians and Hessians.
    m: (3,) S: (3,3) R_eff: (2,2) g/dg/d2g: mapping functions
    returns: mu_y: (2,) Sigma_y: (2,2) Gamma_xy: (3,2)
    """
    Jm = dg(m)
    mu0 = g(m)
    dim_y = int(mu0.size1())
    hessians = [h_fn(m) for h_fn in d2g]

    aux1 = []
    for H_i in hessians:
        aux1.append(ca.trace(ca.mtimes(H_i, S)))

    aux2 = ca.MX.zeros(dim_y, dim_y)
    for i, H_i in enumerate(hessians):
        for j, H_j in enumerate(hessians):
            aux2[i, j] = ca.trace(ca.mtimes([H_i, S, H_j, S]))

    mu = mu0 + 0.5 * ca.vertcat(*aux1)
    Sigma = ca.mtimes([Jm, S, Jm.T]) + 0.5 * aux2 + R_eff
    Gamma = ca.mtimes(S, Jm.T)
    return mu, Sigma, Gamma


def risk_ca(mu, Sigma, goal_mu, goal_S, one_sided=True):
    """
    KL-Divergence based instrumental risk cost.
    mu: (2,) Sigma: (2,2) goal_mu: (2,) goal_S: (2,2)

    The KL between the predicted observation and the goal prior is V-shaped in
    the predicted covariance, with its minimum where the two are equal. Only the
    right branch expresses a navigation preference: a belief WIDER than the task
    tolerance is a problem, a belief SHARPER than it is not. Taking the KL
    unmodified therefore pays the planner to let the belief drift whenever the
    prediction is sharper than the goal prior, which is the whole operating range
    here (belief 2-13 cm against a 35 cm tolerance).

    ``one_sided`` clips the covariance ARGUMENT up to the goal prior, so a
    sharper-than-tolerance prediction is scored as if it sat exactly at
    tolerance. Certainty is then neither rewarded nor punished, and widening past
    the tolerance carries the full KL penalty. The mean (goal-distance) part is
    untouched.
    """
    Sigma_pd = _ensure_symmetric_pd(Sigma)
    if one_sided:
        # The goal prior is isotropic here, so clipping the eigenvalues reduces
        # to clipping the diagonal scale. Blend so the expression stays smooth.
        goal_scale = _logdet_small_pd(_ensure_symmetric_pd(goal_S))
        floor = ca.exp(goal_scale / ca.DM(float(goal_mu.size1())))
        Sigma_pd = _ensure_symmetric_pd(
            Sigma_pd + ca.fmax(floor - ca.trace(Sigma_pd) / float(goal_mu.size1()), 0.0)
            * ca.DM.eye(int(Sigma_pd.size1())))
    goal_S_pd = _ensure_symmetric_pd(goal_S)
    diff = goal_mu - mu
    if goal_S_pd.sparsity().is_diag():
        # The goal prior is diagonal. Explicit componentwise division is the
        # same diagonal solve, and avoids CasADi 3.7.2's DiagSolve serialization
        # defect (a saved function otherwise cannot be loaded in another run).
        diagonal = ca.diag(goal_S_pd)
        solved_cov = Sigma_pd / ca.repmat(diagonal, 1, Sigma_pd.size2())
        solved_diff = diff / diagonal
    else:
        solved_cov = ca.solve(goal_S_pd, Sigma_pd)
        solved_diff = ca.solve(goal_S_pd, diff)
    trace_term = ca.trace(solved_cov)
    quad_term = ca.mtimes([diff.T, solved_diff])
    dim = int(goal_mu.size1())
    logdet_goal = _logdet_small_pd(goal_S_pd)
    logdet_sigma = _logdet_small_pd(Sigma_pd)
    return 0.5 * (trace_term - dim + quad_term[0, 0] + logdet_goal - logdet_sigma)


def ambiguity_ca(Sigma, Gamma, S):
    """
    Expected information gain (epistemic uncertainty reduction).
    Sigma: (2,2) Gamma: (3,2) S: (3,3)
    """
    S_pd = _ensure_symmetric_pd(S)
    Sigma_cond = Sigma - ca.mtimes([Gamma.T, ca.solve(S_pd, Gamma)])
    Sigma_cond = _ensure_symmetric_pd(Sigma_cond)
    dim = int(Sigma_cond.size1())
    logdet = _logdet_small_pd(Sigma_cond)
    return 0.5 * (dim * math.log(2.0 * math.pi * math.e) + logdet)


def state_posterior_cov_ca(S, Sigma, Gamma):
    """Expected state covariance after the planner-facing camera update."""
    S_pd = _ensure_symmetric_pd(S)
    Sigma_pd = _ensure_symmetric_pd(Sigma)
    S_post = S_pd - ca.mtimes([Gamma, ca.solve(Sigma_pd, Gamma.T)])
    return _ensure_symmetric_pd(S_post)


# ---------------------------------------------------------------------------
# Hit/miss expected-belief mixture (opt-in; see CasadiEfeParams.use_hit_miss_mixture)
#
# The frozen path precision-blends the visibility probability into ONE effective
# covariance (_blend_observation_covariance_ca) and then takes ONE deterministic
# ET update with it. That conflates two different random quantities: whether a
# usable detection ARRIVES (Bernoulli) and how accurate it is GIVEN that it
# arrived (Gaussian). A missed detection produces no filter update at all — it is
# not a Gaussian measurement with a huge R.
#
# The mixture below models them separately. As a side effect it needs no miss
# endpoint at all, which dissolves the unreconciled 40 px (offline) vs 120 px
# (runtime) r_miss constant that reliability.covariance_mapping.MissEndpointPolicy
# still refuses to bless: on this path the miss branch simply takes no update.
# ---------------------------------------------------------------------------

def _as_dm(M):
    """Accept numpy / list / DM / MX uniformly, without copying symbolics."""
    if isinstance(M, (ca.DM, ca.MX, ca.SX)):
        return M
    return ca.DM(np.asarray(M, dtype=float))


def _r_cond_expr(params: CasadiEfeParams, state, R_cond_state=None):
    """Conditional measurement covariance R_cond, given a usable detection.

    This is the ONLY measurement covariance on the mixture path. Availability is
    carried by the Bernoulli branch weight, never by inflating R.

    Sources, in priority order:

    1. ``R_cond_state(state)`` — a measured, spatially varying field R_cond(x).
       Nothing is wired to this yet; the hook exists so the companion
       measurement workstream can drop a field in without touching this file.
    2. ``params.R_cond`` — a measured constant, once one is recorded.
    3. ``params.R_visible`` — **the current default and the current reality.**
       No R_cond field has been measured yet, so the mixture reuses the existing
       visible-regime covariance unchanged. That is a placeholder standing in for
       an unmeasured quantity, not an estimate of it.
    """
    if R_cond_state is not None:
        return _as_dm(R_cond_state(state))
    if params.R_cond is not None:
        return _as_dm(params.R_cond)
    return _as_dm(params.R_visible)


def _obs_bias_expr(params: CasadiEfeParams, state, obs_bias_state=None, dim=2):
    """Conditional measurement bias b(x), in pixels. Defaults to zero.

    Same contract as _r_cond_expr: a hook for a measured field, a constant
    fallback, and zero when nothing has been measured. No value is invented.
    """
    if obs_bias_state is not None:
        return ca.reshape(_as_dm(obs_bias_state(state)), dim, 1)
    if params.obs_bias is not None:
        return ca.reshape(_as_dm(params.obs_bias), dim, 1)
    return ca.DM.zeros(dim, 1)


def hit_miss_posterior_ca(S, J, R_cond, p_use):
    """Two-branch expected belief under Bernoulli(p_use) detection availability.

    ``S`` is the predicted (prior) state covariance P-, ``J`` the observation
    Jacobian H, ``R_cond`` the conditional measurement covariance, ``p_use`` the
    probability that a usable measurement arrives.

        P_hit  = P- - P- H^T (H P- H^T + R_cond)^-1 H P-     (a measurement arrives)
        P_miss = P-                                          (nothing arrives)
        E[P+]  = p_use * P_hit + (1 - p_use) * P_miss

    Returns ``(P_mix, P_hit, S_sym, Sigma)`` where ``Sigma`` is the innovation
    covariance ``H P- H^T + R_cond``.

    PSD hygiene. ``P_hit`` is built in **Joseph form**,

        P_hit = (I - K H) P- (I - K H)^T + K R_cond K^T,   K = P- H^T Sigma^-1,

    which is a sum of two congruences of PSD matrices and is therefore PSD *by
    construction*, not by repair. That matters: the algebraically equivalent
    short form ``P- - K Sigma K^T`` is a difference, and differences of nearly
    equal covariances are exactly how the 2026-07-29 indefinite-belief bug was
    produced. Everything here is whole-matrix; no block is ever spliced, and no
    eigenvalue floor is applied to the state-space result at all. The only floor
    in the expression is INNOVATION_COV_FLOOR_PX2 on the matrix being inverted.

    A convex combination of two PSD matrices is PSD, so ``P_mix`` inherits
    PSD-ness from its branches; ``p_use in [0, 1]`` is guaranteed upstream by the
    clipping in ``_visibility_effective_score_ca``.

    Everything is a smooth CasADi expression — products, one linear solve, and a
    convex combination. No ``if`` on a symbolic value, so the NLP stays
    differentiable.
    """
    _require_casadi()
    n = int(S.size1())
    S_sym = 0.5 * (S + S.T)
    R_sym = 0.5 * (R_cond + R_cond.T)

    Sigma = ca.mtimes([J, S_sym, J.T]) + R_sym
    Sigma = 0.5 * (Sigma + Sigma.T) + INNOVATION_COV_FLOOR_PX2 * ca.DM.eye(int(Sigma.size1()))

    # K = P- H^T Sigma^-1, written as a solve against the symmetric Sigma.
    K = ca.solve(Sigma, ca.mtimes(J, S_sym)).T
    A = ca.DM.eye(n) - ca.mtimes(K, J)
    P_hit = ca.mtimes([A, S_sym, A.T]) + ca.mtimes([K, R_sym, K.T])
    P_hit = 0.5 * (P_hit + P_hit.T)

    P_mix = p_use * P_hit + (1.0 - p_use) * S_sym
    return P_mix, P_hit, S_sym, Sigma


def expected_posterior_cov_ca(S, J, R_cond, p_use):
    """E[P+] over the hit/miss mixture. See :func:`hit_miss_posterior_ca`."""
    return hit_miss_posterior_ca(S, J, R_cond, p_use)[0]


def _differential_entropy_ca(P):
    dim = int(P.size1())
    return 0.5 * (dim * math.log(2.0 * math.pi * math.e) + _logdet_small_pd(0.5 * (P + P.T)))


def _anchored_ambiguity_ca(P, P_ref):
    """Ambiguity measured against an ideally observed pose.

        J_amb = 0.5 * max( log( |P| / |P_ref| ), 0 )

    The ET1 ambiguity is an ABSOLUTE differential entropy,
    ``0.5(D_y log 2*pi*e + log|R_eff|)``, so it carries a large
    route-independent additive constant whose SIGN depends only on the units
    ``R_eff`` is written in. On a fixed horizon every candidate accumulates that
    constant the same number of times and it cancels. Under the smooth arrival
    gate duration is a free variable, so it instead contributes ``constant *
    T``: a duration term carrying the sign of an arbitrary unit choice, which
    can reward a route for lasting longer.

    Dividing by ``P_ref`` cancels the constant (including ``D_y log 2*pi*e``),
    leaving a dimensionless quantity. Clipping at zero means no route can be paid
    for taking longer. The result differs from Kouw Lemma 1 by an additive
    constant only, so on a fixed horizon it is rank-identical to the published
    objective.
    """
    ratio = _logdet_small_pd(0.5 * (P + P.T)) - _logdet_small_pd(0.5 * (P_ref + P_ref.T))
    return 0.5 * ca.fmax(ratio, 0.0)


def expected_posterior_uncertainty_ca(S, J, R_cond, p_use):
    """Expected posterior uncertainty (differential entropy) over the mixture.

        E[H(P+)] = p_use * H(P_hit) + (1 - p_use) * H(P-)
                 = H(P-) - p_use * I(x; y)

    i.e. the prior uncertainty minus the *availability-weighted* information gain
    — which is exactly the epistemic term of EFE, written honestly: a measurement
    that never arrives buys no information.

    Note this is E[H], the expectation of the branch entropies, not H(E[P]) the
    entropy of the mixture covariance. E[H] is the correct expected uncertainty,
    because the realised posterior is one branch or the other and never the
    averaged matrix. Both are smooth; E[H] is itself a convex combination of two
    smooth scalars, so it costs nothing extra. (Trace-based readouts do not
    distinguish the two — trace is linear — so the fig_e1 sweep is unaffected.)

    This REPLACES the observation-space ambiguity term on the mixture path. It
    has to: with R_cond spatially constant, ``logdet(R_cond)`` is constant, so
    the frozen ambiguity term carries no visibility signal once the miss endpoint
    is gone. It also fixes a units artifact — a constant px^2 ambiguity scaled by
    availability makes the sign of the visibility preference depend on whether
    ``logdet(R_cond)`` happens to be positive. Here the term is in state units
    throughout and is monotone decreasing in ``p_use`` by construction, since
    ``I(x; y) >= 0``.

    SCALE NOTE (reported, not tuned): the frozen term is a px^2 log-determinant
    and this one is an m^2/rad^2 log-determinant, so the two are not on the same
    numeric scale and ``ambiguity_scale`` does not mean the same thing across the
    flag. That is a fact about the change, recorded here for whoever runs the
    closed-loop campaign. No reweighting is proposed or applied.
    """
    _P_mix, P_hit, S_sym, _Sigma = hit_miss_posterior_ca(S, J, R_cond, p_use)
    return p_use * _differential_entropy_ca(P_hit) + (1.0 - p_use) * _differential_entropy_ca(S_sym)


def visibility_aware_unicycle_efe_ca(
    u_flat,
    m0,
    S0,
    goal_obs,
    goal_xy,
    progress_index0,
    params: CasadiEfeParams,
    g,
    dg,
    *,
    approx='ET1',
    d2g=None,
    p_vis_state=None,
    nogo_cost=None,
    nogo_belief_cost=None,
    R_cond_state=None,
    obs_bias_state=None,
    R_plan_state=None,
):
    """
    Core Expected Free Energy functional for a unicycle agent.
    Iteratively propagates Gaussian state (m, S) over params.time_horizon.
    Goal-seeking emerges from the EFE goal-prior in the risk term (no external
    goal-distance reward).

    Two measurement models, selected by ``params.use_hit_miss_mixture``:

    * OFF (default, FROZEN): one deterministic ET update with the precision-blend
      covariance ``R_plan(p_vis)``. Untouched, bit-for-bit.
    * ON: the availability variable is Bernoulli, so every measurement-dependent
      term is evaluated in both branches and averaged with weights
      ``(p_use, 1 - p_use)``:

          G_t = p_use * [risk(hit)] + (1 - p_use) * [risk(miss)]  +  E[H(P+)]

      The hit branch's predicted observation carries ``R_cond`` (and bias ``b``);
      the miss branch has no measurement, so no measurement noise and no bias —
      only the projected belief compared against the goal prior. The epistemic
      term is the expected posterior uncertainty over the mixture.
    """
    m = m0
    S = S0
    total_risk = 0
    total_amb = 0
    total_control = 0
    total_nogo = 0
    denom = float(max(params.goal_progress_n_steps, 1))  # goal-prior anneal schedule length

    for t in range(params.time_horizon):
        u_t = ca.vertcat(u_flat[2 * t], u_flat[2 * t + 1])
        m_prev = m
        m = unicycle_step_ca(m_prev, u_t, params.dt)
        F = unicycle_jacobian_ca(m_prev, u_t, params.dt)
        Q_t = unicycle_process_noise_ca(
            params.process_noise_xy,
            params.process_noise_theta,
            params.dt,
            m_prev[2],
            u_t[0],
            u_t[1],
            params.process_noise_psd,
        )
        S = ca.mtimes([F, S, F.T]) + Q_t

        p_vis = 1.0
        if p_vis_state is not None:
            p_vis = expected_visibility_ca(
                m,
                S,
                p_vis_state,
                kappa=params.visibility_sigma_kappa,
            )
        p_vis_eff = _visibility_effective_score_ca(p_vis, params)

        progress = (progress_index0 + float(t)) / denom
        goal_cov_t = goal_obs_cov_ca_for_progress(params, progress)
        weight_t = params.discount_gamma ** t

        if params.use_hit_miss_mixture:
            # p_use: probability a USABLE measurement arrives. Sourced from the
            # same (frozen, unchanged) expected-visibility field as before; the
            # observability workstream is measuring a proper p_c(x, y) to put here.
            p_use = p_vis_eff
            R_cond = _r_cond_expr(params, m, R_cond_state)
            bias = _obs_bias_expr(params, m, obs_bias_state)

            # One ET call serves both branches: ET1/ET2 add R_eff as the final
            # term, so the no-measurement branch is exactly Sigma_hit - R_cond.
            # Cheaper than two calls and algebraically identical.
            if approx == 'ET1':
                mu, Sigma_hit, Gamma = et1_ca(m, S, R_cond, g, dg)
            elif approx == 'ET2':
                mu, Sigma_hit, Gamma = et2_ca(m, S, R_cond, g, dg, d2g or [])
            else:
                raise RuntimeError(f"Unsupported CasADi approximation: {approx}")
            Sigma_miss = Sigma_hit - R_cond

            # Risk: expectation over the two branches. The bias is a SENSOR bias,
            # so it shifts the predicted observation only when one is received.
            risk_hit = risk_ca(mu + bias, Sigma_hit, goal_obs, goal_cov_t)
            risk_miss = risk_ca(mu, Sigma_miss, goal_obs, goal_cov_t)
            total_risk += (weight_t * params.risk_scale
                           * (p_use * risk_hit + (1.0 - p_use) * risk_miss))

            # Epistemic: expected posterior uncertainty over the mixture, i.e.
            # H(P-) - p_use * I(x; y). Replaces the observation-space ambiguity
            # term (see expected_posterior_uncertainty_ca for why it has to).
            J_t = dg(m)
            total_amb += (weight_t * params.ambiguity_scale
                          * expected_posterior_uncertainty_ca(S, J_t, R_cond, p_use))

            total_control += weight_t * params.control_weight * ca.sumsqr(u_t)
            if nogo_belief_cost is not None and params.use_belief_nogo_cost:
                S_drive = expected_posterior_cov_ca(S, J_t, R_cond, p_use)
                total_nogo += weight_t * nogo_belief_cost(m, S_drive)
            elif nogo_cost is not None:
                total_nogo += weight_t * nogo_cost(m)
            continue

        # --- FROZEN precision-blend path (do not modify) ----------------------
        # An opt-in network supplies a full covariance in the SAME fixed camera
        # cost chart. With no adapter, the published single-camera path is unchanged.
        R_plan = (_blend_observation_covariance_ca(p_vis_eff, params)
                  if R_plan_state is None else R_plan_state(m, S))
        if approx == 'ET1':
            mu, Sigma, Gamma = et1_ca(m, S, R_plan, g, dg)
        elif approx == 'ET2':
            mu, Sigma, Gamma = et2_ca(m, S, R_plan, g, dg, d2g or [])
        else:
            raise RuntimeError(f"Unsupported CasADi approximation: {approx}")

        total_risk += weight_t * params.risk_scale * risk_ca(mu, Sigma, goal_obs, goal_cov_t)
        total_amb += weight_t * params.ambiguity_scale * ambiguity_ca(Sigma, Gamma, S)
        total_control += weight_t * params.control_weight * ca.sumsqr(u_t)
        if nogo_belief_cost is not None and params.use_belief_nogo_cost:
            S_drive = state_posterior_cov_ca(S, Sigma, Gamma)
            total_nogo += weight_t * nogo_belief_cost(m, S_drive)
        elif nogo_cost is not None:
            total_nogo += weight_t * nogo_cost(m)

    # Normalise per-step sums by the effective discounted horizon so that changing
    # time_horizon does not rescale the weight balance between terms.
    H_eff = sum(params.discount_gamma ** t for t in range(params.time_horizon))
    inv_H = 1.0 / max(H_eff, 1e-8)
    return (total_risk * inv_H + total_amb * inv_H
            + total_control * inv_H + total_nogo * inv_H)


def make_metric_network_efe_valgrad_fn(
    params: CasadiEfeParams,
    expected_belief_state,
    *,
    effective_covariance=None,
    reference_covariance=None,
    goal_std_m,
    goal_std_start_m=None,
    arrival_radius_m=0.0,
    arrival_softness_m=0.25,
    nogo_cost=None,
    nogo_belief_cost=None,
    obstacle_substeps=1,
    control_block_steps=1,
):
    """Build a world-XY expected-belief objective for a camera network.

    The network callback returns the exact one-step expectation over conditionally
    independent per-camera Bernoulli hit events. Each hit uses its conditional
    metric covariance; a miss performs no update. The expected posterior is used
    as the covariance for the next planning step.
    """
    _require_casadi()
    goal_std_m = float(goal_std_m)
    if not np.isfinite(goal_std_m) or goal_std_m <= 0.:
        raise ValueError('metric network goal_std_m must be finite and positive')
    # Anchor for the ambiguity term. Required whenever an ambiguity path is
    # live: without it the term is an absolute entropy and the arrival gate
    # turns its constant into a duration reward (see _anchored_ambiguity_ca).
    if reference_covariance is None:
        raise ValueError(
            'metric network objective requires reference_covariance: the '
            'ambiguity term is anchored to the ideally observed pose')
    R_ref_np = np.asarray(reference_covariance, dtype=float)
    if R_ref_np.shape != (2, 2) or not np.isfinite(R_ref_np).all():
        raise ValueError('reference covariance must be a finite 2x2 matrix')
    R_ref_np = 0.5 * (R_ref_np + R_ref_np.T)
    if np.linalg.eigvalsh(R_ref_np).min() <= 0.:
        raise ValueError('reference covariance must be positive definite')
    R_ref_dm = ca.DM(R_ref_np)
    # A constant goal prior makes the risk term a squared distance to the goal,
    # so the objective reduces to shortest path and the belief terms cannot
    # affect the route. Annealing the prior from loose to tight reproduces the
    # pixel-objective schedule already used by goal_obs_cov_ca_for_progress:
    # early steps are free to be shaped by the belief terms, late steps must
    # arrive. goal_std_start_m=None keeps the previous constant behaviour.
    if goal_std_start_m is None:
        goal_std_start_m = goal_std_m
    goal_std_start_m = float(goal_std_start_m)
    if not np.isfinite(goal_std_start_m) or goal_std_start_m <= 0.:
        raise ValueError('metric network goal_std_start_m must be finite and positive')
    obstacle_substeps = int(obstacle_substeps)
    if obstacle_substeps < 1:
        raise ValueError('obstacle_substeps must be positive')
    control_block_steps = int(control_block_steps)
    if control_block_steps < 1:
        raise ValueError('control_block_steps must be positive')

    control_blocks = int(math.ceil(params.time_horizon / control_block_steps))
    u_flat = ca.MX.sym('u_flat', control_blocks * params.Du)
    m0 = ca.MX.sym('m0', 3)
    S0 = ca.MX.sym('S0', 3, 3)
    goal_obs = ca.MX.sym('goal_obs', 2)
    goal_xy = ca.MX.sym('goal_xy', 2)
    progress_index0 = ca.MX.sym('progress_index0')
    m = m0
    S = S0

    goal_cov_const = (goal_std_m ** 2) * ca.DM.eye(2)
    anneal_denom = float(max(params.goal_progress_n_steps, 1))
    # Steps after the predicted mean reaches the goal are not part of the plan
    # being evaluated: the horizon is fixed, so a short route would otherwise
    # bank its remaining steps parked at the goal and accumulate that cell's
    # ambiguity. A route must be costed by what it takes to DRIVE it. The gate
    # is a smooth sigmoid in the distance to the goal so the objective stays
    # differentiable; ``arrival_softness_m`` sets its width.
    arrival_radius = float(arrival_radius_m)
    arrival_softness = max(float(arrival_softness_m), 1e-6)
    active = 1.0
    total_risk = 0.
    total_amb = 0.
    total_control = 0.
    total_nogo = 0.
    total_active_weight = 0.
    for t in range(params.time_horizon):
        block = t // control_block_steps
        u_t = ca.vertcat(u_flat[2*block], u_flat[2*block+1])
        m_prev = m
        m = unicycle_step_ca(m_prev, u_t, params.dt)
        F = unicycle_jacobian_ca(m_prev, u_t, params.dt)
        Q_t = unicycle_process_noise_ca(
            params.process_noise_xy, params.process_noise_theta,
            params.dt, m_prev[2], u_t[0], u_t[1], params.process_noise_psd,
        )
        S = .5 * (F @ S @ F.T + Q_t + (F @ S @ F.T + Q_t).T)
        weight_t = params.discount_gamma ** t
        if goal_std_start_m == goal_std_m:
            goal_cov_t = goal_cov_const
        else:
            progress_t = (progress_index0 + float(t)) / anneal_denom
            a_t = _smoothstep_ca(ca.power(
                _clip_expr(progress_t, 0.0, 1.0), params.goal_tightening_power))
            sigma_t = (1.0 - a_t) * goal_std_start_m + a_t * goal_std_m
            goal_cov_t = ca.diag(ca.vertcat(
                ca.power(sigma_t, 2), ca.power(sigma_t, 2)))
        if arrival_radius > 0.:
            # 1 while still driving, decaying to 0 once inside the goal region.
            # ``norm_2`` has an undefined derivative at exactly zero.  Seed
            # routes deliberately reach the goal and then remain there for the
            # fixed-horizon tail, so that cusp produced a finite objective with
            # a NaN gradient and made every goal-reaching CasADi initialization
            # fall back before optimization.  This smooth norm is numerically
            # identical at navigation scale and has a finite zero-distance
            # derivative.
            reached = ca.sqrt(ca.sumsqr(m[:2] - goal_xy) + 1.0e-12)
            active = active * (1.0 / (1.0 + ca.exp(
                -(reached - arrival_radius) / arrival_softness)))
        total_active_weight += active * weight_t
        if params.terminal_risk_only:
            # Overwrite rather than accumulate: after the loop this holds the
            # risk of the FINAL belief only, so the term carries no duration.
            # It keeps the arrival gate and the discount so a route that is
            # already parked at the goal is scored at the goal, and it uses the
            # step's own annealed goal covariance, which is the tightest one.
            total_risk = active * weight_t * params.risk_scale * risk_ca(
                m[:2], S[:2, :2], goal_xy, goal_cov_t,
            )
        else:
            total_risk += active * weight_t * params.risk_scale * risk_ca(
                m[:2], S[:2, :2], goal_xy, goal_cov_t,
            )
        S_post, expected_entropy = expected_belief_state(m, S)
        if effective_covariance is None:
            total_amb += active * weight_t * params.ambiguity_scale * expected_entropy
        else:
            # Kouw (IWAI 2024) Lemma 1 with the first-order extended transform
            # of Theorem 1: Sigma - Gamma^T S^-1 Gamma reduces exactly to the
            # observation covariance, so ambiguity is 0.5(Dy log 2*pi*e +
            # log|R|). With one fixed sensor that is constant over states; here
            # R is the availability-weighted commissioned field, so the term
            # varies over the workspace and prefers well-observed states.
            R_eff = effective_covariance(m, S)
            # ANCHORED to the ideally observed pose: see
            # ``_anchored_ambiguity_ca``. The raw Lemma 1 entropy is absolute,
            # so under the arrival gate its route-independent constant became a
            # duration term signed by a unit choice.
            total_amb += (active * weight_t * params.ambiguity_scale
                          * _anchored_ambiguity_ca(.5 * (R_eff + R_eff.T), R_ref_dm))
        total_control += active * weight_t * params.control_weight * ca.sumsqr(u_t)
        if nogo_belief_cost is not None and params.use_belief_nogo_cost:
            total_nogo += active * weight_t * nogo_belief_cost(m, S_post)
        elif nogo_cost is not None:
            # With a 1 s global step, endpoint-only costs can miss a curved
            # transition through a prism or outside the driveable union. Sample
            # the same constant-twist motion used by the rollout validator.
            transition_cost = 0.
            for substep in range(1, obstacle_substeps + 1):
                substate = unicycle_step_ca(
                    m_prev, u_t, params.dt * float(substep) / obstacle_substeps)
                transition_cost += nogo_cost(substate)
            total_nogo += active * weight_t * transition_cost / obstacle_substeps
        S = S_post

    # IWAI-like fixed-horizon accounting for variable-duration global routes:
    # compare their mean active discounted stage cost. A spatially constant
    # ambiguity can no longer become an accidental route-length term.
    objective = ((total_risk + total_amb + total_control + total_nogo)
                 / ca.fmax(total_active_weight, 1e-8))
    gradient = ca.gradient(objective, u_flat)
    # Keep the common six-input interface so cache loading and the optimizer do
    # not need a second calling convention. goal_obs and progress are deliberate
    # compatibility inputs; only goal_xy is used by this metric objective.
    valgrad = ca.Function(
        'metric_network_efe_valgrad',
        [u_flat, m0, S0, goal_obs, goal_xy, progress_index0],
        [objective, gradient],
    )
    return _make_valgrad_wrapper(valgrad)


def _make_valgrad_wrapper(valgrad):
    """Wrap a built valgrad ca.Function as a numpy-in/out callable."""

    def _wrapper(
        u_val,
        m_val,
        S_val,
        goal_obs_val,
        goal_xy_val,
        progress_index0_val,
    ):
        val, grad = valgrad(
            np.asarray(u_val, dtype=float).reshape((-1, 1)),
            np.asarray(m_val, dtype=float).reshape((3, 1)),
            np.asarray(S_val, dtype=float).reshape((3, 3)),
            np.asarray(goal_obs_val, dtype=float).reshape((2, 1)),
            np.asarray(goal_xy_val, dtype=float).reshape((2, 1)),
            np.asarray(float(progress_index0_val), dtype=float),
        )
        return float(np.asarray(val, dtype=float).reshape(-1)[0]), np.asarray(grad, dtype=float).reshape(-1)

    # Expose the immutable graph for serialization / optional compiled evaluation.
    _wrapper.casadi_function = valgrad
    return _wrapper


def make_efe_valgrad_fn(
    params: CasadiEfeParams,
    H,
    *,
    approx='ET1',
    p_vis_state=None,
    nogo_cost=None,
    nogo_belief_cost=None,
    R_cond_state=None,
    obs_bias_state=None,
    R_plan_state=None,
):
    _require_casadi()
    approx = str(approx or 'ET1').upper()
    if approx not in ('ET1', 'ET2'):
        raise RuntimeError("CasADi EFE path supports only ET1 or ET2")
    if R_plan_state is not None and params.use_hit_miss_mixture:
        raise RuntimeError('network score proxy cannot be used as a Bernoulli hit/miss likelihood')
    if (R_cond_state is not None or obs_bias_state is not None) and not params.use_hit_miss_mixture:
        # Fail loudly rather than silently ignoring them: the frozen precision-blend
        # path has no place to put a spatially varying R_cond or a bias term, and a
        # caller passing them clearly expects the mixture.
        raise RuntimeError(
            "R_cond_state/obs_bias_state require params.use_hit_miss_mixture=True; "
            "the frozen precision-blend path ignores them"
        )

    g = make_g_from_homography(H)

    state_sym = ca.MX.sym('state_for_jac', 3)
    g_expr = g(state_sym)
    dg = ca.Function(
        'homography_uv_jac',
        [state_sym],
        [ca.jacobian(g_expr, state_sym)],
        ['state'],
        ['J'],
    )
    d2g = None
    if approx == 'ET2':
        d2g = []
        for i in range(int(g_expr.size1())):
            H_i, _ = ca.hessian(g_expr[i], state_sym)
            d2g.append(
                ca.Function(
                    f'homography_uv_hess_{i}',
                    [state_sym],
                    [H_i],
                    ['state'],
                    ['H'],
                )
            )

    u_flat = ca.MX.sym('u_flat', params.time_horizon * params.Du)
    m0 = ca.MX.sym('m0', 3)
    S0 = ca.MX.sym('S0', 3, 3)
    goal_obs = ca.MX.sym('goal_obs', 2)
    goal_xy = ca.MX.sym('goal_xy', 2)
    progress_index0 = ca.MX.sym('progress_index0')

    objective = visibility_aware_unicycle_efe_ca(
        u_flat,
        m0,
        S0,
        goal_obs,
        goal_xy,
        progress_index0,
        params,
        g,
        dg,
        approx=approx,
        d2g=d2g,
        p_vis_state=p_vis_state,
        nogo_cost=nogo_cost,
        nogo_belief_cost=nogo_belief_cost,
        R_cond_state=R_cond_state,
        obs_bias_state=obs_bias_state,
        R_plan_state=R_plan_state,
    )
    gradient = ca.gradient(objective, u_flat)
    valgrad = ca.Function(
        'visibility_aware_efe_valgrad',
        [u_flat, m0, S0, goal_obs, goal_xy, progress_index0],
        [objective, gradient],
        ['u_flat', 'm0', 'S0', 'goal_obs', 'goal_xy', 'progress_index0'],
        ['objective', 'gradient'],
    )

    return _make_valgrad_wrapper(valgrad)
