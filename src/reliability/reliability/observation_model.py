"""Factorized camera observation model: availability, bias, conditional noise, outliers.

This module holds the *measurement-side* algebra of the observation model that
the rest of the stack already assumes but never wrote down in one place:

    p(d, a, z | x, c, q) = p(d | x, q) · p(a | d, x, q) · p(z | a, d, x, c, q)
                            ^availability   ^valid association   ^conditional noise

The three factors are deliberately kept apart, because they enter planning and
filtering differently:

* **Availability** ``p_use = p_det · p_qual`` is a Bernoulli *event* probability.
  It belongs in a branch weight, never inside a covariance. The labels behind it
  are already frozen in :mod:`reliability.observation_opportunity`
  (``detection_label`` / ``quality_label`` / ``usable_label``); this module only
  provides the product and the planner-facing consequences.
* **Conditional accuracy** decomposes additively in *innovation* space
  (:func:`innovation_covariance`)::

      S = H P H' + R_pixel + R_cal + R_time + R_model

  Each term has a different physical source and a different estimation route, so
  they are separate arguments rather than one fitted blob. In particular
  ``H P H'`` is state uncertainty the filter already carries — learning it again
  inside a camera ``R`` double-counts it.
* **Systematic error** (bias ``b``) is *not* variance. Inflating ``R`` to cover a
  mean offset makes every good detection less informative while leaving the
  offset in place. Bias belongs in the mean function; see
  ``reliability.projection`` for the deployed along-bearing correction and
  ``experiments/external_camera_bias_model/`` for what it leaves behind.

Planner consequence (the part with teeth). The historical planner path folds
availability into one effective covariance — either the deployed diagonal
precision blend (``reliability.single_camera_adapter.precision_blend_covariance``,
mirroring ``planning.core.casadi_efe._blend_observation_covariance_ca``) or its
``R_miss -> inf`` limit :func:`scaled_covariance_baseline` (``R / p``). Both
then take ONE deterministic Kalman update. That is not what happens: a missed
detection yields *no update at all*, not a weak Gaussian one. The honest
expected posterior branches::

    E[P+] = p_use · P_hit + (1 - p_use) · P-

(:func:`expected_posterior_branch`, the stdlib twin of
``planning.core.casadi_efe.hit_miss_posterior_ca``). When an interface insists on
a single ``R_plan``, :func:`equivalent_isotropic_covariance` inverts the
criterion numerically instead of guessing a scaling law — and it makes explicit
that the answer depends on ``P-`` and ``H`` as well as on position.

Everything here is a pure function over plain nested sequences (stdlib only, no
numpy, no CasADi), measurement dimension 2 and arbitrary state dimension, so
offline tooling and ROS nodes can share it. Nothing in this module reads a clock,
a frame, or ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from reliability.contracts import (
    ContractValidationError,
    _finite_float,
    _finite_probability,
)
from reliability.conditional_covariance import (
    _log_det_2x2,
    _quad_form_inverse_2x2,
)
from reliability.fusion import (
    _mat_add,
    _mat_inv_2x2,
    _matrix_2x2,
    _symmetrize,
    _validate_spd,
)

Matrix2x2 = tuple[tuple[float, float], tuple[float, float]]
Matrix = tuple[tuple[float, ...], ...]

_LOG_TWO_PI = math.log(2.0 * math.pi)

#: Criterion names accepted by :func:`posterior_criterion` /
#: :func:`equivalent_isotropic_covariance`.
CRITERIA = ("trace", "position_trace", "logdet", "max_eig")

#: Upper bound on the isotropic variance search in
#: :func:`equivalent_isotropic_covariance`, as a multiple of the prior
#: measurement-space scale. Reaching it means "no finite R reproduces the target",
#: which is the correct answer at ``p_use = 0``.
_SIGMA2_SEARCH_DECADES = 24.0


# --------------------------------------------------------------------------- #
# 1. Availability: p_use = p_det * p_qual                                      #
# --------------------------------------------------------------------------- #


def usable_probability(p_detect: float, p_valid: float) -> float:
    """``p_use = p_det · p_qual``: probability a *usable* measurement arrives.

    ``p_detect`` is the probability the detector returns a robot candidate at all
    (``detection_label`` in :mod:`reliability.observation_opportunity`);
    ``p_valid`` is the probability that candidate is a valid localization given it
    exists (``quality_label`` given ``detection_label``). Keeping them apart
    matters because they fail for different reasons and in different places: a
    miss is a range/FOV/occlusion event, an invalid association is a detector or
    gating event. Only the product drives the branch weight.
    """

    p_d = _finite_probability(p_detect, field_name="p_detect")
    p_a = _finite_probability(p_valid, field_name="p_valid")
    return float(p_d * p_a)


# --------------------------------------------------------------------------- #
# 2. Additive innovation-covariance decomposition                              #
# --------------------------------------------------------------------------- #


def state_projection_covariance(
    jacobian: Sequence[Sequence[float]],
    cov_state: Sequence[Sequence[float]],
) -> Matrix2x2:
    """``H P H'`` — the measurement-space image of state uncertainty.

    This term is *already* in the filter's innovation covariance. It is listed
    here so that a fitted residual covariance can be corrected for it: residuals
    measured against a smoothed state estimate carry ``H P^s H'`` on top of the
    true measurement noise, and subtracting it is what makes the remainder a
    measurement property rather than a filter property.
    """

    h = _rect(jacobian, "jacobian", rows=2)
    p = _square(cov_state, "cov_state", dim=len(h[0]))
    return _symmetrize(_mat2_from_congruence(h, p))


def time_sync_covariance(
    jacobian: Sequence[Sequence[float]],
    state_rate: Sequence[float],
    sigma_tau_s: float,
) -> Matrix2x2:
    """``R_time = (H ẋ) σ_τ² (H ẋ)'`` — timestamp/latency uncertainty.

    An image whose effective exposure time is uncertain by ``σ_τ`` images a state
    that is displaced by ``ẋ σ_τ``. The resulting measurement-space term is
    rank-1 along the *direction of motion* and grows with speed, which is the
    principled reason the same camera at the same place can be more accurate on a
    slow pass than a fast one. It is PSD but singular by construction (a rank-1
    outer product), so it is a term to *add*, never a standalone ``R``.
    """

    h = _rect(jacobian, "jacobian", rows=2)
    rate = _vector(state_rate, "state_rate", dim=len(h[0]))
    sigma = _finite_float(sigma_tau_s, field_name="sigma_tau_s")
    if sigma < 0.0:
        raise ContractValidationError(f"sigma_tau_s must be >= 0, got {sigma}")
    drift = _mat_vec_rect(h, rate)
    var = sigma * sigma
    return (
        (var * drift[0] * drift[0], var * drift[0] * drift[1]),
        (var * drift[0] * drift[1], var * drift[1] * drift[1]),
    )


def calibration_covariance(
    jacobian_calibration: Sequence[Sequence[float]],
    cov_calibration: Sequence[Sequence[float]],
) -> Matrix2x2:
    """``R_cal = H_c Σ_c H_c'`` — first-order calibration uncertainty.

    CAVEAT, and it is not a small one: calibration error is *shared across
    frames*, not resampled per frame. A camera whose yaw is wrong by one degree
    is wrong by one degree on every consecutive measurement. Adding this term to
    a per-frame ``R`` therefore models a persistent offset as independent noise,
    which makes a filter **overconfident** over a burst of frames even though each
    single-frame covariance looks conservative.

    Use it for *marginal* uncertainty statements (how wrong can this camera be at
    this position) and for planning; do NOT wire it into the live per-frame
    update. The deployed treatment is the correct one for a filter: calibrate
    offline, freeze, and model what remains as a spatial bias field.
    """

    h_c = _rect(jacobian_calibration, "jacobian_calibration", rows=2)
    sigma_c = _square(cov_calibration, "cov_calibration", dim=len(h_c[0]))
    return _symmetrize(_mat2_from_congruence(h_c, sigma_c))


def innovation_covariance(
    *terms: Sequence[Sequence[float]],
    require_spd: bool = True,
) -> Matrix2x2:
    """Sum measurement-space covariance terms into one innovation covariance.

    ``S = H P H' + R_pixel + R_cal + R_time + R_model``. Individual terms may be
    singular (``R_time`` always is); the *sum* must be SPD, which is checked
    unless ``require_spd`` is False. Symmetrized on the way out.
    """

    if not terms:
        raise ContractValidationError("innovation_covariance needs at least one term")
    total: Matrix2x2 = ((0.0, 0.0), (0.0, 0.0))
    for index, term in enumerate(terms):
        matrix = _matrix_2x2(term, f"terms[{index}]")
        if not _is_psd_2x2(matrix):
            raise ContractValidationError(
                f"terms[{index}] must be positive semidefinite; "
                "every innovation-covariance contribution is a covariance"
            )
        total = _mat_add(total, matrix)
    out = _symmetrize(total)
    if require_spd:
        _validate_spd(out, "innovation_covariance")
    return out


# --------------------------------------------------------------------------- #
# 3. Planner-facing belief propagation                                         #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BranchPosterior:
    """Expected posterior under Bernoulli(``p_use``) measurement availability."""

    cov_expected: Matrix          #: E[P+] = p·P_hit + (1-p)·P-
    cov_hit: Matrix               #: posterior if a usable measurement arrives
    cov_miss: Matrix              #: posterior if none arrives (== the prior)
    innovation_cov: Matrix2x2     #: H P- H' + R_cond
    p_use: float


def expected_posterior_branch(
    cov_prior: Sequence[Sequence[float]],
    jacobian: Sequence[Sequence[float]],
    r_cond: Sequence[Sequence[float]],
    p_use: float,
    *,
    innovation_floor: float = 0.0,
) -> BranchPosterior:
    """``E[P+] = p_use · P_hit + (1 - p_use) · P-`` (stdlib twin of the runtime).

    ``P_hit`` is built in **Joseph form**,
    ``(I - K H) P- (I - K H)' + K R K'`` with ``K = P- H' S^-1``, so it is PSD by
    construction rather than by repair — the same choice, and for the same
    reason, as ``planning.core.casadi_efe.hit_miss_posterior_ca``, whose numbers
    this function reproduces (asserted in
    ``tests/reliability/test_observation_model.py``).

    ``r_cond`` is the covariance *given that a usable measurement arrived*. It is
    never scaled by availability here; that is the whole point of the branch
    form, and it is why this path needs no ``r_miss`` endpoint constant.

    ``innovation_floor`` adds ``floor·I`` to the matrix being inverted and ONLY to
    that matrix, exactly as ``INNOVATION_COV_FLOOR_PX2`` does in the runtime (the
    ``K R K'`` term keeps the unfloored ``r_cond``). Pass the runtime constant to
    reproduce shipped numbers bit-for-bit; leave it at zero for clean algebra.
    """

    p = _finite_probability(p_use, field_name="p_use")
    h = _rect(jacobian, "jacobian", rows=2)
    n = len(h[0])
    prior = _square(cov_prior, "cov_prior", dim=n)
    _validate_symmetric_pd(prior, "cov_prior")
    cond = _matrix_2x2(r_cond, "r_cond")
    _validate_spd(cond, "r_cond")

    floor = _finite_float(innovation_floor, field_name="innovation_floor")
    if floor < 0.0:
        raise ContractValidationError(f"innovation_floor must be >= 0, got {floor}")
    prior_sym = _sym(prior)
    innovation = _symmetrize(_mat_add(_mat2_from_congruence(h, prior_sym), cond))
    innovation = ((innovation[0][0] + floor, innovation[0][1]), (innovation[1][0], innovation[1][1] + floor))
    _validate_spd(innovation, "innovation covariance H P H' + r_cond")

    # K = P- H' S^-1  (n x 2)
    ph_t = _matmul(prior_sym, _transpose(h))
    gain = _matmul(ph_t, _mat_inv_2x2(innovation))
    identity = _eye(n)
    a = _matsub(identity, _matmul(gain, h))
    cov_hit = _sym(
        _matadd(
            _matmul(_matmul(a, prior_sym), _transpose(a)),
            _matmul(_matmul(gain, cond), _transpose(gain)),
        )
    )
    cov_expected = _sym(_matadd(_matscale(p, cov_hit), _matscale(1.0 - p, prior_sym)))
    return BranchPosterior(
        cov_expected=cov_expected,
        cov_hit=cov_hit,
        cov_miss=prior_sym,
        innovation_cov=innovation,
        p_use=p,
    )


def kalman_posterior(
    cov_prior: Sequence[Sequence[float]],
    jacobian: Sequence[Sequence[float]],
    r_effective: Sequence[Sequence[float]],
) -> Matrix:
    """One deterministic Joseph-form Kalman posterior with covariance ``r_effective``.

    This is what every single-``R`` planner interface actually computes. It equals
    the hit branch of :func:`expected_posterior_branch`; the difference between the
    two models is entirely in *which* ``R`` is fed here and whether a miss branch
    exists at all.
    """

    return expected_posterior_branch(cov_prior, jacobian, r_effective, 1.0).cov_hit


def scaled_covariance_baseline(
    r_cond: Sequence[Sequence[float]],
    p_use: float,
    *,
    p_floor: float = 1.0e-4,
) -> Matrix2x2:
    """``R / p_use`` — the classic expected-information scaling, as a baseline.

    Motivated by ``E[I] ≈ p · I_det`` on the *information* side, which is exact:
    the expected information matrix of a Bernoulli measurement really is
    ``p H' R^-1 H``. What is NOT exact is inverting that back into a covariance
    and pretending a single Gaussian update with ``R/p`` reproduces the expected
    posterior — Kalman updates are nonlinear in ``R``, so the two disagree, and
    :func:`equivalent_isotropic_covariance` measures by how much.

    Note this is exactly the ``R_miss -> inf`` limit of the deployed precision
    blend (``reliability.single_camera_adapter.precision_blend_covariance``,
    itself an exact mirror of
    ``planning.core.casadi_efe._blend_observation_covariance_ca``), so the
    critique is not aimed at a straw man: it is the same family as the shipped
    mapping, with the miss endpoint sent to infinity.
    """

    p = _finite_probability(p_use, field_name="p_use")
    cond = _matrix_2x2(r_cond, "r_cond")
    _validate_spd(cond, "r_cond")
    floor = _finite_float(p_floor, field_name="p_floor")
    if not 0.0 < floor <= 1.0:
        raise ContractValidationError("p_floor must be in (0, 1]")
    scale = 1.0 / max(p, floor)
    return (
        (cond[0][0] * scale, cond[0][1] * scale),
        (cond[1][0] * scale, cond[1][1] * scale),
    )


# --------------------------------------------------------------------------- #
# 4. Equivalent single-R for interfaces that demand one                        #
# --------------------------------------------------------------------------- #


def posterior_criterion(cov: Sequence[Sequence[float]], criterion: str = "trace") -> float:
    """Scalar uncertainty readout ``Φ(P)`` used to compare belief propagations.

    ``trace`` (total variance), ``position_trace`` (the leading 2x2 block only —
    the quantity a navigation planner cares about), ``logdet`` (proportional to
    differential entropy), ``max_eig`` (worst-case direction, via deterministic
    power iteration).
    """

    matrix = _square(cov, "cov", dim=len(tuple(cov)))
    if criterion == "trace":
        return float(sum(matrix[i][i] for i in range(len(matrix))))
    if criterion == "position_trace":
        if len(matrix) < 2:
            raise ContractValidationError("position_trace needs a 2x2 leading block")
        return float(matrix[0][0] + matrix[1][1])
    if criterion == "logdet":
        return _logdet_pd(matrix)
    if criterion == "max_eig":
        return _max_eigenvalue_sym(matrix)
    raise ContractValidationError(f"criterion must be one of {CRITERIA}, got {criterion!r}")


@dataclass(frozen=True)
class EquivalentCovariance:
    """Result of matching a single-``R`` update to a branched expected posterior."""

    sigma2: float            #: isotropic variance, measurement units squared
    criterion: str
    target: float            #: Φ(E[P+])
    achieved: float          #: Φ(P+(σ_eff² I))
    reached: bool            #: False when no finite R reproduces the target
    iterations: int

    @property
    def covariance(self) -> Matrix2x2:
        return ((self.sigma2, 0.0), (0.0, self.sigma2))


def equivalent_isotropic_covariance(
    cov_prior: Sequence[Sequence[float]],
    jacobian: Sequence[Sequence[float]],
    r_cond: Sequence[Sequence[float]],
    p_use: float,
    *,
    criterion: str = "position_trace",
    tolerance: float = 1.0e-10,
    max_iterations: int = 200,
) -> EquivalentCovariance:
    """Smallest-surprise bridge: the ``σ_eff² I`` whose ONE update matches ``E[P+]``.

    Solves ``Φ(P+(P-, H, σ² I)) = Φ(E[P+])`` for ``σ²`` by bisection. This is
    well posed because every criterion in :data:`CRITERIA` is monotone
    non-decreasing in ``σ²`` (more measurement noise cannot shrink a posterior in
    the Loewner order), and because the target is bracketed:
    ``Φ(P_hit) ≤ Φ(E[P+]) ≤ Φ(P-)`` with the upper end attained only at
    ``p_use = 0``, where the honest answer is "no finite R" —
    reported as ``reached=False`` rather than as a large number.

    The signature is the point of the function: ``σ_eff²`` depends on the prior
    ``P-`` and the Jacobian ``H``, not on position alone. Any interface that
    caches a single ``R_plan(s)`` per position is therefore making an
    approximation that this function quantifies.
    """

    if criterion not in CRITERIA:
        raise ContractValidationError(f"criterion must be one of {CRITERIA}, got {criterion!r}")
    branch = expected_posterior_branch(cov_prior, jacobian, r_cond, p_use)
    target = posterior_criterion(branch.cov_expected, criterion)
    tol = _finite_float(tolerance, field_name="tolerance")
    if tol <= 0.0:
        raise ContractValidationError("tolerance must be > 0")

    cond = _matrix_2x2(r_cond, "r_cond")
    scale = max(cond[0][0], cond[1][1], 1.0e-12)
    lo = scale * 10.0 ** (-_SIGMA2_SEARCH_DECADES / 2.0)
    hi = scale * 10.0 ** (_SIGMA2_SEARCH_DECADES / 2.0)

    def phi(sigma2: float) -> float:
        posterior = expected_posterior_branch(
            cov_prior, jacobian, ((sigma2, 0.0), (0.0, sigma2)), 1.0
        ).cov_hit
        return posterior_criterion(posterior, criterion)

    phi_hi = phi(hi)
    if phi_hi < target:
        # Target is above anything a finite isotropic R can produce: p_use == 0
        # (or numerically indistinguishable from it).
        return EquivalentCovariance(
            sigma2=hi,
            criterion=criterion,
            target=target,
            achieved=phi_hi,
            reached=False,
            iterations=0,
        )

    iterations = 0
    for iterations in range(1, int(max_iterations) + 1):
        mid = math.sqrt(lo * hi)  # geometric bisection: σ² spans many decades
        value = phi(mid)
        if abs(value - target) <= tol * max(1.0, abs(target)):
            lo = hi = mid
            break
        if value < target:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1.0 + 1.0e-12:
            break
    sigma2 = math.sqrt(lo * hi)
    return EquivalentCovariance(
        sigma2=sigma2,
        criterion=criterion,
        target=target,
        achieved=phi(sigma2),
        reached=True,
        iterations=iterations,
    )


# --------------------------------------------------------------------------- #
# 5. Robust conditional likelihoods (heavy tails are not variance)             #
# --------------------------------------------------------------------------- #


def student_t_scatter_from_covariance(
    covariance: Sequence[Sequence[float]], nu: float
) -> Matrix2x2:
    """Scatter ``Σ`` of a 2-D Student-t with ``nu`` dof and the given covariance.

    ``Cov = ν/(ν-2) Σ`` for ``ν > 2``, so ``Σ = (ν-2)/ν · Cov``. Use this when
    comparing a Gaussian against a Student-t at *matched second moment* —
    otherwise the heavy-tailed model wins simply by being wider, which says
    nothing about tail shape.
    """

    matrix = _matrix_2x2(covariance, "covariance")
    _validate_spd(matrix, "covariance")
    nu_f = _finite_float(nu, field_name="nu")
    if nu_f <= 2.0:
        raise ContractValidationError("nu must be > 2 for a Student-t covariance to exist")
    factor = (nu_f - 2.0) / nu_f
    return (
        (matrix[0][0] * factor, matrix[0][1] * factor),
        (matrix[1][0] * factor, matrix[1][1] * factor),
    )


def student_t_nll(
    residuals: Sequence[Sequence[float]],
    scatters: Sequence[Sequence[Sequence[float]]],
    nu: float,
) -> float:
    """Mean negative log-likelihood of 2-D Student-t residuals with scatter ``Σ_t``.

    ``-log p = -lgamma((ν+2)/2) + lgamma(ν/2) + log(νπ) + ½log|Σ| +
    ((ν+2)/2) log(1 + Δ/ν)``, ``Δ = e'Σ^-1e``. Directly comparable to
    :func:`reliability.conditional_covariance.matrix_nll` (the Gaussian case) —
    same units, same per-sample mean.
    """

    nu_f = _finite_float(nu, field_name="nu")
    if nu_f <= 0.0:
        raise ContractValidationError("nu must be > 0")
    pairs = _pairs(residuals, scatters)
    const = -math.lgamma((nu_f + 2.0) / 2.0) + math.lgamma(nu_f / 2.0) + math.log(nu_f * math.pi)
    total = 0.0
    for residual, scatter in pairs:
        maha = _quad_form_inverse_2x2(residual, scatter)
        total += (
            const
            + 0.5 * _log_det_2x2(scatter)
            + 0.5 * (nu_f + 2.0) * math.log1p(maha / nu_f)
        )
    return total / len(pairs)


def contaminated_gaussian_nll(
    residuals: Sequence[Sequence[float]],
    inlier_covariances: Sequence[Sequence[Sequence[float]]],
    outlier_covariances: Sequence[Sequence[Sequence[float]]],
    pi_outlier: float,
) -> float:
    """Mean NLL of a two-component ``(1-π)N(0,R_in) + π N(0,R_out)`` mixture.

    The model of record for "large residuals are a *different process*" — a wrong
    box, a stale frame, a partial occlusion — rather than the tail of one wide
    Gaussian. Fitting one Gaussian to contaminated residuals inflates ``R`` for
    every well-behaved detection too; a mixture keeps the inlier covariance sharp
    and prices the outliers separately.
    """

    pi = _finite_probability(pi_outlier, field_name="pi_outlier")
    inliers = _pairs(residuals, inlier_covariances)
    outliers = _pairs(residuals, outlier_covariances)
    total = 0.0
    for (residual, r_in), (_, r_out) in zip(inliers, outliers):
        log_in = _gaussian_logpdf_2d(residual, r_in)
        log_out = _gaussian_logpdf_2d(residual, r_out)
        if pi <= 0.0:
            total -= log_in
        elif pi >= 1.0:
            total -= log_out
        else:
            total -= _log_sum_exp(math.log1p(-pi) + log_in, math.log(pi) + log_out)
    return total / len(inliers)


def _gaussian_logpdf_2d(residual: tuple[float, float], covariance: Matrix2x2) -> float:
    maha = _quad_form_inverse_2x2(residual, covariance)
    return -0.5 * (maha + _log_det_2x2(covariance) + 2.0 * _LOG_TWO_PI)


def _log_sum_exp(a: float, b: float) -> float:
    hi, lo = (a, b) if a >= b else (b, a)
    return hi + math.log1p(math.exp(lo - hi))


# --------------------------------------------------------------------------- #
# Small dense linear algebra (stdlib only; measurement dim 2, state dim n)     #
# --------------------------------------------------------------------------- #


def _rect(value: Sequence[Sequence[float]], name: str, *, rows: int) -> Matrix:
    matrix = tuple(tuple(_finite_float(v, field_name=name) for v in row) for row in value)
    if len(matrix) != rows:
        raise ContractValidationError(f"{name} must have {rows} rows, got {len(matrix)}")
    width = len(matrix[0]) if matrix else 0
    if width == 0 or any(len(row) != width for row in matrix):
        raise ContractValidationError(f"{name} must be a non-empty rectangular matrix")
    return matrix


def _square(value: Sequence[Sequence[float]], name: str, *, dim: int) -> Matrix:
    matrix = tuple(tuple(_finite_float(v, field_name=name) for v in row) for row in value)
    if len(matrix) != dim or any(len(row) != dim for row in matrix):
        raise ContractValidationError(f"{name} must be {dim}x{dim}")
    return matrix


def _vector(value: Sequence[float], name: str, *, dim: int) -> tuple[float, ...]:
    vector = tuple(_finite_float(v, field_name=name) for v in value)
    if len(vector) != dim:
        raise ContractValidationError(f"{name} must have {dim} entries, got {len(vector)}")
    return vector


def _transpose(matrix: Matrix) -> Matrix:
    return tuple(zip(*matrix))


def _matmul(a: Matrix, b: Matrix) -> Matrix:
    if len(a[0]) != len(b):
        raise ContractValidationError("matrix shapes do not conform")
    b_t = _transpose(b)
    return tuple(
        tuple(sum(x * y for x, y in zip(row, column)) for column in b_t) for row in a
    )


def _matadd(a: Matrix, b: Matrix) -> Matrix:
    return tuple(tuple(x + y for x, y in zip(ra, rb)) for ra, rb in zip(a, b))


def _matsub(a: Matrix, b: Matrix) -> Matrix:
    return tuple(tuple(x - y for x, y in zip(ra, rb)) for ra, rb in zip(a, b))


def _matscale(scalar: float, matrix: Matrix) -> Matrix:
    return tuple(tuple(scalar * v for v in row) for row in matrix)


def _mat_vec_rect(matrix: Matrix, vector: Sequence[float]) -> tuple[float, ...]:
    return tuple(sum(v * x for v, x in zip(row, vector)) for row in matrix)


def _eye(n: int) -> Matrix:
    return tuple(tuple(1.0 if i == j else 0.0 for j in range(n)) for i in range(n))


def _sym(matrix: Matrix) -> Matrix:
    transposed = _transpose(matrix)
    return tuple(
        tuple(0.5 * (a + b) for a, b in zip(ra, rb)) for ra, rb in zip(matrix, transposed)
    )


def _mat2_from_congruence(h: Matrix, p: Matrix) -> Matrix2x2:
    """``H P H'`` for a 2-row ``H`` — returned as a plain 2x2 tuple."""

    product = _matmul(_matmul(h, p), _transpose(h))
    return (
        (float(product[0][0]), float(product[0][1])),
        (float(product[1][0]), float(product[1][1])),
    )


def _cholesky(matrix: Matrix, name: str) -> list[list[float]]:
    n = len(matrix)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            total = matrix[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if total <= 0.0:
                    raise ContractValidationError(f"{name} must be positive definite")
                lower[i][j] = math.sqrt(total)
            else:
                lower[i][j] = total / lower[j][j]
    return lower


def _validate_symmetric_pd(matrix: Matrix, name: str) -> None:
    n = len(matrix)
    scale = max(1.0, max(abs(v) for row in matrix for v in row))
    for i in range(n):
        for j in range(i):
            if abs(matrix[i][j] - matrix[j][i]) > 1.0e-9 * scale:
                raise ContractValidationError(f"{name} must be symmetric")
    _cholesky(_sym(matrix), name)


def _logdet_pd(matrix: Matrix) -> float:
    lower = _cholesky(_sym(matrix), "cov")
    return 2.0 * sum(math.log(lower[i][i]) for i in range(len(matrix)))


def _max_eigenvalue_sym(
    matrix: Matrix, *, iterations: int = 500, tolerance: float = 1.0e-14
) -> float:
    """Largest eigenvalue of a symmetric PSD matrix by shifted power iteration.

    The shift ``trace`` keeps every eigenvalue of ``A + trace·I`` positive, so the
    iteration converges to the *largest* eigenvalue rather than the largest in
    magnitude. The start vector ``(1, 1/2, 1/3, ...)`` is deterministic and, being
    non-uniform, is not orthogonal to the dominant eigenvector for any matrix this
    module sees (state dimension <= 3 in practice).
    """

    n = len(matrix)
    if n == 1:
        return float(matrix[0][0])
    shift = sum(matrix[i][i] for i in range(n))
    shifted = tuple(
        tuple(matrix[i][j] + (shift if i == j else 0.0) for j in range(n)) for i in range(n)
    )
    vector = [1.0 / (i + 1) for i in range(n)]
    norm = math.sqrt(sum(v * v for v in vector))
    vector = [v / norm for v in vector]
    value = 0.0
    for _ in range(iterations):
        product = [sum(shifted[i][j] * vector[j] for j in range(n)) for i in range(n)]
        norm = math.sqrt(sum(v * v for v in product))
        if norm <= 0.0:
            return 0.0
        product = [v / norm for v in product]
        new_value = sum(
            product[i] * sum(shifted[i][j] * product[j] for j in range(n)) for i in range(n)
        )
        if abs(new_value - value) <= tolerance * max(1.0, abs(new_value)):
            value = new_value
            vector = product
            break
        value, vector = new_value, product
    return float(value - shift)


def _is_psd_2x2(matrix: Matrix2x2) -> bool:
    a, b = matrix[0]
    _, d = matrix[1]
    scale = max(1.0, abs(a), abs(b), abs(d))
    min_eigenvalue = 0.5 * (a + d) - math.hypot(0.5 * (a - d), b)
    return min_eigenvalue >= -1.0e-12 * scale


def _pairs(
    residuals: Sequence[Sequence[float]],
    matrices: Sequence[Sequence[Sequence[float]]],
) -> list[tuple[tuple[float, float], Matrix2x2]]:
    if len(residuals) != len(matrices):
        raise ContractValidationError("residuals and matrices must have equal length")
    if not residuals:
        raise ContractValidationError("residuals must be non-empty")
    out: list[tuple[tuple[float, float], Matrix2x2]] = []
    for index, (residual, matrix) in enumerate(zip(residuals, matrices)):
        pair = tuple(_finite_float(v, field_name=f"residuals[{index}]") for v in residual)
        if len(pair) != 2:
            raise ContractValidationError(f"residuals[{index}] must have 2 entries")
        validated = _matrix_2x2(matrix, f"matrices[{index}]")
        _validate_spd(validated, f"matrices[{index}]")
        out.append(((pair[0], pair[1]), validated))
    return out
