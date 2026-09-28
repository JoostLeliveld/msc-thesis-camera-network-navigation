"""Contract + parity tests for the factorized observation model.

The load-bearing tests here are the two parity ones: the stdlib branch posterior
must reproduce the CasADi runtime expression, and the stdlib precision blend must
reproduce the deployed one. Everything else in this module is only trustworthy
offline if those hold.
"""

from __future__ import annotations

import math

import pytest

from reliability.conditional_covariance import matrix_nll
from reliability.contracts import ContractValidationError
from reliability.single_camera_adapter import precision_blend_covariance
from reliability.observation_model import (
    CRITERIA,
    calibration_covariance,
    contaminated_gaussian_nll,
    equivalent_isotropic_covariance,
    expected_posterior_branch,
    innovation_covariance,
    kalman_posterior,
    posterior_criterion,
    scaled_covariance_baseline,
    state_projection_covariance,
    student_t_nll,
    student_t_scatter_from_covariance,
    time_sync_covariance,
    usable_probability,
)


def _flat(matrix) -> list[float]:
    """pytest.approx does not nest, so matrices are compared flattened."""

    return [float(value) for row in matrix for value in row]


H = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
PRIOR = (
    (900.0, 225.0, 0.0),
    (225.0, 900.0, 0.0),
    (0.0, 0.0, 0.0076),
)
R_COND = ((6.25, 0.0), (0.0, 6.25))


# ------------------------------------------------------------------ availability


def test_usable_probability_is_the_product_of_the_two_labels():
    assert usable_probability(0.8, 0.5) == pytest.approx(0.4)
    assert usable_probability(1.0, 1.0) == 1.0
    assert usable_probability(0.0, 1.0) == 0.0


def test_usable_probability_rejects_out_of_range():
    with pytest.raises(ContractValidationError):
        usable_probability(1.2, 0.5)


# ------------------------------------------------------- innovation decomposition


def test_state_projection_picks_the_position_block():
    assert state_projection_covariance(H, PRIOR) == ((900.0, 225.0), (225.0, 900.0))


def test_time_sync_covariance_is_rank_one_along_the_motion_direction():
    # 1 m/s along +x, 20 ms timing uncertainty -> 0.02 m std along x only.
    term = time_sync_covariance(H, (1.0, 0.0, 0.0), 0.02)
    assert term[0][0] == pytest.approx(0.02**2)
    assert term[1][1] == pytest.approx(0.0)
    determinant = term[0][0] * term[1][1] - term[0][1] * term[1][0]
    assert determinant == pytest.approx(0.0, abs=1e-18)


def test_time_sync_covariance_grows_quadratically_with_speed():
    slow = time_sync_covariance(H, (0.5, 0.0, 0.0), 0.02)
    fast = time_sync_covariance(H, (1.0, 0.0, 0.0), 0.02)
    assert fast[0][0] == pytest.approx(4.0 * slow[0][0])


def test_time_sync_covariance_rejects_negative_sigma():
    with pytest.raises(ContractValidationError):
        time_sync_covariance(H, (1.0, 0.0, 0.0), -0.01)


def test_calibration_covariance_propagates_a_scalar_yaw_error():
    # A 1 deg yaw error at 10 m lever arm -> 0.1745 m lateral.
    lever = 10.0
    sigma_deg = 1.0
    h_c = ((0.0,), (lever,))
    sigma_c = ((math.radians(sigma_deg) ** 2,),)
    term = calibration_covariance(h_c, sigma_c)
    assert math.sqrt(term[1][1]) == pytest.approx(lever * math.radians(sigma_deg))
    assert term[0][0] == pytest.approx(0.0)


def test_innovation_covariance_sums_terms_and_accepts_singular_contributions():
    total = innovation_covariance(
        state_projection_covariance(H, PRIOR),
        R_COND,
        time_sync_covariance(H, (1.0, 0.0, 0.0), 0.02),
    )
    assert total[0][0] == pytest.approx(900.0 + 6.25 + 0.0004)
    assert total[1][1] == pytest.approx(900.0 + 6.25)


def test_innovation_covariance_rejects_a_non_psd_term():
    with pytest.raises(ContractValidationError):
        innovation_covariance(R_COND, ((-1.0, 0.0), (0.0, 1.0)))


# ------------------------------------------------------------- branch propagation


def test_branch_endpoints_are_the_two_deterministic_cases():
    hit = expected_posterior_branch(PRIOR, H, R_COND, 1.0)
    miss = expected_posterior_branch(PRIOR, H, R_COND, 0.0)
    assert _flat(hit.cov_expected) == pytest.approx(_flat(hit.cov_hit))
    assert _flat(miss.cov_expected) == pytest.approx(_flat(PRIOR))


def test_branch_expectation_is_linear_in_p_use():
    half = expected_posterior_branch(PRIOR, H, R_COND, 0.5)
    for i in range(3):
        for j in range(3):
            expected = 0.5 * half.cov_hit[i][j] + 0.5 * half.cov_miss[i][j]
            assert half.cov_expected[i][j] == pytest.approx(expected)


def test_branch_posterior_is_psd_across_the_whole_p_use_range():
    for k in range(0, 101):
        branch = expected_posterior_branch(PRIOR, H, R_COND, k / 100.0)
        # Cholesky inside posterior_criterion raises if the matrix is not PD.
        assert posterior_criterion(branch.cov_expected, "logdet") < math.inf
        assert posterior_criterion(branch.cov_expected, "max_eig") > 0.0


def test_kalman_posterior_equals_the_hit_branch():
    branch = expected_posterior_branch(PRIOR, H, R_COND, 0.3)
    assert _flat(kalman_posterior(PRIOR, H, R_COND)) == pytest.approx(_flat(branch.cov_hit))


def test_branch_posterior_matches_the_casadi_runtime_expression():
    """The stdlib twin must reproduce planning.core.casadi_efe.hit_miss_posterior_ca."""

    ca = pytest.importorskip("casadi")
    from planning.core.casadi_efe import INNOVATION_COV_FLOOR_PX2, hit_miss_posterior_ca

    # Passing the runtime's innovation floor makes the two expressions identical
    # rather than merely close, so any real divergence would show up.
    for p_use in (0.0, 0.05, 0.37, 0.5, 0.91, 1.0):
        p_mix, p_hit, _, sigma = hit_miss_posterior_ca(
            ca.DM(PRIOR), ca.DM(H), ca.DM(R_COND), ca.DM(p_use)
        )
        branch = expected_posterior_branch(
            PRIOR, H, R_COND, p_use, innovation_floor=INNOVATION_COV_FLOOR_PX2
        )
        assert _flat(branch.cov_expected) == pytest.approx(
            _flat(ca.DM(p_mix).full()), rel=1e-12, abs=1e-12
        )
        assert _flat(branch.cov_hit) == pytest.approx(
            _flat(ca.DM(p_hit).full()), rel=1e-12, abs=1e-12
        )
        assert _flat(branch.innovation_cov) == pytest.approx(
            _flat(ca.DM(sigma).full()), rel=1e-12, abs=1e-12
        )


# ------------------------------------------------------- the single-R alternatives


def test_deployed_blend_mirror_matches_the_casadi_mapping():
    """reliability.single_camera_adapter is the stdlib mirror of the shipped blend."""

    ca = pytest.importorskip("casadi")
    import numpy as np

    from planning.core.casadi_efe import CasadiEfeParams, _blend_observation_covariance_ca

    r_visible_uv, r_miss_uv = 2.5, 120.0
    params = CasadiEfeParams(
        R_visible=ca.DM(np.diag([r_visible_uv**2, r_visible_uv**2])),
        R_miss=ca.DM(np.diag([r_miss_uv**2, r_miss_uv**2])),
        control_weight=1.0,
        risk_scale=1.0,
        ambiguity_scale=1.0,
        discount_gamma=1.0,
        process_noise_xy=0.01,
        process_noise_theta=0.01,
        visibility_sigma_kappa=1.0,
        goal_prior_u_std_start=1.0,
        goal_prior_v_std_start=1.0,
        goal_prior_u_std_final=1.0,
        goal_prior_v_std_final=1.0,
        goal_tightening_power=1.0,
        goal_progress_n_steps=1,
        use_belief_nogo_cost=False,
        time_horizon=1,
        dt=0.1,
        Du=2,
    )
    for p_use in (1e-4, 0.1, 0.5, 0.9, 1.0 - 1e-4):
        runtime = ca.DM(_blend_observation_covariance_ca(ca.DM(p_use), params))
        mirror = precision_blend_covariance(
            p_use, r_visible_uv=r_visible_uv, r_miss_uv=r_miss_uv
        )
        for i in range(2):
            assert mirror[i][i] == pytest.approx(float(runtime[i, i]), rel=1e-12)


def test_scaled_baseline_is_the_infinite_miss_endpoint_limit_of_the_blend():
    """R/p is not a straw man: it is the deployed blend with R_miss -> inf."""

    r_visible_uv = 2.5
    r_visible = ((r_visible_uv**2, 0.0), (0.0, r_visible_uv**2))
    for p_use in (0.05, 0.25, 0.5, 0.8):
        scaled = scaled_covariance_baseline(r_visible, p_use)
        blended = precision_blend_covariance(
            p_use, r_visible_uv=r_visible_uv, r_miss_uv=1.0e6
        )
        for i in range(2):
            assert blended[i][i] == pytest.approx(scaled[i][i], rel=1e-6)


def test_scaled_baseline_inflates_and_floors_tiny_p():
    doubled = scaled_covariance_baseline(R_COND, 0.5)
    assert doubled[0][0] == pytest.approx(2.0 * R_COND[0][0])
    floored = scaled_covariance_baseline(R_COND, 0.0, p_floor=1e-3)
    assert floored[0][0] == pytest.approx(1000.0 * R_COND[0][0])


# --------------------------------------------------------- equivalent covariance


@pytest.mark.parametrize("criterion", CRITERIA)
def test_equivalent_covariance_reproduces_the_branch_criterion(criterion):
    result = equivalent_isotropic_covariance(PRIOR, H, R_COND, 0.4, criterion=criterion)
    assert result.reached
    assert result.achieved == pytest.approx(result.target, rel=1e-6)
    posterior = kalman_posterior(PRIOR, H, result.covariance)
    assert posterior_criterion(posterior, criterion) == pytest.approx(result.target, rel=1e-6)


def test_equivalent_covariance_recovers_r_cond_exactly_at_p_use_one():
    result = equivalent_isotropic_covariance(PRIOR, H, R_COND, 1.0)
    assert result.sigma2 == pytest.approx(R_COND[0][0], rel=1e-6)


def test_equivalent_covariance_is_monotone_in_p_use():
    sigmas = [
        equivalent_isotropic_covariance(PRIOR, H, R_COND, p).sigma2
        for p in (0.2, 0.4, 0.6, 0.8, 1.0)
    ]
    assert sigmas == sorted(sigmas, reverse=True)


def test_equivalent_covariance_reports_unreachable_at_zero_availability():
    result = equivalent_isotropic_covariance(PRIOR, H, R_COND, 0.0)
    assert not result.reached


def test_equivalent_covariance_depends_on_the_prior_not_only_on_p_use():
    """The headline property: R_eff is R_eff(s, P-, H), not R_eff(s)."""

    tight = tuple(tuple(0.01 * v for v in row) for row in PRIOR)
    loose = tuple(tuple(4.0 * v for v in row) for row in PRIOR)
    a = equivalent_isotropic_covariance(tight, H, R_COND, 0.4).sigma2
    b = equivalent_isotropic_covariance(loose, H, R_COND, 0.4).sigma2
    assert a != pytest.approx(b, rel=1e-3)


def test_equivalent_covariance_rejects_an_unknown_criterion():
    with pytest.raises(ContractValidationError):
        equivalent_isotropic_covariance(PRIOR, H, R_COND, 0.5, criterion="variance")


# ------------------------------------------------------------ robust likelihoods


def test_student_t_scatter_matches_the_requested_covariance():
    nu = 6.0
    scatter = student_t_scatter_from_covariance(R_COND, nu)
    assert scatter[0][0] * nu / (nu - 2.0) == pytest.approx(R_COND[0][0])


def test_student_t_scatter_requires_more_than_two_dof():
    with pytest.raises(ContractValidationError):
        student_t_scatter_from_covariance(R_COND, 2.0)


def test_student_t_approaches_the_gaussian_as_nu_grows():
    residuals = [(0.4, -0.2), (1.1, 0.3), (-0.7, 0.9)]
    covariances = [R_COND] * len(residuals)
    gaussian = matrix_nll(residuals, covariances)
    scatters = [student_t_scatter_from_covariance(R_COND, 5000.0)] * len(residuals)
    heavy = student_t_nll(residuals, scatters, 5000.0)
    assert heavy == pytest.approx(gaussian, abs=5e-3)


def test_student_t_prices_an_outlier_more_cheaply_than_a_gaussian():
    outlier = [(30.0, 30.0)]
    gaussian = matrix_nll(outlier, [R_COND])
    heavy = student_t_nll(outlier, [student_t_scatter_from_covariance(R_COND, 4.0)], 4.0)
    assert heavy < gaussian


def test_contaminated_mixture_beats_one_wide_gaussian_on_contaminated_data():
    inlier = ((0.25, 0.0), (0.0, 0.25))
    outlier = ((100.0, 0.0), (0.0, 100.0))
    residuals = [(0.1, -0.1), (-0.2, 0.15), (0.05, 0.2), (12.0, -9.0)]
    mixture = contaminated_gaussian_nll(
        residuals, [inlier] * 4, [outlier] * 4, pi_outlier=0.25
    )
    # A single Gaussian must widen to cover the outlier, which costs every inlier.
    wide = ((25.0, 0.0), (0.0, 25.0))
    single = matrix_nll(residuals, [wide] * 4)
    assert mixture < single


def test_contaminated_mixture_endpoints_reduce_to_single_components():
    inlier = ((0.25, 0.0), (0.0, 0.25))
    outlier = ((100.0, 0.0), (0.0, 100.0))
    residuals = [(0.1, -0.1), (0.3, 0.2)]
    assert contaminated_gaussian_nll(
        residuals, [inlier] * 2, [outlier] * 2, 0.0
    ) == pytest.approx(matrix_nll(residuals, [inlier] * 2))
    assert contaminated_gaussian_nll(
        residuals, [inlier] * 2, [outlier] * 2, 1.0
    ) == pytest.approx(matrix_nll(residuals, [outlier] * 2))
