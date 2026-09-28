from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "reliability"))

from reliability.contracts import CameraQuality, ContractValidationError  # noqa: E402
from reliability.fusion import (  # noqa: E402
    FuseOrSelectDecision,
    MapObservation,
    expected_information_gain,
    fuse_or_select,
    independent_measurement_fusion_2d,
    joseph_update_2d,
    robust_reweight_covariance,
    select_information_best,
    sequential_kalman_update_2d,
)


def _quality(camera_id: str) -> CameraQuality:
    return CameraQuality(
        camera_id=camera_id,
        p_available=0.9,
        conditional_cov_uv=((4.0, 0.0), (0.0, 4.0)),
        association_confidence=0.9,
        epistemic_score=0.0,
        source_model="test",
    )


def _map_obs(
    camera_id: str,
    xy: tuple[float, float],
    cov: tuple[tuple[float, float], tuple[float, float]] = ((0.05, 0.0), (0.0, 0.05)),
) -> MapObservation:
    return MapObservation(
        camera_id=camera_id,
        timestamp_s=1.0,
        xy_m=xy,
        covariance_m2=cov,
        quality=_quality(camera_id),
    )


def _assert_psd(cov) -> None:
    a, b = cov[0]
    c, d = cov[1]
    assert abs(b - c) < 1.0e-12
    assert a > 0.0 and d > 0.0
    assert a * d - b * c > 0.0


class TestJosephUpdate:
    def test_matches_standard_update_well_conditioned(self):
        prior_mean = (0.0, 0.0)
        prior_cov = ((1.0, 0.1), (0.1, 2.0))
        obs = _map_obs("camera_A", (1.0, 0.5), ((0.5, 0.0), (0.0, 0.5)))

        mean_j, cov_j, nis_j = joseph_update_2d(prior_mean, prior_cov, obs)
        ref = sequential_kalman_update_2d(prior_mean, prior_cov, [obs], nis_gate=None)

        assert mean_j == pytest.approx(ref.mean_xy, abs=1.0e-12)
        for i in range(2):
            for j in range(2):
                assert cov_j[i][j] == pytest.approx(ref.covariance_m2[i][j], abs=1.0e-12)
        assert nis_j == pytest.approx(ref.nis_by_camera["camera_A"], abs=1.0e-12)

    def test_psd_retained_at_extreme_r_ratio(self):
        prior_mean = (0.0, 0.0)
        prior_cov = ((1.0, 0.0), (0.0, 1.0))
        obs = _map_obs("camera_A", (0.3, -0.2), ((1.0e-6, 0.0), (0.0, 1.0e6)))
        mean, cov, nis = joseph_update_2d(prior_mean, prior_cov, obs)
        _assert_psd(cov)
        assert math.isfinite(nis)
        # near-exact x measurement pulls x variance close to R_x, y stays near prior
        assert cov[0][0] < 1.0e-5
        assert cov[1][1] == pytest.approx(1.0, rel=1.0e-3)
        assert mean[0] == pytest.approx(0.3, abs=1.0e-4)

    def test_nis_hand_checked(self):
        # nu = (1, 0), S = P + R = 2 I  =>  d2 = 0.5
        obs = _map_obs("camera_A", (1.0, 0.0), ((1.0, 0.0), (0.0, 1.0)))
        _, _, nis = joseph_update_2d((0.0, 0.0), ((1.0, 0.0), (0.0, 1.0)), obs)
        assert nis == pytest.approx(0.5)

    def test_rejects_invalid_prior_cov(self):
        obs = _map_obs("camera_A", (1.0, 0.0))
        with pytest.raises(ContractValidationError):
            joseph_update_2d((0.0, 0.0), ((1.0, 5.0), (5.0, 1.0)), obs)


class TestIndependentMeasurementFusion:
    def test_equal_covariances_average_measurements_and_halve_covariance(self):
        mean, covariance = independent_measurement_fusion_2d([
            _map_obs("camera_A", (0.0, 2.0), ((0.04, 0.0), (0.0, 0.10))),
            _map_obs("camera_B", (2.0, 4.0), ((0.04, 0.0), (0.0, 0.10))),
        ])
        assert mean == pytest.approx((1.0, 3.0))
        assert covariance[0] == pytest.approx((0.02, 0.0))
        assert covariance[1] == pytest.approx((0.0, 0.05))

    def test_more_precise_measurement_gets_more_weight_without_prior_bias(self):
        mean, covariance = independent_measurement_fusion_2d([
            _map_obs("camera_A", (0.0, 0.0), ((0.01, 0.0), (0.0, 0.01))),
            _map_obs("camera_B", (10.0, 10.0), ((1.0, 0.0), (0.0, 1.0))),
        ])
        assert mean == pytest.approx((10.0 / 101.0, 10.0 / 101.0))
        assert covariance[0][0] == pytest.approx(1.0 / 101.0)
        assert covariance[1][1] == pytest.approx(1.0 / 101.0)

    def test_requires_an_observation(self):
        with pytest.raises(ContractValidationError):
            independent_measurement_fusion_2d([])


class TestRobustReweight:
    R = ((2.0, 0.0), (0.0, 2.0))

    def test_nis_zero_leaves_r_unchanged(self):
        # uncapped Student-t weight would be (dof+2)/dof = 1.5 > 1 and
        # SHRINK R; the implementation caps w at 1 (monotone inflating only)
        out = robust_reweight_covariance(self.R, 0.0, dof=4.0, w_min=0.1)
        assert out[0][0] == pytest.approx(2.0)
        assert out[1][1] == pytest.approx(2.0)

    def test_hand_checked_inflation(self):
        # dof=4, nis=6: w = 6/10 = 0.6  =>  R / 0.6
        out = robust_reweight_covariance(self.R, 6.0, dof=4.0, w_min=0.1)
        assert out[0][0] == pytest.approx(2.0 / 0.6)
        assert out[1][1] == pytest.approx(2.0 / 0.6)
        # dof=4, nis=10: w = 6/14  =>  R * 14/6
        out2 = robust_reweight_covariance(self.R, 10.0, dof=4.0, w_min=0.1)
        assert out2[0][0] == pytest.approx(2.0 * 14.0 / 6.0)

    def test_large_nis_hits_w_min_floor(self):
        out = robust_reweight_covariance(self.R, 1.0e9, dof=4.0, w_min=0.1)
        assert out[0][0] == pytest.approx(2.0 / 0.1)
        assert out[1][1] == pytest.approx(2.0 / 0.1)

    def test_monotone_never_shrinks(self):
        for nis in (0.0, 0.5, 1.9, 2.0, 5.0, 100.0):
            out = robust_reweight_covariance(self.R, nis, dof=4.0, w_min=0.1)
            assert out[0][0] >= self.R[0][0] - 1.0e-12

    def test_input_validation(self):
        with pytest.raises(ContractValidationError):
            robust_reweight_covariance(self.R, -1.0)
        with pytest.raises(ContractValidationError):
            robust_reweight_covariance(self.R, float("nan"))
        with pytest.raises(ContractValidationError):
            robust_reweight_covariance(self.R, 1.0, dof=0.0)
        with pytest.raises(ContractValidationError):
            robust_reweight_covariance(self.R, 1.0, w_min=0.0)
        with pytest.raises(ContractValidationError):
            robust_reweight_covariance(((1.0, 5.0), (5.0, 1.0)), 1.0)


class TestInformationGain:
    def test_hand_checked_logdet(self):
        # P = I, R = I  =>  P+ = 0.5 I, gain = log 1 - log 0.25 = log 4
        gain = expected_information_gain(((1.0, 0.0), (0.0, 1.0)), ((1.0, 0.0), (0.0, 1.0)))
        assert gain == pytest.approx(math.log(4.0))

    def test_positive_and_larger_for_smaller_r(self):
        prior = ((1.0, 0.0), (0.0, 1.0))
        gain_small = expected_information_gain(prior, ((0.01, 0.0), (0.0, 0.01)))
        gain_big = expected_information_gain(prior, ((10.0, 0.0), (0.0, 10.0)))
        assert gain_big > 0.0
        assert gain_small > gain_big


class TestSelectInformationBest:
    def test_picks_lower_r(self):
        prior = ((1.0, 0.0), (0.0, 1.0))
        precise = _map_obs("camera_A", (0.0, 0.0), ((0.01, 0.0), (0.0, 0.01)))
        coarse = _map_obs("camera_B", (0.0, 0.0), ((1.0, 0.0), (0.0, 1.0)))
        best = select_information_best([coarse, precise], prior)
        assert best is precise

    def test_geometry_beats_raw_precision(self):
        # prior is uncertain along x only; the camera precise along x wins
        # even though both have the same det(R)
        prior = ((4.0, 0.0), (0.0, 0.04))
        precise_x = _map_obs("camera_A", (0.0, 0.0), ((0.01, 0.0), (0.0, 10.0)))
        precise_y = _map_obs("camera_B", (0.0, 0.0), ((10.0, 0.0), (0.0, 0.01)))
        best = select_information_best([precise_y, precise_x], prior)
        assert best is precise_x

    def test_empty_returns_none(self):
        assert select_information_best([], ((1.0, 0.0), (0.0, 1.0))) is None


class TestFuseOrSelect:
    TAUS = {"camera_A": 0.9, "camera_B": 0.8, "camera_C": 0.7}
    HEALTHS = {"camera_A": 0.95, "camera_B": 0.9, "camera_C": 0.85}

    def _run(self, observations, taus=None, healths=None, **kw) -> FuseOrSelectDecision:
        params = dict(tau_min=0.5, h_min=0.5, disagreement_gate_m=0.5)
        params.update(kw)
        return fuse_or_select(
            observations,
            taus if taus is not None else self.TAUS,
            healths if healths is not None else self.HEALTHS,
            **params,
        )

    def test_all_consistent_fuses_all(self):
        obs = [
            _map_obs("camera_A", (1.0, 1.0)),
            _map_obs("camera_B", (1.1, 1.0)),
            _map_obs("camera_C", (1.0, 1.1)),
        ]
        decision = self._run(obs)
        assert decision.mode == "fuse"
        assert set(decision.selected) == {"camera_A", "camera_B", "camera_C"}
        assert decision.excluded == ()

    def test_one_outlier_excluded_with_reason(self):
        obs = [
            _map_obs("camera_A", (1.0, 1.0)),
            _map_obs("camera_B", (1.1, 1.0)),
            _map_obs("camera_C", (5.0, 5.0)),  # disagrees with both
        ]
        decision = self._run(obs)
        assert decision.mode == "fuse"
        assert set(decision.selected) == {"camera_A", "camera_B"}
        assert decision.excluded == ("camera_C",)
        assert "camera_C" in decision.reason

    def test_outlier_with_two_cameras_becomes_select(self):
        obs = [
            _map_obs("camera_A", (1.0, 1.0)),
            _map_obs("camera_B", (5.0, 5.0)),
        ]
        # tie on violation count (1 each) -> drop lowest tau (camera_B)
        decision = self._run(obs)
        assert decision.mode == "select"
        assert decision.selected == ("camera_A",)
        assert decision.excluded == ("camera_B",)

    def test_all_below_tau_min_gives_none(self):
        obs = [_map_obs("camera_A", (1.0, 1.0)), _map_obs("camera_B", (1.0, 1.0))]
        decision = self._run(obs, taus={"camera_A": 0.1, "camera_B": 0.2})
        assert decision.mode == "none"
        assert decision.selected == ()
        assert set(decision.excluded) == {"camera_A", "camera_B"}

    def test_low_health_excluded_leaving_single_select(self):
        obs = [_map_obs("camera_A", (1.0, 1.0)), _map_obs("camera_B", (1.0, 1.0))]
        decision = self._run(obs, healths={"camera_A": 0.95, "camera_B": 0.1})
        assert decision.mode == "select"
        assert decision.selected == ("camera_A",)
        assert decision.excluded == ("camera_B",)

    def test_missing_tau_treated_as_failing(self):
        obs = [_map_obs("camera_A", (1.0, 1.0)), _map_obs("camera_D", (1.0, 1.0))]
        decision = self._run(obs)  # camera_D not in TAUS/HEALTHS
        assert decision.mode == "select"
        assert decision.selected == ("camera_A",)
        assert "camera_D" in decision.excluded

    def test_gate_validation(self):
        obs = [_map_obs("camera_A", (1.0, 1.0))]
        with pytest.raises(ContractValidationError):
            self._run(obs, disagreement_gate_m=0.0)
        with pytest.raises(ContractValidationError):
            self._run(obs, tau_min=float("nan"))

    @pytest.mark.parametrize(
        "overrides",
        [
            {"tau_min": -0.01},
            {"tau_min": 1.01},
            {"h_min": -0.01},
            {"h_min": 1.01},
            {"h_min": float("inf")},
        ],
    )
    def test_probability_thresholds_out_of_range_rejected(self, overrides):
        obs = [_map_obs("camera_A", (1.0, 1.0))]
        with pytest.raises(ContractValidationError, match=r"\[0, 1\]"):
            self._run(obs, **overrides)

    @pytest.mark.parametrize(
        ("taus", "healths"),
        [
            ({"camera_A": -0.01}, {"camera_A": 0.9}),
            ({"camera_A": 1.01}, {"camera_A": 0.9}),
            ({"camera_A": float("nan")}, {"camera_A": 0.9}),
            ({"camera_A": 0.9}, {"camera_A": -0.01}),
            ({"camera_A": 0.9}, {"camera_A": 1.01}),
            ({"camera_A": 0.9}, {"camera_A": float("inf")}),
        ],
    )
    def test_invalid_observation_probabilities_rejected(self, taus, healths):
        obs = [_map_obs("camera_A", (1.0, 1.0))]
        with pytest.raises(ContractValidationError, match=r"\[0, 1\]"):
            self._run(obs, taus=taus, healths=healths)
