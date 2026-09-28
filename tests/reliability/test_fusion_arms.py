"""The four fusion rules the six-arm fusion comparison drives.

One test per property the arms are compared on, so a rule cannot be quietly changed into
another rule: F1 picks the most precise camera without a prior, F2 ignores covariance and
weights on geometry alone, F3 assumes independent Gaussian readings, and F4 solves the
camera batch jointly before it constructs one network-level Gaussian.
"""
from __future__ import annotations

import math

import pytest

from reliability.contracts import CameraQuality, ContractValidationError
from reliability.fusion import (
    MapObservation,
    distance_angle_weighted_fusion_2d,
    distance_angle_weights,
    independent_measurement_fusion_2d,
    joint_network_estimate_2d,
    select_smallest_covariance,
)

CAMERA_POSITIONS = {
    "camera_A": (-11.45, -9.45, 5.0),
    "camera_B": (-1.5, -9.72, 5.0),
    "camera_E": (11.45, -9.45, 5.0),
}


def _flat(matrix):
    """pytest.approx does not compare nested tuples, so 2x2 matrices are flattened."""
    return [matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]]


def obs(camera_id: str, x: float, y: float, sx: float, sy: float) -> MapObservation:
    return MapObservation(
        camera_id=camera_id, timestamp_s=1.0, xy_m=(x, y),
        covariance_m2=((sx ** 2, 0.0), (0.0, sy ** 2)),
        quality=CameraQuality(camera_id=camera_id))


@pytest.fixture
def three_readings():
    # sizes taken from one commissioned moment: 3.4x1.3, 0.9x0.5 and 2.9x1.2 cm
    return [obs("camera_A", 0.02, 0.0, 0.034, 0.013),
            obs("camera_B", 0.01, 0.01, 0.009, 0.005),
            obs("camera_E", 0.0, -0.01, 0.029, 0.012)]


class TestSingleBestCamera:
    def test_picks_the_smallest_ellipse(self, three_readings):
        assert select_smallest_covariance(three_readings).camera_id == "camera_B"

    def test_needs_no_prior(self, three_readings):
        # the point of this rule versus select_information_best: it is stateless
        assert select_smallest_covariance(reversed(three_readings)).camera_id == "camera_B"

    def test_no_observations_is_none_not_an_error(self):
        assert select_smallest_covariance([]) is None


class TestDistanceAngleHeuristic:
    def test_weights_sum_to_one(self, three_readings):
        w = distance_angle_weights(three_readings, CAMERA_POSITIONS)
        assert math.isclose(sum(w.values()), 1.0, rel_tol=1e-9)

    def test_nearest_camera_dominates(self, three_readings):
        w = distance_angle_weights(three_readings, CAMERA_POSITIONS)
        assert w["camera_B"] > w["camera_A"] and w["camera_B"] > w["camera_E"]

    def test_ignores_the_covariance_entirely(self, three_readings):
        """The baseline's defining property: geometry only, no measurement quality."""
        before = distance_angle_weights(three_readings, CAMERA_POSITIONS)
        widened = [obs(o.camera_id, o.xy_m[0], o.xy_m[1], 1.0, 1.0) if o.camera_id == "camera_B"
                   else o for o in three_readings]
        assert distance_angle_weights(widened, CAMERA_POSITIONS) == before

    def test_refuses_to_guess_a_camera_position(self, three_readings):
        with pytest.raises(ContractValidationError):
            distance_angle_weights(three_readings, {"camera_A": (0.0, 0.0, 5.0)})

    def test_reports_the_covariance_of_its_own_mean(self, three_readings):
        _mean, cov = distance_angle_weighted_fusion_2d(three_readings, CAMERA_POSITIONS)
        assert cov[0][0] > 0.0 and cov[1][1] > 0.0


class TestJointNetworkEstimator:
    def test_one_camera_is_one_network_report(self, three_readings):
        single = three_readings[:1]
        mean, covariance = joint_network_estimate_2d(single)
        assert mean == single[0].xy_m
        assert _flat(covariance) == pytest.approx(_flat(single[0].covariance_m2), abs=1e-15)

    def test_identical_equal_quality_views_do_not_shrink_by_n(self):
        readings = [obs(camera_id, 1.0, 2.0, 0.1, 0.2)
                    for camera_id in ("camera_A", "camera_B", "camera_E")]
        mean, covariance = joint_network_estimate_2d(readings)
        assert mean == pytest.approx((1.0, 2.0))
        assert _flat(covariance) == pytest.approx([0.01, 0.0, 0.0, 0.04])

    def test_disagreement_widens_the_direction_of_disagreement(self):
        agreed = [obs(camera_id, 0.0, 0.0, 0.1, 0.1)
                  for camera_id in ("camera_A", "camera_B", "camera_E")]
        spread = [obs("camera_A", -0.1, 0.0, 0.1, 0.1),
                  obs("camera_B", 0.0, 0.0, 0.1, 0.1),
                  obs("camera_E", 0.1, 0.0, 0.1, 0.1)]
        covariance_agreed = joint_network_estimate_2d(agreed)[1]
        covariance_spread = joint_network_estimate_2d(spread)[1]
        assert covariance_spread[0][0] > covariance_agreed[0][0]
        assert covariance_spread[1][1] == pytest.approx(covariance_agreed[1][1])

    def test_robust_joint_solution_is_not_independent_pooling(self):
        readings = [obs("camera_A", 0.0, 0.0, 0.1, 0.1),
                    obs("camera_B", 0.0, 0.0, 0.1, 0.1),
                    obs("camera_E", 2.0, 0.0, 0.1, 0.1)]
        independent = independent_measurement_fusion_2d(readings)[0]
        joint = joint_network_estimate_2d(readings)[0]
        assert joint[0] < independent[0]

    def test_result_is_independent_of_message_order(self, three_readings):
        forward = joint_network_estimate_2d(three_readings)
        reverse = joint_network_estimate_2d(list(reversed(three_readings)))
        assert forward[0] == pytest.approx(reverse[0], abs=1e-12)
        assert _flat(forward[1]) == pytest.approx(_flat(reverse[1]), abs=1e-12)

    def test_no_observations_is_refused(self):
        with pytest.raises(ContractValidationError):
            joint_network_estimate_2d([])

    def test_invalid_huber_threshold_is_refused(self, three_readings):
        with pytest.raises(ContractValidationError):
            joint_network_estimate_2d(three_readings, huber_delta=0.0)


class TestCommissionedCovarianceProfile:
    """The profile that states R_pix = sigma_px^2 I and lets geometry do the rest."""

    def test_reads_sigma_from_the_frozen_artifact(self, tmp_path):
        from reliability.nodes.camera_manager_node import load_commissioned_sigma_px

        path = tmp_path / "calibration.json"
        path.write_text('{"calibration": {"sigma_px": 0.7642679454946852}}')
        assert load_commissioned_sigma_px(str(path)) == pytest.approx(0.76426794549)

    def test_refuses_a_nonsense_sigma(self, tmp_path):
        from reliability.nodes.camera_manager_node import load_commissioned_sigma_px

        path = tmp_path / "calibration.json"
        path.write_text('{"calibration": {"sigma_px": 0.0}}')
        with pytest.raises(ValueError):
            load_commissioned_sigma_px(str(path))

    def test_pixel_covariance_is_isotropic(self):
        from reliability.nodes.camera_manager_node import commissioned_pixel_covariance

        cov = commissioned_pixel_covariance(0.76)
        assert cov[0][0] == pytest.approx(0.5776) and cov[1][1] == pytest.approx(0.5776)
        assert cov[0][1] == 0.0 and cov[1][0] == 0.0

    def test_nothing_floors_or_widens_what_the_sensor_stated(self):
        """The commissioned covariance IS the claim under test, so it is reported as is.

        A floor here would let every arm pass the honesty check for the wrong reason.
        """
        from reliability.nodes.camera_manager_node import _fusion_report_covariance

        covariance = ((0.0001, -0.00002), (-0.00002, 0.0004))
        assert _fusion_report_covariance(covariance) == covariance

    def test_only_the_commissioned_shared_term_is_added(self):
        """The one permitted addition: the part the cameras get wrong TOGETHER."""
        from reliability.nodes.camera_manager_node import _fusion_report_covariance

        reported = _fusion_report_covariance(
            ((0.0001, -0.00002), (-0.00002, 0.0004)), common_mode_std_m=0.032
        )
        assert reported[0][0] == pytest.approx(0.0001 + 0.032**2)
        assert reported[1][1] == pytest.approx(0.0004 + 0.032**2)
        # cross terms untouched: a shared radial error is not a rotation of the ellipse
        assert reported[0][1] == pytest.approx(-0.00002)


class TestFixedOffsetObservationModel:
    def test_pushes_the_reading_away_from_the_camera(self):
        from reliability.nodes.camera_manager_node import offset_away_from_camera

        # camera at the origin, reading 4 m east: the robot's centre is further east
        moved = offset_away_from_camera((4.0, 0.0), (0.0, 0.0, 5.0), 0.30)
        assert moved == pytest.approx((4.30, 0.0))

    def test_direction_follows_the_bearing_not_the_axes(self):
        from reliability.nodes.camera_manager_node import offset_away_from_camera

        moved = offset_away_from_camera((3.0, 4.0), (0.0, 0.0, 5.0), 0.5)
        assert moved == pytest.approx((3.3, 4.4))

    def test_a_reading_on_top_of_the_camera_is_left_alone(self):
        from reliability.nodes.camera_manager_node import offset_away_from_camera

        assert offset_away_from_camera((1.0, 1.0), (1.0, 1.0, 5.0), 0.3) == (1.0, 1.0)


class TestOneGateFourRules:
    """The gate is shared method; only the rule differs between arms."""

    @staticmethod
    def _fresh():
        return [obs("camera_A", 0.02, 0.0, 0.034, 0.013),
                obs("camera_B", 0.01, 0.01, 0.009, 0.005),
                obs("camera_E", 0.0, -0.01, 0.029, 0.012)]

    def test_best_single_reports_only_the_camera_it_used(self):
        from reliability.nodes.camera_manager_node import (
            FUSION_RULE_BEST_SINGLE, _gated_fusion)

        result = _gated_fusion(self._fresh(), disagreement_gate_m=0.6,
                               rule=FUSION_RULE_BEST_SINGLE)
        assert result.accepted_camera_ids == ("camera_B",)

    def test_joint_network_has_a_distinct_post_fit_covariance(self):
        from reliability.nodes.camera_manager_node import (
            FUSION_RULE_INDEPENDENT, FUSION_RULE_JOINT_NETWORK, _gated_fusion)

        ind = _gated_fusion(self._fresh(), disagreement_gate_m=0.6,
                            rule=FUSION_RULE_INDEPENDENT)
        net = _gated_fusion(self._fresh(), disagreement_gate_m=0.6,
                            rule=FUSION_RULE_JOINT_NETWORK)
        # With no Huber downweighting, both correctly solve the same GLS normal equations.
        assert net.mean_xy == pytest.approx(ind.mean_xy, abs=1e-12)
        assert _flat(net.covariance_m2) != pytest.approx(_flat(ind.covariance_m2), rel=1e-12)

    def test_every_rule_drops_the_same_outlier(self):
        from reliability.nodes.camera_manager_node import (
            FUSION_RULE_BEST_SINGLE, FUSION_RULE_DISTANCE_ANGLE, FUSION_RULE_INDEPENDENT,
            FUSION_RULE_JOINT_NETWORK, _gated_fusion)

        # camera_D reads 0.6 m away from the others: the same gate must reject it whichever
        # rule follows, or the gate becomes part of the treatment
        readings = self._fresh() + [obs("camera_D", 0.62, 0.0, 0.060, 0.019)]
        positions = dict(CAMERA_POSITIONS, camera_D=(11.45, 7.2, 5.0))
        for rule in (FUSION_RULE_BEST_SINGLE, FUSION_RULE_DISTANCE_ANGLE,
                     FUSION_RULE_INDEPENDENT, FUSION_RULE_JOINT_NETWORK):
            result = _gated_fusion(readings, disagreement_gate_m=0.30, rule=rule,
                                   camera_positions_m=positions)
            assert "camera_D" in result.rejected_camera_ids, rule
            assert "camera_D" not in result.accepted_camera_ids, rule


    def test_distance_angle_needs_camera_positions(self):
        from reliability.nodes.camera_manager_node import (
            FUSION_RULE_DISTANCE_ANGLE, _gated_fusion)

        with pytest.raises(ValueError):
            _gated_fusion(self._fresh(), disagreement_gate_m=0.6,
                          rule=FUSION_RULE_DISTANCE_ANGLE)


class TestCorrectionTimestampCompensation:
    """A correction describes where the robot WAS; it has to be carried to where it is used."""

    def test_moves_the_correction_by_the_motion_over_the_interval(self):
        from reliability.nodes.camera_manager_node import propagate_correction_to_now

        xy, cov, delta = propagate_correction_to_now(
            (1.0, 2.0), ((1e-4, 0.0), (0.0, 1e-4)),
            pose_then=(5.0, 5.0, 0.0), pose_now=(5.077, 5.0, 0.0),
            drift_std_m_per_s=0.0, dt_s=0.35)
        # 0.35 s at 0.22 m/s is 7.7 cm of travel, and the correction moves with it
        assert xy == pytest.approx((1.077, 2.0))
        assert delta == pytest.approx((0.077, 0.0))

    def test_an_error_in_the_belief_cancels_in_the_difference(self):
        from reliability.nodes.camera_manager_node import propagate_correction_to_now

        # the same motion, seen from a belief that is a metre wrong throughout
        a = propagate_correction_to_now((0.0, 0.0), ((1e-4, 0.0), (0.0, 1e-4)),
                                        (5.0, 5.0, 0.0), (5.1, 5.0, 0.0),
                                        drift_std_m_per_s=0.0, dt_s=0.35)[0]
        b = propagate_correction_to_now((0.0, 0.0), ((1e-4, 0.0), (0.0, 1e-4)),
                                        (6.0, 5.0, 0.0), (6.1, 5.0, 0.0),
                                        drift_std_m_per_s=0.0, dt_s=0.35)[0]
        assert a == pytest.approx(b)

    def test_the_propagation_adds_its_own_uncertainty(self):
        from reliability.nodes.camera_manager_node import propagate_correction_to_now

        _xy, cov, _d = propagate_correction_to_now(
            (0.0, 0.0), ((1e-4, 2e-5), (2e-5, 1e-4)), (0.0, 0.0, 0.0), (0.1, 0.0, 0.0),
            drift_std_m_per_s=0.05, dt_s=0.4)
        grown = (0.05 * 0.4) ** 2
        assert cov[0][0] == pytest.approx(1e-4 + grown)
        assert cov[1][1] == pytest.approx(1e-4 + grown)
        assert cov[0][1] == pytest.approx(2e-5)   # the cross term is not touched

    def test_zero_interval_changes_nothing(self):
        from reliability.nodes.camera_manager_node import propagate_correction_to_now

        xy, cov, delta = propagate_correction_to_now(
            (3.0, 4.0), ((1e-4, 0.0), (0.0, 1e-4)), (1.0, 1.0, 0.0), (1.0, 1.0, 0.0),
            drift_std_m_per_s=0.05, dt_s=0.0)
        assert xy == pytest.approx((3.0, 4.0))
        assert delta == pytest.approx((0.0, 0.0))
        assert _flat(cov) == pytest.approx([1e-4, 0.0, 0.0, 1e-4])


class TestResidualIntervalFloor:
    """The leftover interval before a correction is used is a bias along travel, not noise."""

    def test_it_widens_along_travel_and_not_across(self):
        from reliability.nodes.camera_manager_node import propagate_correction_to_now

        # 0.35 s of motion due east at 0.2 m/s, consumed 50 ms later
        _xy, cov, _d = propagate_correction_to_now(
            (0.0, 0.0), ((1e-4, 0.0), (0.0, 1e-4)), (0.0, 0.0, 0.0), (0.07, 0.0, 0.0),
            drift_std_m_per_s=0.0, dt_s=0.35, residual_interval_s=0.05)
        expected = (0.05 * (0.07 / 0.35)) ** 2
        assert cov[0][0] == pytest.approx(1e-4 + expected)   # along travel: widened
        assert cov[1][1] == pytest.approx(1e-4)              # across travel: untouched

    def test_it_follows_the_direction_of_travel(self):
        from reliability.nodes.camera_manager_node import propagate_correction_to_now

        _xy, cov, _d = propagate_correction_to_now(
            (0.0, 0.0), ((1e-4, 0.0), (0.0, 1e-4)), (0.0, 0.0, 0.0), (0.0, 0.07, 0.0),
            drift_std_m_per_s=0.0, dt_s=0.35, residual_interval_s=0.05)
        expected = (0.05 * (0.07 / 0.35)) ** 2
        assert cov[1][1] == pytest.approx(1e-4 + expected)   # now north is the wide one
        assert cov[0][0] == pytest.approx(1e-4)

    def test_it_scales_with_speed(self):
        from reliability.nodes.camera_manager_node import propagate_correction_to_now

        slow = propagate_correction_to_now(
            (0.0, 0.0), ((1e-6, 0.0), (0.0, 1e-6)), (0.0, 0.0, 0.0), (0.035, 0.0, 0.0),
            drift_std_m_per_s=0.0, dt_s=0.35, residual_interval_s=0.05)[1][0][0]
        fast = propagate_correction_to_now(
            (0.0, 0.0), ((1e-6, 0.0), (0.0, 1e-6)), (0.0, 0.0, 0.0), (0.070, 0.0, 0.0),
            drift_std_m_per_s=0.0, dt_s=0.35, residual_interval_s=0.05)[1][0][0]
        assert fast > slow                       # twice the speed, four times the variance
        assert (fast - 1e-6) == pytest.approx(4.0 * (slow - 1e-6), rel=1e-9)

    def test_a_standing_robot_gets_no_extra_width(self):
        from reliability.nodes.camera_manager_node import propagate_correction_to_now

        _xy, cov, _d = propagate_correction_to_now(
            (0.0, 0.0), ((1e-4, 0.0), (0.0, 1e-4)), (1.0, 1.0, 0.0), (1.0, 1.0, 0.0),
            drift_std_m_per_s=0.0, dt_s=0.35, residual_interval_s=0.05)
        assert _flat(cov) == pytest.approx([1e-4, 0.0, 0.0, 1e-4])
