"""The floor that stops repetition from buying confidence it has not earned."""
from __future__ import annotations

import math

import pytest

from reliability.bias_floor import (
    DEFAULT_ACROSS_RAY_SLOPE,
    DEFAULT_ALONG_RAY_SLOPE,
    RETIRED_CONSTANT_FLOOR_M,
    BiasFloorError,
    apply_belief_floor,
    bias_floor_matrix,
    ray_bearing_rad,
)


def _sd(matrix, index):
    return math.sqrt(matrix[index][index])


def test_the_floor_reproduces_the_measured_envelope():
    """0.96 cm at 6 m, 1.92 at 12, 3.20 at 20 -- D22's bound over every measured cell."""
    for range_m, expected_cm in ((6.0, 0.96), (12.0, 1.92), (20.0, 3.20)):
        along = _sd(bias_floor_matrix(range_m, 0.0), 0)
        assert 100.0 * along == pytest.approx(expected_cm, abs=0.01)


def test_the_floor_is_anisotropic_and_follows_the_ray():
    """D21: the residual is along the line of sight, so the floor cannot be a scalar."""
    along_ray = bias_floor_matrix(20.0, 0.0)
    assert _sd(along_ray, 0) > 4.0 * _sd(along_ray, 1), "along must dominate across"

    # rotate the bearing by 90 degrees and the large axis must rotate with it
    rotated = bias_floor_matrix(20.0, math.pi / 2.0)
    assert _sd(rotated, 1) == pytest.approx(_sd(along_ray, 0), rel=1e-9)
    assert _sd(rotated, 0) == pytest.approx(_sd(along_ray, 1), rel=1e-9)

    # at 45 degrees the floor must be correlated, not axis-aligned
    diagonal = bias_floor_matrix(20.0, math.pi / 4.0)
    assert abs(diagonal[0][1]) > 0.5 * diagonal[0][0]


def test_a_flat_floor_is_wrong_at_both_ends_which_is_why_it_was_replaced():
    near = 100.0 * _sd(bias_floor_matrix(6.0, 0.0), 0)
    far = 100.0 * _sd(bias_floor_matrix(20.0, 0.0), 0)
    retired_cm = 100.0 * RETIRED_CONSTANT_FLOOR_M
    assert retired_cm > 2.5 * near, "the flat floor over-bounds where cameras are good"
    assert retired_cm < far, "and under-bounds where they are weakest"


def test_an_overconfident_belief_is_raised_and_a_wide_one_is_left_alone():
    floor = bias_floor_matrix(20.0, 0.0)
    sharp = ((1.0e-6, 0.0), (0.0, 1.0e-6))
    raised = apply_belief_floor(sharp, floor)
    assert _sd(raised, 0) == pytest.approx(_sd(floor, 0), rel=1e-6)

    wide = ((0.25, 0.0), (0.0, 0.25))          # half a metre of sd, far above the floor
    untouched = apply_belief_floor(wide, floor)
    assert _sd(untouched, 0) == pytest.approx(0.5, rel=1e-6)
    assert _sd(untouched, 1) == pytest.approx(0.5, rel=1e-6)


def test_the_floor_binds_along_every_direction_not_just_the_diagonal():
    """The case a cheap elementwise maximum gets wrong, and why the eigenvalue form is used.

    This belief has BOTH diagonal entries above the floor's, so an elementwise max would
    accept it unchanged -- yet it is nearly singular along one diagonal direction, which
    is precisely a filter claiming near-certainty it has not earned.
    """
    a = 4.0e-4                                  # 2 cm sd, isotropic floor
    floor = ((a, 0.0), (0.0, a))
    sharp_diagonal = ((1.5 * a, 1.4 * a), (1.4 * a, 1.5 * a))
    assert sharp_diagonal[0][0] > floor[0][0] and sharp_diagonal[1][1] > floor[1][1]

    smallest_before = min(_eigs(sharp_diagonal))
    assert smallest_before < a, "the belief starts sharper than the floor somewhere"

    raised = apply_belief_floor(sharp_diagonal, floor)
    assert min(_eigs(raised)) == pytest.approx(a, rel=1e-6)
    # and the direction it was already wide in is not narrowed
    assert max(_eigs(raised)) == pytest.approx(max(_eigs(sharp_diagonal)), rel=1e-6)


def _eigs(matrix):
    (p, q), (_q, r) = matrix
    tr, det = p + r, p * r - q * q
    root = math.sqrt(max(tr * tr / 4.0 - det, 0.0))
    return (tr / 2.0 - root, tr / 2.0 + root)


def test_the_bearing_comes_from_the_geometry_and_degenerate_input_is_refused():
    assert ray_bearing_rad((0.0, 0.0), (1.0, 0.0)) == pytest.approx(0.0)
    assert ray_bearing_rad((0.0, 0.0), (0.0, 2.0)) == pytest.approx(math.pi / 2.0)
    with pytest.raises(BiasFloorError):
        ray_bearing_rad((1.0, 1.0), (1.0, 1.0))


@pytest.mark.parametrize("bad", [
    {"range_m": -1.0, "bearing_rad": 0.0},
    {"range_m": float("nan"), "bearing_rad": 0.0},
    {"range_m": 5.0, "bearing_rad": float("inf")},
])
def test_unusable_geometry_is_refused_rather_than_silently_floored(bad):
    with pytest.raises(BiasFloorError):
        bias_floor_matrix(**bad)


def test_a_non_psd_or_asymmetric_belief_is_refused():
    floor = bias_floor_matrix(10.0, 0.0)
    with pytest.raises(BiasFloorError):
        apply_belief_floor(((1.0, 0.2), (0.9, 1.0)), floor)      # asymmetric
    with pytest.raises(BiasFloorError):
        apply_belief_floor(((1.0, 0.0), (0.0, 1.0)), ((0.0, 0.0), (0.0, 0.0)))


def test_the_defaults_are_the_measured_ones_and_across_is_the_smaller():
    assert DEFAULT_ALONG_RAY_SLOPE == pytest.approx(0.0016)
    assert DEFAULT_ACROSS_RAY_SLOPE < DEFAULT_ALONG_RAY_SLOPE / 3.0


def test_the_update_applies_the_floor_to_the_posterior_and_is_off_by_default():
    """Wired into the one place a posterior is formed, and silent unless asked for."""
    from reliability.contracts import CameraQuality
    from reliability.fusion import MapObservation, joseph_update_2d

    observation = MapObservation(
        camera_id="camera_A", timestamp_s=0.0, xy_m=(1.0, 0.0),
        covariance_m2=((1.0e-6, 0.0), (0.0, 1.0e-6)),
        quality=CameraQuality(camera_id="camera_A"),
    )
    prior = ((1.0e-6, 0.0), (0.0, 1.0e-6))

    _mean, without, _nis = joseph_update_2d((0.0, 0.0), prior, observation)
    assert _sd(without, 0) < 0.01, "default must stay unfloored"

    floor = bias_floor_matrix(20.0, 0.0)
    _mean, floored, _nis = joseph_update_2d(
        (0.0, 0.0), prior, observation, belief_floor=floor)
    assert _sd(floored, 0) == pytest.approx(_sd(floor, 0), rel=1e-6)
    assert _sd(floored, 0) > 10.0 * _sd(without, 0), "the floor must actually bind"

    # repetition must not talk it back down: ten more identical sightings
    mean, cov = (0.0, 0.0), prior
    for _ in range(10):
        mean, cov, _nis = joseph_update_2d(mean, cov, observation, belief_floor=floor)
    assert _sd(cov, 0) == pytest.approx(_sd(floor, 0), rel=1e-6), (
        "ten looks at the same place must not shrink a floored belief")


def _observation(camera_id, xy, sigma_m=1.0e-3):
    from reliability.contracts import CameraQuality
    from reliability.fusion import MapObservation

    return MapObservation(
        camera_id=camera_id, timestamp_s=0.0, xy_m=xy,
        covariance_m2=((sigma_m ** 2, 0.0), (0.0, sigma_m ** 2)),
        quality=CameraQuality(camera_id=camera_id),
    )


def test_a_second_viewpoint_can_never_shrink_the_floor():
    """The correction that measurement forced: combination must be conservative.

    This first combined floors by inverse sum, on the reasoning that each camera's bias
    points along its own ray so different bearings partly cancel. Measured, that turned a
    single camera's 2.40 cm floor at 15 m into 0.61 cm with two cameras -- a fourfold
    reduction -- because one camera's across-ray tightness was allowed to constrain the
    direction another was loose in. That is the unearned confidence the floor exists to
    forbid, so the rule is now a bound: at least as wide as every input, in every
    direction.
    """
    from reliability.bias_floor import combine_floors

    east = bias_floor_matrix(20.0, 0.0)                # bias along x
    north = bias_floor_matrix(20.0, math.pi / 2.0)     # bias along y
    both = combine_floors([east, north])

    for single in (east, north):
        for vector in ((1.0, 0.0), (0.0, 1.0), (0.7071, 0.7071)):
            quad_single = (vector[0] * (single[0][0] * vector[0] + single[0][1] * vector[1])
                           + vector[1] * (single[1][0] * vector[0] + single[1][1] * vector[1]))
            quad_both = (vector[0] * (both[0][0] * vector[0] + both[0][1] * vector[1])
                         + vector[1] * (both[1][0] * vector[0] + both[1][1] * vector[1]))
            assert quad_both >= quad_single - 1e-12, (
                "the combined floor must not be narrower than any input in any direction")

    # one floor alone comes back unchanged
    alone = combine_floors([east])
    assert _sd(alone, 0) == pytest.approx(_sd(east, 0), rel=1e-9)


def test_the_sequential_update_floors_once_and_ignores_rejected_cameras():
    from reliability.fusion import sequential_kalman_update_2d

    prior = ((1.0, 0.0), (0.0, 1.0))
    floors = {
        "camera_A": bias_floor_matrix(20.0, 0.0),
        "camera_D": bias_floor_matrix(20.0, 0.0),
    }
    observations = [_observation("camera_A", (0.0, 0.0)), _observation("camera_D", (0.0, 0.0))]

    unfloored = sequential_kalman_update_2d((0.0, 0.0), prior, observations)
    assert _sd(unfloored.covariance_m2, 0) < 0.005, "default stays unfloored"

    floored = sequential_kalman_update_2d(
        (0.0, 0.0), prior, observations, belief_floors=floors)
    assert _sd(floored.covariance_m2, 0) > _sd(unfloored.covariance_m2, 0)
    assert floored.accepted_camera_ids == ("camera_A", "camera_D")

    # a camera the gates threw out contributes no floor, because it contributed no
    # information to bound: put it far away so the disagreement gate rejects it
    with_outlier = observations + [_observation("camera_E", (50.0, 50.0))]
    gated = sequential_kalman_update_2d(
        (0.0, 0.0), prior, with_outlier,
        disagreement_gate_m=1.0,
        belief_floors={**floors, "camera_E": bias_floor_matrix(2.0, 0.0)},
    )
    assert "camera_E" in gated.rejected_camera_ids
    assert _sd(gated.covariance_m2, 0) == pytest.approx(
        _sd(floored.covariance_m2, 0), rel=1e-9), (
        "a rejected camera's floor must not tighten the belief")


def test_the_floor_is_order_independent():
    from reliability.fusion import sequential_kalman_update_2d

    prior = ((1.0, 0.0), (0.0, 1.0))
    floors = {"camera_A": bias_floor_matrix(6.0, 0.0),
              "camera_D": bias_floor_matrix(20.0, math.pi / 3.0)}
    a = [_observation("camera_A", (0.0, 0.0)), _observation("camera_D", (0.0, 0.0))]
    forward = sequential_kalman_update_2d((0.0, 0.0), prior, a, belief_floors=floors)
    backward = sequential_kalman_update_2d((0.0, 0.0), prior, list(reversed(a)),
                                           belief_floors=floors)
    for i in (0, 1):
        assert _sd(forward.covariance_m2, i) == pytest.approx(
            _sd(backward.covariance_m2, i), rel=1e-9)


def test_the_camera_manager_can_apply_the_floor_and_defaults_to_off():
    """The runtime path: the node builds per-camera floors from its own geometry.

    Checked on the source rather than by standing a node up, because the node needs a
    live ROS graph. What matters is that the wiring exists, that the default is off, and
    that the floor is built from the camera's surveyed position rather than a constant.
    """
    from pathlib import Path

    source = (Path(__file__).resolve().parents[2]
              / "src/reliability/reliability/nodes/camera_manager_node.py").read_text()

    assert 'declare_parameter("bias_floor_along_slope_m_per_m", 0.0)' in source, (
        "the floor must default to off: it changes what the filter may believe")
    assert "belief_floors=self._bias_floors(fresh)" in source, "not wired into fusion"
    assert "def _bias_floors" in source

    # built inside the class, not appended after main()
    class_start = source.index("class CameraManagerNode")
    main_guard = source.index('if __name__ ==')
    assert class_start < source.index("def _bias_floors") < main_guard

    # and from the camera's own geometry
    helper = source[source.index("def _bias_floors"):source.index("def _decide_fused")]
    assert "model.cam_pos" in helper and "ray_bearing_rad" in helper
    assert "math.hypot" in helper, "range must come from the geometry, not a constant"


def test_one_positive_slope_is_a_configuration_error_not_a_runtime_crash():
    """A floor with a zero axis is not a covariance.

    The manager used to accept along=0 with across>0, build a singular floor, and die
    at the first fusion with "every floor must be positive definite" -- minutes into a
    run, after the simulator had started. It is a configuration mistake and belongs at
    start-up.
    """
    from reliability.bias_floor import bias_floor_matrix, combine_floors, BiasFloorError

    singular = bias_floor_matrix(10.0, 0.3, along_slope=0.0, across_slope=0.00035)
    with pytest.raises(BiasFloorError, match="positive definite"):
        combine_floors([singular])
