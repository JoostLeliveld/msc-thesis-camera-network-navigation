"""Batch identity, unit invariance and numerical stability of camera fusion."""
from dataclasses import replace
from itertools import permutations
import json

import numpy as np
import pytest

from reliability.contracts import CameraQuality, ContractValidationError
from reliability.fusion import (
    MapObservation,
    distance_angle_weighted_fusion_2d,
    independent_measurement_fusion_2d,
    joint_network_estimate_2d,
    map_observations_from_json,
    map_observations_to_json,
    select_information_best,
    select_smallest_covariance,
    sequential_kalman_update_2d,
)
from reliability.measurement_fusion import gated_measurement_fusion_2d


def reading(camera_id="camera_A", xy=(1., 2.), variance=.01):
    return MapObservation(camera_id, 1., xy, ((variance, 0.), (0., variance)),
                          CameraQuality(camera_id=camera_id))


@pytest.mark.parametrize("stamp", [float("nan"), float("inf"), -float("inf"), -1.])
def test_invalid_timestamp_cannot_enter_a_map_observation(stamp):
    with pytest.raises(ContractValidationError, match="timestamp"):
        replace(reading(), timestamp_s=stamp)


@pytest.mark.parametrize("camera_id", ["", " ", None])
def test_camera_identity_cannot_be_empty(camera_id):
    with pytest.raises(ContractValidationError, match="camera_id"):
        replace(reading(), camera_id=camera_id, quality=CameraQuality())


@pytest.mark.parametrize("missing", ["frame_id", "observations"])
def test_wire_batch_requires_explicit_frame_and_observations(missing):
    payload = json.loads(map_observations_to_json([reading()]))
    del payload[missing]
    with pytest.raises(ContractValidationError):
        map_observations_from_json(json.dumps(payload))


@pytest.mark.parametrize("raw", ["", {}, None])
def test_wire_batch_requires_an_actual_list(raw):
    payload = json.loads(map_observations_to_json([]))
    payload["observations"] = raw
    with pytest.raises(ContractValidationError):
        map_observations_from_json(json.dumps(payload))


@pytest.mark.parametrize("location", ["batch", "observation"])
def test_wire_schema_rejects_unknown_fields(location):
    payload = json.loads(map_observations_to_json([reading()]))
    if location == "batch":
        payload["trace_rule"] = "old_experiment"
    else:
        payload["observations"][0]["trace_rule"] = "old_experiment"
    with pytest.raises(ContractValidationError, match="unknown fields"):
        map_observations_from_json(json.dumps(payload))


def test_map_observation_does_not_coerce_source_or_quality_types():
    with pytest.raises(ContractValidationError, match="source"):
        replace(reading(), source={"old": "metadata"})
    with pytest.raises(ContractValidationError, match="quality"):
        replace(reading(), quality={"camera_id": "camera_A"})


@pytest.mark.parametrize("fuse", [independent_measurement_fusion_2d,
                                 joint_network_estimate_2d,
                                 distance_angle_weighted_fusion_2d])
def test_one_camera_cannot_vote_twice_in_a_fusion_batch(fuse):
    a = reading()
    kwargs = {"camera_positions_m": {a.camera_id: (0., 0., 5.)}} if fuse is distance_angle_weighted_fusion_2d else {}
    with pytest.raises(ContractValidationError, match="duplicate"):
        fuse([a, a], **kwargs)


@pytest.mark.parametrize("fuse", [independent_measurement_fusion_2d, joint_network_estimate_2d])
@pytest.mark.parametrize("scale", [1e-4, 1., 1e4])
def test_changing_position_units_preserves_fusion(fuse, scale):
    # A fixed determinant cutoff used to reject the same problem in smaller units.
    batch = [reading("camera_A", (scale, 2*scale), .01*scale**2),
             reading("camera_B", (scale, 2*scale), .01*scale**2)]
    mean, covariance = fuse(batch)
    np.testing.assert_allclose(np.asarray(mean)/scale, [1., 2.], atol=1e-12)
    expected = .005 if fuse is independent_measurement_fusion_2d else .01
    np.testing.assert_allclose(np.asarray(covariance)/scale**2, expected*np.eye(2), atol=1e-12)


def test_equal_covariance_selectors_have_a_stable_camera_tiebreak():
    batch = [reading("camera_C", (3., 0.)), reading("camera_A", (1., 0.)),
             reading("camera_B", (2., 0.))]
    for order in permutations(batch):
        assert select_smallest_covariance(order).camera_id == "camera_A"
        assert select_information_best(order, np.eye(2)).camera_id == "camera_A"


def test_sequential_update_does_not_cancel_a_precise_posterior_to_zero():
    result = sequential_kalman_update_2d((0., 0.), 1e12*np.eye(2),
                                        [reading(variance=1e-6)])
    np.testing.assert_allclose(result.covariance_m2, 1e-6*np.eye(2), rtol=1e-12, atol=0)


@pytest.mark.parametrize("epsilon", [0., -1., float("nan")])
def test_distance_weights_refuse_an_invalid_regularizer(epsilon):
    with pytest.raises(ContractValidationError, match="epsilon"):
        distance_angle_weighted_fusion_2d([reading()], {"camera_A": (0., 0., 5.)}, epsilon=epsilon)


@pytest.mark.parametrize("iterations", [True, 1.5, float("inf")])
def test_irls_iteration_budget_is_an_integer(iterations):
    with pytest.raises(ContractValidationError, match="max_iterations"):
        joint_network_estimate_2d([reading()], max_iterations=iterations)


def test_all_refused_cameras_do_not_produce_a_fake_measurement():
    batch = [reading("camera_A", (0., 0.)), reading("camera_B", (2., 0.))]
    result = gated_measurement_fusion_2d(batch, disagreement_gate_m=.1, rule="independent")
    assert result.accepted_camera_ids == ()
    assert result.mean_xy is None and result.covariance_m2 is None
    assert result.residuals_m_by_camera == {"camera_A": 1., "camera_B": 1.}


def test_unknown_rule_is_refused_even_when_all_cameras_fail_the_gate():
    batch = [reading("camera_A", (0., 0.)), reading("camera_B", (2., 0.))]
    with pytest.raises(ContractValidationError, match="fusion_rule"):
        gated_measurement_fusion_2d(batch, disagreement_gate_m=.1, rule="legacy_typo")


def test_duplicates_cannot_move_the_gate_median():
    a = reading("camera_A", (0., 0.))
    b = reading("camera_B", (2., 0.))
    with pytest.raises(ContractValidationError, match="duplicate"):
        gated_measurement_fusion_2d([a, a, b], disagreement_gate_m=.1, rule="independent")


def test_three_camera_floor_is_reproducible_for_every_batch_permutation():
    from reliability.bias_floor import bias_floor_matrix

    batch = [reading(camera_id, (0., 0.), variance=1e-6)
             for camera_id in ("camera_A", "camera_B", "camera_C")]
    floors = {obs.camera_id: bias_floor_matrix(10.+3*i, .7*i)
              for i, obs in enumerate(batch)}
    results = [gated_measurement_fusion_2d(order, disagreement_gate_m=.1,
               rule="independent", belief_floors=floors) for order in permutations(batch)]
    for result in results:
        np.testing.assert_allclose(result.covariance_m2, results[0].covariance_m2, atol=1e-15)
        for floor in floors.values():
            assert np.linalg.eigvalsh(np.asarray(result.covariance_m2)-floor).min() >= -1e-12


def test_tiny_covariances_still_validate_symmetry_relative_to_their_units():
    with pytest.raises(ContractValidationError, match="symmetric"):
        replace(reading(), covariance_m2=((1e-12, 4e-13), (0., 1e-12)))


def test_wire_batch_rejects_duplicate_sources_before_filter_side_effects():
    payload = json.loads(map_observations_to_json([reading()]))
    payload["observations"] *= 2
    with pytest.raises(ContractValidationError, match="duplicate"):
        map_observations_from_json(json.dumps(payload))


def test_an_infinite_gate_fuses_every_camera():
    batch = [reading("camera_A", (0., 0.)), reading("camera_B", (2., 0.)), reading("camera_C", (0.1, 0.))]
    result = gated_measurement_fusion_2d(batch, disagreement_gate_m=float("inf"), rule="independent")
    assert set(result.accepted_camera_ids) == {"camera_A", "camera_B", "camera_C"}
    assert result.rejected_camera_ids == ()


@pytest.mark.parametrize("gate", [0., -1., float("nan")])
def test_a_nonpositive_gate_is_refused(gate):
    with pytest.raises(ContractValidationError, match="disagreement_gate_m"):
        gated_measurement_fusion_2d([reading()], disagreement_gate_m=gate, rule="independent")
