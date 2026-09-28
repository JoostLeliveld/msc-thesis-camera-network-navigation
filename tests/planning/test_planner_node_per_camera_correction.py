"""Per-camera sequential updating: one filter, N measurements.

Step 2b. Instead of collapsing the cameras into a fused pose and running a
second filter on top of it, the planner folds each camera's map observation into
the belief one at a time -- each predicted to its own stamp, each carrying its
own covariance, each gated and reason-coded on its own.

What that removes (both verified in code, see the parity audit):
- the DOUBLE FILTER: camera_manager seeds a Kalman update from a median with an
  identity prior and no motion model, then the planner treats the result as an
  independent measurement;
- the loss of per-camera measurement information, since the fused pose carries
  one covariance for all cameras.

NOT covered here: whether that per-camera covariance is itself well specified.
It currently is not -- fusion uses a constant isotropic diag(0.08^2, 0.08^2) and
`project_observation_to_world` propagates no covariance at all. Fixing that is a
separate track; these tests only assert that whatever covariance a camera
carries reaches the filter intact and is used per camera.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from reliability.contracts import CameraQuality
from reliability.fusion import (
    MAP_OBSERVATION_BATCH_SCHEMA,
    MapObservation,
    map_observations_from_json,
    map_observations_to_json,
)

from planning.core import belief_correction as bc

from test_planner_node_correction_wiring import IDX_ACCEPTED, IDX_REJECT_CODE
from test_planner_node_state_correction import (
    IDX_MEASUREMENT_SPACE,
    make_state_node,
)

IDX_CAMERA_INDEX = 46


def observation(camera_id, x, y, *, seconds, var=0.03):
    return MapObservation(
        camera_id=camera_id,
        timestamp_s=float(seconds),
        xy_m=(float(x), float(y)),
        covariance_m2=((float(var), 0.0), (0.0, float(var))),
        quality=CameraQuality(camera_id=camera_id),
        source="test",
    )


def make_per_camera_node(**kwargs):
    node = make_state_node(**kwargs)
    node.state_correction_mode = 'per_camera'
    node._seen_map_observation_stamps = {}
    return node


def test_simultaneous_distinct_cameras_both_contribute_once():
    node = make_per_camera_node(belief_cov=.05, belief_stamp_s=9.9)
    node.heading_update_mode = 'coupled'
    batch = [observation('camera_A', 0., 0., seconds=9.95, var=.01),
             observation('camera_B', 0., 0., seconds=9.95, var=.01)]
    node._apply_map_observations(batch)
    expected = 1./(1./.05 + 2./.01)
    np.testing.assert_allclose(node.belief_S[:2, :2], np.eye(2)*expected, atol=1e-10)
    assert all(d[IDX_ACCEPTED] == 1. for d in node.pixel_correction_diag_pub.published)
    before = node.belief_S.copy()
    node._apply_map_observations(batch)
    np.testing.assert_array_equal(node.belief_S, before)


# --------------------------------------------------------------------------
# Wire format (the manager -> planner seam)
# --------------------------------------------------------------------------

def test_map_observation_round_trips_through_json():
    obs = observation("camera_B", 1.25, -3.5, seconds=12.75, var=0.041)
    restored, frame_id = map_observations_from_json(
        map_observations_to_json([obs], frame_id="map_bev")
    )
    assert frame_id == "map_bev"
    assert len(restored) == 1
    assert restored[0] == obs


def test_batch_preserves_per_camera_covariance_rather_than_collapsing_it():
    """The whole point: each camera keeps its own R across the wire."""
    batch = [
        observation("camera_A", 1.0, 0.0, seconds=1.0, var=0.01),
        observation("camera_D", 1.1, 0.0, seconds=1.0, var=0.90),
    ]
    restored, _ = map_observations_from_json(map_observations_to_json(batch))
    assert [o.covariance_m2[0][0] for o in restored] == [0.01, 0.90]


def test_unknown_schema_is_refused_not_silently_misread():
    payload = json.loads(map_observations_to_json([observation("camera_A", 0, 0, seconds=1.0)]))
    payload["schema"] = "map_observation_batch/999"
    with pytest.raises(Exception) as excinfo:
        map_observations_from_json(json.dumps(payload))
    assert "schema" in str(excinfo.value)
    assert MAP_OBSERVATION_BATCH_SCHEMA in str(excinfo.value)


def test_batch_rejects_a_malformed_observation():
    payload = json.loads(map_observations_to_json([observation("camera_A", 0, 0, seconds=1.0)]))
    del payload["observations"][0]["covariance_m2"]
    with pytest.raises(Exception):
        map_observations_from_json(json.dumps(payload))


# --------------------------------------------------------------------------
# Sequential updating
# --------------------------------------------------------------------------

def test_every_camera_produces_its_own_gated_correction():
    node = make_per_camera_node(belief_xy=(0.0, 0.0), belief_stamp_s=9.90, now_s=10.0)
    node._apply_map_observations([
        observation("camera_A", 0.05, 0.0, seconds=9.93),
        observation("camera_B", 0.06, 0.0, seconds=9.94),
        observation("camera_C", 0.05, 0.01, seconds=9.95),
    ])

    published = node.pixel_correction_diag_pub.published
    assert len(published) == 3, "each camera must be folded in separately"
    assert [d[IDX_CAMERA_INDEX] for d in published] == [0.0, 1.0, 2.0]
    assert all(d[IDX_MEASUREMENT_SPACE] == bc.SPACE_MAP_XY for d in published)
    assert all(d[IDX_ACCEPTED] == 1.0 for d in published)


def test_observations_are_applied_in_timestamp_order():
    """A Kalman update is only valid against a prior predicted to its stamp."""
    node = make_per_camera_node(belief_xy=(0.0, 0.0), belief_stamp_s=9.90, now_s=10.0)
    node._apply_map_observations([
        observation("camera_C", 0.05, 0.0, seconds=9.96),
        observation("camera_A", 0.05, 0.0, seconds=9.92),
        observation("camera_B", 0.05, 0.0, seconds=9.94),
    ])
    # Belief stamp ends at the LATEST observation, and every update was accepted,
    # which can only happen if they were applied oldest-first (an out-of-order
    # one would be dropped as not-newer-than-the-belief).
    assert len(node.pixel_correction_diag_pub.published) == 3
    assert node.belief_stamp.nanosec == 960000000


def test_a_stale_observation_in_the_batch_is_dropped_not_applied_backwards():
    node = make_per_camera_node(belief_xy=(0.0, 0.0), belief_stamp_s=9.95, now_s=10.0)
    node._apply_map_observations([
        observation("camera_A", 0.05, 0.0, seconds=9.90),   # older than the belief
        observation("camera_B", 0.05, 0.0, seconds=9.97),
    ])
    diagnostics = node.pixel_correction_diag_pub.published
    assert len(diagnostics) == 2, "a dropped observation must remain visible"
    assert diagnostics[0][IDX_CAMERA_INDEX] == 0.0
    assert diagnostics[0][IDX_REJECT_CODE] == bc.REJECT_CODES['not_newer_than_belief']
    assert diagnostics[1][IDX_CAMERA_INDEX] == 1.0
    assert diagnostics[1][IDX_ACCEPTED] == 1.0


def test_one_bad_camera_is_rejected_while_the_others_are_accepted():
    """Per-camera gating -- impossible when the cameras arrive pre-fused."""
    node = make_per_camera_node(belief_xy=(0.0, 0.0), belief_stamp_s=9.90, now_s=10.0,
                                belief_cov=1e-3)
    node._apply_map_observations([
        observation("camera_A", 0.01, 0.0, seconds=9.93, var=1e-3),
        observation("camera_B", 4.00, 0.0, seconds=9.94, var=1e-3),   # gross outlier
        observation("camera_C", 0.01, 0.0, seconds=9.95, var=1e-3),
    ])

    codes = [d[IDX_REJECT_CODE] for d in node.pixel_correction_diag_pub.published]
    assert len(codes) == 3
    assert codes[0] == bc.ACCEPTED_CODE
    assert codes[1] != bc.ACCEPTED_CODE, "the outlier camera must be gated out"
    assert codes[2] == bc.ACCEPTED_CODE, "a good camera must survive a bad neighbour"
    # And the belief stayed near the good cameras, not the outlier.
    assert abs(node.belief_m[0]) < 0.5


def test_disabling_per_camera_reanchor_does_not_enable_the_pixel_jump_limit():
    """Recovery policy and measurement space are independent choices.

    Per-camera observations deliberately cannot re-anchor the belief on one
    camera's word.  They are still metric observations, so disabling re-anchor
    must not select paper-1's 0.5 m pixel-update limiter.  That accidental
    coupling rejected routine camera corrections in the live showcase.
    """
    node = make_per_camera_node(
        belief_xy=(0.0, 0.0), belief_stamp_s=9.90, now_s=10.0, belief_cov=1.0
    )
    assert node.state_max_correction_jump_m == 0.0
    assert node.pixel_max_correction_jump_m == pytest.approx(0.5)

    node._apply_map_observations([
        observation("camera_A", 1.0, 0.0, seconds=9.95, var=0.03),
    ])

    diag = node.pixel_correction_diag_pub.published[-1]
    assert diag[IDX_ACCEPTED] == 1.0
    assert diag[IDX_REJECT_CODE] != bc.REJECT_CODES['jump_too_large']


def test_a_low_confidence_camera_moves_the_belief_less_than_a_confident_one():
    """Each camera's own covariance reaches the filter and weights its update."""
    def final_x(var):
        node = make_per_camera_node(belief_xy=(0.0, 0.0), belief_stamp_s=9.90,
                                    now_s=10.0, belief_cov=0.05)
        node._apply_map_observations([observation("camera_A", 0.2, 0.0, seconds=9.95, var=var)])
        return node.belief_m[0]

    confident = final_x(1e-3)
    unsure = final_x(1.0)
    assert confident > unsure, "a tighter R must pull the belief further"
    assert unsure < 0.05


def test_bootstrap_from_the_first_per_camera_observation():
    node = make_per_camera_node()
    node.belief_m = None
    node.belief_S = None
    node.belief_stamp = None
    node._apply_map_observations([observation("camera_A", 2.0, 3.0, seconds=9.95)])
    assert node.belief_m[0] == pytest.approx(2.0)
    assert node.belief_m[1] == pytest.approx(3.0)


def test_empty_batch_is_a_no_op():
    node = make_per_camera_node(belief_xy=(0.0, 0.0))
    before = node.belief_m.copy()
    node._apply_map_observations([])
    np.testing.assert_array_equal(node.belief_m, before)
    assert node.pixel_correction_diag_pub.published == []


# --------------------------------------------------------------------------
# Mode routing -- the two arms must not both correct the belief
# --------------------------------------------------------------------------

def test_per_camera_mode_ignores_the_fused_pose_so_it_is_not_folded_in_twice():
    from test_planner_node_state_correction import state_msg

    node = make_per_camera_node(belief_xy=(0.0, 0.0))
    before = node.belief_m.copy()
    node._state_cb(state_msg(1.0, 0.0, seconds=9.95))
    np.testing.assert_array_equal(node.belief_m, before)
    assert node.state_msg is not None, "the fused pose is still cached for planning"


def test_fused_mode_ignores_a_per_camera_batch():
    from std_msgs.msg import String

    node = make_state_node(belief_xy=(0.0, 0.0))     # state_correction_mode = 'fused'
    before = node.belief_m.copy()
    msg = String()
    msg.data = map_observations_to_json([observation("camera_A", 1.0, 0.0, seconds=9.95)])
    node._map_observations_cb(msg)
    np.testing.assert_array_equal(node.belief_m, before)


def test_a_corrupt_batch_stops_the_run_instead_of_quietly_never_correcting():
    """A malformed batch used to be a warning. That let a drive keep going with an
    estimator that had stopped updating, and the run still scored as valid."""
    from std_msgs.msg import String

    node = make_per_camera_node(belief_xy=(0.0, 0.0))
    before = node.belief_m.copy()
    msg = String()
    msg.data = '{"schema": "nope", "observations": []}'
    with pytest.raises(RuntimeError, match="map-observation batch"):
        node._map_observations_cb(msg)
    np.testing.assert_array_equal(node.belief_m, before)
    assert node._fatal_stop_triggered
    assert any('map-observation batch' in m for _, m in node._logger.messages)


def test_foreign_frame_is_rejected_before_any_state_or_dedup_mutation():
    from std_msgs.msg import String

    node = make_per_camera_node(belief_xy=(0., 0.))
    node._resolve_plan_frame_id = lambda: 'map_bev'
    before = node.belief_m.copy(), node.belief_S.copy()
    errors = []
    node._fatal_experiment_stop = lambda context, exc: errors.append(str(exc))
    msg = String()
    msg.data = map_observations_to_json(
        [observation('camera_A', .1, 0., seconds=9.95)], frame_id='camera_optical')
    node._map_observations_cb(msg)
    assert len(errors) == 1 and 'frame' in errors[0]
    assert node._seen_map_observation_stamps == {}
    np.testing.assert_array_equal(node.belief_m, before[0])
    np.testing.assert_array_equal(node.belief_S, before[1])


def test_parser_failure_returns_even_if_the_stop_handler_returns():
    from std_msgs.msg import String

    node = make_per_camera_node()
    errors = []
    node._fatal_experiment_stop = lambda context, exc: errors.append(str(exc))
    node._apply_map_observations = lambda observations: pytest.fail('applied a malformed batch')
    msg = String()
    msg.data = '{}'
    node._map_observations_cb(msg)
    assert len(errors) == 1
