from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
for relative in ("src/reliability", "src/unav_common"):
    location = str(ROOT / relative)
    if location not in sys.path:
        sys.path.insert(0, location)

from reliability.contracts import CameraObservation, CameraQuality  # noqa: E402
from reliability.projection import (  # noqa: E402
    camera_model_from_world,
    project_observation_to_world,
    project_observation_to_world_with_covariance,
)

WORLD_SDF = ROOT / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf"
CAMERA_INCLUDES = {
    "camera_A": "external_camera",
    "camera_B": "external_camera_b",
    "camera_C": "external_camera_c",
    "camera_D": "external_camera_d",
}


def _observation(
    camera_id: str,
    *,
    u: float,
    v: float,
    conditional_cov_uv=((2.5**2, 0.0), (0.0, 2.5**2)),
) -> CameraObservation:
    return CameraObservation(
        camera_id=camera_id,
        timestamp_s=10.0,
        pixel_uv=(u, v),
        detection_valid=True,
        detector_score=0.9,
        measurement_age_s=0.02,
        calibration_id=f"warehouse_full_4cam_{camera_id}",
        image_frame_id=camera_id,
        conditional_cov_uv=conditional_cov_uv,
    )


def test_projection_rejects_invalid_detection():
    model = camera_model_from_world(WORLD_SDF, include_name="external_camera")
    invalid = CameraObservation(camera_id="camera_A", timestamp_s=1.0, detection_valid=False)
    assert project_observation_to_world(invalid, model) is None


def test_paper1_pixel_covariance_is_propagated_as_full_map_ellipse():
    model = camera_model_from_world(WORLD_SDF, include_name="external_camera_c")
    observation = _observation(
        "camera_C",
        u=710.0,
        v=470.0,
        conditional_cov_uv=((6.25, 1.5), (1.5, 25.0)),
    )
    projected = project_observation_to_world_with_covariance(observation, model)
    assert projected is not None
    world_xy, covariance = projected
    assert world_xy == pytest.approx(
        project_observation_to_world(observation, model)
    )
    assert covariance[0][1] == pytest.approx(covariance[1][0])
    assert abs(covariance[0][1]) > 1.0e-8
    assert covariance[0][0] > 0.0
    assert covariance[1][1] > 0.0
    assert covariance[0][0] * covariance[1][1] - covariance[0][1] ** 2 > 0.0


def test_projected_covariance_scales_linearly_with_pixel_covariance():
    model = camera_model_from_world(WORLD_SDF, include_name="external_camera")
    base = project_observation_to_world_with_covariance(
        _observation(
            "camera_A", u=640.0, v=500.0,
            conditional_cov_uv=((6.25, 0.0), (0.0, 6.25)),
        ),
        model,
    )
    scaled = project_observation_to_world_with_covariance(
        _observation(
            "camera_A", u=640.0, v=500.0,
            conditional_cov_uv=((25.0, 0.0), (0.0, 25.0)),
        ),
        model,
    )
    assert base is not None and scaled is not None
    for row in range(2):
        for column in range(2):
            assert scaled[1][row][column] == pytest.approx(
                4.0 * base[1][row][column], rel=1.0e-9, abs=1.0e-12
            )


def test_all_four_cameras_project_into_world_bounds():
    for camera_id, include in CAMERA_INCLUDES.items():
        model = camera_model_from_world(WORLD_SDF, include_name=include)
        world_xy = project_observation_to_world(
            _observation(camera_id, u=640.0, v=520.0), model
        )
        assert world_xy is not None
        x, y = world_xy
        assert -12.5 <= x <= 12.5
        assert -10.5 <= y <= 10.5


def test_node_module_imports_without_ros():
    """The node module must stay importable in non-ROS tooling contexts."""

    from reliability.nodes import camera_manager_node

    assert hasattr(camera_manager_node, "CameraManagerNode")
    assert hasattr(camera_manager_node, "main")
    quality = camera_manager_node._contract_quality(
        _observation("camera_B", u=100.0, v=200.0)
    )
    assert quality.camera_id == "camera_B"
    assert quality.p_available == 1.0
    assert quality.source_model == "live_contract"


def test_gp_query_uses_state_nearest_camera_capture_time():
    from reliability.nodes.camera_manager_node import _nearest_state_xy

    history = [
        (9.7, (-1.0, 2.0)),
        (10.04, (3.0, 4.0)),
        (10.4, (8.0, 9.0)),
    ]
    assert _nearest_state_xy(history, 10.0, max_delta_s=0.10) == (3.0, 4.0)


def test_gp_query_rejects_state_outside_sync_window():
    from reliability.nodes.camera_manager_node import _nearest_state_xy

    assert _nearest_state_xy(
        [(9.5, (1.0, 2.0))], 10.0, max_delta_s=0.35
    ) is None


def test_camera_observation_roundtrip_preserves_detector_batch_identity():
    observation = CameraObservation(
        camera_id="camera_A", timestamp_s=10.0, detection_valid=False,
        source_batch_id="strict:camera_A@100,camera_B@101",
    )
    restored = CameraObservation.from_json(observation.to_json())
    assert restored.source_batch_id == observation.source_batch_id


def test_asynchronous_camera_positions_are_aligned_before_fusion():
    from reliability.fusion import MapObservation
    from reliability.nodes.camera_manager_node import align_observations_to_common_time

    def quality(camera_id):
        return CameraQuality(
            camera_id=camera_id, p_available=1.0,
            conditional_cov_uv=((1.0, 0.0), (0.0, 1.0)),
            association_confidence=1.0,
        )
    old = MapObservation(
        camera_id="camera_A", timestamp_s=10.0, xy_m=(1.0, 2.0),
        covariance_m2=((0.01, 0.0), (0.0, 0.01)), quality=quality("camera_A"),
    )
    new = MapObservation(
        camera_id="camera_B", timestamp_s=10.1, xy_m=(1.02, 2.0),
        covariance_m2=((0.01, 0.0), (0.0, 0.01)), quality=quality("camera_B"),
    )
    aligned, rejected, stamp = align_observations_to_common_time(
        [old, new],
        [(10.0, (0.0, 0.0, 0.0)), (10.1, (0.02, 0.0, 0.0))],
        max_pose_delta_s=0.01,
        drift_std_m_per_s=0.1,
    )

    assert rejected == []
    assert stamp == pytest.approx(10.1)
    assert aligned[0].xy_m == pytest.approx((1.02, 2.0))
    assert aligned[0].timestamp_s == pytest.approx(10.1)
    assert aligned[0].covariance_m2[0][0] == pytest.approx(0.0101)





def test_fusion_is_prior_free_and_rejects_a_metric_outlier():
    """The gate is shared method: it holds for every rule, and needs no prior."""
    from reliability.fusion import MapObservation
    from reliability.nodes.camera_manager_node import _gated_fusion

    def observation(camera_id, x):
        return MapObservation(
            camera_id=camera_id,
            timestamp_s=1.0,
            xy_m=(x, 0.0),
            covariance_m2=((0.04, 0.0), (0.0, 0.04)),
            quality=CameraQuality(
                camera_id=camera_id,
                p_available=0.9,
                conditional_cov_uv=((6.25, 0.0), (0.0, 6.25)),
                association_confidence=0.9,
            ),
        )

    result = _gated_fusion(
        [
            observation("camera_A", 0.0),
            observation("camera_B", 0.2),
            observation("camera_C", 5.0),
        ],
        disagreement_gate_m=0.6,
        rule="independent",
    )
    assert result.accepted_camera_ids == ("camera_A", "camera_B")
    assert result.rejected_camera_ids == ("camera_C",)
    assert result.mean_xy == pytest.approx((0.1, 0.0))
    assert result.covariance_m2[0] == pytest.approx((0.02, 0.0))
    assert result.covariance_m2[1] == pytest.approx((0.0, 0.02))


def test_fusion_candidates_share_manager_gates_and_drop_cached_skewed_views():
    from reliability.camera_manager import CameraManager, CameraManagerConfig
    from reliability.fusion import MapObservation
    from reliability.nodes.camera_manager_node import _synchronous_fusion_candidates

    def map_obs(camera_id, timestamp_s, *, association=0.9):
        return MapObservation(
            camera_id=camera_id,
            timestamp_s=timestamp_s,
            xy_m=(0.0, 0.0),
            covariance_m2=((0.1, 0.0), (0.0, 0.1)),
            quality=CameraQuality(
                camera_id=camera_id,
                p_available=0.9,
                conditional_cov_uv=((1.0, 0.0), (0.0, 1.0)),
                association_confidence=association,
                epistemic_score=0.0,
                source_model="test",
            ),
        )

    manager = CameraManager(CameraManagerConfig(
        min_spatial_trust=0.1,
        min_association_confidence=0.5,
        max_measurement_age_s=1.0,
        age_decay_s=1.0,
    ))
    fresh, scores, rejected, skewed = _synchronous_fusion_candidates(
        manager,
        now_s=10.0,
        observations=[
            map_obs("camera_A", 10.0),
            map_obs("camera_B", 9.6),
            map_obs("camera_C", 10.0, association=0.2),
        ],
        max_timestamp_spread_s=0.25,
    )

    assert [obs.camera_id for obs in fresh] == ["camera_A"]
    assert scores["camera_A"] > 0.0
    assert skewed == ["camera_B"]
    assert "association_below_minimum" in rejected["camera_C"]


def test_projection_is_plain_inverse_perspective_mapping():
    """The deployed measurement is a ray-plane intersection and nothing else.

    Evidence for having no parameters at all: on 1844 real detections every fitted
    correction scored worse than none (raw IPM 66.6 mm, v4 70.1 mm, v3 74.5 mm, v2
    68.2 mm) -- logs/studies/pixel_ground_path/e7_ipm_zero_parameter/RESULTS.md.
    A regression that reintroduces a contact plane or a per-camera offset fails here.
    """

    import inspect

    from reliability import projection as projection_module

    model = camera_model_from_world(WORLD_SDF, include_name="external_camera_c")
    observation = _observation("camera_C", u=640.0, v=500.0)
    assert project_observation_to_world(observation, model) == pytest.approx(
        model.pixel_to_world(640.0, 500.0)
    )

    signature = inspect.signature(project_observation_to_world)
    assert list(signature.parameters) == ["observation", "camera"]

    for deleted in (
        "_project_pixel_to_world",
        "load_projection_calibration",
        "load_projection_contact_z",
        "projection_kwargs_for_camera",
    ):
        assert not hasattr(projection_module, deleted), (
            f"{deleted} was deleted on 2026-08-07; reintroducing it re-opens a "
            "measured-harmful degree of freedom"
        )


# --------------------------------------------------------------------------
# Starting the belief: the only gate that runs without a prior
# --------------------------------------------------------------------------

def _bootstrap_observation(camera_id, x, y=0.0):
    from reliability.fusion import MapObservation

    return MapObservation(
        camera_id=camera_id,
        timestamp_s=1.0,
        xy_m=(float(x), float(y)),
        covariance_m2=((0.04, 0.0), (0.0, 0.04)),
        quality=CameraQuality(camera_id=camera_id),
    )


def test_bootstrap_admits_the_disagreement_the_observation_model_itself_creates():
    """Two CORRECT readings of the same robot are not in the same place.

    The box bottom-centre lands short along each camera's own bearing by a
    commissioned 0.204-0.456 m, so two cameras can put the same stationary robot
    0.91 m apart with no error at all. Measured on the frozen capture, the
    closest-agreeing pair at a position is a median 0.27 m apart. A gate tighter
    than the model's own spread rejects the truth and the robot never starts.
    """
    from reliability.nodes.camera_manager_node import _largest_agreeing_group

    two_correct_readings = [
        _bootstrap_observation("camera_A", 0.0),
        _bootstrap_observation("camera_B", 0.6),
    ]
    assert len(_largest_agreeing_group(two_correct_readings, 0.91)) == 2
    assert len(_largest_agreeing_group(two_correct_readings, 0.30)) == 0


def test_bootstrap_takes_the_agreeing_group_and_leaves_the_odd_camera_out():
    """One camera locked onto something else must not block start-up.

    Requiring every camera in view to agree makes a single mis-association a
    deadlock: no belief, so no prediction, so no correction, so no belief.
    """
    from reliability.nodes.camera_manager_node import _largest_agreeing_group

    group = _largest_agreeing_group(
        [
            _bootstrap_observation("camera_A", 0.0),
            _bootstrap_observation("camera_B", 0.3),
            _bootstrap_observation("camera_C", 0.5),
            _bootstrap_observation("camera_D", 7.0),   # a different aisle
        ],
        0.91,
    )
    assert sorted(o.camera_id for o in group) == ["camera_A", "camera_B", "camera_C"]


def test_bootstrap_refuses_a_single_camera_however_confident():
    """The belief may never be started on one camera's unchecked word."""
    from reliability.nodes.camera_manager_node import _largest_agreeing_group

    assert _largest_agreeing_group([_bootstrap_observation("camera_A", 0.0)], 0.91) == []


def test_bootstrap_group_choice_does_not_depend_on_arrival_order():
    from reliability.nodes.camera_manager_node import _largest_agreeing_group

    readings = [
        _bootstrap_observation("camera_A", 0.0),
        _bootstrap_observation("camera_B", 0.4),
        _bootstrap_observation("camera_C", 9.0),
        _bootstrap_observation("camera_D", 9.3),
    ]
    first = sorted(o.camera_id for o in _largest_agreeing_group(readings, 0.91))
    second = sorted(
        o.camera_id for o in _largest_agreeing_group(list(reversed(readings)), 0.91)
    )
    assert first == second
