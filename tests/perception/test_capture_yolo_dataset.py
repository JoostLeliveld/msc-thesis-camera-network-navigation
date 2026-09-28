from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / 'pipeline' / 'capture'
SRC_UNAV = REPO_ROOT / 'src' / 'unav_common'
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_UNAV) not in sys.path:
    sys.path.insert(0, str(SRC_UNAV))

from capture_yolo_dataset import (
    _CaptureOutputGuard,
    _build_simulation_asset_inventory,
    _canonical_inventory_sha256,
    _capture_transport_environment,
    _filter_projectable_pose_records,
    _filter_pose_records,
    _mask_area_gate_for_range,
    _project_robot_bbox,
    _route_exclusion_segments,
    validate_sample_quality,
)
from unav_common.camera_model import ObliqueCameraModel
from unav_common.occlusion_geometry import AxisAlignedPrism


def _camera() -> ObliqueCameraModel:
    return ObliqueCameraModel(
        cam_pos=(0.0, -5.0, 4.0),
        look_at=(0.0, 0.0, 0.0),
        img_width=640,
        img_height=360,
        fov_h_rad=1.5708,
    )


def _base_kwargs(camera: ObliqueCameraModel) -> dict:
    return {
        'robot_label': 23,
        'epsilon_ratio': 0.01,
        'bottom_band_px': 3.0,
        'min_mask_area': 20.0,
        'min_mask_bbox_w': 4.0,
        'min_mask_bbox_h': 4.0,
        'max_mask_border_fraction': 0.0,
        'min_rgb_robot_color_fraction': 0.01,
        'disable_rgb_color_check': False,
        'robot_color_profile': 'label_vs_background',
        'camera': camera,
        'x': 0.0,
        'y': 0.0,
        'yaw': 0.0,
        'robot_z': 0.05,
        'box_length': 0.22,
        'box_width': 0.22,
        'box_height': 0.20,
        'max_expected_center_error_px': 50.0,
        'min_visible_height_fraction': 0.55,
        'max_bottom_occlusion_px': 20.0,
    }


def _draw_robot_patch(
    *,
    camera: ObliqueCameraModel,
    image_bgr: np.ndarray,
    labels: np.ndarray,
    offset_px: tuple[int, int] = (0, 0),
    draw_rgb: bool = True,
) -> tuple[int, int, int, int]:
    bbox = _project_robot_bbox(
        camera,
        x=0.0,
        y=0.0,
        yaw=0.0,
        z=0.05,
        box_length=0.22,
        box_width=0.22,
        box_height=0.20,
    )
    assert bbox is not None
    dx, dy = offset_px
    x0 = int(round(bbox[0])) + dx
    y0 = int(round(bbox[1])) + dy
    x1 = int(round(bbox[2])) + dx
    y1 = int(round(bbox[3])) + dy
    x0 = max(0, min(labels.shape[1] - 1, x0))
    y0 = max(0, min(labels.shape[0] - 1, y0))
    x1 = max(x0 + 1, min(labels.shape[1], x1))
    y1 = max(y0 + 1, min(labels.shape[0], y1))
    labels[y0:y1, x0:x1] = 23
    if draw_rgb:
        cv2.rectangle(image_bgr, (x0, y0), (x1, y1), (40, 45, 220), thickness=-1)
        cv2.circle(image_bgr, ((x0 + x1) // 2, y0 + 2), 3, (220, 100, 40), thickness=-1)
    return x0, y0, x1, y1


def test_validate_sample_accepts_visible_synchronized_segmentation() -> None:
    camera = _camera()
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    labels = np.zeros((360, 640), dtype=np.uint32)
    _draw_robot_patch(camera=camera, image_bgr=image, labels=labels)

    result = validate_sample_quality(image_bgr=image, labels=labels, **_base_kwargs(camera))

    assert result.accepted
    assert result.reason == ''
    assert result.mask_area_px > 20.0
    assert result.rgb_robot_color_fraction > 0.01
    assert math.isfinite(result.expected_center_error_px)
    assert result.expected_center_error_px < 10.0
    # Fully-visible robot: visible silhouette matches the projected box and the
    # contact row is not occluded.
    assert result.visible_height_fraction > 0.9
    assert abs(result.bottom_occlusion_px) < 5.0


def test_validate_sample_rejects_bottom_occluded_robot() -> None:
    camera = _camera()
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    labels = np.zeros((360, 640), dtype=np.uint32)
    bbox = _project_robot_bbox(
        camera, x=0.0, y=0.0, yaw=0.0, z=0.05,
        box_length=0.22, box_width=0.22, box_height=0.20,
    )
    assert bbox is not None
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    # A foreground rack hides the bottom ~65%: only the top of the robot is labelled,
    # so the visible silhouette is short and its bottom sits well above the true
    # ground-contact row. This is exactly the bad box-bottom label we must drop.
    y_cut = y0 + int(round(0.35 * (y1 - y0)))
    labels[y0:y_cut, x0:x1] = 23
    cv2.rectangle(image, (x0, y0), (x1, y_cut), (40, 45, 220), thickness=-1)

    result = validate_sample_quality(image_bgr=image, labels=labels, **_base_kwargs(camera))

    assert not result.accepted
    assert result.reason in ('occluded_low_visible_height', 'occluded_bottom_hidden')
    assert result.visible_height_fraction < 0.55
    assert result.localization_qualified is False
    assert result.occlusion_state in {'low_visible_height', 'bottom_hidden'}


def test_validate_sample_rejects_projection_mismatch() -> None:
    camera = _camera()
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    labels = np.zeros((360, 640), dtype=np.uint32)
    _draw_robot_patch(camera=camera, image_bgr=image, labels=labels, offset_px=(140, 0))

    result = validate_sample_quality(image_bgr=image, labels=labels, **_base_kwargs(camera))

    assert not result.accepted
    assert result.reason == 'projection_mismatch'


def test_validate_sample_rejects_label_without_visible_rgb_robot() -> None:
    camera = _camera()
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    labels = np.zeros((360, 640), dtype=np.uint32)
    _draw_robot_patch(camera=camera, image_bgr=image, labels=labels, draw_rgb=False)

    result = validate_sample_quality(image_bgr=image, labels=labels, **_base_kwargs(camera))

    assert not result.accepted
    assert result.reason == 'rgb_robot_not_visible'


def test_pose_filter_removes_collision_and_known_non_driveable_positions() -> None:
    records = [
        {'x': 0.0, 'y': 0.0, 'yaw': 0.0},
        {'x': 2.0, 'y': 0.0, 'yaw': 0.0},
        {'x': 4.0, 'y': 0.0, 'yaw': 0.0},
    ]
    prism = AxisAlignedPrism(
        name='rack', xmin=-0.2, xmax=0.2, ymin=-0.2, ymax=0.2, zmin=0.0, zmax=2.0
    )
    kept, counts = _filter_pose_records(
        records,
        traversable_regions=[{'xmin': -5.0, 'xmax': 5.0, 'ymin': -1.0, 'ymax': 1.0}],
        excluded_regions=[{'xmin': 1.8, 'xmax': 2.2, 'ymin': -0.2, 'ymax': 0.2}],
        region_shrink_m=0.0,
        collision_prisms=(prism,),
        collision_clearance_m=0.25,
        camera_xy=(0.0, -10.0),
        min_camera_range_m=0.0,
        max_camera_range_m=20.0,
    )

    assert kept == [records[2]]
    assert counts['collision_clearance'] == 1
    assert counts['known_non_driveable_region'] == 1


def test_pose_filter_applies_camera_range_stratum() -> None:
    records = [
        {'x': 0.0, 'y': 0.0, 'yaw': 0.0},
        {'x': 0.0, 'y': 10.0, 'yaw': 0.0},
        {'x': 0.0, 'y': 20.0, 'yaw': 0.0},
    ]
    kept, counts = _filter_pose_records(
        records,
        traversable_regions=[],
        excluded_regions=[],
        region_shrink_m=0.0,
        collision_prisms=(),
        collision_clearance_m=0.0,
        camera_xy=(0.0, 0.0),
        min_camera_range_m=5.0,
        max_camera_range_m=16.0,
    )

    assert kept == [records[1]]
    assert counts['below_min_camera_range'] == 1
    assert counts['above_max_camera_range'] == 1


def test_projection_filter_keeps_border_hard_cases_but_removes_offscreen_poses() -> None:
    camera = _camera()
    records = [
        {'x': 0.0, 'y': 0.0, 'yaw': 0.0},
        {'x': 50.0, 'y': 0.0, 'yaw': 0.0},
        {'x': 0.0, 'y': -20.0, 'yaw': 0.0},
    ]

    kept, counts = _filter_projectable_pose_records(
        records,
        camera=camera,
        image_width=640,
        image_height=360,
        robot_z=0.05,
        box_length=0.22,
        box_width=0.22,
        box_height=0.20,
    )

    assert kept == [records[0]]
    assert counts['kept'] == 1
    assert counts['projection_outside_image'] + counts['projection_behind_camera'] == 2


def test_far_range_mask_gate_preserves_small_object_examples_explicitly() -> None:
    kwargs = {'near_min_area_px': 80.0, 'far_start_m': 12.0, 'far_min_area_px': 40.0}

    assert _mask_area_gate_for_range(11.99, **kwargs) == 80.0
    assert _mask_area_gate_for_range(12.0, **kwargs) == 40.0
    assert _mask_area_gate_for_range(16.0, **kwargs) == 40.0


def test_detector_training_route_exclusion_includes_every_lateral_variant() -> None:
    study = REPO_ROOT / 'pipeline/capture/multicamera_study.yaml'
    segments = _route_exclusion_segments(
        study,
        route_names=['south_to_north_handover'],
    )

    # The northbound route's +0.5 m left offset is x=-2.0; all three variants
    # are reserved from the detector train/val grid.
    assert len(segments) == 3
    assert sorted(round(segment[0], 2) for segment in segments) == [-2.0, -1.5, -1.0]


def test_capture_source_does_not_retry_deterministic_label_quality_failures() -> None:
    source = (SCRIPT_DIR / 'capture_yolo_dataset.py').read_text(encoding='utf-8')

    assert 'label-quality failures at an exact' in source
    assert 'Only capture/synchronization exceptions' in source


def test_observation_capture_retains_rejected_rgb_frames_without_making_them_training_samples() -> None:
    source = (SCRIPT_DIR / 'capture_yolo_dataset.py').read_text(encoding='utf-8')

    assert '--save-observation-frames' in source
    assert "out_dir / 'observation_frames'" in source
    assert "'observation_image': observation_image_rel" in source
    assert 'including samples rejected by the semantic-label quality gate' in source


def test_capture_transport_contract_requires_local_isolation_and_marks_override_diagnostic() -> None:
    isolated = {
        'ROS_LOCALHOST_ONLY': '1',
        'IGN_IP': '127.0.0.1',
        'GZ_IP': '127.0.0.1',
        'ROS_DOMAIN_ID': '79',
        'IGN_PARTITION': 'fourcam_capture_A_001',
    }

    contract = _capture_transport_environment(isolated)

    assert contract['training_eligible'] is True
    assert contract['isolation_verified'] is True
    assert contract['diagnostic_override_used'] is False
    assert contract['violations'] == []
    assert contract['observed_values'] == isolated

    unisolated = dict(isolated, IGN_IP='0.0.0.0')
    with pytest.raises(RuntimeError, match='Training-grade capture requires isolated'):
        _capture_transport_environment(unisolated)

    diagnostic = _capture_transport_environment(
        unisolated,
        allow_unisolated_transport=True,
    )
    assert diagnostic['training_eligible'] is False
    assert diagnostic['isolation_verified'] is False
    assert diagnostic['diagnostic_override_used'] is True
    assert any('IGN_IP' in violation for violation in diagnostic['violations'])


def test_capture_inventory_fingerprints_world_models_launch_and_robot_assets() -> None:
    inventory = _build_simulation_asset_inventory(
        world_path=REPO_ROOT / 'src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf',
        world_profiles_path=REPO_ROOT / 'src/experiments/config/world_profiles.yaml',
        route_exclusion_config_path=(
            REPO_ROOT
            / 'pipeline/capture/multicamera_study.yaml'
        ),
    )

    assert inventory['file_count'] == len(inventory['files'])
    assert inventory['aggregate_sha256'] == _canonical_inventory_sha256(
        inventory['files']
    )
    roles = {
        role
        for entry in inventory['files']
        for role in entry['roles']
    }
    assert {
        'capture_script',
        'world',
        'world_profiles',
        'route_exclusion_config',
        'sim_launch',
        'robot_description',
    } <= roles
    assert {
        'external_camera',
        'external_camera_b',
        'external_camera_c',
        'external_camera_d',
    } <= set(inventory['referenced_model_names'])
    assert all(
        f'model_asset:{model_name}' in roles
        for model_name in inventory['referenced_model_names']
    )


def test_interrupted_capture_is_atomically_quarantined_with_files_preserved(
    tmp_path: Path,
) -> None:
    output = tmp_path / 'camera_C'
    (output / 'images/train').mkdir(parents=True)
    payload = output / 'images/train/sample_000000.png'
    payload.write_bytes(b'partial-image')
    guard = _CaptureOutputGuard(output, camera_id='camera_C')

    quarantined = guard.quarantine(KeyboardInterrupt())

    assert quarantined is not None
    assert not output.exists()
    assert (quarantined / 'images/train/sample_000000.png').read_bytes() == b'partial-image'
    state = __import__('json').loads(
        (quarantined / '.capture_failed.json').read_text(encoding='utf-8')
    )
    assert state['status'] == 'interrupted'
    assert state['training_eligible'] is False
    assert not (quarantined / '.complete').exists()
    assert guard.quarantine(RuntimeError('second call')) == quarantined


def test_capture_completion_marker_hashes_manifest_and_clears_in_progress(
    tmp_path: Path,
) -> None:
    output = tmp_path / 'camera_A'
    output.mkdir()
    guard = _CaptureOutputGuard(output, camera_id='camera_A')
    manifest = output / 'dataset_manifest.json'
    manifest.write_text('{"status":"complete"}\n', encoding='utf-8')

    guard.complete(manifest)

    completion = __import__('json').loads(
        (output / '.complete').read_text(encoding='utf-8')
    )
    assert completion['status'] == 'complete'
    assert completion['training_eligible'] is True
    assert len(completion['dataset_manifest_sha256']) == 64
    assert not (output / '.capture_in_progress.json').exists()


def test_diagnostic_transport_completion_marker_is_not_training_eligible(
    tmp_path: Path,
) -> None:
    output = tmp_path / 'camera_A_diagnostic'
    output.mkdir()
    guard = _CaptureOutputGuard(
        output,
        camera_id='camera_A',
        training_eligible=False,
    )
    manifest = output / 'dataset_manifest.json'
    manifest.write_text('{"status":"complete"}\n', encoding='utf-8')

    guard.complete(manifest)

    completion = __import__('json').loads(
        (output / '.complete').read_text(encoding='utf-8')
    )
    assert completion['training_eligible'] is False


def test_grey_robot_passes_the_livery_agnostic_gate_but_not_the_marker_gate() -> None:
    """The regression test for a capture pipeline that rejected its own robot.

    Every dataset before 2026-08-20 was captured with a red-and-blue TurtleBot3, so
    the RGB support gate looked for red or blue pixels. ``warehouse_amr`` is charcoal,
    grey and hazard yellow, and measured on real warehouse_v2 frames the red-or-blue
    fraction inside its label box is 0.0000 to 0.0034 -- below the 0.015 default, so
    the gate would have thrown away essentially every sample of the robot it was
    supposed to be capturing. The gate that survives changing the robot compares the
    pixels under the label with the background around it instead.
    """
    camera = _camera()
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    image[:, :] = (150, 152, 154)                      # lit concrete, no red, no blue
    labels = np.zeros((360, 640), dtype=np.uint32)
    x0, y0, x1, y1 = _draw_robot_patch(
        camera=camera, image_bgr=image, labels=labels, draw_rgb=False
    )
    cv2.rectangle(image, (x0, y0), (x1, y1), (44, 42, 40), thickness=-1)   # charcoal

    agnostic = validate_sample_quality(
        image_bgr=image, labels=labels,
        **{**_base_kwargs(camera), 'robot_color_profile': 'label_vs_background'},
    )
    assert agnostic.accepted, agnostic.reason

    marker = validate_sample_quality(
        image_bgr=image, labels=labels,
        **{**_base_kwargs(camera), 'robot_color_profile': 'marker_disks_red_blue'},
    )
    assert not marker.accepted
    assert marker.reason == 'rgb_robot_not_visible'


def test_marker_livery_still_passes_its_own_profile() -> None:
    """The old profile must keep working, so an old capture stays reproducible."""
    camera = _camera()
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    labels = np.zeros((360, 640), dtype=np.uint32)
    _draw_robot_patch(camera=camera, image_bgr=image, labels=labels)

    result = validate_sample_quality(
        image_bgr=image, labels=labels,
        **{**_base_kwargs(camera), 'robot_color_profile': 'marker_disks_red_blue'},
    )
    assert result.accepted, result.reason
    assert result.rgb_robot_color_fraction > 0.01
