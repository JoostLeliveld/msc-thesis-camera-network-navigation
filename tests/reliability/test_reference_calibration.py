"""Calibration contract and actual camera-manager ordering against recorded samples."""
import collections
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/icra_commissioning'))
sys.path.insert(0, str(ROOT / 'experiments/camera_observation_characterization'))
from reliability.reference_calibration import ReferenceCalibration

ARTIFACT = ROOT / 'logs/studies/icra_commissioning_20260905/network_planner/reference_calibration.json'
MEAN = ROOT / 'logs/perception_models/box_feature_bias_correction_20260831/models.joblib'


@pytest.fixture
def small(tmp_path):
    mean = tmp_path / 'mean'; mean.write_bytes(b'frozen mean')
    data = dict(schema='camera_reference_calibration.v1', frame='map_bev',
                reference='robot_ground_reference_xy', covariance_units='m2',
                mean_order='bbox_feature_nn_then_subtract_bias',
                mean_checkpoint_sha256=hashlib.sha256(mean.read_bytes()).hexdigest(),
                cameras={'camera_A': dict(bias_m=[.1, -.2], R_m2=[[.04, .01], [.01, .09]])})
    path = tmp_path / 'calibration.json'; path.write_text(json.dumps(data))
    return path, mean, data


@pytest.mark.parametrize('field,value', [
    ('frame', 'camera'), ('covariance_units', 'pixels2'), ('reference', 'bbox_bottom'),
    ('mean_order', 'bias_then_nn'), ('mean_checkpoint_sha256', 'wrong')])
def test_rejects_incompatible_measurement(small, field, value):
    path, mean, data = small
    data[field] = value; path.write_text(json.dumps(data))
    with pytest.raises(ValueError): ReferenceCalibration(path, mean, ['camera_A'])


@pytest.mark.parametrize('R', [[[1, 2], [2, 1]], [[1, .2], [.3, 1]], [[0, 0], [0, 0]], [[1, 0], [0, float('nan')]]])
def test_rejects_invalid_covariance(small, R):
    path, mean, data = small
    data['cameras']['camera_A']['R_m2'] = R; path.write_text(json.dumps(data))
    with pytest.raises(ValueError): ReferenceCalibration(path, mean, ['camera_A'])


def test_missing_camera_and_incorrect_checkpoint_are_refused(small):
    path, mean, _ = small
    with pytest.raises(ValueError, match='missing'): ReferenceCalibration(path, mean, ['camera_B'])
    mean.write_bytes(b'changed mean')
    with pytest.raises(ValueError, match='hash'): ReferenceCalibration(path, mean, ['camera_A'])


@pytest.mark.skipif(not ARTIFACT.exists(), reason='local frozen calibration unavailable')
def test_actual_manager_matches_frozen_offline_mean_and_R(monkeypatch):
    import joblib
    import reliability.nodes.camera_manager_node as manager
    from reliability.contracts import CameraObservation
    from reliability.learned_box_correction import LearnedBoxCorrection
    from reliability.manager_state import ManagerInputs
    from derive_interpretations import camera_models
    models = joblib.load(ROOT / 'logs/studies/icra_commissioning_20260905/models.joblib')
    calibration = ReferenceCalibration(ARTIFACT, MEAN, [f'camera_{c}' for c in 'ABCDE'])
    nn = LearnedBoxCorrection(MEAN)
    geometry = camera_models(json.loads((ROOT / 'logs/perception_datasets/warehouse_v2_bbox_characterization_20260831/capture_manifest.json').read_text()))
    path = ROOT / 'logs/perception_datasets/warehouse_v2_bbox_characterization_20260831/bias_update_interpretations.csv'
    with path.open() as stream:
        # Keep camera support in this equality check; no performance estimate is made.
        rows = [r for r in csv.DictReader(stream) if r['nn_valid'] == '1' and r['split'] == 'test']
    chosen = []
    for camera in calibration.bias:
        chosen.extend([r for r in rows if r['camera_id'] == camera][:20])
    assert len(chosen) == 100
    monkeypatch.setattr(manager, '_with_provider_quality', lambda observation, *args: observation)
    for row in chosen:
        camera = row['camera_id']
        raw = (float(row['raw_x']), float(row['raw_y']))
        monkeypatch.setattr(manager, 'project_observation_to_world_with_covariance',
                            lambda *args: (raw, ((1., 0.), (0., 1.))))
        # Valid contract, with no robot truth or reference fields in deployed inputs.
        contract = CameraObservation(camera_id=camera, timestamp_s=1., detection_valid=True,
            source_batch_id='frozen_fixture', calibration_id=camera, image_frame_id=camera,
            pixel_uv=(float(row['u_bbox_bottom']), float(row['v_bbox_bottom'])),
            bbox_bottom_uv=(float(row['u_bbox_bottom']), float(row['v_bbox_bottom'])),
            selected_pixel_source='bbox_bottom',
            detector_score=float(row['confidence']), detector_score_raw=float(row['confidence']),
            bbox_xyxy=tuple(float(row[k]) for k in ('x0', 'y0', 'x1', 'y1')))
        fake = SimpleNamespace(_latest={camera: contract}, commissioned_pixel_cov_by_camera={},
            commissioned_pixel_cov=((1., 0.), (0., 1.)), camera_models={camera: geometry[camera]},
            camera_calibration_ids_by_camera={camera: camera}, camera_image_frame_ids_by_camera={camera: camera},
            _decision_snapshot=ManagerInputs('frozen_fixture', (contract,), (), False, None, None),
            covariance_profile=manager.COMMISSIONED_REFERENCE_COVARIANCE, _belief_query_history=[],
            reliability_query_max_time_delta_s=.2, admission_gate=False, silhouette_correction=False,
            observation_model=manager.OBSERVATION_MODEL_LEARNED_NN, learned_correction=nn,
            reference_calibration=calibration, _gate_rejections=collections.Counter(),
            _silhouette_status_by_camera={}, _detection_extras_by_camera={},
            _reliability_query_source_by_camera={}, replay_config=None)
        observations = manager.CameraManagerNode._map_observations(fake, 1.)
        assert len(observations) == 1
        expected_z, expected_R = models[camera, 'constant'].predict([
            dict(z=np.array([float(row['nn_x']), float(row['nn_y'])]))])
        np.testing.assert_allclose(observations[0].xy_m, expected_z[0], atol=1e-9)
        np.testing.assert_allclose(observations[0].covariance_m2, expected_R[0], atol=1e-12)
        assert fake._detection_extras_by_camera[camera]['raw_obs_x'] == raw[0]
