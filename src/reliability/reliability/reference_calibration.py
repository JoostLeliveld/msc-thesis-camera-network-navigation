"""Frozen residual calibration AFTER the bbox-feature reference-position NN.

This defines a measurement of reference XY in map_bev, metres, h(x)=[x,y].
It is a conditional residual covariance, not a fusion posterior or score field.
"""
from pathlib import Path
import hashlib
import json
import re
import numpy as np


class ReferenceCalibration:
    def __init__(self, path, mean_checkpoint, camera_ids, *, expected_sha256=None,
                 expected_source_hashes=None, loaded_mean_sha256=None):
        self.path = Path(path).resolve()
        encoded = self.path.read_bytes()
        self.sha256 = hashlib.sha256(encoded).hexdigest()
        if expected_sha256 is not None and self.sha256 != expected_sha256:
            raise ValueError('reference calibration artifact hash differs')
        data = json.loads(encoded)
        required = dict(schema='camera_reference_calibration.v1',
                        frame='map_bev', reference='robot_ground_reference_xy',
                        covariance_units='m2', mean_order='bbox_feature_nn_then_subtract_bias')
        for key, value in required.items():
            if data.get(key) != value:
                raise ValueError(f'reference calibration {key} must be {value}')
        # Prefer the identity of bytes the mean wrapper actually deserialized.
        mean_hash = (loaded_mean_sha256 or getattr(mean_checkpoint, 'sha256', None)
                     or hashlib.sha256(Path(mean_checkpoint).read_bytes()).hexdigest())
        if data.get('mean_checkpoint_sha256') != mean_hash:
            raise ValueError('reference calibration mean checkpoint hash differs')
        self.bias, self.covariance = {}, {}
        camera_ids = tuple(camera_ids)
        if not camera_ids or len(set(camera_ids)) != len(camera_ids):
            raise ValueError('reference camera request must contain unique camera IDs')
        for camera in camera_ids:
            if camera not in data['cameras']:
                raise ValueError(f'reference calibration missing {camera}')
            entry = data['cameras'][camera]
            bias, R = np.asarray(entry['bias_m'], float), np.asarray(entry['R_m2'], float)
            if bias.shape != (2,) or R.shape != (2, 2) or not np.isfinite(bias).all() or not np.isfinite(R).all():
                raise ValueError(f'nonfinite or wrong-shaped calibration for {camera}')
            if not np.allclose(R, R.T, atol=1e-12, rtol=0):
                raise ValueError(f'non-symmetric covariance for {camera}')
            try:
                np.linalg.cholesky(R)
            except np.linalg.LinAlgError as exc:
                raise ValueError(f'non-positive-definite covariance for {camera}') from exc
            self.bias[camera], self.covariance[camera] = bias, R
        sources = data.get('source_hashes')
        if not isinstance(sources, dict) or not sources or any(
            not isinstance(k, str) or not k or not isinstance(v, str) or not re.fullmatch('[0-9a-f]{64}', v)
            for k, v in sources.items()):
            raise ValueError('reference calibration source hashes are missing or malformed')
        if expected_source_hashes is not None and sources != expected_source_hashes:
            raise ValueError('reference calibration source hashes differ from expected lineage')
        # Immutable inputs must still match. Source-code hashes are archived lineage;
        # they are not a demand that a repaired checkout equal historical software.
        root = Path(__file__).resolve().parents[3]
        for relative, expected in sources.items():
            if Path(relative).suffix in ('.json', '.joblib', '.npz', '.csv'):
                source = root / relative
                if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                    raise ValueError(f'reference calibration source artifact hash differs: {relative}')
        self.source_hashes = dict(sources)
        self.metadata = data

    def apply(self, camera_id, nn_xy):
        """Call once on a fresh NN reference reading, before any robot update."""
        if camera_id not in self.bias:
            raise ValueError(f'unsupported reference camera: {camera_id}')
        z = np.asarray(nn_xy, float)
        if z.shape != (2,) or not np.isfinite(z).all():
            raise ValueError('reference reading must be finite XY metres')
        z = z - self.bias[camera_id]
        return tuple(z), tuple(tuple(row) for row in self.covariance[camera_id])
