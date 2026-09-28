"""Apply the packaged box-feature bias correction to a live camera reading.

This is the deployment side of `experiments/camera_observation_characterization/`. The
model was fitted offline on a spatial checkerboard split of the frozen characterization
capture; here it only runs forward.

Two properties matter for it to be a legitimate runtime model:

*   Every input is available online. The model consumes the raw back-projection, the
    detected box, the detector's confidence and the camera's identity and pose. It does
    NOT consume the robot's position, its heading, the true range, or any segmentation --
    the packaged artifact lists those as excluded, and this module cannot supply them.
*   The features are built by the SAME code path the fit used. Rather than restating the
    feature vector here, the ordering is checked against the artifact's own
    `feature_names` on load, so a change to either side fails loudly instead of silently
    feeding the network a permuted input.

The correction is predicted in the camera-ray frame -- along the ray away from the camera,
and to its left -- because that is the frame the residual was regressed in. Applying it in
world coordinates directly would rotate the correction by the camera's bearing.
"""
from __future__ import annotations

import math
import hashlib
import io
from pathlib import Path
from typing import Sequence

FEATURE_NAMES: tuple[str, ...] = (
    'range_m', 'inv_range', 'box_w_frac', 'box_h_frac', 'box_aspect',
    'u_frac', 'v_frac', 'bearing_cos', 'bearing_sin', 'confidence',
)

#: Schema the loader understands. A different schema means the feature contract may have
#: changed, so the model is refused rather than run on a guess.
SUPPORTED_SCHEMA = 'box_feature_bias_correction.joblib.v1'


class LearnedBoxCorrection:
    """Forward-only wrapper around the packaged neural box correction."""

    def __init__(self, artifact_path: str | Path, *, expected_sha256: str | None = None) -> None:
        import joblib  # imported here so the runtime only needs it when this model is used

        path = Path(artifact_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f'learned box-correction artifact not found: {path}')
        data = path.read_bytes()
        self.sha256 = hashlib.sha256(data).hexdigest()
        if expected_sha256 is not None and self.sha256 != expected_sha256:
            raise ValueError('learned correction artifact hash differs from expected identity')
        payload = joblib.load(io.BytesIO(data))

        schema = str(payload.get('schema', ''))
        if schema != SUPPORTED_SCHEMA:
            raise ValueError(
                f'learned box correction has schema {schema!r}, this runtime understands '
                f'{SUPPORTED_SCHEMA!r}; refusing to run a model whose feature contract '
                f'may differ')

        model = payload.get('neural_model')
        if model is None or not hasattr(model, 'predict'):
            raise ValueError('artifact carries no usable neural_model')

        camera_ids = [str(value) for value in payload.get('camera_ids') or []]
        if not camera_ids:
            raise ValueError('artifact carries no camera_ids, so identity cannot be encoded')
        if len(set(camera_ids)) != len(camera_ids) or any(not c for c in camera_ids):
            raise ValueError('artifact camera_ids must be unique nonempty identities')
        if payload.get('target') != 'truth minus raw projection in along/across camera-ray coordinates':
            raise ValueError('learned correction target/reference semantics are unsupported')

        # The artifact's own ordering is authoritative. Check it rather than trusting that
        # this file and the fit have stayed in step.
        expected = list(FEATURE_NAMES) + [f'is_{camera}' for camera in camera_ids]
        recorded = [str(value) for value in payload.get('feature_names') or []]
        if recorded != expected:
            raise ValueError(
                'learned box-correction feature order does not match this runtime.\n'
                f'  artifact: {recorded}\n  runtime : {expected}')

        geometry = payload.get('camera_geometry') or {}
        if not geometry:
            raise ValueError('artifact carries no camera_geometry')
        if set(geometry) != set(camera_ids):
            raise ValueError('camera geometry must match exactly the trained camera registry')

        self.path = path
        self.model = model
        self.camera_ids = camera_ids
        self.target = str(payload.get('target', ''))
        self._geometry = {
            str(camera): {
                'xy': (float(entry['xy'][0]), float(entry['xy'][1])),
                'yaw': float(entry['yaw']),
                'width': float(entry['width']),
                'height': float(entry['height']),
            }
            for camera, entry in geometry.items()
        }
        for camera, entry in self._geometry.items():
            values = [*entry['xy'], entry['yaw'], entry['width'], entry['height']]
            if not all(math.isfinite(v) for v in values) or min(entry['width'], entry['height']) <= 0:
                raise ValueError(f'nonfinite or invalid learned camera geometry: {camera}')

    # -- feature construction -------------------------------------------------
    def _features(self, camera_id: str, raw_xy: Sequence[float],
                  bbox_xyxy: Sequence[float], confidence: float) -> list[float] | None:
        geometry = self._geometry.get(camera_id)
        if geometry is None:
            return None
        camera_xy = geometry['xy']
        dx = float(raw_xy[0]) - camera_xy[0]
        dy = float(raw_xy[1]) - camera_xy[1]
        distance = math.hypot(dx, dy)
        if not math.isfinite(distance) or distance <= 0.0:
            return None
        bearing = math.atan2(dy, dx) - geometry['yaw']

        width, height = geometry['width'], geometry['height']
        box_w = (float(bbox_xyxy[2]) - float(bbox_xyxy[0])) / width
        box_h = (float(bbox_xyxy[3]) - float(bbox_xyxy[1])) / height
        if not (math.isfinite(box_w) and math.isfinite(box_h)):
            return None
        # The bottom-centre pixel, which is the point the raw projection came from.
        u_bottom = 0.5 * (float(bbox_xyxy[0]) + float(bbox_xyxy[2]))
        v_bottom = float(bbox_xyxy[3])

        row = [
            distance,
            1.0 / max(distance, 1e-3),
            box_w,
            box_h,
            box_w * width / max(box_h * height, 1.0),
            u_bottom / width,
            v_bottom / height,
            math.cos(bearing),
            math.sin(bearing),
            float(confidence),
        ]
        row.extend(1.0 if camera_id == candidate else 0.0 for candidate in self.camera_ids)
        if not all(math.isfinite(value) for value in row):
            return None
        return row

    # -- public API -----------------------------------------------------------
    def correct(self, camera_id: str, raw_xy: Sequence[float],
                bbox_xyxy: Sequence[float] | None,
                confidence: float) -> tuple[float, float] | None:
        """Return the corrected ground point, or None if the reading cannot be corrected.

        Returning None is deliberate: a reading the model cannot describe is left to the
        caller to handle, rather than silently passed through as if it had been corrected.
        """
        if bbox_xyxy is None or len(bbox_xyxy) != 4 or raw_xy is None or len(raw_xy) != 2:
            return None
        try:
            if not all(math.isfinite(float(v)) for v in (*raw_xy, *bbox_xyxy, confidence)):
                return None
            if not 0 <= float(confidence) <= 1 or float(bbox_xyxy[2]) <= float(bbox_xyxy[0]) or float(bbox_xyxy[3]) <= float(bbox_xyxy[1]):
                return None
            row = self._features(camera_id, raw_xy, bbox_xyxy, confidence)
        except (ValueError, TypeError, IndexError, OverflowError):
            return None
        if row is None:
            return None
        try:
            predicted = self.model.predict([row])[0]
        except Exception:
            return None
        along, across = float(predicted[0]), float(predicted[1])
        if not (math.isfinite(along) and math.isfinite(across)):
            return None

        # The residual was regressed in the camera-ray frame, so it is applied there.
        camera_xy = self._geometry[camera_id]['xy']
        dx = float(raw_xy[0]) - camera_xy[0]
        dy = float(raw_xy[1]) - camera_xy[1]
        norm = math.hypot(dx, dy)
        if norm <= 1e-12:
            return None
        unit = (dx / norm, dy / norm)
        left = (-unit[1], unit[0])
        return (
            float(raw_xy[0]) + along * unit[0] + across * left[0],
            float(raw_xy[1]) + along * unit[1] + across * left[1],
        )
