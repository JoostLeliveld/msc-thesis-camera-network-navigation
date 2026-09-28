"""Legacy compatibility reader for superseded availability artifacts.

The canonical thesis method does not instantiate this model. New campaigns use
``reliability.planning_information`` and direct per-camera expected information.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from reliability.bernoulli_gp import model_from_serialized, predict_laplace


SCHEMA = "thesis_commissioned_availability_model.v2"
FEATURE_NAMES = ("robot_x_m", "robot_y_m")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CommissionedAvailabilityModel:
    """Validate and query a frozen Bernoulli-GP field using camera ID and position only."""

    def __init__(
        self,
        artifact_path: str | Path,
        *,
        expected_sha256: str | None = None,
    ) -> None:
        path = Path(artifact_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"availability artifact not found: {path}")
        self.sha256 = _sha256(path)
        if expected_sha256 and self.sha256 != expected_sha256:
            raise ValueError("commissioned availability artifact hash differs from expected identity")
        payload = json.loads(path.read_text())
        if payload.get("schema") != SCHEMA:
            raise ValueError("availability must use the position-only Bernoulli-GP schema")
        if payload.get("final_audit_accessed") is not False:
            raise ValueError("availability artifact must be fitted without final-audit access")
        if tuple(payload.get("feature_names") or ()) != FEATURE_NAMES:
            raise ValueError("availability inputs must be exactly ground-plane x and y")
        if (payload.get("likelihood"), payload.get("inference"), payload.get("kernel")) != (
            "Bernoulli", "Laplace", "isotropic_rbf"
        ):
            raise ValueError("unsupported position-only availability formulation")
        model_id = str(payload.get("model", ""))
        if not model_id.startswith("A") or "gp_position" not in model_id:
            raise ValueError("availability artifact must contain a position GP")
        probability_clip = tuple(float(value) for value in payload["probability_clip"])
        if len(probability_clip) != 2 or not 0.0 < probability_clip[0] < probability_clip[1] < 1.0:
            raise ValueError("invalid availability probability clip")
        cameras = tuple(str(value) for value in payload.get("camera_order") or ())
        if cameras != tuple(f"camera_{letter}" for letter in "ABCDE"):
            raise ValueError("availability camera order differs from runtime")
        uncertainty_penalty = float(payload.get("uncertainty_penalty", 0.0))
        if not math.isfinite(uncertainty_penalty) or uncertainty_penalty < 0.0:
            raise ValueError("invalid GP uncertainty penalty")
        parameter_name = str(payload.get("parameter_file") or "")
        if not parameter_name or Path(parameter_name).name != parameter_name:
            raise ValueError("GP parameter file must be adjacent to the artifact")
        parameter_path = path.parent / parameter_name
        if not parameter_path.is_file() or _sha256(parameter_path) != payload.get("parameter_file_sha256"):
            raise ValueError("GP parameter file differs from its frozen identity")
        parameters = {}
        with np.load(parameter_path, allow_pickle=False) as arrays:
            for camera in cameras:
                entry = payload["parameters"][camera]
                prefix = str(entry["array_prefix"])
                model = {
                    key: np.asarray(arrays[f"{prefix}__{key}"], dtype=float)
                    for key in ("centres_xy_m", "alpha", "sqrt_weight", "cholesky")
                }
                model.update({
                    key: entry[key] for key in (
                        "prior_mean_logit", "length_scale_m", "latent_variance", "jitter"
                    )
                })
                parameters[camera] = model_from_serialized(model)
        self._clip = probability_clip
        self._uncertainty_penalty = uncertainty_penalty
        self._parameters = parameters
        self.camera_ids = cameras
        self.path = path
        self.parameter_path = parameter_path
        self.model_id = model_id

    def probability(self, camera_id: str, x: float, y: float) -> float:
        """Query the legacy availability field; absent from canonical campaigns."""
        if camera_id not in self.camera_ids:
            raise ValueError(f"unknown commissioned camera {camera_id!r}")
        position = np.asarray([[float(x), float(y)]], dtype=float)
        if not np.isfinite(position).all():
            raise ValueError("availability query position must be finite")
        probability, _standard_deviation = predict_laplace(
            self._parameters[camera_id],
            position,
            uncertainty_penalty=self._uncertainty_penalty,
        )
        return float(np.clip(probability[0], self._clip[0], self._clip[1]))
