from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
location = str(ROOT / "src/reliability")
if location not in sys.path:
    sys.path.insert(0, location)

from reliability.bernoulli_gp import fit_laplace_binomial  # noqa: E402
from reliability.commissioned_availability import CommissionedAvailabilityModel  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gp_artifact(path: Path) -> str:
    cameras = [f"camera_{letter}" for letter in "ABCDE"]
    fitted = fit_laplace_binomial(
        np.asarray([[-1.0, 0.0], [1.0, 0.0]]),
        np.asarray([0.0, 10.0]), np.asarray([10.0, 10.0]),
        length_scale_m=0.8, latent_variance=4.0, prior_probability=0.5,
    )
    arrays = {}
    for camera in cameras:
        for key in ("centres_xy_m", "alpha", "sqrt_weight", "cholesky"):
            arrays[f"{camera}__{key}"] = fitted[key]
    parameter_path = path.parent / "availability_gp_parameters.npz"
    np.savez_compressed(parameter_path, **arrays)
    payload = {
        "schema": "thesis_commissioned_availability_model.v2",
        "model": "A1_gp_position_ls0.8",
        "final_audit_accessed": False,
        "camera_order": cameras,
        "feature_names": ["robot_x_m", "robot_y_m"],
        "probability_clip": [0.001, 0.999],
        "likelihood": "Bernoulli",
        "inference": "Laplace",
        "kernel": "isotropic_rbf",
        "uncertainty_penalty": 0.0,
        "parameter_file": parameter_path.name,
        "parameter_file_sha256": _sha256(parameter_path),
        "parameters": {
            camera: {
                "array_prefix": camera,
                "prior_mean_logit": fitted["prior_mean_logit"],
                "length_scale_m": fitted["length_scale_m"],
                "latent_variance": fitted["latent_variance"],
                "jitter": fitted["jitter"],
            }
            for camera in cameras
        },
    }
    path.write_text(json.dumps(payload))
    return _sha256(path)


def test_runtime_query_is_position_only_and_hash_locked(tmp_path):
    path = tmp_path / "availability.json"
    digest = _gp_artifact(path)
    model = CommissionedAvailabilityModel(path, expected_sha256=digest)
    low = model.probability("camera_A", -1.0, 0.0)
    high = model.probability("camera_A", 1.0, 0.0)
    assert low < 0.5 < high
    assert model.probability("camera_A", 1.0, 0.0) == pytest.approx(high)
    with pytest.raises(ValueError, match="hash differs"):
        CommissionedAvailabilityModel(path, expected_sha256="0" * 64)


def test_runtime_rejects_unknown_camera_and_nonfinite_position(tmp_path):
    path = tmp_path / "availability.json"
    _gp_artifact(path)
    model = CommissionedAvailabilityModel(path)
    with pytest.raises(ValueError, match="unknown commissioned camera"):
        model.probability("camera_Z", 0.0, 0.0)
    with pytest.raises(ValueError, match="position must be finite"):
        model.probability("camera_A", float("nan"), 0.0)


def test_parameter_identity_is_locked(tmp_path):
    path = tmp_path / "availability.json"
    _gp_artifact(path)
    parameter_path = tmp_path / "availability_gp_parameters.npz"
    parameter_path.write_bytes(parameter_path.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="parameter file differs"):
        CommissionedAvailabilityModel(path)
