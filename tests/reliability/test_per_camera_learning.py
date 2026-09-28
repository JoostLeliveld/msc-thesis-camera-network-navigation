from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "reliability"))

from reliability import (  # noqa: E402
    CameraPriorMap,
    LearningConfig,
    ReliabilityObservation,
    fuse_camera_prior_maps,
    learn_per_camera_reliability,
    observations_from_camera_rows,
)


def _prior(camera_id: str, probability: float = 0.25) -> CameraPriorMap:
    p = np.full((3, 3), float(probability), dtype=float)
    return CameraPriorMap(
        camera_id=camera_id,
        xs=(0.0, 1.0, 2.0),
        ys=(0.0, 1.0, 2.0),
        probability=p,
        fov_mask=np.ones_like(p, dtype=bool),
    )


def test_per_camera_learning_keeps_camera_fields_isolated() -> None:
    priors = {
        "camera_A": _prior("camera_A"),
        "camera_B": _prior("camera_B"),
    }

    result = learn_per_camera_reliability(
        priors,
        [
            ReliabilityObservation(camera_id="camera_A", xy_m=(1.0, 1.0), detected_probability=1.0),
            ReliabilityObservation(camera_id="camera_B", xy_m=(2.0, 2.0), detected_probability=0.0),
        ],
        LearningConfig(prior_strength=2.0, alpha_floor=0.0, beta_floor=0.0),
    )

    a_post = result.camera_maps["camera_A"].posterior_probability
    b_post = result.camera_maps["camera_B"].posterior_probability

    assert a_post[1, 1] > 0.25
    assert a_post[2, 2] == pytest.approx(0.25)
    assert b_post[2, 2] < 0.25
    assert b_post[1, 1] == pytest.approx(0.25)
    assert result.fused_posterior.best_camera_id[1, 1] == "camera_A"


def test_learning_supports_camera_specific_prior_strength() -> None:
    priors = fuse_camera_prior_maps({"camera_A": _prior("camera_A"), "camera_B": _prior("camera_B")})

    result = learn_per_camera_reliability(
        priors,
        [ReliabilityObservation(camera_id="camera_A", xy_m=(1.0, 1.0), detected_probability=0.0)],
        LearningConfig(prior_strength={"camera_A": 10.0, "camera_B": 1.0}, alpha_floor=0.0, beta_floor=0.0),
    )

    # Strong prior resists one miss: alpha=2.5, beta=7.5+1 -> p ~= 0.227.
    assert result.camera_maps["camera_A"].posterior_probability[1, 1] == pytest.approx(2.5 / 11.0)
    assert result.camera_maps["camera_B"].posterior_probability[1, 1] == pytest.approx(0.25)


def test_observations_from_camera_rows_builds_probability_records() -> None:
    rows = [
        {"state_x": "1.0", "state_y": "2.0", "yolo_detected_after_threshold": "1", "stamp": "3.0"},
        {"state_x": "bad", "state_y": "2.0", "yolo_detected_after_threshold": "1", "stamp": "4.0"},
    ]

    observations = observations_from_camera_rows(rows, camera_id="camera_A")

    assert len(observations) == 1
    assert observations[0].xy_m == (1.0, 2.0)
    assert observations[0].detected_probability == 1.0
    assert observations[0].timestamp_s == 3.0
