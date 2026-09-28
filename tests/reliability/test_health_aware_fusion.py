"""Unit test for the B6 health-aware fusion replay mode (the containment 'full method').

Controlled synthetic scenario (allowed: this tests that the CODE behaves correctly —
it is NOT paper evidence). A dominant, accurate camera `cam_good` anchors the belief
near ground truth; a second camera `cam_bad` carries a PERSISTENT position bias sized
to EVADE the per-frame NIS gate (so naive fusion M5 keeps fusing it and drifts). This
is the regime innovation-health is designed for — a good-majority/dominant anchor, so
the biased camera's innovation reveals its ~full bias (with two equal-weight cameras
the belief drags to the midpoint and masks the fault; detection then needs cross-camera
disagreement, per the pre-registration).

B6 must: (1) drive cam_bad's debounced health to DEGRADED (detect the drift),
(2) NOT flag the healthy cam_good (no false alarm), (3) contain the fault — lower
RMSE and final error than naive fusion. GT is used ONLY to score error; it never
enters the filter (B6 acts purely on innovation/NIS).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "reliability"))

from reliability.contracts import CameraQuality  # noqa: E402
from reliability.fusion import MapObservation  # noqa: E402
from reliability.replay import (  # noqa: E402
    EvaluationFrame,
    ReplayConfig,
    ReplayFrame,
    ReplayMode,
    run_replay,
)

R_GOOD = 0.03 ** 2         # dominant, accurate camera -> anchors the belief near GT
R_BAD = 0.10 ** 2          # biased camera; S = cov + R_BAD ~= 0.0105
BIAS_BAD_M = 0.28          # persistent bias: full-bias NIS ~= 7.5 (< 9.21 gate; gate-evading)
N = 40


def _frames():
    frames, evals = [], []
    for i in range(N):
        t = i * 0.2
        gx, gy = 0.05 * i, 0.0
        obs = (
            MapObservation(
                camera_id="cam_good", timestamp_s=t, xy_m=(gx, gy),
                covariance_m2=((R_GOOD, 0.0), (0.0, R_GOOD)),
                quality=CameraQuality(camera_id="cam_good", p_available=0.9), source="test",
            ),
            MapObservation(
                camera_id="cam_bad", timestamp_s=t, xy_m=(gx + BIAS_BAD_M, gy),
                covariance_m2=((R_BAD, 0.0), (0.0, R_BAD)),
                quality=CameraQuality(camera_id="cam_bad", p_available=0.9), source="test",
            ),
        )
        frames.append(ReplayFrame(timestamp_s=t, odometry_xy_m=(gx, gy), observations=obs))
        evals.append(EvaluationFrame(timestamp_s=t, truth_xy_m=(gx, gy)))
    return frames, evals


def _run(mode):
    frames, evals = _frames()
    return run_replay(frames, ReplayConfig(mode=mode, nis_gate=9.21), evaluation_frames=evals)


def test_bias_is_gate_evading_for_naive_fusion():
    # Precondition: naive fusion does NOT gate the bias out per-frame (else M5 would
    # trivially "contain" it and there would be nothing for B6 to improve).
    m5 = _run(ReplayMode.SEQUENTIAL_FUSION)
    accepted_bad = sum("cam_bad" in s.accepted_camera_ids for s in m5.steps)
    assert accepted_bad >= N // 2


def test_b6_flags_the_biased_camera_and_not_the_good_one():
    b6 = _run(ReplayMode.HEALTH_AWARE_FUSION)
    assert all(s.health_by_camera for s in b6.steps[1:])          # health timeseries populated
    bad_states = [s.health_state_by_camera.get("cam_bad") for s in b6.steps]
    good_states = [s.health_state_by_camera.get("cam_good") for s in b6.steps]
    assert "DEGRADED" in bad_states                               # drift detected + escalated
    assert "DEGRADED" not in good_states                          # healthy camera never hard-flagged
    assert good_states[-1] == "HEALTHY"


def test_b6_contains_the_fault_better_than_naive_fusion():
    m5 = _run(ReplayMode.SEQUENTIAL_FUSION).metrics
    b6 = _run(ReplayMode.HEALTH_AWARE_FUSION).metrics
    assert m5 is not None and b6 is not None
    # containment: once the drift is caught, B6's belief coasts on the good camera.
    # Assert on ramp-robust metrics (final + RMSE), not p95 (the pre-DEGRADED ramp
    # frames inflate a high percentile).
    assert b6.final_error_m < m5.final_error_m
    assert b6.rmse_m < m5.rmse_m
