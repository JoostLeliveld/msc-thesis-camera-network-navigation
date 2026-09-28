"""Every error column must name the instant it describes, and use it.

These lock the three defects found in the 2026-08-28 logging audit, each of which was
silent: nothing raised, every column parsed, and the numbers were wrong by a factor that
looked like a property of the camera network.

1. Ground truth is buffered WITH its own stamp, and an estimate is scored against the
   truth at the estimate's stamp -- not the latest truth held at log time.
2. The interpolator refuses outside the buffered interval instead of clamping to an
   endpoint, because a clamped pose is a real pose from the wrong instant and every
   statistic downstream accepts it.
3. Wheel odometry is not called truth.

Parsed/exercised without a ROS runtime: the arithmetic under test is in static methods
and in a small pure helper, so the tests bind them to a stub rather than starting a node.
"""

from __future__ import annotations

import ast
import math
from collections import deque
from pathlib import Path

import pytest


LOGGER_PATH = (
    Path(__file__).resolve().parents[2]
    / "src" / "experiments" / "experiments" / "nodes" / "experiment_logger.py"
)


def _load_helpers():
    """Bind the pure buffer helpers onto a stub, straight from the source file."""
    tree = ast.parse(LOGGER_PATH.read_text())
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "ExperimentLogger")
    wanted = {"_interpolate_pose_buffer", "_gt_at", "_error_against_gt_at", "_wrap_angle"}
    picked = [n for n in cls.body
              if isinstance(n, (ast.FunctionDef,)) and n.name in wanted]
    assert wanted <= {n.name for n in picked}, sorted(wanted - {n.name for n in picked})
    module = ast.Module(body=[ast.ClassDef(
        name="ExperimentLogger", bases=[], keywords=[], body=picked,
        decorator_list=[])], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict = {"math": math}
    exec(compile(module, str(LOGGER_PATH), "exec"), namespace)  # noqa: S102
    return namespace["ExperimentLogger"]


@pytest.fixture(scope="module")
def helpers():
    return _load_helpers()


@pytest.fixture(scope="module")
def stub_factory(helpers):
    """A minimal object carrying the logger's buffer arithmetic and a truth buffer."""

    class _Stub(helpers):
        def __init__(self, samples):
            self._gt_buf = deque(samples)

    return _Stub


def _straight_line(n=11, dt=0.1, speed=1.0):
    """Truth moving along +x at a known speed, so travel is exactly predictable."""
    return [(i * dt, i * dt * speed, 0.0, 0.0) for i in range(n)]


def test_interpolates_between_samples(helpers, stub_factory):
    stub = stub_factory(_straight_line())
    ok, x, y, _yaw = helpers._gt_at(stub, 0.25)
    assert ok
    assert x == pytest.approx(0.25, abs=1e-9)
    assert y == pytest.approx(0.0, abs=1e-9)


def test_refuses_outside_the_buffer_instead_of_clamping(helpers, stub_factory):
    """A clamped endpoint is a real pose from the wrong instant -- the exact failure."""
    stub = stub_factory(_straight_line())
    for stamp in (-0.5, 5.0, float("nan")):
        ok, x, y, _yaw = helpers._gt_at(stub, stamp)
        assert not ok, stamp
        assert math.isnan(x) and math.isnan(y)


def test_empty_buffer_is_not_an_answer(helpers, stub_factory):
    ok, *_ = helpers._gt_at(stub_factory([]), 0.5)
    assert not ok


def test_scores_the_estimate_at_its_own_stamp_not_the_latest_truth(helpers, stub_factory):
    """The 100 ms publish lag must not appear in the error.

    Truth runs at 1 m/s. An estimate that is exactly right about where the robot was
    0.1 s ago must score 0 cm, not the 10 cm the robot travelled since.
    """
    stub = stub_factory(_straight_line())
    estimate_stamp = 0.5
    error, gx, gy = helpers._error_against_gt_at(stub, 0.5, 0.0, estimate_stamp)
    assert error == pytest.approx(0.0, abs=1e-9)
    assert (gx, gy) == pytest.approx((0.5, 0.0), abs=1e-9)

    # Scored against the truth 0.1 s later -- the old behaviour -- it would read 0.1 m.
    late, *_ = helpers._error_against_gt_at(stub, 0.5, 0.0, estimate_stamp + 0.1)
    assert late == pytest.approx(0.1, abs=1e-9)


def test_missing_stamp_gives_nan_never_a_silent_fallback(helpers, stub_factory):
    stub = stub_factory(_straight_line())
    error, gx, gy = helpers._error_against_gt_at(stub, 0.5, 0.0, float("nan"))
    assert math.isnan(error) and math.isnan(gx) and math.isnan(gy)


def _source() -> str:
    return LOGGER_PATH.read_text()


def test_ground_truth_callback_keeps_the_transform_stamp():
    """Discarding `tr.header.stamp` is what made every column unalignable."""
    source = _source()
    start = source.index("def _ground_truth_cb")
    body = source[start:source.index("\n    def ", start + 1)]
    assert "tr.header.stamp" in body, (
        "_ground_truth_cb must keep the transform's own stamp; without it the only "
        "instant truth can be paired with is the log clock")
    assert "_gt_buf.append" in body


def test_wheel_odometry_is_not_called_truth():
    """`_latest_truth_pose` returned /odom. The name is the whole defect."""
    source = _source()
    assert "def _latest_truth_pose" not in source and "._latest_truth_pose(" not in source, (
        "wheel odometry must not be named truth: it drifts 24 cm median and 2.4 m worst "
        "on these drives, against errors reported in centimetres")
    assert "def _latest_odom_map_pose" in source
    assert "_truth_buf" not in source, (
        "the odometry buffer must not be named truth either -- it was the only "
        "interpolatable pose buffer, so every capture-time metric used it")


def test_final_goal_distance_uses_ground_truth():
    """It measured wheel drift while the goal-reached decision used ground truth."""
    source = _source()
    start = source.index("final_goal_distance = math.nan")
    body = source[start:start + 1400]
    assert "self._gt_xy" in body
    assert "'final_goal_distance_reference': 'ground_truth'" in source


def test_fusion_observations_can_be_deduplicated():
    """Each detection is written ~4x; a row must say which repeat it is."""
    source = _source()
    for column in ("'obs_repeat'", "'obs_seq'", "'gt_x_at_obs'", "'fused_stamp'"):
        assert column in source, column


def test_manifest_declares_its_logging_schema():
    """Runs carry the conventions they were written under, so a scorer can refuse
    a drive it cannot score instead of silently mixing two definitions of a column.
    Bump this deliberately, together with the comment block that says what changed."""
    assert "'logging_schema_version': 9" in _source()
    assert "'belief_prediction_ledger': 'belief_predictions.jsonl'" in _source()


def test_schema_3_records_batch_identity_and_ground_truth_never_terminates():
    """Batch identity (2026-08-28 audit) and offline-only collision scoring."""
    source = _source()
    # a detector batch is one identity, so one round cannot be assimilated four times
    assert "'source_batch_id'" in source
    # ground truth never terminates a run: collision is scored offline from the pose log
    assert "'termination_reference': 'planner_belief_or_timeout'" in source
    assert "'ground_truth_pose.csv'" in source
    assert "terminate_on_geom_collision" not in source
