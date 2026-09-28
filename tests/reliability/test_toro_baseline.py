from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "reliability"))

from reliability.contracts import CameraQuality, ContractValidationError  # noqa: E402
from reliability.fusion import MapObservation, _validate_spd  # noqa: E402
from reliability.toro_baseline import (  # noqa: E402
    CalibrationPoint,
    ToroCovarianceModel,
    bin_observations,
    constant_velocity_predict,
)


COV_A = ((0.04, 0.0), (0.0, 0.09))
COV_B = ((0.25, 0.05), (0.05, 0.16))
COV_C = ((0.01, 0.0), (0.0, 0.01))


def _point(camera_id: str, x: float, y: float, cov=COV_A, count: int = 20) -> CalibrationPoint:
    return CalibrationPoint(
        camera_id=camera_id,
        position_xy=(x, y),
        covariance_xy=cov,
        sample_count=count,
    )


def _map_obs(camera_id: str, timestamp_s: float, x: float = 0.0, y: float = 0.0) -> MapObservation:
    return MapObservation(
        camera_id=camera_id,
        timestamp_s=timestamp_s,
        xy_m=(x, y),
        covariance_m2=((0.01, 0.0), (0.0, 0.01)),
        quality=CameraQuality(camera_id=camera_id, p_available=0.9),
    )


# ---------------------------------------------------------------------------
# CalibrationPoint validation
# ---------------------------------------------------------------------------


def test_calibration_point_rejects_non_spd_covariance() -> None:
    with pytest.raises(ContractValidationError):
        _point("camera_A", 0.0, 0.0, cov=((1.0, 2.0), (2.0, 1.0)))  # det < 0
    with pytest.raises(ContractValidationError):
        _point("camera_A", 0.0, 0.0, cov=((1.0, 0.5), (0.4, 1.0)))  # asymmetric
    with pytest.raises(ContractValidationError):
        _point("camera_A", 0.0, 0.0, cov=((0.0, 0.0), (0.0, 1.0)))  # zero diagonal


def test_calibration_point_rejects_bad_sample_count_and_empty_camera() -> None:
    with pytest.raises(ContractValidationError):
        _point("camera_A", 0.0, 0.0, count=0)
    with pytest.raises(ContractValidationError):
        _point("", 0.0, 0.0)


@pytest.mark.parametrize("sample_count", [True, 1.0, 1.5, "2"])
def test_calibration_point_sample_count_requires_an_integer(sample_count: object) -> None:
    with pytest.raises(ContractValidationError, match="positive integer"):
        CalibrationPoint(
            camera_id="camera_A",
            position_xy=(0.0, 0.0),
            covariance_xy=COV_A,
            sample_count=sample_count,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Nearest-point covariance lookup
# ---------------------------------------------------------------------------


def test_nearest_point_lookup_returns_closest_covariance() -> None:
    model = ToroCovarianceModel(
        [
            _point("camera_A", 0.0, 0.0, cov=COV_A),
            _point("camera_A", 10.0, 0.0, cov=COV_B),
        ]
    )
    assert model.covariance_for("camera_A", (1.0, 0.5)) == COV_A
    assert model.covariance_for("camera_A", (9.0, -0.5)) == COV_B


def test_nearest_point_lookup_exact_hit() -> None:
    model = ToroCovarianceModel(
        [
            _point("camera_A", 0.0, 0.0, cov=COV_A),
            _point("camera_A", 10.0, 0.0, cov=COV_B),
        ]
    )
    assert model.covariance_for("camera_A", (10.0, 0.0)) == COV_B


def test_nearest_point_lookup_tie_breaks_to_first_in_order() -> None:
    model = ToroCovarianceModel(
        [
            _point("camera_A", 0.0, 0.0, cov=COV_A),
            _point("camera_A", 2.0, 0.0, cov=COV_B),
        ]
    )
    # (1, 0) is equidistant from both points; the earliest point wins.
    assert model.covariance_for("camera_A", (1.0, 0.0)) == COV_A


def test_nearest_point_no_interpolation() -> None:
    model = ToroCovarianceModel(
        [
            _point("camera_A", 0.0, 0.0, cov=COV_A),
            _point("camera_A", 10.0, 0.0, cov=COV_B),
        ]
    )
    # Even very close to the midpoint the result is one stored matrix verbatim.
    assert model.covariance_for("camera_A", (4.999, 0.0)) == COV_A
    assert model.covariance_for("camera_A", (5.001, 0.0)) == COV_B


def test_unknown_camera_raises() -> None:
    model = ToroCovarianceModel([_point("camera_A", 0.0, 0.0)])
    with pytest.raises(ContractValidationError):
        model.covariance_for("camera_B", (0.0, 0.0))
    with pytest.raises(ContractValidationError):
        model.in_validated_fov("camera_B", (0.0, 0.0))


def test_empty_model_raises_on_any_query() -> None:
    model = ToroCovarianceModel([])
    with pytest.raises(ContractValidationError):
        model.covariance_for("camera_A", (0.0, 0.0))


def test_lookup_covariance_is_spd_passthrough() -> None:
    model = ToroCovarianceModel([_point("camera_A", 0.0, 0.0, cov=COV_B)])
    cov = model.covariance_for("camera_A", (3.0, 3.0))
    _validate_spd(cov, "lookup result")  # must still be SPD, bit-identical
    assert cov == COV_B


# ---------------------------------------------------------------------------
# Validated-FOV gate (convex hull of calibration points)
# ---------------------------------------------------------------------------


def _square_model() -> ToroCovarianceModel:
    corners = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
    points = [_point("camera_A", x, y) for x, y in corners]
    # An interior calibration point must not affect the hull.
    points.append(_point("camera_A", 1.0, 1.0))
    return ToroCovarianceModel(points)


def test_fov_gate_inside() -> None:
    model = _square_model()
    assert model.in_validated_fov("camera_A", (1.0, 1.0))
    assert model.in_validated_fov("camera_A", (0.1, 1.9))


def test_fov_gate_outside() -> None:
    model = _square_model()
    assert not model.in_validated_fov("camera_A", (3.0, 1.0))
    assert not model.in_validated_fov("camera_A", (-0.001, 1.0))
    assert not model.in_validated_fov("camera_A", (1.0, 2.001))


def test_fov_gate_edge_and_vertex_are_inside() -> None:
    model = _square_model()
    assert model.in_validated_fov("camera_A", (0.0, 1.0))  # edge
    assert model.in_validated_fov("camera_A", (2.0, 2.0))  # vertex


def test_fov_gate_degenerate_two_points_is_segment() -> None:
    model = ToroCovarianceModel(
        [_point("camera_A", 0.0, 0.0), _point("camera_A", 4.0, 0.0)]
    )
    assert model.in_validated_fov("camera_A", (2.0, 0.0))
    assert not model.in_validated_fov("camera_A", (2.0, 0.5))
    assert not model.in_validated_fov("camera_A", (5.0, 0.0))


def test_fov_gate_degenerate_single_point() -> None:
    model = ToroCovarianceModel([_point("camera_A", 1.0, 1.0)])
    assert model.in_validated_fov("camera_A", (1.0, 1.0))
    assert not model.in_validated_fov("camera_A", (1.1, 1.0))


def test_fov_gate_per_camera_regions_are_independent() -> None:
    model = ToroCovarianceModel(
        [
            _point("camera_A", 0.0, 0.0),
            _point("camera_A", 2.0, 0.0),
            _point("camera_A", 1.0, 2.0),
            _point("camera_B", 10.0, 10.0),
            _point("camera_B", 12.0, 10.0),
            _point("camera_B", 11.0, 12.0),
        ]
    )
    assert model.in_validated_fov("camera_A", (1.0, 0.5))
    assert not model.in_validated_fov("camera_B", (1.0, 0.5))
    assert model.in_validated_fov("camera_B", (11.0, 10.5))


# ---------------------------------------------------------------------------
# Time binning (B2a)
# ---------------------------------------------------------------------------


def test_bin_observations_empty() -> None:
    assert bin_observations([]) == []


def test_bin_observations_groups_within_bin() -> None:
    obs = [
        _map_obs("camera_A", 0.00),
        _map_obs("camera_B", 0.03),
        _map_obs("camera_C", 0.079),
    ]
    bins = bin_observations(obs)
    assert len(bins) == 1
    assert [o.camera_id for o in bins[0]] == ["camera_A", "camera_B", "camera_C"]


def test_bin_observations_exact_bin_edge_starts_new_bin() -> None:
    obs = [_map_obs("camera_A", 0.0), _map_obs("camera_B", 0.08)]
    bins = bin_observations(obs)
    assert len(bins) == 2
    assert bins[0][0].camera_id == "camera_A"
    assert bins[1][0].camera_id == "camera_B"


def test_bin_observations_edge_robust_to_float_rounding() -> None:
    # 0.16 / 0.08 == 1.9999999999999998 in binary floating point; a naive
    # floor would misplace the exact-edge observation into the earlier bin.
    obs = [_map_obs("camera_A", 0.08), _map_obs("camera_B", 0.16)]
    bins = bin_observations(obs)
    assert len(bins) == 2


def test_bin_observations_sorts_out_of_order_timestamps() -> None:
    obs = [
        _map_obs("camera_C", 0.20),
        _map_obs("camera_A", 0.01),
        _map_obs("camera_B", 0.09),
    ]
    bins = bin_observations(obs)
    assert [o.camera_id for group in bins for o in group] == [
        "camera_A",
        "camera_B",
        "camera_C",
    ]
    assert [len(group) for group in bins] == [1, 1, 1]


def test_bin_observations_custom_width_and_gap() -> None:
    obs = [
        _map_obs("camera_A", 0.0),
        _map_obs("camera_B", 0.4),
        _map_obs("camera_C", 3.0),
    ]
    bins = bin_observations(obs, bin_width_s=0.5)
    assert len(bins) == 2  # empty bins between 0.5 s and 3.0 s are not emitted
    assert [o.camera_id for o in bins[0]] == ["camera_A", "camera_B"]
    assert [o.camera_id for o in bins[1]] == ["camera_C"]


def test_bin_observations_rejects_bad_width() -> None:
    with pytest.raises(ContractValidationError):
        bin_observations([], bin_width_s=0.0)
    with pytest.raises(ContractValidationError):
        bin_observations([], bin_width_s=-0.08)


# ---------------------------------------------------------------------------
# Constant-velocity prediction
# ---------------------------------------------------------------------------


IDENTITY_4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def test_cv_predict_zero_dt_is_identity() -> None:
    state = (1.0, -2.0, 0.3, 0.7)
    new_state, new_cov = constant_velocity_predict(state, IDENTITY_4, dt=0.0, q=2.0)
    assert new_state == state
    assert new_cov == IDENTITY_4


def test_cv_predict_mean_moves_with_velocity() -> None:
    state = (1.0, 2.0, 0.5, -0.5)
    new_state, _ = constant_velocity_predict(state, IDENTITY_4, dt=0.4, q=0.0)
    assert new_state[0] == pytest.approx(1.2)
    assert new_state[1] == pytest.approx(1.8)
    assert new_state[2] == pytest.approx(0.5)
    assert new_state[3] == pytest.approx(-0.5)


def test_cv_predict_covariance_grows_with_dt() -> None:
    def trace(matrix) -> float:
        return sum(matrix[i][i] for i in range(4))

    _, cov_small = constant_velocity_predict((0.0, 0.0, 0.0, 0.0), IDENTITY_4, dt=0.1, q=1.0)
    _, cov_large = constant_velocity_predict((0.0, 0.0, 0.0, 0.0), IDENTITY_4, dt=1.0, q=1.0)
    assert trace(cov_small) > trace(IDENTITY_4)
    assert trace(cov_large) > trace(cov_small)
    # Position-velocity cross terms must grow too (white-accel coupling).
    assert cov_large[0][2] > cov_small[0][2] > 0.0


def test_cv_predict_matches_hand_computed_one_step() -> None:
    # P = I, dt = 0.5, q = 2.0:
    #   F P F' = [[1.25, 0, 0.5, 0], [0, 1.25, 0, 0.5], [0.5, 0, 1, 0], [0, 0.5, 0, 1]]
    #   Q      = [[1/12, 0, 1/4, 0], [0, 1/12, 0, 1/4], [1/4, 0, 1, 0], [0, 1/4, 0, 1]] * 2 ... expanded below
    dt, q = 0.5, 2.0
    _, cov = constant_velocity_predict((0.0, 0.0, 0.0, 0.0), IDENTITY_4, dt=dt, q=q)
    q_pp = q * dt**3 / 3.0  # 0.0833...
    q_pv = q * dt**2 / 2.0  # 0.25
    q_vv = q * dt  # 1.0
    expected = (
        (1.0 + dt**2 + q_pp, 0.0, dt + q_pv, 0.0),
        (0.0, 1.0 + dt**2 + q_pp, 0.0, dt + q_pv),
        (dt + q_pv, 0.0, 1.0 + q_vv, 0.0),
        (0.0, dt + q_pv, 0.0, 1.0 + q_vv),
    )
    for i in range(4):
        for j in range(4):
            assert cov[i][j] == pytest.approx(expected[i][j], abs=1e-12)


def test_cv_predict_result_is_symmetric() -> None:
    cov0 = (
        (2.0, 0.1, 0.3, 0.0),
        (0.1, 1.5, 0.0, 0.2),
        (0.3, 0.0, 0.8, 0.05),
        (0.0, 0.2, 0.05, 0.9),
    )
    _, cov = constant_velocity_predict((0.0, 0.0, 1.0, 1.0), cov0, dt=0.25, q=0.5)
    for i in range(4):
        for j in range(4):
            assert cov[i][j] == pytest.approx(cov[j][i], abs=1e-12)
            assert math.isfinite(cov[i][j])


def test_cv_predict_accepts_positive_semidefinite_covariance() -> None:
    # Outer product vv' is rank one, hence PSD but singular.
    vector = (1.0, -2.0, 0.5, 3.0)
    rank_one_cov = tuple(
        tuple(vector[i] * vector[j] for j in range(4)) for i in range(4)
    )
    _, covariance = constant_velocity_predict(
        (0.0, 0.0, 0.0, 0.0), rank_one_cov, dt=0.0, q=0.0
    )
    for i in range(4):
        for j in range(4):
            assert covariance[i][j] == pytest.approx(rank_one_cov[i][j])


def test_cv_predict_accepts_roundoff_at_semidefinite_boundary() -> None:
    nearly_rank_one = (
        (1.0, 1.0, 0.0, 0.0),
        (1.0, 1.0 - 5.0e-11, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
    )
    constant_velocity_predict(
        (0.0, 0.0, 0.0, 0.0), nearly_rank_one, dt=0.0, q=0.0
    )


def test_cv_predict_rejects_symmetric_indefinite_covariance() -> None:
    indefinite = (
        (1.0, 2.0, 0.0, 0.0),
        (2.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    with pytest.raises(ContractValidationError, match="positive semidefinite"):
        constant_velocity_predict(
            (0.0, 0.0, 0.0, 0.0), indefinite, dt=0.1, q=1.0
        )


def test_cv_predict_rejects_bad_inputs() -> None:
    with pytest.raises(ContractValidationError):
        constant_velocity_predict((0.0, 0.0, 0.0), IDENTITY_4, dt=0.1, q=1.0)
    with pytest.raises(ContractValidationError):
        constant_velocity_predict((0.0, 0.0, 0.0, 0.0), IDENTITY_4, dt=-0.1, q=1.0)
    with pytest.raises(ContractValidationError):
        constant_velocity_predict((0.0, 0.0, 0.0, 0.0), IDENTITY_4, dt=0.1, q=-1.0)
    asymmetric = (
        (1.0, 0.5, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    with pytest.raises(ContractValidationError):
        constant_velocity_predict((0.0, 0.0, 0.0, 0.0), asymmetric, dt=0.1, q=1.0)
