"""Regression tests for the no-go cost's oriented-body clearance."""

import json
import math

import pytest
import numpy as np

from planning.core.nogo_cost import NogoCostConfig, NogoZoneCostModel


def lane_scene():
    return json.dumps({
        "prisms": [{
            "xmin": -3.0,
            "xmax": 3.0,
            "ymin": -0.35,
            "ymax": 0.35,
            "zmin": 0.0,
            "zmax": 1.0,
        }],
    })


def test_aligned_body_uses_lateral_extent_not_longitudinal_extent():
    model = NogoZoneCostModel(NogoCostConfig(
        geometry_json=lane_scene(),
        mode="keep_in",
        weight=1.0,
        safe_distance=0.325,
        robot_half_length=0.4,
        robot_half_width=0.275,
        body_margin=0.05,
    ))

    # The exact body has 7.5 cm to the lane edge; the separate margin leaves
    # 2.5 cm.  The superseded max-axis surrogate returned -10 cm here.
    assert model.clearance_state_np([0.0, 0.0, 0.0]) == pytest.approx(0.025)


def test_sideways_body_still_fails_the_same_narrow_lane():
    model = NogoZoneCostModel(NogoCostConfig(
        geometry_json=lane_scene(),
        mode="keep_in",
        weight=1.0,
        safe_distance=0.325,
        robot_half_length=0.4,
        robot_half_width=0.275,
        body_margin=0.05,
    ))

    assert model.clearance_state_np([0.0, 0.0, math.pi / 2.0]) < 0.0


def test_point_surrogate_is_unchanged_when_no_body_is_configured():
    model = NogoZoneCostModel(NogoCostConfig(
        geometry_json=lane_scene(),
        mode="keep_in",
        weight=1.0,
        safe_distance=0.325,
    ))

    assert model.clearance_state_np([0.0, 0.0, 0.0]) == pytest.approx(0.025)


def test_shape_warning_starts_inside_five_centimetres_and_matches_casadi():
    model = NogoZoneCostModel(NogoCostConfig(
        geometry_json=lane_scene(), mode="keep_in", weight=40.0,
        safe_distance=0.325, warning_band=0.05, near_weight=50.0,
        robot_half_length=0.4, robot_half_width=0.275,
        body_margin=0.0,
    ))
    penalty_ca = model.make_penalty_state_casadi()

    # Centred body clearance is 7.5 cm: outside the 5 cm warning band.
    safe = np.asarray([0.0, 0.0, 0.0])
    assert model.clearance_state_np(safe) == pytest.approx(0.075)
    assert model.penalty_state_np(safe) == pytest.approx(0.0)
    assert float(penalty_ca(safe)) == pytest.approx(0.0)

    edge = np.asarray([0.0, 0.025, 0.0])
    assert model.clearance_state_np(edge) == pytest.approx(0.05)
    assert model.penalty_state_np(edge) == pytest.approx(0.0, abs=1e-24)
    assert float(penalty_ca(edge)) == pytest.approx(0.0, abs=1e-24)

    # Move 1 mm farther: body clearance is 4.9 cm and the warning is on.
    near = np.asarray([0.0, 0.026, 0.0])
    assert model.clearance_state_np(near) == pytest.approx(0.049)
    assert model.penalty_state_np(near) > 0.0
    assert float(penalty_ca(near)) == pytest.approx(
        model.penalty_state_np(near), rel=1e-9, abs=1e-9)
