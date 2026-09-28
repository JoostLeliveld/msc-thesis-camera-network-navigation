"""Analytic footprint cases and continuous-motion counterexamples."""
from types import SimpleNamespace
import math
import numpy as np
import pytest
from unav_common.rectangular_footprint import RectangularFootprint


def prism(xmin, ymin, xmax, ymax):
    return SimpleNamespace(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)


def test_aligned_body_fits_seventy_centimetre_aisle_but_sideways_does_not():
    model = RectangularFootprint([prism(-3, -.35, 3, .35)], keep_in=True)
    assert model.clearance([0, 0, 0]) == pytest.approx(.075)
    assert model.clearance([0, 0, math.pi/2]) < 0
    assert model.sweep_clearance([-1, 0, 0], [1, 0, 0]) > 0


def test_rotation_sweep_catches_corner_with_clear_endpoints():
    model = RectangularFootprint([prism(.42, -.02, .44, .02)])
    assert model.clearance([0, 0, 0]) > 0
    assert model.clearance([0, 0, math.pi]) > 0
    assert model.sweep_clearance([0, 0, 0], [0, 0, math.pi], yaw_delta=math.pi) < 0


def test_translation_sweep_catches_thin_wall():
    model = RectangularFootprint([prism(.001, -1, .002, 1)])
    assert model.clearance([-1, 0, 0]) > 0
    assert model.clearance([1, 0, 0]) > 0
    assert model.sweep_clearance([-1, 0, 0], [1, 0, 0]) < 0


def test_constant_twist_arc_is_checked_as_well_as_euler_segment():
    model = RectangularFootprint([prism(.30, .30, .34, .34)], length=.02, width=.02)
    start, end = [0, 0, 0], [1, 0, math.pi]
    assert model.sweep_clearance(start, end, yaw_delta=math.pi) > 0
    assert model.sweep_clearance(start, end, control=[1, math.pi], dt=1) < 0


def test_union_seam_is_not_an_obstacle():
    model = RectangularFootprint([prism(-2, -1, 0, 1), prism(0, -1, 2, 1)], keep_in=True)
    assert model.clearance([0, 0, 0]) == pytest.approx(.725)


def test_corners_inside_do_not_hide_a_hole_under_body():
    model = RectangularFootprint([prism(-1, -1, -.1, 1), prism(.1, -1, 1, 1),
                                  prism(-.1, -.99, .1, -.1), prism(-.1, .1, .1, .99)], keep_in=True)
    assert model.clearance([0, 0, 0]) < 0


def test_l_corner_cannot_be_cut_with_body_edges():
    model = RectangularFootprint([prism(-2, -.35, .35, .35), prism(-.35, -.35, .35, 2)], keep_in=True)
    assert model.clearance([-.2, .2, math.pi/4]) < 0


def test_unresolved_sweep_fails_closed():
    model = RectangularFootprint([prism(-3, -.3, 3, .3)], keep_in=True)
    assert model.sweep_clearance([-1, 0, 0], [1, 0, 0], max_depth=0) < 0


def test_wraparound_uses_short_turn_unless_explicitly_overridden():
    model = RectangularFootprint([prism(-2, -.35, 2, .35)], keep_in=True)
    assert model.sweep_clearance([0, 0, math.pi-.01], [0, 0, -math.pi+.01]) > 0
    assert model.sweep_clearance([0, 0, math.pi-.01], [0, 0, -math.pi+.01], yaw_delta=2*math.pi) < 0


def test_required_margin_refines_the_certificate_instead_of_false_refusal():
    model = RectangularFootprint([prism(1.0, -2.0, 2.0, 2.0)], .8, .55)
    value = model.sweep_clearance(
        [0.0, 0.0, 0.0], [0.0, 0.0, math.pi / 4.0],
        yaw_delta=math.pi / 4.0, required_clearance=0.50,
    )
    assert value >= 0.50


@pytest.mark.parametrize('length,width', [(0,.55), (.8,-1), (np.nan,.55), (.8,np.inf)])
def test_invalid_dimensions_rejected(length, width):
    with pytest.raises(ValueError):
        RectangularFootprint([], length, width)


def test_empty_keep_in_is_not_permission_to_drive_anywhere():
    assert RectangularFootprint([], keep_in=True).clearance([0, 0, 0]) == -math.inf
    assert RectangularFootprint([]).clearance([0, 0, 0]) == math.inf
