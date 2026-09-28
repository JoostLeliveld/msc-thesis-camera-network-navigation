"""Systematic wheel errors in the simulated encoder (UMBmark error model)."""
import math

import pytest

from sim.encoder_noise_node import body_velocity_from_poses, systematic_wheel_velocities

B = 0.44


def test_zero_errors_are_the_identity():
    assert systematic_wheel_velocities(0.8, 0.3, 0.0, 1.0, B) == pytest.approx((0.8, 0.3))


def test_diameter_mismatch_bends_straight_driving():
    e = 0.00121
    v_enc, w_enc = systematic_wheel_velocities(1.0, 0.0, e, 1.0, B)
    assert v_enc == pytest.approx(1.0)
    # 10 m straight -> heading error e * 10 / b
    assert w_enc * 10.0 == pytest.approx(-e * 10.0 / B)
    assert abs(math.degrees(w_enc * 10.0)) == pytest.approx(1.58, abs=0.01)


def test_wheelbase_error_scales_every_turn():
    rho = 337.2 / 340.0
    _, w_enc = systematic_wheel_velocities(0.0, 1.0, 0.0, rho, B)
    assert math.degrees((w_enc - 1.0) * math.pi / 2) == pytest.approx(-0.74, abs=0.01)


def test_true_body_velocity_from_two_poses():
    v, w = 0.9, 0.4
    dt = 0.01
    th0 = 1.0
    th1 = th0 + w * dt
    x1 = (v / w) * (math.sin(th1) - math.sin(th0))
    y1 = -(v / w) * (math.cos(th1) - math.cos(th0))
    assert body_velocity_from_poses(0.0, 0.0, th0, x1, y1, th1, dt) == pytest.approx((v, w), rel=1e-6)


def test_heading_change_wraps():
    _, w = body_velocity_from_poses(0, 0, math.pi - 0.01, 0, 0, -math.pi + 0.01, 0.1)
    assert w == pytest.approx(0.2)
