"""Regression tests for the frozen optional coherent-drift configuration."""
import math

import numpy as np
import pytest

from planning.core.dynamics import (
    COHERENT_HEADING_RAD,
    COHERENT_SPEED_SCALE,
    coherent_drift_block,
    coherent_drift_increment,
    unicycle_process_noise,
)


def test_off_by_default_leaves_process_covariance_bit_identical():
    """Existing campaigns must be unaffected until one opts in."""
    for theta, v, dt in ((0.0, 0.22, 0.1), (1.1, 0.15, 0.4), (-2.0, 0.05, 0.05)):
        base = unicycle_process_noise(0.01, 0.02, dt, theta=theta, v=v)
        default = unicycle_process_noise(0.01, 0.02, dt, theta=theta, v=v,
                                         coherent_drift=False)
        np.testing.assert_array_equal(base, default)


def test_enabling_only_adds_covariance():
    """A process-noise term may widen the belief, never sharpen it."""
    off = unicycle_process_noise(0.01, 0.02, 0.1, theta=0.3, v=0.22)
    on = unicycle_process_noise(0.01, 0.02, 0.1, theta=0.3, v=0.22, coherent_drift=True)
    delta = on - off
    assert np.all(np.linalg.eigvalsh(delta[:2, :2]) >= -1e-15)
    assert on[0, 0] > off[0, 0] and on[1, 1] > off[1, 1]


def test_heading_row_untouched():
    """The coherent terms are positional; Gate 0 did not correct heading."""
    block = coherent_drift_block(1.0, 0.7)
    assert block[2, 2] == 0.0
    assert np.allclose(block[2, :], 0.0) and np.allclose(block[:, 2], 0.0)


def test_terms_land_along_and_across_the_heading():
    """along = speed scale, across = held heading offset, in the body frame."""
    distance, theta = 2.0, 0.0
    block = coherent_drift_block(distance, theta)
    # theta = 0 => forward is +x, lateral is +y.
    assert block[0, 0] == pytest.approx((COHERENT_SPEED_SCALE * distance) ** 2)
    assert block[1, 1] == pytest.approx((COHERENT_HEADING_RAD * distance) ** 2)


def test_rotates_with_heading():
    """The same drive rotated 90 deg puts the same variance on the other axis."""
    a = coherent_drift_block(1.5, 0.0)
    b = coherent_drift_block(1.5, math.pi / 2)
    assert b[1, 1] == pytest.approx(a[0, 0])
    assert b[0, 0] == pytest.approx(a[1, 1])


def test_grows_with_distance_not_time():
    """Coherent drift scales with distance travelled: doubling it quadruples variance."""
    one = coherent_drift_block(1.0, 0.4)
    two = coherent_drift_block(2.0, 0.4)
    np.testing.assert_allclose(two, 4.0 * one, rtol=1e-12)


def test_a_stationary_robot_accumulates_none():
    """v = 0 travels no distance, so a coherent scale error contributes nothing."""
    moving = unicycle_process_noise(0.01, 0.02, 0.1, theta=0.0, v=0.22, coherent_drift=True)
    still = unicycle_process_noise(0.01, 0.02, 0.1, theta=0.0, v=0.0, coherent_drift=True)
    baseline = unicycle_process_noise(0.01, 0.02, 0.1, theta=0.0, v=0.0)
    np.testing.assert_allclose(still, baseline, rtol=1e-12)
    assert moving[0, 0] > still[0, 0]


def test_fixes_the_anisotropy_that_gate0_found():
    """The white model is ~1550x stiffer across the path than along it; this is the bug.

    The fix is only visible INTEGRATED over a window, and only if the per-step increments
    telescope: coherent drift grows as distance^2, so summing block(step) once per step
    would re-introduce the sqrt(n) averaging the term exists to prevent.
    """
    white_step = unicycle_process_noise(0.01, 0.02, 0.1, theta=0.0, v=0.22)
    assert white_step[0, 0] / white_step[1, 1] > 1000.0

    def integrate(coherent):
        total = np.zeros((3, 3))
        travelled = 0.0
        for _ in range(50):                      # 5 s at 0.1 s steps, 0.22 m/s
            total += unicycle_process_noise(
                0.01, 0.02, 0.1, theta=0.0, v=0.22,
                coherent_drift=coherent, distance_travelled_m=travelled,
            )
            travelled += 0.22 * 0.1
        return total

    white, fixed = integrate(False), integrate(True)
    # Over 5 s of straight driving the white model puts almost nothing across the path.
    assert math.sqrt(white[1, 1]) < 0.002                     # under 2 mm
    # Gate 0 measured 2.43 cm of real cross-track drift at 5 s (19 drives). The corrected
    # model must land on that order, not 7x below it.
    assert 0.015 < math.sqrt(fixed[1, 1]) < 0.035
    assert fixed[0, 0] / fixed[1, 1] < 30.0


def test_increments_telescope_to_the_total():
    """Step-by-step integration must equal the closed form for the whole distance.

    This is the property that keeps a HELD offset from averaging away like white noise.
    """
    theta, total_distance, steps = 0.6, 1.1, 40
    step = total_distance / steps
    summed = np.zeros((3, 3))
    for index in range(steps):
        summed += coherent_drift_increment(index * step, (index + 1) * step, theta)
    np.testing.assert_allclose(summed, coherent_drift_block(total_distance, theta), rtol=1e-10)


def test_summing_blocks_naively_would_understate_it():
    """Guards the mistake the telescoping form exists to prevent."""
    theta, total_distance, steps = 0.0, 1.1, 50
    naive = sum(coherent_drift_block(total_distance / steps, theta) for _ in range(steps))
    correct = coherent_drift_block(total_distance, theta)
    # Naive per-step summation is sqrt(n) too small in standard deviation.
    assert math.sqrt(correct[1, 1] / naive[1, 1]) == pytest.approx(math.sqrt(steps), rel=1e-9)


def test_fallback_branch_also_honours_the_flag():
    """The simplified diagonal branch (no theta/v) must not silently ignore it."""
    plain = unicycle_process_noise(0.01, 0.02, 0.1, base_dt=0.1)
    flagged = unicycle_process_noise(0.01, 0.02, 0.1, base_dt=0.1, coherent_drift=True)
    # No theta/v means no distance is known, so it must fall through unchanged.
    np.testing.assert_array_equal(plain, flagged)
