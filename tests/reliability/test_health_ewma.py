from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "reliability"))

from reliability.health_ewma import (  # noqa: E402
    CalibrationHealthState,
    HealthDebouncer,
    HealthDebouncerConfig,
    InnovationHealthConfig,
    InnovationHealthMonitor,
    isolate_suspect_camera,
)


def _monitor(**overrides) -> InnovationHealthMonitor:
    defaults = dict(
        rho=0.5,
        rho_bias=0.5,
        m0=2.0,
        eta0=3.0,
        eta1=1.0,
        eta2=1.0,
        eta3=2.0,
        eta4=1.0,
        drop_rate_window=4,
    )
    defaults.update(overrides)
    return InnovationHealthMonitor(config=InnovationHealthConfig(**defaults))


class TestConfigValidation:
    def test_rho_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            InnovationHealthConfig(rho=0.0)
        with pytest.raises(ValueError):
            InnovationHealthConfig(rho=1.5)
        with pytest.raises(ValueError):
            InnovationHealthConfig(rho_bias=-0.1)

    def test_negative_eta_rejected(self):
        with pytest.raises(ValueError):
            InnovationHealthConfig(eta1=-1.0)
        with pytest.raises(ValueError):
            InnovationHealthConfig(eta4=float("nan"))

    def test_bad_m0_and_window_rejected(self):
        with pytest.raises(ValueError):
            InnovationHealthConfig(m0=0.0)
        with pytest.raises(ValueError):
            InnovationHealthConfig(m0=float("inf"))
        with pytest.raises(ValueError):
            InnovationHealthConfig(drop_rate_window=0)


class TestEwmaAlgebra:
    def test_nis_ewma_hand_checked(self):
        mon = _monitor()  # rho = 0.5, m0 = 2.0
        assert mon.nis_ewma == pytest.approx(2.0)
        mon.update(nis=4.0, innovation_uv=None, dropped=False)
        assert mon.nis_ewma == pytest.approx(0.5 * 2.0 + 0.5 * 4.0)  # 3.0
        mon.update(nis=4.0, innovation_uv=None, dropped=False)
        assert mon.nis_ewma == pytest.approx(0.5 * 3.0 + 0.5 * 4.0)  # 3.5

    def test_bias_ewma_hand_checked(self):
        mon = _monitor()  # rho_bias = 0.5, b init (0, 0)
        mon.update(nis=None, innovation_uv=(2.0, 0.0), dropped=False)
        assert mon.bias_ewma == pytest.approx((1.0, 0.0))
        mon.update(nis=None, innovation_uv=(2.0, 0.0), dropped=False)
        assert mon.bias_ewma == pytest.approx((1.5, 0.0))

    def test_none_skips_ewma_but_counts_drop_window(self):
        mon = _monitor(drop_rate_window=2)
        mon.update(nis=None, innovation_uv=None, dropped=True)
        assert mon.nis_ewma == pytest.approx(2.0)  # untouched
        assert mon.bias_ewma == pytest.approx((0.0, 0.0))
        assert mon.drop_rate == pytest.approx(1.0)
        mon.update(nis=2.0, innovation_uv=(0.0, 0.0), dropped=False)
        assert mon.drop_rate == pytest.approx(0.5)

    def test_drop_rate_sliding_window(self):
        mon = _monitor(drop_rate_window=4)
        for dropped in (True, True, False, False):
            mon.update(nis=None, innovation_uv=None, dropped=dropped)
        assert mon.drop_rate == pytest.approx(0.5)
        # window slides: the two drops fall out
        for _ in range(4):
            mon.update(nis=None, innovation_uv=None, dropped=False)
        assert mon.drop_rate == pytest.approx(0.0)


class TestHealthResponse:
    def test_biased_stream_drives_health_down(self):
        mon = _monitor(rho_bias=0.2, eta2=3.0)
        h0 = mon.health
        healths = []
        for _ in range(30):
            # NIS at nominal mean, but innovation direction is constant:
            # the bias EWMA is the only detector that fires.
            healths.append(mon.update(nis=2.0, innovation_uv=(1.0, 0.5), dropped=False))
        bias_norm = (mon.bias_ewma[0] ** 2 + mon.bias_ewma[1] ** 2) ** 0.5
        assert bias_norm > 0.9  # converged toward |(1, 0.5)| ≈ 1.118
        assert healths[-1] < h0
        assert healths[-1] < 0.5 * h0

    def test_monotonic_in_nis(self):
        good, bad = _monitor(), _monitor()
        for mon in (good, bad):
            mon.update(nis=2.0, innovation_uv=(0.0, 0.0), dropped=False)
        h_good = good.update(nis=2.0, innovation_uv=None, dropped=False)
        h_bad = bad.update(nis=50.0, innovation_uv=None, dropped=False)
        assert h_bad < h_good

    def test_monotonic_in_drop_rate(self):
        good, bad = _monitor(), _monitor()
        h_good = good.update(nis=None, innovation_uv=None, dropped=False)
        h_bad = bad.update(nis=None, innovation_uv=None, dropped=True)
        assert h_bad < h_good

    def test_monotonic_in_cross_disagreement(self):
        good, bad = _monitor(), _monitor()
        h_none = good.update(nis=2.0, innovation_uv=None, dropped=False, cross_disagreement=None)
        h_zero = _monitor().update(nis=2.0, innovation_uv=None, dropped=False, cross_disagreement=0.0)
        h_bad = bad.update(nis=2.0, innovation_uv=None, dropped=False, cross_disagreement=3.0)
        assert h_none == pytest.approx(h_zero)  # None -> zero contribution
        assert h_bad < h_none

    def test_nis_below_nominal_never_boosts_health(self):
        # max(0, m - m0) clamps: very small NIS is not "extra healthy".
        base, low = _monitor(), _monitor()
        h_base = base.update(nis=2.0, innovation_uv=None, dropped=False)
        h_low = low.update(nis=0.0, innovation_uv=None, dropped=False)
        assert h_low == pytest.approx(h_base)

    def test_invalid_update_inputs_rejected(self):
        mon = _monitor()
        with pytest.raises(ValueError):
            mon.update(nis=-1.0, innovation_uv=None, dropped=False)
        with pytest.raises(ValueError):
            mon.update(nis=None, innovation_uv=(1.0,), dropped=False)
        with pytest.raises(ValueError):
            mon.update(nis=None, innovation_uv=None, dropped=False, cross_disagreement=-0.5)


class TestHealthDebouncer:
    def _debouncer(self, m_s=2, m_d=2, m_r=2, m_h=2) -> HealthDebouncer:
        return HealthDebouncer(HealthDebouncerConfig(m_s=m_s, m_d=m_d, m_r=m_r, m_h=m_h))

    def test_config_validation(self):
        with pytest.raises(ValueError):
            HealthDebouncerConfig(m_s=0)
        with pytest.raises(ValueError):
            HealthDebouncerConfig(m_h=-1)

    def test_single_bad_window_never_leaves_healthy(self):
        deb = self._debouncer(m_s=3)
        assert deb.step(False) == CalibrationHealthState.HEALTHY
        assert deb.step(True) == CalibrationHealthState.HEALTHY
        # non-consecutive inconsistents never accumulate
        for _ in range(10):
            deb.step(False)
            state = deb.step(True)
        assert state == CalibrationHealthState.HEALTHY

    def test_full_transition_walk(self):
        deb = self._debouncer(m_s=2, m_d=2, m_r=2, m_h=2)
        # HEALTHY --2 inconsistent--> SUSPECT
        deb.step(False)
        assert deb.state == CalibrationHealthState.HEALTHY
        assert deb.step(False) == CalibrationHealthState.SUSPECT
        # SUSPECT --2 more inconsistent--> DEGRADED
        deb.step(False)
        assert deb.state == CalibrationHealthState.SUSPECT
        assert deb.step(False) == CalibrationHealthState.DEGRADED
        # DEGRADED is absorbing without reset()
        for consistent in (True, True, True, False):
            assert deb.step(consistent) == CalibrationHealthState.DEGRADED
        # DEGRADED + reset() --> RECOVERING
        assert deb.reset() == CalibrationHealthState.RECOVERING
        # RECOVERING --2 consistent--> HEALTHY
        deb.step(True)
        assert deb.state == CalibrationHealthState.RECOVERING
        assert deb.step(True) == CalibrationHealthState.HEALTHY

    def test_suspect_recovers_to_healthy(self):
        deb = self._debouncer(m_s=2, m_r=3)
        deb.step(False)
        deb.step(False)
        assert deb.state == CalibrationHealthState.SUSPECT
        deb.step(True)
        deb.step(True)
        assert deb.state == CalibrationHealthState.SUSPECT
        assert deb.step(True) == CalibrationHealthState.HEALTHY

    def test_recovering_relapse_falls_back_to_degraded(self):
        deb = self._debouncer(m_s=2, m_d=2, m_h=3)
        for _ in range(4):
            deb.step(False)
        assert deb.state == CalibrationHealthState.DEGRADED
        deb.reset()
        deb.step(True)
        deb.step(True)
        # documented decision: any inconsistent during RECOVERING -> DEGRADED
        assert deb.step(False) == CalibrationHealthState.DEGRADED
        # and it needs a fresh reset() to try again
        assert deb.step(True) == CalibrationHealthState.DEGRADED
        assert deb.reset() == CalibrationHealthState.RECOVERING

    def test_reset_is_noop_outside_degraded(self):
        deb = self._debouncer()
        assert deb.reset() == CalibrationHealthState.HEALTHY

    def test_response_policy(self):
        assert HealthDebouncer.response_policy(CalibrationHealthState.HEALTHY) == "accept"
        assert HealthDebouncer.response_policy(CalibrationHealthState.SUSPECT) == "inflate"
        assert HealthDebouncer.response_policy(CalibrationHealthState.DEGRADED) == "reject"
        assert HealthDebouncer.response_policy(CalibrationHealthState.RECOVERING) == "slow_reentry"
        with pytest.raises(ValueError):
            HealthDebouncer.response_policy("BROKEN")


class TestIsolateSuspectCamera:
    THRESHOLD = 9.21

    def test_three_camera_odd_one_out(self):
        pairwise = {("A", "B"): 20.0, ("A", "C"): 18.0, ("B", "C"): 1.0}
        assert isolate_suspect_camera(pairwise, self.THRESHOLD) == "A"

    def test_ambiguous_pattern_returns_none(self):
        # single bad pair among 3 cameras: could be either A or B
        pairwise = {("A", "B"): 20.0, ("A", "C"): 1.0, ("B", "C"): 1.0}
        assert isolate_suspect_camera(pairwise, self.THRESHOLD) is None
        # disjoint bad pairs among 4 cameras: no common camera
        pairwise4 = {
            ("A", "B"): 20.0,
            ("C", "D"): 20.0,
            ("A", "C"): 1.0,
            ("A", "D"): 1.0,
            ("B", "C"): 1.0,
            ("B", "D"): 1.0,
        }
        assert isolate_suspect_camera(pairwise4, self.THRESHOLD) is None

    def test_two_cameras_always_none(self):
        assert isolate_suspect_camera({("A", "B"): 100.0}, self.THRESHOLD) is None

    def test_all_consistent_returns_none(self):
        pairwise = {("A", "B"): 1.0, ("A", "C"): 2.0, ("B", "C"): 0.5}
        assert isolate_suspect_camera(pairwise, self.THRESHOLD) is None

    def test_missing_pair_evidence_returns_none(self):
        # A disagrees with both observed peers, but without B-C evidence it
        # is impossible to establish that B and C agree with each other.
        pairwise = {("A", "B"): 20.0, ("C", "A"): 18.0}
        assert isolate_suspect_camera(pairwise, self.THRESHOLD) is None

    def test_four_camera_partial_incident_disagreements_return_none(self):
        # A disagrees with B and C but agrees with D, so A is not a valid
        # odd-one-out even though it is common to every bad pair.
        pairwise = {
            ("A", "B"): 20.0,
            ("A", "C"): 18.0,
            ("A", "D"): 1.0,
            ("B", "C"): 1.0,
            ("B", "D"): 1.0,
            ("C", "D"): 1.0,
        }
        assert isolate_suspect_camera(pairwise, self.THRESHOLD) is None

    def test_reversed_duplicate_pair_rejected(self):
        pairwise = {
            ("A", "B"): 20.0,
            ("B", "A"): 20.0,
            ("A", "C"): 18.0,
            ("B", "C"): 1.0,
        }
        with pytest.raises(ValueError, match="duplicate evidence"):
            isolate_suspect_camera(pairwise, self.THRESHOLD)

    @pytest.mark.parametrize("threshold", [-0.1, float("nan"), float("inf")])
    def test_invalid_threshold_rejected(self, threshold):
        with pytest.raises(ValueError):
            isolate_suspect_camera({("A", "B"): 1.0}, threshold)

    def test_invalid_pair_key_rejected(self):
        with pytest.raises(ValueError):
            isolate_suspect_camera({("A", "A"): 20.0}, self.THRESHOLD)
