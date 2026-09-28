"""Invalid numerical results must never masquerade as accepted corrections."""
from dataclasses import replace

import numpy as np
import pytest

from planning.core import belief_correction as bc


def linearization():
    return bc.Linearization(z=np.array([.01, -.02]), mu_y=np.zeros(2),
                            Gamma=np.eye(3)[:, :2], Sigma_y=2*np.eye(2),
                            R_eff=np.eye(2), S_eff=np.eye(3))


@pytest.mark.parametrize("field", ["z", "mu_y", "Gamma", "Sigma_y", "R_eff", "S_eff"])
def test_nonfinite_measurement_moments_are_refused(field):
    lin = linearization()
    value = np.array(getattr(lin, field), copy=True)
    value.flat[0] = np.nan
    messages = []
    result = bc.compute_update(np.zeros(3), replace(lin, **{field: value}),
                               cov_eig_floor=1e-9, on_shape_error=messages.append)
    assert result is None
    assert messages


@pytest.mark.parametrize("field,value", [("Gamma", np.ones(2)), ("Gamma", np.ones((2, 2))),
                                        ("S_eff", np.eye(2)), ("R_eff", np.eye(3))])
def test_malformed_moment_shapes_are_refused_without_a_numpy_exception(field, value):
    assert bc.compute_update(np.zeros(3), replace(linearization(), **{field: value}),
                             cov_eig_floor=1e-9) is None


@pytest.mark.parametrize("covariance", [np.diag([-1., 1.]), np.array([[1., 2.], [0., 1.]])])
def test_invalid_innovation_covariance_is_not_silently_symmetrized_or_inverted(covariance):
    assert bc.compute_update(np.zeros(3), replace(linearization(), Sigma_y=covariance),
                             cov_eig_floor=1e-9) is None


@pytest.mark.parametrize("age,dt,reason", [(np.nan, .1, bc.RejectReason.STALE_AGE),
                                          (0., np.nan, bc.RejectReason.DT_IMPLAUSIBLE),
                                          (0., -.1, bc.RejectReason.DT_IMPLAUSIBLE)])
def test_execution_requires_known_chronological_times_before_reading_state(age, dt, reason):
    class Source:
        def snapshot(self):
            pytest.fail("invalid timing reached the state snapshot")

    outcome = bc.apply_correction(source=Source(), gates=bc.CorrectionGates(skip_stale=False),
                                  replay=None, age=age, dt_s=dt)
    assert outcome.reason is reason
    assert not outcome.accepted


def test_nonfinite_computed_nis_cannot_bypass_the_statistical_gate(monkeypatch):
    snapshot = bc.CorrectionSnapshot(np.zeros(3), np.eye(3), None, np.zeros(2), np.zeros(2))
    source = bc.FusedMapMeasurementSource(snapshot_fn=lambda: snapshot,
                                          measurement_cov_fn=lambda: np.eye(2))
    monkeypatch.setattr(bc, "normalized_innovation_squared", lambda *args: np.nan)
    result = bc.apply_correction(source=source, gates=bc.CorrectionGates(),
        replay=lambda m, P, *args: (m, P, {}), age=.1, dt_s=.1)
    assert result.reason is bc.RejectReason.UPDATE_FAILED
    assert result.next_m is None and result.next_S is None
