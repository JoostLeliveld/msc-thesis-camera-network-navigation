"""Innovation-driven calibration-health layer (plan 08, health monitor v2).

This module is the *calibration-health* companion to
:mod:`reliability.health` (`CameraHealthMachine`), which remains the
availability layer.  Everything here is driven purely by operational
evidence — NIS, innovation vectors, drop events and cross-camera
disagreement — never by ground truth.

Components
----------
- ``InnovationHealthMonitor``: per-camera continuous health ``h in (0, 1)``
  from EWMAs of NIS and of the innovation mean (persistent-bias detector),
  a sliding-window drop rate, and a cross-camera disagreement term:

      h = sigmoid(eta0 - eta1*max(0, m - m0) - eta2*|b| - eta3*drop_rate - eta4*cross)

- ``HealthDebouncer``: debounced HEALTHY/SUSPECT/DEGRADED/RECOVERING state
  machine so single bad windows never flip the state, plus the response
  policy mapping (accept / inflate / reject / slow_reentry).

- ``isolate_suspect_camera``: odd-one-out isolation from pairwise
  disagreement with >= 3 cameras; with only 2 cameras the evidence is
  inherently ambiguous and ``None`` is returned.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import math
from typing import Mapping


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-min(x, 700.0)))
    e = math.exp(max(x, -700.0))
    return e / (1.0 + e)


def _require_finite(value: float, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _require_rate(value: float, name: str) -> float:
    out = _require_finite(value, name)
    if not 0.0 < out <= 1.0:
        raise ValueError(f"{name} must be in (0, 1]")
    return out


@dataclass(frozen=True)
class InnovationHealthConfig:
    """Configuration for :class:`InnovationHealthMonitor`.

    ``m0`` defaults to 2.0, the nominal mean of a chi-square NIS with
    2 degrees of freedom (2-D position innovation).
    """

    rho: float = 0.1
    rho_bias: float = 0.05
    m0: float = 2.0
    eta0: float = 3.0
    eta1: float = 1.0
    eta2: float = 1.0
    eta3: float = 2.0
    eta4: float = 1.0
    drop_rate_window: int = 20

    def __post_init__(self) -> None:
        object.__setattr__(self, "rho", _require_rate(self.rho, "rho"))
        object.__setattr__(self, "rho_bias", _require_rate(self.rho_bias, "rho_bias"))
        m0 = _require_finite(self.m0, "m0")
        if m0 <= 0.0:
            raise ValueError("m0 must be positive")
        object.__setattr__(self, "m0", m0)
        object.__setattr__(self, "eta0", _require_finite(self.eta0, "eta0"))
        for name in ("eta1", "eta2", "eta3", "eta4"):
            value = _require_finite(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        window = self.drop_rate_window
        if not isinstance(window, int) or isinstance(window, bool) or window < 1:
            raise ValueError("drop_rate_window must be a positive int")


class InnovationHealthMonitor:
    """Per-camera continuous calibration-health monitor.

    Maintains:
    - ``m``: EWMA of NIS (initialised at the nominal mean ``m0``),
    - ``b``: EWMA of the raw innovation vector (persistent-bias detector;
      a biased camera has consistent-direction innovations even when its
      NIS looks tolerable),
    - drop rate over a sliding window of the last ``drop_rate_window``
      update calls.

    ``update`` with ``nis=None`` / ``innovation_uv=None`` means "no
    measurement this cycle": the corresponding EWMA is left untouched,
    but the call still counts toward the drop window (``dropped=True``
    marks it as a drop).
    """

    def __init__(self, camera_id: str = "camera_A", config: InnovationHealthConfig | None = None):
        self.camera_id = str(camera_id)
        self.config = config or InnovationHealthConfig()
        self._m = float(self.config.m0)
        self._b = (0.0, 0.0)
        self._drops: deque[bool] = deque(maxlen=self.config.drop_rate_window)
        self._health = self._compute_health(cross=0.0)

    @property
    def nis_ewma(self) -> float:
        return self._m

    @property
    def bias_ewma(self) -> tuple[float, float]:
        return self._b

    @property
    def drop_rate(self) -> float:
        if not self._drops:
            return 0.0
        return sum(1.0 for d in self._drops if d) / float(len(self._drops))

    @property
    def health(self) -> float:
        return self._health

    def update(
        self,
        nis: float | None,
        innovation_uv: tuple[float, float] | None,
        dropped: bool,
        cross_disagreement: float | None = None,
    ) -> float:
        cfg = self.config
        if nis is not None:
            value = _require_finite(nis, "nis")
            if value < 0.0:
                raise ValueError("nis must be non-negative")
            self._m = (1.0 - cfg.rho) * self._m + cfg.rho * value
        if innovation_uv is not None:
            if len(innovation_uv) != 2:
                raise ValueError("innovation_uv must be a 2-element sequence")
            nu = (
                _require_finite(innovation_uv[0], "innovation_uv[0]"),
                _require_finite(innovation_uv[1], "innovation_uv[1]"),
            )
            self._b = (
                (1.0 - cfg.rho_bias) * self._b[0] + cfg.rho_bias * nu[0],
                (1.0 - cfg.rho_bias) * self._b[1] + cfg.rho_bias * nu[1],
            )
        self._drops.append(bool(dropped))
        if cross_disagreement is None:
            cross = 0.0
        else:
            cross = _require_finite(cross_disagreement, "cross_disagreement")
            if cross < 0.0:
                raise ValueError("cross_disagreement must be non-negative")
        self._health = self._compute_health(cross=cross)
        return self._health

    def _compute_health(self, *, cross: float) -> float:
        cfg = self.config
        bias_norm = math.hypot(self._b[0], self._b[1])
        score = (
            cfg.eta0
            - cfg.eta1 * max(0.0, self._m - cfg.m0)
            - cfg.eta2 * bias_norm
            - cfg.eta3 * self.drop_rate
            - cfg.eta4 * cross
        )
        return _sigmoid(score)


class CalibrationHealthState(str, Enum):
    """Calibration-health states (distinct from health.CameraHealthState)."""

    HEALTHY = "HEALTHY"
    SUSPECT = "SUSPECT"
    DEGRADED = "DEGRADED"
    RECOVERING = "RECOVERING"


@dataclass(frozen=True)
class HealthDebouncerConfig:
    """Consecutive-window counts for :class:`HealthDebouncer`.

    - ``m_s``: consecutive inconsistent samples in HEALTHY before SUSPECT.
    - ``m_d``: consecutive inconsistent samples in SUSPECT before DEGRADED.
    - ``m_r``: consecutive consistent samples in SUSPECT before HEALTHY.
    - ``m_h``: consecutive consistent samples in RECOVERING before HEALTHY.
    """

    m_s: int = 3
    m_d: int = 3
    m_r: int = 3
    m_h: int = 5

    def __post_init__(self) -> None:
        for name in ("m_s", "m_d", "m_r", "m_h"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive int")


class HealthDebouncer:
    """Debounced calibration-health state machine (plan 08).

    Transitions (all counts are *consecutive*; an opposite sample resets
    the corresponding counter):

    - HEALTHY  --(m_s inconsistent)--> SUSPECT
    - SUSPECT  --(m_d inconsistent)--> DEGRADED
    - SUSPECT  --(m_r consistent)-->   HEALTHY
    - DEGRADED --reset()-->            RECOVERING (DEGRADED is otherwise absorbing)
    - RECOVERING --(m_h consistent)--> HEALTHY

    Design decision (documented per plan 08): any inconsistent sample in
    RECOVERING falls back to DEGRADED (not merely a counter reset) — a
    camera that misbehaves during slow re-entry has not proven itself and
    must be explicitly reset() again. This is the conservative choice and
    keeps "single rejected observation never flips a HEALTHY state" while
    making re-entry strict.

    Response policy: HEALTHY=accept, SUSPECT=inflate (covariance inflation
    only), DEGRADED=reject (hard reject), RECOVERING=slow_reentry.
    """

    def __init__(self, config: HealthDebouncerConfig | None = None):
        self.config = config or HealthDebouncerConfig()
        self._state = CalibrationHealthState.HEALTHY
        self._consistent = 0
        self._inconsistent = 0

    @property
    def state(self) -> CalibrationHealthState:
        return self._state

    @property
    def consecutive_consistent(self) -> int:
        return self._consistent

    @property
    def consecutive_inconsistent(self) -> int:
        return self._inconsistent

    def step(self, consistent: bool) -> CalibrationHealthState:
        cfg = self.config
        if consistent:
            self._consistent += 1
            self._inconsistent = 0
        else:
            self._inconsistent += 1
            self._consistent = 0

        if self._state == CalibrationHealthState.HEALTHY:
            if self._inconsistent >= cfg.m_s:
                self._transition(CalibrationHealthState.SUSPECT)
        elif self._state == CalibrationHealthState.SUSPECT:
            if self._inconsistent >= cfg.m_d:
                self._transition(CalibrationHealthState.DEGRADED)
            elif self._consistent >= cfg.m_r:
                self._transition(CalibrationHealthState.HEALTHY)
        elif self._state == CalibrationHealthState.DEGRADED:
            pass  # absorbing until an explicit reset()
        elif self._state == CalibrationHealthState.RECOVERING:
            if not consistent:
                self._transition(CalibrationHealthState.DEGRADED)
            elif self._consistent >= cfg.m_h:
                self._transition(CalibrationHealthState.HEALTHY)
        return self._state

    def reset(self) -> CalibrationHealthState:
        """Operator/recalibration acknowledgement: DEGRADED -> RECOVERING.

        A no-op in any other state (there is nothing to recover from).
        """

        if self._state == CalibrationHealthState.DEGRADED:
            self._transition(CalibrationHealthState.RECOVERING)
        return self._state

    def _transition(self, state: CalibrationHealthState) -> None:
        self._state = state
        self._consistent = 0
        self._inconsistent = 0

    @staticmethod
    def response_policy(state: CalibrationHealthState) -> str:
        policies = {
            CalibrationHealthState.HEALTHY: "accept",
            CalibrationHealthState.SUSPECT: "inflate",
            CalibrationHealthState.DEGRADED: "reject",
            CalibrationHealthState.RECOVERING: "slow_reentry",
        }
        try:
            return policies[CalibrationHealthState(state)]
        except (KeyError, ValueError) as exc:
            raise ValueError(f"unknown calibration health state: {state!r}") from exc


def isolate_suspect_camera(
    pairwise_d2: Mapping[tuple[str, str], float],
    threshold: float,
) -> str | None:
    """Odd-one-out isolation from pairwise disagreement statistics.

    ``pairwise_d2`` maps unordered camera pairs to a disagreement statistic
    (e.g. squared Mahalanobis distance / NIS-like d2). Reversed keys denote
    the same unordered pair and are therefore rejected as duplicate evidence.
    A camera is isolated only from a complete pairwise graph with >= 3
    cameras, when all of its incident pairs are above threshold and every
    pair among the remaining cameras is consistent (at or below threshold).
    Anything else — including only 2 cameras or incomplete evidence — is
    ambiguous and returns ``None``.
    """

    threshold = _require_finite(threshold, "threshold")
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")

    cameras: set[str] = set()
    values_by_pair: dict[frozenset[str], float] = {}
    for pair, d2 in pairwise_d2.items():
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("pairwise_d2 keys must be pairs of distinct camera ids")
        a, b = str(pair[0]), str(pair[1])
        if a == b:
            raise ValueError("pairwise_d2 keys must be pairs of distinct camera ids")
        canonical_pair = frozenset((a, b))
        if canonical_pair in values_by_pair:
            raise ValueError(f"duplicate evidence for unordered camera pair {pair!r}")
        cameras.update((a, b))
        value = _require_finite(d2, f"pairwise_d2[{pair!r}]")
        values_by_pair[canonical_pair] = value

    if len(cameras) < 3:
        return None

    camera_ids = sorted(cameras)
    expected_pairs = {
        frozenset((camera_a, camera_b))
        for index, camera_a in enumerate(camera_ids)
        for camera_b in camera_ids[index + 1 :]
    }
    if set(values_by_pair) != expected_pairs:
        return None

    suspects: list[str] = []
    for suspect in camera_ids:
        others = [camera_id for camera_id in camera_ids if camera_id != suspect]
        all_incident_pairs_bad = all(
            values_by_pair[frozenset((suspect, other))] > threshold
            for other in others
        )
        all_other_pairs_consistent = all(
            values_by_pair[frozenset((camera_a, camera_b))] <= threshold
            for index, camera_a in enumerate(others)
            for camera_b in others[index + 1 :]
        )
        if all_incident_pairs_bad and all_other_pairs_consistent:
            suspects.append(suspect)

    return suspects[0] if len(suspects) == 1 else None
