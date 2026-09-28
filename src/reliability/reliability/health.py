"""Camera health state machine for occlusion/dropout/reacquisition logic."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any

from reliability.contracts import CameraObservation


class CameraHealthState(str, Enum):
    TRACKING = "TRACKING"
    DEGRADED = "DEGRADED"
    LOST = "LOST"
    REACQUIRING = "REACQUIRING"


@dataclass(frozen=True)
class CameraHealthConfig:
    stale_age_s: float = 1.25
    degraded_miss_count: int = 2
    lost_miss_count: int = 5
    degraded_high_nis_count: int = 2
    lost_high_nis_count: int = 4
    high_nis_threshold: float = 9.21
    reacquire_required_hits: int = 2
    reacquire_max_nis: float = 9.21
    low_reliability_threshold: float = 0.2


@dataclass(frozen=True)
class CameraHealth:
    camera_id: str
    state: CameraHealthState
    consecutive_hits: int
    consecutive_misses: int
    consecutive_high_nis: int
    last_timestamp_s: float
    reason: str = ""

    @property
    def tracking(self) -> bool:
        return self.state == CameraHealthState.TRACKING

    @property
    def lost(self) -> bool:
        return self.state == CameraHealthState.LOST


class CameraHealthMachine:
    """Small deterministic camera-health state machine.

    The machine intentionally uses only operational evidence: detection validity,
    age/staleness, NIS/gating status, and predicted availability.
    """

    def __init__(
        self,
        camera_id: str = "camera_A",
        config: CameraHealthConfig | None = None,
        *,
        initial_state: CameraHealthState = CameraHealthState.TRACKING,
    ):
        self.camera_id = str(camera_id)
        self.config = config or CameraHealthConfig()
        self._state = initial_state
        self._hits = 0
        self._misses = 0
        self._high_nis = 0
        self._last_timestamp_s = math.nan
        self._reason = "init"

    @property
    def health(self) -> CameraHealth:
        return CameraHealth(
            camera_id=self.camera_id,
            state=self._state,
            consecutive_hits=self._hits,
            consecutive_misses=self._misses,
            consecutive_high_nis=self._high_nis,
            last_timestamp_s=self._last_timestamp_s,
            reason=self._reason,
        )

    def update(
        self,
        observation: CameraObservation | None,
        *,
        nis: float | None = None,
        accepted: bool | None = None,
        reliability: float | None = None,
    ) -> CameraHealth:
        cfg = self.config
        if observation is not None:
            self._last_timestamp_s = float(observation.timestamp_s)
            camera_id = observation.camera_id or self.camera_id
            self.camera_id = str(camera_id)

        stale = bool(
            observation is None
            or observation.measurement_age_s > max(float(cfg.stale_age_s), 0.0)
        )
        valid_detection = bool(observation is not None and observation.detection_valid and not stale)
        if reliability is None and observation is not None:
            reliability = float(observation.availability_probability)
        low_reliability = bool(
            reliability is not None
            and math.isfinite(float(reliability))
            and float(reliability) < float(cfg.low_reliability_threshold)
        )
        gate_rejected = accepted is False
        high_nis = bool(
            nis is not None
            and math.isfinite(float(nis))
            and float(nis) > float(cfg.high_nis_threshold)
        )

        good_hit = valid_detection and not gate_rejected and not high_nis and not low_reliability
        bad_sample = (not valid_detection) or gate_rejected or high_nis or low_reliability

        if good_hit:
            self._hits += 1
            self._misses = 0
            self._high_nis = 0
            self._reason = "consistent_detection"
        else:
            self._hits = 0
            if bad_sample:
                self._misses += 1
            if high_nis or gate_rejected:
                self._high_nis += 1
            self._reason = self._bad_reason(
                stale=stale,
                gate_rejected=gate_rejected,
                high_nis=high_nis,
                low_reliability=low_reliability,
                valid_detection=valid_detection,
            )

        if self._state == CameraHealthState.TRACKING:
            if self._misses >= cfg.lost_miss_count or self._high_nis >= cfg.lost_high_nis_count:
                self._state = CameraHealthState.LOST
            elif self._misses >= cfg.degraded_miss_count or self._high_nis >= cfg.degraded_high_nis_count:
                self._state = CameraHealthState.DEGRADED
        elif self._state == CameraHealthState.DEGRADED:
            if self._misses >= cfg.lost_miss_count or self._high_nis >= cfg.lost_high_nis_count:
                self._state = CameraHealthState.LOST
            elif good_hit:
                self._state = CameraHealthState.TRACKING
        elif self._state == CameraHealthState.LOST:
            if valid_detection:
                self._state = CameraHealthState.REACQUIRING
                self._hits = 1 if good_hit or self._nis_ok_for_reacquire(nis, cfg) else 0
        elif self._state == CameraHealthState.REACQUIRING:
            if not valid_detection or gate_rejected or not self._nis_ok_for_reacquire(nis, cfg):
                self._state = CameraHealthState.LOST
                self._hits = 0
            elif self._hits >= cfg.reacquire_required_hits:
                self._state = CameraHealthState.TRACKING

        return self.health

    @staticmethod
    def _nis_ok_for_reacquire(nis: float | None, cfg: CameraHealthConfig) -> bool:
        if nis is None:
            return True
        try:
            value = float(nis)
        except (TypeError, ValueError):
            return True
        return (not math.isfinite(value)) or value <= float(cfg.reacquire_max_nis)

    @staticmethod
    def _bad_reason(
        *,
        stale: bool,
        gate_rejected: bool,
        high_nis: bool,
        low_reliability: bool,
        valid_detection: bool,
    ) -> str:
        if stale:
            return "stale_or_missing"
        if not valid_detection:
            return "detector_miss"
        if gate_rejected:
            return "gate_rejected"
        if high_nis:
            return "high_nis"
        if low_reliability:
            return "low_reliability"
        return "bad_sample"


def update_health_by_camera(
    machines: dict[str, CameraHealthMachine],
    observation: CameraObservation,
    *,
    config: CameraHealthConfig | None = None,
    **kwargs: Any,
) -> CameraHealth:
    camera_id = str(observation.camera_id or "camera_A")
    if camera_id not in machines:
        machines[camera_id] = CameraHealthMachine(camera_id=camera_id, config=config)
    return machines[camera_id].update(observation, **kwargs)
