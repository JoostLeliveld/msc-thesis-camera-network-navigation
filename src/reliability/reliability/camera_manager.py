"""Operational camera selection and hysteretic handover.

This module deliberately accepts only :class:`~reliability.fusion.MapObservation`
records.  A map observation already represents a detected, associated, and
projectable camera measurement; ground truth and evaluation labels cannot enter
this interface.  The manager selects at most one observation.  Fusion remains a
separate, optional experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

from reliability.contracts import ContractValidationError
from reliability.fusion import MapObservation
from unav_common.config import parse_bool


@dataclass(frozen=True)
class CameraManagerConfig:
    """Operational gates and switching hysteresis for a camera manager.

    ``min_spatial_trust`` is evaluated through the conservative operational
    score ``p_available * association * exp(-epistemic) * freshness``.  This is
    intentionally a scalar selection criterion; converting the selected
    observation to estimator covariance remains the estimator/reliability
    provider's responsibility.
    """

    min_spatial_trust: float = 0.45
    min_association_confidence: float = 0.70
    max_measurement_age_s: float = 0.15
    age_decay_s: float = 0.15
    candidate_score_margin: float = 0.08
    required_consecutive_better_frames: int = 3
    max_cross_camera_disagreement_m: float = 0.30
    max_overlap_time_delta_s: float = 0.05
    require_consistency_when_source_available: bool = True
    fallback_on_active_camera_loss: bool = True
    allowed_camera_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "min_spatial_trust",
            "min_association_confidence",
            "candidate_score_margin",
        ):
            object.__setattr__(self, name, _probability(getattr(self, name), name))
        for name in (
            "max_measurement_age_s",
            "max_cross_camera_disagreement_m",
            "max_overlap_time_delta_s",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        object.__setattr__(self, "age_decay_s", _positive(self.age_decay_s, "age_decay_s"))
        if isinstance(self.required_consecutive_better_frames, bool):
            raise ContractValidationError("required_consecutive_better_frames must be a positive integer")
        try:
            required = int(self.required_consecutive_better_frames)
        except (TypeError, ValueError) as exc:
            raise ContractValidationError("required_consecutive_better_frames must be a positive integer") from exc
        if required < 1 or required != self.required_consecutive_better_frames:
            raise ContractValidationError("required_consecutive_better_frames must be a positive integer")
        object.__setattr__(self, "required_consecutive_better_frames", required)
        ids = tuple(str(camera_id).strip() for camera_id in self.allowed_camera_ids)
        if any(not camera_id for camera_id in ids) or len(set(ids)) != len(ids):
            raise ContractValidationError("allowed_camera_ids must contain unique non-empty IDs")
        object.__setattr__(self, "allowed_camera_ids", ids)
        for name in (
            "require_consistency_when_source_available",
            "fallback_on_active_camera_loss",
        ):
            try:
                value = parse_bool(getattr(self, name), field_name=name)
            except ValueError as exc:
                raise ContractValidationError(str(exc)) from exc
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class CameraManagerDecision:
    """One operational selection decision, suitable for an experiment log."""

    timestamp_s: float
    selected_observation: MapObservation | None
    active_camera_id: str
    previous_camera_id: str
    switched: bool
    fallback: bool
    candidate_camera_id: str
    candidate_streak: int
    scores_by_camera: Mapping[str, float] = field(default_factory=dict)
    rejected_by_camera: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()

    @property
    def selected_camera_id(self) -> str:
        return self.selected_observation.camera_id if self.selected_observation is not None else ""

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe, operational-only decision summary."""

        return {
            "timestamp_s": self.timestamp_s,
            "selected_camera_id": self.selected_camera_id,
            "active_camera_id": self.active_camera_id,
            "previous_camera_id": self.previous_camera_id,
            "switched": self.switched,
            "fallback": self.fallback,
            "candidate_camera_id": self.candidate_camera_id,
            "candidate_streak": self.candidate_streak,
            "scores_by_camera": {key: float(value) for key, value in self.scores_by_camera.items()},
            "rejected_by_camera": {key: list(value) for key, value in self.rejected_by_camera.items()},
            "reasons": list(self.reasons),
        }


class CameraManager:
    """Select a reliable camera with gates, state, and bounded handover rate."""

    def __init__(self, config: CameraManagerConfig | None = None) -> None:
        self.config = config or CameraManagerConfig()
        self._active_camera_id = ""
        self._contender_camera_id = ""
        self._contender_streak = 0
        self._contender_last_timestamp_s = math.nan

    @property
    def active_camera_id(self) -> str:
        return self._active_camera_id

    def reset(self) -> None:
        """Forget active/contender state before a new replay or run."""

        self._active_camera_id = ""
        self._contender_camera_id = ""
        self._contender_streak = 0
        self._contender_last_timestamp_s = math.nan

    def select(
        self,
        *,
        timestamp_s: float,
        observations: Sequence[MapObservation],
    ) -> CameraManagerDecision:
        """Return one selected observation or an explicit no-observation state.

        A normally healthy active source is held until another source is better
        by ``candidate_score_margin`` for the configured consecutive-frame
        count.  When the active source itself is ineligible, a valid fallback is
        selected immediately (if enabled), because holding a failed source is
        less conservative than changing it.
        """

        timestamp = _finite(timestamp_s, "timestamp_s")
        previous_id = self._active_camera_id
        candidates, scores, rejected = self.eligible_observations(
            timestamp_s=timestamp, observations=observations
        )
        active = candidates.get(previous_id)

        if not previous_id:
            selected = _best(candidates, scores)
            self._active_camera_id = selected.camera_id if selected else ""
            self._clear_contender()
            return self._decision(
                timestamp=timestamp,
                selected=selected,
                previous_id="",
                switched=False,
                fallback=False,
                scores=scores,
                rejected=rejected,
                reasons=("initial_selection",) if selected else ("no_eligible_camera",),
            )

        if active is None:
            selected = _best(candidates, scores)
            self._clear_contender()
            if selected is not None and self.config.fallback_on_active_camera_loss:
                self._active_camera_id = selected.camera_id
                return self._decision(
                    timestamp=timestamp,
                    selected=selected,
                    previous_id=previous_id,
                    switched=selected.camera_id != previous_id,
                    fallback=True,
                    scores=scores,
                    rejected=rejected,
                    reasons=("active_camera_ineligible", "fallback_selection"),
                )
            self._active_camera_id = ""
            return self._decision(
                timestamp=timestamp,
                selected=None,
                previous_id=previous_id,
                switched=False,
                fallback=False,
                scores=scores,
                rejected=rejected,
                reasons=("active_camera_ineligible", "no_eligible_fallback"),
            )

        best = _best(candidates, scores)
        if best is None or best.camera_id == active.camera_id:
            self._clear_contender()
            return self._decision(
                timestamp=timestamp,
                selected=active,
                previous_id=previous_id,
                switched=False,
                fallback=False,
                scores=scores,
                rejected=rejected,
                reasons=("active_camera_best",),
            )

        advantage = scores[best.camera_id] - scores[active.camera_id]
        if advantage < self.config.candidate_score_margin:
            self._clear_contender()
            return self._decision(
                timestamp=timestamp,
                selected=active,
                previous_id=previous_id,
                switched=False,
                fallback=False,
                scores=scores,
                rejected=rejected,
                reasons=("candidate_margin_not_met",),
            )

        consistency_reason = self._consistency_rejection(active, best)
        if consistency_reason:
            rejected = _with_rejection(rejected, best.camera_id, consistency_reason)
            self._clear_contender()
            return self._decision(
                timestamp=timestamp,
                selected=active,
                previous_id=previous_id,
                switched=False,
                fallback=False,
                scores=scores,
                rejected=rejected,
                reasons=(consistency_reason,),
            )

        self._advance_contender(best.camera_id, best.timestamp_s)
        if self._contender_streak < self.config.required_consecutive_better_frames:
            return self._decision(
                timestamp=timestamp,
                selected=active,
                previous_id=previous_id,
                switched=False,
                fallback=False,
                scores=scores,
                rejected=rejected,
                reasons=("candidate_hysteresis_pending",),
            )

        self._active_camera_id = best.camera_id
        self._clear_contender()
        return self._decision(
            timestamp=timestamp,
            selected=best,
            previous_id=previous_id,
            switched=True,
            fallback=False,
            scores=scores,
            rejected=rejected,
            reasons=("hysteresis_handover",),
        )

    def eligible_observations(
        self,
        *,
        timestamp_s: float,
        observations: Sequence[MapObservation],
    ) -> tuple[dict[str, MapObservation], dict[str, float], dict[str, tuple[str, ...]]]:
        """Return the current non-mutating eligibility decision.

        Selection and fusion must start from the same operational contract.  A
        fusion experiment must not accept a stale, low-trust, or poorly
        associated camera merely because it does not use hysteresis.  This
        method deliberately does not change the active-camera state.
        """
        timestamp = _finite(timestamp_s, "timestamp_s")
        return self._eligible_by_camera(timestamp, _camera_batch(observations))

    def _eligible_by_camera(
        self,
        timestamp_s: float,
        observations: Sequence[MapObservation],
    ) -> tuple[dict[str, MapObservation], dict[str, float], dict[str, tuple[str, ...]]]:
        candidates: dict[str, MapObservation] = {}
        scores: dict[str, float] = {}
        rejected: dict[str, tuple[str, ...]] = {}
        for observation in observations:
            camera_id = observation.camera_id
            reasons = self._eligibility_reasons(timestamp_s, observation)
            if reasons:
                rejected = _with_rejection(rejected, camera_id, *reasons)
                continue
            score = operational_camera_score(observation, timestamp_s=timestamp_s, age_decay_s=self.config.age_decay_s)
            candidates[camera_id] = observation
            scores[camera_id] = score
        return candidates, scores, rejected

    def _eligibility_reasons(self, timestamp_s: float, observation: MapObservation) -> tuple[str, ...]:
        reasons: list[str] = []
        camera_id = observation.camera_id
        if self.config.allowed_camera_ids and camera_id not in self.config.allowed_camera_ids:
            reasons.append("camera_not_allowed")
        age = observation_age_s(observation, timestamp_s=timestamp_s)
        if float(observation.timestamp_s) > timestamp_s:
            reasons.append("measurement_future_dated")
        elif observation.quality.stale or age > self.config.max_measurement_age_s:
            reasons.append("measurement_stale")
        if observation.quality.association_confidence < self.config.min_association_confidence:
            reasons.append("association_below_minimum")
        score = operational_camera_score(observation, timestamp_s=timestamp_s, age_decay_s=self.config.age_decay_s)
        if score < self.config.min_spatial_trust:
            reasons.append("spatial_trust_below_minimum")
        return tuple(reasons)

    def _consistency_rejection(self, active: MapObservation, candidate: MapObservation) -> str:
        if not self.config.require_consistency_when_source_available:
            return ""
        if abs(candidate.timestamp_s - active.timestamp_s) > self.config.max_overlap_time_delta_s:
            return "overlap_not_synchronous"
        disagreement = math.hypot(candidate.xy_m[0] - active.xy_m[0], candidate.xy_m[1] - active.xy_m[1])
        if disagreement > self.config.max_cross_camera_disagreement_m:
            return "cross_camera_disagreement"
        return ""

    def _advance_contender(self, camera_id: str, observation_timestamp_s: float) -> None:
        """Advance hysteresis only for a newer frame from the same contender.

        The live node calls :meth:`select` from a fixed-rate timer and retains
        the latest observation between detector callbacks.  Counting every
        timer call would therefore turn one slow camera frame into several
        fictitious "consecutive better frames".  Replay already supplies one
        call per frame, so keying the streak to the observation timestamp keeps
        live and replay semantics aligned.
        """

        stamp = _finite(observation_timestamp_s, "observation_timestamp_s")
        if self._contender_camera_id == camera_id:
            if stamp <= self._contender_last_timestamp_s:
                return
            self._contender_streak += 1
            self._contender_last_timestamp_s = stamp
        else:
            self._contender_camera_id = camera_id
            self._contender_streak = 1
            self._contender_last_timestamp_s = stamp

    def _clear_contender(self) -> None:
        self._contender_camera_id = ""
        self._contender_streak = 0
        self._contender_last_timestamp_s = math.nan

    def _decision(
        self,
        *,
        timestamp: float,
        selected: MapObservation | None,
        previous_id: str,
        switched: bool,
        fallback: bool,
        scores: Mapping[str, float],
        rejected: Mapping[str, tuple[str, ...]],
        reasons: tuple[str, ...],
    ) -> CameraManagerDecision:
        return CameraManagerDecision(
            timestamp_s=timestamp,
            selected_observation=selected,
            active_camera_id=self._active_camera_id,
            previous_camera_id=previous_id,
            switched=switched,
            fallback=fallback,
            candidate_camera_id=self._contender_camera_id,
            candidate_streak=self._contender_streak,
            scores_by_camera=dict(scores),
            rejected_by_camera=dict(rejected),
            reasons=reasons,
        )


def observation_age_s(observation: MapObservation, *, timestamp_s: float) -> float:
    """Return non-negative age in seconds without treating future stamps as fresh."""

    now = _finite(timestamp_s, "timestamp_s")
    age = now - float(observation.timestamp_s)
    return math.inf if age < 0.0 else age


def operational_camera_score(
    observation: MapObservation,
    *,
    timestamp_s: float,
    age_decay_s: float = 0.15,
) -> float:
    """Conservative operational trust used only for selection/handover ranking."""

    decay = _positive(age_decay_s, "age_decay_s")
    quality = observation.quality
    age_factor = math.exp(-observation_age_s(observation, timestamp_s=timestamp_s) / decay)
    score = (
        float(quality.p_available)
        * float(quality.association_confidence)
        * math.exp(-float(quality.epistemic_score))
        * age_factor
    )
    return float(min(max(score, 0.0), 1.0))


def _best(candidates: Mapping[str, MapObservation], scores: Mapping[str, float]) -> MapObservation | None:
    if not candidates:
        return None
    return min(
        candidates.values(),
        key=lambda observation: (
            -scores[observation.camera_id],
            -observation.timestamp_s,
            observation.camera_id,
        ),
    )


def _camera_batch(observations: Sequence[MapObservation]) -> tuple[MapObservation, ...]:
    """Require one current observation per physical camera."""
    items = tuple(observations)
    seen: set[str] = set()
    for observation in items:
        if not isinstance(observation, MapObservation):
            raise ContractValidationError("camera manager entries must be MapObservation objects")
        if observation.camera_id in seen:
            raise ContractValidationError(
                f"duplicate camera in manager batch: {observation.camera_id!r}"
            )
        seen.add(observation.camera_id)
    return items


def _with_rejection(
    rejected: Mapping[str, tuple[str, ...]],
    camera_id: str,
    *reasons: str,
) -> dict[str, tuple[str, ...]]:
    out = dict(rejected)
    current = list(out.get(camera_id, ()))
    current.extend(reason for reason in reasons if reason and reason not in current)
    out[camera_id] = tuple(current)
    return out


def _finite(value: float, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"{name} must be numeric") from exc
    if not math.isfinite(out):
        raise ContractValidationError(f"{name} must be finite")
    return out


def _nonnegative(value: float, name: str) -> float:
    out = _finite(value, name)
    if out < 0.0:
        raise ContractValidationError(f"{name} must be non-negative")
    return out


def _positive(value: float, name: str) -> float:
    out = _finite(value, name)
    if out <= 0.0:
        raise ContractValidationError(f"{name} must be positive")
    return out


def _probability(value: float, name: str) -> float:
    out = _nonnegative(value, name)
    if out > 1.0:
        raise ContractValidationError(f"{name} must be in [0, 1]")
    return out
