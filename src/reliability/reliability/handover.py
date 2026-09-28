"""Handover uncertainty diagnostics for calibrated multi-camera fusion.

The functions in this module use only operational inputs: camera IDs, timestamps,
map-space observations, per-camera quality, staleness, and A/B disagreement. They
do not decide absolute truth; they quantify how much extra uncertainty should be
carried while switching from one camera source to another.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

from reliability.contracts import CameraQuality, ContractValidationError
from reliability.fusion import MapObservation


@dataclass(frozen=True)
class HandoverUncertaintyConfig:
    """Weights and gates for source-switch covariance inflation."""

    disagreement_gate_m: float = 0.30
    freshness_gate_s: float = 0.10
    max_overlap_dt_s: float = 0.05
    max_covariance_inflation: float = 9.0
    switch_penalty: float = 0.10
    no_overlap_penalty: float = 0.35
    disagreement_weight: float = 0.35
    freshness_weight: float = 0.15
    quality_drop_weight: float = 0.20
    low_quality_weight: float = 0.15
    stale_penalty: float = 0.15

    def __post_init__(self) -> None:
        object.__setattr__(self, "disagreement_gate_m", _positive(self.disagreement_gate_m, "disagreement_gate_m"))
        object.__setattr__(self, "freshness_gate_s", _positive(self.freshness_gate_s, "freshness_gate_s"))
        object.__setattr__(self, "max_overlap_dt_s", _nonnegative(self.max_overlap_dt_s, "max_overlap_dt_s"))
        max_inflation = _positive(self.max_covariance_inflation, "max_covariance_inflation")
        if max_inflation < 1.0:
            raise ContractValidationError("max_covariance_inflation must be >= 1")
        object.__setattr__(self, "max_covariance_inflation", max_inflation)
        for name in (
            "switch_penalty",
            "no_overlap_penalty",
            "disagreement_weight",
            "freshness_weight",
            "quality_drop_weight",
            "low_quality_weight",
            "stale_penalty",
        ):
            object.__setattr__(self, name, _unit_interval(getattr(self, name), name))


@dataclass(frozen=True)
class HandoverUncertaintyDiagnostic:
    previous_camera_id: str
    selected_camera_id: str
    switched: bool
    overlap_available: bool
    disagreement_m: float
    age_gap_s: float
    previous_quality_score: float
    selected_quality_score: float
    quality_drop: float
    uncertainty_score: float
    covariance_inflation: float
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "previous_camera_id": self.previous_camera_id,
            "selected_camera_id": self.selected_camera_id,
            "switched": self.switched,
            "overlap_available": self.overlap_available,
            "disagreement_m": _none_nan(self.disagreement_m),
            "age_gap_s": _none_nan(self.age_gap_s),
            "previous_quality_score": _none_nan(self.previous_quality_score),
            "selected_quality_score": _none_nan(self.selected_quality_score),
            "quality_drop": _none_nan(self.quality_drop),
            "uncertainty_score": self.uncertainty_score,
            "covariance_inflation": self.covariance_inflation,
            "reasons": list(self.reasons),
        }


def assess_handover_uncertainty(
    *,
    previous_camera_id: str | None,
    selected_observation: MapObservation | None,
    candidate_observations: Sequence[MapObservation] = (),
    previous_observation: MapObservation | None = None,
    config: HandoverUncertaintyConfig | None = None,
) -> HandoverUncertaintyDiagnostic:
    """Score source-switch uncertainty from operational camera evidence.

    `candidate_observations` should contain the near-synchronous observations
    available at the current replay/runtime step. When switching from A to B,
    an overlapping A observation near B's timestamp is the strongest evidence
    that the handover is well calibrated.
    """

    cfg = config or HandoverUncertaintyConfig()
    previous_id = str(previous_camera_id or (previous_observation.camera_id if previous_observation else ""))
    if selected_observation is None:
        return HandoverUncertaintyDiagnostic(
            previous_camera_id=previous_id,
            selected_camera_id="",
            switched=bool(previous_id),
            overlap_available=False,
            disagreement_m=math.nan,
            age_gap_s=math.nan,
            previous_quality_score=_quality_score(previous_observation.quality) if previous_observation else math.nan,
            selected_quality_score=0.0,
            quality_drop=1.0,
            uncertainty_score=1.0,
            covariance_inflation=cfg.max_covariance_inflation,
            reasons=("no_selected_camera",),
        )

    selected_id = selected_observation.camera_id
    switched = bool(previous_id and previous_id != selected_id)
    candidates = _with_selected(candidate_observations, selected_observation)
    previous_current = _nearest_camera_observation(
        candidates,
        camera_id=previous_id,
        timestamp_s=selected_observation.timestamp_s,
        max_dt_s=cfg.max_overlap_dt_s,
    )
    overlap_available = switched and previous_current is not None

    if previous_current is not None:
        disagreement = _distance_m(previous_current, selected_observation)
        age_gap = abs(float(previous_current.timestamp_s) - float(selected_observation.timestamp_s))
        previous_quality = _quality_score(previous_current.quality)
    else:
        disagreement = math.nan
        age_gap = (
            max(0.0, float(selected_observation.timestamp_s) - float(previous_observation.timestamp_s))
            if previous_observation is not None
            else 0.0
        )
        previous_quality = _quality_score(previous_observation.quality) if previous_observation else math.nan

    selected_quality = _quality_score(selected_observation.quality)
    quality_drop = max(0.0, (previous_quality if math.isfinite(previous_quality) else selected_quality) - selected_quality)

    reasons: list[str] = []
    uncertainty = 0.0
    if switched:
        uncertainty += cfg.switch_penalty
        reasons.append("camera_switch")
        if overlap_available:
            disagreement_ratio = min(max(disagreement, 0.0) / cfg.disagreement_gate_m, 1.0)
            uncertainty += cfg.disagreement_weight * disagreement_ratio
            if disagreement_ratio >= 1.0:
                reasons.append("overlap_disagreement")
            elif disagreement_ratio > 0.0:
                reasons.append("overlap_residual")
            age_ratio = min(max(age_gap, 0.0) / cfg.freshness_gate_s, 1.0)
            uncertainty += cfg.freshness_weight * age_ratio
            if age_ratio >= 1.0:
                reasons.append("async_handover")
        else:
            uncertainty += cfg.no_overlap_penalty
            reasons.append("no_overlap_confirmation")

    if quality_drop > 0.0:
        uncertainty += cfg.quality_drop_weight * min(quality_drop, 1.0)
        if quality_drop >= 0.20:
            reasons.append("quality_drop")

    low_quality = max(0.0, 1.0 - selected_quality)
    if low_quality > 0.0:
        uncertainty += cfg.low_quality_weight * low_quality
        if selected_quality < 0.50:
            reasons.append("low_selected_quality")

    if selected_observation.quality.stale:
        uncertainty += cfg.stale_penalty
        reasons.append("selected_stale")

    score = min(max(float(uncertainty), 0.0), 1.0)
    inflation = 1.0 + score * (cfg.max_covariance_inflation - 1.0)
    return HandoverUncertaintyDiagnostic(
        previous_camera_id=previous_id,
        selected_camera_id=selected_id,
        switched=switched,
        overlap_available=overlap_available,
        disagreement_m=float(disagreement),
        age_gap_s=float(age_gap),
        previous_quality_score=float(previous_quality),
        selected_quality_score=float(selected_quality),
        quality_drop=float(quality_drop),
        uncertainty_score=score,
        covariance_inflation=float(inflation),
        reasons=tuple(dict.fromkeys(reasons)),
    )


def inflate_map_observation_covariance(
    observation: MapObservation,
    inflation: float,
    *,
    source: str = "handover_uncertainty",
) -> MapObservation:
    """Return `observation` with its 2-D covariance multiplied by `inflation`."""

    factor = _positive(inflation, "inflation")
    return MapObservation(
        camera_id=observation.camera_id,
        timestamp_s=observation.timestamp_s,
        xy_m=observation.xy_m,
        covariance_m2=(
            (observation.covariance_m2[0][0] * factor, observation.covariance_m2[0][1] * factor),
            (observation.covariance_m2[1][0] * factor, observation.covariance_m2[1][1] * factor),
        ),
        quality=observation.quality,
        source=source,
    )


def handover_adjusted_observation(
    *,
    previous_camera_id: str | None,
    selected_observation: MapObservation | None,
    candidate_observations: Sequence[MapObservation] = (),
    previous_observation: MapObservation | None = None,
    config: HandoverUncertaintyConfig | None = None,
) -> tuple[MapObservation | None, HandoverUncertaintyDiagnostic]:
    """Assess handover uncertainty and inflate the selected observation."""

    diagnostic = assess_handover_uncertainty(
        previous_camera_id=previous_camera_id,
        selected_observation=selected_observation,
        candidate_observations=candidate_observations,
        previous_observation=previous_observation,
        config=config,
    )
    if selected_observation is None:
        return None, diagnostic
    return (
        inflate_map_observation_covariance(
            selected_observation,
            diagnostic.covariance_inflation,
            source=f"{selected_observation.source or 'map_observation'}:handover_inflated",
        ),
        diagnostic,
    )


def _quality_score(quality: CameraQuality | None) -> float:
    if quality is None:
        return math.nan
    score = (
        float(quality.p_available)
        * float(quality.association_confidence)
        * math.exp(-max(float(quality.epistemic_score), 0.0))
    )
    if quality.stale:
        score *= 0.5
    return float(min(max(score, 0.0), 1.0))


def _nearest_camera_observation(
    observations: Sequence[MapObservation],
    *,
    camera_id: str,
    timestamp_s: float,
    max_dt_s: float,
) -> MapObservation | None:
    if not camera_id:
        return None
    best: MapObservation | None = None
    best_dt = math.inf
    for obs in observations:
        if obs.camera_id != camera_id:
            continue
        dt = abs(float(obs.timestamp_s) - float(timestamp_s))
        if dt <= max_dt_s and dt < best_dt:
            best = obs
            best_dt = dt
    return best


def _with_selected(observations: Sequence[MapObservation], selected: MapObservation) -> tuple[MapObservation, ...]:
    out = list(observations)
    key = (selected.camera_id, round(float(selected.timestamp_s), 9), selected.xy_m)
    if all((obs.camera_id, round(float(obs.timestamp_s), 9), obs.xy_m) != key for obs in out):
        out.append(selected)
    return tuple(out)


def _distance_m(a: MapObservation, b: MapObservation) -> float:
    return float(math.hypot(a.xy_m[0] - b.xy_m[0], a.xy_m[1] - b.xy_m[1]))


def _positive(value: float, field_name: str) -> float:
    out = _finite(value, field_name)
    if out <= 0.0:
        raise ContractValidationError(f"{field_name} must be positive")
    return out


def _nonnegative(value: float, field_name: str) -> float:
    out = _finite(value, field_name)
    if out < 0.0:
        raise ContractValidationError(f"{field_name} must be non-negative")
    return out


def _unit_interval(value: float, field_name: str) -> float:
    out = _nonnegative(value, field_name)
    if out > 1.0:
        raise ContractValidationError(f"{field_name} must be in [0, 1]")
    return out


def _finite(value: float, field_name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"{field_name} must be numeric") from exc
    if not math.isfinite(out):
        raise ContractValidationError(f"{field_name} must be finite")
    return out


def _none_nan(value: float) -> float | None:
    return None if isinstance(value, float) and math.isnan(value) else float(value)
