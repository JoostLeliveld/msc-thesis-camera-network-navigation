"""Camera-overlap validation utilities for extension calibration gates."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Sequence

from reliability.contracts import ContractValidationError
from reliability.fusion import MapObservation
from reliability.replay import ReplayFrame


@dataclass(frozen=True)
class CameraOverlapPair:
    camera_a_id: str
    camera_b_id: str
    timestamp_a_s: float
    timestamp_b_s: float
    xy_a_m: tuple[float, float]
    xy_b_m: tuple[float, float]
    disagreement_m: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_a_id": self.camera_a_id,
            "camera_b_id": self.camera_b_id,
            "timestamp_a_s": self.timestamp_a_s,
            "timestamp_b_s": self.timestamp_b_s,
            "xy_a_m": list(self.xy_a_m),
            "xy_b_m": list(self.xy_b_m),
            "disagreement_m": self.disagreement_m,
        }


@dataclass(frozen=True)
class CameraOverlapValidationSummary:
    camera_a_id: str
    camera_b_id: str
    pair_count: int
    mean_disagreement_m: float
    max_disagreement_m: float
    rms_disagreement_m: float
    max_allowed_disagreement_m: float
    pass_gate: bool
    pairs: tuple[CameraOverlapPair, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_a_id": self.camera_a_id,
            "camera_b_id": self.camera_b_id,
            "pair_count": self.pair_count,
            "mean_disagreement_m": _none_nan(self.mean_disagreement_m),
            "max_disagreement_m": _none_nan(self.max_disagreement_m),
            "rms_disagreement_m": _none_nan(self.rms_disagreement_m),
            "max_allowed_disagreement_m": self.max_allowed_disagreement_m,
            "pass_gate": self.pass_gate,
            "pairs": [pair.to_dict() for pair in self.pairs],
        }


@dataclass(frozen=True)
class CameraOverlapTrustSummary:
    camera_a_id: str
    camera_b_id: str
    pair_count: int
    disagreement_gate_m: float
    mean_disagreement_m: float
    median_disagreement_m: float
    p90_disagreement_m: float
    max_disagreement_m: float
    outlier_rate: float
    mean_delta_b_minus_a_m: tuple[float, float]
    bias_norm_m: float
    pair_trust_score: float
    pass_gate: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_a_id": self.camera_a_id,
            "camera_b_id": self.camera_b_id,
            "pair_count": self.pair_count,
            "disagreement_gate_m": self.disagreement_gate_m,
            "mean_disagreement_m": _none_nan(self.mean_disagreement_m),
            "median_disagreement_m": _none_nan(self.median_disagreement_m),
            "p90_disagreement_m": _none_nan(self.p90_disagreement_m),
            "max_disagreement_m": _none_nan(self.max_disagreement_m),
            "outlier_rate": _none_nan(self.outlier_rate),
            "mean_delta_b_minus_a_m": list(self.mean_delta_b_minus_a_m),
            "bias_norm_m": _none_nan(self.bias_norm_m),
            "pair_trust_score": _none_nan(self.pair_trust_score),
            "pass_gate": self.pass_gate,
        }


def overlap_pairs_from_frames(
    frames: Sequence[ReplayFrame],
    *,
    camera_a_id: str = "camera_A",
    camera_b_id: str = "camera_B",
    max_time_delta_s: float = 0.05,
) -> tuple[CameraOverlapPair, ...]:
    """Build A/B overlap pairs from replay frames or near-synchronous frames."""

    observations: list[MapObservation] = []
    for frame in frames:
        observations.extend(frame.observations)
    return match_overlap_observations(
        observations,
        camera_a_id=camera_a_id,
        camera_b_id=camera_b_id,
        max_time_delta_s=max_time_delta_s,
    )


def match_overlap_observations(
    observations: Iterable[MapObservation],
    *,
    camera_a_id: str = "camera_A",
    camera_b_id: str = "camera_B",
    max_time_delta_s: float = 0.05,
) -> tuple[CameraOverlapPair, ...]:
    """Greedily match near-synchronous map observations from two cameras."""

    max_dt = _nonnegative(max_time_delta_s, "max_time_delta_s")
    obs_a = sorted((obs for obs in observations if obs.camera_id == camera_a_id), key=lambda obs: obs.timestamp_s)
    obs_b = sorted((obs for obs in observations if obs.camera_id == camera_b_id), key=lambda obs: obs.timestamp_s)
    used_b: set[int] = set()
    pairs: list[CameraOverlapPair] = []
    for a in obs_a:
        best_idx = None
        best_dt = math.inf
        for idx, b in enumerate(obs_b):
            if idx in used_b:
                continue
            dt = abs(float(a.timestamp_s) - float(b.timestamp_s))
            if dt <= max_dt and dt < best_dt:
                best_idx = idx
                best_dt = dt
        if best_idx is None:
            continue
        b = obs_b[best_idx]
        used_b.add(best_idx)
        disagreement = math.hypot(a.xy_m[0] - b.xy_m[0], a.xy_m[1] - b.xy_m[1])
        pairs.append(
            CameraOverlapPair(
                camera_a_id=camera_a_id,
                camera_b_id=camera_b_id,
                timestamp_a_s=float(a.timestamp_s),
                timestamp_b_s=float(b.timestamp_s),
                xy_a_m=a.xy_m,
                xy_b_m=b.xy_m,
                disagreement_m=float(disagreement),
            )
        )
    return tuple(pairs)


def validate_camera_overlap(
    frames: Sequence[ReplayFrame],
    *,
    camera_a_id: str = "camera_A",
    camera_b_id: str = "camera_B",
    max_time_delta_s: float = 0.05,
    max_allowed_disagreement_m: float = 0.30,
) -> CameraOverlapValidationSummary:
    pairs = overlap_pairs_from_frames(
        frames,
        camera_a_id=camera_a_id,
        camera_b_id=camera_b_id,
        max_time_delta_s=max_time_delta_s,
    )
    return summarize_overlap_pairs(
        pairs,
        camera_a_id=camera_a_id,
        camera_b_id=camera_b_id,
        max_allowed_disagreement_m=max_allowed_disagreement_m,
    )


def summarize_overlap_trust(
    pairs: Sequence[CameraOverlapPair],
    *,
    camera_a_id: str = "camera_A",
    camera_b_id: str = "camera_B",
    disagreement_gate_m: float = 0.30,
    min_pair_count: int = 1,
    max_outlier_rate: float = 0.05,
) -> CameraOverlapTrustSummary:
    """Summarize overlap pairs as a calibration/trust diagnostic.

    Without an external reference this cannot assign absolute blame to either
    camera. It estimates the pair's consistency, the systematic B-minus-A bias,
    and a heuristic pair-trust score for fusion/selection ablations.
    """

    gate = _nonnegative(disagreement_gate_m, "disagreement_gate_m")
    if int(min_pair_count) < 0:
        raise ContractValidationError("min_pair_count must be non-negative")
    max_outlier_rate = _probability(max_outlier_rate, "max_outlier_rate")
    selected = [
        pair
        for pair in pairs
        if pair.camera_a_id == camera_a_id and pair.camera_b_id == camera_b_id
        and math.isfinite(pair.disagreement_m)
    ]
    values = [pair.disagreement_m for pair in selected]
    deltas = [
        (pair.xy_b_m[0] - pair.xy_a_m[0], pair.xy_b_m[1] - pair.xy_a_m[1])
        for pair in selected
    ]
    if not values:
        return CameraOverlapTrustSummary(
            camera_a_id=camera_a_id,
            camera_b_id=camera_b_id,
            pair_count=0,
            disagreement_gate_m=gate,
            mean_disagreement_m=math.nan,
            median_disagreement_m=math.nan,
            p90_disagreement_m=math.nan,
            max_disagreement_m=math.nan,
            outlier_rate=math.nan,
            mean_delta_b_minus_a_m=(math.nan, math.nan),
            bias_norm_m=math.nan,
            pair_trust_score=0.0,
            pass_gate=False,
        )
    mean = sum(values) / len(values)
    median = _quantile(values, 0.50)
    p90 = _quantile(values, 0.90)
    max_value = max(values)
    outlier_rate = sum(1 for value in values if value > gate) / len(values)
    dx = sum(delta[0] for delta in deltas) / len(deltas)
    dy = sum(delta[1] for delta in deltas) / len(deltas)
    bias_norm = math.hypot(dx, dy)
    scale = max(gate, 1.0e-6)
    pair_trust = math.exp(-p90 / scale) * max(1.0 - outlier_rate, 0.0)
    pass_gate = bool(
        len(values) >= int(min_pair_count)
        and outlier_rate <= max_outlier_rate
        and p90 <= gate
    )
    return CameraOverlapTrustSummary(
        camera_a_id=camera_a_id,
        camera_b_id=camera_b_id,
        pair_count=len(values),
        disagreement_gate_m=gate,
        mean_disagreement_m=float(mean),
        median_disagreement_m=float(median),
        p90_disagreement_m=float(p90),
        max_disagreement_m=float(max_value),
        outlier_rate=float(outlier_rate),
        mean_delta_b_minus_a_m=(float(dx), float(dy)),
        bias_norm_m=float(bias_norm),
        pair_trust_score=float(pair_trust),
        pass_gate=pass_gate,
    )


def overlap_trust_from_frames(
    frames: Sequence[ReplayFrame],
    *,
    camera_a_id: str = "camera_A",
    camera_b_id: str = "camera_B",
    max_time_delta_s: float = 0.05,
    disagreement_gate_m: float = 0.30,
    min_pair_count: int = 1,
    max_outlier_rate: float = 0.05,
) -> CameraOverlapTrustSummary:
    pairs = overlap_pairs_from_frames(
        frames,
        camera_a_id=camera_a_id,
        camera_b_id=camera_b_id,
        max_time_delta_s=max_time_delta_s,
    )
    return summarize_overlap_trust(
        pairs,
        camera_a_id=camera_a_id,
        camera_b_id=camera_b_id,
        disagreement_gate_m=disagreement_gate_m,
        min_pair_count=min_pair_count,
        max_outlier_rate=max_outlier_rate,
    )


def summarize_overlap_pairs(
    pairs: Sequence[CameraOverlapPair],
    *,
    camera_a_id: str = "camera_A",
    camera_b_id: str = "camera_B",
    max_allowed_disagreement_m: float = 0.30,
) -> CameraOverlapValidationSummary:
    limit = _nonnegative(max_allowed_disagreement_m, "max_allowed_disagreement_m")
    values = [pair.disagreement_m for pair in pairs if math.isfinite(pair.disagreement_m)]
    mean = sum(values) / len(values) if values else math.nan
    max_value = max(values) if values else math.nan
    rms = math.sqrt(sum(value * value for value in values) / len(values)) if values else math.nan
    pass_gate = bool(values) and max_value <= limit
    return CameraOverlapValidationSummary(
        camera_a_id=camera_a_id,
        camera_b_id=camera_b_id,
        pair_count=len(values),
        mean_disagreement_m=mean,
        max_disagreement_m=max_value,
        rms_disagreement_m=rms,
        max_allowed_disagreement_m=limit,
        pass_gate=pass_gate,
        pairs=tuple(pairs),
    )


def _nonnegative(value: float, field_name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"{field_name} must be numeric") from exc
    if not math.isfinite(out) or out < 0.0:
        raise ContractValidationError(f"{field_name} must be finite and non-negative")
    return out


def _probability(value: float, field_name: str) -> float:
    out = _nonnegative(value, field_name)
    if out > 1.0:
        raise ContractValidationError(f"{field_name} must be in [0, 1]")
    return out


def _quantile(values: Sequence[float], q: float) -> float:
    vals = sorted(float(value) for value in values)
    if not vals:
        return math.nan
    q = min(max(float(q), 0.0), 1.0)
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _none_nan(value: float) -> float | None:
    return None if isinstance(value, float) and math.isnan(value) else float(value)
