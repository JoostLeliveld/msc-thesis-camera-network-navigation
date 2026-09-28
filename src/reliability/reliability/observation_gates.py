"""Versioned deterministic observation gates.

``evaluate_observation_opportunity(raw_record, gate_config)`` turns one normalized raw
opportunity (produced by the offline exporter from operational logs — never GT) into an
:class:`~reliability.observation_opportunity.ObservationOpportunity`, assigning
``detection_label`` / ``quality_label`` / ``usable_label`` and a single controlled
:class:`~reliability.observation_opportunity.FailureReason` (the earliest failing gate).

Every check is independently toggleable through :class:`UsableObservationGateConfig` so
that (a) datasets whose logs lack a signal can disable the corresponding check explicitly
and reproducibly, and (b) the A5/A6 ablation (quality gate without vs with confidence) is a
config flip, not a code change.  The chosen config is hashable (``config_hash``) and is
recorded in every dataset manifest.

If an *enabled* check lacks its required input on a record, the record is labelled
``UNKNOWN`` — it is never silently coerced into a negative.

The commissioning sensor gate is a stricter use of this module.  It may inspect only
frame/detector fields, raw-box geometry, and whether the calibrated projection exists.
It must not inspect association, tracking, a localizer decision, an innovation, or the
current belief.  :func:`evaluate_sensor_gate` enforces that separation explicitly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping

from reliability.contracts import (
    ContractValidationError,
    LeakageError,
    reject_evaluation_only_keys,
)
from reliability.observation_opportunity import (
    FailureReason,
    ObservationOpportunity,
    SourceDomain,
)


@dataclass(frozen=True)
class UsableObservationGateConfig:
    """Pre-registered, frozen gate thresholds and enable flags."""

    image_width_px: int
    image_height_px: int

    # detection availability (D_c)
    check_frame_freshness: bool = True
    max_frame_age_ms: float = 500.0

    # quality checks (U_c | D_c)
    check_confidence: bool = True
    confidence_threshold: float = 0.25
    check_class: bool = False
    expected_class: str | None = None
    check_box_size: bool = False
    min_bbox_width_px: float = 0.0
    min_bbox_height_px: float = 0.0
    check_projection: bool = True
    check_association: bool = True
    require_track: bool = False
    check_edge_clip: bool = True
    edge_check_mode: str = "selected_pixel"
    min_edge_distance_px: float = 4.0
    check_localizer_accept: bool = True

    gate_id: str = "usable_observation_gate_v1"

    def __post_init__(self) -> None:
        if self.image_width_px <= 0 or self.image_height_px <= 0:
            raise ContractValidationError("image dimensions must be positive")
        for name in (
            "max_frame_age_ms",
            "confidence_threshold",
            "min_bbox_width_px",
            "min_bbox_height_px",
            "min_edge_distance_px",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ContractValidationError(f"{name} must be finite and non-negative")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ContractValidationError("confidence_threshold must be in [0, 1]")
        if self.check_class and not self.expected_class:
            raise ContractValidationError("check_class requires expected_class")
        if self.edge_check_mode not in {"selected_pixel", "full_bbox"}:
            raise ContractValidationError(
                "edge_check_mode must be 'selected_pixel' or 'full_bbox'"
            )

    def assert_belief_independent(self) -> None:
        """Reject checks that belong to data association or estimator gating."""

        forbidden = []
        if self.check_association:
            forbidden.append("check_association")
        if self.require_track:
            forbidden.append("require_track")
        if self.check_localizer_accept:
            forbidden.append("check_localizer_accept")
        if forbidden:
            raise ContractValidationError(
                "commissioning sensor gate must be belief-independent; disable "
                + ", ".join(forbidden)
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UsableObservationGateConfig":
        if not isinstance(payload, Mapping):
            raise ContractValidationError("gate config payload must be a mapping")
        known = {f_.name for f_ in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = set(payload) - known
        if unknown:
            raise ContractValidationError("gate config unknown fields: " + ", ".join(sorted(unknown)))
        return cls(**dict(payload))

    @classmethod
    def from_yaml(cls, path: str) -> "UsableObservationGateConfig":
        import yaml  # local import keeps the contract importable without pyyaml

        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        if not isinstance(payload, Mapping):
            raise ContractValidationError(f"{path} must contain a mapping")
        return cls.from_dict(payload)


class _MissingInput(Exception):
    """Raised internally when an enabled check has no input to evaluate."""


def _get_bool(raw: Mapping[str, Any], key: str, default: bool | None = None) -> bool:
    if key not in raw or raw[key] is None:
        if default is None:
            raise _MissingInput(key)
        return default
    value = raw[key]
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return bool(value)


def _get_float(raw: Mapping[str, Any], key: str) -> float | None:
    if key not in raw or raw[key] is None:
        return None
    try:
        out = float(raw[key])
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _edge_distance_px(u: float | None, v: float | None, width: int, height: int) -> float | None:
    if u is None or v is None:
        return None
    return float(min(u, v, width - u, height - v))


def _bbox_edge_distance_px(
    xmin: float | None,
    ymin: float | None,
    xmax: float | None,
    ymax: float | None,
    width: int,
    height: int,
) -> float | None:
    """Smallest padding between the complete box and any image boundary."""

    if None in (xmin, ymin, xmax, ymax):
        return None
    return float(min(xmin, ymin, width - xmax, height - ymax))


def _classify(
    raw: Mapping[str, Any],
    cfg: UsableObservationGateConfig,
    *,
    selected_u: float | None,
    selected_v: float | None,
    edge_distance_px: float | None,
    bbox_width: float | None,
    bbox_height: float | None,
) -> tuple[int, int, FailureReason]:
    """Return (detection_label, quality_label, failure_reason) via ordered gates."""

    # --- detection availability -------------------------------------------------------
    frame_expected = _get_bool(raw, "frame_expected", default=True)
    frame_received = _get_bool(raw, "frame_received", default=True)
    if frame_expected and not frame_received:
        return 0, 0, FailureReason.NO_FRAME

    if cfg.check_frame_freshness:
        frame_age = _get_float(raw, "frame_age_ms")
        if frame_age is None:
            return 0, 0, FailureReason.UNKNOWN
        if frame_age > cfg.max_frame_age_ms:
            return 0, 0, FailureReason.STALE_FRAME

    detection_received = _get_bool(raw, "detection_received", default=False)
    if not detection_received:
        return 0, 0, FailureReason.NO_DETECTION

    # detection exists -> detection_label = 1; now evaluate quality gates in order.
    try:
        if cfg.check_confidence:
            confidence = _get_float(raw, "detector_confidence")
            if confidence is None:
                return 1, 0, FailureReason.UNKNOWN
            if confidence < cfg.confidence_threshold:
                return 1, 0, FailureReason.LOW_CONFIDENCE

        if cfg.check_class:
            detector_class = raw.get("detector_class")
            if detector_class is None:
                return 1, 0, FailureReason.UNKNOWN
            if str(detector_class) != str(cfg.expected_class):
                return 1, 0, FailureReason.WRONG_CLASS_OR_ID

        if cfg.check_box_size:
            if bbox_width is None or bbox_height is None:
                return 1, 0, FailureReason.UNKNOWN
            if (
                bbox_width < cfg.min_bbox_width_px
                or bbox_height < cfg.min_bbox_height_px
            ):
                return 1, 0, FailureReason.BOX_TOO_SMALL

        if cfg.check_projection:
            if not _get_bool(raw, "projection_valid"):
                return 1, 0, FailureReason.INVALID_PROJECTION

        if cfg.check_association:
            if not _get_bool(raw, "association_valid"):
                return 1, 0, FailureReason.INVALID_ASSOCIATION

        if cfg.require_track:
            if not _get_bool(raw, "tracking_valid"):
                return 1, 0, FailureReason.INVALID_TRACK

        if cfg.check_edge_clip:
            if edge_distance_px is None:
                return 1, 0, FailureReason.UNKNOWN
            if edge_distance_px < cfg.min_edge_distance_px:
                return 1, 0, FailureReason.CLIPPED_OR_EDGE

        if cfg.check_localizer_accept:
            if not _get_bool(raw, "accepted_by_localizer"):
                return 1, 0, FailureReason.REJECTED_BY_LOCALIZER
    except _MissingInput:
        return 1, 0, FailureReason.UNKNOWN

    return 1, 1, FailureReason.USABLE


@dataclass(frozen=True)
class SensorGateResult:
    """Outcome of the belief-independent gate used to define ``q_sensor``."""

    admitted: bool
    reason: str
    gate_id: str
    gate_config_hash: str


def evaluate_sensor_gate(
    raw_record: Mapping[str, Any],
    gate_config: UsableObservationGateConfig,
) -> SensorGateResult:
    """Evaluate the deterministic pre-estimator gate used during commissioning.

    Ground truth and belief-derived checks are rejected.  NIS is intentionally absent:
    innovation gating occurs later, after correction and the matched covariance query.
    """

    if not isinstance(raw_record, Mapping):
        raise ContractValidationError("raw_record must be a mapping")
    reject_evaluation_only_keys(raw_record, context="evaluate_sensor_gate")
    gate_config.assert_belief_independent()

    bbox_xmin = _get_float(raw_record, "bbox_xmin")
    bbox_ymin = _get_float(raw_record, "bbox_ymin")
    bbox_xmax = _get_float(raw_record, "bbox_xmax")
    bbox_ymax = _get_float(raw_record, "bbox_ymax")
    bbox_width = bbox_height = bbox_center_u = bbox_ymax_for_pixel = None
    if None not in (bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax):
        bbox_width = bbox_xmax - bbox_xmin
        bbox_height = bbox_ymax - bbox_ymin
        bbox_center_u = 0.5 * (bbox_xmin + bbox_xmax)
        bbox_ymax_for_pixel = bbox_ymax

    selected_u = _get_float(raw_record, "selected_pixel_u")
    selected_v = _get_float(raw_record, "selected_pixel_v")
    if selected_u is None and bbox_center_u is not None:
        selected_u = bbox_center_u
        selected_v = bbox_ymax_for_pixel
    edge_distance_px = None
    if gate_config.edge_check_mode == "full_bbox":
        edge_distance_px = _bbox_edge_distance_px(
            bbox_xmin,
            bbox_ymin,
            bbox_xmax,
            bbox_ymax,
            gate_config.image_width_px,
            gate_config.image_height_px,
        )
    else:
        edge_distance_px = _get_float(raw_record, "edge_distance_px")
    if edge_distance_px is None and gate_config.edge_check_mode == "selected_pixel":
        edge_distance_px = _edge_distance_px(
            selected_u,
            selected_v,
            gate_config.image_width_px,
            gate_config.image_height_px,
        )

    _, quality_label, reason = _classify(
        raw_record,
        gate_config,
        selected_u=selected_u,
        selected_v=selected_v,
        edge_distance_px=edge_distance_px,
        bbox_width=bbox_width,
        bbox_height=bbox_height,
    )
    return SensorGateResult(
        admitted=bool(quality_label),
        reason=reason.value,
        gate_id=gate_config.gate_id,
        gate_config_hash=gate_config.config_hash(),
    )


def evaluate_observation_opportunity(
    raw_record: Mapping[str, Any],
    gate_config: UsableObservationGateConfig,
) -> ObservationOpportunity:
    """Evaluate one normalized raw opportunity into a labelled ObservationOpportunity.

    ``raw_record`` must contain operational fields only.  Any evaluation-only key raises
    :class:`LeakageError`; ``state_source`` of ``GT`` is rejected.
    """

    if not isinstance(raw_record, Mapping):
        raise ContractValidationError("raw_record must be a mapping")
    reject_evaluation_only_keys(raw_record, context="evaluate_observation_opportunity")

    state_source = str(raw_record.get("state_source", "")).upper()
    if state_source == SourceDomain.GT.value:
        raise LeakageError("state_source=GT is evaluation-only and cannot enter the gate")

    width = gate_config.image_width_px
    height = gate_config.image_height_px

    bbox_xmin = _get_float(raw_record, "bbox_xmin")
    bbox_ymin = _get_float(raw_record, "bbox_ymin")
    bbox_xmax = _get_float(raw_record, "bbox_xmax")
    bbox_ymax = _get_float(raw_record, "bbox_ymax")
    bbox_width = bbox_height = bbox_area = bbox_center_u = bbox_center_v = None
    if None not in (bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax):
        bbox_width = bbox_xmax - bbox_xmin
        bbox_height = bbox_ymax - bbox_ymin
        bbox_area = bbox_width * bbox_height
        bbox_center_u = 0.5 * (bbox_xmin + bbox_xmax)
        bbox_center_v = 0.5 * (bbox_ymin + bbox_ymax)

    selected_u = _get_float(raw_record, "selected_pixel_u")
    selected_v = _get_float(raw_record, "selected_pixel_v")
    if selected_u is None and bbox_center_u is not None:
        # default selected pixel = bottom-centre of the box (localizer convention)
        selected_u = bbox_center_u
        selected_v = bbox_ymax
    edge_distance_px = None
    if gate_config.edge_check_mode == "full_bbox":
        edge_distance_px = _bbox_edge_distance_px(
            bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, width, height
        )
    else:
        edge_distance_px = _get_float(raw_record, "edge_distance_px")
    if edge_distance_px is None and gate_config.edge_check_mode == "selected_pixel":
        edge_distance_px = _edge_distance_px(selected_u, selected_v, width, height)

    detection_label, quality_label, reason = _classify(
        raw_record,
        gate_config,
        selected_u=selected_u,
        selected_v=selected_v,
        edge_distance_px=edge_distance_px,
        bbox_width=bbox_width,
        bbox_height=bbox_height,
    )
    usable_label = detection_label & quality_label

    state_cov = raw_record.get("state_covariance")
    if state_cov is not None:
        state_cov = tuple(float(v) for v in state_cov)

    return ObservationOpportunity(
        timestamp=float(raw_record.get("timestamp", 0.0)),
        run_id=str(raw_record.get("run_id", "")),
        route_id=str(raw_record.get("route_id", "")),
        camera_id=str(raw_record.get("camera_id", "")),
        state_x=float(raw_record.get("state_x", 0.0)),
        state_y=float(raw_record.get("state_y", 0.0)),
        state_yaw=float(raw_record.get("state_yaw", 0.0)),
        state_source=state_source or SourceDomain.BELIEF.value,
        state_covariance=state_cov,
        frame_expected=_get_bool(raw_record, "frame_expected", default=True),
        frame_received=_get_bool(raw_record, "frame_received", default=True),
        frame_age_ms=_get_float(raw_record, "frame_age_ms"),
        detection_received=_get_bool(raw_record, "detection_received", default=False),
        detector_class=(None if raw_record.get("detector_class") is None else str(raw_record["detector_class"])),
        detector_confidence=_get_float(raw_record, "detector_confidence"),
        confidence_threshold=gate_config.confidence_threshold,
        bbox_xmin=bbox_xmin,
        bbox_ymin=bbox_ymin,
        bbox_xmax=bbox_xmax,
        bbox_ymax=bbox_ymax,
        bbox_width=bbox_width,
        bbox_height=bbox_height,
        bbox_area=bbox_area,
        bbox_center_u=bbox_center_u,
        bbox_center_v=bbox_center_v,
        selected_pixel_u=selected_u,
        selected_pixel_v=selected_v,
        image_width=width,
        image_height=height,
        edge_distance_px=edge_distance_px,
        projection_attempted=_get_bool(raw_record, "projection_attempted", default=False),
        projection_valid=_get_bool(raw_record, "projection_valid", default=False),
        association_attempted=_get_bool(raw_record, "association_attempted", default=False),
        association_valid=_get_bool(raw_record, "association_valid", default=False),
        tracking_available=_get_bool(raw_record, "tracking_available", default=False),
        tracking_valid=_get_bool(raw_record, "tracking_valid", default=False),
        accepted_by_localizer=_get_bool(raw_record, "accepted_by_localizer", default=False),
        detection_label=detection_label,
        quality_label=quality_label,
        usable_label=usable_label,
        failure_reason=reason.value,
    )
