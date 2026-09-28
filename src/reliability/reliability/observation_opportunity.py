"""Usable-observation contract (P1 of the p_det / p_qual / p_use refocus).

One :class:`ObservationOpportunity` record must exist for *every* expected camera
opportunity, including misses.  This is the central difference from the legacy
``opportunity.build_opportunity_row`` (which returns ``None`` on a gate failure and
therefore silently drops negatives).  Here a gate failure produces a record with the
appropriate labels and a controlled :class:`FailureReason`.

Label semantics (frozen):

    detection_label = 1  iff the detector produced a robot-class detection candidate
                          (a box) for that camera opportunity, *independent of the
                          confidence threshold*.
    quality_label   = 1  iff detection_label == 1 AND all enabled frozen operational
                          quality checks pass (confidence, projection, freshness,
                          association, track, edge-clip, class/id, localizer accept).
    usable_label    = detection_label AND quality_label.

Because quality_label already requires detection_label, ``usable_label == quality_label``
whenever the record is well-formed; the product form is kept explicit to match the
p_use = p_det * p_qual decomposition and to make an accidental inconsistency detectable.

The confidence threshold is a *quality* check (see :class:`FailureReason.LOW_CONFIDENCE`),
not part of detection_label — this is what makes the A5/A6 ablations (quality gate with vs
without confidence) well defined.

No Gazebo ground truth enters this contract.  Any ``gt_*`` / ``eval_*`` / oracle key in a
raw record raises :class:`~reliability.contracts.LeakageError`, and ``state_source`` of
``GT`` is rejected outright.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Mapping

from reliability.contracts import (
    ContractValidationError,
    LeakageError,
    reject_evaluation_only_keys,
)

OBS_SCHEMA_VERSION = "obs_opportunity_v1"


class FailureReason(str, Enum):
    """Controlled enumeration of why a camera opportunity was not usable.

    Ordered by evaluation priority (earliest failing gate wins).  ``USABLE`` is the only
    value with ``usable_label == 1``.  ``UNKNOWN`` marks a record the gate could not
    decide from the available operational fields (the exporter counts and may drop these;
    it must never be silently treated as a negative).
    """

    NO_FRAME = "NO_FRAME"
    STALE_FRAME = "STALE_FRAME"
    NO_DETECTION = "NO_DETECTION"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    WRONG_CLASS_OR_ID = "WRONG_CLASS_OR_ID"
    BOX_TOO_SMALL = "BOX_TOO_SMALL"
    INVALID_PROJECTION = "INVALID_PROJECTION"
    INVALID_ASSOCIATION = "INVALID_ASSOCIATION"
    INVALID_TRACK = "INVALID_TRACK"
    CLIPPED_OR_EDGE = "CLIPPED_OR_EDGE"
    REJECTED_BY_LOCALIZER = "REJECTED_BY_LOCALIZER"
    USABLE = "USABLE"
    UNKNOWN = "UNKNOWN"


class SourceDomain(str, Enum):
    """Provenance tag every field must carry (task realism rule §2)."""

    GT = "GT"        # Gazebo ground truth — evaluation only, never in this contract
    ODOM = "ODOM"    # wheel odometry
    BELIEF = "BELIEF"  # planner belief / EKF posterior
    STATE = "STATE"  # state-estimate node output
    PIXEL = "PIXEL"  # image-plane detector output
    MODEL = "MODEL"  # a fitted artifact


_OPERATIONAL_STATE_SOURCES = frozenset({SourceDomain.ODOM.value, SourceDomain.BELIEF.value, SourceDomain.STATE.value})

# Static provenance map for the record's own fields.  Used to populate ``source_labels``
# and to document, per field, where the value legitimately comes from.
FIELD_SOURCE_DOMAINS: dict[str, str] = {
    "timestamp": SourceDomain.STATE.value,
    "run_id": SourceDomain.STATE.value,
    "route_id": SourceDomain.STATE.value,
    "camera_id": SourceDomain.STATE.value,
    "state_x": "STATE",
    "state_y": "STATE",
    "state_yaw": "STATE",
    "state_covariance": "STATE",
    "frame_expected": SourceDomain.PIXEL.value,
    "frame_received": SourceDomain.PIXEL.value,
    "frame_age_ms": SourceDomain.PIXEL.value,
    "detection_received": SourceDomain.PIXEL.value,
    "detector_class": SourceDomain.PIXEL.value,
    "detector_confidence": SourceDomain.PIXEL.value,
    "bbox": SourceDomain.PIXEL.value,
    "selected_pixel": SourceDomain.PIXEL.value,
    "projection_valid": SourceDomain.STATE.value,
    "association_valid": SourceDomain.STATE.value,
    "tracking_valid": SourceDomain.STATE.value,
    "accepted_by_localizer": SourceDomain.STATE.value,
    "detection_label": SourceDomain.PIXEL.value,
    "quality_label": SourceDomain.MODEL.value,
    "usable_label": SourceDomain.MODEL.value,
}


def _finite(value: Any, *, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ContractValidationError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(out):
        raise ContractValidationError(f"{name} must be finite, got {value!r}")
    return out


def _opt_finite(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    return _finite(value, name=name)


def _label(value: Any, *, name: str) -> int:
    ivalue = int(value)
    if ivalue not in (0, 1):
        raise ContractValidationError(f"{name} must be 0 or 1, got {value!r}")
    return ivalue


@dataclass(frozen=True)
class ObservationOpportunity:
    """One camera opportunity (detection *or* miss) with frozen usability labels.

    Miss records carry ``None`` for detector-conditional fields (bbox, confidence,
    selected pixel).  ``detection_label`` / ``quality_label`` / ``usable_label`` and
    ``failure_reason`` are always populated.
    """

    # identity / context ---------------------------------------------------------------
    timestamp: float
    run_id: str
    route_id: str
    camera_id: str

    # operational state (never GT) -----------------------------------------------------
    state_x: float
    state_y: float
    state_yaw: float
    state_source: str
    state_covariance: tuple[float, float, float] | None  # (xx, xy, yy), optional

    # frame liveness -------------------------------------------------------------------
    frame_expected: bool
    frame_received: bool
    frame_age_ms: float | None

    # detection ------------------------------------------------------------------------
    detection_received: bool
    detector_class: str | None
    detector_confidence: float | None
    confidence_threshold: float

    # bounding box / image geometry ----------------------------------------------------
    bbox_xmin: float | None
    bbox_ymin: float | None
    bbox_xmax: float | None
    bbox_ymax: float | None
    bbox_width: float | None
    bbox_height: float | None
    bbox_area: float | None
    bbox_center_u: float | None
    bbox_center_v: float | None
    selected_pixel_u: float | None
    selected_pixel_v: float | None
    image_width: int
    image_height: int
    edge_distance_px: float | None

    # downstream operational checks ----------------------------------------------------
    projection_attempted: bool
    projection_valid: bool
    association_attempted: bool
    association_valid: bool
    tracking_available: bool
    tracking_valid: bool
    accepted_by_localizer: bool

    # labels ---------------------------------------------------------------------------
    detection_label: int
    quality_label: int
    usable_label: int

    # provenance / bookkeeping ---------------------------------------------------------
    failure_reason: str
    source_labels: dict[str, str] = field(default_factory=dict)
    schema_version: str = OBS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OBS_SCHEMA_VERSION:
            raise ContractValidationError(f"Unsupported schema_version {self.schema_version!r}")
        if not self.run_id or not self.camera_id:
            raise ContractValidationError("ObservationOpportunity needs run_id and camera_id")
        if self.state_source not in _OPERATIONAL_STATE_SOURCES:
            raise LeakageError(
                f"state_source must be one of {sorted(_OPERATIONAL_STATE_SOURCES)}; "
                f"got {self.state_source!r} (GT is evaluation-only and forbidden here)"
            )
        if self.failure_reason not in {member.value for member in FailureReason}:
            raise ContractValidationError(f"unknown failure_reason {self.failure_reason!r}")

        object.__setattr__(self, "timestamp", _finite(self.timestamp, name="timestamp"))
        for name in ("state_x", "state_y", "state_yaw"):
            object.__setattr__(self, name, _finite(getattr(self, name), name=name))
        object.__setattr__(self, "confidence_threshold", _finite(self.confidence_threshold, name="confidence_threshold"))

        if self.state_covariance is not None:
            cov = tuple(_finite(v, name="state_covariance") for v in self.state_covariance)
            if len(cov) != 3:
                raise ContractValidationError("state_covariance must be (xx, xy, yy)")
            object.__setattr__(self, "state_covariance", cov)

        object.__setattr__(self, "detection_label", _label(self.detection_label, name="detection_label"))
        object.__setattr__(self, "quality_label", _label(self.quality_label, name="quality_label"))
        object.__setattr__(self, "usable_label", _label(self.usable_label, name="usable_label"))

        # Structural invariants of the decomposition.
        if self.quality_label == 1 and self.detection_label == 0:
            raise ContractValidationError("quality_label=1 requires detection_label=1")
        if self.usable_label != (self.detection_label & self.quality_label):
            raise ContractValidationError("usable_label must equal detection_label AND quality_label")
        if (self.failure_reason == FailureReason.USABLE.value) != (self.usable_label == 1):
            raise ContractValidationError("failure_reason USABLE iff usable_label == 1")

        # Firewall: no evaluation-only key may ride along in source_labels/metadata.
        reject_evaluation_only_keys(dict(self.source_labels), context="ObservationOpportunity.source_labels")

        if not self.source_labels:
            object.__setattr__(self, "source_labels", dict(FIELD_SOURCE_DOMAINS))
        if self.source_labels.get("state_x", "STATE") == SourceDomain.GT.value:
            raise LeakageError("source_labels marks state as GT")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ObservationOpportunity":
        if not isinstance(payload, Mapping):
            raise ContractValidationError("ObservationOpportunity payload must be a mapping")
        reject_evaluation_only_keys(payload, context="ObservationOpportunity")
        known = {f_.name for f_ in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = set(payload) - known
        if unknown:
            raise ContractValidationError("ObservationOpportunity unknown fields: " + ", ".join(sorted(unknown)))
        data = dict(payload)
        for key in ("state_covariance",):
            if data.get(key) is not None:
                data[key] = tuple(data[key])
        return cls(**data)  # type: ignore[arg-type]
