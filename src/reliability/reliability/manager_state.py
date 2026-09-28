"""Validated operational belief snapshots for camera admission, never motion."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json

from reliability.contracts import ContractValidationError
from unav_common.operational_belief import (
    OperationalBelief, operational_belief_from_json, operational_belief_invalid_reason,
)


@dataclass(frozen=True)
class ManagerInputs:
    source_batch_id: str
    contracts: tuple
    belief_poses: tuple
    has_anchor: bool
    belief_identity: tuple | None
    motion: object
    belief_predictions: tuple = ()


class AdmissionBeliefHistory:
    """One bound producer epoch, monotonic revisions and prediction targets.

    A new committed revision invalidates predictions made from the old anchor.
    Epoch changes require the coordinated runtime restart, not a guess based on
    arrival order. Invalid snapshots retain their watermark and remove readiness.
    """
    def __init__(self, frame_id, capacity=400):
        self.frame_id, self.capacity = frame_id, capacity
        self.latest: OperationalBelief | None = None
        self.records: tuple[OperationalBelief, ...] = ()
        self._watermark = None
        self._last_payload = None
        self.reason = "missing_operational_belief"

    def invalidate(self, reason):
        self.records = ()
        self.reason = str(reason)

    def offer(self, payload, *, now_ns=None):
        try:
            if not isinstance(payload, dict):
                raise ValueError("operational belief must be an object")
            epoch, revision, target = payload.get("epoch"), payload.get("revision"), payload.get("state_stamp_ns")
            if not isinstance(epoch, str) or not epoch.strip() or type(revision) is not int or revision < 0:
                raise ValueError("invalid operational belief identity")
            if target is not None and (type(target) is not int or target < 0):
                raise ValueError("invalid operational belief target")
            previous_key = self._watermark
            if previous_key is not None and epoch != previous_key[0]:
                raise ValueError("belief epoch changed; coordinated restart required")
            # An explicit absent/invalid target invalidates the current revision
            # without comparing None to a numeric prediction timestamp.
            target_key = target if target is not None else (
                previous_key[2] if previous_key and revision == previous_key[1] else -1)
            key = (epoch, revision, target_key)
            if previous_key is not None and key < previous_key:
                return False
            if key == previous_key:
                encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
                if encoded != self._last_payload:
                    raise ValueError("conflicting operational belief revision/target")
                return False
            # Retain an identified refusal watermark before numerical validation;
            # neither an old delivery nor the same key can revive a poisoned prior.
            self._watermark, self._last_payload = key, None
            encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
            self._last_payload = encoded
            item = operational_belief_from_json(json.dumps(payload, allow_nan=False))
            if item.frame_id != self.frame_id:
                raise ValueError("operational belief frame differs from manager")
            if item.valid:
                # Shared numerical/support validation; camera-prior time matching
                # remains the existing max-delta gate, not a new age threshold.
                if now_ns is not None and item.state_stamp_ns > now_ns:
                    raise ValueError("future_belief_state")
                reason = operational_belief_invalid_reason(item,
                    now_ns=item.state_stamp_ns, expected_frame=self.frame_id, max_age_s=0.)
                if reason == "unsupported_motion":
                    item = replace(item, valid=False, invalid_reason=reason)
                elif reason:
                    raise ValueError(reason)
        except (TypeError, ValueError, OverflowError) as exc:
            self.invalidate(exc)
            raise ContractValidationError(str(exc)) from exc
        previous = self.latest
        if previous is not None:
            if item.revision == previous.revision:
                if item.anchor_stamp_ns != previous.anchor_stamp_ns:
                    self.invalidate("anchor_changed_without_revision")
                    raise ContractValidationError("belief revision changed anchor without revision increment")
        reset = previous is None or item.revision != previous.revision or not item.valid
        self.records = (() if reset else self.records)
        if item.valid:
            self.records = (*self.records, item)[-self.capacity:]
        self.latest = item
        self.reason = "" if item.valid else item.invalid_reason
        return True

    def poses(self):
        return tuple((item.state_stamp_ns / 1e9, item.mean) for item in self.records)

    def predictions(self):
        """Timestamped means and covariances for measurement admission.

        The covariance is not motion evidence. It is retained only so an online
        observation model can marginalise a nuisance orientation instead of
        treating the point-estimate yaw as known exactly.
        """

        return tuple(
            (item.state_stamp_ns / 1e9, item.mean, item.covariance)
            for item in self.records
        )
