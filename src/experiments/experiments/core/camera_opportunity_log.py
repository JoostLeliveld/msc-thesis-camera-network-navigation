"""Lossless append-only JSONL delivery journals used by the experiment logger.

The journals retain the exact string delivered by ROS before interpreting it. A
canonical CSV may suppress an identical retransmission, but this layer never does.
State used for numbering and duplicate classification advances only after a row is
written and flushed successfully.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def _finite_stamp(value: Any, *, field_name: str) -> float:
    stamp = float(value)
    if not math.isfinite(stamp):
        raise ValueError(f"{field_name} is not finite")
    return stamp


def _strict_object(payload: str) -> dict[str, Any]:
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("payload is not an object")
    # Python's decoder accepts NaN and Infinity. Re-encoding strictly catches
    # them recursively before they can make the audit journal itself fail.
    json.dumps(value, allow_nan=False)
    return value


class JsonlDeliveryLog:
    """Record every raw string delivery and its parse result in write order."""

    def __init__(self, handle, *, schema: str = "runtime_event_delivery.v1"):
        self.handle = handle
        self.schema = str(schema)
        self.rows = 0
        self._event_payload_sha256: dict[str, str] = {}

    def append(self, topic: str, payload: str, receive_stamp: float) -> dict[str, Any]:
        raw = str(payload)
        record: dict[str, Any] = {
            "schema": self.schema,
            "topic": str(topic),
            "receive_stamp_s": None,
            "delivery_index": self.rows,
            "raw_payload": raw,
        }
        try:
            record["receive_stamp_s"] = _finite_stamp(
                receive_stamp, field_name="receive_stamp_s"
            )
            parsed = _strict_object(raw)
            record.update(parsed_payload=parsed, valid_json_object=True)
            event_id = str(parsed.get("event_id", "") or "").strip()
            if event_id:
                digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
                previous = self._event_payload_sha256.get(event_id)
                record.update(
                    event_id=event_id,
                    duplicate_delivery=previous is not None,
                    conflicting_delivery=previous is not None and previous != digest,
                    payload_sha256=digest,
                )
        except (ValueError, TypeError, OverflowError) as exc:
            record.update(valid_json_object=False, reason=str(exc))

        encoded = json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n"
        self.handle.write(encoded)
        self.handle.flush()
        self.rows += 1
        if record.get("event_id") and not record.get("duplicate_delivery"):
            self._event_payload_sha256[record["event_id"]] = record["payload_sha256"]
        return record


class CameraOpportunityLog:
    """Raw detector-output ledger, including misses and conflicting deliveries."""

    def __init__(self, handle):
        self.handle = handle
        self.seen: dict[tuple[str, str], str] = {}
        self.rows = 0

    def append(self, camera: str, payload: str, receive_stamp: float) -> dict[str, Any]:
        raw = str(payload)
        record: dict[str, Any] = {
            "schema": "camera_opportunity_log.v2",
            "topic_camera": str(camera),
            "receive_stamp_s": None,
            "delivery_index": self.rows,
            "raw_payload": raw,
        }
        identity = None
        digest = None
        try:
            record["receive_stamp_s"] = _finite_stamp(
                receive_stamp, field_name="receive_stamp_s"
            )
            observation = _strict_object(raw)
            if observation.get("schema_version") != "phase0.v1":
                raise ValueError("unsupported camera observation schema_version")
            _finite_stamp(observation["timestamp_s"], field_name="timestamp_s")
            batch = observation["source_batch_id"]
            if not isinstance(batch, str) or not batch:
                raise ValueError("missing batch identity")
            if observation["camera_id"] != camera:
                raise ValueError("topic camera and observation camera disagree")
            if not isinstance(observation["detection_valid"], bool):
                raise ValueError("detection_valid is not boolean")
            identity = (str(camera), batch)
            digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            previous = self.seen.get(identity)
            record.update(
                observation=observation,
                duplicate=previous is not None,
                conflicting_duplicate=previous is not None and previous != digest,
                payload_sha256=digest,
                valid_contract=True,
            )
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            record.update(valid_contract=False, reason=str(exc))

        encoded = json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n"
        self.handle.write(encoded)
        self.handle.flush()
        self.rows += 1
        if identity is not None and identity not in self.seen:
            self.seen[identity] = str(digest)
        return record
