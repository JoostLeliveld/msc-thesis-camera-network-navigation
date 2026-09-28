"""Observed identity contract for the strict four-camera detector runtime.

The commissioning recorder must not infer the model or batching topology from
its own command line.  The running detector publishes this compact, canonical
JSON contract after a successful model load and warm-up; the recorder compares
it with its frozen expectation before it writes a single operational row.

This module deliberately has no ROS dependency so the publisher and recorder
can share the byte-level validation logic in unit tests and command-line tools.
"""

from __future__ import annotations

import hashlib
from collections import namedtuple
import json
import math
from pathlib import Path
from typing import Any, Mapping


#: Every contract version this module can validate, and what is fixed in each.
#:
#: A contract is a RECORD of how a run was configured, so an old record must stay
#: valid forever: the frozen four-camera commissioning campaign has ledgers that
#: declare v1, and they have to keep validating after the live runtime moves on.
#: Before this table existed, the validator only knew the current version, so any
#: change to the live topology retroactively invalidated the frozen study's ledgers
#: -- which is what happened when the batch went from A--D to A--E.
#:
#: `build_batched_runtime_contract` always emits the CURRENT version. Validation
#: uses the version the payload itself declares.
_ContractSchema = namedtuple("_ContractSchema", ("camera_order", "frame_policy"))
_KNOWN_CONTRACT_SCHEMAS: dict[str, "_ContractSchema"] = {
    # the frozen warehouse_full_4cam commissioning campaign
    "four_camera_detector_runtime_contract.v1": _ContractSchema(
        camera_order=("camera_A", "camera_B", "camera_C", "camera_D"),
        frame_policy="strict_new_unique_stamp_latest_only_no_reuse",
    ),
    # current: warehouse_v2's five wall cameras, and pending frames grouped by
    # stamp so a consumer slower than the frame rate cannot break a forming round
    "four_camera_detector_runtime_contract.v2": _ContractSchema(
        camera_order=("camera_A", "camera_B", "camera_C", "camera_D", "camera_E"),
        frame_policy="strict_stamp_bucketed_round_no_reuse",
    ),
}
RUNTIME_CONTRACT_SCHEMA_VERSION = "four_camera_detector_runtime_contract.v2"
RUNTIME_CONTRACT_TOPIC = "/perception/four_camera_detector_runtime_contract"

BATCHED_RUNTIME_MODE = "batched_four_camera"
BATCHED_EXECUTABLE = "batched_four_camera_yolo_node"
BATCHED_MODEL_FORMAT = "native_ultralytics"
BATCHED_CAMERA_ORDER = _KNOWN_CONTRACT_SCHEMAS[RUNTIME_CONTRACT_SCHEMA_VERSION].camera_order
BATCHED_FRAME_POLICY = _KNOWN_CONTRACT_SCHEMAS[RUNTIME_CONTRACT_SCHEMA_VERSION].frame_policy
BATCHED_INFERENCE_TIMING_SEMANTICS = "full_batch_wall_ms_repeated_per_camera_result"
BATCHED_FAULT_POLICY = "fatal_process_exit_and_launch_shutdown_no_synthetic_miss"
BATCHED_SYNCHRONIZATION_MISS_POLICY = "no_output_detected_by_liveness_gate"

EXECUTABLE_SOURCE_KEY = "perception.nodes.batched_four_camera_yolo_node"
FOUR_CAMERA_BATCH_SOURCE_KEY = "perception.core.four_camera_batch"
RUNTIME_CONTRACT_SOURCE_KEY = "perception.core.four_camera_runtime_contract"
YOLO_SELECTION_SOURCE_KEY = "perception.core.yolo_selection"
REQUIRED_SOURCE_KEYS = (
    EXECUTABLE_SOURCE_KEY,
    FOUR_CAMERA_BATCH_SOURCE_KEY,
    RUNTIME_CONTRACT_SOURCE_KEY,
    YOLO_SELECTION_SOURCE_KEY,
)

_CONTRACT_KEYS = frozenset(
    {
        "schema_version",
        "runtime_mode",
        "executable",
        "model_format",
        "model_path",
        "model_sha256",
        "imgsz",
        "confidence_threshold",
        "iou_threshold",
        "use_masks",
        "model_instances",
        "batch_size",
        "camera_order",
        "shared_device",
        "cpu_threads",
        "interop_threads",
        "opencv_threads",
        "max_batch_stamp_skew_s",
        "max_pending_wall_s",
        "frame_policy",
        "inference_timing_semantics",
        "fault_policy",
        "synchronization_miss_policy",
        "executable_source_sha256",
        "source_hashes",
        "contract_sha256",
    }
)


class RuntimeContractError(ValueError):
    """Raised when a detector identity contract is absent, malformed, or stale."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of an exact local file without following ambiguity."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeContractError(f"required runtime file is missing: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes_from_paths(paths: Mapping[str, Path]) -> dict[str, str]:
    """Hash the exact runtime source modules using stable logical names."""

    supplied = set(paths)
    expected = set(REQUIRED_SOURCE_KEYS)
    if supplied != expected:
        missing = sorted(expected.difference(supplied))
        unexpected = sorted(supplied.difference(expected))
        raise RuntimeContractError(
            "runtime source set is not exact: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {key: sha256_file(Path(paths[key])) for key in REQUIRED_SOURCE_KEYS}


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def runtime_contract_sha256(contract: Mapping[str, Any]) -> str:
    """Hash a contract while excluding its self-referential digest field."""

    payload = dict(contract)
    payload.pop("contract_sha256", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def runtime_contract_json(contract: Mapping[str, Any]) -> str:
    """Return the canonical JSON form after validating its full schema."""

    validated = validate_batched_runtime_contract(contract)
    return _canonical_bytes(validated).decode("utf-8")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeContractError(f"{field} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise RuntimeContractError(f"{field} must be a finite number")
    return numeric


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeContractError(f"{field} must be a positive integer")
    return int(value)


def _require_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeContractError(f"{field} must be a non-empty string")
    return value


def build_batched_runtime_contract(
    *,
    model_path: Path | str,
    model_sha256: str,
    imgsz: int,
    confidence_threshold: float,
    iou_threshold: float,
    use_masks: bool,
    shared_device: str,
    cpu_threads: int,
    interop_threads: int,
    opencv_threads: int,
    max_batch_stamp_skew_s: float,
    max_pending_wall_s: float,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Build the one supported observed contract for the batched runtime."""

    contract: dict[str, Any] = {
        "schema_version": RUNTIME_CONTRACT_SCHEMA_VERSION,
        "runtime_mode": BATCHED_RUNTIME_MODE,
        "executable": BATCHED_EXECUTABLE,
        "model_format": BATCHED_MODEL_FORMAT,
        "model_path": str(Path(model_path).expanduser().resolve()),
        "model_sha256": str(model_sha256),
        "imgsz": int(imgsz),
        "confidence_threshold": float(confidence_threshold),
        "iou_threshold": float(iou_threshold),
        "use_masks": bool(use_masks),
        "model_instances": 1,
        "batch_size": len(BATCHED_CAMERA_ORDER),
        "camera_order": list(BATCHED_CAMERA_ORDER),
        "shared_device": str(shared_device).strip() or "auto",
        "cpu_threads": int(cpu_threads),
        "interop_threads": int(interop_threads),
        "opencv_threads": int(opencv_threads),
        "max_batch_stamp_skew_s": float(max_batch_stamp_skew_s),
        "max_pending_wall_s": float(max_pending_wall_s),
        "frame_policy": BATCHED_FRAME_POLICY,
        "inference_timing_semantics": BATCHED_INFERENCE_TIMING_SEMANTICS,
        "fault_policy": BATCHED_FAULT_POLICY,
        "synchronization_miss_policy": BATCHED_SYNCHRONIZATION_MISS_POLICY,
        "executable_source_sha256": str(source_hashes.get(EXECUTABLE_SOURCE_KEY, "")),
        "source_hashes": {key: str(source_hashes.get(key, "")) for key in REQUIRED_SOURCE_KEYS},
    }
    contract["contract_sha256"] = runtime_contract_sha256(contract)
    return validate_batched_runtime_contract(contract)


def validate_batched_runtime_contract(contract: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Validate the complete, versioned batched contract and its self-hash."""

    if not isinstance(contract, Mapping):
        raise RuntimeContractError("runtime contract must be a JSON object")
    payload = dict(contract)
    supplied_keys = set(payload)
    if supplied_keys != _CONTRACT_KEYS:
        missing = sorted(_CONTRACT_KEYS.difference(supplied_keys))
        unexpected = sorted(supplied_keys.difference(_CONTRACT_KEYS))
        raise RuntimeContractError(
            "runtime contract schema mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    schema = _KNOWN_CONTRACT_SCHEMAS.get(payload["schema_version"])
    if schema is None:
        raise RuntimeContractError(
            "unsupported runtime contract schema_version "
            f"{payload['schema_version']!r}; known: {sorted(_KNOWN_CONTRACT_SCHEMAS)}"
        )
    if payload["runtime_mode"] != BATCHED_RUNTIME_MODE:
        raise RuntimeContractError("runtime contract is not the batched four-camera mode")
    if payload["executable"] != BATCHED_EXECUTABLE:
        raise RuntimeContractError("runtime contract executable is not the batched node")
    if payload["model_format"] != BATCHED_MODEL_FORMAT:
        raise RuntimeContractError("runtime contract model format is not native Ultralytics")
    for field in ("model_path", "shared_device"):
        _require_string(payload[field], field=field)
    for field in (
        "model_sha256",
        "executable_source_sha256",
        "contract_sha256",
    ):
        if not _is_sha256(payload[field]):
            raise RuntimeContractError(f"{field} must be a lowercase SHA-256 digest")
    for field, expected in (
        ("model_instances", 1),
        ("batch_size", len(schema.camera_order)),
    ):
        actual = _positive_int(payload[field], field=field)
        if actual != expected:
            raise RuntimeContractError(f"{field} must be {expected} for the batched runtime")
    for field in ("imgsz", "cpu_threads", "interop_threads", "opencv_threads"):
        _positive_int(payload[field], field=field)
    if not isinstance(payload["use_masks"], bool):
        raise RuntimeContractError("use_masks must be boolean")
    for field in ("confidence_threshold", "iou_threshold"):
        value = _finite_float(payload[field], field=field)
        if not 0.0 <= value <= 1.0:
            raise RuntimeContractError(f"{field} must lie in [0, 1]")
    skew = _finite_float(payload["max_batch_stamp_skew_s"], field="max_batch_stamp_skew_s")
    pending = _finite_float(payload["max_pending_wall_s"], field="max_pending_wall_s")
    if skew < 0.0 or pending <= 0.0:
        raise RuntimeContractError("runtime contract timing bounds are invalid")
    if tuple(payload["camera_order"]) != schema.camera_order:
        raise RuntimeContractError(
            f"runtime contract camera order for {payload['schema_version']} is not "
            "exactly " + ", ".join(schema.camera_order)
        )
    fixed_strings = {
        "frame_policy": schema.frame_policy,
        "inference_timing_semantics": BATCHED_INFERENCE_TIMING_SEMANTICS,
        "fault_policy": BATCHED_FAULT_POLICY,
        "synchronization_miss_policy": BATCHED_SYNCHRONIZATION_MISS_POLICY,
    }
    for field, expected in fixed_strings.items():
        if payload[field] != expected:
            raise RuntimeContractError(f"runtime contract {field} differs from the fixed policy")
    source_hashes = payload["source_hashes"]
    if not isinstance(source_hashes, Mapping) or set(source_hashes) != set(REQUIRED_SOURCE_KEYS):
        raise RuntimeContractError("runtime contract source_hashes are incomplete")
    for key in REQUIRED_SOURCE_KEYS:
        if not _is_sha256(source_hashes[key]):
            raise RuntimeContractError(f"source hash is invalid for {key}")
    if payload["executable_source_sha256"] != source_hashes[EXECUTABLE_SOURCE_KEY]:
        raise RuntimeContractError("executable source hash does not match source_hashes")
    expected_digest = runtime_contract_sha256(payload)
    if payload["contract_sha256"] != expected_digest:
        raise RuntimeContractError("runtime contract SHA-256 does not match its payload")
    # JSON round-trip makes callers retain plain, immutable-compatible values
    # rather than references owned by a ROS callback.
    return json.loads(_canonical_bytes(payload).decode("utf-8"))


def compare_batched_runtime_contract(
    observed: Mapping[str, Any] | Any,
    expected: Mapping[str, Any] | Any,
) -> dict[str, Any]:
    """Fail closed unless every observed deployment field equals the freeze."""

    observed_payload = validate_batched_runtime_contract(observed)
    expected_payload = validate_batched_runtime_contract(expected)
    mismatches: list[str] = []
    for field in sorted(_CONTRACT_KEYS.difference({"contract_sha256"})):
        observed_value = observed_payload[field]
        expected_value = expected_payload[field]
        if isinstance(expected_value, float):
            if not math.isclose(
                float(observed_value), float(expected_value), rel_tol=0.0, abs_tol=1.0e-12
            ):
                mismatches.append(field)
        elif observed_value != expected_value:
            mismatches.append(field)
    if mismatches:
        raise RuntimeContractError(
            "observed detector runtime differs from the frozen expectation: "
            + ", ".join(mismatches)
        )
    if observed_payload["contract_sha256"] != expected_payload["contract_sha256"]:
        raise RuntimeContractError("matching runtime fields produced inconsistent contract hashes")
    return observed_payload
