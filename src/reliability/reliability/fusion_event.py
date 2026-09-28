"""Immutable correction envelopes and their publication transaction.

The correction is a measurement, never a posterior. Capture operands, common-time
operands and the final correction retain separate values and exact timestamps.
Publication status belongs in the append-only journal, outside the payload hash.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

from reliability.contracts import CameraObservation, ContractValidationError, _as_pair, _as_matrix_2x2, _validate_spd_2x2
from reliability.fusion import MapObservation, camera_measurement_batch


def canonical_json(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _identity(value, name):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ContractValidationError(f"{name} must be a nonempty canonical string")
    return value


def _stamp(value, name):
    if type(value) is not int or value < 0:
        raise ContractValidationError(f"{name} must be nonnegative integer ns")
    return value


@dataclass(frozen=True)
class FusedCorrectionEvent:
    json: str

    @property
    def payload(self):
        return json.loads(self.json)

    @classmethod
    def create(cls, *, source_batch_id, epoch, publication_seq, frame_id,
               common_capture_stamp_ns, correction_stamp_ns, xy, covariance_m2,
               accepted_camera_ids, contracts, capture_observations, aligned_observations,
               motion_support_by_camera, model, belief_identity=None,
               common_time_xy=None, common_time_covariance_m2=None, correction_motion_support=None):
        _identity(source_batch_id, "source_batch_id")
        _identity(epoch, "epoch")
        _identity(frame_id, "frame_id")
        _stamp(publication_seq, "publication_seq")
        common_ns = _stamp(common_capture_stamp_ns, "common_capture_stamp_ns")
        corrected_ns = _stamp(correction_stamp_ns, "correction_stamp_ns")
        if corrected_ns < common_ns:
            raise ContractValidationError("correction stamp precedes common capture")
        mean = _as_pair(xy, field_name="xy")
        covariance = _as_matrix_2x2(covariance_m2, field_name="covariance_m2")
        _validate_spd_2x2(covariance, field_name="covariance_m2")
        common_xy = mean if common_time_xy is None else _as_pair(common_time_xy, field_name="common_time_xy")
        common_cov = covariance if common_time_covariance_m2 is None else _as_matrix_2x2(
            common_time_covariance_m2, field_name="common_time_covariance_m2")
        _validate_spd_2x2(common_cov, field_name="common_time_covariance_m2")
        if corrected_ns != common_ns:
            _supported_interval(correction_motion_support, common_ns, corrected_ns, frame_id)
            delta = correction_motion_support["delta_xy_m"]
            if any(not math.isclose(mean[i], common_xy[i]+delta[i], abs_tol=1e-10, rel_tol=1e-12) for i in range(2)):
                raise ContractValidationError("correction mean differs from supported displacement")
        elif mean != common_xy:
            raise ContractValidationError("same-time correction changed the fused mean")
        originals = {o.camera_id: o for o in camera_measurement_batch(capture_observations)}
        aligned = {o.camera_id: o for o in camera_measurement_batch(aligned_observations)}
        accepted = tuple(accepted_camera_ids)
        if not accepted or len(set(accepted)) != len(accepted):
            raise ContractValidationError("accepted cameras must be unique and nonempty")
        by_camera, member_ids, epochs = {}, set(), set()
        for observation in contracts:
            if observation.camera_id in by_camera:
                raise ContractValidationError("duplicate source camera")
            if observation.source_batch_id != source_batch_id or observation.capture_stamp_ns is None:
                raise ContractValidationError("source member lacks physical batch/capture identity")
            if observation.source_frame_id in member_ids:
                raise ContractValidationError("duplicate source frame identity")
            member_ids.add(observation.source_frame_id)
            epochs.add(observation.producer_epoch)
            by_camera[observation.camera_id] = observation
        if len(epochs) != 1 or not all(c in by_camera and c in originals and c in aligned for c in accepted):
            raise ContractValidationError("fused members have missing operands or mixed producer epochs")
        if not set(originals) <= set(by_camera) or not set(aligned) <= set(originals):
            raise ContractValidationError("camera operands do not match the physical batch")
        rows = []
        for camera in sorted(by_camera):
            source = by_camera[camera]
            original, common = originals.get(camera), aligned.get(camera)
            if original is not None:
                tolerance = max(1e-9, math.ulp(source.timestamp_s))
                if abs(original.timestamp_s-source.timestamp_s) > tolerance:
                    raise ContractValidationError("capture operand was relabelled")
            if common is not None:
                if round(common.timestamp_s*1e9) != common_ns and abs(common.timestamp_s-common_ns/1e9) > math.ulp(common_ns/1e9):
                    raise ContractValidationError("aligned operand has another common time")
                if source.capture_stamp_ns > common_ns:
                    raise ContractValidationError("aligned operand precedes its capture")
                support = motion_support_by_camera.get(camera)
                if source.capture_stamp_ns != common_ns:
                    _supported_interval(support, source.capture_stamp_ns, common_ns, frame_id)
                    if any(not math.isclose(common.xy_m[i], original.xy_m[i]+support["delta_xy_m"][i],
                                             abs_tol=1e-10, rel_tol=1e-12) for i in range(2)):
                        raise ContractValidationError("aligned mean differs from supported displacement")
            rows.append(dict(camera_id=camera, producer_epoch=source.producer_epoch,
                             source_frame_id=source.source_frame_id,
                             detector_invocation_id=source.detector_invocation_id,
                             capture_stamp_ns=source.capture_stamp_ns,
                             source_observation=source.to_dict(),
                             capture_observation=None if original is None else original.to_dict(),
                             common_observation=None if common is None else common.to_dict(),
                             used=camera in accepted,
                             motion_support=motion_support_by_camera.get(camera)))
        payload = dict(schema_version=2, source_batch_id=source_batch_id,
                       event_id=f"{epoch}:fusion:{publication_seq}", epoch=epoch,
                       publication_seq=publication_seq, frame_id=frame_id,
                       common_capture_stamp_ns=common_ns, correction_stamp_ns=corrected_ns,
                       common_capture_stamp=common_ns/1e9, correction_stamp=corrected_ns/1e9,
                       xy=list(mean), covariance_m2=[list(row) for row in covariance],
                       accepted_camera_ids=sorted(accepted), members=rows,
                       member_ids=[by_camera[c].source_frame_id for c in sorted(by_camera)],
                       accepted_member_ids=[by_camera[c].source_frame_id for c in sorted(accepted)],
                       observation_model=dict(model), admission_belief_identity=belief_identity)
        payload.update(common_time_xy=list(common_xy),
                       common_time_covariance_m2=[list(row) for row in common_cov],
                       correction_motion_support=correction_motion_support)
        payload["payload_sha256"] = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
        return cls(canonical_json(payload))

    @classmethod
    def from_json(cls, text):
        """Validate untrusted wire content, including member and time consistency."""
        payload = json.loads(text)
        if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int or payload["schema_version"] != 2:
            raise ContractValidationError("unsupported fused correction event schema")
        claimed = payload.pop("payload_sha256", None)
        expected = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
        if claimed != expected:
            raise ContractValidationError("fused correction payload hash mismatch")
        payload["payload_sha256"] = claimed
        try:
            members = payload["members"]
            event = cls.create(source_batch_id=payload["source_batch_id"], epoch=payload["epoch"],
                publication_seq=payload["publication_seq"], frame_id=payload["frame_id"],
                common_capture_stamp_ns=payload["common_capture_stamp_ns"], correction_stamp_ns=payload["correction_stamp_ns"],
                xy=payload["xy"], covariance_m2=payload["covariance_m2"], accepted_camera_ids=payload["accepted_camera_ids"],
                contracts=[CameraObservation.from_dict(m["source_observation"]) for m in members],
                capture_observations=[MapObservation.from_dict(m["capture_observation"]) for m in members if m["capture_observation"] is not None],
                aligned_observations=[MapObservation.from_dict(m["common_observation"]) for m in members if m["common_observation"] is not None],
                motion_support_by_camera={m["camera_id"]: m["motion_support"] for m in members},
                model=payload["observation_model"], belief_identity=payload["admission_belief_identity"],
                common_time_xy=payload["common_time_xy"], common_time_covariance_m2=payload["common_time_covariance_m2"],
                correction_motion_support=payload["correction_motion_support"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractValidationError(f"invalid fused correction event: {exc}") from exc
        if event.json != canonical_json(payload):
            raise ContractValidationError("fused correction fields disagree with immutable members")
        return event


def _supported_interval(support, start_ns, end_ns, frame_id):
    if (not isinstance(support, dict) or support.get("supported") is not True
            or support.get("start_ns") != start_ns or support.get("end_ns") != end_ns
            or support.get("frame_id") != frame_id):
        raise ContractValidationError("aligned operand has no verified motion interval/frame")
    _as_pair(support.get("delta_xy_m"), field_name="motion delta_xy_m")


def publish_fused_event(event, *, journal, publish_envelope, publish_decision,
                        publish_compatibility=()):
    """Persist intent first and never retry a correction after ambiguous publication.

    Each sink is attempted independently; compatibility errors cannot prevent
    the immutable decision from being attempted. Any error remains a fatal
    integrity event for the owner; its complete prefix is already journalled.
    """
    payload = event.payload
    identity = dict(source_batch_id=payload["source_batch_id"], correction_event_id=payload["event_id"],
                    payload_sha256=payload["payload_sha256"], stage="manager")
    journal.append(dict(identity, status="fusion_prepared", envelope=payload))
    failures = []
    for name, publish in (("fused_envelope", lambda: publish_envelope(event.json)),
                          ("manager_decision", publish_decision),
                          *((f"compatibility_{i}", callback) for i, callback in enumerate(publish_compatibility))):
        journal.append(dict(identity, status="publication_attempted", surface=name))
        try:
            publish()
        except Exception as exc:
            failures.append((name, exc))
            journal.append(dict(identity, status="publication_error", surface=name, reason=str(exc)))
        else:
            journal.append(dict(identity, status="publication_returned", surface=name))
    if failures:
        raise RuntimeError("fusion publication failed: " + "; ".join(f"{name}: {exc}" for name, exc in failures)) from failures[0][1]
