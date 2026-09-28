"""P1 gate/label/firewall unit tests (Gate 1).

Every FailureReason must be reachable, the earliest failing gate must win, labels must
satisfy usable = detection AND quality, and the firewall must reject GT.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from reliability.contracts import LeakageError
from reliability.observation_gates import (
    UsableObservationGateConfig,
    evaluate_observation_opportunity,
)
from reliability.observation_opportunity import (
    FailureReason,
    ObservationOpportunity,
    OBS_SCHEMA_VERSION,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_YAML = REPO_ROOT / "config" / "sensor_gate.yaml"


def _cfg(**overrides) -> UsableObservationGateConfig:
    base = dict(image_width_px=640, image_height_px=360)
    base.update(overrides)
    return UsableObservationGateConfig(**base)


def _usable_raw() -> dict:
    """A raw record that passes every gate (USABLE)."""
    return {
        "timestamp": 1.0,
        "run_id": "run_A",
        "route_id": "route_1",
        "camera_id": "camera_A",
        "state_x": 3.0,
        "state_y": 2.0,
        "state_yaw": 0.1,
        "state_source": "BELIEF",
        "frame_expected": True,
        "frame_received": True,
        "frame_age_ms": 40.0,
        "detection_received": True,
        "detector_class": "robot",
        "detector_confidence": 0.8,
        "bbox_xmin": 300.0,
        "bbox_ymin": 150.0,
        "bbox_xmax": 340.0,
        "bbox_ymax": 220.0,
        "projection_attempted": True,
        "projection_valid": True,
        "association_attempted": True,
        "association_valid": True,
        "tracking_available": True,
        "tracking_valid": True,
        "accepted_by_localizer": True,
    }


# --- happy path -----------------------------------------------------------------------

def test_usable_record_all_pass():
    row = evaluate_observation_opportunity(_usable_raw(), _cfg())
    assert (row.detection_label, row.quality_label, row.usable_label) == (1, 1, 1)
    assert row.failure_reason == FailureReason.USABLE.value
    assert row.schema_version == OBS_SCHEMA_VERSION
    # bbox derived fields
    assert row.bbox_width == 40.0 and row.bbox_height == 70.0
    assert row.bbox_area == 40.0 * 70.0
    # selected pixel defaults to bottom-centre of the box
    assert row.selected_pixel_u == 320.0 and row.selected_pixel_v == 220.0


# --- each FailureReason reachable in isolation ---------------------------------------

@pytest.mark.parametrize(
    "mutate, expected_reason, expected_det, expected_qual",
    [
        (lambda r: r.update(frame_expected=True, frame_received=False), FailureReason.NO_FRAME, 0, 0),
        (lambda r: r.update(frame_age_ms=900.0), FailureReason.STALE_FRAME, 0, 0),
        (lambda r: r.update(detection_received=False), FailureReason.NO_DETECTION, 0, 0),
        (lambda r: r.update(detector_confidence=0.1), FailureReason.LOW_CONFIDENCE, 1, 0),
        (lambda r: r.update(projection_valid=False), FailureReason.INVALID_PROJECTION, 1, 0),
        (lambda r: r.update(association_valid=False), FailureReason.INVALID_ASSOCIATION, 1, 0),
        (lambda r: r.update(accepted_by_localizer=False), FailureReason.REJECTED_BY_LOCALIZER, 1, 0),
        # bbox at the top-left corner -> selected bottom-centre near edge
        (lambda r: r.update(bbox_xmin=0.0, bbox_ymin=0.0, bbox_xmax=2.0, bbox_ymax=1.0),
         FailureReason.CLIPPED_OR_EDGE, 1, 0),
    ],
)
def test_single_gate_failures(mutate, expected_reason, expected_det, expected_qual):
    raw = _usable_raw()
    mutate(raw)
    row = evaluate_observation_opportunity(raw, _cfg())
    assert row.failure_reason == expected_reason.value
    assert row.detection_label == expected_det
    assert row.quality_label == expected_qual
    assert row.usable_label == (expected_det & expected_qual)


def test_wrong_class_reachable():
    cfg = _cfg(check_class=True, expected_class="robot")
    raw = _usable_raw()
    raw["detector_class"] = "person"
    row = evaluate_observation_opportunity(raw, cfg)
    assert row.failure_reason == FailureReason.WRONG_CLASS_OR_ID.value
    assert (row.detection_label, row.quality_label) == (1, 0)


def test_invalid_track_reachable():
    cfg = _cfg(require_track=True)
    raw = _usable_raw()
    raw["tracking_valid"] = False
    row = evaluate_observation_opportunity(raw, cfg)
    assert row.failure_reason == FailureReason.INVALID_TRACK.value
    assert (row.detection_label, row.quality_label) == (1, 0)


# --- priority ordering: earliest failing gate wins -----------------------------------

def test_priority_no_detection_beats_low_confidence():
    raw = _usable_raw()
    raw["detection_received"] = False
    raw["detector_confidence"] = 0.01
    raw["projection_valid"] = False
    row = evaluate_observation_opportunity(raw, _cfg())
    assert row.failure_reason == FailureReason.NO_DETECTION.value


def test_priority_low_confidence_beats_projection():
    raw = _usable_raw()
    raw["detector_confidence"] = 0.05
    raw["projection_valid"] = False
    row = evaluate_observation_opportunity(raw, _cfg())
    assert row.failure_reason == FailureReason.LOW_CONFIDENCE.value


def test_priority_projection_beats_association():
    raw = _usable_raw()
    raw["projection_valid"] = False
    raw["association_valid"] = False
    row = evaluate_observation_opportunity(raw, _cfg())
    assert row.failure_reason == FailureReason.INVALID_PROJECTION.value


# --- A5/A6 ablation: confidence off vs on --------------------------------------------

def test_confidence_off_ablation_promotes_low_confidence_to_usable():
    raw = _usable_raw()
    raw["detector_confidence"] = 0.05  # below 0.25
    on = evaluate_observation_opportunity(raw, _cfg(check_confidence=True))
    off = evaluate_observation_opportunity(raw, _cfg(check_confidence=False))
    assert on.failure_reason == FailureReason.LOW_CONFIDENCE.value
    assert off.failure_reason == FailureReason.USABLE.value and off.usable_label == 1


# --- UNKNOWN when an enabled check lacks its input -----------------------------------

def test_unknown_when_enabled_check_missing_input():
    raw = _usable_raw()
    del raw["frame_age_ms"]  # freshness enabled but no age
    row = evaluate_observation_opportunity(raw, _cfg(check_frame_freshness=True))
    assert row.failure_reason == FailureReason.UNKNOWN.value

    raw2 = _usable_raw()
    raw2["detector_confidence"] = None  # detection exists but confidence unknown
    row2 = evaluate_observation_opportunity(raw2, _cfg(check_confidence=True))
    assert row2.failure_reason == FailureReason.UNKNOWN.value
    assert row2.detection_label == 1 and row2.usable_label == 0


# --- firewall: GT never enters --------------------------------------------------------

def test_gt_state_source_rejected():
    raw = _usable_raw()
    raw["state_source"] = "GT"
    with pytest.raises(LeakageError):
        evaluate_observation_opportunity(raw, _cfg())


def test_evaluation_only_key_rejected():
    raw = _usable_raw()
    raw["gt_x"] = 3.01
    with pytest.raises(LeakageError):
        evaluate_observation_opportunity(raw, _cfg())


def test_ground_truth_localization_error_rejected():
    raw = _usable_raw()
    raw["ground_truth_localization_error_m"] = 0.2
    with pytest.raises(LeakageError):
        evaluate_observation_opportunity(raw, _cfg())


# --- contract invariants --------------------------------------------------------------

def test_contract_rejects_inconsistent_labels():
    import dataclasses

    row = evaluate_observation_opportunity(_usable_raw(), _cfg())
    with pytest.raises(Exception):
        dataclasses.replace(row, usable_label=0)  # USABLE reason but usable=0


def test_roundtrip_from_dict():
    row = evaluate_observation_opportunity(_usable_raw(), _cfg())
    again = ObservationOpportunity.from_dict(row.to_dict())
    assert again.to_dict() == row.to_dict()


# --- config -------------------------------------------------------------------------

def test_frozen_yaml_loads_and_hashes():
    cfg = UsableObservationGateConfig.from_yaml(str(GATE_YAML))
    assert cfg.confidence_threshold == 0.25
    assert cfg.image_width_px == 1280 and cfg.image_height_px == 720
    # the fitted gate: confidence and projection only (no edge, size or class check)
    assert (cfg.check_edge_clip, cfg.check_box_size, cfg.check_class) == (False, False, False)
    assert len(cfg.config_hash()) == 16
    # hash is stable and sensitive to a threshold change
    assert cfg.config_hash() == UsableObservationGateConfig.from_yaml(str(GATE_YAML)).config_hash()
    changed = UsableObservationGateConfig.from_dict({**cfg.to_dict(), "confidence_threshold": 0.05})
    assert changed.config_hash() != cfg.config_hash()


def test_every_failure_reason_has_a_producing_case():
    """Guard: the enum and the reachable set stay in sync (except UNKNOWN/edge covered above)."""
    produced = set()
    cfg = _cfg(check_class=True, expected_class="robot", require_track=True,
               check_box_size=True, min_bbox_width_px=16.0, min_bbox_height_px=16.0)
    cases = [
        {"frame_expected": True, "frame_received": False},
        {"frame_age_ms": 9e9},
        {"detection_received": False},
        {"detector_confidence": 0.01},
        {"detector_class": "person"},
        {"projection_valid": False},
        {"association_valid": False},
        {"tracking_valid": False},
        {"bbox_xmin": 300.0, "bbox_ymin": 290.0, "bbox_xmax": 340.0, "bbox_ymax": 358.0},  # bottom centre 2 px from the edge
        {"bbox_xmin": 300.0, "bbox_ymin": 150.0, "bbox_xmax": 302.0, "bbox_ymax": 151.0},  # too small
        {"accepted_by_localizer": False},
        {},  # usable
    ]
    for patch in cases:
        raw = _usable_raw()
        raw.update(patch)
        produced.add(evaluate_observation_opportunity(raw, cfg).failure_reason)
    expected = {r.value for r in FailureReason} - {FailureReason.UNKNOWN.value}
    assert expected <= produced
