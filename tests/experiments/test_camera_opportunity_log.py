import io
import json
from experiments.core.camera_opportunity_log import CameraOpportunityLog, JsonlDeliveryLog


def observation(hit=False, stamp=1., batch="batch1", camera="camera_A"):
    return json.dumps(dict(schema_version="phase0.v1", camera_id=camera,
                           timestamp_s=stamp, source_batch_id=batch,
                           detection_valid=hit, bbox_xyxy=[1, 2, 3, 4] if hit else None))


def test_miss_and_duplicate_are_retained_and_distinguished():
    stream = io.StringIO(); log = CameraOpportunityLog(stream)
    first = log.append("camera_A", observation(), 1.2)
    repeat = log.append("camera_A", observation(), 1.4)
    other = log.append("camera_A", observation(True, 2., "batch2"), 2.3)
    assert first["observation"]["detection_valid"] is False
    assert not first["duplicate"] and repeat["duplicate"] and not other["duplicate"]
    assert len(stream.getvalue().splitlines()) == 3
    assert first["observation"]["timestamp_s"] == 1.
    assert first["receive_stamp_s"] == 1.2
    assert first["raw_payload"] == observation()


def test_malformed_and_wrong_camera_deliveries_remain_auditable():
    stream = io.StringIO(); log = CameraOpportunityLog(stream)
    for payload in ("bad JSON", observation(camera="camera_B"), observation(batch="")):
        result = log.append("camera_A", payload, 3.)
        assert not result["valid_contract"] and result["reason"]
        assert result["raw_payload"] == payload
    assert len(stream.getvalue().splitlines()) == 3


def test_unknown_schema_and_nested_nonfinite_are_retained_without_claiming_identity():
    stream = io.StringIO(); log = CameraOpportunityLog(stream)
    future = json.loads(observation()); future["schema_version"] = "future"
    nonfinite = json.loads(observation()); nonfinite["diagnostics"] = {"score": float("nan")}
    for payload in (json.dumps(future), json.dumps(nonfinite)):
        result = log.append("camera_A", payload, 3.)
        assert not result["valid_contract"] and result["raw_payload"] == payload
    assert log.rows == 2 and not log.seen
    assert len(stream.getvalue().splitlines()) == 2


def test_conflicting_same_camera_batch_is_retained_and_classified():
    stream = io.StringIO(); log = CameraOpportunityLog(stream)
    first = log.append("camera_A", observation(stamp=1.), 2.)
    conflict = log.append("camera_A", observation(stamp=1.1), 2.1)
    assert not first["duplicate"]
    assert conflict["duplicate"] and conflict["conflicting_duplicate"]
    assert len(stream.getvalue().splitlines()) == 2


def test_failed_write_does_not_claim_delivery_or_identity():
    class Broken:
        def write(self, _value):
            raise OSError("disk full")

        def flush(self):
            raise AssertionError("unreachable")

    log = CameraOpportunityLog(Broken())
    try:
        log.append("camera_A", observation(), 2.)
    except OSError:
        pass
    else:
        raise AssertionError("write failure was hidden")
    assert log.rows == 0 and not log.seen


def test_generic_delivery_log_retains_malformed_duplicate_and_conflict():
    stream = io.StringIO(); log = JsonlDeliveryLog(stream)
    first = json.dumps({"event_id": "e1", "value": 1})
    original = log.append("/event", first, 1.)
    duplicate = log.append("/event", first, 1.1)
    conflict = log.append("/event", json.dumps({"event_id": "e1", "value": 2}), 1.2)
    malformed = log.append("/event", "{broken", 1.3)
    assert not original["duplicate_delivery"]
    assert duplicate["duplicate_delivery"] and not duplicate["conflicting_delivery"]
    assert conflict["duplicate_delivery"] and conflict["conflicting_delivery"]
    assert not malformed["valid_json_object"] and malformed["raw_payload"] == "{broken"
    assert len(stream.getvalue().splitlines()) == 4
