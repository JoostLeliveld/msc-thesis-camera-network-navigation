import json

import pytest

from unav_common.terminal_stop import (
    TERMINAL_STOP_ACK_SCHEMA,
    TerminalStopAck,
    TerminalStopRequest,
    terminal_stop_ack_from_json,
    terminal_stop_ack_to_json,
    terminal_stop_request_from_json,
    terminal_stop_request_to_json,
)


def test_terminal_stop_request_round_trip_preserves_identity():
    request = TerminalStopRequest("logger:run:terminal:12", "run", "goal_reached", 12)
    assert terminal_stop_request_from_json(
        terminal_stop_request_to_json(request)
    ) == request


def test_terminal_stop_ack_round_trip_preserves_component_status():
    ack = TerminalStopAck(
        "logger:run:terminal:12", "planner", "stop_command_published", 14,
        "command_stop_generation=3",
    )
    assert terminal_stop_ack_from_json(terminal_stop_ack_to_json(ack)) == ack


@pytest.mark.parametrize(
    "mutation",
    [
        {"component": "ground_truth"},
        {"status": "merely_received"},
        {"request_id": ""},
        {"acknowledgement_stamp_ns": -1},
        {"acknowledgement_stamp_ns": True},
        {"schema": "terminal_stop_ack.v0"},
    ],
)
def test_terminal_stop_ack_rejects_unowned_or_malformed_claims(mutation):
    payload = {
        "schema": TERMINAL_STOP_ACK_SCHEMA,
        "request_id": "logger:run:terminal:12",
        "component": "detector",
        "status": "outcome_stream_quiescent",
        "acknowledgement_stamp_ns": 15,
        "detail": "durable_session_stopped",
    }
    payload.update(mutation)
    with pytest.raises(ValueError):
        terminal_stop_ack_from_json(json.dumps(payload))
