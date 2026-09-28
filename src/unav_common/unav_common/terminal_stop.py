"""Identity-bearing terminal stop handshake shared by runtime components.

The experiment logger owns the terminal decision.  It publishes one immutable
request; the planner, camera manager and detector acknowledge the same request
only after they have entered their terminal state.  The acknowledgement is an
operational boundary, not a claim that a command was physically applied.
"""
from __future__ import annotations

from dataclasses import dataclass
import json


TERMINAL_STOP_REQUEST_TOPIC = "/experiment/terminal_stop_request"
TERMINAL_STOP_ACK_TOPIC = "/experiment/terminal_stop_ack"
TERMINAL_STOP_REQUEST_SCHEMA = "terminal_stop_request.v1"
TERMINAL_STOP_ACK_SCHEMA = "terminal_stop_ack.v1"
TERMINAL_COMPONENTS = frozenset(("planner", "camera_manager", "detector"))
TERMINAL_ACK_STATUS_BY_COMPONENT = {
    "planner": "stop_command_published",
    "camera_manager": "correction_stream_quiescent",
    "detector": "outcome_stream_quiescent",
}


def _nonempty(value, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be a nonempty string without surrounding whitespace")
    return value


def _nonnegative_int(value, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class TerminalStopRequest:
    request_id: str
    run_id: str
    reason: str
    decision_stamp_ns: int

    def __post_init__(self):
        _nonempty(self.request_id, "request_id")
        _nonempty(self.run_id, "run_id")
        _nonempty(self.reason, "reason")
        _nonnegative_int(self.decision_stamp_ns, "decision_stamp_ns")


@dataclass(frozen=True)
class TerminalStopAck:
    request_id: str
    component: str
    status: str
    acknowledgement_stamp_ns: int
    detail: str = ""

    def __post_init__(self):
        _nonempty(self.request_id, "request_id")
        if self.component not in TERMINAL_COMPONENTS:
            raise ValueError(f"unsupported terminal component: {self.component!r}")
        _nonempty(self.status, "status")
        expected_status = TERMINAL_ACK_STATUS_BY_COMPONENT[self.component]
        if self.status != expected_status:
            raise ValueError(
                f"terminal component {self.component!r} must report "
                f"status {expected_status!r}, got {self.status!r}"
            )
        _nonnegative_int(self.acknowledgement_stamp_ns, "acknowledgement_stamp_ns")
        if not isinstance(self.detail, str):
            raise ValueError("detail must be a string")


def terminal_stop_request_to_json(request: TerminalStopRequest) -> str:
    if not isinstance(request, TerminalStopRequest):
        raise TypeError("expected TerminalStopRequest")
    return json.dumps(
        {
            "schema": TERMINAL_STOP_REQUEST_SCHEMA,
            "request_id": request.request_id,
            "run_id": request.run_id,
            "reason": request.reason,
            "decision_stamp_ns": request.decision_stamp_ns,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def terminal_stop_request_from_json(text: str) -> TerminalStopRequest:
    try:
        value = json.loads(text)
        if not isinstance(value, dict) or value.get("schema") != TERMINAL_STOP_REQUEST_SCHEMA:
            raise ValueError("unsupported terminal stop request schema")
        return TerminalStopRequest(
            request_id=value["request_id"],
            run_id=value["run_id"],
            reason=value["reason"],
            decision_stamp_ns=value["decision_stamp_ns"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid terminal stop request: {exc}") from exc


def terminal_stop_ack_to_json(ack: TerminalStopAck) -> str:
    if not isinstance(ack, TerminalStopAck):
        raise TypeError("expected TerminalStopAck")
    return json.dumps(
        {
            "schema": TERMINAL_STOP_ACK_SCHEMA,
            "request_id": ack.request_id,
            "component": ack.component,
            "status": ack.status,
            "acknowledgement_stamp_ns": ack.acknowledgement_stamp_ns,
            "detail": ack.detail,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def terminal_stop_ack_from_json(text: str) -> TerminalStopAck:
    try:
        value = json.loads(text)
        if not isinstance(value, dict) or value.get("schema") != TERMINAL_STOP_ACK_SCHEMA:
            raise ValueError("unsupported terminal stop acknowledgement schema")
        return TerminalStopAck(
            request_id=value["request_id"],
            component=value["component"],
            status=value["status"],
            acknowledgement_stamp_ns=value["acknowledgement_stamp_ns"],
            detail=value.get("detail", ""),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid terminal stop acknowledgement: {exc}") from exc
