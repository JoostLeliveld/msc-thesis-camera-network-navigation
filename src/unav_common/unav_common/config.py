"""Small, ROS-free configuration contracts shared across runtime packages."""

from __future__ import annotations

import math


LOCAL_CONTROLLER_TYPES = frozenset({
    "ff_fb",
    "hyst_damp",
    "pure_pursuit",
    "turn_then_go",
    "turn_then_go_recovery",
})


def parse_bool(value: object, *, field_name: str = "value") -> bool:
    """Parse a boolean without Python's truthiness surprises."""
    if not isinstance(value, (str, bytes)) and hasattr(value, "item"):
        scalar = value.item()
        if scalar is not value:
            return parse_bool(scalar, field_name=field_name)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isfinite(numeric) and numeric in (0.0, 1.0):
            return bool(numeric)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean, got {value!r}")


def local_controller_type(value: object) -> str:
    """Return a supported controller name, rejecting typos and empty values."""
    normalized = str(value).strip().lower()
    if normalized not in LOCAL_CONTROLLER_TYPES:
        supported = ", ".join(sorted(LOCAL_CONTROLLER_TYPES))
        raise ValueError(
            f"unsupported local_controller_type {normalized!r}; expected one of: {supported}"
        )
    return normalized


def parse_bev_affine_calibration(value: object) -> tuple[float, ...] | None:
    """Parse the optional six coefficient BEV affine calibration.

    Empty input explicitly selects the constant offset model. Any nonempty input
    configures the affine model, so malformed coefficients are errors rather
    than permission to silently change calibration models.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    parts = [part.strip() for part in text.replace(";", ",").split(",")]
    if len(parts) != 6 or any(not part for part in parts):
        raise ValueError(
            "bev_affine_calibration must contain exactly 6 comma-separated coefficients"
        )
    try:
        coefficients = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ValueError("bev_affine_calibration coefficients must be numeric") from exc
    if not all(math.isfinite(coefficient) for coefficient in coefficients):
        raise ValueError("bev_affine_calibration coefficients must be finite")
    return coefficients
