"""Leakage-firewall helpers for reliability-model data paths."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import yaml

from reliability.contracts import EVALUATION_ONLY_FIELD_NAMES, LeakageError
from unav_common.config import parse_bool


DEFAULT_FIREWALL_CONFIG = (
    Path(__file__).resolve().parents[1] / "config" / "leakage_firewall.yaml"
)


def load_firewall_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path or DEFAULT_FIREWALL_CONFIG).expanduser().resolve()
    with cfg_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Firewall config must be a mapping: {cfg_path}")
    return payload


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _regex_hits(patterns: Iterable[str], text: str) -> list[str]:
    hits = []
    for pattern in patterns:
        if re.search(str(pattern), text, flags=re.IGNORECASE):
            hits.append(str(pattern))
    return hits


def _token_hits(tokens: Iterable[str], text: str) -> list[str]:
    lowered = str(text).lower()
    return [str(token) for token in tokens if str(token).lower() in lowered]


def _as_config_bool(value: Any) -> bool:
    """Interpret common launch/YAML boolean spellings without executing config."""
    return parse_bool(value, field_name="firewall flag")


def validate_feature_columns(columns: Iterable[str], cfg: Mapping[str, Any] | None = None) -> None:
    """Fail if model feature columns include evaluation-only evidence."""

    cfg = dict(cfg or load_firewall_config())
    exact = set(str(item).lower() for item in cfg.get("evaluation_only_columns", []))
    exact.update(str(item).lower() for item in EVALUATION_ONLY_FIELD_NAMES)
    patterns = [str(item) for item in cfg.get("evaluation_only_patterns", [])]
    bad: list[str] = []
    for column in columns:
        name = str(column).strip()
        lowered = name.lower()
        if lowered in exact or _regex_hits(patterns, lowered):
            bad.append(name)
    if bad:
        raise LeakageError("Model features include evaluation-only columns: " + ", ".join(sorted(set(bad))))


def validate_training_loader_sources(
    sources: Iterable[str],
    *,
    context: str = "normal_training_loader",
    cfg: Mapping[str, Any] | None = None,
) -> None:
    """Fail if a normal training loader reads GT/oracle topics or paths."""

    cfg = dict(cfg or load_firewall_config())
    allowed = set(str(item).lower() for item in cfg.get("allowed_evaluation_contexts", []))
    if str(context).lower() in allowed:
        return
    source_cfg = dict(cfg.get("forbidden_training_sources", {}) or {})
    topic_patterns = [str(item) for item in source_cfg.get("topics", [])]
    path_tokens = [str(item) for item in source_cfg.get("path_tokens", [])]
    bad: list[str] = []
    for source in sources:
        text = str(source).strip()
        if _regex_hits(topic_patterns, text) or _token_hits(path_tokens, text):
            bad.append(text)
    if bad:
        raise LeakageError(
            f"{context} opens ground-truth/oracle sources: " + ", ".join(sorted(set(bad)))
        )


def validate_planner_facing_imports(
    import_targets: Iterable[str],
    cfg: Mapping[str, Any] | None = None,
) -> None:
    """Fail if planner-facing reliability providers import evaluation-only modules."""

    cfg = dict(cfg or load_firewall_config())
    forbidden = [str(item).lower() for item in cfg.get("planner_facing_forbidden_imports", [])]
    bad: list[str] = []
    for target in import_targets:
        lowered = str(target).strip().lower()
        if any(item in lowered for item in forbidden):
            bad.append(str(target))
    if bad:
        raise LeakageError(
            "Planner-facing provider imports evaluation-only modules: "
            + ", ".join(sorted(set(bad)))
        )


def scan_import_targets(paths: Iterable[str | Path]) -> list[str]:
    """Collect import targets from Python files without executing them."""

    imports: list[str] = []
    import_re = re.compile(r"^\s*(?:from\s+([A-Za-z0-9_\.]+)\s+import|import\s+([A-Za-z0-9_\.]+))")
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file() or path.suffix != ".py":
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            match = import_re.match(line)
            if not match:
                continue
            imports.append(match.group(1) or match.group(2) or "")
    return [item for item in imports if item]


def validate_planner_facing_import_paths(
    paths: Iterable[str | Path],
    cfg: Mapping[str, Any] | None = None,
) -> None:
    validate_planner_facing_imports(scan_import_targets(paths), cfg=cfg)


def _walk_config_items(payload: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path, value
            yield from _walk_config_items(value, prefix=path)
    elif isinstance(payload, list):
        for idx, value in enumerate(payload):
            yield from _walk_config_items(value, prefix=f"{prefix}[{idx}]")


def validate_config_sources(
    payload: Mapping[str, Any],
    *,
    context: str = "normal_runtime",
    cfg: Mapping[str, Any] | None = None,
) -> None:
    """Fail if normal runtime config names GT as a state or reliability source."""

    cfg = dict(cfg or load_firewall_config())
    allowed = set(str(item).lower() for item in cfg.get("allowed_evaluation_contexts", []))
    if str(context).lower() in allowed:
        return
    normal_cfg = dict(cfg.get("normal_runtime_forbidden_config_values", {}) or {})
    source_tokens = [str(item) for item in normal_cfg.get("state_or_reliability_source_tokens", [])]
    forbidden_true_flags = set(str(item).lower() for item in normal_cfg.get("forbidden_true_flags", []))

    bad: list[str] = []
    for path, value in _walk_config_items(payload):
        key = path.split(".")[-1].split("[")[0].lower()
        if key in forbidden_true_flags and _as_config_bool(value):
            bad.append(f"{path}={value!r}")
        if ("state_source" in key or "reliability_source" in key) and isinstance(value, str):
            if _token_hits(source_tokens, value):
                bad.append(f"{path}={value!r}")
    if bad:
        raise LeakageError("Normal runtime config enables GT reliability/state source: " + ", ".join(bad))


def assert_operational_feature_table(columns: Iterable[str], sources: Iterable[str]) -> None:
    """Convenience check for early Phase 1 exporters/loaders."""

    validate_feature_columns(columns)
    validate_training_loader_sources(sources)
