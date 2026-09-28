#!/usr/bin/env python3
"""Run a visibility-comparison campaign from a locked config file.

Usage:
    python campaign_runner.py --config scripts/visibility_comparison/warehouse_visibility_campaign.yaml [--dry-run] [--resume]

Each run result is written immediately to campaign_log.json so the campaign
can be interrupted and resumed with --resume (already-completed runs are skipped).
Completion reasons are exactly: goal_reached, timeout_after_first_cmd, collision.
"""

from __future__ import annotations

import argparse
import collections
import csv
import fcntl
import json
import math
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
# The directories whose bytes actually execute during a run. Provenance is scoped to
# these so that editing analysis or figure code elsewhere in the checkout does not
# abort a multi-hour campaign; it mirrors what the source snapshot below copies.
EXECUTABLE_SOURCE_PATHS = ('src', 'scripts/visibility_comparison')
LOGS_ROOT = REPO_ROOT / 'logs' / 'visibility_comparison'
UNAV_COMMON_SRC = REPO_ROOT / 'src' / 'unav_common'
if str(UNAV_COMMON_SRC) not in sys.path:
    sys.path.insert(0, str(UNAV_COMMON_SRC))

from unav_common.preselected_route import (  # noqa: E402
    canonicalize_polyline_json,
    route_sha256,
    sha256_file,
    validate_preselected_route,
)
from unav_common.navigation_parameters import validate_navigation_parameters, PARAMETERS
from unav_common.manifest import atomic_write_json, git_provenance  # noqa: E402
from unav_common.config import parse_bool  # noqa: E402
from unav_common.correction_ledger import validate_correction_ledger  # noqa: E402
from unav_common.camera_outcomes import read_journal  # noqa: E402
from unav_common.terminal_stop import (  # noqa: E402
    TERMINAL_ACK_STATUS_BY_COMPONENT,
)

# Map condition ID to planner name (must match ALLOWED_PLANNERS in launch file).
#: Every terminal outcome a correction may have; mirrors the planner and the logger.
#: An outcome outside this set means a correction went unaccounted, which is what
#: invalidates a run -- not a refusal that recorded its reason.
KNOWN_ASSIMILATION_STATUSES = frozenset({
    'accepted', 'accepted_bootstrap', 'reanchored', 'rejected', 'dropped',
})

CONDITION_PLANNER = {
    'global_intact': 'visibility_aware_efe',
    'global_removal': 'visibility_aware_efe',
    'per_camera_intact': 'visibility_aware_efe',
    'per_camera_removal': 'visibility_aware_efe',
    'spatial_intact': 'visibility_aware_efe',
    'spatial_removal': 'visibility_aware_efe',
}

PRESELECTED_ROUTE_KEYS = (
    'preselected_route_json',
    'preselected_route_sha256',
    'preselected_route_source_path',
    'preselected_route_source_sha256',
    'preselected_route_clearance_m',
    'preselected_route_endpoint_tolerance_m',
    'preselected_route_sample_step_m',
)

BOOL_CONFIG_KEYS = frozenset({
    'headless', 'use_rviz', 'reset_world', 'use_command_noise',
    'use_encoder_noise', 'use_odom_for_predict', 'optimizer_multistart',
    'optimizer_multistart_include_direct', 'yolo_use_masks',
    'yolo_use_torchscript', 'yolo_inference_in_callback',
    'require_state_correction_envelope', 'use_pixel_correction',
    'skip_stale_pixel_correction', 'debug_runtime', 'optimizer_warm_start',
    'use_hierarchical', 'global_use_ambiguity', 'local_use_ambiguity',
    'local_use_obs_risk', 'global_optimizer_multistart',
    'local_optimizer_multistart', 'local_use_visibility_model',
    'local_use_belief_nogo_cost', 'local_replan_on_waypoint_change',
    'latency_compensate_plan_handoff', 'use_nogo_cost',
    'use_belief_nogo_cost', 'use_hit_miss_mixture',
    'bridge_camera_b', 'bridge_camera_c',
    'bridge_camera_d', 'multicam_belief', 'manager_require_source_batch_id',
    'manager_use_task_start_as_bootstrap_prior',
    'manager_bootstrap_prior_counts_as_support',
    'initial_belief_from_task_start',
    'manager_commissioned_per_camera_sigma',
    'manager_correction_timestamp_compensation',
    'manager_require_consistency_when_source_available', 'manager_fusion_mode',
    'manager_require_gp_artifacts', 'state_correction_ekf',
    'wait_for_belief_before_first_goal', 'multicam_scheduled',
    'cleanup_sim_stragglers', 'enable_mission', 'nvidia_offload',
    'lockstep',
})

CAMPAIGN_METADATA_KEYS = frozenset({
    'conditions', 'tasks', 'study_title', 'study_comparison', 'cleanup_mode',
    'cleanup_sim_stragglers', 'ros_domain_id_base', 'world_profiles',
    'tasks_yaml', 'gp_artifact', 'route_selection_manifest_path',
    'route_selection_manifest_sha256',
    'removed_camera_id',
    'camera_network_expected_sha256',
    'planning_information_method', 'planning_artifact_schema',
})


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses duplicate mapping keys."""


def _construct_unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(
                f'duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}'
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _load_config(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=_UniqueKeyLoader)
    _validate_config(cfg, path)
    return cfg


def _validate_config(cfg: dict, path: Path) -> None:
    if not isinstance(cfg, dict):
        raise ValueError(f'campaign config {path} must be a mapping')
    unknown_top = set(cfg) - _known_campaign_keys()
    if unknown_top:
        raise ValueError(f'{path}: unknown campaign keys: {sorted(unknown_top)}')
    validate_navigation_parameters(cfg)
    if cfg.get('planning_information_method') is not None:
        if cfg['planning_information_method'] != 'inverse_of_matched_runtime_covariance':
            raise ValueError(f'{path}: unsupported planning information method')
        if cfg.get('planning_artifact_schema') != 'camera_network.matched_covariance_precision.v1':
            raise ValueError(f'{path}: matched covariance method requires its canonical artifact schema')
    # A solving planner must be given the objective that reads the per-arm
    # camera fields. With global_planner_mode: efe and legacy_pixel_chart the
    # planner never queries a camera field, so every arm produces the same
    # route and the campaign measures nothing. That combination invalidated an
    # earlier four-arm campaign, so it is refused rather than warned about.
    if str(cfg.get('global_planner_mode', '')) == 'efe':
        objective = str(cfg.get('camera_network_objective', '') or 'legacy_pixel_chart')
        if objective != 'metric_expected_belief':
            raise ValueError(
                f'{path}: global_planner_mode: efe requires '
                f'camera_network_objective: metric_expected_belief, not {objective!r}. '
                'Only that objective loads the per-arm planner fields; with any other '
                'the arms are identical.'
            )
    cleanup_mode = cfg.get('cleanup_mode', 'isolated')
    if cleanup_mode != 'isolated':
        raise ValueError('campaign execution requires cleanup_mode: isolated')
    if cfg.get('cleanup_sim_stragglers', False):
        raise ValueError('isolated cleanup forbids global straggler cleanup')
    for key in ('world', 'launch_file', 'conditions', 'tasks',
                'yolo_model', 'horizon', 'dt', 'goal_success_radius',
                'run_timeout_after_first_cmd_s'):
        if key not in cfg:
            raise RuntimeError(f"Campaign config {path} is missing required key: '{key}'")
    for key in ('yolo_model',):
        if str(cfg[key]).startswith('[FILL'):
            raise RuntimeError(
                f"Campaign config {path} has unfilled placeholder for '{key}': {cfg[key]!r}\n"
                f"Fill in all [FILL] entries before running."
            )
    for key in ('observation_risk_scale', 'ambiguity_term_scale'):
        if key in cfg and str(cfg[key]).startswith('[FILL'):
            raise RuntimeError(
                f"Campaign config {path} has unfilled placeholder for '{key}': {cfg[key]!r}\n"
                f"Verify the lambda mapping against the planner source and fill it in."
            )
    if not isinstance(cfg['conditions'], dict) or not cfg['conditions']:
        raise ValueError(f'{path}: conditions must be a nonempty mapping')
    if not isinstance(cfg['tasks'], dict) or not cfg['tasks']:
        raise ValueError(f'{path}: tasks must be a nonempty mapping')
    route_manifest = str(cfg.get('route_selection_manifest_path', '') or '').strip()
    route_manifest_sha = str(
        cfg.get('route_selection_manifest_sha256', '') or ''
    ).strip().lower()
    if bool(route_manifest) != bool(route_manifest_sha):
        raise ValueError(
            f'{path}: route-selection manifest path and SHA-256 must be supplied together'
        )
    if route_manifest:
        route_manifest_path = _resolve_repo_path(route_manifest, strict=True)
        actual = sha256_file(route_manifest_path)
        if actual != route_manifest_sha:
            raise ValueError(
                f'{path}: route-selection manifest SHA-256 mismatch: '
                f'expected {route_manifest_sha}, got {actual}'
            )
    for condition_id in cfg['conditions']:
        if condition_id not in CONDITION_PLANNER:
            raise RuntimeError(
                f"Campaign config {path} uses unsupported active condition '{condition_id}'. "
                f"Allowed conditions are: {', '.join(CONDITION_PLANNER)}"
            )
        condition_cfg = cfg['conditions'].get(condition_id, {}) or {}
        if not isinstance(condition_cfg, dict):
            raise RuntimeError(
                f"Condition '{condition_id}' in {path} must be a mapping"
            )
        declared_planner = condition_cfg.get('planner')
        expected_planner = CONDITION_PLANNER[condition_id]
        if declared_planner is not None and str(declared_planner) != expected_planner:
            raise RuntimeError(
                f"Condition '{condition_id}' declares planner {declared_planner!r}, "
                f"but the active runner contract requires {expected_planner!r}"
            )
        unknown = set(condition_cfg) - (_known_campaign_keys() | {'label', 'planner'})
        if unknown:
            raise ValueError(
                f'{path}: condition {condition_id!r} has unknown keys: {sorted(unknown)}'
            )
        _validate_explicit_scalar_types(condition_cfg, f'{path}: condition {condition_id}')
        network_path = condition_cfg.get('camera_network_artifact_path')
        network_sha = condition_cfg.get('camera_network_expected_sha256')
        if network_path and network_sha:
            actual_network_sha = sha256_file(
                _resolve_repo_path(str(network_path), strict=True)
            )
            if str(network_sha) != actual_network_sha:
                raise ValueError(
                    f'{path}: condition {condition_id!r} camera-network SHA-256 '
                    f'mismatch: expected {network_sha}, got {actual_network_sha}'
                )
    normalized_cells = set()
    for task_name, task_cfg in cfg['tasks'].items():
        if not isinstance(task_name, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', task_name):
            raise ValueError(f'{path}: invalid task name {task_name!r}')
        if not isinstance(task_cfg, dict):
            raise ValueError(f'{path}: task {task_name!r} must be a mapping')
        unknown = set(task_cfg) - (_known_campaign_keys() | {
            'conditions', 'seeds', 'preselected_routes', 'condition_overrides',
        })
        if unknown:
            raise ValueError(
                f'{path}: task {task_name!r} has unknown keys: {sorted(unknown)}'
            )
        conditions = task_cfg.get('conditions')
        seeds = task_cfg.get('seeds')
        if not isinstance(conditions, list) or not conditions:
            raise ValueError(f'{path}: task {task_name!r} conditions must be a nonempty list')
        if len(set(conditions)) != len(conditions):
            raise ValueError(f'{path}: task {task_name!r} has duplicate conditions')
        if not isinstance(seeds, list) or not seeds:
            raise ValueError(f'{path}: task {task_name!r} seeds must be a nonempty list')
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
            raise ValueError(f'{path}: task {task_name!r} seeds must be integers')
        if len(set(seeds)) != len(seeds):
            raise ValueError(f'{path}: task {task_name!r} has duplicate seeds')
        routes = task_cfg.get('preselected_routes', {}) or {}
        if not isinstance(routes, dict):
            raise ValueError(f'{path}: task {task_name!r} preselected_routes must be a mapping')
        for route_condition, route_cfg in routes.items():
            if route_condition not in conditions:
                raise ValueError(
                    f'{path}: task {task_name!r} has a route for inactive condition '
                    f'{route_condition!r}'
                )
            if not isinstance(route_cfg, dict):
                raise ValueError(
                    f'{path}: task {task_name!r} route {route_condition!r} must be a mapping'
                )
            unknown_route = set(route_cfg) - set(PRESELECTED_ROUTE_KEYS)
            if unknown_route:
                raise ValueError(
                    f'{path}: task {task_name!r} route {route_condition!r} has '
                    f'unknown keys: {sorted(unknown_route)}'
                )
            _validate_explicit_scalar_types(
                route_cfg, f'{path}: task {task_name}/{route_condition} route'
            )
        overrides = task_cfg.get('condition_overrides', {}) or {}
        if not isinstance(overrides, dict):
            raise ValueError(f'{path}: task {task_name!r} condition_overrides must be a mapping')
        for override_condition, override_cfg in overrides.items():
            if override_condition not in conditions:
                raise ValueError(
                    f'{path}: task {task_name!r} overrides inactive condition '
                    f'{override_condition!r}'
                )
            if not isinstance(override_cfg, dict):
                raise ValueError(
                    f'{path}: task {task_name!r} condition override '
                    f'{override_condition!r} must be a mapping'
                )
            unknown_override = set(override_cfg) - _known_campaign_keys()
            if unknown_override:
                raise ValueError(
                    f'{path}: task {task_name!r} condition override '
                    f'{override_condition!r} has unknown keys: {sorted(unknown_override)}'
                )
            _validate_explicit_scalar_types(
                override_cfg, f'{path}: task {task_name}/{override_condition} override'
            )
        _validate_explicit_scalar_types(task_cfg, f'{path}: task {task_name}')
        for condition_id in task_cfg.get('conditions', []):
            if condition_id not in CONDITION_PLANNER:
                raise RuntimeError(
                    f"Task '{task_name}' in {path} uses unsupported active condition "
                    f"'{condition_id}'. Allowed conditions are: {', '.join(CONDITION_PLANNER)}"
                )
            if condition_id not in cfg['conditions']:
                raise ValueError(
                    f'{path}: task {task_name!r} references undeclared condition '
                    f'{condition_id!r}'
                )
            for seed in seeds:
                cell = (task_name, condition_id, seed)
                if cell in normalized_cells:
                    raise ValueError(f'{path}: duplicate campaign cell {cell!r}')
                normalized_cells.add(cell)

    _validate_explicit_scalar_types(cfg, str(path))

    active_cells = [
        (task_name, condition_id)
        for task_name, task_cfg in cfg['tasks'].items()
        for condition_id in task_cfg.get('conditions', [])
    ]
    for task_name, condition_id in active_cells:
        _validate_visibility_runtime_bundle(cfg, path, task_name, condition_id)
        _validate_perception_runtime_bundle(cfg, path, task_name, condition_id)
    for task_name, condition_id in active_cells:
        layers = (cfg, cfg['tasks'][task_name], cfg['conditions'][condition_id] or {},
                  _route_overrides(cfg, task_name, condition_id))
        explicit_keys = set().union(*(layer.keys() for layer in layers))
        try:
            validate_navigation_parameters({
                key: _effective_value(cfg, task_name, condition_id, key)
                for key in PARAMETERS & explicit_keys
            })
        except ValueError as exc:
            raise ValueError(f'{path}: {task_name}/{condition_id}: {exc}') from exc
        operational_timeout = _effective_value(
            cfg, task_name, condition_id, 'operational_belief_timeout_s'
        )
        if operational_timeout is not None:
            try:
                operational_timeout = float(operational_timeout)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f'{path}: {task_name}/{condition_id}: '
                    'operational_belief_timeout_s must be numeric'
                ) from exc
            if not math.isfinite(operational_timeout) or operational_timeout <= 0.0:
                raise ValueError(
                    f'{path}: {task_name}/{condition_id}: '
                    'operational_belief_timeout_s must be finite and positive'
                )
    needs_gp = any(
        CONDITION_PLANNER[condition_id] == 'visibility_aware_efe'
        and not _effective_value(cfg, task_name, condition_id, 'camera_network_artifact_path')
        and str(_effective_value(cfg, task_name, condition_id, 'global_planner_mode') or 'efe')
        != 'preselected_route'
        for task_name, condition_id in active_cells
    )
    if needs_gp:
        if 'gp_artifact' not in cfg or str(cfg.get('gp_artifact', '')).startswith('[FILL'):
            raise RuntimeError(
                f"Campaign config {path} requires a filled gp_artifact for its "
                "visibility-aware global-planner condition"
            )

    for task_name, condition_id in active_cells:
        network = _effective_value(cfg, task_name, condition_id, 'camera_network_artifact_path')
        if network:
            if cfg.get('gp_artifact') or _effective_value(cfg, task_name, condition_id, 'visibility_artifact_path'):
                raise RuntimeError('choose one network artifact or legacy GP artifact for this campaign')
            if CONDITION_PLANNER[condition_id] != 'visibility_aware_efe' or str(
                _effective_value(cfg, task_name, condition_id, 'global_planner_mode') or 'efe'
            ) == 'preselected_route':
                raise RuntimeError('network field requires a solved visibility-aware global plan')
            _resolve_repo_path(str(network), strict=True)

    if any(
        str(_effective_value(cfg, task_name, condition_id, 'global_planner_mode') or '')
        == 'preselected_route'
        for task_name, condition_id in active_cells
    ):
        _validate_preselected_campaign_routes(cfg, path)


def _validate_visibility_runtime_bundle(
    cfg: dict, config_path: Path, task_name: str, condition_id: str
) -> None:
    """Prove that the visibility-residual correction and its matched R deploy together."""
    expected = lambda key: _effective_value(cfg, task_name, condition_id, key)
    observation = str(expected('manager_observation_model') or 'raw_box')
    covariance = str(expected('manager_covariance_profile') or '')
    uses_bundle = observation == 'visibility_patch' or covariance == 'commissioned_visibility_r'
    if not uses_bundle:
        return
    if observation != 'visibility_patch' or covariance != 'commissioned_visibility_r':
        raise ValueError(
            f'{config_path}: {task_name}/{condition_id}: visibility_patch and '
            'commissioned_visibility_r must be selected together'
        )
    declared_path = str(expected('manager_visibility_sensor_model_path') or '').strip()
    if not declared_path:
        raise ValueError(
            f'{config_path}: {task_name}/{condition_id}: commissioned visibility-residual bundle is missing'
        )
    model_path = _resolve_repo_path(declared_path, strict=True)
    model_sha = sha256_file(model_path)
    declared_sha = str(
        expected('manager_visibility_sensor_model_expected_sha256') or '').strip().lower()
    if declared_sha != model_sha:
        raise ValueError(
            f'{config_path}: {task_name}/{condition_id}: commissioned visibility-residual bundle hash mismatch'
        )
    try:
        manifest = json.loads(model_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'{config_path}: malformed commissioned visibility-residual bundle') from exc
    if manifest.get('schema') == 'commissioned_visibility_sensor_model.v2':
        if manifest.get('status') != 'frozen_before_audit' or manifest.get('audit_accessed') is not False:
            raise ValueError(
                f'{config_path}: {task_name}/{condition_id}: canonical visibility bundle is not frozen')
        expected_covariance = {
            'global_intact': 'R0_global_full', 'global_removal': 'R0_global_full',
            'per_camera_intact': 'R1_per_camera_full', 'per_camera_removal': 'R1_per_camera_full',
            'spatial_intact': 'R2_spatial_residual', 'spatial_removal': 'R2_spatial_residual',
        }[condition_id]
        if (manifest.get('mean_model') != 'box_mlp_visibility_residual'
                or manifest.get('runtime_covariance_model') != expected_covariance):
            raise ValueError(
                f'{config_path}: {task_name}/{condition_id}: canonical correction/R pairing mismatch')
        for key in ('correction_base', 'correction_patch', 'covariance_models'):
            entry = manifest.get(key)
            artifact = Path(str((entry or {}).get('path', ''))).expanduser()
            if (not isinstance(entry, dict) or not artifact.is_file()
                    or sha256_file(artifact) != str(entry.get('sha256', ''))):
                raise ValueError(
                    f'{config_path}: {task_name}/{condition_id}: canonical bundle {key} hash mismatch')
        return
    # The deployed bundle and this guard were written in different naming
    # generations of the same model: the artifact records the mean chain as
    # ``M4_visibility_patch_residual`` where the guard was written against
    # ``box_mlp_visibility_residual``. Both name the box-MLP base plus the gated
    # visibility-patch residual. Accept either spelling rather than regenerate
    # the commissioned artifact for a rename; the artifact HASHES below are
    # unchanged and still bind the actual files.
    required_identity = {
        'schema': ('commissioned_visibility_sensor_model.v1',),
        'mean_model': ('box_mlp_visibility_residual', 'M4_visibility_patch_residual'),
        'runtime_covariance_model': ('R4_image_conditioned_scale',),
    }
    for key, accepted in required_identity.items():
        if manifest.get(key) not in accepted:
            raise ValueError(
                f'{config_path}: {task_name}/{condition_id}: visibility-residual bundle {key} mismatch'
            )
    # Same naming-generation difference as the identity block above: the
    # deployed bundle prefixes the two mean-chain artifacts M3_/M4_. Accept
    # either spelling. The sha256 check immediately below is unchanged, so the
    # exact bytes of every artifact are still bound.
    artifacts = {
        'correction_base': ('box_mlp_fit_only.joblib', 'M3_box_mlp_fit_only.joblib'),
        'correction_patch': ('box_mlp_visibility_residual_fit_only.pt',
                             'M4_visibility_patch_fit_only.pt'),
        'parameters': ('commissioned_visibility_parameters.npz',),
    }
    for key, basenames in artifacts.items():
        entry = manifest.get(key)
        if not isinstance(entry, dict) or Path(str(entry.get('path', ''))).name not in basenames:
            raise ValueError(
                f'{config_path}: {task_name}/{condition_id}: visibility-residual bundle {key} is missing'
            )
        artifact = Path(str(entry['path'])).expanduser()
        if not artifact.is_file() or sha256_file(artifact) != str(entry.get('sha256', '')):
            raise ValueError(
                f'{config_path}: {task_name}/{condition_id}: visibility-residual bundle {key} hash mismatch'
            )
    runtime = manifest.get('runtime_query', {})
    detector_rate = float(runtime.get('detector_rate_hz', float('nan')))
    fusion_rate = float(runtime.get('fusion_rate_hz', float('nan')))
    configured_rate = float(expected('manager_decision_rate_hz') or float('nan'))
    if detector_rate != 5.0 or fusion_rate != configured_rate:
        raise ValueError(
            f'{config_path}: {task_name}/{condition_id}: visibility-residual bundle was calibrated for '
            f'{fusion_rate:g} Hz fusion from {detector_rate:g} Hz detections, but campaign '
            f'requests {configured_rate:g} Hz fusion'
        )


def _validate_perception_runtime_bundle(
    cfg: dict, config_path: Path, task_name: str, condition_id: str
) -> None:
    expected = lambda key: _effective_value(cfg, task_name, condition_id, key)
    observation = str(expected('manager_observation_model') or 'raw_box')
    covariance = str(expected('manager_covariance_profile') or '')
    methods = {'global_residual', 'per_camera_residual', 'hierarchical_residual',
               'spatial_residual', 'joint_rgb_gaussian'}
    uses_bundle = observation in methods or covariance == 'commissioned_perception_r'
    if not uses_bundle:
        return
    if observation not in methods or covariance != 'commissioned_perception_r':
        raise ValueError(
            f'{config_path}: {task_name}/{condition_id}: a commissioned perception '
            'method and commissioned_perception_r must be selected together'
        )
    declared_path = str(expected('manager_perception_sensor_model_path') or '').strip()
    if not declared_path:
        raise ValueError(
            f'{config_path}: {task_name}/{condition_id}: perception sensor-model package is missing'
        )
    model_path = _resolve_repo_path(declared_path, strict=True)
    actual_sha = sha256_file(model_path)
    declared_sha = str(
        expected('manager_perception_sensor_model_expected_sha256') or ''
    ).strip().lower()
    if actual_sha != declared_sha:
        raise ValueError(
            f'{config_path}: {task_name}/{condition_id}: perception sensor-model hash mismatch'
        )
    try:
        package = json.loads(model_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'{config_path}: malformed perception sensor-model package') from exc
    if package.get('schema') != 'commissioned_perception_runtime_model.v1':
        raise ValueError(f'{config_path}: unsupported perception sensor-model schema')
    if package.get('method_id') != observation:
        raise ValueError(
            f'{config_path}: {task_name}/{condition_id}: configured method and package differ'
        )


def _terminate_process_group(pgid: int, *, grace_s: float = 3.0) -> None:
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    time.sleep(grace_s)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


_OWN_NODE_PATTERNS = [
    'yolo_robot_detector_node',
    'pixel_to_bev_state_node',
    'goal_mission_node',
    'goal_marker_node',
    'experiment_logger',
    'install/planning/lib/planning/efe_agent',
    'wait_for_odom',
    'wait_for_clock',
    'reset_world',
    # Sim noise nodes MUST be reaped: a leftover encoder_noise_node from a prior
    # run publishes a second (pi-offset) heading on /odom_noisy, corrupting the
    # planner's heading. This caused C1's spurious start-divergence (2026-06-03).
    'encoder_noise_node',
    'actuation_noise_node',
]


def _reap_own_node_stragglers() -> None:
    for pattern in _OWN_NODE_PATTERNS:
        try:
            subprocess.run(['pkill', '-f', pattern], timeout=2, capture_output=True, check=False)
        except (subprocess.TimeoutExpired, OSError):
            pass


def _reap_sim_stragglers(world: str) -> None:
    """Stop stale Gazebo servers for this campaign world.

    Gazebo Transport is outside ROS_DOMAIN_ID. A leftover server with the same
    world name can publish duplicate clocks/sensors into a fresh ROS run.
    """
    world_name = str(world or '').strip()
    if not world_name:
        return
    for pattern in (
        f'ign gazebo.*{world_name}',
        f'ruby /usr/bin/ign gazebo.*{world_name}',
        f'gz sim.*{world_name}',
    ):
        try:
            subprocess.run(['pkill', '-f', pattern], timeout=2, capture_output=True, check=False)
        except (subprocess.TimeoutExpired, OSError):
            pass


# Heavy stragglers that were NOT being reaped and accumulate CPU across a campaign:
# the ros_gz bridges, robot_state_publisher, rviz, and the bare Gazebo server/client
# (which often re-parents away from the launch process group and so survives killpg).
_EXTRA_STRAGGLER_PATTERNS = [
    'ros_gz_bridge', 'parameter_bridge', 'ros_gz_image', 'image_bridge',
    'robot_state_publisher', 'rviz', 'static_transform_publisher',
    'ros2 launch experiments', 'spawn_entity',
]
_SIM_STRAGGLER_PATTERNS = [
    'ign gazebo', 'gz sim', 'gzserver', 'gzclient', 'ruby /usr/bin/ign gazebo',
]


def _all_straggler_patterns() -> list:
    return list(_OWN_NODE_PATTERNS) + _EXTRA_STRAGGLER_PATTERNS + _SIM_STRAGGLER_PATTERNS


def _patterns_alive(patterns) -> list:
    alive = []
    for p in patterns:
        try:
            r = subprocess.run(['pgrep', '-f', p], capture_output=True, timeout=2)
            if r.returncode == 0 and r.stdout.strip():
                alive.append(p)
        except (subprocess.TimeoutExpired, OSError):
            pass
    return alive


def _force_fresh(*, settle_s: float = 1.5, max_wait_s: float = 25.0) -> None:
    """Guarantee a clean slate before/after a run: SIGTERM all known run processes,
    then SIGKILL and POLL until none remain (or timeout). This prevents Gazebo/bridge
    stragglers from accumulating and slowly starving the next run's global solve."""
    patterns = _all_straggler_patterns()
    for p in patterns:
        try:
            subprocess.run(['pkill', '-TERM', '-f', p], timeout=2, capture_output=True, check=False)
        except (subprocess.TimeoutExpired, OSError):
            pass
    time.sleep(settle_s)
    deadline = time.time() + max_wait_s
    while True:
        for p in patterns:
            try:
                subprocess.run(['pkill', '-KILL', '-f', p], timeout=2, capture_output=True, check=False)
            except (subprocess.TimeoutExpired, OSError):
                pass
        time.sleep(0.5)
        alive = _patterns_alive(patterns)
        if not alive:
            return
        if time.time() > deadline:
            print(f'  WARN: stragglers still alive after force-clean: {alive}')
            return


def _run_key(task: str, condition: str, seed: int) -> str:
    return f'{task}__{condition}__seed{seed}'


def _pids_with_run_token(token: str, proc_root: Path = Path('/proc')) -> list[int]:
    """Find only descendants carrying this run's inherited environment marker."""
    if not token or '\x00' in token:
        raise ValueError('nonempty run token required for scoped cleanup')
    marker = f'UNAV_CAMPAIGN_RUN_TOKEN={token}'.encode()
    pids = []
    for proc in proc_root.iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if marker in (proc / 'environ').read_bytes().split(b'\x00'):
                pids.append(int(proc.name))
        except (OSError, PermissionError):
            continue
    return pids


def _cleanup_owned_run(token: str) -> None:
    """Reap reparented Gazebo children without touching other ROS experiments."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in _pids_with_run_token(token):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        time.sleep(0.5)


def _load_run_log(log_path: Path) -> dict:
    if not log_path.is_file():
        return {}
    try:
        payload = json.loads(log_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f'malformed campaign ledger {log_path}: {exc}') from exc
    except OSError as exc:
        raise RuntimeError(f'cannot read campaign ledger {log_path}: {exc}') from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f'campaign ledger {log_path} must contain a JSON object')
    return payload


@contextmanager
def _directory_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _save_run_log_unlocked(log_path: Path, log: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # A stopped writer must not truncate the only campaign ledger. The temporary
    # file is on the same filesystem so replacement is atomic.
    payload = json.dumps(log, indent=2, default=str)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                dir=log_path.parent, prefix=log_path.name + '.', suffix='.tmp',
                delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, log_path)
        temporary = None
        directory_fd = os.open(log_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _save_run_log(log_path: Path, log: dict) -> None:
    with _directory_lock(log_path.parent):
        _save_run_log_unlocked(log_path, log)


def _update_run_log(log_path: Path, key: str, entry: dict) -> dict:
    """Update one cell under a lock, preserving other writers' completed cells."""
    with _directory_lock(log_path.parent):
        ledger = _load_run_log(log_path)
        ledger[key] = dict(entry)
        _save_run_log_unlocked(log_path, ledger)
        return ledger


@contextmanager
def _exclusive_lease(path: Path, owner: dict):
    """Hold an exclusive nonblocking resource lease for this process lifetime."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open('a+', encoding='utf-8')
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            stream.seek(0)
            current = stream.read().strip() or '<unknown owner>'
            raise RuntimeError(f'resource lease already held: {path}: {current}') from exc
        stream.seek(0)
        stream.truncate()
        stream.write(json.dumps(owner, sort_keys=True))
        stream.flush()
        os.fsync(stream.fileno())
        yield
    finally:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


def _resolve_repo_path(path_str: str, *, strict: bool = False) -> Path:
    """Resolve config paths, treating relative paths as repo-root relative."""
    expanded = os.path.expandvars(str(path_str))
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve(strict=strict)


def _resolve_for_compare(path_str: str) -> Path:
    return _resolve_repo_path(path_str, strict=False)


def _verify_installed_world_matches_checkout(world_file: str) -> Path:
    """Return the SDF Gazebo will load, refusing a stale installed copy."""
    source = REPO_ROOT / 'src' / 'sim' / 'gazebo_worlds' / 'worlds' / world_file
    if not source.is_file():
        raise RuntimeError(f'checkout world does not exist: {source}')
    try:
        from ament_index_python.packages import get_package_share_directory
        installed = (
            Path(get_package_share_directory('sim'))
            / 'gazebo_worlds' / 'worlds' / world_file
        )
    except Exception as exc:
        raise RuntimeError(
            'sim package is not available in the active ROS environment'
        ) from exc
    if not installed.is_file():
        raise RuntimeError(f'installed simulator world does not exist: {installed}')
    if sha256_file(source) != sha256_file(installed):
        raise RuntimeError(
            f'installed simulator world is stale relative to checkout: {installed}'
        )
    return installed.resolve()


def _checkout_pythonpath() -> str:
    """Prefer this checkout's Python packages over possibly stale installs."""
    package_roots = sorted({
        str(setup.parent.resolve()) for setup in (REPO_ROOT / 'src').glob('*/setup.py')
    })
    if not package_roots:
        raise RuntimeError('no checkout Python packages found under src/*/setup.py')
    existing = os.environ.get('PYTHONPATH', '')
    return os.pathsep.join(package_roots + ([existing] if existing else []))


def _freeze_campaign_source(log_root: Path, config_path: Path, provenance: dict) -> Path:
    """Freeze the executable source/config bytes once for this campaign root."""
    snapshot = log_root / 'source_snapshot'
    identity_path = snapshot / 'source_identity.json'
    if snapshot.exists():
        try:
            recorded = json.loads(identity_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f'malformed existing campaign source snapshot: {exc}') from exc
        if recorded.get('git_provenance') != provenance:
            raise RuntimeError('campaign source snapshot differs from executable checkout')
        return snapshot

    temporary = log_root / f'.source_snapshot.{uuid.uuid4().hex}.tmp'
    temporary.mkdir(parents=False, exist_ok=False)
    suffixes = {'.py', '.yaml', '.yml', '.json', '.xml', '.sdf', '.urdf', '.xacro'}
    try:
        for source_root in (REPO_ROOT / 'src', REPO_ROOT / 'scripts' / 'visibility_comparison'):
            for source in source_root.rglob('*'):
                if (not source.is_file() or '__pycache__' in source.parts
                        or (source.suffix not in suffixes and source.name not in {'setup.py', 'package.xml'})):
                    continue
                destination = temporary / source.relative_to(REPO_ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        config_destination = temporary / 'campaign_config' / config_path.name
        config_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, config_destination)
        atomic_write_json(str(temporary / 'source_identity.json'), {
            'git_provenance': provenance,
            'campaign_config_sha256': sha256_file(config_path),
            'executable_source_root': str(REPO_ROOT),
        })
        os.replace(temporary, snapshot)
        temporary = None
        directory_fd = os.open(log_root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
    return snapshot


def _as_bool(value) -> bool:
    try:
        return parse_bool(value)
    except ValueError as exc:
        raise ValueError(f"invalid boolean value {value!r}") from exc


def _route_overrides(cfg: dict, task_name: str, condition_id: str) -> dict:
    task_cfg = cfg.get('tasks', {}).get(task_name, {}) or {}
    routes = task_cfg.get('preselected_routes', {}) or {}
    if not isinstance(routes, dict):
        raise RuntimeError(
            f"Task '{task_name}' preselected_routes must be a mapping by condition"
        )
    route_cfg = routes.get(condition_id, {}) or {}
    if not isinstance(route_cfg, dict):
        raise RuntimeError(
            f"Task '{task_name}' route '{condition_id}' must be a mapping"
        )
    return route_cfg


def _effective_value(cfg: dict, task_name: str, condition_id: str, key: str):
    """Resolve route > task-condition > condition > task > campaign overrides."""

    task_cfg = cfg.get('tasks', {}).get(task_name, {}) or {}
    condition_cfg = cfg.get('conditions', {}).get(condition_id, {}) or {}
    task_condition_cfg = (task_cfg.get('condition_overrides', {}) or {}).get(
        condition_id, {}) or {}
    route_cfg = _route_overrides(cfg, task_name, condition_id)
    if key in route_cfg:
        return route_cfg[key]
    if key in task_condition_cfg:
        return task_condition_cfg[key]
    if key in condition_cfg:
        return condition_cfg[key]
    if key in task_cfg:
        return task_cfg[key]
    return cfg.get(key)


def _resolved_cell_config(cfg: dict, task_name: str, condition_id: str) -> dict:
    """Materialize route > condition > task > campaign precedence once."""
    task_cfg = cfg.get('tasks', {}).get(task_name, {}) or {}
    condition_cfg = cfg.get('conditions', {}).get(condition_id, {}) or {}
    task_condition_cfg = (task_cfg.get('condition_overrides', {}) or {}).get(
        condition_id, {}) or {}
    route_cfg = _route_overrides(cfg, task_name, condition_id)
    resolved = dict(cfg)
    for layer in (task_cfg, condition_cfg, task_condition_cfg, route_cfg):
        for key, value in layer.items():
            if key not in {'conditions', 'seeds', 'preselected_routes',
                           'condition_overrides', 'label', 'planner'}:
                resolved[key] = value
    return resolved


def _task_spec_from_yaml(cfg: dict, task_name: str, config_path: Path) -> dict:
    tasks_path = _resolve_repo_path(
        cfg.get('tasks_yaml', 'src/experiments/config/tasks.yaml'), strict=True
    )
    payload = yaml.safe_load(tasks_path.read_text(encoding='utf-8')) or {}
    world_tasks = (payload.get('tasks') or {}).get(cfg['world'])
    if not isinstance(world_tasks, list):
        raise RuntimeError(
            f"Campaign {config_path}: no tasks list for world {cfg['world']!r} "
            f"in {tasks_path}"
        )
    for task in world_tasks:
        if isinstance(task, dict) and task.get('name') == task_name:
            return task
    raise RuntimeError(
        f"Campaign {config_path}: task {task_name!r} is not registered for "
        f"world {cfg['world']!r} in {tasks_path}"
    )


def _profile_driveable_geometry(cfg: dict, config_path: Path) -> str:
    profiles_path = _resolve_repo_path(
        cfg.get('world_profiles', 'src/experiments/config/world_profiles.yaml'),
        strict=True,
    )
    payload = yaml.safe_load(profiles_path.read_text(encoding='utf-8')) or {}
    profile = (payload.get('worlds') or {}).get(cfg['world'])
    if not isinstance(profile, dict):
        raise RuntimeError(
            f"Campaign {config_path}: world {cfg['world']!r} is not present in "
            f"{profiles_path}"
        )
    prisms = []
    for region in profile.get('known_2d_regions', []) or []:
        if str(region.get('type', '')).strip().lower() != 'traversable':
            continue
        try:
            prisms.append({
                'name': str(region.get('name', 'lane')),
                'xmin': float(region['xmin']),
                'xmax': float(region['xmax']),
                'ymin': float(region['ymin']),
                'ymax': float(region['ymax']),
                'zmin': 0.0,
                'zmax': 0.1,
            })
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Campaign {config_path}: malformed traversable region {region!r}"
            ) from exc
    if not prisms:
        raise RuntimeError(
            f"Campaign {config_path}: world profile has no traversable driveable regions"
        )
    return json.dumps({'prisms': prisms, 'model_name': 'driveable_region'})


def _validate_preselected_campaign_routes(cfg: dict, config_path: Path) -> None:
    """Make ``--dry-run`` exercise the same route identity/geometry gate as launch."""

    profile_driveable = None
    for task_name, task_cfg in cfg['tasks'].items():
        for condition_id in task_cfg.get('conditions', []):
            mode = str(
                _effective_value(cfg, task_name, condition_id, 'global_planner_mode') or ''
            ).strip().lower()
            route_cfg = _route_overrides(cfg, task_name, condition_id)
            if mode != 'preselected_route':
                if route_cfg:
                    raise RuntimeError(
                        f"Task {task_name!r} condition {condition_id!r} supplies a "
                        "preselected route but global_planner_mode is not preselected_route"
                    )
                continue
            if not _as_bool(
                _effective_value(cfg, task_name, condition_id, 'use_hierarchical')
            ):
                raise RuntimeError(
                    f"Task {task_name!r} condition {condition_id!r}: "
                    "preselected_route requires use_hierarchical=true"
                )
            if _as_bool(
                _effective_value(
                    cfg, task_name, condition_id, 'use_diagnostic_odom_localization')
            ):
                raise RuntimeError(
                    f"Task {task_name!r} condition {condition_id!r}: "
                    "use_diagnostic_odom_localization feeds raw odometry to the planner "
                    "as its belief, bypassing the cameras entirely. It is a controller "
                    "diagnostic and can never produce a closed-loop result."
                )
            identity_keys = PRESELECTED_ROUTE_KEYS[:4]
            missing = [key for key in identity_keys if not str(route_cfg.get(key, '') or '')]
            if missing:
                raise RuntimeError(
                    f"Task {task_name!r} condition {condition_id!r} is missing "
                    f"route-specific fields: {', '.join(missing)}"
                )

            task = _task_spec_from_yaml(cfg, task_name, config_path)
            if task.get('waypoints'):
                raise RuntimeError(
                    f"Task {task_name!r} is a waypoint tour; preselected_route accepts "
                    "one start-to-goal polyline"
                )
            try:
                start = (float(task['start']['x']), float(task['start']['y']))
                goal = (float(task['goal']['x']), float(task['goal']['y']))
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Task {task_name!r} has malformed start/goal coordinates"
                ) from exc

            driveable = _effective_value(
                cfg, task_name, condition_id, 'driveable_geometry_json'
            )
            if not str(driveable or '').strip():
                if profile_driveable is None:
                    profile_driveable = _profile_driveable_geometry(cfg, config_path)
                driveable = profile_driveable

            endpoint_value = _effective_value(
                cfg, task_name, condition_id,
                'preselected_route_endpoint_tolerance_m',
            )
            sample_step_value = _effective_value(
                cfg, task_name, condition_id, 'preselected_route_sample_step_m'
            )
            endpoint_tolerance = float(
                0.25 if endpoint_value is None else endpoint_value
            )
            sample_step = float(
                0.04 if sample_step_value is None else sample_step_value
            )
            if not 0.0 <= endpoint_tolerance <= 0.25:
                raise RuntimeError(
                    "preselected_route_endpoint_tolerance_m must be within [0, 0.25]"
                )
            if not 0.0 < sample_step <= 0.04:
                raise RuntimeError(
                    "preselected_route_sample_step_m must be within (0, 0.04]"
                )
            source_path = _resolve_repo_path(
                route_cfg['preselected_route_source_path'], strict=True
            )
            try:
                validate_preselected_route(
                    route_cfg['preselected_route_json'],
                    route_cfg['preselected_route_sha256'],
                    start_xy=start,
                    goal_xy=goal,
                    driveable_geometry_json=str(driveable),
                    declared_clearance_m=float(
                        _effective_value(
                            cfg,
                            task_name,
                            condition_id,
                            'preselected_route_clearance_m',
                        )
                        or 0.25
                    ),
                    source_path=source_path,
                    expected_source_sha256=route_cfg[
                        'preselected_route_source_sha256'
                    ],
                    endpoint_tolerance_m=endpoint_tolerance,
                    sample_step_m=sample_step,
                )
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    f"Campaign {config_path}: route gate failed for task "
                    f"{task_name!r}, condition {condition_id!r}: {exc}"
                ) from exc


def _float_close(a, b, *, tol: float = 1e-8) -> bool:
    try:
        fa = float(a)
        fb = float(b)
    except (TypeError, ValueError):
        return False
    return math.isfinite(fa) and math.isfinite(fb) and abs(fa - fb) <= tol


def _load_run_manifest(run_dir: Path) -> dict:
    p = run_dir / 'run_manifest.json'
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}


def _verify_preselected_run_artifacts(
    run_dir: Path,
    cfg: dict,
    task_name: str,
    condition_id: str,
    *,
    manifest: dict | None = None,
) -> tuple[bool, str]:
    """Require the executed route bytes and no-global-solve provenance on disk."""

    mode = str(
        _effective_value(cfg, task_name, condition_id, 'global_planner_mode') or ''
    ).strip().lower()
    if mode != 'preselected_route':
        return True, ''
    route_cfg = _route_overrides(cfg, task_name, condition_id)
    try:
        points, canonical = canonicalize_polyline_json(
            route_cfg['preselected_route_json']
        )
    except (KeyError, ValueError) as exc:
        return False, f'invalid configured preselected route: {exc}'
    expected_hash = str(route_cfg.get('preselected_route_sha256', '') or '')
    if route_sha256(canonical) != expected_hash:
        return False, 'configured canonical preselected-route hash no longer matches'
    expected_source = str(
        _resolve_repo_path(route_cfg.get('preselected_route_source_path', ''), strict=False)
    )
    expected_source_hash = str(
        route_cfg.get('preselected_route_source_sha256', '') or ''
    )

    run_manifest = manifest if manifest is not None else _load_run_manifest(run_dir)
    strict_manifest_values = {
        'global_planner_mode': 'preselected_route',
        'preselected_route_json': canonical,
        'preselected_route_sha256': expected_hash,
        'preselected_route_source_path': expected_source,
        'preselected_route_source_sha256': expected_source_hash,
    }
    for key, expected in strict_manifest_values.items():
        actual = str(run_manifest.get(key, '') or '')
        if key == 'preselected_route_source_path':
            if _resolve_for_compare(actual) != _resolve_for_compare(expected):
                return False, f'run manifest {key} mismatch'
        elif actual != expected:
            return False, f'run manifest {key} mismatch'

    route_path = run_dir / 'preselected_route.json'
    try:
        route_bytes = route_path.read_bytes()
        route_text = route_bytes.decode('utf-8')
    except (OSError, UnicodeDecodeError) as exc:
        return False, f'missing/malformed exact preselected route artifact: {exc}'
    if route_text != canonical:
        return False, 'preselected_route.json bytes differ from canonical campaign route'
    if route_sha256(route_text) != expected_hash:
        return False, 'preselected_route.json has the wrong route SHA-256'

    provenance_path = run_dir / 'preselected_route_provenance.json'
    meta_path = run_dir / 'global_plan_meta.json'
    try:
        provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f'missing/malformed preselected route provenance: {exc}'
    if provenance.get('validation_status') != 'passed':
        return False, 'preselected route provenance does not record a passed gate'
    if provenance.get('route_sha256') != expected_hash:
        return False, 'preselected route provenance hash mismatch'
    if provenance.get('route_points') != [[x, y] for x, y in points]:
        return False, 'preselected route provenance coordinates mismatch'
    if provenance.get('source_sha256') != expected_source_hash:
        return False, 'preselected route source provenance hash mismatch'
    if _resolve_for_compare(provenance.get('source_path', '')) != _resolve_for_compare(
        expected_source
    ):
        return False, 'preselected route source provenance path mismatch'
    if meta.get('global_planner_mode') != 'preselected_route':
        return False, 'global plan metadata does not identify preselected_route mode'
    if meta.get('global_solve_invoked') is not False:
        return False, 'global plan metadata does not prove the global solve was skipped'
    if meta.get('route_sha256') != expected_hash:
        return False, 'global plan metadata route hash mismatch'
    return True, ''


def _terminal_summary_outcome(summary: dict | None) -> tuple[bool, str, str]:
    """Classify only a fully committed, evidence-complete terminal summary."""
    if not isinstance(summary, dict):
        return False, 'infra_invalid', 'missing_or_malformed_summary'
    for field in ('completed', 'valid_run', 'data_files_closed'):
        if summary.get(field) is not True:
            return False, 'infra_invalid', f'summary_{field}_not_true'
    # Newer logger schemas emit these fields.  If present, they are mandatory:
    # a physical route outcome is not evidence-complete until the operational
    # stop and producer-quiescence handshake has been verified.
    for field in (
        'evidence_complete',
        'terminal_stop_verified',
        'producer_quiescence_acknowledged',
    ):
        if field in summary and summary.get(field) is not True:
            return False, 'infra_invalid', f'summary_{field}_not_true'
    if 'terminal_stop_request_id' in summary:
        request_id = summary.get('terminal_stop_request_id')
        if not isinstance(request_id, str) or not request_id:
            return False, 'infra_invalid', 'summary_terminal_stop_request_id_missing'
        acknowledgements = summary.get('terminal_stop_acknowledgements')
        if not isinstance(acknowledgements, dict):
            return False, 'infra_invalid', 'summary_terminal_stop_acknowledgements_missing'
        for component, expected_status in TERMINAL_ACK_STATUS_BY_COMPONENT.items():
            ack = acknowledgements.get(component)
            if not isinstance(ack, dict) or ack.get('status') != expected_status:
                return False, 'infra_invalid', f'summary_{component}_terminal_ack_invalid'
    reason = str(summary.get('completion_reason', '') or '')
    if reason in ('goal_reached', 'goal_reached_stable'):
        return True, 'goal_reached', reason
    if reason == 'timeout_after_first_cmd':
        return True, 'timeout', reason
    if reason == 'stuck':
        return True, 'stuck', reason
    if reason == 'goal_loiter_timeout':
        return True, 'goal_loiter_timeout', reason
    return False, 'infra_invalid', f'unrecognized_terminal_reason:{reason or "missing"}'


def _existing_entry_matches_config(
    entry: dict,
    cfg: dict,
    *,
    expected_cell: tuple[str, str, int] | None = None,
    allow_imported_provenance: bool = False,
) -> tuple[bool, str]:
    if not isinstance(entry, dict):
        return False, 'campaign entry is not an object'
    if expected_cell is not None:
        actual_cell = (entry.get('task'), entry.get('condition'), entry.get('seed'))
        if actual_cell != expected_cell:
            return False, f'ledger cell identity mismatch: {actual_cell!r} != {expected_cell!r}'
    if ('_execution_options' in cfg
            and entry.get('execution_options') != cfg['_execution_options']):
        return False, 'planner execution options differ'
    condition_id = str(entry.get('condition', ''))
    run_dir_str = str(entry.get('run_dir', '') or '')
    if not run_dir_str:
        return False, 'missing run_dir'
    manifest = _load_run_manifest(Path(run_dir_str))
    if not manifest:
        return False, f'missing run_manifest.json in {run_dir_str}'
    if int(manifest.get('logging_schema_version', 0) or 0) < 4:
        return False, 'run predates logging schema 4 (source-batch assimilation)'
    if manifest.get('goal_termination_reference') != 'planner_belief':
        return False, 'run did not use planner_belief for goal termination'
    expected_provenance = cfg.get('_git_provenance', {}) or {}
    provenance_keys = (
        'git_sha', 'git_status_sha256', 'git_diff_sha256',
        'git_untracked_content_sha256',
    )
    actual_provenance = {key: manifest.get(key) for key in provenance_keys}
    expected_provenance_subset = {
        key: expected_provenance.get(key) for key in provenance_keys
    }
    if actual_provenance != expected_provenance_subset:
        imported_provenance = entry.get('imported_executable_provenance')
        imported_source = str(entry.get('imported_from_campaign_root', '') or '')
        if not (
            allow_imported_provenance
            and imported_source
            and imported_provenance == actual_provenance
        ):
            differing = next(
                key for key in provenance_keys
                if actual_provenance[key] != expected_provenance_subset[key]
            )
            return False, f'{differing} differs from the executable checkout'

    task_name = str(entry.get('task', '') or '')
    expected_manifest_identity = {
        'task': task_name,
        'planner': CONDITION_PLANNER.get(condition_id),
        'world': cfg.get('world'),
        'seed': entry.get('seed'),
    }
    for key, expected in expected_manifest_identity.items():
        if manifest.get(key) != expected:
            return False, f'run manifest {key} mismatch: {manifest.get(key)!r} != {expected!r}'
    expected_world_path = str(cfg.get('_world_sdf_path', '') or '')
    if expected_world_path:
        if _resolve_for_compare(manifest.get('world_sdf_path', '')) != _resolve_for_compare(
            expected_world_path
        ):
            return False, 'world SDF path mismatch'
        if manifest.get('world_sdf_sha256') != cfg.get('_world_sdf_sha256'):
            return False, 'world SDF content hash mismatch'

    summary = _read_run_summary(Path(run_dir_str))
    terminal_ok, terminal_outcome, terminal_reason = _terminal_summary_outcome(summary)
    if not terminal_ok:
        return False, terminal_reason
    if entry.get('outcome') != terminal_outcome:
        return False, 'campaign outcome disagrees with terminal summary'
    if entry.get('completion_reason') != str(summary.get('completion_reason', '') or ''):
        return False, 'campaign completion_reason disagrees with terminal summary'
    if entry.get('attempt_evidence_complete') is not True:
        return False, 'campaign attempt evidence verdict is not complete'
    verdict_path = Path(str(entry.get('run_log_dir', '') or '')) / 'attempt_evidence_verdict.json'
    try:
        verdict = json.loads(verdict_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f'missing or malformed attempt evidence verdict: {exc}'
    if (not isinstance(verdict, dict) or verdict.get('complete') is not True
            or verdict.get('attempt_id') != entry.get('attempt_id')
            or verdict.get('run_dir') != run_dir_str):
        return False, 'attempt evidence verdict identity mismatch'
    assimilation_ok, assimilation_reason = _verify_correction_assimilations(
        Path(run_dir_str)
    )
    if not assimilation_ok:
        return False, f'correction evidence invalid: {assimilation_reason}'

    def expected_value(key: str):
        return _effective_value(cfg, task_name, condition_id, key)

    multicam_value = expected_value('multicam_belief')
    if multicam_value is not None and _as_bool(multicam_value):
        attempt_dir = Path(str(entry.get('run_log_dir', '') or ''))
        journal_ok, journal_verdict = _verify_detector_journal(
            attempt_dir
        )
        if not journal_ok:
            return False, str(journal_verdict.get('reason', 'detector journal invalid'))
        if entry.get('detector_journal_sha256') != journal_verdict.get('journal_sha256'):
            return False, 'detector journal content hash mismatch'
        manager_ok, manager_verdict = _verify_outcome_journal(
            attempt_dir, 'manager_outcomes.jsonl', 'manager'
        )
        if not manager_ok:
            return False, str(manager_verdict.get('reason', 'manager journal invalid'))
        if entry.get('manager_journal_sha256') != manager_verdict.get('journal_sha256'):
            return False, 'manager journal content hash mismatch'

    expected_yolo_model = str(_resolve_repo_path(cfg['yolo_model'], strict=False))
    actual_yolo_model = str(manifest.get('yolo_model', '') or '')
    if _resolve_for_compare(actual_yolo_model) != _resolve_for_compare(expected_yolo_model):
        return False, f'yolo_model mismatch: run used {actual_yolo_model or "<missing>"}, config expects {expected_yolo_model}'
    expected_model_hash = cfg.get('_yolo_model_sha256') or sha256_file(expected_yolo_model)
    if manifest.get('yolo_model_sha256') != expected_model_hash:
        return False, 'yolo_model content hash mismatch'
    campaign_path = str(cfg.get('_campaign_config_path', '') or '')
    expected_campaign_hash = cfg.get('_campaign_config_sha256')
    if not campaign_path or manifest.get('campaign_config_sha256') != expected_campaign_hash:
        return False, 'campaign config content hash mismatch'

    numeric_keys = (
        'horizon', 'dt', 'goal_success_radius', 'goal_success_hold_s',
        'operational_belief_timeout_s',
        'optimizer_terminal_goal_tolerance_m',
        'run_timeout_after_first_cmd_s', 'r_visible_uv', 'r_miss_uv',
        'process_noise_xy', 'process_noise_theta', 'risk_weight_obs',
        'encoder_wheel_diameter_ratio_error', 'encoder_wheelbase_ratio',
        'ambiguity_weight', 'observation_risk_scale', 'ambiguity_term_scale',
        'control_weight', 'v_max', 'discount_gamma', 'yolo_conf_threshold',
        'yolo_predict_conf_floor',
        'yolo_iou_threshold', 'odom_heading_timeout_s', 'optimizer_maxiter',
        'optimizer_maxfun', 'optimizer_ftol', 'optimizer_gtol',
        'goal_prior_u_std_start', 'goal_prior_v_std_start',
        'goal_prior_u_std_final', 'goal_prior_v_std_final',
        'goal_tightening_power', 'nogo_safe_distance',
        'nogo_logbarrier_eps',
        'nogo_belief_kappa',
        'pixel_correction_nis_threshold',
        'robot_collision_radius_m', 'robot_length_m', 'robot_width_m',
        'camera_network_updates_per_step',
        'optimizer_control_block_steps',
        'global_horizon', 'global_dt', 'local_horizon', 'local_plan_rate',
        'local_optimizer_maxiter',
        'local_nogo_safe_distance',
        'local_goal_prior_u_std_start', 'local_goal_prior_v_std_start',
        'local_goal_prior_u_std_final', 'local_goal_prior_v_std_final',
        'waypoint_spacing_m', 'waypoint_arrival_radius_m',
        'local_replan_min_remaining_s', 'cmd_publish_rate',
        'encoder_noise_linear_slip_mean',
        'encoder_noise_linear_slip_std',
        'encoder_noise_angular_slip_mean',
        'encoder_noise_angular_slip_std',
        'encoder_noise_linear_additive_std',
        'encoder_noise_angular_additive_std',
        'encoder_noise_correlation_alpha',
        'preselected_route_clearance_m',
        'preselected_route_endpoint_tolerance_m',
        'preselected_route_sample_step_m',
        'state_reanchor_m', 'state_max_predict_dt_s', 'state_reject_inflate_m2',
        'stale_belief_inflate_m2_per_s', 'stale_belief_inflate_cap_m2',
        'yolo_max_batch_stamp_skew_s',
        'manager_min_spatial_trust', 'manager_decision_rate_hz',
        'manager_fusion_disagreement_gate_m',
        'manager_bootstrap_min_cameras', 'manager_bootstrap_max_disagreement_m',
        'manager_fusion_max_timestamp_spread_s',
        'manager_commissioned_sigma_px', 'manager_fusion_common_mode_std_m',
        'manager_correction_residual_interval_s',
        'manager_correction_propagation_drift_std',
        'manager_max_measurement_age_s', 'manager_age_decay_s',
        'manager_min_association_confidence',
        'manager_required_consecutive_better_frames',
        'manager_max_cross_camera_disagreement_m',
    )
    for key in numeric_keys:
        expected = expected_value(key)
        if expected is not None:
            if key not in manifest:
                return False, f'{key} missing from run manifest'
            if not _float_close(manifest.get(key), expected):
                return False, f'{key} mismatch: run used {manifest.get(key, "<missing>")}, config expects {expected}'

    bool_keys = (
        'use_nogo_cost',
        'use_belief_nogo_cost',
        'use_command_noise',
        'use_encoder_noise',
        'use_odom_for_predict',
        'optimizer_multistart',
        'optimizer_multistart_include_direct',
        'use_hierarchical',
        'global_use_ambiguity',
        'local_use_ambiguity',
        'local_use_obs_risk',
        'global_optimizer_multistart',
        'local_optimizer_multistart',
        'local_use_visibility_model',
        'local_use_belief_nogo_cost',
        'local_replan_on_waypoint_change',
        'latency_compensate_plan_handoff',
        'use_diagnostic_odom_localization',
        'use_hit_miss_mixture',
        'manager_require_source_batch_id',
        'require_state_correction_envelope',
        'manager_commissioned_per_camera_sigma',
        'manager_correction_timestamp_compensation',
        'manager_require_consistency_when_source_available',
        'manager_fusion_mode', 'manager_require_gp_artifacts',
        'manager_use_task_start_as_bootstrap_prior',
        'manager_bootstrap_prior_counts_as_support',
    )
    for key in bool_keys:
        expected = expected_value(key)
        if expected is not None:
            if key not in manifest:
                return False, f'{key} missing from run manifest'
            try:
                actual_bool = _as_bool(manifest.get(key))
                expected_bool = _as_bool(expected)
            except ValueError as exc:
                return False, f'{key} has invalid boolean encoding: {exc}'
            if actual_bool != expected_bool:
                return False, f'{key} mismatch: run used {manifest.get(key)}, config expects {expected}'

    string_keys = (
        'nogo_mode',
        'nogo_penalty_type',
        'yolo_device',
        'optimizer_initial_routes_json',
        'optimizer_route_seed_mode',
        'local_nogo_penalty_type',
        'heading_update_mode',
        'process_noise_model',
        'state_correction_mode',
        'local_controller_type',
        'global_planner_mode',
        'manager_gp_artifact_template', 'manager_camera_ids',
        'manager_covariance_profile', 'manager_commissioned_calibration_path',
        'manager_commissioned_world_covariance_path',
        'manager_fusion_rule', 'manager_observation_model',
        'manager_sensor_gate_config_path',
        'manager_availability_model_path',
        'manager_availability_model_expected_sha256',
        'manager_learned_correction_path',
        'manager_visibility_sensor_model_path',
        'manager_visibility_sensor_model_expected_sha256',
        'manager_perception_sensor_model_path',
        'manager_perception_sensor_model_expected_sha256',
    )
    for key in string_keys:
        expected = expected_value(key)
        if expected is not None:
            if key not in manifest:
                return False, f'{key} missing from run manifest'
            actual = str(manifest.get(key, ''))
            if key in ('manager_commissioned_calibration_path',
                       'manager_commissioned_world_covariance_path',
                       'manager_sensor_gate_config_path',
                       'manager_availability_model_path',
                       'manager_visibility_sensor_model_path',
                       'manager_perception_sensor_model_path'):
                matches = _resolve_for_compare(actual) == _resolve_for_compare(str(expected))
            else:
                matches = actual == str(expected)
            if not matches:
                return False, f'{key} mismatch: run used {actual!r}, config expects {expected!r}'

    calibration_path = expected_value('manager_commissioned_calibration_path')
    if calibration_path:
        expected_calibration_hash = sha256_file(
            _resolve_repo_path(str(calibration_path), strict=True)
        )
        if manifest.get('manager_commissioned_calibration_sha256') != expected_calibration_hash:
            return False, 'manager commissioned calibration content hash mismatch'

    world_covariance_path = expected_value('manager_commissioned_world_covariance_path')
    if world_covariance_path:
        expected_world_hash = sha256_file(
            _resolve_repo_path(str(world_covariance_path), strict=True)
        )
        if manifest.get('manager_commissioned_world_covariance_sha256') != expected_world_hash:
            return False, 'manager commissioned world covariance content hash mismatch'

    visibility_model_path = expected_value('manager_visibility_sensor_model_path')
    if visibility_model_path:
        expected_visibility_hash = sha256_file(
            _resolve_repo_path(str(visibility_model_path), strict=True)
        )
        if manifest.get('manager_visibility_sensor_model_sha256') != expected_visibility_hash:
            return False, 'manager visibility sensor-model content hash mismatch'

    perception_model_path = expected_value('manager_perception_sensor_model_path')
    if perception_model_path:
        expected_perception_hash = sha256_file(
            _resolve_repo_path(str(perception_model_path), strict=True)
        )
        if manifest.get('manager_perception_sensor_model_sha256') != expected_perception_hash:
            return False, 'manager perception sensor-model content hash mismatch'

    sensor_gate_path = expected_value('manager_sensor_gate_config_path')
    if sensor_gate_path:
        expected_sensor_gate_hash = sha256_file(
            _resolve_repo_path(str(sensor_gate_path), strict=True)
        )
        if manifest.get('manager_sensor_gate_config_sha256') != expected_sensor_gate_hash:
            return False, 'manager sensor-gate content hash mismatch'

    route_matches, route_reason = _verify_preselected_run_artifacts(
        Path(run_dir_str), cfg, task_name, condition_id, manifest=manifest
    )
    if not route_matches:
        return False, route_reason

    # Only an actually solved visibility-aware planner consumes the GP; C1
    # (constant_R_efe) and C0 (geometric_shortest_path) are camera-model-free.
    if (
        CONDITION_PLANNER.get(condition_id) == 'visibility_aware_efe'
        and str(expected_value('global_planner_mode') or 'efe') != 'preselected_route'
    ):
        network = expected_value('camera_network_artifact_path')
        if network:
            expected = _resolve_repo_path(str(network), strict=True)
            if _resolve_for_compare(manifest.get('camera_network_artifact_path', '')) != _resolve_for_compare(str(expected)):
                return False, 'camera network artifact path mismatch'
            if manifest.get('camera_network_artifact_sha256') != sha256_file(expected):
                return False, 'camera network artifact content hash mismatch'
            if manifest.get('camera_network_expected_sha256') != sha256_file(expected):
                return False, 'camera network consumer expectation mismatch'
            if not isinstance(manifest.get('camera_network_source_hashes'), dict) or not manifest.get(
                'camera_network_source_hashes'
            ):
                return False, 'camera network source provenance is missing'
            camera_ids = manifest.get('camera_network_camera_ids')
            if (not isinstance(camera_ids, list) or not camera_ids
                    or len(camera_ids) != len(set(camera_ids))):
                return False, 'camera network roster provenance is missing or malformed'
            expected_active = [
                value.strip() for value in str(
                    expected_value('camera_network_active_camera_ids') or '').split(',')
                if value.strip()
            ]
            if manifest.get('camera_network_active_camera_ids') != expected_active:
                return False, 'active planning-camera set mismatch'
            expected_objective = str(
                expected_value('camera_network_objective') or 'legacy_pixel_chart')
            if manifest.get('camera_network_objective') != expected_objective:
                return False, 'camera network objective mismatch'
            expected_goal_std = float(expected_value('network_goal_std_m') or 0.10)
            if float(manifest.get('network_goal_std_m', float('nan'))) != expected_goal_std:
                return False, 'camera network metric goal width mismatch'
            expected_updates = int(expected_value('camera_network_updates_per_step') or 1)
            if int(manifest.get('camera_network_updates_per_step', -1)) != expected_updates:
                return False, 'camera network update cadence mismatch'
            if manifest.get('visibility_artifact_path'):
                return False, 'network run unexpectedly also used a legacy visibility artifact'
            return True, ''
        actual = str(manifest.get('visibility_artifact_path', '') or '')
        expected = str(_resolve_repo_path(cfg['gp_artifact'], strict=False))
        if _resolve_for_compare(actual) != _resolve_for_compare(expected):
            return (
                False,
                f'visibility_artifact_path mismatch: run used {actual or "<missing>"}, config expects {expected}',
            )
    return True, ''


def _build_run_matrix(cfg: dict) -> list[tuple[str, str, int]]:
    """Return list of (task_name, condition_id, seed) in deterministic order."""
    runs = []
    for task_name, task_cfg in cfg['tasks'].items():
        for condition_id in task_cfg['conditions']:
            for seed in task_cfg['seeds']:
                runs.append((task_name, condition_id, seed))
    return runs


def _ros_domain_for_run(cfg: dict, run_idx: int) -> str | None:
    """Return the per-run ROS domain when campaign isolation is enabled."""
    if 'ros_domain_id_base' not in cfg:
        return None
    return str(int(cfg['ros_domain_id_base']) + int(run_idx))


def _build_launch_cmd(cfg: dict, task_name: str, condition_id: str, seed: int, log_dir: Path) -> list[str]:
    cfg = _resolved_cell_config(cfg, task_name, condition_id)
    planner = CONDITION_PLANNER[condition_id]
    global_mode = str(
        _effective_value(cfg, task_name, condition_id, 'global_planner_mode') or 'efe'
    ).strip().lower()
    gp_artifact = None
    network_artifact = _effective_value(cfg, task_name, condition_id, 'camera_network_artifact_path')
    if planner == 'visibility_aware_efe' and global_mode != 'preselected_route' and not network_artifact:
        gp_artifact = str(_resolve_repo_path(cfg['gp_artifact'], strict=True))
    yolo_model = str(_resolve_repo_path(cfg['yolo_model'], strict=True))
    odom_topic = str(cfg.get('odom_topic', '/odom_noisy'))
    if not _as_bool(cfg.get('use_encoder_noise', True)) and odom_topic == '/odom_noisy':
        odom_topic = '/odom'

    cmd = [
        'ros2', 'launch', 'experiments', str(cfg['launch_file']),
        f'world:={cfg["world"]}',
        f'task:={task_name}',
        f'planner:={planner}',
        f'seed:={seed}',
        f'log_dir:={log_dir}',
        f'outcome_journal_path:={log_dir / "detector_outcomes.jsonl"}',
        f'manager_outcome_journal_path:={log_dir / "manager_outcomes.jsonl"}',
        f'campaign_config_path:={cfg.get("_campaign_config_path", "")}',
        f'perception_backend:={cfg.get("perception_backend", "yolo")}',
        f'horizon:={cfg["horizon"]}',
        f'dt:={cfg["dt"]}',
        f'goal_success_radius:={cfg["goal_success_radius"]}',
        f'goal_success_hold_s:={cfg.get("goal_success_hold_s", 2.0)}',
        f'operational_belief_timeout_s:={cfg.get("operational_belief_timeout_s", 0.5)}',
        f'run_timeout_after_first_cmd_s:={cfg["run_timeout_after_first_cmd_s"]}',
        f'auto_stop_on_goal:=true',
        f'headless:={str(cfg.get("headless", False)).lower()}',
        f'nvidia_offload:={str(cfg.get("nvidia_offload", True)).lower()}',
        f'lockstep:={str(cfg.get("lockstep", False)).lower()}',
        f'lockstep_control_step_iterations:={cfg.get("lockstep_control_step_iterations", 100)}',
        f'lockstep_camera_every_control_steps:={cfg.get("lockstep_camera_every_control_steps", 2)}',
        f'lockstep_max_control_steps:={cfg.get("lockstep_max_control_steps", 0)}',
        f'use_rviz:={str(cfg.get("use_rviz", False)).lower()}',
        f'reset_world:={str(cfg.get("reset_world", False)).lower()}',
        f'r_visible_uv:={cfg.get("r_visible_uv", 2.5)}',
        f'r_miss_uv:={cfg.get("r_miss_uv", 120.0)}',
        f'discount_gamma:={cfg.get("discount_gamma", 0.995)}',
        f'v_max:={cfg.get("v_max", 0.22)}',
        f'max_predict_speed_mps:={cfg.get("max_predict_speed_mps", 0.0)}',
        f'state_correction_mode:={cfg.get("state_correction_mode", "fused")}',
        f'state_max_correction_jump_m:={cfg.get("state_max_correction_jump_m", 0.0)}',
        f'process_noise_xy:={cfg.get("process_noise_xy", 0.012)}',
        f'process_noise_theta:={cfg.get("process_noise_theta", 0.05)}',
        f'process_noise_model:={cfg.get("process_noise_model", "encoder")}',
        f'control_weight:={cfg.get("control_weight", 0.0)}',
        f'optimizer_maxiter:={cfg.get("optimizer_maxiter", 80)}',
        f'optimizer_maxfun:={cfg.get("optimizer_maxfun", 500)}',
        f'optimizer_multistart:={str(cfg.get("optimizer_multistart", False)).lower()}',
        f'optimizer_multistart_include_direct:={str(cfg.get("optimizer_multistart_include_direct", True)).lower()}',
        f'optimizer_terminal_goal_tolerance_m:={cfg.get("optimizer_terminal_goal_tolerance_m", 0.0)}',
        f'use_command_noise:={str(cfg.get("use_command_noise", True)).lower()}',
        f'use_encoder_noise:={str(cfg.get("use_encoder_noise", True)).lower()}',
        f'use_odom_for_predict:={str(cfg.get("use_odom_for_predict", True)).lower()}',
        f'odom_topic:={odom_topic}',
        f'command_noise_linear_slip_mean:={cfg.get("command_noise_linear_slip_mean", 0.03)}',
        f'command_noise_linear_slip_std:={cfg.get("command_noise_linear_slip_std", 0.06)}',
        f'command_noise_angular_slip_std:={cfg.get("command_noise_angular_slip_std", 0.04)}',
        f'command_noise_linear_additive_std:={cfg.get("command_noise_linear_additive_std", 0.008)}',
        f'command_noise_angular_additive_std:={cfg.get("command_noise_angular_additive_std", 0.035)}',
        f'command_noise_correlation_alpha:={cfg.get("command_noise_correlation_alpha", 0.85)}',
        f'encoder_noise_linear_slip_mean:={cfg.get("encoder_noise_linear_slip_mean", 0.0)}',
        f'encoder_noise_linear_slip_std:={cfg.get("encoder_noise_linear_slip_std", 0.125)}',
        f'encoder_noise_angular_slip_mean:={cfg.get("encoder_noise_angular_slip_mean", 0.0)}',
        f'encoder_noise_angular_slip_std:={cfg.get("encoder_noise_angular_slip_std", 0.075)}',
        f'encoder_noise_linear_additive_std:={cfg.get("encoder_noise_linear_additive_std", 0.004)}',
        f'encoder_noise_angular_additive_std:={cfg.get("encoder_noise_angular_additive_std", 0.05)}',
        f'encoder_noise_correlation_alpha:={cfg.get("encoder_noise_correlation_alpha", 0.80)}',
        f'encoder_wheel_diameter_ratio_error:={cfg.get("encoder_wheel_diameter_ratio_error", 0.00121)}',
        f'encoder_wheelbase_ratio:={cfg.get("encoder_wheelbase_ratio", 337.2 / 340.0)}',
        f'odom_heading_timeout_s:={cfg.get("odom_heading_timeout_s", 0.75)}',
        f'yolo_model:={yolo_model}',
        f'yolo_device:={cfg.get("yolo_device", "")}',
        f'yolo_imgsz:={cfg.get("yolo_imgsz", 640)}',
        f'yolo_conf_threshold:={cfg.get("yolo_conf_threshold", 0.25)}',
        f'yolo_predict_conf_floor:={cfg.get("yolo_predict_conf_floor", 0.05)}',
        f'yolo_iou_threshold:={cfg.get("yolo_iou_threshold", 0.45)}',
        f'yolo_target_class:={cfg.get("yolo_target_class", "robot")}',
        f'yolo_class_id:={cfg.get("yolo_class_id", -1)}',
        f'yolo_use_masks:={str(cfg.get("yolo_use_masks", True)).lower()}',
        f'yolo_min_mask_area_px:={cfg.get("yolo_min_mask_area_px", 12.0)}',
        f'yolo_mask_bottom_band_px:={cfg.get("yolo_mask_bottom_band_px", 3.0)}',
        f'yolo_min_bbox_area_px:={cfg.get("yolo_min_bbox_area_px", 0.0)}',
        f'yolo_max_batch_stamp_skew_s:={cfg.get("yolo_max_batch_stamp_skew_s", 0.05)}',
        f'yolo_debug_frame_dir:={cfg.get("yolo_debug_frame_dir", "")}',
        f'yolo_debug_crop_dir:={cfg.get("yolo_debug_crop_dir", "")}',
        f'yolo_use_torchscript:={str(cfg.get("yolo_use_torchscript", False)).lower()}',
        f'yolo_compiled_model:={cfg.get("yolo_compiled_model", "")}',
        f'yolo_warmup_iters:={cfg.get("yolo_warmup_iters", 3)}',
        f'yolo_inference_in_callback:={str(cfg.get("yolo_inference_in_callback", True)).lower()}',
        # 0 disables. Non-zero makes the detector report, at that period, how many
        # frames each camera delivered, what the batcher decided, and which cameras
        # each unfinished round is still waiting on -- the only way to see a batcher
        # that has silently stopped producing batches.
        f'yolo_runtime_trace_period_s:={cfg.get("yolo_runtime_trace_period_s", 0.0)}',
        # Recovery policy: what the belief does when a correction is refused or
        # cannot be replayed. Passed explicitly so it lands in the run manifest.
        f'state_reanchor_m:={cfg.get("state_reanchor_m", 0.0)}',
        f'state_max_predict_dt_s:={cfg.get("state_max_predict_dt_s", 1.5)}',
        f'state_reject_inflate_m2:={cfg.get("state_reject_inflate_m2", 0.0)}',
        f'stale_belief_inflate_m2_per_s:={cfg.get("stale_belief_inflate_m2_per_s", 0.0)}',
        f'stale_belief_inflate_cap_m2:={cfg.get("stale_belief_inflate_cap_m2", 0.0)}',
        f'require_state_correction_envelope:={str(cfg.get("require_state_correction_envelope", False)).lower()}',
    ]
    input_defaults = {
        'world_profiles': 'src/experiments/config/world_profiles.yaml',
        'tasks_yaml': 'src/experiments/config/tasks.yaml',
    }
    for argument_name, default_path in input_defaults.items():
        configured_path = cfg.get(argument_name, default_path)
        cmd.append(
            f'{argument_name}:='
            f'{_resolve_repo_path(str(configured_path), strict=True)}'
        )
    if global_mode == 'preselected_route':
        cmd.append(f'comparison_method_id:=closed_loop_{condition_id}')

    # Planner-specific args: pass GP artifact only for the visibility-aware
    # planner (C2). C1 (constant_R_efe) and C0 (geometric_shortest_path) are
    # camera-model-free and must not receive it.
    if gp_artifact is not None:
        cmd.append(f'visibility_artifact_path:={gp_artifact}')
    if network_artifact:
        cmd.append(f'camera_network_artifact_path:={_resolve_repo_path(str(network_artifact), strict=True)}')
        active_camera_ids = str(
            _effective_value(
                cfg, task_name, condition_id, 'camera_network_active_camera_ids') or '')
        if not active_camera_ids.strip():
            raise ValueError('camera-network runs require camera_network_active_camera_ids')
        cmd.append(f'camera_network_active_camera_ids:={active_camera_ids}')

    for key in (
        'observation_risk_scale', 'ambiguity_term_scale',
        'risk_weight_obs', 'ambiguity_weight',
        'camera_network_objective', 'network_goal_std_m', 'network_goal_std_start_m',
        'kouw_et1_ambiguity',
        'camera_network_updates_per_step',
        'optimizer_control_block_steps',
        'belief_publish_rate',
        'heading_update_mode',
        'use_pixel_correction', 'pixel_topic', 'command_noise_output_topic',
        'pixel_timeout_s', 'skip_stale_pixel_correction',
        'bev_y_calibration_offset_m', 'bev_affine_calibration', 'pixel_max_correction_jump_m',
        'pixel_correction_nis_threshold', 'use_diagnostic_odom_localization',
        'debug_runtime',
        'optimizer_ftol', 'optimizer_gtol', 'optimizer_warm_start',
        'optimizer_initial_routes_json',
        'optimizer_terminal_goal_tolerance_m',
        'optimizer_route_seed_mode',
        'driveable_geometry_json',
        'use_hierarchical', 'global_horizon', 'global_dt', 'local_horizon',
        'local_plan_rate', 'local_optimizer_maxiter',
        'global_use_ambiguity', 'local_use_ambiguity', 'local_use_obs_risk',
        'global_optimizer_multistart', 'local_optimizer_multistart',
        'local_use_visibility_model', 'local_use_belief_nogo_cost',
        'local_nogo_penalty_type',
        'local_nogo_safe_distance',
        'local_goal_prior_u_std_start', 'local_goal_prior_v_std_start',
        'local_goal_prior_u_std_final', 'local_goal_prior_v_std_final',
        'waypoint_spacing_m', 'waypoint_arrival_radius_m',
        'local_replan_min_remaining_s', 'local_replan_on_waypoint_change',
        'latency_compensate_plan_handoff',
        'local_controller_type',
        'cmd_publish_rate',
        'goal_prior_u_std_start', 'goal_prior_v_std_start',
        'goal_prior_u_std_final', 'goal_prior_v_std_final',
        'goal_tightening_power',
        'use_nogo_cost', 'nogo_mode', 'nogo_penalty_type',
        'nogo_safe_distance',
        'nogo_logbarrier_eps', 'nogo_warning_band', 'nogo_near_weight',
        'use_belief_nogo_cost',
        'nogo_belief_kappa',
        'use_hit_miss_mixture',
        'robot_collision_radius_m', 'robot_length_m', 'robot_width_m',
        'global_planner_mode',
        *PRESELECTED_ROUTE_KEYS,
        'bridge_camera_b', 'bridge_camera_c', 'bridge_camera_d',
        'multicam_belief', 'manager_gp_artifact_template',
        'manager_min_spatial_trust',
        'manager_decision_rate_hz',
        'manager_fusion_disagreement_gate_m',
        'manager_require_source_batch_id',
        'manager_bootstrap_min_cameras',
        'manager_bootstrap_max_disagreement_m',
        'manager_use_task_start_as_bootstrap_prior',
        'manager_bootstrap_prior_counts_as_support',
        'initial_belief_from_task_start', 'initial_belief_sigma_xy_m',
        'initial_belief_sigma_theta_rad',
        'manager_fusion_max_timestamp_spread_s',
        'manager_covariance_profile',
        'manager_commissioned_calibration_path', 'manager_commissioned_sigma_px',
        'manager_commissioned_world_covariance_path',
        'manager_commissioned_per_camera_sigma',
        'manager_fusion_common_mode_std_m',
        'manager_fusion_rule', 'manager_observation_model',
        'manager_sensor_gate_config_path',
        'manager_availability_model_path',
        'manager_availability_model_expected_sha256',
        'manager_learned_correction_path',
        'manager_visibility_sensor_model_path',
        'manager_visibility_sensor_model_expected_sha256',
        'manager_perception_sensor_model_path',
        'manager_perception_sensor_model_expected_sha256',
        'manager_correction_timestamp_compensation',
        'manager_correction_residual_interval_s',
        'manager_correction_propagation_drift_std',
        'manager_max_measurement_age_s', 'manager_age_decay_s',
        'manager_min_association_confidence',
        'manager_required_consecutive_better_frames',
        'manager_max_cross_camera_disagreement_m',
        'manager_require_consistency_when_source_available',
        'manager_camera_ids', 'manager_fusion_mode', 'manager_require_gp_artifacts',
        'state_correction_ekf',
        'wait_for_belief_before_first_goal', 'initial_belief_max_sigma_m',
        'multicam_scheduled', 'scheduled_coverage_artifact',
        'scheduled_report_std_m', 'scheduled_rate_hz',
        'stuck_window_s', 'stuck_max_displacement_m',
        'stuck_max_goal_improvement_m', 'stuck_cmd_fraction_min',
        'stuck_idle_cmd_fraction_max',
        'enable_mission', 'simple_tracker_yaw_gate_rad',
        'ff_fb_turn_rate_limit_rad_s', 'ff_fb_corner_crawl_speed_mps',
        'ff_fb_pivot_heading_error_rad',
    ):
        val = _effective_value(cfg, task_name, condition_id, key)
        if key == 'preselected_route_json' and val is not None:
            _points, val = canonicalize_polyline_json(str(val))
        elif key == 'preselected_route_source_path' and val is not None:
            val = str(_resolve_repo_path(str(val), strict=True))
        elif key in {
            'manager_commissioned_calibration_path',
            'manager_commissioned_world_covariance_path',
            'manager_sensor_gate_config_path',
            'manager_availability_model_path',
            'manager_learned_correction_path',
            'manager_visibility_sensor_model_path',
            'manager_perception_sensor_model_path',
            'scheduled_coverage_artifact',
        } and val:
            val = str(_resolve_repo_path(str(val), strict=True))
        elif key == 'manager_gp_artifact_template' and val:
            val = str(_resolve_repo_path(str(val), strict=False))
        if val is not None and not str(val).startswith('[FILL'):
            cmd.append(f'{key}:={val}')

    # Drop empty-valued launch args (e.g. yolo_device when unset): ros2 launch
    # rejects a bare 'name:=', and omitting it lets the launch file use its own
    # default (the behaviour earlier runs relied on).
    cmd = [arg for arg in cmd if not arg.endswith(':=')]
    return cmd


def _constant_strings(value) -> set[str]:
    strings = set()
    if isinstance(value, str):
        if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', value):
            strings.add(value)
    elif isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            strings.update(_constant_strings(item))
    elif hasattr(value, 'co_consts'):
        for item in value.co_consts:
            strings.update(_constant_strings(item))
    return strings


def _known_campaign_keys() -> set[str]:
    """Keys consumed by command construction plus explicit campaign metadata."""
    return (
        set(CAMPAIGN_METADATA_KEYS)
        | set(PRESELECTED_ROUTE_KEYS)
        | _constant_strings(_build_launch_cmd.__code__)
        | {
            'world', 'launch_file', 'yolo_model', 'horizon', 'dt',
            'goal_success_radius', 'run_timeout_after_first_cmd_s',
        }
    )


def _validate_explicit_scalar_types(values: dict, location: str) -> None:
    for key, value in values.items():
        if key in {'conditions', 'tasks', 'seeds', 'preselected_routes', 'label', 'planner'}:
            continue
        if value is None:
            raise ValueError(f'{location}: explicit {key} may not be null')
        if key in BOOL_CONFIG_KEYS and not isinstance(value, bool):
            raise ValueError(
                f'{location}: {key} must be a YAML boolean, got {value!r}'
            )


def _read_run_summary(run_dir: Path) -> dict | None:
    summary_path = run_dir / 'run_summary.json'
    if not summary_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        return summary if isinstance(summary, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _verify_outcome_journal(
    attempt_dir: Path, filename: str, producer_label: str
) -> tuple[bool, dict]:
    """Validate the durable producer close marker after launch shutdown."""
    journal_path = attempt_dir / filename
    if not journal_path.is_file():
        return False, {'reason': f'missing_{producer_label}_outcome_journal'}
    try:
        rows = list(read_journal(journal_path))
    except (OSError, ValueError) as exc:
        return False, {'reason': f'malformed_{producer_label}_outcome_journal:{exc}'}
    if not rows:
        return False, {'reason': f'empty_{producer_label}_outcome_journal'}
    epochs = {str(row.get('producer_epoch', '') or '') for row in rows}
    if len(epochs) != 1 or '' in epochs:
        return False, {'reason': f'{producer_label}_journal_epoch_mismatch'}
    if rows[-1].get('status') != 'session_stopped':
        return False, {'reason': f'{producer_label}_session_not_closed'}
    return True, {
        'reason': 'validated',
        'journal_path': str(journal_path),
        'journal_sha256': sha256_file(journal_path),
        'producer_epoch': next(iter(epochs)),
        'event_count': len(rows),
        'last_event_id': rows[-1].get('event_id'),
        'last_event_sha256': rows[-1].get('event_sha256'),
        'terminal_stop_request_id': rows[-1].get('terminal_stop_request_id'),
    }


def _verify_detector_journal(attempt_dir: Path) -> tuple[bool, dict]:
    return _verify_outcome_journal(
        attempt_dir, 'detector_outcomes.jsonl', 'detector'
    )


def _verify_correction_assimilations(run_dir: Path) -> tuple[bool, str]:
    """Every published fused correction must have one terminal filter outcome."""
    publications_path = run_dir / 'correction_publications.csv'
    observations_path = run_dir / 'fusion_observations.csv'
    assimilations_path = run_dir / 'correction_assimilations.csv'
    source_path = publications_path if publications_path.is_file() else observations_path
    schema_version = int(
        _load_run_manifest(run_dir).get('logging_schema_version', 0) or 0
    )
    if not source_path.is_file():
        return False, 'missing correction publication evidence'
    if not assimilations_path.is_file():
        return False, 'missing correction_assimilations.csv'
    try:
        with source_path.open(encoding='utf-8', newline='') as stream:
            publication_rows = list(csv.DictReader(stream))
        with assimilations_path.open(encoding='utf-8', newline='') as stream:
            assimilation_rows = list(csv.DictReader(stream))
    except (OSError, csv.Error) as exc:
        return False, f'cannot read correction evidence: {exc}'
    def exact_ns(value):
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        if isinstance(value, str) and re.fullmatch(r'0|[1-9][0-9]*', value):
            return int(value)
        return value

    def json_field(value):
        if not isinstance(value, str):
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    # correction_publications.csv is the identity-bearing envelope stream. Older
    # schema fixtures only have fusion_observations.csv, whose per-camera/display
    # rows are not publications; reduce that fallback to one batch identity.
    publications = []
    legacy_ids = set()
    for row in publication_rows:
        source_batch_id = row.get('source_batch_id')
        if source_path == observations_path and schema_version < 4:
            if source_batch_id in legacy_ids:
                continue
            legacy_ids.add(source_batch_id)
            publications.append(source_batch_id)
            continue
        publication = {'source_batch_id': source_batch_id}
        stamp = row.get('correction_stamp') if source_path == publications_path else row.get('fused_stamp')
        if stamp not in (None, ''):
            publication['correction_stamp'] = stamp
        if source_path == publications_path:
            publication['frame_id'] = row.get('frame_id')
            publication['event_id'] = row.get('event_id')
            publication['epoch'] = row.get('epoch')
            publication['publication_seq'] = exact_ns(row.get('publication_seq'))
            publication['correction_stamp_ns'] = exact_ns(row.get('correction_stamp_ns'))
            publication['member_ids'] = json_field(row.get('member_ids'))
            publication['payload_sha256'] = row.get('payload_sha256')
        else:
            payload_keys = (
                'fused_x', 'fused_y', 'fused_cov_xx', 'fused_cov_xy', 'fused_cov_yy'
            )
            if any(key in row for key in payload_keys):
                publication['payload'] = {
                    key: row.get(key) for key in payload_keys
                }
                try:
                    publication['payload'] = {
                        key: float(value)
                        for key, value in publication['payload'].items()
                    }
                except (TypeError, ValueError):
                    pass
        publications.append(publication)
    outcomes = []
    for row in assimilation_rows:
        status = str(row.get('status', '') or '').strip()
        outcome = {
            'source_batch_id': row.get('source_batch_id'),
            'status': status,
            'reason': row.get('reason'),
            'accepted': (
                row.get('accepted') if row.get('accepted') not in (None, '')
                else (status in {'accepted', 'accepted_bootstrap', 'reanchored'}
                      if schema_version < 4 else None)
            ),
        }
        for key in ('correction_stamp', 'apply_stamp'):
            if row.get(key) not in (None, ''):
                outcome[key] = row.get(key)
        for key in ('correction_stamp_ns', 'apply_stamp_ns'):
            if row.get(key) not in (None, ''):
                outcome[key] = exact_ns(row.get(key))
        if row.get('source_epoch') not in (None, ''):
            outcome['epoch'] = row.get('source_epoch')
        if row.get('source_event_id') not in (None, ''):
            outcome['event_id'] = row.get('source_event_id')
        if row.get('source_member_ids') not in (None, ''):
            outcome['member_ids'] = json_field(row.get('source_member_ids'))
        if row.get('source_payload_sha256') not in (None, ''):
            outcome['payload_sha256'] = row.get('source_payload_sha256')
        outcomes.append(outcome)
    require_timestamps = all(
        isinstance(record, dict)
        and record.get('correction_stamp') not in (None, '') for record in publications
    ) and all(
        record.get('correction_stamp') not in (None, '')
        and record.get('apply_stamp') not in (None, '') for record in outcomes
    )
    validation = validate_correction_ledger(
        publications, outcomes, require_timestamps=require_timestamps,
        allow_repeated_publications=True,
    )
    if not validation.valid:
        issue = validation.errors[0]
        return False, f'{issue.code}: {issue.message}'
    return True, ''


def _attempt_run_dir(attempt_dir: Path) -> Path | None:
    """Return the sole run created by one attempt; refuse ambiguous ownership."""
    if not attempt_dir.is_dir():
        return None
    candidates = [
        child for child in attempt_dir.iterdir()
        if child.is_dir() and child.name.startswith('experiment_')
    ]
    if len(candidates) > 1:
        raise RuntimeError(
            f'attempt directory contains multiple experiment runs: {attempt_dir}'
        )
    return candidates[0] if candidates else None


def _command_activity(run_dir: Path | None) -> tuple[int, bool]:
    """Return (experiment rows, whether any nonzero command has been logged)."""
    if run_dir is None:
        return 0, False
    path = run_dir / 'experiment.csv'
    if not path.is_file():
        return 0, False
    command_keys = ('cmd_v', 'cmd_w', 'cmd_raw_v', 'cmd_raw_w', 'exec_cmd_v', 'exec_cmd_w')
    rows = 0
    try:
        with path.open('r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                rows += 1
                for key in command_keys:
                    try:
                        value = float(row.get(key, 'nan'))
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(value) and abs(value) > 1e-6:
                        return rows, True
    except OSError:
        return rows, False
    return rows, False


def main() -> int:
    parser = argparse.ArgumentParser(description='Run a locked visibility-comparison campaign.')
    parser.add_argument('--config', default='scripts/visibility_comparison/warehouse_visibility_campaign.yaml',
                        help='Path to the locked campaign config YAML.')
    parser.add_argument('--log-root', default=str(LOGS_ROOT / 'warehouse_visibility_campaign_v1'),
                        help='Root directory for all run logs.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be run without executing.')
    parser.add_argument('--resume', action='store_true',
                        help='Skip runs already marked completed in campaign_log.json.')
    parser.add_argument(
        '--allow-imported-provenance', action='store_true',
        help='Permit completed ledger entries explicitly imported from an older campaign '
             'only when their recorded original executable provenance matches their run manifest. '
             'New attempts still require the current executable provenance.',
    )
    # Both caps are sized from solve times measured on 2026-09-15 with
    # optimizer_control_block_steps=1: 8 uncontended solves, median 96 s and
    # worst 238 s, and one solve sharing the machine with a live campaign at
    # 304 s. The previous 270/420 s pair was sized against a "contended tail to
    # ~220 s" that the 304 s measurement falsifies, so slow solves were being
    # guillotined with no command and scored infra_invalid.
    parser.add_argument('--run-timeout', type=float, default=900.0,
                        help='Wall-clock timeout per run including simulator startup (seconds). '
                             'Covers ~60s startup, the first-cmd solve cap, and ~40s of driving for the '
                             'longest route, with slack. Re-measure before lowering it.')
    parser.add_argument('--first-cmd-timeout', type=float, default=480.0,
                        help='Kill a live run when experiment.csv has rows but no nonzero command for this many seconds; <=0 disables. '
                             'Must exceed the worst-case global-solve wall time (304s measured under contention) or slow solves are '
                             'guillotined mid-optimization with no command (the dominant past failure mode).')
    parser.add_argument('--cleanup-delay', type=float, default=8.0,
                        help='Sleep between runs for process cleanup (seconds).')
    parser.add_argument('--planner-cache-dir', type=Path,
                        default=REPO_ROOT / 'logs/cache/casadi',
                        help='Reusable CasADi function cache; keys include model parameters and source hashes.')
    parser.add_argument('--no-planner-cache', action='store_true',
                        help='Build planner functions from scratch in each process.')
    parser.add_argument('--planner-jit', action='store_true',
                        help='Opt in to compiled objective/gradient evaluation; recorded for resume checks.')
    parser.add_argument('--max-new-runs-per-condition', type=int, default=0,
                        help='Smoke-test cap per condition; 0 runs the complete matrix. '
                             'A later --resume without this cap continues the campaign.')
    parser.add_argument('--only-condition', action='append', default=[],
                        help='Run only the named condition; repeat to select multiple conditions. '
                             'The complete campaign configuration is still validated and frozen.')
    parser.add_argument('--only-task', action='append', default=[],
                        help='Run only the named task; repeat to select multiple tasks. '
                             'The complete campaign configuration is still validated and frozen.')
    args = parser.parse_args()

    config_path = _resolve_repo_path(args.config, strict=False)
    if not config_path.is_file():
        print(f'ERROR: config file not found: {config_path}', file=sys.stderr)
        return 1

    cfg = _load_config(config_path)
    cfg.setdefault('operational_belief_timeout_s', 0.5)
    # Runtime provenance: the logger snapshots and hashes these exact bytes. Keep this
    # out-of-band key separate from the scientific YAML fields consumed by validation.
    cfg['_campaign_config_path'] = str(config_path)
    cfg['_campaign_config_sha256'] = sha256_file(config_path)
    cfg['_yolo_model_sha256'] = sha256_file(
        _resolve_repo_path(cfg['yolo_model'], strict=True)
    )
    cfg['_world_sdf_path'] = str(
        _verify_installed_world_matches_checkout(cfg['world'])
    )
    cfg['_world_sdf_sha256'] = sha256_file(cfg['_world_sdf_path'])
    cfg['_git_provenance'] = git_provenance(str(REPO_ROOT), EXECUTABLE_SOURCE_PATHS)
    log_root = Path(args.log_root).expanduser().resolve()
    campaign_log_path = log_root / 'campaign_log.json'
    ros_log_dir = Path(os.environ.get('ROS_LOG_DIR') or (log_root / '_ros_logs')).expanduser().resolve()
    if not args.dry_run:
        log_root.mkdir(parents=True, exist_ok=True)
        ros_log_dir.mkdir(parents=True, exist_ok=True)
    child_env = dict(os.environ)
    child_env['ROS_LOG_DIR'] = str(ros_log_dir)
    child_env['PYTHONPATH'] = _checkout_pythonpath()
    child_env['UNAV_EXECUTABLE_SOURCE_ROOT'] = str(REPO_ROOT)
    child_env['UNAV_EXECUTABLE_SOURCE_PATHS'] = os.pathsep.join(
        EXECUTABLE_SOURCE_PATHS
    )
    cache_dir = '' if args.no_planner_cache else str(args.planner_cache_dir.expanduser().resolve())
    child_env['UNAV_CASADI_CACHE_DIR'] = cache_dir
    child_env['UNAV_CASADI_JIT'] = '1' if args.planner_jit else '0'
    cfg['_execution_options'] = dict(planner_cache_dir=cache_dir, planner_jit=args.planner_jit)

    run_matrix = _build_run_matrix(cfg)
    if args.only_condition:
        requested_conditions = set(args.only_condition)
        unknown_conditions = requested_conditions.difference(cfg['conditions'])
        if unknown_conditions:
            raise ValueError(
                'unknown --only-condition value(s): '
                + ', '.join(sorted(unknown_conditions))
            )
        run_matrix = [
            cell for cell in run_matrix if cell[1] in requested_conditions
        ]
    if args.only_task:
        requested_tasks = set(args.only_task)
        unknown_tasks = requested_tasks.difference(cfg['tasks'])
        if unknown_tasks:
            raise ValueError(
                'unknown --only-task value(s): ' + ', '.join(sorted(unknown_tasks))
            )
        run_matrix = [cell for cell in run_matrix if cell[0] in requested_tasks]
    if not args.dry_run and cfg.get('ros_domain_id_base') is None:
        raise RuntimeError(
            'campaign execution requires ros_domain_id_base for scoped cleanup'
        )
    if 'ros_domain_id_base' in cfg:
        domain_base = int(cfg['ros_domain_id_base'])
        domain_max = domain_base + max(len(run_matrix) - 1, 0)
        if domain_base < 0 or domain_max > 232:
            raise RuntimeError(
                f'ros_domain_id_base={domain_base} with {len(run_matrix)} runs would use '
                f'ROS_DOMAIN_ID up to {domain_max}; expected range is 0..232.'
            )
    existing_log = _load_run_log(campaign_log_path) if args.resume else {}

    print(f'Campaign: {len(run_matrix)} runs total')
    print(f'Config: {config_path}')
    print(f'Log root: {log_root}')
    print(f'Campaign log: {campaign_log_path}')
    print(f'Planner function cache: {cache_dir or "disabled"}; JIT: {args.planner_jit}')
    print(f'ROS log dir: {ros_log_dir}')
    if cfg.get('cleanup_sim_stragglers', False):
        print(f'Gazebo cleanup: enabled for {cfg["world"]}')
    if args.dry_run:
        print('DRY RUN — no processes will be started.\n')

    isolated = True

    campaign_log = dict(existing_log)
    campaign_lease = None
    source_snapshot = None
    if not args.dry_run:
        campaign_lease = _exclusive_lease(
            log_root / '.campaign.lock',
            {'pid': os.getpid(), 'config': str(config_path),
             'started_at': datetime.now().isoformat()},
        )
        campaign_lease.__enter__()
        source_snapshot = _freeze_campaign_source(
            log_root, config_path, cfg['_git_provenance']
        )

    new_runs_by_condition = collections.Counter()
    for run_idx, (task_name, condition_id, seed) in enumerate(run_matrix):
        key = _run_key(task_name, condition_id, seed)
        label = f'[{run_idx + 1}/{len(run_matrix)}] task={task_name} condition={condition_id} seed={seed}'

        if args.resume and key in campaign_log and campaign_log[key].get('outcome') not in (None, 'infra_invalid'):
            matches, reason = _existing_entry_matches_config(
                campaign_log[key], cfg,
                expected_cell=(task_name, condition_id, seed),
                allow_imported_provenance=args.allow_imported_provenance,
            )
            if not matches:
                raise RuntimeError(
                    f'Cannot resume campaign with stale run entry for {label}: {reason}. '
                    'Start a fresh log root or rerun the campaign without --resume.'
                )
            print(f'  SKIP (already done): {label}')
            continue

        if (args.max_new_runs_per_condition > 0
                and new_runs_by_condition[condition_id] >= args.max_new_runs_per_condition):
            continue
        new_runs_by_condition[condition_id] += 1

        cell_log_dir = log_root / task_name / condition_id / f'seed{seed}'
        attempt_id = uuid.uuid4().hex
        run_log_dir = cell_log_dir / 'attempts' / attempt_id
        if not args.dry_run:
            run_log_dir.mkdir(parents=True, exist_ok=False)

        cmd = _build_launch_cmd(cfg, task_name, condition_id, seed, run_log_dir)
        ros_domain_id = _ros_domain_for_run(cfg, run_idx)
        print(f'\n{label}')
        if ros_domain_id is not None:
            print(f'  ROS_DOMAIN_ID: {ros_domain_id}')
        print('  CMD:', ' '.join(str(p) for p in cmd))

        if args.dry_run:
            continue

        if (git_provenance(str(REPO_ROOT), EXECUTABLE_SOURCE_PATHS)
                != cfg['_git_provenance']):
            raise RuntimeError(
                'executable checkout changed after campaign source identity was frozen '
                f'(scope: {", ".join(EXECUTABLE_SOURCE_PATHS)})'
            )

        run_entry: dict = {
            'task': task_name,
            'condition': condition_id,
            'seed': seed,
            'planner': CONDITION_PLANNER[condition_id],
            'global_planner_mode': str(
                _effective_value(
                    cfg, task_name, condition_id, 'global_planner_mode'
                ) or 'efe'
            ),
            'preselected_route_sha256': _effective_value(
                cfg, task_name, condition_id, 'preselected_route_sha256'
            ),
            'run_log_dir': str(run_log_dir),
            'attempt_id': attempt_id,
            'started_at': datetime.now().isoformat(),
            'outcome': None,
            'completion_reason': None,
            'goal_reached': None,
            'crashed': None,
            'path_length_m': None,
            'mean_belief_error_gt_m': None,
            'elapsed_after_first_cmd_s': None,
            'minimum_goal_distance': None,
            'ros_domain_id': ros_domain_id,
            'first_cmd_timeout_s': args.first_cmd_timeout,
            'execution_options': dict(cfg['_execution_options']),
            'source_snapshot': str(source_snapshot),
        }
        previous_entry = campaign_log.get(key)
        attempt_history = []
        if isinstance(previous_entry, dict):
            attempt_history.extend(previous_entry.get('attempts', []))
            previous_snapshot = {
                k: v for k, v in previous_entry.items() if k != 'attempts'
            }
            attempt_history.append(previous_snapshot)
        run_entry['attempts'] = attempt_history
        resolved_config = _resolved_cell_config(cfg, task_name, condition_id)
        artifact_paths = {
            'campaign_config': str(config_path),
            'world_sdf': cfg['_world_sdf_path'],
            'yolo_model': str(_resolve_repo_path(cfg['yolo_model'], strict=True)),
        }
        if cfg.get('route_selection_manifest_path'):
            artifact_paths['route_selection_manifest'] = str(_resolve_repo_path(
                cfg['route_selection_manifest_path'], strict=True
            ))
        for field in (
            'camera_network_artifact_path', 'manager_commissioned_calibration_path',
            'manager_commissioned_world_covariance_path',
            'manager_sensor_gate_config_path',
            'manager_learned_correction_path', 'preselected_route_source_path',
            'manager_visibility_sensor_model_path',
            'manager_perception_sensor_model_path',
        ):
            value = resolved_config.get(field)
            if value:
                artifact_paths[field] = str(_resolve_repo_path(str(value), strict=True))
        atomic_write_json(str(run_log_dir / 'attempt_manifest.json'), {
            'attempt_id': attempt_id,
            'cell': {'task': task_name, 'condition': condition_id, 'seed': seed},
            'command': cmd,
            'resolved_config': json.loads(json.dumps(resolved_config, default=str)),
            'artifacts': {
                name: {'path': path, 'sha256': sha256_file(path)}
                for name, path in artifact_paths.items()
            },
            'git_provenance': cfg['_git_provenance'],
            'source_snapshot': str(source_snapshot),
            'created_at': datetime.now().isoformat(),
        })
        campaign_log = _update_run_log(campaign_log_path, key, run_entry)

        run_env = dict(child_env)
        if ros_domain_id is not None:
            run_env['ROS_DOMAIN_ID'] = ros_domain_id
        run_token = f'unav_{os.getpid()}_{time.time_ns()}_{run_idx}'
        if isolated:
            run_env.update(UNAV_CAMPAIGN_RUN_TOKEN=run_token,
                           IGN_PARTITION=run_token, GZ_PARTITION=run_token)
            run_entry['transport_partition'] = run_token
            run_entry['cleanup_mode'] = 'isolated'
            campaign_log = _update_run_log(campaign_log_path, key, run_entry)

        domain_lease = _exclusive_lease(
            Path(tempfile.gettempdir()) / 'unav_campaign_leases'
            / f'ros_domain_{ros_domain_id}.lock',
            {'pid': os.getpid(), 'campaign_root': str(log_root),
             'attempt_id': attempt_id, 'run_token': run_token},
        )
        domain_lease.__enter__()
        process = None
        pgid = None
        process_returncode = None

        timed_out = False
        no_first_cmd_timeout = False
        spawn_error = ''
        started_at = time.monotonic()
        first_cmd_watch_started_at = None
        first_command_seen = False
        try:
            process = subprocess.Popen(cmd, start_new_session=True, env=run_env)
            pgid = os.getpgid(process.pid)
            while True:
                if process.poll() is not None:
                    process_returncode = process.returncode
                    break
                elapsed_wall = time.monotonic() - started_at
                if elapsed_wall >= args.run_timeout:
                    timed_out = True
                    print(f'  Wall-clock timeout after {args.run_timeout:.0f}s — killing.')
                    break
                if not first_command_seen and args.first_cmd_timeout > 0:
                    live_run_dir = _attempt_run_dir(run_log_dir)
                    rows, first_command_seen = _command_activity(live_run_dir)
                    if first_command_seen:
                        first_cmd_watch_started_at = None
                    elif rows > 0:
                        if first_cmd_watch_started_at is None:
                            first_cmd_watch_started_at = time.monotonic()
                        elif (time.monotonic() - first_cmd_watch_started_at) >= args.first_cmd_timeout:
                            no_first_cmd_timeout = True
                            print(
                                f'  INFRA INVALID: no nonzero command after '
                                f'{args.first_cmd_timeout:.0f}s of logged experiment rows — killing.'
                            )
                            break
                time.sleep(2.0)
        except OSError as exc:
            spawn_error = f'{type(exc).__name__}: {exc}'
        finally:
            if pgid is not None:
                _terminate_process_group(pgid)
            _cleanup_owned_run(run_token)
            domain_lease.__exit__(None, None, None)

        # Read run summary written by experiment_logger
        run_dir = _attempt_run_dir(run_log_dir)
        summary = _read_run_summary(run_dir) if run_dir else None
        route_artifact_ok = True
        route_artifact_reason = ''
        assimilation_ok = True
        assimilation_reason = ''
        multicam_value = _effective_value(
            cfg, task_name, condition_id, 'multicam_belief'
        )
        detector_journal_required = (
            _as_bool(multicam_value) if multicam_value is not None else False
        )
        detector_journal_ok = not detector_journal_required
        detector_journal_verdict = {
            'reason': 'not_required' if not detector_journal_required else 'not_checked'
        }
        manager_journal_ok = not detector_journal_required
        manager_journal_verdict = {
            'reason': 'not_required' if not detector_journal_required else 'not_checked'
        }
        if run_dir is not None:
            route_artifact_ok, route_artifact_reason = _verify_preselected_run_artifacts(
                run_dir, cfg, task_name, condition_id
            )
            assimilation_ok, assimilation_reason = _verify_correction_assimilations(run_dir)
        if detector_journal_required:
            detector_journal_ok, detector_journal_verdict = _verify_detector_journal(
                run_log_dir
            )
            manager_journal_ok, manager_journal_verdict = _verify_outcome_journal(
                run_log_dir, 'manager_outcomes.jsonl', 'manager'
            )
            terminal_stop_request_id = (
                summary.get('terminal_stop_request_id')
                if isinstance(summary, dict) else None
            )
            if detector_journal_ok and (
                detector_journal_verdict.get('terminal_stop_request_id')
                != terminal_stop_request_id
            ):
                detector_journal_ok = False
                detector_journal_verdict['reason'] = (
                    'detector_terminal_stop_request_identity_mismatch'
                )
            if manager_journal_ok and (
                manager_journal_verdict.get('terminal_stop_request_id')
                != terminal_stop_request_id
            ):
                manager_journal_ok = False
                manager_journal_verdict['reason'] = (
                    'manager_terminal_stop_request_identity_mismatch'
                )

        terminal_ok, terminal_outcome, terminal_reason = _terminal_summary_outcome(summary)
        if spawn_error:
            outcome = 'infra_invalid'
            completion_reason = 'spawn_error'
            print(f'  INFRA INVALID: {spawn_error}')
        elif timed_out:
            outcome = 'infra_invalid'
            completion_reason = 'wall_clock_timeout'
            print('  INFRA INVALID: campaign wall-clock timeout.')
        elif process_returncode not in (0, None):
            outcome = 'infra_invalid'
            completion_reason = 'launch_process_failed'
            print(f'  INFRA INVALID: launch exited with status {process_returncode}.')
        elif no_first_cmd_timeout:
            outcome = 'infra_invalid'
            completion_reason = 'no_first_cmd_timeout'
            print(f'  INFRA INVALID: live run never produced a nonzero command.')
        elif summary is None:
            outcome = 'infra_invalid'
            completion_reason = 'no_summary'
            print(f'  INFRA INVALID: no run_summary.json found in {run_log_dir}')
        elif not route_artifact_ok:
            outcome = 'infra_invalid'
            completion_reason = 'wrong_route_artifact'
            print(f'  INFRA INVALID: {route_artifact_reason}')
        elif not assimilation_ok:
            outcome = 'infra_invalid'
            completion_reason = 'correction_assimilation_invalid'
            print(f'  INFRA INVALID: {assimilation_reason}')
        elif not detector_journal_ok:
            outcome = 'infra_invalid'
            completion_reason = 'detector_journal_invalid'
            print(f'  INFRA INVALID: {detector_journal_verdict["reason"]}')
        elif not manager_journal_ok:
            outcome = 'infra_invalid'
            completion_reason = 'manager_journal_invalid'
            print(f'  INFRA INVALID: {manager_journal_verdict["reason"]}')
        elif not terminal_ok:
            outcome = 'infra_invalid'
            completion_reason = terminal_reason
            print(f'  INFRA INVALID: {terminal_reason}')
        else:
            outcome = terminal_outcome
            completion_reason = str(summary.get('completion_reason', ''))

        run_entry.update({
            'finished_at': datetime.now().isoformat(),
            'outcome': outcome,
            'completion_reason': completion_reason,
            'goal_reached': outcome == 'goal_reached',
            'crashed': bool(summary.get('crashed', False)) if summary else None,
            'path_length_m': summary.get('path_length_m') if summary else None,
            'mean_belief_error_gt_m': (summary.get('mean_belief_error_gt_after_first_cmd_m',
                                                    summary.get('mean_belief_error_gt_m')) if summary else None),
            'elapsed_after_first_cmd_s': summary.get('elapsed_after_first_cmd_s') if summary else None,
            'minimum_goal_distance': summary.get('minimum_goal_distance') if summary else None,
            'run_dir': str(run_dir) if run_dir else None,
            'route_artifact_verified': route_artifact_ok if run_dir else False,
            'route_artifact_verification_reason': route_artifact_reason,
            'correction_assimilation_verified': assimilation_ok if run_dir else False,
            'correction_assimilation_verification_reason': assimilation_reason,
            'process_returncode': process_returncode,
            'spawn_error': spawn_error or None,
            'detector_journal_required': detector_journal_required,
            'detector_journal_verified': detector_journal_ok,
            'detector_journal_sha256': detector_journal_verdict.get('journal_sha256'),
            'detector_journal_verification_reason': detector_journal_verdict.get('reason'),
            'manager_journal_verified': manager_journal_ok,
            'manager_journal_sha256': manager_journal_verdict.get('journal_sha256'),
            'manager_journal_verification_reason': manager_journal_verdict.get('reason'),
            'attempt_evidence_complete': outcome != 'infra_invalid',
        })
        atomic_write_json(str(run_log_dir / 'attempt_evidence_verdict.json'), {
            'attempt_id': attempt_id,
            'complete': outcome != 'infra_invalid',
            'outcome': outcome,
            'completion_reason': completion_reason,
            'run_dir': str(run_dir) if run_dir else None,
            'detector_journal': detector_journal_verdict,
            'manager_journal': manager_journal_verdict,
            'route_artifact_verified': route_artifact_ok if run_dir else False,
            'correction_assimilation_verified': assimilation_ok if run_dir else False,
            'written_at': datetime.now().isoformat(),
        })
        if outcome != 'infra_invalid':
            evidence_ok, evidence_reason = _existing_entry_matches_config(
                run_entry, cfg,
                expected_cell=(task_name, condition_id, seed),
            )
            if not evidence_ok:
                run_entry['terminal_completion_reason'] = completion_reason
                run_entry['outcome'] = outcome = 'infra_invalid'
                run_entry['completion_reason'] = completion_reason = 'evidence_identity_mismatch'
                run_entry['goal_reached'] = False
                run_entry['attempt_evidence_complete'] = False
                run_entry['evidence_validation_reason'] = evidence_reason
                print(f'  INFRA INVALID: {evidence_reason}')
                atomic_write_json(
                    str(run_log_dir / 'attempt_evidence_verdict.json'), {
                        'attempt_id': attempt_id,
                        'complete': False,
                        'outcome': outcome,
                        'completion_reason': completion_reason,
                        'terminal_completion_reason': run_entry.get(
                            'terminal_completion_reason'),
                        'run_dir': str(run_dir) if run_dir else None,
                        'detector_journal': detector_journal_verdict,
                        'manager_journal': manager_journal_verdict,
                        'route_artifact_verified': route_artifact_ok if run_dir else False,
                        'correction_assimilation_verified': assimilation_ok if run_dir else False,
                        'evidence_validation_reason': evidence_reason,
                        'written_at': datetime.now().isoformat(),
                    },
                )
        campaign_log = _update_run_log(campaign_log_path, key, run_entry)

        goal_str = 'YES' if outcome == 'goal_reached' else 'no'
        print(f'  -> outcome={outcome}, goal={goal_str}, reason={completion_reason}')

    if args.dry_run:
        print('\n=== Dry run complete ===')
    else:
        print('\n=== Campaign complete ===')
        total = len(run_matrix)
        completed = sum(1 for e in campaign_log.values() if e.get('outcome') not in (None, 'infra_invalid'))
        goals = sum(1 for e in campaign_log.values() if e.get('outcome') == 'goal_reached')
        infra = sum(1 for e in campaign_log.values() if e.get('outcome') == 'infra_invalid')
        print(f'  {completed}/{total} runs completed, {goals} goal_reached, {infra} infra_invalid')
        print(f'  Full log: {campaign_log_path}')
    if campaign_lease is not None:
        campaign_lease.__exit__(None, None, None)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
