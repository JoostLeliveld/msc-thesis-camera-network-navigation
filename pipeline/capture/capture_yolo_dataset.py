#!/usr/bin/env python3
"""Capture a YOLO-seg dataset from Gazebo semantic segmentation.

The capture path deliberately uses simulator-native semantic labels. Each
accepted sample must pass all checks:

* the robot has been teleported and a settle interval has elapsed;
* fresh RGB and semantic-label frames arrive after that settle interval;
* RGB and label headers are synchronized;
* the semantic mask is visible and not a tiny/truncated sliver;
* the mask lands near the commanded pose under the configured camera model;
* the robot is not substantially occluded by a foreground rack/box (the visible
  silhouette height and ground-contact row must match the projected robot box);
* the RGB crop under the label contains visible robot-colored pixels;
* the RGB frame is not an exact duplicate of an already accepted sample.

The script writes the YOLO dataset plus diagnostics, overlays, a contact sheet,
and a manifest. If a dataset cannot pass the final duplicate/acceptance checks,
it fails loudly instead of producing a plausible-looking but unusable dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Twist
from rclpy.node import Node
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from sensor_msgs.msg import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
for rel in ('src/experiments', 'src/perception', 'src/unav_common'):
    path = str((REPO_ROOT / rel).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)

from dataset_split_utils import assign_splits, build_pose_records, evenly_spaced_yaws
from experiments.core.world_profiles import compute_look_at_from_pose, load_profile
from perception.core.ros_image import image_msg_to_bgr8
from unav_common.camera_model import ObliqueCameraModel
from unav_common.occlusion_geometry import (
    profile_collision_scene,
    signed_distance_to_union_xy,
)


CAMERA_MODEL_TO_ID = {
    'external_camera': 'camera_A',
    **{
        f'external_camera_{suffix.lower()}': f'camera_{suffix}'
        for suffix in 'BCDEFGHIJKL'
    },
}

CAPTURE_INVOCATION_SCHEMA_VERSION = 'yolo_capture_invocation.v1'
SIMULATION_ASSET_INVENTORY_SCHEMA_VERSION = 'yolo_capture_simulation_assets.v1'
CAPTURE_STATE_SCHEMA_VERSION = 'yolo_capture_state.v1'
CAPTURE_COMPLETION_SCHEMA_VERSION = 'yolo_capture_completion.v1'
CAPTURE_TRANSPORT_SCHEMA_VERSION = 'yolo_capture_transport.v1'
# A dataset capture talks to a local simulator and ROS graph.  Leaving either
# transport on its host/network default allowed an unrelated Gazebo process to
# leak frames into a capture during commissioning.  Pin these values for every
# training-grade capture and retain the observed environment in the manifest.
TRAINING_CAPTURE_TRANSPORT_VALUES = {
    'ROS_LOCALHOST_ONLY': '1',
    'IGN_IP': '127.0.0.1',
    'GZ_IP': '127.0.0.1',
}
CAPTURE_TRANSPORT_VARIABLES = (
    *TRAINING_CAPTURE_TRANSPORT_VALUES,
    'ROS_DOMAIN_ID',
    'IGN_PARTITION',
)
_INVENTORY_IGNORED_PARTS = frozenset({'__pycache__'})
_INVENTORY_IGNORED_SUFFIXES = frozenset({'.pyc', '.pyo'})
_ACTIVE_CAPTURE_OUTPUT_GUARD: '_CaptureOutputGuard | None' = None


def _timestamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def _stamp_ns(msg: Image) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def _sha1_array(arr: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(arr)
    h = hashlib.sha1()
    h.update(str(contiguous.shape).encode('ascii'))
    h.update(str(contiguous.dtype).encode('ascii'))
    h.update(contiguous.tobytes())
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _logical_inventory_path(path: Path) -> str:
    """Return a relocation-stable repository path when one is available."""

    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _inventory_regular_files(path: Path) -> list[Path]:
    """Enumerate deterministic input files, rejecting absent/unsafe roots."""

    candidate = path.expanduser().resolve()
    if candidate.is_file():
        return [candidate]
    if not candidate.is_dir():
        raise RuntimeError(f'Required capture provenance asset does not exist: {candidate}')
    files: list[Path] = []
    for item in sorted(candidate.rglob('*'), key=lambda value: value.as_posix()):
        if any(part in _INVENTORY_IGNORED_PARTS for part in item.parts):
            continue
        if item.suffix.lower() in _INVENTORY_IGNORED_SUFFIXES:
            continue
        resolved = item.resolve()
        if resolved.is_file():
            files.append(resolved)
    if not files:
        raise RuntimeError(f'Required capture provenance asset directory is empty: {candidate}')
    return files


def _model_uris_from_xml(path: Path) -> set[str]:
    """Extract model:// references from a local SDF/config XML input."""

    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise RuntimeError(f'Cannot parse capture simulation asset {path}: {exc}') from exc
    uris: set[str] = set()
    for element in root.iter():
        if element.tag.rsplit('}', 1)[-1] not in {'uri', 'filename'}:
            continue
        text = str(element.text or '').strip()
        if text.startswith('model://'):
            uris.add(text)
    return uris


def _model_name_from_uri(uri: str) -> str:
    relative = str(uri)[len('model://'):].strip('/')
    model_name = relative.split('/', 1)[0]
    if not model_name or model_name in {'.', '..'}:
        raise RuntimeError(f'Invalid model URI in capture simulation asset: {uri!r}')
    return model_name


def _resolve_local_model_dir(model_name: str, sim_root: Path) -> Path:
    candidates = (
        sim_root / 'models' / model_name,
        sim_root / 'models_external' / model_name,
        REPO_ROOT / 'src' / 'sim' / 'models' / model_name,
        REPO_ROOT / 'src' / 'sim' / 'models_external' / model_name,
        REPO_ROOT / 'install' / 'sim' / 'share' / 'sim' / 'models' / model_name,
        REPO_ROOT / 'install' / 'sim' / 'share' / 'sim' / 'models_external' / model_name,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise RuntimeError(
        f'Cannot resolve local model://{model_name} for capture provenance; '
        f'searched {[str(value) for value in candidates]}'
    )


def _canonical_inventory_sha256(entries: Iterable[Mapping[str, Any]]) -> str:
    canonical = [
        {
            'logical_path': str(entry['logical_path']),
            'roles': sorted(str(role) for role in entry['roles']),
            'sha256': str(entry['sha256']),
            'size_bytes': int(entry['size_bytes']),
        }
        for entry in entries
    ]
    canonical.sort(key=lambda entry: (entry['logical_path'], entry['roles']))
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=True,
        allow_nan=False,
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _build_simulation_asset_inventory(
    *,
    world_path: Path,
    world_profiles_path: Path,
    route_exclusion_config_path: Path,
) -> dict[str, Any]:
    """Fingerprint the local simulator inputs that determine captured pixels.

    The world owns its inline geometry and declares model:// dependencies.  We
    recursively discover those local model packages and hash every file in
    each package so SDF, meshes, and textures are frozen together.  The
    standard sim launch and robot-description trees are included because the
    robot's rendered geometry and semantic label are capture inputs even though
    the robot is spawned outside this process.
    """

    world = world_path.expanduser().resolve()
    profiles = world_profiles_path.expanduser().resolve()
    routes = route_exclusion_config_path.expanduser().resolve()
    capture_script = Path(__file__).resolve()
    if not world.is_file():
        raise RuntimeError(f'Resolved capture world is not a file: {world}')

    # .../sim/gazebo_worlds/worlds/<world>.sdf -> .../sim
    try:
        sim_root = world.parents[2]
    except IndexError as exc:
        raise RuntimeError(f'Cannot derive sim asset root from world path: {world}') from exc
    launch_root = sim_root / 'launch'
    if not launch_root.is_dir():
        launch_root = REPO_ROOT / 'src' / 'sim' / 'launch'
    robot_description_root = sim_root / 'robot_description'
    if not robot_description_root.is_dir():
        robot_description_root = REPO_ROOT / 'src' / 'sim' / 'robot_description'

    paths_to_roles: dict[Path, set[str]] = {}

    def add(path: Path, role: str) -> None:
        for item in _inventory_regular_files(path):
            paths_to_roles.setdefault(item, set()).add(role)

    add(capture_script, 'capture_script')
    add(world, 'world')
    add(profiles, 'world_profiles')
    add(routes, 'route_exclusion_config')
    for launch_name in (
        'bringup_sim.launch.py',
        'gazebo.launch.py',
        'robot_description.launch.py',
    ):
        add(launch_root / launch_name, 'sim_launch')
    add(robot_description_root, 'robot_description')

    referenced_uris = set(_model_uris_from_xml(world))
    pending_models = {_model_name_from_uri(uri) for uri in referenced_uris}
    resolved_models: dict[str, Path] = {}
    while pending_models:
        model_name = sorted(pending_models)[0]
        pending_models.remove(model_name)
        if model_name in resolved_models:
            continue
        model_root = _resolve_local_model_dir(model_name, sim_root)
        resolved_models[model_name] = model_root
        add(model_root, f'model_asset:{model_name}')
        for xml_path in sorted(model_root.rglob('*'), key=lambda value: value.as_posix()):
            if xml_path.suffix.lower() not in {'.sdf', '.config'} or not xml_path.is_file():
                continue
            nested_uris = _model_uris_from_xml(xml_path)
            referenced_uris.update(nested_uris)
            pending_models.update(
                _model_name_from_uri(uri)
                for uri in nested_uris
                if _model_name_from_uri(uri) not in resolved_models
            )

    entries: list[dict[str, Any]] = []
    for path, roles in sorted(paths_to_roles.items(), key=lambda item: _logical_inventory_path(item[0])):
        stat_before = path.stat()
        digest = _sha256_file(path)
        stat_after = path.stat()
        if (
            stat_before.st_size != stat_after.st_size
            or stat_before.st_mtime_ns != stat_after.st_mtime_ns
        ):
            raise RuntimeError(f'Capture provenance asset changed while it was hashed: {path}')
        entries.append({
            'logical_path': _logical_inventory_path(path),
            'path': str(path),
            'roles': sorted(roles),
            'sha256': digest,
            'size_bytes': int(stat_after.st_size),
        })

    aggregate = _canonical_inventory_sha256(entries)
    return {
        'schema_version': SIMULATION_ASSET_INVENTORY_SCHEMA_VERSION,
        'aggregate_sha256': aggregate,
        'file_count': len(entries),
        'total_size_bytes': int(sum(int(entry['size_bytes']) for entry in entries)),
        'referenced_model_uris': sorted(referenced_uris),
        'referenced_model_names': sorted(resolved_models),
        'files': entries,
    }


def _jsonable_argument_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return 'Infinity' if value > 0.0 else '-Infinity'
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable_argument_value(item) for item in value]
    raise RuntimeError(f'Cannot preserve capture argument of type {type(value).__name__}: {value!r}')


def _capture_invocation(args: argparse.Namespace) -> dict[str, Any]:
    return {
        'schema_version': CAPTURE_INVOCATION_SCHEMA_VERSION,
        'python_executable': str(Path(sys.executable).expanduser().resolve()),
        'argv': [str(value) for value in sys.argv],
        'working_directory': str(Path.cwd().resolve()),
        'resolved_arguments': {
            key: _jsonable_argument_value(value)
            for key, value in sorted(vars(args).items())
        },
    }


def _capture_transport_environment(
    environment: Mapping[str, str] | None = None,
    *,
    allow_unisolated_transport: bool = False,
) -> dict[str, Any]:
    """Validate and preserve the local transport boundary for a capture.

    ``IGN_PARTITION`` is deliberately unique per capture invocation; it is not
    required to match across source cameras.  In contrast, the loopback values
    prevent another host/process from joining the simulation transport, while a
    declared ROS domain prevents accidental reuse of an ambient graph.  The
    override exists only to preserve a failed/diagnostic run for investigation;
    it records ``training_eligible=false`` and the merger refuses it.
    """

    values_source = os.environ if environment is None else environment
    observed = {
        key: str(values_source.get(key, '')).strip()
        for key in CAPTURE_TRANSPORT_VARIABLES
    }
    violations: list[str] = []
    for key, expected in TRAINING_CAPTURE_TRANSPORT_VALUES.items():
        if observed[key] != expected:
            violations.append(f'{key} must be {expected!r}, got {observed[key]!r}')

    domain_value = observed['ROS_DOMAIN_ID']
    try:
        domain_id = int(domain_value)
    except ValueError:
        domain_id = -1
    if not domain_value or not (0 <= domain_id <= 232):
        violations.append(
            'ROS_DOMAIN_ID must be an explicitly set integer in the inclusive range 0..232'
        )

    partition = observed['IGN_PARTITION']
    if not partition or any(character.isspace() for character in partition):
        violations.append('IGN_PARTITION must be a non-empty, whitespace-free token')

    isolation_verified = not violations
    if violations and not allow_unisolated_transport:
        raise RuntimeError(
            'Training-grade capture requires isolated local ROS/Gazebo transport: '
            + '; '.join(violations)
            + '. Set ROS_LOCALHOST_ONLY=1, IGN_IP=127.0.0.1, GZ_IP=127.0.0.1, '
            'ROS_DOMAIN_ID, and a unique IGN_PARTITION. '
            '--allow-unisolated-transport is diagnostic-only and cannot be merged for training.'
        )
    return {
        'schema_version': CAPTURE_TRANSPORT_SCHEMA_VERSION,
        'required_values': dict(TRAINING_CAPTURE_TRANSPORT_VALUES),
        'observed_values': observed,
        'isolation_verified': isolation_verified,
        'training_eligible': isolation_verified,
        'diagnostic_override_used': bool(violations and allow_unisolated_transport),
        'violations': violations,
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + '\n'
    try:
        with temporary.open('x', encoding='utf-8') as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _next_quarantine_path(output_dir: Path) -> Path:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base = output_dir.with_name(f'{output_dir.name}.failed_{timestamp}')
    candidate = base
    index = 1
    while candidate.exists():
        candidate = output_dir.with_name(f'{base.name}_{index:02d}')
        index += 1
    return candidate


class _CaptureOutputGuard:
    """Own an output until a success marker or atomic failure quarantine exists."""

    def __init__(
        self,
        output_dir: Path,
        *,
        camera_id: str,
        training_eligible: bool = True,
    ) -> None:
        self.output_dir = output_dir.resolve()
        self.camera_id = str(camera_id)
        self.training_eligible = bool(training_eligible)
        self.active = True
        self.quarantined_to: Path | None = None
        _atomic_write_json(
            self.output_dir / '.capture_in_progress.json',
            {
                'schema_version': CAPTURE_STATE_SCHEMA_VERSION,
                'status': 'in_progress',
                'training_eligible': False,
                'camera_id': self.camera_id,
                'output_dir': str(self.output_dir),
                'started_at': datetime.now().isoformat(),
            },
        )

    def quarantine(self, exc: BaseException) -> Path | None:
        if not self.active:
            return self.quarantined_to
        destination = _next_quarantine_path(self.output_dir)
        interrupted = isinstance(exc, (KeyboardInterrupt, SystemExit))
        state = {
            'schema_version': CAPTURE_STATE_SCHEMA_VERSION,
            'status': 'interrupted' if interrupted else 'failed',
            'training_eligible': False,
            'camera_id': self.camera_id,
            'failure_type': type(exc).__name__,
            'failure_reason': str(exc) or type(exc).__name__,
            'original_output_dir': str(self.output_dir),
            'quarantined_output_dir': str(destination),
            'failed_at': datetime.now().isoformat(),
            'recovery': (
                'Files are preserved for diagnosis only. Never merge or train from this '
                'directory; start a fresh capture at a new/original output path.'
            ),
        }
        try:
            if self.output_dir.is_dir():
                _atomic_write_json(self.output_dir / '.capture_failed.json', state)
                in_progress = self.output_dir / '.capture_in_progress.json'
                if in_progress.exists():
                    in_progress.unlink()
                os.replace(self.output_dir, destination)
                self.quarantined_to = destination
        finally:
            self.active = False
        return self.quarantined_to

    def complete(self, manifest_path: Path) -> None:
        if not self.active:
            raise RuntimeError('Capture output guard is no longer active')
        manifest_sha256 = _sha256_file(manifest_path)
        _atomic_write_json(
            self.output_dir / '.complete',
            {
                'schema_version': CAPTURE_COMPLETION_SCHEMA_VERSION,
                'status': 'complete',
                'training_eligible': self.training_eligible,
                'camera_id': self.camera_id,
                'dataset_manifest': 'dataset_manifest.json',
                'dataset_manifest_sha256': manifest_sha256,
                'completed_at': datetime.now().isoformat(),
            },
        )
        in_progress = self.output_dir / '.capture_in_progress.json'
        if in_progress.exists():
            in_progress.unlink()
        self.active = False


def _quarantine_active_capture(exc: BaseException) -> None:
    global _ACTIVE_CAPTURE_OUTPUT_GUARD
    guard = _ACTIVE_CAPTURE_OUTPUT_GUARD
    if guard is None or not guard.active:
        return
    try:
        destination = guard.quarantine(exc)
        if destination is not None:
            print(
                f'Capture did not complete; preserved diagnostic files in {destination}',
                file=sys.stderr,
                flush=True,
            )
    except Exception as quarantine_exc:  # pragma: no cover - last-resort reporting
        print(
            f'WARNING: failed to quarantine incomplete capture {guard.output_dir}: '
            f'{quarantine_exc}',
            file=sys.stderr,
            flush=True,
        )


def _read_uint_label_map(msg: Image) -> np.ndarray:
    encoding = str(msg.encoding or '').strip().lower()
    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)
    if height <= 0 or width <= 0:
        raise ValueError(f'Invalid label image shape height={height}, width={width}')

    raw = np.frombuffer(msg.data, dtype=np.uint8)
    expected = height * step
    if raw.size < expected:
        raise ValueError(f'Label image has {raw.size} bytes, expected at least {expected}')
    rows = raw[:expected].reshape(height, step)

    if encoding in ('rgb8', 'bgr8', '8uc3'):
        min_step = width * 3
        if step < min_step:
            raise ValueError(f'Label image step {step} too small for encoding {encoding}')
        pixels = rows[:, :min_step].reshape(height, width, 3)
        return pixels[..., 2].astype(np.uint32)

    if encoding in ('mono8', '8uc1'):
        min_step = width
        if step < min_step:
            raise ValueError(f'Label image step {step} too small for encoding {encoding}')
        return rows[:, :min_step].reshape(height, width).astype(np.uint32)

    if encoding in ('mono16', '16uc1', '16sc1'):
        min_step = width * 2
        if step < min_step:
            raise ValueError(f'Label image step {step} too small for encoding {encoding}')
        view = rows[:, :min_step].reshape(height, width, 2)
        return view.view(np.uint16).reshape(height, width).astype(np.uint32)

    if encoding in ('32sc1', '32uc1'):
        min_step = width * 4
        if step < min_step:
            raise ValueError(f'Label image step {step} too small for encoding {encoding}')
        view = rows[:, :min_step].reshape(height, width, 4)
        dtype = np.int32 if encoding == '32sc1' else np.uint32
        return view.view(dtype).reshape(height, width).astype(np.uint32)

    raise ValueError(
        f'Unsupported segmentation labels encoding {msg.encoding!r}; '
        'expected rgb8/bgr8/mono8/mono16/32sc1/32uc1'
    )


def _polygon_from_mask(mask_u8: np.ndarray, epsilon_ratio: float) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area <= 0.0:
        return None
    epsilon = max(1.0, float(epsilon_ratio) * cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, epsilon, True)
    pts = approx.reshape(-1, 2).astype(float)
    if pts.shape[0] < 3:
        return None
    return pts


def _write_seg_label(path: Path, polygon: np.ndarray | None, img_w: int, img_h: int) -> None:
    if polygon is None or len(polygon) < 3:
        path.write_text('', encoding='utf-8')
        return
    values = ['0']
    for x, y in polygon:
        values.append(f'{np.clip(x / float(img_w), 0.0, 1.0):.8f}')
        values.append(f'{np.clip(y / float(img_h), 0.0, 1.0):.8f}')
    path.write_text(' '.join(values) + '\n', encoding='utf-8')


def _mask_bbox(mask_u8: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.where(mask_u8 > 0)
    if xs.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def _mask_bottom(polygon: np.ndarray | None, band_px: float) -> tuple[float, float]:
    if polygon is None or len(polygon) < 3:
        return math.nan, math.nan
    v_bottom = float(np.max(polygon[:, 1]))
    band = polygon[polygon[:, 1] >= v_bottom - max(float(band_px), 0.0)]
    if band.size == 0:
        return float(np.mean(polygon[:, 0])), v_bottom
    return float(np.mean(band[:, 0])), v_bottom


def _border_fraction(mask_u8: np.ndarray, bbox: tuple[float, float, float, float]) -> float:
    h, w = mask_u8.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    if x1 <= x0 or y1 <= y0:
        return 1.0
    border = np.zeros_like(mask_u8, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    # Also treat bbox clipped to the image edge as border contact.
    if x0 <= 0 or y0 <= 0 or x1 >= w or y1 >= h:
        edge_contact = 1.0
    else:
        edge_contact = 0.0
    mask = mask_u8 > 0
    if not np.any(mask):
        return 1.0
    return max(float(np.count_nonzero(mask & border)) / float(np.count_nonzero(mask)), edge_contact)


#: The RGB support gate answers one question: do the pixels under the semantic label
#: actually look like the robot? It is what catches a label box that has drifted onto
#: the floor or a rack. The predicate is therefore a property of the ROBOT'S LIVERY and
#: cannot be hard-coded: the TurtleBot3 Burger was red-and-blue, while ``warehouse_amr``
#: is charcoal with a hazard-yellow band and no red or blue anywhere on it.
#:
#: Each profile lists the colour signatures that must ALL be present. A conjunction is
#: what makes the gate mean something in this warehouse: floor markings, rack guards and
#: rack-end protectors are hazard yellow too, so "some yellow is in the crop" is not
#: evidence of a robot -- but "yellow AND a dark body in the same crop" is, because the
#: painted floor has no dark body and a shadow has no yellow.
def _sig_red(b, g, r):
    return (r > 135) & (g < 135) & (b < 145) & ((r - g) > 30) & ((r - b) > 20)


def _sig_blue(b, g, r):
    return (b > 130) & (r < 155) & (g < 180) & ((b - r) > 20)


def _sig_marker_red_or_blue(b, g, r):
    return _sig_red(b, g, r) | _sig_blue(b, g, r)


def _sig_hazard_yellow(b, g, r):
    """Saturated warm yellow: the AMR's bumper band (0.85, 0.65, 0.05 linear)."""
    return (r > 120) & (g > 80) & ((r - b) > 55) & ((g - b) > 35) & (r >= g)


def _sig_dark_body(b, g, r):
    """Charcoal chassis and black sensor pods, which the lit concrete floor lacks."""
    mx = np.maximum(np.maximum(b, g), r)
    mn = np.minimum(np.minimum(b, g), r)
    return (mx < 105) & ((mx - mn) < 45)


ROBOT_COLOR_SIGNATURES = {
    'marker_red_or_blue': _sig_marker_red_or_blue,
    'hazard_yellow': _sig_hazard_yellow,
    'dark_body': _sig_dark_body,
}

#: The livery-agnostic alternative, and the reason it exists: a hand-tuned colour
#: predicate has to be re-tuned for every robot and every lighting change, and both
#: changed here at once. What the gate is really for is catching a label that does
#: not agree with the pixels -- a stale RGB frame from before the teleport, or a
#: label box that has drifted onto the floor. That can be tested without knowing
#: anything about the livery: the pixels INSIDE the semantic mask should not look
#: like the background immediately AROUND it. Measured per sample, so it adapts to
#: the local floor, the local shadow and the local paint by construction.
BACKGROUND_CONTRAST_SIGNATURE = 'background_contrast'
BACKGROUND_CONTRAST_RING_PX = 10
BACKGROUND_CONTRAST_MAHALANOBIS = 3.0


def _label_background_contrast(
    image_bgr: np.ndarray,
    mask_u8: np.ndarray,
    bbox: tuple[float, float, float, float],
    ring_px: int = BACKGROUND_CONTRAST_RING_PX,
) -> float:
    """Fraction of masked pixels whose colour is not plausible background.

    Background is estimated from a ring around the label's bounding box, so the
    comparison is local: a robot standing on bright concrete and the same robot in
    an aisle shadow are both measured against their own surroundings.
    """
    h, w = image_bgr.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    xi0, yi0 = max(0, x0), max(0, y0)
    xi1, yi1 = min(w, x1), min(h, y1)
    if xi1 <= xi0 or yi1 <= yi0:
        return 0.0
    xo0, yo0 = max(0, x0 - int(ring_px)), max(0, y0 - int(ring_px))
    xo1, yo1 = min(w, x1 + int(ring_px)), min(h, y1 + int(ring_px))
    outer = image_bgr[yo0:yo1, xo0:xo1].astype(float)
    ring_select = np.ones(outer.shape[:2], dtype=bool)
    ring_select[yi0 - yo0:yi1 - yo0, xi0 - xo0:xi1 - xo0] = False
    ring = outer[ring_select]
    inside_mask = mask_u8[yi0:yi1, xi0:xi1] > 0
    inside = image_bgr[yi0:yi1, xi0:xi1].astype(float)[inside_mask]
    if len(ring) < 30 or len(inside) < 10:
        return 0.0
    mu = np.median(ring, axis=0)
    # The +4 floor on the diagonal keeps a perfectly flat background (a synthetic
    # render of bare concrete has almost no colour variance) from making every
    # pixel infinitely far away.
    cov = np.cov(ring.T) + np.eye(3) * 4.0
    delta = inside - mu
    distance = np.sqrt(np.einsum('ij,jk,ik->i', delta, np.linalg.inv(cov), delta))
    return float(np.mean(distance > BACKGROUND_CONTRAST_MAHALANOBIS))

#: profile name -> the signatures that must all be present.
ROBOT_COLOR_PROFILES: dict[str, tuple[str, ...]] = {
    # TurtleBot3 Burger with the cyan/magenta marker disks (every dataset before
    # 2026-08-20). Kept so an old capture can be reproduced exactly.
    'marker_disks_red_blue': ('marker_red_or_blue',),
    # warehouse_amr by its livery. Measured 2026-08-21 on real warehouse_v2 frames:
    # the hazard band is only 0.2-5% of the crop at 12-19 m, and 'dark_body' also
    # fires on shadowed concrete, so this profile is kept as a diagnostic and is NOT
    # the default. Colour alone does not separate this robot from this floor.
    'amr_charcoal_hazard': ('dark_body', 'hazard_yellow'),
    # The default: livery-agnostic, and the only one that survives changing either
    # the robot or the lighting.
    'label_vs_background': (BACKGROUND_CONTRAST_SIGNATURE,),
}
DEFAULT_ROBOT_COLOR_PROFILE = 'label_vs_background'


def _robot_color_support_parts(
    image_bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
    profile: str,
    pad_px: int = 4,
    mask_u8: np.ndarray | None = None,
) -> dict[str, float]:
    """Fraction of the crop matching each signature the profile requires."""
    signatures = ROBOT_COLOR_PROFILES[profile]
    parts: dict[str, float] = {}
    if BACKGROUND_CONTRAST_SIGNATURE in signatures:
        parts[BACKGROUND_CONTRAST_SIGNATURE] = (
            _label_background_contrast(image_bgr, mask_u8, bbox)
            if mask_u8 is not None else 0.0
        )
        signatures = tuple(s for s in signatures if s != BACKGROUND_CONTRAST_SIGNATURE)
        if not signatures:
            return parts
    h, w = image_bgr.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    x0 = max(0, x0 - int(pad_px))
    y0 = max(0, y0 - int(pad_px))
    x1 = min(w, x1 + int(pad_px))
    y1 = min(h, y1 + int(pad_px))
    if x1 <= x0 or y1 <= y0:
        parts.update({name: 0.0 for name in signatures})
        return parts
    crop = image_bgr[y0:y1, x0:x1]
    b = crop[:, :, 0].astype(int)
    g = crop[:, :, 1].astype(int)
    r = crop[:, :, 2].astype(int)
    denominator = float(max(crop.shape[0] * crop.shape[1], 1))
    parts.update({
        name: float(np.count_nonzero(ROBOT_COLOR_SIGNATURES[name](b, g, r))) / denominator
        for name in signatures
    })
    return parts


def _robot_color_support_bgr(
    image_bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
    pad_px: int = 4,
    profile: str = DEFAULT_ROBOT_COLOR_PROFILE,
    mask_u8: np.ndarray | None = None,
) -> float:
    """The gate value: the WEAKEST required signature, so every one has to be present."""
    parts = _robot_color_support_parts(image_bgr, bbox, profile, pad_px=pad_px, mask_u8=mask_u8)
    return min(parts.values()) if parts else 0.0


def _project_world_point(camera: ObliqueCameraModel, xyz: np.ndarray) -> np.ndarray | None:
    cam_pt = camera.R @ (np.asarray(xyz, dtype=float) - camera.cam_pos)
    if cam_pt[2] <= 1e-6:
        return None
    pixel_h = camera.K @ cam_pt
    return np.asarray([pixel_h[0] / pixel_h[2], pixel_h[1] / pixel_h[2]], dtype=float)


def _project_robot_bbox(
    camera: ObliqueCameraModel,
    *,
    x: float,
    y: float,
    yaw: float,
    z: float,
    box_length: float,
    box_width: float,
    box_height: float,
) -> tuple[float, float, float, float] | None:
    """Projected support used only to audit synthetic detector labels.

    This geometry decides whether the rendered robot can support a valid training label.
    It is never used as a localization observation or correction input. The `box_*`
    arguments remain for the coarse-prism audit option.
    """
    return _project_robot_shape(camera, x=x, y=y, yaw=yaw, z=z, shape="visual_hull",
                                box_length=box_length, box_width=box_width,
                                box_height=box_height)


def _project_robot_shape(
    camera: ObliqueCameraModel,
    *,
    x: float,
    y: float,
    yaw: float,
    z: float,
    shape: str = "visual_hull",
    box_length: float = 0.80,
    box_width: float = 0.55,
    box_height: float = 0.35,
) -> tuple[float, float, float, float] | None:
    if shape == "visual_hull":
        from unav_common.robot_hull import VISUAL_HULL
        local = np.asarray(VISUAL_HULL, dtype=float)
    else:
        hx = 0.5 * float(box_length)
        hy = 0.5 * float(box_width)
        local = np.asarray([
            [-hx, -hy, 0.0],
            [-hx, hy, 0.0],
            [hx, -hy, 0.0],
            [hx, hy, 0.0],
            [-hx, -hy, box_height],
            [-hx, hy, box_height],
            [hx, -hy, box_height],
            [hx, hy, box_height],
        ], dtype=float)
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    rot = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    world = (rot @ local.T).T
    world[:, 0] += float(x)
    world[:, 1] += float(y)
    world[:, 2] += float(z)
    pts = []
    for corner in world:
        uv = _project_world_point(camera, corner)
        if uv is None:
            return None
        pts.append(uv)
    arr = np.asarray(pts, dtype=float)
    return (
        float(np.min(arr[:, 0])),
        float(np.min(arr[:, 1])),
        float(np.max(arr[:, 0])),
        float(np.max(arr[:, 1])),
    )


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return 0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return {'n': 0, 'min': None, 'median': None, 'mean': None, 'p90': None, 'max': None}
    arr = np.asarray(vals, dtype=float)
    return {
        'n': int(arr.size),
        'min': float(np.min(arr)),
        'median': float(np.median(arr)),
        'mean': float(np.mean(arr)),
        'p90': float(np.quantile(arr, 0.90)),
        'max': float(np.max(arr)),
    }


@dataclass
class FramePair:
    image_bgr: np.ndarray
    labels: np.ndarray
    image_stamp_ns: int
    label_stamp_ns: int
    image_count: int
    label_count: int
    stamp_delta_s: float
    set_pose_latency_s: float
    pair_wait_s: float


@dataclass
class LabelQuality:
    accepted: bool
    reason: str
    polygon: np.ndarray | None
    raw_mask: np.ndarray
    mask_area_px: float
    mask_bbox: tuple[float, float, float, float] | None
    mask_bbox_w: float
    mask_bbox_h: float
    mask_bottom_u: float
    mask_bottom_v: float
    border_fraction: float
    rgb_robot_color_fraction: float
    expected_bbox: tuple[float, float, float, float] | None
    expected_center_error_px: float
    visible_height_fraction: float
    bottom_occlusion_px: float
    localization_qualified: bool
    occlusion_state: str
    image_sha1: str
    label_sha1: str


def validate_sample_quality(
    *,
    image_bgr: np.ndarray,
    labels: np.ndarray,
    robot_label: int,
    epsilon_ratio: float,
    bottom_band_px: float,
    min_mask_area: float,
    min_mask_bbox_w: float,
    min_mask_bbox_h: float,
    max_mask_border_fraction: float,
    min_rgb_robot_color_fraction: float,
    disable_rgb_color_check: bool,
    robot_color_profile: str,
    camera: ObliqueCameraModel,
    x: float,
    y: float,
    yaw: float,
    robot_z: float,
    box_length: float,
    box_width: float,
    box_height: float,
    max_expected_center_error_px: float,
    min_visible_height_fraction: float,
    max_bottom_occlusion_px: float,
) -> LabelQuality:
    raw_mask = (labels == int(robot_label)).astype(np.uint8) * 255
    mask_area = float(cv2.countNonZero(raw_mask))
    bbox = _mask_bbox(raw_mask)
    polygon = _polygon_from_mask(raw_mask, float(epsilon_ratio)) if mask_area > 0 else None
    mask_bottom_u, mask_bottom_v = _mask_bottom(polygon, float(bottom_band_px))
    image_sha1 = _sha1_array(image_bgr)
    label_sha1 = _sha1_array(raw_mask)
    bbox_w = float(bbox[2] - bbox[0]) if bbox is not None else 0.0
    bbox_h = float(bbox[3] - bbox[1]) if bbox is not None else 0.0
    border_fraction = _border_fraction(raw_mask, bbox) if bbox is not None else 1.0
    rgb_support = (
        _robot_color_support_bgr(
            image_bgr, bbox, profile=str(robot_color_profile), mask_u8=raw_mask
        )
        if bbox is not None else 0.0
    )
    expected_bbox = _project_robot_bbox(
        camera,
        x=float(x),
        y=float(y),
        yaw=float(yaw),
        z=float(robot_z),
        box_length=float(box_length),
        box_width=float(box_width),
        box_height=float(box_height),
    )
    expected_center_error = math.nan
    # Occlusion metrics. gz semantic segmentation is rendering-based, so a robot
    # hidden behind a foreground rack/box yields a SHORTER visible silhouette than
    # the full projected bounding prism, and its visible bottom sits ABOVE where the
    # true ground contact projects. Both are measured in image (pixel) space, so they
    # are independent of the world-projection accuracy under investigation.
    #  - visible_height_fraction: visible mask height / expected projected height.
    #    Drops toward 0 as the robot is occluded.
    #  - bottom_occlusion_px: expected contact row - visible mask bottom row.
    #    Positive & large => the ground-contact (box-bottom label) is occluded.
    visible_height_fraction = math.nan
    bottom_occlusion_px = math.nan
    if bbox is not None and expected_bbox is not None:
        cx, cy = _bbox_center(bbox)
        ex, ey = _bbox_center(expected_bbox)
        expected_center_error = float(math.hypot(cx - ex, cy - ey))
        expected_h = float(expected_bbox[3] - expected_bbox[1])
        if expected_h > 1e-6:
            visible_height_fraction = float(bbox_h / expected_h)
        if math.isfinite(mask_bottom_v):
            bottom_occlusion_px = float(expected_bbox[3] - mask_bottom_v)

    reason = ''
    localization_qualified = True
    occlusion_state = 'clear'
    if (math.isfinite(visible_height_fraction)
            and visible_height_fraction < float(min_visible_height_fraction)):
        localization_qualified = False
        occlusion_state = 'low_visible_height'
    if (math.isfinite(bottom_occlusion_px)
            and bottom_occlusion_px > float(max_bottom_occlusion_px)):
        localization_qualified = False
        occlusion_state = 'bottom_hidden'
    if mask_area < float(min_mask_area):
        reason = 'small_mask'
    elif bbox is None:
        reason = 'empty_mask'
    elif polygon is None:
        reason = 'empty_polygon'
    elif bbox_w < float(min_mask_bbox_w) or bbox_h < float(min_mask_bbox_h):
        reason = 'small_bbox'
    elif border_fraction > float(max_mask_border_fraction):
        reason = 'touches_image_border'
    elif expected_bbox is None:
        reason = 'commanded_pose_not_projectable'
    elif expected_center_error > float(max_expected_center_error_px):
        reason = 'projection_mismatch'
    elif (math.isfinite(visible_height_fraction)
          and visible_height_fraction < float(min_visible_height_fraction)):
        reason = 'occluded_low_visible_height'
    elif (math.isfinite(bottom_occlusion_px)
          and bottom_occlusion_px > float(max_bottom_occlusion_px)):
        reason = 'occluded_bottom_hidden'
    elif (not bool(disable_rgb_color_check)) and rgb_support < float(min_rgb_robot_color_fraction):
        reason = 'rgb_robot_not_visible'

    return LabelQuality(
        accepted=(reason == ''),
        reason=reason,
        polygon=polygon,
        raw_mask=raw_mask,
        mask_area_px=mask_area,
        mask_bbox=bbox,
        mask_bbox_w=bbox_w,
        mask_bbox_h=bbox_h,
        mask_bottom_u=float(mask_bottom_u),
        mask_bottom_v=float(mask_bottom_v),
        border_fraction=float(border_fraction),
        rgb_robot_color_fraction=float(rgb_support),
        expected_bbox=expected_bbox,
        expected_center_error_px=float(expected_center_error),
        visible_height_fraction=float(visible_height_fraction),
        bottom_occlusion_px=float(bottom_occlusion_px),
        localization_qualified=bool(localization_qualified),
        occlusion_state=str(occlusion_state),
        image_sha1=image_sha1,
        label_sha1=label_sha1,
    )


class TeleportYoloDatasetCapture(Node):
    def __init__(
        self,
        *,
        world_name: str,
        image_topic: str,
        labels_topic: str,
        robot_z: float,
        settle_s: float,
        image_timeout_s: float,
        sync_slop_s: float,
        min_new_rgb_frames: int,
        min_new_label_frames: int,
        buffer_size: int,
        zero_cmd_topic: str,
    ):
        super().__init__('capture_yolo_dataset')
        self.world_name = str(world_name)
        self.robot_z = float(robot_z)
        self.settle_s = float(settle_s)
        self.image_timeout_s = float(image_timeout_s)
        self.sync_slop_s = float(max(sync_slop_s, 0.0))
        self.min_new_rgb_frames = max(int(min_new_rgb_frames), 1)
        self.min_new_label_frames = max(int(min_new_label_frames), 1)
        self.service_name = f'/world/{self.world_name}/set_pose'
        self.client = self.create_client(SetEntityPose, self.service_name)
        self.rgb_count = 0
        self.label_count = 0
        self.image_buffer = deque(maxlen=max(int(buffer_size), 4))
        self.label_buffer = deque(maxlen=max(int(buffer_size), 4))
        self.last_label_error = ''
        self.zero_cmd_pub = self.create_publisher(Twist, str(zero_cmd_topic), 1)
        self.create_subscription(Image, str(image_topic), self._image_cb, 10)
        self.create_subscription(Image, str(labels_topic), self._labels_cb, 10)

    def _image_cb(self, msg: Image) -> None:
        self.rgb_count += 1
        self.image_buffer.append((int(self.rgb_count), _stamp_ns(msg), image_msg_to_bgr8(msg)))

    def _labels_cb(self, msg: Image) -> None:
        try:
            labels = _read_uint_label_map(msg)
        except Exception as exc:
            self.last_label_error = str(exc)
            self.get_logger().error(f'Failed to decode segmentation labels image: {exc}')
            return
        self.label_count += 1
        self.label_buffer.append((int(self.label_count), _stamp_ns(msg), labels))

    def _publish_zero_cmd(self, repetitions: int = 3) -> None:
        msg = Twist()
        for _ in range(max(int(repetitions), 1)):
            self.zero_cmd_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.01)

    def wait_for_ready(self, timeout_s: float) -> None:
        end = time.monotonic() + max(float(timeout_s), 0.0)
        service_ready = False
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if (not service_ready) and self.client.wait_for_service(timeout_sec=0.05):
                service_ready = True
            if service_ready and self.image_buffer and self.label_buffer:
                return
        extra = f' Last label decode error: {self.last_label_error}' if self.last_label_error else ''
        raise RuntimeError(
            f'Timed out waiting for {self.service_name}, RGB image, and segmentation labels. '
            'Launch sim bringup with bridge_segmentation:=true for dataset capture.'
            + extra
        )

    def _set_robot_pose(self, *, x: float, y: float, yaw: float) -> float:
        self._publish_zero_cmd()
        req = SetEntityPose.Request()
        req.entity = Entity(name='turtlebot3')
        req.entity.type = Entity.MODEL
        req.pose.position.x = float(x)
        req.pose.position.y = float(y)
        req.pose.position.z = float(self.robot_z)
        half = 0.5 * float(yaw)
        req.pose.orientation.z = math.sin(half)
        req.pose.orientation.w = math.cos(half)
        future = self.client.call_async(req)
        start = time.monotonic()
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.05)
            if (time.monotonic() - start) > 10.0:
                raise RuntimeError(f'Timed out waiting for {self.service_name} response')
        result = future.result()
        latency_s = float(time.monotonic() - start)
        if result is None or not bool(getattr(result, 'success', False)):
            raise RuntimeError(
                f'Set pose request failed at x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}'
            )
        self._publish_zero_cmd()
        return latency_s

    def _find_synced_pair(self, before_rgb: int, before_label: int) -> FramePair | None:
        if self.rgb_count < before_rgb + self.min_new_rgb_frames:
            return None
        if self.label_count < before_label + self.min_new_label_frames:
            return None
        image_candidates = [item for item in self.image_buffer if item[0] > before_rgb]
        label_candidates = [item for item in self.label_buffer if item[0] > before_label]
        best = None
        best_key = None
        for label_count, label_stamp, labels in label_candidates:
            for image_count, image_stamp, image in image_candidates:
                delta_s = abs(image_stamp - label_stamp) * 1e-9
                if delta_s > self.sync_slop_s:
                    continue
                key = (min(image_count, label_count), -delta_s)
                if best_key is None or key > best_key:
                    best_key = key
                    best = (image_count, image_stamp, image, label_count, label_stamp, labels, delta_s)
        if best is None:
            return None
        image_count, image_stamp, image, label_count, label_stamp, labels, delta_s = best
        return FramePair(
            image_bgr=image.copy(),
            labels=labels.copy(),
            image_stamp_ns=int(image_stamp),
            label_stamp_ns=int(label_stamp),
            image_count=int(image_count),
            label_count=int(label_count),
            stamp_delta_s=float(delta_s),
            set_pose_latency_s=math.nan,
            pair_wait_s=math.nan,
        )

    def capture_at_pose(self, *, x: float, y: float, yaw: float) -> FramePair:
        set_pose_latency_s = self._set_robot_pose(x=x, y=y, yaw=yaw)

        settle_end = time.monotonic() + max(self.settle_s, 0.0)
        while rclpy.ok() and time.monotonic() < settle_end:
            self._publish_zero_cmd(repetitions=1)
            rclpy.spin_once(self, timeout_sec=0.05)

        # The freshness baseline is after settling, not before teleport. This is
        # what prevents accepting frames rendered while the robot was still being
        # moved to the new pose.
        before_rgb = int(self.rgb_count)
        before_label = int(self.label_count)
        wait_start = time.monotonic()
        deadline = wait_start + max(self.image_timeout_s, 0.0)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            pair = self._find_synced_pair(before_rgb, before_label)
            if pair is not None:
                pair.set_pose_latency_s = set_pose_latency_s
                pair.pair_wait_s = float(time.monotonic() - wait_start)
                return pair
        latest_delta = math.nan
        if self.image_buffer and self.label_buffer:
            latest_delta = abs(self.image_buffer[-1][1] - self.label_buffer[-1][1]) * 1e-9
        raise RuntimeError(
            'No fresh synchronized RGB/segmentation pair after teleport settle '
            f'(rgb_count={self.rgb_count}, label_count={self.label_count}, '
            f'latest_delta={latest_delta:.3f}s, required <= {self.sync_slop_s:.3f}s).'
        )


def _in_any_region(x: float, y: float, regions: list[dict], shrink_m: float) -> bool:
    for r in regions:
        if (
            float(r['xmin']) + shrink_m <= x <= float(r['xmax']) - shrink_m
            and float(r['ymin']) + shrink_m <= y <= float(r['ymax']) - shrink_m
        ):
            return True
    return False


def _in_expanded_region(x: float, y: float, region: dict, expansion_m: float) -> bool:
    return (
        float(region['xmin']) - expansion_m <= x <= float(region['xmax']) + expansion_m
        and float(region['ymin']) - expansion_m <= y <= float(region['ymax']) + expansion_m
    )


def _filter_pose_records(
    records: list[dict[str, float | int]],
    *,
    traversable_regions: list[dict],
    excluded_regions: list[dict],
    region_shrink_m: float,
    collision_prisms: tuple,
    collision_clearance_m: float,
    camera_xy: tuple[float, float],
    min_camera_range_m: float,
    max_camera_range_m: float,
    exclusion_segments: tuple[tuple[float, float, float, float], ...] = (),
    route_exclusion_buffer_m: float = 0.0,
) -> tuple[list[dict[str, float | int]], dict[str, int]]:
    """Apply deterministic geometry/range filters to the common pose grid.

    Split labels are assigned on the complete common A--D grid before this
    camera-dependent filter runs.  A record is kept only when its ground
    position is in a declared traversable region, outside every known
    non-driveable region and parsed collision prism (with robot clearance),
    and in the requested camera-range stratum.
    """

    counts: Counter[str] = Counter()
    kept: list[dict[str, float | int]] = []
    clearance = max(float(collision_clearance_m), 0.0)
    minimum_range = max(float(min_camera_range_m), 0.0)
    maximum_range = float(max_camera_range_m)
    route_buffer = max(float(route_exclusion_buffer_m), 0.0)
    for record in records:
        x = float(record['x'])
        y = float(record['y'])
        if any(
            _point_segment_distance(x, y, *segment) <= route_buffer
            for segment in exclusion_segments
        ):
            counts['evaluation_route_exclusion'] += 1
            continue
        if traversable_regions and not _in_any_region(x, y, traversable_regions, float(region_shrink_m)):
            counts['outside_traversable_region'] += 1
            continue
        if any(_in_expanded_region(x, y, region, clearance) for region in excluded_regions):
            counts['known_non_driveable_region'] += 1
            continue
        if collision_prisms:
            distance = float(
                signed_distance_to_union_xy(
                    collision_prisms,
                    np.asarray([[x, y]], dtype=float),
                    keep_in=False,
                )[0]
            )
            if distance <= clearance:
                counts['collision_clearance'] += 1
                continue
        camera_range = math.hypot(x - float(camera_xy[0]), y - float(camera_xy[1]))
        if camera_range < minimum_range:
            counts['below_min_camera_range'] += 1
            continue
        if math.isfinite(maximum_range) and camera_range > maximum_range:
            counts['above_max_camera_range'] += 1
            continue
        kept.append(record)
    counts['kept'] = len(kept)
    return kept, dict(counts)


def _filter_projectable_pose_records(
    records: list[dict[str, float | int]],
    *,
    camera: ObliqueCameraModel,
    image_width: int,
    image_height: int,
    robot_z: float,
    box_length: float,
    box_width: float,
    box_height: float,
) -> tuple[list[dict[str, float | int]], dict[str, int]]:
    """Remove commanded poses whose projected robot box misses the image.

    This is a calibration-geometry planning filter, not an oracle label: it
    never reads rendered RGB, semantic labels, or evaluation truth.  Border-
    clipped and occluded poses remain eligible as long as the nominal box has
    non-zero image intersection, preserving hard detector examples while
    avoiding repeated captures of robots that are deterministically off-screen.
    """

    width = int(image_width)
    height = int(image_height)
    if width <= 0 or height <= 0:
        raise ValueError('image dimensions must be positive')
    kept: list[dict[str, float | int]] = []
    counts: Counter[str] = Counter()
    for record in records:
        bbox = _project_robot_bbox(
            camera,
            x=float(record['x']),
            y=float(record['y']),
            yaw=float(record['yaw']),
            z=float(robot_z),
            box_length=float(box_length),
            box_width=float(box_width),
            box_height=float(box_height),
        )
        if bbox is None or not all(math.isfinite(value) for value in bbox):
            counts['projection_behind_camera'] += 1
            continue
        x0, y0, x1, y1 = bbox
        intersects = x1 > 0.0 and y1 > 0.0 and x0 < float(width) and y0 < float(height)
        if not intersects:
            counts['projection_outside_image'] += 1
            continue
        kept.append(record)
    counts['kept'] = len(kept)
    return kept, dict(counts)


def _negative_no_opportunity_reason(
    record: Mapping[str, float | int],
    *,
    camera: ObliqueCameraModel,
    image_width: int,
    image_height: int,
    robot_z: float,
    box_length: float,
    box_width: float,
    box_height: float,
) -> str:
    """Certify a negative only when the nominal robot cannot enter the image.

    Occlusion is deliberately not considered: a robot hidden by a rack is not
    a background negative. It remains a detector opportunity/hard positive.
    """

    bbox = _project_robot_bbox(
        camera,
        x=float(record['x']),
        y=float(record['y']),
        yaw=float(record['yaw']),
        z=float(robot_z),
        box_length=float(box_length),
        box_width=float(box_width),
        box_height=float(box_height),
    )
    if bbox is None or not all(math.isfinite(value) for value in bbox):
        return 'projection_behind_camera'
    x0, y0, x1, y1 = bbox
    intersects = (
        x1 > 0.0
        and y1 > 0.0
        and x0 < float(image_width)
        and y0 < float(image_height)
    )
    return '' if intersects else 'projection_outside_image'


def _point_segment_distance(
    x: float,
    y: float,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> float:
    dx = float(x1) - float(x0)
    dy = float(y1) - float(y0)
    denominator = dx * dx + dy * dy
    if denominator <= 1.0e-12:
        return math.hypot(float(x) - float(x0), float(y) - float(y0))
    t = ((float(x) - float(x0)) * dx + (float(y) - float(y0)) * dy) / denominator
    t = min(max(t, 0.0), 1.0)
    return math.hypot(float(x) - (float(x0) + t * dx), float(y) - (float(y0) + t * dy))


def _route_exclusion_segments(
    study_path: Path,
    *,
    route_names: list[str],
) -> tuple[tuple[float, float, float, float], ...]:
    if not route_names:
        return ()
    payload = yaml.safe_load(study_path.read_text(encoding='utf-8')) or {}
    routes = {
        str(item.get('name')): item
        for item in payload.get('collection', {}).get('routes', [])
        if isinstance(item, dict) and item.get('name')
    }
    offsets = [float(value) for value in payload.get('collection', {}).get('lateral_offsets_m', [0.0])]
    segments: list[tuple[float, float, float, float]] = []
    for route_name in route_names:
        if route_name not in routes:
            raise RuntimeError(
                f'Unknown --exclude-route {route_name!r}; available: {", ".join(sorted(routes))}'
            )
        route = routes[route_name]
        x0 = float(route['start']['x'])
        y0 = float(route['start']['y'])
        x1 = float(route['goal']['x'])
        y1 = float(route['goal']['y'])
        dx = x1 - x0
        dy = y1 - y0
        length = math.hypot(dx, dy)
        if length <= 1.0e-9:
            raise RuntimeError(f'Excluded route {route_name!r} has coincident endpoints')
        left_x = -dy / length
        left_y = dx / length
        for offset in offsets:
            segments.append((
                x0 + offset * left_x,
                y0 + offset * left_y,
                x1 + offset * left_x,
                y1 + offset * left_y,
            ))
    return tuple(segments)


def _camera_range_bin(range_m: float) -> str:
    if range_m < 8.0:
        return 'lt_8m'
    if range_m < 12.0:
        return '8_to_12m'
    if range_m <= 16.0:
        return '12_to_16m'
    return 'gt_16m'


def _mask_area_gate_for_range(
    range_m: float,
    *,
    near_min_area_px: float,
    far_start_m: float,
    far_min_area_px: float,
) -> float:
    """Return the pre-declared small-object mask gate for a ground range."""

    if float(range_m) >= float(far_start_m):
        return max(float(far_min_area_px), 0.0)
    return max(float(near_min_area_px), 0.0)


def _camera_from_profile(camera_pose: list[float], intrinsics: dict) -> ObliqueCameraModel:
    cam_pos = [float(v) for v in camera_pose[:3]]
    rpy = [float(v) for v in camera_pose[3:6]]
    look_at = compute_look_at_from_pose(cam_pos, *rpy)
    return ObliqueCameraModel(
        cam_pos=cam_pos,
        look_at=look_at,
        img_width=int(intrinsics['img_width']),
        img_height=int(intrinsics['img_height']),
        fov_h_rad=float(intrinsics['fov_h_rad']),
    )


def _draw_overlay(
    image_bgr: np.ndarray,
    quality: LabelQuality,
    *,
    text: str,
) -> np.ndarray:
    out = image_bgr.copy()
    if quality.polygon is not None:
        pts = quality.polygon.astype(np.int32).reshape(-1, 1, 2)
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], (0, 180, 0))
        out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0.0)
        cv2.polylines(out, [pts], True, (0, 220, 0), 2)
    if quality.mask_bbox is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in quality.mask_bbox]
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 2)
    if quality.expected_bbox is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in quality.expected_bbox]
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 0, 0), 2)
    if math.isfinite(quality.mask_bottom_u) and math.isfinite(quality.mask_bottom_v):
        cv2.circle(out, (int(round(quality.mask_bottom_u)), int(round(quality.mask_bottom_v))), 5, (0, 255, 255), -1)
    cv2.putText(out, text[:150], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, text[:150], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def _write_contact_sheet(paths: list[Path], output_path: Path, *, cols: int = 5, tile_w: int = 320) -> str:
    if not paths:
        return ''
    imgs = []
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = tile_w / float(max(w, 1))
        tile_h = max(1, int(round(h * scale)))
        imgs.append(cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA))
    if not imgs:
        return ''
    tile_h = max(img.shape[0] for img in imgs)
    rows = int(math.ceil(len(imgs) / float(cols)))
    sheet = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    sheet[:, :] = (24, 24, 24)
    for idx, img in enumerate(imgs):
        y = (idx // cols) * tile_h
        x = (idx % cols) * tile_w
        sheet[y:y + img.shape[0], x:x + img.shape[1]] = img
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)
    return str(output_path)


def _archive_existing_path(path: Path) -> Path:
    archive_root = path.parent / 'archive'
    archive_root.mkdir(parents=True, exist_ok=True)
    target = archive_root / f'{path.name}_{_timestamp()}'
    shutil.move(str(path), str(target))
    return target


def _fixed_fieldnames() -> list[str]:
    return [
        'camera_id', 'camera_model', 'camera_range_m', 'camera_range_bin', 'min_mask_area_applied_px',
        'sample_kind', 'sample_index', 'attempt', 'split', 'accepted', 'rejection_reason',
        'geometry_certified_no_opportunity', 'no_opportunity_reason', 'semantic_robot_pixels',
        'image', 'label', 'mask', 'preview', 'observation_image',
        'robot_x', 'robot_y', 'robot_yaw',
        'image_stamp_s', 'label_stamp_s', 'stamp_delta_s',
        'set_pose_latency_s', 'settle_s', 'pair_wait_s',
        'image_count', 'label_count',
        'mask_area_px', 'mask_bbox_x0', 'mask_bbox_y0', 'mask_bbox_x1', 'mask_bbox_y1',
        'mask_bbox_w', 'mask_bbox_h', 'mask_bottom_u', 'mask_bottom_v',
        'border_fraction', 'rgb_robot_color_fraction',
        'expected_bbox_x0', 'expected_bbox_y0', 'expected_bbox_x1', 'expected_bbox_y1',
        'expected_center_error_px', 'visible_height_fraction', 'bottom_occlusion_px',
        'localization_qualified', 'occlusion_state',
        'image_sha1', 'label_sha1',
    ]


def _diagnostic_row(
    *,
    sample_index: int,
    attempt: int,
    split: str,
    accepted: bool,
    reason: str,
    pair: FramePair | None,
    quality: LabelQuality | None,
    x: float,
    y: float,
    yaw: float,
    settle_s: float,
    sample_kind: str = 'positive',
    geometry_certified_no_opportunity: bool = False,
    no_opportunity_reason: str = '',
    semantic_robot_pixels: int | None = None,
    image_rel: str = '',
    label_rel: str = '',
    mask_rel: str = '',
    preview_rel: str = '',
    observation_image_rel: str = '',
) -> dict:
    row = {
        'sample_kind': str(sample_kind),
        'sample_index': int(sample_index),
        'attempt': int(attempt),
        'split': str(split),
        'accepted': int(bool(accepted)),
        'rejection_reason': str(reason),
        'geometry_certified_no_opportunity': int(bool(geometry_certified_no_opportunity)),
        'no_opportunity_reason': str(no_opportunity_reason),
        'semantic_robot_pixels': (
            int(semantic_robot_pixels) if semantic_robot_pixels is not None else math.nan
        ),
        'image': image_rel,
        'label': label_rel,
        'mask': mask_rel,
        'preview': preview_rel,
        'observation_image': observation_image_rel,
        'robot_x': float(x),
        'robot_y': float(y),
        'robot_yaw': float(yaw),
        'image_stamp_s': math.nan,
        'label_stamp_s': math.nan,
        'stamp_delta_s': math.nan,
        'set_pose_latency_s': math.nan,
        'settle_s': float(settle_s),
        'pair_wait_s': math.nan,
        'image_count': math.nan,
        'label_count': math.nan,
        'mask_area_px': math.nan,
        'mask_bbox_x0': math.nan,
        'mask_bbox_y0': math.nan,
        'mask_bbox_x1': math.nan,
        'mask_bbox_y1': math.nan,
        'mask_bbox_w': math.nan,
        'mask_bbox_h': math.nan,
        'mask_bottom_u': math.nan,
        'mask_bottom_v': math.nan,
        'border_fraction': math.nan,
        'rgb_robot_color_fraction': math.nan,
        'expected_bbox_x0': math.nan,
        'expected_bbox_y0': math.nan,
        'expected_bbox_x1': math.nan,
        'expected_bbox_y1': math.nan,
        'expected_center_error_px': math.nan,
        'visible_height_fraction': math.nan,
        'bottom_occlusion_px': math.nan,
        'localization_qualified': 0,
        'occlusion_state': 'unknown',
        'image_sha1': '',
        'label_sha1': '',
    }
    if pair is not None:
        row.update({
            'image_stamp_s': float(pair.image_stamp_ns) * 1e-9,
            'label_stamp_s': float(pair.label_stamp_ns) * 1e-9,
            'stamp_delta_s': float(pair.stamp_delta_s),
            'set_pose_latency_s': float(pair.set_pose_latency_s),
            'pair_wait_s': float(pair.pair_wait_s),
            'image_count': int(pair.image_count),
            'label_count': int(pair.label_count),
        })
    if quality is not None:
        bbox = quality.mask_bbox or (math.nan, math.nan, math.nan, math.nan)
        expected = quality.expected_bbox or (math.nan, math.nan, math.nan, math.nan)
        row.update({
            'mask_area_px': float(quality.mask_area_px),
            'mask_bbox_x0': float(bbox[0]),
            'mask_bbox_y0': float(bbox[1]),
            'mask_bbox_x1': float(bbox[2]),
            'mask_bbox_y1': float(bbox[3]),
            'mask_bbox_w': float(quality.mask_bbox_w),
            'mask_bbox_h': float(quality.mask_bbox_h),
            'mask_bottom_u': float(quality.mask_bottom_u),
            'mask_bottom_v': float(quality.mask_bottom_v),
            'border_fraction': float(quality.border_fraction),
            'rgb_robot_color_fraction': float(quality.rgb_robot_color_fraction),
            'expected_bbox_x0': float(expected[0]),
            'expected_bbox_y0': float(expected[1]),
            'expected_bbox_x1': float(expected[2]),
            'expected_bbox_y1': float(expected[3]),
            'expected_center_error_px': float(quality.expected_center_error_px),
            'visible_height_fraction': float(quality.visible_height_fraction),
            'bottom_occlusion_px': float(quality.bottom_occlusion_px),
            'localization_qualified': int(quality.localization_qualified),
            'occlusion_state': str(quality.occlusion_state),
            'image_sha1': quality.image_sha1,
            'label_sha1': quality.label_sha1,
        })
        row['semantic_robot_pixels'] = int(quality.mask_area_px)
    return row


def main() -> int:
    global _ACTIVE_CAPTURE_OUTPUT_GUARD
    parser = argparse.ArgumentParser(description='Capture a YOLO-seg dataset from Gazebo semantic segmentation labels.')
    parser.add_argument('--world', default='warehouse_aws.world.sdf')
    parser.add_argument('--world-profiles', default=str((REPO_ROOT / 'src' / 'experiments' / 'config' / 'world_profiles.yaml').resolve()))
    parser.add_argument('--out', default='', help='Output dataset folder; defaults under logs/')
    parser.add_argument('--plan-only', action='store_true',
                        help='Validate provenance inputs and print the filtered capture plan without creating output or requiring ROS topics.')
    parser.add_argument('--replace', action='store_true', help='Delete an existing output folder before capture.')
    parser.add_argument('--archive-existing', action='store_true', help='Move an existing output folder under sibling archive/ before capture.')
    parser.add_argument('--sample-nx', type=int, default=16)
    parser.add_argument('--sample-ny', type=int, default=14)
    parser.add_argument('--yaw-samples', type=int, default=8)
    parser.add_argument('--wall-margin', type=float, default=0.65)
    parser.add_argument('--sample-min-x', type=float, default=None,
                        help='Optional camera-specific sampling bound; defaults to the world profile bound plus wall margin.')
    parser.add_argument('--sample-max-x', type=float, default=None)
    parser.add_argument('--sample-min-y', type=float, default=None)
    parser.add_argument('--sample-max-y', type=float, default=None)
    parser.add_argument('--val-fraction', type=float, default=0.20)
    parser.add_argument('--split-mode', choices=('cyclic', 'yaw_bucket', 'spatial_cell', 'spatial_yaw_bucket'), default='spatial_cell',
                        help='Grouped train/val split. spatial_cell (default) keeps every heading at one x/y in one split.')
    parser.add_argument('--spatial-block-size', type=int, default=2)
    parser.add_argument('--split-seed', type=int, default=0)
    # warehouse_amr, the platform since 2026-08-20: 0.80 x 0.55 m body, deck top at
    # 0.35 m, base_footprint on the floor. The previous defaults (0.22 x 0.22 x 0.20)
    # were the TurtleBot3 Burger and would project a prism a quarter of the right size.
    parser.add_argument('--robot-z', type=float, default=0.0)
    parser.add_argument('--robot-box-length', type=float, default=0.80)
    parser.add_argument('--robot-box-width', type=float, default=0.55)
    parser.add_argument('--robot-box-height', type=float, default=0.35)
    parser.add_argument('--settle-s', type=float, default=0.80)
    parser.add_argument('--image-timeout-s', type=float, default=8.0)
    parser.add_argument('--sync-slop-ms', type=float, default=60.0)
    parser.add_argument('--min-new-rgb-frames', type=int, default=3)
    parser.add_argument('--min-new-label-frames', type=int, default=1)
    parser.add_argument('--max-sample-attempts', type=int, default=4)
    parser.add_argument('--buffer-size', type=int, default=90)
    parser.add_argument('--preview-count', type=int, default=160)
    parser.add_argument('--rejected-preview-count', type=int, default=80)
    parser.add_argument('--save-masks', action='store_true', help='Save accepted binary masks for provenance.')
    parser.add_argument(
        '--save-observation-frames',
        action='store_true',
        help=(
            'Retain one synchronized RGB frame for every successfully captured planned pose, '
            'including samples rejected by the semantic-label quality gate. Accepted rows '
            'reuse their normal image path; rejected rows are written under observation_frames/. '
            'This is for detector miss/error characterization, not detector training.'
        ),
    )
    parser.add_argument('--min-mask-area', type=float, default=80.0)
    parser.add_argument('--far-range-start-m', type=float, default=12.0,
                        help='Ground range at which the explicit small-object mask threshold applies.')
    parser.add_argument('--far-min-mask-area', type=float, default=40.0,
                        help='Minimum native-resolution visible-mask pixels at/above --far-range-start-m.')
    parser.add_argument('--min-mask-bbox-w', type=float, default=8.0)
    parser.add_argument('--min-mask-bbox-h', type=float, default=8.0)
    parser.add_argument('--max-mask-border-fraction', type=float, default=0.0)
    parser.add_argument('--bottom-band-px', type=float, default=3.0)
    parser.add_argument('--epsilon-ratio', type=float, default=0.010)
    parser.add_argument('--min-rgb-robot-color-fraction', type=float, default=0.0,
                        help=('Minimum support for EVERY signature the livery profile '
                              'requires. DEFAULT 0 = measured and recorded per sample, but '
                              'not enforced. It is a diagnostic rather than a gate because '
                              'the failure it was invented for -- a label map that does not '
                              'match the RGB frame under it -- is already prevented twice by '
                              'construction: only frames captured AFTER the teleport are '
                              'candidates (--min-new-rgb-frames), and the RGB and label '
                              'headers must agree within --sync-slop-ms. A third check that '
                              'has to be re-tuned for every robot livery and every lighting '
                              'change is a false-reject risk without a matching benefit: the '
                              'red-or-blue version silently discarded every sample of '
                              'warehouse_amr. Set it above 0 to enforce.'))
    parser.add_argument('--robot-color-profile', default=DEFAULT_ROBOT_COLOR_PROFILE,
                        choices=sorted(ROBOT_COLOR_PROFILES),
                        help=('Which livery the RGB support gate should look for. '
                              'marker_disks_red_blue reproduces every capture made before '
                              '2026-08-20 (TurtleBot3 Burger with marker disks); '
                              'amr_charcoal_hazard is warehouse_amr.'))
    parser.add_argument('--disable-rgb-color-check', action='store_true')
    parser.add_argument('--max-expected-center-error-px', type=float, default=90.0)
    parser.add_argument('--min-visible-height-fraction', type=float, default=0.55,
                        help='Reject occluded samples whose visible mask height is below this '
                             'fraction of the projected robot-box height. Set 0.0 to disable.')
    parser.add_argument('--max-bottom-occlusion-px', type=float, default=20.0,
                        help='Reject samples whose visible mask bottom sits more than this many '
                             'pixels above the projected ground-contact row (bottom occluded by a '
                             'foreground rack/box). Set a large value to disable.')
    parser.add_argument(
        '--occlusion-policy',
        choices=('reject', 'visible-mask-positive'),
        default='reject',
        help=(
            'reject keeps a localization-qualified dataset. visible-mask-positive also writes '
            'rendered partial silhouettes as detector positives but marks them localization_qualified=0.'
        ),
    )
    parser.add_argument('--max-final-duplicate-fraction', type=float, default=0.02)
    parser.add_argument('--min-accepted-samples', type=int, default=400)
    parser.add_argument('--min-accept-fraction', type=float, default=0.25)
    parser.add_argument(
        '--negative-samples-per-camera',
        type=int,
        default=0,
        help=(
            'Opt-in count of unique train-only no-opportunity background frames. '
            'A negative requires a fresh synchronized pair, zero semantic robot pixels, '
            'and a nominal robot box fully outside the image. Existing positive-only '
            'smoke commands remain unchanged at the default 0.'
        ),
    )
    parser.add_argument('--image-topic', default='/external_camera/image_raw')
    parser.add_argument('--labels-topic', default='/external_camera/segmentation/labels_map')
    parser.add_argument('--zero-cmd-topic', default='/cmd_vel')
    parser.add_argument('--robot-label', type=int, default=23)
    parser.add_argument('--skip-region-filter', action='store_true',
                        help='Disable known_2d_regions traversability filter even when the profile defines regions.')
    parser.add_argument('--region-shrink-m', type=float, default=0.05,
                        help='Shrink each traversable region boundary inward before testing grid points.')
    parser.add_argument('--collision-clearance-m', type=float, default=0.25,
                        help='Reject commanded poses this close to parsed collision geometry/non-driveable regions.')
    parser.add_argument('--skip-collision-filter', action='store_true',
                        help='Disable parsed-world collision filtering (unsafe; intended only for geometry debugging).')
    parser.add_argument('--skip-projection-filter', action='store_true',
                        help='Disable the nominal camera-box/image intersection prefilter (debug only).')
    parser.add_argument('--min-camera-range-m', type=float, default=0.0)
    parser.add_argument('--max-camera-range-m', type=float, default=float('inf'))
    parser.add_argument('--camera-model', default='external_camera',
                        help='World include whose pose defines projection checks (external_camera or external_camera_b..l).')
    parser.add_argument('--camera-id', default='',
                        help='Stable provenance ID; inferred as camera_A..camera_L for known camera models.')
    parser.add_argument('--allow-topic-model-mismatch', action='store_true',
                        help='Allow image/label topics outside the selected camera-model namespace.')
    parser.add_argument('--exclude-route', action='append', default=[],
                        help='Study route to reserve for detector testing; repeat for multiple routes.')
    parser.add_argument(
        '--route-exclusion-config',
        type=Path,
        default=REPO_ROOT / 'pipeline/capture/multicamera_study.yaml',
    )
    parser.add_argument('--route-exclusion-buffer-m', type=float, default=0.75)
    parser.add_argument(
        '--allow-unisolated-transport',
        action='store_true',
        help=(
            'Permit a non-isolated capture for diagnosis only. Its manifest and completion '
            'marker are marked non-training-eligible and the four-camera merger rejects it.'
        ),
    )
    args = parser.parse_args()

    transport_environment = _capture_transport_environment(
        allow_unisolated_transport=bool(args.allow_unisolated_transport),
    )

    camera_model = str(args.camera_model).strip()
    if not camera_model:
        raise RuntimeError('--camera-model must not be empty')
    camera_id = str(args.camera_id).strip() or CAMERA_MODEL_TO_ID.get(camera_model, camera_model)
    expected_topic_prefix = f'/{camera_model}/'
    if not bool(args.allow_topic_model_mismatch):
        mismatched = [
            topic for topic in (str(args.image_topic), str(args.labels_topic))
            if not topic.startswith(expected_topic_prefix)
        ]
        if mismatched:
            raise RuntimeError(
                f'Topics {mismatched} do not match --camera-model {camera_model!r}; '
                'select the matching topics or pass --allow-topic-model-mismatch explicitly.'
            )

    profile, intrinsics, world_path, camera_pose = load_profile(
        str(args.world_profiles), str(args.world), camera_model=camera_model
    )
    camera = _camera_from_profile(camera_pose, intrinsics)
    vis = dict(profile.get('visibility_defaults') or {})
    xmin = float(vis.get('visibility_map_min_x', -3.0)) + float(args.wall_margin)
    xmax = float(vis.get('visibility_map_max_x', 3.0)) - float(args.wall_margin)
    ymin = float(vis.get('visibility_map_min_y', -3.0)) + float(args.wall_margin)
    ymax = float(vis.get('visibility_map_max_y', 3.0)) - float(args.wall_margin)
    if args.sample_min_x is not None:
        xmin = max(xmin, float(args.sample_min_x))
    if args.sample_max_x is not None:
        xmax = min(xmax, float(args.sample_max_x))
    if args.sample_min_y is not None:
        ymin = max(ymin, float(args.sample_min_y))
    if args.sample_max_y is not None:
        ymax = min(ymax, float(args.sample_max_y))
    if xmax <= xmin or ymax <= ymin:
        raise RuntimeError('Invalid sampling bounds after wall margin')

    xs = np.linspace(xmin, xmax, max(int(args.sample_nx), 1))
    ys = np.linspace(ymin, ymax, max(int(args.sample_ny), 1))
    yaws = np.asarray(evenly_spaced_yaws(max(int(args.yaw_samples), 1)), dtype=float)
    unfiltered_pose_records = build_pose_records(xs, ys, yaws)
    # Assign on the complete common grid before any camera-dependent range or
    # visibility filtering.  Otherwise A/B/C/D see different group sets and
    # the same physical pose can silently become train for one camera and val
    # for another.
    unfiltered_split_labels = assign_splits(
        unfiltered_pose_records,
        val_fraction=float(args.val_fraction),
        split_mode=str(args.split_mode),
        seed=int(args.split_seed),
        spatial_block_size=int(args.spatial_block_size),
    )
    split_by_pose_index = {
        (int(record['x_idx']), int(record['y_idx']), int(record['yaw_idx'])): split
        for record, split in zip(unfiltered_pose_records, unfiltered_split_labels)
    }
    known_regions = list(profile.get('known_2d_regions') or [])
    traversable = [] if args.skip_region_filter else [
        r for r in known_regions if str(r.get('type', '')).strip().lower() == 'traversable'
    ]
    #: Region types that describe where the robot MAY be rather than where it may not.
    #: A ``site_boundary`` is the operating field -- an inclusion bound covering the whole
    #: building -- so treating it as an obstacle rejects every pose in the world. That is
    #: exactly what happened: the "everything that is not traversable is an obstacle" rule
    #: was written on 2026-07-20, `site_boundary` entries were added to the profiles on
    #: 2026-07-29, and no capture has been attempted since, so a total blocker sat latent
    #: in every world that declares one (warehouse_full_4cam, warehouse_v2 and its
    #: shipout pair). The 2026-07-24 four-camera capture predates the profile entry, which
    #: is why it succeeded.
    NON_BLOCKING_REGION_TYPES = {'traversable', 'site_boundary'}
    excluded_regions = [] if args.skip_collision_filter else [
        r for r in known_regions
        if str(r.get('type', '')).strip().lower() not in NON_BLOCKING_REGION_TYPES
    ]
    collision_prisms = ()
    if not args.skip_collision_filter:
        collision_prisms = profile_collision_scene(str(world_path), profile).prisms
    route_config_path = args.route_exclusion_config.expanduser().resolve()
    exclusion_segments = _route_exclusion_segments(
        route_config_path,
        route_names=[str(value) for value in args.exclude_route],
    )
    capture_script_path = Path(__file__).resolve()
    capture_script_sha256 = _sha256_file(capture_script_path)
    capture_invocation = _capture_invocation(args)
    simulation_asset_inventory = _build_simulation_asset_inventory(
        world_path=Path(world_path),
        world_profiles_path=Path(args.world_profiles),
        route_exclusion_config_path=route_config_path,
    )
    pose_records, pose_filter_counts = _filter_pose_records(
        unfiltered_pose_records,
        traversable_regions=traversable,
        excluded_regions=excluded_regions,
        region_shrink_m=float(args.region_shrink_m),
        collision_prisms=tuple(collision_prisms),
        collision_clearance_m=float(args.collision_clearance_m),
        camera_xy=(float(camera_pose[0]), float(camera_pose[1])),
        min_camera_range_m=float(args.min_camera_range_m),
        max_camera_range_m=float(args.max_camera_range_m),
        exclusion_segments=exclusion_segments,
        route_exclusion_buffer_m=float(args.route_exclusion_buffer_m),
    )
    # Negative candidates retain the common-grid split and use the identical
    # driveability, collision, and held-out-route filters. They are selected
    # only from the train split and must be geometrically outside the image;
    # an in-frustum robot hidden by an occluder is never a negative.
    negative_spatial_records, _ = _filter_pose_records(
        unfiltered_pose_records,
        traversable_regions=traversable,
        excluded_regions=excluded_regions,
        region_shrink_m=float(args.region_shrink_m),
        collision_prisms=tuple(collision_prisms),
        collision_clearance_m=float(args.collision_clearance_m),
        camera_xy=(float(camera_pose[0]), float(camera_pose[1])),
        min_camera_range_m=0.0,
        max_camera_range_m=float('inf'),
        exclusion_segments=exclusion_segments,
        route_exclusion_buffer_m=float(args.route_exclusion_buffer_m),
    )
    negative_candidates: list[tuple[dict[str, float | int], str]] = []
    for record in negative_spatial_records:
        split = split_by_pose_index[
            (int(record['x_idx']), int(record['y_idx']), int(record['yaw_idx']))
        ]
        if split != 'train':
            continue
        reason = _negative_no_opportunity_reason(
            record,
            camera=camera,
            image_width=int(intrinsics['img_width']),
            image_height=int(intrinsics['img_height']),
            robot_z=float(args.robot_z),
            box_length=float(args.robot_box_length),
            box_width=float(args.robot_box_width),
            box_height=float(args.robot_box_height),
        )
        if reason:
            negative_candidates.append((record, reason))
    negative_target = int(args.negative_samples_per_camera)
    if negative_target < 0:
        raise RuntimeError('--negative-samples-per-camera must be non-negative')
    if negative_target > 0 and not negative_candidates:
        raise RuntimeError(
            'No train-split geometry-certified no-opportunity pose remains after '
            'driveability, collision, and held-out-route exclusions.'
        )
    if not bool(args.skip_projection_filter):
        pose_records, projection_filter_counts = _filter_projectable_pose_records(
            pose_records,
            camera=camera,
            image_width=int(intrinsics['img_width']),
            image_height=int(intrinsics['img_height']),
            robot_z=float(args.robot_z),
            box_length=float(args.robot_box_length),
            box_width=float(args.robot_box_width),
            box_height=float(args.robot_box_height),
        )
        for reason, count in projection_filter_counts.items():
            if reason != 'kept':
                pose_filter_counts[reason] = int(pose_filter_counts.get(reason, 0)) + int(count)
        pose_filter_counts['kept'] = len(pose_records)
    if not pose_records:
        raise RuntimeError(f'All commanded poses were removed by geometry/range filters: {pose_filter_counts}')
    planned = [(float(item['x']), float(item['y']), float(item['yaw'])) for item in pose_records]
    split_labels = [
        split_by_pose_index[(int(record['x_idx']), int(record['y_idx']), int(record['yaw_idx']))]
        for record in pose_records
    ]
    if len(planned) < int(args.min_accepted_samples):
        raise RuntimeError(
            'Capture plan is geometrically incapable of meeting the acceptance gate: '
            f'{len(planned)} projectable poses < min_accepted_samples '
            f'{int(args.min_accepted_samples)}; filter_counts={pose_filter_counts}'
        )
    planned_split_counts = Counter(split_labels)
    if any(planned_split_counts.get(split, 0) == 0 for split in ('train', 'val')):
        raise RuntimeError(
            'Capture plan has no projectable pose in both grouped splits: '
            f'{dict(planned_split_counts)}; increase spatial grid density without changing the split seed.'
        )
    if bool(args.plan_only):
        print(json.dumps({
            'world': str(args.world),
            'camera_id': camera_id,
            'camera_model': camera_model,
            'sampling_bounds': {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax},
            'unfiltered_pose_count': len(unfiltered_pose_records),
            'projectable_pose_count': len(planned),
            'split_counts': dict(planned_split_counts),
            'filter_counts': pose_filter_counts,
            'negative_candidate_count': len(negative_candidates),
            'negative_candidate_preview': [
                {
                    'x': float(record['x']),
                    'y': float(record['y']),
                    'yaw': float(record['yaw']),
                    'reason': reason,
                }
                for record, reason in negative_candidates[:16]
            ],
            'simulation_asset_inventory_sha256': simulation_asset_inventory['aggregate_sha256'],
        }, indent=2, sort_keys=True))
        return 0

    if str(args.out).strip():
        out_dir = Path(args.out).expanduser().resolve()
    else:
        out_dir = (REPO_ROOT / 'logs' / f'warehouse_yolo_dataset_{_timestamp()}').resolve()
    archived_to = ''
    if out_dir.exists():
        if bool(args.archive_existing):
            archived_to = str(_archive_existing_path(out_dir))
        elif bool(args.replace):
            shutil.rmtree(out_dir)
        else:
            raise RuntimeError(f'Output folder already exists: {out_dir}')

    for split in ('train', 'val'):
        (out_dir / 'images' / split).mkdir(parents=True, exist_ok=False)
        (out_dir / 'labels' / split).mkdir(parents=True, exist_ok=False)
        if bool(args.save_masks):
            (out_dir / 'masks' / split).mkdir(parents=True, exist_ok=False)
    (out_dir / 'audit' / 'accepted').mkdir(parents=True, exist_ok=True)
    (out_dir / 'audit' / 'rejected').mkdir(parents=True, exist_ok=True)
    if bool(args.save_observation_frames):
        (out_dir / 'observation_frames').mkdir(parents=True, exist_ok=True)
    output_guard = _CaptureOutputGuard(
        out_dir,
        camera_id=camera_id,
        training_eligible=bool(transport_environment['training_eligible']),
    )
    _ACTIVE_CAPTURE_OUTPUT_GUARD = output_guard

    diagnostics: list[dict] = []
    accepted_preview_paths: list[Path] = []
    rejected_preview_paths: list[Path] = []
    accepted_hashes: list[str] = []
    seen_image_hashes: set[str] = set()
    rejected_counts: Counter[str] = Counter()
    accepted = 0
    negative_accepted = 0
    rejected_samples = 0

    print(
        f'capture plan camera={camera_id} poses={len(planned)} '
        f'train={planned_split_counts.get("train", 0)} '
        f'val={planned_split_counts.get("val", 0)} filters={pose_filter_counts} '
        f'negative_target={negative_target} negative_candidates={len(negative_candidates)}',
        flush=True,
    )

    node: TeleportYoloDatasetCapture | None = None
    try:
        rclpy.init()
        node = TeleportYoloDatasetCapture(
            world_name=str(profile['world_name']),
            image_topic=str(args.image_topic),
            labels_topic=str(args.labels_topic),
            robot_z=float(args.robot_z),
            settle_s=float(args.settle_s),
            image_timeout_s=float(args.image_timeout_s),
            sync_slop_s=float(args.sync_slop_ms) / 1000.0,
            min_new_rgb_frames=int(args.min_new_rgb_frames),
            min_new_label_frames=int(args.min_new_label_frames),
            buffer_size=int(args.buffer_size),
            zero_cmd_topic=str(args.zero_cmd_topic),
        )
        node.wait_for_ready(timeout_s=30.0)
        for sample_index, (x, y, yaw) in enumerate(planned):
            split = str(split_labels[sample_index])
            sample_accepted = False
            last_reason = 'not_attempted'
            for attempt in range(max(int(args.max_sample_attempts), 1)):
                pair: FramePair | None = None
                quality: LabelQuality | None = None
                try:
                    pair = node.capture_at_pose(x=x, y=y, yaw=yaw)
                    if pair.labels.shape[:2] != pair.image_bgr.shape[:2]:
                        raise RuntimeError(
                            f'Segmentation label map shape {pair.labels.shape[:2]} does not match RGB image shape {pair.image_bgr.shape[:2]}'
                        )
                    quality = validate_sample_quality(
                        image_bgr=pair.image_bgr,
                        labels=pair.labels,
                        robot_label=int(args.robot_label),
                        epsilon_ratio=float(args.epsilon_ratio),
                        bottom_band_px=float(args.bottom_band_px),
                        min_mask_area=_mask_area_gate_for_range(
                            math.hypot(x - float(camera_pose[0]), y - float(camera_pose[1])),
                            near_min_area_px=float(args.min_mask_area),
                            far_start_m=float(args.far_range_start_m),
                            far_min_area_px=float(args.far_min_mask_area),
                        ),
                        min_mask_bbox_w=float(args.min_mask_bbox_w),
                        min_mask_bbox_h=float(args.min_mask_bbox_h),
                        max_mask_border_fraction=float(args.max_mask_border_fraction),
                        min_rgb_robot_color_fraction=float(args.min_rgb_robot_color_fraction),
                        disable_rgb_color_check=bool(args.disable_rgb_color_check),
                        robot_color_profile=str(args.robot_color_profile),
                        camera=camera,
                        x=x,
                        y=y,
                        yaw=yaw,
                        robot_z=float(args.robot_z),
                        box_length=float(args.robot_box_length),
                        box_width=float(args.robot_box_width),
                        box_height=float(args.robot_box_height),
                        max_expected_center_error_px=float(args.max_expected_center_error_px),
                        min_visible_height_fraction=float(args.min_visible_height_fraction),
                        max_bottom_occlusion_px=float(args.max_bottom_occlusion_px),
                    )
                    last_reason = quality.reason
                    if (
                        str(args.occlusion_policy) == 'visible-mask-positive'
                        and quality.reason in {'occluded_low_visible_height', 'occluded_bottom_hidden'}
                        and (
                            bool(args.disable_rgb_color_check)
                            or quality.rgb_robot_color_fraction >= float(args.min_rgb_robot_color_fraction)
                        )
                    ):
                        # Gazebo labels only the rendered/visible silhouette.
                        # It is a valid detector segmentation target but its
                        # bottom point is explicitly forbidden for projection.
                        quality.accepted = True
                        quality.reason = ''
                        last_reason = ''
                    if quality.accepted and quality.image_sha1 in seen_image_hashes:
                        quality.accepted = False
                        quality.reason = 'duplicate_image'
                        last_reason = quality.reason
                    if quality.accepted:
                        stem = f'sample_{sample_index:06d}'
                        image_path = out_dir / 'images' / split / f'{stem}.png'
                        label_path = out_dir / 'labels' / split / f'{stem}.txt'
                        cv2.imwrite(str(image_path), pair.image_bgr)
                        _write_seg_label(label_path, quality.polygon, pair.image_bgr.shape[1], pair.image_bgr.shape[0])
                        mask_rel = ''
                        if bool(args.save_masks):
                            mask_path = out_dir / 'masks' / split / f'{stem}.png'
                            cv2.imwrite(str(mask_path), quality.raw_mask)
                            mask_rel = str(mask_path.relative_to(out_dir))
                        preview_rel = ''
                        if len(accepted_preview_paths) < max(int(args.preview_count), 0):
                            overlay = _draw_overlay(
                                pair.image_bgr,
                                quality,
                                text=(
                                    f'ok idx={sample_index} try={attempt} area={quality.mask_area_px:.0f} '
                                    f'err={quality.expected_center_error_px:.1f}px vh={quality.visible_height_fraction:.2f} '
                                    f'bgap={quality.bottom_occlusion_px:.0f}px rgb={quality.rgb_robot_color_fraction:.3f}'
                                ),
                            )
                            preview_path = out_dir / 'audit' / 'accepted' / f'{stem}.jpg'
                            cv2.imwrite(str(preview_path), overlay)
                            accepted_preview_paths.append(preview_path)
                            preview_rel = str(preview_path.relative_to(out_dir))
                        diagnostics.append(_diagnostic_row(
                            sample_index=sample_index,
                            attempt=attempt,
                            split=split,
                            accepted=True,
                            reason='',
                            pair=pair,
                            quality=quality,
                            x=x,
                            y=y,
                            yaw=yaw,
                            settle_s=float(args.settle_s),
                            image_rel=str(image_path.relative_to(out_dir)),
                            label_rel=str(label_path.relative_to(out_dir)),
                            mask_rel=mask_rel,
                            preview_rel=preview_rel,
                            observation_image_rel=(
                                str(image_path.relative_to(out_dir))
                                if bool(args.save_observation_frames) else ''
                            ),
                        ))
                        accepted += 1
                        accepted_hashes.append(quality.image_sha1)
                        seen_image_hashes.add(quality.image_sha1)
                        sample_accepted = True
                        break
                    rejected_counts[quality.reason] += 1
                    observation_image_rel = ''
                    if bool(args.save_observation_frames):
                        observation_path = (
                            out_dir / 'observation_frames'
                            / f'sample_{sample_index:06d}_try{attempt}.png'
                        )
                        if not cv2.imwrite(str(observation_path), pair.image_bgr):
                            raise RuntimeError(
                                f'Failed to write observation frame {observation_path}'
                            )
                        observation_image_rel = str(observation_path.relative_to(out_dir))
                    if len(rejected_preview_paths) < max(int(args.rejected_preview_count), 0):
                        overlay = _draw_overlay(
                            pair.image_bgr,
                            quality,
                            text=(
                                f'reject={quality.reason} idx={sample_index} try={attempt} '
                                f'area={quality.mask_area_px:.0f} err={quality.expected_center_error_px:.1f}px '
                                f'vh={quality.visible_height_fraction:.2f} bgap={quality.bottom_occlusion_px:.0f}px'
                            ),
                        )
                        preview_path = out_dir / 'audit' / 'rejected' / f'sample_{sample_index:06d}_try{attempt}.jpg'
                        cv2.imwrite(str(preview_path), overlay)
                        rejected_preview_paths.append(preview_path)
                    diagnostics.append(_diagnostic_row(
                        sample_index=sample_index,
                        attempt=attempt,
                        split=split,
                        accepted=False,
                        reason=quality.reason,
                        pair=pair,
                        quality=quality,
                        x=x,
                        y=y,
                        yaw=yaw,
                        settle_s=float(args.settle_s),
                        observation_image_rel=observation_image_rel,
                    ))
                    # In the fixed world, label-quality failures at an exact
                    # commanded pose are deterministic. Repeating them used to
                    # multiply capture time without creating an independent
                    # example. Only capture/synchronization exceptions below
                    # consume another configured attempt.
                    break
                except Exception as exc:
                    if not rclpy.ok():
                        raise RuntimeError(
                            'ROS context shut down during capture; aborting instead of '
                            'turning transport failure into repeated rejected samples.'
                        ) from exc
                    last_reason = f'capture_error:{exc}'
                    rejected_counts['capture_error'] += 1
                    diagnostics.append(_diagnostic_row(
                        sample_index=sample_index,
                        attempt=attempt,
                        split=split,
                        accepted=False,
                        reason=last_reason,
                        pair=pair,
                        quality=quality,
                        x=x,
                        y=y,
                        yaw=yaw,
                        settle_s=float(args.settle_s),
                    ))
            if not sample_accepted:
                rejected_samples += 1
                if not last_reason:
                    rejected_counts['unknown'] += 1
            if (sample_index + 1) % 25 == 0:
                print(f'captured {sample_index + 1}/{len(planned)} planned, accepted={accepted}, rejected_samples={rejected_samples}', flush=True)

        for negative_index, (record, certification) in enumerate(negative_candidates):
            if negative_accepted >= negative_target:
                break
            x = float(record['x'])
            y = float(record['y'])
            yaw = float(record['yaw'])
            pair: FramePair | None = None
            try:
                pair = node.capture_at_pose(x=x, y=y, yaw=yaw)
                if pair.labels.shape[:2] != pair.image_bgr.shape[:2]:
                    raise RuntimeError(
                        f'Segmentation label map shape {pair.labels.shape[:2]} does not '
                        f'match RGB image shape {pair.image_bgr.shape[:2]}'
                    )
            except Exception as exc:
                if not rclpy.ok():
                    raise RuntimeError(
                        'ROS context shut down during negative capture.'
                    ) from exc
                rejected_counts['negative_capture_error'] += 1
                diagnostics.append(_diagnostic_row(
                    sample_index=negative_index,
                    attempt=0,
                    split='train',
                    accepted=False,
                    reason=f'negative_capture_error:{exc}',
                    pair=pair,
                    quality=None,
                    x=x,
                    y=y,
                    yaw=yaw,
                    settle_s=float(args.settle_s),
                    sample_kind='negative',
                    geometry_certified_no_opportunity=True,
                    no_opportunity_reason=certification,
                ))
                continue

            semantic_robot_pixels = int(np.count_nonzero(pair.labels == int(args.robot_label)))
            image_sha1 = _sha1_array(pair.image_bgr)
            rejection_reason = ''
            if semantic_robot_pixels != 0:
                rejection_reason = 'negative_contains_semantic_robot_pixels'
            elif image_sha1 in seen_image_hashes:
                rejection_reason = 'duplicate_negative_image'
            if rejection_reason:
                rejected_counts[rejection_reason] += 1
                row = _diagnostic_row(
                    sample_index=negative_index,
                    attempt=0,
                    split='train',
                    accepted=False,
                    reason=rejection_reason,
                    pair=pair,
                    quality=None,
                    x=x,
                    y=y,
                    yaw=yaw,
                    settle_s=float(args.settle_s),
                    sample_kind='negative',
                    geometry_certified_no_opportunity=True,
                    no_opportunity_reason=certification,
                    semantic_robot_pixels=semantic_robot_pixels,
                )
                row['image_sha1'] = image_sha1
                row['label_sha1'] = _sha1_array(pair.labels)
                diagnostics.append(row)
                continue

            stem = f'negative_{negative_accepted:06d}'
            image_path = out_dir / 'images' / 'train' / f'{stem}.png'
            label_path = out_dir / 'labels' / 'train' / f'{stem}.txt'
            if not cv2.imwrite(str(image_path), pair.image_bgr):
                raise RuntimeError(f'Failed to write negative image {image_path}')
            _write_seg_label(
                label_path,
                None,
                pair.image_bgr.shape[1],
                pair.image_bgr.shape[0],
            )
            mask_rel = ''
            if bool(args.save_masks):
                mask_path = out_dir / 'masks' / 'train' / f'{stem}.png'
                empty_mask = np.zeros(pair.labels.shape[:2], dtype=np.uint8)
                if not cv2.imwrite(str(mask_path), empty_mask):
                    raise RuntimeError(f'Failed to write negative mask {mask_path}')
                mask_rel = str(mask_path.relative_to(out_dir))
            row = _diagnostic_row(
                sample_index=negative_index,
                attempt=0,
                split='train',
                accepted=True,
                reason='',
                pair=pair,
                quality=None,
                x=x,
                y=y,
                yaw=yaw,
                settle_s=float(args.settle_s),
                sample_kind='negative',
                geometry_certified_no_opportunity=True,
                no_opportunity_reason=certification,
                semantic_robot_pixels=0,
                image_rel=str(image_path.relative_to(out_dir)),
                label_rel=str(label_path.relative_to(out_dir)),
                mask_rel=mask_rel,
            )
            row.update({
                'mask_area_px': 0.0,
                'localization_qualified': 0,
                'occlusion_state': 'not_applicable',
                'image_sha1': image_sha1,
                'label_sha1': hashlib.sha1(b'').hexdigest(),
            })
            diagnostics.append(row)
            negative_accepted += 1
            accepted_hashes.append(image_sha1)
            seen_image_hashes.add(image_sha1)

        if negative_accepted < negative_target:
            raise RuntimeError(
                'Negative background gate failed: accepted '
                f'{negative_accepted}/{negative_target} unique zero-robot frames from '
                f'{len(negative_candidates)} certified candidates.'
            )
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception as exc:
                print(f'WARNING: capture node cleanup failed: {exc}', file=sys.stderr)
        # rclpy's SIGINT handler may already have shut the context down. Calling
        # shutdown twice raises RCLError and used to mask the actual interrupt.
        if rclpy.ok():
            rclpy.shutdown()

    data_yaml = {
        'path': str(out_dir),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'robot'},
        'task': 'segment',
    }
    (out_dir / 'data.yaml').write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding='utf-8')

    with (out_dir / 'label_diagnostics.csv').open('w', newline='', encoding='utf-8') as handle:
        for row in diagnostics:
            camera_range = math.hypot(
                float(row['robot_x']) - float(camera_pose[0]),
                float(row['robot_y']) - float(camera_pose[1]),
            )
            row.update({
                'camera_id': camera_id,
                'camera_model': camera_model,
                'camera_range_m': float(camera_range),
                'camera_range_bin': _camera_range_bin(camera_range),
                'min_mask_area_applied_px': (
                    0.0
                    if str(row.get('sample_kind')) == 'negative'
                    else _mask_area_gate_for_range(
                        camera_range,
                        near_min_area_px=float(args.min_mask_area),
                        far_start_m=float(args.far_range_start_m),
                        far_min_area_px=float(args.far_min_mask_area),
                    )
                ),
            })
        writer = csv.DictWriter(handle, fieldnames=_fixed_fieldnames())
        writer.writeheader()
        writer.writerows(diagnostics)

    accepted_rows = [r for r in diagnostics if int(r['accepted']) == 1]
    accepted_positive_rows = [
        row for row in accepted_rows if str(row.get('sample_kind')) == 'positive'
    ]
    total_accepted = accepted + negative_accepted
    duplicate_images = sum(count for count in Counter(accepted_hashes).values() if count > 1)
    duplicate_fraction = duplicate_images / float(max(len(accepted_hashes), 1))
    split_counts = Counter(str(r['split']) for r in accepted_rows)
    positive_split_counts = Counter(str(r['split']) for r in accepted_positive_rows)
    accepted_fraction = accepted / float(max(len(planned), 1))
    accepted_sheet = _write_contact_sheet(accepted_preview_paths, out_dir / 'audit' / 'accepted_contact_sheet.jpg')
    rejected_sheet = _write_contact_sheet(rejected_preview_paths, out_dir / 'audit' / 'rejected_contact_sheet.jpg')
    audit = {
        'unfiltered_planned_samples': int(len(unfiltered_pose_records)),
        'planned_samples': int(len(planned)),
        'pose_filter_counts': pose_filter_counts,
        'accepted_samples': int(total_accepted),
        'positive_planned_samples': int(len(planned)),
        'positive_accepted_samples': int(accepted),
        'negative_candidate_samples': int(len(negative_candidates)),
        'negative_target_samples': int(negative_target),
        'negative_accepted_samples': int(negative_accepted),
        'rejected_planned_samples': int(rejected_samples),
        'accepted_fraction': float(accepted_fraction),
        'split_counts': dict(split_counts),
        'positive_split_counts': dict(positive_split_counts),
        'accepted_by_sample_kind': dict(
            Counter(str(r['sample_kind']) for r in accepted_rows)
        ),
        'accepted_by_range_bin': dict(Counter(str(r['camera_range_bin']) for r in accepted_rows)),
        'accepted_by_occlusion_state': dict(Counter(str(r['occlusion_state']) for r in accepted_rows)),
        'localization_qualified_samples': int(sum(int(r['localization_qualified']) for r in accepted_rows)),
        'rejection_counts_by_attempt': dict(rejected_counts),
        'duplicate_accepted_images': int(duplicate_images),
        'duplicate_accepted_fraction': float(duplicate_fraction),
        'stamp_delta_s': _stats(float(r['stamp_delta_s']) for r in accepted_positive_rows),
        'set_pose_latency_s': _stats(float(r['set_pose_latency_s']) for r in accepted_positive_rows),
        'pair_wait_s': _stats(float(r['pair_wait_s']) for r in accepted_positive_rows),
        'mask_area_px': _stats(float(r['mask_area_px']) for r in accepted_positive_rows),
        'expected_center_error_px': _stats(float(r['expected_center_error_px']) for r in accepted_positive_rows),
        'visible_height_fraction': _stats(float(r['visible_height_fraction']) for r in accepted_positive_rows),
        'bottom_occlusion_px': _stats(float(r['bottom_occlusion_px']) for r in accepted_positive_rows),
        'rgb_robot_color_fraction': _stats(float(r['rgb_robot_color_fraction']) for r in accepted_positive_rows),
        'accepted_contact_sheet': accepted_sheet,
        'rejected_contact_sheet': rejected_sheet,
    }
    final_capture_script_sha256 = _sha256_file(capture_script_path)
    if final_capture_script_sha256 != capture_script_sha256:
        raise RuntimeError(
            'Capture script changed while collection was running; the partial dataset is '
            'not provenance-safe and must not be merged for training.'
        )
    final_asset_inventory = _build_simulation_asset_inventory(
        world_path=Path(world_path),
        world_profiles_path=Path(args.world_profiles),
        route_exclusion_config_path=route_config_path,
    )
    if (
        final_asset_inventory['aggregate_sha256']
        != simulation_asset_inventory['aggregate_sha256']
    ):
        raise RuntimeError(
            'A simulation/input asset changed while collection was running; the partial '
            'dataset is not provenance-safe and must not be merged for training.'
        )
    manifest = {
        'status': 'complete',
        'world': str(args.world),
        'source': 'gazebo_semantic_segmentation',
        'camera_id': camera_id,
        'camera_model': camera_model,
        'image_topic': str(args.image_topic),
        'labels_topic': str(args.labels_topic),
        'world_profiles_path': str(Path(args.world_profiles).expanduser().resolve()),
        'world_profiles_sha256': _sha256_file(Path(args.world_profiles).expanduser().resolve()),
        'world_path': str(Path(world_path).resolve()),
        'world_sha256': _sha256_file(Path(world_path).resolve()),
        'capture_script_path': str(capture_script_path),
        'capture_script_sha256': capture_script_sha256,
        'capture_invocation': capture_invocation,
        'transport_environment': transport_environment,
        'simulation_asset_inventory': simulation_asset_inventory,
        'excluded_evaluation_routes': [str(value) for value in args.exclude_route],
        'route_exclusion_config': str(route_config_path),
        'route_exclusion_config_sha256': (
            _sha256_file(route_config_path) if route_config_path.is_file() else None
        ),
        'route_exclusion_buffer_m': float(args.route_exclusion_buffer_m),
        'archived_existing_to': archived_to,
        'robot_label': int(args.robot_label),
        'camera_pose_xyz_rpy': [float(v) for v in camera_pose],
        'camera_intrinsics': dict(intrinsics),
        'sample_nx': int(args.sample_nx),
        'sample_ny': int(args.sample_ny),
        'yaw_samples': int(args.yaw_samples),
        'yaw_values_rad': [float(v) for v in yaws.tolist()],
        'split_mode': str(args.split_mode),
        'split_seed': int(args.split_seed),
        'spatial_block_size': int(args.spatial_block_size),
        'capture_thresholds': {
            'settle_s': float(args.settle_s),
            'image_timeout_s': float(args.image_timeout_s),
            'sync_slop_ms': float(args.sync_slop_ms),
            'min_new_rgb_frames': int(args.min_new_rgb_frames),
            'min_new_label_frames': int(args.min_new_label_frames),
            'max_sample_attempts': int(args.max_sample_attempts),
            'min_mask_area': float(args.min_mask_area),
            'far_range_start_m': float(args.far_range_start_m),
            'far_min_mask_area': float(args.far_min_mask_area),
            'min_mask_bbox_w': float(args.min_mask_bbox_w),
            'min_mask_bbox_h': float(args.min_mask_bbox_h),
            'max_mask_border_fraction': float(args.max_mask_border_fraction),
            'min_rgb_robot_color_fraction': float(args.min_rgb_robot_color_fraction),
            'robot_color_profile': str(args.robot_color_profile),
            'robot_color_signatures': list(ROBOT_COLOR_PROFILES[str(args.robot_color_profile)]),
            'max_expected_center_error_px': float(args.max_expected_center_error_px),
            'min_visible_height_fraction': float(args.min_visible_height_fraction),
            'max_bottom_occlusion_px': float(args.max_bottom_occlusion_px),
            'occlusion_policy': str(args.occlusion_policy),
            'max_final_duplicate_fraction': float(args.max_final_duplicate_fraction),
            'min_accepted_samples': int(args.min_accepted_samples),
            'min_accept_fraction': float(args.min_accept_fraction),
            'negative_samples_per_camera': int(negative_target),
            'save_observation_frames': bool(args.save_observation_frames),
            'negative_split': 'train',
            'negative_semantic_robot_pixels': 0,
            'negative_no_opportunity_reasons': [
                'projection_behind_camera',
                'projection_outside_image',
            ],
            'robot_box_length': float(args.robot_box_length),
            'robot_box_width': float(args.robot_box_width),
            'robot_box_height': float(args.robot_box_height),
            'robot_z': float(args.robot_z),
            'collision_clearance_m': float(args.collision_clearance_m),
            'skip_collision_filter': bool(args.skip_collision_filter),
            'skip_projection_filter': bool(args.skip_projection_filter),
            'min_camera_range_m': float(args.min_camera_range_m),
            'max_camera_range_m': (
                float(args.max_camera_range_m) if math.isfinite(float(args.max_camera_range_m)) else None
            ),
        },
        'audit': audit,
        'timestamp': _timestamp(),
        'notes': [
            'Semantic segmentation is used only for offline training labels.',
            'The runtime detector still consumes raw RGB images.',
            'Freshness is measured after the teleport settle interval, so pre-settle frames cannot be accepted.',
            'Accepted samples require synchronized RGB and semantic labels plus projection, visibility, and duplicate checks.',
            'Rejected samples are not written as YOLO training negatives by default.',
            'Opt-in negatives are unique train-only augmentation frames with fresh synchronized RGB/semantic input, exactly zero robot-label pixels, and a geometry-certified robot box outside the image.',
            'A fully occluded in-frustum robot is never a negative. In this fixed scene, one unique background per camera is augmentation, not independent false-positive evidence.',
            'Deterministic label-quality rejections are attempted once; max_sample_attempts applies to transient capture/synchronization exceptions.',
            'Partial silhouettes are accepted only when occlusion_policy=visible-mask-positive; '
            'those rows have localization_qualified=0 and must not calibrate a bottom-point projection.',
            'When save_observation_frames is enabled, observation_image names the RGB input '
            'for every successfully synchronized planned pose, whether or not semantic-label '
            'quality admitted it to the detector-training dataset.',
        ],
    }
    failures = []
    if accepted < int(args.min_accepted_samples):
        failures.append(f'accepted {accepted} < min_accepted_samples {int(args.min_accepted_samples)}')
    if accepted_fraction < float(args.min_accept_fraction):
        failures.append(f'accepted_fraction {accepted_fraction:.3f} < {float(args.min_accept_fraction):.3f}')
    if duplicate_fraction > float(args.max_final_duplicate_fraction):
        failures.append(f'duplicate_accepted_fraction {duplicate_fraction:.3f} > {float(args.max_final_duplicate_fraction):.3f}')
    if positive_split_counts.get('train', 0) <= 0 or positive_split_counts.get('val', 0) <= 0:
        failures.append(f'missing positive train/val accepted split: {dict(positive_split_counts)}')
    if negative_accepted != negative_target:
        failures.append(
            f'negative_accepted_samples {negative_accepted} != target {negative_target}'
        )
    if failures:
        raise RuntimeError('Dataset failed quality gates: ' + '; '.join(failures))

    manifest_path = out_dir / 'dataset_manifest.json'
    _atomic_write_json(manifest_path, manifest)
    output_guard.complete(manifest_path)
    _ACTIVE_CAPTURE_OUTPUT_GUARD = None

    print(f'Wrote simulator-segmentation YOLO dataset to {out_dir}')
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == '__main__':
    try:
        exit_code = main()
    except BaseException as exc:
        _quarantine_active_capture(exc)
        raise
    raise SystemExit(exit_code)
