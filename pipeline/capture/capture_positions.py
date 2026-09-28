#!/usr/bin/env python3
"""Capture one synchronized five-camera RGB batch at each warehouse pose.

This is an observation-characterization capture, not a detector-training dataset. It keeps
every fresh RGB frame, including an empty/occluded view, so later YOLO miss and
false-positive maps have an actual image behind every attempted opportunity. Semantic labels
are optional diagnostics and are not required by the main capture.
"""
from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import math
import os
import sys
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.parameter import Parameter
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from sensor_msgs.msg import Image

REPO = Path(__file__).resolve().parents[2]
for rel in ('pipeline/capture', 'src/experiments', 'src/perception', 'src/unav_common'):
    value = str((REPO / rel).resolve())
    if value not in sys.path:
        sys.path.insert(0, value)

from capture_yolo_dataset import (  # noqa: E402
    _camera_from_profile,
    _capture_transport_environment,
    _filter_pose_records,
    _project_robot_bbox,
    _read_uint_label_map,
    _sha1_array,
    _sha256_file,
    _stamp_ns,
)
from dataset_split_utils import build_pose_records, evenly_spaced_yaws  # noqa: E402
from experiments.core.world_profiles import load_profile  # noqa: E402
from perception.core.ros_image import image_msg_to_bgr8  # noqa: E402
from unav_common.occlusion_geometry import profile_collision_scene  # noqa: E402
from unav_common.rectangular_footprint import RectangularFootprint  # noqa: E402
from unav_common.capture_integrity import (  # noqa: E402
    CaptureIndexWriter, atomic_bytes, atomic_json, capture_lock,
)
from audit_capture_resume import require_resume  # noqa: E402


CAMERAS = (
    ('camera_A', 'external_camera'),
    ('camera_B', 'external_camera_b'),
    ('camera_C', 'external_camera_c'),
    ('camera_D', 'external_camera_d'),
    ('camera_E', 'external_camera_e'),
)
FIELDS = (
    'pose_id', 'position_id', 'x_idx', 'y_idx', 'heading_id', 'repetition_id',
    'source_batch_id',
    'capture_session_id', 'image_id', 'image_stamp_ns', 'raw_image_sha1',
    'command_issue_ns', 'command_ack_ns', 'settle_barrier_ns',
    'dataset_split', 'random_draw_index',
    'camera_id', 'camera_model', 'image', 'capture_status', 'capture_error',
    'robot_x', 'robot_y', 'robot_yaw', 'camera_range_m',
    'image_stamp_s', 'label_stamp_s', 'stamp_delta_s', 'batch_image_span_s',
    'image_sha1', 'semantic_robot_pixels', 'line_of_sight',
    'mask_x0', 'mask_y0', 'mask_x1', 'mask_y1', 'mask_bottom_u', 'mask_bottom_v',
    'expected_x0', 'expected_y0', 'expected_x1', 'expected_y1', 'nominal_in_frame',
)


@dataclass(frozen=True)
class CameraSpec:
    camera_id: str
    model: str
    image_topic: str
    labels_topic: str
    camera: object
    pose: tuple[float, ...]
    image_width: int
    image_height: int


@dataclass
class Pair:
    image: np.ndarray
    labels: np.ndarray | None
    image_stamp_ns: int
    label_stamp_ns: int | None
    stamp_delta_s: float


class CaptureEpochError(RuntimeError):
    """A clock reset invalidates every outstanding capture in this process."""


def _closest_timestamp_batch(
    candidates: dict[str, list[Pair]], *, max_span_s: float
) -> dict[str, Pair] | None:
    """Choose the closest all-camera timestamp combination.

    Camera callbacks do not necessarily arrive in timestamp order across topics. Selecting
    each topic's latest fresh frame independently can therefore merge adjacent 5 Hz rounds
    into one nominal batch. Anchor each candidate timestamp in turn, take the closest frame
    from every camera, and retain the newest minimum-span combination.
    """
    if not candidates or any(not items for items in candidates.values()):
        return None
    anchors = sorted({item.image_stamp_ns for items in candidates.values() for item in items})
    best: dict[str, Pair] | None = None
    best_key: tuple[int, int] | None = None
    for anchor in anchors:
        selected = {
            camera_id: min(
                items,
                key=lambda item: (abs(item.image_stamp_ns - anchor), -item.image_stamp_ns),
            )
            for camera_id, items in candidates.items()
        }
        stamps = [item.image_stamp_ns for item in selected.values()]
        span_ns = max(stamps) - min(stamps)
        # First minimize the span; then prefer the newest coherent round.
        key = (span_ns, -min(stamps))
        if best_key is None or key < best_key:
            best_key, best = key, selected
    assert best_key is not None
    if best_key[0] * 1e-9 > float(max_span_s):
        return None
    return best


class FiveCameraCapture(Node):
    def __init__(
        self,
        *,
        world_name: str,
        cameras: tuple[CameraSpec, ...],
        robot_z: float,
        settle_s: float,
        timeout_s: float,
        sync_slop_s: float,
        batch_sync_slop_s: float,
        min_new_rgb: int,
        min_new_labels: int,
        buffer_size: int,
        with_semantic: bool,
    ) -> None:
        super().__init__('capture_bbox_characterization', parameter_overrides=[
            Parameter('use_sim_time', value=True)])
        self.robot_z = float(robot_z)
        self.settle_s = float(settle_s)
        self.timeout_s = float(timeout_s)
        self.sync_slop_s = float(sync_slop_s)
        self.batch_sync_slop_s = float(batch_sync_slop_s)
        self.min_new_rgb = max(int(min_new_rgb), 1)
        self.min_new_labels = max(int(min_new_labels), 1)
        self.with_semantic = bool(with_semantic)
        self._last_clock_ns = 0
        self._barrier_ns = 0
        self._last_consumed = {spec.camera_id: 0 for spec in cameras}
        self._last_rgb_stamp = {spec.camera_id: 0 for spec in cameras}
        self._last_label_stamp = {spec.camera_id: 0 for spec in cameras}
        self.capture_context = {}
        self.rgb_count = {spec.camera_id: 0 for spec in cameras}
        self.label_count = {spec.camera_id: 0 for spec in cameras}
        self.rgb = {spec.camera_id: deque(maxlen=max(int(buffer_size), 8)) for spec in cameras}
        self.labels = {spec.camera_id: deque(maxlen=max(int(buffer_size), 8)) for spec in cameras}
        self.last_label_error = {spec.camera_id: '' for spec in cameras}
        self.client = self.create_client(SetEntityPose, f'/world/{world_name}/set_pose')
        self.zero_cmd = self.create_publisher(Twist, '/cmd_vel', 1)
        for spec in cameras:
            self.create_subscription(
                Image,
                spec.image_topic,
                functools.partial(self._rgb_cb, spec.camera_id),
                10,
            )
            if self.with_semantic:
                self.create_subscription(
                    Image,
                    spec.labels_topic,
                    functools.partial(self._labels_cb, spec.camera_id),
                    10,
                )

    def _rgb_cb(self, camera_id: str, msg: Image) -> None:
        stamp = _stamp_ns(msg)
        if stamp <= self._last_rgb_stamp[camera_id]:
            return  # repeated or reordered delivery is not new physical evidence
        image = image_msg_to_bgr8(msg)
        self._last_rgb_stamp[camera_id] = stamp
        self.rgb_count[camera_id] += 1
        self.rgb[camera_id].append(
            (self.rgb_count[camera_id], stamp, image)
        )

    def _labels_cb(self, camera_id: str, msg: Image) -> None:
        stamp = _stamp_ns(msg)
        if stamp <= self._last_label_stamp[camera_id]:
            return
        try:
            labels = _read_uint_label_map(msg)
        except Exception as exc:  # pragma: no cover - ROS transport path
            self.last_label_error[camera_id] = str(exc)
            return
        self.label_count[camera_id] += 1
        self._last_label_stamp[camera_id] = stamp
        self.labels[camera_id].append(
            (self.label_count[camera_id], stamp, labels)
        )

    def _spin(self, timeout: float = 0.04) -> None:
        self.zero_cmd.publish(Twist())
        rclpy.spin_once(self, timeout_sec=float(timeout))
        self._sim_time_ns()

    def _sim_time_ns(self) -> int:
        stamp = self.get_clock().now().nanoseconds
        if stamp < self._last_clock_ns:
            raise CaptureEpochError('simulation clock rewound; restart the capture process')
        self._last_clock_ns = stamp
        return stamp

    def wait_ready(self, camera_ids: tuple[str, ...], timeout_s: float = 45.0) -> None:
        deadline = time.monotonic() + float(timeout_s)
        while rclpy.ok() and time.monotonic() < deadline:
            self._spin()
            service = self.client.wait_for_service(timeout_sec=0.02)
            ready = all(self.rgb[cid] for cid in camera_ids)
            if self.with_semantic:
                ready = ready and all(self.labels[cid] for cid in camera_ids)
            if service and ready and self._sim_time_ns() > 0:
                return
        missing = [
            cid for cid in camera_ids
            if not self.rgb[cid] or (self.with_semantic and not self.labels[cid])
        ]
        raise RuntimeError(f'Capture transport not ready; missing camera streams: {missing}')

    def _set_pose(self, x: float, y: float, yaw: float) -> None:
        request = SetEntityPose.Request()
        request.entity = Entity(name='turtlebot3', type=Entity.MODEL)
        request.pose.position.x = float(x)
        request.pose.position.y = float(y)
        request.pose.position.z = float(self.robot_z)
        request.pose.orientation.z = math.sin(0.5 * float(yaw))
        request.pose.orientation.w = math.cos(0.5 * float(yaw))
        future = self.client.call_async(request)
        deadline = time.monotonic() + 10.0
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            self._spin()
        result = future.result() if future.done() else None
        if result is None or not bool(getattr(result, 'success', False)):
            raise RuntimeError(f'set_pose failed at ({x:.3f}, {y:.3f}, {yaw:.3f})')

    def _pair_candidates(
        self, camera_id: str, before_rgb: int, before_labels: int
    ) -> list[Pair]:
        barrier = max(self._barrier_ns, self._last_consumed[camera_id])
        rgb_candidates = [item for item in self.rgb[camera_id]
                          if item[0] > before_rgb and barrier < item[1] <= self._sim_time_ns()]
        if len({item[1] for item in rgb_candidates}) < self.min_new_rgb:
            return []
        if not self.with_semantic:
            return [
                Pair(image.copy(), None, rgb_stamp, None, math.nan)
                for _count, rgb_stamp, image in rgb_candidates
            ]
        label_candidates = [item for item in self.labels[camera_id]
                            if item[0] > before_labels and barrier < item[1] <= self._sim_time_ns()]
        if len({item[1] for item in label_candidates}) < self.min_new_labels:
            return []
        paired: list[Pair] = []
        for rgb_count, rgb_stamp, image in rgb_candidates:
            best = None
            best_key = None
            for label_count, label_stamp, labels in label_candidates:
                delta = abs(rgb_stamp - label_stamp) * 1e-9
                if delta > self.sync_slop_s:
                    continue
                key = (min(rgb_count, label_count), -delta)
                if best_key is None or key > best_key:
                    best_key = key
                    best = Pair(image.copy(), labels.copy(), rgb_stamp, label_stamp, delta)
            if best is not None:
                paired.append(best)
        return paired

    def _batch(
        self,
        camera_ids: tuple[str, ...],
        before_rgb: dict[str, int],
        before_labels: dict[str, int],
    ) -> dict[str, Pair] | None:
        candidates = {
            camera_id: self._pair_candidates(
                camera_id, before_rgb[camera_id], before_labels[camera_id]
            )
            for camera_id in camera_ids
        }
        return _closest_timestamp_batch(candidates, max_span_s=self.batch_sync_slop_s)

    def capture(self, x: float, y: float, yaw: float, camera_ids: tuple[str, ...]) -> dict[str, Pair]:
        if not all(math.isfinite(v) for v in (x, y, yaw)):
            raise ValueError('commanded capture pose must be finite')
        if len(set(camera_ids)) != len(camera_ids) or set(camera_ids) != set(self.rgb):
            raise ValueError('capture must request the exact unique camera registry')
        issued = self._sim_time_ns()
        if issued <= 0:
            raise RuntimeError('positive simulation clock required before commanding a pose')
        self._set_pose(x, y, yaw)
        ack = self._sim_time_ns()
        self._barrier_ns = ack + math.ceil(self.settle_s * 1e9)
        self.capture_context = dict(command_issue_ns=issued, command_ack_ns=ack,
                                    settle_barrier_ns=self._barrier_ns)
        settle_deadline = time.monotonic() + self.timeout_s
        while rclpy.ok() and self._sim_time_ns() < self._barrier_ns:
            if time.monotonic() >= settle_deadline:
                raise RuntimeError('simulation did not reach the settle barrier')
            self._spin()
        before_rgb = dict(self.rgb_count)
        before_labels = dict(self.label_count)
        deadline = time.monotonic() + self.timeout_s
        while rclpy.ok() and time.monotonic() < deadline:
            self._spin()
            pairs = self._batch(camera_ids, before_rgb, before_labels)
            if pairs is not None:
                for camera, pair in pairs.items():
                    self._last_consumed[camera] = pair.image_stamp_ns
                return pairs
        missing = [
            cid for cid in camera_ids
            if not self._pair_candidates(cid, before_rgb[cid], before_labels[cid])
        ]
        raise RuntimeError(
            'fresh synchronized frame timeout; '
            f'missing={missing}, required_batch_span_s<={self.batch_sync_slop_s}'
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _add_rgb_read_noise(
    image: np.ndarray,
    *,
    stddev_dn: float,
    seed: int,
    pose_id: int,
    repetition_id: int,
    camera_index: int,
) -> np.ndarray:
    """Apply a reproducible independent camera read-noise realization.

    ``stddev_dn`` is in 8-bit digital-number units. Seed components identify one
    camera-pose repetition, so resumed captures reproduce exactly the same image.
    """
    stddev = float(stddev_dn)
    if not math.isfinite(stddev) or stddev < 0.0:
        raise ValueError('rgb-noise-stddev-dn must be finite and non-negative')
    if stddev == 0.0:
        return image.copy()
    rng = np.random.default_rng(np.random.SeedSequence([
        int(seed), int(pose_id), int(repetition_id), int(camera_index)
    ]))
    noisy = image.astype(np.float32) + rng.normal(0.0, stddev, size=image.shape)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


def _mask_geometry(labels: np.ndarray, robot_label: int) -> dict[str, float | int]:
    ys, xs = np.nonzero(labels == int(robot_label))
    if not xs.size:
        return {
            'semantic_robot_pixels': 0, 'line_of_sight': 0,
            'mask_x0': math.nan, 'mask_y0': math.nan,
            'mask_x1': math.nan, 'mask_y1': math.nan,
            'mask_bottom_u': math.nan, 'mask_bottom_v': math.nan,
        }
    bottom = int(ys.max())
    bottom_x = xs[ys >= bottom - 2]
    return {
        'semantic_robot_pixels': int(xs.size), 'line_of_sight': 1,
        'mask_x0': int(xs.min()), 'mask_y0': int(ys.min()),
        'mask_x1': int(xs.max()) + 1, 'mask_y1': int(ys.max()) + 1,
        'mask_bottom_u': float(np.mean(bottom_x)), 'mask_bottom_v': float(bottom + 1),
    }


def _expected_geometry(spec: CameraSpec, x: float, y: float, yaw: float) -> dict[str, float | int]:
    box = _project_robot_bbox(
        spec.camera,
        x=float(x), y=float(y), yaw=float(yaw), z=0.0,
        box_length=0.80, box_width=0.55, box_height=0.35,
    )
    if box is None:
        return {
            'expected_x0': math.nan, 'expected_y0': math.nan,
            'expected_x1': math.nan, 'expected_y1': math.nan, 'nominal_in_frame': 0,
        }
    x0, y0, x1, y1 = (float(value) for value in box)
    intersects = x1 > 0 and y1 > 0 and x0 < spec.image_width and y0 < spec.image_height
    return {
        'expected_x0': x0, 'expected_y0': y0,
        'expected_x1': x1, 'expected_y1': y1,
        'nominal_in_frame': int(intersects),
    }


def _camera_specs(world_profiles: Path, world: str) -> tuple[tuple[CameraSpec, ...], dict, str]:
    specs = []
    base_profile = None
    world_path = ''
    for camera_id, model in CAMERAS:
        profile, intrinsics, resolved_world, pose = load_profile(
            str(world_profiles), str(world), camera_model=model
        )
        base_profile = profile if base_profile is None else base_profile
        world_path = resolved_world
        specs.append(CameraSpec(
            camera_id=camera_id,
            model=model,
            image_topic=f'/{model}/image_raw',
            labels_topic=f'/{model}/segmentation/labels_map',
            camera=_camera_from_profile(pose, intrinsics),
            pose=tuple(float(value) for value in pose),
            image_width=int(intrinsics['img_width']),
            image_height=int(intrinsics['img_height']),
        ))
    assert base_profile is not None
    return tuple(specs), base_profile, world_path


def _pose_plan(profile: dict, world_path: str, args: argparse.Namespace) -> tuple[list[dict], dict]:
    vis = dict(profile.get('visibility_defaults') or {})
    xmin = float(vis['visibility_map_min_x']) + float(args.wall_margin)
    xmax = float(vis['visibility_map_max_x']) - float(args.wall_margin)
    ymin = float(vis['visibility_map_min_y']) + float(args.wall_margin)
    ymax = float(vis['visibility_map_max_y']) - float(args.wall_margin)
    known = list(profile.get('known_2d_regions') or [])
    traversable = [
        item for item in known
        if str(item.get('type', '')).strip().lower() == 'traversable'
    ]
    excluded = [
        item for item in known
        if str(item.get('type', '')).strip().lower() not in {'traversable', 'site_boundary'}
    ]
    # Every collision object the world profile names: the model groups AND the
    # top-level included objects (forklift, pallets, bin, pallet jack). Missing the
    # latter let 46 top-up poses put the robot inside an object.
    prisms = profile_collision_scene(world_path, profile).prisms
    def filter_records(records: list[dict]) -> tuple[list[dict], dict[str, int]]:
        return _filter_pose_records(
            records,
            traversable_regions=traversable,
            excluded_regions=excluded,
            region_shrink_m=float(args.region_shrink_m),
            collision_prisms=tuple(prisms),
            collision_clearance_m=float(args.collision_clearance_m),
            camera_xy=(0.0, 0.0),
            min_camera_range_m=0.0,
            max_camera_range_m=float('inf'),
        )

    if args.pose_file is not None:
        pose_file = args.pose_file.expanduser().resolve()
        payload = json.loads(pose_file.read_text(encoding='utf-8'))
        source = payload.get('poses') if isinstance(payload, dict) else payload
        if not isinstance(source, list) or not source:
            raise ValueError('--pose-file must contain a non-empty JSON list or {"poses": [...]}')
        records = []
        positions: dict[tuple[float, float], int] = {}
        for index, item in enumerate(source):
            if not isinstance(item, dict):
                raise ValueError(f'pose-file entry {index} is not an object')
            x, y, yaw = (float(item[key]) for key in ('x', 'y', 'yaw'))
            position_key = (x, y)
            if position_key not in positions:
                positions[position_key] = len(positions)
            records.append({
                'x': x, 'y': y, 'yaw': yaw,
                'x_idx': index, 'y_idx': index,
                # A predeclared file carries its own heading index.  Falling back to
                # the flat entry index keeps older single-heading files working.
                'yaw_idx': int(item.get('heading_id', index)),
                'position_id': positions[position_key],
                'dataset_split': str(item.get('stratum', 'commissioning')),
                'random_draw_index': '',
            })

        if args.pose_validity == 'footprint':
            # The declared known_2d_regions encode the PLANNER's keep-out envelope:
            # where the robot may drive.  A capture pose only requires that the body
            # physically fits, so validity here is the oriented footprint against the
            # collision scene.  The planner's envelope is deliberately not consulted.
            footprint = RectangularFootprint(
                tuple(prisms),
                length=float(args.robot_length_m),
                width=float(args.robot_width_m),
            )
            counts: Counter[str] = Counter()
            kept = []
            for record in records:
                clearance = footprint.clearance(
                    (float(record['x']), float(record['y']), float(record['yaw']))
                )
                if clearance < float(args.body_clearance_m):
                    counts['body_clearance'] += 1
                    continue
                record['body_clearance_m'] = round(float(clearance), 4)
                counts['kept'] += 1
                kept.append(record)
            if not kept:
                raise RuntimeError('pose-file left no footprint-valid entries')
            sampling = 'predeclared_pose_file_footprint_validity'
        else:
            kept, counts = filter_records(records)
            if len(kept) != len(records):
                rejected = sorted(set(range(len(records))) - {int(row['x_idx']) for row in kept})
                raise RuntimeError(f'pose-file contains collision-invalid entries: {rejected}')
            sampling = 'predeclared_pose_file'

        return kept, {
            'bounds': [xmin, xmax, ymin, ymax],
            'filter_counts': dict(counts),
            'sampling': sampling,
            'pose_validity': str(args.pose_validity),
            'robot_length_m': float(args.robot_length_m),
            'robot_width_m': float(args.robot_width_m),
            'body_clearance_m': float(args.body_clearance_m),
            'pose_file': str(pose_file),
            'pose_file_sha256': _sha256(pose_file),
            'pose_labels': [str(item.get('stratum', f'pose_{index}'))
                            for index, item in enumerate(source)],
        }

    random_per_split = int(args.random_poses_per_split)
    if random_per_split > 0:
        if int(args.max_poses) > 0:
            raise ValueError('--max-poses cannot be combined with --random-poses-per-split')
        tile_m = float(args.spatial_holdout_tile_m)
        if tile_m <= 0.0:
            raise ValueError('--spatial-holdout-tile-m must be positive')
        rng = np.random.default_rng(int(args.random_seed))
        draw_count = max(20 * random_per_split, 10000)
        records = []
        for draw_index in range(draw_count):
            records.append({
                'x': float(rng.uniform(xmin, xmax)),
                'y': float(rng.uniform(ymin, ymax)),
                'yaw': float(rng.uniform(-math.pi, math.pi)),
                'x_idx': draw_index,
                'y_idx': draw_index,
                'yaw_idx': draw_index,
                'random_draw_index': draw_index,
            })
        kept, counts = filter_records(records)
        pools = {'train': [], 'validation': []}
        for record in kept:
            tile_x = math.floor((float(record['x']) - xmin) / tile_m)
            tile_y = math.floor((float(record['y']) - ymin) / tile_m)
            split = 'train' if (tile_x + tile_y) % 2 == 0 else 'validation'
            record['dataset_split'] = split
            pools[split].append(record)
        shortages = {
            split: len(rows) for split, rows in pools.items()
            if len(rows) < random_per_split
        }
        if shortages:
            raise RuntimeError(
                f'Not enough collision-free random poses for requested split sizes: {shortages}'
            )
        chosen = pools['train'][:random_per_split] + pools['validation'][:random_per_split]
        order = rng.permutation(len(chosen))
        kept = [chosen[int(index)] for index in order]
        for position_id, record in enumerate(kept):
            record['position_id'] = position_id
        return kept, {
            'bounds': [xmin, xmax, ymin, ymax],
            'filter_counts': counts,
            'sampling': 'seeded_continuous_uniform_xy_yaw_after_collision_filter',
            'random_seed': int(args.random_seed),
            'random_poses_per_split': random_per_split,
            'spatial_split': f'{tile_m:g} m checkerboard tiles',
            'spatial_holdout_tile_m': tile_m,
            'random_candidate_draws': draw_count,
            'eligible_by_split': {key: len(value) for key, value in pools.items()},
        }

    records = build_pose_records(
        np.linspace(xmin, xmax, int(args.sample_nx)),
        np.linspace(ymin, ymax, int(args.sample_ny)),
        np.asarray(evenly_spaced_yaws(int(args.yaw_samples)), dtype=float),
    )
    kept, counts = filter_records(records)
    if int(args.max_poses) > 0:
        kept = kept[:int(args.max_poses)]
    positions = {}
    for record in kept:
        key = (int(record['x_idx']), int(record['y_idx']))
        if key not in positions:
            positions[key] = len(positions)
        record['position_id'] = positions[key]
    return kept, {'bounds': [xmin, xmax, ymin, ymax], 'filter_counts': counts}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--world', default='warehouse_v2.world.sdf')
    parser.add_argument('--world-profiles', type=Path, default=REPO / 'src/experiments/config/world_profiles.yaml')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--sample-nx', type=int, default=33)
    parser.add_argument('--sample-ny', type=int, default=28)
    parser.add_argument('--yaw-samples', type=int, default=8)
    parser.add_argument(
        '--pose-file', type=Path,
        help='Predeclared JSON poses [{x,y,yaw,stratum}, ...]; overrides grid/random sampling.',
    )
    parser.add_argument(
        '--random-poses-per-split', type=int, default=0,
        help='Use this many continuous random poses in each frozen train/validation split.',
    )
    parser.add_argument('--random-seed', type=int, default=20260901)
    parser.add_argument('--spatial-holdout-tile-m', type=float, default=2.0)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--max-poses', type=int, default=0)
    parser.add_argument('--wall-margin', type=float, default=0.65)
    parser.add_argument('--region-shrink-m', type=float, default=0.05)
    parser.add_argument('--collision-clearance-m', type=float, default=0.25)
    parser.add_argument('--robot-z', type=float, default=0.0)
    parser.add_argument('--robot-label', type=int, default=23)
    parser.add_argument('--settle-s', type=float, default=0.80)
    parser.add_argument('--image-timeout-s', type=float, default=8.0)
    parser.add_argument('--sync-slop-ms', type=float, default=60.0)
    parser.add_argument(
        '--batch-sync-slop-ms', type=float, default=50.0,
        help='Maximum timestamp span across the five selected RGB frames.',
    )
    parser.add_argument('--min-new-rgb-frames', type=int, default=3)
    parser.add_argument('--min-new-label-frames', type=int, default=1)
    parser.add_argument('--buffer-size', type=int, default=90)
    parser.add_argument('--max-attempts', type=int, default=3)
    parser.add_argument(
        '--rgb-noise-stddev-dn', type=float, default=0.0,
        help=(
            'Optional independent Gaussian RGB read noise, in 8-bit digital-number units, '
            'applied after ROS transport and before hashing/saving/detector inference.'
        ),
    )
    parser.add_argument(
        '--rgb-noise-seed', type=int, default=20260903,
        help='Base seed for --rgb-noise-stddev-dn; each camera-pose-repeat has its own stream.',
    )
    parser.add_argument(
        '--with-semantic', action='store_true',
        help='Also wait for semantic-label images. Optional diagnostic; not used by YOLO.',
    )
    parser.add_argument(
        '--pose-validity', choices=('declared_regions', 'footprint'),
        default='declared_regions',
        help=(
            'How a --pose-file entry is judged valid. "declared_regions" (default) '
            'applies the planner keep-out map from world_profiles.yaml. "footprint" '
            'instead requires only that the oriented robot body clears the collision '
            'scene, which is the physical constraint on standing still to be imaged.'
        ),
    )
    parser.add_argument('--robot-length-m', type=float, default=0.80)
    parser.add_argument('--robot-width-m', type=float, default=0.55)
    parser.add_argument(
        '--body-clearance-m', type=float, default=0.05,
        help='Minimum oriented-body clearance required under --pose-validity footprint.',
    )
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume a status=running capture after its last successful pose batch.',
    )
    args = parser.parse_args()

    transport = _capture_transport_environment(allow_unisolated_transport=False)
    specs, profile, world_path = _camera_specs(args.world_profiles, args.world)
    poses, plan_meta = _pose_plan(profile, world_path, args)
    planned_rows = len(poses) * max(int(args.repeats), 1) * len(specs)
    plan = {
        'world': str(args.world),
        'pose_count': len(poses),
        'position_count': len({int(row['position_id']) for row in poses}),
        'heading_count': (
            'predeclared' if args.pose_file is not None
            else ('continuous' if int(args.random_poses_per_split) > 0 else int(args.yaw_samples))
        ),
        'camera_count': len(specs),
        'repeats': max(int(args.repeats), 1),
        'planned_rows': planned_rows,
        'batch_sync_slop_ms': float(args.batch_sync_slop_ms),
        **plan_meta,
    }
    if args.plan_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=bool(args.resume))
    with capture_lock(out):
        return _run_capture(args, out, specs, profile, world_path, poses, plan, transport)


def _append_attempt(out: Path, record: dict) -> None:
    with (out / 'capture_attempts.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def _run_capture(args, out, specs, profile, world_path, poses, plan, transport) -> int:
    manifest_path = out / 'capture_manifest.json'
    script = Path(__file__).resolve()
    helper = REPO / 'pipeline/capture/capture_yolo_dataset.py'
    expected_manifest = {
        'status': 'running',
        'schema': 'bbox_characterization_capture.v3',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'plan': plan,
        'transport': transport,
        'pose_plan': [{k: float(r[k]) if k in ('x', 'y', 'yaw') else int(r[k])
                       for k in ('x', 'y', 'yaw', 'position_id', 'yaw_idx')} for r in poses],
        'capture_policy': {k: getattr(args, k) for k in (
            'robot_z', 'robot_label', 'settle_s', 'image_timeout_s', 'sync_slop_ms',
            'batch_sync_slop_ms', 'min_new_rgb_frames', 'min_new_label_frames',
            'buffer_size', 'max_attempts', 'with_semantic')},
        'source_files': {rel: _sha256(REPO / rel) for rel in (
            'pipeline/capture/capture_positions.py',
            'pipeline/capture/audit_capture_resume.py',
            'pipeline/capture/capture_yolo_dataset.py',
            'pipeline/capture/dataset_split_utils.py',
            'src/perception/perception/core/ros_image.py',
            'src/experiments/experiments/core/world_profiles.py',
            'src/unav_common/unav_common/capture_integrity.py',
            'src/unav_common/unav_common/camera_model.py')},
        'world_path': str(Path(world_path).resolve()),
        'world_sha256': _sha256(Path(world_path)),
        'world_profiles_path': str(args.world_profiles.resolve()),
        'world_profiles_sha256': _sha256(args.world_profiles.resolve()),
        'capture_script_sha256': _sha256(script),
        'capture_helper_sha256': _sha256(helper),
        'cameras': [
            {
                'camera_id': spec.camera_id, 'camera_model': spec.model,
                'image_topic': spec.image_topic, 'labels_topic': spec.labels_topic,
                'pose_xyz_rpy': list(spec.pose),
                'image_width': spec.image_width, 'image_height': spec.image_height,
            }
            for spec in specs
        ],
        'evaluation_only_inputs': (
            ['commanded_robot_pose', 'semantic_label_map']
            if bool(args.with_semantic) else ['commanded_robot_pose']
        ),
        'sensor_input': 'raw_rgb',
        'sensor_perturbation': {
            'type': (
                'independent_gaussian_rgb_read_noise_after_ros_transport'
                if float(args.rgb_noise_stddev_dn) > 0.0 else 'none'
            ),
            'mean_dn': 0.0,
            'stddev_dn': float(args.rgb_noise_stddev_dn),
            'base_seed': int(args.rgb_noise_seed),
            'clipped_to_uint8': True,
            'detector_input': True,
            'scope': (
                'controlled sensor-read-noise repeat experiment; does not model scene, '
                'illumination, pose, calibration, or renderer variation'
            ),
        },
        'sensor_output_to_be_computed': 'one_frozen_yolo_bounding_box',
        'limited_diagnostic_capture': bool(int(args.max_poses) > 0),
    }
    index_path = out / 'capture_index.csv'
    start_pose_id = 0
    rows_written = 0
    failures = 0
    existing_rows: list[dict[str, str]] = []
    if bool(args.resume):
        if not out.is_dir() or not manifest_path.is_file() or not index_path.is_file():
            raise RuntimeError('--resume requires an existing capture manifest and index')
        report = require_resume(out, expected_manifest)
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        with index_path.open(newline='', encoding='utf-8') as handle:
            existing_rows = list(csv.DictReader(handle))
        start_pose_id = report['next_pose_id']
        start_batch_index = report['next_batch_index']
        rows_written = len(existing_rows)
        failures = len({(r['pose_id'], r['repetition_id']) for r in existing_rows
                        if r['capture_status'] == 'failed'})
        manifest['resume_count'] = int(manifest.get('resume_count', 0)) + 1
        manifest.setdefault('resume_events', []).append({
            'utc': datetime.now(timezone.utc).isoformat(),
            'start_pose_id': start_pose_id,
            'rows_retained': rows_written,
            'retained_failed_batches': failures,
        })
        atomic_json(manifest_path, manifest)
    else:
        if manifest_path.exists() or index_path.exists():
            raise RuntimeError(f'Output already exists: {out}')
        start_batch_index = 0
        for spec in specs:
            (out / spec.camera_id / 'images').mkdir(parents=True, exist_ok=False)
        manifest = expected_manifest
        atomic_json(manifest_path, manifest)

    session_id = uuid.uuid4().hex
    manifest.setdefault('capture_sessions', []).append({
        'capture_session_id': session_id, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'transport': transport, 'start_batch_index': start_batch_index})
    atomic_json(manifest_path, manifest)
    # Exact configuration bytes are retained separately; their hashes remain in the manifest.
    provenance = out / 'provenance'
    provenance.mkdir(exist_ok=True)
    for name, source, expected in (
        ('world.sdf', Path(world_path), manifest['world_sha256']),
        ('world_profiles.yaml', args.world_profiles, manifest['world_profiles_sha256'])):
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError(f'configuration changed while opening capture: {source}')
        dest = provenance / name
        if dest.exists() and dest.read_bytes() != data:
            raise RuntimeError(f'changed capture configuration snapshot: {dest}')
        if not dest.exists():
            atomic_bytes(dest, data)

    camera_ids = tuple(spec.camera_id for spec in specs)
    spec_by_id = {spec.camera_id: spec for spec in specs}
    camera_index_by_id = {camera_id: index for index, camera_id in enumerate(camera_ids)}
    rclpy.init()
    node = FiveCameraCapture(
        world_name=str(profile['world_name']), cameras=specs,
        robot_z=float(args.robot_z), settle_s=float(args.settle_s),
        timeout_s=float(args.image_timeout_s), sync_slop_s=float(args.sync_slop_ms) / 1000.0,
        batch_sync_slop_s=float(args.batch_sync_slop_ms) / 1000.0,
        min_new_rgb=int(args.min_new_rgb_frames), min_new_labels=int(args.min_new_label_frames),
        buffer_size=int(args.buffer_size), with_semantic=bool(args.with_semantic),
    )
    try:
        node.wait_ready(camera_ids)
        image_path_by_hash: dict[tuple[str, str], Path] = {
            (row['camera_id'], row['image_sha1']): out / row['image']
            for row in existing_rows if row['capture_status'] == 'ok' and row['image_sha1']
        }
        writer = CaptureIndexWriter(index_path, FIELDS, len(specs), existing_rows)
        for pose_id, record in enumerate(poses):
            if pose_id < start_pose_id:
                continue
            x, y, yaw = float(record['x']), float(record['y']), float(record['yaw'])
            for repetition in range(max(int(args.repeats), 1)):
                if pose_id * plan['repeats'] + repetition < start_batch_index:
                    continue
                pairs = None
                error = ''
                for attempt in range(max(int(args.max_attempts), 1)):
                    attempt_id = f'{session_id}/{pose_id}:{repetition}/{attempt}'
                    attempt_record = dict(attempt_id=attempt_id, pose_id=pose_id,
                                          repetition_id=repetition, x=x, y=y, yaw=yaw,
                                          status='started', utc=datetime.now(timezone.utc).isoformat())
                    _append_attempt(out, attempt_record)
                    try:
                        pairs = node.capture(x, y, yaw, camera_ids)
                        _append_attempt(out, dict(attempt_record, status='captured',
                            capture_context=node.capture_context,
                            camera_stamps_ns={c: p.image_stamp_ns for c, p in pairs.items()}))
                        break
                    except CaptureEpochError as exc:
                        _append_attempt(out, dict(attempt_record, status='epoch_aborted', reason=str(exc)))
                        raise
                    except Exception as exc:  # pragma: no cover - ROS transport path
                        error = f'attempt_{attempt}:{exc}'
                        _append_attempt(out, dict(attempt_record, status='failed', reason=str(exc)))
                if pairs is None:
                    failures += 1
                    for camera_id in camera_ids:
                        spec = spec_by_id[camera_id]
                        row = {field: '' for field in FIELDS}
                        row.update({
                            'pose_id': pose_id, 'position_id': int(record['position_id']),
                            'x_idx': int(record['x_idx']), 'y_idx': int(record['y_idx']),
                            'heading_id': int(record['yaw_idx']), 'repetition_id': repetition,
                            'source_batch_id': f'pose_{pose_id:06d}_r{repetition:02d}',
                            'capture_session_id': session_id,
                            'dataset_split': record.get('dataset_split', ''),
                            'random_draw_index': record.get('random_draw_index', ''),
                            'camera_id': camera_id, 'camera_model': spec.model,
                            'capture_status': 'failed', 'capture_error': error,
                            'robot_x': x, 'robot_y': y, 'robot_yaw': yaw,
                        })
                        writer.writerow(row)
                        rows_written += 1
                    writer.flush()
                    continue
                stamps = [pair.image_stamp_ns for pair in pairs.values()]
                batch_span_s = (max(stamps) - min(stamps)) * 1e-9
                for camera_id, pair in pairs.items():
                    spec = spec_by_id[camera_id]
                    detector_image = _add_rgb_read_noise(
                        pair.image,
                        stddev_dn=float(args.rgb_noise_stddev_dn),
                        seed=int(args.rgb_noise_seed),
                        pose_id=pose_id,
                        repetition_id=repetition,
                        camera_index=camera_index_by_id[camera_id],
                    )
                    image_sha1 = _sha1_array(detector_image)
                    image_path = image_path_by_hash.get((camera_id, image_sha1))
                    if image_path is None:
                        image_path = (
                            out / camera_id / 'images'
                            / f'pose_{pose_id:06d}_r{repetition:02d}.png'
                        )
                        ok, encoded = cv2.imencode('.png', detector_image)
                        if not ok:
                            raise RuntimeError(f'Failed to encode {image_path}')
                        atomic_bytes(image_path, encoded.tobytes())
                        image_path_by_hash[(camera_id, image_sha1)] = image_path
                    expected = _expected_geometry(spec, x, y, yaw)
                    row = {field: '' for field in FIELDS}
                    row.update({
                        'pose_id': pose_id, 'position_id': int(record['position_id']),
                        'x_idx': int(record['x_idx']), 'y_idx': int(record['y_idx']),
                        'heading_id': int(record['yaw_idx']), 'repetition_id': repetition,
                        'source_batch_id': f'pose_{pose_id:06d}_r{repetition:02d}',
                        'capture_session_id': session_id,
                        'dataset_split': record.get('dataset_split', ''),
                        'random_draw_index': record.get('random_draw_index', ''),
                        'camera_id': camera_id, 'camera_model': spec.model,
                        'image': str(image_path.relative_to(out)),
                        'capture_status': 'ok', 'capture_error': '',
                        'robot_x': x, 'robot_y': y, 'robot_yaw': yaw,
                        'camera_range_m': math.hypot(x - spec.pose[0], y - spec.pose[1]),
                        'image_stamp_s': pair.image_stamp_ns * 1e-9,
                        'label_stamp_s': (
                            pair.label_stamp_ns * 1e-9
                            if pair.label_stamp_ns is not None else ''
                        ),
                        'stamp_delta_s': pair.stamp_delta_s,
                        'batch_image_span_s': batch_span_s,
                        'image_sha1': image_sha1,
                        'raw_image_sha1': _sha1_array(pair.image),
                        'image_stamp_ns': pair.image_stamp_ns,
                        'image_id': f'{session_id}/{camera_id}@{pair.image_stamp_ns}',
                        **node.capture_context,
                        **expected,
                    })
                    if pair.labels is not None:
                        row.update(_mask_geometry(pair.labels, int(args.robot_label)))
                    writer.writerow(row)
                    rows_written += 1
                writer.flush()
            if (pose_id + 1) % 25 == 0:
                print(
                    f'captured {pose_id + 1}/{len(poses)} poses; '
                    f'rows={rows_written} failed_batches={failures}',
                    flush=True,
                )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    manifest.update({
        'status': 'complete' if failures == 0 else 'complete_with_failed_batches',
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'rows_written': rows_written,
        'failed_batches': failures,
        'capture_index_sha256': _sha256(index_path),
    })
    atomic_json(manifest_path, manifest)
    print(json.dumps({key: manifest[key] for key in ('status', 'rows_written', 'failed_batches')}, indent=2))
    return 0 if failures == 0 else 2


if __name__ == '__main__':
    raise SystemExit(main())
