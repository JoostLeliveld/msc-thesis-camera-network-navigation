#!/usr/bin/env python3
"""Deterministic grouped split helpers for perception datasets."""

from __future__ import annotations

import hashlib
import math
from typing import Iterable

import numpy as np


def evenly_spaced_yaws(yaw_count: int) -> list[float]:
    yaw_count = max(int(yaw_count), 1)
    return [float(v) for v in np.linspace(0.0, 2.0 * math.pi, yaw_count, endpoint=False)]


def yaw_bucket_index(yaw_rad: float, bucket_count: int) -> int:
    bucket_count = max(int(bucket_count), 1)
    wrapped = float(yaw_rad) % (2.0 * math.pi)
    return int(math.floor((wrapped / (2.0 * math.pi)) * bucket_count)) % bucket_count


def build_pose_records(
    xs: Iterable[float],
    ys: Iterable[float],
    yaws: Iterable[float],
    *,
    yaw_bucket_count: int | None = None,
) -> list[dict[str, float | int]]:
    yaw_values = [float(v) for v in yaws]
    bucket_count = int(yaw_bucket_count) if yaw_bucket_count is not None else max(len(yaw_values), 1)
    records: list[dict[str, float | int]] = []
    for y_idx, y in enumerate(float(v) for v in ys):
        for x_idx, x in enumerate(float(v) for v in xs):
            for yaw_idx, yaw in enumerate(yaw_values):
                records.append({
                    'x': float(x),
                    'y': float(y),
                    'yaw': float(yaw),
                    'x_idx': int(x_idx),
                    'y_idx': int(y_idx),
                    'yaw_idx': int(yaw_idx),
                    'yaw_bucket': int(yaw_bucket_index(yaw, bucket_count)),
                })
    return records


def _stable_rank(key: str, seed: int) -> int:
    digest = hashlib.sha1(f'{int(seed)}|{key}'.encode('utf-8')).hexdigest()
    return int(digest[:12], 16)


def _group_key(
    record: dict[str, float | int],
    *,
    split_mode: str,
    spatial_block_size: int,
) -> str | None:
    split_mode = str(split_mode).strip().lower()
    if split_mode == 'cyclic':
        return None
    if split_mode == 'yaw_bucket':
        return f"yaw:{int(record['yaw_bucket'])}"
    x_block = int(record['x_idx']) // max(int(spatial_block_size), 1)
    y_block = int(record['y_idx']) // max(int(spatial_block_size), 1)
    if split_mode == 'spatial_cell':
        return f"xy:{x_block}:{y_block}"
    if split_mode == 'spatial_yaw_bucket':
        return f"xyyaw:{x_block}:{y_block}:{int(record['yaw_bucket'])}"
    raise ValueError(f'Unsupported split_mode: {split_mode!r}')


def assign_splits(
    records: list[dict[str, float | int]],
    *,
    val_fraction: float,
    split_mode: str,
    seed: int = 0,
    spatial_block_size: int = 2,
) -> list[str]:
    val_fraction = float(val_fraction)
    if not math.isfinite(val_fraction) or not 0 <= val_fraction <= 1:
        raise ValueError('val_fraction must be finite and in [0, 1]')
    split_mode = str(split_mode).strip().lower()
    if not records:
        return []
    if val_fraction in (0.0, 1.0):
        return ['val' if val_fraction == 1.0 else 'train'] * len(records)
    if split_mode == 'cyclic':
        val_every = max(int(round(1.0 / max(val_fraction, 1e-6))), 2)
        return ['val' if (idx % val_every == 0) else 'train' for idx in range(len(records))]

    group_to_indices: dict[str, list[int]] = {}
    for idx, record in enumerate(records):
        key = _group_key(record, split_mode=split_mode, spatial_block_size=spatial_block_size)
        if key is None:
            raise RuntimeError('Grouped split expected a non-null key')
        group_to_indices.setdefault(key, []).append(idx)

    group_keys = sorted(group_to_indices.keys())
    if len(group_keys) == 1:
        return ['train'] * len(records)

    n_val_groups = int(round(val_fraction * len(group_keys)))
    n_val_groups = max(1, min(n_val_groups, len(group_keys) - 1))
    ranked = sorted(group_keys, key=lambda key: (_stable_rank(key, int(seed)), key))
    val_groups = set(ranked[:n_val_groups])
    return ['val' if _group_key(record, split_mode=split_mode, spatial_block_size=spatial_block_size) in val_groups else 'train' for record in records]
