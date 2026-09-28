from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / 'pipeline' / 'capture'
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset_split_utils import assign_splits, build_pose_records, yaw_bucket_index


def test_yaw_bucket_index_wraps_into_expected_range() -> None:
    assert yaw_bucket_index(0.0, 8) == 0
    assert yaw_bucket_index(2.0 * 3.141592653589793, 8) == 0
    assert yaw_bucket_index(-0.1, 8) == 7


def test_grouped_spatial_split_produces_both_train_and_val() -> None:
    records = build_pose_records([0.0, 1.0, 2.0, 3.0], [0.0, 1.0], [0.0, 1.57])
    splits = assign_splits(
        records,
        val_fraction=0.25,
        split_mode='spatial_cell',
        seed=0,
        spatial_block_size=2,
    )
    assert 'train' in splits
    assert 'val' in splits


def test_yaw_bucket_split_holds_out_entire_yaw_groups() -> None:
    records = build_pose_records([0.0, 1.0], [0.0, 1.0], [0.0, 1.57, 3.14, 4.71], yaw_bucket_count=4)
    splits = assign_splits(
        records,
        val_fraction=0.25,
        split_mode='yaw_bucket',
        seed=1,
        spatial_block_size=1,
    )
    bucket_to_split = {}
    for record, split in zip(records, splits):
        bucket = int(record['yaw_bucket'])
        bucket_to_split.setdefault(bucket, split)
        assert bucket_to_split[bucket] == split
