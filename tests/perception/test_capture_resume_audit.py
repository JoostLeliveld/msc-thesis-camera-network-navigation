from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    'resume_audit', ROOT / 'pipeline/capture/audit_capture_resume.py'
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def capture(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE.shutil, 'disk_usage', lambda _: SimpleNamespace(free=10**10))
    repo = tmp_path / 'repo'
    source_paths = [repo / name for name in (
        'poses.json', 'world.sdf', 'profiles.yaml',
        'pipeline/capture/capture_positions.py',
        'pipeline/capture/capture_yolo_dataset.py',
    )]
    for path in source_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('frozen')
    p = tmp_path / 'capture'
    p.mkdir()
    manifest = {
        'status': 'running',
        'plan': {'pose_count': 3, 'planned_rows': 15, 'repeats': 1,
                 'batch_sync_slop_ms': 50, 'pose_file': str(source_paths[0]),
                 'pose_file_sha256': MODULE.sha256(source_paths[0])},
        'cameras': [{'camera_id': f'camera_{c}'} for c in 'ABCDE'],
    }
    for key, path in zip(['world', 'world_profiles', 'capture_script', 'capture_helper'], source_paths[1:]):
        manifest[key + '_sha256'] = MODULE.sha256(path)
        manifest[key + '_path'] = str(path)
    (p / 'capture_manifest.json').write_text(json.dumps(manifest))
    pixels = np.arange(12 * 12 * 3, dtype=np.uint8).reshape(12, 12, 3)
    assert cv2.imwrite(str(p / 'image.png'), pixels)
    rows = [dict(pose_id=0, repetition_id=0, camera_id=f'camera_{c}',
                 source_batch_id='pose_000000_r00', capture_status='ok',
                 image='image.png', image_sha1=MODULE.image_hash(pixels),
                 image_stamp_s=1, batch_image_span_s=0) for c in 'ABCDE']
    write_rows(p, rows)
    return p, repo, rows


def write_rows(p, rows):
    with (p / 'capture_index.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_valid_prefix_is_ready(capture):
    p, repo, _ = capture
    result = MODULE.audit(p, repo=repo)
    assert result['ready']
    assert result['next_pose_id'] == 1
    assert result['remaining_camera_opportunities'] == 10


@pytest.mark.parametrize('defect', ['duplicate_camera', 'missing_camera', 'pose_gap', 'timing', 'corrupt_image', 'changed_code'])
def test_refuses_unsafe_resume(capture, defect):
    p, repo, rows = capture
    if defect == 'duplicate_camera':
        rows[-1]['camera_id'] = rows[0]['camera_id']
    elif defect == 'missing_camera':
        rows.pop()
    elif defect == 'pose_gap':
        for row in rows:
            row['pose_id'] = 1
            row['source_batch_id'] = 'pose_000001_r00'
    elif defect == 'timing':
        rows[-1]['image_stamp_s'] = 1.2  # reported span remains falsely zero
    elif defect == 'corrupt_image':
        (p / 'image.png').write_bytes(b'broken')
    elif defect == 'changed_code':
        (repo / 'pipeline/capture/capture_yolo_dataset.py').write_text('changed')
    write_rows(p, rows)
    assert not MODULE.audit(p, repo=repo)['ready']


def test_refuses_low_space_and_does_not_modify_capture(capture, monkeypatch):
    p, repo, _ = capture
    before = {f.name: f.read_bytes() for f in p.iterdir()}
    monkeypatch.setattr(MODULE.shutil, 'disk_usage', lambda _: SimpleNamespace(free=1))
    result = MODULE.audit(p, repo=repo)
    assert not result['checks']['space_for_remaining_without_deduplication']
    assert before == {f.name: f.read_bytes() for f in p.iterdir()}


def test_webp_is_verified_but_budget_uses_png(capture):
    p, repo, rows = capture
    pixels = cv2.imread(str(p / 'image.png'))
    assert cv2.imwrite(str(p / 'image.webp'), pixels, [cv2.IMWRITE_WEBP_QUALITY, 101])
    for row in rows:
        row['image'] = 'image.webp'
    write_rows(p, rows)
    result = MODULE.audit(p, repo=repo, reserve_bytes=0)
    assert result['ready']
    _, encoded = cv2.imencode('.png', pixels)
    assert result['space']['estimated_required_bytes'] >= 10 * len(encoded)
