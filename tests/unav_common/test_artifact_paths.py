from pathlib import Path
import hashlib
import pytest
from unav_common.artifact_paths import thesis_artifact_path
from reliability.commissioned_visibility import _read_verified


def checkout(tmp_path):
    root = tmp_path / 'relocated'
    (root / 'pipeline').mkdir(parents=True)
    (root / 'pyproject.toml').write_text('')
    return root


def test_relocated_bundle_is_preferred_to_existing_original(tmp_path):
    root = checkout(tmp_path)
    old = tmp_path / 'original/logs/thesis/fits/model.bin'
    old.parent.mkdir(parents=True)
    old.write_bytes(b'wrong original')
    new = root / 'logs/thesis/fits/model.bin'
    new.parent.mkdir(parents=True)
    new.write_bytes(b'locked bytes')
    anchor = new.parent / 'manifest.json'
    entry = {'path': str(old), 'sha256': hashlib.sha256(new.read_bytes()).hexdigest()}
    assert _read_verified(entry, label='test', anchor=anchor) == (new, b'locked bytes')
    new.write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash differs'):
        _read_verified(entry, label='test', anchor=anchor)
    new.unlink()
    with pytest.raises(FileNotFoundError):
        _read_verified(entry, label='test', anchor=anchor)


def test_relative_bundle_path_and_external_path(tmp_path):
    root = checkout(tmp_path)
    anchor = root / 'pipeline/analyze_campaign.py'
    rel = Path('logs/thesis/final_campaign/run')
    assert thesis_artifact_path(rel, anchor=anchor) == root / rel
    external = tmp_path / 'standalone.bin'
    assert thesis_artifact_path(external, anchor=anchor) == external
    with pytest.raises(ValueError, match='escape'):
        thesis_artifact_path('logs/thesis/../../secret', anchor=anchor)
