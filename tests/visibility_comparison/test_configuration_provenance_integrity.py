"""Regression checks for strict campaign identity and durable attempt ownership."""

import importlib.util
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    'configuration_campaign',
    ROOT / 'pipeline/campaign_runner.py',
)
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


def _config(tmp_path: Path) -> dict:
    network = tmp_path / 'network.npz'
    detector = tmp_path / 'detector.pt'
    network.write_bytes(b'network')
    detector.write_bytes(b'detector')
    return {
        'world': 'warehouse_v2.world.sdf',
        'launch_file': 'warehouse_primary_comparison.launch.py',
        'conditions': {'P2': {'camera_network_artifact_path': str(network)}},
        'tasks': {'task': {'conditions': ['P2'], 'seeds': [0]}},
        'yolo_model': str(detector),
        'horizon': 1,
        'dt': 0.25,
        'goal_success_radius': 0.35,
        'run_timeout_after_first_cmd_s': 1,
        'global_planner_mode': 'efe',
    }


def test_typo_and_duplicate_yaml_keys_are_rejected(tmp_path):
    cfg = _config(tmp_path)
    cfg['run_timeout_after_first_cmd_typo_s'] = 4
    with pytest.raises(ValueError, match='unknown campaign keys'):
        campaign._validate_config(cfg, tmp_path / 'campaign.yaml')
    duplicate = tmp_path / 'duplicate.yaml'
    duplicate.write_text('world: one\nworld: two\n')
    with pytest.raises(ValueError, match='duplicate YAML key'):
        campaign._load_config(duplicate)


def test_false_and_zero_survive_precedence(tmp_path):
    cfg = _config(tmp_path)
    cfg['use_rviz'] = True
    cfg['optimizer_maxiter'] = 12
    cfg['tasks']['task']['use_rviz'] = False
    cfg['conditions']['P2']['optimizer_maxiter'] = 0
    resolved = campaign._resolved_cell_config(cfg, 'task', 'P2')
    assert resolved['use_rviz'] is False
    assert resolved['optimizer_maxiter'] == 0


def test_one_attempt_cannot_select_another_summary(tmp_path):
    attempt = tmp_path / 'attempt'
    (attempt / 'experiment_a').mkdir(parents=True)
    (attempt / 'experiment_b').mkdir()
    with pytest.raises(RuntimeError, match='multiple experiment runs'):
        campaign._attempt_run_dir(attempt)


def test_simultaneous_campaign_leases_are_refused(tmp_path):
    lease_path = tmp_path / 'resource.lock'
    with campaign._exclusive_lease(lease_path, {'campaign': 'first'}):
        with pytest.raises(RuntimeError, match='resource lease already held'):
            with campaign._exclusive_lease(lease_path, {'campaign': 'second'}):
                pass


@pytest.mark.parametrize('field', [
    'completed', 'valid_run', 'data_files_closed',
])
def test_partial_summary_is_not_completed(field):
    summary = {
        'completed': True,
        'valid_run': True,
        'evidence_complete': True,
        'data_files_closed': True,
        'completion_reason': 'goal_reached',
    }
    summary[field] = False
    assert campaign._terminal_summary_outcome(summary)[0] is False


def test_logger_evidence_requires_terminal_stop_and_producer_close_verdict():
    summary = {
        'completed': True,
        'valid_run': True,
        'evidence_complete': False,
        'data_files_closed': True,
        'completion_reason': 'goal_reached',
    }
    assert campaign._terminal_summary_outcome(summary) == (
        False, 'infra_invalid', 'summary_evidence_complete_not_true'
    )


def test_detector_journal_requires_durable_close_marker(tmp_path):
    from unav_common.camera_outcomes import OutcomeJournal

    journal = OutcomeJournal(tmp_path / 'detector_outcomes.jsonl', 'producer')
    journal.append({'status': 'published'})
    ok, verdict = campaign._verify_detector_journal(tmp_path)
    assert not ok and verdict['reason'] == 'detector_session_not_closed'
    journal.append({'status': 'session_stopped'})
    journal.close()
    ok, verdict = campaign._verify_detector_journal(tmp_path)
    assert ok and verdict['event_count'] == 2


def test_empty_but_complete_correction_ledger_is_accounted(tmp_path):
    (tmp_path / 'correction_publications.csv').write_text(
        'source_batch_id,correction_stamp,correction_stamp_ns\n'
    )
    (tmp_path / 'correction_assimilations.csv').write_text(
        'source_batch_id,correction_stamp,apply_stamp,status,reason,accepted\n'
    )
    assert campaign._verify_correction_assimilations(tmp_path) == (True, '')


def test_malformed_campaign_ledger_is_not_treated_as_empty(tmp_path):
    ledger = tmp_path / 'campaign_log.json'
    ledger.write_text('{broken')
    with pytest.raises(RuntimeError, match='malformed campaign ledger'):
        campaign._load_run_log(ledger)


def test_duplicate_snapshot_basenames_are_preserved(tmp_path):
    from experiments.core import manifest

    first = tmp_path / 'a' / 'same.yaml'
    second = tmp_path / 'b' / 'same.yaml'
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text('a: 1\n')
    second.write_text('b: 2\n')
    run = tmp_path / 'run'
    run.mkdir()
    snapshots = manifest.snapshot_configs(str(run), [str(first), str(second)])
    assert len(snapshots) == 2
    assert {Path(path).read_text() for path in snapshots.values()} == {'a: 1\n', 'b: 2\n'}


def test_manifest_provenance_is_cached_per_run(tmp_path, monkeypatch):
    from experiments.core import manifest

    calls = []
    monkeypatch.setattr(
        manifest.common_manifest,
        'git_provenance',
        lambda _root, _paths=None: calls.append((_root, _paths)) or {
            'git_sha': f'sha-{len(calls)}'
        },
    )
    manifest._RUN_PROVENANCE_CACHE.clear()
    run_a = tmp_path / 'a'
    run_b = tmp_path / 'b'
    manifest.write_manifest(str(run_a), {}, str(tmp_path))
    manifest.write_manifest(str(run_a), {'rewrite': True}, str(tmp_path))
    manifest.write_manifest(str(run_b), {}, str(tmp_path))
    assert len(calls) == 2
    assert json.loads((run_a / 'run_manifest.json').read_text())['git_sha'] == 'sha-1'
    assert json.loads((run_b / 'run_manifest.json').read_text())['git_sha'] == 'sha-2'


def test_manifest_uses_campaign_executable_scope(tmp_path, monkeypatch):
    from experiments.core import manifest

    calls = []
    monkeypatch.setenv(
        'UNAV_EXECUTABLE_SOURCE_PATHS',
        os.pathsep.join(('src', 'scripts/visibility_comparison')),
    )
    monkeypatch.setattr(
        manifest.common_manifest,
        'git_provenance',
        lambda root, paths=None: calls.append((root, paths)) or {'git_sha': 'scoped'},
    )
    manifest._RUN_PROVENANCE_CACHE.clear()
    manifest.write_manifest(str(tmp_path / 'run'), {}, str(tmp_path))
    assert calls == [
        (str(tmp_path.resolve()), ('src', 'scripts/visibility_comparison'))
    ]
