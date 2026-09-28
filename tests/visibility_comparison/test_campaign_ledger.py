"""Interrupted campaign persistence must preserve the last complete ledger."""
import importlib.util
from pathlib import Path
import json
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('ledger_campaign',ROOT/'pipeline/campaign_runner.py')
campaign=importlib.util.module_from_spec(spec);spec.loader.exec_module(campaign)

def test_failed_replacement_preserves_completed_runs(tmp_path,monkeypatch):
    ledger=tmp_path/'campaign.json'
    old={'completed_seed':{'outcome':'goal_reached'}}
    campaign._save_run_log(ledger,old)
    def interrupted(*args):raise OSError('replacement interrupted')
    monkeypatch.setattr(campaign.os,'replace',interrupted)
    with pytest.raises(OSError):campaign._save_run_log(ledger,{'new_seed':{}})
    assert json.loads(ledger.read_text())==old
    assert list(tmp_path.iterdir())==[ledger]

def test_successful_replacement_keeps_all_entries(tmp_path):
    ledger=tmp_path/'campaign.json'
    campaign._save_run_log(ledger,{'seed110':{'outcome':'goal_reached'}})
    new={'seed110':{'outcome':'goal_reached'},'seed111':{'outcome':None}}
    campaign._save_run_log(ledger,new)
    assert json.loads(ledger.read_text())==new


def test_scoped_cleanup_only_selects_exact_run_marker(tmp_path):
    for pid, environment in {'100': b'UNAV_CAMPAIGN_RUN_TOKEN=ours\0ROS_DOMAIN_ID=191\0',
                             '101': b'UNAV_CAMPAIGN_RUN_TOKEN=ours_other\0',
                             '102': b'UNAV_CAMPAIGN_RUN_TOKEN=theirs\0',
                             '103': b'OTHER=UNAV_CAMPAIGN_RUN_TOKEN=ours\0'}.items():
        path = tmp_path / pid; path.mkdir(); (path / 'environ').write_bytes(environment)
    assert campaign._pids_with_run_token('ours', tmp_path) == [100]
    with pytest.raises(ValueError): campaign._pids_with_run_token('', tmp_path)
