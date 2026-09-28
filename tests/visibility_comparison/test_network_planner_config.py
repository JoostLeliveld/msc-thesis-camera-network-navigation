"""A network run must actually solve with its declared artifact, not a legacy GP."""
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('network_campaign',ROOT/'pipeline/campaign_runner.py')
campaign=importlib.util.module_from_spec(spec);spec.loader.exec_module(campaign)


def config(tmp_path):
    artifact=tmp_path/'network.npz';artifact.write_bytes(b'configuration fixture')
    detector=tmp_path/'detector.pt';detector.write_bytes(b'configuration fixture')
    return dict(world='warehouse_v2.world.sdf',launch_file='warehouse_primary_comparison.launch.py',
        conditions={'spatial_intact':dict(camera_network_artifact_path=str(artifact),
            camera_network_active_camera_ids='camera_A,camera_B,camera_C,camera_D,camera_E')},
        camera_network_objective='metric_expected_belief',
        tasks={'fusion_network_traverse':dict(conditions=['spatial_intact'],seeds=[513])},
        yolo_model=str(detector),horizon=40,dt=.25,goal_success_radius=.35,
        run_timeout_after_first_cmd_s=30,global_planner_mode='efe')


def test_network_configuration_does_not_require_or_send_legacy_gp(tmp_path):
    cfg=config(tmp_path)
    campaign._validate_config(cfg,tmp_path/'config.yaml')
    cmd=campaign._build_launch_cmd(cfg,'fusion_network_traverse','spatial_intact',513,tmp_path/'logs')
    assert any(s.startswith('camera_network_artifact_path:=') for s in cmd)
    assert 'planner:=visibility_aware_efe' in cmd
    assert not any(s.startswith('visibility_artifact_path:=') for s in cmd)


def test_preselected_route_cannot_masquerade_as_network_planning(tmp_path):
    cfg=config(tmp_path);cfg['global_planner_mode']='preselected_route'
    with pytest.raises(RuntimeError,match='solved visibility-aware'):
        campaign._validate_config(cfg,tmp_path/'config.yaml')


def test_absent_field_still_requires_explicit_legacy_artifact(tmp_path):
    cfg=config(tmp_path);cfg['conditions']['spatial_intact']={}
    with pytest.raises(RuntimeError,match='gp_artifact'):
        campaign._validate_config(cfg,tmp_path/'config.yaml')


def test_ambiguous_legacy_and_network_artifacts_are_refused(tmp_path):
    cfg=config(tmp_path);cfg['gp_artifact']='old_field.npz'
    with pytest.raises(RuntimeError,match='choose one'):
        campaign._validate_config(cfg,tmp_path/'config.yaml')


def test_node_passes_field_to_global_planner_only():
    # Exercise the actual construction method without starting a ROS graph.
    from planning.nodes.unicycle_planner_node import UnicyclePlannerNode
    class NodeFixture:
        camera_network_artifact_path='/frozen/network.npz'
        camera_network_objective='metric_expected_belief'
        network_goal_std_m=.15
        PLANNER_CLASS=staticmethod(lambda **kwargs:kwargs)
        def __getattr__(self,name):return 1.
    node=NodeFixture()
    for enabled in (False,True):
        def g(key):return enabled if key=='use_visibility_model' else 1.
        result=UnicyclePlannerNode._build_planner_instance(node,g,lambda key,default:default,bool)
        assert result['camera_network_artifact_path']==('/frozen/network.npz' if enabled else '')
        assert result['camera_network_objective']==(
            'metric_expected_belief' if enabled else 'legacy_pixel_chart')


def test_metric_network_mode_is_explicit_in_launch_command(tmp_path):
    cfg=config(tmp_path)
    cfg['camera_network_objective']='metric_expected_belief'
    cfg['network_goal_std_m']=.15
    campaign._validate_config(cfg,tmp_path/'config.yaml')
    cmd=campaign._build_launch_cmd(cfg,'fusion_network_traverse','spatial_intact',513,tmp_path/'logs')
    assert 'camera_network_objective:=metric_expected_belief' in cmd
    assert 'network_goal_std_m:=0.15' in cmd
