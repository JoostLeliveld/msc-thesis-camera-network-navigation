"""Independent acceptance/selection regressions for investigation 08 repairs."""
from types import SimpleNamespace as NS
import numpy as np
import pytest

from planning.core.plan_validation import validate_plan_result, validate_covariance
from planning.core.rollout import rollout_unicycle
from test_camera_network import make_planner, write_network


def result_fixture():
    controls=np.array([[.2,0.],[.2,0.]])
    states=rollout_unicycle(np.zeros(3),controls,.25)
    return NS(controls=controls,states=states,total_cost=1.,risk_cost=.4,
              rollout_valid=True,terminal_goal_distance_pred=0.,
              min_predicted_obstacle_distance_m=np.inf,
              optimizer_success=False,optimizer_status=1)


def validate(result, **overrides):
    kwargs=dict(initial_state=np.zeros(3),initial_covariance=np.eye(3)*.01,
        goal_xy=np.array([.1,0.]),dt=.25,control_bounds=((0.,.22),(-1.,1.)),
        expected_horizon=2,require_complete=True,terminal_tolerance_m=.01)
    kwargs.update(overrides)
    return validate_plan_result(result,**kwargs)


def test_iteration_limit_can_be_feasible_and_returns_owned_immutable_data():
    result=result_fixture();checked=validate(result)
    assert checked.valid
    result.controls[0,0]=7.
    result.states[0,0]=8.
    assert checked.controls[0,0]==.2 and checked.states[0,0]==0.
    with pytest.raises(ValueError): checked.controls.setflags(write=True)


@pytest.mark.parametrize('kind',['missing_validity','nan_control','bounds','nan_state',
    'wrong_motion','nan_cost','missing_cost','nan_component','bad_clearance','fake_gap','partial'])
def test_malformed_candidates_are_refused(kind):
    result=result_fixture()
    if kind=='missing_validity':del result.rollout_valid
    elif kind=='nan_control':result.controls[0,0]=np.nan
    elif kind=='bounds':result.controls[0,1]=2.
    elif kind=='nan_state':result.states[1,0]=np.nan
    elif kind=='wrong_motion':result.states[1,0]+=.2
    elif kind=='nan_cost':result.total_cost=np.nan
    elif kind=='missing_cost':del result.total_cost
    elif kind=='nan_component':result.risk_cost=np.inf
    elif kind=='bad_clearance':result.min_predicted_obstacle_distance_m=-np.inf
    elif kind=='fake_gap':result.terminal_goal_distance_pred=1.
    elif kind=='partial':
        result.controls[:]=0.;result.states[:]=0.;result.terminal_goal_distance_pred=.1
    assert not validate(result).valid


@pytest.mark.parametrize('covariance',[np.diag([.1,.1,-.02]),
    np.array([[1.,2.,0.],[2.,1.,0.],[0.,0.,1.]]),
    np.array([[1.,.1,0.],[0.,1.,0.],[0.,0.,1.]]),np.eye(3)*np.nan])
def test_full_prior_is_checked_before_planner_state_changes(tmp_path,covariance):
    net=write_network(tmp_path/'net.npz');planner=make_planner(net.path)
    with pytest.raises(ValueError):planner.plan(np.zeros(3),covariance,[.1,0.])
    assert planner._prev_goal_xy is None and planner.prev_controls_flat is None
    assert not validate(result_fixture(),initial_covariance=covariance).valid


def test_semidefinite_prior_is_a_supported_deterministic_limit():
    np.testing.assert_array_equal(validate_covariance(np.zeros((3,3))),np.zeros((3,3)))
    with pytest.raises(ValueError):validate_covariance(np.zeros((3,3)),positive_definite=True)


def test_geometry_refusal_overrides_success_status():
    result=result_fixture();result.optimizer_success=True;result.optimizer_status=0
    checked=validate(result,geometry_validator=lambda states,controls:(False,'thin obstacle'))
    assert not checked.valid and checked.reason=='thin obstacle'


def test_nonfinite_optimizer_or_candidate_cannot_hide_a_finite_seed(tmp_path,monkeypatch):
    net=write_network(tmp_path/'net.npz');planner=make_planner(net.path)
    original=planner._evaluate_candidate_controls
    calls=[]
    def faulty(*a,**kw):
        value=original(*a,**kw);calls.append(value)
        if len(calls)==2:value['total_cost']=np.nan
        return value
    monkeypatch.setattr(planner,'_evaluate_candidate_controls',faulty)
    monkeypatch.setattr('planning.planners.base_planner.minimize',lambda *a,**kw:
        NS(x=np.tile([.2,0.],5),success=False,status=1,nit=1,nfev=1,message='limit'))
    result=planner.plan(np.zeros(3),np.eye(3)*.01,[.1,0.])
    assert np.isfinite(result.total_cost) and np.isfinite(result.controls).all()
    assert len(calls)==2
    assert np.all(result.controls==0.),'finite seed must replace rejected NaN optimizer candidate'
    assert result.selected_source.startswith('seed:')
    result.controls[0,0]=99.
    assert planner.prev_controls_flat[0]==0.,'result must not alias the next warm start'


def test_all_nonfinite_candidates_raise_and_do_not_install_warm_start(tmp_path,monkeypatch):
    net=write_network(tmp_path/'net.npz');planner=make_planner(net.path)
    monkeypatch.setattr(planner,'_evaluate_candidate_controls',lambda *a,**kw:
        dict(total_cost=np.nan,scaled_total=np.nan,metrics={},controls_flat=np.zeros(10)))
    monkeypatch.setattr('planning.planners.base_planner.minimize',lambda *a,**kw:
        NS(x=np.zeros(10),success=True,status=0,nit=1,nfev=1,message='done'))
    with pytest.raises(RuntimeError,match='no finite solution'):planner.plan(np.zeros(3),np.eye(3),[1.,0.])
    assert planner.prev_controls_flat is None


def test_corrupt_warm_start_is_replaced_with_existing_cold_initialization(tmp_path):
    net=write_network(tmp_path/'net.npz');planner=make_planner(net.path)
    planner.optimizer_warm_start=True;planner.prev_controls_flat=np.full(10,np.nan)
    np.testing.assert_array_equal(planner._initial_controls_flat(),np.zeros(10))
    assert planner.prev_controls_flat is None


def test_unsupported_coherent_drift_cannot_optimize_different_dynamics(tmp_path):
    net=write_network(tmp_path/'net.npz');planner=make_planner(net.path)
    planner.coherent_drift=True
    with pytest.raises(ValueError,match='matching symbolic'):
        planner._get_casadi_valgrad([.1,0.],None,use_observation_risk=True,use_ambiguity_term=True)
