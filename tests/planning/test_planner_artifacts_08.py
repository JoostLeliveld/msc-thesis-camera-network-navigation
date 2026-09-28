"""Frozen-content, layout, domain and selected-objective regressions."""
import hashlib
import json
from pathlib import Path

import casadi as ca
import numpy as np
import pytest

from planning.core.camera_network import CameraNetworkModel,projection_jacobian
from planning.core.visibility_gp_map import GPVisibilityMapConfig,GPVisibilityMapModel
from planning.planners.base_planner import UnicyclePlannerBase
from test_camera_network import write_network


def payload(path):
    with np.load(path,allow_pickle=False) as value:return {k:value[k].copy() for k in value.files}


def legacy(path,grid=None,**changes):
    grid=np.full((3,3),.8) if grid is None else grid
    data=dict(xs=np.array([-2.,0.,2.]),ys=np.array([-2.,0.,2.]),
        P_mean_map=grid,P_conservative_plan_map=grid,camera_pos=np.array([-5.,-5.,5.]),target_height=0.)
    data.update(changes);np.savez(path,**data)
    return GPVisibilityMapModel(GPVisibilityMapConfig(artifact_path=str(path)))


def planner(path=None,**changes):
    settings=dict(horizon=4,dt=.25,v_min=0.,v_max=.22,w_min=-1.,w_max=1.,
        control_weight=.02,process_noise_xy=.01,process_noise_theta=.02,
        goal_sigma_uv=30.,risk_weight_obs=1.,ambiguity_weight=1.,optimizer_maxiter=2,
        optimizer_gtol=1e-5,optimizer_warm_start=False,seed=210,
        camera_params=dict(cam_pos=(-5.,-5.,5.),look_at=(0.,0.,0.),img_width=1280,img_height=720,fov_h_rad=1.2))
    if path:settings.update(use_visibility_model=True,visibility_artifact_path=str(path))
    settings.update(changes)
    return UnicyclePlannerBase(**settings)


def test_network_bytes_identity_and_immutable_loaded_snapshot(tmp_path):
    path=tmp_path/'net.npz';net=write_network(path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    sources=dict(net.metadata['source_hashes'])
    checked=CameraNetworkModel(path,expected_sha256=digest,expected_source_hashes=sources,
                               expected_camera_ids=['camera_B','camera_A'])
    before=checked.query(np.zeros(3))['score'].copy()
    for value in (net.R,net.R_miss,net.precision,net.miss_precision,net.xs,net.ys,net.fields['score']):
        with pytest.raises(ValueError):value.setflags(write=True)
    with pytest.raises(TypeError):net.fields['score']=np.zeros_like(net.fields['score'])
    with pytest.raises(AttributeError):net.R=np.eye(2)
    changed=payload(path);changed['score'][:]=.1;np.savez(path,**changed)
    np.testing.assert_array_equal(checked.query(np.zeros(3))['score'],before)
    assert CameraNetworkModel(path).signature != checked.signature
    with pytest.raises(ValueError,match='SHA-256'):CameraNetworkModel(path,expected_sha256=digest)


@pytest.mark.parametrize('kind',['missing_sources','bad_digest','source_mismatch','roster','extra_R','missing_R',
                                'empty_id','numeric_ids','two_dimensional_ids'])
def test_network_provenance_and_covariance_axis_fail_closed(tmp_path,kind):
    path=tmp_path/'net.npz';net=write_network(path);data=payload(path)
    metadata=json.loads(data['metadata_json'].item());kwargs={}
    if kind=='missing_sources':metadata.pop('source_hashes')
    elif kind=='bad_digest':metadata['source_hashes']={'x':'bad'}
    elif kind=='source_mismatch':kwargs['expected_source_hashes']={'different':'a'*64}
    elif kind=='roster':kwargs['expected_camera_ids']=['camera_A']
    elif kind=='extra_R':data['R_cond_m2']=np.concatenate([data['R_cond_m2'],data['R_cond_m2'][:1]])
    elif kind=='missing_R':data['R_cond_m2']=data['R_cond_m2'][:1]
    elif kind=='empty_id':data['camera_ids']=np.array(['','camera_B'])
    elif kind=='numeric_ids':data['camera_ids']=np.array([1,2])
    elif kind=='two_dimensional_ids':data['camera_ids']=data['camera_ids'][:,None]
    data['metadata_json']=json.dumps(metadata);np.savez(path,**data)
    with pytest.raises(ValueError):CameraNetworkModel(path,**kwargs)


def test_hash_and_parser_share_bytes_even_if_path_is_replaced(tmp_path,monkeypatch):
    path=tmp_path/'net.npz';old=write_network(path);new=payload(path);new['score'][:]=.1
    original=Path.read_bytes
    def swapped(p):
        content=original(p)
        if p==path.resolve():np.savez(p,**new)
        return content
    monkeypatch.setattr(Path,'read_bytes',swapped)
    model=CameraNetworkModel(path)
    assert model.sha256==old.sha256
    np.testing.assert_array_equal(model.query(np.zeros(3))['score'],old.query(np.zeros(3))['score'])


def test_zero_variance_misses_preserve_prior_and_information_refuses_singular_prior(tmp_path):
    model=write_network(tmp_path/'net.npz',availability=0.)
    P=np.zeros((3,3))
    np.testing.assert_array_equal(model.forecast_posterior(np.zeros(3),P),P)
    with pytest.raises(ValueError):model.forecast_posterior(np.zeros(3),P,'information')
    with pytest.raises(ValueError):model.query_belief(np.zeros(3),np.diag([1.,1.,-.1]))


def test_fixed_chart_has_one_singularity_boundary(tmp_path):
    net=write_network(tmp_path/'net.npz');H=np.array([[0.,0.,1.],[0.,1.,0.],[1.,0.,0.]])
    m=ca.MX.sym('m',3);P=ca.MX.sym('P',3,3)
    fn=ca.Function('domain08',[m,P],[net.make_proxy_covariance_casadi(H)(m,P)])
    for x in (0.,1e-9):
        with pytest.raises(ValueError):projection_jacobian(H,[x,0.,0.])
        assert not np.isfinite(np.asarray(fn([x,0.,0.],np.eye(3)))).all()
    assert np.isfinite(np.asarray(fn([1.,0.,0.],np.eye(3)))).all()


def test_legacy_boundary_policy_matches_between_optimizer_and_selection(tmp_path):
    path=tmp_path/'gp.npz';model=legacy(path);p=planner(path)
    m=ca.MX.sym('state',3);fn=ca.Function('legacy08',[m],[model.make_prob_state_casadi()(m)])
    for x in (2.,2.+1e-10,3.,-3.):
        assert p.visibility_probability([x,0.,0.]) == pytest.approx(float(fn([x,0.,0.])))
    assert float(fn([2.+1e-10,0.,0.]))==pytest.approx(model.min_prob)
    with pytest.raises(ValueError):model.P_conservative_plan_map.setflags(write=True)


def test_same_path_size_mean_can_no_longer_collide_in_legacy_identity(tmp_path):
    path=tmp_path/'gp.npz';a=np.array([[.2,.3,.4],[.4,.5,.6],[.6,.7,.8]])
    first=legacy(path,a);before=first.prob_state_np([-2.,-2.,0.])
    second=legacy(path,a[::-1].copy())
    assert first.signature!=second.signature
    assert first.prob_state_np([-2.,-2.,0.])==before


@pytest.mark.parametrize('changes',[{'xs':np.array([0.,0.,1.])},{'ys':np.array([0.,np.nan,2.])},
    {'P_conservative_plan_map':np.full((3,3),np.nan)},{'P_mean_map':np.full((3,3),1.1)}])
def test_invalid_legacy_arrays_are_not_clipped_into_a_model(tmp_path,changes):
    with pytest.raises((ValueError,RuntimeError)):legacy(tmp_path/'gp.npz',**changes)


@pytest.mark.parametrize('method',['ET1','ET2'])
@pytest.mark.parametrize('x',[0.,3.])
@pytest.mark.parametrize('terms',[(True,True),(False,True),(True,False),(False,False)])
def test_optional_mixture_uses_same_objective_for_selection(tmp_path,method,x,terms):
    path=tmp_path/'gp.npz';legacy(path)
    p=planner(path,use_hit_miss_mixture=True,approx_method=method,use_obs_risk=terms[0],use_ambiguity=terms[1])
    m=np.array([x,.2,.1]);P=np.diag([.2,.3,.05]);goal=np.array([1.,1.,0.]);obs=p._goal_obs(goal)
    u=np.tile([.1,.03],p.horizon)
    fn=p._get_casadi_valgrad(goal,obs,use_observation_risk=terms[0],use_ambiguity_term=terms[1])
    value,gradient=fn(u,m,P,obs,goal[:2],0.)
    total,parts=p._evaluate_controls(u,m,P,goal,obs,None,True)
    heff=sum(p.discount_gamma**i for i in range(p.horizon))
    np.testing.assert_allclose(total/heff,value,rtol=1e-7,atol=1e-7)
    assert total==pytest.approx(sum(parts[k] for k in ('risk_cost','ambiguity_cost','control_cost','obstacle_cost')))
    candidate=p._evaluate_candidate_controls(u,m,P,goal,obs,None,(1.,1.))
    assert candidate['total_cost']==pytest.approx(total)
    public=p.evaluate_rollout_controls(m,P,goal[:2],u.reshape(-1,2))
    assert public['total_cost']==pytest.approx(total)


def test_geometry_cache_identity_preserves_small_changes_in_embedded_constants():
    from planning.core.nogo_cost import NogoCostConfig,NogoZoneCostModel
    scene=json.dumps({'prisms':[dict(xmin=-1.,xmax=1.,ymin=-1.,ymax=1.,zmin=0.,zmax=2.)]})
    model=NogoZoneCostModel(NogoCostConfig(geometry_json=scene,weight=1.,safe_distance=.35))
    original=UnicyclePlannerBase._geometry_cache_identity(model)
    model.safe_distance+=1e-10
    assert UnicyclePlannerBase._geometry_cache_identity(model)!=original
    original=UnicyclePlannerBase._geometry_cache_identity(model)
    model._xmins=model._xmins+1e-10
    assert UnicyclePlannerBase._geometry_cache_identity(model)!=original
