"""Network algebra, planner wiring and gradients; synthetic fixtures only."""
import json
import hashlib
import numpy as np
import pytest
from planning.core.camera_network import (
    AMBIGUITY_INFORMATION_REGULARIZER_M2_INV,
    CameraNetworkModel,
    projection_jacobian,
)
from planning.planners.base_planner import UnicyclePlannerBase


def write_network(path, score=.5, availability=.4, spatial=False):
    xs=ys=np.array([-3.,0.,3.]); X,Y=np.meshgrid(xs,ys)
    rho=np.stack([np.full((3,3),score),np.full((3,3),score)])
    if spatial: rho=np.stack([.5+.05*X+.02*Y,.55-.03*X+.04*Y])
    R=np.array([[[.01,.006],[.006,.16]],[[.12,-.008],[-.008,.015]]])
    meta=dict(schema='camera_network.iwai.v1',reference='robot_ground_reference_xy',
        frame='map_bev',covariance_units='m2',score_target='detector_score_with_miss_zero',
        availability_target='valid_detection_finite_ground_projection',evidence='synthetic_test_fixture',
        source_hashes={'synthetic_fixture': hashlib.sha256(b'camera-network-test-v1').hexdigest()})
    np.savez(path,xs=xs,ys=ys,camera_ids=['camera_A','camera_B'],score=rho,
        availability=np.full((2,3,3),availability),R_cond_m2=R,R_miss_proxy_m2=R+25*np.eye(2),
        metadata_json=json.dumps(meta))
    return CameraNetworkModel(path)


def write_information_network(path, information=None):
    xs = ys = np.array([-1.0, 0.0, 1.0])
    if information is None:
        information = np.asarray([
            [[4.0, 0.5], [0.5, 2.0]],
            [[1.0, -0.1], [-0.1, 3.0]],
        ])
    field = np.broadcast_to(
        np.asarray(information)[:, None, None], (2, 3, 3, 2, 2)).copy()
    support = np.broadcast_to(np.asarray([12.0, 8.0])[:, None, None], (2, 3, 3)).copy()
    meta = dict(
        schema='camera_network.thesis_stage09.v3',
        reference='robot_ground_reference_xy', frame='map_bev',
        covariance_units='m2', information_units='m-2',
        planning_target='admitted_runtime_precision_else_zero',
        source_hashes={
            'synthetic_fixture': hashlib.sha256(b'direct-information-test-v1').hexdigest()
        },
    )
    np.savez(
        path, xs=xs, ys=ys, camera_ids=['camera_A', 'camera_B'],
        expected_information_m2_inv=field, opportunity_support=support,
        metadata_json=json.dumps(meta),
    )
    return CameraNetworkModel(path)


def test_direct_information_update_and_camera_removal(tmp_path):
    net = write_information_network(tmp_path / 'information.npz')
    state = np.zeros(3)
    prior = np.diag([0.2, 0.3, 0.05])
    total = net.query(state)['expected_information'].sum(axis=0)
    expected_precision = np.linalg.inv(prior)
    expected_precision[:2, :2] += total
    expected = np.linalg.inv(expected_precision)
    actual, _ = net.expected_belief(state, prior)
    np.testing.assert_allclose(actual, expected)

    without_b = CameraNetworkModel(net.path, cameras=['camera_A'])
    removed, _ = without_b.expected_belief(state, prior)
    removed_precision = np.linalg.inv(prior)
    removed_precision[:2, :2] += net.query(state)['expected_information'][0]
    np.testing.assert_allclose(removed, np.linalg.inv(removed_precision))
    assert np.trace(removed[:2, :2]) > np.trace(actual[:2, :2])


def test_final_bayesian_direct_information_schema(tmp_path):
    path = tmp_path / 'final_bayesian_information.npz'
    xs = ys = np.asarray([0.0, 1.0])
    information = np.broadcast_to(
        np.eye(2), (1, len(ys), len(xs), 2, 2)).copy()
    metadata = {
        'schema': 'camera_network.final_bayesian_planning.v1',
        'reference': 'robot_ground_reference_xy',
        'frame': 'map_bev',
        'fit_role': 'D_R',
        'D_eval_accessed': False,
        'planning_target': 'admitted_runtime_precision_else_zero',
        'source_hashes': {'fixture': hashlib.sha256(b'final-bayesian').hexdigest()},
    }
    np.savez(
        path, xs=xs, ys=ys, camera_ids=['camera_A'],
        expected_information_m2_inv=information,
        opportunity_support=np.ones((1, len(ys), len(xs))),
        metadata_json=json.dumps(metadata),
    )
    network = CameraNetworkModel(path)
    assert network.direct_information
    np.testing.assert_allclose(
        network.query(np.asarray([0.5, 0.5, 0.0]))['expected_information'][0],
        np.eye(2))

    metadata['D_eval_accessed'] = True
    np.savez(
        path, xs=xs, ys=ys, camera_ids=['camera_A'],
        expected_information_m2_inv=information,
        opportunity_support=np.ones((1, len(ys), len(xs))),
        metadata_json=json.dumps(metadata),
    )
    with pytest.raises(ValueError, match='must not access D_eval'):
        CameraNetworkModel(path)


def test_matched_covariance_precision_schema_uses_no_opportunity_target(tmp_path):
    path = tmp_path / 'matched_precision.npz'
    xs = ys = np.asarray([0.0, 1.0])
    precision = np.broadcast_to(
        np.diag([4.0, 2.0]), (1, len(ys), len(xs), 2, 2)).copy()
    metadata = {
        'schema': 'camera_network.matched_covariance_precision.v1',
        'reference': 'robot_ground_reference_xy',
        'frame': 'map_bev',
        'planning_target': 'inverse_of_matched_runtime_covariance',
        'additional_planning_fit': False,
        'detector_opportunities_used': False,
        'gate_outcomes_used': False,
        'source_hashes': {'fixture': hashlib.sha256(b'matched-precision').hexdigest()},
    }
    np.savez(
        path, xs=xs, ys=ys, camera_ids=['camera_A'],
        matched_precision_m2_inv=precision,
        residual_support=np.ones((1, len(ys), len(xs))),
        metadata_json=json.dumps(metadata),
    )
    network = CameraNetworkModel(path)
    np.testing.assert_allclose(
        network.query(np.asarray([0.5, 0.5, 0.0]))['expected_information'][0],
        np.diag([4.0, 2.0]))
    assert 'residual_support' in network.fields
    assert 'opportunity_support' not in network.fields


def test_direct_information_bilinear_interpolation_is_psd_and_heading_invariant(tmp_path):
    path = tmp_path / 'interpolated_information.npz'
    xs = ys = np.asarray([0.0, 1.0])
    field = np.zeros((1, 2, 2, 2, 2))
    field[0, 0, 0] = [[4.0, 1.5], [1.5, 1.0]]
    field[0, 0, 1] = [[1.0, -0.8], [-0.8, 1.0]]
    field[0, 1, 0] = [[2.0, 0.0], [0.0, 0.2]]
    field[0, 1, 1] = [[0.3, 0.1], [0.1, 3.0]]
    meta = dict(
        schema='camera_network.thesis_stage09.v3',
        reference='robot_ground_reference_xy', frame='map_bev', covariance_units='m2',
        planning_target='admitted_runtime_precision_else_zero',
        source_hashes={'fixture': hashlib.sha256(b'interpolation').hexdigest()},
    )
    np.savez(path, xs=xs, ys=ys, camera_ids=['camera_A'],
             expected_information_m2_inv=field,
             opportunity_support=np.ones((1, 2, 2)), metadata_json=json.dumps(meta))
    net = CameraNetworkModel(path)
    first = net.query(np.asarray([0.5, 0.5, -2.0]))['expected_information'][0]
    second = net.query(np.asarray([0.5, 0.5, 2.0]))['expected_information'][0]
    np.testing.assert_allclose(first, field.mean(axis=(1, 2))[0])
    np.testing.assert_allclose(second, first)
    assert np.linalg.eigvalsh(first).min() >= -1e-12


def test_direct_information_rejects_branch_semantics(tmp_path):
    net = write_information_network(tmp_path / 'information.npz')
    prior = np.diag([0.2, 0.3, 0.05])
    with pytest.raises(ValueError, match='information approximation'):
        net.forecast_posterior(np.zeros(3), prior)
    posterior = net.forecast_posterior(np.zeros(3), prior, mode='information')
    assert np.trace(posterior[:2, :2]) < np.trace(prior[:2, :2])


def test_direct_information_numpy_and_casadi_match(tmp_path):
    ca = pytest.importorskip('casadi')
    net = write_information_network(tmp_path / 'information.npz')
    state = np.asarray([0.2, -0.1, 0.0])
    prior = np.asarray([[0.2, 0.01, 0.02], [0.01, 0.3, -0.01], [0.02, -0.01, 0.05]])
    m = ca.MX.sym('m_direct', 3)
    P = ca.MX.sym('P_direct', 3, 3)
    posterior_expr, ambiguity_expr = net.make_expected_belief_casadi(
        opportunities=2)(m, P)
    function = ca.Function(
        'direct_information_belief', [m, P], [posterior_expr, ambiguity_expr])
    actual_posterior, actual_ambiguity = function(state, prior)
    expected_posterior, expected_ambiguity = net.expected_belief(
        state, prior, opportunities=2)
    np.testing.assert_allclose(
        np.asarray(actual_posterior), expected_posterior, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(
        float(actual_ambiguity), expected_ambiguity, rtol=1e-9, atol=1e-10)


def test_direct_effective_covariance_uses_fixed_information_regularizer(tmp_path):
    zero = np.zeros((2, 2, 2))
    blind = write_information_network(tmp_path / 'blind.npz', information=zero)
    state = np.zeros(3)
    prior = np.diag([0.2, 0.3, 0.05])
    expected = np.eye(2) / AMBIGUITY_INFORMATION_REGULARIZER_M2_INV
    np.testing.assert_allclose(
        blind.effective_observation_covariance(state, prior), expected)

    informed = write_information_network(tmp_path / 'informed.npz')
    informed_R = informed.effective_observation_covariance(state, prior)
    # Adding positive-semidefinite camera information can only reduce the
    # equivalent ambiguity covariance in the Loewner order.
    assert np.linalg.eigvalsh(expected - informed_R).min() >= -1e-12


def test_direct_effective_covariance_numpy_and_casadi_match(tmp_path):
    ca = pytest.importorskip('casadi')
    net = write_information_network(tmp_path / 'information.npz')
    state = np.asarray([0.2, -0.1, 0.0])
    prior = np.diag([0.2, 0.3, 0.05])
    m = ca.MX.sym('m_effective', 3)
    P = ca.MX.sym('P_effective', 3, 3)
    function = ca.Function(
        'direct_effective_covariance', [m, P],
        [net.make_effective_covariance_casadi()(m, P)])
    actual = np.asarray(function(state, prior))
    expected = net.effective_observation_covariance(state, prior)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)


def test_real_planner_accepts_direct_information_artifact(tmp_path):
    pytest.importorskip('casadi')
    net = write_information_network(tmp_path / 'information.npz')
    planner = make_planner(
        net.path,
        camera_network_objective='metric_expected_belief',
        camera_network_camera_ids='camera_A,camera_B',
        camera_network_active_camera_ids='camera_A',
        network_goal_std_m=0.2,
    )
    assert planner.camera_network.camera_ids == ('camera_A',)
    state = np.asarray([0.0, 0.0, 0.0])
    prior = np.diag([0.05, 0.04, 0.03])
    goal = np.asarray([0.6, 0.2, 0.0])
    controls = np.tile([0.15, 0.02], planner.horizon)
    goal_obs = planner._goal_obs(goal)
    evaluate = planner._get_casadi_valgrad(
        goal, goal_obs, use_observation_risk=True, use_ambiguity_term=True)
    value, gradient = evaluate(controls, state, prior, goal_obs, goal[:2], 0.0)
    assert np.isfinite(value)
    assert np.isfinite(gradient).all()
    diagnostic = planner.planning_visibility_diagnostics(state, prior)
    assert np.isnan(diagnostic['p_vis'])
    assert diagnostic['expected_information_trace'] > 0.0


def test_complementarity_preserves_full_directional_information(tmp_path):
    net=write_network(tmp_path/'field.npz',score=1.)
    covariance=net.proxy_ground_covariance(np.ones(2))
    expected=np.linalg.solve(np.linalg.solve(net.R[0],np.eye(2))+
        np.linalg.solve(net.R[1],np.eye(2)),np.eye(2))
    np.testing.assert_allclose(covariance,expected)
    assert abs(covariance[0,1])>1e-6
    for R in net.R: assert np.linalg.eigvalsh(R-covariance).min()>0
    reverse=CameraNetworkModel(net.path,cameras=['camera_B','camera_A'])
    np.testing.assert_allclose(reverse.proxy_ground_covariance(np.ones(2)),covariance)


def test_single_camera_is_the_matrix_iwai_precision_blend(tmp_path):
    net=write_network(tmp_path/'field.npz')
    one=CameraNetworkModel(net.path,cameras=['camera_A'])
    expected=np.linalg.solve(.3*one.precision[0]+.7*one.miss_precision[0],np.eye(2))
    np.testing.assert_allclose(one.proxy_ground_covariance([.3]),expected)


def test_misses_do_not_create_measurements_in_the_reference(tmp_path):
    net=write_network(tmp_path/'field.npz',score=.9,availability=0.)
    P=np.diag([.2,.3,.05])
    np.testing.assert_allclose(net.forecast_posterior([0.,0.,0.],P),P)
    np.testing.assert_allclose(net.forecast_posterior([0.,0.,0.],P,'information'),P)
    # A finite IWAI cost proxy remains distinct from the no-observation branch.
    assert np.isfinite(net.proxy_ground_covariance([0.,0.])).all()


def test_branch_reference_exposes_expected_information_optimism(tmp_path):
    net=write_network(tmp_path/'field.npz')
    P=np.array([[.2,.01,.006],[.01,.3,-.002],[.006,-.002,.05]])
    branch=net.forecast_posterior([0.,0.,0.],P)
    information=net.forecast_posterior([0.,0.,0.],P,'information')
    assert np.linalg.eigvalsh(branch).min()>0
    assert np.linalg.eigvalsh(branch-information).min()>-1e-12
    assert np.trace(branch[:2,:2])>np.trace(information[:2,:2])


def test_all_hits_matches_the_joint_linear_update_including_heading_cross_covariance(tmp_path):
    net=write_network(tmp_path/'field.npz',availability=1.)
    P=np.array([[.2,.01,.03],[.01,.3,-.01],[.03,-.01,.05]])
    joint_precision=np.linalg.solve(P,np.eye(3))
    joint_precision[:2,:2]+=net.precision.sum(axis=0)
    expected=np.linalg.solve(joint_precision,np.eye(3))
    np.testing.assert_allclose(net.forecast_posterior([0.,0.,.4],P),expected,atol=1e-12)
    assert expected[2,2]<P[2,2]


def test_numpy_and_casadi_match_inside_and_outside_support(tmp_path):
    ca=pytest.importorskip('casadi')
    net=write_network(tmp_path/'field.npz',spatial=True)
    H=np.array([[120.,5.,640.],[2.,140.,400.],[.03,.02,1.]])
    m=ca.MX.sym('m',3);P=ca.MX.sym('P',3,3)
    func=ca.Function('network_cov',[m,P],[net.make_proxy_covariance_casadi(H)(m,P)])
    cov=np.diag([.05,.04,.03])
    for state in ([.35,.42,.1],[4.,5.,.1],[2.9,.1,.1]):
        actual=np.asarray(func(state,cov))
        expected=net.planning_diagnostics(state,cov,H)['R_plan']
        np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-8)
    np.testing.assert_array_equal(net.query([4.,5.,0.])['availability'],[0.,0.])
    with pytest.raises(ValueError):net.query([0.,0.,0.,.9])


@pytest.mark.parametrize('opportunities',[1,5])
def test_metric_expected_belief_numpy_and_casadi_match(tmp_path,opportunities):
    ca=pytest.importorskip('casadi')
    net=write_network(tmp_path/'field.npz',availability=.4,spatial=True)
    state=np.array([.35,.42,.1])
    P=np.array([[.2,.01,.03],[.01,.3,-.01],[.03,-.01,.05]])
    m=ca.MX.sym('m',3);S=ca.MX.sym('S',3,3)
    P_expr,H_expr=net.make_expected_belief_casadi(opportunities=opportunities)(m,S)
    func=ca.Function('expected_belief',[m,S],[P_expr,H_expr])
    actual_P,actual_H=func(state,P)
    expected_P,expected_H=net.expected_belief(state,P,opportunities=opportunities)
    np.testing.assert_allclose(np.asarray(actual_P),expected_P,rtol=1e-8,atol=1e-9)
    np.testing.assert_allclose(float(actual_H),expected_H,rtol=1e-8,atol=1e-9)
    assert np.trace(expected_P[:2,:2]) < np.trace(P[:2,:2])


def test_repeated_camera_opportunities_contract_covariance_and_validate_count(tmp_path):
    net=write_network(tmp_path/'field.npz',availability=.4,spatial=True)
    state=np.array([.35,.42,.1])
    P=np.array([[.2,.01,.03],[.01,.3,-.01],[.03,-.01,.05]])
    one,_=net.expected_belief(state,P,opportunities=1)
    five,_=net.expected_belief(state,P,opportunities=5)
    assert np.trace(five[:2,:2]) < np.trace(one[:2,:2])
    for invalid in (0,-1,True,1.5):
        with pytest.raises(ValueError,match='positive integer'):
            net.expected_belief(state,P,opportunities=invalid)


def make_planner(path, **overrides):
    from planning.planners.base_planner import UnicyclePlannerBase
    settings=dict(horizon=5,dt=.25,v_min=0.,v_max=.5,w_min=-1.,w_max=1.,
        control_weight=.02,process_noise_xy=.01,process_noise_theta=.02,
        goal_sigma_uv=30.,risk_weight_obs=1.,ambiguity_weight=1.,optimizer_maxiter=15,
        optimizer_gtol=1e-5,optimizer_warm_start=False,seed=513,use_visibility_model=True,
        camera_network_artifact_path=str(path),camera_params=dict(cam_pos=(-5.,-5.,5.),
            look_at=(0.,0.,0.),img_width=1280,img_height=720,fov_h_rad=1.2))
    settings.update(overrides)
    return UnicyclePlannerBase(**settings)


def test_real_planner_uses_network_cost_and_correct_gradient(tmp_path):
    pytest.importorskip('casadi')
    net=write_network(tmp_path/'field.npz',spatial=True)
    planner=make_planner(
        net.path, camera_network_objective='metric_expected_belief')
    state=np.array([.35,.42,.1]);P=np.diag([.05,.04,.03]);goal=np.array([1.2,1.,0.])
    goal_obs=planner._goal_obs(goal)
    evaluate=planner._get_casadi_valgrad(goal,goal_obs,use_observation_risk=True,use_ambiguity_term=True)
    u=np.tile([.22,.06],5)
    value,gradient=evaluate(u,state,P,goal_obs,goal[:2],0.)
    numerical=[]
    for i in range(len(u)):
        delta=np.zeros_like(u);delta[i]=1e-5
        numerical.append((evaluate(u+delta,state,P,goal_obs,goal[:2],0.)[0]-
            evaluate(u-delta,state,P,goal_obs,goal[:2],0.)[0])/2e-5)
    np.testing.assert_allclose(gradient,numerical,rtol=2e-4,atol=2e-5)
    # Both backends expose the locked active-discount-normalized objective.
    numpy_value=planner._evaluate_controls(u,state,P,goal,goal_obs,None)
    np.testing.assert_allclose(value,numpy_value,rtol=1e-7,atol=1e-6)
    with pytest.raises(RuntimeError,match='not a fresh measurement'):
        planner.observation_model_with_visibility(state,P)
    solved=planner.plan(state,P,goal[:2])
    assert np.isfinite(solved.total_cost)
    assert np.isfinite(solved.controls).all()


def test_metric_network_objective_is_independent_of_fixed_camera_chart(tmp_path):
    pytest.importorskip('casadi')
    net=write_network(tmp_path/'field.npz',availability=.4,spatial=True)
    metric=dict(camera_network_objective='metric_expected_belief',network_goal_std_m=.2)
    first=make_planner(net.path,**metric)
    second=make_planner(net.path,**dict(metric,camera_params=dict(
        cam_pos=(7.,-2.,9.),look_at=(1.,1.,0.),img_width=800,img_height=600,fov_h_rad=.8)))
    state=np.array([.35,.42,.1]);P=np.diag([.05,.04,.03]);goal=np.array([1.2,1.,0.])
    u=np.tile([.22,.06],5)
    values=[]
    for planner in (first,second):
        goal_obs=planner._goal_obs(goal)
        evaluate=planner._get_casadi_valgrad(
            goal,goal_obs,use_observation_risk=True,use_ambiguity_term=True)
        value,gradient=evaluate(u,state,P,goal_obs,goal[:2],0.)
        assert np.isfinite(value) and np.isfinite(gradient).all()
        numerical=[]
        for i in range(len(u)):
            delta=np.zeros_like(u);delta[i]=1e-5
            numerical.append((evaluate(u+delta,state,P,goal_obs,goal[:2],0.)[0]-
                evaluate(u-delta,state,P,goal_obs,goal[:2],0.)[0])/2e-5)
        np.testing.assert_allclose(gradient,numerical,rtol=3e-4,atol=3e-5)
        numpy_value=planner._evaluate_controls(u,state,P,goal,goal_obs,None)
        np.testing.assert_allclose(value,numpy_value,rtol=1e-7,atol=1e-6)
        values.append(value)
    np.testing.assert_allclose(values[0],values[1],rtol=1e-12,atol=1e-12)


def test_metric_network_gradient_is_finite_when_route_reaches_exact_goal(tmp_path):
    pytest.importorskip('casadi')
    net=write_network(tmp_path/'field.npz',availability=.4,spatial=True)
    planner=make_planner(
        net.path, camera_network_objective='metric_expected_belief',
        network_goal_std_m=.2, optimizer_terminal_goal_tolerance_m=.1,
    )
    state=np.array([0.,0.,0.]);P=np.diag([.05,.04,.03]);goal=np.array([0.,0.,0.])
    goal_obs=planner._goal_obs(goal)
    evaluate=planner._get_casadi_valgrad(
        goal,goal_obs,use_observation_risk=True,use_ambiguity_term=True)
    value,gradient=evaluate(np.zeros(10),state,P,goal_obs,goal[:2],0.)
    assert np.isfinite(value)
    assert np.isfinite(gradient).all()


def test_locked_active_normalization_removes_constant_field_duration_cost(tmp_path):
    net = write_network(tmp_path/'constant.npz', availability=.4, spatial=False)
    planner = make_planner(
        net.path, camera_network_objective='metric_expected_belief',
        use_obs_risk=False, control_weight=0.0,
        optimizer_terminal_goal_tolerance_m=.1,
    )
    state = np.array([0., 0., 0.])
    covariance = np.diag([.05, .04, .03])
    goal = np.array([.5, 0.])
    early_arrival = np.tile([.5, 0.], (planner.horizon, 1))
    slow_route = np.tile([.2, 0.], (planner.horizon, 1))
    early = planner.evaluate_rollout_controls(
        state, covariance, goal, early_arrival)['ambiguity_cost']
    slow = planner.evaluate_rollout_controls(
        state, covariance, goal, slow_route)['ambiguity_cost']
    np.testing.assert_allclose(early, slow, rtol=1e-12, atol=1e-12)


def test_sparse_control_blocks_keep_full_rollout_and_exact_gradient(tmp_path):
    pytest.importorskip('casadi')
    net=write_network(tmp_path/'field.npz',availability=.4,spatial=True)
    planner=make_planner(
        net.path, camera_network_objective='metric_expected_belief',
        network_goal_std_m=.2, optimizer_control_block_steps=2,
    )
    state=np.array([.35,.42,.1]);P=np.diag([.05,.04,.03]);goal=np.array([1.2,1.,0.])
    goal_obs=planner._goal_obs(goal)
    evaluate=planner._get_casadi_valgrad(
        goal,goal_obs,use_observation_risk=True,use_ambiguity_term=True)
    # ceil(5 / 2) blocks, two control variables each.
    sparse=np.array([.20,.05,.24,-.04,.10,.0])
    value,gradient=evaluate(sparse,state,P,goal_obs,goal[:2],0.)
    numerical=[]
    for i in range(len(sparse)):
        delta=np.zeros_like(sparse);delta[i]=1e-5
        numerical.append((evaluate(sparse+delta,state,P,goal_obs,goal[:2],0.)[0]-
            evaluate(sparse-delta,state,P,goal_obs,goal[:2],0.)[0])/2e-5)
    np.testing.assert_allclose(gradient,numerical,rtol=3e-4,atol=3e-5)
    expanded=np.repeat(sparse.reshape(3,2),2,axis=0)[:5].reshape(-1)
    numpy_value=planner._evaluate_controls(expanded,state,P,goal,goal_obs,None)
    np.testing.assert_allclose(value,numpy_value,rtol=1e-7,atol=1e-6)
    solved=planner.plan(state,P,goal[:2])
    np.testing.assert_allclose(solved.controls[0],solved.controls[1])
    np.testing.assert_allclose(solved.controls[2],solved.controls[3])


def test_invalid_covariance_and_unknown_masks_fail(tmp_path):
    net=write_network(tmp_path/'field.npz')
    with pytest.raises(ValueError,match='unknown camera'):
        CameraNetworkModel(net.path,cameras=['camera_Z'])
    with np.load(net.path) as data: payload={k:data[k] for k in data.files}
    payload['R_cond_m2'][0,0,0]=-1
    np.savez(tmp_path/'bad.npz',**payload)
    with pytest.raises(ValueError,match='positive definite'):
        CameraNetworkModel(tmp_path/'bad.npz')


def test_route_retracing_gate_rejects_reverse_lane_traversal():
    clean = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.],
                      [2., 1., 0.], [3., 1., 0.]])
    retracing = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.],
                          [1., 0., 0.], [1., 1., 0.]])
    assert not UnicyclePlannerBase._trajectory_retraces_lane(clean)
    assert UnicyclePlannerBase._trajectory_retraces_lane(retracing)
