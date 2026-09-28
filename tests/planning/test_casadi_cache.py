import json

import numpy as np
import pytest

from planning.core import casadi_efe
from planning.core.casadi_cache import FunctionCache, compile_function, function_key
from planning.planners.base_planner import UnicyclePlannerBase


def planner(**overrides):
    settings = dict(horizon=6, dt=.25, v_min=0., v_max=.22, w_min=-.8, w_max=.8,
                    control_weight=.02, process_noise_xy=.01, process_noise_theta=.02,
                    goal_sigma_uv=30., risk_weight_obs=1., ambiguity_weight=1.,
                    optimizer_maxiter=10, optimizer_gtol=1e-5, optimizer_warm_start=False, seed=5,
                    camera_params=dict(cam_pos=[-5., -5.5, 4.8], look_at=[1.5, 1.5, 0.],
                                       img_width=1280, img_height=720, fov_h_rad=1.2))
    settings.update(overrides)
    return UnicyclePlannerBase(**settings)


def test_global_validity_uses_swept_rectangle_in_narrow_lane():
    geometry = json.dumps({'prisms': [dict(xmin=-3, xmax=3, ymin=-.35, ymax=.35,
                                           zmin=0, zmax=1)]})
    p = planner(horizon=1, dt=1., use_nogo_cost=True, nogo_mode='keep_in',
                nogo_weight=40., nogo_safe_distance=.55, driveable_geometry_json=geometry)
    # Rectangle fits even though the centre is inside the soft 0.55 m band.
    valid = p._trajectory_plan_diagnostics(np.zeros(3), np.eye(3)*.01, [[.1, 0]], [1, 0])
    assert valid['rollout_valid']
    # Sideways rectangle is wider than this lane.
    invalid = p._trajectory_plan_diagnostics(np.array([0, 0, np.pi/2]), np.eye(3)*.01, [[0, 0]], [1, 0])
    assert not invalid['rollout_valid']


def test_global_validity_checks_rotation_between_clear_endpoints():
    geometry = json.dumps({'prisms': [dict(xmin=.42, xmax=.44, ymin=-.02, ymax=.02,
                                           zmin=0, zmax=1)]})
    p = planner(horizon=1, dt=1., collision_geometry_json=geometry)
    result = p._trajectory_plan_diagnostics(np.zeros(3), np.eye(3)*.01, [[0, np.pi]], [1, 0])
    assert not result['rollout_valid']


def function(p):
    return p._get_casadi_valgrad([2., 1.], None,
                               use_observation_risk=True, use_ambiguity_term=True)


def evaluate(fn):
    return fn(np.tile([.15, .03], 6), [-1., .5, .2], np.diag([.04, .05, .02]),
              [640., 380.], [2., 1.], 0.)


def test_new_planner_loads_without_rebuilding_and_preserves_values(monkeypatch, tmp_path):
    monkeypatch.setenv('UNAV_CASADI_CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('UNAV_CASADI_JIT', '0')
    first = function(planner())
    expected = evaluate(first)
    assert first.cache_info['status'] == 'miss'
    def forbidden(*args, **kwargs):
        raise AssertionError('cache hit rebuilt the function')
    monkeypatch.setattr(casadi_efe, 'make_efe_valgrad_fn', forbidden)
    second = function(planner())
    assert second.cache_info['status'] == 'hit'
    actual = evaluate(second)
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])


def test_noise_camera_and_execution_changes_invalidate_identity(monkeypatch):
    monkeypatch.setenv('UNAV_CASADI_JIT', '0')
    p = planner()
    def key():
        return function_key(p._autodiff_cache_key([2., 1.], [640., 380.], True, True))
    old = key()
    p.process_noise_theta *= 2
    assert key() != old
    old = key()
    p.camera.H[0, 0] += 1
    assert key() != old
    old = key()
    monkeypatch.setenv('UNAV_CASADI_JIT', '1')
    assert key() != old


def test_damaged_cache_rebuilds_instead_of_using_wrong_math(monkeypatch, tmp_path):
    monkeypatch.setenv('UNAV_CASADI_CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('UNAV_CASADI_JIT', '0')
    original = function(planner())
    path = next(tmp_path.glob('*.json'))
    record = json.loads(path.read_text())
    record['sha256'] = 'damaged'
    path.write_text(json.dumps(record))
    with pytest.warns(RuntimeWarning, match='Ignoring unusable'):
        rebuilt = function(planner())
    assert rebuilt.cache_info['status'] == 'miss'
    np.testing.assert_array_equal(evaluate(rebuilt)[1], evaluate(original)[1])


def test_compiled_function_and_embedded_reload_preserve_value_and_gradient(tmp_path):
    native = function(planner())
    compiled = compile_function(native.casadi_function)
    cache = FunctionCache(tmp_path)
    cache.save('compiled_fixture', compiled)
    loaded = cache.load('compiled_fixture')
    assert loaded is not None
    for raw in (compiled, loaded):
        actual = evaluate(casadi_efe._make_valgrad_wrapper(raw))
        expected = evaluate(native)
        np.testing.assert_allclose(actual[0], expected[0], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(actual[1], expected[1], rtol=1e-9, atol=1e-9)
