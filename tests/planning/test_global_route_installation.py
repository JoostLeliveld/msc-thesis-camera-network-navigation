"""Route admission regressions: real callback, fake solver/clock, no ROS graph."""
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest
from geometry_msgs.msg import PoseStamped
from test_runtime_transactions import command_node, Publisher
from test_planner_node_correction_wiring import _Logger, stamp


def route_node():
    n = command_node()
    n._command_stop_generation = 0
    n._active_plan_request = None
    n._active_controls = None
    n._active_plan_started_at = None
    n._fatal_stop_triggered = False
    n.use_hierarchical = True
    n._hier_phase = 'GLOBAL'
    n.global_planner_mode = 'efe'
    n._global_solve_done = False
    n._global_goal_xy = None
    n.goal_replan_move_m = 1.
    n._waypoints = []
    n._wp_idx = 0
    n.waypoint_spacing_m = .2
    n.optimizer_route_seed_mode = 'explicit'
    n.driveable_geometry_json = ''
    n.goal_msg = PoseStamped(); n.goal_msg.pose.position.x = 2.
    n.goal_msg.header.frame_id = 'map_bev'
    n._goal_received_logged = True
    n._update_goal_progress_origin = lambda _: None
    n._snapshot_plan_inputs = lambda: {'goal': n.goal_msg, 'pixel_stamp': None, 'state': None}
    n.belief_m = np.zeros(3); n.belief_S = np.eye(3)*.01; n.belief_stamp = stamp(9.9)
    n._belief_epoch = 0; n._belief_revision = 0
    n._resolve_belief_for_planning = lambda: (n.belief_m.copy(),n.belief_S.copy(),dict(
        belief_epoch=n._belief_epoch,belief_revision=n._belief_revision,
        belief_valid=True,motion_supported=True,belief_frame_id='map_bev'))
    n.get_logger = lambda: _Logger()
    n.published_routes = []
    n._publish_plan_and_metrics = lambda *a, **kw: n.published_routes.append(a[0])
    n._save_global_plan_artifacts = lambda *a, **kw: None
    n._save_preselected_route_artifacts = lambda *a, **kw: None
    n._build_path_message = lambda *a, **kw: a[0]
    n.path_pub = Publisher()
    n._pending_plan_started_at = None
    n.latency_compensate_plan_handoff = False
    n.v_min=0.; n.v_max=.22; n.w_min=-1.; n.w_max=1.
    n.planner = NS(_controls_for_waypoints=lambda *a: np.zeros(4),prev_controls_flat=None,
                   collision_cost_model=None,nogo_cost_model=None)
    n.global_planner = NS(plan=lambda *a, **kw: valid_result(),
        optimizer_terminal_goal_tolerance_m=.35,collision_cost_model=None,nogo_cost_model=None,
        dt=1.,collision_sweep_clearance_np=lambda *a: float('inf'),
        driveable_sweep_clearance_np=lambda *a: float('inf'))
    return n


def valid_result():
    return NS(states=np.array([[i*.2,0.,0.] for i in range(11)]),
              controls=np.tile([.2,0.],(10,1)), rollout_valid=True,
              min_predicted_obstacle_distance_m=1., terminal_goal_distance_pred=0.,
              total_cost=1.)


@pytest.mark.parametrize('kind',['nan','invalid','partial','missing_validity'])
def test_bad_global_result_cannot_become_local_route(kind):
    n=route_node(); r=valid_result()
    if kind=='nan': r.states[1,0]=np.nan
    if kind=='invalid': r.rollout_valid=False
    if kind=='partial': r.states[-1,0]=.2
    if kind=='missing_validity': del r.rollout_valid
    n.global_planner.plan=lambda *a,**kw:r
    n._plan_once()
    assert n._hier_phase=='GLOBAL'
    assert not n._waypoints
    assert not n.published_routes
    assert n.planner.prev_controls_flat is None


@pytest.mark.parametrize('change',['stop','goal','clock','epoch'])
def test_global_return_cannot_cross_request_boundary(change):
    n=route_node(); entered=threading.Event(); release=threading.Event(); errors=[]
    def solve(*a,**kw):
        entered.set(); assert release.wait(3.)
        return valid_result()
    n.global_planner.plan=solve
    def run():
        try:n._plan_once()
        except BaseException as e:errors.append(e)
    worker=threading.Thread(target=run); worker.start(); assert entered.wait(3.)
    if change=='stop':n._publish_safe_stop_command()
    if change=='goal':
        goal=PoseStamped(); goal.header.frame_id='map_bev'; goal.pose.position.x=2.1
        n._goal_cb(goal)
    if change=='clock':n._clock.seconds=5.
    if change=='epoch':n._belief_epoch+=1
    release.set(); worker.join(3.)
    assert not worker.is_alive() and not errors
    assert n._hier_phase=='GLOBAL'
    assert not n._waypoints and not n.published_routes


def test_fresh_route_after_stop_installs():
    n=route_node();n._publish_safe_stop_command();n._plan_once()
    assert n._hier_phase=='LOCAL'
    assert n._waypoints[-1]==(2.,0.)
    assert n.published_routes
    assert all(m.linear.x==m.angular.z==0. for m in n.cmd_pub.messages)


def test_slow_global_result_revalidates_supported_current_state_without_local_dt_limit():
    n=route_node(); checked=[]
    def solve(*a,**kw):
        n._clock.seconds=100.
        n._belief_revision+=1
        n.belief_m=np.array([.4,.1,0.])
        return valid_result()
    n.global_planner.plan=solve
    n.global_planner.collision_sweep_clearance_np=lambda start,end: checked.append(start.copy()) or 1.
    n._plan_once()
    assert n._hier_phase=='LOCAL'
    np.testing.assert_array_equal(checked[0],n.belief_m)


def test_new_correction_unsafe_entry_connector_rejects_route():
    n=route_node()
    def solve(*a,**kw):
        n._belief_revision+=1;n.belief_m=np.array([.4,1.,0.])
        return valid_result()
    n.global_planner.plan=solve
    n.global_planner.collision_sweep_clearance_np=lambda start,end: -1. if start[1]>.5 else 1.
    n._plan_once()
    assert n._hier_phase=='GLOBAL' and not n.published_routes


def test_unsupported_fresh_belief_cannot_install_route():
    n=route_node(); resolve=n._resolve_belief_for_planning
    def solve(*a,**kw):
        def unsupported():
            m,S,meta=resolve();meta['motion_supported']=False
            return m,S,meta
        n._resolve_belief_for_planning=unsupported
        return valid_result()
    n.global_planner.plan=solve;n._plan_once()
    assert n._hier_phase=='GLOBAL' and not n._waypoints


@pytest.mark.parametrize('branch',['efe_exception','geometric_empty','preselected'])
def test_other_global_branches_cannot_bypass_admission(branch,monkeypatch):
    n=route_node()
    if branch=='efe_exception':
        n._global_solve_done=True
        n.global_planner._parse_initial_routes=lambda x:x
        def broken(*a,**kw): raise RuntimeError('failed solve')
        n.global_planner.plan=broken
    elif branch=='geometric_empty':
        n.global_planner_mode='geometric_shortest_path'
    else:
        n.global_planner_mode='preselected_route'
        n._preselected_route_points=[(0.,0.),(2.,0.)]
        n._preselected_route_provenance={'registered_goal_xy':[2.,0.]}
        n._preselected_route_sha256='test'
        n.preselected_route_sha256='test'
        n.global_planner.collision_sweep_clearance_np=lambda *a:-1.
    n._plan_once()
    assert n._hier_phase=='GLOBAL' and not n._waypoints and not n.published_routes


def test_valid_preselected_route_is_unchanged():
    n=route_node(); n.global_planner_mode='preselected_route'
    n._preselected_route_points=[(0.,0.),(.73,0.),(2.,0.)]
    n._preselected_route_provenance={'registered_goal_xy':[2.,0.]}
    n.preselected_route_sha256='test'
    n._plan_once()
    assert n._waypoints==n._preselected_route_points
    assert n._hier_phase=='LOCAL'


def test_preselected_route_without_global_planner_uses_validated_tolerance_and_local_geometry():
    n=route_node(); n.global_planner_mode='preselected_route'; n.global_planner=None
    n._preselected_route_points=[(0.,0.),(.73,0.),(2.-1.e-13,0.)]
    n._preselected_route_provenance={
        'registered_goal_xy':[2.,0.], 'endpoint_tolerance_m':.25,
    }
    n.preselected_route_sha256='test'
    checked=[]
    n.planner.collision_sweep_clearance_np=lambda start,end: checked.append(('collision',start.copy(),end.copy())) or 1.
    n.planner.driveable_sweep_clearance_np=lambda start,end: checked.append(('driveable',start.copy(),end.copy())) or 1.
    n._plan_once()
    assert n._hier_phase=='LOCAL'
    assert n._waypoints==n._preselected_route_points
    assert checked


def _admission_checks(belief_xy):
    n=route_node(); n.global_planner_mode='preselected_route'; n.global_planner=None
    n.waypoint_arrival_radius_m=.1
    n.belief_m=np.array([belief_xy[0], belief_xy[1], 0.])
    n._preselected_route_points=[(0.,0.),(.2,0.),(1.,0.),(2.,0.)]
    n._preselected_route_provenance={
        'registered_goal_xy':[2.,0.], 'endpoint_tolerance_m':.5,
    }
    n.preselected_route_sha256='test'
    checked=[]
    n.planner.collision_sweep_clearance_np=lambda start,end: checked.append(end.copy()) or 1.
    n.planner.driveable_sweep_clearance_np=lambda start,end: checked.append(end.copy()) or 1.
    n._plan_once()
    assert n._hier_phase=='LOCAL'
    return checked


def test_preselected_route_admission_checks_only_live_connector_and_first_leg():
    checked=_admission_checks((-.3, 0.))
    assert len(checked)==8
    assert max(float(end[0]) for end in checked)<=.2


def test_route_start_inside_arrival_radius_needs_no_connector_turn():
    # The follower treats a waypoint within its arrival radius as reached, so the
    # admission check must not demand an in-place turn toward it.
    checked=_admission_checks((.03, .04))
    assert len(checked)==4
    assert all(float(end[0])==.2 for end in checked[2:])


def test_preselected_route_admission_rejects_changed_coordinates():
    n=route_node(); n.global_planner_mode='preselected_route'; n.global_planner=None
    n._preselected_route_points=[(0.,0.),(.2,0.),(2.,0.)]
    n._preselected_route_provenance={
        'registered_goal_xy':[2.,0.], 'endpoint_tolerance_m':0.,
    }
    n.preselected_route_sha256='test'
    safe,reason=n._global_route_candidate_safe(
        [(0.,0.),(.3,0.),(2.,0.)], np.array([2.,0.]), np.zeros(3))
    assert not safe and reason=='preselected_route_identity_mismatch'


def test_zero_endpoint_tolerance_accepts_json_roundoff_only():
    n=route_node(); n.global_planner_mode='preselected_route'; n.global_planner=None
    n._preselected_route_points=[(0.,0.),(2.+4.e-15,0.)]
    n._preselected_route_provenance={
        'registered_goal_xy':[2.,0.], 'endpoint_tolerance_m':0.,
    }
    n.preselected_route_sha256='test'
    n.planner.collision_sweep_clearance_np=lambda *args: 1.
    n.planner.driveable_sweep_clearance_np=lambda *args: 1.
    n._plan_once()
    assert n._hier_phase=='LOCAL'
    assert n._waypoints==n._preselected_route_points


def test_diagnostic_odom_has_stable_route_admission_identity():
    n=route_node()
    n.use_diagnostic_odom_localization=True
    n.diagnostic_odom_pose=(1.,2.,.3)
    m,P,meta=n._resolve_diagnostic_odom_belief_for_planning()
    np.testing.assert_allclose(m,[1.,2.,.3])
    np.testing.assert_allclose(np.diag(P),[1.e-4]*3)
    assert meta['belief_valid'] and meta['motion_supported']
    assert meta['belief_epoch']=='diagnostic_odom'
    assert meta['belief_revision']==0
    assert meta['belief_frame_id']=='map_bev'
    assert n._execution_belief_is_current(meta)


def test_stop_during_validation_cannot_erase_newer_replacement():
    n=route_node();replacement=np.array([[.1,0.]])
    check=n._global_route_candidate_safe
    def replace(*a,**kw):
        outcome=check(*a,**kw)
        n._publish_safe_stop_command()
        n._waypoints=[(9.,9.)];n._active_controls=replacement
        return outcome
    n._global_route_candidate_safe=replace
    n._plan_once()
    assert n._waypoints==[(9.,9.)] and n._active_controls is replacement
    assert not n.published_routes


def test_expiry_diagnostic_callback_cannot_clear_its_replacement():
    n=route_node();n._active_plan_request=n._capture_plan_request();n._clock.seconds=20.
    replacement=np.array([[.1,0.]])
    n._warn_once_about_expired_tape=lambda _:setattr(n,'_active_controls',replacement)
    assert n._install_control_tape(np.array([[.2,0.]]),original_len=1)=='expired'
    assert n._active_controls is replacement


def test_failed_diagnostic_does_not_prevent_fatal_zero():
    n=route_node();n._fatal_stop_triggered=True
    def fail(_):raise RuntimeError('diagnostic unavailable')
    n.active_execution_diag_pub.publish=fail
    n._publish_command(.2,.1)
    assert n.cmd_pub.messages[-1].linear.x==n.cmd_pub.messages[-1].angular.z==0.


def test_atomic_mission_goal_owns_identity_and_ignores_legacy_resends():
    from unav_common.mission_goal import make_mission_goal,mission_goal_to_json
    n=route_node()
    goal=make_mission_goal(mission_epoch='m',stamp_ns=1,frame_id='map_bev',
                           waypoints=[(2.,0.)],tour_index=0)
    n._mission_goal_cb(NS(data=mission_goal_to_json(goal)))
    generation=n._command_stop_generation
    n._mission_goal_cb(NS(data=mission_goal_to_json(goal)))
    legacy=PoseStamped();legacy.pose.position.x=-100.
    n._goal_cb(legacy)
    assert n.goal_msg.pose.position.x==2.
    assert n._command_stop_generation==generation


def test_completed_mission_cannot_restart_on_delayed_active_resend():
    from dataclasses import replace
    from unav_common.mission_goal import make_mission_goal,mission_goal_to_json
    n=route_node();goal=make_mission_goal(mission_epoch='m',stamp_ns=1,frame_id='map_bev',
        waypoints=[(2.,0.)],tour_index=0)
    for item in (goal,replace(goal,status='completed'),goal):
        n._mission_goal_cb(NS(data=mission_goal_to_json(item)))
    assert n.goal_msg is None
    assert n._active_controls is None


def prepare_live_tape_node():
    n=route_node();n.use_hierarchical=True;n._hier_phase='LOCAL'
    n.planner.collision_cost_model=None;n.planner.nogo_cost_model=None
    n._active_plan_request=n._capture_plan_request()
    n._remember_request_belief(n._resolve_belief_for_planning()[2])
    return n


def test_correction_revalidates_tape_without_extending_its_lifetime():
    n=prepare_live_tape_node()
    assert n._install_control_tape(np.array([[.2,0.],[.2,0.]]),original_len=2)=='installed'
    installed=n._active_plan_started_at
    n._belief_revision+=1;n.belief_m=np.array([.01,0.,0.]);n._clock.seconds=10.1
    n._publish_active_plan_command()
    assert n.cmd_pub.messages[-1].linear.x==.2
    assert n._active_plan_started_at==installed


def test_local_install_trusts_globally_admitted_route_and_retains_full_tape():
    from planning.core.tracker_guard import ControlSafetyResult,SafetyFailure
    n=prepare_live_tape_node()
    controls=np.array([[.2,0.],[.2,.1],[.1,-.1]])
    checked=[]
    def safe(candidate,*_):
        checked.append(candidate.copy())
        return ControlSafetyResult(len(candidate),'safe',SafetyFailure.NONE)
    n._simple_plan_safe_to_execute=safe
    assert n._install_control_tape(controls,original_len=3)=='installed'
    assert checked == []
    np.testing.assert_array_equal(n._active_controls,controls)


def test_command_timer_does_not_re_veto_globally_admitted_route():
    from planning.core.tracker_guard import ControlSafetyResult,SafetyFailure
    n=prepare_live_tape_node()
    controls=np.array([[.2,0.],[.2,.1],[.1,-.1]])
    assert n._install_control_tape(controls,original_len=3)=='installed'
    checked=[]
    def safe(candidate,*_):
        checked.append(candidate.copy())
        return ControlSafetyResult(1,'safe',SafetyFailure.NONE)
    n._simple_plan_safe_to_execute=safe
    n._belief_revision+=1
    n._publish_active_plan_command()
    assert checked == []
    np.testing.assert_array_equal(n._active_controls,controls)


def test_belief_shift_does_not_stop_globally_admitted_route():
    from planning.core.tracker_guard import ControlSafetyResult,SafetyFailure
    n=prepare_live_tape_node()
    n._install_control_tape(np.array([[.2,0.]]),original_len=1)
    n._belief_revision+=1
    n._simple_plan_safe_to_execute=lambda *a:ControlSafetyResult(0,'unsafe',SafetyFailure.COLLISION)
    n._publish_active_plan_command()
    assert n._active_controls is not None
    assert n.cmd_pub.messages[-1].linear.x != 0.


def test_small_goal_change_revokes_tape_immediately():
    n=prepare_live_tape_node()
    n._install_control_tape(np.array([[.2,0.]]),original_len=1)
    goal=PoseStamped();goal.header.frame_id='map_bev';goal.pose.position.x=2.01
    n._goal_cb(goal)
    assert n._active_controls is None
    assert n.cmd_pub.messages[-1].linear.x==0.


def test_global_install_uses_real_supported_belief_record_after_correction():
    from test_planner_node_state_correction import make_state_node
    n=route_node(); f=make_state_node(belief_stamp_s=10.,now_s=10.)
    for key,value in vars(f).items():
        if key not in vars(n):setattr(n,key,value)
    del n._resolve_belief_for_planning
    n.belief_stamp=stamp(10.)
    n.state_correction_ekf=True;n.use_pixel_correction=False
    n.use_diagnostic_odom_localization=False;n.require_state_correction_envelope=True
    n._belief_frame_id='map_bev'
    n._correction_lock=threading.RLock()
    n.heading_update_mode='coupled'
    n.stale_belief_inflate_m2_per_s=0.;n.stale_belief_inflate_cap_m2=0.
    checked=[]
    n.global_planner.collision_sweep_clearance_np=lambda start,end:checked.append(start.copy()) or 1.
    def solve(*a,**kw):
        n._commit_belief(np.array([.01,0.,0.]),np.eye(3)*.01,stamp(10.))
        return valid_result()
    n.global_planner.plan=solve
    n._plan_once()
    assert n._hier_phase=='LOCAL'
    assert n._belief_revision==1
    assert checked[0][0]==pytest.approx(.01)


def test_geometric_route_uses_same_admission(monkeypatch):
    import unav_common.lane_graph_routes as routes
    n=route_node();n.global_planner_mode='geometric_shortest_path'
    n.driveable_geometry_json='test-map'
    monkeypatch.setattr(routes,'generate_route_seeds',lambda *a:[dict(name='safe',waypoints=[(0.,0.),(2.,0.)])])
    n._plan_once()
    assert n._hier_phase=='LOCAL' and n._waypoints==[(0.,0.),(2.,0.)]
    assert n.path_pub.messages


def test_clear_waypoints_do_not_hide_unsafe_segment():
    n=route_node()
    n.global_planner.collision_sweep_clearance_np=lambda start,end: -1. if start[0]<.95<end[0] else 1.
    n._plan_once()
    assert n._hier_phase=='GLOBAL' and not n._waypoints


def test_expired_callback_never_logs_nonzero_after_fatal_latch():
    n=prepare_live_tape_node()
    n._install_control_tape(np.array([[.2,0.]]),original_len=1)
    n._fatal_stop_triggered=True
    n._publish_active_plan_command()
    assert n.cmd_pub.messages[-1].linear.x==0.
    assert n.active_execution_diag_pub.messages[-1].data[5]==0.
