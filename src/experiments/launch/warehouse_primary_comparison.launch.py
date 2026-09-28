from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

# Quick launch:
# ros2 launch experiments warehouse_primary_comparison.launch.py \
#     planner:=visibility_aware_efe task:=full_traverse_handover seed:=0 \
#     yolo_model:=logs/perception_models/warehouse_yolo_detector_v1/model.pt \
#     comparison_method_id:=efe_main

DEFAULT_PLANNER = 'visibility_aware_efe'
ALLOWED_PLANNERS = ('visibility_aware_efe', 'constant_R_efe', 'geometric_shortest_path')
PLANNER_DESCRIPTION = (
    'Primary thesis comparison: visibility_aware_efe (C2) | constant_R_efe (C1) | '
    'geometric_shortest_path (C0 conventional-navigation baseline)'
)


def _planner_precision_arguments():
    return [
        DeclareLaunchArgument('camera_network_artifact_path', default_value=''),
        DeclareLaunchArgument('camera_network_active_camera_ids', default_value=''),
        DeclareLaunchArgument('camera_network_objective', default_value='legacy_pixel_chart'),
        DeclareLaunchArgument('network_goal_std_m', default_value='0.10'),
        DeclareLaunchArgument('kouw_et1_ambiguity', default_value='true',
                              description='Kouw ET1 ambiguity on the commissioned R; see docs/PLANNER.md'),
        DeclareLaunchArgument('network_goal_std_start_m', default_value='-1.0',
                              description='Loose goal-prior sigma at rollout start; <=0 keeps the constant prior'),
        DeclareLaunchArgument('camera_network_updates_per_step', default_value='1'),
        DeclareLaunchArgument('optimizer_control_block_steps', default_value='1'),
        DeclareLaunchArgument('horizon', default_value='40'),
        DeclareLaunchArgument('dt', default_value='0.25'),
        DeclareLaunchArgument('v_max', default_value='0.22'),
        DeclareLaunchArgument('max_predict_speed_mps', default_value='0.0'),
        DeclareLaunchArgument('state_correction_mode', default_value='fused'),
        DeclareLaunchArgument('state_max_correction_jump_m', default_value='0.0'),
        DeclareLaunchArgument('discount_gamma', default_value='0.995'),
        DeclareLaunchArgument('goal_prior_u_std_start', default_value='80.0'),
        DeclareLaunchArgument('goal_prior_v_std_start', default_value='80.0'),
        DeclareLaunchArgument('goal_prior_u_std_final', default_value='4.0'),
        DeclareLaunchArgument('goal_prior_v_std_final', default_value='4.0'),
        DeclareLaunchArgument('goal_tightening_power', default_value='0.45'),
        DeclareLaunchArgument('r_visible_uv', default_value='2.5'),
        DeclareLaunchArgument('r_miss_uv', default_value='120.0'),
        DeclareLaunchArgument('pixel_correction_nis_threshold', default_value='9.21',
                              description='Reject pixel corrections with 2D NIS above this threshold; 0 disables NIS gating.'),
        DeclareLaunchArgument('state_reanchor_m', default_value='0.0'),
        DeclareLaunchArgument('state_max_predict_dt_s', default_value='1.5'),
        DeclareLaunchArgument('state_reject_inflate_m2', default_value='0.0'),
        DeclareLaunchArgument('robot_length_m', default_value='0.8'),
        DeclareLaunchArgument('robot_width_m', default_value='0.55'),
        DeclareLaunchArgument('robot_collision_radius_m', default_value='0.48541219597369'),
        DeclareLaunchArgument('odom_heading_timeout_s', default_value='0.75',
                              description='Maximum odometry age used by the pixel-to-BEV state projection orientation field.'),
        DeclareLaunchArgument('heading_update_mode', default_value='coupled'),
        DeclareLaunchArgument('local_controller_type', default_value='ff_fb',
                              description='Waypoint-tracking law; canonical thesis campaigns use ff_fb'),
        DeclareLaunchArgument('use_nogo_cost', default_value='auto'),
        DeclareLaunchArgument('nogo_penalty_type', default_value='warning_band'),
        DeclareLaunchArgument('nogo_weight', default_value='40.0'),
        DeclareLaunchArgument('nogo_safe_distance', default_value='0.0',
                              description='Legacy centre-point radius; zero for exact rectangular-footprint runs.'),
        DeclareLaunchArgument('nogo_logbarrier_eps', default_value='0.001'),
        DeclareLaunchArgument('nogo_warning_band', default_value='0.05'),
        DeclareLaunchArgument('nogo_near_weight', default_value='50.0'),
        DeclareLaunchArgument('use_belief_nogo_cost', default_value='false'),
        DeclareLaunchArgument('nogo_belief_kappa', default_value='1.0'),
        DeclareLaunchArgument('use_odom_for_predict', default_value='true'),
        DeclareLaunchArgument('odom_topic', default_value='/odom_noisy'),
        DeclareLaunchArgument('optimizer_multistart', default_value='false',
                              description='Offer multiple optimizer-init seeds; lowest-EFE candidate wins. Same seeds for all conditions.'),
        DeclareLaunchArgument('optimizer_multistart_include_direct', default_value='true',
                              description='Include a steer-straight-to-goal seed when multistart is on.'),
        DeclareLaunchArgument('optimizer_initial_routes_json', default_value='',
                              description='Optional JSON list of named waypoint routes used only as optimizer seeds (not mission waypoints).'),
        DeclareLaunchArgument('optimizer_terminal_goal_tolerance_m', default_value='0.0',
                              description='If positive, valid multistart candidates reaching this terminal goal tolerance outrank incomplete candidates before EFE cost comparison.'),
        DeclareLaunchArgument('optimizer_route_seed_mode', default_value='explicit',
                              description="Multistart route-seed source: 'explicit' uses optimizer_initial_routes_json; 'lane_graph' generates condition-neutral lane-centre Manhattan seeds from the driveable map."),
        DeclareLaunchArgument('use_hierarchical', default_value='false'),
        DeclareLaunchArgument('global_planner_mode', default_value='efe',
                              description="Global route source: 'efe' (one-shot global EFE solve, C1/C2), 'geometric_shortest_path' (C0), or 'preselected_route' (hash-bound external polyline; no global solve)."),
        DeclareLaunchArgument('preselected_route_json', default_value='',
                              description='Exactly one JSON [[x,y],...] polyline; canonical bytes are SHA-256 checked before launch.'),
        DeclareLaunchArgument('preselected_route_sha256', default_value='',
                              description='Expected SHA-256 of the canonical preselected route JSON.'),
        DeclareLaunchArgument('preselected_route_source_path', default_value='',
                              description='Original offline route-geometry artifact from which the polyline was selected.'),
        DeclareLaunchArgument('preselected_route_source_sha256', default_value='',
                              description='Expected SHA-256 of the complete source route-geometry artifact.'),
        DeclareLaunchArgument('preselected_route_clearance_m', default_value='0.25',
                              description='Declared minimum clearance inside the frozen driveable union.'),
        DeclareLaunchArgument('preselected_route_endpoint_tolerance_m', default_value='0.25',
                              description='Maximum start/end error; frozen protocol forbids values above 0.25 m.'),
        DeclareLaunchArgument('preselected_route_sample_step_m', default_value='0.04',
                              description='Maximum segment-walk step for clearance validation; must be <=0.04 m.'),
        DeclareLaunchArgument('global_horizon', default_value='60'),
        DeclareLaunchArgument('global_dt', default_value='0.0',
                              description='Global planner step size; 0.0 inherits dt.'),
        DeclareLaunchArgument('local_horizon', default_value='12'),
        DeclareLaunchArgument('local_plan_rate', default_value='4.0'),
        DeclareLaunchArgument('local_optimizer_maxiter', default_value='60'),
        DeclareLaunchArgument('global_use_ambiguity', default_value='true'),
        DeclareLaunchArgument('local_use_ambiguity', default_value='false'),
        DeclareLaunchArgument('global_optimizer_multistart', default_value='true'),
        DeclareLaunchArgument('local_optimizer_multistart', default_value='true'),
        DeclareLaunchArgument('local_use_visibility_model', default_value='false'),
        DeclareLaunchArgument('local_use_belief_nogo_cost', default_value='false'),
        DeclareLaunchArgument('local_nogo_penalty_type', default_value=''),
        DeclareLaunchArgument('local_nogo_weight', default_value='-1.0'),
        DeclareLaunchArgument('local_nogo_safe_distance', default_value='-1.0'),
        DeclareLaunchArgument('local_goal_prior_u_std_start', default_value='-1.0'),
        DeclareLaunchArgument('local_goal_prior_v_std_start', default_value='-1.0'),
        DeclareLaunchArgument('local_goal_prior_u_std_final', default_value='-1.0'),
        DeclareLaunchArgument('local_goal_prior_v_std_final', default_value='-1.0'),
        DeclareLaunchArgument('waypoint_spacing_m', default_value='0.2'),
        DeclareLaunchArgument('waypoint_arrival_radius_m', default_value='0.1'),
        DeclareLaunchArgument('local_replan_min_remaining_s', default_value='0.0',
                              description='Skip local replanning while the active control tape has more than this many seconds remaining.'),
        DeclareLaunchArgument('local_replan_on_waypoint_change', default_value='false',
                              description='In hierarchical mode, replan local controls only when the waypoint changes or the active tape expires.'),
        DeclareLaunchArgument('latency_compensate_plan_handoff', default_value='false',
                              description='Start executing a solved local plan at the control index matching solver latency.'),
        DeclareLaunchArgument('simple_tracker_yaw_gate_rad', default_value='0.65',
                              description='Heading error above which the FF/FB tracker uses its corner crawl speed.'),
        DeclareLaunchArgument('ff_fb_turn_rate_limit_rad_s', default_value='0.80',
                              description='Angular-speed limit used by the FF/FB tracker.'),
        DeclareLaunchArgument('ff_fb_corner_crawl_speed_mps', default_value='0.18',
                              description='Forward speed used by the FF/FB tracker through a large heading error.'),
        DeclareLaunchArgument('ff_fb_pivot_heading_error_rad', default_value='2.60',
                              description='Heading error above which the FF/FB tracker rotates in place.'),
        DeclareLaunchArgument('cmd_publish_rate', default_value='10.0'),
    ]


def _launch_setup(context, *args, **kwargs):
    from experiments.core.visibility_launch_common import (
        build_agent_runtime_actions,
        parse_common_launch_config,
        resolve_world_setup,
    )

    cfg = parse_common_launch_config(context)
    planner = str(cfg.get('planner', DEFAULT_PLANNER) or DEFAULT_PLANNER).strip()
    if planner not in ALLOWED_PLANNERS:
        raise RuntimeError(f"planner must be one of: {', '.join(ALLOWED_PLANNERS)}")

    requested_global_mode = str(cfg.get('global_planner_mode', 'efe') or 'efe').strip().lower()
    cfg['planner'] = planner

    if planner == 'constant_R_efe':
        cfg['use_visibility_model'] = False
        # C1 is constant-observability EFE: risk and ambiguity remain active,
        # but the observation covariance is spatially uniform instead of GP-based.
        cfg['use_ambiguity'] = True
        cfg['use_obs_risk'] = True
        cfg['global_planner_mode'] = 'efe'
    elif planner == 'geometric_shortest_path':
        # C0 conventional-navigation baseline: geometry-only shortest-path route
        # over the same driveable + no-go geometry, tracked by the same local
        # controller. No camera-reliability model and no EFE reasoning; the
        # one-shot global EFE solve is skipped, so the EFE terms are unused.
        cfg['use_visibility_model'] = False
        cfg['global_planner_mode'] = 'geometric_shortest_path'
        cfg['use_ambiguity'] = False
        cfg['use_obs_risk'] = False
    else:
        cfg['use_visibility_model'] = True
        cfg['global_planner_mode'] = 'efe'

    if requested_global_mode == 'preselected_route':
        # This is an execution source, not a fourth planner condition. Both
        # route arms use the same belief/filter/local-tracker configuration and
        # skip every global EFE/shortest-path solve.
        cfg['global_planner_mode'] = 'preselected_route'
        cfg['use_visibility_model'] = False
        cfg['use_ambiguity'] = False
        cfg['use_obs_risk'] = False

    cfg = resolve_world_setup(cfg)
    return build_agent_runtime_actions(cfg)


def generate_launch_description():
    world_profiles_default = PathJoinSubstitution([
        FindPackageShare('experiments'), 'config', 'world_profiles.yaml',
    ])
    tasks_default = PathJoinSubstitution([
        FindPackageShare('experiments'), 'config', 'tasks.yaml',
    ])

    return LaunchDescription([
        DeclareLaunchArgument('planner', default_value=DEFAULT_PLANNER, description=PLANNER_DESCRIPTION),
        DeclareLaunchArgument('world', default_value='warehouse_full_4cam.world.sdf'),
        DeclareLaunchArgument('world_profiles', default_value=world_profiles_default, description='World profile YAML'),
        DeclareLaunchArgument('tasks_yaml', default_value=tasks_default, description='Task YAML'),
        DeclareLaunchArgument('task', default_value='', description='Task name; empty uses the world profile recommended_task'),
        DeclareLaunchArgument('seed', default_value='0'),
        DeclareLaunchArgument('comparison_method_id', default_value=''),
        DeclareLaunchArgument('auto_stop_on_goal', default_value='true'),
        DeclareLaunchArgument('enable_mission', default_value='true',
                              description='Publish a navigation goal (true). Set false for a commissioning coverage drive: no goal, EFE planner silent, belief EKF still runs while an external /cmd_vel source drives.'),
        DeclareLaunchArgument(
            'wait_for_belief_before_first_goal',
            default_value='false',
            description='Hold the first mission goal until /planner_belief satisfies the configured xy-sigma readiness gate.',
        ),
        DeclareLaunchArgument(
            'initial_belief_max_sigma_m',
            default_value='0.0',
            description='Maximum xy sigma for the initial-belief readiness gate; <=0 accepts the first finite belief.',
        ),
        DeclareLaunchArgument(
            'operational_belief_timeout_s', default_value='0.5',
            description='Maximum age of the operational predicted belief used to release a mission goal.'),
        DeclareLaunchArgument('goal_success_radius', default_value='0.20'),
        DeclareLaunchArgument('goal_success_hold_s', default_value='2.0'),
        DeclareLaunchArgument('goal_stable_radius', default_value='0.20'),
        DeclareLaunchArgument('goal_stable_hold_s', default_value='2.0'),
        DeclareLaunchArgument('goal_stable_max_displacement_m', default_value='0.04'),
        DeclareLaunchArgument('run_timeout_after_first_cmd_s', default_value='75.0'),
        DeclareLaunchArgument('first_cmd_linear_eps', default_value='0.02'),
        DeclareLaunchArgument('first_cmd_angular_eps', default_value='0.10'),
        DeclareLaunchArgument('stuck_window_s', default_value='8.0'),
        DeclareLaunchArgument('stuck_max_displacement_m', default_value='0.08'),
        DeclareLaunchArgument('stuck_max_goal_improvement_m', default_value='0.05'),
        DeclareLaunchArgument('stuck_cmd_fraction_min', default_value='0.50'),
        DeclareLaunchArgument('stuck_idle_cmd_fraction_max', default_value='0.10'),
        DeclareLaunchArgument('use_command_noise', default_value='true'),
        DeclareLaunchArgument('use_encoder_noise', default_value='true'),
        DeclareLaunchArgument('encoder_noise_linear_slip_mean', default_value='0.0',
                              description='Mean multiplicative linear encoder slip used for /odom_noisy '
                                          '(0: scale-calibrated odometry).'),
        DeclareLaunchArgument('encoder_noise_linear_slip_std', default_value='0.05',
                              description='Stddev of multiplicative linear encoder slip used for /odom_noisy.'),
        DeclareLaunchArgument('encoder_noise_angular_slip_mean', default_value='0.00',
                              description='Mean multiplicative angular encoder slip used for /odom_noisy.'),
        DeclareLaunchArgument('encoder_noise_angular_slip_std', default_value='0.03',
                              description='Stddev of multiplicative angular encoder slip used for /odom_noisy.'),
        DeclareLaunchArgument('encoder_noise_linear_additive_std', default_value='0.004',
                              description='Stddev of additive linear encoder noise used for /odom_noisy.'),
        DeclareLaunchArgument('encoder_noise_angular_additive_std', default_value='0.020',
                              description='Stddev of additive angular encoder noise used for /odom_noisy.'),
        DeclareLaunchArgument('encoder_noise_correlation_alpha', default_value='0.80',
                              description='AR(1) correlation of encoder slip states.'),
        DeclareLaunchArgument('process_noise_model', default_value='encoder',
                              description="'encoder': Q set from the simulated encoder noise; 'constant_psd': legacy."),
        DeclareLaunchArgument('encoder_wheel_diameter_ratio_error', default_value='0.00121',
                              description='Systematic right/left wheel diameter mismatch (UMBmark LabMate).'),
        DeclareLaunchArgument('encoder_wheelbase_ratio', default_value='0.991765',
                              description='True/nominal wheelbase ratio (UMBmark LabMate 337.2/340).'),
        DeclareLaunchArgument('yolo_model', default_value='', description='Local path to a trained YOLO .pt model'),
        DeclareLaunchArgument(
            'outcome_journal_path', default_value='',
            description='Durable JSONL detector-outcome journal; campaigns set an attempt-owned path.'),
        DeclareLaunchArgument(
            'manager_outcome_journal_path', default_value='',
            description='Durable JSONL camera-manager outcome journal; campaigns set a distinct attempt-owned path.'),
        DeclareLaunchArgument('yolo_device', default_value='', description='Ultralytics device string; empty lets Ultralytics choose'),
                # 960, matching every trained model. Until 2026-08-21 this defaulted to 640
        # while all five checkpoints were trained at imgsz 960, so inference ran at a
        # resolution the weights had never seen. Measured on 200 real val frames with
        # the frozen four-camera detector: the median bottom-edge error against the
        # ground-truth box improves from 3.24 to 2.52 px (mask) and 2.88 to 2.40 px
        # (box) going from 640 to 960, and recall is unchanged. 1280 is better again
        # (2.83 / 2.02) but a five-image batch costs 164 ms against the 200 ms the
        # 5 Hz cameras allow, leaving nothing for the rest of the stack; 960 costs
        # 99 ms. The measurement matters because the MEASUREMENT is the mask's bottom
        # edge, so inference resolution is a measurement parameter, not a speed knob.
        DeclareLaunchArgument('yolo_imgsz', default_value='960'),
        DeclareLaunchArgument('yolo_conf_threshold', default_value='0.25'),
        DeclareLaunchArgument('yolo_predict_conf_floor', default_value='0.05',
                              description='Internal pre-NMS floor; must not exceed the reported confidence threshold.'),
        DeclareLaunchArgument('yolo_iou_threshold', default_value='0.45'),
        DeclareLaunchArgument('yolo_target_class', default_value='robot'),
        DeclareLaunchArgument('yolo_class_id', default_value='-1'),
        DeclareLaunchArgument('yolo_use_masks', default_value='true', description='Use YOLO segmentation masks for pixel reference when available'),
        DeclareLaunchArgument('yolo_min_mask_area_px', default_value='12.0'),
        DeclareLaunchArgument('yolo_mask_bottom_band_px', default_value='3.0'),
        DeclareLaunchArgument('yolo_min_bbox_area_px', default_value='0.0'),
        DeclareLaunchArgument(
            'yolo_max_batch_stamp_skew_s', default_value='0.05',
            description='Capture-stamp grouping tolerance; must remain below the 0.20 s camera period.'),
        DeclareLaunchArgument('yolo_debug_frame_dir', default_value=''),
        DeclareLaunchArgument('yolo_debug_crop_dir', default_value=''),
        DeclareLaunchArgument('yolo_use_torchscript', default_value='false',
                              description='Load the TorchScript export of the model (single C++ forward dispatch; bit-identical detections)'),
        DeclareLaunchArgument('yolo_runtime_backend', default_value='native',
                              description='Detector backend: native (evidence default) or torchscript (diagnostic successor).'),
        DeclareLaunchArgument('yolo_compiled_model', default_value='',
                              description='Fixed-shape .torchscript artifact required by yolo_runtime_backend:=torchscript.'),
        DeclareLaunchArgument('yolo_input_transport', default_value='ros',
                              description='Camera input: ros (default) or direct_gz (diagnostic, bypasses RGB bridge).'),
        DeclareLaunchArgument('yolo_runtime_trace_period_s', default_value='0.0'),
        DeclareLaunchArgument('yolo_warmup_iters', default_value='3',
                              description='Dummy inferences at startup to pay lazy CUDA/JIT init off the hot path'),
        DeclareLaunchArgument('yolo_inference_in_callback', default_value='true',
                              description='Run inference synchronously in the image callback (single thread, no GIL contention)'),
        DeclareLaunchArgument('use_pixel_correction', default_value='true'),
        DeclareLaunchArgument('use_diagnostic_odom_localization', default_value='false',
                              description='DIAGNOSTIC ONLY: feed raw odometry (not ground '
                                          'truth) to the planner as its belief, bypassing '
                                          'the cameras. Never true in a comparison run.'),
        DeclareLaunchArgument('multicam_belief', default_value='false',
                              description='Multi-camera belief mode: replace the single-cam detector+pixel_to_bev '
                                          'with the batched 4-cam detector + active camera_manager -> /state/bev.'),
        DeclareLaunchArgument('manager_gp_artifact_template', default_value='',
                              description='Per-camera GP npz path template with {camera_id} for camera_manager selection.'),
        DeclareLaunchArgument('manager_min_spatial_trust', default_value='0.15',
                              description='Min spatial trust to release a camera_manager correction (lower = more corrections).'),
        DeclareLaunchArgument('manager_decision_rate_hz', default_value='5.0'),
        DeclareLaunchArgument('manager_camera_ids', default_value='',
                              description='Comma-separated camera_ids the fusion manager uses (e.g. "camera_A,camera_B,camera_C,camera_D"); empty = all 4. Restrict to one for a single-camera localization baseline.'),
        DeclareLaunchArgument('manager_fusion_mode', default_value='true',
                              description='true = covariance-weighted fusion of all in-view cameras; false = hysteretic single-camera SELECTION (handover). Toggle for the selection-vs-fusion comparison.'),
        DeclareLaunchArgument('manager_fusion_disagreement_gate_m', default_value='0.6',
                              description='Maximum residual from the robust fusion seed before a camera is excluded.'),
        DeclareLaunchArgument('manager_require_source_batch_id', default_value='true',
                              description='Reject detector observations that cannot prove which inference batch produced them.'),
        DeclareLaunchArgument('manager_bootstrap_min_cameras', default_value='2',
                              description='Minimum agreeing cameras required for prior-free belief initialisation.'),
        DeclareLaunchArgument('manager_bootstrap_max_disagreement_m', default_value='0.30',
                              description='Maximum pairwise camera disagreement allowed during prior-free initialisation.'),
        DeclareLaunchArgument('manager_use_task_start_as_bootstrap_prior', default_value='false',
                              description='Use the declared physical task start only until the recursive camera belief is anchored.'),
        DeclareLaunchArgument('initial_belief_from_task_start', default_value='false',
                              description='Start the belief at the declared task start pose (no camera bootstrap).'),
        DeclareLaunchArgument('initial_belief_sigma_xy_m', default_value='0.10'),
        DeclareLaunchArgument('initial_belief_sigma_theta_rad', default_value='0.2617993877991494'),
        DeclareLaunchArgument('manager_bootstrap_prior_counts_as_support', default_value='false',
                              description='Count the declared task-start prior as one bootstrap support source, while still requiring at least one admitted camera within the bootstrap disagreement bound.'),
        DeclareLaunchArgument('manager_require_gp_artifacts', default_value='true',
                              description='true = per-camera reliability GP sets each observation covariance (GP). false = no GP; every observation uses a fixed covariance (non-GP baseline). Set with an empty manager_gp_artifact_template for the non-GP arm.'),
        DeclareLaunchArgument('manager_fusion_max_timestamp_spread_s', default_value='0.05',
                              description='Maximum timestamp spread among views fused into one correction; older cached views are excluded. Must stay BELOW the detector period (0.20 s at 5 Hz) or a whole stale round can be fused as if simultaneous.'),
        DeclareLaunchArgument('manager_covariance_profile', default_value='commissioned_sigma_px',
                              description='The sensor model. commissioned_sigma_px states '
                                          'R_pix = sigma_px^2 I from the frozen calibration and lets '
                                          'each camera geometry size the ellipse on the floor. '
                                          'commissioned_world_R instead states the residual scatter '
                                          'measured directly on the floor per camera and detector '
                                          'confidence, and applies the offset commissioned with it; '
                                          'the pixel route understates the held-out error about 300x '
                                          'in variance. Nothing downstream floors or inflates either.'),
        DeclareLaunchArgument('manager_commissioned_world_covariance_path', default_value='',
                              description='commissioning.json the commissioned_world_R profile reads '
                                          'its per-camera per-confidence covariance and offset from. '
                                          'Required by that profile: the covariance is read, never '
                                          'typed in.'),
        DeclareLaunchArgument('manager_commissioned_calibration_path', default_value='',
                              description='calibration.json the commissioned_sigma_px profile reads '
                                          'sigma_px from. Required by that profile: the detector noise '
                                          'is read, never typed in.'),
        DeclareLaunchArgument('manager_commissioned_sigma_px', default_value='0.0',
                              description='Deliberate override of the commissioned sigma_px (px). '
                                          '0 = read it from the calibration artifact.'),
        DeclareLaunchArgument('manager_commissioned_per_camera_sigma', default_value='false',
                              description='Give each camera its own commissioned pixel noise '
                                          'instead of the pooled one. Commissioning measures '
                                          'both; pooling is a choice, and on these cameras one '
                                          'camera needs ~3x the variance of another.'),
        DeclareLaunchArgument('manager_fusion_common_mode_std_m', default_value='0.0',
                              description='The error the cameras make together (m), added \n'
                                          'back after they are combined. Commissioned once \n'
                                          'against ground truth; 0 reproduces the previous \n'
                                          'independent-cameras assumption.'),
        DeclareLaunchArgument('manager_correction_timestamp_compensation', default_value='false',
                              description='Carry each fused correction forward from the pose it '
                                          'describes to the pose it is used on. Off reproduces '
                                          'the historical behaviour, which applies a ~400 ms old '
                                          'correction as if it were current (8.2 cm of lag bias '
                                          'at 0.22 m/s).'),
        DeclareLaunchArgument('manager_correction_propagation_drift_std', default_value='0.05',
                              description='Uncertainty the propagation itself adds, per second '
                                          'of correction age (m/s).'),
        DeclareLaunchArgument('manager_correction_residual_interval_s', default_value='0.05',
                              description='Interval between carrying a correction forward and '
                                          'consuming it. Declared as uncertainty along the '
                                          'direction of travel, because it is a bias there.'),
        DeclareLaunchArgument('manager_sensor_gate_config_path', default_value='',
                              description='Versioned deterministic belief-independent sensor '
                                          'gate YAML for commissioned navigation.'),
        DeclareLaunchArgument('manager_fusion_rule', default_value='independent',
                              description='How several cameras become one measurement: '
                                          'best_single | distance_angle | independent | '
                                          'joint_network. Campaigns name this explicitly.'),
        DeclareLaunchArgument('manager_observation_model', default_value='raw_box',
                              description='Commissioned observation family: raw_box, learned_nn, '
                                          'or visibility_patch.'),
        DeclareLaunchArgument('manager_availability_model_path', default_value='',
                              description='Frozen Stage-08 Q1 geometry availability JSON artifact.'),
        DeclareLaunchArgument('manager_availability_model_expected_sha256', default_value='',
                              description='Exact SHA-256 required for the Stage-08 artifact.'),
        DeclareLaunchArgument('manager_learned_correction_path', default_value='',
                              description='Packaged neural box-correction artifact. Required by the '
                                          'learned_nn observation model.'),
        DeclareLaunchArgument('manager_visibility_sensor_model_path', default_value='',
                              description='Hash-bound M4 visibility correction and matched R package.'),
        DeclareLaunchArgument('manager_visibility_sensor_model_expected_sha256', default_value='',
                              description='Exact SHA-256 required for the M4 sensor-model package.'),
        DeclareLaunchArgument('manager_perception_sensor_model_path', default_value='',
                              description='Hash-bound commissioned perception mean-plus-R package.'),
        DeclareLaunchArgument('manager_perception_sensor_model_expected_sha256', default_value='',
                              description='Exact SHA-256 required for the perception sensor model.'),
        DeclareLaunchArgument('manager_max_measurement_age_s', default_value='1.25',
                              description='Maximum correction age admitted by selection or fusion; should not exceed planner freshness.'),
        DeclareLaunchArgument('manager_age_decay_s', default_value='1.25'),
        DeclareLaunchArgument('manager_min_association_confidence', default_value='0.30'),
        DeclareLaunchArgument('manager_required_consecutive_better_frames', default_value='1'),
        DeclareLaunchArgument('manager_max_cross_camera_disagreement_m', default_value='1.0'),
        DeclareLaunchArgument('manager_require_consistency_when_source_available', default_value='false'),
        DeclareLaunchArgument('state_correction_ekf', default_value='',
                              description='Multicam: fuse /state/bev via latency-compensated EKF (paper belief filter) instead of hard-reset. Empty=multicam default ON.'),
        DeclareLaunchArgument(
            'require_state_correction_envelope', default_value='false',
            description='Require source-batch-identified fused corrections for the EKF.'),
        DeclareLaunchArgument(
            'stale_belief_inflate_m2_per_s', default_value='0.0',
            description='Optional declared xy covariance penalty per stale second.'),
        DeclareLaunchArgument(
            'stale_belief_inflate_cap_m2', default_value='0.0',
            description='Cap for the optional stale-belief covariance penalty.'),
        DeclareLaunchArgument('multicam_scheduled', default_value='false',
                              description='Reliability-aware scheduled detector: one inference/cycle on the coverage-best camera -> /state/bev.'),
        DeclareLaunchArgument('scheduled_coverage_artifact', default_value='',
                              description='npz with per-camera P_camera_X_map + xs/ys used to pick the camera per belief position.'),
        DeclareLaunchArgument('scheduled_report_std_m', default_value='0.15'),
        DeclareLaunchArgument('scheduled_rate_hz', default_value='5.0'),
        DeclareLaunchArgument('scheduled_selection_mode', default_value='coverage_best_with_fallback',
                              description='coverage_best_with_fallback for deployment; round_robin for commissioning evidence collection.'),
        DeclareLaunchArgument('pixel_topic', default_value='/perception/pixel_pose',
                              description='World-pixel measurement topic the planner corrects on. '
                                          'Point at a fault-injector output (e.g. /perception/pixel_pose_faulted) '
                                          'for controlled calibration-fault ablations; default preserves the locked path.'),
        DeclareLaunchArgument('command_noise_output_topic', default_value='/cmd_vel',
                              description='Topic the actuation-noise node publishes to. Route to /cmd_vel_pregate '
                                          'and interpose a safe-degradation gate (N3 safe-stop) before /cmd_vel; '
                                          'default /cmd_vel preserves the locked path.'),
        *_planner_precision_arguments(),
        DeclareLaunchArgument('enable_logging', default_value='true'),
        DeclareLaunchArgument('log_dir', default_value='logs/experiments'),
        DeclareLaunchArgument('campaign_config_path', default_value='',
                              description='Exact campaign YAML to hash and snapshot into every run.'),
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument(
            'nvidia_offload', default_value='true',
            description='Explicitly select NVIDIA PRIME/EGL rendering for headless camera sensors.'),
        DeclareLaunchArgument('use_rviz', default_value='false'),
        DeclareLaunchArgument('reset_world', default_value='false'),
        DeclareLaunchArgument('lockstep', default_value='false',
                              description='Pause Gazebo and advance only after the odometry, controller, detector, and manager barriers complete.'),
        DeclareLaunchArgument('lockstep_control_step_iterations', default_value='100'),
        DeclareLaunchArgument('lockstep_camera_every_control_steps', default_value='2'),
        DeclareLaunchArgument('lockstep_max_control_steps', default_value='0'),
        DeclareLaunchArgument('bridge_camera_a', default_value='true',
                              description='Bridge primary external-camera RGB/camera_info; false only with yolo_input_transport:=direct_gz.'),
        DeclareLaunchArgument('bridge_camera_b', default_value='true',
                              description='Bridge the north-west external camera RGB/camera_info topics.'),
        DeclareLaunchArgument('bridge_camera_c', default_value='true',
                              description='Bridge the south-east external camera RGB/camera_info topics.'),
        DeclareLaunchArgument('bridge_camera_d', default_value='true',
                              description='Bridge the north-east external camera RGB/camera_info topics.'),
        OpaqueFunction(function=_launch_setup),
    ])
