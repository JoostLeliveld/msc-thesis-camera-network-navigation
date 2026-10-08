# Thesis → code

Where each part of the thesis lives in this repository. Section names follow the
submitted thesis; figure numbers follow [`figures/README.md`](../figures/README.md).

## Method sections

| Thesis section | Offline (fitting) | Online (ROS runtime) | Further reading |
| --- | --- | --- | --- |
| Measurement-model data collection | [`pipeline/capture/`](../pipeline/capture), [`pipeline/dataset.py`](../pipeline/dataset.py), [`pipeline/dataset_lock.json`](../pipeline/dataset_lock.json), [`pipeline/audit_dataset.py`](../pipeline/audit_dataset.py) | — | [METHOD §2](METHOD.md) |
| Detector and observation admission (appendix) | [`pipeline/detector/`](../pipeline/detector), [`pipeline/detect.py`](../pipeline/detect.py), [`pipeline/gate.py`](../pipeline/gate.py), [`config/sensor_gate.yaml`](../config/sensor_gate.yaml) | [`batched_four_camera_yolo_node.py`](../src/perception/perception/nodes/batched_four_camera_yolo_node.py) | [METHOD §3](METHOD.md) |
| Systematic-displacement correction | [`pipeline/fit_correction.py`](../pipeline/fit_correction.py) | `CommissionedVisibilitySensorModel.correction_ray` in [`commissioned_visibility.py`](../src/reliability/reliability/commissioned_visibility.py) | [METHOD §4](METHOD.md) |
| Measurement covariance (R0, R1, R2) | [`pipeline/corrected_residuals.py`](../pipeline/corrected_residuals.py), [`pipeline/fit_covariance.py`](../pipeline/fit_covariance.py), [`pipeline/evaluate_ddev.py`](../pipeline/evaluate_ddev.py), [`pipeline/package_runtime.py`](../pipeline/package_runtime.py) | `_ray_r_covariance`, `_spatial_covariance` in `commissioned_visibility.py` | [METHOD §5](METHOD.md) |
| Multi-camera fusion | [`pipeline/final_audit_fusion.py`](../pipeline/final_audit_fusion.py) (offline evaluation) | `CameraManagerNode._decide_fused` in [`camera_manager_node.py`](../src/reliability/reliability/nodes/camera_manager_node.py); `independent_measurement_fusion_2d` in [`fusion.py`](../src/reliability/reliability/fusion.py) | [METHOD](METHOD.md) |
| State estimation | — | [`belief_correction.py`](../src/planning/planning/core/belief_correction.py), [`dynamics.py`](../src/planning/planning/core/dynamics.py), [`encoder_noise_model.py`](../src/planning/planning/core/encoder_noise_model.py) | [PROCESS_NOISE](PROCESS_NOISE.md) |
| Belief-space route planning | [`pipeline/planning_precision.py`](../pipeline/planning_precision.py) (information field), [`pipeline/solve_routes.py`](../pipeline/solve_routes.py) | [`casadi_efe.py`](../src/planning/planning/core/casadi_efe.py), [`camera_network.py`](../src/planning/planning/core/camera_network.py), [`base_planner.py`](../src/planning/planning/planners/base_planner.py), [`efe_agent_node.py`](../src/planning/planning/nodes/efe_agent_node.py) | [PLANNER](PLANNER.md) |
| Simulation actuation and encoder noise (appendix) | — | [`src/sim/sim/encoder_noise_node.py`](../src/sim/sim/encoder_noise_node.py), [`src/sim_command_guard/`](../src/sim_command_guard) | [PROCESS_NOISE](PROCESS_NOISE.md) |

## Experiments

| Thesis section | Run | Analyse | Result file |
| --- | --- | --- | --- |
| Localisation on held-out camera data | [`pipeline/audit_protocol.py`](../pipeline/audit_protocol.py), [`pipeline/final_audit.py`](../pipeline/final_audit.py), [`pipeline/final_audit_fusion.py`](../pipeline/final_audit_fusion.py) | [`pipeline/analyze_error_distribution.py`](../pipeline/analyze_error_distribution.py) | [`localisation.json`](../results/localisation.json), [`fusion.json`](../results/fusion.json), [`error_distribution.json`](../results/error_distribution.json) |
| Navigation under camera dropout | [`pipeline/routes.sh`](../pipeline/routes.sh), [`pipeline/campaign.sh`](../pipeline/campaign.sh), [`pipeline/campaign_runner.py`](../pipeline/campaign_runner.py), tasks in [`pipeline/tasks.yaml`](../pipeline/tasks.yaml) | [`pipeline/analyze_campaign.py`](../pipeline/analyze_campaign.py), [`pipeline/score_collisions.py`](../pipeline/score_collisions.py), [`pipeline/analyze_temporal_correlation.py`](../pipeline/analyze_temporal_correlation.py), [`figures/make_runtime_coverage.py`](../figures/make_runtime_coverage.py) | [`navigation_runs.csv`](../results/navigation_runs.csv), [`navigation_summary.json`](../results/navigation_summary.json), [`collisions.json`](../results/collisions.json), [`runtime_coverage.json`](../results/runtime_coverage.json), [`temporal_correlation.json`](../results/temporal_correlation.json) |

## Figures and tables

All generators write to `logs/thesis/figures/`; `bash figures/regenerate.sh`
runs them in order.

| Thesis figure | Generator | Needs private data |
| --- | --- | :---: |
| 1 · warehouse setup | [`make_thesis_setup.py`](../figures/make_thesis_setup.py) | no |
| 2 · measurement chain | [`make_measurement_chain.py`](../figures/make_measurement_chain.py) | yes |
| 4 · recorded drive | [`make_problem_statement_drive.py`](../figures/make_problem_statement_drive.py) | yes |
| 6 · data partitions | [`make_data_roles.py`](../figures/make_data_roles.py) | yes |
| 7 · covariance field | [`make_field_construction.py`](../figures/make_field_construction.py) | yes |
| 8 · correction | [`make_correction.py`](../figures/make_correction.py) | yes |
| 9 · corrected run | [`make_navigation_single_run.py`](../figures/make_navigation_single_run.py) | yes |
| 10, 15–18 · dropout tasks | [`make_removal_mechanism.py TASK`](../figures/make_removal_mechanism.py) | yes |
| 11 · workspace map | [`make_driveable_map.py`](../figures/make_driveable_map.py) | no |
| 12 · camera views | [`make_camera_views.py`](../figures/make_camera_views.py) | yes |
| 13–14 · error distribution and correlation | [`make_error_structure.py`](../figures/make_error_structure.py) | yes |
| Tables I–III and in-text numbers | [`make_dropins.py`](../figures/make_dropins.py) → `dropins.tex` | yes |

The ray-basis and method-flow diagrams are TikZ in the thesis source and have
no generator here.
