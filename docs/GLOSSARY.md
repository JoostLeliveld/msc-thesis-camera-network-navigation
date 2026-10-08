# Glossary

Terms as the thesis uses them, with the name they have in the code where it
differs.

## Models and conditions

| Term | Meaning | In code |
| --- | --- | --- |
| **R** | Measurement covariance of one corrected camera position (2×2, m²) | `R`, `ray_r_*` |
| **R0** | One covariance for all cameras and positions | `global`, `R0_global_full` |
| **R1** | One covariance per camera | `per_camera`, `R1_per_camera_full` |
| **R2** | Covariance per camera and floor position, from the 16 nearest fitted positions | `spatial`, `R2_spatial_residual` |
| **R_proj** | Baseline: pixel noise pushed through the projection Jacobian | `rproj` |
| **intact / removal** | All five cameras, or one camera switched off for the whole run | `state` column; `*_intact`, `*_removal` conditions |
| **condition** | Covariance model × network state, e.g. `spatial_removal` | `condition` column in `results/navigation_runs.csv` |
| **seed** | Noise seed of a campaign repetition: 91500, 91501, 91502 | `seed` |

## Data roles

All headings, repetitions and cameras at one physical position stay in the same
role.

| Thesis | Code | Used for |
| --- | --- | --- |
| D_μ | `D_mu` | Fitting the bias correction |
| D_R | `D_R` | Fitting R0, R1, R2 |
| D_val | **`D_dev`** | Sensitivity checks of the fixed R2 constants |
| D_test | `final_audit` | The single held-out evaluation |

## Measurement chain

| Term | Meaning |
| --- | --- |
| **image point** | Bottom of the detected robot (mask bottom, else box bottom) |
| **raw position** | Image point projected onto the floor along the camera ray |
| **admission gate** | Detector confidence ≥ 0.25 and a valid floor projection; nothing else ([`config/sensor_gate.yaml`](../config/sensor_gate.yaml)) |
| **bias correction** | Network predicting the along-ray and across-ray offset of the raw position |
| **along-ray / across-ray** | Floor axes parallel and perpendicular to the camera's viewing direction; R is fitted in this frame |
| **visibility patch** | 16×16 representation of the detected crop, second input of the correction network (`visibility_patch` observation model) |
| **residual** | Corrected position minus reference position |
| **reference pose** | Ground-truth pose of the robot during data capture |
| **NIS** | Normalised innovation squared, `νᵀ S⁻¹ ν`; used as a gate in the EKF and as a calibration check |

## Estimation and planning

| Term | Meaning |
| --- | --- |
| **belief** | EKF state estimate: mean and covariance of the robot pose |
| **Q** | Process noise, set from the injected encoder noise ([`PROCESS_NOISE.md`](PROCESS_NOISE.md)) |
| **fusion** | Inverse-covariance weighting of camera positions from the same detector round |
| **EFE** | Expected free energy: the planner's route cost |
| **risk** | EFE term: divergence of the predicted position from the goal prior |
| **ambiguity** | EFE term: expected uncertainty after the camera updates along the route |
| **information field** | Map over the floor of the camera precision `Σᵢ Rᵢ(p)⁻¹` that the planner reads |
| **route seed** | Hand-specified aisle route used to start the optimiser (`pipeline/tasks.yaml`) |
| **lockstep** | Gazebo is paused and stepped only after every node has reported, so slow detection cannot desynchronise simulated time |
| **collision** | The robot footprint leaves the driveable region; scored offline (`pipeline/score_collisions.py`) |

## Tasks

Task identifiers were frozen before cameras B and C were swapped. **The letter
in the identifier is not always the removed camera.**

| Task identifier | Camera removed |
| --- | --- |
| `thesis10_camera_a_western_dock_detour` | A |
| `thesis10_camera_b_cross_warehouse_detour` | **C** |
| `thesis10_camera_c_inner_warehouse_detour` | **B** |
| `thesis10_camera_e_eastern_detour` | **D** |
| `thesis10_camera_e_long_cross_warehouse_detour` | E |

The authoritative source is `removed_camera_id` in
[`pipeline/execution_template.yaml`](../pipeline/execution_template.yaml).
