# Method

This document describes the method implemented in this repository and used for
the thesis experiments.

## 1. Scope

A mobile robot navigates a known simulated warehouse using wheel odometry and
position measurements from five fixed external cameras. The thesis tests
whether a camera- and position-dependent measurement covariance improves
multi-camera fusion and belief-space route planning, especially when one camera
is dropped.

The compared measurement models are:

- R0: one global residual covariance;
- R1: one residual covariance per camera; and
- R2: a camera- and position-dependent residual covariance.

All three conditions share the detector, correction, dynamics, process noise,
fusion rule, EKF, planner, route initializations, controller, tasks and seeds.

## 2. Data and partitions

Camera observations are paired with recorded reference poses in the simulated
warehouse. A complete physical position is the independent partition unit: all
headings, repetitions and cameras at that position stay in one role.

The final split contains:

| Role | Purpose | Positions |
| --- | --- | ---: |
| D_mu | fit the systematic-displacement correction | 1,326 |
| D_R | fit R0, R1 and R2 | 1,333 |
| D_val (`D_dev` in the code) | sensitivity checks of the fixed R2 constants | 335 |
| D_test | one held-out evaluation | 168 |

Exact source paths, hashes, exclusions and opportunity counts are frozen in
`pipeline/dataset_lock.json`. Missing detections and deterministic gate
refusals remain in opportunity denominators.

## 3. Camera observation

Each camera is geometrically calibrated. The admitted image reference point is
the bottom centre of the frozen detector box. The inverse ground-plane
homography maps it to a raw position measurement.

The runtime sensor gate is deterministic and belief-independent:

1. detector confidence reaches 0.25; and
2. the bottom centre has a valid ground-plane projection.

No ground truth, innovation, NIS, box-size threshold or image-edge threshold is
used by the gate.

## 4. Systematic-displacement correction

One shared neural correction is trained on D_mu. Its structured inputs describe
camera identity, camera-to-measurement geometry, detector-box geometry and
confidence. A 16 by 16 visibility representation from the detected crop supplies
the second input branch.

The network predicts an along-ray and across-ray correction. It is trained
deterministically on CPU with position-balanced SmoothL1 loss. Once fitted, it
is frozen before covariance fitting or evaluation.

## 5. Residual covariance

Covariances are fitted to corrected residuals on D_R. Repeated measurements at
one physical position are first reduced to one second moment, so every position
has equal aggregate weight.

All models use the same inverse-Wishart posterior mean with:

- prior covariance `100 I m2`;
- prior strength `2.5e-6`; and
- eigenvalue floor `1e-6 m2`.

R0 pools every camera and position. R1 pools positions separately for each
camera. R2 uses the 16 nearest distinct positions of that camera with a 0.4 m
Gaussian distance kernel. With little local support, R2 approaches the broad
prior.

The ray-frame covariance is rotated into the world frame at the runtime query
position. The same resulting covariance is used in fusion, the EKF and route
planning.

## 6. Multi-camera fusion

Every admitted measurement is corrected before fusion. For a synchronous batch
of conditionally independent corrected measurements `z_i` with covariances
`R_i`, fusion is:

```text
R_f^-1 = sum_i R_i^-1
z_f    = R_f sum_i R_i^-1 z_i
```

A physical camera frame contributes at most once. No median disagreement gate
or sequential cascade is used.

## 7. State estimation

The robot belief is a Gaussian over planar position and heading. It starts from
the declared task pose with covariance:

```text
diag(0.10^2, 0.10^2, (15 deg)^2)
```

Timestamped encoder velocity propagates the unicycle EKF. The input-dependent
process covariance is set from the simulated encoder model and is identical in
the estimator and planner; see `docs/PROCESS_NOISE.md`.

The fused camera position is accepted when its two-dimensional NIS is at most
9.21. Accepted updates use Joseph-form covariance correction. Cameras do not
observe heading directly; heading changes only through position-heading
cross-covariance.

## 8. Route planning

The planner predicts belief mean and covariance for every candidate control
sequence. Expected camera information is evaluated at five planar sigma points
of the predicted position belief. Active-camera information matrices are added,
and camera dropout excludes the affected camera from this sum.

The route objective is the arrival-gated, discounted mean of:

- Gaussian goal risk;
- expected camera ambiguity; and
- the obstacle penalty evaluated at the same belief sigma points.

The goal-prior standard deviation decreases smoothly from 5.0 m to 0.10 m.
The obstacle penalty is zero beyond a 0.05 m warning band and rises
quadratically near and inside obstacles. The planner also enforces exact
swept-footprint feasibility with the 0.80 by 0.55 m rectangular robot body.

One global route is optimized before execution from the same task-specific
initializations in all experimental arms. The route is not replanned online.
See `docs/PLANNER.md` for the implemented objective and parameters.

## 9. Route execution

The shared `ff_fb` waypoint follower combines route-tangent feedforward
with heading and cross-track feedback, corner preview and final-segment braking.
It tracks from the current belief, never from ground truth.

A run stops when the belief mean remains within 0.10 m of the goal for 2 s.
Ground truth does not stop the robot.

## 10. Navigation experiment

The experiment crosses:

- three covariance models: R0, R1 and R2;
- two network states: intact and one task-specific camera dropped;
- five predeclared tasks; and
- three matched noise seeds.

This gives 90 runs. The active tasks and route initializations are defined in
`pipeline/tasks.yaml`. Each dropout applies to both route planning and runtime
fusion for the complete run.

## 11. Outcome and evidence rules

A run is successful only when:

1. required evidence is complete;
2. final true goal distance is below 0.30 m; and
3. the offline swept-footprint audit finds no departure from the driveable
   region.

Collision scoring uses ground-truth poses only after execution. The simulator
does not provide a collision signal to the controller.

The reported navigation metrics are:

- success by matched task and seed;
- fused camera error at fusion timestamps;
- belief error at belief timestamps;
- planar belief major-axis sigma;
- missed accepted-update fraction;
- route changes between matched intact and dropout conditions; and
- fused and belief NIS/containment.

The analysis reads only the final campaign folder and stops on incomplete or
inconsistent run evidence.

## 12. Use of ground truth

Ground truth is used to:

- provide offline correction targets;
- score localization and navigation after execution; and
- support offline collision auditing.

Ground truth is not used in:

- detector admission;
- runtime correction or covariance query;
- fusion or estimator updates;
- route planning or tracking;
- goal detection; or
- runtime collision decisions.

## 13. Known limitations

- Evaluation covers one simulated installation and five tasks.
- Residuals are heavy-tailed and temporally correlated.
- Camera availability is not modelled separately from covariance conditional on
  an admitted observation.
- Position-only updates can rotate the estimated heading through
  position-heading cross-covariance. Near image, visibility or map edges, a
  biased anisotropic position update may therefore affect tracking.
- Routes are selected from finite initializations and are not replanned online.
