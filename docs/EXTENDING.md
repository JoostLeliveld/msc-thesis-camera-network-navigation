# Extending this work

Starting points for the changes a follow-up project is most likely to make.
Read the [code tour](CODE_TOUR.md) first. After any change, run
`python3 -m pytest -q`; the tests encode most of the contracts listed here.

## Run something first

```bash
# Code-only: no simulator, no private data
python3 -m pytest -q
(cd results && sha256sum -c SHA256SUMS)
python3 figures/make_driveable_map.py        # writes logs/thesis/figures/driveable_map.png
```

With the ROS workspace built (see the main [README](../README.md#setup)),
`ros2 launch experiments warehouse_primary_comparison.launch.py --show-args`
lists every runtime argument with its description.

## Add, move or remove a camera

The five cameras are named `camera_A` … `camera_E` throughout. To change the
network:

1. **World.** Add or move the `external_camera*` include in
   [`warehouse_v2.world.sdf`](../src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf)
   and add a model under `src/sim/models/` with its own image topic. Bridge the
   topic in [`bringup_sim.launch.py`](../src/sim/launch/bringup_sim.launch.py).
2. **Detector.** Extend `CAMERA_TOPICS` in
   [`batched_four_camera_yolo_node.py`](../src/perception/perception/nodes/batched_four_camera_yolo_node.py)
   and `CAMERA_ORDER` in [`four_camera_batch.py`](../src/perception/perception/core/four_camera_batch.py).
3. **Offline scripts.** Several scripts define
   `CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")`
   (`git grep -n "CAMERAS ="` lists them all).
4. **Data and models.** A new or moved camera has no correction or covariance
   yet. Capture reference poses for it, then rerun `pipeline/refit.sh` (see
   [REPRODUCING](REPRODUCING.md)). R2 is local, so it is only valid where the
   camera was captured.

To drop a camera in a run without changing the world, use the
`removed_camera_id`, `manager_camera_ids` and `camera_network_active_camera_ids`
overrides, as the dropout conditions in
[`pipeline/execution_template.yaml`](../pipeline/execution_template.yaml) do.

## Change the warehouse or the tasks

- Geometry: [`world/warehouse_v2.py`](../world/warehouse_v2.py) describes the
  layout. Keep-out regions and the driveable region are in
  [`src/experiments/config/world_profiles.yaml`](../src/experiments/config/world_profiles.yaml).
  Collision scoring uses that driveable region, so update both.
- [`world/world_freeze_manifest.json`](../world/world_freeze_manifest.json)
  records the hash of the world used for data capture. Recorded data are only
  valid for that world: a changed world needs a new capture and refit.
- Tasks and their route seeds are in [`pipeline/tasks.yaml`](../pipeline/tasks.yaml);
  per-task dropout settings are in the two `pipeline/*_template.yaml` files.
  [`world/route_tasks.py`](../world/route_tasks.py) checks that a task offers
  genuinely different routes before it is used.

## Change the measurement model

- New correction network: train in [`pipeline/fit_correction.py`](../pipeline/fit_correction.py)
  and load it in `CommissionedVisibilitySensorModel` in
  [`commissioned_visibility.py`](../src/reliability/reliability/commissioned_visibility.py).
- New covariance model: add it next to R0/R1/R2 in
  [`pipeline/fit_covariance.py`](../pipeline/fit_covariance.py), package it in
  [`pipeline/package_runtime.py`](../pipeline/package_runtime.py), and add a
  branch in `_ray_r_covariance`.
- Evaluate on `D_val` (`D_dev` in code) with [`pipeline/evaluate_ddev.py`](../pipeline/evaluate_ddev.py)
  while you develop. `D_test` (`final_audit`) is used once, at the end.

## Change the planner

- The objective is in [`casadi_efe.py`](../src/planning/planning/core/casadi_efe.py)
  and is described in [PLANNER](PLANNER.md).
- Constants used in the thesis are code defaults. Overriding a locked value
  prints `overrides the locked value`. Read the comment next to that value in
  [`base_planner.py`](../src/planning/planning/planners/base_planner.py)
  before changing it: several of them record a failure the value prevents.
- Compare against the shortest-path baseline with
  `planner:=geometric_shortest_path`.

## Known limitations

- The simulator renders all cameras on one CPU thread. Camera updates are
  limited by rendering, not by the detector, and a campaign takes several hours.
- Lockstep runs are repeatable in their inputs but not bitwise deterministic.
- Corrections and covariances are fitted for this warehouse and these camera
  poses. The aim is to calibrate a fixed site fully, not to generalise to new
  sites.
- The recorded images, weights and logs are not public; see [DATA](DATA.md).
