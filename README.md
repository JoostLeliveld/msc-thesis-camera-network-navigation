# Camera-network belief-space navigation

Code and reproducibility pipeline for the thesis *Camera-Network Modelling for
Belief-Space Robot Navigation*.

Five fixed cameras localize a mobile robot in a simulated warehouse. The
pipeline learns a camera-aware position correction and compares global,
per-camera and spatial residual-covariance models. The same covariance is used
for multi-camera fusion, EKF updates and expected-free-energy route planning.

## Final result

The final experiment contains 90 evidence-valid runs: five tasks, three matched
seeds, three covariance models and two camera-network states. With one camera
dropped, the spatial model succeeds in 15/15 runs, compared with 11/15 for the
global model and 7/15 for the per-camera model.

Machine-readable results are generated in:

```text
logs/thesis/final_campaign/analysis/
  runs.csv
  summary.json
  collisions.json
  runtime_coverage.json
  temporal_correlation.json
```

Raw data, fitted artifacts and run logs are intentionally excluded from Git.
Their exact paths and hashes are recorded in `pipeline/dataset_lock.json` and
the campaign manifest.

## Repository layout

```text
config/      Runtime sensor-gate configuration
docs/        Method, planner and process-noise specifications
figures/     Generators for thesis figures
pipeline/    Dataset, fitting, route, campaign and analysis pipeline
src/         ROS 2 packages and Gazebo simulation
tests/       Unit, integration and reproducibility tests
world/       Canonical warehouse geometry and route utilities
```

Versioned identifiers that remain in frozen data paths are provenance labels,
not alternative active methods.

## Environment

- Ubuntu with ROS 2 Humble
- Gazebo/Ignition used by the ROS packages
- Python dependencies from `requirements.txt`

Build and load the workspace:

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Reproduce the thesis pipeline

The stages are intentionally linear:

```bash
# 1. Validate the frozen dataset.
python3 pipeline/audit_dataset.py

# 2. Refit correction and covariance artifacts.
bash pipeline/refit.sh

# 3. Recreate the detector-backed held-out audit in a new output directory.
python3 pipeline/final_audit.py \
  --protocol logs/thesis/final_audit_protocol.json \
  --output logs/thesis/reproduction/final_audit

# 4. Recreate its fusion audit.
python3 pipeline/final_audit_fusion.py \
  --audit logs/thesis/reproduction/final_audit \
  --runtime-root logs/thesis/fits/runtime_r012 \
  --rproj-ddev logs/thesis/fits/ddev_evaluation/manifest.json \
  --output logs/thesis/reproduction/final_audit_fusion

# 5. Solve and bind the 30 routes.
bash pipeline/routes.sh

# 6. Only when intentionally rerunning the complete campaign.
bash pipeline/campaign.sh

# 7. Analyze the final campaign.
python3 pipeline/analyze_campaign.py
python3 pipeline/analyze_temporal_correlation.py
python3 figures/make_runtime_coverage.py
```

Figure generators in `figures/` read only the frozen thesis artifacts.

## Verification

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 -m pytest -q
```

Before launching Gazebo or a campaign, confirm that no other simulator is
running:

```bash
pgrep -af "ros2 launch|ign gazebo|campaign_runner"
```

## Authoritative documentation

1. `docs/METHOD.md` - final scientific and experimental method.
2. `docs/PLANNER.md` - implemented planner objective.
3. `docs/PROCESS_NOISE.md` - encoder-derived process noise.
4. `docs/STATE.md` - final artifacts, results and repository state.

This repository is a history-free snapshot of the code used for the final
thesis campaign.

## Related publication

The earlier IWAI 2026 study and its separate code snapshot are available at
<https://github.com/JoostLeliveld/iwai2026-camera-reliability-efe>. That paper
reports a separate 40-run experiment and is not an additional arm of the thesis
campaign.

## Licence

Original project code is available under the MIT License. ROS packages that
declare Apache-2.0 in their package metadata remain under Apache-2.0. See
[`LICENSES/README.md`](LICENSES/README.md).
