# Camera-network belief-space navigation

Code for the MSc thesis *Camera-Network Modelling for Belief-Space Robot
Navigation* (Eindhoven University of Technology, 2026).

Five fixed cameras localise a mobile robot in a simulated warehouse. The code
learns a camera-dependent correction of the projected robot position, fits
global, per-camera and spatial covariance models of the remaining error, and
uses the covariance for multi-camera fusion, EKF updates and
expected-free-energy route planning.

## Repository layout

```text
config/      Sensor-gate configuration used at runtime
docs/        Method, planner and process-noise descriptions
figures/     Generators for every thesis figure
pipeline/    Data collection, fitting, audits, route solving, campaign and analysis
src/         ROS 2 packages and the Gazebo simulation
tests/       Unit and integration tests
world/       Warehouse geometry and route utilities
```

Each package in `src/` has its own README. The method is described in
[`docs/METHOD.md`](docs/METHOD.md), the planner objective in
[`docs/PLANNER.md`](docs/PLANNER.md) and the process noise in
[`docs/PROCESS_NOISE.md`](docs/PROCESS_NOISE.md).

## Setup

Requirements: Ubuntu 22.04, ROS 2 Humble, Gazebo Fortress and the Python
packages in `requirements.txt`.

```bash
pip install -r requirements.txt
bash src/sim/fetch_external_models.sh   # floor/steel textures and crate mesh
source /opt/ros/humble/setup.bash
colcon build --symlink-install --base-paths src
source install/setup.bash
```

`--base-paths src` keeps colcon from picking up the source snapshots stored
with the run logs.

## Data

Recorded camera data, fitted models and run logs are not stored in Git. They
live under `logs/`, and `pipeline/dataset_lock.json` records the path and hash
of every frozen input. The recorded camera data and campaign logs are available
from the author on request.

With the data in place, the thesis results are in:

```text
logs/thesis/final_audit/               held-out correction and covariance audit
logs/thesis/final_audit_fusion/        held-out fusion audit
logs/thesis/analysis/                  error-distribution analysis
logs/thesis/final_campaign/analysis/   navigation campaign: runs.csv, summary.json,
                                       collisions.json, runtime_coverage.json,
                                       temporal_correlation.json
```

## Reproducing the results

```bash
# 1. Check the frozen dataset against the lock.
python3 pipeline/audit_dataset.py

# 2. Fit the correction and covariance models.
bash pipeline/refit.sh

# 3. Held-out audit of the correction and covariance models.
python3 pipeline/final_audit.py \
  --protocol logs/thesis/final_audit_protocol.json \
  --output logs/thesis/reproduction/final_audit

# 4. Held-out fusion audit.
python3 pipeline/final_audit_fusion.py \
  --audit logs/thesis/reproduction/final_audit \
  --runtime-root logs/thesis/fits/runtime_r012 \
  --rproj-ddev logs/thesis/fits/ddev_evaluation/manifest.json \
  --output logs/thesis/reproduction/final_audit_fusion
python3 pipeline/analyze_error_distribution.py

# 5. Solve the 30 routes (5 tasks x 3 models x 2 network states).
bash pipeline/routes.sh

# 6. Run the 90-run navigation campaign (several hours).
bash pipeline/campaign.sh

# 7. Analyse the campaign.
python3 pipeline/analyze_campaign.py
python3 pipeline/score_collisions.py
python3 pipeline/analyze_temporal_correlation.py
python3 figures/make_runtime_coverage.py

# 8. Regenerate the figures (written to logs/thesis/figures/).
for f in figures/make_*.py; do python3 "$f"; done
```

`figures/make_removal_mechanism.py` takes a task name, for example
`thesis10_camera_a_western_dock_detour`. Run only one simulator at a time.

## Tests

```bash
source install/setup.bash
python3 -m pytest -q
```

Tests that need the frozen data are skipped when `logs/` is absent.

## Related work

The earlier IWAI 2026 paper and its code are at
<https://github.com/JoostLeliveld/iwai2026-camera-reliability-efe>. It reports a
separate experiment.

## Licence

Original code is MIT-licensed. ROS packages that declare Apache-2.0 in their
package metadata remain under Apache-2.0. Third-party models keep their own
licences. See [`LICENSES/README.md`](LICENSES/README.md).
