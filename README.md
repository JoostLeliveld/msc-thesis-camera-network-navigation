# Camera-network belief-space navigation

Code and aggregate results for Joost Leliveld's MSc thesis **Camera-Network
Modelling for Belief-Space Robot Navigation** (Eindhoven University of
Technology, 2026).

Five fixed cameras localise a mobile robot in a simulated warehouse. A learned
correction removes systematic error in the projected robot position. Global,
per-camera and spatial covariance models then weight camera fusion, EKF updates
and expected-free-energy route planning. The final campaign contains 90 runs:
five tasks, three covariance models, two network states and three noise seeds.

## Start here

- **Read the results:** [results/README.md](results/README.md) and the included
  aggregate JSON/CSV files require no simulator or external data.
- **Understand the method:** [METHOD](docs/METHOD.md), [PLANNER](docs/PLANNER.md)
  and [PROCESS_NOISE](docs/PROCESS_NOISE.md).
- **Install and run tests:** [Setup](#setup) below.
- **Reanalyse the recorded experiments or run a new campaign:**
  [Reproduction guide](docs/REPRODUCING.md).
- **Obtain the images, trained models and full logs:**
  [Data inventory and access](docs/DATA.md).

The code and aggregate results are public. The recorded images, trained weights
and full run logs are available from the author on request. A code-only clone
can run the unit tests but cannot reproduce training or navigation without
those inputs. The original experiment identities are preserved in the results;
the submission repository also contains later documentation, figure and
portability fixes.

## Repository layout

```text
config/      Sensor-admission configuration
docs/       Method, planner, process noise, data and reproduction guides
figures/     Thesis figure and table generators
pipeline/    Capture, fitting, audits, route solving, campaign and analysis
results/     Frozen aggregate thesis results and their checksums
src/         ROS 2 packages, each with a README, and Gazebo simulation
tests/      Unit and integration tests
world/       Warehouse geometry and route utilities
```

## Setup

Use **Ubuntu 22.04, Python 3.10, ROS 2 Humble and Gazebo Fortress**. Follow the [ROS 2 Humble installation guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
and [initialise rosdep](https://docs.ros.org/en/humble/Tutorials/Intermediate/Rosdep.html)
before running the commands below. The campaign templates use CUDA device `0`:
a compatible NVIDIA GPU/driver and a CUDA-enabled PyTorch build are needed for
the configured campaign. The unit tests and recorded-result analysis do not
require running Gazebo. Offline detector inference can also use CPU.

```bash
git clone https://github.com/JoostLeliveld/msc-thesis-camera-network-navigation.git
cd msc-thesis-camera-network-navigation
sudo apt-get install python3-venv python3-colcon-common-extensions python3-rosdep \
  ros-humble-ros-gz ros-humble-cv-bridge curl unzip zstd
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src --rosdistro humble -r -y

# The system packages expose Humble's Python bindings to this environment.
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python3 -m pip install -r requirements-dev.txt -c requirements-lock.txt
python3 -m pip check
```

The NumPy/OpenCV bounds keep the environment on NumPy 1.x for Humble's compiled
Python bindings. See [ENVIRONMENT.md](docs/ENVIRONMENT.md) for the review
machine's versions and the limits of reproducibility across environments.
`requirements-lock.txt` pins the resolved Linux/Python 3.10 dependency set;
`requirements.txt` and `requirements-dev.txt` declare the supported ranges.

For simulation, fetch third-party assets and build the ROS packages:

```bash
bash src/sim/fetch_external_models.sh
python3 -m colcon build --symlink-install --base-paths src
source install/setup.bash
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

Use `--base-paths src`: archived logs contain source snapshots that must not be
rediscovered as live ROS packages. Run commands from the repository root.

## Tests

```bash
python3 -m pytest -q
```

Tests requiring recorded data, fetched simulator assets or unavailable ROS
components skip when those inputs are absent. Passing code-only tests does not
constitute a full simulation rerun. Use a Git clone rather than a downloaded ZIP
because the path-portability check uses the Git file inventory.

## Reproducing results

The [reproduction guide](docs/REPRODUCING.md) gives separate, ordered commands
for analysing the frozen evidence and for fitting models/running a fresh
campaign. It also identifies expected outputs, hardware needs and failure
recovery. `bash figures/regenerate.sh` regenerates the figure set, including
all five dropout tasks, after the required evidence and analysis are present.

See [VERIFICATION.md](docs/VERIFICATION.md) for checks performed on this submission.

## Citation and contact

Use the GitHub **Cite this repository** entry or [CITATION.cff](CITATION.cff).
Record the commit hash used with `git rev-parse HEAD` when reporting a rerun.
For the thesis data bundle, contact Joost Leliveld at
[j.j.p.leliveld@student.tue.nl](mailto:j.j.p.leliveld@student.tue.nl), specifying
whether you need the recorded-evidence bundle or the refitting inputs.

The [earlier IWAI 2026 code](https://github.com/JoostLeliveld/iwai2026-camera-reliability-efe)
reports a separate experiment and is not the implementation of this thesis.

## Licence

Original code is MIT-licensed. ROS packages declaring Apache-2.0 retain that
licence. Third-party assets and dependencies retain their own terms; see
[LICENSES/README.md](LICENSES/README.md). Installing a dependency does not
relicense it under this repository's MIT licence.
