# Reproducing the thesis results

Run all commands at the repository root, after completing the root README's
setup. Keep the recorded thesis evidence and a new experiment in **separate
checkouts**. Do not overwrite the original campaign manifests or relabel new
runs as the recorded thesis experiment.

## 1. Analyse the recorded thesis evidence

Obtain and extract the **recorded-evidence bundle** described in [DATA.md](DATA.md)
so that `logs/thesis/` exists immediately below the repository root. Work with a
copy: analysis commands write derived outputs, while source logs and their
provenance stay unchanged. Activate the Python environment; source the ROS
setup if the installed geometry libraries require it.

```bash
source /opt/ros/humble/setup.bash
source .venv/bin/activate
python3 pipeline/analyze_campaign.py
python3 pipeline/analyze_error_distribution.py
python3 pipeline/analyze_temporal_correlation.py
python3 figures/make_runtime_coverage.py
python3 pipeline/final_audit_corrected_xy.py
bash figures/regenerate.sh
```

`analyze_campaign.py` already performs the swept-footprint collision audit.
It checks every recorded pose and can take substantially longer than the other
analysis commands; the script prints progress as run scores become available.
`score_collisions.py` is an optional per-run diagnostic that requires explicit
run directories; it is not a separate argument-free campaign step.

Outputs:

| Path below `logs/thesis/` | Contents |
| --- | --- |
| `final_campaign/analysis/runs.csv` | One row for each of 90 task/model/state/seed combinations |
| `final_campaign/analysis/summary.json` | Condition aggregates and matched comparisons |
| `final_campaign/analysis/collisions.json` | Swept-footprint collision evidence |
| `final_campaign/analysis/runtime_coverage.json` | Fused-measurement and belief containment |
| `final_campaign/analysis/temporal_correlation.json` | Time-gap and distance-bin error correlations |
| `analysis/error_distribution.json` | Held-out residual and cross-camera diagnostics |
| `final_audit_corrected_xy.npz` | Reconstructed observations, checked against recorded errors |
| `figures/` | PDFs, PNGs and `dropins.tex` table rows |

The runtime model reader maps historical `logs/thesis/...` paths into the
current checkout and still verifies model-file hashes. The campaign analysis
similarly relocates ledger run paths. Original JSON provenance is not rewritten.
Other absolute paths are not implicitly relocated. Compare outputs with the
included [aggregate results](../results/README.md); plotting-library versions
may change figure bytes without changing the numbers.

The ray-basis and method-flow diagrams are TikZ figures in the thesis LaTeX
source; the code repository generates the other 16 PDF figures. See the
[figure inventory](../figures/README.md).

## 2. Refit from the frozen camera observations

Use a separate clean clone with the **refitting-input bundle**, not the recorded
fits/audits/campaign. The bundle includes the raw captures, their partition and
provenance files, and the frozen detector checkpoint. Install dependencies and
fetch simulator assets as in the README. Activate the Python environment.

```bash
python3 pipeline/audit_dataset.py
bash pipeline/refit.sh
python3 pipeline/audit_protocol.py
python3 pipeline/final_audit.py \
  --protocol logs/thesis/final_audit_protocol.json \
  --output logs/thesis/final_audit --device 0
python3 pipeline/final_audit_fusion.py \
  --audit logs/thesis/final_audit \
  --runtime-root logs/thesis/fits/runtime_r012 \
  --rproj-ddev logs/thesis/fits/ddev_evaluation/manifest.json \
  --output logs/thesis/final_audit_fusion
python3 pipeline/analyze_error_distribution.py
python3 pipeline/final_audit_corrected_xy.py
```

For CPU detector inference, use `DETECTOR_DEVICE=cpu bash pipeline/refit.sh` and
`--device cpu` for `final_audit.py`. This does not change the GPU setting in the
navigation templates. CPU and GPU inference are not promised to be bit-identical.
Correction fitting itself is deterministic CPU training. Do not tune the method
against the held-out audit results.

The protocol step freezes the hashes of the newly fitted inputs before the
held-out audit. It is deliberately absent from the recorded-evidence workflow:
regenerating an old protocol would destroy the distinction between original
and reproduced evidence.

## 3. Solve routes and execute a fresh campaign

Build/source the ROS workspace and ensure CUDA device 0 is available. The
tracked tree must be clean because the campaign manifest binds the code commit.
If experimenting with different settings, commit them in your own branch before
running and report the changed configuration.

```bash
source /opt/ros/humble/setup.bash
source .venv/bin/activate
source install/setup.bash
bash pipeline/routes.sh
bash pipeline/campaign.sh
```

`routes.sh` generates configuration from the tracked templates, solves 30 routes,
binds three per-seed execution configurations, and runs the follower replay
check. Everything is written under `logs/thesis/final_campaign/`. The replay must
pass for all 30 routes before campaign execution. `campaign.sh` verifies the
code/configuration manifest and executes the three seeds in sequence. Run only
one simulator at a time. Allow several hours; the exact time depends on the
machine. The script requires at least 6 GiB of free space before each seed, but
that lower bound excludes raw images, model fitting and setup files.

After execution, run the analysis/figure commands in section 1. Figure generation
uses the captured setup image in `figures/assets/gazebo_plan_view.png`,
included in this repository. A new image at
`logs/thesis/figures/gazebo_plan_view.png` overrides it. To capture another setup image, see
`figures/capture_plan_view.py` (requires the simulator).

## Recovery and interpretation

- A missing input is not a zero-observation result. Obtain the full corresponding
  bundle rather than creating empty directories to bypass an error.
- `refit.sh` skips completed output directories. A `.incomplete` directory marks
  a failed stage; inspect its log and move the failed output aside before retrying.
- Route solving skips task directories already present. To change fits/settings,
  use a fresh checkout/output tree instead of combining old routes with new models.
- `campaign.sh [SEED ...]` supports seeds 91500, 91501 and 91502. Its runner resumes
  completed work from the seed ledger. Hash or commit drift stops the campaign.
- Do not run `campaign.sh` on the recorded bundle: its historical commit and
  input identities intentionally differ from a new submission checkout.
- The thesis tests one simulated warehouse. Tests and reproduction checks do not
  establish hardware performance, generalisation or a failure rate outside it.
