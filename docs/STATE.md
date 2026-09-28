# Final thesis repository state

Status: complete and submission-ready as of 2026-09-28.

## Canonical experiment

The final campaign is rooted at:

```text
logs/thesis/final_campaign/
```

It contains 90 evidence-valid runs:

- five predeclared start-goal tasks;
- three matched noise seeds;
- global, per-camera and spatial covariance models; and
- intact and task-specific camera-dropout states.

The final analysis is:

```text
logs/thesis/final_campaign/analysis/
  runs.csv
  summary.json
  collisions.json
  runtime_coverage.json
  temporal_correlation.json
  trajectories.png
```

## Final results

| Condition | Success | Fused RMSE | Belief error | Belief sigma | Missed updates |
| --- | ---: | ---: | ---: | ---: | ---: |
| Global, intact | 14/15 | 5.32 cm | 6.81 cm | 5.11 cm | 8.5% |
| Global, dropout | 11/15 | 6.75 cm | 15.86 cm | 53.69 cm | 18.4% |
| Per-camera, intact | 14/15 | 5.86 cm | 4.49 cm | 4.71 cm | 6.2% |
| Per-camera, dropout | 7/15 | 10.66 cm | 20.01 cm | 47.15 cm | 26.8% |
| Spatial, intact | 15/15 | 4.31 cm | 3.14 cm | 4.19 cm | 4.1% |
| Spatial, dropout | 15/15 | 4.58 cm | 3.10 cm | 3.66 cm | 5.7% |

The swept-footprint audit finds 14 departures: five global, nine per-camera and
zero spatial. The spatial model changes route in 12/15 matched dropout pairs,
covering four of the five tasks. The two constant models change route in 0/15.

## Held-out localization

- Correction position-balanced RMSE: 36.72 cm to 5.36 cm on the held-out test
  partition used by the thesis.
- Spatial fusion RMSE: 2.69 cm, 34.7% below equal weighting.
- The geometric baseline uses a fitted pixel scale of 2.931 px.

## Frozen inputs

- Dataset lock: `pipeline/dataset_lock.json`
- Dataset partitions: complete physical positions
- Detector: frozen YOLO11n at 960 px input
- Covariance models: global R0, per-camera R1 and spatial R2
- Spatial constants: 16 neighbours, 0.4 m length scale and broad
  `100 I m2` prior with strength `2.5e-6`
- World: `src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf`
- Tasks: `pipeline/tasks.yaml`
- Execution controller: `ff_fb`
- Process noise: encoder-derived model in `docs/PROCESS_NOISE.md`

The version labels in frozen capture paths record the chronology of data
collection. They are not selectable alternatives.

## Reproduction entry points

| Stage | Command |
| --- | --- |
| Dataset audit | `python3 pipeline/audit_dataset.py` |
| Fit correction and covariance | `bash pipeline/refit.sh` |
| Held-out audit | `python3 pipeline/final_audit.py --protocol logs/thesis/final_audit_protocol.json --output <new-directory>` |
| Route solve | `bash pipeline/routes.sh` |
| Campaign rerun | `bash pipeline/campaign.sh` |
| Campaign analysis | `python3 pipeline/analyze_campaign.py` |
| Collision audit | `python3 pipeline/score_collisions.py` |

## Submission revisions

- `da37b327` - canonical final-campaign analysis
- `266f949f` - final analysis paths and retained obstacle penalty
- `9832cd6c` - measurement-chain notation aligned with the thesis

Rejected experiments, superseded tasks and development campaigns remain available in Git
history. They are intentionally absent from the active repository documentation.
