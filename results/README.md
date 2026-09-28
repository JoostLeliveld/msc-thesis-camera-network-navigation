# Recorded thesis results

These files are copies of the aggregate reports from the final thesis evidence,
not results of a new simulation run. The navigation campaign has 90 valid runs,
15 per condition. Raw images, model weights and per-frame logs are supplied on
request; see [DATA.md](../docs/DATA.md).

| Condition | Successful runs | Mean belief error (cm) |
| --- | ---: | ---: |
| Global, intact | 14/15 | 6.81 |
| Global, dropout | 11/15 | 15.86 |
| Per-camera, intact | 14/15 | 4.49 |
| Per-camera, dropout | 7/15 | 20.01 |
| Spatial, intact | 15/15 | 3.14 |
| Spatial, dropout | 15/15 | 3.10 |

On held-out camera batches, spatial fusion reduces RMSE by 34.7% relative to
equal weighting. The correction reduces position-balanced held-out RMSE from
36.72 to 5.36 cm. These are different evaluation populations from the closed-loop
navigation campaign.

| File | Original path below `logs/thesis/` |
| --- | --- |
| `localisation.json` | `final_audit/report.json` |
| `fusion.json` | `final_audit_fusion/report.json` |
| `navigation_summary.json` | `final_campaign/analysis/summary.json` |
| `navigation_runs.csv` | `final_campaign/analysis/runs.csv` |
| `runtime_coverage.json` | `final_campaign/analysis/runtime_coverage.json` |
| `temporal_correlation.json` | `final_campaign/analysis/temporal_correlation.json` |
| `error_distribution.json` | `analysis/error_distribution.json` |
| `collisions.json` | `final_campaign/analysis/collisions.json` |

Verify these distributed files with:

```bash
(cd results && sha256sum -c SHA256SUMS)
```

Keep the original `logs/thesis/` layout when using the full evidence bundle; the
pipeline does not consume this public summary directory as a substitute for
raw evidence. JSON provenance fields, including original paths and historical
code hashes, are preserved. `D_dev` in older artifacts is the thesis validation
partition; `final_audit` is its held-out test partition.

Table III's fused error is RMSE pooled over fusion decisions; belief error and
belief sigma are means over runs. Containment uses pooled errors. The temporal
analysis records both time-gap bins and displacement bins; displacement is the
Euclidean separation of the two ground-truth positions, not accumulated path
length. The first distance bin is below 0.05 m.
