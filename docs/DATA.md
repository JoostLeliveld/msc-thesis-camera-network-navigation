# Data and model access

The public repository includes code and aggregate results. Camera images,
trained weights and full campaign logs are available from Joost Leliveld:
[j.j.p.leliveld@student.tue.nl](mailto:j.j.p.leliveld@student.tue.nl).
Request the **recorded-evidence bundle** to inspect/reanalyse the thesis or the
**refitting-input bundle** for a new fit and campaign. There is no public download
URL for these bundles. Extract into a separate checkout with paths preserved.

## Recorded-evidence bundle

| Path below the checkout | Purpose |
| --- | --- |
| `logs/thesis/captures/` | Images, indices, pose plans, partition files and capture provenance; includes the versioned v5/v8/v9/v10/v11 dependencies of `pipeline/dataset.py` |
| `logs/thesis/detector/training/imgsz960/upper_finetune/weights/best.pt` | Frozen YOLO11n detector |
| `logs/thesis/fits/` | Correction, residuals, covariance, validation evaluation, runtime models and planning fields |
| `logs/thesis/final_audit_protocol.json` | Original locked held-out evaluation inputs |
| `logs/thesis/final_audit/` | Held-out records, report and completion evidence |
| `logs/thesis/final_audit_fusion/` | Held-out multi-camera fusion records and report |
| `logs/thesis/final_audit_corrected_xy.npz` | Corrected held-out positions for figures; can be reconstructed |
| `logs/thesis/final_campaign/routes/` | Solved routes, solver manifests and follower replay report |
| `logs/thesis/final_campaign/campaign_configs/` | Original per-seed execution settings |
| `logs/thesis/final_campaign/campaign/` | Three seed ledgers, source snapshots, per-run CSVs, summaries and event records |
| `logs/thesis/analysis/`, `logs/thesis/final_campaign/analysis/` | Derived diagnostics and campaign aggregates |

The refitting-input bundle contains the captures and detector checkpoint, with their
supporting provenance. The setup image is included in `figures/assets/`. Do not include the recorded `fits/`,
audits or campaign output in a fresh-refit checkout.

## Integrity and provenance

`pipeline/dataset_lock.json` defines the final physical-position partition and
records input SHA-256 values. Historical capture directory versions are inputs
to that one dataset, not competing final experiments. All headings and camera
views at one physical position share a partition.

The original campaign used source commits recorded in
`results/navigation_summary.json`; per-seed source snapshots are supplied with
the full evidence. Those development-repository commits need not exist in this
public repository's history. Later submission fixes do not retroactively alter
the experiment identity. New experiments record their own commit and hashes.

Runtime manifests may retain the original machine's absolute paths. Readers
relocate known `logs/thesis/` references and validate model hashes instead of
editing those frozen manifests. Keep the complete directory layout.

Some event-delivery JSONL files are losslessly compressed with Zstandard. The
standard analyses use the CSV/summary records; `zstd -d FILE.jsonl.zst` restores
an event stream if required. Keep originals when investigating provenance.
