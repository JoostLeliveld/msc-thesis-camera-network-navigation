# Thesis figures and tables

From the repository root, `bash figures/regenerate.sh` generates all 16 PDF/PNG
figures below and `dropins.tex` under `logs/thesis/figures/`. It requires the
recorded-evidence bundle and the analyses in [REPRODUCING.md](../docs/REPRODUCING.md).
The script stops on the first failed generator. The ray-basis and method-flow
TikZ diagrams are maintained with the thesis LaTeX source.

| Thesis figure | Generator | Main evidence |
| --- | --- | --- |
| 1: warehouse setup | `make_thesis_setup.py` | Bundled plan-view image and world camera poses |
| 2: measurement chain | `make_measurement_chain.py` | Capture images, admitted detector observations and world geometry |
| 4: recorded drive | `make_problem_statement_drive.py` | Recorded A-west spatial-model run |
| 6: data partitions | `make_data_roles.py` | Frozen capture/partition loader |
| 7: covariance field | `make_field_construction.py` | Residuals, fitted covariances and planning fields |
| 8: correction | `make_correction.py` | Held-out corrected positions and report |
| 9: corrected run | `make_navigation_single_run.py` | Recorded A-west spatial-model run |
| 10, 15–18: dropout tasks | `make_removal_mechanism.py TASK` | Planned routes, executed poses and collision scores |
| 11: workspace map | `make_driveable_map.py` | World geometry and driveable regions |
| 12: camera views | `make_camera_views.py` | Recorded camera images |
| 13–14: error distribution/correlation | `make_error_structure.py` | Held-out NIS and runtime correlation reports |
| Tables I–III and numerical checks | `make_dropins.py` | Held-out reports, campaign summary and runtime coverage |

`make_runtime_coverage.py` writes a JSON diagnostic rather than a figure; it runs
before table generation. `regenerate.sh` explicitly visits all five tasks rather
than relying on `make_removal_mechanism.py`'s single-task default.

The dropout colour bar is half the trace of the summed camera precision,
`0.5 tr sum_i R_i(p)^-1`, in m^-2. The correlation plot uses the Euclidean
separation of the ground-truth positions, rather than accumulated travel distance.
Neither plotting correction changes the recorded observations or outcomes.

`dropins.tex` contains table **bodies** and comments identifying evaluation
populations; it is not a standalone LaTeX document and does not edit the thesis.
Primary correction numbers use D_test. Additional D_val (`D_dev` in code)
statistics are labelled separately.
