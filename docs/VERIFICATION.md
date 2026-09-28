# Submission verification

Checks performed on 28 September 2026 against the submission source. See
[ENVIRONMENT.md](ENVIRONMENT.md) for installed package versions. Generated
analysis and figures are kept outside tracked source; original thesis evidence
is not rewritten.

- Code-only suite: 1,602 passed, 16 skipped. The skips cover inputs/components
  absent from a code-only clone. The suite includes relocation tests that reject
  a changed model hash and do not fall back to an existing original-machine file.
- Python 3.10 dependency resolution: the complete set in `requirements-lock.txt`
  resolves successfully with pip. This was a dry run, not a clean OS installation.
- Relocated held-out evidence: all 1,304 corrected observations at 165 positions
  reconstruct exactly; every stored array matches the original artifact.
- Held-out error distribution: the regenerated JSON equals the archived report.
- Full campaign reanalysis: all 90 run rows, condition aggregates and collision
  reports exactly match the archive after rescoring 747,058 ground-truth poses.
- Runtime containment: the regenerated JSON equals the archived report.
- Between-frame correlation: the regenerated time-gap and distance-bin JSON
  equals the archived report, using 21,087 frames across 90 runs.
- Configuration preparation: generated campaign configurations and bound
  all three per-seed execution configurations to the recorded solved routes
  in a separate checkout.
- ROS workspace build: all eight packages build with the documented colcon command.
- Campaign input validation: 94 files including all 30 solved routes and the
  passing archived follower-replay report.
- Figure regeneration: the complete script produces 16 PDF/PNG figures and
  table rows, including all five dropout tasks. Representative plots and the
  recovered setup image were visually inspected.
- Documentation links, Python/shell syntax and distributed-result checksums pass.

The submission cleanup does not rerun detector training, correction fitting,
the route optimiser or the 90-run Gazebo campaign. Those are separate workloads
in [REPRODUCING.md](REPRODUCING.md). The recorded campaign remains the source of
the reported scientific results.
