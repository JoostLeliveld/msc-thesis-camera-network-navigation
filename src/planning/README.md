# Planning

This package contains the existing IWAI expected-free-energy planner and its belief rollout.
The thesis keeps this planner fixed. Commissioning changes only the future camera model
supplied to the rollout as the inverse of the matched runtime covariance
`R_i(p, psi)` of the selected correction.

Process covariance `Q` is a frozen configuration input. It is not estimated, compared, or
selected by the thesis pipeline.

Primary files:

- `planning/nodes/unicycle_planner_node.py`: ROS wrapper and belief loop;
- `planning/planners/base_planner.py`: route optimization;
- `planning/core/casadi_efe.py`: EFE objective;
- `planning/core/dynamics.py`: frozen unicycle process model;
- `planning/core/visibility_gp_map.py`: planner-facing commissioned field adapter.

Final run comparisons must follow the reporting rules in `docs/METHOD.md` §11.
