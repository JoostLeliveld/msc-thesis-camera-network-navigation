# planning

Belief-space route planner and the robot's EKF.

- `planning/nodes/unicycle_planner_node.py`: EKF on encoder odometry and fused camera
  measurements, and the planner interface.
- `planning/nodes/efe_agent_node.py`: runtime node used in the campaign. It extends the
  planner node with the waypoint tracker that follows the route from the belief.
- `planning/planners/base_planner.py`: route optimisation from the route initialisations.
- `planning/core/casadi_efe.py`: expected-free-energy objective (risk, ambiguity, no-go
  penalty).
- `planning/core/camera_network.py`: expected camera information from the covariance
  model.
- `planning/core/belief_correction.py`: EKF camera update with the NIS gate.
- `planning/core/encoder_noise_model.py`: process noise derived from the encoder model
  (see `docs/PROCESS_NOISE.md`).
- `planning/core/nogo_cost.py`: no-go penalty at the belief sigma points.

The objective is described in `docs/PLANNER.md`.
