# experiments

Launch file and nodes for one navigation run.

- `launch/warehouse_primary_comparison.launch.py`: launches the simulation, detector,
  camera manager, planner and logger for one run. `pipeline/campaign_runner.py` starts it
  once per task, model, network state and seed.
- `experiments/core/visibility_launch_common.py`: launch defaults and argument handling.
- `experiments/nodes/experiment_logger.py`: writes the run manifest and CSV logs.
- `experiments/nodes/goal_mission_node.py`: goal and belief-based stopping rule.
- `config/world_profiles.yaml`: world geometry, camera metadata and map bounds.

The campaign tasks are defined in `pipeline/tasks.yaml`.
