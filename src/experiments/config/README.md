# Experiment configuration

This directory contains the reusable ROS experiment registry. The submitted thesis
campaign uses `warehouse_v2.world.sdf`. Its immutable campaign-facing task definitions
live in `pipeline/tasks.yaml`; the broader `tasks.yaml` file remains a simulator fixture
for package-level tests and exploratory launches.

- `world_profiles.yaml`: world geometry, camera metadata, map bounds, and planner defaults.
- `warehouse_v2_keepout.json`: collision-derived keep-out geometry for the final world.
- `tasks.yaml`: generic launch-task registry.

For exact reproduction, start at the repository root `README.md` and use the locked files
under `pipeline/`. A profile entry never silently selects a learned artifact; campaign
configurations must name those artifacts explicitly.
