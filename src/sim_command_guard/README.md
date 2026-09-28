# sim_command_guard

Gazebo system plugin and lockstep scheduler.

- `src/command_guard_system.cc` applies the velocity command inside Gazebo and
  reports the executed command.
- `src/lockstep_scheduler.cc` steps the simulation in lockstep with the planner
  and camera pipeline, so that simulation speed does not change the results.
