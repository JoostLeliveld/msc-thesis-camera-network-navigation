# sim

Gazebo simulation of the warehouse, the five wall cameras and the robot.

| File | Role |
| --- | --- |
| `launch/bringup_sim.launch.py` | starts Gazebo, the ROS bridges and the robot |
| `gazebo_worlds/worlds/warehouse_v2.world.sdf` | the warehouse used in the thesis |
| `models/` | camera, scenery and plan-view camera models |
| `robot_description/urdf/warehouse_amr.urdf.xacro` | the 0.80 x 0.55 m robot |
| `sim/encoder_noise_node.py` | encoder odometry with additive noise and AR(1) slip |
| `sim/actuation_noise_node.py` | noise on the executed velocity command |
| `sim/wait_for_odom.py`, `sim/wait_for_clock.py` | start-up gates |
| `fetch_external_models.sh` | downloads the two third-party assets the world renders with |

Run `bash src/sim/fetch_external_models.sh` once before the first launch.
