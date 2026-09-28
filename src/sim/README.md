# Simulator

This ROS 2 package supplies the physical plant and sensing environment used by the thesis:
the Gazebo warehouse, wall cameras, AMR description, startup gates, and command/encoder
noise.

| File | Role |
| --- | --- |
| `launch/bringup_sim.launch.py` | simulator bringup used by the experiment package |
| `gazebo_worlds/worlds/warehouse_v2.world.sdf` | final warehouse, camera network, and collision geometry |
| `models/external_camera/model.sdf` | external camera model |
| `robot_description/urdf/warehouse_amr.urdf.xacro` | AMR description |
| `sim/actuation_noise_node.py` | optional executed-command perturbation |
| `sim/encoder_noise_node.py` | noisy odometry stream used by the final campaign |
| `sim/wait_for_odom.py` | launch startup gate |

The submitted campaign uses `warehouse_v2.world.sdf`. Other worlds are retained only where
package-level simulator tests require them; they are not alternative thesis results.
