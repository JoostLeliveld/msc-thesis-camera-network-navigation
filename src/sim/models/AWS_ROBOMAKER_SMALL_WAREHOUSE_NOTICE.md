# AWS RoboMaker Small Warehouse asset notice

Selected models in this directory come from the official AWS Robotics
repository:

- Repository: https://github.com/aws-robotics/aws-robomaker-small-warehouse-world
- Branch/source layout used: `ros2`
- Package name: `aws_robomaker_small_warehouse_world`
- License: MIT-0, copied in `aws_robomaker_small_warehouse_LICENSE_MIT0.txt`

The active Experiment B world is:

- `src/sim/gazebo_worlds/worlds/warehouse_aws.world.sdf`

That world is a simplified, auditable extraction of the top-right shelf section
of the official/JdeRobot warehouse layout.  It uses primitive shelf and box
geometry rather than the full decorative AWS scene so the paper task is visually
unambiguous.  The primitive geometry is the experiment contract; the copied AWS
assets remain available for provenance and future visual variants.
