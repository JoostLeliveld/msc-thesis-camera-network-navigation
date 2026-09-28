from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation (Gazebo) clock if true",
    )
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_lidar_arg = DeclareLaunchArgument(
        "use_lidar",
        default_value="true",
        description="Enable LiDAR sensor plugin in the URDF",
    )
    use_lidar = LaunchConfiguration("use_lidar")
    show_pose_markers_arg = DeclareLaunchArgument(
        "show_pose_markers",
        default_value="false",
        description=(
            "Render the opt-in cyan/magenta pose-keypoint marker disks. "
            "Burger only: warehouse_amr carries no marker disks and ignores it."
        ),
    )
    show_pose_markers = LaunchConfiguration("show_pose_markers")
    robot_model_arg = DeclareLaunchArgument(
        "robot_model",
        default_value="warehouse_amr",
        description=(
            "Robot URDF stem in sim/robot_description/urdf. "
            "'warehouse_amr' is the 0.80 x 0.55 m low-deck AMR used from "
            "2026-08-20; pass 'turtlebot3_burger' to reproduce any campaign "
            "captured before that."
        ),
    )
    robot_model = LaunchConfiguration("robot_model")

    robot_description = Command([
        "xacro ",
        PathJoinSubstitution([
            FindPackageShare("sim"),
            "robot_description",
            "urdf",
        ]),
        "/",
        robot_model,
        ".urdf.xacro",
        " ",
        "use_lidar:=",
        use_lidar,
        " ",
        "show_pose_markers:=",
        show_pose_markers,
    ])



    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        parameters=[{
            "robot_description": ParameterValue(robot_description, value_type=str),
            "use_sim_time": use_sim_time
        }],
        output="screen"
    )

    return LaunchDescription([
        use_sim_time_arg,
        use_lidar_arg,
        show_pose_markers_arg,
        robot_model_arg,
        robot_state_publisher,
    ])
