import os

from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    DeclareLaunchArgument,
    RegisterEventHandler,
    OpaqueFunction,
    EmitEvent,
    LogInfo,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from ament_index_python.packages import get_package_share_directory


def _reject_unsafe_in_place_reset(context, *args, **kwargs):
    if LaunchConfiguration("reset_world").perform(context).lower() == "true":
        raise RuntimeError(
            "reset_world:=true is unsupported: Gazebo Sim 6 reset.all rewinds time "
            "without clearing robot pose, odometry, or the DiffDrive target. Restart "
            "the complete launch for a new simulator epoch."
        )
    return []


def _continue_only_on_success(next_action, gate_name):
    def _handler(event, _context):
        if event.returncode == 0:
            return [next_action]
        reason = f"{gate_name} exited with code {event.returncode}; aborting simulator startup"
        return [LogInfo(msg=reason), EmitEvent(event=Shutdown(reason=reason))]
    return _handler



def generate_launch_description():
    # A--D are retained as named variables below for backwards-compatible
    # launch APIs. Optional larger camera registries add E--L; build those declarations and isolated
    # bridges mechanically so camera topic/model suffixes cannot drift apart.
    extra_camera_suffixes = tuple("efghijkl")
    extra_segmentation_args = []
    extra_segmentation_flags = {}
    extra_camera_args = []
    extra_camera_flags = {}
    for suffix in extra_camera_suffixes:
        segmentation_name = f"bridge_segmentation_{suffix}"
        camera_name = f"bridge_camera_{suffix}"
        extra_segmentation_args.append(DeclareLaunchArgument(
            segmentation_name,
            default_value="false",
            description=f"Bridge camera-{suffix.upper()} semantic labels for offline dataset capture",
        ))
        extra_segmentation_flags[suffix] = LaunchConfiguration(segmentation_name)
        extra_camera_args.append(DeclareLaunchArgument(
            camera_name,
            default_value="false",
            description=f"Bridge extension-only /external_camera_{suffix} RGB and camera_info topics",
        ))
        extra_camera_flags[suffix] = LaunchConfiguration(camera_name)

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
        description="Render the opt-in cyan/magenta pose-keypoint marker disks",
    )
    show_pose_markers = LaunchConfiguration("show_pose_markers")
    robot_model_arg = DeclareLaunchArgument(
        "robot_model",
        default_value="warehouse_amr",
        description=(
            "Robot URDF stem. 'warehouse_amr' is the 0.80 x 0.55 m low-deck AMR "
            "used from 2026-08-20; 'turtlebot3_burger' reproduces earlier runs."
        ),
    )
    robot_model = LaunchConfiguration("robot_model")
    bridge_scan_arg = DeclareLaunchArgument(
        "bridge_scan",
        default_value="true",
        description="Bridge /scan between Gazebo and ROS",
    )
    bridge_scan = LaunchConfiguration("bridge_scan")
    bridge_segmentation_arg = DeclareLaunchArgument(
        "bridge_segmentation",
        default_value="false",
        description="Bridge external-camera semantic segmentation labels for offline dataset capture",
    )
    bridge_segmentation = LaunchConfiguration("bridge_segmentation")
    bridge_segmentation_b_arg = DeclareLaunchArgument(
        "bridge_segmentation_b",
        default_value="false",
        description="Bridge camera-B semantic labels for offline dataset capture",
    )
    bridge_segmentation_b = LaunchConfiguration("bridge_segmentation_b")
    bridge_segmentation_c_arg = DeclareLaunchArgument(
        "bridge_segmentation_c",
        default_value="false",
        description="Bridge camera-C semantic labels for offline dataset capture",
    )
    bridge_segmentation_c = LaunchConfiguration("bridge_segmentation_c")
    bridge_segmentation_d_arg = DeclareLaunchArgument(
        "bridge_segmentation_d",
        default_value="false",
        description="Bridge camera-D semantic labels for offline dataset capture",
    )
    bridge_segmentation_d = LaunchConfiguration("bridge_segmentation_d")
    bridge_camera_a_arg = DeclareLaunchArgument(
        "bridge_camera_a",
        default_value="true",
        description="Bridge the primary /external_camera RGB and camera_info topics",
    )
    bridge_camera_a = LaunchConfiguration("bridge_camera_a")
    bridge_camera_b_arg = DeclareLaunchArgument(
        "bridge_camera_b",
        default_value="false",
        description="Bridge extension-only /external_camera_b RGB and camera_info topics",
    )
    bridge_camera_b = LaunchConfiguration("bridge_camera_b")
    bridge_camera_c_arg = DeclareLaunchArgument(
        "bridge_camera_c",
        default_value="false",
        description="Bridge extension-only /external_camera_c RGB and camera_info topics",
    )
    bridge_camera_c = LaunchConfiguration("bridge_camera_c")
    bridge_camera_d_arg = DeclareLaunchArgument(
        "bridge_camera_d",
        default_value="false",
        description="Bridge extension-only /external_camera_d RGB and camera_info topics",
    )
    bridge_camera_d = LaunchConfiguration("bridge_camera_d")
    bridge_overview_camera_arg = DeclareLaunchArgument(
        "bridge_overview_camera",
        default_value="false",
        description="Bridge the presentation-only full-facility overview camera",
    )
    bridge_overview_camera = LaunchConfiguration("bridge_overview_camera")
    world_arg = DeclareLaunchArgument(
        "world",
        default_value="warehouse_aws.world.sdf",
        description="World file under sim/gazebo_worlds/worlds",
    )
    world = LaunchConfiguration("world")
    world_name_arg = DeclareLaunchArgument(
        "world_name",
        default_value=PythonExpression([
            "'",
            LaunchConfiguration("world"),
            "'.replace('.world.sdf','').replace('.sdf','')"
        ]),
        description="World name (used for /world/<name>/clock bridging)",
    )
    world_name = LaunchConfiguration("world_name")
    headless_arg = DeclareLaunchArgument(
        "headless",
        default_value="false",
        description="Run Gazebo server-only if true",
    )
    headless = LaunchConfiguration("headless")
    nvidia_offload_arg = DeclareLaunchArgument(
        "nvidia_offload",
        default_value="true",
        description="Route headless camera rendering through NVIDIA PRIME/EGL",
    )
    nvidia_offload = LaunchConfiguration("nvidia_offload")
    reset_world_arg = DeclareLaunchArgument(
        "reset_world",
        default_value="false",
        description="Unsupported in-place reset guard; use a complete launch restart",
    )
    spawn_x_arg = DeclareLaunchArgument(
        "spawn_x",
        default_value="0.0",
        description="Robot spawn x position",
    )
    spawn_y_arg = DeclareLaunchArgument(
        "spawn_y",
        default_value="0.0",
        description="Robot spawn y position",
    )
    spawn_z_arg = DeclareLaunchArgument(
        "spawn_z",
        default_value="0.05",
        description="Robot spawn z position",
    )
    spawn_yaw_arg = DeclareLaunchArgument(
        "spawn_yaw",
        default_value="0.0",
        description="Robot spawn yaw (radians)",
    )
    spawn_x = LaunchConfiguration("spawn_x")
    spawn_y = LaunchConfiguration("spawn_y")
    spawn_z = LaunchConfiguration("spawn_z")
    spawn_yaw = LaunchConfiguration("spawn_yaw")

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("sim"),
                "launch",
                "gazebo.launch.py"
            ])
        ),
        launch_arguments={
            "world": world,
            "headless": headless,
            "nvidia_offload": nvidia_offload,
        }.items(),
    )

    robot_description = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("sim"),
                "launch",
                "robot_description.launch.py"
            ])
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "use_lidar": use_lidar,
            "show_pose_markers": show_pose_markers,
            "robot_model": robot_model,
        }.items(),
    )

    wait_for_clock = Node(
        package="sim",
        executable="wait_for_clock",
        name="wait_for_clock",
        output="screen",
        parameters=[{
            "topic": "/clock",
            "timeout_s": 30.0,
            "min_messages": 3,
        }],
    )
    clock_throttle = Node(
        package="sim",
        executable="clock_throttle_node",
        name="clock_throttle",
        output="screen",
        parameters=[{
            "input_topic": "/clock_full",
            "output_topic": "/clock",
            "publish_rate_hz": 50.0,
        }],
    )

    spawn = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name", "turtlebot3",
            "-topic", "robot_description",
            "-x", spawn_x,
            "-y", spawn_y,
            "-z", spawn_z,
            "-Y", spawn_yaw,
        ],
        output="screen"
    )

    spawn_after_clock = RegisterEventHandler(
        OnProcessExit(
            target_action=wait_for_clock,
            on_exit=_continue_only_on_success(spawn, "wait_for_clock"),
        ),
    )



    clock_arg = PythonExpression([
        "'/world/' + '",
        world_name,
        "' + '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'"
    ])
    set_pose_service_arg = PythonExpression([
        "'/world/' + '",
        world_name,
        "' + '/set_pose@ros_gz_interfaces/srv/SetEntityPose'"
    ])
    clock_remap_src = PythonExpression([
        "'/world/' + '",
        world_name,
        "' + '/clock'"
    ])
    # Ground-truth pose: Gazebo publishes every moving entity's world pose on
    # /world/<name>/dynamic_pose/info (gz.msgs.Pose_V). Bridge it so the logger
    # can measure localization/belief error against TRUE pose instead of the
    # DiffDrive wheel odometry (/odom), which itself drifts in turns.
    gt_pose_src = PythonExpression([
        "'/world/' + '",
        world_name,
        "' + '/dynamic_pose/info'"
    ])
    gt_pose_arg = PythonExpression([
        "'/world/' + '",
        world_name,
        "' + '/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V'"
    ])

    ros_gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/model/turtlebot3/cmd_vel_guard_input@geometry_msgs/msg/Twist]gz.msgs.Twist",
            "/model/turtlebot3/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            "/model/turtlebot3/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
            "/model/turtlebot3/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model",
            "/model/turtlebot3/actuation_outcome@std_msgs/msg/String[gz.msgs.StringMsg",
            # Segmentation stays out of the runtime bridge. Dataset capture
            # enables it through the separate conditional bridge below so a
            # stalled semantic stream cannot take the RGB bridge down with it.
            #
            # Gazebo renders semantic segmentation at 1280x720 and it is a
            # dataset-only cost. Runtime YOLO consumes only RGB images.
            #
            # YOLO uses its own internal mask from the plain RGB image; these
            # semantic segmentation topics are never consumed at runtime.
            # "/external_camera/segmentation/colored_map@sensor_msgs/msg/Image[gz.msgs.Image",
            # "/external_camera/segmentation/labels_map@sensor_msgs/msg/Image[gz.msgs.Image",
            clock_arg,
            set_pose_service_arg,
        ],
        remappings=[
            ("/model/turtlebot3/cmd_vel_guard_input", "/cmd_vel"),
            ("/model/turtlebot3/odometry", "/odom"),
            ("/model/turtlebot3/tf", "/tf"),
            ("/model/turtlebot3/joint_states", "/joint_states"),
            ("/model/turtlebot3/actuation_outcome", "/sim/actuation_outcome"),
            (clock_remap_src, "/clock_full"),
        ],
        output="screen",
    )
    # Ground-truth pose on a SEPARATE bridge node (isolated from the others)
    # so it can never take the main /odom + camera bridge down.
    ros_gz_groundtruth_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[gt_pose_arg],
        remappings=[(gt_pose_src, "/ground_truth_tf")],
        output="screen",
    )
    ros_gz_segmentation_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/external_camera/segmentation/labels_map@sensor_msgs/msg/Image[gz.msgs.Image",
        ],
        output="screen",
        condition=IfCondition(bridge_segmentation),
    )
    # Keep every semantic stream on an isolated bridge.  The label images are
    # large and intentionally opt-in; four-camera dataset capture enables only
    # the camera being collected so a slow semantic renderer cannot stall the
    # operational RGB/odometry bridge or the other cameras.
    ros_gz_segmentation_b_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/external_camera_b/segmentation/labels_map@sensor_msgs/msg/Image[gz.msgs.Image",
        ],
        output="screen",
        condition=IfCondition(bridge_segmentation_b),
    )
    ros_gz_segmentation_c_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/external_camera_c/segmentation/labels_map@sensor_msgs/msg/Image[gz.msgs.Image",
        ],
        output="screen",
        condition=IfCondition(bridge_segmentation_c),
    )
    ros_gz_segmentation_d_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/external_camera_d/segmentation/labels_map@sensor_msgs/msg/Image[gz.msgs.Image",
        ],
        output="screen",
        condition=IfCondition(bridge_segmentation_d),
    )
    ros_gz_camera_b_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/external_camera_b/image_raw@sensor_msgs/msg/Image[gz.msgs.Image",
            "/external_camera_b/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
        ],
        output="screen",
        condition=IfCondition(bridge_camera_b),
    )
    # Camera A is isolated too.  This keeps the default ROS contract unchanged
    # while allowing the diagnostic direct-Gazebo detector to remove its large
    # image conversion without risking odometry/control bridging.
    ros_gz_camera_a_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/external_camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image",
            "/external_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
        ],
        output="screen",
        condition=IfCondition(bridge_camera_a),
    )
    ros_gz_camera_c_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/external_camera_c/image_raw@sensor_msgs/msg/Image[gz.msgs.Image",
            "/external_camera_c/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
        ],
        output="screen",
        condition=IfCondition(bridge_camera_c),
    )
    ros_gz_camera_d_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/external_camera_d/image_raw@sensor_msgs/msg/Image[gz.msgs.Image",
            "/external_camera_d/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
        ],
        output="screen",
        condition=IfCondition(bridge_camera_d),
    )
    extra_segmentation_bridges = [
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            arguments=[
                f"/external_camera_{suffix}/segmentation/labels_map"
                "@sensor_msgs/msg/Image[gz.msgs.Image",
            ],
            name=f"ros_gz_segmentation_{suffix}_bridge",
            output="screen",
            condition=IfCondition(extra_segmentation_flags[suffix]),
        )
        for suffix in extra_camera_suffixes
    ]
    extra_camera_bridges = [
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            arguments=[
                f"/external_camera_{suffix}/image_raw"
                "@sensor_msgs/msg/Image[gz.msgs.Image",
                f"/external_camera_{suffix}/camera_info"
                "@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            ],
            name=f"ros_gz_camera_{suffix}_bridge",
            output="screen",
            condition=IfCondition(extra_camera_flags[suffix]),
        )
        for suffix in extra_camera_suffixes
    ]
    ros_gz_overview_camera_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/presentation_overview_camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image",
            "/presentation_overview_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
        ],
        output="screen",
        condition=IfCondition(bridge_overview_camera),
    )
    ros_gz_scan_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
        ],
        output="screen",
        condition=IfCondition(bridge_scan),
    )

    return LaunchDescription([
        use_sim_time_arg,
        use_lidar_arg,
        show_pose_markers_arg,
        robot_model_arg,
        bridge_scan_arg,
        world_arg,
        world_name_arg,
        headless_arg,
        nvidia_offload_arg,
        bridge_segmentation_arg,
        bridge_segmentation_b_arg,
        bridge_segmentation_c_arg,
        bridge_segmentation_d_arg,
        *extra_segmentation_args,
        bridge_camera_a_arg,
        bridge_camera_b_arg,
        bridge_camera_c_arg,
        bridge_camera_d_arg,
        *extra_camera_args,
        bridge_overview_camera_arg,
        reset_world_arg,
        OpaqueFunction(function=_reject_unsafe_in_place_reset),
        spawn_x_arg,
        spawn_y_arg,
        spawn_z_arg,
        spawn_yaw_arg,
        gazebo,
        robot_description,
        ros_gz_bridge,
        clock_throttle,
        spawn_after_clock,
        wait_for_clock,
        ros_gz_segmentation_bridge,
        ros_gz_segmentation_b_bridge,
        ros_gz_segmentation_c_bridge,
        ros_gz_segmentation_d_bridge,
        *extra_segmentation_bridges,
        ros_gz_camera_a_bridge,
        ros_gz_camera_b_bridge,
        ros_gz_camera_c_bridge,
        ros_gz_camera_d_bridge,
        *extra_camera_bridges,
        ros_gz_overview_camera_bridge,
        ros_gz_groundtruth_bridge,
        ros_gz_scan_bridge,
    ])
