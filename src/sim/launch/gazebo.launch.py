from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration, PythonExpression
from launch.actions import SetEnvironmentVariable
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    sim_pkg_share = get_package_share_directory("sim")
    world_arg = DeclareLaunchArgument(
        "world",
        default_value="warehouse_aws.world.sdf",
        description="World file under sim/gazebo_worlds/worlds",
    )
    headless_arg = DeclareLaunchArgument(
        "headless",
        default_value="false",
        description="Run Gazebo server-only if true",
    )
    nvidia_offload_arg = DeclareLaunchArgument(
        "nvidia_offload",
        default_value="true",
        description=(
            "Route Gazebo's OGRE rendering (camera sensor) to the discrete "
            "NVIDIA GPU via PRIME render offload. On hybrid Intel+NVIDIA "
            "systems the gz server otherwise renders on the integrated GPU, "
            "capping the camera sensor rate. Set false on machines without an "
            "NVIDIA GPU."
        ),
    )
    world_path = PathJoinSubstitution([
        FindPackageShare("sim"),
        "gazebo_worlds",
        "worlds",
        LaunchConfiguration("world"),
    ])
    sim_pkg_share_parent = os.path.dirname(sim_pkg_share)

    gz_resource_paths = [
        sim_pkg_share_parent,
        os.path.join(sim_pkg_share, "models"),
        # Third-party Fuel assets (gitignored, fetched separately). Kept out of
        # the tracked models dir because they are tens of MB each.
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(sim_pkg_share)))), "src", "sim", "models_external"),
        os.path.join(sim_pkg_share, "gazebo_worlds", "models"),
        os.path.join(sim_pkg_share, "gazebo_worlds"),
        os.path.join(sim_pkg_share, "robot_description"),
        sim_pkg_share
    ]

    set_gz_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=":".join(gz_resource_paths)
    )
    # ROS 2 Humble's ros_gz packages on this machine target Gazebo Sim 6
    # (Fortress). Fortress still consumes the IGN_* spelling, while newer
    # Gazebo releases consume GZ_*. Keep both names identical so a world never
    # appears to load while its model:// camera assets are unresolved.
    set_ign_resource_path = SetEnvironmentVariable(
        name="IGN_GAZEBO_RESOURCE_PATH",
        value=":".join(gz_resource_paths)
    )

    # PRIME render offload: make the gz server's OGRE (camera sensor) GL context
    # bind to the discrete NVIDIA GPU instead of the integrated Intel GPU. On
    # this hybrid laptop the discrete Quadro sits idle while the iGPU renders the
    # camera, which is the binding limiter on the camera-sensor update rate. The
    # gz server (started by the included gz_sim.launch.py) inherits these because
    # they are set before the include in this scope. Verified: with both set,
    # `glxinfo` reports the Quadro instead of the Intel iGPU.
    set_prime_offload = SetEnvironmentVariable(
        name="__NV_PRIME_RENDER_OFFLOAD",
        value="1",
        condition=IfCondition(LaunchConfiguration("nvidia_offload")),
    )
    set_glx_vendor = SetEnvironmentVariable(
        name="__GLX_VENDOR_LIBRARY_NAME",
        value="nvidia",
        condition=IfCondition(LaunchConfiguration("nvidia_offload")),
    )
    # The server-only camera path uses EGL rather than GLX.  PRIME's GLX
    # variables alone leave GLVND free to select Mesa, which then tries (and on
    # this hybrid laptop fails) to create a DRI2 screen.  This matches the
    # already commissioned four-camera launch and makes the headless sensor
    # renderer select NVIDIA as well.
    set_egl_vendor = SetEnvironmentVariable(
        name="__EGL_VENDOR_LIBRARY_FILENAMES",
        value="/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
        condition=IfCondition(LaunchConfiguration("nvidia_offload")),
    )


    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("ros_gz_sim"),
                "launch",
                "gz_sim.launch.py"
            ])
        ]),
        launch_arguments={
            "gz_args": [
                PythonExpression([
                    "'-r -s --headless-rendering ' if '",
                    LaunchConfiguration("headless"),
                    "'.strip().lower() in ('1', 'true', 'yes', 'on') else '-r '",
                ]),
                world_path,
            ],
            # Pin the ABI used by /opt/ros/humble's ros_gz_bridge. Both
            # `ign gazebo` (Fortress 6) and `gz sim` (Harmonic 8) are installed
            # locally; allowing the latter to start yields advertised camera
            # bridges with no ROS image messages.
            "gz_version": "6",
        }.items()
    )

    def check_world_exists(context):
        """Fail here, with the path, if the world file is not installed.

        `world_path` is a lazy substitution, so a missing or stale world reaches
        the gz server as a path it cannot open and the run dies several seconds
        later inside Gazebo's own output. That happened for real: `colcon build
        --symlink-install` leaves DANGLING symlinks in install/ when a world is
        deleted from src/, so `ls` showed worlds that no longer existed and a
        launch could name one. Resolve the path up front and say so plainly.
        """
        resolved = world_path.perform(context)
        if not os.path.exists(resolved):
            worlds_dir = os.path.dirname(resolved)
            try:
                available = sorted(
                    f for f in os.listdir(worlds_dir)
                    if f.endswith(".sdf")
                    and os.path.exists(os.path.join(worlds_dir, f))
                )
            except OSError:
                available = []
            raise RuntimeError(
                f"world file does not exist: {resolved}\n"
                f"available worlds: {', '.join(available) or '(none)'}\n"
                "If it was deleted from src/, clear stale install symlinks with:\n"
                "  find install/sim build/sim -xtype l -delete"
            )
        return []

    return LaunchDescription([
        world_arg,
        headless_arg,
        nvidia_offload_arg,
        set_gz_resource_path,
        set_ign_resource_path,
        set_prime_offload,
        set_glx_vendor,
        set_egl_vendor,
        OpaqueFunction(function=check_world_exists),
        gazebo,
    ])
