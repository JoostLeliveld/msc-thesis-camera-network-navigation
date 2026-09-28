from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )
    cam_x_arg = DeclareLaunchArgument('cam_x', default_value='-3')
    cam_y_arg = DeclareLaunchArgument('cam_y', default_value='-3')
    cam_z_arg = DeclareLaunchArgument('cam_z', default_value='6.0')
    cam_qx_arg = DeclareLaunchArgument('cam_qx', default_value='0.141234')
    cam_qy_arg = DeclareLaunchArgument('cam_qy', default_value='-0.340969')
    cam_qz_arg = DeclareLaunchArgument('cam_qz', default_value='0.355669')
    cam_qw_arg = DeclareLaunchArgument('cam_qw', default_value='0.858658')
    odom_x_arg = DeclareLaunchArgument('odom_x', default_value='0.0')
    odom_y_arg = DeclareLaunchArgument('odom_y', default_value='0.0')
    odom_z_arg = DeclareLaunchArgument('odom_z', default_value='0.0')
    odom_qx_arg = DeclareLaunchArgument('odom_qx', default_value='0.0')
    odom_qy_arg = DeclareLaunchArgument('odom_qy', default_value='0.0')
    odom_qz_arg = DeclareLaunchArgument('odom_qz', default_value='0.0')
    odom_qw_arg = DeclareLaunchArgument('odom_qw', default_value='1.0')

    return LaunchDescription([
        use_sim_time,
        cam_x_arg,
        cam_y_arg,
        cam_z_arg,
        cam_qx_arg,
        cam_qy_arg,
        cam_qz_arg,
        cam_qw_arg,
        odom_x_arg,
        odom_y_arg,
        odom_z_arg,
        odom_qx_arg,
        odom_qy_arg,
        odom_qz_arg,
        odom_qw_arg,
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_bev_to_odom',
            arguments=[
                '--x', LaunchConfiguration('odom_x'),
                '--y', LaunchConfiguration('odom_y'),
                '--z', LaunchConfiguration('odom_z'),
                '--qx', LaunchConfiguration('odom_qx'),
                '--qy', LaunchConfiguration('odom_qy'),
                '--qz', LaunchConfiguration('odom_qz'),
                '--qw', LaunchConfiguration('odom_qw'),
                '--frame-id', 'map_bev',
                '--child-frame-id', 'odom',
            ],
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_bev_to_camera',
            arguments=[
                '--x', LaunchConfiguration('cam_x'),
                '--y', LaunchConfiguration('cam_y'),
                '--z', LaunchConfiguration('cam_z'),
                '--qx', LaunchConfiguration('cam_qx'),
                '--qy', LaunchConfiguration('cam_qy'),
                '--qz', LaunchConfiguration('cam_qz'),
                '--qw', LaunchConfiguration('cam_qw'),
                '--frame-id', 'map_bev',
                '--child-frame-id', 'external_camera/link/camera',
            ],
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
        )
    ])
