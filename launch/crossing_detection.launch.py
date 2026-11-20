import os

from ament_index_python import get_package_share_directory
from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Generate the launch description for the crossing_detection node."""
    package_dir = get_package_share_directory("crossing_detection")

    # Declare launch argument with default value
    config_arg = DeclareLaunchArgument(
        "config_file",
        default_value=os.path.join(package_dir, "config", "crossing_detection.yaml"),
        description="Path to the config file for crossing detection",
    )

    config_file = LaunchConfiguration("config_file")

    # Create the crossing detection node
    crossing_detection_node = Node(
        package="crossing_detection",
        executable="crossing_detection_node",
        name="crossing_detection_node",
        output="screen",
        parameters=[config_file],
        emulate_tty=True,
    )

    return LaunchDescription([crossing_detection_node])
