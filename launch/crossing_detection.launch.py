import os

from ament_index_python import get_package_share_directory
from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """
    Generate the launch description for the crossing_detection node.

    This launch exposes the default parameters defined in the node's
    constructor so you can override them at runtime.
    """
    debug = LaunchConfiguration("debug")
    params_file = LaunchConfiguration("params_file")
    image_subscriber = LaunchConfiguration("image_subscriber")
    debug_image_publisher = LaunchConfiguration("debug_image_publisher")
    result_publisher = LaunchConfiguration("result_publisher")
    state = LaunchConfiguration("state")
    image_path = LaunchConfiguration("image_path")
    compute_crossing_center = LaunchConfiguration("compute_crossing_center")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "debug", default_value="False", description="Enable debug mode"
            ),
            DeclareLaunchArgument(
                "image_subscriber",
                default_value="/camera/birds_eye",
                description="Image subscriber topic",
            ),
            DeclareLaunchArgument(
                "debug_image_publisher",
                default_value="/crossing_detection/debug/image",
                description="Debug image publisher topic",
            ),
            DeclareLaunchArgument(
                "result_publisher",
                default_value="/crossing_detection/result",
                description="Result publisher topic",
            ),
            DeclareLaunchArgument(
                "state",
                default_value="1",
                description="Initial node state (NodeState value)",
            ),
            DeclareLaunchArgument(
                "image_path",
                default_value="resources/img/example.png",
                description="Relative image path inside package",
            ),
            DeclareLaunchArgument(
                "compute_crossing_center",
                default_value="False",
                description="Whether to compute crossing center",
            ),
            DeclareLaunchArgument(
                "params_file",
                default_value=os.path.join(
                    get_package_share_directory("crossing_detection"),
                    "config",
                    "ros_params.yaml",
                ),
                description="Path to the ROS parameters file",
            ),
            Node(
                package="crossing_detection",
                namespace="",  # Is also the namespace for loading the params
                executable="crossing_detection_node",
                name="crossing_detection_node",
                parameters=[
                    {"debug": debug},
                    {"image_subscriber": image_subscriber},
                    {"debug_image_publisher": debug_image_publisher},
                    {"result_publisher": result_publisher},
                    {"state": state},
                    {"image_path": image_path},
                    {"compute_crossing_center": compute_crossing_center},
                    params_file,
                ],
            ),
        ]
    )
