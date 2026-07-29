from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    statenav_src = EnvironmentVariable(
        "STATENAV_SRC",
        default_value="/home/rowan-l/ros2_ws/statenav_ws/src/state_nav",
    )

    bag_path = LaunchConfiguration("bag_path")
    rviz_config = LaunchConfiguration("rviz_config")
    planner_delay = LaunchConfiguration("planner_delay")
    bag_delay = LaunchConfiguration("bag_delay")
    rviz_delay = LaunchConfiguration("rviz_delay")
    start = LaunchConfiguration("start")
    goal = LaunchConfiguration("goal")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "bag_path",
                default_value=PathJoinSubstitution([statenav_src, "ROS", "bag_isaac"]),
                description="Path to the rosbag directory used for demo replay.",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=PathJoinSubstitution([statenav_src, "ROS", "rviz_setting.rviz"]),
                description="Path to the RViz2 config file.",
            ),
            DeclareLaunchArgument(
                "planner_delay",
                default_value="3.0",
                description="Seconds to wait before starting global_planning.",
            ),
            DeclareLaunchArgument(
                "bag_delay",
                default_value="6.0",
                description="Seconds to wait before playing the demo rosbag.",
            ),
            DeclareLaunchArgument(
                "rviz_delay",
                default_value="8.0",
                description="Seconds to wait before opening RViz2.",
            ),
            DeclareLaunchArgument(
                "play_bag",
                default_value="true",
                description="Set false to skip rosbag replay.",
            ),
            DeclareLaunchArgument(
                "start_rviz",
                default_value="true",
                description="Set false to skip RViz2.",
            ),
            DeclareLaunchArgument(
                "start",
                default_value="",
                description="Optional planning start as [x,y]. Empty uses planning_config.yaml initial_start.",
            ),
            DeclareLaunchArgument(
                "goal",
                default_value="",
                description="Optional planning goal as [x,y]. Empty uses planning_config.yaml global_goal.",
            ),
            Node(
                package="statenav_global",
                executable="traversability_estimation",
                name="worldmodel_node",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": True,
                        "start": start,
                        "goal": goal,
                    }
                ],
            ),
            TimerAction(
                period=planner_delay,
                actions=[
                    Node(
                        package="statenav_global",
                        executable="global_planning",
                        name="planning_node",
                        output="screen",
                        parameters=[
                            {
                                "use_sim_time": True,
                                "start": start,
                                "goal": goal,
                            }
                        ],
                    )
                ],
            ),
            TimerAction(
                period=bag_delay,
                condition=IfCondition(LaunchConfiguration("play_bag")),
                actions=[
                    ExecuteProcess(
                        cmd=["ros2", "bag", "play", bag_path, "--clock"],
                        output="screen",
                    )
                ],
            ),
            TimerAction(
                period=rviz_delay,
                condition=IfCondition(LaunchConfiguration("start_rviz")),
                actions=[
                    ExecuteProcess(
                        cmd=["rviz2", "-d", rviz_config],
                        output="screen",
                    )
                ],
            ),
        ]
    )
