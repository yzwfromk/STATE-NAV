#!/usr/bin/env python3
import ast
import numpy as np
import os
from omegaconf import OmegaConf
from pathlib import Path as PathLib

import statenav_global
from statenav_global.world_model import *
import statenav_global.world_model.WorldModel as WorldModel
import statenav_global.world_model.globalmap as globalmap_module
from statenav_global.utility.ros_utils import publish_costmap_float32multiarray, publish_costmap_gridmap, Rviz_vis_travmap

import rclpy
import rclpy.parameter
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.clock import ClockType
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray, Bool
from grid_map_msgs.msg import GridMap


import threading

import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

cfg = OmegaConf.load(PathLib(statenav_global.__file__).parent / "configs/planning_config.yaml")


def parse_xy_override(value, name):
    if isinstance(value, str):
        raw_value = value.strip()
        if not raw_value:
            return None

        try:
            value = ast.literal_eval(raw_value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"{name} must be formatted as [x,y], got: {raw_value}") from exc

    if value is None:
        return None

    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values, got: {value}")

    try:
        return [float(value[0]), float(value[1])]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} values must be numeric, got: {value}") from exc


class WorldModelNode(Node):
    """
    WorldModelNode manages WorldModel and map updates.

    Architecture:
    - WorldModel runs in THIS process (writer) - updates maps in shared memory
    - Planner runs in SEPARATE process (reader) - reads maps from shared memory
    - Communication: Shared memory (maps) + ROS topics (coordination)

    Can run in two modes:
    1. Named shared memory: use_shared_memory=True (multiprocess, maps visible to Planner)
    2. No shared memory: use_shared_memory=False (single process, publishes /global_costmap)
    """

    def __init__(self, use_shared_memory=None, use_sim_time: bool = False):
        super().__init__(
            'worldmodel_node',
            parameter_overrides=[
                rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, use_sim_time)
            ]
        )
        clock_type = self.get_clock().clock_type
        if clock_type != ClockType.ROS_TIME:
            self.get_logger().warning(
                f"[WorldModel] use_sim_time is OFF (clock={clock_type.name}). "
                "Pass --ros-args -p use_sim_time:=true to use /clock."
            )
        else:
            self.get_logger().info("[WorldModel] Using ROS time from /clock")

        os.nice(-10)

        cfg_path = PathLib(statenav_global.__file__).parent / "configs/planning_config.yaml"
        self.cfg = OmegaConf.load(cfg_path)
        xy_parameter_descriptor = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter("start", "", descriptor=xy_parameter_descriptor)
        self.declare_parameter("goal", "", descriptor=xy_parameter_descriptor)
        start_override = parse_xy_override(self.get_parameter("start").value, "start")
        goal_override = parse_xy_override(self.get_parameter("goal").value, "goal")
        if start_override is not None:
            self.cfg.initial_start = start_override
            self.get_logger().info(f"[WorldModel] Overriding initial_start from launch: {start_override}")
        if goal_override is not None:
            self.cfg.global_goal = goal_override
            self.get_logger().info(f"[WorldModel] Overriding global_goal from launch: {goal_override}")
        WorldModel.cfg = self.cfg
        globalmap_module.cfg = self.cfg

        print("=" * 80)

        self.initialization_time = self.cfg.initialization_time
        self.debugging_visualization = self.cfg.debugging_visualization
        self.map_visualization_interval = self.cfg.map_visualization_interval



        # Determine shared memory mode
        self.use_shared_memory = use_shared_memory if use_shared_memory is not None else (
            self.cfg.get('use_shared_memory', False)
        )

        self._init_map_and_worldmodel()
        self._setup_ros_communication()

        self.get_logger().info(f"[WorldModel] WorldModelNode initialized {'with' if self.use_shared_memory else 'without'} shared memory")




####################################### INITIALIZATION #######################################

    def _init_map_and_worldmodel(self):
        """Initialize map and world model based on shared memory mode."""
        if self.use_shared_memory:
            self.get_logger().info("[WorldModel] Initializing WorldModel with shared memory (named)")
            self.world_model = WorldModel.WorldModel(use_shared_memory=True)
        else:
            self.get_logger().info("[WorldModel] Initializing map without shared memory (direct creation)")
            self.world_model = WorldModel.WorldModel(use_shared_memory=False)

    def _setup_ros_communication(self):
        """Setup ROS2 subscribers and publishers."""
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        self.create_subscription(PoseStamped, "/robot/pose",
                                 self.world_model.global_map.pose_callback, qos_profile)
        self.create_subscription(GridMap, "/elevation_mapping_node/elevation_map_filter",
                                 self.world_model.global_map.robo_centric_map_callback, qos_profile)
        # Publishers
        self.replanning_signal_pub = self.create_publisher(Bool, "/replanning_signal", qos_profile)

        self.use_gridmap_msg = self.cfg.get('use_gridmap_msg', False)
        self.frame_id = self.cfg.frame_id

        if not self.use_shared_memory:
            msg_type = GridMap if self.use_gridmap_msg else Float32MultiArray
            self.global_costmap_pub = self.create_publisher(msg_type, "/global_costmap", qos_profile)
            self.get_logger().info(f"[WorldModel] Using {'GridMap' if self.use_gridmap_msg else 'Float32MultiArray'} for costmap publishing")

        # Optional Rviz2 visualization: elevation map colored by north-direction cmd_v, at max 0.5 Hz
        if self.cfg.get('pub_travmap_visualization', False):
            self.travmap_vis_pub = self.create_publisher(GridMap, "/travmap_visualization", qos_profile)
            self.get_logger().info("[WorldModel] Publishing travmap visualization on /travmap_visualization (elevation + cmd_v @ north, 0.5 Hz)")
        else:
            self.travmap_vis_pub = None




####################################### UPDATE MAP #######################################

    def _update_map(self):
        """Update the global map."""
        visualize_flag = False
        current_time = self.get_clock().now().nanoseconds / 1e9
        if self.debugging_visualization and current_time - self.last_visualization_time > self.map_visualization_interval:
            visualize_flag = True
            self.last_visualization_time = current_time

        visualize_flag = False # Temp
        self.world_model.global_map.Update_map(
            path_plan=None,
            visualize_map=visualize_flag,
        )




####################################### PUBLISH #######################################

    def _publish_costmap(self):
        """Publish full costmap (non-shared-memory mode only)."""
        publish_func = publish_costmap_gridmap if self.use_gridmap_msg else publish_costmap_float32multiarray
        publish_func(
            self.world_model.global_map, self.frame_id, self.global_costmap_pub,
            node=self
        )

    def _publish_replanning_signal(self):
        """Notify the planner process to replan (shared-memory mode)."""
        msg = Bool()
        msg.data = True
        self.replanning_signal_pub.publish(msg)

    def _run_main_loop(self, current_time):
        """Map update, costmap publish, and replanning signals each loop iteration."""
        if current_time <= self.initialization_time:
            return

        self._update_map()

        if self.world_model.global_map.is_TraversabilityMap_built:
            if not self.use_shared_memory:
                self._publish_costmap()

            # Publish Rviz2 elevation+travmap visualization at max 0.5 Hz
            if self.travmap_vis_pub is not None:
                if current_time - self.last_travmap_vis_pub_time >= 2.0:
                    Rviz_vis_travmap(self.world_model.global_map, self.travmap_vis_pub, self, self.frame_id)
                    self.last_travmap_vis_pub_time = current_time


        self._publish_replanning_signal()








####################################### RUN #######################################

    def run(self):
        """Main execution loop."""
        self.get_logger().info("[WorldModel] WorldModelNode running. Waiting for map data...")

        spin_thread = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)
        spin_thread.start()

        self._last_loop_log_time = 0.0
        self._last_rostime_log_time = 0.0
        self.last_visualization_time = 0.0
        self.last_travmap_vis_pub_time = 0.0  # tracks last publish time for /travmap_visualization (0.5 Hz cap)

        while rclpy.ok():
            current_time = self.get_clock().now().nanoseconds / 1e9

            if current_time - self._last_loop_log_time > 1.0:
                print("\n==================================== LOOP ====================================")
                self._last_loop_log_time = current_time

            if current_time - self._last_rostime_log_time > 1.0:
                self.get_logger().info("[WorldModel] RCLPY in loop. Time is %.2f" % current_time)
                self._last_rostime_log_time = current_time

            try:
                self._run_main_loop(current_time)

            except Exception as e:
                self.get_logger().error(f"[WorldModel] Error in WorldModelNode run loop: {e}")
                import traceback
                self.get_logger().error(f"[WorldModel] {traceback.format_exc()}")




####################################### MAIN #######################################

def main():
    rclpy.init()
    node = None
    use_shared_memory = cfg.get('use_shared_memory', False)
    try:
        node = WorldModelNode(use_shared_memory=use_shared_memory, use_sim_time=True)
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.get_logger().info("[WorldModel] WorldModelNode shutting down")
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
