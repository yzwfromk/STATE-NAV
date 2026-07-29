#!/usr/bin/env python3
import ast
import numpy as np
import math
from omegaconf import OmegaConf

import statenav_global
from statenav_global import CMDbasedMap
from statenav_global.world_model.globalmap_with_backend import CMDbasedMapReaderProxy
from statenav_global.world_model.shared_memory_backend import SharedMemoryBackend
from statenav_global.utility.ros_utils import populate_map_from_float32multiarray, populate_map_from_gridmap

# Note: tf2_ros imports available if needed for future TF operations
# from tf2_ros import TransformListener, Buffer, TransformException
# import tf2_geometry_msgs

import rclpy
import rclpy.parameter
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.clock import ClockType
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray, Bool
from grid_map_msgs.msg import GridMap
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

import threading
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

from pathlib import Path as PathLib
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


def set_random_seed(seed):
    rng = np.random.RandomState(seed)
    print(f"Set random seed to {seed} in numpy.")
    return rng


def wrap_to_pi(angle):
    """Wraps an angle to the range [-π, π] using atan2."""
    return np.arctan2(np.sin(angle), np.cos(angle))






















################################################### Helper Functions for ROS ###################################################





































class PlanningNode(Node):
    """Main planning node that subscribes to map and publishes paths."""

    def __init__(self, use_sim_time: bool = False):
        """
        Initializescore_travcmd PlanningNode
        """
        super().__init__(
            'planning_node',
            parameter_overrides=[
                rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, use_sim_time)
            ]
        )
        clock_type = self.get_clock().clock_type
        if clock_type != ClockType.ROS_TIME:
            self.get_logger().warning(
                f"[PathPlanner] use_sim_time is OFF (clock={clock_type.name}). "
                "Pass --ros-args -p use_sim_time:=true to use /clock."
            )
        else:
            self.get_logger().info("[PathPlanner] Using ROS time from /clock")

        # Load configuration
        self.cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        xy_parameter_descriptor = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter("start", "", descriptor=xy_parameter_descriptor)
        self.declare_parameter("goal", "", descriptor=xy_parameter_descriptor)
        start_override = parse_xy_override(self.get_parameter("start").value, "start")
        goal_override = parse_xy_override(self.get_parameter("goal").value, "goal")
        if start_override is not None:
            self.cfg.initial_start = start_override
            self.get_logger().info(f"[PathPlanner] Overriding initial_start from launch: {start_override}")
        if goal_override is not None:
            self.cfg.global_goal = goal_override
            self.get_logger().info(f"[PathPlanner] Overriding global_goal from launch: {goal_override}")

        # Check if using shared memory mode
        self.use_shared_memory = self.cfg.get('use_shared_memory', False)
        
        # Calculate task extent
        self.task_extent = [
            self.cfg.env_extent[0] + 1*self.cfg.local_patch_size,
            self.cfg.env_extent[1] - 1*self.cfg.local_patch_size,
            self.cfg.env_extent[2] + 1*self.cfg.local_patch_size,
            self.cfg.env_extent[3] - 1*self.cfg.local_patch_size
        ]
        diagonal = math.hypot(
            self.task_extent[1] - self.task_extent[0],
            self.task_extent[3] - self.task_extent[2]
        )
        
        global_goal = self.cfg.global_goal
        










        # Initialize map based on mode
        if self.use_shared_memory:
            # Shared memory mode: attach to shared memory created by WorldModel
            self.get_logger().info("[PathPlanner] Initializing in SHARED MEMORY mode (reader, named)")
            backend = SharedMemoryBackend(mode='reader')
            
            # Note: CMDbasedMapReaderProxy only supports CMDbasedMap for now
            # If you need other map types, you'll need to create reader proxies for them
            if self.cfg.trav_option == "Proposed":
                self.global_map = CMDbasedMapReaderProxy(backend)
                # Set goal after initialization
                self.global_map.goal_x = global_goal[0]
                self.global_map.goal_y = global_goal[1]
            else:
                self.get_logger().warning(f"[PathPlanner] Shared memory mode currently only supports CMDbasedMap (trav_option: Proposed).")
                self.get_logger().warning(f"[PathPlanner] Requested: {self.cfg.trav_option}. Falling back to ROS message mode.")
                self.use_shared_memory = False
        
        if not self.use_shared_memory:
            # ROS message mode: create map and subscribe to topics
            self.get_logger().info("[PathPlanner] Initializing in ROS MESSAGE mode")
            if self.cfg.trav_option == "Proposed":
                self.global_map = CMDbasedMap(
                    env_xmin=self.cfg.env_extent[0], 
                    env_xmax=self.cfg.env_extent[1], 
                    env_ymin=self.cfg.env_extent[2], 
                    env_ymax=self.cfg.env_extent[3],
                    goal_x=global_goal[0], 
                    goal_y=global_goal[1],
                    which_layer=self.cfg.which_layer,
                    preest_update_resolution=self.cfg.trav_estimation_resoultion, 
                    instab_limit=self.cfg.instability_limit,
                    load_Travformer = False)
            else:
                raise ValueError(f"Invalid trav_option: {self.cfg.trav_option}")
        
        self.get_logger().info(f"[PathPlanner] Map class initialized. Type: {type(self.global_map).__name__}, Mode: {'SHARED MEMORY' if self.use_shared_memory else 'ROS MESSAGE'}")
        













        # Initialize RRT planner
        self.initial_start = self.cfg.initial_start
        self.global_goal = self.cfg.global_goal
        self.heading_start = np.deg2rad(self.cfg.heading_start)
        
        self.rng = set_random_seed(self.cfg.seed)
        
        iter_max_global = self.cfg.iter_max
        branch_length_max = self.cfg.branch_length_max_ratio * diagonal
        search_radius = self.cfg.search_radius_ratio * diagonal
        
        self.global_planner = statenav_global.planners.GlobalRRTStar(
            self.task_extent, self.rng, self.initial_start, self.global_goal, self.heading_start,
            goal_radius=diagonal * self.cfg.goal_radius_ratio,
            branch_length_max=branch_length_max,
            search_radius=search_radius,
            decrease_search_radius=True,
            iter_max=iter_max_global,
            convergence_threshold=self.cfg.convergence_ratio,
            switch_to_informed_from_thisiter=self.cfg.switch_to_informed_from_thisiter,
            sampling_dist=self.cfg.sampling_dist,
            num_samplingpoints=self.cfg.num_samplingpoints,
            default_obstacle_clearance=self.cfg.obs_clearance,
        )
        self.global_planner.global_map = self.global_map

        # ROS2 setup
        self.frame_id = self.cfg.get('frame_id', 'map')

        # QoS profile for subscribers
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribers (different based on mode)
        if self.use_shared_memory:
            # Shared memory mode: subscribe to pose and replanning signal via ROS
            self.create_subscription(PoseStamped, "/robot/pose", self.pose_callback, qos_profile)
            self.create_subscription(Bool, "/replanning_signal", self.replanning_callback, qos_profile)
            self.get_logger().info("[PathPlanner] Subscribed to /robot/pose and /replanning_signal (shared memory mode)")
        else:
            # ROS message mode: subscribe to map topics and pose
            self.use_gridmap_msg = self.cfg.get('use_gridmap_msg', False)
            if self.use_gridmap_msg:
                self.create_subscription(GridMap, "/global_costmap", self.map_callback, qos_profile)
                self.get_logger().info("[PathPlanner] Subscribed to /global_costmap (GridMap)")
            else:
                self.create_subscription(Float32MultiArray, "/global_costmap", self.map_callback, qos_profile)
                self.get_logger().info("[PathPlanner] Subscribed to /global_costmap (Float32MultiArray)")

            self.create_subscription(PoseStamped, "/robot/pose", self.pose_callback, qos_profile)
            self.get_logger().info("[PathPlanner] Subscribed to /robot/pose")

        print(self.global_map.robot_heading)




        # Publishers
        self.global_path_pub = self.create_publisher(Path, "/global_path", qos_profile)
        self.get_logger().info("[PathPlanner] Publishers initialized: /global_path")













        
        # For getting next waypoint for resetting the tree
        self.waypoint_lookahead_time = self.cfg.waypoint_lookahead_time
        
        # State
        self.last_plan_time = 0
        self.planning_in_progress = False
        self.initialization_time = self.cfg.initialization_time
        self.replanning_needed = True  # For shared memory mode
        # TODO: Handle replanning needed later

        # Handling map visualization
        self.debugging_visualization = self.cfg.get('debugging_visualization', False)
        self.map_visualization_interval = self.cfg.get('map_visualization_interval', 1.0)
        self._last_map_visualization_time = 0.0

        self.get_logger().info(f"[PathPlanner] Planning Node initialized with {self.cfg.trav_option} map type!")
    





    def plan_and_publish(self):
        """Run RRT planning and publish the path."""
        if not self.global_map.is_TraversabilityMap_built:
            self.get_logger().warning("[PathPlanner] Map not built yet, cannot plan")
            return

        self.planning_in_progress = True

        try:
            ros_t = self.get_clock().now().nanoseconds / 1e9
            rx, ry, rh = self.global_map.robot_x, self.global_map.robot_y, self.global_map.robot_heading
            self.get_logger().info("=" * 72)
            self.get_logger().info(
                f"[PathPlanner] Replanning | t={ros_t:.2f}s | "
                f"robot=({rx:.2f}, {ry:.2f}, {np.rad2deg(rh):.1f}deg)"
            )

            self.global_planner.replan(
                initial_start=(self.initial_start[0], self.initial_start[1]),
                total_lookahead_time=self.waypoint_lookahead_time,
                plot_map = False,
            )

            self.publish_path()
            
            # NOTE: Uncomment this to visualize other traversability maps
            self._visualize_planning_maps(ros_t)

            self.last_plan_time = self.get_clock().now().nanoseconds / 1e9

        except Exception as e:
            self.get_logger().error("[PathPlanner] Error during planning: %s" % str(e))
            import traceback
            traceback.print_exc()
        finally:
            self.planning_in_progress = False




####################################### PUBLISHING #######################################

    @staticmethod
    def _path_to_xy_list(path):
        """Convert planner path (Node objects or xy tuples) to [[x, y], ...]."""
        xy_path = []
        for pt in path:
            if hasattr(pt, 'x') and hasattr(pt, 'y'):
                xy_path.append([pt.x, pt.y])
            else:
                xy_path.append([float(pt[0]), float(pt[1])])
        return xy_path

    def _visualize_planning_maps(self, current_time):
        """Show elevation + traversability (cmd) maps with the planned path overlay."""
        if not self.debugging_visualization:
            return
        if current_time - self._last_map_visualization_time < self.map_visualization_interval:
            return
        if len(self.global_planner.path) == 0:
            return
        if not hasattr(self.global_map, 'visualize_maps'):
            return

        self._last_map_visualization_time = current_time
        path_xy = self._path_to_xy_list(self.global_planner.path)
        self.global_map.visualize_maps(self.global_map.robot_heading, path_xy)

    def publish_path(self):
        """Publish the planned path as a ROS Path message."""
        if len(self.global_planner.path) == 0:
            self.get_logger().warning("[PathPlanner] No path to publish")
            return
        
        global_path_msg = Path()
        global_path_msg.header.frame_id = self.frame_id
        global_path_msg.header.stamp = self.get_clock().now().to_msg()
        
        # Handle different path formats
        for pt in self.global_planner.path:
            pose = PoseStamped()
            pose.header.frame_id = self.frame_id
            pose.header.stamp = global_path_msg.header.stamp
            
            # Handle both tuple/list format and Node object format
            if hasattr(pt, 'x') and hasattr(pt, 'y'):
                # Node object (from shared memory mode)
                pose.pose.position.x = pt.x
                pose.pose.position.y = pt.y
            else:
                # Tuple/list format (from ROS message mode)
                pose.pose.position.x = float(pt[0])
                pose.pose.position.y = float(pt[1])
            
            # Get elevation z for this waypoint from the elevation map; fallback to 1.0
            z = 1.0
            if self.global_map.is_ElevationMap_built and self.global_map.ElevationMap is not None:
                row, col = self.global_map.xy2grid(pose.pose.position.x, pose.pose.position.y)
                if row is not None and col is not None:
                    elev = self.global_map.ElevationMap[row, col]
                    if not np.isnan(elev):
                        z = float(elev)
            pose.pose.position.z = z
            pose.pose.orientation.w = 1.0
            global_path_msg.poses.append(pose)
        
        self.global_path_pub.publish(global_path_msg)
        # self.get_logger().info("Published path with %d waypoints", len(self.global_planner.path))



####################################### CALLBACKS #######################################

    def pose_callback(self, msg):
        """Update robot pose from PoseStamped message."""
        p = msg.pose.position
        q = msg.pose.orientation
        
        yaw = np.arctan2(
            2 * (q.w * q.z + q.x * q.y),
            1 - 2 * (q.y * q.y + q.z * q.z)
        )
        
        self.global_map.robot_x = p.x
        self.global_map.robot_y = p.y
        self.global_map.robot_heading = wrap_to_pi(yaw)
    
    def replanning_callback(self, msg):
        """Callback for replanning signal via ROS topic"""
        if msg.data:
            self.replanning_needed = True

    def map_callback(self, msg):
        """Handle incoming map message and trigger planning (ROS message mode only)."""
        if self.use_shared_memory:
            self.get_logger().warning("[PathPlanner] map_callback called in shared memory mode. This should not happen.")
            return
        
        if self.planning_in_progress:
            # Note: ROS2 throttling is done differently - using timer-based approach or manual check
            if not hasattr(self, '_last_warning_time'):
                self._last_warning_time = 0.0
            current_time = self.get_clock().now().nanoseconds / 1e9
            if current_time - self._last_warning_time > 1.0:
                self.get_logger().warning("[PathPlanner] Planning in progress, skipping map update")
                self._last_warning_time = current_time
            return

        if self.get_clock().now().nanoseconds / 1e9 < self.initialization_time:
            return
        
        # Update map data using helper functions
        success = False
        if self.use_gridmap_msg:
            success = populate_map_from_gridmap(self.global_map, msg, node=self)
        else:
            success = populate_map_from_float32multiarray(self.global_map, msg, node=self)
        
        if not success:
            self.get_logger().warning("[PathPlanner] Failed to update map from message")
            return

        # Trigger planning for each map update
        self.plan_and_publish()
    
    





####################################### RUN #######################################

    def run(self):
        # Spin in a background thread so ROS callbacks are always processed
        spin_thread = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)
        spin_thread.start()

        if self.use_shared_memory:
            self.get_logger().info("[PathPlanner] Planning Node started (shared memory mode). Waiting for maps and replanning signals...")
            planning_rate = self.create_rate(1)  # 1 Hz

            self._last_wait_log_time = 0.0

            while rclpy.ok():

                if not self.global_map.is_TraversabilityMap_built:
                    current_time = self.get_clock().now().nanoseconds / 1e9
                    if current_time - self._last_wait_log_time > 5.0:
                        self.get_logger().info("[PathPlanner] Waiting for maps to be built...")
                        self._last_wait_log_time = current_time
                    planning_rate.sleep()
                    continue

                if self.replanning_needed and not self.planning_in_progress:
                    self.plan_and_publish()
                    self.replanning_needed = False

                planning_rate.sleep()
        else:
            self.get_logger().info("[PathPlanner] Planning Node started (ROS message mode). Waiting for map updates...")
            # plan_and_publish is called in map_callback; just keep alive until shutdown
            spin_thread.join()

def main():
    rclpy.init()
    try:
        node = PlanningNode(use_sim_time=True)
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
