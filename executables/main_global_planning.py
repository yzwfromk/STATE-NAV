from time import time
import numpy as np
import math
from omegaconf import OmegaConf

import statenav_global
from statenav_global import CMDbasedMap, LearnedInSMap, IHMCMap, QuadrupedMap



import transforms3d as tf3
import tf.transformations as tf_trans





import rospy
from std_msgs.msg import Float32MultiArray
from grid_map_msgs.msg import GridMap
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

from pathlib import Path as PathLib
cfg = OmegaConf.load(PathLib(__file__).parents[0] / "configs/planning_config.yaml")


def set_random_seed(seed):
    rng = np.random.RandomState(seed)
    print(f"Set random seed to {seed} in numpy.")
    return rng


def wrap_to_pi(angle):
    """Wraps an angle to the range [-π, π] using atan2."""
    return np.arctan2(np.sin(angle), np.cos(angle))






















################################################### Helper Functions for ROS ###################################################





def populate_map_from_float32multiarray(global_map, msg):
    """Populate existing map object from Float32MultiArray message."""
    if len(msg.data) < 8:
        rospy.logwarn("Map message too short, skipping")
        return False
        
    data = msg.data
    idx = 0
        
    if not global_map.is_TraversabilityMap_built: # If the map is not built, extract the metadata
        # Extract metadata
        global_map.map_resolution = data[idx]; idx += 1
        global_map.TraversabilityMap_theta_resolution = data[idx]; idx += 1
        global_map.TraversabilityMap_xmin = data[idx]; idx += 1
        global_map.TraversabilityMap_xmax = data[idx]; idx += 1
        global_map.TraversabilityMap_ymin = data[idx]; idx += 1
        global_map.TraversabilityMap_ymax = data[idx]; idx += 1
        global_map.TraversabilityMap_theta_min = data[idx]; idx += 1
        global_map.TraversabilityMap_theta_max = data[idx]; idx += 1
        
        # Calculate dimensions
        global_map.TraversabilityMap_size_rows = int((global_map.TraversabilityMap_xmax - global_map.TraversabilityMap_xmin) / global_map.map_resolution)
        global_map.TraversabilityMap_size_cols = int((global_map.TraversabilityMap_ymax - global_map.TraversabilityMap_ymin) / global_map.map_resolution)
        global_map.TraversabilityMap_size_layers = int(2 * np.round(np.pi / global_map.TraversabilityMap_theta_resolution))

    else:
        idx = 8 # Skip the metadata but keep the data index
        
    # Calculate expected data size
    expected_size = global_map.TraversabilityMap_size_rows * global_map.TraversabilityMap_size_cols * global_map.TraversabilityMap_size_layers * 4
    actual_size = len(data) - idx
    
    if actual_size < expected_size:
        rospy.logwarn(f"Map data incomplete. Expected {expected_size}, got {actual_size}")
        return False
    
    # Reshape data into 4D array [rows, cols, theta_layers, 4_channels]
    map_data = np.array(data[idx:idx+expected_size], dtype=np.float32)
    global_map.TraversabilityMap = map_data.reshape(
        (global_map.TraversabilityMap_size_rows, 
         global_map.TraversabilityMap_size_cols, 
         global_map.TraversabilityMap_size_layers, 
         4)
    )
    
    global_map.is_TraversabilityMap_built = True # Avoiding unnecessary initialization of the map. Assuming the map metadata does not change.
    # rospy.loginfo(f"Map updated: {global_map.TraversabilityMap_size_rows}x{global_map.TraversabilityMap_size_cols}x{global_map.TraversabilityMap_size_layers}")
    return True


def populate_map_from_gridmap(global_map, msg):
    """Populate existing map object from GridMap message."""
    # Extract metadata from GridMap info
    global_map.map_resolution = msg.info.resolution
    length_x = msg.info.length_x
    length_y = msg.info.length_y
    
    # Calculate map bounds from center position
    center_x = msg.info.pose.position.x
    center_y = msg.info.pose.position.y
    
    global_map.TraversabilityMap_xmin = center_x - length_x / 2.0
    global_map.TraversabilityMap_xmax = center_x + length_x / 2.0
    global_map.TraversabilityMap_ymin = center_y - length_y / 2.0
    global_map.TraversabilityMap_ymax = center_y + length_y / 2.0
    
    # Get dimensions from first layer
    if len(msg.layers) == 0 or len(msg.data) == 0:
        rospy.logwarn("GridMap has no layers")
        return False
    
    first_layer = msg.data[0]
    if len(first_layer.layout.dim) < 2:
        rospy.logwarn("GridMap layer has insufficient dimensions")
        return False
    
    # Note: GridMap uses column_index, row_index order
    num_cols = first_layer.layout.dim[0].size
    num_rows = first_layer.layout.dim[1].size
    
    global_map.TraversabilityMap_size_rows = num_rows
    global_map.TraversabilityMap_size_cols = num_cols
    
    # Extract theta information from metadata layers
    theta_resolution = None
    theta_min = None
    theta_max = None
    num_theta_layers = None
    
    for i, layer_name in enumerate(msg.layers):
        if i < len(msg.data):
            if layer_name == "theta_resolution":
                theta_resolution = msg.data[i].data[0]
            elif layer_name == "theta_min":
                theta_min = msg.data[i].data[0]
            elif layer_name == "theta_max":
                theta_max = msg.data[i].data[0]
            elif layer_name == "num_theta_layers":
                num_theta_layers = int(msg.data[i].data[0])
    
    # If metadata not found, infer from layer names
    if theta_resolution is None:
        theta_indices = set()
        for layer_name in msg.layers:
            if "_theta_" in layer_name:
                try:
                    theta_idx = int(layer_name.split("_theta_")[-1])
                    theta_indices.add(theta_idx)
                except:
                    pass
        
        if len(theta_indices) > 0:
            num_theta_layers = len(theta_indices)
            if theta_min is None:
                theta_min = -np.pi
            if theta_max is None:
                theta_max = np.pi - (np.pi / num_theta_layers)
            if theta_resolution is None:
                theta_resolution = (theta_max - theta_min) / (num_theta_layers - 1) if num_theta_layers > 1 else np.pi / 4
    
    if num_theta_layers is None:
        rospy.logwarn("Could not determine number of theta layers, defaulting to 8")
        num_theta_layers = 8
        theta_min = -np.pi
        theta_max = np.pi - np.pi/4
        theta_resolution = np.pi / 4
    
    global_map.TraversabilityMap_size_layers = num_theta_layers
    global_map.TraversabilityMap_theta_min = theta_min if theta_min is not None else -np.pi
    global_map.TraversabilityMap_theta_max = theta_max if theta_max is not None else (np.pi - np.pi/4)
    global_map.TraversabilityMap_theta_resolution = theta_resolution if theta_resolution is not None else np.pi/4
    
    # Reconstruct 4D array from GridMap layers
    channel_names = ["cmd_v", "cmd_w", "auxiliary_score", "auxiliary_score_std"]
    
    global_map.TraversabilityMap = np.full(
        (num_rows, num_cols, num_theta_layers, 4), 
        np.nan, 
        dtype=np.float32
    )
    
    layer_idx = 0
    for theta_layer in range(num_theta_layers):
        for channel_idx, channel_name in enumerate(channel_names):
            layer_name = f"{channel_name}_theta_{theta_layer}"
            
            if layer_idx < len(msg.layers) and layer_idx < len(msg.data):
                if msg.layers[layer_idx] == layer_name:
                    layer_data = np.array(msg.data[layer_idx].data, dtype=np.float32)
                    
                    # Reshape from column-major to row-major
                    layer_data_reshaped = layer_data.reshape((num_cols, num_rows)).T
                    
                    global_map.TraversabilityMap[:, :, theta_layer, channel_idx] = layer_data_reshaped
                    layer_idx += 1
                else:
                    rospy.logwarn(f"Expected layer {layer_name}, got {msg.layers[layer_idx]}")
                    layer_idx += 1
    
    global_map.is_TraversabilityMap_built = True
    # rospy.loginfo(f"Map updated from GridMap: {num_rows}x{num_cols}x{num_theta_layers}")
    return True
































# Todos: Make the global_map a shared memory between the main_traversability_estimation.py and main_global_planning.py



class PlanningNode:
    """Main planning node that subscribes to map and publishes paths."""
    
    def __init__(self):
        # Load configuration
        self.cfg = cfg
        
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
        
        if self.cfg.trav_option == "Proposed" and self.cfg.planner_option == "safecmd":
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
                load_NN = False
            )
        elif self.cfg.trav_option == "Proposed" and self.cfg.planner_option == "score":
            self.global_map = LearnedInSMap(
                env_xmin=self.cfg.env_extent[0], 
                env_xmax=self.cfg.env_extent[1], 
                env_ymin=self.cfg.env_extent[2], 
                env_ymax=self.cfg.env_extent[3],
                goal_x=global_goal[0], 
                goal_y=global_goal[1],
                which_layer=self.cfg.which_layer,
                preest_update_resolution=self.cfg.trav_estimation_resoultion, 
                instab_limit=self.cfg.instability_limit,
                load_NN = False
            )
        else:
            raise ValueError(f"Invalid trav_option or planner_option: {self.cfg.trav_option} or {self.cfg.planner_option}")
        
        rospy.loginfo(f"Map class initialized. Type: {type(self.global_map).__name__}")
        
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
            default_obstacle_clearance=self.cfg.obs_clearance
        )
        self.global_planner.global_map = self.global_map
        
        # ROS setup
        rospy.init_node('Planning_Node', anonymous=True)
        
        # Check message type
        self.use_gridmap_msg = self.cfg.get('use_gridmap_msg', False)
        self.frame_id = self.cfg.get('frame_id', 'map')
        
        # Subscribers
        if self.use_gridmap_msg:
            rospy.Subscriber("/global_costmap", GridMap, self.map_callback, queue_size=1)
            rospy.loginfo("Subscribed to /global_costmap (GridMap)")
        else:
            rospy.Subscriber("/global_costmap", Float32MultiArray, self.map_callback, queue_size=1)
            rospy.loginfo("Subscribed to /global_costmap (Float32MultiArray)")
        
        rospy.Subscriber("/robot/pose", PoseStamped, self.pose_callback, queue_size=1)
        rospy.loginfo("Subscribed to /robot/pose")

        print(self.global_map.robot_heading)
        
        # Publishers
        self.global_path_pub = rospy.Publisher("/global_path", Path, queue_size=1)
        rospy.loginfo("Publisher initialized: /global_path")
        
        # For getting next waypoint for resetting the tree
        self.step_T = self.cfg.step_T
        self.RRT_getwaypoint_steps = self.cfg.RRT_getwaypoint_steps
        
        # State
        self.last_plan_time = 0
        self.planning_in_progress = False
        self.initialization_time = self.cfg.initialization_time
        
        rospy.loginfo(f"Planning Node initialized with {self.cfg.trav_option}/{self.cfg.planner_option} map type!")
    
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


    
    def map_callback(self, msg):
        """Handle incoming map message and trigger planning."""
        if self.planning_in_progress:
            rospy.logwarn_throttle(1.0, "Planning in progress, skipping map update")
            return
        
        if rospy.get_time() < self.initialization_time:
            return
        
        # Update map data using helper functions
        success = False
        if self.use_gridmap_msg:
            success = populate_map_from_gridmap(self.global_map, msg)
        else:
            success = populate_map_from_float32multiarray(self.global_map, msg)
        
        if not success:
            rospy.logwarn("Failed to update map from message")
            return

        # Trigger planning for each map update
        self.plan_and_publish()
    
    def plan_and_publish(self):
        """Run RRT planning and publish the path."""
        if not self.global_map.is_TraversabilityMap_built:
            rospy.logwarn("Map not built yet, cannot plan")
            return
        
        self.planning_in_progress = True
        
        try:
            
            self.global_planner.replan(initial_start=(self.initial_start[0], self.initial_start[1]), step_T=self.step_T, RRT_getwaypoint_steps=self.RRT_getwaypoint_steps, plot_map=False)
            # Publish path
            self.publish_path()
            
            self.last_plan_time = rospy.get_time()
            
        except Exception as e:
            rospy.logerr("Error during planning: %s", str(e))
            import traceback
            traceback.print_exc()
        finally:
            self.planning_in_progress = False
    
    def publish_path(self):
        """Publish the planned path as a ROS Path message."""
        if len(self.global_planner.path) == 0:
            rospy.logwarn("No path to publish")
            return
        
        global_path_msg = Path()
        global_path_msg.header.frame_id = self.frame_id
        global_path_msg.header.stamp = rospy.Time.now()
        
        for pt in self.global_planner.path:
            pose = PoseStamped()
            pose.header.frame_id = self.frame_id
            pose.header.stamp = global_path_msg.header.stamp
            pose.pose.position.x = pt[0]
            pose.pose.position.y = pt[1]
            pose.pose.position.z = 1.0
            pose.pose.orientation.w = 1.0
            global_path_msg.poses.append(pose)
        
        self.global_path_pub.publish(global_path_msg)
        # rospy.loginfo("Published path with %d waypoints", len(self.global_planner.path))
    
    def run(self):

       rospy.loginfo("Planning Node started. Waiting for map updates...")
       
       planning_rate = rospy.Rate(1)  # 1 Hz rate
       
       while not rospy.is_shutdown():
           
            # self.plan_and_publish()
            # since plan_and_publish is called in the map_callback, we don't need to call it here
            # if we call it here, threading and locking issues need to be handled.
            planning_rate.sleep()

def main():
    try:
        node = PlanningNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        rospy.signal_shutdown("Finished execution")


if __name__ == '__main__':
    main()