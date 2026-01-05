from time import time
import numpy as np
import torch
import os
import math
from omegaconf import OmegaConf
import psutil

import statenav_global
from statenav_global.mapping import *

import transforms3d as tf3


import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Float32MultiArray, MultiArrayLayout, MultiArrayDimension
from grid_map_msgs.msg import GridMap



import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)



from pathlib import Path as PathLib
cfg = OmegaConf.load(PathLib(__file__).parents[0] / "configs/planning_config.yaml")


def set_random_seed(seed):
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    print(f"Set random seed to {seed} in numpy and torch.")
    return rng

def wrap_to_pi(angle):
    """Wraps an angle to the range [-π, π] using atan2."""
    return np.arctan2(np.sin(angle), np.cos(angle))











################################################### Helper Functions for ROS ###################################################



def publish_costmap_float32multiarray(global_map, frame_id, global_costmap_pub, pub_locally, global_planner, 
                                       step_T, localmap_getwaypoint_horizonmultiplier, MPC_horizon):
    """
    Publish costmap using Float32MultiArray (original method).
    """
    global_costmap_topic = Float32MultiArray()

    if pub_locally:
        [cmd_v, cmd_w] = global_map.get_cmd_limits()
        waypoint = global_planner.get_waypoint(global_map.robot_x, global_map.robot_y, 
                                                global_map.robot_heading, step_T, 
                                                localmap_getwaypoint_horizonmultiplier*MPC_horizon, 
                                                cmd_v, cmd_w)
        xmin = np.min([global_map.robot_x, waypoint[0]])
        xmax = np.max([global_map.robot_x, waypoint[0]])
        ymin = np.min([global_map.robot_y, waypoint[1]])
        ymax = np.max([global_map.robot_y, waypoint[1]])
        padding = 1
    else:
        xmin = global_map.TraversabilityMap_xmin
        xmax = global_map.TraversabilityMap_xmax
        ymin = global_map.TraversabilityMap_ymin
        ymax = global_map.TraversabilityMap_ymax
        padding = 0

    xmin_padded = np.max([xmin - padding, global_map.TraversabilityMap_xmin])
    xmax_padded = np.min([xmax + padding, global_map.TraversabilityMap_xmax])
    ymin_padded = np.max([ymin - padding, global_map.TraversabilityMap_ymin])
    ymax_padded = np.min([ymax + padding, global_map.TraversabilityMap_ymax])
    row_xminpadded = int((global_map.TraversabilityMap_xmax - xmin_padded)/global_map.map_resolution)
    row_xmaxpadded = int((global_map.TraversabilityMap_xmax - xmax_padded)/global_map.map_resolution)
    col_yminpadded = int((global_map.TraversabilityMap_ymax - ymin_padded)/global_map.map_resolution)
    col_ymaxpadded = int((global_map.TraversabilityMap_ymax - ymax_padded)/global_map.map_resolution)

    # Boundary checks
    if row_xminpadded >= global_map.TraversabilityMap_size_rows:
        row_xminpadded = global_map.TraversabilityMap_size_rows-1
        xmin_padded = global_map.TraversabilityMap_xmin
    if row_xmaxpadded < 0:
        row_xmaxpadded = 0
        xmax_padded = global_map.TraversabilityMap_xmax
    if col_yminpadded >= global_map.TraversabilityMap_size_cols:
        col_yminpadded = global_map.TraversabilityMap_size_cols - 1
        ymin_padded = global_map.TraversabilityMap_ymin
    if col_ymaxpadded < 0:
        col_ymaxpadded = 0
        ymax_padded = global_map.TraversabilityMap_ymax

    xmax_padded = (row_xminpadded - row_xmaxpadded) * global_map.map_resolution + xmin_padded + 1e-4
    ymax_padded = (col_yminpadded - col_ymaxpadded) * global_map.map_resolution + ymin_padded + 1e-4

    # Add metadata
    global_costmap_topic.data.append(global_map.map_resolution)
    global_costmap_topic.data.append(global_map.TraversabilityMap_theta_resolution)
    global_costmap_topic.data.append(xmin_padded)
    global_costmap_topic.data.append(xmax_padded)
    global_costmap_topic.data.append(ymin_padded)
    global_costmap_topic.data.append(ymax_padded)
    global_costmap_topic.data.append(global_map.TraversabilityMap_theta_min)
    global_costmap_topic.data.append(global_map.TraversabilityMap_theta_max)

    # Add data
    for i in range(row_xmaxpadded, row_xminpadded):
        for j in range(col_ymaxpadded, col_yminpadded):
            for k in range(global_map.TraversabilityMap_size_layers):
                global_costmap_topic.data.append(global_map.TraversabilityMap[i,j,k,0])
                global_costmap_topic.data.append(global_map.TraversabilityMap[i,j,k,1])
                global_costmap_topic.data.append(global_map.TraversabilityMap[i,j,k,2])
                global_costmap_topic.data.append(global_map.TraversabilityMap[i,j,k,3])

    global_costmap_pub.publish(global_costmap_topic)




def publish_costmap_gridmap(global_map, frame_id, global_costmap_pub, pub_locally, global_planner,
                            step_T, localmap_getwaypoint_horizonmultiplier, MPC_horizon):
    """
    Publish costmap using GridMap message type.
    """
    # Calculate region bounds (same logic as Float32MultiArray)
    if pub_locally:
        [cmd_v, cmd_w] = global_map.get_cmd_limits()
        waypoint = global_planner.get_waypoint(global_map.robot_x, global_map.robot_y,
                                                global_map.robot_heading, step_T,
                                                localmap_getwaypoint_horizonmultiplier*MPC_horizon,
                                                cmd_v, cmd_w)
        xmin = np.min([global_map.robot_x, waypoint[0]])
        xmax = np.max([global_map.robot_x, waypoint[0]])
        ymin = np.min([global_map.robot_y, waypoint[1]])
        ymax = np.max([global_map.robot_y, waypoint[1]])
        padding = 1
    else:
        xmin = global_map.TraversabilityMap_xmin
        xmax = global_map.TraversabilityMap_xmax
        ymin = global_map.TraversabilityMap_ymin
        ymax = global_map.TraversabilityMap_ymax
        padding = 0

    xmin_padded = np.max([xmin - padding, global_map.TraversabilityMap_xmin])
    xmax_padded = np.min([xmax + padding, global_map.TraversabilityMap_xmax])
    ymin_padded = np.max([ymin - padding, global_map.TraversabilityMap_ymin])
    ymax_padded = np.min([ymax + padding, global_map.TraversabilityMap_ymax])
    row_xminpadded = int((global_map.TraversabilityMap_xmax - xmin_padded)/global_map.map_resolution)
    row_xmaxpadded = int((global_map.TraversabilityMap_xmax - xmax_padded)/global_map.map_resolution)
    col_yminpadded = int((global_map.TraversabilityMap_ymax - ymin_padded)/global_map.map_resolution)
    col_ymaxpadded = int((global_map.TraversabilityMap_ymax - ymax_padded)/global_map.map_resolution)

    # Boundary checks
    if row_xminpadded >= global_map.TraversabilityMap_size_rows:
        row_xminpadded = global_map.TraversabilityMap_size_rows-1
        xmin_padded = global_map.TraversabilityMap_xmin
    if row_xmaxpadded < 0:
        row_xmaxpadded = 0
        xmax_padded = global_map.TraversabilityMap_xmax
    if col_yminpadded >= global_map.TraversabilityMap_size_cols:
        col_yminpadded = global_map.TraversabilityMap_size_cols - 1
        ymin_padded = global_map.TraversabilityMap_ymin
    if col_ymaxpadded < 0:
        col_ymaxpadded = 0
        ymax_padded = global_map.TraversabilityMap_ymax

    xmax_padded = (row_xminpadded - row_xmaxpadded) * global_map.map_resolution + xmin_padded + 1e-4
    ymax_padded = (col_yminpadded - col_ymaxpadded) * global_map.map_resolution + ymin_padded + 1e-4

    # Calculate grid dimensions
    num_rows = row_xminpadded - row_xmaxpadded
    num_cols = col_yminpadded - col_ymaxpadded

    # Initialize GridMap message
    grid_map_msg = GridMap()
    grid_map_msg.info.header.frame_id = frame_id
    grid_map_msg.info.header.stamp = rospy.Time.now()

    # Set GridMap metadata
    grid_map_msg.info.resolution = global_map.map_resolution
    grid_map_msg.info.length_x = xmax_padded - xmin_padded
    grid_map_msg.info.length_y = ymax_padded - ymin_padded
    grid_map_msg.info.pose.position.x = xmin_padded + (xmax_padded - xmin_padded) / 2.0
    grid_map_msg.info.pose.position.y = ymin_padded + (ymax_padded - ymin_padded) / 2.0
    grid_map_msg.info.pose.position.z = 0.0
    grid_map_msg.info.pose.orientation.w = 1.0

    # Channel names for the 4 data channels
    channel_names = ["cmd_v", "cmd_w", "auxiliary_score", "auxiliary_score_std"]

    # Create layers for each (theta_layer, channel) combination
    for theta_layer in range(global_map.TraversabilityMap_size_layers):
        theta_value = global_map.TraversabilityMap_theta_min + theta_layer * global_map.TraversabilityMap_theta_resolution
        
        for channel_idx, channel_name in enumerate(channel_names):
            # Create layer name with theta index
            layer_name = f"{channel_name}_theta_{theta_layer}"
            grid_map_msg.layers.append(layer_name)

            # Create Float32MultiArray for this layer
            layer_data = Float32MultiArray()
            
            # Set up layout dimensions (row-major order)
            layout = MultiArrayLayout()
            
            # Height dimension (rows)
            dim_row = MultiArrayDimension()
            dim_row.label = "row_index"
            dim_row.size = num_rows
            dim_row.stride = 1
            
            # Width dimension (cols)
            dim_col = MultiArrayDimension()
            dim_col.label = "column_index"
            dim_col.size = num_cols
            dim_col.stride = num_rows
            
            layout.dim = [dim_col, dim_row]
            layout.data_offset = 0
            layer_data.layout = layout

            # Extract and append data for this layer (row-major order)
            for j in range(col_ymaxpadded, col_yminpadded):
                for i in range(row_xmaxpadded, row_xminpadded):
                    value = global_map.TraversabilityMap[i, j, theta_layer, channel_idx]
                    layer_data.data.append(float(value))

            grid_map_msg.data.append(layer_data)

    # Add metadata layers for theta information
    # These can be used by subscribers to understand theta layer structure
    metadata_layer_names = [
        "theta_resolution",
        "theta_min", 
        "theta_max",
        "num_theta_layers",
        "xmin",
        "xmax",
        "ymin",
        "ymax"
    ]
    metadata_values = [
        global_map.TraversabilityMap_theta_resolution,
        global_map.TraversabilityMap_theta_min,
        global_map.TraversabilityMap_theta_max,
        float(global_map.TraversabilityMap_size_layers),
        xmin_padded,
        xmax_padded,
        ymin_padded,
        ymax_padded
    ]

    # Create single-value metadata layers (repeated for each cell for consistency)
    for meta_idx, meta_name in enumerate(metadata_layer_names):
        grid_map_msg.layers.append(meta_name)
        meta_layer_data = Float32MultiArray()
        meta_layout = MultiArrayLayout()
        
        meta_dim_row = MultiArrayDimension()
        meta_dim_row.label = "row_index"
        meta_dim_row.size = num_rows
        meta_dim_row.stride = 1
        
        meta_dim_col = MultiArrayDimension()
        meta_dim_col.label = "column_index"
        meta_dim_col.size = num_cols
        meta_dim_col.stride = num_rows
        
        meta_layout.dim = [meta_dim_col, meta_dim_row]
        meta_layout.data_offset = 0
        meta_layer_data.layout = meta_layout
        
        # Fill with constant metadata value
        meta_value = metadata_values[meta_idx]
        for j in range(num_cols):
            for i in range(num_rows):
                meta_layer_data.data.append(float(meta_value))
        
        grid_map_msg.data.append(meta_layer_data)

    # Publish
    global_costmap_pub.publish(grid_map_msg)















def main():

    os.nice(-10)
    

    home_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__name__)) , os.pardir))
    rng = set_random_seed(cfg.seed)

    #Environment Properties
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)


    


    cfg.task_extent = [cfg.env_extent[0] + 1*cfg.local_patch_size, cfg.env_extent[1] - 1*cfg.local_patch_size, cfg.env_extent[2] + 1*cfg.local_patch_size, cfg.env_extent[3] - 1*cfg.local_patch_size ]
    diagonal = math.hypot(cfg.task_extent[1] - cfg.task_extent[0], cfg.task_extent[3] - cfg.task_extent[2])
   
    initial_start = cfg.initial_start
    global_goal = cfg.global_goal
    heading_start =  np.deg2rad(cfg.heading_start)

    # Global Map
    if cfg.trav_option == "Proposed" and cfg.planner_option == "safecmd":
        global_map = CMDbasedMap(env_xmin=cfg.env_extent[0], env_xmax=cfg.env_extent[1], env_ymin=cfg.env_extent[2], env_ymax=cfg.env_extent[3],
                            goal_x=global_goal[0], goal_y=global_goal[1],
                            which_layer = cfg.which_layer,
                            preest_update_resolution=cfg.trav_estimation_resoultion, instab_limit = cfg.instability_limit) # 0.37
    elif cfg.trav_option == "Proposed" and cfg.planner_option == "score":
        global_map = LearnedInSMap(env_xmin=cfg.env_extent[0], env_xmax=cfg.env_extent[1], env_ymin=cfg.env_extent[2], env_ymax=cfg.env_extent[3],
                            goal_x=global_goal[0], goal_y=global_goal[1],
                            which_layer = cfg.which_layer,
                            preest_update_resolution=cfg.trav_estimation_resoultion, instab_limit = cfg.instability_limit) # 0.37
    else:
        raise ValueError(f"Invalid trav_option: {cfg.trav_option}")

    initialization_time = cfg.initialization_time




    ############################## ROS INITIALIZATION ########################################
    print("Initializing... ROS Node")
    rospy.init_node('Global_Planner', anonymous=True)
    rate = rospy.Rate(1) 

    robot_pose_listner = rospy.Subscriber("/robot/pose", PoseStamped, global_map.pose_callback, queue_size=1) # warning: timestamp is not correct
    robocentric_map_listner = rospy.Subscriber("/elevation_mapping/elevation_map_filter", GridMap, global_map.robo_centric_map_callback,  queue_size=1)

    use_gridmap_msg = cfg.get('use_gridmap_msg', False)  # Default to False for backward compatibility
    frame_id = cfg.frame_id

    do_RRT_globalplanning = cfg.do_RRT_globalplanning
    asynchronous_globalplanning = cfg.asynchronous_globalplanning
    if not do_RRT_globalplanning and asynchronous_globalplanning:
        raise ValueError("asynchronous_globalplanning can be True only when do_RRT_globalplanning is True")


    pub_locally = cfg.pub_locally
    if asynchronous_globalplanning and pub_locally:
        raise ValueError("pub_locally is only supported when asynchronous_globalplanning is False")
    if not do_RRT_globalplanning and pub_locally:
        raise ValueError("pub_locally is only supported when do_RRT_globalplanning is True")
    localmap_getwaypoint_horizonmultiplier = cfg.localmap_getwaypoint_horizonmultiplier
    RRT_getwaypoint_steps = cfg.RRT_getwaypoint_steps
    MPC_horizon = cfg.MPC_horizon
    step_T = cfg.step_T

    debugging_visualization = cfg.debugging_visualization

    ############################## ROS Publisher Setup ##############################
    if use_gridmap_msg:
        global_costmap_pub = rospy.Publisher("/global_costmap", GridMap, queue_size=1)
        print("Using GridMap message type for costmap publishing")
    else:
        global_costmap_pub = rospy.Publisher("/global_costmap", Float32MultiArray, queue_size=1)
        print("Using Float32MultiArray message type for costmap publishing")

    global_path_pub = rospy.Publisher("/global_path", Path, queue_size=1)
    obstacle_list_pub = rospy.Publisher("/obstacle_list", Float32MultiArray, queue_size=1)
    print("Initialized! ROS Node")





    ############################## Global Planner Configurations ##############################
    iter_max_global = cfg.iter_max
    branch_length_max = cfg.branch_length_max_ratio * diagonal
    search_radius = cfg.search_radius_ratio * diagonal
    decrease_search_radius = True

    if do_RRT_globalplanning and not asynchronous_globalplanning:
        global_planner = statenav_global.planners.GlobalRRTStar(
                    cfg.task_extent, rng, initial_start, global_goal, heading_start,
                    goal_radius = diagonal * cfg.goal_radius_ratio, branch_length_max=branch_length_max, search_radius=search_radius, decrease_search_radius=decrease_search_radius, \
                    iter_max=iter_max_global, convergence_threshold=cfg.convergence_ratio,
                    switch_to_informed_from_thisiter = cfg.switch_to_informed_from_thisiter, sampling_dist=cfg.sampling_dist, num_samplingpoints=cfg.num_samplingpoints,
                    default_obstacle_clearance=cfg.obs_clearance)
        global_planner.global_map = global_map # pass by reference

        print("Initialized planner GlobalRRTStar")
    else:
        global_planner = None
    











    ############################## Main Loop ##############################
    count = 0
    while not rospy.is_shutdown():

        count += 1
        rospy.loginfo_throttle(1.0, "\n==================================== LOOP ====================================")
        rospy.loginfo_throttle(1.0, "ROSPY in loop. ROSTime is, %f and count is %d", rospy.get_time(), count)

        # ==================================== ROS  ====================================

        if rospy.get_time() > initialization_time:

            if do_RRT_globalplanning and not asynchronous_globalplanning:
                global_map.Update_map(global_planner.path, visualize_map=debugging_visualization)
            else:
                global_map.Update_map(visualize_map=debugging_visualization)


            if global_map.is_TraversabilityMap_built:
                
                if do_RRT_globalplanning and not asynchronous_globalplanning:
                    global_planner.replan(initial_start=(initial_start[0], initial_start[1]), step_T=step_T, RRT_getwaypoint_steps=RRT_getwaypoint_steps, plot_map=False)

                # ==================================== ROS Publishing ====================================

                # Publish costmap (toggle between Float32MultiArray and GridMap)
                if use_gridmap_msg:
                    publish_costmap_gridmap(global_map, frame_id, global_costmap_pub, pub_locally, global_planner,
                                            step_T, localmap_getwaypoint_horizonmultiplier, MPC_horizon)
                else:
                    publish_costmap_float32multiarray(global_map, frame_id, global_costmap_pub, pub_locally, global_planner,
                                                    step_T, localmap_getwaypoint_horizonmultiplier, MPC_horizon)


                obstacle_list_topic = Float32MultiArray()
                for obs in global_map.obs_list:
                    obstacle_list_topic.data.append(obs[0])
                    obstacle_list_topic.data.append(obs[1])
                obstacle_list_pub.publish(obstacle_list_topic)


                if do_RRT_globalplanning and not asynchronous_globalplanning:


                    global_path_msg = Path()
                    global_path_msg.header.frame_id = frame_id
                    global_path_msg.header.stamp = rospy.Time.now()
                    for pt in global_planner.path:
                        pose = PoseStamped()
                        pose.header.frame_id = frame_id
                        pose.header.stamp = global_path_msg.header.stamp
                        pose.pose.position.x = pt[0]
                        pose.pose.position.y = pt[1]
                        pose.pose.position.z = 1
                        pose.pose.orientation.w = 1
                        global_path_msg.poses.append(pose)
                    global_path_pub.publish(global_path_msg)





                # print("ROS Topic published!")









    #----------
    #Signal Shutdown when loop is exited
    rate.sleep()
    rospy.signal_shutdown("Finished execution")




if __name__ == '__main__':

    try:
        main()
    except rospy.ROSInterruptException:
        pass
    finally:
        # Call rospy.signal_shutdown() to stop the node when the code is finished
        rospy.signal_shutdown("Finished execution")