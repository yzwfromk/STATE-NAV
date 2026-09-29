"""Shared QoS ladder for every statenav / gsa ROS node.

Single source of truth so the statenav nodes, the gsa nodes and RViz agree on the
wire contract.  A copy of this file lives in the gsa package — keep the two in sync.

QoS ladder, ordered by increasing delivery guarantee.

Compatibility rule used throughout: a subscriber connects only if its *requested*
QoS is no stricter than the publisher's *offered* QoS.  So a BEST_EFFORT subscriber
reads a RELIABLE publisher, but a RELIABLE subscriber never sees a BEST_EFFORT one.

This module lives in the standalone `statenav_fsm` package, which is installed into every
interpreter that needs it (statenav venv, Isaac Sim's bundled python, the gsa and
elevation_mapping workspaces).  It is kept out of statenav_global on purpose: importing
anything from statenav_global executes its __init__, which pulls torch and every ROS
message package — far too heavy for a consumer that only wants four QoS constants.

NOTE: importing this module imports rclpy.qos, and nothing else.  Consumers that defer
their rclpy imports (hazard_detection_node.py, VamosPlanner.register_ros_subscribers())
must import this module lazily, inside the function, for the same reason.

The four profiles are module-level singletons, matching the rclpy convention
(rclpy.qos.qos_profile_sensor_data etc.).  Treat them as immutable: they are shared by
every node in the process, and create_publisher()/create_subscription() copy the settings
at endpoint-creation time, so mutating one afterwards would not update live endpoints but
would corrupt every endpoint created later.
"""

import rclpy.qos

# level1: lossy / lowest latency. High-rate, self-superseding sensor streams
#         (point clouds, images) where a stale frame is worse than a dropped one.
#         Use for: exteroceptive sensors.
#           - robot/aggregated_cloud       (PointCloud2)
#           - robot/lidar/point_cloud      (PointCloud2)
#           - robot/front_cam/rgb_image    (Image)
#           - robot/front_cam/depth_image  (Image)
#           - robot/front_cam/camera_info  (CameraInfo)
#         gsa outputs on this level (image / pointcloud traffic):
#           - /segmentation_mask   (Image)
#           - /object_cloud        (PointCloud2)
#           - /recon_cloud         (PointCloud2)
#           - /detected_objects    (DetectedObjectArray — embeds a PointCloud2 per object)
#           - path_clearance debug image (Image)
#         statenav endpoints on this level (all subscriptions):
#           - /elevation_mapping_node/elevation_map_filter  (GridMap, WorldModelNode)
#           - /detected_objects                             (DetectedObjectArray, WorldModelNode)
#           - robot/front_cam/camera_info                   (CameraInfo, WorldModelNode fov_angle)
#           - robot/front_cam/rgb_image + depth_image       (Image, VamosPlanner RGBD sync)
#         NOTE: BEST_EFFORT publishers are invisible to RELIABLE subscribers
#         (RViz2 defaults / plain `ros2 topic echo`).
qos_profile_level1 = rclpy.qos.QoSProfile(
    depth=1,
    reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
    durability=rclpy.qos.DurabilityPolicy.VOLATILE,
    history=rclpy.qos.HistoryPolicy.KEEP_LAST
)

# level2: reliable but latest-only. Retransmits, yet a new sample evicts an
#         unacked one, so high-rate topics can still drop.
#         Use for: low-rate FSM state / one-shot event flags, where only the
#         newest value is meaningful and a superseded one can be discarded.
#           - /fsm_state             (String:  'WP_TRACKING' | 'INTERACTION')
#           - /interaction_success   (Bool)
#           - /interaction_failure   (Bool, IsaacLab task_failure -> FSMNode)
#           - /interaction_failure_progress (Bool, global planner's INTERACTION progress
#                                      monitor -> FSMNode; separate topic so the two failure
#                                      votes keep independent streaks and are OR-ed)
#         gsa topics on this level:
#           - /clearability            (String, VLM result — newest verdict wins)
#           - /VLMevaluation_signal    (Bool, one-shot trigger)
#           - /target_obstacle_live_pos (PointStamped, gsam SAM2 target tracker -> planner and
#                                        WorldModel; <= 2 Hz live position of the object being
#                                        pushed. Both statenav subscriptions use level3's deeper
#                                        buffer, which is compatible with this publisher —
#                                        reliability and durability match, depth is per-endpoint)
#           - global path subscription (Path, a new plan supersedes the old one)
#         statenav topics on this level (also full-snapshot payloads, where the newest
#         message fully replaces the previous one):
#           - /failure_recovered        (Bool, local planner -> FSMNode)
#           - /target_obstacle          (Float32MultiArray, planner -> WorldModel/FSM)
#           - /target_obstacle_name     (String, WorldModel -> consumers; VLM label of the
#                                        obstacle /target_obstacle resolves to, published
#                                        only when a labeled target is resolved)
#           - /replanning_signal        (Bool, WorldModel -> planner)
#           - /manipulatable_obstacles  (Float32MultiArray, whole obstacle list each publish)
#           - /global_costmap           (GridMap | Float32MultiArray, whole map each publish)
#           - /global_path              (Path, whole trajectory each replan)
#           - RViz2 debug topics: /travmap_visualization, /graph_visualization,
#             /obstacle_markers, /planning_routes, /planning_cost_table — kept RELIABLE
#             rather than level1 because RViz2 displays default to RELIABLE and would
#             not see a BEST_EFFORT publisher, and none of them is a sensor stream.
qos_profile_level2 = rclpy.qos.QoSProfile(
    depth=1,
    reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
    durability=rclpy.qos.DurabilityPolicy.VOLATILE,
    history=rclpy.qos.HistoryPolicy.KEEP_LAST
)

# level3: reliable with a 10-sample buffer, so a burst is queued instead of
#         dropped. Use for: continuous rigid-body state and control streams
#         (odom / TF / pose) where the consumer integrates a history, not just
#         the newest sample.
#           - robot/odom      (Odometry)
#           - robot/pose      (PoseStamped)
#           - robot/cmd_vel   (TwistStamped, subscription)
#           - TF broadcasts   (handled by TransformBroadcaster, own QoS)
#         gsa topics on this level:
#           - /tf, /tf_static  (TFMessage — one topic multiplexes many broadcasters,
#                               so depth=1 would drop sibling transforms)
#         statenav endpoints on this level:
#           - /robot/pose      (PoseStamped, subscribed by WorldModelNode, FSMNode
#                               and PathPlannerNode)
qos_profile_level3 = rclpy.qos.QoSProfile(
    depth=10,
    reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
    durability=rclpy.qos.DurabilityPolicy.VOLATILE,
    history=rclpy.qos.HistoryPolicy.KEEP_LAST
)

# level4: latched (TRANSIENT_LOCAL). Late-joining subscribers still receive the
#         last published sample. Requires RELIABLE.
#         Use for: special one-shot / configuration topics published once at
#         startup, whose subscribers may not exist yet at publish time.
#           - /logging/log_dir   (String, experiment log path)
#         gsa topics on this level:
#           - /grounding_busy    (Bool, advisory GPU-contention flag; the late-joining
#                                 hazard node must read the last state immediately)
#         statenav topics on this level:
#           - /planning_config_path  (String, published once by WorldModelNode so the
#                                     planner and the logging tooling load the same YAML)
qos_profile_level4 = rclpy.qos.QoSProfile(
    depth=1,
    reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
    durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
    history=rclpy.qos.HistoryPolicy.KEEP_LAST,
)
