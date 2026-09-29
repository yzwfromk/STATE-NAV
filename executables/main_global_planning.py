#!/usr/bin/env python3
import numpy as np
import math
from omegaconf import OmegaConf

import os
# Global-planner process runs matplotlib headless (no GUI windows). The shared statenav_global
# modules read STATENAV_MPL_BACKEND at import time (see plotting.py / globalmap.py), so it MUST
# be set before importing statenav_global below. setdefault lets an explicit external override win.
os.environ.setdefault('STATENAV_MPL_BACKEND', 'Agg')

import statenav_global
from statenav_global import CMDbasedMap
from statenav_global.world_model.globalmap_with_backend import CMDbasedMapReaderProxy
from statenav_global.world_model.shared_memory_backend import SharedMemoryBackend
from statenav_global.utility.ros_utils import populate_map_from_float32multiarray, populate_map_from_gridmap, _elevation_z
from statenav_global.utility.utils import wrap_to_pi

# Note: tf2_ros imports available if needed for future TF operations
# from tf2_ros import TransformListener, Buffer, TransformException
# import tf2_geometry_msgs

import rclpy
import rclpy.parameter
from rclpy.node import Node
from rclpy.clock import ClockType
# Shared QoS ladder — single source of truth for the wire contract across statenav.
from statenav_fsm.qos_profiles import (
    qos_profile_level2,
    qos_profile_level3,
    qos_profile_level4,
)
# Sim-time stamping for this node's ROS log lines. The rcutils "[INFO] [<stamp>]" prefix is
# wall clock and cannot be retargeted to /clock, so the mixin prepends the node's ROS clock
# reading to the MESSAGE instead (see statenav_fsm/sim_time_logger.py).
from statenav_fsm.sim_time_logger import SimTimeLoggerMixin
from std_msgs.msg import Float32MultiArray, Bool, String
from grid_map_msgs.msg import GridMap
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

import threading
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

from pathlib import Path as PathLib
cfg = OmegaConf.load(PathLib(statenav_global.__file__).parent / "configs/planning_config.yaml")


def set_random_seed(seed):
    rng = np.random.RandomState(seed)
    print(f"Set random seed to {seed} in numpy.")
    return rng








################################################### Helper Functions for ROS ###################################################





































class PlanningNode(SimTimeLoggerMixin, Node):
    """Main planning node that subscribes to map and publishes paths.

    SimTimeLoggerMixin is listed FIRST on purpose: its get_logger() must win the MRO over
    Node.get_logger(), otherwise nothing is stamped with sim time.
    """

    def __init__(self, use_sim_time: bool = False):
        """
        Initialize PlanningNode
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

        os.nice(-15)

        # Load configuration
        self.cfg = cfg

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
            self.global_map = CMDbasedMapReaderProxy(backend)
            # Set goal after initialization
            self.global_map.goal_x = global_goal[0]
            self.global_map.goal_y = global_goal[1]

        if not self.use_shared_memory:
            # ROS message mode: create map and subscribe to topics
            self.get_logger().info("[PathPlanner] Initializing in ROS MESSAGE mode")
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
                load_Travformer=False,
            )
        
        self.get_logger().info(f"[PathPlanner] Map class initialized. Type: {type(self.global_map).__name__}, Mode: {'SHARED MEMORY' if self.use_shared_memory else 'ROS MESSAGE'}")
        













        # Initialize global planner
        self.initial_start = self.cfg.initial_start
        self.global_goal = self.cfg.global_goal
        self.heading_start = np.deg2rad(self.cfg.heading_start)

        self.rng = set_random_seed(self.cfg.seed)

        # Planner factory: select implementation via planning_config.yaml
        # global_planner_type: 'Astar' (default)
        planner_type = self.cfg.get('global_planner_type', 'Astar')
        self.planner_type = planner_type

        if planner_type == 'Astar':
            # Heading-aware sparse A* on the shared-memory graph (Phase C).
            # Heading-layer geometry comes from the world-model instance (the
            # authoritative writer-side source), so it is never duplicated as a
            # second config location; A*-specific scalars come from graph_astar_*.
            self.global_planner = statenav_global.planners.GlobalAStar(
                self.task_extent, self.rng,
                tuple(self.initial_start), tuple(self.global_goal), self.heading_start,
                theta_min=self.global_map.TraversabilityMap_theta_min,
                theta_resolution=self.global_map.TraversabilityMap_theta_resolution,
                theta_layers=self.global_map.TraversabilityMap_size_layers,
                snapshot_max_retries=self.cfg.get('graph_astar_snapshot_max_retries', 50),
                epsilon=self.cfg.get('graph_astar_epsilon', 1.0e-12),
                heuristic=self.cfg.get('graph_astar_heuristic', 'reverse_dijkstra'),
                heuristic_weight=self.cfg.get('graph_astar_heuristic_weight', 1.5),
                alt_landmark_count=self.cfg.get('graph_astar_alt_landmarks', 4),
                speed_discount=self.cfg.get('speed_discount', 1.0),
                arc_radius_factor=self.cfg.get('arc_radius_factor', 1.5),
            )
        else:
            # No silent fallback: an unrecognised value is a config error.
            # Constructing a different planner instead would hide the typo until
            # the robot is already driving on the wrong plan.
            raise ValueError(
                f"Unknown global_planner_type={planner_type!r} in planning_config.yaml. "
                f"Valid options: 'Astar'.")

        # Post-init attachments
        self.global_planner.global_map = self.global_map

        # Path smoothing config
        self.global_planner.path_smoothing_enable = bool(
            self.cfg.get('path_smoothing_enable', True))
        self.global_planner.path_smoothing_radius = float(
            self.cfg.get('path_smoothing_radius', 0.5))
        self.global_planner.path_smoothing_step = float(
            self.cfg.get('path_smoothing_step', 0.25))

        # ROS2 setup
        self.frame_id = self.cfg.get('frame_id', 'map')

        # QoS from the shared ladder (statenav_fsm.qos_profiles):
        #   level3 -> /robot/pose, the continuous pose stream the planner replans from
        #   level2 -> state / event flags and full-snapshot payloads (costmap,
        #             global path) plus the RViz2 debug topics, which stay
        #             RELIABLE because RViz2 displays default to RELIABLE
        #   level4 -> /logging/log_dir, latched (see below)
        # level1 is unused here: this node consumes no sensor streams.

        # Subscribers (different based on mode)
        if self.use_shared_memory:
            # Shared memory mode: subscribe to pose and replanning signal via ROS
            self.create_subscription(PoseStamped, "/robot/pose", self.pose_callback, qos_profile_level3)
            self.create_subscription(Bool, "/replanning_signal", self.replanning_callback, qos_profile_level2)
            self.get_logger().info("[PathPlanner] Subscribed to /robot/pose and /replanning_signal (shared memory mode)")
        else:
            # ROS message mode: subscribe to map topics and pose
            self.use_gridmap_msg = self.cfg.get('use_gridmap_msg', False)
            if self.use_gridmap_msg:
                self.create_subscription(GridMap, "/global_costmap", self.map_callback, qos_profile_level2)
                self.get_logger().info("[PathPlanner] Subscribed to /global_costmap (GridMap)")
            else:
                self.create_subscription(Float32MultiArray, "/global_costmap", self.map_callback, qos_profile_level2)
                self.get_logger().info("[PathPlanner] Subscribed to /global_costmap (Float32MultiArray)")

            self.create_subscription(PoseStamped, "/robot/pose", self.pose_callback, qos_profile_level3)
            self.get_logger().info("[PathPlanner] Subscribed to /robot/pose")

        # Subscribe to FSM state (both modes)
        self.create_subscription(String, "/fsm_state", self._fsm_state_callback, qos_profile_level2)
        self.get_logger().info("[PathPlanner] Subscribed to /fsm_state")

        # Subscribe (latched) to the experiment log directory published by IsaacLab when
        # logging starts. On receipt we schedule a one-time dump of the initial elevation
        # map and traversability map into
        # <log_dir>/initial/. transient_local matches the latched publisher so we still
        # receive it even if IsaacLab published before this node subscribed. level4 of the
        # shared ladder is exactly that latched (TRANSIENT_LOCAL) profile.
        self.create_subscription(String, "/logging/log_dir", self._log_dir_callback, qos_profile_level4)
        self.get_logger().info("[PathPlanner] Subscribed to /logging/log_dir (latched)")
        # One-time snapshot state
        self._log_snapshot_dir = None       # <log_dir>/initial target directory (set on log_dir msg)
        self._snapshot_dump_pending = False # True once a log_dir is received, until dump succeeds
        self._snapshot_dumped = False       # latched True after a successful dump (fires once)

        print(self.global_map.robot_heading)




        # Publishers
        self.global_path_pub = self.create_publisher(Path, "/global_path", qos_profile_level2)
        self.get_logger().info("[PathPlanner] Publishers initialized: /global_path")













        
        # For getting next waypoint for resetting the tree
        # Always sourced from the dedicated global-planner key (independent of waypoint_lookahead_time).
        self.waypoint_lookahead_time = self.cfg.globalplanner_lookahead_time
        
        # State
        self.last_plan_time = 0
        self.planning_in_progress = False
        self.initialization_time = self.cfg.initialization_time
        # Replan request plumbing (shared-memory mode). The event is the wake-up edge; the
        # sticky flag is the request itself, so a request that arrives while the planner
        # cannot act (map not built, before initialization_time, FSM not in WP_TRACKING)
        # is retained instead of being consumed by a no-op dispatch.
        self._replan_event = threading.Event()
        # NO bootstrap set() here on purpose: the planner plans ONLY on a received request, and
        # /replanning_signal from WorldModel is now its ONLY source (see _fsm_state_callback).
        # The signal is held there until initialization_time, while WorldModel builds the
        # map/graph first. A bootstrap request here would defeat that gate
        # by planning prematurely.
        self._replan_pending = False
        self._new_map_event = threading.Event()  # signals a new map is ready (ROS message mode)

        # FSM state tracking
        self.current_fsm_state = "WP_TRACKING"

        self.get_logger().info(f"[PathPlanner] Planning Node initialized with CMDbasedMap!")
    





####################################### PUBLISHING #######################################

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
            
            # Get elevation z for this waypoint from the elevation map; fallback to 1.0.
            # Uses the shared _elevation_z() helper so the path floats above terrain
            # instead of being buried in it.
            pose.pose.position.z = _elevation_z(
                self.global_map, pose.pose.position.x, pose.pose.position.y, 0.3, 1.0
            )   # surface + 0.3 m, else 1.0 m
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
            self._replan_event.set()

    def _fsm_state_callback(self, msg):
        """Track FSM state."""
        raw = msg.data
        if raw.startswith("WP_TRACKING_"):
            base_state = "WP_TRACKING"
        else:
            base_state = raw

        prev_state = self.current_fsm_state
        self.current_fsm_state = base_state

        # → any non-WP_TRACKING state: drop any replan request that has not been served
        # yet (see _drop_replan_request).
        if base_state != "WP_TRACKING" and base_state != prev_state:
            self._drop_replan_request(f"→ {base_state}")

        # NO self-triggered replan on WP_TRACKING re-entry, on purpose: planning is driven
        # STRICTLY by /replanning_signal from WorldModel.
        # WorldModel requests a replan on EVERY WP_TRACKING re-entry (its _fsm_state_callback
        # ends with a safety net for exactly that).



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

        # Signal the main run loop that a new map is ready
        self._new_map_event.set()
    
    
####################################### FSM PER-STATE ACTIONS #######################################

    def _run_WP_TRACKING(self, current_time):
        """Actions executed each loop iteration while in WP_TRACKING state."""
        if current_time <= self.initialization_time:
            return
        self.plan_and_publish()
        # After the first post-init plan, dump the initial maps/planner image once
        # (only when an experiment log dir has been received and a valid plan/map exists).
        self._maybe_dump_initial_snapshot()

####################################### REPLAN REQUEST #######################################
    def _drop_replan_request(self, context):
        """Discard an unserved replan request, so planning resumes only on a NEW signal.

        Only WP_TRACKING plans. A request that was raised before the FSM left WP_TRACKING
        describes the world as it was before that transition, so it is not served later.

        Dropping cannot stall the planner: WorldModel requests a replan on EVERY WP_TRACKING
        re-entry. Dropping a REQUEST loses no map information
        either — every replan reads the newest graph snapshot when it runs.

        context : str  short tag for the log line, e.g. "→ INITIALIZATION_HOLD"; only the
                       wording of the message depends on it.
        Called from the spin thread (_fsm_state_callback); both writes are
        single attribute / Event operations.
        """
        if self._replan_pending or self._replan_event.is_set():
            self.get_logger().info(
                f"[PathPlanner] {context}: dropping the unserved replan request "
                f"(pending={self._replan_pending}, signal_queued={self._replan_event.is_set()}); "
                f"planning resumes on the next /replanning_signal from WorldModel.")
        self._replan_pending = False
        self._replan_event.clear()



####################################### INITIAL SNAPSHOT DUMP #######################################

    def _log_dir_callback(self, msg):
        """Receive the experiment log directory (latched) and schedule a one-time snapshot dump."""
        import os
        log_dir = msg.data
        if not log_dir:
            return
        # Target <log_dir>/initial/ ; the actual dump happens after the next post-init plan.
        self._log_snapshot_dir = os.path.join(log_dir, "initial")
        if not self._snapshot_dumped:
            self._snapshot_dump_pending = True
        self.get_logger().info(
            f"[PathPlanner] Received log dir; will dump initial snapshot to {self._log_snapshot_dir}"
        )

    def _maybe_dump_initial_snapshot(self):
        """Dump the initial maps exactly once, when a log dir has been
        received and a valid traversability map + plan are available. Deferred (retried on
        the next plan) if prerequisites are not yet met."""
        if not self._snapshot_dump_pending or self._snapshot_dumped:
            return
        if not self.global_map.is_TraversabilityMap_built:
            self.get_logger().info("[PathPlanner] Snapshot deferred: traversability map not built yet")
            return
        if len(self.global_planner.path) == 0:
            self.get_logger().info("[PathPlanner] Snapshot deferred: no plan available yet")
            return
        try:
            self._dump_maps_and_graph(self._log_snapshot_dir)
            self._snapshot_dumped = True
            self._snapshot_dump_pending = False
            self.get_logger().info(f"[PathPlanner] Initial snapshot dumped to {self._log_snapshot_dir}")
        except Exception as e:
            import traceback
            self.get_logger().error(f"[PathPlanner] Initial snapshot dump failed: {e}")
            traceback.print_exc()

    def _dump_maps_and_graph(self, out_dir):
        """Render the initial maps into out_dir.

        Reuses the exact visualization paths the user referenced. Each sub-dump is guarded
        independently so one failing (e.g. missing planner state) does not block the others:
          - global_map.visualize_maps(..., save_dir)  -> 'World Elevation Map.pdf', 'Traversability Map.pdf'

        Rendering is forced headless: the plotting helpers pop up GUI windows
        (plt.show(block=False)/plt.pause()). The global-planner process selects the non-GUI 'Agg'
        backend (STATENAV_MPL_BACKEND=Agg, set at this module's top), so windows never open here;
        this block additionally switches to 'Agg' for the dump as defense-in-depth (no-op when
        already headless) so the snapshot saves to PDF WITHOUT pop-ups even if an interactive
        backend is ever selected, then restores the previous backend. Switching closes open
        figures, but every figure here is created guarded by plt.fignum_exists(), so it is
        recreated on demand.
        """
        import os, warnings
        import matplotlib
        import matplotlib.pyplot as plt
        os.makedirs(out_dir, exist_ok=True)

        gm = self.global_map
        planner = self.global_planner

        # Force headless rendering for the whole snapshot so no GUI windows pop up.
        _prev_backend = matplotlib.get_backend()
        _switched = False
        try:
            # Switch only if not already on a headless Agg backend. Compare the EXACT name:
            # interactive backends ('QtAgg', 'TkAgg', ...) all contain the substring 'agg',
            # so a substring test would wrongly skip the switch.
            if _prev_backend.lower() != "agg":
                plt.switch_backend("Agg")   # non-GUI: savefig works, show()/pause() are no-ops
                _switched = True

            # Silence the expected "non-GUI backend cannot show the figure" warnings that the
            # plotting helpers' plt.show(block=False) emit under the Agg backend.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # (1)+(2) Elevation + traversability maps, with the current plan overlaid.
                try:
                    theta_robot = wrap_to_pi(gm.robot_heading)          # robot heading (rad), wrapped to (-pi, pi]
                    gm.visualize_maps(theta_robot, planner.path, save_dir=out_dir)
                except Exception as e:
                    self.get_logger().error(f"[PathPlanner] Snapshot: elevation/traversability map dump failed: {e}")
        finally:
            # Always restore the previous (GUI) backend, even if a sub-dump raised.
            if _switched:
                try:
                    plt.switch_backend(_prev_backend)
                except Exception as e:
                    self.get_logger().warning(f"[PathPlanner] Snapshot: could not restore backend '{_prev_backend}': {e}")








    def _print_replan_header(self, ros_t):
        """Print the one-line planner status header (separator + status line).

        ros_t : float   node ROS clock [s] (sim time whenever /clock drives it)
        """
        rx, ry, rh = self.global_map.robot_x, self.global_map.robot_y, self.global_map.robot_heading  # robot pose [m, m, rad]
        replan_msg = (
            f"[PathPlanner] Replanning | t={ros_t:.2f}s | "
            f"robot=({rx:.2f}, {ry:.2f}, {np.rad2deg(rh):.1f}deg)"
        )
        # Printed instead of ROS-logged so the turn header carries no rcutils
        # "[INFO] [<wall clock>] [planning_node]:" prefix. t= is the node's ROS clock,
        # i.e. sim time whenever /clock drives it, so no time information is lost.
        print("=" * 72)
        print(replan_msg)

    def plan_and_publish(self, forced_start=None):
        """Run global planning and publish the path."""
        if not self.global_map.is_TraversabilityMap_built:
            self.get_logger().warning("[PathPlanner] Map not built yet, cannot plan")
            return

        self.planning_in_progress = True

        try:
            # Single replan status line: ROS time, robot pose
            ros_t = self.get_clock().now().nanoseconds / 1e9
            self._print_replan_header(ros_t)

            self.global_planner.replan(
                initial_start=(self.initial_start[0], self.initial_start[1]),
                total_lookahead_time=self.waypoint_lookahead_time,
                plot_map=False,
                forced_start=forced_start
            )

            # Single plan-summary line: solver time + plan ETA.
            comp_t = getattr(self.global_planner, 'last_computation_time', None)  # solve time [s]; None if the planner does not report one (or the replan was skipped)
            summary = "[PathPlanner]"
            if comp_t is not None:
                summary += f" Computation Time: {np.round(comp_t, 3)} s |"
            if len(self.global_planner.path) > 0:
                summary += f" Navigation Cost (ETA): {np.round(self.global_planner.plan_cost, 3)} s |"
            else:
                summary += " No path |"
            print(summary.rstrip(" |"))

            # Publish path
            self.publish_path()

            
            

            self.last_plan_time = self.get_clock().now().nanoseconds / 1e9

        except Exception as e:
            self.get_logger().error("[PathPlanner] Error during planning: %s" % str(e))
            import traceback
            traceback.print_exc()
        finally:
            self.planning_in_progress = False




####################################### RUN #######################################

    def run(self):
        # Spin in a background thread so ROS callbacks are always processed
        spin_thread = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)
        spin_thread.start()

        if self.use_shared_memory:
            self.get_logger().info("[PathPlanner] Planning Node started (shared memory mode). Waiting for maps and replanning signals...")

            self._last_wait_log_time = 0.0

            while rclpy.ok():
                # Block until a replan is requested. The timeout is NOT a planning rate: it only
                # bounds the "waiting for maps" log and re-checks rclpy.ok() / FSM state. Wall-clock
                # based on purpose — create_rate() blocks forever if /clock stops under sim time.
                if self._replan_event.wait(timeout=1.0):
                    self._replan_event.clear()    # cleared BEFORE planning: a signal arriving during
                    self._replan_pending = True   # the solve re-sets it and is served next iteration

                current_time = self.get_clock().now().nanoseconds / 1e9

                if not self.global_map.is_TraversabilityMap_built:
                    if current_time - self._last_wait_log_time > 5.0:
                        self.get_logger().info("[PathPlanner] Waiting for maps to be built...")
                        self._last_wait_log_time = current_time
                    continue      # request kept in _replan_pending; event is clear so no spin

                if current_time <= self.initialization_time:
                    continue      # _run_WP_TRACKING early-returns before init; keep the request

                if not self._replan_pending:
                    # Visibility guard for the removed bootstrap request: with no self-request, an
                    # absent /replanning_signal means the planner legitimately never plans, which
                    # would otherwise look like a silent hang. Logged only while no plan exists at
                    # all (steady-state idling between replans is normal), throttled to 5 s.
                    if (len(self.global_planner.path) == 0
                            and current_time - self._last_wait_log_time > 5.0):
                        self._last_wait_log_time = current_time
                        self.get_logger().info(
                            "[PathPlanner] Maps ready, no plan yet: waiting for the first "
                            "/replanning_signal from WorldModel (held there until "
                            "initialization_time)")
                    continue

                if self.current_fsm_state == "WP_TRACKING":
                    self._replan_pending = False
                    self._run_WP_TRACKING(current_time)
        else:
            self.get_logger().info("[PathPlanner] Planning Node started (ROS message mode). Waiting for map updates...")
            while rclpy.ok():
                triggered = self._new_map_event.wait(timeout=1.0)
                self._new_map_event.clear()
                if not triggered:
                    continue
                current_time = self.get_clock().now().nanoseconds / 1e9
                if self.current_fsm_state == "WP_TRACKING":
                    self._run_WP_TRACKING(current_time)

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
