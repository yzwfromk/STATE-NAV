#!/usr/bin/env python3
import numpy as np
import os
from omegaconf import OmegaConf
from pathlib import Path as PathLib

# Use PyTorch's expandable-segments CUDA allocator so freed VRAM can be reused for
# allocations of any size (avoids caching-allocator fragmentation OOM when the GPU is
# shared with other processes). Read when torch's CUDA allocator initializes, so it
# MUST be set before torch is first used; statenav_global imports torch transitively
# (globalmap.py). setdefault lets an explicit external override win. Requires torch>=2.1.
# Skipped on Jetson/Tegra (L4T, detected via /etc/nv_tegra_release): expandable_segments
# relies on the CUDA virtual-memory-management driver calls (cuMemAddressReserve/
# cuMemCreate), which are not reliably supported on the Orin's integrated GPU driver.
# There it makes torch.load(..., map_location='cuda') raise "CUDA driver error: out of
# memory" regardless of actual free VRAM (see docker/JETSON_SETUP.md).
if not os.path.exists('/etc/nv_tegra_release'):
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import statenav_global
from statenav_global.world_model import *
import statenav_global.world_model.WorldModel as WorldModel
from statenav_global.ros.ros_utils import Rviz_vis_graph
# Shared QoS ladder — single source of truth for the wire contract across statenav.
from statenav_fsm.qos_profiles import (
    qos_profile_level2,
    qos_profile_level3,
    qos_profile_level4,
)
# Sim-time console logging: the rcutils "[1786660163.342...]" prefix of every ROS log line is
# wall clock and cannot be changed, so the node's ROS (sim) clock is prepended to the MESSAGE.
from statenav_fsm.sim_time_logger import SimTimeLoggerMixin

import rclpy
import rclpy.parameter
from rclpy.node import Node
from rclpy.clock import ClockType
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, String
from grid_map_msgs.msg import GridMap
from visualization_msgs.msg import MarkerArray


import threading

import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

from statenav_global.core.config import load_planning_config, planning_config_path
cfg = load_planning_config()   # $STATENAV_PLANNING_CONFIG, else the packaged planning_config.yaml


class WorldModelNode(SimTimeLoggerMixin, Node):
    """
    WorldModelNode manages WorldModel and map updates.

    Architecture:
    - WorldModel runs in THIS process (writer) - updates maps in shared memory
    - Planner runs in SEPARATE process (reader) - reads maps from shared memory
    - Communication: Shared memory (maps) + ROS topics (coordination)

    Runs only with named shared memory (use_shared_memory=True): traversability lives in the
    shared lattice graph, which the planner and the MPPI read directly. The former
    use_shared_memory=False mode (dense /global_costmap over ROS) was removed with the dense map.
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

        cfg_path = planning_config_path()   # same resolution as the library modules (core.config)
        self.cfg_path = cfg_path  # advertised over ROS so planner/logging load the same config
        self.cfg = OmegaConf.load(cfg_path)

        print("=" * 80)

        self.initialization_time = self.cfg.initialization_time
        self.debugging_visualization = self.cfg.debugging_visualization
        self.map_visualization_interval = self.cfg.map_visualization_interval
        self.travmap_vis_pub_interval = self.cfg.travmap_vis_pub_interval



        # Determine shared memory mode
        self.use_shared_memory = use_shared_memory if use_shared_memory is not None else (
            self.cfg.get('use_shared_memory', False)
        )

        self.use_shared_memory = use_shared_memory if use_shared_memory is not None else (
            self.cfg.get('use_shared_memory', False)
        )
        # Guard: the shared graph (the only traversability store) needs shared memory; the
        # ROS-message mode that published a dense /global_costmap no longer exists.
        if not self.use_shared_memory:
            raise RuntimeError(
                f"[WorldModel] use_shared_memory={self.use_shared_memory!r} in {cfg_path}, but only "
                f"shared-memory mode is supported: traversability lives only in the shared graph "
                f"(the dense /global_costmap mode was removed). Set use_shared_memory: True.")

        self.current_fsm_state = "WP_TRACKING"

        # Sim time [s] of the previous publish of each signal; None until the first publish.
        # Tracked node-locally, at the actual publish site, so the reported gap is independent
        # of WorldModel's internal trigger baselines. Assigned in __init__
        # rather than run() so a callback-driven publish before the loop starts cannot miss it.
        self._last_replanning_pub_time = None

        # First-replan gate bookkeeping (see _handle_replanning_signals).
        # _first_replan_gate_logged: one-shot, so the initialization-time release is printed
        #                            exactly once instead of on every later publish.
        # _last_replan_hold_log_time: sim time [s] of the last "holding" line, throttled to 1 Hz so
        #                            a held request is visible during init without flooding.
        self._first_replan_gate_logged = False
        self._last_replan_hold_log_time = None

        self._init_map_and_worldmodel()
        self._setup_ros_communication()

        self.get_logger().info(f"[WorldModel] WorldModelNode initialized {'with' if self.use_shared_memory else 'without'} shared memory")




####################################### INITIALIZATION #######################################

    def _init_map_and_worldmodel(self):
        """Initialize map and world model based on shared memory mode."""
        # Shared memory only (guarded in __init__).
        self.get_logger().info("[WorldModel] Initializing WorldModel with shared memory (named)")
        self.world_model = WorldModel.WorldModel(use_shared_memory=True)

    def _setup_ros_communication(self):
        """Setup ROS2 subscribers and publishers.

        QoS: every endpoint below takes a level from the shared ladder in
        statenav_fsm.qos_profiles (see that module for the level definitions and
        the full per-topic registry). level1 = BEST_EFFORT sensor streams, level2 = RELIABLE
        latest-only state/flags, level3 = RELIABLE 10-deep pose stream, level4 = latched.
        """
        # Subscribers
        self.create_subscription(PoseStamped, "/robot/pose",
                                 self.world_model.global_map.pose_callback, qos_profile_level3)
        self.create_subscription(GridMap, "/elevation_mapping_node/elevation_map_filter",
                                 self.world_model.global_map.robo_centric_map_callback, qos_profile_level2)

        # Publishers
        self.replanning_signal_pub = self.create_publisher(Bool, "/replanning_signal", qos_profile_level2)
        # FSMNode owns the FSM; subscribe to its published state
        self.create_subscription(String, "/fsm_state", self._fsm_state_callback, qos_profile_level2)

        # Advertise the loaded planning_config.yaml path so other processes (planner) and
        # sim-logging tooling can load the exact same config. Latched (TRANSIENT_LOCAL,
        # depth 1) so late-joining subscribers still receive it; published once here.
        self.planning_config_path_pub = self.create_publisher(String, "/planning_config_path", qos_profile_level4)
        config_path_msg = String()
        config_path_msg.data = str(self.cfg_path)
        self.planning_config_path_pub.publish(config_path_msg)
        self.get_logger().info(f"[WorldModel] Advertised planning_config path on /planning_config_path: {self.cfg_path}")


        self.frame_id = self.cfg.frame_id

        # Optional Rviz2 visualization of the shared traversability graph (nodes + edges).
        # Switched by the existing pub_travmap_visualization key (its dense /travmap_visualization
        # GridMap was removed with the dense map) and, as before, always on for non-Astar planners.
        astar_active = str(self.cfg.get('global_planner_type', 'Astar')).lower() == 'astar'
        if self.cfg.get('pub_travmap_visualization', False) or not astar_active:
            self.graph_vis_pub = self.create_publisher(MarkerArray, "/graph_visualization", qos_profile_level2)
            self.get_logger().info("[WorldModel] Publishing graph visualization on /graph_visualization")
        else:
            self.graph_vis_pub = None




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

    def _handle_replanning_signals(self, current_time):
        """Publish replanning signal when WorldModel requests a replan.

        current_time: seconds from this node's ROS clock (sim-time safe), used by the
                      initialization gate below. Passed in rather than re-sampled so the gate
                      decides on the same instant as the rest of the tick.
        """
        if self.world_model.send_replanning:
            # Initialization gate: no /replanning_signal may leave WorldModel until
            # initialization_time has passed. Map/graph construction continues
            # during this window, so the first plan sees the map built by the release time.
            # The request is NOT consumed while held: send_replanning stays True and
            # map_change_stats is not reset (notify_replanning_published is not called), so the
            # held request is published on the tick the gate opens.
            if current_time <= self.initialization_time:
                # Throttled to 1 Hz: makes a held request visible during initialization.
                if (self._last_replan_hold_log_time is None
                        or current_time - self._last_replan_hold_log_time >= 1.0):
                    self._last_replan_hold_log_time = current_time
                    self.get_logger().info(
                        f"[WorldModel] Holding /replanning_signal until initialization_time "
                        f"{self.initialization_time - current_time:.2f}s "
                        f"left before the initialization_time release")
                return

            # One-shot: report when initialization_time released the gate.
            if not self._first_replan_gate_logged:
                self._first_replan_gate_logged = True
                self.get_logger().info(
                    f"[WorldModel] First replan gate released by initialization_time "
                    f"({self.initialization_time:.2f}s)")

            msg = Bool()
            msg.data = True
            self.replanning_signal_pub.publish(msg)
            self.world_model.send_replanning = False
            # Sim time of this publish, read once and reused for the log, the local gap
            # tracker and the gate latch, so all three refer to the same instant.
            now = self.get_clock().now().nanoseconds / 1e9  # [s] node ROS clock = sim time
            # Sim-time gap to the previous /replanning_signal (any source: this gate or an FSM
            # transition).
            dt_txt = ("dt=first" if self._last_replanning_pub_time is None
                      else f"dt={now - self._last_replanning_pub_time:.3f}s")
            # Map/graph change accumulated since that previous publish, i.e. exactly the
            # counters the replan gate decided on (see WorldModel._check_replan_trigger).
            # Read BEFORE notify_replanning_published() below, which zeroes them.
            #   toorisky_flips  : nodes entering/leaving the too-risky set (Tier 1, fires at >0)
            #   nodes_changed   : graph nodes whose payload changed value (Tier 2, thresholded)
            # All zero with a dt near replan_max_interval means Tier 3 (staleness) fired; all
            # zero at the main-loop period means the no-graph-writer fallback is replanning
            # every tick.
            stats = getattr(self.world_model.global_map, 'map_change_stats', None)
            chg_txt = (f"toorisky_flips={stats['graph_toorisky_flips']}, "
                       f"nodes_changed={stats['graph_nodes_value_changed']}"
                       if stats is not None else "change=n/a (no map_change_stats)")
            # Log every publish so the replan trigger is visible; capped by the main-loop rate
            # (worldmodel_rate_hz). FSM state is included because the trigger is state-dependent.
            self.get_logger().info(
                f"[WorldModel] Published /replanning_signal "
                f"(fsm={self.current_fsm_state}, {dt_txt}, {chg_txt})")
            self._last_replanning_pub_time = now
            # Latch the change gate: this replan serves everything accumulated so far.
            self.world_model.notify_replanning_published(now)



####################################### FSM PER-STATE ACTIONS #######################################

    def _fsm_state_callback(self, msg):
        """Track FSM state and handle WP_TRACKING re-entry transitions."""
        raw = msg.data
        # Two collapses to a base state. WP_TRACKING_SUCCESS is the transition tag; the three
        # INITIALIZATION_<PHASE> strings collapse to ONE state so the whole init window is a
        # single state entry here. Without that, SETTLING -> ROTATE -> HOLD would each count as
        # a transition below.
        base_state = ("WP_TRACKING" if raw.startswith("WP_TRACKING_")
                      else "INITIALIZATION" if raw.startswith("INITIALIZATION")
                      else raw)

        prev_state = self.current_fsm_state

        # TODO: This can be moved under WorldModel

        # Log every FSM state transition (raw msg shown when it differs from base_state)
        if prev_state != base_state:
            suffix = f" (raw: {raw})" if raw != base_state else ""
            self.get_logger().info(f"[WorldModel] FSM: {prev_state} → {base_state}{suffix}")

        # Safety net for the planner's single trigger: the planner plans ONLY on
        # /replanning_signal now, so every re-entry into WP_TRACKING has to
        # carry a request from here or the planner simply never plans again.
        if (base_state == "WP_TRACKING" and prev_state != "WP_TRACKING"
                and not self.world_model.send_replanning):
            self.get_logger().warning(
                f"[WorldModel] FSM: {prev_state} → WP_TRACKING (raw: {raw}) carried no replan "
                f"request; requesting one so the planner cannot stall.")
            self.world_model.send_replanning = True

        # CAUTION: Need to assign the incoming state after processing the transition
        # to prevent the main loop runs before processing the transition
        # and executes the wrong per-state code.
        self.current_fsm_state = base_state


    def _run_WP_TRACKING(self, current_time):
        """Actions executed each loop iteration while in WP_TRACKING state."""
        # Startup-only ordering: the first elevation/traversability update allocates the dense
        # maps and, in shared-memory Astar mode, constructs the graph.
        map_initialized_this_tick = False
        if not self.world_model.global_map.is_TraversabilityMap_built:
            self._update_map()
            map_initialized_this_tick = self.world_model.global_map.is_TraversabilityMap_built
            if not map_initialized_this_tick:
                return

        # Guard the construction invariant explicitly. CMDbasedMapWithBackend.Initilize_map()
        # builds the graph (every planner type: it is the only traversability store) before it
        # lets the traversability map become ready, so reaching this state without a writer
        # means startup is inconsistent.
        if self.world_model.global_map.graph_writer is None:
            raise RuntimeError(
                "[WorldModel] Traversability map is built but the required shared graph "
                "writer is missing.")

        # The startup tick already updated once to create the map/graph and must not
        # run the expensive Travformer update twice.
        if not map_initialized_this_tick:
            self._update_map()

        if self.world_model.global_map.is_TraversabilityMap_built:
            # Publish Rviz2 graph visualization
            if self.graph_vis_pub is not None:
                Rviz_vis_graph(
                    self.world_model.global_map,
                    self.graph_vis_pub,
                    self,
                    self.frame_id,
                )


        # Request a replan only when the map/graph actually changed enough to affect the plan
        # (see WorldModel._check_replan_trigger). OR-in, never overwrite: callback-driven sites
        # (FSM transitions) may already have set it.
        if self.world_model._check_replan_trigger(current_time):
            self.world_model.send_replanning = True
        self._handle_replanning_signals(current_time)








####################################### RUN #######################################

    def run(self):
        """Main execution loop."""
        self.get_logger().info("[WorldModel] WorldModelNode running. Waiting for map data...")

        spin_thread = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)
        spin_thread.start()

        # Main-loop rate limiter. Without it run() free-runs: it pins a full core, starves the
        # rclpy.spin thread above (GIL) so pose/elevation callbacks get delayed, and republishes
        # /replanning_signal on every iteration. Rate uses this node's clock, so under
        # use_sim_time it follows /clock. Created after spin_thread.start() so the executor picks
        # up its timer (same order as main_global_planning.py).
        worldmodel_rate_hz = self.cfg.get('worldmodel_rate_hz', 1)  # (Hz) target loop cadence
        worldmodel_rate = self.create_rate(worldmodel_rate_hz)
        self.get_logger().info(f"[WorldModel] Main loop rate limited to {worldmodel_rate_hz} Hz")

        self._last_rostime_log_time = 0.0
        self.last_visualization_time = 0.0
        self.last_travmap_vis_pub_time = 0.0  # tracks last publish time for /travmap_visualization (0.5 Hz cap)

        while rclpy.ok():
            current_time = self.get_clock().now().nanoseconds / 1e9

            # 1 Hz liveness heartbeat, so quiet ticks
            # still show the loop is running. Throttled on current_time — this node's
            # ROS clock — rather than with rclpy's throttle_duration_sec, whose default time
            # source is the wall clock and would drift from the [sim ...] stamp on the line.
            if current_time - self._last_rostime_log_time > 1.0:
                # Visual separator so each 1 Hz heartbeat block is easy to spot in the console.
                self.get_logger().info("=" * 60)
                self.get_logger().info("[WorldModel] Updated with new measurements")
                # self.get_logger().info("[WorldModel] RCLPY in loop. Time is %.2f" % current_time)
                self._last_rostime_log_time = current_time

            try:
                # INITIALIZATION runs the SAME per-state work as WP_TRACKING, deliberately: the
                # init window is when the map/graph get allocated.
                # Skipping it leaves the shared-memory maps unwritten — which blocks the local
                # planner's reader attach for the whole window, so nothing commands the init
                # sweep — and leaves the first plan blind. /replanning_signal is NOT released
                # early by this: _handle_replanning_signals holds it on its own
                # initialization_time gate, which is a clock gate independent of the FSM state.
                if self.current_fsm_state in ("WP_TRACKING", "INITIALIZATION"):
                    self._run_WP_TRACKING(current_time)

            except Exception as e:
                self.get_logger().error(f"[WorldModel] Error in WorldModelNode run loop: {e}")
                import traceback
                self.get_logger().error(f"[WorldModel] {traceback.format_exc()}")

            # Sleep the remainder of the loop period. Placed outside the try/except so the rate
            # is honored even when the state handler raised (otherwise a persistent exception
            # would turn this back into a busy loop). On overrun it returns immediately.
            worldmodel_rate.sleep()




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
