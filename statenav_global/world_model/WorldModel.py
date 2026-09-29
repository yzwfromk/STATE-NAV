try:
    import rospy
except ImportError:
    rospy = None

import math

from statenav_global.world_model.globalmap import CMDbasedMap
from statenav_global.world_model.globalmap_with_backend import CMDbasedMapWithBackend
from pathlib import Path
from omegaconf import OmegaConf

# Load config from package data
_PKG_ROOT = Path(__file__).resolve().parent.parent
cfg = OmegaConf.load(_PKG_ROOT / "configs/planning_config.yaml")


class WorldModel:
    """
    WorldModel manages the global map.
    
    When use_shared_memory=True:
    - Uses CMDbasedMapWithBackend (writer mode)
    - Creates shared memory that Planner (separate process) can read
    - Maps are updated in shared memory, accessible by Planner
    """
    
    @staticmethod
    def create_map_from_config(cfg=None, global_goal=None):
        """
        Create a CMDbasedMap instance from config.

        Args:
            cfg: Optional config object. If None, uses module-level cfg.
            global_goal: Optional global goal [x, y]. If None, uses cfg.global_goal.

        Returns:
            CMDbasedMap instance
        """
        if cfg is None:
            cfg = globals()['cfg']
        if global_goal is None:
            global_goal = cfg.global_goal

        return CMDbasedMap(
            env_xmin=cfg.env_extent[0], env_xmax=cfg.env_extent[1],
            env_ymin=cfg.env_extent[2], env_ymax=cfg.env_extent[3],
            goal_x=global_goal[0], goal_y=global_goal[1],
            which_layer=cfg.which_layer,
            preest_update_resolution=cfg.trav_estimation_resoultion,
            instab_limit=cfg.instability_limit,
            load_Travformer=True,
        )

    def __init__(self, use_shared_memory=False, backend=None):
        """
        Initialize WorldModel

        Args:
            use_shared_memory: If True, uses shared memory backend for maps
                              (enables multiprocess access by Planner)
            backend: Optional SharedMemoryBackend or LocalMemoryBackend instance.
                    If provided, uses this backend instead of creating a new one.
        """

        # Initialize map with or without shared memory backend
        if use_shared_memory or backend is not None:
            # Use shared memory backend (writer mode)
            # This creates shared memory that Planner can attach to
            if backend is None:
                # Create backend with shared_metadata if provided
                from statenav_global.world_model.shared_memory_backend import SharedMemoryBackend
                backend = SharedMemoryBackend(mode='writer')
            
            self.global_map = CMDbasedMapWithBackend(
                env_xmin=cfg.env_extent[0],
                env_xmax=cfg.env_extent[1],
                env_ymin=cfg.env_extent[2],
                env_ymax=cfg.env_extent[3],
                goal_x=cfg.global_goal[0],
                goal_y=cfg.global_goal[1],
                which_layer=cfg.which_layer,
                preest_update_resolution=cfg.trav_estimation_resoultion,
                instab_limit=cfg.instability_limit,
                load_Travformer=True,
                backend=backend,  # Pass the backend directly
                use_shared_memory=True
            )
            if rospy:
                rospy.loginfo("[WorldModel] WorldModel initialized with shared memory backend (writer mode)")
        else:
            # Use regular map (single process) - create based on config
            self.global_map = WorldModel.create_map_from_config()


        # Global Planning
        self.send_replanning = False

        # ── Replanning trigger (map/graph change gate) ───────────────────────────────
        # There is no minimum-interval floor: the gate is evaluated once per WorldModel main
        # loop tick, so worldmodel_rate_hz IS the ceiling on trigger frequency. Safe because
        # the planner clears its request before solving (requests coalesce, never queue).
        self.replan_min_changed_nodes = cfg.get('replan_min_changed_nodes', 12)
        self.replan_max_interval = cfg.get('replan_max_interval', 0.0)   # [s] staleness refresh, 0 = off
        self._last_replan_publish_time = None
        self._replan_gate_disabled_warned = False
        self._check_replan_trigger_config()

    ####################### REPLANNING TRIGGER (map/graph change gate) #######################

    def _check_replan_trigger_config(self):
        """
        Validate the replanning-trigger configuration once at construction time.
        Reports instead of raising, so a questionable config still runs with the behavior
        spelled out.
        """
        print(f"[WorldModel] Replan trigger: too-risky-flip fires immediately, "
              f"cost drift fires at >= {self.replan_min_changed_nodes} changed graph nodes "
              f"(eps={cfg.get('replan_cmd_change_eps', 0.05)}), "
              f"max_interval={self.replan_max_interval:.2f} s "
              f"({'staleness refresh on' if self.replan_max_interval > 0.0 else 'staleness refresh off'})")

        if self.replan_min_changed_nodes <= 0:
            print(f"[WorldModel] WARNING: replan_min_changed_nodes="
                  f"{self.replan_min_changed_nodes} <= 0: every WP_TRACKING tick with any graph "
                  f"write will request a replan (equivalent to the old unconditional behavior). "
                  f"Set it > 0 to gate on actual map change.")

    def _check_replan_trigger(self, current_time):
        """
        Decide whether a global replan should be requested, from how much the map/graph
        actually changed since the last published /replanning_signal.

        Tier 1 (structural)  : any node entering/leaving the
                               too-risky set — the only graph changes that add or remove edges.
                               Fires immediately, no threshold.
        Tier 2 (cost drift)  : >= replan_min_changed_nodes graph nodes whose payload actually
                               changed value (not merely got rewritten).
        Tier 3 (staleness)   : replan_max_interval elapsed with no change at all; 0 disables.

        No SIDE EFFECT here — the counters are latched by notify_replanning_published(), which
        runs when the signal is actually published (from any source, not only this gate).

        Args:
            current_time: REQUIRED seconds from the caller's ROS clock (sim-time safe).
                          Only used by Tier 3. No wall-clock fallback: mixing wall time
                          into the staleness delta would fire (or starve) Tier 3 at the
                          sim/wall rate ratio instead of at replan_max_interval.

        Returns:
            bool: True if /replanning_signal should be requested on this tick.
        """
        # Guard: Tier 3 is a time delta; refuse to run on an absent/garbage clock.
        if current_time is None or not math.isfinite(float(current_time)):
            raise ValueError(
                f"[WorldModel] _check_replan_trigger requires current_time from the caller's ROS "
                f"clock, got {current_time!r}. Pass node.get_clock().now().nanoseconds / 1e9 "
                f"(see WorldModelNode._handle_replanning_signals).")

        # Guard: the gate is defined purely on shared-graph writer counters. With no graph
        # writer nothing ever increments them, so gating would mean never replanning again.
        # The graph is built only when global_planner_type == "Astar" (see _build_shared_graph),
        # and never on a reader proxy / use_shared_memory: False. Fall back to the previous
        # unconditional behavior and say so once.
        if getattr(self.global_map, 'graph_writer', None) is None:
            if not self._replan_gate_disabled_warned:
                print("[WorldModel] WARNING: global_map has no shared-graph writer, so the "
                      "replanning change gate has no counters to read; falling back to a "
                      "replan request on every WP_TRACKING tick (previous behavior). "
                      "Expected with global_planner_type != Astar or use_shared_memory: False.")
                self._replan_gate_disabled_warned = True
            return True

        stats = self.global_map.map_change_stats

        # Tier 1 — structural / topology change
        if stats['graph_toorisky_flips'] > 0:
            return True

        # Tier 2 — accumulated cost drift
        if stats['graph_nodes_value_changed'] >= self.replan_min_changed_nodes:
            return True

        # Tier 3 — staleness refresh
        if self.replan_max_interval > 0.0:
            if self._last_replan_publish_time is None:
                return True
            if current_time - self._last_replan_publish_time >= self.replan_max_interval:
                return True

        return False

    def notify_replanning_published(self, current_time=None):
        """
        Latch, called right after /replanning_signal is actually published — from ANY source
        (this gate or an FSM transition). Resets
        the change counters and the staleness clock together, so change accumulated up to
        this point is considered served by that replan, and change arriving afterwards starts
        a fresh accumulation.
        """
        stats = getattr(self.global_map, 'map_change_stats', None)
        if stats is not None:
            for key in stats:
                stats[key] = 0
        self._last_replan_publish_time = current_time





















    
