#
#   Unified local planner: velocity control for wp_tracking.
#   FSM state topic: /fsm_state (std_msgs/String: "initialization_settling" |
#   "initialization_rotate" | "initialization_hold" | "wp_tracking",
#   plus the transition tags in FSM_TRANSITION_TAGS)
#   The initialization_* phases are scheduled by NavigationFSM and COMMANDED HERE: this node
#   issues the standstill and the in-place yaw sweep that open every run (see run()).
#

import sys
import os
import math
import signal
import threading
import dataclasses
from collections import deque
from collections.abc import Callable

# main_mppi.py lives in state_nav/executables/; the statenav_mppi package sits under the repo root.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('../../')

import numpy as np

# ── MPPI local planner ────────────────────────────────────────────────────────
from statenav_mppi.mppi_planners.mppi_planner import MPPIConfig, MPPIPlanner, MPPIPathTrackingResult

# ── MPPI plan visualization (RViz MarkerArray, replaces the old Matplotlib figure) ──
from statenav_global.ros.mppi_rviz import MPPIMarkerPublisher

# ── ROS2 ──────────────────────────────────────────────────────────────────────
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import SingleThreadedExecutor
from rclpy.exceptions import ROSInterruptException
from rclpy.signals import SignalHandlerOptions
# Shared QoS ladder — single source of truth for the wire contract across statenav,
# gsa, elevation_mapping, IsaacLab and this repo. Requires the standalone
# statenav_fsm package: pip install -e <path-to>/state_nav/statenav_fsm
from statenav_fsm.qos_profiles import (
    qos_profile_level2,
    qos_profile_level3,
)
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import String

# ── Traversability map reader ─────────────────────────────────────────────────
from statenav_global.core.shared_memory_backend import SharedMemoryBackend
from statenav_global.world_model.globalmap_with_backend import CMDbasedMapReaderProxy

# ── Config ────────────────────────────────────────────────────────────────────
from statenav_global.core.config import load_planning_config

cfg = load_planning_config()   # $STATENAV_PLANNING_CONFIG, else the packaged planning_config.yaml


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


# FSM transition tags: /fsm_state carries one of these INSTEAD of a state name for
# fsm_transition_tag_duration seconds after the tagged transition (see main_fsm.py). They exist
# for WorldModel / PlanningNode; they name NO new local-planner behaviour — the state each one
# tags is the value here.
# Mapped rather than ignored: an unmapped tag falls through to the unknown-state branch of
# fsm_state_callback, which leaves self.fsm_state at its PREVIOUS value.
FSM_TRANSITION_TAGS: dict[str, str] = {"wp_tracking_success": "wp_tracking"}


# Ordered (previous, next) FSM transitions whose warm-start plan SURVIVES the change. None:
# every transition changes what the plan was solved against and drops it in run().
WARMSTART_KEEP_TRANSITIONS: frozenset[tuple[str, str]] = frozenset()

# A NEWLY received global path now drops the warm start UNCONDITIONALLY, so there is no distance
# limit to configure any more (the former WARMSTART_MAX_END_DIST_M). The measured test that
# replaced it -- MPPIPlanner.drop_plan_if_off_path, kept for its own callers and tests -- asked
# whether the cached plan still ENDED near the new path, and answered "yes" for exactly the
# reroute this planner handles worst: a path that shifts laterally by less than the limit but
# demands a large heading change at its start. Keeping the plan there re-seeds the tick with a
# nominal aimed at the OLD corridor. Dropping it re-seeds with zeros, which
# MPPICore._project_and_rollout turns into a genuine hold-position prior -- but only because
# mppi_lin_acc_ub is now >= maximum_cmd_v / ctrl_period, so the knot-zero rate floor
# max(0, previous_v - lin_acc * dt) reaches 0 and CommandConstraint._expand_to_boundary leaves
# an exactly-zero linear command alone. At a smaller rate bound that same zero seed is grown
# back onto the coupled simplex as FULL SPEED STRAIGHT AHEAD, which is why the two changes
# belong together.


# ─────────────────────────────────────────────────────────────────────────────
# Controller parameters
# ─────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────
# Local Planner Node
# ─────────────────────────────────────────────────────────────────────────────

class LocalPlannerNode(Node):
    """Unified planner: velocity control for tracking."""

    def __init__(self):
        super().__init__(
            "local_planner",
            parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)],
        )

        os.nice(-20) 

        
        # ── Robot pose / velocity state ────────────────────────────────────
        # Latest pose as ONE tuple (x_m, y_m, z_m, yaw_rad), not separate attributes.
        # pose_callback now runs in the executor thread (see the executor block below), so four
        # independent assignments could be read half-updated by the control loop — x from one
        # message and yaw from the next. Rebinding a single tuple is one bytecode op under the
        # GIL, so run() always sees a self-consistent pose.
        self.msg_pose: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        # True once /robot/pose has been received at least once. msg_pose starts at all-zeros,
        # which is a VALID pose, so the init sweep (which latches its reference heading from it)
        # needs to know whether that tuple is a measurement or the placeholder.
        self._pose_received: bool = False
        self.pose_timestamp = 0

        # Moving average window — configurable; used in twist_callback to compute avg_v / avg_w
        self.MA_WINDOW_S: float = float(cfg.ma_window_s)

        # Moving-average buffers: each entry is (wall_time_s, value)
        self._v_history: deque = deque()   # linear  velocity samples (m/s)
        self._w_history: deque = deque()   # angular velocity samples (rad/s)
        self.avg_v: float = 0.0            # current moving-average linear  velocity (m/s)
        self.avg_w: float = 0.0            # current moving-average angular velocity (rad/s)

        # Liveness tracking for the measured-velocity stream (/robot/twist on hardware,
        # /robot/odom in sim — whichever is publishing). avg_v/avg_w are the ONLY measured
        # feedback in the wp_tracking law: they drive the lag roll-forward (predict_pose) and the
        # speed-profile scan horizon. When the stream dies they silently hold their init value of
        # 0.0, which makes predict_pose return the pose unchanged and collapses the horizon to
        # its floor — i.e. the planner degrades into a no-lag controller and overshoots without
        # saying so. -1.0 marks "no sample ever received"; the guard treats that as stale from
        # startup, which is exactly how the sim-side missing publisher was found.
        self._last_vel_t: float = -1.0   # node-clock time (s) of the last velocity sample
        # Age (s) past which the velocity stream counts as stale. Three control periods: long
        # enough that ordinary jitter or a single dropped sample does not trip it, short enough
        # that a dead publisher is reported within a third of a second.
        self.VEL_STALE_S: float = 3.0 * float(cfg.ctrl_period)

        # ── Path / FSM state ───────────────────────────────────────────────
        self.path: np.ndarray = np.empty((0, 2))
        # Set by global_path_callback when a new global path is accepted, cleared by run() after
        # the warm-start check. A flag rather than the check itself because a reset_plan() issued
        # from the executor thread can be overwritten by a solve() already in flight, which loses
        # the drop silently; run() acts on it at the tick boundary instead.
        self._new_path_pending: bool = False
        # Current FSM state: "initialization" | "wp_tracking" | "interaction" | "failure_recovery".
        # INITIALIZATION is the SAFE DEFAULT, not just the first state of a run: this node now owns
        # the init motion, so a planner that has not yet heard from the FSM must hold position
        # rather than fall through to wp_tracking and start tracking whatever it has.
        self.fsm_state: str = "initialization"
        # Sub-phase of "initialization", from the INITIALIZATION_<PHASE> string the FSM publishes:
        # "settling" | "rotate" | "hold". Only read while fsm_state == "initialization".
        self.init_phase: str = "settling"
        # FSM state seen on the previous control tick; drives transition-hook dispatch in run().
        # Initialized equal to fsm_state so no spurious enter-hook fires on the first tick.
        self._prev_mode: str = "initialization"

        # ── Config / goal ─────────────────────────────────────────────────
        self.GOAL_THRESHOLD: float = float(cfg.goal_threshold)   # m
        self.CTRL_PERIOD: float    = float(cfg.ctrl_period)
        # 'mppi' — the only legal value: the pure-pursuit planner was removed. Kept as a field
        # so the effective planner is still named in the startup config report and a stale
        # config fails at construction rather than running MPPI under the wrong name.
        self.LOCAL_PLANNER_TYPE: str = str(
            getattr(cfg, "local_planner_type", "mppi")).strip().lower()
        # NOTE: the per-mode command ceilings (conservative_speed_*, maximum_cmd_*) now live in
        # MPPIConfig and are
        # validated by its __post_init__, because the ceiling is a pure function of the mode
        # (MPPIPlanner._mode_cmd_limits) and nothing this node measures enters it. They are read
        # back from self.planner.config for the startup report only.
        # Fixed waypoint lookahead distance (m).
        self.LOOKAHEAD_DIST: float = float(cfg.waypoint_lookahead_distance)
        # Total angular lag (s) from issuing a command to the robot actually turning at it:
        # actuator response + velocity-estimator group delay + control period. Every geometric
        # quantity in wp_tracking is evaluated at the pose rolled forward by this much.
        # Read from mppi_lag_time: that is the same physical quantity (plant step-response
        # settling time) and it is the one the MPPI rollout integrates, so the node reports the
        # lag the planner actually compensates for. The former cmd_lag_time key was a
        # pure-pursuit-era duplicate of it and no longer exists in planning_config.yaml.
        self.CMD_LAG_TIME: float       = float(cfg.mppi_lag_time)
        # Spacing (m) of the global path as actually delivered on /global_path. The planner
        # samples at cfg.sampling_dist, but the final smoothing pass (when enabled) re-spaces the
        # output at cfg.path_smoothing_step — see the path-smoothing block in planning_config.yaml.
        self.PATH_SPACING: float = (float(cfg.path_smoothing_step) if bool(cfg.path_smoothing_enable)
                                    else float(cfg.sampling_dist))
        # Distance (m) to the last path node at which wp_tracking declares arrival and holds.
        # Numerical robustness, not a tuning knob: once the robot is on top of the final node the
        # polyline has no forward direction left, the bearing to the waypoint saturates at the
        # +/-90 deg clamp, and the resulting phantom curvature (2/LOOKAHEAD_DIST) makes the robot
        # orbit the endpoint at cmd_w_limit/kappa forever instead of stopping. DERIVED from the
        # delivered path spacing (not a fixed 0.15 m) so the "last segment always has a direction"
        # property survives enabling path smoothing, which widens the spacing to 0.25 m.
        self.PATH_END_STOP: float = 1.5 * self.PATH_SPACING
        # ── Initialization motion (this node commands it, the FSM schedules it) ───────────
        # The init window used to be a sim-time gate here that only held position while IsaacLab
        # rotated the robot itself. IsaacLab no longer touches the command: NavigationFSM publishes
        # INITIALIZATION_SETTLING / _ROTATE / _HOLD on /fsm_state, and the branch at the top of
        # run() issues every /robot/cmd_vel of those phases. Hence no initialization_time here —
        # this node reads only WHAT to command, never WHEN.
        self.INIT_YAW_RATE: float = float(cfg.init_yaw_rate)                    # rad/s commanded during the sweep
        self.INIT_YAW_AMP: float = math.radians(float(cfg.init_yaw_amplitude_deg))  # rad; sweep half-range
        # Pacing of the init branch alone. That branch runs no MPPI solve, so it ticks far faster
        # than ctrl_period; the sweep's phase switches are detected once per tick, so the tick
        # period IS the overshoot at the commanded yaw rate (0.3 s -> ~17 deg, 0.05 s -> ~3 deg).
        self.INIT_CTRL_PERIOD: float = float(cfg.init_ctrl_period)
        self.INIT_RECENTER_TOL: float = 0.05    # rad; |rel yaw| below this ends the recentre phase
        # Sweep state, valid only inside the ROTATE phase. _init_yaw0 is the reference heading,
        # captured on the FIRST rotate tick (not at node start) so it is the heading the robot
        # actually holds when the sweep begins.
        self._init_yaw0: float | None = None
        self._init_sweep_phase: str = "left"    # "left" -> "right" -> "recenter" -> "done"
        # Plan visualization on/off; the flag is the same one main_worldmodel.py and
        # main_global_planning.py already honour. For MPPI this now gates the RViz
        # MarkerArray publisher (see mppi_rviz), which costs ~0.03 ms per tick instead of
        # the ~20 ms matplotlib redraw that used to dominate this node — so it no longer
        # has to be switched off per machine for timing reasons. With the flag False no
        # publisher is created and the planner keeps its no-op visualizer.
        self.VISUALIZE: bool = bool(cfg.debugging_visualization)
        # Per-tick trace output on/off (wp_tracking diagnostic and the [mode] vx/wz
        # heartbeat). Read once here rather than per tick; same
        # cfg.debug_print flag GlobalMap already honours. Warnings and errors are
        # deliberately NOT gated on it — only the routine per-tick trace.
        self.DEBUG_PRINT: bool = bool(cfg.debug_print)
        # Control-loop iteration counter. Printed in the per-tick trace delimiter so a gap in
        # the numbering identifies a dropped or stalled tick, which at 1/ctrl_period is
        # otherwise invisible in a log that only carries throttled messages.
        self._tick_count: int = 0
        self.goal_x: float = cfg.global_goal[0]
        self.goal_y: float = cfg.global_goal[1]

        # ── Traversability map reader (shared with planner) ────────────────────
        # _backend must be kept alive on self; if only held locally it is GC'd
        # while _map still holds a dangling shared-memory handle.
        self._backend = SharedMemoryBackend(mode='reader')
        self._map     = CMDbasedMapReaderProxy(self._backend)

        # ── Local planner (MPPI) ───────────────────────────────────────────
        # MPPI takes its whole tuning from cfg through MPPIConfig.from_source, so the factory
        # needs nothing beyond the config and the shared map reader.
        # MPPI is the only local planner in this package (the pure-pursuit controller and
        # its one-type factory are gone), so a stale local_planner_type is rejected here
        # rather than silently running MPPI under the wrong name.
        if self.LOCAL_PLANNER_TYPE != "mppi":
            raise ValueError(
                "local_planner_type must be 'mppi' (the only local planner in this package; "
                f"'pure_pursuit' was removed), got {self.LOCAL_PLANNER_TYPE!r}")
        self.planner = MPPIPlanner(MPPIConfig.from_source(cfg), self._map)
        self._prev_ros_time = self.get_clock().now()

        # ── Stall boost (wp_tracking) ──────────────────────────────────────
        # The locomotion policy can refuse to follow a slow command -- a pivot at the
        # conservative_speed_w floor MPPI is capped to in a too-risky cell (navigation-2026-10-01
        # runs 009/027) -- and MPPI then re-issues the same command forever because the state never
        # changes. _update_stall_boost swaps in a config with higher conservative_speed_v/w floors
        # while that stall lasts. See the "Stall boost" block in planning_config.yaml. Missing
        # keys default to OFF so an older config behaves unchanged.
        self.STALL_ENABLE: bool = bool(getattr(cfg, "stall_boost_enable", False))
        self.STALL_TIME: float = float(getattr(cfg, "stall_time", 1.5))                  # (s) stall must persist this long
        self.STALL_CMD_V_MIN: float = float(getattr(cfg, "stall_cmd_v_min", 0.2))        # (m/s) |v_cmd| at least this = linear axis commanded
        self.STALL_CMD_W_MIN: float = float(getattr(cfg, "stall_cmd_w_min", 0.4))        # (rad/s) |w_cmd| at least this = angular axis commanded
        self.STALL_MEAS_V_MAX: float = float(getattr(cfg, "stall_meas_v_max", 0.1))      # (m/s) |avg_v| below this = not translating
        self.STALL_MEAS_W_MAX: float = float(getattr(cfg, "stall_meas_w_max", 0.1))      # (rad/s) |avg_w| below this = not turning
        self.STALL_BOOST_V: float = float(getattr(cfg, "stall_boost_v", 1.0))            # (m/s) conservative_speed_v while boosted
        self.STALL_BOOST_W: float = float(getattr(cfg, "stall_boost_w", 1.0))            # (rad/s) conservative_speed_w while boosted
        self.STALL_RELEASE_DIST: float = float(getattr(cfg, "stall_release_dist", 0.3))  # (m) translation since engage that ends the boost
        # FSM modes the boost may run in.
        self._STALL_MODES: tuple[str, ...] = ("wp_tracking",)
        # The pivot_stall_* keys were renamed to stall_*; a config still carrying them would
        # otherwise run with the boost silently OFF.
        if getattr(cfg, "pivot_stall_boost_enable", None) is not None:
            print("[WARNING][config] pivot_stall_* keys are no longer read: they were renamed to "
                  "stall_* (see the 'Stall boost' block in planning_config.yaml)")
        # The two configs the boost toggles between; MPPIConfig is frozen, so toggling is a
        # reference swap of self.planner.config. Both point at the nominal one while disabled.
        self._stall_cfg_nominal: MPPIConfig = self.planner.config
        self._stall_cfg_boost: MPPIConfig = self.planner.config
        if self.STALL_ENABLE:
            _nominal_v = float(self.planner.config.conservative_speed_v)
            _nominal_w = float(self.planner.config.conservative_speed_w)
            _thresholds = (self.STALL_TIME, self.STALL_CMD_V_MIN, self.STALL_CMD_W_MIN,
                           self.STALL_MEAS_V_MAX, self.STALL_MEAS_W_MAX,
                           self.STALL_BOOST_V, self.STALL_BOOST_W, self.STALL_RELEASE_DIST)
            # Every assumption the detector relies on; any violation aborts startup with the
            # reason named, so a misconfigured boost can never run silently disabled.
            _problems = []
            if not all(math.isfinite(_x) and _x > 0.0 for _x in _thresholds):
                _problems.append(f"every stall_* value must be finite and > 0, got "
                                 f"time={self.STALL_TIME} cmd_v_min={self.STALL_CMD_V_MIN} "
                                 f"cmd_w_min={self.STALL_CMD_W_MIN} "
                                 f"meas_v_max={self.STALL_MEAS_V_MAX} "
                                 f"meas_w_max={self.STALL_MEAS_W_MAX} "
                                 f"boost_v={self.STALL_BOOST_V} boost_w={self.STALL_BOOST_W} "
                                 f"release_dist={self.STALL_RELEASE_DIST}")
            if not (self.STALL_BOOST_V >= _nominal_v and self.STALL_BOOST_W >= _nominal_w
                    and (self.STALL_BOOST_V > _nominal_v or self.STALL_BOOST_W > _nominal_w)):
                _problems.append(f"stall_boost_v/w={self.STALL_BOOST_V:.2f}/{self.STALL_BOOST_W:.2f} "
                                 f"must be >= conservative_speed_v/w={_nominal_v:.2f}/{_nominal_w:.2f} "
                                 f"with at least one strictly above, or the boost changes nothing")
            if self.STALL_CMD_V_MIN > _nominal_v or self.STALL_CMD_W_MIN > _nominal_w:
                # In a too-risky cell |v_cmd| <= conservative_speed_v and |w_cmd| <=
                # conservative_speed_w, so an axis whose threshold sits above it never counts.
                _problems.append(f"stall_cmd_v_min/w_min={self.STALL_CMD_V_MIN:.2f}/"
                                 f"{self.STALL_CMD_W_MIN:.2f} > conservative_speed_v/w="
                                 f"{_nominal_v:.2f}/{_nominal_w:.2f}: a capped command is never detected")
            if (self.STALL_MEAS_V_MAX >= self.STALL_CMD_V_MIN
                    or self.STALL_MEAS_W_MAX >= self.STALL_CMD_W_MIN):
                _problems.append(f"stall_meas_v_max/w_max={self.STALL_MEAS_V_MAX:.2f}/"
                                 f"{self.STALL_MEAS_W_MAX:.2f} must be < stall_cmd_v_min/w_min="
                                 f"{self.STALL_CMD_V_MIN:.2f}/{self.STALL_CMD_W_MIN:.2f}, or a robot "
                                 f"moving as commanded reads as stalled")
            if not _problems:
                # MPPIConfig re-validates on replace (e.g. conservative_speed_v/w <= maximum_cmd_v/w),
                # so its message is reported as-is.
                try:
                    self._stall_cfg_boost = dataclasses.replace(
                        self.planner.config, conservative_speed_v=self.STALL_BOOST_V,
                        conservative_speed_w=self.STALL_BOOST_W)
                except ValueError as exc:
                    _problems.append(f"boosted MPPIConfig rejected: {exc}")
            if _problems:
                raise ValueError(
                    "stall boost misconfigured — fix the stall_* keys in planning_config.yaml or set "
                    "stall_boost_enable: false: " + "; ".join(_problems))
        if self.STALL_ENABLE:
            print(f"[config] stall boost: ON ({', '.join(self._STALL_MODES)}) stall = every "
                  f"commanded axis (|v_cmd|>={self.STALL_CMD_V_MIN:.2f} m/s, "
                  f"|w_cmd|>={self.STALL_CMD_W_MIN:.2f} rad/s) unrealized "
                  f"(|avg_v|<{self.STALL_MEAS_V_MAX:.2f} m/s, |avg_w|<{self.STALL_MEAS_W_MAX:.2f} rad/s) "
                  f"for {self.STALL_TIME:.2f} s -> conservative_speed_v/w "
                  f"{self._stall_cfg_nominal.conservative_speed_v:.2f}/"
                  f"{self._stall_cfg_nominal.conservative_speed_w:.2f} -> "
                  f"{self.STALL_BOOST_V:.2f}/{self.STALL_BOOST_W:.2f} until the robot moves "
                  f"{self.STALL_RELEASE_DIST:.2f} m, stops commanding motion, or changes mode")
        else:
            print("[config] stall boost: OFF")
        self._stall_last_cmd: tuple[float, float] = (0.0, 0.0)  # (v m/s, w rad/s) published last tick
        self._stall_timer_s: float = 0.0    # (s) continuous stalled time so far
        self._stall_active: bool = False    # True while self.planner.config is the boosted one
        self._stall_mode: str | None = None   # FSM mode of the previous update; a change resets
        self._stall_engage_xy: tuple[float, float] = (0.0, 0.0)  # (x m, y m) robot position at engage

        # ── QoS / subscriptions / publishers ──────────────────────────────
        # QoS levels come from the shared ladder (statenav_fsm.qos_profiles); see that
        # module for the level definitions and the full per-topic registry.
        # This node uses level2 and level3 only, so only those two are imported; level1 and
        # level4 stay available in the shared module for future topics.
        # Topics not named in the ladder are placed by the same criteria:
        #   - /robot/twist     continuous rigid-body state, same stream class as
        #                      /robot/pose, and twist_callback integrates a moving
        #                      average over a history of samples          -> level3
        #   - /robot/odom      already named on the level3 ladder; same stream class
        #                      and same moving-average consumer as /robot/twist -> level3
        #   - /global_path     low-rate replanning output; a newer path fully
        #                      supersedes the previous one                -> level2
        # /robot/twist and /robot/odom are the HARDWARE and SIMULATION velocity sources
        # respectively — the zed_wrapper publishes only the former, IsaacLab's ros_manager only
        # the latter. Both are subscribed so avg_v/avg_w stay alive in either deployment without
        # a mode flag; see _ingest_velocity() for how the two streams share one average.
        self.create_subscription(PoseStamped,       "/robot/pose",      self.pose_callback,             qos_profile_level3)
        self.create_subscription(TwistStamped,      "/robot/twist",     self.twist_callback,            qos_profile_level3)
        self.create_subscription(Odometry,          "/robot/odom",      self.odom_callback,             qos_profile_level3)
        self.create_subscription(Path,              "/global_path",     self.global_path_callback,      qos_profile_level2)
        self.create_subscription(String,            "/fsm_state",       self.fsm_state_callback,        qos_profile_level2)

        # /robot/cmd_vel is the control stream consumed by the robot-side level3 subscription.
        self.cmd_pub                  = self.create_publisher(TwistStamped, "/robot/cmd_vel",       qos_profile_level3)

        # ── MPPI plan visualization ───────────────────────────────────────
        # The planner is constructed ROS-free, so the marker publisher is built here
        # and injected. Only built when VISUALIZE is set.
        # level2 is RELIABLE, which RViz's default subscription requires — a
        # BEST_EFFORT publisher would silently never match and the topic would look
        # alive under `topic hz` while RViz showed nothing.
        # Two flags: debugging_visualization is the node-wide gate and mppi_visualization is
        # the MPPI-private one. Both must be set, so existing configs behave unchanged. Both
        # are read off the raw cfg here — MPPIConfig no longer carries a visualization field,
        # because the planner itself never consulted it: attaching the sink IS the switch.
        self._plan_markers = None
        if self.VISUALIZE and bool(getattr(cfg, "mppi_visualization", True)):
            self._plan_markers = MPPIMarkerPublisher(
                self, self.planner,
                ctrl_period=self.CTRL_PERIOD,
                qos_profile=qos_profile_level2,
            )
            self.planner.set_visualizer(self._plan_markers)

        # ── Callback executor (background thread) ──────────────────────────
        # run() used to service callbacks with a single rclpy.spin_once(timeout_sec=0) per
        # control tick, then time.sleep() for the rest of the period. spin_once() executes
        # exactly ONE ready work item, so with five subscriptions plus /clock (use_sim_time)
        # a 10 Hz loop consumed roughly one message per 100 ms and the depth-10 level3 queues
        # stayed permanently full: the pose read by run() was the OLDEST queued sample, and
        # twist_callback almost never ran, which pinned avg_v/avg_w at 0 and silently disabled
        # the lag compensation the wp_tracking law depends on for stability.
        # Spinning in a daemon thread decouples message intake from the control period, so each
        # tick reads the freshest pose/twist. Thread safety: every callback either assigns a
        # scalar attribute, rebinds an ndarray reference, or does deque append/popleft — all
        # single bytecode operations under the GIL, so no additional locking is required.
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        # Report the coupled config values once and warn on the combinations that silently
        # throttle the command instead of failing loudly.
        self._check_config_invariants()


####################################### CALLBACKS #######################################

    def pose_callback(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        q = msg.pose.orientation
        # Single atomic rebind so the control loop can never mix components of two messages
        self.msg_pose = (p.x, p.y, p.z,
                         math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                    1.0 - 2.0 * (q.y ** 2 + q.z ** 2)))
        self._pose_received = True

    def twist_callback(self, msg: TwistStamped) -> None:
        """/robot/twist — the HARDWARE velocity source (zed_wrapper/robot_twist_publisher.py).

        Not published in simulation; see odom_callback for the sim path.
        """
        self.pose_timestamp = msg.header.stamp
        self._ingest_velocity("/robot/twist", msg.header.stamp,
                              msg.twist.linear.x, msg.twist.angular.z)

    def odom_callback(self, msg: Odometry) -> None:
        """/robot/odom — the SIMULATION velocity source (IsaacLab ros_manager, 100 Hz).

        IsaacLab publishes robot/odom and robot/pose but never /robot/twist, so without this
        the whole wp_tracking law ran with avg_v = avg_w = 0 in sim: no lag roll-forward and a
        speed-profile horizon pinned to its floor.

        odom.twist is body-frame (child_frame_id="base", filled from base_lin_vel_b /
        base_ang_vel_b), i.e. the same (v, w) convention /robot/twist publishes and the unicycle
        model in predict_pose expects — so both streams can feed one moving average.
        """
        self._ingest_velocity("/robot/odom", msg.header.stamp,
                              msg.twist.twist.linear.x, msg.twist.twist.angular.z)

    def _ingest_velocity(self, source: str, stamp, v: float, w: float) -> None:
        """Single entry point for measured body-frame velocity, from either source topic.

        Both /robot/twist (hardware) and /robot/odom (sim) carry the same quantity, so they
        share one moving average rather than one winning a mode flag. If both happen to be up
        at once — hardware plus a bag replay — the samples simply interleave: same quantity,
        and the window is bounded by TIME not by sample count, so only the effective rate
        doubles.

        Inputs : source (str) topic name, used only in diagnostics; stamp (builtin_interfaces
                 Time) message header stamp; v (m/s) forward velocity, w (rad/s) yaw rate,
                 both body-frame.
        Post-condition: avg_v/avg_w reflect every sample within MA_WINDOW_S of `stamp`, and
                 _last_vel_t marks this arrival for the staleness guard.
        """
        # Arrival stamp for the staleness guard in _track_path_common. Taken from the NODE clock,
        # not the header: this measures "is the stream alive", and a publisher that leaves
        # header.stamp unfilled (0) would otherwise look infinitely stale even while publishing.
        self._last_vel_t = self.get_clock().now().nanoseconds * 1e-9

        # Update moving-average buffers for avg_v / avg_w
        # Use header stamp (sim-time aware) instead of wall clock
        now_s = stamp.sec + stamp.nanosec * 1e-9
        # Guard: an unfilled header.stamp (all samples at t=0) makes the cutoff below prune
        # nothing, so the deque grows without bound and the "moving" average is taken over the
        # whole run — including the standstill before launch, which drags avg_v toward 0 and
        # therefore disables lag compensation just as a dead stream would. Fall back to the node
        # clock, which is the same time base and is always populated.
        if now_s <= 0.0:
            now_s = self._last_vel_t
            self.get_logger().warn(
                f"[velocity] {source} header.stamp is unset (0) — the moving-average window "
                f"cannot prune by message time; falling back to the node clock. Fix the "
                f"publisher: avg_v/avg_w feed the lag compensation and the speed-profile horizon.",
                throttle_duration_sec=5.0)
        cutoff = now_s - self.MA_WINDOW_S
        self._v_history.append((now_s, v))
        self._w_history.append((now_s, w))
        # Drop samples older than the window
        while self._v_history and self._v_history[0][0] < cutoff:
            self._v_history.popleft()
        while self._w_history and self._w_history[0][0] < cutoff:
            self._w_history.popleft()
        self.avg_v = sum(_v for _, _v in self._v_history) / len(self._v_history)
        self.avg_w = sum(_w for _, _w in self._w_history) / len(self._w_history)

    def global_path_callback(self, msg: Path) -> None:
        points = [[p.pose.position.x, p.pose.position.y] for p in msg.poses]
        # A trackable path needs at least one segment: the local planner projects the robot onto
        # a polyline, and 0 or 1 points carry no forward direction. Reject the message and keep
        # the last accepted path rather than overwriting a good path with an untrackable one.
        # The global planner never publishes an empty Path (it logs "No path to publish" and
        # returns), so no intentional stop signal is discarded here.
        if len(points) < 2:
            self.get_logger().warn(
                f"[global_path] rejected: {len(points)} pose(s), need >= 2 "
                f"— keeping previous path (len={self.path.shape[0]})",
                throttle_duration_sec=1.0)
            return
        self.path = np.array(points)
        # Raised AFTER self.path is bound so run() can never see the flag paired with the previous
        # path. The warm-start decision itself belongs to the tick boundary, not to this thread.
        self._new_path_pending = True

    def fsm_state_callback(self, msg: String) -> None:
        raw   = msg.data
        state = raw.strip().lower()
        # Resolve a transition tag to the state it tags; anything else passes through unchanged
        # and still reaches the unknown-state warning below.
        state = FSM_TRANSITION_TAGS.get(state, state)

        # This callback only validates and records the state. All entry/exit side effects
        # live in the FSM TRANSITION HOOKS region and are dispatched from run(), where the
        # tick's pose and trav-map cmd limits are available.
        # INITIALIZATION_<PHASE> is split into the base state plus its sub-phase: the init branch
        # in run() issues a different command per phase, while every transition-hook path below
        # only ever sees the base "initialization". Handled before the membership test because the
        # phase suffix would otherwise make each of the three a separate unknown state.
        if state.startswith("initialization"):
            self.fsm_state = "initialization"
            self.init_phase = state[len("initialization_"):] if "_" in state else "settling"
        elif state in ("wp_tracking", "interaction", "failure_recovery"):
            self.fsm_state = state
        else:
            print("[Warning] Unknown FSM state: ", state)
            pass   # ignore unknown values

    def _check_config_invariants(self) -> None:
        """Validate the coupled config values once at startup and print the effective numbers.

        Every coupling checked here degrades silently at runtime — it scales the command or the
        waypoint distance down instead of raising — so each one is surfaced explicitly.
        """
        # Every ceiling below is read back from the planner's own config, which validated them
        # in MPPIConfig.__post_init__ — reporting the values the planner will actually use rather
        # than re-reading the raw YAML, so a mapping mistake shows up in this report instead of
        # hiding behind two independent reads of the same key.
        _mppi = self.planner.config
        v_max = float(_mppi.maximum_cmd_v)

        # The world model publishes the travgraph -- the MPPI's only terrain source -- for every
        # global_planner_type (state_nav removed the dense TraversabilityMap, so the graph is its
        # only traversability store), so no planner-type restriction applies here any more.
        print(f"[config] MPPI terrain source: travgraph (lattice {_mppi.travgraph_resolution:.3f} m, "
              f"sigmoid k={_mppi.travgraph_sigmoid_k:g}, rasterize={_mppi.travgraph_rasterize}; "
              f"dense TraversabilityMap not read)")

        # The conservative speed decides the wp_tracking floor, so report the effective numbers
        # once, next to the actuator ceilings they are measured against.
        print(f"[config] conservative speed: v={_mppi.conservative_speed_v:.2f} m/s, "
              f"w={_mppi.conservative_speed_w:.2f} rad/s "
              f"(actuator ceilings v={v_max:.2f}, w={float(_mppi.maximum_cmd_w):.2f}; "
              f"planner={self.LOCAL_PLANNER_TYPE})")
        print("[config]   wp_tracking domain = max(conservative, travmap value)")
        # Every mode now plans against the same lookahead waypoint: get_waypoint queried at
        # horizon * mppi_dt * lookahead_time_scale seconds and max(conservative_speed_v, the
        # mode ceiling), so the window is the same one the global planners aim at instead of a
        # per-mode arc budget.
        _horizon_time_s = (float(cfg.mppi_horizon) * float(cfg.mppi_dt)
                           * float(_mppi.lookahead_time_scale))
        print(f"[config]   local reference = get_waypoint at {_horizon_time_s:.2f} s "
              f"({float(cfg.mppi_horizon) * float(cfg.mppi_dt):.2f} s horizon x "
              f"{_mppi.lookahead_time_scale:.2f}) x "
              f"max(conservative {_mppi.conservative_speed_v:.2f}, mode ceiling) m/s "
              f"-> {_horizon_time_s * float(_mppi.conservative_speed_v):.2f} m at the "
              f"conservative floor, {_horizon_time_s * v_max:.2f} m at the actuator maximum")
        # Warm-start invalidation policy. No distance limit is reported any more: a new global
        # path drops the plan unconditionally (see the WARMSTART note at the top of this file),
        # so the only surviving policy is the FSM-transition one.
        _warmstart_kept = ", ".join(
            sorted(f"{prev} -> {nxt}" for prev, nxt in WARMSTART_KEEP_TRANSITIONS))
        print(f"[config] warm start: kept across {_warmstart_kept or '(nothing)'}, dropped on "
              f"every other FSM transition; dropped unconditionally on every newly accepted "
              f"global path")
        # The zero seed a dropped plan re-enters solve() with is only a hold-position prior while
        # the knot-zero rate box can actually reach v = 0 in one control period. Below that the
        # coupled simplex grows it back to the linear ceiling, so the drop re-seeds the tick at
        # full speed straight ahead — the opposite of what dropping it is for.
        _seed_zero_acc = v_max / self.CTRL_PERIOD if self.CTRL_PERIOD > 0.0 else float("inf")
        if float(_mppi.lin_acc) < _seed_zero_acc:
            print(f"[WARNING][config]   mppi_lin_acc_ub={float(_mppi.lin_acc):.2f} m/s^2 is below "
                  f"maximum_cmd_v/ctrl_period = {_seed_zero_acc:.2f} m/s^2 — the knot-zero rate "
                  f"floor stays positive above {float(_mppi.lin_acc) * self.CTRL_PERIOD:.2f} m/s, "
                  f"so no rollout can carry v = 0: turn-in-place is unreachable and a dropped "
                  f"warm start re-seeds at the linear ceiling instead of at rest")
        # kappa_max: the tightest curve holdable at full speed, i.e. the sharpest turn the robot
        # can take without having to slow down first
        kappa_max = float(cfg.maximum_cmd_w) / v_max if v_max > 1e-9 else float('inf')
        print(f"[config] wp_tracking law: lookahead={self.LOOKAHEAD_DIST:.2f} m  "
              f"mppi_lag_time={self.CMD_LAG_TIME:.2f} s  "
              f"lin_rate=+/-{float(cfg.mppi_lin_acc_ub):.2f} m/s^2  "
              f"ang_acc={float(cfg.mppi_ang_acc_ub):.2f} rad/s^2  ma_window={self.MA_WINDOW_S:.2f} s")
        print(f"[config] at maximum_cmd_v={v_max:.2f} m/s: lag advance "
              f"{v_max * self.CMD_LAG_TIME:.2f} m, tightest curve holdable at full speed "
              f"R={1.0 / kappa_max if kappa_max > 0 else 0:.2f} m")
        print(f"[config] path spacing {self.PATH_SPACING:.2f} m "
              f"({'smoothed: path_smoothing_step' if bool(cfg.path_smoothing_enable) else 'raw: sampling_dist'})"
              f" -> path-end stop at {self.PATH_END_STOP:.2f} m  "
              f"visualization={'ON' if self.VISUALIZE else 'OFF'}")

        # Lag compensation is a stability condition, not a refinement: with mppi_lag_time at 0 the
        # loop limit-cycles on a straight path and saturates well below the speed limit.
        if self.CMD_LAG_TIME <= 0.0:
            print(f"[WARNING][config] mppi_lag_time={self.CMD_LAG_TIME:.2f} s disables lag compensation "
                  f"— expect oscillation on straight segments and roughly a third of the speed limit")
        # The estimator delay is one component of the total lag; if the moving average alone exceeds
        # it, the total lag is under-counted no matter what the actuator does.
        if self.MA_WINDOW_S / 2.0 > self.CMD_LAG_TIME:
            print(f"[WARNING][config] ma_window_s/2 ({self.MA_WINDOW_S / 2.0:.2f} s) exceeds "
                  f"mppi_lag_time={self.CMD_LAG_TIME:.2f} s — the estimator delay alone is larger than "
                  f"the lag being compensated; shorten ma_window_s or raise mppi_lag_time")
        # Above this speed the lag advance exceeds the lookahead, so the pose the waypoint is
        # queried from sits beyond its own waypoint. Informational only: capping the advance was
        # measured to roughly double the cross-track error, so it is left uncapped on purpose.
        # Worth knowing which speeds run in that regime.
        v_overtake = self.LOOKAHEAD_DIST / self.CMD_LAG_TIME if self.CMD_LAG_TIME > 0 else float('inf')
        if v_overtake < v_max:
            print(f"[config] above {v_overtake:.2f} m/s the lag advance exceeds the lookahead "
                  f"({v_max * self.CMD_LAG_TIME:.2f} m vs {self.LOOKAHEAD_DIST:.2f} m at "
                  f"maximum_cmd_v) — left uncapped deliberately; raise "
                  f"waypoint_lookahead_distance if cornering degrades at top speed")

        # PATH_END_STOP is derived from the delivered path spacing, so widening that spacing (a
        # larger path_smoothing_step) grows the arrival radius too. Past the lookahead it would
        # declare arrival before tracking ever engages.
        if self.PATH_END_STOP >= self.LOOKAHEAD_DIST:
            print(f"[WARNING][config] path-end stop {self.PATH_END_STOP:.2f} m (= 1.5 x path spacing "
                  f"{self.PATH_SPACING:.2f} m) >= waypoint_lookahead_distance {self.LOOKAHEAD_DIST:.2f} m "
                  f"— wp_tracking would hold position instead of tracking; lower path_smoothing_step "
                  f"or raise waypoint_lookahead_distance")

    def _warn_if_velocity_stale(self) -> bool:
        """Per-tick liveness check on the measured-velocity stream. True when it is stale.

        Stale means NEITHER source is delivering: /robot/twist (hardware) and /robot/odom (sim)
        both feed _ingest_velocity, and only one of them exists in any given deployment, so the
        check is on the shared arrival stamp rather than per-topic.

        Warn-only by design: the wp_tracking law still produces a usable (if lag-blind) command
        from the un-advanced pose, and holding position on a dropped velocity stream would be a
        worse failure than tracking without lag compensation. The point is that the degradation
        stops being invisible — previously the only symptom was `avg=0.00` buried in the
        per-tick trace, with overshoot as the observable effect.

        Reports the derived quantity that actually changes behaviour, so the log says what
        broke rather than only that something did:
          - lag roll-forward = avg_v*CMD_LAG_TIME (m) and avg_w*CMD_LAG_TIME (deg), frozen at
            whatever they held when the stream died — 0/0 from startup, which makes
            predict_pose() a no-op via its at-rest early return
        """
        age = self.get_clock().now().nanoseconds * 1e-9 - self._last_vel_t
        # _last_vel_t < 0 means no sample has ever arrived; treat as stale from startup.
        if self._last_vel_t >= 0.0 and age <= self.VEL_STALE_S:
            return False

        # Built separately rather than inline: a nested f-string inside the message below is
        # a syntax error before Python 3.12, and this node runs on 3.10.
        _what = ("has never been received" if self._last_vel_t < 0.0
                 else f"is stale by {age:.2f} s")
        self.get_logger().error(
            f"[wp_tracking] measured velocity {_what} on either /robot/twist (hardware) or "
            f"/robot/odom (sim) "
            f"(threshold {self.VEL_STALE_S:.2f} s) — avg_v={self.avg_v:.2f} m/s "
            f"avg_w={self.avg_w:+.2f} rad/s are frozen, so the lag roll-forward is stuck at "
            f"{self.avg_v * self.CMD_LAG_TIME:.2f} m / "
            f"{math.degrees(self.avg_w * self.CMD_LAG_TIME):+.1f} deg. "
            f"Expect overshoot and limit-cycling. Check whichever publisher this deployment "
            f"uses is alive and that its QoS is compatible with level3 (RELIABLE, depth 10): "
            f"ros2 topic hz /robot/odom /robot/twist",
            throttle_duration_sec=2.0)
        return True


####################################### FSM PER-STATE ACTIONS #######################################

    def _init_sweep_yaw_rate(self) -> float:
        """Yaw rate (rad/s) for this tick of the INITIALIZATION_ROTATE sweep.

        An in-place +/-INIT_YAW_AMP scan that gives perception a first look around before
        navigation starts: 'left' (to +amp) -> 'right' (to -amp) -> 'recenter' (back to the start
        heading) -> 'done' (0.0 from then on). Closed-loop on /robot/pose yaw and carrying NO time
        budget of its own: the FSM ends the phase, and a sweep that has not converged by then is
        simply stopped by INITIALIZATION_HOLD. The phase tests fall through rather than chaining
        on elif so a switch takes effect on the tick that detects it.
        """
        # The reference heading is latched from a measured pose, never from the all-zeros
        # placeholder msg_pose starts at — that would make the sweep relative to the world frame
        # and send the recentre to heading 0 instead of back to where the robot stands.
        if not self._pose_received:
            self.get_logger().warning(
                "[init] ROTATE phase but no /robot/pose received yet — commanding zero yaw until "
                "one arrives.", throttle_duration_sec=1.0)
            return 0.0

        robot_yaw = wrap_to_pi(self.msg_pose[3])
        if self._init_yaw0 is None:
            self._init_yaw0 = robot_yaw
            print(f"[init] sweep start: yaw0={math.degrees(robot_yaw):+.1f}deg, "
                  f"amplitude=+/-{math.degrees(self.INIT_YAW_AMP):.0f}deg at "
                  f"{self.INIT_YAW_RATE:.2f} rad/s")
        # Heading relative to the reference, wrapped to (-pi, pi]; stays within +/-amp all sweep.
        rel = wrap_to_pi(robot_yaw - self._init_yaw0)

        if self._init_sweep_phase == "left":
            if rel < self.INIT_YAW_AMP:
                return +self.INIT_YAW_RATE
            self._init_sweep_phase = "right"
        if self._init_sweep_phase == "right":
            if rel > -self.INIT_YAW_AMP:
                return -self.INIT_YAW_RATE
            self._init_sweep_phase = "recenter"
        if self._init_sweep_phase == "recenter":
            if abs(rel) > self.INIT_RECENTER_TOL:
                return -self.INIT_YAW_RATE if rel > 0.0 else +self.INIT_YAW_RATE
            self._init_sweep_phase = "done"
            print(f"[init] sweep complete: within {math.degrees(abs(rel)):.1f}deg of the start "
                  "heading; holding zero until the FSM leaves INITIALIZATION")
        return 0.0

    def _update_stall_boost(self, mode: str, dt: float, robot_x: float, robot_y: float) -> None:
        """Choose the MPPIConfig this tick solves with: nominal, or boosted conservative_speed_v/w.

        Not an FSM state and issues no command. conservative_speed_v/w are the floors
        _command_limits caps MPPI to in a too-risky cell, and the locomotion policy can refuse to
        follow a command at those rates while MPPI re-issues it (navigation-2026-10-01 runs
        009/027: v_cmd ~ 0, |w_cmd| ~ 0.6, measured yaw rate ~ 0 until timeout).

          engage  — mode in _STALL_MODES and unchanged since the previous update, the command
                    published last tick asked for motion on at least one axis
                    (|v| >= STALL_CMD_V_MIN or |w| >= STALL_CMD_W_MIN), and EVERY axis it asked
                    for stayed unrealized (|avg_v| < STALL_MEAS_V_MAX, |avg_w| <
                    STALL_MEAS_W_MAX) for STALL_TIME. With |v| below STALL_CMD_V_MIN this is
                    exactly the former pivot-only rule.
          release — the mode changed or is not in _STALL_MODES, that command no longer asks for
                    motion on either axis (hold), or the robot has translated STALL_RELEASE_DIST
                    from the engage position. Measured motion alone does NOT release, or the boost
                    would be dropped on the tick it starts working; a pivot never translates, so
                    the boost holds for the whole pivot as before.

        Args:
            mode: this tick's FSM state name.
            dt: (s) this tick's control interval.
            robot_x, robot_y: (m) this tick's robot position, for the release distance.
        """
        if not self.STALL_ENABLE:
            return
        last_v, last_w = self._stall_last_cmd
        # Which axes the command published last tick actually asked for.
        drive = abs(last_v) >= self.STALL_CMD_V_MIN
        turn = abs(last_w) >= self.STALL_CMD_W_MIN
        # A mode switch starts a new stall episode: the
        # previous mode's command and timer are not evidence about this one.
        mode_changed = mode != self._stall_mode
        self._stall_mode = mode
        qualifies = mode in self._STALL_MODES and not mode_changed and (drive or turn)
        # (m) translation since the boost engaged; 0 while not engaged. A non-finite pose gives
        # NaN, which never satisfies the release test, so the other release paths still apply.
        moved = (math.hypot(robot_x - self._stall_engage_xy[0], robot_y - self._stall_engage_xy[1])
                 if self._stall_active else 0.0)

        # ── Release: another state owns this tick, a hold was issued, or the robot got out ──
        if not qualifies or moved >= self.STALL_RELEASE_DIST:
            self._stall_timer_s = 0.0
            if self._stall_active:
                self._stall_active = False
                self.planner.config = self._stall_cfg_nominal
                print(f"[stall_boost] released (mode={mode}, last cmd v={last_v:+.2f} "
                      f"w={last_w:+.2f}, moved {moved:.2f} m since engage) — "
                      f"conservative_speed_v/w back to "
                      f"{self._stall_cfg_nominal.conservative_speed_v:.2f}/"
                      f"{self._stall_cfg_nominal.conservative_speed_w:.2f}")
            return
        if self._stall_active:
            return   # still stalled or just getting out under the boost: keep it

        # ── Accumulate: motion is commanded but no commanded axis is moving ──
        # A dead velocity stream freezes avg_v/avg_w (0.0 from startup), which would read as "not
        # moving" without being evidence of it, so the timer neither advances nor clears then.
        vel_age = self.get_clock().now().nanoseconds * 1e-9 - self._last_vel_t
        if self._last_vel_t < 0.0 or vel_age > self.VEL_STALE_S:
            return
        stalled = ((not drive or abs(self.avg_v) < self.STALL_MEAS_V_MAX)
                   and (not turn or abs(self.avg_w) < self.STALL_MEAS_W_MAX))
        if stalled:
            self._stall_timer_s += dt
        else:
            self._stall_timer_s = 0.0

        # ── Engage: next solve gets the higher linear and angular floors ──
        if self._stall_timer_s >= self.STALL_TIME:
            self._stall_active = True
            self.planner.config = self._stall_cfg_boost
            self._stall_engage_xy = (float(robot_x), float(robot_y))
            # (vmax, wmax) the last command was clipped by at the robot's cell, or None before the
            # first solve. The boost changes an axis only where this sits below its boosted floor.
            knot0 = self.planner.last_knot_zero_limits
            knot0_txt = ("n/a" if knot0 is None
                         else f"v={float(knot0[0]):.2f} w={float(knot0[1]):.2f}")
            print(f"[stall_boost] ENGAGED in {mode} after {self._stall_timer_s:.2f} s: last cmd "
                  f"v={last_v:+.2f} w={last_w:+.2f} but measured avg_v={self.avg_v:+.3f} m/s "
                  f"avg_w={self.avg_w:+.3f} rad/s — conservative_speed_v/w "
                  f"{self._stall_cfg_nominal.conservative_speed_v:.2f}/"
                  f"{self._stall_cfg_nominal.conservative_speed_w:.2f} -> "
                  f"{self.STALL_BOOST_V:.2f}/{self.STALL_BOOST_W:.2f} (knot0 limits before boost: "
                  f"{knot0_txt})")

    def _run_WP_TRACKING(self, robot_x: float, robot_y: float, robot_yaw: float,
                         at_goal: bool, active_path: np.ndarray,
                         dt: float) -> tuple[float, float, np.ndarray | None]:
        """Actions executed each control tick while in WP_TRACKING.

        Holds only what is this state's own; every hold-position guard and the result drain
        live in _track_path_common(), which this method delegates to.
        The two things owned here:

          the polyline — active_path, the global path.
          the planner entrypoint — MPPIPlanner.track_path(), whose command ceiling is the actuator
              maximum and whose local-reference length is speed-derived and shortened on curves.
        """
        return self._track_path_common(
            "wp_tracking", robot_x, robot_y, robot_yaw, at_goal, active_path, dt,
            self.planner.track_path)

    def _track_path_common(self, mode: str, robot_x: float, robot_y: float, robot_yaw: float,
                           at_goal: bool, active_path: np.ndarray,
                           dt: float,
                           track_fn: Callable[..., MPPIPathTrackingResult],
                           ) -> tuple[float, float, np.ndarray | None]:
        """The path-tracking law of WP_TRACKING. Not an FSM state itself.

        Holds every hold-position guard, the planner call and the warning/diagnostic drain.
        What the calling _run_<STATE> method decides arrives as an argument — active_path (the
        polyline) and track_fn (the planner entrypoint, which owns the command ceilings and the
        local-reference length as its own mode policy). `mode` is the tag on every log line
        below.

        Lag-aware path following; the maths lives in MPPIPlanner._track_path().

        Everything geometric is evaluated at a pose rolled forward by CMD_LAG_TIME, the total
        angular lag (actuator + estimator + control period). This is a stability requirement,
        not a refinement: without it the loop limit-cycles on a straight path and saturates at
        roughly a third of the speed limit. at_goal is evaluated on the actual pose by the caller.
        """
        if at_goal:
            self.planner.reset()   # commanded velocity is zero — keep the rate-bound anchor honest
            return 0.0, 0.0, None

        # ── Arrival at the end of the path ────────────────────────────────
        # Checked before any geometry: sitting on the final node leaves the polyline with no
        # forward direction, the bearing to the waypoint pins at the +/-90 deg clamp, and the
        # phantom curvature that produces makes the robot orbit the endpoint at a steady
        # cmd_w_limit/kappa instead of stopping. at_goal only covers the case where the path end
        # IS the global goal.
        if active_path.shape[0] > 0:
            end_dist = math.hypot(float(active_path[-1, 0]) - robot_x,
                                  float(active_path[-1, 1]) - robot_y)
            if end_dist < self.PATH_END_STOP:
                self.planner.reset()
                self.get_logger().info(
                    f"[{mode}] reached the end of the path ({end_dist:.2f} m < "
                    f"{self.PATH_END_STOP:.2f} m) — holding position",
                    throttle_duration_sec=1.0)
                return 0.0, 0.0, active_path[-1]

        # Velocity-stream liveness remains a node concern because it depends on ROS time and
        # ROS logging. The planner continues on the un-advanced pose as a safe fallback.
        self._warn_if_velocity_stale()

        # ── No trackable path ─────────────────────────────────────────────
        # Reached before the first global path arrives (self.path is still empty). Handing the
        # planner a polyline with no forward direction only produces a shape-error warning that
        # hides the real situation, so the situation is named here and the robot holds.
        # global_path_callback rejects sub-two-point messages, so an accepted path can no
        # longer reach this branch. Kept below the
        # velocity-liveness check so a dead velocity publisher is still reported during the
        # pre-path startup window, which is where that warning earns its keep.
        if active_path.shape[0] < 2:
            self.planner.reset()
            self.get_logger().info(
                f"[{mode}] no global path received yet — holding position"
                if self.path.shape[0] == 0 else
                f"[{mode}] active path has {active_path.shape[0]} point(s), need >= 2 "
                f"(global path len={self.path.shape[0]}) — holding position",
                throttle_duration_sec=1.0)
            return 0.0, 0.0, None

        # The entrypoint is supplied by the calling state method (_run_WP_TRACKING passes
        # MPPIPlanner.track_path). The ceilings, the local-reference
        # length and the mode tag on every diagnostic all come from the planner's own mode policy,
        # so nothing terrain-derived is computed here.
        result = track_fn(
            robot_x, robot_y, robot_yaw, self.avg_v, self.avg_w, dt, active_path)
        for warning in result.warnings:
            self.get_logger().warn(warning, throttle_duration_sec=1.0)
        if self.DEBUG_PRINT and result.diagnostic is not None:
            self.get_logger().info(
                result.diagnostic,
                # Throttled just under ctrl_period so no control tick is dropped by timing jitter:
                # at exactly ctrl_period a tick arriving 1 us early is silently skipped, which
                # would leave gaps in the trace this line exists to produce.
                throttle_duration_sec=self.CTRL_PERIOD * 0.9)
        return result.v_cmd, result.w_cmd, result.waypoint

    def _publish_state_output(self, mode: str, vx: float, wz: float, at_goal: bool) -> None:
        """Publish the output owned by the active FSM state.

        Every state publishes one TwistStamped per tick. Skipping the publish here would leave
        the robot without a command stream AND leave the planner's rate anchor
        (record_issued_command below) stale, which knot zero of the next solve is bounded against.
        """
        cmd = TwistStamped()
        cmd.header.stamp    = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base"
        cmd.twist.linear.x  = vx
        cmd.twist.angular.z = wz
        if at_goal:
            cmd.twist.angular.x = 1.0   # FSM stop indicator
        self.cmd_pub.publish(cmd)
        self.planner.record_issued_command(vx, wz)


####################################### RUN #######################################

    def run(self) -> None:
        # Loop pacing, SIM clock. create_rate() sleeps on the NODE clock, so with use_sim_time=True
        # each iteration waits until CTRL_PERIOD of /clock has elapsed rather than CTRL_PERIOD of
        # wall time. The previous time.monotonic()/time.sleep() pacing made the control period a
        # wall quantity, which then leaked into the control law: on a sim slower than realtime the
        # true ROS delta fell below CTRL_PERIOD and the max(dt, CTRL_PERIOD) floor below replaced
        # it, so every dt consumer was effectively paced
        # by the wall clock. With sim-time pacing the raw delta is already ~CTRL_PERIOD and that
        # floor only fires on a genuine clock stall, which is what it was written for.
        # Blocking here is safe: callbacks are serviced by the executor thread started in
        # __init__, so rate.sleep() stalls only this loop.
        # CAVEAT: if /clock stops (sim paused or dead) rate.sleep() blocks until it resumes and no
        # command is published meanwhile — intended, a frozen sim should freeze the controller.
        # Startup visibility for that caveat: with use_sim_time=True and no /clock publisher yet
        # the rate timer never fires, so the loop below blocks on its first sleep and emits nothing
        # further. Name the condition once here so a stalled start is diagnosable rather than mute.
        if self.get_clock().now().nanoseconds == 0:
            self.get_logger().warning(
                "[init] ROS clock still reads 0 — waiting for /clock (use_sim_time=True). The "
                "control loop paces on sim time and will not tick until the simulator publishes "
                "/clock; no command is sent meanwhile.")
        rate = self.create_rate(1.0 / self.CTRL_PERIOD)
        # Separate, faster pacing for the initialization branch only. That branch runs no MPPI
        # solve, and its sweep switches direction on a per-tick heading test, so pacing it at
        # CTRL_PERIOD would overshoot each switch by CTRL_PERIOD * INIT_YAW_RATE (~17 deg at
        # 0.3 s and 1.0 rad/s) — more than the recentre tolerance it has to hit.
        init_rate = self.create_rate(1.0 / self.INIT_CTRL_PERIOD)
        while rclpy.ok():
            # No spin_once() here: callbacks are serviced continuously by the executor thread
            # started in __init__, so this tick reads the freshest pose/twist rather than one
            # message per 100 ms out of a permanently full queue. See the executor block there.

            # Sim-time stamp of this tick (use_sim_time=True), used by the trace below and the
            # initialization branch. The init window is no longer a time comparison here: the FSM
            # owns those boundaries and publishes the resulting phase.
            sim_time = self.get_clock().now().nanoseconds * 1e-9

            # ── Per-tick trace delimiter ──────────────────────────────────────
            # One separator per control loop iteration, emitted BEFORE the initialization gate
            # so that EVERY tick opens a block — including ticks that hold position and never
            # reach the planner at all, which are otherwise indistinguishable in the log from a
            # loop that stopped running. Everything the tick prints (warnings, the planner
            # diagnostic, the MPPI tuning trace, the heartbeat) lands underneath it.
            # print() rather than get_logger() so the delimiter shares stdout with the
            # planner's own trace and the block stays contiguous; the ROS logger writes to
            # rosout on a separate stream and would interleave.
            self._tick_count += 1
            if self.DEBUG_PRINT:
                print(f"{'=' * 78}\n"
                      f"[tick {self._tick_count}] sim_time={sim_time:.3f}s "
                      f"fsm={self.fsm_state}")

            # ── Initialization phase (FSM-scheduled, planner-commanded) ───────
            # NavigationFSM decides WHICH init phase the system is in and publishes it on
            # /fsm_state; this branch issues the only /robot/cmd_vel of that phase — zero while
            # settling and during the trailing hold, an in-place yaw sweep in between. IsaacLab
            # does not touch the command any more.
            if self.fsm_state == "initialization":
                wz = self._init_sweep_yaw_rate() if self.init_phase == "rotate" else 0.0
                cmd = TwistStamped()
                cmd.header.stamp    = self.get_clock().now().to_msg()
                cmd.header.frame_id = "base"
                cmd.twist.angular.z = wz   # linear.x / linear.y stay 0: in-place rotation only
                self.cmd_pub.publish(cmd)
                self.get_logger().info(
                    f"[init] phase={self.init_phase} t={sim_time:.2f}s wz={wz:+.2f} rad/s "
                    f"(sweep={self._init_sweep_phase})",
                    throttle_duration_sec=1.0,
                )
                # Keep _prev_ros_time current so the first post-init dt is ~one
                # control period instead of the full initialization span.
                self._prev_ros_time = self.get_clock().now()
                init_rate.sleep()   # sim-clock pacing at INIT_CTRL_PERIOD, see run()'s preamble
                continue

            _now_ros_time       = self.get_clock().now()
            dt                  = (_now_ros_time - self._prev_ros_time).nanoseconds * 1e-9
            dt                  = max(dt, self.CTRL_PERIOD)  # prevent zero-dt when the sim clock stalls between control ticks
            self._prev_ros_time = _now_ros_time

            # One read of the pose tuple: all four components come from the same message
            robot_x, robot_y, robot_z, _yaw = self.msg_pose
            robot_yaw = wrap_to_pi(_yaw)

            dist_to_goal = math.hypot(robot_x - self.goal_x, robot_y - self.goal_y)
            at_goal      = dist_to_goal < self.GOAL_THRESHOLD

            vx = wz = 0.0
            wp   = None
            # mode is driven directly by the FSM state: "wp_tracking" | "interaction" | "failure_recovery"
            mode = self.fsm_state

            # Update robot pose on map before querying traversability limits and waypoint
            self._map.robot_x       = robot_x
            self._map.robot_y       = robot_y
            self._map.robot_heading = robot_yaw
            # Same samples as get_cmd_limits(), read from the travgraph the planner plans on, so
            # the dense map is never touched.
            cmd_v_limit, cmd_w_limit = self.planner.travgraph_cmd_limits(
                robot_x, robot_y, robot_yaw, float(cfg.lookahead_distance),
                float(cfg.safecmd_sampling_dist), math.radians(float(cfg.lookahead_angle)))
            # Hand the limits to the planner purely so publish_plan() can display the real terrain
            # limits. Without this the plotted
            # "limit" showed the constructor defaults during wp_tracking.
            self.planner.set_cmd_limits(cmd_v_limit, cmd_w_limit)

            # ── FSM transition hooks ──────────────────────────────────────────
            # Fired here rather than in fsm_state_callback so the hooks see this tick's pose
            # and the same cmd limits the per-state action below will use.
            if mode != self._prev_mode:
                # A state change invalidates the warm start unless the pair is in the keep-set:
                # the cached plan was solved against the previous state's command ceiling,
                # reference budget and rollout geometry. Explicit policy rather than a side
                # effect of the reversed_geometry check inside solve(). reset_plan(), not
                # reset() — the rate anchor is kept so the next command is still bounded from
                # the speed the robot is actually moving at.
                if (self._prev_mode, mode) not in WARMSTART_KEEP_TRANSITIONS:
                    self.planner.reset_plan()
                self._prev_mode = mode

            # ── New-path warm-start drop ──────────────────────────────────────
            # Detected in global_path_callback (a flag, nothing more) and acted on HERE, at the
            # tick boundary, for the same reason the FSM hooks are: this touches planner state a
            # solve() may be in the middle of. Runs after the hooks so a plan this tick already
            # dropped is simply dropped again, which reset_plan() is idempotent about. The flag is
            # cleared unconditionally, so one accepted path message causes at most one drop.
            if self._new_path_pending:
                self._new_path_pending = False
                # UNCONDITIONAL: no endpoint-distance test any more. The plan was solved
                # against the previous polyline, and the case the distance test kept it for
                # — a small lateral shift with a large heading change at the start — is
                # exactly the case a stale plan tracks worst. reset_plan() keeps the rate
                # anchor (_last_issued_command), so the next command is still bounded from
                # the speed the robot is actually moving at.
                # last_result, not nominal_controls: the latter copies the plan off the
                # GPU, which is a device sync on the control thread just to test for None.
                # Both are cleared by reset_plan() and both are written by every solve.
                _ws_had_plan = self.planner.last_result is not None
                self.planner.reset_plan()
                if _ws_had_plan:
                    print(f"[warmstart] new global path (len={self.path.shape[0]}) in "
                          f"{mode}: plan dropped (unconditional)")

            # Stall boost: picks the planner config for this tick's solve, so it must run
            # before any planner call and releases before any mode switch can solve.
            self._update_stall_boost(mode, dt, robot_x, robot_y)

            # ── Dispatch to per-state action ──────────────────────────────────
            # One _run_<STATE> method per FSM state, each owning what only that state knows: the
            # polyline it tracks and which planner entrypoint it calls. The law lives in
            # _track_path_common(), and the command ceilings plus the local-reference length are
            # the planner's own mode policy there.
            if mode == "wp_tracking":
                vx, wz, wp = self._run_WP_TRACKING(
                    robot_x, robot_y, robot_yaw, at_goal, self.path, dt)
            else:
                # Reached only if the FSM enters a state this node does not handle
                # (fsm_state_callback still accepts the FSM's 'interaction' and
                # 'failure_recovery'). Named rather than falling
                # through mutely with vx = wz = 0.0, which is indistinguishable in the log from a
                # legitimate hold.
                self.planner.reset()
                self.get_logger().error(
                    f"[{mode}] unknown FSM state (expected 'wp_tracking') — holding position")

            # ── Publish ──────────────────────────────────────────────────────
            self._publish_state_output(mode, vx, wz, at_goal)
            self._stall_last_cmd = (vx, wz)   # read by next tick's _update_stall_boost

            # ── Plot ─────────────────────────────────────────────────────────
            # Skipped entirely when visualization is off. For MPPI this now only builds and
            # publishes a MarkerArray (~0.03 ms); the ~20 ms matplotlib redraw that used to push
            # the tick past ctrl_period on slower machines is gone. See mppi_rviz.
            if self.VISUALIZE:
                # Keep the planner-facing (x, y, yaw) indices stable and append z only
                # for RViz, where the MPPI rollout uses the robot's current height.
                robot_pose_arr = np.array([robot_x, robot_y, robot_yaw, robot_z])
                self.planner.publish_plan(self.path, wp, robot_pose_arr, vx, wz,
                                       mode=mode, time=_now_ros_time.nanoseconds * 1e-9)

            if self.DEBUG_PRINT:
                self.get_logger().info(
                    f"[{mode}] vx={vx:.3f}  wz={wz:.3f}  dist_to_goal={dist_to_goal:.2f} m",
                    throttle_duration_sec=1.0,
                )

            # Tick cost measured on the SIM clock against a sim-clock budget: _now_ros_time is this
            # tick's start stamp, so both sides of the comparison are /clock seconds. This fires
            # when the tick body consumes more than one control period of sim time, which is the
            # "cannot keep up" criterion that matters for a sim-paced loop.
            tick_elapsed = (self.get_clock().now() - _now_ros_time).nanoseconds * 1e-9
            if tick_elapsed > self.CTRL_PERIOD * 1.5:
                self.get_logger().warning(
                    f"[{mode}] control is slower than required: {tick_elapsed:.3f}s > "
                    f"{self.CTRL_PERIOD:.3f}s (sim time)"
                )
            rate.sleep()


# ── Termination handling ──────────────────────────────────────────────────────
# Wall-clock budget between a stop request and the unconditional exit. Long enough for one MPPI
# tick plus rclpy teardown, short enough to stay well inside the experiment runner's SIGINT phase
# (shutdown_interrupt_s, 15 s by default) so escalation to SIGTERM/SIGKILL is never needed.
_TERMINATION_GRACE_S = 3.0
# Per-thread wait, in main()'s finally, for the daemon threads still inside rclpy (shutdown worker,
# spin thread). Two joins fit inside _TERMINATION_GRACE_S, so the watchdog stays the backstop.
_THREAD_JOIN_TIMEOUT_S = 1.0


def _install_termination_handlers() -> list[threading.Thread]:
    """Make SIGINT/SIGTERM actually stop this node, even after the sim clock has died.

    rclpy.init() installs its own SIGINT/SIGTERM handling by default, but that handler shuts the
    context down at the C level only: Context.on_shutdown() callbacks registered from Python never
    run, so the Rate that run() sleeps on is never woken. With use_sim_time=True and /clock gone -
    exactly what a teardown looks like, since the simulator is stopped before this node - the
    rate.sleep() at the end of run() blocks forever, `while rclpy.ok()` is never re-evaluated, and
    the process can only be removed with SIGKILL (observed: exit 137 on every experiment teardown).

    This handler instead calls rclpy.try_shutdown() from Python, which does run those callbacks:
    Rate._on_shutdown() sets the wake event, the blocked sleep() raises ROSInterruptException, and
    run() unwinds through main()'s finally block like a normal Ctrl-C. Two safety nets on top:
    a watchdog thread exits unconditionally if the loop is stuck somewhere a context shutdown
    cannot reach (an MPPI solve, a shared-memory read), and a second signal exits immediately.

    The real teardown runs on its own thread, never inside the handler: Context.shutdown() takes
    the context handle and its lock, and the interrupted main thread may already hold them.

    Returns the list the handler appends that shutdown worker to, so main() can join it: returning
    while a daemon thread is still inside rcl races native process teardown (exit 139 / 137).
    """
    stopping = threading.Event()
    shutdown_workers: list[threading.Thread] = []   # started by the handler, joined by main()

    def _shutdown_worker(signum: int) -> None:
        """Perform the graceful context shutdown off the signal handler."""
        try:
            rclpy.try_shutdown()
        except Exception as error:  # a failed teardown must never keep the process alive
            print(f"[local_planner] rclpy.try_shutdown() failed while handling signal {signum}: "
                  f"{error!r}. Falling back to the {_TERMINATION_GRACE_S:.1f}s hard exit.",
                  file=sys.stderr, flush=True)

    def _handle_termination(signum, _frame) -> None:
        if stopping.is_set():
            # Asked twice: leave now, skipping every remaining cleanup.
            os._exit(128 + signum)
        stopping.set()
        print(f"[local_planner] signal {signum} received, shutting down (hard exit after "
              f"{_TERMINATION_GRACE_S:.1f}s if the control loop is blocked).",
              file=sys.stderr, flush=True)
        worker = threading.Thread(
            target=_shutdown_worker, args=(signum,), name="mpc-shutdown", daemon=True
        )
        worker.start()
        shutdown_workers.append(worker)   # appended after start(): join() on it is always valid
        watchdog = threading.Timer(_TERMINATION_GRACE_S, os._exit, args=(128 + signum,))
        watchdog.daemon = True
        watchdog.start()

    signal.signal(signal.SIGINT, _handle_termination)
    signal.signal(signal.SIGTERM, _handle_termination)
    return shutdown_workers


def main() -> None:
    os.nice(-20)
    # SignalHandlerOptions.NO: the handlers installed below replace rclpy's, which cannot unblock
    # the sim-clock Rate this node sleeps on. See _install_termination_handlers().
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    shutdown_workers = _install_termination_handlers()
    node = None
    try:
        node = LocalPlannerNode()
        node.run()
    except (KeyboardInterrupt, ROSInterruptException):
        # ROSInterruptException is now the normal stop path: rate.sleep() raises it as soon as the
        # signal handler shuts the context down.
        pass
    finally:
        # Stop the callback executor before tearing the context down, otherwise the spin thread
        # is still inside rclpy when rclpy.shutdown() invalidates the context and prints a
        # spurious traceback on Ctrl-C. Guarded on `node` because construction itself can raise
        # (config validation, shared-memory reader) before the executor exists.
        if node is not None and getattr(node, "_executor", None) is not None:
            node._executor.shutdown()
        # Let the signal handler's rclpy.try_shutdown() finish before deciding on rclpy.shutdown().
        for worker in shutdown_workers:
            worker.join(timeout=_THREAD_JOIN_TIMEOUT_S)
        if rclpy.ok():
            rclpy.shutdown()
        # Join the daemon spin thread last, once the context is down and spin() has returned:
        # returning while it is still inside rcl lets native teardown race it (exit 139 / 137).
        spin_thread = getattr(node, "_spin_thread", None)
        if spin_thread is not None:
            spin_thread.join(timeout=_THREAD_JOIN_TIMEOUT_S)


if __name__ == "__main__":
    main()
