import math
import os
import warnings
from enum import Enum, auto
from pathlib import Path
from omegaconf import OmegaConf

# Name of the environment variable that overrides the planning config location.
# Set this in repos that do not ship statenav_global (IsaacLab, elevation_mapping, gsa).
CONFIG_ENV_VAR = "STATENAV_PLANNING_CONFIG"

# Config keys this FSM reads, with the fallback value used when no config file is found.
# Kept here (rather than inline) so the "running on defaults" warning can name them.
_CONFIG_DEFAULTS = {
    'radius_obs_affected_in_travmap': 0.2,
    'obstacle_interaction_distance_threshold': 0.5,
    'interaction_time_limit': 10.0,
    # Initialization window (see _in_INITIALIZATION). The FSM owns the SCHEDULE only; the local
    # planner issues the commands. init_yaw_rate / init_yaw_amplitude_deg are not read here, but
    # they are what the startup fit-check below sizes the sweep with.
    'initialization_time': 17.0,
    'cloud_settle_time': 3.0,
    'init_hold_time': 1.0,
    'init_yaw_rate': 1.0,
    'init_yaw_amplitude_deg': 90.0,
}


def _resolve_config(config_path=None):
    """Locate and load planning_config.yaml, returning (cfg, source_description).

    This package is installed into several interpreters that do NOT ship
    statenav_global (Isaac Sim's python, the gsa / elevation_mapping workspaces), so the
    config location cannot be derived from this file's own path the way it was when this
    module lived inside statenav_global. Resolution order:

      1. explicit `config_path` argument
      2. $STATENAV_PLANNING_CONFIG
      3. the configs/ directory of an installed statenav_global (imported lazily, so
         statenav_global stays an optional dependency)
      4. built-in defaults, with a warning naming every path that was tried

    Returns an OmegaConf node for 1-3, or None for 4 (caller falls back to
    _CONFIG_DEFAULTS). Loading is deliberately done at NavigationFSM construction rather
    than at module import, so `import statenav_fsm.NavigationFSM` never touches the disk.
    """
    attempted = []  # human-readable list of sources tried, used in the fallback warning

    # 1) Explicit path wins — the caller knows best.
    if config_path is not None:
        p = Path(config_path).expanduser().resolve()
        attempted.append(f"explicit config_path={p}")
        if p.is_file():
            return OmegaConf.load(p), str(p)
        raise FileNotFoundError(
            f"[NavigationFSM] config_path was given but does not exist: {p}"
        )

    # 2) Environment variable — the mechanism for repos without statenav_global.
    env_path = os.environ.get(CONFIG_ENV_VAR)
    if env_path:
        p = Path(env_path).expanduser().resolve()
        attempted.append(f"${CONFIG_ENV_VAR}={p}")
        if p.is_file():
            return OmegaConf.load(p), str(p)
        warnings.warn(
            f"[NavigationFSM] ${CONFIG_ENV_VAR} points at a missing file: {p}. "
            "Continuing with the next resolution step.",
            RuntimeWarning, stacklevel=3,
        )
    else:
        attempted.append(f"${CONFIG_ENV_VAR} (unset)")

    # 3) Installed statenav_global — the in-repo path, kept working unchanged.
    #    Imported inside the function so statenav_global remains optional: interpreters
    #    without it (and without its torch / ROS message deps) just fall through.
    try:
        import statenav_global  # noqa: F401  (imported for its filesystem location only)
        p = Path(statenav_global.__file__).resolve().parent / "configs/planning_config.yaml"
        attempted.append(f"statenav_global package config={p}")
        if p.is_file():
            return OmegaConf.load(p), str(p)
    except Exception as e:  # ImportError, or any failure inside statenav_global's __init__
        attempted.append(f"statenav_global import failed ({type(e).__name__}: {e})")

    # 4) Nothing found — run on defaults, but say so loudly. Silence here would be
    #    dangerous: e.g. interaction_time_limit defaults to 10.0s while the shipped
    #    planning_config.yaml sets 3.0s, so a silent fallback changes FSM timing.
    warnings.warn(
        "[NavigationFSM] No planning config found; falling back to built-in defaults "
        f"{_CONFIG_DEFAULTS}. Tried, in order: " + "; ".join(attempted) + ". "
        f"Set ${CONFIG_ENV_VAR} or pass NavigationFSM(config_path=...) to silence this.",
        RuntimeWarning, stacklevel=3,
    )
    return None, "built-in defaults"


class FSMState(Enum):
    INITIALIZATION = auto()
    WP_TRACKING = auto()
    INTERACTION = auto()
    FAILURE_RECOVERY = auto()


# Sub-phases of INITIALIZATION, in order. They are published as INITIALIZATION_<PHASE> on
# /fsm_state (see published_state_name), mirroring the WP_TRACKING_SUCCESS tag convention: every
# consumer that only cares about the base state collapses the string on "INITIALIZATION".
INIT_PHASES = ('SETTLING', 'ROTATE', 'HOLD')


class NavigationFSM:
    """Finite State Machine for navigation mode management.

    Pure FSM logic with no external dependencies. State data (robot pose,
    target obstacle) is injected by the owning node before each update() call.

    States:
        INITIALIZATION   : startup window (entry state); stand still, sweep the yaw, stand still
        WP_TRACKING      : following global waypoints
        INTERACTION      : robot is manipulating a nearby obstacle
        FAILURE_RECOVERY : robot retreats along reversed path after a failed manipulation

    Transitions:
        INITIALIZATION   -> WP_TRACKING      : the init schedule below reaches initialization_time
        WP_TRACKING      -> INTERACTION      : robot within distance threshold of target_obstacle
        INTERACTION      -> WP_TRACKING      : success (manipulation completed)
        INTERACTION      -> FAILURE_RECOVERY : failure (timeout or explicit failure signal)
        FAILURE_RECOVERY -> WP_TRACKING      : local planner publishes /failure_recovered=True

    INITIALIZATION is TIMING ONLY. This FSM decides which init phase the system is in and
    publishes it; the local planner (Bipedal_Planning_MPC) issues every /robot/cmd_vel of that
    phase, and IsaacLab only feeds the /clock these boundaries are measured on. Nothing here
    commands motion.
    """

    def __init__(self, config_path=None):
        # Config is resolved per-instance (see _resolve_config) rather than at module
        # import, so this module is importable in interpreters that have no planning
        # config on disk. self.config_source records where the values came from and is
        # worth logging by the owning node.
        cfg, self.config_source = _resolve_config(config_path)
        if cfg is None:
            cfg = _CONFIG_DEFAULTS  # dict; .get() below works identically on both

        # INITIALIZATION is the ENTRY state: the run starts inside the init window, and nothing
        # downstream may treat the robot as navigating until this FSM says WP_TRACKING.
        self.FSM_STATE = FSMState.INITIALIZATION
        self._interaction_mode_start_time = None
        self._radius_obs_affected_in_travmap = cfg.get('radius_obs_affected_in_travmap', 0.2)
        self._obstacle_interaction_distance_threshold = cfg.get('obstacle_interaction_distance_threshold', 0.5)
        self._interaction_time_limit = cfg.get('interaction_time_limit', 10.0)

        # ── Initialization schedule ────────────────────────────────────────────────────────
        # Three ABSOLUTE sim-time boundaries, resolved once here and never recomputed: the phase
        # is a pure function of /clock, so an FSM that restarts mid-run lands in the phase the
        # simulation is actually in rather than replaying the window.
        self._initialization_time = float(cfg.get('initialization_time', 17.0))  # window end (s)
        self._cloud_settle_time = float(cfg.get('cloud_settle_time', 3.0))       # SETTLING end (s)
        self._init_hold_time = float(cfg.get('init_hold_time', 1.0))             # trailing hold (s)
        # ROTATE end = window end minus the trailing hold. Ordering is enforced rather than
        # assumed: a config where the holds overlap would otherwise produce a negative-length
        # phase and a state string no consumer expects.
        self._init_rotate_end = self._initialization_time - self._init_hold_time
        if not (0.0 <= self._cloud_settle_time <= self._init_rotate_end <= self._initialization_time):
            warnings.warn(
                f"[NavigationFSM] Init schedule is not ordered: cloud_settle_time="
                f"{self._cloud_settle_time:.2f}s, initialization_time - init_hold_time="
                f"{self._init_rotate_end:.2f}s, initialization_time="
                f"{self._initialization_time:.2f}s. Clamping so the phases stay in order; the "
                "ROTATE phase may end up empty.",
                RuntimeWarning, stacklevel=2,
            )
            self._cloud_settle_time = max(0.0, min(self._cloud_settle_time, self._initialization_time))
            self._init_rotate_end = max(self._cloud_settle_time,
                                        min(self._init_rotate_end, self._initialization_time))

        # Fit check (warn only): a full sweep rotates 4*amplitude, so at the commanded rate — the
        # best case, the policy tracks a yaw command with lag — it needs this much of the ROTATE
        # window. If it does not fit, the sweep is truncated by the HOLD phase every run, which
        # leaves the robot off its start heading; say so here rather than let it look like a
        # planner bug.
        _yaw_rate = float(cfg.get('init_yaw_rate', 1.0))
        _amp = math.radians(float(cfg.get('init_yaw_amplitude_deg', 90.0)))
        _sweep_time = 4.0 * _amp / _yaw_rate if _yaw_rate > 0.0 else float('inf')
        _rotate_window = self._init_rotate_end - self._cloud_settle_time
        if _sweep_time > _rotate_window:
            warnings.warn(
                f"[NavigationFSM] ROTATE window is {_rotate_window:.2f}s "
                f"(cloud_settle_time={self._cloud_settle_time:.2f}s -> "
                f"{self._init_rotate_end:.2f}s) but a full "
                f"+/-{math.degrees(_amp):.0f}deg sweep needs {_sweep_time:.2f}s at "
                f"init_yaw_rate={_yaw_rate:.2f} rad/s EVEN WITH PERFECT TRACKING. The local "
                "planner's sweep will be cut off by INITIALIZATION_HOLD. Raise "
                "initialization_time, or lower init_yaw_amplitude_deg / cloud_settle_time.",
                RuntimeWarning, stacklevel=2,
            )

        # Current sub-phase of INITIALIZATION; published as INITIALIZATION_<phase>.
        self._init_phase = INIT_PHASES[0]
        self._init_clock_warned = False    # one-shot: /clock still reads 0 (see _in_INITIALIZATION)

        # ONE-SHOT vote latches, replacing the former continuous-True streak windows
        # (failure_flag_window_duration / success_flag_window_duration, both removed): the FIRST
        # True message on a topic decides that vote for the rest of the episode. A later False
        # does NOT clear the latch, so a vote can no longer be lost between two update() calls
        # and no publisher has to re-send it to keep a streak alive.
        # The two failure sources keep SEPARATE latches because they are different verdicts —
        # IsaacLab's task_failure (robot stability) and the global planner's INTERACTION progress
        # (stall) monitor on /interaction_failure_progress — and neither one's False may cancel
        # the other. They are OR-ed in _in_INTERACTION.
        # All three are cleared on BOTH INTERACTION edges (_on_enter/_on_exit_INTERACTION), so a
        # vote that arrived while the FSM was elsewhere can never decide the next episode.
        self._failure_flag = False           # /interaction_failure          (task_failure, stability)
        self._progress_failure_flag = False  # /interaction_failure_progress (stall monitor)
        self._success_flag = False           # /interaction_success          (bypass probe)

        self._last_time = None  # last current_time seen by update()

        self._failure_recovered_flag: bool = False  # set by failure_recovered_callback; triggers FAILURE_RECOVERY → WP_TRACKING

        # Injected each loop by FSMNode from ROS topic data
        self.robot_x = None
        self.robot_y = None
        self.target_obstacle_x = None
        self.target_obstacle_y = None
        self.target_obstacle_radius = None
        self.last_exit_success = None  # set by _on_exit_INTERACTION, read by FSMNode


    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, current_time):
        """Evaluate and execute state transitions. Call every loop iteration."""
        # Backwards-clock guard: sim time only moves forward within a run, so a jump back means
        # the simulator restarted while this FSM kept running. Every init boundary and the
        # INTERACTION timeout are absolute sim times, so from here on they describe a run that no
        # longer exists. Reported, not repaired: a correct recovery is a node restart.
        if self._last_time is not None and current_time < self._last_time - 1.0:
            print(f"[NavigationFSM] CLOCK WENT BACKWARDS: {self._last_time:.2f}s -> "
                  f"{current_time:.2f}s. The simulator most likely restarted under this FSM, so "
                  f"state={self.FSM_STATE.name} and every time-based transition still refer to "
                  "the previous run. Restart the FSM node.")
        self._last_time = current_time
        if self.FSM_STATE == FSMState.INITIALIZATION:
            self._in_INITIALIZATION(current_time)
        elif self.FSM_STATE == FSMState.WP_TRACKING:
            self._in_WP_TRACKING(current_time)
        elif self.FSM_STATE == FSMState.INTERACTION:
            self._in_INTERACTION(current_time)
        elif self.FSM_STATE == FSMState.FAILURE_RECOVERY:
            self._in_FAILURE_RECOVERY(current_time)

    def published_state_name(self):
        """Return the string the owning node publishes on /fsm_state for the current state.

        INITIALIZATION carries its sub-phase (INITIALIZATION_SETTLING / _ROTATE / _HOLD) because
        the local planner issues a different command in each; every other state publishes its
        plain name. Consumers that only need the base state collapse on the "INITIALIZATION"
        prefix, exactly as they already do for the WP_TRACKING_SUCCESS transition tag.
        """
        if self.FSM_STATE == FSMState.INITIALIZATION:
            return f"INITIALIZATION_{self._init_phase}"
        return self.FSM_STATE.name

    def interaction_failure_callback(self, msg):
        """ROS callback: failure vote from the local MPC planner (std_msgs/Bool).

        ONE-SHOT: the first True latches and is never cleared by a later False — only the
        INTERACTION edges clear it. A False therefore carries no information at all here."""
        if msg.data:
            self._failure_flag = True

    def interaction_progress_failure_callback(self, msg):
        """ROS callback: failure vote from the global planner's INTERACTION progress monitor
        (std_msgs/Bool) — the robot stopped making progress on the push. Latched exactly like
        interaction_failure_callback above but SEPARATELY, so neither source's False can
        cancel the other's True; _in_INTERACTION fails the push when EITHER latch is set."""
        if msg.data:
            self._progress_failure_flag = True

    def failure_recovered_callback(self, msg):
        """ROS callback: recovery-complete signal from local planner (std_msgs/Bool).
        Sets flag to trigger FAILURE_RECOVERY -> WP_TRACKING on the next update()."""
        if msg.data:
            self._failure_recovered_flag = True

    def interaction_success_callback(self, msg):
        """ROS callback: success vote from the global planner's bypass probe (std_msgs/Bool).

        ONE-SHOT like the two failure latches. No time base is needed any more, so the former
        `_last_time is None` drop (which existed only because the streak had to be stamped) is
        gone: a True that arrives before the first update() is now kept, not discarded."""
        if msg.data:
            self._success_flag = True

    # ------------------------------------------------------------------
    # Per-state guards
    # ------------------------------------------------------------------

    def _in_INITIALIZATION(self, current_time):
        """Guard: walk the init phase schedule, then hand over to WP_TRACKING.

        TIMING ONLY. The phase is a pure function of sim time against the three boundaries
        resolved in __init__; the local planner reads the published phase and issues the command
        (zero in SETTLING and HOLD, an in-place yaw sweep in ROTATE).
        """
        # /clock has not started yet: with use_sim_time the node clock reads exactly 0 until the
        # simulator publishes, so the FSM correctly sits in the first phase. Name it once —
        # from the outside a waiting FSM and a wedged one look identical.
        if current_time <= 0.0:
            if not self._init_clock_warned:
                self._init_clock_warned = True
                print(f"[NavigationFSM] sim clock still reads 0: waiting for /clock. Holding "
                      f"INITIALIZATION_{INIT_PHASES[0]} (robot stands still) until the simulator "
                      "publishes. This is normal before IsaacLab starts.")
            return

        # Ordered latest-first so each tick lands in exactly one phase.
        if current_time >= self._initialization_time:
            self._on_exit_INITIALIZATION(current_time)
        elif current_time >= self._init_rotate_end:
            self._set_init_phase('HOLD', current_time)
        elif current_time >= self._cloud_settle_time:
            self._set_init_phase('ROTATE', current_time)
        # else: stay in SETTLING, the entry phase

    def _set_init_phase(self, phase, current_time):
        """Advance the init sub-phase, reporting the boundary exactly once.

        Called every tick, so the equality check is what keeps the 50 Hz loop from printing the
        same boundary forever.
        """
        if self._init_phase == phase:
            return
        print(f"[NavigationFSM] INITIALIZATION {self._init_phase} -> {phase} at "
              f"t={current_time:.2f}s")
        self._init_phase = phase

    def _in_WP_TRACKING(self, current_time):
        """Guard: WP_TRACKING -> INTERACTION when robot reaches target_obstacle."""
        if (self.target_obstacle_x is None or self.target_obstacle_y is None
                or self.target_obstacle_radius is None
                or self.robot_x is None or self.robot_y is None):
            return

        dist = math.hypot(self.target_obstacle_x - self.robot_x,
                          self.target_obstacle_y - self.robot_y)
        # target_obstacle_radius is r_a; enter outside the planning affected zone with
        # the configured additional margin.
        interaction_distance = (self.target_obstacle_radius
                                + self._radius_obs_affected_in_travmap
                                + self._obstacle_interaction_distance_threshold)
        if dist <= interaction_distance:
            self._on_enter_INTERACTION(current_time)

    def _in_INTERACTION(self, current_time):
        """Guard: INTERACTION -> WP_TRACKING on success or failure.

        Deliberately SILENT: this runs on every FSMNode loop iteration, so anything printed here
        floods the console for the whole episode. The episode is reported exactly twice — once on
        entry and once on exit, with the reason — and nowhere else.
        """
        elapsed = current_time - self._interaction_mode_start_time

        # Failure: timeout OR either failure latch set. The two sources are OR-ed and latched
        # independently: IsaacLab's task_failure (stability) and the global planner's progress
        # monitor answer different questions, and one True from either is enough to abandon the
        # push. Resolved to a REASON string (None = no failure) so the single exit line can say
        # which of the three actually fired; the order below is the reporting priority only.
        if elapsed > self._interaction_time_limit:
            failure_reason = f"timeout ({elapsed:.1f}s > {self._interaction_time_limit:.1f}s)"
        elif self._failure_flag:
            failure_reason = "task failure (/interaction_failure)"
        elif self._progress_failure_flag:
            failure_reason = "no push progress (/interaction_failure_progress)"
        else:
            failure_reason = None

        # Success: one True on /interaction_success ends the push — TBA: real-time object pos no longer overlaps with path
        if self._success_flag:
            self._on_exit_INTERACTION(success=True, reason="bypass proved (/interaction_success)")
        elif failure_reason is not None:
            self._on_exit_INTERACTION(success=False, reason=failure_reason)
        # Success is prioritized: if both are true, Success takes precedence.

    # ------------------------------------------------------------------
    # Entry / exit actions
    # ------------------------------------------------------------------

    def _on_exit_INITIALIZATION(self, current_time):
        """Exit action: the init window is over, navigation starts."""
        self.FSM_STATE = FSMState.WP_TRACKING
        # A large overshoot means this FSM joined a run already past initialization_time (node
        # restarted, or started after the simulator). The window is NOT replayed in that case —
        # the schedule is absolute sim time — so the robot never swept. Say which happened.
        late = current_time - self._initialization_time
        late_note = (f" — entered {late:.2f}s LATE, so the init sweep never ran under this FSM"
                     if late > 1.0 else "")
        print(f"[NavigationFSM] INITIALIZATION -> WP_TRACKING | schedule complete at "
              f"t={current_time:.2f}s (initialization_time="
              f"{self._initialization_time:.2f}s){late_note}")

    def _on_enter_INTERACTION(self, current_time):
        """Entry action: transition into INTERACTION state."""
        self.FSM_STATE = FSMState.INTERACTION
        self._interaction_mode_start_time = current_time
        # Clear every vote latch on the ENTRY edge too: a True that arrived while the FSM was in
        # WP_TRACKING (the planner keeps publishing) would otherwise decide this episode before
        # the robot has touched anything.
        self._failure_flag = False
        self._progress_failure_flag = False
        self._success_flag = False
        print(f"[NavigationFSM] WP_TRACKING -> INTERACTION | target_obstacle=({self.target_obstacle_x:.2f}, {self.target_obstacle_y:.2f})")

    def _on_exit_INTERACTION(self, success, reason=""):
        """Exit action: transition out of INTERACTION.
        Success → WP_TRACKING; failure → FAILURE_RECOVERY.

        reason : str   why the episode ended, from _in_INTERACTION; printed on the ONE line that
                       reports this episode's outcome (the per-tick status display is gone)."""
        # Clear interaction bookkeeping regardless of outcome
        self._interaction_mode_start_time = None
        self._failure_flag = False
        self._progress_failure_flag = False
        self._success_flag = False
        # Clear target obstacle so the FSM does not immediately re-enter INTERACTION
        self.target_obstacle_x = None
        self.target_obstacle_y = None
        self.target_obstacle_radius = None

        # The leading "\n" these two prints used to carry is gone with the per-tick "\r" status
        # line they had to break out of; nothing is left mid-line to terminate.
        if success:
            self.FSM_STATE = FSMState.WP_TRACKING
            self.last_exit_success = True
            print(f"[NavigationFSM] INTERACTION -> WP_TRACKING | success=True | {reason}")
        else:
            self.last_exit_success = None  # ensure no stale True from a prior success cycle
            self._on_enter_FAILURE_RECOVERY(reason)

    def _in_FAILURE_RECOVERY(self, current_time):
        """Guard: FAILURE_RECOVERY -> WP_TRACKING when local planner signals recovery complete."""
        if self._failure_recovered_flag:
            self._on_exit_FAILURE_RECOVERY()

    def _on_enter_FAILURE_RECOVERY(self, reason=""):
        """Entry action: transition into FAILURE_RECOVERY state.

        reason : str   the failure that ended the INTERACTION, forwarded from
                       _on_exit_INTERACTION so this line alone explains the episode."""
        self.FSM_STATE = FSMState.FAILURE_RECOVERY
        self._failure_recovered_flag = False
        print(f"[NavigationFSM] INTERACTION -> FAILURE_RECOVERY | {reason}")

    def _on_exit_FAILURE_RECOVERY(self):
        """Exit action: transition out of FAILURE_RECOVERY back to WP_TRACKING."""
        self.FSM_STATE = FSMState.WP_TRACKING
        self._failure_recovered_flag = False
        print("[NavigationFSM] FAILURE_RECOVERY -> WP_TRACKING")
