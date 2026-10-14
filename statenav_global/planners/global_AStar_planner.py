"""
Heading-aware sparse A* over one captured GraphSnapshot (Phase C).

See docs/shared_memory_graph_plan_3_phase_C_onward.md for the full contract.

The planner is graph-native: it acquires one process-local `GraphSnapshot`
per planning invocation (a detached seqlock copy of the writer's shared-memory
graph) and runs the complete search against that frozen state. Topology comes
only from `snapshot.neighbors(slot)`; coordinates from `snapshot.node_x/node_y`;
directional command limits from the `cmd_v` / `cmd_w` node payloads. Graph edges
define every allowed geometric transition; there is no radius query inside A*.

Search-state semantics are exact `(x, y, theta)`:
  - `(x, y)`  is the physical position of one active graph node (via its slot),
  - `theta`   is the robot's EXACT wrapped arrival heading at that node.
Three heading quantities are kept distinct (design section 8):
  - `theta_current` : exact heading stored in the current SearchState,
  - `theta_move`    : exact geometric direction of a candidate outgoing edge,
  - `layer`         : discrete payload-layer index derived from `theta_move`
                      only, used solely to read directional `cmd_v/cmd_w`.
Only `theta_move` is quantized (to select a payload layer); turn geometry and
state identity always use exact headings so payload discretization never
corrupts the turn cost.

This module deliberately keeps its top-level imports light (numpy + stdlib +
BasePlanner). `shared_graph`, the shared `GRAPH_LAYOUT`, and `plotting` are
imported lazily inside methods, because `statenav_global.world_model.__init__`
pulls the heavy torch/ROS world-model stack and `planners` is imported before
`world_model` in the package `__init__` (a top-level import would be circular).
"""

import heapq
import importlib.util
import itertools
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import BasePlanner

# Import reusable utility helpers without making the standalone planner import
# depend on the ROS/torch package import chain. The file fallback supports the
# test harness that loads this module under a synthetic parent package.
try:
    from statenav_global.utility.heuristic import (
        build_heuristic,
        normalize_heuristic_name,
    )
    from statenav_global.utility.utils import (
        transition_cost,
        wrap_to_pi,
    )
except ModuleNotFoundError:
    # Loading the parent utility package can fail outside the development
    # container because its legacy exports import torch. These helpers are
    # dependency-free, so load only utility/utils.py in that environment.
    _utils_path = Path(__file__).resolve().parents[1] / "utility" / "utils.py"
    _utils_spec = importlib.util.spec_from_file_location("_global_astar_utils", _utils_path)
    if _utils_spec is None or _utils_spec.loader is None:
        raise ImportError(f"Unable to load utility helpers from {_utils_path}")
    _utils = importlib.util.module_from_spec(_utils_spec)
    _utils_spec.loader.exec_module(_utils)
    wrap_to_pi = _utils.wrap_to_pi
    transition_cost = _utils.transition_cost
    _heuristic_path = Path(__file__).resolve().parents[1] / "utility" / "heuristic.py"
    _heuristic_spec = importlib.util.spec_from_file_location(
        "_global_astar_heuristic", _heuristic_path)
    if _heuristic_spec is None or _heuristic_spec.loader is None:
        raise ImportError(f"Unable to load heuristic helpers from {_heuristic_path}")
    _heuristic = importlib.util.module_from_spec(_heuristic_spec)
    sys.modules[_heuristic_spec.name] = _heuristic
    _heuristic_spec.loader.exec_module(_heuristic)
    build_heuristic = _heuristic.build_heuristic
    normalize_heuristic_name = _heuristic.normalize_heuristic_name


# ---------------------------------------------------------------------------
# Immutable, hashable search state — used as a dict/set key (design 5.1).
# `theta` is the exact wrapped arrival heading, NOT a payload-layer index.
# Immutability is a search-state requirement only; it does not imply the
# GraphSnapshot itself is immutable.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SearchState:
    node_slot: int   # snapshot-local handle for (x, y); NOT a cross-cycle id
    theta: float     # exact wrapped arrival heading in [-pi, pi)


class GlobalAStar(BasePlanner):
    """
    Heading-aware sparse A* planner on the shared-memory graph (Phase C).

    Owns one `SharedGraphReader` and reuses one `GraphSnapshot` buffer across
    invocations. Public surface is what the planning node touches
    (`replan(...)`, `path`, `plan_cost`, `plot_map`).
    """

    # Constructor Method
    def __init__(
        self,
        task_extent: List[float],
        rng: np.random.RandomState,

        x_start: tuple,
        x_goal: tuple,
        heading_start: float,

        *,
        # Heading-layer geometry — MUST match the writer's authoritative values
        # (theta_min / theta_resolution / theta_layers). Passed in by the
        # selector from the world-model instance so it is never duplicated as a
        # second config source; consistency against the snapshot payloads is
        # re-checked at plan time (payload column count == theta_layers).
        theta_min: float,
        theta_resolution: float,
        theta_layers: int,

        # A*-specific scalars (planning_config.yaml: graph_astar_*).
        snapshot_max_retries: int = 50,
        epsilon: float = 1.0e-12,
        heuristic: str = "reverse_dijkstra",
        heuristic_weight: float = 1.5,
        alt_landmark_count: int = 4,

        # planning_config.yaml: speed_discount — fraction of stored travmap
        # v/w assumed achievable; inflates travel-time by 1/it.
        speed_discount: float = 1.0,

        # Arc-radius approximation of the curvature-limited edge model
        # (planning_config.yaml: arc_radius_factor). Calibration pair with
        # speed_discount; the default keeps the value the model shipped with.
        arc_radius_factor: float = 1.5,

        attach_timeout: float = 10.0,
        graph_layout=None,
        graph_reader=None,
    ) -> None:

        # Initialize BasePlanner
        super().__init__(task_extent, rng)

        # --- start / goal / heading (world coordinates) -------------------
        self.x_start = (float(x_start[0]), float(x_start[1]))
        self.x_goal = (float(x_goal[0]), float(x_goal[1]))
        self.heading_start = float(heading_start)

        # --- heading-layer geometry + config validation (design 7.3) ------
        self._validate_config(theta_min, theta_resolution, theta_layers,
                              epsilon,
                              snapshot_max_retries, speed_discount,
                              arc_radius_factor,
                              heuristic, heuristic_weight, alt_landmark_count)
        self.theta_min = float(theta_min)
        self.theta_resolution = float(theta_resolution)
        self.theta_layers = int(theta_layers)
        self.snapshot_max_retries = int(snapshot_max_retries)
        self.epsilon = float(epsilon)
        self.speed_discount = float(speed_discount)
        self.arc_radius_factor = float(arc_radius_factor)
        self.heuristic_name = normalize_heuristic_name(heuristic)
        self.heuristic_weight = float(heuristic_weight)
        self.alt_landmark_count = int(alt_landmark_count)
        self.attach_timeout = float(attach_timeout)

        # --- shared-graph ownership ---------------------------------------
        # graph_layout defaults to the shared GRAPH_LAYOUT (single source of
        # truth for writer + reader). Imported lazily to keep this module free
        # of the heavy world-model import chain at package-load time.
        if graph_layout is None:
            from statenav_global.core.graph_layout import GRAPH_LAYOUT
            graph_layout = GRAPH_LAYOUT
        self.graph_layout = graph_layout
        # One SharedGraphReader (created lazily on first plan unless injected)
        # and one reusable GraphSnapshot buffer (allocated on first snapshot).
        self.graph_reader = graph_reader
        self.graph_snapshot = None

        # --- public planner state (the fields the node reads) ------------
        # public path: list of [x, y]; graph-node coords when path smoothing is
        # disabled, otherwise the smoothed polyline (endpoints still exact
        # graph-node coords). state_path / node_path always stay raw.
        self.path = []
        self.plan_cost = 0.0      # final search cost (estimated travel time, s)
        self.is_goal_reached = False

        # --- diagnostics (design 5.3 / 19) --------------------------------
        self.total_plan_time = 0.0     # wall time of the whole plan() call (s)
        self.snapshot_copy_time = 0.0  # seqlock snapshot acquisition time (s)
        self.search_time = 0.0         # A* loop time (s)
        self.expanded_count = 0        # number of expanded (closed) states
        self.start_slot = -1           # resolved start slot (this invocation)
        self.goal_slot = -1            # resolved goal slot (this invocation)
        self.no_path_reason = None     # human-readable ordinary-failure reason
        # Diagnostic-only path representations (valid for this invocation only;
        # snapshot-local slots are never persisted as cross-cycle identifiers).
        self.state_path = []           # list of (x, y, theta) tuples
        self.node_path = []            # list of snapshot-local node slots

        # --- world-model attached post-init by the node -------------------
        self.global_map = None

        # Plotting object. Imported lazily; non-essential for A*, so a
        # failure to build it must not break planning.
        try:
            from statenav_global.utility import plotting
            self.plotting = plotting.Plotting(self.x_start, self.x_goal,
                                              task_extent=task_extent)
        except Exception as exc:   # pragma: no cover - matplotlib/display issue
            print(f"[GlobalAStar] plotting init skipped: {exc}")
            self.plotting = None

    # ------------------------------------------------------------------
    # Configuration validation (design 7.3)
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_config(theta_min, theta_resolution, theta_layers,
                         epsilon,
                         snapshot_max_retries, speed_discount,
                         arc_radius_factor,
                         heuristic, heuristic_weight, alt_landmark_count):
        """Fail loud on invalid configuration before any planning can run."""
        if not math.isfinite(float(theta_min)):
            raise ValueError(f"theta_min must be finite, got {theta_min!r}")
        tr = float(theta_resolution)
        if not math.isfinite(tr) or tr <= 0.0:
            raise ValueError(f"theta_resolution must be finite and > 0, got {theta_resolution!r}")
        if isinstance(theta_layers, bool) or not isinstance(theta_layers, (int, np.integer)) \
                or int(theta_layers) < 1:
            raise ValueError(f"theta_layers must be a positive integer, got {theta_layers!r}")
        # The layers must cover exactly one full revolution (design 7.3).
        full_rev = int(theta_layers) * tr
        if abs(full_rev - 2.0 * math.pi) > 1e-6:
            raise ValueError(
                f"theta_layers * theta_resolution = {full_rev} must cover one full "
                f"revolution (2*pi = {2.0 * math.pi}); layers={theta_layers}, "
                f"resolution={theta_resolution}")
        ep = float(epsilon)
        if not math.isfinite(ep) or ep <= 0.0:
            raise ValueError(f"epsilon must be finite and > 0, got {epsilon!r}")
        if isinstance(snapshot_max_retries, bool) \
                or not isinstance(snapshot_max_retries, (int, np.integer)) \
                or int(snapshot_max_retries) < 1:
            raise ValueError(
                f"snapshot_max_retries must be a positive integer, got {snapshot_max_retries!r}")
        # speed_discount must be a strictly-positive fraction (dividing by it must
        # be well-defined); values in (0, 1] inflate cost, 1.0 is a no-op.
        sd = float(speed_discount)
        if not math.isfinite(sd) or sd <= 0.0:
            raise ValueError(f"speed_discount must be finite and > 0, got {speed_discount!r}")
        # arc_radius_factor scales the arc radius R = factor * L / dtheta, so it must be
        # strictly positive; 0 or negative would make every turning edge cost inf or
        # negative-speed nonsense.
        arf = float(arc_radius_factor)
        if not math.isfinite(arf) or arf <= 0.0:
            raise ValueError(
                f"arc_radius_factor must be finite and > 0, got {arc_radius_factor!r}")
        normalize_heuristic_name(heuristic)
        hw = float(heuristic_weight)
        if not math.isfinite(hw) or hw < 1.0:
            raise ValueError(
                f"heuristic_weight must be finite and >= 1, got {heuristic_weight!r}")
        if isinstance(alt_landmark_count, bool) \
                or not isinstance(alt_landmark_count, (int, np.integer)) \
                or int(alt_landmark_count) < 1:
            raise ValueError(
                f"alt_landmark_count must be a positive integer, got "
                f"{alt_landmark_count!r}")

    # ------------------------------------------------------------------
    # Reader / snapshot lifecycle (design 6)
    # ------------------------------------------------------------------

    def _ensure_reader(self):
        """
        Create the single SharedGraphReader lazily on first use (the writer's
        graph SHM may not exist at planner-construction time). Never attaches
        shared-memory blocks inside the search loop.
        """
        if self.graph_reader is not None:
            return
        # Lazy import (see module docstring): avoids the heavy world-model
        # import chain at package-load time.
        from statenav_global.core.shared_graph import (
            SharedGraphReader, SharedGraphError,
        )
        try:
            self.graph_reader = SharedGraphReader(self.graph_layout,
                                                  attach_timeout=self.attach_timeout)
        except SharedGraphError as exc:
            # Missing / unpublished shared graph is a programming/config error,
            # not an ordinary no-path result (design 19).
            raise RuntimeError(
                f"GlobalAStar: could not attach to the shared graph — is the "
                f"world-model writer running (it builds the graph at map init)? "
                f"({exc})") from exc

    def close(self):
        """Release the reader's SHM handles. Never unlinks writer-owned blocks."""
        if self.graph_reader is not None:
            try:
                self.graph_reader.close()
            except Exception:
                pass
            self.graph_reader = None

    # ------------------------------------------------------------------
    # Snapshot-array validation (design 7.2)
    # ------------------------------------------------------------------

    def _validate_snapshot_arrays(self, snapshot, cmd_v: np.ndarray, cmd_w: np.ndarray):
        """
        Validate the captured snapshot's core arrays and payload shapes before
        entering the A* loop. Raises on malformed rank/shape (programming/schema
        error); never silently truncates or reshapes.
        """
        node_active = np.asarray(snapshot.node_active)
        node_x = np.asarray(snapshot.node_x)
        node_y = np.asarray(snapshot.node_y)
        edge_length = np.asarray(snapshot.edge_length)
        # core arrays must be 1-D
        for name, arr in (("node_active", node_active), ("node_x", node_x),
                          ("node_y", node_y), ("edge_length", edge_length)):
            if arr.ndim != 1:
                raise ValueError(f"snapshot.{name} must be 1-D, got shape {arr.shape}")
        node_rows = len(node_active)
        # node_x / node_y must index every node row
        if len(node_x) < node_rows or len(node_y) < node_rows:
            raise ValueError(
                f"snapshot node_x/node_y rows ({len(node_x)}/{len(node_y)}) < "
                f"node_active rows ({node_rows})")
        # cmd_v / cmd_w must be rank 2, exactly theta_layers columns, enough rows
        for name, arr in (("cmd_v", cmd_v), ("cmd_w", cmd_w)):
            if arr.ndim != 2:
                raise ValueError(f"payload {name!r} must be rank 2, got shape {arr.shape}")
            if arr.shape[1] != self.theta_layers:
                raise ValueError(
                    f"payload {name!r} has {arr.shape[1]} heading layers, expected "
                    f"theta_layers={self.theta_layers} (writer/planner layer mismatch)")
            if arr.shape[0] < node_rows:
                raise ValueError(
                    f"payload {name!r} has {arr.shape[0]} rows < node_active rows "
                    f"({node_rows})")

    # ------------------------------------------------------------------
    # Admissible stored-speed heuristic (design 14)
    # ------------------------------------------------------------------

    def _compute_speed_bound(self, snapshot, cmd_v: np.ndarray) -> float:
        """
        `v_upper` = maximum finite, strictly positive stored `cmd_v` over active
        snapshot nodes. Entries still carrying the writer `Initialguess` value
        are valid stored values and included; inactive nodes and non-finite /
        non-positive values are excluded. Absence of any positive finite stored
        speed fails loud (design 19).
        """
        active = np.asarray(snapshot.node_active, dtype=bool)   # (n_max,)
        values = cmd_v[: len(active)]                           # (n_max, L) aligned to active
        # broadcast active over the L heading columns; keep only usable speeds
        valid = active[:, None] & np.isfinite(values) & (values > 0.0)
        if not np.any(valid):
            raise RuntimeError(
                "GlobalAStar heuristic: no positive finite cmd_v exists on any "
                "active snapshot node; cannot form an admissible speed bound")
        return float(np.max(values[valid]))

    def _heading_to_layer(self, theta_move: float) -> int:
        """Map an exact movement heading with the writer's layer geometry."""
        if self.global_map is None or not callable(
                getattr(self.global_map, "get_theta_layer", None)):
            raise RuntimeError(
                "GlobalAStar: global_map with get_theta_layer() must be attached "
                "before planning")
        return int(self.global_map.get_theta_layer(theta_move))

    # ------------------------------------------------------------------
    # Neighbor expansion (design 11) — exact-heading successor generation
    # ------------------------------------------------------------------

    def _successors(self, snapshot, cmd_v: np.ndarray, cmd_w: np.ndarray,
                    current: SearchState):
        """
        Yield (successor_state, edge_cost) for each explicit graph neighbor of
        `current` whose transition cost is finite. Follows graph edges only (no
        radius query). The successor heading is `theta_move` because after
        traversing the edge the robot is assumed aligned with it.
        """
        slot_i = current.node_slot
        if self.global_map is None or not callable(
                getattr(self.global_map, "get_theta_layer", None)):
            raise RuntimeError(
                "GlobalAStar: global_map with get_theta_layer() must be attached "
                "before planning")
        x_i = float(snapshot.node_x[slot_i])
        y_i = float(snapshot.node_y[slot_i])
        if not (math.isfinite(x_i) and math.isfinite(y_i)):
            raise ValueError(
                f"GlobalAStar: node slot {slot_i} has non-finite position "
                f"({x_i}, {y_i})")

        for target_raw, edge_raw in snapshot.neighbors(slot_i):
            target = int(target_raw)
            edge_index = int(edge_raw)

            x_j = float(snapshot.node_x[target])
            y_j = float(snapshot.node_y[target])
            if not (math.isfinite(x_j) and math.isfinite(y_j)):
                raise ValueError(
                    f"GlobalAStar: node slot {target} has non-finite position "
                    f"({x_j}, {y_j})")

            # exact outgoing-edge direction; only THIS is quantized to a layer
            theta_move = wrap_to_pi(math.atan2(y_j - y_i, x_j - x_i))
            layer = self._heading_to_layer(theta_move)

            # stored directional commands (already hold Initialguess or estimate)
            v_i = float(cmd_v[slot_i, layer])
            v_j = float(cmd_v[target, layer])
            w_i = float(cmd_w[slot_i, layer])

            edge_cost = transition_cost(
                edge_length=float(snapshot.edge_length[edge_index]),
                theta_current=current.theta,
                theta_move=theta_move,
                v_i=v_i,
                v_j=v_j,
                w_i=w_i,
                speed_discount=self.speed_discount,
                arc_radius_factor=self.arc_radius_factor,
            )
            if math.isfinite(edge_cost):
                yield SearchState(target, theta_move), edge_cost

    # ------------------------------------------------------------------
    # Sparse heap-based A* core (design 15)
    # ------------------------------------------------------------------

    def _build_search_heuristic(self, snapshot, cmd_v: np.ndarray,
                                cmd_w: np.ndarray, goal_slot: int,
                                v_upper: float):
        """
        Build the configured heuristic for ONE (snapshot, goal_slot) pair.

        Factored out of _search so a caller running SEVERAL searches toward the
        SAME goal_slot on the same snapshot can build it once and pass it in:
        graph-aware strategies such as reverse_dijkstra precompute a full
        backward Dijkstra over the whole graph here, and that table is
        identical for every such search.
        """
        return build_heuristic(
            name=self.heuristic_name,
            snapshot=snapshot,
            cmd_v=cmd_v,
            cmd_w=cmd_w,
            goal_slot=goal_slot,
            v_upper=v_upper,
            speed_discount=self.speed_discount,
            arc_radius_factor=self.arc_radius_factor,
            heading_to_layer=self._heading_to_layer,
            transition_cost=transition_cost,
            wrap_to_pi=wrap_to_pi,
            weight=self.heuristic_weight,
            alt_landmark_count=self.alt_landmark_count,
        )

    def _search(self, snapshot, cmd_v: np.ndarray, cmd_w: np.ndarray,
                start_slot: int, goal_slot: int, start_heading: float,
                v_upper: float, heuristic=None):
        """
        Run sparse A* from `start_slot` (facing `start_heading`) to any arrival
        heading at `goal_slot`. Returns
        (success, goal_state, g_score, came_from, expanded_count).

        `heuristic` is an already-built callable for THIS goal_slot and
        snapshot, or None (default, unchanged behavior) to build the configured
        one here.

        Sparse containers because the graph is not assumed rectangular. Heap
        entries are (f_score, g_at_push, insertion_counter, state); the counter
        prevents Python from ever ordering SearchState objects on tied priority.
        """
        eps = self.epsilon
        start_state = SearchState(start_slot, wrap_to_pi(start_heading))

        # start == goal: succeed immediately with zero cost (design 15).
        if start_slot == goal_slot:
            return True, start_state, {start_state: 0.0}, {}, 0

        # Prepare the configured strategy once for this frozen snapshot and
        # target. Graph-aware heuristics may precompute reverse shortest paths.
        # A caller that already built it for this goal_slot passes it in and
        # skips that precomputation (see _build_search_heuristic).
        if heuristic is None:
            heuristic = self._build_search_heuristic(
                snapshot, cmd_v, cmd_w, goal_slot, v_upper)

        g_score: Dict[SearchState, float] = {start_state: 0.0}
        came_from: Dict[SearchState, SearchState] = {}
        closed = set()
        open_heap: List[Tuple[float, float, int, SearchState]] = []
        counter = itertools.count()          # unique insertion order tie-break
        expanded_count = 0

        heapq.heappush(
            open_heap,
            (heuristic(start_state), 0.0, next(counter), start_state),
        )

        while open_heap:
            _, pushed_g, _, current = heapq.heappop(open_heap)
            best_g = g_score.get(current, math.inf)

            # discard superseded / stale heap entries (centralized epsilon)
            if pushed_g > best_g + eps:
                continue
            if current in closed:
                continue

            closed.add(current)
            expanded_count += 1

            # goal test on the node slot only (any arrival heading accepted)
            if current.node_slot == goal_slot:
                return True, current, g_score, came_from, expanded_count

            for successor, edge_cost in self._successors(snapshot, cmd_v, cmd_w, current):
                if successor in closed:
                    continue
                candidate = best_g + edge_cost
                old = g_score.get(successor, math.inf)
                if candidate >= old - eps:
                    continue
                came_from[successor] = current
                g_score[successor] = candidate
                priority = candidate + heuristic(successor)
                heapq.heappush(
                    open_heap,
                    (priority, candidate, next(counter), successor),
                )

        # open list exhausted without reaching the goal: ordinary no-path
        return False, None, g_score, came_from, expanded_count

    # ------------------------------------------------------------------
    # Path reconstruction (design 16)
    # ------------------------------------------------------------------

    def _reconstruct(self, snapshot, came_from: Dict[SearchState, SearchState],
                     goal_state: SearchState):
        """
        Follow `came_from` backward from `goal_state` and build three
        representations:
          - state_path : semantic (x, y, theta) tuples read directly from the
                         snapshot node coordinates,
          - node_path  : snapshot-local slots (diagnostics only),
          - path       : public [x, y] graph-node polyline (start_slot..goal_slot).
        No raw start/goal query point is added outside the graph.
        """
        states = [goal_state]
        current = goal_state
        while current in came_from:
            current = came_from[current]
            states.append(current)
        states.reverse()

        state_path = []
        node_path = []
        path = []
        for s in states:
            x = float(snapshot.node_x[s.node_slot])
            y = float(snapshot.node_y[s.node_slot])
            state_path.append((x, y, s.theta))
            node_path.append(int(s.node_slot))
            path.append([x, y])
        return state_path, node_path, path

    # ------------------------------------------------------------------
    # Isolated start/goal recovery — re-snap to the nearest usable node
    # ------------------------------------------------------------------
    # A node is "isolated" when A* can neither leave it (nor, on the symmetric
    # graph, enter it): _successors yields nothing because every incident edge
    # has a non-finite transition_cost — all-non-finite stored cmd_v/cmd_w on
    # the node (a gray / non-traversable node) or no active neighbors at all.
    # nearest_node() returns such a slot without complaint, after which the
    # search fails with cost inf. These helpers instead re-resolve start/goal to
    # the nearest USABLE node, with a warning, so a start/goal that happens to
    # land on an isolated node no longer poisons the whole plan.

    # Heading-alignment weight for the isolated-node re-snap ranking:
    #   score = dist * (1 + K * herr/pi)
    # so a candidate directly behind the robot (herr = pi) ranks as if it were
    # (1 + K)x farther. Distance stays dominant; heading only biases among
    # comparably close candidates. Tunable; K = 0.0 == pure nearest-node.
    _ISOLATED_RESNAP_HEADING_WEIGHT = 0.5
    # Upper bound on isolation probes during the fallback scan so a pathological
    # map (most nodes isolated) cannot make one resolution O(N). Candidates are
    # probed nearest-first, so this cap is only ever reached in degenerate maps.
    _ISOLATED_RESNAP_MAX_PROBES = 256

    def _slot_is_isolated(self, snapshot, cmd_v: np.ndarray, cmd_w: np.ndarray,
                          slot: int) -> bool:
        """
        True when node `slot` yields NO admissible successor (every incident
        edge has a non-finite transition_cost, or the node has no active
        neighbors). Heading-independent: transition_cost only returns inf on
        invalid geometry / non-positive-non-finite command limits, never from
        the turn term, so the probe heading (0.0) cannot change the result.
        Delegates to self._successors. Read-only over the snapshot (invariant I7).
        """
        probe = SearchState(int(slot), 0.0)   # heading irrelevant to admissibility
        for _successor, _edge_cost in self._successors(snapshot, cmd_v, cmd_w, probe):
            return False   # first admissible successor -> not isolated
        return True

    def _resolve_nonisolated_slot(self, snapshot, cmd_v: np.ndarray,
                                  cmd_w: np.ndarray, x: float, y: float,
                                  heading, role: str) -> int:
        """
        Resolve world point (x, y) to a USABLE graph node slot.

        Fast path (unchanged behavior): the ordinary snapshot.nearest_node(x, y),
        returned as-is when it is off-graph (-1; caller keeps its existing
        not-on-graph handling) or already non-isolated (the common case — so
        healthy resolutions stay byte-for-byte identical and print nothing).

        Fallback (nearest node isolated): scan active nodes nearest-first and
        return the first NON-isolated one, warning with the re-snap distance.
        `heading` (radians) biases the ranking toward nodes in front of the
        robot when not None (used for the start); None ranks by pure distance
        (used for the goal, where the robot heading is meaningless). Returns the
        original isolated slot (with a warning) when no usable node exists, so
        the subsequent search fails loudly with the normal no-path reason.
        """
        slot = int(snapshot.nearest_node(x, y))
        if slot < 0:
            return slot   # off-graph: unchanged behavior (caller reports it)
        if not self._slot_is_isolated(snapshot, cmd_v, cmd_w, slot):
            return slot   # nearest node usable: no change, no warning

        # --- fallback: nearest node is isolated -> nearest usable node --------
        node_active = np.asarray(snapshot.node_active)                     # (n_max,) 1 = active
        n_rows = len(node_active)
        node_x = np.asarray(snapshot.node_x)[:n_rows].astype(np.float64)   # (n_max,) world x
        node_y = np.asarray(snapshot.node_y)[:n_rows].astype(np.float64)   # (n_max,) world y
        active_slots = np.flatnonzero(node_active == 1)                    # (M,) active node slots

        dx = node_x[active_slots] - float(x)                               # (M,) offset to query
        dy = node_y[active_slots] - float(y)
        dist = np.hypot(dx, dy)                                            # (M,) Euclidean distance [m]
        if heading is None:
            score = dist                                                   # goal: pure nearest
        else:
            # Bearing from the query point to each candidate vs. the robot
            # heading; penalize misaligned (behind-the-robot) candidates.
            bearing = np.arctan2(dy, dx)                                   # (M,) [-pi, pi]
            herr = np.abs(wrap_to_pi(bearing - float(heading)))            # (M,) [0, pi]
            score = dist * (1.0 + self._ISOLATED_RESNAP_HEADING_WEIGHT * (herr / np.pi))
        order = np.argsort(score, kind='stable')                          # best (nearest/aligned) first

        probes = 0                                                         # bounded isolation-probe counter
        for j in order:
            cand = int(active_slots[j])
            if cand == slot:
                continue   # the already-known isolated nearest node
            if probes >= self._ISOLATED_RESNAP_MAX_PROBES:
                break      # bound worst-case cost on degenerate graphs
            probes += 1
            if not self._slot_is_isolated(snapshot, cmd_v, cmd_w, cand):
                print(f"[GlobalAStar] {role} nearest node slot {slot} at "
                      f"({x:.3f}, {y:.3f}) is isolated (no usable graph edge); "
                      f"re-snapping to node slot {cand} at "
                      f"({node_x[cand]:.3f}, {node_y[cand]:.3f}), "
                      f"{dist[j]:.3f} m away.")
                return cand

        # No usable node found within the probe budget: keep the isolated slot
        # and warn; the caller's search then reports the normal no-path reason.
        print(f"[GlobalAStar] {role} nearest node slot {slot} at "
              f"({x:.3f}, {y:.3f}) is isolated and NO non-isolated active node "
              f"was found within {probes} probes; planning will report no path.")
        return slot

    # ------------------------------------------------------------------
    # Path-smoothing traversability gate (graph-native)
    # ------------------------------------------------------------------

    def _graph_point_is_traversable(self, snapshot, cmd_v: np.ndarray,
                                    cmd_w: np.ndarray, x: float, y: float) -> bool:
        """
        True when the free-space point (x, y) resolves to an ACTIVE snapshot node
        within the graph's own default_lookup_radius. This is the gate path
        smoothing consults for every point it emits: a corner cut into an
        off-graph region (a hole, or outside the mapped area) is rejected and the
        smoother reverts to the original on-graph polyline there.

        Deliberately does NOT probe node isolation. The isolation test walks the
        node's incident edges and rejects far more aggressively than "is this on
        the mapped graph", which made the gate veto ordinary corner cuts; the
        original path is already connected by construction, so connectivity is
        not what this gate is for. cmd_v / cmd_w are still taken (and checked for
        presence) so the signature stays the invocation-context handle.

        Read-only over the snapshot (invariant I7). Returns False rather than
        raising on a lookup failure, so a lookup problem degrades to "no
        smoothing here" instead of losing the whole plan.
        """
        if snapshot is None or cmd_v is None or cmd_w is None:
            return False   # no live snapshot context: reject rather than guess
        try:
            slot = int(snapshot.nearest_node(float(x), float(y)))
        except Exception as exc:
            print(f"[GlobalAStar] smoothing gate: nearest_node({x:.3f}, {y:.3f}) "
                  f"failed ({exc}); rejecting this smoothing step.")
            return False
        return slot >= 0   # < 0: no active node within the lookup radius (off-graph)

    # ------------------------------------------------------------------
    # Full planning pipeline for one invocation (design 6/7/8/15/16)
    # ------------------------------------------------------------------

    def plan(self, start=None, goal=None, start_heading=None) -> bool:
        """
        Acquire one local GraphSnapshot and run the complete heading-aware A*
        search on it. Sets `path` / `plan_cost` / diagnostics and returns True
        on success, False for an ordinary planning failure (start/goal not in
        the graph, or no path). Raises on programming/schema errors.

        `start` / `goal` are (x, y) world points; `start_heading` is the robot's
        arrival heading at `start`. When omitted they fall back to the stored
        constructor values.
        """
        t_start = time.perf_counter()

        # --- reset per-invocation outputs / diagnostics -------------------
        self.path = []
        self.state_path = []
        self.node_path = []
        # A raw graph A* path has no mandatory pass-through waypoint, so it also
        # has no smoothing anchors.
        self._path_anchor_indices = []
        self.plan_cost = 0.0
        self.is_goal_reached = False
        self.no_path_reason = None
        self.start_slot = -1
        self.goal_slot = -1
        self.expanded_count = 0
        self.snapshot_copy_time = 0.0
        self.search_time = 0.0

        # --- physical input validation (design 7.1) -----------------------
        start = self.x_start if start is None else start
        goal = self.x_goal if goal is None else goal
        start_heading = self.heading_start if start_heading is None else start_heading
        start_x, start_y = self._validate_point(start, "start")
        goal_x, goal_y = self._validate_point(goal, "goal")
        start_heading = wrap_to_pi(start_heading)             # wrap once

        # --- one snapshot per invocation (design 6) -----------------------
        self._ensure_reader()
        t_snap = time.perf_counter()
        # First call allocates the buffer; later calls reuse it via into=.
        self.graph_snapshot = self.graph_reader.snapshot(
            into=self.graph_snapshot,
            max_retries=self.snapshot_max_retries,
        )
        snapshot = self.graph_snapshot
        self.snapshot_copy_time = time.perf_counter() - t_snap

        # --- resolve + validate cached payload views once (design 6/7.2) --
        cmd_v = np.asarray(snapshot.node_payload("cmd_v"))    # (n_max, L) stored linear limits
        cmd_w = np.asarray(snapshot.node_payload("cmd_w"))    # (n_max, L) stored angular limits
        self._validate_snapshot_arrays(snapshot, cmd_v, cmd_w)

        # --- start / goal resolution via the graph's own lookup (design 8)-
        # Re-snap past an isolated nearest node (start biased by heading, goal by
        # distance); identical to nearest_node when that node is usable / off-graph.
        start_slot = self._resolve_nonisolated_slot(
            snapshot, cmd_v, cmd_w, start_x, start_y, start_heading, role="start")
        goal_slot = self._resolve_nonisolated_slot(
            snapshot, cmd_v, cmd_w, goal_x, goal_y, None, role="goal")
        if start_slot < 0:
            self.no_path_reason = (
                f"start ({start_x:.3f}, {start_y:.3f}) is not represented by any "
                f"active graph node within the lookup radius")
            self.total_plan_time = time.perf_counter() - t_start
            return False
        if goal_slot < 0:
            self.no_path_reason = (
                f"goal ({goal_x:.3f}, {goal_y:.3f}) is not represented by any "
                f"active graph node within the lookup radius")
            self.total_plan_time = time.perf_counter() - t_start
            return False
        self.start_slot = start_slot
        self.goal_slot = goal_slot

        # --- admissible speed bound (design 14) ---------------------------
        v_upper = self._compute_speed_bound(snapshot, cmd_v)

        # --- sparse A* (design 15) ----------------------------------------
        t_search = time.perf_counter()
        success, goal_state, g_score, came_from, expanded = self._search(
            snapshot, cmd_v, cmd_w, start_slot, goal_slot, start_heading, v_upper)
        self.search_time = time.perf_counter() - t_search
        self.expanded_count = int(expanded)

        if not success:
            self.no_path_reason = (
                f"no graph path from start slot {start_slot} to goal slot "
                f"{goal_slot} (disconnected or all transitions blocked)")
            self.total_plan_time = time.perf_counter() - t_start
            return False

        # --- path reconstruction + outputs (design 16) --------------------
        self.state_path, self.node_path, self.path = self._reconstruct(
            snapshot, came_from, goal_state)
        # Look-ahead circle smoothing of the public path. state_path / node_path
        # keep the RAW graph-node result (they carry snapshot slots and exact
        # arrival headings, which only exist on graph nodes), so after this line
        # self.path is NO LONGER index-aligned with them. The gate is passed
        # explicitly because the snapshot and payload views are plan-local here,
        # not instance state.
        self.path = self._apply_path_smoothing(
            self.path,
            is_valid_fn=lambda px, py: self._graph_point_is_traversable(
                snapshot, cmd_v, cmd_w, px, py),
        )
        self.plan_cost = float(g_score[goal_state])
        self.is_goal_reached = True
        self.total_plan_time = time.perf_counter() - t_start
        return True

    @staticmethod
    def _validate_point(pt, name: str) -> Tuple[float, float]:
        """Require a physical point with finite x and y (design 7.1)."""
        try:
            x = float(pt[0])
            y = float(pt[1])
        except (TypeError, ValueError, IndexError):
            raise ValueError(f"{name} must provide finite (x, y), got {pt!r}")
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"{name} has non-finite coordinates ({x}, {y})")
        return x, y

    # ------------------------------------------------------------------
    # Planning-start selection
    # ------------------------------------------------------------------

    # Lookahead thresholds, expressed in graph-lattice cells (R =
    # global_map.TravUpdate_XYResolution). A* snaps every start to a lattice
    # node, so anything below one cell is invisible to the search.
    #   MIN_CELLS       : floor on the lookahead distance, so the projected start
    #                     can actually reach a different node than the robot's.
    #   MIN_CHORD_CELLS : below this the robot->waypoint chord carries no usable
    #                     direction (pose noise dominates) and no real motion.
    #   GOAL_GAP_CELLS  : never start this close to the goal, or start_slot ==
    #                     goal_slot returns a 1-point "plan" with cost 0.
    _LOOKAHEAD_MIN_CELLS = 1.5
    _LOOKAHEAD_MIN_CHORD_CELLS = 0.5
    _LOOKAHEAD_GOAL_GAP_CELLS = 1.0

    def _path_tangent_at(self, wx: float, wy: float):
        """
        Direction (rad) of the self.path segment carrying the lookahead point
        (wx, wy) — the heading the robot is expected to hold when it arrives
        there, which is what A* stores in SearchState.theta.

        Uses the previous plan's public path (list of [x, y]); the segment is
        taken forward from the nearest path node, since get_waypoint always
        projects the point forward along the path. Returns None when the path
        is too short or the local segment is degenerate, so the caller can fall
        back to the robot->waypoint chord. Never mutates planner state.
        """
        pts = np.asarray(self.path, dtype=float)          # (N, 2) previous public path
        if pts.ndim != 2 or pts.shape[0] < 2:
            return None
        d = np.hypot(pts[:, 0] - wx, pts[:, 1] - wy)      # (N,) distance to the lookahead point
        i = int(np.argmin(d))
        a, b = (pts[i], pts[i + 1]) if i + 1 < pts.shape[0] else (pts[i - 1], pts[i])
        tx, ty = float(b[0] - a[0]), float(b[1] - a[1])   # segment direction vector
        if math.hypot(tx, ty) < 1e-6:
            return None
        return math.atan2(ty, tx)

    def _compute_start(self, initial_start, total_lookahead_time):
        """
        Resolve the planning start pose (sx, sy, sh) without mutating any
        planner state. Applies a 2-branch priority (waypoint
        lookahead > current robot pose) over the A* tuple endpoints
        (self.x_start / self.x_goal).


        Returns:
            (sx, sy, sh): planning start x, y (world coords) and heading (rad).
        """
        # Graph node spacing (m). Every threshold below is scaled by it, so an
        # absent / invalid value must fail loud rather than silently disable the
        # guards. Set by CMDbasedMap.__init__ from cfg.trav_estimation_resoultion
        # in every mode (ROS / shared-memory / backend reader).
        lattice_R = getattr(self.global_map, "TravUpdate_XYResolution", None)
        if lattice_R is None or not math.isfinite(float(lattice_R)) or float(lattice_R) <= 0.0:
            raise RuntimeError(
                f"GlobalAStar._compute_start: global_map.TravUpdate_XYResolution "
                f"must be a positive finite graph spacing, got {lattice_R!r}")
        lattice_R = float(lattice_R)

        # --- Branch 1: waypoint lookahead along the last path -------------
        # Project a point (scaled) reachable_radius ahead of the robot on
        # self.path so the plan starts roughly where the robot will be once
        # planning finishes. get_waypoint returns False when self.path is empty.
        [cmd_v_limit, cmd_w_limit] = self.global_map.get_cmd_limits()
        # Effective lookahead DISTANCE (m), floored at _LOOKAHEAD_MIN_CELLS cells.
        # get_waypoint's reachable radius is total_lookahead_time * cmd_v_limit,
        # so passing the distance with cmd_v_limit pinned to 1.0 makes the radius
        # exactly this value.
        lookahead_dist = max(float(total_lookahead_time) * float(cmd_v_limit),
                             self._LOOKAHEAD_MIN_CELLS * lattice_R)
        waypoint = self.global_map.get_waypoint(
            self.global_map.robot_x, self.global_map.robot_y,
            self.global_map.robot_heading,
            lookahead_dist,
            1.0, cmd_w_limit, self.path)

        # Use the lookahead point only when it is valid, actually moves the start,
        # is not effectively at the goal, and the robot has already left the config
        # initial_start (not the very first plan). All thresholds are lattice-scaled
        # because the start is snapped to a graph node before the search runs.
        if waypoint is not False:
            # chord: robot -> lookahead point distance (m) and fallback direction
            dx_wp = float(waypoint[0]) - self.global_map.robot_x
            dy_wp = float(waypoint[1]) - self.global_map.robot_y
            chord = math.hypot(dx_wp, dy_wp)
            # goal_gap: distance from the lookahead point to the global goal (m).
            # The old test compared coordinates for exact equality, which can never
            # match in A*: self.path holds graph-node / smoothed coordinates while
            # self.x_goal is the raw config goal. A metric gap is what was meant.
            goal_gap = math.hypot(float(waypoint[0]) - self.x_goal[0],
                                  float(waypoint[1]) - self.x_goal[1])
            # "Has the robot actually left the configured initial start?" Compared
            # against the LIVE robot pose: self.x_start is overwritten with the
            # previous lookahead start on every replan, so the original test only
            # ever suppressed the very first cycle. initial_start None means there
            # is no config start to be pinned at, so the lookahead is allowed.
            moved_off_initial = initial_start is None or math.hypot(
                self.global_map.robot_x - float(initial_start[0]),
                self.global_map.robot_y - float(initial_start[1])
            ) > self._LOOKAHEAD_MIN_CHORD_CELLS * lattice_R

            if chord > self._LOOKAHEAD_MIN_CHORD_CELLS * lattice_R \
                    and goal_gap > self._LOOKAHEAD_GOAL_GAP_CELLS * lattice_R \
                    and moved_off_initial:
                # Start heading = the heading the robot will HAVE on arriving at the
                # lookahead point (SearchState.theta is an ARRIVAL heading), i.e. the
                # path tangent there. The robot->waypoint chord is only the fallback:
                # after snapping, the start node can sit up to R/sqrt(2) from the
                # projected point, and a short chord is dominated by pose noise.
                sh = self._path_tangent_at(float(waypoint[0]), float(waypoint[1]))
                if sh is None:
                    sh = math.atan2(dy_wp, dx_wp)
                return float(waypoint[0]), float(waypoint[1]), float(sh)

            print(f" [Lookahead] rejected (chord {chord:.3f} m, goal gap "
                  f"{goal_gap:.3f} m, moved_off_initial {moved_off_initial}); "
                  f"starting from the robot pose.")

        # --- Branch 2: current robot pose (fallback) ----------------------
        return (float(self.global_map.robot_x),
                float(self.global_map.robot_y),
                float(self.global_map.robot_heading))

    # ------------------------------------------------------------------
    # Node-facing entry point
    # ------------------------------------------------------------------

    def replan(self, initial_start=None, total_lookahead_time=0.5,
               plot_map=False):
        """
        Node-facing replan wrapper. Determines the planning start pose (waypoint
        lookahead > current robot pose), keeps the fixed global goal, runs the graph
        A*, and logs timing/diagnostics.
        """
        if self.global_map is None:
            raise RuntimeError("GlobalAStar.replan: global_map is not attached")
        if not self.global_map.is_TraversabilityMap_built:
            # Nothing to plan on yet.
            return

        start_time = time.time()
        print("\nReplanning Global Path (A* on shared graph)!")

        # --- select the planning start (waypoint lookahead > robot pose) -
        # Shared 2-branch selection (see _compute_start). initial_start is threaded
        # through so the lookahead's "robot has moved off the config start" guard
        # can fire; the executable already passes it to replan().
        sx, sy, sh = self._compute_start(initial_start, total_lookahead_time)

        gx, gy = self.x_goal
        self.x_start = (sx, sy)
        self.heading_start = sh

        print(f" Start: ({sx:.3f}, {sy:.3f}), heading {np.rad2deg(sh):.1f} deg")
        print(f" Goal:  ({gx:.3f}, {gy:.3f})")

        success = self.plan(start=(sx, sy), goal=(gx, gy), start_heading=sh)

        # Lookahead-start recovery: a start projected FORWARD can leave the mapped
        # graph (map boundary / hole) or land on a disconnected pocket while the
        # robot's own pose is still plannable. plan() has already cleared
        # self.path, so without this retry the whole cycle publishes nothing.
        if not success:
            rx = float(self.global_map.robot_x)
            ry = float(self.global_map.robot_y)
            rh = float(self.global_map.robot_heading)
            if (rx, ry) != (sx, sy):
                print(f"  [Retry] lookahead start ({sx:.3f}, {sy:.3f}) failed "
                      f"({self.no_path_reason}); retrying from the robot pose "
                      f"({rx:.3f}, {ry:.3f}).")
                sx, sy, sh = rx, ry, rh
                self.x_start = (sx, sy)
                self.heading_start = sh
                success = self.plan(start=(sx, sy), goal=(gx, gy), start_heading=sh)

        # --- diagnostics summary (design 19) ------------------------------
        print(f"  A* Computation Time: {np.round(time.time() - start_time, 3)} s "
              f"(snapshot {self.snapshot_copy_time * 1e3:.1f} ms, "
              f"search {self.search_time * 1e3:.1f} ms, "
              f"expanded {self.expanded_count} states)")
        if success:
            print(f"  Resolved start slot {self.start_slot} -> goal slot {self.goal_slot}; "
                  f"path nodes: {len(self.path)}")
            print(f' Total Navigation Cost (Estimated Traversal Time): '
                  f'{np.round(self.plan_cost, 3)} s')
            print("Global Path Replanned!\n")
        else:
            print(f"  No path: {self.no_path_reason}\n")

        if plot_map:
            self.plot_map()

    # ------------------------------------------------------------------
    # Plotting (the node passes plot_map=False)
    # ------------------------------------------------------------------

    def plot_map(self):
        if self.plotting is None:
            return
        if self.path:
            self.plotting.plot_grid(
                f"A* on shared graph, ETA = {np.round(self.plan_cost, 1)}")
        else:
            self.plotting.plot_grid("Plain Environment")
