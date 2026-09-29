from abc import ABCMeta, abstractmethod
from typing import List
import importlib.util
import math
from pathlib import Path
import numpy as np

# Path-smoothing helper, resolved lazily and cached (see _get_smoother below).
# It must NOT be a top-level import: statenav_global/__init__ imports `planners`
# first, planners/__init__ imports THIS module first, so importing
# statenav_global.utility here would re-enter a half-initialized package (and
# utility/__init__ re-exports plotting, pulling the ROS/torch chain). Resolving
# on first use instead means the package is fully initialized by then, and the
# standalone test harness — which loads base_planner.py directly and expects
# only abc/typing/numpy — keeps working.
_smooth_path_lookahead = None


def _get_smoother():
    """
    Return utils.smooth_path_lookahead, importing it on first use.

    Falls back to loading utility/utils.py by path when the package import
    fails, which is the case outside the dev container: utility/__init__ pulls
    plotting -> ROS, while utils.py itself needs only math + numpy. Same
    fallback pattern global_AStar_planner.py uses for its helpers.
    """
    global _smooth_path_lookahead
    if _smooth_path_lookahead is None:
        try:
            from statenav_global.utility.utils import smooth_path_lookahead
        except ModuleNotFoundError:
            _utils_path = Path(__file__).resolve().parents[1] / "utility" / "utils.py"
            _spec = importlib.util.spec_from_file_location("_base_planner_utils", _utils_path)
            if _spec is None or _spec.loader is None:
                raise ImportError(f"Unable to load utility helpers from {_utils_path}")
            _utils = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_utils)
            smooth_path_lookahead = _utils.smooth_path_lookahead
        _smooth_path_lookahead = smooth_path_lookahead
    return _smooth_path_lookahead


class BasePlanner(metaclass=ABCMeta):
    def __init__(self, task_extent: List[float], rng: np.random.RandomState) -> None:
        self.task_extent = task_extent
        self.rng = rng

        # --- path smoothing (look-ahead circle corner cutting) -------------
        # Shared by every planner engine; the planning node overwrites these
        # from planning_config.yaml right after construction (same post-init
        # attach pattern as global_map).
        self.path_smoothing_enable = True    # False publishes the raw planner path
        self.path_smoothing_radius = 0.5     # R (m): look-ahead circle radius
        self.path_smoothing_step = 0.25      # d (m): advance per iteration; must be < R
        # Indices into self.path that smoothing must preserve verbatim.
        # Rebuilt on every plan().
        self._path_anchor_indices = []

    @abstractmethod
    def plan(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Path smoothing (look-ahead circle / pursuit-style corner cutting)
    # ------------------------------------------------------------------

    def _smoothing_point_is_valid(self, x: float, y: float) -> bool:
        """
        Traversability gate consulted for every smoothed point. Default: accept
        everything, i.e. purely geometric smoothing.

        Engines that can cheaply answer "is this free-space point traversable?"
        override this (GlobalAStar queries the invocation GraphSnapshot). An
        engine WITHOUT a real gate still gets smoothing, but a corner cut is
        then never rejected and CAN cross non-traversable terrain.
        """
        return True

    def _smoothing_sample_step(self) -> float:
        """
        Chord-sampling spacing (m) handed to the validity gate: HALF a map cell.

        The smoothing step is 0.25 m by default, i.e. several cells, so the
        chord must be sampled rather than only its endpoint. Half a cell (not a
        full cell) guarantees that any cell the chord crosses over at least half
        its width gets probed; the graph's own lookup tolerance is 0.75 cells,
        so the probes overlap and no thin blocked strip slips between them.
        Only a corner graze shorter than half a cell can still be missed.

        Falls back to the smoothing step when the map resolution is not
        available (no global_map attached, or no map received yet), which only
        makes the gate coarser — never unsafe to call.
        """
        res = getattr(getattr(self, "global_map", None), "map_resolution", None)
        try:
            res = float(res)
        except (TypeError, ValueError):
            return float(self.path_smoothing_step)
        if not math.isfinite(res) or res <= 0.0:
            return float(self.path_smoothing_step)
        return 0.5 * res

    def _apply_path_smoothing(self, path, is_valid_fn=None):
        """
        Smooth the just-planned `path` and return the result.

        Args:
            path       : list of [x, y] — the raw planner output.
            is_valid_fn: explicit traversability gate; defaults to the
                         self._smoothing_point_is_valid hook. GlobalAStar.plan()
                         passes a closure instead, because its snapshot and
                         payload views are plan-local rather than instance state.

        Guarded end-to-end: any smoothing failure reports the reason and returns
        the ORIGINAL path, which lies on graph edges (A*) and is therefore
        traversable by construction. A
        smoother bug can degrade path quality but never break navigation.
        """
        if not self.path_smoothing_enable:
            return path
        if path is None or len(path) < 3:
            return path   # nothing to cut: a 2-point path is already a straight line

        if is_valid_fn is None:
            is_valid_fn = self._smoothing_point_is_valid
        try:
            smoothed, stats = _get_smoother()(
                path,
                smoothing_radius=self.path_smoothing_radius,
                step=self.path_smoothing_step,
                anchor_indices=self._path_anchor_indices,
                is_valid_fn=is_valid_fn,
                sample_step=self._smoothing_sample_step(),
            )
        except Exception as exc:
            print(f"[Planning] Path smoothing FAILED (R={self.path_smoothing_radius}, "
                  f"d={self.path_smoothing_step}, {len(path)} nodes, "
                  f"anchors={self._path_anchor_indices}): {exc}. "
                  f"Publishing the unsmoothed path.")
            return path

        if not smoothed:
            print(f"[Planning] Path smoothing produced an empty path from "
                  f"{len(path)} nodes; publishing the unsmoothed path.")
            return path

        # Report what the smoother had to give up on. rejected_steps > 0 means
        # the terrain gate vetoed corner cuts and those stretches fell back to
        # the original path — expected near tight/gray regions, but a large
        # count means R is too aggressive for this map.
        msg = (f"[Planning] Path smoothing: {stats['n_in']} -> {stats['n_out']} nodes "
               f"(R={self.path_smoothing_radius}, d={self.path_smoothing_step}, "
               f"anchors={self._path_anchor_indices})")
        if stats['rejected_steps'] or stats['unsafe_snapbacks'] or stats['iter_cap_hit']:
            msg += (f"; gate-rejected steps: {stats['rejected_steps']}, "
                    f"unsafe snap-backs: {stats['unsafe_snapbacks']}, "
                    f"iteration cap hit: {stats['iter_cap_hit']}")
        print(msg)
        return smoothed
