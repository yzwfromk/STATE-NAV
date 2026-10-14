"""Traversability-map layer: snapshot cache, footprint reduction and terrain cost.

Everything that reads the shared StateNav map sits here, between the solver and the
path-tracking layer:

    MPPICore -> MPPISolver -> TerrainMPPI -> PathTrackingMPPI -> MPPIPlanner

It answers the one question MPPICore declares but cannot resolve -- what command domain
is in force at a given state -- and it prices staying somewhere the map allows only small
commands.  Terrain values come from the world model's shared travgraph, sigmoid-interpolated
onto the map's fine grid at runtime (mppi_travgraph.py); of the shared map only its metadata
(bounds, resolution, size, built flag) is read, never the dense TraversabilityMap.  No
writer-side change is required.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import torch

from statenav_mppi.mppi_core.mppi_core import _TWO_PI
from statenav_mppi.mppi_core.mppi_constraints import ConstraintTermTable
from statenav_mppi.mppi_core.mppi_validity import ValidityTermTable
from statenav_mppi.mppi_core.mppi_costs import CostContext, CostTermTable
from statenav_mppi.mppi_core.mppi_solver import MPPISolver
from statenav_mppi.mppi_core.mppi_types import MPPIConfig, MPPIInput
from statenav_mppi.mppi_planners.mppi_travgraph import TravGraphTerrain


class _TravmapTerm:
    """The terrain penalty, plus the per-knot map snapshot the tuning trace prints.

    An object rather than a bound method for the same reason _ProgressTerm is one:
    CostTermTable.debug_lines() finds debug_line() by attribute lookup on the registered
    callable.  The arithmetic stays on the planner in _term_Travmap; this only owns the
    snapshot.

    Travmap's LEVEL alone cannot be acted on.  It is a mean of ``1/safe_v + 1/safe_w``
    over the horizon, floored per channel at the too-risky stamp, so one number covers
    "the whole plan sits in mildly slow terrain" and "two knots clip a blocked cell and
    the rest are free" -- which want opposite responses.  The per-knot limits below
    separate them, and they are the only record of what the map said: the map itself is
    read from shared memory and never lands in a log.

    Registered run-independently (fn=, not make=), so ONE instance serves every tick.  The
    snapshot is therefore cleared on every batch that is not the settled batch-of-one,
    which is the LAST evaluation of a tick -- so debug_line() can only ever report the
    plan actually being issued, never a sample batch and never a previous tick's.
    """

    def __init__(self, planner: "TerrainMPPI") -> None:
        self._planner = planner
        # (safe_v, safe_w) of the issued plan, or None outside a batch-of-one evaluation.
        self._debug: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        # (x, y, point_v) each (1,T,1+N) raw footprint samples of the issued plan, plus (1,T)
        # movement heading; debug_print only, so the control path pays no extra lookup.
        self._points: Optional[tuple[torch.Tensor, ...]] = None

    def __call__(self, ctx: CostContext) -> torch.Tensor:
        value = self._planner._term_Travmap(ctx)
        # (...,T) footprint-reduced limits.  _term_Travmap has just memoized these on ctx,
        # so this is a cache hit and costs no second map lookup.
        safe_v, safe_w, _ = self._planner._cached_footprint_limits(ctx)
        self._debug = (
            (safe_v.detach(), safe_w.detach())
            if safe_v.ndim == 2 and safe_v.shape[0] == 1 else None)
        self._points = None
        if self._debug is not None and bool(self._planner.config.debug_print):
            # Re-read the raw per-point map values of the issued plan with the SAME heading
            # _cached_footprint_limits uses, so the trace shows what the reduction was fed.
            heading = ctx.states[..., 2]
            if ctx.data.reversed_geometry:
                heading = self._planner._wrap_torch(heading + math.pi)
            x, y = self._planner._footprint_sample_positions(ctx.states)
            point_v, _ = self._planner._lookup_cached_map(
                x, y, heading[..., None].expand_as(x))
            self._points = (x.detach(), y.detach(), point_v.detach(), heading.detach())
        return value

    def debug_line(self) -> Optional[str]:
        """One trace line, or None when this evaluation was not the issued batch-of-one.

        Resolved here rather than inside __call__ for the reason _ProgressTerm gives:
        the snapshot holds TENSORS, and reading them per sample batch would force a
        device sync on the control path.

        ``channels`` names whether the map's linear and angular layers actually carry
        independent information.  When they are identical the term is 2/safe_v wearing
        two names, toorisky_w is dead weight, and no angular-only retune can do anything.
        """
        snapshot = self._debug
        if snapshot is None:
            return None
        config = self._planner.config
        safe_v = snapshot[0].reshape(-1).cpu().numpy()
        safe_w = snapshot[1].reshape(-1).cpu().numpy()
        floor_v, floor_w = float(config.too_risky_v), float(config.too_risky_w)
        # (T,) the SAME flooring _inverse_limit applies: non-finite and at-or-below-floor
        # limits both pay exactly 1/floor, so the per-knot cost below matches the term.
        def _cost(values: np.ndarray, floor: float) -> np.ndarray:
            usable = np.isfinite(values) & (values > floor)
            return 1.0 / np.where(usable, values, floor)
        per_knot = _cost(safe_v, floor_v) + _cost(safe_w, floor_w)
        channels = ("IDENTICAL -- the map's angular layer carries no extra information"
                    if np.allclose(safe_v, safe_w, rtol=0.0, atol=1.0e-6)
                    else "independent")
        def series(values: np.ndarray, digits: int) -> str:
            return "[" + " ".join(f"{float(value):.{digits}f}" for value in values) + "]"
        text = (
            f"[mppi][dbg]   Travmap: safe_v={series(safe_v, 3)} "
            f"safe_w={series(safe_w, 3)} m/s,rad/s (channels {channels})\n"
            f"[mppi][dbg]     per-knot 1/safe_v+1/safe_w={series(per_knot, 1)} "
            f"mean={float(np.mean(per_knot)):.2f} "
            f"(blocked-cell cap {1.0 / floor_v + 1.0 / floor_w:.1f} at toorisky "
            f"{floor_v:.3f}/{floor_w:.3f})")
        points = self._points
        geometry = self._planner._map_geometry
        if points is None or geometry is None:
            return text
        # (T,1+N) raw point samples; index 0 is the body centre, 1.. the footprint ring.
        px, py, pv = (t.reshape(-1, t.shape[-1]).cpu().numpy() for t in points[:3])
        heading_deg = np.degrees(points[3].reshape(-1).cpu().numpy())
        _, xmax, _, ymax, resolution, _, _, layers = geometry
        lines = [text, (
            f"[mppi][dbg]     footprint points (raw safe_v, blocked = <= toorisky {floor_v:.3f}): "
            f"map res={float(resolution):.3f} xmax={float(xmax):.2f} ymax={float(ymax):.2f} "
            f"layers={int(layers)}")]
        for k in range(pv.shape[0]):
            blocked = ~(np.isfinite(pv[k]) & (pv[k] > floor_v))
            worst = int(np.argmin(np.where(np.isfinite(pv[k]), pv[k], -np.inf)))
            # Centre of the map cell the body centre falls in, for comparison with graph nodes.
            cell_x = float(xmax) - float(resolution) * (
                math.floor((float(xmax) - px[k, 0]) / float(resolution)) + 0.5)
            cell_y = float(ymax) - float(resolution) * (
                math.floor((float(ymax) - py[k, 0]) / float(resolution)) + 0.5)
            lines.append(
                f"[mppi][dbg]       k{k} centre=({px[k, 0]:+.2f},{py[k, 0]:+.2f}) "
                f"cell=({cell_x:+.2f},{cell_y:+.2f}) heading={heading_deg[k]:+.0f}deg "
                f"centre_v={pv[k, 0]:.3f} min_v={pv[k, worst]:.3f} at "
                f"({px[k, worst]:+.2f},{py[k, worst]:+.2f}) blocked={int(blocked.sum())}/{pv.shape[1]}")
        return "\n".join(lines)


class TerrainMPPI(MPPISolver):
    """MPPI solver plus version-aware traversability-map caching and terrain costs."""

    def __init__(
        self,
        config: MPPIConfig,
        global_map: Any = None,
        travgraph_reader: Any = None,
    ) -> None:
        super().__init__(config)
        self.global_map = global_map
        self._footprint_offsets = self._build_footprint_offsets()
        # The terrain source: the world model's lattice graph, interpolated at runtime.  Its
        # reader is travgraph_reader when injected, else the one an in-memory map carries as
        # `travgraph_reader` (demo, tests), else the shared-memory graph, attached lazily.
        self._travgraph: Optional[TravGraphTerrain] = None
        if global_map is not None:
            self._travgraph = TravGraphTerrain(
                config.travgraph_resolution, config.travgraph_sigmoid_k,
                reader=(travgraph_reader if travgraph_reader is not None
                        else getattr(global_map, "travgraph_reader", None)),
                rasterize=config.travgraph_rasterize,
                window_radius=config.travgraph_window_radius)

        self._map_version: Optional[int] = None
        # The plot background is one heading layer of the command map, copied lazily by
        # visualization_snapshot() rather than on every map refresh, so a headless run never
        # pays for it.  The key is (map version, layer) rather than the version alone because
        # the layer follows the robot heading, which changes far more often than the map.
        self._trav_layer_snapshot: Optional[np.ndarray] = None
        self._trav_layer_key: Optional[tuple[int, int]] = None
        # Colour limits taken over every heading layer so the background does not re-scale each
        # time the robot rotates into a different layer.  Cached per map version.
        self._trav_clim: Optional[tuple[float, float]] = None
        self._trav_clim_version: Optional[int] = None
        # Latched once the heading-bin convention has been checked against the map metadata.
        self._layer_convention_checked = False
        self._map_geometry: Optional[tuple[float, float, float, float, float, int, int, int]] = None




####################################### FOOTPRINT STENCIL #######################################
# The ring of sample points every map query is reduced over, so a hazard beside the
# body is seen and not only one directly under its centre.

    def _build_footprint_offsets(self) -> torch.Tensor:
        """Return the cached ``(1 + N, 2)`` ring footprint stencil in metres.

        Row 0 is the state centre so a hazard directly under the body still
        reaches the harmonic mean; the remaining ``footprint_samples`` rows are
        evenly spaced on the ``footprint_radius`` circle.  The ring is
        rotation invariant, so the offsets are reused as world-frame deltas.
        """
        radius = float(self.config.footprint_radius)
        count = int(self.config.footprint_samples)
        # (N,) evenly spaced circumference angles in radians, starting at +x.
        angles = torch.arange(
            count, device=self.device, dtype=self.dtype,
        ) * (2.0 * math.pi / count)
        # (N, 2) ring offsets, each exactly `radius` metres from the centre.
        ring = torch.stack(
            (radius * torch.cos(angles), radius * torch.sin(angles)), dim=-1)
        centre = torch.zeros((1, 2), device=self.device, dtype=self.dtype)
        return torch.cat((centre, ring), dim=0)

    def _footprint_sample_positions(
        self, states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Offset each state's ``(x, y)`` onto the cached ring stencil.

        Returns two tensors shaped ``states.shape[:-1] + (1 + N,)``.  A circle
        is rotation invariant, so body yaw does not change the sampled set and
        no per-state rotation is applied.
        """
        x = states[..., 0, None] + self._footprint_offsets[:, 0]
        y = states[..., 1, None] + self._footprint_offsets[:, 1]
        return x, y




####################################### MAP SNAPSHOT CACHE #######################################
# One pinned copy of the shared map per version, so every rollout in a tick reads the
# same terrain and the writer can advance underneath us without tearing a solve.

    def _require_map_snapshot(self, context: str) -> None:
        """Raise unless a map snapshot is pinned for this tick.

        Guards the precondition that every rollout in a tick reads ONE terrain version.
        Without it a forgotten pin fails SILENTLY and INCONSISTENTLY: _constraint_Terrain
        returns the caller's unconstrained limits while _lookup_cached_map returns +inf,
        which _inverse_limit floors -- so the terrain COST becomes maximal at the same
        moment the command DOMAIN becomes unbounded.
        """
        if self.global_map is not None and not self._map_snapshot_pinned():
            raise RuntimeError(
                f"{context}: no traversability snapshot is pinned; the per-run entrypoint "
                "must call _ensure_map_cache() before solve() (see the map-snapshot cache "
                "contract in mppi_terrain.py)")

    def _ensure_map_cache(self, center: Optional[Any] = None) -> None:
        """Pin the travgraph version every rollout of this tick interpolates.

        The fine grid comes from the map's METADATA (bounds, resolution, size), never from a
        dense TraversabilityMap array: the writer's interpolation is defined on that grid, and
        the cell/layer indexing below runs on it.  The version is the graph's own seqlock
        counter, so a re-snapshot costs O(nodes), and only when the writer published an edit.
        ``center`` is the robot's world (x, y, ...) the travgraph read window follows
        (travgraph_window_radius); None keeps the last one.
        """
        if self.global_map is None:
            return
        if hasattr(self.global_map, "is_TraversabilityMap_built"):
            if not bool(self.global_map.is_TraversabilityMap_built):
                raise RuntimeError("traversability map is not built")
        global_map = self.global_map
        resolution = float(global_map.map_resolution)
        if not math.isfinite(resolution) or resolution <= 0.0:
            raise ValueError("invalid map resolution")
        rows = int(global_map.map_size_rows)
        cols = int(global_map.map_size_cols)
        layers = int(global_map.TraversabilityMap_size_layers)
        if rows <= 0 or cols <= 0 or layers <= 0:
            raise ValueError(
                f"invalid map size for the travgraph grid: rows={rows} cols={cols} layers={layers}")
        geometry = (
            float(global_map.map_xmin), float(global_map.map_xmax),
            float(global_map.map_ymin), float(global_map.map_ymax),
            resolution, rows, cols, layers,
        )
        self._travgraph.refresh(geometry, self.device, self.dtype, center)
        self._map_geometry = geometry
        self._map_version = self._travgraph.version

    def _map_snapshot_pinned(self) -> bool:
        """True when this tick's travgraph tables are pinned on the device."""
        return (self._map_geometry is not None and self._travgraph is not None
                and self._travgraph.is_ready)




####################################### HEADING LAYERS #######################################
# The map is binned by heading; these resolve a heading to its bin and flag a map
# whose published convention disagrees with the one every lookup assumes.

    def _heading_layer_index(self, heading: float) -> Optional[int]:
        """Return the heading layer whose bin centre is nearest ``heading`` radians.

        Runs the identical float32 arithmetic as _lookup_cached_map() - deliberately through
        torch on a one-element tensor rather than a re-derived float64 scalar formula - so a
        plotted layer is bit-for-bit the layer the solver is constrained by at that heading.
        A float64 restatement disagrees with the solver for headings sitting exactly on a bin
        boundary, which is precisely where an operator would question the plot.  Returns None
        before the first successful map snapshot (the layer count is unknown) or for a
        non-finite heading, which has no bin.
        """
        if self._map_geometry is None:
            return None
        layers = int(self._map_geometry[7])
        if layers <= 0 or not math.isfinite(heading):
            return None
        heading_tensor = torch.tensor(
            float(heading), device=self.device, dtype=self.dtype)
        wrapped = self._wrap_torch(heading_tensor)
        layer = torch.floor(
            torch.remainder(wrapped + math.pi, _TWO_PI) / (_TWO_PI / layers) + 0.5)
        return int(torch.remainder(layer.to(torch.long), layers))

    def _warn_layer_convention_once(self) -> None:
        """Warn once if the map's heading-bin metadata contradicts the assumed convention.

        Both _lookup_cached_map() and _heading_layer_index() assume the layer axis starts at
        theta_min = -pi and covers one full revolution in `layers` uniform bins.  A map that
        publishes a different origin or resolution would make every layer lookup - control and
        plot alike - read the wrong heading bin, so surface it instead of failing silently.
        Maps that do not publish the metadata (the demo maps) are skipped, not flagged.
        """
        if self._layer_convention_checked or self._map_geometry is None:
            return
        self._layer_convention_checked = True
        layers = int(self._map_geometry[7])
        if layers <= 0:
            return
        theta_min = getattr(self.global_map, "TraversabilityMap_theta_min", None)
        theta_resolution = getattr(
            self.global_map, "TraversabilityMap_theta_resolution", None)
        expected_resolution = _TWO_PI / layers
        problems: list[str] = []
        if theta_min is not None and not math.isclose(
                float(theta_min), -math.pi, abs_tol=1.0e-6):
            problems.append(
                f"theta_min={float(theta_min):.6f} (expected {-math.pi:.6f})")
        if theta_resolution is not None and not math.isclose(
                float(theta_resolution), expected_resolution, rel_tol=1.0e-6):
            problems.append(
                f"theta_resolution={float(theta_resolution):.6f} "
                f"(expected {expected_resolution:.6f} for {layers} layers)")
        if problems:
            print(
                "[MPPI] traversability heading-layer convention mismatch: "
                + ", ".join(problems)
                + " - layer lookups (control and plot) may read the wrong heading bin")




####################################### MAP LOOKUP #######################################
# Batched world-frame queries against the pinned snapshot.  Nothing here rejects a
# sample: out-of-extent, too-risky and non-finite points are all the caller's to own.

    def lookup_map(
        self, x: torch.Tensor, y: torch.Tensor, heading: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Refresh the shared-map cache, then batch-lookup world-frame states."""
        self._ensure_map_cache()
        return self._lookup_cached_map(x, y, heading)

    def _lookup_cached_map(
        self, x: torch.Tensor, y: torch.Tensor, heading: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch lookup against the map snapshot pinned by the current solve.

        Returns the raw ``(safe_v, safe_w)`` command channels only: the cell
        acceptance test this used to compute is gone, so nothing here rejects a
        sample.  Consequences the callers must own:
          * Out-of-extent samples are NOT flagged.  row/col are clamped below,
            so a point outside the map silently reads the nearest edge cell and
            inherits its limits.
          * Too-risky cells are NOT flagged.  The writer stamps toorisky_v/w
            into the command channels of regions it invalidates, so those cells
            now surface as very small positive limits and are penalized by the
            soft map cost instead of being rejected.
          * Non-finite x/y are NOT flagged.  Indexing stays in range because of
            the clamp, and non-finite rollouts are still dropped by CostManager.
          * Cells outside the travgraph read window or next to a graph hole read NaN
            (mppi_travgraph.py), which _validity_MapUsable hard-vetoes.
        """
        if not self._map_snapshot_pinned():
            infinity = torch.full_like(x, float("inf"))
            return infinity, infinity

        # _map_geometry layout: (xmin, xmax, ymin, ymax, resolution, rows, cols,
        # layers).  xmin/ymin were only needed by the dropped extent test.
        _, xmax, _, ymax, resolution, rows, cols, layers = self._map_geometry
        row = torch.floor((xmax - x) / resolution).to(torch.long).clamp(0, rows - 1)
        col = torch.floor((ymax - y) / resolution).to(torch.long).clamp(0, cols - 1)
        wrapped = self._wrap_torch(heading)
        layer = torch.floor(torch.remainder(wrapped + math.pi, _TWO_PI) / (_TWO_PI / layers) + 0.5)
        layer = torch.remainder(layer.to(torch.long), layers)

        # (..., 2) the writer's sigmoid interpolation of the lattice graph at this fine cell
        # (mppi_travgraph.py); no dense map is read.
        cmd = self._travgraph.gather(row, col, layer)
        # Per-cell TravMap_MetaInfo is deliberately not consulted: once
        # is_TraversabilityMap_built is set (enforced by _ensure_map_cache), the
        # whole extent is planned over and never-estimated cells are accepted at
        # their InitialGuess command values.
        return cmd[..., 0], cmd[..., 1]




####################################### COMMAND-LIMIT REDUCTION #######################################
# Per-point map limits and the reduction helpers shared by both consumers: the terrain
# COST reduces them over the footprint, the command rectangle reads the body-centre cell
# alone, and the (safe_v, safe_w) pair a state is constrained by follows from the latter.

    @staticmethod
    def _inverse_limit(values: torch.Tensor, floor: float) -> torch.Tensor:
        """Return ``1/limit`` with unusable limits pinned to ``floor``.

        ``values`` holds safe command limits of any shape.  ``floor`` is the
        SAME channel's configured too-risky stamp -- ``config.too_risky_v``
        (``toorisky_v``) for a linear channel, ``config.too_risky_w``
        (``toorisky_w``) for an angular one -- so a too-risky cell pays exactly
        the maximum ``1/floor`` that channel can cost and no more.  Non-finite,
        non-positive, and below-floor entries all reduce to the floor, so the
        result is finite and bounded by ``1/floor``: an unclamped ``1/limit``
        explodes on near-zero limits (the footprint reduction can reach ~1e-6)
        and would collapse the softmax onto a single rollout or turn zero/NaN
        limits into non-finite costs.  Cells that cannot constrain a command are
        still hard-rejected through the validity tensor; this clamp only keeps
        their cost from poisoning the softmax with NaN or inf before that
        rejection happens.  ``__post_init__`` guarantees ``floor > 0``.
        """
        floor_values = torch.full_like(values, float(floor))
        sanitized = torch.where(
            torch.isfinite(values) & (values > float(floor)), values, floor_values)
        return torch.reciprocal(sanitized)

    @staticmethod
    def _power_mean_positive(
        values: torch.Tensor, floor: float, order: float = -1.0,
    ) -> torch.Tensor:
        """Reduce the last axis without allowing invalid limits to make NaNs.

        ``order`` is the power-mean order p: -1 is the harmonic mean, 0 the
        geometric mean.  Flooring happens BEFORE the reduction at every order, so
        a non-finite, non-positive or below-stamp cell still collapses to the
        channel stamp; ``floor`` must therefore be the same channel stamp the map
        cost later applies to this reduction's output.

        At the default p = -1, reducing the same floored reciprocals the map cost
        uses keeps the identity ``1 / harmonic_mean == mean(1 / floored point
        limits)``: one blocked footprint point out of ``N`` then raises the cost
        by ``1/N`` of the maximum instead of saturating the whole footprint,
        which a much smaller floor would do.

        Above p = -1 that identity is deliberately given up, and the LINEARITY in
        the blocked-point count with it.  Both ENDS stay pinned -- a power mean of
        identical values is that value, so an all-free and an all-blocked
        footprint reduce exactly as they do at p = -1 -- while a PARTIAL overlap
        reduces to a larger limit and so costs less.  That is the one way to relax
        what a footprint grazing an obstacle edge pays without also lowering the
        all-blocked barrier; see MPPIConfig.travmap_footprint_harmonic_power for the
        numbers and for why a positive order is rejected.
        """
        # (..., N) 1/limit at every point, bounded by 1/floor.  The whole reduction is
        # expressed in these reciprocals so the flooring rule lives in _inverse_limit alone:
        # with r = 1/s and q = -p, the power mean of the floored limits s is
        # M_p = mean(r ** q) ** (-1 / q).
        reciprocals = TerrainMPPI._inverse_limit(values, floor)
        if float(order) == -1.0:
            # The q = 1 case of the formula below, kept as the original expression so the
            # default path stays bit-for-bit what it was rather than routing through pow().
            return values.shape[-1] / torch.sum(reciprocals, dim=-1)
        if float(order) == 0.0:
            # Geometric mean: the q -> 0 limit, where the formula below is degenerate.
            return torch.exp(-torch.mean(torch.log(reciprocals), dim=-1))
        q = -float(order)
        return torch.pow(
            torch.mean(torch.pow(reciprocals, q), dim=-1), -1.0 / q)

    @staticmethod
    def _usable_limits(
        safe_v: torch.Tensor, safe_w: torch.Tensor,
    ) -> torch.Tensor:
        """True where both looked-up channels can act as command constraints.

        This is the only acceptance test left: a limit is unusable when it is
        non-finite or non-positive, because neither can bound a command.  Small
        positive limits (including the writer's toorisky_v/w stamps) are usable
        and are handled by the soft map cost instead.
        """
        return (
            torch.isfinite(safe_v)
            & torch.isfinite(safe_w)
            & (safe_v > 0.0)
            & (safe_w > 0.0)
        )

    def _footprint_map_limits(
        self,
        states: torch.Tensor,
        movement_heading: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Lookup and reduce safe command limits over every footprint point."""
        x, y = self._footprint_sample_positions(states)
        heading = movement_heading[..., None].expand_as(x)
        point_v, point_w = self._lookup_cached_map(x, y, heading)
        point_valid = self._usable_limits(point_v, point_w)
        safe_v = self._power_mean_positive(
            point_v, self.config.too_risky_v,
            self.config.travmap_footprint_harmonic_power)
        safe_w = self._power_mean_positive(
            point_w, self.config.too_risky_w,
            self.config.travmap_footprint_harmonic_power)
        return safe_v, safe_w, torch.all(point_valid, dim=-1)

    def _centre_map_limits(
        self,
        states: torch.Tensor,
        movement_heading: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Lookup safe command limits at the body CENTRE cell only, no ring.

        Same channels, same flooring and the same usability test as
        ``_footprint_map_limits``; the only difference is the sampled SET -- one
        point instead of ``1 + footprint_samples``.  The footprint stencil is
        reserved for the terrain COST, so the command rectangle is decided by the
        cell the body centre actually occupies and a hazard beside the body is
        priced rather than throttling the command.

        ``states`` is ``(..., 5)`` and ``movement_heading`` is ``(...)``; all three
        returned tensors are ``(...)``.
        """
        # (...,) one sample per state: the map cell under the body centre.
        point_v, point_w = self._lookup_cached_map(
            states[..., 0], states[..., 1], movement_heading)
        point_valid = self._usable_limits(point_v, point_w)
        # The SAME floored reciprocal the footprint reduction applies, over a
        # one-element axis -- a one-point power mean IS that floor, at every order.  Keeping it
        # means a non-finite, non-positive or below-stamp cell still collapses to
        # the channel's too-risky stamp instead of widening the command domain,
        # which a raw lookup (+inf is the fold identity) would silently do.
        # Left at the default order deliberately: travmap_footprint_harmonic_power is the terrain
        # COST's knob, and over one point any other order would return the same value up
        # to float32 round-trip noise -- which the command domain has no reason to pay.
        safe_v = self._power_mean_positive(
            point_v[..., None], self.config.too_risky_v)
        safe_w = self._power_mean_positive(
            point_w[..., None], self.config.too_risky_w)
        return safe_v, safe_w, point_valid

    def _build_constraint_table(self) -> ConstraintTermTable:
        table = super()._build_constraint_table()
        table.add("Terrain", fn=self._constraint_Terrain)
        return table

    def _constraint_Terrain(
        self, state: torch.Tensor, data: MPPIInput,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ``(...)`` command ceilings the map cell under ``state`` allows.

        ``state`` is ``(..., 5)`` and both returned tensors are ``(...)``.  A pure
        TIGHTENER: the caller's mode ceiling is the seed MPPICore._command_limits folds
        this into, and the escape floor that keeps a too-risky cell from deadlocking the
        planner is applied there too.  Neither belongs to the terrain -- one is the mode's,
        the other is core's anti-deadlock policy -- and this term used to own both only
        because it was the one function that computed the rectangle.

        ``+inf`` when there is no map snapshot (direct helper use, or a planner that has
        not refreshed): the identity of the fold, so the caller's ceiling is left in place
        rather than a terrain constraint being invented.  solve() cannot reach that branch
        because _ensure_map_cache() runs first.
        """
        template = state[..., 0]
        if not self._map_snapshot_pinned():
            infinity = torch.full_like(template, float("inf"))
            return infinity, infinity

        movement_heading = state[..., 2]
        if data.reversed_geometry:
            movement_heading = self._wrap_torch(movement_heading + math.pi)
        safe_v, safe_w, _ = self._centre_map_limits(
            state, movement_heading)
        # Cells that cannot bound a command (NaN or negative) drop to zero, and so to the
        # escape floor _command_limits applies, instead of silently widening the domain.
        safe_v = torch.where(
            torch.isnan(safe_v) | (safe_v < 0.0),
            torch.zeros_like(safe_v), safe_v)
        safe_w = torch.where(
            torch.isnan(safe_w) | (safe_w < 0.0),
            torch.zeros_like(safe_w), safe_w)
        return safe_v, safe_w




####################################### TERRAIN COST #######################################

    def _cached_footprint_limits(
        self, ctx: CostContext,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(K,T)`` safe_v, safe_w and ``(K,T)`` usability, memoized for this ONE batch.

        Shared by _term_Travmap (the cost) and _validity_MapUsable (the veto): both need
        the same footprint reduction, and it is the dominant per-batch map cost, so running
        it twice per batch would be a real regression rather than a rounding error.  The
        memo lives on ctx.cache, which is per-evaluation, so the sample batch and the
        settled batch-of-one can never share one -- and the two managers must therefore be
        handed the SAME context, which MPPIProblem.score() is what guarantees.
        """
        cached = ctx.cache.get("footprint_limits")
        if cached is None:
            self._require_map_snapshot("_cached_footprint_limits")
            movement_heading = ctx.states[..., 2]
            if ctx.data.reversed_geometry:
                movement_heading = self._wrap_torch(movement_heading + math.pi)
            cached = self._footprint_map_limits(
                ctx.states, movement_heading)
            ctx.cache["footprint_limits"] = cached
        return cached

    def _build_cost_table(self) -> CostTermTable:
        table = super()._build_cost_table()
        # The registered callable is the term OBJECT, not the bound method: it publishes
        # the per-knot map limits through debug_line(), which CostTermTable.debug_lines()
        # finds by attribute lookup and a bound method does not carry.
        table.add("Travmap", self.config.weight_Travmap, fn=_TravmapTerm(self))
        # Per-metre terrain share; skipped at weight 0 so the historical objective is unchanged.
        if float(self.config.weight_TravmapDistance) > 0.0:
            table.add("TravmapDistance", self.config.weight_TravmapDistance,
                      fn=self._term_TravmapDistance)
        return table

    def _build_validity_table(self) -> ValidityTermTable:
        table = super()._build_validity_table()
        table.add("MapUsable", fn=self._validity_MapUsable)
        return table

    def _term_Travmap(self, ctx: CostContext) -> torch.Tensor:
        """Return ``(K,)`` UNWEIGHTED mean terrain penalty.

        Map values are safe command targets, not physical actual-velocity
        limits. States have shape ``(K,T,5)``. The cost is the terrain-only
        traversability penalty ``1/safe_v + 1/safe_w`` (inverse command units),
        time-averaged before ``weight_Travmap``, which the manager applies:
        staying where the map allows only small commands is expensive regardless
        of what is commanded there, so rollouts are steered away from risky cells
        instead of merely being penalized for exceeding their limits.  Out-of-extent
        and too-risky cells are not rejected: only limits that cannot constrain a
        command (non-finite or non-positive) are reported, and that rejection is now
        _validity_MapUsable's, not this term's.
        """
        # (K,T) footprint-reduced safe limits; the usability channel is the veto's.
        safe_v, safe_w, _ = self._cached_footprint_limits(ctx)
        # (K,T) per-step terrain cost: s/m from the linear channel bounded by
        # 1/too_risky_v, plus s/rad from the angular channel bounded by 1/too_risky_w.
        inverse_limits = (
            self._inverse_limit(safe_v, self.config.too_risky_v)
            + self._inverse_limit(safe_w, self.config.too_risky_w))
        return torch.mean(inverse_limits, dim=-1)

    def _term_TravmapDistance(self, ctx: CostContext) -> torch.Tensor:
        """Return ``(K,)`` UNWEIGHTED mean terrain penalty per distance driven.

        Per knot: ``max(0, 1/safe_v + 1/safe_w - c_free) * ds / (conservative_speed_v * mppi_dt)``
        with ``c_free = 1/maximum_cmd_v + 1/maximum_cmd_w`` and ``ds`` the metres that knot
        drives.  A knot that does not translate (an in-place pivot) pays nothing, and one
        driven at the conservative speed -- the domain cap once the body centre is in a
        too-risky cell -- pays Travmap's own per-knot value less the free baseline.
        """
        states = ctx.states
        if states.ndim < 2 or states.shape[-1] < 2:
            raise ValueError(
                f"_term_TravmapDistance needs (..., T, 5) rollout states, got shape "
                f"{tuple(states.shape)}")
        # (K,T) footprint-reduced safe limits; a cache hit after Travmap, so no second map lookup.
        safe_v, safe_w, _ = self._cached_footprint_limits(ctx)
        # (K,T) Travmap's per-knot cost above the free-terrain baseline, so free ground costs 0.
        free_baseline = (1.0 / float(self.config.maximum_cmd_v)
                         + 1.0 / float(self.config.maximum_cmd_w))
        excess = torch.clamp(
            self._inverse_limit(safe_v, self.config.too_risky_v)
            + self._inverse_limit(safe_w, self.config.too_risky_w)
            - free_baseline, min=0.0)
        # (K,T,2) knot positions and (K,T,2) the position entering each knot; knot 0 enters
        # from data.pose, which is the rollout's own initial state (MPPISolver).
        positions = states[..., :2]
        origin = torch.as_tensor(
            np.asarray(ctx.data.pose[:2], dtype=np.float32),
            device=positions.device, dtype=positions.dtype)
        previous = torch.cat(
            (origin.expand(positions.shape[:-2] + (1, 2)), positions[..., :-1, :]), dim=-2)
        # (K,T) metres driven per knot, in units of the conservative step (validated > 0).
        step_ratio = torch.linalg.norm(positions - previous, dim=-1) / (
            float(self.config.conservative_speed_v) * float(self.config.mppi_dt))
        return torch.mean(excess * step_ratio, dim=-1)

    def _validity_MapUsable(self, ctx: CostContext) -> torch.Tensor:
        """``(K,T)`` True where every footprint point of the knot can bound a command.

        Independent of weight_Travmap BY CONSTRUCTION now, rather than by the accident of
        how one function unpacked a tuple: this lives in a registry that has no weights at
        all, so a zero-weighted map cost still vetoes an unusable cell.
        """
        _, _, valid_map = self._cached_footprint_limits(ctx)
        return valid_map




####################################### MAP QUERIES #######################################

    def travgraph_cmd_limits(
        self,
        x: float,
        y: float,
        heading: float,
        lookahead_distance: float,
        sampling_dist: float,
        lookahead_angle: float,
    ) -> tuple[float, float]:
        """CMDbasedMap.get_cmd_limits() answered from the travgraph instead of the dense map.

        Display only: main.py hands it to set_cmd_limits() for the HUD.  Same samples -- every
        ``sampling_dist`` metres out to ``lookahead_distance`` along ``heading`` and ``heading +-
        lookahead_angle`` (rad) -- the same int() cell / nearest-layer rules and the same plain
        mean.  Returns the too-risky pair when no snapshot can be pinned (the tick that follows
        reports why) and the actuator maxima when every sample is off the map, as the dense
        query does.  Runs on the CPU tables, so it costs no device round trip.
        """
        if self._travgraph is None:
            raise RuntimeError(
                "travgraph_cmd_limits() needs a map reader: this planner was built with "
                "global_map=None, so it has no terrain source")
        try:
            # (x, y) is the robot pose (main.py), so the read window follows the robot here too.
            self._ensure_map_cache((x, y))
        except (RuntimeError, ValueError):
            # Same answer as the dense query before its map exists; the planner tick re-raises
            # the identical error through _prepare_problem, which is where it is reported.
            return float(self.config.too_risky_v), float(self.config.too_risky_w)
        if not self._map_snapshot_pinned():
            return float(self.config.too_risky_v), float(self.config.too_risky_w)
        _, xmax, _, ymax, resolution, rows, cols, layers = self._map_geometry
        count = int(float(lookahead_distance) / float(sampling_dist))
        # (3,) ray headings wrapped to [-pi, pi); (count,) sample distances along each ray.
        theta = np.remainder(
            float(heading) + np.array([-1.0, 0.0, 1.0]) * float(lookahead_angle) + math.pi,
            _TWO_PI) - math.pi
        distance = np.arange(count, dtype=np.float64) * float(sampling_dist)
        # (count, 3) sample positions and their int()-truncated cells, as get_cmd_limits indexes.
        px = float(x) + distance[:, None] * np.cos(theta)[None, :]
        py = float(y) + distance[:, None] * np.sin(theta)[None, :]
        row = np.trunc((float(xmax) - px) / float(resolution)).astype(np.int64)
        col = np.trunc((float(ymax) - py) / float(resolution)).astype(np.int64)
        layer = np.broadcast_to(np.remainder(np.floor(
            np.remainder(theta + math.pi, _TWO_PI) / (_TWO_PI / layers) + 0.5), layers).astype(np.int64),
            row.shape)
        inside = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
        if not inside.any():
            return float(self.config.maximum_cmd_v), float(self.config.maximum_cmd_w)
        values = self._travgraph.gather_numpy(row[inside], col[inside], layer[inside])  # (n, 2)
        return float(np.mean(values[:, 0])), float(np.mean(values[:, 1]))




####################################### VISUALIZATION SNAPSHOT #######################################
# Read-only copies of the cached map for plotting.  Never on the control path.

    def _ensure_trav_layer_snapshot(self, heading: float) -> Optional[int]:
        """Cache the cmd_v channel of the heading layer nearest ``heading`` for plotting.

        Returns the layer index that was cached, or None when no layer can be resolved.  The
        (rows, cols) copy is taken only when the (map version, layer) key changes, so a
        stationary robot re-uses the previous copy and a rotating one pays a single
        device-to-host copy per heading bin it enters.
        """
        layer = self._heading_layer_index(heading)
        if layer is None or not self._map_snapshot_pinned() or self._map_version is None:
            self._trav_layer_snapshot = None
            self._trav_layer_key = None
            return None
        key = (int(self._map_version), int(layer))
        if self._trav_layer_snapshot is not None and self._trav_layer_key == key:
            return layer
        # (rows, cols) m/s - per-cell safe linear-velocity limit at this heading layer.
        self._trav_layer_snapshot = self._travgraph.layer_image(
            layer, int(self._map_geometry[5]), int(self._map_geometry[6]))
        self._trav_layer_key = key
        return layer

    def _ensure_trav_clim(self) -> Optional[tuple[float, float]]:
        """Return plot colour limits over every heading layer, cached per map version.

        Scaling on the displayed layer alone would make the background colours jump whenever
        the robot rotates into another bin, so the range is taken over the whole cmd_v channel.
        """
        if not self._map_snapshot_pinned():
            return None
        if self._trav_clim is not None and self._trav_clim_version == self._map_version:
            return self._trav_clim
        # Interpolation is convex over the lattice nodes, so their range is the map's.
        value_range = self._travgraph.value_range()
        if value_range is None:
            self._trav_clim = None
        else:
            low, high = value_range
            # A uniform map (every cell identical) would give a degenerate range.
            self._trav_clim = (low, high) if high > low else (low, low + 1.0e-6)
        self._trav_clim_version = self._map_version
        return self._trav_clim

    def visualization_snapshot(self, heading: float = 0.0) -> Optional[dict[str, Any]]:
        """Return the plot background for ``heading`` without exposing mutable shared arrays.

        ``trav`` is the (rows, cols) cmd_v channel of the heading layer nearest ``heading`` in
        m/s, ``layer`` is that layer index and ``clim`` the colour range over all layers.
        Callers cache on (``version``, ``layer``): the map version alone is not enough because
        the background follows the robot heading.
        """
        if self.global_map is None:
            return None
        self._ensure_map_cache()
        self._warn_layer_convention_once()
        layer = self._ensure_trav_layer_snapshot(heading)
        return {
            "version": self._map_version,
            "layer": layer,
            "trav": self._trav_layer_snapshot,
            "clim": self._ensure_trav_clim(),
            "geometry": self._map_geometry,
        }




####################################### CUDA OUT-OF-MEMORY FALLBACK #######################################
# Only the map-shaped part of the migration lives here; MPPICore._fallback_to_cpu owns
# the device switch, the generators and the sample count, and calls the hook below.

    def _migrate_device_caches(self, target: torch.device) -> Optional[str]:
        """Move this layer's device tensors to ``target``; see MPPICore for the contract.

        Two caches, both rebuilt rather than copied: the travgraph tables, re-read from the
        graph on the next pin (O(nodes) values plus geometry tables, cheaper and safer than a
        copy off an exhausted GPU), and the footprint stencil.  Nothing is ever dropped
        without a rebuild, so the note from the lower layers is passed through unchanged.
        """
        note = super()._migrate_device_caches(target)
        if self._travgraph is not None:
            self._travgraph.drop_device_tensors()
        self._map_version = None
        # Rebuilt rather than copied: the stencil is (1 + footprint_samples, 2) and is a
        # pure function of config, so a device-to-host copy would buy nothing.  It reads
        # self.device, which _fallback_to_cpu() has already switched to ``target``.
        self._footprint_offsets = self._build_footprint_offsets()
        return note
