"""Travgraph terrain source: the MPPI's runtime view of the world model's lattice graph.

The world model keeps two representations of the same traversability estimate: the dense
TraversabilityMap (fine cells, map_resolution) and the shared-memory travgraph (one node per
coarse estimation-lattice point, trav_estimation_resoultion apart, carrying per-heading-layer
cmd_v / cmd_w).  The dense map is the lattice values sigmoid-interpolated onto the fine grid
(CMDbasedMap._compute_travmap, "Interpolation").  This module rebuilds exactly that
interpolation AT RUNTIME from the graph alone, so the planner no longer reads the dense map.

How a fine cell (row, col) resolves, reproducing the writer:
  * row/col are the planner's own floor() cell indices, unchanged.
  * Each axis picks its ANCHOR lattice node: the nearest node at or above the cell in world
    coordinate (smallest fine index <= the cell's), and the next node below it.  The writer
    paints each quadrant as an (N+1)x(N+1) patch, N = floor(R / map_resolution); when R / res is
    not an integer (0.5 / 0.04 = 12.5) adjacent patches overlap by one fine row/col and the dense
    map keeps whichever was written LAST.  Within one estimation batch the writer walks the
    lattice in ascending x then y, so the higher-x / higher-y corner's patch wins -- this module
    anchors there.  Over 214 recorded snapshots it reproduced 94 % of the dense cell values
    exactly (88-97 % per scene); 5 % are later batches having overwritten an overlap row in the
    other order (<= one sigmoid step apart), 0.4 % are fine-resolution stamps that a 0.5 m
    lattice cannot represent.
  * Weight toward the lower neighbour along an axis is w[i], i = fine offset from the anchor and
    w the writer's normalized sigmoid over linspace(0, 1, N + 1): bilinear in (w_x, w_y).

Cost structure (what keeps the per-query path cheap):
  * per fine cell, geometry only, built once: the anchor square index and the bilinear basis
    (w_x, w_y, w_x * w_y)                                     -> 2 gathers per query
  * per graph version: the lattice values re-expressed as bilinear coefficients
    (c0, c1, c2, c3) per (square, layer), so a query is c0 + basis . (c1, c2, c3)
                                                               -> 1 gather + 3 elementwise ops
  * the flat cell index is one fused add and the (square, layer) pair one 2-D gather, so a
    query is 7 kernel launches on top of the dense path's row/col/layer arithmetic
  * per tick: only graph_seq is read; the payload copy happens on a version change and is
    O(nodes * layers), not O(fine cells * layers) as the dense snapshot was.
  * rasterize (default): the interpolation above runs ONCE per graph version over every fine
    cell and layer on the device, and a query is then the dense path's single gather.  The
    writer changes the graph ~2x per second while a tick issues ~40 lookups, and on a GPU shared
    with the simulator and perception every extra small kernel queues: per-query evaluation
    measured +15 % mean solve time in closed loop, the per-version grid none.  Costs the
    (rows, cols, layers, 2) float32 grid the dense snapshot used to occupy (7.8 MB at 350^2 x 8).
    The grid is rebuilt one heading layer at a time into its own buffer, so a rebuild's
    transient device memory is ~1/layers of a one-shot evaluation (908 MB -> ~1/8 on a 50 m map).
  * per graph version, only what CHANGED is redone: the nodes whose values differ from the
    pinned version select the <= 4 lattice squares each one is a corner of; only those squares'
    coefficients are recomputed and uploaded and only their fine cells re-interpolated.  The
    lattice layout (slot -> lattice index, its validation) is redone only when node positions
    change.  The graph copy and the change detection stay O(nodes): the reader offers a full
    snapshot and no change log.  A full rebuild is kept for the first pin, a geometry or device
    change, and recovery after a refresh that failed part-way.
  * window (window_radius finite): only the lattice nodes within window_radius (+ one lattice
    step) of the robot are read, found through the reader's bucket index (bucket_size == R on
    the writer's grid, so only the window's buckets are touched) and copied under the graph's
    seqlock.  The read and the change detection are then O(window), and nodes may be inserted or
    deleted.  A lattice point without an active node (a hole) and every point outside the window
    hold NaN, which the interpolation spreads to the fine cells of the <= 4 squares it is a
    corner of; TerrainMPPI._usable_limits() hard-vetoes those non-finite limits.
"""

from __future__ import annotations

import math
import time
from typing import Any, Optional

import numpy as np
import torch


# Payload names of the shared GRAPH_LAYOUT (statenav_global.core.graph_layout).
_CMD_V = "cmd_v"
_CMD_W = "cmd_w"
# Tolerance (m) for "a node's stored float32 coordinate sits on the lattice".  The writer's own
# check is 8 * float32 eps * |coord| (~1e-5 m on a 15 m map); this only rejects a graph that is
# not the lattice at all, so it is set well above float noise and far below any real offset.
_LATTICE_COORD_TOL_M = 1.0e-3
# Seqlock retry budget of the windowed read and the backoff between attempts (the values the
# reader's own snapshot() uses), so a writer batch in progress is waited out, not spun on.
_WINDOW_MAX_RETRIES = 50
_WINDOW_RETRY_SLEEP_S = 5.0e-4
# Reader attributes the windowed read uses (SharedGraphReader's read-only shared-memory views).
_WINDOW_READER_API = ("graph_seq", "n_max", "bounds", "bucket_size", "bucket_head",
                      "node_next_in_bucket", "node_x", "node_y", "node_active", "node_payload")


def sigmoid_weights(segments: int, sharpness: float) -> np.ndarray:
    """``(segments + 1,)`` float64 normalized sigmoid, verbatim CMDbasedMap._compute_travmap.

    w[0] == 0, w[-1] == 1 and w[i] + w[segments - i] == 1, so the interpolation is symmetric.
    """
    t = np.linspace(0.0, 1.0, segments + 1)
    sig = 1.0 / (1.0 + np.exp(-sharpness * (t - 0.5)))
    return (sig - sig[0]) / (sig[-1] - sig[0])


def fine_index(coord_max: float, coords: np.ndarray, resolution: float, size: int) -> np.ndarray:
    """Fine-grid index of world coordinates with CMDbasedMap.xy2grid's exact arithmetic.

    ``int()`` truncation of a float64 quotient, with index == size clamped to size - 1 (the
    writer's own convention for a point on the far map edge).  ``coords`` is ``(n,)``.
    """
    index = np.array([int((coord_max - float(c)) / resolution) for c in coords], dtype=np.int64)
    index[index == size] = size - 1
    return index


def axis_anchor_table(
    node_fine: np.ndarray, size: int, segments: int, weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per fine index along one axis: ``(anchor lattice index, weight toward lower node)``.

    ``node_fine`` is ``(n,)`` the fine index of lattice nodes in ASCENDING world coordinate,
    hence non-increasing.  The anchor is the node with the smallest lattice index whose fine
    index is <= the cell's (the nearest node at or above the cell); the lower neighbour is
    anchor - 1.  Cells past the last node on either side clamp onto it with weight 0.
    Both returned arrays are ``(size,)``.
    """
    fine = np.arange(size, dtype=np.int64)
    # (size,) count of nodes strictly below the cell in fine index == first node at/above it.
    first = np.sum(node_fine[None, :] > fine[:, None], axis=1)
    beyond_top = first >= node_fine.size
    anchor = np.where(beyond_top, node_fine.size - 1, first)
    offset = np.where(beyond_top, 0, fine - node_fine[anchor])
    weight = weights[np.minimum(offset, segments)]
    # No lower neighbour (anchor 0): the coefficient table carries a zero slope there anyway.
    weight = np.where(anchor == 0, 0.0, weight)
    return anchor.astype(np.int64), weight


class _LatticeSnapshot:
    """The slice of the GraphSnapshot API TravGraphTerrain reads, over plain arrays."""

    def __init__(self, graph_seq: int, x: np.ndarray, y: np.ndarray,
                 payloads: dict[str, np.ndarray]) -> None:
        self.graph_seq = graph_seq
        self.node_hwm = int(x.size)
        self.node_x = x                                         # (n,) float32 m
        self.node_y = y                                         # (n,) float32 m
        self.node_active = np.ones(x.size, dtype=np.uint8)      # (n,) every node active
        self._payloads = payloads                               # name -> (n, layers) float32

    def node_payload(self, name: str) -> np.ndarray:
        return self._payloads[name]


class InMemoryTravGraphReader:
    """SharedGraphReader stand-in over an in-memory lattice, for maps without a world model.

    Implements the reader protocol TravGraphTerrain uses (``graph_seq``, ``snapshot(into=)``,
    and the bucket-index view of _WINDOW_READER_API for the windowed read)
    so the demo's flat map and the tests plan on the same code path as the robot.  ``values``
    is ``(nx, ny, layers, 2)`` float32 (cmd_v, cmd_w) of the lattice node at world
    ``((kx0 + i) * resolution, (ky0 + j) * resolution)``.  set_values() publishes a new
    version, as one writer edit batch would.
    """

    def __init__(self, values: np.ndarray, kx0: int, ky0: int, resolution: float) -> None:
        self.graph_seq = 2
        self._snapshot: Optional[_LatticeSnapshot] = None
        self.set_values(values, kx0, ky0, resolution)

    @classmethod
    def uniform(cls, bounds: tuple[float, float, float, float], resolution: float,
                layers: int, cmd_v: float, cmd_w: float) -> "InMemoryTravGraphReader":
        """Every lattice point inside ``bounds`` (xmin, xmax, ymin, ymax) at one command pair."""
        xmin, xmax, ymin, ymax = (float(b) for b in bounds)
        kx0, kx1 = math.ceil(xmin / resolution - 1e-9), math.floor(xmax / resolution + 1e-9)
        ky0, ky1 = math.ceil(ymin / resolution - 1e-9), math.floor(ymax / resolution + 1e-9)
        if kx1 < kx0 or ky1 < ky0:
            raise ValueError(
                f"bounds {bounds} contain no {resolution} m lattice point; the travgraph "
                "terrain needs at least one node")
        values = np.empty((kx1 - kx0 + 1, ky1 - ky0 + 1, int(layers), 2), dtype=np.float32)
        values[..., 0] = cmd_v
        values[..., 1] = cmd_w
        return cls(values, kx0, ky0, resolution)

    def set_values(self, values: np.ndarray, kx0: int, ky0: int, resolution: float) -> None:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 4 or values.shape[-1] != 2:
            raise ValueError(f"lattice values must be (nx, ny, layers, 2), got {values.shape}")
        nx, ny = values.shape[:2]
        ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        x = ((kx0 + ii.reshape(-1)) * float(resolution)).astype(np.float32)
        y = ((ky0 + jj.reshape(-1)) * float(resolution)).astype(np.float32)
        self.graph_seq += 2
        self._snapshot = _LatticeSnapshot(self.graph_seq, x, y, {
            "cmd_v": values[..., 0].reshape(nx * ny, -1).copy(),
            "cmd_w": values[..., 1].reshape(nx * ny, -1).copy()})
        # Bucket-index view laid out as the writer's create_grid lattice: bucket_size == resolution
        # and bounds half a step outside the outer nodes, so bucket (i, j) holds only slot i*ny + j.
        self.n_max = nx * ny
        self.node_x, self.node_y = self._snapshot.node_x, self._snapshot.node_y
        self.node_active = self._snapshot.node_active
        self.bucket_size = float(resolution)
        self.bounds = ((kx0 - 0.5) * resolution, (kx0 + nx - 0.5) * resolution,
                       (ky0 - 0.5) * resolution, (ky0 + ny - 0.5) * resolution)
        self.bucket_head = np.arange(nx * ny, dtype=np.int32).reshape(nx, ny)   # (nx, ny) slot
        self.node_next_in_bucket = np.full(nx * ny, -1, dtype=np.int32)         # (n,) no chains

    def node_payload(self, name: str) -> np.ndarray:
        return self._snapshot.node_payload(name)

    def snapshot(self, into: Any = None, max_retries: int = 50) -> _LatticeSnapshot:
        return self._snapshot

    def close(self) -> None:
        return None


class TravGraphTerrain:
    """Version-cached travgraph lattice plus the sigmoid-interpolation tables.

    Owns its SharedGraphReader (created lazily on first use, like GlobalAStar) unless a reader
    is injected, which is how tests supply a graph without shared memory.  The reader needs only
    ``graph_seq`` and ``snapshot(into=...)``; the snapshot only ``graph_seq``, ``node_hwm``,
    ``node_x``, ``node_y``, ``node_active`` and ``node_payload(name)``.  With a finite
    ``window_radius`` the reader is read directly instead, through _WINDOW_READER_API.
    """

    def __init__(
        self,
        resolution: float,
        sigmoid_k: float,
        reader: Any = None,
        attach_timeout: float = 10.0,
        rasterize: bool = True,
        window_radius: float = math.inf,
    ) -> None:
        if not math.isfinite(float(resolution)) or float(resolution) <= 0.0:
            raise ValueError(f"travgraph lattice resolution must be finite and > 0, got {resolution!r}")
        if not math.isfinite(float(sigmoid_k)) or float(sigmoid_k) <= 0.0:
            raise ValueError(f"travgraph sigmoid sharpness must be finite and > 0, got {sigmoid_k!r}")
        if math.isnan(float(window_radius)) or float(window_radius) <= 0.0:
            raise ValueError(
                f"travgraph window radius must be > 0 m (inf reads the whole graph), got {window_radius!r}")
        self.resolution = float(resolution)   # R (m): lattice spacing the writer estimates on
        self.sigmoid_k = float(sigmoid_k)     # interp_sigmoid_k of the writer's S-curve
        self._reader = reader
        self._attach_timeout = float(attach_timeout)
        self.rasterize = bool(rasterize)      # per-version full-grid evaluation (see module doc)
        # (m) half side of the square read around the robot; inf = whole graph via snapshot().
        self.window_radius = float(window_radius)
        self._center: Optional[tuple[float, float]] = None    # world (x, y) the window follows
        # (bounds, (kx0, ky0, nx, ny)): the R-lattice inside the graph's bounds, per bounds.
        self._frame: Optional[tuple[tuple, tuple[int, int, int, int]]] = None
        # (i0, i1, j0, j1) half-open lattice box of the pinned version; None on the snapshot path.
        self._window_box: Optional[tuple[int, int, int, int]] = None
        self._window_holes = 0                # holes in the last window read, reported on change
        self._snapshot: Any = None            # reused GraphSnapshot buffer (reader.snapshot(into=))
        self.version: Optional[int] = None    # graph_seq of the pinned snapshot
        self._cols = 0                        # fine-grid column count the cell tables index with
        self._cell_key: Optional[tuple] = None
        # CPU (NumPy) tables: the upload source, the HUD query and the plot background.
        self._np_cell_square: Optional[np.ndarray] = None  # (rows*cols,) int64 anchor square index
        self._np_cell_basis: Optional[np.ndarray] = None   # (rows*cols, 3) float32 (wx, wy, wx*wy)
        self._np_coeff: Optional[np.ndarray] = None        # (nx*ny, layers, 4, 2) float32 (c0..c3) x (v, w)
        self._np_lattice: Optional[np.ndarray] = None      # (nx, ny, layers, 2) float32 node values
        # Node layout of the last snapshot: copies of its (hwm,) x / y / active arrays and what
        # they resolve to, so validation and slot -> lattice indexing rerun only when they change.
        self._layout: Optional[dict[str, Any]] = None
        # Device copies of the same three tables; None until refresh() pins a snapshot.
        self._device: Optional[torch.device] = None
        self._cell_square: Optional[torch.Tensor] = None
        self._cell_basis: Optional[torch.Tensor] = None
        self._coeff: Optional[torch.Tensor] = None
        # (rows, cols, layers, 2) interpolated (cmd_v, cmd_w) of the pinned version, rasterize only.
        self._grid: Optional[torch.Tensor] = None

    @property
    def is_ready(self) -> bool:
        """True once a snapshot is pinned and its tables are on the device."""
        return (self._coeff is not None and self._cell_square is not None
                and self._cell_basis is not None)

    # ------------------------------------------------------------------ reader / snapshot

    def _ensure_reader(self) -> Any:
        if self._reader is None:
            # Lazy import, as GlobalAStar does: keeps the MPPI importable without state_nav, and
            # the writer's graph blocks need not exist when the planner is constructed.
            # statenav_global.core only (numpy + omegaconf): no world-model / ROS stack is loaded.
            from statenav_global.core.graph_layout import GRAPH_LAYOUT
            from statenav_global.core.shared_graph import (
                SharedGraphError, SharedGraphReader)
            try:
                self._reader = SharedGraphReader(GRAPH_LAYOUT, attach_timeout=self._attach_timeout)
            except SharedGraphError as exc:
                raise RuntimeError(
                    "travgraph terrain: could not attach to the shared graph -- is the world "
                    "model running with global_planner_type == 'Astar' (the only mode that "
                    f"builds it)? ({exc})") from exc
        return self._reader

    def current_version(self) -> int:
        """The writer's live graph_seq: one shared-memory read, no copy."""
        return int(self._ensure_reader().graph_seq)

    def _read_lattice(self, layers: int) -> tuple[int, np.ndarray, int, int]:
        """Snapshot the graph and regrid it onto the lattice.

        Returns ``(graph_seq, values, kx0, ky0)`` with ``values`` the ``(nx, ny, layers, 2)``
        float32 (cmd_v, cmd_w) of every lattice node, x/y index ascending in world coordinate,
        and ``kx0 / ky0`` the integer lattice index of values[0, 0] (world = k * resolution).
        """
        reader = self._ensure_reader()
        self._snapshot = reader.snapshot(into=self._snapshot)
        snap = self._snapshot
        hwm = int(snap.node_hwm)
        # (hwm,) node core arrays; views into the snapshot buffer, which the next snapshot reuses.
        raw_x = np.asarray(snap.node_x[:hwm])
        raw_y = np.asarray(snap.node_y[:hwm])
        raw_active = np.asarray(snap.node_active[:hwm])
        layout = self._layout
        if not (layout is not None and np.array_equal(layout["x"], raw_x)
                and np.array_equal(layout["y"], raw_y)
                and np.array_equal(layout["active"], raw_active)):
            layout = self._lattice_layout(raw_x, raw_y, raw_active)
            self._layout = layout
        active = layout["active"] == 1
        cmd_v = np.asarray(snap.node_payload(_CMD_V)[:hwm], dtype=np.float32)[active]  # (n, L)
        cmd_w = np.asarray(snap.node_payload(_CMD_W)[:hwm], dtype=np.float32)[active]  # (n, L)
        if cmd_v.ndim != 2 or cmd_v.shape[1] != layers or cmd_w.shape != cmd_v.shape:
            raise RuntimeError(
                f"travgraph terrain: node payloads cmd_v{cmd_v.shape}/cmd_w{cmd_w.shape} do not "
                f"carry the map's {layers} heading layers")
        nx, ny = layout["shape"]
        values = np.empty((nx, ny, layers, 2), dtype=np.float32)
        values[layout["i"], layout["j"], :, 0] = cmd_v
        values[layout["i"], layout["j"], :, 1] = cmd_w
        return int(snap.graph_seq), values, layout["kx0"], layout["ky0"]

    def _lattice_layout(self, raw_x: np.ndarray, raw_y: np.ndarray,
                        raw_active: np.ndarray) -> dict[str, Any]:
        """Resolve and validate where each active node sits on the lattice.

        ``raw_x / raw_y / raw_active`` are the snapshot's ``(hwm,)`` node arrays.  Returns their
        copies (the cache key), the ``(n,)`` lattice indices i / j of the active nodes in slot
        order, the lattice origin kx0 / ky0 and the lattice shape (nx, ny).
        """
        active = raw_active == 1
        x = raw_x.astype(np.float64)[active]                                 # (n,) m
        y = raw_y.astype(np.float64)[active]                                 # (n,) m
        if x.size == 0:
            raise RuntimeError("travgraph terrain: the shared graph has no active node")
        resolution = self.resolution
        kx = np.rint(x / resolution).astype(np.int64)
        ky = np.rint(y / resolution).astype(np.int64)
        off_lattice = (np.abs(kx * resolution - x) > _LATTICE_COORD_TOL_M) | (
            np.abs(ky * resolution - y) > _LATTICE_COORD_TOL_M)
        if off_lattice.any():
            bad = int(np.argmax(off_lattice))
            raise RuntimeError(
                f"travgraph terrain: {int(off_lattice.sum())} node(s) are off the "
                f"{resolution} m lattice, e.g. ({x[bad]:.4f}, {y[bad]:.4f}); the planner's "
                "trav_estimation_resoultion must match the world model's")
        kx0, ky0 = int(kx.min()), int(ky.min())
        nx, ny = int(kx.max()) - kx0 + 1, int(ky.max()) - ky0 + 1
        flat = (kx - kx0) * ny + (ky - ky0)
        if nx * ny != x.size or np.unique(flat).size != x.size:
            raise RuntimeError(
                f"travgraph terrain: {x.size} active nodes do not tile a full {nx}x{ny} lattice "
                "(missing or duplicate lattice points); this module needs the writer's grid graph")
        return {"x": raw_x.copy(), "y": raw_y.copy(), "active": raw_active.copy(),
                "i": kx - kx0, "j": ky - ky0, "kx0": kx0, "ky0": ky0, "shape": (nx, ny)}

    # ------------------------------------------------------------------ windowed read

    def _lattice_frame(self) -> tuple[int, int, int, int]:
        """``(kx0, ky0, nx, ny)``: every R-lattice point inside the graph's bounds.

        The writer's grid stores bounds half a lattice step outside its outer nodes, so this is
        exactly its node lattice; insert_node() rejects positions outside the bounds, so no node
        can lie outside it.  Same +-1e-9 index nudges as the world model's lattice index rule.
        """
        reader = self._ensure_reader()
        if self._frame is None:
            missing = [name for name in _WINDOW_READER_API if not hasattr(reader, name)]
            if missing:
                raise RuntimeError(
                    f"travgraph window read: reader {type(reader).__name__} lacks {missing}; set "
                    "mppi_travgraph_window_radius to .inf to read it through snapshot() instead")
        bounds = tuple(float(b) for b in reader.bounds)
        if self._frame is None or self._frame[0] != bounds:
            xmin, xmax, ymin, ymax = bounds
            res = self.resolution
            kx0, kx1 = math.ceil(xmin / res - 1e-9), math.floor(xmax / res + 1e-9)
            ky0, ky1 = math.ceil(ymin / res - 1e-9), math.floor(ymax / res + 1e-9)
            if kx1 < kx0 or ky1 < ky0:
                raise RuntimeError(
                    f"travgraph window read: graph bounds x[{xmin}, {xmax}] y[{ymin}, {ymax}] "
                    f"contain no {res} m lattice point")
            self._frame = (bounds, (kx0, ky0, kx1 - kx0 + 1, ky1 - ky0 + 1))
            # A new frame re-indexes the lattice: the next read starts from all-NaN.
            self._window_box = None
        return self._frame[1]

    def _window_box_for(self, center: Optional[Any]) -> Optional[tuple[int, int, int, int]]:
        """Half-open lattice index box ``(i0, i1, j0, j1)`` the next read covers, or None.

        None when window_radius is infinite (the whole-graph snapshot path).  Otherwise the
        nodes within window_radius of the center widened by one lattice step, so every fine cell
        within window_radius has all four corners of its square read; the whole lattice before
        any center is known.  ``center`` (world x, y, ...) replaces the stored one when given.
        """
        if not math.isfinite(self.window_radius):
            return None
        if center is not None:
            cx, cy = float(center[0]), float(center[1])
            if not (math.isfinite(cx) and math.isfinite(cy)):
                raise ValueError(f"travgraph window: non-finite robot position ({cx}, {cy})")
            self._center = (cx, cy)
        kx0, ky0, nx, ny = self._lattice_frame()
        if self._center is None:
            return (0, nx, 0, ny)
        cx, cy = self._center
        res, radius = self.resolution, self.window_radius
        # Lattice indices of the nodes within radius plus one node each side, clamped to the frame.
        i0 = min(max(math.ceil((cx - radius) / res) - 1 - kx0, 0), nx)
        i1 = min(max(math.floor((cx + radius) / res) + 2 - kx0, 0), nx)
        j0 = min(max(math.ceil((cy - radius) / res) - 1 - ky0, 0), ny)
        j1 = min(max(math.floor((cy + radius) / res) + 2 - ky0, 0), ny)
        return (i0, i1, j0, j1)

    def _read_window(
        self, layers: int, box: tuple[int, int, int, int],
    ) -> tuple[int, np.ndarray, int, int, tuple[int, int, int, int]]:
        """Read the lattice nodes inside ``box`` through the reader's bucket index.

        Returns ``(graph_seq, values, kx0, ky0, span)``: ``values`` is the ``(nx, ny, layers, 2)``
        float32 lattice holding the box's nodes and NaN everywhere else -- holes (lattice points
        without an active node) and the points outside the box -- and ``span`` the
        ``(i0, i1, j0, j1)`` index range outside which ``values`` equals the pinned lattice.
        Only the box's buckets and their chains are touched, and the copy is kept only when the
        graph's seqlock reads the same even graph_seq before and after it.
        """
        reader = self._ensure_reader()
        kx0, ky0, nx, ny = self._lattice_frame()
        i0, i1, j0, j1 = box
        res = self.resolution
        # World rectangle of the box: half a lattice step around its outer nodes.
        x_lo, x_hi = (kx0 + i0 - 0.5) * res, (kx0 + i1 - 0.5) * res
        y_lo, y_hi = (ky0 + j0 - 0.5) * res, (ky0 + j1 - 0.5) * res
        # Bucket index range the rectangle overlaps, clamped like the writer's _bucket_of().
        bxmin, _, bymin, _ = (float(b) for b in reader.bounds)
        bucket = float(reader.bucket_size)
        n_bx, n_by = reader.bucket_head.shape
        bx0 = min(max(math.floor((x_lo - bxmin) / bucket), 0), n_bx)
        bx1 = min(max(math.ceil((x_hi - bxmin) / bucket), 0), n_bx)
        by0 = min(max(math.floor((y_lo - bymin) / bucket), 0), n_by)
        by1 = min(max(math.ceil((y_hi - bymin) / bucket), 0), n_by)
        if i0 >= i1 or j0 >= j1:
            bx1, by1 = bx0, by0
            print(f"[travgraph] the robot at {self._center} is more than {self.window_radius:g} m "
                  "outside the graph lattice: no node is read and every cell is hard-vetoed")

        n_max = int(reader.n_max)
        next_in_bucket = reader.node_next_in_bucket
        # The reader's fenced seqlock load when it has one (required off x86-64), else graph_seq.
        read_seq = getattr(reader, "_read_seq", lambda: reader.graph_seq)
        seq = None
        for _ in range(_WINDOW_MAX_RETRIES):
            seq = int(read_seq())
            if seq < 4 or seq % 2 != 0:
                # Writer mid-batch, or no complete graph published yet (graph_seq < 4).
                time.sleep(_WINDOW_RETRY_SLEEP_S)
                continue
            # Walk the box's bucket chains one link depth at a time: (m,) slots per depth.
            chain = np.asarray(reader.bucket_head[bx0:bx1, by0:by1]).reshape(-1)
            chain = chain[chain != -1]
            found, count, broken = [], 0, False
            while chain.size:
                count += chain.size
                # An out-of-range link, or more slots than the pool holds (a cycle): a torn walk.
                if chain.min() < 0 or chain.max() >= n_max or count > n_max:
                    broken = True
                    break
                found.append(chain)
                chain = np.asarray(next_in_bucket[chain])
                chain = chain[chain != -1]
            slots = (np.concatenate(found).astype(np.int64) if found and not broken
                     else np.empty(0, dtype=np.int64))                       # (k,) node slots
            # (k,) coordinates [m] / active flags and (k, layers) payloads, copied out of the graph.
            x = np.asarray(reader.node_x[slots], dtype=np.float64)
            y = np.asarray(reader.node_y[slots], dtype=np.float64)
            active = np.asarray(reader.node_active[slots]) == 1
            cmd_v = np.asarray(reader.node_payload(_CMD_V)[slots], dtype=np.float32)
            cmd_w = np.asarray(reader.node_payload(_CMD_W)[slots], dtype=np.float32)
            if int(read_seq()) == seq:
                if broken:
                    raise RuntimeError(
                        f"travgraph window read: the chains of buckets x[{bx0}, {bx1}) y[{by0}, "
                        f"{by1}) hold an out-of-range slot or a cycle (n_max={n_max}) at a stable "
                        f"graph_seq {seq}: the shared graph is corrupt")
                break
            time.sleep(_WINDOW_RETRY_SLEEP_S)
        else:
            raise RuntimeError(
                f"travgraph window read: no consistent copy in {_WINDOW_MAX_RETRIES} attempts "
                f"(last graph_seq {seq}): the writer stayed mid-batch or kept editing during the copy")

        # Active nodes inside the box rectangle; each must sit on its own lattice point.
        keep = active & (x >= x_lo) & (x < x_hi) & (y >= y_lo) & (y < y_hi)
        x, y, cmd_v, cmd_w = x[keep], y[keep], cmd_v[keep], cmd_w[keep]
        kx = np.rint(x / res).astype(np.int64)
        ky = np.rint(y / res).astype(np.int64)
        off_lattice = (np.abs(kx * res - x) > _LATTICE_COORD_TOL_M) | (
            np.abs(ky * res - y) > _LATTICE_COORD_TOL_M)
        if off_lattice.any():
            bad = int(np.argmax(off_lattice))
            raise RuntimeError(
                f"travgraph terrain: {int(off_lattice.sum())} node(s) in the window are off the "
                f"{res} m lattice, e.g. ({x[bad]:.4f}, {y[bad]:.4f}); the planner's "
                "trav_estimation_resoultion must match the world model's")
        i, j = kx - kx0, ky - ky0                                       # (n,) lattice indices
        flat = i * ny + j
        if np.unique(flat).size != flat.size:
            raise RuntimeError(
                f"travgraph terrain: {flat.size - np.unique(flat).size} duplicate node(s) on a "
                "lattice point in the window; this module needs one node per lattice point")
        if cmd_v.ndim != 2 or cmd_v.shape[1] != layers or cmd_w.shape != cmd_v.shape:
            raise RuntimeError(
                f"travgraph terrain: node payloads cmd_v{cmd_v.shape}/cmd_w{cmd_w.shape} do not "
                f"carry the map's {layers} heading layers")

        # Start from the pinned lattice when the last refresh completed on this frame, so only
        # the union of the old and new boxes can differ; otherwise from all-NaN.
        pinned = self._np_lattice
        if (self.version is not None and self._window_box is not None and pinned is not None
                and pinned.shape == (nx, ny, layers, 2)):
            p0, p1, q0, q1 = self._window_box
            span = (min(i0, p0), max(i1, p1), min(j0, q0), max(j1, q1))
            values = pinned.copy()
            values[span[0]:span[1], span[2]:span[3]] = np.nan
        else:
            span = (0, nx, 0, ny)
            values = np.full((nx, ny, layers, 2), np.nan, dtype=np.float32)
        values[i, j, :, 0] = cmd_v
        values[i, j, :, 1] = cmd_w

        # Lattice points of the box without an active node: vetoed (NaN), reported on change.
        holes = (i1 - i0) * (j1 - j0) - flat.size if (i1 > i0 and j1 > j0) else 0
        if holes != self._window_holes:
            if holes:
                filled = np.zeros((i1 - i0, j1 - j0), dtype=bool)
                filled[i - i0, j - j0] = True
                hi, hj = np.argwhere(~filled)[0]
                print(f"[travgraph] {holes} lattice point(s) in the {self.window_radius:g} m window "
                      f"have no active node, e.g. ({(kx0 + i0 + hi) * res:.2f}, "
                      f"{(ky0 + j0 + hj) * res:.2f}); the cells around them are hard-vetoed")
            else:
                print(f"[travgraph] every lattice point in the {self.window_radius:g} m window has "
                      "an active node again")
            self._window_holes = holes
        return seq, values, kx0, ky0, span

    # ------------------------------------------------------------------ table builders

    def _build_cell_tables(
        self, geometry: tuple, nx: int, ny: int, kx0: int, ky0: int,
    ) -> None:
        """Per fine cell: anchor square index and bilinear basis.  Geometry only."""
        _, xmax, _, ymax, map_resolution, rows, cols, _ = geometry
        segments = math.floor(self.resolution / float(map_resolution))
        if segments < 1:
            raise ValueError(
                f"travgraph terrain: lattice resolution {self.resolution} m is finer than the map "
                f"resolution {map_resolution} m; the writer's interpolation needs >= 1 segment")
        weights = sigmoid_weights(segments, self.sigmoid_k)
        # Lattice world coordinates as the writer computes them (R * integer, float64).
        node_rows = fine_index(float(xmax), (kx0 + np.arange(nx)) * self.resolution,
                               float(map_resolution), int(rows))
        node_cols = fine_index(float(ymax), (ky0 + np.arange(ny)) * self.resolution,
                               float(map_resolution), int(cols))
        anchor_x, weight_x = axis_anchor_table(node_rows, int(rows), segments, weights)
        anchor_y, weight_y = axis_anchor_table(node_cols, int(cols), segments, weights)
        # (rows, cols) flat anchor square index into the (nx*ny, layers, ...) coefficient table.
        square = anchor_x[:, None] * ny + anchor_y[None, :]
        wx = np.broadcast_to(weight_x[:, None], (int(rows), int(cols)))
        wy = np.broadcast_to(weight_y[None, :], (int(rows), int(cols)))
        basis = np.stack((wx, wy, wx * wy), axis=-1)                    # (rows, cols, 3)
        self._np_cell_square = np.ascontiguousarray(square.reshape(-1), dtype=np.int64)
        self._np_cell_basis = np.ascontiguousarray(basis.reshape(-1, 3), dtype=np.float32)
        self._cols = int(cols)

    @staticmethod
    def _square_coefficients(values: np.ndarray, squares: np.ndarray) -> np.ndarray:
        """``(s, layers, 4, 2)`` bilinear coefficients of the given flat anchor squares.

        ``squares`` is ``(s,)`` flat ids i * ny + j.  For anchor A = (i, j) with lower neighbours
        X = (i-1, j), Y = (i, j-1), XY = (i-1, j-1) (clamped at the lattice edge, giving zero
        slope):  v = c0 + wx c1 + wy c2 + wx wy c3 with c0 = A, c1 = X - A, c2 = Y - A,
        c3 = (XY - Y) - (X - A).  Equal corners reproduce A exactly, so uniform terrain and
        too-risky plateaus keep their stamped value bit-for-bit.  The one place the formula is
        written, so a full build and a changed-squares update produce identical coefficients.
        """
        si, sj = np.divmod(squares, values.shape[1])
        li, lj = np.maximum(si - 1, 0), np.maximum(sj - 1, 0)
        a, x, y, xy = values[si, sj], values[li, sj], values[si, lj], values[li, lj]  # (s, L, 2)
        coeff = np.stack((a, x - a, y - a, (xy - y) - (x - a)), axis=2)            # (s, L, 4, 2)
        return np.ascontiguousarray(coeff, dtype=np.float32)

    # ------------------------------------------------------------------ public API

    def refresh(self, geometry: tuple, device: torch.device, dtype: torch.dtype,
                center: Optional[Any] = None) -> None:
        """Pin the current graph version on ``device``; a no-op when nothing changed.

        ``geometry`` is TerrainMPPI's fine-grid tuple (xmin, xmax, ymin, ymax, resolution, rows,
        cols, layers) -- the grid the planner indexes, which the interpolation is defined on.
        ``center`` is the robot's world (x, y, ...) the read window follows when window_radius
        is finite; None keeps the previous center (the whole lattice before any is given).
        """
        layers = int(geometry[7])
        # (i0, i1, j0, j1) lattice box to read, or None for the whole-graph snapshot path.
        box = self._window_box_for(center)
        if (self.is_ready and self._device == device and box == self._window_box
                and self.current_version() == self.version):
            return
        if box is None:
            version, values, kx0, ky0 = self._read_lattice(layers)
            span = (0, values.shape[0], 0, values.shape[1])
        else:
            version, values, kx0, ky0, span = self._read_window(layers, box)
        rows, cols = int(geometry[5]), int(geometry[6])
        cell_key = (tuple(float(g) for g in geometry), values.shape[:2], kx0, ky0, str(device))
        # Every table matches a completed earlier version on this geometry and device -- the
        # only state a changed-nodes-only update may build on.  `version is not None` is what
        # says the last refresh completed (see below).
        incremental = (self.is_ready and self.version is not None and self._device == device
                       and cell_key == self._cell_key and self._np_lattice is not None
                       and (self._grid is not None) == self.rasterize)
        if incremental:
            # (nx, ny) True where any layer / channel of a node differs from the pinned version.
            # Only `span` can differ; NaN (a vetoed node) staying NaN is no change.
            changed = np.zeros(values.shape[:2], dtype=bool)
            new = values[span[0]:span[1], span[2]:span[3]]
            old = self._np_lattice[span[0]:span[1], span[2]:span[3]]
            changed[span[0]:span[1], span[2]:span[3]] = np.any(
                (new != old) & ~(np.isnan(new) & np.isnan(old)), axis=(2, 3))
            if not changed.any():
                # Identical values (a third of the writer's versions: re-estimates, edge-only
                # edits) leave every table valid; only the pin moves.
                self._window_box = box
                self.version = version
                return
        # No version is pinned while tables are being modified: a refresh that fails part-way
        # (e.g. CUDA OOM) can then never pass for a valid snapshot, and the next one rebuilds.
        self.version = None
        if incremental:
            self._update_changed(values, changed, rows, cols, layers)
        else:
            if cell_key != self._cell_key or self._cell_square is None:
                self._build_cell_tables(geometry, values.shape[0], values.shape[1], kx0, ky0)
                self._cell_square = torch.as_tensor(self._np_cell_square, device=device)
                self._cell_basis = torch.as_tensor(self._np_cell_basis, device=device, dtype=dtype)
                self._cell_key = cell_key
            self._np_lattice = values
            self._np_coeff = self._square_coefficients(
                values, np.arange(values.shape[0] * values.shape[1]))
            self._coeff = torch.as_tensor(self._np_coeff, device=device, dtype=dtype)
            if self.rasterize:
                self._grid = self._rasterize(rows, cols, layers, device)
            else:
                self._grid = None
        self._window_box = box
        self._device = device
        self.version = version

    def _update_changed(self, values: np.ndarray, changed: np.ndarray,
                        rows: int, cols: int, layers: int) -> None:
        """Redo coefficients, their upload and the grid for the squares changed nodes touch.

        ``changed`` is the ``(nx, ny)`` node mask.  A node (i, j) is a corner of the squares
        anchored at (i, j), (i+1, j), (i, j+1) and (i+1, j+1); nothing else reads its value.
        """
        nx, ny = values.shape[:2]
        ci, cj = np.nonzero(changed)
        # (4 * c,) candidate anchors per changed node, kept inside the lattice.
        ai = np.concatenate((ci, ci + 1, ci, ci + 1))
        aj = np.concatenate((cj, cj, cj + 1, cj + 1))
        inside = (ai < nx) & (aj < ny)
        squares = np.unique(ai[inside] * ny + aj[inside])                     # (s,) flat ids
        coeff = self._square_coefficients(values, squares)                    # (s, L, 4, 2)
        self._np_lattice = values
        self._np_coeff[squares] = coeff
        squares_t = torch.as_tensor(squares, device=self._device)
        # Upload only the changed rows of the (nx*ny, L, 4, 2) device table.
        self._coeff[squares_t] = torch.as_tensor(coeff, device=self._device, dtype=self._coeff.dtype)
        if self._grid is not None:
            self._raster_squares(squares_t, rows, cols, layers)

    def _raster_squares(self, squares: torch.Tensor, rows: int, cols: int, layers: int) -> None:
        """Re-interpolate, in place, only the fine cells whose anchor square is in ``squares``.

        The cells are found on the device: mark the ``(s,)`` squares in an ``(nx*ny,)`` mask and
        select the cells whose ``_cell_square`` is marked -- one pass over the cell table, no
        per-square host loop.  Per heading layer, so transient memory stays bounded for any
        number of changed nodes.  Same arithmetic per cell as _rasterize(), so bit-identical.
        """
        dirty = torch.zeros(self._coeff.shape[0], dtype=torch.bool, device=self._device)
        dirty[squares] = True
        cells = torch.nonzero(dirty[self._cell_square], as_tuple=True)[0]   # (m,) flat fine cells
        square = self._cell_square[cells]                                    # (m,)
        basis = self._cell_basis[cells]                                      # (m, 3)
        grid = self._grid.view(rows * cols, layers, 2)
        for layer in range(layers):
            grid[cells, layer] = self._blend(self._coeff[square, layer], basis)

    def _rasterize(self, rows: int, cols: int, layers: int, device: torch.device) -> torch.Tensor:
        """``(rows, cols, layers, 2)`` interpolation of every fine cell, one heading layer at a time.

        Chunked because the transient tensors of the interpolation (4x2 coefficients and their
        weighted products per cell) would otherwise cover every layer at once -- 1 GB on a
        50x50 m map, on a GPU shared with the simulator and perception.  Written into the
        current grid's buffer when its shape, device and dtype still fit, so a rebuild never
        holds two grids.  Same arithmetic per cell as _interpolate(), so bit-identical to it.
        """
        grid = self._grid
        if (grid is None or tuple(grid.shape) != (rows, cols, layers, 2)
                or grid.device != device or grid.dtype != self._coeff.dtype):
            grid = torch.empty((rows, cols, layers, 2), device=device, dtype=self._coeff.dtype)
        # Over the whole grid the flat cell index is 0..rows*cols-1, so the per-cell tables are
        # read as-is: (rows, cols) anchor square and (rows, cols, 3) basis, shared by all layers.
        square = self._cell_square.view(rows, cols)
        basis = self._cell_basis.view(rows, cols, 3)
        for layer in range(layers):
            grid[:, :, layer] = self._blend(self._coeff[square, layer], basis)
        return grid

    def gather(self, row: torch.Tensor, col: torch.Tensor, layer: torch.Tensor) -> torch.Tensor:
        """``(..., 2)`` interpolated (cmd_v, cmd_w) at fine cells; inputs are equal-shape long."""
        if self._grid is not None:
            return self._grid[row, col, layer]
        return self._interpolate(row, col, layer)

    def _interpolate(self, row: torch.Tensor, col: torch.Tensor, layer: torch.Tensor) -> torch.Tensor:
        """The sigmoid interpolation itself, evaluated at the given fine cells (any equal shape)."""
        flat = torch.add(col, row, alpha=self._cols)                     # (...,) fine cell, one kernel
        coeff = self._coeff[self._cell_square[flat], layer]               # (..., 4, 2) one 2-D gather
        basis = self._cell_basis[flat]                                    # (..., 3)
        return self._blend(coeff, basis)

    @staticmethod
    def _blend(coeff: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        """``(..., 2)`` c0 + wx c1 + wy c2 + wx wy c3 from ``(..., 4, 2)`` coeff, ``(..., 3)`` basis.

        The one place the bilinear blend is written, so the rasterized grid and the per-query
        path run identical arithmetic and stay bit-identical.
        """
        return coeff[..., 0, :] + (coeff[..., 1:, :] * basis[..., :, None]).sum(dim=-2)

    def gather_numpy(self, row: np.ndarray, col: np.ndarray, layer: np.ndarray) -> np.ndarray:
        """CPU twin of gather() over the same tables: no device round trip for small queries."""
        flat = np.asarray(row, dtype=np.int64) * self._cols + np.asarray(col, dtype=np.int64)
        coeff = self._np_coeff[self._np_cell_square[flat], np.asarray(layer, dtype=np.int64)]
        basis = self._np_cell_basis[flat]
        return coeff[..., 0, :] + (coeff[..., 1:, :] * basis[..., :, None]).sum(axis=-2)

    def layer_image(self, layer: int, rows: int, cols: int) -> np.ndarray:
        """``(rows, cols)`` float32 interpolated cmd_v of one heading layer, for plotting."""
        flat = np.arange(rows * cols, dtype=np.int64)
        coeff = self._np_coeff[self._np_cell_square[flat], int(layer)]               # (n, 4, 2)
        values = coeff[:, 0, :] + (coeff[:, 1:, :] * self._np_cell_basis[flat][:, :, None]).sum(axis=1)
        return np.asarray(values[:, 0].reshape(rows, cols), dtype=np.float32)

    def value_range(self) -> Optional[tuple[float, float]]:
        """(min, max) cmd_v over every node and layer: the interpolated range, all layers."""
        if self._np_lattice is None:
            return None
        values = self._np_lattice[..., 0]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return None
        return float(finite.min()), float(finite.max())

    def drop_device_tensors(self) -> None:
        """Forget the device copies so the next refresh() rebuilds them (OOM migration)."""
        self._cell_square = None
        self._cell_basis = None
        self._coeff = None
        self._grid = None
        self._cell_key = None
        self._device = None
        self.version = None

    def close(self) -> None:
        """Release the reader's shared-memory handles; never unlinks writer-owned blocks."""
        if self._reader is not None and hasattr(self._reader, "close"):
            self._reader.close()
        self._reader = None
