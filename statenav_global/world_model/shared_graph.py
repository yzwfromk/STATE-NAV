"""
Shared-Memory Graph Storage Core (Part 1a / Phase A backend)

Slot-pool graph living in POSIX shared memory, per
docs/shared_memory_graph_design.md and
docs/shared_memory_graph_plan_1a_phase_A_backend.md.

Roles
-----
- SharedGraphWriter : owns and creates every SHM block; the ONLY process that
  may mutate shared state (single-writer invariant). All mutation happens
  inside a seqlock window (graph_seq odd = mid-edit, even = quiescent).
- SharedGraphReader : attaches by name, validates block presence and exact
  byte sizes against the shared GraphLayout, and produces GraphSnapshot
  replicas via a seqlock retry loop. It has NO mutators.
- GraphSnapshot     : fully functional offline replica (allocator state
  included) exposing the same mutation engine over private local arrays with
  a no-op edit guard. Replica edits never touch shared memory and are wiped
  on the next snapshot(into=...) overwrite.

One mutation engine (_GraphEngine) implements every structural/payload
operation once over an "array bundle": SHM arrays + seqlock guard for the
writer, local arrays + no-op guard for the replica.

Data structure (see design doc section 2)
-----------------------------------------
- Node pool  (n_max rows): x, y (float32), active (uint8), generation,
  update_version, first_edge, next_in_bucket (int32). next_in_bucket doubles
  as the node free-list link while the slot is free.
- Edge pool  (e_max rows, directed; an undirected edge is two rows):
  target, next (int32), length (float32, immutable, auto-computed).
  next doubles as the edge free-list link; target == -1 marks a free row.
- bucket_head[n_bx, n_by] (int32): spatial index chain heads.
- One parallel SHM block per payload property (never packed structs).

Platform / concurrency assumptions (documented, guarded at startup)
-------------------------------------------------------------------
- Memory ordering: aligned int32 stores are single-copy atomic on x86-64 and
  on ARMv8 alike, but only x86-64 (TSO) also guarantees that stores are not
  reordered with other stores, so only there do plain NumPy writes implement
  the seqlock on their own. On any other machine every sequence store/load is
  paired with an explicit seq_cst fence (_seq_fence, resolved from libatomic);
  a machine where that fence cannot be resolved is rejected at writer/reader
  construction by _require_supported_platform().
- Counters (graph_seq, generation, update_version) are int32 and are only
  compared by equality/inequality, so wraparound is harmless in principle.
  NumPy raises on int32 overflow at assignment; reaching 2**31 edits is not
  a practical concern, so this is documented rather than guarded.
- Single mutator process. The seqlock protects readers against the writer,
  never writers against each other.
- Reader processes may run in any process tree, including an independently
  launched one: on Python < 3.13 attaching registers the block with the
  reader's own multiprocessing resource tracker, which would unlink
  writer-owned blocks at reader exit, so every block a reader attaches is
  disowned immediately via _untrack_shm(). Unlinking stays writer-only.
"""

import atexit
import ctypes
import math
import platform
import signal
import sys
import time
from types import MappingProxyType
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from multiprocessing import shared_memory
# Reader-side resource-tracker helper, shared with SharedMemoryBackend so both
# readers (map blocks and graph blocks) disown attached blocks the same way.
from statenav_global.world_model.shared_memory_backend import (
    _untrack_shm, _unlink_shm, _CLEANUP_SIGNALS)


# ---------------------------------------------------------------------------
# Errors — every expected failure raises one of these with actionable detail.
# ---------------------------------------------------------------------------

class SharedGraphError(RuntimeError):
    """Base class for shared-graph backend errors."""


class GraphAttachError(SharedGraphError):
    """Reader could not attach: missing/mis-sized blocks or unpublished init."""


class GraphCapacityError(SharedGraphError):
    """A pool (shared writer pool or replica local headroom) is exhausted."""


class GraphCorruptionError(SharedGraphError):
    """A structural invariant does not hold (cycle, half edge pair, ...)."""


class SnapshotRetryError(SharedGraphError):
    """snapshot() could not obtain a torn-free copy within max_retries."""


# ---------------------------------------------------------------------------
# Metadata shared memory layout (mirrors statenav_meta_shm conventions)
# ---------------------------------------------------------------------------

_GRAPH_META_SHM_NAME = 'statenav_graph_meta_shm'

# One aligned structured record. graph_seq: odd = writer mid-edit, even =
# quiescent. node_hwm / edge_hwm are EXCLUSIVE high-water marks: the used
# prefix is always [0, hwm).
_GRAPH_META_DTYPE = np.dtype([
    ('graph_seq',             np.int32),
    ('node_free_head',        np.int32),
    ('edge_free_head',        np.int32),
    ('node_hwm',              np.int32),
    ('edge_hwm',              np.int32),
    ('n_max',                 np.int32),
    ('e_max',                 np.int32),
    ('n_bx',                  np.int32),
    ('n_by',                  np.int32),
    ('xmin',                  np.float64),
    ('xmax',                  np.float64),
    ('ymin',                  np.float64),
    ('ymax',                  np.float64),
    ('bucket_size',           np.float64),
    ('default_lookup_radius', np.float64),   # consumed by Part 1b only
], align=True)

# Fail loud at import if the aligned-struct / 4-byte-aligned seqlock
# assumptions ever break (they cannot with the layout above, but the seqlock
# is only correct if they hold).
if not _GRAPH_META_DTYPE.isalignedstruct:
    raise RuntimeError("shared_graph: _GRAPH_META_DTYPE is not an aligned struct")
if _GRAPH_META_DTYPE.fields['graph_seq'][1] % 4 != 0:
    raise RuntimeError("shared_graph: graph_seq is not 4-byte aligned; "
                       "atomic int32 store assumption would not hold")

_SUPPORTED_MACHINES = ('x86_64', 'amd64')

# memory_order_seq_cst / __ATOMIC_SEQ_CST, the order argument of the C11
# atomic_thread_fence() exported by libatomic. seq_cst compiles to a full
# barrier (dmb ish on aarch64) and is the only order that gives the seqlock
# both the store-store (writer) and load-load (reader) ordering it needs.
_ATOMIC_SEQ_CST = 5


def _resolve_seq_fence():
    """
    Resolve the seqlock memory barrier for this machine, once at import.

    Returns (fence_callable, error_detail). error_detail is None when a usable
    barrier exists.

    - x86-64 (TSO): the required store-store / load-load ordering is
      architectural, so the fence is a no-op and the seqlock keeps its original
      zero-cost form. Nothing about the x86-64 code path changes.
    - Any other machine (aarch64, ...): a real seq_cst fence is mandatory.
      Without it the odd-sequence store may become visible to a reader AFTER
      the payload stores it is supposed to shield, and the reader's s1 == s2
      check would then accept a torn snapshot instead of retrying.

    Never raises at import: shared_graph is imported unconditionally by
    globalmap.py, so an import-time raise would break every non-Astar planner
    on an unsupported machine too. Failure is reported by the returned
    error_detail and turned into an actionable error by
    _require_supported_platform(), i.e. only when a writer/reader is built.
    """
    machine = platform.machine().lower()
    if machine in _SUPPORTED_MACHINES:
        return (lambda: None), None
    try:
        # C11 spelling 'atomic_thread_fence' (LIBATOMIC_1.2); the GCC builtin
        # '__atomic_thread_fence' is inlined by the compiler and is NOT an
        # exported symbol, so it cannot be used here.
        _lib = ctypes.CDLL('libatomic.so.1')
        _fence = getattr(_lib, 'atomic_thread_fence')
        _fence.argtypes = [ctypes.c_int]
        _fence.restype = None
        _fence(_ATOMIC_SEQ_CST)     # probe once here, never fail mid-batch
    except (OSError, AttributeError) as exc:
        detail = f"{type(exc).__name__}: {exc}"

        def _missing_fence():
            # Unreachable via the public API (_require_supported_platform runs
            # first); loud rather than silently unordered if that ever changes.
            raise RuntimeError(
                f"shared_graph: no seqlock memory barrier available on "
                f"'{machine}' ({detail}); _require_supported_platform() should "
                f"have rejected this machine before any shared access")

        return _missing_fence, detail
    return (lambda: _fence(_ATOMIC_SEQ_CST)), None


# Module-level seqlock barrier: no-op on x86-64, real seq_cst fence elsewhere.
_seq_fence, _SEQ_FENCE_ERROR = _resolve_seq_fence()


def _require_supported_platform():
    """
    Guard the seqlock's memory-ordering requirements for this machine.

    x86-64 satisfies them architecturally (TSO). Any other machine is accepted
    only when the explicit seq_cst fence was resolved at import; that fence is
    what actually orders the odd/even sequence stores against the payload
    stores there.
    """
    machine = platform.machine().lower()
    if machine in _SUPPORTED_MACHINES:
        return
    if _SEQ_FENCE_ERROR is None:
        return
    raise RuntimeError(
        f"shared_graph on '{machine}' needs an explicit memory barrier for its "
        f"seqlock (the x86-64 TSO store-store / load-load guarantees do not hold "
        f"here), but libatomic's atomic_thread_fence could not be resolved: "
        f"{_SEQ_FENCE_ERROR}. Install libatomic1 in this image (it ships with the "
        f"gcc runtime).")


# ---------------------------------------------------------------------------
# Core array specs and SHM block naming (single source of truth)
# ---------------------------------------------------------------------------

# Node pool core fields, one SHM block each, shape (n_max,).
_NODE_CORE_SPECS = (
    ('x',              np.float32),  # world position
    ('y',              np.float32),
    ('active',         np.uint8),    # 1 = live; 0 = free or deleted
    ('generation',     np.int32),    # +1 each slot recycle (identity counter)
    ('update_version', np.int32),    # +1 each payload rewrite (data counter)
    ('first_edge',     np.int32),    # head of out-edge chain (-1 = none)
    ('next_in_bucket', np.int32),    # bucket chain link / free-list link
)

# Edge pool core fields, one SHM block each, shape (e_max,).
_EDGE_CORE_SPECS = (
    ('target', np.int32),    # destination node slot; -1 = free row sentinel
    ('next',   np.int32),    # next edge of same owner / free-list link
    ('length', np.float32),  # endpoint distance, auto-computed, immutable
)

_BUCKET_HEAD_SHM_NAME = 'graph_bucket_head_shm'

# Payload names that would collide with mutator parameter names. Part 1b adds
# 'tol' and 'connect_radius' because those are frontend control arguments that
# sit alongside **payload kwargs (insert_node_at, add_edge_at, ...). 'theta',
# 'angle_tol', and the local 'full_fields' need no reservation: they belong to
# signatures that carry no **payload.
_RESERVED_PAYLOAD_NAMES = frozenset({'self', 'slot', 'indexed', 'x', 'y', 'radius',
                                     'tol', 'connect_radius'})

# Small bounded backoff between snapshot retries so the retry loop cannot
# tight-spin through all attempts faster than a normal short writer batch.
_SNAPSHOT_RETRY_SLEEP_S = 5e-4

# Guard object so GraphSnapshot can only be built through the sanctioned
# factory paths (SharedGraphReader.snapshot / GraphSnapshot.copy).
_SNAPSHOT_CONSTRUCTION_TOKEN = object()


def _node_core_block_name(field: str) -> str:
    return f'graph_node_{field}_shm'


def _edge_core_block_name(field: str) -> str:
    return f'graph_edge_{field}_shm'


def _node_payload_block_name(name: str) -> str:
    return f'graph_node_payload_{name}_shm'


def _edge_payload_block_name(name: str) -> str:
    return f'graph_edge_payload_{name}_shm'


def _expected_data_blocks(layout: 'GraphLayout', n_max: int, e_max: int,
                          n_bx: int, n_by: int) -> Dict[str, Tuple[tuple, np.dtype]]:
    """
    Every data SHM block implied by capacities + the shared layout, as
    {block_name: (array_shape, dtype)}. Used identically by the writer
    (creation), the reader (presence + exact-size validation), and stale
    cleanup — one source of truth for the block set.
    """
    blocks: Dict[str, Tuple[tuple, np.dtype]] = {}
    for field, dt in _NODE_CORE_SPECS:
        blocks[_node_core_block_name(field)] = ((n_max,), np.dtype(dt))
    for field, dt in _EDGE_CORE_SPECS:
        blocks[_edge_core_block_name(field)] = ((e_max,), np.dtype(dt))
    blocks[_BUCKET_HEAD_SHM_NAME] = ((n_bx, n_by), np.dtype(np.int32))
    # Node and edge payloads live in separate namespaces, so the same logical
    # payload name may exist on both sides without colliding.
    for name, (shape, dt, _default) in layout.node_payloads.items():
        blocks[_node_payload_block_name(name)] = ((n_max,) + shape, dt)
    for name, (shape, dt, _default) in layout.edge_payloads.items():
        blocks[_edge_payload_block_name(name)] = ((e_max,) + shape, dt)
    return blocks


# ---------------------------------------------------------------------------
# Shared payload-layout contract
# ---------------------------------------------------------------------------

class GraphLayout:
    """
    Immutable Python-side payload layout shared by writer and reader.

    Both sides must be constructed from the SAME layout definition (import it
    from one application module). Payload-layout changes are deployed by
    restarting the graph processes and recreating the SHM blocks.

    Each spec maps  name -> (per_element_shape, dtype)  or
    name -> (per_element_shape, dtype, default)  where the stored array has
    shape (capacity, *per_element_shape). `default` is the value every
    payload row holds wherever it is (re)initialized (graph creation,
    node-slot recycle, edge-row free, replica beyond-prefix reset): a scalar
    (broadcast over per_element_shape) or an array of exactly
    per_element_shape. When omitted (or None) the implicit default is NaN
    for float/complex payloads ("never written" sentinel) and 0 for
    integer/bool payloads, which cannot represent NaN.
    """

    def __init__(self, node_payloads=None, edge_payloads=None):
        self._node_payloads = self._normalize(node_payloads or {}, 'node')
        self._edge_payloads = self._normalize(edge_payloads or {}, 'edge')

    @staticmethod
    def _normalize(spec, kind: str):
        out = {}
        for name, entry in dict(spec).items():
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError(f"{kind} payload name {name!r} must be a valid identifier")
            if name in _RESERVED_PAYLOAD_NAMES:
                raise ValueError(
                    f"{kind} payload name {name!r} is reserved (collides with a "
                    f"mutator parameter); reserved names: {sorted(_RESERVED_PAYLOAD_NAMES)}")
            try:
                parts = tuple(entry)
            except TypeError:
                parts = ()
            if len(parts) == 2:
                (shape, dt), default = parts, None
            elif len(parts) == 3:
                shape, dt, default = parts
            else:
                raise ValueError(
                    f"{kind} payload {name!r}: spec must be (per_element_shape, dtype) "
                    f"or (per_element_shape, dtype, default), got {entry!r}")
            if isinstance(shape, (int, np.integer)):
                shape = (int(shape),)
            shape = tuple(int(d) for d in shape)
            if any(d < 1 for d in shape):
                raise ValueError(f"{kind} payload {name!r}: shape {shape} has a dim < 1")
            dt = np.dtype(dt)
            if dt.hasobject:
                raise ValueError(
                    f"{kind} payload {name!r}: object dtype {dt} cannot live in shared memory")
            out[name] = (shape, dt,
                         GraphLayout._normalize_default(kind, name, shape, dt, default))
        return MappingProxyType(out)

    @staticmethod
    def _normalize_default(kind: str, name: str, shape: tuple, dt: np.dtype,
                           default) -> np.ndarray:
        """
        Validate and canonicalize one payload default into a read-only array
        of exactly per-element `shape` and dtype `dt`. `default=None` selects
        the implicit default: NaN for float/complex payloads, 0 for
        integer/bool payloads (which cannot represent NaN). All checks fail
        loud at layout construction, never at insert/free time.
        """
        if default is None:
            # dt.type(0) yields a correctly-typed zero (e.g. np.uint8(0),
            # np.bool_(False)) so the cast checks below pass for every kind.
            default = np.nan if dt.kind in 'fc' else dt.type(0)
        try:
            arr = np.asarray(default)
        except Exception as exc:
            raise ValueError(
                f"{kind} payload {name!r}: default cannot be converted to an "
                f"array: {exc}")
        if arr.dtype.hasobject:
            raise ValueError(f"{kind} payload {name!r}: default is not numeric")
        if arr.shape not in ((), shape):
            raise ValueError(
                f"{kind} payload {name!r}: default shape {arr.shape} must be () "
                f"(scalar, broadcast) or the per-element shape {shape}")
        # Fail loud on NaN/inf toward a dtype that cannot represent it —
        # casting would silently produce garbage (e.g. NaN -> INT_MIN).
        if dt.kind not in 'fc' and arr.dtype.kind in 'fc' and not np.all(np.isfinite(arr)):
            raise ValueError(
                f"{kind} payload {name!r}: non-finite default {default!r} cannot "
                f"be represented in {dt}; pick a finite sentinel (e.g. -1)")
        if arr.dtype.kind in 'iu' and dt.kind in 'iu':
            # integer -> integer: same value-range rule as payload writes.
            info = np.iinfo(dt)
            if arr.size and (int(arr.min()) < info.min or int(arr.max()) > info.max):
                raise ValueError(
                    f"{kind} payload {name!r}: default outside {dt} range "
                    f"[{info.min}, {info.max}]: {default!r}")
        elif not np.can_cast(arr.dtype, dt, casting='same_kind'):
            raise ValueError(
                f"{kind} payload {name!r}: default dtype {arr.dtype} does not "
                f"safely cast to {dt} (same_kind); e.g. pass 0.0 (not 0) for a "
                f"float payload")
        out = np.broadcast_to(arr.astype(dt, copy=False), shape).copy()
        out.flags.writeable = False   # shared immutable row template
        return out

    @property
    def node_payloads(self):
        return self._node_payloads

    @property
    def edge_payloads(self):
        return self._edge_payloads

    @staticmethod
    def _specs_equal(a, b) -> bool:
        """Spec-map equality. NaN-aware: NaN defaults must compare equal,
        otherwise identical writer/reader layouts would never match."""
        if a.keys() != b.keys():
            return False
        for name in a:
            shape_a, dt_a, default_a = a[name]
            shape_b, dt_b, default_b = b[name]
            if (shape_a != shape_b or dt_a != dt_b
                    or not np.array_equal(default_a, default_b, equal_nan=True)):
                return False
        return True

    def __eq__(self, other):
        if not isinstance(other, GraphLayout):
            return NotImplemented
        return (self._specs_equal(self._node_payloads, other._node_payloads)
                and self._specs_equal(self._edge_payloads, other._edge_payloads))

    def __repr__(self):
        return (f"GraphLayout(node_payloads={dict(self._node_payloads)!r}, "
                f"edge_payloads={dict(self._edge_payloads)!r})")


# ---------------------------------------------------------------------------
# Edit guards — the only place the seqlock counter is touched
# ---------------------------------------------------------------------------

class _NoOpEditGuard:
    """Edit guard for the local snapshot replica: no seqlock needed."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _SeqlockEditGuard:
    """
    Reentrant, process-local seqlock guard around writer mutations.

    Only the depth 0 -> 1 transition flips graph_seq even -> odd and only the
    depth 1 -> 0 transition flips it back (inner nested entries never touch
    the sequence), so public mutators compose freely inside edit_batch().
    The outer exit publishes the even state in finally position: preflight
    validation runs before the first structural write, so an exception that
    propagates out of a batch leaves a structurally consistent graph.
    """

    def __init__(self, meta: np.ndarray):
        self._meta = meta          # (1,) structured record view into SHM
        self._depth = 0            # process-local nesting depth

    def __enter__(self):
        if self._depth == 0:
            seq = int(self._meta[0]['graph_seq'])
            if seq % 2 != 0:
                # Never silently "fix" this: an odd sequence at outer entry
                # means a second mutator or a crashed batch.
                raise GraphCorruptionError(
                    f"graph_seq={seq} is odd at edit-batch entry: another writer "
                    f"is active or a previous batch crashed mid-edit; the "
                    f"single-writer invariant is broken")
            self._meta[0]['graph_seq'] = seq + 1   # odd: batch in progress
            # Release barrier: the odd sequence must be observable BEFORE any
            # payload store of this batch, otherwise a reader could sample the
            # still-even old sequence twice around a torn copy (no-op on x86-64).
            _seq_fence()
        self._depth += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        self._depth -= 1
        if self._depth == 0:
            # Publish quiescent (even) state — finally semantics via with.
            # Release barrier first: every payload store of this batch must be
            # observable before the even sequence that advertises it as complete.
            _seq_fence()
            self._meta[0]['graph_seq'] = int(self._meta[0]['graph_seq']) + 1
        return False


# ---------------------------------------------------------------------------
# Read-only state view shared by writer, reader, and snapshot
# ---------------------------------------------------------------------------

class _GraphStateView:
    """
    Binds the array bundle and exposes read-only introspection. All returned
    array views are non-writable: every mutation must go through the engine
    API so it is covered by the edit guard.
    """

    def _bind_views(self, meta, node_core, edge_core, bucket_head,
                    node_payloads, edge_payloads, layout):
        self._meta = meta                    # (1,) _GRAPH_META_DTYPE record
        self._node_core = node_core          # {field: (n_max,) array}
        self._edge_core = edge_core          # {field: (e_max,) array}
        self._bucket_head = bucket_head      # (n_bx, n_by) int32 chain heads
        self._node_payloads = node_payloads  # {name: (n_max, *shape) array}
        self._edge_payloads = edge_payloads  # {name: (e_max, *shape) array}
        self._layout = layout
        # Per-payload default row templates ({name: read-only array of
        # per-element shape}): the reset value used wherever a payload row is
        # (re)initialized (insert recycle, edge free, replica reset).
        self._node_payload_defaults = {n: spec[2]
                                       for n, spec in layout.node_payloads.items()}
        self._edge_payload_defaults = {n: spec[2]
                                       for n, spec in layout.edge_payloads.items()}
        # Hot-path aliases (same underlying buffers as the dicts above).
        self._node_x = node_core['x']
        self._node_y = node_core['y']
        self._node_active = node_core['active']
        self._node_generation = node_core['generation']
        self._node_update_version = node_core['update_version']
        self._node_first_edge = node_core['first_edge']
        self._node_next_in_bucket = node_core['next_in_bucket']
        self._edge_target = edge_core['target']
        self._edge_next = edge_core['next']
        self._edge_length = edge_core['length']

    def _drop_views(self):
        """Release every array reference so SHM handles can be closed."""
        for attr in ('_meta', '_node_core', '_edge_core', '_bucket_head',
                     '_node_payloads', '_edge_payloads',
                     '_node_x', '_node_y', '_node_active', '_node_generation',
                     '_node_update_version', '_node_first_edge',
                     '_node_next_in_bucket',
                     '_edge_target', '_edge_next', '_edge_length'):
            setattr(self, attr, None)

    @staticmethod
    def _readonly(arr: np.ndarray) -> np.ndarray:
        view = arr.view()
        view.flags.writeable = False
        return view

    # --- scalar metadata -------------------------------------------------

    @property
    def graph_seq(self) -> int:
        return int(self._meta[0]['graph_seq'])

    @property
    def n_max(self) -> int:
        return int(self._meta[0]['n_max'])

    @property
    def e_max(self) -> int:
        return int(self._meta[0]['e_max'])

    @property
    def node_hwm(self) -> int:
        """Exclusive node high-water mark: used prefix is [0, node_hwm)."""
        return int(self._meta[0]['node_hwm'])

    @property
    def edge_hwm(self) -> int:
        """Exclusive edge high-water mark: used prefix is [0, edge_hwm)."""
        return int(self._meta[0]['edge_hwm'])

    @property
    def node_free_head(self) -> int:
        return int(self._meta[0]['node_free_head'])

    @property
    def edge_free_head(self) -> int:
        return int(self._meta[0]['edge_free_head'])

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        m = self._meta[0]
        return (float(m['xmin']), float(m['xmax']),
                float(m['ymin']), float(m['ymax']))

    @property
    def bucket_size(self) -> float:
        return float(self._meta[0]['bucket_size'])

    @property
    def default_lookup_radius(self) -> float:
        return float(self._meta[0]['default_lookup_radius'])

    @property
    def layout(self) -> GraphLayout:
        return self._layout

    # --- read-only array views -------------------------------------------

    @property
    def node_x(self):
        return self._readonly(self._node_x)

    @property
    def node_y(self):
        return self._readonly(self._node_y)

    @property
    def node_active(self):
        return self._readonly(self._node_active)

    @property
    def node_generation(self):
        return self._readonly(self._node_generation)

    @property
    def node_update_version(self):
        return self._readonly(self._node_update_version)

    @property
    def node_first_edge(self):
        return self._readonly(self._node_first_edge)

    @property
    def node_next_in_bucket(self):
        return self._readonly(self._node_next_in_bucket)

    @property
    def edge_target(self):
        return self._readonly(self._edge_target)

    @property
    def edge_next(self):
        return self._readonly(self._edge_next)

    @property
    def edge_length(self):
        return self._readonly(self._edge_length)

    @property
    def bucket_head(self):
        return self._readonly(self._bucket_head)

    def node_payload(self, name: str) -> np.ndarray:
        """Read-only view of a node payload array, shape (n_max, *per_element)."""
        try:
            return self._readonly(self._node_payloads[name])
        except KeyError:
            raise KeyError(f"unknown node payload {name!r}; available: "
                           f"{sorted(self._node_payloads)}") from None

    def edge_payload(self, name: str) -> np.ndarray:
        """Read-only view of an edge payload array, shape (e_max, *per_element)."""
        try:
            return self._readonly(self._edge_payloads[name])
        except KeyError:
            raise KeyError(f"unknown edge payload {name!r}; available: "
                           f"{sorted(self._edge_payloads)}") from None

    def validate_ref(self, slot: int, generation: int) -> bool:
        """
        Intra-cycle debug helper only (see design: slot-lifetime rule). True
        iff the slot is live and still holds the same-identity node. No
        correctness may depend on this across planning cycles.
        """
        slot = int(slot)
        if slot < 0 or slot >= self.n_max:
            return False
        return (int(self._node_active[slot]) == 1
                and int(self._node_generation[slot]) == int(generation))


# ---------------------------------------------------------------------------
# The one shared mutation engine
# ---------------------------------------------------------------------------

class _GraphEngine(_GraphStateView):
    """
    Every structural and payload mutation, implemented once over the bound
    array bundle. SharedGraphWriter binds SHM arrays + a seqlock guard;
    GraphSnapshot binds local arrays + a no-op guard. All normal validation
    finishes BEFORE the first write; multi-edge operations preflight every
    required edge and resource first (failure atomicity).
    """

    def _bind_state(self, meta, node_core, edge_core, bucket_head,
                    node_payloads, edge_payloads, layout, edit_guard,
                    is_replica: bool):
        self._bind_views(meta, node_core, edge_core, bucket_head,
                         node_payloads, edge_payloads, layout)
        self._edit_guard = edit_guard
        self._is_replica = is_replica

    # ------------------------------------------------------------------
    # Public batch guard
    # ------------------------------------------------------------------

    def edit_batch(self):
        """
        Reentrant edit window. On the writer this holds graph_seq odd for the
        whole (possibly nested) batch; public mutators called inside do not
        touch the sequence at inner depths. On a snapshot replica it is a
        no-op. Keep writer batches short (one map-update's worth) so reader
        snapshot retries stay rare and bounded.
        """
        return self._edit_guard

    # ------------------------------------------------------------------
    # Validation helpers (never write)
    # ------------------------------------------------------------------

    def _pool_context(self) -> str:
        return "snapshot-replica local pool" if self._is_replica else "shared writer pool"

    def _require_node_slot(self, slot, op: str) -> int:
        if isinstance(slot, bool) or not isinstance(slot, (int, np.integer)):
            raise ValueError(f"{op}: node slot must be an integer, got {slot!r}")
        slot = int(slot)
        if slot < 0 or slot >= self.n_max:
            raise ValueError(f"{op}: node slot {slot} out of range [0, {self.n_max})")
        return slot

    def _require_active_node(self, slot, op: str) -> int:
        slot = self._require_node_slot(slot, op)
        if int(self._node_active[slot]) != 1:
            raise ValueError(
                f"{op}: node slot {slot} is not active (free or deleted); "
                f"node_hwm={self.node_hwm}, generation={int(self._node_generation[slot])}")
        return slot

    def _validate_position(self, x, y, op: str) -> Tuple[float, float]:
        """
        Validate finite, in-bounds coordinates and return them rounded to the
        float32 values that will actually be stored. All bucket and length
        math uses the stored-float32 values so recomputation (e.g. bucket
        lookup at delete time) can never disagree with insertion.
        """
        try:
            xf64, yf64 = float(x), float(y)
        except (TypeError, ValueError):
            raise ValueError(f"{op}: coordinates must be scalars, got ({x!r}, {y!r})")
        if not (math.isfinite(xf64) and math.isfinite(yf64)):
            raise ValueError(f"{op}: non-finite coordinates ({xf64}, {yf64})")
        xmin, xmax, ymin, ymax = self.bounds
        if not (xmin <= xf64 <= xmax and ymin <= yf64 <= ymax):
            raise ValueError(
                f"{op}: position ({xf64}, {yf64}) outside world bounds "
                f"x[{xmin}, {xmax}] y[{ymin}, {ymax}]")
        return float(np.float32(xf64)), float(np.float32(yf64))

    def _bucket_of(self, xf: float, yf: float) -> Tuple[int, int]:
        """
        Bucket cell of a stored position. Indices are unconditionally clamped
        into the grid: validated in-bounds coordinates can only fall outside
        by the exact-max-bound cell or float32 rounding at the border, and
        clamping keeps both in the last valid cell.
        """
        m = self._meta[0]
        b = float(m['bucket_size'])
        n_bx, n_by = int(m['n_bx']), int(m['n_by'])
        bx = int(math.floor((xf - float(m['xmin'])) / b))
        by = int(math.floor((yf - float(m['ymin'])) / b))
        return (min(max(bx, 0), n_bx - 1), min(max(by, 0), n_by - 1))

    def _walk_bucket_chain(self, bx: int, by: int) -> Iterator[int]:
        """Yield node slots of one bucket chain, guarding cycles/bad indices."""
        n_max = self.n_max
        s = int(self._bucket_head[bx, by])
        steps = 0
        while s != -1:
            if s < 0 or s >= n_max:
                raise GraphCorruptionError(
                    f"bucket({bx},{by}) chain holds out-of-range slot {s} (n_max={n_max})")
            yield s
            s = int(self._node_next_in_bucket[s])
            steps += 1
            if steps > n_max:
                raise GraphCorruptionError(f"bucket({bx},{by}) chain has a cycle")

    def _walk_edge_chain(self, head: int, what: str) -> Iterator[int]:
        """Yield edge rows of one adjacency chain, guarding cycles/bad indices."""
        e_max = self.e_max
        e = int(head)
        steps = 0
        while e != -1:
            if e < 0 or e >= e_max:
                raise GraphCorruptionError(
                    f"{what}: chain holds out-of-range edge row {e} (e_max={e_max})")
            yield e
            e = int(self._edge_next[e])
            steps += 1
            if steps > e_max:
                raise GraphCorruptionError(f"{what}: adjacency chain has a cycle")

    def _find_directed_edge(self, owner: int, target: int) -> Optional[int]:
        """Edge row owner->target, or None if absent."""
        for e in self._walk_edge_chain(int(self._node_first_edge[owner]),
                                       f"node {owner} adjacency"):
            if int(self._edge_target[e]) == target:
                return e
        return None

    def _require_free_edge_rows(self, k: int, op: str):
        """
        Verify >= k free edge rows by walking at most k free-list links.
        Never mutates the free list — this is the all-or-nothing gate for
        multi-edge operations.
        """
        cur = self.edge_free_head
        for i in range(k):
            if cur == -1:
                raise GraphCapacityError(
                    f"{op}: needs {k} free edge rows but only {i} available in the "
                    f"{self._pool_context()} (e_max={self.e_max}, edge_hwm={self.edge_hwm})"
                    + ("; re-snapshot with a larger edge_headroom" if self._is_replica else ""))
            if cur < 0 or cur >= self.e_max:
                raise GraphCorruptionError(f"{op}: edge free list holds out-of-range row {cur}")
            if int(self._edge_target[cur]) != -1:
                raise GraphCorruptionError(
                    f"{op}: edge free-list row {cur} has target="
                    f"{int(self._edge_target[cur])} (expected -1 free sentinel)")
            cur = int(self._edge_next[cur])

    # --- payload validation -------------------------------------------

    @staticmethod
    def _cast_payload_value(kind: str, name: str, value, expected_shape: tuple,
                            dt: np.dtype) -> np.ndarray:
        try:
            arr = np.asarray(value)
        except Exception as exc:
            raise ValueError(f"{kind} payload {name!r}: cannot convert value to array: {exc}")
        if arr.dtype.hasobject:
            raise ValueError(f"{kind} payload {name!r}: value is not numeric")
        if arr.dtype.kind in 'iu' and dt.kind in 'iu':
            # integer -> integer: any width/signedness is fine as long as the
            # actual values fit the target range (fail loud on wraparound
            # instead of silently truncating, e.g. 300 -> uint8).
            info = np.iinfo(dt)
            if arr.size and (int(arr.min()) < info.min or int(arr.max()) > info.max):
                raise ValueError(
                    f"{kind} payload {name!r}: value(s) outside {dt} range "
                    f"[{info.min}, {info.max}]: min={arr.min()}, max={arr.max()}")
        elif not np.can_cast(arr.dtype, dt, casting='same_kind'):
            raise ValueError(
                f"{kind} payload {name!r}: value dtype {arr.dtype} does not safely "
                f"cast to {dt} (same_kind)")
        if arr.shape != expected_shape:
            raise ValueError(
                f"{kind} payload {name!r}: value shape {arr.shape} != expected "
                f"{expected_shape}")
        return arr.astype(dt, copy=False)

    def _prepare_payload_writes(self, kind: str, spec_map, array_map,
                                indexed, full, op: str,
                                allow_empty: bool = False) -> List[tuple]:
        """
        Validate a payload-update request completely before any write.
        Returns [(target_array, index_tuple_within_row, cast_value), ...].
        `indexed` maps name -> (index, value); `full` maps name -> full row
        value of exactly per_element_shape (unambiguous by construction: a
        full update must match the whole per-element shape).
        """
        indexed = dict(indexed) if indexed else {}
        full = dict(full) if full else {}
        overlap = set(indexed) & set(full)
        if overlap:
            raise ValueError(
                f"{op}: fields {sorted(overlap)} appear in both indexed and full updates")
        if not indexed and not full and not allow_empty:
            raise ValueError(f"{op}: empty update (no indexed and no full fields)")

        writes: List[tuple] = []
        for name, value in full.items():
            if name not in spec_map:
                raise KeyError(f"{op}: unknown {kind} payload {name!r}; "
                               f"available: {sorted(spec_map)}")
            shape, dt, _ = spec_map[name]
            writes.append((array_map[name], (),
                           self._cast_payload_value(kind, name, value, shape, dt)))
        for name, entry in indexed.items():
            if name not in spec_map:
                raise KeyError(f"{op}: unknown {kind} payload {name!r}; "
                               f"available: {sorted(spec_map)}")
            shape, dt, _ = spec_map[name]
            if shape == ():
                raise ValueError(
                    f"{op}: {kind} payload {name!r} is scalar per element and "
                    f"cannot take an indexed update; pass it as a full field")
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise ValueError(
                    f"{op}: indexed entry for {name!r} must be (index, value), got {entry!r}")
            index, value = entry
            if isinstance(index, (int, np.integer)) and not isinstance(index, bool):
                idx = (int(index),)
            else:
                try:
                    idx = tuple(int(i) for i in index)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"{op}: index for {name!r} must be an int or tuple of ints, "
                        f"got {index!r}")
            if len(idx) > len(shape):
                raise ValueError(
                    f"{op}: index {idx} for {name!r} has more dims than payload "
                    f"shape {shape}")
            for axis, i in enumerate(idx):
                if i < 0 or i >= shape[axis]:
                    raise ValueError(
                        f"{op}: index {idx} for {name!r} out of range for shape {shape} "
                        f"(axis {axis}: 0 <= {i} < {shape[axis]} violated)")
            expected = shape[len(idx):]
            writes.append((array_map[name], idx,
                           self._cast_payload_value(kind, name, value, expected, dt)))
        return writes

    # ------------------------------------------------------------------
    # Internal allocation / linked-list primitives (call inside the guard)
    # ------------------------------------------------------------------

    def _pop_free_node(self) -> int:
        s = self.node_free_head
        if s == -1:
            # Callers preflight this; kept as an internal last-line guard.
            raise GraphCapacityError(f"node pool exhausted in the {self._pool_context()}")
        if s < 0 or s >= self.n_max:
            raise GraphCorruptionError(f"node free list holds out-of-range slot {s}")
        self._meta[0]['node_free_head'] = int(self._node_next_in_bucket[s])
        return s

    def _pop_free_edge(self) -> int:
        e = self.edge_free_head
        if e == -1:
            raise GraphCapacityError(f"edge pool exhausted in the {self._pool_context()}")
        if e < 0 or e >= self.e_max:
            raise GraphCorruptionError(f"edge free list holds out-of-range row {e}")
        if int(self._edge_target[e]) != -1:
            raise GraphCorruptionError(
                f"edge free-list row {e} not marked free (target != -1)")
        self._meta[0]['edge_free_head'] = int(self._edge_next[e])
        if e >= self.edge_hwm:
            self._meta[0]['edge_hwm'] = e + 1   # first use of a fresh row
        return e

    def _free_edge_row(self, e: int):
        """Canonical edge free: reset payloads to their layout defaults and
        length to 0, -1 sentinel, push on free list."""
        for name, arr in self._edge_payloads.items():
            arr[e] = self._edge_payload_defaults[name]
        self._edge_length[e] = 0.0
        self._edge_target[e] = -1
        self._edge_next[e] = self.edge_free_head
        self._meta[0]['edge_free_head'] = e

    def _unlink_edge_from_chain(self, owner: int, e: int, op: str):
        """Remove edge row e from owner's adjacency chain (walk-to-predecessor)."""
        head = int(self._node_first_edge[owner])
        if head == e:
            self._node_first_edge[owner] = int(self._edge_next[e])
            return
        prev = None
        for cur in self._walk_edge_chain(head, f"{op}: node {owner} adjacency"):
            if cur == e:
                self._edge_next[prev] = int(self._edge_next[e])
                return
            prev = cur
        raise GraphCorruptionError(
            f"{op}: edge row {e} not found in node {owner}'s adjacency chain")

    def _unlink_node_from_bucket(self, slot: int, bx: int, by: int, op: str):
        """Remove a node from its bucket chain (walk-to-predecessor)."""
        head = int(self._bucket_head[bx, by])
        if head == slot:
            self._bucket_head[bx, by] = int(self._node_next_in_bucket[slot])
            return
        prev = None
        for cur in self._walk_bucket_chain(bx, by):
            if cur == slot:
                self._node_next_in_bucket[prev] = int(self._node_next_in_bucket[slot])
                return
            prev = cur
        raise GraphCorruptionError(
            f"{op}: node {slot} not found in bucket({bx},{by}) chain")

    def _link_edge_pair(self, a: int, b: int, payload_writes: List[tuple]) -> Tuple[int, int]:
        """
        Allocate + fully initialize + splice one undirected edge pair. Call
        only inside the guard, after preflight guaranteed two free rows.
        Both rows are completely initialized (target, length, payloads)
        BEFORE being spliced into the adjacency chains.
        """
        e_ab = self._pop_free_edge()
        e_ba = self._pop_free_edge()
        # Immutable geometric length from the stored float32 endpoints.
        length = np.float32(math.hypot(float(self._node_x[a]) - float(self._node_x[b]),
                                       float(self._node_y[a]) - float(self._node_y[b])))
        for e in (e_ab, e_ba):
            for arr, idx, value in payload_writes:   # mirrored payloads
                arr[(e,) + idx] = value
            self._edge_length[e] = length
        self._edge_target[e_ab] = b
        self._edge_target[e_ba] = a
        # Splice LAST: a reader-visible chain never reaches a half-built row.
        self._edge_next[e_ab] = int(self._node_first_edge[a])
        self._node_first_edge[a] = e_ab
        self._edge_next[e_ba] = int(self._node_first_edge[b])
        self._node_first_edge[b] = e_ba
        return e_ab, e_ba

    def _remove_incident_pairs(self, slot: int, pairs: List[Tuple[int, int]], op: str):
        """
        Unlink and free every preflighted incident edge pair of `slot` (call
        inside the guard). The reverse edge is re-located at mutation time so
        neighbor-chain predecessors are always current. Freed slot-owned rows
        may linger in slot's own chain until the caller rewrites first_edge —
        invisible outside the batch.
        """
        for e, t in pairs:
            rev = self._find_directed_edge(t, slot)
            if rev is None:
                raise GraphCorruptionError(
                    f"{op}: reverse edge {t}->{slot} vanished mid-removal")
            self._unlink_edge_from_chain(t, rev, op)
            self._free_edge_row(rev)
            self._free_edge_row(e)

    def _preflight_incident_pairs(self, slot: int, op: str) -> List[Tuple[int, int]]:
        """
        Validate every incident directed edge of `slot` before any write:
        in-range active target, no self-loop, no duplicate target, and a
        reverse edge present in the target's chain. Returns [(edge, target)].
        """
        pairs: List[Tuple[int, int]] = []
        seen_targets = set()
        for e in self._walk_edge_chain(int(self._node_first_edge[slot]),
                                       f"{op}: node {slot} adjacency"):
            t = int(self._edge_target[e])
            if t < 0 or t >= self.n_max:
                raise GraphCorruptionError(
                    f"{op}: edge {e} of node {slot} targets out-of-range slot {t}")
            if t == slot:
                raise GraphCorruptionError(f"{op}: node {slot} has a self-loop edge {e}")
            if t in seen_targets:
                raise GraphCorruptionError(
                    f"{op}: node {slot} has duplicate edges toward node {t}")
            seen_targets.add(t)
            if int(self._node_active[t]) != 1:
                raise GraphCorruptionError(
                    f"{op}: edge {e} of node {slot} targets inactive node {t}")
            if self._find_directed_edge(t, slot) is None:
                raise GraphCorruptionError(
                    f"{op}: edge {e} ({slot}->{t}) has no reverse edge {t}->{slot}; "
                    f"edge pair is half-broken")
            pairs.append((e, t))
        return pairs

    # ------------------------------------------------------------------
    # Public mutators
    # ------------------------------------------------------------------

    def insert_node(self, x, y, **payload) -> int:
        """
        Allocate a node at (x, y) with optional full-row payload values.
        Returns the node slot. Ordering follows the design: the slot becomes
        active LAST, so a concurrent reader can never see a half-initialized
        node as live.
        """
        op = 'insert_node'
        # ---- preflight: coordinates, payloads, capacity, bucket (no writes)
        xf, yf = self._validate_position(x, y, op)
        writes = self._prepare_payload_writes(
            'node', self._layout.node_payloads, self._node_payloads,
            None, payload, op, allow_empty=True)
        if self.node_free_head == -1:
            raise GraphCapacityError(
                f"{op}: node pool exhausted in the {self._pool_context()} "
                f"(n_max={self.n_max}, node_hwm={self.node_hwm})"
                + ("; re-snapshot with a larger node_headroom" if self._is_replica else ""))
        bx, by = self._bucket_of(xf, yf)

        with self._edit_guard:
            s = self._pop_free_node()
            # Canonical recycle reset: a recycled slot must not leak stale
            # adjacency or payload data. generation is NOT reset (identity
            # counter) and update_version is NEVER reset (monotonic data
            # counter for the replan gate).
            self._node_active[s] = 0
            self._node_first_edge[s] = -1
            for name, arr in self._node_payloads.items():
                arr[s] = self._node_payload_defaults[name]
            # Position + caller payloads.
            self._node_x[s] = xf
            self._node_y[s] = yf
            for arr, idx, value in writes:
                arr[(s,) + idx] = value
            # Bucket chain push (head insertion).
            self._node_next_in_bucket[s] = int(self._bucket_head[bx, by])
            self._bucket_head[bx, by] = s
            # One data-version bump per insert.
            self._node_update_version[s] += 1
            # Activate LAST.
            self._node_active[s] = 1
            if s >= self.node_hwm:
                self._meta[0]['node_hwm'] = s + 1   # fresh row first used
        return int(s)

    def delete_node(self, slot: int):
        """
        Delete an active node and all incident edge pairs (O(degree)).
        Ordering follows the design: active = 0 FIRST so the node is
        instantly invisible to bucket and adjacency walks.
        """
        op = 'delete_node'
        slot = self._require_active_node(slot, op)
        # ---- preflight (no writes): bucket membership + reverse edges
        bx, by = self._bucket_of(float(self._node_x[slot]), float(self._node_y[slot]))
        occurrences = sum(1 for s in self._walk_bucket_chain(bx, by) if s == slot)
        if occurrences != 1:
            raise GraphCorruptionError(
                f"{op}: node {slot} appears {occurrences} times in bucket({bx},{by}) "
                f"chain (expected exactly 1)")
        pairs = self._preflight_incident_pairs(slot, op)

        with self._edit_guard:
            self._node_active[slot] = 0                       # invisible FIRST
            self._remove_incident_pairs(slot, pairs, op)
            # Bucket unlink must precede the free-list push below, which
            # repurposes next_in_bucket[slot] as the free-list link.
            self._unlink_node_from_bucket(slot, bx, by, op)
            self._node_first_edge[slot] = -1
            self._node_generation[slot] += 1                  # identity bump
            self._node_next_in_bucket[slot] = self.node_free_head
            self._meta[0]['node_free_head'] = slot

    def add_edge(self, slot_a: int, slot_b: int, **payload) -> Tuple[int, int]:
        """
        Insert one undirected edge as two directed rows with mirrored
        payloads and auto-computed immutable length. Returns (e_ab, e_ba).
        """
        op = 'add_edge'
        a = self._require_active_node(slot_a, op)
        b = self._require_active_node(slot_b, op)
        if a == b:
            raise ValueError(f"{op}: self-edge on node {a} is not allowed")
        fwd = self._find_directed_edge(a, b)
        rev = self._find_directed_edge(b, a)
        if fwd is not None and rev is not None:
            raise ValueError(f"{op}: edge pair {a}<->{b} already exists "
                             f"(rows {fwd} and {rev})")
        if (fwd is None) != (rev is None):
            raise GraphCorruptionError(
                f"{op}: half-broken pair between {a} and {b}: forward row "
                f"{fwd}, reverse row {rev}")
        writes = self._prepare_payload_writes(
            'edge', self._layout.edge_payloads, self._edge_payloads,
            None, payload, op, allow_empty=True)
        self._require_free_edge_rows(2, op)     # all-or-nothing gate

        with self._edit_guard:
            e_ab, e_ba = self._link_edge_pair(a, b, writes)
        return e_ab, e_ba

    def remove_edge(self, slot_a: int, slot_b: int):
        """Remove the undirected edge pair between two active nodes."""
        op = 'remove_edge'
        a = self._require_active_node(slot_a, op)
        b = self._require_active_node(slot_b, op)
        if a == b:
            raise ValueError(f"{op}: self-edge on node {a} cannot exist")
        # ---- preflight both directed rows before unlinking either
        fwd = self._find_directed_edge(a, b)
        rev = self._find_directed_edge(b, a)
        if fwd is None and rev is None:
            raise ValueError(f"{op}: no edge pair between nodes {a} and {b}")
        if fwd is None or rev is None:
            raise GraphCorruptionError(
                f"{op}: half-broken pair between {a} and {b}: forward row "
                f"{fwd}, reverse row {rev}")

        with self._edit_guard:
            self._unlink_edge_from_chain(a, fwd, op)
            self._unlink_edge_from_chain(b, rev, op)
            self._free_edge_row(fwd)
            self._free_edge_row(rev)

    def remove_edges_of_node(self, slot: int) -> int:
        """
        Remove every incident edge pair, leaving the node active, in its
        bucket, and degree zero (detached nodes are valid). Returns the
        number of removed pairs.
        """
        op = 'remove_edges_of_node'
        slot = self._require_active_node(slot, op)
        pairs = self._preflight_incident_pairs(slot, op)

        with self._edit_guard:
            self._remove_incident_pairs(slot, pairs, op)
            self._node_first_edge[slot] = -1
        return len(pairs)

    def connect_node(self, slot: int, radius, **payload) -> List[Tuple[int, int]]:
        """
        Wire an existing active node to every active in-radius node it is not
        already connected to (mirrored payloads on every created pair).
        All-or-nothing: if fewer than 2*k edge rows are free for the k needed
        pairs, nothing is mutated. Returns the created (e_ab, e_ba) pairs.
        """
        op = 'connect_node'
        slot = self._require_active_node(slot, op)
        try:
            r = float(radius)
        except (TypeError, ValueError):
            raise ValueError(f"{op}: radius must be a scalar, got {radius!r}")
        if not math.isfinite(r) or r < 0.0:
            raise ValueError(f"{op}: radius must be finite and >= 0, got {r}")
        writes = self._prepare_payload_writes(
            'edge', self._layout.edge_payloads, self._edge_payloads,
            None, payload, op, allow_empty=True)
        # ---- candidate set: active in-radius, minus self and existing
        candidates = self.radius_query(float(self._node_x[slot]),
                                       float(self._node_y[slot]), r)
        existing = {t for t, _ in self.neighbors(slot)}
        chosen = [int(c) for c in candidates if int(c) != slot and int(c) not in existing]
        # ---- all-or-nothing capacity gate (walks <= 2k links, no mutation)
        self._require_free_edge_rows(2 * len(chosen), op)

        created: List[Tuple[int, int]] = []
        with self._edit_guard:
            for c in chosen:
                created.append(self._link_edge_pair(slot, c, writes))
        return created

    # ------------------------------------------------------------------
    # Payload updates (seq-guarded like every mutation)
    # ------------------------------------------------------------------

    def update_node_payload(self, slot: int, indexed=None, **full):
        """
        Update node payload fields. `indexed` maps name -> (index, value)
        for per-layer writes, e.g. indexed={"cmd_v": (layer, value)}; keyword
        fields carry full per-element rows. The two forms cannot overlap and
        an empty update raises, so an indexed update can never be confused
        with a legitimate full payload of shape (2,). One successful call
        increments update_version exactly once.
        """
        op = 'update_node_payload'
        slot = self._require_active_node(slot, op)
        writes = self._prepare_payload_writes(
            'node', self._layout.node_payloads, self._node_payloads,
            indexed, full, op)
        with self._edit_guard:
            for arr, idx, value in writes:
                arr[(slot,) + idx] = value
            self._node_update_version[slot] += 1

    def update_edge_payload(self, edge: int, indexed=None, **full):
        """Update payload fields of one allocated directed edge row."""
        op = 'update_edge_payload'
        if isinstance(edge, bool) or not isinstance(edge, (int, np.integer)):
            raise ValueError(f"{op}: edge row must be an integer, got {edge!r}")
        edge = int(edge)
        if edge < 0 or edge >= self.e_max:
            raise ValueError(f"{op}: edge row {edge} out of range [0, {self.e_max})")
        if int(self._edge_target[edge]) < 0:
            raise ValueError(
                f"{op}: edge row {edge} is free (target == -1); only allocated "
                f"edges can carry payload updates")
        writes = self._prepare_payload_writes(
            'edge', self._layout.edge_payloads, self._edge_payloads,
            indexed, full, op)
        with self._edit_guard:
            for arr, idx, value in writes:
                arr[(edge,) + idx] = value

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def neighbors(self, slot: int) -> Iterator[Tuple[int, int]]:
        """
        Iterate (target_slot, edge_row) over the out-edge chain of an active
        node (newest-first chain order). Mutating the graph while iterating
        is undefined; materialize with list() first if needed.
        """
        slot = self._require_active_node(slot, 'neighbors')

        def _iter():
            for e in self._walk_edge_chain(int(self._node_first_edge[slot]),
                                           f"neighbors of node {slot}"):
                yield int(self._edge_target[e]), e
        return _iter()

    def radius_query(self, x, y, r) -> np.ndarray:
        """
        Active node slots within distance r of (x, y), as a sorted int32
        array. Scans ceil(r / bucket_size) buckets each way around the
        (clamped) center cell, clips at grid borders, and exact-filters by
        squared distance; complete because bucket_size-sized cells bound the
        per-axis displacement.
        """
        op = 'radius_query'
        try:
            xq, yq, rq = float(x), float(y), float(r)
        except (TypeError, ValueError):
            raise ValueError(f"{op}: x, y, r must be scalars, got ({x!r}, {y!r}, {r!r})")
        if not (math.isfinite(xq) and math.isfinite(yq) and math.isfinite(rq)):
            raise ValueError(f"{op}: non-finite query ({xq}, {yq}, r={rq})")
        if rq < 0.0:
            raise ValueError(f"{op}: negative radius {rq}")
        m = self._meta[0]
        b = float(m['bucket_size'])
        n_bx, n_by = int(m['n_bx']), int(m['n_by'])
        # Clamped center cell: exact max-bound coordinates land in the last
        # bucket, matching node placement; out-of-bounds centers stay complete
        # because the scan window is only ever widened by the clamp.
        cbx = min(max(int(math.floor((xq - float(m['xmin'])) / b)), 0), n_bx - 1)
        cby = min(max(int(math.floor((yq - float(m['ymin'])) / b)), 0), n_by - 1)
        d = int(math.ceil(rq / b)) if rq > 0.0 else 0
        r2 = rq * rq
        found: List[int] = []
        for bx in range(max(0, cbx - d), min(n_bx - 1, cbx + d) + 1):
            for by in range(max(0, cby - d), min(n_by - 1, cby + d) + 1):
                for s in self._walk_bucket_chain(bx, by):
                    if int(self._node_active[s]) != 1:
                        continue   # mid-batch transient or corrupt entry
                    dx = float(self._node_x[s]) - xq
                    dy = float(self._node_y[s]) - yq
                    if dx * dx + dy * dy <= r2:
                        found.append(s)
        return np.array(sorted(found), dtype=np.int32)

    # ------------------------------------------------------------------
    # World-coordinate frontend (Part 1b)
    #
    # Callers work with physical x, y, theta; slots appear only as returned
    # handles. Implemented on the shared engine so every frontend exists on
    # both the writer (mutates SHM) and GraphSnapshot (mutates its private
    # copy). Uniform resolution semantics:
    #   - a position resolves to the NEAREST ACTIVE node within a radius;
    #     tol/max_radius = None selects the instance default_lookup_radius,
    #     an explicit value is used literally;
    #   - ties resolve deterministically by minimizing (squared_distance,
    #     slot);
    #   - LOOKUPS return -1/None on a miss; MUTATORS raise with the query
    #     point, effective tolerance, and nearest active-node distance
    #     (silently mutating nothing is not allowed);
    #   - a direction resolves to the incident edge whose geometric heading
    #     best matches theta under wrapped angular error, tie-broken by
    #     (angle_error, target_slot, edge_index) independent of chain order.
    # These are thin: resolve position/direction -> slot(s) -> delegate to the
    # Part 1a backend. No new storage and no new persistent invariants.
    # ------------------------------------------------------------------

    def _effective_tol(self, tol) -> float:
        """None -> instance default_lookup_radius; else the literal value."""
        return self.default_lookup_radius if tol is None else float(tol)

    # --- lookups (never raise on a resolution miss) -------------------

    def nearest_node(self, x, y, max_radius=None) -> int:
        """
        Nearest ACTIVE node slot within max_radius of (x, y), or -1 on a miss.
        max_radius=None uses default_lookup_radius. radius_query returns slots
        in ascending order, so np.argmin (first minimum) reproduces the
        (squared_distance, slot) tie-break: the smallest slot wins on exact
        midpoints.
        """
        r = self._effective_tol(max_radius)
        # radius_query validates finite coordinates and a finite, non-negative
        # radius (raises), so invalid inputs raise here too (not "misses").
        cand = self.radius_query(x, y, r)          # (k,) int32, ascending slots
        if cand.size == 0:
            return -1
        xq, yq = float(x), float(y)
        dx = self._node_x[cand].astype(np.float64) - xq   # (k,) squared-dist parts
        dy = self._node_y[cand].astype(np.float64) - yq
        d2 = dx * dx + dy * dy
        return int(cand[int(np.argmin(d2))])

    def nodes_in_radius(self, x, y, r) -> np.ndarray:
        """All active node slots within r of (x, y) — alias of radius_query."""
        return self.radius_query(x, y, r)

    def neighbors_at(self, x, y, tol=None) -> np.ndarray:
        """
        Adjacent node slots of the node nearest (x, y), sorted ascending.
        Returns an empty int32 array on a resolution miss (lookup, no raise).
        """
        s = self.nearest_node(x, y, tol)
        if s == -1:
            return np.empty(0, dtype=np.int32)
        return np.array(sorted(int(t) for t, _ in self.neighbors(s)), dtype=np.int32)

    def find_edge(self, pos_a, pos_b, tol=None):
        """
        Resolve both endpoints and return the directed pair (e_ab, e_ba) where
        e_ab is the row a->b (endpoint ORDER defines which is e_ab). None if
        either endpoint misses, both resolve to the same node, or no edge
        connects them.
        """
        a = self.nearest_node(pos_a[0], pos_a[1], tol)
        b = self.nearest_node(pos_b[0], pos_b[1], tol)
        if a == -1 or b == -1 or a == b:
            return None
        fwd = self._find_directed_edge(a, b)
        rev = self._find_directed_edge(b, a)
        if fwd is None or rev is None:
            return None
        return (fwd, rev)

    @staticmethod
    def _validate_direction(theta, angle_tol, op) -> Tuple[float, float]:
        """Validate finite theta and angle_tol in [0, pi]; return them as floats."""
        try:
            th = float(theta)
            atol = float(angle_tol)
        except (TypeError, ValueError):
            raise ValueError(f"{op}: theta and angle_tol must be scalars, got "
                             f"theta={theta!r}, angle_tol={angle_tol!r}")
        if not math.isfinite(th):
            raise ValueError(f"{op}: theta must be finite, got {theta!r}")
        if not math.isfinite(atol) or not (0.0 <= atol <= math.pi):
            raise ValueError(f"{op}: angle_tol must be finite in [0, pi], got {angle_tol!r}")
        return th, atol

    def _select_edge_toward(self, src, theta, atol, op):
        """
        Incident edge of active `src` whose heading best matches theta. Wrapped
        angular error abs(atan2(sin(dtheta), cos(dtheta))); ties broken by
        minimizing (angle_error, target_slot, edge_index) so the result is
        independent of adjacency-chain order. Returns (edge_index, src, target)
        or None if no incident edge lies within atol.
        """
        sx, sy = float(self._node_x[src]), float(self._node_y[src])
        best = None   # (angle_error, target_slot, edge_index)
        for t, e in self.neighbors(src):
            # geometric heading of the edge from src to its target
            etheta = math.atan2(float(self._node_y[t]) - sy,
                                float(self._node_x[t]) - sx)
            err = abs(math.atan2(math.sin(etheta - theta), math.cos(etheta - theta)))
            if err <= atol:
                key = (err, int(t), int(e))
                if best is None or key < best:
                    best = key
        if best is None:
            return None
        return (best[2], int(src), best[1])

    def edge_toward(self, x, y, theta, angle_tol, tol=None):
        """
        Incident edge of the node nearest (x, y) whose heading best matches
        theta. Returns (edge_index, slot_src, slot_dst) or None on a node miss
        or when no incident edge lies within angle_tol. Direction arguments are
        validated even when the position misses.
        """
        op = 'edge_toward'
        th, atol = self._validate_direction(theta, angle_tol, op)
        s = self.nearest_node(x, y, tol)
        if s == -1:
            return None
        return self._select_edge_toward(s, th, atol, op)

    # --- mutator resolution (raise on a miss) -------------------------

    def _nearest_active(self, x, y):
        """
        (distance, slot) of the nearest active node over [0, node_hwm), or
        (None, -1) when the graph has no active node. Error-path helper only.
        """
        nh = self.node_hwm
        if nh == 0:
            return None, -1
        active = np.flatnonzero(np.asarray(self._node_active[:nh]) == 1)
        if active.size == 0:
            return None, -1
        dx = self._node_x[active].astype(np.float64) - float(x)
        dy = self._node_y[active].astype(np.float64) - float(y)
        d2 = dx * dx + dy * dy
        i = int(np.argmin(d2))
        return math.sqrt(float(d2[i])), int(active[i])

    def _resolve_mutator_node(self, x, y, tol, op) -> int:
        """
        Resolve (x, y) to the nearest active node within the effective
        tolerance for a mutator, raising an actionable error on a miss that
        includes the query point, effective tolerance, and nearest active-node
        distance.
        """
        r = self._effective_tol(tol)
        s = self.nearest_node(x, y, r)   # validates finite coords + radius
        if s == -1:
            dist, near = self._nearest_active(x, y)
            if near == -1:
                detail = "no active nodes exist in the graph"
            else:
                detail = f"nearest active node is {near} at distance {dist:.6g}"
            raise ValueError(
                f"{op}: no active node within tol={r:.6g} of "
                f"({float(x):.6g}, {float(y):.6g}); {detail}")
        return s

    # --- mutators (writer mutates SHM; replica mutates its private copy) ---

    def insert_node_at(self, x, y, connect_radius=0.0, **node_payload) -> int:
        """
        Insert a node at (x, y), optionally auto-connecting it to every active
        node within connect_radius (the PRM primitive). connect_radius == 0
        creates a detached node. node_payload forwards to insert_node; any
        auto-created edges keep their layout-defined defaults (use add_edge_at /
        connect_node_at for explicit edge payloads).

        WRITER-FATAL capacity policy (see plan): if edge capacity is
        insufficient, connect_node raises GraphCapacityError before adding any
        edge, but the newly inserted node may remain active and detached. No
        rollback of the inserted node is attempted; the caller terminates and
        cleans up the writer.
        """
        op = 'insert_node_at'
        try:
            cr = float(connect_radius)
        except (TypeError, ValueError):
            raise ValueError(f"{op}: connect_radius must be a scalar, got {connect_radius!r}")
        if not math.isfinite(cr) or cr < 0.0:
            raise ValueError(f"{op}: connect_radius must be finite and >= 0, "
                             f"got {connect_radius!r}")
        # insert + optional auto-connect in one seqlock window
        with self._edit_guard:
            s = self.insert_node(x, y, **node_payload)
            if cr > 0.0:
                self.connect_node(s, cr)
        return int(s)

    def delete_node_at(self, x, y, tol=None) -> int:
        """Delete the active node nearest (x, y). Returns the deleted slot."""
        op = 'delete_node_at'
        s = self._resolve_mutator_node(x, y, tol, op)
        self.delete_node(s)
        return s

    def delete_nodes_in_radius(self, x, y, r) -> int:
        """
        Delete every active node within r of (x, y) (region invalidation).
        Empty regions return 0 without raising. Failure-atomic across the whole
        bulk: every target is preflighted (active, single bucket membership,
        valid incident/reverse edges) BEFORE the first write, so a malformed
        target cannot leave earlier targets deleted.
        """
        op = 'delete_nodes_in_radius'
        # radius_query already returns active, de-duplicated, ascending slots.
        targets = [int(t) for t in self.radius_query(x, y, r)]
        if not targets:
            return 0
        # ---- preflight EVERY target before any write ---------------------
        for slot in targets:
            self._require_active_node(slot, op)
            bx, by = self._bucket_of(float(self._node_x[slot]), float(self._node_y[slot]))
            occ = sum(1 for s in self._walk_bucket_chain(bx, by) if s == slot)
            if occ != 1:
                raise GraphCorruptionError(
                    f"{op}: node {slot} appears {occ} times in bucket({bx},{by}) "
                    f"chain (expected exactly 1)")
            self._preflight_incident_pairs(slot, op)   # validates reverse edges
        # ---- execute as one batch; delete_node re-preflights the evolving
        #      state, so an edge between two targets (freed when the first
        #      target goes) is handled correctly on the second target -------
        with self._edit_guard:
            for slot in targets:
                self.delete_node(slot)
        return len(targets)

    def add_edge_at(self, pos_a, pos_b, tol=None, **edge_payload) -> Tuple[int, int]:
        """Add an undirected edge between the nodes nearest pos_a and pos_b."""
        op = 'add_edge_at'
        a = self._resolve_mutator_node(pos_a[0], pos_a[1], tol, op)
        b = self._resolve_mutator_node(pos_b[0], pos_b[1], tol, op)
        return self.add_edge(a, b, **edge_payload)

    def remove_edge_at(self, pos_a, pos_b, tol=None):
        """Remove the undirected edge between the nodes nearest pos_a and pos_b."""
        op = 'remove_edge_at'
        a = self._resolve_mutator_node(pos_a[0], pos_a[1], tol, op)
        b = self._resolve_mutator_node(pos_b[0], pos_b[1], tol, op)
        self.remove_edge(a, b)

    def remove_edge_toward(self, x, y, theta, angle_tol, tol=None):
        """
        Remove the undirected pair selected by direction: the incident edge of
        the node nearest (x, y) whose heading best matches theta (same wrapped
        angle and tie-break as edge_toward). Raises on a node miss or when no
        incident edge lies within angle_tol.
        """
        op = 'remove_edge_toward'
        th, atol = self._validate_direction(theta, angle_tol, op)
        s = self._resolve_mutator_node(x, y, tol, op)
        result = self._select_edge_toward(s, th, atol, op)
        if result is None:
            raise ValueError(
                f"{op}: node {s} at ({float(x):.6g}, {float(y):.6g}) has no incident "
                f"edge within angle_tol={atol:.6g} of theta={th:.6g}")
        _e, src, dst = result
        self.remove_edge(src, dst)

    def isolate_node_at(self, x, y, tol=None) -> int:
        """Strip all edges from the node nearest (x, y), leaving it detached."""
        op = 'isolate_node_at'
        s = self._resolve_mutator_node(x, y, tol, op)
        return self.remove_edges_of_node(s)

    def connect_node_at(self, x, y, radius, tol=None, **edge_payload):
        """
        Wire the node nearest (x, y) to every active node within `radius`.
        `radius` is REQUIRED (no default, never substituted by
        default_lookup_radius): it selects connection candidates, while `tol`
        only resolves (x, y) to the source node.
        """
        op = 'connect_node_at'
        s = self._resolve_mutator_node(x, y, tol, op)
        return self.connect_node(s, radius, **edge_payload)

    def update_node_payload_at(self, x, y, tol=None, *, indexed=None, **full_fields):
        """
        Update node payloads of the node nearest (x, y). Delegates to the Part
        1a updater unchanged: one field cannot appear in both indexed and
        full_fields, an empty update raises, and one successful call bumps
        update_version exactly once.
        """
        op = 'update_node_payload_at'
        s = self._resolve_mutator_node(x, y, tol, op)
        self.update_node_payload(s, indexed=indexed, **full_fields)

    def update_edge_payload_at(self, pos_a, pos_b, tol=None, *, indexed=None, **full_fields):
        """
        Update the UNDIRECTED edge between the resolved endpoints: preflight
        both directed rows and the field set once, then mirror the same update
        onto e_ab and e_ba failure-atomically (nothing is written if either row
        is missing or any field is invalid). Meaningful only when the layout
        declares edge payloads.
        """
        op = 'update_edge_payload_at'
        a = self._resolve_mutator_node(pos_a[0], pos_a[1], tol, op)
        b = self._resolve_mutator_node(pos_b[0], pos_b[1], tol, op)
        if a == b:
            raise ValueError(f"{op}: both endpoints resolve to the same node {a}")
        fwd = self._find_directed_edge(a, b)
        rev = self._find_directed_edge(b, a)
        if fwd is None or rev is None:
            raise ValueError(
                f"{op}: no edge pair between resolved nodes {a} and {b} "
                f"(forward row {fwd}, reverse row {rev})")
        # Validate the whole field set once (raises before any write).
        writes = self._prepare_payload_writes(
            'edge', self._layout.edge_payloads, self._edge_payloads,
            indexed, full_fields, op)
        with self._edit_guard:
            for e in (fwd, rev):                # mirror onto both directed rows
                for arr, idx, value in writes:
                    arr[(e,) + idx] = value


# ---------------------------------------------------------------------------
# Grid topology generator (Part 1b create_grid helper — generic, domain-free)
# ---------------------------------------------------------------------------

def _generate_grid_topology(xmin, xmax, ymin, ymax, resolution):
    """
    Vectorized regular-grid topology generator. Produces only coordinates and
    connectivity; the generic bulk loader populates the ordinary pools.

    Returns
    -------
    positions : (N, 2) float32 — cell-center world coords in ROW-MAJOR order
                (slot i = r*cols + c). Callers must never persist slot<->(r,c).
    undirected : (P, 2) int32 — 8-neighbor endpoint pairs, each once (u < v).
    stored_bounds : (xmin, xmax, ymin, ymax) — bounds EXTENDED down to anchor
                at xmax/ymax so every cell center and outer boundary is in
                bounds.
    rows, cols : int — ceil-based grid dimensions.

    Fixed coordinate convention for cell (r, c):
        x = xmax - (r + 0.5) * resolution     (rows grow with decreasing x)
        y = ymax - (c + 0.5) * resolution     (cols grow with decreasing y)
    """
    rows = max(1, int(math.ceil((xmax - xmin) / resolution)))
    cols = max(1, int(math.ceil((ymax - ymin) / resolution)))
    # Row-major (r, c) index grids: slot i = r*cols + c.
    r_idx, c_idx = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    r_flat = r_idx.ravel()          # (N,) row index per slot
    c_flat = c_idx.ravel()          # (N,) col index per slot
    # Cell-center coordinates (float64 math, stored as float32).
    px = xmax - (r_flat + 0.5) * resolution
    py = ymax - (c_flat + 0.5) * resolution
    positions = np.stack([px, py], axis=1).astype(np.float32)   # (N, 2)
    # 8-neighbor undirected pairs, each emitted once via the s1 < s2 filter.
    slot = r_flat * cols + c_flat                                # == arange(N)
    offsets = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
               (0, 1), (1, -1), (1, 0), (1, 1))
    pair_chunks = []
    for dr, dc in offsets:
        r2 = r_flat + dr
        c2 = c_flat + dc
        valid = (r2 >= 0) & (r2 < rows) & (c2 >= 0) & (c2 < cols)
        s1 = slot[valid]
        s2 = (r2 * cols + c2)[valid]
        keep = s1 < s2                                           # dedup pairs
        if np.any(keep):
            pair_chunks.append(np.stack([s1[keep], s2[keep]], axis=1))
    if pair_chunks:
        undirected = np.concatenate(pair_chunks, axis=0).astype(np.int32)
    else:
        undirected = np.empty((0, 2), dtype=np.int32)           # 1x1 grid
    stored_bounds = (xmax - rows * resolution, xmax,
                     ymax - cols * resolution, ymax)
    return positions, undirected, stored_bounds, rows, cols


# ---------------------------------------------------------------------------
# Writer: creates and owns every SHM block
# ---------------------------------------------------------------------------

class SharedGraphWriter(_GraphEngine):
    """
    Creates the shared graph and is the single process allowed to mutate it.

    Initialization publication: the metadata block is created first with
    graph_seq = 1 (odd), every core/payload block is created and initialized,
    and only the very last write publishes graph_seq = 2. A reader therefore
    never accepts a half-created graph — it waits for an even sequence.

    Lifecycle mirrors SharedMemoryBackend: stale blocks from a crashed writer
    are auto-cleaned on creation, cleanup runs on atexit/SIGINT/SIGTERM and
    on context-manager exit, close() drops local handles, unlink() removes
    the named blocks. Using the writer after cleanup raises.
    """

    def __init__(self, n_max: int, e_max: int,
                 bounds: Tuple[float, float, float, float],
                 bucket_size: float, layout: GraphLayout,
                 default_lookup_radius: Optional[float] = None):
        _require_supported_platform()
        # ---- parameter validation (fail loud before touching /dev/shm) ----
        if not isinstance(layout, GraphLayout):
            raise TypeError(f"layout must be a GraphLayout, got {type(layout).__name__}")
        if not isinstance(n_max, (int, np.integer)) or n_max < 1:
            raise ValueError(f"n_max must be an int >= 1, got {n_max!r}")
        if not isinstance(e_max, (int, np.integer)) or e_max < 2:
            raise ValueError(f"e_max must be an int >= 2 (one undirected edge needs "
                             f"two rows), got {e_max!r}")
        try:
            xmin, xmax, ymin, ymax = (float(v) for v in bounds)
        except (TypeError, ValueError):
            raise ValueError(f"bounds must be (xmin, xmax, ymin, ymax), got {bounds!r}")
        if not all(math.isfinite(v) for v in (xmin, xmax, ymin, ymax)):
            raise ValueError(f"bounds must be finite, got {bounds!r}")
        if not (xmin < xmax and ymin < ymax):
            raise ValueError(f"bounds must satisfy xmin < xmax and ymin < ymax, "
                             f"got {bounds!r}")
        bucket_size = float(bucket_size)
        if not math.isfinite(bucket_size) or bucket_size <= 0.0:
            raise ValueError(f"bucket_size must be finite and > 0, got {bucket_size!r}")
        if default_lookup_radius is None:
            # Part 1b consumes this; default: half a bucket as the position-
            # resolution tolerance.
            default_lookup_radius = 0.5 * bucket_size
        default_lookup_radius = float(default_lookup_radius)
        if not math.isfinite(default_lookup_radius) or default_lookup_radius <= 0.0:
            raise ValueError(f"default_lookup_radius must be finite and > 0, "
                             f"got {default_lookup_radius!r}")
        n_max, e_max = int(n_max), int(e_max)
        n_bx = max(1, int(math.ceil((xmax - xmin) / bucket_size)))
        n_by = max(1, int(math.ceil((ymax - ymin) / bucket_size)))

        self.shm_blocks: Dict[str, shared_memory.SharedMemory] = {}
        self._closed = False
        # signum -> handler this writer displaced, restored on cleanup()
        self._prev_handlers: Dict[int, object] = {}
        try:
            # ---- 1) metadata first, published ODD so readers wait --------
            meta_shm = self._create_block(_GRAPH_META_SHM_NAME, _GRAPH_META_DTYPE.itemsize)
            self.shm_blocks[_GRAPH_META_SHM_NAME] = meta_shm
            meta = np.ndarray(1, dtype=_GRAPH_META_DTYPE, buffer=meta_shm.buf)
            meta[:] = np.zeros(1, dtype=_GRAPH_META_DTYPE)
            meta[0]['graph_seq'] = 1          # odd: initialization in progress
            _seq_fence()                      # odd visible before any init write
            # ---- 2) geometry and capacities -------------------------------
            meta[0]['n_max'] = n_max
            meta[0]['e_max'] = e_max
            meta[0]['n_bx'] = n_bx
            meta[0]['n_by'] = n_by
            meta[0]['xmin'] = xmin
            meta[0]['xmax'] = xmax
            meta[0]['ymin'] = ymin
            meta[0]['ymax'] = ymax
            meta[0]['bucket_size'] = bucket_size
            meta[0]['default_lookup_radius'] = default_lookup_radius
            # ---- 3) create every core and payload block -------------------
            block_specs = _expected_data_blocks(layout, n_max, e_max, n_bx, n_by)
            views: Dict[str, np.ndarray] = {}
            for name, (shape, dt) in block_specs.items():
                nbytes = int(np.prod(shape, dtype=np.int64)) * dt.itemsize
                shm = self._create_block(name, nbytes)
                self.shm_blocks[name] = shm
                views[name] = np.ndarray(shape, dtype=dt, buffer=shm.buf)
            node_core = {f: views[_node_core_block_name(f)] for f, _ in _NODE_CORE_SPECS}
            edge_core = {f: views[_edge_core_block_name(f)] for f, _ in _EDGE_CORE_SPECS}
            bucket_head = views[_BUCKET_HEAD_SHM_NAME]
            node_payloads = {n: views[_node_payload_block_name(n)]
                             for n in layout.node_payloads}
            edge_payloads = {n: views[_edge_payload_block_name(n)]
                             for n in layout.edge_payloads}
            # ---- 4) initialize pools, free lists, sentinels ---------------
            node_core['x'][:] = 0.0
            node_core['y'][:] = 0.0
            node_core['active'][:] = 0
            node_core['generation'][:] = 0
            node_core['update_version'][:] = 0
            node_core['first_edge'][:] = -1
            # Node free list 0 -> 1 -> ... -> n_max-1 -> -1 through
            # next_in_bucket (free slots are never in a bucket).
            node_core['next_in_bucket'][:] = np.arange(1, n_max + 1, dtype=np.int32)
            node_core['next_in_bucket'][n_max - 1] = -1
            edge_core['target'][:] = -1
            edge_core['length'][:] = 0.0
            # Edge free list 0 -> 1 -> ... -> e_max-1 -> -1 through next.
            edge_core['next'][:] = np.arange(1, e_max + 1, dtype=np.int32)
            edge_core['next'][e_max - 1] = -1
            bucket_head[:] = -1
            for name, arr in node_payloads.items():
                arr[:] = layout.node_payloads[name][2]   # per-payload default
            for name, arr in edge_payloads.items():
                arr[:] = layout.edge_payloads[name][2]
            meta[0]['node_free_head'] = 0
            meta[0]['edge_free_head'] = 0
            meta[0]['node_hwm'] = 0            # exclusive HWMs
            meta[0]['edge_hwm'] = 0
            # ---- bind the shared mutation engine --------------------------
            self._bind_state(meta, node_core, edge_core, bucket_head,
                             node_payloads, edge_payloads, layout,
                             _SeqlockEditGuard(meta), is_replica=False)
            # ---- cleanup handlers (mirrors SharedMemoryBackend) -----------
            atexit.register(self._cleanup_on_exit)
            try:
                # Same set the map backend guards (Ctrl+C, kill, terminal hangup,
                # Ctrl+\); each displaced handler is remembered for cleanup().
                for _sig in _CLEANUP_SIGNALS:
                    self._prev_handlers[_sig] = signal.signal(_sig, self._signal_handler)
            except ValueError:
                # signal.signal only works in the main thread; atexit still
                # covers normal termination.
                pass
            # ---- 5) publish: even sequence only after the final init write
            _seq_fence()                      # every init write visible first
            meta[0]['graph_seq'] = 2
        except Exception:
            # A half-created graph must not leak named blocks.
            self._drop_views()
            for shm in self.shm_blocks.values():
                try:
                    shm.close()
                    _unlink_shm(shm)   # unlink() also unregisters from the tracker
                except Exception:
                    pass
            self.shm_blocks = {}
            raise

        total_mb = sum(shm.size for shm in self.shm_blocks.values()) / 1e6
        print(f"Created shared graph: n_max={n_max}, e_max={e_max}, "
              f"buckets={n_bx}x{n_by}, {len(self.shm_blocks)} blocks, "
              f"{total_mb:.1f}MB")

    # ------------------------------------------------------------------
    # Grid initializer (Part 1b) — generic vectorized bulk topology loader
    # ------------------------------------------------------------------

    @classmethod
    def create_grid(cls, xmin, xmax, ymin, ymax, resolution, layout, *,
                    n_max=None, e_max=None, bucket_size=None,
                    isolate_border_width=0.0,
                    isolate_border_bounds=None,
                    isolated_node_payloads=None) -> 'SharedGraphWriter':
        """
        Build a writer holding a regular 8-connected grid, bulk-loaded into the
        SAME generic node/edge pools every other graph uses. The grid is only an
        initialization recipe: after construction the writer retains no grid
        identity, metadata, cell index, or resolution field, and every
        frontend/snapshot works without a grid-specific branch.

        `layout` is the same immutable shared node/edge GraphLayout required by
        the ordinary constructor. default_lookup_radius is always set internally
        to 0.75*resolution (covers every cell corner but never reaches an
        orthogonal neighbor at distance `resolution`); it is a lookup radius
        only and is never a connection radius. To restore an isolated grid
        node's 8 neighbors via connect_node_at the caller must explicitly pass
        radius = 1.5*resolution (the completed graph stores no resolution).

        `isolate_border_width` is an optional grid-only initialization policy.
        Nodes whose centers are closer than this width to any side of
        `isolate_border_bounds` remain active and spatially indexed but start
        with degree zero. If no isolation bounds are supplied, the grid input
        bounds are used. The default width of zero preserves the ordinary fully
        connected grid. `isolated_node_payloads` optionally supplies full-field
        payload values for those isolated nodes; values may be scalars that
        broadcast over their declared per-node payload shapes.

        Publication follows the Part 1b sequence contract: the ordinary
        constructor leaves graph_seq = 2 (storage ready, no content), this
        method sets graph_seq = 3 (population in progress), bulk-populates via a
        private direct-write path, and writes graph_seq = 4 (first complete
        graph published) ONLY after successful completion. The bulk population
        does NOT run through _SeqlockEditGuard, whose finally path would publish
        an even sequence after an exception. On any failure after the writer
        exists, 4 is never published and every claimed SHM block is unlinked.
        """
        op = 'create_grid'
        # ---- 1) validate bounds/resolution (no SHM yet) ------------------
        if not isinstance(layout, GraphLayout):
            raise TypeError(f"layout must be a GraphLayout, got {type(layout).__name__}")
        try:
            xmin, xmax, ymin, ymax, resolution = (float(xmin), float(xmax),
                                                  float(ymin), float(ymax),
                                                  float(resolution))
        except (TypeError, ValueError):
            raise ValueError(f"{op}: bounds and resolution must be scalars")
        if not all(math.isfinite(v) for v in (xmin, xmax, ymin, ymax, resolution)):
            raise ValueError(f"{op}: bounds and resolution must be finite, got "
                             f"x[{xmin}, {xmax}] y[{ymin}, {ymax}] res={resolution}")
        if not (xmax > xmin and ymax > ymin):
            raise ValueError(f"{op}: extents must be positive (xmax > xmin, "
                             f"ymax > ymin), got x[{xmin}, {xmax}] y[{ymin}, {ymax}]")
        if resolution <= 0.0:
            raise ValueError(f"{op}: resolution must be > 0, got {resolution}")

        try:
            isolate_border_width = float(isolate_border_width)
        except (TypeError, ValueError):
            raise ValueError(
                f"{op}: isolate_border_width must be a scalar, got "
                f"{isolate_border_width!r}")
        if not math.isfinite(isolate_border_width) or isolate_border_width < 0.0:
            raise ValueError(
                f"{op}: isolate_border_width must be finite and >= 0, got "
                f"{isolate_border_width}")

        if isolate_border_bounds is None:
            border_xmin, border_xmax, border_ymin, border_ymax = xmin, xmax, ymin, ymax
        else:
            try:
                border_xmin, border_xmax, border_ymin, border_ymax = (
                    float(v) for v in isolate_border_bounds)
            except (TypeError, ValueError):
                raise ValueError(
                    f"{op}: isolate_border_bounds must be "
                    f"(xmin, xmax, ymin, ymax), got {isolate_border_bounds!r}")
            if not all(math.isfinite(v) for v in
                       (border_xmin, border_xmax, border_ymin, border_ymax)):
                raise ValueError(
                    f"{op}: isolate_border_bounds must be finite, got "
                    f"{isolate_border_bounds!r}")
            if not (border_xmax > border_xmin and border_ymax > border_ymin):
                raise ValueError(
                    f"{op}: isolate_border_bounds must satisfy xmin < xmax and "
                    f"ymin < ymax, got {isolate_border_bounds!r}")

        isolated_payload_values = {}
        for name, value in dict(isolated_node_payloads or {}).items():
            if name not in layout.node_payloads:
                raise KeyError(
                    f"{op}: unknown isolated node payload {name!r}; available: "
                    f"{sorted(layout.node_payloads)}")
            shape, dt, _ = layout.node_payloads[name]
            isolated_payload_values[name] = GraphLayout._normalize_default(
                'node', name, shape, dt, value)

        # ---- 2) generate topology + preflight float32 representability ---
        positions, undirected, stored_bounds, rows, cols = \
            _generate_grid_topology(xmin, xmax, ymin, ymax, resolution)

        # A border-isolated node stays in the ordinary active-node pool and
        # spatial buckets, but no initial edge may be incident to it. Filtering
        # the undirected pairs before bulk loading establishes degree zero
        # atomically as part of the grid initializer's 2 -> 3 -> 4 publication.
        isolated = np.zeros(positions.shape[0], dtype=bool)  # (N,) isolated-node mask
        if isolate_border_width > 0.0:
            px64 = positions[:, 0].astype(np.float64)  # (N,) grid-node world x
            py64 = positions[:, 1].astype(np.float64)  # (N,) grid-node world y
            isolated = (((px64 - border_xmin) < isolate_border_width)
                        | ((border_xmax - px64) < isolate_border_width)
                        | ((py64 - border_ymin) < isolate_border_width)
                        | ((border_ymax - py64) < isolate_border_width))  # (N,)
            if undirected.shape[0]:
                connected_pairs = ~(isolated[undirected[:, 0]]
                                    | isolated[undirected[:, 1]])
                undirected = undirected[connected_pairs]

        # adjacent cell centers must remain distinct in float32 storage
        row_x = (xmax - (np.arange(rows) + 0.5) * resolution).astype(np.float32)
        col_y = (ymax - (np.arange(cols) + 0.5) * resolution).astype(np.float32)
        if not (np.all(np.isfinite(row_x)) and np.all(np.isfinite(col_y))
                and np.all(np.isfinite(positions))):
            raise ValueError(f"{op}: cell centers collapse to non-finite float32 for "
                             f"bounds x[{xmin}, {xmax}] y[{ymin}, {ymax}] res={resolution}")
        if np.unique(row_x).size != rows or np.unique(col_y).size != cols:
            raise ValueError(
                f"{op}: adjacent cell centers collapse in float32 (rows={rows}, "
                f"cols={cols}, res={resolution}); increase resolution or shrink the "
                f"world extent so centers stay distinct")
        # every generated edge must have strictly positive stored length
        if undirected.shape[0]:
            pu = positions[undirected[:, 0]].astype(np.float64)
            pv = positions[undirected[:, 1]].astype(np.float64)
            elen = np.hypot(pu[:, 0] - pv[:, 0], pu[:, 1] - pv[:, 1])
            if np.any(elen <= 0.0):
                raise ValueError(f"{op}: some generated edges have zero length in "
                                 f"float32; resolution {resolution} is too small")

        # ---- 3) capacities + bucket size (preflight before creating SHM) -
        N = int(positions.shape[0])           # == rows*cols
        M = int(2 * undirected.shape[0])      # directed edge rows
        n_max_eff = N if n_max is None else int(n_max)
        e_max_eff = (rows * cols * 8) if e_max is None else int(e_max)
        if n_max_eff < N:
            raise ValueError(f"{op}: n_max={n_max_eff} cannot hold {N} grid nodes")
        if e_max_eff < max(2, M):
            raise ValueError(f"{op}: e_max={e_max_eff} cannot hold {M} directed edge "
                             f"rows (minimum {max(2, M)})")
        if bucket_size is None:
            bucket_size_eff = resolution
        else:
            bucket_size_eff = float(bucket_size)
            if not math.isfinite(bucket_size_eff) or bucket_size_eff <= 0.0:
                raise ValueError(f"{op}: bucket_size must be finite and > 0, "
                                 f"got {bucket_size!r}")

        # ---- 4) construct writer (graph_seq = 2), then publish 2 -> 3 -> 4
        writer = cls(n_max=n_max_eff, e_max=e_max_eff, bounds=stored_bounds,
                     bucket_size=bucket_size_eff, layout=layout,
                     default_lookup_radius=0.75 * resolution)
        try:
            writer._meta[0]['graph_seq'] = 3        # initial population in progress
            _seq_fence()                            # odd visible before population
            writer._bulk_initialize_topology(positions, undirected)
            for name, value in isolated_payload_values.items():
                writer._node_payloads[name][isolated] = value
            # final validation before publishing the graph as complete
            if int(writer._meta[0]['node_hwm']) != N or int(writer._meta[0]['edge_hwm']) != M:
                raise GraphCorruptionError(
                    f"{op}: post-population HWMs "
                    f"(node={int(writer._meta[0]['node_hwm'])}, "
                    f"edge={int(writer._meta[0]['edge_hwm'])}) do not match generated "
                    f"counts (N={N}, M={M})")
            _seq_fence()                            # population visible first
            writer._meta[0]['graph_seq'] = 4        # publish only on success
        except Exception:
            # Never leave a half-populated graph named; never publish 4.
            writer.cleanup()
            raise
        return writer

    def _bulk_initialize_topology(self, node_positions, undirected_edges):
        """
        Vectorized bulk population of the ordinary generic pools from a
        precomputed topology. Writes node/edge core arrays, bucket chains,
        adjacency chains, edge lengths, free lists, and HWMs DIRECTLY (no
        per-element insert_node/add_edge, no _SeqlockEditGuard — create_grid
        owns the seqlock through its 2 -> 3 -> 4 publication path). Generic:
        reusable later for bulk PRM / imported-topology construction.

        node_positions   : (N, 2) float32 world coordinates; slot i = row i.
        undirected_edges : (P, 2) int32 endpoint pairs, each undirected edge
                           once; becomes M = 2*P directed rows [0, M).

        The result is observationally equivalent to allocating the same
        topology through insert_node/add_edge: active rows [0, N),
        generation 0, update_version 1, node_hwm = N, edge_hwm = M, canonical
        free chains for the remaining capacity, and layout-default payloads
        everywhere (node/edge payload rows already hold their defaults from
        construction). create_grid may overwrite selected node payload rows
        before publication when its border-isolation policy requests that.
        """
        n_max, e_max = self.n_max, self.e_max
        positions = np.asarray(node_positions, dtype=np.float32)  # (N, 2)
        N = int(positions.shape[0])
        edges = np.asarray(undirected_edges, dtype=np.int64).reshape(-1, 2)  # (P, 2)
        P = int(edges.shape[0])
        M = 2 * P
        # ---- last-line capacity guards (create_grid preflights already) --
        if N > n_max:
            raise GraphCapacityError(
                f"_bulk_initialize_topology: {N} nodes exceed n_max={n_max}")
        if M > e_max:
            raise GraphCapacityError(
                f"_bulk_initialize_topology: {M} directed edge rows exceed e_max={e_max}")

        nc, ec = self._node_core, self._edge_core
        px = positions[:, 0]           # (N,) float32 x
        py = positions[:, 1]           # (N,) float32 y
        # ---- 1) node core prefix [0, N) ----------------------------------
        nc['x'][:N] = px
        nc['y'][:N] = py
        nc['active'][:N] = 1
        nc['generation'][:N] = 0
        nc['update_version'][:N] = 1   # one insert bump each (never reset)
        nc['first_edge'][:N] = -1      # set from adjacency chains below

        # ---- 2) bucket chains (head-insertion emulation in slot order) ---
        m = self._meta[0]
        b = float(m['bucket_size'])
        xmin_s, ymin_s = float(m['xmin']), float(m['ymin'])
        n_bx, n_by = int(m['n_bx']), int(m['n_by'])
        # Bucket cell of each stored position, clamped exactly like _bucket_of.
        bx = np.clip(np.floor((px.astype(np.float64) - xmin_s) / b).astype(np.int64),
                     0, n_bx - 1)
        by = np.clip(np.floor((py.astype(np.float64) - ymin_s) / b).astype(np.int64),
                     0, n_by - 1)
        bflat = bx * n_by + by         # (N,) flattened (row-major) bucket id
        next_ib = np.full(N, -1, dtype=np.int32)
        self._bucket_head[:] = -1
        if N > 0:
            # stable argsort groups equal buckets while preserving ascending
            # slot order within a group -> emulates head-insertion in slot order
            order = np.argsort(bflat, kind='stable')     # slots grouped by bucket
            sorted_b = bflat[order]
            same_prev = np.zeros(N, dtype=bool)
            same_prev[1:] = sorted_b[1:] == sorted_b[:-1]
            pos = np.flatnonzero(same_prev)              # link to previous in group
            next_ib[order[pos]] = order[pos - 1].astype(np.int32)
            last_of = np.ones(N, dtype=bool)             # last (highest) slot = head
            last_of[:-1] = sorted_b[:-1] != sorted_b[1:]
            hp = np.flatnonzero(last_of)
            self._bucket_head.reshape(-1)[sorted_b[hp]] = order[hp].astype(np.int32)
        nc['next_in_bucket'][:N] = next_ib               # free tail rebuilt in step 4

        # ---- 3) edge rows [0, M) + adjacency chains ----------------------
        if P > 0:
            u = edges[:, 0]
            v = edges[:, 1]
            # directed rows: 2p = u->v (owner u), 2p+1 = v->u (owner v)
            owners = np.empty(M, dtype=np.int64)
            targets = np.empty(M, dtype=np.int32)
            owners[0::2] = u
            owners[1::2] = v
            targets[0::2] = v.astype(np.int32)
            targets[1::2] = u.astype(np.int32)
            # immutable geometric length from stored float32 endpoints
            # (float64 hypot -> float32, matching the engine's _link_edge_pair)
            dux = px[u].astype(np.float64) - px[v].astype(np.float64)
            duy = py[u].astype(np.float64) - py[v].astype(np.float64)
            length = np.hypot(dux, duy).astype(np.float32)   # (P,)
            ec['target'][:M] = targets
            ec['length'][0:M:2] = length
            ec['length'][1:M:2] = length
            # adjacency chains (head-insertion emulation in row order)
            edge_next = np.full(M, -1, dtype=np.int32)
            row_order = np.argsort(owners, kind='stable')    # rows grouped by owner
            sorted_owner = owners[row_order]
            same_prev = np.zeros(M, dtype=bool)
            same_prev[1:] = sorted_owner[1:] == sorted_owner[:-1]
            pos = np.flatnonzero(same_prev)
            edge_next[row_order[pos]] = row_order[pos - 1].astype(np.int32)
            ec['next'][:M] = edge_next
            last_of = np.ones(M, dtype=bool)
            last_of[:-1] = sorted_owner[:-1] != sorted_owner[1:]
            hp = np.flatnonzero(last_of)
            nc['first_edge'][sorted_owner[hp]] = row_order[hp].astype(np.int32)

        # ---- 4) free lists + HWMs ----------------------------------------
        # Node free tail: rows [N, n_max) chained N -> N+1 -> ... -> -1.
        if n_max > N:
            if n_max - 1 > N:
                nc['next_in_bucket'][N:n_max - 1] = np.arange(N + 1, n_max, dtype=np.int32)
            nc['next_in_bucket'][n_max - 1] = -1
            m['node_free_head'] = N
        else:
            m['node_free_head'] = -1
        # Edge free tail: rows [M, e_max) chained M -> M+1 -> ... -> -1.
        if e_max > M:
            if e_max - 1 > M:
                ec['next'][M:e_max - 1] = np.arange(M + 1, e_max, dtype=np.int32)
            ec['next'][e_max - 1] = -1
            m['edge_free_head'] = M
        else:
            m['edge_free_head'] = -1
        m['node_hwm'] = N              # exclusive HWMs
        m['edge_hwm'] = M

    # ------------------------------------------------------------------
    # SHM block creation with stale-block auto-clean (crash recovery)
    # ------------------------------------------------------------------

    @staticmethod
    def _create_block(name: str, size: int) -> shared_memory.SharedMemory:
        if size < 1:
            raise ValueError(f"shared-memory block '{name}' would have size {size}")
        try:
            return shared_memory.SharedMemory(create=True, size=size, name=name)
        except FileExistsError:
            # Stale block from a previous crash — clean and retry, mirroring
            # SharedMemoryBackend._cleanup_stale_memory.
            print(f"Warning: stale shared-memory block '{name}' detected "
                  f"(previous crash?). Auto-cleaning...")
            try:
                stale = shared_memory.SharedMemory(name=name)
                stale.close()
                stale.unlink()
            except FileNotFoundError:
                pass
            return shared_memory.SharedMemory(create=True, size=size, name=name)

    # ------------------------------------------------------------------
    # Lifecycle (mirrors SharedMemoryBackend cleanup pattern)
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.cleanup()
        return False

    def _signal_handler(self, signum, frame):
        self.cleanup()
        sys.exit(0)

    def _cleanup_on_exit(self):
        self.cleanup()

    def close(self):
        """Close this process's handles (views + mmaps); blocks stay named."""
        self._drop_views()
        for shm in self.shm_blocks.values():
            try:
                shm.close()
            except Exception:
                pass   # e.g. BufferError if a caller still holds a view

    def unlink(self):
        """Remove every named block from the system (writer owns them).

        SharedMemory.unlink() already unregisters the block from the
        multiprocessing resource tracker (Python >= 3.9), so no explicit
        unregister call is needed here; _unlink_shm only re-registers first so
        that unregister always matches an entry a reader may have disowned.
        """
        for shm in self.shm_blocks.values():
            try:
                _unlink_shm(shm)
            except Exception:
                pass

    def cleanup(self):
        """Idempotent close + unlink; also restores signal handlers."""
        if self._closed:
            return
        self._closed = True
        self.close()
        self.unlink()
        self.shm_blocks = {}
        atexit.unregister(self._cleanup_on_exit)
        # Restore the handlers we displaced so a cleaned-up writer does not
        # keep hijacking the host process's signals.
        try:
            for _sig, _prev in self._prev_handlers.items():
                if _prev is not None:
                    signal.signal(_sig, _prev)
        except (ValueError, TypeError):
            pass
        print("Shared graph writer cleaned up")


# ---------------------------------------------------------------------------
# Reader: attach + validate + seqlock snapshot
# ---------------------------------------------------------------------------

class SharedGraphReader(_GraphStateView):
    """
    Attaches to a writer-created graph by name and produces GraphSnapshot
    replicas. Has NO mutators (single-writer invariant). Attachment waits for
    the writer with the existing backend's timeout pattern, refuses to accept
    an unpublished (odd-sequence) initialization, and validates that every
    block implied by the shared layout exists with the exact byte size before
    creating any NumPy view. On failure every block attached during the
    attempt is closed.
    """

    def __init__(self, layout: GraphLayout, attach_timeout: float = 10.0):
        _require_supported_platform()
        if not isinstance(layout, GraphLayout):
            raise TypeError(f"layout must be a GraphLayout, got {type(layout).__name__}")
        attach_timeout = float(attach_timeout)
        if attach_timeout <= 0.0:
            raise ValueError(f"attach_timeout must be > 0, got {attach_timeout}")

        self._closed = False
        self.last_snapshot_attempts = 0
        deadline = time.monotonic() + attach_timeout
        attached: Dict[str, shared_memory.SharedMemory] = {}
        try:
            # ---- metadata block: wait-attach with timeout ------------------
            meta_shm = self._wait_attach(_GRAPH_META_SHM_NAME, deadline, attach_timeout)
            attached[_GRAPH_META_SHM_NAME] = meta_shm
            if meta_shm.size < _GRAPH_META_DTYPE.itemsize:
                raise GraphAttachError(
                    f"metadata block '{_GRAPH_META_SHM_NAME}' has {meta_shm.size} bytes, "
                    f"expected at least {_GRAPH_META_DTYPE.itemsize}; schema mismatch "
                    f"between writer and reader builds")
            meta = np.ndarray(1, dtype=_GRAPH_META_DTYPE, buffer=meta_shm.buf)
            # ---- wait for content publication (even, graph_seq >= 4) -------
            # Part 1b sequence contract: 1 = block creation, 2 = storage
            # initialized but no graph content published, 3 = initial
            # population in progress, 4 = first complete graph published,
            # then ordinary batches continue 4 -> 5 -> 6 -> ... A reader that
            # observes 1/2/3 keeps waiting until the writer publishes content.
            t0 = time.monotonic()
            while True:
                seq = int(meta[0]['graph_seq'])
                # Acquire barrier: the geometry/capacity reads below must not be
                # hoisted above this publication check on a weakly ordered machine.
                _seq_fence()
                if seq >= 4 and seq % 2 == 0:
                    break
                if time.monotonic() > deadline:
                    raise GraphAttachError(
                        f"graph_seq={seq} stayed odd/unpublished (< 4) for "
                        f"{time.monotonic() - t0:.2f}s: writer crashed "
                        f"mid-initialization or mid-batch, is still initializing, or "
                        f"has not performed its first content-publishing mutation; "
                        f"increase attach_timeout if initialization is legitimately slow")
                time.sleep(0.01)
            # ---- metadata sanity before trusting any capacity --------------
            n_max, e_max = int(meta[0]['n_max']), int(meta[0]['e_max'])
            n_bx, n_by = int(meta[0]['n_bx']), int(meta[0]['n_by'])
            xmin, xmax = float(meta[0]['xmin']), float(meta[0]['xmax'])
            ymin, ymax = float(meta[0]['ymin']), float(meta[0]['ymax'])
            bucket_size = float(meta[0]['bucket_size'])
            if (n_max < 1 or e_max < 2 or n_bx < 1 or n_by < 1
                    or not (xmin < xmax and ymin < ymax)
                    or not math.isfinite(bucket_size) or bucket_size <= 0.0):
                raise GraphAttachError(
                    f"metadata is internally invalid: n_max={n_max}, e_max={e_max}, "
                    f"buckets={n_bx}x{n_by}, bounds=({xmin},{xmax},{ymin},{ymax}), "
                    f"bucket_size={bucket_size}")
            # ---- attach every expected block, exact byte-size check --------
            block_specs = _expected_data_blocks(layout, n_max, e_max, n_bx, n_by)
            views: Dict[str, np.ndarray] = {}
            for name, (shape, dt) in block_specs.items():
                shm = self._wait_attach(name, deadline, attach_timeout)
                attached[name] = shm
                expected = int(np.prod(shape, dtype=np.int64)) * dt.itemsize
                if shm.size != expected:
                    raise GraphAttachError(
                        f"block '{name}' has {shm.size} bytes, expected exactly "
                        f"{expected} for shape {shape} dtype {dt} — the writer was "
                        f"created with a different payload layout or capacity")
                views[name] = np.ndarray(shape, dtype=dt, buffer=shm.buf)
        except Exception:
            # Close every block attached during this failed attempt.
            for shm in attached.values():
                try:
                    shm.close()
                except Exception:
                    pass
            raise

        self.shm_blocks = attached
        node_core = {f: views[_node_core_block_name(f)] for f, _ in _NODE_CORE_SPECS}
        edge_core = {f: views[_edge_core_block_name(f)] for f, _ in _EDGE_CORE_SPECS}
        node_payloads = {n: views[_node_payload_block_name(n)] for n in layout.node_payloads}
        edge_payloads = {n: views[_edge_payload_block_name(n)] for n in layout.edge_payloads}
        self._bind_views(meta, node_core, edge_core, views[_BUCKET_HEAD_SHM_NAME],
                         node_payloads, edge_payloads, layout)

    @staticmethod
    def _wait_attach(name: str, deadline: float,
                     attach_timeout: float) -> shared_memory.SharedMemory:
        """Attach to a named block, waiting for the writer (existing pattern)."""
        while True:
            try:
                shm = shared_memory.SharedMemory(name=name)
                # Attaching registered this writer-owned block with the reader's
                # resource tracker; disown it so a reader that exits before the
                # writer cannot unlink the graph out from under it.
                _untrack_shm(shm)
                return shm
            except FileNotFoundError:
                if time.monotonic() > deadline:
                    raise GraphAttachError(
                        f"timeout ({attach_timeout}s) waiting for shared-memory "
                        f"block '{name}' — is the graph writer running, and do "
                        f"writer and reader share the same GraphLayout?")
                time.sleep(0.05)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self):
        """Close attached handles. The reader never unlinks writer-owned blocks."""
        if self._closed:
            return
        self._closed = True
        self._drop_views()
        for shm in self.shm_blocks.values():
            try:
                shm.close()
            except Exception:
                pass
        self.shm_blocks = {}

    # ------------------------------------------------------------------
    # Seqlock snapshot
    # ------------------------------------------------------------------

    def _read_seq(self) -> int:
        # Fenced on BOTH sides so this one helper serves both seqlock loads of
        # snapshot(): the leading fence orders the preceding array reads before
        # the s2 load, the trailing fence keeps the following array reads from
        # being hoisted above the s1 load. Both are no-ops on x86-64.
        _seq_fence()
        seq = int(self._meta[0]['graph_seq'])
        _seq_fence()
        return seq

    def snapshot(self, max_retries: int = 50, into: Optional['GraphSnapshot'] = None,
                 node_headroom: int = 256, edge_headroom: int = 1024) -> 'GraphSnapshot':
        """
        Copy the graph into a fully functional offline replica via the
        seqlock retry loop:

            s1 = graph_seq                  (odd -> backoff, retry)
            copy rows [0, node_hwm) / [0, edge_hwm), bucket_head, metadata
            s2 = graph_seq
            accept only if s1 == s2 (and even)

        Pass `into=` a previous GraphSnapshot to overwrite it in place (its
        local what-if edits are completely removed; it is reallocated
        transparently if the writer HWM outgrew it). On SnapshotRetryError
        the contents of `into` are undefined and must not be used.
        `node_headroom` / `edge_headroom` rows of local free capacity are
        reserved beyond the writer prefix for replica what-if edits.
        """
        if self._closed:
            raise SharedGraphError("snapshot: reader is closed")
        if not isinstance(max_retries, (int, np.integer)) or max_retries < 1:
            raise ValueError(f"max_retries must be an int >= 1, got {max_retries!r}")
        for label, hr in (('node_headroom', node_headroom), ('edge_headroom', edge_headroom)):
            if not isinstance(hr, (int, np.integer)) or hr < 1:
                raise ValueError(f"{label} must be an int >= 1, got {hr!r}")
        if into is not None:
            if not isinstance(into, GraphSnapshot):
                raise TypeError(f"into must be a GraphSnapshot, got {type(into).__name__}")
            if into._layout != self._layout:
                raise ValueError("into: replica layout differs from this reader's layout")
            if into._bucket_head is not None and into._bucket_head.shape != self._bucket_head.shape:
                raise ValueError(
                    f"into: replica bucket grid {into._bucket_head.shape} differs from "
                    f"writer grid {self._bucket_head.shape}; it belongs to a different graph")
        snap = into if into is not None else GraphSnapshot(_SNAPSHOT_CONSTRUCTION_TOKEN,
                                                           self._layout)

        shm_n_max, shm_e_max = self.n_max, self.e_max
        odd_seen = 0
        torn_seen = 0
        first_seq = None
        last_seq = None
        t_start = time.monotonic()
        for attempt in range(1, int(max_retries) + 1):
            s1 = self._read_seq()
            if first_seq is None:
                first_seq = s1
            last_seq = s1
            if s1 < 4 or s1 % 2 != 0:
                # Writer mid-edit, or content not yet published (graph_seq < 4
                # per the Part 1b publication contract): brief bounded backoff
                # instead of tight-spinning.
                odd_seen += 1
                time.sleep(_SNAPSHOT_RETRY_SLEEP_S)
                continue
            meta_copy = self._meta.copy()          # (1,) record snapshot
            node_hwm = int(meta_copy[0]['node_hwm'])
            edge_hwm = int(meta_copy[0]['edge_hwm'])
            if not (0 <= node_hwm <= shm_n_max and 0 <= edge_hwm <= shm_e_max):
                # The writer never stores out-of-range HWMs, even mid-batch.
                raise GraphCorruptionError(
                    f"snapshot: HWMs out of range (node_hwm={node_hwm}/{shm_n_max}, "
                    f"edge_hwm={edge_hwm}/{shm_e_max}); shared metadata is corrupt")
            snap._ensure_local_capacity(node_hwm + int(node_headroom),
                                        edge_hwm + int(edge_headroom),
                                        self._bucket_head.shape)
            # ---- raw prefix copy (validated by the s2 re-read below) ------
            for field, src in self._node_core.items():
                np.copyto(snap._node_core[field][:node_hwm], src[:node_hwm])
            for field, src in self._edge_core.items():
                np.copyto(snap._edge_core[field][:edge_hwm], src[:edge_hwm])
            for name, src in self._node_payloads.items():
                np.copyto(snap._node_payloads[name][:node_hwm], src[:node_hwm])
            for name, src in self._edge_payloads.items():
                np.copyto(snap._edge_payloads[name][:edge_hwm], src[:edge_hwm])
            np.copyto(snap._bucket_head, self._bucket_head)
            s2 = self._read_seq()
            if s2 == s1:
                snap._finalize_from_copy(meta_copy, node_hwm, edge_hwm)
                self.last_snapshot_attempts = attempt
                return snap
            torn_seen += 1
            time.sleep(_SNAPSHOT_RETRY_SLEEP_S)

        self.last_snapshot_attempts = int(max_retries)
        elapsed = time.monotonic() - t_start
        if torn_seen == 0:
            detail = (f"graph_seq stayed odd the whole time "
                      f"(first={first_seq}, last={last_seq}): the writer crashed "
                      f"mid-batch or a batch is running far longer than expected")
        else:
            detail = (f"{torn_seen} torn copies and {odd_seen} odd reads "
                      f"(first_seq={first_seq}, last_seq={last_seq}): the writer is "
                      f"mutating faster than the copy completes; shorten writer "
                      f"batches or raise max_retries")
        raise SnapshotRetryError(
            f"snapshot failed after {max_retries} attempts in {elapsed:.3f}s — {detail}. "
            f"If a reused 'into' replica was passed, its contents are now undefined.")


# ---------------------------------------------------------------------------
# Snapshot replica: same engine, private arrays, no-op guard
# ---------------------------------------------------------------------------

class GraphSnapshot(_GraphEngine):
    """
    Fully functional offline replica of the graph, allocator state included.
    Exposes the writer's exact mutation engine on private local arrays with a
    no-op edit guard, so the planner can run what-if edits with the same API.

    Slot-lifetime rule (design doc): slots are meaningful only within one
    planning cycle's snapshot. Replica edits are never written back and are
    completely removed when the replica is overwritten via
    reader.snapshot(into=this). All cross-cycle state must carry world
    coordinates, never slots.
    """

    def __init__(self, _token=None, _layout=None):
        if _token is not _SNAPSHOT_CONSTRUCTION_TOKEN:
            raise TypeError(
                "GraphSnapshot cannot be constructed directly; use "
                "SharedGraphReader.snapshot() or GraphSnapshot.copy()")
        self._layout = _layout
        self._node_cap = 0        # local node capacity (writer hwm + headroom)
        self._edge_cap = 0        # local edge capacity
        self._meta = None
        self._node_core = None
        self._edge_core = None
        self._bucket_head = None
        self._node_payloads = None
        self._edge_payloads = None

    # ------------------------------------------------------------------
    # Local buffer management (used by SharedGraphReader.snapshot)
    # ------------------------------------------------------------------

    def _ensure_local_capacity(self, node_cap: int, edge_cap: int, bucket_shape: tuple):
        """
        Grow-only (re)allocation of the private replica arrays. Reallocation
        is transparent: the GraphSnapshot object identity is preserved, only
        its buffers are replaced (the subsequent copy + finalize rebuild the
        full state, so no stale content can survive a reallocation).
        """
        changed = False
        if self._node_core is None or self._node_cap < node_cap:
            self._node_core = {f: np.zeros(node_cap, dtype=dt)
                               for f, dt in _NODE_CORE_SPECS}
            self._node_payloads = {n: np.zeros((node_cap,) + shape, dtype=dt)
                                   for n, (shape, dt, _d) in self._layout.node_payloads.items()}
            self._node_cap = node_cap
            changed = True
        if self._edge_core is None or self._edge_cap < edge_cap:
            self._edge_core = {f: np.zeros(edge_cap, dtype=dt)
                               for f, dt in _EDGE_CORE_SPECS}
            self._edge_payloads = {n: np.zeros((edge_cap,) + shape, dtype=dt)
                                   for n, (shape, dt, _d) in self._layout.edge_payloads.items()}
            self._edge_cap = edge_cap
            changed = True
        if self._bucket_head is None or self._bucket_head.shape != tuple(bucket_shape):
            self._bucket_head = np.full(bucket_shape, -1, dtype=np.int32)
            changed = True
        if self._meta is None:
            self._meta = np.zeros(1, dtype=_GRAPH_META_DTYPE)
            changed = True
        if changed:
            self._bind_state(self._meta, self._node_core, self._edge_core,
                             self._bucket_head, self._node_payloads,
                             self._edge_payloads, self._layout,
                             _NoOpEditGuard(), is_replica=True)

    def _finalize_from_copy(self, meta_copy: np.ndarray, node_hwm: int, edge_hwm: int):
        """
        Turn an accepted raw prefix copy into a coherent standalone replica:

        1. adopt the copied metadata (geometry, HWMs, counters, source seq),
           overriding capacities with the local buffer sizes;
        2. reset EVERY local row beyond the writer prefix, so nothing from a
           previous replica lifetime (local nodes/edges/payloads) survives;
        3. rebuild the free lists: copied in-prefix order is preserved, the
           first link leaving [0, hwm) is redirected to the local headroom
           chain (free-list truncation rule).

        Together with the prefix/bucket_head/metadata overwrite in
        snapshot(), a reused replica becomes observationally identical to a
        fresh snapshot of the same writer state, apart from buffer capacity.
        """
        m = self._meta
        m[:] = meta_copy
        m[0]['n_max'] = self._node_cap
        m[0]['e_max'] = self._edge_cap
        # ---- 2) reset all local rows beyond the writer prefix -------------
        nc, ec = self._node_core, self._edge_core
        nc['x'][node_hwm:] = 0.0
        nc['y'][node_hwm:] = 0.0
        nc['active'][node_hwm:] = 0
        nc['generation'][node_hwm:] = 0
        nc['update_version'][node_hwm:] = 0
        nc['first_edge'][node_hwm:] = -1
        nc['next_in_bucket'][node_hwm:] = -1    # rebuilt by truncation below
        for name, arr in self._node_payloads.items():
            arr[node_hwm:] = self._node_payload_defaults[name]
        ec['target'][edge_hwm:] = -1
        ec['length'][edge_hwm:] = 0.0
        ec['next'][edge_hwm:] = -1              # rebuilt by truncation below
        for name, arr in self._edge_payloads.items():
            arr[edge_hwm:] = self._edge_payload_defaults[name]
        # ---- 3) free-list truncation + local headroom chaining ------------
        m[0]['node_free_head'] = _truncate_free_list(
            nc['next_in_bucket'], int(meta_copy[0]['node_free_head']),
            node_hwm, self._node_cap, 'node')
        m[0]['edge_free_head'] = _truncate_free_list(
            ec['next'], int(meta_copy[0]['edge_free_head']),
            edge_hwm, self._edge_cap, 'edge')

    # ------------------------------------------------------------------
    # Independent deep copy (side-by-side what-if comparison)
    # ------------------------------------------------------------------

    def copy(self) -> 'GraphSnapshot':
        """Independent deep copy of this replica, allocator state included."""
        if self._meta is None:
            raise SharedGraphError("copy: this GraphSnapshot was never populated")
        dup = GraphSnapshot(_SNAPSHOT_CONSTRUCTION_TOKEN, self._layout)
        dup._ensure_local_capacity(self._node_cap, self._edge_cap,
                                   self._bucket_head.shape)
        dup._meta[:] = self._meta
        for field, src in self._node_core.items():
            np.copyto(dup._node_core[field], src)
        for field, src in self._edge_core.items():
            np.copyto(dup._edge_core[field], src)
        for name, src in self._node_payloads.items():
            np.copyto(dup._node_payloads[name], src)
        for name, src in self._edge_payloads.items():
            np.copyto(dup._edge_payloads[name], src)
        np.copyto(dup._bucket_head, self._bucket_head)
        return dup


def _truncate_free_list(next_arr: np.ndarray, copied_head: int, hwm: int,
                        local_cap: int, pool_name: str) -> int:
    """
    Free-list truncation for the replica (see plan, "Free-list truncation").

    The writer's free chain may leave [0, hwm) and continue through
    never-used SHM rows the replica did not copy. Rebuild locally:

    - chain the local headroom rows  hwm -> hwm+1 -> ... -> local_cap-1 -> -1;
    - preserve the copied free-list order within [0, hwm);
    - replace the first link that leaves [0, hwm) (or the terminal -1) with
      the first local headroom row;
    - if the copied head itself is outside [0, hwm), the headroom row IS the
      head.

    Returns the local free head. `next_arr` rows >= hwm are owned by this
    function; rows < hwm must already hold the copied writer links.
    """
    if local_cap > hwm:
        if local_cap - 1 > hwm:
            next_arr[hwm:local_cap - 1] = np.arange(hwm + 1, local_cap, dtype=np.int32)
        next_arr[local_cap - 1] = -1
        first_local = hwm
    else:
        # No headroom (cannot happen: snapshot() enforces headroom >= 1).
        first_local = -1
    if copied_head < 0 or copied_head >= hwm:
        return first_local
    cur = copied_head
    steps = 0
    while True:
        nxt = int(next_arr[cur])
        if nxt < 0 or nxt >= hwm:
            # First link leaving the copied prefix (never-used SHM tail or
            # terminal -1): splice the local headroom chain here.
            next_arr[cur] = first_local
            return copied_head
        cur = nxt
        steps += 1
        if steps > hwm:
            raise GraphCorruptionError(
                f"snapshot: copied {pool_name} free list has a cycle within "
                f"[0, {hwm}) — accepted copy is corrupt")
