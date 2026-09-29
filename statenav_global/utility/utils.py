"""
utils for collision check
@author: huiming zhou
"""

import math
import numpy as np
import os
import sys
from pathlib import Path

def get_project_root():
    """
    Find the project root directory by looking for marker files.

    Returns:
        Path: Path to the project root directory (where setup.py, package.xml, or CMakeLists.txt exists)
    """
    # Start from this file's directory
    current = Path(__file__).resolve()

    # Walk up the directory tree looking for project root markers
    for parent in [current] + list(current.parents):
        # Check for common project root markers
        if (parent / "setup.py").exists() or \
           (parent / "package.xml").exists() or \
           (parent / "CMakeLists.txt").exists():
            return parent

    # Fallback: if no marker found, assume project root is 3 levels up from this file
    # (statenav_global/utility/utils.py -> statenav_global -> state_nav)
    return current.parents[2]


def get_source_inference_dir():
    """
    Return the path to statenav_global/inference/ in the SOURCE tree.

    When running from an installed colcon package, __file__ points to the
    install tree which lacks heavy assets (model weights, checkpoints/).
    This helper detects that case and resolves to the source tree instead.
    """
    # Direct path: statenav_global/inference/ relative to this file
    # (this file is statenav_global/utility/utils.py)
    local_inference = Path(__file__).resolve().parent.parent / "inference"

    # If the assets exist locally, we're running from source — use it directly
    if (local_inference / "checkpoints").exists():
        return str(local_inference)

    # We're in the install tree. Resolve to source via colcon workspace layout:
    #   install/<pkg>/lib/python3.x/site-packages/statenav_global/utility/utils.py
    #   -> <ws_root>/src/state_nav/statenav_global/inference/
    parts = Path(__file__).resolve().parts
    try:
        idx = parts.index("install")
        ws_root = Path(*parts[:idx])
        src_inference = ws_root / "src" / "state_nav" / "statenav_global" / "inference"
        if src_inference.exists():
            return str(src_inference)
    except ValueError:
        pass

    # Fallback: return the local path (caller will get FileNotFoundError with a clear path)
    return str(local_inference)


def set_random_seed(seed):
    # Import torch only when seeding it so math helpers remain dependency-free.
    import torch

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    print(f"Set random seed to {seed} in numpy and torch.")
    return rng

















# Math

def wrap_to_pi(angle):
    """
    Wrap finite scalar or array angles into [-π, π). π maps to -π so every
    physical heading has one canonical representation.

    Scalar inputs return ``float``; array-like inputs return an ``ndarray`` to
    preserve the existing utility API. NaN/inf raises rather than propagating
    through geometry and layer calculations.
    """
    angle_array = np.asarray(angle)
    if not np.all(np.isfinite(angle_array)):
        raise ValueError(f"wrap_to_pi: non-finite angle {angle!r}")
    wrapped = (angle_array + np.pi) % (2.0 * np.pi) - np.pi
    return float(wrapped) if np.ndim(angle_array) == 0 else wrapped


def heading_to_layer(theta_move, theta_min, theta_resolution, theta_layers):
    """
    Deterministic nearest-layer conversion of an exact movement heading to a
    discrete payload layer. The caller supplies the authoritative layer
    geometry; only the heading is quantized and the result is modulo-wrapped.

    Example (theta_min=-π, theta_resolution=π/4, theta_layers=8):
      heading ~0 -> layer 4, heading ~π/2 -> layer 6, π -> layer 0.
    """
    wrapped = wrap_to_pi(theta_move)
    offset = (wrapped - theta_min) % (2.0 * np.pi)
    return int(math.floor(offset / theta_resolution + 0.5)) % int(theta_layers)


def harmonic_mean_speed(v_i, v_j):
    """
    Return the harmonic mean of two already-validated endpoint linear speeds:

        v_h = 2 v_i v_j / (v_i + v_j)
    """
    return 2.0 * v_i * v_j / (v_i + v_j)


def transition_cost(edge_length, theta_current, theta_move, v_i, v_j, w_i,
                    speed_discount=1.0, arc_radius_factor=1.5):
    """
    Return a directional edge travel-time cost, or ``math.inf`` if invalid.

    Cost = (T_linear + T_turn) / speed_discount, where linear travel uses the
    harmonic-mean endpoint speed and turn time uses the source angular limit.
    ``speed_discount`` is the fraction of stored command limits assumed
    achievable. ``arc_radius_factor`` is the local arc-radius approximation of
    the curvature-limited model below (planning_config.yaml:
    arc_radius_factor); the default preserves the value this model shipped with.
    Invalid geometry or non-positive/non-finite command limits are
    never clamped into valid values.
    """
    # Reject invalid geometry and command limits before any division.
    if not math.isfinite(edge_length) or edge_length <= 0.0:
        return math.inf
    if not math.isfinite(v_i) or v_i <= 0.0:
        return math.inf
    if not math.isfinite(v_j) or v_j <= 0.0:
        return math.inf
    if not math.isfinite(w_i) or w_i <= 0.0:
        return math.inf

    v_h = harmonic_mean_speed(v_i, v_j)
    turn_angle = abs(wrap_to_pi(theta_move - theta_current))
    
    # --- Original point-turn model: rotate in place, then translate (times add) ---
    # linear_time = edge_length / v_h
    # angular_time = turn_angle / w_i
    # return (linear_time + angular_time) / speed_discount

    # --- Curvature-limited arc model: turn and translate together (bottleneck dominates) ---
    # Approximate the edge as a constant-curvature arc of radius R = 2*edge_length/turn_angle.
    # On that arc the source angular limit caps the achievable linear speed to v = min(v_h, w_i*R);
    # travel time is edge_length / v, with no separate additive turn term.
    if turn_angle <= 0.0:
        v = v_h                                            # straight edge: no curvature limit
    else:
        R = arc_radius_factor * edge_length / turn_angle   # local arc-radius approximation [m]
        v = min(v_h, w_i * R)                              # curvature-limited traversal speed [m/s]
    return (edge_length / v) / speed_discount


# ---------------------------------------------------------------------------
# Look-ahead circle path smoothing (pursuit-style corner cutting)
# ---------------------------------------------------------------------------
# From the current smoothing point P, intersect a circle of radius R with the
# remaining original polyline, take the forward-most hit L, and advance P by one
# step d along the chord P -> L. Repeat. Corners are therefore cut before they
# are reached; R controls how much is cut and d controls the output spacing.
#
# The smoothed polyline leaves the original polyline, so it is NOT traversable
# by construction. `is_valid_fn` is the gate that keeps it usable: every
# candidate step is sampled and, when a sample is rejected, the smoother reverts
# to the ORIGINAL vertices for that stretch (which are on-graph by construction).

def _polyline_dedupe(pts, eps=1e-9):
    """
    Drop consecutive duplicate vertices so no zero-length segment reaches the
    smoother (the risk-aware branch concatenates routes and can repeat a point).

    Args:
        pts: (N, 2) float array of [x, y] polyline vertices.
    Returns:
        deduped   : (M, 2) float array, M <= N, first/last vertex preserved.
        orig_to_new: list of length N mapping each ORIGINAL index to the index
                     in `deduped` of the surviving vertex at or before it (used
                     to remap anchor indices across the dedupe).
    """
    n = len(pts)
    keep = [0]
    orig_to_new = [0] * n
    for i in range(1, n):
        if math.hypot(pts[i, 0] - pts[keep[-1], 0], pts[i, 1] - pts[keep[-1], 1]) > eps:
            keep.append(i)
        orig_to_new[i] = len(keep) - 1
    return pts[keep], orig_to_new


def _forward_circle_hit(pts, k0, t0, cx, cy, radius):
    """
    Forward-most intersection of circle((cx, cy), radius) with polyline `pts`,
    considering only positions STRICTLY AHEAD of the progress position (k0, t0).

    The progress position is an exact (segment index, parameter) pair, not just
    a segment index. That matters: near the end of a segment the circle also has
    a root BEHIND the current point on that same segment, and accepting it makes
    the smoother walk backwards and then forwards forever (a livelock, since the
    smoothed point never reaches the polyline end). Rejecting t < t0 on segment
    k0 removes that root.

    "Forward-most" (last segment with a hit, farthest root on it) is what lets
    the smoother short-circuit a switchback rather than latching onto its near
    backward branch. Cost is O(N) per call; N is a few hundred nodes at most.

    Args:
        pts    : (N, 2) float array of polyline vertices.
        k0, t0 : progress position — segment index and parameter in [0, 1].
    Returns:
        (hit_x, hit_y), segment_index, t   -- or (None, -1, 0.0) when the circle
        has no forward intersection (remaining path shorter than radius, or the
        smoothing point drifted more than radius off the polyline).
    """
    best = None
    best_k = -1
    best_t = 0.0
    n_seg = len(pts) - 1
    for k in range(k0, n_seg):
        t_min = t0 if k == k0 else 0.0   # never look behind the progress position
        ax = pts[k, 0]
        ay = pts[k, 1]
        dx = pts[k + 1, 0] - ax          # segment direction (not normalized)
        dy = pts[k + 1, 1] - ay
        fx = ax - cx                     # circle center -> segment start
        fy = ay - cy
        a = dx * dx + dy * dy
        if a <= 0.0:
            continue                     # zero-length segment (defensive; deduped upstream)
        b = 2.0 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - radius * radius
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            continue                     # circle misses this segment's line
        sq = math.sqrt(disc)
        t_hit = None
        # larger root first: the farthest point of this segment inside the circle
        for t in ((-b + sq) / (2.0 * a), (-b - sq) / (2.0 * a)):
            if t_min <= t <= 1.0:
                t_hit = t
                break
        if t_hit is None:
            continue                     # no forward intersection within this segment
        best = (ax + t_hit * dx, ay + t_hit * dy)
        best_k = k                       # keep scanning: a later segment overrides
        best_t = t_hit
    return best, best_k, best_t


def _closest_segment_index(pts, lo, hi, px, py):
    """
    Index of the segment in pts[lo..hi] whose closest point to (px, py) is
    nearest — i.e. where the smoothed point currently sits along the ORIGINAL
    polyline.

    Only called when the validity gate rejects a step (rare), so the bounded
    scan costs nothing in the common path. It exists because the smoothed point
    and the look-ahead point have DIFFERENT progress: the look-ahead runs up to
    one radius ahead, so using its segment index as a snap-back target would
    skip every original vertex in between.
    """
    best_d2 = float('inf')
    best_k = lo
    for k in range(lo, hi + 1):
        ax = pts[k, 0]
        ay = pts[k, 1]
        dx = pts[k + 1, 0] - ax
        dy = pts[k + 1, 1] - ay
        a = dx * dx + dy * dy
        if a <= 0.0:
            t = 0.0                      # zero-length segment (defensive)
        else:
            t = ((px - ax) * dx + (py - ay) * dy) / a
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        cx = ax + t * dx                 # closest point on this segment
        cy = ay + t * dy
        d2 = (px - cx) * (px - cx) + (py - cy) * (py - cy)
        if d2 < best_d2:
            best_d2 = d2
            best_k = k
    return best_k


def _segment_is_valid(p0, p1, is_valid_fn, sample_step):
    """
    Sample the chord p0 -> p1 every `sample_step` metres (endpoint included) and
    require every sample to pass `is_valid_fn(x, y)`.

    p0 is assumed already validated, so only samples in (0, 1] are tested.
    Sampling (rather than testing only p1) is what prevents a step longer than
    one map cell from jumping over a thin non-traversable strip.
    """
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    seg_len = math.hypot(dx, dy)
    n_samples = max(1, int(math.ceil(seg_len / sample_step)))
    for i in range(1, n_samples + 1):
        t = i / n_samples
        if not is_valid_fn(p0[0] + t * dx, p0[1] + t * dy):
            return False
    return True


def _smooth_one_polyline(pts, radius, step, is_valid_fn, sample_step, stats):
    """
    Run the look-ahead circle smoother over ONE anchor-free polyline.

    Args:
        pts        : (N, 2) float array, N >= 3, deduped.
        radius     : R, look-ahead circle radius [m].
        step       : d, incremental advance per iteration [m], 0 < d < R.
        is_valid_fn: None, or f(x, y) -> bool traversability gate.
        sample_step: chord-sampling spacing [m] used by the gate.
        stats      : dict mutated in place with diagnostic counters.
    Returns:
        list of [x, y] — starts at pts[0] and ends exactly at pts[-1].
    """
    n = len(pts)
    last = (float(pts[-1, 0]), float(pts[-1, 1]))

    # --- iteration budget: enough for the whole polyline at step d, plus one
    # snap-back per vertex, plus slack. Purely a runaway-loop backstop.
    total_len = 0.0
    for i in range(n - 1):
        total_len += math.hypot(pts[i + 1, 0] - pts[i, 0], pts[i + 1, 1] - pts[i, 1])
    max_iters = int(total_len / step) * 4 + 2 * n + 20

    P = (float(pts[0, 0]), float(pts[0, 1]))   # current smoothing point
    out = [[P[0], P[1]]]
    # Progress position along the ORIGINAL polyline as an exact (segment index,
    # parameter) pair. Monotone non-decreasing: the look-ahead point may never
    # move backward, which is what guarantees the loop terminates.
    k = 0
    t_prog = 0.0
    snap_floor = 0   # highest ORIGINAL vertex index already emitted by a snap-back

    it = 0
    while it < max_iters:
        it += 1

        # --- termination: close enough to the end of this polyline ---------
        if math.hypot(last[0] - P[0], last[1] - P[1]) <= step:
            break

        # --- 1) look-ahead circle -> forward-most hit L on the original path
        L, k_hit, t_hit = _forward_circle_hit(pts, k, t_prog, P[0], P[1], radius)
        if L is None:
            # No forward intersection: the remaining path is shorter than R (the
            # normal end-game), or P has drifted > R off the path. Aim straight
            # at the polyline end; the validity gate below still applies.
            L = last
        else:
            k = k_hit                    # _forward_circle_hit never looks behind (k, t_prog)
            t_prog = t_hit

        # --- 2/3) advance one step d along the chord P -> L ------------------
        ux = L[0] - P[0]
        uy = L[1] - P[1]
        norm = math.hypot(ux, uy)
        accepted = False
        if norm > 1e-12:
            cand = (P[0] + step * ux / norm, P[1] + step * uy / norm)
            if is_valid_fn is None or _segment_is_valid(P, cand, is_valid_fn, sample_step):
                out.append([cand[0], cand[1]])
                P = cand
                accepted = True
        if accepted:
            continue

        # --- 4) gate rejected the cut (or L coincided with P): snap back -----
        # Fall back to the next ORIGINAL vertex AHEAD of the current progress
        # position. Progress (k, t_prog) sits on segment k, so that vertex is
        # pts[k + 1]; original vertices are on-graph and therefore traversable
        # by construction. Repeated rejections then walk the original polyline
        # vertex by vertex, reproducing it exactly.
        #
        # The target MUST be derived from where P ACTUALLY is, which is neither
        # of the two obvious shortcuts:
        #   - an independent counter that lags the progress pointer re-emits
        #     vertices P has already passed, sending the path back to the start
        #     and retracing the whole polyline from there;
        #   - the look-ahead pointer k runs up to one radius AHEAD of P, so
        #     k + 1 skips every original vertex between P and the look-ahead.
        # Projecting P onto the polyline gives the segment it sits alongside;
        # the next vertex after that is the correct fallback. snap_floor then
        # forces strict advancement, which is what makes this branch terminate.
        stats['rejected_steps'] += 1
        seg = _closest_segment_index(pts, snap_floor, k, P[0], P[1])
        j = max(seg + 1, snap_floor + 1)
        if j > n - 1:
            break                        # no original vertex left ahead: end-game
        target = (float(pts[j, 0]), float(pts[j, 1]))
        # P can already sit exactly on pts[j] (it was snapped there, or a step
        # landed on it); emitting it again would put a zero-length segment in
        # the output. Advancing the pointers alone still guarantees progress.
        if math.hypot(target[0] - P[0], target[1] - P[1]) > 1e-9:
            if is_valid_fn is not None and not _segment_is_valid(
                    P, target, is_valid_fn, sample_step):
                # P was validated when it was emitted and pts[j] is on the
                # original (traversable-by-construction) path, so this chord
                # should be clean; count it instead of hiding it.
                stats['unsafe_snapbacks'] += 1
            out.append([target[0], target[1]])
        P = target
        snap_floor = j
        # Resume the circle search at the snapped vertex (start of its outgoing
        # segment). j == n-1 (snapped onto the final vertex) instead pins the
        # progress to the end of the last segment.
        k = min(j, n - 2)
        t_prog = 1.0 if j > k else 0.0

    if it >= max_iters:
        stats['iter_cap_hit'] += 1

    # --- endpoint preservation: always finish exactly on the original end ---
    if math.hypot(out[-1][0] - last[0], out[-1][1] - last[1]) > 1e-9:
        out.append([last[0], last[1]])
    return out


def smooth_path_lookahead(path, smoothing_radius, step, anchor_indices=(),
                          is_valid_fn=None, sample_step=None):
    """
    Smooth `path` by look-ahead circle corner cutting (see module section above).

    Args:
        path            : sequence of [x, y] — the planned path (N >= 3 to do
                          any work; shorter paths are returned unchanged).
        smoothing_radius: R [m], look-ahead circle radius. Larger R cuts corners
                          harder and further ahead.
        step            : d [m], advance per iteration and therefore the output
                          vertex spacing. Must satisfy 0 < d < R.
        anchor_indices  : indices into `path` that MUST survive smoothing. The path is
                          split at each anchor and every piece smoothed on its
                          own, so an anchor is never cut away. Endpoints are
                          always preserved and need not be listed.
        is_valid_fn     : None, or f(x, y) -> bool answering "is this free-space
                          point traversable?". Without it the smoother is purely
                          geometric and CAN cut through non-traversable terrain.
        sample_step     : chord-sampling spacing [m] for is_valid_fn; defaults
                          to `step`. Set to about one map cell so a step longer
                          than a cell cannot jump over a blocked cell.

    Returns:
        (smoothed, stats) where `smoothed` is a list of [x, y] floats with
        smoothed[0] == path[0] and smoothed[-1] == path[-1], and `stats` is a
        dict of diagnostic counters:
            rejected_steps   : smoothing steps vetoed by is_valid_fn
            unsafe_snapbacks : snap-back chords that themselves failed the gate
            iter_cap_hit     : pieces that hit the runaway-loop iteration cap
            n_in / n_out     : vertex counts before / after

    Raises:
        ValueError on invalid parameters or non-finite path coordinates.
    """
    # --- parameter validation (fail loud; these are config values) ----------
    R = float(smoothing_radius)
    d = float(step)
    if not math.isfinite(R) or R <= 0.0:
        raise ValueError(f"smoothing_radius must be finite and > 0, got {smoothing_radius!r}")
    if not math.isfinite(d) or d <= 0.0:
        raise ValueError(f"step must be finite and > 0, got {step!r}")
    if d >= R:
        raise ValueError(
            f"step ({d}) must be strictly smaller than smoothing_radius ({R}); "
            f"otherwise one advance overshoots the look-ahead point and the "
            f"smoother degenerates into the original polyline")
    s_step = d if sample_step is None else float(sample_step)
    if not math.isfinite(s_step) or s_step <= 0.0:
        raise ValueError(f"sample_step must be finite and > 0, got {sample_step!r}")

    stats = {'rejected_steps': 0, 'unsafe_snapbacks': 0, 'iter_cap_hit': 0,
             'n_in': 0, 'n_out': 0}

    if path is None or len(path) == 0:
        return [], stats
    pts_in = np.asarray(path, dtype=np.float64)          # (N, 2) input polyline
    if pts_in.ndim != 2 or pts_in.shape[1] != 2:
        raise ValueError(f"path must be shaped (N, 2), got {pts_in.shape}")
    if not np.all(np.isfinite(pts_in)):
        bad = int(np.argmax(~np.all(np.isfinite(pts_in), axis=1)))
        raise ValueError(
            f"path has non-finite coordinates at index {bad}: {pts_in[bad].tolist()}")
    stats['n_in'] = int(pts_in.shape[0])
    if pts_in.shape[0] < 3:
        out = [[float(p[0]), float(p[1])] for p in pts_in]
        stats['n_out'] = len(out)
        return out, stats

    # --- dedupe, then remap the anchors onto the deduped indexing -----------
    pts, orig_to_new = _polyline_dedupe(pts_in)          # (M, 2)
    n = int(pts.shape[0])
    if n < 3:
        out = [[float(p[0]), float(p[1])] for p in pts]
        stats['n_out'] = len(out)
        return out, stats

    anchors = set()
    for a in anchor_indices or ():
        a = int(a)
        if 0 < a < stats['n_in'] - 1:                    # endpoints are implicit anchors
            a_new = orig_to_new[a]
            if 0 < a_new < n - 1:
                anchors.add(a_new)

    # --- smooth each anchor-delimited piece independently -------------------
    # boundaries: [0, anchor_0, ..., anchor_k, n-1]; consecutive pairs are the
    # pieces. Anchors therefore appear verbatim in the output.
    boundaries = [0] + sorted(anchors) + [n - 1]
    out = []
    for b in range(len(boundaries) - 1):
        s_i = boundaries[b]
        e_i = boundaries[b + 1]
        piece = pts[s_i:e_i + 1]                         # (P, 2) sub-polyline, endpoints inclusive
        if piece.shape[0] < 3:
            piece_out = [[float(p[0]), float(p[1])] for p in piece]
        else:
            piece_out = _smooth_one_polyline(piece, R, d, is_valid_fn, s_step, stats)
        if out:
            piece_out = piece_out[1:]                    # shared anchor already emitted
        out.extend(piece_out)

    stats['n_out'] = len(out)
    return out, stats



# Geometry

# Module-level utility functions that can be imported directly
def get_ray(start, end):
    """Calculate ray origin and direction from start to end point."""
    orig = [start.x, start.y]
    direc = [end.x - start.x, end.y - start.y]
    return orig, direc


def get_dist(start, end):
    """Calculate Euclidean distance between two nodes."""
    return math.hypot(end.x - start.x, end.y - start.y)


def is_intersect_rec(start, end, o, d, a, b):
    v1 = [o[0] - a[0], o[1] - a[1]]
    v2 = [b[0] - a[0], b[1] - a[1]]
    v3 = [-d[1], d[0]]

    div = np.dot(v2, v3)

    if div == 0:
        return False

    t1 = np.linalg.norm(np.cross(v2, v1)) / div
    t2 = np.dot(v1, v3) / div

    if t1 >= 0 and 0 <= t2 <= 1:
        shot = BasicNode((o[0] + t1 * d[0], o[1] + t1 * d[1]))
        dist_obs = get_dist(start, shot)
        dist_seg = get_dist(start, end)
        if dist_obs <= dist_seg:
            return True

    return False

def is_intersect_circle(o, d, a, r, delta = 0):
    d2 = np.dot(d, d)

    if d2 == 0:
        return False

    t = np.dot([a[0] - o[0], a[1] - o[1]], d) / d2

    if 0 <= t <= 1:
        shot = BasicNode((o[0] + t * d[0], o[1] + t * d[1]))
        if get_dist(shot, BasicNode(a)) <= r + delta:
            return True

    return False



def is_inside_circle(node, circle, delta=None):

    """
    
    Args:
        node: BasicNode object with x and y attributes
        circle: Tuple, containing (x, y, r) representing circle
        delta: Tolerance value for point-in-circle check (default: None)
    
    Returns:
        bool: True if point is inside circle, False otherwise
    """
    
    if delta is None:
        delta = 0.00

    x, y, r = circle
    if math.hypot(node.x - x, node.y - y) <= r + delta:
        return True
    return False


def is_inside_circles(node, circles, delta=None):
    if delta is None:
        delta = 0.00

    for circle in circles:
        if is_inside_circle(node, circle, delta):
            return True
    return False


class BasicNode:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None


class Utils:
    def __init__(self):
        self.delta = 0.01#0.5
        self.obs_circle = []
        self.obs_rectangle = []
        self.obs_boundary = []

    def get_obs_vertex(self):
        delta = self.delta
        obs_list = []

        for (ox, oy, w, h) in self.obs_rectangle:
            vertex_list = [[ox - delta, oy - delta],
                           [ox + w + delta, oy - delta],
                           [ox + w + delta, oy + h + delta],
                           [ox - delta, oy + h + delta]]
            obs_list.append(vertex_list)
            
        delta = 0.01
        for (ox, oy, w, h) in self.obs_boundary:
            vertex_list = [[ox - delta, oy - delta],
                           [ox + w + delta, oy - delta],
                           [ox + w + delta, oy + h + delta],
                           [ox - delta, oy + h + delta]]
            obs_list.append(vertex_list)

        return obs_list


    def is_collision(self, start, end, delta=None):
        
        if self.is_inside_obs(start, delta) or self.is_inside_obs(end, delta):
            return True

        o, d = get_ray(start, end)
        obs_vertex = self.get_obs_vertex()

        for (v1, v2, v3, v4) in obs_vertex:
            if self.is_intersect_rec(start, end, o, d, v1, v2):
                return True
            if self.is_intersect_rec(start, end, o, d, v2, v3):
                return True
            if self.is_intersect_rec(start, end, o, d, v3, v4):
                return True
            if self.is_intersect_rec(start, end, o, d, v4, v1):
                return True

        for (x, y, r) in self.obs_circle:
            if self.is_intersect_circle(o, d, (x, y), r):
                return True

        return False
    
                

    def is_inside_obs(self, node, delta=None):
        
        if delta is None:
            delta = self.delta

        for (x, y, r) in self.obs_circle:
            if math.hypot(node.x - x, node.y - y) <= r + delta:
                return True

        for (x, y, w, h) in self.obs_rectangle:
            if 0 <= node.x - (x - delta) <= w + 2 * delta \
                    and 0 <= node.y - (y - delta) <= h + 2 * delta:
                return True

        for (x, y, w, h) in self.obs_boundary:
            if 0 <= node.x - (x) <= w \
                    and 0 <= node.y - (y) <= h:
                return True

        return False


# Expose free functions as class-level utilities for convenient access
Utils.set_random_seed = set_random_seed
Utils.wrap_to_pi = wrap_to_pi
Utils.heading_to_layer = heading_to_layer
Utils.harmonic_mean_speed = harmonic_mean_speed
Utils.transition_cost = transition_cost
Utils.get_ray = get_ray
Utils.get_dist = get_dist
Utils.is_intersect_rec = is_intersect_rec
Utils.is_intersect_circle = is_intersect_circle
Utils.is_inside_circle = is_inside_circle
