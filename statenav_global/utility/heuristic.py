"""Selectable heuristic strategies for the heading-aware sparse A* planner.

All distance-like values returned by this module are estimated traversal times
in seconds. The graph snapshot and command arrays are read-only inputs owned by
``GlobalAStar``; this module never mutates planner or shared-memory state.

``euclidean``, ``relaxed_dijkstra``, ``reverse_dijkstra``,
``one_step_lookahead``, and ``alt`` are admissible for the current
curvature-limited edge-cost model. ``heading_euclidean`` assumes additive
point-turn cost, while ``manhattan`` and ``weighted_euclidean`` can overestimate
on an 8-connected / irregular graph; those three modes are experimental.
"""

import heapq
import math
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np


SUPPORTED_HEURISTICS = (
    "euclidean",
    "relaxed_dijkstra",
    "reverse_dijkstra",
    "one_step_lookahead",
    "heading_euclidean",
    "alt",
    "weighted_euclidean",
    "manhattan",
)


def normalize_heuristic_name(name: str) -> str:
    """Normalize and validate a configured A* heuristic name."""
    normalized = str(name).strip().lower().replace("-", "_")
    aliases = {
        "one_step": "one_step_lookahead",
        "heading": "heading_euclidean",
        "weighted": "weighted_euclidean",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_HEURISTICS:
        raise ValueError(
            f"graph_astar_heuristic must be one of {SUPPORTED_HEURISTICS}, "
            f"got {name!r}")
    return normalized


@dataclass(frozen=True)
class HeuristicContext:
    """Read-only data shared by the per-state heuristic functions."""

    snapshot: object
    goal_slot: int
    goal_x: float
    goal_y: float
    v_upper: float
    speed_discount: float
    arc_radius_factor: float
    wrap_to_pi: Callable[[float], float]


def euclidean_heuristic(state, context: HeuristicContext) -> float:
    """Straight-line travel time at the global maximum stored linear speed."""
    slot = int(state.node_slot)
    x = float(context.snapshot.node_x[slot])
    y = float(context.snapshot.node_y[slot])
    distance = math.hypot(x - context.goal_x, y - context.goal_y)
    return distance / (context.v_upper * context.speed_discount)


def manhattan_heuristic(state, context: HeuristicContext) -> float:
    """Axis-aligned travel-time estimate for comparison experiments.

    This can overestimate on the current 8-connected or irregular graph because
    a diagonal edge can be shorter than its Manhattan distance.
    """
    slot = int(state.node_slot)
    x = float(context.snapshot.node_x[slot])
    y = float(context.snapshot.node_y[slot])
    distance = abs(x - context.goal_x) + abs(y - context.goal_y)
    return distance / (context.v_upper * context.speed_discount)


def heading_euclidean_heuristic(
        state, context: HeuristicContext, w_upper: float) -> float:
    """Experimental Euclidean plus point-turn estimate."""
    slot = int(state.node_slot)
    if slot == context.goal_slot:
        return 0.0
    x = float(context.snapshot.node_x[slot])
    y = float(context.snapshot.node_y[slot])
    dx = context.goal_x - x
    dy = context.goal_y - y
    linear_time = math.hypot(dx, dy) / (
        context.v_upper * context.speed_discount)
    goal_direction = math.atan2(dy, dx)
    turn = abs(context.wrap_to_pi(goal_direction - float(state.theta)))
    angular_time = turn / (w_upper * context.speed_discount)
    return linear_time + angular_time


def weighted_euclidean_heuristic(
        state, context: HeuristicContext, weight: float) -> float:
    """Weighted-A* priority contribution based on the Euclidean lower bound."""
    return weight * euclidean_heuristic(state, context)


def _build_relaxed_adjacency(
        snapshot, cmd_v: np.ndarray, speed_discount: float,
        heading_to_layer: Callable[[float], int]) -> Tuple[
            List[List[Tuple[int, float]]], List[List[Tuple[int, float]]]]:
    """Build forward/reverse adjacency using linear time and no turn cost."""
    active = np.asarray(snapshot.node_active, dtype=bool)  # shape (n_max,)
    node_count = len(active)
    # Each list entry stores (neighbor_slot, relaxed_seconds) for one active node.
    forward = [[] for _ in range(node_count)]
    reverse = [[] for _ in range(node_count)]

    for source_raw in np.flatnonzero(active):
        source = int(source_raw)
        x_i = float(snapshot.node_x[source])
        y_i = float(snapshot.node_y[source])
        for target_raw, edge_raw in snapshot.neighbors(source):
            target = int(target_raw)
            edge_index = int(edge_raw)
            if target < 0 or target >= node_count or not active[target]:
                continue
            x_j = float(snapshot.node_x[target])
            y_j = float(snapshot.node_y[target])
            theta_move = math.atan2(y_j - y_i, x_j - x_i)
            layer = int(heading_to_layer(theta_move))
            v_i = float(cmd_v[source, layer])
            v_j = float(cmd_v[target, layer])
            edge_length = float(snapshot.edge_length[edge_index])
            if (not math.isfinite(edge_length) or edge_length <= 0.0
                    or not math.isfinite(v_i) or v_i <= 0.0
                    or not math.isfinite(v_j) or v_j <= 0.0):
                continue
            harmonic_speed = 2.0 * v_i * v_j / (v_i + v_j)
            edge_cost = edge_length / (harmonic_speed * speed_discount)
            forward[source].append((target, edge_cost))
            reverse[target].append((source, edge_cost))
    return forward, reverse


def _build_fastest_layer_reverse_adjacency(
        snapshot, cmd_v: np.ndarray,
        speed_discount: float) -> List[List[Tuple[int, float]]]:
    """Build reverse edges using each node's fastest finite linear command."""
    active = np.asarray(snapshot.node_active, dtype=bool)  # shape (n_max,)
    node_count = len(active)
    linear_speeds = cmd_v[:node_count]                     # shape (n_max, theta_layers)

    # Ignore turn cost and heading-layer constraints for this relaxation. Using
    # each node's fastest positive linear command keeps every relaxed edge cost
    # less than or equal to its corresponding real transition cost.
    valid_layers = (
        active[:, None]
        & np.isfinite(linear_speeds) & (linear_speeds > 0.0)
    )
    layer_slowness = np.full(
        linear_speeds.shape, math.inf, dtype=np.float64)    # shape (n_max, theta_layers)
    layer_slowness[valid_layers] = 1.0 / linear_speeds[valid_layers]
    best_node_slowness = np.min(layer_slowness, axis=1)    # shape (n_max,)

    # Preserve the sparse directed topology while reversing each usable edge.
    # Endpoint-slowness averaging is algebraically identical to dividing edge
    # length by the harmonic mean of the two endpoint speeds.
    reverse = [[] for _ in range(node_count)]
    for source_raw in np.flatnonzero(active):
        source = int(source_raw)
        source_slowness = float(best_node_slowness[source])
        if not math.isfinite(source_slowness):
            continue
        for target_raw, edge_raw in snapshot.neighbors(source):
            target = int(target_raw)
            edge_index = int(edge_raw)
            if target < 0 or target >= node_count or not active[target]:
                continue
            target_slowness = float(best_node_slowness[target])
            edge_length = float(snapshot.edge_length[edge_index])
            if (not math.isfinite(target_slowness)
                    or not math.isfinite(edge_length) or edge_length <= 0.0):
                continue
            edge_cost = edge_length * 0.5 * (
                source_slowness + target_slowness) / speed_discount
            reverse[target].append((source, edge_cost))
    return reverse


def _dijkstra_distances(
        adjacency: List[List[Tuple[int, float]]], source: int) -> np.ndarray:
    """Return shortest distances from one source over a non-negative graph."""
    distances = np.full(len(adjacency), math.inf, dtype=np.float64)  # shape (n_max,)
    distances[source] = 0.0
    queue = [(0.0, int(source))]
    while queue:
        current_cost, current = heapq.heappop(queue)
        if current_cost > distances[current]:
            continue
        for target, edge_cost in adjacency[current]:
            candidate = current_cost + edge_cost
            if candidate < distances[target]:
                distances[target] = candidate
                heapq.heappush(queue, (candidate, target))
    return distances


def relaxed_dijkstra_heuristic(state, cost_to_goal: np.ndarray) -> float:
    """Exact cost-to-go in the graph with all angular costs removed."""
    return float(cost_to_goal[int(state.node_slot)])


def reverse_dijkstra_heuristic(
        state, context: HeuristicContext, cost_to_goal: np.ndarray) -> float:
    """Look up cost-to-go computed from the goal over reversed graph edges.

    ``cost_to_goal`` has shape ``(n_max,)`` and contains relaxed traversal time
    in seconds for every graph slot. It is prepared once per planning snapshot.
    A Euclidean lower bound provides a finite fallback for unreachable entries.
    """
    value = float(cost_to_goal[int(state.node_slot)])
    if math.isfinite(value):
        return value
    return euclidean_heuristic(state, context)


def one_step_lookahead_heuristic(
        state, context: HeuristicContext, cmd_v: np.ndarray, cmd_w: np.ndarray,
        relaxed_cost_to_goal: np.ndarray,
        heading_to_layer: Callable[[float], int],
        transition_cost: Callable[..., float]) -> float:
    """Charge the exact first edge, then use relaxed cost-to-go thereafter."""
    source = int(state.node_slot)
    if source == context.goal_slot:
        return 0.0
    x_i = float(context.snapshot.node_x[source])
    y_i = float(context.snapshot.node_y[source])
    best = math.inf
    for target_raw, edge_raw in context.snapshot.neighbors(source):
        target = int(target_raw)
        edge_index = int(edge_raw)
        remaining = float(relaxed_cost_to_goal[target])
        if not math.isfinite(remaining):
            continue
        x_j = float(context.snapshot.node_x[target])
        y_j = float(context.snapshot.node_y[target])
        theta_move = math.atan2(y_j - y_i, x_j - x_i)
        layer = int(heading_to_layer(theta_move))
        first_edge = transition_cost(
            edge_length=float(context.snapshot.edge_length[edge_index]),
            theta_current=float(state.theta),
            theta_move=theta_move,
            v_i=float(cmd_v[source, layer]),
            v_j=float(cmd_v[target, layer]),
            w_i=float(cmd_w[source, layer]),
            speed_discount=context.speed_discount,
            arc_radius_factor=context.arc_radius_factor,
        )
        if math.isfinite(first_edge):
            best = min(best, first_edge + remaining)
    return best


def _select_landmarks(snapshot, count: int) -> List[int]:
    """Choose deterministic, spatially spread active graph landmarks."""
    active_slots = np.flatnonzero(np.asarray(snapshot.node_active, dtype=bool))
    if len(active_slots) == 0:
        return []
    count = min(int(count), len(active_slots))
    # Coordinates have shape (num_active, 2) and remain in snapshot slot order.
    coordinates = np.column_stack((
        np.asarray(snapshot.node_x)[active_slots],
        np.asarray(snapshot.node_y)[active_slots],
    )).astype(np.float64, copy=False)
    selected_local = [0]
    min_distance_sq = np.sum(
        (coordinates - coordinates[selected_local[0]]) ** 2, axis=1)
    while len(selected_local) < count:
        next_local = int(np.argmax(min_distance_sq))
        if next_local in selected_local:
            break
        selected_local.append(next_local)
        distance_sq = np.sum(
            (coordinates - coordinates[next_local]) ** 2, axis=1)
        min_distance_sq = np.minimum(min_distance_sq, distance_sq)
    return [int(active_slots[index]) for index in selected_local]


def alt_heuristic(
        state, goal_slot: int,
        landmark_tables: List[Tuple[np.ndarray, np.ndarray]]) -> float:
    """Directed ALT lower bound from precomputed landmark distance tables."""
    slot = int(state.node_slot)
    best = 0.0
    for from_landmark, to_landmark in landmark_tables:
        # Directed triangle inequalities provide two independent lower bounds.
        if (math.isfinite(from_landmark[goal_slot])
                and math.isfinite(from_landmark[slot])):
            best = max(
                best, float(from_landmark[goal_slot] - from_landmark[slot]))
        if (math.isfinite(to_landmark[slot])
                and math.isfinite(to_landmark[goal_slot])):
            best = max(
                best, float(to_landmark[slot] - to_landmark[goal_slot]))
    return max(0.0, best)


def build_heuristic(
        name: str, snapshot, cmd_v: np.ndarray, cmd_w: np.ndarray,
        goal_slot: int, v_upper: float, speed_discount: float,
        heading_to_layer: Callable[[float], int],
        transition_cost: Callable[..., float],
        wrap_to_pi: Callable[[float], float],
        weight: float = 1.5, alt_landmark_count: int = 4,
        arc_radius_factor: float = 1.5) -> Callable:
    """Prepare one configured heuristic and return a cached state callable."""
    heuristic_name = normalize_heuristic_name(name)
    goal_x = float(snapshot.node_x[goal_slot])
    goal_y = float(snapshot.node_y[goal_slot])
    context = HeuristicContext(
        snapshot=snapshot,
        goal_slot=int(goal_slot),
        goal_x=goal_x,
        goal_y=goal_y,
        v_upper=float(v_upper),
        speed_discount=float(speed_discount),
        arc_radius_factor=float(arc_radius_factor),
        wrap_to_pi=wrap_to_pi,
    )

    if heuristic_name == "euclidean":
        evaluate = lambda state: euclidean_heuristic(state, context)
    elif heuristic_name == "manhattan":
        evaluate = lambda state: manhattan_heuristic(state, context)
    elif heuristic_name == "heading_euclidean":
        active = np.asarray(snapshot.node_active, dtype=bool)  # shape (n_max,)
        values = cmd_w[: len(active)]                          # shape (n_max, theta_layers)
        valid = active[:, None] & np.isfinite(values) & (values > 0.0)
        if not np.any(valid):
            raise RuntimeError(
                "heading_euclidean heuristic requires a positive finite cmd_w")
        w_upper = float(np.max(values[valid]))
        evaluate = lambda state: heading_euclidean_heuristic(
            state, context, w_upper)
    elif heuristic_name == "weighted_euclidean":
        evaluate = lambda state: weighted_euclidean_heuristic(
            state, context, float(weight))
    elif heuristic_name == "reverse_dijkstra":
        # Relax heading and turn cost, then propagate from the goal backward.
        reverse = _build_fastest_layer_reverse_adjacency(
            snapshot, cmd_v, speed_discount)
        cost_to_goal = _dijkstra_distances(reverse, int(goal_slot))
        evaluate = lambda state: reverse_dijkstra_heuristic(
            state, context, cost_to_goal)
    else:
        # Graph-aware heuristics share one relaxed adjacency built from the
        # frozen snapshot's directional linear speeds, shape (n_max, degree).
        forward, reverse = _build_relaxed_adjacency(
            snapshot, cmd_v, speed_discount, heading_to_layer)
        if heuristic_name == "relaxed_dijkstra":
            # Run Dijkstra once from the goal over reversed edges so each table
            # entry is the relaxed shortest cost from that node to the goal.
            cost_to_goal = _dijkstra_distances(reverse, int(goal_slot))
            evaluate = lambda state: relaxed_dijkstra_heuristic(
                state, cost_to_goal)
        elif heuristic_name == "one_step_lookahead":
            cost_to_goal = _dijkstra_distances(reverse, int(goal_slot))
            evaluate = lambda state: one_step_lookahead_heuristic(
                state, context, cmd_v, cmd_w, cost_to_goal,
                heading_to_layer, transition_cost)
        else:
            landmarks = _select_landmarks(snapshot, int(alt_landmark_count))
            landmark_tables = [
                (_dijkstra_distances(forward, landmark),
                 _dijkstra_distances(reverse, landmark))
                for landmark in landmarks
            ]
            evaluate = lambda state: alt_heuristic(
                state, int(goal_slot), landmark_tables)

    # SearchState is immutable/hashable. Cache avoids repeated graph lookups when
    # a state is inserted into the heap more than once with an improved g-score.
    cache = {}

    def cached(state) -> float:
        if state not in cache:
            cache[state] = float(evaluate(state))
        return cache[state]

    return cached
