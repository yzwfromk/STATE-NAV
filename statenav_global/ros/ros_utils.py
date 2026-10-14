"""
ROS2 utilities for mapping module.
Contains ROS2 message publishing functions for traversability maps.
"""

import time

import matplotlib
import numpy as np
from std_msgs.msg import (
    ColorRGBA,
    Float32MultiArray,
    MultiArrayLayout,
    MultiArrayDimension,
)
from grid_map_msgs.msg import GridMap
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from statenav_global.core.shared_graph import GraphCorruptionError


# Matplotlib coolwarm_r defines node colors; RViz interpolates those endpoint
# colors across edges.
_COOLWARM_R = matplotlib.colormaps["coolwarm_r"]
_GRAPH_VMAX_WARNING_EMITTED = False
_GRAPH_TIMING_LOGGED = False

_GRAPH_NODE_Z_OFFSET = 0.1
_GRAPH_NODE_DIAMETER = 0.15
_GRAPH_EDGE_WIDTH = 0.03
_GRAPH_EDGE_ALPHA = 0.9


def _log_warning(msg, node=None):
    """Helper function for logging warnings (compatible with ROS2 or print)."""
    if node is not None:
        node.get_logger().warning(msg)
    else:
        print(f"[WARN] {msg}")


def _log_info(msg, node=None):
    """Helper function for logging info (compatible with ROS2 or print)."""
    if node is not None:
        node.get_logger().info(msg)
    else:
        print(f"[INFO] {msg}")


def _elevation_z(global_map, x, y, offset, fallback):
    """Return the elevation-map height at world (x, y) plus offset, else fallback.

    Reuses the map's own xy2grid() convention so marker heights cannot drift from
    the world-coordinate convention used elsewhere.

    Args:
        global_map: Map instance exposing ElevationMap + xy2grid(); may be None.
        x, y:       world coordinates in metres of the query point.
        offset:     metres added above the surface so the marker sits just above terrain.
        fallback:   z (metres) used when elevation is unavailable / NaN / out of range.
    """
    # No map or no elevation grid (e.g. planning process in ROS-message mode)
    if global_map is None or not global_map.is_ElevationMap_built or global_map.ElevationMap is None:
        return fallback
    row, col = global_map.xy2grid(float(x), float(y))
    if row is None or col is None:   # query point outside the map bounds
        return fallback
    elevation = global_map.ElevationMap[row, col]   # scalar metres, NaN for unobserved cells
    if not np.isfinite(elevation):
        return fallback
    return float(elevation) + offset


def _graph_node_rgba(cmd_v, viz_vmax, node=None):
    """Reduce heading-layer cmd_v values and map them to coolwarm_r colors."""
    global _GRAPH_VMAX_WARNING_EMITTED

    cmd_v = np.asarray(cmd_v)  # cmd_v (H, 8): command velocity by heading layer
    finite = np.isfinite(cmd_v)
    finite_count = finite.sum(axis=1)
    finite_sum = np.where(finite, cmd_v, 0.0).sum(axis=1)

    mean_v = np.full(cmd_v.shape[0], np.nan, dtype=np.float32)  # mean_v (H,)
    valid_mean = finite_count > 0
    mean_v[valid_mean] = finite_sum[valid_mean] / finite_count[valid_mean]

    viz_vmax = float(viz_vmax)
    if not np.isfinite(viz_vmax) or viz_vmax <= 0.0:
        viz_vmax = 1.0
        if not _GRAPH_VMAX_WARNING_EMITTED:
            _log_warning(
                "[GraphVisualization] viz_vmax is non-finite or non-positive; "
                "using 1.0 for color normalization",
                node,
            )
            _GRAPH_VMAX_WARNING_EMITTED = True

    finite_mean = np.isfinite(mean_v)
    t = np.zeros(mean_v.shape[0], dtype=np.float64)
    t[finite_mean] = np.clip(mean_v[finite_mean] / viz_vmax, 0.0, 1.0)

    rgba = np.empty((mean_v.shape[0], 4), dtype=np.float64)  # rgba (H, 4)
    rgba[finite_mean] = _COOLWARM_R(t[finite_mean])
    rgba[~finite_mean] = (0.5, 0.5, 0.5, 1.0)
    return rgba


def _graph_node_elevations(global_map, x, y, active):
    """Return graph-node elevations with the visualization offset applied."""
    z = np.full(x.shape[0], _GRAPH_NODE_Z_OFFSET, dtype=np.float32)
    if global_map.ElevationMap is None:
        return z

    for slot in np.flatnonzero(active):
        # Deliberately reuse xy2grid() so visualization cannot drift from the
        # map's existing world-coordinate convention.
        row, col = global_map.xy2grid(float(x[slot]), float(y[slot]))
        if row is None or col is None:
            continue
        elevation = global_map.ElevationMap[row, col]
        if np.isfinite(elevation):
            z[slot] = float(elevation) + _GRAPH_NODE_Z_OFFSET
    return z


def _recover_graph_edge_owners(node_first_edge, edge_next, active, edge_hwm):
    """Recover the source-node slot for each allocated adjacency edge row."""
    owner = np.full(edge_hwm, -1, dtype=np.int64)  # owner (E,): source slot per row
    slots = np.flatnonzero(active)
    frontier = np.asarray(node_first_edge[slots], dtype=np.int64)
    frontier_owner = slots.astype(np.int64, copy=False)

    # Edge rows do not store their source node, so ownership must be recovered
    # from adjacency-chain membership. In a valid graph, propagation depth is
    # bounded by the maximum node degree. The explicit edge_hwm guard prevents
    # a corrupted cycle from hanging the world-model process.
    iteration_count = 0
    while frontier.size:
        valid = frontier >= 0
        frontier = frontier[valid]
        frontier_owner = frontier_owner[valid]
        if frontier.size == 0:
            break
        if np.any(frontier >= edge_hwm):
            bad_row = int(frontier[frontier >= edge_hwm][0])
            raise GraphCorruptionError(
                f"Adjacency-chain edge row {bad_row} is outside edge_hwm "
                f"{edge_hwm}."
            )

        owner[frontier] = frontier_owner
        frontier = edge_next[frontier]

        iteration_count += 1
        if iteration_count > edge_hwm:
            raise GraphCorruptionError(
                "Adjacency-chain traversal exceeded edge_hwm; "
                "the graph may contain an edge-chain cycle."
            )

    return owner, iteration_count


def _extract_undirected_graph_edges(gw, active):
    """Return active endpoint slots for each undirected graph edge exactly once."""
    edge_hwm = gw.edge_hwm
    owner, iteration_count = _recover_graph_edge_owners(
        np.asarray(gw.node_first_edge[:active.shape[0]]),
        np.asarray(gw.edge_next[:edge_hwm]),
        active,
        edge_hwm,
    )
    target = np.asarray(gw.edge_target[:edge_hwm])

    target_active = np.zeros(edge_hwm, dtype=bool)
    valid_target = (target >= 0) & (target < active.shape[0])
    target_active[valid_target] = active[target[valid_target]]
    sel = valid_target & target_active & (owner >= 0) & (owner < target)

    a = owner[sel]  # a (U,): first endpoint node slots
    b = target[sel].astype(np.int64, copy=False)  # b (U,): second endpoint slots
    return a, b, iteration_count


def _graph_point(x, y, z):
    point = Point()
    point.x = float(x)
    point.y = float(y)
    point.z = float(z)
    return point


def _graph_color(rgba, alpha=None):
    color = ColorRGBA()
    color.r = float(rgba[0])
    color.g = float(rgba[1])
    color.b = float(rgba[2])
    color.a = float(rgba[3] if alpha is None else alpha)
    return color


def Rviz_vis_graph(global_map, pub, node, frame_id):
    """Publish the shared-memory graph as node and edge RViz2 markers."""
    global _GRAPH_TIMING_LOGGED

    gw = global_map.graph_writer
    if gw is None or pub is None:
        return
    if node is None:
        _log_warning(
            "[GraphVisualization] node is None, cannot get clock stamp - skipping",
            node,
        )
        return

    started_at = time.perf_counter()
    stamp = node.get_clock().now().to_msg()
    marker_array = MarkerArray()

    delete_all = Marker()
    delete_all.header.frame_id = frame_id
    delete_all.header.stamp = stamp
    delete_all.action = Marker.DELETEALL
    marker_array.markers.append(delete_all)

    node_hwm = gw.node_hwm
    if node_hwm == 0:
        pub.publish(marker_array)
        return

    # Graph mutation and visualization both run on the main-loop thread, so
    # direct graph_writer reads are safe without taking a snapshot.
    x = np.asarray(gw.node_x[:node_hwm])  # x (H,): node world x coordinates
    y = np.asarray(gw.node_y[:node_hwm])
    active = np.asarray(gw.node_active[:node_hwm], dtype=bool)
    cmd_v = np.asarray(gw.node_payload("cmd_v")[:node_hwm])

    rgba = _graph_node_rgba(cmd_v, global_map.viz_vmax, node)
    z = _graph_node_elevations(global_map, x, y, active)
    a, b, _ = _extract_undirected_graph_edges(gw, active)

    node_marker = Marker()
    node_marker.header.frame_id = frame_id
    node_marker.header.stamp = stamp
    node_marker.ns = "graph_nodes"
    node_marker.id = 0
    node_marker.type = Marker.SPHERE_LIST
    node_marker.action = Marker.ADD
    node_marker.pose.orientation.w = 1.0
    node_marker.scale.x = _GRAPH_NODE_DIAMETER
    node_marker.scale.y = _GRAPH_NODE_DIAMETER
    node_marker.scale.z = _GRAPH_NODE_DIAMETER

    for slot in np.flatnonzero(active):
        node_marker.points.append(_graph_point(x[slot], y[slot], z[slot]))
        node_marker.colors.append(_graph_color(rgba[slot]))
    marker_array.markers.append(node_marker)

    edge_marker = Marker()
    edge_marker.header.frame_id = frame_id
    edge_marker.header.stamp = stamp
    edge_marker.ns = "graph_edges"
    edge_marker.id = 0
    edge_marker.type = Marker.LINE_LIST
    edge_marker.action = Marker.ADD
    edge_marker.pose.orientation.w = 1.0
    edge_marker.scale.x = _GRAPH_EDGE_WIDTH

    for slot_a, slot_b in zip(a, b):
        edge_marker.points.append(_graph_point(x[slot_a], y[slot_a], z[slot_a]))
        edge_marker.points.append(_graph_point(x[slot_b], y[slot_b], z[slot_b]))
        edge_marker.colors.append(_graph_color(rgba[slot_a], _GRAPH_EDGE_ALPHA))
        edge_marker.colors.append(_graph_color(rgba[slot_b], _GRAPH_EDGE_ALPHA))
    marker_array.markers.append(edge_marker)

    if not _GRAPH_TIMING_LOGGED:
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        _log_info(
            f"[GraphVisualization] Built {int(active.sum())} node and "
            f"{a.size} edge markers in {elapsed_ms:.1f} ms",
            node,
        )
        _GRAPH_TIMING_LOGGED = True

    pub.publish(marker_array)
