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
from statenav_global.world_model.shared_graph import GraphCorruptionError


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


def publish_costmap_float32multiarray(global_map, frame_id, global_costmap_pub, node=None):
    """
    Publish costmap using Float32MultiArray (original method).

    Args:
        global_map: Map instance (BaseMap or subclass)
        frame_id: ROS frame ID
        global_costmap_pub: ROS2 publisher for costmap
        node: ROS2 node instance (optional, for logging)
    """
    # Full-map grid index bounds
    row_xmin, row_xmax = global_map.map_size_rows - 1, 0
    col_ymin, col_ymax = global_map.map_size_cols - 1, 0
    xmin_p, ymin_p = global_map.map_xmin, global_map.map_ymin
    xmax_p = row_xmin * global_map.map_resolution + xmin_p + 1e-4
    ymax_p = col_ymin * global_map.map_resolution + ymin_p + 1e-4

    msg = Float32MultiArray()
    # Metadata
    msg.data.extend([
        global_map.map_resolution, global_map.TraversabilityMap_theta_resolution,
        xmin_p, xmax_p, ymin_p, ymax_p,
        global_map.TraversabilityMap_theta_min, global_map.TraversabilityMap_theta_max
    ])
    # Data
    for i in range(0, global_map.map_size_rows):
        for j in range(0, global_map.map_size_cols):
            for k in range(global_map.TraversabilityMap_size_layers):
                msg.data.extend(global_map.TraversabilityMap[i, j, k, :])
    
    global_costmap_pub.publish(msg)


def publish_costmap_gridmap(global_map, frame_id, global_costmap_pub, node=None):
    """
    Publish costmap using GridMap message type.

    Args:
        global_map: Map instance (BaseMap or subclass)
        frame_id: ROS frame ID
        global_costmap_pub: ROS2 publisher for costmap
        node: ROS2 node instance (required for time stamps)
    """
    # Full-map grid index bounds
    row_xmin, row_xmax = global_map.map_size_rows - 1, 0
    col_ymin, col_ymax = global_map.map_size_cols - 1, 0
    xmin_p, ymin_p = global_map.map_xmin, global_map.map_ymin
    xmax_p = row_xmin * global_map.map_resolution + xmin_p + 1e-4
    ymax_p = col_ymin * global_map.map_resolution + ymin_p + 1e-4

    num_rows, num_cols = row_xmin, col_ymin
    
    grid_map_msg = GridMap()
    grid_map_msg.header.frame_id = frame_id
    if node is not None:
        grid_map_msg.header.stamp = node.get_clock().now().to_msg()
    grid_map_msg.info.resolution = global_map.map_resolution
    grid_map_msg.info.length_x = xmax_p - xmin_p
    grid_map_msg.info.length_y = ymax_p - ymin_p
    grid_map_msg.info.pose.position.x = (xmin_p + xmax_p) / 2.0
    grid_map_msg.info.pose.position.y = (ymin_p + ymax_p) / 2.0
    grid_map_msg.info.pose.position.z = 0.0
    grid_map_msg.info.pose.orientation.w = 1.0
    
    channel_names = ["cmd_v", "cmd_w", "auxiliary_score", "auxiliary_score_std"]
    
    # Create layers for each (theta_layer, channel) combination
    for theta_layer in range(global_map.TraversabilityMap_size_layers):
        for channel_idx, channel_name in enumerate(channel_names):
            layer_name = f"{channel_name}_theta_{theta_layer}"
            grid_map_msg.layers.append(layer_name)
            
            layer_data = Float32MultiArray()
            layout = MultiArrayLayout()
            layout.dim = [
                MultiArrayDimension(label="column_index", size=num_cols, stride=num_rows),
                MultiArrayDimension(label="row_index", size=num_rows, stride=1)
            ]
            layout.data_offset = 0
            layer_data.layout = layout
            
            # Extract data (row-major order)
            for j in range(col_ymax, col_ymin):
                for i in range(row_xmax, row_xmin):
                    layer_data.data.append(float(global_map.TraversabilityMap[i, j, theta_layer, channel_idx]))
            
            grid_map_msg.data.append(layer_data)
    
    # Add metadata layers
    metadata = {
        "theta_resolution": global_map.TraversabilityMap_theta_resolution,
        "theta_min": global_map.TraversabilityMap_theta_min,
        "theta_max": global_map.TraversabilityMap_theta_max,
        "num_theta_layers": float(global_map.TraversabilityMap_size_layers),
        "xmin": xmin_p, "xmax": xmax_p, "ymin": ymin_p, "ymax": ymax_p
    }
    
    for meta_name, meta_value in metadata.items():
        grid_map_msg.layers.append(meta_name)
        meta_layer = Float32MultiArray()
        meta_layer.layout = layout
        meta_layer.data = [float(meta_value)] * (num_rows * num_cols)
        grid_map_msg.data.append(meta_layer)
    
    global_costmap_pub.publish(grid_map_msg)


def populate_map_from_float32multiarray(global_map, msg, node=None):
    """Populate existing map object from Float32MultiArray message."""
    if len(msg.data) < 8:
        _log_warning("[PathPlanner] Map message too short, skipping", node)
        return False
        
    data = msg.data
    idx = 0
        
    if not global_map.is_TraversabilityMap_built: # If the map is not built, extract the metadata
        # Extract metadata
        global_map.map_resolution = data[idx]; idx += 1
        global_map.TraversabilityMap_theta_resolution = data[idx]; idx += 1
        global_map.map_xmin = data[idx]; idx += 1
        global_map.map_xmax = data[idx]; idx += 1
        global_map.map_ymin = data[idx]; idx += 1
        global_map.map_ymax = data[idx]; idx += 1
        global_map.TraversabilityMap_theta_min = data[idx]; idx += 1
        global_map.TraversabilityMap_theta_max = data[idx]; idx += 1
        
        # Calculate dimensions
        global_map.map_size_rows = int((global_map.map_xmax - global_map.map_xmin) / global_map.map_resolution)
        global_map.map_size_cols = int((global_map.map_ymax - global_map.map_ymin) / global_map.map_resolution)
        global_map.TraversabilityMap_size_layers = int(2 * np.round(np.pi / global_map.TraversabilityMap_theta_resolution))

    else:
        idx = 8 # Skip the metadata but keep the data index
        
    # Calculate expected data size
    expected_size = global_map.map_size_rows * global_map.map_size_cols * global_map.TraversabilityMap_size_layers * 4
    actual_size = len(data) - idx
    
    if actual_size < expected_size:
        _log_warning(f"[PathPlanner] Map data incomplete. Expected {expected_size}, got {actual_size}", node)
        return False
    
    # Reshape data into 4D array [rows, cols, theta_layers, 4_channels]
    map_data = np.array(data[idx:idx+expected_size], dtype=np.float32)
    global_map.TraversabilityMap = map_data.reshape(
        (global_map.map_size_rows, 
         global_map.map_size_cols, 
         global_map.TraversabilityMap_size_layers, 
         4)
    )
    
    global_map.is_TraversabilityMap_built = True # Avoiding unnecessary initialization of the map. Assuming the map metadata does not change.
    # _log_info(f"Map updated: {global_map.map_size_rows}x{global_map.map_size_cols}x{global_map.TraversabilityMap_size_layers}", node)
    return True


def populate_map_from_gridmap(global_map, msg, node=None):
    """Populate existing map object from GridMap message."""
    # Extract metadata from GridMap info
    global_map.map_resolution = msg.info.resolution
    length_x = msg.info.length_x
    length_y = msg.info.length_y
    
    # Calculate map bounds from center position
    center_x = msg.info.pose.position.x
    center_y = msg.info.pose.position.y
    
    global_map.map_xmin = center_x - length_x / 2.0
    global_map.map_xmax = center_x + length_x / 2.0
    global_map.map_ymin = center_y - length_y / 2.0
    global_map.map_ymax = center_y + length_y / 2.0
    
    # Get dimensions from first layer
    if len(msg.layers) == 0 or len(msg.data) == 0:
        _log_warning("[PathPlanner] GridMap has no layers", node)
        return False

    first_layer = msg.data[0]
    if len(first_layer.layout.dim) < 2:
        _log_warning("[PathPlanner] GridMap layer has insufficient dimensions", node)
        return False
    
    # Note: GridMap uses column_index, row_index order
    num_cols = first_layer.layout.dim[0].size
    num_rows = first_layer.layout.dim[1].size
    
    global_map.map_size_rows = num_rows
    global_map.map_size_cols = num_cols
    
    # Extract theta information from metadata layers
    theta_resolution = None
    theta_min = None
    theta_max = None
    num_theta_layers = None
    
    for i, layer_name in enumerate(msg.layers):
        if i < len(msg.data):
            if layer_name == "theta_resolution":
                theta_resolution = msg.data[i].data[0]
            elif layer_name == "theta_min":
                theta_min = msg.data[i].data[0]
            elif layer_name == "theta_max":
                theta_max = msg.data[i].data[0]
            elif layer_name == "num_theta_layers":
                num_theta_layers = int(msg.data[i].data[0])
    
    # If metadata not found, infer from layer names
    if theta_resolution is None:
        theta_indices = set()
        for layer_name in msg.layers:
            if "_theta_" in layer_name:
                try:
                    theta_idx = int(layer_name.split("_theta_")[-1])
                    theta_indices.add(theta_idx)
                except:
                    pass
        
        if len(theta_indices) > 0:
            num_theta_layers = len(theta_indices)
            if theta_min is None:
                theta_min = -np.pi
            if theta_max is None:
                theta_max = np.pi - (np.pi / num_theta_layers)
            if theta_resolution is None:
                theta_resolution = (theta_max - theta_min) / (num_theta_layers - 1) if num_theta_layers > 1 else np.pi / 4
    
    if num_theta_layers is None:
        _log_warning("[PathPlanner] Could not determine number of theta layers, defaulting to 8", node)
        num_theta_layers = 8
        theta_min = -np.pi
        theta_max = np.pi - np.pi/4
        theta_resolution = np.pi / 4
    
    global_map.TraversabilityMap_size_layers = num_theta_layers
    global_map.TraversabilityMap_theta_min = theta_min if theta_min is not None else -np.pi
    global_map.TraversabilityMap_theta_max = theta_max if theta_max is not None else (np.pi - np.pi/4)
    global_map.TraversabilityMap_theta_resolution = theta_resolution if theta_resolution is not None else np.pi/4
    
    # Reconstruct 4D array from GridMap layers
    channel_names = ["cmd_v", "cmd_w", "auxiliary_score", "auxiliary_score_std"]
    
    global_map.TraversabilityMap = np.full(
        (num_rows, num_cols, num_theta_layers, 4), 
        np.nan, 
        dtype=np.float32
    )
    
    layer_idx = 0
    for theta_layer in range(num_theta_layers):
        for channel_idx, channel_name in enumerate(channel_names):
            layer_name = f"{channel_name}_theta_{theta_layer}"
            
            if layer_idx < len(msg.layers) and layer_idx < len(msg.data):
                if msg.layers[layer_idx] == layer_name:
                    layer_data = np.array(msg.data[layer_idx].data, dtype=np.float32)
                    
                    # Reshape from column-major to row-major
                    layer_data_reshaped = layer_data.reshape((num_cols, num_rows)).T
                    
                    global_map.TraversabilityMap[:, :, theta_layer, channel_idx] = layer_data_reshaped
                    layer_idx += 1
                else:
                    _log_warning(f"[PathPlanner] Expected layer {layer_name}, got {msg.layers[layer_idx]}", node)
                    layer_idx += 1

    global_map.is_TraversabilityMap_built = True
    # _log_info(f"Map updated from GridMap: {num_rows}x{num_cols}x{num_theta_layers}", node)
    return True


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


def Rviz_vis_travmap(global_map, pub, node, frame_id):
    """Publish elevation + north-direction cmd_v as a GridMap for Rviz2 via grid_map_rviz_plugin.
    Intended to be called at low frequency (<=0.5 Hz). NaN cells are rendered as empty."""
    import array as _array

    if not global_map.is_TraversabilityMap_built:
        return

    # Spatial dimensions of the global map grid
    num_rows = global_map.map_size_rows  # x-axis extent in cells
    num_cols = global_map.map_size_cols  # y-axis extent in cells

    # Lazy-allocate flat float32 buffers on first call (one per layer); safe since map
    # dimensions are fixed by the time is_TraversabilityMap_built is True
    if global_map._vis_flat_bufs is None:
        _n = num_rows * num_cols
        global_map._vis_flat_bufs = [np.empty(_n, dtype=np.float32),   # elevation layer [num_rows*num_cols]
                                      np.empty(_n, dtype=np.float32)]    # cmd_v layer     [num_rows*num_cols]

    # Theta layer index for north (0 deg): idx = (0 - theta_min) / theta_resolution
    north_theta_idx = int(round((0.0 - global_map.TraversabilityMap_theta_min) / global_map.TraversabilityMap_theta_resolution))

    # Build GridMap header and spatial metadata
    grid_map_msg = GridMap()
    grid_map_msg.header.frame_id = frame_id
    grid_map_msg.header.stamp = node.get_clock().now().to_msg()
    grid_map_msg.info.resolution = float(global_map.map_resolution)
    grid_map_msg.info.length_x = float(global_map.map_xmax - global_map.map_xmin)
    grid_map_msg.info.length_y = float(global_map.map_ymax - global_map.map_ymin)
    grid_map_msg.info.pose.position.x = float((global_map.map_xmin + global_map.map_xmax) / 2.0)
    grid_map_msg.info.pose.position.y = float((global_map.map_ymin + global_map.map_ymax) / 2.0)
    grid_map_msg.info.pose.position.z = 0.0
    grid_map_msg.info.pose.orientation.w = 1.0

    # Layers: (layer_name, 2D array of shape [num_rows, num_cols])
    layers_to_publish = [
        ("elevation", global_map.ElevationMap),                                  # global elevation grid [num_rows, num_cols]
        ("cmd_v",     global_map.TraversabilityMap[:, :, north_theta_idx, 0]),   # cmd_v at north heading [num_rows, num_cols]
    ]

    for layer_idx, (layer_name, data_array) in enumerate(layers_to_publish):
        grid_map_msg.layers.append(layer_name)

        layer_data = Float32MultiArray()
        layout = MultiArrayLayout()

        # Column-major layout: outer dim = cols (y-axis), inner dim = rows (x-axis)
        # Required by grid_map convention for correct Rviz2 display
        dim_col = MultiArrayDimension()
        dim_col.label = "column_index"
        dim_col.size = num_cols
        dim_col.stride = num_rows

        dim_row = MultiArrayDimension()
        dim_row.label = "row_index"
        dim_row.size = num_rows
        dim_row.stride = 1

        layout.dim = [dim_col, dim_row]
        layout.data_offset = 0
        layer_data.layout = layout

        # Fill pre-allocated buffer in column-major order via a single typed copy:
        # reshape(num_cols, num_rows) makes the flat buffer a view with shape [num_cols, num_rows],
        # then copying data_array.T (shape [num_cols, num_rows]) writes buf[j*num_rows+i]=data[i,j].
        # This avoids intermediate ravel/astype allocations. NaN values pass through unchanged.
        np.copyto(global_map._vis_flat_bufs[layer_idx].reshape(num_cols, num_rows), data_array.T)
        # array.array avoids creating per-element Python float objects (5-10x faster than tolist())
        layer_data.data = _array.array('f', global_map._vis_flat_bufs[layer_idx])

        grid_map_msg.data.append(layer_data)

    pub.publish(grid_map_msg)
