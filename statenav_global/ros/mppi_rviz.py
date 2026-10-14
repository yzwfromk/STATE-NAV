"""RViz MarkerArray visualization for the MPPI local planner.

Replaces the former interactive Matplotlib figure.  The per-tick cost drops from a
full canvas rasterization (~20 ms, measured at 480x400) to marker payload
construction plus a handful of elevation lookups (~0.03 ms), because RViz does the
rendering in its own process.

Only the three planner-private overlays are published here.  The traversability
background and the global path are deliberately NOT republished: the world model
already publishes /travmap_visualization (GridMap) and the global planner already
publishes /global_path, so duplicating them would cost bandwidth and hide the
originals behind an opaque second copy.

This module owns every rclpy / visualization_msgs import.  mppi_planner.py stays
ROS-free so headless tests and non-ROS callers keep working; main.py constructs
this object and injects it through MPPIPlanner.set_visualizer().
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
from rclpy.duration import Duration
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray


# ── Marker identity ───────────────────────────────────────────────────────────
# Fixed (namespace, id) pairs.  Every tick republishes ALL of them, either as ADD
# with this tick's geometry or as DELETE when the element does not exist right now.
# An ADD on an existing id replaces it in RViz, so no id can ever show stale data,
# and the array length is constant so RViz never accumulates markers.
MARKER_NAMESPACE = "mppi_plan"
MARKER_ID_ROLLOUT = 0     # MPPI predicted rollout        (LINE_STRIP)
MARKER_ID_WAYPOINT = 1    # lookahead waypoint            (SPHERE)
MARKER_ID_TEXT = 3        # cmd / limit / solve-time HUD  (TEXT_VIEW_FACING)

# ── Colours (r, g, b, a) ──────────────────────────────────────────────────────
COLOR_ROLLOUT = (1.0, 1.0, 0.0, 1.0)    # yellow
COLOR_WAYPOINT = (0.0, 1.0, 0.0, 1.0)   # green, matches the old "go" marker
COLOR_TEXT = (1.0, 1.0, 1.0, 1.0)       # white

# ── Geometry ──────────────────────────────────────────────────────────────────
LINE_WIDTH_ROLLOUT = 0.05    # m, LINE_STRIP scale.x
WAYPOINT_DIAMETER = 0.15     # m, SPHERE scale.{x,y,z}
TEXT_HEIGHT = 0.25           # m, TEXT_VIEW_FACING scale.z

# ── Height placement ──────────────────────────────────────────────────────────
# Ground markers sit at (terrain elevation at their own xy) + ELEV_BASE_OFFSET +
# their own small per-marker offset.  The rollout instead stays flat at the robot
# pose z so it represents the planner trajectory in the robot's current xy plane.
# The per-marker offsets exist only to separate overlapping ground markers so they
# cannot z-fight with each other or with the GridMap.
ELEV_BASE_OFFSET = 0.2       # m above the terrain surface
Z_OFFSET_ROLLOUT = 0.0       # m, preserve the robot pose z exactly
Z_OFFSET_WAYPOINT = 0.06     # m, above terrain so the sphere remains visible
Z_OFFSET_TEXT = 1.0          # m, floats above the robot so it does not overlap the plan

# Markers expire this many control periods after publication.  Per-tick correctness
# is handled by the explicit DELETE above; this only covers the planner node dying,
# which no republish can catch.  5x (not 2-3x) because a legitimately slow tick must
# not blink the markers: main.py only warns past 1.5x ctrl_period.
LIFETIME_CTRL_PERIODS = 5.0


def _elevation_z(global_map: Any, x: float, y: float,
                 offset: float, fallback: float) -> float:
    """Return the elevation-map height at world (x, y) plus offset, else fallback.

    Local copy of statenav_global.ros.ros_utils._elevation_z.  It is duplicated
    rather than imported on purpose: that module evaluates
    matplotlib.colormaps["coolwarm_r"] at import time, and importing Matplotlib into
    the control process is exactly what this file exists to avoid.  Keep the
    xy2grid() call below in sync with that original so marker heights cannot drift
    from the world-coordinate convention the rest of the stack uses.

    Args:
        global_map: map exposing ElevationMap + xy2grid() (may be None, e.g. tests).
        x, y:       world coordinates in metres of the query point.
        offset:     metres added above the surface so the marker sits above terrain.
        fallback:   z (metres) used when elevation is unavailable / NaN / out of range.
    """
    # No map or no elevation grid yet (startup, before the first elevation message).
    if (global_map is None
            or not getattr(global_map, "is_ElevationMap_built", False)
            or getattr(global_map, "ElevationMap", None) is None):
        return fallback

    # Guard BEFORE xy2grid(): it does int((map_xmax - x)/map_resolution) with no
    # finite check, so a NaN x raises ValueError and an inf x raises OverflowError.
    # That exception would latch visualization off for the rest of the run.
    if not (math.isfinite(x) and math.isfinite(y)):
        return fallback

    # Silent in-bounds precheck.  xy2grid() prints an unthrottled "Out of range."
    # line for every out-of-extent query; the MPPI rollout extends up to
    # horizon*dt*maximum_cmd_v ahead of the robot and is NOT constrained to the map
    # (mppi_planner._lookup_cached_map clamps the grid index, not the state), so it
    # crosses the boundary while the robot itself is still legitimately inside.
    try:
        if not (global_map.map_xmin <= x <= global_map.map_xmax
                and global_map.map_ymin <= y <= global_map.map_ymax):
            return fallback
    except AttributeError:
        return fallback

    row, col = global_map.xy2grid(float(x), float(y))
    if row is None or col is None:   # query point outside the map bounds
        return fallback
    elevation = global_map.ElevationMap[row, col]   # scalar metres, NaN if unobserved
    if not np.isfinite(elevation):
        return fallback
    return float(elevation) + offset


def _finite_xy(x: float, y: float) -> bool:
    """True when both coordinates can safely become a marker position."""
    return math.isfinite(x) and math.isfinite(y)


def status_text(mode: str, stamp_s: float, v: float, w: float,
                v_limit: float, w_limit: float, solve_ms: float,
                has_plan: bool = True) -> str:
    """Build the compact FSM-state / command HUD string.

    ``has_plan`` is False when the planner returned no
    predicted path this tick; that is otherwise visible only as a MISSING rollout marker,
    which reads identically to "visualization is broken" and hides a held position.

    ``v_limit``/``w_limit`` are the limits the command was actually PROJECTED against, so
    the percentages read as domain usage: 100% means no retuning can raise the command.
    The caller passes the knot-zero command domain for them (see _build_text); passing the
    caller's terrain ceiling instead reports headroom the projection never offered.
    """
    v_use = abs(v) / v_limit if v_limit > 1.0e-9 else 0.0
    w_use = abs(w) / w_limit if w_limit > 1.0e-9 else 0.0
    head = f"[{mode}] t={stamp_s:.1f} solve={solve_ms:.0f}ms"
    if not has_plan:
        head += " NO PLAN"
    # Limits print at the command's own precision now that they are the per-cell domain:
    # that is a continuous harmonic mean, and one decimal rounded 0.28 to "0.3", leaving
    # the printed ratio visibly disagreeing with the percentage next to it.
    return (f"{head}\n"
            f"cmd. v={v:.2f}/{v_limit:.2f}({v_use * 100:.0f}%) "
            f"w={w:.2f}/{w_limit:.2f}({w_use * 100:.0f}%)")


class MPPIMarkerPublisher:
    """Publish the MPPI plan as a MarkerArray, replacing the Matplotlib figure.

    Duck-types the two methods MPPIPlanner calls on its visualizer slot:
    ``update()`` (per control tick) and ``reset_history()`` (planner reset).
    """

    def __init__(self, node: Any, planner: Any, *, ctrl_period: float,
                 topic: str = "/mppi_plan_markers", frame_id: str = "map",
                 qos_profile: Any = None) -> None:
        """
        Args:
            node:        rclpy Node owning the publisher and the clock.
            planner:     MPPIPlanner; read for global_map and last_cmd_limits.
            ctrl_period: control period in seconds, sets the marker lifetime.
            topic:       MarkerArray topic name.
            frame_id:    TF frame every marker is stamped in.
            qos_profile: QoS for the publisher; must be RELIABLE for RViz's default
                         subscription to connect (BEST_EFFORT silently never matches).
        """
        self.node = node
        self.planner = planner
        self.frame_id = frame_id
        self.enabled = True
        self._failed = False   # latched after an error, mirrors the old visualizer

        if qos_profile is None:
            self._publisher = node.create_publisher(MarkerArray, topic, 1)
        else:
            self._publisher = node.create_publisher(MarkerArray, topic, qos_profile)
        self._lifetime = Duration(
            seconds=float(ctrl_period) * LIFETIME_CTRL_PERIODS).to_msg()

    # ── Marker construction helpers ───────────────────────────────────────────

    def _new_marker(self, marker_id: int, marker_type: int) -> Marker:
        """Return an ADD marker with the shared header/ns/lifetime already filled."""
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = MARKER_NAMESPACE
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        # An all-zero quaternion makes RViz log "uninitialized quaternion" per marker.
        marker.pose.orientation.w = 1.0
        marker.lifetime = self._lifetime
        return marker

    def _delete_marker(self, marker_id: int) -> Marker:
        """Return a DELETE for one id.  Harmless no-op if RViz never saw that id."""
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = MARKER_NAMESPACE
        marker.id = marker_id
        marker.action = Marker.DELETE
        return marker

    @staticmethod
    def _set_color(marker: Marker, rgba: tuple[float, float, float, float]) -> None:
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = rgba

    # ── Public API (duck-typed against the former MPPIVisualizer) ─────────────

    def reset_history(self) -> None:
        """Drop every marker in this namespace on planner reset.

        DELETEALL is correct here and only here: using it per tick would delete and
        re-add the markers each frame, which flickers visibly at the control rate.
        """
        if not self.enabled or self._failed:
            return
        try:
            clear = Marker()
            clear.header.frame_id = self.frame_id
            clear.header.stamp = self.node.get_clock().now().to_msg()
            clear.ns = MARKER_NAMESPACE
            clear.action = Marker.DELETEALL
            array = MarkerArray()
            array.markers = [clear]
            self._publisher.publish(array)
        except Exception as exc:   # visualization must never stop command publication
            self._failed = True
            print(f"[MPPI rviz] disabled after error in reset_history: {exc}")

    def update(
        self,
        path: np.ndarray,
        waypoint: Optional[np.ndarray],
        robot_pose: np.ndarray,
        v: float,
        w: float,
        result: Optional[Any],
        *,
        mode: str,
        source: str,
        stamp_s: float,
    ) -> None:
        """Publish this tick's MarkerArray.  Signature matches MPPIVisualizer.update().

        ``path`` (the global path) and ``source`` are accepted for call-site
        compatibility and intentionally unused: /global_path already carries the
        former, and the latter only labelled the old figure.

        Args:
            path:       (N, 2) global path in world xy.  Unused, see above.
            waypoint:   (2,) lookahead waypoint in world xy, or None.
            robot_pose: (4,) robot pose (x, y, yaw, z).  A legacy (3,) input uses
                        the robot-position elevation fallback for rollout z.
            v, w:       issued linear / angular command.
            result:     MPPIResult; predicted_path is (H + 1, 3) as (x, y, yaw).
            mode:       Current FSM mode string, shown in the HUD text.
            stamp_s:    sim-clock seconds, shown in the HUD text.
        """
        if not self.enabled or self._failed:
            return
        try:
            global_map = getattr(self.planner, "global_map", None)
            robot_x, robot_y = float(robot_pose[0]), float(robot_pose[1])

            # Per-tick shared height fallback for ground markers and legacy callers.
            # Cascade: marker elevation -> robot-position elevation -> ground at z=0.
            z_base = _elevation_z(global_map, robot_x, robot_y,
                                  ELEV_BASE_OFFSET, ELEV_BASE_OFFSET)
            robot_z = (float(robot_pose[3]) if len(robot_pose) >= 4
                       else z_base)

            # Every id is rebuilt on every tick, ADD or DELETE.
            markers = [
                self._build_rollout(result, robot_z),
                self._build_waypoint(waypoint, global_map, z_base),
                self._build_text(mode, stamp_s, v, w, result,
                                 robot_x, robot_y, global_map, z_base),
            ]
            array = MarkerArray()
            array.markers = markers
            self._publisher.publish(array)
        except Exception as exc:   # visualization must never stop command publication
            self._failed = True
            print(f"[MPPI rviz] disabled after error: {exc}")

    # ── Per-element builders; each returns ADD or DELETE for its fixed id ─────

    def _build_rollout(self, result: Optional[Any], robot_z: float) -> Marker:
        """MPPI predicted rollout as a yellow LINE_STRIP at the robot pose z.

        NOTE ON COORDINATES: predicted_path columns are (x, y, yaw) in the world
        frame.  The old Matplotlib code plotted them as (y, x) purely to orient the
        2D axes; that swap must NOT be carried over here or the whole plan renders
        mirrored across the diagonal.
        """
        predicted = None if result is None else getattr(result, "predicted_path", None)
        if predicted is None or len(predicted) < 2 or not math.isfinite(robot_z):
            return self._delete_marker(MARKER_ID_ROLLOUT)

        marker = self._new_marker(MARKER_ID_ROLLOUT, Marker.LINE_STRIP)
        marker.scale.x = LINE_WIDTH_ROLLOUT
        self._set_color(marker, COLOR_ROLLOUT)
        for state in predicted:
            x, y = float(state[0]), float(state[1])
            if not _finite_xy(x, y):
                # MPPIResult.valid already rejects non-finite paths, so this only
                # fires if an unvalidated path ever reaches here.  Drop the whole
                # strip rather than emit a NaN vertex: RViz rejects the array.
                return self._delete_marker(MARKER_ID_ROLLOUT)
            marker.points.append(Point(
                x=x, y=y,
                z=robot_z + Z_OFFSET_ROLLOUT))
        return marker

    def _build_waypoint(self, waypoint: Optional[np.ndarray], global_map: Any,
                        z_base: float) -> Marker:
        """Lookahead waypoint as a green SPHERE at world (wp[0], wp[1])."""
        if waypoint is None:
            return self._delete_marker(MARKER_ID_WAYPOINT)
        x, y = float(waypoint[0]), float(waypoint[1])
        if not _finite_xy(x, y):
            return self._delete_marker(MARKER_ID_WAYPOINT)

        marker = self._new_marker(MARKER_ID_WAYPOINT, Marker.SPHERE)
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = _elevation_z(
            global_map, x, y,
            ELEV_BASE_OFFSET + Z_OFFSET_WAYPOINT, z_base + Z_OFFSET_WAYPOINT)
        marker.scale.x = WAYPOINT_DIAMETER
        marker.scale.y = WAYPOINT_DIAMETER
        marker.scale.z = WAYPOINT_DIAMETER
        self._set_color(marker, COLOR_WAYPOINT)
        return marker

    def _build_text(self, mode: str, stamp_s: float, v: float, w: float,
                    result: Optional[Any], robot_x: float, robot_y: float,
                    global_map: Any, z_base: float) -> Marker:
        """Command / limit / solve-time HUD, floating above the robot."""
        if not _finite_xy(robot_x, robot_y):
            return self._delete_marker(MARKER_ID_TEXT)

        # Denominator for the usage percentages: the command domain knot zero -- the
        # command being displayed -- was actually projected against.  That is the cell
        # under the robot floored to conservative_speed_v/w, NOT the caller's terrain
        # limit, which the cell may tighten far below: a command pinned AT a 0.5 rad/s
        # cell ceiling printed as "0.47/1.0 (47%)" and read as unused authority that the
        # projection was never going to give.  last_cmd_limits remains the fallback for the
        # ticks before the first solve of a plan, where no domain has been measured yet.
        knot_zero_limits = self.planner.last_knot_zero_limits
        v_limit, w_limit = (
            self.planner.last_cmd_limits if knot_zero_limits is None
            else knot_zero_limits)
        solve_ms = 0.0 if result is None else float(result.solve_time_s) * 1.0e3

        marker = self._new_marker(MARKER_ID_TEXT, Marker.TEXT_VIEW_FACING)
        marker.pose.position.x = robot_x
        marker.pose.position.y = robot_y
        marker.pose.position.z = _elevation_z(
            global_map, robot_x, robot_y,
            ELEV_BASE_OFFSET + Z_OFFSET_TEXT, z_base + Z_OFFSET_TEXT)
        marker.scale.z = TEXT_HEIGHT
        self._set_color(marker, COLOR_TEXT)
        # has_plan mirrors exactly what _build_rollout draws, so the HUD and the missing
        # yellow line can never disagree about whether a plan exists.
        has_plan = (result is not None
                    and getattr(result, "predicted_path", None) is not None
                    and len(result.predicted_path) >= 2)
        marker.text = status_text(
            mode, stamp_s, v, w, v_limit, w_limit, solve_ms,
            has_plan)
        return marker
