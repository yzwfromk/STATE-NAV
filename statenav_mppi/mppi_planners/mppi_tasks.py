"""Task layer: what WP_TRACKING adds.

A mixin over PathTrackingMPPI, composed into the single MPPIPlanner in
mppi_planner.py.  The class here owns only what its mode ADDS to the generic tracker --
an entrypoint -- while everything else stays one implementation below it.

There is no mode registry.  Each mixin states its own policy inline at its entrypoint --
the log tag, the command ceiling and the rollout geometry -- because the mixin IS the mode,
and a bool cannot name a mode that does not exist.
"""

from __future__ import annotations

import math

import numpy as np

from statenav_mppi.mppi_core.mppi_types import MPPIPathTrackingResult




####################################### WP TRACKING #######################################
# Nominal navigation: actuator-maximum ceiling, tapered toward the path end.  Everything
# else it needs already lives in PathTrackingMPPI.

class WpTrackingMixin:
    """WP_TRACKING entrypoint.  Mixed into MPPIPlanner over PathTrackingMPPI."""

    def _goal_taper_caps(
        self, robot_x: float, robot_y: float, path: np.ndarray,
    ) -> tuple[float, float]:
        """Return this tick's ``(v, w)`` mode ceiling, tapered toward the path end.

        The ceiling is walked from the actuator maxima DOWN to the conservative speed as
        the robot closes on ``path[-1]``, and never past it.  Under the coupled simplex
        equality law the projection spends whatever rectangle it is handed, so shrinking
        the rectangle is the only way to slow the robot down -- no cost term can do it.

        The profile is the braking law ``v(d) = sqrt(2 * a * d)``: the speed from which
        deceleration ``a`` still bleeds off inside ``d``.  Following it costs exactly ``a``,
        so with ``a = goal_taper_acc_frac * lin_acc`` and that fraction below 1 the rate box
        can always follow the taper -- which is what stops the taper from driving the rate
        box and the simplex disjoint and spraying the rate-infeasible warning at every
        arrival.  A linear ramp has no such property; it demands unbounded deceleration as
        the distance goes to zero.

        Both axes are scaled by the SAME factor.  Tapering ``v`` alone would leave the
        angular ceiling at its maximum, and the equality law would then pour the freed
        command budget into yaw -- the robot would swing about instead of slowing down.

        ``d_end`` is the straight-line distance to the last path point, deliberately the
        same quantity main.py's PATH_END_STOP guard measures, so the two agree on where the
        goal is by construction rather than by coincidence.
        """
        maximum_v = float(self.config.maximum_cmd_v)
        maximum_w = float(self.config.maximum_cmd_w)
        # The taper's landing rectangle.  min() with the maxima so a config whose
        # conservative speed exceeds the actuator maximum cannot RAISE the ceiling; this
        # mirrors the same guard in MPPICore._command_limits.
        floor_v = min(maximum_v, float(self.config.conservative_speed_v))
        floor_w = min(maximum_w, float(self.config.conservative_speed_w))
        # A path with no points has no end to approach.  _track_path rejects it a moment
        # later with a tagged reason; returning the untapered ceiling here keeps this
        # helper from being the thing that reports it.
        if path is None or len(path) == 0:
            return maximum_v, maximum_w
        d_end = math.hypot(
            float(path[-1][0]) - robot_x, float(path[-1][1]) - robot_y)
        if not math.isfinite(d_end):
            # Non-finite pose or path point.  _track_path validates both and reports it;
            # leaving the ceiling untapered means this helper adds no second failure mode.
            return maximum_v, maximum_w
        # (m) distance still available to bleed speed off in, measured to the point where
        # the taper must already be AT the conservative speed.
        usable = max(0.0, d_end - float(self.config.goal_taper_hold_m))
        taper_acc = float(self.config.goal_taper_acc_frac) * float(self.config.lin_acc)
        # Dimensionless similarity factor in (0, 1] applied to BOTH axes.
        scale = min(1.0, math.sqrt(2.0 * taper_acc * usable) / maximum_v)
        return (
            max(floor_v, maximum_v * scale),
            max(floor_w, maximum_w * scale),
        )

    def track_path(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        avg_v: float,
        avg_w: float,
        dt: float,
        path: np.ndarray,
    ) -> MPPIPathTrackingResult:
        """Run one WP_TRACKING path-tracking tick.

        The command ceiling is the actuator maximum; the local reference is the lookahead
        waypoint every mode shares (see _select_reference).
        """
        return self._track_path(
            robot_x, robot_y, robot_yaw, avg_v, avg_w, dt, path,
            tag="wp_tracking",
            # The ceiling is the upper clip of every per-cell command domain and the
            # conservative speed is its lower clip, so the domain each knot optimizes in is
            # clip(travmap_value, min(ceiling, conservative), ceiling).  Choosing the ceiling
            # per task is how the required domain is produced, and it is a pure
            # function of the task -- no trav-map value from the caller enters it.
            # Here it is the ACTUATOR MAXIMA, so the domain becomes
            # max(conservative_speed, travmap value): terrain may only RAISE the command and
            # the single cell under the robot no longer caps the whole rollout.
            # The map COST is unaffected by this choice -- it keeps using the raw travmap
            # values, so a too-risky cell still costs 1/toorisky_v + 1/toorisky_w per knot
            # and rollouts are steered away from it even where the domain there is the
            # conservative speed.
            # ... except within goal_taper_hold_m + the braking distance of the path end,
            # where _goal_taper_caps walks BOTH ceilings down to the conservative speed so
            # the robot arrives slowly.  Inert outside that band, where it returns exactly
            # the actuator maxima this comment describes.
            cmd_limits=self._goal_taper_caps(robot_x, robot_y, path),
            reversed_geometry=False)
