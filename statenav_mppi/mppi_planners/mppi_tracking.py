"""Path-tracking layer: local reference, path cost terms and one tick.

The generic tracker.  It turns a global polyline into the forward interval the horizon
plans against and registers the terms that price a rollout against that interval:

    MPPICore -> MPPISolver -> TerrainMPPI -> PathTrackingMPPI -> MPPIPlanner

The MPPI update itself is NOT here.  It is MPPISolver.solve(), two layers down, because
nothing in it is path-flavoured: this layer's job is to hand it an MPPIProblem -- a pinned
map snapshot and an objective already bound to this tick's reference -- and to turn the
result into the node-facing command.

Everything here is task-agnostic: _track_path() is HANDED its tag, its command ceiling
and its rollout geometry, so the task entrypoint in
mppi_tasks.py is a thin wrapper rather than a copy of a tick -- and there is no mode
registry in between, because each mixin IS its mode.
"""

from __future__ import annotations

import math
import time
from functools import partial
from typing import Optional

import numpy as np
import torch

from statenav_mppi.mppi_core.mppi_validity import ValidityTermTable, screen
from statenav_mppi.mppi_core.mppi_costs import CostContext, CostTermTable, MPPIProblem
from statenav_mppi.mppi_utils.mppi_path_geometry import (
    advance_along_polyline,
    extend_path_end,
    project_point_to_polyline,
    project_to_reference,
    sanitize_path,
)
from statenav_mppi.mppi_planners.mppi_terrain import TerrainMPPI
from statenav_mppi.mppi_core.mppi_types import (
    MPPIInput,
    MPPIPathTrackingResult,
    MPPIResult,
    PathProjection,
)


# Slack on the Progress Lipschitz bound, dimensionless.  A rollout is measured as straight
# chords between its knots while the arc it covers is measured on the reference's own much
# finer polyline, so even a plan that follows a curve EXACTLY reads as travelling slightly
# less than it advances.  The gap is the chord-vs-arc error, (s/R)^2/24 for knot spacing s
# on radius R: 1.9% at the worst case this planner can command (v=1.35 m/s over mppi_dt=0.5
# on the R=1.0 m curve that is the tightest holdable at full speed), and under 0.5% for
# anything gentler.  10% therefore clears the discretization by 5x while still clipping the
# projection jumps this bound exists for, which run 1.5x to 12x over.  Raising it past ~1.3
# would start letting real corner-cuts through.
_PROGRESS_CLIP_TOLERANCE = 1.10


class _ProgressTerm:
    """The along-path progress reward, bound to ONE run.

    An object rather than a functools.partial:
    it keeps the snapshot the tuning trace prints, and CostTermTable.debug_lines() finds
    debug_line() by attribute lookup on the registered callable -- which a partial does not
    forward.  The snapshot cannot outlive its run, because the term IS the run: one is
    built per CostTermTable.bind(), and the two evaluations of a tick (the sample batch,
    then the settled batch-of-one) share it, so what debug_line() reports is the ISSUED
    plan's clip and not some rollout's.

    The arithmetic stays on the planner in _term_Progress; this only owns the snapshot.
    """

    def __init__(self, planner, reference: torch.Tensor) -> None:
        self._planner = planner
        # (M,2) local reference this run's costs are measured against.
        self._reference = reference
        # Last evaluation's pre/post-clip tensors, or None before the first call.
        self._debug: Optional[dict] = None

    def __call__(self, ctx: CostContext) -> torch.Tensor:
        record: dict = {}
        value = self._planner._term_Progress(
            ctx, reference=self._reference, record=record)
        self._debug = record
        return value

    def debug_line(self) -> Optional[str]:
        """One trace line, or None when the term has not been evaluated this run.

        Resolved here rather than inside __call__: the snapshot holds TENSORS, and reading
        them per rollout batch would force a device sync on the control path.

        `ratio` is credited arc per metre driven.  Following the reference exactly gives
        1.00 and never clips; a value the CLIP had to pull down means the plan was cutting
        across a bend in the path and being paid the bend's whole arc for the chord.
        """
        snapshot = self._debug
        if snapshot is None:
            return None
        advance = snapshot["advance"]
        travelled = snapshot["travelled"]
        clipped = snapshot["clipped"]
        raw_mean = float(torch.mean(advance).item())
        kept_mean = float(torch.mean(clipped).item())
        travelled_mean = float(torch.mean(travelled).item())
        # A knot counts as clipped only past a tolerance: the bound is an equality for a
        # plan that follows the path, so exact-equality counting would report float noise.
        clipped_knots = int(torch.sum(advance > clipped + 1.0e-4).item())
        total_knots = int(advance.reshape(-1).numel())
        driven = float(travelled.reshape(-1)[-1].item())
        # Undefined rather than infinite when the plan holds position: no distance driven
        # means no ratio, and 0/0 would print as nan without saying why.
        ratio = (f"{raw_mean / travelled_mean:.2f} -> {kept_mean / travelled_mean:.2f}"
                 if travelled_mean > 1.0e-3 else "n/a (plan holds position)")
        return (
            f"[mppi][dbg]   Progress: mean arc raw={raw_mean:.2f}m "
            f"clipped={kept_mean:.2f}m | ratio (arc per metre driven) {ratio} | "
            f"{clipped_knots}/{total_knots} knots clipped, plan drives {driven:.2f}m")


class _CorridorTerm:
    """The corridor funnel veto, bound to ONE run.

    An object rather than a partial for the same two reasons _ProgressTerm is one: it keeps
    the snapshot the tuning trace prints, and ValidityManager.debug_lines() finds
    debug_line() by attribute lookup on the registered callable.  The arithmetic stays on
    the planner in _validity_Corridor; this owns the snapshot and the per-tick abstention
    count.

    TWO snapshots, because the two evaluations of a tick answer different questions.  The
    SAMPLE batch says how much of the batch the funnel admitted -- a number that is only
    readable next to the abstention count -- and the settled batch-of-one says whether the
    plan actually being ISSUED stays inside it, which is the invariant this term exists to
    hold.  A single snapshot would report whichever ran last and silently mix the two.

    Everything is stored UNRESOLVED, as tensors: reading them here would force a device
    sync on the control path, so debug_line() resolves them and only when the trace is on.
    """

    def __init__(self, planner: "PathTrackingMPPI", reference: torch.Tensor) -> None:
        self._planner = planner
        # (M,2) local reference this run's cross-track is measured against.
        self._reference = reference
        # Last SAMPLE-batch and last BATCH-OF-ONE evaluation, or None before each first.
        self._batch: Optional[dict] = None
        self._issued: Optional[dict] = None
        # How many SAMPLE batches this run has judged and how many of them the funnel had
        # to abstain on.  Counted over sample batches alone: a batch-of-one "abstention" is
        # only the settled plan sitting outside the funnel, which the issued line below
        # reports in its own words, and folding the two together would make the number that
        # matters -- how often the veto went inert on a real batch -- unreadable.  The count
        # is accumulated as a 0-d TENSOR so it costs no device sync per batch.
        self._evaluations = 0
        self._abstentions: Optional[torch.Tensor] = None

    def __call__(self, ctx: CostContext) -> torch.Tensor:
        record: dict = {}
        value = self._planner._validity_Corridor(
            ctx, reference=self._reference, record=record)
        # K == 1 is the settled plan being priced (by the guard, or for the trace); anything
        # wider is a sample batch.
        if int(ctx.states.shape[0]) == 1:
            self._issued = record
        else:
            self._batch = record
            self._evaluations += 1
            abstained = record["abstained"].to(torch.long)
            self._abstentions = (
                abstained if self._abstentions is None
                else self._abstentions + abstained)
        return value

    def debug_line(self) -> Optional[str]:
        """One or two trace lines, or None before the term has been evaluated.

        The funnel is reported as the BOUND it applied, not as a pass/fail count alone: a
        run where the corridor abstains often is a run where corridor_half_width or
        corridor_decay is mis-sized for the manoeuvre the path is demanding, and that must
        not be silent -- an abstaining term is an ABSENT term, and the plan is then only as
        path-adherent as the cost weights make it.
        """
        snapshot = self._batch or self._issued
        if snapshot is None:
            return None
        config = self._planner.config
        bound = snapshot["bound"].reshape(-1).cpu().numpy()          # (T,) m
        abstentions = (0 if self._abstentions is None
                       else int(self._abstentions.reshape(-1)[0].item()))
        line = (
            f"[mppi][dbg]   Corridor: d0={float(snapshot['d0']):.2f}m "
            f"bound=[" + " ".join(f"{float(value):.2f}" for value in bound) + "]m "
            f"(w={float(config.corridor_half_width):.2f} "
            f"decay={float(config.corridor_decay):.2f})")
        if self._batch is not None:
            inside = int(self._batch["inside"].sum().item())
            total = int(self._batch["inside"].numel())
            line += (f" | batch {inside}/{total} inside, ABSTAINED on "
                     f"{abstentions}/{self._evaluations} sample batch(es)")
            if bool(self._batch["abstained"]):
                # Last sample batch had nothing inside: report the relaxed band it kept instead.
                kept = int(self._batch["admitted"].sum().item())
                line += (f" -> RELAXED: kept {kept}/{total} within "
                         f"+{float(config.corridor_abstain_slack):.2f}m of the least "
                         f"violation +{float(self._batch['least_excess']):.2f}m")
        if self._issued is not None:
            # (T,) cross-track of the plan being issued, against the same bound.
            issued = self._issued["cross_track"].reshape(-1).cpu().numpy()
            worst = int(np.argmax(issued - bound[:issued.shape[0]]))
            verdict = ("INSIDE" if float(issued[worst]) <= float(bound[worst]) + 1.0e-6
                       else "OUTSIDE -- the funnel was relaxed on this plan")
            line += (
                f"\n[mppi][dbg]     issued xt=["
                + " ".join(f"{float(value):.2f}" for value in issued)
                + f"]m worst knot {worst} ({float(issued[worst]):.2f}m vs "
                f"{float(bound[worst]):.2f}m) -> {verdict}")
        return line


class _ReverseHeadingTerm:
    """The reverse-heading funnel veto, bound to ONE run.  INERT outside reversed geometry.

    An object rather than a partial for _CorridorTerm's two reasons -- it keeps the
    snapshot the tuning trace prints, and ValidityManager.debug_lines() finds debug_line()
    by attribute lookup on the registered callable.  The arithmetic stays on the planner in
    _validity_ReverseHeading; this owns the snapshot, the abstention count and the gate.

    THE GATE IS BOUND ONCE, HERE, not tested per batch: the validity table is built once at
    planner construction and one planner serves every mode, so the term is registered
    unconditionally and decides at bind() whether this tick is one it judges.  A forward
    tick therefore pays one bool and returns an all-True mask -- it never projects, never
    reads _yaw_error and prints no trace line.  The name stays in the "valid:" signature of
    every mode's trace, which is the honest reading: the term is REGISTERED everywhere and
    ACTIVE only where reversed_geometry is.

    Two snapshots for _CorridorTerm's reason: the SAMPLE batch says how much of the batch
    the funnel admitted, the batch-of-one says whether the plan being ISSUED stays inside
    it, and a single snapshot would report whichever ran last.
    """

    def __init__(
        self, planner: "PathTrackingMPPI", data: MPPIInput, reference: torch.Tensor,
    ) -> None:
        self._planner = planner
        # (M,2) local reference this run's heading error is measured against.
        self._reference = reference
        # The gate: only a reversed rollout has a reverse-only linear domain to protect.
        self._active = bool(data.reversed_geometry)
        # Last SAMPLE-batch and last BATCH-OF-ONE evaluation, or None before each first.
        self._batch: Optional[dict] = None
        self._issued: Optional[dict] = None
        # Sample batches judged, and how many of them the funnel had to abstain on.  A
        # 0-d TENSOR, so accumulating it costs no device sync on the control path.
        self._evaluations = 0
        self._abstentions: Optional[torch.Tensor] = None

    def __call__(self, ctx: CostContext) -> torch.Tensor:
        if not self._active:
            # (K,) all-True: registered but not judging this mode.
            return torch.ones(
                ctx.states.shape[0], device=ctx.states.device, dtype=torch.bool)
        record: dict = {}
        value = self._planner._validity_ReverseHeading(
            ctx, reference=self._reference, record=record)
        # K == 1 is the settled plan being priced (by the guard, or for the trace);
        # anything wider is a sample batch.
        if int(ctx.states.shape[0]) == 1:
            self._issued = record
        else:
            self._batch = record
            self._evaluations += 1
            abstained = record["abstained"].to(torch.long)
            self._abstentions = (
                abstained if self._abstentions is None
                else self._abstentions + abstained)
        return value

    def debug_line(self) -> Optional[str]:
        """One or two trace lines; None in a forward mode or before the first evaluation.

        Reported as the BOUND it applied, not as a pass/fail count alone, for the reason
        _CorridorTerm's line is: a run that abstains often is a run whose limit or decay is
        mis-sized for the manoeuvre the retreat demands, and an abstaining term is an
        ABSENT term -- the heading is then only as constrained as weight_PathAligned makes
        it, which is the situation this veto exists because of.
        """
        if not self._active:
            return None
        snapshot = self._batch or self._issued
        if snapshot is None:
            return None
        config = self._planner.config
        bound = snapshot["bound"].reshape(-1).cpu().numpy()          # (T,) rad
        e0 = float(snapshot["e0"])
        abstentions = (0 if self._abstentions is None
                       else int(self._abstentions.reshape(-1)[0].item()))
        line = (
            f"[mppi][dbg]   ReverseHeading: e0={e0:.2f}rad ({math.degrees(e0):.0f}deg) "
            f"bound=[" + " ".join(f"{float(value):.2f}" for value in bound) + "]rad "
            f"(limit={float(config.recovery_heading_limit):.2f} "
            f"decay={float(config.recovery_heading_decay):.2f})")
        if self._batch is not None:
            aligned = int(self._batch["aligned"].sum().item())
            total = int(self._batch["aligned"].numel())
            line += (f" | batch {aligned}/{total} aligned, ABSTAINED on "
                     f"{abstentions}/{self._evaluations} sample batch(es)")
        if self._issued is not None:
            # (T,) heading error of the plan being issued, against the same bound.
            issued = self._issued["heading_error"].reshape(-1).cpu().numpy()
            worst = int(np.argmax(issued - bound[:issued.shape[0]]))
            verdict = ("INSIDE" if float(issued[worst]) <= float(bound[worst]) + 1.0e-6
                       else "OUTSIDE -- the funnel abstained on this plan")
            line += (
                f"\n[mppi][dbg]     issued |yaw_err|=["
                + " ".join(f"{float(value):.2f}" for value in issued)
                + f"]rad worst knot {worst} ({float(issued[worst]):.2f}rad vs "
                f"{float(bound[worst]):.2f}rad) -> {verdict}")
        return line


class PathTrackingMPPI(TerrainMPPI):
    """Terrain-aware MPPI that tracks a polyline; mode-agnostic by construction."""





####################################### PLAN INVALIDATION #######################################
# Path-aware warm-start invalidation.  The unconditional drop lives on MPPICore (reset_plan);
# what needs the polyline -- deciding whether the cached plan is still ABOUT the path in front
# of the robot -- belongs here, with the rest of the path geometry.

    def drop_plan_if_off_path(
        self, path: np.ndarray, max_end_distance_m: float,
    ) -> tuple[bool, Optional[float], str]:
        """Drop the warm-start plan when its rollout END no longer sits near ``path``.

        Called by the node when a NEW global path is accepted.  The cached plan was solved
        against the PREVIOUS polyline, and the cheapest test of whether it still means
        anything is where it ENDS: one horizon (horizon * mppi_dt seconds) ahead, which is
        where a reroute separates from the old route while ordinary cross-track error does
        not.  The plan is kept whole or dropped whole -- there is no partial credit for a
        plan whose far half belongs to a corridor that no longer exists.

        Inputs : ``path`` the (N, 2) world-frame global path that just arrived;
                 ``max_end_distance_m`` (m) the distance beyond which the plan is stale.
        Returns: ``(dropped, end_distance_m, reason)``.  ``end_distance_m`` is the measured
                 distance from the plan's last rollout point to ``path``, or None when no
                 measurement was possible; ``reason`` names the branch that ran, so the
                 caller can log WHY a plan survived or died rather than just that it did.

        The measured endpoint is ONE control period stale -- MPPIResult.predicted_path starts
        at the pose of the solve that produced it -- which is deliberate: this is a coarse
        relevance test, not a metric.  Never raises: the caller is a control loop with no
        try/except around its body.  Touches only the plan state, never the rate anchor.
        """
        if self._nominal is None:
            return False, None, "no cached plan"
        # (T+1, 3) [x, y, yaw] rollout of the plan the warm start holds, or None when the last
        # tick produced no valid solve.  Written by the SAME solve that wrote _nominal, so its
        # last row is this warm start's own endpoint and not some earlier plan's.
        predicted = None if self._last_result is None else self._last_result.predicted_path
        if predicted is None or predicted.ndim != 2 or predicted.shape[0] < 1:
            # Nothing to vouch for the plan against the new path.  Dropping is the conservative
            # branch: the alternative keeps a plan that nothing has checked, and the last tick
            # failed anyway.
            self.reset_plan()
            return True, None, "no rollout to measure"
        try:
            projection = project_point_to_polyline(
                path, np.asarray(predicted[-1, :2], dtype=np.float32))
        except (TypeError, ValueError):
            # Unusable new path: fewer than two distinct points, or non-finite coordinates.
            # _track_path rejects it on this tick too, so drop rather than keep a plan measured
            # against nothing.
            self.reset_plan()
            return True, None, "unusable path"
        distance = math.sqrt(max(0.0, float(projection.distance_sq)))
        if distance > float(max_end_distance_m):
            self.reset_plan()
            return True, distance, "off path"
        return False, distance, "on path"




####################################### LOCAL REFERENCE #######################################
# The forward interval of the global path the horizon is priced against.

    # (m) Shortest TOTAL reference length that can still define a direction.  Below it the
    # reference carries no forward bearing and every path cost degenerates, so a waypoint this
    # close to the robot -- the path ended under it -- is reported instead of planned against.
    # 1 mm is far below any real lookahead.
    _MIN_REFERENCE_SEGMENT_M: float = 1.0e-3
    # (m) Below this, two consecutive reference knots are the SAME point and the segment
    # between them has zero length, which makes project_to_reference divide by zero and every
    # path cost NaN.  Reached where the cut meets a vertex: the robot's foot can coincide with
    # the next vertex, and the waypoint with the previous one.  Matches sanitize_path's own
    # de-duplication threshold, so no vertex that survived it is dropped here.
    _DUPLICATE_KNOT_M: float = 1.0e-6

    def _select_reference(self, data: MPPIInput) -> np.ndarray:
        """Return the ``(2, 2)`` local reference: the robot, and the lookahead waypoint.

        ``advance_along_polyline`` decides HOW FAR the window reaches -- the robot's own foot
        is walked FORWARD ALONG THE PATH by the horizon's own span worth of travel, so the
        waypoint sits where the rollout ends -- and the reference is the PATH cut between the
        robot and it::

            reference[0]    = the robot's foot on the path      (m, world)
            reference[1:-1] = the path's own vertices in between
            reference[-1]   = the waypoint                      (m, world)

        with one exception, at the head: when the foot CLAMPS to the path's first vertex --
        the robot has not reached the path yet, which is the normal state while the global
        planner keeps snapping its start to a graph node ahead of the robot -- the measured
        position is prepended, so reference[0] is the robot itself and the gap between it
        and the path becomes a lead-in segment the objective can see.  See the branch below
        for what silently breaks without it.

        The vertices are what carry the path's shape into the objective.  Every path cost is a
        projection onto these segments, so a rollout that traces the path pays zero cross-track
        and one that cuts the inside of a bend pays for the deviation -- which a straight
        robot-to-waypoint chord would instead have rewarded.

        The budget is ARC, not a circle drawn around the robot: it is spent on the path's own
        segments, so a winding path returns a waypoint that is nearer in a straight line but
        the same length of TRAVEL away, and the window does not shrink into a bend.  Because
        the walk starts at the robot's own foot and only ever moves forward, the waypoint
        cannot land behind it -- which is why nothing here has to test that it is ahead.

        The reach speed is ``max(conservative_speed_v, cmd_v_limit)``: the conservative speed
        is the command domain available in EVERY cell, so flooring there stops a terrain- or
        taper-lowered ceiling from pulling the waypoint in exactly where the plan most needs
        to see ahead.  The lookahead time is the horizon's own span scaled by
        ``lookahead_time_scale``, which is how far PAST its last knot the plan is asked to
        aim; see the config field for why that is above 1.

        The polyline all of that is measured on is the path CONTINUED past its own last
        vertex (extend_path_end), unless the run asked for it not to be.  Without the
        continuation the window is clipped by whatever path remains, so on the last metres
        the reference collapses toward the final vertex: the cross-track term stops
        measuring cross-track and becomes a radial pull toward that point, progress
        saturates, and the coupled simplex -- which spends the whole command rectangle at
        every knot -- pours the unspendable forward budget into yaw.  The continuation is
        not a route: the guard below refuses to plan from a foot that has already reached
        the real path's end, so it can only ever price the approach, never extend it.  The
        robot's own foot is projected on the REAL path alone; the window's far end is the
        only query the continuation ever answers.

        ONE window serves every mode.  There is no per-task override and no fallback: a plan
        priced against a window nobody asked for is worse than a tick that holds and says why,
        so every unusable waypoint raises ValueError, which _prepare_problem turns into a held
        position carrying the reason.
        """
        # (m/s) reach speed of the lookahead; see the docstring for the floor.
        cmd_v = max(
            float(self.config.conservative_speed_v), float(data.cmd_v_limit))
        # (s) the horizon's own span, scaled: at scale 1.0 the waypoint sits exactly where
        # the rollout ends, and above it the waypoint stays ahead of the last knot.
        lookahead_time = (float(self.config.horizon) * float(self.config.mppi_dt)
                          * float(self.config.lookahead_time_scale))
        # ── the polyline this tick is measured on ──
        # (N, 2) the REAL path, sanitized exactly as every projection below sanitizes it,
        # so the arc length measured from it is directly comparable with theirs.
        real_points = sanitize_path(data.path)
        # (m) total arc of the real path.  The continuation appended below is a costing
        # device rather than a route, so this is where "the path ends" for the guard.
        real_path_arc = float(np.sum(
            np.linalg.norm(np.diff(real_points, axis=0), axis=1)))
        # (M, 2) what the waypoint query, both projections and the cut all read.
        # data.path itself is never modified.
        tracked_path = real_points
        if data.extend_reference_past_end:
            # ONE window arc budget: the foot below is projected onto the REAL path, so it
            # sits at most real_path_arc - _MIN_REFERENCE_SEGMENT_M along it, and the walk
            # never spends more than one budget of arc past the foot.  A single budget
            # therefore always contains the waypoint rather than clamping to its last
            # vertex.  A SECOND budget was needed only while the foot could itself land on
            # the continuation, which the real-path projection below no longer permits.
            tracked_path = extend_path_end(
                real_points, lookahead_time * cmd_v)
        # ── cut the path forward from the robot by the arc budget ──
        # Where the robot sits ON the polyline.  The walk below starts from this projection
        # and shares its prefix sum, so the two arc lengths are directly comparable.
        # Projected on the REAL path, never on the continuation: project_point_to_polyline
        # is an unconstrained nearest-segment search, so a path that bends back on itself --
        # or whose final tangent points back over the robot, which is what a route that
        # approaches its goal from the far side does -- can put a VIRTUAL tail vertex nearer
        # to the robot than any real segment.  The foot then reads as tens of metres of arc,
        # past the real end, and the guard below holds a robot that has in fact not started;
        # standing still, the next replan returns the same path and the same projection, so
        # the hold never lifts.  tracked_path[:len(real_points)] IS real_points, so this
        # projection's segment_index and prefix sum stay valid for the walk below.
        try:
            start = project_point_to_polyline(
                real_points, np.asarray(data.pose[:2], dtype=np.float32))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"could not project the robot onto the path: {exc}") from exc
        # The continuation prices the approach; it is not a route to follow.  Once the
        # robot's own foot has reached the real path's last vertex there is nothing left to
        # track, and planning along the virtual tail would drive it straight past the end.
        # The node's arrival guard normally holds the robot before this fires; this is the
        # backstop for an approach that passes the end wide enough to miss that guard's
        # radius, and for a run whose arrival guard is disabled.  Tested BEFORE the walk
        # below so a foot past the end is reported as what it is rather than as a window
        # that had nothing left to spend its budget on.  The tolerance is the same
        # millimetre the reference needs to carry a direction, which also absorbs the last
        # float32 bit of disagreement between this sum and the projection's own prefix sum.
        if (data.extend_reference_past_end
                and start.arc_length >= real_path_arc - self._MIN_REFERENCE_SEGMENT_M):
            raise ValueError(
                f"the robot's foot sits {start.arc_length:.2f} m along a "
                f"{real_path_arc:.2f} m path: the path ends here, and the virtual "
                f"continuation past its end is priced against, never followed")
        # The window's far end: lookahead_time * cmd_v metres of TRAVEL further along the
        # path, measured from the robot's OWN foot.  A robot that has not reached the path
        # yet spends the whole budget on the path itself and the lead-in prepended below is
        # arc on TOP of that, so the window runs longer than the budget -- deliberately
        # harmless here: Progress is absolute metres advanced and only CLIPS at the reference
        # end, so a generous window merely saturates later, while a short one is what
        # degenerates the objective (see extend_path_end).
        end = advance_along_polyline(tracked_path, start, lookahead_time * cmd_v)

        # (N, 2) this tick's polyline -- the real path, continued past its end unless the
        # run opted out.  Sanitized above, so its vertices are finite and distinct.
        points = tracked_path
        # (M, 2) the robot's foot on the path, the path's OWN vertices in between, then the
        # waypoint.  The vertices carry the path's shape into the cost, which is the whole
        # point of cutting instead of drawing a chord: a rollout that traces the path pays
        # zero cross-track, and cutting the inside of a bend is penalized again.
        reference = np.vstack((
            start.point[None, :],
            points[start.segment_index + 1:end.segment_index + 1],
            end.point[None, :],
        )).astype(np.float32)
        # A foot that clamped to the polyline's FIRST vertex means the robot has not reached
        # the path yet: the projection ran off the back of segment zero, so start.point is
        # that vertex rather than a point beside the robot.  The global planner snaps its
        # start onto a graph node ahead of the robot, so the leading ~0.75 m of every replan
        # sits in FRONT of it, and that gap is a dead zone for the whole objective:
        # project_to_reference clamps the arc at 0 across it, so Progress is identically
        # zero (nothing rewards closing the gap and the coupled simplex spends the unclaimed
        # linear budget as yaw), and the tangent PathAligned measures against is the path's
        # own first segment rather than the bearing the robot must travel to reach it.
        # Prepending the measured position turns the gap into a real lead-in segment with a
        # real bearing and a real arc.  Costs nothing when the robot is already on the path:
        # the foot is then interior (fraction > 0) and this does not fire, and on the exact
        # vertex the duplicate-knot filter below drops the repeated point.
        if start.segment_index == 0 and start.fraction <= 0.0:
            reference = np.vstack((
                np.asarray(data.pose[:2], dtype=np.float32)[None, :],
                reference,
            )).astype(np.float32)
        # Drop a knot that repeats the previous one -- the cut lands on a vertex whenever the
        # robot or the waypoint projects exactly onto it -- since a zero-length segment makes
        # every path cost NaN.
        keep = np.ones(reference.shape[0], dtype=bool)
        keep[1:] = (np.linalg.norm(np.diff(reference, axis=0), axis=1)
                    > self._DUPLICATE_KNOT_M)
        reference = reference[keep]
        # (m) the arc the horizon is being asked to cover, along the path rather than across it.
        span = (0.0 if reference.shape[0] < 2 else
                float(np.sum(np.linalg.norm(np.diff(reference, axis=0), axis=1))))
        if reference.shape[0] < 2 or span < self._MIN_REFERENCE_SEGMENT_M:
            raise ValueError(
                f"the path between the robot and the lookahead waypoint is {span:.4f} m over "
                f"{reference.shape[0]} point(s), under the "
                f"{self._MIN_REFERENCE_SEGMENT_M:.3f} m a reference needs to carry a "
                f"direction (the path ends here)")
        return reference




####################################### PATH PROJECTION #######################################
# Batched nearest-segment projection; the geometric primitive every path cost is
# built on.  The implementation is
# mppi_path_geometry.project_to_reference -- it reads no planner state, so it belongs
# with the rest of the polyline geometry.  Kept bound here under its original name
# because the cost terms reach for it through the planner.

    _project_to_reference = staticmethod(project_to_reference)




####################################### PATH COSTS #######################################
# Per-term costs over the local reference.  Each returns (K,) UNWEIGHTED; the manager
# applies the weight registered below, so a term can be retuned without touching the
# others.  All four share ONE projection of the batch onto this run's reference, memoized
# on the evaluation context: computing it per term would run the batched nearest-segment
# search four times over the same (K,T,2) positions.

    def _build_cost_table(self) -> CostTermTable:
        table = super()._build_cost_table()
        # `reference` is this tick's (M,2) local-reference tensor; every path term is bound
        # to it and to nothing else.  Each entry carries its own maker rather than sharing
        # one partial() closure, because Progress needs to be an OBJECT: it publishes a
        # debug line, and CostTermTable.debug_lines() looks debug_line() up as an attribute
        # of the registered callable, which functools.partial does not forward.  Written
        # out per entry so the registration ORDER -- and therefore the order the trace
        # prints the objective in -- is unchanged.
        for name, weight, make, reward in (
            ("PathTracking",          self.config.weight_PathTracking,
             lambda data, reference: partial(self._term_PathTracking, reference=reference), False),
            ("PathAligned",           self.config.weight_PathAligned,
             lambda data, reference: partial(self._term_PathAligned, reference=reference), False),
            ("Progress",              self.config.weight_Progress,
             lambda data, reference: _ProgressTerm(self, reference), True),
            ("PathTracking_Terminal", self.config.weight_PathTracking_Terminal,
             lambda data, reference: partial(self._term_PathTracking_Terminal, reference=reference), False),
        ):
            table.add(name, weight, reward=reward, make=make)
        return table

    @staticmethod
    def _require_reference(reference: Optional[torch.Tensor]) -> torch.Tensor:
        """Raise unless this run was bound to a local reference.

        Every path term needs one, and a run without one is a construction error at the
        entrypoint, not something to discover as a shape mismatch inside the projection.
        """
        if reference is None:
            raise RuntimeError(
                "path cost terms need a local reference, but this run was bound with "
                "reference_points=None; the per-run entrypoint must pass the (M, 2) "
                "reference tensor to CostTermTable.bind()")
        return reference

    def _projection(
        self, ctx: CostContext, reference: Optional[torch.Tensor],
    ) -> PathProjection:
        """Memoized ``(K,T)`` projection of this batch onto this run's reference.

        One CostManager is bound to exactly one reference, and one CostContext to exactly
        one batch, so a plain key is unambiguous.
        """
        if "projection" not in ctx.cache:
            ctx.cache["projection"] = project_to_reference(
                ctx.states[..., :2], self._require_reference(reference),
                validate_finite=False)
        return ctx.cache["projection"]

    def _yaw_error(
        self, ctx: CostContext, reference: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Memoized ``(K,T)`` wrapped heading error against the reference tangent.

        The reversed modes track the SAME polyline backwards, so their reference yaw is
        flipped by pi.  This is the only place that flip is applied.
        """
        if "yaw_error" not in ctx.cache:
            reference_yaw = self._projection(ctx, reference).tangent_yaw
            if ctx.data.reversed_geometry:
                reference_yaw = self._wrap_torch(reference_yaw + math.pi)
            ctx.cache["yaw_error"] = self._wrap_torch(
                ctx.states[..., 2] - reference_yaw)
        return ctx.cache["yaw_error"]

    def _term_PathTracking(
        self, ctx: CostContext, *, reference: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return ``(K,)`` UNWEIGHTED mean squared cross-track distance (m^2).

        The time mean preserves squared-metre units before ``weight_PathTracking``.
        Inputs are trusted finite projections; the VALIDITY manager rejects non-finite
        rollouts.
        """
        return torch.mean(self._projection(ctx, reference).distance_sq, dim=-1)

    def _term_PathAligned(
        self, ctx: CostContext, *, reference: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return ``(K,)`` UNWEIGHTED mean squared heading error (rad^2).

        Wrapped yaw errors are squared and averaged over time, yielding squared-radian
        units before ``weight_PathAligned``.
        """
        return torch.mean(self._yaw_error(ctx, reference) ** 2, dim=-1)

    def _term_Progress(
        self, ctx: CostContext, *, reference: Optional[torch.Tensor],
        record: Optional[dict] = None,
    ) -> torch.Tensor:
        """Return ``(K,)`` UNWEIGHTED mean arc advanced past the measured pose (m).

        Averaging progress over the horizon rewards reaching the same terminal arc
        earlier, while projection clipping prevents progress beyond the reference
        endpoint.  Registered with ``reward=True``, so the manager SUBTRACTS it and the
        stored term stays negative exactly as the trace has always printed it.

        The advance is CLIPPED to the distance the rollout actually drove, because
        project_to_reference() takes the globally nearest segment of the whole polyline --
        no arc window, no monotonicity -- and nearest-segment projection onto a non-convex
        polyline is not 1-Lipschitz.  Where the reference doubles back around an obstacle,
        a rollout that leaves one leg and lands on the other is credited the entire bend it
        skipped: a hairpin whose legs sit 0.5 m apart pays ~6.6 m of "progress" for 0.5 m
        of travel, which at weight_Progress outbids the cross-track penalty for cutting the
        corner by more than an order of magnitude.  Since the loop in a global path is
        precisely how it routes AROUND something, the unclipped term pays the robot to
        drive through the obstacle the path was avoiding.

        ``record``, when given, receives the pre- and post-clip tensors for the tuning
        trace.  They are stored UNRESOLVED: reading them here would force a device sync on
        the control path, so _ProgressTerm.debug_line() resolves them instead.
        """
        initial_position = torch.as_tensor(
            np.asarray(ctx.data.pose[:2], dtype=np.float32),
            device=self.device, dtype=self.dtype)
        initial_arc = project_to_reference(
            initial_position, self._require_reference(reference),
            validate_finite=False).arc_length
        # (...,T) raw arc every knot claims past the pose's own projection.
        advance = self._projection(ctx, reference).arc_length - initial_arc
        # (...,T,2) knot positions, and the same row shifted one knot back so that knot
        # zero's predecessor is the measured pose -- the same pose-then-knots polyline
        # solve() rebuilds for the visualiser.
        positions = ctx.states[..., :2]
        previous = torch.cat(
            (initial_position.expand(*positions.shape[:-2], 1, 2),
             positions[..., :-1, :]), dim=-2)
        # (...,T) PATH LENGTH driven by each knot, not straight-line displacement from the
        # pose: a rollout that follows a bend faithfully must stay unclipped, and its chord
        # is shorter than its arc, so a displacement bound would penalise exactly the
        # behaviour this term is meant to reward.
        travelled = torch.cumsum(
            torch.sqrt(torch.sum((positions - previous) ** 2, dim=-1)), dim=-1)
        # (...,T) the bound itself: the Lipschitz limit the projection should have had,
        # widened by the chord-vs-arc slack a knot-sampled rollout cannot avoid (see
        # _PROGRESS_CLIP_TOLERANCE).  Symmetric, because the same projection jump runs
        # BACKWARDS across a bend too, and there it manufactures a penalty rather than a
        # reward -- clipping only the reward side would leave that half of the artifact in
        # place and bias the term against ever approaching a doubled-back reference.
        bound = travelled * _PROGRESS_CLIP_TOLERANCE
        clipped = torch.minimum(torch.maximum(advance, -bound), bound)
        if record is not None:
            record["advance"] = advance
            record["travelled"] = travelled
            record["clipped"] = clipped
        return torch.mean(clipped, dim=-1)

    def _term_PathTracking_Terminal(
        self, ctx: CostContext, *, reference: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return ``(K,)`` UNWEIGHTED endpoint squared cross-track distance (m^2).

        Not normalized before ``weight_PathTracking_Terminal``.  Endpoint projection
        clipping is performed by project_to_reference; a cache hit on _projection.
        """
        return self._projection(ctx, reference).distance_sq[..., -1]




####################################### PATH VALIDITY #######################################
# The corridor funnel: the one path rule that is a VETO rather than a price.  It is here
# and not in the cost table because a WEIGHT cannot express it -- see _validity_Corridor.

    def _build_validity_table(self) -> ValidityTermTable:
        table = super()._build_validity_table()
        # Registered with make=, not fn=: the veto needs this run's reference, and the
        # registry already binds one (ValidityTermTable.bind passes reference_points, which
        # _prepare_problem supplies).  No plumbing beyond this line.
        table.add("Corridor", make=lambda data, reference: _CorridorTerm(self, reference))
        # The heading twin of the corridor, and registered the same way -- except that it
        # also needs `data`, because the mode it judges is carried there.  Registered for
        # every mode and ACTIVE only where data.reversed_geometry is; see
        # _ReverseHeadingTerm for why the gate is bound here rather than tested per batch.
        table.add("ReverseHeading",
                  make=lambda data, reference: _ReverseHeadingTerm(self, data, reference))
        return table

    def _validity_Corridor(
        self, ctx: CostContext, *, reference: Optional[torch.Tensor],
        record: Optional[dict] = None,
    ) -> torch.Tensor:
        """``(K,)`` True where the rollout stays inside the cross-track funnel.

        The bound at rollout knot ``k`` is::

            w_k = max(corridor_half_width, d0 * corridor_decay**k)

        with ``d0`` the robot's OWN cross-track this tick, projected onto the same local
        reference the path costs are measured against.  So:

          * ``d0 <= corridor_half_width`` -- a plain tube of that half-width.  The permitted
            one-cell dodge around a too-risky patch lives inside it, and the "leave the path
            and route around" excursion does not.
          * ``d0 >  corridor_half_width`` -- the bound starts AT d0 and walks toward the
            tube at ``corridor_decay`` per knot, so the admissible rollouts are the ones
            that do not move further out (at the shipped decay of 1.0) and the ones
            converging at that rate (below it).  See MPPIConfig.corridor_decay for why the
            shipped value is 1.0 and what a smaller one costs.

        WHY A VETO AND NOT A WEIGHT.  MPPI selects on the SPREAD of a term across samples;
        a term that adds the same number to every sample cancels exactly in the softmax.
        PathTracking's level divided by its own spread is 1.6 on the runs that reach the
        goal and 3.7-5.5 on the ones that fail -- its authority DECREASES as the error
        grows -- and raising weight_PathTracking scales level and spread together, so the
        argmax does not move.  A corridor has to be a constraint to be a corridor.

        WHY A FUNNEL AND NOT A TUBE.  There are two ways the robot can start outside a
        tube, and _select_reference only handles one of them: when the path starts ahead of
        the robot, the lead-in prepend makes reference[0] the robot itself and d0 is 0 by
        construction.  When the robot is laterally off the MIDDLE of the path the foot
        projects interior, no lead-in is prepended, and a tube would veto every rollout
        including the ones driving back.  The funnel is feasible at every tick by
        construction -- the first knot's bound is d0 itself -- never admits moving further
        out, and is one rule in both regimes.  It never vetoes where the robot IS, only
        where it is going.

        ABSTENTION.  When no rollout satisfies the bound the term does not let screen()
        produce "all rollouts invalid", which would hold the position -- and a corridor that
        stops the robot is worse than no corridor.  It RELAXES instead of vanishing: only the
        rollouts whose worst-knot excess over the bound is within corridor_abstain_slack of
        the least-violating one stay admissible, so the batch is never empty and the veto
        still points back at the path.  The abstention is recorded and printed, never
        silent: a run that abstains often is a
        run whose corridor_decay is demanding lateral convergence the manoeuvre cannot
        supply (an in-place pivot converges at zero), and that is a tuning fault to see
        rather than a behaviour to inherit.

        The projection is the memoized one the four path costs already share, so the funnel
        costs no extra geometry; ``d0`` is computed exactly as _term_Progress computes its
        ``initial_arc``.  ``record``, when given, receives the snapshot _CorridorTerm's
        trace line resolves -- tensors, unresolved, for the reason given there.
        """
        reference_points = self._require_reference(reference)
        # (K,T) cross-track of every knot of every rollout.  clamp(min=0) only guards the
        # float32 negative-zero the squared distance can carry into sqrt.
        cross_track = torch.sqrt(
            torch.clamp(self._projection(ctx, reference).distance_sq, min=0.0))
        # (2,) the measured pose, projected onto the SAME reference: the funnel's mouth.
        initial_position = torch.as_tensor(
            np.asarray(ctx.data.pose[:2], dtype=np.float32),
            device=self.device, dtype=self.dtype)
        d0 = float(torch.sqrt(torch.clamp(project_to_reference(
            initial_position, reference_points,
            validate_finite=False).distance_sq, min=0.0)))
        if not math.isfinite(d0):
            # Unreachable -- MPPIProblem validates the pose finite and the reference is
            # built from a sanitized path -- but a non-finite mouth would make every bound
            # non-finite and veto the whole batch, so degrade to the plain tube instead.
            d0 = 0.0
        half_width = float(self.config.corridor_half_width)
        horizon = int(cross_track.shape[-1])
        # (T,) the funnel itself.  Knot 0 is bounded by d0 exactly, which is what makes the
        # set non-empty by construction: a rollout that holds its cross-track satisfies the
        # first knot, and only the knots AFTER it are asked to converge.
        knots = torch.arange(
            horizon, device=cross_track.device, dtype=cross_track.dtype)
        bound = torch.clamp(
            d0 * float(self.config.corridor_decay) ** knots, min=half_width)
        # (K,T) then (K,): a rollout is admissible only if EVERY knot of it is.  A
        # non-finite cross-track compares False here and is rejected, which agrees with
        # FiniteState rejecting the same rollout.
        inside = cross_track <= bound[None, :]
        admissible = torch.all(inside, dim=-1)
        # 0-d bool: did anything survive?  Kept as a TENSOR and folded in with where() so
        # the abstention costs no device sync on the control path.
        any_valid = torch.any(admissible)
        # (K,) worst-knot excess over the bound (m): <= 0 inside, +inf for a non-finite rollout,
        # so FiniteState's rejects can never be the ones the relaxed corridor keeps.
        excess = torch.amax(cross_track - bound[None, :], dim=-1)
        excess = torch.where(
            torch.isfinite(excess), excess, torch.full_like(excess, float("inf")))
        # Graceful abstention: keep the least-violating band; all-True only if none is finite.
        least_excess = torch.min(excess)
        relaxed = torch.where(
            torch.isfinite(least_excess),
            excess <= least_excess + float(self.config.corridor_abstain_slack),
            torch.ones_like(admissible))
        valid = torch.where(any_valid, admissible, relaxed)
        if record is not None:
            record["d0"] = d0
            record["bound"] = bound
            # The PRE-abstention mask, deliberately: reporting the returned one would make
            # an abstained tick read as "every rollout was inside the funnel", which is the
            # opposite of what happened.
            record["inside"] = admissible
            record["cross_track"] = cross_track
            record["abstained"] = ~any_valid
            # What the relaxed corridor kept, and the least violation it was measured from.
            record["admitted"] = valid
            record["least_excess"] = least_excess
        return valid

    def _validity_ReverseHeading(
        self, ctx: CostContext, *, reference: Optional[torch.Tensor],
        record: Optional[dict] = None,
    ) -> torch.Tensor:
        """``(K,)`` True where a REVERSED rollout keeps facing back down its own reference.

        The bound at rollout knot ``k`` is::

            h_k = max(recovery_heading_limit, e0 * recovery_heading_decay**k)

        with ``e0`` the robot's OWN heading error this tick, measured at the pose against
        the same local reference the path costs are measured against, and every error
        taken against the REVERSED tangent -- _yaw_error's pi flip, which is what makes
        zero error mean "backing straight down the retreat" rather than "driving down it".

          * ``e0 <= recovery_heading_limit`` -- a plain cap of that half-width.  The
            steering a retreat needs to follow a curving route lives inside it; a turn that
            puts the reference BEHIND the robot does not.
          * ``e0 >  recovery_heading_limit`` -- the bound starts AT e0 and walks toward the
            cap at ``recovery_heading_decay`` per knot, so the admissible rollouts are the
            ones that do not turn further off (at the shipped decay of 1.0) and the ones
            unwinding at that rate (below it).

        WHY A VETO AND NOT A WEIGHT.  weight_PathAligned is the price of this same error
        and it demonstrably loses the bid: on the failed retreat this term was written for,
        the tick that spun the robot 180 deg traded PathAligned +10 -> +253 for Travmap
        +361 -> +113 and came out 12 cheaper overall, because a rollout that stops
        translating stops paying the terrain.  Raising weight_PathAligned to win that trade
        would have to outbid a term whose magnitude is set by the map, not by the geometry,
        and it would distort every well-behaved tick to do it.  A heading limit has to be a
        constraint to be a limit.

        WHY IT MATTERS ONLY IN REVERSED GEOMETRY.  A forward mode that ends up facing the
        wrong way can drive out of it: its linear domain contains the command that fixes
        the error.  A reversed one cannot -- CommandConstraint.build opens the negative half
        of the linear domain and closes the positive half to braking only -- so past a
        quarter turn EVERY admissible command drives the robot further from the retreat
        target, and the mode is trapped in a state the cost can see and no command can
        leave.  That is the failure this veto removes, and it removes it by making the
        state unreachable rather than by pricing the exit.

        WHY A FUNNEL AND NOT A CAP.  The trapped state is exactly the state a flat cap
        cannot handle: at e0 ~ pi every rollout violates it, the batch empties, and the
        solver holds position -- one stall traded for another.  Knot 0's bound is e0
        itself, so the set is non-empty by construction, the term never vetoes where the
        robot IS FACING (only where it is turning to), and the same rule reads as
        "never turn further off" from inside the cap and outside it alike.

        ABSTENTION.  When no rollout satisfies the bound the term returns an all-True mask
        rather than letting screen() empty the batch, for _validity_Corridor's reason and
        with the same visibility: the abstention is counted and printed, never silent.

        JOINT FEASIBILITY.  This is the third hard veto on a tick that already carries
        MapUsable and Corridor, and each abstains only on its OWN empty set, so an empty
        INTERSECTION still reaches screen() and holds the position with a reason.  That
        exposure is not new (MapUsable and Corridor have always had it) and is bounded
        here: a reversed reference is rebuilt from the current pose every tick, so
        Corridor's d0 is 0 and its bound is the plain tube, which the rollouts this funnel
        admits -- the ones backing down that very line -- sit inside.

        The heading error is the memoized ``(K,T)`` tensor _term_PathAligned already
        prices, so the funnel costs no extra geometry beyond the one pose projection it
        needs for ``e0``.  ``record``, when given, receives the snapshot
        _ReverseHeadingTerm's trace line resolves -- tensors, unresolved, for the reason
        given there.
        """
        reference_points = self._require_reference(reference)
        # (K,T) |wrapped heading error| against the reversed reference tangent.  Cache hit
        # on _yaw_error: PathAligned has already built it for this batch.
        heading_error = torch.abs(self._yaw_error(ctx, reference))
        # (2,) the measured pose, projected onto the SAME reference: the funnel's mouth.
        initial_position = torch.as_tensor(
            np.asarray(ctx.data.pose[:2], dtype=np.float32),
            device=self.device, dtype=self.dtype)
        initial_tangent = project_to_reference(
            initial_position, reference_points,
            validate_finite=False).tangent_yaw
        # The pi flip again, on the measured pose this time, so e0 is the same quantity the
        # (K,T) errors above are: zero when the robot is backing straight down the route.
        e0 = float(torch.abs(self._wrap_torch(
            float(ctx.data.pose[2]) - (initial_tangent + math.pi))))
        if not math.isfinite(e0):
            # Unreachable -- MPPIProblem validates the pose finite and the reference is
            # built from a sanitized path -- but a non-finite mouth would make every bound
            # non-finite and veto the whole batch, so degrade to the plain cap instead.
            e0 = 0.0
        limit = float(self.config.recovery_heading_limit)
        horizon = int(heading_error.shape[-1])
        # (T,) the funnel itself.  Knot 0 is bounded by e0 exactly, which is what makes the
        # set non-empty by construction: a rollout that holds its heading satisfies the
        # first knot, and only the knots AFTER it are asked to unwind.
        knots = torch.arange(
            horizon, device=heading_error.device, dtype=heading_error.dtype)
        bound = torch.clamp(
            e0 * float(self.config.recovery_heading_decay) ** knots, min=limit)
        # (K,T) then (K,): a rollout is admissible only if EVERY knot of it is.  A
        # non-finite heading compares False here and is rejected, which agrees with
        # FiniteState rejecting the same rollout.
        aligned = heading_error <= bound[None, :]
        admissible = torch.all(aligned, dim=-1)
        # 0-d bool: did anything survive?  Kept as a TENSOR and folded in with where() so
        # the abstention costs no device sync on the control path.
        any_valid = torch.any(admissible)
        valid = torch.where(any_valid, admissible, torch.ones_like(admissible))
        if record is not None:
            record["e0"] = e0
            record["bound"] = bound
            # The PRE-abstention mask, deliberately: reporting the returned one would make
            # an abstained tick read as "every rollout was inside the funnel", which is the
            # opposite of what happened.
            record["aligned"] = admissible
            record["heading_error"] = heading_error
            record["abstained"] = ~any_valid
        return valid




####################################### COST COMPOSITION #######################################

    def _costs(
        self,
        states: torch.Tensor,
        controls: torch.Tensor,
        data: MPPIInput,
        reference: np.ndarray,
        terms: Optional[dict] = None,
    ) -> torch.Tensor:
        """Bind the registered objective to ``data``/``reference`` and score one batch.

        A convenience wrapper, NOT the control path: solve() binds once per tick and
        evaluates the resulting CostManager two or three times, so going through here
        would re-allocate the term closures on every batch.  Kept because scoring an
        arbitrary batch against a tick's inputs is exactly what a parity check or an
        offline analysis wants, and because it keeps one stable entry to the objective.

        ``terms``, when given, is filled with the ``(K,)`` weighted components under the
        names the debug trace prints, before the admissibility screen replaces a rejected
        rollout's total with ``inf``.

        Binds the validity table too, and screens through it, because the contract this
        wrapper has always had is "the number solve() would score this batch at" -- and a
        cost sum that skipped the vetoes would no longer be that number.
        """
        reference_points = torch.as_tensor(
            reference, device=self.device, dtype=self.dtype)
        costs = self._cost_table.bind(data, reference_points)
        validity = self._validity_table.bind(data, reference_points)
        # One context for both, as MPPIProblem.score() does: the terrain cost and the
        # terrain veto share a memoized footprint lookup on it.
        ctx = CostContext(states=states, controls=controls, data=data)
        screened, _ = screen(
            costs.evaluate(ctx, terms=terms), validity.evaluate(ctx))
        return screened




####################################### PATH-TRACKING TICK #######################################
# The shared body of every task entrypoint: validate, pin the terrain, build this tick's
# objective, solve, and turn the result into the node-facing command plus diagnostics.
# Never raises into the control loop.

    def _prepare_problem(
        self, data: MPPIInput, start: float,
    ) -> tuple[Optional[MPPIProblem], Optional[MPPIResult]]:
        """Pin the terrain snapshot and bind this tick's objective to it.

        The two preconditions MPPIProblem states and solve() does not check are owned
        here: ONE map version serving every rollout of the tick, and a fully bound
        CostManager and ValidityManager.  Both used to live inside solve(); they belong
        to the per-run entrypoint, which is where the tick begins.

        Returns ``(problem, None)`` when both hold, or ``(None, invalid_result)`` when the
        tick must be rejected.  Never raises: the caller is a control loop.
        """
        try:
            # The travgraph read window follows this tick's pose (data.pose is (x, y, yaw)).
            self._ensure_map_cache(data.pose)
        except torch.cuda.OutOfMemoryError as exc:
            # OutOfMemoryError subclasses RuntimeError, so it must be caught
            # before the generic handler below.  The map cache upload is the
            # largest single CUDA allocation in a tick and is where OOM under
            # external GPU pressure actually surfaces.  This branch KEEPS the warm
            # start (_fallback_to_cpu carries _nominal over); the one below drops it.
            return None, self._invalid_result(start, self._fallback_to_cpu(exc))
        except (RuntimeError, ValueError) as exc:
            self.reset()
            return None, self._invalid_result(start, str(exc))
        try:
            reference = self._select_reference(data)
            reference_points = torch.as_tensor(
                reference, device=self.device, dtype=self.dtype)
            problem = MPPIProblem(
                data=data,
                costs=self._cost_table.bind(data, reference_points),
                validity=self._validity_table.bind(data, reference_points),
                reference=reference)
        except ValueError as exc:
            return None, self._invalid_result(start, str(exc))
        return problem, None

    def _track_path(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        avg_v: float,
        avg_w: float,
        dt: float,
        path: np.ndarray,
        *,
        tag: str,
        cmd_limits: tuple[float, float],
        reversed_geometry: bool,
    ) -> MPPIPathTrackingResult:
        """Shared body of the path-tracking entrypoints.

        Everything a task differs by is an ARGUMENT, not a lookup: ``tag`` is the log
        prefix, ``cmd_limits`` the (v, w) command ceiling this task plans in, and
        ``reversed_geometry`` its rollout geometry.  Each mixin supplies its own,
        next to the reasoning for it.  The local reference is NOT among them: every mode
        plans against the same lookahead waypoint (see _select_reference).

        Every rejection path returns valid=False with a tagged reason rather than raising,
        so one bad tick holds position instead of killing the control loop.
        """
        cmd_v_limit, cmd_w_limit = cmd_limits
        try:
            clean_path = sanitize_path(path)
        except (TypeError, ValueError) as exc:
            self.reset()
            return MPPIPathTrackingResult(
                0.0, 0.0, None, None,
                (f"[{tag}][MPPI] {exc} — holding position",),
                valid=False,
            )
        scalars = (robot_x, robot_y, robot_yaw, avg_v, avg_w, dt,
                   cmd_v_limit, cmd_w_limit)
        if not all(math.isfinite(float(value)) for value in scalars):
            self.reset()
            return MPPIPathTrackingResult(
                0.0, 0.0, None, None,
                (f"[{tag}][MPPI] state, timing, and limits must be finite "
                 "— holding position",),
                valid=False,
            )
        if dt <= 0.0 or cmd_v_limit <= 0.0 or cmd_w_limit <= 0.0:
            self.reset()
            return MPPIPathTrackingResult(
                0.0, 0.0, None, None,
                (f"[{tag}][MPPI] expected positive dt/limits, got "
                 f"dt={dt}, v={cmd_v_limit}, w={cmd_w_limit} — holding position",),
                valid=False,
            )

        pose = np.array([robot_x, robot_y, robot_yaw], dtype=np.float32)
        velocity = np.array([avg_v, avg_w], dtype=np.float32)
        self.set_cmd_limits(cmd_v_limit, cmd_w_limit)
        data = MPPIInput(
            tag=tag,
            reversed_geometry=reversed_geometry,
            pose=pose,
            velocity=velocity,
            control_dt=float(dt),
            path=clean_path,
            cmd_v_limit=float(cmd_v_limit),
            cmd_w_limit=float(cmd_w_limit),
        )
        # The terrain snapshot and this tick's objective are pinned before the solve, not
        # inside it, so every rollout of the tick reads one map version and one reference.
        start = time.perf_counter()
        problem, rejection = self._prepare_problem(data, start)
        result = rejection if problem is None else self.solve(problem)
        self._last_result = result
        warnings = result.warnings
        if not result.valid:
            warnings += (
                f"[{tag}][MPPI] {result.reason} — holding position",
            )
        target = result.reference_endpoint
        target_text = (
            "reference=unavailable" if target is None
            else f"reference_end=({target[0]:.2f}, {target[1]:.2f})")
        diagnostic = (
            f"[{tag}][MPPI] {target_text} "
            f"solve={result.solve_time_s * 1.0e3:.1f} ms "
            f"cost={result.min_cost:.3f}"
        )
        # The CUDA fallback only warns on the tick that failed; latch it into the
        # periodic diagnostic so a degraded planner stays visible to the operator.
        if self._cuda_fallback_active:
            diagnostic += f" [CPU fallback, {self.num_samples} samples]"
        return MPPIPathTrackingResult(
            float(result.v), float(result.w), target, diagnostic, warnings,
            valid=result.valid)
