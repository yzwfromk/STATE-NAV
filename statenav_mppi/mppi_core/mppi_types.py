"""Configuration and data contracts shared by every MPPI planner layer.

Split out of mppi_planner.py so the frozen dataclasses that cross the planner/node
boundary -- MPPIConfig, MPPIInput and the three result types -- can be imported
without pulling in the solver itself.  Nothing here holds planner state or touches
the traversability map; MPPIConfig is the single place tuning is validated.
"""

from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass, fields
from typing import Any, Mapping, Optional

import numpy as np
import torch




####################################### CONFIG DISCOVERY #######################################
# Locating planning_config.yaml and reading its keys.  The YAML is the single source
# of every default below; _MAPPING is the field-name -> YAML-key translation.

def find_default_config_path() -> Path | None:
    """Find planning_config.yaml: $STATENAV_PLANNING_CONFIG, else the packaged statenav_global config.

    Uses the same resolver as every statenav_global module and node (statenav_global.core.config),
    so an override reaches the MPPI too. Returns None when the resolved file does not exist.
    """
    from statenav_global.core.config import planning_config_path
    candidate = planning_config_path()
    return candidate if candidate.is_file() else None


_MAPPING: dict[str, str] = {
    "device": "mppi_device",
    "horizon": "mppi_horizon",
    "num_samples": "mppi_num_samples",
    "cpu_num_samples": "mppi_cpu_num_samples",
    "temperature": "mppi_lambda",
    "adaptive_temperature": "mppi_adaptive_temperature",
    "adaptive_temperature_target_ess": "mppi_adaptive_temperature_target_ess",
    "adaptive_temperature_max_scale": "mppi_adaptive_temperature_max_scale",
    "noise_v": "mppi_noise_v",
    "noise_w": "mppi_noise_w",
    "noise_smoothing_taps": "mppi_noise_smoothing_taps",
    "iterations": "mppi_iterations",
    "seed": "seed",
    "mppi_dt": "mppi_dt",
    "lag_time": "mppi_lag_time",
    "lookahead_time_scale": "mppi_lookahead_time_scale",
    "lin_acc": "mppi_lin_acc_ub",
    "ang_acc": "mppi_ang_acc_ub",
    "recovery_reverse_scale": "mppi_recovery_reverse_scale",
    "weight_PathTracking": "mppi_weight_PathTracking",
    "weight_PathAligned": "mppi_weight_PathAligned",
    "weight_Progress": "mppi_weight_Progress",
    "weight_PathTracking_Terminal": "mppi_weight_PathTracking_Terminal",
    "weight_CtrlEffort": "mppi_weight_CtrlEffort",
    "weight_Accel": "mppi_weight_Accel",
    "weight_Travmap": "mppi_weight_Travmap",
    "weight_TravmapDistance": "mppi_weight_TravmapDistance",
    "travmap_footprint_harmonic_power": "mppi_travmap_footprint_harmonic_power",
    "too_risky_v": "toorisky_v",
    "too_risky_w": "toorisky_w",
    "conservative_speed_v": "conservative_speed_v",
    "conservative_speed_w": "conservative_speed_w",
    "maximum_cmd_v": "maximum_cmd_v",
    "maximum_cmd_w": "maximum_cmd_w",
    "goal_taper_acc_frac": "mppi_goal_taper_acc_frac",
    "goal_taper_hold_m": "mppi_goal_taper_hold_m",
    "map_snapshot_retries": "mppi_map_snapshot_retries",
    "travgraph_resolution": "trav_estimation_resoultion",
    "travgraph_sigmoid_k": "interp_sigmoid_k",
    "travgraph_rasterize": "mppi_travgraph_rasterize",
    "travgraph_window_radius": "mppi_travgraph_window_radius",
    "footprint_radius": "mppi_footprint_radius",
    "footprint_samples": "mppi_footprint_samples",
    "guard_margin": "mppi_guard_margin",
    "corridor_half_width": "mppi_corridor_half_width",
    "corridor_decay": "mppi_corridor_decay",
    "corridor_abstain_slack": "mppi_corridor_abstain_slack",
    "recovery_heading_limit": "mppi_recovery_heading_limit",
    "recovery_heading_decay": "mppi_recovery_heading_decay",
    "debug_print": "debug_print",
}


def _load_yaml_defaults() -> dict[str, Any]:
    """Read default values dynamically from planning_config.yaml."""
    path = find_default_config_path()
    if path is None or not path.is_file():
        return {}
    try:
        try:
            from omegaconf import OmegaConf
            loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        except ImportError:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
    except Exception:
        return {}

    if not isinstance(loaded, Mapping):
        return {}

    defaults = {}
    for field_name, yaml_key in _MAPPING.items():
        if yaml_key in loaded:
            defaults[field_name] = loaded[yaml_key]
    return defaults


_DEFAULTS: dict[str, Any] = _load_yaml_defaults()


def _cfg_value(source: Any, name: str, default: Any) -> Any:
    """Read a value from OmegaConf, a mapping, or a plain object."""
    if source is None:
        return default
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)




####################################### PLANNER CONFIGURATION #######################################

@dataclass(frozen=True)
class MPPIConfig:
    """MPPI sampling, constraints, costs, and debug settings.
    
    All default values are loaded dynamically from planning_config.yaml.
    """

    device: str = _DEFAULTS["device"]
    horizon: int = _DEFAULTS["horizon"]
    num_samples: int = _DEFAULTS["num_samples"]
    cpu_num_samples: int = _DEFAULTS["cpu_num_samples"]
    temperature: float = _DEFAULTS["temperature"]
    noise_v: float = _DEFAULTS["noise_v"]
    noise_w: float = _DEFAULTS["noise_w"]
    # Width in KNOTS of the symmetric kernel that smooths the sampled command noise ALONG the
    # horizon (see MPPICore._build_noise_filter).  1 keeps the historical i.i.d. per-knot
    # noise; 3 and 5 correlate neighbouring knots so ONE draw proposes a sustained manoeuvre
    # instead of jitter that largely cancels over the rollout.  That is what the horizon TAIL
    # needs: a command's influence on the cost falls off with the square of the rollout time
    # remaining after it, so an isolated late-knot perturbation is invisible to the softmax
    # while the same perturbation held across several knots carries the leverage of the
    # earliest one.  Must be ODD -- an even kernel has no centre knot and would shift the
    # noise half a knot forward in time, biasing knot zero, the command actually issued.
    # The per-knot marginal std stays exactly noise_v / noise_w at every width, so this key
    # changes the SHAPE of the exploration and never its magnitude.
    # .get so an older YAML still loads.
    noise_smoothing_taps: int = _DEFAULTS.get("noise_smoothing_taps", 1)
    # MPPI update passes per control tick.  A tick used to be exactly one sample-score-settle
    # pass, so the plan could move by at most one noise-scale step per control period and the
    # tail -- see noise_smoothing_taps above for why it is the slow part -- converged over
    # many ticks instead of within one.  Each extra pass re-samples around the plan the
    # previous one settled on and costs one more full batch, so solve time scales linearly
    # with this: budget num_samples against ctrl_period accordingly.
    # .get so an older YAML still loads.
    iterations: int = _DEFAULTS.get("iterations", 1)
    seed: int = _DEFAULTS["seed"]
    mppi_dt: float = _DEFAULTS["mppi_dt"]
    # Dimensionless multiplier on the horizon's own span (horizon * mppi_dt) when the local
    # reference asks the map reader for its lookahead waypoint -- see
    # PathTrackingMPPI._select_reference.  At 1.0 the waypoint sits exactly where the last
    # rollout knot can reach, which leaves the plan no reference beyond its own endpoint: the
    # terminal term then prices the endpoint against the very point the rollout is trying to
    # arrive at, and progress saturates there.  Above 1.0 the waypoint stays ahead of the
    # rollout, so the plan is always steering toward path it has not covered yet.  It scales
    # the TIME, not the speed, so the reach it produces still tracks the command ceiling.
    # .get so an older YAML still loads.
    lookahead_time_scale: float = _DEFAULTS.get("lookahead_time_scale", 1.5)
    lag_time: float = _DEFAULTS["lag_time"]

    # One symmetric linear rate bound (mppi_lin_acc_ub) for BOTH directions: braking and
    # accelerating are held to the same |dv/dt|, so there is no separate deceleration key.
    lin_acc: float = _DEFAULTS["lin_acc"]
    ang_acc: float = _DEFAULTS["ang_acc"]
    recovery_reverse_scale: float = _DEFAULTS["recovery_reverse_scale"]

    weight_PathTracking: float = _DEFAULTS["weight_PathTracking"]
    weight_PathAligned: float = _DEFAULTS["weight_PathAligned"]
    weight_Progress: float = _DEFAULTS["weight_Progress"]
    weight_PathTracking_Terminal: float = _DEFAULTS["weight_PathTracking_Terminal"]
    weight_CtrlEffort: float = _DEFAULTS["weight_CtrlEffort"]
    weight_Accel: float = _DEFAULTS["weight_Accel"]
    weight_Travmap: float = _DEFAULTS["weight_Travmap"]
    # Per-metre twin of weight_Travmap (TerrainMPPI._term_TravmapDistance).  0 leaves the term
    # unregistered, i.e. exactly the historical per-knot-only terrain cost.
    weight_TravmapDistance: float = _DEFAULTS.get("weight_TravmapDistance", 0.0)
    # Power-mean ORDER p of the footprint reduction the terrain COST is built on
    # (TerrainMPPI._power_mean_positive).  -1 is the harmonic mean, the historical behaviour;
    # 0 is the geometric mean.  Only the terrain COST's 1 + footprint_samples ring reads it:
    # the command rectangle samples the body CENTRE alone, and a power mean over ONE point is
    # that point at every order, so the command domain and the MapUsable veto are untouched
    # by this key BY CONSTRUCTION.
    #
    # What it buys: both ENDS of the reduction are pinned at every p, because a power mean of
    # identical values is that value.  An all-free footprint and an all-blocked one therefore
    # cost exactly what they cost at p = -1, while a PARTIAL overlap costs less.  That is the
    # only way to relax what a footprint GRAZING an obstacle edge pays without also lowering
    # the all-blocked barrier: weight_Travmap and the too-risky stamp both scale the two
    # together, so neither can separate them.  At weight_Travmap 20, too_risky 0.01 and 9
    # footprint points, n blocked points cost (last column against weight_PathTracking 600):
    #
    #       p     n=1     n=2     n=5     n=9      n=1 as an equivalent cross-track
    #   -1.00     471     912    2235    4000      0.86 m   <- harmonic, historical
    #   -0.75     273     602    1876    4000      0.64 m
    #   -0.50     140     334    1410    4000      0.43 m
    #   -0.25      76     164     884    4000      0.28 m
    #    0.00      51      88     451    4000      0.19 m   <- geometric
    #
    # Must be <= 0.  A POSITIVE order leans the reduction toward the FREE points instead of
    # the blocked ones, which prices a mostly-blocked footprint as nearly free -- at p = +0.5
    # five of nine points blocked costs 121, a 0.45 m detour -- and would drive the robot
    # through obstacles; __post_init__ rejects it rather than warning.  Floored at -4 for
    # float32 headroom, because the reduction raises 1/too_risky_v to the power -p.
    # .get so an older YAML still loads, at the historical harmonic mean.
    travmap_footprint_harmonic_power: float = _DEFAULTS.get("travmap_footprint_harmonic_power", -1.0)

    # These values identify a current-cell too-risky stamp so the caller's raw
    # map limit can be raised to the configured command-domain floor. They do
    # not reject rollouts: stamped cells remain planned over at a large soft
    # map cost.
    too_risky_v: float = _DEFAULTS["too_risky_v"]
    too_risky_w: float = _DEFAULTS["too_risky_w"]
    # Conservative speed: the per-axis command domain that is available in EVERY
    # cell, however bad the map says it is. Terrain may only raise it. It keeps a
    # too-risky cell drivable without changing its 0.01/0.01 stamp or the terrain
    # cost derived from that stamp.
    conservative_speed_v: float = _DEFAULTS.get("conservative_speed_v", 0.5)
    conservative_speed_w: float = _DEFAULTS.get("conservative_speed_w", 0.3)

    # ── Per-mode command ceilings ─────────────────────────────────────────────────────────
    # The ceiling every per-cell domain is clipped to from above; the conservative speed above
    # is the floor it is clipped to from below.  Owned here rather than by the node because the
    # ceiling is a pure function of the mode (each task mixin passes its own cmd_limits= to
    # _track_path), not of anything the node measures: WP_TRACKING gets the actuator maxima
    # so terrain may only RAISE the command.
    maximum_cmd_v: float = _DEFAULTS.get("maximum_cmd_v", 1.0)
    maximum_cmd_w: float = _DEFAULTS.get("maximum_cmd_w", 1.0)
    # ── Goal approach taper (WP_TRACKING only) ────────────────────────────────────────
    # Near the end of the path the mode ceiling above is walked DOWN to the conservative
    # speed, so the robot arrives slowly instead of at the actuator maximum.  It has to be
    # the CEILING and not a cost: under the coupled simplex equality law the projection
    # spends whatever rectangle it is given, so a penalty cannot slow the robot -- only a
    # smaller rectangle can (see CommandConstraint.project).
    #
    # The taper stops AT the conservative speed and never below it, which is what keeps it
    # free of edge cases: _command_limits floors every rectangle at
    # min(ceiling, conservative_speed), so a taper that stayed above that floor never
    # fights it, MPPIProblem's strictly-positive command-limit check is never approached,
    # and _select_reference's speed-derived window cannot collapse.
    #
    # acc_frac is the share of lin_acc the taper may demand.  Following
    # v(d) = sqrt(2 * a * d) costs exactly `a` of deceleration, so a fraction BELOW 1
    # guarantees the rate box can always follow the taper and the rate box and the simplex
    # can never be driven disjoint by it (which would emit the rate-infeasible warning and
    # hand the tick to the safety-first fallback).  0.5 leaves a factor-of-two margin.
    goal_taper_acc_frac: float = _DEFAULTS.get("goal_taper_acc_frac", 0.5)
    # Remaining distance (m) at which the taper has fully reached the conservative speed.
    # Should be >= the node's own path-end stop radius (main.py PATH_END_STOP = 1.5 x path
    # spacing), so the robot is already at the conservative speed when that hard stop takes
    # over; a larger value merely holds the conservative speed for longer, which is safe.
    goal_taper_hold_m: float = _DEFAULTS.get("goal_taper_hold_m", 0.35)
    map_snapshot_retries: int = _DEFAULTS["map_snapshot_retries"]
    # TerrainMPPI reads traversability only from the world model's shared lattice graph, which
    # it sigmoid-interpolates at runtime (mppi_travgraph.py); the dense TraversabilityMap is
    # never read.  The three keys below configure that.
    # (m) Node pitch of the travgraph -- the writer's own estimation-lattice resolution (shared
    # key trav_estimation_resoultion).  A graph whose nodes are not on this lattice is rejected.
    travgraph_resolution: float = _DEFAULTS.get("travgraph_resolution", 0.5)
    # Sharpness k of the writer's S-curve between lattice nodes (shared key interp_sigmoid_k).
    # Must equal the world model's value for the interpolation to reproduce its dense map.
    travgraph_sigmoid_k: float = _DEFAULTS.get("travgraph_sigmoid_k", 10.0)
    # True: evaluate the travgraph interpolation once per graph version over the whole fine grid
    # on the device, so each rollout lookup is a single gather (the dense path's cost) for the
    # (rows, cols, layers, 2) grid in device memory.  False: evaluate it per query (7 extra kernel
    # launches per lookup, no grid) -- only worth it for a map too large to hold as a grid.
    travgraph_rasterize: bool = _DEFAULTS.get("travgraph_rasterize", True)
    # (m) Half side of the square around the robot inside which the travgraph is read, through
    # the graph's bucket index (O(window) per refresh, not O(graph)).  Lattice points outside it
    # and holes (lattice points without an active node) read NaN and are hard-vetoed.  inf reads
    # the whole graph through a full snapshot, the pre-window path, which rejects holes.
    travgraph_window_radius: float = _DEFAULTS.get("travgraph_window_radius", 10.0)

    # Radius in metres of the ring the map cost samples around each predicted
    # (x, y); it must cover the whole body extent, not just a margin.
    footprint_radius: float = _DEFAULTS.get("footprint_radius", 0.40)
    # Evenly spaced circumference samples per state; the stencil keeps the
    # state centre in addition to these, so the lookup uses 1 + N points.
    footprint_samples: int = _DEFAULTS.get("footprint_samples", 8)

    # Per-tick MPPI tuning trace on/off.  Deliberately the SAME cfg.debug_print key main.py
    # already reads for its own per-tick trace, so one flag switches the whole node's routine
    # output instead of adding an mppi-private second one.  It gates the extra batch-of-one
    # cost evaluation in solve() as well as the print, so it costs nothing while off.
    debug_print: bool = _DEFAULTS.get("debug_print", False)

    # ESS-based temperature control.  With this on, `temperature` above stops being the
    # value the softmax uses and becomes the FLOOR of a per-pass search: see
    # MPPICore.adaptive_temperature for why a temperature fixed in config can only be right
    # at one cost scale.  Off restores the fixed-temperature behaviour exactly.
    adaptive_temperature: bool = _DEFAULTS.get("adaptive_temperature", True)
    # Target effective sample size, as a FRACTION of the finite samples in the batch. Too
    # small and the update stays a near-argmin -- one rollout, redrawn every tick; too large
    # and the weights approach uniform, which drives the update toward zero and pins the
    # plan on its warm start.
    adaptive_temperature_target_ess: float = _DEFAULTS.get(
        "adaptive_temperature_target_ess", 0.10)
    # Ceiling on that search, as a multiple of `temperature`. Bounds how far a batch the
    # cost cannot separate is allowed to push the weights toward uniform.
    adaptive_temperature_max_scale: float = _DEFAULTS.get(
        "adaptive_temperature_max_scale", 1000.0)

    # ── Best-sample guard (MPPISolver._adopt_best_sample) ─────────────────────────────
    # Relative slack, dimensionless, on "is the settled weighted mean actually worse than
    # the cheapest rollout the same batch drew".  The MPPI update is a softmax-weighted
    # MEAN of the candidates, and on a bimodal batch -- turn left of the hazard, turn right
    # of it -- the mean lands BETWEEN the modes, where the cost is higher than either.  The
    # guard adopts the cheapest drawn sample instead whenever
    #     settled_cost > best_sample_cost + guard_margin * |best_sample_cost|
    # so it is the margin, not the guard, that stops the plan chattering between the mean
    # and a sample on float noise.  0 would swap plans on a rounding step; large values
    # disable the guard by never firing.  Adoption is counted in the tick trace.
    guard_margin: float = _DEFAULTS.get("guard_margin", 0.10)

    # ── Corridor funnel veto (PathTrackingMPPI._validity_Corridor) ────────────────────
    # (m) Half-width of the tube the rollout's cross-track must stay inside while the robot
    # is already ON the path.  Sized from the geometry it has to permit, not from the cost
    # scale: shifting the mppi_footprint_radius ring clear of ONE bad map cell
    # (trav_estimation_resoultion) needs ~0.5 m of lateral room, so this is the smallest
    # half-width that still allows a genuine one-cell dodge around a too-risky patch while
    # forbidding the "abandon the path and route around it" excursions the terrain cost
    # would otherwise buy.
    corridor_half_width: float = _DEFAULTS.get("corridor_half_width", 0.60)
    # Per-knot shrink factor of that bound while the robot starts OUTSIDE the tube, in
    # (0, 1].  The bound at rollout knot k is max(corridor_half_width, d0 * decay**k) for
    # d0 the robot's own current cross-track, so the admissible set is exactly the rollouts
    # that converge back and the term never vetoes where the robot IS, only where it is
    # going.  1.0 is the plain non-increasing funnel ("never move further out"); below 1.0
    # it additionally DEMANDS lateral convergence at every knot.
    #
    # 1.0 by measurement, not by caution.  A robot sitting beside the path has to TURN
    # before it can converge, and it cannot do both in the same knot: at 1.35 m/s over a
    # 0.5 s knot, closing the 15 % of d0 that 0.85 asks for at knot 1 needs a heading ~50
    # deg off the path before the turn has happened.  Measured on the flat-map bench at
    # d0 = 2.5 m and d0 = 1.2 m, decay 0.85 admitted 0 of 256 samples and abstained on
    # EVERY batch -- an abstaining term is an absent term -- while 0.95 admitted 202/256
    # and 1.00 admitted 232/256, both vetoing the diverging samples.  So 0.85 buys nothing
    # except a silent corridor; lower this only with the trace's abstention count in view.
    corridor_decay: float = _DEFAULTS.get("corridor_decay", 1.0)
    # (m) Graceful abstention: when no rollout fits the funnel, keep only those whose worst-knot
    # excess over the bound is within this slack of the least-violating one (0 keeps the ties).
    corridor_abstain_slack: float = _DEFAULTS.get("corridor_abstain_slack", 0.10)

    # ── Reverse-heading funnel veto (PathTrackingMPPI._validity_ReverseHeading) ───────
    # (rad) The heading half-width the rollout's yaw error against the REVERSED reference
    # tangent must stay inside.  Applies to reversed_geometry ticks only, and is inert
    # everywhere else.
    #
    # pi/2 is the sign of the thing, not a tuning choice: the reverse-only linear domain
    # (see CommandConstraint.build) makes "backwards" the only direction the mode can
    # drive, so a heading more than a quarter turn off the reversed tangent points the one
    # admissible direction AWAY from the retreat target.  At exactly pi/2 the reverse
    # velocity's component along the retreat is zero; past it, it is negative and every
    # admissible command makes the retreat worse.  Raising this toward pi disables the
    # veto (every wrapped error is <= pi); lowering it demands a tighter reverse.
    recovery_heading_limit: float = _DEFAULTS.get(
        "recovery_heading_limit", 0.5 * math.pi)
    # Per-knot shrink factor of that bound while the robot starts OUTSIDE the limit, in
    # (0, 1].  The bound at rollout knot k is max(recovery_heading_limit, e0 * decay**k)
    # for e0 the robot's own current heading error, so -- exactly as corridor_decay does
    # for cross-track -- the term never vetoes where the robot IS FACING, only where it is
    # turning to.  This feasibility is not decorative: a robot that has already spun past
    # the limit is the state the veto exists to prevent, and a flat cap would empty the
    # batch on every tick there and hold the position instead of unwinding the spin.
    # 1.0 is the plain non-increasing funnel ("never turn further off"); below 1.0 it
    # additionally DEMANDS angular convergence at every knot, which a rollout that must
    # first brake out of a turn cannot always supply -- watch the trace's abstention count.
    recovery_heading_decay: float = _DEFAULTS.get("recovery_heading_decay", 1.0)

    def __post_init__(self) -> None:
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be one of: auto, cpu, cuda")
        for name in (
            "horizon", "num_samples", "cpu_num_samples", "map_snapshot_retries",
            "footprint_samples", "iterations", "noise_smoothing_taps",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        # Symmetric-kernel requirement, checked here rather than at the draw: an even width
        # has no centre knot, so the filtered noise would be shifted half a knot in time and
        # knot zero would be biased by knots it should not see.  Capped at the horizon
        # because a kernel wider than the plan mixes every knot into every other one, which
        # is not "correlated exploration" any more but a single scalar per sample.
        taps = int(self.noise_smoothing_taps)
        if taps % 2 == 0 or taps > int(self.horizon):
            raise ValueError(
                f"noise_smoothing_taps must be odd and <= horizon ({self.horizon}), got "
                f"{self.noise_smoothing_taps!r}: an even width shifts the sampled noise half "
                f"a knot forward in time and biases knot zero, and a width past the horizon "
                f"collapses the whole plan onto one random scalar per sample")
        for name in (
            "temperature", "noise_v", "noise_w", "mppi_dt",
            "lookahead_time_scale",
            "lin_acc", "ang_acc",
            "conservative_speed_v", "conservative_speed_w",
            "footprint_radius",
            "travgraph_resolution", "travgraph_sigmoid_k",
            # Strictly positive, not merely non-negative: these two are the per-channel
            # floor the map cost divides by (see _inverse_limit), so a zero stamp would
            # make every terrain cost +inf and collapse the softmax onto one rollout.
            "too_risky_v", "too_risky_w",
        ):
            if not math.isfinite(float(getattr(self, name))) or float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        # The ESS target is a fraction of the batch, so it has to land in (0, 1]. A target
        # of exactly 1.0 asks for uniform weights and therefore a zero update, which is a
        # legal thing to configure but never a useful one -- hence the upper bound is
        # inclusive and the warning lives in the field comment, not in a rejection.
        if not (0.0 < float(self.adaptive_temperature_target_ess) <= 1.0):
            raise ValueError(
                f"adaptive_temperature_target_ess must be in (0, 1], got "
                f"{self.adaptive_temperature_target_ess!r}: it is the share of the sample "
                f"batch the weighted update should average over")
        # Below 1.0 the ceiling would sit under the floor and the search would invert.
        if (not math.isfinite(float(self.adaptive_temperature_max_scale))
                or float(self.adaptive_temperature_max_scale) < 1.0):
            raise ValueError(
                f"adaptive_temperature_max_scale must be finite and >= 1.0, got "
                f"{self.adaptive_temperature_max_scale!r}: it multiplies the configured "
                f"temperature ({self.temperature!r}) to bound the search from above, so a "
                f"value below 1.0 would put that ceiling under its own floor")
        if not math.isfinite(float(self.lag_time)) or self.lag_time < 0.0:
            raise ValueError("lag_time must be finite and non-negative")
        if not 0.0 < self.recovery_reverse_scale <= 1.0:
            raise ValueError("recovery_reverse_scale must be in (0, 1]")
        if not 0.0 < float(self.goal_taper_acc_frac) < 1.0:
            raise ValueError(
                f"goal_taper_acc_frac must be in (0, 1), got "
                f"{self.goal_taper_acc_frac!r}: it is the share of lin_acc the goal taper "
                f"is allowed to demand, and at 1.0 the taper would ask for the whole rate "
                f"bound, leaving the projection no margin to follow it")
        if (not math.isfinite(float(self.goal_taper_hold_m))
                or float(self.goal_taper_hold_m) < 0.0):
            raise ValueError(
                f"goal_taper_hold_m must be finite and non-negative, got "
                f"{self.goal_taper_hold_m!r}")
        for name in (
            "weight_PathTracking", "weight_PathAligned", "weight_Progress",
            "weight_PathTracking_Terminal", "weight_CtrlEffort", "weight_Accel",
            "weight_Travmap", "weight_TravmapDistance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        # Footprint-reduction order.  Only p <= 0 leans the reduction toward the WORST
        # footprint point, which is what a safety reduction has to do; a positive order leans
        # it toward the free points and would price a mostly-blocked footprint as nearly free.
        # Rejected rather than warned about, because the failure is silent and the sign of
        # this key is exactly the thing that is easy to get wrong.  The lower bound keeps
        # (1 / too_risky_v) ** -p inside float32.
        _footprint_order = float(self.travmap_footprint_harmonic_power)
        if not math.isfinite(_footprint_order) or not (-4.0 <= _footprint_order <= 0.0):
            raise ValueError(
                f"travmap_footprint_harmonic_power must be finite and in [-4.0, 0.0], got "
                f"{self.travmap_footprint_harmonic_power!r}: -1 is the harmonic mean and 0 the "
                f"geometric mean, and a POSITIVE order leans the footprint reduction toward "
                f"the free points, pricing a mostly-blocked footprint as nearly free")
        # Actuator ceilings.
        for name in ("maximum_cmd_v", "maximum_cmd_w"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        # The conservative speed is the FLOOR of every per-cell domain and the ceiling is its
        # upper clip, so a ceiling below the floor would inverse the domain in every cell.
        if float(self.maximum_cmd_v) < float(self.conservative_speed_v):
            raise ValueError(
                f"maximum_cmd_v ({self.maximum_cmd_v:.3f}) must be >= conservative_speed_v "
                f"({self.conservative_speed_v:.3f}): the conservative speed is the per-cell "
                f"domain FLOOR and the maximum is its ceiling")
        if float(self.maximum_cmd_w) < float(self.conservative_speed_w):
            raise ValueError(
                f"maximum_cmd_w ({self.maximum_cmd_w:.3f}) must be >= conservative_speed_w "
                f"({self.conservative_speed_w:.3f})")
        # Non-negative, not strictly positive: 0 is the legitimate "adopt whenever the
        # cheapest sample is cheaper at all" setting.  Negative would adopt a sample that is
        # WORSE than the settled plan, which inverts the guard.
        if (not math.isfinite(float(self.guard_margin))
                or float(self.guard_margin) < 0.0):
            raise ValueError(
                f"guard_margin must be finite and non-negative, got "
                f"{self.guard_margin!r}: it is the relative slack on 'the settled mean is "
                f"worse than the cheapest drawn sample', and a negative value would adopt "
                f"samples that are worse than the plan the update settled on")
        # Strictly positive: a zero or negative tube admits nothing at all once the robot
        # is on the path, and the corridor term would veto every rollout on every tick.
        if (not math.isfinite(float(self.corridor_half_width))
                or float(self.corridor_half_width) <= 0.0):
            raise ValueError(
                f"corridor_half_width must be finite and positive, got "
                f"{self.corridor_half_width!r}")
        # (0, 1]: at or below 0 the funnel collapses to the tube in one knot, and above 1 it
        # would WIDEN with the horizon, which is the opposite of a funnel.
        if not (math.isfinite(float(self.corridor_decay))
                and 0.0 < float(self.corridor_decay) <= 1.0):
            raise ValueError(
                f"corridor_decay must be in (0, 1], got {self.corridor_decay!r}: it is the "
                f"per-knot shrink factor of the cross-track bound while the robot starts "
                f"outside the corridor, so a value above 1 would let a rollout leave the "
                f"path faster the further ahead it plans")
        # >= 0: a negative slack would keep nothing when the funnel is infeasible and empty the batch.
        if (not math.isfinite(float(self.corridor_abstain_slack))
                or float(self.corridor_abstain_slack) < 0.0):
            raise ValueError(
                f"corridor_abstain_slack must be finite and >= 0 m, got "
                f"{self.corridor_abstain_slack!r}: it is how far past the least-violating "
                f"rollout the relaxed corridor still admits, so a negative value admits none")
        # (0, pi) exclusive at BOTH ends: at or below 0 the funnel admits nothing once the
        # heading is settled, and at or above pi it admits everything, because a wrapped
        # error is always <= pi -- a silently disabled veto rather than a loose one.
        if not (math.isfinite(float(self.recovery_heading_limit))
                and 0.0 < float(self.recovery_heading_limit) < math.pi):
            raise ValueError(
                f"recovery_heading_limit must be in (0, pi) rad, got "
                f"{self.recovery_heading_limit!r}: it is the heading half-width a reversed "
                f"rollout must stay inside, and pi or more is inert because every wrapped "
                f"yaw error already satisfies it")
        # (0, 1], for corridor_decay's reason: above 1 the bound would WIDEN with the
        # horizon, so a rollout could turn further off the reference the further ahead it
        # plans -- the opposite of a funnel.
        if not (math.isfinite(float(self.recovery_heading_decay))
                and 0.0 < float(self.recovery_heading_decay) <= 1.0):
            raise ValueError(
                f"recovery_heading_decay must be in (0, 1], got "
                f"{self.recovery_heading_decay!r}: it is the per-knot shrink factor of the "
                f"heading bound while the robot starts outside the limit, so a value above "
                f"1 would let a reversed rollout turn further off the further ahead it plans")
        # The travgraph window must hold every footprint point a rollout can reach: its edge reads
        # NaN, so a window smaller than the reach would veto rollouts there like a wall.
        rollout_reach = (int(self.horizon) * float(self.mppi_dt) * float(self.maximum_cmd_v)
                         + float(self.footprint_radius))
        if (math.isnan(float(self.travgraph_window_radius))
                or float(self.travgraph_window_radius) < rollout_reach):
            raise ValueError(
                f"travgraph_window_radius ({self.travgraph_window_radius!r} m) must be >= the "
                f"rollout reach horizon * mppi_dt * maximum_cmd_v + footprint_radius = "
                f"{rollout_reach:.2f} m (inf reads the whole graph): the travgraph is read only "
                f"inside this window and everything outside it is hard-vetoed")

    @classmethod
    def from_file(cls, config_path: str | Path) -> "MPPIConfig":
        """Load configuration directly from a YAML file path."""
        path = Path(config_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        try:
            from omegaconf import OmegaConf
            loaded = OmegaConf.load(path)
        except ImportError:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)

        return cls.from_source(loaded)

    @classmethod
    def from_source(cls, source: Any = None) -> "MPPIConfig":
        """Build from planning_config.yaml, dict, file path, or object."""
        if source is None:
            config_path = find_default_config_path()
            if config_path is not None:
                return cls.from_file(config_path)
        elif isinstance(source, (str, Path)):
            return cls.from_file(source)

        yaml_defaults = _load_yaml_defaults()
        kwargs = {}
        for field_info in fields(cls):
            name = field_info.name
            yaml_key = _MAPPING.get(name, name)
            
            # Read from provided source
            val = _cfg_value(source, yaml_key, None)
            if val is None and yaml_key != name:
                val = _cfg_value(source, name, None)
            
            # Fallback dynamically to planning_config.yaml value
            if val is None:
                val = yaml_defaults.get(name, getattr(field_info, "default", None))
            
            kwargs[name] = val
        return cls(**kwargs)




####################################### SOLVER INPUT #######################################
# One tick's worth of everything solve() reads.  Frozen, so a caller can never
# mutate it underneath a rollout.

@dataclass(frozen=True)
class MPPIInput:
    # Log/HUD prefix for this run ("wp_tracking", ...).  NOTHING dispatches
    # on it: every behavioural difference a mode used to carry is now either a field below
    # or an argument the entrypoint passes, so this is a cosmetic string and an unknown
    # value cannot produce a wrong command.
    tag: str
    # Rollout GEOMETRY of this run: the map-lookup heading, the reference yaw and the
    # linear command domain are all flipped when it is True (see _constraint_Terrain,
    # _term_Travmap, _yaw_error and _project_and_rollout).  Everything else drives forward.
    #
    # The warm-start plan is invalidated when a transition crosses THIS boundary, NOT
    # whenever the tag changes.
    reversed_geometry: bool
    pose: np.ndarray
    velocity: np.ndarray
    control_dt: float
    path: np.ndarray
    cmd_v_limit: float
    cmd_w_limit: float
    # Continue `path` past its last vertex along the final tangent when the local reference
    # is selected (mppi_path_geometry.extend_path_end, used by
    # PathTrackingMPPI._select_reference).  True for every mode that FOLLOWS a route: the
    # continuation keeps the reference a full-length forward interval on the last metres,
    # where it would otherwise collapse onto the final vertex and turn every path cost into
    # a radial pull toward one point.  It never tells the robot to drive past the end --
    # _select_reference refuses to plan from a foot already past it, and the node's arrival
    # guard normally holds before that.
    # NO mode sets it False today.
    #
    # The flag stays rather than being folded into the extension, because the collapse it
    # prevents is a property of a SHORT REFERENCE and not of any one mode: a future
    # two-point route that genuinely must not be overrun would want to opt out, and would
    # then also want _select_reference's foot guard, which this flag gates.
    extend_reference_past_end: bool = True




####################################### SOLVER OUTPUTS #######################################
# What the solver and the two tracking entrypoints hand back to the node.

@dataclass(frozen=True)
class MPPIResult:
    v: float
    w: float
    valid: bool
    solve_time_s: float
    predicted_path: Optional[np.ndarray]
    min_cost: float
    reason: str
    reference_endpoint: Optional[np.ndarray] = None
    warnings: tuple[str, ...] = ()
    # Per-tick MPPI tuning trace, or None when config.debug_print is off.  Carried on the
    # result as well as printed so a test can assert on it without capturing stdout.
    debug: Optional[str] = None


@dataclass(frozen=True)
class MPPIPathTrackingResult:
    """Command and diagnostics returned through main.py's existing contract."""

    v_cmd: float
    w_cmd: float
    waypoint: Optional[np.ndarray]
    diagnostic: Optional[str]
    warnings: tuple[str, ...]
    valid: bool = True




####################################### INTERNAL GEOMETRY #######################################
# Batched path-projection result; produced and consumed inside the planner only.

@dataclass(frozen=True)
class PathProjection:
    """Batched nearest-segment values sharing the positions' leading shape."""

    distance_sq: torch.Tensor
    segment_index: torch.Tensor
    fraction: torch.Tensor
    arc_length: torch.Tensor
    tangent_yaw: torch.Tensor
