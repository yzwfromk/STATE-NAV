"""Constraints on the command itself: the admissible set at one knot, and the projection.

A CONSTRAINT here shapes the command BEFORE it is ever rolled out -- the rate box, the
cell rectangle, the coupled simplex.  It is enforced by PROJECTION inside the causal
loop, so a constrained command is not "rejected", it is moved onto the boundary and the
rollout continues.

That is a different mechanism from a VALIDITY test, which judges a rollout AFTER the
fact and produces a boolean mask -- see mppi_validity.py.  The two used to share the
word "constraint", which made it impossible to say which one a sentence meant:

    constraint (here)      command -> admissible command      projection, per knot
    validity   (there)     rollout -> True / False            veto, per batch

CommandConstraint is a value object rather than a registry, on two grounds:

  1. The projection is ORDER-COUPLED -- coupled_vmax depends on coupled_w, which depends
     on v_now.  Independent registered predicates cannot express that.
  2. It sits inside a T-iteration loop running at ~20 Hz, where per-term Python dispatch
     is not free.

What CAN become a registry is the ``(vmax, wmax)`` INPUT to it: limiters combine with an
elementwise minimum, which is commutative, so they really are independent.  That is the
intended home of a future ConstraintTermTable in this module -- the name is deliberately
left free for it.

Nothing here imports the planner, and nothing here imports mppi_costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from statenav_mppi.mppi_core.mppi_types import MPPIConfig, MPPIInput


# Slack on the coupled simplex feasibility test, in dimensionless domain-usage units.
# A knot whose rate box only just touches the simplex must not be reported infeasible
# because the two bounds disagree in the last float32 bit.
_SIMPLEX_TOLERANCE = 1.0e-7
# Floor under both simplex normalizers, so a collapsed domain divides to a FINITE number
# instead of NaN.  What it does NOT do is decide anything: project() discards the floored
# lanes through `positive_limits`, which selects the plain rectangle clip for them.
#
# Reachability, since this is the assumption the whole guard rests on: on every control
# path both normalizers are strictly positive.  MPPICore._command_limits floors
# vmax/wmax at min(cmd_limit, conservative_speed_v/w); MPPIConfig validates both
# conservative speeds strictly positive, and MPPIProblem validates both command limits
# strictly positive.  So the floor is defensive against a future constraint term,
# not something today's stack exercises.
#
# One deliberate consequence for feasible(): the knot-zero warning used to restate this
# arithmetic in scalar Python WITHOUT the floor, where a fully collapsed domain divided
# 0/0 to NaN and reported "infeasible" by accident.  Sharing one implementation adopts the
# floored form, under which that same case reports "feasible" -- correctly, since a zero
# command is then the only admissible one and it IS admissible.  Unreachable today for the
# reason above; the point is that the two copies can no longer disagree about it.
_DENOMINATOR_FLOOR = 1.0e-9
# Floor on domain usage below which a command has no DIRECTION to expand along.  Purely a
# NaN guard, not a policy knob: the expansion scale is 1/usage, and usage == 0 would make
# it 0 * inf.  Reachable on a real path -- MPPISolver seeds `nominal` with zeros on every
# mode transition and projects it before the first sample is drawn -- so this is not
# defensive.  1e-6 of a 1.5 m/s domain is 1.5e-6 m/s, far under any actuator resolution.
#
# NO LONGER LOAD-BEARING, and kept only so the note above is not lost.  It described the
# RAY growth, whose scale was 1/usage; the linear-only _expand_to_boundary that replaced it
# never divides by usage, so there is no 0 * inf to guard and nothing in this module reads
# this value.  The zero case is now decided by the sign of the admissible interval, on an
# exact IEEE comparison and deliberately WITHOUT a tolerance band -- a band here would widen
# the discontinuity the growth already has at zero instead of removing it.
_DIRECTION_FLOOR = 1.0e-6




####################################### PER-KNOT COMMAND CONSTRAINT #######################################

@dataclass(frozen=True)
class CommandConstraint:
    """The admissible command set at ONE knot, plus the projection onto it.

    Shape- and dtype-agnostic on purpose.  The batched rollout builds it from ``(K,)``
    float32 tensors and the knot-zero feasibility warning from 0-d float64 ones, and both
    run the SAME arithmetic -- which is the entire reason this type exists.  The warning
    used to restate the rate box and the simplex test in scalar Python, unguarded, and the
    two copies had already drifted: the batched path clamped its denominators and gated the
    simplex behind positive limits, and the scalar copy did neither.

    Every field is a tensor of one broadcastable shape:

      v_lower/v_upper       rate box ∩ cell rectangle, linear -- the interval a command
      w_lower/w_upper       may actually be clipped into at this knot, angular
      v_lo/v_hi             the cell rectangle ALONE, rate-free; the safety-first fallback
                            projects onto this when the rate box and the simplex are disjoint
      wmax                  simplex normalizer, angular (the cell's angular ceiling)
      domain_vmax           simplex normalizer, linear = max(|v_lo|, |v_hi|)
      domain_denominator    wmax and domain_vmax floored by _DENOMINATOR_FLOOR.  Derived
      angular_denominator   fields rather than locals so the two clamps are paid once per
                            knot instead of once in project() and again in feasible().
    """

    v_lower: torch.Tensor
    v_upper: torch.Tensor
    w_lower: torch.Tensor
    w_upper: torch.Tensor
    v_lo: torch.Tensor
    v_hi: torch.Tensor
    wmax: torch.Tensor
    domain_vmax: torch.Tensor
    domain_denominator: torch.Tensor
    angular_denominator: torch.Tensor

    @staticmethod
    def build(
        vmax: torch.Tensor,
        wmax: torch.Tensor,
        previous_v: torch.Tensor,
        previous_w: torch.Tensor,
        dt: float,
        config: MPPIConfig,
        *,
        reversed_geometry: bool,
    ) -> "CommandConstraint":
        """Intersect the rate box with the cell rectangle for one knot.

        ``vmax``/``wmax`` are the command-domain limits of the cell being occupied NOW --
        the causal choice, because the state after an interval depends on the command being
        projected.  ``previous_v``/``previous_w`` is the command the rate bound is measured
        from (the last issued command at knot zero, the preceding projected command after),
        and ``dt`` the interval that bound spans: the measured ``control_dt`` at knot zero,
        the fixed ``mppi_dt`` after it.

        ``reversed_geometry`` opens the negative half of the linear domain and closes the
        positive half to BRAKING ONLY: the forward ceiling of the rate box becomes the
        rate-limited step toward zero, ``max(0, previous_v - lin_acc * dt)``, so a reversed
        tick still brakes out of its forward speed at the physical rate limit but can
        never accelerate forward again.  Driving forward is therefore not something the
        reversed cost has to outbid -- it is not in the domain at all.

        The mode RECTANGLE ``v_lo``/``v_hi`` is deliberately left alone by that rule: it is
        what the simplex normalizer (``domain_vmax``) and the rate-free fallback of
        project() are defined against, so capping it instead of the rate box would both
        reshape the simplex and let the fallback step the command discontinuously to zero.
        """
        if reversed_geometry:
            v_lo, v_hi = -vmax * config.recovery_reverse_scale, vmax
        else:
            v_lo, v_hi = torch.zeros_like(vmax), vmax
        # (...,) upper edge of the rate box.  Reversed geometry caps it at the brake ramp:
        # strictly positive only while previous_v is still too fast for one interval at
        # lin_acc to reach zero, and exactly 0 from that interval on -- including when
        # previous_v is already negative, where ``previous_v + lin_acc * dt`` alone would
        # let a retreating command cross back through zero into forward drive.
        rate_upper = previous_v + config.lin_acc * dt
        if reversed_geometry:
            rate_upper = torch.minimum(
                rate_upper,
                torch.clamp(previous_v - config.lin_acc * dt, min=0.0))
        domain_vmax = torch.maximum(torch.abs(v_lo), torch.abs(v_hi))
        return CommandConstraint(
            v_lower=torch.maximum(v_lo, previous_v - config.lin_acc * dt),
            v_upper=torch.minimum(v_hi, rate_upper),
            w_lower=torch.maximum(-wmax, previous_w - config.ang_acc * dt),
            w_upper=torch.minimum(wmax, previous_w + config.ang_acc * dt),
            v_lo=v_lo,
            v_hi=v_hi,
            wmax=wmax,
            domain_vmax=domain_vmax,
            domain_denominator=torch.clamp(domain_vmax, min=_DENOMINATOR_FLOOR),
            angular_denominator=torch.clamp(wmax, min=_DENOMINATOR_FLOOR),
        )

    @classmethod
    def from_scalars(
        cls,
        vmax: float,
        wmax: float,
        previous_v: float,
        previous_w: float,
        dt: float,
        config: MPPIConfig,
        *,
        reversed_geometry: bool,
    ) -> "CommandConstraint":
        """Build from Python floats, as 0-d float64 CPU tensors.

        Same arguments as build(), and it IS build() -- only the shapes differ.  That is
        the whole point of this type being shape-agnostic: the knot-zero feasibility
        report runs the identical arithmetic the batched projection runs one knot later,
        so the two cannot drift apart the way the hand-restated scalar copy did.

        The dtype is deliberate.  float64 reproduces bit-for-bit the Python-float
        arithmetic the knot-zero test used to do inline, so adopting the shared
        implementation cannot move a marginal verdict by a float32 rounding step.  CPU
        because this is one scalar per tick, and a device round-trip for it would cost
        more than the arithmetic.
        """
        def scalar(value: float) -> torch.Tensor:
            return torch.tensor(float(value), dtype=torch.float64)

        return cls.build(
            scalar(vmax), scalar(wmax), scalar(previous_v), scalar(previous_w),
            float(dt), config,
            reversed_geometry=reversed_geometry)

    def feasible(self) -> torch.Tensor:
        """Boolean of the field shape: True where SOME command satisfies every constraint.

        The test is the simplex one: the cheapest reachable point of
        the rate box (the closest either interval can get to zero on each axis) must still
        fit inside the domain simplex.

        The two denominators are the floored ones, so a cell that collapses a lane's domain
        to zero yields a finite verdict rather than a NaN that compares False by accident.
        """
        interval_ordered = (
            (self.v_lower <= self.v_upper) & (self.w_lower <= self.w_upper))
        # (...,) smallest |command| the rate interval permits on each axis: zero when the
        # interval straddles zero, otherwise the nearer endpoint.
        minimum_abs_v = torch.where(
            (self.v_lower <= 0.0) & (self.v_upper >= 0.0),
            torch.zeros_like(self.v_lower),
            torch.minimum(torch.abs(self.v_lower), torch.abs(self.v_upper)),
        )
        minimum_abs_w = torch.where(
            (self.w_lower <= 0.0) & (self.w_upper >= 0.0),
            torch.zeros_like(self.w_lower),
            torch.minimum(torch.abs(self.w_lower), torch.abs(self.w_upper)),
        )
        return interval_ordered & (
            minimum_abs_v / self.domain_denominator
            + minimum_abs_w / self.angular_denominator
            <= 1.0 + _SIMPLEX_TOLERANCE
        )

    def _expand_to_boundary(
        self, v: torch.Tensor, w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Grow one in-domain command onto the simplex boundary along the LINEAR axis.

        ``v``/``w`` must already lie inside BOTH the rate box and the simplex -- that is
        what project()'s coupled clip produces, and the growth below is only correct for
        such a point: it is derived as one-way from that assumption.

        The growth is LINEAR-ONLY, and that asymmetry is the whole point.  Growing along
        the v/w ray -- which is what this used to do -- makes braking and turning the SAME
        action: whenever the rate box floors |v| above what the objective wants, the only
        way the update can express "slower" is "turn harder", and the linear budget the
        objective declined to spend is handed to yaw that nothing asked for.  That is the
        in-place-spin failure: |v| + |w| pinned at the ceiling on every knot while the
        robot rotates through a full turn without advancing, the sign of the rotation set
        by whatever few degrees of heading error happened to be there when the braking
        started.  extend_path_end's docstring already names one instance of it (progress
        saturating at the path end); the rate box floors |v| for the same effect on any
        approach to a corner, which is not something a longer reference can fix.

        Holding |w| at the value the sampler and the cost actually chose, and spending the
        remaining budget on |v|, keeps "every knot spends the whole command budget" intact
        on the axis that carries the robot forward while leaving the angular command
        alone.  The admissible SET is unchanged -- still rate box n simplex -- so
        feasible() and infeasible_reason() describe this projection exactly as before, and
        the result can only ever have LESS |w| and MORE |v| than the ray growth produced.

        A knot that genuinely wants to pivot is unaffected: at |w| == wmax the allowance
        below is zero, so turn-in-place is still reachable -- it now has to be REQUESTED
        rather than arrived at by braking.

        Two things stop the growth, in this order: the simplex allowance left over once
        |w| is paid for, and the rate box.  Whichever binds first wins, so a ramp out of
        standstill is rate-limited rather than reported infeasible.

        A knot whose linear command is EXACTLY zero is grown along the sign the admissible
        interval supplies, and left alone only when that interval admits both signs.  The
        long comment on ``zero_sign`` below is the argument for it; the short version is
        that an ungrown zero is a fixed point of this projection and the sampler produces
        those zeros in bulk, so leaving them alone let the plan get stuck at standstill.
        """
        # (...,) linear magnitude the simplex still allows once |w| is paid for.  This IS
        # the boundary |v|/domain_vmax + |w|/wmax == 1 solved for |v|, so a command grown
        # to it sits exactly ON the boundary, as the ray growth did.
        allowance = self.domain_vmax * torch.clamp(
            1.0 - torch.abs(w) / self.angular_denominator, min=0.0)
        # (...,) grow-only, exactly as the scale clamp was: a knot whose usage already
        # rounds past 1.0 (the rate box and the safety set are disjoint there, feasible()
        # is False and project() substitutes the safety-first fallback) keeps its own
        # magnitude instead of being shrunk onto the boundary by this method.
        magnitude = torch.maximum(torch.abs(v), allowance)
        # (...,) the sign to grow a knot whose linear command is EXACTLY zero along.  A zero
        # carries no direction of its own, but the admissible interval [v_lower, v_upper] --
        # the rate box intersected with the cell rectangle -- usually does: where that
        # interval is one-sided, every nonzero command it admits has the SAME sign, so
        # growing along that sign invents nothing.  It is the only direction the growth
        # could ever take.  Both modes that matter are one-sided:
        #
        #   wp_tracking        v_lo = 0, so the interval is [0, v_upper]  -> +
        #   reversed_geometry  rate_upper collapses to exactly 0 once previous_v can reach
        #                      zero in one interval (see build()), so the interval is
        #                      [v_lower, 0]                              -> -
        #
        # Where the interval straddles zero both signs are admissible, the choice really is
        # arbitrary, and the zero is left alone exactly as it always was.  That third branch
        # is DEFENSIVE against a future build(), not a case today's stack reaches: forward
        # geometry has v_lower = max(0, previous_v - lin_acc * dt) >= 0, so it can never be
        # negative, and in reversed geometry v_upper > 0 requires previous_v > lin_acc * dt,
        # which forces v_lower = previous_v - lin_acc * dt > 0 as well.  So every REACHABLE
        # lane is one-sided (or collapsed to [0, 0], where the only admissible command is
        # the zero and leaving it alone is right), and the equality holds unconditionally
        # wherever a direction exists at all.
        #
        # WHY this is not "inventing a heading".  Leaving the zero ungrown made it a FIXED
        # POINT of the projection, and the sampler manufactures those zeros in bulk: the
        # noise is symmetric and centred on the warm start, so once a settled plan reads
        # zero the nominal sits ON the one-sided bound and about half the batch proposes a
        # command the rectangle clips to an exact zero.  Those lanes all score identically,
        # the guard adopts one, the warm start is zero again, and the plan can never leave
        # standstill -- an absorbing state that ended three runs of the 2026-09-10 suite in
        # simulation_timeout (009, 013 and 021, every one
        # of their off-simplex ticks an exact zero and not one a rate-box ramp).  Under this
        # rule a stop has to be REQUESTED through the domain -- cmd_v_limit = 0 collapses
        # domain_vmax and project()'s `positive_limits` select discards the coupled branch --
        # rather than arrived at by the sampler.  Every path that commands a true stop
        # already does that or bypasses the projection entirely.
        forward_only = (self.v_lower >= 0.0) & (self.v_upper > 0.0)
        reverse_only = (self.v_upper <= 0.0) & (self.v_lower < 0.0)
        zero_sign = torch.where(
            forward_only, torch.ones_like(magnitude),
            torch.where(
                reverse_only, -torch.ones_like(magnitude),
                torch.zeros_like(magnitude)))
        # Every nonzero command keeps its own sign, so the growth never reverses a knot.
        signed = torch.where(
            v < 0.0, -magnitude,
            torch.where(v > 0.0, magnitude, zero_sign * magnitude))
        # The rate box is the second and final bound on the growth.
        return (
            torch.minimum(torch.maximum(signed, self.v_lower), self.v_upper),
            w,
        )

    def project(
        self, raw_v: torch.Tensor, raw_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project one knot's raw command targets onto this domain.

        ``raw_v``/``raw_w`` are the sampled targets of this knot, of a shape broadcastable
        against the fields; the returned pair has that same shape.  ``raw_v``/``raw_w`` are
        read twice -- once through the rate box, once through the rate-FREE rectangle for
        the safety-first fallback -- so neither is overwritten here.

        On the coupled branch the simplex is an EQUALITY, not a ceiling:

            |v| / domain_vmax  +  |w| / wmax  ==  1

        The clip below moves a raw target INTO the domain and _expand_to_boundary() then
        grows it back OUT onto the boundary along the LINEAR axis alone, so the projection
        selects the fastest command carrying the angular rate the sampler asked for rather
        than the nearest admissible one.  Growing along the v/w RAY instead -- which this
        used to do -- spent the linear budget the objective declined as unrequested yaw,
        which is the in-place-spin failure _expand_to_boundary now documents at length.
        The admissible SET is unchanged -- still rate box ∩ simplex -- so
        feasible() and infeasible_reason() describe this projection exactly as before.

        One documented exception to the equality, by construction rather than by a
        branch here: the safety-first fallback stays shrink-only, because it fires only on
        the tick that is
        already relaxing the rate bound and emitting an operator warning.  The rate box
        also legitimately holds a command below the boundary -- that is a ramp, not a
        violation, and it is why the growth is capped by the box instead of overriding it.
        """
        v_now = torch.minimum(torch.maximum(raw_v, self.v_lower), self.v_upper)
        w_now = torch.minimum(torch.maximum(raw_w, self.w_lower), self.w_upper)

        # Per-sample replacement of the former scalar `domain_vmax > 0 and
        # wmax > 0` guard: a cell may collapse one sample's domain while
        # leaving the rest of the batch untouched.
        positive_limits = (self.domain_vmax > 0.0) & (self.wmax > 0.0)
        # (...,) True where the rate box and the safety set still intersect; the one
        # feasibility test, shared with the knot-zero warning rather than restated.
        rate_and_safety_feasible = self.feasible()

        # Project onto the rate-bounded rectangle intersected with the
        # coupled command diamond. Preserve linear braking first and
        # reduce angular demand; only reduce |v| when angular rate
        # bounds prevent enough steering reduction.
        coupled_wmax = self.wmax * torch.clamp(
            1.0 - torch.abs(v_now) / self.domain_denominator, min=0.0)
        coupled_w = torch.clamp(w_now, -coupled_wmax, coupled_wmax)
        coupled_w = torch.minimum(
            torch.maximum(coupled_w, self.w_lower), self.w_upper)

        coupled_vmax = self.domain_vmax * torch.clamp(
            1.0 - torch.abs(coupled_w) / self.angular_denominator, min=0.0)
        coupled_v = torch.clamp(v_now, -coupled_vmax, coupled_vmax)
        coupled_v = torch.minimum(
            torch.maximum(coupled_v, self.v_lower), self.v_upper)

        # The law is EQUALITY, not a ceiling: having clipped INTO the domain, grow back
        # out onto its boundary along the same ray, so every knot spends the whole command
        # budget the cell allows and the sampler's v/w ratio decides only how it is split.
        coupled_v, coupled_w = self._expand_to_boundary(coupled_v, coupled_w)

        # If a terrain-limit change makes the rate box and safety set
        # disjoint, no command can satisfy both. Apply the approved
        # safety-first policy: command bounds and simplex remain hard,
        # while solve() reports the unavoidable rate relaxation.
        safe_v = torch.minimum(torch.maximum(raw_v, self.v_lo), self.v_hi)
        safe_w = torch.clamp(raw_w, -self.wmax, self.wmax)
        safe_usage = (
            torch.abs(safe_v) / self.domain_denominator
            + torch.abs(safe_w) / self.angular_denominator
        )
        safe_scale = torch.clamp(safe_usage, min=1.0)
        safe_v = safe_v / safe_scale
        safe_w = safe_w / safe_scale
        coupled_v = torch.where(rate_and_safety_feasible, coupled_v, safe_v)
        coupled_w = torch.where(rate_and_safety_feasible, coupled_w, safe_w)
        return (
            torch.where(positive_limits, coupled_v, v_now),
            torch.where(positive_limits, coupled_w, w_now),
        )

    def infeasible_reason(self) -> str:
        """The operator-facing warning naming which policy was prioritized instead.

        Operators recognise this verbatim; it is not to be reworded.
        """
        return (
            "first-command rate constraints infeasible after command limits "
            "changed; command-domain and coupled safety limits prioritized"
        )




####################################### COMMAND RECTANGLE REGISTRY #######################################

# fn(state, data) -> (vmax, wmax) of the state's leading shape.  Both are CEILINGS on the
# magnitude of the command at this knot, and the manager folds them with an elementwise
# MINIMUM, so a registered term can only ever TIGHTEN the rectangle -- never widen it.
# +inf means "this term has nothing to say here", which is the identity of that fold.
ConstraintFn = Callable[[torch.Tensor, MPPIInput], tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class ConstraintTermEntry:
    """One command constraint's structure.  Built once at construction, never per knot."""

    name: str            # attribution key
    fn: ConstraintFn


class ConstraintTermTable:
    """Immutable, ordered table of the terms that TIGHTEN the command rectangle.

    Composed bottom layer up via _build_constraint_table(), calling super() FIRST and
    appending, exactly as the cost and validity tables are -- so adding a constraint is one
    edit in the file that owns it.

    Order is legibility, not semantics: the fold is an elementwise minimum, which is
    commutative, so permuting the table cannot change the rectangle.  It fixes the order
    the trace names them in, and nothing else.

    Two constraints are deliberately NOT in here, and both for the same reason -- the
    projection has to be able to tell them apart from the rectangle:

      * the RATE box (lin_acc/ang_acc x dt).  CommandConstraint.build() intersects it in
        afterwards, because project() must be able to DROP it: when the rate box and the
        simplex are disjoint the safety-first fallback keeps the rectangle and the simplex
        and relaxes the rate bound.  A term folded in here would be indistinguishable from
        the rectangle at that point.
      * the coupled SIMPLEX, whose normalizers (domain_vmax, wmax) are defined against the
        rectangle ALONE.  ``coupled_wmax = wmax * (1 - |v_now| / domain_denominator)`` reads
        the rectangle for its normalizers and the rate-clipped box for ``v_now`` in ONE
        expression, so the two subsets cannot be collapsed into a single ordered fold.

    The caller's mode ceiling (``cmd_v_limit``/``cmd_w_limit``) is not a term either: it is
    the SEED the fold starts from, so it applies even on the warm-start pass that skips
    every registered term.
    """

    def __init__(self) -> None:
        self._entries: list[ConstraintTermEntry] = []
        self._names: set[str] = set()

    @property
    def entries(self) -> tuple[ConstraintTermEntry, ...]:
        """Registered constraints in fold order."""
        return tuple(self._entries)

    def add(self, name: str, *, fn: ConstraintFn) -> None:
        """Register one command constraint.

        Raises on a duplicate or empty name -- a construction error, not a tick error: a
        control tick must never be the place a mis-registered constraint is discovered.
        """
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"command constraint name must be a non-empty string, got {name!r}")
        if name in self._names:
            raise ValueError(
                f"command constraint {name!r} is already registered; names are the "
                f"attribution keys and must be unique (registered so far: "
                f"{sorted(self._names)})")
        self._entries.append(ConstraintTermEntry(name=name, fn=fn))
        self._names.add(name)

    def signature(self) -> str:
        """``"Mode & Terrain"`` -- the rectangle's composition, for the trace.

        Names the SEED first, because it is what the fold starts from and it is the one
        constraint that survives the warm-start pass.
        """
        return " & ".join(("Mode",) + tuple(e.name for e in self._entries))
