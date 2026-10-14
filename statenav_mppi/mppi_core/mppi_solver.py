"""The MPPI update itself: sample, score, reweight, settle.

Between the generic core and the traversability layer:

    MPPICore -> MPPISolver -> TerrainMPPI -> PathTrackingMPPI -> MPPIPlanner

It sits below the map and below the reference path on purpose.  Everything one update
needs arrives in the MPPIProblem the per-run entrypoint hands it -- this tick's inputs and
an objective already bound to them -- so the only thing this layer asks of the layers ABOVE
it is a command-constraint term, which MPPICore's registry already declares.  There is no map pin and no
reference selection here: one map version and one reference must serve every rollout of a
tick, and the tick begins at the entrypoint, not here.

The two step primitives stay separate on purpose: "a batch was drawn and priced" and "the
weighted update was applied" are distinct steps of one update, and solve() reads better as
those two calls than as one twenty-line block.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from statenav_mppi.mppi_core.mppi_core import MPPICore
from statenav_mppi.mppi_core.mppi_constraints import CommandConstraint
from statenav_mppi.mppi_core.mppi_validity import ValidityVerdict
from statenav_mppi.mppi_core.mppi_costs import CostContext, MPPIProblem
from statenav_mppi.mppi_utils.mppi_debug import SolveTrace, format_tick_trace
from statenav_mppi.mppi_core.mppi_types import MPPIInput, MPPIResult


# ── Deterministic pivot seeds (see MPPISolver._write_seed_rows) ──────────────────────
# Domain-usage levels u = |w| / wmax the seeded rows hold, and the sign set they are held
# in.  The coupled simplex is an EQUALITY, so u alone determines the whole command --
# v = vmax * (1 - u) -- and one row per (u, sign) therefore spans the entire
# one-dimensional control manifold rather than one axis of a two-dimensional box.  u = 1.0
# is the pure pivot: zero linear command, the full angular ceiling, zero lateral excursion.
# The three interior values are spaced to bracket the arc radii the sampler already
# reaches (u ~ 0.55) and the ones it cannot (R = 0.33 m at u = 0.75).
_SEED_ANGULAR_FRACTIONS: tuple[float, ...] = (0.4, 0.7, 0.9, 1.0)
# Knots a seed HOLDS its level for before tapering back to the warm start.  A manoeuvre
# has to survive several knots to be worth anything to the softmax -- a command's influence
# on the cost falls off with the square of the rollout time remaining after it -- and
# tapering rather than holding to the horizon end keeps the seed a proposal about the NEXT
# second, not a demand that the whole plan be a pivot.
_SEED_HOLD_KNOTS: int = 3


@dataclass(frozen=True)
class MPPIStep:
    """One sampled batch: the proposals, their rollout and their scores.

    Returned by _sample_and_score() and consumed by _settle(), so the sampled batch travels
    between the two as one named value instead of six loose tensors.
    """

    proposed: torch.Tensor         # (K,T,2) command proposals BEFORE projection
    candidates: torch.Tensor       # (K,T,2) the same proposals AFTER projection
    states: torch.Tensor           # (K,T,5) rollout of candidates
    costs: torch.Tensor            # (K,) total cost; +inf where the rollout was rejected
    weights: torch.Tensor          # (K,) softmax weights; 0 where the rollout was rejected
    all_invalid: bool              # True when no sample survived, so no update is defined
    terms: Optional[dict]          # {name: (K,)} components, or None when not collected
    verdict: ValidityVerdict       # which validity term eliminated which sample
    temperature: float             # softmax temperature THIS batch was weighted at
    seed_count: int = 0            # how many of the LAST rows are deterministic seeds


@dataclass(frozen=True)
class _GuardOutcome:
    """What the best-sample guard found on ONE pass, and whether it swapped the plan.

    Carried out of _adopt_best_sample() so solve() can both report it in the trace and
    RE-USE its cost evaluation: the guard prices the settled batch-of-one, which is exactly
    the evaluation the trace would otherwise repeat.
    """

    settled_cost: float               # screened total of the weighted-mean plan
    best_cost: float                  # screened total of the cheapest drawn sample
    best_index: int                   # which sample that was, or -1 when none was finite
    adopted: bool                     # True when the sample replaced the settled plan
    terms: Optional[dict]             # {name: (1,)} components of the SETTLED plan, or None
    verdict: Optional[ValidityVerdict]  # validity of that same settled plan


class MPPISolver(MPPICore):
    """One MPPI update over an MPPIProblem: no map, no reference path, no task."""




####################################### STEP PRIMITIVES #######################################
# Draw-and-price, then weight-and-settle.  Split so solve() reads as the two steps of one
# MPPI update rather than as a single block, and so the sampled batch that travels between
# them is a named value with a documented shape contract.

    def _sample_and_score(
        self,
        problem: MPPIProblem,
        nominal: torch.Tensor,
        initial_state: torch.Tensor,
        *,
        generator: torch.Generator,
        collect_terms: bool = False,
        seed_limits: Optional[tuple[float, float]] = None,
    ) -> MPPIStep:
        """Draw ``num_samples`` perturbations of ``nominal``, roll them out and price them.

        ``nominal`` is ``(T, 2)`` and ``initial_state`` ``(5,)``.  The horizon is read off
        ``nominal`` rather than from the config, so the shape contract is local to the call.
        ``generator`` is passed rather than taken from the planner so the sample sequence is
        an explicit input of this step instead of hidden planner state.

        ``collect_terms`` fills MPPIStep.terms with each term's ``(K,)`` contribution; it
        costs nothing beyond the dict, because the evaluation happens either way.

        ``seed_limits`` is the ``(vmax, wmax)`` command rectangle the deterministic seeds
        below are expressed as a fraction of -- knot zero's, so a seeded pivot is a pivot
        against the domain the command actually being issued is projected onto.  None falls
        back to the caller's mode ceiling.
        """
        horizon = int(nominal.shape[-2])
        sigma = torch.tensor(
            (self.config.noise_v, self.config.noise_w), device=self.device)
        # (K,T,2) command proposals: the warm start plus command noise whose per-knot std is
        # sigma and whose correlation ACROSS knots is set by config.noise_smoothing_taps.
        noise = self._draw_noise(horizon, sigma, generator=generator)
        proposed = nominal[None, :, :] + noise
        # COVERAGE, not distribution: the last few rows are overwritten with deterministic
        # manoeuvres the Gaussian cannot reach (see _write_seed_rows).  Written BEFORE the
        # projection, so a seed is clipped onto the rate box and the simplex and screened by
        # the validity table exactly like a drawn sample -- it can neither violate the
        # command domain nor bypass a veto.
        seed_count = self._write_seed_rows(
            proposed, nominal, seed_limits, problem.data)
        candidates, states = self._project_and_rollout(
            proposed, initial_state, problem.data)
        terms: Optional[dict] = {} if collect_terms else None
        # ONE context for both managers: the terrain cost and the terrain veto share a
        # memoized footprint lookup on it, so a second context would double the dominant
        # per-batch map cost.  MPPIProblem.score() is what enforces that.
        costs, verdict = problem.score(
            CostContext(states=states, controls=candidates, data=problem.data),
            terms=terms)
        # The softmax temperature is in COST UNITS, so it is fitted to THIS batch's costs
        # rather than read straight off the config: a value that averages properly on open
        # ground collapses the same softmax onto a single rollout once the terrain penalty
        # raises the cost scale.  Per PASS, not per tick, because each MPPI iteration
        # resamples around the plan the previous one settled on and so has its own spread.
        temperature = (
            self.adaptive_temperature(
                costs,
                self.config.temperature,
                self.config.adaptive_temperature_target_ess,
                self.config.adaptive_temperature_max_scale)
            if bool(self.config.adaptive_temperature)
            else float(self.config.temperature))
        weights, all_invalid = self.stable_weights(costs, temperature)
        return MPPIStep(
            proposed=proposed, candidates=candidates, states=states, costs=costs,
            weights=weights, all_invalid=all_invalid, terms=terms, verdict=verdict,
            temperature=temperature, seed_count=seed_count)

    def _settle(
        self,
        problem: MPPIProblem,
        nominal: torch.Tensor,
        initial_state: torch.Tensor,
        step: MPPIStep,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Apply the weighted update and re-project it; ``None`` when nothing survived.

        Returns ``(controls (T,2), states (T,5))``.  The update is the cost-weighted mean of
        the projected candidates' deviation from ``nominal``, which is what makes the
        settled plan admissible: the deviations are measured AFTER projection, so a sample
        that was clipped onto a bound contributes the bound and not its raw proposal.
        """
        if step.all_invalid:
            return None
        update = torch.sum(
            step.weights[:, None, None] * (step.candidates - nominal[None, :, :]),
            dim=0)
        return self._project_and_rollout(
            nominal + update, initial_state, problem.data)




####################################### COVERAGE AND ADOPTION #######################################
# The two halves of "the update must be able to CHOOSE a manoeuvre the Gaussian cannot
# average its way into".  _write_seed_rows puts that manoeuvre in the batch; the guard is
# what lets it be issued, because a softmax-weighted mean of 1015 arcs and 9 pivots is an
# arc.  Neither changes the admissible set: seeds are projected and screened like any
# sample, and the guard only ever adopts a rollout that already survived the screen.

    def _write_seed_rows(
        self,
        proposed: torch.Tensor,
        nominal: torch.Tensor,
        seed_limits: Optional[tuple[float, float]],
        data: MPPIInput,
    ) -> int:
        """Overwrite the LAST rows of ``proposed`` with deterministic manoeuvre seeds.

        ``proposed`` is the ``(K,T,2)`` raw proposal batch, mutated IN PLACE; ``nominal`` is
        the ``(T,2)`` warm start.  Returns how many rows were written -- 0 when the batch is
        too small to spare them, so a caller reading MPPIStep.seed_count can tell "no seed
        was admissible" from "no seed was written".

        Overwriting rather than appending is deliberate: ``K``, every tensor shape and the
        sample count the trace prints stay identical, and the ~1 % of the batch spent here
        costs no extra rollout.

        WHY the batch needs them.  The coupled simplex is an equality, so after projection
        the command at a knot is the ONE-dimensional choice u = |w| / wmax, with
        v = vmax * (1 - u) and turn radius R = v / w.  The reachable range of a smoothed
        Gaussian around a near-straight warm start is u in [0, ~0.55] -- the 3-tap knot
        smoothing removes exactly the tail a pivot needs, leaving an effective per-knot
        sigma of ~0.20 -- so u >= 0.9 is a >5 sigma event and is drawn 0 times in 1024.  The
        robot therefore takes the tightest arc it CAN draw (R ~ 1 m) where the path asks for
        a turn in place, which costs 2x the time and ~2 m of lateral excursion a corridor
        cannot afford.  u = 1 already satisfies the simplex equality (0/vmax + wmax/wmax = 1)
        and _command_limits floors ``vmax``, never ``v``, so the pivot is already IN the
        admissible set: this is purely a sampling-coverage problem.

        WHY BOTH AXES ARE WRITTEN, when u alone determines the command.  The projection
        does derive |v| from |w| -- _expand_to_boundary grows the linear command into
        whatever the simplex leaves once |w| is paid for -- but it derives it in the OTHER
        order first (coupled_wmax = wmax * (1 - |v|/domain_vmax)).  So writing w alone and
        leaving v at the warm start would let a large warm-start v clip the seeded w away.
        Writing the simplex point itself, v = vmax * (1 - u), is what makes each seed the
        manoeuvre it is meant to be -- and it is a fixed point of the projection, so a seed
        that the rate box does not bind comes out of it unchanged.

        This paragraph used to add that the growth is a no-op on a command whose linear
        part is exactly zero, so that writing w with v = 0 would degenerate every level
        into "rotate in place at u * wmax".  That is no longer true, and the reason matters
        for REVERSED geometry, where these seeds are written forward (levels scale a
        non-negative `linear`) into a domain whose only admissible sign is minus: every
        seed clipped to v = 0 and the batch really did degenerate into nine pivots, which
        is the failure this paragraph warns about, arriving through the projection instead
        of through the seed arithmetic.  _expand_to_boundary now takes the sign of a zero
        from the admissible interval, so such a seed clips to zero and is then grown to
        -vmax * (1 - u) -- the reverse arc the level names.  The arithmetic here needs no
        sign flip of its own; it self-repairs through the projection.
        """
        rows = int(proposed.shape[0])
        horizon = int(proposed.shape[-2])
        # One row per (level, sign) plus one unperturbed warm start as the control.
        count = 2 * len(_SEED_ANGULAR_FRACTIONS) + 1
        if rows <= count or horizon <= 0:
            # Nothing to spare: seeding here would leave the batch with too few DRAWN
            # samples for the softmax to be an average at all.  Silent by design -- it can
            # only happen on a batch far smaller than any configured one -- but visible,
            # because the returned 0 reaches the trace.
            return 0
        # (m/s, rad/s) the rectangle the levels are a fraction OF.  Knot zero's cell domain
        # when the caller supplied it, so a seeded u matches the u the tick trace's pivot
        # probe measures; the mode ceiling otherwise.  A non-finite or non-positive angular
        # ceiling would make every seed the warm start again, so it is rejected.
        linear, ceiling = (
            seed_limits if seed_limits is not None
            else (float(data.cmd_v_limit), float(data.cmd_w_limit)))
        linear, ceiling = float(linear), float(ceiling)
        if not math.isfinite(ceiling) or ceiling <= 0.0:
            linear, ceiling = float(data.cmd_v_limit), float(data.cmd_w_limit)
        if not math.isfinite(ceiling) or ceiling <= 0.0:
            return 0
        if not math.isfinite(linear) or linear < 0.0:
            linear = 0.0
        # (T,) how much of each knot belongs to the SEED rather than to the warm start: a
        # full hold over the first knots, then a linear taper to 0 at the last knot, so the
        # seed proposes a manoeuvre now and hands the horizon tail back to the warm start.
        hold = min(int(_SEED_HOLD_KNOTS), horizon)
        alpha = torch.ones(horizon, device=proposed.device, dtype=proposed.dtype)
        if horizon > hold:
            alpha[hold:] = torch.linspace(
                1.0, 0.0, horizon - hold + 1,
                device=proposed.device, dtype=proposed.dtype)[1:]
        # (N-1,) the simplex point of each (level, sign): the signed angular target and the
        # linear command the equality |v|/vmax + |w|/wmax == 1 leaves for it.
        levels = torch.tensor(
            [level for level in _SEED_ANGULAR_FRACTIONS for _ in (1.0, -1.0)],
            device=proposed.device, dtype=proposed.dtype)
        signs = torch.tensor(
            [sign for _ in _SEED_ANGULAR_FRACTIONS for sign in (1.0, -1.0)],
            device=proposed.device, dtype=proposed.dtype)
        targets = signs * levels * ceiling
        # (N-1,T) blend of the held target and the warm start's own plan, per axis.
        angular = (alpha[None, :] * targets[:, None]
                   + (1.0 - alpha[None, :]) * nominal[None, :, 1])
        forward = (alpha[None, :] * (linear * (1.0 - levels))[:, None]
                   + (1.0 - alpha[None, :]) * nominal[None, :, 0])
        seeds = torch.zeros(
            (count, horizon, 2), device=proposed.device, dtype=proposed.dtype)
        seeds[:count - 1, :, 0] = forward
        seeds[:count - 1, :, 1] = angular
        # The control row: the warm start itself, unperturbed.  It is what makes "the batch
        # held something better than the plan" a statement about the seeds and the draws
        # rather than about the warm start having been discarded.
        seeds[count - 1] = nominal
        proposed[rows - count:] = seeds
        return count

    def _adopt_best_sample(
        self,
        problem: MPPIProblem,
        step: MPPIStep,
        updated: torch.Tensor,
        predicted: torch.Tensor,
        *,
        collect_terms: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, _GuardOutcome]:
        """Replace the settled plan by the cheapest drawn sample when it is meaningfully worse.

        ``updated`` is the ``(T,2)`` plan _settle() produced and ``predicted`` its ``(T,5)``
        rollout; the returned pair is either those two or the cheapest sample and ITS
        rollout.  Runs per PASS, not per tick: each MPPI iteration resamples around the
        previous one's result, so pass two must resample around whatever was actually
        adopted.

        The MPPI update is ``nominal + sum_i w_i (candidate_i - nominal)`` -- a
        softmax-weighted MEAN.  On a bimodal batch (turn left of the hazard, turn right of
        it) the mean lands BETWEEN the modes, where the cost is higher than either, and the
        adaptive temperature cannot fix it: the ESS controller is already pinned at its 5 %
        floor on those ticks, so there is no sharpening left to do.  This is the mechanism
        that converts a drawn pivot into an issued one.

        Nothing needs re-projecting and no veto is bypassed: ``step.candidates`` is already
        post-projection, ``step.states`` is its own rollout, and a sample with a +inf cost
        (which is what the validity screen produces) can never be the argmin.

        The margin is relative to the MAGNITUDE of the best cost rather than the plain
        ``best * (1 + margin)`` of a strictly positive cost: the total is a sum of penalties
        MINUS the progress reward and may legitimately go negative, and there the plain form
        would put the threshold BELOW the value it is meant to sit above and invert the test.
        """
        # The batch-of-one evaluation of the plan about to be issued.  Unconditional now --
        # it used to happen only behind debug_print -- because it is the guard's own
        # question; the trace re-uses it rather than scoring a second time.
        issued_terms: Optional[dict] = {} if collect_terms else None
        total, verdict = problem.score(
            CostContext(states=predicted[None, ...], controls=updated[None, ...],
                        data=problem.data),
            terms=issued_terms)
        settled_cost = float(total.reshape(-1)[0])
        finite = torch.isfinite(step.costs)
        if not bool(torch.any(finite)):
            # Unreachable on this path -- _settle() returned a plan, so the batch had a
            # survivor -- but the argmin below would be meaningless on an all-+inf row.
            return updated, predicted, _GuardOutcome(
                settled_cost=settled_cost, best_cost=float("inf"), best_index=-1,
                adopted=False, terms=issued_terms, verdict=verdict)
        # +inf is what the validity screen writes, so it can never win an argmin over a row
        # that has at least one finite entry -- which the guard above has just established.
        index = int(torch.argmin(step.costs).item())
        best_cost = float(step.costs[index])
        margin = float(self.config.guard_margin)
        threshold = best_cost + margin * abs(best_cost)
        # A non-finite settled cost means the WEIGHTED MEAN itself was vetoed -- possible
        # even when every sample it averaged was admissible, because the mean of two
        # admissible plans need not be one -- and adopting is then the only way to issue a
        # plan the validity table accepts.
        adopted = (not math.isfinite(settled_cost)) or settled_cost > threshold
        if not adopted:
            return updated, predicted, _GuardOutcome(
                settled_cost=settled_cost, best_cost=best_cost, best_index=index,
                adopted=False, terms=issued_terms, verdict=verdict)
        # Cloned, not sliced: a view would keep the whole (K,T,.) batch alive through
        # self._nominal until the next tick overwrites it.
        return (
            step.candidates[index].clone(),
            step.states[index].clone(),
            _GuardOutcome(
                settled_cost=settled_cost, best_cost=best_cost, best_index=index,
                adopted=True, terms=issued_terms, verdict=verdict),
        )




####################################### SOLVE #######################################
# One MPPI update: shift the warm start, sample, project, score, reweight, re-project.
# Everything it needs arrives in the MPPIProblem -- there is no map pin and no reference
# selection here, because one map version and one reference must serve the whole tick and
# the tick begins at the entrypoint, not here.

    def solve(self, problem: MPPIProblem) -> MPPIResult:
        start = time.perf_counter()
        data = problem.data
        # MPPIProblem.__post_init__ has already run every shape, finiteness and
        # positivity check this used to repeat; pose is re-read only for the result.
        pose = np.asarray(data.pose, dtype=np.float32)

        try:
            pose_t = torch.as_tensor(pose, device=self.device, dtype=self.dtype)
            velocity_t = torch.as_tensor(
                np.asarray(data.velocity, dtype=np.float32),
                device=self.device, dtype=self.dtype)
            initial_state = torch.cat((pose_t, velocity_t), dim=0)
            # Knot zero is bounded by the cell the robot currently occupies, so
            # feasibility is reported against those limits rather than the
            # caller's alone.  This needs the map snapshot the entrypoint pinned.
            knot_zero_vmax, knot_zero_wmax = self._command_limits(
                initial_state, data)
            # Publish that pair for the HUD immediately, before anything below can fail:
            # it is the only denominator the issued command's domain usage means anything
            # against.  The visualizer used to divide by the caller's limits, which on a
            # map-tightened cell reported a command sitting ON its ceiling as half-used.
            self.record_knot_zero_limits(
                float(knot_zero_vmax), float(knot_zero_wmax))
            # The SAME admissible set _project_and_rollout will build for knot zero, in
            # scalar shape.  Reported rather than enforced: when the rate box and the
            # safety set are disjoint no command satisfies both, and the approved policy
            # is that CommandConstraint.project() relaxes the rate bound and still issues
            # a command -- so this is an operator-facing warning, never a veto.
            knot_zero = CommandConstraint.from_scalars(
                float(knot_zero_vmax), float(knot_zero_wmax),
                *self.last_issued_command, float(data.control_dt), self.config,
                reversed_geometry=data.reversed_geometry)
            result_warnings = (
                () if bool(knot_zero.feasible())
                else (knot_zero.infeasible_reason(),))
            reversed_geometry = data.reversed_geometry
            if self._nominal is None or self._last_reversed != reversed_geometry:
                nominal = torch.zeros((self.config.horizon, 2), device=self.device, dtype=self.dtype)
                self._last_reversed = reversed_geometry
            else:
                # The elapsed control interval is known only when this solve
                # begins, so time-align the previous plan here instead of
                # assuming that exactly one fixed prediction knot elapsed.
                nominal = self._shift_nominal_for_elapsed_time(
                    self._nominal, data.control_dt)
            nominal, _ = self._project_and_rollout(
                nominal, initial_state, data, use_map_limits=False)
            # Keep the cached plan on the current time grid even if every new
            # rollout is rejected, and prevent a failed mode transition from
            # retaining controls that belonged to the previous mode.
            self._nominal = nominal

            # Collecting the per-sample components costs nothing extra: it fills a dict on
            # the cost evaluation solve() already performs.  None keeps that dict unbuilt
            # when the trace is off.
            # config.iterations sample-score-settle passes per tick, each RE-SAMPLING around
            # the plan the previous pass settled on.  One pass moves the plan by at most a
            # noise-scale step, and the horizon's TAIL needs several of them: a command's
            # influence on the cost falls off with the square of the rollout time remaining
            # after it, so the last knots converge far slower than knot zero.  Spreading
            # those passes across control ticks instead is what let a plan hold a straight
            # tail into the start of a turn.  _settle() returns an already-projected
            # sequence, so every pass samples around an admissible plan exactly as the
            # first one does.
            #
            # `step` stays bound to the last pass that PRODUCED a plan, never to a failed
            # one: the trace, the weights and min_cost below all describe the plan being
            # issued, and a batch where nothing survived has no finite cost to report.
            step: Optional[MPPIStep] = None
            failed: Optional[MPPIStep] = None
            failed_index = 0
            updated: Optional[torch.Tensor] = None
            predicted: Optional[torch.Tensor] = None
            # Best-sample guard bookkeeping.  `guard` is the outcome of the pass that
            # produced the plan being issued; `guard_is_last` says whether its batch-of-one
            # evaluation is still the MOST RECENT one of this tick, which is the only
            # condition under which the trace may re-use it instead of scoring again.
            guard: Optional[_GuardOutcome] = None
            guard_is_last = False
            adoptions = 0
            for iteration in range(int(self.config.iterations)):
                guard_is_last = False
                pass_step = self._sample_and_score(
                    problem, nominal, initial_state, generator=self._generator,
                    collect_terms=bool(self.config.debug_print),
                    seed_limits=(float(knot_zero_vmax), float(knot_zero_wmax)))
                settled = self._settle(problem, nominal, initial_state, pass_step)
                if settled is None:
                    # Nothing admissible in THIS pass.  A later pass failing does not
                    # invalidate the plan an earlier one already settled on, so keep that
                    # plan and stop iterating; only a first-pass failure leaves the tick
                    # with nothing to issue.
                    failed, failed_index = pass_step, iteration
                    break
                step = pass_step
                updated, predicted = settled
                # The update is a weighted MEAN, and a mean of a bimodal batch can be worse
                # than either mode.  Adopt the cheapest rollout the same batch drew when it
                # is, BEFORE the warm start is written: `nominal` below feeds the next pass,
                # and self._nominal after the loop feeds the next tick, so both follow the
                # plan that is actually issued.
                updated, predicted, guard = self._adopt_best_sample(
                    problem, pass_step, updated, predicted,
                    collect_terms=bool(self.config.debug_print))
                guard_is_last = True
                adoptions += int(guard.adopted)
                nominal = updated
            if updated is None or predicted is None or step is None:
                # This is the ONE tick the trace never gets to explain -- it is built
                # further down, after a settled plan exists -- so the attribution is
                # carried in the reason instead.  "all rollouts invalid" alone left a tick
                # where the map killed everything indistinguishable from one where the
                # rollout diverged, and the two have opposite responses.
                counts = {} if failed is None else failed.verdict.rejection_counts()
                detail = ", ".join(
                    f"{name} {count}" for name, count in sorted(
                        counts.items(), key=lambda item: (-item[1], item[0])))
                return self._invalid_result(
                    start,
                    "all rollouts invalid"
                    + (f" ({detail})" if detail else ""))
            if failed is not None:
                # The tick still issues the earlier pass's plan, so this is a warning and
                # not a rejection -- but a batch going fully inadmissible one pass after a
                # successful one says the samples are sitting on the edge of the admissible
                # set, which is worth seeing before it turns into a held position.
                result_warnings += (
                    f"MPPI iteration {failed_index + 1}/{int(self.config.iterations)} found "
                    f"no admissible rollout; issuing the plan settled by iteration "
                    f"{failed_index}",)
            costs, weights = step.costs, step.weights
            command = updated[0]
            # Store the optimized sequence on its current time grid.  The next
            # solve will shift it once that tick's measured control_dt is known.
            self._nominal = updated.detach()

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - start
            command_np = command.detach().cpu().numpy()
            predicted_np = predicted.detach().cpu().numpy()
            predicted_path = np.vstack((pose[None, :], predicted_np[:, :3]))
            min_cost = float(torch.min(costs[torch.isfinite(costs)]).detach().cpu())

            # Per-tick tuning trace. Built only when debug_print is on: it costs one extra
            # batch-of-one cost evaluation plus a device sync, so it is not produced
            # unconditionally and then thrown away. Guarded because a trace is diagnostic
            # output, not control output -- a failure here is reported and the tick still
            # issues its command, rather than being turned into "invalid result" by solve()'s
            # own ValueError handler.
            debug = None
            if bool(self.config.debug_print):
                try:
                    # The batch-of-one evaluation of the plan being ISSUED -- not of the
                    # cheapest sample -- because the question the trace answers is what the
                    # command being sent paid.  The guard has already priced exactly that
                    # plan, so it is scored again ONLY when the guard's evaluation no longer
                    # describes it: when the guard adopted a sample after scoring the mean,
                    # or when a later pass ran a batch after it.  Scoring last also matters
                    # beyond the cost: a term that keeps a snapshot (Travmap, Progress,
                    # Corridor) publishes its debug line from its LAST evaluation.
                    reuse = (
                        guard is not None and guard_is_last and not guard.adopted
                        and guard.terms is not None and guard.verdict is not None)
                    if reuse:
                        issued_terms = guard.terms
                        issued_total = guard.settled_cost
                        issued_verdict = guard.verdict
                    else:
                        issued_terms = {}
                        issued_masked_total, issued_verdict = problem.score(
                            CostContext(states=predicted[None, ...],
                                        controls=updated[None, ...], data=data),
                            terms=issued_terms)
                        issued_total = float(issued_masked_total.reshape(-1)[0])
                    debug = format_tick_trace(
                        problem.costs, problem.validity, SolveTrace(
                        data=data,
                        config=self.config,
                        settled_controls=updated,
                        costs=costs,
                        weights=weights,
                        temperature=step.temperature,
                        proposed=step.proposed,
                        candidates=step.candidates,
                        sample_terms=step.terms,
                        issued_terms=issued_terms,
                        issued_masked_total=issued_total,
                        verdict=step.verdict,
                        issued_verdict=issued_verdict,
                        seed_count=int(step.seed_count),
                        guard_adoptions=adoptions,
                        guard_settled_cost=(
                            float("nan") if guard is None else guard.settled_cost),
                        guard_best_cost=(
                            float("nan") if guard is None else guard.best_cost),
                        constraint_signature=self._constraint_table.signature(),
                        knot_zero_vmax=float(knot_zero_vmax),
                        knot_zero_wmax=float(knot_zero_wmax),
                        solve_time_s=elapsed,
                        reference_points=(
                            None if problem.reference is None
                            else problem.reference),
                    ))
                    print(debug)
                except (RuntimeError, ValueError, IndexError, TypeError) as exc:
                    debug = (f"[mppi][dbg] trace unavailable: "
                             f"{type(exc).__name__}: {exc}")
                    print(debug)

            valid = bool(np.all(np.isfinite(command_np)) and np.all(np.isfinite(predicted_path)) and math.isfinite(min_cost))
            reason = "ok" if valid else "non-finite MPPI result"
            if valid:
                self.record_issued_command(
                    float(command_np[0]), float(command_np[1]))
            else:
                self.record_issued_command(0.0, 0.0)
            return MPPIResult(
                float(command_np[0]), float(command_np[1]), valid, elapsed,
                predicted_path if valid else None, min_cost, reason,
                reference_endpoint=(
                    problem.reference[-1].astype(np.float32, copy=True)
                    if valid and problem.reference is not None else None),
                warnings=result_warnings if valid else (),
                debug=debug,
            )
        except ValueError as exc:
            return self._invalid_result(start, str(exc))
        except torch.cuda.OutOfMemoryError:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            self.reset()
            return self._invalid_result(start, "CUDA out of memory")
