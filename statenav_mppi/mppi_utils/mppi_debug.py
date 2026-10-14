"""The per-tick MPPI tuning trace: pure formatting, no planner.

Reached only when config.debug_print is true.

The routine "solve=.. cost=.." diagnostic says how long a tick took and what the best
sample scored; neither answers the question that actually blocks tuning, which is WHICH
term of the objective is setting the command and whether the sampler is healthy enough
for that answer to mean anything.  This module answers both.  It issues no command,
mutates no planner state, and is invoked behind a guard in solve() so a formatting bug
here can never take a control tick down.

Everything it reads arrives in one SolveTrace value plus the tick's CostManager and
ValidityManager, so the term set, the reward flags, the per-term debug lines and the
validity set are all read off the registries instead of being restated here -- adding a
cost term or a validity term changes nothing in this file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from statenav_mppi.mppi_core.mppi_validity import ValidityManager, ValidityVerdict
from statenav_mppi.mppi_core.mppi_costs import CostManager
from statenav_mppi.mppi_core.mppi_types import MPPIConfig, MPPIInput


@dataclass(frozen=True)
class SolveTrace:
    """Everything one tick's trace needs, collected by solve() as it goes.

    Carries ``data`` and ``config`` whole rather than restating the eight scalars the
    trace reads off them; both are frozen dataclasses, and neither is the planner.
    """

    data: MPPIInput                    # this tick's inputs
    config: MPPIConfig
    settled_controls: torch.Tensor     # (T, 2) projected commands this tick actually issues
    costs: torch.Tensor                # (K,) sample costs; +inf where the rollout was rejected
    weights: torch.Tensor              # (K,) softmax weights; 0 where the rollout was rejected
    temperature: float                 # temperature those weights were built at, not config's
    proposed: torch.Tensor             # (K, T, 2) command proposals BEFORE projection
    candidates: torch.Tensor           # (K, T, 2) the same proposals AFTER projection
    sample_terms: dict                 # {name: (K,)} cost components of every sample
    issued_terms: dict                 # {name: (1,)} components of the SETTLED batch-of-one
    issued_masked_total: float       # that plan's total after the validity screen
    verdict: ValidityVerdict         # which validity term eliminated which SAMPLE
    issued_verdict: ValidityVerdict  # the same for the SETTLED batch-of-one
    constraint_signature: str        # how the command rectangle under the robot is composed
    knot_zero_vmax: float              # (m/s)   command domain of the cell under the robot
    knot_zero_wmax: float              # (rad/s) command domain of the cell under the robot
    solve_time_s: float
    reference_points: Any              # (N, 2) local reference the costs were measured against
    # ── best-sample guard and deterministic seeds ─────────────────────────────────────
    # Defaulted, so a caller built before these existed still constructs; solve() always
    # supplies them.  seed_count is how many of the LAST rows of the batch are the
    # deterministic manoeuvre seeds, which is what makes "were the seeds admissible" a
    # question the trace can answer without being told which rows they are.
    seed_count: int = 0
    guard_adoptions: int = 0           # passes this tick that issued a sample, not the mean
    guard_settled_cost: float = float("nan")   # weighted-mean plan's cost, last pass
    guard_best_cost: float = float("nan")      # cheapest drawn sample's cost, last pass


def _format_series(values: np.ndarray, digits: int = 2) -> str:
    """Render a per-knot command series ``(T,)`` as one compact bracketed line."""
    return "[" + " ".join(f"{float(value):+.{digits}f}" for value in values) + "]"


# (-) Knot-zero command shape that counts as "turn in place".  The coupled simplex makes
# the two conditions one -- after projection |v|/vmax + |w|/wmax == 1, so an angular
# command at its ceiling already forces the linear one to zero -- but both are tested.
_PIVOT_W_FRACTION: float = 0.90
_PIVOT_V_FRACTION: float = 0.10


def _pivot_probe_lines(
    trace: SolveTrace, issued_values: dict[str, float],
) -> tuple[str, ...]:
    """Report the cheapest TURN-IN-PLACE sample against the plan being issued.

    The rest of the trace says what the issued command paid and which term has the widest
    spread; neither answers the question that comes up whenever the robot arcs forward off
    a path it should be re-acquiring, which is "was a pivot ever on the table, and if so
    what outbid it".  Those are two different failures with opposite fixes -- the sampler
    never drawing one is a noise/warm-start problem, the objective refusing one is a
    weight problem -- and the level and spread lines cannot tell them apart.

    So: mask the batch to samples whose KNOT ZERO is a pivot, take the cheapest, and print
    its per-term decomposition as a DELTA against the issued plan.  The term with the
    largest positive delta is the one paying for the pivot not to happen.

    Reads only what SolveTrace already carries -- candidates, costs and the per-sample
    term components the trace collects anyway -- so it costs one masked argmin and no
    extra cost evaluation.  Comparing a raw SAMPLE against the SETTLED weighted mean is
    deliberate and is the comparison that matters: the settled plan is what was sent.
    """
    if not trace.sample_terms:
        return ()
    vmax, wmax = float(trace.knot_zero_vmax), float(trace.knot_zero_wmax)
    if not (math.isfinite(vmax) and math.isfinite(wmax)) or vmax <= 0.0 or wmax <= 0.0:
        return ()
    # (K,) knot-zero command magnitudes AFTER projection: what the rollout actually flew,
    # not the raw proposal the projection may have clipped onto a bound.
    knot_zero = trace.candidates[:, 0, :].detach()
    linear, angular = knot_zero[:, 0].abs(), knot_zero[:, 1].abs()
    finite = torch.isfinite(trace.costs)
    w_gate, v_gate = _PIVOT_W_FRACTION * wmax, _PIVOT_V_FRACTION * vmax
    mask = (angular >= w_gate) & (linear <= v_gate) & finite
    drawn = int(mask.sum().item())
    header = (
        f"[mppi][dbg]   pivot probe: {drawn}/{int(trace.costs.numel())} admissible "
        f"sample(s) turn in place at knot 0 (|w|>={w_gate:.2f}, |v|<={v_gate:.2f})")
    if drawn == 0:
        if not bool(finite.any()):
            return (header + " -- no sample survived the validity screen",)
        # (K,) how pivot-SHAPED each sample is, so a tick with no pivot still says whether
        # the sampler came close.  A value near +1 is a pivot; near -1 is full ahead.
        shape = torch.where(
            finite, angular / wmax - linear / vmax,
            torch.full_like(angular, -math.inf))
        nearest = int(torch.argmax(shape).item())
        return (
            header + f" -- nearest drawn: v={float(knot_zero[nearest, 0]):+.2f} "
            f"w={float(knot_zero[nearest, 1]):+.2f}; the SAMPLER never proposed one, so "
            f"no cost term rejected it (raise noise_w, or the warm start is too straight)",)
    # The cheapest pivot: argmin over the masked subset, taken on the SCREENED costs so a
    # sample the validity table rejected can never be reported as the best alternative.
    masked = torch.where(mask, trace.costs, torch.full_like(trace.costs, math.inf))
    best = int(torch.argmin(masked).item())
    pivot = {name: float(tensor[best]) for name, tensor in trace.sample_terms.items()}
    pivot_total, issued_total = sum(pivot.values()), sum(issued_values.values())
    # Per-term (pivot - issued).  Positive entries are what the pivot pays EXTRA, i.e. the
    # terms that vetoed it; negative entries are what it would have saved.
    delta = {name: value - issued_values.get(name, 0.0) for name, value in pivot.items()}
    against = max(delta, key=delta.get)
    verdict = ("CHEAPER than the issued plan -- the update averaged it away, not the "
               "objective" if pivot_total < issued_total else
               f"dearer by {pivot_total - issued_total:+.2f}, mostly {against}")
    commands = trace.candidates[best].detach().cpu().numpy()          # (T,2) the pivot plan
    return (
        header,
        f"[mppi][dbg]     best pivot: v={_format_series(commands[:, 0])} "
        f"w={_format_series(commands[:, 1])} total={pivot_total:+.2f} "
        f"vs issued {issued_total:+.2f} -> {verdict}",
        f"[mppi][dbg]     delta (pivot - issued): "
        + " ".join(f"{name}={value:+.2f}" for name, value in delta.items())
        + f" | worst against the pivot: {against} {delta[against]:+.2f}",
    )


def _guard_lines(trace: SolveTrace) -> tuple[str, ...]:
    """Report the best-sample guard and the deterministic seeds -- one line, always.

    Two facts that are invisible in every other section and that decide whether A and B
    are doing anything at all:

      * whether the plan being issued is the softmax-weighted MEAN or a rollout that was
        adopted whole because the mean priced worse than it.  An adopted plan is a single
        ``nominal + one noise draw``, so its horizon tail is unsmoothed and the next warm
        start is rougher -- adoption has to be countable, not silent.
      * how many of the deterministic seeds survived the validity screen.  A seeded pivot
        that is vetoed every tick is a corridor or map problem, not a sampling one, and
        the pivot probe below cannot tell the two apart on its own.

    Reads only what SolveTrace carries; costs nothing beyond the formatting.
    """
    settled, best = trace.guard_settled_cost, trace.guard_best_cost
    if math.isfinite(settled) and math.isfinite(best) and abs(best) > 1.0e-9:
        # Positive: the mean priced WORSE than the cheapest rollout the batch drew, which
        # is what the guard exists to catch.
        excess = f"{100.0 * (settled - best) / abs(best):+.0f}%"
    else:
        excess = "n/a"
    verdict = (f"ADOPTED the sample ({int(trace.guard_adoptions)} pass(es))"
               if int(trace.guard_adoptions) > 0 else "issued the weighted mean")
    seeds = int(trace.seed_count)
    if seeds <= 0:
        seed_text = "seeds off (batch too small to spare rows)"
    else:
        # The seeds are the LAST `seed_count` rows of the batch by construction, so their
        # admissibility is read straight off the screened costs.
        admissible = int(torch.isfinite(trace.costs[-seeds:]).sum().item())
        # The level set itself is not restated here: it lives on MPPISolver, which imports
        # THIS module, so quoting it would either invert that dependency or silently drift.
        seed_text = (f"seeds {admissible}/{seeds} admissible "
                     f"(u levels in both signs, + the warm start)")
    return (
        f"[mppi][dbg]   guard: settled={settled:+.2f} best sample={best:+.2f} "
        f"({excess}) -> {verdict} | {seed_text}",
    )


def format_tick_trace(
    costs: CostManager, validity: ValidityManager, trace: SolveTrace,
) -> str:
    """Build the per-tick tuning trace for the plan that is about to be issued.

    Both registries are passed rather than restated here: the term set, the reward flags,
    the per-term debug lines AND the validity set are all read off them, so adding either
    a cost term or a validity term changes nothing in this file.
    """
    data = trace.data
    config = trace.config
    # ── (1) Cost decomposition of the ISSUED plan ─────────────────────────────────
    # Evaluated on the settled batch-of-one rollout, not on the cheapest sample: the
    # question is what the command being sent paid, not what some rejected sample paid.
    # Re-uses THIS tick's bound objective rather than binding a fresh one, so a term
    # that keeps a snapshot reports the ISSUED plan's geometry.
    values = {name: float(tensor.reshape(-1)[0])
              for name, tensor in trace.issued_terms.items()}
    total = sum(values.values())
    # Only the penalties compete for "what is capping the command".  progress is the one
    # negative term (a reward for moving along the reference), so it is reported AGAINST
    # the penalty sum rather than inside it -- when it is small next to the dominant
    # penalty, the cost-optimal command is a crawl no matter how wide the domain is.
    reward_names = {term.name for term in costs.terms if term.reward}
    penalties = {name: value for name, value in values.items()
                 if name not in reward_names}
    penalty_sum = sum(penalties.values())
    dominant = max(penalties, key=penalties.get) if penalties else "n/a"
    dominant_share = (
        100.0 * penalties[dominant] / penalty_sum
        if abs(penalty_sum) > 1.0e-9 else float("nan"))
    # cost units bought by moving forward: the reward terms, reported positive
    reward = -sum(values.get(name, 0.0) for name in reward_names)
    reward_share = (
        100.0 * reward / abs(penalty_sum) if abs(penalty_sum) > 1.0e-9 else float("nan"))
    rejected_plan = not math.isfinite(trace.issued_masked_total)
    # WHICH validity term rejected the plan about to be issued.  "[PLAN REJECTED]" alone
    # left the operator to guess between a tight map, a diverged rollout and a non-finite
    # cost.
    issued_dominant = trace.issued_verdict.dominant_reason()
    rejected_plan_text = (
        "" if not rejected_plan
        else f"  [PLAN REJECTED BY {issued_dominant or 'the validity screen'}]")

    # ── (1b) Which term actually SELECTS the command ──────────────────────────────
    # The level above says what the issued plan paid; it does not say what chose it.  A
    # term that is large but identical across every sample (a uniform map floor, say)
    # dominates the decomposition while contributing nothing to the softmax -- only the
    # part of a term that VARIES with the command can move the argmin.  So the spread
    # across samples, not the level, is what to retune when the command will not grow.
    spread = {}
    for name, tensor in trace.sample_terms.items():
        finite_term = tensor[torch.isfinite(tensor)]
        spread[name] = (
            float(torch.std(finite_term).item()) if finite_term.numel() > 1
            else float("nan"))
    selector = max(
        (name for name in spread if math.isfinite(spread[name])),
        key=lambda name: spread[name], default="n/a")

    # ── (1c) Per-term internals ───────────────────────────────────────────────────
    # A term whose LEVEL alone is unreadable publishes its own line.  The
    # line comes off the bound term, which is the object that actually holds the
    # snapshot, instead of a planner attribute.
    term_lines = costs.debug_lines()

    # ── (1d) The alternative the update did not take ─────────────────────────────
    # Sections (1) and (1b) describe the command that WAS chosen.  This one describes the
    # best turn-in-place sample that was not, and which term paid to keep it that way --
    # the question every "why does it arc forward instead of re-acquiring the path" tick
    # raises, and the one thing the level/spread pair cannot answer.
    probe_lines = _guard_lines(trace) + _pivot_probe_lines(trace, values)
    # Vetoes publish their internals here for the same reason cost terms do: a rejection
    # COUNT cannot distinguish a tight bound from a mis-sized one that had to abstain.
    probe_lines += validity.debug_lines()

    # ── (2) Sampler health ───────────────────────────────────────────────────────
    # ESS = 1/sum(w^2) is the effective number of samples the update averaged over.
    # ESS ~ 1     -> temperature too low: the update is one sample, high variance.
    # ESS ~ K     -> temperature too high: the weights are uniform and the update is ~0.
    # spread ~ 0  -> the cost cannot tell the samples apart; noise_v/w is too small, or
    #                every rollout is dominated by the same constant term.
    finite = torch.isfinite(trace.costs)
    rejected = int((~finite).sum().item())
    # WHY those samples died, per validity term.  Without this a tick where the map killed
    # every sample is indistinguishable from one where the map was merely tight -- and the
    # tuning response to the two is opposite.  Per-term and therefore not a partition: a
    # sample failing two terms is counted by both, so the parts need not sum to
    # `rejected`.  Ordered by count, then name, so consecutive ticks read the same way.
    rejections = trace.verdict.rejection_counts()
    rejected_detail = (
        "" if not rejections else " (" + ", ".join(
            f"{name} {count}" for name, count in sorted(
                rejections.items(), key=lambda item: (-item[1], item[0]))) + ")")
    finite_costs = trace.costs[finite]
    has_finite = bool(finite_costs.numel())
    cost_min = float(torch.min(finite_costs).item()) if has_finite else float("nan")
    cost_median = float(torch.median(finite_costs).item()) if has_finite else float("nan")
    weight_sq = float(torch.sum(trace.weights ** 2).item())
    ess = 1.0 / weight_sq if weight_sq > 0.0 else float("nan")
    weight_max = float(torch.max(trace.weights).item())
    # The temperature the weights were actually built at, annotated with the configured
    # floor whenever the ESS search had to raise it.  Without the floor beside it a raised
    # lambda reads as a config change, and a tick whose cost scale exploded is
    # indistinguishable from one that was simply retuned between runs.
    temperature_note = (
        f" (floor {float(config.temperature):.3f})"
        if trace.temperature > float(config.temperature) else "")

    # ── (3) Domain usage of the issued command ───────────────────────────────────
    # v_use + w_use is the coupled simplex _project_and_rollout enforces.  At ~1.0 the
    # command sits ON the envelope and no cost retuning can raise it.  Reading a value
    # BELOW 1.0 changed with the equality law: the simplex is no longer a ceiling the
    # objective may leave unspent, so below 1.0 does NOT mean "the objective chose to go
    # slower" -- the projection grows every coupled command back onto the boundary.  It
    # means the RATE BOX bound first (lin_acc/ang_acc x dt could not reach the boundary in
    # one interval), i.e. the robot is ramping, or the safety-first fallback fired on this
    # tick, which the accompanying rate-infeasible warning names.  A steady-state tick that
    # is neither should read 1.00; a persistent value below it is the thing to investigate.
    #
    # READ THE RAMP EXPLANATION ABOVE SCEPTICALLY, because it used to be the whole story
    # and was the wrong place to look.  A third cause dominated in practice: a knot whose
    # linear command was EXACTLY zero was skipped by the growth entirely and sat off the
    # boundary at any usage at all.  Across the 30 runs of the 2026-09-10 suite every
    # single off-simplex tick was that case and not one was a rate-box ramp, and in three
    # runs it became an absorbing standstill that ended them in simulation_timeout.
    # _expand_to_boundary now takes the sign of such a zero from the admissible interval,
    # so on a one-sided interval -- which is every reachable lane in wp_tracking and in a
    # braked reversed rollout -- the zero is grown like any other command.  What remains
    # below 1.0 is therefore a real ramp, the safety-first fallback, or a collapsed domain,
    # and a FULL STOP is now reported as `stop` rather than as a usage of 0.00, so it can
    # no longer be misread as either.
    settled = trace.settled_controls.detach().cpu().numpy()      # (T, 2) issued command plan
    v_cmd = float(settled[0, 0])
    w_cmd = float(settled[0, 1])
    v_use = abs(v_cmd) / trace.knot_zero_vmax if trace.knot_zero_vmax > 1.0e-9 else float("nan")
    w_use = abs(w_cmd) / trace.knot_zero_wmax if trace.knot_zero_wmax > 1.0e-9 else float("nan")
    # A full stop is named rather than printed as a usage of 0.00.  The two are the same
    # number and NOT the same event: 0.00 in the old trace could be a ramp from standstill,
    # a collapsed domain, or the ungrown-zero stall, and an operator scanning for "why is
    # this below 1.00" had no way to tell.  `stop` says the issued command is zero on both
    # axes.  After the zero-sign rule that is no longer something the objective can choose,
    # so it means the DOMAIN left no room -- a rectangle collapsed by the caller's limit or
    # by the map, an admissible interval that is exactly [0, 0] (reachable when an open-loop
    # reverse hands the rate anchor a previous_v at or below -lin_acc * dt), or the
    # safety-first fallback, which its own warning names.  All three point at the limits and
    # the rate anchor rather than at the cost weights, which is the diagnostic the bare
    # 0.00 could not give.
    #
    # The tolerance is for float noise, not policy: settled_controls is a weighted mean that
    # is re-projected, so a genuine stop can arrive as 1e-17 rather than as an exact zero,
    # and an exact comparison would miss precisely the case this is here to name.  1e-9 m/s
    # is far under any actuator resolution.
    full_stop = abs(v_cmd) <= 1.0e-9 and abs(w_cmd) <= 1.0e-9
    simplex_text = (
        "simplex=stop" if full_stop
        else f"simplex={v_use + w_use:.2f}/1.00")

    # ── (4) Knot-zero clipping ───────────────────────────────────────────────────
    # How much of the sampled noise survives projection.  Outside reversed geometry the
    # linear domain is one-sided (v_lo = 0), so a noise_v comparable to the domain width
    # clips a large share of samples back onto a bound; those samples carry
    # effective_noise = bound - nominal and drag the update toward that bound, which is
    # indistinguishable from "MPPI refuses to speed up" when read off the command alone.
    # lo/hi are the percentages pushed UP off the lower bound and DOWN off the upper one.
    delta = (trace.candidates[:, 0, :] - trace.proposed[:, 0, :]).detach()   # (K, 2) projection correction
    tolerance = 1.0e-6
    clip_lo = (delta > tolerance).to(delta.dtype).mean(dim=0) * 100.0
    clip_hi = (delta < -tolerance).to(delta.dtype).mean(dim=0) * 100.0
    mean_delta = delta.mean(dim=0)
    clip_lo_np = clip_lo.cpu().numpy()
    clip_hi_np = clip_hi.cpu().numpy()
    mean_delta_np = mean_delta.cpu().numpy()

    # ── (5) Reference geometry the costs were measured against ───────────────────
    # A reference that is shorter than the horizon can travel silently caps progress: the
    # projection clips at the endpoint, so extra speed stops earning any reward at all.
    points = np.asarray(trace.reference_points, dtype=float)      # (N, 2) local reference polyline
    reference_len = (
        float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        if points.shape[0] > 1 else 0.0)
    end = points[-1]
    d_end = math.hypot(
        float(end[0]) - float(data.pose[0]), float(end[1]) - float(data.pose[1]))
    # Distance the horizon can actually cover at the reach speed the reference was sized
    # with -- the ceiling floored at the conservative speed, as _select_reference floors it --
    # for comparison with reference_len.
    horizon_reach = (
        max(float(config.conservative_speed_v), float(data.cmd_v_limit))
        * config.horizon * config.mppi_dt * config.lookahead_time_scale)

    lines = [
        f"[mppi][dbg] tag={data.tag} K={int(trace.costs.numel())} T={config.horizon} "
        f"mppi_dt={config.mppi_dt:.2f} lag={config.lag_time:.2f} "
        f"ctrl_dt={float(data.control_dt):.3f} solve={trace.solve_time_s * 1.0e3:.0f}ms "
        f"| obj: {costs.signature()} | valid: {validity.signature()}",
        "[mppi][dbg] cost(issued): "
        + " ".join(f"{name}={value:+.2f}" for name, value in values.items())
        + f" | total={total:+.2f}"
        + rejected_plan_text,
        f"[mppi][dbg]   level: dominant penalty={dominant} ({dominant_share:.0f}% of "
        f"{penalty_sum:+.2f}); progress reward={reward:+.2f} ({reward_share:.1f}% of "
        f"penalties)",
        "[mppi][dbg]   selection (std across K -- this is what picks the command): "
        + " ".join(f"{name}={value:.2f}" for name, value in spread.items())
        + f" -> {selector}",
        f"[mppi][dbg] sampler: lambda={trace.temperature:.3f}{temperature_note} "
        f"noise_v={config.noise_v:.2f} noise_w={config.noise_w:.2f} "
        f"taps={int(config.noise_smoothing_taps)} iters={int(config.iterations)} "
        f"ESS={ess:.1f}/{int(trace.costs.numel())} wmax={weight_max:.3f} | "
        f"cost min={cost_min:.2f} p50={cost_median:.2f} "
        f"spread={cost_median - cost_min:.2f} rejected={rejected}{rejected_detail}",
        f"[mppi][dbg] domain[{trace.constraint_signature}]: "
        f"knot0 vmax={trace.knot_zero_vmax:.3f} wmax={trace.knot_zero_wmax:.3f} "
        f"(caller {float(data.cmd_v_limit):.3f}/{float(data.cmd_w_limit):.3f}) | "
        f"use v={v_use * 100.0:.0f}% w={w_use * 100.0:.0f}% "
        + simplex_text,
        f"[mppi][dbg] issued v={_format_series(settled[:, 0])} "
        f"w={_format_series(settled[:, 1])}",
        f"[mppi][dbg] clip@knot0: v lo={float(clip_lo_np[0]):.0f}% "
        f"hi={float(clip_hi_np[0]):.0f}% dv={float(mean_delta_np[0]):+.3f} | "
        f"w lo={float(clip_lo_np[1]):.0f}% hi={float(clip_hi_np[1]):.0f}% "
        f"dw={float(mean_delta_np[1]):+.3f}",
        f"[mppi][dbg] reference: len={reference_len:.2f}m pts={points.shape[0]} "
        f"end=({float(end[0]):.2f}, {float(end[1]):.2f}) d_end={d_end:.2f}m "
        f"horizon_reach={horizon_reach:.2f}m",
    ]
    for offset, line in enumerate(term_lines + probe_lines):
        # Directly under the selection line, which is where the pivot probe's verdict
        # belongs too.
        lines.insert(4 + offset, line)
    return "\n".join(lines)
