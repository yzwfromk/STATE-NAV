"""Generic MPPI machinery: sampling grid, rollout projection and plan bookkeeping.

The bottom layer of the planner stack.  It knows about a unicycle state, a command
sequence and the rate/simplex constraints that shape one, and nothing else -- no
traversability map, no reference path, no task.  Those arrive in the layers above:

    MPPICore -> MPPISolver -> TerrainMPPI -> PathTrackingMPPI -> MPPIPlanner

The one thing this layer needs but cannot supply is what TIGHTENS the command
rectangle at a given state, which is a map query.  It is declared here as the
_constraint_table registry, and TerrainMPPI registers the traversability lookup into it.
"""

from __future__ import annotations

import math
import time
from typing import Any, Optional

import numpy as np
import torch

from statenav_mppi.mppi_core.mppi_constraints import CommandConstraint, ConstraintTermTable
from statenav_mppi.mppi_core.mppi_validity import ValidityTermTable
from statenav_mppi.mppi_core.mppi_costs import CostContext, CostTermTable
from statenav_mppi.mppi_core.mppi_dynamics import step_actuator_lag
from statenav_mppi.mppi_core.mppi_types import MPPIConfig, MPPIInput, MPPIResult


_TWO_PI = 2.0 * math.pi

# Geometric-bisection steps in MPPICore.adaptive_temperature.  Each step halves the RATIO
# of the bracket, so 24 steps resolve a 1000x search range to ~1.0000004x -- far past what
# the ESS target needs, and still only 24 softmaxes over a (K,) CPU row per MPPI pass.
_TEMPERATURE_SEARCH_STEPS = 24


class MPPICore:
    """Batched unicycle MPPI: noise, warm start, constraint projection and rollout."""

    def __init__(self, config: MPPIConfig) -> None:
        self.config = config
        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("mppi_device=cuda but CUDA is not available")
        device_name = "cuda" if config.device == "auto" and torch.cuda.is_available() else config.device
        if device_name == "auto":
            device_name = "cpu"
        self.device = torch.device(device_name)
        self.dtype = torch.float32
        self.num_samples = config.num_samples if self.device.type == "cuda" else config.cpu_num_samples
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(int(config.seed))
        # Knot-axis smoothing of the sampled command noise: the (2, 1, taps) kernel and the
        # (T,) per-knot variance normalizer that holds its marginal std at noise_v / noise_w.
        # Both are None at noise_smoothing_taps == 1, which is plain white noise.  Config
        # derived and tiny, so _fallback_to_cpu REBUILDS them instead of copying them.
        self._noise_kernel, self._noise_gain = self._build_noise_filter()
        self._nominal: Optional[torch.Tensor] = None
        # Rollout geometry the cached plan was built under: True for the reversed modes, None
        # before the first solve.  Compared instead of the tag STRING -- see
        # MPPIInput.reversed_geometry.
        self._last_reversed: Optional[bool] = None
        self._last_issued_command = torch.zeros(
            2, device=self.device, dtype=self.dtype)
        self._last_cmd_v_limit = 0.0
        self._last_cmd_w_limit = 0.0
        # (vmax, wmax) command domain knot zero was ACTUALLY projected against on the last
        # solve, or None before the first one.  Display-only bookkeeping, kept here beside
        # the caller limits because it is the pair the issued command was clipped by.
        self._last_knot_zero_limits: Optional[tuple[float, float]] = None
        self._last_result: Optional[MPPIResult] = None
        # Visualization sink.  Defaults to a no-op so this module stays ROS-free and
        # constructs identically in headless tests; main.py injects the real RViz
        # marker publisher through set_visualizer() when visualization is enabled.
        self._visualizer = _NullVisualizer()
        # Latched once a CUDA OOM has permanently migrated this planner to CPU.
        self._cuda_fallback_active = False
        # Structure of the objective: names, weights, order, reward flags.  Composed and
        # VALIDATED here, once -- a duplicate name or a bad weight is a construction error,
        # not something to find out about on a control tick.  Built last in __init__ so
        # every layer's overrides see a fully assigned self.config.
        self._cost_table = self._build_cost_table()
        # Structure of the validity test: names and order.  Same three lifetimes and
        # the same construction-time validation as the cost table above -- a duplicate
        # term name is a wiring bug, not something to discover on a control tick.
        self._validity_table = self._build_validity_table()
        # Structure of the command rectangle: which terms tighten it, and in what order.
        # Same construction-time validation as the two tables above.
        self._constraint_table = self._build_constraint_table()




####################################### PLAN STATE #######################################
# The warm-started nominal plan and the last solve's result.  reset_plan() drops the
# plan alone; reset() also clears the command the rate bound is measured from.

    @property
    def nominal_controls(self) -> Optional[np.ndarray]:
        if self._nominal is None:
            return None
        return self._nominal.detach().cpu().numpy().copy()

    @property
    def last_result(self) -> Optional[MPPIResult]:
        return self._last_result

    def reset_plan(self) -> None:
        """Discard the warm-start plan without changing the command anchor."""
        self._nominal = None
        self._last_reversed = None
        self._last_result = None
        # Dropped WITH _last_result: both describe the last tick that produced a plan, so
        # the HUD must not keep normalizing by a domain measured before the reset.
        self._last_knot_zero_limits = None

    def reset(self) -> None:
        """Reset both the warm-start plan and previously issued command."""
        self.reset_plan()
        self._last_issued_command.zero_()




####################################### COMMAND BOOKKEEPING #######################################
# The command the outer loop actually issued, which knot zero's rate bound is
# measured against, plus the terrain limits recorded for diagnostics and plotting.

    @property
    def last_cmd_limits(self) -> tuple[float, float]:
        return self._last_cmd_v_limit, self._last_cmd_w_limit

    @property
    def last_knot_zero_limits(self) -> Optional[tuple[float, float]]:
        """Knot-zero command domain of the last solve, or None before the first one.

        This is the pair the issued command was actually clipped by -- the cell under the
        robot, floored to conservative_speed_v/w -- and NOT ``last_cmd_limits``, which is
        the caller's terrain ceiling that the cell may tighten far below.  The two differ
        on every cell the map rates worse than the floor, and reporting usage against the
        caller's pair there invents headroom: a command pinned AT the cell ceiling of
        0.5 rad/s reads as "47% of 1.0 rad/s".
        """
        return self._last_knot_zero_limits

    @property
    def last_issued_command(self) -> tuple[float, float]:
        """Return the command used to constrain the next candidate knot zero."""
        command = self._last_issued_command.detach().cpu().numpy()
        return float(command[0]), float(command[1])

    def record_issued_command(self, v_cmd: float, w_cmd: float) -> None:
        """Record the finite command actually issued by the outer control loop."""
        values = (float(v_cmd), float(w_cmd))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("issued command must be finite")
        self._last_issued_command = torch.tensor(
            values, device=self.device, dtype=self.dtype)

    def set_cmd_limits(self, cmd_v_limit: float, cmd_w_limit: float) -> None:
        """Record current terrain limits for diagnostics and plotting."""
        self._last_cmd_v_limit = float(cmd_v_limit)
        self._last_cmd_w_limit = float(cmd_w_limit)

    def record_knot_zero_limits(self, vmax: float, wmax: float) -> None:
        """Record the knot-zero command domain of the solve now running, for display.

        Called by solve() once the cell limits are known.  Deliberately unvalidated, like
        set_cmd_limits above: a non-finite bound is a projection fault that FiniteControl
        already reports on the control path, and raising from display bookkeeping would
        bury it under solve()'s generic "invalid result" instead.  A non-finite value here
        prints as ``nan`` in the HUD, which is the honest reading.
        """
        self._last_knot_zero_limits = (float(vmax), float(wmax))




####################################### VISUALIZATION SINK #######################################
# Injected, never constructed here, so this module never imports rclpy.

    def set_visualizer(self, visualizer: Any) -> None:
        """Attach a visualization sink, or detach with None.

        The sink must provide ``update(...)`` with publish_plan()'s argument list and
        ``reset_history()``.  Injected rather than constructed here so this module
        never imports rclpy: see mppi_rviz.MPPIMarkerPublisher.
        """
        self._visualizer = _NullVisualizer() if visualizer is None else visualizer

    def reset_visualization(self) -> None:
        """Clear plot-only history without changing controller state."""
        self._visualizer.reset_history()

    def publish_plan(
        self,
        path: np.ndarray,
        waypoint: Optional[np.ndarray],
        robot_pose: np.ndarray,
        v_cmd: float,
        w_cmd: float,
        mode: str = "wp_tracking",
        time: float = 0.0,
    ) -> None:
        """Publish the MPPI plan overlay for this tick as an RViz MarkerArray."""
        self._visualizer.update(
            path, waypoint, robot_pose, v_cmd, w_cmd, self._last_result,
            mode=mode, source="mppi", stamp_s=float(time))




####################################### MATH UTILITIES #######################################
# Stateless helpers: angle wrapping, the finite-only softmax, the ESS-based temperature
# search that keeps that softmax from degenerating into an argmin, and the warm-start
# resampling that puts a previous plan back on the current time grid.

    @staticmethod
    def _wrap_torch(angle: torch.Tensor) -> torch.Tensor:
        return torch.remainder(angle + math.pi, _TWO_PI) - math.pi

    @staticmethod
    def _softmax_over_shifted(
        shifted: torch.Tensor, temperature: float,
    ) -> torch.Tensor:
        """Return the ``(n,)`` softmax of ``-shifted/temperature`` for min-shifted costs.

        ``shifted`` is a ``(n,)`` non-negative row -- the finite costs minus their own
        minimum.  The -80 logit floor keeps a far-off sample at a representable weight
        instead of underflowing, and it MUST be the same floor the temperature search
        prices: search one clamp and weight with another, and the ESS the search targets is
        not the ESS the returned weights have.  Both callers therefore go through here
        rather than restating the expression.
        """
        logits = torch.clamp(-shifted / temperature, min=-80.0, max=0.0)
        return torch.softmax(logits, dim=0)

    @staticmethod
    def effective_sample_size(weights: torch.Tensor) -> float:
        """Return ``1/sum(w^2)``: how many samples a weighted update really averaged.

        ``weights`` is a ``(n,)`` normalized softmax row.  1.0 means the update IS one
        sample -- the softmax has degenerated into an argmin -- and ``n`` means the weights
        are uniform and the update is ~0.
        """
        square_sum = float(torch.sum(weights * weights).item())
        if not math.isfinite(square_sum) or square_sum <= 0.0:
            raise ValueError(
                f"effective sample size undefined: sum(w^2)={square_sum!r} over "
                f"{int(weights.numel())} weights; a normalized softmax row cannot sum to "
                "zero or to a non-finite value, so the weights are not normalized")
        return 1.0 / square_sum

    @staticmethod
    def adaptive_temperature(
        costs: torch.Tensor,
        base_temperature: float,
        target_ess_frac: float,
        max_scale: float,
    ) -> float:
        """Return the temperature whose softmax averages ``target_ess_frac`` of the batch.

        ``costs`` is the raw ``(K,)`` cost row, +inf included; the search runs on exactly
        the finite subset stable_weights() is about to weight, so the ESS aimed for here is
        the ESS the issued weights actually have.

        The temperature is in COST UNITS, so one value fixed in config can only be right at
        one cost scale.  The terrain penalty moves that scale by ~100x between open ground
        and a footprint clipping a too-risky cell, and at the tight end a fixed temperature
        makes ``exp(-(S - S_min)/lambda)`` underflow for every sample but one: the softmax
        becomes an argmin, the "average of K rollouts" is one rollout, and because the noise
        is redrawn every tick that one rollout changes every tick.  Raising the temperature
        until enough samples carry weight is what makes the update an average again.

        RAISE-ONLY by construction -- ``base_temperature`` is the floor of the search, so a
        batch whose ESS already clears the target keeps exactly the behaviour it had before
        this existed.  The ceiling is ``base_temperature * max_scale``: ESS increases
        monotonically with the temperature, so an uncapped search would happily answer
        "uniform weights, zero update" for a batch the cost genuinely cannot separate.
        """
        finite = torch.isfinite(costs)
        finite_costs = costs[finite]
        count = int(finite_costs.numel())
        # Fewer than two survivors: ESS is 1 at every temperature, so there is nothing to
        # search for and the configured floor is the only defensible answer.
        if count < 2:
            return float(base_temperature)
        # (n,) non-negative shifted costs, moved to CPU ONCE.  The bisection reads ESS back
        # as a Python float on every step, and doing that on the device would sync the GPU
        # once per step instead of once per solve.
        shifted = (finite_costs - torch.min(finite_costs)).detach().to("cpu")
        # Effective sample count to aim for, clamped into [1, n]: a target outside that
        # range is unreachable and would only drive the bisection onto a bound.
        target = min(max(float(target_ess_frac) * count, 1.0), float(count))

        def ess_at(temperature: float) -> float:
            return MPPICore.effective_sample_size(
                MPPICore._softmax_over_shifted(shifted, temperature))

        low = float(base_temperature)
        if ess_at(low) >= target:
            return low
        high = low * float(max_scale)
        if ess_at(high) < target:
            # The ceiling cannot reach the target.  It is still the best available answer,
            # and the tick trace prints the temperature actually used, so a run pinned here
            # is visible rather than silent.
            return high
        # Geometric bisection: the answer sits orders of magnitude above the floor (10 ->
        # ~700 on the collapses seen in the logs), so halving the RATIO converges in a fixed
        # handful of steps where halving the interval would crawl.  The loop keeps the
        # invariant ess(low) < target <= ess(high).
        for _ in range(_TEMPERATURE_SEARCH_STEPS):
            middle = math.sqrt(low * high)
            if ess_at(middle) < target:
                low = middle
            else:
                high = middle
        return high

    @staticmethod
    def stable_weights(costs: torch.Tensor, temperature: float) -> tuple[torch.Tensor, bool]:
        """Finite-only softmax; returns ``all_invalid=True`` when no sample survived."""
        finite = torch.isfinite(costs)
        weights = torch.zeros_like(costs)
        if not bool(torch.any(finite)):
            return weights, True
        finite_costs = costs[finite]
        minimum = torch.min(finite_costs)
        finite_weights = MPPICore._softmax_over_shifted(
            finite_costs - minimum, temperature)
        weights[finite] = finite_weights
        return weights, False

    def _shift_nominal_for_elapsed_time(
        self, controls: torch.Tensor, control_dt: float,
    ) -> torch.Tensor:
        """Advance a ``(T, 2)`` warm start by the measured loop interval.

        The rollout grid remains fixed at ``mppi_dt``.  A control interval that
        falls between two prediction knots is represented by linear
        interpolation, while samples beyond the previous horizon hold its last
        command.  The caller re-projects the result onto the current command and
        rate constraints before using it as the next nominal sequence.
        """
        if controls.shape != (self.config.horizon, 2):
            raise ValueError(
                "nominal controls must have shape "
                f"({self.config.horizon}, 2), got {tuple(controls.shape)}")
        elapsed = float(control_dt)
        if not math.isfinite(elapsed) or elapsed <= 0.0:
            raise ValueError("control_dt must be finite and positive")

        # Resample the previous command knots at t = control_dt + k*mppi_dt.
        knot_shift = elapsed / self.config.mppi_dt
        if knot_shift >= self.config.horizon - 1:
            return controls[-1:].expand_as(controls).clone()
        positions = (
            torch.arange(
                self.config.horizon,
                device=controls.device,
                dtype=controls.dtype,
            )
            + knot_shift
        )
        lower_unclamped = torch.floor(positions).to(torch.long)
        upper_unclamped = lower_unclamped + 1
        lower = torch.clamp(lower_unclamped, 0, self.config.horizon - 1)
        upper = torch.clamp(upper_unclamped, 0, self.config.horizon - 1)
        fraction = (positions - lower_unclamped.to(positions.dtype)).unsqueeze(-1)
        return torch.lerp(controls[lower], controls[upper], fraction)

    def _build_noise_filter(
        self,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Build the knot-axis noise kernel and its exact per-knot variance normalizer.

        Returns ``(kernel (2, 1, taps), gain (T,))``, or ``(None, None)`` at width 1 --
        which IS the historical i.i.d. white noise, so the filter is skipped entirely there
        rather than run as a one-tap identity that would only cost dispatch.

        The weights are the normalized binomial (Pascal) row of that width -- the discrete
        Gaussian -- symmetric, so the kernel has a centre knot and introduces no time shift.
        MPPIConfig.__post_init__ rejects an even width for exactly that reason.

        ``gain`` is EXACT rather than the interior ``sqrt(sum(c**2))``: the padded convolution
        is a linear map ``y = A x`` along the knot axis, filtering the identity basis gives
        ``A``, and ``Var(y_i) = ||A_i||^2`` for unit-variance ``x``.  It has to be per-knot
        because REPLICATE padding leaves the two edge knots with a larger variance than the
        interior (0.625 against 0.375 at width 3), and one of those edges is the horizon tail,
        the part of the plan this filter exists to help.  Dividing by it keeps the drawn
        noise at exactly noise_v / noise_w on EVERY knot.
        """
        taps = int(self.config.noise_smoothing_taps)
        if taps <= 1:
            return None, None
        horizon = int(self.config.horizon)
        # (taps,) binomial row, e.g. [0.25, 0.50, 0.25] at width 3.
        row = np.ones(1, dtype=np.float64)
        for _ in range(taps - 1):
            row = np.convolve(row, np.ones(2, dtype=np.float64))
        weights = torch.as_tensor(
            row / row.sum(), device=self.device, dtype=self.dtype)
        # (2, 1, taps): the SAME kernel for the linear and the angular channel, shaped for
        # conv1d(groups=2) so the two are filtered independently in one call.
        kernel = weights.reshape(1, 1, taps).repeat(2, 1, 1)
        padding = taps // 2
        # (T, 1, T) identity basis: sample j is the one-hot e_j along the knot axis, so
        # filtering it yields column j of A.
        basis = torch.eye(
            horizon, device=self.device, dtype=self.dtype).reshape(horizon, 1, horizon)
        columns = torch.nn.functional.conv1d(
            torch.nn.functional.pad(basis, (padding, padding), mode="replicate"),
            kernel[:1],
        ).squeeze(1)                                        # (T_j, T_i), == A[i, j]
        return kernel, torch.sqrt(torch.sum(columns ** 2, dim=0))

    def _draw_noise(
        self, horizon: int, sigma: torch.Tensor, *, generator: torch.Generator,
    ) -> torch.Tensor:
        """Return ``(K, T, 2)`` command noise whose per-knot std is ``sigma``.

        At noise_smoothing_taps == 1 this is the i.i.d. white noise the sampler has always
        drawn.  Above 1 the draw is CORRELATED ACROSS KNOTS, so one sample proposes a
        sustained manoeuvre rather than per-knot jitter that largely cancels over the
        rollout.  See MPPIConfig.noise_smoothing_taps for why the horizon tail needs that.

        ``sigma`` is the ``(2,)`` per-channel std; ``generator`` is passed in rather than read
        off the planner for the same reason _sample_and_score takes it, so the sample sequence
        stays an explicit input of the step.
        """
        white = torch.randn(
            (self.num_samples, horizon, 2),
            generator=generator, device=self.device, dtype=self.dtype)
        if self._noise_kernel is None or self._noise_gain is None:
            return white * sigma
        # The normalizer is built once for config.horizon.  A batch of a different horizon
        # would silently be scaled by the wrong per-knot gain, so name the assumption that
        # broke instead of broadcasting through it.
        if horizon != int(self._noise_gain.shape[0]):
            raise ValueError(
                f"noise smoothing is built for horizon {int(self._noise_gain.shape[0])} but "
                f"a batch of horizon {horizon} was requested; the per-knot variance "
                f"normalizer would not match")
        # (K,T,2) -> (K,2,T): conv1d filters the LAST axis, and the knot axis is the one that
        # has to be smoothed.  Replicate padding extends the edge knots by their own value
        # instead of pulling them toward zero; the gain above corrects the variance that
        # leaves them with.
        padding = int(self._noise_kernel.shape[-1]) // 2
        smoothed = torch.nn.functional.conv1d(
            torch.nn.functional.pad(
                white.permute(0, 2, 1), (padding, padding), mode="replicate"),
            self._noise_kernel, groups=2)
        return smoothed.permute(0, 2, 1) / self._noise_gain[None, :, None] * sigma




####################################### COMMAND PROJECTION AND ROLLOUT #######################################
# The causal pass that turns raw command targets into a feasible sequence and the
# states it produces.  The one thing a mode changes here is the rollout GEOMETRY, and
# that arrives as MPPIInput.reversed_geometry -- a bool, so it cannot be unknown.

    def _build_constraint_table(self) -> ConstraintTermTable:
        """Register the command constraints this layer owns.  Overrides MUST call super() first.

        MPPICore registers none, and that is not an oversight.  The two constraints it
        owns are both structural rather than registered, because the projection has to be
        able to tell them apart from the rectangle -- the rate box is DROPPED by the
        safety-first fallback, and the coupled simplex is NORMALIZED by the rectangle
        alone.  See ConstraintTermTable's docstring.  The caller's mode ceiling is the
        seed of the fold below, not a term, so that it survives the warm-start pass.

        TerrainMPPI appends the traversability lookup; that is the one thing this layer
        needs and cannot supply.
        """
        return ConstraintTermTable()

    def _command_limits(
        self, state: torch.Tensor, data: MPPIInput, use_terms: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ``(...)`` command rectangle ceilings in force at ``state``.

        ``state`` is ``(..., 5)`` and both returned tensors are ``(...)``.  The fold is:

            seed      the caller's mode ceiling, cmd_v_limit / cmd_w_limit
            tighten   elementwise minimum over every registered term, in table order
            floor     min(ceiling, conservative_speed_v/w)

        The FLOOR is core policy, not any one term's: a term may tighten the rectangle but
        may not deadlock the planner.  Without it a stamped 0.01 m/s cell would pin the
        command domain to zero and the robot could not drive out of it.  Independent
        ``conservative_speed_v/w`` values preserve useful linear and angular authority
        instead of deriving both axes from one linear-speed fraction.  It can never widen
        past the ceiling, because it is the ceiling's own minimum with the escape speed,
        which is also why it is inert when no term tightened anything.

        ``use_terms=False`` keeps the caller's ceiling and skips every registered term, for
        the warm-start pass whose result is only a cached fallback plan that is always
        re-projected -- with the terms -- before any command derived from it is issued.
        """
        global_v = max(0.0, float(data.cmd_v_limit))
        global_w = max(0.0, float(data.cmd_w_limit))
        template = state[..., 0]
        # (...,) seed: the mode ceiling, broadcast to the batch shape.
        vmax = torch.full_like(template, global_v)
        wmax = torch.full_like(template, global_w)
        if use_terms:
            for entry in self._constraint_table.entries:
                term_v, term_w = entry.fn(state, data)
                vmax = torch.minimum(vmax, term_v)
                wmax = torch.minimum(wmax, term_w)
        floor_v = min(global_v, float(self.config.conservative_speed_v))
        floor_w = min(global_w, float(self.config.conservative_speed_w))
        return (
            torch.clamp(vmax, min=floor_v),
            torch.clamp(wmax, min=floor_w),
        )

    def _project_and_rollout(
        self, controls: torch.Tensor, initial_state: torch.Tensor,
        data: MPPIInput, *, use_map_limits: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project command targets and integrate them in one causal pass.

        ``controls`` has shape ``(..., T, 2)`` and ``initial_state`` ``(5,)`` or
        anything broadcastable to ``(..., 5)``; the returned projected commands
        are ``(..., T, 2)`` and the predicted states ``(..., T, 5)``. Knot zero
        is constrained relative to the last issued command over ``control_dt``;
        future knots use the preceding projected command over fixed ``mppi_dt``.

        Command bounds and the coupled simplex are normalized by the map cell
        under the state that *enters* each interval, which is the only causal
        choice: the state after an interval depends on the command being
        projected. Projection and integration therefore interleave here instead
        of running as two separate passes.

        ``use_map_limits=False`` projects against the caller's limits only. It
        exists for the warm-start pass, whose output is a cached fallback plan
        that is always re-projected — with the map — before any command derived
        from it is issued; skipping the per-knot lookup there avoids a full
        batch-of-one lookup pass that is pure dispatch overhead.
        """
        controls = controls.clone()
        recovery = data.reversed_geometry
        # (..., 5) physical state entering the interval currently projected.
        state = torch.broadcast_to(
            initial_state.to(device=controls.device, dtype=controls.dtype),
            controls.shape[:-2] + (5,),
        ).clone()

        anchor = self._last_issued_command
        previous_v = torch.full_like(
            controls[..., 0, 0], float(anchor[0]))
        previous_w = torch.full_like(
            controls[..., 0, 1], float(anchor[1]))
        projected = []
        states = []
        for index in range(controls.shape[-2]):
            dt = float(data.control_dt) if index == 0 else self.config.mppi_dt
            # (...,) per-sample command domain of the cell being occupied now.
            vmax, wmax = self._command_limits(state, data, use_map_limits)
            # The admissible command set at THIS knot -- rate box, cell rectangle and coupled
            # simplex -- built once and projected onto.  Built from the
            # SAME type solve() builds the knot-zero feasibility report from, so the
            # projection and the warning about it cannot drift apart.
            constraint = CommandConstraint.build(
                vmax, wmax, previous_v, previous_w, dt, self.config,
                reversed_geometry=recovery)
            v_now, w_now = constraint.project(
                controls[..., index, 0], controls[..., index, 1])

            command = torch.stack((v_now, w_now), dim=-1)
            projected.append(command)
            # The rollout grid stays at mppi_dt for every knot, including the
            # first one whose rate bound used the measured control_dt.
            state = step_actuator_lag(
                state, command, self.config.mppi_dt, self.config.lag_time)
            states.append(state)
            previous_v, previous_w = v_now, w_now
        return torch.stack(projected, dim=-2), torch.stack(states, dim=-2)




####################################### CONTROL COSTS #######################################
# The two cost terms that read only the command sequence, so they need neither a
# map nor a reference path.  Registered here; every layer above appends its own.

    def _build_cost_table(self) -> CostTermTable:
        """Register the terms this layer owns.  Overrides MUST call super() first.

        Called once from __init__.  The composed order is the order the terms are summed
        and the order the trace prints them, so appending after super() is what makes the
        objective read bottom layer up: core, terrain, tracking, task.
        """
        table = CostTermTable()
        table.add("CtrlEffort", self.config.weight_CtrlEffort, fn=self._term_CtrlEffort)
        table.add("Accel", self.config.weight_Accel, fn=self._term_Accel)
        return table


    def _term_CtrlEffort(self, ctx: CostContext) -> torch.Tensor:
        """Return ``(K,)`` UNWEIGHTED running effort for ``(K,T,2)`` commands.

        Linear and angular commands are normalized by their positive current
        limits, squared, and time-averaged, so the result is dimensionless
        before ``weight_CtrlEffort``, which the manager applies.  MPPIProblem
        validates both denominators.
        """
        effort = (
            (ctx.controls[..., 0] / float(ctx.data.cmd_v_limit)) ** 2
            + (ctx.controls[..., 1] / float(ctx.data.cmd_w_limit)) ** 2
        )
        return torch.mean(effort, dim=-1)

    def _term_Accel(self, ctx: CostContext) -> torch.Tensor:
        """Return ``(K,)`` UNWEIGHTED running rate cost for ``(K,T,2)`` commands.

        Knot zero uses ``control_dt`` relative to the last issued command;
        future knots use the fixed MPPI interval. Linear acceleration is
        normalized by ``lin_acc`` in BOTH directions -- braking and accelerating
        share one symmetric rate bound, so the sign of dv/dt does not change the
        denominator -- and angular acceleration uses ``ang_acc``. Squared
        normalized rates are dimensionless and time-averaged. Configuration and
        ``control_dt`` validation reject non-positive denominators before division.
        """
        controls = ctx.controls
        control_dt = float(ctx.data.control_dt)
        if not math.isfinite(control_dt) or control_dt <= 0.0:
            raise ValueError("control_dt must be finite and positive")
        previous = self._last_issued_command.to(
            device=controls.device, dtype=controls.dtype)
        first_acceleration = (controls[..., :1, :] - previous) / control_dt
        future_acceleration = torch.diff(controls, dim=-2) / self.config.mppi_dt
        acceleration = torch.cat((first_acceleration, future_acceleration), dim=-2)
        linear = acceleration[..., 0]
        normalized_linear = linear / self.config.lin_acc
        normalized_angular = acceleration[..., 1] / self.config.ang_acc
        normalized_sq = normalized_linear ** 2 + normalized_angular ** 2
        return torch.mean(normalized_sq, dim=-1)



####################################### VALIDITY #######################################

    def _build_validity_table(self) -> ValidityTermTable:
        """Register the vetoes this layer owns.  Overrides MUST call super() first.

        Called once from __init__, exactly like _build_cost_table.  These two used to be
        three hardcoded lines inside CostManager.evaluate, owned by nobody and reported
        under no name; the third of them, the total-cost finiteness test, cannot be a
        context-only predicate and is folded in by screen() under the reserved
        FiniteCost key instead.
        """
        table = ValidityTermTable()
        table.add("FiniteState", fn=self._validity_FiniteState)
        table.add("FiniteControl", fn=self._validity_FiniteControl)
        return table

    def _validity_FiniteState(self, ctx: CostContext) -> torch.Tensor:
        """(K,) True where every knot of the ``(K,T,5)`` rollout is finite.

        A non-finite state means the integration diverged, so nothing measured on that
        rollout -- including its cost -- describes a plan that could be executed.
        """
        return torch.all(torch.isfinite(ctx.states), dim=(-2, -1))

    def _validity_FiniteControl(self, ctx: CostContext) -> torch.Tensor:
        """(K,) True where every knot of the ``(K,T,2)`` command sequence is finite.

        Projection cannot produce a non-finite command from finite bounds, so this fires
        only when a bound itself was non-finite; it is the cheap structural check that
        keeps such a sequence from being issued.
        """
        return torch.all(torch.isfinite(ctx.controls), dim=(-2, -1))


####################################### CUDA OUT-OF-MEMORY FALLBACK #######################################
    # ------------------------------------------------------------------
    # CUDA out-of-memory fallback
    #
    # The GPU can be saturated by other processes on a shared machine.  Without
    # a fallback every tick re-raises OOM and the robot holds position forever,
    # because the failure path used to drop the map cache and then have to
    # re-upload it (the largest single allocation of a tick) on a full GPU.
    # Instead we migrate the planner to CPU once and keep planning at the
    # reduced cpu_num_samples rollout count.  The migration is deliberately
    # one-way: re-probing CUDA from the control loop would risk OOM churn.
    #
    # Owned here rather than in TerrainMPPI because only three of these lines are
    # about the map: device, num_samples, both generators, _nominal, _last_result
    # and the latch itself are all core state.  Layers with device-resident tensors
    # of their own move them through the _migrate_device_caches() hook below.
    # ------------------------------------------------------------------

    @property
    def cuda_fallback_active(self) -> bool:
        """True once a CUDA OOM has permanently moved this planner to the CPU."""
        return self._cuda_fallback_active

    def _migrate_device_caches(self, target: torch.device) -> Optional[str]:
        """Hook: move this layer's device-resident tensors to ``target``.

        Called by _fallback_to_cpu() AFTER ``self.device`` has been switched to ``target``,
        so an override may both copy tensors (``.to(target)`` works from either device) and
        rebuild config-derived ones off ``self.device``.  Overrides must call super() first
        and return a human-readable note if they had to DROP a cache instead of moving it,
        or None on success.  MPPICore has no layer-local device cache beyond the ones
        _fallback_to_cpu() moves itself, so this default does nothing.
        """
        return None

    def _fallback_to_cpu(self, exc: BaseException) -> str:
        """Migrate cached planner tensors from CUDA to CPU after an OOM.

        Device-to-host copies reuse memory that is already allocated on the GPU,
        so they are safe to perform while the device is full.  Preserving the
        traversability-map snapshot this way means the next tick does not have
        to re-read it from the shared global map.

        Returns a human-readable reason for the caller to surface as a warning.
        """
        detail = str(exc).split("\n", 1)[0].strip()
        if self.device.type != "cuda":
            # Already on CPU: this is host memory pressure, nowhere left to fall
            # back to, so report it plainly instead of pretending to recover.
            return f"{self.device.type} out of memory: {detail}"

        # Cached device tensor owned by this layer: the nominal control sequence
        # (horizon, 2).  Layer-local caches (the map snapshot, the footprint stencil)
        # are handled by the _migrate_device_caches() overrides below.
        moved_nominal: Optional[torch.Tensor] = None
        migration_error: Optional[str] = None
        try:
            if self._nominal is not None:
                moved_nominal = self._nominal.detach().to("cpu")
        except RuntimeError as move_exc:  # includes torch.cuda.OutOfMemoryError
            # Could not copy the cache off the device; drop it so the next tick
            # rebuilds it on CPU instead of reusing a tensor that still lives on
            # the exhausted GPU.
            migration_error = str(move_exc).split("\n", 1)[0].strip()
            moved_nominal = None

        self.device = torch.device("cpu")
        self.num_samples = self.config.cpu_num_samples
        # torch.Generator is bound to a device, so it must be rebuilt.  Reseed
        # from the config to keep sampling reproducible across the migration.
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(int(self.config.seed))
        # Config derived, so REBUILT on the new device rather than copied: there is nothing
        # in it to preserve, and a rebuild cannot fail the way a device-to-host copy can.
        self._noise_kernel, self._noise_gain = self._build_noise_filter()
        self._nominal = moved_nominal
        self._last_issued_command = self._last_issued_command.detach().to(self.device)
        # Layer-local device tensors (the map snapshot, the footprint stencil) move last,
        # once self.device already names the new device.
        layer_error = self._migrate_device_caches(self.device)
        self._last_result = None
        self._cuda_fallback_active = True
        # Release the vacated CUDA blocks back to the driver so the process that
        # is actually starved for memory can use them.
        torch.cuda.empty_cache()

        reason = (
            f"CUDA out of memory, falling back to CPU MPPI "
            f"({self.num_samples} samples): {detail}"
        )
        if migration_error is not None:
            reason += f"; nominal plan dropped (device-to-host copy failed: {migration_error})"
        if layer_error is not None:
            reason += f"; {layer_error}"
        return reason




####################################### RESULT HELPERS #######################################

    def _invalid_result(self, start: float, reason: str) -> MPPIResult:
        self.record_issued_command(0.0, 0.0)
        return MPPIResult(
            0.0, 0.0, False, time.perf_counter() - start,
            None, float("inf"), reason,
        )




####################################### NULL VISUALIZATION SINK #######################################

class _NullVisualizer:
    """Inert visualization sink used when nothing is attached.

    MPPIPlanner always holds a visualizer, so publish_plan() and reset_visualization()
    stay unconditional call sites.  This keeps the planner importable and runnable
    with no ROS, no display and no visualization dependency at all; the live
    implementation is mppi_rviz.MPPIMarkerPublisher, injected by main.py.
    """

    def reset_history(self) -> None:
        return

    def update(self, *args: Any, **kwargs: Any) -> None:
        return
