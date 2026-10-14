"""Common first-order actuator dynamics for batched MPPI rollouts."""

from __future__ import annotations

import math

import torch


_TWO_PI = 2.0 * math.pi
_SETTLING_REMAINDER = 0.05


def _lag_coefficients(mppi_dt: float, lag_time: float) -> tuple[float, float]:
    step = float(mppi_dt)
    settling = float(lag_time)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("mppi_dt must be finite and positive")
    if not math.isfinite(settling) or settling < 0.0:
        raise ValueError("mppi_lag_time must be finite and non-negative")
    if settling == 0.0:
        return 1.0, 0.0
    tau = settling / -math.log(_SETTLING_REMAINDER)
    beta = -math.expm1(-step / tau)
    interval_initial_weight = (tau / step) * beta
    return beta, interval_initial_weight


def lag_response_factor(mppi_dt: float, lag_time: float) -> float:
    """Return the exact discrete response for a 95%-settling-time parameter.

    ``mppi_dt`` and ``lag_time`` are seconds. ``lag_time == 0`` selects an
    instantaneous response without evaluating a zero time constant.
    """
    beta, _ = _lag_coefficients(mppi_dt, lag_time)
    return beta


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, _TWO_PI) - math.pi


def step_actuator_lag(
    state: torch.Tensor,
    command: torch.Tensor,
    mppi_dt: float,
    lag_time: float,
) -> torch.Tensor:
    """Advance one ``(..., 5)`` state by one ``(..., 2)`` command interval.

    This is the single interval that ``rollout_actuator_lag`` repeats; it is
    exposed so a caller that must interleave per-knot command projection with
    integration (the map-limited simplex projection) reuses this integrator
    instead of duplicating it.  ``state`` is ``[x, y, yaw, v_actual, w_actual]``
    and ``command`` is ``[v_cmd, w_cmd]``; both are trusted finite here because
    the batched entry point above validates them.
    """
    step = float(mppi_dt)
    beta, interval_initial_weight = _lag_coefficients(step, lag_time)
    v_actual = state[..., 3]
    w_actual = state[..., 4]
    v_command = command[..., 0]
    w_command = command[..., 1]
    v_next = v_actual + beta * (v_command - v_actual)
    w_next = w_actual + beta * (w_command - w_actual)
    v_average = (
        v_command + interval_initial_weight * (v_actual - v_command))
    w_average = (
        w_command + interval_initial_weight * (w_actual - w_command))
    yaw_delta = w_average * step
    midpoint_yaw = state[..., 2] + 0.5 * yaw_delta
    x_next = state[..., 0] + v_average * torch.cos(midpoint_yaw) * step
    y_next = state[..., 1] + v_average * torch.sin(midpoint_yaw) * step
    yaw_next = _wrap_angle(state[..., 2] + yaw_delta)
    return torch.stack(
        (x_next, y_next, yaw_next, v_next, w_next), dim=-1,
    )


def rollout_actuator_lag(
    initial_state: torch.Tensor,
    controls: torch.Tensor,
    mppi_dt: float,
    lag_time: float,
    *,
    validate_finite: bool = True,
) -> torch.Tensor:
    """Roll out command targets through common lag into physical state.

    Args:
        initial_state: Physical state shaped ``(..., 5)`` as
            ``[x, y, yaw, v_actual, w_actual]``.
        controls: Command targets shaped ``(..., T, 2)`` as
            ``[v_cmd, w_cmd]``.
        mppi_dt: Fixed interval between predicted knots in seconds.
        lag_time: Time in seconds to reach 95% of a constant step target.

    Returns:
        Predicted physical states shaped ``(..., T, 5)``. Pose uses the
        interval-average actual velocities and midpoint heading.
    """
    if controls.ndim < 2 or controls.shape[-1] != 2 or controls.shape[-2] <= 0:
        raise ValueError("controls must have shape (..., T, 2) with T > 0")
    if initial_state.ndim < 1 or initial_state.shape[-1] != 5:
        raise ValueError("initial_state must have shape (..., 5)")
    if validate_finite:
        if not bool(torch.all(torch.isfinite(initial_state))):
            raise ValueError("initial_state must be finite")
        if not bool(torch.all(torch.isfinite(controls))):
            raise ValueError("controls must be finite")

    step = float(mppi_dt)
    batch_shape = controls.shape[:-2]
    state = initial_state.to(device=controls.device, dtype=controls.dtype)
    try:
        state = torch.broadcast_to(state, batch_shape + (5,)).clone()
    except RuntimeError as exc:
        raise ValueError(
            "initial_state batch shape must broadcast to controls"
        ) from exc

    states = []
    for index in range(controls.shape[-2]):
        state = step_actuator_lag(
            state, controls[..., index, :], step, lag_time)
        states.append(state)
    return torch.stack(states, dim=-2)
