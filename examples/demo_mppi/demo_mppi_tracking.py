"""ROS-free closed-loop plant for the isolated MPPI shared-map demo."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path as FilePath
import sys
from typing import Optional

import numpy as np
import torch

DEMO_DIRECTORY = FilePath(__file__).resolve().parent
TRAVERSABILITY_DIRECTORY = DEMO_DIRECTORY.parent
PROJECT_ROOT = TRAVERSABILITY_DIRECTORY.parent
sys.path.insert(0, str(TRAVERSABILITY_DIRECTORY))
sys.path.insert(0, str(PROJECT_ROOT))

from statenav_mppi.mppi_core.mppi_dynamics import rollout_actuator_lag
from statenav_mppi.mppi_utils.mppi_path_geometry import project_point_to_polyline, sanitize_path
from statenav_mppi.mppi_planners.mppi_planner import MPPIPlanner


def _wrap_to_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def path_progress_ratio(path: np.ndarray, xy: np.ndarray) -> float:
    """Return clamped global-path progress from measured polyline arc length."""
    projection = project_point_to_polyline(path, xy)
    total_length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    return float(np.clip(projection.arc_length / total_length, 0.0, 1.0))


def _settling_time_from_response(control_period: float, response: float) -> float:
    """Convert one-tick response into the equivalent 95%-settling time."""
    if response >= 1.0:
        return 0.0
    tau = -float(control_period) / math.log1p(-float(response))
    return tau * -math.log(0.05)


def cross_track_error(path: np.ndarray, xy: np.ndarray) -> float:
    segment_start = path[:-1]
    segment = np.diff(path, axis=0)
    denominator = np.sum(segment * segment, axis=1)
    fraction = np.sum((np.asarray(xy)[None, :] - segment_start) * segment, axis=1)
    fraction = np.clip(fraction / np.maximum(denominator, 1.0e-12), 0.0, 1.0)
    projection = segment_start + fraction[:, None] * segment
    return float(np.sqrt(np.min(np.sum(
        (projection - np.asarray(xy)[None, :]) ** 2, axis=1))))


@dataclass(frozen=True)
class ClosedLoopTrackingConfig:
    control_period: float = 0.2
    max_v: float = 2.0
    max_w: float = 1.0
    plant_response: float = 0.72
    goal_tolerance_m: float = 0.32
    trail_length: int = 500

    def __post_init__(self) -> None:
        for name in ("control_period", "max_v", "max_w", "goal_tolerance_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 < self.plant_response <= 1.0:
            raise ValueError("plant_response must be in (0, 1]")
        if self.trail_length <= 1:
            raise ValueError("trail_length must be greater than one")


@dataclass(frozen=True)
class ClosedLoopTrackingStep:
    pose: np.ndarray
    velocity: np.ndarray
    command: np.ndarray
    target: np.ndarray
    warnings: tuple[str, ...]
    diagnostic: Optional[str]
    source: str
    cross_track_m: float
    path_progress: float
    goal_distance_m: float
    goal_reached: bool


class ClosedLoopMPPITracker:
    """Advance an isolated first-order unicycle with production MPPI commands."""

    def __init__(
        self,
        planner: MPPIPlanner,
        tracking_config: ClosedLoopTrackingConfig,
    ) -> None:
        self.planner = planner
        self.config = tracking_config
        self.path = np.empty((0, 2), dtype=np.float32)
        self.pose = np.zeros(3, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self._initialized = False
        self._goal_reached = False
        self._progress = 0.0
        self._trail: list[np.ndarray] = []
        self._targets: list[np.ndarray] = []
        self._commands: list[np.ndarray] = []
        self._errors: list[float] = []
        self._solve_ms: list[float] = []
        self.mppi_steps = 0
        self.held_steps = 0

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def goal_reached(self) -> bool:
        return self._goal_reached

    @property
    def trail(self) -> np.ndarray:
        return np.asarray(self._trail, dtype=np.float32).reshape(-1, 2)

    @property
    def targets(self) -> np.ndarray:
        return np.asarray(self._targets, dtype=np.float32).reshape(-1, 2)

    @property
    def commands(self) -> np.ndarray:
        return np.asarray(self._commands, dtype=np.float32).reshape(-1, 2)

    @property
    def errors(self) -> np.ndarray:
        return np.asarray(self._errors, dtype=np.float32)

    @property
    def solve_ms(self) -> np.ndarray:
        return np.asarray(self._solve_ms, dtype=np.float32)

    def _initialize_at_path_start(self) -> None:
        tangent = self.path[1] - self.path[0]
        self.pose = np.array([
            self.path[0, 0], self.path[0, 1],
            math.atan2(float(tangent[1]), float(tangent[0])),
        ], dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self._trail = [self.pose[:2].copy()]
        self._targets = []
        self._commands = []
        self._errors = []
        self._solve_ms = []
        self._progress = 0.0
        self._goal_reached = False
        self.mppi_steps = 0
        self.held_steps = 0

    def set_path(self, path: np.ndarray, *, reset_pose: bool = False) -> bool:
        """Install a path, preserving simulated state across ordinary replans."""
        clean = sanitize_path(path)
        unchanged = self.path.shape == clean.shape and np.allclose(self.path, clean)
        if unchanged and not reset_pose:
            return False
        first_path = not self._initialized
        self.path = clean
        if first_path or reset_pose:
            self.planner.reset()
            self._initialize_at_path_start()
        else:
            self.planner.reset_plan()
        self._initialized = True
        self._goal_reached = False
        return True

    def reset(self) -> None:
        if self.path.shape[0] < 2:
            return
        self.planner.reset()
        self._initialize_at_path_start()
        self._initialized = True

    def _current_limits(self) -> tuple[float, float, Optional[str]]:
        global_map = self.planner.global_map
        if global_map is not None and hasattr(
            global_map, "is_TraversabilityMap_built",
        ) and not bool(global_map.is_TraversabilityMap_built):
            return 0.0, 0.0, "traversability map is not built"

        # The cell under the simulated pose, through the planner's own travgraph lookup: the
        # map's get_cmd_limits() would read the dense TraversabilityMap, which the MPPI never does.
        scalar = lambda value: torch.tensor(  # noqa: E731
            [value], device=self.planner.device, dtype=self.planner.dtype)
        try:
            with torch.no_grad():
                safe_v, safe_w = self.planner.lookup_map(
                    scalar(float(self.pose[0])),
                    scalar(float(self.pose[1])),
                    scalar(float(self.pose[2])),
                )
        except RuntimeError as exc:
            return 0.0, 0.0, str(exc)
        # lookup_map no longer reports cell acceptance, so this fallback applies
        # the same usability rule the planner keeps: a limit must be finite and
        # positive to cap a command.  Small positive limits (too-risky stamps)
        # are honoured as slow commands instead of holding position.
        limit_v = float(safe_v[0].detach().cpu())
        limit_w = float(safe_w[0].detach().cpu())
        if (
            not math.isfinite(limit_v) or not math.isfinite(limit_w)
            or limit_v <= 0.0 or limit_w <= 0.0
        ):
            return 0.0, 0.0, (
                f"current simulated pose has no usable map command limit "
                f"(safe_v={limit_v}, safe_w={limit_w})"
            )
        return (
            min(self.config.max_v, limit_v),
            min(self.config.max_w, limit_w),
            None,
        )

    def _stationary_step(
        self,
        source: str,
        warnings: tuple[str, ...] = (),
        target: Optional[np.ndarray] = None,
        diagnostic: Optional[str] = None,
    ) -> ClosedLoopTrackingStep:
        stationary_target = (
            self.path[-1].copy() if target is None
            else np.asarray(target, dtype=np.float32).reshape(2).copy())
        error = cross_track_error(self.path, self.pose[:2])
        distance = float(np.linalg.norm(self.pose[:2] - self.path[-1]))
        return ClosedLoopTrackingStep(
            self.pose.copy(), self.velocity.copy(),
            np.zeros(2, dtype=np.float32), stationary_target, warnings, diagnostic,
            source, error, self._progress, distance, self._goal_reached,
        )

    def hold(
        self,
        source: str,
        warnings: tuple[str, ...] = (),
        *,
        target: Optional[np.ndarray] = None,
        diagnostic: Optional[str] = None,
        solve_ms: float = 0.0,
    ) -> ClosedLoopTrackingStep:
        """Stop the isolated plant and record a retryable held tick."""
        if not self._initialized:
            raise RuntimeError("waiting for a valid global path")
        self.velocity[:] = 0.0
        self.planner.record_issued_command(0.0, 0.0)
        self.held_steps += 1
        return self._record(self._stationary_step(
            source, warnings, target, diagnostic), solve_ms=solve_ms)

    def advance(self) -> ClosedLoopTrackingStep:
        if not self._initialized:
            raise RuntimeError("waiting for a valid global path")
        if self._goal_reached:
            self.velocity[:] = 0.0
            self.planner.record_issued_command(0.0, 0.0)
            step = self._stationary_step("hold(goal_reached)")
            if not self._commands or not np.allclose(self._commands[-1], 0.0):
                return self._record(step)
            return step

        cmd_v_limit, cmd_w_limit, map_warning = self._current_limits()
        if map_warning is not None:
            return self.hold("hold(map_not_ready)", (map_warning,))

        result = self.planner.track_path(
            float(self.pose[0]), float(self.pose[1]), float(self.pose[2]),
            float(self.velocity[0]), float(self.velocity[1]),
            self.config.control_period, self.path,
        )
        target = self.path[-1].copy() if result.waypoint is None else np.asarray(
            result.waypoint, dtype=np.float32).reshape(2)
        command = np.array([result.v_cmd, result.w_cmd], dtype=np.float32)
        solve_ms = 0.0
        if result.diagnostic is not None and "solve=" in result.diagnostic:
            try:
                solve_ms = float(result.diagnostic.split("solve=", 1)[1].split()[0])
            except (ValueError, IndexError):
                solve_ms = 0.0
        if not result.valid:
            return self.hold(
                "hold(mppi)", result.warnings, target=target,
                diagnostic=result.diagnostic, solve_ms=solve_ms)
        source = "mppi"
        self.mppi_steps += 1

        dt = self.config.control_period
        initial_state = torch.as_tensor(
            np.concatenate((self.pose, self.velocity)),
            device=self.planner.device,
            dtype=self.planner.dtype,
        )
        controls = torch.as_tensor(
            command[None, :],
            device=self.planner.device,
            dtype=self.planner.dtype,
        )
        lag_time = _settling_time_from_response(
            dt, self.config.plant_response)
        with torch.no_grad():
            next_state = rollout_actuator_lag(
                initial_state, controls, dt, lag_time)[0]
        next_state_np = next_state.detach().cpu().numpy()
        self.pose[:] = next_state_np[:3]
        self.velocity[:] = next_state_np[3:]

        self._progress = max(
            self._progress, path_progress_ratio(self.path, self.pose[:2]))
        error = cross_track_error(self.path, self.pose[:2])
        goal_distance = float(np.linalg.norm(self.pose[:2] - self.path[-1]))
        self._goal_reached = goal_distance <= self.config.goal_tolerance_m
        if self._goal_reached:
            source = "mppi(goal_reached)"

        return self._record(ClosedLoopTrackingStep(
            self.pose.copy(), self.velocity.copy(), command, target.copy(),
            result.warnings, result.diagnostic, source, error,
            self._progress, goal_distance, self._goal_reached,
        ), solve_ms=solve_ms)

    def _record(
        self, step: ClosedLoopTrackingStep, *, solve_ms: float = 0.0,
    ) -> ClosedLoopTrackingStep:
        self._trail.append(step.pose[:2].copy())
        if len(self._trail) > self.config.trail_length:
            self._trail = self._trail[-self.config.trail_length:]
        self._targets.append(step.target.copy())
        self._commands.append(step.command.copy())
        self._errors.append(step.cross_track_m)
        self._solve_ms.append(float(solve_ms))
        return step

    def metrics(self) -> dict[str, object]:
        errors = self.errors
        solve_ms = self.solve_ms
        commands = self.commands
        command_boundaries = np.vstack((
            np.zeros((1, 2), dtype=np.float32), commands))
        if (
            self._goal_reached and commands.shape[0] > 0
            and not np.allclose(commands[-1], 0.0)
        ):
            command_boundaries = np.vstack((
                command_boundaries,
                np.zeros((1, 2), dtype=np.float32),
            ))
        command_rate = (
            np.diff(command_boundaries, axis=0) / self.config.control_period)
        goal_distance = (
            float(np.linalg.norm(self.pose[:2] - self.path[-1]))
            if self.path.shape[0] else float("inf")
        )
        return {
            "device": self.planner.device.type,
            "samples": self.planner.num_samples,
            "horizon": self.planner.config.horizon,
            "control_period_s": self.config.control_period,
            "steps": int(errors.size),
            "mppi_steps": self.mppi_steps,
            "held_steps": self.held_steps,
            "path_progress": self._progress,
            "rms_cross_track_m": (
                float(np.sqrt(np.mean(errors * errors)))
                if errors.size else float("inf")),
            "max_cross_track_m": (
                float(np.max(errors)) if errors.size else float("inf")),
            "final_goal_distance_m": goal_distance,
            "solve_median_ms": (
                float(np.median(solve_ms)) if solve_ms.size else 0.0),
            "solve_p95_ms": (
                float(np.percentile(solve_ms, 95)) if solve_ms.size else 0.0),
            "solve_max_ms": (
                float(np.max(solve_ms)) if solve_ms.size else 0.0),
            "command_rate_rms": (
                float(np.sqrt(np.mean(command_rate * command_rate)))
                if command_rate.size else 0.0),
            "goal_reached": self._goal_reached,
        }
