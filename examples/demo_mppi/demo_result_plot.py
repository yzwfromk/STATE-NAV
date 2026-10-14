"""Save a compact, post-run summary figure for the isolated MPPI demo."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np


def _xy(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.empty((0, 2), dtype=float)
    array = array.reshape(-1, array.shape[-1])
    if array.shape[1] < 2:
        raise ValueError("path and trail points must have at least two coordinates")
    return array[:, :2]


def _series(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _plot_terrain(axis, terrain_snapshot: Optional[dict[str, Any]]) -> None:
    if not terrain_snapshot:
        return
    # (rows, cols) m/s cmd_v channel of the heading layer the snapshot was taken for.
    trav = terrain_snapshot.get("trav")
    geometry = terrain_snapshot.get("geometry")
    if trav is None or geometry is None:
        return
    image = np.asarray(trav, dtype=float)
    image = np.squeeze(image)
    if image.ndim > 2:
        image = image[0]
    if image.ndim != 2:
        return
    image = np.flip(image.T, axis=(0, 1))
    try:
        xmin, xmax, ymin, ymax = (float(value) for value in geometry[:4])
    except (TypeError, ValueError, IndexError):
        return
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return
    axis.imshow(
        image,
        extent=(xmin, xmax, ymin, ymax),
        origin="lower",
        cmap="coolwarm_r",
        alpha=0.55,
        aspect="equal",
    )


def save_demo_result_plot(
    *,
    output_path: Path,
    path: Any,
    trail: Any,
    commands: Any,
    errors: Any,
    solve_ms: Any,
    control_period: float,
    goal_reached: bool,
    terrain_snapshot: Optional[dict[str, Any]] = None,
) -> Path:
    """Write the final path, tracking error, and applied-speed summary.

    The plotting dependency is imported lazily so normal demo operation does
    not depend on a GUI backend and plotting failures can be handled by the
    caller at goal time.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path_xy = _xy(path)
    trail_xy = _xy(trail)
    command_array = np.asarray(commands, dtype=float)
    if command_array.size == 0:
        command_array = np.empty((0, 2), dtype=float)
    command_array = command_array.reshape(-1, command_array.shape[-1])
    if command_array.shape[1] < 2:
        raise ValueError("commands must contain linear and angular velocity")
    command_array = command_array[:, :2]
    error_array = _series(errors)
    solve_array = _series(solve_ms)
    periods = float(control_period)
    if not np.isfinite(periods) or periods <= 0.0:
        raise ValueError("control_period must be finite and positive")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(22.95, 11.56), dpi=100)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.45, 1.0), hspace=0.28)
    map_axis = figure.add_subplot(grid[:, 0])
    error_axis = figure.add_subplot(grid[0, 1])
    speed_axis = figure.add_subplot(grid[1, 1])

    _plot_terrain(map_axis, terrain_snapshot)
    if path_xy.size:
        map_axis.plot(path_xy[:, 0], path_xy[:, 1], "--", color="#1565c0", lw=2.2,
                      label="global path")
        map_axis.scatter(*path_xy[0], color="#2e7d32", s=70, zorder=4, label="start")
        map_axis.scatter(*path_xy[-1], color="#c62828", s=70, zorder=4, label="goal")
    if trail_xy.size:
        map_axis.plot(trail_xy[:, 0], trail_xy[:, 1], color="#ef6c00", lw=2.5,
                      label="robot trail")
        map_axis.scatter(*trail_xy[-1], color="#ef6c00", s=55, zorder=5, label="final pose")
    map_axis.set_title("MPPI global-path tracking")
    map_axis.set_xlabel("x [m]")
    map_axis.set_ylabel("y [m]")
    map_axis.grid(True, alpha=0.25)
    map_axis.set_aspect("equal", adjustable="datalim")
    handles, labels = map_axis.get_legend_handles_labels()
    if handles:
        map_axis.legend(loc="best")

    count = max(error_array.size, solve_array.size)
    time = np.arange(count, dtype=float) * periods
    if error_array.size:
        error_axis.plot(np.arange(error_array.size) * periods, error_array,
                        color="#6a1b9a", lw=2, label="cross-track error")
    error_axis.set_ylabel("error [m]", color="#6a1b9a")
    error_axis.tick_params(axis="y", labelcolor="#6a1b9a")
    error_axis.grid(True, alpha=0.25)
    solve_axis = error_axis.twinx()
    if solve_array.size:
        solve_axis.plot(np.arange(solve_array.size) * periods, solve_array,
                        color="#00838f", lw=1.8, label="MPPI solve time")
    solve_axis.set_ylabel("solve time [ms]", color="#00838f")
    solve_axis.tick_params(axis="y", labelcolor="#00838f")
    error_axis.set_title("Tracking error and MPPI solve time")
    error_axis.set_xlabel("time [s]")
    if count:
        error_axis.set_xlim(0.0, max(float(time[-1]), periods))

    command_time = np.arange(command_array.shape[0], dtype=float) * periods
    if command_array.shape[0]:
        speed_axis.plot(command_time, command_array[:, 0], color="#1b5e20", lw=2,
                        label="linear v")
        speed_axis.plot(command_time, command_array[:, 1], color="#e65100", lw=2,
                        label="angular w")
    speed_axis.set_title("Applied command")
    speed_axis.set_xlabel("time [s]")
    speed_axis.set_ylabel("velocity")
    speed_axis.grid(True, alpha=0.25)
    speed_axis.legend(loc="best")

    status = "goal reached" if goal_reached else "goal not reached"
    figure.suptitle(f"MPPI demo result — {status}", fontsize=16)
    figure.savefig(output, dpi=100)
    plt.close(figure)
    return output
