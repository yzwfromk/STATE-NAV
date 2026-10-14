"""ROS-free synthetic global paths for isolated MPPI debugging."""

from __future__ import annotations

import numpy as np


_BASE_POLYLINES = {
    "straight": np.array([[0, 0], [8, 0]], dtype=np.float64),
    "right_angle": np.array(
        [[0, 0], [4, 0], [4, 4]], dtype=np.float64),
    "detour": np.array(
        [[0, 0], [2, 0], [2, 3], [6, 3], [6, 0], [8, 0]],
        dtype=np.float64,
    ),
}


def _resample_polyline(
    points: np.ndarray,
    spacing: float,
    *,
    retain_vertices: bool,
) -> np.ndarray:
    """Resample a known-valid polyline by measured arc length."""
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    arc = np.arange(0.0, cumulative[-1], spacing, dtype=np.float64)
    additions = cumulative if retain_vertices else np.array([cumulative[-1]])
    arc = np.unique(np.concatenate((arc, additions, [cumulative[-1]])))

    segment = np.searchsorted(cumulative, arc, side="right") - 1
    segment = np.clip(segment, 0, len(segment_lengths) - 1)
    fraction = (arc - cumulative[segment]) / segment_lengths[segment]
    return points[segment] + fraction[:, None] * (
        points[segment + 1] - points[segment])


def generate_path_scenario(
    path_type: str,
    spacing: float = 0.25,
    scale: float = 1.0,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> np.ndarray:
    """Return one finite, approximately uniform synthetic path in world xy."""
    name = str(path_type).strip()
    if name not in (*_BASE_POLYLINES, "s_curve"):
        raise ValueError(
            f"unknown path_type {name!r}; expected straight, s_curve, "
            "right_angle, or detour")

    scalars = {
        "spacing": float(spacing),
        "scale": float(scale),
        "origin_x": float(origin_x),
        "origin_y": float(origin_y),
    }
    if not all(np.isfinite(value) for value in scalars.values()):
        raise ValueError("path geometry parameters must be finite")
    if scalars["spacing"] <= 0.0:
        raise ValueError("spacing must be positive")
    if scalars["scale"] <= 0.0:
        raise ValueError("scale must be positive")

    retain_vertices = name != "s_curve"
    if retain_vertices:
        base = _BASE_POLYLINES[name].copy()
    else:
        parameter = np.linspace(0.0, 1.0, 1001, dtype=np.float64)
        base = np.column_stack((
            8.0 * parameter,
            1.5 * np.sin(2.0 * np.pi * parameter),
        ))

    scaled = base * scalars["scale"]
    sampled = _resample_polyline(
        scaled, scalars["spacing"], retain_vertices=retain_vertices)
    sampled += np.array(
        [scalars["origin_x"], scalars["origin_y"]], dtype=np.float64)
    result = sampled.astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError("generated path must be finite")
    if np.any(np.linalg.norm(np.diff(result, axis=0), axis=1) <= 1.0e-6):
        raise ValueError(
            "spacing is too small to produce distinct float32 points")
    return result


def path_tangent_headings(points: np.ndarray) -> np.ndarray:
    """Return the outgoing segment heading at each point in a finite path."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1:] != (2,) or values.shape[0] < 2:
        raise ValueError("points must have shape (N, 2) with N >= 2")
    if not np.all(np.isfinite(values)):
        raise ValueError("points must be finite")

    delta = np.diff(values, axis=0)
    if np.any(np.linalg.norm(delta, axis=1) <= 1.0e-6):
        raise ValueError("path points must be distinct")
    segment_heading = np.arctan2(delta[:, 1], delta[:, 0])
    return np.concatenate((segment_heading, segment_heading[-1:]))
