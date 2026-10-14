"""Finite polyline geometry: the two projections every path cost is built on."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from statenav_mppi.mppi_core.mppi_types import PathProjection


def sanitize_path(path: np.ndarray) -> np.ndarray:
    """Return a finite, de-duplicated path with at least two unique points."""
    points = np.asarray(path, dtype=np.float32)
    if points.ndim != 2 or points.shape[1:] != (2,) or points.shape[0] < 2:
        raise ValueError("global path must have shape (N, 2) with N >= 2")
    if not np.all(np.isfinite(points)):
        raise ValueError("global path contains non-finite coordinates")
    keep = np.ones(points.shape[0], dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > 1.0e-6
    points = points[keep]
    if points.shape[0] < 2:
        raise ValueError("global path must contain at least two distinct points")
    return points


def extend_path_end(path: np.ndarray, extension_m: float) -> np.ndarray:
    """Return ``path`` continued straight past its last vertex along its final tangent.

    Inputs : ``path`` an ``(N, 2)`` polyline; ``extension_m`` (m) > 0, how far past
             ``path[-1]`` the continuation runs.
    Returns: an ``(N + K, 2)`` float32 polyline whose first N points are ``path``
             sanitized and whose K appended points walk on in the direction of the last
             real segment, the last of them exactly ``extension_m`` past ``path[-1]``.

    The continuation is a COSTING DEVICE, not a route.  It exists so the local reference
    keeps a full-length forward interval -- and with it a tangent, a cross-track direction
    and an unsaturated arc -- while the robot closes the last metres of a path.  Without
    it the reference collapses onto the final vertex: every path cost degenerates into a
    radial pull toward that one point, progress saturates, and the coupled command simplex
    spends as yaw the budget it can no longer spend on progress.

    Nothing here decides where the robot STOPS.  A caller that plans along the
    continuation must refuse to plan from a pose whose foot has already passed
    ``path[-1]`` -- see PathTrackingMPPI._select_reference, which owns that guard.

    The appended vertices are spaced at the path's own median segment length, so the
    continuation looks like the rest of the polyline to a consumer that walks vertices
    (globalmap.get_waypoint does).  At least one vertex is always appended.
    """
    points = sanitize_path(path)
    extension = float(extension_m)
    if not np.isfinite(extension) or extension <= 0.0:
        raise ValueError(
            f"path extension must be finite and positive, got {extension_m!r}")
    # (2,) unit direction of the last real segment.  sanitize_path keeps consecutive
    # points more than 1e-6 m apart, so this norm cannot be zero.
    tangent = points[-1] - points[-2]
    tangent = tangent / float(np.linalg.norm(tangent))
    # (m) vertex spacing of the continuation: the path's own median segment length,
    # floored so a pathologically fine path cannot produce thousands of virtual points
    # and capped at the extension so at least one vertex is emitted.
    spacing = float(np.median(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    spacing = min(max(spacing, 0.05), extension)
    count = max(1, int(np.ceil(extension / spacing)))
    # (K,) arc distances past points[-1]; (K, 2) the virtual vertices themselves.
    steps = np.linspace(extension / count, extension, count, dtype=np.float32)
    virtual = points[-1][None, :] + steps[:, None] * tangent[None, :]
    return np.vstack((points, virtual)).astype(np.float32)


@dataclass(frozen=True)
class PolylineProjection:
    """Nearest finite-segment projection of one planar query point."""

    segment_index: int
    fraction: float
    point: np.ndarray
    distance_sq: float
    arc_length: float


def _finite_point(point: np.ndarray) -> np.ndarray:
    value = np.asarray(point, dtype=np.float32)
    if value.shape != (2,) or not np.all(np.isfinite(value)):
        raise ValueError("query point must contain finite [x, y]")
    return value


def project_point_to_polyline(
    path: np.ndarray, point: np.ndarray,
) -> PolylineProjection:
    """Project a ``(2,)`` point onto the nearest segment of an ``(N, 2)`` path.

    Arc length is measured in metres using each finite segment's actual length.
    Only the segment fraction is clipped; the query point remains unchanged.
    """
    points = sanitize_path(path)
    query = _finite_point(point)
    segment = np.diff(points, axis=0)
    length_sq = np.sum(segment * segment, axis=1)
    if not np.all(np.isfinite(length_sq)) or np.any(length_sq <= 0.0):
        raise ValueError("global path must contain a finite segment")

    raw_fraction = np.sum(
        (query[None, :] - points[:-1]) * segment, axis=1,
    ) / length_sq
    fraction = np.clip(raw_fraction, 0.0, 1.0)
    projections = points[:-1] + fraction[:, None] * segment
    distance_sq = np.sum((query[None, :] - projections) ** 2, axis=1)
    segment_index = int(np.argmin(distance_sq))
    segment_length = np.sqrt(length_sq)
    prefix = np.concatenate(([0.0], np.cumsum(segment_length)))

    return PolylineProjection(
        segment_index=segment_index,
        fraction=float(fraction[segment_index]),
        point=projections[segment_index].astype(np.float32, copy=True),
        distance_sq=float(distance_sq[segment_index]),
        arc_length=float(
            prefix[segment_index]
            + fraction[segment_index] * segment_length[segment_index]
        ),
    )


def advance_along_polyline(
    path: np.ndarray, start: PolylineProjection, arc_m: float,
) -> PolylineProjection:
    """Return the point ``arc_m`` metres FORWARD ALONG ``path`` from ``start``.

    The lookahead as an ARC budget rather than a circle: ``arc_m`` metres are spent on the
    path's OWN segments, so a winding path yields a point that is nearer in a straight line
    but exactly ``arc_m`` of TRAVEL away, and the window a horizon is priced over does not
    shrink into a bend.  ``project_point_to_polyline`` answers "where am I on this path";
    this answers "where is the path ``arc_m`` further on".  Both measure arc on the same
    prefix sum, so their arc lengths are directly comparable.

    Inputs : ``path`` an ``(N, 2)`` polyline, sanitized as the projection that produced
             ``start`` sanitized it; ``start`` the projection to walk forward from;
             ``arc_m`` (m) the finite, non-negative arc budget.
    Returns: a PolylineProjection ON the polyline -- ``distance_sq`` is 0 by construction --
             whose ``arc_length`` is ``start.arc_length + arc_m``, short of that only when
             the path ENDS first, in which case its last vertex is returned.  Extrapolating
             past the end would invent a route; extend_path_end is how a caller asks for one.

    Never walks backwards: the returned arc length is always >= ``start.arc_length``, which
    is what makes a separate "is the result actually ahead of the robot?" test unnecessary
    at the call site.
    """
    points = sanitize_path(path)
    budget = float(arc_m)
    if not np.isfinite(budget) or budget < 0.0:
        raise ValueError(
            f"arc budget must be finite and non-negative, got {arc_m!r}")
    # (M, 2) segment vectors and (M,) their lengths.  sanitize_path guarantees M >= 1 and
    # every length > 1e-6 m, so no division below can be by zero.
    segment = np.diff(points, axis=0)
    segment_length = np.linalg.norm(segment, axis=1)
    index = int(start.segment_index)
    if not 0 <= index < segment_length.shape[0]:
        raise ValueError(
            f"start.segment_index {index} is outside the {segment_length.shape[0]} "
            f"segment(s) of this {points.shape[0]}-point path: the projection was taken "
            f"on a DIFFERENT polyline than the one being walked")
    # Walk forward, spending the budget on the part of each segment that lies AHEAD of the
    # current fraction.  ``fraction`` is where on segment ``index`` the walk stands now.
    fraction = float(start.fraction)
    remaining = budget                                            # (m) budget not yet spent
    while index < segment_length.shape[0]:
        ahead = (1.0 - fraction) * float(segment_length[index])   # (m) unspent arc on this segment
        if remaining <= ahead:
            # Clamped: float error on a budget that lands exactly on the far vertex must not
            # push the fraction past it and index a point off the segment.
            fraction = min(1.0, fraction + remaining / float(segment_length[index]))
            break
        remaining -= ahead
        index += 1
        fraction = 0.0
    if index >= segment_length.shape[0]:
        # The budget outlived the path: clamp to its last vertex.
        index = int(segment_length.shape[0]) - 1
        fraction = 1.0
    # (M+1,) arc at every vertex, measured exactly as project_point_to_polyline measures it,
    # so a caller can subtract the two arc lengths.
    prefix = np.concatenate(([0.0], np.cumsum(segment_length)))
    return PolylineProjection(
        segment_index=index,
        fraction=float(fraction),
        point=(points[index] + fraction * segment[index]).astype(np.float32, copy=True),
        distance_sq=0.0,
        arc_length=float(prefix[index] + fraction * segment_length[index]),
    )


def project_to_reference(
    positions: torch.Tensor,
    reference: torch.Tensor,
    *,
    validate_finite: bool = True,
) -> PathProjection:
    """Project ``(..., 2)`` positions onto a finite ``(M, 2)`` polyline.

    Distances are squared metres, fractions are dimensionless and clipped
    to ``[0, 1]``, and arc lengths are measured metres from reference[0].

    A free function rather than a planner method: it reads no planner state, so it
    belongs with the polyline geometry rather than with the layer that used to own it
    (PathTrackingMPPI).
    """
    if positions.ndim < 1 or positions.shape[-1] != 2:
        raise ValueError("positions must have shape (..., 2)")
    if reference.ndim != 2 or reference.shape[0] < 2 or reference.shape[1] != 2:
        raise ValueError("reference must have shape (M, 2) with M >= 2")
    if validate_finite and (
        not bool(torch.all(torch.isfinite(positions)))
        or not bool(torch.all(torch.isfinite(reference)))
    ):
        raise ValueError("positions and reference must be finite")

    segment_start = reference[:-1]
    segment_delta = torch.diff(reference, dim=0)
    segment_length_sq = torch.sum(segment_delta * segment_delta, dim=-1)
    if validate_finite and bool(torch.any(segment_length_sq <= 0.0)):
        raise ValueError("reference must contain finite distinct segments")
    relative = positions[..., None, :] - segment_start
    fraction = torch.sum(relative * segment_delta, dim=-1) / segment_length_sq
    fraction = torch.clamp(fraction, 0.0, 1.0)
    projected = segment_start + fraction[..., None] * segment_delta
    all_distance_sq = torch.sum(
        (positions[..., None, :] - projected) ** 2, dim=-1)
    distance_sq, segment_index = torch.min(all_distance_sq, dim=-1)
    selected_fraction = torch.gather(
        fraction, -1, segment_index[..., None]).squeeze(-1)
    segment_length = torch.sqrt(segment_length_sq)
    prefix = torch.cat((
        torch.zeros(1, device=reference.device, dtype=reference.dtype),
        torch.cumsum(segment_length, dim=0),
    ))
    arc_length = (
        prefix[segment_index]
        + selected_fraction * segment_length[segment_index]
    )
    tangent = torch.atan2(segment_delta[:, 1], segment_delta[:, 0])
    return PathProjection(
        distance_sq=distance_sq,
        segment_index=segment_index,
        fraction=selected_fraction,
        arc_length=arc_length,
        tangent_yaw=tangent[segment_index],
    )
