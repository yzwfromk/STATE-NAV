"""Uniformly traversable in-memory map for the standalone MPPI demo."""

from __future__ import annotations

import math

import numpy as np

from statenav_mppi.mppi_planners.mppi_travgraph import InMemoryTravGraphReader


class FlatDemoMap:
    """Expose the production MPPI map-reader contract without shared memory.

    The planner reads only map metadata plus a travgraph, so the flat terrain is a uniform
    in-memory lattice graph (``travgraph_reader``) on the planner's lattice resolution.
    """

    def __init__(
        self,
        *,
        half_extent_m: float = 20.0,
        resolution_m: float = 0.25,
        heading_layers: int = 36,
        max_v: float = 2.0,
        max_w: float = 1.0,
        lattice_resolution_m: float = 0.5,
    ) -> None:
        scalars = {
            "half_extent_m": float(half_extent_m),
            "resolution_m": float(resolution_m),
            "max_v": float(max_v),
            "max_w": float(max_w),
            "lattice_resolution_m": float(lattice_resolution_m),
        }
        for name, value in scalars.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(heading_layers, bool)
            or not isinstance(heading_layers, (int, np.integer))
            or int(heading_layers) <= 0
        ):
            raise ValueError("heading_layers must be a positive integer")

        self.map_xmin = -scalars["half_extent_m"]
        self.map_xmax = scalars["half_extent_m"]
        self.map_ymin = -scalars["half_extent_m"]
        self.map_ymax = scalars["half_extent_m"]
        self.map_resolution = scalars["resolution_m"]
        cells = int(math.ceil(
            2.0 * scalars["half_extent_m"] / self.map_resolution)) + 1
        if cells < 2:
            raise ValueError(
                "flat demo map must contain at least two cells per axis")

        layers = int(heading_layers)
        # Fine-grid metadata the planner indexes on: (cells, cells) cells x `layers` headings.
        self.map_size_rows = cells
        self.map_size_cols = cells
        self.TraversabilityMap_size_layers = layers
        # Uniform terrain: every lattice node at (max_v, max_w) on every heading layer.  Must sit
        # on MPPIConfig.travgraph_resolution, which the planner checks every node against.
        self.travgraph_reader = InMemoryTravGraphReader.uniform(
            (self.map_xmin, self.map_xmax, self.map_ymin, self.map_ymax),
            scalars["lattice_resolution_m"], layers, scalars["max_v"], scalars["max_w"])
        self.is_TraversabilityMap_built = True
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_heading = 0.0
