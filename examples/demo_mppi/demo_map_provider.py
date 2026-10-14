"""Select the map source used by the isolated MPPI ROS demo."""

from __future__ import annotations

from typing import Any, Optional

from demo_mppi.flat_demo_map import FlatDemoMap


def _create_shared_map() -> tuple[Any, Any]:
    """Attach to the production shared map only when explicitly selected."""
    from statenav_global.world_model.globalmap_with_backend import (
        CMDbasedMapReaderProxy,
    )
    from statenav_global.world_model.shared_memory_backend import (
        SharedMemoryBackend,
    )

    backend = SharedMemoryBackend(mode="reader")
    return backend, CMDbasedMapReaderProxy(backend)


def create_demo_map(
    *,
    use_shared_map: bool,
    half_extent_m: float,
    resolution_m: float,
    heading_layers: int,
    max_v: float,
    max_w: float,
    lattice_resolution_m: float = 0.5,
) -> tuple[Optional[Any], Any]:
    """Return the selected backend handle and planner-compatible map."""
    if use_shared_map:
        return _create_shared_map()
    return None, FlatDemoMap(
        half_extent_m=half_extent_m,
        resolution_m=resolution_m,
        heading_layers=heading_layers,
        max_v=max_v,
        max_w=max_w,
        lattice_resolution_m=lattice_resolution_m,
    )
