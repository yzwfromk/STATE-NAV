"""Terrain-aware Model Predictive Path Integral (MPPI) local planner.

The implementation is intentionally self-contained: PyTorch supplies batched
rollouts, while the StateNav shared map is consumed through its existing public
NumPy arrays.  No writer-side map changes are required.

This module is now the composition point and the public import surface.  The
implementation is layered across eleven modules, each owning one concern:

    mppi_core/mppi_types.py         MPPIConfig and the frozen input/result contracts
    mppi_core/mppi_costs.py         the cost registry: term table, bound objective, MPPIProblem
    mppi_core/mppi_constraints.py   CommandConstraint, and the command-rectangle registry
    mppi_core/mppi_validity.py      the validity registry: which rollouts the update may use
    mppi_core/mppi_core.py          MPPICore -- noise, rollout projection, plan bookkeeping
    mppi_core/mppi_solver.py        MPPISolver -- one MPPI update over an MPPIProblem
    mppi_core/mppi_dynamics.py      the actuator-lag rollout
    mppi_utils/mppi_debug.py        the per-tick tuning trace, as pure formatting
    mppi_utils/mppi_path_geometry.py  polyline validation, projection, arc-length reference
    mppi_planners/mppi_terrain.py   TerrainMPPI -- map cache, footprint reduction, terrain cost
    mppi_planners/mppi_tracking.py  PathTrackingMPPI -- reference, path cost terms, one tick
    mppi_planners/mppi_tasks.py     the task mixin

Every name this module used to define is re-exported below, so
``from statenav_mppi.mppi_planners.mppi_planner import ...`` is the single public import
surface for main.py, the demo node and the tests.
"""

from __future__ import annotations

# Re-exports, not uses.  The underscore-prefixed names are imported deliberately: they
# were module-level names here before the split and existing call sites still reach for
# them through this module (test_mppi_rviz.py constructs mppi_planner._NullVisualizer).
# Removing any of them is a breaking change, so they are bound in _REEXPORTED below
# rather than left looking like dead imports.
from statenav_mppi.mppi_core.mppi_core import MPPICore, _NullVisualizer
from statenav_mppi.mppi_planners.mppi_tasks import (
    WpTrackingMixin,
)
from statenav_mppi.mppi_core.mppi_solver import MPPISolver
from statenav_mppi.mppi_planners.mppi_terrain import TerrainMPPI
from statenav_mppi.mppi_planners.mppi_tracking import PathTrackingMPPI
from statenav_mppi.mppi_core.mppi_types import (
    MPPIConfig,
    MPPIInput,
    MPPIPathTrackingResult,
    MPPIResult,
    PathProjection,
    _DEFAULTS,
    _MAPPING,
    _cfg_value,
    _load_yaml_defaults,
    find_default_config_path,
)


class MPPIPlanner(
    WpTrackingMixin,
    PathTrackingMPPI,
):
    """Batched unicycle MPPI with version-aware traversability-map caching.

    Method resolution order, most specific first:

        MPPIPlanner
          WpTrackingMixin         track_path()
          PathTrackingMPPI        _track_path(), reference and path cost terms
          TerrainMPPI             map cache, footprint limits, terrain cost term
          MPPISolver              solve(), _sample_and_score(), _settle()
          MPPICore                noise, rollout projection, plan state

    The constructor is inherited unchanged: MPPIPlanner(config, global_map=None).
    """


# Private names this module used to define, kept importable for existing call sites.
# Bound here so a linter sees them as used and a reader sees they are intentional.
_REEXPORTED = (
    _NullVisualizer,
    _DEFAULTS,
    _MAPPING,
    _cfg_value,
    _load_yaml_defaults,
)


__all__ = [
    "MPPIPlanner",
    "MPPIConfig",
    "MPPIInput",
    "MPPIResult",
    "MPPIPathTrackingResult",
    "PathProjection",
    "MPPICore",
    "MPPISolver",
    "TerrainMPPI",
    "PathTrackingMPPI",
    "WpTrackingMixin",
    "find_default_config_path",
]
