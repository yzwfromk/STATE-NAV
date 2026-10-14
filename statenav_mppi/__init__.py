"""Terrain-aware MPPI local planner, split into three layers.

    mppi_core/      the solver stack: config/contracts, dynamics, cost registry,
                    plan bookkeeping and the MPPI update itself
    mppi_planners/  the planners built on that stack: terrain, path tracking,
                    the task mixins and MPPIPlanner, the composition point
    mppi_utils/     helpers with no planner state: polyline geometry, the tuning
                    trace and the RViz MarkerArray publisher

Every ``__init__.py`` in this tree is deliberately kept free of re-exports.
mppi_core.mppi_costs imports mppi_utils.mppi_path_geometry while
mppi_utils.mppi_debug imports mppi_core.mppi_costs, so the two subpackages form a
cycle at package level.  That is harmless only while importing a subpackage runs
no module code -- re-exporting here would turn it into a real circular import.
Import the leaf module directly: ``from statenav_mppi.mppi_core.mppi_types import MPPIConfig``.
"""
