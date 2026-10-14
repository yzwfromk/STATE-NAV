"""Contract layer shared by the world model (writer) and the planners / MPPI (readers).

Holds only numpy-level pieces: the shared-memory map backend, the shared-memory graph,
and the graph payload layout (graph_layout.GRAPH_LAYOUT). Nothing here may import torch,
matplotlib, scipy, ROS, or another statenav_global subpackage, so a reader process can
attach to the world model's shared memory without the world-model stack.

This __init__ is INTENTIONALLY EMPTY (same rule as statenav_fsm): a re-export here would
make every consumer pay for every submodule. Import submodules explicitly:
    from statenav_global.core.graph_layout import GRAPH_LAYOUT
    from statenav_global.core.shared_graph import SharedGraphReader
"""
