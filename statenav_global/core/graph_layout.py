"""Shared-memory graph payload contract: the one GraphLayout writer and readers agree on.

Moved out of world_model/globalmap.py (which re-exports every name below) so readers such
as GlobalAStar and the MPPI can import the layout without globalmap's torch / ROS stack.
Dependencies: numpy, omegaconf (the fresh-node cmd defaults come from planning_config.yaml).
"""
from enum import IntEnum

import numpy as np

from statenav_global.core.config import load_planning_config
from statenav_global.core.shared_graph import GraphLayout

# Same config file globalmap.py reads (core.config.planning_config_path()), so the writer's
# map defaults and this layout's fresh-node defaults come from one source.
cfg = load_planning_config()


class TravmapInfo_IntEnum(IntEnum):

    TRAV_NotInitialized = 0
    TRAV_NEEDUPDATE = 1
    TRAV_Estimated = 2


# ---------------------------------------------------------------------------
# Shared-memory graph payload layout (Part 2)
# ---------------------------------------------------------------------------
# One module-level layout imported identically by the writer (the world model,
# globalmap.py) and the reader (planner, Part 3). Do NOT reconstruct an equivalent
# dict at other call sites.
#
# The fresh-node command defaults are the map's InitialGuess_cmd_v / _w, i.e.
# cfg.default_safe_cmd_v / cfg.default_safe_cmd_w. They are read from the same
# config source rather than duplicated as literals, so a fresh graph node's 8 cmd
# layers hold the map's Initialguess state exactly. `meta` defaults to
# TRAV_NotInitialized on every layer (nothing estimated yet).
INITIAL_CMD_V_DEFAULT = float(cfg.default_safe_cmd_v)   # scalar broadcast over 8 heading layers
INITIAL_CMD_W_DEFAULT = float(cfg.default_safe_cmd_w)   # scalar broadcast over 8 heading layers

# Number of heading layers the graph payloads carry. The graph layout is a fixed
# module constant, so it hard-codes the 8-layer heading discretization
# (theta_min = -pi, theta_resolution = pi/4). _build_shared_graph guards that the
# live map instance actually uses this many layers before constructing the graph.
_GRAPH_HEADING_LAYERS = 8

# Node payloads are THE traversability store (there is no dense grid): cmd_v/cmd_w
# (float32, one value per heading layer) and meta (per-layer TravmapInfo_IntEnum
# status, the Step-9 lock gate's input).
# Edge payloads are empty: edges carry only generic graph-core data (immutable
# geometric length); heading-aware traversal cost is derived planner-side in
# Part 3, never stored on edges.
GRAPH_LAYOUT = GraphLayout(
    node_payloads={
        "cmd_v": ((_GRAPH_HEADING_LAYERS,), np.float32, INITIAL_CMD_V_DEFAULT),
        "cmd_w": ((_GRAPH_HEADING_LAYERS,), np.float32, INITIAL_CMD_W_DEFAULT),
        "meta": (
            (_GRAPH_HEADING_LAYERS,),
            np.uint8,
            int(TravmapInfo_IntEnum.TRAV_NotInitialized),
        ),
    },
    edge_payloads={},
)
