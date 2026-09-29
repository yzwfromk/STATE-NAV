from time import time
import numpy as np
import os
import datetime
import math
import random
import threading
import matplotlib
# Headless 'Agg' backend by default, so merely importing this module never opens a pop-up
# window and never requires a Qt binding (e.g. on the Jetson, or in the MPC venv which has
# no Qt binding). NOTE: the previous 'QtAgg if DISPLAY else Agg' rule silently re-enabled
# GUI windows in processes that had explicitly asked for 'Agg' — this module is imported
# AFTER utility/plotting.py in the statenav_global package, and matplotlib.use() defaults to
# force=True, so it overrode the earlier headless choice. An explicit MPLBACKEND or
# STATENAV_MPL_BACKEND env var still wins (set either to 'QtAgg' for interactive debugging).
if not os.environ.get('MPLBACKEND'):
    matplotlib.use(os.environ.get('STATENAV_MPL_BACKEND', 'Agg'))
import matplotlib.pyplot as plt
from enum import IntEnum
import hydra
from omegaconf import DictConfig, OmegaConf


import torch


from statenav_global.utility import Utils
from statenav_global.utility.utils import get_source_inference_dir
from statenav_global.utility.plotting import (
    RC_PARAMS, FS_TICK, FS_LABEL, FS_LEGEND, FS_TITLE,
    PLOT_DPI, PLOT_FORMAT, FONT_PROP, BaseFigure, MapFigure
)
# Shared-memory graph backend (Part 1a/1b). Only the writer-side symbols used by
# the Part 2 grid-topology integration are imported here; the layout below is the
# single definition writer and reader (planner) must both import.
from statenav_global.world_model.shared_graph import (
    GraphLayout, SharedGraphWriter, SharedGraphError,
)





from std_msgs.msg import Float32MultiArray
from grid_map_msgs.msg import GridMap
from geometry_msgs.msg import PoseStamped


import tf_transformations as tf_trans
import transforms3d as tf3
from scipy.ndimage import map_coordinates
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)




from pathlib import Path
_PKG_ROOT = Path(__file__).resolve().parent.parent
cfg = OmegaConf.load(_PKG_ROOT / "configs/planning_config.yaml")


    
lookahead_distance = cfg.lookahead_distance

safecmd_sampling_dist = cfg.safecmd_sampling_dist
lookahead_angle = cfg.lookahead_angle

instability_std_multiplier = cfg.instability_std_multiplier










# batch putting of local elev map to global elev map
# 



class ElevmapInfo_IntEnum(IntEnum):

    # Lower number means more primitive status
    # Higher number implies lower number status

    ELEV_NotInitialized = 0
    ELEV_MEASURED = 1

class TravmapInfo_IntEnum(IntEnum):

    TRAV_NotInitialized = 0
    TRAV_NEEDUPDATE = 1
    TRAV_Estimated = 2


# ---------------------------------------------------------------------------
# Shared-memory graph payload layout (Part 2)
# ---------------------------------------------------------------------------
# One module-level layout imported identically by the writer (this world model)
# and the reader (planner, Part 3). Do NOT reconstruct an equivalent dict at
# other call sites.
#
# The fresh-node command defaults are bound to the exact values the existing
# `Initilize_map` writes into TraversabilityMap on initialization
# (TraversabilityMap[:,:,:,0] = InitialGuess_cmd_v, [:,:,:,1] = InitialGuess_cmd_w),
# which are cfg.default_safe_cmd_v / cfg.default_safe_cmd_w. They are read from
# the same config source rather than duplicated as literals, so a fresh graph
# node's 8 cmd layers match the map's Initialguess state exactly. `meta` defaults
# to TRAV_NotInitialized on every layer, matching the initial TravMap_MetaInfo.
INITIAL_CMD_V_DEFAULT = float(cfg.default_safe_cmd_v)   # scalar broadcast over 8 heading layers
INITIAL_CMD_W_DEFAULT = float(cfg.default_safe_cmd_w)   # scalar broadcast over 8 heading layers

# Number of heading layers the graph payloads carry. The graph layout is a fixed
# module constant, so it hard-codes the 8-layer heading discretization
# (theta_min = -pi, theta_resolution = pi/4). _build_shared_graph guards that the
# live map instance actually uses this many layers before constructing the graph.
_GRAPH_HEADING_LAYERS = 8

# Node payloads mirror TraversabilityMap cmd_v/cmd_w (float32, one value per
# heading layer) and TravMap_MetaInfo (per-layer TravmapInfo_IntEnum status).
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




















class BaseMap:
    def __init__(self, env_xmin, env_xmax, env_ymin, env_ymax, goal_x, goal_y, which_layer, preest_update_resolution, instab_limit, load_Travformer = True):

        self.debug_print = cfg.debug_print

        # Throttle counter for debug printouts (prints every 50 calls to avoid log flooding)
        self._dbg_callback_count = 0
        self._dbg_update_count = 0

        self.goal_x = goal_x
        self.goal_y = goal_y


        # Robocentric Local Elevation Map from ROS
        self.is_RobocentricMap_received = False
        self.layer_data = {}
        self._map_data_lock = threading.Lock()

        self.RobocentricMap_CenterPosOffset_Worldfr = None # Not necessarily identical to the robot pos
        self.RobocentricMap_CenterRotOffset_Worldfr = None
        
        self.map_resolution = None # shared across all maps
        self.RobocentricMap_LengthY = None
        self.RobocentricMap_LengthX = None
        self.RobocentricMap_Rows = None
        self.RobocentricMap_Cols = None
 
        # Robot position and heading in the world frame
        self.robot_x = cfg.initial_start[0] # initialized
        self.robot_y = cfg.initial_start[1]
        self.robot_roll = 0.0  # initialized
        self.robot_pitch = 0.0  # initialized
        self.robot_heading = cfg.heading_start


        # World frame Global Elevation Map
        self.is_ElevationMap_built = False
        self.ElevationMap = None
        self.map_xmin = env_xmin
        self.map_xmax = env_xmax
        self.map_size_x = self.map_xmax - self.map_xmin
        self.map_size_rows = None

        self.map_ymin = env_ymin
        self.map_ymax = env_ymax
        self.map_size_y = self.map_ymax - self.map_ymin
        self.map_size_cols = None

        # ElevationMap update methods
        self.which_layer = which_layer


        # 3D (x,y,theta(=heading angle)) Traversability map that contains the pre-estimated cmd v and w. World frame
        self.is_TraversabilityMap_built = False
        self.TraversabilityMap = None
        self.TraversabilityMap_theta_min = -np.pi
        self.TraversabilityMap_theta_resolution = np.pi/4 # Do not change. 
        self.TraversabilityMap_theta_max = np.pi - self.TraversabilityMap_theta_resolution
        self.TraversabilityMap_size_layers = 2 * int(np.round(np.pi / self.TraversabilityMap_theta_resolution))
        print("Traversability Map size layers: ", self.TraversabilityMap_size_layers)




        # Trav and Velocity


        self.InitialGuess_cmd_v = None
        self.InitialGuess_cmd_w = None


        self.maximum_cmd_v = cfg.maximum_cmd_v
        self.maximum_cmd_w = cfg.maximum_cmd_w





        
        

        # TraversabilityMap update methods
        self.TravUpdate_XYResolution = preest_update_resolution
        self.TravUpdate_Pts = np.empty((0, 2))

        # ── Replanning-gate change accounting ────────────────────────────────────────
        # Accumulated map/graph change since the last published /replanning_signal.
        # Reset ONLY by WorldModel.notify_replanning_published(), so sub-threshold change
        # keeps accumulating across ticks instead of being discarded each tick.
        #   graph_toorisky_flips      : nodes that entered or left the too-risky set (edges changed)
        #   graph_nodes_value_changed : nodes whose cmd_v/cmd_w/meta payload actually changed value
        self.map_change_stats = {
            'graph_toorisky_flips': 0,
            'graph_nodes_value_changed': 0,
        }
        # Per-layer |Δcmd| below which a rewrite is treated as "same value" (no change).
        self._replan_cmd_change_eps = float(cfg.get('replan_cmd_change_eps', 0.05))

        self.elevation_diff_threshold = cfg.elevation_diff_threshold
        self.RandomRate_TravUpdate = cfg.RandomRate_TravUpdate


        self.ElevmapInfo_IntEnum = ElevmapInfo_IntEnum
        self.is_ElevMap_MetaInfo_built = False
        self.ElevMap_MetaInfo = None

        self.TravmapInfo_IntEnum = TravmapInfo_IntEnum
        self.is_TravMap_MetaInfo_built = False
        self.TravMap_MetaInfo = None
        
        self.TravMap_MetaInfo_theta_min = self.TraversabilityMap_theta_min
        self.TravMap_MetaInfo_theta_resolution = self.TraversabilityMap_theta_resolution
        self.TravMap_MetaInfo_theta_max = self.TraversabilityMap_theta_max
        self.TravMap_MetaInfo_size_layers = self.TraversabilityMap_size_layers



        # Visualization
        self.viz_channel = 0  # default channel
        self.viz_vmin = 0.0
        self.viz_vmax = self.maximum_cmd_v
        self.viz_title = "Traversability Map"
        self.viz_cbar_label = "Traversability Score"

        # Pre-allocated flat float32 buffers for ros_utils.Rviz_vis_travmap (one per layer)
        # Allocated in Initilize_map once map dimensions are known; shape: [num_rows * num_cols]
        self._vis_flat_bufs = None


        # ML related
        self.device = 'cuda'




        # Travformer related
        self.local_patch_size = cfg.local_patch_size  # in meters
        self.elevonly_vw2instab_model = None

        def _load_model(checkpoint_path, model_type='Travformer', device='cpu'):

            from statenav_global.inference.uncertainty_models import ElevationOnlyNetworkMLL
            if model_type == 'Travformer':
                model = ElevationOnlyNetworkMLL()

            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            model.to(device)
            model.eval()
            return model


        if load_Travformer:
            inference_dir = get_source_inference_dir()
            elevonly_vw2instab_model_checkpoint = os.path.join(inference_dir + "/checkpoints/checkpoint_MLL_23.pth")
            self.elevonly_vw2instab_model = _load_model(elevonly_vw2instab_model_checkpoint, model_type='Travformer', device=self.device)
            print("Travformer model loaded.")
        else:
            print("Skipping Travformer model loading (read-only mode).")







    @torch.no_grad()
    def TravFormer_Inference(self, target_instability, elevation_map_numpy, commands_numpy, model, device='cpu'):

        cmdvorcmdw = 0 if commands_numpy[0,0] != 0 else 1

        # Preprocess inputs
        elevation_map = torch.tensor(elevation_map_numpy.copy(), dtype=torch.float32, requires_grad=True).unsqueeze(1).to(device)
        commands = torch.tensor(commands_numpy.copy(), dtype=torch.float32, requires_grad=True).to(device)


        pred, pred_logstd = model(elevation_map, commands)
        pred_logstd = torch.exp(pred_logstd)

        if target_instability is not None:
            mean_grad_commands = torch.autograd.grad(pred, commands, grad_outputs=torch.ones_like(pred), retain_graph=True, create_graph=True)[0]
            std_grad_commands = torch.autograd.grad(pred_logstd, commands, grad_outputs=torch.ones_like(pred_logstd), create_graph=True, retain_graph=True, allow_unused=True)[0]

            pred = pred.cpu().detach().numpy()
            pred_logstd = pred_logstd.cpu().detach().numpy()
            mean_grad_commands = mean_grad_commands.cpu().detach().numpy()
            std_grad_commands = std_grad_commands.cpu().detach().numpy()

            residual = (target_instability - ((pred[:,0]) + instability_std_multiplier*np.exp((pred_logstd[:,0])) ) )
            total_gradient = (mean_grad_commands[:,cmdvorcmdw]) + instability_std_multiplier*(std_grad_commands[:,cmdvorcmdw]) * np.exp((pred_logstd[:,0])) 
            delta_vorw = total_gradient**(-1) * residual
            target_vorw = (commands_numpy[:,cmdvorcmdw]) + delta_vorw

        else:
            pred = pred.cpu().detach().numpy()
            pred_logstd = pred_logstd.cpu().detach().numpy()

        return pred, pred_logstd




    def _initialize_traversability_map_values(self):
        """Initialize every traversability-map command with the neutral guess."""
        if (self.TraversabilityMap is None or self.TraversabilityMap.ndim != 4
                or self.TraversabilityMap.shape[3] != 4):
            raise ValueError(
                "_initialize_traversability_map_values requires TraversabilityMap "
                f"shape (rows, cols, layers, 4), got "
                f"{None if self.TraversabilityMap is None else self.TraversabilityMap.shape}")
        expected_meta_shape = self.TraversabilityMap.shape[:3]
        if self.TravMap_MetaInfo is None or self.TravMap_MetaInfo.shape != expected_meta_shape:
            raise ValueError(
                "_initialize_traversability_map_values requires TravMap_MetaInfo "
                f"shape {expected_meta_shape}, got "
                f"{None if self.TravMap_MetaInfo is None else self.TravMap_MetaInfo.shape}")

        self.TraversabilityMap[:,:,:,0] = self.InitialGuess_cmd_v
        self.TraversabilityMap[:,:,:,1] = self.InitialGuess_cmd_w
        self.TraversabilityMap[:,:,:,2] = np.nan
        self.TraversabilityMap[:,:,:,3] = np.nan
        self.TravMap_MetaInfo[:] = TravmapInfo_IntEnum.TRAV_NotInitialized.value












    def bump_map_version(self):
        """
        Publish "the dense map content changed" to the backend, ONCE per map-write
        transaction.

        The backend version counter is the only cross-process "map changed" signal
        (SharedMemoryBackend.increment_version / LocalMemoryBackend.increment_version),
        so EVERY path that mutates ElevationMap / TraversabilityMap / ElevMap_MetaInfo /
        TravMap_MetaInfo must call this once it has finished writing.

        One bump per Update_map / Initilize_map.

        getattr guard: BaseMap also runs without any backend (plain CMDbasedMap in the
        unit tests). The reader-mode guard keeps a reader proxy
        (CMDbasedMapReaderProxy, which inherits every write method) from bumping the
        writer's shared counter should it ever call one of them.
        """
        backend = getattr(self, 'backend', None)
        if backend is not None and getattr(backend, 'mode', 'writer') != 'reader':
            backend.increment_version()


    def Initilize_map(self):

        if self.is_RobocentricMap_received:

        # Initialize global map
            if not self.is_ElevationMap_built:
                self.is_ElevationMap_built = True

            self.map_size_rows = int(self.map_size_x/self.map_resolution)
            self.map_size_cols = int(self.map_size_y/self.map_resolution)

            self.ElevationMap = np.full((self.map_size_rows, self.map_size_cols), np.nan)

        # Build global trav map  (flag not initialized here though)
        # 3D (x,y,theta(=heading angle)) global map that contains the pre-estimated cmd v and w array
            self.TraversabilityMap = np.full((self.map_size_rows, self.map_size_cols, self.TraversabilityMap_size_layers, 4), np.nan)

        # Build metainfo map
            self.ElevMap_MetaInfo = np.full((self.map_size_rows, self.map_size_cols), ElevmapInfo_IntEnum.ELEV_NotInitialized.value)
            self.TravMap_MetaInfo = np.full((self.map_size_rows, self.map_size_cols, self.TravMap_MetaInfo_size_layers), TravmapInfo_IntEnum.TRAV_NotInitialized.value)
            self._initialize_traversability_map_values()

        


    

        


    def Update_map(self, path_plan = None, visualize_map = False):
        
        """
        # In grid map, the left top corner ( = (0,0) ) is the max x and y point in the map, regardless of the robot pose.
        # increasing row/col indexing results in decreasing x/y in world coordinate.
        # x increases as row decreases, y increases as col decreases
        # This rule is the same both for local and global map
        """


        if not self.is_ElevationMap_built:
            self.Initilize_map() # it goes to backend Initilize_map if backend is running

        if self.is_RobocentricMap_received and self.is_ElevationMap_built:

            # Snapshot the callback-owned data atomically so the entire Update_map
            # iteration uses a consistent (position, layers) pair from the same message.
            with self._map_data_lock:
                layer_data_snap = dict(self.layer_data)
                center_pos_snap = self.RobocentricMap_CenterPosOffset_Worldfr

            # Update global map
            # rospy.loginfo("Available layers in self.layer_data: %s", list(layer_data_snap.keys()))

            # local_map : (H, W) float – robocentric elevation grid received this tick
            local_map = layer_data_snap[self.which_layer[0]]
            H, W = local_map.shape
            # half_r / half_c: pixel offset from the local map's top-left corner to its center cell,
            # used to convert (row, col) indices into world-frame metric offsets
            half_r = int(self.RobocentricMap_Rows / 2)
            half_c = int(self.RobocentricMap_Cols / 2)
            R         = self.TravUpdate_XYResolution  # coarse grid resolution (m) used to quantise TravUpdate candidate positions
            half_patch = 0.5 * self.local_patch_size  # half the Travformer inference window (m); candidates closer to the map edge than this are skipped
            NINIT_T = self.TravmapInfo_IntEnum.TRAV_NotInitialized.value  # int sentinel: traversability not yet estimated for a direction
            NINIT_E = self.ElevmapInfo_IntEnum.ELEV_NotInitialized.value  # int sentinel: elevation cell never written

            # ---------------------------------------------------------------
            # Step 1: Map every local cell to its world-frame position and
            #         corresponding index in the global elevation map.
            #
            # Convention (shared by local and global maps):
            #   row 0 = highest x; row increases → x decreases
            #   col 0 = highest y; col increases → y decreases
            # ---------------------------------------------------------------

            # row_idx / col_idx : (H, W) int – full index grids for the local map
            row_idx, col_idx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

            # global_x_all / global_y_all : (H, W) float – world-frame metric position of each local cell
            global_x_all = (half_r - row_idx) * self.map_resolution + center_pos_snap[0]
            global_y_all = (half_c - col_idx) * self.map_resolution + center_pos_snap[1]

            # row_global_all / col_global_all : (H, W) int – corresponding row/col in the global elevation map
            row_global_all = ((self.map_xmax - global_x_all) / self.map_resolution).astype(int)
            col_global_all = ((self.map_ymax - global_y_all) / self.map_resolution).astype(int)

            # ---------------------------------------------------------------
            # Step 2: Build a validity mask and flatten to per-valid-cell vectors.
            #
            # A cell is valid if it has sensor data (non-NaN) AND its world
            # position falls within the pre-allocated global map extent.
            # Cells outside the global map are silently dropped (the map does
            # not auto-expand).
            # ---------------------------------------------------------------

            # valid : (H, W) bool – True for cells that have data and lie inside the global map
            valid = (~np.isnan(local_map)
                     & (row_global_all >= 0) & (row_global_all < self.map_size_rows)
                     & (col_global_all >= 0) & (col_global_all < self.map_size_cols))

            # All vectors below are shape (N,) where N = number of valid cells this tick

            vrl = row_idx[valid]       # local map row indices
            vcl = col_idx[valid]       # local map col indices
            vrg = row_global_all[valid]  # global map row indices
            vcg = col_global_all[valid]  # global map col indices
            vgx = global_x_all[valid]  # world-frame x (m)
            vgy = global_y_all[valid]  # world-frame y (m)
            local_vals = local_map[vrl, vcl]              # (N,) float – incoming elevation values
            elev_meta  = self.ElevMap_MetaInfo[vrg, vcg]  # (N,) int   – current ElevmapInfo_IntEnum status per cell

            # ---------------------------------------------------------------
            # Step 3: Classify each valid cell into one of three triggering
            #         conditions that require queueing nearby grid points for
            #         traversability re-estimation.
            #
            #   Case A – Cell newly seen: global cell had no data yet.
            #            Queue the coarse-grid neighbourhood so the Travformer
            #            can estimate traversability for this region.
            #
            #   Case B – Terrain changed: elevation differs from the stored
            #            value by more than elevation_diff_threshold.
            #            The traversability estimate is stale and must be
            #            recomputed regardless of current TravMap state.
            #
            #   Case C – Gap-filling probe: the global cell is already
            #            measured, but some traversability directions are still
            #            unestimated (e.g. the wavefront hasn't reached here
            #            yet). A random subsample of such cells is queued to
            #            gradually fill these gaps.
            #
            # Cases are mutually exclusive in priority order A > B > C.
            # ---------------------------------------------------------------

            # case_a : (N,) bool – cells seen for the first time
            case_a = elev_meta == NINIT_E

            # elev_diff : (N,) float – |new elevation − stored elevation| per cell
            elev_diff = np.abs(local_vals - self.ElevationMap[vrg, vcg])
            # case_b : (N,) bool – cells whose terrain changed significantly
            case_b = (~case_a
                      & (elev_diff > self.elevation_diff_threshold)
                      & (elev_meta <= ElevmapInfo_IntEnum.ELEV_MEASURED.value))

            # case_c : (N,) bool – stochastic gap-filling; computed only for cells not already handled
            # by A/B to match the original per-cell RNG consumption pattern
            can_be_c = ~case_a & ~case_b  # (N,) bool – cells eligible for Case C
            case_c = np.zeros(len(vgx), dtype=bool)
            if np.any(can_be_c):
                # trav_any_uninit : (K,) bool – at least one heading direction has no traversability estimate
                #   K = number of can_be_c cells; TravMap_MetaInfo[r,c] has shape (num_layers,) per cell
                trav_any_uninit = np.any(self.TravMap_MetaInfo[vrg[can_be_c], vcg[can_be_c]] == NINIT_T, axis=-1)
                # rand_pass : (K,) bool – random gate so only ~RandomRate_TravUpdate fraction are probed
                rand_pass = np.random.random(int(np.sum(can_be_c))) < self.RandomRate_TravUpdate
                case_c[can_be_c] = trav_any_uninit & rand_pass

            # ---------------------------------------------------------------
            # Step 4: Collect traversability update candidate positions into
            #         self.TravUpdate_Pts.
            #
            # TravUpdate_Pts : (M, 2) float – world-frame (x, y) positions of
            #   coarse-grid points that need Travformer inference. M grows as
            #   new terrain is discovered and shrinks after inference runs.
            #
            # Candidates are quantised to the coarse TravUpdate_XYResolution
            # grid so the Travformer is invoked at well-spaced locations, not
            # once per fine-resolution elevation cell.
            #
            # Cases A and B use the floor-aligned 2×2 neighbourhood of the
            # triggering cell's coarse-grid square (4 candidate positions).
            # Case C uses the single nearest rounded grid point and only adds
            # it if at least one neighbouring grid point is already estimated
            # (so the estimation wavefront expands inward, not outward).
            # ---------------------------------------------------------------

            # existing_set: Python set of (x, y) float64 tuples mirroring TravUpdate_Pts for O(1) dedup
            existing_set = set(map(tuple, self.TravUpdate_Pts)) if len(self.TravUpdate_Pts) > 0 else set()
            new_pts = []  # accumulates new (2,) arrays before a single vstack at the end

            def _floor_to_cands(gx, gy):
                # gx, gy : (K,) float – world coords of triggering cells
                # Returns (4*M, 2) float – 4 floor-grid neighbours for each of M unique floor points.
                # np.arange(fx-R, fx+R, R) gives [fx-R, fx], so offsets span {-R, 0} × {-R, 0}.
                fx = R * np.floor(gx / R)
                fy = R * np.floor(gy / R)
                uniq = np.unique(np.stack([fx, fy], axis=1), axis=0)  # (M, 2) – deduplicated floor points
                offsets = np.array([[-R, -R], [-R, 0.], [0., -R], [0., 0.]])  # (4, 2)
                return (uniq[:, None, :] + offsets[None]).reshape(-1, 2)      # (4M, 2)

            def _base_filter(pts):
                # pts : (P, 2) float – candidate world positions
                # Removes candidates outside the global map and within half_patch of any map boundary
                # (the Travformer needs a full patch around each inference point).
                # Returns (pts_filtered, rg, cg) each of length ≤ P.
                rg = ((self.map_xmax - pts[:, 0]) / self.map_resolution).astype(int)  # (P,) global row
                cg = ((self.map_ymax - pts[:, 1]) / self.map_resolution).astype(int)  # (P,) global col
                keep = ((rg >= 0) & (rg < self.map_size_rows)
                        & (cg >= 0) & (cg < self.map_size_cols)
                        & (np.abs(self.map_xmax - pts[:, 0]) >= half_patch)
                        & (np.abs(pts[:, 0] - self.map_xmin) >= half_patch)
                        & (np.abs(self.map_ymax - pts[:, 1]) >= half_patch)
                        & (np.abs(pts[:, 1] - self.map_ymin) >= half_patch))
                return pts[keep], rg[keep], cg[keep]

            def _add_new(pts):
                # pts : (Q, 2) float – pre-filtered candidates to add
                # Skips any point already in TravUpdate_Pts (via set lookup) and appends the rest.
                for pt in pts:
                    t = tuple(pt)
                    if t not in existing_set:
                        existing_set.add(t)
                        new_pts.append(pt)

            # --- Case A: newly discovered cells ---
            # Queue the coarse-grid neighbourhood only if every traversability direction at
            # that candidate is still uninitialized (any already-estimated neighbour is left alone).
            if np.any(case_a):
                cands, crg, ccg = _base_filter(_floor_to_cands(vgx[case_a], vgy[case_a]))
                if len(cands) > 0:
                    # all_uninit : (Q,) bool – True if NO direction at this candidate has been estimated yet
                    all_uninit = np.all(self.TravMap_MetaInfo[crg, ccg] == NINIT_T, axis=-1)
                    _add_new(cands[all_uninit])

            # --- Case B: terrain changed ---
            # Re-queue unconditionally — the existing traversability estimate is stale.
            if np.any(case_b):
                cands, crg, ccg = _base_filter(_floor_to_cands(vgx[case_b], vgy[case_b]))
                _add_new(cands)

            # --- Case C: gap-filling wavefront ---
            # Add a rounded candidate only if it neighbours already-estimated terrain (wavefront
            # expansion inward) and still has at least one unestimated direction.
            if np.any(case_c):
                # rounded_x/y : (K,) float – cell's world position snapped to the nearest coarse-grid point
                rounded_x = R * np.round(vgx[case_c] / R)
                rounded_y = R * np.round(vgy[case_c] / R)
                # uniq_c : (M, 2) float – deduplicated rounded candidate positions
                uniq_c = np.unique(np.stack([rounded_x, rounded_y], axis=1), axis=0)
                cands, crg, ccg = _base_filter(uniq_c)  # (Q, 2), (Q,), (Q,)

                if len(cands) > 0:
                    # Build the 4 floor-aligned neighbours for each candidate to check whether
                    # the estimation wavefront has already reached this vicinity.
                    offsets = np.array([[-R, -R], [-R, 0.], [0., -R], [0., 0.]])  # (4, 2)
                    nbr     = cands[:, None, :] + offsets[None]  # (Q, 4, 2) – 4 neighbours per candidate
                    nbr_rg  = ((self.map_xmax - nbr[:, :, 0]) / self.map_resolution).astype(int)  # (Q, 4)
                    nbr_cg  = ((self.map_ymax - nbr[:, :, 1]) / self.map_resolution).astype(int)  # (Q, 4)
                    # nbr_valid : (Q, 4) bool – which neighbours fall inside the global map
                    nbr_valid = ((nbr_rg >= 0) & (nbr_rg < self.map_size_rows)
                                 & (nbr_cg >= 0) & (nbr_cg < self.map_size_cols))

                    # near_preest : (Q,) bool – True if any in-bounds neighbour has at least one
                    # estimated traversability direction (i.e. the wavefront has reached nearby)
                    near_preest = np.zeros(len(cands), dtype=bool)
                    for j in range(4):
                        v = nbr_valid[:, j]  # (Q,) bool – which candidates have a valid j-th neighbour
                        if not np.any(v):
                            continue
                        # Clamp out-of-bounds indices to 0 for safe array access; masked out by `v` below
                        rg_j = np.where(v, nbr_rg[:, j], 0)
                        cg_j = np.where(v, nbr_cg[:, j], 0)
                        # has_nonnull : (Q,) bool – neighbour j is valid AND has some estimated direction
                        has_nonnull = v & np.any(self.TravMap_MetaInfo[rg_j, cg_j] != NINIT_T, axis=-1)
                        near_preest |= has_nonnull

                    # Discard candidates not adjacent to the estimation wavefront
                    cands = cands[near_preest]
                    crg   = crg[near_preest]
                    ccg   = ccg[near_preest]

                    if len(cands) > 0:
                        # any_uninit : (Q,) bool – candidate still has at least one unestimated direction
                        any_uninit = np.any(self.TravMap_MetaInfo[crg, ccg] == NINIT_T, axis=-1)
                        _add_new(cands[any_uninit])

            # Commit all new candidates in one allocation instead of one np.append per point
            if new_pts:
                self.TravUpdate_Pts = np.vstack([self.TravUpdate_Pts, np.array(new_pts)])

            # ---------------------------------------------------------------
            # Step 5: Write the new elevation values into the global map.
            # ---------------------------------------------------------------

            self.ElevationMap[vrg, vcg] = local_vals
            # newly_init : (N,) bool – cells written for the very first time; promote their status
            newly_init = elev_meta == NINIT_E
            self.ElevMap_MetaInfo[vrg[newly_init], vcg[newly_init]] = ElevmapInfo_IntEnum.ELEV_MEASURED.value




            # ---------------------------------------------------------------
            # Step 6: Prune TravUpdate_Pts of stale entries.
            #
            # Remove any queued candidate whose elevation is NaN in the
            # (now-updated) global map — Travformer inference requires valid
            # elevation data.
            # ---------------------------------------------------------------

            deleting_indices = []
            for i in range(self.TravUpdate_Pts.shape[0]):

                estimating_gridpt = self.TravUpdate_Pts[i]  # (2,) float – world (x, y) of this candidate
                estimating_pt_global_row = int( (self.map_xmax - estimating_gridpt[0])/self.map_resolution )
                estimating_pt_global_col = int( (self.map_ymax - estimating_gridpt[1])/self.map_resolution )

                if self.is_TraversabilityMap_built and np.isnan(self.ElevationMap[estimating_pt_global_row, estimating_pt_global_col]):
                    # This pt is nan. no need to preestimate. turning off only when this is the first time to build preest map
                    deleting_indices.append(i)

            self.TravUpdate_Pts = np.delete(self.TravUpdate_Pts, deleting_indices, axis=0)





            if visualize_map:
                # Update the traversability map
                print("\nUpdating traversability map......")
                start_time = time()
                print(" Current robot position: {:.1f}, {:.1f}, {:.0f}".format(self.robot_x, self.robot_y, np.rad2deg(self.robot_heading)))
                print(" Global map range:      x-axis: ", self.map_xmin, self.map_xmax, "y-axis: ", self.map_ymin, self.map_ymax)
                print(" Estimating", self.TravUpdate_Pts.shape[0], " points in total...")
                

            self.get_travmap(self.TravUpdate_Pts, 0, 0, 0, 0)

            if visualize_map:
                print("Traversability map updated. Total time of estimation: ", time()-start_time, "\n")

            theta_to_goal = np.arctan2(self.goal_y - self.robot_y, self.goal_x - self.robot_x)
            theta_robot = Utils.wrap_to_pi(self.robot_heading)
            if visualize_map and self.is_TraversabilityMap_built:
                self.visualize_maps(theta_robot, path_plan)







# TODO: Handle travmap
# right size for trav update


    def visualize_maps(self, map_theta, path_plan = None, save_dir = None):
        # save_dir: when given, save the Elevation and Traversability figures as PDFs into
        # this directory (used by the one-time experiment snapshot dump).

        theta_layer = self.get_theta_layer(map_theta)


        path_plan = np.array(path_plan).reshape(-1,2) if path_plan is not None else None
        if path_plan is not None and path_plan.shape[0] > 0:
            path_plan_InGridMap = np.zeros_like(path_plan)
            for i in range(path_plan.shape[0]):
                path_plan_InGridMap[i,:] = self.xy2grid(path_plan[i,0], path_plan[i,1])




        # ####################################### Elevation map #######################################
        # Only visualize elevation map if it exists
        if self.ElevationMap is not None:
            if not hasattr(self, '_mapfig_elev'):
                self._mapfig_elev = MapFigure(1, (0, 0, 480, 420))
            self._mapfig_elev.get_or_create()
            plt.imshow(self.ElevationMap[:,:], cmap='viridis')
            plt.title("Global Elevation Map", fontproperties=FONT_PROP, fontsize=FS_TITLE)
            if path_plan is not None and path_plan.shape[0] > 0:
                plt.plot([x[1] for x in path_plan_InGridMap], [x[0] for x in path_plan_InGridMap], "Dr", markersize=3)
                plt.plot([x[1] for x in path_plan_InGridMap], [x[0] for x in path_plan_InGridMap], '-r', linewidth=1.5)
                plt.plot(path_plan_InGridMap[0,1], path_plan_InGridMap[0,0], marker="D",color=(0,1,1), markersize=6)
                plt.plot(path_plan_InGridMap[-1,1], path_plan_InGridMap[-1,0], marker="*",color=(1,1,0), markersize=10)

            self._mapfig_elev.apply_colorbar('Height (m)')
            self._mapfig_elev.apply_axes_style(
                self.ElevationMap.shape[:2],
                (self.map_xmin, self.map_xmax),
                (self.map_ymin, self.map_ymax)
            )
            # plt.minorticks_on()
            # One-time experiment snapshot: save the elevation figure into the log dir.
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, 'World Elevation Map' + '.pdf'), format=PLOT_FORMAT, dpi=PLOT_DPI, bbox_inches='tight')
            else:
                # Interactive use only: pop up the window. Skipped in headless snapshot mode
                # (save_dir given) so the one-time experiment dump never opens GUI windows.
                self._mapfig_elev.show()






        ####################################### Traversability map #######################################
        if not hasattr(self, '_mapfig_trav'):
            self._mapfig_trav = MapFigure(2, (0, 660, 480, 420))
        self._mapfig_trav.get_or_create()

        plt.imshow(self.TraversabilityMap[:,:,theta_layer,self.viz_channel], cmap='coolwarm_r', vmin=self.viz_vmin, vmax=self.viz_vmax)
        plt.title(f"Traversability Map", fontproperties=FONT_PROP, fontsize=FS_TITLE)
        self._mapfig_trav.apply_colorbar(self.viz_cbar_label)

        if path_plan is not None and path_plan.shape[0] > 0:
            # plt.plot([x[1] for x in path_plan_InGridMap], [x[0] for x in path_plan_InGridMap], "Dr", markersize=3)
            plt.plot([x[1] for x in path_plan_InGridMap], [x[0] for x in path_plan_InGridMap], '--', color='lime', linewidth=4, label = 'Global Path Plan')
            # plt.plot(path_plan_InGridMap[0,1], path_plan_InGridMap[0,0], marker="D",color=(0,1,1), markersize=6)
            # plt.plot(path_plan_InGridMap[-1,1], path_plan_InGridMap[-1,0], marker="*",color=(1,1,0), markersize=10)
            pass

        self._mapfig_trav.apply_axes_style(
            self.TraversabilityMap.shape[:2],
            (self.map_xmin, self.map_xmax),
            (self.map_ymin, self.map_ymax),
            grid_kwargs={}
        )
        self._mapfig_trav.apply_legend(loc='upper left')
        # One-time experiment snapshot: save the traversability figure into the log dir.
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'Traversability Map' + '.pdf'), format=PLOT_FORMAT, dpi=PLOT_DPI, bbox_inches='tight')
        else:
            # Interactive use only: pop up the window. Skipped in headless snapshot mode
            # (save_dir given) so the one-time experiment dump never opens GUI windows.
            self._mapfig_trav.show()


    def get_travmap(self, estimating_xypts_set, xmin, xmax, ymin, ymax):
        self._compute_travmap(estimating_xypts_set, xmin, xmax, ymin, ymax)

        if not self.is_TraversabilityMap_built:
            self.is_TraversabilityMap_built = True
            if hasattr(self, 'backend'):
                self.backend.set_trav_map_built(True)

        self.TravUpdate_Pts = np.empty((0, 2))

    def _compute_travmap(self, estimating_xypts_set, xmin, xmax, ymin, ymax):
        raise NotImplementedError("Subclasses must implement _compute_travmap")






                



    
    def _compute_patch_indices(self, x_center, y_center, yaw, width, height):
        """Compute the map row/col indices for a yaw-aligned patch centered at (x_center, y_center).
        SHOULD BE APPLIED ONLY FOR FUll SIZE MAP

        width: local x-axis extent (meters)
        height: local y-axis extent (meters)
        Returns rows, cols, valid, (half_width_grid, half_height_grid).

        N = 2 * (half_width_grid) * (half_height_grid)
        rows: 2D numpy array of shape (N, 1) with the row indices of the patch
        cols: 2D numpy array of shape (N, 1) with the column indices of the patch
        valid: 1D boolean array of shape (N,) indicating whether each index is valid (in the map)
        (half_width_grid, half_height_grid): tuple of half the patch size in grid cells
        """
        row_center = int((self.map_xmax - x_center) / self.map_resolution)
        col_center = int((self.map_ymax - y_center) / self.map_resolution)
        givenpoint_tf_grid = tf3.affines.compose(
            np.array([row_center, col_center, 0]),
            tf3.euler.euler2mat(0, 0, yaw, 'sxyz'),
            np.ones(3)
        )
        half_width_grid  = np.round(0.5 * width  / self.map_resolution).astype(int)
        half_height_grid = np.round(0.5 * height / self.map_resolution).astype(int)
        local_patch_index = np.array([
            [i, j, 0]
            for i in range(-half_width_grid, half_width_grid)
            for j in range(-half_height_grid, half_height_grid)
        ])
        local_patch_index_homogeneous = np.hstack(
            (local_patch_index, np.ones((local_patch_index.shape[0], 1)))
        )
        local_patch_index_transformed = np.round(
            np.dot(givenpoint_tf_grid, local_patch_index_homogeneous.T).T[:, :2]
        ).astype(int)

        rows = local_patch_index_transformed[:, 0]
        cols = local_patch_index_transformed[:, 1]
        valid = (rows >= 0) & (rows < self.map_size_rows) & \
                (cols >= 0) & (cols < self.map_size_cols)

        return rows, cols, valid, (half_width_grid, half_height_grid)

    def get_patch_in_map(self, x_center, y_center, yaw, width, height, map):
        """
        Get a local patch of the global elevation map, centered at (x_center, y_center). yaw-aligned direction.

        SHOULD BE APPLIED ONLY FOR FUll SIZE MAP

        x_center, y_center: in the world frame
        width: local x-axis extent in meters
        height: local y-axis extent in meters
        yaw: in radian
        map: numpy array with at least 2 dimensions (H, W, ...)

        Returns:
        local_patch: array of shape (half_width_grid*2, half_height_grid*2, ...) matching extra map dimensions
        ismapnan: a boolean indicating whether the local patch contains any nan value
        """
        ismapnan = False

        if map is not None:
            rows, cols, valid, (half_width_grid, half_height_grid) = \
                self._compute_patch_indices(x_center, y_center, yaw, width, height)

            safe_rows = np.where(valid, rows, 0)
            safe_cols = np.where(valid, cols, 0)
            gathered = map[safe_rows, safe_cols]  # shape: (N,) or (N, ...) for extra dims
            extra_dims = gathered.shape[1:]
            valid_broadcast = valid.reshape(-1, *([1] * len(extra_dims)))
            nan_fill = np.full(extra_dims, np.nan, dtype=gathered.dtype)
            values = np.where(valid_broadcast, gathered, nan_fill)

            ismapnan = bool(np.any(np.isnan(values)))
            local_patch = values.reshape((half_width_grid * 2, half_height_grid * 2) + extra_dims)

            return local_patch, ismapnan

    def set_patch_in_map(self, x_center, y_center, yaw, width, height, map, patch):
        """
        Write a local patch into the global map at (x_center, y_center), yaw-aligned.
        patch: 2D (or higher-D) numpy array or scalar value.
        SHOULD BE APPLIED ONLY FOR FUll SIZE MAP

        Uses an inverse-transform approach so that every global map cell whose
        centre falls inside the rotated patch rectangle is written.  Cells that
        the forward-transform would have missed (rotation-induced holes) are
        filled by bilinear interpolation (float patches) or nearest-neighbour
        (integer patches).
        """
        if map is None or patch is None:
            return

        row_center = int((self.map_xmax - x_center) / self.map_resolution)
        col_center = int((self.map_ymax - y_center) / self.map_resolution)
        hw = int(np.round(0.5 * width  / self.map_resolution))
        hh = int(np.round(0.5 * height / self.map_resolution))

        # Rotation matrix (same z-axis convention as _compute_patch_indices)
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        R     = np.array([[cos_yaw, -sin_yaw],
                          [sin_yaw,  cos_yaw]])
        R_inv = R.T   # [[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]]

        # Axis-aligned bounding box of the rotated patch in global map space
        corners = np.array([[-hw, -hh], [hw, -hh], [-hw, hh], [hw, hh]], dtype=float)
        corners_g = (R @ corners.T).T + [row_center, col_center]

        row_min = max(int(np.floor(corners_g[:, 0].min())), 0)
        row_max = min(int(np.ceil( corners_g[:, 0].max())), self.map_size_rows - 1)
        col_min = max(int(np.floor(corners_g[:, 1].min())), 0)
        col_max = min(int(np.ceil( corners_g[:, 1].max())), self.map_size_cols - 1)

        if row_min > row_max or col_min > col_max:
            return

        # Dense grid of every global cell in the bounding box
        gr, gc = np.meshgrid(
            np.arange(row_min, row_max + 1),
            np.arange(col_min, col_max + 1),
            indexing='ij'
        )

        # Inverse-rotate to local patch coordinates
        drow = (gr - row_center).astype(float)
        dcol = (gc - col_center).astype(float)
        local_i = R_inv[0, 0] * drow + R_inv[0, 1] * dcol
        local_j = R_inv[1, 0] * drow + R_inv[1, 1] * dcol

        # Keep only cells whose centre lies inside the patch region
        # (half-open interval matches the original range(-hw, hw))
        inside = (local_i >= -hw) & (local_i < hw) & \
                 (local_j >= -hh) & (local_j < hh)

        if not isinstance(patch, np.ndarray):
            map[gr[inside], gc[inside]] = patch
            return

        if patch.shape[0] != hw * 2 or patch.shape[1] != hh * 2:
            raise ValueError(
                f"Patch shape {patch.shape[:2]} does not match expected ({hw * 2}, {hh * 2})."
            )

        # Continuous patch indices (0-based) for interpolation
        pi = local_i[inside] + hw   # in [0, 2*hw)
        pj = local_j[inside] + hh   # in [0, 2*hh)
        coords = np.array([pi, pj])

        # Bilinear for floats, nearest-neighbour for integers
        interp_order = 0 if np.issubdtype(patch.dtype, np.integer) else 1
        extra_dims = patch.shape[2:]

        def _interp_slice(slc):
            vals = map_coordinates(slc.astype(float), coords,
                                   order=interp_order, mode='nearest')
            if np.issubdtype(patch.dtype, np.integer):
                vals = np.round(vals).astype(patch.dtype)
            return vals

        if len(extra_dims) == 0:
            map[gr[inside], gc[inside]] = _interp_slice(patch)
        else:
            out = np.empty((int(inside.sum()),) + extra_dims, dtype=patch.dtype)
            for idx in np.ndindex(extra_dims):
                out[(slice(None),) + idx] = _interp_slice(patch[(Ellipsis,) + idx])
            map[gr[inside], gc[inside]] = out





    def get_theta_layer(self, theta):
        return Utils.heading_to_layer(theta, self.TraversabilityMap_theta_min,
                                      self.TraversabilityMap_theta_resolution,
                                      self.TraversabilityMap_size_layers)

    def xy2grid(self, x, y):
        
        row = int( (self.map_xmax - x)/self.map_resolution )
        col = int( (self.map_ymax - y)/self.map_resolution )
        
        if row < 0 or row > self.map_size_rows:
            print(f"Out of range. x: {x}, y: {y}, row: {row}, col: {col}")
            return None, None
        elif row == self.map_size_rows:
            row = self.map_size_rows - 1

        if col < 0 or col > self.map_size_cols:
            print(f"Out of range. x: {x}, y: {y}, row: {row}, col: {col}")
            return None, None
        elif col == self.map_size_cols:
            col = self.map_size_cols - 1

        return row, col

    def grid2xy(self, row, col):
        x = self.map_xmax - row * self.map_resolution
        y = self.map_ymax - col * self.map_resolution
        return x, y




    def get_cmd_limits(self):

        # TODO: there could be latency issue for robot's position when called by MPC in the future.


        if self.map_resolution is None or self.ElevationMap is None or self.TraversabilityMap is None:
            return self.toorisky_v, self.toorisky_w
        
        cmd_v_limit = self.maximum_cmd_v
        cmd_w_limit = self.maximum_cmd_w


        discretization = safecmd_sampling_dist
        num_points = int(lookahead_distance/discretization)
        angle_tolerance = np.deg2rad(lookahead_angle)

        counter = 0
        for i in range(num_points):
            delta_dist = i * discretization
            
            for j in range(-1, 2):
                theta = self.robot_heading + j * angle_tolerance
                theta = Utils.wrap_to_pi(theta)

                x = self.robot_x + delta_dist * np.cos(theta)
                y = self.robot_y + delta_dist * np.sin(theta)

                row = int( (self.map_xmax - x)/self.map_resolution )
                col = int( (self.map_ymax - y)/self.map_resolution )
                

                theta_layer = self.get_theta_layer(theta)

                if row < 0 or row >= self.map_size_rows or col < 0 or col >= self.map_size_cols:
                    continue

                cmd_v = self.TraversabilityMap[row, col, theta_layer, 0]
                cmd_w = self.TraversabilityMap[row, col, theta_layer, 1]


                cmd_v_limit = ( cmd_v_limit * counter + cmd_v )/(counter + 1)
                cmd_w_limit = ( cmd_w_limit * counter + cmd_w )/(counter + 1)
                counter += 1

                # if cmd_v != 0:
                #     cmd_v_limit = np.min([cmd_v_limit, cmd_v])
                #     cmd_w_limit = np.min([cmd_w_limit, cmd_w])
                # else:
                #     print("You are near obstacles")
        

        return cmd_v_limit, cmd_w_limit


    def get_waypoint(self, robot_x, robot_y, robot_heading, total_lookahead_time, cmd_v_limit, cmd_w_limit, path):
        """
        This function takes in the current position of the robot and returns the next waypoint (in the global coordinate system)
        that the robot should move to. The waypoint is a point on the global path at exactly reachable_radius distance from
        the robot, where reachable_radius = total_lookahead_time * cmd_v_limit.

        The function finds the path node nearest to the robot, advances past any nodes that are behind the robot, then
        interpolates to the exact point where the path crosses the reachable_radius circle. If the remaining path is shorter
        than reachable_radius, the last path node (clamped to reachable_radius) is returned.

        Parameters
        ----------
        robot_x : float
            The current x-coordinate of the robot in the global coordinate system.
        robot_y : float
            The current y-coordinate of the robot in the global coordinate system.
        robot_heading : float
            The current heading of the robot in radians (reserved for future use).
        total_lookahead_time : float
            Total lookahead time in seconds. The local planner (main.py::_select_waypoint) pins
            cmd_v_limit/cmd_w_limit to 1.0 and passes cfg.waypoint_lookahead_distance here, so
            reachable_radius below collapses to that fixed distance in metres.
        cmd_v_limit : float
            Maximum linear velocity limit.
        cmd_w_limit : float
            Maximum angular velocity limit (reserved for future use).
        path : array_like
            The global path as a numpy array of shape (N, 2) where each row is [x, y].

        Returns
        -------
        waypoint : array_like or False
            The coordinates of the next waypoint in the global coordinate system, or False if no path is available
            or the traversability map is not yet built.
        """
        path = np.array(path)  # (N, 2) array of [x, y] path nodes in global frame
        if path.shape[0] == 0:
            print(' Getting Waypoint: No path received')
            return False

        # Clamp velocity limits to sensible minimums to avoid zero-radius degenerate cases
        cmd_v_limit = max(cmd_v_limit, 0.1)
        # reachable_radius: how far (meters) the robot can travel within total_lookahead_time at cmd_v_limit
        reachable_radius = total_lookahead_time * cmd_v_limit

        def _waypoint_at_radius(target_node):
            # Interpolate from robot toward target_node, clamped to reachable_radius.
            # Returns a point at reachable_radius distance if target is beyond it; otherwise returns target itself.
            robot_pos = np.array([robot_x, robot_y])
            dist = math.hypot(float(target_node[0]) - robot_x, float(target_node[1]) - robot_y)
            if dist > reachable_radius:
                return robot_pos + (target_node - robot_pos) * (reachable_radius / dist)
            return target_node

        # --- Step 1: Find the path node closest to the robot ---
        # waypoint_node_index: index of the path node with minimum Euclidean distance to the robot; used as the starting search point
        waypoint_node_index = int(np.argmin([math.hypot(path[nd,0] - robot_x, path[nd,1] - robot_y)
                                        for nd in range(path.shape[0])]))

        # If the nearest node is already the last node, aim toward the goal clamped to reachable_radius
        if waypoint_node_index >= path.shape[0] - 1:
            # Interpolate toward path[-1] at exactly reachable_radius instead of
            # teleporting to it, so the waypoint stays at a constant lookahead distance.
            return _waypoint_at_radius(path[-1, :])

        # --- Step 2: Find an intermediate goal node to determine the forward direction ---
        # intermediate_goal_radius: twice the reachable radius, used to identify a node clearly ahead of the robot
        min_cmd = max(cmd_v_limit, 0.25)
        intermediate_goal_radius = 2 * total_lookahead_time * min_cmd
        intermediate_goal_node_index = waypoint_node_index
        for node_idx in range(waypoint_node_index,path.shape[0]):
            if math.hypot(path[node_idx,0] - robot_x, path[node_idx,1] - robot_y) >= intermediate_goal_radius:
                intermediate_goal_node_index = node_idx
                break
        # robot_to_intermediate_goal_distance: distance from robot to the chosen intermediate goal node
        robot_to_intermediate_goal_distance = math.hypot(path[intermediate_goal_node_index,0] - robot_x, path[intermediate_goal_node_index,1] - robot_y)

        # --- Step 3: Advance waypoint_node_index past nodes that are "behind" the robot ---
        # A node is considered behind the robot if it is farther from the intermediate goal than the robot itself is.
        # This handles the case where the nearest node is behind the robot's current position along the path.
        while waypoint_node_index < path.shape[0] - 1:

            # wp_to_intermediate_goal_distance: distance from candidate waypoint node to the intermediate goal
            wp_to_intermediate_goal_distance = math.hypot(path[waypoint_node_index,0] - path[intermediate_goal_node_index,0], path[waypoint_node_index,1] - path[intermediate_goal_node_index,1])

            if wp_to_intermediate_goal_distance > robot_to_intermediate_goal_distance:
                # This waypoint is behind the robot. Proceed to the next node
                waypoint_node_index += 1

            else:
                break

        # If advancing pushed us to the last node, aim toward the end of the path
        if waypoint_node_index >= path.shape[0] - 1:
            print(' Getting Waypoint: Getting the end of the path')
            return _waypoint_at_radius(path[-1, :])

        # --- Step 4: Find the exact waypoint on the reachable_radius boundary ---
        # Only executed when the traversability map is available; returns False otherwise (map not yet ready).
        if self.is_TraversabilityMap_built:

            # Scan forward from waypoint_node_index to find the first node that exceeds reachable_radius
            for node in range(waypoint_node_index, path.shape[0]):

                robot_to_this_node_distance = math.hypot(path[node,0] - robot_x, path[node,1] - robot_y)
                if robot_to_this_node_distance > reachable_radius: # we got to a node that is further than the reachable radius

                    if node == waypoint_node_index: # the nearest hopeful node is already further than the reachable radius
                        return _waypoint_at_radius(path[node, :])

                    # This means that the found node is not the nearest hopeful node.
                    # We need to get intermediate point between this found node and its previous node
                    previous_node = node - 1
                    robot_to_previous_node_distance = math.hypot(path[previous_node,0] - robot_x, path[previous_node,1] - robot_y)
                    if robot_to_previous_node_distance > reachable_radius:
                        raise ValueError(
                            f'[DEBUG INFO] The previous node: {path[previous_node,:]} with distance {robot_to_previous_node_distance}'
                            f' is further than the reachable radius: {reachable_radius}; '
                            f'The found node: {path[node,:]} with distance {robot_to_this_node_distance}'
                            f' is further than the reachable radius: {reachable_radius}; '
                            f'But it is not the nearest hopeful node. This should not happen.'
                        )
                    else: # robot_to_previous_node_distance < reachable_radius
                        # Interpolate to the exact reachable_radius boundary between previous and current node
                        # alpha: fractional position along the segment [previous_node, node] where the boundary is crossed
                        alpha = (reachable_radius - robot_to_previous_node_distance) / (robot_to_this_node_distance - robot_to_previous_node_distance)
                        # This is not exactly accurate equation but good approximation
                        return path[previous_node,:] + (path[node,:] - path[previous_node,:]) * alpha

            # All remaining nodes are within reachable_radius; aim at the final node clamped to reachable_radius
            return _waypoint_at_radius(path[-1, :])

        return False

    
    def pose_callback(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        quat = [q.x, q.y, q.z, q.w]
        roll, pitch, yaw = tf_trans.euler_from_quaternion(quat)
        msg_yaw = yaw
        self.robot_x = p.x
        self.robot_y = p.y
        self.robot_roll = Utils.wrap_to_pi(roll)
        self.robot_pitch = Utils.wrap_to_pi(pitch)
        self.robot_heading = Utils.wrap_to_pi(yaw)
        # rospy.loginfo_throttle(1.0, "[pose_callback] x: %.2f, y: %.2f, yaw: %.2f rad", self.robot_x, self.robot_y, self.robot_heading)


    def robo_centric_map_callback(self, msg):

        if not self.is_RobocentricMap_received:
            self.is_RobocentricMap_received = True
        # start = time()
        # rospy.loginfo("Received GridMap message")

        # rospy.loginfo("Header:\n%s" % msg.info.header)

        # Get the dimensions of the map
        self.map_resolution = msg.info.resolution
        self.RobocentricMap_LengthY = msg.info.length_y
        self.RobocentricMap_LengthX = msg.info.length_x

        self.RobocentricMap_Rows = int(self.RobocentricMap_LengthY / self.map_resolution)
        self.RobocentricMap_Cols = int(self.RobocentricMap_LengthX / self.map_resolution)
        # rospy.loginfo("Map dimensions: %dx%d" % (self.RobocentricMap_Rows, self.RobocentricMap_Cols))

        # Build layer arrays before acquiring the lock to minimize lock hold time
        layers = msg.layers
        new_center_pos = (msg.info.pose.position.x, msg.info.pose.position.y, msg.info.pose.position.z)
        new_center_rot = (msg.info.pose.orientation.x, msg.info.pose.orientation.y, msg.info.pose.orientation.z, msg.info.pose.orientation.w)
        new_layer_data = {}
        for i, layer in enumerate(layers):
            if not layer in self.which_layer:
                continue
            data_layer = np.array(msg.data[i].data)
            new_layer_data[layer] = data_layer.reshape((self.RobocentricMap_Rows, self.RobocentricMap_Cols)).transpose() #Don't know why but transpose is needed. row indexing

        # Atomically publish the new position and layer data so Update_map always
        # sees a consistent pair (same message's position + same message's data).
        with self._map_data_lock:
            self.RobocentricMap_CenterPosOffset_Worldfr = new_center_pos
            self.RobocentricMap_CenterRotOffset_Worldfr = new_center_rot
            self.layer_data.update(new_layer_data)
        # rospy.loginfo("Map position: %s" % str(self.RobocentricMap_CenterPosOffset_Worldfr))
        # rospy.loginfo("Layers: %s" % layers)

        # rospy.loginfo("Received GridMap message in %f seconds" % (time() - start))



























                                        








class CMDbasedMap(BaseMap):
    def __init__(self, env_xmin, env_xmax, env_ymin, env_ymax, goal_x, goal_y, which_layer, preest_update_resolution, instab_limit, load_Travformer = True):
        super().__init__(env_xmin, env_xmax, env_ymin, env_ymax, goal_x, goal_y, which_layer, preest_update_resolution, instab_limit, load_Travformer)


        self.InitialGuess_cmd_v = cfg.default_safe_cmd_v
        self.InitialGuess_cmd_w = cfg.default_safe_cmd_w


        # trav as velcity-based method specific
        self.toorisky_v = cfg.toorisky_v
        self.toorisky_w = cfg.toorisky_w
        self.v_res = self.maximum_cmd_v / cfg.num_commands
        self.w_res = self.maximum_cmd_w / cfg.num_commands
        # TreatObs_cmd_v: minimum speed bin considered safe; patches failing at this speed keep toorisky_v sentinel
        self.TreatObs_cmd_v = cfg.TreatObs_cmd_v
        if self.TreatObs_cmd_v > self.maximum_cmd_v:
            raise ValueError(
                f"TreatObs_cmd_v ({self.TreatObs_cmd_v}) must be <= maximum_cmd_v ({self.maximum_cmd_v}). "
                "No speed bins would exist and every cell would be marked toorisky."
            )

        self.instability_limit = instab_limit # = \delta_limit
        self.interp_sigmoid_k = cfg.interp_sigmoid_k  # sharpness of S-curve interpolation between quadrant corners (from planning_config.yaml)

        # Visualization
        self.viz_channel = 0  # default channel
        self.viz_vmin = 0.0
        self.viz_vmax = self.maximum_cmd_v
        self.viz_title = "Traversability Map"
        self.viz_cbar_label = "Linear Velocity (m/s)"

        # ── Shared-memory graph (Part 2) ─────────────────────────────────────
        # Owned only by the writer-side map when global_planner_type == "Astar";
        # created lazily by _build_shared_graph() once the coarse lattice geometry
        # is known. Left None everywhere else (base map, reader proxy, other
        # planner types), which disables every graph hook below.
        self.graph_writer = None
        # Retained coarse estimation-lattice CENTER bounds (kx_min*R ... kx_max*R),
        # the expected-update domain used to classify write-back points. Distinct
        # from create_grid's half-cell-extended storage bounds. Set in
        # _build_shared_graph().
        self._graph_lattice_bounds = None      # (lxmin, lxmax, lymin, lymax)
        # float32-roundoff tolerance for stored-coordinate equality checks; set
        # from the actual snapped bounds in _build_shared_graph().
        self._graph_coord_tol = None


    def _initialize_traversability_map_values(self):
        """Initialize neutral commands, then mark the non-inferable map border too-risky."""
        super()._initialize_traversability_map_values()

        if (self.map_resolution is None or not np.isfinite(self.map_resolution)
                or self.map_resolution <= 0.0):
            raise ValueError(
                "_initialize_traversability_map_values requires a finite positive "
                f"map_resolution, got {self.map_resolution}")
        if not np.isfinite(self.local_patch_size) or self.local_patch_size <= 0.0:
            raise ValueError(
                "_initialize_traversability_map_values requires a finite positive "
                f"local_patch_size, got {self.local_patch_size}")

        # half_patch_cells is the half-side of each TravFormer elevation input in
        # fine-map cells. Border cells closer than this cannot provide a complete
        # patch, so every heading layer starts at the exact too-risky sentinel.
        half_patch_cells = int(np.round(0.5 * self.local_patch_size / self.map_resolution))
        if half_patch_cells <= 0:
            return

        rows, cols = self.TraversabilityMap.shape[:2]
        row_indices = np.arange(rows)[:, None]  # (rows, 1) fine-map row index
        col_indices = np.arange(cols)[None, :]  # (1, cols) fine-map column index
        boundary_mask = ((row_indices < half_patch_cells)
                         | (row_indices >= rows - half_patch_cells)
                         | (col_indices < half_patch_cells)
                         | (col_indices >= cols - half_patch_cells))  # (rows, cols)
        self.TraversabilityMap[:, :, :, 0][boundary_mask] = self.toorisky_v
        self.TraversabilityMap[:, :, :, 1][boundary_mask] = self.toorisky_w
        self.TravMap_MetaInfo[boundary_mask] = TravmapInfo_IntEnum.TRAV_Estimated.value


    # =====================================================================
    # Shared-memory graph integration (Part 2) — one contiguous region.
    #
    # Responsibilities, all writer-side and all no-ops when self.graph_writer
    # is None (base map, reader proxy, or global_planner_type != "Astar"):
    #   _build_shared_graph()        : lattice-aligned graph construction
    #   _close_shared_graph()        : shutdown cleanup
    #   _graph_layer_too_risky_mask()/_graph_node_has_too_risky() : the single
    #                                  too-risky predicate (shared with tests)
    #   _apply_graph_update()        : mirror one map-update's coarse-lattice
    #                                  cmd_v/cmd_w/meta into node payloads and
    #                                  maintain the aggregate too-risky topology
    #
    # The graph is a MIRROR of the existing TraversabilityMap / TravMap_MetaInfo
    # coarse-lattice state, not a second estimator: the capture site (Step 9)
    # carries the exact cmd_v/cmd_w/meta it writes to
    # the dense grid into the record (direct write, no read-back), and
    # _apply_graph_update never recomputes values or heading-layer selection.
    # =====================================================================

    def _build_shared_graph(self):
        """
        Construct the lattice-aligned shared graph exactly once, when
        global_planner_type == "Astar". No-op for every other planner type
        (leaves self.graph_writer = None) and if already built.

        Node centers land on the estimation lattice (world multiples of
        TravUpdate_XYResolution clipped to the map bounds) via create_grid's
        cell-edge bounds, so runtime write-back points resolve by coordinate.
        No grid-index-to-slot logic is created or retained.
        """
        # Idempotent: build at most once.
        if self.graph_writer is not None:
            return
        # Existing planner selector gates construction (no graph-specific flag).
        if str(cfg.get('global_planner_type', 'Astar')) != "Astar":
            return
        # Geometry must be known (Initilize_map has run and set map bounds/res).
        if self.map_resolution is None or not self.is_ElevationMap_built:
            raise SharedGraphError(
                "_build_shared_graph called before map geometry is known "
                f"(map_resolution={self.map_resolution}, "
                f"is_ElevationMap_built={self.is_ElevationMap_built})")
        # The module-level GRAPH_LAYOUT hard-codes _GRAPH_HEADING_LAYERS heading
        # layers; refuse to build if the live map uses a different count.
        if int(self.TraversabilityMap_size_layers) != _GRAPH_HEADING_LAYERS:
            raise SharedGraphError(
                f"_build_shared_graph: map has {self.TraversabilityMap_size_layers} "
                f"heading layers but GRAPH_LAYOUT is fixed at {_GRAPH_HEADING_LAYERS}; "
                f"update GRAPH_LAYOUT before enabling the graph")

        R = float(self.TravUpdate_XYResolution)   # coarse estimation-lattice resolution (m)
        if not (R > 0.0):
            raise SharedGraphError(f"_build_shared_graph: non-positive resolution R={R}")

        # ---- retained integer lattice index range within the map bounds -----
        # Lattice points are world multiples of R; the extreme multiples that
        # fall inside [map_xmin, map_xmax] x [map_ymin, map_ymax] define the
        # lattice. round-then-clip removes float noise near an exact multiple.
        def _k_min(lo):
            # smallest integer k with k*R >= lo; the -1e-9 nudge keeps an lo that
            # is already an exact multiple of R from rounding up a spurious cell.
            return math.ceil(lo / R - 1e-9)

        def _k_max(hi):
            # largest integer k with k*R <= hi (symmetric +1e-9 nudge).
            return math.floor(hi / R + 1e-9)

        kx_min = _k_min(self.map_xmin)
        kx_max = _k_max(self.map_xmax)
        ky_min = _k_min(self.map_ymin)
        ky_max = _k_max(self.map_ymax)
        rows_expected = kx_max - kx_min + 1        # number of distinct x lattice points
        cols_expected = ky_max - ky_min + 1        # number of distinct y lattice points
        if rows_expected < 1 or cols_expected < 1:
            raise SharedGraphError(
                f"_build_shared_graph: empty lattice (rows={rows_expected}, "
                f"cols={cols_expected}) for map x[{self.map_xmin}, {self.map_xmax}] "
                f"y[{self.map_ymin}, {self.map_ymax}] at R={R}")

        lattice_xmin = kx_min * R
        lattice_xmax = kx_max * R
        lattice_ymin = ky_min * R
        lattice_ymax = ky_max * R
        # create_grid is cell-centered; extend bounds by half a cell so generated
        # cell centers land exactly on the lattice extrema (see plan §"touchpoints").
        snapped_xmin = lattice_xmin - R / 2.0
        snapped_xmax = lattice_xmax + R / 2.0
        snapped_ymin = lattice_ymin - R / 2.0
        snapped_ymax = lattice_ymax + R / 2.0

        # Guard: the snapped cell-edge bounds must reproduce EXACTLY the expected
        # counts under create_grid's ceil(extent / resolution) rule; a float-noise
        # extra row/col would silently break coordinate resolution later.
        rows_grid = max(1, int(math.ceil((snapped_xmax - snapped_xmin) / R)))
        cols_grid = max(1, int(math.ceil((snapped_ymax - snapped_ymin) / R)))
        if rows_grid != rows_expected or cols_grid != cols_expected:
            raise SharedGraphError(
                f"_build_shared_graph: snapped-bounds grid ({rows_grid}x{cols_grid}) "
                f"does not match the retained lattice index range "
                f"({rows_expected}x{cols_expected}); geometry/config mismatch "
                f"(R={R}, x[{self.map_xmin}, {self.map_xmax}] "
                f"y[{self.map_ymin}, {self.map_ymax}])")

        # float32-roundoff tolerance for stored-coordinate equality (node coords
        # are float32). Scaled by the largest coordinate magnitude in play.
        max_abs = max(abs(snapped_xmin), abs(snapped_xmax),
                      abs(snapped_ymin), abs(snapped_ymax), 1.0)
        coord_tol = 8.0 * float(np.finfo(np.float32).eps) * max_abs

        # ---- construct (n_max/e_max/bucket_size auto-sized by create_grid) ---
        writer = SharedGraphWriter.create_grid(
            xmin=snapped_xmin,
            xmax=snapped_xmax,
            ymin=snapped_ymin,
            ymax=snapped_ymax,
            resolution=R,
            layout=GRAPH_LAYOUT,
            isolate_border_width=0.5 * self.local_patch_size,
            isolate_border_bounds=(self.map_xmin, self.map_xmax,
                                   self.map_ymin, self.map_ymax),
            isolated_node_payloads={
                "cmd_v": float(self.toorisky_v),
                "cmd_w": float(self.toorisky_w),
                "meta": int(TravmapInfo_IntEnum.TRAV_Estimated.value),
            },
        )
        try:
            # ---- post-construction verification against the expected lattice -
            active = np.asarray(writer.node_active[:writer.node_hwm])
            n_active = int(np.count_nonzero(active == 1))
            if n_active != rows_expected * cols_expected:
                raise SharedGraphError(
                    f"_build_shared_graph: graph has {n_active} active nodes, "
                    f"expected {rows_expected * cols_expected} "
                    f"({rows_expected}x{cols_expected})")
            nx = np.asarray(writer.node_x[:writer.node_hwm])[active == 1]
            ny = np.asarray(writer.node_y[:writer.node_hwm])[active == 1]
            if not (np.all(nx >= snapped_xmin - coord_tol)
                    and np.all(nx <= snapped_xmax + coord_tol)
                    and np.all(ny >= snapped_ymin - coord_tol)
                    and np.all(ny <= snapped_ymax + coord_tol)):
                raise SharedGraphError(
                    "_build_shared_graph: some node centers fall outside the "
                    "snapped storage bounds; grid geometry is inconsistent")
            # every node center must sit on a lattice multiple of R (within tol)
            kx = np.round((nx.astype(np.float64) - lattice_xmin) / R)
            ky = np.round((ny.astype(np.float64) - lattice_ymin) / R)
            if not (np.all(np.abs(kx * R + lattice_xmin - nx) <= coord_tol)
                    and np.all(np.abs(ky * R + lattice_ymin - ny) <= coord_tol)):
                raise SharedGraphError(
                    "_build_shared_graph: node centers are not aligned to the "
                    f"estimation lattice (R={R}); float32 roundoff exceeded "
                    f"coord_tol={coord_tol:.3g}")
        except Exception:
            # Never retain a graph that fails verification.
            writer.cleanup()
            raise

        # Commit the owned writer and the runtime-resolution parameters.
        self.graph_writer = writer
        self._graph_lattice_bounds = (lattice_xmin, lattice_xmax,
                                      lattice_ymin, lattice_ymax)
        self._graph_coord_tol = coord_tol

        print(f"[shared-graph] built lattice graph: {n_active} nodes "
              f"({rows_expected}x{cols_expected}), edge_hwm={writer.edge_hwm}, "
              f"n_max={writer.n_max}, e_max={writer.e_max}, R={R}, "
              f"snapped bounds x[{snapped_xmin}, {snapped_xmax}] "
              f"y[{snapped_ymin}, {snapped_ymax}], lookup_radius={writer.default_lookup_radius}. "
              f"Edges store immutable geometric length only (no cmd/status/cost).")

    def _close_shared_graph(self):
        """Idempotently close + unlink the owned graph (shutdown hook)."""
        gw = self.graph_writer
        if gw is not None:
            self.graph_writer = None
            gw.cleanup()

    def _graph_layer_too_risky_mask(self, cmd_v_row, cmd_w_row, meta_row):
        """
        (num_layers,) bool mask: True for each heading layer that has estimated
        meta AND is left at the too-risky command sentinel. This is the
        SINGLE too-risky predicate, reused by _apply_graph_update and the Part 2 tests.

        A layer is too-risky iff meta is TRAV_Estimated and
        cmd_v/cmd_w equal (toorisky_v, toorisky_w). Meta alone is never
        sufficient: the exact sentinel pair is still required.

        cmd_v_row / cmd_w_row : (num_layers,) command values for one node.
        meta_row              : (num_layers,) per-layer TravmapInfo_IntEnum status.
        """
        TRAV_Estimated = int(self.TravmapInfo_IntEnum.TRAV_Estimated.value)
        tv = np.float32(self.toorisky_v)          # float32 sentinel matching stored payload
        tw = np.float32(self.toorisky_w)
        meta = np.asarray(meta_row).astype(np.int64)
        supported_meta = (meta == TRAV_Estimated)
        at_sentinel = ((np.asarray(cmd_v_row, dtype=np.float32) == tv)
                       & (np.asarray(cmd_w_row, dtype=np.float32) == tw))
        return supported_meta & at_sentinel

    def _graph_node_has_too_risky(self, cmd_v_row, cmd_w_row, meta_row):
        """True iff any heading layer has supported meta and the too-risky sentinel pair."""
        return bool(self._graph_layer_too_risky_mask(cmd_v_row, cmd_w_row, meta_row).any())

    def _apply_graph_update(self, graph_records):
        """
        Mirror one map update's coarse-lattice write-back into the shared graph
        and maintain the aggregate too-risky topology invariant.

        graph_records : list of (x, y, theta_layer, cmd_v, cmd_w, meta) captured
            at the point each value is DECIDED — the _compute_travmap Step-9
            loop. The payload is
            carried in the record (direct write); TraversabilityMap /
            TravMap_MetaInfo are NOT read back here, so the graph no longer
            depends on the dense grid (still a mirror, never a second
            calculation — the carried values are the exact values the dense
            write used).

        All predictable coordinate/payload validation completes before the outer
        edit_batch() opens. Any graph-mutation exception propagates out (the
        caller must stop, not keep using the graph): edit_batch() is only a
        reader publication barrier, not a rollback.
        """
        gw = self.graph_writer
        if gw is None or not graph_records:
            return

        R = float(self.TravUpdate_XYResolution)
        tol = float(self._graph_coord_tol)
        lxmin, lxmax, lymin, lymax = self._graph_lattice_bounds
        n_layers = int(self.TraversabilityMap_size_layers)
        lookup_r = gw.default_lookup_radius       # 0.75 * R (lookup only)

        # ---- 1) preflight: resolve slots, read final map values, group -------
        # by_slot: slot -> list of (theta_layer, cmd_v_f32, cmd_w_f32, meta_int)
        by_slot = {}
        slot_xy = {}                              # slot -> (stored_x, stored_y) for reconnect queries
        out_of_domain = 0                         # per-update skip counter (logged once)
        for rec in graph_records:
            x, y, theta_layer = float(rec[0]), float(rec[1]), int(rec[2])
            # Classify against the RETAINED lattice-center bounds (not the
            # half-cell-extended storage bounds), with the float tolerance.
            if not (lxmin - tol <= x <= lxmax + tol and lymin - tol <= y <= lymax + tol):
                out_of_domain += 1
                continue
            if theta_layer < 0 or theta_layer >= n_layers:
                raise SharedGraphError(
                    f"_apply_graph_update: heading layer {theta_layer} out of range "
                    f"[0, {n_layers}) for point ({x:.6g}, {y:.6g})")
            # Resolve once with the default lookup radius; -1 on a miss.
            slot = gw.nearest_node(x, y)
            if slot == -1:
                raise SharedGraphError(
                    f"_apply_graph_update: in-domain point ({x:.6g}, {y:.6g}) did not "
                    f"resolve to any graph node within lookup_radius={lookup_r:.6g}; "
                    f"lattice-center bounds x[{lxmin}, {lxmax}] y[{lymin}, {lymax}], "
                    f"R={R}, coord_tol={tol:.3g} — lattice/graph mismatch")
            # A lookup hit alone is insufficient (radius is 0.75*R): require the
            # stored float32 coordinate to equal the expected lattice coordinate.
            resolved_x = gw.node_x[slot]
            resolved_y = gw.node_y[slot]
            if not (np.isclose(resolved_x, x, rtol=0.0, atol=tol)
                    and np.isclose(resolved_y, y, rtol=0.0, atol=tol)):
                raise SharedGraphError(
                    f"_apply_graph_update: point ({x:.6g}, {y:.6g}) resolved to node "
                    f"{slot} at ({float(resolved_x):.6g}, {float(resolved_y):.6g}) but the "
                    f"stored coordinate differs by more than coord_tol={tol:.3g}; "
                    f"lattice-center bounds x[{lxmin}, {lxmax}] y[{lymin}, {lymax}], "
                    f"R={R}, lookup_radius={lookup_r:.6g} — off-lattice write-back point")
            # Unpack the CARRIED payload decided at the capture site (direct
            # write — no dense-grid read-back). The coarse point's own fine cell
            # keeps its Step-9 value through the interpolation pass (the cell is
            # a bilinear corner), so the carried value equals exactly the
            # cmd_v/cmd_w/meta the dense map holds for this point+layer.
            cmd_v = np.float32(rec[3])
            cmd_w = np.float32(rec[4])
            # Finiteness guard kept on the carried values (defensive; the
            # capture sites already guarantee finite commands).
            if not (np.isfinite(cmd_v) and np.isfinite(cmd_w)):
                raise SharedGraphError(
                    f"_apply_graph_update: non-finite carried command ({cmd_v}, {cmd_w}) "
                    f"at ({x:.6g}, {y:.6g}) layer {theta_layer}; cannot mirror into the graph")
            meta_value = int(rec[5])
            by_slot.setdefault(int(slot), []).append((theta_layer, cmd_v, cmd_w, meta_value))
            slot_xy[int(slot)] = (float(resolved_x), float(resolved_y))

        if out_of_domain:
            # One summary per map update, not one line per point.
            print(f"[shared-graph] skipped {out_of_domain} write-back point(s) outside "
                  f"the retained lattice-center bounds x[{lxmin}, {lxmax}] "
                  f"y[{lymin}, {lymax}]")

        if not by_slot:
            return

        # ---- 2) stage final per-node too-risky state (independent of order) --
        # Live read-only views into the writer payload arrays (reflect writes
        # made inside the batch below once step 3.1 runs).
        cur_cmd_v = gw.node_payload('cmd_v')      # (n_max, n_layers) float32
        cur_cmd_w = gw.node_payload('cmd_w')
        cur_meta = gw.node_payload('meta')        # (n_max, n_layers) uint8
        old_has_too_risky = {}
        final_has_too_risky = {}
        for slot, updates in by_slot.items():
            staged_v = np.array(cur_cmd_v[slot], dtype=np.float32)   # (n_layers,) copy
            staged_w = np.array(cur_cmd_w[slot], dtype=np.float32)
            staged_m = np.array(cur_meta[slot], dtype=np.uint8)
            old_has_too_risky[slot] = self._graph_node_has_too_risky(staged_v, staged_w, staged_m)
            for (L, v, w, mv) in updates:          # apply every layer update to the staged copy
                staged_v[L] = v
                staged_w[L] = w
                staged_m[L] = mv
            final_has_too_risky[slot] = self._graph_node_has_too_risky(staged_v, staged_w, staged_m)

            # ── Replanning-gate accounting ───────────────────────────────────────────
            # A node counts as changed only when its FINAL payload actually differs from what
            # the graph already holds. Re-estimating unchanged terrain — the Case-C random
            # gap-filling path, which fires every tick at RandomRate_TravUpdate — rewrites
            # identical values and therefore contributes nothing. This is what makes the gate
            # selective; a naive len(graph_records) count would be large on every tick.
            eps = self._replan_cmd_change_eps
            if (np.any(np.abs(staged_v - cur_cmd_v[slot]) > eps)
                    or np.any(np.abs(staged_w - cur_cmd_w[slot]) > eps)
                    or np.any(staged_m != cur_meta[slot])):
                self.map_change_stats['graph_nodes_value_changed'] += 1
            # Tier 1: a flip is exactly what adds/removes edges in steps 3.2-3.4 below, i.e.
            # the only graph change that can make a path infeasible or unlock a shortcut.
            if old_has_too_risky[slot] != final_has_too_risky[slot]:
                self.map_change_stats['graph_toorisky_flips'] += 1

        # ---- 3) apply everything under one outer edit_batch (publication) ----
        with gw.edit_batch():
            # 3.1 payload updates through explicit indexed={...} syntax; one
            #     three-field call bumps this node's update_version once.
            for slot, updates in by_slot.items():
                for (L, v, w, mv) in updates:
                    gw.update_node_payload(
                        slot,
                        indexed={
                            "cmd_v": (L, v),
                            "cmd_w": (L, w),
                            "meta": (L, mv),
                        },
                    )
            # 3.2 isolate every affected node whose FINAL state is too-risky
            #     (degree 0). remove_edges_of_node on an already-detached active
            #     node is a no-op returning 0.
            for slot in by_slot:
                if final_has_too_risky[slot]:
                    gw.remove_edges_of_node(slot)
            # 3.3 reconnect only nodes that transitioned too-risky -> not-too-risky,
            #     restoring their orthogonal/diagonal grid neighbors via a generic
            #     radius query at exactly 2.0*default_lookup_radius (== 1.5*R for
            #     the lattice graph, since create_grid stores default_lookup_radius
            #     = 0.75*R). Sourcing the radius from the graph frontend keeps this
            #     reconnect off the map's fixed-resolution field. Collect the
            #     generic in-radius candidates considered for the final isolation pass.
            reconnect_candidates = set()
            for slot in by_slot:
                if old_has_too_risky[slot] and not final_has_too_risky[slot]:
                    sx, sy = slot_xy[slot]
                    gw.connect_node(slot, radius=2.0 * gw.default_lookup_radius)
                    for c in gw.radius_query(sx, sy, 2.0 * gw.default_lookup_radius):
                        c = int(c)
                        if c != slot:
                            reconnect_candidates.add(c)
            # 3.4 final isolation pass: a generic reconnect may have wired an edge
            #     to a candidate that itself still has a too-risky layer (its own
            #     current or this-cycle final state). Re-isolate every such
            #     candidate so no edge is incident to a too-risky node.
            for c in reconnect_candidates:
                if c in final_has_too_risky:
                    still_too_risky = final_has_too_risky[c]
                else:
                    # Not updated this cycle: its live payload is unchanged.
                    still_too_risky = self._graph_node_has_too_risky(
                        cur_cmd_v[c], cur_cmd_w[c], cur_meta[c])
                if still_too_risky:
                    gw.remove_edges_of_node(c)


    def _compute_travmap(self, estimating_xypts_set, estimating_area_xmin, estimating_area_xmax, estimating_area_ymin, estimating_area_ymax):
        
        if estimating_xypts_set is None:
            print(" Estimating area: x-axis: {:.1f}, {:.1f}, y-axis: {:.1f}, {:.1f}".format(estimating_area_xmin, estimating_area_xmax, estimating_area_ymin, estimating_area_ymax))
                    
            if estimating_area_xmin < self.map_xmin or estimating_area_xmax > self.map_xmax or estimating_area_ymin < self.map_ymin or estimating_area_ymax > self.map_ymax:
                print("The estimating area is out of the global map.")


            # Grid points in the estimating area, grid size of self.TravUpdate_XYResolution. xyz coordinates
            estimating_xycpts_set = np.array([ [x,y,c] for x in np.arange(estimating_area_xmin, estimating_area_xmax + self.TravUpdate_XYResolution, self.TravUpdate_XYResolution) for y in np.arange(estimating_area_ymin, estimating_area_ymax + self.TravUpdate_XYResolution, self.TravUpdate_XYResolution)\
                                                for c in np.arange(self.TraversabilityMap_theta_min, self.TraversabilityMap_theta_max + self.TraversabilityMap_theta_resolution, self.TraversabilityMap_theta_resolution) ])
        
        else:
            # estimating_xypts_set is ndarray of [x,y]. I want to add one more dimension for theta
            _thetas = np.arange(self.TraversabilityMap_theta_min, self.TraversabilityMap_theta_max + self.TraversabilityMap_theta_resolution, self.TraversabilityMap_theta_resolution)
            estimating_xycpts_set = np.column_stack([
                np.repeat(estimating_xypts_set, len(_thetas), axis=0),
                np.tile(_thetas, len(estimating_xypts_set))
            ])






        if self.is_RobocentricMap_received and self.is_ElevationMap_built and estimating_xycpts_set.shape[0] != 0:

            # ── Step 1: Unpack query points ──────────────────────────────────────────
            # N: total number of (x, y, theta) points whose traversability will be estimated.
            N      = estimating_xycpts_set.shape[0]
            xs     = estimating_xycpts_set[:, 0]   # world x,   shape (N,)
            ys     = estimating_xycpts_set[:, 1]   # world y,   shape (N,)
            thetas = estimating_xycpts_set[:, 2]   # heading,   shape (N,)

            # ── Step 2: Patch geometry ────────────────────────────────────────────────
            # Each query point gets a square elevation patch of side P (in pixels) centered on it.
            # half_w: half-side in pixels; P: full side length in pixels.
            half_w = int(np.round(0.5 * self.local_patch_size / self.map_resolution))
            P = 2 * half_w

            # local_i, local_j: per-pixel row/col offsets from patch center in body frame, shape (P*P,).
            # Built once and shared across all N points.
            ii, jj = np.meshgrid(range(-half_w, half_w), range(-half_w, half_w), indexing='ij')
            local_i = ii.ravel()
            local_j = jj.ravel()

            # ── Step 3: Body-frame → map-pixel transform ──────────────────────────────
            # Map convention: row = (xmax - world_x) / res, col = (ymax - world_y) / res.
            # row_centers, col_centers: map-pixel of each query point center, shape (N,).
            row_centers = ((self.map_xmax - xs) / self.map_resolution).astype(int)
            col_centers = ((self.map_ymax - ys) / self.map_resolution).astype(int)
            cos_t = np.cos(thetas)   # shape (N,)
            sin_t = np.sin(thetas)   # shape (N,)

            # Rotate body-frame pixel offsets by theta, then shift to map-frame center.
            # global_rows[n,k] = cos(θ_n)*local_i[k] - sin(θ_n)*local_j[k] + row_center_n
            # global_cols[n,k] = sin(θ_n)*local_i[k] + cos(θ_n)*local_j[k] + col_center_n
            # Shape: (N, P*P)
            global_rows = np.round(
                cos_t[:, None] * local_i - sin_t[:, None] * local_j + row_centers[:, None]
            ).astype(int)
            global_cols = np.round(
                sin_t[:, None] * local_i + cos_t[:, None] * local_j + col_centers[:, None]
            ).astype(int)

            # ── Step 4: Elevation lookup with bounds check ────────────────────────────
            # valid: True where the rotated pixel falls inside the elevation map, shape (N, P*P).
            valid = (
                (global_rows >= 0) & (global_rows < self.map_size_rows) &
                (global_cols >= 0) & (global_cols < self.map_size_cols)
            )
            # Clamp OOB indices to 0 so array lookup never faults, then replace those values with NaN.
            safe_rows = np.where(valid, global_rows, 0)
            safe_cols = np.where(valid, global_cols, 0)
            gathered = self.ElevationMap[safe_rows, safe_cols]   # (N, P*P)
            gathered = np.where(valid, gathered, np.nan)

            # A patch is unusable if ANY pixel is NaN (OOB or missing elevation data).
            ismapnan_arr = np.any(np.isnan(gathered), axis=1)    # (N,) bool

            all_patches = gathered.reshape(N, P, P)              # (N, P, P)

            # ── Step 5: Build xyc_patchindex ─────────────────────────────────────────
            # xyc_patchindex: (N, 4) — columns [x, y, theta, patch_index].
            # patch_index = -1  → patch was NaN / unusable (will use InitialGuess later).
            # patch_index = k   → patch is the k-th row in patch_tensor (0-indexed, 0 ≤ k < M).
            xyc_patchindex = np.concatenate(
                (estimating_xycpts_set, np.full((N, 1), -1, dtype=int)), axis=1
            )
            valid_mask = ~ismapnan_arr       # (N,) — True for fully-observed patches
            M = int(valid_mask.sum())        # number of usable patches

            if M > 0:
                valid_patches = all_patches[valid_mask]                # (M, P, P)
                # Height normalization: subtract each patch's 5th-percentile elevation so the
                # model sees relative relief rather than absolute height.
                offsets = np.percentile(valid_patches.reshape(M, -1), 5, axis=1)  # (M,)
                valid_patches = valid_patches - offsets[:, None, None]
                patch_tensor = valid_patches                           # (M, P, P)
                # Assign sequential indices 0..M-1 to valid rows so they can look up target_v/w later.
                xyc_patchindex[valid_mask, 3] = np.arange(M)
            else:
                patch_tensor = np.empty((0, P, P))

            # ── Step 6: Batch padding and duplication ─────────────────────────────────
            # The model requires batch sizes that are multiples of 64.
            # '% 64' on the outer expression prevents adding 64 spurious zeros when M is already a multiple of 64.
            pad = (64 - patch_tensor.shape[0] % 64) % 64
            patch_tensor_extended = np.concatenate(
                (patch_tensor, np.zeros((pad, P, P))), axis=0
            )  # (size, P, P), size = next multiple of 64 ≥ M
            # Duplicate: first 'size' rows → cmd_v evaluation, next 'size' rows → cmd_w evaluation.
            # TEMPORARY PATCH: cmd_w is derived from cmd_v after the chunk loop, so the cmd_w
            # half is never evaluated → no duplication (halves the inference batch).
            # patch_cat_tensor = np.concatenate((patch_tensor_extended, patch_tensor_extended), axis=0)  # (2*size, P, P)
            patch_cat_tensor = patch_tensor_extended  # (size, P, P), cmd_v evaluation only
            size = patch_tensor_extended.shape[0]  # multiple of 64; only indices 0..M-1 hold real patches

            # ── Step 7: Candidate speed bins ──────────────────────────────────────────
            # target_v/w: (size, 1) — best safe speed per patch slot.
            # Initialized to sentinel values (toorisky) meaning "no safe speed found yet".
            target_v = np.full((size, 1), self.toorisky_v)
            v_res = self.v_res
            # v_bins: candidate linear speeds in descending order [v_max, ..., TreatObs_cmd_v].
            # Lower bound is TreatObs_cmd_v: patches failing at every bin keep toorisky_v (treated as obstacle).
            # Descending so the largest feasible speed is assigned first and locked in by the sentinel guard.
            v_bins = np.arange(self.TreatObs_cmd_v, self.maximum_cmd_v + v_res, v_res)
            v_bins = v_bins[::-1]

            target_w = np.full((size, 1), self.toorisky_w)
            w_res = self.w_res
            # w_bins: candidate angular speeds in descending order [w_max, ..., w_res].
            w_bins = np.arange(w_res, self.maximum_cmd_w + w_res, w_res)
            w_bins = w_bins[::-1]

            # ── Step 8: Chunked TravFormer inference ──────────────────────────────────
            # CMD_CHUNK: number of velocity bins batched into a single GPU forward pass.
            # Higher = fewer kernel launches (faster), but scales GPU memory by CMD_CHUNK.
            # Halved automatically on CUDA OOM until it reaches 1 (original per-bin behavior).
            # Skip inference entirely when size==0 (all patches were NaN/OOB): every
            # xyc_patchindex entry already has patchindex==-1, so Step 9 will fall back
            # to InitialGuess for all points without needing TravFormer output.
            CMD_CHUNK = 2
            n_bins = len(v_bins)  # total number of candidate speed levels to evaluate
            chunk_start = 0
            while size > 0 and chunk_start < n_bins:
                # chunk_v/chunk_w: (n_chunk,) — velocity bins in this chunk (descending order,
                # so the largest feasible speed is assigned first and later bins are skipped by the mask guard)
                chunk_v = v_bins[chunk_start : chunk_start + CMD_CHUNK]
                # chunk_w = w_bins[chunk_start : chunk_start + CMD_CHUNK]
                n_chunk = len(chunk_v)  # ≤ CMD_CHUNK; may be smaller for the last chunk

                # patch_chunk: (n_chunk*2*size, P, P)
                # patch_cat_tensor is (2*size, P, P): first half for cmd_v eval, second for cmd_w eval.
                # Replicated n_chunk times so that each bin j occupies rows [j*2*size : (j+1)*2*size].
                # TEMPORARY PATCH: patch_cat_tensor is (size, P, P) now, so patch_chunk is
                # (n_chunk*size, P, P) and bin j occupies rows [j*size : (j+1)*size].
                patch_chunk = np.concatenate([patch_cat_tensor] * n_chunk, axis=0)

                # cmd_chunk: (n_chunk*2*size, 2) — command pairs [cmd_v, cmd_w].
                # For bin j:
                #   rows [j*2*size       : j*2*size+size] → col 0 = v, col 1 = 0  (v-only evaluation)
                #   rows [j*2*size+size  : (j+1)*2*size ] → col 0 = 0, col 1 = w  (w-only evaluation)
                # TEMPORARY PATCH: cmd_w is no longer swept, so only the v-rows remain:
                #   rows [j*size : (j+1)*size] → col 0 = v, col 1 = 0  (v-only evaluation)
                # The second column MUST be kept (the model slices commands[:, 1:2] and its
                # cmd_proj expects 2*(2*num_freqs+1) inputs), it just stays zero.
                # cmd_chunk = np.zeros((n_chunk * 2 * size, 2))
                cmd_chunk = np.zeros((n_chunk * size, 2))
                # for j, (v, w) in enumerate(zip(chunk_v, chunk_w)):
                #     cmd_chunk[j*2*size        : j*2*size + size,  0] = v
                #     cmd_chunk[j*2*size + size : (j+1)*2*size,     1] = w
                for j, v in enumerate(chunk_v):
                    cmd_chunk[j*size : (j+1)*size, 0] = v

                try:
                    start = time()
                    # instab_mean_cat, instab_std_cat: each (n_chunk*2*size, 1)
                    [instab_mean_cat, instab_std_cat] = self.TravFormer_Inference(None, patch_chunk, cmd_chunk, model=self.elevonly_vw2instab_model, device=self.device)
                except RuntimeError as e:
                    err = str(e).lower()
                    # Catch both OOM and "invalid configuration argument" (CUDA grid-dim overflow
                    # when batch×num_heads exceeds the 65535 hardware limit at large resolutions).
                    if ('out of memory' in err or 'invalid configuration argument' in err) and CMD_CHUNK > 1:
                        CMD_CHUNK = max(1, CMD_CHUNK // 2)
                        del patch_chunk, cmd_chunk  # free CPU memory before retry
                        if self.device != 'cpu':
                            torch.cuda.empty_cache()
                        continue  # retry this chunk with halved size
                    raise

                # VaR_chunk: (n_chunk, 2*size) — Value-at-Risk instability per (bin, patch).
                # VaR_chunk[j, :size]  → v-evaluation VaR for bin j across all patches
                # VaR_chunk[j, size:]  → w-evaluation VaR for bin j across all patches
                # TEMPORARY PATCH: only the v-half is inferred → (n_chunk, size).
                # VaR_chunk[j] → v-evaluation VaR for bin j across all patches.
                # VaR_chunk = (instab_mean_cat[:,0] + instability_std_multiplier*instab_std_cat[:,0]).reshape(n_chunk, 2*size)
                VaR_chunk = (instab_mean_cat[:,0] + instability_std_multiplier*instab_std_cat[:,0]).reshape(n_chunk, size)

                # for j, (v, w) in enumerate(zip(chunk_v, chunk_w)):
                for j, v in enumerate(chunk_v):
                    # VaR = VaR_chunk[j]  # (2*size,)
                    VaR = VaR_chunk[j]  # (size,)
                    # Assign v only to patches still at sentinel (not yet assigned a safe speed)
                    # that now pass the VaR threshold at this speed level.
                    mask_v = (VaR[:size] < self.instability_limit) & (target_v[:, 0] == self.toorisky_v)
                    target_v[mask_v, 0] = v
                    # mask_w = (VaR[size:] < self.instability_limit) & (target_w[:, 0] == self.toorisky_w)
                    # target_w[mask_w, 0] = w
                    # TEMPORARY PATCH: derive cmd_w from the cmd_v just assigned. Patches that
                    # never pass any bin keep the (toorisky_v, toorisky_w) init pair untouched,
                    # which _graph_layer_too_risky_mask requires as an exact pair.
                    target_w[mask_v, 0] = self.maximum_cmd_w * v / self.maximum_cmd_v

                chunk_start += n_chunk

                # Early exit: all patches have been assigned a safe speed for both v and w
                # TEMPORARY PATCH: target_w is derived from target_v afterwards and stays at its
                # sentinel during the loop, so only the v condition may be tested here.
                # if not (target_v[:, 0] == self.toorisky_v).any() and not (target_w[:, 0] == self.toorisky_w).any():
                if not (target_v[:, 0] == self.toorisky_v).any():
                    break
            


            # TEMPORARY PATCH: get target_w by target_v.
            # Moved into the Step 8 bin loop (target_w[mask_v, 0] = ...) so the too-risky
            # sentinel pair survives without a repair pass; rebinding target_w here would
            # overwrite that sentinel again.
            # target_w = self.maximum_cmd_w * target_v / self.maximum_cmd_v
            # Restore the exact too-risky sentinel pair: _graph_layer_too_risky_mask requires
            # cmd_v == toorisky_v AND cmd_w == toorisky_w, but the scaling above would map
            # toorisky_v to maximum_cmd_w*toorisky_v/maximum_cmd_v (≠ toorisky_w).
            # target_w[target_v == self.toorisky_v] = self.toorisky_w

            
            # ── Step 8b: Propagate toorisky across all theta directions per (x,y) ─────
            # If ANY heading at (x, y) fails (toorisky_v OR toorisky_w), force every
            # heading at that (x, y) to toorisky — unsafe from one direction means the
            # whole cell is treated as an obstacle.
            # estimating_xycpts_set is built by tiling all thetas over each (x,y), so
            # consecutive blocks of n_thetas entries share the same (x,y).
            n_thetas = self.TraversabilityMap_size_layers
            if N % n_thetas == 0:
                # patchindices_2d: (N//n_thetas, n_thetas) — patchindex per (x,y) group
                patchindices_2d = xyc_patchindex[:, 3].astype(int).reshape(-1, n_thetas)
                for _i in range(patchindices_2d.shape[0]):
                    _valid_idx = patchindices_2d[_i][patchindices_2d[_i] >= 0]
                    if len(_valid_idx) == 0:
                        continue
                    if np.any(target_v[_valid_idx, 0] == self.toorisky_v):
                        target_v[_valid_idx, 0] = self.toorisky_v
                        target_w[_valid_idx, 0] = self.toorisky_w

            # ── Step 9: Write safe speeds back to TraversabilityMap ──────────────────
            # Iterate all N query points in the same order as estimating_xycpts_set so that
            # patchindex_counter stays aligned with xyc_patchindex rows.
            #
            # Shared-graph mirror hook (Part 2): collect the (x, y, heading layer)
            # touched by this write-back — the update membership and heading-layer
            # selection computed here, NOT recomputed. None when no graph is owned
            # (base map, reader proxy, or non-"Astar" planner), which disables the
            # hook. Each record CARRIES the cmd_v/cmd_w/meta decided here (direct
            # write); _apply_graph_update no longer reads the dense map back.
            graph_records = [] if self.graph_writer is not None else None
            patchindex_counter = 0
            for xyc in estimating_xycpts_set:

                x     = xyc[0]
                y     = xyc[1]
                theta = xyc[2]

                # patchindex: index into target_v/w for this point.
                # -1 means the patch was NaN → fall back to the initial-guess safe speed.
                patchindex = xyc_patchindex[patchindex_counter, 3].astype(int)
                patchindex_counter += 1

                if patchindex == -1:
                    cmd_v = self.InitialGuess_cmd_v
                    cmd_w = self.InitialGuess_cmd_w
                else:
                    # target_v/w[patchindex] holds the largest speed that passed the VaR threshold,
                    # or toorisky_v/w (sentinel) if no speed was safe for this patch.
                    cmd_v = target_v[patchindex, 0]
                    cmd_w = target_w[patchindex, 0]

                # The TraversabilityMap is updated in grid cells of size TravUpdate_XYResolution.
                # Compute the world-space corners of the square cell centered at (x, y):
                #   BL = bottom-left (smaller x, smaller y)
                #   TR = top-right   (larger  x, larger  y)
                square_BL_xyc = np.array([x - self.TravUpdate_XYResolution/2, y - self.TravUpdate_XYResolution/2])
                square_TR_xyc = np.array([x + self.TravUpdate_XYResolution/2, y + self.TravUpdate_XYResolution/2])

                # Convert world corners to map pixel indices.
                # Map convention: row = (xmax - world_x)/res  → larger x = smaller row (TR → smaller row).
                #                 col = (ymax - world_y)/res  → larger y = smaller col (TR → smaller col).
                # So row_start < row_end and col_start < col_end, giving a valid slice.
                square_BL_entry = np.array([ round( (self.map_xmax - square_BL_xyc[0])/self.map_resolution ), round( (self.map_ymax - square_BL_xyc[1])/self.map_resolution ) ])
                square_TR_entry = np.array([ round( (self.map_xmax - square_TR_xyc[0])/self.map_resolution ), round( (self.map_ymax - square_TR_xyc[1])/self.map_resolution ) ])

                # Discretize the canonical [-π, π) heading to the nearest layer.
                theta = Utils.wrap_to_pi(theta)
                theta_layer = self.get_theta_layer(theta)

                # Clip pixel bounds to stay within the map and write cmd_v, cmd_w, and status.
                row_start = int(np.clip(square_TR_entry[0], 0, self.map_size_rows))
                row_end   = int(np.clip(square_BL_entry[0], 0, self.map_size_rows))
                col_start = int(np.clip(square_TR_entry[1], 0, self.map_size_cols))
                col_end   = int(np.clip(square_BL_entry[1], 0, self.map_size_cols))
                self.TraversabilityMap[row_start:row_end, col_start:col_end, theta_layer, 0] = cmd_v
                self.TraversabilityMap[row_start:row_end, col_start:col_end, theta_layer, 1] = cmd_w
                self.TravMap_MetaInfo[row_start:row_end, col_start:col_end, theta_layer] = self.TravmapInfo_IntEnum.TRAV_Estimated.value

                # Shared-graph mirror hook (Part 2): record this coarse point +
                # heading layer for the post-interpolation graph mirror, carrying
                # the exact cmd_v/cmd_w decided above plus TRAV_Estimated (direct
                # write — no dense read-back in _apply_graph_update).
                if graph_records is not None:
                    graph_records.append((
                        float(x), float(y), int(theta_layer),
                        float(cmd_v), float(cmd_w),
                        int(self.TravmapInfo_IntEnum.TRAV_Estimated.value)))










            # Interpolation
            N = math.floor(self.TravUpdate_XYResolution / self.map_resolution)
            R = self.TravUpdate_XYResolution
            _t   = np.linspace(0.0, 1.0, N + 1)
            _sig = 1.0 / (1.0 + np.exp(-self.interp_sigmoid_k * (_t - 0.5)))
            xi_w = (_sig - _sig[0]) / (_sig[-1] - _sig[0])    # normalize to [0,1]; f(0.5)=0.5 exactly
            yi_w = xi_w

            # (dxTL, dyTL, dxBR, dyBR, dxTR, dyTR) world-space offsets from BL corner
            # quadrants 0-3 are defined in array space (same as original)
            _q_offsets = [
                ( R,  0,  0, -R,  R, -R),  # q0
                ( 0,  R,  R,  0,  R,  R),  # q1
                (-R,  0,  0,  R, -R,  R),  # q2
                ( 0, -R, -R,  0, -R, -R),  # q3
            ]

            for xyc in estimating_xycpts_set:
                theta = Utils.wrap_to_pi(xyc[2])
                theta_layer = self.get_theta_layer(theta)

                row_BL, col_BL = self.xy2grid(xyc[0], xyc[1])
                if row_BL is None or col_BL is None:
                    continue

                for quadrant, (dxTL, dyTL, dxBR, dyBR, dxTR, dyTR) in enumerate(_q_offsets):
                    row_TL, col_TL = self.xy2grid(xyc[0] + dxTL, xyc[1] + dyTL)
                    if row_TL is None or col_TL is None:
                        continue
                    row_BR, col_BR = self.xy2grid(xyc[0] + dxBR, xyc[1] + dyBR)
                    if row_BR is None or col_BR is None:
                        continue
                    row_TR, col_TR = self.xy2grid(xyc[0] + dxTR, xyc[1] + dyTR)
                    if row_TR is None or col_TR is None:
                        continue

                    # Corner values for both map layers at once: shape (2,)
                    BL = self.TraversabilityMap[row_BL, col_BL, theta_layer, :]
                    TL = self.TraversabilityMap[row_TL, col_TL, theta_layer, :]
                    BR = self.TraversabilityMap[row_BR, col_BR, theta_layer, :]
                    TR = self.TraversabilityMap[row_TR, col_TR, theta_layer, :]

                    # Bilinear interpolation over the (N+1)x(N+1) patch, both layers at once
                    bottom = BL + xi_w[:, None] * (BR - BL)                         # (N+1, 2)
                    top    = TL + xi_w[:, None] * (TR - TL)                         # (N+1, 2)
                    patch  = (1 - yi_w[:, None, None]) * bottom[None] \
                           +      yi_w[:, None, None]  * top[None]                  # (N+1, N+1, 2)

                    # Write patch into the map; orientation per quadrant derived from loop indices:
                    #   q0: TravMap[row_BL-yi, col_BL+xi]  →  patch[::-1]
                    #   q1: TravMap[row_BL-xi, col_BL-yi]  →  patch[::-1,::-1].transpose(1,0,2)
                    #   q2: TravMap[row_BL+yi, col_BL-xi]  →  patch[:,    ::-1]
                    #   q3: TravMap[row_BL+xi, col_BL+yi]  →  patch.transpose(1,0,2)
                    if quadrant == 0:
                        _interp_vals = patch[::-1]
                        _row_sl, _col_sl = slice(row_BL-N, row_BL+1),  slice(col_BL,   col_BL+N+1)
                    elif quadrant == 1:
                        _interp_vals = patch[::-1, ::-1].transpose(1, 0, 2)
                        _row_sl, _col_sl = slice(row_BL-N, row_BL+1),  slice(col_BL-N, col_BL+1)
                    elif quadrant == 2:
                        _interp_vals = patch[:, ::-1]
                        _row_sl, _col_sl = slice(row_BL,   row_BL+N+1), slice(col_BL-N, col_BL+1)
                    elif quadrant == 3:
                        _interp_vals = patch.transpose(1, 0, 2)
                        _row_sl, _col_sl = slice(row_BL,   row_BL+N+1), slice(col_BL,   col_BL+N+1)
                    self.TraversabilityMap[_row_sl, _col_sl, theta_layer, :] = _interp_vals

            # ── Shared-graph mirror + topology (Part 2) ──────────────────────────
            # The map is now final for this update (Step 9 + interpolation done).
            # Mirror the coarse-lattice cmd_v/cmd_w/meta into node payloads and
            # maintain the aggregate too-risky topology inside one edit_batch().
            # No-op when no graph is owned; graph-mutation exceptions propagate.
            if graph_records is not None:
                self._apply_graph_update(graph_records)
