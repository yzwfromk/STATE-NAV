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
# the Part 2 grid-topology integration are imported here; the layout (core.graph_layout,
# re-exported below) is the single definition writer and reader (planner) must both import.
from statenav_global.core.shared_graph import (
    GraphLayout, SharedGraphWriter, SharedGraphError,
)





# No ROS imports here: this module must import without a ROS environment. The ROS-facing
# callbacks (pose_callback, robo_centric_map_callback) read their messages by attribute only,
# and quaternion -> Euler uses transforms3d (bit-identical to tf_transformations' sxyz path).
import transforms3d as tf3
from scipy.ndimage import map_coordinates
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)




from pathlib import Path
_PKG_ROOT = Path(__file__).resolve().parent.parent
from statenav_global.core.config import load_planning_config
cfg = load_planning_config()   # $STATENAV_PLANNING_CONFIG, else the packaged planning_config.yaml


    
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

# The graph payload contract (TravmapInfo_IntEnum, GRAPH_LAYOUT and its heading-layer
# count / fresh-node cmd defaults) moved to statenav_global.core.graph_layout so readers
# (GlobalAStar, MPPI) can import it without this module's torch / ROS stack. Re-exported
# here so existing `from ...globalmap import GRAPH_LAYOUT` style imports keep working.
from statenav_global.core.graph_layout import (
    TravmapInfo_IntEnum, INITIAL_CMD_V_DEFAULT, INITIAL_CMD_W_DEFAULT,
    _GRAPH_HEADING_LAYERS, GRAPH_LAYOUT,
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


        # 3D (x,y,theta(=heading angle)) traversability (pre-estimated cmd v and w, world frame) lives
        # ONLY in the shared lattice graph (graph_writer); there is no dense grid. These attributes keep
        # their legacy names: the readiness flag and the heading-layer geometry of the graph payloads.
        self.is_TraversabilityMap_built = False
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

        # Per-layer traversability status values; stored in the graph's "meta" node payload.
        self.TravmapInfo_IntEnum = TravmapInfo_IntEnum



        # Visualization
        self.viz_channel = 0  # default channel
        self.viz_vmin = 0.0
        self.viz_vmax = self.maximum_cmd_v
        self.viz_title = "Traversability Map"
        self.viz_cbar_label = "Traversability Score"


        # ML related
        self.device = 'cuda'




        # Travformer related
        self.local_patch_size = cfg.local_patch_size  # in meters
        self.elevonly_vw2instab_model = None
        # Max elevation patches per TravFormer forward pass (bounds GPU memory, ~0.165 MB per patch).
        # Rounded down to a multiple of 64, the batch granularity _compute_travmap pads to.
        self.travformer_max_batch_patches = int(cfg.get('travformer_max_batch_patches', 30000)) // 64 * 64
        if self.travformer_max_batch_patches < 64:
            raise ValueError(f"planning_config.yaml: travformer_max_batch_patches must be >= 64, got "
                             f"{cfg.get('travformer_max_batch_patches')}")

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













    def bump_map_version(self):
        """
        Publish "the map content changed" to the backend, ONCE per map-write
        transaction.

        The backend version counter is the only cross-process "map changed" signal
        (SharedMemoryBackend.increment_version / LocalMemoryBackend.increment_version),
        so EVERY path that mutates ElevationMap / ElevMap_MetaInfo or the shared graph's
        traversability payloads must call this once it has finished writing.

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

        # Build metainfo map. Traversability has no dense grid: it lives only in the shared
        # graph, which CMDbasedMapWithBackend.Initilize_map builds (_build_shared_graph).
            self.ElevMap_MetaInfo = np.full((self.map_size_rows, self.map_size_cols), ElevmapInfo_IntEnum.ELEV_NotInitialized.value)

        


    

        


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
            # grid_map cell centres (getPositionFromIndex): x(r) = center + ((H-1)/2 - r)*res. Taken from
            # local_map's own (H, W): rows run along x, i.e. the transposed GridMap's length_x axis.
            half_r = 0.5 * (H - 1)
            half_c = 0.5 * (W - 1)
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
            # One rounding for the whole local grid (the global cell containing local cell (0, 0)'s centre),
            # then integer offsets: a per-cell floor/truncation flips on cell boundaries and leaves holes.
            r0 = int(np.round((self.map_xmax - global_x_all[0, 0]) / self.map_resolution - 0.5))
            c0 = int(np.round((self.map_ymax - global_y_all[0, 0]) / self.map_resolution - 0.5))
            row_global_all = r0 + row_idx
            col_global_all = c0 + col_idx

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
                #   K = number of can_be_c cells; each cell reads the (num_layers,) graph meta of the
                #   lattice node whose R x R estimation cell contains it
                trav_any_uninit = np.any(self._graph_node_rows(np.stack([vgx[can_be_c], vgy[can_be_c]], axis=1), 'meta', NINIT_T) == NINIT_T, axis=-1)
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
                # The corners of the R x R square CONTAINING the cell, {fx, fx+R} x {fy, fy+R}: the only
                # lattice points whose +-local_patch_size/2 window (> R/2) can contain it. ({-R, 0}
                # offsets queued the square on the -x/-y side, which never sees the cell, and missed fx+R.)
                fx = R * np.floor(gx / R)
                fy = R * np.floor(gy / R)
                uniq = np.unique(np.stack([fx, fy], axis=1), axis=0)  # (M, 2) – deduplicated floor points
                offsets = np.array([[0., 0.], [0., R], [R, 0.], [R, R]])  # (4, 2)
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
                    all_uninit = np.all(self._graph_node_rows(cands, 'meta', NINIT_T) == NINIT_T, axis=-1)
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
                    # Symmetric: the candidate itself and its +-x / +-y lattice neighbours (the former
                    # {-R, 0} x {-R, 0} set only looked to -x/-y, so the wavefront grew one way only).
                    offsets = np.array([[0., 0.], [-R, 0.], [R, 0.], [0., -R], [0., R]])  # (5, 2)
                    nbr     = cands[:, None, :] + offsets[None]  # (Q, 5, 2) – neighbours per candidate
                    nbr_rg  = ((self.map_xmax - nbr[:, :, 0]) / self.map_resolution).astype(int)  # (Q, 4)
                    nbr_cg  = ((self.map_ymax - nbr[:, :, 1]) / self.map_resolution).astype(int)  # (Q, 4)
                    # nbr_valid : (Q, 4) bool – which neighbours fall inside the global map
                    nbr_valid = ((nbr_rg >= 0) & (nbr_rg < self.map_size_rows)
                                 & (nbr_cg >= 0) & (nbr_cg < self.map_size_cols))

                    # near_preest : (Q,) bool – True if any in-bounds neighbour has at least one
                    # estimated traversability direction (i.e. the wavefront has reached nearby)
                    near_preest = np.zeros(len(cands), dtype=bool)
                    for j in range(len(offsets)):
                        v = nbr_valid[:, j]  # (Q,) bool – which candidates have a valid j-th neighbour
                        if not np.any(v):
                            continue
                        # has_nonnull : (Q,) bool – neighbour j is valid AND its lattice node has some
                        # estimated direction (off-map neighbours are masked out by `v`)
                        has_nonnull = v & np.any(self._graph_node_rows(nbr[:, j, :], 'meta', NINIT_T) != NINIT_T, axis=-1)
                        near_preest |= has_nonnull

                    # Discard candidates not adjacent to the estimation wavefront
                    cands = cands[near_preest]
                    crg   = crg[near_preest]
                    ccg   = ccg[near_preest]

                    if len(cands) > 0:
                        # any_uninit : (Q,) bool – candidate still has at least one unestimated direction
                        any_uninit = np.any(self._graph_node_rows(cands, 'meta', NINIT_T) == NINIT_T, axis=-1)
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
        # save_dir: when given, save the Elevation figure as a PDF into this directory (used by the
        # one-time experiment snapshot dump). Traversability has no dense grid to draw any more;
        # the graph is drawn by the planner's 'traversability_graph' dump and /graph_visualization.
        # map_theta is kept for call-site compatibility (it selected the dense map's heading layer).


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


        if self.map_resolution is None or self.ElevationMap is None or not self.is_TraversabilityMap_built:
            return self.toorisky_v, self.toorisky_w

        cmd_v_limit = self.maximum_cmd_v
        cmd_w_limit = self.maximum_cmd_w


        discretization = safecmd_sampling_dist
        num_points = int(lookahead_distance/discretization)
        angle_tolerance = np.deg2rad(lookahead_angle)

        # Fan of in-map sample points ahead of the robot, in sampling order: world (x, y) and
        # heading layer per sample; looked up from the shared graph in ONE batch below.
        sample_xs, sample_ys, sample_layers = [], [], []
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

                sample_xs.append(x)
                sample_ys.append(y)
                sample_layers.append(theta_layer)

        # (n, 2) [cmd_v, cmd_w] per sample: the values of its nearest graph node; then the same
        # running average as before.
        sample_cmds = self.graph_cmd_at(sample_xs, sample_ys, sample_layers)
        counter = 0
        for cmd_v, cmd_w in sample_cmds:
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
            Total lookahead time in seconds. A caller that pins cmd_v_limit/cmd_w_limit to 1.0
            and passes a distance in metres here makes reachable_radius below collapse to that
            fixed distance.
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

    
    def pose_callback(self, msg):   # msg: geometry_msgs/PoseStamped, or any object with .pose.position/.orientation
        p = msg.pose.position
        q = msg.pose.orientation
        quat = [q.x, q.y, q.z, q.w]
        # transforms3d takes [w, x, y, z]; 'sxyz' matches tf_transformations.euler_from_quaternion.
        roll, pitch, yaw = tf3.euler.quat2euler([quat[3], quat[0], quat[1], quat[2]], 'sxyz')
        msg_yaw = yaw
        self.robot_x = p.x
        self.robot_y = p.y
        self.robot_roll = Utils.wrap_to_pi(roll)
        self.robot_pitch = Utils.wrap_to_pi(pitch)
        self.robot_heading = Utils.wrap_to_pi(yaw)
        # rospy.loginfo_throttle(1.0, "[pose_callback] x: %.2f, y: %.2f, yaw: %.2f rad", self.robot_x, self.robot_y, self.robot_heading)


    def robo_centric_map_callback(self, msg):

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
        # elevation_mapping publishes the RAW base position as info.pose (elevation_mapping_node.py
        # publish_map: _map_t), while its grid centre moves in whole cells (move_to): snap x/y back
        # onto the cell grid, else every local map lands up to res/2 off per axis.
        new_center_pos = (round(msg.info.pose.position.x / self.map_resolution) * self.map_resolution,
                          round(msg.info.pose.position.y / self.map_resolution) * self.map_resolution,
                          msg.info.pose.position.z)
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
        # Raised only once the layers are in place: Update_map (main loop) must never see the flag
        # with no layer data (KeyError on the first map).
        if not self.is_RobocentricMap_received:
            self.is_RobocentricMap_received = True
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
        # The ONLY traversability store. Owned by the writer-side map (every planner
        # type); created by _build_shared_graph() once the coarse lattice geometry
        # is known (its create_grid also seeds the neutral interior and the
        # too-risky, isolated map border). None on a reader proxy, which reads it
        # through graph_cmd_at's lazily attached SharedGraphReader instead.
        self.graph_writer = None
        # Retained coarse estimation-lattice CENTER bounds (kx_min*R ... kx_max*R),
        # the expected-update domain used to classify write-back points. Distinct
        # from create_grid's half-cell-extended storage bounds. Set in
        # _build_shared_graph().
        self._graph_lattice_bounds = None      # (lxmin, lxmax, lymin, lymax)
        # float32-roundoff tolerance for stored-coordinate equality checks; set
        # from the actual snapped bounds in _build_shared_graph().
        self._graph_coord_tol = None
        # Reader-side graph access for graph_cmd_at (planner process): one lazily
        # attached SharedGraphReader and one reused GraphSnapshot buffer.
        self._graph_lookup_reader = None
        self._graph_lookup_snapshot = None


    # =====================================================================
    # Shared-memory graph integration (Part 2) — one contiguous region.
    #
    # Responsibilities. Writer-side unless noted; the writer methods raise
    # SharedGraphError when no graph is owned (_require_graph_writer), because
    # the graph is the ONLY traversability store (there is no dense grid):
    #   _build_shared_graph()        : lattice-aligned graph construction
    #   _close_shared_graph()        : shutdown cleanup (also the reader lookup)
    #   _graph_layer_too_risky_mask()/_graph_node_has_too_risky() : the single
    #                                  too-risky predicate (shared with tests)
    #   _apply_graph_update()        : write one map-update's coarse-lattice
    #                                  cmd_v/cmd_w/meta into node payloads and
    #                                  maintain the aggregate too-risky topology
    #   _graph_slots_at() / _graph_node_rows() : world points -> lattice node
    #                                  slots / payload rows (status gates)
    #   graph_cmd_at()               : writer OR reader — world points -> cmd of
    #                                  the nearest lattice node
    #
    # The capture site (Step 9)
    # carries the exact cmd_v/cmd_w/meta it decided into the record (direct
    # write), and _apply_graph_update never recomputes values or heading-layer
    # selection.
    # =====================================================================

    def _build_shared_graph(self):
        """
        Construct the lattice-aligned shared graph exactly once, for EVERY
        planner type: it is the only traversability store. No-op if already
        built.

        Node centers land on the estimation lattice (world multiples of
        TravUpdate_XYResolution clipped to the map bounds) via create_grid's
        cell-edge bounds, so runtime write-back points resolve by coordinate.
        No grid-index-to-slot logic is created or retained.
        """
        # Idempotent: build at most once.
        if self.graph_writer is not None:
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
        """Idempotently close + unlink the owned graph (shutdown hook); close a reader lookup."""
        gw = self.graph_writer
        if gw is not None:
            self.graph_writer = None
            gw.cleanup()
        # Reader-side lookup handles (graph_cmd_at on a reader proxy): close, never unlink.
        rd = getattr(self, '_graph_lookup_reader', None)
        if rd is not None:
            self._graph_lookup_reader = None
            self._graph_lookup_snapshot = None
            rd.close()

    def _require_graph_writer(self, where):
        """Return the owned graph writer; raise when there is none (it is the only trav store)."""
        gw = getattr(self, 'graph_writer', None)
        if gw is None:
            raise SharedGraphError(
                f"{where}: this map owns no shared graph writer, and the shared graph is the "
                f"ONLY traversability store (the dense TraversabilityMap was removed). Writer-side "
                f"traversability code needs CMDbasedMapWithBackend.Initilize_map (which builds the "
                f"graph) to have run; map type={type(self).__name__}, "
                f"is_ElevationMap_built={getattr(self, 'is_ElevationMap_built', None)}")
        return gw

    def _graph_slots_at(self, pts):
        """
        (M, 2) world [x, y] -> (M,) int slot of the lattice node whose R x R estimation cell
        contains each point (its nearest lattice point: the cell Step 9 writes for that node),
        -1 when that lattice point lies outside the retained lattice-center bounds.
        nearest_node runs once per DISTINCT lattice point, and its hit must sit exactly on
        the lattice point (same check as _apply_graph_update).
        """
        gw = self._require_graph_writer('_graph_slots_at')
        R = float(self.TravUpdate_XYResolution)
        tol = float(self._graph_coord_tol)
        lxmin, lxmax, lymin, lymax = self._graph_lattice_bounds
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] == 0:
            return np.empty(0, dtype=np.int64)
        # k: (M, 2) integer lattice index of each point's nearest lattice point
        k = np.round(pts / R).astype(np.int64)
        uniq, inv = np.unique(k, axis=0, return_inverse=True)       # (U, 2), (M,)
        uniq_slots = np.full(len(uniq), -1, dtype=np.int64)          # (U,) slot per lattice point
        for u, (kx, ky) in enumerate(uniq):
            lx, ly = float(kx) * R, float(ky) * R
            if not (lxmin - tol <= lx <= lxmax + tol and lymin - tol <= ly <= lymax + tol):
                continue                                             # no node: outside the lattice
            slot = gw.nearest_node(lx, ly)
            if (slot == -1 or abs(float(gw.node_x[slot]) - lx) > tol
                    or abs(float(gw.node_y[slot]) - ly) > tol):
                raise SharedGraphError(
                    f"_graph_slots_at: in-domain lattice point ({lx:.6g}, {ly:.6g}) resolved to "
                    f"slot {slot}"
                    + ("" if slot == -1 else
                       f" at ({float(gw.node_x[slot]):.6g}, {float(gw.node_y[slot]):.6g})")
                    + f"; expected a node exactly there (coord_tol={tol:.3g}, R={R}) — "
                      f"lattice/graph mismatch")
            uniq_slots[u] = slot
        return uniq_slots[np.asarray(inv).reshape(-1)]

    def _graph_node_rows(self, pts, payload, fill):
        """
        (M, 2) world [x, y] -> (M, L) float64 copy of the named node payload ('cmd_v',
        'cmd_w' or 'meta') of each point's lattice node (_graph_slots_at), `fill` where a
        point has no node. With payload='meta' and fill=TRAV_NotInitialized this is the
        per-cell traversability status the removed TravMap_MetaInfo held.
        """
        slots = self._graph_slots_at(pts)                                    # (M,) int, -1 = none
        rows = np.full((len(slots), int(self.TraversabilityMap_size_layers)), fill,
                       dtype=np.float64)                                     # (M, L)
        ok = slots >= 0
        if ok.any():
            rows[ok] = np.asarray(self.graph_writer.node_payload(payload))[slots[ok]]
        return rows

    def _graph_lattice_values(self):
        """
        One coherent read of the graph's cmd payloads regridded onto the lattice. Returns
        (values, kx0, ky0): values (nx, ny, L, 2) float64 [cmd_v, cmd_w], x/y index ascending
        in world coordinate, and kx0/ky0 the integer lattice index of values[0, 0]
        (world = k * R). The writer reads its own arrays; a reader proxy (no writer) takes a
        torn-free snapshot through a lazily attached SharedGraphReader. Requires every active
        node on the R-lattice, tiling a full nx x ny grid (the writer's create_grid graph).
        """
        src = getattr(self, 'graph_writer', None)
        if src is None:
            # Reader process (planner proxy): attach once, then one snapshot per call, reusing
            # one buffer. Attaching fails loudly when the world model has not built the graph.
            if self._graph_lookup_reader is None:
                from statenav_global.world_model.shared_graph import SharedGraphReader
                self._graph_lookup_reader = SharedGraphReader(
                    GRAPH_LAYOUT, attach_timeout=float(cfg.get('graph_astar_attach_timeout', 10.0)))
            self._graph_lookup_snapshot = self._graph_lookup_reader.snapshot(
                into=self._graph_lookup_snapshot,
                max_retries=int(cfg.get('graph_astar_snapshot_max_retries', 50)))
            src = self._graph_lookup_snapshot
        hwm = int(src.node_hwm)
        active = np.asarray(src.node_active[:hwm]) == 1                       # (hwm,) bool
        x = np.asarray(src.node_x[:hwm], dtype=np.float64)[active]            # (n,) node x [m]
        y = np.asarray(src.node_y[:hwm], dtype=np.float64)[active]            # (n,) node y [m]
        if x.size == 0:
            raise SharedGraphError("_graph_lattice_values: the shared graph has no active node")
        R = float(self.TravUpdate_XYResolution)
        kx = np.rint(x / R).astype(np.int64)                                  # (n,) lattice index x
        ky = np.rint(y / R).astype(np.int64)                                  # (n,) lattice index y
        # Guard: every node on the lattice, and the nodes tile a full grid with no duplicate.
        tol = 1.0e-3                                                          # [m], far above float32 noise
        off = (np.abs(kx * R - x) > tol) | (np.abs(ky * R - y) > tol)
        kx0, ky0 = int(kx.min()), int(ky.min())
        nx, ny = int(kx.max()) - kx0 + 1, int(ky.max()) - ky0 + 1
        flat = (kx - kx0) * ny + (ky - ky0)
        if off.any() or nx * ny != x.size or np.unique(flat).size != x.size:
            raise SharedGraphError(
                f"_graph_lattice_values: {x.size} active nodes do not tile a full {nx}x{ny} "
                f"R={R} m lattice ({int(off.sum())} off-lattice); trav_estimation_resoultion "
                f"must match the world model's and the graph must be its create_grid lattice")
        n_layers = int(self.TraversabilityMap_size_layers)
        values = np.empty((nx, ny, n_layers, 2), dtype=np.float64)            # (nx, ny, L, 2)
        values[kx - kx0, ky - ky0, :, 0] = np.asarray(src.node_payload('cmd_v')[:hwm])[active]
        values[kx - kx0, ky - ky0, :, 1] = np.asarray(src.node_payload('cmd_w')[:hwm])[active]
        return values, kx0, ky0

    def graph_cmd_at(self, xs, ys, layers):
        """
        Nearest-node traversability lookup: (n,) world x, y [m] and heading layers ->
        (n, 2) float64 [cmd_v, cmd_w] of the lattice node nearest each point (the node whose
        R x R estimation cell contains it — the same rounding _graph_slots_at and the A*
        planner use). Lattice indices are clamped to the graph's extent, so every point gets a
        real node (a point past the outermost node reads that node). Works on the writer and on
        a reader proxy (see _graph_lattice_values); one coherent graph read per call.
        """
        xs = np.asarray(xs, dtype=np.float64).reshape(-1)          # (n,) world x per query [m]
        ys = np.asarray(ys, dtype=np.float64).reshape(-1)          # (n,) world y per query [m]
        layers = np.asarray(layers, dtype=np.int64).reshape(-1)    # (n,) heading layer per query
        if not (xs.shape == ys.shape == layers.shape):
            raise ValueError(f"graph_cmd_at: xs{xs.shape}, ys{ys.shape} and layers{layers.shape} "
                             f"must have equal length")
        if xs.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        n_layers = int(self.TraversabilityMap_size_layers)
        if not (np.isfinite(xs).all() and np.isfinite(ys).all()):
            raise ValueError(f"graph_cmd_at: non-finite query point(s): "
                             f"{int((~np.isfinite(xs) | ~np.isfinite(ys)).sum())} of {xs.size}")
        if layers.min() < 0 or layers.max() >= n_layers:
            raise ValueError(f"graph_cmd_at: heading layer out of range — "
                             f"[{layers.min()}, {layers.max()}] of {n_layers}")
        values, kx0, ky0 = self._graph_lattice_values()            # (nx, ny, L, 2)
        nx, ny = values.shape[:2]
        R = float(self.TravUpdate_XYResolution)
        # (n,) lattice index of each point's nearest node, clamped onto the existing lattice
        ix = np.clip(np.round(xs / R).astype(np.int64) - kx0, 0, nx - 1)
        iy = np.clip(np.round(ys / R).astype(np.int64) - ky0, 0, ny - 1)
        return values[ix, iy, layers]                              # (n, 2) [cmd_v, cmd_w]

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
        Write one map update's coarse-lattice values into the shared graph (the
        only traversability store) and maintain the aggregate too-risky topology
        invariant.

        graph_records : list of (x, y, theta_layer, cmd_v, cmd_w, meta) captured
            at the point each value is DECIDED — the _compute_travmap Step-9
            loop. The payload is
            carried in the record (direct write), never recomputed here.

        All predictable coordinate/payload validation completes before the outer
        edit_batch() opens. Any graph-mutation exception propagates out (the
        caller must stop, not keep using the graph): edit_batch() is only a
        reader publication barrier, not a rollback. Raises SharedGraphError when
        records arrive but no graph is owned (they would otherwise be lost).
        """
        if not graph_records:
            return
        gw = self._require_graph_writer('_apply_graph_update')

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
            # write): the exact cmd_v/cmd_w/meta for this point+layer.
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
            # round, not astype(int): a lattice point lies exactly on a cell boundary, where truncation
            # flips between two cells by float rounding and shifts the patch by one cell
            row_centers = np.round((self.map_xmax - xs) / self.map_resolution).astype(int)
            col_centers = np.round((self.map_ymax - ys) / self.map_resolution).astype(int)
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
            # CMD_CHUNK: number of velocity bins batched into a single inference call.
            # Higher = fewer kernel launches (faster); GPU memory scales with the PATCHES per
            # forward pass, so CMD_CHUNK is derived from the patch cap MAX_PATCHES below.
            # On CUDA OOM the patch cap is halved (down to 64 patches) and the chunk retried.
            # Skip inference entirely when size==0 (all patches were NaN/OOB): every
            # xyc_patchindex entry already has patchindex==-1, so Step 9 will fall back
            # to InitialGuess for all points without needing TravFormer output.
            # MAX_PATCHES: max patches per forward pass (planning_config travformer_max_batch_patches,
            # a multiple of 64). Bins per call = MAX_PATCHES // size, but never more than the former
            # fixed 2: the early exit below is checked once per call, and runtime updates (size up to
            # MAX_PATCHES / 2) keep exactly their previous batching. A call larger than the cap (one bin
            # of more than MAX_PATCHES patches) runs as MAX_PATCHES-sized slices (see the inference call).
            MAX_PATCHES = self.travformer_max_batch_patches
            MAX_BINS_PER_CALL = 2
            CMD_CHUNK = max(1, min(MAX_BINS_PER_CALL, MAX_PATCHES // size)) if size > 0 else 1
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
                    # Forward passes of at most MAX_PATCHES patches (GPU memory cap). Eval mode (BatchNorm on
                    # running stats, no dropout), so a patch's output does not depend on its slice.
                    outs = [self.TravFormer_Inference(None, patch_chunk[s:s + MAX_PATCHES], cmd_chunk[s:s + MAX_PATCHES],
                                                      model=self.elevonly_vw2instab_model, device=self.device)
                            for s in range(0, len(cmd_chunk), MAX_PATCHES)]
                    instab_mean_cat = np.concatenate([m for m, _ in outs], axis=0)
                    instab_std_cat = np.concatenate([sd for _, sd in outs], axis=0)
                except RuntimeError as e:
                    err = str(e).lower()
                    # Catch both OOM and "invalid configuration argument" (CUDA grid-dim overflow
                    # when batch×num_heads exceeds the 65535 hardware limit at large resolutions).
                    if ('out of memory' in err or 'invalid configuration argument' in err) and MAX_PATCHES > 64:
                        print(f"[GlobalMap] TravFormer pass of up to {MAX_PATCHES} patches failed ({e}); "
                              f"retrying with half the patch cap")
                        # halve the pass that actually failed (min(cap, rows of this call)); halving a
                        # cap larger than the call would retry the identical pass
                        MAX_PATCHES = max(64, min(MAX_PATCHES, len(cmd_chunk)) // 2 // 64 * 64)
                        CMD_CHUNK = max(1, min(MAX_BINS_PER_CALL, MAX_PATCHES // size))
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

            # ── Step 9: Write safe speeds back to the shared graph ───────────────────
            # Iterate all N query points in the same order as estimating_xycpts_set so that
            # patchindex_counter stays aligned with xyc_patchindex rows.
            #
            # graph_records: one record per (x, y, heading layer) this write-back decides,
            # CARRYING its cmd_v/cmd_w/meta (direct write); _apply_graph_update writes them
            # all into the graph (the only traversability store) after the loop.
            self._require_graph_writer('_compute_travmap')
            graph_records = []
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
                    # NaN in the patch: InitialGuess but TRAV_NotInitialized, so Case A re-queues the node
                    # once the missing cells are measured (a TRAV_Estimated stamp is never re-queued)
                    meta = int(self.TravmapInfo_IntEnum.TRAV_NotInitialized.value)
                else:
                    # target_v/w[patchindex] holds the largest speed that passed the VaR threshold,
                    # or toorisky_v/w (sentinel) if no speed was safe for this patch.
                    cmd_v = target_v[patchindex, 0]
                    cmd_w = target_w[patchindex, 0]
                    meta = int(self.TravmapInfo_IntEnum.TRAV_Estimated.value)

                # Discretize the canonical [-π, π) heading to the nearest layer.
                theta = Utils.wrap_to_pi(theta)
                theta_layer = self.get_theta_layer(theta)

                graph_records.append((
                    float(x), float(y), int(theta_layer),
                    float(cmd_v), float(cmd_w),
                    meta))

            # No fine-grid interpolation pass any more: consumers that need values between
            # lattice nodes interpolate the graph at read time (graph_cmd_at, MPPI's travgraph).

            # ── Shared-graph write + topology (Part 2) ───────────────────────────
            # Write the coarse-lattice cmd_v/cmd_w/meta into node payloads and maintain the
            # aggregate too-risky topology inside one edit_batch(). Graph-mutation
            # exceptions propagate.
            self._apply_graph_update(graph_records)
