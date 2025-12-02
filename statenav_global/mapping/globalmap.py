from time import time
import numpy as np
import torch
import os
import datetime
import math
import random
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from enum import IntEnum


from statenav_global.mapping.uncertainty_models import *

import hydra
from omegaconf import DictConfig, OmegaConf


import rospy
from std_msgs.msg import Float32MultiArray
from grid_map_msgs.msg import GridMap
from geometry_msgs.msg import PoseStamped


import tf.transformations as tf_trans
import transforms3d as tf3
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)




from pathlib import Path
cfg = OmegaConf.load(Path(__file__).parents[2] / "executables/configs/planning_config.yaml")

do_RRT_globalplanning = cfg.do_RRT_globalplanning

    
step_T = cfg.step_T
localmap_getwaypoint_horizonmultiplier = cfg.localmap_getwaypoint_horizonmultiplier
MPC_horizon = cfg.MPC_horizon
cost_weight = cfg.cost_weight
cost_power = cfg.cost_power
sampling_dist = cfg.sampling_dist
lookahead_distance = cfg.lookahead_distance

safecmd_sampling_dist = cfg.safecmd_sampling_dist
lookahead_angle = cfg.lookahead_angle

instability_std_multiplier = cfg.instability_std_multiplier
RRT_getwaypoint_steps = cfg.RRT_getwaypoint_steps


def set_random_seed(seed):
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    print(f"Set random seed to {seed} in numpy and torch.")
    return rng

def wrap_to_pi(angle):
    """Wraps an angle to the range [-π, π] using atan2."""
    return np.arctan2(np.sin(angle), np.cos(angle))





font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
times_new_roman = fm.FontProperties(fname=font_path)

FS_TICK: int = 15
FS_LABEL: int = 15
FS_LEGEND: int = 15
FS_TITLE: int = 20
PLOT_DPI: int=1200
PLOT_FORMAT: str='pdf'
RC_PARAMS: dict = {
    # Set background and border settings
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 2,
    'xtick.color': 'black',
    'ytick.color': 'black',
    "font.family": "Times New Roman",
    "font.family": 'serif',
    "font.serif": "Times New Roman",
}
savefigureonce_elevmap = cfg.savemap
savefigureonce_travmap = cfg.savemap





# batch putting of local elev map to global elev map
# 



class ElevmapInfo_IntEnum(IntEnum):

    # Lower number means more primitive status
    # Higher number implies lower number status

    ELEV_NotInitialized = 0
    ELEV_MEASURED = 1
    ELEV_NeedReconstruction = 2
    ELEV_Reconstructed = 3

class TravmapInfo_IntEnum(IntEnum):

    TRAV_NotInitialized = 0
    TRAV_NEEDUPDATE = 1
    TRAV_Estimated = 2
    StaticObstacle = 3
    


# localmap_from_globalmap_offset = 

# localmap_in_globalframe = 
# localmap_nan_mask = np.isnan(nan_map)
# self.ElevationMap[~localmap_nan_mask] = local_map[~localmap_nan_mask]



# updating_ElevationMap = ElevationMap[local_map_range_x, local_map_range_y]
# updating_metainfo_map = MetainfoMap[local_map_range_x, local_map_range_y, :]

# nearedge_mask = (localmap_in_globalframe < 0.5*self.local_patch_size) | (localmap_in_globalframe > self.ElevationMap_size - 0.5*self.local_patch_size)


# Trav_UPDATE_NEEDED_XYMASK = ~nan_map and ~nearedge_mask and \
#     (
#         ( (updating_ElevationMap == np.nan) and ~np.any(updating_metainfo_map[,,:] != UPDATE_NEEDED_TYPE.UPDATE_NOT_NEEDED) )\  
#         or ( np.abs(updating_ElevationMap - local_map) > self.elevation_diff_threshold)\
#             or (   )
        
        
#     )
#     # problem: nan value updates need to trigger nearby area as well by preest update res


# # priority 1: InitialGuess_cmd_v, obstacle_cmd_v stuffs?
# # elevmap recon and travmap update -> get box coordinates. take 3mx3m map area from elevmap and erase the box. give it to NN. get the recon. reassign to elevmap and travmap. But mark it as 'recon' and don't get from incoming elevmap
# # map publication & importing in RRT*
# # batch processing


# Metainfo_Map[local_map_range_x, local_map_range_y, :] = Trav_UPDATE_NEEDED_XYMASK


    
# MetainfoMap[local_map_range_x, local_map_range_y] = MetainfoMap[local_map_range_x, local_map_range_y]
# if ElevationMap[local_map_range_x, local_map_range_y] is nan:




















class BaseMap:
    def __init__(self, env_xmin, env_xmax, env_ymin, env_ymax, goal_x, goal_y, which_layer, preest_update_resolution, instab_limit, load_NN = True):


        self.goal_x = goal_x
        self.goal_y = goal_y


        # Robocentric Local Elevation Map from ROS
        self.is_RobocentricMap_received = False
        self.layer_data = {}

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
        self.robot_heading = cfg.heading_start


        # World frame Global Elevation Map
        self.is_ElevationMap_built = False
        self.ElevationMap = None
        self.ElevationMap_xmin = env_xmin
        self.ElevationMap_xmax = env_xmax
        self.ElevationMap_size_x = self.ElevationMap_xmax - self.ElevationMap_xmin
        self.ElevationMap_size_rows = None

        self.ElevationMap_ymin = env_ymin
        self.ElevationMap_ymax = env_ymax
        self.ElevationMap_size_y = self.ElevationMap_ymax - self.ElevationMap_ymin
        self.ElevationMap_size_cols = None

        # ElevationMap update methods
        self.which_layer = which_layer
        self.mean_over_maps = cfg.mean_over_maps # Boolean. If true, the first layer is used for data and the second layer is used for deciding if it is NaN


        # 3D (x,y,theta(=heading angle)) Traversability map that contains the pre-estimated cmd v and w. World frame
        self.is_TraversabilityMap_built = False
        self.TraversabilityMap = None
        self.TraversabilityMap_xmin = self.ElevationMap_xmin
        self.TraversabilityMap_xmax = self.ElevationMap_xmax
        self.TraversabilityMap_size_x = self.ElevationMap_xmax - self.ElevationMap_xmin
        self.TraversabilityMap_size_rows = None

        self.TraversabilityMap_ymin = self.ElevationMap_ymin
        self.TraversabilityMap_ymax = self.ElevationMap_ymax
        self.TraversabilityMap_size_y = self.ElevationMap_ymax - self.ElevationMap_ymin
        self.TraversabilityMap_size_cols = None

        self.TraversabilityMap_theta_min = -np.pi
        self.TraversabilityMap_theta_resolution = np.pi/4
        self.TraversabilityMap_theta_max = np.pi - self.TraversabilityMap_theta_resolution
        self.TraversabilityMap_size_layers = 2 * int(np.round(np.pi / self.TraversabilityMap_theta_resolution))
        print("Traversability Map size layers: ", self.TraversabilityMap_size_layers)




        # Trav and Velocity


        self.InitialGuess_cmd_v = None
        self.InitialGuess_cmd_w = None

        self.InitialGuess_auxiliary_score = None
        self.InitialGuess_auxiliary_score_std = None


        self.maximum_cmd_v = cfg.maximum_cmd_v
        self.maximum_cmd_w = cfg.maximum_cmd_w





        
        

        # TraversabilityMap update methods
        self.TravUpdate_XYResolution = preest_update_resolution
        self.TravUpdate_Pts = np.empty((0, 2))

        self.elevation_diff_threshold = cfg.elevation_diff_threshold
        self.RandomRate_TravUpdate = cfg.RandomRate_TravUpdate


        self.ElevmapInfo_IntEnum = ElevmapInfo_IntEnum
        self.is_ElevMap_MetaInfo_built = False
        self.ElevMap_MetaInfo = None
        self.ElevMap_MetaInfo_xmin = self.ElevationMap_xmin
        self.ElevMap_MetaInfo_xmax = self.ElevationMap_xmax
        self.ElevMap_MetaInfo_size_x = self.ElevationMap_xmax - self.ElevationMap_xmin
        self.ElevMap_MetaInfo_size_rows = None

        self.ElevMap_MetaInfo_ymin = self.ElevationMap_ymin
        self.ElevMap_MetaInfo_ymax = self.ElevationMap_ymax
        self.ElevMap_MetaInfo_size_y = self.ElevationMap_ymax - self.ElevationMap_ymin
        self.ElevMap_MetaInfo_size_cols = None


        
        self.TravmapInfo_IntEnum = TravmapInfo_IntEnum
        self.is_TravMap_MetaInfo_built = False
        self.TravMap_MetaInfo = None
        self.TravMap_MetaInfo_xmin = self.ElevationMap_xmin
        self.TravMap_MetaInfo_xmax = self.ElevationMap_xmax
        self.TravMap_MetaInfo_size_x = self.ElevationMap_xmax - self.ElevationMap_xmin
        self.TravMap_MetaInfo_size_rows = None

        self.TravMap_MetaInfo_ymin = self.ElevationMap_ymin
        self.TravMap_MetaInfo_ymax = self.ElevationMap_ymax
        self.TravMap_MetaInfo_size_y = self.ElevationMap_ymax - self.ElevationMap_ymin
        self.TravMap_MetaInfo_size_cols = None
        
        self.TravMap_MetaInfo_theta_min = self.TraversabilityMap_theta_min
        self.TravMap_MetaInfo_theta_resolution = self.TraversabilityMap_theta_resolution
        self.TravMap_MetaInfo_theta_max = self.TraversabilityMap_theta_max
        self.TravMap_MetaInfo_size_layers = self.TraversabilityMap_size_layers




        # Visualization
        self.viz_channel = 0  # default channel
        self.viz_vmin = 0.0
        self.viz_vmax = 0.5
        self.viz_title = "Traversability Map"
        self.viz_cbar_label = "Traversability Score"





        # NN related
        self.local_patch_size = cfg.local_patch_size  # in meters
        self.elevonly_vw2instab_model = None
        self.model_type = 'MLL'
        self.device = 'cuda'

                # Conditionally load model only if needed
        if load_NN:
            root_dir = os.path.dirname(os.path.abspath(__file__))
            elevonly_vw2instab_model_checkpoint = os.path.join(root_dir + "/checkpoints/elevonly_vw2instab.pth")
            
            def _load_model(checkpoint_path, model_type='MLL', device='cpu'):
                model = ElevationOnlyNetworkMLL()
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                model.to(device)
                model.eval()
                return model

            self.elevonly_vw2instab_model = _load_model(elevonly_vw2instab_model_checkpoint, model_type=self.model_type, device=self.device)
            print("Neural network model loaded.")

        else:
            print("Skipping neural network model loading (read-only mode).")










    @torch.no_grad()
    def _nn_inference(self, target_instability, elevation_map_numpy, commands_numpy, model, model_type='MLL', device='cpu'):

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



    def Initilize_map(self):

        if self.is_RobocentricMap_received:

        # Initialize global map
            if not self.is_ElevationMap_built:
                self.is_ElevationMap_built = True

            self.ElevationMap_size_rows = int(self.ElevationMap_size_x/self.map_resolution)
            self.ElevationMap_size_cols = int(self.ElevationMap_size_y/self.map_resolution)

            self.ElevationMap = np.full((self.ElevationMap_size_rows, self.ElevationMap_size_cols), np.nan)

        # Build global trav map  (flag not initialized here though)
        # 3D (x,y,theta(=heading angle)) global map that contains the pre-estimated cmd v and w array
            self.TraversabilityMap_size_rows = int(self.TraversabilityMap_size_x/self.map_resolution)
            self.TraversabilityMap_size_cols = int(self.TraversabilityMap_size_y/self.map_resolution)
            
            self.TraversabilityMap = np.full((self.TraversabilityMap_size_rows, self.TraversabilityMap_size_cols, self.TraversabilityMap_size_layers, 4), np.nan)
            self.TraversabilityMap[:,:,:,0] = self.InitialGuess_cmd_v
            self.TraversabilityMap[:,:,:,1] = self.InitialGuess_cmd_w
            self.TraversabilityMap[:,:,:,2] = self.InitialGuess_auxiliary_score
            self.TraversabilityMap[:,:,:,3] = self.InitialGuess_auxiliary_score_std

        # Build metainfo map
            self.ElevMap_MetaInfo_size_rows = int(self.ElevMap_MetaInfo_size_x/self.map_resolution)
            self.ElevMap_MetaInfo_size_cols = int(self.ElevMap_MetaInfo_size_y/self.map_resolution)
            self.ElevMap_MetaInfo = np.full((self.ElevMap_MetaInfo_size_rows, self.ElevMap_MetaInfo_size_cols), ElevmapInfo_IntEnum.ELEV_NotInitialized.value)


            self.TravMap_MetaInfo_size_rows = int(self.TravMap_MetaInfo_size_x/self.map_resolution)
            self.TravMap_MetaInfo_size_cols = int(self.TravMap_MetaInfo_size_y/self.map_resolution)
            self.TravMap_MetaInfo = np.full((self.TravMap_MetaInfo_size_rows, self.TravMap_MetaInfo_size_cols, self.TravMap_MetaInfo_size_layers), TravmapInfo_IntEnum.TRAV_NotInitialized.value)

        


    

        


    def Update_map(self, path_plan = None, visualize_map = False):
        
        '''
        # In grid map, the left top corner ( = (0,0) ) is the max x and y point in the map, regardless of the robot pose.
        # increasing row/col indexing results in decreasing x/y in world coordinate.
        # x increases as row decreases, y increases as col decreases
        # This rule is the same both for local and global map
        '''


        if not self.is_ElevationMap_built:
            self.Initilize_map()

        if self.is_RobocentricMap_received and self.is_ElevationMap_built:


            # Update global map
            # rospy.loginfo("Available layers in self.layer_data: %s", list(self.layer_data.keys()))

            if len(self.which_layer) == 1:
                local_map = self.layer_data[self.which_layer[0]]
                nan_map = local_map
            else:

                if self.mean_over_maps == True:
                    local_map = np.mean([self.layer_data[self.which_layer[i]] for i in range(len(self.which_layer))], axis=0)
                    nan_map = self.layer_data[self.which_layer[1]]
                else:
                    local_map = self.layer_data[self.which_layer[0]]
                    nan_map = self.layer_data[self.which_layer[1]]
                # the result will be NaN if one of the layers is NaN

            for row in range(local_map.shape[0]):
                for col in range(local_map.shape[1]):

                    if not np.isnan(nan_map[row, col]):

                        # for the current row,col in the local map, get the global x and y in the world frame
                        global_x = ( int( self.RobocentricMap_Rows/2) - row ) * self.map_resolution + self.RobocentricMap_CenterPosOffset_Worldfr[0]
                        global_y = ( int( self.RobocentricMap_Cols/2) - col ) * self.map_resolution + self.RobocentricMap_CenterPosOffset_Worldfr[1]

                        # corresponding row and col index in the global map
                        row_global = int( (self.ElevationMap_xmax - global_x)/self.map_resolution )
                        col_global = int( (self.ElevationMap_ymax - global_y)/self.map_resolution )
                        if row_global < 0 or row_global >= self.ElevationMap_size_rows or col_global < 0 or col_global >= self.ElevationMap_size_cols:
                            continue
                            # TODO: expand the global map by resizing it if out-of-range point is detected

                        


                        # if global map was previously nan
                        if self.ElevMap_MetaInfo[row_global, col_global] == self.ElevmapInfo_IntEnum.ELEV_NotInitialized.value:
                        # if np.isnan(self.ElevationMap[row_global, col_global]):

                            preest_res_floor_x = self.TravUpdate_XYResolution * math.floor(global_x/self.TravUpdate_XYResolution)
                            preest_res_floor_y = self.TravUpdate_XYResolution * math.floor(global_y/self.TravUpdate_XYResolution)

                            for estimating_gridpt in np.array([ [x,y] for x in np.arange(preest_res_floor_x - self.TravUpdate_XYResolution, preest_res_floor_x + self.TravUpdate_XYResolution, self.TravUpdate_XYResolution)\
                                                                        for y in np.arange(preest_res_floor_y - self.TravUpdate_XYResolution, preest_res_floor_y + self.TravUpdate_XYResolution, self.TravUpdate_XYResolution)]):
                                
                                estimating_pt_global_row = int( (self.ElevationMap_xmax - estimating_gridpt[0])/self.map_resolution )
                                estimating_pt_global_col = int( (self.ElevationMap_ymax - estimating_gridpt[1])/self.map_resolution )

                                if estimating_pt_global_row < 0 or estimating_pt_global_row >= self.ElevationMap_size_rows\
                                    or estimating_pt_global_col < 0 or estimating_pt_global_col >= self.ElevationMap_size_cols:
                                    # This pt is out of the global map. no need to preestimate
                                    continue
                                elif abs(self.ElevationMap_xmax - estimating_gridpt[0]) < 0.5*self.local_patch_size or abs(estimating_gridpt[0] - self.TraversabilityMap_xmin) < 0.5*self.local_patch_size\
                                    or abs(self.ElevationMap_ymax - estimating_gridpt[1]) < 0.5*self.local_patch_size or abs(estimating_gridpt[1] - self.TraversabilityMap_ymin) < 0.5*self.local_patch_size:
                                    # This pt is near the edge of the global map. no need to preestimate
                                    continue
                                elif np.any(np.all(self.TravUpdate_Pts == estimating_gridpt, axis=1)):
                                    # This pt is already in the list
                                    continue
                                elif np.any(self.TravMap_MetaInfo[estimating_pt_global_row, estimating_pt_global_col] != self.TravmapInfo_IntEnum.TRAV_NotInitialized.value):
                                # elif np.any(self.TraversabilityMap[estimating_pt_global_row, estimating_pt_global_col, :, 0] != self.InitialGuess_cmd_v):
                                    # This pt is already preestimated in some direction or obstacle. no need to preestimate
                                    continue
                                else:
                                    self.TravUpdate_Pts = np.append(self.TravUpdate_Pts, np.array([estimating_gridpt]), axis=0)
                        
                        # if the new value is significantly different from the old value, update the TraversabilityMap. 
                        # But if the region is under reconstruction, the case is handled separately.
                        elif np.abs(local_map[row, col] - self.ElevationMap[row_global, col_global]) > self.elevation_diff_threshold \
                            and self.ElevMap_MetaInfo[row_global, col_global] <= ElevmapInfo_IntEnum.ELEV_MEASURED.value:


                            preest_res_floor_x = self.TravUpdate_XYResolution * math.floor(global_x/self.TravUpdate_XYResolution)
                            preest_res_floor_y = self.TravUpdate_XYResolution * math.floor(global_y/self.TravUpdate_XYResolution)

                            for estimating_gridpt in np.array([ [x,y] for x in np.arange(preest_res_floor_x - self.TravUpdate_XYResolution, preest_res_floor_x + self.TravUpdate_XYResolution, self.TravUpdate_XYResolution)\
                                                                        for y in np.arange(preest_res_floor_y - self.TravUpdate_XYResolution, preest_res_floor_y + self.TravUpdate_XYResolution, self.TravUpdate_XYResolution)]):
                                
                                estimating_pt_global_row = int( (self.ElevationMap_xmax - estimating_gridpt[0])/self.map_resolution )
                                estimating_pt_global_col = int( (self.ElevationMap_ymax - estimating_gridpt[1])/self.map_resolution )

                                if estimating_pt_global_row < 0 or estimating_pt_global_row >= self.ElevationMap_size_rows\
                                    or estimating_pt_global_col < 0 or estimating_pt_global_col >= self.ElevationMap_size_cols:
                                    # This pt is out of the global map. no need to preestimate
                                    continue
                                elif abs(self.ElevationMap_xmax - estimating_gridpt[0]) < 0.5*self.local_patch_size or abs(estimating_gridpt[0] - self.TraversabilityMap_xmin) < 0.5*self.local_patch_size\
                                    or abs(self.ElevationMap_ymax - estimating_gridpt[1]) < 0.5*self.local_patch_size or abs(estimating_gridpt[1] - self.TraversabilityMap_ymin) < 0.5*self.local_patch_size:
                                    # This pt is near the edge of the global map. no need to preestimate
                                    continue
                                elif np.any(np.all(self.TravUpdate_Pts == estimating_gridpt, axis=1)):
                                    # This pt is already in the list
                                    continue
                                else:
                                    self.TravUpdate_Pts = np.append(self.TravUpdate_Pts, np.array([estimating_gridpt]), axis=0)



                        # For the cells that are nan-assigned, randomly take some of them to update if possible
                        # elif (np.any(self.TraversabilityMap[row_global, col_global, :, 0] == self.InitialGuess_cmd_v)\
                        elif (np.any(self.TravMap_MetaInfo[row_global, col_global, :] == self.TravmapInfo_IntEnum.TRAV_NotInitialized.value)\
                                and np.random.random() < self.RandomRate_TravUpdate): # To handle any missing preestimated points in the middle of already preestimated area
                            
                            preest_res_rounded_x = self.TravUpdate_XYResolution * round(global_x/self.TravUpdate_XYResolution)
                            preest_res_rounded_y = self.TravUpdate_XYResolution * round(global_y/self.TravUpdate_XYResolution)

                            near_preesimated_area = False
                            for nearby_gridpt in np.array([ [x,y] for x in np.arange(preest_res_rounded_x - self.TravUpdate_XYResolution, preest_res_rounded_x + self.TravUpdate_XYResolution, self.TravUpdate_XYResolution)\
                                                                        for y in np.arange(preest_res_rounded_y - self.TravUpdate_XYResolution, preest_res_rounded_y + self.TravUpdate_XYResolution, self.TravUpdate_XYResolution)]):
                                
                                nearby_gridpt_global_row = int( (self.ElevationMap_xmax - nearby_gridpt[0])/self.map_resolution )
                                nearby_gridpt_global_col = int( (self.ElevationMap_ymax - nearby_gridpt[1])/self.map_resolution )
                                
                                if nearby_gridpt_global_row < 0 or nearby_gridpt_global_row >= self.ElevationMap_size_rows\
                                    or nearby_gridpt_global_col < 0 or nearby_gridpt_global_col >= self.ElevationMap_size_cols:
                                    # This pt is out of the global map.
                                    continue
                                #if np.any(self.TraversabilityMap[nearby_gridpt_global_row, nearby_gridpt_global_col, :, 0] != self.InitialGuess_cmd_v):
                                if np.any(self.TravMap_MetaInfo[nearby_gridpt_global_row, nearby_gridpt_global_col, :] != self.TravmapInfo_IntEnum.TRAV_NotInitialized.value):
                                    near_preesimated_area = True
                                    break


                            if near_preesimated_area:

                                estimating_pt_global_row = int( (self.ElevationMap_xmax - preest_res_rounded_x)/self.map_resolution )
                                estimating_pt_global_col = int( (self.ElevationMap_ymax - preest_res_rounded_y)/self.map_resolution )


                                if estimating_pt_global_row < 0 or estimating_pt_global_row >= self.ElevationMap_size_rows\
                                    or estimating_pt_global_col < 0 or estimating_pt_global_col >= self.ElevationMap_size_cols:
                                    # This pt is out of the global map. no need to preestimate
                                    continue
                                elif abs(self.ElevationMap_xmax - preest_res_rounded_x) < 0.5*self.local_patch_size or abs(preest_res_rounded_x - self.TraversabilityMap_xmin) < 0.5*self.local_patch_size\
                                    or abs(self.ElevationMap_ymax - preest_res_rounded_y) < 0.5*self.local_patch_size or abs(preest_res_rounded_y - self.TraversabilityMap_ymin) < 0.5*self.local_patch_size:
                                    # This pt is near the edge of the global map. no need to preestimate
                                    continue
                                elif np.any(np.all(self.TravUpdate_Pts == [preest_res_rounded_x, preest_res_rounded_y], axis=1)):
                                    # This pt is already in the list
                                    continue
                                #elif np.all(self.TraversabilityMap[estimating_pt_global_row, estimating_pt_global_col, :, 0] != self.InitialGuess_cmd_v):
                                elif np.all(self.TravMap_MetaInfo[estimating_pt_global_row, estimating_pt_global_col, :] != self.TravmapInfo_IntEnum.TRAV_NotInitialized.value):
                                    # This pt is already preestimated in EVERY DIRECTIOn. no need to preestimate
                                    continue
                                else:
                                    self.TravUpdate_Pts = np.append(self.TravUpdate_Pts, np.array([[preest_res_rounded_x, preest_res_rounded_y]]), axis=0)
                        
                            


                        self.ElevationMap[row_global, col_global] = local_map[row, col]
                        if self.ElevMap_MetaInfo[row_global, col_global] == ElevmapInfo_IntEnum.ELEV_NotInitialized.value:
                            self.ElevMap_MetaInfo[row_global, col_global] = ElevmapInfo_IntEnum.ELEV_MEASURED.value




                              

        # Get traversability map
        if self.is_RobocentricMap_received and self.is_ElevationMap_built:

            deleting_indices = []
            for i in range(self.TravUpdate_Pts.shape[0]):

                estimating_gridpt = self.TravUpdate_Pts[i]
                estimating_pt_global_row = int( (self.ElevationMap_xmax - estimating_gridpt[0])/self.map_resolution )
                estimating_pt_global_col = int( (self.ElevationMap_ymax - estimating_gridpt[1])/self.map_resolution )

                if self.is_TraversabilityMap_built and np.isnan(self.ElevationMap[estimating_pt_global_row, estimating_pt_global_col]): 
                    # This pt is nan. no need to preestimate. turning off only when this is the first time to build preest map
                    deleting_indices.append(i)
                # elif self.TraversabilityMap[estimating_pt_global_row, estimating_pt_global_col, 0, 0] == self.obstacle_cmd_v:
                elif np.any(self.TravMap_MetaInfo[estimating_pt_global_row, estimating_pt_global_col, :] == self.TravmapInfo_IntEnum.StaticObstacle.value):
                    # This pt is already preestimated as obstacle. no need to preestimate
                    deleting_indices.append(i)
        
            self.TravUpdate_Pts = np.delete(self.TravUpdate_Pts, deleting_indices, axis=0)








            print("\nUpdating traversability map......")
            start_time = time()
            print(" Current robot position: {:.1f}, {:.1f}, {:.0f}".format(self.robot_x, self.robot_y, np.rad2deg(self.robot_heading)))
            print(" Global map range:      x-axis: ", self.ElevationMap_xmin, self.ElevationMap_xmax, "y-axis: ", self.ElevationMap_ymin, self.ElevationMap_ymax)
            
            
            # for i in range(self.TravUpdate_Pts.shape[0]):
            #     x = self.TravUpdate_Pts[i, 0]
            #     y = self.TravUpdate_Pts[i, 1]
            #     row = int((x - self.ElevationMap_xmin)/self.map_resolution)
            #     col = int((y - self.ElevationMap_ymin)/self.map_resolution)
            #     self.travmap_metainfo_grid[row, col] = self.MetaInfo_IntEnum.UPDATE_NEEDED.value


            self.get_travmap(self.TravUpdate_Pts, 0, 0, 0, 0)

            if not self.is_TraversabilityMap_built:
                self.is_TraversabilityMap_built = True
            self.TravUpdate_Pts = np.empty((0, 2))
            print("Traversability map updated. Total time of estimation: ", time()-start_time, "\n")






            theta_to_goal = np.arctan2(self.goal_y - self.robot_y, self.goal_x - self.robot_x)
            theta_robot = wrap_to_pi(self.robot_heading)
            if visualize_map:
                self.visualize_maps(theta_robot, path_plan)








    def visualize_maps(self, map_theta, path_plan = None):
        

        theta_layer = int(np.round((map_theta-self.TraversabilityMap_theta_min)/self.TraversabilityMap_theta_resolution))
        if theta_layer == self.TraversabilityMap_size_layers:
            theta_layer = 0 # pi = -pi


        path_plan = np.array(path_plan).reshape(-1,2) if path_plan is not None else None
        if path_plan is not None and path_plan.shape[0] > 0:
            path_plan_InGridMap = np.zeros_like(path_plan)
            for i in range(path_plan.shape[0]):
                path_plan_InGridMap[i,:] = self.xy2grid(path_plan[i,0], path_plan[i,1])




        # ####################################### Elevation map #######################################
        # Only visualize elevation map if it exists
        if self.ElevationMap is not None:
            global savefigureonce_elevmap
            if savefigureonce_elevmap:
                plt.close(1)
                plt.figure(1)
            else:
                # plt.ion()
                # fig, ax = plt.subplots(num=1)
                plt.close(1)
                plt.figure(1)
            manager = plt.get_current_fig_manager()
            manager.window.setGeometry(0, 0, 480, 420)  # Adjust the position and size as needed
            plt.rcParams.update(RC_PARAMS)
            plt.imshow(self.ElevationMap[:,:], cmap='viridis')
            plt.title("Global Elevation Map", fontproperties=times_new_roman, fontsize=FS_TITLE)
            if path_plan is not None and path_plan.shape[0] > 0:
                plt.plot([x[1] for x in path_plan_InGridMap], [x[0] for x in path_plan_InGridMap], "Dr", markersize=3) 
                plt.plot([x[1] for x in path_plan_InGridMap], [x[0] for x in path_plan_InGridMap], '-r', linewidth=1.5)
                plt.plot(path_plan_InGridMap[0,1], path_plan_InGridMap[0,0], marker="D",color=(0,1,1), markersize=6)
                plt.plot(path_plan_InGridMap[-1,1], path_plan_InGridMap[-1,0], marker="*",color=(1,1,0), markersize=10)
            
            cbar = plt.colorbar()
            cbar.set_label('Height (m)', fontsize=FS_LEGEND, fontproperties=times_new_roman)  # Addi
            plt.xlabel("Y-axis", fontsize=FS_LABEL, fontproperties=times_new_roman)
            plt.ylabel("X-axis", fontsize=FS_LABEL, fontproperties=times_new_roman)
            plt.xticks(np.linspace(0, self.ElevationMap.shape[1], int(self.ElevationMap_ymax-self.ElevationMap_ymin)+1), np.linspace(self.ElevationMap_ymax, self.ElevationMap_ymin, int(self.ElevationMap_ymax-self.ElevationMap_ymin+1) ))
            plt.yticks(np.linspace(0, self.ElevationMap.shape[0], int(self.ElevationMap_xmax-self.ElevationMap_xmin)+1), np.linspace(self.ElevationMap_xmax, self.ElevationMap_xmin, int(self.ElevationMap_xmax-self.ElevationMap_xmin+1) ))
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            # plt.minorticks_on()
            plt.tick_params(axis='both', which='major', labelsize=FS_TICK, width=2)
            for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
                label.set_fontproperties(times_new_roman)
                label.set_fontsize(FS_TICK)

            if savefigureonce_elevmap and path_plan is not None and path_plan.shape[0] > 0:
                figure_path = os.path.join(Path(__file__).parents[1], "./")
                os.makedirs(figure_path, exist_ok=True)
                plt.savefig(figure_path + 'World Elevation Map' + '.pdf', format=PLOT_FORMAT, dpi=PLOT_DPI, bbox_inches='tight')
                savefigureonce_elevmap = False
            plt.show(block=False)
            plt.pause(0.001)      






        ####################################### Traversability map #######################################
        global savefigureonce_travmap
        if savefigureonce_travmap:
            plt.close(2)
            plt.figure(2)
        else:
            plt.ion()
            fig, ax = plt.subplots(num=2)
            plt.clf()
            # plt.close(2)
            # plt.figure(2)
        manager = plt.get_current_fig_manager()
        # manager.window.setGeometry(0, 660, 480, 420)   # Adjust the position and size as needed
        manager.window.setGeometry(240, 0, 480, 420)
        plt.rcParams.update(RC_PARAMS)


        plt.imshow(self.TraversabilityMap[:,:,theta_layer,self.viz_channel], cmap='coolwarm_r', vmin=self.viz_vmin, vmax=self.viz_vmax)
        plt.title(f"Traversability Map", fontproperties=times_new_roman, fontsize=FS_TITLE)
        cbar = plt.colorbar()
        cbar.set_label(self.viz_cbar_label, fontproperties=times_new_roman, fontsize=FS_LEGEND)  #
        

        if path_plan is not None and path_plan.shape[0] > 0:
            # plt.plot([x[1] for x in path_plan_InGridMap], [x[0] for x in path_plan_InGridMap], "Dr", markersize=3) 
            plt.plot([x[1] for x in path_plan_InGridMap], [x[0] for x in path_plan_InGridMap], '--', color='lime', linewidth=4, label = 'Global Path Plan')
            # plt.plot(path_plan_InGridMap[0,1], path_plan_InGridMap[0,0], marker="D",color=(0,1,1), markersize=6)
            # plt.plot(path_plan_InGridMap[-1,1], path_plan_InGridMap[-1,0], marker="*",color=(1,1,0), markersize=10)
            pass

        plt.xlabel("Y-axis", fontproperties=times_new_roman, fontsize=FS_LABEL)
        plt.ylabel("X-axis", fontproperties=times_new_roman, fontsize=FS_LABEL)
        plt.xticks(np.linspace(0, self.TraversabilityMap.shape[1], int(self.TraversabilityMap_ymax-self.TraversabilityMap_ymin+1)), np.linspace(self.TraversabilityMap_ymax, self.TraversabilityMap_ymin, int(self.TraversabilityMap_ymax-self.TraversabilityMap_ymin+1)))
        plt.yticks(np.linspace(0, self.TraversabilityMap.shape[0], int(self.TraversabilityMap_xmax-self.TraversabilityMap_xmin+1)), np.linspace(self.TraversabilityMap_xmax, self.TraversabilityMap_xmin, int(self.TraversabilityMap_xmax-self.TraversabilityMap_xmin+1)))
        plt.grid(True)
        plt.tick_params(axis='both', which='major', labelsize=FS_TICK, width=2)
        for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
            label.set_fontproperties(times_new_roman)
            label.set_fontsize(FS_TICK)
        legend = plt.legend(loc='upper left', prop=times_new_roman, fontsize=FS_LEGEND)
        for text in legend.get_texts():
            text.set_fontsize(FS_LEGEND)  # 🔹 Force larger size

        if savefigureonce_travmap and path_plan is not None and path_plan.shape[0] > 0:
            figure_path = os.path.join(Path(__file__).parents[1], "./")
            os.makedirs(figure_path, exist_ok=True)
            plt.savefig(figure_path + 'Traversability Map' + '.pdf', format=PLOT_FORMAT, dpi=PLOT_DPI, bbox_inches='tight')
            savefigureonce_travmap = False
        plt.show(block=False)
        plt.pause(0.001)


    def get_travmap(self, estimating_xypts_set, xmin, xmax, ymin, ymax):
        # Base implementation or raise NotImplementedError
        raise NotImplementedError("Subclasses must implement get_travmap")






                



    
    def get_patch_in_elevmap(self, x_center, y_center, yaw, size):


        """
        Get a local patch of the global elevation map, centered at (x_center, y_center) with size x size. yaw-aligned direction 
        x_center, y_center: in the world frame
        size: in meters. width (=height) of the local patch
        yaw: in radian

        Returns:
        local_patch: a 2D array of size x size, which is the local patch of the global elevation map
        ismapnan: a boolean indicating whether the local patch contains any nan value
        """
        ismapnan = False

        if self.ElevationMap is not None:
            row_center = int( (self.ElevationMap_xmax - x_center)/self.map_resolution )
            col_center = int( (self.ElevationMap_ymax - y_center)/self.map_resolution )
            givenpoint_tf_grid = tf3.affines.compose(np.array([row_center, col_center, 0]), tf3.euler.euler2mat(0, 0, yaw, 'sxyz'), np.ones(3))

            halfsize_grid = np.round(0.5*size/self.map_resolution).astype(int)
            local_patch_index = np.array([ [i,j,0] for i in range(-halfsize_grid, halfsize_grid) for j in range(-halfsize_grid, halfsize_grid) ])
            # transform the local_patch_index by rotation of yaw and translation of the given point
            local_patch_index_homogeneous = np.hstack((local_patch_index, np.ones((local_patch_index.shape[0], 1))))
            local_patch_index_transformed = np.round( np.dot(givenpoint_tf_grid, local_patch_index_homogeneous.transpose()).transpose()[:, :2] ).astype(int)
            


            local_patch_flattened = np.full((halfsize_grid*2 * halfsize_grid*2), np.nan)
            # Get elevation map data at the indexes of the local_patch_index_transformed from the gripmaplistener
            for i in range(local_patch_index_transformed.shape[0]):
                
                if local_patch_index_transformed[i, 0] < 0 or local_patch_index_transformed[i, 0] >= self.ElevationMap_size_rows or local_patch_index_transformed[i, 1] < 0 or local_patch_index_transformed[i, 1] >= self.ElevationMap_size_cols:
                    local_patch_flattened[i] = np.nan
                    ismapnan = True

                elif np.isnan(self.ElevationMap[local_patch_index_transformed[i, 0], local_patch_index_transformed[i, 1]]):
                    local_patch_flattened[i] = np.nan
                    ismapnan = True
                else:
                    local_patch_flattened[i] = self.ElevationMap[local_patch_index_transformed[i, 0], local_patch_index_transformed[i, 1]]


            local_patch = local_patch_flattened.reshape((halfsize_grid*2, halfsize_grid*2))
            
            return local_patch, ismapnan



    def xy2grid(self, x, y):
        
        row = int( (self.TraversabilityMap_xmax - x)/self.map_resolution )
        col = int( (self.TraversabilityMap_ymax - y)/self.map_resolution )
        
        if row < 0 or row > self.TraversabilityMap_size_rows:
            print(f"Out of range. x: {x}, y: {y}, row: {row}, col: {col}")
            return None, None
        elif row == self.TraversabilityMap_size_rows:
            row = self.TraversabilityMap_size_rows - 1

        if col < 0 or col > self.TraversabilityMap_size_cols:
            print(f"Out of range. x: {x}, y: {y}, row: {row}, col: {col}")
            return None, None
        elif col == self.TraversabilityMap_size_cols:
            col = self.TraversabilityMap_size_cols - 1

        return row, col

    def grid2xy(self, row, col):
        x = self.TraversabilityMap_xmax - row * self.map_resolution
        y = self.TraversabilityMap_ymax - col * self.map_resolution
        return x, y


    def get_edge_cost(self, x1,y1,c1, x2,y2, mode, sampling_distance, num_interpoints):
        # Base implementation or raise NotImplementedError
        raise NotImplementedError("Subclasses must implement get_edge_cost")



    def get_cmd_limits(self):


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
                theta = wrap_to_pi(theta)

                x = self.robot_x + delta_dist * np.cos(theta)
                y = self.robot_y + delta_dist * np.sin(theta)

                row = int( (self.TraversabilityMap_xmax - x)/self.map_resolution )
                col = int( (self.TraversabilityMap_ymax - y)/self.map_resolution )
                

                theta_layer = int(np.round((theta-self.TraversabilityMap_theta_min)/self.TraversabilityMap_theta_resolution))
                if theta_layer == self.TraversabilityMap_size_layers:
                    theta_layer = 0 # pi = -pi

                if row < 0 or row >= self.TraversabilityMap_size_rows or col < 0 or col >= self.TraversabilityMap_size_cols:
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



    
    def pose_callback(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        quat = [q.x, q.y, q.z, q.w]
        roll, pitch, yaw = tf_trans.euler_from_quaternion(quat)
        msg_yaw = yaw
        self.robot_x = p.x
        self.robot_y = p.y
        self.robot_heading = wrap_to_pi(yaw)
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

        # Get the position of the grid map in the world frame
        self.RobocentricMap_CenterPosOffset_Worldfr = (msg.info.pose.position.x, msg.info.pose.position.y, msg.info.pose.position.z)
        self.RobocentricMap_CenterRotOffset_Worldfr = (msg.info.pose.orientation.x, msg.info.pose.orientation.y, msg.info.pose.orientation.z, msg.info.pose.orientation.w)
        # rospy.loginfo("Map position: %s" % str(self.RobocentricMap_CenterPosOffset_Worldfr))

        # Get the grid map layers
        layers = msg.layers
        # rospy.loginfo("Layers: %s" % layers)
        for i, layer in enumerate(layers):
            if not layer in self.which_layer:
                continue
            data_layer = np.array(msg.data[i].data)
            data_layer_reshaped = data_layer.reshape((self.RobocentricMap_Rows, self.RobocentricMap_Cols)).transpose() #Don't know why but transpose is needed. row indexing 
            self.layer_data[layer] = data_layer_reshaped
            

        # rospy.loginfo("Received GridMap message in %f seconds" % (time() - start))
































                                        








class CMDbasedMap(BaseMap):
    def __init__(self, env_xmin, env_xmax, env_ymin, env_ymax, goal_x, goal_y, which_layer, preest_update_resolution, instab_limit, load_NN = True):
        super().__init__(env_xmin, env_xmax, env_ymin, env_ymax, goal_x, goal_y, which_layer, preest_update_resolution, instab_limit, load_NN)


        self.InitialGuess_cmd_v = cfg.default_safe_cmd_v
        self.InitialGuess_cmd_w = cfg.default_safe_cmd_w

        self.InitialGuess_auxiliary_score = np.nan
        self.InitialGuess_auxiliary_score_std = np.nan


        # trav as velcity-based method specific
        self.toorisky_v = cfg.toorisky_v
        self.toorisky_w = cfg.toorisky_w
        self.v_res = self.maximum_cmd_v / cfg.num_commands
        self.w_res = self.maximum_cmd_w / cfg.num_commands

        self.instability_limit = instab_limit # = \delta_limit
        self.obstacle_cmd_v_threshold = cfg.obstacle_cmd_v_threshold
        self.obs_list = []

        # Visualization
        self.viz_channel = 0  # default channel
        self.viz_vmin = 0.0
        self.viz_vmax = 0.5
        self.viz_title = "Traversability Map"
        self.viz_cbar_label = "Linear Velocity (m/s)"





    def get_travmap(self, estimating_xypts_set, estimating_area_xmin, estimating_area_xmax, estimating_area_ymin, estimating_area_ymax):
        
        if estimating_xypts_set is None:
            print(" Estimating area: x-axis: {:.1f}, {:.1f}, y-axis: {:.1f}, {:.1f}".format(estimating_area_xmin, estimating_area_xmax, estimating_area_ymin, estimating_area_ymax))
                    
            if estimating_area_xmin < self.ElevationMap_xmin or estimating_area_xmax > self.ElevationMap_xmax or estimating_area_ymin < self.ElevationMap_ymin or estimating_area_ymax > self.ElevationMap_ymax:
                print("The estimating area is out of the global map.")


            # Grid points in the estimating area, grid size of self.TravUpdate_XYResolution. xyz coordinates
            estimating_xycpts_set = np.array([ [x,y,c] for x in np.arange(estimating_area_xmin, estimating_area_xmax + self.TravUpdate_XYResolution, self.TravUpdate_XYResolution) for y in np.arange(estimating_area_ymin, estimating_area_ymax + self.TravUpdate_XYResolution, self.TravUpdate_XYResolution)\
                                                for c in np.arange(self.TraversabilityMap_theta_min, self.TraversabilityMap_theta_max + self.TraversabilityMap_theta_resolution, self.TraversabilityMap_theta_resolution) ])
        
        else:
            # estimating_xypts_set is ndarray of [x,y]. I want to add one more dimension for theta
            estimating_xycpts_set = np.array([ [xy[0], xy[1], c] for xy in estimating_xypts_set\
                                               for c in np.arange(self.TraversabilityMap_theta_min, self.TraversabilityMap_theta_max + self.TraversabilityMap_theta_resolution, self.TraversabilityMap_theta_resolution) ])

        print(" Estimating", estimating_xypts_set.shape[0], " points in total...")
        # print(" Estimating xypts are: \n", estimating_xypts_set)





        if self.is_RobocentricMap_received and self.is_ElevationMap_built and estimating_xycpts_set.shape[0] != 0:
            
            counter = 0
            total_pts = estimating_xycpts_set.shape[0]


            xyc_patchindex = np.concatenate((estimating_xycpts_set, np.zeros((estimating_xycpts_set.shape[0],1), dtype=int)), axis=1)
            patchindex_counter = 0
            patch_tensor = np.empty((0, 2*np.round(0.5*self.local_patch_size/self.map_resolution).astype(int), 2*np.round(0.5*self.local_patch_size/self.map_resolution).astype(int)))


            for xyc in estimating_xycpts_set:
                
                x = xyc[0]
                y = xyc[1]
                theta = xyc[2]

                patch, ismapnan = self.get_patch_in_elevmap(x, y, theta, self.local_patch_size)

                if ismapnan:
                    xyc_patchindex[patchindex_counter, 3] = -1 # -1 means NaN 
                    patchindex_counter += 1
                    continue


                # offset compensation for the patch (getting rid of absolute height)
                # patch_at_center = patch[int(patch.shape[0]//2 + 3), patch.shape[1] // 2 - 4]
                patch_at_center = np.percentile(patch, 5)
                patch = patch - patch_at_center
                patch_tensor = np.concatenate((patch_tensor, patch[np.newaxis, :, :]), axis=0)
                xyc_patchindex[patchindex_counter, 3] = patch_tensor.shape[0] - 1
                patchindex_counter += 1

            
            # Inference with patch_tensor
            # patch_tensor_extended = np.concatenate((patch_tensor, np.zeros((64-patch_tensor.shape[0]%64, patch_tensor.shape[1], patch_tensor.shape[2]))), axis=0)
            # cmd_v_tensor = np.concatenate( (np.full((patch_tensor_extended.shape[0],1), 0.5), np.full((patch_tensor_extended.shape[0],1), 0)), axis=1)

            # [instab, std_pred, target_v] = self._nn_inference(self.instability_limit, patch_tensor_extended, cmd_v_tensor, model=self.elevonly_v2instab_model, model_type=self.model_type, device=self.device)
             # max_iter = 2
            # iter = 0
            # while iter < max_iter:

            #     if iter==0 and target_v > 0.5:
            #         target_v = 0.5
            #         break
            #     elif target_v < 0:
            #         target_v = 0.01
            
            #     if abs(instab[0][0] + 2*math.exp(std_pred[0][0]) - self.instability_limit) < 0.1:
            #         break
                
            #     [instab, std_pred, target_v] = self._nn_inference(self.instability_limit, patch, [target_v,0], model=self.elevonly_v2instab_model, model_type=self.model_type, device=self.device)

            #     iter += 1
                    
            # if target_v > 0.5:
            #     target_v = 0.5
            # elif target_v < 0:
            #     target_v = 0.01
            # cmd_v = target_v


            # [instab, std_pred, target_v] = self._nn_inference(self.instability_limit, patch, [0.5,0], model=self.elevonly_v2instab_model, model_type=self.model_type, device=self.device)
            # self.mean_time = (self.mean_time*self.inference_counter + (time() - start))/(self.inference_counter+1)
            # self.inference_counter += 1
            # using another model
            # [instab, std_pred, target_v] = self._nn_inference(self.instability_limit, patch, [0.5,0], model=self.elevonly_lateraldrift_model, model_type=self.model_type, device=self.device)
            
            patch_tensor_extended = np.concatenate((patch_tensor, np.zeros((64-patch_tensor.shape[0]%64, patch_tensor.shape[1], patch_tensor.shape[2]))), axis=0) # to make the batch size 64 multiple
            patch_cat_tensor = np.concatenate((patch_tensor_extended, patch_tensor_extended), axis=0) # to get both for cmdv and cmdw
            size = patch_tensor_extended.shape[0]

            target_v = np.full((size,1), self.toorisky_v)
            v_res = self.v_res
            v_bins = np.arange(v_res, self.maximum_cmd_v+v_res, v_res)
            v_bins = v_bins[::-1]

            target_w = np.full((size,1), self.toorisky_w)
            w_res = self.w_res
            w_bins = np.arange(w_res, self.maximum_cmd_w+w_res, w_res)
            w_bins = w_bins[::-1]


            print_once = True
            mean_time = 0
            inference_counter = 0
            for i in range(len(v_bins)):
                v = v_bins[i]
                w = w_bins[i]
                cmd_cat_tensor = np.zeros((2*size, 2))
                cmd_cat_tensor[:size,0] = v
                cmd_cat_tensor[size:,1] = w

                start = time()
                [instab_mean_cat, instab_std_cat] = self._nn_inference(None, patch_cat_tensor, cmd_cat_tensor, model=self.elevonly_vw2instab_model, model_type=self.model_type, device=self.device)
                mean_time = (mean_time*inference_counter + (time() - start))/(inference_counter+1)
                inference_counter += 1

                VaR_instab_cat = instab_mean_cat[:,0] + instability_std_multiplier*instab_std_cat[:,0]

                for i in range(size):
                    if VaR_instab_cat[i] < self.instability_limit and target_v[i] == self.toorisky_v:
                        target_v[i] = v
                    if VaR_instab_cat[size + i] < self.instability_limit and target_w[i] == self.toorisky_w:
                        target_w[i] = w
            




            
            patchindex_counter = 0
            for xyc in estimating_xycpts_set:

                x = xyc[0]
                y = xyc[1]
                theta = xyc[2]

                cmd_v = 0
                cmd_w = 0

                patchindex = xyc_patchindex[patchindex_counter, 3].astype(int)
                patchindex_counter += 1

                if patchindex == -1: # NaN
                    cmd_v = self.InitialGuess_cmd_v
                    cmd_w = self.InitialGuess_cmd_w
                else:
                    cmd_v = target_v[patchindex,0]
                    cmd_w = target_w[patchindex,0]
                    
                    
                    isobsinthelist = False
                    for obs in self.obs_list:
                        if x == obs[0] and y == obs[1]:
                            isobsinthelist = True
                            break

                    if cmd_v < self.obstacle_cmd_v_threshold and not isobsinthelist: # if cmd_v is too low, consider it as an obstacle.
                        row, col = self.xy2grid(x, y)
                        if np.all(self.TraversabilityMap[row, col, :, 0] < self.obstacle_cmd_v_threshold):
                            self.obs_list.append([x,y])
                    elif cmd_v >= self.obstacle_cmd_v_threshold and isobsinthelist: # If it was obstacle but now okay with the updated perception, remove it from the list
                        self.obs_list.remove([x,y])

                    #
                    


        

                # Assign cmd_v and cmd_w to the 3D map. Square with size of 'self.TravUpdate_XYResolution' centered at x and y at each pt has the same cmd_v and cmd_w
                square_BL_xyc = np.array([x - self.TravUpdate_XYResolution/2, y - self.TravUpdate_XYResolution/2])
                square_TR_xyc = np.array([x + self.TravUpdate_XYResolution/2, y + self.TravUpdate_XYResolution/2])

                square_BL_entry = np.array([ round( (self.TraversabilityMap_xmax - square_BL_xyc[0])/self.map_resolution ), round( (self.TraversabilityMap_ymax - square_BL_xyc[1])/self.map_resolution ) ])
                square_TR_entry = np.array([ round( (self.TraversabilityMap_xmax - square_TR_xyc[0])/self.map_resolution ), round( (self.TraversabilityMap_ymax - square_TR_xyc[1])/self.map_resolution ) ])
                
                theta = wrap_to_pi(theta)
                theta_layer = int(np.round((theta-self.TraversabilityMap_theta_min)/self.TraversabilityMap_theta_resolution))
                if theta_layer == self.TraversabilityMap_size_layers:
                    theta_layer = 0 # pi = -pi

                for row in range(square_TR_entry[0], square_BL_entry[0]):
                    for col in range(square_TR_entry[1], square_BL_entry[1]):
                        
                        if row < 0 or row >= self.TraversabilityMap_size_rows or col < 0 or col >= self.TraversabilityMap_size_cols:
                            continue
                        else:
                            self.TraversabilityMap[row, col, theta_layer, 0] = cmd_v
                            self.TraversabilityMap[row, col, theta_layer, 1] = cmd_w
                            self.TravMap_MetaInfo[row, col, theta_layer] = self.TravmapInfo_IntEnum.TRAV_Estimated.value



            # Interpolation
            for xyc in estimating_xycpts_set:

                isobsinthelist = False
                for obs in self.obs_list:
                    if xyc[0] == obs[0] and xyc[1] == obs[1]:
                        isobsinthelist = True
                        break

                if isobsinthelist:
                    # continue
                    pass
                    # Don't interpolate and keep it  for dangerous area. interpolating makes the dangerous area smaller


                update_size = math.floor(self.TravUpdate_XYResolution/self.map_resolution)



                for quadrant in range(0, 4): # quadrants are defined in array space
                    if quadrant == 0:
                        x_BLCornerQuadrant = xyc[0]
                        y_BLCornerQuadrant = xyc[1]

                        x_TLCornerQuadrant = xyc[0] + self.TravUpdate_XYResolution
                        y_TLCornerQuadrant = xyc[1]

                        x_BRCornerQuadrant = xyc[0]
                        y_BRCornerQuadrant = xyc[1] - self.TravUpdate_XYResolution

                        x_TRCornerQuadrant = xyc[0] + self.TravUpdate_XYResolution
                        y_TRCornerQuadrant = xyc[1] - self.TravUpdate_XYResolution
                        

                    elif quadrant == 1: # 90 degree rotated ccw
                        x_BLCornerQuadrant = xyc[0]
                        y_BLCornerQuadrant = xyc[1]

                        x_TLCornerQuadrant = xyc[0]
                        y_TLCornerQuadrant = xyc[1] + self.TravUpdate_XYResolution

                        x_BRCornerQuadrant = xyc[0] + self.TravUpdate_XYResolution
                        y_BRCornerQuadrant = xyc[1]

                        x_TRCornerQuadrant = xyc[0] + self.TravUpdate_XYResolution
                        y_TRCornerQuadrant = xyc[1] + self.TravUpdate_XYResolution


                    elif quadrant == 2: # 180 degree rotated ccw
                        x_BLCornerQuadrant = xyc[0]
                        y_BLCornerQuadrant = xyc[1]

                        x_TLCornerQuadrant = xyc[0] - self.TravUpdate_XYResolution
                        y_TLCornerQuadrant = xyc[1]

                        x_BRCornerQuadrant = xyc[0]
                        y_BRCornerQuadrant = xyc[1] + self.TravUpdate_XYResolution

                        x_TRCornerQuadrant = xyc[0] - self.TravUpdate_XYResolution
                        y_TRCornerQuadrant = xyc[1] + self.TravUpdate_XYResolution


                    elif quadrant == 3: # 270 degree rotated ccw
                        x_BLCornerQuadrant = xyc[0]
                        y_BLCornerQuadrant = xyc[1]

                        x_TLCornerQuadrant = xyc[0] 
                        y_TLCornerQuadrant = xyc[1] - self.TravUpdate_XYResolution

                        x_BRCornerQuadrant = xyc[0] - self.TravUpdate_XYResolution
                        y_BRCornerQuadrant = xyc[1]

                        x_TRCornerQuadrant = xyc[0] - self.TravUpdate_XYResolution
                        y_TRCornerQuadrant = xyc[1] - self.TravUpdate_XYResolution






                    theta = xyc[2]
                    theta = wrap_to_pi(theta)
                    theta_layer = math.floor(np.round((theta-self.TraversabilityMap_theta_min)/self.TraversabilityMap_theta_resolution))
                    #Code not implemented which will be used for checking the duplicate quadrants
                    if theta_layer == self.TraversabilityMap_size_layers:
                        theta_layer = 0 # pi = -pi



                    for map_layer in range(4):

                        row_BLCornerQuadrant, col_BLCornerQuadrant = self.xy2grid(x_BLCornerQuadrant, y_BLCornerQuadrant)
                        if row_BLCornerQuadrant is None or col_BLCornerQuadrant is None:
                            continue
                        Trav_BLCornerQuadrant = self.TraversabilityMap[row_BLCornerQuadrant,col_BLCornerQuadrant,theta_layer,map_layer]

                        row_TLCornerQuadrant, col_TLCornerQuadrant = self.xy2grid(x_TLCornerQuadrant, y_TLCornerQuadrant)
                        if row_TLCornerQuadrant is None or col_TLCornerQuadrant is None:
                            continue
                        Trav_TLCornerQuadrant = self.TraversabilityMap[row_TLCornerQuadrant,col_TLCornerQuadrant,theta_layer,map_layer]

                        row_TRCornerQuadrant, col_TRCornerQuadrant = self.xy2grid(x_TRCornerQuadrant, y_TRCornerQuadrant)
                        if row_TRCornerQuadrant is None or col_TRCornerQuadrant is None:
                            continue
                        Trav_TRCornerQuadrant = self.TraversabilityMap[row_TRCornerQuadrant,col_TRCornerQuadrant,theta_layer,map_layer]

                        row_BRCornerQuadrant, col_BRCornerQuadrant = self.xy2grid(x_BRCornerQuadrant, y_BRCornerQuadrant)
                        if row_BRCornerQuadrant is None or col_BRCornerQuadrant is None:
                            continue
                        Trav_BRCornerQuadrant = self.TraversabilityMap[row_BRCornerQuadrant,col_BRCornerQuadrant,theta_layer,map_layer]

                        for x_in_quadrant in range(0, (update_size+1) ):#//2+1):
                            for y_in_quadrant in range(0, (update_size+1) ):#//2+1):
                                trav = ( 1 - y_in_quadrant/update_size) * ( (x_in_quadrant/update_size) * (Trav_BRCornerQuadrant - Trav_BLCornerQuadrant) + Trav_BLCornerQuadrant ) \
                                    + (y_in_quadrant/update_size) * ( (x_in_quadrant/update_size) * (Trav_TRCornerQuadrant - Trav_TLCornerQuadrant) + Trav_TLCornerQuadrant )

                                if quadrant == 0:
                                    self.TraversabilityMap[row_BLCornerQuadrant - y_in_quadrant, col_BLCornerQuadrant + x_in_quadrant, theta_layer, map_layer] = trav
                                elif quadrant == 1:
                                    self.TraversabilityMap[row_BLCornerQuadrant - x_in_quadrant, col_BLCornerQuadrant - y_in_quadrant, theta_layer, map_layer] = trav
                                elif quadrant == 2:
                                    self.TraversabilityMap[row_BLCornerQuadrant + y_in_quadrant, col_BLCornerQuadrant - x_in_quadrant, theta_layer, map_layer] = trav
                                elif quadrant == 3:
                                    self.TraversabilityMap[row_BLCornerQuadrant + x_in_quadrant, col_BLCornerQuadrant + y_in_quadrant, theta_layer, map_layer] = trav






    def get_edge_cost(self, x1,y1,c1, x2,y2, mode, sampling_distance, num_interpoints):

        '''
        Get travel time for the vertex btwn nodes (x1,y1) and (x2,y2)
        preestimated cmd_v and cmd_w are used to calculate the travel time
        '''

        if x1 == x2 and y1 == y2:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!! You are getting the cost btwn the same nodes !!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print('x1, y1, x2, y2', x1, y1, x2, y2)
            return 0

        if not self.is_TraversabilityMap_built:
            return math.hypot(x2-x1, y2-y1)
        

        vertex_distance = math.hypot(x2-x1, y2-y1)


        if abs(x2 - x1) < 1e-3:
            c2 = np.pi/2
        else:
            c2 = (y2-y1)/(x2-x1)
        theta_1 = wrap_to_pi(c1)
        theta_2 = np.arctan2(y2-y1, x2-x1)

        # Get the equation of line between (x1,y1,c1) and (x2,y2,c2) 
        my = c2
        ny = y1 - my*x1

        mth = wrap_to_pi(theta_2 - theta_1) / (x2 - x1)
        nth = theta_1 - mth*x1

        # convert the x1 and x2 to the row index in the global map preestimated
        x_min = min(x1, x2)
        x_max = max(x1, x2)
        y_min = min(y1, y2)
        y_max = max(y1, y2)

        row_xmin = int( (self.TraversabilityMap_xmax - x_min)/self.map_resolution )
        row_xmax = int( (self.TraversabilityMap_xmax - x_max)/self.map_resolution )
        col_ymin = int( (self.TraversabilityMap_ymax - y_min)/self.map_resolution )
        col_ymax = int( (self.TraversabilityMap_ymax - y_max)/self.map_resolution )




        # Get the travel time for the line between (x1,y1) and (x2,y2)
        travel_time = 0


        if row_xmin == row_xmax and col_ymin == col_ymax:

            theta_1 = wrap_to_pi(theta_1)
            theta_layer = int(np.round((theta_1-self.TraversabilityMap_theta_min)/self.TraversabilityMap_theta_resolution))
            if theta_layer == self.TraversabilityMap_size_layers:
                theta_layer = 0 # pi = -pi
            v = self.TraversabilityMap[row_xmin, col_ymin, theta_layer][0]
            w = self.TraversabilityMap[row_xmin, col_ymin, theta_layer][1]

            segment_length = vertex_distance
            segment_anglechange = np.abs( (segment_length/vertex_distance) * wrap_to_pi(theta_2 - theta_1))


            travel_time = segment_length/v + segment_anglechange/(w/step_T)

        else:

            if mode == 'ByCell':
                intersections_row = np.arange( row_xmax + 1, row_xmin + 1, 1)
                intersections_x = self.TraversabilityMap_xmax - intersections_row*self.map_resolution
                intersections_y = my*intersections_x + ny
                intersections_th = mth*intersections_x + nth
                intersections_1 = np.concatenate((intersections_x.reshape(-1,1), intersections_y.reshape(-1,1), intersections_th.reshape(-1,1)), axis=1)

                intersections_col = np.arange( col_ymax + 1, col_ymin + 1, 1)
                intersections_y = self.TraversabilityMap_ymax - intersections_col*self.map_resolution
                intersections_x = (intersections_y - ny)/my
                intersections_th = mth*intersections_x + nth
                intersections_2 = np.concatenate((intersections_x.reshape(-1,1), intersections_y.reshape(-1,1), intersections_th.reshape(-1,1)), axis=1)

                intersections = np.concatenate((intersections_1, intersections_2), axis=0)
                intersections = np.concatenate((intersections, np.array([[x1,y1,c1], [x2,y2,c2]])), axis=0)
                intersections = intersections[intersections[:,0].argsort()]

            elif mode == 'SamplingInEdge':
                
                if sampling_distance is not None:
                    num_interpoints = int(math.hypot(x2-x1, y2-y1)/sampling_distance) + 1
                elif num_interpoints is not None:
                    pass
                else:
                    raise ValueError("Either sampling_distance or num_interpoints should be provided")

                x_interpoints = np.linspace(x1, x2, num_interpoints)
                y_interpoints = np.linspace(y1, y2, num_interpoints)
                c_interpoints = np.linspace(c1, c2, num_interpoints)
                intersections = np.concatenate((x_interpoints.reshape(-1,1), y_interpoints.reshape(-1,1), c_interpoints.reshape(-1,1)), axis=1)

            else:
                raise ValueError("mode should be either 'ByCell' or 'SamplingInEdge'")

            for i in range(intersections.shape[0]-1):

                x_i1 = intersections[i,0]
                y_i1 = intersections[i,1]
                c_i1 = intersections[i,2]
                x_i2 = intersections[i+1,0]
                y_i2 = intersections[i+1,1]
                c_i2 = intersections[i+1,2]

                [row, col] = self.xy2grid(x_i1, y_i1)

                c_i1 = wrap_to_pi(c_i1)
                if np.isnan(c_i1) or np.isnan((c_i1-self.TraversabilityMap_theta_min)/self.TraversabilityMap_theta_resolution):
                    print("c_i1 is NaN")
                    print(x1, y1, x2, y2, c1, c2)
                    print(intersections)
                theta_layer = int(np.round((c_i1-self.TraversabilityMap_theta_min)/self.TraversabilityMap_theta_resolution))
                if theta_layer == self.TraversabilityMap_size_layers:
                    theta_layer = 0 # pi = -pi


                # print(intersections, i)
                # print(x_i1, y_i1, x_i2, y_i2, row, col, theta_layer)
                # print(row, col, theta_layer)
                segment_length = math.hypot(x_i2-x_i1, y_i2-y_i1)
                segment_anglechange = np.abs( (segment_length/vertex_distance) * wrap_to_pi(c_i2 - c_i1) )
                v = self.TraversabilityMap[row, col, theta_layer, 0]
                w = self.TraversabilityMap[row, col, theta_layer, 1]


                travel_time += segment_length/v + segment_anglechange/(w/step_T)




        return travel_time











class ScoreBasedMap(BaseMap):
    
    def __init__(self, env_xmin, env_xmax, env_ymin, env_ymax, goal_x, goal_y, which_layer, preest_update_resolution, instab_limit, load_NN = True):
        super().__init__(env_xmin, env_xmax, env_ymin, env_ymax, goal_x, goal_y, which_layer, preest_update_resolution, instab_limit, load_NN)


        self.InitialGuess_cmd_v = cfg.scorebased_safe_cmd_v
        self.InitialGuess_cmd_w = cfg.scorebased_safe_cmd_w

        self.InitialGuess_auxiliary_score = 0.5 # used for preest[:,:,:,2] which is for trav score not ld
        self.InitialGuess_auxiliary_score_std = 0
        
        self.local_cmd_option = cfg.local_cmd_option
        self.cost_at_best = None

        # Visualization
        self.viz_channel = 2  # default channel
        self.viz_vmin = 0.0
        self.viz_vmax = 1
        self.viz_title = "Traversability Map"
        self.viz_cbar_label = "Traversability Score"




    def convert_to_score(self, cost):

        if self.cost_at_best is None:
            raise ValueError("cost_at_best is not set")
        
        trav_score = self.cost_at_best / cost if cost > 1e-2 else 1

        if trav_score > 1:
            trav_score = 1

        return trav_score




    def get_edge_cost(self, x1,y1,c1, x2,y2, mode, sampling_distance, num_interpoints):



        if x1 == x2 and y1 == y2:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!! You are getting the cost btwn the same nodes !!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print('x1, y1, x2, y2', x1, y1, x2, y2)
            return 0

        if not self.is_RobocentricMap_received or not self.is_ElevationMap_built or not self.is_TraversabilityMap_built:
            return math.hypot(x2-x1, y2-y1)
        

        vertex_distance = math.hypot(x2-x1, y2-y1)


        if abs(x2 - x1) < 1e-3:
            c2 = np.pi/2
        else:
            c2 = (y2-y1)/(x2-x1)
        theta_1 = wrap_to_pi(c1)
        theta_2 = np.arctan2(y2-y1, x2-x1)

        # Get the equation of line between (x1,y1,c1) and (x2,y2,c2) 
        my = c2
        ny = y1 - my*x1

        mth = wrap_to_pi(theta_2 - theta_1) / (x2 - x1)
        nth = theta_1 - mth*x1

        # convert the x1 and x2 to the row index in the global map preestimated
        x_min = min(x1, x2)
        x_max = max(x1, x2)
        y_min = min(y1, y2)
        y_max = max(y1, y2)

        row_xmin = int( (self.TraversabilityMap_xmax - x_min)/self.map_resolution )
        row_xmax = int( (self.TraversabilityMap_xmax - x_max)/self.map_resolution )
        col_ymin = int( (self.TraversabilityMap_ymax - y_min)/self.map_resolution )
        col_ymax = int( (self.TraversabilityMap_ymax - y_max)/self.map_resolution )




        # Get the travel time for the line between (x1,y1) and (x2,y2)
        cost = 0


        if row_xmin == row_xmax and col_ymin == col_ymax:

            theta_1 = wrap_to_pi(theta_1)
            theta_layer = int(np.round((theta_1-self.TraversabilityMap_theta_min)/self.TraversabilityMap_theta_resolution))
            if theta_layer == self.TraversabilityMap_size_layers:
                theta_layer = 0 # pi = -pi
            v = self.TraversabilityMap[row_xmin, col_ymin, theta_layer][0]
            w = self.TraversabilityMap[row_xmin, col_ymin, theta_layer][1]

            segment_length = vertex_distance
            
            cost = ( 1 + cost_weight * (1/self.TraversabilityMap[row_xmin, col_ymin, theta_layer, 2])**cost_power)* segment_length

                

        else:


            if mode == 'ByCell':
                intersections_row = np.arange( row_xmax + 1, row_xmin + 1, 1)
                intersections_x = self.TraversabilityMap_xmax - intersections_row*self.map_resolution
                intersections_y = my*intersections_x + ny
                intersections_th = mth*intersections_x + nth
                intersections_1 = np.concatenate((intersections_x.reshape(-1,1), intersections_y.reshape(-1,1), intersections_th.reshape(-1,1)), axis=1)

                intersections_col = np.arange( col_ymax + 1, col_ymin + 1, 1)
                intersections_y = self.TraversabilityMap_ymax - intersections_col*self.map_resolution
                intersections_x = (intersections_y - ny)/my
                intersections_th = mth*intersections_x + nth
                intersections_2 = np.concatenate((intersections_x.reshape(-1,1), intersections_y.reshape(-1,1), intersections_th.reshape(-1,1)), axis=1)

                intersections = np.concatenate((intersections_1, intersections_2), axis=0)
                intersections = np.concatenate((intersections, np.array([[x1,y1,c1], [x2,y2,c2]])), axis=0)
                intersections = intersections[intersections[:,0].argsort()]

            elif mode == 'SamplingInEdge':
                
                if sampling_distance is not None:
                    num_interpoints = int(math.hypot(x2-x1, y2-y1)/sampling_distance) + 1
                elif num_interpoints is not None:
                    pass
                else:
                    raise ValueError("Either sampling_distance or num_interpoints should be provided")

                x_interpoints = np.linspace(x1, x2, num_interpoints)
                y_interpoints = np.linspace(y1, y2, num_interpoints)
                c_interpoints = np.linspace(c1, c2, num_interpoints)
                intersections = np.concatenate((x_interpoints.reshape(-1,1), y_interpoints.reshape(-1,1), c_interpoints.reshape(-1,1)), axis=1)

            else:
                raise ValueError("mode should be either 'ByCell' or 'SamplingInEdge'")


            trav_cost_sum = 0
            for i in range(intersections.shape[0]):

                x_i1 = intersections[i,0]
                y_i1 = intersections[i,1]
                c_i1 = intersections[i,2]
                # x_i2 = intersections[i+1,0]
                # y_i2 = intersections[i+1,1]

                row = int( (self.TraversabilityMap_xmax - x_i1)/self.map_resolution )
                col = int( (self.TraversabilityMap_ymax - y_i1)/self.map_resolution )
                c_i1 = wrap_to_pi(c_i1)
                theta_layer = int(np.round((c_i1-self.TraversabilityMap_theta_min)/self.TraversabilityMap_theta_resolution))
                if theta_layer == self.TraversabilityMap_size_layers:
                    theta_layer = 0 # pi = -pi


                # segment_length = math.hypot(x_i2-x_i1, y_i2-y_i1)
                # v = self.TraversabilityMap[row, col, theta_layer, 0]
                # w = self.TraversabilityMap[row, col, theta_layer, 1]

                trav_cost_sum += (1/self.TraversabilityMap[row, col, theta_layer, 2])
            
            

            cost = ( 1 + cost_weight * (trav_cost_sum/(intersections.shape[0]))**cost_power )* vertex_distance
                   




        return cost












class LearnedInSMap(ScoreBasedMap):

    def __init__(self, env_xmin, env_xmax, env_ymin, env_ymax, goal_x, goal_y, which_layer, preest_update_resolution, instab_limit, load_NN = True):
        super().__init__(env_xmin, env_xmax, env_ymin, env_ymax, goal_x, goal_y, which_layer, preest_update_resolution, instab_limit, load_NN)
        
        self.cost_at_best = 2.14 # at v = 0.5 on flat terrain, the prediction was mean = 2, std = 0.07 -> VaR instab = 2.14


    def get_travmap(self, estimating_xypts_set, xmin, xmax, ymin, ymax):
        
        local_update_pts_list = np.empty((0, 2))

        if estimating_xypts_set is None:

            xmin = self.TravUpdate_XYResolution * np.round(xmin/self.TravUpdate_XYResolution)
            xmax = self.TravUpdate_XYResolution * np.round(xmax/self.TravUpdate_XYResolution)
            ymin = self.TravUpdate_XYResolution * np.round(ymin/self.TravUpdate_XYResolution)
            ymax = self.TravUpdate_XYResolution * np.round(ymax/self.TravUpdate_XYResolution)
            for estimating_gridpt in np.array([ [x,y] for x in np.arange(xmin, xmax + self.TravUpdate_XYResolution, self.TravUpdate_XYResolution)\
                                                        for y in np.arange(ymin, ymax + self.TravUpdate_XYResolution, self.TravUpdate_XYResolution)]):
                
                estimating_pt_global_row = int( (self.ElevationMap_xmax - estimating_gridpt[0])/self.map_resolution )
                estimating_pt_global_col = int( (self.ElevationMap_ymax - estimating_gridpt[1])/self.map_resolution )

                if estimating_pt_global_row < 0 or estimating_pt_global_row >= self.ElevationMap_size_rows\
                    or estimating_pt_global_col < 0 or estimating_pt_global_col >= self.ElevationMap_size_cols:
                    # This pt is out of the global map. no need to preestimate
                    continue
                elif np.any(self.TraversabilityMap[estimating_pt_global_row, estimating_pt_global_col, :, 2] != self.InitialGuess_auxiliary_score):
                    # This pt is already preestimated for lateral drift. no need to preestimate
                    continue
                elif np.isnan(self.ElevationMap[estimating_pt_global_row, estimating_pt_global_col]):
                    # This pt is nan. no need to preestimate
                    continue
                elif self.TraversabilityMap[estimating_pt_global_row, estimating_pt_global_col, 0, 0] == self.obstacle_cmd_v:
                    # This pt is already preestimated as obstacle. no need to preestimate
                    continue
                elif abs(self.ElevationMap_xmax - estimating_gridpt[0]) < 0.5*self.local_patch_size or abs(estimating_gridpt[0] - self.TraversabilityMap_xmin) < 0.5*self.local_patch_size\
                    or abs(self.ElevationMap_ymax - estimating_gridpt[1]) < 0.5*self.local_patch_size or abs(estimating_gridpt[1] - self.TraversabilityMap_ymin) < 0.5*self.local_patch_size:
                    # This pt is near the edge of the global map. no need to preestimate
                    continue
                # elif np.any(np.all(local_update_pts_list == estimating_gridpt, axis=1)):
                #     # This pt is already in the list
                #     continue
                else:
                    local_update_pts_list = np.append(local_update_pts_list, np.array([estimating_gridpt]), axis=0)
                    

            print("  Estimating", local_update_pts_list.shape[0], " points in total...")
            # print(local_update_pts_list)
            estimating_xycpts_set = np.array([ [xy[0], xy[1], c] for xy in local_update_pts_list\
                                                for c in np.arange(self.TraversabilityMap_theta_min, self.TraversabilityMap_theta_max + self.TraversabilityMap_theta_resolution, self.TraversabilityMap_theta_resolution) ])


        else:
            # estimating_xypts_set is ndarray of [x,y]. I want to add one more dimension for theta
            estimating_xycpts_set = np.array([ [xy[0], xy[1], c] for xy in estimating_xypts_set\
                                               for c in np.arange(self.TraversabilityMap_theta_min, self.TraversabilityMap_theta_max + self.TraversabilityMap_theta_resolution, self.TraversabilityMap_theta_resolution) ])

            print(" Estimating", estimating_xypts_set.shape[0], " points in total...")








        if self.is_RobocentricMap_received and self.is_ElevationMap_built and estimating_xycpts_set.shape[0] != 0:
            
            counter = 0
            total_pts = estimating_xycpts_set.shape[0]

            xyc_patchindex = np.concatenate((estimating_xycpts_set, np.zeros((estimating_xycpts_set.shape[0],1), dtype=int)), axis=1)
            patchindex_counter = 0
            patch_tensor = np.empty((0, 2*np.round(0.5*self.local_patch_size/self.map_resolution).astype(int), 2*np.round(0.5*self.local_patch_size/self.map_resolution).astype(int)))
            safe_cmdv_tensor = np.empty((0, 2))

            for xyc in estimating_xycpts_set:
                
                x = xyc[0]
                y = xyc[1]
                theta = xyc[2]

                patch, ismapnan = self.get_patch_in_elevmap(x, y, theta, self.local_patch_size)

                
                

                if ismapnan:
                    xyc_patchindex[patchindex_counter, 3] = -1 # -1 means NaN 
                    patchindex_counter += 1
                    continue
                

                else:
                    # absolute height offset compensation
                    patch_at_center = np.percentile(patch, 5)
                    patch = patch - patch_at_center
                    patch_tensor = np.concatenate((patch_tensor, patch[np.newaxis, :, :]), axis=0)

                    row_global = int( (self.ElevationMap_xmax - x)/self.map_resolution )
                    col_global = int( (self.ElevationMap_ymax - y)/self.map_resolution )
                    theta_layer = int(np.round((theta-self.TraversabilityMap_theta_min)/self.TraversabilityMap_theta_resolution))
                    if theta_layer == self.TraversabilityMap_size_layers:
                        theta_layer = 0
                    safe_cmd_v = self.maximum_cmd_v
                    safe_cmdv_tensor = np.concatenate((safe_cmdv_tensor, np.array([[safe_cmd_v, 0]])), axis=0)
                    xyc_patchindex[patchindex_counter, 3] = patch_tensor.shape[0] - 1
                    patchindex_counter += 1



            patch_tensor_extended = np.concatenate((patch_tensor, np.zeros((64-patch_tensor.shape[0]%64, patch_tensor.shape[1], patch_tensor.shape[2]))), axis=0)
            safe_cmdv_tensor_extended = np.concatenate((safe_cmdv_tensor, np.zeros((64-safe_cmdv_tensor.shape[0]%64, safe_cmdv_tensor.shape[1]))), axis=0)
            [instability_mean_tensor, instability_std_tensor] = self._nn_inference(None, patch_tensor_extended, safe_cmdv_tensor_extended, model=self.elevonly_vw2instab_model, model_type=self.model_type, device=self.device)
            
            

            patchindex_counter = 0
            for xyc in estimating_xycpts_set:

                x = xyc[0]
                y = xyc[1]
                theta = xyc[2]

                instability_mean = 0
                instability_std = 0
                traversability_score = 0
                cost = 0

                patchindex = xyc_patchindex[patchindex_counter, 3].astype(int)
                patchindex_counter += 1




                if patchindex == -1: # NaN
                    instability_mean = self.InitialGuess_auxiliary_score
                    instability_std  = self.InitialGuess_auxiliary_score_std
                    traversability_score = self.InitialGuess_auxiliary_score
                elif patchindex == -2: # OBSTACLE
                    instability_mean = self.InitialGuess_auxiliary_score
                    instability_std  = self.InitialGuess_auxiliary_score_std
                    traversability_score = self.InitialGuess_auxiliary_score
                else:
                    instability_mean = instability_mean_tensor[patchindex]
                    instability_std = instability_std_tensor[patchindex]
                    cost = instability_mean + instability_std_multiplier*instability_std
                    traversability_score = self.convert_to_score(cost)


                # Assign cmd_v and cmd_w to the 3D map. Square with size of 'self.TravUpdate_XYResolution' centered at x and y at each pt has the same cmd_v and cmd_w
                square_BL_xyc = np.array([x - self.TravUpdate_XYResolution/2, y - self.TravUpdate_XYResolution/2])
                square_TR_xyc = np.array([x + self.TravUpdate_XYResolution/2, y + self.TravUpdate_XYResolution/2])

                square_BL_entry = np.array([ round( (self.TraversabilityMap_xmax - square_BL_xyc[0])/self.map_resolution ), round( (self.TraversabilityMap_ymax - square_BL_xyc[1])/self.map_resolution ) ])
                square_TR_entry = np.array([ round( (self.TraversabilityMap_xmax - square_TR_xyc[0])/self.map_resolution ), round( (self.TraversabilityMap_ymax - square_TR_xyc[1])/self.map_resolution ) ])
                
                theta = wrap_to_pi(theta)
                theta_layer = int(np.round((theta-self.TraversabilityMap_theta_min)/self.TraversabilityMap_theta_resolution))
                if theta_layer == self.TraversabilityMap_size_layers:
                    theta_layer = 0 # pi = -pi

                for row in range(square_TR_entry[0], square_BL_entry[0]):
                    for col in range(square_TR_entry[1], square_BL_entry[1]):
                        
                        if row < 0 or row >= self.TraversabilityMap_size_rows or col < 0 or col >= self.TraversabilityMap_size_cols:
                            continue
                        else:

                            if self.local_cmd_option == 'score_travcmd':
                                self.TraversabilityMap[row, col, theta_layer, 0] = self.maximum_cmd_v * traversability_score
                                self.TraversabilityMap[row, col, theta_layer, 1] = self.maximum_cmd_w * traversability_score
                            else:
                                self.TraversabilityMap[row, col, theta_layer, 0] = self.InitialGuess_cmd_v + 0.01 # To mark that this point is already estimated
                                self.TraversabilityMap[row, col, theta_layer, 1] = self.InitialGuess_cmd_w + 0.001

                            self.TraversabilityMap[row, col, theta_layer, 2] = traversability_score
                            self.TravMap_MetaInfo[row, col, theta_layer] = self.TravmapInfo_IntEnum.TRAV_Estimated.value





            for xyc in estimating_xycpts_set:
                check_bound_x = int((self.TraversabilityMap_xmax - xyc[0])/self.map_resolution)
                check_bound_y = int((self.TraversabilityMap_ymax - xyc[1])/self.map_resolution)
                # 2 nested for loops to thake the corner of the reference points
                for Row_TLCorner in np.arange(xyc[0], xyc[0] + 2 * self.TravUpdate_XYResolution, self.TravUpdate_XYResolution):
                    for Col_TLCorner in np.arange(xyc[1], xyc[1] + 2 * self.TravUpdate_XYResolution, self.TravUpdate_XYResolution):
                        #Converting the size into Grid Scale
                        x1 = int((self.TraversabilityMap_xmax - Row_TLCorner)/self.map_resolution)
                        y1 = int((self.TraversabilityMap_ymax - Col_TLCorner)/self.map_resolution)
                        update_size = int(self.TravUpdate_XYResolution/self.map_resolution)
                        #Coordinate Points for x2 and y2 points
                        x2 = x1 + update_size
                        y2 = y1 + update_size
                        #Checking Boundary Condition
                        if x1 < 0 or x2 > self.TraversabilityMap_size_rows or y1 < 0 or y2 > self.TraversabilityMap_size_cols:
                            continue
                        else:
                            theta = xyc[2]
                            theta = wrap_to_pi(theta)
                            theta_layer = int(np.round((theta-self.TraversabilityMap_theta_min)/self.TraversabilityMap_theta_resolution))
                            #Code not implemented which will be used for checking the duplicate quadrants
                            if theta_layer == self.TraversabilityMap_size_layers:
                                theta_layer = 0 # pi = -pi
                            if x2 > check_bound_x:
                                boundcheck_x = check_bound_x + 1
                            else:
                                boundcheck_x = check_bound_x - 1
                            if y2 > check_bound_y:
                                boundcheck_y = check_bound_y + 1
                            else:
                                boundcheck_y = check_bound_y - 1
                                
                            # Each cmd_v and cmd_w values of each quadrants corner
                            for map_layer in range(3):
                                try:
                                    x2y2_pred = self.TraversabilityMap[x2,y2,theta_layer,map_layer]
                                    x1y2_pred = self.TraversabilityMap[x1,y2,theta_layer,map_layer]
                                    x2y1_pred = self.TraversabilityMap[x2,y1,theta_layer,map_layer]
                                    x1y1_pred = self.TraversabilityMap[x1,y1,theta_layer,map_layer]
                                except:
                                    print("Error: x2,y2,x1,y1,theta_layer,map_layer", x2,y2,x1,y1,theta_layer,map_layer)
                                    continue

                                edge_pred = np.array([x1y1_pred, x2y1_pred, x1y2_pred, x2y2_pred])
                                #Looping through the quadrants
                                for row3 in range(x1,x2+1):
                                    for col3 in range(y1,y2+1):
                                        if x1 == x2 or y1 == y2:
                                            continue                                                                  

                                        else:
                                            self.TraversabilityMap[row3, col3, theta_layer, map_layer] = ((col3 - y2) / (y1 - y2)) * ((row3 - x2)/(x1-x2)*x1y1_pred + (row3 - x1)/(x2-x1)*x2y1_pred) +  ((col3 - y1) / (y2 -y1)) * ((row3 - x2)/(x1-x2)*x1y2_pred + (row3 - x1)/(x2-x1)*x2y2_pred)
                                            
                                            






class IHMCMap(ScoreBasedMap):

    def __init__(self, env_xmin, env_xmax, env_ymin, env_ymax, goal_x, goal_y, which_layer, preest_update_resolution, instab_limit):
        super().__init__(env_xmin, env_xmax, env_ymin, env_ymax, goal_x, goal_y, which_layer, preest_update_resolution, instab_limit)

        # IHMC Costs start strictly from 0
        # Padding to prevent deviding by 0
        # Later for planning, the cost become 1 + (1/cost_at_best)*cost
        self.cost_at_best = 0.01


        self.IHMC_parameters = {
            "Traversibility height window deadband": 0.15,
            "Min normal angle to penalize for traversibility": 45,
            "Max normal angle to penalize for traversibility": 75,
            "Traversibility incline weight": 0,
            "Traversibility height window width": 0.2,
            "Half stance width": 0.175,
            "Traversibility step weight": 2,
            "Traversibility stance weight": 4,

            "Max penalized roll angle": 7,
            "Roll cost deadband": 1.5,
            "Roll cost weight": 1,

            "step_length": 0.2,

            "Visualization": True,

        }



    def get_travmap(self, estimating_xypts_set, xmin, xmax, ymin, ymax):
        
        local_update_pts_list = np.empty((0, 2))

        if estimating_xypts_set is None:

            xmin = self.TravUpdate_XYResolution * np.round(xmin/self.TravUpdate_XYResolution)
            xmax = self.TravUpdate_XYResolution * np.round(xmax/self.TravUpdate_XYResolution)
            ymin = self.TravUpdate_XYResolution * np.round(ymin/self.TravUpdate_XYResolution)
            ymax = self.TravUpdate_XYResolution * np.round(ymax/self.TravUpdate_XYResolution)
            for estimating_gridpt in np.array([ [x,y] for x in np.arange(xmin, xmax + self.TravUpdate_XYResolution, self.TravUpdate_XYResolution)\
                                                        for y in np.arange(ymin, ymax + self.TravUpdate_XYResolution, self.TravUpdate_XYResolution)]):
                
                estimating_pt_global_row = int( (self.ElevationMap_xmax - estimating_gridpt[0])/self.map_resolution )
                estimating_pt_global_col = int( (self.ElevationMap_ymax - estimating_gridpt[1])/self.map_resolution )

                if estimating_pt_global_row < 0 or estimating_pt_global_row >= self.ElevationMap_size_rows\
                    or estimating_pt_global_col < 0 or estimating_pt_global_col >= self.ElevationMap_size_cols:
                    # This pt is out of the global map. no need to preestimate
                    continue
                elif np.any(self.TraversabilityMap[estimating_pt_global_row, estimating_pt_global_col, :, 2] != self.InitialGuess_auxiliary_score):
                    # This pt is already preestimated . no need to preestimate
                    continue
                elif np.isnan(self.ElevationMap[estimating_pt_global_row, estimating_pt_global_col]):
                    # This pt is nan. no need to preestimate
                    continue
                elif self.TraversabilityMap[estimating_pt_global_row, estimating_pt_global_col, 0, 0] == self.obstacle_cmd_v:
                    # This pt is already preestimated as obstacle. no need to preestimate
                    continue
                elif abs(self.ElevationMap_xmax - estimating_gridpt[0]) < 0.5*self.local_patch_size or abs(estimating_gridpt[0] - self.TraversabilityMap_xmin) < 0.5*self.local_patch_size\
                    or abs(self.ElevationMap_ymax - estimating_gridpt[1]) < 0.5*self.local_patch_size or abs(estimating_gridpt[1] - self.TraversabilityMap_ymin) < 0.5*self.local_patch_size:
                    # This pt is near the edge of the global map. no need to preestimate
                    continue
                # elif np.any(np.all(local_update_pts_list == estimating_gridpt, axis=1)):
                #     # This pt is already in the list
                #     continue
                else:
                    local_update_pts_list = np.append(local_update_pts_list, np.array([estimating_gridpt]), axis=0)
                    

            print("  Estimating", local_update_pts_list.shape[0], " points in total...")
            # print(local_update_pts_list)
            estimating_xycpts_set = np.array([ [xy[0], xy[1], c] for xy in local_update_pts_list\
                                                for c in np.arange(self.TraversabilityMap_theta_min, self.TraversabilityMap_theta_max + self.TraversabilityMap_theta_resolution, self.TraversabilityMap_theta_resolution) ])


        else:
            # estimating_xypts_set is ndarray of [x,y]. I want to add one more dimension for theta
            estimating_xycpts_set = np.array([ [xy[0], xy[1], c] for xy in estimating_xypts_set\
                                               for c in np.arange(self.TraversabilityMap_theta_min, self.TraversabilityMap_theta_max + self.TraversabilityMap_theta_resolution, self.TraversabilityMap_theta_resolution) ])

            print(" Estimating", estimating_xypts_set.shape[0], " points in total...")









        if self.is_RobocentricMap_received and self.is_ElevationMap_built and estimating_xycpts_set.shape[0] != 0:
            
            for xyc in estimating_xycpts_set:

                x = xyc[0]
                y = xyc[1]
                theta = xyc[2]

                theta = wrap_to_pi(theta)
                theta_layer = int(np.round((theta-self.TraversabilityMap_theta_min)/self.TraversabilityMap_theta_resolution))
                if theta_layer == self.TraversabilityMap_size_layers:
                    theta_layer = 0 # pi = -pi




                cost = 0
                invalid_flag = False
                x2 = x
                y2 = y
                x1 = x2 - self.IHMC_parameters["step_length"] * np.cos(theta)
                y1 = y2 - self.IHMC_parameters["step_length"] * np.sin(theta)

                if not self.is_RobocentricMap_received or not self.is_ElevationMap_built:
                    continue
                    


                row_x2 =  int( (self.TraversabilityMap_xmax - x2)/self.map_resolution )
                col_y2 =  int( (self.TraversabilityMap_ymax - y2)/self.map_resolution )
                if row_x2 < 0 or row_x2 >= self.TraversabilityMap_size_rows or col_y2 < 0 or col_y2 >= self.TraversabilityMap_size_cols:
                    invalid_flag = True
                    
                else:                   
                    z2 = self.ElevationMap[row_x2, col_y2]
                theta2 = np.arctan2(y2-y1, x2-x1)
                

                row_x1 =  int( (self.TraversabilityMap_xmax - x1)/self.map_resolution )
                col_y1 =  int( (self.TraversabilityMap_ymax - y1)/self.map_resolution )
                if row_x1 < 0 or row_x1 >= self.TraversabilityMap_size_rows or col_y1 < 0 or col_y1 >= self.TraversabilityMap_size_cols:
                    invalid_flag = True
                else:                   
                    z1 = self.ElevationMap[row_x1, col_y1]

                distance = math.hypot(x2-x1, y2-y1)




                if not invalid_flag:
                    # line 363-393 of ihmc-footstep-planning/src/main/java/us/ihmc/footstepPlanning/bodyPath/AStarBodyPathPlanner.java. use RANSAC Trav
                    parentheight = z1
                    nodeheight = z2
                    minheight = min(parentheight, nodeheight) - self.IHMC_parameters["Traversibility height window width"]
                    maxheight = max(parentheight, nodeheight) + self.IHMC_parameters["Traversibility height window width"]
                    avgheight = (parentheight + nodeheight)/2
                    windowwidth = (maxheight - minheight)/2

                    jl_center = np.array([x2, y2, 0]) + self.IHMC_parameters["Half stance width"]*tf3.euler.euler2mat(0, 0, math.pi/2, 'sxyz').transpose() @ np.array([np.cos(theta2), np.sin(theta2), 0])
                    patch_jl, ismapnan = self.get_patch_in_elevmap(jl_center[0], jl_center[1], theta2, 0.22) # size of the robot feet
                    if ismapnan:
                        invalid_flag = True
                    else:                   
                        t_jl = self.computetraversability(patch_jl, ismapnan, minheight, maxheight, avgheight, windowwidth, self.map_resolution)

                    jr_center = np.array([x2, y2, 0]) + self.IHMC_parameters["Half stance width"]*tf3.euler.euler2mat(0, 0, -math.pi/2, 'sxyz').transpose() @ np.array([np.cos(theta2), np.sin(theta2), 0])
                    patch_jr, ismapnan = self.get_patch_in_elevmap(jr_center[0], jr_center[1], theta2, 0.22) # size of the robot feet
                    if ismapnan:
                        invalid_flag = True
                    else:                   
                        t_jr = self.computetraversability(patch_jr, ismapnan, minheight, maxheight, avgheight, windowwidth, self.map_resolution)

                    il_center = np.array([x1, y1, 0]) + self.IHMC_parameters["Half stance width"]*tf3.euler.euler2mat(0, 0, math.pi, 'sxyz').transpose() @ np.array([np.cos(theta2), np.sin(theta2), 0])
                    patch_il, ismapnan = self.get_patch_in_elevmap(il_center[0], il_center[1], theta2, 0.22) # size of the robot feet
                    if ismapnan:
                        invalid_flag = True
                    else:                   
                        t_il = self.computetraversability(patch_il, ismapnan, minheight, maxheight, avgheight, windowwidth, self.map_resolution)

                    ir_center = np.array([x1, y1, 0]) + self.IHMC_parameters["Half stance width"]*tf3.euler.euler2mat(0, 0, -math.pi, 'sxyz').transpose() @ np.array([np.cos(theta2), np.sin(theta2), 0])
                    patch_ir, ismapnan = self.get_patch_in_elevmap(ir_center[0], ir_center[1], theta2, 0.22) # size of the robot feet
                    if ismapnan:
                        invalid_flag = True
                    else:                   
                        t_ir = self.computetraversability(patch_ir, ismapnan, minheight, maxheight, avgheight, windowwidth, self.map_resolution)

                    if not invalid_flag:
                        stepTraversability = max(t_jl, t_jr)
                        stanceTraversability = max(math.sqrt(t_jl*t_ir), math.sqrt(t_jr*t_il))
                        cost += self.IHMC_parameters["Traversibility step weight"] * (1-stepTraversability) + self.IHMC_parameters["Traversibility stance weight"] * (1-stanceTraversability)



                # line 456-485 of ihmc-footstep-planning/src/main/java/us/ihmc/footstepPlanning/bodyPath/AStarBodyPathPlanner.java
                # get surface patch's normal vector
                if not invalid_flag:
                    incline_theta_ij = np.arctan2(z2 - z1, distance)
                    patch, ismapnan = self.get_patch_in_elevmap(x2, y2, 0, self.IHMC_parameters["Half stance width"])
                    if ismapnan:
                        invalid_flag = True
                    else:

                        # patch = least_squares_fitting_plane(patch)
                        dx, dy = np.gradient(patch)
                        dx, dy = -dx, -dy
                        normals = np.dstack((-dx/self.map_resolution, -dy/self.map_resolution, np.ones_like(patch)))
                        normals = normals / np.linalg.norm(normals, axis=2, keepdims=True)
                        normals = np.median(normals, axis=(0,1))

                        
                        effectiveroll = np.arcsin( abs(np.dot(normals, np.array([-np.sin(theta2), np.cos(theta2), 0]))) )
                        maxangle = np.deg2rad(self.IHMC_parameters["Max penalized roll angle"] - self.IHMC_parameters["Roll cost deadband"])
                        inclinescale = max(0, min(abs(incline_theta_ij) / maxangle, 1))
                        rollangleDeadbanded = max(0, abs(effectiveroll) - np.deg2rad(self.IHMC_parameters["Roll cost deadband"]))
                        cost += self.IHMC_parameters["Roll cost weight"] * inclinescale * rollangleDeadbanded
                    


                # line 405-414 of ihmc-footstep-planning/src/main/java/us/ihmc/footstepPlanning/bodyPath/AStarBodyPathPlanner.java
                inclinecost = 0 # the cost weight is set 0 by default in the code
                cost += inclinecost

                traversability_score = self.InitialGuess_auxiliary_score if invalid_flag == True else self.convert_to_score(cost)






        

                # Assign cmd_v and cmd_w to the 3D map. Square with size of 'self.TravUpdate_XYResolution' centered at x and y at each pt has the same cmd_v and cmd_w
                square_BL_xyc = np.array([x - self.TravUpdate_XYResolution/2, y - self.TravUpdate_XYResolution/2])
                square_TR_xyc = np.array([x + self.TravUpdate_XYResolution/2, y + self.TravUpdate_XYResolution/2])

                square_BL_entry = np.array([ round( (self.TraversabilityMap_xmax - square_BL_xyc[0])/self.map_resolution ), round( (self.TraversabilityMap_ymax - square_BL_xyc[1])/self.map_resolution ) ])
                square_TR_entry = np.array([ round( (self.TraversabilityMap_xmax - square_TR_xyc[0])/self.map_resolution ), round( (self.TraversabilityMap_ymax - square_TR_xyc[1])/self.map_resolution ) ])
                

                for row in range(square_TR_entry[0], square_BL_entry[0]):
                    for col in range(square_TR_entry[1], square_BL_entry[1]):
                        
                        if row < 0 or row >= self.TraversabilityMap_size_rows or col < 0 or col >= self.TraversabilityMap_size_cols:
                            continue
                        else:

                            if self.local_cmd_option == 'score_travcmd':
                                self.TraversabilityMap[row, col, theta_layer, 0] = self.maximum_cmd_v * traversability_score
                                self.TraversabilityMap[row, col, theta_layer, 1] = self.maximum_cmd_w * traversability_score
                            else:
                                self.TraversabilityMap[row, col, theta_layer, 0] = self.InitialGuess_cmd_v + 0.01
                                self.TraversabilityMap[row, col, theta_layer, 1] = self.InitialGuess_cmd_w + 0.001
                            self.TraversabilityMap[row, col, theta_layer, 2] = traversability_score
                            self.TravMap_MetaInfo[row, col, theta_layer] = self.TravmapInfo_IntEnum.TRAV_Estimated.value
                            





            for xyc in estimating_xycpts_set:
                check_bound_x = int((self.TraversabilityMap_xmax - xyc[0])/self.map_resolution)
                check_bound_y = int((self.TraversabilityMap_ymax - xyc[1])/self.map_resolution)
                # 2 nested for loops to thake the corner of the reference points
                for Row_TLCorner in np.arange(xyc[0], xyc[0] + 2 * self.TravUpdate_XYResolution, self.TravUpdate_XYResolution):
                    for Col_TLCorner in np.arange(xyc[1], xyc[1] + 2 * self.TravUpdate_XYResolution, self.TravUpdate_XYResolution):
                        #Converting the size into Grid Scale
                        x1 = int((self.TraversabilityMap_xmax - Row_TLCorner)/self.map_resolution)
                        y1 = int((self.TraversabilityMap_ymax - Col_TLCorner)/self.map_resolution)
                        update_size = int(self.TravUpdate_XYResolution/self.map_resolution)
                        #Coordinate Points for x2 and y2 points
                        x2 = x1 + update_size
                        y2 = y1 + update_size
                        #Checking Boundary Condition
                        if x1 < 0 or x2 > self.TraversabilityMap_size_rows or y1 < 0 or y2 > self.TraversabilityMap_size_cols:
                            continue
                        else:
                            theta = xyc[2]
                            theta = wrap_to_pi(theta)
                            theta_layer = int(np.round((theta-self.TraversabilityMap_theta_min)/self.TraversabilityMap_theta_resolution))
                            if theta_layer == self.TraversabilityMap_size_layers:
                                theta_layer = 0 # pi = -pi

                            if x2 > check_bound_x:
                                boundcheck_x = check_bound_x + 1
                            else:
                                boundcheck_x = check_bound_x - 1
                            if y2 > check_bound_y:
                                boundcheck_y = check_bound_y + 1
                            else:
                                boundcheck_y = check_bound_y - 1
                            # Each cmd_v and cmd_w values of each quadrants corner
                            for map_layer in range(3):
                                x2y2_pred = self.TraversabilityMap[x2,y2,theta_layer,map_layer]
                                x1y2_pred = self.TraversabilityMap[x1,y2,theta_layer,map_layer]
                                x2y1_pred = self.TraversabilityMap[x2,y1,theta_layer,map_layer]
                                x1y1_pred = self.TraversabilityMap[x1,y1,theta_layer,map_layer]

                                edge_pred = np.array([x1y1_pred, x2y1_pred, x1y2_pred, x2y2_pred])
                                #Looping through the quadrants
                                for row3 in range(x1,x2+1):
                                    for col3 in range(y1,y2+1):
                                        if x1 == x2 or y1 == y2:
                                            continue                                                                  

                                        else:
                                            self.TraversabilityMap[row3, col3, theta_layer, map_layer] = ((col3 - y2) / (y1 - y2)) * ((row3 - x2)/(x1-x2)*x1y1_pred + (row3 - x1)/(x2-x1)*x2y1_pred) +  ((col3 - y1) / (y2 -y1)) * ((row3 - x2)/(x1-x2)*x1y2_pred + (row3 - x1)/(x2-x1)*x2y2_pred)
                                            
                                            

    def computetraversability(self, patch, ismapnan, minheight, maxheight, avgheight, windowwidth, cellresolution):
            
        t_ = 0
        if ismapnan:
            t_ = 0
        else:
            patch = patch[:, int(0.25*patch.shape[1]):int(0.75*patch.shape[1])] # middle half of the patch since foot width is roughly half of the length
            for row in range(patch.shape[0]):
                for col in range(patch.shape[1]):
                    height = patch[row, col]

                    if height > minheight and height < maxheight:

                        deltaheight = max(0, abs(avgheight - height) - self.IHMC_parameters["Traversibility height window deadband"])
                        cellpercentage = 1 - deltaheight/windowwidth
                        nonGroundDiscount = 1 # not explained in detail in the paper

                        minNormalToPenalize = np.rad2deg(self.IHMC_parameters["Min normal angle to penalize for traversibility"])
                        maxNormalToPenalize = np.rad2deg(self.IHMC_parameters["Max normal angle to penalize for traversibility"])
                        dx, dy = np.gradient(patch)
                        dx, dy = -dx, -dy
                        normal = np.array([-dx[row, col]/cellresolution, -dy[row, col]/cellresolution, 1])
                        normal = normal / np.linalg.norm(normal)
                        incline = max(0, np.arccos(normal[2]) - minNormalToPenalize)
                        inclinealpha = max(0, min((maxNormalToPenalize - incline) / (maxNormalToPenalize - minNormalToPenalize), 1) )

                        
                        t_ += nonGroundDiscount*(1 - self.IHMC_parameters["Traversibility incline weight"])*cellpercentage + self.IHMC_parameters["Traversibility incline weight"]*inclinealpha

            t_ = t_ / (patch.shape[0]*patch.shape[1])
        
        return t_
















class QuadrupedMap(ScoreBasedMap):

    def __init__(self, env_xmin, env_xmax, env_ymin, env_ymax, goal_x, goal_y, which_layer, preest_update_resolution, instab_limit):
        super().__init__(env_xmin, env_xmax, env_ymin, env_ymax, goal_x, goal_y, which_layer, preest_update_resolution, instab_limit)


    def get_travmap(self, estimating_xypts_set, xmin, xmax, ymin, ymax):

        if not self.is_ElevationMap_built:
            self.Initilize_map()

        if self.is_RobocentricMap_received and self.is_ElevationMap_built:


            print(self.layer_data.keys())

            # Update global map
            local_traversability = self.layer_data['erosion']
            nan_map = self.layer_data[self.which_layer[1]]

            for row in range(local_traversability.shape[0]):
                for col in range(local_traversability.shape[1]):

                    if not np.isnan(nan_map[row, col]):

                        # for the current row,col in the local map, get the global x and y in the world frame
                        global_x = ( int( self.RobocentricMap_Rows/2) - row ) * self.map_resolution + self.RobocentricMap_CenterPosOffset_Worldfr[0]
                        global_y = ( int( self.RobocentricMap_Cols/2) - col ) * self.map_resolution + self.RobocentricMap_CenterPosOffset_Worldfr[1]

                        # corresponding row and col index in the global map
                        row_global = int( (self.ElevationMap_xmax - global_x)/self.map_resolution )
                        col_global = int( (self.ElevationMap_ymax - global_y)/self.map_resolution )
                        if row_global < 0 or row_global >= self.ElevationMap_size_rows or col_global < 0 or col_global >= self.ElevationMap_size_cols:
                            continue
                            # TODO: expand the global map by resizing it if out-of-range point is detected

                        if self.local_cmd_option == 'score_travcmd':
                            self.TraversabilityMap[row_global, col_global, :, 0] = self.maximum_cmd_v * local_traversability[row, col]
                            self.TraversabilityMap[row_global, col_global, :, 1] = self.maximum_cmd_w * local_traversability[row, col]
                        self.TraversabilityMap[row_global, col_global, :, 2] = local_traversability[row, col]
                        self.TravMap_MetaInfo[row, col, :] = self.TravmapInfo_IntEnum.TRAV_Estimated.value



