"""
Environment for rrt_2D
@author: huiming zhou
"""


class Env:
    def __init__(self):
        self.x_range = (0, 5)
        self.y_range = (0, 5)
        self.obs_boundary = self.obs_boundary()
        self.obs_circle = self.obs_circle()
        self.obs_rectangle = self.obs_rectangle()
        self.obstacle = []


    #Define obstable as follow: [x_0 y_0, +width, +height]
    # :                +------------------+
    # :                |                  |
    # :              height               |
    # :                |                  |
    # :               (xy)---- width -----+
    
    @staticmethod
    def obs_boundary():
        obs_boundary = [
            #  [0, 0, 0.1, 4],
            # [0, 4, 4, 0.1],
            # [0.1, 0, 4, 0.1],
            # [4, 0.1, 0.1, 4]
            # [0, 0, 1, 30],
            # [0, 30, 50, 1],
            # [1, 0, 50, 1],
            # [50, 1, 1, 30]
        ]
        return obs_boundary

    @staticmethod
    def obs_rectangle():
        obs_rectangle = [
            # [14, 12, 8, 2],
            # [18, 22, 8, 3],
            # # [26, 7, 2, 12],
            # [32, 14, 10, 2]
        ]
        return obs_rectangle

    @staticmethod
    def obs_circle():
        obs_cir = [
            # [2,2,0.1],
            # [1,2,0.3],
            # [2.5,0.3,0.2],
            # [1.5,1,0.15],
            # [2.5,1.5,0.15]
            
            # [7, 12, 3],
            # [46, 20, 2],
            # [15, 5, 2],
            # [37, 7, 3],
            # [37, 23, 3]
        ]

        return obs_cir
    