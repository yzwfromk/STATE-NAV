"""
utils for collision check
@author: huiming zhou
"""

import math
import numpy as np
import os
import sys

# sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
#                 "/../../Sampling_based_Planning/")

import env
from rrt import Node


class Utils:
    def __init__(self):
        self.env = env.Env()

        self.delta = 0.01#0.5
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary
        self.obstacle = []

    def update_obs(self, obs_cir, obs_bound, obs_rec):
        self.obs_circle = obs_cir
        self.obs_boundary = obs_bound
        self.obs_rectangle = obs_rec

    def get_obs_vertex(self):
        delta = self.delta
        obs_list = []

        for (ox, oy, w, h) in self.obs_rectangle:
            vertex_list = [[ox - delta, oy - delta],
                           [ox + w + delta, oy - delta],
                           [ox + w + delta, oy + h + delta],
                           [ox - delta, oy + h + delta]]
            obs_list.append(vertex_list)
            
        delta = 0.01
        for (ox, oy, w, h) in self.obs_boundary:
            vertex_list = [[ox - delta, oy - delta],
                           [ox + w + delta, oy - delta],
                           [ox + w + delta, oy + h + delta],
                           [ox - delta, oy + h + delta]]
            obs_list.append(vertex_list)

        return obs_list

    def is_intersect_rec(self, start, end, o, d, a, b):
        v1 = [o[0] - a[0], o[1] - a[1]]
        v2 = [b[0] - a[0], b[1] - a[1]]
        v3 = [-d[1], d[0]]

        div = np.dot(v2, v3)

        if div == 0:
            return False

        t1 = np.linalg.norm(np.cross(v2, v1)) / div
        t2 = np.dot(v1, v3) / div

        if t1 >= 0 and 0 <= t2 <= 1:
            shot = Node((o[0] + t1 * d[0], o[1] + t1 * d[1]))
            dist_obs = self.get_dist(start, shot)
            dist_seg = self.get_dist(start, end)
            if dist_obs <= dist_seg:
                return True

        return False

    def is_intersect_circle(self, o, d, a, r):
        d2 = np.dot(d, d)
        delta = self.delta

        if d2 == 0:
            return False

        t = np.dot([a[0] - o[0], a[1] - o[1]], d) / d2

        if 0 <= t <= 1:
            shot = Node((o[0] + t * d[0], o[1] + t * d[1]))
            if self.get_dist(shot, Node(a)) <= r + delta:
                return True

        return False

    def is_collision(self, start, end, delta=None):
        
        if self.is_inside_obs(start, delta) or self.is_inside_obs(end, delta):
            return True

        o, d = self.get_ray(start, end)
        obs_vertex = self.get_obs_vertex()

        for (v1, v2, v3, v4) in obs_vertex:
            if self.is_intersect_rec(start, end, o, d, v1, v2):
                return True
            if self.is_intersect_rec(start, end, o, d, v2, v3):
                return True
            if self.is_intersect_rec(start, end, o, d, v3, v4):
                return True
            if self.is_intersect_rec(start, end, o, d, v4, v1):
                return True

        for (x, y, r) in self.obs_circle:
            if self.is_intersect_circle(o, d, [x, y], r):
                return True
            
        for path in self.obstacle:
            if self.is_intersect_polygon(start, end, path):
                return True

        return False
    
    
    def is_intersect_polygon(self, start, end, path):
        # Calculate the distance between the two points
        dist = self.get_dist(start, end)
        
        # Calculate the number of steps needed
        num_steps = int(dist / 0.01)
        
        if num_steps > 1:
            # Calculate the step size in each dimension
            step_x = (end.x - start.x) / num_steps
            step_y = (end.y - start.y) / num_steps
            
            # Interpolated Nodes
            for i in range(num_steps):
                if i == 0:
                    continue
                
                new_x = start.x + i * step_x
                new_y = start.y + i * step_y
                
                interp_node = Node((new_x, new_y))
                if self.point_in_polygon(interp_node, path.vertices):
                    return True
                

    def is_inside_obs(self, node, delta=None):
        
        if delta is None:
            delta = self.delta

        for (x, y, r) in self.obs_circle:
            if math.hypot(node.x - x, node.y - y) <= r + delta:
                return True

        for (x, y, w, h) in self.obs_rectangle:
            if 0 <= node.x - (x - delta) <= w + 2 * delta \
                    and 0 <= node.y - (y - delta) <= h + 2 * delta:
                return True

        # for (x, y, w, h) in self.obs_boundary:
        #     if 0 <= node.x - (x - delta) <= w + 2 * delta \
        #             and 0 <= node.y - (y - delta) <= h + 2 * delta:
        #         return True
            
        for (x, y, w, h) in self.obs_boundary:
            if 0 <= node.x - (x) <= w \
                    and 0 <= node.y - (y) <= h:
                return True
            
        for path in self.obstacle:
            # print(path)
            if self.point_in_polygon(node, path.vertices):
                return True

        return False
    
    # Checking if a point is inside a polygon
    def point_in_polygon(self, node, polygon):
        
        polygon = np.array(polygon)
        
        num_vertices = len(polygon)
        x, y = node.x, node.y
        inside = False
    
        # Store the first point in the polygon and initialize the second point
        p1 = polygon[0]
    
        # Loop through each edge in the polygon
        for i in range(1, num_vertices + 1):
            # Get the next point in the polygon
            p2 = polygon[i % num_vertices]
    
            # Check if the point is above the minimum y coordinate of the edge
            if y + self.delta > min(p1[1], p2[1]):
                # Check if the point is below the maximum y coordinate of the edge
                if y - self.delta <= max(p1[1], p2[1]):
                    # Check if the point is to the left of the maximum x coordinate of the edge
                    if x - self.delta <= max(p1[0], p2[0]):
                        # Calculate the x-intersection of the line connecting the point to the edge
                        x_intersection = (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
    
                        # Check if the point is on the same line as the edge or to the left of the x-intersection
                        if p1[0] == p2[0] or x <= x_intersection:
                            # Flip the inside flag
                            inside = not inside
    
            # Store the current point as the first point for the next iteration
            p1 = p2
    
        # Return the value of the inside flag
        return inside

    @staticmethod
    def get_ray(start, end):
        orig = [start.x, start.y]
        direc = [end.x - start.x, end.y - start.y]
        return orig, direc

    @staticmethod
    def get_dist(start, end):
        return math.hypot(end.x - start.x, end.y - start.y)
    
    
