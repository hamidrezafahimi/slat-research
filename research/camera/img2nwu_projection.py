import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../lib")
from mapper3D_helper import *


altitude = 2
gepimg = calc_ground_depth(80.0, 0, (176,1241), fixed_alt=10.0, horizon_pitch_rad=-0.034)
pc1, _ = depthImage2pointCloud(gepimg, 80.0, 0, -0.15, 0)
gpc = np.zeros_like(pc1).astype(np.float32)
gpc[:,:,2] = -abs(altitude)
scale_factor = gpc[:,:,2] / pc1[:,:,2]
gep_pc_scaled, _ = depthImage2pointCloud(gepimg, 80.0, 0, -0.15, 0, scale_factor=scale_factor)
points = gep_pc_scaled.reshape(-1, 3)
colors = np.zeros_like(points)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd])
