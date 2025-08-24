import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../lib")
from kinematics.pose import Pose
from mapper3D_helper import *
import numpy as np
import open3d as o3d
import cv2

def plot_point_cloud_and_save(point_cloud, visualization_image=None, save=False, filename = "output"):
    points = point_cloud.reshape(-1, 3)
    geometries = []
    assert visualization_image is not None
    h, w, _ = visualization_image.shape
    print(point_cloud.shape)
    assert point_cloud.shape[:2] == (h, w), "Image and point cloud dimensions mismatch."
    colors = visualization_image.reshape(-1, 3).astype(np.float32) / 255.0

    # Point cloud geometry
    pcd = o3d.geometry.PointCloud()
    red_point = np.array([[0.0, 0.0, 0.0]])
    red_color = np.array([[1.0, 0.0, 0.0]])
    points = np.vstack([points, red_point])
    colors = np.vstack([colors, red_color])
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    geometries.append(pcd)


    # Display
    o3d.visualization.draw_geometries(geometries)

    if save:
        output_filename = str(filename) + ".pcd"
        o3d.io.write_point_cloud(output_filename, pcd)
    
def generate_gep_data(pose: Pose):
    # Compute ground elevation profile - What the true background must be, with 255.0 as max val
    gep_depth = calc_ground_depth(hfov_deg, pitch_rad=pose._pitch_rad, 
                                    output_shape=metric_depth.shape)
    gep_pc_scaled = project3DAndScale(gep_depth, pose, hfov_deg, metric_depth.shape)
    return gep_pc_scaled, gep_depth


metric_depth_path, metric_bg_path, color_path = sys.argv[1:4]

hfov_deg = 80.0

p = Pose(x=0,y=0,z=1.7,roll=0,pitch=-0.15,yaw=0)

metric_depth = np.loadtxt(metric_depth_path, delimiter=',', dtype=np.float32)[200:,:]
metric_bg = np.loadtxt(metric_bg_path, delimiter=',', dtype=np.float32)[200:,:]
visimg = cv2.imread(color_path)[200:,:]

metric_depth_s = metric_depth / np.max(metric_depth) * 255.0
metric_bg_s = metric_bg / np.max(metric_depth) * 255.0
gep_pc_scaled, gep_depth = generate_gep_data(pose=p)

rd_pc_scaled = project3DAndScale(metric_depth_s, p, hfov_deg, metric_depth.shape)

bg_pc_scaled = project3DAndScale(metric_bg_s, p, hfov_deg, metric_depth.shape)

# if ref_mode == RefusionMode.Replace_25D:
# bg_scaled = np.where(metric_bg_s < metric_depth_s, metric_depth_s, metric_bg_s)
# fimg = move_depth(metric_depth_s, bg_scaled, gep_depth)
# refused_pc = project3DAndScale(fimg, p, hfov_deg, metric_depth.shape)
# elif ref_mode == RefusionMode.Unfold:
# refused_pc = unfold_depth(rd_pc_scaled, bg_pc_scaled, gep_pc_scaled)
# elif ref_mode == RefusionMode.Drop:
refused_pc = drop_depth(rd_pc_scaled, bg_pc_scaled, gep_pc_scaled)

plot_point_cloud_and_save(refused_pc, visualization_image= visimg)
# plot_point_cloud_and_save(bg_pc_scaled, visualization_image= visimg)