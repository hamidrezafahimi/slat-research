import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../lib")
from kinematics.pose import Pose
from mapper3D_helper import *
import numpy as np
import open3d as o3d
import cv2

def plot_point_cloud_and_save(point_cloud, visualization_image=None, save=False, filename = "pcd_file"):
    points = point_cloud.reshape(-1, 3)
    geometries = []
    assert visualization_image is not None
    h, w, _ = visualization_image.shape
    print(point_cloud.shape)
    assert point_cloud.shape[:2] == (h, w), "Image and point cloud dimensions mismatch."
    colors = visualization_image.reshape(-1, 3).astype(np.float32) / 255.0
    # Point cloud geometry
    pcd = o3d.geometry.PointCloud()
    # red_point = np.array([[0.0, 0.0, 0.0]])
    # red_color = np.array([[1.0, 0.0, 0.0]])
    # points = np.vstack([points, red_point])
    # colors = np.vstack([colors, red_color])
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    geometries.append(pcd)
    geometries.append(axis)
    # Display
    o3d.visualization.draw_geometries(geometries)
    if save:
        output_filename = str(filename) + ".pcd"
        o3d.io.write_point_cloud(output_filename, pcd)
    

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py metric_depth.csv color.png")
        sys.exit(1)

    metric_depth_path, color_path = sys.argv[1:3]

    # hfov_deg = 80.0
    hfov_deg = 66.0

    # p = Pose(x=0,y=0,z=1.7,roll=0,pitch=-0.15,yaw=0)
    p = Pose(x=0,y=0,z=9.4,roll=0,pitch=-0.78,yaw=0)

    # metric_depth = np.loadtxt(metric_depth_path, delimiter=',', dtype=np.float32)[200:,:]
    metric_depth = np.loadtxt(metric_depth_path, delimiter=',', dtype=np.float32)

    rd_pc_scaled = project3DAndScale(metric_depth, p, hfov_deg, metric_depth.shape)

    # plot_point_cloud_and_save(rd_pc_scaled, visualization_image= cv2.imread(color_path)[200:,:], save=True)
    plot_point_cloud_and_save(rd_pc_scaled, visualization_image= cv2.imread(color_path), save=True)