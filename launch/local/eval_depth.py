import sys
import numpy as np
import os
import open3d as o3d

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../lib")
from mapper3D_helper import *
from mapper3D import Mapper3D

roll = 0
yaw = 0
pitch = -0.02
hfov = 80
z = 2

def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <depth_image1.pcd> <depth_image2.pcd>")
        sys.exit(1)

    depth_path1, depth_path2 = sys.argv[1:3]
    p1 = o3d.io.read_point_cloud(depth_path1)
    p2 = o3d.io.read_point_cloud(depth_path2)

    # Convert points to NumPy arrays
    pts1 = np.asarray(p1.points)
    pts2 = np.asarray(p2.points)
    all_pts = np.vstack([pts1, pts2])

    # Convert colors to NumPy arrays
    col1 = np.asarray(p1.colors)
    col2 = np.asarray(p2.colors)
    col2_modified = np.ones_like(col2)
    col2_modified[:, 2] = 0  # set blue channel to 0
    all_colors = np.vstack([col1, col2_modified])

    # Create new point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()

