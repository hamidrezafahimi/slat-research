import sys
import numpy as np
import os
import open3d as o3d

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../lib")
from mapper3D_helper import *
from mapper3D import Mapper3D

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <depth_image1.pcd>")
        sys.exit(1)

    depth_path1 = sys.argv[1]
    p1 = o3d.io.read_point_cloud(depth_path1)

    # Convert points to NumPy arrays
    pts1 = np.asarray(p1.points)

    # Convert colors to NumPy arrays
    col1 = np.asarray(p1.colors)

    # Create new point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts1)
    pcd.colors = o3d.utility.Vector3dVector(col1)

    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()

