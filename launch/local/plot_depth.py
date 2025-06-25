import sys
import numpy as np
import os
import cv2
import open3d as o3d

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../lib")
from mapper3D_helper import *
from mapper3D import Mapper3D

roll = 0
yaw = 0
pitch = -0.1
hfov = 80
z = 4

def save_point_cloud(points, colors, filename):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize RGB to [0,1]

    ext = os.path.splitext(filename)[1].lower()
    if ext == '.pcd':
        o3d.io.write_point_cloud(filename, pc, write_ascii=True)
        print(f"Saved point cloud as PCD: {filename}")
    elif ext == '.ply':
        o3d.io.write_point_cloud(filename, pc, write_ascii=True)
        print(f"Saved point cloud as PLY: {filename}")
    else:
        print(f"Unsupported file format: {ext}. Use .pcd or .ply only.")

def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <depth_image.csv> <color_image.jpg> [output.pcd|output.ply]")
        sys.exit(1)

    depth_img_path, color_img_path = sys.argv[1:3]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None

    depth_image = np.loadtxt(depth_img_path, delimiter=',', dtype=np.float32)
    color_img = cv2.imread(color_img_path)

    depth_pc = depthImage2pointCloud(depth_image, roll_rad=roll, pitch_rad=pitch, 
                                     yaw_rad=yaw, horizontal_fov=hfov, abs_alt=abs(z))

    mapper = Mapper3D(vis=True, plot=True, color_mode='image', backend='matplotlib')
    mapper.plot_point_cloud(depth_pc, color_img)

    if output_path:
        xyz = depth_pc.reshape(-1, 3)
        rgb = color_img.reshape(-1, 3)
        save_point_cloud(xyz, rgb, output_path)

if __name__ == '__main__':
    main()
