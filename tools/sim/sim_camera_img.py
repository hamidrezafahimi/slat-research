import numpy as np
import cv2
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../lib")
from sim.camera import SimpleCamera
from kinematics.transformations import transform_kps_nwu2camera
from utils.plotting import set_axes_equal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ------------------ Input ------------------
pts_nwu = np.array([
    [0, -1.0, -1.0, 0.0],
    [1, -1.0,  1.0, 0.0],
    [2, 1.0,  1.0, 0.0],
    [3, 1.0, -1.0, 0.0]
])

camera_pose_nwu = np.array([10.0, 0.0, 10.0])
roll = 0.0
pitch = -0.78
yaw = -3.14

pts_cam = transform_kps_nwu2camera (
        pts_nwu, cam_pose=camera_pose_nwu,
        roll=roll, pitch=pitch, yaw=yaw
    )

# ------------------ Plot NWU Frame ------------------
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter(pts_nwu[:, 1], pts_nwu[:, 2], pts_nwu[:, 3], c='blue', label='NWU Points')
for pt in pts_nwu:
    ax1.text(*pt[1:], f'N{pt[0]}', color='blue')

ax1.scatter(*camera_pose_nwu, color='green', s=100, label='Camera Position')

ax1.set_title("NWU Frame")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")
ax1.legend()
ax1.view_init(elev=30, azim=-45)
set_axes_equal(ax1)

# ------------------ Plot Camera Frame ------------------
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.scatter(pts_cam[:, 1], pts_cam[:, 2], pts_cam[:, 3], c='red', label='Camera Frame Points')
for pt in pts_cam:
    ax2.text(*pt[1:], f'C{pt[0]}', color='red')

ax2.set_title("Camera Frame")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")
ax2.legend()
ax2.view_init(elev=30, azim=-45)
set_axes_equal(ax2)

plt.show()

# ------------------ Optional 2D Projection ------------------
cam = SimpleCamera(hfov_deg=66, show=True, image_shape=(640, 480))
cam.project_3dTo2d_pc(pts_cam)
