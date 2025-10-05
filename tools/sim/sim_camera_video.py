import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../lib")
from sim.camera import SimpleCamera
from kinematics.transformations import transform_kps_nwu2camera
from utils.plotting import set_axes_equal

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ------------------ Static NWU Keypoints ------------------
pts_nwu = np.array([
    [0, -1.0, -1.0, 0.0],
    [1, -1.0,  1.0, 0.0],
    [2,  1.0,  1.0, 0.0],
    [3,  1.0, -1.0, 0.0]
])

# ------------------ Simulated Camera Trajectory ------------------
N = 5
camera_trajectory = np.array([
    [10.0, 0.0+i, 10.0 - 0.5 * i] for i in range(N)
])  # Example: moving right + descending

roll = 0.0
pitch = -0.78
yaw = -3.14

# ------------------ Plot Static 3D NWU Frame ------------------
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter(pts_nwu[:, 1], pts_nwu[:, 2], pts_nwu[:, 3], c='blue', label='NWU Points')
for pt in pts_nwu:
    ax1.text(*pt[1:], f'N{pt[0]}', color='blue')

ax1.plot(camera_trajectory[:, 0], camera_trajectory[:, 1], camera_trajectory[:, 2], 
         'g-o', label='Camera Trajectory')
ax1.set_title("NWU Frame")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")
ax1.legend()
ax1.view_init(elev=30, azim=-45)
set_axes_equal(ax1)

# ------------------ Plot Camera Frame Points (at t=0) ------------------
# Only for visualization purposes; camera frame changes per timestep
pts_cam_init = transform_kps_nwu2camera(
    pts_nwu, cam_pose=camera_trajectory[0],
    roll=roll, pitch=pitch, yaw=yaw
)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.scatter(pts_cam_init[:, 1], pts_cam_init[:, 2], pts_cam_init[:, 3], 
            c='red', label='Camera Frame Points (t=0)')
for pt in pts_cam_init:
    ax2.text(*pt[1:], f'C{pt[0]}', color='red')

ax2.set_title("Camera Frame (Initial)")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")
ax2.legend()
ax2.view_init(elev=30, azim=-45)
set_axes_equal(ax2)

plt.show()

# ------------------ Loop Over Time: Project Points ------------------
cam = SimpleCamera(hfov_deg=66, show=True, image_shape=(640, 480))

for t in range(N):
    cam_pose = camera_trajectory[t]
    pts_cam_t = transform_kps_nwu2camera(
        pts_nwu, cam_pose=cam_pose,
        roll=roll, pitch=pitch, yaw=yaw
    )
    uvs = cam.project_3dTo2d_pc(pts_cam_t)

    print(f"Frame {t}: Press any key in the image window to continue …")
    key = cv2.waitKey(0)
    if key == 27:  # ESC to break early
        break

cv2.destroyAllWindows()
