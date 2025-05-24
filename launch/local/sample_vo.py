import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../lib")
from sim.camera import SimpleCamera
# from vo.core_vo import VIO_MonoVO
from vo.mono_vo import MonoStreamVO
from utils.transformations import transform_kps_nwu2camera
from utils.plotting import set_axes_equal, plot_3d_kps, get_new_ax3d

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import json

from cam_traj import generate_chess_points, cameras_around_chess


def load_scene(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Convert lists back to NumPy arrays
    pts_nwu = np.array(data["points_nwu"], dtype=np.float32)    # shape (M, 4)
    traj = np.array(data["trajectory"], dtype=np.float32)       # shape (N, 6)
    
    return pts_nwu, traj


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python load_scene.py <path_to_json>")
        sys.exit(1)

    json_path = sys.argv[1]
    pts_nwu, traj = load_scene(json_path)

    print("Loaded points (NWU):", pts_nwu.shape)
    print("Loaded trajectory:", traj.shape)

    # ------------------ Plot Static 3D NWU Frame ------------------
    plot_3d_kps(pts_nwu, "GT in NWU", pts_color='blue', camera_trajectory=traj)

    # ------------------ Plot Camera Frame Points (at t=0) ------------------
    # Only for visualization purposes; camera frame changes per timestep
    pts_cam_init = transform_kps_nwu2camera(
        pts_nwu, cam_pose=traj[0, :3],
        roll=traj[0, 4], pitch=traj[0, 5], yaw=traj[0, 5])

    plot_3d_kps(pts_cam_init, "GT in CAM", pts_color='red', plt_show=True)

    # ------------------ Loop Over Time: Project Points ------------------
    cam = SimpleCamera(hfov_deg=66, show=True, image_shape=(640, 480), ambif=False)

    vo = MonoStreamVO(cam.getK())

    ax = get_new_ax3d()
    # ax2 = get_new_ax3d()
    k = 0
    np.set_printoptions(suppress=True)
    for t in range(traj.shape[0]):
        cam_pose = traj[t, :]
        pts_cam_t = transform_kps_nwu2camera(pts_nwu, cam_pose=traj[t, :3], roll=traj[t, 3], 
            pitch=traj[t, 4], yaw=traj[t, 5])
        uvs = cam.project(pts_cam_t)

        pts_vo = vo.do_vo(uvs)
        print(f"iteration {k} - outputs: ", pts_vo)

        key = cv2.waitKey(1)
        if key == 27:  # ESC to break early
            break
        
        if k > 1:
            plot_3d_kps(pts_vo, ax=ax, pts_color='red', label='vo', title='vo points', 
                plt_pause=0.1, 
                cla=True)
            plot_3d_kps(pts_cam_t, "GT in CAM", pts_color='blue', label='gt', ax=ax, 
                plt_pause=5)
        k += 1

    cv2.destroyAllWindows()
