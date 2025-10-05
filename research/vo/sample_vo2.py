import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../lib")
from sim.camera import SimpleCamera
# from vo.core_vo import VIO_MonoVO
from vo.mono_vo2 import MonoStreamVO
from kinematics.transformations import transform_kps_nwu2camera
from utils.plotting import set_axes_equal, plot_3d_kps, get_new_ax3d

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import json

def generate_random_dist_data(pts_nwu, cam_pose, percent_with_dist=20, seed=None):
    """
    Generate Nx2 array of [id, dist_to_cam], where dist is computed for a random subset,
    and the rest are None.

    Parameters
    ----------
    pts_nwu : ndarray of shape (N, 4)
        Each row is [id, x, y, z]
    cam_pose : array-like of shape (3,)
        Camera position [x, y, z] in NWU
    percent_with_dist : float
        Percentage (0–100) of points to assign distances to.
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    random_dist_data : ndarray of shape (N, 2), where dist is float or None
    """
    if seed is not None:
        np.random.seed(seed)

    N = pts_nwu.shape[0]
    ids = pts_nwu[:, 0].astype(int)
    pts_xyz = pts_nwu[:, 1:]

    num_with_dist = int(np.round(N * percent_with_dist / 100))

    # Compute all distances to camera
    dists = np.linalg.norm(pts_xyz - cam_pose[:3], axis=1)

    # Choose which points to keep distances for
    selected = np.random.choice(N, size=num_with_dist, replace=False)

    # Fill with None, then set selected
    dist_data = np.full(N, None, dtype=object)
    dist_data[selected] = dists[selected]

    return np.column_stack((ids, dist_data))


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
        random_dist_data = generate_random_dist_data(pts_nwu, cam_pose)

        print(random_dist_data)
        pts_vo = vo.do_vo(uvs, random_dist_data)
        print(f"iteration {k}")# - outputs: ", pts_vo)

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
