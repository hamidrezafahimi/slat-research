import math
import numpy as np
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../lib")
from sim.camera import SimpleCamera
from sim.box import Box
from sim.trajectory_generation import get_cam_pose, get_cam_trajectory
from utils.transformations import transform_kps_nwu2camera   # your helper  :contentReference[oaicite:0]{index=0}
from utils.plotting import visualize_trajectory_and_points
import cv2
import json

# Single point sample
# if __name__ == '__main__':
#     cam = SimpleCamera(hfov_deg=66, show=True, log=True, image_shape=(640, 480))
#     box = Box(anchor=(0,0,0), dims=(5,5,3))
#     pts_nwu = box.get_random_points(10)
#     cam_pose, roll, pitch, yaw = get_cam_pose(cam, box, D=9.0)
#     pts_cam_t = transform_kps_nwu2camera(pts_nwu, cam_pose=cam_pose, roll=roll, pitch=pitch, 
#             yaw=yaw)
#     uvs = cam.project(pts_cam_t)
#     cv2.waitKey()
#     visualize_trajectory_and_points(pts_nwu, np.array([[cam_pose[0], cam_pose[1], cam_pose[2],
#         roll, pitch, yaw]]))


# # Trajectory sample
if __name__ == '__main__':
    # ❶ create a *silent* camera for sampling, then a *visual* one for playback
    cam_sampler = SimpleCamera(hfov_deg=66, image_shape=(640, 480))
    cam_viewer  = SimpleCamera(hfov_deg=66, image_shape=(640, 480),
                               show=True, log=True, ambif=False)

    box      = Box(anchor=(0, 0, 0), dims=(4, 4, 3))
    pts_nwu  = box.get_random_on_nodes()          # shape (M,4)

    # ❷ build a 10-pose tour that never moves more than 5 m per step
    traj = get_cam_trajectory(cam_sampler, box,
                              N=10, D=10.0, step_size=5.0,
                              log_every=2, smooth=True, smooth_upsample=2)
    
    print("pts_nwu:", pts_nwu.shape, "traj:", traj.shape)

    # — save to JSON before visualizing or playing back —
    data = {
        "points_nwu": pts_nwu.tolist(),        # [[id, x, y, z], ...]
        "trajectory": traj.tolist(),           # [[x, y, z, roll, pitch, yaw], ...]
    }
    with open("scene_with_traj.json", "w") as f:
        json.dump(data, f, indent=4)
    print("Saved points & trajectory to scene_with_traj.json")

    # ❸ visualize & play back
    visualize_trajectory_and_points(pts_nwu, traj)
    for pose in traj:
        x, y, z, r, p, yw = pose
        pts_cam = transform_kps_nwu2camera(
            pts_nwu,
            cam_pose=np.array([x, y, z]),
            roll=r, pitch=p, yaw=yw,
        )
        cam_viewer.project(pts_cam)          # will raise if any point fails
        cv2.waitKey(300)                     # ~3 fps preview

    print("Trajectory done – press any key to quit.")
    cv2.waitKey()