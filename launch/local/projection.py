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
import argparse

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

def main():
    parser = argparse.ArgumentParser(
        description="Generate and visualize a camera trajectory over a box with configurable parameters."
    )
    parser.add_argument(
        '--hfov_deg_base', type=float, default=132,
        help="Horizontal FOV for the sampler camera (degrees)"
    )
    parser.add_argument(
        '--image_shape_base', type=int, nargs=2, metavar=('WIDTH','HEIGHT'),
        default=[880, 660],
        help="Width and height for the sampler camera image (e.g. 880 660)"
    )
    parser.add_argument(
        '--hfov_deg_view', type=float, default=66,
        help="Horizontal FOV for the viewer camera (degrees)"
    )
    parser.add_argument(
        '--image_shape_view', type=int, nargs=2, metavar=('WIDTH','HEIGHT'),
        default=[440, 330],
        help="Width and height for the viewer camera image (e.g. 440 330)"
    )
    parser.add_argument(
        '--box_dims', type=float, nargs=3, metavar=('DIM_X','DIM_Y','DIM_Z'),
        default=[4, 4, 4],
        help="Dimensions of the box in NWU coordinates (e.g. 7 7 4)"
    )
    parser.add_argument(
        '--traj_N', type=int, default=10,
        help="Number of poses in the trajectory (N)"
    )
    parser.add_argument(
        '--traj_D', type=float, default=7.0,
        help="Initial distance constraint D (max distance from first pose)"
    )
    parser.add_argument(
        '--traj_step_size', type=float, default=1.0,
        help="Maximum step size between successive poses (in meters)"
    )
    parser.add_argument(
        '--no_imshow',
        dest='imshow',
        action='store_false',
        help="If present, turn off image display (via cv2.imshow) for cam_viewer. Default: images are shown."
    )
    parser.add_argument(
        '--no_plot',
        dest='plot',
        action='store_false',
        help="If present, skip calling visualize_trajectory_and_points. Default: trajectory is plotted."
    )

    # By default, imshow and plot are True when flags are not given
    parser.set_defaults(imshow=True, plot=True)

    args = parser.parse_args()

    # Unpack command-line arguments
    hfov_deg_base = args.hfov_deg_base
    image_shape_base = tuple(args.image_shape_base)
    hfov_deg_view = args.hfov_deg_view
    image_shape_view = tuple(args.image_shape_view)
    box_dims = tuple(args.box_dims)
    traj_N = args.traj_N
    traj_D = args.traj_D
    traj_step_size = args.traj_step_size
    do_imshow = args.imshow
    do_plot = args.plot

    # ❶ Create cameras
    cam_sampler = SimpleCamera(hfov_deg=hfov_deg_base, image_shape=image_shape_base)
    cam_viewer  = SimpleCamera(
        hfov_deg=hfov_deg_view,
        image_shape=image_shape_view,
        show=do_imshow,
        log=True,
        ambif=False
    )

    # Create the box and sample points
    box = Box(anchor=(0, 0, 0), dims=box_dims)
    pts_nwu = box.get_random_on_nodes()  # shape (M, 4)

    # ❷ Build trajectory
    traj = get_cam_trajectory(
        cam_sampler,
        box,
        N=traj_N,
        D=traj_D,
        step_size=traj_step_size,
        log_every=2,
        smooth=True,
        smooth_upsample=2
    )

    print("pts_nwu:", pts_nwu.shape, "traj:", traj.shape)

    # Save to JSON
    data = {
        "camera": {
            "width": image_shape_view[0],
            "height": image_shape_view[1],
            "hfov_deg": hfov_deg_view
        },
        "points_nwu": pts_nwu.tolist(),       # [[id, x, y, z], ...]
        "trajectory": traj.tolist(),          # [[x, y, z, roll, pitch, yaw], ...]
    }
    with open("scene_with_traj.json", "w") as f:
        json.dump(data, f, indent=4)
    print("Saved points & trajectory to scene_with_traj.json")

    # ❸ Optionally plot in 3D
    if do_plot:
        visualize_trajectory_and_points(pts_nwu, traj)

    # ❹ Optionally iterate through trajectory and show images
    for pose in traj:
        x, y, z, r, p, yw = pose
        pts_cam = transform_kps_nwu2camera(
            pts_nwu,
            cam_pose=np.array([x, y, z]),
            roll=r, pitch=p, yaw=yw,
        )
        # Only call cam_viewer.project if imshow is enabled
        if do_imshow:
            cam_viewer.project(pts_cam)   # will raise if any point fails
            cv2.waitKey(300)              # ~3 fps preview

    print("Done. Press any key to quit.")
    if do_imshow:
        cv2.waitKey()

if __name__ == "__main__":
    main()
