import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../lib")
from sim.camera import SimpleCamera
# from vo.core_vo import VIO_MonoVO
from vo.mono_vo import MonoStreamVO
from utils.transformations import transform_kps_nwu2camera
from utils.plotting import plot_3d_kps, get_new_ax3d

import numpy as np
import cv2
import json
import argparse


def load_scene(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Convert lists back to NumPy arrays
    pts_nwu = np.array(data["points_nwu"], dtype=np.float32)    # shape (M, 4)
    traj = np.array(data["trajectory"], dtype=np.float32)       # shape (N, 6)
    shape = (data["camera"]["width"], data["camera"]["height"])
    hfov_deg = data["camera"]["hfov_deg"]
    
    return pts_nwu, traj, shape, hfov_deg


import argparse
import json
import numpy as np
import cv2

# Import or define these before running:
# from your_module import (
#     load_scene,
#     plot_3d_kps,
#     transform_kps_nwu2camera,
#     SimpleCamera,
#     MonoStreamVO,
#     get_new_ax3d
# )

def main():
    parser = argparse.ArgumentParser(
        description="Load a scene JSON, visualize ground truth, run VO with optional noise, and save VO + GT output."
    )
    parser.add_argument(
        'json_path',
        help="Path to the scene JSON (contains points_nwu, trajectory, camera hfov/shape)."
    )
    parser.add_argument(
        '--noise_std',
        type=float,
        default=0.0,
        help="Standard deviation of projection noise (default = 0 → no noise)."
    )
    parser.add_argument(
        '--no_imshow',
        dest='imshow',
        action='store_false',
        help="If present, turn off image display (cv2.imshow) for cam_viewer. Default: images shown."
    )
    parser.add_argument(
        '--no_plot',
        dest='plot',
        action='store_false',
        help="If present, skip calling plot_3d_kps. Default: trajectory/VO plots shown."
    )
    parser.add_argument(
        '--out_vo',
        type=str,
        default=None,
        help="If provided, save VO + GT outputs per frame to this JSON file."
    )

    # By default, imshow and plot are True
    parser.set_defaults(imshow=True, plot=True)

    args = parser.parse_args()
    do_imshow = args.imshow
    do_plot = args.plot
    out_vo_path = args.out_vo
    noise_std = args.noise_std

    # Load scene: returns (pts_nwu (M×4), traj (N×6), image_shape (W,H), hfov_deg)
    pts_nwu, traj, image_shape, hfov_deg = load_scene(args.json_path)

    print("Loaded points (NWU):", pts_nwu.shape)
    print("Loaded trajectory:", traj.shape)

    # ------------------ Plot Static 3D NWU Frame ------------------
    if do_plot:
        plot_3d_kps(
            pts_nwu,
            title="GT in NWU",
            pts_color='blue',
            camera_trajectory=traj
        )

    # ------------------ Plot Camera Frame Points (t=0) ------------------
    x0, y0, z0 = traj[0, :3]
    roll0, pitch0, yaw0 = traj[0, 3], traj[0, 4], traj[0, 5]
    pts_cam_init = transform_kps_nwu2camera(
        pts_nwu,
        cam_pose=np.array([x0, y0, z0]),
        roll=roll0,
        pitch=pitch0,
        yaw=yaw0
    )
    if do_plot:
        plot_3d_kps(
            pts_cam_init,
            title="GT in CAM (t=0)",
            pts_color='red',
            plt_show=True
        )

    # ------------------ Loop Over Time: Project Points with VO ------------------
    show_noise_flag = noise_std > 0.0
    cam = SimpleCamera(
        hfov_deg=hfov_deg,
        show=do_imshow,
        image_shape=tuple(image_shape),
        ambif=False,
        noise_std=noise_std,
        show_noise=show_noise_flag,
        report_disp=show_noise_flag
    )

    vo = MonoStreamVO(cam.getK())
    ax = get_new_ax3d()
    np.set_printoptions(suppress=True)

    # Prepare to collect results if out_vo_path is given
    vo_results = [] if out_vo_path is not None else None
    displacement_list = []

    for k in range(traj.shape[0]):
        # Extract ground-truth pose
        x, y, z = traj[k, :3]
        roll, pitch, yaw = traj[k, 3], traj[k, 4], traj[k, 5]

        # Compute ground-truth camera-frame points
        pts_cam_t = transform_kps_nwu2camera(
            pts_nwu,
            cam_pose=np.array([x, y, z]),
            roll=roll,
            pitch=pitch,
            yaw=yaw
        )

        # VO prediction
        uvs = cam.project(pts_cam_t)
        pts_vo = vo.do_vo(uvs)
        print(f"Iteration {k} → VO outputs: {pts_vo.shape[0]} points")

        # Collect displacement metric
        displacement_list.append(cam.mean_pct)

        # If requested, save both VO points and GT (camera-frame) points
        if out_vo_path is not None:
            vo_results.append({
                "frame": k,
                "vo_points": pts_vo.tolist(),
                "gt_points": pts_cam_t.tolist()
            })

        if do_imshow:
            key = cv2.waitKey(1)
            if key == 27:  # ESC key to break early
                break

        if do_plot and k > 1:
            # Plot Vo points (green) and GT camera-frame points (blue) on same axes
            plot_3d_kps(
                pts_vo,
                ax=ax,
                pts_color='green',
                label='VO',
                title=f'Frame {k}: VO vs. GT in CAM',
                plt_pause=0.1,
                cla=True
            )
            plot_3d_kps(
                pts_cam_t,
                ax=ax,
                pts_color='blue',
                label='GT',
                plt_pause=0.1
            )

    if do_imshow:
        cv2.destroyAllWindows()

    # ------------------ Compute and Save Ratio if requested ------------------
    if out_vo_path is not None:
        # Compute average displacement over all iterations
        avg_disp = float(np.mean(displacement_list)) if displacement_list else 0.0
        ratio = noise_std / avg_disp if avg_disp != 0 else None

        output_data = {
            "vo_results": vo_results,
            "noise_disp_ratio": ratio
        }
        with open(out_vo_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"VO + GT output (with noise_disp_ratio={ratio}) saved to {out_vo_path}")

if __name__ == "__main__":
    main()

