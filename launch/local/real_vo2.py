import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../lib")
from sim.camera import SimpleCamera
# from vo.core_vo import VIO_MonoVO
from vo.mono_vo2 import MonoStreamVO
from utils.transformations import transform_kps_camera2nwu
from utils.plotting import set_axes_equal, plot_3d_kps, get_new_ax3d

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import json
import random


def load_keypoints_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    # Convert to list of (id, u, v)
    img1_kps = [(int(k), v["x"], v["y"]) for k, v in data["image1"].items()]
    img2_kps = [(int(k), v["x"], v["y"]) for k, v in data["image2"].items()]

    uvs1 = np.array(img1_kps, dtype=np.float32)
    uvs2 = np.array(img2_kps, dtype=np.float32)

    return uvs1, uvs2


def render_depth_circles(uvs: np.ndarray, pts: np.ndarray, width: int, height: int, radius: int) -> np.ndarray:
    """
    Render a monochrome image with circles at (u,v) positions from `uvs`,
    brightness based on 3D distance from origin computed from `pts`.

    Only uses IDs present in both `uvs` and `pts`.

    Parameters:
        uvs (np.ndarray): Nx3 array of [id, u, v] coordinates.
        pts (np.ndarray): Mx4 array of [id, x, y, z] coordinates.
        width (int): Width of the output image.
        height (int): Height of the output image.
        radius (int): Radius of the circles to draw.

    Returns:
        np.ndarray: 2D monochrome image (uint8).
    """
    # Convert to dictionaries for fast lookup
    uvs_dict = {int(row[0]): row[1:] for row in uvs}
    pts_dict = {int(row[0]): row[1:] for row in pts}

    # Find common IDs
    common_ids = sorted(set(uvs_dict.keys()) & set(pts_dict.keys()))
    if not common_ids:
        return np.zeros((height, width), dtype=np.uint8)  # return blank if no overlap

    # Extract matched data
    matched_uvs = np.array([uvs_dict[i] for i in common_ids])
    matched_pts = np.array([pts_dict[i] for i in common_ids])

    # Compute distances
    distances = np.sqrt(np.sum(matched_pts ** 2, axis=1))

    # Normalize distances: closest = 255, farthest = 0
    d_min = distances.min()
    d_max = distances.max()
    if d_max == d_min:
        normalized = np.full_like(distances, 255, dtype=np.uint8)
    else:
        normalized = 255.0 * (1.0 - (distances - d_min) / (d_max - d_min))
        normalized = normalized.astype(np.uint8)

    # Create black image
    img = np.zeros((height, width), dtype=np.uint8)

    # Draw circles
    for (u, v), brightness in zip(matched_uvs, normalized):
        center = (int(round(u)), int(round(v)))
        cv2.circle(img, center, radius, int(brightness), -1)

    return img


if __name__ == '__main__':
    # Usage:
    #   python run_vo_from_json.py <keypoints.json> <depth_dir> [percent]
    #
    #   <keypoints.json> : JSON file with frames of keypoints.
    #   <depth_dir>      : Directory containing depth images (one per frame).
    #   [percent]        : Optional integer percentage of keypoints to sample distance for (default: 20).
    #
    # Example:
    #   python run_vo_from_json.py keypoints.json depth_images 20

    if not (3 <= len(sys.argv) <= 4):
        print("Usage: python run_vo_from_json.py <keypoints.json> <depth_dir> [percent]")
        sys.exit(1)

    json_path = sys.argv[1]
    depth_dir = sys.argv[2]
    try:
        percent = int(sys.argv[3]) if len(sys.argv) == 4 else 20
    except ValueError:
        print("Error: [percent] must be an integer.")
        sys.exit(1)
    if not (0 <= percent <= 100):
        print("Error: [percent] must be between 0 and 100.")
        sys.exit(1)

    # Load the JSON data of keypoints
    with open(json_path, "r") as f:
        data = json.load(f)

    # Gather and sort depth-image filenames
    depth_files = [
        os.path.join(depth_dir, fname)
        for fname in sorted(os.listdir(depth_dir))
        if os.path.isfile(os.path.join(depth_dir, fname))
    ]
    if len(depth_files) != len(data):
        print(
            f"Error: Number of depth images ({len(depth_files)}) "
            f"does not match number of frames in JSON ({len(data)})."
        )
        sys.exit(1)

    # Initialize camera and VO
    cam = SimpleCamera(hfov_deg=66, show=True, image_shape=(440, 330), ambif=False)
    vo = MonoStreamVO(cam.getK())

    ax = get_new_ax3d()
    np.set_printoptions(suppress=True)

    # Iterate over all frames
    for k, frame_data in enumerate(data):
        frame_index = frame_data["frame"]
        keypoints = frame_data["keypoints"]
        print(f"Processing frame {frame_index} (index {k})")

        # Extract UVs array of shape (N, 3): [id, u, v]
        uvs = np.array([[kp[0], kp[1], kp[2]] for kp in keypoints])
        if uvs.shape[0] == 0:
            print("No keypoints in this frame; exiting loop.")
            break

        # -- Read and preprocess the corresponding depth image --
        depth_path = depth_files[k]
        depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        if depth_img is None:
            print(f"Error: could not load depth image '{depth_path}'.")
            sys.exit(1)
        # Invert the depth image: new_depth = 255 - original
        depth_img = 255 - depth_img

        # -- Build dist_data for this frame --
        # dist_data: list of [id, dist] or [id, None], length = number of keypoints
        num_points = uvs.shape[0]
        num_sample = int(np.ceil((percent / 100.0) * num_points))
        all_indices = list(range(num_points))
        sampled_indices = set(random.sample(all_indices, k=num_sample))

        dist_data = []
        height, width = depth_img.shape
        for idx in range(num_points):
            pid, uf, vf = uvs[idx]
            pid = int(pid)
            if idx in sampled_indices:
                # Convert float uv to nearest integer pixel coordinates
                u_int = int(round(uf))
                v_int = int(round(vf))
                # Clamp to image boundaries
                u_int = max(0, min(u_int, width - 1))
                v_int = max(0, min(v_int, height - 1))
                dist = float(depth_img[v_int, u_int])
                dist_data.append([pid, dist])
            else:
                dist_data.append([pid, None])

        # -- Run VO for this frame --
        pts_vo = vo.do_vo(uvs, dist_data)
        print(f"VO returned {pts_vo.shape[0]} points.")

        if pts_vo.shape[0] != 0:
            # Transform to NWU and plot
            pts_nwu = transform_kps_camera2nwu(pts_vo, np.array([0, 0, 0]), 0, -0.78, -3.14)
            plot_3d_kps(
                pts_nwu,
                ax=ax,
                pts_color='green',
                label=f'vo{k}',
                title=f'VO Output Frame {k}',
                plt_pause=0.1,
                write_id=True,
                cla=True
            )
        else:
            pts_nwu = np.empty((0, 4))

        key = cv2.waitKey(1)
        if key == 27:  # ESC to quit
            print("ESC pressed; exiting.")
            break

    # After the loop: show final VO results
    if 'pts_vo' in locals() and pts_vo.shape[0] != 0:
        plot_3d_kps(
            pts_vo,
            ax=ax,
            pts_color='green',
            label=f'vo_final',
            title=f'VO Output Final Frame',
            plt_show=True,
            write_id=True,
            cla=True
        )

    # Render depth circles on last frame's UVs/VO (if any)
    if 'pts_vo' in locals() and pts_vo.shape[0] != 0 and 'uvs' in locals():
        img_out = render_depth_circles(uvs, pts_vo, 440, 330, 5)
        cv2.imwrite("vo_out.png", img_out)
        print("Saved depth-rendered image as 'vo_out.png'.")

    # Prepare and save JSON with VO results
    output_vo = pts_vo.tolist() if ('pts_vo' in locals() and pts_vo is not None) else []
    output_nwu = pts_nwu.tolist() if ('pts_nwu' in locals() and pts_nwu is not None) else []
    output_dict = {
        "pts_vo": output_vo,     # [[id, x, y, z], ...]
        "pts_nwu": output_nwu    # [[id, x, y, z], ...]
    }
    with open("vo_output.json", "w") as f_out:
        json.dump(output_dict, f_out, indent=2)
    print("Saved VO output to 'vo_output.json'.")    # cv2.waitKey()

    # cv2.destroyAllWindows()
