import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../lib")
from sim.camera import SimpleCamera
# from vo.core_vo import VIO_MonoVO
from vo.mono_vo import MonoStreamVO
from utils.transformations import transform_kps_camera2nwu
from utils.plotting import set_axes_equal, plot_3d_kps, get_new_ax3d

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import json

"""Monocular VO sample"""


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
    # Load the JSON data
    if len(sys.argv) != 2:
        print("Usage: python run_vo_from_json.py <keypoints.json>")
        sys.exit(1)
    json_path = sys.argv[1]
    with open(json_path, "r") as f:
        data = json.load(f)

    cam = SimpleCamera(hfov_deg=66, show=True, image_shape=(440, 330), ambif=False)
    vo = MonoStreamVO(cam.getK())

    ax = get_new_ax3d()
    np.set_printoptions(suppress=True)

    # Iterate over all frames
    k = 0
    for frame_data in data:
        frame_index = frame_data["frame"]
        keypoints = frame_data["keypoints"]
        print("processing frame", frame_index)

        # Extract UVs
        uvs = np.array([[kp[0], kp[1], kp[2]] for kp in keypoints])  # Ignore the keypoint index (kp[0])

        if uvs.shape[0] == 0:
            break

        pts_vo = vo.do_vo(uvs)
        print(pts_vo.shape)

        if pts_vo.shape[0] != 0:
            pts_nwu = transform_kps_camera2nwu(pts_vo, np.array([0,0,0]), 0, -0.78, -3.14)
            # Example plotting (update with real 3D plotting if needed)
            plot_3d_kps(pts_nwu, ax=ax, pts_color='green', label=f'vo{k}', 
                        title=f'VO Output Frame {k}', plt_pause=0.1, write_id=True, cla=True)

        k += 1
        key = cv2.waitKey(1)
        if key == 27:  # ESC to quit
            break
    
    # plot_3d_kps(pts_nwu, ax=ax, pts_color='green', label=f'vo{k}', 
    #             title=f'VO Output Frame {k}', plt_show=True, write_id=True, cla=True)
    plot_3d_kps(pts_vo, ax=ax, pts_color='green', label=f'vo{k}', 
                title=f'VO Output Frame {k}', plt_show=True, write_id=True, cla=True)
    
    img = render_depth_circles(uvs, pts_vo, 440, 330, 5)
    # cv2.imshow("img", img)
    cv2.imwrite("vo_out.png", img)

    import json

    # Convert arrays to lists with IDs
    output_vo = pts_vo.tolist() if pts_vo is not None else []
    output_nwu = pts_nwu.tolist() if 'pts_nwu' in locals() else []

    # Wrap into a dictionary
    output_dict = {
        "pts_vo": output_vo,     # shape: (N, 4) → [[id, x, y, z], ...]
        "pts_nwu": output_nwu    # shape: (N, 4) → [[id, x, y, z], ...]
    }

    # Save to JSON
    with open("vo_output.json", "w") as f_out:
        json.dump(output_dict, f_out, indent=2)
    print("Saved VO output to vo_output.json")
    # cv2.waitKey()

    # cv2.destroyAllWindows()
