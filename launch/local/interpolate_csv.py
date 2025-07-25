import os
import numpy as np
import cv2
import pandas as pd
import glob
from scipy.interpolate import griddata

# === Configuration ===
image_folder = "/home/ali/Datasets/street_kitti_1/image"
lidar_folder = "/home/ali/Datasets/street_kitti_1/pcd360"
calib_file = "/home/ali/Datasets/street_kitti_1/calib.txt"

output_csv_folder = "data/depth_csv_matrix/"
depth_pcd_folder = "data/depth_pcd/"
cam_coords_csv_folder = "data/pcd_cam_coords/"

SAVE_DEPTH_CSV = True
SAVE_DEPTH_IMAGE = True
SAVE_CAM_COORDS_CSV = True

# === Create output folders if needed ===
os.makedirs(output_csv_folder, exist_ok=True)
os.makedirs(depth_pcd_folder, exist_ok=True)
os.makedirs(cam_coords_csv_folder, exist_ok=True)

# === Load Calibration ===
def load_calib(filepath):
    calib = {}
    with open(filepath, 'r') as f:
        for line in f:
            if ":" not in line:
                continue
            key, val = line.strip().split(":", 1)
            calib[key] = np.array([float(x) for x in val.strip().split()])
    return calib

calib = load_calib(calib_file)
tr_key = "Tr_velo_to_cam" if "Tr_velo_to_cam" in calib else "Tr"
Tr = np.vstack((calib[tr_key].reshape(3, 4), [0, 0, 0, 1]))  # (4x4)
P2 = calib["P2"].reshape(3, 4)

# === Process Each Image ===
image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))

for img_path in image_paths:
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    lidar_path = os.path.join(lidar_folder, base_name + ".bin")
    output_matrix_csv = os.path.join(output_csv_folder, base_name + ".csv")
    depth_image_path = os.path.join(depth_pcd_folder, base_name + ".jpg")
    cam_coords_csv = os.path.join(cam_coords_csv_folder, base_name + ".csv")

    if not os.path.exists(lidar_path):
        print(f"Skipping {base_name}: no corresponding LiDAR file.")
        continue

    # Load LiDAR
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    points_hom = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))  # (N, 4)

    # Transform to camera coordinate
    points_cam = (Tr @ points_hom.T).T[:, :3]
    mask_front = points_cam[:, 2] > 0
    points_cam = points_cam[mask_front]

    # === Optional: Save transformed 3D points in camera coordinates ===
    if SAVE_CAM_COORDS_CSV:
        pd.DataFrame(points_cam, columns=["X_cam", "Y_cam", "Z_cam"]).to_csv(cam_coords_csv, index=False)
        print(f"📌 Saved 3D camera coordinates to {cam_coords_csv}")

    # Project to image
    proj = (P2 @ np.hstack((points_cam, np.ones((points_cam.shape[0], 1)))).T).T
    proj[:, 0] /= proj[:, 2]
    proj[:, 1] /= proj[:, 2]

    image = cv2.imread(img_path)
    if image is None:
        print(f"Failed to load image: {img_path}")
        continue
    img_h, img_w = image.shape[:2]

    u = np.round(proj[:, 0]).astype(int)
    v = np.round(proj[:, 1]).astype(int)
    z = points_cam[:, 2]

    valid = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    u = u[valid]
    v = v[valid]
    z = z[valid]

    if len(z) == 0:
        print(f"⚠️  No valid depth points for {base_name}, skipping.")
        continue

    # Interpolate raw depth to full image size
    grid_x, grid_y = np.meshgrid(np.arange(img_w), np.arange(img_h))
    known_points = np.stack([u, v], axis=-1)
    known_values = z

    interpolated = griddata(
        known_points, known_values,
        (grid_x, grid_y),
        method='linear',
        fill_value=0
    )

    # === Save interpolated depth as 2D matrix CSV ===
    if SAVE_DEPTH_CSV:
        pd.DataFrame(interpolated).to_csv(output_matrix_csv, index=False, header=False)
        print(f"📄 Saved 2D matrix depth CSV to {output_matrix_csv}")

    # === Optional visualization ===
    if SAVE_DEPTH_IMAGE:
        normalized = (interpolated - interpolated.min()) / (interpolated.max() - interpolated.min() + 1e-8)
        depth_image = np.clip(normalized * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(depth_image_path, depth_image)
        print(f"🖼️  Saved interpolated depth image to {depth_image_path}")

print("✅ All done generating matrix CSVs, images, and camera-coordinate point clouds.")
