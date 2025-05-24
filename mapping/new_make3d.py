import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def convert_depthImage_to_pointCloud(depth_image, horizontal_fov_deg, pitch_deg, max_dist_m):
    H, W = depth_image.shape
    hfov_rad = np.radians(horizontal_fov_deg)

    # Intrinsics
    focal_length = (W / 2) / np.tan(hfov_rad / 2)
    cx, cy = W / 2, H / 2
    x_grid, y_grid = np.meshgrid(np.arange(W), np.arange(H))
    X = (x_grid - cx) / focal_length
    Y = (y_grid - cy) / focal_length
    Z = np.ones_like(X)

    # Normalize
    norm = np.sqrt(X**2 + Y**2 + Z**2)
    unit_vectors = np.stack((X / norm, Y / norm, Z / norm), axis=-1)

    # Depth values: all 255 → 0 meters
    depth_meters = (255 - depth_image.astype(np.float32)) * (max_dist_m / 255.0)
    point_cloud = unit_vectors * depth_meters[..., np.newaxis]

    # Rotate according to pitch (around X-axis)
    theta = np.radians(-pitch_deg)  # aircraft convention
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)]
    ], dtype=np.float32)

    rotated_points = point_cloud.reshape(-1, 3) @ Rx.T
    return rotated_points.reshape(H, W, 3)


# === Test ===
if __name__ == "__main__":
    # All-zero image simulates all points at max_dist
    H, W = 100, 100
    depth_image = np.zeros((H, W), dtype=np.uint8)  # All pixel values = 0
    hfov = 66
    pitch = -90
    max_dist = 28

    point_cloud = convert_depthImage_to_pointCloud(depth_image, hfov, pitch, max_dist)

    # Extract Z values and check
    Z = point_cloud[..., 2]
    print("Z range:", Z.min(), "to", Z.max())