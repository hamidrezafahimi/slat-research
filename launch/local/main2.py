import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator

def infer_grid_shape(points):
    x_unique = np.unique(points[:, 0])
    y_unique = np.unique(points[:, 1])
    W = x_unique.shape[0]
    H = y_unique.shape[0]
    assert H * W == points.shape[0], "Point cloud is not a full grid"
    return H, W

def calc_distance(pc_up: np.ndarray, pc_down: np.ndarray) -> np.ndarray:
    H, W, _ = pc_up.shape
    up_flat = pc_up.reshape(-1, 3)
    down_flat = pc_down.reshape(-1, 3)

    xy_down = down_flat[:, :2]
    z_down = down_flat[:, 2]
    interp = LinearNDInterpolator(xy_down, z_down, fill_value=np.nan)

    projected_z = np.empty_like(up_flat[:, 2])
    for i, (x, y, _) in enumerate(up_flat):
        z_proj = interp(x, y)
        if np.isnan(z_proj):
            dists = np.linalg.norm(xy_down - [x, y], axis=1)
            z_proj = z_down[np.argmin(dists)]
        projected_z[i] = z_proj

    vertical_distances = up_flat[:, 2] - projected_z
    return vertical_distances.reshape((H, W, 1))

# ---------- Main Script ----------

# Load point clouds
pcd_up = o3d.io.read_point_cloud("inclined_plane.pcd")
pcd_down = o3d.io.read_point_cloud("flat_surface.pcd")

# Convert to NumPy arrays
points_up = np.asarray(pcd_up.points)
points_down = np.asarray(pcd_down.points)

# Infer grid shapes
H_up, W_up = infer_grid_shape(points_up)
H_down, W_down = infer_grid_shape(points_down)

# Determine crop shape
H = min(H_up, H_down)
W = min(W_up, W_down)
print(f"Using cropped shape: H={H}, W={W}")

# Crop and reshape both clouds
pc_up_grid = points_up[:H*W].reshape((H, W, 3))
pc_down_grid = points_down[:H*W].reshape((H, W, 3))

# Compute vertical distances
distance_matrix = calc_distance(pc_up_grid, pc_down_grid)  # Shape: (H, W, 1)

height, width, _ = distance_matrix.shape
xx, yy = np.meshgrid(np.arange(width), np.arange(height))

# Flatten the grid and distance values
points = np.vstack((xx.flatten(), yy.flatten(), distance_matrix.flatten())).T

# Normalize the distances for color mapping
norm_distances = (distance_matrix.flatten() - np.min(distance_matrix.flatten())) / (np.max(distance_matrix.flatten()) - np.min(distance_matrix.flatten()))

# Create a color map based on the normalized distances
colors = plt.cm.viridis(norm_distances)[:, :3]  # Extract RGB values from the colormap

# Create a point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Visualize the point cloud using Open3D
o3d.visualization.draw_geometries([pcd], window_name="Vertical Distance Heatmap")

# Optionally, you can save the point cloud as a PLY file if needed
o3d.io.write_point_cloud("distance_heatmap.ply", pcd)