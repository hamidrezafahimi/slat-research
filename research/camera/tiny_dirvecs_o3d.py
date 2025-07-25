import numpy as np
import open3d as o3d

def img2dirVecsCam(output_shape, hfov_degs):
    H, W = output_shape
    aspect_ratio = H / W
    hfov_rad = np.deg2rad(hfov_degs)
    vfov_rad = 2 * np.arctan(np.tan(hfov_rad / 2) * aspect_ratio)
    x_angles = np.linspace(-hfov_rad / 2, hfov_rad / 2, W)
    y_angles = np.linspace(-vfov_rad / 2, vfov_rad / 2, H)
    theta, phi = np.meshgrid(x_angles, y_angles)
    x = np.tan(theta)
    y = np.tan(phi)
    z = np.ones_like(x)
    dirs1 = np.stack((x, y, z), axis=-1)
    norms = np.linalg.norm(dirs1, axis=-1, keepdims=True)
    return dirs1 / norms  # shape (H, W, 3)

# Visualization using Open3D
def visualize_rays_open3d(dirs, scale=1.0):
    H, W, _ = dirs.shape
    origin = np.array([[0, 0, 0]])

    points = []
    lines = []
    colors = []

    idx = 0
    for i in range(H):
        for j in range(W):
            d = dirs[i, j] * scale
            points.append(origin[0])
            points.append(d)
            lines.append([idx, idx + 1])
            colors.append([0, 0, 1])  # Blue
            idx += 2

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([line_set],
        zoom=0.7,
        front=[0.0, 0.0, -1.0],
        lookat=[0, 0, 1],
        up=[0, -1, 0]
    )

# Run the test
hfov = 90
output_shape = (3, 3)
dirs = img2dirVecsCam(output_shape, hfov)
visualize_rays_open3d(dirs, scale=1.0)
