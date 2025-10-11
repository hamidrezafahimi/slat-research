#!/usr/bin/env python3
import open3d as o3d
import numpy as np
import argparse

def create_xy_grid(size=1.0, step=0.1):
    """Return an XY grid (LineSet) centered at the origin."""
    xs = np.arange(-size, size + step, step)
    ys = np.arange(-size, size + step, step)
    points, lines = [], []

    for i, x in enumerate(xs):
        points.append([x, -size, 0])
        points.append([x, size, 0])
        lines.append([2 * i, 2 * i + 1])

    offset = len(points)
    for j, y in enumerate(ys):
        points.append([-size, y, 0])
        points.append([ size, y, 0])
        lines.append([offset + 2 * j, offset + 2 * j + 1])

    grid = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    grid.colors = o3d.utility.Vector3dVector([[0.7, 0.7, 0.7] for _ in lines])
    return grid

def show_point_clouds(pcd_paths):
    pcds = []
    colors = [
        [0.2, 0.6, 0.9],  # bluish
        [0.9, 0.2, 0.2],  # reddish
    ]
    for i, path in enumerate(pcd_paths):
        pcd = o3d.io.read_point_cloud(path)
        if len(pcd.points) == 0:
            raise RuntimeError(f"Point cloud is empty: {path}")
        col = colors[i % len(colors)]
        pcd.paint_uniform_color(col)
        pcds.append(pcd)

    # Estimate grid size from first cloud
    aabb = pcds[0].get_axis_aligned_bounding_box()
    extent = np.linalg.norm(aabb.get_extent())
    grid = create_xy_grid(size=extent * 0.6, step=extent * 0.1)

    o3d.visualization.draw_geometries(pcds + [grid],
                                      window_name="Point Clouds with XY plane")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show one or two point clouds with XY plane grid.")
    parser.add_argument("pcd_files", nargs="+", help="Path(s) to .pcd/.ply/.xyz file(s). One or two allowed.")
    args = parser.parse_args()

    if not (1 <= len(args.pcd_files) <= 2):
        raise ValueError("Please specify one or two point cloud files.")

    show_point_clouds(args.pcd_files)
