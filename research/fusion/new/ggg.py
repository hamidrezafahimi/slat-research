#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def pcd2pcm(pcd: o3d.geometry.PointCloud, H: int, W: int) -> np.ndarray:
    """Convert a point cloud to an H x W x 3 numpy array of colors."""
    pts = np.asarray(pcd.colors, dtype=np.float32)
    if pts.shape[0] != H * W:
        raise ValueError(f"Point cloud has {pts.shape[0]} points, "
                         f"but expected {H*W} for reshaping.")
    return pts.reshape((H, W, 3))


def main():
    ap = argparse.ArgumentParser(description="Visualize PCD colors and real XY in Open3D.")
    ap.add_argument("--pcd", required=True, help="Path to input .pcd file")
    ap.add_argument("--H", type=int, required=True, help="Image height")
    ap.add_argument("--W", type=int, required=True, help="Image width")
    ap.add_argument("--save", default=None, help="Optional: path to save RGB image")
    args = ap.parse_args()

    pcd_path = Path(args.pcd)
    if not pcd_path.exists():
        raise FileNotFoundError(f"PCD not found: {pcd_path}")

    # --- Load point cloud
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    if not pcd.has_points():
        raise ValueError("PCD has no points.")
    if not pcd.has_colors():
        raise ValueError("PCD has no per-point colors.")

    pts = np.asarray(pcd.points)
    n = pts.shape[0]
    expected = args.H * args.W
    if n != expected:
        raise ValueError(f"Expected {expected} points (H*W), got {n}.")

    # --- Make RGB image (from colors)
    rgb = pcd2pcm(pcd, args.H, args.W)

    plt.figure(figsize=(8, 8))
    plt.imshow(rgb)
    plt.title("RGB Image (from PCD colors)")
    plt.axis('off')
    plt.tight_layout()
    if args.save:
        plt.imsave(args.save, np.clip(rgb, 0, 1))
        print(f"Saved RGB image to: {args.save}")
    plt.show()

    # --- 3D visualization in Open3D
    print("Opening Open3D window for true XY layout...")
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries([pcd, frame],
                                      window_name="True XY Visualization",
                                      width=960, height=720,
                                      point_show_normal=False)


if __name__ == "__main__":
    main()
