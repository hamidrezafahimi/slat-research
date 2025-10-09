#!/usr/bin/env python3
import argparse
import open3d as o3d
import numpy as np

def downsample_pcd(pcd: o3d.geometry.PointCloud, num_dst: int) -> o3d.geometry.PointCloud:
    """
    Downsample the point cloud to a target number of points.

    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud.
        num_dst (int): Target number of points.

    Returns:
        o3d.geometry.PointCloud: Downsampled point cloud.
    """
    cloud_pts = np.asarray(pcd.points, dtype=float)
    num_orig = float(cloud_pts.shape[0])
    num_dst = float(num_dst)

    if num_dst < num_orig:
        ratio = num_dst / num_orig
    else:
        ratio = 1.0

    num_points = cloud_pts.shape[0]
    new_num_points = int(num_points * ratio)

    indices = np.random.choice(num_points, new_num_points, replace=False)
    cloud_pts_downsampled = cloud_pts[indices]

    if pcd.has_colors():
        orig_cols = np.asarray(pcd.colors, dtype=float)
        orig_cols_downsampled = orig_cols[indices]

    pcd_downsampled = o3d.geometry.PointCloud()
    pcd_downsampled.points = o3d.utility.Vector3dVector(cloud_pts_downsampled)

    if pcd.has_colors():
        pcd_downsampled.colors = o3d.utility.Vector3dVector(orig_cols_downsampled)

    return pcd_downsampled


def main():
    parser = argparse.ArgumentParser(description="Downsample a point cloud to a target number of points.")
    parser.add_argument("--in", dest="input_file", required=True, help="Input point cloud file (.pcd, .ply, etc.)")
    parser.add_argument("--out", dest="output_file", required=True, help="Output point cloud file (.pcd, .ply, etc.)")
    parser.add_argument("--target", dest="target_points", type=int, required=True, help="Target number of points")

    args = parser.parse_args()

    # Load input point cloud
    pcd = o3d.io.read_point_cloud(args.input_file)
    print(f"Loaded {len(pcd.points)} points from {args.input_file}")

    # Downsample
    pcd_down = downsample_pcd(pcd, args.target_points)
    print(f"Downsampled to {len(pcd_down.points)} points")

    # Save output
    o3d.io.write_point_cloud(args.output_file, pcd_down)
    print(f"Saved downsampled point cloud to {args.output_file}")


if __name__ == "__main__":
    main()
