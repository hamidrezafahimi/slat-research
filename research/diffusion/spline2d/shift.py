#!/usr/bin/env python3
"""
refactor_bspline_shift.py

Refactor of the original script into a single computation function that
takes input paths + parameters and returns the shifted control points.
Mesh construction and visualization are done outside the function.

Dependencies / assumptions:
 - numpy as np
 - open3d as o3d
 - helper functions available in the same module or imported:
     infer_grid(ctrl_pts)
     reorder_ctrl_points_rowmajor(ctrl_pts)
     bspline_surface_mesh_from_ctrl(ctrl_pts_rowmajor, gw, gh, su, sv)
     project_external_along_normals_noreject(ext_pts, mesh)
 These helpers are assumed to behave the same as in your original code.
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d

import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../../lib")
# --- IMPORT or define your helper functions here ---
from diffusion.helper import compute_shifted_ctrl_points

# -------------------------
# Example usage (CLI)
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Compute shifted spline control points.")
    ap.add_argument("--pcd", required=True, type=Path, help="Path to input .pcd file")
    ap.add_argument("--ctrl", required=True, type=Path, help="Path to control points CSV (x,y,z)")
    ap.add_argument("--su", type=int, default=100, help="Surface samples along U (columns / width)")
    ap.add_argument("--sv", type=int, default=100, help="Surface samples along V (rows / height)")
    ap.add_argument("--voxel", type=float, default=0.0, help="Optional voxel size to downsample input PCD")
    ap.add_argument("--k", type=float, default=1.1, help="Coef K (used only for mesh translate)")
    ap.add_argument("--out-csv", required=True, type=Path, default=Path("shifted_spline_mesh.csv"),
                    help="Output CSV path for shifted spline control points (x,y,z)")
    args = ap.parse_args()

    # Call the refactored function
        # --- Load PCD ---
    pcd_path = Path(args.pcd)
    ctrl_csv_path = Path(args.ctrl)

    print(f"[compute_shifted_ctrl_points] Reading PCD: {pcd_path}")
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    ext_pts = np.asarray(pcd.points, dtype=float)
    ctrl_pts = np.loadtxt(ctrl_csv_path, delimiter=",", dtype=float)
    if pcd.is_empty():
        raise RuntimeError("Input point cloud is empty or failed to load.")
    shifted_ctrl_pts, mesh, shft, mesh_shifted = compute_shifted_ctrl_points(
        ext_pts=ext_pts,
        ctrl_pts=ctrl_pts,
        su=args.su,
        sv=args.sv,
        k=args.k,
    )

    # Save shifted control points CSV (same behavior as original script)
    np.savetxt(args.out_csv, shifted_ctrl_pts, fmt="%.9f", delimiter=",", comments="")
    print(f"[main] Shifted control points saved to: {args.out_csv}")

    # Load the (optionally downsampled) point cloud for visualization
    pcd = o3d.io.read_point_cloud(str(args.pcd))
    if args.voxel and args.voxel > 0.0:
        pcd = pcd.voxel_down_sample(voxel_size=float(args.voxel))
    if not pcd.has_colors():
        pcd.paint_uniform_color([0.55, 0.55, 0.55])

    print("[main] Opening viewer (press 'q' to close)...")
    try:
        o3d.visualization.draw_geometries(
            [pcd, mesh, mesh_shifted],
            window_name="PCD + Initial Spline + Shifted Spline"
        )
    except AttributeError:
        o3d.visualization.draw(
            [pcd, mesh, mesh_shifted],
            title="PCD + Initial Spline + Shifted Spline"
        )


if __name__ == "__main__":
    main()

