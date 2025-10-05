#!/usr/bin/env python3
"""
PCD → B-spline surface projection visualizer (three viewers).

Usage:
  python pcd_to_bspline_projection.py \
      --pcd path/to/cloud.pcd \
      --ctrl path/to/ctrl_points.csv \
      --su 100 --sv 100 \
      --k 3 \
      [--voxel 0.0] [--max-lines 0] [--output path/to/output.pcd]

Notes:
- su × sv controls the sampling resolution of the surface mesh.
- k is kept for compatibility with the old IDW method (unused by normal-ray projection).
- If --voxel > 0, the input PCD is voxel-downsampled to that size (meters).
- If --max-lines > 0, at most that many random point-pairs will be drawn in total.
- Color convention: ORANGE = above (dz>0), GREEN = below (dz≤0).
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
import sys
import random
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../../lib")
from geom.surfaces import bspline_surface_mesh_from_ctrl, project_external_along_normals_noreject,\
    infer_grid
from utils.o3dviz import estimate_marker_radius, make_lineset_for_pairs
from diffusion.helper import reorder_ctrl_points_rowmajor
from helper import *

def main():
    ap = argparse.ArgumentParser(description="Project PCD points onto a B-spline surface defined by control points.")
    ap.add_argument("--pcd", required=True, type=Path, help="Path to input .pcd file")
    ap.add_argument("--ctrl", required=True, type=Path, help="Path to control points CSV (x,y,z)")
    ap.add_argument("--su", type=int, default=100, help="Surface samples along U (columns / width)")
    ap.add_argument("--sv", type=int, default=100, help="Surface samples along V (rows / height)")
    ap.add_argument("--k", type=int, default=3, help="k-NN for IDW projection (unused for normal-ray method)")
    ap.add_argument("--voxel", type=float, default=0.0, help="Optional voxel size to downsample input PCD")
    ap.add_argument("--max-lines", type=int, default=0, help="Global cap on number of connecting lines (0 = all)")
    ap.add_argument("--output", type=Path, help="Output path to save projected points as a PCD file")
    args = ap.parse_args()

    # --- Load data ---
    print(f"Reading PCD: {args.pcd}")
    pcd = o3d.io.read_point_cloud(str(args.pcd))
    if pcd.is_empty():
        print("ERROR: Input point cloud is empty or failed to load.", file=sys.stderr)
        sys.exit(1)
    if args.voxel and args.voxel > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(args.voxel))

    ext_pts = np.asarray(pcd.points, dtype=float)
    print(f"Input PCD points: {ext_pts.shape[0]}")

    print(f"Reading control CSV: {args.ctrl}")
    ctrl_pts = np.loadtxt(args.ctrl, delimiter=",", dtype=float)

    # --- Infer grid dimensions and reorder to row-major by (y,x) ---
    gw, gh = infer_grid(ctrl_pts)
    print(f"Inferred control grid: gw={gw}, gh={gh}  (total={ctrl_pts.shape[0]})")
    ctrl_pts_sorted = reorder_ctrl_points_rowmajor(ctrl_pts)

    # --- Build the B-spline surface mesh ---
    print(f"Building B-spline surface mesh with su={args.su}, sv={args.sv} ...")
    mesh = bspline_surface_mesh_from_ctrl(ctrl_pts_sorted, gw, gh, args.su, args.sv)
    mesh.compute_vertex_normals()  # ensure normals exist

    # --- Project points onto the surface along local normals ---
    print(f"Projecting {ext_pts.shape[0]} points along mesh normals (ray casting ±n)...")
    proj = project_external_along_normals_noreject(ext_pts, mesh)
    # proj, idx = project_external_along_normals_andreject(ext_pts, mesh)
    # proj = projp[idx]
    # src_valid = ext_pts[idx]

    # --- Add grey spheres on spline projection points ---
    grey_spheres = []
    r = estimate_marker_radius(mesh, ext_pts)  # Estimate marker radius based on scene scale
    for p in ctrl_pts_sorted:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=r)
        sphere.paint_uniform_color([0.5, 0.5, 0.5])  # Grey color for the spheres
        sphere.translate(p, relative=False)
        sphere.compute_vertex_normals()
        grey_spheres.append(sphere)

    # --- Build visuals for Viewer #1 (everything) ---
    pcd_in_all = o3d.geometry.PointCloud()
    pcd_in_all.points = o3d.utility.Vector3dVector(ext_pts)
    pcd_in_all.colors = pcd.colors  # Keep original colors of the point cloud

    pcd_proj_all = o3d.geometry.PointCloud()
    pcd_proj_all.points = o3d.utility.Vector3dVector(proj)
    pcd_proj_all.paint_uniform_color([1.0, 0.0, 0.0])

    mesh.paint_uniform_color([0.2, 0.7, 1.0])

    # Lines: ORANGE for above, GREEN for below
    # lines_above = make_lineset_for_pairs(src_valid, proj, color=(1.0, 0.5, 0.0))  # orange
    # lines_below = make_lineset_for_pairs(src_valid, proj, color=(0.0, 1.0, 0.0))  # green
    lines_above = make_lineset_for_pairs(ext_pts, proj, color=(1.0, 0.5, 0.0))  # orange
    lines_below = make_lineset_for_pairs(ext_pts, proj, color=(0.0, 1.0, 0.0))  # green

    # --- Viewer #1 ---
    print("Viewer #1: all geometry (including grey spheres at spline projections). Close (press 'q') to open Viewer #2...")
    try:
        o3d.visualization.draw_geometries([mesh, pcd_in_all, pcd_proj_all, lines_above, lines_below, *grey_spheres],
                                          window_name="PCD → B-spline projection (Viewer #1)",
                                          point_show_normal=False)
    except AttributeError:
        o3d.visualization.draw([mesh, pcd_in_all, pcd_proj_all, lines_above, lines_below, *grey_spheres],
                               title="PCD → B-spline projection (Viewer #1)")

    # Save projected points to PCD if the --output argument is provided
    if args.output:
        print(f"Saving projected points to {args.output}")
        pcd_proj = o3d.geometry.PointCloud()
        pcd_proj.points = o3d.utility.Vector3dVector(proj)
        o3d.io.write_point_cloud(str(args.output), pcd_proj)
        print("Projected points saved successfully.")

if __name__ == "__main__":
    main()
