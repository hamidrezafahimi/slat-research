import argparse
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# --- local lib paths / helpers ---
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../../../lib"))
from geom.surfaces import bspline_surface_mesh_from_ctrl, infer_grid
from utils.o3dviz import make_lineset_for_pairs, estimate_marker_radius
from diffusion.helper import reorder_ctrl_points_rowmajor, score_based_downsample
from diffusion.scoring import *


# =========================
# Utilities
# =========================


def visualize_score_heatmap(
    ext_pts: np.ndarray,
    scores01: np.ndarray,
    title: str,
    cmap: str = "viridis",
    invert_for_jet: bool = False
):
    """
    Single visualization: color points by a score in [0,1].
    - If using 'jet' and you want '1' to appear BLUE (like the old vis#2),
      pass invert_for_jet=True (because jet maps 0→blue, 1→red).
    """
    s = np.asarray(scores01, dtype=float)
    s = np.clip(s, 0.0, 1.0)
    if invert_for_jet:
        s = 1.0 - s

    # map to colors and show
    cmap_fn = getattr(plt.cm, cmap)
    colors = cmap_fn(s)[:, :3]

    pcd_col = o3d.geometry.PointCloud()
    pcd_col.points = o3d.utility.Vector3dVector(ext_pts)
    pcd_col.colors = o3d.utility.Vector3dVector(colors.astype(float))

    try:
        o3d.visualization.draw_geometries([pcd_col], window_name=title, point_show_normal=False)
    except AttributeError:
        o3d.visualization.draw([pcd_col], title=title)



# =========================
# CLI entry
# =========================

def main():
    ap = argparse.ArgumentParser(
        description="Compute NDF, Smoothness (adaptive R), and Compound scores for a PCD projected onto a B-spline surface."
    )
    ap.add_argument("--pcd", required=True, type=Path, help="Path to input .pcd file")
    ap.add_argument("--ctrl", required=True, type=Path, help="Path to control points CSV (x,y,z)")
    ap.add_argument("--su", type=int, default=100, help="Surface samples along U")
    ap.add_argument("--sv", type=int, default=100, help="Surface samples along V")
    ap.add_argument("--voxel", type=float, default=0.0, help="Optional voxel size to downsample input PCD")
    ap.add_argument("--M", type=int, default=10, help="Required neighbors inside adaptive R for smoothness")
    ap.add_argument("--K", type=int, default=10, help="Target avg neighbors to size adaptive R")
    args = ap.parse_args()

    # Load / prep PCD
    print(f"Reading PCD: {args.pcd}")
    pcd = o3d.io.read_point_cloud(str(args.pcd))
    if pcd.is_empty():
        print("ERROR: Input point cloud is empty or failed to load.", file=sys.stderr)
        sys.exit(1)
    if args.voxel and args.voxel > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(args.voxel))

    # Run main pipeline
    print("Computing scores (NDF, Smoothness, Compound)...")
    ext_pts = np.asarray(pcd.points, dtype=float)

    # Load & prep control points
    ctrl_raw = np.loadtxt(args.ctrl, delimiter=",", dtype=float)
    gw, gh = infer_grid(ctrl_raw)
    ctrl_sorted = reorder_ctrl_points_rowmajor(ctrl_raw)

    # Build mesh & project
    mesh = bspline_surface_mesh_from_ctrl(ctrl_sorted, gw, gh, args.su, args.sv)
    compound, smoothness, ndf_score, mesh, proj_pts = associate_compound_score(
        ext_pts=ext_pts,
        mesh=mesh,
        K_for_radius=args.K
    )

    ext_pts = np.asarray(pcd.points, dtype=float)

    # --- Viewer #1: Geometry and projection ---
    print("Viewer #1: all geometry (including grey spheres at spline projections). Close (press 'q') to open Viewer #2...")
    grey_spheres = []
    r = estimate_marker_radius(mesh, ext_pts)  # Estimate marker radius based on scene scale
    ctrl_pts = np.loadtxt(args.ctrl, delimiter=",", dtype=float)
    ctrl_pts_sorted = reorder_ctrl_points_rowmajor(ctrl_pts)
    for p in ctrl_pts_sorted:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=r)
        sphere.paint_uniform_color([0.5, 0.5, 0.5])  # Grey color for the spheres
        sphere.translate(p, relative=False)
        sphere.compute_vertex_normals()
        grey_spheres.append(sphere)

    lines_above = make_lineset_for_pairs(ext_pts, proj_pts, color=(1.0, 0.5, 0.0))  # orange
    lines_below = make_lineset_for_pairs(ext_pts, proj_pts, color=(0.0, 1.0, 0.0))  # green

    try:
        o3d.visualization.draw_geometries([mesh, pcd, *grey_spheres, lines_above, lines_below],
                                          window_name="PCD → B-spline projection (Viewer #1)",
                                          point_show_normal=False)
    except AttributeError:
        o3d.visualization.draw([mesh, pcd, *grey_spheres, lines_above, lines_below],
                               title="PCD → B-spline projection (Viewer #1)")

    # --- Viewer #2: NDF heatmap ---
    print("Viewer #2: NDF Heatmap...")
    print(ext_pts.shape[0])
    fasd, sc, _ = score_based_downsample(ext_pts.copy(), ndf_score)
    print(fasd.shape[0])
    visualize_score_heatmap(
        # ext_pts=ext_pts,
        # scores01=ndf_score,
        ext_pts=fasd,
        scores01=sc,
        title="Visualization #2 — NDF Score (1=blue, 0=red)",
        cmap="jet",
        invert_for_jet=True
    )

    # --- Viewer #3: Smoothness heatmap ---
    print("Viewer #3: Smoothness Heatmap...")
    visualize_score_heatmap(
        ext_pts=ext_pts,
        scores01=smoothness,
        title="Visualization #3 — Smoothness Score (1=smoothest)",
        cmap="viridis",
        invert_for_jet=False
    )

    # --- Viewer #4: Compound heatmap ---
    print("Viewer #4: Compound Heatmap...")
    visualize_score_heatmap(
        ext_pts=ext_pts,
        scores01=compound,
        title="Visualization #4 — Compound Score (smoothness × ndf score)",
        cmap="viridis",
        invert_for_jet=False
    )


if __name__ == "__main__":
    main()