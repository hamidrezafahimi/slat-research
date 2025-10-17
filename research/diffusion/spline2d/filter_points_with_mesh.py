"""
Filter points by their normal-to-mesh distance (above/below) with asymmetric thresholds.

Usage (examples):
  python filter_points_by_mesh_normals.py \
      --cloud data/scene.pcd \
      --ctrl data/ctrl_pts.csv \
      --su 120 --sv 120 --k 1.1 \
      --up-offset 0.5 \
      --down-offset 0.30 \
      --save-filtered outputs/filtered_points.pcd

What gets shown:
  - Spline mesh (painted)
  - Raw point cloud (original colors if present)
  - Filtered points in red
"""
from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import open3d as o3d
# Local imports
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../../lib")
from geom.surfaces import bspline_surface_mesh_from_ctrl, \
    project_external_along_normals_noreject, filter_mesh_neighbors
from geom.grids import reorder_ctrl_points_rowmajor, infer_grid


# -----------------------------------------------------------------------------
# Helper to compute the exact boolean mask (used only for saving colors)
# -----------------------------------------------------------------------------
def _compute_filter_mask_for_colors(
    ext_pts: np.ndarray,
    ctrl_pts: np.ndarray,
    su: int,
    sv: int,
    k: float,
    up_offset: float,
    down_offset: float,
) -> np.ndarray:
    """
    Recomputes the same boolean keep mask as filter_mesh_neighbors, so we can
    slice the original per-point colors consistently. This does not change
    the public behavior of filter_mesh_neighbors (stays lean).
    """
    ext = np.asarray(ext_pts, dtype=float)[:, :3]

    ctrl = np.asarray(ctrl_pts, dtype=float)
    if ctrl.ndim == 1:
        ctrl = ctrl[None, :]
    ctrl = ctrl[:, :3]

    gw, gh = infer_grid(ctrl)
    ctrl_rowmajor = reorder_ctrl_points_rowmajor(ctrl)
    mesh = bspline_surface_mesh_from_ctrl(ctrl_rowmajor, gw, gh, su, sv)
    mesh.compute_vertex_normals()

    proj = project_external_along_normals_noreject(ext, mesh)
    dz = ext[:, 2] - proj[:, 2]
    eps = 1e-12
    below_mask = dz <= eps

    if np.any(below_mask):
        diffs_below = ext[below_mask] - proj[below_mask]
        disp = float(np.linalg.norm(diffs_below, axis=1).max(initial=0.0))
    else:
        disp = 0.0

    shft = float(k) * disp

    diffs_all = ext - proj
    lens_all = np.linalg.norm(diffs_all, axis=1)
    above_mask = dz > eps

    thr_above = up_offset * shft
    thr_below = down_offset

    keep_above = above_mask & (lens_all < thr_above)
    keep_below = (~above_mask) & (lens_all < thr_below)
    keep = keep_above | keep_below
    return keep


# -----------------------------------------------------------------------------
# CLI Utilities (I/O + Visualization)
# -----------------------------------------------------------------------------
def _read_ctrl_csv(path: str) -> np.ndarray:
    """
    Reads control points CSV with at least 3 columns (x,y,z).
    Accepts header/no-header; extra columns are ignored.
    """
    try:
        import pandas as pd
        arr = pd.read_csv(path, header=None).values
    except Exception:
        # Fallback tolerant loader
        arr = None
        for delim in [",", None, " ", "\t", ";"]:
            try:
                arr = np.genfromtxt(path, delimiter=delim)
                if arr is not None:
                    break
            except Exception:
                arr = None
        if arr is None:
            raise

    if arr.ndim == 1:
        if arr.size < 3:
            raise RuntimeError("Control CSV must have at least 3 numbers (x,y,z).")
    else:
        if arr.shape[1] < 3:
            raise RuntimeError("Control CSV must have at least 3 columns (x,y,z).")
    return arr


def _load_ext_cloud_points(path: str) -> tuple[np.ndarray, o3d.geometry.PointCloud]:
    """
    Loads an external point cloud and returns (Nx3 float array, o3d.pcd with colors if present).
    Supports: .pcd/.ply/.xyz/.xyzn/.xyzrgb, and .npy (Nx3)
    """
    ext = os.path.splitext(path.lower())[1]
    if ext == ".npy":
        pts = np.asarray(np.load(path), dtype=float)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise RuntimeError("NPY must be of shape (N,>=3).")
        pts = pts[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        # no colors available
        return pts, pcd

    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise RuntimeError(f"Failed to load point cloud: {path}")
    pts = np.asarray(pcd.points, dtype=float)
    return pts, pcd


def _ensure_colors(pcd: o3d.geometry.PointCloud, rgb=(0.6, 0.6, 0.6)) -> o3d.geometry.PointCloud:
    """
    Ensures pcd has colors; if not, paints it uniformly.
    """
    if len(pcd.colors) == 0:
        pcd.paint_uniform_color(rgb)
    return pcd


def _build_mesh_for_viz(ctrl_pts: np.ndarray, su: int, sv: int) -> o3d.geometry.TriangleMesh:
    """
    Build and return a colored mesh for visualization.
    (Yes, we build it here for viz even though the filter function builds its own mesh internally.)
    """
    ctrl = np.asarray(ctrl_pts, dtype=float)
    if ctrl.ndim == 1:
        ctrl = ctrl[None, :]
    ctrl = ctrl[:, :3]

    gw, gh = infer_grid(ctrl)
    ctrl_rowmajor = reorder_ctrl_points_rowmajor(ctrl)
    mesh = bspline_surface_mesh_from_ctrl(ctrl_rowmajor, gw, gh, su, sv)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.20, 0.70, 1.00])  # cyan-ish
    return mesh


def visualize(mesh: o3d.geometry.TriangleMesh,
              raw_pcd: o3d.geometry.PointCloud,
              filtered_pts: np.ndarray,
              point_size: int = 3):
    """
    Show mesh + raw cloud (original colors) + filtered points (red).
    """
    pcd_flt = o3d.geometry.PointCloud()
    if filtered_pts.size > 0:
        pcd_flt.points = o3d.utility.Vector3dVector(filtered_pts)
        pcd_flt.paint_uniform_color([1.0, 0.0, 0.0])  # red

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Mesh + Raw Cloud + Filtered Points", width=1440, height=900)
    vis.add_geometry(mesh)
    vis.add_geometry(frame)

    _ensure_colors(raw_pcd)
    vis.add_geometry(raw_pcd)

    if filtered_pts.size > 0:
        vis.add_geometry(pcd_flt)

    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    opt.mesh_show_back_face = True
    opt.mesh_color_option = o3d.visualization.MeshColorOption.Color

    vis.run()
    vis.destroy_window()


# -----------------------------------------------------------------------------
# Main (CLI)
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Filter points by normal-to-mesh distance (asymmetric above/below) and visualize."
    )
    ap.add_argument("--cloud", required=True, help="Input point cloud (.pcd/.ply/.xyz/… or .npy Nx3).")
    ap.add_argument("--ctrl", required=True, help="Control points CSV (>=3 columns: x,y,z).")
    ap.add_argument("--su", type=int, default=100, help="Spline subdivisions along U.")
    ap.add_argument("--sv", type=int, default=100, help="Spline subdivisions along V.")
    ap.add_argument("--up-offset", type=float, default=0.5,
                    help="ABOVE threshold scales with shft: keep if dist < up_offset * shft.")
    ap.add_argument("--down-offset", type=float, default=0.30,
                    help="BELOW absolute threshold: keep if dist < down_offset.")
    ap.add_argument("--save-filtered", type=str, default="",
                    help="Optional path to save filtered points as PCD (with original colors if available).")
    ap.add_argument("--point-size", type=int, default=3, help="Point size for visualization.")
    args = ap.parse_args()

    # Load inputs
    ctrl_pts = _read_ctrl_csv(args.ctrl)
    ext_pts, raw_pcd = _load_ext_cloud_points(args.cloud)

    # Run filtering (function returns only filtered points)
    filtered_pts = filter_mesh_neighbors(
        ext_pts=ext_pts,
        ctrl_pts=ctrl_pts,
        su=args.su,
        sv=args.sv,
        up_offset=args.up_offset,
        down_offset=args.down_offset,
    )

    # Report counts
    print(f"[filter] kept {filtered_pts.shape[0]} / {ext_pts.shape[0]} points")

    # Save filtered as colored PCD (colors copied from original input cloud if present)
    if args.save_filtered:
        pcd_out = o3d.geometry.PointCloud()
        if filtered_pts.size > 0:
            pcd_out.points = o3d.utility.Vector3dVector(filtered_pts)

            # If original cloud had colors, recompute the keep mask and slice colors accordingly.
            if len(raw_pcd.colors) > 0:
                keep = _compute_filter_mask_for_colors(
                    ext_pts=ext_pts,
                    ctrl_pts=ctrl_pts,
                    su=args.su,
                    sv=args.sv,
                    k=args.k,
                    up_offset=args.up_offset,
                    down_offset=args.down_offset,
                )
                cols = np.asarray(raw_pcd.colors, dtype=float)
                if cols.shape[0] != ext_pts.shape[0]:
                    # Safety: shapes must match; fall back to uniform if mismatch.
                    print("[warn] color/points count mismatch; saving uniform gray for filtered points.")
                    pcd_out.paint_uniform_color([0.6, 0.6, 0.6])
                else:
                    pcd_out.colors = o3d.utility.Vector3dVector(cols[keep])
            else:
                # Input had no colors (e.g., .npy). Use a neutral gray.
                pcd_out.paint_uniform_color([0.6, 0.6, 0.6])

        o3d.io.write_point_cloud(args.save_filtered, pcd_out, write_ascii=False, compressed=True)
        print(f"[io] Wrote filtered points (+colors if available) -> {args.save_filtered}")

    # Build mesh (for visualization)
    mesh = _build_mesh_for_viz(ctrl_pts, args.su, args.sv)

    # Visualize
    visualize(mesh, raw_pcd, filtered_pts, point_size=args.point_size)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
