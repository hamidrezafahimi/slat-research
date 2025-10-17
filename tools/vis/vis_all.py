#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d

def load_ctrl_xyz(csv_path: Path) -> np.ndarray:
    arr = np.loadtxt(str(csv_path), delimiter=",")
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] == 2:
        arr = np.c_[arr[:, 0], arr[:, 1], np.zeros(len(arr))]
    elif arr.shape[1] >= 3:
        arr = arr[:, :3]
    else:
        raise ValueError("Control CSV must have 2 or 3 columns (x,y or x,y,z).")
    return arr.astype(float)

def make_ctrl_lines(ctrl_xyz: np.ndarray, grid_w: int, grid_h: int) -> o3d.geometry.LineSet:
    if grid_w * grid_h != len(ctrl_xyz):
        raise ValueError("grid dims do not match number of control points")
    lines = []
    for j in range(grid_h):
        for i in range(grid_w):
            idx = j * grid_w + i
            if i + 1 < grid_w:
                lines.append([idx, idx + 1])
            if j + 1 < grid_h:
                lines.append([idx, idx + grid_w])
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(ctrl_xyz)
    ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(
        np.tile(np.array([[0.0, 0.9, 0.0]]), (len(lines), 1))
    )
    return ls

def coord_frame_for(geoms: list[o3d.geometry.Geometry]) -> o3d.geometry.TriangleMesh:
    """Union AABB to scale a coordinate frame (no '+' on AABBs)."""
    mins, maxs = [], []
    for g in geoms:
        try:
            bb = g.get_axis_aligned_bounding_box()
            mn = np.asarray(bb.get_min_bound(), dtype=float)
            mx = np.asarray(bb.get_max_bound(), dtype=float)
            if np.all(np.isfinite(mn)) and np.all(np.isfinite(mx)):
                mins.append(mn); maxs.append(mx)
        except Exception:
            pass
    if not mins:
        size = 1.0
    else:
        mn = np.min(np.vstack(mins), axis=0)
        mx = np.max(np.vstack(maxs), axis=0)
        extent = float(np.linalg.norm(mx - mn))
        size = max(0.15 * extent, 1e-3)
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)

def show_legacy(geoms, point_size: float = 2.0, bg_black: bool = True, screenshot: Path | None = None):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PCD + Mesh + Control", width=1280, height=800, visible=True)
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    opt.background_color = np.array([0, 0, 0]) if bg_black else np.array([1, 1, 1])
    vis.poll_events()
    vis.update_renderer()
    if screenshot is not None:
        vis.capture_screen_image(str(screenshot), do_render=True)
        print(f"[ok] Saved screenshot -> {screenshot}")
    vis.run()
    vis.destroy_window()

def show_draw_api(pcd, mesh, ctrl_pcd, ctrl_lines, axes, point_size: float, bg_black: bool):
    """Try modern draw() API (Open3D >= 0.17)."""
    bg = [0, 0, 0, 1] if bg_black else [1, 1, 1, 1]
    items = [
        {"name": "pcd",  "geometry": pcd,      "material": {"point_size": float(point_size)}},
        {"name": "mesh", "geometry": mesh},
        {"name": "ctrl", "geometry": ctrl_pcd, "material": {"point_size": float(point_size)}},
        {"name": "axes", "geometry": axes},
    ]
    if ctrl_lines is not None:
        items.append({"name": "ctrl_grid", "geometry": ctrl_lines})
    # If your Open3D doesn't support dict items, this will raise — caller will catch.
    o3d.visualization.draw(items, show_skybox=False, bg_color=bg, title="PCD + Mesh + Control")

def main():
    ap = argparse.ArgumentParser(description="Visualize original PCD + fitted mesh + control net")
    ap.add_argument("--pcd", type=Path, required=True, help="Path to original point cloud (.pcd/.ply/.xyz/...)")
    ap.add_argument("--mesh", type=Path, required=True, help="Path to fitted mesh (.ply/.obj)")
    ap.add_argument("--ctrl", type=Path, required=True, help="Path to control points CSV (x,y[,z])")
    ap.add_argument("--grid-w", type=int, default=None, help="Control grid width (columns, u)")
    ap.add_argument("--grid-h", type=int, default=None, help="Control grid height (rows, v)")
    ap.add_argument("--point-size", type=float, default=2.5, help="Point size for PCD & control net")
    ap.add_argument("--bg", choices=["black", "white"], default="black", help="Background color")
    ap.add_argument("--screenshot", type=Path, default=None, help="Optional path to save a screenshot (legacy mode)")
    ap.add_argument("--use-draw", action="store_true", help="Force modern draw() API; otherwise use legacy Visualizer")
    args = ap.parse_args()

    # Load
    pcd = o3d.io.read_point_cloud(str(args.pcd))
    if pcd.is_empty(): raise RuntimeError(f"Failed to read point cloud: {args.pcd}")
    mesh = o3d.io.read_triangle_mesh(str(args.mesh))
    if mesh.is_empty(): raise RuntimeError(f"Failed to read mesh: {args.mesh}")
    if not mesh.has_vertex_normals(): mesh.compute_vertex_normals()
    ctrl_xyz = load_ctrl_xyz(args.ctrl)
    ctrl_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ctrl_xyz))

    # Colors
    pcd.paint_uniform_color([0.2, 0.7, 1.0])
    mesh.paint_uniform_color([0.95, 0.55, 0.2])
    ctrl_pcd.paint_uniform_color([0.0, 0.9, 0.0])

    ctrl_lines = None
    if args.grid_w is not None and args.grid_h is not None:
        ctrl_lines = make_ctrl_lines(ctrl_xyz, args.grid_w, args.grid_h)

    axes = coord_frame_for([pcd, mesh, ctrl_pcd] + ([ctrl_lines] if ctrl_lines is not None else []))
    bg_black = (args.bg == "black")

    if args.use_draw:
        try:
            show_draw_api(pcd, mesh, ctrl_pcd, ctrl_lines, axes, args.point_size, bg_black)
            return
        except Exception as e:
            print(f"[warn] draw() API failed ({e}). Falling back to legacy Visualizer.")

    # Legacy (works on all versions)
    geoms = [pcd, mesh, ctrl_pcd, axes] + ([ctrl_lines] if ctrl_lines is not None else [])
    show_legacy(geoms, point_size=args.point_size, bg_black=bg_black, screenshot=args.screenshot)

if __name__ == "__main__":
    main()
