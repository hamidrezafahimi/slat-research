#!/usr/bin/env python3
"""
projection_scorer_final_gui.py

- Reads a point cloud in main()
- generate_xy_spline(...) builds the surface at the cloud CG
- class Projection3DScorer encapsulates:
    * __init__: per-point smoothness via self_associate_smoothness()
    * score(): project to ppoints + scoreSurface() (sum of point_scores)
    * draw(): returns Open3D geometries (no materials)
- OPTIONAL: --show_gui displays with SceneWidget + PBR materials (defaultLit) for the spline
"""

import argparse
import numpy as np
import open3d as o3d

from helper import *

# ---------- GUI & materials (used only with --show_gui) ----------
try:
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering
    _GUI_OK = True
except Exception:
    _GUI_OK = False

import numpy as np
import open3d as o3d


# =================== helpers: geometry ===================
def mat_points(size=4.0):
    m = rendering.MaterialRecord()
    m.shader = "defaultUnlit"
    m.point_size = float(size)
    return m

def mat_mesh():
    m = rendering.MaterialRecord()
    m.shader = "defaultLit"
    m.base_color = (0.7, 0.7, 0.9, 1.0)
    m.base_roughness = 0.8
    return m

def visualize_with_materials(ctrl_pcd, surf_mesh, ext_pcd, proj_pcd=None):
    gui.Application.instance.initialize()
    window = gui.Application.instance.create_window("Smoothness + Surface", 1024, 768)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    window.add_child(scene)

    scene.scene.add_geometry("ctrl_pcd", ctrl_pcd, mat_points(8.0))
    scene.scene.add_geometry("surf_mesh", surf_mesh, mat_mesh())
    scene.scene.add_geometry("ext_pcd", ext_pcd, mat_points(2.0))
    if proj_pcd is not None:
        scene.scene.add_geometry("proj_pcd", proj_pcd, mat_points(2.0))

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    scene.scene.add_geometry("axis", axis, mat_mesh())

    # fit camera
    if proj_pcd is not None:
        aabbs = [g.get_axis_aligned_bounding_box() for g in [ctrl_pcd, surf_mesh, ext_pcd, proj_pcd]]
    else:
        aabbs = [g.get_axis_aligned_bounding_box() for g in [ctrl_pcd, surf_mesh, ext_pcd]]
    mins = np.min([a.min_bound for a in aabbs], axis=0)
    maxs = np.max([a.max_bound for a in aabbs], axis=0)
    center = 0.5 * (mins + maxs)
    extent = max(maxs - mins)
    eye = center + np.array([0, -3.0 * extent, 1.8 * extent])
    up = np.array([0, 0, 1])
    scene.scene.camera.look_at(center, eye, up)

    gui.Application.instance.run()

def show_gui_with_materials(geoms: list[o3d.geometry.Geometry]):
    if not _GUI_OK:
        raise RuntimeError("Open3D GUI not available. Install open3d with GUI support.")
    gui.Application.instance.initialize()
    win = gui.Application.instance.create_window("Projection3DScorer — PBR Surface", 1280, 800)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(win.renderer)
    win.add_child(scene)

    # Add geoms with appropriate materials
    for i, g in enumerate(geoms):
        name = f"g{i}"
        if isinstance(g, o3d.geometry.TriangleMesh):
            scene.scene.add_geometry(name, g, _mat_mesh())
        elif isinstance(g, o3d.geometry.PointCloud):
            scene.scene.add_geometry(name, g, _mat_points(2.0))
        else:
            scene.scene.add_geometry(name, g, _mat_mesh())

    # Camera fit
    aabbs = [g.get_axis_aligned_bounding_box() for g in geoms]
    mins = np.min([a.min_bound for a in aabbs], axis=0)
    maxs = np.max([a.max_bound for a in aabbs], axis=0)
    center = 0.5 * (mins + maxs)
    extent = max(maxs - mins)
    eye = center + np.array([0, -3.0 * extent, 1.8 * extent])
    up = np.array([0, 0, 1])
    scene.scene.camera.look_at(center, eye, up)

    gui.Application.instance.run()

# =================== main() ===================
def main():
    ap = argparse.ArgumentParser(description="Projection3DScorer (with optional PBR GUI)")
    ap.add_argument("--in", dest="inp", required=True, help="Input point cloud (.pcd/.ply/.xyz)")
    ap.add_argument("--grid_w", type=int, default=6)
    ap.add_argument("--grid_h", type=int, default=4)
    ap.add_argument("--samples_u", type=int, default=40)
    ap.add_argument("--samples_v", type=int, default=40)
    ap.add_argument("--margin", type=float, default=0.02)

    ap.add_argument("--max_dz", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=None)

    ap.add_argument("--ext_mode", choices=["black", "smoothness", "original", "null"], default="smoothness")
    ap.add_argument("--pp_mode",  choices=["black", "score", "original", "null"], default="score")
    ap.add_argument("--plot_spline", action="store_true", help="plot spline")

    ap.add_argument("--show", action="store_true", help="Classic viewer (no PBR)")
    ap.add_argument("--show_gui", action="store_true", help="SceneWidget viewer (PBR spline)")
    ap.add_argument("--save_proj", default=None)
    ap.add_argument("--save_colored", default=None)
    ap.add_argument("--spline_data", default=None,
                help="CSV of control points (x,y,z). If set, overrides auto spline generation.")

    args = ap.parse_args()

    # read cloud
    pcd = o3d.io.read_point_cloud(args.inp)
    if len(pcd.points) == 0:
        raise SystemExit("[ERROR] Empty point cloud.")
    cloud_pts = np.asarray(pcd.points, dtype=float)
    orig_cols = np.asarray(pcd.colors, dtype=float) if pcd.has_colors() else None

    # --- spline creation ---
    if args.spline_data:
        ctrl_pts = np.loadtxt(args.spline_data, delimiter=",", dtype=float)
        mesh, W, H = generate_spline(ctrl_pts, samples_u=args.samples_u, samples_v=args.samples_v)
        center_xy = np.array([ctrl_pts[:,0].mean(), ctrl_pts[:,1].mean()])
        z0 = float(ctrl_pts[:,2].mean())
    else:
        # --- Auto-generate spline as before ---
        mesh, ctrl_pts, W, H, center_xy, z0 = generate_xy_spline(
            cloud_pts, grid_w=args.grid_w, grid_h=args.grid_h,
            samples_u=args.samples_u, samples_v=args.samples_v, margin=args.margin
        )

    # scorer
    scorer = Projection3DScorer(
        cloud_pts, mesh,
        kmin_neighbors=8, neighbor_cap=64,
        max_delta_z=args.max_dz, tau=args.tau,
        original_colors=orig_cols
    )

    overall, pjs = scorer.score(mesh)
    print(f"[OVERALL] sum(point_scores) = {overall:.6f}")

    geoms = scorer.draw(ctrl_pts=ctrl_pts, pj_pts=pjs, W=W, H=H,
                        ext_mode=args.ext_mode, pp_mode=args.pp_mode)

    if args.save_colored:
        o3d.io.write_point_cloud(args.save_colored, geoms[0], write_ascii=True)
        print(f"[i] saved external colored: {args.save_colored}")
    if args.save_proj:
        o3d.io.write_point_cloud(args.save_proj, geoms[1], write_ascii=True)
        print(f"[i] saved projected colored: {args.save_proj}")

    if args.show:
        if args.plot_spline:
            visualize_with_materials(
                ctrl_pcd=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ctrl_pts)), 
                surf_mesh=mesh, 
                ext_pcd=geoms[0], 
                proj_pcd=geoms[1]
            )
        else:
            o3d.visualization.draw_geometries(geoms)

    if args.show_gui:
        show_gui_with_materials(geoms)


if __name__ == "__main__":
    main()
