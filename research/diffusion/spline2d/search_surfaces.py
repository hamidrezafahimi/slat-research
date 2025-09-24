#!/usr/bin/env python3
"""
random_spline_search.py

- Loads cloud
- Builds CG-centered XY flat surface via helper.generate_xy_spline (inside RandomSurfacer)
- RandomSurfacer.generate(N) -> try N random control grids (fixed x,y; random z in [Z_down,Z_up])
- Score each with Projection3DScorer; pick best; save CSV
- Final visualization uses Open3D GUI with materials (your style)
  * --show-sandwich overlays top/mid/bottom surfaces
"""

import argparse
import os
import sys
import csv
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

# ---- Import exactly what you asked to keep external in helper.py ----
try:
    from helper import generate_xy_spline, RandomSurfacer, Projection3DScorer
except ImportError as e:
    print("ERROR: Could not import from helper.py (need generate_xy_spline, RandomSurfacer, Projection3DScorer).", file=sys.stderr)
    raise

# =================== helpers: materials viewer (your style) ===================
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

def mat_mesh_tinted(rgba=(0.7, 0.95, 0.7, 0.5)):
    m = rendering.MaterialRecord()
    m.shader = "defaultLit"
    m.base_color = tuple(float(x) for x in rgba)
    m.base_roughness = 0.8
    return m

def _fit_camera(scene, geoms):
    aabbs = [g.get_axis_aligned_bounding_box() for g in geoms if g is not None]
    if not aabbs:
        return
    mins = np.min([a.min_bound for a in aabbs], axis=0)
    maxs = np.max([a.max_bound for a in aabbs], axis=0)
    center = 0.5 * (mins + maxs)
    extent = float(max(maxs - mins))
    if extent <= 0:
        extent = 1.0
    eye = center + np.array([0, -3.0 * extent, 1.8 * extent])
    up = np.array([0, 0, 1])
    scene.camera.look_at(center, eye, up)

def visualize_with_materials(ctrl_pcd, surf_mesh, ext_pcd, proj_pcd=None):
    gui.Application.instance.initialize()
    window = gui.Application.instance.create_window("Best Spline (Materials Viewer)", 1024, 768)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    window.add_child(scene)

    if ctrl_pcd is not None:
        scene.scene.add_geometry("ctrl_pcd", ctrl_pcd, mat_points(8.0))
    if surf_mesh is not None:
        scene.scene.add_geometry("surf_mesh", surf_mesh, mat_mesh())
    if ext_pcd is not None:
        scene.scene.add_geometry("ext_pcd", ext_pcd, mat_points(2.0))
    if proj_pcd is not None:
        scene.scene.add_geometry("proj_pcd", proj_pcd, mat_points(2.0))

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    scene.scene.add_geometry("axis", axis, mat_mesh())

    _fit_camera(scene.scene, [ctrl_pcd, surf_mesh, ext_pcd, proj_pcd, axis])
    gui.Application.instance.run()

def visualize_with_materials_sandwich(ctrl_pcd, best_mesh, ext_pcd, top_mesh, mid_mesh, bot_mesh, proj_pcd=None):
    """Same look, but overlays top/mid/bottom."""
    gui.Application.instance.initialize()
    window = gui.Application.instance.create_window("Best Spline + Sandwich (Materials Viewer)", 1280, 900)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    window.add_child(scene)

    if ctrl_pcd is not None:
        scene.scene.add_geometry("ctrl_pcd", ctrl_pcd, mat_points(8.0))
    if best_mesh is not None:
        scene.scene.add_geometry("best_mesh", best_mesh, mat_mesh())
    if ext_pcd is not None:
        scene.scene.add_geometry("ext_pcd", ext_pcd, mat_points(2.0))
    if proj_pcd is not None:
        scene.scene.add_geometry("proj_pcd", proj_pcd, mat_points(2.0))

    # top/mid/bottom, lightly tinted
    if top_mesh is not None:
        scene.scene.add_geometry("top_mesh", top_mesh, mat_mesh_tinted((0.7, 0.95, 0.7, 0.55)))
    if mid_mesh is not None:
        scene.scene.add_geometry("mid_mesh", mid_mesh, mat_mesh_tinted((0.7, 0.7, 0.9, 0.35)))
    if bot_mesh is not None:
        scene.scene.add_geometry("bottom_mesh", bot_mesh, mat_mesh_tinted((0.95, 0.7, 0.7, 0.55)))

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    scene.scene.add_geometry("axis", axis, mat_mesh())

    _fit_camera(scene.scene, [ctrl_pcd, best_mesh, ext_pcd, proj_pcd, top_mesh, mid_mesh, bot_mesh, axis])
    gui.Application.instance.run()

# =================== I/O helpers ===================
def save_ctrl_csv(path: str, ctrl_pts: np.ndarray):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for row in ctrl_pts:
            w.writerow([f"{row[0]:.9f}", f"{row[1]:.9f}", f"{row[2]:.9f}"])

# =================== main ===================
def main():
    ap = argparse.ArgumentParser(description="Random spline search around CG-centered XY base surface; score via Projection3DScorer; view with materials.")
    ap.add_argument("--cloud", required=True, help="Path to external point cloud.")
    ap.add_argument("--N", type=int, default=100, help="Number of random spline candidates.")
    ap.add_argument("--grid_w", type=int, default=6)
    ap.add_argument("--grid_h", type=int, default=4)
    ap.add_argument("--samples_u", type=int, default=40)
    ap.add_argument("--samples_v", type=int, default=40)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--kmin_neighbors", type=int, default=8)
    ap.add_argument("--neighbor_cap", type=int, default=64)
    ap.add_argument("--max_delta_z", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-best", default="best_ctrl.csv")
    ap.add_argument("--no-viz", action="store_true")
    ap.add_argument("--show-sandwich", action="store_true",
                    help="Overlay top/mid(bottom=base)/bottom surfaces in the final viewer.")
    ap.add_argument("--show-projected", action="store_true",
                    help="Also render the projected points (IDW projection onto best surface).")
    args = ap.parse_args()

    if args.seed >= 0:
        np.random.seed(args.seed)

    # RandomSurfacer builds: base_mesh, top_mesh, bottom_mesh, ctrl_pts, cloud_pts, etc.
    rs = RandomSurfacer(
        cloud=args.cloud,
        grid_w=args.grid_w,
        grid_h=args.grid_h,
        samples_u=args.samples_u,
        samples_v=args.samples_v,
        margin=args.margin,
    )

    base_mesh = rs.base_mesh
    cloud_pts = rs.cloud_pts

    p3ds = Projection3DScorer(
        cloud_pts=cloud_pts,
        spline_mesh=base_mesh,
        kmin_neighbors=args.kmin_neighbors,
        neighbor_cap=args.neighbor_cap,
        max_delta_z=args.max_delta_z,
        tau=args.tau,
        original_colors=None,
    )

    base_score, _ = p3ds.score(base_mesh)
    print(f"[info] Base flat-surface score = {base_score:.6f}")

    candidates = rs.generate(args.N)
    if not candidates:
        print("[warn] N <= 0; saving base ctrl and exiting.")
        save_ctrl_csv(args.save_best, rs.ctrl_pts)
        return

    best_idx = -1
    best_score = -np.inf
    best_ctrl = None
    best_mesh = None
    best_proj = None

    for i, ctrl in enumerate(candidates):
        mesh = rs._mesh_from_ctrl(ctrl)
        score, proj_pts = p3ds.score(mesh)

        if score > best_score:
            best_score = score
            best_idx = i
            best_ctrl = ctrl
            best_mesh = mesh
            best_proj = proj_pts

        if (i + 1) % max(1, args.N // 10) == 0 or i == args.N - 1:
            print(f"[progress] {i+1}/{args.N} | best idx={best_idx} score={best_score:.6f}")

    if best_ctrl is None:
        print("[warn] No best candidate; saving base ctrl.")
        best_ctrl = rs.ctrl_pts
        best_mesh = base_mesh
        # recompute projection for viewing, if requested
        _, best_proj = p3ds.score(best_mesh)

    save_ctrl_csv(args.save_best, best_ctrl)
    print(f"[done] Best control grid saved: {args.save_best} (idx={best_idx}, score={best_score:.6f})")

    # Prepare final geoms for your viewer
    ctrl_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(best_ctrl))
    ctrl_pcd.paint_uniform_color([1.0, 0.2, 0.2])

    ext_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud_pts))
    ext_pcd.paint_uniform_color([0.2, 0.6, 1.0])

    proj_pcd = None
    if args.show_projected and best_proj is not None:
        proj_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(best_proj))
        # color projected points by score if you later expose p3ds._point_scores;
        # for now, keep neutral:
        proj_pcd.paint_uniform_color([0.1, 0.8, 0.1])

    if not args.no_viz:
        if args.show_sandwich:
            # use RandomSurfacer’s prebuilt top/mid/base(bottom) meshes
            # (mid = base_mesh; top/bottom were created in __init__)
            # ensure normals for proper shading
            for m in [best_mesh, rs.top_mesh, rs.base_mesh, rs.bottom_mesh]:
                if m is not None and not m.has_vertex_normals():
                    m.compute_vertex_normals()
            visualize_with_materials_sandwich(
                ctrl_pcd=ctrl_pcd,
                best_mesh=best_mesh,
                ext_pcd=ext_pcd,
                top_mesh=rs.top_mesh,
                mid_mesh=rs.base_mesh,
                bot_mesh=rs.bottom_mesh,
                proj_pcd=proj_pcd,
            )
        else:
            if best_mesh is not None and not best_mesh.has_vertex_normals():
                best_mesh.compute_vertex_normals()
            visualize_with_materials(
                ctrl_pcd=ctrl_pcd,
                surf_mesh=best_mesh,
                ext_pcd=ext_pcd,
                proj_pcd=proj_pcd,
            )

if __name__ == "__main__":
    main()
