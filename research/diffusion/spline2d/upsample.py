import argparse
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../../lib")
# --- IMPORT or define your helper functions here ---
from diffusion.helper import create_grid_on_surface
from geom.surfaces import infer_grid, generate_spline


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

def visualize_with_materials(ctrl_pcd, surf_mesh, ext_pcd, proj_pcd=None, grid_pcd=None):
    gui.Application.instance.initialize()
    window = gui.Application.instance.create_window("Spline Visualization with Grid", 1024, 768)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    window.add_child(scene)

    scene.scene.add_geometry("ctrl_pcd", ctrl_pcd, mat_points(8.0))
    scene.scene.add_geometry("surf_mesh", surf_mesh, mat_mesh())
    scene.scene.add_geometry("ext_pcd", ext_pcd, mat_points(2.0))
    if proj_pcd is not None:
        scene.scene.add_geometry("proj_pcd", proj_pcd, mat_points(2.0))
    if grid_pcd is not None:
        scene.scene.add_geometry("grid_pcd", grid_pcd, mat_points(4.0))

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    scene.scene.add_geometry("axis", axis, mat_mesh())

    # fit camera
    if proj_pcd is not None:
        aabbs = [g.get_axis_aligned_bounding_box() for g in [ctrl_pcd, surf_mesh, ext_pcd, proj_pcd, grid_pcd]]
    else:
        aabbs = [g.get_axis_aligned_bounding_box() for g in [ctrl_pcd, surf_mesh, ext_pcd, grid_pcd]]
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
    ap = argparse.ArgumentParser(description="Spline Visualization with Grid on Surface")
    ap.add_argument("--spline_data", required=True, help="CSV of control points (x,y,z).")
    ap.add_argument("--samples_u", type=int, default=10)
    ap.add_argument("--samples_v", type=int, default=10)
    ap.add_argument("--grid_w", type=int, default=6)
    ap.add_argument("--grid_h", type=int, default=4)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--output_csv", required=True, help="Output CSV file to save grid points.")
    args = ap.parse_args()

    # Load control points from the provided CSV file
    ctrl_pts = np.loadtxt(args.spline_data, delimiter=",", dtype=float)

    gw, gh = infer_grid(ctrl_pts)

    # Generate spline (assuming `generate_xy_spline` exists and works)
    mesh, W, H = generate_spline(ctrl_pts)

    # Prepare point cloud for control points
    ctrl_pcd = o3d.geometry.PointCloud()
    ctrl_pcd.points = o3d.utility.Vector3dVector(ctrl_pts)

    # External point cloud (assuming it exists and is generated elsewhere)
    ext_pcd = o3d.geometry.PointCloud()
    ext_pcd.points = o3d.utility.Vector3dVector(ctrl_pts)  # Placeholder for external points

    # Projected point cloud (optional, assuming it's also generated)
    proj_pcd = o3d.geometry.PointCloud()  # Placeholder

    # Create mesh grid points on the surface
    grid_points_3d, grid_pcd = create_grid_on_surface(mesh, args.samples_u, args.samples_v)

    # Save the grid points to CSV file
    np.savetxt(args.output_csv, grid_points_3d, delimiter=",", fmt="%f")

    # Visualize with materials if --show_gui is set
    visualize_with_materials(ctrl_pcd, mesh, ext_pcd, proj_pcd, grid_pcd)

if __name__ == "__main__":
    main()
