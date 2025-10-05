import numpy as np
import open3d as o3d
from scipy.interpolate import Rbf
import csv
import argparse

from helper import (
    infer_grid,
    bspline_surface_mesh_from_ctrl,
    project_external_along_normals,   # normal-ray projection (as used previously)
    reorder_ctrl_points_rowmajor,     # row-major reorder (y, then x)
    unique_sorted_with_tol            # Assuming you have this function imported from helper
)


# ---- Materials
def _mat_points(size=4.0, color=(0.8, 0.2, 0.2)):
    m = o3d.visualization.rendering.MaterialRecord()
    m.shader = "defaultUnlit"
    m.point_size = float(size)
    return m

def _mat_mesh(backface_culling=True):
    m = o3d.visualization.rendering.MaterialRecord()
    m.shader = "defaultLit"
    m.base_color = (0.7, 0.7, 0.9, 1.0)
    m.base_roughness = 0.8
    return m

def upsample_ctrl_points_rbf(ctrl_pts_flat, new_grid_w, new_grid_h, grid_w, grid_h):
    """
    Upsample the control points using Radial Basis Function (RBF) interpolation
    from a 3x3 grid to a new grid (new_grid_w x new_grid_h).
    """
    u, v = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
    
    u_flat = u.flatten()
    v_flat = v.flatten()
    
    rbf_x = Rbf(u_flat, v_flat, ctrl_pts_flat[:, 0], function='multiquadric', epsilon=2)
    rbf_y = Rbf(u_flat, v_flat, ctrl_pts_flat[:, 1], function='multiquadric', epsilon=2)
    rbf_z = Rbf(u_flat, v_flat, ctrl_pts_flat[:, 2], function='multiquadric', epsilon=2)
    
    new_u = np.linspace(0, grid_w - 1, new_grid_w)
    new_v = np.linspace(0, grid_h - 1, new_grid_h)
    new_u_grid, new_v_grid = np.meshgrid(new_u, new_v)
    
    new_u_flat = new_u_grid.flatten()
    new_v_flat = new_v_grid.flatten()
    
    upsampled_x = rbf_x(new_u_flat, new_v_flat)
    upsampled_y = rbf_y(new_u_flat, new_v_flat)
    upsampled_z = rbf_z(new_u_flat, new_v_flat)
    
    upsampled_ctrl_pts = np.stack([upsampled_x, upsampled_y, upsampled_z], axis=-1)
    return upsampled_ctrl_pts.reshape(new_grid_h, new_grid_w, 3)

def write_ctrl_points_to_csv(ctrl_pts, filename="upsampled_ctrl_points.csv"):
    """
    Write the control points to a CSV file.
    """
    grid_h, grid_w, _ = ctrl_pts.shape
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for i in range(grid_h):
            for j in range(grid_w):
                writer.writerow(ctrl_pts[i, j])

def visualize_bspline(ctrl_pts, grid_w, grid_h, su, sv, ctrl_pts_initial=None, ctrl_pts_initial_color=(0.8, 0.2, 0.2)):
    """Visualize the B-spline surface generated from the control points."""
    ctrl_pts_flat = ctrl_pts.reshape(-1, 3)
    mesh = bspline_surface_mesh_from_ctrl(ctrl_pts_flat, grid_w, grid_h, su, sv)
    
    ctrl_points_geometries = o3d.geometry.PointCloud()
    ctrl_points_geometries.points = o3d.utility.Vector3dVector(ctrl_pts.reshape(-1, 3))

    if ctrl_pts_initial is not None:
        ctrl_points_initial_geometries = []
        for pt in ctrl_pts_initial.reshape(-1, 3):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=6.0)
            sphere.translate(pt)
            ctrl_points_initial_geometries.append(sphere)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    vis.add_geometry(mesh)
    vis.get_render_option().point_size = 2
    
    vis.add_geometry(ctrl_points_geometries)
    
    if ctrl_pts_initial is not None:
        for sphere in ctrl_points_initial_geometries:
            vis.add_geometry(sphere)
    
    vis.run()
    vis.destroy_window()


# Step 1: Load original control points from CSV
def load_ctrl_points_from_csv(filename):
    ctrl_pts = np.loadtxt(filename, delimiter=",")  
    return ctrl_pts.reshape(-1, 3)  # Assuming the data is structured (N, 3)

# Step 2: Infer grid dimensions from the control points
def get_grid_dimensions(ctrl_points):
    grid_w, grid_h = infer_grid(ctrl_points)
    return grid_w, grid_h

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process control points and upsample using RBF interpolation.")
    parser.add_argument("input_csv", type=str, help="Path to the input CSV file containing original control points.")
    parser.add_argument("output_csv", type=str, help="Path to the output CSV file to save upsampled control points.")
    parser.add_argument("new_grid_w", type=int, help="Width of the new grid (upsampled).")
    parser.add_argument("new_grid_h", type=int, help="Height of the new grid (upsampled).")
    
    args = parser.parse_args()

    # Step 1: Load original control points from the input CSV
    ctrl_points_initial = load_ctrl_points_from_csv(args.input_csv)
    
    # Step 2: Infer grid dimensions from the original control points
    grid_w, grid_h = get_grid_dimensions(ctrl_points_initial)

    # Step 3: Upsample the control points using RBF interpolation
    upsampled_ctrl_points = upsample_ctrl_points_rbf(ctrl_points_initial, new_grid_w=args.new_grid_w, 
                                                     new_grid_h=args.new_grid_h, grid_w=grid_w, grid_h=grid_h)

    # Step 4: Write the upsampled control points to the output CSV
    write_ctrl_points_to_csv(upsampled_ctrl_points, filename=args.output_csv)

    # Step 5: Visualize the original control points and the upsampled surface
    visualize_bspline(upsampled_ctrl_points, grid_w=args.new_grid_w, grid_h=args.new_grid_h, su=100, sv=100, ctrl_pts_initial=ctrl_points_initial)


if __name__ == "__main__":
    main()
