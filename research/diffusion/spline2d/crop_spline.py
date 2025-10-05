from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from helper import *


def visualize_with_materials(surf_mesh1, surf_mesh2=None):
# def visualize_with_materials(surf_mesh1):
    gui.Application.instance.initialize()
    window = gui.Application.instance.create_window("Smoothness + Surface", 1024, 768)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    window.add_child(scene)

    # Add the first control points and mesh
    scene.scene.add_geometry("surf_mesh1", surf_mesh1, mat_mesh())
    
    if surf_mesh2 is not None:
        # Add the second control points and mesh
        scene.scene.add_geometry("surf_mesh2", surf_mesh2, mat_mesh())

    # Add axis
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    scene.scene.add_geometry("axis", axis, mat_mesh())

    # Fit camera to all geometries
    if surf_mesh2 is not None:
        aabbs = [g.get_axis_aligned_bounding_box() for g in [surf_mesh1, surf_mesh2]]
    else:
        aabbs = [g.get_axis_aligned_bounding_box() for g in [surf_mesh1]]
    
    mins = np.min([a.min_bound for a in aabbs], axis=0)
    maxs = np.max([a.max_bound for a in aabbs], axis=0)
    center = 0.5 * (mins + maxs)
    extent = max(maxs - mins)
    
    eye = center + np.array([0, -3.0 * extent, 1.8 * extent])
    up = np.array([0, 0, 1])
    scene.scene.camera.look_at(center, eye, up)

    # Run the viewer
    gui.Application.instance.run()

# def align_meshes_with_control_points(ctrl_pts: np.ndarray, nei_p: np.ndarray) -> np.ndarray:
#     """
#     Aligns the smaller control points grid (`nei_p`) with the original grid (`ctrl_pts`).
#     It ensures that the interpolated grid matches the original mesh's layout exactly.
#     """
#     # Reorder control points (ensure correct spatial alignment)
#     gw, gh = infer_grid(ctrl_pts)
    
#     # Interpolate nei_p to match the full control points grid size (e.g., 10x10)
#     target_grid_size = gw  # Target grid size should be the same as the original grid
    
#     return expanded_ctrl_pts


def compute_average_z_distance(mesh1: o3d.geometry.TriangleMesh, mesh2: o3d.geometry.TriangleMesh) -> float:
    """
    Compute the average Z-distance between two meshes (mesh1 and mesh2).
    The meshes must have the same number of vertices and aligned spatially.
    """
    # Extract vertex positions
    vertices1 = np.asarray(mesh1.vertices)
    vertices2 = np.asarray(mesh2.vertices)
    
    if vertices1.shape != vertices2.shape:
        raise ValueError("Meshes must have the same number of vertices to compute Z-distance.")
    
    # Compute the Z-coordinate difference
    z_diff = np.abs(vertices1[:, 2] - vertices2[:, 2])
    
    # Compute the average Z-distance
    avg_z_distance = np.mean(z_diff)
    
    return avg_z_distance

def get_submesh_on_index(ind, gw, gh, ctrl_pts, su, sv):
    nei_p, dims, k = extract_adjacent_nodes(25, gw, gh, ctrl_pts)
    target_grid_size = 5
    expanded_ctrl_pts = interpolate_ctrl_points_to_larger_grid(nei_p, target_grid_size)
    return bspline_surface_mesh_from_ctrl(expanded_ctrl_pts, target_grid_size, target_grid_size, su=su, sv=sv)


def main():
    ap = argparse.ArgumentParser(description="Project PCD points onto a B-spline surface defined by control points.")
    ap.add_argument("--ctrl", required=True, type=Path, help="Path to control points CSV (x,y,z)")
    ap.add_argument("--su", type=int, default=100, help="Surface samples along U (columns / width)")
    ap.add_argument("--sv", type=int, default=100, help="Surface samples along V (rows / height)")
    ap.add_argument("--k", type=int, default=3, help="k-NN for IDW projection (unused for normal-ray method)")
    ap.add_argument("--max-lines", type=int, default=0, help="Global cap on number of connecting lines (0 = all)")
    args = ap.parse_args()

    print(f"Reading control CSV: {args.ctrl}")
    ctrl_pts = np.loadtxt(args.ctrl, delimiter=",", dtype=float)

    # --- Infer grid dimensions and reorder to row-major by (y,x) ---
    gw, gh = infer_grid(ctrl_pts)
    print(f"Inferred control grid: gw={gw}, gh={gh}  (total={ctrl_pts.shape[0]})")
    ctrl_pts_sorted = reorder_ctrl_points_rowmajor(ctrl_pts)

    # --- Build the B-spline surface mesh ---
    print(f"Building B-spline surface mesh with su={args.su}, sv={args.sv} ...")
    
    target_idx = 25
    ctrl_pts[target_idx, 2] += 40
    mesh2 = bspline_surface_mesh_from_ctrl(ctrl_pts, gw, gh, args.su, args.sv)
    
    mesh1 = get_submesh_on_index(target_idx, gw, gh, ctrl_pts, args.su, args.sv )

    mesh2.compute_vertex_normals()  # ensure normals exist
    mesh2.paint_uniform_color([0.0, 0.0, 1.0])

    mesh1.compute_vertex_normals()  # ensure normals exist
    mesh1.paint_uniform_color([0.0, 0.7, 1.0])

    # visualize_with_materials(mesh2)
    visualize_with_materials(mesh2, mesh1)

if __name__ == "__main__":
    main()