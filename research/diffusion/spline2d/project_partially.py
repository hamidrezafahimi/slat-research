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

# >>> Your helper with the provided functions <<<
# It must define: infer_grid, bspline_surface_mesh_from_ctrl, project_external_to_surface_idw,
# plus reorder_ctrl_points_rowmajor and the grid helpers you used.
from helper import *

# -----------------------------
# NEW: robust normal-ray projector
# -----------------------------
def _interp_face_normal_at_bary(mesh_legacy: o3d.geometry.TriangleMesh,
                                tri_id: int,
                                uv: np.ndarray) -> np.ndarray:
    """Interpolate vertex normals at a point on a triangle given (u,v)."""
    tris = np.asarray(mesh_legacy.triangles, dtype=np.int32)
    vnorms = np.asarray(mesh_legacy.vertex_normals, dtype=float)
    i0, i1, i2 = tris[tri_id]
    u, v = float(uv[0]), float(uv[1])
    w = 1.0 - u - v
    n = w * vnorms[i0] + u * vnorms[i1] + v * vnorms[i2]
    n_norm = np.linalg.norm(n)
    if n_norm == 0 or not np.isfinite(n_norm):
        # fallback to un-interpolated face normal
        v = np.asarray(mesh_legacy.vertices, dtype=float)
        a, b, c = v[i0], v[i1], v[i2]
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm == 0 or not np.isfinite(n_norm):
            return np.array([0.0, 0.0, 1.0], dtype=float)
        return n / n_norm
    return n / n_norm


def project_points_via_normal_raycast(points: np.ndarray,
                                      mesh_legacy: o3d.geometry.TriangleMesh,
                                      angle_tol_deg: float = 80.0,
                                      hit_pos_tol_rel: float = 1e-1) -> tuple[np.ndarray, np.ndarray]:
    """
    Project each input point along the local mesh normal direction, but ONLY keep points
    whose ±normal-direction rays actually hit the mesh near their closest surface point.

    Returns:
      proj_pts: (M,3) array of accepted projected points
      idx:      (M,)   indices into `points` of the accepted projections
    """
    if points.size == 0:
        return np.empty((0, 3), float), np.empty((0,), int)

    # Ensure vertex normals exist for interpolation
    if len(mesh_legacy.vertex_normals) == 0:
        mesh_legacy.compute_vertex_normals()

    # Scene scale for tolerances
    verts = np.asarray(mesh_legacy.vertices, float)
    if verts.size == 0:
        return np.empty((0, 3), float), np.empty((0,), int)
    mn, mx = verts.min(axis=0), verts.max(axis=0)
    diag = float(np.linalg.norm(mx - mn)) or 1.0
    hit_pos_tol = hit_pos_tol_rel * diag
    cos_thresh = float(np.cos(np.deg2rad(angle_tol_deg)))

    # Prefer Open3D tensor raycasting if available
    has_tensor = hasattr(o3d, "t") and hasattr(o3d.t, "geometry") and hasattr(o3d.t.geometry, "RaycastingScene")
    if has_tensor:
        # Build tensor scene
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(tmesh)

        p_t = o3d.core.Tensor(points.astype(np.float32))
        closest = scene.compute_closest_points(p_t)
        c = closest["points"].numpy()                     # (N,3) closest points on surface
        tri_ids = closest["primitive_ids"].numpy()        # (N,) triangle ids
        uv = closest["primitive_uvs"].numpy()             # (N,2) (u,v) on that tri

        # Interpolate a normal for each closest point
        n_interp = np.zeros_like(c)
        for i in range(points.shape[0]):
            if tri_ids[i] < 0:
                n_interp[i] = np.array([0., 0., 1.], float)
            else:
                n_interp[i] = _interp_face_normal_at_bary(mesh_legacy, int(tri_ids[i]), uv[i])

        v = points - c
        v_norm = np.linalg.norm(v, axis=1)
        # Discard degenerate cases up-front
        nonzero = v_norm > 0

        # Choose ±normal direction that points toward the surface (i.e., opposite of v)
        s = np.sign(np.einsum("ij,ij->i", v, n_interp))
        s[s == 0] = 1.0
        d = (-s[:, None]) * n_interp

        # Build rays: [ox, oy, oz, dx, dy, dz]
        rays = np.concatenate([points.astype(np.float32), d.astype(np.float32)], axis=1)
        hits = scene.cast_rays(o3d.core.Tensor(rays))
        t_hit = hits["t_hit"].numpy()                     # inf where no hit
        hit_ok = np.isfinite(t_hit)

        # Compute hit positions and test proximity to the closest points
        hit_pts = points + d * t_hit[:, None]
        close_to_closest = np.linalg.norm(hit_pts - c, axis=1) <= hit_pos_tol

        # Also require alignment with normal within an angle tolerance
        # |cos(theta)| = |v·n|/(||v||*||n||)  (||n||=1 here)
        with np.errstate(invalid="ignore", divide="ignore"):
            cosang = np.abs(np.einsum("ij,ij->i", v, n_interp) / (v_norm + 1e-12))

        valid = hit_ok & nonzero & close_to_closest & (cosang >= cos_thresh)

        idx = np.nonzero(valid)[0]
        return hit_pts[idx], idx

    # -------- Fallback path (no tensor ray caster): alignment-only gate --------
    # Uses only closest point + normal alignment to reject boundary-snapping.
    # This won’t “prove” an intersection but works well to drop tangential/boundary snaps.
    # Build a fast closest-point structure via RaycastingScene if available; otherwise sample densely.
    # Here we do a simple KD on mesh vertices as a cheap proxy, then refine via triangle plane.
    # If your environment lacks o3d.t, consider upgrading Open3D for robust ray casting.
    tri = np.asarray(mesh_legacy.triangles, dtype=np.int32)
    ver = np.asarray(mesh_legacy.vertices, dtype=float)
    vnorms = np.asarray(mesh_legacy.vertex_normals, dtype=float)
    # Quick AABB tree for triangles is not exposed in legacy; use a coarse nearest-vertex proxy:
    kdtree = o3d.geometry.KDTreeFlann(o3d.utility.Vector3dVector(ver))

    proj_list = []
    idx_list = []
    for i, p in enumerate(points):
        # nearest vertex -> take one of its incident triangles (best-effort)
        _, idxs, _ = kdtree.search_knn_vector_3d(p, 1)
        if not idxs:
            continue
        v_id = int(idxs[0])
        # find triangles that include this vertex
        tri_ids = np.nonzero((tri == v_id).any(axis=1))[0]
        if tri_ids.size == 0:
            continue
        best_h = None
        best_d = np.inf
        for tid in tri_ids:
            i0, i1, i2 = tri[tid]
            a, b, c = ver[i0], ver[i1], ver[i2]
            n = np.cross(b - a, c - a)
            nn = np.linalg.norm(n)
            if nn == 0:
                continue
            n = n / nn
            # closest point on the plane
            t = np.dot(a - p, n)
            h = p + n * t
            # barycentric test
            v0, v1, v2 = b - a, c - a, h - a
            d00 = np.dot(v0, v0)
            d01 = np.dot(v0, v1)
            d11 = np.dot(v1, v1)
            d20 = np.dot(v2, v0)
            d21 = np.dot(v2, v1)
            denom = d00 * d11 - d01 * d01
            if denom == 0:
                continue
            v = (d11 * d20 - d01 * d21) / denom
            w = (d00 * d21 - d01 * d20) / denom
            u = 1 - v - w
            inside = (u >= -1e-8) and (v >= -1e-8) and (w >= -1e-8)
            if not inside:
                continue
            # alignment check with interpolated normal
            n_interp = (u * vnorms[i0] + v * vnorms[i1] + w * vnorms[i2])
            nn2 = np.linalg.norm(n_interp)
            if nn2 == 0:
                continue
            n_interp = n_interp / nn2
            vec = p - h
            lv = np.linalg.norm(vec)
            if lv == 0:
                continue
            cosang = abs(np.dot(vec / lv, n_interp))
            if cosang >= cos_thresh:
                d2 = lv
                if d2 < best_d:
                    best_d = d2
                    best_h = h
        if best_h is not None:
            proj_list.append(best_h)
            idx_list.append(i)

    if not proj_list:
        return np.empty((0, 3), float), np.empty((0,), int)
    return np.vstack(proj_list).astype(float), np.asarray(idx_list, dtype=int)


def make_lineset_for_pairs(src_pts: np.ndarray, dst_pts: np.ndarray, color=(1.0, 0.0, 1.0)) -> o3d.geometry.LineSet:
    """Build a LineSet connecting each src point to its corresponding dst point."""
    if src_pts.shape != dst_pts.shape:
        raise ValueError("Source and destination point arrays must have the same shape.")
    n = src_pts.shape[0]
    if n == 0:
        return o3d.geometry.LineSet()
    points = np.vstack([src_pts, dst_pts])
    lines = np.column_stack([np.arange(n, dtype=np.int32), np.arange(n, 2*n, dtype=np.int32)])
    colors = np.tile(np.asarray(color, float)[None, :], (n, 1))
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points.astype(float))
    ls.lines = o3d.utility.Vector2iVector(lines.astype(np.int32))
    ls.colors = o3d.utility.Vector3dVector(colors.astype(float))
    return ls


def _estimate_marker_radius(mesh: o3d.geometry.TriangleMesh, points: np.ndarray) -> float:
    """Heuristic sphere radius based on scene scale."""
    verts = np.asarray(mesh.vertices)
    all_pts = verts if points.size == 0 else np.vstack([verts, points])
    if all_pts.shape[0] < 2:
        return 1e-2
    mn = all_pts.min(axis=0)
    mx = all_pts.max(axis=0)
    diag = np.linalg.norm(mx - mn)
    if not np.isfinite(diag) or diag <= 0:
        return 1e-2
    return float(0.01 * diag)  # 1% of scene diagonal

def get_submesh_on_index(ind, gw, gh, ctrl_pts, su, sv):
    nei_p, dims, k = extract_adjacent_nodes(25, gw, gh, ctrl_pts)
    target_grid_size = 10
    expanded_ctrl_pts = interpolate_ctrl_points_to_larger_grid(nei_p, target_grid_size)
    return bspline_surface_mesh_from_ctrl(expanded_ctrl_pts, target_grid_size, target_grid_size, su=su, sv=sv)


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
    mesh1 = get_submesh_on_index(25, gw, gh, ctrl_pts_sorted, args.su, args.sv)
    mesh.compute_vertex_normals()  # ensure normals exist

    # --- Project points along mesh normals, but only keep real ray intersections ---
    print(f"Projecting {ext_pts.shape[0]} points along mesh normals (±n with intersection gating)...")
    proj_pts, valid_idx = project_points_via_normal_raycast(ext_pts, mesh)
    print(f"Valid projections: {proj_pts.shape[0]}  |  skipped: {ext_pts.shape[0] - proj_pts.shape[0]}")

    # --- Add grey spheres on spline control points ---
    grey_spheres = []
    r = _estimate_marker_radius(mesh, ext_pts)  # Estimate marker radius based on scene scale
    for p in ctrl_pts_sorted:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=r)
        sphere.paint_uniform_color([0.5, 0.5, 0.5])  # Grey color for the spheres
        sphere.translate(p, relative=False)
        sphere.compute_vertex_normals()
        grey_spheres.append(sphere)

    # --- Build visuals for Viewer #1 (everything) ---
    # full input cloud (unchanged)
    pcd_in_all = o3d.geometry.PointCloud()
    pcd_in_all.points = o3d.utility.Vector3dVector(ext_pts)
    pcd_in_all.colors = pcd.colors  # Keep original colors of the point cloud

    # projected subset only
    pcd_proj_all = o3d.geometry.PointCloud()
    pcd_proj_all.points = o3d.utility.Vector3dVector(proj_pts)
    pcd_proj_all.paint_uniform_color([1.0, 0.0, 0.0])

    mesh.paint_uniform_color([0.2, 0.7, 1.0])
    mesh1.paint_uniform_color([0.2, 0.7, 0])

    # Build lines only for valid indices
    src_valid = ext_pts[valid_idx]

    # Color by vertical difference as your note states
    dz = src_valid[:, 2] - proj_pts[:, 2]
    above_mask = dz > 0
    below_mask = ~above_mask

    lines_above = make_lineset_for_pairs(src_valid[above_mask], proj_pts[above_mask], color=(1.0, 0.5, 0.0))  # ORANGE
    lines_below = make_lineset_for_pairs(src_valid[below_mask], proj_pts[below_mask], color=(0.0, 1.0, 0.0))  # GREEN


    o3d.visualization.draw_geometries([mesh, mesh1, pcd_in_all, pcd_proj_all, lines_above, lines_below, *grey_spheres],
                                        window_name="PCD → B-spline projection (Viewer #1)",
                                        point_show_normal=False)

    # Save projected points to PCD if the --output argument is provided
    if args.output:
        print(f"Saving projected points to {args.output}")
        pcd_proj = o3d.geometry.PointCloud()
        pcd_proj.points = o3d.utility.Vector3dVector(proj_pts)
        o3d.io.write_point_cloud(str(args.output), pcd_proj)
        print("Projected points saved successfully.")

if __name__ == "__main__":
    main()
