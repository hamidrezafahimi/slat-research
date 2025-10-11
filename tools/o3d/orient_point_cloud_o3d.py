#!/usr/bin/env python3
import argparse
import numpy as np
import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../lib")
from kinematics.clouds import orient_point_cloud_cgplane_global, apply_transform


import open3d as o3d


def create_xy_grid(size: float, step: float = None) -> o3d.geometry.LineSet:
    if step is None or step <= 0:
        step = max(size / 10.0, 1e-3)
    n = max(int(np.ceil(size / step)), 1)
    xs = np.linspace(-n * step, n * step, 2 * n + 1)
    ys = xs.copy()

    points = []
    lines = []

    for i, x in enumerate(xs):
        points.append([x, -n * step, 0.0])
        points.append([x,  n * step, 0.0])
        lines.append([2 * i, 2 * i + 1])

    offset = len(points)
    for j, y in enumerate(ys):
        points.append([-n * step, y, 0.0])
        points.append([ n * step, y, 0.0])
        lines.append([offset + 2 * j, offset + 2 * j + 1])

    grid = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(points)),
        lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32)),
    )
    colors = np.tile(np.array([[0.7, 0.7, 0.7]]), (len(lines), 1))
    grid.colors = o3d.utility.Vector3dVector(colors)
    return grid

def scene_union_aabb(aabb1, aabb2):
    minb = np.minimum(np.asarray(aabb1.get_min_bound()), np.asarray(aabb2.get_min_bound()))
    maxb = np.maximum(np.asarray(aabb1.get_max_bound()), np.asarray(aabb2.get_max_bound()))
    return o3d.geometry.AxisAlignedBoundingBox(minb, maxb)

def compute_rms_indexed(A_pts: np.ndarray, B_pts: np.ndarray) -> float:
    """Root-mean-square error assuming same index ordering."""
    if A_pts.shape != B_pts.shape:
        return float("nan")
    diff = A_pts - B_pts
    return float(np.sqrt((diff * diff).sum(axis=1).mean()))

def compute_rmse_nn(A: o3d.geometry.PointCloud, B: o3d.geometry.PointCloud):
    """Nearest-neighbor RMSE and max distance from B to A."""
    if len(A.points) == 0 or len(B.points) == 0:
        return float("nan"), float("nan")
    kdtree = o3d.geometry.KDTreeFlann(A)
    dists = []
    for p in np.asarray(B.points):
        _, idx, _ = kdtree.search_knn_vector_3d(p, 1)
        q = np.asarray(A.points)[idx[0]]
        d = np.linalg.norm(p - q)
        dists.append(d)
    dists = np.asarray(dists)
    rmse = float(np.sqrt((dists ** 2).mean()))
    dmax = float(dists.max())
    return rmse, dmax

def visualize_three(pcd_a, pcd_b, pcd_c, window_title="Original (RED) | Oriented (GREEN) | Restored (BLUE)"):
    left = o3d.geometry.PointCloud(pcd_a); left.paint_uniform_color([0.85, 0.2, 0.2])   # RED
    mid  = o3d.geometry.PointCloud(pcd_b); mid.paint_uniform_color([0.2, 0.7, 0.2])     # GREEN
    right= o3d.geometry.PointCloud(pcd_c); right.paint_uniform_color([0.2, 0.4, 0.85])  # BLUE

    aabb_left = left.get_axis_aligned_bounding_box()
    aabb_mid  = mid.get_axis_aligned_bounding_box()
    aabb_right= right.get_axis_aligned_bounding_box()

    extent = np.linalg.norm(aabb_left.get_extent())
    plane_half_size = max(extent * 0.6, 0.1)
    grid_left = create_xy_grid(plane_half_size, step=plane_half_size / 10.0)
    grid_mid  = create_xy_grid(plane_half_size, step=plane_half_size / 10.0)
    grid_right= create_xy_grid(plane_half_size, step=plane_half_size / 10.0)

    dx1 = (aabb_left.get_extent()[0] + aabb_mid.get_extent()[0]) * 1.2
    dx2 = (aabb_mid.get_extent()[0] + aabb_right.get_extent()[0]) * 1.2

    mid.translate([dx1, 0, 0], relative=True);       grid_mid.translate([dx1, 0, 0], relative=True)
    right.translate([dx1 + dx2, 0, 0], relative=True); grid_right.translate([dx1 + dx2, 0, 0], relative=True)

    o3d.visualization.draw_geometries(
        [left, grid_left, mid, grid_mid, right, grid_right],
        window_name=window_title
    )

def visualize_overlay_original_restored(orig, restored, window_title="Overlay: Original (RED) vs Restored (BLUE)"):
    left = o3d.geometry.PointCloud(orig); left.paint_uniform_color([0.85, 0.2, 0.2])   # RED
    right= o3d.geometry.PointCloud(restored); right.paint_uniform_color([0.2, 0.4, 0.85])  # BLUE
    aabb = left.get_axis_aligned_bounding_box()
    extent = np.linalg.norm(aabb.get_extent())
    grid = create_xy_grid(max(extent * 0.6, 0.1), step=max(extent * 0.6, 0.1)/10.0)
    o3d.visualization.draw_geometries([left, right, grid], window_name=window_title)

def render_offscreen_png_three(pcd_a, pcd_b, pcd_c, out_path):
    try:
        from open3d.visualization.rendering import OffscreenRenderer, MaterialRecord
    except Exception:
        return False, "Offscreen renderer not available in this build."

    w, h = 1920, 1080
    renderer = OffscreenRenderer(w, h)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])

    mat_pts = MaterialRecord(); mat_pts.shader = "defaultUnlit"; mat_pts.point_size = 2.0
    mat_lines = MaterialRecord(); mat_lines.shader = "unlitLine"; mat_lines.line_width = 1.0

    left = o3d.geometry.PointCloud(pcd_a); left.paint_uniform_color([0.85, 0.2, 0.2])
    mid  = o3d.geometry.PointCloud(pcd_b); mid.paint_uniform_color([0.2, 0.7, 0.2])
    right= o3d.geometry.PointCloud(pcd_c); right.paint_uniform_color([0.2, 0.4, 0.85])

    aabb_left = left.get_axis_aligned_bounding_box()
    aabb_mid  = mid.get_axis_aligned_bounding_box()
    aabb_right= right.get_axis_aligned_bounding_box()

    extent = np.linalg.norm(aabb_left.get_extent())
    plane_half_size = max(extent * 0.6, 0.1)
    grid_left = create_xy_grid(plane_half_size, step=plane_half_size / 10.0)
    grid_mid  = create_xy_grid(plane_half_size, step=plane_half_size / 10.0)
    grid_right= create_xy_grid(plane_half_size, step=plane_half_size / 10.0)

    dx1 = (aabb_left.get_extent()[0] + aabb_mid.get_extent()[0]) * 1.2
    dx2 = (aabb_mid.get_extent()[0] + aabb_right.get_extent()[0]) * 1.2

    mid.translate([dx1, 0, 0], relative=True);       grid_mid.translate([dx1, 0, 0], relative=True)
    right.translate([dx1 + dx2, 0, 0], relative=True); grid_right.translate([dx1 + dx2, 0, 0], relative=True)

    renderer.scene.add_geometry("left", left, mat_pts)
    renderer.scene.add_geometry("grid_left", grid_left, mat_lines)
    renderer.scene.add_geometry("mid", mid, mat_pts)
    renderer.scene.add_geometry("grid_mid", grid_mid, mat_lines)
    renderer.scene.add_geometry("right", right, mat_pts)
    renderer.scene.add_geometry("grid_right", grid_right, mat_lines)

    union = scene_union_aabb(aabb_left, mid.get_axis_aligned_bounding_box())
    union = scene_union_aabb(union, right.get_axis_aligned_bounding_box())
    target = union.get_center()
    diag = np.linalg.norm(union.get_extent())
    eye = target + np.array([0.0, -diag * 2.0, diag * 0.8])
    up = np.array([0.0, 0.0, 1.0])
    renderer.setup_camera(60.0, eye, target, up)

    renderer.scene.scene.set_sun_light(
        np.array([0.577, -0.577, -0.577]),
        np.array([1.0, 1.0, 1.0]),
        75000
    )
    renderer.scene.scene.enable_sun_light(True)

    img = renderer.render_to_image()
    ok = o3d.io.write_image(out_path, img)
    return ok, None if ok else "Failed to write image."

def render_offscreen_png_overlay(orig, restored, out_path):
    try:
        from open3d.visualization.rendering import OffscreenRenderer, MaterialRecord
    except Exception:
        return False, "Offscreen renderer not available in this build."
    w, h = 1600, 1000
    renderer = OffscreenRenderer(w, h)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])

    mat_pts = MaterialRecord(); mat_pts.shader = "defaultUnlit"; mat_pts.point_size = 2.0
    mat_lines = MaterialRecord(); mat_lines.shader = "unlitLine"; mat_lines.line_width = 1.0

    left = o3d.geometry.PointCloud(orig); left.paint_uniform_color([0.85, 0.2, 0.2])
    right= o3d.geometry.PointCloud(restored); right.paint_uniform_color([0.2, 0.4, 0.85])

    aabb = left.get_axis_aligned_bounding_box()
    grid = create_xy_grid(max(np.linalg.norm(aabb.get_extent()) * 0.6, 0.1))

    renderer.scene.add_geometry("orig", left, mat_pts)
    renderer.scene.add_geometry("restored", right, mat_pts)
    renderer.scene.add_geometry("grid", grid, mat_lines)

    union = scene_union_aabb(aabb, right.get_axis_aligned_bounding_box())
    target = union.get_center()
    diag = np.linalg.norm(union.get_extent())
    eye = target + np.array([0.0, -diag * 2.0, diag * 0.8])
    up = np.array([0.0, 0.0, 1.0])
    renderer.setup_camera(60.0, eye, target, up)

    renderer.scene.scene.set_sun_light(
        np.array([0.577, -0.577, -0.577]),
        np.array([1.0, 1.0, 1.0]),
        75000
    )
    renderer.scene.scene.enable_sun_light(True)

    img = renderer.render_to_image()
    ok = o3d.io.write_image(out_path, img)
    return ok, None if ok else "Failed to write image."

def main():
    parser = argparse.ArgumentParser(
        description="Orient a point cloud so a CG-constrained best-fit plane (computed from ALL points) becomes parallel to the XY plane. "
                    "Shows Original (RED) | Oriented (GREEN) | Restored (BLUE)."
    )
    parser.add_argument("--input", "-i", type=str, required=True, help="Input point cloud (e.g., .pcd, .ply, .xyz)")
    parser.add_argument("--output", "-o", type=str, default="out_oriented.pcd", help="Output oriented point cloud path")
    parser.add_argument("--restored-output", "-r", type=str, default="out_restored.pcd", help="Output restored (inverse-transformed) point cloud path")
    parser.add_argument("--save-png", type=str, default=None, help="Optional path to save an offscreen PNG render (three-view).")
    parser.add_argument("--save-overlay-png", type=str, default=None, help="Optional path to save an offscreen PNG of Original vs Restored overlay.")
    parser.add_argument("--no-gui", action="store_true", help="Skip interactive windows (useful on headless systems)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    pcd = o3d.io.read_point_cloud(args.input)
    if len(pcd.points) == 0:
        print("ERROR: Loaded point cloud is empty.", file=sys.stderr)
        sys.exit(1)

    # Orient using global (ALL points) CG plane
    oriented, T = orient_point_cloud_cgplane_global(pcd)
    o3d.io.write_point_cloud(args.output, oriented, write_ascii=False, compressed=True)
    print(f"[OK] Saved oriented point cloud: {args.output}")

    # Inverse transform analytically for exact reversal
    R = T[:3, :3]; t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3]  = -R.T @ t

    restored = apply_transform(oriented, T_inv)
    o3d.io.write_point_cloud(args.restored_output, restored, write_ascii=False, compressed=True)
    print(f"[OK] Saved restored point cloud (after inverse transform): {args.restored_output}")

    # Verification numbers
    A = np.asarray(pcd.points)
    C = np.asarray(restored.points)
    rms_index = compute_rms_indexed(A, C)
    rmse_nn, dmax_nn = compute_rmse_nn(pcd, restored)
    print(f"[VERIFY] Indexed RMS(original, restored) = {rms_index:.9e}")
    print(f"[VERIFY] NN-based  RMSE(original, restored) = {rmse_nn:.9e}, max = {dmax_nn:.9e}")

    print("[INFO] 4x4 transform (homogeneous) that ORIENTS (Original -> Oriented):")
    with np.printoptions(precision=6, suppress=True):
        print(T)
    print("[INFO] T_inv (Oriented -> Original) is computed analytically as [R^T | -R^T t].")

    # Renders / windows
    if args.save_png is not None:
        ok, msg = render_offscreen_png_three(pcd, oriented, restored, args.save_png)
        if ok:
            print(f"[OK] Saved triple-view PNG to: {args.save_png}")
        else:
            print(f"[WARN] Could not save triple-view PNG: {msg}")

    if args.save_overlay_png is not None:
        ok, msg = render_offscreen_png_overlay(pcd, restored, args.save_overlay_png)
        if ok:
            print(f"[OK] Saved overlay PNG to: {args.save_overlay_png}")
        else:
            print(f"[WARN] Could not save overlay PNG: {msg}")

    if not args.no_gui:
        visualize_three(pcd, oriented, restored, window_title="Original (RED) | Oriented (GREEN) | Restored (BLUE) — side-by-side")
        visualize_overlay_original_restored(pcd, restored, window_title="Overlay at same pose: Original (RED) vs Restored (BLUE)")

if __name__ == "__main__":
    main()
