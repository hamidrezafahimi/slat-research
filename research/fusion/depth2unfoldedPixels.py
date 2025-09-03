#!/usr/bin/env python3
"""
Multi-point surface unfold on metric-depth input (ONLY).

Input:
  --metric_depth metrics_depth.csv
  [--hfov_deg 66.0] [--pose_x ... --pose_y ... --pose_z ... --pose_roll ... --pose_pitch ... --pose_yaw ...]

Behavioral contracts preserved:
- Visualization artifacts, colors, geometry names → EXACTLY as before (when --vis-debug true).
- Unfolding mathematics and terminal report structure → EXACTLY as before.
- All points are painted black (no color.png needed).
- 10 random (i,j) from HxW grid by default (configurable via --n_greens).
"""

import argparse
import time
import numpy as np
import os, sys
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(DIR_PATH + "/../../lib")
from unfold_helper import *

try:
    from kinematics.pose import Pose
    from mapper3D_helper import project3DAndScale
except Exception as e:
    raise SystemExit(f"Required modules not found: {e}\n"
                     f"Make sure Pose and project3DAndScale are importable like in depth2pcd.py")

# -------------------- main (METRIC DEPTH ONLY) --------------------
def main():
    ap = argparse.ArgumentParser(
        description="Multi-point unfold (metric-depth only) with original visualization/report."
    )
    ap.add_argument("--metric_depth", required=True, help="Path to metrics_depth.csv")
    ap.add_argument("--k_plane", type=int, default=50)
    ap.add_argument("--k_proj", type=int, default=40)
    ap.add_argument("--allow_backward", action="store_true")
    ap.add_argument("--n_mid", type=int, default=10)
    ap.add_argument("--spline_samples", type=int, default=200)
    ap.add_argument("--spline_color", type=float, nargs=3, default=[1.0, 1.0, 0.0])
    ap.add_argument("--show_green_to_orange", dest="show_green_to_orange", action="store_true")
    ap.add_argument("--no-show_green_to_orange", dest="show_green_to_orange", action="store_false")
    ap.add_argument("--n_greens", type=int, default=10)

    # Camera/pose params (same style as your depth2pcd)
    ap.add_argument("--hfov_deg", type=float, default=66.0)
    ap.add_argument("--pose_x", type=float, default=0.0)
    ap.add_argument("--pose_y", type=float, default=0.0)
    ap.add_argument("--pose_z", type=float, default=9.4)
    ap.add_argument("--pose_roll", type=float, default=0.0)
    ap.add_argument("--pose_pitch", type=float, default=-0.78)
    ap.add_argument("--pose_yaw", type=float, default=0.0)

    # NEW: visualization/debug toggle
    ap.add_argument("--vis-debug", type=lambda x: str(x).lower() in ["true","1","yes"],
                    default=True, help="Enable visualization/debug artifacts (default: true)")

    ap.set_defaults(show_green_to_orange=True)
    args = ap.parse_args()

    # CONFIG update (unchanged math)
    CONFIG.k_plane = args.k_plane
    CONFIG.k_proj = args.k_proj
    CONFIG.allow_backward = args.allow_backward
    CONFIG.n_mid = args.n_mid
    CONFIG.spline_samples = args.spline_samples
    CONFIG.spline_color = args.spline_color
    CONFIG.show_green_to_orange = args.show_green_to_orange
    CONFIG.n_greens = args.n_greens
    CONFIG.vis_debug = args.vis_debug

    t0 = time.time()

    # 1) Load metric depth
    metric_depth = np.loadtxt(args.metric_depth, delimiter=',', dtype=np.float32)
    H, W = metric_depth.shape[:2]

    # 2) Build pose & HxWx3 via project3DAndScale
    pose = Pose(x=args.pose_x, y=args.pose_y, z=args.pose_z,
                roll=args.pose_roll, pitch=args.pose_pitch, yaw=args.pose_yaw)
    xyz_img = project3DAndScale(metric_depth, pose, args.hfov_deg, metric_depth.shape)

    # 3) Flatten to pcd (black)
    pts_flat = xyz_img.reshape(-1, 3)
    mask = np.isfinite(pts_flat).all(axis=1)
    pts_flat = pts_flat[mask]
    if pts_flat.size == 0:
        raise SystemExit("All projected points are non-finite.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_flat)
    pcd.paint_uniform_color([0.0, 0.0, 0.0])

    # Finite pts for math (unchanged)
    pts = finite_points_from_pcd(pcd)
    if pts.shape[0] != len(pcd.points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color([0.0, 0.0, 0.0])

    # Step 1: surface-interior-cg via cg + local plane hit (unchanged)
    surface_interior_cg, aux1 = find_surface_mass_anchor(pts)
    cg = aux1["cg"]

    # Log base positions (EXACT text) — only when debugging
    def fmt(p): return "x={:.6f}, y={:.6f}, z={:.6f}".format(p[0], p[1], p[2])
    if CONFIG.vis_debug:
        print("cg:                  " + fmt(cg))
        print("surface-interior-cg: " + fmt(surface_interior_cg))

    # Step 2: choose distinct random (i,j) from HxW
    points = []
    used_keys = set()
    tries = CONFIG.n_greens * 6
    rng = random.Random()
    for _ in range(tries):
        i = rng.randrange(H)
        j = rng.randrange(W)
        key = (i, j)
        if key in used_keys:
            continue
        p_ij = xyz_img[i, j]
        if not np.all(np.isfinite(p_ij)):
            continue
        if np.linalg.norm(p_ij - surface_interior_cg) <= 1e-12:
            continue
        points.append(p_ij)
        used_keys.add(key)
        if len(points) >= CONFIG.n_greens:
            break
    if CONFIG.vis_debug and len(points) < CONFIG.n_greens:
        print(f"[WARN] Only found {len(points)} unique (i,j) points (requested {CONFIG.n_greens}).")

    # Step 3: loop over points (UNCHANGED math & per-item prints behind flag)
    items = []
    for i, p in enumerate(points, 1):
        dir_vec, spline_len, aux2 = find_surface_point_unfold_params(pts, surface_interior_cg, p)
        white_len = float(np.linalg.norm(p - surface_interior_cg))  # radial-dir-(id) length

        unfolded_i = unfold_surface_point(surface_interior_cg, dir_vec, spline_len)

        if CONFIG.vis_debug:
            radial_proj_i = unfold_surface_point(surface_interior_cg, dir_vec, white_len)
            print(f"[{i:02d}] point-{i}:  x={p[0]:.6f}, y={p[1]:.6f}, z={p[2]:.6f}")
            print(f"     unfolded-point-{i}:  x={unfolded_i[0]:.6f}, y={unfolded_i[1]:.6f}, z={unfolded_i[2]:.6f}")
            print(f"     cyan-dir: ({dir_vec[0]:.6f}, {dir_vec[1]:.6f}, {dir_vec[2]:.6f})")
            print(f"     unfold-spline-{i} length = {spline_len:.6f}")
            print(f"     radial-dir-{i} length    = {white_len:.6f}")
            if white_len > spline_len:
                print("     [DEBUG] WARNING: white > spline (unexpected).")
            else:
                print("     [DEBUG] OK: white <= spline.")
            d_unfold = float(np.linalg.norm(unfolded_i - surface_interior_cg))
            d_radial = float(np.linalg.norm(radial_proj_i - surface_interior_cg))
            if d_radial <= d_unfold + 1e-12:
                print("     [DEBUG] OK: radial-projection-point nearer than unfolded-point.")
            else:
                print("     [DEBUG] WARNING: radial-projection farther than unfolded (unexpected).")

            items.append(dict(
                point=p,
                mids=aux2["mids"],
                proj_points=aux2["proj_points"],
                curve=aux2["curve"],
                unfolded=unfolded_i,
                radial_proj=radial_proj_i,
            ))

    print("Done in %.2f ms" % (1000.0 * (time.time() - t0)))

    # Step 4: visualize once for all points (only when debugging)
    if CONFIG.vis_debug:
        visualize_scene_multi(
            pcd=pcd, cg=cg, surface_interior_cg=surface_interior_cg, items=items,
            spline_color=CONFIG.spline_color,
            show_green_to_orange=CONFIG.show_green_to_orange
        )

if __name__ == "__main__":
    main()
