#!/usr/bin/env python3
import argparse
import time
import numpy as np
import open3d as o3d
import random
from types import SimpleNamespace
import os, sys
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(DIR_PATH + "/../../lib")
from unfold_helper import *

def main():
    ap = argparse.ArgumentParser(
        description="Multi-point unfold with renamed entities and plane G."
    )
    ap.add_argument("--pcd", default="pcd_file.pcd", help="Path to .pcd")
    ap.add_argument("--k_plane", type=int, default=50, help="k-NN for local plane fit (surface-interior-cg calc)")
    ap.add_argument("--k_proj", type=int, default=40, help="k-NN for local plane fit used by radial-dir-midpoint projections")
    ap.add_argument("--voxel", type=float, default=0.0, help="Optional voxel downsample size (0=off)")
    ap.add_argument("--paint_black", action="store_true", help="Render all points black")
    ap.add_argument("--allow_backward", action="store_true", help="Allow plane intersection behind cg")
    ap.add_argument("--n_mid", type=int, default=10, help="Number of radial-dir-midpoints per point-(id)")
    ap.add_argument("--spline_samples", type=int, default=200, help="Samples along unfold-spline")
    ap.add_argument("--spline_color", type=float, nargs=3, default=[1.0, 1.0, 0.0], help="RGB for unfold-spline polylines")
    ap.add_argument("--show_green_to_orange", dest="show_green_to_orange", action="store_true",
                    help="Show dashed lines from point-(id) to unfolded-/radial-projection-point-(id) (default: on)")
    ap.add_argument("--no-show_green_to_orange", dest="show_green_to_orange", action="store_false",
                    help="Hide dashed lines from point-(id) to unfolded-/radial-projection-point-(id)")
    ap.add_argument("--n_greens", type=int, default=10, help="How many point-(id) to draw")

    # NEW: visualization/debug toggle
    ap.add_argument("--vis-debug", type=lambda x: str(x).lower() in ["true","1","yes"],
                    default=True, help="Enable visualization/debug artifacts (default: true)")

    ap.set_defaults(show_green_to_orange=True)
    args = ap.parse_args()

    # CONFIG update
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
    pcd = o3d.io.read_point_cloud(args.pcd)
    if pcd is None or len(pcd.points) == 0:
        raise SystemExit(f"Failed to read any points from: {args.pcd}")

    if args.voxel > 0.0:
        pcd = pcd.voxel_down_sample(args.voxel)

    if args.paint_black or not pcd.has_colors():
        pcd.paint_uniform_color([0.0, 0.0, 0.0])

    pts = finite_points_from_pcd(pcd)
    if pts.shape[0] != len(pcd.points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color([0.0, 0.0, 0.0])

    # Step 1: surface-interior-cg via cg + local plane hit
    surface_interior_cg, aux1 = find_surface_mass_anchor(pts)
    cg = aux1["cg"]

    # Log base positions (only when debugging)
    def fmt(p): return "x={:.6f}, y={:.6f}, z={:.6f}".format(p[0], p[1], p[2])
    if CONFIG.vis_debug:
        print("cg:                  " + fmt(cg))
        print("surface-interior-cg: " + fmt(surface_interior_cg))

    # Step 2: choose distinct random point-(id) not equal to surface-interior-cg
    points = []
    used_idx = set()
    for _ in range(CONFIG.n_greens * 3):
        gi = random.randrange(pts.shape[0])
        if gi in used_idx:
            continue
        g = pts[gi]
        if np.linalg.norm(g - surface_interior_cg) <= 1e-12:
            continue
        points.append(g)
        used_idx.add(gi)
        if len(points) >= CONFIG.n_greens:
            break
    if CONFIG.vis_debug and len(points) < CONFIG.n_greens:
        print(f"[WARN] Only found {len(points)} unique point-(id) (requested {CONFIG.n_greens}).")

    # Step 3: loop over points
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
