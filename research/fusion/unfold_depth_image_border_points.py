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
- Border-selected (i,j) from HxW grid by default; can cap with --n_greens.
"""

import argparse
import time
import numpy as np
import os, sys
import os, sys
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(DIR_PATH + "/../../lib")
from unfold_helper import *  # brings CONFIG, math/vis helpers, and border_grid_indices
from typing import Optional, Dict

# ---------- project3DAndScale + Pose (same as in your depth2pcd flow) ----------
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(DIR_PATH + "/../../lib")
try:
    from kinematics.pose import Pose
    from mapper3D_helper import project3DAndScale
except Exception as e:
    raise SystemExit(f"Required modules not found: {e}\n"
                     f"Make sure Pose and project3DAndScale are importable like in depth2pcd.py")

# -------------------- CLI main (keeps original log/vis behavior) --------------------
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
    ap.add_argument("--n_greens", type=int, default=0, help="Cap number of border points (0 = use all).")
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

    # CONFIG update (unchanged)
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
    pose_kwargs = dict(
        x=args.pose_x, y=args.pose_y, z=args.pose_z,
        roll=args.pose_roll, pitch=args.pose_pitch, yaw=args.pose_yaw
    )

    # Run the high-level API
    max_pts = args.n_greens if args.n_greens and args.n_greens > 0 else None

    # 1) Load metric depth
    metric_depth = np.loadtxt(args.metric_depth, delimiter=',', dtype=np.float32)
    # 2) Build pose & HxWx3 via project3DAndScale
    pose = Pose(**{
        "x":  pose_kwargs.get("x", 0.0),
        "y":  pose_kwargs.get("y", 0.0),
        "z":  pose_kwargs.get("z", 9.4),
        "roll":  pose_kwargs.get("roll", 0.0),
        "pitch": pose_kwargs.get("pitch", -0.78),
        "yaw":   pose_kwargs.get("yaw", 0.0),
    })
    xyz_img = project3DAndScale(metric_depth, pose, args.hfov_deg, metric_depth.shape)
    unfolded_pts, extras = unfold_image_border_points(
        xyz_img=xyz_img,
        hfov_deg=args.hfov_deg,
        pose_kwargs=pose_kwargs,
        max_points=max_pts
    )

    # Print cg / surface-interior-cg only when debugging (those lines are printed inside API)
    print("Done in %.2f ms" % (1000.0 * (time.time() - t0)))

    # Visualize once for all points (only when debugging)
    if CONFIG.vis_debug:
        visualize_scene_multi(
            pcd=extras["pcd"],
            cg=extras["cg"],
            surface_interior_cg=extras["surface_interior_cg"],
            items=extras["items"],
            spline_color=CONFIG.spline_color,
            show_green_to_orange=CONFIG.show_green_to_orange
        )

if __name__ == "__main__":
    main()
