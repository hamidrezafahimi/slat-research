#!/usr/bin/env python3
"""
Unfold image *borders*:

1) Compute orange (unfolded) points for a uniform subset of border pixels (same math as before).
2) Fit four cubic splines (top/bottom/left/right) through those orange points.
3) For EVERY border pixel, suggest an orange position by evaluating the respective spline
   at its normalized coordinate t in [0,1]; draw these as tiny red dots on plane G.

All visualization and per-point logs respect --vis-debug (default: true).
"""

import argparse
import time
import os, sys
import numpy as np
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(DIR_PATH + "/../../lib")
from unfold_helper import *
try:
    from kinematics.pose import Pose  # noqa: F401 (only to check import; helper uses lazy inside)
    from mapper3D_helper import project3DAndScale  # noqa: F401
except Exception as e:
    raise SystemExit(f"Required modules not found: {e}\n"
                     f"Make sure Pose and project3DAndScale are importable like in depth2pcd.py")
import json

def main():
    ap = argparse.ArgumentParser(
        description="Unfold image borders with orange splines and tiny red suggestions on plane G."
    )
    ap.add_argument("--metric_depth", required=True, help="Path to metrics_depth.csv")

    # Core params (unchanged)
    ap.add_argument("--k_plane", type=int, default=5)
    ap.add_argument("--k_proj", type=int, default=4)
    ap.add_argument("--allow_backward", action="store_true")
    ap.add_argument("--n_mid", type=int, default=10)
    ap.add_argument("--spline_samples", type=int, default=200)
    ap.add_argument("--spline_color", type=float, nargs=3, default=[1.0, 1.0, 0.0])
    ap.add_argument("--show_green_to_orange", dest="show_green_to_orange", action="store_true")
    ap.add_argument("--no-show_green_to_orange", dest="show_green_to_orange", action="store_false")
    ap.add_argument("--n_greens", type=int, default=0, help="Cap number of sampled border points (0 = use all from uniform grid).")

    # Camera/pose params
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

    # Wire CONFIG
    CONFIG.k_plane = args.k_plane
    CONFIG.k_proj = args.k_proj
    CONFIG.allow_backward = args.allow_backward
    CONFIG.n_mid = args.n_mid
    CONFIG.spline_samples = args.spline_samples
    CONFIG.spline_color = args.spline_color
    CONFIG.show_green_to_orange = args.show_green_to_orange
    CONFIG.vis_debug = args.vis_debug

    t0 = time.time()
    pose_kwargs = dict(
        x=args.pose_x, y=args.pose_y, z=args.pose_z,
        roll=args.pose_roll, pitch=args.pose_pitch, yaw=args.pose_yaw
    )
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
    # Run new borders API
    result, extras = unfold_image_borders(
        xyz_img=xyz_img,
        hfov_deg=args.hfov_deg,
        pose_kwargs=pose_kwargs,
        max_points=max_pts
    )

    # --- NEW: save dense_red poses as JSON ---
    out_json = os.path.splitext(args.metric_depth)[0] + "_dense_red.json"

    # Convert numpy arrays to lists for JSON serialization
    dense_red_serializable = {
        side: arr.tolist() if arr is not None else None
        for side, arr in result["dense_red"].items()
    }

    with open(out_json, "w") as f:
        json.dump(dense_red_serializable, f, indent=2)

    print(f"Saved dense_red poses to {out_json}")

    # Minimal console timing (heavier per-point logs stay guarded by vis_debug)
    print("Done in %.2f ms" % (1000.0 * (time.time() - t0)))

    # Visualize (orange sample items + tiny red dense suggestions)
    if CONFIG.vis_debug:
        visualize_scene_borders(
            pcd=extras["pcd"],
            cg=extras["cg"],
            surface_interior_cg=extras["surface_interior_cg"],
            items=extras["items"],
            red_points_by_side=result["dense_red"],
            spline_color=CONFIG.spline_color,
            show_green_to_orange=CONFIG.show_green_to_orange
        )

if __name__ == "__main__":
    main()
