#!/usr/bin/env python3
# Depth image from spline control grid using your camera & transformations libs
#
# - Camera: SimpleCamera (hfov=80°, image 1241×376) at (0,0,0)
# - Geometry: control-point grid loaded from ../diffusion/spline_ctrl.csv (x,y,z in NWU)
# - Transform: handled inside camera.project_3dTo2d_sp via your utils/transformations.py
# - Rendering: per-pixel ray/triangle intersection; each pixel stores metric range (m)
# - Output: CSV with float numbers; NaN where the ray misses the surface
#
# Files produced:
#   depth_spline_simplecamera_1241x376_hfov80.csv

import os
import sys
import math
from pathlib import Path
import numpy as np

# --- imports (classic style) ---
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../../lib"))

from sim.camera import SimpleCamera  # uses your updated camera.py internally

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py spline.csv")
        sys.exit(1)

    ctrl_path = sys.argv[1]

    # --- paths ---
    out_csv   = "depth_spline.csv"

    # --- load control points (x,y,z in NWU) ---
    pts = np.loadtxt(ctrl_path, delimiter=",", dtype=float)  # shape (N,3)

    # --- camera setup ---
    # W, H = 1241, 376
    W, H = 440, 330
    hfov_deg = 66.0
    cam = SimpleCamera((W, H), hfov_deg=hfov_deg, show=False)

    # Mandatory rpy for spline-projection (roll, pitch, yaw). Adjust if needed.
    # rpy = (0.0, -0.02, 0.0)  # radians; use degrees if you prefer, camera.py handles both
    rpy = (0.0, -0.78, 0.0)  # radians; use degrees if you prefer, camera.py handles both

    # --- rasterize spline control grid to per-pixel metric range ---
    # project_3dTo2d_sp returns a depth map (H, W), NaN where no hit.
    depth = cam.project_3dTo2d_sp(pts, rpy=rpy, cam_pose_xyz=(0.0, 0.0, 0.0))

    # --- save CSV (float numbers) ---
    np.savetxt(out_csv, depth.astype(np.float32), delimiter=",", fmt="%.6f")
    print(f"Saved depth CSV: {out_csv}  shape={depth.shape}, hfov={hfov_deg}°, image={W}x{H}")
