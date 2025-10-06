import argparse
import sys
import time
import numpy as np
import open3d as o3d

from helper import *
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../../lib")
from diffusion.tunning import Optimizer
from diffusion.scoring import Projection3DScorer
from diffusion.viz import ClassicViewer
from diffusion.config import BGPatternDiffuserConfig
from diffusion.helper import downsample_pcd
from geom.surfaces import cg_centeric_xy_spline, infer_grid, generate_spline

# ---------------- small utilities ----------------
def factor_pairs(n):
    pairs = []
    for a in range(1, int(np.sqrt(n)) + 1):
        if n % a == 0:
            pairs.append((a, n // a))
            if a != n // a:
                pairs.append((n // a, a))
    pairs.sort(key=lambda ab: (-ab[0], ab[1]))
    return pairs

def print_manual(args, N):
    if args.verbosity == "none":
        return
    print("\n=== Z-GD — Classic Viewer (no overlays) ===")
    print(" SPACE : single-step for current control point (updates one index)")
    print(" A     : run one full serial iteration (update all control points once), then stop")
    print(" M     : run one 'move-all' iteration (uniform Z shift; preserves shape)")
    print(" O     : toggle continuous optimization loop (start/stop). One full iteration per animation frame.")
    print(" J / K : prev / next control point")
    print(" R     : reset control points")
    print(" Q     : quit")
    print("-------------------------------------------")
    print(f" grid: {args.grid_w} x {args.grid_h}   N={N}")
    print(f" samples_u,v: {args.samples_u}, {args.samples_v}")
    print(f" continuous --max_iters: {args.iters}  --tol: {args.tol}")
    print(f" verbosity: {args.verbosity}   fast: {args.fast}")
    # Note: --move_all only affects --fast(headless) mode; GUI keys are always available.
    print("===========================================\n")

# ---------------- CLI & main ----------------
def main():
    ap = argparse.ArgumentParser(description="Classic Open3D viewer + terminal-logged Z-only optimization of control points.")
    ap.add_argument("--in", dest="inp", required=True, help="Input point cloud (.pcd/.ply/.xyz/.npy/.npz)")
    ap.add_argument("--initial_guess", required=True, help="CSV of control points (x,y,z)")
    ap.add_argument("--base_ctrl", type=str, help="CSV of control points to generate ground smoothness criteria")
    ap.add_argument("--samples_u", type=int, default=40)
    ap.add_argument("--samples_v", type=int, default=40)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--kmin_neighbors", type=int, default=8)
    ap.add_argument("--max_dz", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--eps", type=float, default=1e-3, help="Epsilon for central diff grad wrt z.")
    ap.add_argument("--iters", type=int, default=100, help="Headless: run N full iterations without viewer.")
    ap.add_argument("--max_iters", type=int, default=100, help="Max full iterations in continuous mode.")
    ap.add_argument("--tol", type=float, default=1e-8, help="Stopping threshold for continuous mode on max |alpha*d|.")
    ap.add_argument("--downsample", type=float, default=None, help="Stopping threshold for continuous mode on max |alpha*d|.")
    ap.add_argument("--verbosity", choices=["full", "tiny", "none"], default="full",
                    help="Terminal verbosity: full = detailed, tiny = only score+iter, none = silent")
    ap.add_argument("--fast", action="store_true", help="Fast headless mode: no viz, minimal logging, skip heavy ops.")
    ap.add_argument("--move_all", action="store_true",
        help="FAST mode only: if set, optimize a single global Z-offset each iteration (uniform displacement; shape preserved).",
    )

    args = ap.parse_args()

    # load cloud
    cloud_pts = None
    orig_cols = None
    pcd = o3d.io.read_point_cloud(args.inp)
    if pcd.is_empty():
        raise SystemExit("[ERROR] Empty point cloud.")
    if args.downsample:
        pcd = downsample_pcd(pcd, args.downsample)
    cloud_pts = np.asarray(pcd.points, dtype=float)
    if pcd.has_colors():
        orig_cols = np.asarray(pcd.colors, dtype=float)

    # load control net
    ctrl_pts = np.loadtxt(args.initial_guess, delimiter=",", dtype=float)
    if ctrl_pts.ndim != 2 or ctrl_pts.shape[1] != 3:
        raise SystemExit("[ERROR] --initial_guess must be (N,3) CSV of x,y,z.")

    # automatically infer grid size from x,y of ctrl points
    N = ctrl_pts.shape[0]

    args.grid_w, args.grid_h = infer_grid(ctrl_pts)
    if args.verbosity != "none":
        print(f"[auto] grid inferred from spline CSV: grid_w={args.grid_w} grid_h={args.grid_h} N={N}")

    cfg = BGPatternDiffuserConfig(
        hfov_deg=90.0,
        output_dir="",
        fast=args.fast
    )

    if args.base_ctrl:
        base_mesh, W, H = generate_spline(np.loadtxt(args.base_ctrl, delimiter=",", dtype=float),
                                          args.samples_u, args.samples_v)
        print(f"[auto] Base mesh generated from {args.base_ctrl}")
    else:
        # create centered-XY base mesh for scorer (never reset later)
        base_mesh, base_ctrl_grid, _, _, center_xy, z0 = cg_centeric_xy_spline(
            cloud_pts, args.grid_w, args.grid_h, args.samples_u, args.samples_v, margin=0.02
        )
        W, H = infer_grid(base_ctrl_grid)
        print(f"[auto] Base mesh automatically generated parallel to xy plane")


    # FAST (headless) path: here --move_all decides behavior
    scorer = Projection3DScorer(cfg)
    scorer.reset(cloud_pts, smoothness_base_mesh=base_mesh, max_dz=args.max_dz)
    if args.fast:
        optimizer = Optimizer(cfg)
        iters_done, final_score, final_z = optimizer.tune(ctrl_pts.copy(), scorer, iters=args.iters,
                                                          alpha=args.alpha)
        if args.verbosity != "none":
            mode = "uniform shift (--move_all)" if args.move_all else "per-point"
            print(f"[fast] mode={mode} done iters={iters_done} score={final_score:.6f}")
        print("Z vals for given ctrl: ")
        final = ctrl_pts.copy()
        final[:,2] = final_z
        np.savetxt(args.out, final, delimiter=',')
        print(f"Saved output into {args.out}")
    else:
        # GUI path: keys SPACE/A/O do per-point; M does uniform shift; unaffected by --move_all
        print_manual(args, len(ctrl_pts))
        ClassicViewer(cfg, cloud_pts, ctrl_pts, scorer, W, H, _alpha=args.alpha, iters=args.iters)


if __name__ == "__main__":
    main()
