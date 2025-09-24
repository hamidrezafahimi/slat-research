#!/usr/bin/env python3
import argparse, os, numpy as np, open3d as o3d
from helper import RandomSurfacer


# ------------- CLI demo -------------
def _demo():
    ap = argparse.ArgumentParser(description="Generate N random spline surfaces and visualize one.")
    ap.add_argument("--cloud", required=True, help="Point cloud (.pcd/.ply/.xyz/.npy/.npz)")
    ap.add_argument("--grid_w", type=int, default=6); ap.add_argument("--grid_h", type=int, default=4)
    ap.add_argument("--samples_u", type=int, default=40); ap.add_argument("--samples_v", type=int, default=40)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--num", type=int, default=3, help="How many random surfaces to generate")
    ap.add_argument("--viz", action="store_true", help="Visualize the last random surface as a mesh")
    args = ap.parse_args()

    rs = RandomSurfacer(args.cloud, args.grid_w, args.grid_h, args.samples_u, args.samples_v, args.margin)
    random_sets = rs.generate(args.num)
    print(f"Generated {len(random_sets)} random control grids; each shape: {(random_sets[0].shape if random_sets else None)}")
    print(f"Z bounds: [{rs.Z_down_scalar:.6f}, {rs.Z_up_scalar:.6f}]  (OF={rs.OF:.6f}, z0={rs.z0:.6f})")

    if args.viz and random_sets:
        rs.draw(random_sets[-1])

if __name__ == "__main__":
    _demo()
