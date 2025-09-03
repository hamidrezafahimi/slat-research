#!/usr/bin/env python3
import os, sys, argparse, time
import numpy as np
import open3d as o3d

# Project + pose like your other flows
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(DIR_PATH + "/../../lib")
from kinematics.pose import Pose
from mapper3D_helper import project3DAndScale

# Pull helpers + CONFIG + visual primitives
from unfold_helper import (  # noqa: E402
    CONFIG,
    unfold_surface,            # full-surface interpolation
    unfold_image_borders,      # dense red + green→orange borders
    make_sphere, make_polyline, make_lineset,
    _make_plane_G, _make_dashed_line,
)

def _make_pcd(xyz: np.ndarray, color=(0, 0, 0), subsample=1, voxel=0.0):
    """Generic point cloud builder with optional subsampling/voxelization."""
    pts = xyz.reshape(-1, 3)
    if subsample and subsample > 1:
        pts = pts[::subsample]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float32))
    if voxel and voxel > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(voxel))
    if color is not None:
        col = np.tile(np.asarray(color, dtype=np.float32), (len(pcd.points), 1))
        pcd.colors = o3d.utility.Vector3dVector(col)
    return pcd

def _pack_all_red_dense_as_pcd(dense_red_dict):
    chunks = []
    for s in ("top", "right", "bottom", "left"):
        if s in dense_red_dict and dense_red_dict[s] is not None:
            Y = np.asarray(dense_red_dict[s], dtype=float)
            if Y.size: chunks.append(Y)
    if not chunks:
        return None
    reds = np.vstack(chunks)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(reds.astype(np.float32))
    pcd.paint_uniform_color([1.0, 0.0, 0.0])  # tiny red
    return pcd

def main():
    ap = argparse.ArgumentParser(
        description="Unfold full surface + visualize xyz_img (black), unfolded (light green), dense red borders, and green→orange flows."
    )
    ap.add_argument("--metric_depth", required=True, help="Path to metrics_depth.csv")

    # Core params
    ap.add_argument("--k_plane", type=int, default=5)
    ap.add_argument("--k_proj", type=int, default=4)
    ap.add_argument("--allow_backward", action="store_true")
    ap.add_argument("--n_mid", type=int, default=10)
    ap.add_argument("--spline_samples", type=int, default=200)
    ap.add_argument("--spline_color", type=float, nargs=3, default=[1.0, 1.0, 0.0])
    ap.add_argument("--show_green_to_orange", dest="show_green_to_orange", action="store_true")
    ap.add_argument("--no-show_green_to_orange", dest="show_green_to_orange", action="store_false")
    ap.add_argument("--n_greens", type=int, default=0,
                    help="Cap number of sampled border points (0 = default set).")

    # Camera/pose
    ap.add_argument("--hfov_deg", type=float, default=66.0)
    ap.add_argument("--pose_x", type=float, default=0.0)
    ap.add_argument("--pose_y", type=float, default=0.0)
    ap.add_argument("--pose_z", type=float, default=9.4)
    ap.add_argument("--pose_roll", type=float, default=0.0)
    ap.add_argument("--pose_pitch", type=float, default=-0.78)
    ap.add_argument("--pose_yaw", type=float, default=0.0)

    # Viz & perf
    ap.add_argument("--vis-debug", type=lambda x: str(x).lower() in ["true", "1", "yes"],
                    default=True, help="Enable visualization/debug artifacts")
    ap.add_argument("--subsample", type=int, default=1, help="Subsample for unfolded surface")
    ap.add_argument("--voxel", type=float, default=0.0, help="Voxel size for unfolded surface")

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

    # Load metric depth & project to xyz_img
    metric_depth = np.loadtxt(args.metric_depth, delimiter=',', dtype=np.float32)
    pose = Pose(x=args.pose_x, y=args.pose_y, z=args.pose_z,
                roll=args.pose_roll, pitch=args.pose_pitch, yaw=args.pose_yaw)
    xyz_img = project3DAndScale(metric_depth, pose, args.hfov_deg, metric_depth.shape)

    # A) full-surface interpolation (light green)
    t0 = time.time()
    unfolded_surface, ctx = unfold_surface(
        xyz_img=xyz_img,
        hfov_deg=args.hfov_deg,
        pose_kwargs=dict(x=args.pose_x, y=args.pose_y, z=args.pose_z,
                         roll=args.pose_roll, pitch=args.pose_pitch, yaw=args.pose_yaw)
    )
    tA = 1000.0 * (time.time() - t0)

    # B) borders API (dense reds + sampled border→orange)
    max_pts = args.n_greens if args.n_greens and args.n_greens > 0 else None
    t1 = time.time()
    borders, extras = unfold_image_borders(
        xyz_img=xyz_img,
        hfov_deg=args.hfov_deg,
        pose_kwargs=dict(x=args.pose_x, y=args.pose_y, z=args.pose_z,
                         roll=args.pose_roll, pitch=args.pose_pitch, yaw=args.pose_yaw),
        max_points=max_pts
    )
    tB = 1000.0 * (time.time() - t1)

    print(f"[Timing] unfold_surface: {tA:.2f} ms, unfold_image_borders: {tB:.2f} ms")

    if not CONFIG.vis_debug:
        return

    draw = []

    # Anchors & plane G
    pcd_black_src = extras["pcd"]
    cg = extras["cg"]
    sicg = extras["surface_interior_cg"]
    bbox = pcd_black_src.get_axis_aligned_bounding_box()
    plane_G = _make_plane_G(bbox, sicg[2], color=[0.2, 0.6, 1.0])
    draw.append(plane_G)
    diag = np.linalg.norm(bbox.get_extent()) if bbox is not None else 1.0
    draw.append(make_sphere(cg, max(1e-6, 0.0020 * diag), [1.0, 0.0, 0.0]))
    draw.append(make_sphere(sicg, max(1e-6, 0.0018 * diag), [0.0, 0.0, 1.0]))

    # 1) Initial xyz_img in black
    pcd_initial = _make_pcd(xyz_img, color=(0, 0, 0), subsample=args.subsample, voxel=args.voxel)
    draw.append(pcd_initial)

    # 2) Unfolded surface in light green
    pcd_unfold = _make_pcd(unfolded_surface, color=(0.5, 1.0, 0.5),
                           subsample=args.subsample, voxel=args.voxel)
    draw.append(pcd_unfold)

    # 3) Dense red border points
    red_pcd = _pack_all_red_dense_as_pcd(borders["dense_red"])
    if red_pcd is not None:
        draw.append(red_pcd)

    # 4) Green→Orange samples & splines
    r_point = max(1e-6, 0.0018 * diag)
    r_mid_purple = max(1e-6, 0.0015 * diag)
    r_mid_yellow = max(1e-6, 0.0015 * diag)
    r_unfolded = max(1e-6, 0.0018 * diag)
    dash_len = 0.02 * diag
    gap_len  = 0.015 * diag

    for it in extras["items"]:
        point_i = it["point"]
        mids_purple = it.get("mids", np.empty((0, 3)))
        mids_yellow = it.get("proj_points", np.empty((0, 3)))
        curve = it.get("curve", None)
        unfolded_i = it.get("unfolded", None)
        radial_proj_i = it.get("radial_proj", None)

        # green start point
        draw.append(make_sphere(point_i, r_point, [0.0, 1.0, 0.0]))
        draw.append(make_lineset(np.vstack([sicg, point_i]),
                                 np.array([[0, 1]], np.int32), [1.0, 1.0, 1.0]))

        # purple & yellow mids
        for M in mids_purple:
            draw.append(make_sphere(M, r_mid_purple, [1.0, 0.0, 1.0]))
        for P in mids_yellow:
            draw.append(make_sphere(P, r_mid_yellow, [1.0, 1.0, 0.0]))

        # yellow spline
        if curve is not None and curve.shape[0] >= 2:
            ls = make_polyline(curve, CONFIG.spline_color)
            if ls is not None: draw.append(ls)

        # orange unfolded point + dashed link
        if unfolded_i is not None:
            draw.append(make_sphere(unfolded_i, r_unfolded, [1.0, 0.5, 0.0]))
            if CONFIG.show_green_to_orange:
                ds = _make_dashed_line(point_i, unfolded_i, [1.0, 0.5, 0.0], dash_len, gap_len)
                if ds is not None: draw.append(ds)

        # grey radial projection + dashed link
        if radial_proj_i is not None:
            draw.append(make_sphere(radial_proj_i, r_unfolded, [0.6, 0.6, 0.6]))
            if CONFIG.show_green_to_orange:
                ds_g = _make_dashed_line(point_i, radial_proj_i, [0.6, 0.6, 0.6], dash_len, gap_len)
                if ds_g is not None: draw.append(ds_g)

    # Show everything in one viewer
    o3d.visualization.draw_geometries(
        draw,
        window_name="xyz_img (black) + unfolded (light green) + dense red + green→orange",
        width=1400, height=900
    )

if __name__ == "__main__":
    main()
