#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../../lib")
from fusion.helper import depyramidize_pointCloud

ORIGIN = np.array([0.0, 0.0, 0.0], dtype=np.float64)

# ----------------------- Viz helpers --------------------
def parse_color(s: str):
    r, g, b = map(float, s.split(","))
    return [r, g, b]

def make_xy_plane_mesh(xmin, xmax, ymin, ymax, z=0.0, color=(0.7, 0.7, 0.7)):
    verts = np.array(
        [[xmin, ymin, z],
         [xmax, ymin, z],
         [xmax, ymax, z],
         [xmin, ymax, z]], dtype=np.float64
    )
    tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts),
        triangles=o3d.utility.Vector3iVector(tris),
    )
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh

def show_geometries_cross_version(geoms, title: str, point_size: int = 2):
    try:
        o3d.visualization.draw(geoms, title=title, point_size=int(point_size))
        return
    except Exception:
        pass
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1600, height=1000)
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    opt.background_color = np.array([1.0, 1.0, 1.0])
    vis.run()
    vis.destroy_window()
# -------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Depyramidize a whole point cloud (vectorized) and visualize.")
    ap.add_argument("--in", dest="inp", required=True, help="Input .pcd/.ply")
    ap.add_argument("--plane-pad", type=float, default=0.05,
                    help="Pad fraction to extend the plane across XY bounds of cloud & intersections.")
    ap.add_argument("--dep-color", default="0.2,1.0,0.2", help="RGB for depyramidized points (dep_points)")
    ap.add_argument("--origline-color", default="1.0,1.0,0.2", help="RGB for dep-origin points (dep_origins)")
    ap.add_argument("--point-size", type=int, default=2, help="Viewer point size (int)")
    ap.add_argument("--save-dir", default="", help="Optional dir to save outputs: dep_points.pcd, dep_origins.pcd")
    args = ap.parse_args()

    # Load cloud
    inp = Path(args.inp)
    if not inp.exists():
        raise FileNotFoundError(str(inp))
    pcd = o3d.io.read_point_cloud(str(inp))
    if not pcd.has_colors():
        pcd.paint_uniform_color([0.85, 0.85, 0.85])
    P = np.asarray(pcd.points)

    # Core depyramidization
    dep_points, dep_origins, info = depyramidize_pointCloud(P)

    # Build clouds
    dep_pcd = o3d.geometry.PointCloud()
    dep_pcd.points = o3d.utility.Vector3dVector(dep_points)
    dep_pcd.paint_uniform_color(parse_color(args.dep_color))

    dep_origin_pcd = o3d.geometry.PointCloud()
    dep_origin_pcd.points = o3d.utility.Vector3dVector(dep_origins)
    dep_origin_pcd.paint_uniform_color(parse_color(args.origline_color))

    # Plane mesh covering original XY and intersections XY (with padding)
    I = info["blue"]
    xy_all = np.vstack([P[:, :2], I[:, :2]])
    xy_min = xy_all.min(axis=0)
    xy_max = xy_all.max(axis=0)
    pad = max(args.plane_pad, 0.0)
    xpad = pad * max(xy_max[0]-xy_min[0], 1e-6)
    ypad = pad * max(xy_max[1]-xy_min[1], 1e-6)
    plane_mesh = make_xy_plane_mesh(
        xy_min[0]-xpad, xy_max[0]+xpad, xy_min[1]-ypad, xy_max[1]+ypad,
        z=float(np.min(P[:, 2])), color=(0.7, 0.7, 0.7)
    )

    # Axes
    bb = np.vstack([P, dep_points, dep_origins])
    mn, mx = bb.min(axis=0), bb.max(axis=0)
    axes_size = max(1e-3, 0.1 * float((mx - mn).max()))
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axes_size, origin=[0, 0, 0])

    # Stats
    p2p = info["point2plane"]
    p2o = info["plane2origin"]  # <— CHANGED
    print(f"point2plane:   min={p2p.min():.6f}, mean={p2p.mean():.6f}, max={p2p.max():.6f}")
    print(f"plane2origin:  min={p2o.min():.6f}, mean={p2o.mean():.6f}, max={p2o.max():.6f}")

    # Viz
    show_geometries_cross_version(
        [plane_mesh, axes, pcd, dep_pcd, dep_origin_pcd],
        "Depyramidized clouds (plane2origin) + original",
        point_size=args.point_size
    )

if __name__ == "__main__":
    main()
