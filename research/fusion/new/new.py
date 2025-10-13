import argparse
import numpy as np
import open3d as o3d
import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../../lib")
from geom.rectification import rectify_xy_proj


# ----------------------------- I/O helpers -----------------------------
def ensure_colors(pcd: o3d.geometry.PointCloud, default_rgb=(0.6, 0.6, 0.6)) -> None:
    """Guarantee the PCD has colors; if not, fill with a default."""
    if not pcd.has_colors():
        n = np.asarray(pcd.points).shape[0]
        pcd.colors = o3d.utility.Vector3dVector(np.tile(default_rgb, (n, 1)))


def pcd2xyz_grid(pcd: o3d.geometry.PointCloud, H: int, W: int) -> np.ndarray:
    """Return an H×W×3 numpy array of XYZ from a row-major point cloud."""
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.shape[0] != H * W:
        raise ValueError(f"Point cloud has {pts.shape[0]} points, but expected {H*W}.")
    return pts.reshape((H, W, 3))


def pcd2rgb_grid(pcd: o3d.geometry.PointCloud, H: int, W: int) -> np.ndarray:
    """Return an H×W×3 numpy array of RGB in [0,1] (row-major)."""
    cols = np.asarray(pcd.colors, dtype=np.float32)
    if cols.shape[0] != H * W:
        raise ValueError(f"Point cloud has {cols.shape[0]} points, but expected {H*W}.")
    return cols.reshape((H, W, 3))


# ----------------------------- Visualization helpers -----------------------------
def mark_marginals_inplace(pcd_like: o3d.geometry.PointCloud, H: int, W: int) -> None:
    """Color x-marginals (left/right) red; y-marginals (top/bottom) blue; corners magenta."""
    n = H * W
    colors = np.asarray(pcd_like.colors, dtype=float)
    if colors.shape[0] != n:
        raise ValueError(f"Cloud has {colors.shape[0]} points; expected {n} (H*W).")
    rows = np.repeat(np.arange(H), W)
    cols = np.tile(np.arange(W), H)
    x_mask = (cols == 0) | (cols == W - 1)
    y_mask = (rows == 0) | (rows == H - 1)
    both = x_mask & y_mask
    colors[y_mask] = np.array([0.0, 0.0, 1.0])
    colors[x_mask] = np.array([1.0, 0.0, 0.0])
    colors[both]   = np.array([1.0, 0.0, 1.0])
    pcd_like.colors = o3d.utility.Vector3dVector(colors)


def build_corner_lineset(xy_flat: np.ndarray, H: int, W: int, draw_diagonals: bool) -> o3d.geometry.LineSet:
    """Colored rectangle (and optional diagonals) using XY in row-major order."""
    TL, TR, BL, BR = 0, W - 1, (H - 1) * W, H * W - 1
    corner_pts = np.vstack([xy_flat[TL], xy_flat[TR], xy_flat[BR], xy_flat[BL]])  # TL, TR, BR, BL
    corner_pts3 = np.hstack([corner_pts, np.zeros((4, 1))])  # lines at z=0 (reference)
    lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
    if draw_diagonals:
        lines += [[0, 2], [1, 3]]
        colors += [[1, 0, 1], [0, 1, 1]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corner_pts3)
    ls.lines  = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=float))
    return ls

# ----------------------------- CLI / Visualization -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Rectify XY projection (equalize row/column adjacent distances) and visualize with original Z."
    )
    ap.add_argument("--in", dest="inp", required=True, help="Input .pcd path")
    ap.add_argument("--out", dest="out", default=None, help="(Optional) Save rectified PCD to this path")
    ap.add_argument("--H", type=int, required=True, help="Image height")
    ap.add_argument("--W", type=int, required=True, help="Image width")
    ap.add_argument("--frame", type=float, default=1.0, help="Coordinate frame size")
    ap.add_argument("--draw_diagonals", action="store_true", help="Draw diagonals between opposite corners")
    args = ap.parse_args()

    # Load cloud
    pcd = o3d.io.read_point_cloud(args.inp)
    if pcd.is_empty():
        raise RuntimeError(f"Loaded empty point cloud from: {args.inp}")
    ensure_colors(pcd, default_rgb=(0.55, 0.55, 0.55))

    # Build grids
    xyz_grid = pcd2xyz_grid(pcd, args.H, args.W)    # H×W×3
    rgb_grid = pcd2rgb_grid(pcd, args.H, args.W)    # H×W×3

    # --- Core rectification (returns H×W×3) ---
    rect_grid = rectify_xy_proj(xyz_grid)           # H×W×3

    # Prepare output point cloud for visualization/saving
    rect_pts_flat = rect_grid.reshape(-1, 3)
    colors_flat   = rgb_grid.reshape(-1, 3)

    proj = o3d.geometry.PointCloud()
    proj.points = o3d.utility.Vector3dVector(rect_pts_flat)
    proj.colors = o3d.utility.Vector3dVector(colors_flat)
    mark_marginals_inplace(proj, args.H, args.W)

    # Colored rectangle overlay (built from rectified XY; at z=0 for reference)
    xy_rect_flat = rect_pts_flat[:, :2]
    corner_lines = build_corner_lineset(xy_rect_flat, args.H, args.W, args.draw_diagonals)

    # Optional save
    if args.out is not None:
        ok = o3d.io.write_point_cloud(args.out, proj)
        if not ok:
            raise RuntimeError(f"Failed to write rectified PCD to: {args.out}")
        print(f"[✔] Saved rectified PCD → {args.out}")

    # Visualize
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.frame, origin=[0, 0, 0])
    o3d.visualization.draw_geometries(
        [proj, corner_lines, frame],
        window_name="Rectified XY (columns & rows equalized) • Z inherited",
        width=1280, height=800
    )


if __name__ == "__main__":
    main()
