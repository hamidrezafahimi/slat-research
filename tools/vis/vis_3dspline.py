#!/usr/bin/env python3
"""
visualize_spline.py — read control points from CSV and render the B-spline surface
with the exact same visualization (materials/texturing) as app.py.

Usage:
  python visualize_spline.py --spline_data spline_ctrl.csv
                             [--samples_u 40] [--samples_v 40]
                             [--show_ctrl] [--cloud cloud.pcd]
"""

import argparse
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

# ---------- B-spline helpers (clamped uniform cubic) ----------
def clamped_uniform_knot_vector(n_ctrl: int, degree: int):
    p = degree
    m = n_ctrl + p + 1
    kv = np.zeros(m, dtype=float)
    kv[:p+1] = 0.0
    kv[-(p+1):] = 1.0
    interior_count = n_ctrl - p - 1
    if interior_count > 0:
        interior = np.linspace(0.0, 1.0, interior_count + 2)[1:-1]
        kv[p+1 : m-(p+1)] = interior
    return kv

def bspline_basis_all(n_ctrl: int, degree: int, kv: np.ndarray, t: float):
    p = degree
    lo, hi = kv[p], kv[-p-1]
    if t <= lo: t = lo + 1e-12
    if t >= hi: t = hi - 1e-12

    N = np.zeros(n_ctrl)
    tmp = np.zeros(len(kv) - 1, dtype=float)
    for j in range(len(tmp)):
        tmp[j] = 1.0 if (kv[j] <= t < kv[j+1]) else 0.0
    for d in range(1, p+1):
        for j in range(len(tmp) - d):
            left = right = 0.0
            dl = kv[j+d] - kv[j]
            dr = kv[j+d+1] - kv[j+1]
            if dl > 0:
                left = (t - kv[j]) / dl * tmp[j]
            if dr > 0:
                right = (kv[j+d+1] - t) / dr * tmp[j+1]
            tmp[j] = left + right
    N[:n_ctrl] = tmp[:n_ctrl]
    return N

def sample_bspline_surface(ctrl_pts, gw, gh, samples_u=40, samples_v=40):
    p = q = 3
    U = clamped_uniform_knot_vector(gw, p)
    V = clamped_uniform_knot_vector(gh, q)

    us = np.linspace(0, 1, samples_u)
    vs = np.linspace(0, 1, samples_v)
    Bu = np.stack([bspline_basis_all(gw, p, U, u) for u in us], axis=0)  # (Mu, gw)
    Bv = np.stack([bspline_basis_all(gh, q, V, v) for v in vs], axis=0)  # (Mv, gh)

    P = ctrl_pts.reshape(gh, gw, 3)

    S = np.zeros((samples_v, samples_u, 3), dtype=float)
    for k in range(3):
        Gk = P[..., k]
        inner_u = np.tensordot(Bu, Gk.T, axes=(1,1))
        S[..., k] = np.tensordot(Bv, inner_u.T, axes=(1,0))

    verts = S.reshape(-1, 3)
    tris = []
    for j in range(samples_v - 1):
        for i in range(samples_u - 1):
            a = j * samples_u + i
            b = a + 1
            c = a + samples_u
            d = c + 1
            tris.append([a, c, b])
            tris.append([b, c, d])
    tris = np.asarray(tris, dtype=np.int32)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(tris)
    mesh.compute_vertex_normals()
    return mesh

# ---------- CSV + grid inference ----------
def load_ctrl_points_csv(path: str) -> np.ndarray:
    try:
        arr = np.loadtxt(path, delimiter=",")
    except Exception:
        arr = np.loadtxt(path)
    arr = np.atleast_2d(arr)
    if arr.shape[1] != 3:
        raise ValueError(f"Expected Nx3 CSV, got shape {arr.shape}")
    return arr.astype(float)

def infer_grid_wh_from_points(pts: np.ndarray):
    xs, ys = pts[:,0], pts[:,1]
    rx = np.round(xs / (np.max(np.abs(xs)) or 1.0) * 1e8)
    ry = np.round(ys / (np.max(np.abs(ys)) or 1.0) * 1e8)
    gw = len(np.unique(rx))
    gh = len(np.unique(ry))
    n = pts.shape[0]
    if gw * gh != n:
        facs = [(w, n // w) for w in range(2, n + 1) if n % w == 0]
        facs.sort(key=lambda ab: abs(ab[0] - gw))
        if not facs:
            raise ValueError("Cannot infer a rectangular grid from points.")
        gw, gh = facs[0]
    return int(gw), int(gh)

# ---------- Viewer (matches app.py look) ----------
class Viewer:
    def __init__(self, mesh, ctrl_pts=None, cloud=None):
        self.window = gui.Application.instance.create_window("Spline Viewer", 1000, 750)
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene)

        # Materials to mirror app.py
        self.surf_mat = rendering.MaterialRecord()
        self.surf_mat.shader = "defaultLit"
        self.surf_mat.base_color = (0.7, 0.7, 0.9, 1.0)
        self.surf_mat.base_metallic = 0.0
        self.surf_mat.base_roughness = 0.8
        self.surf_mat.base_reflectance = 0.5

        self.points_mat = rendering.MaterialRecord()
        self.points_mat.shader = "defaultUnlit"
        self.points_mat.point_size = 10.0

        self.ext_mat = rendering.MaterialRecord()
        self.ext_mat.shader = "defaultUnlit"
        self.ext_mat.point_size = 4.0

        # Add geometries
        self.scene.scene.add_geometry("spline_surf", mesh, self.surf_mat)

        geoms_for_fit = [mesh]

        if ctrl_pts is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(ctrl_pts)
            pcd.colors = o3d.utility.Vector3dVector(
                np.tile(np.array([[0.2, 0.8, 1.0]]), (len(ctrl_pts), 1))
            )
            self.scene.scene.add_geometry("ctrl_pts", pcd, self.points_mat)
            geoms_for_fit.append(pcd)

        if cloud is not None:
            ext = o3d.io.read_point_cloud(cloud)
            if not ext.is_empty():
                if not ext.has_colors():
                    ext.paint_uniform_color([0.5, 0.5, 0.5])
                self.scene.scene.add_geometry("external_pcd", ext, self.ext_mat)
                geoms_for_fit.append(ext)
            else:
                print(f"[WARN] Cloud '{cloud}' has no points; skipping.")

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        axis_mat = rendering.MaterialRecord()
        axis_mat.shader = "defaultLit"
        self.scene.scene.add_geometry("axis", axis, axis_mat)
        geoms_for_fit.append(axis)

        # Camera fit to all
        mins = []
        maxs = []
        for g in geoms_for_fit:
            aabb = g.get_axis_aligned_bounding_box()
            mins.append(np.asarray(aabb.min_bound))
            maxs.append(np.asarray(aabb.max_bound))
        mins = np.vstack(mins).min(axis=0)
        maxs = np.vstack(maxs).max(axis=0)
        big = o3d.geometry.AxisAlignedBoundingBox(mins, maxs)
        self.scene.setup_camera(60, big, big.get_center())

        # Close on Q
        self.window.set_on_key(self._on_key)

    def _on_key(self, event):
        if event.type == gui.KeyEvent.Type.DOWN and event.key == gui.KeyName.Q:
            gui.Application.instance.quit()
            return True
        return False

    def run(self):
        gui.Application.instance.run()

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spline_data", required=True, help="Path to spline_ctrl.csv (Nx3).")
    ap.add_argument("--samples_u", type=int, default=40, help="Samples along u.")
    ap.add_argument("--samples_v", type=int, default=40, help="Samples along v.")
    ap.add_argument("--show_ctrl", action="store_true", help="Also render the control points.")
    ap.add_argument("--cloud", type=str, default=None, help="Optional external PCD/PLY/XYZ file.")
    args = ap.parse_args()

    ctrl = load_ctrl_points_csv(args.spline_data)
    gw, gh = infer_grid_wh_from_points(ctrl)
    print(f"[OK] Loaded {len(ctrl)} ctrl points. Inferred grid: {gw}×{gh}")

    mesh = sample_bspline_surface(ctrl, gw, gh, samples_u=args.samples_u, samples_v=args.samples_v)

    # Start GUI with app.py-style rendering
    gui.Application.instance.initialize()
    viewer = Viewer(mesh, ctrl_pts=ctrl if args.show_ctrl else None, cloud=args.cloud)
    viewer.run()

if __name__ == "__main__":
    main()
