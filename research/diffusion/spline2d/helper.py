import numpy as np
# from typing import Optional, Tuple, List, Dict, Union
# from dataclasses import dataclass
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os, numpy as np, open3d as o3d
# ---------- optional (faster KDTree + pdist) ----------
import numpy as np
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


from scipy.spatial import cKDTree
_SCIPY = True

# ------------- base “spline” scaffolding -------------
def _extent_wh_center_xy(cloud_pts: np.ndarray):
    mins = cloud_pts[:, :2].min(axis=0); maxs = cloud_pts[:, :2].max(axis=0)
    W = float(maxs[0] - mins[0]); H = float(maxs[1] - mins[1])
    center_xy = 0.5 * (mins + maxs)
    return W, H, center_xy

def _ctrl_points_grid(grid_w: int, grid_h: int, W: float, H: float, center_xy: np.ndarray, z0: float):
    xs = np.linspace(-0.5 * W, 0.5 * W, grid_w) + center_xy[0]
    ys = np.linspace(-0.5 * H, 0.5 * H, grid_h) + center_xy[1]
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    Z = np.full_like(X, z0, dtype=float)
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

def _flat_patch_mesh(center_xy: np.ndarray, W: float, H: float, z0: float, su: int, sv: int):
    su = max(2, int(su)); sv = max(2, int(sv))
    xs = np.linspace(-0.5 * W, 0.5 * W, su) + center_xy[0]
    ys = np.linspace(-0.5 * H, 0.5 * H, sv) + center_xy[1]
    X, Y = np.meshgrid(xs, ys, indexing="xy"); Z = np.full_like(X, z0, dtype=float)
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    tris = []
    for j in range(sv - 1):
        for i in range(su - 1):
            v0 = j * su + i; v1 = v0 + 1; v2 = v0 + su; v3 = v2 + 1
            tris.append([v0, v2, v1]); tris.append([v1, v2, v3])
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(np.asarray(verts, np.float64)),
        triangles=o3d.utility.Vector3iVector(np.asarray(tris, np.int32)),
    )
    mesh.compute_vertex_normals()
    return mesh

def build_surface_mesh(ctrl_points: np.ndarray, gw: int, gh: int,
                       samples_u: int = 40, samples_v: int = 40) -> o3d.geometry.TriangleMesh:
    verts, tris = sample_bspline_surface(ctrl_points, gw, gh, samples_u, samples_v)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(tris)
    mesh.compute_vertex_normals()
    return mesh

def sample_bspline_surface(ctrl_pts: np.ndarray, gw: int, gh: int,
                           samples_u: int = 40, samples_v: int = 40) -> tuple[np.ndarray, np.ndarray]:
    p = q = 3
    # Clamp degrees to grid
    p = min(p, gw - 1)
    q = min(q, gh - 1)

    U = clamped_uniform_knot_vector(gw, p)
    V = clamped_uniform_knot_vector(gh, q)

    us = np.linspace(0, 1, samples_u)
    vs = np.linspace(0, 1, samples_v)

    Bu = np.stack([bspline_basis_all(gw, p, U, u) for u in us], axis=0)  # (Mu, gw)
    Bv = np.stack([bspline_basis_all(gh, q, V, v) for v in vs], axis=0)  # (Mv, gh)

    # ctrl_pts will be shaped to (gh, gw, 3) BEFORE calling this function
    P = ctrl_pts.reshape(gh, gw, 3)

    S = np.zeros((samples_v, samples_u, 3), dtype=float)
    for k in range(3):
        Gk = P[..., k]  # (gh, gw)
        inner_u = np.tensordot(Bu, Gk.transpose(0, 1), axes=(1, 1))         # (Mu, gh)
        S[..., k] = np.tensordot(Bv, inner_u.transpose(1, 0), axes=(1, 0))  # (Mv, Mu)

    verts = S.reshape(-1, 3)
    tris = []
    for j in range(samples_v - 1):
        for i in range(samples_u - 1):
            a = j * samples_u + i
            b = a + 1
            c = a + samples_u
            d = c + 1
            tris.append([a, c, b]); tris.append([b, c, d])
    return verts, np.asarray(tris, dtype=np.int32)


def clamped_uniform_knot_vector(n_ctrl: int, degree: int) -> np.ndarray:
    """
    Clamped uniform knots in [0,1], length n_ctrl + degree + 1.
    Interior knots are uniformly spaced.
    """
    n_ctrl = int(n_ctrl)
    degree = int(degree)
    m = n_ctrl + degree + 1
    kv = np.zeros(m, dtype=float)
    kv[-(degree + 1):] = 1.0
    interior = n_ctrl - degree - 1
    if interior > 0:
        kv[degree + 1:degree + 1 + interior] = np.linspace(0.0, 1.0, interior + 2)[1:-1]
    return kv


def bspline_basis_all(n_ctrl: int, degree: int, knots: np.ndarray, u: float) -> np.ndarray:
    """
    Return all N_i,p(u) for i=0..n_ctrl-1 (global vector).
    Cox–de Boor using local nonzeros scattered into a length-n_ctrl vector.
    """
    p = int(degree)
    U = knots
    n = int(n_ctrl)
    N = np.zeros(n, dtype=float)

    # find span (largest s.t. U[span] <= u < U[span+1]), clamped at the end
    if u >= U[-p-1]:
        span = n - 1
    else:
        span = np.searchsorted(U, u, side='right') - 1
        span = max(p, min(span, n - 1))

    # local basis
    Nloc = np.zeros(p + 1, dtype=float)
    Nloc[0] = 1.0
    left = np.zeros(p + 1, dtype=float)
    right = np.zeros(p + 1, dtype=float)
    for j in range(1, p + 1):
        left[j] = u - U[span + 1 - j]
        right[j] = U[span + j] - u
        saved = 0.0
        for r in range(j):
            denom = right[r + 1] + left[j - r]
            a = 0.0 if denom == 0.0 else Nloc[r] / denom
            temp = a * right[r + 1]
            Nloc[r] = saved + temp
            saved = a * left[j - r]
        Nloc[j] = saved

    start = span - p
    N[start:start + p + 1] = Nloc
    return N

# =================== 2) IDW projection ===================
def project_external_to_surface_idw(ext_pts: np.ndarray, surf_mesh: o3d.geometry.TriangleMesh,
                                    k: int = 3, eps: float = 1e-9) -> np.ndarray:
    if ext_pts.size == 0:
        return np.empty((0, 3), dtype=float)
    verts = np.asarray(surf_mesh.vertices)
    if verts.size == 0:
        return np.empty((0, 3), dtype=float)

    surf_xy = verts[:, :2]
    surf_z  = verts[:, 2]
    ext_xy  = ext_pts[:, :2]

    if _SCIPY and surf_xy.shape[0] >= k:
        tree = cKDTree(surf_xy)
        d, idx = tree.query(ext_xy, k=k, workers=-1)
        if k == 1:
            d = d[:, None]; idx = idx[:, None]
        w = 1.0 / (d + eps)
        z_neighbors = surf_z[idx]
        zs = (w * z_neighbors).sum(axis=1) / w.sum(axis=1)
    else:
        pc_xy = o3d.geometry.PointCloud()
        pc_xy.points = o3d.utility.Vector3dVector(
            np.column_stack([surf_xy, np.zeros((surf_xy.shape[0],), dtype=surf_xy.dtype)])
        )
        kdt = o3d.geometry.KDTreeFlann(pc_xy)
        N = ext_xy.shape[0]
        zs = np.empty((N,), dtype=float)
        for i, (x, y) in enumerate(ext_xy):
            cnt, idxs, d2 = kdt.search_knn_vector_3d([float(x), float(y), 0.0], k)
            if cnt == 0:
                zs[i] = 0.0; continue
            d = np.sqrt(np.asarray(d2)[:cnt]) + eps
            w = 1.0 / d
            neigh = surf_z[np.asarray(idxs[:cnt], dtype=int)]
            zs[i] = float((w * neigh).sum() / w.sum())

    return np.column_stack([ext_pts[:, 0], ext_pts[:, 1], zs])


def mat_points(size=4.0):
    m = rendering.MaterialRecord()
    m.shader = "defaultUnlit"
    m.point_size = float(size)
    return m

def mat_mesh():
    m = rendering.MaterialRecord()
    m.shader = "defaultLit"
    m.base_color = (0.7, 0.7, 0.9, 1.0)
    m.base_roughness = 0.8
    return m

def visualize_with_materials(ctrl_pcd, surf_mesh, ext_pcd, proj_pcd=None):
    gui.Application.instance.initialize()
    window = gui.Application.instance.create_window("Smoothness + Surface", 1024, 768)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    window.add_child(scene)

    scene.scene.add_geometry("ctrl_pcd", ctrl_pcd, mat_points(8.0))
    scene.scene.add_geometry("surf_mesh", surf_mesh, mat_mesh())
    scene.scene.add_geometry("ext_pcd", ext_pcd, mat_points(2.0))
    if proj_pcd is not None:
        scene.scene.add_geometry("proj_pcd", proj_pcd, mat_points(2.0))

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    scene.scene.add_geometry("axis", axis, mat_mesh())

    # fit camera
    if proj_pcd is not None:
        aabbs = [g.get_axis_aligned_bounding_box() for g in [ctrl_pcd, surf_mesh, ext_pcd, proj_pcd]]
    else:
        aabbs = [g.get_axis_aligned_bounding_box() for g in [ctrl_pcd, surf_mesh, ext_pcd]]
    mins = np.min([a.min_bound for a in aabbs], axis=0)
    maxs = np.max([a.max_bound for a in aabbs], axis=0)
    center = 0.5 * (mins + maxs)
    extent = max(maxs - mins)
    eye = center + np.array([0, -3.0 * extent, 1.8 * extent])
    up = np.array([0, 0, 1])
    scene.scene.camera.look_at(center, eye, up)
    gui.Application.instance.run()



