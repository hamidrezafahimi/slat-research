#!/usr/bin/env python3
"""
helper.py — shared utilities for spline surfacing, projection, and loss.

- B-spline surface sampling (builds a TriangleMesh from control points)
- Grid + geometry helpers
- IDW projection of external cloud to current surface mesh (fast path via SciPy)
- Loss calculation and colorization
"""
from __future__ import annotations
import numpy as np
import open3d as o3d

# Optional SciPy KD-tree
try:
    from scipy.spatial import cKDTree  # type: ignore
    _SCIPY_AVAILABLE = True
except Exception:
    _SCIPY_AVAILABLE = False

# ---------- CSV I/O (for --spline_data) ----------
def load_ctrl_points_csv(path: str) -> np.ndarray:
    try:
        arr = np.loadtxt(path, delimiter=",")
    except Exception:
        arr = np.loadtxt(path)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 3)
    if arr.shape[1] != 3:
        raise ValueError(f"Expected Nx3 CSV, got shape {arr.shape}")
    return arr

def infer_grid_wh_from_points(pts: np.ndarray) -> tuple[int, int]:
    xs, ys = pts[:, 0], pts[:, 1]
    def _uniq_len(v):
        scale = max(np.max(np.abs(v)), 1.0)
        return np.unique(np.round(v/scale*1e8)).size
    gw = _uniq_len(xs)
    gh = _uniq_len(ys)
    n = pts.shape[0]
    if gw * gh != n:
        facs = [(w, n//w) for w in range(2, n+1) if n % w == 0]
        facs.sort(key=lambda ab: abs(ab[0] - gw))
        if not facs:
            raise ValueError("Cannot infer rectangular grid from CSV points.")
        gw, gh = facs[0]
    return int(gw), int(gh)

# ---------- B-spline sampling (for visualization mesh) ----------
def clamped_uniform_knot_vector(n_ctrl: int, degree: int) -> np.ndarray:
    p = degree
    m = n_ctrl + p + 1
    kv = np.zeros(m, dtype=float)
    kv[:p+1] = 0.0
    kv[-(p+1):] = 1.0
    interior = n_ctrl - p - 1
    if interior > 0:
        kv[p+1:m-(p+1)] = np.linspace(0.0, 1.0, interior+2)[1:-1]
    return kv

def bspline_basis_all(n_ctrl: int, degree: int, kv: np.ndarray, t: float) -> np.ndarray:
    p = degree
    lo, hi = kv[p], kv[-p-1]
    t = np.clip(t, lo + 1e-12, hi - 1e-12)
    N = np.zeros(n_ctrl)
    tmp = np.array([1.0 if (kv[j] <= t < kv[j+1]) else 0.0 for j in range(len(kv)-1)], dtype=float)
    for d in range(1, p+1):
        for j in range(len(tmp)-d):
            left = 0.0; right = 0.0
            dl = kv[j+d] - kv[j]
            dr = kv[j+d+1] - kv[j+1]
            if dl > 0: left  = (t - kv[j]) / dl * tmp[j]
            if dr > 0: right = (kv[j+d+1] - t) / dr * tmp[j+1]
            tmp[j] = left + right
    N[:n_ctrl] = tmp[:n_ctrl]
    return N

def sample_bspline_surface(ctrl_pts: np.ndarray, gw: int, gh: int,
                           samples_u: int = 40, samples_v: int = 40) -> tuple[np.ndarray, np.ndarray]:
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
        inner_u = np.tensordot(Bu, Gk.transpose(0, 1), axes=(1, 1))
        S[..., k] = np.tensordot(Bv, inner_u.transpose(1, 0), axes=(1, 0))
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

def build_surface_mesh(ctrl_points: np.ndarray, gw: int, gh: int,
                       samples_u: int = 40, samples_v: int = 40) -> o3d.geometry.TriangleMesh:
    verts, tris = sample_bspline_surface(ctrl_points, gw, gh, samples_u, samples_v)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(tris)
    mesh.compute_vertex_normals()
    return mesh

# ---------- Grid / geometry helpers ----------
def make_grid_points(grid_w: int, grid_h: int, metric_w: float, metric_h: float,
                     center_xy=(0.0, 0.0), z0: float = 0.0) -> np.ndarray:
    cx, cy = center_xy
    xs = np.linspace(cx - metric_w/2, cx + metric_w/2, grid_w)
    ys = np.linspace(cy - metric_h/2, cy + metric_h/2, grid_h)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.full_like(xx, z0, dtype=float)
    return np.column_stack((xx.ravel(), yy.ravel(), zz.ravel())).astype(float)

def make_o3d_pcd(pts: np.ndarray, color=(0.2, 0.8, 1.0)) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    col = np.array(color, dtype=float).reshape(1, 3)
    pcd.colors = o3d.utility.Vector3dVector(np.tile(col, (pts.shape[0], 1)))
    return pcd

def extent_wh_from_points_xy(pts: np.ndarray) -> tuple[float, float, np.ndarray]:
    mins = pts[:, :2].min(axis=0)
    maxs = pts[:, :2].max(axis=0)
    return float(maxs[0]-mins[0]), float(maxs[1]-mins[1]), ((mins+maxs)*0.5).astype(float)

# ---------- IDW projection + loss ----------
def project_external_to_surface_idw(ext_pts: np.ndarray,
                                    surf_mesh: o3d.geometry.TriangleMesh,
                                    k: int = 3, eps: float = 1e-9) -> np.ndarray:
    """
    For each external point E(x,y,ze), estimate z_surf(x,y) via inverse-distance
    weighting of k nearest surface vertices in XY. Returns Nx3 projected points.
    """
    if ext_pts.size == 0:
        return np.empty((0, 3), dtype=float)
    verts = np.asarray(surf_mesh.vertices)
    if verts.size == 0:
        return np.empty((0, 3), dtype=float)

    surf_xy = verts[:, :2]
    surf_z  = verts[:, 2]
    ext_xy  = ext_pts[:, :2]

    if _SCIPY_AVAILABLE and surf_xy.shape[0] >= k:
        tree = cKDTree(surf_xy)
        d, idx = tree.query(ext_xy, k=k, workers=-1)
        if k == 1:
            d = d[:, None]
            idx = idx[:, None]
        w = 1.0 / (d + eps)
        z_neighbors = surf_z[idx]           # (N,k)
        zs = (w * z_neighbors).sum(axis=1) / w.sum(axis=1)
    else:
        # Fallback: Open3D KDTreeFlann on XY embedded in 3D
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
                zs[i] = 0.0
                continue
            d = np.sqrt(np.asarray(d2)[:cnt]) + eps
            w = 1.0 / d
            neigh = surf_z[np.asarray(idxs[:cnt], dtype=int)]
            zs[i] = float((w * neigh).sum() / w.sum())

    return np.column_stack([ext_pts[:, 0], ext_pts[:, 1], zs])

def calc_loss(ext_pts: np.ndarray, spline_pts: np.ndarray, loss_thresh: float = 0.2) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Compute |Δz| per-point between external and projected spline points.
    Returns: (loss_val, mask_bad, colors_rgb)
    """
    if ext_pts.size == 0 or spline_pts.size == 0 or ext_pts.shape[0] != spline_pts.shape[0]:
        return 0, np.zeros((0,), dtype=bool), np.zeros((0, 3), dtype=float)
    dz = np.abs(ext_pts[:, 2] - spline_pts[:, 2])
    mask = dz > float(loss_thresh)
    colors = np.empty((ext_pts.shape[0], 3), dtype=float)
    colors[~mask] = [1.0, 0.0, 0.0]  # red OK
    colors[ mask] = [0.0, 0.0, 1.0]  # blue bad
    return int(mask.sum()), mask, colors


def project_mode_original(ext_pts: np.ndarray,
                          surf_mesh,
                          k: int,
                          eps: float,
                          loss_thresh: float):
    """
    Directly project ext_pts onto surface via IDW-over-surface-verts.
    Returns (spline_pts, colors, loss_val)
    Mirrors original app.py behavior.  (|Δz| loss vs ext_pts)
    """
    spline_pts = project_external_to_surface_idw(ext_pts, surf_mesh, k=k, eps=eps)
    loss_val, mask, colors = calc_loss(ext_pts, spline_pts, loss_thresh)
    return spline_pts, colors, float(loss_val)


def project_mode_indirect(ext_pts: np.ndarray,
                          surf_mesh,
                          k: int,
                          eps: float,
                          xy_cell: float,
                          loss_thresh: float = 0.2):
    """
    Indirect pipeline:
      1) ext -> z=0 (keep x,y; set z=0)
      2) grid on XY with cell=xy_cell
      3) keep one point per occupied cell (at cell center, z=0)
      4) project these centers to the surface (IDW)
      5) compute loss/colors exactly like 'original':
         - for each cell center, take the nearest ext-XY point's Z as reference
         - compare to projected spline Z (|Δz|) and color red/blue.
    Returns (spline_pts, colors, loss_val)
    """
    if ext_pts.size == 0:
        return ext_pts, np.empty((0, 3)), 0.0

    # (1) ext -> z=0
    ext2xy = np.column_stack([ext_pts[:, 0], ext_pts[:, 1],
                              np.zeros((ext_pts.shape[0],), dtype=float)])
    xy = ext2xy[:, :2]
    mins = xy.min(axis=0)
    cell = max(float(xy_cell), 1e-12)

    # (2) occupancy on XY
    ix = np.floor((xy[:, 0] - mins[0]) / cell).astype(np.int64)
    iy = np.floor((xy[:, 1] - mins[1]) / cell).astype(np.int64)
    occ = np.unique(np.stack([ix, iy], axis=1), axis=0)
    if occ.size == 0:
        return np.empty((0, 3)), np.empty((0, 3)), 0.0

    # (3) cell centers (z=0)
    centers_x = mins[0] + (occ[:, 0].astype(float) + 0.5) * cell
    centers_y = mins[1] + (occ[:, 1].astype(float) + 0.5) * cell
    xy_centers = np.column_stack([centers_x, centers_y,
                                  np.zeros_like(centers_x)])

    # (4) project centers to spline
    spline_pts = project_external_to_surface_idw(xy_centers, surf_mesh, k=k, eps=eps)

    # (5) build a same-length "external-like" set by borrowing Z from nearest ext-XY
    ext_xy = ext_pts[:, :2]
    centers_xy = xy_centers[:, :2]

    if _SCIPY_AVAILABLE and ext_xy.shape[0] > 0:
        tree = cKDTree(ext_xy)
        d, idx = tree.query(centers_xy, k=1, workers=-1)
        ref_z = ext_pts[idx, 2]  # nearest neighbor Z for each center
    else:
        # Fallback KD-tree with Open3D
        pc_xy = o3d.geometry.PointCloud()
        pc_xy.points = o3d.utility.Vector3dVector(
            np.column_stack([ext_xy, np.zeros((ext_xy.shape[0],), dtype=ext_xy.dtype)])
        )
        kdt = o3d.geometry.KDTreeFlann(pc_xy)
        ref_z = np.empty((centers_xy.shape[0],), dtype=float)
        for i, (x, y) in enumerate(centers_xy):
            cnt, idxs, d2 = kdt.search_knn_vector_3d([float(x), float(y), 0.0], 1)
            if cnt == 0:
                ref_z[i] = 0.0
            else:
                ref_z[i] = float(ext_pts[int(idxs[0]), 2])

    # Now compute loss/colors exactly like original
    ext_like = np.column_stack([xy_centers[:, 0], xy_centers[:, 1], ref_z])
    loss_val, mask, colors = calc_loss(ext_like, spline_pts, loss_thresh)

    return spline_pts, colors, float(loss_val)

def project_mode_dedup(ext_pts: np.ndarray,
                       surf_mesh,
                       samples_u: int,
                       samples_v: int,
                       k: int,
                       eps: float,
                       loss_thresh: float):
    """
    Deduplicate ext points by XY cells aligned to surface sampling grid
    (samples_u x samples_v), then project the reduced set.
    Returns (spline_pts, colors, loss_val) computed on the dedup set.
    """
    if ext_pts.size == 0:
        return ext_pts, np.empty((0, 3)), None

    verts = np.asarray(surf_mesh.vertices)
    if verts.size == 0:
        return np.empty((0, 3)), np.empty((0, 3)), None

    mins = verts[:, :2].min(axis=0)
    maxs = verts[:, :2].max(axis=0)
    span = np.maximum(maxs - mins, 1e-12)
    cu = max(int(samples_u), 1)
    cv = max(int(samples_v), 1)
    cell_w = span[0] / cu
    cell_h = span[1] / cv

    xy = ext_pts[:, :2]
    cx = np.floor((xy[:, 0] - mins[0]) / cell_w).astype(int)
    cy = np.floor((xy[:, 1] - mins[1]) / cell_h).astype(int)
    cx = np.clip(cx, 0, cu - 1)
    cy = np.clip(cy, 0, cv - 1)

    seen = set()
    keep_idx = []
    for i, key in enumerate(zip(cx.tolist(), cy.tolist())):
        if key in seen:
            continue
        seen.add(key)
        keep_idx.append(i)

    ext_dedup = ext_pts[np.array(keep_idx, dtype=int)] if keep_idx else ext_pts[:0]

    spline_pts = project_external_to_surface_idw(ext_dedup, surf_mesh, k=k, eps=eps)
    loss_val, mask, colors = calc_loss(ext_dedup, spline_pts, loss_thresh)
    return spline_pts, colors, float(loss_val)
