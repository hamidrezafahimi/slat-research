# unfold_api.py
"""
Single-call API:
    unfolded = unfold_surface_point(surface, point, *, k_plane=50, k_proj=40, n_mid=10, spline_samples=200)

- surface: Open3D PointCloud or (N,3) numpy array (the black surface)
- point: (3,) numpy array-like (the green point on the surface)
- returns: (3,) numpy array (the orange unfolded point in the XY-parallel plane through the surface’s interior-CG)

Matches the math/behavior of your original script while removing the gray-point logic and all visualization.
"""

from __future__ import annotations
import numpy as np

# Optional SciPy (falls back to Catmull–Rom if missing)
try:
    from scipy.interpolate import CubicSpline
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# ---------- small utilities ----------
def _as_points(surface):
    # Accept Open3D PointCloud or numpy
    try:
        import open3d as o3d  # only if user passed an o3d object
        if isinstance(surface, o3d.geometry.PointCloud):
            pts = np.asarray(surface.points)
        else:
            pts = np.asarray(surface, dtype=float)
    except Exception:
        pts = np.asarray(surface, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("surface must be Nx3 points or an Open3D PointCloud")
    # keep finite
    mask = np.isfinite(pts).all(axis=1)
    pts = pts[mask]
    if pts.shape[0] == 0:
        raise ValueError("No finite points in surface.")
    return pts

def _center_of_mass(pts):  # mean
    return pts.mean(axis=0)

def _fit_local_plane(patch):
    c = patch.mean(axis=0)
    _, _, Vt = np.linalg.svd(patch - c, full_matrices=False)
    n = Vt[-1, :]
    n /= (np.linalg.norm(n) + 1e-12)
    return n, c

def _kdtree(pts):
    # Tiny, dependency-free KD lookup using Open3D if present; else brute-force fallback
    try:
        import open3d as o3d
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts)
        return ("o3d", o3d.geometry.KDTreeFlann(pc), pc)
    except Exception:
        return ("np", pts, None)

def _knn(tree, q, k):
    kind = tree[0]
    if kind == "o3d":
        kd = tree[1]
        _, idxs, _ = kd.search_knn_vector_3d(q.astype(float), k)
        return np.asarray(idxs, dtype=int)
    # Numpy fallback
    pts = tree[1]
    d2 = np.sum((pts - q[None, :])**2, axis=1)
    return np.argsort(d2)[:k]

def _ray_plane_intersect(origin, dir_vec, plane_pt, plane_n, forward_only=True):
    d = dir_vec / (np.linalg.norm(dir_vec) + 1e-12)
    denom = float(np.dot(plane_n, d))
    if abs(denom) < 1e-12:
        return None
    t = float(np.dot(plane_n, plane_pt - origin) / denom)
    if forward_only and t <= 0:
        return None
    return origin + t * d

def _sample_segment_interior(a, b, n):
    if n <= 0: return np.empty((0, 3), float)
    a = np.asarray(a, float); b = np.asarray(b, float)
    ts = (np.arange(n, dtype=float) + 1.0) / (n + 1.0)
    return a[None, :] + ts[:, None] * (b - a)[None, :]

def _orth_proj_to_surface(M, pts, kdt, k_neighbors):
    k = int(np.clip(k_neighbors, 3, pts.shape[0]))
    nn = _knn(kdt, M, k)
    n, c = _fit_local_plane(pts[nn])
    d = float(np.dot(M - c, n))
    return M - d * n

def _chord_param(P):
    diffs = np.linalg.norm(np.diff(P, axis=0), axis=1)
    t = np.zeros(P.shape[0], float)
    t[1:] = np.cumsum(np.sqrt(diffs + 1e-12))
    if t[-1] > 0: t /= t[-1]
    return t

def _catmull_rom(P, t, ts):
    # Centrally parameterized C-R with simple end handling
    Q, tq = P, t
    seg_idx = np.clip(np.searchsorted(tq, ts, side='right') - 1, 0, len(tq)-2)
    def get(i): return Q[int(np.clip(i, 0, len(Q)-1))]
    out = []
    for s, u in zip(seg_idx, ts):
        i0, i1, i2, i3 = s-1, s, s+1, s+2
        P0, P1, P2, P3 = get(i0), get(i1), get(i2), get(i3)
        t0, t1, t2, t3 = tq[max(i0,0)], tq[i1], tq[i2], tq[min(i3, len(tq)-1)]
        if t2 - t1 < 1e-12:
            tau = 0.0
        else:
            tau = float(np.clip((u - t1) / (t2 - t1), 0.0, 1.0))
        def _tan(Pm, P0, P1, tm, t0, t1):
            denom = (t1 - tm) if (t1 - tm) > 1e-12 else 1e-12
            return (P1 - Pm) / denom
        m1 = 0.5 * ((t2 - t1) * _tan(P0, P1, P2, t0, t1, t2) +
                    (t1 - t0) * _tan(P1, P2, P3, t1, t2, t3))
        m2 = 0.5 * ((t3 - t2) * _tan(P1, P2, P3, t1, t2, t3) +
                    (t2 - t1) * _tan(P0, P1, P2, t0, t1, t2))
        tt, tt2, tt3 = tau, tau*tau, tau*tau*tau
        h00, h10, h01, h11 = 2*tt3-3*tt2+1, tt3-2*tt2+tt, -2*tt3+3*tt2, tt3-tt2
        C = h00*P1 + h10*m1 + h01*P2 + h11*m2
        out.append(C)
    return np.asarray(out, float)

def _cubic_curve(points, samples=200):
    P = np.asarray(points, float)
    if P.shape[0] < 2: return P.copy()
    if P.shape[0] == 2:
        t = np.linspace(0, 1, samples)
        return (1 - t)[:, None] * P[0] + t[:, None] * P[1]
    t = _chord_param(P)
    ts = np.linspace(0, 1, samples)
    if _HAS_SCIPY and P.shape[0] >= 3:
        xs = CubicSpline(t, P[:, 0], bc_type="natural")(ts)
        ys = CubicSpline(t, P[:, 1], bc_type="natural")(ts)
        zs = CubicSpline(t, P[:, 2], bc_type="natural")(ts)
        return np.stack([xs, ys, zs], axis=1)
    return _catmull_rom(P, t, ts)

# ---------- exact algorithm, compact form ----------
def unfold_surface_point(surface, point, *, k_plane=50, k_proj=40, n_mid=10, spline_samples=200, allow_backward=False):
    """
    Compute the orange unfolded point for a given green point on the black surface.
    The XY-parallel target plane is z = z(surface-interior-cg).

    Returns (3,) numpy array for the unfolded point.
    """
    pts = _as_points(surface)
    point = np.asarray(point, dtype=float).reshape(3)

    # 1) Surface interior CG (blue) via CG + local-plane intersection along its normal.
    cg = _center_of_mass(pts)
    kdt_all = _kdtree(pts)
    # nearest to CG, then a larger local patch for stable normal
    idx0 = _knn(kdt_all, cg, 1)[0]
    nn = _knn(kdt_all, pts[idx0], int(np.clip(k_plane, 3, pts.shape[0])))
    n, plane_pt = _fit_local_plane(pts[nn])
    # ensure normal points from cg -> plane_pt direction
    if np.dot(n, plane_pt - cg) < 0.0:
        n = -n
    sicg = _ray_plane_intersect(cg, n, plane_pt, n, forward_only=(not allow_backward))
    if sicg is None or not np.all(np.isfinite(sicg)):
        raise RuntimeError("Failed to compute surface-interior-cg")

    # 2) Build yellow unfold path: split (sicg -> point) into n_mid mids,
    #    orthogonally project mids to local planes → control points for a C^1 spline.
    mids = _sample_segment_interior(sicg, point, n_mid)
    kdt = kdt_all  # reuse
    proj_pts = np.array([_orth_proj_to_surface(M, pts, kdt, k_proj) for M in mids], dtype=float)
    ctrl = np.vstack([sicg[None, :], proj_pts, point[None, :]])
    curve = _cubic_curve(ctrl, samples=int(max(2, spline_samples)))

    # 3) Distance along the yellow curve + initial direction at its start (cyan).
    if curve.shape[0] >= 2:
        diffs = np.diff(curve, axis=0)
        seglens = np.linalg.norm(diffs, axis=1)
        total_len = float(np.sum(seglens))
        dir_vec = (curve[1] - curve[0])
    else:
        total_len = float(np.linalg.norm(point - sicg))
        dir_vec = (point - sicg)

    # 4) “Unfold” by marching total_len in the plane z = sicg.z along the XY
    #    projection of the cyan direction.
    dir_xy = np.array([dir_vec[0], dir_vec[1], 0.0], float)
    nrm = np.linalg.norm(dir_xy)
    if nrm < 1e-12:
        dir_xy = np.array([1.0, 0.0, 0.0], float)
        nrm = 1.0
    u = dir_xy / nrm
    out = sicg + total_len * u
    out[2] = sicg[2]  # stay on plane G

    return out



import open3d as o3d

if __name__ == '__main__':

    pcd = o3d.io.read_point_cloud("your_surface.pcd")   # the black surface
    green = np.array([10.15946354, 5.28891609, -4.12390131], dtype=float)         # a point on that surface
    orange = unfold_surface_point(pcd, green)           # ← single high-level call
    print("unfolded:", orange)