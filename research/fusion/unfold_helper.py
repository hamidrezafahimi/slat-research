import open3d as o3d
import random
from types import SimpleNamespace
import numpy as np

# Optional SciPy for cubic spline interpolation (unchanged)
try:
    from scipy.interpolate import CubicSpline
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

from scipy.spatial import cKDTree

# -------------------- border index helpers --------------------
def border_grid_indices(shape):
    """
    Given image shape (h, w), return a (N, 2) array of (row, col) integer
    pixel indices that lie ONLY on the image borders (corners included).
    Spacing along each axis is as equal as possible with integer pixels.
    """
    h, w = int(shape[0]), int(shape[1])
    assert h >= 2 and w >= 2, "shape must be at least 2x2"

    def _evenly_spaced_indices(n: int, max_index: int) -> np.ndarray:
        # produce n integers from 0..max_index with near-uniform steps
        assert n >= 2
        intervals = n - 1
        base = max_index // intervals
        rem = max_index % intervals
        out = [0]
        pos = 0
        err = 0
        for _ in range(intervals):
            step = base
            err += rem
            if err >= intervals:
                step += 1
                err -= intervals
            pos += step
            out.append(pos)
        out[-1] = max_index
        return np.asarray(out, dtype=np.int32)

    target_step = max(10, min(h, w) // 6)
    nx = max(2, (w - 1) // target_step + 1)
    ny = max(2, (h - 1) // target_step + 1)

    xs = _evenly_spaced_indices(nx, w - 1)
    ys = _evenly_spaced_indices(ny, h - 1)

    # Assemble borders: top, bottom, left, right
    top    = np.stack([np.zeros_like(xs),       xs], axis=1)                 # (0, x)
    bottom = np.stack([np.full_like(xs, h - 1), xs], axis=1)                 # (h-1, x)
    left   = np.stack([ys,                      np.zeros_like(ys)], axis=1)  # (y, 0)
    right  = np.stack([ys,                      np.full_like(ys, w - 1)], axis=1)  # (y, w-1)

    rc = np.concatenate([top, bottom, left, right], axis=0).astype(np.int32)

    # Deduplicate corners while preserving first occurrences
    keys = rc[:, 0] * w + rc[:, 1]
    _, uniq_idx = np.unique(keys, return_index=True)
    rc = rc[np.sort(uniq_idx)]
    return rc


def border_all_indices(shape):
    """
    Dense border indices for 'every single' border pixel.
    Returns dict with 'top','bottom','left','right' -> array[(N,2)] of (i,j).
    Corners will be present in both touching sides; users can dedup if needed.
    """
    h, w = int(shape[0]), int(shape[1])
    assert h >= 2 and w >= 2, "shape must be at least 2x2"
    top    = np.stack([np.zeros(w, dtype=np.int32),              np.arange(w, dtype=np.int32)], axis=1)
    bottom = np.stack([np.full(w, h - 1, dtype=np.int32),        np.arange(w, dtype=np.int32)], axis=1)
    left   = np.stack([np.arange(h, dtype=np.int32),             np.zeros(h, dtype=np.int32)], axis=1)
    right  = np.stack([np.arange(h, dtype=np.int32),             np.full(h, w - 1, dtype=np.int32)], axis=1)
    return dict(top=top, bottom=bottom, left=left, right=right)


# -------------------- point-cloud helpers (unchanged) --------------------
def finite_points_from_pcd(pcd):
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        raise ValueError("Point cloud has no points.")
    mask = np.isfinite(pts).all(axis=1)
    pts = pts[mask]
    if pts.size == 0:
        raise ValueError("All points are non-finite (NaN/Inf).")
    return pts

def compute_center_of_mass(pts):
    return pts.mean(axis=0)

def build_kdtree_from_points(pts):
    # Fast SciPy KD-tree (balanced + compact)
    return cKDTree(pts, leafsize=32, compact_nodes=True, balanced_tree=True)

def nearest_index(kdtree, q, k=1):
    # cKDTree.query returns (dist, idx)
    _, idx = kdtree.query(q, k=k)
    return int(idx) if k == 1 else idx


def fit_local_plane(patch):
    c = patch.mean(axis=0)
    _, _, Vt = np.linalg.svd(patch - c, full_matrices=False)
    n = Vt[-1, :]
    n = n / (np.linalg.norm(n) + 1e-12)
    return n, c

def ray_plane_intersection(origin, dir_vec, plane_pt, plane_n, forward_only=True):
    dirn = dir_vec / (np.linalg.norm(dir_vec) + 1e-12)
    denom = float(np.dot(plane_n, dirn))
    if abs(denom) < 1e-12:
        return None
    t = float(np.dot(plane_n, plane_pt - origin) / denom)
    if forward_only and t <= 0:
        return None
    return origin + t * dirn

def make_sphere(center, radius, color):
    m = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    m.compute_vertex_normals()
    m.paint_uniform_color(color)
    m.translate(center)
    return m

def make_lineset(points_np, lines_idx, color_rgb):
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points_np)
    ls.lines = o3d.utility.Vector2iVector(lines_idx)
    ls.colors = o3d.utility.Vector3dVector([color_rgb for _ in range(len(lines_idx))])
    return ls

def make_polyline(points_np, color_rgb):
    if points_np.shape[0] < 2:
        return None
    lines = np.column_stack([
        np.arange(points_np.shape[0]-1, dtype=np.int32),
        np.arange(1, points_np.shape[0], dtype=np.int32)
    ])
    return make_lineset(points_np, lines, color_rgb)

def sample_segment_interior(a, b, n):
    if n <= 0:
        return np.empty((0, 3), dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ts = (np.arange(n, dtype=float) + 1.0) / (n + 1.0)
    return a[None, :] + ts[:, None] * (b - a)[None, :]

def orthogonal_projection_to_local_surface(M, pts, kdtree, k_neighbors):
    k_use = min(max(3, k_neighbors), pts.shape[0])
    # query k nearest neighbors of M
    _, nn_idx = kdtree.query(M, k=k_use)
    patch = pts[np.atleast_1d(nn_idx)]
    n, c = fit_local_plane(patch)
    d = float(np.dot(M - c, n))
    P = M - d * n
    return P

def chord_length_param(points):
    diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
    t = np.zeros(points.shape[0], dtype=float)
    t[1:] = np.cumsum(np.sqrt(diffs + 1e-12))
    if t[-1] > 0:
        t /= t[-1]
    return t

def cubic_spline_interp(points, n_samples=200):
    P = np.asarray(points, dtype=float)
    if P.shape[0] < 2:
        return P.copy()
    if P.shape[0] == 2:
        t = np.linspace(0, 1, n_samples)
        return (1 - t)[:, None] * P[0] + t[:, None] * P[1]
    t = chord_length_param(P)
    ts = np.linspace(0, 1, n_samples)
    if HAS_SCIPY and P.shape[0] >= 3:
        xs = CubicSpline(t, P[:, 0], bc_type="natural")(ts)
        ys = CubicSpline(t, P[:, 1], bc_type="natural")(ts)
        zs = CubicSpline(t, P[:, 2], bc_type="natural")(ts)
        return np.stack([xs, ys, zs], axis=1)
    return catmull_rom_sample(P, t, ts)

def catmull_rom_sample(P, t, ts):
    Q = P
    tq = t
    seg_idx = np.searchsorted(tq, ts, side='right') - 1
    seg_idx = np.clip(seg_idx, 0, len(tq) - 2)
    def get(idx):
        idx = int(np.clip(idx, 0, len(Q)-1))
        return Q[idx]
    samples = []
    for s, u in zip(seg_idx, ts):
        i0, i1, i2, i3 = s-1, s, s+1, s+2
        P0, P1, P2, P3 = get(i0), get(i1), get(i2), get(i3)
        t0, t1, t2, t3 = tq[max(i0,0)], tq[i1], tq[i2], tq[min(i3, len(tq)-1)]
        if t2 - t1 < 1e-12:
            tau = 0.0
        else:
            tau = (u - t1) / (t2 - t1)
            tau = np.clip(tau, 0.0, 1.0)
        def alpha_tan(Pm, P0, P1, tm, t0, t1):
            denom = (t1 - tm) if (t1 - tm) > 1e-12 else 1e-12
            return (P1 - Pm) / denom
        m1 = ( (t2 - t1) * (alpha_tan(P0, P1, P2, t0, t1, t2)) +
               (t1 - t0) * (alpha_tan(P1, P2, P3, t1, t2, t3)) ) * 0.5
        m2 = ( (t3 - t2) * (alpha_tan(P1, P2, P3, t1, t2, t3)) +
               (t2 - t1) * (alpha_tan(P0, P1, P2, t0, t1, t2)) ) * 0.5
        tt, tt2, tt3 = tau, tau*tau, tau*tau*tau
        h00, h10, h01, h11 = 2*tt3-3*tt2+1, tt3-2*tt2+tt, -2*tt3+3*tt2, tt3-tt2
        C = (h00*P1 + h10*m1 + h01*P2 + h11*m2)
        samples.append(C)
    return np.asarray(samples, dtype=float)

def _skew(v):
    x, y, z = v
    return np.array([[0, -z,  y],
                     [z,  0, -x],
                     [-y, x,  0]], dtype=float)

def rot_from_a_to_b(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b); s = np.linalg.norm(v); c = np.dot(a, b)
    if s < 1e-12:
        if c > 0.0:
            return np.eye(3)
        axis = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(axis, a)) > 0.9:
            axis = np.array([0.0, 0.0, 1.0])
        v = np.cross(a, axis); v = v / (np.linalg.norm(v) + 1e-12)
        K = _skew(v); return np.eye(3) + K + K @ K
    K = _skew(v)
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s**2 + 1e-12))

def make_arrow_from_to(start, end, color, thin_factor=1.0):
    start = np.asarray(start, float); end = np.asarray(end, float)
    v = end - start; L = float(np.linalg.norm(v))
    if L < 1e-12:
        m = o3d.geometry.TriangleMesh.create_sphere(radius=1e-6)
        m.paint_uniform_color(color); m.translate(start); return m
    cone_len = 0.20 * L; cyl_len = max(L - cone_len, 1e-9)
    shaft_r = 0.02 * L * thin_factor; head_r = 2.0 * shaft_r
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=shaft_r, cone_radius=head_r,
        cylinder_height=cyl_len, cone_height=max(L-cyl_len, 1e-9))
    arrow.compute_vertex_normals(); arrow.paint_uniform_color(color)
    R = rot_from_a_to_b(np.array([0.0, 0.0, 1.0]), v)
    arrow.rotate(R, center=np.array([0.0, 0.0, 0.0])); arrow.translate(start)
    return arrow


# -------------------- runtime config (UNCHANGED core + vis_debug) --------------------
CONFIG = SimpleNamespace(
    k_plane=50,
    k_proj=40,
    allow_backward=False,
    n_mid=10,
    spline_samples=200,
    spline_color=[1.0, 1.0, 0.0],  # yellow for unfold-spline
    show_green_to_orange=True,
    n_greens=10,
    vis_debug=True,
)


# -------------------- core functions (UNCHANGED math) --------------------
def find_surface_mass_anchor(surface_pts):
    cg = compute_center_of_mass(surface_pts)
    kdt_all = build_kdtree_from_points(surface_pts)
    ni = nearest_index(kdt_all, cg)
    k_use = min(max(3, CONFIG.k_plane), surface_pts.shape[0])

    # cKDTree: get neighbor indices around the nearest surface point
    _, nn_idx = kdt_all.query(surface_pts[ni], k=k_use)
    patch = surface_pts[np.atleast_1d(nn_idx)]

    n, c_on_plane = fit_local_plane(patch)
    if np.dot(n, c_on_plane - cg) < 0.0:
        n = -n
    surface_interior_cg = ray_plane_intersection(
        cg, n, c_on_plane, n, forward_only=(not CONFIG.allow_backward)
    )
    if surface_interior_cg is None or not np.all(np.isfinite(surface_interior_cg)):
        raise SystemExit("Could not compute 'surface-interior-cg'. Adjust parameters.")
    aux = dict(cg=cg, kdtree=kdt_all, nn_patch_idx=np.atleast_1d(nn_idx),
               plane_normal=n, plane_point=c_on_plane)
    return surface_interior_cg, aux

def find_surface_point_unfold_params(surface_pts, surface_interior_cg, point_i):
    mids = sample_segment_interior(surface_interior_cg, point_i, CONFIG.n_mid)
    kdt_all = build_kdtree_from_points(surface_pts)
    proj_points = [orthogonal_projection_to_local_surface(M, surface_pts, kdt_all, CONFIG.k_proj) for M in mids]
    ctrl_pts_np = np.asarray([surface_interior_cg, *proj_points, point_i], dtype=float)
    curve = cubic_spline_interp(ctrl_pts_np, n_samples=CONFIG.spline_samples)

    if curve.shape[0] >= 2:
        v = curve[1] - curve[0]
        dir_vec = v / (np.linalg.norm(v) + 1e-12)
    else:
        v = point_i - surface_interior_cg
        dir_vec = v / (np.linalg.norm(v) + 1e-12)

    diffs = np.diff(curve, axis=0) if curve.shape[0] >= 2 else np.empty((0,3))
    seglens = np.linalg.norm(diffs, axis=1) if diffs.size else np.array([0.0])
    dist = float(np.sum(seglens))

    aux = dict(mids=mids, proj_points=np.asarray(proj_points, dtype=float), curve=curve)
    return dir_vec, dist, aux

def unfold_surface_point(surface_interior_cg, dir_vec, dist):
    # move in plane G along projected cyan dir
    dir_xy = np.array([dir_vec[0], dir_vec[1], 0.0], dtype=float)
    nrm = np.linalg.norm(dir_xy)
    if nrm < 1e-12:
        dir_xy = np.array([1.0, 0.0, 0.0], dtype=float); nrm = 1.0
    u = dir_xy / nrm
    p = np.asarray(surface_interior_cg, dtype=float) + dist * u
    p[2] = surface_interior_cg[2]
    return p


# -------------------- visualization (existing) --------------------
def _make_dashed_line(a, b, color_rgb, dash_len, gap_len):
    a = np.asarray(a, float); b = np.asarray(b, float)
    v = b - a; L = float(np.linalg.norm(v))
    if L < 1e-12: return None
    u = v / L; pts = [a]; lines = []; pos = 0.0; draw_on = True; idx = 0
    while pos < L - 1e-12:
        step = dash_len if draw_on else gap_len
        step = min(step, L - pos)
        next_pt = a + (pos + step) * u
        pts.append(next_pt)
        if draw_on: lines.append([idx, idx + 1])
        idx += 1; pos += step; draw_on = not draw_on
    return make_lineset(np.vstack(pts), np.asarray(lines, np.int32), color_rgb)

def _make_plane_G(bbox, z_val, color=[0.2, 0.6, 1.0]):
    min_bound = bbox.min_bound
    max_bound = bbox.max_bound
    corners = np.array([
        [min_bound[0], min_bound[1], z_val],
        [max_bound[0], min_bound[1], z_val],
        [max_bound[0], max_bound[1], z_val],
        [min_bound[0], max_bound[1], z_val],
    ], dtype=float)
    tris = np.array([[0, 1, 2], [0, 2, 3], [2, 1, 0], [3, 2, 0]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(corners)
    mesh.triangles = o3d.utility.Vector3iVector(tris)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh

def visualize_scene_multi(pcd, cg, surface_interior_cg, items, spline_color, show_green_to_orange=True):
    if not CONFIG.vis_debug:
        return  # skip any visualization work entirely

    draw = [pcd]
    bbox = pcd.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(bbox.get_extent()) if bbox is not None else 1.0

    plane_G = _make_plane_G(bbox, surface_interior_cg[2], color=[0.2, 0.6, 1.0])
    draw.append(plane_G)

    r_cg = max(1e-6, 0.0020 * diag)
    r_sicg = max(1e-6, 0.0018 * diag)
    draw.append(make_sphere(cg, r_cg, [1.0, 0.0, 0.0]))
    draw.append(make_sphere(surface_interior_cg, r_sicg, [0.0, 0.0, 1.0]))

    r_point = max(1e-6, 0.0018 * diag)
    r_mid_purple = max(1e-6, 0.0015 * diag)
    r_mid_yellow = max(1e-6, 0.0015 * diag)
    r_unfolded = max(1e-6, 0.0018 * diag)
    r_radial_proj = max(1e-6, 0.0018 * diag)

    dash_len = 0.02 * diag
    gap_len  = 0.015 * diag

    for idx, it in enumerate(items, 1):
        point_i = it["point"]
        mids_purple = it.get("mids", np.empty((0,3)))
        mids_yellow = it.get("proj_points", np.empty((0,3)))
        curve = it.get("curve", None)
        unfolded_i = it.get("unfolded", None)
        radial_proj_i = it.get("radial_proj", None)

        draw.append(make_sphere(point_i, r_point, [0.0, 1.0, 0.0]))
        draw.append(make_lineset(np.vstack([surface_interior_cg, point_i]),
                                 np.array([[0, 1]], np.int32), [1.0, 1.0, 1.0]))
        for M in mids_purple:
            draw.append(make_sphere(M, r_mid_purple, [1.0, 0.0, 1.0]))
        for P in mids_yellow:
            draw.append(make_sphere(P, r_mid_yellow, [1.0, 1.0, 0.0]))

        if curve is not None and curve.shape[0] >= 2:
            ls = make_polyline(curve, CONFIG.spline_color)
            if ls is not None: draw.append(ls)
            p0, p1 = curve[0], curve[1]
            draw.append(make_arrow_from_to(p0, p1 + 19*(p1 - p0), [0.0, 1.0, 1.0], thin_factor=0.5))

        if unfolded_i is not None:
            draw.append(make_sphere(unfolded_i, r_unfolded, [1.0, 0.5, 0.0]))
            if show_green_to_orange:
                ds = _make_dashed_line(point_i, unfolded_i, [1.0, 0.5, 0.0], dash_len, gap_len)
                if ds is not None: draw.append(ds)

        if radial_proj_i is not None:
            draw.append(make_sphere(radial_proj_i, r_radial_proj, [0.6, 0.6, 0.6]))
            if show_green_to_orange:
                ds_g = _make_dashed_line(point_i, radial_proj_i, [0.6, 0.6, 0.6], dash_len, gap_len)
                if ds_g is not None: draw.append(ds_g)

    o3d.visualization.draw_geometries(
        draw,
        window_name="cg & surface-interior-cg with N×{point-(id), radial-dir-(id), midpoints, unfold-spline-(id), unfolded-/radial-projection-point-(id)} + plane G",
        width=1400, height=900
    )


# -------------------- new: spline evaluation utility --------------------
def eval_spline_through_points(points, ts_query):
    """
    Fit a C^2 cubic (SciPy) or Catmull-Rom fallback through 'points' and evaluate at ts_query in [0,1].
    """
    P = np.asarray(points, dtype=float)
    if P.shape[0] == 0:
        return np.empty((0, 3), dtype=float)
    if P.shape[0] == 1:
        return np.repeat(P, repeats=len(ts_query), axis=0)
    t_knots = chord_length_param(P)
    ts_query = np.asarray(ts_query, dtype=float)
    ts_query = np.clip(ts_query, 0.0, 1.0)
    if HAS_SCIPY and P.shape[0] >= 3:
        xs = CubicSpline(t_knots, P[:, 0], bc_type="natural")(ts_query)
        ys = CubicSpline(t_knots, P[:, 1], bc_type="natural")(ts_query)
        zs = CubicSpline(t_knots, P[:, 2], bc_type="natural")(ts_query)
        return np.stack([xs, ys, zs], axis=1)
    # fallback to Catmull-Rom eval at custom ts
    return catmull_rom_sample(P, t_knots, ts_query)


# -------------------- old high-level (kept for compatibility) --------------------
def unfold_image_border_points(
    metric_depth_csv: str,
    hfov_deg: float = 66.0,
    pose_kwargs=None,
    max_points=None,
):
    """
    OLD high-level: read metric depth CSV, select uniform border pixels (i,j),
    run the EXISTING unfolding pipeline, and return the orange (unfolded) points.

    Returns
    -------
    unfolded_points : (N, 3) float32
    extras : dict with pcd/cg/surface_interior_cg/items
    """
    # Lazy imports to avoid hard dependency here
    import os, sys
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(DIR_PATH + "/../../lib")
    from kinematics.pose import Pose
    from mapper3D_helper import project3DAndScale

    pose_kwargs = pose_kwargs or {}
    # 1) Load metric depth
    metric_depth = np.loadtxt(metric_depth_csv, delimiter=',', dtype=np.float32)
    H, W = metric_depth.shape[:2]

    # 2) Build pose & HxWx3 via project3DAndScale
    pose = Pose(**{
        "x":  pose_kwargs.get("x", 0.0),
        "y":  pose_kwargs.get("y", 0.0),
        "z":  pose_kwargs.get("z", 9.4),
        "roll":  pose_kwargs.get("roll", 0.0),
        "pitch": pose_kwargs.get("pitch", -0.78),
        "yaw":   pose_kwargs.get("yaw", 0.0),
    })
    xyz_img = project3DAndScale(metric_depth, pose, hfov_deg, metric_depth.shape)

    # 3) Flatten to pcd (black)
    pts_flat = xyz_img.reshape(-1, 3)
    mask = np.isfinite(pts_flat).all(axis=1)
    pts_flat = pts_flat[mask]
    if pts_flat.size == 0:
        raise SystemExit("All projected points are non-finite.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_flat)
    pcd.paint_uniform_color([0.0, 0.0, 0.0])

    # Finite pts for math (unchanged)
    pts = finite_points_from_pcd(pcd)
    if pts.shape[0] != len(pcd.points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color([0.0, 0.0, 0.0])

    # Step 1: surface-interior-cg via cg + local plane hit (unchanged)
    surface_interior_cg, aux1 = find_surface_mass_anchor(pts)
    cg = aux1["cg"]

    # Step 2: choose border (i,j) from HxW
    rc_all = border_grid_indices((H, W))
    if max_points is not None and max_points > 0:
        rc_use = rc_all[:max_points]
    else:
        rc_use = rc_all

    # Convert (i,j) -> 3D points, skip non-finite & duplicates of SICG
    points = []
    for (i, j) in rc_use:
        p_ij = xyz_img[i, j]
        if not np.all(np.isfinite(p_ij)):
            continue
        if np.linalg.norm(p_ij - surface_interior_cg) <= 1e-12:
            continue
        points.append(p_ij)
    if len(points) == 0:
        raise SystemExit("[ERROR] No valid border points found.")

    # Step 3: loop over points (UNCHANGED math)
    items = []
    unfolded_pts = []
    for idx, p in enumerate(points, 1):
        dir_vec, spline_len, aux2 = find_surface_point_unfold_params(pts, surface_interior_cg, p)
        white_len = float(np.linalg.norm(p - surface_interior_cg))
        unfolded_i = unfold_surface_point(surface_interior_cg, dir_vec, spline_len)
        radial_proj_i = unfold_surface_point(surface_interior_cg, dir_vec, white_len)
        unfolded_pts.append(unfolded_i)

        # Per-point printing only if debugging
        if CONFIG.vis_debug:
            print(f"[{idx:02d}] point-{idx}:  x={p[0]:.6f}, y={p[1]:.6f}, z={p[2]:.6f}")
            print(f"     unfolded-point-{idx}:  x={unfolded_i[0]:.6f}, y={unfolded_i[1]:.6f}, z={unfolded_i[2]:.6f}")
            print(f"     cyan-dir: ({dir_vec[0]:.6f}, {dir_vec[1]:.6f}, {dir_vec[2]:.6f})")
            print(f"     unfold-spline-{idx} length = {spline_len:.6f}")
            print(f"     radial-dir-{idx} length    = {white_len:.6f}")

        if CONFIG.vis_debug:
            items.append(dict(
                point=p,
                mids=aux2["mids"],
                proj_points=aux2["proj_points"],
                curve=aux2["curve"],
                unfolded=unfolded_i,
                radial_proj=radial_proj_i,
            ))

    extras = dict(
        pcd=pcd,
        cg=cg,
        surface_interior_cg=surface_interior_cg,
        items=items,
        xyz_img=xyz_img,  # expose for reuse
    )
    return np.asarray(unfolded_pts, dtype=np.float32), extras


def unfold_image_borders(
    metric_depth_csv: str,
    hfov_deg: float = 66.0,
    pose_kwargs=None,
    max_points=None,
):
    """
    Superset of unfold_image_border_points:
      1) Compute orange (unfolded) points for a UNIFORM subset of border pixels.
      2) Build four splines that each go *corner→corner*, in this exact order:
         TOP:    top-left  -> top-right
         RIGHT:  top-right -> bottom-right
         BOTTOM: bottom-right -> bottom-left
         LEFT:   bottom-left -> top-left
         (each passes all mid-orange points on that side, if any)
      3) For EVERY border pixel, suggest an orange position by evaluating the
         respective side spline at that pixel’s normalized coordinate t∈[0,1].
      4) Returns the sampled oranges and the dense (tiny red) suggestions.

    Returns
    -------
    result : dict with:
      - 'unfolded_sample': (Ns,3) orange points for the sampled border set
      - 'splines': dict side -> {'P': (Nk,3) sample oranges incl. corners,
                                 'ts_dense': (Nd,), 'Y_dense': (Nd,3)}
      - 'dense_red': dict side -> (Nd,3) suggested orange positions (tiny red dots)
    extras : dict from unfold_image_border_points (pcd, cg, sicg, items, xyz_img, ...)
    """

    # --- 0) run the existing pipeline to get sample oranges + scene context ---
    unfolded_sample, extras = unfold_image_border_points(
        metric_depth_csv=metric_depth_csv,
        hfov_deg=hfov_deg,
        pose_kwargs=pose_kwargs,
        max_points=max_points
    )

    # --- local helpers (self-contained so only this function needs replacing) ---
    def _eval_spline_at_t(P_sample, t_sample, ts_query):
        """
        Fit/evaluate a cubic through (t_sample, P_sample) and eval at ts_query in [0,1].
        Uses SciPy CubicSpline if available; otherwise Catmull-Rom on the same t-domain.
        """
        P = np.asarray(P_sample, float)
        t = np.asarray(t_sample, float)
        tq = np.asarray(ts_query, float)

        if P.shape[0] == 0:
            return np.empty((0, 3), float)
        if P.shape[0] == 1:
            return np.repeat(P, repeats=len(tq), axis=0)

        # strictly increasing unique t
        t_u, idx = np.unique(t, return_index=True)
        P_u = P[idx]
        tq = np.clip(tq, 0.0, 1.0)

        try:
            from scipy.interpolate import CubicSpline  # local import
            if P_u.shape[0] >= 3:
                xs = CubicSpline(t_u, P_u[:, 0], bc_type="natural")(tq)
                ys = CubicSpline(t_u, P_u[:, 1], bc_type="natural")(tq)
                zs = CubicSpline(t_u, P_u[:, 2], bc_type="natural")(tq)
                return np.stack([xs, ys, zs], axis=1)
        except Exception:
            pass
        # fallback: Catmull-Rom on supplied t-knots
        return catmull_rom_sample(P_u, t_u, tq)

    def _fit_eval_segment(t_sample, P_sample, ts_dense):
        # sort by t, then evaluate
        t_sample = np.asarray(t_sample, float)
        P_sample = np.asarray(P_sample, float)
        order = np.argsort(t_sample)
        t_sorted = t_sample[order]
        P_sorted = P_sample[order]
        Y = _eval_spline_at_t(P_sorted, t_sorted, ts_dense)
        return dict(P=P_sorted, ts_dense=ts_dense, Y_dense=Y), Y

    # compute "orange" for a specific border pixel (i,j)
    def _orange_for_ij(i, j):
        p_ij = extras["xyz_img"][i, j]
        dir_vec, spline_len, _ = find_surface_point_unfold_params(
            np.asarray(extras["pcd"].points),          # surface_pts
            extras["surface_interior_cg"],             # SICG
            p_ij                                       # green on the surface
        )
        return unfold_surface_point(extras["surface_interior_cg"], dir_vec, spline_len)

    # --- 1) rebuild the filtered (i,j) list in the SAME order as the sampled items ---
    metric_depth = np.loadtxt(metric_depth_csv, delimiter=',', dtype=np.float32)
    H, W = metric_depth.shape[:2]

    rc_sample = border_grid_indices((H, W))
    if max_points is not None and max_points > 0:
        rc_sample = rc_sample[:max_points]

    filtered_rc = []
    for (i, j) in rc_sample:
        p_ij = extras["xyz_img"][i, j]
        if not np.all(np.isfinite(p_ij)):
            continue
        if np.linalg.norm(p_ij - extras["surface_interior_cg"]) <= 1e-12:
            continue
        filtered_rc.append((i, j))

    # sanity: items were generated in this same filtered order
    assert len(filtered_rc) == len(extras["items"]), \
        "Mismatch between filtered border pixels and generated sample items."

    # collect the unfolded oranges in the same order (already produced)
    sample_unfolded_ordered = np.asarray([it["unfolded"] for it in extras["items"]], float)

    # --- 2) split samples by side, storing *t* per your traversal directions ---
    def side_of(i, j):
        if i == 0:     return "top"
        if i == H - 1: return "bottom"
        if j == 0:     return "left"
        if j == W - 1: return "right"
        return None

    side_groups = dict(top=[], bottom=[], left=[], right=[])
    for (i, j), P_orange in zip(filtered_rc, sample_unfolded_ordered):
        s = side_of(i, j)
        if s is None:
            continue
        # normalized coordinate along traversal for each side:
        if s == "top":      t =  j / (W - 1.0)                 # left -> right
        elif s == "right":  t =  i / (H - 1.0)                 # top -> bottom
        elif s == "bottom": t = (W - 1.0 - j) / (W - 1.0)      # right -> left
        else:               t = (H - 1.0 - i) / (H - 1.0)      # left: bottom -> top
        side_groups[s].append((t, P_orange))

    # --- 3) compute corner oranges (endpoints) ---
    tl = (0, 0)
    tr = (0, W - 1)
    br = (H - 1, W - 1)
    bl = (H - 1, 0)

    P_tl = _orange_for_ij(*tl)
    P_tr = _orange_for_ij(*tr)
    P_br = _orange_for_ij(*br)
    P_bl = _orange_for_ij(*bl)

    # --- 4) per-side corner→corner spline fit + dense evaluation ---
    splines = {}
    dense_red = {}

    # TOP: tl -> tr (t=j/(W-1))
    top_pts = sorted(side_groups["top"], key=lambda x: x[0])
    t_top = [0.0] + [t for (t, _) in top_pts] + [1.0]
    P_top = [P_tl] + [P for (_, P) in top_pts] + [P_tr]
    ts_top_dense = np.linspace(0.0, 1.0, W)
    splines["top"], dense_red["top"] = _fit_eval_segment(t_top, P_top, ts_top_dense)

    # RIGHT: tr -> br (t=i/(H-1))
    right_pts = sorted(side_groups["right"], key=lambda x: x[0])
    t_right = [0.0] + [t for (t, _) in right_pts] + [1.0]
    P_right = [P_tr] + [P for (_, P) in right_pts] + [P_br]
    ts_right_dense = np.linspace(0.0, 1.0, H)
    splines["right"], dense_red["right"] = _fit_eval_segment(t_right, P_right, ts_right_dense)

    # BOTTOM: br -> bl (t=(W-1-j)/(W-1))
    bottom_pts = sorted(side_groups["bottom"], key=lambda x: x[0])
    t_bottom = [0.0] + [t for (t, _) in bottom_pts] + [1.0]
    P_bottom = [P_br] + [P for (_, P) in bottom_pts] + [P_bl]
    ts_bottom_dense = np.linspace(0.0, 1.0, W)
    splines["bottom"], dense_red["bottom"] = _fit_eval_segment(t_bottom, P_bottom, ts_bottom_dense)

    # LEFT: bl -> tl (t=(H-1-i)/(H-1))
    left_pts = sorted(side_groups["left"], key=lambda x: x[0])
    t_left = [0.0] + [t for (t, _) in left_pts] + [1.0]
    P_left = [P_bl] + [P for (_, P) in left_pts] + [P_tl]
    ts_left_dense = np.linspace(0.0, 1.0, H)
    splines["left"], dense_red["left"] = _fit_eval_segment(t_left, P_left, ts_left_dense)

    # --- 5) pack results ---
    result = dict(
        unfolded_sample=np.asarray(sample_unfolded_ordered, dtype=np.float32),
        splines=splines,
        dense_red=dense_red,
    )
    return result, extras


# -------------------- NEW visualization that also draws tiny red dots --------------------
def visualize_scene_borders(pcd, cg, surface_interior_cg, items, red_points_by_side,
                            spline_color, show_green_to_orange=True):
    if not CONFIG.vis_debug:
        return

    draw = [pcd]
    bbox = pcd.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(bbox.get_extent()) if bbox is not None else 1.0

    # Plane G + anchors
    plane_G = _make_plane_G(bbox, surface_interior_cg[2], color=[0.2, 0.6, 1.0])
    draw.append(plane_G)
    r_cg = max(1e-6, 0.0020 * diag)
    r_sicg = max(1e-6, 0.0018 * diag)
    draw.append(make_sphere(cg, r_cg, [1.0, 0.0, 0.0]))
    draw.append(make_sphere(surface_interior_cg, r_sicg, [0.0, 0.0, 1.0]))

    # Reuse the per-item rendering from visualize_scene_multi but inline it here
    r_point = max(1e-6, 0.0018 * diag)
    r_mid_purple = max(1e-6, 0.0015 * diag)
    r_mid_yellow = max(1e-6, 0.0015 * diag)
    r_unfolded = max(1e-6, 0.0018 * diag)
    r_radial_proj = max(1e-6, 0.0018 * diag)

    dash_len = 0.02 * diag
    gap_len  = 0.015 * diag

    for idx, it in enumerate(items, 1):
        point_i = it["point"]
        mids_purple = it.get("mids", np.empty((0,3)))
        mids_yellow = it.get("proj_points", np.empty((0,3)))
        curve = it.get("curve", None)
        unfolded_i = it.get("unfolded", None)
        radial_proj_i = it.get("radial_proj", None)

        draw.append(make_sphere(point_i, r_point, [0.0, 1.0, 0.0]))
        draw.append(make_lineset(np.vstack([surface_interior_cg, point_i]),
                                 np.array([[0, 1]], np.int32), [1.0, 1.0, 1.0]))
        for M in mids_purple:
            draw.append(make_sphere(M, r_mid_purple, [1.0, 0.0, 1.0]))
        for P in mids_yellow:
            draw.append(make_sphere(P, r_mid_yellow, [1.0, 1.0, 0.0]))

        if curve is not None and curve.shape[0] >= 2:
            ls = make_polyline(curve, CONFIG.spline_color)
            if ls is not None: draw.append(ls)
            p0, p1 = curve[0], curve[1]
            draw.append(make_arrow_from_to(p0, p1 + 19*(p1 - p0), [0.0, 1.0, 1.0], thin_factor=0.5))

        if unfolded_i is not None:
            draw.append(make_sphere(unfolded_i, r_unfolded, [1.0, 0.5, 0.0]))
            if show_green_to_orange:
                ds = _make_dashed_line(point_i, unfolded_i, [1.0, 0.5, 0.0], dash_len, gap_len)
                if ds is not None: draw.append(ds)

        if radial_proj_i is not None:
            draw.append(make_sphere(radial_proj_i, r_radial_proj, [0.6, 0.6, 0.6]))
            if show_green_to_orange:
                ds_g = _make_dashed_line(point_i, radial_proj_i, [0.6, 0.6, 0.6], dash_len, gap_len)
                if ds_g is not None: draw.append(ds_g)

    # Tiny red dots: as a point cloud to get the same tiny size as black pc
    all_red = []
    for s in ("top", "bottom", "left", "right"):
        Y = red_points_by_side.get(s, None)
        if Y is not None and len(Y) > 0:
            all_red.append(np.asarray(Y, dtype=float))
    if len(all_red) > 0:
        reds = np.vstack(all_red)
        red_pcd = o3d.geometry.PointCloud()
        red_pcd.points = o3d.utility.Vector3dVector(reds)
        red_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # tiny red points
        draw.append(red_pcd)

    o3d.visualization.draw_geometries(
        draw,
        window_name="Borders: orange splines + tiny red suggestions on plane G",
        width=1400, height=900
    )

def unfold_surface(metric_depth_csv: str,
                   hfov_deg: float = 66.0,
                   pose_kwargs=None):
    """
    Unfold the entire surface (every pixel) onto plane G using border reds.

    Steps:
      1) Calls unfold_image_borders(...) to compute dense red suggestions on
         TOP/RIGHT/BOTTOM/LEFT borders (one red per border pixel).
      2) For each interior pixel (i, j), fetch the four red points that share
         its row or column:
            T = red at (i=0,     j=j)      # top
            B = red at (i=H-1,   j=j)      # bottom
            L = red at (i=i,     j=0)      # left
            R = red at (i=i,     j=W-1)    # right
         (Note: bottom/left sequences in 'dense_red' were generated in reverse
          parameterization; we reindex them so T,B,L,R all line up with j or i.)
      3) Horizontal & vertical linear interpolation on plane G:
            t_x = j / (W-1)
            t_y = i / (H-1)
            Qh  = L + t_x * (R - L)        # along the row i
            Qv  = T + t_y * (B - T)        # along the column j
            Q   = 0.5 * (Qh + Qv)          # cross-consistent average
      4) Returns:
            unfolded_surface : (H, W, 3) float32 array of G-plane 3D points
            context          : dict with useful intermediates

    Parameters
    ----------
    metric_depth_csv : str
        Path to input metric depth CSV (HxW floating depths).
    hfov_deg : float
        Horizontal FOV used by project3DAndScale (kept for parity).
    pose_kwargs : dict | None
        Pose overrides forwarded to unfold_image_borders.

    Notes
    -----
    - All red points already lie on plane G, so the interpolation is planar.
    - This function does not re-run per-pixel unfolding; it strictly interpolates
      from border reds in O(HW) time with vectorization.

    """
    # 0) Build border reds & scene context
    borders, extras = unfold_image_borders(
        metric_depth_csv=metric_depth_csv,
        hfov_deg=hfov_deg,
        pose_kwargs=pose_kwargs,
        max_points=None,
    )
    dense_red = borders["dense_red"]  # side -> (Nd, 3)

    # 1) Image shape
    metric_depth = np.loadtxt(metric_depth_csv, delimiter=',', dtype=np.float32)
    H, W = metric_depth.shape[:2]

    # 2) Grab per-side dense reds and align their indexing with (i, j)
    #    TOP:    index j -> top[j]
    #    RIGHT:  index i -> right[i]
    #    BOTTOM: generated right->left, so index j -> bottom_aligned[j] = bottom[W-1 - j]
    #    LEFT:   generated bottom->top, so index i -> left_aligned[i]   = left[H-1  - i]
    top    = np.asarray(dense_red["top"],    dtype=float)           # (W, 3)
    right  = np.asarray(dense_red["right"],  dtype=float)           # (H, 3)
    bottom = np.asarray(dense_red["bottom"], dtype=float)           # (W, 3)
    left   = np.asarray(dense_red["left"],   dtype=float)           # (H, 3)

    if top.shape[0] != W or bottom.shape[0] != W:
        raise ValueError("Unexpected TOP/BOTTOM red densities (expected length = W).")
    if left.shape[0] != H or right.shape[0] != H:
        raise ValueError("Unexpected LEFT/RIGHT red densities (expected length = H).")

    bottom_aligned = bottom[::-1].copy()   # now index by j
    left_aligned   = left[::-1].copy()     # now index by i

    # 3) Vectorized cross-interpolation on plane G
    #    Build 2D grids of i and j, and pull corresponding border reds.
    ii = np.arange(H, dtype=float)[:, None]           # (H,1)
    jj = np.arange(W, dtype=float)[None, :]           # (1,W)
    tx = (jj / max(W - 1.0, 1.0))                     # (1,W)
    ty = (ii / max(H - 1.0, 1.0))                     # (H,1)

    # Broadcast L(i), R(i) to (H,W,3)
    L = left_aligned[:,  None, :]                     # (H,1,3)
    R = right[        :,  None, :]                    # (H,1,3)
    Qh = L + tx[..., None] * (R - L)                  # (H,W,3)

    # Broadcast T(j), B(j) to (H,W,3)
    T = top[          None, :, :]                     # (1,W,3)
    B = bottom_aligned[None, :, :]                    # (1,W,3)
    Qv = T + ty[..., None] * (B - T)                  # (H,W,3)

    # Final point per pixel: average the two linear predictions
    unfolded_surface = 0.5 * (Qh + Qv)
    unfolded_surface = unfolded_surface.astype(np.float32, copy=False)

    # 4) Optional: expose a little context for debugging/visualization
    context = dict(
        top=top, right=right, bottom=bottom_aligned, left=left_aligned,
        pcd=extras["pcd"],
        surface_interior_cg=extras["surface_interior_cg"],
        cg=extras["cg"],
        xyz_img=extras["xyz_img"],        # original black points
    )
    return unfolded_surface, context
