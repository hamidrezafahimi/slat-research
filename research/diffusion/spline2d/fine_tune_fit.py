"""
Fit a smooth B-spline surface to a point cloud by solving for control-point Z
given fixed control-point X,Y from a CSV grid. Balances data fidelity and
smoothness (thin-plate-like via second differences).

Usage (typical):
  python fit_spline_surface_to_pcd.py \
      --pcd /mnt/data/ds.pcd \
      --ctrl /mnt/data/fgs.csv \
      --grid-w 20 --grid-h 15 \
      --su 160 --sv 120 \
      --lambda-smooth 10.0 \
      --voxel 0.25 \
      --iters 2 \
      --out out_spline_fit \
      --visualize \
      --dst-grid-w 30 --dst-grid-h 22   # (optional) upsample initial-guess grid

CSV format:
  Either 2 or 3 columns.
    • If 3 columns (X,Y,Z): Z from CSV is used as the INITIAL GUESS (and then optimized).
      If --dst-grid-* upsamples, initial Z is bilinearly interpolated from CSV Z.
    • If 2 columns (X,Y): Z is initialized from the point cloud (kNN median).
Order is row-major by default: rows are v (j), columns are u (i).

Visualization (when --visualize is set):
  - Input PCD points     (cyan)
  - Fitted spline mesh   (orange)  + fitted control points (green)
  - Initial spline mesh  (gray)    + initial control points (magenta)
"""

from __future__ import annotations
import argparse, os, sys, math
from pathlib import Path
import numpy as np
import open3d as o3d

# SciPy is assumed to exist per user request
import scipy.sparse as sp
from scipy.sparse import csr_matrix, diags, kron, eye
from scipy.sparse.linalg import spsolve

# --------------------------- B-spline basics ---------------------------

def _uniform_clamped_knots(n: int, p: int) -> np.ndarray:
    if n <= 0:
        return np.array([0.0, 1.0], dtype=float)
    p = int(max(0, min(p, n - 1)))
    m = n + p + 1
    U = np.empty(m, dtype=float)
    U[:p + 1] = 0.0
    U[-(p + 1):] = 1.0
    num_interior = n - p - 1
    if num_interior > 0:
        interior = np.linspace(0.0, 1.0, num_interior + 2, dtype=float)[1:-1]
        U[p + 1: m - (p + 1)] = interior
    return U

def _find_span(n: int, p: int, u: float, U: np.ndarray) -> int:
    if u >= U[n]:
        return n - 1
    low, high = p, n
    mid = (low + high) // 2
    while not (U[mid] <= u < U[mid+1]):
        if u < U[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid

def _basis_funs(span: int, u: float, p: int, U: np.ndarray) -> np.ndarray:
    N = np.zeros(p+1, dtype=float)
    left = np.zeros(p+1, dtype=float)
    right = np.zeros(p+1, dtype=float)
    N[0] = 1.0
    for j in range(1, p+1):
        left[j] = u - U[span+1-j]
        right[j] = U[span+j] - u
        saved = 0.0
        for r in range(j):
            denom = right[r+1] + left[j-r]
            val = 0.0 if denom == 0.0 else N[r] / denom
            temp = val * right[r+1]
            N[r] = saved + temp
            saved = val * left[j-r]
        N[j] = saved
    return N

def bspline_surface_mesh_from_ctrl(ctrl_pts_flat: np.ndarray, grid_w: int, grid_h: int,
                                   su: int, sv: int) -> o3d.geometry.TriangleMesh:
    P = ctrl_pts_flat.reshape(grid_h, grid_w, 3).astype(float)  # [v=j, u=i]
    n_u, n_v = grid_w, grid_h
    p_u = min(3, max(0, n_u - 1))
    p_v = min(3, max(0, n_v - 1))

    if n_u < 2 or n_v < 2:
        X, Y, Z = P[...,0], P[...,1], P[...,2]
        uu = np.linspace(0, n_u - 1, su); vv = np.linspace(0, n_v - 1, sv)
        Uidx = np.clip(np.searchsorted(np.arange(n_u), uu) - 1, 0, max(0, n_u - 2))
        Vidx = np.clip(np.searchsorted(np.arange(n_v), vv) - 1, 0, max(0, n_v - 2))
        XX = np.zeros((sv, su)); YY = np.zeros((sv, su)); ZZ = np.zeros((sv, su))
        for a, j in enumerate(Vidx):
            v0 = j; v1 = min(j+1, n_v-1); tv = (vv[a]-v0) if n_v>1 else 0.0
            for b, i in enumerate(Uidx):
                u0 = i; u1 = min(i+1, n_u-1); tu = (uu[b]-u0) if n_u>1 else 0.0
                def bl(M):
                    return ((1-tu)*(1-tv)*M[v0,u0] + tu*(1-tv)*M[v0,u1] +
                            (1-tu)*tv*M[v1,u0] + tu*tv*M[v1,u1])
                XX[a,b]=bl(X); YY[a,b]=bl(Y); ZZ[a,b]=bl(Z)
        verts = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)
    else:
        U = _uniform_clamped_knots(n_u, p_u)
        V = _uniform_clamped_knots(n_v, p_v)
        us = np.linspace(0.0, 1.0, su)
        vs = np.linspace(0.0, 1.0, sv)
        Bu = np.zeros((su, n_u)); Bv = np.zeros((sv, n_v))
        for b, u in enumerate(us):
            span_u = _find_span(n_u, p_u, u, U)
            Bu[b, span_u-p_u: span_u+1] = _basis_funs(span_u, u, p_u, U)
        for a, v in enumerate(vs):
            span_v = _find_span(n_v, p_v, v, V)
            Bv[a, span_v-p_v: span_v+1] = _basis_funs(span_v, v, p_v, V)
        Xc, Yc, Zc = P[...,0], P[...,1], P[...,2]
        XX = Bv @ Xc @ Bu.T
        YY = Bv @ Yc @ Bu.T
        ZZ = Bv @ Zc @ Bu.T
        verts = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)

    tris=[]
    for j in range(sv-1):
        for i in range(su-1):
            v0=j*su+i; v1=v0+1; v2=v0+su; v3=v2+1
            tris.append([v0, v2, v1]); tris.append([v1, v2, v3])
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(np.asarray(verts,np.float64)),
        triangles=o3d.utility.Vector3iVector(np.asarray(tris,np.int32)),
    )
    mesh.compute_vertex_normals()
    return mesh

def _eval_component(Bu, Bv, C: np.ndarray) -> np.ndarray:
    return Bv @ C @ Bu.T

# --------------------------- Fitting / helpers ---------------------------

def _infer_grid_hw(n_ctrl: int) -> tuple[int,int]:
    best = (1, n_ctrl); best_diff = n_ctrl-1
    for h in range(1, int(math.sqrt(n_ctrl))+1):
        if n_ctrl % h == 0:
            w = n_ctrl // h
            if abs(w - h) < best_diff:
                best = (h,w); best_diff = abs(w-h)
    return best

def _load_ctrl_xy(csv_path: Path) -> tuple[np.ndarray, bool]:
    arr = np.loadtxt(str(csv_path), delimiter=",")
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < 2:
        raise ValueError("Control CSV must have at least 2 columns (x,y).")
    has_z = arr.shape[1] >= 3
    ctrl_arr = arr[:, :3] if has_z else np.c_[arr[:,0], arr[:,1], np.zeros(len(arr))]
    return ctrl_arr, has_z

def _nearest_z_init(ctrl_xy: np.ndarray, pts: np.ndarray, k: int = 20, radius: float | None = None) -> np.ndarray:
    from sklearn.neighbors import KDTree
    tree = KDTree(pts[:, :2])
    if radius is not None and radius > 0:
        ind = tree.query_radius(ctrl_xy[:, :2], r=radius)
        zs = np.zeros(len(ctrl_xy))
        for i, idxs in enumerate(ind):
            zs[i] = np.median(pts[idxs, 2]) if len(idxs) else np.nan
    else:
        _, idx = tree.query(ctrl_xy[:, :2], k=min(k, len(pts)))
        zs = np.median(pts[idx, 2], axis=1)
    if np.isnan(zs).any():
        A = np.c_[pts[:,0], pts[:,1], np.ones(len(pts))]
        x, *_ = np.linalg.lstsq(A, pts[:,2], rcond=None)  # z ≈ ax+by+c
        a,b,c = x
        mask = np.isnan(zs)
        zs[mask] = a*ctrl_xy[mask,0] + b*ctrl_xy[mask,1] + c
    return zs

def _build_bases(n_u, n_v, p_u, p_v, su, sv):
    U = _uniform_clamped_knots(n_u, p_u)
    V = _uniform_clamped_knots(n_v, p_v)
    us = np.linspace(0.0, 1.0, su)
    vs = np.linspace(0.0, 1.0, sv)
    Bu = np.zeros((su, n_u)); Bv = np.zeros((sv, n_v))
    for b, u in enumerate(us):
        span_u = _find_span(n_u, p_u, u, U)
        Bu[b, span_u-p_u: span_u+1] = _basis_funs(span_u, u, p_u, U)
    for a, v in enumerate(vs):
        span_v = _find_span(n_v, p_v, v, V)
        Bv[a, span_v-p_v: span_v+1] = _basis_funs(span_v, v, p_v, V)
    return U, V, Bu, Bv, us, vs

def _freeze_uv_from_xy(ctrl_xy_grid, Bu, Bv, pts, su, sv):
    Xc = ctrl_xy_grid[...,0]; Yc = ctrl_xy_grid[...,1]
    XX = _eval_component(Bu, Bv, Xc)
    YY = _eval_component(Bu, Bv, Yc)
    xy_samples = np.c_[XX.ravel(), YY.ravel()]
    try:
        from sklearn.neighbors import KDTree
        kdt = KDTree(xy_samples); _, idx = kdt.query(pts[:, :2], k=1)
        idx = idx[:,0]
    except Exception:
        def nn(x):
            d = np.sum((xy_samples - x[:2])**2, axis=1)
            return np.argmin(d)
        idx = np.array([nn(p) for p in pts])
    a = idx // su
    b = idx % su
    v = a / max(1, sv-1)
    u = b / max(1, su-1)
    return u, v

def _assemble_data_matrix(u, v, n_u, n_v, p_u, p_v, U, V):
    N = len(u)
    nnz_per_row = (p_u+1)*(p_v+1)
    rows = np.zeros(N*nnz_per_row, dtype=np.int64)
    cols = np.zeros(N*nnz_per_row, dtype=np.int64)
    vals = np.zeros(N*nnz_per_row, dtype=np.float64)
    cursor = 0
    for i in range(N):
        ui, vi = u[i], v[i]
        span_u = _find_span(n_u, p_u, ui, U)
        span_v = _find_span(n_v, p_v, vi, V)
        Nu = _basis_funs(span_u, ui, p_u, U)
        Nv = _basis_funs(span_v, vi, p_v, V)
        for a in range(p_v+1):
            j = span_v - p_v + a
            for b in range(p_u+1):
                iu = span_u - p_u + b
                col = j*n_u + iu
                rows[cursor] = i
                cols[cursor] = col
                vals[cursor] = Nv[a] * Nu[b]
                cursor += 1
    A = csr_matrix((vals, (rows, cols)), shape=(N, n_u*n_v))
    return A

def _laplacian_2nd_diff(n_u, n_v, weight=1.0):
    def D2(n):
        e = np.ones(n)
        data = np.array([e, -2*e, e])
        offsets = np.array([-1, 0, 1])
        return diags(data, offsets, shape=(n, n), format='csr')
    Du2 = D2(n_u); Dv2 = D2(n_v)
    L = sp.vstack([kron(eye(n_v, format='csr'), Du2, format='csr'),
                   kron(Dv2, eye(n_u, format='csr'), format='csr')], format='csr')
    return weight * L

# --------------------------- Grid upsampling ---------------------------

def _upsample_ctrl_grid_bilinear(ctrl_grid: np.ndarray, dst_w: int, dst_h: int) -> np.ndarray:
    """
    Bilinearly upsample a (H,W,3) control grid to (dst_h, dst_w, 3).
    Interpolates X, Y, Z independently on the grid (not along the B-spline surface).
    """
    src_h, src_w, _ = ctrl_grid.shape
    if src_h == dst_h and src_w == dst_w:
        return ctrl_grid.copy()

    # Parametric positions in source index space
    # Handle degenerate 1-sized dimensions
    xs = np.linspace(0.0, src_w-1, dst_w) if dst_w > 1 else np.array([0.0])
    ys = np.linspace(0.0, src_h-1, dst_h) if dst_h > 1 else np.array([0.0])
    xi = np.clip(xs, 0, src_w-1)
    yi = np.clip(ys, 0, src_h-1)

    x0 = np.floor(xi).astype(int); x1 = np.clip(x0+1, 0, src_w-1)
    y0 = np.floor(yi).astype(int); y1 = np.clip(y0+1, 0, src_h-1)
    tx = (xi - x0)[None, :]                 # shape (1, dst_w)
    ty = (yi - y0)[:, None]                 # shape (dst_h, 1)

    def interp_channel(C):
        # C shape: (src_h, src_w)
        C00 = C[y0[:,None], x0[None,:]]     # (dst_h, dst_w)
        C10 = C[y0[:,None], x1[None,:]]
        C01 = C[y1[:,None], x0[None,:]]
        C11 = C[y1[:,None], x1[None,:]]
        # Bilinear: (1-tx)(1-ty)C00 + tx(1-ty)C10 + (1-tx)ty C01 + tx ty C11
        return ((1-tx)*(1-ty)*C00 + tx*(1-ty)*C10 + (1-tx)*ty*C01 + tx*ty*C11)

    X = ctrl_grid[...,0]; Y = ctrl_grid[...,1]; Z = ctrl_grid[...,2]
    Xn = interp_channel(X); Yn = interp_channel(Y); Zn = interp_channel(Z)
    return np.dstack([Xn, Yn, Zn])

def _draw_safe(geoms, window_name="Open3D"):
    try:
        o3d.visualization.draw(geoms)
    except Exception:
        pure = []
        for g in geoms:
            if isinstance(g, dict) and "geometry" in g:
                pure.append(g["geometry"])
            else:
                pure.append(g)
        o3d.visualization.draw_geometries(pure, window_name=window_name)

# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--pcd", type=Path, required=True, help="Input point cloud")
    ap.add_argument("--ctrl", type=Path, required=True, help="CSV control grid (x,y or x,y,z)")
    ap.add_argument("--grid-w", type=int, default=None, help="Control grid width (u dimension, columns)")
    ap.add_argument("--grid-h", type=int, default=None, help="Control grid height (v dimension, rows)")
    ap.add_argument("--order", choices=["rowmajor","colmajor"], default="rowmajor", help="CSV control ordering")

    ap.add_argument("--dst-grid-w", type=int, default=None, help="(Optional) Destination grid width for upsampling")
    ap.add_argument("--dst-grid-h", type=int, default=None, help="(Optional) Destination grid height for upsampling")

    ap.add_argument("--su", type=int, default=160, help="Surface samples along u for UV freezing/meshing")
    ap.add_argument("--sv", type=int, default=120, help="Surface samples along v for UV freezing/meshing")
    ap.add_argument("--voxel", type=float, default=0.0, help="Voxel size for downsampling PCD (0=off)")
    ap.add_argument("--lambda-smooth", type=float, default=10.0, help="Smoothing weight λ")
    ap.add_argument("--init-radius", type=float, default=0.0, help="Radius (XY) for Z initialization if CSV lacks Z (0→kNN median)")
    ap.add_argument("--init-k", type=int, default=20, help="k for kNN Z initialization if CSV lacks Z")
    ap.add_argument("--iters", type=int, default=2, help="Reproject UV and re-solve iterations")
    ap.add_argument("--out", type=Path, default=Path("out_spline_fit"), help="Output directory")
    ap.add_argument("--visualize", action="store_true", help="Show Open3D visualization")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Load point cloud
    pcd = o3d.io.read_point_cloud(str(args.pcd))
    if pcd.is_empty():
        raise RuntimeError(f"Failed to read PCD: {args.pcd}")
    if args.voxel and args.voxel > 0:
        pcd = pcd.voxel_down_sample(args.voxel)
    pts = np.asarray(pcd.points)
    if pts.shape[1] != 3:
        raise RuntimeError("Point cloud must be 3D points")

    # Load control grid (X,Y,(Z?))
    ctrl, ctrl_has_z = _load_ctrl_xy(args.ctrl)  # (N,3), flag
    n_ctrl = ctrl.shape[0]
    if args.grid_w is None or args.grid_h is None:
        gh, gw = _infer_grid_hw(n_ctrl)
        print(f"[info] Inferred grid_h={gh}, grid_w={gw} from {n_ctrl} control points.")
        args.grid_h = gh; args.grid_w = gw
    if args.grid_w * args.grid_h != n_ctrl:
        raise ValueError(f"grid_w*grid_h != number of control points ({args.grid_w}*{args.grid_h} != {n_ctrl})")

    # Reorder into (grid_h, grid_w, 3) as row-major
    if args.order == "rowmajor":
        ctrl_grid = ctrl.reshape(args.grid_h, args.grid_w, 3).copy()
    else:
        ctrl_grid = ctrl.reshape(args.grid_w, args.grid_h, 3).transpose(1,0,2).copy()

    # Optional upsampling of the INITIAL GUESS grid
    dst_w, dst_h = args.dst_grid_w, args.dst_grid_h
    do_upsample = (dst_w is not None and dst_h is not None) and (dst_w > args.grid_w or dst_h > args.grid_h)
    if do_upsample:
        print(f"[info] Upsampling initial control grid from ({args.grid_w}x{args.grid_h}) → ({dst_w}x{dst_h})")
        ctrl_grid = _upsample_ctrl_grid_bilinear(ctrl_grid, dst_w=dst_w, dst_h=dst_h)
        # Update grid sizes post-upsampling
        args.grid_w, args.grid_h = dst_w, dst_h

    # Keep a copy of the *initial* control grid (after optional upsampling)
    initial_ctrl_grid = ctrl_grid.copy()

    # Initialize Z if CSV had no Z
    if ctrl_has_z:
        print("[info] Using Z from CSV as the initial guess"
              + (" (upsampled bilinearly)" if do_upsample else "") + ".")
    else:
        print("[info] CSV lacks Z; initializing Z by local median of PCD"
              + (" after XY upsampling" if do_upsample else "") + ".")
        init_z = _nearest_z_init(ctrl_grid.reshape(-1,3), pts, k=args.init_k,
                                 radius=(None if args.init_radius<=0 else args.init_radius))
        ctrl_grid[...,2] = init_z.reshape(args.grid_h, args.grid_w)
        initial_ctrl_grid = ctrl_grid.copy()

    # Degrees & bases (based on possibly upsampled control grid)
    n_u, n_v = args.grid_w, args.grid_h
    p_u = min(3, max(0, n_u-1))
    p_v = min(3, max(0, n_v-1))
    U, V, Bu, Bv, us_samp, vs_samp = _build_bases(n_u, n_v, p_u, p_v, args.su, args.sv)

    # Freeze UV for each point from XY-only surface
    u, v = _freeze_uv_from_xy(ctrl_grid, Bu, Bv, pts, args.su, args.sv)

    # Smoothness operator
    L = _laplacian_2nd_diff(n_u, n_v, weight=1.0)

    # Solve iteratively for Z-control values (starting from current ctrl_grid Z)
    z_ctrl = ctrl_grid[...,2].reshape(-1).copy()

    for it in range(max(1, args.iters)):
        print(f"[solve] Iteration {it+1}/{args.iters}")
        A = _assemble_data_matrix(u, v, n_u, n_v, p_u, p_v, U, V)
        z_obs = pts[:,2]

        ATA = (A.T @ A)
        reg = args.lambda_smooth * (L.T @ L)
        rhs = A.T @ z_obs
        M = ATA + reg
        z_ctrl = spsolve(M, rhs)

        ctrl_grid[...,2] = z_ctrl.reshape(args.grid_h, args.grid_w)

    # Build meshes for export/vis
    ctrl_pts_flat = ctrl_grid.reshape(-1,3)
    mesh_fitted = bspline_surface_mesh_from_ctrl(ctrl_pts_flat, args.grid_w, args.grid_h, args.su, args.sv)

    init_ctrl_flat = initial_ctrl_grid.reshape(-1,3)
    mesh_initial = bspline_surface_mesh_from_ctrl(init_ctrl_flat, args.grid_w, args.grid_h, args.su, args.sv)

    # Save outputs (fitted)
    np.savetxt(args.out / "ctrl_fitted_xyz.csv", ctrl_pts_flat, delimiter=",", fmt="%.6f")
    o3d.io.write_triangle_mesh(str(args.out / "spline_mesh.ply"), mesh_fitted, write_triangle_uvs=False)
    print(f"[ok] Wrote: {args.out/'ctrl_fitted_xyz.csv'} and {args.out/'spline_mesh.ply'}")

    # Evaluate distances (point→mesh) for fitted and initial
    try:
        d_fit = np.array(mesh_fitted.compute_point_cloud_distance(pcd), dtype=float)
        msg = (f"[fit] point→mesh (fitted): mean={d_fit.mean():.4f}, "
               f"rmse={math.sqrt((d_fit**2).mean()):.4f}, "
               f"p50={np.percentile(d_fit,50):.4f}, "
               f"p90={np.percentile(d_fit,90):.4f}, max={d_fit.max():.4f}")
        print(msg)
        np.savetxt(args.out / "point_to_mesh_distances_fitted.csv", d_fit, delimiter=",", fmt="%.6f")
    except Exception as e:
        print(f"[warn] Could not compute distances to fitted mesh: {e}")
    try:
        d_ini = np.array(mesh_initial.compute_point_cloud_distance(pcd), dtype=float)
        msg = (f"[init] point→mesh (initial): mean={d_ini.mean():.4f}, "
               f"rmse={math.sqrt((d_ini**2).mean()):.4f}, "
               f"p50={np.percentile(d_ini,50):.4f}, "
               f"p90={np.percentile(d_ini,90):.4f}, max={d_ini.max():.4f}")
        print(msg)
        np.savetxt(args.out / "point_to_mesh_distances_initial.csv", d_ini, delimiter=",", fmt="%.6f")
    except Exception as e:
        print(f"[warn] Could not compute distances to initial mesh: {e}")

    # Visualization
    if args.visualize:
        pcd_vis = o3d.geometry.PointCloud(pcd)
        pcd_vis.paint_uniform_color([0.2, 0.7, 1.0])       # cyan

        mesh_fit_vis = o3d.geometry.TriangleMesh(mesh_fitted)
        mesh_fit_vis.paint_uniform_color([0.95, 0.55, 0.20])  # orange

        ctrl_fit_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ctrl_pts_flat))
        ctrl_fit_pcd.paint_uniform_color([0.0, 0.9, 0.0])   # green

        mesh_ini_vis = o3d.geometry.TriangleMesh(mesh_initial)
        mesh_ini_vis.paint_uniform_color([0.6, 0.6, 0.6])   # gray

        ctrl_ini_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(init_ctrl_flat))
        ctrl_ini_pcd.paint_uniform_color([1.0, 0.0, 1.0])   # magenta

        geoms = [
            {"name":"pcd",     "geometry": pcd_vis},
            {"name":"fit",     "geometry": mesh_fit_vis},
            {"name":"ctrl_fit","geometry": ctrl_fit_pcd},
            {"name":"init",    "geometry": mesh_ini_vis},
            {"name":"ctrl_ini","geometry": ctrl_ini_pcd},
        ]
        print("[viz] Colors — PCD: cyan, Fitted: orange+green, Initial: gray+magenta")
        _draw_safe(geoms, window_name="Spline fit: data vs initial vs fitted")

if __name__ == "__main__":
    main()
