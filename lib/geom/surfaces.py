import numpy as np, open3d as o3d

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

# infer grid sizes (unique Xs and Ys with tolerance)
def unique_sorted_with_tol(a, atol):
    a_sorted = np.sort(a)
    uniq = [a_sorted[0]]
    for v in a_sorted[1:]:
        if abs(v - uniq[-1]) > atol:
            uniq.append(v)
    return np.asarray(uniq, dtype=float)

def infer_grid(ctrl_points,
        tol: float = 0.59):
    P = np.asarray(ctrl_points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or P.size == 0:
        raise ValueError("ctrl_points must be a non-empty (N,3) array")
    xs, ys = P[:, 0], P[:, 1]

    # extents
    W = float(xs.max() - xs.min())
    H = float(ys.max() - ys.min())

    tol_x = max(tol, 1e-8 * max(1.0, W))
    tol_y = max(tol, 1e-8 * max(1.0, H))
    ux = unique_sorted_with_tol(xs, tol_x)  # gw
    uy = unique_sorted_with_tol(ys, tol_y)  # gh
    gw, gh = len(ux), len(uy)
    if gw * gh != len(P):
        raise ValueError(f"Control net is not a full grid: {len(P)} pts vs {gw}×{gh}")
    return gw, gh


def generate_spline(ctrl_points: np.ndarray,
                       samples_u: int = 40, samples_v: int = 40):
    """
    Build a B-spline surface directly from control points (x,y,z).

    Returns:
        mesh: o3d TriangleMesh
        W, H: XY extents of the control net
    """
    P = np.asarray(ctrl_points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or P.size == 0:
        raise ValueError("ctrl_points must be a non-empty (N,3) array")
    xs, ys = P[:, 0], P[:, 1]

    # extents
    W = float(xs.max() - xs.min())
    H = float(ys.max() - ys.min())

    gw, gh = infer_grid(ctrl_points)

    # reorder control points into row-major (v=Y first, then u=X)
    # i.e., sort by (y, x) ascending, then reshape (gh, gw, 3)
    order = np.lexsort((xs, ys))          # primary: ys, secondary: xs
    ctrl_sorted = P[order, :]
    ctrl_grid = ctrl_sorted.reshape(gh, gw, 3)

    # build mesh with your sampler
    mesh = bspline_surface_mesh_from_ctrl(ctrl_pts_flat=ctrl_grid, grid_w=gw, grid_h=gh, su=samples_u, sv=samples_v)
    return mesh, W, H

def cg_centeric_xy_spline(cloud_pts: np.ndarray,
                       grid_w: int = 6, grid_h: int = 4,
                       samples_u: int = 40, samples_v: int = 40,
                       margin: float = 0.02):
    W, H, center_xy = _extent_wh_center_xy(cloud_pts)
    W *= (1.0 + margin); H *= (1.0 + margin)
    z0 = float(cloud_pts[:, 2].mean())
    ctrl_pts = _ctrl_points_grid(grid_w, grid_h, W, H, center_xy, z0)
    mesh = _flat_patch_mesh(center_xy, W, H, z0, samples_u, samples_v)
    return mesh, ctrl_pts, W, H, center_xy, z0

def bspline_surface_mesh_from_ctrl(ctrl_pts_flat: np.ndarray, grid_w: int, grid_h: int,
                                    su: int, sv: int) -> o3d.geometry.TriangleMesh:
    """Build a smooth clamped tensor-product B-spline surface (degree ≤ 3) from a (grid_h*grid_w,3) control grid.
       The spline degree is chosen adaptively per axis: p_u = min(3, grid_w-1), p_v = min(3, grid_h-1).
    """
    P = ctrl_pts_flat.reshape(grid_h, grid_w, 3).astype(float)  # [v=j(row), u=i(col)]
    n_u, n_v = grid_w, grid_h

    # Choose highest valid degrees up to cubic so smaller grids (e.g., 3x3) still produce a spline.
    p_u = min(3, max(0, n_u - 1))
    p_v = min(3, max(0, n_v - 1))

    # Degenerate fallback only if we can't form a surface (need at least 2 control points per axis).
    if n_u < 2 or n_v < 2:
        # Fallback: bilinear upsample using control grid directly
        X = P[..., 0]; Y = P[..., 1]; Z = P[..., 2]
        uu = np.linspace(0, n_u - 1, su); vv = np.linspace(0, n_v - 1, sv)
        Uidx = np.clip(np.searchsorted(np.arange(n_u), uu) - 1, 0, max(0, n_u - 2))
        Vidx = np.clip(np.searchsorted(np.arange(n_v), vv) - 1, 0, max(0, n_v - 2))
        XX = np.zeros((sv, su)); YY = np.zeros((sv, su)); ZZ = np.zeros((sv, su))
        for a, j in enumerate(Vidx):
            v0 = j; v1 = min(j + 1, n_v - 1); tv = (vv[a] - v0) if n_v > 1 else 0.0
            for b, i in enumerate(Uidx):
                u0 = i; u1 = min(i + 1, n_u - 1); tu = (uu[b] - u0) if n_u > 1 else 0.0
                def bl(M):
                    return ((1 - tu) * (1 - tv) * M[v0, u0] +
                            tu * (1 - tv) * M[v0, u1] +
                            (1 - tu) * tv * M[v1, u0] +
                            tu * tv * M[v1, u1])
                XX[a, b] = bl(X); YY[a, b] = bl(Y); ZZ[a, b] = bl(Z)
        verts = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)
    else:
        # Open-uniform, clamped knot vectors with chosen degrees
        U = _uniform_clamped_knots(n_u, p_u)
        V = _uniform_clamped_knots(n_v, p_v)

        us = np.linspace(0.0, 1.0, su)
        vs = np.linspace(0.0, 1.0, sv)

        # Precompute basis across all samples
        Bu = np.zeros((su, n_u))
        Bv = np.zeros((sv, n_v))

        for b, u in enumerate(us):
            span_u = _find_span(n_u, p_u, u, U)
            Nu = _basis_funs(span_u, u, p_u, U)  # length p_u+1
            Bu[b, span_u - p_u: span_u + 1] = Nu

        for a, v in enumerate(vs):
            span_v = _find_span(n_v, p_v, v, V)
            Nv = _basis_funs(span_v, v, p_v, V)  # length p_v+1
            Bv[a, span_v - p_v: span_v + 1] = Nv

        # Evaluate S(v,u) = sum_j sum_i Bv[a,j] * Bu[b,i] * P[j,i]
        Xc, Yc, Zc = P[..., 0], P[..., 1], P[..., 2]
        XX = Bv @ Xc @ Bu.T
        YY = Bv @ Yc @ Bu.T
        ZZ = Bv @ Zc @ Bu.T
        verts = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)

    # Connectivity (unchanged)
    tris = []
    for j in range(sv - 1):
        for i in range(su - 1):
            v0 = j * su + i; v1 = v0 + 1; v2 = v0 + su; v3 = v2 + 1
            tris.append([v0, v2, v1]); tris.append([v1, v2, v3])

    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(np.asarray(verts, np.float64)),
        triangles=o3d.utility.Vector3iVector(np.asarray(tris,  np.int32)),
    )
    mesh.compute_vertex_normals()
    return mesh

def _uniform_clamped_knots(n: int, p: int):
    # n = number of control points; p = degree
    m = n + p + 1
    U = np.zeros(m, dtype=float); U[-(p+1):] = 1.0
    r = n - p - 1  # number of internal knots
    if r > 0:
        for k in range(1, r + 1):
            U[p + k] = k / (r + 1)
    return U

def _find_span(n: int, p: int, u: float, U: np.ndarray):
    # n = number of control points (n), valid span in [p, n-1]
    if u >= U[n]:  # clamp to last span
        return n - 1
    low, high = p, n - 1
    while low <= high:
        mid = (low + high) // 2
        if u < U[mid]:
            high = mid - 1
        elif u >= U[mid + 1]:
            low = mid + 1
        else:
            return mid
    return max(p, min(n - 1, low))


def _basis_funs(span: int, u: float, p: int, U: np.ndarray):
    N = np.zeros(p + 1, dtype=float)
    left = np.zeros(p + 1, dtype=float)
    right = np.zeros(p + 1, dtype=float)
    N[0] = 1.0
    for j in range(1, p + 1):
        left[j] = u - U[span + 1 - j]
        right[j] = U[span + j] - u
        saved = 0.0
        for r in range(0, j):
            denom = right[r + 1] + left[j - r]
            term = 0.0 if denom == 0.0 else N[r] / denom
            temp = term * right[r + 1]
            N[r] = saved + temp
            saved = term * left[j - r]
        N[j] = saved
    return N

# Precise
def project_external_along_normals_noreject(ext_pts: np.ndarray, mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """
    Project each external point onto the mesh by casting rays along ± the
    local surface normal (nearest vertex normal to the external point).
    Uses open3d.t.geometry.RaycastingScene when available; otherwise falls back
    to nearest-vertex positions.
    """
    if ext_pts.size == 0 or len(mesh.triangles) == 0:
        return np.empty((0, 3), dtype=float)

    n = nearest_vertex_normals(ext_pts, mesh)

    try:
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(tmesh)

        origins = ext_pts.astype(np.float32)
        dirs_pos = n.astype(np.float32)      # +n
        dirs_neg = (-n).astype(np.float32)   # -n

        rays_pos = np.concatenate([origins, dirs_pos], axis=1)  # [N,6]
        rays_neg = np.concatenate([origins, dirs_neg], axis=1)  # [N,6]
        rays = np.vstack([rays_pos, rays_neg]).astype(np.float32)  # [2N,6]

        t_rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
        ans = scene.cast_rays(t_rays)
        t_hit = ans["t_hit"].numpy()  # [2N], inf if no hit

        N = origins.shape[0]
        t_pos = t_hit[:N]
        t_neg = t_hit[N:]

        hit_pos = origins + dirs_pos * t_pos[:, None]
        hit_neg = origins + dirs_neg * t_neg[:, None]
        mask_pos = np.isfinite(t_pos)
        mask_neg = np.isfinite(t_neg)

        chosen = np.empty_like(origins, dtype=np.float32)
        chosen[:] = np.nan

        both = mask_pos & mask_neg
        only_pos = mask_pos & ~mask_neg
        only_neg = ~mask_pos & mask_neg

        if np.any(both):
            pick_pos = t_pos[both] <= t_neg[both]
            idx_both = np.where(both)[0]
            idx_pos = idx_both[pick_pos]
            idx_neg = idx_both[~pick_pos]
            chosen[idx_pos] = hit_pos[idx_pos]
            chosen[idx_neg] = hit_neg[idx_neg]
        if np.any(only_pos):
            idx = np.where(only_pos)[0]
            chosen[idx] = hit_pos[idx]
        if np.any(only_neg):
            idx = np.where(only_neg)[0]
            chosen[idx] = hit_neg[idx]

        # Fallbacks for misses: snap to nearest vertex
        misses = np.isnan(chosen).any(axis=1)
        if np.any(misses):
            verts = np.asarray(mesh.vertices)
            try:
                from scipy.spatial import cKDTree
                tree = cKDTree(verts)
                _, idx = tree.query(origins[misses], k=1)
            except Exception:
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(verts)
                kdt = o3d.geometry.KDTreeFlann(pc)
                idx_list = []
                for p in origins[misses]:
                    cnt, inds, _ = kdt.search_knn_vector_3d(p.astype(float), 1)
                    idx_list.append(inds[0] if cnt > 0 else 0)
                idx = np.asarray(idx_list, dtype=int)
            chosen[misses] = verts[idx].astype(np.float32)

        return chosen.astype(np.float64)

    except Exception as e:
        print(f"[WARN] RaycastingScene unavailable or failed ({e}); falling back to nearest-vertex projection.", file=sys.stderr)

    # Fallback: nearest vertex positions
    verts = np.asarray(mesh.vertices)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(verts)
        _, idx = tree.query(ext_pts, k=1)
    except Exception:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(verts)
        kdt = o3d.geometry.KDTreeFlann(pc)
        idx = []
        for p in ext_pts:
            cnt, inds, _ = kdt.search_knn_vector_3d(p.astype(float), 1)
            idx.append(inds[0] if cnt > 0 else 0)
        idx = np.asarray(idx, dtype=int)
    return verts[idx].astype(float)

def project_external_along_normals_andreject(
    ext_pts: np.ndarray,
    mesh: o3d.geometry.TriangleMesh,
    angle_tol_deg: float = 50.0,
    tangential_tol_rel: float = 2e-2,
    near_vertex_tol_scale: float = 6.0,
    keep_length: bool = False,
):
    """
    Fast normal-ray projection that ONLY keeps points whose ±normal ray actually hits the mesh
    and that are plausibly 'over' the surface (coverage gating via nearest vertex cKDTree).

    Returns:
      - if keep_length=False (default): (proj_pts (M,3), valid_idx (M,))
      - if keep_length=True: proj_full (N,3) with NaNs at invalid rows
    """
    if ext_pts.size == 0 or len(mesh.triangles) == 0:
        return (np.empty((0, 3), float), np.empty((0,), int)) if not keep_length else np.empty((0,3), float)

    # Ensure normals
    if len(mesh.vertex_normals) == 0:
        mesh.compute_vertex_normals()

    verts = np.asarray(mesh.vertices, dtype=float)
    vnorms = np.asarray(mesh.vertex_normals, dtype=float)
    tris = np.asarray(mesh.triangles, dtype=np.int32)

    # Scene scale
    mn, mx = verts.min(axis=0), verts.max(axis=0)
    diag = float(np.linalg.norm(mx - mn)) or 1.0
    tangential_tol = tangential_tol_rel * diag
    cos_thresh = float(np.cos(np.deg2rad(angle_tol_deg)))

    # ---- Nearest-vertex lookup (prefer SciPy cKDTree; fall back to Open3D KDTreeFlann) ----
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(verts)
        # nearest vertex per external point
        _, vidx = tree.query(ext_pts, k=1)
        vidx = vidx.astype(np.int32)

        # per-vertex nearest-neighbor distance (exclude self) for local scale
        # k=2 gives [self, nn]; we take the second
        d2, _ = tree.query(verts, k=2)
        nn_dist = d2[:, 1]  # (V,)
        # Replace zeros (isolated or duplicate vertices) with global tiny scale
        nn_dist[~np.isfinite(nn_dist) | (nn_dist <= 0)] = 1e-6 * diag
    except Exception:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(verts)
        kdt = o3d.geometry.KDTreeFlann(pc)
        vidx = np.empty((ext_pts.shape[0],), dtype=np.int32)
        for i, p in enumerate(ext_pts):
            cnt, inds, _ = kdt.search_knn_vector_3d(p.astype(float), 1)
            vidx[i] = inds[0] if cnt > 0 else 0
        # coarse global scale as fallback
        nn_dist = np.full((verts.shape[0],), 1e-3 * diag, dtype=float)

    v_near = verts[vidx]           # (N,3)
    n_near = vnorms[vidx]          # (N,3)
    # normalize normals (robust)
    n_len = np.linalg.norm(n_near, axis=1, keepdims=True)
    n_len[n_len == 0] = 1.0
    n_near = n_near / n_len

    # Vector from nearest vertex to external point
    v = ext_pts - v_near
    v_len = np.linalg.norm(v, axis=1)
    # normal component
    dn = np.einsum("ij,ij->i", v, n_near)
    # tangential component magnitude
    vt = v - dn[:, None] * n_near
    vt_len = np.linalg.norm(vt, axis=1)

    # Gate 1: alignment with normal and small tangential offset (point roughly above the surface patch)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = np.abs(dn) / (v_len + 1e-12)
    gate_align = cosang >= cos_thresh
    gate_tan   = vt_len <= (tangential_tol + near_vertex_tol_scale * nn_dist[vidx])
    candidate = gate_align & gate_tan & np.isfinite(v_len)

    if not np.any(candidate):
        if keep_length:
            out = np.full((ext_pts.shape[0], 3), np.nan, dtype=float)
            return out
        return np.empty((0, 3), float), np.empty((0,), int)

    # Choose ray directions toward the surface: -sign(dn) * n
    sgn = np.sign(dn)
    sgn[sgn == 0] = 1.0
    dirs = (-sgn[:, None]) * n_near

    # ---- Raycast only the candidates (tensor path); fallback approximates with plane hit ----
    hits_ok = np.zeros(ext_pts.shape[0], dtype=bool)
    hit_pts = np.full((ext_pts.shape[0], 3), np.nan, dtype=float)

    has_tensor = hasattr(o3d, "t") and hasattr(o3d.t, "geometry") and hasattr(o3d.t.geometry, "RaycastingScene")
    cand_idx = np.nonzero(candidate)[0]
    O = ext_pts[cand_idx].astype(np.float32)
    D = dirs[cand_idx].astype(np.float32)

    if has_tensor:
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(tmesh)
        rays = np.concatenate([O, D], axis=1).astype(np.float32)  # [M,6]
        ans = scene.cast_rays(o3d.core.Tensor(rays))
        t_hit = ans["t_hit"].numpy()  # inf if no hit
        ok = np.isfinite(t_hit)
        H = O + D * t_hit[:, None]

        # Gate 2: require the hit to land near the nearest vertex (local scale)
        vnear_c = v_near[cand_idx]
        local_tol = near_vertex_tol_scale * nn_dist[vidx[cand_idx]]
        near_ok = np.linalg.norm(H - vnear_c, axis=1) <= (local_tol + tangential_tol)

        final_ok = ok & near_ok
        hit_pts[cand_idx[final_ok]] = H[final_ok].astype(np.float64)
        hits_ok[cand_idx[final_ok]] = True
    else:
        # Fallback: approximate by intersecting with the tangent plane at the nearest vertex
        # (keeps the coverage gates; doesn’t guarantee a true mesh hit but avoids boundary snaps)
        n_c = n_near[cand_idx]
        v_c = v[cand_idx]
        # move along -dn to the plane through v_near with normal n
        H = ext_pts[cand_idx] - (np.einsum("ij,ij->i", v_c, n_c))[:, None] * n_c
        vnear_c = v_near[cand_idx]
        local_tol = near_vertex_tol_scale * nn_dist[vidx[cand_idx]]
        near_ok = np.linalg.norm(H - vnear_c, axis=1) <= (local_tol + tangential_tol)
        final_ok = near_ok  # plane “hit” accepted if near the vertex
        hit_pts[cand_idx[final_ok]] = H[final_ok].astype(np.float64)
        hits_ok[cand_idx[final_ok]] = True

    valid_idx = np.nonzero(hits_ok)[0]
    proj_pts = hit_pts[valid_idx]

    if keep_length:
        out = np.full((ext_pts.shape[0], 3), np.nan, dtype=float)
        out[valid_idx] = proj_pts
        return out

    return proj_pts, valid_idx

def nearest_vertex_normals(query_pts: np.ndarray, mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """Return per-point normals taken from the nearest mesh vertex for each query point."""
    verts = np.asarray(mesh.vertices)
    vnorms = np.asarray(mesh.vertex_normals)
    if vnorms.shape != verts.shape or vnorms.size == 0:
        mesh.compute_vertex_normals()
        vnorms = np.asarray(mesh.vertex_normals)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(verts)
        _, idx = tree.query(query_pts, k=1)
    except Exception:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(verts)
        kdt = o3d.geometry.KDTreeFlann(pc)
        idx = []
        for p in query_pts:
            cnt, inds, _ = kdt.search_knn_vector_3d(p.astype(float), 1)
            idx.append(inds[0] if cnt > 0 else 0)
        idx = np.asarray(idx, dtype=int)
    n = vnorms[idx]
    lens = np.linalg.norm(n, axis=1, keepdims=True)
    bad = lens[:, 0] < 1e-12
    if np.any(bad):
        n[bad] = np.array([0.0, 0.0, 1.0])
        lens[bad] = 1.0
    n = n / lens
    return n

