import numpy as np
import open3d as o3d
try:
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def add_grid_and_axes(extent=50, step=1):
    """Return a grey XY-grid and XYZ axes for context."""
    pts, lines = [], []
    for y in np.arange(-extent, extent+1e-6, step):
        pts += [[-extent, y, 0], [ extent, y, 0]]
        lines.append([len(pts)-2, len(pts)-1])
    for x in np.arange(-extent, extent+1e-6, step):
        pts += [[x, -extent, 0], [ x, extent, 0]]
        lines.append([len(pts)-2, len(pts)-1])
    grid = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines)
    )
    grid.colors = o3d.utility.Vector3dVector([[0.2,0.2,0.2]] * len(lines))
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    return grid, axes

def create_normal_lines(pcd, voxel_size=0.02, normal_length=1.0):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20)
    )
    pts = np.asarray(pcd_down.points)
    nrm = np.asarray(pcd_down.normals)
    seg_pts, seg_lines = [], []
    for i, (p, v) in enumerate(zip(pts, nrm)):
        seg_pts.append(p)
        seg_pts.append(p + v * normal_length)
        seg_lines.append([2*i, 2*i+1])
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(seg_pts),
        lines=o3d.utility.Vector2iVector(seg_lines)
    )
    ls.colors = o3d.utility.Vector3dVector([[0,0,1]] * len(seg_lines))
    return ls

def print_report(query_pts, best_in_idxs, best_out_idxs,
                 all_normals, all_dists, all_projs):
    H, W, k, _ = all_normals.shape
    sep = "-" * 60
    print("\nProjection Report:\n" + sep)
    for i in range(H):
        for j in range(W):
            bi, bo = best_in_idxs[i,j], best_out_idxs[i,j]
            q = query_pts[i,j]
            print(f"Query[{i},{j}] = {q.tolist()}")
            if bi >= 0:
                n = all_normals[i,j,bi]
                d = all_dists[i,j,bi]
                p = all_projs[i,j,bi]
                print(f"  Inside  → idx={bi:2d}, dist={d:7.3f}, "
                      f"normal={n.round(3).tolist()}, proj={p.round(3).tolist()}")
            else:
                print("  Inside  → none")
            if bo >= 0:
                n = all_normals[i,j,bo]
                d = all_dists[i,j,bo]
                p = all_projs[i,j,bo]
                print(f"  Outside → idx={bo:2d}, dist={d:7.3f}, "
                      f"normal={n.round(3).tolist()}, proj={p.round(3).tolist()}")
            else:
                print("  Outside → none")
        print(sep)

def xyz_image_to_o3d_pcd(xyz: np.ndarray, *, drop_invalid: bool = True) -> o3d.geometry.PointCloud:
    """
    Convert an H×W×3 (or N×3) array of XYZ coordinates to an Open3D PointCloud.

    Parameters
    ----------
    xyz : np.ndarray
        Array of shape (H, W, 3) or (N, 3). Units are up to you (e.g., meters).
        Invalid/missing points should be np.nan.
    drop_invalid : bool, default True
        If True, remove any rows containing NaN/Inf before creating the point cloud.

    Returns
    -------
    o3d.geometry.PointCloud
        Unorganized point cloud (Open3D does not keep H/W organization).
    """
    if xyz.ndim == 3 and xyz.shape[2] == 3:
        pts = xyz.reshape(-1, 3)
    elif xyz.ndim == 2 and xyz.shape[1] == 3:
        pts = xyz
    else:
        raise ValueError("xyz must be (H, W, 3) or (N, 3)")

    pts = np.asarray(pts, dtype=np.float64)

    if drop_invalid:
        mask = np.isfinite(pts).all(axis=1)
        pts = pts[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

def make_flat_plane_xyz(H: int, W: int, size_m: float = 10.0, centered: bool = True, noise_std: float = 0.0):
    """
    Create an H×W×3 array of XYZ points on a flat plane.

    Parameters
    ----------
    H, W : int
        Image height and width.
    size_m : float, default 10.0
        Side length of the square plane in meters.
    centered : bool, default True
        If True, plane spans [-size/2, size/2] in X and Y.
        If False, plane spans [0, size] in X and Y.
    noise_std : float, default 0.0
        Optional Gaussian noise (meters) added to Z (and a tiny XY) to simulate roughness.

    Returns
    -------
    xyz : np.ndarray, shape (H, W, 3), dtype float32
        Points on the z=0 plane.
    """
    if centered:
        x = np.linspace(-size_m/2, size_m/2, W, dtype=np.float32)
        y = np.linspace(-size_m/2, size_m/2, H, dtype=np.float32)
    else:
        x = np.linspace(0.0, size_m, W, dtype=np.float32)
        y = np.linspace(0.0, size_m, H, dtype=np.float32)

    X, Y = np.meshgrid(x, y)           # X: (H,W), Y: (H,W)
    Z = np.zeros_like(X, dtype=np.float32)

    if noise_std > 0:
        # Small XY jitter (1/10 of Z noise) plus Z roughness
        rng = np.random.default_rng()
        X = X + (noise_std * 0.1) * rng.standard_normal(X.shape, dtype=np.float32)
        Y = Y + (noise_std * 0.1) * rng.standard_normal(Y.shape, dtype=np.float32)
        Z = Z + noise_std * rng.standard_normal(Z.shape, dtype=np.float32)

    xyz = np.stack([X, Y, Z], axis=-1).astype(np.float32)  # (H, W, 3)
    return xyz

def pcd_to_image(pcd_file, H, W):
    pcd = o3d.io.read_point_cloud(pcd_file)
    pcd_data = np.asarray(pcd.points)
    mask = ~np.all(pcd_data == [0, 0, 0], axis=1)
    filtered_points = pcd_data[mask]
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    pcd_data = np.asarray(pcd.points)
    if pcd_data.shape[1] != 3:
        raise ValueError("pcd_data must have shape (N, 3), where N is the number of points.")
    
    if pcd_data.shape[0] != H * W:
        raise ValueError(f"Number of points in pcd_data ({pcd_data.shape[0]}) does not match the expected size (H * W = {H * W}).")

    image = pcd_data.reshape(H, W, 3)

    return image

def crop_corner(arr, corner, percentage):
    H, W, _ = arr.shape
    h_crop = int(H * percentage)
    w_crop = int(W * percentage)

    if corner == 'up-left':
        return arr[:h_crop, :w_crop, :]
    elif corner == 'up-right':
        return arr[:h_crop, -w_crop:, :]
    elif corner == 'bottom-left':
        return arr[-h_crop:, :w_crop, :]
    elif corner == 'bottom-right':
        return arr[-h_crop:, -w_crop:, :]
    else:
        raise ValueError("corner must be one of 'up-left', 'up-right', 'bottom-left', 'bottom-right'")

def _normalize_rows(x, eps=1e-12):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n

def project_points_multi_fast(
    pcd,
    query_pts,
    k=10,
    just_vals=False,
    get_norm=False,
    batch_size=50_000,
    dtype=np.float32,
    print_log=False,
    visualize=False,
    just_proj = False
):
    """
    Fast batched variant with optional logging and visualization.

    Returns:
      - if just_vals: (H, W, 1) float32 of signed distances
      - if get_norm:  (H, W, 3) float32 normal-distance vectors
      - else:         (H, W, 1) object array of dicts (matching original)
    """
    if not pcd.has_normals():
        raise ValueError("Estimate normals on the point cloud first.")

    # Pull data once, normalize, cast to float32 for bandwidth
    pts   = np.asarray(pcd.points, dtype=dtype)
    norms = _normalize_rows(np.asarray(pcd.normals, dtype=dtype))

    n_pts = pts.shape[0]
    k_eff = int(min(k, n_pts))

    # Bounds for inside/outside classification (XY box)
    pts_xy = pts[:, :2]
    min_x, max_x = pts_xy[:, 0].min(), pts_xy[:, 0].max()
    min_y, max_y = pts_xy[:, 1].min(), pts_xy[:, 1].max()

    H, W, _ = query_pts.shape
    Q = H * W
    q_all = np.asarray(query_pts, dtype=dtype).reshape(Q, 3)

    # Build KD-tree once
    if not _HAS_SCIPY:
        raise ImportError(
            "SciPy is required for the fast batched KD-tree path. "
            "Install it or switch to sklearn NearestNeighbors."
        )
    tree = cKDTree(pts)

    # Output flat buffers (later reshaped)
    out_dist = np.full((Q,), np.nan, dtype=dtype)
    out_nvec = np.full((Q, 3), np.nan, dtype=dtype)
    out_proj = np.full((Q, 3), np.nan, dtype=dtype)
    out_idx  = np.full((Q,), -1, dtype=np.int32)

    # Optional per-k caches (only if logging or viz)
    keep_candidates = (print_log or visualize)
    if keep_candidates:
        all_normals_flat = np.zeros((Q, k_eff, 3), dtype=dtype)
        all_dists_flat   = np.zeros((Q, k_eff),    dtype=dtype)
        all_projs_flat   = np.zeros((Q, k_eff, 3), dtype=dtype)
        best_in_flat     = np.full((Q,), -1, dtype=np.int32)
        best_out_flat    = np.full((Q,), -1, dtype=np.int32)

    # Process queries in batches to limit memory
    for start in range(0, Q, batch_size):
        end = min(start + batch_size, Q)
        qb = q_all[start:end]  # (B,3)
        B = qb.shape[0]

        # KNN indices for the whole batch
        _, idxs = tree.query(qb, k=k_eff, workers=-1)
        idxs = np.asarray(idxs, dtype=np.intp)
        if idxs.ndim == 1:  # ensure (B, k)
            idxs = idxs[:, None]

        # Gather neighbor points and normals
        neigh_pts = pts[idxs]      # (B,k,3)
        neigh_n   = norms[idxs]    # (B,k,3)

        # Signed distances and foot points
        vecs   = qb[:, None, :] - neigh_pts                  # (B,k,3)
        dists  = np.einsum("bki,bki->bk", vecs, neigh_n)     # (B,k)
        projs  = qb[:, None, :] - dists[..., None] * neigh_n # (B,k,3)

        # Inside/outside mask by XY bounds
        px = projs[..., 0]
        py = projs[..., 1]
        in_mask  = (px >= min_x) & (px <= max_x) & (py >= min_y) & (py <= max_y)
        out_mask = ~in_mask

        # Best inside / outside by smallest |distance|
        absd = np.abs(dists)

        absd_in = np.where(in_mask, absd, np.inf)
        bi = np.argmin(absd_in, axis=1)          # (B,)
        best_in = absd_in[np.arange(B), bi]
        has_in = np.isfinite(best_in)
        d_in = dists[np.arange(B), bi]
        d_in[~has_in] = np.inf

        absd_out = np.where(out_mask, absd, np.inf)
        bo = np.argmin(absd_out, axis=1)
        best_out = absd_out[np.arange(B), bo]
        has_out = np.isfinite(best_out)
        d_out = dists[np.arange(B), bo]
        d_out[~has_out] = np.inf

        # Final selection between inside/outside
        choose_in = (np.abs(d_in) <= np.abs(d_out))
        sel_idx   = np.where(choose_in, bi, bo)         # (B,)
        sel_dist  = np.where(choose_in, d_in, d_out)    # (B,)

        valid = np.isfinite(sel_dist)
        sel_idx_final  = np.where(valid, sel_idx, -1)
        sel_dist_final = np.where(valid, sel_dist, np.nan)

        # Write outputs
        out_dist[start:end] = sel_dist_final
        out_idx[start:end]  = sel_idx_final

        if valid.any():
            vi = np.where(valid)[0]
            kk = sel_idx_final[vi]

            n_sel = neigh_n[vi, kk]            # (V,3)
            p_sel = projs[vi, kk]              # (V,3)
            d_sel = sel_dist_final[vi][:, None]# (V,1)

            out_nvec[start:end][vi] = n_sel * d_sel
            out_proj[start:end][vi] = p_sel

        # Keep candidate fields if requested
        if keep_candidates:
            all_normals_flat[start:end] = neigh_n
            all_dists_flat[start:end]   = dists
            all_projs_flat[start:end]   = projs
            best_in_flat[start:end]     = bi
            best_out_flat[start:end]    = bo

    # Quick-return formats
    if just_vals:
        return out_dist.reshape(H, W, 1).astype(np.float32)

    if get_norm:
        return out_nvec.reshape(H, W, 3).astype(np.float32)
    
    if just_proj:
        return out_proj.reshape(H, W, 3).astype(np.float32)

    # Optional log and visualization
    if keep_candidates:
        all_normals = all_normals_flat.reshape(H, W, k_eff, 3)
        all_dists   = all_dists_flat.reshape(H, W, k_eff)
        all_projs   = all_projs_flat.reshape(H, W, k_eff, 3)
        best_in_idxs  = best_in_flat.reshape(H, W)
        best_out_idxs = best_out_flat.reshape(H, W)

        # Print log if requested
        if print_log:
            print_report(
                query_pts,
                best_in_idxs, best_out_idxs,
                all_normals, all_dists, all_projs
            )

        # Visualize if requested
        if visualize:
            # Prepare flattened arrays (float64 for Open3D)
            N      = H * W
            q_flat = query_pts.reshape(N, 3).astype(np.float64)
            pk     = all_projs.reshape(N, k_eff, 3).astype(np.float64)

            # Purple: all candidate lines
            q_rep     = np.repeat(q_flat, k_eff, axis=0)
            p_rep     = pk.reshape(N * k_eff, 3)
            pts_all   = np.vstack([q_rep, p_rep])
            lines_all = [[i, i + N * k_eff] for i in range(N * k_eff)]
            ls_all = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(pts_all),
                lines=o3d.utility.Vector2iVector(lines_all)
            )
            ls_all.colors = o3d.utility.Vector3dVector([[0.5, 0, 0.5]] * len(lines_all))

            # Green/red: best inside/outside
            pts_in, lines_in = [], []
            pts_out, lines_out = [], []
            bi_flat = best_in_idxs.flatten()
            bo_flat = best_out_idxs.flatten()
            for idx in range(N):
                bi = bi_flat[idx]
                if bi >= 0:
                    s, e = q_flat[idx], pk[idx, bi]
                    b = len(pts_in)
                    pts_in += [s, e]
                    lines_in.append([b, b + 1])
                bo = bo_flat[idx]
                if bo >= 0:
                    s, e = q_flat[idx], pk[idx, bo]
                    b = len(pts_out)
                    pts_out += [s, e]
                    lines_out.append([b, b + 1])

            ls_in = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(np.asarray(pts_in, dtype=np.float64)),
                lines=o3d.utility.Vector2iVector(lines_in)
            )
            ls_in.colors = o3d.utility.Vector3dVector([[0, 1, 0]] * len(lines_in))

            ls_out = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(np.asarray(pts_out, dtype=np.float64)),
                lines=o3d.utility.Vector2iVector(lines_out)
            )
            ls_out.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(lines_out))

            # Points
            qp = o3d.geometry.PointCloud()
            qp.points = o3d.utility.Vector3dVector(q_flat)
            qp.paint_uniform_color([0, 1, 0])

            pp_in  = [pk[i, bi_flat[i]] for i in range(N) if bi_flat[i] >= 0]
            print(pp_in)
            pp_out = [pk[i, bo_flat[i]] for i in range(N) if bo_flat[i] >= 0]
            pc_in  = o3d.geometry.PointCloud()
            pc_in.points = o3d.utility.Vector3dVector(np.asarray(pp_in, dtype=np.float64))
            pc_in.paint_uniform_color([0, 0, 1])
            pc_out = o3d.geometry.PointCloud()
            pc_out.points = o3d.utility.Vector3dVector(np.asarray(pp_out, dtype=np.float64))
            pc_out.paint_uniform_color([1, 0, 0])

            # Grid, axes, downsampled normals (helpers must be defined)
            grid, axes   = add_grid_and_axes()
            normals_ls   = create_normal_lines(pcd)

            # o3d.visualization.draw_geometries([
            #     pcd, qp, pc_in, pc_out,
            #     ls_all, ls_in, ls_out,
            #     grid, axes, normals_ls
            # ])
            o3d.visualization.draw_geometries([
                pcd, qp, pc_in, pc_out,
                ls_all, ls_in, ls_out,
                
                
            ])

    # Full nested dict object array (note: building 200k dicts is inherently heavy)
    full = np.empty((H, W, 1), dtype=object)
    flat_idx = 0
    for i in range(H):
        for j in range(W):
            di = out_dist[flat_idx]
            ni = out_nvec[flat_idx]
            pi = out_proj[flat_idx]
            ii = int(out_idx[flat_idx])
            full[i, j, 0] = {
                "query_point":            q_all[flat_idx],
                "projected_index":        ii if ii >= 0 else None,
                "projected_point":        None if np.isnan(pi).any() else pi,
                "normal_distance_vector": None if np.isnan(ni).any() else ni,
                "distance":               None if np.isnan(di) else float(di),
            }
            flat_idx += 1
    return full

def euclidean_distance_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute per-pixel Euclidean distance between two H×W×3 arrays.

    Parameters
    ----------
    a, b : np.ndarray
        Arrays of shape (H, W, 3). Types can be integer or float.

    Returns
    -------
    np.ndarray
        Distance map of shape (H, W, 1), dtype float32.
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: a{a.shape} vs b{b.shape}")
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"Expected shape (H, W, 3), got {a.shape}")

    # Cast once to float for precise diff; keepdims=True → (H, W, 1)
    diff = a.astype(np.float32) - b.astype(np.float32)
    dist = np.linalg.norm(diff, axis=2, keepdims=True)  # (H, W, 1)
    return dist
