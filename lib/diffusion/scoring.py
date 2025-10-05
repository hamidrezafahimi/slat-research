import numpy as np, open3d as o3d
from scipy.spatial import cKDTree, distance
_SCIPY = True
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/..")
from geom.surfaces import project_external_along_normals_andreject, \
    project_external_along_normals_noreject, nearest_vertex_normals
from .config import BGPatternDiffuserConfig
from .helper import score_based_downsample

# Utility: compute base smoothness from a neighbor set and a tangent plane (V, n_hat)
def _base_smoothness_on_plane(neigh_pts: np.ndarray, V: np.ndarray, n_hat: np.ndarray) -> float:
    m = neigh_pts.shape[0]
    if m < 2:
        return 0.0
    # project onto plane: Q_proj = Q - ((Q - V)·n) n
    diff = neigh_pts - V[None, :]
    t = np.einsum("ij,j->i", diff, n_hat)  # (m,)
    proj = neigh_pts - t[:, None] * n_hat[None, :]  # (m,3)

    # pairwise distances
    eps = 1e-9
    if _SCIPY:
        from scipy.spatial import distance
        D3 = distance.pdist(neigh_pts, metric="euclidean")
        Dp = distance.pdist(proj, metric="euclidean")
    else:
        D3_list, Dp_list = [], []
        for a in range(m - 1):
            dif3 = neigh_pts[a+1:] - neigh_pts[a]
            d3 = np.sqrt(np.sum(dif3 * dif3, axis=1))
            difp = proj[a+1:] - proj[a]
            dp = np.sqrt(np.sum(difp * difp, axis=1))
            D3_list.append(d3); Dp_list.append(dp)
        D3 = np.concatenate(D3_list) if D3_list else np.array([], dtype=float)
        Dp = np.concatenate(Dp_list) if Dp_list else np.array([], dtype=float)
    if D3.size == 0:
        return 0.0
    rel = np.abs(Dp - D3) / (D3 + eps)
    s_base = 1.0 - float(np.mean(rel))
    return float(np.clip(s_base, 0.0, 1.0))

def _eps():
    return 1e-12

def normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mn = float(np.min(x))
    mx = float(np.max(x))
    rng = mx - mn
    if rng <= _eps():
        return np.zeros_like(x)
    return (x - mn) / rng

def calc_pcd_bbox(points: np.ndarray):
    """Axis-aligned bounding box (min, max)."""
    pts = np.asarray(points, dtype=float)
    return pts.min(axis=0), pts.max(axis=0)

def average_density(points: np.ndarray) -> float:
    """Points per unit volume using AABB volume (robust to degenerate dims)."""
    mn, mx = calc_pcd_bbox(points)
    extents = np.maximum(mx - mn, _eps())
    vol = float(np.prod(extents))
    n = points.shape[0]
    return n / vol

def adapt_radius_for_avg_neighbors(points: np.ndarray, K: int) -> float:
    """
    Pick radius R so a random ball contains ~K points on average, based on global density.
    V_ball = 4/3 π R^3;  K ≈ density * V_ball  ⇒  R = ((3*K)/(4π*density))^(1/3)
    """
    d = average_density(points)
    R = ((3.0 * max(K, 1)) / (4.0 * np.pi * max(d, _eps()))) ** (1.0 / 3.0)
    return float(R)

def calculate_ndf_distances(ext_pts: np.ndarray, proj_pts: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """
    Raw NDF line lengths with sign (negative for below, positive for above).
    """
    # Calculate distances along the normal direction
    diff = proj_pts - ext_pts
    distances = np.linalg.norm(diff, axis=1)

    # Check the sign of the distance based on the normal direction
    # If the dot product of the difference vector and the normal is negative, it's below the surface
    sign = np.sign(np.sum(diff * normals, axis=1))
    
    return sign * distances

def project_and_dist(ext_pts, mesh):
    proj_pts = project_external_along_normals_noreject(ext_pts, mesh)
    normals = nearest_vertex_normals(ext_pts, mesh)
    ndf_dists = calculate_ndf_distances(ext_pts, proj_pts, normals)
    return proj_pts, ndf_dists

def ndf_to_score(ndf_dists: np.ndarray) -> np.ndarray:
    """
    Map raw NDF distances to [0,1] where:
      1 = most negative NDF distance (below the mesh),
      0 = most positive NDF distance (above the mesh).
    """
    # Invert the NDF distances: more negative -> higher score
    norm = normalize01(-ndf_dists)  # Negate the distances to invert the range
    return norm  # No need to invert again as we want negative to map to 1 and positive to 0

def calc_pcd_bbox(pcd_points: np.ndarray) -> np.ndarray:
    """Calculate the bounding box of the point cloud."""
    min_corner = np.min(pcd_points, axis=0)
    max_corner = np.max(pcd_points, axis=0)
    return min_corner, max_corner

def calculate_average_density(pcd_points: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray) -> float:
    """Calculate the average point density within the bounding box."""
    bbox_volume = np.prod(bbox_max - bbox_min)
    num_points = pcd_points.shape[0]
    density = num_points / bbox_volume
    return density

def adapt_radius(pcd_points: np.ndarray, k: int, desired_neighbors: int = 10) -> float:
    """
    Adapt the radius R for each point to have approximately K neighbors on average based on point density.
    """
    # Step 1: Calculate the bounding box of the point cloud
    bbox_min, bbox_max = calc_pcd_bbox(pcd_points)
    
    # Step 2: Calculate the average point density
    density = calculate_average_density(pcd_points, bbox_min, bbox_max)

    # Step 3: Estimate the radius R to have K neighbors
    # Volume of a ball in 3D: V = (4/3) * pi * R^3
    # We want the ball's volume to contain approximately K points based on the density
    # The volume of the ball that contains K points is V = K / density
    # Solving for R: R = ((3 * K) / (4 * pi * density)) ** (1/3)
    estimated_radius = ((3 * desired_neighbors) / (4 * np.pi * density)) ** (1/3)
    return estimated_radius

def calculate_smoothness_score(ext_pts: np.ndarray, ndfDists: np.ndarray, k: int) -> np.ndarray:
    """Calculate smoothness scores with adaptive radius R."""
    # Step 1: Adapt radius R for each point based on point cloud density
    R = adapt_radius(ext_pts, k)
    
    # Step 2: Proceed with the smoothness score calculation using the adapted radius
    smoothness_scores = np.zeros(ext_pts.shape[0], dtype=float)
    kdtree = cKDTree(ext_pts)

    for i, point in enumerate(ext_pts):
        # Query neighbors within the adaptive radius R using KDTree
        neighbors_idx = kdtree.query_ball_point(point, R)

        # Ensure there are at least k neighbors
        if len(neighbors_idx) < k:
            smoothness_scores[i] = 0
        else:
            # Calculate the absolute differences of NDF distances between the point and its neighbors
            ndf_diff = np.abs(ndfDists[neighbors_idx] - ndfDists[i])
            
            # Calculate the mean of absolute differences
            mean_ndf_diff = np.mean(ndf_diff)
            
            # Smoothness is inversely proportional to the mean NDF difference
            smoothness_scores[i] = 1 - np.clip(mean_ndf_diff, 0, 1)

    return smoothness_scores



# =========================
# Main pipeline (as a function)
# =========================

def associate_compound_score(
    ext_pts: np.ndarray,
    mesh,
    K_for_radius: int = 10 # target average neighbors used to size R
):
    """
    End-to-end pipeline starting from a PCD:
      1) Build B-spline mesh from control points.
      2) Project external PCD points to the surface (normal rays).
      3) Compute raw NDF distances -> ndf_score in [0,1] with 1=best (blue), 0=worst (red).
      4) Compute smoothness in [0,1] with adaptive radius.
      5) Compound = smoothness * ndf_score.
    Returns:
      compound_scores, smoothness_scores, ndf_scores
    """
    mesh.compute_vertex_normals()
    proj_pts, ndf_dists = project_and_dist(ext_pts, mesh)
    ndf_scores = ndf_to_score(ndf_dists)

    # Smoothness (adaptive R from density)
    smoothness_scores = calculate_smoothness_score(
        ext_pts=ext_pts,
        ndfDists=ndf_dists,
        k=K_for_radius
    )

    # Compound
    # compound_scores = smoothness_scores * ndf_scores
    compound_scores = ndf_scores
    # compound_scores = smoothness_scores

    return compound_scores, smoothness_scores, ndf_scores, mesh, proj_pts
    # return compound_scores, None, ndf_scores, mesh, proj_pts
    # return compound_scores, smoothness_scores, None, mesh, proj_pts


class Projection3DScorer:
    def __init__(self, config: BGPatternDiffuserConfig):
        self.config = config
        self.maxDZ = None
        self.fineTune = False
    
    def reset(self, _cloud_pts: np.ndarray,
              smoothness_base_mesh: o3d.geometry.TriangleMesh, max_dz: float,
              original_colors: np.ndarray | None = None, fine_tune: bool = False):
        self.maxDZ = max_dz
        self.tau = max(1e-9, self.maxDZ / 3.0)
        self.cloud_pts = np.asarray(_cloud_pts, dtype=float)
        self.smoothness_base_mesh = smoothness_base_mesh
        print("[Projection3DScorer] Calculating smoothness score ...")
        self.fineTune = fine_tune
        if self.fineTune:
            _scores, _, _, _, _ = associate_compound_score(
                    ext_pts=self.cloud_pts,
                    mesh=smoothness_base_mesh,
                    K_for_radius=self.config.scoring_smoothness_k)
        else: 
            _scores = self.self_associate_smoothness()

        downsampled_pts, self.smoothness_scores, _ = score_based_downsample(self.cloud_pts, 
                                                                            _scores,
                                                                            target_fraction=self.config.scorebased_downsample_target)
        il = self.cloud_pts.shape[0]                                                                                
        self.cloud_pts = np.asarray(downsampled_pts, dtype=float)
        print(f"[Projection3DScorer] Point clound downsampled from {il} to {downsampled_pts.shape[0]}")
        self._point_scores = None
        self.original_colors = None
        if original_colors is not None and len(original_colors) == len(self.cloud_pts):
            self.original_colors = np.asarray(original_colors, dtype=float)

    # ----- reset spline mesh -----
    def reset_smoothness_base_mesh(self, new_mesh: o3d.geometry.TriangleMesh):
        """
        Replace the class's spline mesh with a new one, and recompute
        all smoothness scores since smoothness relies on the mesh normals.

        Args:
            new_mesh (o3d.geometry.TriangleMesh): the new spline surface
        """
        if not isinstance(new_mesh, o3d.geometry.TriangleMesh):
            raise TypeError("reset_smoothness_base_mesh expects an open3d TriangleMesh")

        self.smoothness_base_mesh = new_mesh
        if not self.smoothness_base_mesh.has_vertex_normals():
            self.smoothness_base_mesh.compute_vertex_normals()

        # Clear caches
        self._point_scores = None

        # Recompute smoothness scores
        # self.smoothness_scores = self.self_associate_smoothness()
        if self.fineTune:
            self.smoothness_scores, _, _, _, _ = associate_compound_score(
                    ext_pts=self.cloud_pts,
                    mesh=self.smoothness_base_mesh,
                    K_for_radius=self.config.scoring_smoothness_k
                )
        else: 
            self.smoothness_scores = self.self_associate_smoothness()

    # ----- encapsulated smoothness (project neighbors along spline normals) -----
    def self_associate_smoothness(self) -> np.ndarray:
        """
        For each point P in the cloud:
        1) Take k_min nearest neighbors (including P).
        2) Find nearest vertex V on the current spline mesh to P and get its normal n.
        3) Project all those neighbors onto the tangent plane at V by moving them along n:
            Q_proj = Q - dot(Q - V, n_hat) * n_hat
        4) Compute base smoothness by comparing pairwise distances in 3D vs. on that plane:
            s_base = 1 - mean(|d_plane - d_3D| / (d_3D + eps))
        5) Apply penalties (same as before):
            - penalty_neighbors (count-based)
            - penalty_radius with R = distance to k-th NN, normalized by R95

        Returns per-point smoothness scores in [0, 1].
        """
        pts = self.cloud_pts
        n = pts.shape[0]
        scores = np.zeros(n, dtype=float)
        rng = np.random.default_rng(2)

        # --- prepare mesh vertex data & KD-tree for nearest-vertex queries ---
        mesh = self.smoothness_base_mesh
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()

        m_verts = np.asarray(mesh.vertices, dtype=float)
        m_norms = np.asarray(mesh.vertex_normals, dtype=float)

        # Guard: if mesh has no vertices, return zeros
        if m_verts.size == 0:
            return scores

        # KD-tree over mesh vertices (SciPy fast path; Open3D fallback)
        use_scipy_mesh = False
        try:
            if _SCIPY:
                from scipy.spatial import cKDTree as _Tree
                mesh_tree = _Tree(m_verts)
                use_scipy_mesh = True
            else:
                raise RuntimeError
        except Exception:
            mesh_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(m_verts))
            mesh_kdt = o3d.geometry.KDTreeFlann(mesh_pc)

        # --- robust neighborhood scale for radius penalty (same as before) ---
        R95 = self._estimate_R95(pts, k=self.config.scoring_smoothness_kmin_neighbors + 1)

        # --- neighbor search over cloud points (SciPy fast path; Open3D fallback) ---
        if _SCIPY:
            from scipy.spatial import cKDTree
            kdt = cKDTree(pts)

            for i in range(n):
                # k-NN (including self)
                dists, idxs = kdt.query(pts[i], k=self.config.scoring_smoothness_kmin_neighbors + 1)
                if np.isscalar(dists):  # edge case k=1
                    dists = np.array([0.0, float(dists)])
                    idxs = np.array([i, int(idxs)])
                R = float(dists[-1])

                neigh_idx = np.asarray(idxs, dtype=int)
                if len(neigh_idx) > self.config.scoring_smoothness_neighbors_cap:
                    neigh_idx = rng.choice(neigh_idx, size=self.config.scoring_smoothness_neighbors_cap, replace=False)
                neigh_pts = pts[neigh_idx]
                count = len(neigh_idx)

                # nearest mesh vertex to P -> (V, n_hat)
                if use_scipy_mesh:
                    _, j = mesh_tree.query(pts[i], k=1)
                    V = m_verts[j]
                    n_vec = m_norms[j]
                else:
                    _, jlist, _ = mesh_kdt.search_knn_vector_3d(pts[i], 1)
                    j = int(jlist[0])
                    V = m_verts[j]
                    n_vec = m_norms[j]

                # normalize normal; fallback to +Z if degenerate
                n_norm = np.linalg.norm(n_vec)
                n_hat = (n_vec / n_norm) if n_norm > 1e-12 else np.array([0.0, 0.0, 1.0], dtype=float)

                # base smoothness from projecting neighbors along the spline normal onto tangent plane
                s_base = _base_smoothness_on_plane(neigh_pts, V, n_hat)

                # penalties (unchanged)
                penalty_neighbors = np.clip((count - 1) / float(self.config.scoring_smoothness_kmin_neighbors), 0.0, 1.0)
                penalty_radius = 1.0 / (1.0 + (R / R95))

                scores[i] = float(np.clip(s_base * penalty_neighbors * penalty_radius, 0.0, 1.0))

        else:
            # Open3D fallback for both cloud NN and mesh-vertex NN
            cloud_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
            cloud_kdt = o3d.geometry.KDTreeFlann(cloud_pc)
            mesh_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(m_verts))
            mesh_kdt = o3d.geometry.KDTreeFlann(mesh_pc)

            for i in range(n):
                cnt, idxs, d2 = cloud_kdt.search_knn_vector_3d(pts[i], self.config.scoring_smoothness_kmin_neighbors + 1)
                if cnt == 0:
                    scores[i] = 0.0
                    continue
                R = float(np.sqrt(d2[min(cnt, self.config.scoring_smoothness_kmin_neighbors) - 1])) if cnt > 0 else 0.0

                neigh_idx = np.asarray(idxs[:cnt], dtype=int)
                if len(neigh_idx) > self.config.scoring_smoothness_neighbors_cap:
                    neigh_idx = rng.choice(neigh_idx, size=self.config.scoring_smoothness_neighbors_cap, replace=False)
                neigh_pts = pts[neigh_idx]
                count = len(neigh_idx)

                # nearest mesh vertex to P
                _, jlist, _ = mesh_kdt.search_knn_vector_3d(pts[i], 1)
                j = int(jlist[0])
                V = m_verts[j]
                n_vec = m_norms[j]
                n_norm = np.linalg.norm(n_vec)
                n_hat = (n_vec / n_norm) if n_norm > 1e-12 else np.array([0.0, 0.0, 1.0], dtype=float)

                s_base = _base_smoothness_on_plane(neigh_pts, V, n_hat)

                penalty_neighbors = np.clip((count - 1) / float(self.config.scoring_smoothness_kmin_neighbors), 0.0, 1.0)
                penalty_radius = 1.0 / (1.0 + (R / R95))

                scores[i] = float(np.clip(s_base * penalty_neighbors * penalty_radius, 0.0, 1.0))
        return scores

    # ----- main callables -----
    def score(self, mesh, rejection=False) -> float:
        # if rejection:
        #     # _projected_pts, idx = project_external_along_normals_andreject(self.cloud_pts, mesh)

        #     total = self.scoreSurface(_projected_pts, idx)
        # else:
        # _projected_pts = project_external_along_normals_noreject(self.cloud_pts, mesh)
        _projected_pts, _ = project_and_dist(self.cloud_pts, mesh)
        total = self.scoreSurface(_projected_pts)
        return float(total), _projected_pts

    def scoreSurface(self, proj_pts, idx=None) -> float:
        """
        point_score emphasizes smoothness 2× over distance:
        gain(dz) = exp(-|dz| / tau) for |dz| <= max_delta_z, else 0
        point_score = (smoothness^2) * (gain^1) within cutoff; else 0
        Returns SUM of point_scores.
        """
        dz = np.linalg.norm(self.cloud_pts - proj_pts, axis=1)
        within = (dz <= self.maxDZ)

        gain = np.zeros_like(dz, dtype=float)
        gain[within] = np.exp(-dz[within] / self.tau)

        if idx is None:
            s = np.clip(self.smoothness_scores, 0.0, 1.0)
        else:
            s = np.clip(self.smoothness_scores, 0.0, 1.0)[idx]
        
        ps = np.zeros_like(s, dtype=float)
        # ▶ emphasize smoothness twice as much as distance (2:1 in log-domain)
        ps[within] = (s[within] ** 2) * (gain[within] ** 1)

        # Linear 2:1 blend (still zero outside cutoff)
        # w_s, w_g = 2.0, 1.0
        # ps[within] = ((w_s * s[within]) + (w_g * gain[within])) / (w_s + w_g)

        self._point_scores = ps
        return float(np.sum(ps))
    
    # ----- drawables (no materials; usable in any viewer) -----
    def draw(self, ctrl_pts: np.ndarray = None, pj_pts: np.ndarray = None,
             W: float = None, H: float = None,
             ext_mode: str = "smoothness", pp_mode: str = "score"):
        if pj_pts is None:
            pj_pts = project_external_along_normals_andreject(self.cloud_pts, self.smoothness_base_mesh)
        if self._point_scores is None:
            _ = self.scoreSurface()

        # external colors
        if ext_mode == "black":
            ext_colors = np.tile([[0, 0, 0]], (len(self.cloud_pts), 1))
        elif ext_mode == "original" and self.original_colors is not None:
            ext_colors = np.clip(self.original_colors, 0.0, 1.0)
        elif ext_mode == "smoothness":
            ext_colors = self._heatmap_smoothness(self.smoothness_scores)
        else:
            ext_colors = None

        # ppoints colors
        if pp_mode == "black":
            pp_colors = np.tile([[0, 0, 0]], (len(pj_pts), 1))
        elif pp_mode == "original" and self.original_colors is not None:
            pp_colors = np.clip(self.original_colors, 0.0, 1.0)
        elif pp_mode == "score":
            pp_colors = self._heatmap_score(np.clip(self._point_scores, 0.0, 1.0))
        else:
            pp_colors = None

        if ext_colors is not None:
            pcd_ext = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.cloud_pts))
            pcd_ext.colors = o3d.utility.Vector3dVector(ext_colors)
        else:
            pcd_ext = o3d.geometry.PointCloud(o3d.utility.Vector3dVector([]))
            pcd_ext.colors = o3d.utility.Vector3dVector([])

        if pp_colors is not None:
            pcd_pp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pj_pts))
            pcd_pp.colors = o3d.utility.Vector3dVector(pp_colors)
        else:
            pcd_pp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector([]))
            pcd_pp.colors = o3d.utility.Vector3dVector([])

        geoms = [pcd_ext, pcd_pp]
        return geoms

    # ----- helpers -----
    @staticmethod
    def _heatmap_smoothness(s: np.ndarray) -> np.ndarray:
        # red (1) ↔ blue (0)
        r = s; g = np.zeros_like(s); b = 1.0 - s
        return np.stack([r, g, b], axis=1)

    @staticmethod
    def _heatmap_score(q: np.ndarray) -> np.ndarray:
        # green (1) ↔ purple (0)
        good = np.array([0.0, 1.0, 0.0])
        bad  = np.array([0.5, 0.0, 0.5])
        return (bad[None, :] * (1.0 - q[:, None]) + good[None, :] * q[:, None])

    @staticmethod
    def _ctrl_points_as_spheres(ctrl_pts: np.ndarray, W: float, H: float):
        if ctrl_pts.size == 0:
            return []
        radius = max(1e-6, 0.01 * min(W, H))
        geoms = []
        for p in ctrl_pts:
            s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            s.translate(p.tolist())
            s.compute_vertex_normals()
            s.paint_uniform_color([0.9, 0.1, 0.1])  # red spheres
            geoms.append(s)
        return geoms

    @staticmethod
    def _estimate_R95(pts: np.ndarray, k: int = 9, sample_size: int = 5000) -> float:
        n = pts.shape[0]
        idx = np.random.default_rng(1).choice(n, size=min(sample_size, n), replace=False)
        if _SCIPY:
            kdt = cKDTree(pts)
            dists, _ = kdt.query(pts[idx], k=k)
            r9 = dists[:, -1]
        else:
            tree = o3d.geometry.KDTreeFlann(o3d.utility.Vector3dVector(pts))
            r9 = []
            for i in idx:
                _, _, d2 = tree.search_knn_vector_3d(pts[i], k)
                r9.append(np.sqrt(d2[-1]) if len(d2) else 0.0)
            r9 = np.array(r9)
        R95 = float(np.percentile(r9, 95)) if r9.size > 0 else 1.0
        return max(R95, 1e-9)

    @staticmethod
    def _smoothness_from_neighbors(neigh_pts: np.ndarray) -> float:
        m = neigh_pts.shape[0]
        if m < 2:
            return 0.0
        if _SCIPY:
            D3 = distance.pdist(neigh_pts, metric="euclidean")
            Dxy = distance.pdist(neigh_pts[:, :2], metric="euclidean")
        else:
            D3, Dxy = [], []
            for a in range(m - 1):
                dif = neigh_pts[a + 1:] - neigh_pts[a]
                D3.append(np.linalg.norm(dif, axis=1))
                Dxy.append(np.linalg.norm(dif[:, :2], axis=1))
            D3 = np.concatenate(D3) if D3 else np.array([], dtype=float)
            Dxy = np.concatenate(Dxy) if Dxy else np.array([], dtype=float)
        if D3.size == 0:
            return 0.0
        eps = 1e-9
        rel = np.abs(Dxy - D3) / (D3 + eps)
        s_base = 1.0 - float(np.mean(rel))
        return float(np.clip(s_base, 0.0, 1.0))
