import numpy as np
from scipy.spatial import cKDTree
# from typing import Optional, Tuple, List, Dict, Union
# from dataclasses import dataclass
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os, numpy as np, open3d as o3d
# ---------- optional (faster KDTree + pdist) ----------
import time
import secrets
import numpy as np
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


try:
    from scipy.spatial import cKDTree, distance
    _SCIPY = True
except Exception:
    _SCIPY = False

# =================== 3) scorer class ===================
class Projection3DScorer:
    def __init__(self,
                 cloud_pts: np.ndarray,
                 spline_mesh: o3d.geometry.TriangleMesh,
                 kmin_neighbors: int = 8,
                 neighbor_cap: int = 64,
                 max_delta_z: float = 0.5,
                 tau: float = None,
                 original_colors: np.ndarray | None = None):
        self.cloud_pts = np.asarray(cloud_pts, dtype=float)
        self.spline_mesh = spline_mesh
        self.kmin_neighbors = int(kmin_neighbors)
        self.neighbor_cap = int(neighbor_cap)
        self.max_delta_z = float(max_delta_z)
        self.tau = float(tau) if tau is not None else max(1e-9, self.max_delta_z / 3.0)

        self.original_colors = None
        if original_colors is not None and len(original_colors) == len(self.cloud_pts):
            self.original_colors = np.asarray(original_colors, dtype=float)

        # per-point smoothness
        self.smoothness_scores = self.self_associate_smoothness()

        # lazy caches
        self._point_scores = None

    # # ----- encapsulated smoothness -----
    # def self_associate_smoothness(self) -> np.ndarray:
    #     pts = self.cloud_pts
    #     n = pts.shape[0]
    #     scores = np.zeros(n, dtype=float)
    #     rng = np.random.default_rng(2)

    #     R95 = self._estimate_R95(pts, k=self.kmin_neighbors + 1)

    #     if _SCIPY:
    #         kdt = cKDTree(pts)
    #         for i in range(n):
    #             dists, _ = kdt.query(pts[i], k=self.kmin_neighbors + 1)
    #             R = float(dists[-1])
    #             neigh_idx = kdt.query_ball_point(pts[i], r=R)
    #             if len(neigh_idx) < self.kmin_neighbors:
    #                 R = min(R * 2.0, 1.5 * R95)
    #                 neigh_idx = kdt.query_ball_point(pts[i], r=R)
    #             if len(neigh_idx) > self.neighbor_cap:
    #                 neigh_idx = list(rng.choice(neigh_idx, size=self.neighbor_cap, replace=False))

    #             neigh_pts = pts[neigh_idx]
    #             count = len(neigh_idx)

    #             s_base = self._smoothness_from_neighbors(neigh_pts)
    #             penalty_neighbors = np.clip((count - 1) / float(self.kmin_neighbors), 0.0, 1.0)
    #             penalty_radius = 1.0 / (1.0 + (R / R95))
    #             scores[i] = float(np.clip(s_base * penalty_neighbors * penalty_radius, 0.0, 1.0))
    #     else:
    #         tree = o3d.geometry.KDTreeFlann(o3d.utility.Vector3dVector(pts))
    #         for i in range(n):
    #             _, _, d2_knn = tree.search_knn_vector_3d(pts[i], self.kmin_neighbors + 1)
    #             R = float(np.sqrt(d2_knn[-1])) if len(d2_knn) else 0.0
    #             _, idx_r, _ = tree.search_radius_vector_3d(pts[i], R)
    #             if len(idx_r) < self.kmin_neighbors:
    #                 R = min(R * 2.0, 1.5 * R95)
    #                 _, idx_r, _ = tree.search_radius_vector_3d(pts[i], R)
    #             if len(idx_r) > self.neighbor_cap:
    #                 idx_r = list(rng.choice(idx_r, size=self.neighbor_cap, replace=False))

    #             neigh_pts = pts[idx_r]
    #             count = len(idx_r)

    #             s_base = self._smoothness_from_neighbors(neigh_pts)
    #             penalty_neighbors = np.clip((count - 1) / float(self.kmin_neighbors), 0.0, 1.0)
    #             penalty_radius = 1.0 / (1.0 + (R / R95))
    #             scores[i] = float(np.clip(s_base * penalty_neighbors * penalty_radius, 0.0, 1.0))

    #     return scores

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
        mesh = self.spline_mesh
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
        R95 = self._estimate_R95(pts, k=self.kmin_neighbors + 1)

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

        # --- neighbor search over cloud points (SciPy fast path; Open3D fallback) ---
        if _SCIPY:
            from scipy.spatial import cKDTree
            kdt = cKDTree(pts)

            for i in range(n):
                # k-NN (including self)
                dists, idxs = kdt.query(pts[i], k=self.kmin_neighbors + 1)
                if np.isscalar(dists):  # edge case k=1
                    dists = np.array([0.0, float(dists)])
                    idxs = np.array([i, int(idxs)])
                R = float(dists[-1])

                neigh_idx = np.asarray(idxs, dtype=int)
                if len(neigh_idx) > self.neighbor_cap:
                    neigh_idx = rng.choice(neigh_idx, size=self.neighbor_cap, replace=False)
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
                penalty_neighbors = np.clip((count - 1) / float(self.kmin_neighbors), 0.0, 1.0)
                penalty_radius = 1.0 / (1.0 + (R / R95))

                scores[i] = float(np.clip(s_base * penalty_neighbors * penalty_radius, 0.0, 1.0))

        else:
            # Open3D fallback for both cloud NN and mesh-vertex NN
            cloud_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
            cloud_kdt = o3d.geometry.KDTreeFlann(cloud_pc)
            mesh_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(m_verts))
            mesh_kdt = o3d.geometry.KDTreeFlann(mesh_pc)

            for i in range(n):
                cnt, idxs, d2 = cloud_kdt.search_knn_vector_3d(pts[i], self.kmin_neighbors + 1)
                if cnt == 0:
                    scores[i] = 0.0
                    continue
                R = float(np.sqrt(d2[min(cnt, self.kmin_neighbors) - 1])) if cnt > 0 else 0.0

                neigh_idx = np.asarray(idxs[:cnt], dtype=int)
                if len(neigh_idx) > self.neighbor_cap:
                    neigh_idx = rng.choice(neigh_idx, size=self.neighbor_cap, replace=False)
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

                penalty_neighbors = np.clip((count - 1) / float(self.kmin_neighbors), 0.0, 1.0)
                penalty_radius = 1.0 / (1.0 + (R / R95))

                scores[i] = float(np.clip(s_base * penalty_neighbors * penalty_radius, 0.0, 1.0))

        return scores

    # ----- main callables -----
    def score(self, spline_mesh) -> float:
        _projected_pts = project_external_to_surface_idw(self.cloud_pts, spline_mesh)
        total = self.scoreSurface(_projected_pts)
        return float(total), _projected_pts

    # ----- reset spline mesh -----
    def reset_spline_mesh(self, new_mesh: o3d.geometry.TriangleMesh):
        """
        Replace the class's spline mesh with a new one, and recompute
        all smoothness scores since smoothness relies on the mesh normals.

        Args:
            new_mesh (o3d.geometry.TriangleMesh): the new spline surface
        """
        if not isinstance(new_mesh, o3d.geometry.TriangleMesh):
            raise TypeError("reset_spline_mesh expects an open3d TriangleMesh")

        self.spline_mesh = new_mesh
        if not self.spline_mesh.has_vertex_normals():
            self.spline_mesh.compute_vertex_normals()

        # Clear caches
        self._point_scores = None

        # Recompute smoothness scores
        self.smoothness_scores = self.self_associate_smoothness()

    # ----- new surface scoring -----
    def scoreSurface(self, pj_pts) -> float:
        """
        point_score emphasizes smoothness 2× over distance:
        gain(dz) = exp(-|dz| / tau) for |dz| <= max_delta_z, else 0
        point_score = (smoothness^2) * (gain^1) within cutoff; else 0
        Returns SUM of point_scores.
        """
        P  = self.cloud_pts
        PJ = pj_pts
        dz = np.abs(PJ[:, 2] - P[:, 2])
        within = (dz <= self.max_delta_z)

        gain = np.zeros_like(dz, dtype=float)
        gain[within] = np.exp(-dz[within] / self.tau)

        s = np.clip(self.smoothness_scores, 0.0, 1.0)
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
            pj_pts = project_external_to_surface_idw(self.cloud_pts, self.spline_mesh)
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

class RandomSurfacer:
    """
    - Builds base flat surface at z0 and its parallel top/bottom (z0±OF).
    - Precomputes Z_up/Z_down for each control point by vertical projection.
    - generate(N) -> list[np.ndarray]: N random control grids with fixed (x,y) and z ~ U[Z_down, Z_up].
    - draw(random_ctrl_pts): renders ext cloud, base/top/bottom AND a *smooth random B-spline surface* from random_ctrl_pts.
    """

    def __init__(self, cloud, grid_w=6, grid_h=4, samples_u=40, samples_v=40, margin=0.02, seed=None):
        # ---- robust, instance-scoped RNG (ignores global np.random.seed) ----
        if seed is None:
            # Mix multiple entropy sources so two identical constructions in the same process still differ.
            entropy = [
                secrets.randbits(32),
                int(time.time_ns() & 0xFFFFFFFF),
                grid_w, grid_h, samples_u, samples_v
            ]
            ss = np.random.SeedSequence(entropy)
            self.rng = np.random.default_rng(ss)
        else:
            # Reproducible stream if you want it:
            self.rng = np.random.default_rng(seed)

        self.cloud_pts = _load_cloud_points(cloud)
        self.grid_w, self.grid_h = int(grid_w), int(grid_h)
        self.samples_u, self.samples_v = int(samples_u), int(samples_v)
        self.margin = float(margin)

        (self.base_mesh, self.ctrl_pts, self.W, self.H,
         self.center_xy, self.z0) = generate_xy_spline(
            self.cloud_pts, self.grid_w, self.grid_h, self.samples_u, self.samples_v, self.margin
        )

        z_mean = float(self.cloud_pts[:, 2].mean())
        z_min  = float(self.cloud_pts[:, 2].min())
        self.OF = z_mean - z_min

        # (Open3D note: .clone() is safest on newer versions)
        try:
            self.top_mesh = self.base_mesh.clone()
            self.bottom_mesh = self.base_mesh.clone()
        except AttributeError:
            self.top_mesh = o3d.geometry.TriangleMesh(self.base_mesh)
            self.bottom_mesh = o3d.geometry.TriangleMesh(self.base_mesh)

        self.top_mesh.translate((0, 0, +self.OF))
        self.bottom_mesh.translate((0, 0, -self.OF))

        self.Z_up_scalar   = self.z0 + self.OF
        self.Z_down_scalar = self.z0 - self.OF
        self.Z_up   = np.full((self.grid_h, self.grid_w), self.Z_up_scalar,   dtype=float)
        self.Z_down = np.full((self.grid_h, self.grid_w), self.Z_down_scalar, dtype=float)

        self.ctrl_xy = self.ctrl_pts[:, :2].copy()

    def generate(self, N: int) -> list[np.ndarray]:
        N = int(N)
        if N <= 0:
            return []
        z_lo = self.Z_down.ravel()
        z_hi = self.Z_up.ravel()
        outs = []
        for _ in range(N):
            # Use the instance RNG so draws differ across instances/runs even with same params
            z = self.rng.uniform(low=z_lo, high=z_hi, size=z_lo.shape).astype(float)
            outs.append(np.column_stack([self.ctrl_xy, z]))
        return outs

    def _mesh_from_ctrl(self, ctrl_pts_flat: np.ndarray) -> o3d.geometry.TriangleMesh:
        return bspline_surface_mesh_from_ctrl(
            ctrl_pts_flat, self.grid_w, self.grid_h, self.samples_u, self.samples_v
        )

    def draw(self, random_ctrl_pts: np.ndarray | None = None):
        ctrl_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.ctrl_pts))
        ctrl_pcd.paint_uniform_color([1.0, 0.6, 0.2])

        ext_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.cloud_pts))
        ext_pcd.paint_uniform_color([0.2, 0.6, 1.0])

        rnd_mesh = None
        if random_ctrl_pts is not None:
            rnd_mesh = self._mesh_from_ctrl(random_ctrl_pts)

        gui.Application.instance.initialize()
        window = gui.Application.instance.create_window("RandomSurfacer Viewer", 1280, 900)
        scene = gui.SceneWidget()
        scene.scene = rendering.Open3DScene(window.renderer)
        window.add_child(scene)

        scene.scene.add_geometry("ext_pcd", ext_pcd, mat_points(2.0))
        scene.scene.add_geometry("ctrl_pcd", ctrl_pcd, mat_points(8.0))
        scene.scene.add_geometry("base_mesh", self.base_mesh, mat_mesh())
        scene.scene.add_geometry("top_mesh", self.top_mesh, _mat_mesh_tinted((0.7, 0.95, 0.7, 0.55)))
        scene.scene.add_geometry("bottom_mesh", self.bottom_mesh, _mat_mesh_tinted((0.95, 0.7, 0.7, 0.55)))
        if rnd_mesh is not None:
            rnd_mesh.compute_vertex_normals()
            scene.scene.add_geometry("random_spline", rnd_mesh, _mat_mesh_tinted((0.8, 0.8, 0.2, 0.9)))

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        scene.scene.add_geometry("axis", axis, mat_mesh())

        # Fit camera to everything
        geoms = [ext_pcd, ctrl_pcd, self.base_mesh, self.top_mesh, self.bottom_mesh]
        if rnd_mesh is not None:
            geoms.append(rnd_mesh)
        aabbs = [g.get_axis_aligned_bounding_box() for g in geoms]
        mins = np.min([a.min_bound for a in aabbs], axis=0)
        maxs = np.max([a.max_bound for a in aabbs], axis=0)
        center = 0.5 * (mins + maxs)
        extent = float(np.max(maxs - mins))
        eye = center + np.array([0.0, -3.0 * extent, 1.8 * extent])
        up = np.array([0.0, 0.0, 1.0])
        scene.scene.camera.look_at(center, eye, up)

        gui.Application.instance.run()

# =================== 2) IDW projection ===================
def project_external_to_surface_idw(ext_pts: np.ndarray,
                                    surf_mesh: o3d.geometry.TriangleMesh,
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

# ---------------- I/O ----------------
def _load_cloud_points(path_or_arr):
    if isinstance(path_or_arr, np.ndarray):
        pts = np.asarray(path_or_arr, dtype=float)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError("cloud_pts ndarray must be (N,3)+")
        return pts[:, :3].copy()
    path = str(path_or_arr)
    ext = os.path.splitext(path)[1].lower()
    if ext in [".npy", ".npz"]:
        arr = np.load(path)
        pts = arr["points"] if isinstance(arr, np.lib.npyio.NpzFile) and "points" in arr else np.array(arr)
        pts = np.asarray(pts, dtype=float)
    else:
        pcd = o3d.io.read_point_cloud(path)
        if pcd.is_empty():
            raise ValueError(f"Failed to read or empty point cloud: {path}")
        pts = np.asarray(pcd.points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("Point cloud must be (N,3)+")
    return pts[:, :3].copy()

def generate_xy_spline(cloud_pts: np.ndarray,
                       grid_w: int = 6, grid_h: int = 4,
                       samples_u: int = 40, samples_v: int = 40,
                       margin: float = 0.02):
    W, H, center_xy = _extent_wh_center_xy(cloud_pts)
    W *= (1.0 + margin); H *= (1.0 + margin)
    z0 = float(cloud_pts[:, 2].mean())
    ctrl_pts = _ctrl_points_grid(grid_w, grid_h, W, H, center_xy, z0)
    mesh = _flat_patch_mesh(center_xy, W, H, z0, samples_u, samples_v)
    return mesh, ctrl_pts, W, H, center_xy, z0

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

def bspline_surface_mesh_from_ctrl(ctrl_pts_flat: np.ndarray, grid_w: int, grid_h: int,
                                    su: int, sv: int) -> o3d.geometry.TriangleMesh:
    """Build a smooth bicubic tensor-product B-spline surface from a (grid_h*grid_w,3) control grid."""
    P = ctrl_pts_flat.reshape(grid_h, grid_w, 3).astype(float)  # [v=j(row), u=i(col)]
    n_u, n_v, p = grid_w, grid_h, 3
    if n_u < 4 or n_v < 4:
        # Fallback: bilinear upsample using control grid directly
        X = P[..., 0]; Y = P[..., 1]; Z = P[..., 2]
        uu = np.linspace(0, n_u - 1, su); vv = np.linspace(0, n_v - 1, sv)
        Uidx = np.clip(np.searchsorted(np.arange(n_u), uu) - 1, 0, n_u - 2)
        Vidx = np.clip(np.searchsorted(np.arange(n_v), vv) - 1, 0, n_v - 2)
        XX = np.zeros((sv, su)); YY = np.zeros((sv, su)); ZZ = np.zeros((sv, su))
        for a, j in enumerate(Vidx):
            v0 = j; v1 = j + 1; tv = (vv[a] - v0)
            for b, i in enumerate(Uidx):
                u0 = i; u1 = i + 1; tu = (uu[b] - u0)
                # bilinear blend
                def bl(M):
                    return (1 - tu) * (1 - tv) * M[v0, u0] + tu * (1 - tv) * M[v0, u1] + (1 - tu) * tv * M[v1, u0] + tu * tv * M[v1, u1]
                XX[a, b] = bl(X); YY[a, b] = bl(Y); ZZ[a, b] = bl(Z)
        verts = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)
    else:
        U = _uniform_clamped_knots(n_u, p)
        V = _uniform_clamped_knots(n_v, p)
        us = np.linspace(0.0, 1.0, su)
        vs = np.linspace(0.0, 1.0, sv)
        # Precompute basis for all sample params (sparse across local spans)
        Bu = np.zeros((su, n_u))
        Bv = np.zeros((sv, n_v))
        for b, u in enumerate(us):
            span_u = _find_span(n_u, p, u, U)
            Nu = _basis_funs(span_u, u, p, U)
            Bu[b, span_u - p: span_u + 1] = Nu
        for a, v in enumerate(vs):
            span_v = _find_span(n_v, p, v, V)
            Nv = _basis_funs(span_v, v, p, V)
            Bv[a, span_v - p: span_v + 1] = Nv
        # Evaluate: S(v,u) = sum_j sum_i Bv[a,j] * Bu[b,i] * P[j,i]
        Xc, Yc, Zc = P[..., 0], P[..., 1], P[..., 2]
        XX = Bv @ Xc @ Bu.T
        YY = Bv @ Yc @ Bu.T
        ZZ = Bv @ Zc @ Bu.T
        verts = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)

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

# ------------- B-spline surface evaluation (bicubic, clamped, uniform) -------------
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

def mat_points(size=4.0):
    m = rendering.MaterialRecord(); m.shader = "defaultUnlit"; m.point_size = float(size); return m

def mat_mesh():
    m = rendering.MaterialRecord(); m.shader = "defaultLit"
    m.base_color = (0.7, 0.7, 0.9, 1.0); m.base_roughness = 0.8; return m

def _mat_mesh_tinted(rgba):
    m = rendering.MaterialRecord(); m.shader = "defaultLit"
    m.base_color = tuple(float(c) for c in rgba); m.base_roughness = 0.8; return m



def generate_spline(ctrl_points: np.ndarray,
                       samples_u: int = 40, samples_v: int = 40,
                       tol: float = 1e-9):
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

    # infer grid sizes (unique Xs and Ys with tolerance)
    def unique_sorted_with_tol(a, atol):
        a_sorted = np.sort(a)
        uniq = [a_sorted[0]]
        for v in a_sorted[1:]:
            if abs(v - uniq[-1]) > atol:
                uniq.append(v)
        return np.asarray(uniq, dtype=float)

    tol_x = max(tol, 1e-8 * max(1.0, W))
    tol_y = max(tol, 1e-8 * max(1.0, H))
    ux = unique_sorted_with_tol(xs, tol_x)  # gw
    uy = unique_sorted_with_tol(ys, tol_y)  # gh
    gw, gh = len(ux), len(uy)
    if gw * gh != len(P):
        raise ValueError(f"Control net is not a full grid: {len(P)} pts vs {gw}×{gh}")

    # reorder control points into row-major (v=Y first, then u=X)
    # i.e., sort by (y, x) ascending, then reshape (gh, gw, 3)
    order = np.lexsort((xs, ys))          # primary: ys, secondary: xs
    ctrl_sorted = P[order, :]
    ctrl_grid = ctrl_sorted.reshape(gh, gw, 3)

    # build mesh with your sampler
    mesh = build_surface_mesh(ctrl_grid.reshape(-1, 3), gw, gh, samples_u, samples_v)
    return mesh, W, H


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