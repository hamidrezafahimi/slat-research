import numpy as np
import open3d as o3d

def _eps():
    return 1e-12

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

def estimate_plane_normal_cg_global(pcd: o3d.geometry.PointCloud) -> (np.ndarray, np.ndarray):
    """
    Estimate a plane that is constrained to pass through the centroid (CG)
    using ALL points. The plane normal is the smallest-eigenvector of the
    covariance of (points - CG).
    Returns (normal, centroid).
    """
    pts = np.asarray(pcd.points)
    if pts.shape[0] < 3:
        raise RuntimeError("Not enough points to estimate a plane.")
    cg = pcd.get_center()
    X = pts - cg
    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, np.argmin(eigvals)]
    return normal, cg

def calc_pcd_bbox(pcd_points: np.ndarray) -> np.ndarray:
    """Calculate the bounding box of the point cloud."""
    min_corner = np.min(pcd_points, axis=0)
    max_corner = np.max(pcd_points, axis=0)
    return min_corner, max_corner
