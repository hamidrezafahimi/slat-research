import open3d as o3d
from .transformations import rodrigues_rotation_matrix
import numpy as np
from geom.clouds import estimate_plane_normal_cg_global

def orient_point_cloud_cgplane_global(pcd: o3d.geometry.PointCloud):
    """
    Orient the point cloud by:
      1) Estimating the plane through the centroid using ALL points (global PCA around CG).
      2) Rotating about the centroid so the plane becomes parallel to the XY plane (normal -> +Z).
    Returns (oriented_cloud, T) where T is the 4x4 transform Original->Oriented.
    """
    n, c = estimate_plane_normal_cg_global(pcd)
    if n[2] < 0:
        n = -n
    R = rodrigues_rotation_matrix(n, np.array([0.0, 0.0, 1.0]))

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = c - R @ c  # rotate about centroid
    oriented = apply_transform(pcd, T)
    return oriented, T

def apply_transform(pcd: o3d.geometry.PointCloud, T: np.ndarray) -> o3d.geometry.PointCloud:
    """Apply a 4x4 homogeneous transform to a point cloud and return a copy."""
    q = o3d.geometry.PointCloud(pcd)
    R = T[:3, :3]
    t = T[:3, 3]
    q.rotate(R, center=(0.0, 0.0, 0.0))
    q.translate(t, relative=True)
    return q

def apply_transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Apply a 4x4 homogeneous transform to an array of 3D points.

    Args
    ----
    points : (N, 3) float array
        Input points.
    T : (4, 4) float array
        Homogeneous transform matrix. Interpreted as x' = R x + t,
        where R = T[:3,:3], t = T[:3,3].

    Returns
    -------
    points_out : (N, 3) float array
        Transformed points.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"`points` must be (N,3), got {points.shape}")
    if T.shape != (4, 4):
        raise ValueError(f"`T` must be (4,4), got {T.shape}")

    R = T[:3, :3]
    t = T[:3, 3]
    # x' = R x + t  (apply row-wise: P @ R^T + t)
    return points @ R.T + t