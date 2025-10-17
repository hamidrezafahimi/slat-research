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

def downsample_pcd(pcd: o3d.geometry.PointCloud, num_dst: int) -> o3d.geometry.PointCloud:
    """
    Automatically downsample the point cloud to a given ratio or half its size.
    
    Args:
    - pcd (o3d.geometry.PointCloud): Input point cloud to be downsampled.
    - ratio (float): Ratio to downsample the point cloud. 
                     Default is 0.5 (downsample to half).
    
    Returns:
    - pcd_downsampled (o3d.geometry.PointCloud): Downsampled point cloud.
    """
    # Convert point cloud to numpy array
    cloud_pts = np.asarray(pcd.points, dtype=float)
    num_orig = float(cloud_pts.shape[0])
    num_dst = float(num_dst)
    if num_dst < num_orig:
        ratio = num_dst / num_orig
    else:
        ratio = 1.0

    # Calculate the number of points after downsampling
    num_points = cloud_pts.shape[0]
    new_num_points = int(num_points * ratio)

    # Randomly select indices to downsample
    indices = np.random.choice(num_points, new_num_points, replace=False)

    # Downsample the points
    cloud_pts_downsampled = cloud_pts[indices]

    # If the point cloud has colors, downsample colors as well
    if pcd.has_colors():
        orig_cols = np.asarray(pcd.colors, dtype=float)
        orig_cols_downsampled = orig_cols[indices]
    
    # Create a new point cloud from the downsampled points
    pcd_downsampled = o3d.geometry.PointCloud()
    pcd_downsampled.points = o3d.utility.Vector3dVector(cloud_pts_downsampled)

    if pcd.has_colors():
        pcd_downsampled.colors = o3d.utility.Vector3dVector(orig_cols_downsampled)

    return pcd_downsampled

