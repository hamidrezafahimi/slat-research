import numpy as np, open3d as o3d


def pcd2pcdArr(pcd):
    return np.asarray(pcd.points, dtype=float)

def pcm2pcdArr(pcm):
    return pcm.reshape(-1, 3)

def pcm2pcd(pcm, visualization_image):
    points = pcm2pcdArr(pcm)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    h, w, _ = visualization_image.shape
    assert pcm.shape[:2] == (h, w), "Image and point cloud dimensions mismatch."
    colors = visualization_image.reshape(-1, 3).astype(np.float32) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, [2,1,0]])
    return pcd

