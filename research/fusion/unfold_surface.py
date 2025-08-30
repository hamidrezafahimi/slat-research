from unfold_helper import *

unfolded, ctx = unfold_surface("/media/hamid/Workspace/slat-research/data/research/depth_spline_viot.csv")
# e.g., to visualize:
pcd_unfold = o3d.geometry.PointCloud()
pcd_unfold.points = o3d.utility.Vector3dVector(unfolded.reshape(-1, 3))
o3d.visualization.draw_geometries([pcd_unfold])