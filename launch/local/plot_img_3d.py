import matplotlib.pyplot as plt
import cv2 
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import open3d as o3d

# bgi = cv2.imread('/home/hamid/w/DATA/27/pattern_y.png', cv2.IMREAD_GRAYSCALE)
bgi = cv2.imread('/media/hamid/Workspace/DATA/272/pattern.png', cv2.IMREAD_GRAYSCALE)[:,:800]
bg = bgi.astype(np.float32)
# dpth = cv2.imread('/home/hamid/w/DATA/27/00000027.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)
dpth = np.loadtxt('/home/hamid/w/DATA/272/asfd.csv', delimiter=',', dtype=np.float32)
cl = cv2.imread('/media/hamid/Workspace/DATA/27/fasdfdf.png').astype(np.float32)[:,:800]
dpth *= 255.0 / np.max(dpth)

cv2.imwrite('/home/hamid/fasd.png', dpth)

dpth = dpth[:,:800]

x_coords = np.linspace(0, bg.shape[1], bg.shape[1])
y_coords = np.linspace(0, bg.shape[0], bg.shape[0])
X, Y = np.meshgrid(x_coords, y_coords)
 
# ==========

# cv2.imshow('s', bgi)
# cv2.waitKey()

# ==========

# print(bg[0,   0     ], 
#       bg[0,   800  ], 
#       bg[375, 0     ], 
#       bg[375, 800  ])

# ==========

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, bg, color='blue')
# ax.plot_surface(X, Y, dpth, color='red')
# plt.show()

# ==========
points1 = np.hstack([X.reshape(-1,1), Y.reshape(-1,1), bg.reshape(-1,1)])
colors1 = cv2.cvtColor(bgi, cv2.COLOR_GRAY2BGR).astype(np.float32).reshape(-1,3) / 255.0
colors1[:,2] = 1

points2 = np.hstack([X.reshape(-1,1), Y.reshape(-1,1), dpth.reshape(-1,1)])
colors2 = cl.reshape(-1,3) / 255.0

# points3 = np.array([[0,   0,     -1396.0],
#                     [0,   800,  -1396.0],
#                     [375, 0,     203.0],
#                     [375, 800,  234.0]])
pcd = o3d.geometry.PointCloud()

# points = np.vstack([points1, points2])
# colors = np.vstack([colors1, colors2])
# pcd.points = o3d.utility.Vector3dVector(points)
# pcd.colors = o3d.utility.Vector3dVector(colors)

pcd.points = o3d.utility.Vector3dVector(points2)
pcd.colors = o3d.utility.Vector3dVector(colors2)


# 1. Define vertices for the surface
# For a simple plane, we can define 4 vertices
vertices = np.array([[0.0,    0.0,   32],
                     [800.0, 0.0,   32],
                     [0.0,    375.0, 6],
                     [800.0, 375.0, 6]])

# # 2. Define triangles using vertex indices
# # Each row represents a triangle, defined by 3 vertex indices
triangles = np.array([
    [0, 1, 2],  # First triangle
    [1, 3, 2]   # Second triangle
])

# 3. Create a TriangleMesh object
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(vertices)
mesh.triangles = o3d.utility.Vector3iVector(triangles)
mesh.compute_vertex_normals()

o3d.visualization.draw_geometries([pcd, mesh])
# o3d.visualization.draw_geometries([pcd])
