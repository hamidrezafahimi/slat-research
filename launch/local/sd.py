import open3d as o3d
import numpy as np

# 1. Define vertices for the surface
# For a simple plane, we can define 4 vertices
vertices = np.array([[0.0,   0.0,     0.0],
                     [375.0, 0.0,     203.0],
                     [0.0,   1240.0,  0.0],
                     [375.0, 1240.0,  203.0]])

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

# # (Optional) Compute normals for proper lighting and rendering
# mesh.compute_vertex_normals()

# 4. Visualize the mesh
o3d.visualization.draw_geometries([mesh], window_name="Simple Surface Example")