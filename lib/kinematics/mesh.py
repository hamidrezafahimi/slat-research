import open3d as o3d
import numpy as np

def translate_mesh(mesh: o3d.geometry.TriangleMesh, translation: np.ndarray) -> o3d.geometry.TriangleMesh:
    """
    Translate an Open3D mesh by a constant [x, y, z] vector.

    Args:
        mesh (o3d.geometry.TriangleMesh): The input mesh.
        translation (array-like): A 3-element vector [dx, dy, dz].

    Returns:
        o3d.geometry.TriangleMesh: A new translated mesh.
    """
    if not isinstance(translation, np.ndarray):
        translation = np.array(translation, dtype=float)

    # Make a copy so the original mesh is unchanged
    mesh_copy = o3d.geometry.TriangleMesh(mesh)
    
    # Apply translation (relative=True means additive offset)
    mesh_copy.translate(translation, relative=True)
    return mesh_copy