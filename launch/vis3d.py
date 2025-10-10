import numpy as np
import open3d as o3d


def euler_to_rotation(phi, theta, psi):
    """Convert Euler angles (in radians) to rotation matrix (Z-Y-X order)."""
    Rz = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi),  np.cos(psi), 0],
        [0, 0, 1]
    ])
    Ry = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]
    ])
    return Rz @ Ry @ Rx


def create_ground_grid(size=10, step=1):
    """Create a simple ground grid."""
    lines = []
    points = []
    for i in np.arange(-size, size + step, step):
        # parallel to X
        points.append([i, -size, 0])
        points.append([i, size, 0])
        lines.append([len(points) - 2, len(points) - 1])
        # parallel to Y
        points.append([-size, i, 0])
        points.append([size, i, 0])
        lines.append([len(points) - 2, len(points) - 1])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[0.7, 0.7, 0.7]] * len(lines))
    return line_set


def create_camera_frame(position, R, scale=0.5):
    """Create a small coordinate frame representing the camera."""
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
    frame.rotate(R, center=(0, 0, 0))
    frame.translate(position)
    return frame


def create_normal_vector(position, R, length=1.0):
    """Show camera's forward direction as a line (normal vector)."""
    z_axis = R[:, 2]  # forward direction (camera's +Z)
    start = np.array(position)
    end = start + length * z_axis
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector([start, end])
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # red line
    return line_set


def visualize_camera(x, y, z, phi_deg, theta_deg, psi_deg):
    # Convert degrees to radians
    phi, theta, psi = np.radians([phi_deg, theta_deg, psi_deg])

    # Compute rotation
    R = euler_to_rotation(phi, theta, psi)

    # Create geometry
    grid = create_ground_grid(size=10, step=0.1)
    camera = create_camera_frame([x, y, z], R, scale=0.3)
    normal = create_normal_vector([x, y, z], R, length=1.0)

    # Visualize
    o3d.visualization.draw_geometries([grid, camera, normal],
                                      window_name='3D Camera Attitude',
                                      width=1000, height=800,
                                      point_show_normal=False)


if __name__ == "__main__":
    # Example camera pose
    x, y, z = 0.0, 0.0, 0.5      # position in meters
    phi, theta, psi = 90, 0, 0  # roll, pitch, yaw in degrees

    visualize_camera(x, y, z, phi, theta, psi)
