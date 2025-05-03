import numpy as np

def rotation_matrix_x(phi):
    """Rotation about x-axis."""
    c = np.cos(phi)
    s = np.sin(phi)
    return np.array([
        [1, 0, 0],
        [0, c, s],
        [0, -s, c]
    ])

def rotation_matrix_y(theta):
    """Rotation about y-axis."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, 0, -s],
        [0, 1, 0],
        [s, 0, c]
    ])

def rotation_matrix_z(psi):
    """Rotation about z-axis."""
    c = np.cos(psi)
    s = np.sin(psi)
    return np.array([
        [c, s, 0],
        [-s, c, 0],
        [0, 0, 1]
    ])

def body_to_earth_matrix(roll, pitch, yaw):
    """Construct full rotation matrix from Earth to Body using 3-2-1 convention."""
    Rz = rotation_matrix_z(yaw)
    Ry = rotation_matrix_y(pitch)
    Rx = rotation_matrix_x(roll)
    return Rx @ Ry @ Rz

def camera_to_body_matrix():
    """Fixed rotations from Body to Camera frame."""
    Rz_90 = rotation_matrix_z(np.pi/2)
    Rx_90 = rotation_matrix_x(np.pi/2)
    return Rx_90 @ Rz_90

def transform_camera_to_earth(points_camera, roll, pitch, yaw):
    """Transform (N,3) points from Camera frame to Earth frame."""
    C_cb = camera_to_body_matrix()
    C_be = body_to_earth_matrix(roll, pitch, yaw)
    
    # Transpose because we are moving backward
    C_bc = C_cb.T
    C_eb = C_be.T

    # Full rotation matrix: camera to earth
    C_ec = C_eb @ C_bc

    # Apply to all points
    points_camera = np.asarray(points_camera)
    points_earth = points_camera @ C_ec.T  # Notice transpose here for proper broadcasting
    return points_earth

# Example usage:
if __name__ == "__main__":
    # Given roll, pitch, yaw in radians
    roll = np.deg2rad(10)    # example roll +10 degrees
    pitch = np.deg2rad(5)    # example pitch +5 degrees
    yaw = np.deg2rad(30)     # example yaw +30 degrees

    # Example (N, 3) points in camera frame
    points_camera = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    # Transform to Earth frame
    points_earth = transform_camera_to_earth(points_camera, roll, pitch, yaw)

    print("Points in Earth frame:\n", points_earth)
