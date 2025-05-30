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

def transform_kps_nwu2camera(points, cam_pose, roll, pitch, yaw):
    ids = points[:,0:1]
    xyz = points[:,1:4]
    xyx_transformed = transform_nwu_to_camera(xyz, cam_pose, roll, pitch, yaw)
    return np.hstack([ids, xyx_transformed])

def transform_kps_camera2nwu(points, cam_pose, roll, pitch, yaw):
    ids = points[:,0:1]
    xyz = points[:,1:4]
    xyx_transformed = transform_camera_to_nwu(xyz, cam_pose, roll, pitch, yaw)
    return np.hstack([ids, xyx_transformed])

def transform_nwu_to_camera(points, cam_pose, roll, pitch, yaw):
    points_translated = points - cam_pose

    R1 = rotation_matrix_x(np.pi)
    Rpsi = rotation_matrix_z(yaw)
    Rtheta = rotation_matrix_y(pitch)
    Rphi = rotation_matrix_x(roll)
    R5 = rotation_matrix_x(np.pi/2)
    R6 = rotation_matrix_y(np.pi/2)

    R = R6 @ R5 @ Rphi @ Rtheta @ Rpsi @ R1
    points_out = points_translated @ R.T
    return points_out
    # return points_out - cam_pose

def transform_camera_to_nwu(points, cam_pose, roll, pitch, yaw):
    points_translated = points + cam_pose

    R1 = rotation_matrix_x(np.pi)
    Rpsi = rotation_matrix_z(yaw)
    Rtheta = rotation_matrix_y(pitch)
    Rphi = rotation_matrix_x(roll)
    R5 = rotation_matrix_x(np.pi/2)
    R6 = rotation_matrix_y(np.pi/2)

    R = R1 @ Rpsi @ Rtheta @ Rphi @ R5 @ R6
    # points_out = points @ R.T
    points_out = points_translated @ R.T
    return points_out