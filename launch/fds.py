import numpy as np

def rotation_matrix_to_euler(R):
    """
    Convert rotation matrix R to Euler angles (roll, pitch, yaw)
    assuming R = Rz * Ry * Rx.
    Returns angles in radians.
    """
    # Check for gimbal lock
    if abs(R[2,0]) < 1.0:
        pitch = np.arcsin(R[2,0])
        roll  = np.arctan2(-R[2,1], R[2,2])
        yaw   = np.arctan2(-R[1,0], R[0,0])
    else:
        # Gimbal lock: pitch = ±90°
        pitch = np.pi/2 if R[2,0] >= 1 else -np.pi/2
        roll  = np.arctan2(R[0,1], R[1,1])
        yaw   = 0.0  # arbitrary

    pitch = pitch * 180 / 3.1415
    roll  = roll  * 180 / 3.1415
    yaw   = yaw   * 180 / 3.1415

    return roll, pitch, yaw

# Example usage:
cy, sy = np.cos(np.radians(-45)), np.sin(np.radians(-45))
cp, sp = np.cos(np.radians(-5)), np.sin(np.radians(-5))
cr, sr = np.cos(np.radians(-10)), np.sin(np.radians(-10))

R1 = np.array([[cy, -sy, 0],
               [sy,  cy, 0],
               [0,   0,  1]])

R2 = np.array([[cp,  0, sp],
               [0,   1, 0],
               [-sp, 0, cp]])

R3 = np.array([[1, 0,  0],
               [0, cr, -sr],
               [0, sr, cr]])

R = R1 @ R2 @ R3

roll, pitch, yaw = rotation_matrix_to_euler(R)
print("roll:", roll, "pitch:", pitch, "yaw:", yaw)
