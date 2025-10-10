import json
import numpy as np

# Function to extract Euler angles from rotation matrix
def rotation_matrix_to_euler(R):
    """
    Convert rotation matrix R to Euler angles (roll, pitch, yaw)
    assuming R = Rz * Ry * Rx.
    Returns angles in radians.
    """
    if abs(R[2,0]) < 1.0:
        pitch = np.arcsin(R[2,0])
        roll  = np.arctan2(-R[2,1], R[2,2])
        yaw   = np.arctan2(-R[1,0], R[0,0])
    else:
        # Gimbal lock
        pitch = np.pi/2 if R[2,0] >= 1 else -np.pi/2
        roll  = np.arctan2(R[0,1], R[1,1])
        yaw   = 0.0
    return roll, pitch, yaw

# Read JSON file
with open("/home/psash/Desktop/WildUAV/seq00/meta/metadata/000000.json", "r") as f:
    data = json.load(f)

# Extract rotation matrix
R_list = data["rotation"]
R = np.array(R_list)

# Compute Euler angles
roll, pitch, yaw = rotation_matrix_to_euler(R)

print("roll (rad):", roll)
print("pitch (rad):", pitch)
print("yaw (rad):", yaw)

# Optional: convert to degrees
print("roll (deg):", np.degrees(roll))
print("pitch (deg):", np.degrees(pitch))
print("yaw (deg):", np.degrees(yaw))


def rotation_matrix(roll, pitch, yaw):
    cr, sr = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))

    Rz = np.array([[cy,  -sy,  0],
                   [sy, cy,  0],
                   [ 0,   0,  1]])

    Ry = np.array([[cp,  0, sp],
                   [ 0,  1,  0],
                   [-sp,  0,  cp]])

    Rx = np.array([[1,  0,   0],
                   [0,  cr,  -sr],
                   [0, sr,  cr]])

    R = Rz @ Ry @ Rx
    return R