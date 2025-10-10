import numpy as np
import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../lib")
from projection.helper import rotation_matrix_x, rotation_matrix_y, rotation_matrix_z

roll_rad = 0
pitch_rad = -1.57
yaw_rad = 0

# Camera-to-body
Ry = np.array([[ 0,  0, 1], [0, 1, 0], [-1, 0, 0]])
Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
# Body to NED
Rphi   = rotation_matrix_x(-roll_rad)
Rtheta = rotation_matrix_y(-pitch_rad)
Rpsi   = rotation_matrix_z(-yaw_rad)
Rnwu   = rotation_matrix_x(np.pi)

RR = Rpsi @ Rtheta @ Rphi
R = Rnwu @ Rpsi @ Rtheta @ Rphi @ Rx @ Ry
RRR = ((Rnwu.T @ R) @ Ry.T) @ Rx.T
ROT_code = R
# print(RR)
# print(RRR)
# # R = Rpsi @ Rtheta @ Rphi @ Rx @ Ry
# dd = np.array([0,0,1])
# d = dd @ R.T
import json
import numpy as np

# Function to extract Euler angles from rotation matrix
def decodeWildUav(Rot: np.ndarray):
    """
    Convert rotation matrix R to Euler angles (roll, pitch, yaw)
    assuming R = Rz * Ry * Rx.
    Returns angles in radians.
    """
    Ry = np.array([[ 0,  0, 1], [0, 1, 0], [-1, 0, 0]])
    Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    Rnwu   = rotation_matrix_x(np.pi)
    R = ((Rnwu.T @ Rot) @ Ry.T) @ Rx.T
    
    print(Rot)
    print("        ")
    print("        ")
    print("        ")
    print(R)
    
    # Check for gimbal lock
    if abs(R[2,0]) >= 1.0:
        # Gimbal lock, pitch = ±90°
        pitch = np.pi / 2 * np.sign(-R[2,0])
        roll = 0.0
        yaw = np.arctan2(-R[0,1], R[1,1])
    else:
        pitch = np.arcsin(-R[2,0])
        roll = np.arctan2(R[2,1], R[2,2])
        yaw = np.arctan2(R[1,0], R[0,0])
    
    return roll, pitch, yaw

# Read JSON file
with open("/home/psash/Desktop/WildUAV/seq00/meta/metadata/000000.json", "r") as f:
    data = json.load(f)

# Extract rotation matrix
R_list = data["rotation"]
# print(np.array([0,0,1]) @ np.array(R_list).T)

# Compute Euler angles
roll, pitch, yaw = decodeWildUav(np.array(R_list))
print("roll (deg):", np.degrees(roll))
print("pitch (deg):", np.degrees(pitch))
print("yaw (deg):", np.degrees(yaw))

# roll, pitch, yaw = decodeWildUav(ROT_code)
# print("roll (deg):", np.degrees(roll))
# print("pitch (deg):", np.degrees(pitch))
# print("yaw (deg):", np.degrees(yaw))