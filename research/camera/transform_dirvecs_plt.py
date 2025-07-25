import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# Image size
W, H = 16, 9
aspect_ratio = H / W

# FOVs in radians
hfov_rad = np.deg2rad(80)
vfov_rad = 2 * np.arctan(np.tan(hfov_rad / 2) * aspect_ratio)


# Camera center (focal point)
camera_origin = np.array([0.0, 0.0, 10.0])

# Generate angle arrays for each pixel
x_angles = np.linspace(-hfov_rad / 2, hfov_rad / 2, W)
y_angles = np.linspace(-vfov_rad / 2, vfov_rad / 2, H)
print(y_angles)

# Create meshgrid of angles
theta, phi = np.meshgrid(x_angles, y_angles)

# Compute direction vectors (in camera coordinate system)
x = np.tan(theta)
y = np.tan(phi)
z = np.ones_like(x)
# z = np.sqrt(1-x**2-y**2)
# Plot
# fig1 = plt.figure(figsize=(10, 7))
# ax1 = fig1.add_subplot(111, projection='3d')
# ax1.scatter(x,y,svfd)
# ax1.set_xlabel("X")
# ax1.set_ylabel("Y")
# ax1.set_zlabel("Z")

dirs = np.stack((x, y, z), axis=-1)
norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
unit_dirs = dirs / norms

Ry = np.array([[ 0,  0, 1], [0, 1, 0], [-1, 0, 0]])
Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
Rphi   = rotation_matrix_x(0)
Rtheta = rotation_matrix_y(0.15)
Rpsi   = rotation_matrix_z(0)
Rnwu   = rotation_matrix_x(np.pi)
R = Rnwu @ Rpsi @ Rtheta @ Rphi @ Rx @ Ry

dirs_nwu = dirs @ R.T

# Sample fewer vectors for clearer plot
step = 1
sampled_dirs = dirs_nwu[::step, ::step]
H_s, W_s, _ = sampled_dirs.shape

# Reshape
vectors = sampled_dirs.reshape(-1, 3)

# Scale vectors
vec_length = 20
vectors_scaled = vectors * vec_length

# Create repeated origins at camera location
origins = np.tile(camera_origin, (vectors.shape[0], 1))

# Compute intersections with z=0 plane
# Ray: p = o + t*d → find t such that p_z = 0 ⇒ 0 = o_z + t*d_z ⇒ t = -o_z / d_z
t = -camera_origin[2] / vectors[:, 2]
intersections = origins + vectors * t[:, np.newaxis]  # shape (N, 3)

# Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Draw rays (as lines instead of arrows)
for origin, direction in zip(origins, vectors):
    # Choose a far scalar to simulate an "infinite" line
    t_min = 0   # optionally show part of ray behind camera
    t_max = 300

    line_pts = origin[np.newaxis, :] + np.array([t_min, t_max])[:, np.newaxis] * direction[np.newaxis, :]
    ax.plot(line_pts[:, 0], line_pts[:, 1], line_pts[:, 2], color='blue', linewidth=0.4)


# Draw red dots at intersection with XY plane (z=0)
ax.scatter(intersections[:, 0], intersections[:, 1], intersections[:, 2], color='red', s=4, 
           label='Intersections on z=0')

# Axis settings
ax.set_xlim([-50, 50])
ax.set_ylim([-50, 50])
ax.set_zlim([0, 15])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Camera Rays and Intersections with XY Plane")
ax.legend()
plt.tight_layout()
plt.show()
