# project_on_xy_plane.py
#
# Same output as before, but now it relies on the new `project_2dTo3d`
# inside PinholeCamera instead of duplicating the maths.

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../lib")
from sim.camera import SimpleCamera

# ─────────────────────────────────────────────────────────────────────────────
#  Setup
# ─────────────────────────────────────────────────────────────────────────────
W, H         = 640, 480          # pixels
hfov_deg     = 60                # horizontal FOV
altitude_m   = 10.0              # camera height above Z=0 plane
tilt_deg     = -45.0             # nose-down pitch (deg)

cam = SimpleCamera(image_shape=(W, H), hfov_deg=hfov_deg)

# Euler angles for the *NED → FRD* aircraft rotation
rpy = (0.0, tilt_deg, 0.0)       # roll, pitch, yaw   (deg OK)

# ─────────────────────────────────────────────────────────────────────────────
#  Build a sparse pixel grid and back-project to NWU
# ─────────────────────────────────────────────────────────────────────────────
step = 10
u = np.arange(0, W, step)
v = np.arange(0, H, step)
uu, vv = np.meshgrid(u, v)
uv_pairs = np.stack([uu, vv], axis=-1).reshape(-1, 2)

dirs_nwu = cam.project_2dTo3d(uv_pairs, rpy=rpy).reshape(-1, 3)

# Camera origin (NWU)
cam_origin = np.array([0.0, 0.0, altitude_m])

# Intersect with ground plane  Z = 0
dz   = dirs_nwu[:, 2]
mask = dz < 0                      # only rays pointing downward
t    = -altitude_m / dz[mask]
pts  = cam_origin + dirs_nwu[mask] * t[:, None]

print(f"Intersections computed: {pts.shape[0]}")

# ─────────────────────────────────────────────────────────────────────────────
#  Plot
# ─────────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 6))
plt.scatter(pts[:, 0], pts[:, 1], s=1)
plt.xlabel("X (m)  –  North")
plt.ylabel("Y (m)  –  West")
plt.title(f"Ground intersections – Alt {altitude_m} m, Tilt {tilt_deg}°")
plt.axis("equal")
plt.grid(True)
plt.show()
