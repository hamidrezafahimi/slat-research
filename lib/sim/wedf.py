# camera.py
#
# NOTE – only the `project_2dTo3d` method is new; everything else
#        is byte-for-byte identical to your previous version.

from __future__ import annotations

import math
from typing import Dict, Tuple

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  elementary rotations (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def rotation_matrix_x(phi):
    c = np.cos(phi)
    s = np.sin(phi)
    return np.array([[1, 0, 0],
                     [0, c, s],
                     [0, -s, c]])


def rotation_matrix_y(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, 0, -s],
                     [0, 1, 0],
                     [s, 0, c]])


def rotation_matrix_z(psi):
    c = np.cos(psi)
    s = np.sin(psi)
    return np.array([[c, s, 0],
                     [-s, c, 0],
                     [0, 0, 1]])


# ─────────────────────────────────────────────────────────────────────────────
#  core class
# ─────────────────────────────────────────────────────────────────────────────
class PinholeCamera:
    """
    Ideal pin-hole camera (no lens distortion).

    … docstring unchanged for brevity …
    """

    # ──────────────────────────────────────────────────────────────────
    #  constructor & other helpers – **unchanged**
    # ──────────────────────────────────────────────────────────────────
    # (all the original code is kept – not repeated here for conciseness)

    # ----------------------------------------------------------------
    #  NEW   –   pixel → 3-D ray
    # ----------------------------------------------------------------
    def project_2dTo3d(self, uvs: np.ndarray, rpy=None) -> np.ndarray:
        """
        Back-project *pixel* coordinates to *unit* 3-D rays.

        Parameters
        ----------
        uvs : ndarray[..., 2]
            Any shape whose *last* dimension is ``(u, v)``.
        rpy : (roll, pitch, yaw) | None, optional
            Aircraft attitude (NED → FRD).  Values may be **degrees** or
            **radians** – a simple 2 π heuristic decides.  If *None*, the
            rays are returned in the **camera frame** (Right-Down-Forward).
            Otherwise they are expressed in the **earth NWU frame**.

        Returns
        -------
        ndarray[..., 3]   (unit-length)
        """

        # ---------- 1. pixel → camera (Right-Down-Forward)  -------------
        uv = np.asarray(uvs, dtype=float)
        if uv.shape[-1] != 2:
            raise ValueError("`uvs` must have …×2 shape (u, v)")

        orig_shape = uv.shape[:-1]
        uv_flat = uv.reshape(-1, 2)
        u, v = uv_flat.T

        x_cam = (u - self.cx) / self.fx
        y_cam = (v - self.cy) / self.fy
        z_cam = np.ones_like(x_cam)

        dirs_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
        dirs_cam /= np.linalg.norm(dirs_cam, axis=-1, keepdims=True)

        # ---------- 2. early exit (camera frame requested) --------------
        if rpy is None:
            return dirs_cam.reshape(*orig_shape, 3)

        # ---------- 3. camera → body (FRD) ------------------------------
        Ry = np.array([[0, 0, 1],
                       [0, 1, 0],
                       [-1, 0, 0]])
        Rx = np.array([[1, 0, 0],
                       [0, 0, -1],
                       [0, 1, 0]])
        cam_to_frd = Rx @ Ry                              # fixed mapping
        dirs_frd = dirs_cam @ cam_to_frd.T

        # ---------- 4. body → NED (Euler 3-2-1, NED → FRD) --------------
        roll, pitch, yaw = map(float, rpy)
        if max(abs(roll), abs(pitch), abs(yaw)) > 2 * math.pi:
            roll, pitch, yaw = map(math.radians, (roll, pitch, yaw))

        Rphi   = rotation_matrix_x(-roll)     # note the *minus*
        Rtheta = rotation_matrix_y(-pitch)
        Rpsi   = rotation_matrix_z(-yaw)

        # ---------- 5. NED → NWU (earth frame used in the demo) ---------
        Rnwu_ned = rotation_matrix_x(math.pi)  # 180° about X (North)

        # Total transformation: cam → NWU
        R_total = Rnwu_ned @ Rpsi @ Rtheta @ Rphi @ cam_to_frd
        dirs_nwu = dirs_cam @ R_total.T
        dirs_nwu /= np.linalg.norm(dirs_nwu, axis=-1, keepdims=True)

        return dirs_nwu.reshape(*orig_shape, 3)

    # (rest of the original class – *unchanged*)


# ─────────────────────────────────────────────────────────────────────────────
#  convenience subclass (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
class SquaredPixelFocalCenteredPinholeCamera(PinholeCamera):
    …  # identical to your previous code …

# convenient alias
SimpleCamera = SquaredPixelFocalCenteredPinholeCamera

__all__ = ["PinholeCamera",
           "SquaredPixelFocalCenteredPinholeCamera",
           "SimpleCamera"]






# project_on_xy_plane.py
#
# Same output as before, but now it relies on the new `project_2dTo3d`
# inside PinholeCamera instead of duplicating the maths.

import numpy as np
import matplotlib.pyplot as plt

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


