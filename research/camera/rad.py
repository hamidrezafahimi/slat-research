import numpy as np
import matplotlib.pyplot as plt

alt = 10.0
pitch = -np.deg2rad(45)
W, H, hfov = 640, 480, 80
CX = 320
CY = 240
FX = 554
FY = 554
# Distortion parameters
K1 = 0
K2 = 0
K3 = 0
P1 = 0.002  # Tangential distortion parameter 1
P2 = -0.002  # Tangential distortion parameter 2

def pinhole_dir(u, v, unique=True):
    x = (u - CX) / FX
    y = (v - CY) / FY
    z = 1
    if unique:
        return np.array([x, y, z]) / np.linalg.norm([x, y, z])
    else:
        return np.array([x, y, z])


def kitti_dir(u, v, fx=FX, fy=FY, cx=CX, cy=CY, k1=K1, k2=K2, p1=P1, p2=P2, k3=K3):
    """
    KITTI plumb-bob distortion model.
    Applies radial and tangential distortion to normalized pixel coordinates.
    """
    x = (u - cx) / fx
    y = (v - cy) / fy

    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2

    radial = 1 + k1 * r2 + k2 * r4 + k3 * r6

    dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y

    x_dist = x * radial + dx
    y_dist = y * radial + dy

    return np.array([x_dist, y_dist, 1]) / np.linalg.norm([x_dist, y_dist, 1])


def footprint(ray_fn):
    Rx = np.array([[1,0,0],
                   [0,np.cos(pitch), -np.sin(pitch)],
                   [0,np.sin(pitch), np.cos(pitch)]])
    M = np.array([[1,0,0],
                  [0,0,1],
                  [0,-1,0]]) # cam → world
    pts = []
    for v in range(H):
        for u in range(W):
            d = ray_fn(u, v) @ Rx.T @ M.T
            if d[2] < 0: # hits ground
                t = -alt / d[2]
                pts.append(d[:2]*t)
    return np.asarray(pts)


# Generate footprints for all distortion models
pinhole = footprint(pinhole_dir)
kitti = footprint(kitti_dir)

# Plot all seven models
plt.figure(figsize=(18,12))
plt.subplot(2,1,1); plt.scatter(*pinhole.T, s=.1); plt.title("Pinhole"); plt.axis('equal')
plt.subplot(2,1,2); plt.scatter(*kitti.T, s=.1); plt.title("KITTI (Plumb Bob)"); plt.axis('equal')
plt.tight_layout()
plt.show()