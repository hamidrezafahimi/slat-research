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
K1 = -0.1
K2 = 0.01
K3 = 0
P1 = 0.001  # Tangential distortion parameter 1
P2 = 0.001  # Tangential distortion parameter 2

# Alternative radial parameters for inward curvature
K1_inward = 0.15   # Positive K1 for pincushion distortion
K2_inward = 0.02  # Negative K2 to balance higher order effects
K3_inward = 0.001  # Small positive K3

def pinhole_dir(u, v, unique=True):
    x = (u - CX) / FX
    y = (v - CY) / FY
    z = 1
    if unique:
        return np.array([x, y, z]) / np.linalg.norm([x, y, z])
    else:
        return np.array([x, y, z])

def fisheye_dir(u, v):
    dir = pinhole_dir(u, v, False)
    x, y = dir[0], dir[1]
    r = np.hypot(x, y)
    theta = r
    if r == 0:
        return np.array([0, 0, 1])
    phi = np.arctan2(y, x)
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])

def radial_dir(u, v):
    dir = pinhole_dir(u, v)
    x, y = dir[0], dir[1]
    r2 = x * x + y * y
    distortion = 1 + K1 * r2 + K2 * r2**2 + K3 * r2**3
    x_distorted = x * distortion
    y_distorted = y * distortion
    return np.array([x_distorted, y_distorted, 1]) / np.linalg.norm([x_distorted, y_distorted, 1])

def radtan_dir(u, v):
    """
    Radial-tangential (Brown-Conrady) distortion model
    Combines radial and tangential distortion components
    """
    # Start with normalized coordinates
    x = (u - CX) / FX
    y = (v - CY) / FY
    
    # Radial distance squared
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    
    # Radial distortion factor
    radial_factor = 1 + K1 * r2 + K2 * r4 + K3 * r6
    
    # Tangential distortion components
    dx_tangential = 2 * P1 * x * y + P2 * (r2 + 2 * x * x)
    dy_tangential = P1 * (r2 + 2 * y * y) + 2 * P2 * x * y
    
    # Apply both radial and tangential distortion
    x_distorted = x * radial_factor + dx_tangential
    y_distorted = y * radial_factor + dy_tangential
    
    # Create the ray direction
    z = 1
    return np.array([x_distorted, y_distorted, z]) / np.linalg.norm([x_distorted, y_distorted, z])

def tangential_dir(u, v):
    """
    Pure tangential distortion model
    Only applies tangential distortion without radial component
    """
    # Start with normalized coordinates
    x = (u - CX) / FX
    y = (v - CY) / FY
    
    # Radial distance squared (needed for tangential distortion)
    r2 = x * x + y * y
    
    # Tangential distortion components only
    dx_tangential = 2 * P1 * x * y + P2 * (r2 + 2 * x * x)
    dy_tangential = P1 * (r2 + 2 * y * y) + 2 * P2 * x * y
    
    # Apply tangential distortion
    x_distorted = x + dx_tangential
    y_distorted = y + dy_tangential
    
    # Create the ray direction
    z = 1
    return np.array([x_distorted, y_distorted, z]) / np.linalg.norm([x_distorted, y_distorted, z])

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
fisheye = footprint(fisheye_dir)
rad = footprint(radial_dir)
radtan = footprint(radtan_dir)
tangential = footprint(tangential_dir)
# trapezoid_curved = footprint(trapezoid_curved_dir)
kitti = footprint(kitti_dir)

# Plot all seven models
plt.figure(figsize=(18,12))
plt.subplot(3,2,4); plt.scatter(*kitti.T, s=.1); plt.title("KITTI (Plumb Bob)"); plt.axis('equal')
plt.subplot(3,2,1); plt.scatter(*pinhole.T, s=.1); plt.title("Pinhole"); plt.axis('equal')
plt.subplot(3,2,2); plt.scatter(*fisheye.T, s=.1); plt.title("Fisheye"); plt.axis('equal')
plt.subplot(3,2,3); plt.scatter(*rad.T, s=.1); plt.title("Radial (Barrel)"); plt.axis('equal')
plt.subplot(3,2,5); plt.scatter(*radtan.T, s=.1); plt.title("Radial-Tangential"); plt.axis('equal')
plt.subplot(3,2,6); plt.scatter(*tangential.T, s=.1); plt.title("Tangential"); plt.axis('equal')
# plt.subplot(3,3,7); plt.scatter(*trapezoid_curved.T, s=.1); plt.title("Trapezoid Curved"); plt.axis('equal')
plt.tight_layout()
plt.show()