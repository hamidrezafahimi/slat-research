import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv

def depthImage2pointCloud(D, horizontal_fov, pitch_rad, max_dist = 256):
    """
    Computes the unit direction vectors for each pixel in the image coordinate system
    
    Parameters:
    - image_width (int): Width of the image in pixels.
    - image_height (int): Height of the image in pixels.
    - horizontal_fov (float): Horizontal field of view of the camera in degrees.
    
    Returns:
    - np.ndarray: A (H, W, 3) array where each pixel contains its unit direction vector (X, Y, Z).
    """
    # Convert angles to radians
    horizontal_fov_rad = np.radians(horizontal_fov)
    
    # Compute focal length using the horizontal FOV
    image_height, image_width = D.shape
    focal_length = (image_width / 2) / np.tan(horizontal_fov_rad / 2)
    
    # Compute the intrinsic center
    cx, cy = image_width / 2, image_height / 2
    
    # Create a grid of pixel coordinates
    x_indices = np.arange(image_width)
    y_indices = np.arange(image_height)
    
    # Create meshgrid for pixel coordinates
    x_grid, y_grid = np.meshgrid(x_indices, y_indices)
    
    X = (x_grid - cx) / focal_length
    Y = (y_grid - cy) / focal_length
    Z = np.ones_like(X)
    
    # Normalize to unit vectors
    norm = np.sqrt(X**2 + Y**2 + Z**2)
    X /= norm
    Y /= norm
    Z /= norm

    direction_vectors = np.stack((X, Y, Z), axis=-1)

    Ry = np.array([
        [ 0,  0, 1],
        [ 0,  1, 0],
        [ -1,  0, 0]
    ])

    Rx = np.array([
        [1,  0,  0],
        [0,  0,  -1],
        [0, 1,  0]
    ])

    # Combined rotation: first -90° around Z, then -90° around X
    R = Rx @ Ry

    # Apply rotation to all direction vectors
    direction_vectors_ned = direction_vectors @ R.T # shape remains (H, W, 3)

    # pitch_rad is the rotation of camera. so the point cloud must rotate in reverse angle
    Ry = np.array([
        [ np.cos(-pitch_rad),  0, -np.sin(-pitch_rad)],
        [ 0,  1, 0],
        [ np.sin(-pitch_rad),  0, np.cos(-pitch_rad)]
    ])
        
    unit_vectors = direction_vectors_ned @ Ry.T

    # Compute point cloud
    depth_meters = (255 - D) * (max_dist / 255.0)
    return unit_vectors * depth_meters[..., np.newaxis]


def plot_point_cloud(point_cloud):
    # Sample points for visualization
    step = 20
    X_sample = point_cloud[::step, ::step, 0]
    Y_sample = point_cloud[::step, ::step, 1]
    Z_sample = point_cloud[::step, ::step, 2]

    # Plot the point cloud
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(0, 0, 0, color='red')
    ax.plot(X_sample.flatten(), Y_sample.flatten(), Z_sample.flatten(), 'b.', markersize=1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=-180, azim=0)
    plt.show()

import sys
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py depth_image.jpg")
        sys.exit(1)

    depth_path = sys.argv[1]
    
    D = cv.imread(depth_path, cv.IMREAD_GRAYSCALE)
    if D is None:
        raise IOError(f"Could not load {depth_path}. Check the file path.")

    # Example usage
    point_cloud = depthImage2pointCloud(D, horizontal_fov=66, pitch_rad=-0.75)

