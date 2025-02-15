import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv

def generate_unit_direction_vectors(image_width, image_height, horizontal_fov, tilt_angle=0):
    """
    Computes the unit direction vectors for each pixel in the image coordinate system,
    with an optional tilt correction.
    
    Parameters:
    - image_width (int): Width of the image in pixels.
    - image_height (int): Height of the image in pixels.
    - horizontal_fov (float): Horizontal field of view of the camera in degrees.
    - tilt_angle (float): Tilt angle of the camera in degrees (positive looks down).
    
    Returns:
    - np.ndarray: A (H, W, 3) array where each pixel contains its unit direction vector (X, Y, Z).
    """
    # Convert angles to radians
    horizontal_fov_rad = np.radians(horizontal_fov)
    tilt_rad = np.radians(tilt_angle)
    
    # Compute focal length using the horizontal FOV
    focal_length = (image_width / 2) / np.tan(horizontal_fov_rad / 2)
    
    # Compute the intrinsic center
    cx, cy = image_width / 2, image_height / 2
    
    # Create a grid of pixel coordinates
    x_indices = np.arange(image_width)
    y_indices = np.arange(image_height)
    
    # Create meshgrid for pixel coordinates
    x_grid, y_grid = np.meshgrid(x_indices, y_indices)
    
    # Compute direction vectors
    X = (x_grid - cx) / focal_length
    Y = (y_grid - cy) / focal_length
    Z = np.ones_like(X)
    
    # Normalize to unit vectors
    norm = np.sqrt(X**2 + Y**2 + Z**2)
    X /= norm
    Y /= norm
    Z /= norm
    
    # Apply tilt rotation around the X-axis
    Y_tilted = Y * np.cos(tilt_rad) - Z * np.sin(tilt_rad)
    Z_tilted = Y * np.sin(tilt_rad) + Z * np.cos(tilt_rad)
    
    # Stack into a 3D array
    direction_vectors = np.stack((X, Y_tilted, Z_tilted), axis=-1)
    
    return direction_vectors

# Example usage
if __name__ == "__main__":
    image_w, image_h = 440, 330
    horizontal_fov = 66  # Example horizontal FOV in degrees
    tilt_angle = -40  # Example tilt angle in degrees
    
    unit_vectors = generate_unit_direction_vectors(image_w, image_h, horizontal_fov, tilt_angle)
    
    # Generate random depth data
    # np.random.seed(42)  # For reproducibility
    # D = np.random.rand(image_h, image_w) * 10  # Depth values in range [0,10]
    D = cv.imread('/media/hamid/Workspace/thing/depthpro_output.png', cv.IMREAD_GRAYSCALE)

    # Compute point cloud
    point_cloud = unit_vectors * D[..., np.newaxis]
    
    # Sample points for visualization
    step = 40  # Adjust for better visualization
    X_sample = point_cloud[::step, ::step, 0]
    Y_sample = point_cloud[::step, ::step, 1]
    Z_sample = point_cloud[::step, ::step, 2]
    
    # Plot the point cloud
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X_sample.flatten(), Y_sample.flatten(), Z_sample.flatten(), 'b.', markersize=1)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Point Cloud from Depth Data")
    plt.show()
    
    print("Point Cloud Shape:", point_cloud.shape)
    print("Sample Point (center pixel):", point_cloud[image_h // 2, image_w // 2])
