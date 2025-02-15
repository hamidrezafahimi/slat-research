import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

def expand_segments(laplacian_image, mask_image, laplacian_threshold):
    """
    Expands the zero-valued regions in the mask image until the Laplacian values
    at the boundaries reach or exceed the given threshold (considering both +L and -L).

    Args:
        laplacian_image (numpy.ndarray): The Laplacian image (2D array).
        mask_image (numpy.ndarray): The boolean mask image (2D array, 0 for annotated segments, 1 for free areas).
        laplacian_threshold (float): The Laplacian threshold value.

    Returns:
        numpy.ndarray: The expanded mask image.
    """
    # Ensure the inputs are numpy arrays
    laplacian_image = np.asarray(laplacian_image, dtype=np.float32)
    mask_image = np.asarray(mask_image, dtype=np.uint8)

    # Invert the mask so that zero regions are treated as the regions of interest
    inverted_mask = (mask_image == 0).astype(np.uint8)

    # Define a kernel for morphological dilation (4-connectivity)
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=np.uint8)

    # Initialize the expanded mask with the inverted mask
    expanded_mask = inverted_mask.copy()

    while True:
        # Dilate the current mask to expand the region
        dilated_mask = cv.dilate(expanded_mask, kernel, iterations=1)

        # Find the boundary of the newly expanded region
        boundary_mask = dilated_mask - expanded_mask

        # Find the Laplacian values at the boundary
        boundary_laplacian = laplacian_image[boundary_mask == 1]

        # Check where the Laplacian values are within the acceptable range
        boundary_condition = (boundary_laplacian > -laplacian_threshold) & (boundary_laplacian < laplacian_threshold)

        if not np.any(boundary_condition):
            # Stop if no boundary pixel meets the condition
            break

        # Add the valid boundary pixels to the expanded mask
        expanded_mask[boundary_mask == 1] = boundary_condition

    # Convert the expanded mask back to the original mask format (1 for free areas, 0 for segments)
    final_mask = (expanded_mask == 0).astype(np.uint8)

    return final_mask


def compute_second_derivative(array, dx=1):
    """
    Computes the second derivative of a 1D array using finite differences.
    
    Parameters:
        array (np.ndarray): Input 1D array.
        dx (float): Spacing between points in the array.
        
    Returns:
        np.ndarray: Second derivative array.
    """
    second_derivative = np.zeros_like(array)
    # Compute second derivative for interior points
    second_derivative[1:-1] = (array[2:] - 2 * array[1:-1] + array[:-2]) / (dx ** 2)
    # Optionally handle boundary conditions (e.g., extrapolate or leave as zeros)
    return second_derivative


def calcFlatGroundDepth(tilt_angle, h=10, vertical_fov_degrees=0.75*66.0, horizontal_fov_degrees=66.0, resolution=(440, 330)):
    """
    Calculate the depth data for a camera image based on the geometry of light rays.

    Parameters:
        tilt_angle (float): The tilt angle of the camera in radians.
        h (float): Altitude of the camera in meters. Default is 10.
        vertical_fov_degrees (float): Vertical field of view in degrees. Default is 0.75*66.0.
        horizontal_fov_degrees (float): Horizontal field of view in degrees. Default is 66.0.
        resolution (tuple): Resolution of the camera (width, height). Default is (440, 330).

    Returns:
        np.ndarray: Normalized depth data as a 2D numpy array.
    """
    width, height = resolution
    # Convert FOVs to radians
    vertical_fov = np.deg2rad(vertical_fov_degrees)
    horizontal_fov = np.deg2rad(horizontal_fov_degrees)
    # Generate angles for the vertical and horizontal directions
    vertical_angles = (np.linspace(-vertical_fov / 2, vertical_fov / 2, height) + tilt_angle)[::-1]
    horizontal_angles = np.linspace(-horizontal_fov / 2, horizontal_fov / 2, width)
    # Create a 2D array to store depth values
    depth_data = np.zeros((height, width))
    # Calculate depth for each pixel
    projectionYs = np.zeros((height, width))
    projectionXs = np.zeros((height, width))
    for i, v_angle in enumerate(vertical_angles):
        for j, h_angle in enumerate(horizontal_angles):
            if -np.pi / 2 <= v_angle <= 0:  # Ensure valid vertical angles
                r = h / abs(np.sin(v_angle))  # Length in vertical plane
                R = r / abs(np.cos(h_angle))  # Adjust for horizontal angle
                depth_data[i, j] = R
                # point = R * np.array([])
                # projectionXs[i, j] = 
            # else:
            #     depth_data[i, j] = np.inf  # Invalid pixels set to infinity

    max_depth = depth_data.max()
    min_depth = depth_data.min()
    # # dimg = ((1 - depth_data / max_depth)-min_depth)
    # dimg = (1 - depth_data / max_depth)
    # return (255 * dimg).astype(np.uint8)
    # Map depth data into the 0-255 range and shift down to ensure the darkest points are zero
    dimg = (1 - depth_data / max_depth)
    dimg = dimg - dimg.min()  # Shift the values so the minimum is 0

    return (255 * dimg).astype(np.uint8)


# if __name__ == '__main__':
    # Example usage
    # tilt_angle = -0.78

    # plt.figure()
    # img = calcFlatGroundDepth(tilt_angle)
    # for i, val in enumerate(img[:, 220]):
    #     plt.scatter(i, val, color='k')
    # plt.show()
    # cv.imshow("sd", img)
    # cv.waitKey()
