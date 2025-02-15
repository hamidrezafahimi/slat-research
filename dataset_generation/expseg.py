import cv2
import numpy as np
import matplotlib.pyplot as plt

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
        dilated_mask = cv2.dilate(expanded_mask, kernel, iterations=1)

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

# Example usage:
if __name__ == "__main__":
    # Create a sample Laplacian image and mask image
    laplacian_image = np.random.rand(100, 100) * 10  # Simulated Laplacian image
    mask_image = np.ones((100, 100), dtype=np.uint8)
    mask_image[40:60, 40:60] = 0  # Annotated region in the mask

    laplacian_threshold = 5.0

    # Get the expanded mask
    expanded_mask = expand_segments(laplacian_image, mask_image, laplacian_threshold)

    # Display the images
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Laplacian Image")
    plt.imshow(laplacian_image, cmap='gray')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("Initial Mask")
    plt.imshow(mask_image, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Expanded Mask")
    plt.imshow(expanded_mask, cmap='gray')

    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Create a sample Laplacian image and mask image
    laplacian_image = np.random.rand(100, 100) * 10  # Simulated Laplacian image
    mask_image = np.ones((100, 100), dtype=np.uint8)
    mask_image[40:60, 40:60] = 0  # Annotated region in the mask

    laplacian_threshold = 5.0

    # Get the expanded mask
    expanded_mask = expand_segments(laplacian_image, mask_image, laplacian_threshold)

    # Display the images
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Laplacian Image")
    plt.imshow(laplacian_image, cmap='gray')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("Initial Mask")
    plt.imshow(mask_image, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Expanded Mask")
    plt.imshow(expanded_mask, cmap='gray')

    plt.tight_layout()
    plt.show()
