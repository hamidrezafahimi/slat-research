import cv2
import numpy as np

def apply_threshold(image, threshold_pattern):
    """
    Apply the given threshold pattern to the image.
    
    For each pixel, if the image value is greater than the corresponding
    threshold pattern value, set it to 255; otherwise, set it to 0.
    """
    binary_output = (image > threshold_pattern).astype(np.uint8) * 255
    return binary_output

if __name__ == "__main__":
    # Load the depth image and the threshold pattern in grayscale
    depth_img = cv2.imread("depth_image.jpg", cv2.IMREAD_GRAYSCALE)
    if depth_img is None:
        raise IOError("Could not load 'depth_image.jpg'. Check the file path.")

    threshold_pattern = cv2.imread("threshold_pattern.jpg", cv2.IMREAD_GRAYSCALE)
    if threshold_pattern is None:
        raise IOError("Could not load 'threshold_pattern.jpg'. Check the file path.")

    # If the threshold pattern size does not match the depth image, resize it.
    if threshold_pattern.shape != depth_img.shape:
        threshold_pattern = cv2.resize(threshold_pattern, (depth_img.shape[1], depth_img.shape[0]))

    # Apply the threshold pattern to the depth image
    result = apply_threshold(depth_img, threshold_pattern)

    # Display the result
    cv2.imshow("Thresholded Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
