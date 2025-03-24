import cv2
import numpy as np

def adjust_threshold_pattern(base_pattern, translation, slope):
    """
    Adjust the base threshold pattern using translation and a slope (in radians).
    
    For each row r (0-based) in an image of height H:
      adjusted[r, :] = base_pattern[r, :] + translation + (r - center_row) * tan(slope)
    where center_row = (H - 1) / 2.
    
    Parameters:
      base_pattern : 2D numpy array (uint8) representing the loaded threshold pattern.
      translation  : Constant offset (can be negative or positive).
      slope        : Slope angle in radians.
    
    Returns:
      adjusted_pattern: The modified threshold pattern, clipped to [0, 255] (uint8).
    """
    H, W = base_pattern.shape
    center_row = (H - 1) / 2.0
    # Create a row index offset (distance from vertical center)
    row_offsets = np.arange(H, dtype=np.float32) - center_row
    # Compute row-dependent offset using tangent of the slope angle.
    row_offsets = np.tan(slope) * row_offsets
    # Apply translation and row-dependent offset.
    adjusted = base_pattern.astype(np.float32) + translation + row_offsets[:, np.newaxis]
    # Clip to valid grayscale range and convert back to uint8.
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return adjusted

def apply_threshold(image, threshold_pattern):
    """
    Apply the given threshold pattern to the image.
    
    For each pixel, if the image value is greater than the corresponding threshold,
    output 255; otherwise, 0.
    """
    binary_output = (image > threshold_pattern).astype(np.uint8) * 255
    return binary_output

def on_trackbar(val):
    global base_threshold_pattern, depth_img
    
    # Get trackbar positions.
    # Translation: range 0 to 510 with center at 255 corresponding to 0 offset.
    trans_trackbar = cv2.getTrackbarPos("Translation", "Adaptive Threshold")
    translation = trans_trackbar - 255  # negative below center, positive above
    
    # Slope: range 0 to 200 with center at 100 corresponding to 0 radians.
    slope_trackbar = cv2.getTrackbarPos("Slope", "Adaptive Threshold")
    max_angle = 1.57  # Maximum absolute angle in radians (approx. 90°)
    slope = ((slope_trackbar - 100) / 100.0) * max_angle  # Map trackbar value to radians
    
    # Adjust the base threshold pattern.
    adjusted_pattern = adjust_threshold_pattern(base_threshold_pattern, translation, slope)
    
    # Apply thresholding using the adjusted pattern.
    result = apply_threshold(depth_img, adjusted_pattern)
    
    # Display the thresholded image and the adjusted threshold pattern.
    cv2.imshow("Adaptive Threshold", result)
    cv2.imshow("Threshold Pattern", adjusted_pattern)

if __name__ == "__main__":
    # Load the depth image in grayscale.
    depth_img = cv2.imread("depth_image.jpg", cv2.IMREAD_GRAYSCALE)
    if depth_img is None:
        raise IOError("Could not load 'depth_image.jpg'. Check the file path.")
    
    # Load any arbitrary threshold pattern (monochrome image) in unchanged mode.
    base_threshold_pattern = cv2.imread("Untitled.jpeg", cv2.IMREAD_UNCHANGED)
    if base_threshold_pattern is None:
        raise IOError("Could not load 'threshold_pattern.jpg'. This file is mandatory.")
    
    # If the loaded pattern is not single channel, convert it to grayscale.
    if len(base_threshold_pattern.shape) == 3:
        base_threshold_pattern = cv2.cvtColor(base_threshold_pattern, cv2.COLOR_BGR2GRAY)
    
    # Resize the threshold pattern if necessary to match the depth image size.
    if base_threshold_pattern.shape != depth_img.shape:
        base_threshold_pattern = cv2.resize(base_threshold_pattern, (depth_img.shape[1], depth_img.shape[0]))
    
    # Create display windows.
    cv2.namedWindow("Adaptive Threshold", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Threshold Pattern", cv2.WINDOW_NORMAL)
    
    # Create trackbars for translation and slope.
    # Translation trackbar: range from 0 to 510, with 255 as zero offset.
    cv2.createTrackbar("Translation", "Adaptive Threshold", 255, 510, on_trackbar)
    # Slope trackbar: range from 0 to 200, with 100 as zero radians.
    cv2.createTrackbar("Slope", "Adaptive Threshold", 100, 200, on_trackbar)
    
    # Perform the initial thresholding.
    on_trackbar(0)
    
    print("Loaded arbitrary threshold pattern. Adjust the trackbars to translate and rotate it.")
    print("Press 'q' or 'Esc' to quit.")
    
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            break
    
    cv2.destroyAllWindows()
