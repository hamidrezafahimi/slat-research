import cv2
import numpy as np

def compute_threshold_pattern(image_shape, bottom_threshold, slope):
    """
    Compute a 2D threshold pattern image.
    
    Parameters:
      image_shape      : Tuple (H, W) of the input image.
      bottom_threshold : The threshold value at the bottom row.
      slope            : The difference between the bottom and top threshold.
      
    Returns:
      threshold_pattern: 2D numpy array (uint8) where each row contains the computed threshold repeated across all columns.
    """
    H, W = image_shape
    row_indices = np.arange(H)
    # For top row (r=0): threshold = bottom_threshold - slope; for bottom row (r=H-1): threshold = bottom_threshold.
    row_thresholds = bottom_threshold - slope * ((H - 1 - row_indices) / (H - 1))
    row_thresholds = np.clip(row_thresholds, 0, 255).astype(np.uint8)
    threshold_pattern = np.tile(row_thresholds[:, np.newaxis], (1, W))
    return threshold_pattern

def apply_threshold(image, threshold_pattern):
    """
    Apply the threshold pattern to the image.
    
    For each pixel, if the image value > corresponding threshold, set to 255, else 0.
    """
    binary_output = (image > threshold_pattern).astype(np.uint8) * 255
    return binary_output

def on_trackbar(val):
    # Get current trackbar positions
    bottom_threshold = cv2.getTrackbarPos("BottomThreshold", "Adaptive Threshold")
    slope = cv2.getTrackbarPos("Slope", "Adaptive Threshold")
    
    # Compute threshold pattern and apply thresholding
    threshold_pattern = compute_threshold_pattern(img.shape, bottom_threshold, slope)
    result = apply_threshold(img, threshold_pattern)
    
    # Display the results
    cv2.imshow("Adaptive Threshold", result)
    cv2.imshow("Threshold Pattern", threshold_pattern)

if __name__ == "__main__":
    # Load the grayscale image (adjust the path as needed)
    img = cv2.imread("depth_image.jpg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError("Could not load image. Check the file path.")
    
    # Create windows
    cv2.namedWindow("Adaptive Threshold", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Threshold Pattern", cv2.WINDOW_NORMAL)
    
    # Create trackbars for the two parameters
    cv2.createTrackbar("BottomThreshold", "Adaptive Threshold", 255, 255, on_trackbar)
    cv2.createTrackbar("Slope", "Adaptive Threshold", 50, 255, on_trackbar)
    
    # Initial update
    on_trackbar(0)
    
    # Main loop: check for key press events
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == ord('s'):
            # When 's' is pressed, compute and save the current threshold pattern
            bottom_threshold = cv2.getTrackbarPos("BottomThreshold", "Adaptive Threshold")
            slope = cv2.getTrackbarPos("Slope", "Adaptive Threshold")
            threshold_pattern = compute_threshold_pattern(img.shape, bottom_threshold, slope)
            cv2.imwrite("threshold_pattern.jpg", threshold_pattern)
            print("Saved threshold pattern to 'threshold_pattern.jpg'")
        elif key == ord('q') or key == 27:
            # Quit on 'q' or Esc key
            break
    
    cv2.destroyAllWindows()
