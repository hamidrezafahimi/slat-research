import cv2
import numpy as np

# Global variables to hold the loaded threshold pattern and a flag to use it
loaded_threshold = None
use_loaded = False

def compute_threshold_pattern(image_shape, bottom_threshold, slope):
    """
    Compute a 2D threshold pattern image.
    
    For each row r in an image of height H:
      T(r) = bottom_threshold - slope * ((H - 1 - r) / (H - 1))
    
    Parameters:
      image_shape      : Tuple (H, W) of the input image.
      bottom_threshold : Threshold at the bottom row.
      slope            : Difference between the bottom and top row thresholds.
      
    Returns:
      threshold_pattern: 2D numpy array (uint8) of shape (H, W) where each row
                         contains the computed threshold repeated across all columns.
    """
    H, W = image_shape
    row_indices = np.arange(H)
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
    global loaded_threshold, use_loaded
    bottom_threshold = cv2.getTrackbarPos("BottomThreshold", "Adaptive Threshold")
    slope = cv2.getTrackbarPos("Slope", "Adaptive Threshold")
    
    # Use the loaded pattern if flagged; otherwise compute the pattern from trackbar values
    if use_loaded and loaded_threshold is not None:
        threshold_pattern = loaded_threshold
    else:
        threshold_pattern = compute_threshold_pattern(img.shape, bottom_threshold, slope)
    
    result = apply_threshold(img, threshold_pattern)
    cv2.imshow("Adaptive Threshold", result)
    cv2.imshow("Threshold Pattern", threshold_pattern)

if __name__ == "__main__":
    # Load the grayscale image (adjust the filename as needed)
    img = cv2.imread("depth_image.jpg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError("Could not load image. Check the file path.")
    
    # Create display windows
    cv2.namedWindow("Adaptive Threshold", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Threshold Pattern", cv2.WINDOW_NORMAL)
    
    # Create trackbars for the two parameters
    cv2.createTrackbar("BottomThreshold", "Adaptive Threshold", 255, 255, on_trackbar)
    cv2.createTrackbar("Slope", "Adaptive Threshold", 50, 255, on_trackbar)
    
    # Initial update (using computed threshold pattern)
    on_trackbar(0)
    
    print("Press 's' to save the computed threshold pattern to 'threshold_pattern.jpg'.")
    print("Press 'l' to load the threshold pattern from file and use it for thresholding.")
    print("Press 'c' to clear the loaded pattern and revert to computed pattern.")
    print("Press 'q' or 'Esc' to quit.")
    
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == ord('s'):
            # Save the currently computed threshold pattern
            bottom_threshold = cv2.getTrackbarPos("BottomThreshold", "Adaptive Threshold")
            slope = cv2.getTrackbarPos("Slope", "Adaptive Threshold")
            pattern_to_save = compute_threshold_pattern(img.shape, bottom_threshold, slope)
            cv2.imwrite("threshold_pattern.jpg", pattern_to_save)
            print("Saved threshold pattern to 'threshold_pattern.jpg'")
        elif key == ord('l'):
            # Load threshold pattern from file
            loaded = cv2.imread("threshold_pattern.jpg", cv2.IMREAD_GRAYSCALE)
            if loaded is None:
                print("Failed to load threshold pattern from 'threshold_pattern.jpg'")
            else:
                # Resize if necessary to match the image size
                if loaded.shape != img.shape:
                    loaded = cv2.resize(loaded, (img.shape[1], img.shape[0]))
                loaded_threshold = loaded
                use_loaded = True
                print("Loaded threshold pattern from file and now using it.")
                on_trackbar(0)
        elif key == ord('c'):
            # Clear the loaded threshold pattern and revert to computed threshold
            use_loaded = False
            loaded_threshold = None
            print("Cleared loaded threshold pattern. Using computed threshold pattern.")
            on_trackbar(0)
        elif key == ord('q') or key == 27:
            break
    
    cv2.destroyAllWindows()
