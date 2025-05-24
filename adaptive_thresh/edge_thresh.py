import cv2
import numpy as np

def apply_threshold(image, threshold_pattern):
    """Apply thresholding to the image using the given threshold pattern."""
    return ((image > threshold_pattern).astype(np.uint8)) * 255

def detect_and_compute_metric(
        masked_image, 
        template_mask, 
        canny_min=0, 
        canny_max=255, 
        match_threshold=100,
        pos_radius_fraction=0.0001
    ):
    """
    1. Performs Canny edge detection on 'masked_image'.
    2. Detects keypoints and descriptors (ORB) in both 'detected_edges' and 'template_mask'.
    3. Matches descriptors with BFMatcher (Hamming distance).
    4. Applies two filters for "good" matches:
       a) Descriptor distance < match_threshold
       b) Keypoint positional distance < (pos_radius_fraction * image_width * image_height)
    5. Computes a metric in [0..1], where a higher value indicates more similarity
       (based on ratio of good matches to the number of keypoints in the template).
    
    Parameters:
        masked_image (np.ndarray): Binary (thresholded) image.
        template_mask (np.ndarray): Binary edge template image.
        canny_min (int)          : Minimum threshold for Canny edge detection.
        canny_max (int)          : Maximum threshold for Canny edge detection.
        match_threshold (float)  : Maximum Hamming distance for a "good" descriptor match.
        pos_radius_fraction (float): Fraction of total pixels used for positional constraint.
                                     e.g., 0.01 => radius = 1% of (width*height).
    
    Returns:
        metric (float)             : A ratio of good matches to template keypoints (0 if no matches).
        detected_edges (np.ndarray): The binary edge image of 'masked_image'.
        kp_detected (list)         : Keypoints in the detected_edges image.
        kp_template (list)         : Keypoints in the template_mask image.
        good_matches (list)        : Filtered "good" matches (based on match_threshold & positional constraint).
    """
    # 1. Perform Canny edge detection.
    detected_edges = cv2.Canny(masked_image, canny_min, canny_max)
    kernel = np.ones((5, 5), np.uint8) 
    detected_edges = cv2.dilate(detected_edges, kernel, iterations=1)

    # 2. Detect keypoints and descriptors in both images using ORB.
    orb = cv2.ORB_create()
    kp_detected, des_detected = orb.detectAndCompute(detected_edges, None)
    kp_template, des_template = orb.detectAndCompute(template_mask, None)

    # If either image has no descriptors, metric = 0.0
    if des_detected is None or des_template is None:
        return 0.0, detected_edges, kp_detected, kp_template, []

    # 3. Match descriptors using BFMatcher with Hamming distance, crossCheck=True
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_detected, des_template)
    
    # Sort matches by distance (ascending = better match).
    matches = sorted(matches, key=lambda x: x.distance)

    # Determine the actual pixel radius from the fraction
    h, w = masked_image.shape[:2]
    pixel_radius = pos_radius_fraction * (w * h)

    # 4. Filter matches:
    #    a) descriptor distance < match_threshold
    #    b) positional distance < pixel_radius
    good_matches = []
    for m in matches:
        if m.distance < match_threshold:
            # Get (x, y) of keypoints in each image
            (x1, y1) = kp_detected[m.queryIdx].pt
            (x2, y2) = kp_template[m.trainIdx].pt
            
            # Euclidean distance between keypoint positions
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            
            if dist < pixel_radius:
                good_matches.append(m)

    # 5. Compute the ratio of good matches to the total number of keypoints in the template.
    #    (metric in [0..1], where 1 = all template keypoints matched well, 0 = no matches).
    if len(good_matches) == 0:
        metric = 10000000
    else:
        metric = 1 / len(good_matches)
    
    return metric, detected_edges, kp_detected, kp_template, good_matches

import sys

def main():
    # Define file paths
    image_path = "depth_image.jpg"

    if len(sys.argv) < 2:
        print("Usage: python script.py threshold_pattern")
        sys.exit(1)

    threshold_path = sys.argv[1]
    edge_template_path = "edge_image.jpg"

    # 1. Load images in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    threshold_pattern = cv2.imread(threshold_path, cv2.IMREAD_GRAYSCALE)
    edge_template = cv2.imread(edge_template_path, cv2.IMREAD_GRAYSCALE)
    
    # Basic checks
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return
    if threshold_pattern is None:
        print(f"Error: Unable to load threshold pattern from {threshold_path}")
        return
    if edge_template is None:
        print(f"Error: Unable to load edge template image from {edge_template_path}")
        return

    # 2. Apply thresholding to get a masked (binary) image
    masked_image = apply_threshold(image, threshold_pattern)

    # 3. Detect edges + compute match-based metric with normalized positional constraint
    metric, detected_edges, kp_detected, kp_template, good_matches = detect_and_compute_metric(
        masked_image, 
        edge_template, 
        canny_min=0, 
        canny_max=255, 
        match_threshold=100,
        pos_radius_fraction=0.0001  # 1% of total pixels as the positional radius
    )
    
    print("Edge Metric (keypoint-based):", metric)

    # --- Visualization ---
    # Convert single-channel images to color for drawing colored keypoints and lines
    detected_edges_color = cv2.cvtColor(detected_edges, cv2.COLOR_GRAY2BGR)
    template_mask_color = cv2.cvtColor(edge_template, cv2.COLOR_GRAY2BGR)

    # Draw matches (keypoints + lines) on a combined image
    matched_image = cv2.drawMatches(
        detected_edges_color, kp_detected,
        template_mask_color, kp_template,
        good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Show the resulting image with keypoints and match lines
    cv2.imshow("Keypoint Matches with Normalized Positional Constraint", matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
