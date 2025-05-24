import cv2
import numpy as np
import glob
import os
import sys
import math

def get_cell_info(img_shape, cell_width_ratio):
    height, width = img_shape
    cell_width = max(1, int(round(width * cell_width_ratio)))
    # Estimate number of vertical cells so that cell_height ~ cell_width
    estimated_cells_y = max(1, round(height / cell_width))
    cell_height = int(round(height / estimated_cells_y))
    # Calculate number of cells to fully cover the image
    num_cells_y = height // cell_height
    num_cells_x = width // cell_width
    return cell_width, cell_height, num_cells_x, num_cells_y

def line_intersection(line1, line2):
    """
    Compute the intersection of two lines (each defined as ((x1, y1), (x2, y2))).
    Returns (x, y) or None if the lines are parallel.
    """
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1

    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3

    determinant = A1 * B2 - A2 * B1
    if abs(determinant) < 1e-10:
        return None
    x = (B2 * C1 - B1 * C2) / determinant
    y = (A1 * C2 - A2 * C1) / determinant
    return (x, y)

def angle_between_lines(line1, line2):
    """
    Computes the acute angle (in degrees) between two line segments.
    """
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2
    v1 = (x2 - x1, y2 - y1)
    v2 = (x4 - x3, y4 - y3)
    norm1 = math.hypot(*v1)
    norm2 = math.hypot(*v2)
    if norm1 == 0 or norm2 == 0:
        return None
    # Normalize vectors
    v1 = (v1[0] / norm1, v1[1] / norm1)
    v2 = (v2[0] / norm2, v2[1] / norm2)
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    dot = max(min(dot, 1.0), -1.0)
    angle_rad = math.acos(dot)
    angle_deg = math.degrees(angle_rad)
    if angle_deg > 90:
        angle_deg = 180 - angle_deg
    return angle_deg

def detect_grid_corner_at_node(image, node_x, node_y, window_size=20, angle_tol=5, endpoint_thresh=5):
    """
    Extract a window centered at the grid node and check for a 90° corner.
    Steps:
      1. Extract a small window around the grid node.
      2. Run Canny edge detection and detect line segments using HoughLinesP.
      3. For each pair of line segments, compute their intersection (in window coordinates)
         and the angle between them.
      4. If an intersection is near the window center and forms a ~90° angle, return True.
    """
    half = window_size // 2
    h, w = image.shape[:2]
    # Ensure the window remains within image bounds
    x1 = max(0, node_x - half)
    y1 = max(0, node_y - half)
    x2 = min(w, node_x + half)
    y2 = min(h, node_y + half)
    window = image[y1:y2, x1:x2]
    
    gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=10, maxLineGap=5)
    if lines is None:
        return False

    center_x = window.shape[1] // 2
    center_y = window.shape[0] // 2
    
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            line1 = ((lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]))
            line2 = ((lines[j][0][0], lines[j][0][1]), (lines[j][0][2], lines[j][0][3]))
            inter = line_intersection(line1, line2)
            if inter is None:
                continue
            ix, iy = inter
            # Check if the intersection is near the window center
            if abs(ix - center_x) > endpoint_thresh or abs(iy - center_y) > endpoint_thresh:
                continue
            angle = angle_between_lines(line1, line2)
            if angle is None:
                continue
            if abs(angle - 90) <= angle_tol:
                return True
    return False

def process_image(image, cell_width_ratio=0.2, window_size=20, angle_tol=5, endpoint_thresh=5, grid_alpha=0.4):
    """
    Defines a grid on the image using get_cell_info, then iterates over grid nodes
    (excluding nodes on the image margins) to check for a perfect 90° corner.
    If a corner is detected at a node, a red dot is drawn.
    Additionally, a semi-transparent grid is drawn over the image.
    """
    height, width = image.shape[:2]
    cell_width, cell_height, num_cells_x, num_cells_y = get_cell_info((height, width), cell_width_ratio)
    
    # Annotate grid nodes that detect a perfect 90° corner
    for i in range(0, num_cells_x + 1):
        for j in range(0, num_cells_y + 1):
            node_x = i * cell_width
            node_y = j * cell_height
            # Skip nodes that fall on the very margins
            if node_x <= 0 or node_y <= 0 or node_x >= width - 1 or node_y >= height - 1:
                continue
            if detect_grid_corner_at_node(image, node_x, node_y, window_size, angle_tol, endpoint_thresh):
                cv2.circle(image, (node_x, node_y), radius=5, color=(0, 0, 255), thickness=-1)
    
    # Create an empty overlay for grid lines
    grid_overlay = np.zeros_like(image, dtype=np.uint8)
    
    # Draw vertical grid lines
    for i in range(0, num_cells_x + 1):
        x = i * cell_width
        cv2.line(grid_overlay, (x, 0), (x, height-1), color=(0, 255, 0), thickness=1)
    # Draw horizontal grid lines
    for j in range(0, num_cells_y + 1):
        y = j * cell_height
        cv2.line(grid_overlay, (0, y), (width-1, y), color=(0, 255, 0), thickness=1)
    
    # Blend the grid overlay with the annotated image using specified transparency (grid_alpha)
    annotated = cv2.addWeighted(image, 1.0, grid_overlay, grid_alpha, 0)
    return annotated

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <images_directory>")
        sys.exit(1)
    
    images_dir = sys.argv[1]
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
    
    if not image_paths:
        print("No images found in the given directory.")
        sys.exit(1)
    
    for img_path in image_paths:
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load image: {img_path}")
            continue
        
        annotated = process_image(image.copy(), cell_width_ratio=0.2, window_size=20, angle_tol=5, endpoint_thresh=5, grid_alpha=0.4)
        cv2.imshow("Grid Corner Detection", annotated)
        print(f"Showing {img_path}. Press any key to continue (ESC to exit).")
        key = cv2.waitKey(0)
        if key == 27:  # ESC key
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
