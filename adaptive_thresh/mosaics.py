import cv2
import numpy as np
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../depth_pattern_analysis/gt_generation")
from search_threshs import get_cell_info


def compute_mosaic_array(binary, threshold_fraction=0.01, cell_width_ratio=0.1):
    """
    Given a thresholded (binary) image, build a 2D array of mosaic cells:
      - -2 if the cell is "full" (white fraction > threshold_fraction)
      - -1 if the cell is "empty" (white fraction <= threshold_fraction)

    :param binary: 2D numpy array of shape (H, W), values in {0,255}.
    :param threshold_fraction: e.g. 0.01 => >1% white => 'full'.
    :param cell_width_ratio: fraction of image width used as the cell width.
    :return: mosaic_array of shape (num_cells_y, num_cells_x) with values {-2, -1}.
    """
    height, width = binary.shape
    cell_width, cell_height, num_cells_x, num_cells_y = get_cell_info((height, width), cell_width_ratio)

    # Initialize mosaic array
    mosaic_array = np.zeros((num_cells_y, num_cells_x), dtype=int)

    # Loop over each cell
    for row_cell in range(num_cells_y):
        for col_cell in range(num_cells_x):
            y_start = row_cell * cell_height
            y_end   = y_start + cell_height
            x_start = col_cell * cell_width
            x_end   = x_start + cell_width

            # Extract the sub-region of the binary image
            cell = binary[y_start:y_end, x_start:x_end]

            # Calculate fraction of white pixels
            white_pixels = np.count_nonzero(cell == 255)
            total_pixels = cell_height * cell_width
            fraction_white = white_pixels / total_pixels

            # Assign mosaic array values
            if fraction_white > threshold_fraction:
                # 'Full'
                mosaic_array[row_cell, col_cell] = -2
            else:
                # 'Empty'
                mosaic_array[row_cell, col_cell] = -1

    return mosaic_array


def visualize_mosaic_overlay(img_gray, mosaic_array, cell_width_ratio=0.1, alpha=0.5):
    """
    Create a color overlay showing the mosaic array on top of the original grayscale image.
    'Full' cells are painted red; 'Empty' cells are black/transparent.

    :param img_gray: original grayscale image (H x W).
    :param mosaic_array: 2D array of shape (num_cells_y, num_cells_x), values in {-2, -1}.
    :param cell_width_ratio: same fraction used when creating mosaic_array.
    :param alpha: blending factor for overlay (0=only original, 1=only mosaic).
    :return: overlay image in BGR (same size as img_gray).
    """
    height, width = img_gray.shape
    cell_width, cell_height, num_cells_x, num_cells_y = get_cell_info((height, width), cell_width_ratio)

    # Convert original grayscale to BGR
    original_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    # Mosaic BGR (initially all zeros = black)
    mosaic_bgr = np.zeros_like(original_bgr)

    # Paint each cell according to mosaic_array
    red_color = (0, 0, 255)
    for row_cell in range(num_cells_y):
        for col_cell in range(num_cells_x):
            y_start = row_cell * cell_height
            y_end   = y_start + cell_height
            x_start = col_cell * cell_width
            x_end   = x_start + cell_width

            if mosaic_array[row_cell, col_cell] == -2:
                # "Full" => paint red
                mosaic_bgr[y_start:y_end, x_start:x_end] = red_color
            else:
                # "Empty" => keep black (0,0,0)
                pass

    # Blend the mosaic image with the original
    overlay = cv2.addWeighted(mosaic_bgr, alpha, original_bgr, 1 - alpha, 0)
    return overlay


def main():
    # 1. Read the image in grayscale
    img_gray = cv2.imread('edge_image.jpg', cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError("Could not read the image 'edge_image.jpg'")

    # 2. Threshold to ensure it is strictly black/white
    _, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    # 3. Build the mosaic array
    mosaic_array = compute_mosaic_array(
        binary,
        threshold_fraction=0.01,     # More than 1% white => full
        cell_width_ratio=0.1        # Cell width = 10% of image width
    )

    # 4. Visualize the mosaic overlay
    overlay = visualize_mosaic_overlay(
        img_gray,
        mosaic_array,
        cell_width_ratio=0.1,
        alpha=0.5
    )

    # 5. Save the overlay to disk
    cv2.imwrite('mosaic_overlay.jpg', overlay)
    print("Overlay saved as 'mosaic_overlay.jpg'.")

    # 6. Print or return the mosaic array if desired
    #   -1 => empty, -2 => full
    print("Mosaic array (showing first few rows):\n", mosaic_array[:5])


if __name__ == "__main__":
    main()
