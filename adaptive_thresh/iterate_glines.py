import sys
import cv2
import numpy as np
import random
from expansion import get_cell_info


def get_perfect_line_segments(bin_img, min_length_ratio=1/11):
    """
    Detects continuous perfect line segments aligned with internal grid lines,
    splits segments at grid intersections (nodes), and filters short segments.

    Returns: list of sets of (x, y) pixels — one set per segment.
    """
    height, width = bin_img.shape
    min_length = int(round(min_length_ratio * width))

    cell_width, cell_height, num_cells_x, num_cells_y = get_cell_info(bin_img.shape, 0.2)

    vertical_lines = [j * cell_width - 1 for j in range(1, num_cells_x)]
    horizontal_lines = [i * cell_height - 1 for i in range(1, num_cells_y)]
    vertical_set = set(vertical_lines)
    horizontal_set = set(horizontal_lines)

    segments = []

    # --- Vertical perfect segments (along horizontal grid lines) ---
    for y in horizontal_lines:
        if y + 1 >= height:
            continue
        seg = set()
        for x in range(width):
            val1 = bin_img[y, x]
            val2 = bin_img[y + 1, x]
            is_perfect = (val1 == 255 and val2 == 0) or (val1 == 0 and val2 == 255)

            is_node = (x in vertical_set and y in horizontal_set)

            if is_perfect:
                seg.add((x, y))
                seg.add((x, y + 1))
                if is_node and len(seg) >= min_length:
                    segments.append(seg)
                    seg = set()
            else:
                if len(seg) >= min_length:
                    segments.append(seg)
                seg = set()
        if len(seg) >= min_length:
            segments.append(seg)

    # --- Horizontal perfect segments (along vertical grid lines) ---
    for x in vertical_lines:
        if x + 1 >= width:
            continue
        seg = set()
        for y in range(height):
            val1 = bin_img[y, x]
            val2 = bin_img[y, x + 1]
            is_perfect = (val1 == 255 and val2 == 0) or (val1 == 0 and val2 == 255)

            is_node = (x in vertical_set and y in horizontal_set)

            if is_perfect:
                seg.add((x, y))
                seg.add((x + 1, y))
                if is_node and len(seg) >= min_length:
                    segments.append(seg)
                    seg = set()
            else:
                if len(seg) >= min_length:
                    segments.append(seg)
                seg = set()
        if len(seg) >= min_length:
            segments.append(seg)

    return segments

def expand_line_seg(seg, bin_img, edge_img):
    """
    Expands a perfect line segment into the white side, stopping at:
      - a black pixel (0) in bin_img  → expansion_cut_condition_1
      - a white pixel (255) in edge_img → expansion_cut_condition_2

    The segment itself is included in the returned expansion.

    Args:
        seg: set of (x, y) points
        bin_img: binary image (0 or 255)
        edge_img: another binary image for stopping condition

    Returns:
        A set of (x, y) expansion pixels (including the original segment)
    """
    if not seg:
        return set()

    height, width = bin_img.shape
    seg = list(seg)

    xs, ys = zip(*seg)
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)

    # ⬅️ include the original segment in expansion
    expansion_set = set(seg)

    if dx >= dy:
        # Horizontal segment → expand vertically
        col_to_rows = {}
        for x, y in seg:
            col_to_rows.setdefault(x, []).append(y)

        for x in col_to_rows:
            y = sorted(col_to_rows[x])[0]

            # Decide direction
            if y > 0 and bin_img[y - 1, x] == 255:
                direction = -1  # up
            elif y + 1 < height and bin_img[y + 1, x] == 255:
                direction = 1   # down
            else:
                continue

            r = y + direction
            while 0 <= r < height:
                if bin_img[r, x] == 0 or edge_img[r, x] == 255:
                    break
                expansion_set.add((x, r))
                r += direction

    else:
        # Vertical segment → expand horizontally
        row_to_cols = {}
        for x, y in seg:
            row_to_cols.setdefault(y, []).append(x)

        for y in row_to_cols:
            x = sorted(row_to_cols[y])[0]

            # Decide direction
            if x > 0 and bin_img[y, x - 1] == 255:
                direction = -1  # left
            elif x + 1 < width and bin_img[y, x + 1] == 255:
                direction = 1   # right
            else:
                continue

            c = x + direction
            while 0 <= c < width:
                if bin_img[y, c] == 0 or edge_img[y, c] == 255:
                    break
                expansion_set.add((c, y))
                c += direction

    return expansion_set


def expand_all_from_perfect_segments(bin_img, edge_img, min_length_ratio=1/11):
    """
    Complete expansion pipeline:
      - Finds perfect line segments in bin_img (grid-aligned, filtered)
      - Expands each segment based on bin_img and edge_img
      - Returns a set of all expansion pixels (including original segments)

    Args:
        bin_img: binary image (0/255) where expansion starts
        edge_img: binary image (0/255) that blocks expansion (cut_condition_2)
        min_length_ratio: min segment length relative to image width

    Returns:
        Set of (x, y) expansion pixels
    """
    # Step 1: Detect perfect segments
    segments = get_perfect_line_segments(bin_img, min_length_ratio)

    # Step 2: Expand each segment and combine results
    all_expansion = set()
    for seg in segments:
        expanded = expand_line_seg(seg, bin_img, edge_img)
        all_expansion.update(expanded)

    return all_expansion



import os
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python script.py subt.jpg edge.jpg")
        sys.exit(1)

    subt_path = sys.argv[1]
    edge_path = sys.argv[2]

    # Load images in grayscale.
    subt_image = cv2.imread(subt_path, cv2.IMREAD_GRAYSCALE)
    if subt_image is None:
        raise IOError(f"Could not load {subt_path}. Check the file path.")

    currentTMask = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
    if currentTMask is None:
        raise IOError(f"Could not load {edge_path}. Check the file path.")

    if currentTMask.shape != subt_image.shape:
        raise ValueError("Both input images must have the same dimensions.")

    output_dir = "masked_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Binarize image using threshold 25
    subt_image[subt_image < 25] = 0
    subt_image[subt_image >= 25] = 255
    currentTMask[currentTMask < 25] = 0
    currentTMask[currentTMask >= 25] = 255
    # Detect perfect line segments
    expanded_pixels = expand_all_from_perfect_segments(subt_image, currentTMask)

    # Visualize
    vis = cv2.cvtColor(subt_image, cv2.COLOR_GRAY2BGR)
    for x, y in expanded_pixels:
        vis[y, x] = (0, 0, 255)
    cv2.imshow("Expanded Pixels", vis)
    cv2.waitKey(0)
