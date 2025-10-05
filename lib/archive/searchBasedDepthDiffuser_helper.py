import cv2
import numpy as np
import csv

def get_cell_info(img_shape, cell_width_ratio):
    height, width = img_shape
    cell_width = max(1, int(round(width * cell_width_ratio)))
    estimated_cells_y = max(1, round(height / cell_width))
    cell_height = int(round(height / estimated_cells_y))
    num_cells_y = height // cell_height
    num_cells_x = width // cell_width
    return cell_width, cell_height, num_cells_x, num_cells_y

def apply_threshold(image, threshold_pattern):
    return np.where(image > threshold_pattern, 255, 0).astype(np.uint8)

def get_perfect_line_segments(bin_img, min_length_ratio=0.1, wr=0.2):
    """
    Detects continuous perfect line segments aligned with internal grid lines,
    splits segments at grid intersections (nodes), and filters short segments.

    Returns: list of sets of (x, y) pixels — one set per segment.
    """
    height, width = bin_img.shape
    min_length = int(round(min_length_ratio * width))

    cell_width, cell_height, num_cells_x, num_cells_y = get_cell_info(bin_img.shape, wr)

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

def expand_zero_from_perfect_segments(bin_img, edge_img, min_length_ratio=0.1, wr=0.2, 
                                      write_csv=False, prefix="output", show=False):
    """
    Full expansion pipeline:
      - Validates binary images
      - Finds perfect segments and expands them using bin_img and edge_img
      - Optionally writes CSVs and shows visualization

    Args:
        bin_img: binary image (0/255)
        edge_img: binary image (0/255)
        min_length_ratio: min segment length relative to image width
        write_csv: if True, saves pixel pairs and full bin_img to CSV
        prefix: output prefix for CSV files
        show: if True, visualizes red (expansion) and blue (line segments)

    Returns:
        A set of (x, y) expansion pixels
    """
    # Validate input
    if not np.isin(bin_img, [0, 255]).all():
        raise ValueError("bin_img must be binary (only 0 and 255 values)")
    if not np.isin(edge_img, [0, 255]).all():
        raise ValueError("edge_img must be binary (only 0 and 255 values)")

    height, width = bin_img.shape
    segments = get_perfect_line_segments(bin_img, min_length_ratio)

    # Optional CSV output
    if write_csv:
        cell_width, cell_height, num_cells_x, num_cells_y = get_cell_info(bin_img.shape, 0.2)
        vertical_lines = [j * cell_width - 1 for j in range(1, num_cells_x)]
        horizontal_lines = [i * cell_height - 1 for i in range(1, num_cells_y)]

        pixel_pairs = []
        for x in vertical_lines:
            if x + 1 >= width: continue
            for y in range(height):
                pixel_pairs.append([x, y, x + 1, y, int(bin_img[y, x]), int(bin_img[y, x + 1])])
        for y in horizontal_lines:
            if y + 1 >= height: continue
            for x in range(width):
                pixel_pairs.append([x, y, x, y + 1, int(bin_img[y, x]), int(bin_img[y + 1, x])])

        with open(f"{prefix}_pixel_pairs.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x1", "y1", "x2", "y2", "val1", "val2"])
            writer.writerows(pixel_pairs)

        np.savetxt(f"{prefix}_image_data.csv", bin_img, fmt="%d", delimiter=",")

    # Expansion
    all_expansion = set()
    for seg in segments:
        expanded = expand_line_seg(seg, bin_img, edge_img)
        all_expansion.update(expanded)

    # Optional display
    if show:
        vis = cv2.cvtColor(bin_img.copy(), cv2.COLOR_GRAY2BGR)

        # Draw red expansion pixels first
        for x, y in all_expansion:
            if 0 <= x < width and 0 <= y < height:
                vis[y, x] = (0, 0, 255)

        # Draw blue segment pixels on top
        for seg in segments:
            for x, y in seg:
                if 0 <= x < width and 0 <= y < height:
                    vis[y, x] = (255, 0, 0)

        cv2.imshow("Expansion (red) and Line Segments (blue)", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return all_expansion