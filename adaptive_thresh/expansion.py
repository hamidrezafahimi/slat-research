import numpy as np

def get_cell_info(img_shape, cell_width_ratio):
    """
    Provided helper. Returns (cell_width, cell_height, num_cells_x, num_cells_y)
    given the shape and a ratio.
    """
    height, width = img_shape
    cell_width = max(1, int(round(width * cell_width_ratio)))
    estimated_cells_y = max(1, round(height / cell_width))
    cell_height = int(round(height / estimated_cells_y))
    num_cells_y = height // cell_height
    num_cells_x = width // cell_width
    return cell_width, cell_height, num_cells_x, num_cells_y

def find_perfect_line_segments_horizontal(bin_img, r1, r2):
    """
    Looks at two adjacent rows r1, r2 in bin_img.
    Returns a list of (col_start, col_end) segments where
    bin_img[r1, c] == 255, bin_img[r2, c] == 0 OR vice versa,
    continuously across columns.
    """
    height, width = bin_img.shape
    if not (0 <= r1 < height and 0 <= r2 < height):
        return []

    row1 = bin_img[r1, :]
    row2 = bin_img[r2, :]

    segments = []
    in_segment = False
    seg_start = -1

    for c in range(width):
        # “Perfect” if the two adjacent pixels differ exactly as 255/0
        is_perfect = ((row1[c] == 255 and row2[c] == 0) or
                      (row1[c] == 0   and row2[c] == 255))
        if is_perfect and not in_segment:
            seg_start = c
            in_segment = True
        elif not is_perfect and in_segment:
            segments.append((seg_start, c - 1))
            in_segment = False

    # Close any open segment at the last column
    if in_segment:
        segments.append((seg_start, width - 1))

    return segments

def find_perfect_line_segments_vertical(bin_img, c1, c2):
    """
    Looks at two adjacent columns c1, c2 in bin_img.
    Returns a list of (row_start, row_end) segments where
    bin_img[r, c1] == 255, bin_img[r, c2] == 0 OR vice versa,
    continuously across rows.
    """
    height, width = bin_img.shape
    if not (0 <= c1 < width and 0 <= c2 < width):
        return []

    col1 = bin_img[:, c1]
    col2 = bin_img[:, c2]

    segments = []
    in_segment = False
    seg_start = -1

    for r in range(height):
        # “Perfect” if the two adjacent pixels differ exactly as 255/0
        is_perfect = ((col1[r] == 255 and col2[r] == 0) or
                      (col1[r] == 0   and col2[r] == 255))
        if is_perfect and not in_segment:
            seg_start = r
            in_segment = True
        elif not is_perfect and in_segment:
            segments.append((seg_start, r - 1))
            in_segment = False

    if in_segment:
        segments.append((seg_start, height - 1))

    return segments

def expand_horizontal_line(bin_img, edge_img, c_start, c_end, black_row, white_row):
    """
    Given a horizontal perfect-line segment from col c_start..c_end (inclusive),
    where `black_row` is the side with 0 in bin_img and `white_row` is the side
    with 255 in bin_img, expand “downward” or “upward” from the white_row, adding
    points to the expansion set until hitting a black pixel in bin_img or a
    white pixel in edge_img (edge_img[r,c] == 255).
    """
    expansions = []
    height, width = bin_img.shape
    # Determine expansion direction
    step = 1 if white_row > black_row else -1
    r = white_row
    while 0 <= r < height:
        # Check every column in [c_start..c_end]
        for c in range(c_start, c_end + 1):
            # Stop if we see black in bin_img or white (255) in edge_img
            if bin_img[r, c] == 0 or edge_img[r, c] == 255:
                return expansions
        # If safe, add all these pixels to expansions
        for c in range(c_start, c_end + 1):
            expansions.append((r, c))
        r += step

    return expansions

def expand_vertical_line(bin_img, edge_img, r_start, r_end, black_col, white_col):
    """
    Given a vertical perfect-line segment from row r_start..r_end (inclusive),
    where `black_col` is the side with 0 in bin_img and `white_col` is the side
    with 255 in bin_img, expand “rightward” or “leftward” from the white_col,
    adding points to the expansion set until hitting a black pixel in bin_img or
    a white pixel in edge_img (edge_img[r,c] == 255).
    """
    expansions = []
    height, width = bin_img.shape
    step = 1 if white_col > black_col else -1
    c = white_col
    while 0 <= c < width:
        # Check every row in [r_start..r_end]
        for r in range(r_start, r_end + 1):
            if bin_img[r, c] == 0 or edge_img[r, c] == 255:
                return expansions
        # If safe, add all these pixels
        for r in range(r_start, r_end + 1):
            expansions.append((r, c))
        c += step

    return expansions

def expand_black_area(bin_img, edge_img):
    """
    Main routine:
      - Use get_cell_info with (bin_img.shape, 0.2) to find the grid.
      - Locate 'perfect lines' near the horizontal/vertical grid lines.
      - For each perfect line, expand the black area into the white area.
      - Return a set of all expansion coordinates.
    """
    # Get grid info
    cell_width, cell_height, num_cells_x, num_cells_y = get_cell_info(bin_img.shape, 0.2)
    height, width = bin_img.shape
    bin_img[bin_img < 25] = 0
    bin_img[bin_img >= 25] = 255

    expansion_set = set()

    # --- Horizontal lines: possible row-coordinates for the grid ---
    horizontal_candidates = [i * cell_height - 1 for i in range(1, num_cells_y)]
    for r in horizontal_candidates:
        # Check pairs: (r, r+1) and (r-1, r) in-bounds
        # 1) (r, r+1)
        segs = find_perfect_line_segments_horizontal(bin_img, r, r+1)
        for (c_start, c_end) in segs:
            # Figure out which row is black vs white at c_start
            if bin_img[r, c_start] == 0:
                # r is black, r+1 is white
                new_pixels = expand_horizontal_line(bin_img, edge_img,
                                                    c_start, c_end,
                                                    black_row=r,
                                                    white_row=r+1)
            else:
                # r is white, r+1 is black
                new_pixels = expand_horizontal_line(bin_img, edge_img,
                                                    c_start, c_end,
                                                    black_row=r+1,
                                                    white_row=r)
            expansion_set.update(new_pixels)

        # 2) (r-1, r)
        segs = find_perfect_line_segments_horizontal(bin_img, r-1, r)
        for (c_start, c_end) in segs:
            if bin_img[r-1, c_start] == 0:
                new_pixels = expand_horizontal_line(bin_img, edge_img,
                                                    c_start, c_end,
                                                    black_row=r-1,
                                                    white_row=r)
            else:
                new_pixels = expand_horizontal_line(bin_img, edge_img,
                                                    c_start, c_end,
                                                    black_row=r,
                                                    white_row=r-1)
            expansion_set.update(new_pixels)

    # --- Vertical lines: possible column-coordinates for the grid ---
    vertical_candidates = [j * cell_width - 1 for j in range(1, num_cells_x)]
    for c in vertical_candidates:
        # Check pairs: (c, c+1) and (c-1, c)
        # 1) (c, c+1)
        segs = find_perfect_line_segments_vertical(bin_img, c, c+1)
        for (r_start, r_end) in segs:
            if bin_img[r_start, c] == 0:
                # col c is black, col c+1 is white
                new_pixels = expand_vertical_line(bin_img, edge_img,
                                                  r_start, r_end,
                                                  black_col=c,
                                                  white_col=c+1)
            else:
                # col c is white, col c+1 is black
                new_pixels = expand_vertical_line(bin_img, edge_img,
                                                  r_start, r_end,
                                                  black_col=c+1,
                                                  white_col=c)
            expansion_set.update(new_pixels)

        # 2) (c-1, c)
        segs = find_perfect_line_segments_vertical(bin_img, c-1, c)
        for (r_start, r_end) in segs:
            if bin_img[r_start, c-1] == 0:
                new_pixels = expand_vertical_line(bin_img, edge_img,
                                                  r_start, r_end,
                                                  black_col=c-1,
                                                  white_col=c)
            else:
                new_pixels = expand_vertical_line(bin_img, edge_img,
                                                  r_start, r_end,
                                                  black_col=c,
                                                  white_col=c-1)
            expansion_set.update(new_pixels)

    return expansion_set


import cv2
import numpy as np

def scan_grid_for_perfect_lines(img, cell_width_ratio=0.2):
    """
    Scans along grid lines from get_cell_info.
    For each adjacent pixel pair on the grid (one 0, one 255), marks them in red and shows the image.
    """
    cell_width, cell_height, num_cells_x, num_cells_y = get_cell_info(img.shape, cell_width_ratio)
    height, width = img.shape

    # Prepare color image for visualization
    if len(img.shape) == 2:
        vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis_img = img.copy()

    # Track red-marked pixels
    red_pixels = []

    # Horizontal checks
    for i in range(num_cells_y + 1):
        r = i * cell_height
        if r + 1 >= height:
            continue
        for c in range(width):
            p1 = img[r, c]
            p2 = img[r + 1, c]
            if (p1 == 255 and p2 == 0) or (p1 == 0 and p2 == 255):
                print(f"found perfect line at row pair ({r}, {r+1}) col {c}")
                vis_img[r, c] = (0, 0, 255)
                vis_img[r + 1, c] = (0, 0, 255)

    # Vertical checks
    for j in range(num_cells_x + 1):
        c = j * cell_width
        if c + 1 >= width:
            continue
        for r in range(height):
            p1 = img[r, c]
            p2 = img[r, c + 1]
            if (p1 == 255 and p2 == 0) or (p1 == 0 and p2 == 255):
                print(f"found perfect line at col pair ({c}, {c+1}) row {r}")
                vis_img[r, c] = (0, 0, 255)
                vis_img[r, c + 1] = (0, 0, 255)

    # Show result
    cv2.imshow("Perfect Line Pixels (in Red)", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


import sys
import cv2
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

    currentTMask[currentTMask < 25] = 0
    currentTMask[currentTMask >= 25] = 255
    subt_image[subt_image < 25] = 0
    subt_image[subt_image >= 25] = 255

    sett = expand_black_area(subt_image, currentTMask)
    # scan_grid_for_perfect_lines(subt_image)
    # Ensure subt_image is in color
    if len(subt_image.shape) == 2:
        subt_color = cv2.cvtColor(subt_image, cv2.COLOR_GRAY2BGR)
    else:
        subt_color = subt_image.copy()

    # Draw red points on top of the image
    for x, y in sett:
        if 0 <= y < subt_color.shape[0] and 0 <= x < subt_color.shape[1]:
            subt_color[y, x] = (0, 0, 255)  # BGR: Red

    # Draw yellow grid lines
    cell_width, cell_height, num_cells_x, num_cells_y = get_cell_info(subt_image.shape, 0.2)
    height, width = subt_image.shape

    # Draw horizontal grid lines
    for i in range(num_cells_y + 1):
        y = i * cell_height
        cv2.line(subt_color, (0, y), (width - 1, y), (0, 255, 255), 1)  # BGR: Yellow

    # Draw vertical grid lines
    for j in range(num_cells_x + 1):
        x = j * cell_width
        cv2.line(subt_color, (x, 0), (x, height - 1), (0, 255, 255), 1)  # BGR: Yellow


    cv2.imshow("Points on Subt Image", subt_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
