import numpy as np

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
