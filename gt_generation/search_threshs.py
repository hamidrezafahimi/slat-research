import cv2
import numpy as np
import os
import csv
import time
from scipy.interpolate import LSQBivariateSpline
from scipy.spatial import Delaunay
import cv2
import numpy as np
from scipy.interpolate import LSQBivariateSpline
from scipy.spatial import ConvexHull

import cv2
import numpy as np
from scipy.interpolate import LSQBivariateSpline
from scipy.spatial import ConvexHull  # for hull boundary checks

class VisualSplineFit:
    def __init__(self, x_coeffs=4, degree_x=3, degree_y=3):
        """
        Same constructor signature as before.
        """
        self.degree_x = degree_x
        self.degree_y = degree_y
        self.num_coeffs_x = x_coeffs
        self.params_init = False

    def initParams(self, img_shape):
        """
        Prepare (x, y) grids and knot vectors for the spline.
        """
        self.H, self.W = img_shape

        # Rough guess for how many coeffs in y direction
        self.num_coeffs_y = int(np.round((float(self.H) / self.W) * self.num_coeffs_x))

        # x, y in [0..1] for each pixel
        X, Y = np.meshgrid(
            np.linspace(0, 1, self.W),  # shape => (H, W)
            np.linspace(0, 1, self.H)
        )
        self.x_flat = X.ravel()  # shape => (H*W,)
        self.y_flat = Y.ravel()  # shape => (H*W,)

        # Prepare knots
        self.num_knots_x = self.num_coeffs_x + self.degree_x + 1
        self.num_knots_y = self.num_coeffs_y + self.degree_y + 1
        self.num_inner_knots_x = self.num_knots_x - 2 * self.degree_x
        self.num_inner_knots_y = self.num_knots_y - 2 * self.degree_y

        self.inner_knots_x = np.linspace(0, 1, self.num_inner_knots_x + 2)[1:-1]
        self.inner_knots_y = np.linspace(0, 1, self.num_inner_knots_y + 2)[1:-1]

        self.params_init = True

    def fit(self, img_gray, bin_img):
        """
        For interpolation (inside the convex hull of known data) => use spline.
        For extrapolation (outside that hull) => do row-by-row linear extension.
        
        Returns fitted surface as (H x W) float64, with saturation to [0, 255].
        """
        if not self.params_init:
            self.initParams(img_gray.shape)

        # Flatten grayscale intensities
        z_full = img_gray.astype(np.float64).ravel()

        # Known data: e.g. bin == 0
        mask_known = (bin_img.ravel() == 0)
        x_masked = self.x_flat[mask_known]
        y_masked = self.y_flat[mask_known]
        z_masked = z_full[mask_known]

        if len(z_masked) < 2:
            raise ValueError("Not enough known points to fit anything!")

        # 1) Fit Cubic Spline inside the region
        spline = LSQBivariateSpline(
            x_masked,
            y_masked,
            z_masked,
            tx=self.inner_knots_x,
            ty=self.inner_knots_y,
            kx=self.degree_x,
            ky=self.degree_y
        )

        # Evaluate the spline over the full grid => shape (H, W)
        x_eval = np.linspace(0, 1, self.W)
        y_eval = np.linspace(0, 1, self.H)

        z_spline_2d = spline(x_eval, y_eval)  # => shape (len(y_eval), len(x_eval))
        if z_spline_2d.shape != (self.H, self.W):
            z_spline_2d = z_spline_2d.T  # ensure (H, W)

        # 2) Build the convex hull of known points for inside/outside tests
        points_known = np.column_stack([x_masked, y_masked])
        if len(points_known) < 3:
            # With <3 points, we can't form a polygon. Treat everything as outside.
            hull_mask = np.zeros((self.H, self.W), dtype=bool)
        else:
            hull = ConvexHull(points_known)  # hull.vertices => indices
            hull_pts = points_known[hull.vertices]  # (N_hull, 2)

            # We'll do a standard 'point in polygon' test for each pixel
            X_eval, Y_eval = np.meshgrid(x_eval, y_eval)
            coords_eval = np.column_stack((X_eval.ravel(), Y_eval.ravel()))

            inside = _points_in_poly(coords_eval, hull_pts)  # custom helper
            hull_mask = inside.reshape((self.H, self.W))

        # 3) Combine spline + row-based extrapolation
        fitted_2d = np.copy(z_spline_2d)  # start with spline predictions
        outside_mask = ~hull_mask

        # Precompute the row-based boundary for each row: 
        row_boundaries = [None]*self.H
        for row_i in range(self.H):
            inside_cols = np.where(hull_mask[row_i, :])[0]
            if inside_cols.size == 0:
                # No intersection in this row
                row_boundaries[row_i] = None
            else:
                min_col = inside_cols.min()
                max_col = inside_cols.max()
                row_boundaries[row_i] = (min_col, max_col)

        # Now handle each row's outside pixels
        for row_i in range(self.H):
            minmax = row_boundaries[row_i]
            if minmax is None:
                # Entire row is outside => fallback
                fitted_2d[row_i, :] = _fallback_extrapolate_entire_row(
                    row_i, fitted_2d, row_boundaries
                )
                continue

            min_col, max_col = minmax
            z_left  = fitted_2d[row_i, min_col]
            z_right = fitted_2d[row_i, max_col]

            # For columns < min_col => do linear extrapolation from left boundary
            for col_j in range(0, min_col):
                if outside_mask[row_i, col_j]:
                    slope = z_left - fitted_2d[row_i, min_col - 1]
                    slope_col = 1.0
                    dist = (col_j - (min_col - 1))
                    fitted_2d[row_i, col_j] = (
                        fitted_2d[row_i, min_col - 1] + slope * (dist / slope_col)
                    )

            # For columns > max_col => do linear extrapolation from right boundary
            for col_j in range(max_col + 1, self.W):
                if outside_mask[row_i, col_j]:
                    # Simple zero slope in this example
                    fitted_2d[row_i, col_j] = z_right

        # 4) *** SATURATE (CLIP) to [0..255] ***
        np.clip(fitted_2d, 0, 255, out=fitted_2d)
        
        return fitted_2d


# -------------------------------------------------------------------------
# HELPER: Test if a set of points is inside a convex polygon (x,y).
def _points_in_poly(points, hull_pts):
    """
    points: shape (N, 2)
    hull_pts: shape (M, 2) in some order (the hull's vertices).
    Returns: boolean array of length N, True if inside
    """
    if len(hull_pts) < 3:
        return np.zeros(len(points), dtype=bool)
    return _ray_casting(points, hull_pts)


def _ray_casting(points, polygon):
    """ Minimal 'ray casting' approach to test inside vs outside. """
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(len(points), dtype=bool)

    xs = polygon[:, 0]
    ys = polygon[:, 1]
    n = len(polygon)

    for ipt, (px, py) in enumerate(points):
        count_intersect = 0
        for i in range(n):
            x1, y1 = xs[i], ys[i]
            x2, y2 = xs[(i + 1) % n], ys[(i + 1) % n]
            if y1 > y2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
            if (py > y1) and (py <= y2):
                if y2 == y1:
                    continue
                intersect_x = x1 + (py - y1)*(x2 - x1)/(y2 - y1)
                if intersect_x >= px:
                    count_intersect += 1
        if (count_intersect % 2) == 1:
            inside[ipt] = True
    return inside


def _fallback_extrapolate_entire_row(row_i, fitted_2d, row_boundaries):
    """
    If the entire row row_i is outside the hull, fallback approach:
    - Copy from the nearest row above/below that intersects the hull
      or fill zeros if none found.
    """
    H, W = fitted_2d.shape
    above, below = None, None
    # search up
    for r in range(row_i - 1, -1, -1):
        if row_boundaries[r] is not None:
            above = r
            break
    # search down
    for r in range(row_i + 1, H):
        if row_boundaries[r] is not None:
            below = r
            break

    if above is None and below is None:
        # no rows in the entire image intersect => fallback: fill zeros
        return np.zeros(W, dtype=np.float64)

    # If we only found 'above':
    if below is None:
        return fitted_2d[above, :].copy()

    # If we only found 'below':
    if above is None:
        return fitted_2d[below, :].copy()

    # If we have both, do vertical interpolation
    dist_above = row_i - above
    dist_below = below - row_i
    total = dist_above + dist_below
    w_above = dist_below / total
    w_below = dist_above / total
    return w_above * fitted_2d[above, :] + w_below * fitted_2d[below, :]


HUGE_METRIC = 1e9

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

class AutoDepthBGSubtractor:
    def __init__(self, glob_step=3, cell_w_r=0.2, occupied_cell_fraction=0.002, 
                 imwrite_global=False, imwrite_local=False, 
                 log_all=False, log_global=False, log_local=False):
        self.cell_width_ratio = cell_w_r
        self.glob_step = glob_step
        self.imwrite_global = imwrite_global
        self.imwrite_local = imwrite_local
        self.log_all = log_all
        self.log_global = log_global
        self.log_local = log_local

        self.currentTMask = None
        self.currentDImg = None
        self.total_pixels = None
        self.mosaicArray = None
        self.bestCandidates = {}
        self.latest_index = 0
        self.edge_template_threshold = 25
        self.doneWithMosaicArray = False
        self.occupied_cell_fraction = occupied_cell_fraction
        self.canny_min=0
        self.canny_max=255
        self.match_threshold_humming=300
        self.kp_pos_radius_fraction=0.0264
        self.cbachss = 5 # Corner Based Adjacency Check Square Size
        self.spline_offset = 8

        self.spline_fitter = VisualSplineFit(x_coeffs=2, degree_x=2, degree_y=2)

        # We'll store adjacency-failure lines here
        self.adjFailurePixels = set()

        if self.imwrite_global or self.log_all or self.log_global:
            self.glob_dir = "glob_search_output"
            if not os.path.exists(self.glob_dir):
                os.makedirs(self.glob_dir)
        else:
            self.glob_dir = None

        if self.imwrite_local or self.log_local:
            self.loc_dir = "loc_search_output"
            if not os.path.exists(self.loc_dir):
                os.makedirs(self.loc_dir)
        else:
            self.loc_dir = None
    
    def compute_edge_metric(self, masked_image, template_mask):
        detected_edges = cv2.Canny(masked_image, self.canny_min, self.canny_max)
        kernel = np.ones((5, 5), np.uint8) 
        detected_edges = cv2.dilate(detected_edges, kernel, iterations=1) 

        orb = cv2.ORB_create()
        kp_detected, des_detected = orb.detectAndCompute(detected_edges, None)
        kp_template, des_template = orb.detectAndCompute(template_mask, None)

        if des_detected is None or des_template is None:
            return HUGE_METRIC, None, None, None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_detected, des_template)
        matches = sorted(matches, key=lambda x: x.distance)

        h, w = masked_image.shape[:2]
        pixel_radius = self.kp_pos_radius_fraction * np.sqrt(w**2 + h**2)

        good_matches = []
        good_match_coords = []
        for m in matches:
            if m.distance < self.match_threshold_humming:
                (x1, y1) = kp_detected[m.queryIdx].pt
                (x2, y2) = kp_template[m.trainIdx].pt
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if dist < pixel_radius:
                    good_matches.append(m)
                    good_match_coords.append((x1, y1))

        if self.imwrite_local or self.imwrite_global:
            detected_edges_color = cv2.cvtColor(detected_edges, cv2.COLOR_GRAY2BGR)
            template_mask_color = cv2.cvtColor(template_mask, cv2.COLOR_GRAY2BGR)
            matched_image = cv2.drawMatches(
                detected_edges_color, kp_detected,
                template_mask_color, kp_template,
                good_matches, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
        else:
            detected_edges = None
            matched_image = None

        if len(good_matches) == 0:
            return HUGE_METRIC, None, None, None
        else:
            return (1.0 / len(good_matches)), good_match_coords, matched_image, detected_edges


    def compute_threshold_pattern(self, image_shape, bottom_threshold, slope):
        H, W = image_shape
        row_indices = np.arange(H)
        row_thresholds = bottom_threshold - slope * ((H - 1 - row_indices) / (H - 1))
        row_thresholds = np.clip(row_thresholds, 0, 255).astype(np.uint8)
        return np.tile(row_thresholds[:, np.newaxis], (1, W))

    def iterateLinear(self, bt_range, slope_range, step):
        depth_image = self.currentDImg
        template_mask = self.currentTMask
        H, W = depth_image.shape
        all_candidates = []
        for bt in range(bt_range[0], bt_range[1], step):
            for slope in range(slope_range[0], slope_range[1], step):
                threshold_pattern = self.compute_threshold_pattern((H, W), bt, slope)
                binary_output = apply_threshold(depth_image, threshold_pattern)
                
                fg_count = np.count_nonzero(binary_output)
                bg_fraction = 1.0 - (fg_count / self.total_pixels)

                if np.all(binary_output == 0) or np.all(binary_output == 255) or \
                   bg_fraction < 0.25:
                    all_candidates.append({
                        "bottom_threshold": bt,
                        "slope": slope,
                        "metric": HUGE_METRIC,
                        "coords": None,
                        "masked_image": None,
                        "threshold_pattern": None,
                        "kp_image": None,
                        "edge_image": None,
                        "image_name": f"masked_bt{bt:03d}_slope{slope:03d}.jpg"
                    })
                    continue

                metric, coords, kp_image, eg_img = self.compute_edge_metric(binary_output, template_mask)
                all_candidates.append({
                    "bottom_threshold": bt,
                    "slope": slope,
                    "metric": metric,
                    "coords": coords,
                    "masked_image": binary_output,
                    "threshold_pattern": threshold_pattern,
                    "kp_image": kp_image,
                    "edge_image": eg_img,
                    "image_name": f"masked_bt{bt:03d}_slope{slope:03d}.jpg"
                })
        return all_candidates

    def sortGlobal(self):
        t00 = time.time()
        global_candidates_list = self.iterateLinear((0, 256), (0, 256), step=self.glob_step)
        print("iterate time: ", time.time() - t00)
        # Global logs
        if self.log_all and self.glob_dir:
            with open(os.path.join(self.glob_dir, "log_all.csv"), "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["image_name", "bottom_threshold", "slope", "metric"])
                for cand in global_candidates_list:
                    writer.writerow([cand["image_name"],
                                     cand["bottom_threshold"],
                                     cand["slope"],
                                     cand["metric"]])

        if self.log_global and self.glob_dir:
            with open(os.path.join(self.glob_dir, "log_global.csv"), "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["image_name", "bottom_threshold", "slope", "metric"])
                for cand in global_candidates_list:
                    if cand["metric"] < HUGE_METRIC:
                        writer.writerow([cand["image_name"],
                                         cand["bottom_threshold"],
                                         cand["slope"],
                                         cand["metric"]])

        if self.imwrite_global and self.glob_dir:
            for cand in global_candidates_list:
                if cand["metric"] < HUGE_METRIC and cand["masked_image"] is not None:
                    cv2.imwrite(os.path.join(self.glob_dir, cand["image_name"]),
                                cand["masked_image"])
                    local_fname_tp = f"globlist_tp_{cand['image_name']}"
                    local_fname_kp = f"globlist_kp_{cand['image_name']}"
                    local_fname_eg = f"globlist_eg_{cand['image_name']}"
                    if cand["threshold_pattern"] is not None:
                        cv2.imwrite(os.path.join(self.glob_dir, local_fname_tp), cand["threshold_pattern"])
                    if cand["kp_image"] is not None:
                        cv2.imwrite(os.path.join(self.glob_dir, local_fname_kp), cand["kp_image"])
                    if cand["edge_image"] is not None:
                        cv2.imwrite(os.path.join(self.glob_dir, local_fname_eg), cand["edge_image"])

        return sorted(global_candidates_list, key=lambda x: x["metric"])

    def compute_mosaic_array(self):
        self.mosaicArray = np.zeros((self.num_cells_y, self.num_cells_x), dtype=int)
        for row_cell in range(self.num_cells_y):
            for col_cell in range(self.num_cells_x):
                y_start = row_cell * self.cell_height
                y_end   = y_start + self.cell_height
                x_start = col_cell * self.cell_width
                x_end   = x_start + self.cell_width

                cell = self.currentTMask[y_start:y_end, x_start:x_end]
                white_pixels = np.count_nonzero(cell)
                fraction_white = white_pixels / self.total_pixels
                if fraction_white > self.occupied_cell_fraction:
                    self.mosaicArray[row_cell, col_cell] = -2
                else:
                    self.mosaicArray[row_cell, col_cell] = -1

    ########################################################################
    # markBoundaryLine: when occupant_masked line is all zero, we expand line
    # by line, but we also STOP if self.currentTMask has 255 in that line.
    ########################################################################
    def markBoundaryLine(self, occupant_masked, boundary, y1, y2, x1, x2):
        if boundary == 'top':
            r = y1
            while r < y2:
                # occupant's masked row
                rowdata = occupant_masked[r, x1:x2]
                if np.any(rowdata == 0):
                    break
                # If TMask has 255 in this row => stop
                if np.any(self.currentTMask[r, x1:x2] == 255):
                    break
                # Mark adjacency fail
                for c in range(x1, x2):
                    self.adjFailurePixels.add((r, c))
                r += 1

        elif boundary == 'bottom':
            r = y2 - 1
            while r >= y1:
                rowdata = occupant_masked[r, x1:x2]
                if np.any(rowdata == 0):
                    break
                if np.any(self.currentTMask[r, x1:x2] == 255):
                    break
                for c in range(x1, x2):
                    self.adjFailurePixels.add((r, c))
                r -= 1

        elif boundary == 'left':
            c = x1
            while c < x2:
                coldata = occupant_masked[y1:y2, c]
                if np.any(coldata == 0):
                    break
                if np.any(self.currentTMask[y1:y2, c] == 255):
                    break
                for r in range(y1, y2):
                    self.adjFailurePixels.add((r, c))
                c += 1

        else:  # 'right'
            c = x2 - 1
            while c >= x1:
                coldata = occupant_masked[y1:y2, c]
                if np.any(coldata == 0):
                    break
                if np.any(self.currentTMask[y1:y2, c] == 255):
                    break
                for r in range(y1, y2):
                    self.adjFailurePixels.add((r, c))
                c -= 1

    def doesPassAdjacentCondition(self, occupant_id, row_cell, col_cell):
        """
        If occupant boundary is all zero, we call markBoundaryLine(...).
        That sets adjacency fails in self.adjFailurePixels. We return False.
        """
        cand = self.bestCandidates[occupant_id]
        occupant_masked = cand["masked_image"]
        if occupant_masked is None:
            return True

        y1 = row_cell * self.cell_height
        y2 = y1 + self.cell_height
        x1 = col_cell * self.cell_width
        x2 = x1 + self.cell_width

        neighbors = [
            (-1,0,'top'),
            (1,0,'bottom'),
            (0,-1,'left'),
            (0,1,'right')
        ]

        for dy, dx, boundary_tag in neighbors:
            ny = row_cell + dy
            nx = col_cell + dx
            if 0 <= ny < self.num_cells_y and 0 <= nx < self.num_cells_x:
                if self.mosaicArray[ny, nx] in (-1, -2):
                    if boundary_tag == 'top':
                        # occupant top row
                        top_row = occupant_masked[y1, x1:x2]
                        if np.all(top_row==255):
                            self.markBoundaryLine(occupant_masked,'top',y1,y2,x1,x2)
                            return False
                    elif boundary_tag == 'bottom':
                        bot_row = occupant_masked[y2-1, x1:x2]
                        if np.all(bot_row==255):
                            self.markBoundaryLine(occupant_masked,'bottom',y1,y2,x1,x2)
                            return False
                    elif boundary_tag == 'left':
                        left_col = occupant_masked[y1:y2, x1]
                        if np.all(left_col==255):
                            self.markBoundaryLine(occupant_masked,'left',y1,y2,x1,x2)
                            return False
                    else:
                        right_col = occupant_masked[y1:y2, x2-1]
                        if np.all(right_col==255):
                            self.markBoundaryLine(occupant_masked,'right',y1,y2,x1,x2)
                            return False
        return True

    ########################################################################
    # STEP 1: occupant assignment ignoring adjacency
    ########################################################################
    def assign_cells_basic(self, sorted_candidates):
        self.bestCandidates = {}
        for idx, cand in enumerate(sorted_candidates):
            self.bestCandidates[idx] = cand

        self.counts_3d = np.zeros((len(sorted_candidates), self.num_cells_y, self.num_cells_x), dtype=np.int32)
        for idx, cand in enumerate(sorted_candidates):
            if cand["coords"] is None:
                continue
            for (x, y) in cand["coords"]:
                col = int(x // self.cell_width)
                row = int(y // self.cell_height)
                if 0 <= row < self.num_cells_y and 0 <= col < self.num_cells_x:
                    if self.mosaicArray[row, col] == -2:
                        self.counts_3d[idx, row, col] += 1

        occupant_ids_set = set()
        for row_cell in range(self.num_cells_y):
            for col_cell in range(self.num_cells_x):
                if self.mosaicArray[row_cell, col_cell] == -2:
                    cell_counts = self.counts_3d[:, row_cell, col_cell]
                    mx = np.max(cell_counts)
                    if mx > 0:
                        occ_id = np.argmax(cell_counts)
                        occupant_ids_set.add(occ_id)
                        self.mosaicArray[row_cell, col_cell] = occ_id

        # Local logging
        if occupant_ids_set:
            for occ_id in occupant_ids_set:
                cand = self.bestCandidates[occ_id]
                # images
                if self.imwrite_local and self.loc_dir and (cand["masked_image"] is not None):
                    local_fname = f"local_cluster_{occ_id}_{cand['image_name']}"
                    cv2.imwrite(os.path.join(self.loc_dir, local_fname), cand["masked_image"])
                    if cand["threshold_pattern"] is not None:
                        local_fname_tp = f"local_cluster_thpat_{occ_id}_{cand['image_name']}"
                        cv2.imwrite(os.path.join(self.loc_dir, local_fname_tp), cand["threshold_pattern"])
                    if cand["kp_image"] is not None:
                        local_fname_kp = f"local_cluster_kp_{occ_id}_{cand['image_name']}"
                        cv2.imwrite(os.path.join(self.loc_dir, local_fname_kp), cand["kp_image"])
                    if cand["edge_image"] is not None:
                        local_fname_eg = f"local_cluster_eg_{occ_id}_{cand['image_name']}"
                        cv2.imwrite(os.path.join(self.loc_dir, local_fname_eg), cand["edge_image"])

                # csv
                if self.log_local and self.loc_dir:
                    local_log_path = os.path.join(self.loc_dir, "log_local.csv")
                    file_exists = os.path.isfile(local_log_path)
                    with open(local_log_path, "a", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        if not file_exists:
                            writer.writerow(["Index", "image_name", "bottom_threshold", "slope", "metric"])
                        writer.writerow([occ_id,
                                         cand["image_name"],
                                         cand.get("bottom_threshold",-999),
                                         cand.get("slope",-999),
                                         cand.get("metric",HUGE_METRIC)])

    ########################################################################
    # STEP 2: occupant re-assign if adjacency fails
    ########################################################################
    def fix_adjacent_condition(self):
        occupant_ids_set = set()

        for row_cell in range(self.num_cells_y):
            for col_cell in range(self.num_cells_x):
                occupant_id = self.mosaicArray[row_cell, col_cell]
                if occupant_id < 0:
                    continue

                if not self.doesPassAdjacentCondition(occupant_id, row_cell, col_cell):
                    # occupant fails => try next occupant
                    cell_counts = self.counts_3d[:, row_cell, col_cell]
                    sorted_desc = np.argsort(cell_counts)[::-1]
                    chosen_id = occupant_id
                    for cand_id in sorted_desc:
                        if cell_counts[cand_id]==0:
                            break
                        if cand_id == occupant_id:
                            continue
                        # check adjacency for cand_id
                        if self.doesPassAdjacentCondition(cand_id, row_cell, col_cell):
                            chosen_id = cand_id
                            break
                    if chosen_id != occupant_id:
                        self.mosaicArray[row_cell, col_cell] = chosen_id
                        occupant_id = chosen_id

                occupant_ids_set.add(occupant_id)

        # local logging occupant changes
        for occ_id in occupant_ids_set:
            cand = self.bestCandidates.get(occ_id)
            if not cand:
                continue
            if self.imwrite_local and self.loc_dir and (cand["masked_image"] is not None):
                local_fname = f"fixadj_cluster_{occ_id}_{cand['image_name']}"
                cv2.imwrite(os.path.join(self.loc_dir, local_fname), cand["masked_image"])
                if cand["threshold_pattern"] is not None:
                    local_fname_tp = f"fixadj_thpat_{occ_id}_{cand['image_name']}"
                    cv2.imwrite(os.path.join(self.loc_dir, local_fname_tp), cand["threshold_pattern"])
                if cand["kp_image"] is not None:
                    local_fname_kp = f"fixadj_kp_{occ_id}_{cand['image_name']}"
                    cv2.imwrite(os.path.join(self.loc_dir, local_fname_kp), cand["kp_image"])
                if cand["edge_image"] is not None:
                    local_fname_eg = f"fixadj_eg_{occ_id}_{cand['image_name']}"
                    cv2.imwrite(os.path.join(self.loc_dir, local_fname_eg), cand["edge_image"])

            if self.log_local and self.loc_dir:
                local_log_path = os.path.join(self.loc_dir, "log_local.csv")
                file_exists = os.path.isfile(local_log_path)
                with open(local_log_path, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    if not file_exists:
                        writer.writerow(["Index", "image_name", "bottom_threshold", "slope", "metric"])
                    writer.writerow([occ_id,
                                     cand["image_name"],
                                     cand.get("bottom_threshold",-999),
                                     cand.get("slope",-999),
                                     cand.get("metric",HUGE_METRIC)])

    def cornerBasedAdjacencyCheck(self, masked_image):
        """
        For each *internal* grid node (row_node, col_node),
        we examine the four 5x5 squares around that node:
            top-left  => masked_image[node_y-5:node_y,   node_x-5:node_x]
            top-right => masked_image[node_y-5:node_y,   node_x:node_x+5]
            bot-left  => masked_image[node_y:node_y+5,   node_x-5:node_x]
            bot-right => masked_image[node_y:node_y+5,   node_x:node_x+5]

        Each square is considered "fully occupant" if it's all 255 in the masked_image.
        If exactly one of these squares is fully 255, we find the corresponding neighbor cell
        (top-left => cell(row_node-1, col_node-1), etc.) and perform expansions in that subregion:
        - row expansions (top -> down, bottom -> up)
        - column expansions (left -> right, right -> left)
        We add all those occupant pixels to self.adjFailurePixels, but **stop expansions** if we see:
        - masked_image == 0 (background), or
        - self.currentTMask == 255 (some edge region).

        This function is the "corner-based adjacency" check with the **new** annotation:
        masked_image == 255 => occupant / foreground, masked_image == 0 => background.
        """

        H, W = masked_image.shape

        def is_fully_white(r1, r2, c1, c2):
            """
            Returns True if the sub-block in [r1:r2, c1:c2] is entirely 255 in masked_image;
            also clamps out-of-bounds. If out-of-bounds => returns False.
            """
            if r1<0 or c1<0 or r2>H or c2>W:
                return False
            block = masked_image[r1:r2, c1:c2]
            return np.all(block == 255)

        def expand_linewise(row_start, row_end, col_start, col_end, step):
            """
            Expand row by row (vertical) from row_start..row_end with a given step (+1 or -1).
            We add occupant pixels (255) to self.adjFailurePixels,
            but stop if we see masked_image==0 or self.currentTMask==255.
            """
            r = row_start
            while (r < row_end if step>0 else r >= row_end):
                row_slice = masked_image[r, col_start:col_end]
                # if any of them == 0 => occupant region is broken => stop
                if np.any(row_slice == 0):
                    break
                # if TMask has 255 => also stop
                if np.any(self.currentTMask[r, col_start:col_end] == 255):
                    break
                # Mark these occupant pixels
                for c in range(col_start, col_end):
                    self.adjFailurePixels.add((r, c))
                r += step

        def expand_colwise(col_start, col_end, row_start, row_end, step):
            """
            Expand column by column (horizontal) from col_start..col_end with step (+1 or -1).
            Stop expansions if masked_image==0 or TMask==255.
            """
            c = col_start
            while (c < col_end if step>0 else c >= col_end):
                col_slice = masked_image[row_start:row_end, c]
                if np.any(col_slice == 0):
                    break
                if np.any(self.currentTMask[row_start:row_end, c] == 255):
                    break
                for r in range(row_start, row_end):
                    self.adjFailurePixels.add((r, c))
                c += step

        def expand_cell_subregion(r_cell, c_cell):
            """
            Once we identify which cell subregion is occupant-based on the corner check,
            we do expansions on its top, bottom, left, and right boundaries, stopping 
            if masked_image==0 or TMask==255.
            """
            y1 = r_cell * self.cell_height
            y2 = y1 + self.cell_height
            x1 = c_cell * self.cell_width
            x2 = x1 + self.cell_width
            # top boundary => expand downward
            expand_linewise(y1, y2, x1, x2, step=+1)
            # bottom boundary => expand upward
            expand_linewise(y2 - 1, y1 - 1, x1, x2, step=-1)
            # left boundary => expand right
            expand_colwise(x1, x2, y1, y2, step=+1)
            # right boundary => expand left
            expand_colwise(x2 - 1, x1 - 1, y1, y2, step=-1)

        # We'll only iterate over *internal* grid nodes => row_node in [1..num_cells_y-1], col_node in [1..num_cells_x-1]
        # because only those have 4 neighbor cells.
        for row_node in range(1, self.num_cells_y):
            node_y = row_node * self.cell_height
            if node_y < self.cbachss or node_y > H - self.cbachss:
                continue
            for col_node in range(1, self.num_cells_x):
                node_x = col_node * self.cell_width
                if node_x < self.cbachss or node_x > W - self.cbachss:
                    continue

                # check 4 sub-block corners
                top_left  = is_fully_white(node_y - self.cbachss, node_y, node_x - self.cbachss, node_x)
                top_right = is_fully_white(node_y - self.cbachss, node_y, node_x, node_x + self.cbachss)
                bot_left  = is_fully_white(node_y, node_y + self.cbachss, node_x - self.cbachss, node_x)
                bot_right = is_fully_white(node_y, node_y + self.cbachss, node_x, node_x + self.cbachss)

                num_corners = sum([top_left, top_right, bot_left, bot_right])
                if num_corners == 1:
                    # figure out which corner => subregion expansions
                    if top_left:
                        r_cell = row_node - 1
                        c_cell = col_node - 1
                    elif top_right:
                        r_cell = row_node - 1
                        c_cell = col_node
                    elif bot_left:
                        r_cell = row_node
                        c_cell = col_node - 1
                    else:  # bot_right
                        r_cell = row_node
                        c_cell = col_node

                    if (0 <= r_cell < self.num_cells_y) and (0 <= c_cell < self.num_cells_x):
                        expand_cell_subregion(r_cell, c_cell)


    def search(self, template_mask, depth_image, depth_image_name, outdir):
        self.currentTMask = template_mask
        # Binarize at edge threshold
        self.currentTMask[self.currentTMask < self.edge_template_threshold] = 0
        self.currentTMask[self.currentTMask >= self.edge_template_threshold] = 255
        self.currentDImg = depth_image
        self.total_pixels = depth_image.size

        H, W = depth_image.shape
        self.cell_width, self.cell_height, self.num_cells_x, self.num_cells_y = \
            get_cell_info((H, W), self.cell_width_ratio)

        self.adjFailurePixels = set()
        self.bestCandidates = {}
        self.doneWithMosaicArray = False
        self.latest_index = 0

        # 1) build mosaic array
        self.compute_mosaic_array()
        print("Initial Mosaic Array:\n", self.mosaicArray)

        # 2) global sort
        sorted_candidates = self.sortGlobal()

        # 3) occupant assignment ignoring adjacency
        self.assign_cells_basic(sorted_candidates)
        print("Mosaic after assignment:\n", self.mosaicArray)

        # 4) occupant re-assign if adjacency fails
        self.fix_adjacent_condition()
        print("Mosaic after adjacency fix:\n", self.mosaicArray)

        # 5) Build combined_thresh
        combined_thresh = np.zeros((H, W), dtype=np.uint8)
        for row_cell in range(self.num_cells_y):
            for col_cell in range(self.num_cells_x):
                occupant_id = self.mosaicArray[row_cell, col_cell]
                y_start = row_cell*self.cell_height
                y_end   = y_start + self.cell_height
                x_start = col_cell*self.cell_width
                x_end   = x_start + self.cell_width

                if occupant_id >=0:
                    cand = self.bestCandidates[occupant_id]
                    if cand["threshold_pattern"] is not None:
                        combined_thresh[y_start:y_end, x_start:x_end] = \
                            cand["threshold_pattern"][y_start:y_end, x_start:x_end]
                else:
                    combined_thresh[y_start:y_end, x_start:x_end] = 255


        # 6) Mark adjacency-failure lines => 255
        for (py, px) in self.adjFailurePixels:
            combined_thresh[py, px] = 255
        
        self.adjFailurePixels = set()

        # 7) Save final
        _masked = apply_threshold(depth_image, combined_thresh)

        self.cornerBasedAdjacencyCheck(_masked)

        for (py, px) in self.adjFailurePixels:
            combined_thresh[py, px] = 255

        final_masked = apply_threshold(depth_image, combined_thresh)

        kernel = np.ones((3, 3), np.uint8) 
        edg = cv2.erode(self.currentTMask, kernel, iterations=1) 
        final_masked[edg == 255] = 255
        out_fname_c = os.path.join(outdir, f"{os.path.splitext(os.path.basename(depth_image_name))[0]}_bgCurve.jpg")
        out_fname_m = os.path.join(outdir, f"{os.path.splitext(os.path.basename(depth_image_name))[0]}_bgsMasked.jpg")
        masked_image = np.where(final_masked == 0, depth_image, 0).astype(np.uint8)

        # 3) Fit using only pixels where the binary image is 0
        fitted_surface = self.spline_fitter.fit(masked_image, final_masked)
        background_curve = fitted_surface + self.spline_offset
        np.clip(background_curve, 0, 255, out=background_curve)
        m = apply_threshold(depth_image, background_curve)
        new_masked = np.where(m == 0, depth_image, 0).astype(np.uint8)
        cv2.imwrite(out_fname_m, new_masked)
        cv2.imwrite(out_fname_c, background_curve)
        print(f"Saved final => {out_fname_m}")


def main():
    depth_dir = "../data/depth"
    tmask_dir = "../data/depth_edge"
    output_dir = "../data/search_thresh_output"

    depth_files = sorted([f for f in os.listdir(depth_dir) if os.path.isfile(os.path.join(depth_dir, f))])
    tmask_files = sorted([f for f in os.listdir(tmask_dir) if os.path.isfile(os.path.join(tmask_dir, f))])

    if set(depth_files) != set(tmask_files):
        raise ValueError("Depth images and template masks must match")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    searcher = AutoDepthBGSubtractor(
        glob_step=3,
        cell_w_r=0.2,
        occupied_cell_fraction=0.005,
        imwrite_global=True,
        imwrite_local=True,
        log_all=True,
        log_global=True,
        log_local=True
    )

    for fname in depth_files:
        depth_path = os.path.join(depth_dir, fname)
        tmask_path = os.path.join(tmask_dir, fname)

        depth_image = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        if depth_image is None:
            raise IOError(f"Could not load depth image: {depth_path}")
        template_mask = cv2.imread(tmask_path, cv2.IMREAD_GRAYSCALE)
        if template_mask is None:
            raise IOError(f"Could not load template mask: {tmask_path}")
        if depth_image.shape != template_mask.shape:
            raise ValueError(f"Shape mismatch for {fname}")

        basename = os.path.splitext(fname)[0]
        t0 = time.time()
        searcher.search(template_mask, depth_image, basename, output_dir)
        print("time: ", time.time() - t0)
        print("-------------")

if __name__=="__main__":
    main()
