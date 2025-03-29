import cv2
import numpy as np
import os
import csv

from scipy.interpolate import LSQBivariateSpline
HUGE_METRIC = 1e9

###############################################################################
# The VisualSplineFit class from your spline_fit.py, slightly modified to
# accept partial data (NaNs). We skip those from the fitting (white regions).
###############################################################################
class VisualSplineFit:
    def __init__(self, x_coeffs=2, degree_x=3, degree_y=3):
        """
        A simple 2D B-spline-fitting helper for a grayscale image of shape (H, W).
        We default to 4 x-coeffs, 3rd-degree in x, 3rd-degree in y.
        """
        self.degree_x = degree_x
        self.degree_y = degree_y
        self.num_coeffs_x = x_coeffs
        self.params_init = False

    def initParams(self, img_shape):
        """
        Prepare internal arrays and knot vectors for LSQBivariateSpline.
        """
        self.H, self.W = img_shape
        # Adjust num_coeffs_y based on the image aspect ratio
        self.num_coeffs_y = int(round((float(self.H)/self.W) * self.num_coeffs_x))

        # Create [0..1] coordinate grids
        X, Y = np.meshgrid(
            np.linspace(0, 1, self.W),  # length W
            np.linspace(0, 1, self.H)   # length H
        )
        self.x = X.ravel()  # (H*W,) flatten
        self.y = Y.ravel()  # (H*W,)

        # #knots = #coeffs + degree + 1
        self.num_knots_x = self.num_coeffs_x + self.degree_x + 1
        self.num_knots_y = self.num_coeffs_y + self.degree_y + 1

        # Inner knots (excluding the edges)
        # => [1:-1] slices off the first and last of the linspace
        self.num_inner_knots_x = self.num_knots_x - 2 * self.degree_x
        self.num_inner_knots_y = self.num_knots_y - 2 * self.degree_y
        self.inner_knots_x = np.linspace(0, 1, self.num_inner_knots_x + 2)[1:-1]
        self.inner_knots_y = np.linspace(0, 1, self.num_inner_knots_y + 2)[1:-1]

        self.params_init = True

    def fit(self, img):
        """
        Fit a 2D B-spline surface to 'img' (shape HxW). 
        Any NaNs in 'img' are skipped (not used in the fitting).
        Returns the fitted surface (float64, shape HxW).
        """
        if not self.params_init:
            self.initParams(img.shape)

        # Flatten intensities
        z_all = img.astype(np.float64).ravel()
        # Identify which pixels are valid (non-NaN)
        valid_mask = ~np.isnan(z_all)
        if not np.any(valid_mask):
            # Everything is NaN => can't fit; just return zeros
            return np.zeros(img.shape, dtype=np.float64)

        # Subsample only valid pixels
        x_valid = self.x[valid_mask]
        y_valid = self.y[valid_mask]
        z_valid = z_all[valid_mask]

        # Fit B-spline
        spline = LSQBivariateSpline(
            x_valid,
            y_valid,
            z_valid,
            tx=self.inner_knots_x,
            ty=self.inner_knots_y,
            kx=self.degree_x,
            ky=self.degree_y
        )

        # Evaluate on entire [0..1] domain
        x_sorted = np.linspace(0, 1, self.W)
        y_sorted = np.linspace(0, 1, self.H)

        fitted_2d = spline(x_sorted, y_sorted)  # shape => (len(y_sorted), len(x_sorted))
        # If shape is (W, H), transpose
        if fitted_2d.shape == (self.W, self.H):
            fitted_2d = fitted_2d.T

        return fitted_2d


def compute_edge_metric(masked_image, template_mask, 
                        canny_min=0, 
                        canny_max=255, 
                        match_threshold=100,
                        pos_radius_fraction=0.0001):
    detected_edges = cv2.Canny(masked_image, canny_min, canny_max)
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
    pixel_radius = pos_radius_fraction * (w * h)

    good_matches = []
    good_match_coords = []
    for m in matches:
        if m.distance < match_threshold:
            (x1, y1) = kp_detected[m.queryIdx].pt
            (x2, y2) = kp_template[m.trainIdx].pt
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if dist < pixel_radius:
                good_matches.append(m)
                good_match_coords.append((x1, y1))

    detected_edges_color = cv2.cvtColor(detected_edges, cv2.COLOR_GRAY2BGR)
    template_mask_color = cv2.cvtColor(template_mask, cv2.COLOR_GRAY2BGR)
    matched_image = cv2.drawMatches(
        detected_edges_color, kp_detected,
        template_mask_color, kp_template,
        good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    if len(good_matches) == 0:
        return HUGE_METRIC, None, None, None
    else:
        return (1.0 / len(good_matches)), good_match_coords, matched_image, detected_edges

def get_cell_info(img_shape, cell_width_ratio):
    height, width = img_shape

    cell_width = max(1, int(round(width * cell_width_ratio)))

    # Estimate number of vertical cells that gives cell_height ~ cell_width
    estimated_cells_y = max(1, round(height / cell_width))
    cell_height = int(round(height / estimated_cells_y))
    
    # Recalculate to ensure full image coverage
    num_cells_y = height // cell_height  # ceil(height / cell_height)
    num_cells_x = width // cell_width

    print("cell info:", cell_width, cell_height, num_cells_x, num_cells_y)
    return cell_width, cell_height, num_cells_x, num_cells_y

class ThresholdSearcher:
    def __init__(self, glob_step=3, cell_w_r=0.2, occupied_cell_fraction=0.01, 
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
        self.mosaicArray = None
        self.bestCandidates = {}
        self.latest_index = 0
        self.doneWithMosaicArray = False
        self.spline_fitter = VisualSplineFit(x_coeffs=1)
        self.occupied_cell_fraction = occupied_cell_fraction
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

    def compute_threshold_pattern(self, image_shape, bottom_threshold, slope):
        H, W = image_shape
        row_indices = np.arange(H)
        row_thresholds = bottom_threshold - slope * ((H - 1 - row_indices) / (H - 1))
        row_thresholds = np.clip(row_thresholds, 0, 255).astype(np.uint8)
        return np.tile(row_thresholds[:, np.newaxis], (1, W))

    def apply_threshold(self, image, threshold_pattern):
        return ((image > threshold_pattern).astype(np.uint8)) * 255

    def iterateLinear(self, bt_range, slope_range, step):
        depth_image = self.currentDImg
        template_mask = self.currentTMask
        total_pixels = depth_image.size
        H, W = depth_image.shape
        all_candidates = []
        for bt in range(bt_range[0], bt_range[1], step):
            for slope in range(slope_range[0], slope_range[1], step):
                threshold_pattern = self.compute_threshold_pattern((H, W), bt, slope)
                binary_output = self.apply_threshold(depth_image, threshold_pattern)
                
                masked_image = np.where(binary_output == 0, depth_image, 0).astype(np.uint8)
                mask = (masked_image > 4)
                unmasked_count = np.count_nonzero(mask)
                unmasked_fraction = unmasked_count / total_pixels

                if np.all(binary_output == 0) or np.all(binary_output == 255) or \
                    unmasked_fraction < 0.1:
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
                metric, coords, kp_image, eg_img = compute_edge_metric(binary_output, template_mask)
                
                all_candidates.append({
                    "bottom_threshold": bt,
                    "slope": slope,
                    "metric": metric,
                    "coords": coords,
                    "masked_image": masked_image,
                    "threshold_pattern": threshold_pattern,
                    "kp_image": kp_image,
                    "edge_image": eg_img,
                    "image_name": f"masked_bt{bt:03d}_slope{slope:03d}.jpg"
                })
        return all_candidates

    def sortGlobal(self):
        global_candidates_list = self.iterateLinear((0, 256), (0, 256), step=self.glob_step)

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
                white_pixels = np.count_nonzero(cell == 255)
                total_pixels = self.cell_height * self.cell_width
                fraction_white = white_pixels / total_pixels
                if fraction_white > self.occupied_cell_fraction:
                    self.mosaicArray[row_cell, col_cell] = -2
                else:
                    self.mosaicArray[row_cell, col_cell] = -1

    def assign_cells_by_most_coords(self, sorted_candidates):
        self.bestCandidates = {}
        for idx, cand in enumerate(sorted_candidates):
            self.bestCandidates[idx] = cand

        num_candidates = len(sorted_candidates)
        counts_3d = np.zeros((num_candidates, self.num_cells_y, self.num_cells_x), dtype=np.int32)

        for idx, cand in enumerate(sorted_candidates):
            if cand["coords"] is None:
                continue
            for (x, y) in cand["coords"]:
                col = int(x // self.cell_width)
                row = int(y // self.cell_height)
                if 0 <= row < self.num_cells_y and 0 <= col < self.num_cells_x:
                    if self.mosaicArray[row, col] == -2:
                        counts_3d[idx, row, col] += 1

        occupant_ids_set = set()

        for row_cell in range(self.num_cells_y):
            for col_cell in range(self.num_cells_x):
                if self.mosaicArray[row_cell, col_cell] == -2:
                    cell_counts = counts_3d[:, row_cell, col_cell]
                    max_counts = np.max(cell_counts)
                    if max_counts > 0:
                        winner_idx = np.argmax(cell_counts)
                        self.mosaicArray[row_cell, col_cell] = winner_idx
                        occupant_ids_set.add(winner_idx)
                    else:
                        self.mosaicArray[row_cell, col_cell] = -2

        if not np.any(self.mosaicArray == -2):
            self.doneWithMosaicArray = True

        # Local logging for each occupant that claimed at least one cell
        for occupant_id in occupant_ids_set:
            cand = self.bestCandidates[occupant_id]
            if self.imwrite_local and self.loc_dir and (cand["masked_image"] is not None):
                local_fname = f"local_cluster_{occupant_id}_{cand['image_name']}"
                local_fname_tp = f"local_cluster_thpat_{occupant_id}_{cand['image_name']}"
                local_fname_kp = f"local_cluster_kp_{occupant_id}_{cand['image_name']}"
                local_fname_eg = f"local_cluster_eg_{occupant_id}_{cand['image_name']}"
                cv2.imwrite(os.path.join(self.loc_dir, local_fname), cand["masked_image"])
                if cand["threshold_pattern"] is not None:
                    cv2.imwrite(os.path.join(self.loc_dir, local_fname_tp), cand["threshold_pattern"])
                if cand["kp_image"] is not None:
                    cv2.imwrite(os.path.join(self.loc_dir, local_fname_kp), cand["kp_image"])
                if cand["edge_image"] is not None:
                    cv2.imwrite(os.path.join(self.loc_dir, local_fname_eg), cand["edge_image"])

            if self.log_local and self.loc_dir:
                local_log_path = os.path.join(self.loc_dir, "log_local.csv")
                file_exists = os.path.isfile(local_log_path)
                with open(local_log_path, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    if not file_exists:
                        writer.writerow(["Index", "image_name", "bottom_threshold", "slope", "metric"])
                    writer.writerow([occupant_id,
                                     cand["image_name"],
                                     cand.get("bottom_threshold", -999),
                                     cand.get("slope", -999),
                                     cand.get("metric", HUGE_METRIC)])


    ############################################################################
    #  NEW spline-based post-processing that leverages VisualSplineFit
    ############################################################################
    def compute_continuous_spline(self, combined_thresh, output_path):
        """
        1) Convert 255 => NaN so that the white cells are treated as unknown.
        2) Use VisualSplineFit to fit a continuous B-spline surface to the
           non-white data.
        3) Save the resulting fitted image to 'output_path'.

        :param combined_thresh: 2D array (H, W) in uint8
        :param output_path: Path where we save the final fitted image
        :return: fitted_image (uint8)
        """
        # Convert to float, set white => NaN
        combined_thresh_f = combined_thresh.astype(np.float64)
        combined_thresh_f[combined_thresh_f == 255] = np.nan

        # Instantiate the fitter
        spline_fitter = VisualSplineFit(x_coeffs=4, degree_x=3, degree_y=3)
        fitted_float = spline_fitter.fit(combined_thresh_f)

        # Clip to [0..255], convert to uint8
        fitted_uint8 = np.clip(fitted_float, 0, 255).astype(np.uint8)

        # Save
        cv2.imwrite(output_path, fitted_uint8)
        return fitted_uint8
    
    def search(self, template_mask, depth_image, depth_image_name="unknown"):
        self.currentTMask = template_mask
        self.currentDImg = depth_image
        height, width = depth_image.shape
        self.cell_width, self.cell_height, self.num_cells_x, self.num_cells_y = \
            get_cell_info((height, width), self.cell_width_ratio)
        self.doneWithMosaicArray = False
        self.bestCandidates = {}
        self.latest_index = 0

        # 1) Create mosaic array
        self.compute_mosaic_array()  # sets mosaicArray to -2 or -1

        # 2) Sort candidates
        sorted_candidates = self.sortGlobal()

        # 3) Assign cells by "most coords"
        self.assign_cells_by_most_coords(sorted_candidates)
        print("Mosaic Array: ")
        print(self.mosaicArray)

        # We'll store occupant areas with threshold pattern, unoccupied with 255
        combined_thresh = np.zeros((height, width), dtype=np.uint8)

        for row_cell in range(self.num_cells_y):
            for col_cell in range(self.num_cells_x):
                occupant_id = self.mosaicArray[row_cell, col_cell]
                y_start = row_cell * self.cell_height
                y_end   = y_start + self.cell_height
                x_start = col_cell * self.cell_width
                x_end   = x_start + self.cell_width

                if occupant_id >= 0:
                    cand = self.bestCandidates[occupant_id]
                    if cand["threshold_pattern"] is not None:
                        combined_thresh[y_start:y_end, x_start:x_end] = \
                            cand["threshold_pattern"][y_start:y_end, x_start:x_end]
                else:
                    # White fill for unoccupied cells
                    combined_thresh[y_start:y_end, x_start:x_end] = 255

        # 4b) Save raw combined_thresh
        raw_out_fname = f"{depth_image_name}_combined_thresh.jpg"
        cv2.imwrite(raw_out_fname, combined_thresh)
        print(f"Saved raw combined threshold => {raw_out_fname}")

        # -----------------------------------------------------
        # 5) Build a version with occupant cells as-is, but
        #    unoccupied cells => NaN, so the spline can fill them.
        # -----------------------------------------------------
        combined_thresh_for_spline = combined_thresh.astype(np.float32)
        combined_thresh_for_spline[combined_thresh_for_spline == 255] = np.nan

        # 5b) Fit a continuous spline to fill the NaNs
        fitted_float = self.spline_fitter.fit(combined_thresh_for_spline)  # shape => (H, W)

        # 5c) Clip to [0..255], convert to uint8
        fitted_uint8 = np.clip(fitted_float, 0, 255).astype(np.uint8)

        # 5d) Save final "processed" image
        fitted_out_fname = f"{depth_image_name}_processed_combined_thresh.jpg"
        cv2.imwrite(fitted_out_fname, fitted_uint8)
        print(f"Saved processed (spline) combined threshold => {fitted_out_fname}")


def main():
    # Directories for depth images and template masks.
    depth_dir = "../data/depth"
    tmask_dir = "../data/depth_edge"
    output_dir = "../data/search_thresh_output"

    # Get sorted list of filenames.
    depth_files = sorted([f for f in os.listdir(depth_dir) if os.path.isfile(os.path.join(depth_dir, f))])
    tmask_files = sorted([f for f in os.listdir(tmask_dir) if os.path.isfile(os.path.join(tmask_dir, f))])

    # Check that both directories have exactly the same filenames.
    if set(depth_files) != set(tmask_files):
        raise ValueError("Depth images and template masks must have matching filenames.")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a ThresholdSearcher object with desired parameters and logging flags.
    searcher = ThresholdSearcher(glob_step=5)

    # Process each matching pair.
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
            raise ValueError(f"Image shape mismatch for {fname}")

        searcher.search(template_mask, depth_image)

if __name__ == "__main__":
    main()
