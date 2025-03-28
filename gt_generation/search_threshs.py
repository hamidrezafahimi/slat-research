import cv2
import numpy as np
import os
import csv

HUGE_METRIC = 1e9

class ThresholdSearcher:
    def __init__(self, num_local_minima=5, glob_step=5, local_search_radius=5,
                 imwrite_global=False, imwrite_local=False, 
                 log_all=False, log_global=False, log_local=False):
        self.num_local_minima = num_local_minima
        self.glob_step = glob_step
        self.local_search_radius = local_search_radius
        self.imwrite_global = imwrite_global
        self.imwrite_local = imwrite_local
        self.log_all = log_all
        self.log_global = log_global
        self.log_local = log_local
        self.currentTMask = None  # current template mask
        self.currentDImg = None   # current depth image

        # Initialize logging directories in __init__
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

    def compute_overlap_metric(self, masked_img, template_mask):
        mask = (masked_img > 4)
        unmasked_count = np.count_nonzero(mask)
        unmasked_fraction = unmasked_count / self.total_pixels
        overlap_ratio = np.sum(masked_img == template_mask) / self.total_pixels
        overlap_metric = 1 - overlap_ratio
        if unmasked_count < 6 or unmasked_fraction < 0.1 or overlap_ratio < 0.2:
            return HUGE_METRIC
        return overlap_metric

    def compute_edge_metric(self, masked_image, template_mask, 
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
        5. Computes a metric in where a lower value indicates more similarity
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
        # Taking a matrix of size 5 as the kernel 
        kernel = np.ones((5, 5), np.uint8) 
        detected_edges = cv2.dilate(detected_edges, kernel, iterations=1) 

        # 2. Detect keypoints and descriptors in both images using ORB.
        orb = cv2.ORB_create()
        kp_detected, des_detected = orb.detectAndCompute(detected_edges, None)
        kp_template, des_template = orb.detectAndCompute(template_mask, None)

        # If either image has no descriptors, metric = 0.0
        if des_detected is None or des_template is None:
            return HUGE_METRIC

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
            return HUGE_METRIC
        else:
            return 1 / len(good_matches)

    def iterateLinear(self, bt_range, slope_range, step):
        depth_image = self.currentDImg
        template_mask = self.currentTMask
        H, W = depth_image.shape
        all_candidates = []
        for bt in range(bt_range[0], bt_range[1], step):
            for slope in range(slope_range[0], slope_range[1], step):
                threshold_pattern = self.compute_threshold_pattern((H, W), bt, slope)
                binary_output = self.apply_threshold(depth_image, threshold_pattern)
                if np.all(binary_output == 0) or np.all(binary_output == 255):
                    all_candidates.append({
                        "bottom_threshold": bt,
                        "slope": slope,
                        "metric": HUGE_METRIC
                    })
                    continue
                masked_image = np.where(binary_output == 0, depth_image, 0).astype(np.uint8)
                metric = self.compute_edge_metric(masked_image, template_mask)
                candidate = {
                    "bottom_threshold": bt,
                    "slope": slope,
                    "metric": metric,
                    "masked_image": masked_image
                }
                all_candidates.append(candidate)
        return all_candidates

    def searchGlobal(self):
        # Perform a coarse search over the full parameter space.
        global_candidates_list = self.iterateLinear((0, 256), (0, 256), step=self.glob_step)

        # Log all candidates from global search.
        if self.log_all and self.glob_dir:
            with open(os.path.join(self.glob_dir, "log_all.csv"), "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["image_name", "bottom_threshold", "slope", "metric"])
                for cand in global_candidates_list:
                    image_name = f"masked_bt{cand['bottom_threshold']:03d}_slope{cand['slope']:03d}.jpg"
                    writer.writerow([image_name, cand["bottom_threshold"], cand["slope"], cand["metric"]])
                    
        # Log only candidates that passed the skip condition (metric < HUGE_METRIC).
        if self.log_global and self.glob_dir:
            with open(os.path.join(self.glob_dir, "log_global.csv"), "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["image_name", "bottom_threshold", "slope", "metric"])
                for cand in global_candidates_list:
                    if cand["metric"] < HUGE_METRIC:
                        image_name = f"masked_bt{cand['bottom_threshold']:03d}_slope{cand['slope']:03d}.jpg"
                        writer.writerow([image_name, cand["bottom_threshold"], cand["slope"], cand["metric"]])
                        
        # Write out global candidate images that passed the skip condition.
        if self.imwrite_global and self.glob_dir:
            for cand in global_candidates_list:
                if cand["metric"] < HUGE_METRIC and "masked_image" in cand:
                    image_name = f"masked_bt{cand['bottom_threshold']:03d}_slope{cand['slope']:03d}.jpg"
                    cv2.imwrite(os.path.join(self.glob_dir, image_name), cand["masked_image"])
        
        # Keep up to num_local_minima best candidates.
        top_candidates = []
        for candidate in global_candidates_list:
            if len(top_candidates) < self.num_local_minima:
                top_candidates.append(candidate)
            else:
                worst_candidate = max(top_candidates, key=lambda x: x["metric"])
                if candidate["metric"] < worst_candidate["metric"]:
                    top_candidates.remove(worst_candidate)
                    top_candidates.append(candidate)
        top_candidates = sorted(top_candidates, key=lambda x: x["metric"])
        return {i: top_candidates[i] for i in range(len(top_candidates))}

    def findLocalMinima(self, candidates):
        candidate_list = list(candidates.values())
        candidate_list = sorted(candidate_list, key=lambda x: x["bottom_threshold"])
        clusters = []
        current_cluster = []
        for cand in candidate_list:
            if not current_cluster:
                current_cluster.append(cand)
            else:
                if abs(cand["bottom_threshold"] - current_cluster[-1]["bottom_threshold"]) <= self.local_search_radius:
                    current_cluster.append(cand)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [cand]
        if current_cluster:
            clusters.append(current_cluster)
        local_minimas = {}
        for i, cluster in enumerate(clusters):
            avg_bt = sum(c["bottom_threshold"] for c in cluster) / len(cluster)
            avg_slope = sum(c["slope"] for c in cluster) / len(cluster)
            local_minimas[i] = {"bottom_threshold": int(round(avg_bt)), "slope": int(round(avg_slope))}
        print("Local minima clusters:")
        for idx, lm in local_minimas.items():
            print(f"  Cluster {idx}: bottom_threshold = {lm['bottom_threshold']}, slope = {lm['slope']}")
        return local_minimas

    def searchLocal(self, local_minimas):
        refined_candidates = []
        local_log_rows = []  # accumulate local log rows if needed
        if self.imwrite_local or self.log_local:
            loc_dir = self.loc_dir
        for idx, lm in local_minimas.items():
            bt_center = lm["bottom_threshold"]
            slope_center = lm["slope"]
            bt_start = max(0, bt_center - self.local_search_radius)
            bt_end = min(256, bt_center + self.local_search_radius + 1)
            slope_start = max(0, slope_center - self.local_search_radius)
            slope_end = min(256, slope_center + self.local_search_radius + 1)
            candidates = self.iterateLinear((bt_start, bt_end), (slope_start, slope_end), step=1)
            valid_candidates = [c for c in candidates if c.get("metric", HUGE_METRIC) < HUGE_METRIC]
            if valid_candidates:
                best_candidate = min(valid_candidates, key=lambda c: c["metric"])
                best_candidate["image_name"] = f"masked_bt{best_candidate['bottom_threshold']:03d}_slope{best_candidate['slope']:03d}.jpg"
                refined_candidates.append(best_candidate)
                local_log_rows.append([best_candidate["image_name"], best_candidate["bottom_threshold"],
                                         best_candidate["slope"], best_candidate["metric"]])
                if self.imwrite_local and self.loc_dir and "masked_image" in best_candidate:
                    local_fname = f"local_cluster_{idx}_" + best_candidate["image_name"]
                    cv2.imwrite(os.path.join(self.loc_dir, local_fname), best_candidate["masked_image"])
        if self.log_local and self.loc_dir and local_log_rows:
            with open(os.path.join(self.loc_dir, "log_local.csv"), "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["image_name", "bottom_threshold", "slope", "metric"])
                for row in local_log_rows:
                    writer.writerow(row)
        if refined_candidates:
            return min(refined_candidates, key=lambda c: c["metric"])
        else:
            return None

    def search(self, template_mask, depth_image):
        self.currentTMask = template_mask
        self.currentDImg = depth_image
        self.total_pixels = self.currentDImg.size
        global_candidates = self.searchGlobal()
        print("Global search candidates:")
        for cand in global_candidates.values():
            print(f"  bt: {cand['bottom_threshold']}, slope: {cand['slope']}, metric: {cand['metric']}")
        local_minimas = self.findLocalMinima(global_candidates)
        best_candidate = self.searchLocal(local_minimas)
        self.currentTMask = None
        self.currentDImg = None
        return best_candidate

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
    searcher = ThresholdSearcher(num_local_minima=5, glob_step=5, local_search_radius=5)

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

        best_candidate = searcher.search(template_mask, depth_image)
        if best_candidate is None:
            print(f"No valid candidate found for {fname}")
        else:
            out_fname = "best_" + fname
            out_path = os.path.join(output_dir, out_fname)
            cv2.imwrite(out_path, best_candidate["masked_image"])
            print(f"Saved best candidate for {fname} as {out_fname} with metric {best_candidate['metric']}")

if __name__ == "__main__":
    main()
