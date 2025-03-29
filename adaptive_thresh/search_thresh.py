import cv2
import numpy as np
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../depth_pattern_analysis/gt_generation")
from search_threshs import ThresholdSearcher


def main():
    # Load images in grayscale.
    fname = "depth_image.jpg"
    depth_image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    if depth_image is None:
        raise IOError("Could not load depth_image.jpg. Check the file path.")

    template_mask = cv2.imread("edge_image.jpg", cv2.IMREAD_GRAYSCALE)
    if template_mask is None:
        raise IOError("Could not load edge_image.jpg. Check the file path.")
    if template_mask.shape != depth_image.shape:
        raise ValueError("edge_image.jpg must have the same dimensions as depth_image.jpg.")

    # Create a ThresholdSearcher object with desired parameters.
    searcher = ThresholdSearcher(glob_step=3, imwrite_local=True, log_all=True)

    output_dir = "masked_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    best_candidate = searcher.search(template_mask, depth_image, fname[:-4])
    if best_candidate is None:
        print(f"No valid candidate found for {fname}")
    else:
        out_fname = "best_" + fname
        out_path = os.path.join(output_dir, out_fname)
        cv2.imwrite(out_path, best_candidate["masked_image"])
        print(f"Saved best candidate for {fname} as {out_fname} with metric {best_candidate['metric']}")


if __name__ == "__main__":
    main()
