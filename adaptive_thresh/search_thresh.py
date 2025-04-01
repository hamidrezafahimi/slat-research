import cv2
import numpy as np
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../depth_pattern_analysis/gt_generation")
from search_threshs import AutoDepthBGSubtractor


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
    searcher = AutoDepthBGSubtractor()#, imwrite_local=True, log_all=True)

    output_dir = "masked_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    import time
    t0 = time.time()
    searcher.search(template_mask, depth_image, fname[:-4], output_dir)
    print("time: ", time.time() - t0)
    print("-------------")

if __name__ == "__main__":
    main()
