import numpy as np
import os
import csv
import time
import cv2
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../lib")
from searchBasedDepthDiffuser import SearchBasedDepthDiffuser

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

    searcher = SearchBasedDepthDiffuser()

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
