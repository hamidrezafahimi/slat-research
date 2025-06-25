import cv2
import numpy as np
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../lib")
from searchBasedDepthDiffuser import SearchBasedDepthDiffuser


def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py depth_image.jpg edge_image.jpg")
        sys.exit(1)

    depth_path = sys.argv[1]
    edge_path = sys.argv[2]

    # Load images in grayscale.
    depth_image = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    if depth_image is None:
        raise IOError(f"Could not load {depth_path}. Check the file path.")

    template_mask = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
    if template_mask is None:
        raise IOError(f"Could not load {edge_path}. Check the file path.")

    if template_mask.shape != depth_image.shape:
        raise ValueError("Both input images must have the same dimensions.")


    output_dir = "masked_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a ThresholdSearcher object with desired parameters.
    searcher = SearchBasedDepthDiffuser(outdir=output_dir)#, imwrite_local=True, log_all=True)

    import time
    t0 = time.time()
    searcher.diffuse(template_mask, depth_image, depth_path[:-4])
    print("time: ", time.time() - t0)
    print("-------------")

if __name__ == "__main__":
    main()
