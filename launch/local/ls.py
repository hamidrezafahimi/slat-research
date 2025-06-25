import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../lib")
from mapper3D import Mapper3D
import cv2
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <metric_depth.csv> depth.jpg")
        sys.exit(1)

    metric_depth_path, depth_image_path = sys.argv[1:3]

    depth_image = 255 - cv2.imread(depth_image_path, cv2.IMREAD_GRAYSCALE)
    metric_depth = np.loadtxt(metric_depth_path, delimiter=',', dtype=np.float32)

    # depth_image = depth_image[80:, :]
    # metric_depth = metric_depth[80:, :]

    depth_image1 = depth_image.astype(np.float32) * (255.0 / np.max(depth_image.astype(np.float32)))
    depth_image = depth_image1.astype(np.uint8)

    # metric_depth -= np.min(metric_depth)
    metric_depth *= 255.0 / np.max(metric_depth)

    # cv2.imwrite("/home/hamid/asfd.png", (255 - metric_depth).astype(np.uint8))
    np.savetxt("/home/hamid/asfd.csv", metric_depth, delimiter=',')

    print(np.min(metric_depth), np.max(metric_depth), np.min(depth_image), np.max(depth_image))
    cv2.imshow("metric", metric_depth.astype(np.uint8))
    cv2.imshow("visual", depth_image)
    cv2.waitKey()



if __name__ == "__main__":
    main()
