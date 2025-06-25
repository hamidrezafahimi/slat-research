import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../lib")
from mapper3D import Mapper3D
import cv2
import numpy as np

def main():
    if len(sys.argv) < 4:
        print("Usage: python script.py <metric_depth.csv> <background_image.jpg> <color_image.jpg> <mask.png>")
        sys.exit(1)

    metric_depth_path, bg_path, color_path, mask_path = sys.argv[1:6]

    roll = 0
    pitch = -0.78
    yaw = 0
    z = 10

    metric_depth = np.loadtxt(metric_depth_path, delimiter=',', dtype=np.float32)
    bg_image = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE)
    color_img = cv2.imread(color_path)
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    mapper = Mapper3D(vis=True, plot=True, color_mode='image')
    mapper.process(metric_depth, bg_image, color_img, mask_img, roll, pitch, yaw, z, 66.0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
