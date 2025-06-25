import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../lib")
from mapper3D import Mapper3D
import cv2
import numpy as np

def main():
    if len(sys.argv) < 4:
        print("Usage: python script.py <metric_depth.csv> <metric_bg_path.csv> <background_image.jpg> <color_image.jpg> <mask.png>")
        sys.exit(1)

    metric_depth_path, metric_bg_path, color_path, mask_path = sys.argv[1:6]

    roll = 0
    pitch = -0.15
    yaw = 0
    z = 1.7

    metric_depth = np.loadtxt(metric_depth_path, delimiter=',', dtype=np.float32)[200:,:]
    metric_bg = np.loadtxt(metric_bg_path, delimiter=',', dtype=np.float32)[200:,:]
    color_img = cv2.imread(color_path)[200:,:]
    mask_img = 255 - cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)[200:,:]

    mapper = Mapper3D(vis=True, plot=True, color_mode='image')
    mapper.process(metric_depth, metric_bg, color_img, mask_img, roll, pitch, yaw, z, 80.0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
