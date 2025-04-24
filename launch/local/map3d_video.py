import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../lib")
from mapper3D import Mapper3D
import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Usage: python script.py <depth_dir> <background_dir> <color_dir> <meta_txt>")
        sys.exit(1)

    depth_dir, bg_dir, color_dir, meta_file = sys.argv[1:5]

    with open(meta_file, 'r') as f:
        meta_lines = [line.strip().split(',') for line in f if line.strip()]
    meta_data = {
        parts[0]: {'altitude': float(parts[2]), 'pitch': float(parts[4])}
        for parts in meta_lines
    }

    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(".csv")])
    bg_files = sorted([f for f in os.listdir(bg_dir) if f.endswith(".jpg")])
    color_files = sorted([f for f in os.listdir(color_dir) if f.endswith(".jpg")])
    names_meta = list(meta_data.keys())

    mapper = Mapper3D()

    for dfile, bfile, cfile, base_name in zip(depth_files, bg_files, color_files, names_meta):
        assert os.path.splitext(dfile)[0] == os.path.splitext(bfile)[0] == os.path.splitext(cfile)[0] == base_name

        pitch = meta_data[base_name]['pitch']
        altitude = meta_data[base_name]['altitude']

        depth_image = np.loadtxt(os.path.join(depth_dir, dfile), delimiter=',', dtype=np.float32)
        bg_image = 255 - cv2.imread(os.path.join(bg_dir, bfile), cv2.IMREAD_GRAYSCALE)
        color_img = cv2.imread(os.path.join(color_dir, cfile))

        if mapper.process(depth_image, bg_image, color_img, pitch, altitude, vis=True, plot=True):
            plt.pause(0.001)
        else:
            break

    cv2.destroyAllWindows()

