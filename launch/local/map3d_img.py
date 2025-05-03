import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../lib")
from mapper3D import Mapper3D
import cv2
import numpy as np

def main():
    if len(sys.argv) < 5:
        print("Usage: python script.py <depth_image.csv> <background_image.jpg> <color_image.jpg> <meta_txt>")
        sys.exit(1)

    depth_path, bg_path, color_path, meta_file = sys.argv[1:5]
    base_name = os.path.splitext(os.path.basename(depth_path))[0]

    with open(meta_file, 'r') as f:
        meta_lines = [line.strip().split(',') for line in f if line.strip()]
    meta_data = {
        parts[0]: {'x': float(parts[2]), 'y': float(parts[3]), 'z': float(parts[4]),
                   'roll': float(parts[5]), 'pitch': float(parts[6]), 'yaw': float(parts[7])}
        for parts in meta_lines
    }

    if base_name not in meta_data:
        print(f"Metadata for {base_name} not found in {meta_file}")
        sys.exit(1)

    roll = meta_data[base_name]['roll']
    pitch = meta_data[base_name]['pitch']
    yaw = meta_data[base_name]['yaw']
    x = meta_data[base_name]['x']
    y = meta_data[base_name]['y']
    z = meta_data[base_name]['z']

    depth_image = np.loadtxt(depth_path, delimiter=',', dtype=np.float32)
    bg_image = 255 - cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE)
    color_img = cv2.imread(color_path)

    mapper = Mapper3D(vis=True, plot=True)
    mapper.process(depth_image, bg_image, color_img, roll, pitch, yaw, z)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
