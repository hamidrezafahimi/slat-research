import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../lib")
from mapper3D import Mapper3D
import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 5:
        print("Usage: python script.py <depth_image.csv> <background_image.jpg> <color_image.jpg> <meta_txt>")
        sys.exit(1)

    depth_path, bg_path, color_path, meta_file = sys.argv[1:5]

    # Extract the base name (without extension)
    base_name = os.path.splitext(os.path.basename(depth_path))[0]

    # Load meta file and parse altitude/pitch for the base_name
    with open(meta_file, 'r') as f:
        meta_lines = [line.strip().split(',') for line in f if line.strip()]
    meta_data = {
        parts[0]: {'altitude': float(parts[2]), 'pitch': float(parts[4])}
        for parts in meta_lines
    }

    if base_name not in meta_data:
        print(f"Metadata for {base_name} not found in {meta_file}")
        sys.exit(1)

    altitude = meta_data[base_name]['altitude']
    pitch = meta_data[base_name]['pitch']

    # Load images
    depth_image = np.loadtxt(depth_path, delimiter=',', dtype=np.float32)
    bg_image = 255 - cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE)
    color_img = cv2.imread(color_path)

    # Process using Mapper3D
    mapper = Mapper3D()
    mapper.process(depth_image, bg_image, color_img, pitch, altitude, vis=True, plot=True)
    plt.show()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()