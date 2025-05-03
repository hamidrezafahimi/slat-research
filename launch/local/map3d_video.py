import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../lib")
from mapper3D import Mapper3D
import cv2
import numpy as np
from occupancy_grid3d import OccupancyGrid3D
from transforms import transform_camera_to_earth


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Usage: python script.py <depth_dir> <background_dir> <color_dir> <meta_txt>")
        sys.exit(1)

    depth_dir, bg_dir, color_dir, meta_file = sys.argv[1:5]

    with open(meta_file, 'r') as f:
        meta_lines = [line.strip().split(',') for line in f if line.strip()]
    meta_data = {
        parts[0]: {'x': float(parts[2]), 'y': float(parts[3]), 'z': float(parts[4]),
                   'roll': float(parts[5]), 'pitch': float(parts[6]), 'yaw': float(parts[7])}
        for parts in meta_lines
    }

    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(".csv")])
    bg_files = sorted([f for f in os.listdir(bg_dir) if f.endswith(".jpg")])
    color_files = sorted([f for f in os.listdir(color_dir) if f.endswith(".jpg")])
    names_meta = list(meta_data.keys())

    mapper = Mapper3D(block_plot=False)
    og = OccupancyGrid3D(cell_size=0.10)   # 10 cm voxels
    origin_world = None                    # will be filled on first frame

    k = 0
    for dfile, bfile, cfile, base_name in zip(depth_files, bg_files, color_files, names_meta):
        assert os.path.splitext(dfile)[0] == os.path.splitext(bfile)[0] == os.path.splitext(cfile)[0] == base_name

        roll = meta_data[base_name]['roll']
        pitch = meta_data[base_name]['pitch']
        yaw = meta_data[base_name]['yaw']
        x = meta_data[base_name]['x']
        y = meta_data[base_name]['y']
        z = meta_data[base_name]['z']

        depth_image = np.loadtxt(os.path.join(depth_dir, dfile), delimiter=',', dtype=np.float32)
        bg_image = 255 - cv2.imread(os.path.join(bg_dir, bfile), cv2.IMREAD_GRAYSCALE)
        color_img = cv2.imread(os.path.join(color_dir, cfile))

        fimgpc, ret = mapper.process(depth_image, bg_image, color_img, pitch, z)

        pc_cam  = fimgpc.reshape(-1, 3)
        t_c2w   = np.array([x, y, z], dtype=np.float32)

        if origin_world is None:              # set world origin once
            origin_world = t_c2w.copy()

        points_earth = transform_camera_to_earth(pc_cam, roll, pitch, yaw)
        # pc_world = points_earth + (t_c2w - origin_world)
        pc_world = points_earth
        og.update(pc_world, k)
        # og.visualize_open3d(min_count=3)      # interactive

        k += 1
        if not ret:
            break

    og.save_pcd("demo_occ.pcd", min_count=1)
    # cv2.destroyAllWindows()

