import cv2
import os
import numpy as np
import csv
import time
from sampl3d import scale_d_surface
import sys
sys.path.append("/media/hamid/Workspace/thing/lib")
from lines import calcFlatGroundDepth, expand_segments

# List to store all rows of pixel data
output_txt_file = "output_file_paths.txt"  # Path to save the filenames
tilt_file = "/media/hamid/Workspace/viot3/pm1states/camera_states.txt"

with open(tilt_file, "r") as file:
    # Read all lines
    tlines = file.readlines()
    # print(lines[234])


# Directories containing images
directory1 = '/media/hamid/Workspace/viot3/output/put'
directory2 = '/media/hamid/Workspace/viot3/sam/masks'

# Function to extract the numeric part of filenames for sorting
def extract_number(filename):
    # Extract the number from the filename (e.g., '00000234' from '00000234.jpg')
    return int(''.join(filter(str.isdigit, filename)))

# Get sorted lists of filenames from both directories
files1 = sorted(
    [f for f in os.listdir(directory1) if f.endswith('.jpg')],
    key=extract_number
)
files2 = sorted(
    [f for f in os.listdir(directory2) if f.endswith('.jpg')],
    key=lambda x: extract_number(x.split("_masks_jj")[0])  # Remove the "_masks_jj" suffix before sorting
)

# Ensure both directories have the same number of images
assert len(files1) == len(files2), "Mismatch in the number of files between the two directories!"

# Define the output CSV file path
m_csv = 'mmat.csv'
h_csv = 'hmat.csv'
r_csv = 'rmat.csv'
f_csv = 'fmat.csv'
d_csv = 'dmat.csv'
ddd_csv = 'dddmat.csv'
rdd_csv = 'rddmat.csv'

# Open the text file for appending the paths
with open(m_csv, "a") as m_file, open(h_csv, "a") as h_file, open(r_csv, "a") as r_file, \
    open(f_csv, "a") as f_file, open(d_csv, "a") as d_file, open(ddd_csv, "a") as ddd_file, \
    open(rdd_csv, "a") as rdd_file, open(output_txt_file, "w") as file:
    # Loop through the sorted filenames and process corresponding images
    for k, (file1, file2) in enumerate(zip(files1, files2)):
        # Check if the numeric part of the filenames match
        if extract_number(file1) != extract_number(file2.split("_masks_jj")[0]):
            raise ValueError(f"Mismatched filenames: {file1} and {file2}")
        
        # Full paths to the images
        if file1.endswith('.jpg') and file2.endswith('.jpg'):
            image1_path = os.path.join(directory1, file1)
            image2_path = os.path.join(directory2, file2)
            
            print(f"Processing: {image1_path} and {image2_path}")
            # Add your processing code here

            # Read the image in grayscale (assuming monochrome images)
            depth = 255 - cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
            gep_depth = calcFlatGroundDepth(np.float32(tlines[k].strip()))
            h = scale_d_surface(depth, gep_depth)
            residual = h - gep_depth
            # hddot = compute_second_derivative(h)
            # Compute the Laplacian (2nd derivative)
            res_lap = cv2.Laplacian(residual, ddepth=cv2.CV_64F, ksize=3)
            depth_lap = cv2.Laplacian(depth, ddepth=cv2.CV_64F, ksize=3)
            # Optionally normalize for visualization
            # depth_lap_n = cv2.normalize(depth_lap, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # res_lap_n = cv2.normalize(res_lap, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            mask = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
            if depth.shape != mask.shape:
                raise ValueError("The dimensions of depth and mask do not match.")

            ret, thresh1 = cv2.threshold(mask,230,255,cv2.THRESH_BINARY)        
            # e_mask = expand_segments(depth_lap, thresh1, 10)
            # print(np.int32(thresh1[:, 32] > 0))
            # print(thresh1[:, 32] )
            
            # cv2.imshow("f", gep_depth)
            # cv2.imshow("d", depth)
            # cv2.imshow("res", residual)
            # cv2.imshow("ddd", depth_lap)
            # cv2.imshow("mask", thresh1)
            # cv2.imshow("expanded mask", e_mask)
            # cv2.waitKey()
            # time.sleep(100)

            # for c in range(depth.shape[1]):
            #     # h_data = scale_d_surface(depth[:, c], gep_depth[:, c])
            #     np.savetxt(m_file, [np.int32(thresh1[:, c] > 0)], delimiter=',', fmt='%d')
            #     np.savetxt(d_file, [depth[:, c]], delimiter=',', fmt='%d')
            #     np.savetxt(r_file, [residual[:, c]], delimiter=',', fmt='%d')
            #     np.savetxt(f_file, [gep_depth[:, c]], delimiter=',', fmt='%d')
            #     np.savetxt(h_file, [h[:, c]], delimiter=',', fmt='%d')
            #     np.savetxt(ddd_file, [depth_lap[:, c]], delimiter=',', fmt='%d')
            #     np.savetxt(rdd_file, [res_lap[:, c]], delimiter=',', fmt='%d')
            #     file.write(f"{image1_path}, {image2_path}\n")
