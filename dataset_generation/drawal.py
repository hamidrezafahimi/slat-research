import cv2 as cv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# Load the matrices from CSV files
dmat = pd.read_csv('dmat.csv', header=None)
hmat = pd.read_csv('hmat.csv', header=None)
rmat = pd.read_csv('rmat.csv', header=None)
mmat = pd.read_csv('mmat.csv', header=None)
fmat = pd.read_csv('fmat.csv', header=None)
dddmat = pd.read_csv('dddmat.csv', header=None)
rddmat = pd.read_csv('rddmat.csv', header=None)
output_txt_file = "output_file_paths.txt"


# Ensure the matrices have the same dimensions
if rmat.shape != mmat.shape:
    raise ValueError("The dimensions of rmat.csv and mmat.csv do not match.")

with open(output_txt_file, "r") as file:
    # Read all lines
    lines = file.readlines()
    # print(lines[234])

while True:
    # Randomly select a row index
    # random_row_index = np.random.randint(0, rmat.shape[0])
    random_row_index = 19196

    # Extract the selected row from rmat and mmat
    dmat_row = dmat.iloc[random_row_index]
    hmat_row = hmat.iloc[random_row_index]
    rmat_row = rmat.iloc[random_row_index]
    mmat_row = mmat.iloc[random_row_index]
    fmat_row = fmat.iloc[random_row_index]
    dddmat_row = dddmat.iloc[random_row_index]
    rddmat_row = rddmat.iloc[random_row_index]

    # Validate the row number
    if random_row_index < 1 or random_row_index > len(lines):
        raise ValueError(f"Row number {random_row_index} is out of range. File has {len(lines)} rows.")
    
    # Choose a random line
    random_line = lines[random_row_index].strip()
    
    # Split the line into image1_path and image2_path
    image1_path, image2_path = random_line.split(", ")
    print(image1_path, image2_path)
    # Load the image
    image1 = 255 - cv.imread(image1_path, cv.IMREAD_GRAYSCALE)
    image2 = cv.imread(image2_path)
    ret, thresh1 = cv.threshold(image2,230,255,cv.THRESH_BINARY)
    cv.line(image1, (random_row_index % image1.shape[1], 0), (random_row_index % image1.shape[1], image1.shape[0]-1), 255, 2)
    cv.line(image2, (random_row_index % image2.shape[1], 0), (random_row_index % image2.shape[1], image2.shape[0]-1), 255, 2)
    cv.imshow("depth", image1)
    cv.imshow("mask", image2)
    cv.imshow("mask thresh", thresh1)
    cv.waitKey()

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Iterate through the row to plot points based on mmat values
    for i, (d_val, r_val, m_val, f_val, h_val, ddd_val, rdd_val) in \
        enumerate(zip(dmat_row, rmat_row, mmat_row, fmat_row, hmat_row, dddmat_row, rddmat_row)):
        # colord = 'red' if m_val == 1 else 'blue'
        # colorr = 'yellow' if m_val == 1 else 'green'
        # colorh = 'brown' if m_val == 1 else 'grey'
        # plt.plot(i, h_val, color=colorh)
        # plt.plot(i, d_val, color=colord)
        # plt.plot(i, r_val, color=colorr)
        # plt.plot(i, f_val, color='black')
        if i > 0:
            colord = 'blue' if m_val == 1 else 'red'
            colorr = 'green' if m_val == 1 else 'red'
            # colorh = 'grey' if m_val == 1 else 'grey'
            # colord = 'blue' if m_val == 1 else 'red'
            # colorr = 'green' if m_val == 1 else 'red'
            colorh = 'grey' if m_val == 1 else 'red'
            plt.plot([i-1, i], [last_h, h_val], color=colorh, label='H(x)')
            plt.plot([i-1, i], [last_d, d_val], color=colord, label='D(x)')
            plt.plot([i-1, i], [last_r, r_val], color=colorr, label='R(x)')
            plt.plot([i-1, i], [last_f, f_val], color='black', label='F(x)')
            # plt.plot([i-1, i], [last_ddd, ddd_val], color='cyan')
            # plt.plot([i-1, i], [last_rdd, rdd_val], color='pink')
        last_d = d_val
        last_r = r_val
        last_f = f_val
        last_h = h_val
        last_ddd = ddd_val
        last_rdd = rdd_val
        # plt.scatter(i, f_val, color='black')

    # Label the axes and add a title
    plt.xlabel('Index (Row Number)')
    plt.ylabel('Value')
    plt.title(f'Normalized Proximity to Camera (data: {random_row_index})')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys())
    plt.grid(True, linestyle='--', alpha=0.6)

    # Show the plot
    plt.show()
