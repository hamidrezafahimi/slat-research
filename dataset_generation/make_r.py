import cv2
import os
import numpy as np
import csv

# Directory containing the .jpg files
directory = '/home/hamid/viot3/output/put'

# List to store all rows of pixel data
all_pixel_data = []

# Iterate over all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.jpg'):
        image_path = os.path.join(directory, filename)
        print("processing: ", image_path)
        
        # Read the image in grayscale (assuming monochrome images)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Flatten the image to a 1D array (each row of the image becomes a single row in the CSV)
        # flattened_image = image.flatten()
        
        # image[:, column_index]
        # Append this flattened image data as a row
        for c in range(image.shape[1]):
            all_pixel_data.append(image[:, c])

# Convert the list to a numpy array for easier saving
all_pixel_data = np.array(all_pixel_data)

# Define the output CSV file path
output_csv = 'rmat.csv'

# Save the numpy array to a CSV file
np.savetxt(output_csv, all_pixel_data, delimiter=',', fmt='%d')

print(f"CSV file saved to: {output_csv}")
