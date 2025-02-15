import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read the image in grayscale
image_path = "flood_fill.png"  # Change this to the actual image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Resize while keeping aspect ratio
height, width = image.shape
new_height = 32
new_width = int((new_height / height) * width)
resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

# Apply threshold
_, binary_image = cv2.threshold(resized_image, 100, 1, cv2.THRESH_BINARY)

# Get new dimensions
h, w = binary_image.shape

# Generate coordinates
x, y = np.meshgrid(range(w), range(h))
x, y = x.flatten(), y.flatten()
z = np.zeros_like(x)

# Colors based on thresholded values
colors = np.where(binary_image.flatten() == 1, 'red', 'blue')

# Define the fixed point above the XY plane
# fixed_point = np.array([w // 2, h // 2, 20])  # Centered above the plane
fixed_point = np.array([25, 50, -9.4])  # Centered above the plane


# Create 3D plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(x, y, z, c=colors, marker='o')

# Draw lines from the fixed point to blue points (0 values)
for xi, yi, zi, ci in zip(x, y, z, colors):
    if ci == 'blue':  # Only draw lines to blue points
        ax.plot([fixed_point[0], xi], [fixed_point[1], yi], [fixed_point[2], zi], color='blue', alpha=0.5)

# Labels and formatting
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Scatter Plot from Image")
ax.view_init(elev=30, azim=45)

plt.show()
