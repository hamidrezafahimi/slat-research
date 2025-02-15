import cv2
import numpy as np

# Read the image
img = cv2.imread('sam2.png')


# Convert the image to grayscale (as thresholding works on single channel images)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding: any pixel value above the threshold will become white (255), else black (0)
threshold_value = 254  # Set the threshold value (adjust this based on your needs)
_, binary_img = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

# Convert the result back to RGB to maintain color channels, although it's now binary
binary_rgb_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2RGB)

# Save the resulting binary image
cv2.imwrite('binary_image_thresholded.jpg', binary_rgb_img)

# Display the resulting image
cv2.imshow('Binary Image with Thresholding', binary_rgb_img)
cv2.waitKey(0)
cv2.destroyAllWindows()