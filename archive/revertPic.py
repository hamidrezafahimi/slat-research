import cv2

# Read the image in grayscale mode
image = cv2.imread('00000044.jpg', cv2.IMREAD_GRAYSCALE)

# Reverse (invert) the image
inverted_image = 255 - image  # Alternative: cv2.bitwise_not(image)

# Display the original and inverted image
cv2.imshow('Original Image', image)
cv2.imshow('Inverted Image', inverted_image)

# Save the inverted image
cv2.imwrite('inverted_image.png', inverted_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
