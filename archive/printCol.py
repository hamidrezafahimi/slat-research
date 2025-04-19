import cv2

# Read the image in grayscale mode
image = cv2.imread('inverted_image.png', cv2.IMREAD_GRAYSCALE)

print(image.shape)
for k in range(image.shape[0]):
    print(image[k, 276], ",")
