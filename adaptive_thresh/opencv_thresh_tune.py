import cv2
import sys
import os

def nothing(x):
    pass

# Check if image path is provided
if len(sys.argv) < 2:
    print("Usage: python threshold_tuner.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

# Check if file exists
if not os.path.exists(image_path):
    print(f"Error: File '{image_path}' not found.")
    sys.exit(1)

# Load image in grayscale
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Could not read the image.")
    sys.exit(1)

# Create window
cv2.namedWindow('Threshold Tuner')

# Create trackbar
cv2.createTrackbar('Threshold', 'Threshold Tuner', 127, 255, nothing)

while True:
    thresh_val = cv2.getTrackbarPos('Threshold', 'Threshold Tuner')
    _, thresh_img = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY)
    cv2.imshow('Threshold Tuner', thresh_img)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif key == ord('s'):
        filename = f'threshold_{thresh_val}.png'
        cv2.imwrite(filename, thresh_img)
        print(f"Saved as {filename}")

cv2.destroyAllWindows()
