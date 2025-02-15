import cv2
import numpy as np
import os

# Global variables
points = []  # Points buffer to store contour points
drawing_mask = None  # The current mask image being drawn
original_image = None  # Original image
mask = None  # Final mask to store all contours

# Mouse callback function for point annotation
def draw_points(event, x, y, flags, param):
    global points, drawing_mask

    if event == cv2.EVENT_LBUTTONDOWN:
        # Add clicked point to buffer
        points.append((x, y))
        print(f"Point {len(points)} added at: ({x}, {y})")

    # Update display with current points drawn, including all previous contours
    if len(points) > 1:
        temp_mask = mask.copy()  # Use the full mask with previous contours
        # Draw all points so far
        cv2.polylines(temp_mask, [np.array(points, dtype=np.int32)], isClosed=False, color=0, thickness=2)
        overlay_image = cv2.addWeighted(original_image, 1, temp_mask, 0.5, 0)
        cv2.imshow("Image", overlay_image)

# Function to finalize and save the contour to the mask
def finalize_contour():
    global mask, points
    if len(points) > 1:
        # Draw the current contour and fill it in the mask
        cv2.polylines(mask, [np.array(points, dtype=np.int32)], isClosed=True, color=0, thickness=2)
        cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], color=0)
        print(f"Contour added with {len(points)} points.")
    points.clear()  # Clear points buffer after finishing the contour

    # Display all contours so far
    overlay_image = cv2.addWeighted(original_image, 1, mask, 0.5, 0)
    cv2.imshow("Image", overlay_image)

def save_masked_image(path):
    result_image = np.ones_like(mask) * 255
    result_image[mask == 0] = 0
    cv2.imwrite(path + "_mask.png", result_image)
    print("Masked image saved as 'masked_image.png'.")

def main():
    global mask, original_image, drawing_mask

    # Directory containing the .jpg files
    directory = '/home/hamid/thing/output/hj'

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            image_path = os.path.join(directory, filename)

            # Read the input image
            original_image = cv2.imread(image_path)
            if original_image is None:
                print("Error: Image not found.")
                return

            # Create a blank mask with the same dimensions as the input image (255 = white, 0 = black)
            mask = np.ones_like(original_image, dtype=np.uint8) * 255
            drawing_mask = mask.copy()

            # Show the image and set the mouse callback
            cv2.imshow("Image", original_image)
            cv2.setMouseCallback("Image", draw_points)

            while True:
                key = cv2.waitKey(1) & 0xFF

                # Press 's' to save the mask
                if key == ord('s'):
                    save_masked_image(image_path[:-4])

                # Press 'm' to finalize the current contour and start a new one
                elif key == ord('m'):
                    finalize_contour()

                # Press 'q' to exit the program
                elif key == ord('q'):
                    break

            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
