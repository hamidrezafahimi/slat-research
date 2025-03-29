import cv2
import sys

def nothing(x):
    pass

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py path_to_image")
        return

    image_path = sys.argv[1]
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: could not read image {image_path}")
        return

    cv2.namedWindow("Edge Detection")

    # Create trackbars for threshold1 and threshold2 (for Canny)
    cv2.createTrackbar("Threshold1", "Edge Detection", 50, 500, nothing)
    cv2.createTrackbar("Threshold2", "Edge Detection", 150, 500, nothing)

    while True:
        # Get trackbar positions
        t1 = cv2.getTrackbarPos("Threshold1", "Edge Detection")
        t2 = cv2.getTrackbarPos("Threshold2", "Edge Detection")

        # Apply Canny edge detection
        edges = cv2.Canny(img, t1, t2)

        cv2.imshow("Edge Detection", edges)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            cv2.imwrite("edge_image.png", edges)
            print("Saved edge_image.png")
            break
        elif key == 27:  # ESC key
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
