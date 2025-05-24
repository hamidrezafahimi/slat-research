import cv2

# Initialize variables to store the coordinates of the mouse clicks and their IDs
positive_points = []  # List to store positive (green) points
negative_points = []  # List to store negative (red) points
point_id = 0

# Mouse callback function to capture the (x, y) coordinates and draw points
def click_event(event, x, y, flags, param):
    global point_id

    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button (positive point, green)
        print(f"Positive point id = {point_id}, Clicked at [{x}, {y}]")
        
        # Add the point to the positive points list
        positive_points.append(f"[{x}, {y}], ")

        # Draw a circle at the clicked position (Green)
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # Green circle

        # Put the ID text near the point
        cv2.putText(img, str(point_id), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Increment the ID for the next point
        point_id += 1

        # Display the updated image
        cv2.imshow('Image', img)

    elif event == cv2.EVENT_RBUTTONDOWN:  # Right mouse button (negative point, red)
        print(f"Negative point id = {point_id}, Clicked at [{x}, {y}]")
        
        # Add the point to the negative points list
        negative_points.append(f"[{x}, {y}], ")

        # Draw a circle at the clicked position (Red)
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # Red circle

        # Put the ID text near the point
        cv2.putText(img, str(point_id), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Increment the ID for the next point
        point_id += 1

        # Display the updated image
        cv2.imshow('Image', img)

# Function to reset points when 'R' key is pressed
def reset_points():
    global positive_points, negative_points, img, point_id
    # Print positive and negative points
    print ("[", end="")
    for p in positive_points:
        print (p, end="")
    for p in negative_points:
        print (p, end="")
    print ("]")
    print ("-----")
    print ("[", end="")
    for p in positive_points:
        print ("1, ", end="")
    for p in negative_points:
        print ("0, ", end="")
    print ("]")

    # Reset the image to the original state
    img = cv2.imread('00000001.jpg')
    
    # Clear the lists of points and reset the point_id
    positive_points = []
    negative_points = []
    point_id = 0

    # Display the reset image
    cv2.imshow('Image', img)

# Load the image
img = cv2.imread('00000001.jpg')

# Display the image in a window
cv2.imshow('Image', img)

# Set the mouse callback function
cv2.setMouseCallback('Image', click_event)

while True:
    # Wait for keypress
    key = cv2.waitKey(1) & 0xFF
    
    # If 'r' key is pressed, reset the points
    if key == ord('r'):  
        reset_points()
    
    # If 'ESC' key is pressed, break the loop
    elif key == 27:  # ESC key to exit
        break

# Destroy all OpenCV windows after the loop ends
cv2.destroyAllWindows()
