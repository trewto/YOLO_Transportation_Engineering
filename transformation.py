import cv2
import numpy as np

# Load the image
image_path = 'your_image.jpg'  # Replace with your image path
image = cv2.imread(image_path)
height, width = image.shape[:2]

# Define the points of the quadrilateral (source points)
# You should manually choose these points in the image
# Example points (x, y): top-left, top-right, bottom-right, bottom-left
src_points = np.array([[100, 200], [400, 200], [450, 400], [50, 400]], dtype='float32')

# Define the points of the rectangle (destination points)
dst_points = np.array([[0, 0], [400, 0], [400, 300], [0, 300]], dtype='float32')

# Get the transformation matrix
matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# Apply the perspective transformation
warped_image = cv2.warpPerspective(image, matrix, (400, 300))

# Show the original and warped images
cv2.imshow('Original Image', image)
cv2.imshow('Warped Image', warped_image)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
