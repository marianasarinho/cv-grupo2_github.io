import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
  
# Read image
img = cv.imread('foto3.jpg', cv.IMREAD_COLOR) 

# Convert the image to gray-scale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Find the edges in the image using canny detector
edges = cv.Canny(gray, 50, 200)

# Detect points that form a line
lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=10, maxLineGap=250)

# Draw lines on the image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

# Show result
cv.imshow("Result Image", img)