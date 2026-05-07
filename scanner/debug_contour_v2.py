# Save as scanner/debug_contours_v2.py

import cv2
import numpy as np

img = cv2.imread("sample_images/1.jpg")

# Resize
height, width = img.shape[:2]
ratio = 800 / height
resized = cv2.resize(img, (int(width * ratio), 800))

# APPROACH 2: Use thresholding instead of Canny
# The document is white/light, background is dark
# Thresholding separates light from dark directly
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (11, 11), 0)

# Threshold: pixels brighter than threshold become white, rest become black
# This should isolate the white document from the dark background
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Clean up with morphological operations
# Close: fills small holes inside the document
# Open: removes small noise outside the document
kernel = np.ones((5, 5), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

cv2.imwrite("output/debug_thresh.jpg", thresh)

# Find contours on the thresholded image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

image_area = resized.shape[0] * resized.shape[1]
debug_img = resized.copy()

print(f"Total contours found: {len(contours)}")
print(f"\nTop 5 contours:")

for i, contour in enumerate(contours[:5]):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    
    print(f"  #{i}: area={area:.0f} ({area/image_area*100:.1f}%), vertices={len(approx)}")
    
    if i == 0:
        # Draw the largest contour and its approximation
        cv2.drawContours(debug_img, [contour], -1, (0, 255, 0), 2)
        cv2.drawContours(debug_img, [approx], -1, (0, 0, 255), 3)

cv2.imwrite("output/debug_contours_v2.jpg", debug_img)
print("\nSaved debug images to output/")