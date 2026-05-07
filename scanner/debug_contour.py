# Save as scanner/debug_contours.py

import cv2
import numpy as np

img = cv2.imread("sample_images/1.jpg")

# Resize
height, width = img.shape[:2]
ratio = 800 / height
resized = cv2.resize(img, (int(width * ratio), 800))

# Preprocess
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)
kernel = np.ones((3, 3), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=2)  # more dilation to close gaps

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

image_area = resized.shape[0] * resized.shape[1]
debug_img = resized.copy()

print(f"Total contours found: {len(contours)}")
print(f"Image area: {image_area}")
print(f"\nTop 10 contours:")

for i, contour in enumerate(contours[:10]):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    
    print(f"  #{i}: area={area:.0f} ({area/image_area*100:.1f}%), vertices={len(approx)}")
    
    # Draw top 5 contours in different colors
    if i < 5:
        color = [(0,255,0), (255,0,0), (0,0,255), (255,255,0), (0,255,255)][i]
        cv2.drawContours(debug_img, [approx], -1, color, 2)

cv2.imwrite("output/debug_contours.jpg", debug_img)
print("\nSaved debug image to output/debug_contours.jpg")