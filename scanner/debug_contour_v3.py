# Save as scanner/debug_contours_v3.py

import cv2
import numpy as np

img = cv2.imread("sample_images/1.jpg")

# Resize
height, width = img.shape[:2]
ratio = 800 / height
resized = cv2.resize(img, (int(width * ratio), 800))

gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (11, 11), 0)

# Try multiple approaches and pick the best one

def try_find_document(resized, gray, blurred):
    """Try multiple methods to find the document contour."""
    image_area = resized.shape[0] * resized.shape[1]
    
    # METHOD 1: Otsu threshold + morphology
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # METHOD 2: Canny + heavy dilation
    edges = cv2.Canny(blurred, 30, 100)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=3)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=3)
    
    for method_name, binary in [("threshold", thresh), ("canny", edges)]:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contour in contours[:5]:
            area = cv2.contourArea(contour)
            if area < 0.1 * image_area:
                continue
            
            # Get convex hull first (removes concavities)
            hull = cv2.convexHull(contour)
            
            # Then approximate to polygon
            perimeter = cv2.arcLength(hull, True)
            
            # Try different epsilon values for approximation
            for eps_mult in [0.02, 0.03, 0.04, 0.05]:
                approx = cv2.approxPolyDP(hull, eps_mult * perimeter, True)
                
                if len(approx) == 4:
                    print(f"Found with {method_name}, epsilon={eps_mult}, area={area/image_area*100:.1f}%")
                    return approx
    
    # FALLBACK: If no 4-point contour found, use the largest contour's
    # minimum area rectangle
    print("Fallback: Using minimum area rectangle of largest contour")
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    if contours:
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        return box.astype(np.int32).reshape(-1, 1, 2)
    
    return None


contour = try_find_document(resized, gray, blurred)

if contour is not None:
    debug_img = resized.copy()
    cv2.drawContours(debug_img, [contour], -1, (0, 255, 0), 3)
    
    # Draw corner points
    pts = contour.reshape(-1, 2)
    for pt in pts:
        cv2.circle(debug_img, tuple(pt), 8, (0, 0, 255), -1)
    
    cv2.imwrite("output/debug_contours_v3.jpg", debug_img)
    
    # Do the perspective transform
    pts_float = pts.astype(np.float32)
    
    # Order corners
    s = pts_float.sum(axis=1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts_float[np.argmin(s)]
    ordered[2] = pts_float[np.argmax(s)]
    d = np.diff(pts_float, axis=1)
    ordered[1] = pts_float[np.argmin(d)]
    ordered[3] = pts_float[np.argmax(d)]
    
    # Scale back to original image
    ordered = ordered / ratio
    
    # Output dimensions
    width_top = np.linalg.norm(ordered[1] - ordered[0])
    width_bottom = np.linalg.norm(ordered[2] - ordered[3])
    max_width = int(max(width_top, width_bottom))
    
    height_left = np.linalg.norm(ordered[3] - ordered[0])
    height_right = np.linalg.norm(ordered[2] - ordered[1])
    max_height = int(max(height_left, height_right))
    
    dst = np.array([[0,0], [max_width-1,0], [max_width-1,max_height-1], [0,max_height-1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    scanned = cv2.warpPerspective(img, matrix, (max_width, max_height))
    
    cv2.imwrite("output/5_scanned.jpg", scanned)
    print(f"Scanned output: {max_width}x{max_height}")
    print("Saved to output/")
else:
    print("Failed to find document")