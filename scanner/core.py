import cv2
import numpy as np

def resize_image(image, max_height=800):
    """
    Resize image to a manageable size for processing.
    We keep the original for final output but work on a smaller version
    for speed. 800px height is enough for accurate edge detection.
    """
    height, width = image.shape[:2]
    if height > max_height:
        ratio = max_height / height
        new_width = int(width * ratio)
        resized = cv2.resize(image, (new_width, max_height))
        return resized, ratio
    return image, 1.0


def preprocess(image):
    """
    Prepare image for edge detection.
    
    Why each step:
    - Grayscale: edges are about intensity changes, color doesn't help
    - Gaussian blur: removes noise that would create false edges
    - The kernel size (5,5) controls how much smoothing — too little 
      leaves noise, too much blurs real edges
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred


def detect_edges(blurred):
    """
    Canny edge detection finds pixels where intensity changes sharply.
    
    The two thresholds (50, 150) control sensitivity:
    - Below 50: definitely NOT an edge (ignored)
    - Above 150: definitely IS an edge (kept)
    - Between 50-150: kept only if connected to a strong edge
    
    This two-threshold approach prevents noise from being detected
    as edges while keeping weak but real edges that connect to strong ones.
    """
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate edges slightly to close small gaps
    # Without this, the document outline might have breaks
    # and we won't find it as a complete contour
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    return edges

def find_document_contour(edges, image):
    """
    Find the document boundary in the edge image.
    
    How it works:
    1. Find ALL contours (closed shapes) in the edge image
    2. Sort them by area (largest first)
    3. For each contour, try to approximate it as a polygon
    4. If the polygon has exactly 4 vertices — it's a rectangle — that's our document
    
    The approximation step is key. A real document edge isn't a perfect 
    rectangle — it has tiny bumps and irregularities. approxPolyDP smooths 
    the contour and reduces it to its essential shape. A document always 
    reduces to 4 points (corners).
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort by area, largest first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    document_contour = None
    
    for contour in contours[:10]:  # only check the 10 largest
        # Calculate perimeter
        perimeter = cv2.arcLength(contour, True)
        
        # Approximate the contour to a polygon
        # The second parameter (0.02 * perimeter) controls how much 
        # simplification is allowed. Smaller = more precise, larger = more smoothing
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        # If the approximated contour has 4 points, it's a quadrilateral
        if len(approx) == 4:
            # Make sure it's large enough (at least 10% of image area)
            area = cv2.contourArea(approx)
            image_area = image.shape[0] * image.shape[1]
            if area > 0.1 * image_area:
                document_contour = approx
                print(f"Found document contour with area {area:.0f} ({area/image_area*100:.1f}% of image)")
                break
    
    if document_contour is None:
        print("WARNING: No document contour found!")
    
    return document_contour


def order_corners(pts):
    """
    Order the 4 corner points as: top-left, top-right, bottom-right, bottom-left.
    
    This is critical for the perspective transform — if corners are in the 
    wrong order, the image will be warped incorrectly (flipped, rotated, etc.)
    
    The trick: 
    - Top-left has the smallest sum (x+y)
    - Bottom-right has the largest sum (x+y)
    - Top-right has the smallest difference (y-x)
    - Bottom-left has the largest difference (y-x)
    """
    pts = pts.reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]  # top-left
    ordered[2] = pts[np.argmax(s)]  # bottom-right
    
    d = np.diff(pts, axis=1)
    ordered[1] = pts[np.argmin(d)]  # top-right
    ordered[3] = pts[np.argmax(d)]  # bottom-left
    
    return ordered


def perspective_transform(image, corners, ratio):
    """
    Warp the document to a flat, rectangular view.
    
    This is the core of the scanner. Given 4 corner points, we compute
    a transformation matrix that maps those 4 points to a perfect rectangle.
    
    ratio: the resize ratio from earlier — we need to scale corners back 
    to original image coordinates for maximum quality output.
    """
    # Scale corners back to original image size
    corners = corners / ratio
    ordered = order_corners(corners)
    
    tl, tr, br, bl = ordered
    
    # Calculate output dimensions
    # Width = max of top edge and bottom edge
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bottom))
    
    # Height = max of left edge and right edge
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = int(max(height_left, height_right))
    
    # Define destination points (a perfect rectangle)
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)
    
    # Compute and apply the perspective transform
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    
    print(f"Output dimensions: {max_width}x{max_height}")
    
    return warped
# Let's test on your image
if __name__ == "__main__":
    img = cv2.imread("sample_images/1.jpg")
    resized, ratio = resize_image(img)
    blurred = preprocess(resized)
    edges = detect_edges(blurred)
    
    print(f"Original size: {img.shape[1]}x{img.shape[0]}")
    print(f"Resized to: {resized.shape[1]}x{resized.shape[0]}")
    
    # Find document
    contour = find_document_contour(edges, resized)
    
    if contour is not None:
        # Draw contour on resized image for visualization
        debug_img = resized.copy()
        cv2.drawContours(debug_img, [contour], -1, (0, 255, 0), 3)
        cv2.imwrite("output/4_contour_found.jpg", debug_img)
        
        # Perspective transform on ORIGINAL (high-res) image
        scanned = perspective_transform(img, contour.reshape(4, 2).astype(np.float32), ratio)
        cv2.imwrite("output/5_scanned.jpg", scanned)
        
        print("Saved contour visualization and scanned output to output/")
    else:
        print("Could not find document. Try adjusting edge detection parameters.")