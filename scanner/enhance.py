# Save as scanner/enhance.py

import cv2
import numpy as np

def enhance_scan(image):
    """
    Apply multiple enhancement techniques to make the scanned 
    document look like it came from a real scanner.
    """
    
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Adaptive thresholding
    # Unlike simple thresholding (one threshold for entire image),
    # adaptive thresholding calculates a different threshold for 
    # each small region. This handles uneven lighting — if one 
    # corner is darker than another, it still works.
    # 
    # Block size (21): the neighborhood size for computing local threshold
    # C (10): constant subtracted from the mean — controls sensitivity
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 10
    )
    
    # Step 3: Denoise the binary image
    # Remove small speckles that adaptive thresholding creates
    adaptive = cv2.medianBlur(adaptive, 3)
    
    return gray, adaptive


def enhance_scan_color(image):
    """
    Enhance while keeping some color/gray tones.
    Looks more natural than pure black & white.
    """
    
    # CLAHE: Contrast Limited Adaptive Histogram Equalization
    # Improves contrast locally without over-amplifying noise
    # Like adaptive thresholding but for contrast, not binarization
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Sharpen the image
    # The kernel boosts the center pixel and subtracts neighbors,
    # which makes edges (like text) crisper
    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)
    
    return enhanced, sharpened


if __name__ == "__main__":
    # Load the scanned output
    scanned = cv2.imread("output/5_scanned.jpg")
    
    if scanned is None:
        print("No scanned image found. Run debug_contours_v3.py first.")
        exit()
    
    print(f"Input size: {scanned.shape[1]}x{scanned.shape[0]}")
    
    # Apply both enhancement methods
    gray, adaptive = enhance_scan(scanned)
    enhanced, sharpened = enhance_scan_color(scanned)
    
    # Save all versions
    cv2.imwrite("output/6_gray.jpg", gray)
    cv2.imwrite("output/7_adaptive_threshold.jpg", adaptive)
    cv2.imwrite("output/8_clahe_enhanced.jpg", enhanced)
    cv2.imwrite("output/9_sharpened.jpg", sharpened)
    
    print("Saved enhancement results:")
    print("  6_gray.jpg             - Simple grayscale")
    print("  7_adaptive_threshold.jpg - Black & white (scanner look)")
    print("  8_clahe_enhanced.jpg   - Contrast enhanced grayscale")
    print("  9_sharpened.jpg        - Sharpened for OCR")