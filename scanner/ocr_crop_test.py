# Save as scanner/ocr_crop_test.py

import cv2
import pytesseract

# Load the scanned image
img = cv2.imread("output/5_scanned.jpg")

# Crop just the top 40% where text was readable
h = img.shape[0]
top_crop = img[:int(h * 0.4), :]

# Enhance the crop
gray = cv2.cvtColor(top_crop, cv2.COLOR_BGR2GRAY)
adaptive = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 21, 10
)

# OCR
text = pytesseract.image_to_string(adaptive, config='--oem 3 --psm 6')
lines = [l for l in text.split('\n') if l.strip()]

print(f"Lines extracted: {len(lines)}")
print(f"\nExtracted text:")
for l in lines:
    print(f"  {l}")