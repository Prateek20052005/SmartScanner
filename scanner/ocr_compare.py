# Save as scanner/ocr_compare.py

import cv2
import pytesseract
import time

def run_ocr(image_path, name):
    """Run OCR and return the simple text output."""
    print(f"\n--- {name} ---")
    start = time.time()
    
    # Use --psm 6 (assume uniform block of text) for better paragraph handling
    custom_config = '--oem 3 --psm 6'
    text = pytesseract.image_to_string(
        cv2.imread(image_path), 
        config=custom_config
    )
    
    elapsed = time.time() - start
    
    # Count non-empty lines
    lines = [l for l in text.split('\n') if l.strip()]
    
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Lines extracted: {len(lines)}")
    print(f"  First 5 lines:")
    for l in lines[:5]:
        print(f"    {l}")
    
    return text, lines

# Compare all versions
versions = [
    ("output/5_scanned.jpg", "Raw scan (color)"),
    ("output/6_gray.jpg", "Grayscale"),
    ("output/7_adaptive_threshold.jpg", "Adaptive threshold"),
    ("output/9_sharpened.jpg", "Sharpened"),
]

best_text = ""
best_lines = 0
best_name = ""

for path, name in versions:
    text, lines = run_ocr(path, name)
    if len(lines) > best_lines:
        best_lines = len(lines)
        best_text = text
        best_name = name

print(f"\n{'='*60}")
print(f"Best version: {best_name} ({best_lines} lines)")
print(f"{'='*60}")
print(best_text)

# Save best result
with open("output/extracted_text.txt", "w") as f:
    f.write(best_text)
print(f"\nSaved best result to output/extracted_text.txt")