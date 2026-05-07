# scanner/ocr.py

import cv2
import pytesseract
import time

def extract_text(image_path):
    """
    Extract text from a scanned document using Tesseract OCR.
    
    Tesseract works best on clean, high-contrast images — which is 
    exactly what our adaptive thresholding produces.
    """
    
    image = cv2.imread(image_path)
    
    print("Running Tesseract OCR...")
    start = time.time()
    
    # Get detailed output with confidence scores
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    
    elapsed = time.time() - start
    print(f"OCR completed in {elapsed:.1f} seconds")
    
    return data


def format_results(data, confidence_threshold=30):
    """
    Convert Tesseract output into clean readable text.
    
    Tesseract returns word-by-word results with:
    - text: the recognized word
    - conf: confidence (0-100)
    - block_num, line_num, word_num: position in document structure
    
    We use block_num and line_num to reconstruct paragraphs and lines.
    """
    
    lines = {}
    
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        conf = int(data['conf'][i])
        
        if text and conf >= confidence_threshold:
            # Create a unique key for each line
            line_key = (data['block_num'][i], data['line_num'][i])
            
            if line_key not in lines:
                lines[line_key] = {'words': [], 'confidences': []}
            
            lines[line_key]['words'].append(text)
            lines[line_key]['confidences'].append(conf)
    
    # Build formatted output
    formatted_lines = []
    for key in sorted(lines.keys()):
        line_text = " ".join(lines[key]['words'])
        avg_conf = sum(lines[key]['confidences']) / len(lines[key]['confidences'])
        formatted_lines.append((line_text, avg_conf))
    
    full_text = "\n".join([text for text, conf in formatted_lines])
    
    return full_text, formatted_lines


if __name__ == "__main__":
    # Run OCR on the adaptive threshold version
    data = extract_text("output/7_adaptive_threshold.jpg")
    
    full_text, lines = format_results(data)
    
    print(f"\n{'='*60}")
    print("EXTRACTED TEXT")
    print(f"{'='*60}\n")
    print(full_text)
    
    # Save to file
    with open("output/extracted_text.txt", "w") as f:
        f.write(full_text)
    
    print(f"\n{'='*60}")
    print(f"Saved to output/extracted_text.txt")
    
    # Show confidence per line
    print(f"\nLine-by-line confidence:")
    for i, (text, conf) in enumerate(lines):
        print(f"  Line {i+1} ({conf:.0f}%): {text[:80]}...")