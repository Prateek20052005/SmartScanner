# 📄 SmartScanner

A document scanning application that transforms messy phone photos into clean, readable scans with automatic text extraction. Combines classical computer vision techniques with OCR to produce a genuinely useful tool.

## Overview

Point your phone at a document — a receipt, exam paper, printed page, handwritten notes — and SmartScanner will detect the document boundaries, correct the perspective distortion, enhance the image for readability, and extract the text. It handles angled shots, uneven lighting, and cluttered backgrounds.

## How It Works

```
Phone photo → Edge Detection → Corner Finding → Perspective Warp → Enhancement → OCR
```

The pipeline combines six distinct CV techniques, each solving a specific problem:

1. **Otsu Thresholding** separates the bright document from the dark background by automatically finding the optimal brightness threshold
2. **Convex Hull + Polygon Approximation** finds the document boundary and reduces it to exactly 4 corner points, even when edges are noisy
3. **Perspective Transform** maps the tilted quadrilateral to a flat rectangle using a 3×3 projective transformation matrix
4. **Adaptive Thresholding** produces clean black-on-white output by computing a local threshold for each image region, handling uneven lighting
5. **CLAHE Enhancement** improves contrast locally without over-amplifying noise, making faded text readable
6. **Tesseract OCR** extracts text using an LSTM neural network that reads character by character

## Results

### Before → After

| Input | Scanned | Enhanced |
|-------|---------|----------|
| Angled photo with shadows, cluttered background | Perspective-corrected flat view | Clean black & white, scanner-quality output |

The system handles documents photographed at significant angles (up to ~45°), on patterned/textured backgrounds, and with uneven lighting across the page.

### OCR Accuracy

| Document Region | Confidence | Quality |
|----------------|------------|---------|
| Headers & titles | 88-96% | Near-perfect extraction |
| Body text (clean scan) | 75-90% | Good with minor errors |
| Dense paragraphs (distorted) | 40-70% | Degraded — manual corner adjustment helps |

OCR quality depends directly on scan quality. The manual corner adjustment feature allows users to fine-tune detection when auto-detection isn't perfect.

## Features

- **Automatic document detection** using Otsu thresholding + convex hull + polygon approximation
- **Manual corner adjustment** with interactive sliders when auto-detection needs fine-tuning
- **Four enhancement modes**: adaptive threshold (scanner look), CLAHE (contrast boost), sharpened (OCR-optimized), and grayscale
- **Text extraction** with Tesseract OCR
- **Download options** for both the scanned image and extracted text
- **Works on any document** — receipts, printed pages, handwritten notes, forms

## Project Structure

```
SmartScanner/
├── app.py                      # Streamlit application (main entry point)
├── scanner/
│   ├── core.py                 # Document detection & perspective transform
│   ├── enhance.py              # Image enhancement functions
│   ├── ocr.py                  # Text extraction with Tesseract
│   ├── debug_contours.py       # Debugging tool for contour detection
│   ├── debug_contours_v2.py    # Threshold-based detection debugging
│   ├── debug_contours_v3.py    # Final detection with convex hull
│   └── ocr_compare.py          # OCR comparison across enhancement modes
├── sample_images/              # Test photos
├── output/                     # Processing results
├── requirements.txt
└── README.md
```

## Setup & Usage

### Prerequisites

- Python 3.10+
- Tesseract OCR engine

### Install Tesseract

**Mac:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**Windows:**
Download installer from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

### Install Python dependencies

```bash
pip install opencv-python numpy pytesseract streamlit pillow
```

### Run the app

```bash
streamlit run app.py
```

Upload a document photo → adjust corners if needed → click "Scan Document" → choose enhancement mode → download the result.

## Technical Deep Dive

### Why Classical CV Over Deep Learning?

This project deliberately uses classical computer vision rather than a CNN-based document detector. The reasons are practical: classical methods run instantly on any CPU (no GPU needed), require zero training data, and are fully interpretable — when something goes wrong, you can visualize every intermediate step (edges, contours, thresholds) and understand exactly why.

A deep learning approach (like a U-Net for document segmentation) would likely achieve better corner detection accuracy on difficult images, but would require a trained model, more compute, and would be harder to debug. The manual corner adjustment feature bridges the gap — auto-detection handles 80% of cases, and the user fixes the rest in seconds.

### The Corner Detection Challenge

Finding document corners is the hardest part of this pipeline. The naive approach (Canny edges → find quadrilateral contour) fails frequently because text on the document creates thousands of small edge fragments, patterned backgrounds create edges that compete with the document boundary, and shadows or low contrast can break the document edge.

The solution evolved through three iterations. Version 1 used pure Canny edge detection — failed because text edges dominated. Version 2 switched to Otsu thresholding — better, but background regions merging with the document created irregular contours with too many vertices. Version 3 added convex hull before polygon approximation — this smooths out irregularities and reliably produces 4-point approximations, with a minimum-area-rectangle fallback.

### Enhancement Mode Tradeoffs

Each mode optimizes for different use cases:

- **Adaptive threshold** produces the cleanest black-and-white output, ideal for printing or archiving. Handles uneven lighting well. Can lose fine detail in gray-tone images.
- **CLAHE** preserves gray tones while boosting contrast. Better for photos of handwritten notes where stroke thickness carries information.
- **Sharpen** maximizes edge crispness for OCR. Can amplify noise on low-quality inputs.
- **Grayscale** is the safest option — minimal processing, preserves all detail.

### OCR Limitations

Tesseract performs best on clean, high-resolution, well-aligned text. Performance degrades with curved or warped text (partially addressed by perspective correction), very small font sizes, mixed languages in the same document, and heavy background texture bleeding into the scan. For production use, EasyOCR or Google Cloud Vision API would provide better accuracy, especially on challenging inputs.

## What I Learned

- **Image quality cascades**: every stage depends on the previous one. Poor corner detection → distorted scan → bad enhancement → garbage OCR. Fixing the root cause (detection) matters more than compensating downstream.
- **Classical CV is underrated**: edge detection, morphological operations, and contour analysis are powerful tools that run in milliseconds. Not everything needs deep learning.
- **Debugging with intermediate outputs**: saving images at each pipeline stage (edges, contours, threshold, warp) was essential for understanding failures. This is the CV equivalent of print-debugging.
- **User control matters**: auto-detection works most of the time, but providing manual override (corner adjustment) makes the tool reliable for all cases. Real scanner apps (CamScanner, Adobe Scan) all do this.

## Limitations & Future Improvements

- **Multi-page scanning**: currently handles one page at a time. Could add batch processing with PDF output.
- **Deep learning corner detection**: a lightweight CNN or U-Net for document segmentation would improve auto-detection accuracy on difficult backgrounds.
- **Handwriting recognition**: current OCR is optimized for printed text. Adding a handwriting model would expand use cases.
- **Table detection**: identifying and extracting tabular data from scanned documents using contour analysis or a specialized model.
- **Multi-language OCR**: Tesseract supports 100+ languages but currently only English is configured.
- **Mobile deployment**: packaging as a mobile app with live camera preview and real-time edge detection.

## Tech Stack

- **Computer Vision:** OpenCV (edge detection, contours, perspective transform, morphological operations)
- **OCR:** Tesseract (LSTM-based text recognition)
- **Frontend:** Streamlit
- **Image Processing:** NumPy, Pillow