# Save as app.py (in the root SmartScanner directory)

import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
import io
import os

st.set_page_config(page_title="SmartScanner", page_icon="📄", layout="wide")
st.title("📄 SmartScanner — Document Scanner + OCR")

# --- CORE FUNCTIONS ---

def resize_image(image, max_height=800):
    height, width = image.shape[:2]
    if height > max_height:
        ratio = max_height / height
        resized = cv2.resize(image, (int(width * ratio), max_height))
        return resized, ratio
    return image, 1.0


def find_document(image):
    """Auto-detect document corners."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    image_area = image.shape[0] * image.shape[1]
    
    for contour in contours[:5]:
        area = cv2.contourArea(contour)
        if area < 0.1 * image_area:
            continue
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        for eps in [0.02, 0.03, 0.04, 0.05]:
            approx = cv2.approxPolyDP(hull, eps * perimeter, True)
            if len(approx) == 4:
                return approx.reshape(4, 2)
    
    # Fallback: minimum area rectangle
    if contours:
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        return box.astype(np.int32)
    
    return None


def order_corners(pts):
    """Order corners: top-left, top-right, bottom-right, bottom-left."""
    pts = pts.astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1).flatten()
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


def warp_perspective(image, corners):
    """Apply perspective transform to get flat document."""
    ordered = order_corners(corners)
    tl, tr, br, bl = ordered
    
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bottom))
    
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = int(max(height_left, height_right))
    
    dst = np.array([
        [0, 0], [max_width - 1, 0],
        [max_width - 1, max_height - 1], [0, max_height - 1]
    ], dtype=np.float32)
    
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped


def enhance_document(image, mode="adaptive"):
    """Enhance scanned document for readability."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mode == "adaptive":
        enhanced = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 10
        )
        enhanced = cv2.medianBlur(enhanced, 3)
    elif mode == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
    elif mode == "sharpen":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
    else:
        enhanced = gray
    
    return enhanced


def run_ocr(image):
    """Extract text using Tesseract."""
    text = pytesseract.image_to_string(image, config='--oem 3 --psm 6')
    lines = [l for l in text.split('\n') if l.strip()]
    return text, lines


# --- STREAMLIT UI ---

uploaded = st.file_uploader("Upload a document photo", type=['jpg', 'jpeg', 'png'])

if uploaded:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    st.subheader("Step 1: Document Detection")
    
    # Resize for processing
    resized, ratio = resize_image(original)
    
    # Auto-detect corners
    corners = find_document(resized)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Original Image**")
        st.image(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    if corners is not None:
        # Draw detected corners
        debug = resized.copy()
        cv2.drawContours(debug, [corners.astype(np.int32)], -1, (0, 255, 0), 3)
        for pt in corners:
            cv2.circle(debug, tuple(pt.astype(int)), 8, (0, 0, 255), -1)
        
        with col2:
            st.write("**Detected Corners**")
            st.image(cv2.cvtColor(debug, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        # Manual corner adjustment
        st.subheader("Step 2: Adjust Corners (if needed)")
        st.write("Fine-tune the corner positions if auto-detection isn't perfect.")
        
        ordered = order_corners(corners)
        h, w = resized.shape[:2]
        
        adj_col1, adj_col2, adj_col3, adj_col4 = st.columns(4)
        
        with adj_col1:
            st.write("**Top-Left**")
            tl_x = st.slider("TL X", 0, w, int(ordered[0][0]), key="tl_x")
            tl_y = st.slider("TL Y", 0, h, int(ordered[0][1]), key="tl_y")
        
        with adj_col2:
            st.write("**Top-Right**")
            tr_x = st.slider("TR X", 0, w, int(ordered[1][0]), key="tr_x")
            tr_y = st.slider("TR Y", 0, h, int(ordered[1][1]), key="tr_y")
        
        with adj_col3:
            st.write("**Bottom-Right**")
            br_x = st.slider("BR X", 0, w, int(ordered[2][0]), key="br_x")
            br_y = st.slider("BR Y", 0, h, int(ordered[2][1]), key="br_y")
        
        with adj_col4:
            st.write("**Bottom-Left**")
            bl_x = st.slider("BL X", 0, w, int(ordered[3][0]), key="bl_x")
            bl_y = st.slider("BL Y", 0, h, int(ordered[3][1]), key="bl_y")
        
        adjusted_corners = np.array([
            [tl_x, tl_y], [tr_x, tr_y],
            [br_x, br_y], [bl_x, bl_y]
        ], dtype=np.float32)
        
        # Scale corners to original image size
        original_corners = adjusted_corners / ratio
        
        # Show adjusted corners preview
        preview = resized.copy()
        pts = adjusted_corners.astype(np.int32)
        cv2.polylines(preview, [pts], True, (0, 255, 0), 3)
        for pt in pts:
            cv2.circle(preview, tuple(pt), 8, (0, 0, 255), -1)
        st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        # Scan button
        if st.button("📷 Scan Document", type="primary"):
            st.subheader("Step 3: Scanned Output")
            
            # Warp using original high-res image
            scanned = warp_perspective(original, original_corners)
            
            # Enhancement options
            enhance_mode = st.radio(
                "Enhancement mode",
                ["adaptive", "clahe", "sharpen", "grayscale"],
                horizontal=True
            )
            
            enhanced = enhance_document(scanned, mode=enhance_mode)
            
            scan_col1, scan_col2 = st.columns(2)
            with scan_col1:
                st.write("**Raw Scan**")
                st.image(cv2.cvtColor(scanned, cv2.COLOR_BGR2RGB), use_container_width=True)
            with scan_col2:
                st.write("**Enhanced**")
                st.image(enhanced, use_container_width=True)
            
            # OCR
            st.subheader("Step 4: Extracted Text")
            
            with st.spinner("Running OCR..."):
                text, lines = run_ocr(enhanced)
            
            st.text_area("Extracted Text", text, height=400)
            st.write(f"**Lines extracted:** {len(lines)}")
            
            # Download options
            st.subheader("Download")
            
            dl_col1, dl_col2 = st.columns(2)
            
            with dl_col1:
                # Download scanned image
                _, buf = cv2.imencode('.jpg', enhanced)
                st.download_button(
                    "📥 Download Scanned Image",
                    data=buf.tobytes(),
                    file_name="scanned_document.jpg",
                    mime="image/jpeg"
                )
            
            with dl_col2:
                # Download text
                st.download_button(
                    "📥 Download Extracted Text",
                    data=text,
                    file_name="extracted_text.txt",
                    mime="text/plain"
                )
    
    else:
        with col2:
            st.error("Could not detect document corners automatically.")
            st.write("Try uploading a photo with more contrast between the document and background.")