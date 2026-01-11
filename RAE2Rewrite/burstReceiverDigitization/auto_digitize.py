import cv2
import numpy as np
import pandas as pd
import pytesseract
import argparse
import re
import os

# --- Configuration ---
# You may need to point this to your tesseract executable if on Windows
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image_path):
    """
    Loads and prepares the image for both OCR and Data Extraction.
    Returns: Original Gray, Binary (Text), Dots Only (Data)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Binary for OCR (keep text sharp)
    # Adaptive threshold works well for text on varying backgrounds
    binary_text = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 31, 10)

    # 2. Clean for Data Extraction (Grid Removal)
    inv = 255 - gray
    _, binary_data = cv2.threshold(inv, 100, 255, cv2.THRESH_BINARY)
    
    # Aggressive Grid Removal
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    h_lines = cv2.morphologyEx(binary_data, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(binary_data, cv2.MORPH_OPEN, v_kernel)
    
    grid_mask = cv2.add(h_lines, v_lines)
    grid_mask = cv2.dilate(grid_mask, np.ones((5,5), np.uint8)) # Aggressive dilation
    
    dots_only = cv2.subtract(binary_data, grid_mask)
    dots_clean = cv2.morphologyEx(dots_only, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
    
    return gray, binary_text, dots_clean

def get_roi_text(img, x, y, w, h, config='--psm 6'):
    """Extracts text from a specific Region of Interest (ROI)."""
    roi = img[y:y+h, x:x+w]
    text = pytesseract.image_to_string(roi, config=config)
    return text.strip()

def extract_metadata(gray_img):
    """
    Attempts to read Date, Time Range from bottom left
    and Receiver Name from top.
    """
    h, w = gray_img.shape
    
    metadata = {
        "date": "UnknownDate",
        "time_start": "0000",
        "time_end": "0000",
        "receiver": "UnknownReceiver"
    }
    
    # --- 1. Detect Receiver (Top Center/Left) ---
    header_roi = gray_img[0:int(h*0.15), 0:w]
    header_text = pytesseract.image_to_string(header_roi).upper()
    
    if "RCVR 1" in header_text or "RECEIVER 1" in header_text:
        metadata["receiver"] = "BurstReceiver1"
    elif "RCVR 2" in header_text or "RECEIVER 2" in header_text:
        metadata["receiver"] = "BurstReceiver2"

    # --- 2. Detect Date/Time (Bottom Left) ---
    # Usually in the bottom 15% of the image, left 25%
    footer_roi = gray_img[int(h*0.85):h, 0:int(w*0.35)]
    
    # Use specific OCR config for finding digits and dates
    footer_text = pytesseract.image_to_string(footer_roi, config='--psm 6')
    
    # Regex Logic for Dates (e.g., 75/05/03 or 75-5-3)
    # Looks for pattern: 2 digits, separator, 1-2 digits, separator, 1-2 digits
    date_match = re.search(r'(\d{2})[/-](\d{1,2})[/-](\d{1,2})', footer_text)
    if date_match:
        # Normalize to YYYYMMDD
        yy, mm, dd = date_match.groups()
        # Assume 19xx for year
        metadata["date"] = f"19{yy}{int(mm):02d}{int(dd):02d}"
        
    # Regex Logic for Times (Looking for ranges like 0100-0200 or 01:00)
    # This is tricky as handwriting varies. We look for 4-digit blocks.
    times = re.findall(r'\b\d{4}\b', footer_text)
    if len(times) >= 2:
        metadata["time_start"] = times[0]
        metadata["time_end"] = times[1]
    
    return metadata

def extract_channel_labels(gray_img, num_channels=8):
    """
    Scans the right margin to identify channel frequencies.
    Returns a list of labels (e.g., ['110', '096', ...])
    """
    h, w = gray_img.shape
    
    # Define Right Margin ROI
    margin_x = int(w * 0.90) # Last 10% of width
    margin_w = w - margin_x
    
    # Calculate approximate height of one channel
    # We exclude top/bottom margins (approx 10% each)
    graph_top = int(h * 0.10)
    graph_bottom = int(h * 0.90)
    channel_h = (graph_bottom - graph_top) / num_channels
    
    detected_labels = []
    
    # Standard fallback RAE-B frequencies
    defaults = ['110', '096', '083', '067', '055', '044', '035', '025']
    
    for i in range(num_channels):
        # Define ROI for this specific channel label
        y_c = int(graph_top + i * channel_h)
        # Look slightly above/below the center of the channel strip
        roi = gray_img[y_c : y_c + int(channel_h), margin_x : w]
        
        # OCR restricted to digits
        text = pytesseract.image_to_string(roi, config='--psm 7 outputbase digits').strip()
        
        # Validation: Labels are usually 3 digits. 
        if text.isdigit() and len(text) in [2, 3]:
            detected_labels.append(text)
        else:
            # Fallback if OCR fails
            detected_labels.append(defaults[i] + "_est")
            
    return detected_labels

def main():
    parser = argparse.ArgumentParser(description="Auto-Digitize Strip Chart with Metadata OCR")
    parser.add_argument("image", help="Input image path")
    args = parser.parse_args()
    
    print(f"Analyzing {args.image}...")
    gray, binary_text, dots_clean = preprocess_image(args.image)
    
    # 1. Metadata Extraction
    print("Reading metadata (Date/Time/Receiver)...")
    meta = extract_metadata(gray)
    print(f"  > Date: {meta['date']}")
    print(f"  > Time: {meta['time_start']} - {meta['time_end']}")
    print(f"  > Receiver: {meta['receiver']}")
    
    # 2. Channel Label Extraction
    print("Reading channel labels...")
    channel_labels = extract_channel_labels(gray)
    print(f"  > Detected Channels: {channel_labels}")
    
    # 3. Data Extraction Setup
    h, w = dots_clean.shape
    x_start = int(w * 0.10) # Skip left text
    x_end = int(w * 0.90)   # Skip right labels
    y_start = int(h * 0.10)
    y_end = int(h * 0.90)
    
    num_channels = 8
    channel_height = (y_end - y_start) / num_channels
    
    # Sampling Setup (7.68s)
    total_min = 60.0
    sample_period_min = 7.68 / 60.0
    num_samples = int(total_min / sample_period_min) + 1
    sample_x_coords = np.linspace(x_start, x_end, num_samples)
    
    extracted_data = []
    
    # 4. Extraction Loop
    for ch_idx in range(num_channels):
        label = channel_labels[ch_idx]
        ch_top = y_start + ch_idx * channel_height
        
        for t_idx, x_float in enumerate(sample_x_coords):
            x = int(round(x_float))
            time_min = t_idx * sample_period_min
            
            if x >= w: break
            
            # Search window
            strip = dots_clean[int(ch_top+2):int(ch_top+channel_height-2), x-1:x+2]
            
            val = "" # Empty by default
            if np.any(strip):
                M = cv2.moments(strip)
                if M["m00"] > 0:
                    cy = M["m01"] / M["m00"]
                    pixel_y = int(ch_top+2) + cy
                    # Map to 4-12 Scale
                    norm_y = (pixel_y - ch_top) / channel_height
                    val = round(12.0 - norm_y * (12.0 - 4.0), 4)
            
            extracted_data.append({
                "Date": meta['date'],
                "Time_Range": f"{meta['time_start']}-{meta['time_end']}",
                "Time_Offset_Min": round(time_min, 4),
                "Channel": label,
                "Value": val
            })
            
    # 5. Save Output
    output_filename = f"{meta['date']}_{meta['time_start']}-{meta['time_end']}_{meta['receiver']}.csv"
    
    # Sanitize filename (remove bad chars if OCR failed)
    output_filename = output_filename.replace("UnknownDate", "UnknownDate").replace(":", "")
    
    df = pd.DataFrame(extracted_data)
    df.to_csv(output_filename, index=False)
    print(f"\nCompleted! Data saved to: {output_filename}")

if __name__ == "__main__":
    main()