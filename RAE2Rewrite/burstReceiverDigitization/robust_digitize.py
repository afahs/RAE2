import cv2
import numpy as np
import pandas as pd
import google.generativeai as genai
import argparse
import os
import json
import time

# --- CONFIGURATION ---
# valid channels provided by user (used for AI context)
VALID_CHANNELS = [
    "0.025", "0.035", "0.044", "0.055", "0.067", "0.083", "0.096", "0.110",
    "0.130", "0.155", "0.185", "0.210", "0.250", "0.292", "0.360", "0.425",
    "0.475", "0.600", "0.737", "0.870", "1.030", "1.27", "1.45", "1.85",
    "2.20", "2.80", "3.93", "4.70", "6.55", "9.18", "11.8", "13.1"
]

def get_metadata_from_gemini(api_key, image_path):
    """
    Sends image to Gemini to extract Date, Time, Receiver, and Channel Labels.
    """
    genai.configure(api_key=api_key)
    
    # Use Flash for speed/cost, or Pro for maximum reasoning
    model = genai.GenerativeModel('gemini-flash-latest')
    
    # Load image
    with open(image_path, "rb") as f:
        img_data = f.read()

    # Construct Prompt
    prompt = f"""
    Analyze this radio astronomy strip chart. I need to extract metadata to digitize it.
    
    1. **Date:** Find the handwritten date (usually bottom left). Format as YYYYMMDD. If ambiguous, give best guess.
    2. **Time Range:** Find the start and end times (usually bottom left/center). Format as HHMM-HHMM.
    3. **Receiver:** Look at the top header text. Is it "Burst Receiver 1" (or RCVR 1) or "Burst Receiver 2"?
    4. **Channels:** Identify the 8 vertical channel labels on the right side of the graph, from TOP to BOTTOM.
       The labels on the image are likely in kHz (e.g. 110, 096), but they correspond to this list of valid MHz frequencies:
       {VALID_CHANNELS}
       Return the text exactly as written on the chart for the channels.
    
    Return a pure JSON object with no markdown formatting:
    {{
        "date": "YYYYMMDD",
        "time_str": "HHMM-HHMM",
        "receiver": "BurstReceiverX",
        "channels": ["Label1", "Label2", "Label3", "Label4", "Label5", "Label6", "Label7", "Label8"]
    }}
    """
    
    response = model.generate_content([
        {'mime_type': 'image/jpeg', 'data': img_data},
        prompt
    ])
    
    try:
        # Clean response (sometimes models add ```json blocks)
        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        print(f"Error parsing Gemini response: {e}")
        print("Raw response:", response.text)
        return None

def process_image_data(image_path, channels_list, debug_mode=False):
    """
    Performs the local OpenCV grid removal and data extraction.
    Returns (extracted_rows, debug_image)
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create a copy for debugging if requested
    debug_img = img.copy() if debug_mode else None
    
    # Invert & Threshold
    inv = 255 - gray
    _, binary = cv2.threshold(inv, 100, 255, cv2.THRESH_BINARY)
    
    # --- Aggressive Grid Removal ---
    # Horizontal
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    # Vertical
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    
    # Combine & Dilate
    grid_mask = cv2.add(h_lines, v_lines)
    grid_mask = cv2.dilate(grid_mask, np.ones((5,5), np.uint8), iterations=1)
    
    # Subtract Grid
    dots_only = cv2.subtract(binary, grid_mask)
    dots_clean = cv2.morphologyEx(dots_only, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
    
    # --- Determine Bounds ---
    h, w = dots_clean.shape
    x_start = int(w * 0.11)
    x_end = int(w * 0.905)
    y_start = int(h * 0.10) # Below header
    y_end = int(h * 0.88)   # Above footer
    
    if debug_mode:
        # Draw the bounding box (Blue)
        cv2.rectangle(debug_img, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
    
    # --- Setup Extraction ---
    num_channels = 8
    channel_height = (y_end - y_start) / num_channels
    
    # 60 Minutes, 7.68s cadence
    total_min = 60.0
    sample_period_min = 7.68 / 60.0
    num_samples = int(total_min / sample_period_min) + 1
    sample_x_coords = np.linspace(x_start, x_end, num_samples)
    
    extracted_rows = []
    
    # Ensure we have 8 labels, fill if missing
    while len(channels_list) < 8:
        channels_list.append(f"Unknown_{len(channels_list)}")
    
    for ch_idx in range(num_channels):
        label = channels_list[ch_idx]
        ch_top = y_start + ch_idx * channel_height
        
        if debug_mode:
            # Draw channel separator lines (Yellow)
            y_line = int(ch_top)
            cv2.line(debug_img, (x_start, y_line), (x_end, y_line), (0, 255, 255), 1)
        
        for t_idx, x_float in enumerate(sample_x_coords):
            x = int(round(x_float))
            time_min = t_idx * sample_period_min
            
            if x >= w: break
            
            # Search Strip (+/- 1 pixel width, vertical padding)
            strip_y_start = int(ch_top+2)
            strip_y_end = int(ch_top+channel_height-2)
            strip = dots_clean[strip_y_start:strip_y_end, x-1:x+2]
            
            val = "" # Default empty
            if np.any(strip):
                M = cv2.moments(strip)
                if M["m00"] > 0:
                    cy = M["m01"] / M["m00"]
                    pixel_y = strip_y_start + cy
                    
                    # Map to Log Temp Scale (12 Top to 4 Bottom)
                    norm_y = (pixel_y - ch_top) / channel_height
                    val = round(12.0 - norm_y * (12.0 - 4.0), 4)
                    
                    if debug_mode:
                        # Draw Red Dot at detection point
                        cv2.circle(debug_img, (x, int(pixel_y)), 2, (0, 0, 255), -1)
            
            extracted_rows.append({
                "Time_Min": round(time_min, 4),
                "Channel": label,
                "Value": val
            })
            
    return extracted_rows, debug_img

def main():
    parser = argparse.ArgumentParser(description="Hybrid Digitize: Gemini Metadata + OpenCV Extraction")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--key", help="Google Gemini API Key", default=os.environ.get("GEMINI_API_KEY"))
    parser.add_argument("--debug", action="store_true", help="Generate a debug JPEG with overlaid points")
    
    args = parser.parse_args()
    
    if not args.key:
        print("Error: No API Key found.")
        print("Pass it with --key 'YOUR_KEY' or set GEMINI_API_KEY env var.")
        return

    print(f"--- Processing {args.image} ---")
    
    # 1. Ask Gemini for Metadata
    print("Step 1: Consulting Gemini for Metadata (Date, Time, Channels)...")
    meta = get_metadata_from_gemini(args.key, args.image)
    
    if not meta:
        print("Failed to retrieve metadata from Gemini.")
        return
        
    print(f"  > Detected Date: {meta.get('date')}")
    print(f"  > Detected Time: {meta.get('time_str')}")
    print(f"  > Receiver:      {meta.get('receiver')}")
    print(f"  > Channels:      {meta.get('channels')}")
    
    # 2. Extract Data Locally
    print("Step 2: extracting data points locally...")
    # Pass debug flag here
    data_rows, debug_img = process_image_data(args.image, meta.get('channels', []), debug_mode=args.debug)
    
    # 3. Save to CSV
    # Add metadata columns to every row
    final_data = []
    for row in data_rows:
        row['Date'] = meta.get('date')
        row['Time_Range'] = meta.get('time_str')
        row['Receiver'] = meta.get('receiver')
        # Reorder dict for CSV columns
        final_data.append(row)
        
    df = pd.DataFrame(final_data)
    
    # Construct Filename
    # 19750503_0100-0200_BurstReceiver1.csv
    base_name = f"{meta.get('date')}_{meta.get('time_str')}_{meta.get('receiver')}"
    # Clean filename
    clean_name = base_name.replace("/", "").replace(":", "").replace(" ", "")
    csv_filename = f"{clean_name}.csv"
    
    df.to_csv(csv_filename, index=False)
    print(f"--- Success! Saved to {csv_filename} ---")
    
    # 4. Save Debug Image if requested
    if args.debug and debug_img is not None:
        debug_filename = f"debug_{clean_name}.jpg"
        cv2.imwrite(debug_filename, debug_img)
        print(f"--- Debug Image Saved: {debug_filename} ---")

if __name__ == "__main__":
    main()