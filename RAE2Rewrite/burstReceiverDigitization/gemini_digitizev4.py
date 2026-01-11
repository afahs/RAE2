import cv2
import numpy as np
import pandas as pd
import google.generativeai as genai
import json
import math
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from dataclasses import dataclass
from typing import List, Tuple

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------

GOOGLE_API_KEY = "***REMOVED***"
USE_MOCK_GEMINI = True  # Set to False to use real API

# 7.68 seconds per point (from prompt)
POINT_INTERVAL_SECONDS = 7.68

VALID_FREQUENCIES = [
    "0.025", "0.035", "0.044", "0.055", "0.067", "0.083", "0.096", "0.110", 
    "0.130", "0.155", "0.185", "0.210", "0.250", "0.292", "0.360", "0.425", 
    "0.475", "0.600", "0.737", "0.870", "1.030", "1.27", "1.45", "1.85", 
    "2.20", "2.80", "3.93", "4.70", "6.55", "9.18", "11.8", "13.1"
]

@dataclass
class ChartMetadata:
    receiver: str
    date: str
    start_hour: int
    end_hour: int
    min_temp: float
    max_temp: float
    channels: List[str]

# Updated Mock Data
MOCK_METADATA = ChartMetadata(
    receiver="RAE-B BURST RECEIVER 1",
    date="75/5/3",
    start_hour=1,
    end_hour=2,
    min_temp=4.0, 
    max_temp=12.0,
    channels=["0.110", "0.096", "0.083", "0.067", "0.055", "0.044", "0.035", "0.025"]
)

class RadioAstronomyDigitizer:
    def __init__(self, api_key):
        if not USE_MOCK_GEMINI:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp') 

    def extract_metadata(self, image_path: str) -> ChartMetadata:
        if USE_MOCK_GEMINI:
            print("--- WARNING: USING MOCK METADATA (Skipping API call) ---")
            return MOCK_METADATA

        print(f"Analyzing metadata for {image_path} with Gemini...")
        file = genai.upload_file(image_path)
        
        prompt = f"""
        Analyze this radio astronomy chart. Return a JSON object with:
        1. 'receiver': Text at the top.
        2. 'date': Date at bottom left (YYYY/M/D).
        3. 'start_hour': Integer hour at the far left of the time axis.
        4. 'end_hour': Integer hour at the far right of the time axis.
        5. 'min_temp': The lowest number on the vertical Y-axis scale (often 4 or similar).
        6. 'max_temp': The highest number on the vertical Y-axis scale (often 12 or similar).
        7. 'channels': List of 8 frequencies on the right edge (Top to Bottom).
        
        CRITICAL: The channels MUST be selected from this list:
        {json.dumps(VALID_FREQUENCIES)}
        
        Return ONLY valid JSON.
        """

        try:
            response = self.model.generate_content([prompt, file])
            text = response.text.replace('```json', '').replace('```', '')
            data = json.loads(text)
            
            return ChartMetadata(
                receiver=data.get('receiver', 'Unknown'),
                date=data.get('date', '1970/1/1'),
                start_hour=int(data.get('start_hour', 0)),
                end_hour=int(data.get('end_hour', 1)),
                min_temp=float(data.get('min_temp', 4.0)),
                max_temp=float(data.get('max_temp', 12.0)),
                channels=data.get('channels', [])
            )
        except Exception as e:
            print(f"Gemini Error: {e}")
            return MOCK_METADATA

    def preprocess_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 25, 15)
        return binary

    def get_grid_mask(self, binary_img):
        """Returns a mask containing ONLY the horizontal and vertical grid lines."""
        # Horizontal lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        h_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, h_kernel)
        
        # Vertical lines
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        v_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, v_kernel)
        
        grid_mask = cv2.add(h_lines, v_lines)
        grid_mask = cv2.dilate(grid_mask, np.ones((3,3), np.uint8), iterations=2)
        return grid_mask

    def correct_skew(self, image):
        """
        Detects lines in the GRID MASK (not edges) to determine skew angle.
        Uses high-precision Hough Transform (0.1 degree resolution).
        """
        # 1. Get Grid Mask to ignore text/noise
        binary = self.preprocess_image(image)
        grid_mask = self.get_grid_mask(binary)
        
        # 2. High-Res Hough Transform
        # Theta resolution: np.pi/1800 = 0.1 degrees
        lines = cv2.HoughLines(grid_mask, 1, np.pi/1800, 200)
        
        if lines is None:
            return image

        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            # Only consider near-horizontal lines
            if -10 < angle < 10:
                angles.append(angle)
        
        if not angles:
            return image
        
        median_angle = np.median(angles)
        print(f"Detected Skew Angle: {median_angle:.2f} degrees")
        
        if abs(median_angle) < 0.05: # Stricter threshold
            return image

        # 3. Rotate
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated

    def detect_strips_y(self, image, debug=False):
        """
        Detects Y-intervals of the 8 charts using the full width of the image.
        """
        binary = self.preprocess_image(image)
        grid_mask = self.get_grid_mask(binary)
        h, w = grid_mask.shape

        # 1. Find Global X-Bounds first to narrow down the search area
        # This prevents wide margins from skewing the width-based threshold
        col_projection = np.sum(grid_mask, axis=0) / 255.0
        # Columns with > 5% vertical fill are considered part of the grid
        is_grid_col = col_projection > (h * 0.05)
        
        runs = []
        in_run = False
        r_start = 0
        for x, val in enumerate(is_grid_col):
            if val and not in_run:
                in_run = True
                r_start = x
            elif not val and in_run:
                in_run = False
                runs.append((r_start, x))
        if in_run: runs.append((r_start, w))
        
        # Default to full width if no distinct column found
        search_x_start, search_x_end = 0, w
        if runs:
            main_run = max(runs, key=lambda r: r[1]-r[0])
            search_x_start, search_x_end = main_run
        
        search_width = search_x_end - search_x_start
        print(f"Global Grid Column detected: {search_x_start}-{search_x_end} (Width: {search_width})")

        # 2. Project horizontally WITHIN the grid column
        grid_crop = grid_mask[:, search_x_start:search_x_end]
        row_projection = np.sum(grid_crop, axis=1) / 255.0
        
        # Threshold: Row has significant grid content relative to the CHART width
        # Lowered to 5% to be robust against broken lines
        is_grid_row = row_projection > (search_width * 0.05) 
        
        y_ranges = []
        in_run = False
        r_start = 0
        for y, val in enumerate(is_grid_row):
            if val and not in_run:
                in_run = True
                r_start = y
            elif not val and in_run:
                in_run = False
                if (y - r_start) > 20: # Lower minimum height
                    y_ranges.append((r_start, y))
        if in_run and (h - r_start) > 20: 
             y_ranges.append((r_start, h))
        
        # Merge close blocks
        merged_ranges = []
        if y_ranges:
            curr_start, curr_end = y_ranges[0]
            for i in range(1, len(y_ranges)):
                next_start, next_end = y_ranges[i]
                gap = next_start - curr_end
                if gap < 20: # Merge small gaps
                    curr_end = next_end
                else:
                    merged_ranges.append((curr_start, curr_end))
                    curr_start, curr_end = next_start, next_end
            merged_ranges.append((curr_start, curr_end))
        
        # Selection logic
        final_strips = []
        if len(merged_ranges) == 8:
            final_strips = merged_ranges
        elif len(merged_ranges) == 1:
             print("Detected 1 block. Falling back to simple division.")
             sy, ey = merged_ranges[0]
             step = (ey - sy) / 8.0
             final_strips = [(int(sy + i*step), int(sy + (i+1)*step)) for i in range(8)]
        elif len(merged_ranges) == 0:
             print("Error: No grid blocks detected. Using full height division fallback.")
             # Fallback: Divide entire image into 8
             step = h / 8.0
             final_strips = [(int(i*step), int((i+1)*step)) for i in range(8)]
        else:
             print(f"Detected {len(merged_ranges)} blocks. Selecting 8 largest.")
             merged_ranges.sort(key=lambda r: r[1]-r[0], reverse=True)
             final_strips = merged_ranges[:8]
             final_strips.sort(key=lambda r: r[0])

        return final_strips

    def get_strip_x_bounds(self, strip_img):
        """
        Determines the exact X start and End for a specific strip image.
        """
        binary = self.preprocess_image(strip_img)
        grid_mask = self.get_grid_mask(binary)
        h, w = grid_mask.shape
        
        # Project vertically
        col_projection = np.sum(grid_mask, axis=0) / 255.0
        # Threshold: Col has grid lines (vertical ticks)
        is_grid_col = col_projection > (h * 0.1) 
        
        # Find largest run
        runs = []
        in_run = False
        r_start = 0
        for x, val in enumerate(is_grid_col):
            if val and not in_run:
                in_run = True
                r_start = x
            elif not val and in_run:
                in_run = False
                runs.append((r_start, x))
        if in_run: runs.append((r_start, w))
        
        if not runs:
            return 0, w
            
        main_run = max(runs, key=lambda r: r[1]-r[0])
        return max(0, main_run[0] - 2), min(w, main_run[1] + 2)

    def clean_dots(self, roi_img):
        binary = self.preprocess_image(roi_img)
        
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
        
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
        
        grid_mask = cv2.add(h_lines, v_lines)
        grid_mask = cv2.dilate(grid_mask, np.ones((3,3), np.uint8), iterations=2)
        
        dots_only = cv2.subtract(binary, grid_mask)
        dots_separated = cv2.erode(dots_only, np.ones((2,1), np.uint8), iterations=1)
        
        # Restored cleanup
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        dots_clean = cv2.morphologyEx(dots_separated, cv2.MORPH_OPEN, clean_kernel)
        
        return dots_clean

    def calculate_actual_time(self, date_str, start_hour, minute_offset):
        try:
            date_parts = date_str.replace('-', '/').split('/')
            year = int(date_parts[0])
            if year < 100: year += 1900
            
            dt = datetime(year, int(date_parts[1]), int(date_parts[2]), start_hour, 0, 0)
            actual_time = dt + timedelta(minutes=minute_offset)
            return actual_time.strftime("%Y-%m-%d %H:%M:%S")
        except:
            return "N/A"

    def pixel_to_value(self, px, max_px, val_start, val_end):
        ratio = float(px) / float(max_px)
        return val_start + ratio * (val_end - val_start)

    def process_chart(self, image_path, debug=False):
        metadata = self.extract_metadata(image_path)
        print(f"Metadata: {metadata.date} | Temp Range: {metadata.min_temp}-{metadata.max_temp}")

        original = cv2.imread(image_path)
        if original is None: raise ValueError("Image not found.")
        
        # 1. High-Precision Deskew
        deskewed = self.correct_skew(original)
        if debug: cv2.imwrite("debug_1_deskewed.jpg", deskewed)

        # 2. Detect Y Strips (Full Width)
        y_ranges = self.detect_strips_y(deskewed, debug)
        print(f"Detected {len(y_ranges)} strips.")

        extracted_data = []

        if debug:
            debug_img = deskewed.copy()
            cmap = cm.get_cmap('tab10')

        # 3. Process Each Strip Individually
        for i, (y_start, y_end) in enumerate(y_ranges):
            if i >= len(metadata.channels): break
            channel_freq = metadata.channels[i]
            
            # Extract full width strip first
            raw_strip = deskewed[y_start:y_end, :]
            
            # 4. Find Local X-Bounds for this specific strip
            x_start, x_end = self.get_strip_x_bounds(raw_strip)
            
            # Refine strip crop
            strip_roi = raw_strip[:, x_start:x_end]
            roi_w = x_end - x_start
            roi_h = y_end - y_start
            
            # Calibration for this specific strip
            total_hours = metadata.end_hour - metadata.start_hour
            pixels_per_second = roi_w / (total_hours * 3600)
            px_interval = pixels_per_second * POINT_INTERVAL_SECONDS

            if debug:
                # Draw the specific bounding box found for this strip
                cv2.rectangle(debug_img, (x_start, y_start), (x_end, y_end), (255, 255, 0), 2)
                rgba = cmap(i / 8.0)
                bgr_color = (int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255))

            # Clean and Detect
            strip_clean = self.clean_dots(strip_roi)
            contours, _ = cv2.findContours(strip_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                x, y, cw, ch = cv2.boundingRect(cnt)
                
                if area < 2: continue 

                is_cluster = cw > (px_interval * 1.1)
                points_to_process = []

                if not is_cluster:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        points_to_process.append((cx, cy))
                else:
                    curr_x = float(x) + (px_interval / 2.0)
                    end_x = x + cw
                    while curr_x < end_x:
                        sample_x = int(curr_x)
                        col_slice = strip_clean[y:y+ch, sample_x:sample_x+1]
                        if np.sum(col_slice) > 0:
                            ys, _ = np.where(col_slice > 0)
                            local_y = int(np.mean(ys))
                            points_to_process.append((sample_x, y + local_y))
                        curr_x += px_interval

                for (cx, cy) in points_to_process:
                    # Map X,Y to real values
                    time_min = self.pixel_to_value(cx, roi_w, 0, total_hours * 60)
                    actual_time_str = self.calculate_actual_time(metadata.date, metadata.start_hour, time_min)
                    
                    # Y -> Temp
                    temp_val = self.pixel_to_value(cy, roi_h, metadata.max_temp, metadata.min_temp)

                    extracted_data.append({
                        "Receiver": metadata.receiver,
                        "Date": metadata.date,
                        "Channel_Freq": channel_freq,
                        "Time_Offset_Min": round(time_min, 4),
                        "Actual_Time": actual_time_str,
                        "Log_Cal_Temp": round(temp_val, 4),
                    })
                    
                    if debug:
                        # Convert back to global coords for plotting
                        global_x = x_start + cx
                        global_y = y_start + cy
                        cv2.circle(debug_img, (global_x, global_y), 3, bgr_color, -1)

        df = pd.DataFrame(extracted_data)
        if not df.empty:
            df = df.sort_values(by=['Channel_Freq', 'Actual_Time'])
        
        output_csv = f"extracted_{metadata.date.replace('/','-')}.csv"
        df.to_csv(output_csv, index=False)
        print(f"Saved {len(df)} points to {output_csv}")
        
        if debug:
            cv2.imwrite("debug_final_robust.jpg", debug_img)
            plt.figure(figsize=(12, 18))
            plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
            plt.title("Detected Points (Per-Strip Bounds)")
            plt.axis('off')
            plt.show()

if __name__ == "__main__":
    IMAGE_FILE = "/global/cfs/projectdirs/m4895/RAE2Data/burstReceiver/burstRecieverIncompletePictures/cropped/convertedToPNG/PSFP-00103_PSFP-00820_RAE-B_MP28134_0008_cropped.png"
    digitizer = RadioAstronomyDigitizer(api_key=GOOGLE_API_KEY)
    try:
        digitizer.process_chart(IMAGE_FILE, debug=True)
    except Exception as e:
        import traceback
        traceback.print_exc()