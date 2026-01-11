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

# Updated Mock Data to include Temp range
MOCK_METADATA = ChartMetadata(
    receiver="RAE-B BURST RECEIVER 1",
    date="75/5/3",
    start_hour=1,
    end_hour=2,
    min_temp=4.0,  # Default lower bound
    max_temp=12.0, # Default upper bound
    channels=["0.110", "0.096", "0.083", "0.067", "0.055", "0.044", "0.035", "0.025"]
)

class RadioAstronomyDigitizer:
    def __init__(self, api_key):
        if not USE_MOCK_GEMINI:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp') # Using fast flash model

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

    def correct_skew(self, image):
        """
        Detects lines to determine skew angle and rotates the image to align the grid.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        if lines is None:
            return image

        angles = []
        for rho, theta in lines[:, 0]:
            # Convert to degrees
            angle = np.degrees(theta) - 90
            # We only care about nearly horizontal lines (-10 to 10 degrees)
            if -10 < angle < 10:
                angles.append(angle)
        
        if not angles:
            return image
        
        # Median angle is robust against outliers
        median_angle = np.median(angles)
        print(f"Detected Skew Angle: {median_angle:.2f} degrees")
        
        if abs(median_angle) < 0.1:
            return image

        # Rotate
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated

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
        # Dilate slightly to form a solid connected component for the whole table
        grid_mask = cv2.dilate(grid_mask, np.ones((3,3), np.uint8), iterations=2)
        return grid_mask

    def detect_strips(self, image, debug=False):
        """
        Robustly identifies 8 separated chart strips by projecting the grid mask.
        Returns:
            strips: List of (y_start, y_end) tuples
            x_bounds: (x_start, x_end) tuple for the global width
        """
        binary = self.preprocess_image(image)
        grid_mask = self.get_grid_mask(binary)
        h, w = grid_mask.shape

        # 1. Determine X-bounds (Global Width)
        # Find the main column of grids by projecting vertically
        col_projection = np.sum(grid_mask, axis=0) / 255.0
        # Simple threshold: Columns with > 5% filled with lines
        is_grid_col = col_projection > (h * 0.05)
        
        # Find largest continuous horizontal run (the graph column)
        x_start, x_end = 0, w
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
        
        if runs:
            # Pick largest run
            main_run = max(runs, key=lambda r: r[1]-r[0])
            # Add small margin
            x_start = max(0, main_run[0] - 5)
            x_end = min(w, main_run[1] + 5)
        
        print(f"Detected Global X-Bounds: {x_start}-{x_end}")

        # 2. Determine Y-Bounds (Separate Strips)
        # Project horizontally to find rows that have grid lines
        # Only look within the x-bounds we just found to avoid side noise
        grid_crop = grid_mask[:, x_start:x_end]
        row_projection = np.sum(grid_crop, axis=1) / 255.0
        
        # Threshold: Row has significant grid content
        is_grid_row = row_projection > ((x_end - x_start) * 0.05)
        
        y_ranges = []
        in_run = False
        r_start = 0
        for y, val in enumerate(is_grid_row):
            if val and not in_run:
                in_run = True
                r_start = y
            elif not val and in_run:
                in_run = False
                if (y - r_start) > 30: # Minimum height to be a chart
                    y_ranges.append((r_start, y))
        if in_run: 
             if (h - r_start) > 30:
                y_ranges.append((r_start, h))
        
        # 3. Merge / Filter logic
        # Sometimes a single chart has a tiny gap in detection. Merge close blocks.
        merged_ranges = []
        if y_ranges:
            curr_start, curr_end = y_ranges[0]
            for i in range(1, len(y_ranges)):
                next_start, next_end = y_ranges[i]
                gap = next_start - curr_end
                # If gap is small (e.g. < 15px), it's likely a glitch in the same chart
                if gap < 20: 
                    curr_end = next_end # Merge
                else:
                    merged_ranges.append((curr_start, curr_end))
                    curr_start, curr_end = next_start, next_end
            merged_ranges.append((curr_start, curr_end))
        
        # 4. Final Selection
        # We expect 8. 
        if len(merged_ranges) == 8:
            print("Successfully detected exactly 8 separated grids.")
            final_strips = merged_ranges
        elif len(merged_ranges) == 1:
             print("Detected 1 block. Falling back to simple division.")
             sy, ey = merged_ranges[0]
             step = (ey - sy) / 8.0
             final_strips = [(int(sy + i*step), int(sy + (i+1)*step)) for i in range(8)]
        else:
             print(f"Detected {len(merged_ranges)} blocks. Selecting 8 largest.")
             # Sort by height to find the main charts (ignoring small noise blocks)
             merged_ranges.sort(key=lambda r: r[1]-r[0], reverse=True)
             final_strips = merged_ranges[:8]
             # Sort back by Y position
             final_strips.sort(key=lambda r: r[0])
             
             if len(final_strips) < 8:
                 # Fallback if fewer than 8 found
                 print("Warning: Found fewer than 8 charts. Using available blocks.")

        return final_strips, (x_start, x_end)

    def clean_dots(self, roi_img):
        """Remove grid lines from the cropped ROI to leave only dots."""
        binary = self.preprocess_image(roi_img)
        
        # Re-calculate grid mask specifically for the crop
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
        
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
        
        grid_mask = cv2.add(h_lines, v_lines)
        grid_mask = cv2.dilate(grid_mask, np.ones((3,3), np.uint8), iterations=2)
        
        dots_only = cv2.subtract(binary, grid_mask)
        
        # Break horizontal links between close dots
        dots_separated = cv2.erode(dots_only, np.ones((2,1), np.uint8), iterations=1)
        
        # RESTORED: Aggressive 'Open' operation to clean scanning artifacts.
        # This removes small noise but keeps the dots if they are substantial enough.
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
        # 1. Metadata
        metadata = self.extract_metadata(image_path)
        print(f"Metadata: {metadata.date} | Temp Range: {metadata.min_temp}-{metadata.max_temp}")

        # 2. Load & Deskew
        original = cv2.imread(image_path)
        if original is None: raise ValueError("Image not found.")
        
        deskewed = self.correct_skew(original)
        if debug: cv2.imwrite("debug_1_deskewed.jpg", deskewed)

        # 3. Detect Strips (Gap-Aware)
        # Returns list of (y_start, y_end) and global (x_start, x_end)
        y_ranges, (x_start, x_end) = self.detect_strips(deskewed, debug)
        roi_w = x_end - x_start

        # 4. Calibration
        total_hours = metadata.end_hour - metadata.start_hour
        total_seconds = total_hours * 3600
        pixels_per_second = roi_w / total_seconds
        px_interval = pixels_per_second * POINT_INTERVAL_SECONDS
        print(f"Sampling Interval: {px_interval:.2f} pixels")

        extracted_data = []

        if debug:
            debug_img = deskewed.copy()
            cmap = cm.get_cmap('tab10')

        # 5. Iterate over detected strips
        for i, (y_start, y_end) in enumerate(y_ranges):
            if i >= len(metadata.channels): break
            
            channel_freq = metadata.channels[i]
            
            if debug:
                # Draw Box around detected strip
                cv2.rectangle(debug_img, (x_start, y_start), (x_end, y_end), (255, 255, 0), 2)
                rgba = cmap(i / 8.0)
                bgr_color = (int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255))

            # Crop & Clean specific strip
            strip_roi = deskewed[y_start:y_end, x_start:x_end]
            strip_clean = self.clean_dots(strip_roi)
            current_strip_h = y_end - y_start
            
            contours, _ = cv2.findContours(strip_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                x, y, cw, ch = cv2.boundingRect(cnt)
                
                # CHANGED: Lower threshold to capture small/faint points
                if area < 5: continue 

                # CHANGED: More sensitive cluster detection.
                # If width > 1.1x interval (just slightly wider than one point spacing),
                # treat as a cluster to ensure we split touching neighbors.
                is_cluster = cw > (px_interval * 1.5)
                
                points_to_process = []

                if not is_cluster:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        points_to_process.append((cx, cy))
                else:
                    # Resample Cluster
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

                # Map Points
                for (cx, cy) in points_to_process:
                    # Global coordinates for debug
                    global_x = x_start + cx
                    global_y = y_start + cy
                    
                    # X -> Time (Relative to strip width)
                    time_min = self.pixel_to_value(cx, roi_w, 0, total_hours * 60)
                    actual_time_str = self.calculate_actual_time(metadata.date, metadata.start_hour, time_min)
                    
                    # Y -> Temp
                    temp_val = self.pixel_to_value(cy, current_strip_h, metadata.max_temp, metadata.min_temp)

                    extracted_data.append({
                        "Receiver": metadata.receiver,
                        "Date": metadata.date,
                        "Channel_Freq": channel_freq,
                        "Time_Offset_Min": round(time_min, 4),
                        "Actual_Time": actual_time_str,
                        "Log_Cal_Temp": round(temp_val, 4),
                    })
                    
                    if debug:
                        cv2.circle(debug_img, (global_x, global_y), 3, bgr_color, -1)

        # Save
        df = pd.DataFrame(extracted_data)
        if not df.empty:
            df = df.sort_values(by=['Channel_Freq', 'Actual_Time'])
        
        output_csv = f"extracted_{metadata.date.replace('/','-')}.csv"
        df.to_csv(output_csv, index=False)
        print(f"Saved {len(df)} points to {output_csv}")
        
        if debug:
            output_debug = "debug_final_robust.jpg"
            cv2.imwrite(output_debug, debug_img)
            
            plt.figure(figsize=(12, 18))
            plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
            plt.title("Detected Points (Sensitive)")
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