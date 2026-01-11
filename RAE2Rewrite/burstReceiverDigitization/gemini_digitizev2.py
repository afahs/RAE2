import cv2
import numpy as np
import pandas as pd
import google.generativeai as genai
import json
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from dataclasses import dataclass
from typing import List

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------

# Replace with your actual Gemini API Key
GOOGLE_API_KEY = "***REMOVED***"

# Set to True to save API calls while debugging CV logic
USE_MOCK_GEMINI = False

# Known physical constraint from prompt
POINT_INTERVAL_SECONDS = 7.68

# The bounds provided by the user (Top Y, Bottom Y)
RAW_BOUNDS = [
    (50, 425), (425, 800), (800, 1175), (1175, 1550),
    (1550, 1925), (1925, 2275), (2275, 2650), (2650, 3025)
]

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
    channels: List[str]

# Mock data matching the sample image for CV debugging
MOCK_METADATA = ChartMetadata(
    receiver="RAE-B BURST RECEIVER 1",
    date="75/5/3",
    start_hour=1,
    end_hour=2,
    channels=["0.110", "0.096", "0.083", "0.067", "0.055", "0.044", "0.035", "0.025"]
)

class RadioAstronomyDigitizer:
    def __init__(self, api_key):
        if not USE_MOCK_GEMINI:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-flash-latest')

    def extract_metadata(self, image_path: str) -> ChartMetadata:
        if USE_MOCK_GEMINI:
            print("--- WARNING: USING MOCK METADATA (Skipping API call) ---")
            return MOCK_METADATA

        print(f"Analyzing metadata for {image_path} with Gemini...")
        file = genai.upload_file(image_path)
        
        prompt = f"""
        Analyze this radio astronomy chart. I need a JSON output containing:
        1. 'receiver': The text at the very top of the image.
        2. 'date': The date at the bottom left (format YYYY/M/D).
        3. 'start_hour': The integer hour label at the far left of the time axis.
        4. 'end_hour': The integer hour label at the far right of the time axis.
        5. 'channels': A list of exactly 8 frequencies found on the right vertical edge.
        
        CRITICAL: The channels MUST be selected from this specific valid list:
        {json.dumps(VALID_FREQUENCIES)}
        
        Return ONLY valid JSON.
        """

        response = self.model.generate_content([prompt, file])
        text = response.text.replace('```json', '').replace('```', '')
        data = json.loads(text)
        
        return ChartMetadata(
            receiver=data['receiver'],
            date=data['date'],
            start_hour=int(data['start_hour']),
            end_hour=int(data['end_hour']),
            channels=data['channels']
        )

    def preprocess_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 25, 15)
        return binary

    def remove_grid_and_clean(self, binary_img):
        # 1. Grid Detection
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        h_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, h_kernel)
        
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        v_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, v_kernel)
        
        grid_mask = cv2.add(h_lines, v_lines)
        grid_mask = cv2.dilate(grid_mask, np.ones((3,3), np.uint8), iterations=2)

        # 2. Subtract Grid
        dots_only = cv2.subtract(binary_img, grid_mask)
        
        # 3. Separate Clusters (Light erosion)
        separation_kernel = np.ones((2, 2), np.uint8) 
        dots_separated = cv2.erode(dots_only, separation_kernel, iterations=1)
        
        # 4. Clean noise
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        dots_clean = cv2.morphologyEx(dots_separated, cv2.MORPH_OPEN, clean_kernel, iterations=1)
        
        return dots_clean

    def pixel_to_value(self, px, range_min_px, range_max_px, val_start, val_end):
        if range_max_px == range_min_px: return val_start
        ratio = float(px - range_min_px) / float(range_max_px - range_min_px)
        return val_start + ratio * (val_end - val_start)

    def get_scaled_bounds(self, img_h):
        """Scales hardcoded bounds to image height."""
        reference_h = 3300 
        scale = img_h / reference_h
        if 0.95 < scale < 1.05: return RAW_BOUNDS
        return [(int(s*scale), int(e*scale)) for (s, e) in RAW_BOUNDS]

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

    def process_chart(self, image_path, debug=False):
        metadata = self.extract_metadata(image_path)
        
        original = cv2.imread(image_path)
        h, w = original.shape[:2]

        margin_x_left = int(w * 0.1)
        margin_x_right = int(w * 0.05)
        
        roi = original[:, margin_x_left:w-margin_x_right]
        roi_h, roi_w = roi.shape[:2]
        
        # --- CALCULATE PIXEL SPACING ---
        # Calculate how many pixels represent 7.68 seconds
        total_hours = metadata.end_hour - metadata.start_hour
        total_seconds = total_hours * 3600
        pixels_per_second = roi_w / total_seconds
        px_interval = pixels_per_second * POINT_INTERVAL_SECONDS
        
        print(f"Time Scale: {total_seconds}s total width.")
        print(f"Sampling Interval: ~{px_interval:.2f} pixels (corresponds to 7.68s)")

        binary = self.preprocess_image(roi)
        clean_dots_img = self.remove_grid_and_clean(binary)

        extracted_data = []
        bounds = self.get_scaled_bounds(roi_h)

        if debug:
            debug_img = roi.copy()
            cmap = cm.get_cmap('tab10')

        # Iterate over channels
        for i, (y_start, y_end) in enumerate(bounds):
            if i >= len(metadata.channels): break
            channel_freq = metadata.channels[i]
            
            if debug:
                # Cyan boundary lines
                cv2.line(debug_img, (0, y_start), (roi_w, y_start), (255, 255, 0), 2)
                cv2.line(debug_img, (0, y_end), (roi_w, y_end), (255, 255, 0), 2)
                
                # Channel color
                rgba = cmap(i / 8.0)
                bgr_color = (int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255))

            strip = clean_dots_img[y_start:y_end, :]
            strip_height = y_end - y_start
            
            contours, _ = cv2.findContours(strip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                x, y, cw, ch = cv2.boundingRect(cnt)
                
                if area < 10: continue # Skip noise

                # Determine if this contour is a single dot or a cluster
                # If width is significantly larger than our calculated pixel interval, it's a cluster
                is_cluster = cw > (px_interval * 1.5)

                points_to_process = []

                if not is_cluster:
                    # Case 1: Single point (calculate centroid)
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        points_to_process.append((cx, cy))
                else:
                    # Case 2: Cluster / Line of points
                    # We iterate through the bounding box in steps of px_interval
                    # This recovers the data points hidden inside the solid line
                    
                    # Start slightly offset to center the sampling
                    curr_x = float(x) + (px_interval / 2.0)
                    end_x = x + cw
                    
                    while curr_x < end_x:
                        sample_x = int(curr_x)
                        
                        # Optimization: Ensure we are actually ON ink
                        # Look at the vertical slice at sample_x inside the bounding box
                        # and find the center of the ink at that specific X
                        col_slice = strip[y:y+ch, sample_x:sample_x+1]
                        
                        if np.sum(col_slice) > 0: # If there is ink here
                            # Find weighted Y center of this slice
                            ys, _ = np.where(col_slice > 0)
                            if len(ys) > 0:
                                local_y = int(np.mean(ys))
                                points_to_process.append((sample_x, y + local_y))
                        
                        curr_x += px_interval

                # Add processed points to dataset
                for (cx, cy) in points_to_process:
                    global_y = y_start + cy
                    
                    time_min = self.pixel_to_value(cx, 0, roi_w, 0, total_hours * 60)
                    temp_val = self.pixel_to_value(cy, 0, strip_height, 12, 4)
                    
                    actual_time_str = self.calculate_actual_time(metadata.date, metadata.start_hour, time_min)

                    extracted_data.append({
                        "Receiver": metadata.receiver,
                        "Date": metadata.date,
                        "Channel_Freq": channel_freq,
                        "Time_Offset_Min": round(time_min, 4),
                        "Actual_Time": actual_time_str,
                        "Log_Cal_Temp": round(temp_val, 4),
                    })
                    
                    if debug:
                        cv2.circle(debug_img, (cx, global_y), 3, bgr_color, -1)

        # Final Save
        df = pd.DataFrame(extracted_data)
        df = df.sort_values(by=['Channel_Freq', 'Time_Offset_Min'])
        
        output_csv = f"extracted_{metadata.date.replace('/','-')}.csv"
        df.to_csv(output_csv, index=False)
        print(f"Processing complete. Saved {len(df)} data points to {output_csv}")
        
        if debug:
            output_debug = "debug_final_result.jpg"
            cv2.imwrite(output_debug, debug_img)
            
            plt.figure(figsize=(12, 18))
            plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
            plt.title("Detected Points (Resampled Clusters)")
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