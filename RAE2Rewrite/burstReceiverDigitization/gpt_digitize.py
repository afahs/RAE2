import cv2
import numpy as np
import pandas as pd
from openai import OpenAI
import argparse
import os
import json
import base64

# --- CONFIGURATION ---
VALID_CHANNELS = [
    "0.025", "0.035", "0.044", "0.055", "0.067", "0.083", "0.096", "0.110",
    "0.130", "0.155", "0.185", "0.210", "0.250", "0.292", "0.360", "0.425",
    "0.475", "0.600", "0.737", "0.870", "1.030", "1.27", "1.45", "1.85",
    "2.20", "2.80", "3.93", "4.70", "6.55", "9.18", "11.8", "13.1"
]

def get_metadata_and_bounds_from_chatgpt(api_key, image_path, model="gpt-5-mini"):
    """
    Sends image to ChatGPT (OpenAI API) to extract metadata AND the graph's bounding box.
    """
    client = OpenAI(api_key=api_key)

    with open(image_path, "rb") as f:
        img_bytes = f.read()

    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"

    prompt = f"""
    Analyze this radio astronomy strip chart image. I need to find the metadata and the exact location of the data graph area.

    1. **Date & Time:** Find the handwritten date (YYYYMMDD) and time range (HHMM-HHMM) usually in the bottom-left.
    2. **Receiver:** Is it "Burst Receiver 1" or "2" in the top header?
    3. **Channels:** Identify the 8 vertical channel labels on the right margin, from TOP to BOTTOM. 
       They should match THESE valid channel values (kHz): {VALID_CHANNELS}.
    4. **Graph Bounding Box:** Identify the main rectangular area containing the 8 horizontal data strips. 
       Exclude the top header text, bottom footer text, and side margin labels.
       Return the bounding box as percentages of the image size in this format: [ymin_pct, xmin_pct, ymax_pct, xmax_pct].
       For example, [0.10, 0.05, 0.90, 0.95] would mean the graph starts 10% down and 5% from the left.

    Return a pure JSON object:
    {{
        "date": "YYYYMMDD",
        "time_str": "HHMM-HHMM",
        "receiver": "BurstReceiverX",
        "channels": ["110", "096", "...", "025"],
        "bbox_pct": [ymin, xmin, ymax, xmax]
    }}
    """

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{img_b64}"
                            },
                        },
                    ],
                }
            ],
        )

        text = response.choices[0].message.content
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        print(f"AI Error ({model}): {e}")
        return None

def process_image_data_with_bounds(image_path, channels_list, bbox_pct):
    # ... your existing function unchanged ...
    img = cv2.imread(image_path)
    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    _, binary = cv2.threshold(inv, 100, 255, cv2.THRESH_BINARY)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    grid_mask = cv2.add(cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel),
                        cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel))
    grid_mask = cv2.dilate(grid_mask, np.ones((5,5), np.uint8), iterations=1)
    dots_clean = cv2.morphologyEx(cv2.subtract(binary, grid_mask),
                                  cv2.MORPH_OPEN, np.ones((2,2), np.uint8))

    ymin_pct, xmin_pct, ymax_pct, xmax_pct = bbox_pct
    y_start = int(h_img * ymin_pct)
    y_end = int(h_img * ymax_pct)
    x_start = int(w_img * xmin_pct)
    x_end = int(w_img * xmax_pct)

    print(f"  > AI defined graph area: Rows {y_start}-{y_end}, Cols {x_start}-{x_end}")

    num_channels = len(channels_list)
    channel_height = (y_end - y_start) / num_channels

    num_samples = int(60.0 / (7.68 / 60.0)) + 1
    sample_x_coords = np.linspace(x_start, x_end, num_samples)

    extracted_rows = []
    for ch_idx in range(num_channels):
        label = channels_list[ch_idx]
        ch_top = y_start + ch_idx * channel_height

        for t_idx, x_float in enumerate(sample_x_coords):
            x = int(round(x_float))
            time_min = t_idx * (7.68 / 60.0)

            if x >= w_img:
                break

            strip = dots_clean[int(ch_top+2):int(ch_top+channel_height-2), x-1:x+2]

            val = ""
            if np.any(strip):
                M = cv2.moments(strip)
                if M["m00"] > 0:
                    cy = M["m01"] / M["m00"]
                    pixel_y = int(ch_top+2) + cy
                    norm_y = (pixel_y - ch_top) / channel_height
                    val = round(12.0 - norm_y * (12.0 - 4.0), 4)

            extracted_rows.append({
                "Time_Min": round(time_min, 4),
                "Channel": label,
                "Value": val
            })

    return extracted_rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "--key",
        help="OpenAI API Key",
        default=os.environ.get("OPENAI_API_KEY")
    )
    parser.add_argument(
        "--model",
        help="OpenAI model name (e.g. gpt-4o-mini, gpt-4o)",
        default="gpt-4o-mini"
    )
    args = parser.parse_args()

    if not args.key:
        print("Error: Missing OPENAI_API_KEY.")
        return

    print(f"--- Processing {args.image} with model {args.model} ---")

    print("Step 1: Asking AI to find metadata and graph boundaries...")
    meta = get_metadata_and_bounds_from_chatgpt(args.key, args.image, model=args.model)

    if not meta or 'bbox_pct' not in meta:
        print("Failed to get valid data from AI.")
        return

    print(f"  > Found Date: {meta.get('date')}")
    print(f"  > Found Box:  {meta.get('bbox_pct')}")

    print("Step 2: Extracting data points from defined area...")
    data_rows = process_image_data_with_bounds(
        args.image, meta.get('channels'), meta.get('bbox_pct')
    )

    final_data = []
    for row in data_rows:
        row['Date'] = meta.get('date')
        row['Time_Range'] = meta.get('time_str')
        row['Receiver'] = meta.get('receiver')
        final_data.append(row)

    df = pd.DataFrame(final_data)
    filename = f"{meta.get('date')}_{meta.get('time_str')}_{meta.get('receiver')}_{args.model}.csv"
    filename = filename.replace("/", "").replace(":", "")
    df.to_csv(filename, index=False)
    print(f"--- Success! Saved to {filename} ---")

if __name__ == "__main__":
    main()
