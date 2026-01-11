import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent


def main():
    rng = np.random.default_rng(42)

    start = pd.Timestamp("1974-12-01 00:00:00")
    times = pd.date_range(start, periods=51, freq="1min")

    event_specs = [
        ("E1", start + pd.Timedelta(minutes=10), "DISAPPEARANCE"),
        ("E2", start + pd.Timedelta(minutes=20), "REAPPEARANCE"),
        ("E3", start + pd.Timedelta(minutes=30), "DISAPPEARANCE"),
        ("E4", start + pd.Timedelta(minutes=40), "REAPPEARANCE"),
    ]

    channel_freqs = [
        0.025, 0.035, 0.044, 0.055, 0.067, 0.083, 0.096, 0.110,
        0.130, 0.155, 0.185, 0.210, 0.250, 0.292, 0.360, 0.425,
        0.475, 0.600, 0.737, 0.870, 1.030, 1.27, 1.45, 1.85,
        2.20, 2.80, 3.93, 4.70, 6.55, 9.18, 11.8, 13.1,
    ]

    occulted_channels = {10, 11, 12}
    drop_db = 1.0

    event_times = {spec[1]: spec[2] for spec in event_specs}
    occulted = False

    rows = []
    for t in times:
        if t in event_times:
            etype = event_times[t]
            if etype == "DISAPPEARANCE":
                occulted = True
            elif etype == "REAPPEARANCE":
                occulted = False
        for channel_id, freq in enumerate(channel_freqs):
            base = -100.0 + rng.normal(0.0, 0.1)
            if occulted and channel_id in occulted_channels:
                base -= drop_db
            rows.append(
                {
                    "time": t,
                    "channel_id": channel_id,
                    "center_frequency_mhz": freq,
                    "power": base,
                }
            )

    measurements_df = pd.DataFrame(rows)
    measurements_df.to_csv(OUTPUT_DIR / "synthetic_measurements.csv", index=False)

    sep_defaults = {f"{body}_sep_deg": 10.0 for body in [
        "sun", "mercury", "venus", "earth", "mars",
        "jupiter", "saturn", "uranus", "neptune",
    ]}

    events_rows = []
    for event_id, event_time, event_type in event_specs:
        row = {
            "event_id": event_id,
            "event_time": event_time,
            "event_type": event_type,
            "planet": "SATURN",
        }
        row.update(sep_defaults)
        events_rows.append(row)

    events_df = pd.DataFrame(events_rows)
    events_df.to_csv(OUTPUT_DIR / "synthetic_events.csv", index=False)


if __name__ == "__main__":
    main()
