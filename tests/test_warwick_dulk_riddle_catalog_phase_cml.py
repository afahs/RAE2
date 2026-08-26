from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.plot_jupiter_burst_count_vs_cml import cml_frequency_summary
from scripts.plot_warwick_dulk_riddle_catalog_phase_cml import load_catalog, parse_frequency_range


def test_parse_frequency_range_sorts_ambiguous_reversed_range() -> None:
    assert parse_frequency_range("11-20") == (11.0, 20.0)
    assert parse_frequency_range("25-18") == (18.0, 25.0)


def test_load_catalog_handles_next_day_hhmm(tmp_path) -> None:
    path = tmp_path / "wdr.csv"
    pd.DataFrame(
        [
            {
                "date": "1975-02-23",
                "event_start_hhmm": "2535",
                "event_end_hhmm": "2550",
                "intensity": "1",
                "burstiness": "MODERATE",
                "reported_freq_range_mhz": "13-17",
                "pdf_page": "24",
                "notes": "",
            }
        ]
    ).to_csv(path, index=False)
    out = load_catalog(path)
    assert out.loc[0, "event_start_time"] == pd.Timestamp("1975-02-24 01:35")
    assert out.loc[0, "event_end_time"] == pd.Timestamp("1975-02-24 01:50")
    assert out.loc[0, "reported_freq_mid_mhz"] == 15.0


def test_cml_frequency_summary_keeps_zero_exposure_bins_nan() -> None:
    samples = pd.DataFrame(
        {
            "frequency_band": [1, 1],
            "frequency_mhz": [0.45, 0.45],
            "jupiter_cml_spice_deg": [1.0, 2.0],
        }
    )
    bursts = pd.DataFrame(
        {
            "frequency_band": [1],
            "frequency_mhz": [0.45],
            "jupiter_cml_spice_deg": [1.5],
        }
    )

    result = cml_frequency_summary(samples, bursts, bin_deg=10.0)

    populated = result[result["cml_bin_deg"].eq(5.0)].iloc[0]
    empty = result[result["cml_bin_deg"].eq(15.0)].iloc[0]
    assert populated["n_bursts"] == 1
    assert populated["burst_rate_per_100k_samples"] == 50_000.0
    assert np.isnan(empty["burst_rate_per_100k_samples"])
