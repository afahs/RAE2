#!/usr/bin/env python
"""Build raw-power event plots with baseline and fitted step model overlays."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rylevonberg.detection import baseline_matrix, event_template
from rylevonberg.util import datetime_ns


BASE = Path(__file__).resolve().parents[1]
EXAMPLES = BASE / "outputs/slide_assets/individual_event_examples/individual_event_examples.csv"
OUT_DIR = BASE / "outputs/slide_assets/individual_event_examples/model_overlays"

CLEAN_DEFAULT = BASE / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
CLEAN_SUN_DEMO = BASE / "outputs/demo_sun_postnov1974_multiband/01_ingest/cleaned_timeseries.csv"


def _fit_models(t_rel: np.ndarray, y: np.ndarray, event_type: str, timing_offset: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline = baseline_matrix(t_rel, 1)
    beta0, *_ = np.linalg.lstsq(baseline, y, rcond=None)
    baseline_only = baseline @ beta0

    step = event_template(t_rel, event_type, smooth_seconds=0.0, timing_offset_sec=timing_offset)
    design = np.column_stack([baseline, step])
    beta1, *_ = np.linalg.lstsq(design, y, rcond=None)
    model = design @ beta1
    step_component = beta1[-1] * step
    return baseline_only, model, step_component


def _clean_for_source(source: str, cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    key = "sun_demo" if source.lower() == "sun" else "default"
    if key not in cache:
        path = CLEAN_SUN_DEMO if key == "sun_demo" else CLEAN_DEFAULT
        cache[key] = read_table(path, parse_dates=["time"], low_memory=False)
    return cache[key]


def plot_one(clean: pd.DataFrame, event: pd.Series, output_path: Path) -> None:
    source = str(event["source"]).lower()
    freq_band = event.get("internal_frequency_band")
    antenna = event.get("antenna")
    window_s = 900.0 if source == "jupiter" else 600.0
    event_time = pd.Timestamp(event["predicted_event_time"])

    group = clean
    if pd.notna(freq_band):
        group = group[group["frequency_band"] == int(freq_band)]
    if pd.notna(antenna):
        group = group[group["antenna"].astype(str) == str(antenna)]

    rel = (datetime_ns(group["time"]) - event_time.value) / 1e9
    keep = np.abs(rel) <= window_s
    local = group.loc[keep].copy()
    local_rel = rel[keep]
    valid = np.isfinite(local["power"].to_numpy(dtype=float))
    if "is_valid" in local.columns:
        valid &= local["is_valid"].to_numpy(dtype=bool)

    t_fit = local_rel[valid].astype(float)
    y_fit = local.loc[valid, "power"].to_numpy(dtype=float)
    order = np.argsort(t_fit)
    t_fit = t_fit[order]
    y_fit = y_fit[order]
    baseline_only, model, step_component = _fit_models(
        t_fit,
        y_fit,
        str(event["event_type"]),
        float(event["timing_offset_sec"]),
    )

    fig, (ax, ax_resid) = plt.subplots(
        2,
        1,
        figsize=(8, 5.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )
    ax.plot(local_rel / 60.0, local["power"], ".", ms=3, color="0.7", label="raw power samples")
    ax.plot(t_fit / 60.0, y_fit, ".", ms=4, color="#1f77b4", label="valid samples used in fit")
    ax.plot(t_fit / 60.0, baseline_only, "--", lw=1.8, color="#ff7f0e", label="baseline-only fit")
    ax.plot(t_fit / 60.0, model, "-", lw=2.0, color="#d62728", label="baseline + step fit")
    ax.axvline(0.0, color="black", lw=1.1, label="predicted event")
    ax.axvline(float(event["timing_offset_sec"]) / 60.0, color="#17becf", lw=1.1, ls=":", label="fitted step time")
    ax.set_ylabel("Original Ryle-Vonberg power")
    ax.set_title(
        f"{event['source']} {event['event_type']} {event['frequency']} {event['antenna']} "
        f"SNR={float(event['best_abs_snr']):.2f}, dt={float(event['timing_offset_sec']):.0f}s"
    )
    ax.legend(fontsize=7, ncol=2)

    ax_resid.axhline(0.0, color="0.2", lw=0.8)
    ax_resid.plot(t_fit / 60.0, y_fit - model, ".", ms=3, color="0.25")
    ax_resid.set_xlabel("Relative time from predicted event (min)")
    ax_resid.set_ylabel("resid.")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    examples = read_table(EXAMPLES)
    cache: dict[str, pd.DataFrame] = {}
    rows = []
    for i, event in examples.iterrows():
        clean = _clean_for_source(str(event["source"]), cache)
        out = OUT_DIR / f"model_overlay_{i:04d}_{str(event['source']).lower()}.png"
        plot_one(clean, event, out)
        row = event.to_dict()
        row["model_overlay"] = str(out)
        rows.append(row)

    index = pd.DataFrame(rows)
    index.to_csv(OUT_DIR / "model_overlay_index.csv", index=False)

    fig, axes = plt.subplots(3, 2, figsize=(13, 14), constrained_layout=True)
    for ax, (_, row) in zip(axes.ravel(), index.iterrows()):
        img = mpimg.imread(row["model_overlay"])
        ax.imshow(img)
        ax.axis("off")
    fig.savefig(OUT_DIR / "model_overlay_montage.png", dpi=170)
    plt.close(fig)

    print(index[["source", "event_type", "frequency", "antenna", "best_abs_snr", "timing_offset_sec", "model_overlay"]].to_string(index=False))


if __name__ == "__main__":
    main()
