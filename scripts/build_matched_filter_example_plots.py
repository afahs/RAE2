#!/usr/bin/env python
"""Build slide-ready matched-filter event examples."""

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

from rylevonberg.constants import add_frequency_mhz_column
from rylevonberg.detection import baseline_matrix, event_template
from rylevonberg.util import datetime_ns


BASE = Path(__file__).resolve().parents[1]
OUT_DIR = BASE / "outputs/slide_assets/matched_filter_examples"
CLEAN_DEFAULT = BASE / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"


def _read_scored(path: Path) -> pd.DataFrame:
    return add_frequency_mhz_column(read_table(path, low_memory=False))


def _selected_events() -> pd.DataFrame:
    earth_sun = _read_scored(BASE / "outputs/control_survey_earth_sun_postnov1974_v2/07_scoring/scored_detections.csv")
    jupiter = _read_scored(BASE / "outputs/control_survey_jupiter_postnov1974_v1/09_window_sweep/window_900s/scored_detections.csv")
    tables = []
    for source, df, n in [("earth", earth_sun, 2), ("sun", earth_sun, 2), ("jupiter", jupiter, 2)]:
        work = df[
            df["source_name"].astype(str).str.lower().eq(source)
            & df["antenna"].astype(str).eq("rv2_coarse")
            & (pd.to_numeric(df["quality_clean_fraction"], errors="coerce") >= 0.75)
            & pd.to_numeric(df["matched_snr"], errors="coerce").notna()
        ].copy()
        work["abs_matched_snr"] = pd.to_numeric(work["matched_snr"], errors="coerce").abs()
        tables.append(work.sort_values("abs_matched_snr", ascending=False).head(n))
    return pd.concat(tables, ignore_index=True)


def _matched_components(clean: pd.DataFrame, event: pd.Series, window_s: float) -> dict[str, np.ndarray | float]:
    group = clean
    group = group[group["frequency_band"] == int(event["frequency_band"])]
    group = group[group["antenna"].astype(str) == str(event["antenna"])]

    event_time = pd.Timestamp(event["predicted_event_time"])
    rel = (datetime_ns(group["time"]) - event_time.value) / 1e9
    keep = np.abs(rel) <= window_s
    local = group.loc[keep].copy()
    local_rel = rel[keep]
    valid = np.isfinite(local["power"].to_numpy(dtype=float))
    if "is_valid" in local.columns:
        valid &= local["is_valid"].to_numpy(dtype=bool)

    t = local_rel[valid].astype(float)
    y = local.loc[valid, "power"].to_numpy(dtype=float)
    order = np.argsort(t)
    t = t[order]
    y = y[order]

    baseline = baseline_matrix(t, 1)
    beta_y, *_ = np.linalg.lstsq(baseline, y, rcond=None)
    y_baseline = baseline @ beta_y
    yd = y - y_baseline

    template = event_template(t, str(event["event_type"]), smooth_seconds=0.0, timing_offset_sec=0.0)
    beta_h, *_ = np.linalg.lstsq(baseline, template, rcond=None)
    td = template - baseline @ beta_h
    denom = float(np.dot(td, td))
    amp = float(np.dot(yd, td) / denom) if denom > 0 else np.nan
    model_detrended = amp * td

    return {
        "local_rel": np.asarray(local_rel, dtype=float),
        "local_power": local["power"].to_numpy(dtype=float),
        "t": t,
        "y": y,
        "baseline": y_baseline,
        "yd": yd,
        "td": td,
        "amp": amp,
        "model_detrended": model_detrended,
    }


def _plot_event(clean: pd.DataFrame, event: pd.Series, output_path: Path) -> None:
    source = str(event["source_name"]).lower()
    window_s = 900.0 if source == "jupiter" else 600.0
    comp = _matched_components(clean, event, window_s)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.5, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.8]},
        constrained_layout=True,
    )
    ax = axes[0]
    ax.plot(comp["local_rel"] / 60.0, comp["local_power"], ".", ms=3, color="0.72", label="raw power")
    ax.plot(comp["t"] / 60.0, comp["y"], ".", ms=4, color="#1f77b4", label="valid samples")
    ax.plot(comp["t"] / 60.0, comp["baseline"], "--", lw=1.8, color="#ff7f0e", label="baseline fit")
    ax.axvline(0.0, color="black", lw=1.1, label="template centered at dt=0")
    ax.set_ylabel("Original power")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title(
        f"{event['source_name']} {event['event_type']} {float(event['frequency_mhz']):.2f} MHz {event['antenna']} "
        f"matched SNR={float(event['matched_snr']):.2f}"
    )

    ax2 = axes[1]
    ax2.axhline(0.0, color="0.25", lw=0.8)
    ax2.plot(comp["t"] / 60.0, comp["yd"], ".", ms=4, color="#1f77b4", label="detrended power")
    ax2.plot(comp["t"] / 60.0, comp["model_detrended"], "-", lw=2.0, color="#d62728", label="scaled matched template")
    ax2.plot(comp["t"] / 60.0, comp["td"] * np.nanstd(comp["yd"]) / max(np.nanstd(comp["td"]), 1e-12), ":", lw=1.4, color="#2ca02c", label="template shape")
    ax2.axvline(0.0, color="black", lw=1.1)
    ax2.set_xlabel("Relative time from predicted event (min)")
    ax2.set_ylabel("Detrended power")
    ax2.legend(fontsize=7, ncol=2)

    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _write_markdown(summary: pd.DataFrame) -> None:
    cols = [
        "source_name",
        "event_type",
        "predicted_event_time",
        "frequency",
        "antenna",
        "matched_snr",
        "matched_amp",
        "best_empirical_p",
        "quality_clean_fraction",
        "detection_grade",
    ]
    md = [
        "# Matched Filter Event Examples",
        "",
        "These examples show the matched-filter event-level result. The template is fixed at the predicted event time (`dt = 0`). The upper panel shows original power and the fitted local baseline. The lower panel shows detrended power and the scaled matched-filter template.",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in summary.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                if col in {"matched_snr", "matched_amp"}:
                    vals.append(f"{value:.3g}")
                elif col in {"best_empirical_p", "quality_clean_fraction"}:
                    vals.append(f"{value:.4f}")
                else:
                    vals.append(f"{value:.4g}")
            else:
                vals.append(str(value))
        md.append("| " + " | ".join(vals) + " |")
    md.extend(
        [
            "",
            "## Plots",
            "",
            f"- Montage: `{OUT_DIR / 'matched_filter_examples_montage.png'}`",
            f"- Index CSV: `{OUT_DIR / 'matched_filter_examples.csv'}`",
        ]
    )
    (OUT_DIR / "matched_filter_examples.md").write_text("\n".join(md) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean = read_table(CLEAN_DEFAULT, parse_dates=["time"], low_memory=False)
    events = _selected_events()
    rows = []
    for i, event in events.iterrows():
        plot_path = OUT_DIR / f"matched_filter_event_{i:04d}_{str(event['source_name']).lower()}.png"
        _plot_event(clean, event, plot_path)
        row = event.to_dict()
        row["frequency"] = f"{float(event['frequency_mhz']):.2f} MHz"
        row["plot"] = str(plot_path)
        rows.append(row)

    summary = pd.DataFrame(rows)
    keep_cols = [
        "source_name",
        "event_type",
        "predicted_event_time",
        "frequency",
        "frequency_band",
        "frequency_mhz",
        "antenna",
        "matched_amp",
        "matched_snr",
        "best_empirical_p",
        "quality_clean_fraction",
        "detection_grade",
        "plot",
    ]
    summary[keep_cols].to_csv(OUT_DIR / "matched_filter_examples.csv", index=False)
    _write_markdown(summary)

    fig, axes = plt.subplots(3, 2, figsize=(13, 15), constrained_layout=True)
    for ax, (_, row) in zip(axes.ravel(), summary.iterrows()):
        ax.imshow(mpimg.imread(row["plot"]))
        ax.axis("off")
    fig.savefig(OUT_DIR / "matched_filter_examples_montage.png", dpi=170)
    plt.close(fig)
    print(summary[["source_name", "event_type", "frequency", "antenna", "matched_snr", "best_empirical_p", "plot"]].to_string(index=False))


if __name__ == "__main__":
    main()
