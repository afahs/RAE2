"""Diagnostic plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .util import datetime_ns


def plot_event_window(clean_df: pd.DataFrame, event: pd.Series, output_path: str | Path, window_seconds: float = 600.0) -> None:
    group = clean_df
    if pd.notna(event.get("frequency_band")):
        group = group[group["frequency_band"] == event["frequency_band"]]
    if pd.notna(event.get("antenna")):
        group = group[group["antenna"] == event["antenna"]]
    et = pd.Timestamp(event["predicted_event_time"])
    rel = (datetime_ns(group["time"]) - et.value) / 1e9
    keep = abs(rel) <= float(window_seconds)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(rel[keep] / 60.0, group.loc[keep, "power"], ".", ms=2)
    ax.axvline(0.0, color="black", lw=1)
    ax.set_xlabel("Relative time (min)")
    ax.set_ylabel("Ryle-Vonberg power")
    ax.set_title(f"{event.get('source_name', '')} {event.get('event_type', '')}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_stack(stacked_df: pd.DataFrame, output_path: str | Path) -> None:
    if stacked_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for _, grp in stacked_df.groupby([c for c in ["source_name", "frequency_band", "antenna"] if c in stacked_df.columns], dropna=False):
        ax.plot(grp["t_bin_sec"] / 60.0, grp["mean"], lw=1)
    ax.axvline(0.0, color="black", lw=1)
    ax.set_xlabel("Relative time (min)")
    ax.set_ylabel("Mean normalized profile")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
