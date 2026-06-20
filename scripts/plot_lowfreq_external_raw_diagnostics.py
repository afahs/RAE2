#!/usr/bin/env python
"""Raw-data diagnostics for externally selected Sun/Jupiter events.

This intentionally avoids the source-like contrast score in the plots:

- Jupiter panels show lower-V raw power samples around MASER/PADC-selected
  predicted Jupiter events.
- Sun panels show lower-V locally normalized raw-power profile grids for NOAA
  dekameter-selected event times.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.constants import FREQUENCY_MAP_MHZ  # noqa: E402
from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402
from scripts.build_all_frequency_occultation_profile_grids import (  # noqa: E402
    CLEAN,
    _event_window,
    _make_groups,
    collect_profiles,
    summarize_profiles,
)
from scripts.run_low_frequency_external_selectors import (  # noqa: E402
    DEFAULT_FEATURES,
    annotate_jupiter_probability_maps,
    annotate_solar_spectral,
    download_parse_noaa_spectral,
)


ANTENNA = "rv2_coarse"
ANTENNA_LABEL = "lower V"
DEFAULT_OUT = ROOT / "outputs/lower_v_lowfreq_external_raw_diagnostics_v1"
SUN_MODES = ["dek_near_1h", "dek_near_6h", "dek_typeiii_near_6h", "dek_high_day"]
JUPITER_MODE = "maser_zarka_io_top25"
JUPITER_SCORE_COL = "maser_zarka_io_score"


def _read(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return read_table(path, low_memory=False, **kwargs)


def _annotated_features(features_path: Path, out_dir: Path, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = _read(features_path, parse_dates=["predicted_event_time"])
    features = features[features["antenna"].astype(str).eq(ANTENNA)].copy()
    years = pd.to_datetime(features["predicted_event_time"]).dt.year
    spectral = download_parse_noaa_spectral(int(years.min()), int(years.max()), out_dir, refresh=refresh)
    annotated = annotate_solar_spectral(features, spectral)
    annotated, _maps = annotate_jupiter_probability_maps(annotated, out_dir, refresh=refresh)
    return annotated, spectral


def _mode_mask(df: pd.DataFrame, source: str, mode: str) -> pd.Series:
    return (
        df["analysis_source"].astype(str).eq(source)
        & df["control_family"].astype(str).eq("real")
        & df["antenna"].astype(str).eq(ANTENNA)
        & df[mode].astype(bool)
    )


def _plot_sun_mode_grid(
    clean: pd.DataFrame,
    annotated: pd.DataFrame,
    mode: str,
    out_dir: Path,
    window_s: float,
    bin_s: float,
    inner_s: float,
) -> tuple[Path | None, pd.DataFrame]:
    events = annotated[_mode_mask(annotated, "sun", mode)].copy()
    if events.empty:
        return None, pd.DataFrame()

    points = collect_profiles(clean, events, "sun", window_s, bin_s, inner_s)
    summary = summarize_profiles(points)
    if points.empty or summary.empty:
        return None, summary

    points.to_csv(out_dir / f"sun_{mode}_selected_profile_points_{int(window_s)}s.csv", index=False)
    summary.to_csv(out_dir / f"sun_{mode}_selected_profile_summary_{int(window_s)}s.csv", index=False)

    freqs = sorted(summary["frequency_mhz"].dropna().unique())
    event_types = ["disappearance", "reappearance"]
    fig, axes = plt.subplots(len(freqs), 2, figsize=(12.4, max(10, 1.42 * len(freqs))), sharex=True)
    if len(freqs) == 1:
        axes = np.asarray([axes])
    for i, freq in enumerate(freqs):
        for j, event_type in enumerate(event_types):
            ax = axes[i, j]
            sub = summary[
                np.isclose(summary["frequency_mhz"], float(freq))
                & summary["event_type"].astype(str).eq(event_type)
                & summary["antenna"].astype(str).eq(ANTENNA)
            ].sort_values("t_bin_sec")
            if not sub.empty:
                ax.errorbar(
                    sub["t_bin_sec"] / 60.0,
                    sub["median_z_power"],
                    yerr=sub["median_z_power_err"],
                    color="#d95f02",
                    ecolor="#d95f02",
                    marker="o",
                    markersize=2.3,
                    linewidth=1.25,
                    elinewidth=0.65,
                    capsize=1.3,
                )
                n_events = int(sub["n_events"].max())
            else:
                n_events = 0
            ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
            ax.axhline(0.0, color="0.65", linewidth=0.7)
            ax.grid(True, color="0.92", linewidth=0.5)
            ax.set_title(f"{freq:.2f} MHz {event_type}, n={n_events}", fontsize=8.5)
            if j == 0:
                ax.set_ylabel("normalized raw power")
            if i == len(freqs) - 1:
                ax.set_xlabel("minutes from predicted event")
    fig.suptitle(
        f"Sun {ANTENNA_LABEL}: NOAA dekameter-selected events ({mode})\n"
        "Per-event local raw power normalized by sideband median/robust scale; no trendline removal.",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.968])
    path = out_dir / f"sun_{mode}_selected_all_frequency_grid_{int(window_s)}s.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path, summary


def _unique_event_rows(rows: pd.DataFrame) -> pd.DataFrame:
    key = ["event_id", "event_type", "predicted_event_time"]
    work = rows.sort_values(JUPITER_SCORE_COL, ascending=False).drop_duplicates(key, keep="first")
    return work.sort_values(JUPITER_SCORE_COL, ascending=False).reset_index(drop=True)


def _plot_jupiter_raw_single_frequency(
    clean_groups: dict[tuple[int, str], tuple[pd.DataFrame, np.ndarray]],
    selected: pd.DataFrame,
    out_dir: Path,
    window_s: float,
    target_band: int = 1,
    max_events: int = 8,
) -> Path | None:
    events = selected[selected["frequency_band"].astype(int).eq(int(target_band))].copy()
    events = _unique_event_rows(events).head(max_events)
    if events.empty:
        return None
    payload = clean_groups.get((int(target_band), ANTENNA))
    if payload is None:
        return None
    group, group_ns = payload
    n = len(events)
    fig, axes = plt.subplots(n, 1, figsize=(10.8, max(8, 2.2 * n)), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (_, ev) in zip(axes, events.iterrows()):
        local = _event_window(group, group_ns, pd.Timestamp(ev["predicted_event_time"]), window_s)
        if local is None:
            continue
        t, y = local
        ax.plot(t / 60.0, y, marker="o", linestyle="-", markersize=3.0, linewidth=0.8, color="black")
        ax.axvline(0, color="#b2182b", linestyle="--", linewidth=1.0)
        ax.grid(True, color="0.92", linewidth=0.5)
        ax.set_ylabel("raw power")
        ax.set_title(
            f"{pd.Timestamp(ev['predicted_event_time'])}  {ev['event_type']}  "
            f"CML={float(ev['jupiter_cml_spice_deg']):.1f} deg, Io={float(ev['io_phase_spice_deg']):.1f} deg, "
            f"map score={float(ev[JUPITER_SCORE_COL]):.2f}",
            fontsize=8.7,
        )
    axes[-1].set_xlabel("minutes from predicted Jupiter event")
    fig.suptitle(
        f"Jupiter {ANTENNA_LABEL}: raw {FREQUENCY_MAP_MHZ[target_band]:.2f} MHz power for {JUPITER_MODE} selected real events\n"
        "These are measured lower-V samples; invalid samples are omitted; red dashed line is the predicted event time.",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.958])
    path = out_dir / f"jupiter_{JUPITER_MODE}_0p45mhz_raw_event_windows_{int(window_s)}s.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_jupiter_raw_all_frequency_montage(
    clean_groups: dict[tuple[int, str], tuple[pd.DataFrame, np.ndarray]],
    annotated: pd.DataFrame,
    selected_band_rows: pd.DataFrame,
    out_dir: Path,
    window_s: float,
    max_events: int = 6,
) -> Path | None:
    events = _unique_event_rows(selected_band_rows).head(max_events)
    if events.empty:
        return None
    freqs = sorted(FREQUENCY_MAP_MHZ)
    fig, axes = plt.subplots(len(events), len(freqs), figsize=(18.0, max(7.6, 1.55 * len(events))), sharex=True)
    if len(events) == 1:
        axes = np.asarray([axes])
    for row_i, (_, ev) in enumerate(events.iterrows()):
        for col_i, band in enumerate(freqs):
            ax = axes[row_i, col_i]
            payload = clean_groups.get((int(band), ANTENNA))
            if payload is None:
                ax.axis("off")
                continue
            group, group_ns = payload
            local = _event_window(group, group_ns, pd.Timestamp(ev["predicted_event_time"]), window_s)
            if local is not None:
                t, y = local
                ax.plot(t / 60.0, y, marker="o", linestyle="-", markersize=2.1, linewidth=0.65, color="black")
            ax.axvline(0, color="#b2182b", linestyle="--", linewidth=0.75)
            ax.grid(True, color="0.93", linewidth=0.45)
            if row_i == 0:
                ax.set_title(f"{FREQUENCY_MAP_MHZ[band]:.2f} MHz", fontsize=8)
            if col_i == 0:
                ax.set_ylabel(f"{pd.Timestamp(ev['predicted_event_time']).date()}\n{ev['event_type']}\nraw power", fontsize=7)
            if row_i == len(events) - 1:
                ax.set_xlabel("min", fontsize=7)
            ax.tick_params(labelsize=6)
    fig.suptitle(
        f"Jupiter {ANTENNA_LABEL}: raw all-frequency windows for {JUPITER_MODE} selected 0.45 MHz events\n"
        "Each row is one predicted Jupiter event; each panel is actual lower-V raw power with invalid samples omitted.",
        y=0.996,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = out_dir / f"jupiter_{JUPITER_MODE}_raw_all_frequency_event_montage_{int(window_s)}s.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_report(
    out_dir: Path,
    paths: list[Path],
    sun_counts: pd.DataFrame,
    jupiter_selected: pd.DataFrame,
    spectral: pd.DataFrame,
    window_s: float,
) -> Path:
    lines = [
        "# Low-Frequency External Raw Diagnostics",
        "",
        "Purpose: show the external-selector tests closer to the measured data.",
        "",
        "## What Is Plotted",
        "",
        "- Jupiter plots show measured lower-V raw power samples around predicted Jupiter events selected by the MASER/PADC Io-CML probability map.",
        "- Sun grids show measured lower-V power after only per-event local sideband normalization, using NOAA dekameter selected event times.",
        "- Invalid cleaned samples are omitted; no trendline model is subtracted.",
        "- The red dashed vertical line in raw-event plots is the predicted occultation/reappearance time.",
        "",
        f"Window half-width: `{window_s:.0f}` seconds.",
        f"NOAA dekameter spectral events parsed: `{len(spectral)}`.",
        "",
        "## Sun Selected Event Counts",
        "",
        sun_counts.to_string(index=False) if not sun_counts.empty else "(none)",
        "",
        "## Jupiter Selected 0.45 MHz Events",
        "",
        jupiter_selected[
            [
                "event_id",
                "event_type",
                "predicted_event_time",
                "frequency_mhz",
                JUPITER_SCORE_COL,
                "jupiter_cml_spice_deg",
                "io_phase_spice_deg",
                "source_like_log_contrast",
            ]
        ].to_string(index=False)
        if not jupiter_selected.empty
        else "(none)",
        "",
        "## Generated Plots",
        "",
        *[f"- `{p}`" for p in paths],
    ]
    path = out_dir / "lowfreq_external_raw_diagnostics_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--window-s", type=float, default=900.0)
    parser.add_argument("--bin-s", type=float, default=60.0)
    parser.add_argument("--inner-s", type=float, default=15.0)
    parser.add_argument("--refresh-external", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    write_json(
        out_dir / "run_config.json",
        {
            "features": str(args.features),
            "antenna": ANTENNA,
            "sun_modes": SUN_MODES,
            "jupiter_mode": JUPITER_MODE,
            "window_s": float(args.window_s),
            "bin_s": float(args.bin_s),
            "inner_s": float(args.inner_s),
            "software_versions": software_versions(),
        },
    )

    annotated, spectral = _annotated_features(args.features, out_dir, refresh=bool(args.refresh_external))
    clean = _read(CLEAN, parse_dates=["time"])
    clean = clean[clean["antenna"].astype(str).eq(ANTENNA)].copy()
    groups = _make_groups(clean)

    paths: list[Path] = []
    sun_count_rows = []
    for mode in SUN_MODES:
        selected = annotated[_mode_mask(annotated, "sun", mode)].copy()
        sun_count_rows.append(
            {
                "mode": mode,
                "n_frequency_event_rows": int(len(selected)),
                "n_unique_predicted_events": int(
                    selected[["event_id", "event_type", "predicted_event_time"]].drop_duplicates().shape[0]
                )
                if not selected.empty
                else 0,
            }
        )
        path, _summary = _plot_sun_mode_grid(
            clean,
            annotated,
            mode,
            out_dir,
            float(args.window_s),
            float(args.bin_s),
            float(args.inner_s),
        )
        if path is not None:
            paths.append(path)

    jup = annotated[
        _mode_mask(annotated, "jupiter", JUPITER_MODE)
        & annotated["frequency_band"].astype(int).eq(1)
    ].copy()
    jup = _unique_event_rows(jup)
    jup.to_csv(out_dir / f"jupiter_{JUPITER_MODE}_0p45mhz_selected_events.csv", index=False)
    p = _plot_jupiter_raw_single_frequency(groups, jup, out_dir, float(args.window_s), target_band=1)
    if p is not None:
        paths.append(p)
    p = _plot_jupiter_raw_all_frequency_montage(groups, annotated, jup, out_dir, float(args.window_s))
    if p is not None:
        paths.append(p)

    sun_counts = pd.DataFrame(sun_count_rows)
    sun_counts.to_csv(out_dir / "sun_external_selected_event_counts.csv", index=False)
    report = _write_report(out_dir, paths, sun_counts, jup, spectral, float(args.window_s))
    print(report)


if __name__ == "__main__":
    main()
