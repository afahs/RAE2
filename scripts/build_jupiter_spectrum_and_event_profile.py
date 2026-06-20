#!/usr/bin/env python
"""Build Jupiter lower-V spectrum and best-channel event-type profile for slides."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from rylevonberg.table_io import read_table


ROOT = Path(__file__).resolve().parents[1]
SURVEY = ROOT / "outputs/planetary_confirmation_survey/jupiter"
OUT = ROOT / "outputs/slide_assets/channel_strength_spectra"
CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
EVENTS = ROOT / "outputs/planetary_confirmation_survey/events/jupiter_predicted_events.csv"

sys.path.insert(0, str(ROOT))

from rylevonberg.stacking import aligned_profiles


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scan = read_table(SURVEY / "initial_channel_scan.csv")
    lower = scan[
        scan["antenna"].astype(str).eq("rv2_coarse")
        & pd.to_numeric(scan["window_s"], errors="coerce").eq(900.0)
    ].sort_values("frequency_mhz").copy()
    lower.to_csv(OUT / "jupiter_lower_v_snr_spectrum.csv", index=False)
    _plot_spectrum(lower)

    event_type = read_table(SURVEY / "event_type.csv")
    event_type.to_csv(OUT / "jupiter_1p31mhz_lower_v_event_type_summary.csv", index=False)
    profile = _jupiter_event_type_profile(frequency_band=4, antenna="rv2_coarse", window_s=900.0, bin_s=60.0)
    profile.to_csv(OUT / "jupiter_1p31mhz_lower_v_event_type_profile.csv", index=False)
    _plot_event_type_profile(profile)
    _write_readme(lower, event_type)


def _plot_spectrum(lower: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(lower["frequency_mhz"], lower["stacked_snr"], marker="o", lw=2.0, color="#7570b3")
    ax.axhline(0, color="0.25", lw=0.9)
    ax.axhline(3, color="0.65", lw=0.8, ls="--")
    ax.axhline(-3, color="0.65", lw=0.8, ls="--")
    for _, row in lower.iterrows():
        ax.text(row["frequency_mhz"], row["stacked_snr"], f" {row['frequency_mhz']:.2g}", fontsize=7, va="bottom" if row["stacked_snr"] >= 0 else "top")
    ax.set_xscale("log")
    ticks = sorted(lower["frequency_mhz"].unique())
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{x:.2g}" for x in ticks])
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Stacked SNR, lower V")
    ax.set_title("Jupiter lower-V full-band channel-strength spectrum")
    fig.tight_layout()
    fig.savefig(OUT / "jupiter_lower_v_snr_spectrum.png", dpi=180)
    plt.close(fig)


def _jupiter_event_type_profile(frequency_band: int, antenna: str, window_s: float, bin_s: float) -> pd.DataFrame:
    clean = read_table(CLEAN, parse_dates=["time"], low_memory=False)
    events = read_table(EVENTS, parse_dates=["predicted_event_time"], low_memory=False)
    sub = events[
        events["source_name"].astype(str).eq("jupiter")
        & events["frequency_band"].astype(int).eq(int(frequency_band))
        & events["antenna"].astype(str).eq(str(antenna))
    ].copy()
    profiles = aligned_profiles(clean, sub, window_seconds=window_s, bin_seconds=bin_s, normalize=True)
    if profiles.empty:
        return pd.DataFrame()
    profiles["template_aligned_value"] = profiles["profile_value"].astype(float) * profiles["template"].astype(float)
    rows = []
    for keys, grp in profiles.groupby(["source_name", "frequency_band", "frequency_mhz", "antenna", "event_type", "t_bin_sec"], sort=True):
        source, band, mhz, ant, event_type, t_bin = keys
        rows.append(
            {
                "source_name": source,
                "frequency_band": int(band),
                "frequency_mhz": float(mhz),
                "antenna": ant,
                "event_type": event_type,
                "t_bin_sec": float(t_bin),
                "n_samples": int(len(grp)),
                "n_events": int(grp["event_id"].nunique()),
                "mean_template_aligned": float(grp["template_aligned_value"].mean()),
                "median_template_aligned": float(grp["template_aligned_value"].median()),
                "sem_template_aligned": float(grp["template_aligned_value"].std(ddof=1) / (len(grp) ** 0.5)) if len(grp) > 1 else float("nan"),
            }
        )
    return pd.DataFrame.from_records(rows)


def _plot_event_type_profile(profile: pd.DataFrame) -> None:
    if profile.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors = {"disappearance": "#4c78a8", "reappearance": "#d95f02"}
    for event_type, grp in profile.groupby("event_type", sort=True):
        grp = grp.sort_values("t_bin_sec")
        x = grp["t_bin_sec"] / 60.0
        y = grp["mean_template_aligned"]
        sem = grp["sem_template_aligned"]
        label = f"{event_type} ({int(grp['n_events'].max())} events)"
        ax.plot(x, y, marker="o", ms=3, lw=1.8, color=colors.get(str(event_type)), label=label)
        ax.fill_between(x, y - sem, y + sem, color=colors.get(str(event_type)), alpha=0.18, linewidth=0)
    ax.axvline(0.0, color="black", lw=1.0, ls="--", label="predicted occultation time")
    ax.axvline(-0.5, color="0.35", lw=0.9, ls=":", label="best timing scan offset")
    ax.axhline(0.0, color="0.35", lw=0.8)
    ax.set_title("Jupiter 1.31 MHz lower V stacked occultation profile")
    ax.set_xlabel("Relative time from predicted event (min)")
    ax.set_ylabel("Template-aligned normalized residual")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "jupiter_1p31mhz_lower_v_event_type_profile.png", dpi=180)
    plt.close(fig)


def _write_readme(lower: pd.DataFrame, event_type: pd.DataFrame) -> None:
    best = lower.assign(abs_snr=lower["stacked_snr"].abs()).sort_values("abs_snr", ascending=False).iloc[0]
    md = [
        "# Jupiter Lower-V Spectrum And Event-Type Profile",
        "",
        "The spectrum uses the planetary survey initial channel scan, filtered to lower V (`rv2_coarse`) and the 900 s Jupiter target window. The event-type plot uses the best lower-V channel after the timing-offset scan: 1.31 MHz lower V at dt = -30 s.",
        "",
        f"- Spectrum: `{OUT / 'jupiter_lower_v_snr_spectrum.png'}`",
        f"- Event-type profile: `{OUT / 'jupiter_1p31mhz_lower_v_event_type_profile.png'}`",
        f"- Spectrum CSV: `{OUT / 'jupiter_lower_v_snr_spectrum.csv'}`",
        f"- Event-type profile CSV: `{OUT / 'jupiter_1p31mhz_lower_v_event_type_profile.csv'}`",
        f"- Event-type summary CSV: `{OUT / 'jupiter_1p31mhz_lower_v_event_type_summary.csv'}`",
        "",
        f"Strongest lower-V channel at dt=0: {best['frequency_mhz']:.2f} MHz, SNR {best['stacked_snr']:.2f}.",
        "Timing-scan best for this channel: 1.31 MHz, SNR 4.24 at dt = -30 s.",
    ]
    (OUT / "jupiter_spectrum_readme.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
