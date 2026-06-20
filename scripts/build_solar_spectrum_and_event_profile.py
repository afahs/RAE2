#!/usr/bin/env python
"""Build solar lower-V spectrum and best-channel event-type profile for slides."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rylevonberg.detection import baseline_matrix
from rylevonberg.util import datetime_ns, robust_sigma


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/slide_assets/channel_strength_spectra"
SOLAR_RUN = ROOT / "outputs/sun_whole_dataset_validation_allbands_mincontrols/summary/sun_real_stack_summary.csv"
CONTROLLED_RUN = ROOT / "outputs/sun_whole_dataset_validation_highcontrols_bands4_8/summary/sun_whole_dataset_scored_stacks.csv"
CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
EVENTS = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/02_events/predicted_events.csv"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    solar = read_table(SOLAR_RUN)
    controlled = read_table(CONTROLLED_RUN)
    lower = solar[solar["antenna"].astype(str).eq("rv2_coarse")].sort_values("frequency_mhz").copy()
    lower.to_csv(OUT / "sun_lower_v_snr_spectrum.csv", index=False)
    _plot_spectrum(lower)

    controlled_lower = controlled[controlled["antenna"].astype(str).eq("rv2_coarse")].copy()
    best = controlled_lower.assign(abs_snr=controlled_lower["stacked_snr"].abs()).sort_values("abs_snr", ascending=False).iloc[0]
    profile = _solar_event_type_profile(
        frequency_band=int(best["frequency_band"]),
        frequency_mhz=float(best["frequency_mhz"]),
        antenna=str(best["antenna"]),
        window_s=900.0,
        bin_s=60.0,
    )
    profile.to_csv(OUT / "sun_3p93mhz_lower_v_event_type_profile.csv", index=False)
    _plot_event_type_profile(profile, best)
    _write_readme(lower, best)


def _plot_spectrum(lower: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(lower["frequency_mhz"], lower["stacked_snr"], marker="o", lw=2.0, color="#1b9e77")
    status_colors = {"candidate": "#1b9e77", "unresolved": "#d95f02", "not_detected": "0.35", "likely_systematic": "#7570b3"}
    for _, row in lower.iterrows():
        ax.scatter(
            row["frequency_mhz"],
            row["stacked_snr"],
            s=70,
            color=status_colors.get(str(row.get("status")), "#1f77b4"),
            edgecolor="black",
            linewidth=0.5,
            zorder=4,
        )
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
    ax.set_title("Sun lower-V full-band channel-strength spectrum")
    fig.tight_layout()
    fig.savefig(OUT / "sun_lower_v_snr_spectrum.png", dpi=180)
    plt.close(fig)


def _solar_event_type_profile(frequency_band: int, frequency_mhz: float, antenna: str, window_s: float, bin_s: float) -> pd.DataFrame:
    clean = read_table(CLEAN, parse_dates=["time"], low_memory=False)
    events = read_table(EVENTS, parse_dates=["predicted_event_time"], low_memory=False)
    events = events[
        events["source_name"].astype(str).eq("sun")
        & events["frequency_band"].astype(int).eq(frequency_band)
        & events["antenna"].astype(str).eq(antenna)
    ].copy()
    group = clean[
        clean["frequency_band"].astype(int).eq(frequency_band)
        & clean["antenna"].astype(str).eq(antenna)
    ].sort_values("time").reset_index(drop=True)
    t_ns = datetime_ns(group["time"])
    rows = []
    half_ns = int(window_s * 1e9)
    for _, ev in events.iterrows():
        event_ns = pd.Timestamp(ev["predicted_event_time"]).value
        lo = int(np.searchsorted(t_ns, event_ns - half_ns, side="left"))
        hi = int(np.searchsorted(t_ns, event_ns + half_ns, side="right"))
        if hi <= lo:
            continue
        local = group.iloc[lo:hi]
        rel = (datetime_ns(local["time"]) - event_ns).astype(float) / 1e9
        y = pd.to_numeric(local["power"], errors="coerce").to_numpy(dtype=float)
        valid = local["is_valid"].to_numpy(dtype=bool) if "is_valid" in local.columns else np.ones(len(local), dtype=bool)
        keep = valid & np.isfinite(y) & (np.abs(rel) <= window_s)
        if np.count_nonzero(keep) < 6:
            continue
        tr = rel[keep]
        yy = y[keep]
        baseline = baseline_matrix(tr, 1)
        beta, *_ = np.linalg.lstsq(baseline, yy, rcond=None)
        yd = yy - baseline @ beta
        sigma = robust_sigma(yd)
        if np.isfinite(sigma) and sigma > 0:
            yd = yd / sigma
        tbin = np.round(tr / bin_s) * bin_s
        for t_rel, tb, val in zip(tr, tbin, yd):
            rows.append(
                {
                    "source_name": "sun",
                    "frequency_band": frequency_band,
                    "frequency_mhz": frequency_mhz,
                    "antenna": antenna,
                    "event_type": ev["event_type"],
                    "event_id": ev.get("event_id"),
                    "t_rel_sec": float(t_rel),
                    "t_bin_sec": float(tb),
                    "profile_value": float(val),
                }
            )
    prof = pd.DataFrame.from_records(rows)
    if prof.empty:
        return prof
    out_rows = []
    for keys, grp in prof.groupby(["source_name", "frequency_band", "frequency_mhz", "antenna", "event_type", "t_bin_sec"], sort=True):
        source, band, mhz, ant, event_type, tbin = keys
        vals = grp["profile_value"].to_numpy(dtype=float)
        out_rows.append(
            {
                "source_name": source,
                "frequency_band": band,
                "frequency_mhz": mhz,
                "antenna": ant,
                "event_type": event_type,
                "t_bin_sec": tbin,
                "n_samples": int(vals.size),
                "n_events": int(grp["event_id"].nunique()),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "sem": float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else np.nan,
            }
        )
    return pd.DataFrame.from_records(out_rows)


def _plot_event_type_profile(profile: pd.DataFrame, best: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors = {"disappearance": "#4c78a8", "reappearance": "#d95f02"}
    for event_type, grp in profile.groupby("event_type", sort=True):
        grp = grp.sort_values("t_bin_sec")
        label = f"{event_type} ({int(grp['n_events'].max())} events)"
        ax.plot(grp["t_bin_sec"] / 60.0, grp["mean"], marker="o", ms=3, lw=1.8, color=colors.get(str(event_type)), label=label)
        sem = grp["sem"].to_numpy(dtype=float)
        ax.fill_between(grp["t_bin_sec"] / 60.0, grp["mean"] - sem, grp["mean"] + sem, color=colors.get(str(event_type)), alpha=0.18, linewidth=0)
    ax.axvline(0.0, color="black", lw=1.0, label="predicted occultation time")
    ax.axhline(0.0, color="0.35", lw=0.8)
    ax.set_title("Sun 3.93 MHz lower V stacked occultation profile")
    ax.set_xlabel("Relative time from predicted event (min)")
    ax.set_ylabel("Mean normalized, baseline-subtracted power")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "sun_3p93mhz_lower_v_event_type_profile.png", dpi=180)
    plt.close(fig)


def _write_readme(lower: pd.DataFrame, best: pd.Series) -> None:
    md = [
        "# Solar Lower-V Spectrum And Event-Type Profile",
        "",
        "The spectrum uses the full-band solar scan, filtered to lower V (`rv2_coarse`) only. This shows channel strength across all available Ryle-Vonberg bands. The plotted SNR values are stacked SNR values only; randomized-time and off-ephemeris controls are separate tests and are not folded into the SNR.",
        "",
        f"- Spectrum: `{OUT / 'sun_lower_v_snr_spectrum.png'}`",
        f"- Best-channel event-type profile: `{OUT / 'sun_3p93mhz_lower_v_event_type_profile.png'}`",
        f"- Spectrum CSV: `{OUT / 'sun_lower_v_snr_spectrum.csv'}`",
        "",
        f"The event-type profile still uses the best controlled solar channel: {best['frequency_mhz']:.2f} MHz lower V.",
    ]
    (OUT / "sun_spectrum_readme.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
