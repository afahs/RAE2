#!/usr/bin/env python
"""Back-of-envelope antenna-beam modulation check for Fornax A."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

ROOT = Path(__file__).resolve().parents[1]
BEAM_DIR = Path(os.environ.get("RAE2_ANTENNA_DIGITIZATION_DIR", "data/antennaDigitization"))
EVENTS = ROOT / "outputs/control_survey_bright_sources_postnov1974_v1/02_events/predicted_events.csv"
OUT_DIR = ROOT / "outputs/antenna_beam_bote_fornax_v1"


def _periodic_interp(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    xp_ext = np.r_[xp - 360.0, xp, xp + 360.0]
    fp_ext = np.r_[fp, fp, fp]
    return np.interp(x, xp_ext, fp_ext)


def _load_beam(freq_label: str, plane: str) -> pd.DataFrame:
    return read_table(BEAM_DIR / f"{plane}_{freq_label}MHz.csv")


def _limb_rate_deg_s() -> float:
    events = read_table(EVENTS, low_memory=False)
    fa = events[events["source_name"].astype(str).str.lower().eq("fornax_a")].copy()
    pre = pd.to_numeric(fa["pre_limb_angle_deg"], errors="coerce")
    post = pd.to_numeric(fa["post_limb_angle_deg"], errors="coerce")
    gap = pd.to_numeric(fa["gap_seconds"], errors="coerce")
    rate = (post - pre).abs() / gap
    return float(np.nanmedian(rate))


def compute_slope_table(rate_deg_s: float) -> pd.DataFrame:
    rows = []
    for freq_label, freq_mhz in [("1p31", 1.31), ("3p93", 3.93), ("6p55", 6.55)]:
        for plane in ["eplane", "hplane"]:
            beam = _load_beam(freq_label, plane)
            angle = beam["angle_deg"].to_numpy(dtype=float)
            gain_db = beam["gain_dB"].to_numpy(dtype=float)
            gain_lin = 10.0 ** (gain_db / 10.0)
            for center in [60.0, 75.0, 90.0, 105.0, 120.0]:
                for half_width in [5.0, 10.0, 15.0, 30.0]:
                    sample = np.linspace(center - half_width, center + half_width, 401)
                    gdb = _periodic_interp(sample, angle, gain_db)
                    glin = _periodic_interp(sample, angle, gain_lin)
                    slope_db_deg = float(np.polyfit(sample, gdb, 1)[0])
                    slope_lin_deg = float(np.polyfit(sample, glin, 1)[0])
                    g_center = float(_periodic_interp(np.array([center]), angle, gain_lin)[0])
                    rel_slope_deg = slope_lin_deg / g_center if g_center > 0 else np.nan
                    # Time for the local linear trend to change by one local gain value.
                    tau_gain_s = 1.0 / abs(rel_slope_deg * rate_deg_s) if np.isfinite(rel_slope_deg) and rel_slope_deg != 0 else np.nan
                    db_change_20min = slope_db_deg * rate_deg_s * 1200.0
                    rows.append(
                        {
                            "frequency_mhz": freq_mhz,
                            "plane": plane,
                            "center_angle_deg": center,
                            "fit_half_width_deg": half_width,
                            "gain_center_linear": g_center,
                            "gain_center_dB": 10.0 * np.log10(g_center) if g_center > 0 else np.nan,
                            "slope_dB_per_deg": slope_db_deg,
                            "slope_linear_per_deg": slope_lin_deg,
                            "relative_slope_per_deg": rel_slope_deg,
                            "tau_gain_s": tau_gain_s,
                            "dB_change_over_20min": db_change_20min,
                        }
                    )
    return pd.DataFrame.from_records(rows)


def plot_beam_cut(rate_deg_s: float, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, plane in zip(axes, ["eplane", "hplane"]):
        beam = _load_beam("1p31", plane)
        angle = beam["angle_deg"].to_numpy(dtype=float)
        gain_db = beam["gain_dB"].to_numpy(dtype=float)
        ax.plot(angle, gain_db, lw=1.2)
        for center in [90.0, 105.0, 120.0]:
            ax.axvline(center, color="0.65", lw=0.8, ls="--")
        ax.set_xlim(30, 150)
        ax.set_ylim(-21, 1)
        ax.grid(alpha=0.25)
        ax.set_title(f"1.31 MHz {plane}")
        ax.set_xlabel("beam-pattern angle (deg)")
        ax.text(
            0.03,
            0.05,
            f"Fornax limb rate ~{rate_deg_s:.4f} deg/s\n20 min = {rate_deg_s*1200:.1f} deg",
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
        )
    axes[0].set_ylabel("digitized gain (dB)")
    fig.suptitle("Digitized beam structure near lunar-limb angles")
    fig.tight_layout()
    path = out_dir / "fornax_1p31_beam_cut_near_limb.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_point_source_toy(rate_deg_s: float, out_dir: Path) -> Path:
    beam = _load_beam("1p31", "eplane")
    angle = beam["angle_deg"].to_numpy(dtype=float)
    gain_lin = 10.0 ** (beam["gain_dB"].to_numpy(dtype=float) / 10.0)
    t = np.linspace(-1800.0, 1800.0, 241)
    theta0 = 90.0
    theta = theta0 + rate_deg_s * t
    gain = _periodic_interp(theta, angle, gain_lin)
    gain = gain / np.nanmax(gain)
    dis = gain * (t < 0)
    rep = gain * (t > 0)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(t / 60.0, gain, color="0.55", lw=1.5, label="beam gain only")
    ax.plot(t / 60.0, dis, color="#4c78a8", lw=2, label="point-source disappearance toy")
    ax.plot(t / 60.0, rep, color="#d95f02", lw=2, label="point-source reappearance toy")
    ax.axvline(0, color="black", ls=":", lw=0.9)
    ax.set_xlabel("minutes from nominal event")
    ax.set_ylabel("relative point-source contribution")
    ax.set_title("Toy model: point source occulted while beam gain changes")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "fornax_point_source_beam_modulation_toy.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _markdown_table(frame: pd.DataFrame) -> str:
    work = frame.copy()
    for col in work.columns:
        if pd.api.types.is_numeric_dtype(work[col]):
            work[col] = work[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    cols = list(work.columns)
    widths = {col: max(len(str(col)), *(len(str(v)) for v in work[col])) for col in cols}
    lines = [
        "| " + " | ".join(str(col).ljust(widths[col]) for col in cols) + " |",
        "| " + " | ".join("-" * widths[col] for col in cols) + " |",
    ]
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(str(row[col]).ljust(widths[col]) for col in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rate = _limb_rate_deg_s()
    table = compute_slope_table(rate)
    table.to_csv(OUT_DIR / "fornax_beam_slope_timescale_table.csv", index=False)
    p1 = plot_beam_cut(rate, OUT_DIR)
    p2 = plot_point_source_toy(rate, OUT_DIR)

    focus = table[
        table["frequency_mhz"].eq(1.31)
        & table["center_angle_deg"].isin([90.0, 105.0, 120.0])
        & table["fit_half_width_deg"].isin([10.0, 15.0, 30.0])
    ].copy()
    focus = focus.sort_values(["plane", "center_angle_deg", "fit_half_width_deg"])
    cols = [
        "frequency_mhz",
        "plane",
        "center_angle_deg",
        "fit_half_width_deg",
        "gain_center_dB",
        "slope_dB_per_deg",
        "tau_gain_s",
        "dB_change_over_20min",
    ]
    lines = [
        "# Fornax A Antenna-Beam Back-Of-Envelope",
        "",
        "Question: can the digitized antenna pattern turn a point-source occultation into an apparently linear falloff/rise?",
        "",
        "Point-source toy model:",
        "",
        "    P(t) = S * G(theta(t)) * V(t)",
        "",
        "where `V(t)` is the lunar occultation visibility step and `G(theta)` is the antenna gain. For a point source, the lunar visibility is still a step, but the unocculted side can be strongly sloped if the source is moving through a steep beam sidelobe/null.",
        "",
        f"Median Fornax event-table limb-angle rate: `{rate:.5f} deg/s`.",
        f"That is `{rate * 1200.0:.1f} deg` over 20 minutes.",
        "",
        "The digitized 1.31 MHz beam has strong structure near 90-120 deg. That is the relevant order-of-magnitude region for a lunar-limb source when the lower V boresight points toward the Moon.",
        "",
        "## Local Beam Slopes",
        "",
        _markdown_table(focus[cols]),
        "",
        "## Interpretation",
        "",
        "- Around 1.31 MHz, the E-plane gain can change by many dB across the angle swept in a 20 minute window.",
        "- Several local gain timescales are hundreds to a few thousand seconds, comparable to the fitted finite-duration `tau` values for Fornax A.",
        "- Therefore antenna beam modulation can plausibly make the unocculted side of a point-source event look roughly linear over minutes.",
        "- It does not remove the underlying point-source visibility step. A true point source should still switch on/off at the limb, but sparse sampling plus beam modulation can make the stacked profile look ramp-like instead of vertically stepped.",
        "- Because this uses 1D E/H plane cuts and a rough limb-angle rate, it is only a back-of-the-envelope check. A proper test requires projecting the source direction through the full spacecraft/antenna frame and interpolating the 2D beam.",
        "",
        "## Plots",
        "",
        f"- `{p1.name}`",
        f"- `{p2.name}`",
    ]
    (OUT_DIR / "fornax_beam_bote_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_DIR / "fornax_beam_bote_report.md")


if __name__ == "__main__":
    main()
