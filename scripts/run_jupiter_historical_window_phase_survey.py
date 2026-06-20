#!/usr/bin/env python
"""Jupiter survey restricted to historically active Warwick/Dulk/Riddle windows."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES = ROOT / "outputs/jupiter_phase_pattern_survey_v1/jupiter_phase_pattern_sampled_points.csv"
DEFAULT_WINDOWS = ROOT / "configs/jupiter_warwick_dulk_riddle_1975_active_windows.csv"
DEFAULT_OUT = ROOT / "outputs/jupiter_historical_active_windows_v1"

ANTENNA_LABEL = {
    "rv1_coarse": "upper V",
    "rv2_coarse": "lower V",
}


@dataclass(frozen=True)
class Interval:
    label: str
    start: pd.Timestamp
    end: pd.Timestamp


def parse_hhmm(date: pd.Timestamp, value: object) -> pd.Timestamp:
    text = str(value).strip().replace(".", "")
    if not text or text.lower() == "nan":
        raise ValueError(f"missing HHMM value for {date}")
    sign = -1 if text.startswith("-") else 1
    text = text.lstrip("+-").zfill(4)
    hour = int(text[:-2])
    minute = int(text[-2:])
    if minute >= 60:
        raise ValueError(f"malformed HHMM minute in {value!r}")
    if sign < 0:
        return date - pd.Timedelta(hours=hour, minutes=minute)
    day_add, hour = divmod(hour, 24)
    return date + pd.Timedelta(days=day_add, hours=hour, minutes=minute)


def load_windows(path: Path, padding_min: float) -> pd.DataFrame:
    raw = read_table(path, dtype=str)
    rows = []
    for i, row in raw.iterrows():
        date = pd.Timestamp(row["date"])
        start = parse_hhmm(date, row["event_start_hhmm"])
        end = parse_hhmm(date, row["event_end_hhmm"])
        if end <= start:
            end += pd.Timedelta(days=1)
        pad = pd.Timedelta(minutes=float(padding_min))
        freq_text = str(row.get("reported_freq_range_mhz", "")).strip()
        freq_low = np.nan
        freq_high = np.nan
        if "-" in freq_text:
            left, right = freq_text.split("-", 1)
            try:
                freq_low = float(left)
                freq_high = float(right)
            except ValueError:
                pass
        rows.append(
            {
                "historical_window_id": f"w{i + 1:03d}",
                "historical_date": date.date().isoformat(),
                "event_start_time": start,
                "event_end_time": end,
                "expanded_start_time": start - pad,
                "expanded_end_time": end + pad,
                "duration_min": (end - start).total_seconds() / 60.0,
                "expanded_duration_min": (end - start + 2 * pad).total_seconds() / 60.0,
                "intensity": int(row["intensity"]),
                "burstiness": str(row["burstiness"]).strip().upper(),
                "reported_freq_range_mhz": freq_text,
                "reported_freq_low_mhz": freq_low,
                "reported_freq_high_mhz": freq_high,
                "pdf_page": row.get("pdf_page", ""),
                "notes": row.get("notes", ""),
            }
        )
    return pd.DataFrame(rows)


def _intervals_from_windows(windows: pd.DataFrame, start_col: str, end_col: str) -> list[Interval]:
    return [
        Interval(str(row["historical_window_id"]), pd.Timestamp(row[start_col]), pd.Timestamp(row[end_col]))
        for _, row in windows.iterrows()
    ]


def assign_intervals(times: pd.Series, intervals: list[Interval]) -> pd.Series:
    labels = np.full(len(times), "", dtype=object)
    arr = pd.to_datetime(times).to_numpy()
    for iv in intervals:
        mask = (arr >= np.datetime64(iv.start)) & (arr <= np.datetime64(iv.end))
        labels[mask & (labels == "")] = iv.label
    return pd.Series(labels, index=times.index)


def build_shifted_controls(windows: pd.DataFrame, shift_days: list[int], padding_min: float) -> pd.DataFrame:
    pieces = []
    for shift in shift_days:
        shifted = windows.copy()
        shifted["historical_window_id"] = shifted["historical_window_id"].map(lambda x: f"{x}_shift_{shift:+d}d")
        for col in ["event_start_time", "event_end_time", "expanded_start_time", "expanded_end_time"]:
            shifted[col] = pd.to_datetime(shifted[col]) + pd.Timedelta(days=int(shift))
        shifted["control_shift_days"] = int(shift)
        pieces.append(shifted)
    out = pd.concat(pieces, ignore_index=True)
    pad = pd.Timedelta(minutes=float(padding_min))
    out["expanded_start_time"] = pd.to_datetime(out["event_start_time"]) - pad
    out["expanded_end_time"] = pd.to_datetime(out["event_end_time"]) + pad
    return out


def summarize_labeled_samples(samples: pd.DataFrame, label_col: str, high_z: float) -> pd.DataFrame:
    rows = []
    work = samples.copy()
    work["high_power_tail"] = work["daily_z_log_power"] > float(high_z)
    for (antenna, band, freq), grp in work.groupby(["antenna", "frequency_band", "frequency_mhz"], sort=True):
        active = grp[grp[label_col].astype(str).ne("")]
        inactive = grp[grp[label_col].astype(str).eq("")]
        rows.append(
            {
                "antenna": antenna,
                "frequency_band": int(band),
                "frequency_mhz": float(freq),
                "active_n_samples": int(len(active)),
                "inactive_n_samples": int(len(inactive)),
                "active_n_windows": int(active[label_col].nunique()) if len(active) else 0,
                "active_high_tail_fraction": float(active["high_power_tail"].mean()) if len(active) else np.nan,
                "inactive_high_tail_fraction": float(inactive["high_power_tail"].mean()) if len(inactive) else np.nan,
                "active_minus_inactive_high_tail_fraction": (
                    float(active["high_power_tail"].mean() - inactive["high_power_tail"].mean())
                    if len(active) and len(inactive)
                    else np.nan
                ),
                "active_median_daily_z": float(active["daily_z_log_power"].median()) if len(active) else np.nan,
                "inactive_median_daily_z": float(inactive["daily_z_log_power"].median()) if len(inactive) else np.nan,
                "active_minus_inactive_median_daily_z": (
                    float(active["daily_z_log_power"].median() - inactive["daily_z_log_power"].median())
                    if len(active) and len(inactive)
                    else np.nan
                ),
                "active_maser_score_median": float(active["maser_zarka_io_score"].median()) if len(active) else np.nan,
                "inactive_maser_score_median": float(inactive["maser_zarka_io_score"].median()) if len(inactive) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_active_vs_shifted(active_samples: pd.DataFrame, control_samples: pd.DataFrame, high_z: float) -> pd.DataFrame:
    rows = []
    active = active_samples.copy()
    control = control_samples.copy()
    active["high_power_tail"] = active["daily_z_log_power"] > float(high_z)
    control["high_power_tail"] = control["daily_z_log_power"] > float(high_z)
    for (antenna, band, freq), grp in active.groupby(["antenna", "frequency_band", "frequency_mhz"], sort=True):
        ctl = control[
            control["antenna"].astype(str).eq(str(antenna))
            & control["frequency_band"].astype(int).eq(int(band))
            & np.isclose(control["frequency_mhz"].astype(float), float(freq))
        ]
        rows.append(
            {
                "antenna": antenna,
                "frequency_band": int(band),
                "frequency_mhz": float(freq),
                "active_n_samples": int(len(grp)),
                "shifted_control_n_samples": int(len(ctl)),
                "active_high_tail_fraction": float(grp["high_power_tail"].mean()) if len(grp) else np.nan,
                "shifted_control_high_tail_fraction": float(ctl["high_power_tail"].mean()) if len(ctl) else np.nan,
                "active_minus_shifted_high_tail_fraction": (
                    float(grp["high_power_tail"].mean() - ctl["high_power_tail"].mean())
                    if len(grp) and len(ctl)
                    else np.nan
                ),
                "active_median_daily_z": float(grp["daily_z_log_power"].median()) if len(grp) else np.nan,
                "shifted_control_median_daily_z": float(ctl["daily_z_log_power"].median()) if len(ctl) else np.nan,
                "active_minus_shifted_median_daily_z": (
                    float(grp["daily_z_log_power"].median() - ctl["daily_z_log_power"].median())
                    if len(grp) and len(ctl)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_windows(active: pd.DataFrame, high_z: float) -> pd.DataFrame:
    work = active.copy()
    work["high_power_tail"] = work["daily_z_log_power"] > float(high_z)
    rows = []
    for (window_id, antenna, band, freq), grp in work.groupby(
        ["historical_window_id", "antenna", "frequency_band", "frequency_mhz"], sort=True
    ):
        rows.append(
            {
                "historical_window_id": window_id,
                "antenna": antenna,
                "frequency_band": int(band),
                "frequency_mhz": float(freq),
                "n_samples": int(len(grp)),
                "high_power_fraction": float(grp["high_power_tail"].mean()),
                "median_daily_z": float(grp["daily_z_log_power"].median()),
                "max_daily_z": float(grp["daily_z_log_power"].max()),
                "median_maser_score": float(grp["maser_zarka_io_score"].median()),
            }
        )
    return pd.DataFrame(rows)


def phase_binned(active: pd.DataFrame, high_z: float, phase_bin_deg: float, min_count: int) -> pd.DataFrame:
    work = active[active["jupiter_visible_by_moon"].astype(bool)].copy()
    if work.empty:
        return pd.DataFrame()
    work["high_power_tail"] = work["daily_z_log_power"] > float(high_z)
    work["cml_bin_deg"] = np.floor(work["jupiter_cml_spice_deg"] / phase_bin_deg) * phase_bin_deg + 0.5 * phase_bin_deg
    work["io_bin_deg"] = np.floor(work["io_phase_spice_deg"] / phase_bin_deg) * phase_bin_deg + 0.5 * phase_bin_deg
    rows = []
    for keys, grp in work.groupby(["antenna", "frequency_band", "frequency_mhz", "cml_bin_deg", "io_bin_deg"], sort=True):
        if len(grp) < int(min_count):
            continue
        rows.append(
            {
                "antenna": keys[0],
                "frequency_band": int(keys[1]),
                "frequency_mhz": float(keys[2]),
                "cml_bin_deg": float(keys[3]),
                "io_bin_deg": float(keys[4]),
                "n_samples": int(len(grp)),
                "high_power_fraction": float(grp["high_power_tail"].mean()),
                "median_daily_z": float(grp["daily_z_log_power"].median()),
                "median_maser_score": float(grp["maser_zarka_io_score"].median()),
            }
        )
    return pd.DataFrame(rows)


def plot_active_spectrum(active_shifted: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=True)
    for antenna, sub in active_shifted.groupby("antenna", sort=True):
        label = ANTENNA_LABEL.get(str(antenna), str(antenna))
        axes[0].plot(
            sub["frequency_mhz"],
            sub["active_minus_shifted_high_tail_fraction"],
            marker="o",
            lw=1.5,
            label=label,
        )
        axes[1].plot(
            sub["frequency_mhz"],
            sub["active_minus_shifted_median_daily_z"],
            marker="o",
            lw=1.5,
            label=label,
        )
    axes[0].axhline(0, color="0.35", lw=0.9)
    axes[1].axhline(0, color="0.35", lw=0.9)
    axes[0].set_ylabel("active - shifted control\nhigh-tail fraction")
    axes[1].set_ylabel("active - shifted control\nmedian daily z")
    axes[1].set_xlabel("Ryle-Vonberg frequency (MHz)")
    axes[0].legend()
    fig.suptitle("Jupiter historical active windows compared with shifted inactive windows")
    fig.tight_layout()
    path = out_dir / "jupiter_historical_active_vs_shifted_spectrum.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_window_sample_counts(windows: pd.DataFrame, window_summary: pd.DataFrame, out_dir: Path) -> Path:
    counts = (
        window_summary.groupby("historical_window_id")["n_samples"]
        .sum()
        .rename("n_samples")
        .reset_index()
        .merge(windows[["historical_window_id", "event_start_time", "intensity", "burstiness"]], on="historical_window_id", how="right")
        .fillna({"n_samples": 0})
        .sort_values("event_start_time")
    )
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    colors = counts["intensity"].astype(float)
    sc = ax.scatter(pd.to_datetime(counts["event_start_time"]), counts["n_samples"], c=colors, cmap="plasma", s=34)
    ax.set_ylabel("Ryle-Vonberg samples in expanded window")
    ax.set_xlabel("historical Jupiter event time")
    ax.set_title("Historical Jupiter windows: RAE-2 sample coverage")
    fig.colorbar(sc, ax=ax, label="reported historical intensity")
    fig.autofmt_xdate()
    fig.tight_layout()
    path = out_dir / "jupiter_historical_window_sample_coverage.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_raw_time_grid(active: pd.DataFrame, out_dir: Path, top_n_windows: int, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    if active.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No active-window samples", ha="center", va="center")
        path = out_dir / "jupiter_historical_active_raw_time_grid.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path
    top = (
        active.groupby("historical_window_id")["daily_z_log_power"]
        .max()
        .sort_values(ascending=False)
        .head(int(top_n_windows))
        .index.tolist()
    )
    sub = active[active["historical_window_id"].isin(top)].copy()
    if len(sub) > 35000:
        sub = sub.iloc[rng.choice(len(sub), size=35000, replace=False)]
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.2), sharex=True, sharey=True)
    for ax, antenna in zip(axes, ["rv1_coarse", "rv2_coarse"]):
        a = sub[sub["antenna"].astype(str).eq(antenna)]
        if a.empty:
            ax.text(0.5, 0.5, "no samples", transform=ax.transAxes, ha="center", va="center")
            continue
        sc = ax.scatter(
            a["dt_from_window_center_min"],
            a["daily_z_log_power"],
            c=a["frequency_mhz"],
            cmap="viridis",
            s=8,
            alpha=0.55,
            rasterized=True,
        )
        ax.axhline(0, color="0.3", lw=0.8)
        ax.axhline(2.5, color="crimson", lw=0.9, ls="--")
        ax.axvline(0, color="0.2", lw=0.8)
        ax.set_title(ANTENNA_LABEL.get(antenna, antenna))
        ax.set_ylabel("daily-normalized log power")
        ax.set_ylim(-4, 8)
    axes[-1].set_xlabel("minutes from historical event-window center")
    fig.colorbar(sc, ax=axes.tolist(), pad=0.012, label="frequency (MHz)")
    fig.suptitle(f"Raw RAE-2 samples in top {len(top)} historical Jupiter active windows")
    fig.savefig(out_dir / "jupiter_historical_active_raw_time_grid.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_dir / "jupiter_historical_active_raw_time_grid.png"


def plot_phase_maps(phase: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    if phase.empty:
        return paths
    vals = phase["high_power_fraction"].dropna()
    vmax = max(0.02, float(vals.quantile(0.98))) if not vals.empty else 0.1
    for antenna in ["rv1_coarse", "rv2_coarse"]:
        suba = phase[phase["antenna"].astype(str).eq(antenna)]
        if suba.empty:
            continue
        freqs = sorted(suba["frequency_mhz"].unique())
        ncols = 3
        nrows = int(np.ceil(len(freqs) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, max(4.0, 3.2 * nrows)), sharex=True, sharey=True)
        axes = np.asarray(axes).reshape(-1)
        im = None
        for ax, freq in zip(axes, freqs):
            sub = suba[np.isclose(suba["frequency_mhz"].astype(float), float(freq))]
            mat = sub.pivot(index="io_bin_deg", columns="cml_bin_deg", values="high_power_fraction").sort_index()
            if mat.empty:
                ax.axis("off")
                continue
            im = ax.imshow(
                mat.to_numpy(dtype=float),
                origin="lower",
                extent=[
                    float(mat.columns.min() - 7.5),
                    float(mat.columns.max() + 7.5),
                    float(mat.index.min() - 7.5),
                    float(mat.index.max() + 7.5),
                ],
                aspect="auto",
                cmap="magma",
                vmin=0,
                vmax=vmax,
            )
            ax.set_title(f"{float(freq):g} MHz")
            ax.set_xlabel("CML (deg)")
            ax.set_ylabel("Io phase (deg)")
        for ax in axes[len(freqs) :]:
            ax.axis("off")
        fig.suptitle(f"Historical active-window Jupiter phase maps, {ANTENNA_LABEL.get(antenna, antenna)}")
        if im is not None:
            fig.colorbar(im, ax=axes.tolist(), shrink=0.8, pad=0.012, label="high-tail fraction")
        path = out_dir / f"jupiter_historical_active_phase_maps_{antenna}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_active_phase_scatter(active: pd.DataFrame, out_dir: Path, seed: int) -> list[Path]:
    paths = []
    rng = np.random.default_rng(seed)
    for antenna in ["rv1_coarse", "rv2_coarse"]:
        sub = active[active["antenna"].astype(str).eq(antenna) & active["jupiter_visible_by_moon"].astype(bool)].copy()
        if sub.empty:
            continue
        if len(sub) > 25000:
            sub = sub.iloc[rng.choice(len(sub), size=25000, replace=False)]
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        sc = ax.scatter(
            sub["jupiter_cml_spice_deg"],
            sub["io_phase_spice_deg"],
            c=sub["daily_z_log_power"],
            s=7,
            alpha=0.55,
            cmap="coolwarm",
            vmin=-2,
            vmax=5,
            rasterized=True,
        )
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 360)
        ax.set_xlabel("Jupiter CML (deg)")
        ax.set_ylabel("Io phase (deg)")
        ax.set_title(f"Historical active-window sample scatter, {ANTENNA_LABEL.get(antenna, antenna)}")
        fig.colorbar(sc, ax=ax, label="daily-normalized log power")
        fig.tight_layout()
        path = out_dir / f"jupiter_historical_active_phase_scatter_{antenna}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def write_report(
    out_dir: Path,
    windows: pd.DataFrame,
    active: pd.DataFrame,
    control: pd.DataFrame,
    active_shifted_summary: pd.DataFrame,
    phase: pd.DataFrame,
    paths: list[Path],
    config: dict[str, object],
) -> Path:
    top_delta = active_shifted_summary.copy()
    top_delta["abs_delta"] = top_delta["active_minus_shifted_high_tail_fraction"].abs()
    if not phase.empty:
        corr_rows = []
        for (antenna, band, freq), grp in phase.groupby(["antenna", "frequency_band", "frequency_mhz"], sort=True):
            valid = grp[["high_power_fraction", "median_maser_score"]].dropna()
            corr_rows.append(
                {
                    "antenna": antenna,
                    "frequency_mhz": float(freq),
                    "n_phase_bins": int(len(valid)),
                    "spearman_high_frac_vs_maser_score": (
                        float(valid["high_power_fraction"].corr(valid["median_maser_score"], method="spearman"))
                        if len(valid) >= 12
                        else np.nan
                    ),
                }
            )
        corr = pd.DataFrame(corr_rows)
        corr["abs_corr"] = corr["spearman_high_frac_vs_maser_score"].abs()
        corr_text = corr.sort_values("abs_corr", ascending=False).head(18).drop(columns=["abs_corr"]).to_string(index=False)
    else:
        corr_text = "(no phase bins met the minimum sample threshold)"
    lines = [
        "# Jupiter Historical Active-Window Phase Survey",
        "",
        "This run restricts the RAE-2 phase-pattern survey to historical Jupiter activity intervals transcribed from Warwick, Dulk, and Riddle (1975), Report UAG-42.",
        "",
        "The report frequency coverage is 7.6-80 MHz, so the historical table is used as a time/activity selector only. All Ryle-Vonberg frequencies are retained.",
        "",
        "## Configuration",
        "",
        *[f"- `{k}`: `{v}`" for k, v in config.items()],
        "",
        "## Coverage",
        "",
        f"- Historical windows transcribed: `{len(windows)}`",
        f"- Active-window RAE-2 samples: `{len(active)}`",
        f"- Shifted-control RAE-2 samples: `{len(control)}`",
        f"- Historical windows with at least one RAE-2 sample: `{active['historical_window_id'].nunique() if not active.empty else 0}`",
        "",
        "## Largest Active Minus Shifted-Control Differences",
        "",
        top_delta.sort_values("abs_delta", ascending=False)
        .head(18)[
            [
                "antenna",
                "frequency_mhz",
                "active_n_samples",
                "shifted_control_n_samples",
                "active_high_tail_fraction",
                "shifted_control_high_tail_fraction",
                "active_minus_shifted_high_tail_fraction",
                "active_minus_shifted_median_daily_z",
            ]
        ]
        .to_string(index=False),
        "",
        "## Io-CML Correlations Inside Historical Windows",
        "",
        corr_text,
        "",
        "## Interpretation Guardrails",
        "",
        "- These are not occultation detections.",
        "- The historical windows come from ground-based 7.6-80 MHz observations, so they can identify active Jovian intervals but cannot guarantee simultaneous sub-10 MHz RAE-2 emission.",
        "- A convincing result would show enhancement in historical windows relative to shifted controls and also visually concentrate high-power samples in plausible Io-CML regions.",
        "",
        "## Plots",
        "",
        *[f"- `{p}`" for p in paths],
    ]
    path = out_dir / "jupiter_historical_active_window_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--historical-windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--padding-min", type=float, default=30.0)
    parser.add_argument("--control-shift-days", type=int, nargs="*", default=[-7, 7])
    parser.add_argument("--high-z", type=float, default=2.5)
    parser.add_argument("--phase-bin-deg", type=float, default=15.0)
    parser.add_argument("--min-count-per-phase-bin", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260609)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    windows = load_windows(args.historical_windows, padding_min=float(args.padding_min))
    windows.to_csv(args.out_dir / "warwick_dulk_riddle_active_windows_expanded.csv", index=False)

    samples = read_table(args.samples, parse_dates=["time"])
    samples = samples.sort_values("time").reset_index(drop=True)

    active_labels = assign_intervals(samples["time"], _intervals_from_windows(windows, "expanded_start_time", "expanded_end_time"))
    samples["historical_window_id"] = active_labels
    active = samples[samples["historical_window_id"].astype(str).ne("")].copy()
    centers = windows.set_index("historical_window_id")["event_start_time"].combine(
        windows.set_index("historical_window_id")["event_end_time"],
        lambda a, b: pd.Timestamp(a) + (pd.Timestamp(b) - pd.Timestamp(a)) / 2,
    )
    active["historical_window_center"] = active["historical_window_id"].map(centers)
    active["dt_from_window_center_min"] = (
        pd.to_datetime(active["time"]) - pd.to_datetime(active["historical_window_center"])
    ).dt.total_seconds() / 60.0

    shifted = build_shifted_controls(windows, list(args.control_shift_days), padding_min=float(args.padding_min))
    control_labels = assign_intervals(samples["time"], _intervals_from_windows(shifted, "expanded_start_time", "expanded_end_time"))
    control = samples[control_labels.astype(str).ne("")].copy()
    control["shifted_control_window_id"] = control_labels[control_labels.astype(str).ne("")].to_numpy()

    active.to_csv(args.out_dir / "jupiter_historical_active_samples.csv", index=False)
    control.to_csv(args.out_dir / "jupiter_historical_shifted_control_samples.csv", index=False)

    active_all_summary = summarize_labeled_samples(samples, "historical_window_id", high_z=float(args.high_z))
    active_all_summary.to_csv(args.out_dir / "jupiter_historical_active_vs_all_inactive_summary.csv", index=False)

    active_shifted_summary = summarize_active_vs_shifted(active, control, high_z=float(args.high_z))
    active_shifted_summary.to_csv(args.out_dir / "jupiter_historical_active_vs_shifted_summary.csv", index=False)

    window_summary = summarize_windows(active, high_z=float(args.high_z))
    window_summary.to_csv(args.out_dir / "jupiter_historical_window_sample_summary.csv", index=False)

    phase = phase_binned(
        active,
        high_z=float(args.high_z),
        phase_bin_deg=float(args.phase_bin_deg),
        min_count=int(args.min_count_per_phase_bin),
    )
    phase.to_csv(args.out_dir / "jupiter_historical_active_phase_binned_summary.csv", index=False)

    paths = [
        plot_active_spectrum(active_shifted_summary, args.out_dir),
        plot_window_sample_counts(windows, window_summary, args.out_dir),
        plot_raw_time_grid(active, args.out_dir, top_n_windows=18, seed=int(args.seed)),
    ]
    paths.extend(plot_phase_maps(phase, args.out_dir))
    paths.extend(plot_active_phase_scatter(active, args.out_dir, seed=int(args.seed)))

    config = {
        "samples": str(args.samples),
        "historical_windows": str(args.historical_windows),
        "padding_min": float(args.padding_min),
        "control_shift_days": list(args.control_shift_days),
        "high_z": float(args.high_z),
        "phase_bin_deg": float(args.phase_bin_deg),
        "min_count_per_phase_bin": int(args.min_count_per_phase_bin),
    }
    report = write_report(args.out_dir, windows, active, control, active_shifted_summary, phase, paths, config)
    print(report)


if __name__ == "__main__":
    main()
