#!/usr/bin/env python
"""Build inspection plots for the science-baseline Earth/Sun/Jupiter run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.stacking import detrend_profile_values
from rylevonberg.util import datetime_ns


CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
ANT_LABEL = {"rv1_coarse": "upper V", "rv2_coarse": "lower V"}
COLORS = {"rv1_coarse": "#d95f02", "rv2_coarse": "#1b9e77"}
EVENT_COLORS = {"disappearance": "#4c78a8", "reappearance": "#d95f02"}


def _read(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return read_table(path, low_memory=False, **kwargs)


def _score_col(df: pd.DataFrame) -> str:
    return "robust_stack_snr" if "robust_stack_snr" in df.columns else "stacked_snr"


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _source_title(source: str) -> str:
    return source.capitalize()


def _antenna_label(antenna: str) -> str:
    return ANT_LABEL.get(str(antenna), str(antenna))


def _plot_summary(summary: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    vals = pd.to_numeric(summary["best_confirmed_robust_snr"], errors="coerce").to_numpy(dtype=float)
    ax.bar(summary["source_name"], vals, color=["#4c78a8" if v >= 0 else "#d95f02" for v in vals])
    ax.axhline(0.0, color="black", lw=0.8)
    ax.axhline(3.0, color="0.65", lw=0.8, ls="--")
    ax.axhline(-3.0, color="0.65", lw=0.8, ls="--")
    for i, val in enumerate(vals):
        ax.text(i, val, f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8)
    ax.set_title("Updated science-baseline run: best robust stack SNR")
    ax.set_ylabel("Robust stack SNR")
    return _save(fig, out / "science_baseline_best_robust_snr.png")


def _plot_spectrum(source: str, suite: Path, source_summary: pd.Series, out: Path) -> Path | None:
    scan = _read(suite / "initial_channel_scan.csv")
    if scan.empty:
        return None
    score = _score_col(scan)
    target_window = float(source_summary["target_window_s"])
    sub = scan[pd.to_numeric(scan["window_s"], errors="coerce").eq(target_window)].copy()
    if sub.empty:
        return None
    (out / source).mkdir(parents=True, exist_ok=True)
    sub.to_csv(out / source / f"{source}_robust_snr_spectrum.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for antenna, grp in sub.groupby("antenna", sort=True):
        grp = grp.sort_values("frequency_mhz")
        ax.plot(
            grp["frequency_mhz"],
            grp[score],
            marker="o",
            lw=1.9,
            color=COLORS.get(str(antenna)),
            label=_antenna_label(str(antenna)),
        )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.axhline(3.0, color="0.65", lw=0.8, ls="--")
    ax.axhline(-3.0, color="0.65", lw=0.8, ls="--")
    ax.axvline(float(source_summary["target_frequency_mhz"]), color="crimson", lw=1.0, ls=":", label="selected channel")
    ticks = sorted(pd.to_numeric(sub["frequency_mhz"], errors="coerce").dropna().unique())
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{x:.2g}" for x in ticks])
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Robust stack SNR")
    ax.set_title(f"{_source_title(source)} channel-strength spectrum ({int(target_window)} s window)")
    ax.legend(fontsize=8)
    return _save(fig, out / source / f"{source}_robust_snr_spectrum.png")


def _plot_timing(source: str, suite: Path, out: Path) -> Path | None:
    df = _read(suite / "timing_scan.csv")
    if df.empty:
        return None
    score = _score_col(df)
    df = df.sort_values("timing_offset_s")
    best = df.iloc[pd.to_numeric(df[score], errors="coerce").abs().argmax()]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(df["timing_offset_s"], df[score], marker="o", lw=1.6)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.axvspan(-60, 60, color="#1b9e77", alpha=0.12, label="good timing region")
    ax.axvline(0.0, color="0.45", lw=0.9, ls="--", label="predicted time")
    ax.axvline(float(best["timing_offset_s"]), color="crimson", lw=1.0, ls=":", label=f"best {best['timing_offset_s']:.0f}s")
    ax.set_title(f"{_source_title(source)} timing-offset scan")
    ax.set_xlabel("Template timing offset (s)")
    ax.set_ylabel("Robust stack SNR")
    ax.legend(fontsize=8)
    return _save(fig, out / source / f"{source}_timing_scan.png")


def _plot_window(source: str, suite: Path, out: Path) -> Path | None:
    df = _read(suite / "window_sweep.csv")
    if df.empty:
        return None
    score = _score_col(df)
    df = df.sort_values("window_s")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(df["window_s"], df[score], marker="o", lw=1.6)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.axhline(3.0, color="0.65", lw=0.8, ls="--")
    ax.axhline(-3.0, color="0.65", lw=0.8, ls="--")
    ax.set_title(f"{_source_title(source)} window-size repeatability")
    ax.set_xlabel("Window half-width (s)")
    ax.set_ylabel("Robust stack SNR")
    return _save(fig, out / source / f"{source}_window_sweep.png")


def _plot_event_type_bars(source: str, suite: Path, out: Path) -> Path | None:
    df = _read(suite / "event_type.csv")
    if df.empty:
        return None
    score = _score_col(df)
    labels = df["event_type"].astype(str).to_list()
    vals = pd.to_numeric(df[score], errors="coerce").to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.bar(labels, vals, color=[EVENT_COLORS.get(label, "#4c78a8") for label in labels])
    ax.axhline(0.0, color="black", lw=0.8)
    for i, val in enumerate(vals):
        ax.text(i, val, f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8)
    ax.set_title(f"{_source_title(source)} disappearance vs reappearance")
    ax.set_ylabel("Robust stack SNR")
    return _save(fig, out / source / f"{source}_event_type_split.png")


def _plot_quality(source: str, suite: Path, out: Path) -> Path | None:
    timing = _read(suite / "timing_scan.csv")
    strict = _read(suite / "strict_quality.csv")
    if timing.empty:
        return None
    score = _score_col(timing)
    best = timing.iloc[pd.to_numeric(timing[score], errors="coerce").abs().argmax()]
    vals = [float(best[score])]
    labels = ["all usable"]
    if not strict.empty:
        vals.append(float(strict.iloc[0][_score_col(strict)]))
        labels.append("strict quality")
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.bar(labels, vals, color=["#4c78a8" if v >= 0 else "#d95f02" for v in vals])
    ax.axhline(0.0, color="black", lw=0.8)
    for i, val in enumerate(vals):
        ax.text(i, val, f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8)
    ax.set_title(f"{_source_title(source)} quality survival")
    ax.set_ylabel("Robust stack SNR")
    return _save(fig, out / source / f"{source}_quality_survival.png")


def _plot_controls_and_spectral(source: str, suite: Path, out: Path) -> Path | None:
    timing = _read(suite / "timing_scan.csv")
    controls = _read(suite / "wrong_controls.csv")
    if timing.empty and controls.empty:
        return None
    score = _score_col(timing) if not timing.empty else _score_col(controls)
    rows = []
    if not timing.empty:
        best = timing.iloc[pd.to_numeric(timing[score], errors="coerce").abs().argmax()]
        rows.append({"label": "target", "value": float(best[score]), "kind": "target"})
    for _, row in controls.iterrows():
        ctype = str(row.get("control_type", "control"))
        label_type = "spectral support" if ctype == "neighboring_band" else ctype.replace("_", " ")
        rows.append(
            {
                "label": f"{label_type}\n{row['frequency_mhz']:.2g} MHz {_antenna_label(row['antenna'])}",
                "value": float(row[_score_col(controls)]),
                "kind": "spectral" if ctype == "neighboring_band" else "control",
            }
        )
    df = pd.DataFrame(rows)
    (out / source).mkdir(parents=True, exist_ok=True)
    df.to_csv(out / source / f"{source}_target_controls_spectral_support.csv", index=False)
    colors = {"target": "#4c78a8", "control": "#999999", "spectral": "#1b9e77"}
    fig, ax = plt.subplots(figsize=(max(7.0, 1.6 * len(df)), 4.2))
    ax.bar(df["label"], df["value"], color=[colors[k] for k in df["kind"]])
    ax.axhline(0.0, color="black", lw=0.8)
    for i, val in enumerate(df["value"].to_numpy(dtype=float)):
        ax.text(i, val, f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8)
    ax.set_title(f"{_source_title(source)} target, antenna control, and spectral support")
    ax.set_ylabel("Robust stack SNR")
    ax.tick_params(axis="x", labelrotation=20)
    return _save(fig, out / source / f"{source}_controls_and_spectral_support.png")


def _plot_leave_one_month(source: str, suite: Path, out: Path) -> Path | None:
    df = _read(suite / "leave_one_month.csv")
    if df.empty:
        return None
    score = _score_col(df)
    df = df.sort_values("left_out_month")
    vals = pd.to_numeric(df[score], errors="coerce").to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(df["left_out_month"].astype(str), vals, color=["#4c78a8" if v >= 0 else "#d95f02" for v in vals])
    ax.axhline(0.0, color="black", lw=0.8)
    if "full_stacked_snr" in df and score == "stacked_snr":
        ax.axhline(float(df["full_stacked_snr"].iloc[0]), color="crimson", lw=1.0, ls="--", label="full stack")
        ax.legend(fontsize=8)
    ax.axhline(3.0, color="0.65", lw=0.8, ls="--")
    ax.axhline(-3.0, color="0.65", lw=0.8, ls="--")
    ax.set_title(f"{_source_title(source)} leave-one-month-out")
    ax.set_xlabel("Month removed")
    ax.set_ylabel("Robust stack SNR")
    ax.tick_params(axis="x", labelrotation=45)
    return _save(fig, out / source / f"{source}_leave_one_month.png")


def _event_type_profile(clean: pd.DataFrame, events: pd.DataFrame, source_summary: pd.Series) -> pd.DataFrame:
    source = str(source_summary["source_name"])
    band = int(source_summary["target_band"])
    antenna = str(source_summary["target_antenna"])
    window = float(source_summary["target_window_s"])
    timing_offset = float(source_summary["best_timing_offset_s"])
    sub_events = events[
        events["source_name"].astype(str).eq(source)
        & events["frequency_band"].astype(int).eq(band)
        & events["antenna"].astype(str).eq(antenna)
    ].copy()
    group = clean[
        clean["frequency_band"].astype(int).eq(band)
        & clean["antenna"].astype(str).eq(antenna)
    ].sort_values("time").reset_index(drop=True)
    t_ns = datetime_ns(group["time"])
    rows = []
    half_ns = int(window * 1e9)
    for _, ev in sub_events.iterrows():
        event_ns = pd.Timestamp(ev["predicted_event_time"]).value
        lo = int(np.searchsorted(t_ns, event_ns - half_ns, side="left"))
        hi = int(np.searchsorted(t_ns, event_ns + half_ns, side="right"))
        if hi <= lo:
            continue
        local = group.iloc[lo:hi]
        rel = (datetime_ns(local["time"]) - event_ns).astype(float) / 1e9
        y = pd.to_numeric(local["power"], errors="coerce").to_numpy(dtype=float)
        valid = local["is_valid"].to_numpy(dtype=bool) if "is_valid" in local.columns else np.ones(len(local), dtype=bool)
        keep = valid & np.isfinite(y) & (np.abs(rel) <= window)
        if np.count_nonzero(keep) < 6:
            continue
        tr = rel[keep]
        yy = y[keep]
        profile, template, sigma, _n = detrend_profile_values(
            tr,
            yy,
            str(ev["event_type"]),
            baseline_mode="sideband_linear",
            sideband_exclusion_seconds=120.0,
            timing_offset_sec=timing_offset,
            normalize=True,
        )
        tbin = np.round(tr / 60.0) * 60.0
        for tb, val in zip(tbin, profile):
            rows.append(
                {
                    "source_name": source,
                    "frequency_band": band,
                    "frequency_mhz": float(source_summary["target_frequency_mhz"]),
                    "antenna": antenna,
                    "event_type": ev["event_type"],
                    "event_id": ev.get("event_id"),
                    "t_bin_sec": float(tb),
                    "profile_value": float(val),
                    "local_sigma": float(sigma) if np.isfinite(sigma) else np.nan,
                }
            )
    raw = pd.DataFrame.from_records(rows)
    if raw.empty:
        return raw
    out_rows = []
    for keys, grp in raw.groupby(["source_name", "frequency_band", "frequency_mhz", "antenna", "event_type", "t_bin_sec"], sort=True):
        source, band, mhz, antenna, event_type, tbin = keys
        vals = grp["profile_value"].to_numpy(dtype=float)
        out_rows.append(
            {
                "source_name": source,
                "frequency_band": band,
                "frequency_mhz": mhz,
                "antenna": antenna,
                "event_type": event_type,
                "t_bin_sec": tbin,
                "n_samples": int(vals.size),
                "n_events": int(grp["event_id"].nunique()),
                "mean": float(np.nanmean(vals)),
                "median": float(np.nanmedian(vals)),
                "sem": float(np.nanstd(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else np.nan,
            }
        )
    return pd.DataFrame.from_records(out_rows)


def _plot_event_type_profile(source: str, clean: pd.DataFrame, events: pd.DataFrame, source_summary: pd.Series, out: Path) -> Path | None:
    profile = _event_type_profile(clean, events, source_summary)
    if profile.empty:
        return None
    (out / source).mkdir(parents=True, exist_ok=True)
    profile.to_csv(out / source / f"{source}_best_channel_event_type_profile.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    for event_type, grp in profile.groupby("event_type", sort=True):
        grp = grp.sort_values("t_bin_sec")
        x = grp["t_bin_sec"] / 60.0
        y = grp["mean"]
        sem = grp["sem"].fillna(0.0)
        ax.plot(x, y, marker="o", ms=3, lw=1.8, color=EVENT_COLORS.get(str(event_type)), label=f"{event_type} ({int(grp['n_events'].max())} events)")
        ax.fill_between(x, y - sem, y + sem, alpha=0.16, color=EVENT_COLORS.get(str(event_type)), linewidth=0)
    ax.axvline(0.0, color="black", lw=1.0, ls="--", label="predicted time")
    if float(source_summary["best_timing_offset_s"]) != 0.0:
        ax.axvline(float(source_summary["best_timing_offset_s"]) / 60.0, color="crimson", lw=1.0, ls=":", label="best timing offset")
    ax.axhline(0.0, color="0.35", lw=0.8)
    ax.set_title(
        f"{_source_title(source)} {float(source_summary['target_frequency_mhz']):.2f} MHz "
        f"{_antenna_label(source_summary['target_antenna'])} profile"
    )
    ax.set_xlabel("Relative time from predicted event (min)")
    ax.set_ylabel("Mean normalized sideband residual")
    ax.legend(fontsize=8)
    return _save(fig, out / source / f"{source}_best_channel_event_type_profile.png")


def _write_readme(out: Path, summary_path: Path, paths: dict[str, list[Path]], summary_plot: Path | None) -> None:
    lines = [
        "# Science-Baseline Inspection Plots",
        "",
        f"Generated from `{summary_path}`.",
        "",
        "These plots use the updated sideband-linear profile baseline and robust stack SNR where available.",
        "Neighboring frequency bands are labeled as spectral support, not wrong-band controls.",
        "",
    ]
    if summary_plot is not None:
        lines.extend(["## Summary", "", f"- [{summary_plot.name}]({summary_plot.relative_to(out)})", ""])
    for source, source_paths in paths.items():
        lines.extend([f"## {_source_title(source)}", ""])
        for path in source_paths:
            lines.append(f"- [{path.name}]({path.relative_to(out)})")
        lines.append("")
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey-root", default="outputs/planetary_confirmation_survey_science_baseline_v2")
    parser.add_argument("--output-dir", default="outputs/inspection_plots_science_baseline_v2")
    args = parser.parse_args()

    survey = ROOT / args.survey_root
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    summary_path = survey / "planetary_confirmation_summary.csv"
    summary = _read(summary_path)
    clean = _read(CLEAN, parse_dates=["time"])
    events = _read(survey / "events" / "all_planet_predicted_events.csv", parse_dates=["predicted_event_time"])

    paths: dict[str, list[Path]] = {}
    summary_plot = _plot_summary(summary, out) if not summary.empty else None
    for _, row in summary.iterrows():
        source = str(row["source_name"])
        suite = survey / source
        source_paths: list[Path] = []
        for func in [
            _plot_spectrum,
            _plot_timing,
            _plot_window,
            _plot_event_type_bars,
            _plot_event_type_profile,
            _plot_quality,
            _plot_controls_and_spectral,
            _plot_leave_one_month,
        ]:
            if func is _plot_spectrum:
                path = func(source, suite, row, out)
            elif func is _plot_event_type_profile:
                path = func(source, clean, events, row, out)
            else:
                path = func(source, suite, out)
            if path is not None:
                source_paths.append(path)
        paths[source] = source_paths

    _write_readme(out, summary_path, paths, summary_plot)
    print(out / "README.md")


if __name__ == "__main__":
    main()
