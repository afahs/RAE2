#!/usr/bin/env python
"""Build raw-power diagnostics for best Earth/Sun/Jupiter channels."""

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

from rylevonberg.detection import baseline_matrix, event_template
from rylevonberg.util import datetime_ns, robust_sigma


CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
ANT_LABEL = {"rv1_coarse": "upper V", "rv2_coarse": "lower V"}
EVENT_COLORS = {"disappearance": "#4c78a8", "reappearance": "#d95f02"}


def _read(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return read_table(path, low_memory=False, **kwargs)


def _antenna_label(antenna: str) -> str:
    return ANT_LABEL.get(str(antenna), str(antenna))


def _source_title(source: str) -> str:
    return source.capitalize()


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _event_window(group: pd.DataFrame, group_ns: np.ndarray, event: pd.Series, window_s: float) -> tuple[np.ndarray, np.ndarray] | None:
    event_ns = pd.Timestamp(event["predicted_event_time"]).value
    half_ns = int(float(window_s) * 1e9)
    lo = int(np.searchsorted(group_ns, event_ns - half_ns, side="left"))
    hi = int(np.searchsorted(group_ns, event_ns + half_ns, side="right"))
    if hi <= lo:
        return None
    local = group.iloc[lo:hi]
    rel = (datetime_ns(local["time"]) - event_ns).astype(float) / 1e9
    y = pd.to_numeric(local["power"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y) & (np.abs(rel) <= float(window_s))
    if "is_valid" in local.columns:
        valid &= local["is_valid"].to_numpy(dtype=bool)
    if np.count_nonzero(valid) < 8:
        return None
    tr = rel[valid]
    yy = y[valid]
    order = np.argsort(tr)
    return tr[order], yy[order]


def _fit_joint_model(tr: np.ndarray, y: np.ndarray, event_type: str, timing_offset_s: float) -> dict[str, np.ndarray | float]:
    baseline = baseline_matrix(tr, 1)
    template = event_template(tr, event_type, timing_offset_sec=timing_offset_s)
    X = np.column_stack([baseline, template])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    baseline_y = baseline @ beta[:-1]
    model_y = X @ beta
    resid = y - model_y
    sigma = robust_sigma(resid)
    amp = float(beta[-1])
    return {
        "baseline": baseline_y,
        "model": model_y,
        "amp": amp,
        "sigma": float(sigma) if np.isfinite(sigma) else np.nan,
        "amp_over_sigma": float(amp / sigma) if np.isfinite(sigma) and sigma > 0 else np.nan,
    }


def _collect_raw_samples(
    clean: pd.DataFrame,
    events: pd.DataFrame,
    source_summary: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = str(source_summary["source_name"])
    band = int(source_summary["target_band"])
    antenna = str(source_summary["target_antenna"])
    window_s = float(source_summary["target_window_s"])
    timing_offset_s = float(source_summary["best_timing_offset_s"])
    group = clean[
        clean["frequency_band"].astype(int).eq(band)
        & clean["antenna"].astype(str).eq(antenna)
    ].sort_values("time").reset_index(drop=True)
    group_ns = datetime_ns(group["time"])
    sub_events = events[
        events["source_name"].astype(str).eq(source)
        & events["frequency_band"].astype(int).eq(band)
        & events["antenna"].astype(str).eq(antenna)
    ].copy()
    sample_rows = []
    fit_rows = []
    for _, ev in sub_events.iterrows():
        local = _event_window(group, group_ns, ev, window_s)
        if local is None:
            continue
        tr, y = local
        fit = _fit_joint_model(tr, y, str(ev["event_type"]), timing_offset_s)
        event_id = ev.get("event_id")
        fit_rows.append(
            {
                "source_name": source,
                "event_id": event_id,
                "event_type": ev["event_type"],
                "predicted_event_time": ev["predicted_event_time"],
                "frequency_band": band,
                "frequency_mhz": float(source_summary["target_frequency_mhz"]),
                "antenna": antenna,
                "window_s": window_s,
                "timing_offset_s": timing_offset_s,
                "n_samples": int(len(tr)),
                "joint_step_amplitude": float(fit["amp"]),
                "joint_step_sigma": float(fit["sigma"]) if np.isfinite(fit["sigma"]) else np.nan,
                "joint_amp_over_sigma": float(fit["amp_over_sigma"]) if np.isfinite(fit["amp_over_sigma"]) else np.nan,
            }
        )
        tbin = np.round(tr / 60.0) * 60.0
        for trel, tb, yy, base, model in zip(tr, tbin, y, fit["baseline"], fit["model"]):
            sample_rows.append(
                {
                    "source_name": source,
                    "event_id": event_id,
                    "event_type": ev["event_type"],
                    "predicted_event_time": ev["predicted_event_time"],
                    "frequency_band": band,
                    "frequency_mhz": float(source_summary["target_frequency_mhz"]),
                    "antenna": antenna,
                    "window_s": window_s,
                    "timing_offset_s": timing_offset_s,
                    "t_rel_sec": float(trel),
                    "t_bin_sec": float(tb),
                    "raw_power": float(yy),
                    "joint_baseline": float(base),
                    "joint_model": float(model),
                }
            )
    return pd.DataFrame.from_records(sample_rows), pd.DataFrame.from_records(fit_rows)


def _stack_raw(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, grp in samples.groupby(["source_name", "frequency_band", "frequency_mhz", "antenna", "event_type", "t_bin_sec"], sort=True):
        source, band, mhz, antenna, event_type, tbin = keys
        vals = grp["raw_power"].to_numpy(dtype=float)
        rows.append(
            {
                "source_name": source,
                "frequency_band": band,
                "frequency_mhz": mhz,
                "antenna": antenna,
                "event_type": event_type,
                "t_bin_sec": float(tbin),
                "n_samples": int(vals.size),
                "n_events": int(grp["event_id"].nunique()),
                "mean_raw_power": float(np.nanmean(vals)),
                "median_raw_power": float(np.nanmedian(vals)),
                "sem_raw_power": float(np.nanstd(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else np.nan,
            }
        )
    return pd.DataFrame.from_records(rows)


def _plot_raw_stack(source: str, stack: pd.DataFrame, source_summary: pd.Series, out: Path) -> Path | None:
    if stack.empty:
        return None
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for event_type, grp in stack.groupby("event_type", sort=True):
        grp = grp.sort_values("t_bin_sec")
        x = grp["t_bin_sec"] / 60.0
        y = grp["mean_raw_power"]
        sem = grp["sem_raw_power"].fillna(0.0)
        ax.plot(x, y, marker="o", ms=3, lw=1.7, color=EVENT_COLORS.get(str(event_type)), label=f"{event_type} ({int(grp['n_events'].max())} events)")
        ax.fill_between(x, y - sem, y + sem, color=EVENT_COLORS.get(str(event_type)), alpha=0.15, linewidth=0)
    ax.axvline(0.0, color="black", lw=1.0, ls="--", label="predicted time")
    if float(source_summary["best_timing_offset_s"]) != 0.0:
        ax.axvline(float(source_summary["best_timing_offset_s"]) / 60.0, color="crimson", lw=1.0, ls=":", label="best timing offset")
    ax.set_title(
        f"{_source_title(source)} raw power stack: {float(source_summary['target_frequency_mhz']):.2f} MHz "
        f"{_antenna_label(source_summary['target_antenna'])}"
    )
    ax.set_xlabel("Relative time from predicted event (min)")
    ax.set_ylabel("Mean raw power")
    ax.legend(fontsize=8)
    return _save(fig, out / source / f"{source}_raw_power_event_type_stack.png")


def _plot_overlay_montage(
    source: str,
    samples: pd.DataFrame,
    fits: pd.DataFrame,
    source_summary: pd.Series,
    out: Path,
    n_per_type: int = 3,
) -> Path | None:
    if samples.empty or fits.empty:
        return None
    ranked = (
        fits.assign(abs_fit=lambda d: pd.to_numeric(d["joint_amp_over_sigma"], errors="coerce").abs())
        .sort_values("abs_fit", ascending=False)
        .groupby("event_type", sort=True)
        .head(n_per_type)
    )
    if ranked.empty:
        return None
    n = min(len(ranked), 2 * n_per_type)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.2, 3.2 * nrows), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (_, fit) in zip(axes, ranked.head(n).iterrows()):
        ev_samples = samples[samples["event_id"].astype(str).eq(str(fit["event_id"]))].sort_values("t_rel_sec")
        ax.plot(ev_samples["t_rel_sec"] / 60.0, ev_samples["raw_power"], ".", ms=3, color="0.62", label="raw power")
        ax.plot(ev_samples["t_rel_sec"] / 60.0, ev_samples["joint_baseline"], "--", lw=1.5, color="#ff7f0e", label="joint baseline")
        ax.plot(ev_samples["t_rel_sec"] / 60.0, ev_samples["joint_model"], "-", lw=1.7, color="#d62728", label="baseline + step")
        ax.axvline(0.0, color="black", lw=0.9, ls="--")
        if float(source_summary["best_timing_offset_s"]) != 0.0:
            ax.axvline(float(source_summary["best_timing_offset_s"]) / 60.0, color="crimson", lw=0.9, ls=":")
        ax.set_title(
            f"{fit['event_type']} {pd.Timestamp(fit['predicted_event_time']).date()} "
            f"A/sigma={float(fit['joint_amp_over_sigma']):.2f}"
        )
        ax.set_ylabel("Raw power")
    for ax in axes[n:]:
        ax.axis("off")
    axes[0].legend(fontsize=8)
    for ax in axes[-ncols:]:
        ax.set_xlabel("Relative time from predicted event (min)")
    fig.suptitle(
        f"{_source_title(source)} raw power model overlays: {float(source_summary['target_frequency_mhz']):.2f} MHz "
        f"{_antenna_label(source_summary['target_antenna'])}"
    )
    return _save(fig, out / source / f"{source}_raw_power_model_overlay_montage.png")


def _write_readme(out: Path, paths: dict[str, list[Path]]) -> None:
    lines = [
        "# Raw-Power Source Diagnostics",
        "",
        "These plots use original Ryle-Vonberg power samples for the best updated science-baseline channel of each source.",
        "The raw stack is not baseline-subtracted. The model-overlay montage shows raw samples with a joint baseline+step model overplotted.",
        "",
    ]
    for source, source_paths in paths.items():
        lines.extend([f"## {_source_title(source)}", ""])
        for path in source_paths:
            lines.append(f"- [{path.name}]({path.relative_to(out)})")
        lines.append("")
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey-root", default="outputs/planetary_confirmation_survey_science_baseline_v2")
    parser.add_argument("--output-dir", default="outputs/raw_power_diagnostics_science_baseline_v2")
    args = parser.parse_args()

    survey = ROOT / args.survey_root
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    clean = _read(CLEAN, parse_dates=["time"])
    events = _read(survey / "events" / "all_planet_predicted_events.csv", parse_dates=["predicted_event_time"])
    summary = _read(survey / "planetary_confirmation_summary.csv")
    paths: dict[str, list[Path]] = {}
    for _, row in summary.iterrows():
        source = str(row["source_name"])
        (out / source).mkdir(parents=True, exist_ok=True)
        samples, fits = _collect_raw_samples(clean, events, row)
        samples.to_csv(out / source / f"{source}_raw_power_samples.csv", index=False)
        fits.to_csv(out / source / f"{source}_raw_power_joint_fit_summary.csv", index=False)
        stack = _stack_raw(samples)
        stack.to_csv(out / source / f"{source}_raw_power_event_type_stack.csv", index=False)
        source_paths = []
        p = _plot_raw_stack(source, stack, row, out)
        if p is not None:
            source_paths.append(p)
        p = _plot_overlay_montage(source, samples, fits, row, out)
        if p is not None:
            source_paths.append(p)
        paths[source] = source_paths
    _write_readme(out, paths)
    print(out / "README.md")


if __name__ == "__main__":
    main()
