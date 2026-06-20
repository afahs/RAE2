#!/usr/bin/env python
"""Diagnose whether sparse sampling can smear occultation steps into slopes."""

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

from rylevonberg.detection import event_template  # noqa: E402
from rylevonberg.util import datetime_ns, ensure_dir, robust_sigma, software_versions, write_json  # noqa: E402


CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
SUN_EVENTS = ROOT / "outputs/pipeline_confidence_audit_v2/sun_audit_input_events.csv"
BRIGHT_EVENTS = ROOT / "outputs/control_survey_bright_sources_postnov1974_v1/02_events/predicted_events.csv"
PLANET_EVENTS = ROOT / "outputs/planetary_confirmation_survey_science_baseline_v2/events/all_planet_predicted_events.csv"
ANT_LABEL = {"rv1_coarse": "upper V", "rv2_coarse": "lower V"}
ANT_COLOR = {"rv1_coarse": "#4c78a8", "rv2_coarse": "#d95f02"}


def _read(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return read_table(path, low_memory=False, **kwargs)


def _make_groups(clean: pd.DataFrame) -> dict[tuple[int, str], tuple[pd.DataFrame, np.ndarray]]:
    groups = {}
    for (band, antenna), grp in clean.groupby(["frequency_band", "antenna"], sort=True):
        g = grp.sort_values("time").reset_index(drop=True)
        groups[(int(band), str(antenna))] = (g, datetime_ns(g["time"]))
    return groups


def _window_times(
    group: pd.DataFrame,
    group_ns: np.ndarray,
    event_time: pd.Timestamp,
    window_s: float,
) -> np.ndarray:
    event_ns = pd.Timestamp(event_time).value
    half_ns = int(float(window_s) * 1e9)
    lo = int(np.searchsorted(group_ns, event_ns - half_ns, side="left"))
    hi = int(np.searchsorted(group_ns, event_ns + half_ns, side="right"))
    if hi <= lo:
        return np.array([], dtype=float)
    local = group.iloc[lo:hi]
    t = (datetime_ns(local["time"]) - event_ns).astype(float) / 1e9
    y = pd.to_numeric(local["power"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y) & (np.abs(t) <= float(window_s))
    if "is_valid" in local.columns:
        valid &= local["is_valid"].to_numpy(dtype=bool)
    return np.sort(t[valid])


def _ramp_template(t: np.ndarray, event_type: str, transition_s: float) -> np.ndarray:
    """Positive-source occultation template with finite transition duration."""

    point = event_template(t, event_type)
    if transition_s <= 0:
        return point
    x = np.clip(t / float(transition_s), -0.5, 0.5)
    if str(event_type).lower() == "reappearance":
        return x
    return -x


def collect_sampling(
    clean: pd.DataFrame,
    events: pd.DataFrame,
    source: str,
    window_s: float,
    inner_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = _make_groups(clean)
    work = events[events["source_name"].astype(str).str.lower().eq(source.lower())].copy()
    event_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    for _, ev in work.iterrows():
        payload = groups.get((int(ev["frequency_band"]), str(ev["antenna"])))
        if payload is None:
            continue
        group, group_ns = payload
        t = _window_times(group, group_ns, pd.Timestamp(ev["predicted_event_time"]), window_s)
        before = t[t < 0]
        after = t[t > 0]
        nearest_before = float(before.max()) if before.size else np.nan
        nearest_after = float(after.min()) if after.size else np.nan
        nearest_abs = float(np.nanmin(np.abs(t))) if t.size else np.nan
        bracket_gap = float(nearest_after - nearest_before) if np.isfinite(nearest_before) and np.isfinite(nearest_after) else np.nan
        inner_before = int(np.count_nonzero((t >= -float(inner_s)) & (t < 0)))
        inner_after = int(np.count_nonzero((t > 0) & (t <= float(inner_s))))
        large_gap = np.nan
        if t.size >= 2:
            large_gap = float(np.nanmax(np.diff(t)))
        event_rows.append(
            {
                "source_name": source.lower(),
                "event_id": ev.get("event_id"),
                "event_type": ev.get("event_type"),
                "frequency_band": int(ev["frequency_band"]),
                "frequency_mhz": float(ev["frequency_mhz"]),
                "antenna": str(ev["antenna"]),
                "n_valid_samples": int(t.size),
                "nearest_sample_abs_s": nearest_abs,
                "nearest_before_s": nearest_before,
                "nearest_after_s": nearest_after,
                "bracketing_gap_s": bracket_gap,
                "largest_gap_s": large_gap,
                "inner_samples_before": inner_before,
                "inner_samples_after": inner_after,
                "has_both_sides": bool(before.size and after.size),
                "has_inner_bracket": bool(inner_before and inner_after),
            }
        )
        for tv in t:
            sample_rows.append(
                {
                    "source_name": source.lower(),
                    "event_id": ev.get("event_id"),
                    "event_type": ev.get("event_type"),
                    "frequency_band": int(ev["frequency_band"]),
                    "frequency_mhz": float(ev["frequency_mhz"]),
                    "antenna": str(ev["antenna"]),
                    "t_rel_sec": float(tv),
                }
            )
    return pd.DataFrame.from_records(event_rows), pd.DataFrame.from_records(sample_rows)


def summarize_events(event_metrics: pd.DataFrame, inner_s: float) -> pd.DataFrame:
    rows = []
    by = ["source_name", "frequency_band", "frequency_mhz", "antenna", "event_type"]
    for keys, grp in event_metrics.groupby(by, sort=True, dropna=False):
        nearest = pd.to_numeric(grp["nearest_sample_abs_s"], errors="coerce")
        bracket = pd.to_numeric(grp["bracketing_gap_s"], errors="coerce")
        n_valid = pd.to_numeric(grp["n_valid_samples"], errors="coerce")
        rows.append(
            {
                **dict(zip(by, keys)),
                "n_predicted_event_windows": int(len(grp)),
                "median_valid_samples": float(np.nanmedian(n_valid)),
                "median_nearest_sample_abs_s": float(np.nanmedian(nearest)),
                "frac_nearest_le_30s": float(np.nanmean(nearest <= 30.0)),
                "frac_nearest_le_60s": float(np.nanmean(nearest <= 60.0)),
                "frac_nearest_le_120s": float(np.nanmean(nearest <= 120.0)),
                "frac_has_both_sides": float(np.nanmean(grp["has_both_sides"].astype(bool))),
                f"frac_has_samples_both_sides_within_{int(inner_s)}s": float(np.nanmean(grp["has_inner_bracket"].astype(bool))),
                "median_bracketing_gap_s": float(np.nanmedian(bracket)),
                "p90_bracketing_gap_s": float(np.nanpercentile(bracket.dropna(), 90)) if bracket.notna().any() else np.nan,
            }
        )
    return pd.DataFrame.from_records(rows)


def simulate_template_profiles(samples: pd.DataFrame, bin_s: float, window_s: float, transition_grid_s: list[float]) -> pd.DataFrame:
    rows = []
    if samples.empty:
        return pd.DataFrame()
    bins = np.arange(-float(window_s), float(window_s) + float(bin_s), float(bin_s))
    by = ["source_name", "frequency_band", "frequency_mhz", "antenna", "event_type"]
    for keys, grp in samples.groupby(by, sort=True, dropna=False):
        t = grp["t_rel_sec"].to_numpy(dtype=float)
        bin_idx = np.digitize(t, bins) - 1
        for transition_s in transition_grid_s:
            y = _ramp_template(t, str(keys[-1]), float(transition_s))
            for idx in sorted(set(bin_idx)):
                if idx < 0 or idx >= len(bins) - 1:
                    continue
                mask = bin_idx == idx
                if not np.any(mask):
                    continue
                rows.append(
                    {
                        **dict(zip(by, keys)),
                        "transition_s": float(transition_s),
                        "t_bin_sec": float(0.5 * (bins[idx] + bins[idx + 1])),
                        "median_template": float(np.nanmedian(y[mask])),
                        "n_samples": int(np.count_nonzero(mask)),
                    }
                )
    return pd.DataFrame.from_records(rows)


def summarize_template_shape(template_profiles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    by = ["source_name", "frequency_band", "frequency_mhz", "antenna", "event_type", "transition_s"]
    for keys, grp in template_profiles.groupby(by, sort=True, dropna=False):
        g = grp.sort_values("t_bin_sec")
        t = g["t_bin_sec"].to_numpy(dtype=float)
        y = g["median_template"].to_numpy(dtype=float)
        near = np.abs(t) <= 180.0
        side = np.abs(t) >= 300.0
        near_slope = np.nan
        if np.count_nonzero(near) >= 3:
            near_slope = float(np.polyfit(t[near], y[near], 1)[0])
        contrast = np.nan
        if np.any(t < -300.0) and np.any(t > 300.0):
            contrast = float(np.nanmedian(y[t > 300.0]) - np.nanmedian(y[t < -300.0]))
        rows.append(
            {
                **dict(zip(by, keys)),
                "near_event_slope_per_s": near_slope,
                "far_side_contrast": contrast,
                "n_bins": int(len(g)),
                "n_near_bins": int(np.count_nonzero(near)),
                "n_side_bins": int(np.count_nonzero(side)),
            }
        )
    return pd.DataFrame.from_records(rows)


def plot_sampling_summary(summary: pd.DataFrame, source: str, out_dir: Path) -> Path | None:
    sub = summary[summary["source_name"].eq(source)].copy()
    if sub.empty:
        return None
    freqs = sorted(sub["frequency_mhz"].dropna().unique())
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for antenna, grp in sub.groupby("antenna", sort=True):
        label = ANT_LABEL.get(str(antenna), str(antenna))
        color = ANT_COLOR.get(str(antenna))
        g = grp.groupby("frequency_mhz", as_index=False).agg(
            median_nearest_sample_abs_s=("median_nearest_sample_abs_s", "median"),
            median_bracketing_gap_s=("median_bracketing_gap_s", "median"),
            frac_has_inner=("frac_has_samples_both_sides_within_180s", "median"),
        )
        axes[0].plot(g["frequency_mhz"], g["median_nearest_sample_abs_s"], marker="o", label=label, color=color)
        axes[1].plot(g["frequency_mhz"], g["median_bracketing_gap_s"], marker="o", label=label, color=color)
        axes[2].plot(g["frequency_mhz"], g["frac_has_inner"], marker="o", label=label, color=color)
    axes[0].set_ylabel("median |nearest sample| (s)")
    axes[1].set_ylabel("median gap across event (s)")
    axes[2].set_ylabel("fraction bracketed within 180 s")
    axes[2].set_xlabel("frequency (MHz)")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_xticks(freqs)
    axes[0].legend(frameon=False)
    fig.suptitle(f"{source}: sampling support around predicted occultation")
    fig.tight_layout()
    path = out_dir / f"{source}_sampling_support_by_frequency.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_template_smear(template_profiles: pd.DataFrame, source: str, out_dir: Path, frequency_mhz: float | None = None) -> Path | None:
    sub = template_profiles[template_profiles["source_name"].eq(source)].copy()
    if sub.empty:
        return None
    if frequency_mhz is None:
        candidates = sub.groupby("frequency_mhz").size().sort_values(ascending=False)
        frequency_mhz = float(candidates.index[0])
    sub = sub[np.isclose(sub["frequency_mhz"], float(frequency_mhz))]
    if sub.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for j, event_type in enumerate(["disappearance", "reappearance"]):
        ax = axes[j]
        et = sub[sub["event_type"].astype(str).eq(event_type)]
        for transition_s, grp in et.groupby("transition_s", sort=True):
            g = grp.groupby("t_bin_sec", as_index=False)["median_template"].median().sort_values("t_bin_sec")
            ax.plot(g["t_bin_sec"], g["median_template"], marker="o", markersize=2.5, label=f"{transition_s:.0f} s")
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.axhline(0, color="0.6", linewidth=0.7)
        ax.set_title(f"{event_type}, {frequency_mhz:.2f} MHz")
        ax.set_xlabel("seconds from predicted event")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("ideal template sampled at real times")
    axes[1].legend(title="transition", frameon=False, fontsize=8)
    fig.suptitle(f"{source}: sampling-only template smear diagnostic")
    fig.tight_layout()
    path = out_dir / f"{source}_sampling_template_smear_{frequency_mhz:.2f}MHz.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_template_smear_grid(template_profiles: pd.DataFrame, source: str, antenna: str, out_dir: Path) -> Path | None:
    sub = template_profiles[
        template_profiles["source_name"].eq(source) & template_profiles["antenna"].astype(str).eq(str(antenna))
    ].copy()
    if sub.empty:
        return None
    freqs = sorted(sub["frequency_mhz"].dropna().unique())
    transitions = [t for t in [0.0, 300.0, 900.0] if np.any(np.isclose(sub["transition_s"], t))]
    if not transitions:
        transitions = sorted(sub["transition_s"].dropna().unique())[:3]
    colors = {0.0: "#222222", 300.0: "#4c78a8", 900.0: "#d95f02"}
    fig, axes = plt.subplots(len(freqs), 2, figsize=(12, max(10, 1.25 * len(freqs))), sharex=True, sharey=True)
    if len(freqs) == 1:
        axes = np.asarray([axes])
    for i, freq in enumerate(freqs):
        for j, event_type in enumerate(["disappearance", "reappearance"]):
            ax = axes[i, j]
            for transition_s in transitions:
                grp = sub[
                    np.isclose(sub["frequency_mhz"], float(freq))
                    & sub["event_type"].astype(str).eq(event_type)
                    & np.isclose(sub["transition_s"], float(transition_s))
                ]
                if grp.empty:
                    continue
                g = grp.groupby("t_bin_sec", as_index=False)["median_template"].median().sort_values("t_bin_sec")
                ax.plot(
                    g["t_bin_sec"],
                    g["median_template"],
                    linewidth=1.2,
                    marker="o",
                    markersize=2.0,
                    color=colors.get(float(transition_s)),
                    label=f"{transition_s:.0f} s",
                )
            ax.axvline(0, color="black", linestyle="--", linewidth=0.7)
            ax.axhline(0, color="0.6", linewidth=0.7)
            ax.set_title(f"{freq:.2f} MHz {event_type}", fontsize=9)
            if j == 0:
                ax.set_ylabel("sampled ideal template")
            if i == len(freqs) - 1:
                ax.set_xlabel("seconds from predicted event")
            if i == 0 and j == 1:
                ax.legend(title="transition", frameon=False, fontsize=8)
    fig.suptitle(
        f"{source}: all-frequency sampling-only template smear ({ANT_LABEL.get(antenna, antenna)})\n"
        "0 s is a point-step; longer transitions approximate extended-source/timing-smearing ramps.",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    path = out_dir / f"{source}_{antenna}_all_frequency_template_smear_grid.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(out_dir: Path, summary: pd.DataFrame, shape: pd.DataFrame, paths: list[Path], inner_s: float) -> None:
    lines = [
        "# Sampling Smearing Diagnostic",
        "",
        "Question tested: can sparse grouped sampling around predicted occultations make a sharp step look like a line or gradual slope?",
        "",
        "Method:",
        "",
        "- measure where valid samples fall relative to each predicted event;",
        "- measure the gap between the closest sample before and after the predicted event;",
        "- measure whether an event is actually bracketed within the central window;",
        "- sample ideal point-source and finite-duration occultation templates at the same times;",
        "- compare whether the expected sampled template is step-like, under-sampled, or inherently ramp-like.",
        "",
        "Interpretation:",
        "",
        "- if the median bracketing gap is several minutes, the plotted line across zero is mostly interpolation across unobserved time;",
        "- if few events have samples on both sides within the central window, individual event fits are poorly constrained;",
        "- finite source size, limb geometry, and ephemeris/timing error convolve a point step into a ramp;",
        "- for extended sources this effect is physical, but sparse sampling makes it difficult to distinguish from baseline drift.",
        "",
        f"Central bracketing window used here: +/-{inner_s:.0f} s.",
        "",
        "## Strongest Sampling Caveats",
        "",
    ]
    if summary.empty:
        lines.append("No sampling rows were produced.")
    else:
        work = summary.copy()
        work["sampling_risk_score"] = (
            pd.to_numeric(work["median_bracketing_gap_s"], errors="coerce").fillna(9999) / 300.0
            + (1.0 - pd.to_numeric(work[f"frac_has_samples_both_sides_within_{int(inner_s)}s"], errors="coerce").fillna(0))
        )
        cols = [
            "source_name",
            "frequency_mhz",
            "antenna",
            "event_type",
            "n_predicted_event_windows",
            "median_valid_samples",
            "median_nearest_sample_abs_s",
            "median_bracketing_gap_s",
            f"frac_has_samples_both_sides_within_{int(inner_s)}s",
        ]
        lines.append(_markdown_table(work.sort_values("sampling_risk_score", ascending=False)[cols].head(20)))
    lines.extend(["", "## Template Smear", ""])
    if shape.empty:
        lines.append("No simulated template profiles were produced.")
    else:
        point = shape[np.isclose(shape["transition_s"], 0.0)].copy()
        cols = ["source_name", "frequency_mhz", "antenna", "event_type", "far_side_contrast", "n_near_bins", "n_side_bins"]
        lines.append("Point-step templates sampled at the real times retain the expected sign, but sparse near-event bins can make plotted connections look like slopes:")
        lines.append("")
        lines.append(_markdown_table(point[cols].head(20)))
    lines.extend(["", "## Plots", ""])
    lines.extend(f"- `{p.name}`" for p in paths)
    lines.extend(
        [
            "",
            "## Pipeline Implication",
            "",
            "Do not promote a high-SNR result unless the profile has adequate central bracketing and the source-like contrast survives shape checks. "
            "For extended sources, use stack-first fits with finite-duration templates and report the bracketing gap alongside amplitude-like metrics.",
        ]
    )
    (out_dir / "sampling_smearing_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/sampling_smearing_diagnostics_v1"))
    parser.add_argument("--sources", default="earth,sun,jupiter,fornax_a,cyg_a,cas_a")
    parser.add_argument("--window-s", type=float, default=900.0)
    parser.add_argument("--inner-s", type=float, default=180.0)
    parser.add_argument("--bin-s", type=float, default=60.0)
    parser.add_argument("--transition-grid-s", default="0,120,300,600,900")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    sources = [x.strip().lower() for x in str(args.sources).split(",") if x.strip()]
    transition_grid = [float(x) for x in str(args.transition_grid_s).split(",") if x.strip()]
    write_json(
        out_dir / "run_config.json",
        {
            "sources": sources,
            "window_s": float(args.window_s),
            "inner_s": float(args.inner_s),
            "bin_s": float(args.bin_s),
            "transition_grid_s": transition_grid,
            "software_versions": software_versions(),
        },
    )

    clean = _read(CLEAN, parse_dates=["time"])
    events = pd.concat(
        [
            _read(SUN_EVENTS, parse_dates=["predicted_event_time"]),
            _read(BRIGHT_EVENTS, parse_dates=["predicted_event_time"]),
            _read(PLANET_EVENTS, parse_dates=["predicted_event_time"]),
        ],
        ignore_index=True,
    )

    all_events = []
    all_samples = []
    plot_paths: list[Path] = []
    for source in sources:
        event_metrics, sample_times = collect_sampling(clean, events, source, args.window_s, args.inner_s)
        event_metrics.to_csv(out_dir / f"{source}_event_sampling_metrics.csv", index=False)
        sample_times.to_csv(out_dir / f"{source}_sample_offsets.csv", index=False)
        all_events.append(event_metrics)
        all_samples.append(sample_times)

    event_metrics = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    sample_times = pd.concat(all_samples, ignore_index=True) if all_samples else pd.DataFrame()
    summary = summarize_events(event_metrics, args.inner_s)
    summary.to_csv(out_dir / "sampling_support_summary.csv", index=False)
    templates = simulate_template_profiles(sample_times, args.bin_s, args.window_s, transition_grid)
    templates.to_csv(out_dir / "sampling_template_profiles.csv", index=False)
    shape = summarize_template_shape(templates)
    shape.to_csv(out_dir / "sampling_template_shape_summary.csv", index=False)

    for source in sources:
        path = plot_sampling_summary(summary, source, out_dir)
        if path is not None:
            plot_paths.append(path)
        path = plot_template_smear(templates, source, out_dir)
        if path is not None:
            plot_paths.append(path)
        for antenna in ["rv2_coarse", "rv1_coarse"]:
            path = plot_template_smear_grid(templates, source, antenna, out_dir)
            if path is not None:
                plot_paths.append(path)
    write_report(out_dir, summary, shape, plot_paths, args.inner_s)
    print(out_dir / "sampling_smearing_report.md")


if __name__ == "__main__":
    main()
