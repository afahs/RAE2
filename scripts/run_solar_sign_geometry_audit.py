#!/usr/bin/env python
"""Audit why lower-V solar contrasts are anti-template.

The audit deliberately stays close to raw pre/post power measurements.  It
tests four explanations:

1. event-labeling/sign convention;
2. timing bias in the predicted event;
3. baseline/sideband definition;
4. channel dependence across lower-V frequencies.
"""

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
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.util import datetime_ns, ensure_dir, robust_sigma, software_versions, write_json  # noqa: E402


CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
SUN_EVENTS = ROOT / "outputs/pipeline_confidence_audit_v2/sun_audit_input_events.csv"
ANT_LABEL = {"rv1_coarse": "upper V", "rv2_coarse": "lower V"}
BASELINE_LABELS = {
    "sideband_wide": "wide sidebands",
    "sideband_near": "near sidebands",
    "full_halves": "full pre/post halves",
    "linear_detrended_sidebands": "linear detrended sidebands",
}


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return read_table(path, low_memory=False)


def _make_groups(clean: pd.DataFrame) -> dict[tuple[int, str], tuple[pd.DataFrame, np.ndarray]]:
    groups = {}
    for (band, antenna), grp in clean.groupby(["frequency_band", "antenna"], sort=True):
        g = grp.sort_values("time").reset_index(drop=True)
        groups[(int(band), str(antenna))] = (g, datetime_ns(g["time"]))
    return groups


def _event_window(
    group: pd.DataFrame,
    group_ns: np.ndarray,
    event_time: pd.Timestamp,
    window_s: float,
    shift_s: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    event_ns = pd.Timestamp(event_time).value + int(float(shift_s) * 1e9)
    half_ns = int(float(window_s) * 1e9)
    lo = int(np.searchsorted(group_ns, event_ns - half_ns, side="left"))
    hi = int(np.searchsorted(group_ns, event_ns + half_ns, side="right"))
    if hi <= lo:
        return None
    local = group.iloc[lo:hi]
    rel = (datetime_ns(local["time"]) - event_ns).astype(float) / 1e9
    y = pd.to_numeric(local["power"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(y) & (np.abs(rel) <= float(window_s))
    if "is_valid" in local.columns:
        keep &= local["is_valid"].to_numpy(dtype=bool)
    if np.count_nonzero(keep) < 8:
        return None
    order = np.argsort(rel[keep])
    return rel[keep][order], y[keep][order]


def _masks(t: np.ndarray, window_s: float, inner_s: float, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mode == "sideband_wide":
        outer = min(float(window_s) * 0.8, float(window_s) - 1.0)
        pre = (t >= -outer) & (t <= -float(inner_s))
        post = (t >= float(inner_s)) & (t <= outer)
    elif mode == "sideband_near":
        outer = min(float(window_s) * 0.35, float(window_s) - 1.0)
        pre = (t >= -outer) & (t <= -float(inner_s))
        post = (t >= float(inner_s)) & (t <= outer)
    elif mode == "full_halves":
        pre = (t < -float(inner_s)) & (t >= -float(window_s))
        post = (t > float(inner_s)) & (t <= float(window_s))
    elif mode == "linear_detrended_sidebands":
        outer = min(float(window_s) * 0.8, float(window_s) - 1.0)
        pre = (t >= -outer) & (t <= -float(inner_s))
        post = (t >= float(inner_s)) & (t <= outer)
    else:
        raise ValueError(f"unknown baseline mode: {mode}")
    return pre, post, pre | post


def _measure(
    t: np.ndarray,
    y: np.ndarray,
    event_type: str,
    window_s: float,
    inner_s: float,
    baseline_mode: str,
) -> dict[str, float]:
    pre, post, side = _masks(t, window_s, inner_s, baseline_mode)
    if np.count_nonzero(pre) < 2 or np.count_nonzero(post) < 2:
        return {
            "pre_median": np.nan,
            "post_median": np.nan,
            "post_minus_pre_raw": np.nan,
            "expected_contrast_z": np.nan,
            "flipped_sign_contrast_z": np.nan,
            "label_swapped_contrast_z": np.nan,
            "n_pre": int(np.count_nonzero(pre)),
            "n_post": int(np.count_nonzero(post)),
            "local_scale": np.nan,
        }

    yy = y.copy()
    if baseline_mode == "linear_detrended_sidebands" and np.count_nonzero(side) >= 4:
        coef = np.polyfit(t[side], yy[side], deg=1)
        baseline = np.polyval(coef, t)
        level = float(np.nanmedian(yy[side]))
        yy = yy - baseline + level

    pre_med = float(np.nanmedian(yy[pre]))
    post_med = float(np.nanmedian(yy[post]))
    delta = post_med - pre_med
    center = float(np.nanmedian(yy[side]))
    scale = robust_sigma(yy[side] - center)
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(yy[side]))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    expected_raw = delta if str(event_type) == "reappearance" else -delta
    expected = expected_raw / scale
    return {
        "pre_median": pre_med,
        "post_median": post_med,
        "post_minus_pre_raw": float(delta),
        "expected_contrast_z": float(expected),
        "flipped_sign_contrast_z": float(-expected),
        "label_swapped_contrast_z": float(-expected),
        "n_pre": int(np.count_nonzero(pre)),
        "n_post": int(np.count_nonzero(post)),
        "local_scale": float(scale),
    }


def _collect(
    clean: pd.DataFrame,
    events: pd.DataFrame,
    combinations: list[tuple[float, float, str]],
    inner_s: float,
    antenna: str,
) -> pd.DataFrame:
    groups = _make_groups(clean)
    work = events[events["antenna"].astype(str).eq(str(antenna))].copy()
    rows = []
    for _, ev in work.iterrows():
        band = int(ev["frequency_band"])
        payload = groups.get((band, antenna))
        if payload is None:
            continue
        group, group_ns = payload
        event_time = pd.Timestamp(ev["predicted_event_time"])
        local_cache: dict[tuple[float, float], tuple[np.ndarray, np.ndarray] | None] = {}
        for window_s, shift_s, baseline_mode in combinations:
            key = (float(window_s), float(shift_s))
            if key not in local_cache:
                local_cache[key] = _event_window(group, group_ns, event_time, window_s, shift_s)
            local = local_cache[key]
            if local is None:
                continue
            t, y = local
            rows.append(
                {
                    "source_name": "sun",
                    "event_id": ev.get("event_id"),
                    "event_type": ev["event_type"],
                    "predicted_event_time": ev["predicted_event_time"],
                    "month_block": event_time.strftime("%Y-%m"),
                    "frequency_band": band,
                    "frequency_mhz": float(ev["frequency_mhz"]),
                    "antenna": antenna,
                    "window_s": float(window_s),
                    "time_shift_s": float(shift_s),
                    "baseline_mode": baseline_mode,
                    "limb_exclusion_nearest_source": ev.get("limb_exclusion_nearest_source", ""),
                    "limb_exclusion_nearest_abs_deg": ev.get("limb_exclusion_nearest_abs_deg", np.nan),
                    **_measure(t, y, str(ev["event_type"]), window_s, inner_s, baseline_mode),
                }
            )
    return pd.DataFrame.from_records(rows)


def _bootstrap_median(values: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    med = float(np.nanmedian(vals))
    boots = []
    for _ in range(int(n_bootstrap)):
        boots.append(float(np.nanmedian(rng.choice(vals, size=vals.size, replace=True))))
    std = float(np.nanstd(boots, ddof=1)) if len(boots) > 1 else np.nan
    snr = float(med / std) if np.isfinite(std) and std > 0 else np.nan
    return med, std, snr


def _summarize(df: pd.DataFrame, value_col: str, group_cols: list[str], seed: int, n_bootstrap: int) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    rows = []
    for keys, grp in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(group_cols, keys))
        vals = pd.to_numeric(grp[value_col], errors="coerce").dropna().to_numpy(dtype=float)
        median, boot_std, boot_snr = _bootstrap_median(vals, rng, n_bootstrap)
        signs = np.sign(vals)
        n_nonzero = int(np.count_nonzero(signs != 0))
        n_pos = int(np.count_nonzero(signs > 0))
        sign_p = float(binomtest(max(n_pos, n_nonzero - n_pos), n_nonzero, 0.5).pvalue) if n_nonzero else np.nan
        month_medians = grp.groupby("month_block")[value_col].median()
        rows.append(
            {
                **meta,
                "value_col": value_col,
                "n_rows": int(len(grp)),
                "n_events": int(grp["event_id"].nunique()),
                "median": median,
                "bootstrap_std": boot_std,
                "bootstrap_snr": boot_snr,
                "positive_fraction": float(n_pos / n_nonzero) if n_nonzero else np.nan,
                "sign_binomial_p": sign_p,
                "n_months": int(month_medians.size),
                "month_median_min": float(np.nanmin(month_medians)) if len(month_medians) else np.nan,
                "month_median_max": float(np.nanmax(month_medians)) if len(month_medians) else np.nan,
            }
        )
    return pd.DataFrame.from_records(rows)


def _plot_sign_mode_matrix(summary: pd.DataFrame, out_dir: Path) -> Path:
    sub = summary[
        summary["window_s"].eq(300.0)
        & summary["time_shift_s"].eq(0.0)
        & summary["frequency_band"].eq(8)
    ].copy()
    value_cols = ["expected_contrast_z", "flipped_sign_contrast_z", "label_swapped_contrast_z"]
    rows = []
    for value_col in value_cols:
        x = sub[sub["value_col"].eq(value_col)]
        for _, row in x.iterrows():
            rows.append(
                {
                    "test": value_col.replace("_contrast_z", ""),
                    "baseline_mode": row["baseline_mode"],
                    "bootstrap_snr": row["bootstrap_snr"],
                }
            )
    mat = pd.DataFrame(rows).pivot(index="baseline_mode", columns="test", values="bootstrap_snr")
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    data = mat.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="RdBu_r", vmin=-max(5, np.nanmax(np.abs(data))), vmax=max(5, np.nanmax(np.abs(data))))
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=20, ha="right")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels([BASELINE_LABELS.get(x, x) for x in mat.index])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = data[i, j]
            ax.text(j, i, f"{val:.1f}" if np.isfinite(val) else "nan", ha="center", va="center", color="black")
    ax.set_title("Sun lower V 6.55 MHz: sign/label vs baseline test SNR")
    fig.colorbar(im, ax=ax, label="bootstrap SNR of median contrast")
    fig.tight_layout()
    path = out_dir / "solar_sign_baseline_matrix_band8_lower_v.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_timing(summary: pd.DataFrame, out_dir: Path) -> Path:
    sub = summary[
        summary["value_col"].eq("expected_contrast_z")
        & summary["window_s"].eq(300.0)
        & summary["baseline_mode"].eq("sideband_wide")
    ].copy()
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for band, grp in sub.groupby("frequency_band", sort=True):
        grp = grp.sort_values("time_shift_s")
        freq = float(grp["frequency_mhz"].iloc[0])
        lw = 2.2 if band == 8 else 1.2
        alpha = 1.0 if band == 8 else 0.55
        ax.plot(grp["time_shift_s"], grp["bootstrap_snr"], marker="o", lw=lw, alpha=alpha, label=f"{freq:.2f} MHz")
    ax.axhline(0, color="black", lw=0.9)
    ax.axvline(0, color="black", lw=0.9, ls=":")
    ax.set_xlabel("Timing shift applied to predicted event (s)")
    ax.set_ylabel("Expected-sign bootstrap SNR")
    ax.set_title("Sun lower V timing-bias audit across channels")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    path = out_dir / "solar_lower_v_timing_bias_audit.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_baseline_spectrum(summary: pd.DataFrame, out_dir: Path) -> Path:
    sub = summary[
        summary["value_col"].eq("expected_contrast_z")
        & summary["window_s"].eq(300.0)
        & summary["time_shift_s"].eq(0.0)
    ].copy()
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for mode, grp in sub.groupby("baseline_mode", sort=True):
        grp = grp.sort_values("frequency_mhz")
        ax.plot(
            grp["frequency_mhz"],
            grp["bootstrap_snr"],
            marker="o",
            lw=1.7,
            label=BASELINE_LABELS.get(str(mode), str(mode)),
        )
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Expected-sign bootstrap SNR")
    ax.set_title("Sun lower V baseline-definition audit")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "solar_lower_v_baseline_definition_spectrum.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_raw_delta_by_event_type(raw: pd.DataFrame, out_dir: Path) -> Path:
    sub = raw[
        raw["window_s"].eq(300.0)
        & raw["time_shift_s"].eq(0.0)
        & raw["baseline_mode"].eq("sideband_wide")
    ].copy()
    rows = []
    for keys, grp in sub.groupby(["frequency_mhz", "event_type"], sort=True):
        freq, event_type = keys
        rows.append({"frequency_mhz": freq, "event_type": event_type, "median_delta": float(np.nanmedian(grp["post_minus_pre_raw"]))})
    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for event_type, grp in data.groupby("event_type", sort=True):
        ax.plot(grp["frequency_mhz"], grp["median_delta"], marker="o", lw=1.8, label=event_type)
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Median raw post - pre power")
    ax.set_title("Sun lower V raw power delta by event label")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "solar_lower_v_raw_delta_by_event_type.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_report(out_dir: Path, summary: pd.DataFrame, paths: list[Path], config: dict[str, object]) -> None:
    primary = summary[
        summary["value_col"].eq("expected_contrast_z")
        & summary["frequency_band"].eq(8)
        & summary["window_s"].eq(300.0)
        & summary["time_shift_s"].eq(0.0)
        & summary["baseline_mode"].eq("sideband_wide")
    ]
    flipped = summary[
        summary["value_col"].eq("flipped_sign_contrast_z")
        & summary["frequency_band"].eq(8)
        & summary["window_s"].eq(300.0)
        & summary["time_shift_s"].eq(0.0)
        & summary["baseline_mode"].eq("sideband_wide")
    ]
    timing = summary[
        summary["value_col"].eq("expected_contrast_z")
        & summary["frequency_band"].eq(8)
        & summary["window_s"].eq(300.0)
        & summary["baseline_mode"].eq("sideband_wide")
    ].sort_values("bootstrap_snr", ascending=False)
    best_timing = timing.iloc[0] if len(timing) else None
    baseline = summary[
        summary["value_col"].eq("expected_contrast_z")
        & summary["frequency_band"].eq(8)
        & summary["window_s"].eq(300.0)
        & summary["time_shift_s"].eq(0.0)
    ].sort_values("bootstrap_snr", ascending=False)
    best_baseline = baseline.iloc[0] if len(baseline) else None

    lines = [
        "# Solar Sign And Geometry Audit",
        "",
        "This audit tests whether the lower-V solar anti-template behavior is caused by event labeling, sign convention, timing bias, or baseline definition.",
        "",
        "The measurement starts from raw pre/post medians in each event window. The expected signed contrast is:",
        "",
        "- disappearance: `(pre - post) / local_scale`",
        "- reappearance: `(post - pre) / local_scale`",
        "",
        "Positive values are source-like under the current occultation convention.",
        "",
        "## Configuration",
        "",
        f"- antenna: {ANT_LABEL.get(str(config['antenna']), str(config['antenna']))}",
        f"- windows: {config['windows_s']}",
        f"- timing shifts: {config['time_shifts_s']}",
        f"- baseline modes: {config['baseline_modes']}",
        f"- inner exclusion around event: {config['inner_s']} s",
        "",
        "## Key Results",
        "",
    ]
    if len(primary):
        row = primary.iloc[0]
        lines.append(
            f"- Current convention, 6.55 MHz lower V, 300 s, zero shift, wide sidebands: "
            f"median {row['median']:.3f}, bootstrap SNR {row['bootstrap_snr']:.2f}, "
            f"positive fraction {row['positive_fraction']:.3f}."
        )
    if len(flipped):
        row = flipped.iloc[0]
        lines.append(
            f"- Flipping sign or swapping disappearance/reappearance at the same channel gives "
            f"median {row['median']:.3f}, bootstrap SNR {row['bootstrap_snr']:.2f}. "
            "This is expected because both operations reverse the signed contrast."
        )
    if best_timing is not None:
        lines.append(
            f"- Best 6.55 MHz lower-V timing-shift result under the current sign is at "
            f"{best_timing['time_shift_s']:.0f} s with bootstrap SNR {best_timing['bootstrap_snr']:.2f}."
        )
    if best_baseline is not None:
        lines.append(
            f"- Best 6.55 MHz lower-V zero-shift baseline definition under the current sign is "
            f"`{best_baseline['baseline_mode']}` with bootstrap SNR {best_baseline['bootstrap_snr']:.2f}."
        )
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- If flipped/sign-swapped versions are positive while the expected sign is negative, the data are anti-template under the current geometry/sign convention, but this alone does not prove the labels are wrong.",
            "- If one timing shift near zero turns the expected sign positive, timing bias is plausible.",
            "- If a baseline mode turns the expected sign positive while raw deltas remain label-inconsistent, baseline subtraction is the likely driver.",
            "- If all baseline modes and nearby timing shifts remain negative in lower V, the anti-template behavior is not explained by these implementation choices.",
            "",
            "## Generated Diagnostic Plots",
            "",
        ]
    )
    lines.extend(f"- `{path.name}`" for path in paths)
    lines.extend(
        [
            "",
            "## Recommended Pipeline Options",
            "",
            "1. Keep the raw pre/post delta table as a required diagnostic for any solar claim.",
            "2. Treat flipped-sign solar results as a debugging flag, not as a detection, unless an independent geometry audit justifies the sign change.",
            "3. Prefer lower-V-only solar decision metrics; upper-V channels remain controls.",
            "4. Add a timing-bias tolerance only if the timing scan shows a local maximum near zero and not only at large artificial shifts.",
            "5. Require baseline robustness: a solar candidate should keep the same sign across wide sidebands, near sidebands, full halves, and linear-detrended sidebands.",
        ]
    )
    (out_dir / "solar_sign_geometry_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/solar_sign_geometry_audit_v1"))
    parser.add_argument("--antenna", default="rv2_coarse")
    parser.add_argument("--bands", default="8", help="Comma-separated frequency bands to audit; default is the selected solar 6.55 MHz band.")
    parser.add_argument("--windows-s", default="300")
    parser.add_argument("--time-shifts-s", default="-300,-180,-120,-60,-30,0,30,60,120,180,300")
    parser.add_argument("--baseline-modes", default="sideband_wide,sideband_near,full_halves,linear_detrended_sidebands")
    parser.add_argument("--inner-s", type=float, default=15.0)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260518)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    windows = [float(x) for x in str(args.windows_s).split(",") if x]
    shifts = [float(x) for x in str(args.time_shifts_s).split(",") if x]
    baseline_modes = [str(x) for x in str(args.baseline_modes).split(",") if x]
    bands = [int(x) for x in str(args.bands).split(",") if x]
    combinations = sorted(
        {
            (300.0, float(shift), "sideband_wide")
            for shift in shifts
        }
        | {
            (float(window), 0.0, str(mode))
            for window in windows
            for mode in baseline_modes
        }
    )
    config = {
        "cleaned_timeseries": str(CLEAN),
        "sun_events": str(SUN_EVENTS),
        "antenna": str(args.antenna),
        "bands": bands,
        "windows_s": windows,
        "time_shifts_s": shifts,
        "baseline_modes": baseline_modes,
        "measurement_combinations": combinations,
        "inner_s": float(args.inner_s),
        "n_bootstrap": int(args.n_bootstrap),
        "seed": int(args.seed),
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)

    clean = _read(CLEAN)
    events = _read(SUN_EVENTS)
    events = events[events["source_name"].astype(str).str.lower().eq("sun")].copy()
    events = events[events["frequency_band"].astype(int).isin(bands)].copy()

    raw = _collect(clean, events, combinations, args.inner_s, args.antenna)
    raw.to_csv(out_dir / "solar_lower_v_raw_prepost_audit.csv", index=False)

    group_cols = ["frequency_band", "frequency_mhz", "antenna", "window_s", "time_shift_s", "baseline_mode"]
    summaries = []
    for value_col in ["expected_contrast_z", "flipped_sign_contrast_z", "label_swapped_contrast_z"]:
        summaries.append(_summarize(raw, value_col, group_cols, args.seed, args.n_bootstrap))
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(out_dir / "solar_lower_v_sign_geometry_summary.csv", index=False)

    event_type_summary = _summarize(
        raw,
        "expected_contrast_z",
        group_cols + ["event_type"],
        args.seed + 1,
        args.n_bootstrap,
    )
    event_type_summary.to_csv(out_dir / "solar_lower_v_event_type_summary.csv", index=False)

    paths = [
        _plot_sign_mode_matrix(summary, out_dir),
        _plot_timing(summary, out_dir),
        _plot_baseline_spectrum(summary, out_dir),
        _plot_raw_delta_by_event_type(raw, out_dir),
    ]
    _write_report(out_dir, summary, paths, config)
    print(f"Wrote solar sign/geometry audit to {out_dir}")
    print(summary[
        summary["frequency_band"].eq(8)
        & summary["window_s"].eq(300.0)
        & summary["time_shift_s"].eq(0.0)
    ].sort_values(["value_col", "baseline_mode"]).to_string(index=False))


if __name__ == "__main__":
    main()
