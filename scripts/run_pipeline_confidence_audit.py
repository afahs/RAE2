#!/usr/bin/env python
"""End-to-end confidence audit for selected Ryle-Vonberg occultation channels.

The goal is deliberately simple: compare the more complex stack-template fits
against transparent pre/post level contrasts and time-shift controls.
"""

from __future__ import annotations

import argparse
import importlib.util
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

from rylevonberg.util import datetime_ns, ensure_dir, robust_sigma, write_json


CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
ANT_LABEL = {"rv1_coarse": "upper V", "rv2_coarse": "lower V"}


def _read(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return read_table(path, low_memory=False, **kwargs)


def _source_title(source: str) -> str:
    return "Jupiter" if str(source).lower() == "jupiter" else str(source).capitalize()


def _antenna_label(antenna: str) -> str:
    return ANT_LABEL.get(str(antenna), str(antenna))


def _event_window(
    group: pd.DataFrame,
    group_ns: np.ndarray,
    event_time: pd.Timestamp,
    window_s: float,
    time_shift_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    event_ns = pd.Timestamp(event_time).value + int(float(time_shift_s) * 1e9)
    half_ns = int(float(window_s) * 1e9)
    lo = int(np.searchsorted(group_ns, event_ns - half_ns, side="left"))
    hi = int(np.searchsorted(group_ns, event_ns + half_ns, side="right"))
    if hi <= lo:
        return None
    local = group.iloc[lo:hi]
    tr = (datetime_ns(local["time"]) - event_ns).astype(float) / 1e9
    y = pd.to_numeric(local["power"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(y) & (np.abs(tr) <= float(window_s))
    if "is_valid" in local.columns:
        keep &= local["is_valid"].to_numpy(dtype=bool)
    if np.count_nonzero(keep) < 8:
        return None
    tr = tr[keep]
    y = y[keep]
    order = np.argsort(tr)
    return tr[order], y[order]


def _side_masks(t: np.ndarray, inner_s: float, outer_s: float) -> tuple[np.ndarray, np.ndarray]:
    pre = (t >= -float(outer_s)) & (t <= -float(inner_s))
    post = (t >= float(inner_s)) & (t <= float(outer_s))
    return pre, post


def _event_contrast(
    t: np.ndarray,
    y: np.ndarray,
    event_type: str,
    inner_s: float,
    outer_s: float,
    normalize: str,
) -> dict[str, float]:
    pre, post = _side_masks(t, inner_s, outer_s)
    if np.count_nonzero(pre) < 2 or np.count_nonzero(post) < 2:
        return {
            "contrast": np.nan,
            "n_pre": int(np.count_nonzero(pre)),
            "n_post": int(np.count_nonzero(post)),
            "local_scale": np.nan,
        }
    pre_med = float(np.nanmedian(y[pre]))
    post_med = float(np.nanmedian(y[post]))
    side = pre | post
    center = float(np.nanmedian(y[side]))
    scale = robust_sigma(y[side] - center)
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(y[side]))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    if normalize == "zscore":
        denom = scale
    elif normalize == "fractional":
        denom = abs(center) if np.isfinite(center) and abs(center) > 0 else 1.0
    elif normalize == "raw":
        denom = 1.0
    else:
        raise ValueError(f"unknown normalize mode: {normalize}")
    if str(event_type) == "reappearance":
        contrast = (post_med - pre_med) / denom
    else:
        contrast = (pre_med - post_med) / denom
    return {
        "contrast": float(contrast),
        "n_pre": int(np.count_nonzero(pre)),
        "n_post": int(np.count_nonzero(post)),
        "local_scale": float(scale),
    }


def _collect_event_contrasts(
    clean: pd.DataFrame,
    events: pd.DataFrame,
    source_row: pd.Series,
    windows: list[float],
    inner_s: float,
    outer_fraction: float,
    normalize_modes: list[str],
    time_shifts: list[float],
) -> pd.DataFrame:
    source = str(source_row["source_name"])
    band = int(source_row["target_band"])
    antenna = str(source_row["target_antenna"])
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
    rows = []
    for _, ev in sub_events.iterrows():
        for window_s in windows:
            outer_s = min(float(window_s) * float(outer_fraction), float(window_s) - 1.0)
            for shift in time_shifts:
                local = _event_window(group, group_ns, pd.Timestamp(ev["predicted_event_time"]), window_s, shift)
                if local is None:
                    continue
                t, y = local
                for norm in normalize_modes:
                    contrast = _event_contrast(t, y, str(ev["event_type"]), inner_s, outer_s, norm)
                    rows.append(
                        {
                            "source_name": source,
                            "event_id": ev.get("event_id"),
                            "event_type": ev["event_type"],
                            "predicted_event_time": ev["predicted_event_time"],
                            "month_block": pd.Timestamp(ev["predicted_event_time"]).strftime("%Y-%m"),
                            "frequency_band": band,
                            "frequency_mhz": float(source_row["target_frequency_mhz"]),
                            "antenna": antenna,
                            "window_s": float(window_s),
                            "time_shift_s": float(shift),
                            "normalize": norm,
                            **contrast,
                        }
                    )
    return pd.DataFrame.from_records(rows)


def _bootstrap_snr(values: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float, float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    med = float(np.nanmedian(vals))
    sig = robust_sigma(vals)
    robust_se = float(sig / np.sqrt(vals.size)) if np.isfinite(sig) and sig > 0 else np.nan
    robust_snr = float(med / robust_se) if np.isfinite(robust_se) and robust_se > 0 else np.nan
    boots = []
    if vals.size > 2 and n_bootstrap > 0:
        for _ in range(int(n_bootstrap)):
            sample = rng.choice(vals, size=vals.size, replace=True)
            boots.append(float(np.nanmedian(sample)))
    boot_std = float(np.nanstd(boots, ddof=1)) if len(boots) > 1 else np.nan
    boot_snr = float(med / boot_std) if np.isfinite(boot_std) and boot_std > 0 else np.nan
    return med, robust_snr, boot_std, boot_snr


def _summarize_contrasts(events: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    rows = []
    by = ["source_name", "frequency_band", "frequency_mhz", "antenna", "window_s", "time_shift_s", "normalize"]
    for keys, grp in events.groupby(by, dropna=False, sort=True):
        meta = dict(zip(by, keys))
        vals = pd.to_numeric(grp["contrast"], errors="coerce").dropna().to_numpy(dtype=float)
        med, robust_snr, boot_std, boot_snr = _bootstrap_snr(vals, rng, n_bootstrap)
        signs = np.sign(vals)
        n_nonzero = int(np.count_nonzero(signs != 0))
        n_pos = int(np.count_nonzero(signs > 0))
        sign_p = float(binomtest(max(n_pos, n_nonzero - n_pos), n_nonzero, 0.5).pvalue) if n_nonzero else np.nan
        month_medians = grp.groupby("month_block")["contrast"].median()
        rows.append(
            {
                **meta,
                "n_events": int(grp["event_id"].nunique()),
                "median_contrast": med,
                "robust_contrast_snr": robust_snr,
                "event_bootstrap_contrast_std": boot_std,
                "event_bootstrap_contrast_snr": boot_snr,
                "positive_fraction": float(n_pos / n_nonzero) if n_nonzero else np.nan,
                "sign_binomial_p": sign_p,
                "month_median_min": float(np.nanmin(month_medians)) if len(month_medians) else np.nan,
                "month_median_max": float(np.nanmax(month_medians)) if len(month_medians) else np.nan,
                "n_months": int(month_medians.size),
            }
        )
    return pd.DataFrame.from_records(rows)


def _add_time_shift_p(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    out = summary.copy()
    out["time_shift_empirical_p"] = np.nan
    by = ["source_name", "frequency_band", "frequency_mhz", "antenna", "window_s", "normalize"]
    for keys, grp in out.groupby(by, dropna=False, sort=False):
        real = grp[grp["time_shift_s"].eq(0.0)]
        controls = grp[~grp["time_shift_s"].eq(0.0)]
        if real.empty or controls.empty:
            continue
        real_idx = real.index[0]
        real_score = abs(float(real.iloc[0]["event_bootstrap_contrast_snr"]))
        control_scores = pd.to_numeric(controls["event_bootstrap_contrast_snr"], errors="coerce").abs().dropna().to_numpy()
        if control_scores.size:
            out.loc[real_idx, "time_shift_empirical_p"] = (1 + int(np.count_nonzero(control_scores >= real_score))) / (1 + control_scores.size)
    return out


def _plot_summary(source: str, summary: pd.DataFrame, out_dir: Path) -> Path | None:
    sub = summary[summary["source_name"].astype(str).eq(source) & summary["time_shift_s"].eq(0.0)].copy()
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    labels = []
    vals = []
    colors = []
    for _, row in sub.sort_values(["window_s", "normalize"]).iterrows():
        labels.append(f"{int(row['window_s'])}s\n{row['normalize']}")
        vals.append(float(row["event_bootstrap_contrast_snr"]))
        colors.append("#4c78a8" if vals[-1] >= 0 else "#d95f02")
    ax.bar(labels, vals, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(3, color="0.6", lw=0.8, ls="--")
    ax.axhline(-3, color="0.6", lw=0.8, ls="--")
    ax.set_ylabel("Simple pre/post event-bootstrap SNR")
    meta = sub.iloc[0]
    ax.set_title(
        f"{_source_title(source)} simple contrast audit: {float(meta['frequency_mhz']):.2f} MHz {_antenna_label(str(meta['antenna']))}"
    )
    fig.tight_layout()
    path = out_dir / source / f"{source}_simple_contrast_audit.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_report(out_dir: Path, summary: pd.DataFrame, paths: dict[str, Path]) -> None:
    real = summary[summary["time_shift_s"].eq(0.0)].copy()
    real["abs_bootstrap_snr"] = pd.to_numeric(real["event_bootstrap_contrast_snr"], errors="coerce").abs()
    best = real.sort_values(["source_name", "abs_bootstrap_snr"], ascending=[True, False]).groupby("source_name").head(6)
    lines = [
        "# Pipeline Confidence Audit",
        "",
        "This audit re-checks the pipeline with simple statistics before trusting the more complex template fits.",
        "",
        "For each selected channel it computes a per-event signed pre/post contrast:",
        "",
        "- disappearance: pre-event level minus post-event level;",
        "- reappearance: post-event level minus pre-event level.",
        "",
        "Positive contrast is therefore the expected sign for a positive source occultation. The audit then aggregates event contrasts with a median, event bootstrap, sign test, month range, and shifted-time controls.",
        "",
        "## Best Real-Time Simple Contrasts",
        "",
        "```\n"
        + best[
            [
                "source_name",
                "frequency_mhz",
                "antenna",
                "window_s",
                "normalize",
                "n_events",
                "median_contrast",
                "event_bootstrap_contrast_snr",
                "positive_fraction",
                "sign_binomial_p",
                "time_shift_empirical_p",
                "month_median_min",
                "month_median_max",
            ]
        ].to_string(index=False)
        + "\n```",
        "",
        "## Figures",
        "",
    ]
    for source, path in paths.items():
        lines.append(f"- {_source_title(source)}: {path}")
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "Prefer results that are stable across normalization choices, have the expected positive sign, are stronger than shifted-time controls, and do not depend on a single month. A strong template fit with a weak or opposite simple pre/post contrast should be treated as unresolved.",
        ]
    )
    (out_dir / "pipeline_confidence_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_selected_channels(path: Path, sources: list[str]) -> pd.DataFrame:
    table = _read(path)
    table = table[table["source_name"].astype(str).isin(sources)].copy()
    if table.empty:
        raise ValueError(f"No selected channel rows for {sources} in {path}")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", default=str(CLEAN.relative_to(ROOT)))
    parser.add_argument("--selected-channels", default="outputs/fit_inspection_plots_lower_v_fitquality_v1/lower_v_selected_channels.csv")
    parser.add_argument("--output-dir", default="outputs/pipeline_confidence_audit_v1")
    parser.add_argument("--sources", nargs="+", default=["sun", "jupiter"])
    parser.add_argument("--sun-survey-root", default="outputs/planetary_confirmation_survey_sun_earth_excluded_v1")
    parser.add_argument("--default-survey-root", default="outputs/planetary_confirmation_survey_science_baseline_v2")
    parser.add_argument("--windows", nargs="+", type=float, default=[300.0, 600.0, 900.0])
    parser.add_argument("--inner-seconds", type=float, default=60.0)
    parser.add_argument("--outer-fraction", type=float, default=0.8)
    parser.add_argument("--normalize", nargs="+", default=["raw", "fractional", "zscore"])
    parser.add_argument("--time-shifts", nargs="+", type=float, default=[0.0, -7200.0, -3600.0, -1800.0, 1800.0, 3600.0, 7200.0])
    parser.add_argument("--random-time-shifts", type=int, default=0)
    parser.add_argument("--random-shift-min-seconds", type=float, default=1800.0)
    parser.add_argument("--random-shift-max-seconds", type=float, default=21600.0)
    parser.add_argument("--n-bootstrap", type=int, default=512)
    parser.add_argument("--bootstrap-seed", type=int, default=20260514)
    args = parser.parse_args()

    out_dir = ensure_dir(ROOT / args.output_dir)
    clean = _read(ROOT / args.clean, parse_dates=["time"])
    selected = _load_selected_channels(ROOT / args.selected_channels, args.sources)
    time_shifts = list(float(v) for v in args.time_shifts)
    if args.random_time_shifts > 0:
        rng = np.random.default_rng(int(args.bootstrap_seed))
        mags = rng.uniform(float(args.random_shift_min_seconds), float(args.random_shift_max_seconds), int(args.random_time_shifts))
        signs = rng.choice([-1.0, 1.0], size=int(args.random_time_shifts))
        time_shifts = sorted(set([*time_shifts, *[float(m * s) for m, s in zip(mags, signs)]]))
        if 0.0 not in time_shifts:
            time_shifts.insert(0, 0.0)
    all_events = []
    all_contrasts = []
    paths: dict[str, Path] = {}
    for _, row in selected.iterrows():
        source = str(row["source_name"])
        survey_root = ROOT / (args.sun_survey_root if source == "sun" else args.default_survey_root)
        event_file = survey_root / "events" / f"{source}_predicted_events.csv"
        if not event_file.exists():
            event_file = survey_root / "events" / "all_planet_predicted_events.csv"
        events = _read(event_file, parse_dates=["predicted_event_time"])
        events = events[events["source_name"].astype(str).eq(source)].copy()
        events.to_csv(out_dir / f"{source}_audit_input_events.csv", index=False)
        contrasts = _collect_event_contrasts(
            clean,
            events,
            row,
            args.windows,
            args.inner_seconds,
            args.outer_fraction,
            args.normalize,
            time_shifts,
        )
        source_dir = ensure_dir(out_dir / source)
        contrasts.to_csv(source_dir / f"{source}_simple_event_contrasts.csv", index=False)
        summary = _summarize_contrasts(contrasts, args.n_bootstrap, args.bootstrap_seed)
        summary = _add_time_shift_p(summary)
        summary.to_csv(source_dir / f"{source}_simple_contrast_summary.csv", index=False)
        all_contrasts.append(contrasts)
        all_events.append(summary)
        p = _plot_summary(source, summary, out_dir)
        if p is not None:
            paths[source] = p

    contrast_out = pd.concat(all_contrasts, ignore_index=True) if all_contrasts else pd.DataFrame()
    summary_out = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    contrast_out.to_csv(out_dir / "all_simple_event_contrasts.csv", index=False)
    summary_out.to_csv(out_dir / "all_simple_contrast_summary.csv", index=False)
    _write_report(out_dir, summary_out, paths)
    write_json(out_dir / "run_config.json", vars(args))
    print(out_dir / "pipeline_confidence_audit_report.md")
    print(out_dir / "all_simple_contrast_summary.csv")


if __name__ == "__main__":
    main()
