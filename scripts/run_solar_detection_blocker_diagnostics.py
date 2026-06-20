#!/usr/bin/env python
"""Characterize blockers preventing a clean solar occultation detection."""

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

from rylevonberg.util import datetime_ns, ensure_dir, robust_sigma, write_json


CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
ANT_LABEL = {"rv1_coarse": "upper V", "rv2_coarse": "lower V"}
ANT_COLORS = {"rv1_coarse": "#4c78a8", "rv2_coarse": "#d95f02"}
EVENT_COLORS = {"disappearance": "#4c78a8", "reappearance": "#d95f02"}


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


def _event_window(
    group: pd.DataFrame,
    group_ns: np.ndarray,
    event_time: pd.Timestamp,
    window_s: float,
    shift_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    event_ns = pd.Timestamp(event_time).value + int(float(shift_s) * 1e9)
    half_ns = int(float(window_s) * 1e9)
    lo = int(np.searchsorted(group_ns, event_ns - half_ns, side="left"))
    hi = int(np.searchsorted(group_ns, event_ns + half_ns, side="right"))
    if hi <= lo:
        return None
    local = group.iloc[lo:hi]
    rel = (datetime_ns(local["time"]) - event_ns).astype(float) / 1e9
    power = pd.to_numeric(local["power"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(power) & (np.abs(rel) <= float(window_s))
    if "is_valid" in local.columns:
        keep &= local["is_valid"].to_numpy(dtype=bool)
    if np.count_nonzero(keep) < 8:
        return None
    order = np.argsort(rel[keep])
    return rel[keep][order], power[keep][order]


def _contrast(t: np.ndarray, y: np.ndarray, event_type: str, inner_s: float, outer_s: float, normalize: str) -> dict[str, float]:
    pre = (t >= -float(outer_s)) & (t <= -float(inner_s))
    post = (t >= float(inner_s)) & (t <= float(outer_s))
    if np.count_nonzero(pre) < 2 or np.count_nonzero(post) < 2:
        return {"contrast": np.nan, "n_pre": int(np.count_nonzero(pre)), "n_post": int(np.count_nonzero(post))}
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
        raise ValueError(normalize)
    value = (post_med - pre_med) if str(event_type) == "reappearance" else (pre_med - post_med)
    return {"contrast": float(value / denom), "n_pre": int(np.count_nonzero(pre)), "n_post": int(np.count_nonzero(post))}


def _collect(
    clean: pd.DataFrame,
    events: pd.DataFrame,
    windows: list[float],
    normalizes: list[str],
    shifts: list[float],
    inner_s: float,
    outer_fraction: float,
    restrict_band: int | None = None,
    restrict_antenna: str | None = None,
) -> pd.DataFrame:
    groups = _make_groups(clean)
    rows = []
    work = events.copy()
    if restrict_band is not None:
        work = work[work["frequency_band"].astype(int).eq(int(restrict_band))]
    if restrict_antenna is not None:
        work = work[work["antenna"].astype(str).eq(str(restrict_antenna))]
    for _, ev in work.iterrows():
        band = int(ev["frequency_band"])
        antenna = str(ev["antenna"])
        payload = groups.get((band, antenna))
        if payload is None:
            continue
        group, group_ns = payload
        for window_s in windows:
            outer_s = min(float(window_s) * float(outer_fraction), float(window_s) - 1.0)
            for shift_s in shifts:
                local = _event_window(group, group_ns, pd.Timestamp(ev["predicted_event_time"]), window_s, shift_s)
                if local is None:
                    continue
                t, y = local
                for normalize in normalizes:
                    result = _contrast(t, y, str(ev["event_type"]), inner_s, outer_s, normalize)
                    rows.append(
                        {
                            "source_name": "sun",
                            "event_id": ev.get("event_id"),
                            "event_type": ev["event_type"],
                            "predicted_event_time": ev["predicted_event_time"],
                            "month_block": pd.Timestamp(ev["predicted_event_time"]).strftime("%Y-%m"),
                            "frequency_band": band,
                            "frequency_mhz": float(ev["frequency_mhz"]),
                            "antenna": antenna,
                            "window_s": float(window_s),
                            "time_shift_s": float(shift_s),
                            "normalize": normalize,
                            **result,
                        }
                    )
    return pd.DataFrame.from_records(rows)


def _bootstrap_snr(values: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    med = float(np.nanmedian(vals))
    boots = []
    for _ in range(int(n_bootstrap)):
        sample = rng.choice(vals, size=vals.size, replace=True)
        boots.append(float(np.nanmedian(sample)))
    std = float(np.nanstd(boots, ddof=1)) if len(boots) > 1 else np.nan
    snr = float(med / std) if np.isfinite(std) and std > 0 else np.nan
    return med, std, snr


def _summarize(df: pd.DataFrame, by: list[str], n_bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    rows = []
    for keys, grp in df.groupby(by, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(by, keys))
        vals = pd.to_numeric(grp["contrast"], errors="coerce").dropna().to_numpy(dtype=float)
        med, std, snr = _bootstrap_snr(vals, rng, n_bootstrap)
        signs = np.sign(vals)
        n_nonzero = int(np.count_nonzero(signs != 0))
        n_pos = int(np.count_nonzero(signs > 0))
        sign_p = float(binomtest(max(n_pos, n_nonzero - n_pos), n_nonzero, 0.5).pvalue) if n_nonzero else np.nan
        month_med = grp.groupby("month_block")["contrast"].median()
        rows.append(
            {
                **meta,
                "n_events": int(grp["event_id"].nunique()),
                "median_contrast": med,
                "bootstrap_std": std,
                "bootstrap_snr": snr,
                "positive_fraction": float(n_pos / n_nonzero) if n_nonzero else np.nan,
                "sign_binomial_p": sign_p,
                "month_median_min": float(np.nanmin(month_med)) if len(month_med) else np.nan,
                "month_median_max": float(np.nanmax(month_med)) if len(month_med) else np.nan,
                "n_months": int(month_med.size),
            }
        )
    return pd.DataFrame.from_records(rows)


def _plot_spectrum(summary: pd.DataFrame, out_dir: Path) -> Path:
    sub = summary[
        summary["time_shift_s"].eq(0.0)
        & summary["window_s"].eq(300.0)
        & summary["normalize"].eq("zscore")
    ].copy()
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for antenna, grp in sub.groupby("antenna", sort=True):
        grp = grp.sort_values("frequency_mhz")
        ax.plot(
            grp["frequency_mhz"],
            grp["bootstrap_snr"],
            marker="o",
            lw=1.8,
            color=ANT_COLORS.get(str(antenna)),
            label=ANT_LABEL.get(str(antenna), str(antenna)),
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(3, color="0.65", lw=0.8, ls="--")
    ax.axhline(-3, color="0.65", lw=0.8, ls="--")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Simple pre/post bootstrap SNR")
    ax.set_title("Sun Earth-limb-excluded all-channel contrast spectrum")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "sun_all_channel_simple_contrast_spectrum.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_timing_scan(timing: pd.DataFrame, out_dir: Path) -> Path:
    sub = timing[timing["normalize"].eq("zscore")].copy()
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for window_s, grp in sub.groupby("window_s", sort=True):
        grp = grp.sort_values("time_shift_s")
        ax.plot(grp["time_shift_s"], grp["bootstrap_snr"], marker="o", lw=1.5, label=f"{int(window_s)} s")
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(3, color="0.65", lw=0.8, ls="--")
    ax.axhline(-3, color="0.65", lw=0.8, ls="--")
    ax.axvline(0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("Artificial timing shift applied to predicted Sun events (s)")
    ax.set_ylabel("Simple pre/post bootstrap SNR")
    ax.set_title("Sun 6.55 MHz lower V timing-shift scan")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "sun_selected_channel_timing_scan.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_event_type(event_type_summary: pd.DataFrame, out_dir: Path) -> Path:
    sub = event_type_summary[
        event_type_summary["time_shift_s"].eq(0.0)
        & event_type_summary["window_s"].eq(300.0)
        & event_type_summary["normalize"].eq("zscore")
    ].copy()
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    labels = []
    vals = []
    colors = []
    for _, row in sub.sort_values(["frequency_mhz", "antenna", "event_type"]).iterrows():
        labels.append(f"{row['frequency_mhz']:.2g}\n{ANT_LABEL.get(row['antenna'], row['antenna'])}\n{row['event_type'][:4]}")
        vals.append(float(row["bootstrap_snr"]))
        colors.append(EVENT_COLORS.get(str(row["event_type"]), "0.5"))
    ax.bar(range(len(vals)), vals, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(3, color="0.65", lw=0.8, ls="--")
    ax.axhline(-3, color="0.65", lw=0.8, ls="--")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("Bootstrap SNR")
    ax.set_title("Sun event-type split by channel")
    fig.tight_layout()
    path = out_dir / "sun_event_type_channel_split.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _strong_rows(table: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if table.empty:
        return table
    out = table.copy()
    out["abs_snr"] = pd.to_numeric(out["bootstrap_snr"], errors="coerce").abs()
    return out.sort_values("abs_snr", ascending=False).head(n).drop(columns=["abs_snr"])


def _write_report(
    out_dir: Path,
    channel_summary: pd.DataFrame,
    event_type_summary: pd.DataFrame,
    timing_summary: pd.DataFrame,
    veto_counts: pd.DataFrame,
    figure_paths: list[Path],
) -> None:
    channel_focus = channel_summary[
        channel_summary["time_shift_s"].eq(0.0)
        & channel_summary["window_s"].eq(300.0)
        & channel_summary["normalize"].eq("zscore")
    ].copy()
    timing_focus = timing_summary[timing_summary["normalize"].eq("zscore")].copy()
    lines = [
        "# Solar Detection Blocker Diagnostics",
        "",
        "Question: what is preventing a clean positive solar occultation detection?",
        "",
        "The tests here use the Earth-limb-excluded Sun event list. Positive SNR means disappearance drops and reappearance rises, matching a positive-source occultation. Negative SNR means anti-template behavior.",
        "",
        "## Blocker Summary",
        "",
        "1. Earth-limb contamination was tested previously and is not enough: removing events with Earth within 3 degrees of the lunar limb leaves the Sun anti-template behavior.",
        "2. The lower-V 6.55 MHz channel remains negative in both simple pre/post contrasts and stack-first fits.",
        "3. The all-channel scan tests whether the problem is one bad channel or a broader Sun/antenna behavior.",
        "4. The timing scan tests whether the solar response is simply offset from the predicted point-source limb time.",
        "5. The event-type split tests whether disappearance or reappearance is driving the anti-template result.",
        "",
        "## Earth-Limb Veto Counts",
        "",
        "```\n" + (veto_counts.to_string(index=False) if not veto_counts.empty else "No veto-count table found.") + "\n```",
        "",
        "## Strongest All-Channel Simple Contrasts",
        "",
        "```\n"
        + _strong_rows(
            channel_focus[
                [
                    "frequency_band",
                    "frequency_mhz",
                    "antenna",
                    "n_events",
                    "median_contrast",
                    "bootstrap_snr",
                    "positive_fraction",
                    "month_median_min",
                    "month_median_max",
                ]
            ],
            18,
        ).to_string(index=False)
        + "\n```",
        "",
        "## Selected-Channel Timing Scan",
        "",
        "```\n"
        + _strong_rows(
            timing_focus[
                [
                    "window_s",
                    "time_shift_s",
                    "n_events",
                    "median_contrast",
                    "bootstrap_snr",
                    "positive_fraction",
                    "month_median_min",
                    "month_median_max",
                ]
            ],
            12,
        ).to_string(index=False)
        + "\n```",
        "",
        "## Event-Type Split",
        "",
        "```\n"
        + _strong_rows(
            event_type_summary[
                event_type_summary["time_shift_s"].eq(0.0)
                & event_type_summary["window_s"].eq(300.0)
                & event_type_summary["normalize"].eq("zscore")
            ][
                [
                    "frequency_band",
                    "frequency_mhz",
                    "antenna",
                    "event_type",
                    "n_events",
                    "median_contrast",
                    "bootstrap_snr",
                    "positive_fraction",
                ]
            ],
            20,
        ).to_string(index=False)
        + "\n```",
        "",
        "## Figures",
        "",
        *[f"- {path}" for path in figure_paths],
        "",
        "## Current Interpretation",
        "",
        "A proper solar detection is currently blocked by sign/morphology, not just statistical power. The pipeline sees a repeatable Sun-associated response, but the response has the wrong sign under the Earth-validated positive-source convention. This makes the next science task a physical/systematic interpretation problem: antenna response, finite solar disk/extended emission, remaining contaminating geometry, or channel calibration.",
    ]
    (out_dir / "solar_detection_blocker_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", default=str(CLEAN.relative_to(ROOT)))
    parser.add_argument("--events", default="outputs/planetary_confirmation_survey_sun_earth_excluded_v1/events/sun_predicted_events.csv")
    parser.add_argument("--veto-counts", default="outputs/sun_earth_limb_exclusion_audit_v1/earth_limb_exclusion_event_counts.csv")
    parser.add_argument("--output-dir", default="outputs/solar_detection_blocker_diagnostics_v1")
    parser.add_argument("--windows", nargs="+", type=float, default=[300.0, 600.0, 900.0])
    parser.add_argument("--normalizes", nargs="+", default=["zscore", "fractional"])
    parser.add_argument("--timing-shifts", nargs="+", type=float, default=[-600, -480, -360, -240, -180, -120, -60, 0, 60, 90, 120, 180, 240, 360, 480, 600])
    parser.add_argument("--inner-seconds", type=float, default=60.0)
    parser.add_argument("--outer-fraction", type=float, default=0.8)
    parser.add_argument("--n-bootstrap", type=int, default=256)
    parser.add_argument("--bootstrap-seed", type=int, default=20260514)
    args = parser.parse_args()

    out_dir = ensure_dir(ROOT / args.output_dir)
    clean = _read(ROOT / args.clean, parse_dates=["time"])
    events = _read(ROOT / args.events, parse_dates=["predicted_event_time"])
    events = events[events["source_name"].astype(str).eq("sun")].copy()

    all_channel = _collect(
        clean,
        events,
        windows=args.windows,
        normalizes=args.normalizes,
        shifts=[0.0],
        inner_s=args.inner_seconds,
        outer_fraction=args.outer_fraction,
    )
    all_channel.to_csv(out_dir / "sun_all_channel_event_contrasts.csv", index=False)
    channel_summary = _summarize(
        all_channel,
        ["frequency_band", "frequency_mhz", "antenna", "window_s", "time_shift_s", "normalize"],
        args.n_bootstrap,
        args.bootstrap_seed,
    )
    channel_summary.to_csv(out_dir / "sun_all_channel_contrast_summary.csv", index=False)
    event_type_summary = _summarize(
        all_channel,
        ["frequency_band", "frequency_mhz", "antenna", "window_s", "time_shift_s", "normalize", "event_type"],
        args.n_bootstrap,
        args.bootstrap_seed + 1,
    )
    event_type_summary.to_csv(out_dir / "sun_event_type_contrast_summary.csv", index=False)

    timing = _collect(
        clean,
        events,
        windows=args.windows,
        normalizes=args.normalizes,
        shifts=args.timing_shifts,
        inner_s=args.inner_seconds,
        outer_fraction=args.outer_fraction,
        restrict_band=8,
        restrict_antenna="rv2_coarse",
    )
    timing.to_csv(out_dir / "sun_selected_channel_timing_event_contrasts.csv", index=False)
    timing_summary = _summarize(
        timing,
        ["frequency_band", "frequency_mhz", "antenna", "window_s", "time_shift_s", "normalize"],
        args.n_bootstrap,
        args.bootstrap_seed + 2,
    )
    timing_summary.to_csv(out_dir / "sun_selected_channel_timing_summary.csv", index=False)

    veto_counts = _read(ROOT / args.veto_counts)
    if not veto_counts.empty:
        veto_counts = veto_counts[
            veto_counts["frequency_band"].astype(str).isin(["all", "8"])
            & veto_counts["antenna"].astype(str).isin(["all", "rv2_coarse"])
        ].copy()
    figures = [
        _plot_spectrum(channel_summary, out_dir),
        _plot_timing_scan(timing_summary, out_dir),
        _plot_event_type(event_type_summary, out_dir),
    ]
    _write_report(out_dir, channel_summary, event_type_summary, timing_summary, veto_counts, figures)
    write_json(out_dir / "run_config.json", vars(args))
    print(out_dir / "solar_detection_blocker_diagnostics.md")
    print(out_dir / "sun_all_channel_contrast_summary.csv")


if __name__ == "__main__":
    main()
