#!/usr/bin/env python
"""Fast no-trendline channel screen with drift-sensitivity labels."""

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
EARTH_EVENTS = ROOT / "outputs/planetary_confirmation_survey_science_baseline_v2/events/all_planet_predicted_events.csv"
ANT_LABEL = {"rv1_coarse": "upper V", "rv2_coarse": "lower V"}


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
) -> tuple[np.ndarray, np.ndarray] | None:
    event_ns = pd.Timestamp(event_time).value
    half_ns = int(float(window_s) * 1e9)
    lo = int(np.searchsorted(group_ns, event_ns - half_ns, side="left"))
    hi = int(np.searchsorted(group_ns, event_ns + half_ns, side="right"))
    if hi <= lo:
        return None
    local = group.iloc[lo:hi]
    t = (datetime_ns(local["time"]) - event_ns).astype(float) / 1e9
    y = pd.to_numeric(local["power"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y) & (np.abs(t) <= float(window_s))
    if "is_valid" in local.columns:
        valid &= local["is_valid"].to_numpy(dtype=bool)
    if np.count_nonzero(valid) < 8:
        return None
    order = np.argsort(t[valid])
    return t[valid][order], y[valid][order]


def _signed_delta(pre_med: float, post_med: float, event_type: str) -> float:
    return float(post_med - pre_med) if str(event_type) == "reappearance" else float(pre_med - post_med)


def _constant_template_amp(t: np.ndarray, y: np.ndarray, event_type: str) -> float:
    tmpl = event_template(t, event_type, timing_offset_sec=0.0)
    X = np.column_stack([np.ones(len(y)), tmpl])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[-1])


def _measure_event(t: np.ndarray, y: np.ndarray, event_type: str, window_s: float, inner_s: float) -> dict[str, float]:
    pre = t <= -float(inner_s)
    post = t >= float(inner_s)
    inner_pre = (t >= -3 * float(inner_s)) & (t <= -float(inner_s))
    inner_post = (t >= float(inner_s)) & (t <= 3 * float(inner_s))
    side = pre | post
    out = {
        "prepost_amp": np.nan,
        "inner_amp": np.nan,
        "detrended_prepost_amp": np.nan,
        "constant_template_amp": np.nan,
        "n_pre": int(np.count_nonzero(pre)),
        "n_post": int(np.count_nonzero(post)),
        "n_inner_pre": int(np.count_nonzero(inner_pre)),
        "n_inner_post": int(np.count_nonzero(inner_post)),
        "local_sigma": np.nan,
        "linear_drift_window": np.nan,
    }
    if np.count_nonzero(pre) >= 2 and np.count_nonzero(post) >= 2:
        pre_med = float(np.nanmedian(y[pre]))
        post_med = float(np.nanmedian(y[post]))
        out["prepost_amp"] = _signed_delta(pre_med, post_med, event_type)
    if np.count_nonzero(inner_pre) >= 2 and np.count_nonzero(inner_post) >= 2:
        pre_med = float(np.nanmedian(y[inner_pre]))
        post_med = float(np.nanmedian(y[inner_post]))
        out["inner_amp"] = _signed_delta(pre_med, post_med, event_type)
    if np.count_nonzero(side) >= 6:
        center = float(np.nanmedian(y[side]))
        sigma = robust_sigma(y[side] - center)
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = float(np.nanstd(y[side]))
        out["local_sigma"] = float(sigma) if np.isfinite(sigma) and sigma > 0 else np.nan
        coef = np.polyfit(t[side], y[side], 1)
        baseline = np.polyval(coef, t)
        yd = y - baseline + float(np.nanmedian(y[side]))
        out["linear_drift_window"] = float(coef[0] * (2 * float(window_s)))
        if np.count_nonzero(pre) >= 2 and np.count_nonzero(post) >= 2:
            out["detrended_prepost_amp"] = _signed_delta(float(np.nanmedian(yd[pre])), float(np.nanmedian(yd[post])), event_type)
    try:
        out["constant_template_amp"] = _constant_template_amp(t, y, event_type)
    except Exception:
        pass
    return out


def _collect(
    clean: pd.DataFrame,
    events: pd.DataFrame,
    sources: set[str],
    windows: list[float],
    inner_s: float,
) -> pd.DataFrame:
    groups = _make_groups(clean)
    work = events[events["source_name"].astype(str).str.lower().isin(sources)].copy()
    rows = []
    for _, ev in work.iterrows():
        band = int(ev["frequency_band"])
        antenna = str(ev["antenna"])
        payload = groups.get((band, antenna))
        if payload is None:
            continue
        group, group_ns = payload
        for window_s in windows:
            local = _event_window(group, group_ns, pd.Timestamp(ev["predicted_event_time"]), window_s)
            if local is None:
                continue
            t, y = local
            rows.append(
                {
                    "source_name": str(ev["source_name"]).lower(),
                    "event_id": ev.get("event_id"),
                    "event_type": ev.get("event_type"),
                    "predicted_event_time": ev.get("predicted_event_time"),
                    "month_block": pd.Timestamp(ev["predicted_event_time"]).strftime("%Y-%m"),
                    "frequency_band": band,
                    "frequency_mhz": float(ev["frequency_mhz"]),
                    "antenna": antenna,
                    "window_s": float(window_s),
                    **_measure_event(t, y, str(ev["event_type"]), window_s, inner_s),
                }
            )
    return pd.DataFrame.from_records(rows)


def _robust_snr(values: pd.Series) -> tuple[float, float, float, float]:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    med = float(np.nanmedian(vals))
    sig = robust_sigma(vals)
    snr = float(med / (sig / np.sqrt(vals.size))) if np.isfinite(sig) and sig > 0 else np.nan
    pos = float(np.mean(vals > 0.0))
    return med, float(sig) if np.isfinite(sig) else np.nan, snr, pos


def _summarize(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    by = ["source_name", "frequency_band", "frequency_mhz", "antenna", "window_s"]
    methods = ["prepost_amp", "inner_amp", "detrended_prepost_amp", "constant_template_amp"]
    for keys, grp in events.groupby(by, sort=True, dropna=False):
        meta = dict(zip(by, keys))
        row = {
            **meta,
            "n_events": int(grp["event_id"].nunique()),
            "median_abs_drift_over_sigma": np.nan,
        }
        drift = pd.to_numeric(grp["linear_drift_window"], errors="coerce")
        sigma = pd.to_numeric(grp["local_sigma"], errors="coerce")
        ratio = np.abs(drift) / sigma.replace(0, np.nan)
        row["median_abs_drift_over_sigma"] = float(np.nanmedian(ratio)) if np.isfinite(ratio).any() else np.nan
        for method in methods:
            med, sig, snr, pos = _robust_snr(grp[method])
            prefix = method.replace("_amp", "")
            row[f"{prefix}_median"] = med
            row[f"{prefix}_robust_sigma"] = sig
            row[f"{prefix}_snr"] = snr
            row[f"{prefix}_positive_fraction"] = pos
        row.update(_classify(row))
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _classify(row: dict[str, float]) -> dict[str, str | float]:
    raw_snrs = [row.get("prepost_snr", np.nan), row.get("inner_snr", np.nan), row.get("constant_template_snr", np.nan)]
    raw_best = float(np.nanmax(np.abs(raw_snrs))) if np.isfinite(raw_snrs).any() else np.nan
    raw_signs = [np.sign(x) for x in [row.get("prepost_snr", np.nan), row.get("inner_snr", np.nan), row.get("constant_template_snr", np.nan)] if np.isfinite(x) and abs(x) >= 3]
    detrended_snr = float(row.get("detrended_prepost_snr", np.nan))
    prepost_snr = float(row.get("prepost_snr", np.nan))
    inner_snr = float(row.get("inner_snr", np.nan))
    constant_snr = float(row.get("constant_template_snr", np.nan))
    step_like = (
        np.isfinite(prepost_snr)
        and np.isfinite(constant_snr)
        and abs(prepost_snr) >= 3
        and abs(constant_snr) >= 3
        and np.sign(prepost_snr) == np.sign(constant_snr)
    )
    spike_like = (
        np.isfinite(inner_snr)
        and abs(inner_snr) >= 3
        and (not np.isfinite(prepost_snr) or abs(inner_snr) > 2.0 * max(abs(prepost_snr), 1e-9))
        and (not np.isfinite(constant_snr) or abs(inner_snr) > 2.0 * max(abs(constant_snr), 1e-9))
    )
    status = "not_detected"
    reason = "no no-trendline method reaches |SNR| >= 3"
    sign = 0.0
    if raw_signs:
        sign = float(np.sign(np.nanmedian(raw_signs)))
        same_sign_raw = int(np.count_nonzero(np.asarray(raw_signs) == sign))
        if same_sign_raw >= 2:
            if np.isfinite(detrended_snr) and abs(detrended_snr) >= 3 and np.sign(detrended_snr) == sign:
                status = "no_trendline_and_drift_supported"
                reason = "raw/constant and detrended methods agree"
            elif np.isfinite(detrended_snr) and abs(detrended_snr) < 3:
                status = "baseline_sensitive"
                reason = "no-trendline methods agree but detrended pre/post weakens below 3"
            elif np.isfinite(detrended_snr) and np.sign(detrended_snr) != sign and abs(detrended_snr) >= 3:
                status = "baseline_conflicted"
                reason = "detrended pre/post has opposite significant sign"
            else:
                status = "no_trendline_supported"
                reason = "no-trendline methods agree; detrended test unavailable/weakly constrained"
        else:
            status = "method_specific"
            reason = "only one no-trendline method reaches |SNR| >= 3"
    return {
        "best_no_trendline_abs_snr": raw_best,
        "step_like_methods_agree": bool(step_like),
        "spike_like_flag": bool(spike_like),
        "screen_status": status,
        "screen_reason": reason,
    }


def _add_frequency_coherence(summary: pd.DataFrame) -> pd.DataFrame:
    """Mark same-sign support in neighboring frequencies.

    Neighboring-frequency support is treated as positive evidence, not a
    control failure.  Coherence is counted within each source, antenna, and
    window using the signed pre/post metric.
    """

    out = summary.copy()
    out["frequency_coherence_count"] = 0
    out["frequency_coherence_flag"] = "isolated_or_weak"
    for (_source, antenna, window_s), grp in out.groupby(["source_name", "antenna", "window_s"], sort=True):
        grp = grp.sort_values("frequency_mhz")
        idxs = list(grp.index)
        snrs = pd.to_numeric(grp["prepost_snr"], errors="coerce").to_numpy(dtype=float)
        for pos, idx in enumerate(idxs):
            val = snrs[pos]
            if not np.isfinite(val) or abs(val) < 3:
                continue
            sign = np.sign(val)
            count = 1
            for neighbor in [pos - 1, pos + 1]:
                if 0 <= neighbor < len(snrs):
                    nval = snrs[neighbor]
                    if np.isfinite(nval) and abs(nval) >= 3 and np.sign(nval) == sign:
                        count += 1
            out.loc[idx, "frequency_coherence_count"] = count
            out.loc[idx, "frequency_coherence_flag"] = "neighbor_supported" if count >= 2 else "isolated_significant_channel"
    return out


def _plot_ranked(summary: pd.DataFrame, out_dir: Path) -> Path:
    sub = summary.sort_values("best_no_trendline_abs_snr", ascending=False).head(30).copy()
    labels = [
        f"{r.source_name} {r.frequency_mhz:.2f} MHz {ANT_LABEL.get(r.antenna, r.antenna)} {int(r.window_s)}s"
        for r in sub.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(10, max(5, 0.28 * len(sub))))
    y = np.arange(len(sub))
    colors = ["#d95f02" if s == "baseline_sensitive" else "#4c78a8" if "supported" in s else "#888888" for s in sub["screen_status"]]
    ax.barh(y, sub["best_no_trendline_abs_snr"], color=colors)
    ax.axvline(3, color="black", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Best no-trendline |SNR|")
    ax.set_title("No-trendline channel screen ranked results")
    fig.tight_layout()
    path = out_dir / "ranked_no_trendline_channel_screen.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_report(out_dir: Path, summary: pd.DataFrame, plot_path: Path) -> None:
    ranked = summary.sort_values("best_no_trendline_abs_snr", ascending=False)
    top = ranked.head(12)
    lines = [
        "# No-Trendline Channel Screen",
        "",
        "This screen scans channels/windows using no-trendline methods, then compares them with a simple detrended pre/post sensitivity metric.",
        "",
        "Status meanings:",
        "",
        "- `no_trendline_and_drift_supported`: no-trendline and detrended metrics agree.",
        "- `no_trendline_supported`: no-trendline methods agree; detrended result is unavailable or not decisive.",
        "- `baseline_sensitive`: no-trendline methods agree, but detrended pre/post drops below |SNR| 3.",
        "- `baseline_conflicted`: detrended pre/post is significant with the opposite sign.",
        "- `method_specific`: only one no-trendline method is strong.",
        "- `not_detected`: no no-trendline method is strong.",
        "",
        "## Top Ranked Rows",
        "",
        "| source | freq MHz | antenna | window s | best no-trendline | detrended pre/post | status |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for row in top.itertuples():
        lines.append(
            f"| {row.source_name} | {row.frequency_mhz:.2f} | {ANT_LABEL.get(row.antenna, row.antenna)} | {row.window_s:.0f} | "
            f"{row.best_no_trendline_abs_snr:.2f} | {row.detrended_prepost_snr:.2f} | {row.screen_status} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Frequencies should be interpreted as a family. Neighboring-frequency support with the same sign is positive evidence, not a control failure. "
            "Rows that are `baseline_sensitive` can remain on a watch list, but should not be promoted to detections.",
            "",
            "SNR is not sufficient by itself. A channel can have high SNR from a narrow spike rather than a persistent step. "
            "Use `step_like_methods_agree`, `spike_like_flag`, profile grids, antenna consistency, and frequency coherence together.",
            "",
            f"Plot: `{plot_path.name}`",
            "",
        ]
    )
    (out_dir / "no_trendline_channel_screen_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/no_trendline_channel_screen_v1"))
    parser.add_argument("--sources", default="earth,sun,fornax_a,cyg_a,cas_a")
    parser.add_argument("--windows-s", default="300,600,900")
    parser.add_argument("--inner-s", type=float, default=15.0)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    sources = {x.strip().lower() for x in str(args.sources).split(",") if x.strip()}
    windows = [float(x) for x in str(args.windows_s).split(",") if x.strip()]
    config = {
        "sources": sorted(sources),
        "windows_s": windows,
        "inner_s": float(args.inner_s),
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)

    clean = _read(CLEAN, parse_dates=["time"])
    event_frames = [
        _read(SUN_EVENTS, parse_dates=["predicted_event_time"]),
        _read(BRIGHT_EVENTS, parse_dates=["predicted_event_time"]),
        _read(EARTH_EVENTS, parse_dates=["predicted_event_time"]),
    ]
    events = pd.concat([f for f in event_frames if not f.empty], ignore_index=True)
    events = events[events["source_name"].astype(str).str.lower().isin(sources)].copy()
    event_metrics = _collect(clean, events, sources, windows, args.inner_s)
    event_metrics.to_csv(out_dir / "no_trendline_event_metrics.csv", index=False)
    summary = _summarize(event_metrics)
    summary = _add_frequency_coherence(summary)
    summary.to_csv(out_dir / "no_trendline_channel_summary.csv", index=False)
    plot = _plot_ranked(summary, out_dir)
    _write_report(out_dir, summary, plot)
    print(out_dir / "no_trendline_channel_screen_report.md")
    print(summary.sort_values("best_no_trendline_abs_snr", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
