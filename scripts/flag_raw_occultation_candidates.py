#!/usr/bin/env python
"""Flag raw occultation-like event windows for manual inspection.

This is a triage tool, not a detection claim.  It uses predicted event times
only to define candidate windows, then scores simple raw-data pre/post changes
inside each window.  The output is intended to answer: "which raw event
windows should a human inspect first?"
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.util import datetime_ns, ensure_dir, robust_sigma, software_versions, write_json  # noqa: E402


DEFAULT_CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
DEFAULT_SUN_EVENTS = ROOT / "outputs/pipeline_confidence_audit_v2/sun_audit_input_events.csv"
DEFAULT_OUT = ROOT / "outputs/sun_raw_occultation_candidate_flags_v1"

ANT_LABEL = {"rv1_coarse": "upper V", "rv2_coarse": "lower V"}
ANT_COLOR = {"rv1_coarse": "#4c78a8", "rv2_coarse": "#d95f02"}
EXPECTED_SIGN = {"disappearance": -1.0, "reappearance": 1.0}


def _read(path: Path, **kwargs) -> pd.DataFrame:
    return read_table(path, low_memory=False, **kwargs)


def _bool_array(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    return series.astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy(dtype=bool)


def _format_freq(freq_mhz: float) -> str:
    return f"{freq_mhz:.2f}".replace(".", "p")


def _source_label(source: str) -> str:
    return {
        "sun": "Sun",
        "earth": "Earth",
        "jupiter": "Jupiter",
        "fornax_a": "Fornax-A",
        "cyg_a": "Cyg-A",
        "cas_a": "Cas-A",
        "tau_a": "Tau-A",
    }.get(source.lower(), source)


def _make_groups(clean: pd.DataFrame) -> dict[tuple[int, str], tuple[pd.DataFrame, np.ndarray]]:
    groups: dict[tuple[int, str], tuple[pd.DataFrame, np.ndarray]] = {}
    for (band, antenna), grp in clean.groupby(["frequency_band", "antenna"], sort=True):
        g = grp.sort_values("time").reset_index(drop=True)
        groups[(int(band), str(antenna))] = (g, datetime_ns(g["time"]))
    return groups


def _event_window(group: pd.DataFrame, group_ns: np.ndarray, event_time: pd.Timestamp, window_s: float) -> pd.DataFrame:
    event_ns = pd.Timestamp(event_time).value
    half_ns = int(float(window_s) * 1e9)
    lo = int(np.searchsorted(group_ns, event_ns - half_ns, side="left"))
    hi = int(np.searchsorted(group_ns, event_ns + half_ns, side="right"))
    if hi <= lo:
        return pd.DataFrame()
    local = group.iloc[lo:hi].copy()
    local["t_rel_sec"] = (datetime_ns(local["time"]) - event_ns).astype(float) / 1e9
    local = local[np.abs(local["t_rel_sec"]) <= float(window_s)].copy()
    return local


def _valid_mask(local: pd.DataFrame, use_existing_valid: bool) -> np.ndarray:
    power = pd.to_numeric(local["power"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(power) & (power > 0.0)
    if use_existing_valid and "is_valid" in local.columns:
        keep &= _bool_array(local["is_valid"])
    return keep


def _line_slope(t: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(t) & np.isfinite(y)
    if np.count_nonzero(valid) < 3:
        return np.nan
    tt = t[valid] / 60.0
    yy = y[valid]
    if np.nanmax(tt) <= np.nanmin(tt):
        return np.nan
    try:
        return float(np.polyfit(tt, yy, 1)[0])
    except Exception:
        return np.nan


def _score_at_offset(
    t: np.ndarray,
    y: np.ndarray,
    event_type: str,
    offset_s: float,
    inner_s: float,
    prepost_s: float,
    min_side_samples: int,
) -> dict[str, float | int]:
    rel = t - float(offset_s)
    pre = (rel >= -float(prepost_s)) & (rel <= -float(inner_s))
    post = (rel >= float(inner_s)) & (rel <= float(prepost_s))
    n_pre = int(np.count_nonzero(pre))
    n_post = int(np.count_nonzero(post))
    if n_pre < min_side_samples or n_post < min_side_samples:
        return {
            "offset_s": float(offset_s),
            "n_pre": n_pre,
            "n_post": n_post,
            "pre_median": np.nan,
            "post_median": np.nan,
            "raw_delta_post_minus_pre": np.nan,
            "signed_delta": np.nan,
            "noise_scale": np.nan,
            "step_z": np.nan,
            "opposite_step_z": np.nan,
            "fractional_signed_delta": np.nan,
            "support_bins": 0,
            "pre_slope_per_min": np.nan,
            "post_slope_per_min": np.nan,
        }
    pre_vals = y[pre]
    post_vals = y[post]
    pre_med = float(np.nanmedian(pre_vals))
    post_med = float(np.nanmedian(post_vals))
    delta = post_med - pre_med
    expected = EXPECTED_SIGN[event_type]
    signed_delta = expected * delta
    side_vals = np.concatenate([pre_vals, post_vals])
    scale = robust_sigma(side_vals - np.nanmedian(side_vals))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(side_vals, ddof=1)) if len(side_vals) > 1 else np.nan
    denom = scale * np.sqrt(1.0 / max(n_pre, 1) + 1.0 / max(n_post, 1)) if np.isfinite(scale) else np.nan
    step_z = signed_delta / denom if np.isfinite(denom) and denom > 0 else np.nan
    opposite_z = -step_z if np.isfinite(step_z) else np.nan
    level = float(np.nanmedian(np.abs(side_vals)))
    frac = signed_delta / level if np.isfinite(level) and level > 0 else np.nan

    support_bins = 0
    bin_edges = [(-prepost_s, -prepost_s / 2.0), (-prepost_s / 2.0, -inner_s), (inner_s, prepost_s / 2.0), (prepost_s / 2.0, prepost_s)]
    bin_medians: list[float] = []
    for lo, hi in bin_edges:
        mask = (rel >= lo) & (rel <= hi)
        bin_medians.append(float(np.nanmedian(y[mask])) if np.count_nonzero(mask) > 0 else np.nan)
    pre_bin_ref = np.nanmedian(bin_medians[:2])
    for val in bin_medians[2:]:
        if np.isfinite(pre_bin_ref) and np.isfinite(val) and expected * (val - pre_bin_ref) > 0:
            support_bins += 1

    return {
        "offset_s": float(offset_s),
        "n_pre": n_pre,
        "n_post": n_post,
        "pre_median": pre_med,
        "post_median": post_med,
        "raw_delta_post_minus_pre": float(delta),
        "signed_delta": float(signed_delta),
        "noise_scale": float(scale) if np.isfinite(scale) else np.nan,
        "step_z": float(step_z) if np.isfinite(step_z) else np.nan,
        "opposite_step_z": float(opposite_z) if np.isfinite(opposite_z) else np.nan,
        "fractional_signed_delta": float(frac) if np.isfinite(frac) else np.nan,
        "support_bins": int(support_bins),
        "pre_slope_per_min": _line_slope(rel[pre], pre_vals),
        "post_slope_per_min": _line_slope(rel[post], post_vals),
    }


def _score_event(
    local: pd.DataFrame,
    event_type: str,
    scan_offsets_s: np.ndarray,
    inner_s: float,
    prepost_s: float,
    min_side_samples: int,
    use_existing_valid: bool,
) -> dict[str, object]:
    if local.empty:
        return {"usable": False, "primary_failure": "no_samples_in_window"}
    total_samples = int(len(local))
    keep = _valid_mask(local, use_existing_valid)
    valid_samples = int(np.count_nonzero(keep))
    if valid_samples == 0:
        return {
            "usable": False,
            "primary_failure": "no_valid_samples",
            "total_samples": total_samples,
            "valid_samples": 0,
            "valid_fraction": 0.0,
        }
    t = local.loc[keep, "t_rel_sec"].to_numpy(dtype=float)
    y = pd.to_numeric(local.loc[keep, "power"], errors="coerce").to_numpy(dtype=float)
    predicted = _score_at_offset(t, y, event_type, 0.0, inner_s, prepost_s, min_side_samples)
    scans = [_score_at_offset(t, y, event_type, float(offset), inner_s, prepost_s, min_side_samples) for offset in scan_offsets_s]
    scan_frame = pd.DataFrame(scans)
    if scan_frame["step_z"].notna().any():
        best = scan_frame.sort_values(["step_z", "support_bins"], ascending=[False, False]).iloc[0].to_dict()
    else:
        best = predicted
    if scan_frame["opposite_step_z"].notna().any():
        best_opposite = scan_frame.sort_values("opposite_step_z", ascending=False).iloc[0].to_dict()
    else:
        best_opposite = predicted
    outer = np.abs(t) >= float(inner_s)
    side_scale = robust_sigma(y[outer] - np.nanmedian(y[outer])) if np.count_nonzero(outer) else np.nan
    if not np.isfinite(side_scale) or side_scale <= 0:
        side_scale = float(np.nanstd(y[outer], ddof=1)) if np.count_nonzero(outer) > 1 else np.nan
    central = np.abs(t) < float(inner_s)
    central_peak_z = np.nan
    if np.count_nonzero(central) > 0 and np.isfinite(side_scale) and side_scale > 0:
        central_peak_z = float((np.nanmax(np.abs(y[central] - np.nanmedian(y[outer]))) / side_scale))

    failure = ""
    if int(predicted["n_pre"]) < min_side_samples:
        failure = "too_few_pre_samples"
    elif int(predicted["n_post"]) < min_side_samples:
        failure = "too_few_post_samples"
    elif valid_samples / total_samples < 0.5:
        failure = "low_valid_fraction"

    return {
        "usable": failure == "",
        "primary_failure": failure,
        "total_samples": total_samples,
        "valid_samples": valid_samples,
        "valid_fraction": float(valid_samples / total_samples),
        "predicted_n_pre": int(predicted["n_pre"]),
        "predicted_n_post": int(predicted["n_post"]),
        "predicted_pre_median": predicted["pre_median"],
        "predicted_post_median": predicted["post_median"],
        "predicted_raw_delta_post_minus_pre": predicted["raw_delta_post_minus_pre"],
        "predicted_signed_delta": predicted["signed_delta"],
        "predicted_step_z": predicted["step_z"],
        "predicted_fractional_signed_delta": predicted["fractional_signed_delta"],
        "predicted_support_bins": int(predicted["support_bins"]),
        "best_offset_s": best["offset_s"],
        "best_abs_offset_s": abs(float(best["offset_s"])) if np.isfinite(float(best["offset_s"])) else np.nan,
        "best_n_pre": int(best["n_pre"]),
        "best_n_post": int(best["n_post"]),
        "best_pre_median": best["pre_median"],
        "best_post_median": best["post_median"],
        "best_raw_delta_post_minus_pre": best["raw_delta_post_minus_pre"],
        "best_signed_delta": best["signed_delta"],
        "best_step_z": best["step_z"],
        "best_fractional_signed_delta": best["fractional_signed_delta"],
        "best_support_bins": int(best["support_bins"]),
        "best_pre_slope_per_min": best["pre_slope_per_min"],
        "best_post_slope_per_min": best["post_slope_per_min"],
        "best_opposite_offset_s": best_opposite["offset_s"],
        "best_opposite_step_z": best_opposite["opposite_step_z"],
        "best_opposite_support_bins": int(best_opposite["support_bins"]),
        "best_opposite_fractional_signed_delta": best_opposite["fractional_signed_delta"],
        "central_peak_z": central_peak_z,
    }


def _priority_reason(row: pd.Series, min_predicted_z: float, min_best_z: float, max_abs_offset_s: float) -> tuple[str, str]:
    if str(row.get("primary_failure", "")):
        return "unusable", str(row["primary_failure"])
    predicted_z = float(row.get("predicted_step_z", np.nan))
    best_z = float(row.get("best_step_z", np.nan))
    opposite_z = float(row.get("best_opposite_step_z", np.nan))
    offset = float(row.get("best_abs_offset_s", np.nan))
    support = int(row.get("best_support_bins", 0))
    opposite_support = int(row.get("best_opposite_support_bins", 0))
    min_support = int(row.get("min_support_bins", 1))
    if np.isfinite(opposite_z) and opposite_z >= min_best_z and opposite_support >= min_support and (not np.isfinite(best_z) or opposite_z > best_z):
        return "anti_template_review", f"opposite-sign raw step stronger than expected sign; opposite_z={opposite_z:.2f}"
    if (
        np.isfinite(predicted_z)
        and np.isfinite(best_z)
        and predicted_z >= min_predicted_z
        and best_z >= min_best_z
        and np.isfinite(offset)
        and offset <= max_abs_offset_s
        and support >= min_support
    ):
        return "high_priority", f"expected-sign raw step near prediction; predicted_z={predicted_z:.2f}, best_z={best_z:.2f}, offset={offset:.0f}s"
    if np.isfinite(best_z) and best_z >= min_best_z and np.isfinite(offset) and offset <= max_abs_offset_s and support >= min_support:
        return "offset_candidate", f"expected-sign step found near event but weaker at exact prediction; best_z={best_z:.2f}, offset={offset:.0f}s"
    if np.isfinite(predicted_z) and predicted_z >= min_predicted_z:
        return "weak_predicted_candidate", f"expected-sign raw step at predicted time but weak scan support; predicted_z={predicted_z:.2f}"
    return "not_flagged", "no strong raw pre/post change by configured thresholds"


def score_events(
    clean: pd.DataFrame,
    events: pd.DataFrame,
    source: str,
    event_types: set[str],
    antennas: set[str],
    frequencies: set[float] | None,
    window_s: float,
    prepost_s: float,
    inner_s: float,
    scan_radius_s: float,
    scan_step_s: float,
    min_side_samples: int,
    use_existing_valid: bool,
    min_predicted_z: float,
    min_best_z: float,
    max_abs_offset_s: float,
    min_support_bins: int,
) -> pd.DataFrame:
    source = source.lower()
    work = events[
        events["source_name"].astype(str).str.lower().eq(source)
        & events["event_type"].astype(str).isin(event_types)
        & events["antenna"].astype(str).isin(antennas)
    ].copy()
    if frequencies is not None:
        work = work[work["frequency_mhz"].astype(float).round(6).isin({round(f, 6) for f in frequencies})].copy()
    groups = _make_groups(clean)
    offsets = np.arange(-float(scan_radius_s), float(scan_radius_s) + 0.5 * float(scan_step_s), float(scan_step_s))
    rows: list[dict[str, object]] = []
    for ev in work.sort_values(["predicted_event_time", "frequency_mhz", "antenna"]).itertuples(index=False):
        band = int(ev.frequency_band)
        antenna = str(ev.antenna)
        event_type = str(ev.event_type)
        payload = groups.get((band, antenna))
        if payload is None:
            metrics = {"usable": False, "primary_failure": "no_clean_channel_group"}
        else:
            group, group_ns = payload
            local = _event_window(group, group_ns, pd.Timestamp(ev.predicted_event_time), window_s)
            metrics = _score_event(local, event_type, offsets, inner_s, prepost_s, min_side_samples, use_existing_valid)
        row = {
            "source_name": source,
            "event_id": getattr(ev, "event_id"),
            "event_type": event_type,
            "predicted_event_time": getattr(ev, "predicted_event_time"),
            "frequency_band": band,
            "frequency_mhz": float(ev.frequency_mhz),
            "antenna": antenna,
        }
        row.update(metrics)
        row["min_support_bins"] = int(min_support_bins)
        priority, reason = _priority_reason(pd.Series(row), min_predicted_z, min_best_z, max_abs_offset_s)
        row["manual_review_priority"] = priority
        row["manual_review_reason"] = reason
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_candidate_grid(candidates: pd.DataFrame, clean: pd.DataFrame, out_dir: Path, source: str, window_s: float, max_panels: int) -> Path | None:
    flagged = candidates[candidates["manual_review_priority"].isin(["high_priority", "offset_candidate", "anti_template_review", "weak_predicted_candidate"])].copy()
    if flagged.empty:
        return None
    priority_order = {"high_priority": 0, "offset_candidate": 1, "anti_template_review": 2, "weak_predicted_candidate": 3}
    flagged["priority_order"] = flagged["manual_review_priority"].map(priority_order).fillna(9)
    flagged = flagged.sort_values(["priority_order", "best_step_z", "predicted_step_z"], ascending=[True, False, False]).head(max_panels)
    groups = _make_groups(clean)
    n = len(flagged)
    fig, axes = plt.subplots(n, 1, figsize=(12, max(8, 1.55 * n)), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, row in zip(axes, flagged.itertuples(index=False)):
        payload = groups.get((int(row.frequency_band), str(row.antenna)))
        if payload is not None:
            group, group_ns = payload
            local = _event_window(group, group_ns, pd.Timestamp(row.predicted_event_time), window_s)
            keep = _valid_mask(local, use_existing_valid=True) if not local.empty else np.array([], dtype=bool)
            if not local.empty and np.any(keep):
                ax.plot(
                    local.loc[keep, "t_rel_sec"].to_numpy(dtype=float) / 60.0,
                    pd.to_numeric(local.loc[keep, "power"], errors="coerce").to_numpy(dtype=float),
                    ".-",
                    color=ANT_COLOR.get(str(row.antenna), "#d95f02"),
                    markersize=3.5,
                    linewidth=0.7,
                    label=ANT_LABEL.get(str(row.antenna), str(row.antenna)),
                )
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8, label="predicted event time")
        if np.isfinite(row.best_offset_s):
            ax.axvline(float(row.best_offset_s) / 60.0, color="#b00020", linestyle=":", linewidth=0.9, label="best scanned step time")
        ax.grid(True, color="0.9", linewidth=0.5)
        ax.set_ylabel("raw power")
        ax.set_title(
            f"{row.frequency_mhz:.2f} MHz {ANT_LABEL.get(str(row.antenna), row.antenna)} {row.event_type} "
            f"event {int(row.event_id)} | {pd.Timestamp(row.predicted_event_time)} | {row.manual_review_priority} | "
            f"z0={float(row.predicted_step_z):.2f}, zbest={float(row.best_step_z):.2f}, dt={float(row.best_offset_s):.0f}s",
            fontsize=8.5,
            loc="left",
        )
    axes[-1].set_xlabel("minutes from predicted event")
    fig.suptitle(f"{_source_label(source)} raw occultation candidate windows for manual inspection", y=0.996)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    path = out_dir / f"{source}_top_raw_occultation_candidate_windows.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_score_summary(candidates: pd.DataFrame, out_dir: Path, source: str) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    colors = {
        "high_priority": "#1b9e77",
        "offset_candidate": "#7570b3",
        "anti_template_review": "#d95f02",
        "weak_predicted_candidate": "#66a61e",
        "not_flagged": "0.65",
        "unusable": "0.85",
    }
    for priority, grp in candidates.groupby("manual_review_priority", sort=True):
        axes[0].scatter(
            grp["frequency_mhz"],
            grp["predicted_step_z"],
            s=10,
            alpha=0.45,
            color=colors.get(priority, "0.5"),
            label=priority,
        )
        axes[1].scatter(
            grp["frequency_mhz"],
            grp["best_offset_s"],
            s=np.clip(pd.to_numeric(grp["best_step_z"], errors="coerce").fillna(0).to_numpy(dtype=float), 0, 8) * 8 + 8,
            alpha=0.45,
            color=colors.get(priority, "0.5"),
            label=priority,
        )
    axes[0].axhline(0, color="0.5", linewidth=0.8)
    axes[0].set_ylabel("raw expected-sign step z at prediction")
    axes[0].set_title("Prediction-centered raw step scores")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("best scan offset (s)")
    axes[1].set_xlabel("frequency (MHz)")
    axes[1].set_title("Best raw step timing offset; marker size scales with best expected-sign score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=8)
    fig.suptitle(f"{_source_label(source)} raw occultation candidate triage summary", y=0.98)
    fig.tight_layout(rect=[0, 0, 0.88, 0.95])
    path = out_dir / f"{source}_raw_occultation_candidate_score_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_report(
    out_dir: Path,
    source: str,
    candidates: pd.DataFrame,
    shortlist: pd.DataFrame,
    config: dict[str, object],
    plot_paths: list[Path],
) -> None:
    priority_counts = candidates["manual_review_priority"].value_counts().rename_axis("priority").reset_index(name="n_rows")
    summary_cols = ["frequency_mhz", "event_type", "antenna", "manual_review_priority"]
    by_channel = candidates.groupby(summary_cols, dropna=False).size().rename("n_rows").reset_index()
    lines = [
        f"# {_source_label(source)} Raw Occultation Candidate Flags",
        "",
        "This is a manual-inspection triage run, not a detection claim.",
        "",
        "## Method",
        "",
        "For each predicted event/frequency/antenna row:",
        "",
        "1. Extract raw power samples in a fixed window around the predicted event.",
        "2. Drop invalid raw samples only at the sample level.",
        "3. Compare median raw power before and after the event.",
        "4. Use the expected sign: disappearance should decrease, reappearance should increase.",
        "5. Repeat the pre/post comparison over a small timing-offset scan.",
        "6. Flag windows where the expected-sign raw step is strong enough for manual review, or where the opposite sign is stronger.",
        "",
        "No stack SNR, baseline trendline removal, polynomial subtraction, or fitted occultation model is used.",
        "",
        "## Configuration",
        "",
        pd.Series(config).to_string(),
        "",
        "## Priority Counts",
        "",
        priority_counts.to_string(index=False),
        "",
        f"Manual-review shortlist rows: {len(shortlist)}",
        "",
        "## Channel Counts",
        "",
        by_channel.to_string(index=False),
        "",
        "## Outputs",
        "",
        "- `raw_occultation_candidate_scores.csv`: all scored event windows;",
        "- `manual_review_candidate_flags.csv`: only rows selected for review;",
        "- `manual_review_shortlist.csv`: compact top-ranked subset for first-pass inspection;",
        "- `raw_occultation_candidate_channel_summary.csv`: counts by channel/event type/antenna/priority;",
        "- `run_config.json`: exact run configuration;",
    ]
    for path in plot_paths:
        lines.append(f"- `{path.name}`")
    lines += [
        "",
        "## Plot Conventions",
        "",
        "- Orange/blue samples are raw power samples that survived the point-level validity mask and were eligible for scoring.",
        "- Samples removed by the validity mask are not plotted, so they cannot set the y-axis scale.",
        "- The black dashed vertical line is the predicted occultation event time.",
        "- The red dotted vertical line is the timing offset, within the scan range, where the strongest expected-sign raw pre/post step was found. If it lies on the black line, the strongest score is centered at the predicted event time.",
        "",
        "## How To Use",
        "",
        "Start with `manual_review_candidate_flags.csv` and the top-window plot. A useful event should show a visible raw change near the predicted time, not only a distant spike or a monotonic background drift. Opposite-sign rows are included because they are diagnostically important for the Sun.",
    ]
    (out_dir / f"{source}_raw_occultation_candidate_flag_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", default=str(DEFAULT_CLEAN))
    parser.add_argument("--events", default=str(DEFAULT_SUN_EVENTS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--source", default="sun")
    parser.add_argument("--event-types", default="disappearance,reappearance")
    parser.add_argument("--antennas", default="rv1_coarse,rv2_coarse")
    parser.add_argument("--frequencies", default="", help="Comma-separated MHz list. Empty means all event-table frequencies.")
    parser.add_argument("--window-s", type=float, default=600.0)
    parser.add_argument("--prepost-s", type=float, default=300.0)
    parser.add_argument("--inner-s", type=float, default=30.0)
    parser.add_argument("--scan-radius-s", type=float, default=300.0)
    parser.add_argument("--scan-step-s", type=float, default=60.0)
    parser.add_argument("--min-side-samples", type=int, default=4)
    parser.add_argument("--min-predicted-z", type=float, default=2.0)
    parser.add_argument("--min-best-z", type=float, default=3.0)
    parser.add_argument("--max-abs-offset-s", type=float, default=180.0)
    parser.add_argument("--min-support-bins", type=int, default=2)
    parser.add_argument("--shortlist-per-group", type=int, default=3)
    parser.add_argument("--top-plot-panels", type=int, default=36)
    parser.add_argument("--ignore-existing-valid", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    source = str(args.source).lower()
    event_types = {x.strip() for x in str(args.event_types).split(",") if x.strip()}
    antennas = {x.strip() for x in str(args.antennas).split(",") if x.strip()}
    frequencies = {float(x.strip()) for x in str(args.frequencies).split(",") if x.strip()} or None

    events = _read(Path(args.events), parse_dates=["predicted_event_time"])
    event_subset = events[
        events["source_name"].astype(str).str.lower().eq(source)
        & events["event_type"].astype(str).isin(event_types)
        & events["antenna"].astype(str).isin(antennas)
    ].copy()
    if frequencies is not None:
        event_subset = event_subset[event_subset["frequency_mhz"].astype(float).round(6).isin({round(f, 6) for f in frequencies})].copy()
    if event_subset.empty:
        raise SystemExit("No matching events found.")
    bands = sorted(event_subset["frequency_band"].astype(int).unique())

    clean_cols = ["time", "frequency_band", "frequency_mhz", "antenna", "power", "is_valid"]
    clean = _read(Path(args.clean), usecols=clean_cols, parse_dates=["time"])
    clean = clean[clean["frequency_band"].astype(int).isin(bands) & clean["antenna"].astype(str).isin(antennas)].copy()
    candidates = score_events(
        clean=clean,
        events=event_subset,
        source=source,
        event_types=event_types,
        antennas=antennas,
        frequencies=frequencies,
        window_s=float(args.window_s),
        prepost_s=float(args.prepost_s),
        inner_s=float(args.inner_s),
        scan_radius_s=float(args.scan_radius_s),
        scan_step_s=float(args.scan_step_s),
        min_side_samples=int(args.min_side_samples),
        use_existing_valid=not bool(args.ignore_existing_valid),
        min_predicted_z=float(args.min_predicted_z),
        min_best_z=float(args.min_best_z),
        max_abs_offset_s=float(args.max_abs_offset_s),
        min_support_bins=int(args.min_support_bins),
    )
    candidates.to_csv(out_dir / "raw_occultation_candidate_scores.csv", index=False)
    flagged = candidates[candidates["manual_review_priority"].ne("not_flagged") & candidates["manual_review_priority"].ne("unusable")].copy()
    priority_order = {"high_priority": 0, "offset_candidate": 1, "anti_template_review": 2, "weak_predicted_candidate": 3}
    flagged["priority_order"] = flagged["manual_review_priority"].map(priority_order).fillna(9)
    flagged["review_rank_score"] = np.where(
        flagged["manual_review_priority"].eq("anti_template_review"),
        pd.to_numeric(flagged["best_opposite_step_z"], errors="coerce"),
        pd.to_numeric(flagged["best_step_z"], errors="coerce"),
    )
    flagged = flagged.sort_values(["priority_order", "review_rank_score", "predicted_step_z"], ascending=[True, False, False])
    flagged.to_csv(out_dir / "manual_review_candidate_flags.csv", index=False)
    if flagged.empty:
        shortlist = flagged.copy()
    else:
        shortlist = (
            flagged.groupby(["frequency_mhz", "event_type", "antenna", "manual_review_priority"], group_keys=False, dropna=False)
            .head(int(args.shortlist_per_group))
            .sort_values(["priority_order", "frequency_mhz", "event_type", "antenna", "review_rank_score"], ascending=[True, True, True, True, False])
            .copy()
        )
    shortlist.to_csv(out_dir / "manual_review_shortlist.csv", index=False)
    channel_summary = (
        candidates.groupby(["frequency_mhz", "event_type", "antenna", "manual_review_priority"], dropna=False)
        .size()
        .rename("n_rows")
        .reset_index()
    )
    channel_summary.to_csv(out_dir / "raw_occultation_candidate_channel_summary.csv", index=False)

    plot_paths = [_plot_score_summary(candidates, out_dir, source)]
    top_plot_input = shortlist if not shortlist.empty else candidates
    top_path = _plot_candidate_grid(top_plot_input, clean, out_dir, source, float(args.window_s), int(args.top_plot_panels))
    if top_path is not None:
        plot_paths.append(top_path)

    config = {
        "clean": str(Path(args.clean)),
        "events": str(Path(args.events)),
        "source": source,
        "event_types": sorted(event_types),
        "antennas": sorted(antennas),
        "frequencies": sorted(frequencies) if frequencies is not None else "all",
        "window_s": float(args.window_s),
        "prepost_s": float(args.prepost_s),
        "inner_s": float(args.inner_s),
        "scan_radius_s": float(args.scan_radius_s),
        "scan_step_s": float(args.scan_step_s),
        "min_side_samples": int(args.min_side_samples),
        "min_predicted_z": float(args.min_predicted_z),
        "min_best_z": float(args.min_best_z),
        "max_abs_offset_s": float(args.max_abs_offset_s),
        "min_support_bins": int(args.min_support_bins),
        "shortlist_per_group": int(args.shortlist_per_group),
        "use_existing_valid": not bool(args.ignore_existing_valid),
        "n_scored_rows": int(len(candidates)),
        "n_flagged_rows": int(len(flagged)),
        "n_shortlist_rows": int(len(shortlist)),
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)
    _write_report(out_dir, source, candidates, shortlist, config, plot_paths)
    print(out_dir / f"{source}_raw_occultation_candidate_flag_report.md")


if __name__ == "__main__":
    main()
