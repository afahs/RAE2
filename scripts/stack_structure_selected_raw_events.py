#!/usr/bin/env python
"""Stack raw occultation windows selected by manual-inspection triage flags."""

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
DEFAULT_CANDIDATES = ROOT / "outputs/sun_lower_v_raw_occultation_candidate_flags_v2_cleanplot/raw_occultation_candidate_scores.csv"
DEFAULT_OUT = ROOT / "outputs/sun_lower_v_structure_selected_stacks_v1"
ANT_LABEL = {"rv1_coarse": "upper V", "rv2_coarse": "lower V"}
ANT_COLOR = {"rv1_coarse": "#4c78a8", "rv2_coarse": "#d95f02"}
EXPECTED_SIGN = {"disappearance": -1.0, "reappearance": 1.0}


def _read(path: Path, **kwargs) -> pd.DataFrame:
    return read_table(path, low_memory=False, **kwargs)


def _bool_array(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    return series.astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy(dtype=bool)


def _valid_mask(local: pd.DataFrame, use_existing_valid: bool) -> np.ndarray:
    y = pd.to_numeric(local["power"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(y) & (y > 0)
    if use_existing_valid and "is_valid" in local.columns:
        keep &= _bool_array(local["is_valid"])
    return keep


def _mad(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    med = np.nanmedian(vals)
    return float(1.4826 * np.nanmedian(np.abs(vals - med)))


def _robust_sem(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size <= 1:
        return np.nan
    sig = robust_sigma(vals - np.nanmedian(vals))
    if not np.isfinite(sig) or sig <= 0:
        sig = float(np.nanstd(vals, ddof=1))
    return float(sig / np.sqrt(vals.size)) if np.isfinite(sig) else np.nan


def _make_groups(clean: pd.DataFrame) -> dict[tuple[int, str], tuple[pd.DataFrame, np.ndarray]]:
    groups: dict[tuple[int, str], tuple[pd.DataFrame, np.ndarray]] = {}
    for (band, antenna), grp in clean.groupby(["frequency_band", "antenna"], sort=True):
        g = grp.sort_values("time").reset_index(drop=True)
        groups[(int(band), str(antenna))] = (g, datetime_ns(g["time"]))
    return groups


def _event_window(group: pd.DataFrame, group_ns: np.ndarray, center_time: pd.Timestamp, window_s: float) -> pd.DataFrame:
    center_ns = pd.Timestamp(center_time).value
    half_ns = int(float(window_s) * 1e9)
    lo = int(np.searchsorted(group_ns, center_ns - half_ns, side="left"))
    hi = int(np.searchsorted(group_ns, center_ns + half_ns, side="right"))
    if hi <= lo:
        return pd.DataFrame()
    local = group.iloc[lo:hi].copy()
    local["t_rel_sec"] = (datetime_ns(local["time"]) - center_ns).astype(float) / 1e9
    return local[np.abs(local["t_rel_sec"]) <= float(window_s)].copy()


def _normalize(t: np.ndarray, y: np.ndarray, sideband_s: float) -> tuple[np.ndarray, float, float] | None:
    side = np.abs(t) >= float(sideband_s)
    if np.count_nonzero(side) < 6:
        return None
    center = float(np.nanmedian(y[side]))
    scale = robust_sigma(y[side] - center)
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(y[side], ddof=1)) if np.count_nonzero(side) > 1 else np.nan
    if not np.isfinite(scale) or scale <= 0:
        return None
    return (y - center) / scale, center, scale


def collect_selected_profiles(
    clean: pd.DataFrame,
    candidates: pd.DataFrame,
    window_s: float,
    bin_s: float,
    sideband_s: float,
    center_mode: str,
    use_existing_valid: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = _make_groups(clean)
    bins = np.arange(-float(window_s), float(window_s) + float(bin_s), float(bin_s))
    rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    for cand in candidates.itertuples(index=False):
        band = int(cand.frequency_band)
        antenna = str(cand.antenna)
        event_time = pd.Timestamp(cand.predicted_event_time)
        best_offset = float(cand.best_offset_s) if np.isfinite(float(cand.best_offset_s)) else 0.0
        center_time = event_time + pd.to_timedelta(best_offset, unit="s") if center_mode == "best_offset_centered" else event_time
        payload = groups.get((band, antenna))
        base_status = {
            "source_name": str(cand.source_name),
            "event_id": cand.event_id,
            "event_type": str(cand.event_type),
            "predicted_event_time": cand.predicted_event_time,
            "frequency_band": band,
            "frequency_mhz": float(cand.frequency_mhz),
            "antenna": antenna,
            "center_mode": center_mode,
            "best_offset_s": best_offset,
            "manual_review_priority": str(cand.manual_review_priority),
            "predicted_step_z": float(cand.predicted_step_z) if np.isfinite(float(cand.predicted_step_z)) else np.nan,
            "best_step_z": float(cand.best_step_z) if np.isfinite(float(cand.best_step_z)) else np.nan,
            "valid_fraction_score_window": float(cand.valid_fraction) if np.isfinite(float(cand.valid_fraction)) else np.nan,
        }
        if payload is None:
            status_rows.append({**base_status, "used_in_stack": False, "stack_failure": "missing_clean_group", "n_valid_samples": 0})
            continue
        group, group_ns = payload
        local = _event_window(group, group_ns, center_time, window_s)
        if local.empty:
            status_rows.append({**base_status, "used_in_stack": False, "stack_failure": "no_samples", "n_valid_samples": 0})
            continue
        keep = _valid_mask(local, use_existing_valid)
        if np.count_nonzero(keep) < 8:
            status_rows.append({**base_status, "used_in_stack": False, "stack_failure": "too_few_valid_samples", "n_valid_samples": int(np.count_nonzero(keep))})
            continue
        t = local.loc[keep, "t_rel_sec"].to_numpy(dtype=float)
        y = pd.to_numeric(local.loc[keep, "power"], errors="coerce").to_numpy(dtype=float)
        order = np.argsort(t)
        t = t[order]
        y = y[order]
        norm = _normalize(t, y, sideband_s)
        if norm is None:
            status_rows.append({**base_status, "used_in_stack": False, "stack_failure": "normalization_failed", "n_valid_samples": int(len(y))})
            continue
        z, center, scale = norm
        bin_idx = np.digitize(t, bins) - 1
        event_level_pre = (t >= -300) & (t <= -30)
        event_level_post = (t >= 30) & (t <= 300)
        event_signed_contrast = np.nan
        if np.count_nonzero(event_level_pre) >= 2 and np.count_nonzero(event_level_post) >= 2:
            delta = float(np.nanmedian(z[event_level_post]) - np.nanmedian(z[event_level_pre]))
            event_signed_contrast = EXPECTED_SIGN[str(cand.event_type)] * delta
        status_rows.append(
            {
                **base_status,
                "used_in_stack": True,
                "stack_failure": "",
                "n_valid_samples": int(len(y)),
                "normalization_center": center,
                "normalization_scale": scale,
                "event_signed_contrast_z": event_signed_contrast,
            }
        )
        for idx in sorted(set(bin_idx)):
            if idx < 0 or idx >= len(bins) - 1:
                continue
            mask = bin_idx == idx
            if np.count_nonzero(mask) == 0:
                continue
            rows.append(
                {
                    "source_name": str(cand.source_name),
                    "event_id": cand.event_id,
                    "event_type": str(cand.event_type),
                    "predicted_event_time": cand.predicted_event_time,
                    "frequency_band": band,
                    "frequency_mhz": float(cand.frequency_mhz),
                    "antenna": antenna,
                    "center_mode": center_mode,
                    "manual_review_priority": str(cand.manual_review_priority),
                    "best_offset_s": best_offset,
                    "t_bin_sec": float(0.5 * (bins[idx] + bins[idx + 1])),
                    "z_power": float(np.nanmedian(z[mask])),
                    "n_samples": int(np.count_nonzero(mask)),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(status_rows)


def stack_profiles(points: pd.DataFrame) -> pd.DataFrame:
    if points.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    by = ["source_name", "frequency_band", "frequency_mhz", "antenna", "center_mode", "event_type", "t_bin_sec"]
    for keys, grp in points.groupby(by, sort=True, dropna=False):
        vals = pd.to_numeric(grp["z_power"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        rows.append(
            {
                **dict(zip(by, keys)),
                "median_z_power": float(np.nanmedian(vals)),
                "mean_z_power": float(np.nanmean(vals)),
                "robust_sem_z_power": _robust_sem(vals),
                "n_events": int(grp["event_id"].nunique()),
                "n_points": int(vals.size),
            }
        )
    return pd.DataFrame(rows)


def stack_statistics(stack: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if stack.empty:
        return pd.DataFrame()
    by = ["frequency_mhz", "frequency_band", "antenna", "center_mode", "event_type"]
    for keys, grp in stack.groupby(by, sort=True, dropna=False):
        pre = grp[(grp["t_bin_sec"] >= -300) & (grp["t_bin_sec"] <= -60)]["median_z_power"].to_numpy(dtype=float)
        post = grp[(grp["t_bin_sec"] >= 60) & (grp["t_bin_sec"] <= 300)]["median_z_power"].to_numpy(dtype=float)
        event_type = str(keys[-1])
        delta = float(np.nanmedian(post) - np.nanmedian(pre)) if pre.size and post.size else np.nan
        signed = EXPECTED_SIGN[event_type] * delta if event_type in EXPECTED_SIGN and np.isfinite(delta) else np.nan
        sub_status = status[
            np.isclose(status["frequency_mhz"].astype(float), float(keys[0]))
            & status["antenna"].astype(str).eq(str(keys[2]))
            & status["center_mode"].astype(str).eq(str(keys[3]))
            & status["event_type"].astype(str).eq(event_type)
        ].copy()
        used = sub_status[sub_status["used_in_stack"].astype(bool)].copy() if not sub_status.empty else pd.DataFrame()
        event_contrasts = pd.to_numeric(used.get("event_signed_contrast_z", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
        offsets = pd.to_numeric(used.get("best_offset_s", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
        rows.append(
            {
                **dict(zip(by, keys)),
                "n_selected_events": int(len(sub_status)),
                "n_used_events": int(len(used)),
                "use_fraction": float(len(used) / len(sub_status)) if len(sub_status) else np.nan,
                "stack_post_minus_pre_z": delta,
                "stack_expected_signed_contrast_z": signed,
                "median_event_expected_signed_contrast_z": float(np.nanmedian(event_contrasts)) if event_contrasts.size else np.nan,
                "event_expected_signed_contrast_robust_sem": _robust_sem(event_contrasts),
                "fraction_events_expected_positive": float(np.mean(event_contrasts > 0)) if event_contrasts.size else np.nan,
                "median_best_offset_s": float(np.nanmedian(offsets)) if offsets.size else np.nan,
                "mad_best_offset_s": _mad(offsets),
                "median_predicted_step_z": float(pd.to_numeric(used.get("predicted_step_z", pd.Series(dtype=float)), errors="coerce").median()) if not used.empty else np.nan,
                "median_best_step_z": float(pd.to_numeric(used.get("best_step_z", pd.Series(dtype=float)), errors="coerce").median()) if not used.empty else np.nan,
                "median_valid_fraction": float(pd.to_numeric(used.get("valid_fraction_score_window", pd.Series(dtype=float)), errors="coerce").median()) if not used.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_stack_grid(stack: pd.DataFrame, stats: pd.DataFrame, source: str, center_mode: str, out_dir: Path) -> Path:
    sub = stack[stack["center_mode"].astype(str).eq(center_mode)].copy()
    freqs = sorted(sub["frequency_mhz"].dropna().unique())
    event_types = ["disappearance", "reappearance"]
    fig, axes = plt.subplots(len(freqs), 2, figsize=(12, max(10, 1.35 * len(freqs))), sharex=True, sharey=False)
    if len(freqs) == 1:
        axes = np.asarray([axes])
    for i, freq in enumerate(freqs):
        for j, event_type in enumerate(event_types):
            ax = axes[i, j]
            g = sub[np.isclose(sub["frequency_mhz"].astype(float), float(freq)) & sub["event_type"].astype(str).eq(event_type)].sort_values("t_bin_sec")
            if not g.empty:
                ax.errorbar(
                    g["t_bin_sec"],
                    g["median_z_power"],
                    yerr=g["robust_sem_z_power"],
                    color="#d95f02",
                    marker="o",
                    markersize=2.5,
                    linewidth=1.2,
                    elinewidth=0.65,
                    capsize=1.3,
                    label="lower V",
                )
            s = stats[
                np.isclose(stats["frequency_mhz"].astype(float), float(freq))
                & stats["center_mode"].astype(str).eq(center_mode)
                & stats["event_type"].astype(str).eq(event_type)
            ]
            n_used = int(s["n_used_events"].iloc[0]) if not s.empty else 0
            signed = float(s["stack_expected_signed_contrast_z"].iloc[0]) if not s.empty else np.nan
            ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
            ax.axhline(0, color="0.6", linewidth=0.7)
            ax.set_title(f"{freq:.2f} MHz {event_type} | n={n_used} | signed contrast={signed:.2f}", fontsize=8.5)
            if j == 0:
                ax.set_ylabel("normalized raw power")
            if i == len(freqs) - 1:
                ax.set_xlabel("seconds from stack center")
            if i == 0 and j == 1:
                ax.legend(frameon=False, fontsize=8)
    label = "predicted event time" if center_mode == "predicted_centered" else "best raw-step offset"
    fig.suptitle(
        f"{source}: lower-V structure-selected stack, centered on {label}\n"
        "Only high-priority expected-sign raw candidate events are included; no trendline subtraction.",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    path = out_dir / f"{source}_lower_v_structure_selected_stack_{center_mode}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(out_dir: Path, source: str, selected: pd.DataFrame, status: pd.DataFrame, stats: pd.DataFrame, plot_paths: list[Path], config: dict[str, object]) -> None:
    selection_counts = (
        selected.groupby(["frequency_mhz", "event_type", "manual_review_priority"], dropna=False)
        .size()
        .rename("n_selected_rows")
        .reset_index()
    )
    status_counts = (
        status.groupby(["center_mode", "used_in_stack", "stack_failure"], dropna=False)
        .size()
        .rename("n_rows")
        .reset_index()
        if not status.empty
        else pd.DataFrame()
    )
    lines = [
        f"# {source.title()} Lower-V Structure-Selected Stack",
        "",
        "This stack uses only lower-V (`rv2_coarse`) events that passed the raw triage as `high_priority`.",
        "That means the individual raw window had the expected occultation sign near the predicted time and persistent post-event support in the simple pre/post median test.",
        "",
        "This is still a selected stack, not an unbiased detection statistic. It asks: if we keep only events with individually plausible raw morphology, what does their average profile look like?",
        "",
        "## Configuration",
        "",
        pd.Series(config).to_string(),
        "",
        "## Selection Counts",
        "",
        selection_counts.to_string(index=False),
        "",
        "## Stack Extraction Status",
        "",
        status_counts.to_string(index=False) if not status_counts.empty else "No status rows.",
        "",
        "## Stack Statistics",
        "",
        stats.to_string(index=False) if not stats.empty else "No stack statistics.",
        "",
        "## Plot Notes",
        "",
        "- y-axis is per-event normalized raw power: `(raw - sideband median) / sideband robust sigma`.",
        "- No trendline, polynomial, simulator, or fitted baseline has been subtracted.",
        "- `predicted_centered` stacks align events at the predicted occultation time.",
        "- `best_offset_centered` stacks align events at the strongest raw expected-sign pre/post step found by the triage scan.",
        "- `stack_expected_signed_contrast_z` is positive when the stacked profile changes in the expected direction.",
        "",
        "Generated plots:",
        "",
    ]
    lines.extend(f"- `{p.name}`" for p in plot_paths)
    lines += [
        "",
        "Generated tables:",
        "",
        "- `structure_selected_event_status.csv`",
        "- `structure_selected_stack_points.csv`",
        "- `structure_selected_stack_summary.csv`",
        "- `structure_selected_stack_statistics.csv`",
    ]
    (out_dir / f"{source}_lower_v_structure_selected_stack_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", default=str(DEFAULT_CLEAN))
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--source", default="sun")
    parser.add_argument("--antenna", default="rv2_coarse")
    parser.add_argument("--priorities", default="high_priority")
    parser.add_argument("--window-s", type=float, default=600.0)
    parser.add_argument("--bin-s", type=float, default=60.0)
    parser.add_argument("--sideband-s", type=float, default=30.0)
    parser.add_argument("--use-existing-valid", action="store_true", default=True)
    args = parser.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    source = str(args.source).lower()
    priorities = {x.strip() for x in str(args.priorities).split(",") if x.strip()}
    candidates = _read(Path(args.candidates), parse_dates=["predicted_event_time"])
    selected = candidates[
        candidates["source_name"].astype(str).str.lower().eq(source)
        & candidates["antenna"].astype(str).eq(str(args.antenna))
        & candidates["manual_review_priority"].astype(str).isin(priorities)
    ].copy()
    if selected.empty:
        raise SystemExit("No selected candidate rows matched the requested filters.")
    selected.to_csv(out_dir / "structure_selected_candidate_rows.csv", index=False)

    bands = sorted(selected["frequency_band"].astype(int).unique())
    clean_cols = ["time", "frequency_band", "frequency_mhz", "antenna", "power", "is_valid"]
    clean = _read(Path(args.clean), usecols=clean_cols, parse_dates=["time"])
    clean = clean[clean["frequency_band"].astype(int).isin(bands) & clean["antenna"].astype(str).eq(str(args.antenna))].copy()

    all_points = []
    all_status = []
    for center_mode in ["predicted_centered", "best_offset_centered"]:
        points, status = collect_selected_profiles(
            clean=clean,
            candidates=selected,
            window_s=float(args.window_s),
            bin_s=float(args.bin_s),
            sideband_s=float(args.sideband_s),
            center_mode=center_mode,
            use_existing_valid=bool(args.use_existing_valid),
        )
        all_points.append(points)
        all_status.append(status)
    points = pd.concat(all_points, ignore_index=True) if all_points else pd.DataFrame()
    status = pd.concat(all_status, ignore_index=True) if all_status else pd.DataFrame()
    stack = stack_profiles(points)
    stats = stack_statistics(stack, status)

    points.to_csv(out_dir / "structure_selected_stack_points.csv", index=False)
    status.to_csv(out_dir / "structure_selected_event_status.csv", index=False)
    stack.to_csv(out_dir / "structure_selected_stack_summary.csv", index=False)
    stats.to_csv(out_dir / "structure_selected_stack_statistics.csv", index=False)

    plot_paths = []
    if not stack.empty:
        for center_mode in ["predicted_centered", "best_offset_centered"]:
            plot_paths.append(plot_stack_grid(stack, stats, source, center_mode, out_dir))

    config = {
        "clean": str(Path(args.clean)),
        "candidates": str(Path(args.candidates)),
        "source": source,
        "antenna": str(args.antenna),
        "priorities": sorted(priorities),
        "window_s": float(args.window_s),
        "bin_s": float(args.bin_s),
        "sideband_s": float(args.sideband_s),
        "n_selected_candidate_rows": int(len(selected)),
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)
    write_report(out_dir, source, selected, status, stats, plot_paths, config)
    print(out_dir / f"{source}_lower_v_structure_selected_stack_report.md")


if __name__ == "__main__":
    main()
