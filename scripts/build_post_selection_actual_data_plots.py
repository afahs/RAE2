#!/usr/bin/env python
"""Plot actual lower-V data behind the post-selection bias audit."""

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

from rylevonberg.constants import FREQUENCY_MAP_MHZ  # noqa: E402
from rylevonberg.util import datetime_ns, ensure_dir, robust_sigma, write_json  # noqa: E402


CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
AUDIT_ROOT = ROOT / "outputs/post_selection_bias_audit_sun_fornax_dt0_v1"
DEFAULT_OUT = AUDIT_ROOT / "actual_data_plots"
ANTENNA = "rv2_coarse"

GROUP_STYLE = {
    "real_all_usable": ("all usable real", "#808080", "-", 0.85),
    "real_selected": ("selected real", "#000000", "-", 1.0),
    "time_shift": ("selected time-shift controls", "#d95f02", "--", 0.9),
    "randomized_time": ("selected randomized-time controls", "#7570b3", "--", 0.9),
    "offsource": ("selected off-source controls", "#1b9e77", "--", 0.9),
}

GROUP_COUNT_LABEL = {
    "real_all_usable": "all",
    "real_selected": "sel",
    "time_shift": "shift",
    "randomized_time": "rand",
    "offsource": "off",
}


def _read(path: Path, **kwargs) -> pd.DataFrame:
    return read_table(path, low_memory=False, **kwargs)


def _bool_values(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    return series.astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy(dtype=bool)


def _load_clean_groups(path: Path, bands: list[int]) -> dict[tuple[int, str], dict[str, np.ndarray]]:
    cols = ["time", "frequency_band", "frequency_mhz", "antenna", "power", "is_valid"]
    clean = _read(path, usecols=cols, parse_dates=["time"])
    clean = clean[clean["antenna"].astype(str).eq(ANTENNA) & clean["frequency_band"].astype(int).isin(set(bands))].copy()
    groups: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for (band, antenna), grp in clean.groupby(["frequency_band", "antenna"], sort=True):
        g = grp.sort_values("time").reset_index(drop=True)
        power = pd.to_numeric(g["power"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(power) & (power > 0)
        if "is_valid" in g.columns:
            valid &= _bool_values(g["is_valid"])
        groups[(int(band), str(antenna))] = {
            "time_ns": datetime_ns(g["time"]),
            "power": power,
            "valid": valid,
        }
    return groups


def _normalize_window(t: np.ndarray, y: np.ndarray, sideband_s: float) -> tuple[np.ndarray, np.ndarray] | None:
    side = np.abs(t) >= float(sideband_s)
    if np.count_nonzero(side) < 6:
        return None
    center = float(np.nanmedian(y[side]))
    scale = robust_sigma(y[side] - center)
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(y[side], ddof=1)) if np.count_nonzero(side) > 1 else np.nan
    if not np.isfinite(scale) or scale <= 0:
        return None
    return t, (y - center) / scale


def _window_arrays(
    group: dict[str, np.ndarray],
    event_time: pd.Timestamp,
    window_s: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    event_ns = pd.Timestamp(event_time).value
    half_ns = int(float(window_s) * 1e9)
    time_ns = group["time_ns"]
    lo = int(np.searchsorted(time_ns, event_ns - half_ns, side="left"))
    hi = int(np.searchsorted(time_ns, event_ns + half_ns, side="right"))
    if hi <= lo:
        return None
    t = (time_ns[lo:hi] - event_ns).astype(float) / 1e9
    keep = group["valid"][lo:hi] & (np.abs(t) <= float(window_s))
    if np.count_nonzero(keep) < 8:
        return None
    y = group["power"][lo:hi]
    order = np.argsort(t[keep])
    return t[keep][order], y[keep][order]


def _robust_sem(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size <= 1:
        return np.nan
    scale = robust_sigma(vals - np.nanmedian(vals))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(vals, ddof=1))
    return float(scale / np.sqrt(vals.size)) if np.isfinite(scale) and scale > 0 else np.nan


def _deterministic_cap(df: pd.DataFrame, max_per_group: int, seed: int) -> pd.DataFrame:
    if max_per_group <= 0:
        return df
    pieces = []
    keys = ["source_name", "plot_group", "frequency_mhz", "event_type"]
    for _, grp in df.groupby(keys, sort=True, dropna=False):
        if len(grp) <= max_per_group:
            pieces.append(grp)
        else:
            pieces.append(grp.sample(n=max_per_group, random_state=seed).sort_values("event_id"))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=df.columns)


def _prepare_plot_candidates(scores: pd.DataFrame, max_control_per_group: int, seed: int) -> pd.DataFrame:
    work = scores[scores["antenna"].astype(str).eq(ANTENNA)].copy()
    usable = work["usable_bool"] if "usable_bool" in work.columns else work["usable"]
    if usable.dtype != bool:
        usable = usable.astype(str).str.lower().isin(["true", "1", "yes"])
    selected = work["expected_shape_any"] if "expected_shape_any" in work.columns else work["manual_review_priority"].astype(str).isin(
        ["high_priority", "offset_candidate", "weak_predicted_candidate"]
    )
    if selected.dtype != bool:
        selected = selected.astype(str).str.lower().isin(["true", "1", "yes"])

    real_all = work[work["run_role"].eq("real") & usable].copy()
    real_all["plot_group"] = "real_all_usable"
    real_selected = work[work["run_role"].eq("real") & usable & selected].copy()
    real_selected["plot_group"] = "real_selected"
    control_selected = work[work["run_role"].eq("control") & usable & selected].copy()
    control_selected["plot_group"] = control_selected["control_family"].astype(str)
    control_selected = control_selected[control_selected["plot_group"].isin(["time_shift", "randomized_time", "offsource"])].copy()
    control_selected = _deterministic_cap(control_selected, max_control_per_group, seed)
    out = pd.concat([real_all, real_selected, control_selected], ignore_index=True)
    return out.sort_values(["source_name", "plot_group", "frequency_mhz", "event_type", "predicted_event_time"]).reset_index(drop=True)


def collect_actual_profiles(
    scores: pd.DataFrame,
    clean_groups: dict[tuple[int, str], dict[str, np.ndarray]],
    window_s: float,
    bin_s: float,
    sideband_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bins = np.arange(-float(window_s), float(window_s) + float(bin_s), float(bin_s))
    point_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    for row in scores.itertuples(index=False):
        band = int(row.frequency_band)
        payload = clean_groups.get((band, ANTENNA))
        base = {
            "source_name": str(row.source_name),
            "plot_group": str(row.plot_group),
            "event_id": row.event_id,
            "event_type": str(row.event_type),
            "frequency_band": band,
            "frequency_mhz": float(row.frequency_mhz),
            "predicted_event_time": row.predicted_event_time,
        }
        if payload is None:
            status_rows.append({**base, "used_in_plot": False, "failure": "missing_clean_group", "n_valid_samples": 0})
            continue
        local = _window_arrays(payload, pd.Timestamp(row.predicted_event_time), window_s)
        if local is None:
            status_rows.append({**base, "used_in_plot": False, "failure": "no_or_few_samples", "n_valid_samples": 0})
            continue
        norm = _normalize_window(local[0], local[1], sideband_s)
        if norm is None:
            status_rows.append({**base, "used_in_plot": False, "failure": "normalization_failed", "n_valid_samples": len(local[0])})
            continue
        t, z = norm
        status_rows.append({**base, "used_in_plot": True, "failure": "", "n_valid_samples": len(t)})
        bin_idx = np.digitize(t, bins) - 1
        for idx in sorted(set(bin_idx)):
            if idx < 0 or idx >= len(bins) - 1:
                continue
            mask = bin_idx == idx
            if np.count_nonzero(mask) == 0:
                continue
            point_rows.append(
                {
                    **base,
                    "t_bin_sec": float(0.5 * (bins[idx] + bins[idx + 1])),
                    "z_power": float(np.nanmedian(z[mask])),
                    "n_samples": int(np.count_nonzero(mask)),
                }
            )
    return pd.DataFrame(point_rows), pd.DataFrame(status_rows)


def summarize_profiles(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    by = ["source_name", "plot_group", "frequency_band", "frequency_mhz", "event_type", "t_bin_sec"]
    for keys, grp in points.groupby(by, sort=True, dropna=False):
        vals = pd.to_numeric(grp["z_power"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        rows.append(
            {
                **dict(zip(by, keys)),
                "median_z_power": float(np.nanmedian(vals)),
                "robust_sem_z_power": _robust_sem(vals),
                "n_windows": int(grp["event_id"].nunique()),
                "n_points": int(vals.size),
            }
        )
    return pd.DataFrame(rows)


def _plot_grid(
    summary: pd.DataFrame,
    status: pd.DataFrame,
    source: str,
    groups_to_plot: list[str],
    title: str,
    out_path: Path,
) -> None:
    freqs = sorted(summary.loc[summary["source_name"].astype(str).eq(source), "frequency_mhz"].dropna().unique())
    event_types = ["disappearance", "reappearance"]
    fig, axes = plt.subplots(len(freqs), 2, figsize=(13.5, max(10, 1.45 * len(freqs))), sharex=True, sharey=False)
    if len(freqs) == 1:
        axes = np.asarray([axes])
    for i, freq in enumerate(freqs):
        for j, event_type in enumerate(event_types):
            ax = axes[i, j]
            for group in groups_to_plot:
                sub = summary[
                    summary["source_name"].astype(str).eq(source)
                    & np.isclose(summary["frequency_mhz"].astype(float), float(freq))
                    & summary["event_type"].astype(str).eq(event_type)
                    & summary["plot_group"].astype(str).eq(group)
                ].sort_values("t_bin_sec")
                if sub.empty:
                    continue
                label, color, linestyle, alpha = GROUP_STYLE[group]
                ax.errorbar(
                    sub["t_bin_sec"] / 60.0,
                    sub["median_z_power"],
                    yerr=sub["robust_sem_z_power"],
                    marker="o",
                    markersize=2.3,
                    linewidth=1.1 if group != "real_selected" else 1.7,
                    elinewidth=0.55,
                    capsize=1.1,
                    color=color,
                    ecolor=color,
                    linestyle=linestyle,
                    alpha=alpha,
                    label=label,
                )
            ax.axvline(0, color="black", linestyle=":", linewidth=0.8)
            ax.axhline(0, color="0.7", linewidth=0.7)
            used = status[
                status["source_name"].astype(str).eq(source)
                & np.isclose(status["frequency_mhz"].astype(float), float(freq))
                & status["event_type"].astype(str).eq(event_type)
                & status["plot_group"].astype(str).isin(groups_to_plot)
                & status["used_in_plot"].astype(bool)
            ]
            group_counts = used.groupby("plot_group")["event_id"].nunique().to_dict() if not used.empty else {}
            count_text = "  ".join(
                f"{GROUP_COUNT_LABEL.get(g, g)}={group_counts.get(g, 0)}"
                for g in groups_to_plot
                if group_counts.get(g, 0) > 0
            )
            ax.set_title(f"{float(freq):.2f} MHz {event_type}", fontsize=8.5)
            if count_text:
                ax.text(
                    0.01,
                    0.96,
                    count_text,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=6.4,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
                )
            if j == 0:
                ax.set_ylabel("normalized raw power")
            if i == len(freqs) - 1:
                ax.set_xlabel("minutes from predicted event")
            ax.grid(True, color="0.92", linewidth=0.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=min(len(by_label), 4), frameon=False)
    fig.suptitle(title, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", default=str(AUDIT_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--sources", default="sun,fornax_a")
    parser.add_argument("--window-s", type=float, default=600.0)
    parser.add_argument("--bin-s", type=float, default=60.0)
    parser.add_argument("--sideband-s", type=float, default=300.0)
    parser.add_argument("--max-control-per-frequency-event", type=int, default=250)
    parser.add_argument("--random-seed", type=int, default=20260604)
    args = parser.parse_args()

    audit_root = Path(args.audit_root)
    out_dir = ensure_dir(Path(args.out_dir))
    sources = [s.strip().lower() for s in str(args.sources).split(",") if s.strip()]
    scores_path = audit_root / "post_selection_all_scores.csv"
    cols = [
        "source_name",
        "run_role",
        "control_family",
        "control_id",
        "event_id",
        "event_type",
        "predicted_event_time",
        "frequency_band",
        "frequency_mhz",
        "antenna",
        "usable",
        "usable_bool",
        "manual_review_priority",
        "expected_shape_any",
    ]
    print(f"Loading scores from {scores_path}", flush=True)
    scores = _read(scores_path, usecols=lambda c: c in cols, parse_dates=["predicted_event_time"])
    scores = scores[scores["source_name"].astype(str).str.lower().isin(sources)].copy()
    candidates = _prepare_plot_candidates(scores, int(args.max_control_per_frequency_event), int(args.random_seed))
    candidates.to_csv(out_dir / "actual_data_plot_candidate_rows.csv", index=False)

    bands = sorted(candidates["frequency_band"].dropna().astype(int).unique())
    print("Loading lower-V clean groups...", flush=True)
    clean_groups = _load_clean_groups(CLEAN, bands)
    print(f"Collecting profiles for {len(candidates)} candidate/control rows...", flush=True)
    points, status = collect_actual_profiles(
        candidates,
        clean_groups,
        window_s=float(args.window_s),
        bin_s=float(args.bin_s),
        sideband_s=float(args.sideband_s),
    )
    points.to_csv(out_dir / "actual_data_profile_points.csv", index=False)
    status.to_csv(out_dir / "actual_data_profile_status.csv", index=False)
    summary = summarize_profiles(points)
    summary.to_csv(out_dir / "actual_data_profile_summary.csv", index=False)

    plot_paths: list[Path] = []
    for source in sources:
        p1 = out_dir / f"{source}_actual_data_real_all_vs_selected.png"
        _plot_grid(
            summary,
            status,
            source,
            ["real_all_usable", "real_selected"],
            f"{source}: actual lower-V data, all usable real windows versus raw-shape-selected real windows",
            p1,
        )
        plot_paths.append(p1)
        p2 = out_dir / f"{source}_actual_data_selected_real_vs_selected_controls.png"
        _plot_grid(
            summary,
            status,
            source,
            ["real_selected", "time_shift", "randomized_time", "offsource"],
            f"{source}: actual lower-V data, selected real windows versus selected controls",
            p2,
        )
        plot_paths.append(p2)

    config = {
        "audit_root": str(audit_root),
        "clean": str(CLEAN),
        "antenna": ANTENNA,
        "sources": sources,
        "window_s": float(args.window_s),
        "bin_s": float(args.bin_s),
        "sideband_s": float(args.sideband_s),
        "max_control_per_frequency_event": int(args.max_control_per_frequency_event),
        "random_seed": int(args.random_seed),
        "notes": "Plots use normalized raw lower-V power. Controls are selected by the same expected-shape raw selector; control rows are deterministically capped only for plotting cost/readability.",
    }
    write_json(out_dir / "run_config.json", config)
    lines = [
        "# Actual-Data Post-Selection Inspection Plots",
        "",
        "These plots show normalized raw lower-V power samples, not fit scores.",
        "",
        "Two plots are generated per source:",
        "",
        "- `*_real_all_vs_selected.png`: compares all usable real windows with the subset chosen by the raw expected-shape selector.",
        "- `*_selected_real_vs_selected_controls.png`: compares selected real windows with selected time-shift, randomized-time, and off-source control windows.",
        "",
        "Interpretation: if the selected controls look similar to selected real events, then a source-like selected stack is not independent detection evidence. It means the selection rule can find that morphology in wrong-time or wrong-sky data too.",
        "",
        "Control rows were deterministically capped per source/frequency/event type for plotting cost and readability. The summary tables in the parent audit use the full control counts.",
        "",
        "Generated plots:",
        "",
    ]
    for path in plot_paths:
        lines.append(f"- `{path.name}`")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote actual-data plots to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
