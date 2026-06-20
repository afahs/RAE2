#!/usr/bin/env python
"""Build source-level inspection plots from confirmation-suite outputs."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "inspection_plots"


SOURCE_SUITES = {
    "sun": ROOT / "outputs" / "solar_detection_confirmation_suite_v3",
    "jupiter": ROOT / "outputs" / "jupiter_detection_confirmation_suite",
    "fornax_a": ROOT / "outputs" / "fornax_a_detection_confirmation_suite",
    "cas_a": ROOT / "outputs" / "cas_a_detection_confirmation_suite",
    "cyg_a": ROOT / "outputs" / "cyg_a_detection_confirmation_suite",
}


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return read_table(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _clean_name(name: str) -> str:
    return name.replace("_", " ").title().replace("Cyg A", "Cyg A").replace("Cas A", "Cas A")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _bar_labels(ax: plt.Axes, values: list[float]) -> None:
    for i, val in enumerate(values):
        if not np.isfinite(val):
            continue
        va = "bottom" if val >= 0 else "top"
        offset = 0.03 * max(max(abs(v) for v in values if np.isfinite(v)), 1.0)
        ax.text(i, val + (offset if val >= 0 else -offset), f"{val:.2f}", ha="center", va=va, fontsize=8)


def _plot_timing(source: str, suite: Path, out_dir: Path) -> Path | None:
    df = _read(suite / "timing_scan.csv")
    if df.empty:
        return None
    df = df.sort_values("timing_offset_s")
    best = df.iloc[df["stacked_snr"].abs().argmax()]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["timing_offset_s"], df["stacked_snr"], marker="o", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", label="predicted time")
    ax.axvline(best["timing_offset_s"], color="crimson", linewidth=1.0, linestyle=":", label=f"best {best['timing_offset_s']:.0f}s")
    ax.set_title(f"{_clean_name(source)} timing-offset scan")
    ax.set_xlabel("Template timing offset (s)")
    ax.set_ylabel("Stacked SNR")
    ax.legend(loc="best")
    path = out_dir / source / f"{source}_timing_scan.png"
    _save(fig, path)
    return path


def _plot_window(source: str, suite: Path, out_dir: Path) -> Path | None:
    df = _read(suite / "window_sweep.csv")
    if df.empty:
        return None
    df = df.sort_values("window_s")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["window_s"], df["stacked_snr"], marker="o", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{_clean_name(source)} window-size repeatability")
    ax.set_xlabel("Window half-width (s)")
    ax.set_ylabel("Stacked SNR")
    path = out_dir / source / f"{source}_window_sweep.png"
    _save(fig, path)
    return path


def _plot_event_type(source: str, suite: Path, out_dir: Path) -> Path | None:
    df = _read(suite / "event_type.csv")
    if df.empty:
        return None
    labels = df["event_type"].astype(str).tolist()
    vals = pd.to_numeric(df["stacked_snr"], errors="coerce").to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#4c78a8" if v >= 0 else "#d95f02" for v in vals]
    ax.bar(labels, vals, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    _bar_labels(ax, list(vals))
    ax.set_title(f"{_clean_name(source)} event-type split")
    ax.set_ylabel("Stacked SNR")
    path = out_dir / source / f"{source}_event_type_split.png"
    _save(fig, path)
    return path


def _plot_quality_survival(source: str, suite: Path, out_dir: Path) -> Path | None:
    full = _read(suite / "window_sweep.csv")
    strict = _read(suite / "strict_quality.csv")
    if full.empty:
        return None
    best = full.iloc[full["stacked_snr"].abs().argmax()]
    vals = [float(best["stacked_snr"])]
    labels = ["full"]
    if not strict.empty and "stacked_snr" in strict:
        vals.append(float(strict.iloc[0]["stacked_snr"]))
        labels.append("strict quality")
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#4c78a8" if v >= 0 else "#d95f02" for v in vals]
    ax.bar(labels, vals, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    _bar_labels(ax, vals)
    ax.set_title(f"{_clean_name(source)} quality survival")
    ax.set_ylabel("Stacked SNR")
    path = out_dir / source / f"{source}_quality_survival.png"
    _save(fig, path)
    return path


def _plot_wrong_controls(source: str, suite: Path, out_dir: Path) -> Path | None:
    target = _read(suite / "window_sweep.csv")
    controls = _read(suite / "wrong_controls.csv")
    if target.empty and controls.empty:
        return None
    rows = []
    if not target.empty:
        best = target.iloc[target["stacked_snr"].abs().argmax()]
        rows.append({"label": "target", "stacked_snr": float(best["stacked_snr"])})
    for _, row in controls.iterrows():
        label = f"{row.get('control_type', 'control')}\\nB{int(row['frequency_band'])} {row['antenna']}"
        rows.append({"label": label, "stacked_snr": float(row["stacked_snr"])})
    df = pd.DataFrame(rows)
    vals = df["stacked_snr"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(df)), 4))
    colors = ["#4c78a8"] + ["#999999"] * max(len(df) - 1, 0)
    ax.bar(df["label"], vals, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    _bar_labels(ax, list(vals))
    ax.set_title(f"{_clean_name(source)} target vs wrong controls")
    ax.set_ylabel("Stacked SNR")
    ax.tick_params(axis="x", labelrotation=30)
    path = out_dir / source / f"{source}_wrong_controls.png"
    _save(fig, path)
    return path


def _plot_leave_one_month(source: str, suite: Path, out_dir: Path) -> Path | None:
    df = _read(suite / "leave_one_month.csv")
    if df.empty:
        return None
    df = df.sort_values("left_out_month")
    fig, ax = plt.subplots(figsize=(8, 4))
    vals = pd.to_numeric(df["stacked_snr"], errors="coerce").to_numpy(dtype=float)
    ax.bar(df["left_out_month"].astype(str), vals, color=["#4c78a8" if v >= 0 else "#d95f02" for v in vals])
    ax.axhline(0, color="black", linewidth=0.8)
    if "full_stacked_snr" in df:
        ax.axhline(float(df["full_stacked_snr"].iloc[0]), color="crimson", linestyle="--", linewidth=1.0, label="full stack")
        ax.legend(loc="best")
    ax.set_title(f"{_clean_name(source)} leave-one-month-out")
    ax.set_xlabel("Month removed")
    ax.set_ylabel("Stacked SNR")
    ax.tick_params(axis="x", labelrotation=45)
    path = out_dir / source / f"{source}_leave_one_month.png"
    _save(fig, path)
    return path


def _plot_earth(out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    earth_dir = out_dir / "earth"
    stack = _read(ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/04_stack/stacked_detection_summary.csv")
    if not stack.empty:
        top = stack[stack["source_name"].eq("earth")].copy()
        top = top.assign(abs_snr=top["stacked_snr"].abs()).sort_values("abs_snr", ascending=False).head(10)
        labels = [f"B{int(r.frequency_band)}\\n{r.frequency_mhz:.2f} MHz\\n{r.antenna}" for r in top.itertuples()]
        vals = top["stacked_snr"].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(labels, vals, color=["#4c78a8" if v >= 0 else "#d95f02" for v in vals])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Earth positive-control strongest stacked channels")
        ax.set_ylabel("Stacked SNR")
        ax.tick_params(axis="x", labelrotation=30)
        path = earth_dir / "earth_top_stacks.png"
        _save(fig, path)
        paths.append(path)
    summary = _read(ROOT / "outputs/reports/earth_positive_control_assets/earth_band7_rv2_coarse_event_type_summary.csv")
    if not summary.empty:
        labels = summary["event_type"].astype(str).tolist()
        vals = summary["stacked_snr"].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, vals, color=["#4c78a8" if v >= 0 else "#d95f02" for v in vals])
        ax.axhline(0, color="black", linewidth=0.8)
        _bar_labels(ax, list(vals))
        ax.set_title("Earth band 7 rv2_coarse event-type split")
        ax.set_ylabel("Stacked SNR")
        path = earth_dir / "earth_event_type_split.png"
        _save(fig, path)
        paths.append(path)
    src = ROOT / "outputs/reports/earth_positive_control_assets/earth_band7_rv2_coarse_event_type_stack.png"
    stack_profile = _read(ROOT / "outputs/reports/earth_positive_control_assets/earth_band7_rv2_coarse_event_type_stack.csv")
    if not stack_profile.empty:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        colors = {"disappearance": "#4c78a8", "reappearance": "#d95f02"}
        for event_type, grp in stack_profile.groupby("event_type", sort=True):
            grp = grp.sort_values("t_bin_sec")
            ax.plot(
                grp["t_bin_sec"] / 60.0,
                grp["mean"],
                marker="o",
                ms=3,
                lw=1.8,
                color=colors.get(str(event_type)),
                label=f"{event_type} ({int(grp['n_events'].max())} events)",
            )
            if "sem" in grp:
                sem = grp["sem"].to_numpy(dtype=float)
                ax.fill_between(
                    grp["t_bin_sec"] / 60.0,
                    grp["mean"] - sem,
                    grp["mean"] + sem,
                    color=colors.get(str(event_type)),
                    alpha=0.18,
                    linewidth=0,
                )
        ax.axvline(0.0, color="black", lw=1.0, label="predicted occultation time")
        ax.axhline(0.0, color="0.35", lw=0.8)
        ax.set_title("Earth 4.70 MHz lower V stacked occultation profile")
        ax.set_xlabel("Relative time from predicted event (min)")
        ax.set_ylabel("Mean normalized, baseline-subtracted power")
        ax.legend(fontsize=8)
        dest = earth_dir / "earth_band7_rv2_coarse_event_type_profile.png"
        _save(fig, dest)
        paths.append(dest)
    elif src.exists():
        dest = earth_dir / "earth_band7_rv2_coarse_event_type_profile.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        paths.append(dest)
    return paths


def _plot_summary(out_dir: Path) -> Path | None:
    df = _read(ROOT / "outputs/summary/source_detection_technique_summary.csv")
    if df.empty:
        return None
    final = _read(ROOT / "outputs/summary/source_final_summary.csv")
    rows = []
    for _, row in df.iterrows():
        source = row["source"]
        snr = np.nan
        if not final.empty:
            key = str(source).lower().replace(" ", "_")
            match = final[final["source"].eq(key)]
            if not match.empty:
                snr = float(match.iloc[0]["strongest_stacked_snr"])
        rows.append({"source": source, "strongest_stacked_snr": snr, "status": row["final_status"]})
    plot = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 4))
    vals = plot["strongest_stacked_snr"].to_numpy(dtype=float)
    ax.bar(plot["source"], vals, color=["#4c78a8" if v >= 0 else "#d95f02" for v in vals])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Source summary: strongest existing stacked SNR")
    ax.set_ylabel("Stacked SNR")
    ax.tick_params(axis="x", labelrotation=25)
    path = out_dir / "source_summary_strongest_snr.png"
    _save(fig, path)
    return path


def _build_index(paths_by_source: dict[str, list[Path]], summary_path: Path | None) -> None:
    lines = [
        "# Source Inspection Plots",
        "",
        "Plots are generated from the confirmation-suite CSV outputs and existing Earth positive-control outputs.",
        "",
    ]
    if summary_path is not None:
        lines += ["## Cross-Source Summary", "", f"- [{summary_path.name}]({summary_path.relative_to(OUT)})", ""]
    for source, paths in paths_by_source.items():
        lines += [f"## {_clean_name(source)}", ""]
        if not paths:
            lines.append("No plots generated.")
        for path in paths:
            rel = path.relative_to(OUT)
            lines.append(f"- [{path.name}]({rel})")
        lines.append("")
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths_by_source: dict[str, list[Path]] = {}
    paths_by_source["earth"] = _plot_earth(OUT)
    for source, suite in SOURCE_SUITES.items():
        paths: list[Path] = []
        for func in [
            _plot_timing,
            _plot_window,
            _plot_event_type,
            _plot_quality_survival,
            _plot_wrong_controls,
            _plot_leave_one_month,
        ]:
            path = func(source, suite, OUT)
            if path is not None:
                paths.append(path)
        paths_by_source[source] = paths
    summary_path = _plot_summary(OUT)
    _build_index(paths_by_source, summary_path)
    print(OUT / "README.md")


if __name__ == "__main__":
    main()
