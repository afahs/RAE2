#!/usr/bin/env python
"""Compare low-frequency Earth/Sun extraction methods without SNR metrics."""

from __future__ import annotations

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

from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402


EXPECTED_SIGN = {"disappearance": -1.0, "reappearance": 1.0}
LOW_FREQS = [0.45, 0.70, 0.90, 1.31, 2.20]


def _read(path: Path) -> pd.DataFrame:
    return read_table(path, low_memory=False)


def _contrast_from_summary(summary: pd.DataFrame, value_col: str, method: str) -> pd.DataFrame:
    rows = []
    pre = (-180.0, -60.0)
    post = (60.0, 180.0)
    keys = ["source_name", "frequency_band", "frequency_mhz", "antenna", "event_type"]
    for vals_key, grp in summary.groupby(keys, sort=True, dropna=False):
        before = grp[(grp["t_bin_sec"] >= pre[0]) & (grp["t_bin_sec"] <= pre[1])][value_col]
        after = grp[(grp["t_bin_sec"] >= post[0]) & (grp["t_bin_sec"] <= post[1])][value_col]
        if before.empty or after.empty:
            continue
        event_type = str(vals_key[-1])
        delta = float(np.nanmedian(after) - np.nanmedian(before))
        rows.append(
            {
                **dict(zip(keys, vals_key)),
                "method": method,
                "post_minus_pre": delta,
                "source_like_contrast": float(EXPECTED_SIGN[event_type] * delta),
            }
        )
    return pd.DataFrame(rows)


def load_method_contrasts() -> pd.DataFrame:
    parts = []
    for source in ["earth", "sun"]:
        raw = _read(ROOT / f"outputs/all_frequency_profile_grids_v1/{source}_all_frequency_profile_summary_900s.csv")
        raw = raw[raw["antenna"].eq("rv2_coarse")].copy()
        parts.append(_contrast_from_summary(raw, "median_z_power", "raw_lower_v"))
        for label in ["linear", "quadratic"]:
            path = ROOT / f"outputs/{source}_{label}_detrend_profiles_v1/{source}_{label}_residual_prepost_contrast.csv"
            if path.exists():
                df = _read(path)
                df = df[df["antenna"].eq("rv2_coarse")].copy()
                df = df.rename(columns={"source_like_residual_contrast": "source_like_contrast"})
                df["source_name"] = source
                parts.append(
                    df[
                        [
                            "source_name",
                            "method",
                            "frequency_band",
                            "frequency_mhz",
                            "antenna",
                            "event_type",
                            "post_minus_pre_residual",
                            "source_like_contrast",
                        ]
                    ].rename(columns={"post_minus_pre_residual": "post_minus_pre"})
                )
    shift = ROOT / "outputs/shift_control_background_profiles_v1/shift_control_corrected_prepost_contrast_all_sources.csv"
    if shift.exists():
        df = _read(shift)
        df = df[df["antenna"].eq("rv2_coarse")].copy()
        parts.append(
            df[
                [
                    "source_name",
                    "method",
                    "frequency_band",
                    "frequency_mhz",
                    "antenna",
                    "event_type",
                    "post_minus_pre",
                    "source_like_contrast",
                ]
            ]
        )
    common = ROOT / "outputs/antenna_common_mode_profiles_v1/lower_v_common_mode_prepost_contrast_all_sources.csv"
    if common.exists():
        df = _read(common)
        df["antenna"] = "rv2_coarse"
        parts.append(
            df[
                [
                    "source_name",
                    "method",
                    "frequency_band",
                    "frequency_mhz",
                    "antenna",
                    "event_type",
                    "post_minus_pre",
                    "source_like_contrast",
                ]
            ]
        )
    geom = ROOT / "outputs/opposite_event_geometry_background_v1/geometry_background_prepost_contrasts.csv"
    if geom.exists():
        df = _read(geom)
        df["antenna"] = "rv2_coarse"
        parts.append(
            df[
                [
                    "source_name",
                    "method",
                    "frequency_mhz",
                    "antenna",
                    "event_type",
                    "post_minus_pre",
                    "source_like_contrast",
                ]
            ]
        )
    return pd.concat(parts, ignore_index=True, sort=False)


def plot_heatmaps(contrasts: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    methods = [
        "raw_lower_v",
        "pre_average_linear",
        "pre_average_quadratic",
        "shift_control_subtracted",
        "upper_v_common_mode_subtracted",
        "geometry_opposite_event_subtracted",
    ]
    for source in ["earth", "sun"]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)
        for ax, event_type in zip(axes, ["disappearance", "reappearance"]):
            sub = contrasts[
                contrasts["source_name"].eq(source)
                & contrasts["event_type"].eq(event_type)
                & contrasts["frequency_mhz"].isin(LOW_FREQS)
                & contrasts["method"].isin(methods)
            ].copy()
            pivot = sub.pivot_table(index="method", columns="frequency_mhz", values="source_like_contrast", aggfunc="first")
            pivot = pivot.reindex(methods)
            pivot = pivot[[freq for freq in LOW_FREQS if freq in pivot.columns]]
            data = pivot.to_numpy(dtype=float)
            vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
            vmax = max(vmax, 0.25)
            im = ax.imshow(data, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ax.set_title(event_type)
            ax.set_xticks(np.arange(len(pivot.columns)), [f"{float(x):.2f}" for x in pivot.columns], rotation=45, ha="right")
            ax.set_yticks(np.arange(len(pivot.index)), [str(x).replace("_", " ") for x in pivot.index])
            ax.set_xlabel("MHz")
            for y in range(data.shape[0]):
                for x in range(data.shape[1]):
                    if np.isfinite(data[y, x]):
                        ax.text(x, y, f"{data[y, x]:.2f}", ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="source-like contrast")
        fig.suptitle(f"{source.title()} lower-V low-frequency extraction method comparison")
        fig.tight_layout()
        path = out_dir / f"{source}_lowfreq_method_contrast_heatmap.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def write_report(contrasts: pd.DataFrame, paths: list[Path], out_dir: Path) -> None:
    low = contrasts[
        contrasts["source_name"].isin(["earth", "sun"])
        & contrasts["frequency_mhz"].isin(LOW_FREQS)
        & contrasts["antenna"].eq("rv2_coarse")
    ].copy()
    lines = [
        "# Low-Frequency Extraction Method Comparison",
        "",
        "This report compares profile morphology using source-like pre/post contrast, not SNR.",
        "Positive values mean the median profile moves in the expected occultation direction:",
        "down across disappearance and up across reappearance.",
        "",
        "Methods compared:",
        "",
        "- raw lower V normalized profile;",
        "- event-level linear polynomial residual;",
        "- event-level quadratic polynomial residual;",
        "- nearby time-shift control subtraction;",
        "- upper-V common-mode subtraction.",
        "- opposite-event geometry-background subtraction.",
        "",
        "## Low-Frequency Table",
        "",
        low[
            [
                "source_name",
                "method",
                "frequency_mhz",
                "event_type",
                "source_like_contrast",
            ]
        ].sort_values(["source_name", "frequency_mhz", "event_type", "method"]).to_string(index=False),
        "",
        "## Current Read",
        "",
        "- Earth 0.45 MHz remains the control that looks most like a clean occulted-source profile in raw and shift-control views.",
        "- Shift-control subtraction preserves Earth 0.45 and improves Earth 0.70, but does not recover Earth 0.90/1.31/2.20.",
        "- Upper-V common-mode subtraction suppresses several reversals, but it also suppresses Earth 0.45, so it is too aggressive as a production correction.",
        "- Opposite-event geometry-background subtraction is currently the most promising recovery method for the requested hard bins.",
        "- It recovers Earth 0.70 and 0.90 MHz in both event types, and moves Earth 1.31 disappearance just positive.",
        "- It recovers solar 0.45, 0.70, and 1.31 MHz reappearance, which were persistent blockers in earlier methods.",
        "- Earth 2.20 MHz and solar 2.20 MHz remain unresolved.",
        "",
        "## Plots",
        "",
    ]
    lines.extend(f"- `{path}`" for path in paths)
    (out_dir / "lowfreq_extraction_method_comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = ensure_dir(ROOT / "outputs/lowfreq_extraction_method_comparison_v1")
    write_json(out_dir / "run_config.json", {"software_versions": software_versions()})
    contrasts = load_method_contrasts()
    contrasts.to_csv(out_dir / "lowfreq_method_contrasts.csv", index=False)
    paths = plot_heatmaps(contrasts, out_dir)
    write_report(contrasts, paths, out_dir)
    print(out_dir / "lowfreq_extraction_method_comparison_report.md")


if __name__ == "__main__":
    main()
