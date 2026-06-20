#!/usr/bin/env python
"""Build compact summary plots for the latitude morphology audit."""

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

from scripts.build_raw_ecliptic_control_visuals import EVENT_TYPES, FREQS_MHZ, _format_freq  # noqa: E402


OUT = ROOT / "outputs/latitude_morphology_audit_v1"


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return read_table(path, low_memory=False)


def _plot_summary_lines(summary: pd.DataFrame, groups: list[str], title: str, out_path: Path) -> Path:
    fig, axes = plt.subplots(len(FREQS_MHZ), 2, figsize=(12.0, 1.85 * len(FREQS_MHZ)), sharex=True, sharey=True)
    cmap = plt.get_cmap("tab10")
    colors = {group: cmap(i % 10) for i, group in enumerate(groups)}
    for row, freq in enumerate(FREQS_MHZ):
        for col, event_type in enumerate(EVENT_TYPES):
            ax = axes[row, col]
            for group in groups:
                g = summary[
                    np.isclose(summary["frequency_mhz"], freq)
                    & summary["event_type"].astype(str).eq(event_type)
                    & summary["plot_group"].astype(str).eq(group)
                ].sort_values("t_bin_sec")
                if g.empty:
                    continue
                ax.plot(g["t_bin_sec"] / 60.0, g["median_normalized_power"], lw=1.25, color=colors[group], label=group)
            ax.axvline(0.0, color="black", lw=0.8, ls="--")
            ax.axhline(0.0, color="0.55", lw=0.6)
            ax.grid(alpha=0.18)
            if row == 0:
                ax.set_title(event_type)
            if col == 0:
                ax.set_ylabel(f"{freq:.2f} MHz\nnorm. power")
            if row == len(FREQS_MHZ) - 1:
                ax.set_xlabel("minutes from predicted event")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncols=min(4, len(groups)), fontsize=7, frameon=False)
    fig.suptitle(title, y=1.02, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _visual_index(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, grp in summary.groupby(["plot_group", "frequency_mhz", "event_type"], sort=True):
        group, freq, event_type = keys
        pre = grp[(grp["t_bin_sec"] >= -180) & (grp["t_bin_sec"] <= -60)]["median_normalized_power"].median()
        post = grp[(grp["t_bin_sec"] >= 60) & (grp["t_bin_sec"] <= 180)]["median_normalized_power"].median()
        rows.append({"plot_group": group, "frequency_mhz": freq, "event_type": event_type, "post_minus_pre": post - pre})
    return pd.DataFrame(rows)


def _plot_visual_index(gal_index: pd.DataFrame, point_index: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, len(FREQS_MHZ), figsize=(15.5, 6.0), sharey=True)
    gal = gal_index[gal_index["plot_group"].str.startswith("Galactic b=", na=False)].copy()
    gal["galactic_b_bin_deg"] = gal["plot_group"].str.extract(r"([+-]\d+) deg", expand=False).astype(float)
    for col, freq in enumerate(FREQS_MHZ):
        ax = axes[0, col]
        sub = gal[np.isclose(gal["frequency_mhz"], freq)]
        for event_type, color in [("disappearance", "#4c78a8"), ("reappearance", "#d95f02")]:
            g = sub[sub["event_type"].eq(event_type)].sort_values("galactic_b_bin_deg")
            ax.plot(g["galactic_b_bin_deg"], g["post_minus_pre"], marker="o", lw=1.2, color=color, label=event_type)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.axvline(0.0, color="0.7", lw=0.6)
        ax.set_title(f"{freq:.2f} MHz")
        ax.set_xlabel("Galactic b control bin")
        ax.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Galactic controls\npost - pre")
    point_names = [x for x in ["earth moving track", "sun moving track", "tau_a", "vir_a", "sgr_a", "galactic_center", "fornax a fixed source"]]
    name_map = {name.lower(): name for name in point_index["plot_group"].dropna().unique()}
    for col, freq in enumerate(FREQS_MHZ):
        ax = axes[1, col]
        sub = point_index[np.isclose(point_index["frequency_mhz"], freq)].copy()
        groups = [name_map[x] for x in point_names if x in name_map]
        x = np.arange(len(groups))
        width = 0.38
        for i, event_type in enumerate(EVENT_TYPES):
            vals = []
            for group in groups:
                v = sub[sub["plot_group"].eq(group) & sub["event_type"].eq(event_type)]["post_minus_pre"]
                vals.append(float(v.iloc[0]) if not v.empty else np.nan)
            ax.bar(x + (i - 0.5) * width, vals, width=width, label=event_type, alpha=0.85)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=65, ha="right", fontsize=7)
        ax.grid(alpha=0.18, axis="y")
    axes[1, 0].set_ylabel("Point sources\npost - pre")
    axes[0, -1].legend(frameon=False, fontsize=7)
    axes[1, -1].legend(frameon=False, fontsize=7)
    fig.suptitle("Visual before/after index from normalized median profiles")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = OUT / "latitude_and_point_source_visual_index.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _write_report(paths: list[Path], gal_index: pd.DataFrame, point_index: pd.DataFrame) -> Path:
    coord = _read(OUT / "fixed_source_coordinate_audit.csv")
    near = _read(OUT / "near_ecliptic_point_sources.csv")
    # Compact tables focused on the frequency range where the effect is visually strongest.
    focus_freqs = [0.90, 1.31, 2.20]
    gal_focus = gal_index[gal_index["frequency_mhz"].isin(focus_freqs)].sort_values(["frequency_mhz", "event_type", "plot_group"])
    point_focus = point_index[point_index["frequency_mhz"].isin(focus_freqs)].sort_values(["frequency_mhz", "event_type", "plot_group"])
    lines = [
        "# Latitude Morphology Audit Summary",
        "",
        "This report uses compact median-profile plots rather than dense sample-cloud plots to avoid large output files.",
        "All curves are locally normalized per event window and use lower V only.",
        "",
        "## Fixed Source Coordinate Check",
        "",
        coord.to_string(index=False) if not coord.empty else "Coordinate table missing.",
        "",
        "Near-ecliptic point sources tested:",
        "",
        near[["source_name", "ecliptic_lat_deg", "galactic_b_deg"]].to_string(index=False) if not near.empty else "None.",
        "",
        "## Visual Index Tables",
        "",
        "The index below is only a visual descriptor: median normalized power after the event minus before the event.",
        "It is not an SNR or detection score.",
        "",
        "### Galactic Latitude Controls",
        "",
        gal_focus.to_string(index=False),
        "",
        "### Near-Ecliptic Point Sources",
        "",
        point_focus.to_string(index=False),
        "",
        "## Interpretation",
        "",
        "- The Galactic-latitude sweep tests whether the Earth/Sun-like morphology is really tied to low Galactic latitude and diffuse Galactic background.",
        "- The near-ecliptic fixed point-source test separates ecliptic latitude from source class. `vir_a` is useful because it is near the ecliptic but very high Galactic latitude.",
        "- If low Galactic latitude controls resemble Earth/Sun more than high-latitude controls, that supports a diffuse-background replacement interpretation.",
        "- If `vir_a` resembles Earth/Sun despite high Galactic latitude, that supports ecliptic/lunar observing geometry more strongly.",
        "- If neither fixed controls nor near-ecliptic point sources match Earth/Sun cleanly, moving-body time selection remains important.",
        "",
        "## Generated Figures",
        "",
    ]
    lines.extend(f"- `{p}`" for p in paths)
    path = OUT / "latitude_morphology_audit_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    gal_summary = _read(OUT / "galactic_latitude_normalized_profile_summary.csv")
    point_summary = _read(OUT / "near_ecliptic_point_source_normalized_profile_summary.csv")
    if gal_summary.empty or point_summary.empty:
        raise SystemExit("Required summary CSVs are missing. Run run_latitude_morphology_audit.py first.")
    gal_groups = ["Earth moving track", "Sun moving track"] + [f"Galactic b={b:+03d} deg" for b in [-60, -30, -10, 0, 10, 30, 60]] + ["Fornax A fixed source"]
    point_groups = ["Earth moving track", "Sun moving track", "tau_a", "vir_a", "sgr_a", "galactic_center", "Fornax A fixed source"]
    paths = [
        _plot_summary_lines(gal_summary, gal_groups, "Galactic latitude control morphology", OUT / "galactic_latitude_summary_profiles.png"),
        _plot_summary_lines(point_summary, point_groups, "Near-ecliptic point-source morphology", OUT / "near_ecliptic_point_source_summary_profiles.png"),
    ]
    gal_index = _visual_index(gal_summary)
    point_index = _visual_index(point_summary)
    gal_index.to_csv(OUT / "galactic_latitude_visual_index.csv", index=False)
    point_index.to_csv(OUT / "near_ecliptic_point_source_visual_index.csv", index=False)
    paths.append(_plot_visual_index(gal_index, point_index))
    report = _write_report(paths, gal_index, point_index)
    print(report)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
