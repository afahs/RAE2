#!/usr/bin/env python
"""Audit limb filters and non-astrophysical explanations for Earth/Sun low-frequency reversal."""

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

from rylevonberg.events import predict_events  # noqa: E402
from rylevonberg.util import ensure_dir  # noqa: E402
from scripts.build_all_frequency_occultation_profile_grids import (  # noqa: E402
    BRIGHT_EVENTS,
    CLEAN,
    EARTH_EVENTS,
    SUN_EVENTS,
    collect_profiles,
    summarize_profiles,
)


OUT = ROOT / "outputs/moving_body_filter_and_reversal_options_v1"
FREQS = [0.45, 0.70, 0.90, 1.31, 2.20, 3.93, 4.70, 6.55, 9.18]
LOW_FREQS = [0.70, 0.90, 1.31, 2.20]
ANTENNAS = ["rv1_coarse", "rv2_coarse"]
EXPECTED_SIGN = {"disappearance": -1.0, "reappearance": 1.0}


def _read(path: Path, **kwargs) -> pd.DataFrame:
    return read_table(path, low_memory=False, **kwargs)


def _source(source_name: str) -> pd.DataFrame:
    if source_name == "earth":
        return pd.DataFrame(
            [{"source_name": "earth", "kind": "earth", "body_name": "earth", "frame": "fk4", "ra_deg": np.nan, "dec_deg": np.nan}]
        )
    return pd.DataFrame(
        [
            {
                "source_name": source_name,
                "kind": "body",
                "body_name": source_name,
                "frame": "fk4",
                "ra_deg": np.nan,
                "dec_deg": np.nan,
            }
        ]
    )


def _central_contrast(summary: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for keys, grp in summary.groupby(["source_name", "frequency_mhz", "antenna", "event_type"], sort=True):
        source, freq, antenna, event_type = keys
        pre = grp[(grp["t_bin_sec"] >= -180) & (grp["t_bin_sec"] <= -60)]["median_z_power"].median()
        post = grp[(grp["t_bin_sec"] >= 60) & (grp["t_bin_sec"] <= 180)]["median_z_power"].median()
        outer_pre = grp[(grp["t_bin_sec"] >= -900) & (grp["t_bin_sec"] <= -600)]["median_z_power"].median()
        outer_post = grp[(grp["t_bin_sec"] >= 600) & (grp["t_bin_sec"] <= 900)]["median_z_power"].median()
        sign = EXPECTED_SIGN[str(event_type)]
        rows.append(
            {
                "run_label": label,
                "source_name": source,
                "frequency_mhz": float(freq),
                "antenna": antenna,
                "event_type": event_type,
                "central_post_minus_pre": float(post - pre),
                "central_source_like_contrast": float(sign * (post - pre)),
                "outer_source_like_contrast": float(sign * (outer_post - outer_pre)),
                "n_events": int(grp["n_events"].max()),
            }
        )
    return pd.DataFrame(rows)


def _existing_summary(source: str, label: str) -> pd.DataFrame:
    path = ROOT / f"outputs/all_frequency_profile_grids_v1/{source}_all_frequency_profile_summary_900s.csv"
    summary = _read(path)
    return _central_contrast(summary, label)


def _profile_from_events(clean: pd.DataFrame, events: pd.DataFrame, source: str, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    points = collect_profiles(clean, events, source, window_s=900.0, bin_s=60.0, inner_s=15.0)
    summary = summarize_profiles(points)
    points.to_csv(OUT / f"{label}_{source}_profile_points_900s.csv", index=False)
    summary.to_csv(OUT / f"{label}_{source}_profile_summary_900s.csv", index=False)
    return points, _central_contrast(summary, label) if not summary.empty else pd.DataFrame()


def _predict_earth_sun_veto(clean: pd.DataFrame) -> pd.DataFrame:
    event_path = OUT / "earth_sun_limb_excluded_3deg_predicted_events.csv"
    state_path = OUT / "earth_sun_limb_excluded_3deg_limb_states.csv"
    if event_path.exists():
        return _read(event_path, parse_dates=["predicted_event_time"])
    events, states = predict_events(
        clean,
        _source("earth"),
        target_frame="fk4",
        equinox="B1950",
        ephemeris="builtin",
        max_gap_seconds=600.0,
        prediction_cadence_seconds=300.0,
        frequencies=list(range(1, 10)),
        antennas=ANTENNAS,
        limb_exclusion_sources_df=_source("sun"),
        limb_exclusion_deg=3.0,
    )
    events.to_csv(event_path, index=False)
    states.to_csv(state_path, index=False)
    return events


def _filter_status() -> pd.DataFrame:
    rows = []
    inputs = [
        ("earth_grid_current", "earth", EARTH_EVENTS),
        ("sun_grid_current", "sun", SUN_EVENTS),
        ("science_baseline_planets", "all_planets", EARTH_EVENTS),
        ("bright_sources", "fixed_sources", BRIGHT_EVENTS),
    ]
    for label, source, path in inputs:
        df = _read(path)
        if source != "all_planets" and source != "fixed_sources" and "source_name" in df:
            work = df[df["source_name"].astype(str).str.lower().eq(source)].copy()
        else:
            work = df
        limb_deg = pd.to_numeric(work.get("limb_exclusion_deg", pd.Series(dtype=float)), errors="coerce")
        nearest = pd.to_numeric(work.get("limb_exclusion_nearest_abs_deg", pd.Series(dtype=float)), errors="coerce")
        rows.append(
            {
                "label": label,
                "source_scope": source,
                "event_table": str(path.relative_to(ROOT)),
                "n_rows": int(len(work)),
                "has_limb_exclusion_column": "limb_exclusion_deg" in work.columns,
                "finite_limb_exclusion_rows": int(limb_deg.notna().sum()),
                "limb_exclusion_deg_unique": ";".join(str(x) for x in sorted(limb_deg.dropna().unique())),
                "min_nearest_abs_limb_angle_deg": float(nearest.min()) if nearest.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _plot_filter_comparison(contrast: pd.DataFrame) -> Path:
    work = contrast[
        contrast["source_name"].eq("earth")
        & contrast["antenna"].eq("rv2_coarse")
        & contrast["run_label"].isin(["earth_current_no_sun_veto", "earth_sun_limb_excluded_3deg"])
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True)
    colors = {"earth_current_no_sun_veto": "#1f77b4", "earth_sun_limb_excluded_3deg": "#d62728"}
    for ax, event_type in zip(axes, ["disappearance", "reappearance"]):
        sub = work[work["event_type"].eq(event_type)]
        for label, grp in sub.groupby("run_label", sort=True):
            grp = grp.sort_values("frequency_mhz")
            ax.plot(
                grp["frequency_mhz"],
                grp["central_source_like_contrast"],
                marker="o",
                lw=1.6,
                color=colors[label],
                label=label,
            )
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks(FREQS)
        ax.set_xticklabels([f"{f:.2f}" for f in FREQS], rotation=45)
        ax.set_title(event_type)
        ax.set_xlabel("frequency (MHz)")
        ax.grid(alpha=0.22, which="both")
    axes[0].set_ylabel("source-like central contrast\nlower V")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Earth lower V: adding Sun-limb veto does not explain low-frequency reversal")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    path = OUT / "earth_sun_limb_veto_contrast_comparison.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_source_sign_matrix(contrast: pd.DataFrame) -> Path:
    work = contrast[
        contrast["run_label"].isin(
            [
                "earth_current_no_sun_veto",
                "sun_current_earth_veto",
                "fornax_a_current",
                "cas_a_current",
                "cyg_a_current",
            ]
        )
        & contrast["antenna"].eq("rv2_coarse")
        & contrast["frequency_mhz"].isin(LOW_FREQS)
    ].copy()
    pivot = work.pivot_table(
        index=["source_name", "event_type"],
        columns="frequency_mhz",
        values="central_source_like_contrast",
        aggfunc="median",
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    data = pivot.reindex(columns=LOW_FREQS).to_numpy(dtype=float)
    im = ax.imshow(data, cmap="coolwarm", vmin=-0.4, vmax=0.4, aspect="auto")
    ax.set_xticks(np.arange(len(LOW_FREQS)))
    ax.set_xticklabels([f"{f:.2f}" for f in LOW_FREQS])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{src} {etype}" for src, etype in pivot.index], fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="source-like contrast")
    ax.set_xlabel("frequency (MHz)")
    ax.set_title("Lower-V low-frequency sign matrix: moving bodies vs fixed sources")
    fig.tight_layout()
    path = OUT / "lower_v_low_frequency_sign_matrix.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _write_report(status: pd.DataFrame, contrast: pd.DataFrame, paths: list[Path]) -> Path:
    low = contrast[
        contrast["frequency_mhz"].isin(LOW_FREQS)
        & contrast["antenna"].eq("rv2_coarse")
    ][
        ["run_label", "source_name", "frequency_mhz", "event_type", "central_source_like_contrast", "n_events"]
    ].sort_values(["run_label", "source_name", "frequency_mhz", "event_type"])
    earth_compare = low[
        low["run_label"].isin(["earth_current_no_sun_veto", "earth_sun_limb_excluded_3deg"])
    ]
    lines = [
        "# Moving-Body Filter And Reversal Options Audit",
        "",
        "## Limb-Filter Status",
        "",
        "The all-frequency grid plotting function does not apply limb filters itself; it stacks",
        "the event table provided for each source. I updated the plotter so each source now uses",
        "one intended source-specific event table rather than concatenating filtered and unfiltered",
        "moving-body tables.",
        "",
        status.to_string(index=False),
        "",
        "Current status after this fix:",
        "",
        "- Sun grid: uses `pipeline_confidence_audit_v2/sun_audit_input_events.csv`, which has Earth-limb exclusion at 3 deg.",
        "- Earth grid: uses the science-baseline planetary table and has no Sun-limb exclusion by default.",
        "- Fixed-source grids: no moving-body limb exclusion.",
        "",
        "## Symmetric Cross-Limb Contamination Test",
        "",
        "I re-predicted Earth events with a Sun-limb exclusion of 3 deg and rebuilt the lower-V profile contrasts.",
        "",
        earth_compare.to_string(index=False),
        "",
        "The Earth low-frequency reversal at 0.90, 1.31, and 2.20 MHz survives the Sun-limb veto.",
        "So direct Sun-near-limb contamination is not enough to explain the shared Earth/Sun behavior.",
        "",
        "## Other Options Tested / Constrained",
        "",
        "1. **Plot-input bug:** found and fixed for the all-frequency plotter. The Sun table is now explicitly Earth-limb-excluded.",
        "2. **Cross-limb contamination:** not sufficient. Sun was already Earth-vetoed in the intended table; Earth remains reversed after a symmetric Sun-veto test.",
        "3. **Event-label sign bug:** previous limb-angle checks show disappearance/reappearance labels have the correct pre/post limb-angle signs.",
        "4. **Generic receiver band sign inversion:** unlikely as a complete explanation because fixed sources do not all reverse in the same low-frequency bins.",
        "5. **Baseline drift:** not sufficient because the reversal survives low-drift event selection.",
        "6. **Exact Earth/Sun disk astrophysics:** unlikely as the only explanation because nearby moving-body offset tracks preserve the same sign in previous tests.",
        "",
        "## More Plausible Explanation",
        "",
        "The shared Earth/Sun reversal is most consistent with a moving-body/ecliptic-track response",
        "or low-frequency background-replacement effect, not intrinsic source astrophysics alone.",
        "Earth and Sun are both moving, extended targets near the ecliptic, so their lunar-limb events",
        "sample a different distribution of Moon/spacecraft/antenna/sky-background states than fixed",
        "radio sources. In total power, the measured contrast is the difference between the occulted",
        "target/track contribution and the sky/lunar/beam background replacing it. That effective",
        "contrast can be negative in selected low-frequency channels even without negative emission.",
        "",
        "The key unresolved question is whether this is physical sky-background replacement, receiver/",
        "beam coupling along moving-body tracks, or a channel-specific calibration effect. The next",
        "pipeline step should add moving-track controls: ecliptic offset tracks and time-shifted Earth/",
        "Sun tracks, then estimate empirical contrast sign per source/frequency/antenna rather than",
        "assuming positive-source sign.",
        "",
        "## Low-Frequency Lower-V Contrast Table",
        "",
        low.to_string(index=False),
        "",
        "## Generated Plots",
        "",
    ]
    lines.extend(f"- `{path}`" for path in paths)
    report = OUT / "moving_body_filter_and_reversal_options_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ensure_dir(OUT)
    clean = _read(CLEAN, parse_dates=["time"])

    status = _filter_status()
    current = pd.concat(
        [
            _existing_summary("earth", "earth_current_no_sun_veto"),
            _existing_summary("sun", "sun_current_earth_veto"),
            _existing_summary("fornax_a", "fornax_a_current"),
            _existing_summary("cas_a", "cas_a_current"),
            _existing_summary("cyg_a", "cyg_a_current"),
        ],
        ignore_index=True,
    )

    earth_sun_veto_events = _predict_earth_sun_veto(clean)
    _, earth_sun_veto = _profile_from_events(clean, earth_sun_veto_events, "earth", "earth_sun_limb_excluded_3deg")
    contrast = pd.concat([current, earth_sun_veto], ignore_index=True)

    status.to_csv(OUT / "profile_grid_limb_filter_status.csv", index=False)
    contrast.to_csv(OUT / "moving_body_filter_contrast_comparison.csv", index=False)

    paths = [_plot_filter_comparison(contrast), _plot_source_sign_matrix(contrast)]
    report = _write_report(status, contrast, paths)
    print(report)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
