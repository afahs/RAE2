#!/usr/bin/env python
"""Build event-by-event occultograms with fixed time-shift controls."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rylevonberg.occultogram import (  # noqa: E402
    OccultogramConfig,
    compact_occultogram_profiles,
    event_signed_contrasts,
    extract_occultogram_points,
    fold_occultogram_points,
    occultogram_event_metadata,
    summarize_occultogram,
)
from rylevonberg.table_io import output_table_path, read_table, resolve_table_path, write_table  # noqa: E402
from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402


INPUT_ROOT = ROOT / "runtime_inputs/occultogram_v1"
CLEAN = INPUT_ROOT / "cleaned_timeseries.parquet"
SUN_EVENTS = INPUT_ROOT / "sun_events.parquet"
BRIGHT_EVENTS = INPUT_ROOT / "bright_source_events.parquet"
PLANET_EVENTS = INPUT_ROOT / "planet_events_science_baseline.parquet"
OUTER_PLANET_EVENTS = INPUT_ROOT / "outer_planet_events.parquet"
VIRGO_EVENTS = INPUT_ROOT / "virgo_events.parquet"
DEFAULT_SOURCES = ("earth", "sun", "fornax_a", "vir_a")
PLANETS = {"mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"}
OUTER_PLANETS = {"saturn", "uranus", "neptune"}
BRIGHT_SOURCES = {"fornax_a", "cas_a", "cyg_a", "tau_a", "sgr_a", "galactic_center"}
EARTH_GALAXY_DOMINATED_MHZ = {0.70, 0.90, 1.31, 2.20}
SOURCE_LABEL = {
    "earth": "Earth",
    "sun": "Sun",
    "fornax_a": "Fornax A",
    "vir_a": "Virgo A",
    "cas_a": "Cassiopeia A",
    "cyg_a": "Cygnus A",
    "tau_a": "Taurus A",
    "jupiter": "Jupiter",
    "saturn": "Saturn",
    "uranus": "Uranus",
    "neptune": "Neptune",
}


def _read(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    resolved = resolve_table_path(path)
    if not resolved.exists() or resolved.stat().st_size == 0:
        return pd.DataFrame()
    return read_table(resolved, parse_dates=parse_dates, low_memory=False)


def _default_event_path(source: str) -> Path | None:
    if source == "sun":
        return SUN_EVENTS
    if source in OUTER_PLANETS:
        return OUTER_PLANET_EVENTS
    if source in PLANETS:
        return PLANET_EVENTS
    if source == "vir_a":
        return VIRGO_EVENTS
    if source in BRIGHT_SOURCES:
        return BRIGHT_EVENTS
    return None


def load_events_for_source(source: str, event_tables: list[Path] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Load source rows without mixing filtered and unfiltered event suites."""

    paths = list(event_tables or [])
    if not paths:
        default = _default_event_path(source)
        paths = [default] if default is not None else []
    frames = []
    used = []
    for path in paths:
        table = _read(path, parse_dates=["predicted_event_time"])
        if table.empty or "source_name" not in table:
            continue
        selected = table[table["source_name"].astype(str).str.lower().eq(source)].copy()
        if not selected.empty:
            frames.append(selected)
            used.append(str(path))
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), used


def _channel_title(source: str, frequency_mhz: float, antenna: str) -> str:
    label = SOURCE_LABEL.get(source, source.replace("_", " ").title())
    title = f"{label} — {frequency_mhz:.2f} MHz — {antenna}"
    if source == "earth" and any(np.isclose(frequency_mhz, value) for value in EARTH_GALAXY_DOMINATED_MHZ):
        title += "  [Galaxy-dominated band]"
    return title


def _heatmap_matrix(points: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    event_order = (
        points[["event_uid", "event_order"]]
        .drop_duplicates()
        .sort_values("event_order")["event_uid"]
        .tolist()
    )
    time_bins = np.sort(pd.to_numeric(points["t_bin_sec"], errors="coerce").dropna().unique())
    pivot = points.pivot_table(index="event_uid", columns="t_bin_sec", values="normalized_power", aggfunc="median")
    pivot = pivot.reindex(index=event_order, columns=time_bins)
    return pivot.to_numpy(dtype=float), time_bins


def _plot_channel_occultogram(
    folded: pd.DataFrame,
    summary: pd.DataFrame,
    source: str,
    frequency_mhz: float,
    antenna: str,
    output_path: Path,
) -> None:
    target = folded[np.isclose(folded["control_shift_s"], 0.0)].copy()
    matrix, time_bins = _heatmap_matrix(target)
    finite_abs = np.abs(matrix[np.isfinite(matrix)])
    color_limit = float(np.nanquantile(finite_abs, 0.98)) if finite_abs.size else 1.0
    color_limit = min(max(color_limit, 1.0), 6.0)

    fig, (heat_ax, profile_ax) = plt.subplots(
        2,
        1,
        figsize=(11.5, 7.0),
        gridspec_kw={"height_ratios": [2.2, 1.0]},
        sharex=True,
    )
    extent = [time_bins.min(), time_bins.max(), matrix.shape[0] - 0.5, -0.5]
    image = heat_ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-color_limit,
        vmax=color_limit,
        extent=extent,
    )
    heat_ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    heat_ax.set_ylabel("independent events\n(chronological; D/R folded)")
    heat_ax.set_title(_channel_title(source, frequency_mhz, antenna), loc="left", fontsize=11)
    colorbar = fig.colorbar(image, ax=heat_ax, pad=0.012, aspect=28)
    colorbar.set_label("outer-sideband normalized power\n(no detrending)")

    controls = sorted(pd.to_numeric(summary["control_shift_s"], errors="coerce").dropna().unique())
    for shift in controls:
        line = summary[np.isclose(summary["control_shift_s"], shift)].sort_values("t_bin_sec")
        if line.empty:
            continue
        is_target = np.isclose(shift, 0.0)
        profile_ax.plot(
            line["t_bin_sec"],
            line["median_normalized_power"],
            color="black" if is_target else "0.58",
            linewidth=2.0 if is_target else 0.9,
            alpha=1.0 if is_target else 0.72,
            linestyle="-" if is_target else "--",
            label="predicted time" if is_target else f"{shift / 60:+.0f} min control",
            zorder=3 if is_target else 1,
        )
        if is_target:
            use_blocks = pd.to_numeric(line["n_lunar_blocks"], errors="coerce").max() >= 3
            lower_column = "block_q16_normalized_power" if use_blocks else "event_q16_normalized_power"
            upper_column = "block_q84_normalized_power" if use_blocks else "event_q84_normalized_power"
            profile_ax.fill_between(
                line["t_bin_sec"].to_numpy(dtype=float),
                line[lower_column].to_numpy(dtype=float),
                line[upper_column].to_numpy(dtype=float),
                color="black",
                alpha=0.13,
                linewidth=0,
                label="lunar-block 16–84%" if use_blocks else "event 16–84%",
            )
    profile_ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    profile_ax.axhline(0.0, color="0.75", linewidth=0.8)
    profile_ax.set_xlabel("seconds from predicted/control alignment")
    profile_ax.set_ylabel("median normalized power")
    profile_ax.legend(frameon=False, fontsize=8, ncol=3, loc="best")
    fig.suptitle(
        "A localized source should form a repeatable vertical edge at zero in the predicted-time panel, "
        "with weaker or displaced structure in fixed-time controls.",
        fontsize=9.5,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_source_antenna_overview(
    folded: pd.DataFrame,
    summary: pd.DataFrame,
    source: str,
    antenna: str,
    output_path: Path,
) -> None:
    frequencies = sorted(pd.to_numeric(folded["frequency_mhz"], errors="coerce").dropna().unique())
    target_values = pd.to_numeric(
        folded.loc[np.isclose(folded["control_shift_s"], 0.0), "normalized_power"], errors="coerce"
    ).dropna()
    color_limit = float(np.nanquantile(np.abs(target_values), 0.98)) if not target_values.empty else 1.0
    color_limit = min(max(color_limit, 1.0), 6.0)
    fig, axes = plt.subplots(
        len(frequencies),
        2,
        figsize=(14.5, max(9.0, 1.8 * len(frequencies))),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.12, 1.0]},
    )
    last_image = None
    for row_index, frequency_mhz in enumerate(frequencies):
        heat_ax, profile_ax = axes[row_index]
        channel = folded[np.isclose(folded["frequency_mhz"], frequency_mhz)]
        target = channel[np.isclose(channel["control_shift_s"], 0.0)]
        channel_summary = summary[np.isclose(summary["frequency_mhz"], frequency_mhz)]
        matrix, time_bins = _heatmap_matrix(target)
        extent = [time_bins.min(), time_bins.max(), matrix.shape[0] - 0.5, -0.5]
        last_image = heat_ax.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=-color_limit,
            vmax=color_limit,
            extent=extent,
        )
        heat_ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
        foreground_note = " — Galaxy foreground" if source == "earth" and any(
            np.isclose(frequency_mhz, value) for value in EARTH_GALAXY_DOMINATED_MHZ
        ) else ""
        heat_ax.set_title(
            f"{frequency_mhz:.2f} MHz — {matrix.shape[0]} events{foreground_note}",
            loc="left",
            fontsize=8.5,
            color="#8c2d04" if foreground_note else "black",
        )
        heat_ax.set_yticks([])
        heat_ax.set_ylabel("events", fontsize=8)

        for shift in sorted(pd.to_numeric(channel_summary["control_shift_s"], errors="coerce").dropna().unique()):
            line = channel_summary[np.isclose(channel_summary["control_shift_s"], shift)].sort_values("t_bin_sec")
            is_target = np.isclose(shift, 0.0)
            profile_ax.plot(
                line["t_bin_sec"],
                line["median_normalized_power"],
                color="black" if is_target else "0.62",
                linewidth=1.6 if is_target else 0.65,
                alpha=1.0 if is_target else 0.68,
                linestyle="-" if is_target else "--",
                label="predicted" if is_target else (f"{shift / 60:+.0f} min" if row_index == 0 else None),
                zorder=3 if is_target else 1,
            )
            if is_target:
                use_blocks = pd.to_numeric(line["n_lunar_blocks"], errors="coerce").max() >= 3
                lower_column = "block_q16_normalized_power" if use_blocks else "event_q16_normalized_power"
                upper_column = "block_q84_normalized_power" if use_blocks else "event_q84_normalized_power"
                profile_ax.fill_between(
                    line["t_bin_sec"].to_numpy(dtype=float),
                    line[lower_column].to_numpy(dtype=float),
                    line[upper_column].to_numpy(dtype=float),
                    color="black",
                    alpha=0.11,
                    linewidth=0,
                )
        profile_ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
        profile_ax.axhline(0.0, color="0.78", linewidth=0.6)
        profile_ax.tick_params(labelsize=7.5)
        if row_index == 0:
            profile_ax.set_title("predicted median (black) vs fixed-time controls (gray)", loc="left", fontsize=8.5)
            profile_ax.legend(frameon=False, fontsize=7, ncol=5, loc="best")
        if row_index == len(frequencies) - 1:
            heat_ax.set_xlabel("seconds from predicted event")
            profile_ax.set_xlabel("seconds from predicted/control alignment")
        else:
            heat_ax.tick_params(labelbottom=False)
            profile_ax.tick_params(labelbottom=False)
        profile_ax.set_ylabel("median", fontsize=8)

    label = SOURCE_LABEL.get(source, source.replace("_", " ").title())
    fig.suptitle(
        f"{label} — {antenna} — morphology-first all-frequency occultograms\n"
        "Each heatmap row is an event; reappearances are time-reversed so a positive source is high-before/low-after. No detrending or SNR grading.",
        fontsize=11,
        y=0.998,
    )
    if last_image is not None:
        colorbar = fig.colorbar(last_image, ax=axes[:, 0].tolist(), pad=0.008, aspect=45, shrink=0.92)
        colorbar.set_label("normalized raw power", fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
    fig.subplots_adjust(left=0.055, right=0.965, top=0.955, bottom=0.045, hspace=0.34, wspace=0.18)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_occultograms(
    points: pd.DataFrame,
    output_dir: Path,
    channel_plots: bool = False,
    summary: pd.DataFrame | None = None,
) -> list[Path]:
    folded = fold_occultogram_points(points)
    summary = summarize_occultogram(folded) if summary is None or summary.empty else summary
    paths = []
    for (source, antenna), source_antenna in folded.groupby(["source_name", "antenna"], dropna=False, sort=True):
        source_summary = summary[
            summary["source_name"].astype(str).eq(str(source))
            & summary["antenna"].astype(str).eq(str(antenna))
        ]
        path = output_dir / f"{source}_{antenna}_all_frequency_occultograms.png"
        _plot_source_antenna_overview(source_antenna, source_summary, str(source), str(antenna), path)
        paths.append(path)
    if not channel_plots:
        return paths
    channel_columns = ["source_name", "frequency_band", "frequency_mhz", "antenna"]
    for keys, channel in folded.groupby(channel_columns, dropna=False, sort=True):
        source, band, frequency_mhz, antenna = keys
        channel_summary = summary[
            summary["source_name"].astype(str).eq(str(source))
            & summary["frequency_band"].eq(band)
            & np.isclose(summary["frequency_mhz"], float(frequency_mhz))
            & summary["antenna"].astype(str).eq(str(antenna))
        ]
        path = output_dir / f"{source}_band{int(band):02d}_{float(frequency_mhz):.2f}mhz_{antenna}_occultogram.png"
        _plot_channel_occultogram(channel, channel_summary, str(source), float(frequency_mhz), str(antenna), path)
        paths.append(path)
    return paths


def write_report(
    output_dir: Path,
    source_status: list[dict[str, object]],
    plot_paths: list[Path],
    config: OccultogramConfig,
) -> None:
    lines = [
        "# Morphology-First Occultograms",
        "",
        "This product replaces the old stacked-SNR/timing-scan figures as the primary source-detection review.",
        "It does not assign a detection SNR.",
        "",
        "## Reading the plots",
        "",
        "- Each heatmap row is one independent predicted event; disappearance and reappearance are folded to the same high-before/low-after source sign.",
        "- The dashed vertical line is the tested alignment. A source-like edge should repeat across event rows at zero.",
        "- The lower panel compares the predicted alignment with fixed ±10 and ±20 minute controls using the same data and normalization.",
        "- The black uncertainty region uses 29.53059-day lunar-block quantiles when at least three blocks are available; otherwise it uses event quantiles.",
        "- Profiles are raw apart from outer-sideband centering/scaling. No linear or diffuse-background trend is subtracted.",
        "- By default, finite samples are retained even if legacy amplitude/jump flags reject them, so a genuine sharp edge or solar burst is not silently censored.",
        "",
        "## Claim standard",
        "",
        "A detection claim should require a correctly signed, zero-localized edge visible across independent events; agreement between disappearance and folded reappearance; failure of fixed-time, wrong-antenna, and neighboring-frequency controls to reproduce it; and stability across lunar blocks. The event-level signed-contrast table is supporting bookkeeping, not a replacement for the images.",
        "",
        "For Earth, 0.70, 0.90, 1.31, and 2.20 MHz are marked Galaxy-dominated and should not be counted as separate Earth detections even if their profiles contain smooth structure.",
        "",
        "The default extrasolar order is Fornax A then Virgo A because both are far from the Galactic plane/center relative to Cas A, Cyg A, and Tau A. Bright plane sources remain opt-in controls. Outer planets are also opt-in and should initially be reported as exploratory morphology or upper limits; Jupiter should be cross-checked against Io-phase burst diagnostics.",
        "",
        "## Configuration",
        "",
        f"- Window: ±{config.window_seconds:.0f} s",
        f"- Bin size: {config.bin_seconds:.0f} s",
        f"- Normalization sidebands: |t| ≥ {config.sideband_start_seconds:.0f} s",
        f"- Fixed shifts: {', '.join(f'{value:+.0f} s' for value in config.control_shifts_seconds)}",
        f"- Validity mode: `{config.validity_mode}`",
        "",
        "## Event inputs",
        "",
    ]
    for status in source_status:
        lines.append(
            f"- `{status['source_name']}`: {status['n_event_rows']} event rows from "
            f"{', '.join(status['event_tables']) if status['event_tables'] else 'no available table'}"
        )
    lines.extend(["", "## Generated figures", ""])
    lines.extend(f"- `{path.name}`" for path in plot_paths)
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_number_list(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned", default=str(CLEAN))
    parser.add_argument("--profiles", default="", help="Resume plotting from an existing occultogram event-profile table")
    parser.add_argument("--event-table", action="append", default=[], help="Optional event table(s) used for every requested source")
    parser.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/occultograms_morphology_v1"))
    parser.add_argument("--window-s", type=float, default=2400.0)
    parser.add_argument("--bin-s", type=float, default=60.0)
    parser.add_argument("--sideband-start-s", type=float, default=1200.0)
    parser.add_argument("--control-shifts-s", default="-1200,-600,0,600,1200")
    parser.add_argument("--validity-mode", choices=["finite", "pipeline"], default="finite")
    parser.add_argument("--contrast-inner-s", type=float, default=120.0)
    parser.add_argument("--contrast-outer-s", type=float, default=600.0)
    parser.add_argument("--profile-table-format", choices=["parquet", "npy", "csv"], default="parquet")
    parser.add_argument("--channel-plots", action="store_true", help="Also write one enlarged figure per frequency/antenna channel")
    args = parser.parse_args()

    sources = tuple(item.strip().lower() for item in args.sources.split(",") if item.strip())
    config = OccultogramConfig(
        window_seconds=args.window_s,
        bin_seconds=args.bin_s,
        sideband_start_seconds=args.sideband_start_s,
        control_shifts_seconds=_parse_number_list(args.control_shifts_s),
        validity_mode=args.validity_mode,
    )
    output_dir = ensure_dir(args.out_dir)
    if args.profiles:
        profile_path = resolve_table_path(Path(args.profiles))
        if not profile_path.exists():
            raise FileNotFoundError(f"occultogram event-profile table is missing: {args.profiles}")
        points = read_table(profile_path)
        if points.empty:
            raise RuntimeError(f"occultogram event-profile table is empty: {profile_path}")
        summary = _read(output_dir / "occultogram_event_and_lunar_block_summary.csv")
        status_path = output_dir / "event_input_status.json"
        source_status = pd.read_json(status_path).to_dict(orient="records") if status_path.exists() else []
        paths = plot_occultograms(points, output_dir, channel_plots=args.channel_plots, summary=summary)
        write_report(output_dir, source_status, paths, config)
        for path in paths:
            print(path)
        return

    clean = _read(Path(args.cleaned), parse_dates=["time"])
    if clean.empty:
        raise FileNotFoundError(f"cleaned time series is missing or empty: {args.cleaned}")
    event_tables = [Path(value) for value in args.event_table] or None

    source_status = []
    all_events = []
    for source in sources:
        events, used_paths = load_events_for_source(source, event_tables)
        source_status.append({"source_name": source, "n_event_rows": int(len(events)), "event_tables": used_paths})
        if not events.empty:
            all_events.append(events)
    combined_events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    points = extract_occultogram_points(clean, combined_events, config)
    if points.empty:
        raise RuntimeError("no occultogram points were extracted from the requested sources")

    folded = fold_occultogram_points(points)
    summary = summarize_occultogram(folded)
    contrasts = event_signed_contrasts(
        points,
        inner_seconds=args.contrast_inner_s,
        outer_seconds=args.contrast_outer_s,
    )
    profile_path = output_table_path(output_dir, "occultogram_event_profiles", args.profile_table_format)
    write_table(compact_occultogram_profiles(points), profile_path, index=False)
    write_table(
        occultogram_event_metadata(points),
        output_dir / "occultogram_event_metadata.parquet",
        index=False,
    )
    write_table(summary, output_dir / "occultogram_event_and_lunar_block_summary.parquet", index=False)
    write_table(contrasts, output_dir / "occultogram_event_signed_contrasts.parquet", index=False)
    pd.DataFrame(source_status).to_json(output_dir / "event_input_status.json", orient="records", indent=2)
    write_json(
        output_dir / "run_config.json",
        {
            "sources": sources,
            "cleaned": str(args.cleaned),
            "window_seconds": config.window_seconds,
            "bin_seconds": config.bin_seconds,
            "sideband_start_seconds": config.sideband_start_seconds,
            "control_shifts_seconds": config.control_shifts_seconds,
            "validity_mode": config.validity_mode,
            "contrast_inner_seconds": args.contrast_inner_s,
            "contrast_outer_seconds": args.contrast_outer_s,
            "profile_table_format": args.profile_table_format,
            "software_versions": software_versions(),
        },
    )
    paths = plot_occultograms(points, output_dir, channel_plots=args.channel_plots)
    write_report(output_dir, source_status, paths, config)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
