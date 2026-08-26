#!/usr/bin/env python
"""Plot Warwick/Dulk/Riddle UAG-42 Jupiter burst catalog phase diagnostics.

The catalog is a manually transcribed active-window table from Warwick,
Dulk, and Riddle (1975), Report UAG-42.  Each row is treated as one reported
burst interval.  Geometry is evaluated at the interval midpoint, using SPICE
for Jupiter System III CML and Io phase.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.table_io import read_table  # noqa: E402
from rylevonberg.util import ensure_dir, write_json  # noqa: E402
from scripts.run_jupiter_historical_window_phase_survey import parse_hhmm  # noqa: E402
from scripts.run_solar_burst_spice_jupiter_visibility import annotate_jupiter_spice  # noqa: E402


DEFAULT_CATALOG = ROOT / "configs/jupiter_warwick_dulk_riddle_1975_active_windows.csv"
DEFAULT_OUT = ROOT / "outputs/jupiter_warwick_dulk_riddle_catalog_phase_cml_v1"
IO_WINDOWS = [("Io-B/D", 80.0, 130.0), ("Io-A/C", 205.0, 260.0)]


def bin_centers(bin_deg: float) -> np.ndarray:
    return np.arange(0.5 * float(bin_deg), 360.0, float(bin_deg))


def in_windows(values_deg: np.ndarray, windows: list[tuple[str, float, float]]) -> np.ndarray:
    vals = np.asarray(values_deg, dtype=float) % 360.0
    out = np.zeros(vals.shape, dtype=bool)
    for _, lo, hi in windows:
        if lo <= hi:
            out |= (vals >= lo) & (vals < hi)
        else:
            out |= (vals >= lo) | (vals < hi)
    return out


def parse_frequency_range(text: object) -> tuple[float, float]:
    value = str(text).strip()
    if "-" not in value:
        return np.nan, np.nan
    left, right = value.split("-", 1)
    try:
        lo = float(left)
        hi = float(right)
    except ValueError:
        return np.nan, np.nan
    if hi < lo:
        return hi, lo
    return lo, hi


def load_catalog(path: Path) -> pd.DataFrame:
    raw = read_table(path, dtype=str)
    rows: list[dict[str, object]] = []
    for idx, row in raw.iterrows():
        date = pd.Timestamp(row["date"])
        start = parse_hhmm(date, row["event_start_hhmm"])
        end = parse_hhmm(date, row["event_end_hhmm"])
        if end <= start:
            end += pd.Timedelta(days=1)
        freq_low, freq_high = parse_frequency_range(row.get("reported_freq_range_mhz", ""))
        midpoint = start + (end - start) / 2
        rows.append(
            {
                "catalog_event_id": f"wdr_{idx + 1:03d}",
                "event_start_time": start,
                "event_end_time": end,
                "event_mid_time": midpoint,
                "duration_min": float((end - start).total_seconds() / 60.0),
                "intensity": int(row["intensity"]),
                "burstiness": str(row.get("burstiness", "")).strip().upper(),
                "reported_freq_range_mhz": str(row.get("reported_freq_range_mhz", "")).strip(),
                "reported_freq_low_mhz": freq_low,
                "reported_freq_high_mhz": freq_high,
                "reported_freq_mid_mhz": float(np.nanmean([freq_low, freq_high])) if np.isfinite(freq_low + freq_high) else np.nan,
                "pdf_page": row.get("pdf_page", ""),
                "notes": row.get("notes", ""),
            }
        )
    return pd.DataFrame(rows)


def annotate_geometry(catalog: pd.DataFrame, out_dir: Path, refresh_kernels: bool) -> pd.DataFrame:
    features = pd.DataFrame(
        {
            "analysis_source": "jupiter",
            "predicted_event_time": catalog["event_mid_time"],
        }
    )
    geom = annotate_jupiter_spice(features, out_dir=out_dir, refresh_kernels=refresh_kernels)
    cols = [
        "predicted_event_time",
        "jupiter_cml_spice_deg",
        "io_phase_spice_deg",
        "io_phase_spice_reverse_deg",
        "spice_geometry_status",
        "spice_io_abc_gate",
        "spice_io_abc_reversed_gate",
        "spice_io_abc_or_reversed_gate",
    ]
    merged = catalog.merge(
        geom[cols].rename(columns={"predicted_event_time": "event_mid_time"}),
        on="event_mid_time",
        how="left",
    )
    merged["io_phase_expected_window"] = in_windows(merged["io_phase_spice_deg"].to_numpy(dtype=float), IO_WINDOWS)
    return merged


def phase_summary(catalog: pd.DataFrame, value_col: str, bin_deg: float) -> pd.DataFrame:
    centers = bin_centers(bin_deg)
    col = f"{value_col}_bin_deg"
    work = catalog.dropna(subset=[value_col]).copy()
    work[col] = np.floor((work[value_col].astype(float) % 360.0) / float(bin_deg)) * float(bin_deg) + 0.5 * float(bin_deg)
    counts = work.groupby(col, sort=True).size().rename("n_events").reset_index()
    full = pd.DataFrame({col: centers})
    out = full.merge(counts, on=col, how="left")
    out["n_events"] = out["n_events"].fillna(0).astype(int)
    out["fraction_events"] = out["n_events"] / len(work) if len(work) else np.nan
    return out


def cml_frequency_summary(catalog: pd.DataFrame, cml_bin_deg: float, freq_bin_mhz: float) -> pd.DataFrame:
    work = catalog.dropna(subset=["jupiter_cml_spice_deg", "reported_freq_mid_mhz"]).copy()
    work["cml_bin_deg"] = np.floor((work["jupiter_cml_spice_deg"] % 360.0) / cml_bin_deg) * cml_bin_deg + 0.5 * cml_bin_deg
    work["freq_mid_bin_mhz"] = np.floor(work["reported_freq_mid_mhz"] / freq_bin_mhz) * freq_bin_mhz + 0.5 * freq_bin_mhz
    return (
        work.groupby(["freq_mid_bin_mhz", "cml_bin_deg"], sort=True)
        .size()
        .rename("n_events")
        .reset_index()
    )


def plot_io_histogram(catalog: pd.DataFrame, summary: pd.DataFrame, out_dir: Path, bin_deg: float) -> Path:
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.bar(
        summary["io_phase_spice_deg_bin_deg"],
        summary["n_events"],
        width=float(bin_deg) * 0.86,
        color="#7570b3",
        alpha=0.82,
        edgecolor="white",
        linewidth=0.35,
    )
    colors = {"Io-B/D": "#d95f02", "Io-A/C": "#1b9e77"}
    for label, lo, hi in IO_WINDOWS:
        ax.axvspan(lo, hi, color=colors[label], alpha=0.10, lw=0)
        ax.axvline(lo, color=colors[label], lw=1.0)
        ax.axvline(hi, color=colors[label], lw=1.0)
        ax.text((lo + hi) / 2, 0.96, label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)
    n_expected = int(catalog["io_phase_expected_window"].sum())
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 30))
    ax.set_xlabel("Io phase at reported burst midpoint (deg)")
    ax.set_ylabel("catalog event count")
    ax.set_title(f"Warwick/Dulk/Riddle UAG-42 bursts by Io phase ({n_expected}/{len(catalog)} in nominal windows)")
    ax.grid(True, axis="y", color="0.90", lw=0.5)
    fig.tight_layout()
    path = out_dir / "warwick_dulk_riddle_burst_count_io_phase.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_cml_histogram(summary: pd.DataFrame, out_dir: Path, bin_deg: float) -> Path:
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.bar(
        summary["jupiter_cml_spice_deg_bin_deg"],
        summary["n_events"],
        width=float(bin_deg) * 0.86,
        color="#386cb0",
        alpha=0.82,
        edgecolor="white",
        linewidth=0.35,
    )
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 30))
    ax.set_xlabel("Jupiter System III CML at reported burst midpoint (deg)")
    ax.set_ylabel("catalog event count")
    ax.set_title("Warwick/Dulk/Riddle UAG-42 bursts by Jupiter System III longitude")
    ax.grid(True, axis="y", color="0.90", lw=0.5)
    fig.tight_layout()
    path = out_dir / "warwick_dulk_riddle_burst_count_system_iii_cml.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_iocml_scatter(catalog: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    sizes = 30 + 18 * catalog["intensity"].to_numpy(dtype=float)
    sc = ax.scatter(
        catalog["jupiter_cml_spice_deg"],
        catalog["io_phase_spice_deg"],
        c=catalog["reported_freq_mid_mhz"],
        s=sizes,
        cmap="plasma",
        alpha=0.82,
        edgecolor="white",
        linewidth=0.45,
    )
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 360)
    ax.set_xticks(np.arange(0, 361, 60))
    ax.set_yticks(np.arange(0, 361, 60))
    ax.set_xlabel("Jupiter System III CML (deg)")
    ax.set_ylabel("Io phase (deg)")
    ax.set_title("Warwick/Dulk/Riddle catalog burst midpoints in CML-Io phase space")
    ax.grid(True, color="0.90", lw=0.5)
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label("reported frequency midpoint (MHz)")
    fig.tight_layout()
    path = out_dir / "warwick_dulk_riddle_burst_cml_io_phase_scatter.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_cml_frequency(summary: pd.DataFrame, out_dir: Path, cml_bin_deg: float) -> Path:
    if summary.empty:
        raise ValueError("empty CML-frequency summary")
    centers = bin_centers(cml_bin_deg)
    mat = summary.pivot_table(index="freq_mid_bin_mhz", columns="cml_bin_deg", values="n_events", aggfunc="sum").reindex(columns=centers)
    mat = mat.fillna(0)
    vals = mat.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(12.8, 5.8))
    im = ax.imshow(
        vals,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[0.0, 360.0, -0.5, len(mat.index) - 0.5],
        cmap="viridis",
        norm=Normalize(vmin=0.0, vmax=max(1.0, float(np.nanmax(vals)))),
    )
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels([f"{v:.1f}" for v in mat.index])
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 30))
    ax.set_xlabel("Jupiter System III CML at reported burst midpoint (deg)")
    ax.set_ylabel("reported frequency midpoint bin (MHz)")
    ax.set_title("Warwick/Dulk/Riddle burst count by reported frequency and System III longitude")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="catalog event count")
    fig.tight_layout()
    path = out_dir / "warwick_dulk_riddle_burst_count_frequency_system_iii_cml.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(out_dir: Path, catalog: pd.DataFrame, paths: list[Path], config: dict[str, object]) -> Path:
    expected = int(catalog["io_phase_expected_window"].sum())
    abc = int(catalog["spice_io_abc_gate"].sum()) if "spice_io_abc_gate" in catalog else 0
    lines = [
        "# Warwick/Dulk/Riddle UAG-42 Jupiter Burst Phase/CML Analysis",
        "",
        "Each catalog row is one reported active interval from the transcribed UAG-42 table.  Io phase and Jupiter System III CML are evaluated at the interval midpoint.",
        "",
        "The default input is the current local manual transcription used by the RAE-2 Jupiter analysis, not a complete OCR of every 1971-1975 UAG-42 report entry.",
        "",
        "This catalog is at higher frequencies than the RAE-2 bands used in the local burst search, so it should be treated as an external activity and beaming reference rather than a one-to-one frequency match.",
        "",
        "## Configuration",
        "",
        *[f"- `{k}`: `{v}`" for k, v in config.items()],
        "",
        "## Counts",
        "",
        f"- catalog intervals: `{len(catalog)}`",
        f"- midpoint geometry status counts: `{catalog['spice_geometry_status'].value_counts(dropna=False).to_dict()}`",
        f"- in nominal Io-phase-only windows `{IO_WINDOWS}`: `{expected}`",
        f"- in broad SPICE Io-A/B/C CML+Io gates: `{abc}`",
        f"- frequency range in transcribed table: `{catalog['reported_freq_low_mhz'].min():.1f}-{catalog['reported_freq_high_mhz'].max():.1f} MHz`",
        "",
        "## Files",
        "",
        *[f"- `{p}`" for p in paths],
    ]
    path = out_dir / "warwick_dulk_riddle_catalog_phase_cml_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--phase-bin-deg", type=float, default=10.0)
    parser.add_argument("--cml-bin-deg", type=float, default=10.0)
    parser.add_argument("--freq-bin-mhz", type=float, default=2.0)
    parser.add_argument("--refresh-kernels", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    config = {
        "catalog": str(args.catalog),
        "phase_bin_deg": float(args.phase_bin_deg),
        "cml_bin_deg": float(args.cml_bin_deg),
        "freq_bin_mhz": float(args.freq_bin_mhz),
    }
    write_json(out_dir / "warwick_dulk_riddle_catalog_phase_cml_config.json", config)

    catalog = annotate_geometry(load_catalog(args.catalog), out_dir, refresh_kernels=bool(args.refresh_kernels))
    catalog = catalog.sort_values("event_mid_time").reset_index(drop=True)
    catalog_path = out_dir / "warwick_dulk_riddle_catalog_with_spice_geometry.csv"
    catalog.to_csv(catalog_path, index=False)

    io_summary = phase_summary(catalog, "io_phase_spice_deg", bin_deg=float(args.phase_bin_deg))
    cml_summary = phase_summary(catalog, "jupiter_cml_spice_deg", bin_deg=float(args.cml_bin_deg))
    cml_freq = cml_frequency_summary(catalog, cml_bin_deg=float(args.cml_bin_deg), freq_bin_mhz=float(args.freq_bin_mhz))
    io_summary_path = out_dir / "warwick_dulk_riddle_io_phase_summary.csv"
    cml_summary_path = out_dir / "warwick_dulk_riddle_system_iii_cml_summary.csv"
    cml_freq_path = out_dir / "warwick_dulk_riddle_frequency_cml_summary.csv"
    io_summary.to_csv(io_summary_path, index=False)
    cml_summary.to_csv(cml_summary_path, index=False)
    cml_freq.to_csv(cml_freq_path, index=False)

    paths = [
        catalog_path,
        io_summary_path,
        cml_summary_path,
        cml_freq_path,
        plot_io_histogram(catalog, io_summary, out_dir, bin_deg=float(args.phase_bin_deg)),
        plot_cml_histogram(cml_summary, out_dir, bin_deg=float(args.cml_bin_deg)),
        plot_iocml_scatter(catalog, out_dir),
        plot_cml_frequency(cml_freq, out_dir, cml_bin_deg=float(args.cml_bin_deg)),
    ]
    report = write_report(out_dir, catalog, paths, config)
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
