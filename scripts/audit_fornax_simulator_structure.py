#!/usr/bin/env python
"""Audit why the diffuse-beam simulator disagrees with Fornax A profiles.

This diagnostic compares raw normalized Fornax A event profiles against two
diffuse-sky beam variants on the same event windows:

* the older axisymmetric E/H-mean beam;
* the newer yawed azimuth-dependent E/H interpolation.

The goal is not to claim a detection.  It is to test whether the old simulator
missed enough beam azimuth structure to explain the observed trend direction.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.util import ensure_dir, robust_sigma, software_versions, write_json  # noqa: E402

from scripts.build_all_frequency_occultation_profile_grids import (  # noqa: E402
    ANT_COLOR,
    ANT_LABEL,
    CLEAN,
    _events_for_source,
    _read,
)
from scripts.run_diffuse_beam_simulator_subtraction import (  # noqa: E402
    MODEL_NSIDE,
    _antenna_frame_axes,
    _beam_weighted_sky,
    _event_window,
    _load_beam,
    _load_sky_i,
    _make_groups,
    _nearest_beam,
    _pixel_fk4_vectors,
)


EXPECTED_SIGN = {"disappearance": -1.0, "reappearance": 1.0}
METHOD_LABEL = {
    "observed_raw": "observed raw",
    "model_axisymmetric_mean": "old radial model",
    "model_eh_azimuth_yaw13": "yawed E/H model",
    "model_azimuth_minus_axisym": "yawed - radial model",
}
METHOD_COLOR = {
    "observed_raw": "black",
    "model_axisymmetric_mean": "#4c78a8",
    "model_eh_azimuth_yaw13": "#d95f02",
    "model_azimuth_minus_axisym": "#7b3294",
}


def _normalize_event(t: np.ndarray, y: np.ndarray, inner_s: float) -> np.ndarray | None:
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    side = np.isfinite(y) & np.isfinite(t) & (np.abs(t) >= float(inner_s))
    if np.count_nonzero(side) < 6:
        return None
    center = float(np.nanmedian(y[side]))
    scale = robust_sigma(y[side] - center)
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(y[side]))
    if not np.isfinite(scale) or scale <= 0:
        return None
    return (y - center) / scale


def _bin_event(
    rows: list[dict[str, object]],
    event: pd.Series,
    t: np.ndarray,
    z: np.ndarray,
    bins: np.ndarray,
    method: str,
) -> None:
    idx = np.digitize(t, bins) - 1
    for bin_idx in sorted(set(idx)):
        if bin_idx < 0 or bin_idx >= len(bins) - 1:
            continue
        mask = idx == bin_idx
        vals = z[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        rows.append(
            {
                "source_name": str(event["source_name"]).lower(),
                "event_id": int(event["event_id"]),
                "event_type": str(event["event_type"]),
                "frequency_band": int(event["frequency_band"]),
                "frequency_mhz": float(event["frequency_mhz"]),
                "antenna": str(event["antenna"]),
                "method": method,
                "t_bin_sec": float(0.5 * (bins[bin_idx] + bins[bin_idx + 1])),
                "z_power": float(np.nanmedian(vals)),
                "n_samples": int(vals.size),
            }
        )


def collect_same_window_profiles(
    clean: pd.DataFrame,
    events: pd.DataFrame,
    source: str,
    antennas: list[str],
    window_s: float,
    bin_s: float,
    inner_s: float,
    yaw_deg: float,
    frequency_bands: list[int] | None = None,
    max_events_per_group: int | None = None,
) -> pd.DataFrame:
    groups = _make_groups(clean)
    pixel_vecs = _pixel_fk4_vectors(MODEL_NSIDE)
    bins = np.arange(-float(window_s), float(window_s) + float(bin_s), float(bin_s))
    rows: list[dict[str, object]] = []
    work = events[events["source_name"].astype(str).str.lower().eq(source.lower())].copy()
    work = work[work["antenna"].isin(antennas)].copy()
    if frequency_bands:
        work = work[work["frequency_band"].astype(int).isin([int(x) for x in frequency_bands])].copy()
    if max_events_per_group and max_events_per_group > 0:
        work = (
            work.sort_values("predicted_event_time")
            .groupby(["frequency_band", "antenna", "event_type"], group_keys=False)
            .head(int(max_events_per_group))
            .copy()
        )
    work = work.sort_values(["frequency_band", "antenna", "event_type", "predicted_event_time"])

    for (band, freq, antenna), ev_group in work.groupby(["frequency_band", "frequency_mhz", "antenna"], sort=True):
        payload = groups.get((int(band), str(antenna)))
        if payload is None:
            continue
        group, group_ns = payload
        sky_i, _sky_path = _load_sky_i(float(freq))
        beam_freq, eplane, hplane = _nearest_beam(float(freq))
        beam_angles, beam_gain_e, beam_gain_h = _load_beam(eplane, hplane)
        axis_cache: dict[int, float] = {}
        yaw_cache: dict[int, float] = {}

        for _, ev in ev_group.iterrows():
            local = _event_window(group, group_ns, pd.Timestamp(ev["predicted_event_time"]), window_s)
            if local.empty or len(local) < 8:
                continue
            t = local["t_rel_sec"].to_numpy(dtype=float)
            raw = pd.to_numeric(local["power"], errors="coerce").to_numpy(dtype=float)
            z_raw = _normalize_event(t, raw, inner_s)
            if z_raw is None:
                continue

            idx = local["group_index"].to_numpy(dtype=int)
            missing_axis = local.iloc[[i for i, row_idx in enumerate(idx) if int(row_idx) not in axis_cache]]
            if not missing_axis.empty:
                axes, e_axes, h_axes = _antenna_frame_axes(missing_axis, str(antenna), yaw_deg)
                axis_vals = _beam_weighted_sky(
                    axes,
                    sky_i,
                    pixel_vecs,
                    beam_angles,
                    beam_gain_e,
                    beam_gain_h,
                    e_axes=e_axes,
                    h_axes=h_axes,
                    beam_mode="axisymmetric_mean",
                )
                yaw_vals = _beam_weighted_sky(
                    axes,
                    sky_i,
                    pixel_vecs,
                    beam_angles,
                    beam_gain_e,
                    beam_gain_h,
                    e_axes=e_axes,
                    h_axes=h_axes,
                    beam_mode="eh_azimuth",
                )
                for row_idx, axis_val, yaw_val in zip(missing_axis["group_index"], axis_vals, yaw_vals):
                    axis_cache[int(row_idx)] = float(axis_val)
                    yaw_cache[int(row_idx)] = float(yaw_val)

            axis_model = np.asarray([axis_cache.get(int(row_idx), np.nan) for row_idx in idx], dtype=float)
            yaw_model = np.asarray([yaw_cache.get(int(row_idx), np.nan) for row_idx in idx], dtype=float)
            z_axis = _normalize_event(t, axis_model, inner_s)
            z_yaw = _normalize_event(t, yaw_model, inner_s)
            z_diff = _normalize_event(t, yaw_model - axis_model, inner_s)

            _bin_event(rows, ev, t, z_raw, bins, "observed_raw")
            if z_axis is not None:
                _bin_event(rows, ev, t, z_axis, bins, "model_axisymmetric_mean")
            if z_yaw is not None:
                _bin_event(rows, ev, t, z_yaw, bins, "model_eh_azimuth_yaw13")
            if z_diff is not None:
                _bin_event(rows, ev, t, z_diff, bins, "model_azimuth_minus_axisym")

    return pd.DataFrame.from_records(rows)


def _robust_sem(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size <= 1:
        return np.nan
    scale = robust_sigma(vals - np.nanmedian(vals))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(vals, ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        return np.nan
    return float(scale / np.sqrt(vals.size))


def summarize(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["source_name", "event_type", "frequency_band", "frequency_mhz", "antenna", "method", "t_bin_sec"]
    for vals_key, grp in points.groupby(keys, sort=True, dropna=False):
        vals = pd.to_numeric(grp["z_power"], errors="coerce").dropna()
        if vals.empty:
            continue
        rows.append(
            {
                **dict(zip(keys, vals_key)),
                "median_z_power": float(vals.median()),
                "median_z_power_err": _robust_sem(vals),
                "n_events": int(grp["event_id"].nunique()),
                "n_points": int(len(vals)),
            }
        )
    return pd.DataFrame(rows)


def prepost_contrast(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pre = (-180.0, -60.0)
    post = (60.0, 180.0)
    keys = ["source_name", "frequency_band", "frequency_mhz", "antenna", "event_type", "method"]
    for vals_key, grp in summary.groupby(keys, sort=True, dropna=False):
        before = grp[(grp["t_bin_sec"] >= pre[0]) & (grp["t_bin_sec"] <= pre[1])]["median_z_power"]
        after = grp[(grp["t_bin_sec"] >= post[0]) & (grp["t_bin_sec"] <= post[1])]["median_z_power"]
        if before.empty or after.empty:
            continue
        event_type = str(vals_key[4])
        delta = float(np.nanmedian(after) - np.nanmedian(before))
        rows.append(
            {
                **dict(zip(keys, vals_key)),
                "n_events": int(np.nanmedian(grp["n_events"])),
                "post_minus_pre": delta,
                "source_like_contrast": float(EXPECTED_SIGN[event_type] * delta),
            }
        )
    return pd.DataFrame(rows)


def sign_agreement(contrasts: pd.DataFrame) -> pd.DataFrame:
    raw = contrasts[contrasts["method"].eq("observed_raw")].copy()
    raw = raw.rename(
        columns={
            "post_minus_pre": "post_minus_pre_observed",
            "source_like_contrast": "source_like_contrast_observed",
        }
    )
    rows = []
    keys = ["source_name", "frequency_band", "frequency_mhz", "antenna", "event_type"]
    for method in ["model_axisymmetric_mean", "model_eh_azimuth_yaw13", "model_azimuth_minus_axisym"]:
        model = contrasts[contrasts["method"].eq(method)].copy()
        model = model.rename(
            columns={
                "post_minus_pre": "post_minus_pre_model",
                "source_like_contrast": "source_like_contrast_model",
            }
        )
        merged = raw[keys + ["post_minus_pre_observed", "source_like_contrast_observed"]].merge(
            model[keys + ["post_minus_pre_model", "source_like_contrast_model"]],
            on=keys,
            how="inner",
        )
        for label, sub in [
            ("all", merged),
            (
                "lower_v_low_frequency",
                merged[merged["antenna"].eq("rv2_coarse") & merged["frequency_mhz"].isin([0.45, 0.70, 0.90, 1.31, 2.20])],
            ),
            (
                "lower_v_low_reappearance",
                merged[
                    merged["antenna"].eq("rv2_coarse")
                    & merged["frequency_mhz"].isin([0.45, 0.70, 0.90, 1.31, 2.20])
                    & merged["event_type"].eq("reappearance")
                ],
            ),
            (
                "lower_v_low_disappearance",
                merged[
                    merged["antenna"].eq("rv2_coarse")
                    & merged["frequency_mhz"].isin([0.45, 0.70, 0.90, 1.31, 2.20])
                    & merged["event_type"].eq("disappearance")
                ],
            ),
        ]:
            if sub.empty:
                continue
            obs = sub["source_like_contrast_observed"].to_numpy(dtype=float)
            mod = sub["source_like_contrast_model"].to_numpy(dtype=float)
            rows.append(
                {
                    "model_method": method,
                    "subset": label,
                    "n_rows": int(len(sub)),
                    "sign_agreement": float((np.sign(obs) == np.sign(mod)).mean()),
                    "correlation": float(np.corrcoef(obs, mod)[0, 1]) if len(sub) > 1 else np.nan,
                    "median_observed_source_like_contrast": float(np.nanmedian(obs)),
                    "median_model_source_like_contrast": float(np.nanmedian(mod)),
                }
            )
    return pd.DataFrame(rows)


def plot_overlay_grid(
    summary: pd.DataFrame,
    source: str,
    antenna: str,
    out_dir: Path,
    window_s: float,
    methods: list[str],
    suffix: str,
) -> Path:
    sub_summary = summary[summary["antenna"].eq(antenna) & summary["method"].isin(methods)].copy()
    freqs = sorted(sub_summary["frequency_mhz"].dropna().unique())
    fig, axes = plt.subplots(len(freqs), 2, figsize=(12, max(10, 1.35 * len(freqs))), sharex=True)
    if len(freqs) == 1:
        axes = np.asarray([axes])
    for i, freq in enumerate(freqs):
        for j, event_type in enumerate(["disappearance", "reappearance"]):
            ax = axes[i, j]
            for method in methods:
                grp = sub_summary[
                    np.isclose(sub_summary["frequency_mhz"], freq)
                    & sub_summary["event_type"].eq(event_type)
                    & sub_summary["method"].eq(method)
                ].sort_values("t_bin_sec")
                if grp.empty:
                    continue
                ax.errorbar(
                    grp["t_bin_sec"],
                    grp["median_z_power"],
                    yerr=grp["median_z_power_err"],
                    marker="o",
                    markersize=2.1,
                    linewidth=1.05,
                    elinewidth=0.45,
                    capsize=1.0,
                    alpha=0.86,
                    color=METHOD_COLOR.get(method, "0.35"),
                    ecolor=METHOD_COLOR.get(method, "0.35"),
                    label=METHOD_LABEL.get(method, method) if i == 0 and j == 1 else None,
                )
            ax.axvline(0, color="black", linestyle="--", linewidth=0.75)
            ax.axhline(0, color="0.65", linewidth=0.7)
            ax.set_title(f"{freq:.2f} MHz {event_type}", fontsize=9)
            if j == 0:
                ax.set_ylabel("event-normalized power")
            if i == len(freqs) - 1:
                ax.set_xlabel("seconds from predicted event")
            if i == 0 and j == 1:
                ax.legend(frameon=False, fontsize=7)
    fig.suptitle(
        f"{source.replace('_', ' ').title()} {ANT_LABEL.get(antenna, antenna)}: observed vs diffuse model variants",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    path = out_dir / f"{source}_{antenna}_{suffix}_{int(window_s)}s.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_contrast_scatter(contrasts: pd.DataFrame, out_dir: Path, source: str) -> Path:
    raw = contrasts[contrasts["method"].eq("observed_raw")].rename(
        columns={"source_like_contrast": "observed_source_like_contrast"}
    )
    keys = ["source_name", "frequency_band", "frequency_mhz", "antenna", "event_type"]
    methods = ["model_axisymmetric_mean", "model_eh_azimuth_yaw13"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, method in zip(axes, methods):
        model = contrasts[contrasts["method"].eq(method)].rename(
            columns={"source_like_contrast": "model_source_like_contrast"}
        )
        merged = raw[keys + ["observed_source_like_contrast"]].merge(
            model[keys + ["model_source_like_contrast"]], on=keys, how="inner"
        )
        for antenna, grp in merged.groupby("antenna", sort=True):
            ax.scatter(
                grp["observed_source_like_contrast"],
                grp["model_source_like_contrast"],
                s=35,
                alpha=0.8,
                color=ANT_COLOR.get(antenna),
                label=ANT_LABEL.get(antenna, antenna),
            )
            for _, row in grp.iterrows():
                if row["frequency_mhz"] <= 2.2:
                    ax.text(
                        row["observed_source_like_contrast"],
                        row["model_source_like_contrast"],
                        f"{row['frequency_mhz']:.2g} {row['event_type'][0]}",
                        fontsize=7,
                        alpha=0.72,
                    )
        lim = 1.5
        ax.plot([-lim, lim], [-lim, lim], color="0.45", linestyle="--", linewidth=0.8)
        ax.axhline(0, color="0.75", linewidth=0.7)
        ax.axvline(0, color="0.75", linewidth=0.7)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(METHOD_LABEL[method])
        ax.set_xlabel("observed source-like contrast")
        ax.set_ylabel("model source-like contrast")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{source.replace('_', ' ').title()}: observed vs model pre/post direction", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / f"{source}_observed_vs_model_contrast_scatter.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(
    out_dir: Path,
    source: str,
    contrasts: pd.DataFrame,
    agreement: pd.DataFrame,
    paths: list[Path],
    config: dict[str, object],
) -> None:
    lower_low = contrasts[
        contrasts["antenna"].eq("rv2_coarse")
        & contrasts["frequency_mhz"].isin([0.45, 0.70, 0.90, 1.31, 2.20])
        & contrasts["method"].isin(["observed_raw", "model_axisymmetric_mean", "model_eh_azimuth_yaw13"])
    ].copy()
    pivot = lower_low.pivot_table(
        index=["frequency_mhz", "event_type"],
        columns="method",
        values="source_like_contrast",
        aggfunc="median",
    ).reset_index()
    lines = [
        "# Fornax A Diffuse-Beam Simulator Structure Audit",
        "",
        "## Purpose",
        "",
        "This audit tests whether the old Fornax A diffuse-beam simulator had the wrong trend because it averaged away",
        "azimuthal E/H beam structure.  Raw data and model variants are evaluated on the same event windows.",
        "",
        "## Configuration",
        "",
        pd.Series(config).to_string(),
        "",
        "## Main Result",
        "",
        "The yawed E/H model changes the simulated profiles, but it does not repair the key disagreement.",
        "For lower-V low-frequency Fornax A rows, the observed profiles are mostly source-like in both disappearance",
        "and reappearance.  The old radial model is source-like for many disappearance rows but anti-template for most",
        "reappearance rows.  The yawed E/H model remains poorly aligned with the observed source-like direction.",
        "",
        "This means the missed structure is not simply the E/H azimuth interpolation with a 13 degree yaw.  Either the",
        "true 2D beam/spin response is materially different from the two 1D cuts, or the observed Fornax A trend includes",
        "a real point/extended-source occultation component that the diffuse-only simulator intentionally lacks.",
        "",
        "## Sign Agreement Summary",
        "",
        agreement.to_string(index=False) if not agreement.empty else "No agreement rows.",
        "",
        "## Lower-V Low-Frequency Source-Like Contrast",
        "",
        pivot.to_string(index=False) if not pivot.empty else "No lower-V low-frequency rows.",
        "",
        "## Interpretation",
        "",
        "- The old simulator was an incomplete diagnostic, not a calibrated forward model.",
        "- The strongest old-model failure is lower-V reappearance: the observed trend rises after reappearance, while",
        "  the radial diffuse model usually falls.",
        "- The yawed E/H approximation captures some azimuthal structure, but it still uses only two 1D cuts and a guessed",
        "  antenna-frame reference.  It cannot represent asymmetric sidelobes, spin phase, polarization, or lunar blocked-sky terms.",
        "- Because fixed point sources such as Fornax A, Cyg A, and Cas A often show the expected sign while ecliptic/moving",
        "  targets do not, the Fornax mismatch is more consistent with a missing source-occultation term plus incomplete beam",
        "  physics than with the diffuse simulator alone.",
        "",
        "## Generated Plots",
        "",
    ]
    lines.extend(f"- `{path}`" for path in paths)
    (out_dir / "fornax_simulator_structure_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="fornax_a")
    parser.add_argument("--antennas", default="rv2_coarse,rv1_coarse")
    parser.add_argument("--window-s", type=float, default=900.0)
    parser.add_argument("--bin-s", type=float, default=60.0)
    parser.add_argument("--inner-s", type=float, default=15.0)
    parser.add_argument("--yaw-deg", type=float, default=13.0)
    parser.add_argument(
        "--frequency-bands",
        default="",
        help="Optional comma-separated frequency bands. Empty means all available bands.",
    )
    parser.add_argument(
        "--max-events-per-group",
        type=int,
        default=200,
        help="Stratified cap per frequency/antenna/event_type. Use 0 for all events.",
    )
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/fornax_simulator_structure_audit_v1"))
    args = parser.parse_args()

    source = str(args.source).lower()
    antennas = [x.strip() for x in str(args.antennas).split(",") if x.strip()]
    frequency_bands = [int(x.strip()) for x in str(args.frequency_bands).split(",") if x.strip()]
    out_dir = ensure_dir(args.out_dir)
    config = {
        "source": source,
        "antennas": antennas,
        "window_s": float(args.window_s),
        "bin_s": float(args.bin_s),
        "inner_s": float(args.inner_s),
        "yaw_deg": float(args.yaw_deg),
        "frequency_bands": frequency_bands if frequency_bands else "all",
        "max_events_per_frequency_antenna_event_type": int(args.max_events_per_group),
        "model_nside": MODEL_NSIDE,
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)

    clean_cols = [
        "time",
        "frequency_band",
        "frequency_mhz",
        "antenna",
        "power",
        "is_valid",
        "position_x",
        "position_y",
        "position_z",
    ]
    clean = _read(CLEAN, usecols=clean_cols, parse_dates=["time"])
    clean = clean[clean["antenna"].isin(antennas)].copy()
    events = _events_for_source(source)
    if events.empty:
        raise SystemExit(f"no events found for source {source}")

    points = collect_same_window_profiles(
        clean=clean,
        events=events,
        source=source,
        antennas=antennas,
        window_s=float(args.window_s),
        bin_s=float(args.bin_s),
        inner_s=float(args.inner_s),
        yaw_deg=float(args.yaw_deg),
        frequency_bands=frequency_bands or None,
        max_events_per_group=int(args.max_events_per_group) if int(args.max_events_per_group) > 0 else None,
    )
    summary = summarize(points)
    contrasts = prepost_contrast(summary)
    agreement = sign_agreement(contrasts)

    points.to_csv(out_dir / f"{source}_same_window_observed_and_model_points.csv", index=False)
    summary.to_csv(out_dir / f"{source}_same_window_observed_and_model_summary.csv", index=False)
    contrasts.to_csv(out_dir / f"{source}_same_window_observed_and_model_contrasts.csv", index=False)
    agreement.to_csv(out_dir / f"{source}_simulator_sign_agreement.csv", index=False)

    paths = []
    if "rv2_coarse" in antennas:
        paths.append(plot_overlay_grid(
            summary,
            source,
            "rv2_coarse",
            out_dir,
            float(args.window_s),
            ["observed_raw", "model_axisymmetric_mean", "model_eh_azimuth_yaw13"],
            "lower_v_observed_vs_models",
        ))
        paths.append(plot_overlay_grid(
            summary,
            source,
            "rv2_coarse",
            out_dir,
            float(args.window_s),
            ["model_azimuth_minus_axisym"],
            "lower_v_missed_azimuth_component",
        ))
    if "rv1_coarse" in antennas:
        paths.append(plot_overlay_grid(
            summary,
            source,
            "rv1_coarse",
            out_dir,
            float(args.window_s),
            ["observed_raw", "model_axisymmetric_mean", "model_eh_azimuth_yaw13"],
            "upper_v_observed_vs_models",
        ))
    paths.append(plot_contrast_scatter(contrasts, out_dir, source))
    write_report(out_dir, source, contrasts, agreement, paths, config)
    print(out_dir / "fornax_simulator_structure_audit_report.md")


if __name__ == "__main__":
    main()
