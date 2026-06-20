#!/usr/bin/env python
"""Run stack-first finite-duration occultation fits across all frequencies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.detection import baseline_matrix  # noqa: E402
from rylevonberg.stackfit import StackedStepFitConfig, fit_stacked_step, stacked_event_template  # noqa: E402
from rylevonberg.util import datetime_ns, ensure_dir, robust_sigma, software_versions, write_json  # noqa: E402


CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
PLANET_EVENTS = ROOT / "outputs/planetary_confirmation_survey_science_baseline_v2/events/all_planet_predicted_events.csv"
SUN_EVENTS = ROOT / "outputs/pipeline_confidence_audit_v2/sun_audit_input_events.csv"
BRIGHT_EVENTS = ROOT / "outputs/control_survey_bright_sources_postnov1974_v1/02_events/predicted_events.csv"
ANT_LABEL = {"rv1_coarse": "upper V", "rv2_coarse": "lower V"}
ANT_COLOR = {"rv1_coarse": "#4c78a8", "rv2_coarse": "#d95f02"}
VARIANT_LABELS = {
    "raw_zscore_no_baseline": "raw z-score, no trend",
    "sideband_constant_subtracted": "constant sideband",
    "sideband_linear_subtracted": "linear sideband",
}


def _read(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return read_table(path, low_memory=False, **kwargs)


def _safe_scale(values: np.ndarray) -> float:
    scale = robust_sigma(values)
    if np.isfinite(scale) and scale > 0:
        return float(scale)
    scale = float(np.nanstd(values))
    return scale if np.isfinite(scale) and scale > 0 else 1.0


def _profile_variant(t: np.ndarray, y: np.ndarray, variant: str, exclusion_s: float) -> tuple[np.ndarray, int]:
    side = np.abs(t) >= float(exclusion_s)
    if np.count_nonzero(side) < 6:
        side = np.ones(len(y), dtype=bool)
    center = float(np.nanmedian(y[side]))
    scale = _safe_scale(y[side] - center)
    if variant == "raw_zscore_no_baseline":
        return (y - center) / scale, int(np.count_nonzero(side))
    if variant == "sideband_constant_subtracted":
        return (y - center) / scale, int(np.count_nonzero(side))
    if variant == "sideband_linear_subtracted":
        B = baseline_matrix(t[side], 1)
        beta, *_ = np.linalg.lstsq(B, y[side], rcond=None)
        baseline = baseline_matrix(t, 1) @ beta
        resid = y - baseline
        return resid / _safe_scale(resid[side]), int(np.count_nonzero(side))
    raise ValueError(f"unknown variant: {variant}")


def _make_groups(clean: pd.DataFrame) -> dict[tuple[int, str], tuple[pd.DataFrame, np.ndarray]]:
    groups = {}
    for (band, antenna), grp in clean.groupby(["frequency_band", "antenna"], sort=True):
        g = grp.sort_values("time").reset_index(drop=True)
        groups[(int(band), str(antenna))] = (g, datetime_ns(g["time"]))
    return groups


def collect_profiles(
    clean: pd.DataFrame,
    events: pd.DataFrame,
    sources: list[str],
    window_s: float,
    bin_s: float,
    sideband_exclusion_s: float,
    variants: list[str],
    event_time_shift_s: float = 0.0,
) -> pd.DataFrame:
    groups = _make_groups(clean)
    rows = []
    events = events[events["source_name"].astype(str).str.lower().isin(sources)].copy()
    for _, ev in events.iterrows():
        band = int(ev["frequency_band"])
        antenna = str(ev["antenna"])
        payload = groups.get((band, antenna))
        if payload is None:
            continue
        group, group_ns = payload
        original_event_time = pd.Timestamp(ev["predicted_event_time"])
        shifted_event_time = original_event_time + pd.to_timedelta(float(event_time_shift_s), unit="s")
        event_ns = shifted_event_time.value
        half_ns = int(float(window_s) * 1e9)
        lo = int(np.searchsorted(group_ns, event_ns - half_ns, side="left"))
        hi = int(np.searchsorted(group_ns, event_ns + half_ns, side="right"))
        if hi <= lo:
            continue
        local = group.iloc[lo:hi]
        t = (datetime_ns(local["time"]) - event_ns).astype(float) / 1e9
        y = pd.to_numeric(local["power"], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(y) & (np.abs(t) <= float(window_s))
        if "is_valid" in local.columns:
            keep &= local["is_valid"].to_numpy(dtype=bool)
        if np.count_nonzero(keep) < 8:
            continue
        t = t[keep]
        y = y[keep]
        order = np.argsort(t)
        t = t[order]
        y = y[order]
        tbin = np.round(t / float(bin_s)) * float(bin_s)
        for variant in variants:
            profile, n_side = _profile_variant(t, y, variant, sideband_exclusion_s)
            for tr, tb, val in zip(t, tbin, profile):
                rows.append(
                    {
                        "source_name": str(ev["source_name"]).lower(),
                        "event_id": ev.get("event_id"),
                        "event_type": str(ev["event_type"]),
                        "predicted_event_time": ev["predicted_event_time"],
                        "analysis_event_time": shifted_event_time,
                        "event_time_shift_s": float(event_time_shift_s),
                        "frequency_band": band,
                        "frequency_mhz": float(ev["frequency_mhz"]),
                        "antenna": antenna,
                        "window_s": float(window_s),
                        "variant": variant,
                        "t_rel_sec": float(tr),
                        "t_bin_sec": float(tb),
                        "profile_value": float(val),
                        "n_sideband_samples": int(n_side),
                    }
                )
    return pd.DataFrame.from_records(rows)


def stack_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    by = ["source_name", "frequency_band", "frequency_mhz", "antenna", "window_s", "variant", "event_type", "t_bin_sec"]
    for keys, grp in profiles.groupby(by, sort=True, dropna=False):
        vals = pd.to_numeric(grp["profile_value"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        rows.append(
            {
                **dict(zip(by, keys)),
                "mean_profile": float(np.nanmean(vals)),
                "median_profile": float(np.nanmedian(vals)),
                "robust_sem_profile": float(robust_sigma(vals) / np.sqrt(vals.size)) if vals.size > 1 else np.nan,
                "sem_profile": float(np.nanstd(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else np.nan,
                "n_samples": int(vals.size),
                "n_events": int(grp["event_id"].nunique()),
            }
        )
    return pd.DataFrame.from_records(rows)


def fit_grid(
    stacked: pd.DataFrame,
    profiles: pd.DataFrame,
    timing_offsets: list[float],
    transition_durations: list[float],
) -> pd.DataFrame:
    rows = []
    group_cols = ["source_name", "frequency_band", "frequency_mhz", "antenna", "window_s", "variant"]
    event_counts = profiles.groupby(group_cols, sort=True, dropna=False)["event_id"].nunique().rename("total_events_used").reset_index()
    for keys, grp in stacked.groupby(group_cols, sort=True, dropna=False):
        meta = dict(zip(group_cols, keys))
        match = event_counts
        for key, val in meta.items():
            match = match[match[key].eq(val)]
        n_events = int(match["total_events_used"].iloc[0]) if not match.empty else 0
        for stack_group, sub in [("combined", grp), *[(et, g) for et, g in grp.groupby("event_type", sort=True)]]:
            sub = sub.sort_values(["event_type", "t_bin_sec"])
            for baseline_order, model_name in [(0, "constant_plus_finite"), (1, "linear_plus_finite")]:
                cfg = StackedStepFitConfig(
                    baseline_order=baseline_order,
                    timing_offsets_seconds=tuple(float(x) for x in timing_offsets),
                    transition_durations_seconds=tuple(float(x) for x in transition_durations),
                )
                fit = fit_stacked_step(
                    sub["t_bin_sec"].to_numpy(dtype=float),
                    sub["mean_profile"].to_numpy(dtype=float),
                    sub["event_type"].astype(str).to_numpy(),
                    uncertainty=sub["robust_sem_profile"].fillna(sub["sem_profile"]).to_numpy(dtype=float),
                    config=cfg,
                )
                rows.append(
                    {
                        **meta,
                        "stack_group": str(stack_group),
                        "stack_model": model_name,
                        "total_events_used": n_events,
                        "median_events_per_bin": float(np.nanmedian(sub["n_events"])) if not sub.empty else np.nan,
                        **fit,
                    }
                )
    return pd.DataFrame.from_records(rows)


def plot_fit_heatmaps(fits: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    combined = fits[fits["stack_group"].eq("combined") & fits["stack_model"].eq("constant_plus_finite")].copy()
    if combined.empty:
        return paths
    for source, src in combined.groupby("source_name", sort=True):
        variants = [v for v in VARIANT_LABELS if v in set(src["variant"])]
        fig, axes = plt.subplots(len(variants), 3, figsize=(14, max(3.2, 3.2 * len(variants))), squeeze=False, sharex=False)
        for i, variant in enumerate(variants):
            sub = src[src["variant"].eq(variant)].copy()
            for j, (col, title, cmap) in enumerate(
                [
                    ("stack_fit_snr", "finite-template fit SNR", "coolwarm"),
                    ("best_transition_duration_s", "best transition duration (s)", "viridis"),
                    ("delta_bic", "Delta BIC vs baseline", "magma"),
                ]
            ):
                ax = axes[i, j]
                pivot = sub.pivot_table(index="antenna", columns="frequency_mhz", values=col, aggfunc="first")
                order = [a for a in ["rv2_coarse", "rv1_coarse"] if a in pivot.index] + [a for a in pivot.index if a not in ["rv2_coarse", "rv1_coarse"]]
                pivot = pivot.loc[order]
                data = pivot.to_numpy(dtype=float)
                if col == "stack_fit_snr":
                    vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
                    vmin = -vmax
                else:
                    vmin = np.nanmin(data) if np.isfinite(data).any() else 0.0
                    vmax = np.nanmax(data) if np.isfinite(data).any() else 1.0
                im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title(f"{VARIANT_LABELS.get(variant, variant)}: {title}")
                ax.set_yticks(np.arange(len(pivot.index)), [ANT_LABEL.get(str(x), str(x)) for x in pivot.index])
                ax.set_xticks(np.arange(len(pivot.columns)), [f"{float(x):.2f}" for x in pivot.columns], rotation=45, ha="right")
                for y in range(data.shape[0]):
                    for x in range(data.shape[1]):
                        val = data[y, x]
                        if np.isfinite(val):
                            text = f"{val:.1f}" if col != "best_transition_duration_s" else f"{val:.0f}"
                            ax.text(x, y, text, ha="center", va="center", fontsize=7, color="white" if j == 2 else "black")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        fig.suptitle(f"{source}: all-frequency stack-first finite-duration fits", fontsize=13)
        fig.tight_layout()
        path = out_dir / f"{source}_finite_duration_fit_heatmaps.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def _model_curve(sub: pd.DataFrame, fit: pd.Series) -> np.ndarray:
    t = sub["t_bin_sec"].to_numpy(dtype=float)
    y = sub["mean_profile"].to_numpy(dtype=float)
    event_types = sub["event_type"].astype(str).to_numpy()
    tmpl = stacked_event_template(
        t,
        event_types,
        timing_offset_sec=float(fit["best_timing_offset_s"]),
        transition_duration_sec=float(fit.get("best_transition_duration_s", 0.0)),
    )
    B = baseline_matrix(t, int(fit["baseline_order"]))
    X = np.column_stack([B, tmpl])
    sem = sub["robust_sem_profile"].fillna(sub["sem_profile"]).to_numpy(dtype=float)
    sem = np.where(
        np.isfinite(sem) & (sem > 0),
        sem,
        np.nanmedian(sem[np.isfinite(sem) & (sem > 0)]) if np.any(np.isfinite(sem) & (sem > 0)) else 1.0,
    )
    weights = 1.0 / sem**2
    beta, *_ = np.linalg.lstsq(X * np.sqrt(weights)[:, None], y * np.sqrt(weights), rcond=None)
    return X @ beta


def plot_top_profile_fits(stacked: pd.DataFrame, fits: pd.DataFrame, out_dir: Path, n_per_source: int = 4) -> list[Path]:
    paths = []
    combined = fits[
        fits["stack_group"].eq("combined")
        & fits["stack_model"].eq("constant_plus_finite")
        & fits["variant"].eq("raw_zscore_no_baseline")
    ].copy()
    if combined.empty:
        return paths
    combined["abs_snr"] = pd.to_numeric(combined["stack_fit_snr"], errors="coerce").abs()
    for source, src in combined.groupby("source_name", sort=True):
        top = src.sort_values("abs_snr", ascending=False).head(int(n_per_source))
        if top.empty:
            continue
        fig, axes = plt.subplots(len(top), 2, figsize=(13, max(3.0, 2.7 * len(top))), squeeze=False, sharex=True)
        for i, (_, fit) in enumerate(top.iterrows()):
            channel = stacked[
                stacked["source_name"].eq(fit["source_name"])
                & np.isclose(stacked["frequency_mhz"], float(fit["frequency_mhz"]))
                & stacked["antenna"].eq(fit["antenna"])
                & stacked["variant"].eq(fit["variant"])
            ].copy()
            channel = channel.sort_values(["event_type", "t_bin_sec"])
            if channel.empty:
                continue
            model = _model_curve(channel, fit)
            channel = channel.assign(model=model)
            for j, event_type in enumerate(["disappearance", "reappearance"]):
                ax = axes[i, j]
                sub = channel[channel["event_type"].eq(event_type)].sort_values("t_bin_sec")
                if sub.empty:
                    continue
                sem = sub["robust_sem_profile"].fillna(sub["sem_profile"]).fillna(0.0).to_numpy(dtype=float)
                x = sub["t_bin_sec"].to_numpy(dtype=float) / 60.0
                y = sub["mean_profile"].to_numpy(dtype=float)
                ax.plot(x, y, marker="o", ms=3, lw=1.2, color="#333333", label="stack")
                ax.fill_between(x, y - sem, y + sem, color="0.3", alpha=0.15, linewidth=0)
                ax.plot(x, sub["model"], color="#d95f02", lw=2.0, label="finite-duration fit")
                ax.axvline(0.0, color="black", ls=":", lw=0.8)
                ax.axhline(0.0, color="0.7", lw=0.8)
                ax.set_title(f"{event_type}")
                if j == 0:
                    ax.set_ylabel(f"{float(fit['frequency_mhz']):.2f} MHz\n{ANT_LABEL.get(str(fit['antenna']), fit['antenna'])}")
                if i == len(top) - 1:
                    ax.set_xlabel("minutes from predicted event")
                if i == 0 and j == 1:
                    ax.legend(frameon=False, fontsize=8)
            axes[i, 1].text(
                1.02,
                0.5,
                f"SNR={float(fit['stack_fit_snr']):.2f}\nA={float(fit['amplitude']):.3g}\n"
                f"dt={float(fit['best_timing_offset_s']):.0f}s\n"
                f"tau={float(fit['best_transition_duration_s']):.0f}s\n"
                f"DeltaBIC={float(fit['delta_bic']):.1f}",
                transform=axes[i, 1].transAxes,
                va="center",
                ha="left",
                fontsize=8,
            )
        fig.suptitle(f"{source}: strongest raw no-trend finite-duration stack fits", fontsize=13)
        fig.tight_layout(rect=[0, 0, 0.92, 0.97])
        path = out_dir / f"{source}_top_finite_duration_profile_fits.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def _best_rows(fits: pd.DataFrame) -> pd.DataFrame:
    combined = fits[fits["stack_group"].eq("combined")].copy()
    if combined.empty:
        return combined
    combined["abs_snr"] = pd.to_numeric(combined["stack_fit_snr"], errors="coerce").abs()
    return combined.sort_values(["source_name", "abs_snr"], ascending=[True, False]).groupby("source_name").head(8)


def write_report(out_dir: Path, fits: pd.DataFrame, paths: list[Path]) -> None:
    best = _best_rows(fits)
    cols = [
        "source_name",
        "frequency_mhz",
        "antenna",
        "variant",
        "stack_model",
        "total_events_used",
        "amplitude",
        "uncertainty",
        "stack_fit_snr",
        "delta_bic",
        "best_timing_offset_s",
        "best_transition_duration_s",
    ]
    lines = [
        "# Finite-Duration Stack-First Occultation Fits",
        "",
        "This run fits occultation templates after stacking, with a grid over timing offset and finite transition duration.",
        "",
        "Model:",
        "",
        "    y_stack(t) = baseline(t) + A h(t - dt; tau)",
        "",
        "`tau = 0` is the ideal point-source step. Larger `tau` values linearly smear the transition and approximate finite source size, timing scatter, or sparse-sampling broadening.",
        "",
        "Important interpretation:",
        "",
        "- a positive amplitude follows the positive-source occultation convention;",
        "- a negative amplitude is anti-template behavior;",
        "- neighboring frequencies with the same sign are supportive, not a control failure;",
        "- if the best `tau` is very large and Delta BIC is weak, the result is more line/ramp-like than occultation-like;",
        "- compare lower V and upper V. Repeated upper-V dominance remains physically concerning.",
        "",
        "## Best Combined Fits",
        "",
        "```\n" + (best[cols].to_string(index=False) if not best.empty else "No rows.") + "\n```",
        "",
        "## Plots",
        "",
    ]
    lines.extend(f"- `{path.name}`" for path in paths)
    (out_dir / "finite_duration_stackfit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/finite_duration_stackfit_allfreq_v1"))
    parser.add_argument("--sources", default="earth,sun,jupiter")
    parser.add_argument("--window-s", type=float, default=900.0)
    parser.add_argument("--bin-s", type=float, default=60.0)
    parser.add_argument("--sideband-exclusion-s", type=float, default=180.0)
    parser.add_argument("--timing-offsets", default="-300,-240,-180,-120,-60,0,60,120,180,240,300")
    parser.add_argument("--transition-durations", default="0,120,240,360,600,900")
    parser.add_argument("--skip-profile-csv", action="store_true", help="Do not write the large per-sample profile table.")
    parser.add_argument("--event-time-shift-s", type=float, default=0.0, help="Shift analysis event times by this many seconds for time-control runs.")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    sources = [x.strip().lower() for x in str(args.sources).split(",") if x.strip()]
    timing_offsets = [float(x) for x in str(args.timing_offsets).split(",") if x.strip()]
    transition_durations = [float(x) for x in str(args.transition_durations).split(",") if x.strip()]
    variants = list(VARIANT_LABELS)
    write_json(
        out_dir / "run_config.json",
        {
            "sources": sources,
            "window_s": float(args.window_s),
            "bin_s": float(args.bin_s),
            "sideband_exclusion_s": float(args.sideband_exclusion_s),
            "timing_offsets": timing_offsets,
            "transition_durations": transition_durations,
            "variants": variants,
            "event_time_shift_s": float(args.event_time_shift_s),
            "software_versions": software_versions(),
        },
    )

    clean = _read(CLEAN, parse_dates=["time"])
    planet = _read(PLANET_EVENTS, parse_dates=["predicted_event_time"])
    sun = _read(SUN_EVENTS, parse_dates=["predicted_event_time"])
    bright = _read(BRIGHT_EVENTS, parse_dates=["predicted_event_time"])
    events = pd.concat(
        [planet[planet["source_name"].astype(str).str.lower().ne("sun")], sun, bright],
        ignore_index=True,
    )

    profiles = collect_profiles(
        clean,
        events,
        sources,
        args.window_s,
        args.bin_s,
        args.sideband_exclusion_s,
        variants,
        event_time_shift_s=float(args.event_time_shift_s),
    )
    if not args.skip_profile_csv:
        profiles.to_csv(out_dir / "finite_duration_stack_profiles.csv", index=False)
    stacked = stack_profiles(profiles)
    stacked.to_csv(out_dir / "finite_duration_binned_stacks.csv", index=False)
    fits = fit_grid(stacked, profiles, timing_offsets, transition_durations)
    fits.to_csv(out_dir / "finite_duration_stackfit_summary.csv", index=False)
    paths = plot_fit_heatmaps(fits, out_dir)
    paths.extend(plot_top_profile_fits(stacked, fits, out_dir))
    write_report(out_dir, fits, paths)
    print(out_dir / "finite_duration_stackfit_report.md")
    print(out_dir / "finite_duration_stackfit_summary.csv")


if __name__ == "__main__":
    main()
