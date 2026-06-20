#!/usr/bin/env python
"""Compare physical/event covariates for source-like vs anti-template regimes."""

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

from rylevonberg.constants import EARTH_UNIT_COLUMNS, SPACECRAFT_COLUMNS  # noqa: E402
from rylevonberg.events import source_vectors_for_rows  # noqa: E402
from rylevonberg.geometry import normalize_vectors  # noqa: E402
from rylevonberg.util import datetime_ns, ensure_dir  # noqa: E402


EVENT_METRICS = ROOT / "outputs/moving_body_stack_type_subset_tests_v1/moving_body_event_metrics.csv"
EVENTS = ROOT / "outputs/planetary_confirmation_survey_science_baseline_v2/events/all_planet_predicted_events.csv"
CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
OUT = ROOT / "outputs/moving_body_regime_physical_differences_v1"

FOCUS_FREQS = [0.90, 1.31, 2.20, 4.70, 9.18]
LOW_FREQS = [0.90, 1.31, 2.20]
REGIME_THRESHOLD = 0.02


def _classify(vals: pd.Series) -> pd.Series:
    x = pd.to_numeric(vals, errors="coerce")
    out = pd.Series("neutral", index=vals.index, dtype=object)
    out.loc[x > REGIME_THRESHOLD] = "source_like"
    out.loc[x < -REGIME_THRESHOLD] = "anti_template"
    out.loc[~np.isfinite(x)] = "invalid"
    return out


def _ra_dec_from_vec(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unit = normalize_vectors(vec)
    ra = (np.degrees(np.arctan2(unit[:, 1], unit[:, 0])) + 360.0) % 360.0
    dec = np.degrees(np.arcsin(np.clip(unit[:, 2], -1.0, 1.0)))
    return ra, dec


def _limb_position_angle_deg(moon_center: np.ndarray, target: np.ndarray) -> np.ndarray:
    m = normalize_vectors(moon_center)
    s = normalize_vectors(target)
    z = np.array([0.0, 0.0, 1.0])
    pa = np.full(len(m), np.nan)
    for i, (mi, si) in enumerate(zip(m, s)):
        if not np.isfinite(mi).all() or not np.isfinite(si).all():
            continue
        east = np.cross(z, mi)
        if np.linalg.norm(east) < 1e-10:
            east = np.cross(np.array([1.0, 0.0, 0.0]), mi)
        east /= np.linalg.norm(east)
        north = np.cross(mi, east)
        north /= np.linalg.norm(north)
        radial = si - np.dot(si, mi) * mi
        if np.linalg.norm(radial) < 1e-12:
            continue
        radial /= np.linalg.norm(radial)
        # Position angle east of north in the tangent plane around Moon center.
        pa[i] = (np.degrees(np.arctan2(np.dot(radial, east), np.dot(radial, north))) + 360.0) % 360.0
    return pa


def _nearest_base_rows(clean: pd.DataFrame, times: pd.Series) -> pd.DataFrame:
    base = clean.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    base_ns = datetime_ns(base["time"])
    event_ns = pd.to_datetime(times).astype("int64").to_numpy()
    idx = np.searchsorted(base_ns, event_ns, side="left")
    idx = np.clip(idx, 1, len(base_ns) - 1)
    prev_idx = idx - 1
    choose_prev = np.abs(base_ns[prev_idx] - event_ns) <= np.abs(base_ns[idx] - event_ns)
    nearest = np.where(choose_prev, prev_idx, idx)
    return base.iloc[nearest].reset_index(drop=True)


def build_regime_geometry_table() -> pd.DataFrame:
    metrics = read_table(EVENT_METRICS, low_memory=False)
    metrics["regime"] = _classify(metrics["source_like_fractional_contrast"])
    metrics = metrics[metrics["regime"].isin(["source_like", "anti_template", "neutral"])].copy()
    events = read_table(EVENTS, parse_dates=["predicted_event_time"], low_memory=False)
    events = events[events["antenna"].eq("rv2_coarse")].copy()
    geom_cols = [
        "source_name",
        "event_id",
        "event_type",
        "frequency_band",
        "predicted_event_time",
        "pre_limb_angle_deg",
        "post_limb_angle_deg",
        "moon_center_x",
        "moon_center_y",
        "moon_center_z",
        "moon_angular_radius_deg",
    ]
    merged = metrics.merge(events[geom_cols], on=["source_name", "event_id", "event_type", "frequency_band"], how="left")
    merged["predicted_event_time"] = pd.to_datetime(merged["predicted_event_time"])
    merged["date"] = merged["predicted_event_time"].dt.date.astype(str)
    merged["hour_utc"] = merged["predicted_event_time"].dt.hour + merged["predicted_event_time"].dt.minute / 60.0
    merged["day_index"] = (merged["predicted_event_time"] - merged["predicted_event_time"].min()).dt.total_seconds() / 86400.0

    moon = merged[["moon_center_x", "moon_center_y", "moon_center_z"]].to_numpy(dtype=float)
    moon_ra, moon_dec = _ra_dec_from_vec(moon)
    merged["moon_center_ra_deg"] = moon_ra
    merged["moon_center_dec_deg"] = moon_dec

    clean = read_table(
        CLEAN,
        usecols=["time", *SPACECRAFT_COLUMNS, *EARTH_UNIT_COLUMNS],
        parse_dates=["time"],
        low_memory=False,
    )
    target = np.full((len(merged), 3), np.nan)
    for source in ["earth", "sun"]:
        mask = merged["source_name"].eq(source).to_numpy()
        if not mask.any():
            continue
        local = merged.loc[mask].copy()
        base_rows = _nearest_base_rows(clean, local["predicted_event_time"])
        if source == "earth":
            source_row = pd.Series({"source_name": "earth", "kind": "earth", "frame": "fk4"})
        else:
            source_row = pd.Series({"source_name": "sun", "kind": "body", "body_name": "sun", "frame": "fk4"})
        vec = source_vectors_for_rows(
            source_row,
            pd.DatetimeIndex(base_rows["time"]),
            base_rows,
            target_frame="fk4",
            equinox="B1950",
            ephemeris="builtin",
        )
        target[np.where(mask)[0], :] = vec
    target_ra, target_dec = _ra_dec_from_vec(target)
    merged["target_ra_deg"] = target_ra
    merged["target_dec_deg"] = target_dec
    merged["limb_position_angle_deg"] = _limb_position_angle_deg(moon, target)
    # Circular helpers make 0/360 deg continuous for simple median diagnostics.
    pa_rad = np.deg2rad(merged["limb_position_angle_deg"].to_numpy(dtype=float))
    merged["limb_pa_sin"] = np.sin(pa_rad)
    merged["limb_pa_cos"] = np.cos(pa_rad)
    return merged


def numeric_differences(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "source_like_fractional_contrast",
        "near_raw_slope_per_s",
        "far_raw_slope_per_s",
        "pre_far_median",
        "pre_sigma",
        "side_sigma",
        "limb_rate_deg_s",
        "gap_seconds",
        "moon_angular_radius_deg",
        "moon_center_ra_deg",
        "moon_center_dec_deg",
        "target_ra_deg",
        "target_dec_deg",
        "limb_position_angle_deg",
        "limb_pa_sin",
        "limb_pa_cos",
        "hour_utc",
        "day_index",
    ]
    rows = []
    work = df[df["frequency_mhz"].isin(FOCUS_FREQS) & df["regime"].isin(["source_like", "anti_template"])].copy()
    for keys, grp in work.groupby(["source_name", "frequency_mhz", "event_type"], sort=True):
        source, freq, event_type = keys
        src = grp[grp["regime"].eq("source_like")]
        anti = grp[grp["regime"].eq("anti_template")]
        for col in cols:
            a = pd.to_numeric(src[col], errors="coerce").dropna().to_numpy()
            b = pd.to_numeric(anti[col], errors="coerce").dropna().to_numpy()
            if a.size < 5 or b.size < 5:
                continue
            pooled = np.concatenate([a, b])
            scale = np.nanmedian(np.abs(pooled - np.nanmedian(pooled))) * 1.4826
            if not np.isfinite(scale) or scale <= 0:
                scale = np.nanstd(pooled)
            delta = float(np.nanmedian(a) - np.nanmedian(b))
            rows.append(
                {
                    "source_name": source,
                    "frequency_mhz": freq,
                    "event_type": event_type,
                    "variable": col,
                    "source_like_n": int(a.size),
                    "anti_template_n": int(b.size),
                    "source_like_median": float(np.nanmedian(a)),
                    "anti_template_median": float(np.nanmedian(b)),
                    "median_difference_source_minus_anti": delta,
                    "robust_effect_size": float(delta / scale) if np.isfinite(scale) and scale > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def categorical_differences(df: pd.DataFrame) -> pd.DataFrame:
    cats = ["month", "date", "slope_subset", "background_subset", "limb_rate_subset", "gap_subset"]
    rows = []
    work = df[df["frequency_mhz"].isin(FOCUS_FREQS) & df["regime"].isin(["source_like", "anti_template"])].copy()
    for keys, grp in work.groupby(["source_name", "frequency_mhz", "event_type"], sort=True):
        source, freq, event_type = keys
        for cat in cats:
            for label, sub in grp.groupby(cat, sort=True):
                n = len(sub)
                rows.append(
                    {
                        "source_name": source,
                        "frequency_mhz": freq,
                        "event_type": event_type,
                        "variable": cat,
                        "label": str(label),
                        "n_events": int(n),
                        "source_like_fraction": float(sub["regime"].eq("source_like").mean()),
                        "anti_template_fraction": float(sub["regime"].eq("anti_template").mean()),
                        "median_source_like_fractional_contrast": float(np.nanmedian(sub["source_like_fractional_contrast"])),
                    }
                )
    return pd.DataFrame(rows)


def plot_top_effects(effects: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    for source in ["earth", "sun"]:
        sub = effects[
            effects["source_name"].eq(source)
            & effects["frequency_mhz"].isin(LOW_FREQS)
            & effects["variable"].isin(
                [
                    "near_raw_slope_per_s",
                    "far_raw_slope_per_s",
                    "pre_far_median",
                    "pre_sigma",
                    "limb_rate_deg_s",
                    "moon_angular_radius_deg",
                    "limb_pa_sin",
                    "limb_pa_cos",
                    "hour_utc",
                    "day_index",
                ]
            )
        ].copy()
        if sub.empty:
            continue
        pivot = sub.pivot_table(index="variable", columns=["frequency_mhz", "event_type"], values="robust_effect_size", aggfunc="median")
        fig, ax = plt.subplots(figsize=(12, 5.5))
        vals = pivot.to_numpy(dtype=float)
        vmax = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1.0
        im = ax.imshow(vals, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"{c[0]:.2f}\\n{c[1][:4]}" for c in pivot.columns], rotation=0, fontsize=8)
        ax.set_title(f"{source}: source-like minus anti-template robust effect sizes")
        fig.colorbar(im, ax=ax, label="median difference / robust scale")
        fig.tight_layout()
        path = out_dir / f"{source}_lowfreq_regime_physical_effects.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_limb_pa(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    for source in ["earth", "sun"]:
        sub = df[
            df["source_name"].eq(source)
            & df["frequency_mhz"].isin(LOW_FREQS)
            & df["regime"].isin(["source_like", "anti_template"])
        ]
        if sub.empty:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
        for ax, freq in zip(axes, LOW_FREQS):
            f = sub[np.isclose(sub["frequency_mhz"], freq)]
            for regime, color in [("source_like", "#2ca02c"), ("anti_template", "#d62728")]:
                vals = f[f["regime"].eq(regime)]["limb_position_angle_deg"].dropna()
                ax.hist(vals, bins=np.linspace(0, 360, 19), histtype="step", lw=1.7, color=color, label=regime, density=True)
            ax.set_title(f"{freq:.2f} MHz")
            ax.set_xlabel("limb position angle (deg)")
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("density")
        axes[-1].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{source}: limb position angle by event regime")
        fig.tight_layout()
        path = out_dir / f"{source}_lowfreq_limb_pa_by_regime.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def _table(df: pd.DataFrame, cols: list[str], max_rows: int = 80) -> str:
    if df.empty:
        return "(empty)"
    return df[cols].head(max_rows).to_string(index=False)


def write_report(df: pd.DataFrame, effects: pd.DataFrame, cats: pd.DataFrame, paths: list[Path]) -> Path:
    top_effects = (
        effects[effects["frequency_mhz"].isin(LOW_FREQS)]
        .assign(abs_effect=lambda x: x["robust_effect_size"].abs())
        .sort_values(["source_name", "frequency_mhz", "event_type", "abs_effect"], ascending=[True, True, True, False])
        .groupby(["source_name", "frequency_mhz", "event_type"])
        .head(8)
    )
    top_cats = (
        cats[cats["frequency_mhz"].isin(LOW_FREQS) & cats["variable"].isin(["month", "slope_subset", "background_subset"])]
        .assign(majority=lambda x: np.maximum(x["source_like_fraction"], x["anti_template_fraction"]))
        .sort_values(["source_name", "frequency_mhz", "event_type", "majority"], ascending=[True, True, True, False])
        .groupby(["source_name", "frequency_mhz", "event_type"])
        .head(8)
    )
    regime_counts = (
        df[df["frequency_mhz"].isin(LOW_FREQS)]
        .groupby(["source_name", "frequency_mhz", "event_type", "regime"], as_index=False)
        .size()
        .pivot_table(index=["source_name", "frequency_mhz", "event_type"], columns="regime", values="size", fill_value=0)
        .reset_index()
    )
    lines = [
        "# Moving-Body Regime Physical Differences",
        "",
        "Question: what makes the source-like and anti-template event classes incompatible, and do they differ physically?",
        "",
        "Regimes are classified from per-event raw-fractional source-like contrast. Positive means the event follows the expected point-source occultation sign; negative means it goes the opposite way.",
        "",
        "## Regime Counts At Low Frequency",
        "",
        regime_counts.to_string(index=False),
        "",
        "## Largest Numeric Differences",
        "",
        _table(
            top_effects,
            [
                "source_name",
                "frequency_mhz",
                "event_type",
                "variable",
                "source_like_n",
                "anti_template_n",
                "source_like_median",
                "anti_template_median",
                "median_difference_source_minus_anti",
                "robust_effect_size",
            ],
            160,
        ),
        "",
        "## Strong Categorical Separations",
        "",
        _table(
            top_cats,
            [
                "source_name",
                "frequency_mhz",
                "event_type",
                "variable",
                "label",
                "n_events",
                "source_like_fraction",
                "anti_template_fraction",
                "median_source_like_fractional_contrast",
            ],
            160,
        ),
        "",
        "## Plots",
        "",
    ]
    lines.extend(f"- `{p}`" for p in paths)
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The event classes are incompatible because they have opposite local pre/post behavior. For disappearance, source-like events have falling local power through the event while anti-template events have rising local power. For reappearance the relation reverses. A single stack averages these opposite slopes and can hide the fact that ordinary occultation-like events exist.",
            "",
            "The dominant separator is local power trend. Broad-window slope, background level/noise, month, and limb position angle provide secondary structure, but they are weaker than the direct local trend. This points to the Sun/Earth signal being measured on top of a large moving-body/background/antenna response that changes with orbital geometry.",
        ]
    )
    path = OUT / "moving_body_regime_physical_differences_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    ensure_dir(OUT)
    table = build_regime_geometry_table()
    effects = numeric_differences(table)
    cats = categorical_differences(table)
    table.to_csv(OUT / "moving_body_event_regime_geometry_table.csv", index=False)
    effects.to_csv(OUT / "numeric_regime_differences.csv", index=False)
    cats.to_csv(OUT / "categorical_regime_differences.csv", index=False)
    paths = []
    paths.extend(plot_top_effects(effects, OUT))
    paths.extend(plot_limb_pa(table, OUT))
    report = write_report(table, effects, cats, paths)
    print(report)


if __name__ == "__main__":
    main()
