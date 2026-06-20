#!/usr/bin/env python
"""Lower-V-only stack-first occultation detection attempt for Sun and Fornax-A.

This run intentionally avoids raw-shape event selection.  It stacks all
independently usable predicted lower-V event windows, then fits the same
finite-duration occultation template to real and control stacks.
"""

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
from rylevonberg.detection import baseline_matrix  # noqa: E402
from rylevonberg.stackfit import StackedStepFitConfig, fit_stacked_step, stacked_event_template  # noqa: E402
from rylevonberg.util import datetime_ns, ensure_dir, robust_sigma, software_versions, write_json  # noqa: E402


CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
SUN_EVENTS = ROOT / "outputs/pipeline_confidence_audit_v2/sun_audit_input_events.csv"
FORNAX_EVENTS = ROOT / "outputs/control_survey_bright_sources_postnov1974_v1/02_events/predicted_events.csv"
SUN_OFFSOURCE_EVENTS = ROOT / "outputs/sun_whole_dataset_validation_allbands_mincontrols/02_events/sun_offephemeris_predicted_events.csv"
FORNAX_OFFSOURCE_EVENTS = ROOT / "outputs/post_selection_bias_audit_sun_fornax_dt0_v1/controls/fornax_a_offsource_predicted_events.csv"
DEFAULT_OUT = ROOT / "outputs/lower_v_stackfirst_detection_sun_fornax_v1"
ANTENNA = "rv2_coarse"
EVENT_SIGN = {"disappearance": -1.0, "reappearance": 1.0}
CONTROL_STYLE = {
    "time_shift": ("time-shift controls", "#d95f02"),
    "randomized_time": ("randomized-time controls", "#7570b3"),
    "offsource": ("off-source controls", "#1b9e77"),
}
SOURCE_LABEL = {"sun": "Sun", "fornax_a": "Fornax-A"}


def _read(path: Path, **kwargs) -> pd.DataFrame:
    return read_table(path, low_memory=False, **kwargs)


def _bool_values(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    return series.astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy(dtype=bool)


def _robust_sem(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size <= 1:
        return np.nan
    scale = robust_sigma(vals - np.nanmedian(vals))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(vals, ddof=1))
    return float(scale / np.sqrt(vals.size)) if np.isfinite(scale) and scale > 0 else np.nan


def _load_clean_groups(bands: list[int]) -> dict[tuple[int, str], dict[str, np.ndarray]]:
    cols = ["time", "frequency_band", "frequency_mhz", "antenna", "power", "is_valid"]
    clean = _read(CLEAN, usecols=cols, parse_dates=["time"])
    clean = clean[clean["antenna"].astype(str).eq(ANTENNA) & clean["frequency_band"].astype(int).isin(set(bands))].copy()
    groups: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for (band, antenna), grp in clean.groupby(["frequency_band", "antenna"], sort=True):
        g = grp.sort_values("time").reset_index(drop=True)
        power = pd.to_numeric(g["power"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(power) & (power > 0.0)
        if "is_valid" in g.columns:
            valid &= _bool_values(g["is_valid"])
        groups[(int(band), str(antenna))] = {
            "time_ns": datetime_ns(g["time"]),
            "power": power,
            "valid": valid,
        }
    return groups


def _lower_v_after_date(events: pd.DataFrame, source: str, start_date: str) -> pd.DataFrame:
    work = events[
        events["source_name"].astype(str).str.lower().eq(source)
        & events["antenna"].astype(str).eq(ANTENNA)
    ].copy()
    work["predicted_event_time"] = pd.to_datetime(work["predicted_event_time"])
    work = work[work["predicted_event_time"] >= pd.Timestamp(start_date)].copy()
    return work.reset_index(drop=True)


def _load_real_events(start_date: str) -> dict[str, pd.DataFrame]:
    sun = _lower_v_after_date(_read(SUN_EVENTS, parse_dates=["predicted_event_time"]), "sun", start_date)
    fornax = _lower_v_after_date(_read(FORNAX_EVENTS, parse_dates=["predicted_event_time"]), "fornax_a", start_date)
    for source, frame in [("sun", sun), ("fornax_a", fornax)]:
        frame["analysis_source"] = source
        frame["source_name"] = source
        frame["control_family"] = "real"
        frame["control_id"] = "real"
        frame["control_type"] = "true_prediction"
    return {"sun": sun, "fornax_a": fornax}


def _make_time_shift_controls(real_events: pd.DataFrame, shifts_s: list[float]) -> pd.DataFrame:
    tables = []
    source = str(real_events["analysis_source"].iloc[0])
    for shift in shifts_s:
        shifted = real_events.copy()
        shifted["predicted_event_time"] = shifted["predicted_event_time"] + pd.to_timedelta(float(shift), unit="s")
        shifted["source_name"] = source
        shifted["analysis_source"] = source
        shifted["control_family"] = "time_shift"
        shifted["control_id"] = f"shift_{int(shift):+d}s"
        shifted["control_type"] = "temporal_offset"
        shifted["time_shift_s"] = float(shift)
        tables.append(shifted)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def _make_randomized_controls(real_events: pd.DataFrame, clean_times: pd.Series, n_random: int, seed: int, window_s: float) -> pd.DataFrame:
    source = str(real_events["analysis_source"].iloc[0])
    rng = np.random.default_rng(seed)
    unique_times = pd.Series(pd.to_datetime(clean_times).drop_duplicates().sort_values().to_numpy())
    choices = unique_times[
        (unique_times >= unique_times.min() + pd.to_timedelta(float(window_s), unit="s"))
        & (unique_times <= unique_times.max() - pd.to_timedelta(float(window_s), unit="s"))
    ]
    keys = real_events[["event_id", "event_type", "predicted_event_time"]].drop_duplicates().reset_index(drop=True)
    tables = []
    for idx in range(int(n_random)):
        mapping = keys.copy()
        mapping["_event_key"] = np.arange(len(mapping))
        mapping["randomized_event_time"] = rng.choice(choices.to_numpy(), size=len(mapping), replace=True)
        randomized = real_events.merge(mapping[["event_id", "event_type", "predicted_event_time", "_event_key"]], on=["event_id", "event_type", "predicted_event_time"], how="left")
        randomized = randomized.merge(mapping[["_event_key", "randomized_event_time"]], on="_event_key", how="left")
        randomized["predicted_event_time"] = pd.to_datetime(randomized["randomized_event_time"])
        randomized = randomized.drop(columns=["_event_key", "randomized_event_time"])
        randomized["source_name"] = source
        randomized["analysis_source"] = source
        randomized["control_family"] = "randomized_time"
        randomized["control_id"] = f"random_{idx:03d}"
        randomized["control_type"] = "randomized_event_time"
        randomized["random_realization"] = idx
        tables.append(randomized)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def _load_offsource_events(source: str, start_date: str, max_offsource_controls: int) -> pd.DataFrame:
    if source == "sun":
        events = _read(SUN_OFFSOURCE_EVENTS, parse_dates=["predicted_event_time"])
        events = events[events["antenna"].astype(str).eq(ANTENNA)].copy()
        events["control_id"] = events["control_name"].astype(str)
    elif source == "fornax_a":
        events = _read(FORNAX_OFFSOURCE_EVENTS, parse_dates=["predicted_event_time"])
        events = events[events["antenna"].astype(str).eq(ANTENNA)].copy()
        events["control_id"] = events["control_name"].astype(str)
        controls = sorted(events["control_id"].dropna().astype(str).unique())
        if max_offsource_controls > 0:
            controls = controls[: int(max_offsource_controls)]
            events = events[events["control_id"].isin(controls)].copy()
    else:
        raise ValueError(source)
    events["predicted_event_time"] = pd.to_datetime(events["predicted_event_time"])
    events = events[events["predicted_event_time"] >= pd.Timestamp(start_date)].copy()
    events["analysis_source"] = source
    events["source_name"] = source
    events["control_family"] = "offsource"
    events["control_type"] = events.get("control_type", "offsource")
    return events.reset_index(drop=True)


def _load_event_table(
    start_date: str,
    shifts_s: list[float],
    n_random: int,
    random_seed: int,
    window_s: float,
    max_offsource_controls: int,
) -> pd.DataFrame:
    real = _load_real_events(start_date)
    clean_times = _read(CLEAN, usecols=["time"], parse_dates=["time"])["time"]
    tables = []
    for source, real_events in real.items():
        tables.append(real_events)
        tables.append(_make_time_shift_controls(real_events, shifts_s))
        tables.append(
            _make_randomized_controls(
                real_events,
                clean_times,
                n_random=n_random,
                seed=int(random_seed) + sum((i + 1) * ord(ch) for i, ch in enumerate(source)),
                window_s=window_s,
            )
        )
        tables.append(_load_offsource_events(source, start_date, max_offsource_controls))
    events = pd.concat([t for t in tables if not t.empty], ignore_index=True)
    events["event_uid"] = np.arange(len(events), dtype=int)
    return events


def _window(group: dict[str, np.ndarray], event_time: pd.Timestamp, window_s: float) -> tuple[np.ndarray, np.ndarray] | None:
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
    y = group["power"][lo:hi][keep]
    t = t[keep]
    order = np.argsort(t)
    return t[order], y[order]


def _normalize(t: np.ndarray, y: np.ndarray, sideband_s: float) -> tuple[np.ndarray, np.ndarray] | None:
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


def collect_profiles(events: pd.DataFrame, clean_groups: dict[tuple[int, str], dict[str, np.ndarray]], window_s: float, bin_s: float, sideband_s: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    bins = np.arange(-float(window_s), float(window_s) + float(bin_s), float(bin_s))
    point_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    for ev in events.sort_values(["analysis_source", "control_family", "control_id", "predicted_event_time", "frequency_mhz"]).itertuples(index=False):
        band = int(ev.frequency_band)
        payload = clean_groups.get((band, ANTENNA))
        base = {
            "analysis_source": str(ev.analysis_source),
            "control_family": str(ev.control_family),
            "control_id": str(ev.control_id),
            "event_uid": int(ev.event_uid),
            "event_id": getattr(ev, "event_id", np.nan),
            "event_type": str(ev.event_type),
            "predicted_event_time": ev.predicted_event_time,
            "frequency_band": band,
            "frequency_mhz": float(ev.frequency_mhz),
            "antenna": ANTENNA,
        }
        if payload is None:
            status_rows.append({**base, "used_in_stack": False, "failure": "missing_clean_group", "n_valid_samples": 0})
            continue
        local = _window(payload, pd.Timestamp(ev.predicted_event_time), window_s)
        if local is None:
            status_rows.append({**base, "used_in_stack": False, "failure": "no_or_few_samples", "n_valid_samples": 0})
            continue
        norm = _normalize(local[0], local[1], sideband_s)
        if norm is None:
            status_rows.append({**base, "used_in_stack": False, "failure": "normalization_failed", "n_valid_samples": int(len(local[0]))})
            continue
        t, z = norm
        status_rows.append({**base, "used_in_stack": True, "failure": "", "n_valid_samples": int(len(t))})
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


def stack_by_control(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    by = ["analysis_source", "control_family", "control_id", "frequency_band", "frequency_mhz", "event_type", "t_bin_sec"]
    for keys, grp in points.groupby(by, sort=True, dropna=False):
        vals = pd.to_numeric(grp["z_power"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        rows.append(
            {
                **dict(zip(by, keys)),
                "median_z_power": float(np.nanmedian(vals)),
                "mean_z_power": float(np.nanmean(vals)),
                "robust_sem_z_power": _robust_sem(vals),
                "n_windows": int(grp["event_uid"].nunique()),
                "n_points": int(vals.size),
            }
        )
    return pd.DataFrame(rows)


def summarize_control_curves(stacks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    by = ["analysis_source", "control_family", "frequency_band", "frequency_mhz", "event_type", "t_bin_sec"]
    for keys, grp in stacks.groupby(by, sort=True, dropna=False):
        vals = pd.to_numeric(grp["median_z_power"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        rows.append(
            {
                **dict(zip(by, keys)),
                "median_curve": float(np.nanmedian(vals)),
                "q25_curve": float(np.nanquantile(vals, 0.25)),
                "q75_curve": float(np.nanquantile(vals, 0.75)),
                "n_control_groups": int(grp["control_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def fit_stacks(stacks: pd.DataFrame, timing_offsets: list[float], transition_durations: list[float]) -> pd.DataFrame:
    rows = []
    cfg = StackedStepFitConfig(
        baseline_order=0,
        timing_offsets_seconds=tuple(float(x) for x in timing_offsets),
        transition_durations_seconds=tuple(float(x) for x in transition_durations),
        min_bins=8,
    )
    group_cols = ["analysis_source", "control_family", "control_id", "frequency_band", "frequency_mhz"]
    for keys, grp in stacks.groupby(group_cols, sort=True, dropna=False):
        meta = dict(zip(group_cols, keys))
        for stack_group, sub in [("combined", grp), *[(et, g) for et, g in grp.groupby("event_type", sort=True)]]:
            sub = sub.sort_values(["event_type", "t_bin_sec"])
            fit = fit_stacked_step(
                sub["t_bin_sec"].to_numpy(dtype=float),
                sub["median_z_power"].to_numpy(dtype=float),
                sub["event_type"].astype(str).to_numpy(),
                uncertainty=sub["robust_sem_z_power"].to_numpy(dtype=float),
                config=cfg,
            )
            rows.append(
                {
                    **meta,
                    "stack_group": str(stack_group),
                    "n_event_type_groups": int(sub["event_type"].nunique()),
                    "median_windows_per_bin": float(np.nanmedian(sub["n_windows"])) if not sub.empty else np.nan,
                    **fit,
                }
            )
    return pd.DataFrame(rows)


def empirical_fit_summary(fits: pd.DataFrame) -> pd.DataFrame:
    real = fits[(fits["control_family"].eq("real")) & fits["stack_group"].eq("combined")].copy()
    controls = fits[(~fits["control_family"].eq("real")) & fits["stack_group"].eq("combined")].copy()
    rows = []
    for _, row in real.iterrows():
        same = controls[
            controls["analysis_source"].eq(row["analysis_source"])
            & controls["frequency_band"].eq(row["frequency_band"])
        ].copy()
        out = {
            "analysis_source": row["analysis_source"],
            "frequency_band": int(row["frequency_band"]),
            "frequency_mhz": float(row["frequency_mhz"]),
            "real_amplitude": float(row["amplitude"]),
            "real_uncertainty": float(row["uncertainty"]),
            "real_fit_snr": float(row["stack_fit_snr"]),
            "real_delta_bic": float(row["delta_bic"]),
            "real_best_transition_duration_s": float(row.get("best_transition_duration_s", np.nan)),
        }
        for family in ["time_shift", "randomized_time", "offsource"]:
            fam = same[same["control_family"].eq(family)]
            vals = pd.to_numeric(fam["amplitude"], errors="coerce").dropna().to_numpy(dtype=float)
            if len(vals):
                out[f"{family}_n"] = int(len(vals))
                out[f"{family}_median_amplitude"] = float(np.nanmedian(vals))
                out[f"{family}_q25_amplitude"] = float(np.nanquantile(vals, 0.25))
                out[f"{family}_q75_amplitude"] = float(np.nanquantile(vals, 0.75))
                out[f"{family}_p_amp_ge_real"] = float((1 + np.count_nonzero(vals >= float(row["amplitude"]))) / (1 + len(vals)))
            else:
                out[f"{family}_n"] = 0
                out[f"{family}_median_amplitude"] = np.nan
                out[f"{family}_q25_amplitude"] = np.nan
                out[f"{family}_q75_amplitude"] = np.nan
                out[f"{family}_p_amp_ge_real"] = np.nan
        rows.append(out)
    return pd.DataFrame(rows)


def _model_curve(sub: pd.DataFrame, fit: pd.Series) -> np.ndarray:
    t = sub["t_bin_sec"].to_numpy(dtype=float)
    event_types = sub["event_type"].astype(str).to_numpy()
    tmpl = stacked_event_template(
        t,
        event_types,
        timing_offset_sec=float(fit["best_timing_offset_s"]),
        transition_duration_sec=float(fit["best_transition_duration_s"]),
    )
    B = baseline_matrix(t, 0)
    X = np.column_stack([B, tmpl])
    y = sub["median_z_power"].to_numpy(dtype=float)
    sigma = sub["robust_sem_z_power"].to_numpy(dtype=float)
    fallback = np.nanmedian(sigma[np.isfinite(sigma) & (sigma > 0)])
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, fallback)
    weights = 1.0 / sigma**2
    beta, *_ = np.linalg.lstsq(X * np.sqrt(weights)[:, None], y * np.sqrt(weights), rcond=None)
    return X @ beta


def plot_real_vs_controls_grid(stacks: pd.DataFrame, control_curves: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    for source in sorted(stacks["analysis_source"].unique()):
        freqs = sorted(stacks.loc[stacks["analysis_source"].eq(source), "frequency_mhz"].dropna().unique())
        fig, axes = plt.subplots(len(freqs), 2, figsize=(13.5, max(10, 1.45 * len(freqs))), sharex=True, sharey=False)
        if len(freqs) == 1:
            axes = np.asarray([axes])
        for i, freq in enumerate(freqs):
            for j, event_type in enumerate(["disappearance", "reappearance"]):
                ax = axes[i, j]
                real = stacks[
                    stacks["analysis_source"].eq(source)
                    & stacks["control_family"].eq("real")
                    & np.isclose(stacks["frequency_mhz"].astype(float), float(freq))
                    & stacks["event_type"].eq(event_type)
                ].sort_values("t_bin_sec")
                if not real.empty:
                    ax.errorbar(
                        real["t_bin_sec"] / 60.0,
                        real["median_z_power"],
                        yerr=real["robust_sem_z_power"],
                        color="black",
                        ecolor="black",
                        marker="o",
                        markersize=2.4,
                        linewidth=1.5,
                        elinewidth=0.55,
                        capsize=1.1,
                        label="real stack",
                    )
                for family, (label, color) in CONTROL_STYLE.items():
                    ctrl = control_curves[
                        control_curves["analysis_source"].eq(source)
                        & control_curves["control_family"].eq(family)
                        & np.isclose(control_curves["frequency_mhz"].astype(float), float(freq))
                        & control_curves["event_type"].eq(event_type)
                    ].sort_values("t_bin_sec")
                    if ctrl.empty:
                        continue
                    x = ctrl["t_bin_sec"].to_numpy(dtype=float) / 60.0
                    ax.plot(x, ctrl["median_curve"], color=color, lw=1.0, alpha=0.9, label=label)
                    ax.fill_between(x, ctrl["q25_curve"], ctrl["q75_curve"], color=color, alpha=0.13, linewidth=0)
                ax.axvline(0.0, color="black", linestyle=":", linewidth=0.8)
                ax.axhline(0.0, color="0.7", linewidth=0.7)
                n_real = int(real["n_windows"].median()) if not real.empty else 0
                ax.text(0.01, 0.96, f"real n~{n_real}", transform=ax.transAxes, ha="left", va="top", fontsize=6.8, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.0})
                ax.set_title(f"{float(freq):.2f} MHz {event_type}", fontsize=8.5)
                if j == 0:
                    ax.set_ylabel("normalized lower-V power")
                if i == len(freqs) - 1:
                    ax.set_xlabel("minutes from predicted event")
                ax.grid(True, color="0.92", linewidth=0.5)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=min(4, len(by_label)), frameon=False)
        fig.suptitle(f"{SOURCE_LABEL.get(source, source)} lower-V stack-first profiles: real data versus controls", y=0.996)
        fig.tight_layout(rect=[0, 0, 1, 0.965])
        path = out_dir / f"{source}_lower_v_stackfirst_real_vs_controls_grid.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_amplitude_spectrum(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    for source, src in summary.groupby("analysis_source", sort=True):
        fig, ax = plt.subplots(figsize=(9.5, 4.6))
        x = src["frequency_mhz"].to_numpy(dtype=float)
        ax.axhline(0.0, color="0.6", linewidth=0.8)
        for family, (_, color) in CONTROL_STYLE.items():
            q25 = src[f"{family}_q25_amplitude"].to_numpy(dtype=float)
            q75 = src[f"{family}_q75_amplitude"].to_numpy(dtype=float)
            med = src[f"{family}_median_amplitude"].to_numpy(dtype=float)
            ax.fill_between(x, q25, q75, color=color, alpha=0.13)
            ax.plot(x, med, color=color, lw=1.0, marker=".", label=f"{family} median/IQR")
        ax.errorbar(
            x,
            src["real_amplitude"],
            yerr=src["real_uncertainty"],
            color="black",
            marker="o",
            lw=1.4,
            capsize=2,
            label="real stack fit",
        )
        ax.set_xscale("log")
        ax.set_xticks(list(FREQUENCY_MAP_MHZ.values()))
        ax.set_xticklabels([f"{v:g}" for v in FREQUENCY_MAP_MHZ.values()], rotation=45, ha="right")
        ax.set_xlabel("frequency (MHz)")
        ax.set_ylabel("combined-stack positive-source template amplitude")
        ax.set_title(f"{SOURCE_LABEL.get(source, source)} lower-V stack-first fitted amplitude spectrum")
        ax.grid(True, color="0.9", linewidth=0.5)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        path = out_dir / f"{source}_lower_v_stackfirst_amplitude_spectrum.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_top_fit_profiles(stacks: pd.DataFrame, fits: pd.DataFrame, control_curves: pd.DataFrame, summary: pd.DataFrame, out_dir: Path, n_channels: int = 4) -> list[Path]:
    paths = []
    for source, src in summary.groupby("analysis_source", sort=True):
        ranked = src.copy()
        ranked["abs_amplitude"] = pd.to_numeric(ranked["real_amplitude"], errors="coerce").abs()
        ranked = ranked.sort_values(["abs_amplitude", "real_delta_bic"], ascending=[False, False]).head(int(n_channels))
        if ranked.empty:
            continue
        fig, axes = plt.subplots(len(ranked), 2, figsize=(13.5, max(3.0, 2.7 * len(ranked))), squeeze=False, sharex=True)
        for i, (_, row) in enumerate(ranked.iterrows()):
            freq = float(row["frequency_mhz"])
            real_fit = fits[
                fits["analysis_source"].eq(source)
                & fits["control_family"].eq("real")
                & np.isclose(fits["frequency_mhz"].astype(float), freq)
                & fits["stack_group"].eq("combined")
            ].iloc[0]
            channel = stacks[
                stacks["analysis_source"].eq(source)
                & stacks["control_family"].eq("real")
                & np.isclose(stacks["frequency_mhz"].astype(float), freq)
            ].copy()
            channel = channel.sort_values(["event_type", "t_bin_sec"])
            channel["model"] = _model_curve(channel, real_fit)
            for j, event_type in enumerate(["disappearance", "reappearance"]):
                ax = axes[i, j]
                sub = channel[channel["event_type"].eq(event_type)].sort_values("t_bin_sec")
                if not sub.empty:
                    x = sub["t_bin_sec"].to_numpy(dtype=float) / 60.0
                    y = sub["median_z_power"].to_numpy(dtype=float)
                    err = sub["robust_sem_z_power"].to_numpy(dtype=float)
                    ax.errorbar(x, y, yerr=err, color="black", ecolor="black", marker="o", markersize=2.5, lw=1.1, elinewidth=0.55, capsize=1.1, label="real stack")
                    ax.plot(x, sub["model"], color="#e41a1c", lw=1.8, label="fit to real stack")
                for family, (label, color) in CONTROL_STYLE.items():
                    ctrl = control_curves[
                        control_curves["analysis_source"].eq(source)
                        & control_curves["control_family"].eq(family)
                        & np.isclose(control_curves["frequency_mhz"].astype(float), freq)
                        & control_curves["event_type"].eq(event_type)
                    ].sort_values("t_bin_sec")
                    if ctrl.empty:
                        continue
                    x_ctrl = ctrl["t_bin_sec"].to_numpy(dtype=float) / 60.0
                    ax.plot(x_ctrl, ctrl["median_curve"], color=color, lw=0.9, alpha=0.8, label=label)
                    ax.fill_between(x_ctrl, ctrl["q25_curve"], ctrl["q75_curve"], color=color, alpha=0.10, linewidth=0)
                ax.axvline(0.0, color="black", ls=":", lw=0.8)
                ax.axhline(0.0, color="0.7", lw=0.7)
                ax.grid(True, color="0.92", linewidth=0.5)
                ax.set_title(event_type, fontsize=9)
                if j == 0:
                    ax.set_ylabel(f"{freq:.2f} MHz\nnormalized power")
                if i == len(ranked) - 1:
                    ax.set_xlabel("minutes from predicted event")
                if i == 0 and j == 1:
                    ax.legend(frameon=False, fontsize=7)
            axes[i, 1].text(
                1.02,
                0.5,
                f"A={float(row['real_amplitude']):.3f}\n"
                f"unc={float(row['real_uncertainty']):.3f}\n"
                f"fitSNR={float(row['real_fit_snr']):.2f}\n"
                f"DeltaBIC={float(row['real_delta_bic']):.1f}\n"
                f"tau={float(row['real_best_transition_duration_s']):.0f}s\n"
                f"p_shift={float(row['time_shift_p_amp_ge_real']):.2f}\n"
                f"p_off={float(row['offsource_p_amp_ge_real']):.2f}",
                transform=axes[i, 1].transAxes,
                ha="left",
                va="center",
                fontsize=8,
            )
        fig.suptitle(f"{SOURCE_LABEL.get(source, source)} lower-V stack-first largest absolute-amplitude channels", y=0.995)
        fig.tight_layout(rect=[0, 0, 0.91, 0.96])
        path = out_dir / f"{source}_lower_v_stackfirst_top_fit_profiles.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def write_report(out_dir: Path, summary: pd.DataFrame, status: pd.DataFrame, paths: list[Path]) -> None:
    use_counts = (
        status.groupby(["analysis_source", "control_family"])["used_in_stack"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "n_used_windows", "count": "n_candidate_windows"})
    )
    cols = [
        "analysis_source",
        "frequency_mhz",
        "real_amplitude",
        "real_uncertainty",
        "real_fit_snr",
        "real_delta_bic",
        "real_best_transition_duration_s",
        "time_shift_p_amp_ge_real",
        "randomized_time_p_amp_ge_real",
        "offsource_p_amp_ge_real",
    ]
    lines = [
        "# Lower-V Stack-First Detection Attempt: Sun and Fornax-A",
        "",
        "This run ignores upper V entirely and does not select events by raw occultation shape.",
        "",
        "Each predicted lower-V event window is normalized by its outer sidebands, stacked by frequency and event type, and fit after stacking with:",
        "",
        "    y_stack(t) = c + A h(t; tau)",
        "",
        "`A > 0` means disappearance decreases and reappearance increases under the positive-source convention. Controls are extracted from the same lower-V data using wrong times or wrong sky positions.",
        "",
        "## Usable Window Counts",
        "",
        use_counts.to_string(index=False),
        "",
        "## Combined-Stack Fit Summary",
        "",
        summary[cols].sort_values(["analysis_source", "frequency_mhz"]).to_string(index=False),
        "",
        "## Plots",
        "",
    ]
    lines.extend(f"- `{path.name}`" for path in paths)
    lines += [
        "",
        "## How To Read The Plots",
        "",
        "- The all-frequency grids show actual normalized lower-V stacked power for real events in black.",
        "- Colored curves and bands are control-stack medians and interquartile ranges.",
        "- A convincing detection should have real disappearance/reappearance morphology that is not reproduced by wrong-time or wrong-sky controls.",
        "- The top-fit panels overlay the fitted finite-duration template on the real stack, but the controls remain visible on the same axes.",
    ]
    (out_dir / "lower_v_stackfirst_detection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--start-date", default="1974-11-01")
    parser.add_argument("--window-s", type=float, default=900.0)
    parser.add_argument("--bin-s", type=float, default=60.0)
    parser.add_argument("--sideband-s", type=float, default=600.0)
    parser.add_argument("--time-shifts-s", default="-1200,-600,-300,300,600,1200")
    parser.add_argument("--n-random", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=20260604)
    parser.add_argument("--max-offsource-controls", type=int, default=16)
    parser.add_argument("--timing-offsets-s", default="0")
    parser.add_argument("--transition-durations-s", default="0,120,300,600,900")
    args = parser.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    shifts = [float(x.strip()) for x in str(args.time_shifts_s).split(",") if x.strip()]
    timing_offsets = [float(x.strip()) for x in str(args.timing_offsets_s).split(",") if x.strip()]
    transition_durations = [float(x.strip()) for x in str(args.transition_durations_s).split(",") if x.strip()]
    config = {
        "clean": str(CLEAN),
        "antenna": ANTENNA,
        "start_date": str(args.start_date),
        "window_s": float(args.window_s),
        "bin_s": float(args.bin_s),
        "sideband_s": float(args.sideband_s),
        "time_shifts_s": shifts,
        "n_random": int(args.n_random),
        "random_seed": int(args.random_seed),
        "max_offsource_controls": int(args.max_offsource_controls),
        "timing_offsets_s": timing_offsets,
        "transition_durations_s": transition_durations,
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)

    print("Loading event/control tables...", flush=True)
    events = _load_event_table(
        str(args.start_date),
        shifts_s=shifts,
        n_random=int(args.n_random),
        random_seed=int(args.random_seed),
        window_s=float(args.window_s),
        max_offsource_controls=int(args.max_offsource_controls),
    )
    events.to_csv(out_dir / "lower_v_stackfirst_event_rows.csv", index=False)
    bands = sorted(events["frequency_band"].dropna().astype(int).unique())
    print(f"Loading lower-V clean groups for bands {bands}...", flush=True)
    clean_groups = _load_clean_groups(bands)
    print(f"Collecting all usable lower-V profiles from {len(events)} event/control rows...", flush=True)
    points, status = collect_profiles(events, clean_groups, float(args.window_s), float(args.bin_s), float(args.sideband_s))
    points.to_csv(out_dir / "lower_v_stackfirst_profile_points.csv", index=False)
    status.to_csv(out_dir / "lower_v_stackfirst_profile_status.csv", index=False)
    print("Stacking by real/control group...", flush=True)
    stacks = stack_by_control(points)
    stacks.to_csv(out_dir / "lower_v_stackfirst_control_stacks.csv", index=False)
    control_curves = summarize_control_curves(stacks[stacks["control_family"].ne("real")].copy())
    control_curves.to_csv(out_dir / "lower_v_stackfirst_control_curve_summary.csv", index=False)
    print("Fitting finite-duration templates to stacks...", flush=True)
    fits = fit_stacks(stacks, timing_offsets, transition_durations)
    fits.to_csv(out_dir / "lower_v_stackfirst_fit_summary_by_group.csv", index=False)
    summary = empirical_fit_summary(fits)
    summary.to_csv(out_dir / "lower_v_stackfirst_empirical_fit_summary.csv", index=False)
    print("Writing plots...", flush=True)
    paths: list[Path] = []
    paths.extend(plot_real_vs_controls_grid(stacks, control_curves, out_dir))
    paths.extend(plot_amplitude_spectrum(summary, out_dir))
    paths.extend(plot_top_fit_profiles(stacks, fits, control_curves, summary, out_dir, n_channels=4))
    write_report(out_dir, summary, status, paths)
    print(out_dir / "lower_v_stackfirst_detection_report.md")


if __name__ == "__main__":
    main()
